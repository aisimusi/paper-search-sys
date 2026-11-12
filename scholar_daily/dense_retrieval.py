"""Per-paper persistent FAISS indexes over validated retrieval fragments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

from ._compat import FileLock
from .embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from .paper_context import digest
from .pipeline import atomic_text


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _atomic_bytes(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".index-", delete=False) as target:
            temporary = Path(target.name)
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DenseRetriever:
    def __init__(self, config=None, *, embedder=None, cache_dir=None):
        # Optional dependencies never enter the BM25 import path.
        import faiss
        import numpy as np
        self.faiss, self.np = faiss, np
        self.config = config if isinstance(config, EmbeddingConfig) else EmbeddingConfig(**(config or {}))
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder(self.config)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.embedding_signature = self.config.signature()
        self.embedding_signature["implementation"] = type(self.embedder).__module__ + "." + type(self.embedder).__qualname__
        self.last_stats = {}

    def _vectors(self, values, count, *, normalize=True):
        np = self.np
        if np.iscomplexobj(values):
            raise ValueError("Embedding 必须为实数向量")
        with np.errstate(over="ignore", invalid="ignore"):
            vectors = np.array(values, dtype="float32", order="C", copy=True)
        if vectors.shape != (count, self.config.dimension):
            raise ValueError("Embedding 向量数量或维度错误：" + str(vectors.shape))
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding 包含非有限向量")
        norms = np.linalg.norm(vectors.astype("float64"), axis=1)
        if (norms <= 0).any():
            raise ValueError("Embedding 包含零向量")
        if normalize:
            vectors = np.ascontiguousarray(vectors / norms[:, None], dtype="float32")
        elif not np.allclose(norms, 1, atol=1e-5):
            raise ValueError("缓存向量未归一化")
        return vectors

    def _identity(self, chunks):
        first = chunks[0]
        return {
            "schema_version": 1, "index_type": "IndexFlatIP", "metric": "cosine",
            "faiss_version": self.faiss.__version__, "embedding": self.embedding_signature,
            "paper_id": first["paper_id"], "version": first["version"],
            "source_fingerprint": first["source_fingerprint"],
            "offset_unit": first["offset_unit"],
            # Effective chunk boundaries also capture changes to chunking settings.
            "fragments": [{**{k: row[k] for k in (
                "fragment_id", "source_id", "source_kind", "source_sha256", "start", "end")},
                "text_sha256": digest(row["text"])} for row in chunks],
        }

    def _load(self, folder, identity):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("向量缓存元数据已失效")
        binary = (folder / "index.faiss").read_bytes()
        if hashlib.sha256(binary).hexdigest() != manifest["index_sha256"]:
            raise ValueError("FAISS 缓存指纹不匹配")
        index = self.faiss.deserialize_index(self.np.frombuffer(binary, dtype="uint8").copy())
        if (not isinstance(index, self.faiss.IndexFlatIP) or index.d != self.config.dimension
                or index.ntotal != len(identity["fragments"])):
            raise ValueError("FAISS 缓存类型、维度或片段数量不一致")
        self._vectors(index.reconstruct_n(0, index.ntotal), index.ntotal, normalize=False)
        return index

    def _index(self, chunks):
        started = perf_counter()
        identity = self._identity(chunks)
        key = digest(_json(identity))
        root = (self.cache_dir / digest(chunks[0]["paper_id"]) if self.cache_dir is not None
                else Path(chunks[0]["index_path"]).parent / "dense")
        folder = root / key
        folder.mkdir(parents=True, exist_ok=True)
        info = {"paper_id": chunks[0]["paper_id"], "cache_path": str(folder),
                "fragments": len(chunks), "status": "hit"}
        # Readers and writers share a per-key lock; publish the manifest last.
        with FileLock(folder / ".lock"):
            try:
                index = self._load(folder, identity)
            except (FileNotFoundError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                info.update(status="rebuilt", reason=str(exc))
                vectors = self._vectors(self.embedder.encode([c["text"] for c in chunks], query=False), len(chunks))
                index = self.faiss.IndexFlatIP(self.config.dimension)
                index.add(vectors)
                binary = self.faiss.serialize_index(index).tobytes()
                _atomic_bytes(folder / "index.faiss", binary)
                atomic_text(folder / "manifest.json", _json({
                    "identity": identity, "index_sha256": hashlib.sha256(binary).hexdigest(),
                }))
        info["seconds"] = perf_counter() - started
        self.last_stats["papers"].append(info)
        return index

    def retrieve(self, query, chunks, top_k, issues):
        self.last_stats = {"papers": [], "query_seconds": 0.0, "search_seconds": 0.0}
        if not chunks:
            return []
        try:
            started = perf_counter()
            vector = self._vectors(self.embedder.encode([query], query=True), 1)
            self.last_stats["query_seconds"] = perf_counter() - started
        except Exception as exc:
            issues.append({"paper_id": "", "code": "dense_query_failed", "message": str(exc)})
            return []
        papers = {}
        for row in chunks:
            papers.setdefault(row["paper_id"], []).append(row)
        hits = []
        for pid, rows in sorted(papers.items()):
            try:
                index = self._index(rows)
                started = perf_counter()
                # Exact task-sized search; include ties at the top-k boundary.
                scores, positions = index.search(vector, index.ntotal)
                self.last_stats["search_seconds"] += perf_counter() - started
                if not self.np.isfinite(scores).all():
                    raise ValueError("FAISS 返回非有限相似度")
                hits.extend(dict(rows[int(pos)], score=float(score), method="dense")
                            for score, pos in zip(scores[0], positions[0]) if pos >= 0)
            except Exception as exc:
                issues.append({"paper_id": pid, "code": "dense_index_failed", "message": str(exc)})
        return sorted(hits, key=lambda h: (-h["score"], h["paper_id"], h["source_kind"], h["start"], h["fragment_id"]))[:top_k]
