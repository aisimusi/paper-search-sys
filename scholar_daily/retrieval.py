"""Retrieval candidates over existing paper snapshots, not verified evidence."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
from typing import Callable, Literal, Mapping, Sequence, TypedDict

from .paper_context import load_index

DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60


class RetrievalHit(TypedDict):
    paper_id: str
    version: str
    fragment_id: str
    source_fingerprint: str
    source_id: str
    source_kind: str
    source_sha256: str
    index_path: str
    section_id: str
    section: str
    page_start: int | None
    page_end: int | None
    page_basis: str
    start: int
    end: int
    offset_unit: str
    text: str
    score: float
    method: str
    verification_status: Literal["unverified"]


class RetrievalIssue(TypedDict):
    paper_id: str
    code: str
    message: str


class HybridHit(RetrievalHit):
    rrf_score: float
    bm25_score: float | None
    bm25_rank: int | None
    dense_score: float | None
    dense_rank: int | None
    retrieval_sources: list[str]


class RetrievalResult(TypedDict):
    query: str
    method: str
    hits: list[RetrievalHit]
    issues: list[RetrievalIssue]


def lexical_tokens(text: str) -> list[str]:
    """Lowercase ASCII words/numbers and overlapping Han bigrams; no translation.

    An isolated Han character is a token too. Longer Han runs use bigrams only,
    avoiding matches based solely on a common character inside unrelated words.
    """
    result = []
    for match in re.finditer(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+", text.lower()):
        term = match.group()
        if "\u3400" <= term[0] <= "\u9fff" and len(term) > 1:
            result.extend(term[i:i + 2] for i in range(len(term) - 1))
        else:
            result.append(term)
    return result


def bm25_retrieve(query: str, chunks: list[dict], k: int, *,
                  tokenizer: Callable[[str], list[str]] = lexical_tokens,
                  include_zero: bool = False) -> list[dict]:
    """Shared BM25 (k1=1.5, b=.75); ties retain input order.

    ``include_zero`` exists for the historical benchmark's fixed-size context.
    Retrieval candidates use the default positive-score-only behavior.
    """
    if not chunks:
        return []
    counts = [Counter(tokenizer(chunk["text"])) for chunk in chunks]
    lengths = [sum(count.values()) for count in counts]
    avg = sum(lengths) / len(lengths) or 1
    df = Counter(term for count in counts for term in count)
    terms = sorted(set(tokenizer(query)))
    scored = []
    for i, count in enumerate(counts):
        score = 0.0
        for term in terms:
            tf = count[term]
            if tf:
                idf = math.log(1 + (len(chunks) - df[term] + .5) / (df[term] + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * lengths[i] / avg))
        if include_zero or score > 0:
            scored.append((score, i))
    return [dict(chunks[i], score=score)
            for score, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:k]]


def rrf_fuse(bm25_hits: list[dict], dense_hits: list[dict], top_k: int,
             rrf_k: int = DEFAULT_RRF_K) -> list[HybridHit]:
    """Fuse one-based ranks; only a route's first occurrence of an ID votes.

    Identity and tie order are (paper_id, source_fingerprint, fragment_id).
    Raw scores are diagnostics and never enter the RRF score.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k 必须为正整数")
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k 必须为非负整数")
    merged = {}
    for route, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
        seen = set()
        for rank, hit in enumerate(hits, 1):
            identity = tuple(hit[k] for k in ("paper_id", "source_fingerprint", "fragment_id"))
            if identity in seen:
                continue
            seen.add(identity)
            if identity not in merged:
                merged[identity] = dict(hit, method="hybrid", score=0.0, rrf_score=0.0,
                    bm25_score=None, bm25_rank=None, dense_score=None, dense_rank=None,
                    retrieval_sources=[])
            row = merged[identity]
            row[route + "_score"] = hit["score"]
            row[route + "_rank"] = rank
            row["retrieval_sources"].append(route)
            row["rrf_score"] += 1.0 / (rrf_k + rank)
            row["score"] = row["rrf_score"]
    return [hit for _, hit in sorted(merged.items(), key=lambda item: (-item[1]["score"], item[0]))[:top_k]]


class Retriever:
    """Bind a corpus, then call retrieve(query, paper_ids, top_k, method).

    ``paper_contexts`` and ``papers`` accept ResearchState's existing mappings;
    papers must expose ``paper_id`` and ``updated``. ``folder`` optionally
    restricts snapshots to that task's context/papers directory. Callers must
    obtain these records through their existing authorization boundary.
    Each call reloads and validates snapshots; no source or state is changed.
    Dense retrieval writes derived caches and lazily loads optional libraries.
    """

    def __init__(self, paper_contexts: Mapping[str, dict], papers: Mapping[str, dict], *,
                 folder: Path | None = None, dense_config=None, embedder=None,
                 dense_cache_dir: Path | None = None):
        self.paper_contexts = paper_contexts
        self.papers = papers
        self.folder = folder
        self.dense_config = dense_config
        self.embedder = embedder
        self.dense_cache_dir = dense_cache_dir
        self._dense = None

    @property
    def dense_stats(self) -> dict:
        """Latest Dense or Hybrid Dense-branch diagnostics, outside the hit contract."""
        import copy
        return copy.deepcopy(self._dense.last_stats) if self._dense is not None else {}

    @staticmethod
    def _fragments(index, texts, record, query) -> list[dict]:
        chunks, seen = [], set()
        references_requested = bool(re.search(r"\b(?:references?|bibliography)\b|参考文献", query.lower()))
        for kind, source in sorted(index["sources"].items()):
            text = texts[kind]
            sections = {s["section_id"]: s for s in source["sections"]}
            for fragment in sorted(source["fragments"], key=lambda f: (f["start"], f["fragment_id"])):
                section = sections[fragment["section_id"]]
                start, end = fragment["start"], fragment["end"]
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                    raise ValueError("片段字符范围无效")
                if not section["start"] <= start < end <= section["end"]:
                    raise ValueError("片段字符范围超出章节")
                if section["is_reference"] and not references_requested:
                    continue
                identity = (kind, fragment["fragment_id"], start, end)
                if identity in seen:
                    continue
                seen.add(identity)
                chunks.append({
                    "paper_id": index["paper_id"], "version": index["version"],
                    "fragment_id": fragment["fragment_id"],
                    "source_fingerprint": record["source_fingerprint"],
                    "source_id": source["source_id"], "source_kind": kind,
                    "source_sha256": source["sha256"], "index_path": record["index_path"],
                    "section_id": section["section_id"], "section": section["name"],
                    "page_start": fragment["page_start"], "page_end": fragment["page_end"],
                    "page_basis": source["page_basis"], "start": start, "end": end,
                    "offset_unit": index["offset_unit"], "text": text[start:end],
                    "method": "bm25", "verification_status": "unverified",
                })
        return chunks

    def retrieve(self, query: str, paper_ids: Sequence[str], top_k: int = 10,
                 method: str = "bm25", *, candidate_k: int = DEFAULT_CANDIDATE_K,
                 rrf_k: int = DEFAULT_RRF_K) -> RetrievalResult:
        """Rank fragments jointly across selected papers; report per-paper failures.

        Empty/no-token/no-match queries yield no hits. Missing/stale/invalid
        materials are excluded and reported in issues, including on partial
        success. Scores are corpus-relative ranking values, not confidence.
        Hybrid uses a shared candidate_k window per route (>= top_k).
        Dense failures retain their issues and any BM25-only partial hits.
        """
        if not isinstance(query, str):
            raise ValueError("query 必须为字符串")
        if isinstance(paper_ids, (str, bytes)) or not isinstance(paper_ids, Sequence) or any(
                not isinstance(pid, str) or not pid for pid in paper_ids):
            raise ValueError("paper_ids 必须为非空论文 ID 的序列")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k 必须为正整数")
        if method not in ("bm25", "dense", "hybrid"):
            raise ValueError("不支持的检索方法：" + str(method))
        if method == "hybrid":
            if type(candidate_k) is not int or candidate_k < top_k:
                raise ValueError("candidate_k 必须为不小于 top_k 的正整数")
            if type(rrf_k) is not int or rrf_k < 0:
                raise ValueError("rrf_k 必须为非负整数")
        result: RetrievalResult = {"query": query, "method": method, "hits": [], "issues": []}
        if method in ("dense", "hybrid") and self._dense is not None:
            self._dense.last_stats = {}
        if not query.strip() or (method == "bm25" and not lexical_tokens(query)):
            return result
        chunks = []
        for pid in sorted(set(paper_ids)):
            def issue(code, message):
                result["issues"].append({"paper_id": pid, "code": code, "message": message})
            paper = self.papers.get(pid)
            record = self.paper_contexts.get(pid)
            if paper is None:
                issue("paper_missing", "未找到论文元数据")
                continue
            if not record:
                issue("index_missing", "论文尚无原文索引")
                continue
            try:
                index, texts = load_index(record, folder=self.folder)
                if index["paper_id"] != pid or paper["paper_id"] != pid:
                    raise ValueError("原文索引与论文 ID 不一致")
                if index["version"] != paper["updated"]:
                    issue("stale_material", "原文索引与论文版本不一致")
                    continue
                chunks.extend(self._fragments(index, texts, record, query))
            except FileNotFoundError as exc:
                issue("material_missing", str(exc))
            except OSError as exc:
                issue("material_unavailable", str(exc))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                issue("invalid_material", str(exc))
        window = candidate_k if method == "hybrid" else top_k
        if method in ("bm25", "hybrid"):
            result["hits"] = bm25_retrieve(query, chunks, window)
        if method in ("dense", "hybrid") and chunks:
            dense_hits = []
            try:
                if self._dense is None:
                    from .dense_retrieval import DenseRetriever
                    self._dense = DenseRetriever(self.dense_config, embedder=self.embedder,
                                                 cache_dir=self.dense_cache_dir)
                dense_hits = self._dense.retrieve(query, chunks, window, result["issues"])
            except Exception as exc:
                result["issues"].append({"paper_id": "", "code": "dense_initialization_failed",
                                         "message": str(exc)})
            result["hits"] = (rrf_fuse(result["hits"], dense_hits, top_k, rrf_k)
                              if method == "hybrid" else dense_hits)
        return result
