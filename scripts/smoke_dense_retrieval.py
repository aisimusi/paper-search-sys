"""Run a real local embedding smoke test over existing validated paper indexes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scholar_daily.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from scholar_daily.paper_context import digest, load_index
from scholar_daily.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-dir", type=Path)
    source.add_argument("--index", type=Path, action="append")
    parser.add_argument("--model", default=EmbeddingConfig.model_id)
    parser.add_argument("--revision", default=EmbeddingConfig.revision)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--document-prefix", default="passage: ")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import faiss
    import numpy as np
    import torch
    from importlib.metadata import version
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    contexts, papers = {}, {}
    folder = args.task_dir.resolve() if args.task_dir else None
    if folder:
        state = json.loads((folder / "state.json").read_text(encoding="utf-8-sig"))
        contexts, papers = state["paper_contexts"], state["papers"]
    else:
        for path in args.index:
            path = path.resolve()
            index = json.loads(path.read_text(encoding="utf-8-sig"))
            pid = index["paper_id"]
            if pid in papers:
                raise ValueError("重复论文 ID；请为每篇论文选一个版本")
            contexts[pid] = {"index_path": str(path), "source_fingerprint": digest(
                json.dumps(index, ensure_ascii=False, sort_keys=True))[:24]}
            papers[pid] = {"paper_id": pid, "updated": index["version"]}
    ids = sorted(contexts)
    if not ids:
        raise ValueError("所选材料没有现有论文索引")
    config = EmbeddingConfig(model_id=args.model, revision=args.revision, dimension=args.dimension,
                             max_length=args.max_length, batch_size=args.batch_size, device=args.device,
                             query_prefix=args.query_prefix, document_prefix=args.document_prefix)
    encoder = SentenceTransformerEmbedder(config)
    started = perf_counter()
    encoder.encode([args.query], query=True)
    load_seconds = perf_counter() - started
    def new_retriever():
        return Retriever(contexts, papers, folder=folder, dense_config=config,
                         embedder=encoder, dense_cache_dir=args.cache_dir.resolve())
    retriever = new_retriever()
    def measure(instance):
        started = perf_counter()
        result = instance.retrieve(args.query, ids, args.top_k, "dense")
        seconds = perf_counter() - started
        if result["issues"] or not result["hits"]:
            raise ValueError("Dense 冒烟检索失败：" + json.dumps(result, ensure_ascii=False))
        for hit in result["hits"]:
            _, texts = load_index(contexts[hit["paper_id"]], folder=folder)
            if texts[hit["source_kind"]][hit["start"]:hit["end"]] != hit["text"]:
                raise ValueError("检索原文与定位范围不一致")
        return seconds, result, instance.dense_stats
    build_seconds, result, build_stats = measure(retriever)
    warm = [measure(retriever)[0] for _ in range(3)]
    reload_seconds, _, reload_stats = measure(new_retriever())
    if any(row["status"] != "hit" for row in reload_stats["papers"]):
        raise ValueError("新检索器未复用磁盘索引")
    payload = {
        "environment": {"platform": platform.platform(), "python": sys.version,
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "threads": args.threads,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "faiss": faiss.__version__, "numpy": np.__version__,
            "sentence_transformers": version("sentence-transformers"), "transformers": version("transformers")},
        "config": asdict(config), "paper_ids": ids,
        "model_load_and_first_query_seconds": load_seconds,
        "first_retrieve_seconds": build_seconds, "build_stats": build_stats,
        "warm_retrieve_seconds": warm, "warm_median_seconds": statistics.median(warm),
        "new_retriever_disk_cache_seconds": reload_seconds, "reload_stats": reload_stats,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "result"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
