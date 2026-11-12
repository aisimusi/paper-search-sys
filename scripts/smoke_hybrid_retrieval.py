"""Compare BM25/Dense/Hybrid on existing indexes with a real local model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scholar_daily.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from scholar_daily.paper_context import digest, load_index
from scholar_daily.research_calls import read_json, write_json
from scholar_daily.retrieval import Retriever


def identity(hit):
    return tuple(hit[k] for k in ("paper_id", "source_fingerprint", "fragment_id"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query", action="append")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k or args.rrf_k < 0:
        parser.error("要求 1 <= top-k <= candidate-k 且 rrf-k >= 0")
    import faiss
    import torch
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    contexts, papers, texts, titles = {}, {}, {}, {}
    for path in args.index:
        index = read_json(path)
        pid = index["paper_id"]
        if pid in contexts:
            raise ValueError("每篇论文只能选择一个索引版本")
        contexts[pid] = {"index_path": str(path.resolve()), "source_fingerprint": digest(
            json.dumps(index, ensure_ascii=False, sort_keys=True))[:24]}
        papers[pid] = {"paper_id": pid, "updated": index["version"]}
        titles[pid] = index["title"]
        _, texts[pid] = load_index(contexts[pid])
    config = EmbeddingConfig(model_id=args.model)
    encoder = SentenceTransformerEmbedder(config)
    retriever = Retriever(contexts, papers, dense_config=config, embedder=encoder,
                          dense_cache_dir=args.cache_dir.resolve())
    queries = args.query or ["inference latency", "如何降低实时目标检测的推理延迟？",
                            "spatial prior identity association", "目标身份关联与空间先验"]
    started = perf_counter()
    encoder.encode([queries[0]], query=True)
    load_seconds = perf_counter() - started
    comparisons = []
    for query in queries:
        candidates, results, timings, stats = {}, {}, {}, {}
        for method in ("bm25", "dense", "hybrid"):
            started = perf_counter()
            result = retriever.retrieve(query, sorted(papers),
                args.top_k if method == "hybrid" else args.candidate_k, method,
                candidate_k=args.candidate_k, rrf_k=args.rrf_k)
            timings[method] = perf_counter() - started
            if result["issues"] or (method != "bm25" and not result["hits"]):
                raise ValueError("冒烟检索失败：" + json.dumps(result, ensure_ascii=False))
            for row in result["hits"]:
                if (row["verification_status"] != "unverified" or
                        texts[row["paper_id"]][row["source_kind"]][row["start"]:row["end"]] != row["text"]):
                    raise ValueError("原文定位或未核查状态错误")
            if method != "hybrid":
                candidates[method] = result
            if method in {"dense", "hybrid"}:
                stats[method] = retriever.dense_stats
            results[method] = dict(result, hits=result["hits"][:args.top_k])
        ranks = {route: {identity(h): (rank, h["score"]) for rank, h in enumerate(value["hits"], 1)}
                 for route, value in candidates.items()}
        expected = {}
        for route in ranks.values():
            for key, (rank, _) in route.items():
                expected[key] = expected.get(key, 0.0) + 1 / (args.rrf_k + rank)
        ordered = sorted(expected, key=lambda key: (-expected[key], key))[:args.top_k]
        if [identity(h) for h in results["hybrid"]["hits"]] != ordered:
            raise ValueError("融合排名与两路独立候选不一致")
        for row in results["hybrid"]["hits"]:
            key = identity(row)
            for route in ("bm25", "dense"):
                rank, score = ranks[route].get(key, (None, None))
                if (row[route + "_rank"], row[route + "_score"]) != (rank, score):
                    raise ValueError("分路排名或原始分数诊断错误")
            if not math.isclose(row["rrf_score"], expected[key]) or row["score"] != row["rrf_score"]:
                raise ValueError("RRF 分数错误")
        comparisons.append({"query": query, "candidate_results": candidates, "results": results,
                            "candidate_retrieval_seconds": timings, "dense_stats": stats})
    payload = {"environment": {"platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "faiss": faiss.__version__, "threads": args.threads,
        "sentence_transformers": version("sentence-transformers"), "transformers": version("transformers")},
        "config": asdict(config), "top_k": args.top_k, "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k, "papers": titles, "model_load_seconds": load_seconds,
        "comparisons": comparisons}
    write_json(args.output, payload)
    lines = ["# Hybrid 实际样例对照", "", "链路与定位验收；未标注相关性，结果不代表检索质量提升。",
             "", f"配置：Top-K={args.top_k}，每路候选={args.candidate_k}，rrf_k={args.rrf_k}。",
             "纯检索方法先取得相同候选窗口，再展示前 Top-K；JSON 保留完整候选、分路分数与排名。", "",
             "| 查询 | BM25 前三 | Dense 前三 | Hybrid 前三 |", "|---|---|---|---|"]
    def label(row):
        return titles[row["paper_id"]].split(":")[0] + " / " + row["fragment_id"].rsplit(":", 1)[-1]
    for item in comparisons:
        cells = ["；".join(label(h) for h in item["results"][method]["hits"][:3]) or "无词项命中"
                 for method in ("bm25", "dense", "hybrid")]
        lines.append("| " + " | ".join([item["query"].replace("|", "\\|"), *cells]) + " |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "queries": len(comparisons),
        "hybrid_hits": [len(c["results"]["hybrid"]["hits"]) for c in comparisons],
        "model_load_seconds": load_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
