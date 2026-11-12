"""Frozen QASPER-native retrieval comparison; does not score production PDF fragments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scholar_daily.agents import get_backend
from scholar_daily.benchmark import digest, paragraphs, read_json, write_json
from scholar_daily.benchmark_agent import Caller
from scholar_daily.benchmark_holdout import checked_inputs
from scholar_daily.benchmark_metrics import ranked_scores
from scholar_daily.benchmark_rerank import INSTRUCTION, select_ids
from scholar_daily.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from scholar_daily.retrieval import bm25_retrieve, rrf_fuse

ARMS = ("bm25", "dense", "hybrid", "hybrid_llm_rerank")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["selections"],
    "properties": {"selections": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["question_id", "paragraph_ids"],
        "properties": {"question_id": {"type": "string"},
                       "paragraph_ids": {"type": "array", "items": {"type": "string"}}}}}}}


def prepare_cases(gold, inputs):
    cases, exclusions, seen_questions = [], [], set()
    for item in inputs:
        qid, pid = item["question_id"], item["paper_id"]
        if qid in seen_questions:
            raise ValueError("重复 question_id")
        seen_questions.add(qid)
        paper = gold[pid]
        qa = next(q for q in paper["qas"] if q["question_id"] == qid)
        original = paragraphs(paper)
        if original != item["chunks"] or qa["question"] != item["question"]:
            raise ValueError("原生段落或问题与 gold 快照不一致")
        # Freeze one common binary qrel set; never choose a reference per method.
        relevant = sorted({text for row in qa["answers"] if not row["answer"]["unanswerable"]
                           for text in row["answer"]["evidence"]})
        if not relevant:
            exclusions.append({"question_id": qid, "reason": "no_positive_evidence"})
            continue
        if not set(relevant) <= {c["text"] for c in original}:
            exclusions.append({"question_id": qid, "reason": "unmapped_evidence"})
            continue
        fingerprint = digest(json.dumps(original, ensure_ascii=False, sort_keys=True).encode())
        chunks, seen_text = [], set()
        for c in original:
            if c["text"] in seen_text:
                continue
            seen_text.add(c["text"])
            chunks.append(dict(c, paper_id=pid, version="qasper-v0.3-native",
                fragment_id=c["id"], source_fingerprint=fingerprint,
                source_id=c["id"], source_kind="qasper_native_paragraph",
                source_sha256=digest(c["text"].encode()), start=0, end=len(c["text"]),
                offset_unit="Unicode offsets local to this QASPER paragraph; not PDF offsets",
                index_path="", verification_status="unverified"))
        cases.append({"question_id": qid, "paper_id": pid, "title": item["title"],
                      "question": item["question"], "chunks": chunks, "relevant_texts": relevant})
    return cases, exclusions


def scores(ranking, relevant):
    if not relevant:
        raise ValueError("检索评测需要非空固定相关性标注")
    refs = [{"evidence": relevant}]
    return {"recall_at_5": ranked_scores(ranking, refs, 5)["recall"],
            "ndcg_at_10": ranked_scores(ranking, refs, 10)["ndcg"]}


def rerank_payload(case, candidates):
    return [{"question_id": "q0", "title": case["title"], "question": case["question"],
             "select_count": len(candidates),
             "paragraphs": [{k: c[k] for k in ("id", "section", "text")} for c in candidates]}]


def select_reranked(output, candidates):
    ids = select_ids([dict(output, question_id="q0")], [{"paragraphs": candidates}], len(candidates))[0]
    by_id = {c["id"]: c for c in candidates}
    return [by_id[cid] for cid in ids]


def paired_delta(rows, arm, metric, seed=20261020):
    values = [r["arms"][arm]["metrics"][metric] - r["arms"]["bm25"]["metrics"][metric] for r in rows]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return {"delta": statistics.mean(values), "bootstrap_95": [draws[125], draws[4874]],
            "wins": sum(v > 0 for v in values), "ties": sum(v == 0 for v in values),
            "losses": sum(v < 0 for v in values)}


def summarize(rows):
    summary = {}
    for arm in ARMS:
        milliseconds = sorted(r["arms"][arm]["seconds"] * 1000 for r in rows)
        summary[arm] = {metric: statistics.mean(r["arms"][arm]["metrics"][metric] for r in rows)
                        for metric in ("recall_at_5", "ndcg_at_10")}
        summary[arm].update(questions=len(rows), failures=sum(bool(r["arms"][arm].get("error")) for r in rows),
            query_ms_median=statistics.median(milliseconds),
            query_ms_p95=milliseconds[math.ceil(.95 * len(milliseconds)) - 1])
    return summary


def run(args):
    import faiss
    import torch
    from scholar_daily.dense_retrieval import DenseRetriever
    root, study = args.output.resolve(), args.study.resolve()
    root.mkdir(parents=True, exist_ok=True)
    gold_path = study / args.split / "gold.json"
    reservation = read_json(study / "reservation.json")
    if digest(gold_path.read_bytes()) != reservation["hashes"][args.split]["gold"]:
        raise ValueError("Gold 快照指纹不匹配")
    inputs, gold = checked_inputs(study, args.split), read_json(gold_path)
    cases, exclusions = prepare_cases(gold, inputs)
    if not cases:
        raise ValueError("没有统一标注的可评测题目")
    config = EmbeddingConfig(model_id=args.model)
    model_config = read_json(args.rerank_config)
    protocol = {"source_study": str(study), "split": args.split,
        "source_hashes": reservation["hashes"][args.split],
        "unit": "QASPER native paragraphs, exact-text duplicates canonicalized per paper",
        "scope": "each query searches all native paragraphs of its own paper, all four arms identical",
        "qrels": "fixed union of answerable annotators' evidence; binary; unannotated paragraphs score 0",
        "question_ids": [c["question_id"] for c in cases], "excluded": exclusions,
        "candidate_k": 20, "rrf_k": 60, "metrics": ["Recall@5", "binary nDCG@10"],
        "rerank": "existing INSTRUCTION/select_ids; full permutation of Hybrid Top-20; no answers/qrels supplied",
        "rerank_instruction_sha256": digest(INSTRUCTION.encode()), "rerank_config": model_config,
        "embedding": asdict(config), "threads": args.threads, "latency_repeats": 3,
        "latency": "warm, corpus resident, model/index build excluded; Dense includes query encoding and FAISS cache validation; rerank includes Hybrid plus all LLM attempts",
        "diagnostic_windows": [10, 20, 50], "window_analysis": "exploratory on same historical sample; no production tuning",
        "historical_sample": "previously used holdout; current rerun is not a newly sealed unseen evaluation",
        "runner_sha256": digest(Path(__file__).read_bytes())}
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("冻结的评测协议已变化；请使用新目录")
    write_json(protocol_path, protocol)
    write_json(root / "qrels.json", [{k: c[k] for k in ("question_id", "paper_id", "relevant_texts")} for c in cases])
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    encoder = SentenceTransformerEmbedder(config)
    begun = perf_counter()
    encoder.encode([cases[0]["question"]], query=True)
    load_seconds = perf_counter() - begun
    dense = DenseRetriever(config, embedder=encoder, cache_dir=root / "native-vector-cache")
    model = encoder._load()
    tokenizer = model.tokenizer
    diagnostics, rankings = [], {}
    build_seconds = 0.0
    for i, case in enumerate(cases):
        issues = []
        begun = perf_counter()
        full_dense = dense.retrieve(case["question"], case["chunks"], len(case["chunks"]), issues)
        build_seconds += perf_counter() - begun
        if issues:
            raise ValueError("Dense 构建失败：" + json.dumps(issues, ensure_ascii=False))
        full_bm25 = bm25_retrieve(case["question"], case["chunks"], len(case["chunks"]))
        rankings[case["question_id"]] = {"bm25": full_bm25, "dense": full_dense}
        lengths = [len(tokenizer(config.document_prefix + c["text"], truncation=False,
                                 add_special_tokens=True, verbose=False)["input_ids"]) for c in case["chunks"]]
        diagnostics.append({"question_id": case["question_id"], "query": case["question"],
            "query_non_ascii": any(ord(c) > 127 for c in case["question"]),
            "paragraphs": [{"id": c["id"], "characters": len(c["text"]), "tokens_with_prefix": length,
                "truncated": length > config.max_length, "relevant": c["text"] in case["relevant_texts"]}
                for c, length in zip(case["chunks"], lengths)], "build_stats": dense.last_stats})
        print(f"index {i+1}/{len(cases)}", flush=True)
    write_json(root / "input-diagnostics.json", diagnostics)
    workspace = root / "llm-workspace"
    workspace.mkdir(exist_ok=True)
    backend = get_backend(model_config, workspace)
    if backend is None or getattr(backend, "resumable", False):
        raise ValueError("需要可执行的既有 LLM 后端")
    caller = Caller(backend, root, {"max_call_attempts": 2})
    rows = []
    try:
        for i, case in enumerate(cases):
            result_path = root / "questions" / f"{i:04d}.json"
            if result_path.exists():
                row = read_json(result_path)
                if row["question_id"] != case["question_id"]:
                    raise ValueError("已保存的题目与协议不一致")
                rows.append(row)
                continue
            qid = case["question_id"]
            latency = {arm: [] for arm in ARMS[:3]}
            measured = {}
            def retrieve(arm):
                bm25 = bm25_retrieve(case["question"], case["chunks"], 20) if arm != "dense" else []
                semantic = []
                if arm != "bm25":
                    issues = []
                    semantic = dense.retrieve(case["question"], case["chunks"], 20, issues)
                    if issues or any(p["status"] != "hit" for p in dense.last_stats["papers"]):
                        raise ValueError("暖查询发生错误或重建，不能计入该延迟口径")
                return rrf_fuse(bm25, semantic, 20) if arm == "hybrid" else semantic if arm == "dense" else bm25
            for repeat in range(3):
                order = list(ARMS[:3])
                order = order[repeat:] + order[:repeat]
                for arm in order:
                    begun = perf_counter()
                    ranking = retrieve(arm)
                    latency[arm].append(perf_counter() - begun)
                    measured[arm] = ranking
            row = {"question_id": qid, "paper_id": case["paper_id"], "question": case["question"],
                   "relevant_ids": [c["id"] for c in case["chunks"] if c["text"] in case["relevant_texts"]], "arms": {}}
            for arm in ARMS[:3]:
                row["arms"][arm] = {"metrics": scores(measured[arm], case["relevant_texts"]),
                    "ranking": [c["id"] for c in measured[arm]], "seconds": statistics.median(latency[arm]),
                    "repeat_seconds": latency[arm]}
            candidates = measured["hybrid"]
            payload = rerank_payload(case, candidates)
            output, error = [], None
            try:
                result = caller.call(f"{i:04d}/rerank", INSTRUCTION, payload, SCHEMA, "selections",
                                     lambda value, _: select_reranked(value, candidates))
                output = result["q0"]
            except Exception as exc:
                error = str(exc)
            records = [read_json(p) for p in (root / "calls" / f"{i:04d}" / "rerank").glob("*/record.json")]
            llm_seconds = sum(r["wall_seconds"] for r in records)
            row["arms"]["hybrid_llm_rerank"] = {"metrics": scores(output, case["relevant_texts"]),
                "ranking": [c["id"] for c in output], "seconds": latency["hybrid"][-1] + llm_seconds,
                "llm_seconds": llm_seconds, "attempts": len(records), "error": error}
            row["window_diagnostics"] = {}
            for window in (10, 20, 50):
                hybrid = rrf_fuse(rankings[qid]["bm25"][:window], rankings[qid]["dense"][:window], 20)
                row["window_diagnostics"][str(window)] = scores(hybrid, case["relevant_texts"])
            refs = [{"evidence": case["relevant_texts"]}]
            row["hybrid_candidate_recall_at_20"] = ranked_scores(candidates, refs, 20)["recall"]
            write_json(result_path, row)
            rows.append(row)
            print(f"score {i+1}/{len(cases)} rerank_error={bool(error)}", flush=True)
    finally:
        if hasattr(backend, "close"):
            backend.close()
    summary = summarize(rows)
    all_paragraphs = [p for d in diagnostics for p in d["paragraphs"]]
    relevant = [p for p in all_paragraphs if p["relevant"]]
    truncation = {"paragraphs": len(all_paragraphs), "truncated": sum(p["truncated"] for p in all_paragraphs),
        "relevant_paragraphs": len(relevant), "relevant_truncated": sum(p["truncated"] for p in relevant),
        "characters_median": statistics.median(p["characters"] for p in all_paragraphs),
        "characters_max": max(p["characters"] for p in all_paragraphs),
        "non_ascii_queries": sum(d["query_non_ascii"] for d in diagnostics)}
    payload = {"protocol": protocol, "summary": summary, "truncation": truncation,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "faiss": faiss.__version__, "threads": args.threads},
        "model_load_seconds": load_seconds, "index_build_and_prewarm_seconds": build_seconds,
        "paired_vs_bm25": {arm: {metric: paired_delta(rows, arm, metric) for metric in ("recall_at_5", "ndcg_at_10")}
                           for arm in ARMS[1:]},
        "window_sensitivity": {str(w): {metric: statistics.mean(r["window_diagnostics"][str(w)][metric] for r in rows)
                                       for metric in ("recall_at_5", "ndcg_at_10")} for w in (10, 20, 50)},
        "hybrid_candidate_recall_at_20": statistics.mean(r["hybrid_candidate_recall_at_20"] for r in rows),
        "pdf_evaluation": {"status": "awaiting_human_relevance_labels", "scored": False}, "questions": rows}
    write_json(root / "results.json", payload)
    lines = ["# QASPER 原生段落检索对照", "", f"固定复测题 {len(rows)}；统一排除 {len(exclusions)} 题。每题只检索其论文的原生段落全集。",
        "统一采用可回答标注的证据段落并集作为二值 qrels，未标注段落按 0 计分。历史 holdout 已被使用，本次不是新的未见测试集。",
        "PDF 字符片段缺少人工相关性标注，未计算 PDF 检索质量指标；本报告不使用或混合 Evidence F1。", "",
        "| 对照组 | Recall@5 | nDCG@10 | 查询延迟 P50 | P95 | 失败题 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, values in summary.items():
        lines.append(f"| {arm} | {values['recall_at_5']:.4f} | {values['ndcg_at_10']:.4f} | "
                     f"{values['query_ms_median']:.2f} ms | {values['query_ms_p95']:.2f} ms | {values['failures']} |")
    lines += ["", "前三组计时为模型及索引预热后的单题查询，包含查询编码和缓存校验，不含模型加载及首次建索引。",
        "第四组包含 Hybrid 查询和已有 LLM Rerank 的全部尝试；LLM 对同一 Hybrid Top-20 完整排序，未收到答案或 qrels。",
        "全部四组使用相同题目、候选全集和 qrels。错误题保留在分母中，以空排名计分。", "",
        f"截断诊断：{truncation['truncated']}/{truncation['paragraphs']} 段超过 512 token；"
        f"相关段落中 {truncation['relevant_truncated']}/{truncation['relevant_paragraphs']} 段超过限制。",
        "窗口敏感性、成对 bootstrap 区间、逐题排名和调用记录见 results.json、questions/ 和 calls/。",
        "窗口分析是同一历史样本上的探索性检查，不据此选择并宣称最优生产参数。"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rerank-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
