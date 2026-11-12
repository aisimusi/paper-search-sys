"""Read-only paired comparison of QASPER development snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path
import random

from .benchmark import (answer_diagnostics, checked_snapshot, digest, load_evaluator,
                        read_json, retrieval_metrics, write_json)
from .pipeline import atomic_text


def paired_delta(reference: list[dict], candidate: list[dict], seed: int = 20261012):
    if [r["question_id"] for r in reference] != [r["question_id"] for r in candidate]:
        raise ValueError("Paired question IDs/order differ")
    deltas = [b["official_answer_f1"] - a["official_answer_f1"] for a, b in zip(reference, candidate)]
    if not deltas:
        raise ValueError("Empty paired comparison")
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
    return {"questions": len(deltas), "mean_delta": sum(deltas) / len(deltas),
            "bootstrap_95_percentile_interval": [samples[49], samples[1949]],
            "bootstrap_samples": 2000, "seed": seed,
            "improved": sum(d > 1e-12 for d in deltas), "worse": sum(d < -1e-12 for d in deltas)}


def inspect_run(root: Path):
    manifest, raw_gold, tasks = checked_snapshot(root, "dev")
    evaluator = load_evaluator(Path(manifest["data_dir"]))
    gold = evaluator.get_answers_and_evidence(raw_gold, False)
    predictions, seen, results = {}, set(), []
    for path in sorted((root / "dev/batches").glob("*/result.json")):
        result = read_json(path)
        ids = result["question_ids"]
        predicted, errors = set(result["predictions"]), set(result["errors"])
        if (len(ids) != len(set(ids)) or seen.intersection(ids) or not set(ids) <= set(gold)
                or predicted & errors or predicted | errors != set(ids)):
            raise ValueError("Inconsistent batch IDs")
        seen.update(ids)
        predictions.update(result["predictions"])
        results.append(result)
    diagnostics, rows = answer_diagnostics(evaluator, gold, predictions, tasks)
    config = read_json(root / "run-config.json")
    summary = {"path": str(root.resolve()), "questions": len(gold), "completed": len(predictions),
               "failed": sum(len(r["errors"]) for r in results), "pending": len(gold) - len(seen),
               "official": evaluator.evaluate(gold, predictions), "diagnostics": diagnostics,
               "retrieval": retrieval_metrics(evaluator, raw_gold, tasks), "configuration": config,
               "gold_sha256": manifest["splits"]["dev"]["gold_sha256"],
               "manifest_sha256": digest((root / "manifest.json").read_bytes()),
               "batch_wall_seconds_total": sum(r["wall_seconds"] for r in results)}
    usage = {}
    for result in results:
        for event in result["usage"]:
            for key, value in event.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
    retry_results = [read_json(p) for p in sorted((root / "attempts/dev").glob("*/result.json"))]
    summary["retried_batches"] = len(retry_results)
    summary["first_pass"] = read_json(root / "first-pass-metrics.json")["dev"] if (root / "first-pass-metrics.json").exists() else None
    for result in retry_results:
        summary["batch_wall_seconds_total"] += result["wall_seconds"]
        for event in result["usage"]:
            for key, value in event.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
    summary["answer_usage"] = dict(usage)
    rerank_results = [read_json(p) for p in sorted((root / "retrieval-stage/batches").glob("*/result.json"))]
    rerank_usage = {}
    for result in rerank_results:
        for event in result["usage"]:
            for key, value in event.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    rerank_usage[key] = rerank_usage.get(key, 0) + value
                    usage[key] = usage.get(key, 0) + value
    summary["rerank_usage"] = rerank_usage or None
    summary["rerank_wall_seconds_total"] = sum(r["wall_seconds"] for r in rerank_results)
    all_attempts = results + retry_results + rerank_results
    summary["usage_coverage"] = {"attempts": len(all_attempts),
                                 "attempts_with_usage": sum(bool(r["usage"]) for r in all_attempts)}
    summary["usage"] = usage
    return summary, rows


def compare(paths: list[Path], output: Path):
    names = [p.name for p in paths]
    if len(names) != len(set(names)):
        raise ValueError("Run directory names must be unique")
    output = output.resolve()
    if any(output == p.resolve() or p.resolve() in output.parents for p in paths):
        raise ValueError("Comparison output must be outside the source run directories")
    runs, all_rows = {}, {}
    for path in paths:
        summary, rows = inspect_run(path)
        if runs and summary["gold_sha256"] != next(iter(runs.values()))["gold_sha256"]:
            raise ValueError("Gold snapshots differ")
        runs[path.name], all_rows[path.name] = summary, rows
    reference = names[0]
    pairs = {name: paired_delta(all_rows[reference], all_rows[name]) for name in names[1:]}
    payload = {"split": "dev", "reference": reference, "runs": runs, "paired": pairs}
    write_json(output / "comparison.json", payload)
    for name, rows in all_rows.items():
        write_json(output / f"{name}-diagnostics.json", rows)
    lines = ["# QASPER 固定开发集对照", "", "| 设置 | 完成/总题数 | 官方 Answer F1 | 字符规范化诊断 F1 | 官方 Evidence F1 | 一致可回答题 Answer F1 | 证据召回 | 平均段数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in runs.items():
        d, ev = r["diagnostics"], r["retrieval"]
        answerable = d["groups"]["unanimous_answerable"]["official_answer_f1"]
        answerable_text = f"{answerable:.2%}" if answerable is not None else "—"
        lines.append(f"| {name} | {r['completed']}/{r['questions']} | {r['official']['Answer F1']:.2%} | {d['character_normalized_answer_f1']:.2%} | {r['official']['Evidence F1']:.2%} | {answerable_text} | {ev['max_reference_recall_at_k']:.2%} | {ev['mean_context_paragraphs']:.2f} |")
    lines.extend(["", f"配对参照：{reference}。缺失/失败预测按零分计；逐题结果、模型配置、用量及完整状态见 comparison.json。",
                  "", "| 设置 | Answer F1 差值（百分点） | 配对 bootstrap 95% 区间（百分点） | 提升/下降题数 |", "|---|---:|---:|---:|"])
    for name, pair in pairs.items():
        low, high = pair["bootstrap_95_percentile_interval"]
        lines.append(f"| {name} | {100*pair['mean_delta']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] | {pair['improved']}/{pair['worse']} |")
    lines.extend(["", "| 设置 | 失败/待运行 | 输入 tokens（含缓存命中） | 缓存命中 tokens | 输出 tokens | 批次墙钟时间累计（秒） |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, r in runs.items():
        u = r["usage"]
        # Codex input_tokens includes cached_input_tokens; Harness reports them separately.
        inputs = u.get("input_tokens", "未知")
        if "inputTokens" in u and "cacheReadTokens" in u:
            inputs = u["inputTokens"] + u["cacheReadTokens"]
        cached = u.get("cached_input_tokens", u.get("cacheReadTokens", "未知"))
        outputs = u.get("output_tokens", u.get("outputTokens", "未知"))
        seconds = r["batch_wall_seconds_total"] + r["rerank_wall_seconds_total"]
        lines.append(f"| {name} | {r['failed']}/{r['pending']} | {inputs} | {cached} | {outputs} | {seconds:.1f} |")
    lines.extend(["", "重排条件用量与时间包含检索重排及回答两个阶段；各阶段原始计数另存 JSON。上述时间为各批次累计，受并发和网络影响，不能作为模型速度排名；费用未知。"])
    for name, r in runs.items():
        if r["usage_coverage"]["attempts"] != r["usage_coverage"]["attempts_with_usage"]:
            lines.append(f"{name}：部分调用缺少用量记录，表中计数仅代表已记录部分。")
        if r["first_pass"] is not None:
            first = r["first_pass"]
            lines.append(f"{name}：首次通过 {first['completed']}/{first['questions']}，首次官方 Answer F1 {first['official']['Answer F1']:.2%}；{r['retried_batches']} 个全批无效响应批次按统一规则重试一次，最终结果列在主表，用量包含原失败尝试。原响应与评分见该运行的 attempts/ 和 first-pass-metrics.json。")
    lines.extend(["", "区间在固定题目上重采样 2000 次，描述样本不确定性；未估计模型重跑波动，未校正多配置选择偏差。此开发集用于选型，不能替代冻结配置后的独立评测。",
                  "字符规范化分数只诊断文本匹配差异，不替代官方分数。Oracle 提供所有可回答标注的证据并集，空上下文透露不可回答性；一致可回答子组应同时报告。Oracle 分差包含证据选择、噪声及回答交互作用。",
                  "原评分器使用下载包内经 SHA-256 校验的 QASPER v0.3 文件；数据与下载地址见 [AllenAI QASPER](https://huggingface.co/datasets/allenai/qasper/blob/main/qasper.py)。"])
    atomic_text(output / "comparison.md", "\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="First run is the paired reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.runs, args.output)
    print(args.output / "comparison.md")


if __name__ == "__main__":
    main()
