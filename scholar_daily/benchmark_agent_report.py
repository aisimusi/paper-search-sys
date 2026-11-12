"""Matched module ablations, official scores, support diagnostics and complete call costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .benchmark import digest, read_json, write_json
from .benchmark_agent import FIVE_ARMS, check_frozen, implementation_hashes, study_arms
from .benchmark_compare import paired_delta
from .pipeline import atomic_text

LABELS = {"rag5": "Top-5", "rag20": "Top-20", "rag20_gate": "Top-20＋审查拒答",
          "rag20_repair": "Top-20＋修复", "agent": "完整动态 Agent",
          "llm": "直接 LLM 回答", "rag": "单轮检索＋回答", "rerank": "检索＋重排＋回答",
          "dynamic": "检索＋动态 Agent", "verified": "完整 Agent＋证据审查"}


def sum_costs(costs):
    if not costs or any(cost is None for cost in costs) or len({c["currency"] for c in costs}) != 1:
        return None
    return {"amount": sum(c["amount"] for c in costs), "currency": costs[0]["currency"]}


def extended_tables(runs, policy):
    percent = lambda v: "—" if v is None else f"{v:.2%}"
    number = lambda v: "—" if v is None else f"{v:.3f}"
    lines = ["", "| 条件 | Answer EM | Evidence Precision | Evidence Recall | 任务成功率 | 任务修复成功/初始失败 | 人工支持/已核验 | 人工支持率 | 人工覆盖率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, r in runs.items():
        q, h = r.get("quality"), r.get("human_support")
        if q is None or h is None:
            continue
        lines.append(f"| {LABELS[arm]} | {percent(q['answer_exact_match'])} | {percent(q['evidence_precision'])} | {percent(q['evidence_recall'])} | {percent(q['task_success_rate'])} | {q['recovered_tasks']}/{q['initial_task_failures']}（{percent(q['failure_recovery_rate'])}） | {h['supported']}/{h['reviewed']} | {percent(h['support_rate'])} | {percent(h['coverage'])} |")
    lines.extend(["", "EM 沿用官方答案规范化并对多参考取最大值，属于补充指标。Evidence Precision/Recall 使用同一个参考证据集：优先取 Evidence F1 最大者，再依次按 Recall、Precision、标注顺序消除平局；空预测与空参考都记 1。失败及待完成题的质量得分记 0，分母为全部预留题目。",
                  "任务成功要求规范化答案和完整证据集合同时匹配同一条参考标注；它衡量 QASPER 问答与证据定位的严格成功率。任务修复率的分母为已产生初始答案且初始未成功的题目，分子为其中最终成功的题目。人工支持率只统计已核验的正面答案，uncertain 计入分母，pending 不计入；必须结合覆盖率解读。人工表与答案/引用指纹绑定。",
                  "", "| 条件 | 阶段 | K | 有证据参考题数 | Recall@K | MRR@K | nDCG@K | 完整参考覆盖 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    stage_labels = {"candidate": "初检候选", "initial": "首次回答上下文", "final": "最终上下文"}
    for arm, r in runs.items():
        q = r.get("quality")
        if not q or not q["retrieval_applicable"]:
            continue
        for stage, values in q["retrieval"].items():
            for k, m in values.items():
                lines.append(f"| {LABELS[arm]} | {stage_labels[stage]} | {k} | {m['questions']} | {percent(m['recall'])} | {number(m['mrr'])} | {number(m['ndcg'])} | {percent(m['complete_reference'])} |")
    lines.extend(["", "检索指标仅纳入至少含一份非空参考证据的题目；每项指标分别对参考集取最大值，再按题宏平均。段落以原文文本去重，相关性为二值。K 截断实际可用列表，最终上下文按检索/补读进入顺序排列；未召回的题目记 0。直接 LLM 组检索指标不适用。",
                  "", "| 条件 | 总运行秒数 | 平均题延迟秒 | P95 题延迟秒 | 平均在线调用/题 | 平均全链路调用/题 | 调用恢复/失败组 | 在线费用/题 | 评测审查费用 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for arm, r in runs.items():
        p = r.get("performance")
        if not p:
            continue
        currency = p['cost_currency'] or ''
        cost = "—" if p['mean_task_cost'] is None else f"{p['mean_task_cost']:.6f} {currency}"
        audit = "—" if p['evaluation_cost'] is None else f"{p['evaluation_cost']['amount']:.6f} {currency}"
        lines.append(f"| {LABELS[arm]} | {number(p['total_wall_seconds'])} | {number(p['question_latency_mean_seconds'])} | {number(p['question_latency_p95_seconds'])} | {number(p['mean_online_model_attempts'])} | {number(p['mean_model_attempts'])} | {p['recovered_call_groups']}/{p['failed_call_groups']}（{percent(p['call_failure_recovery_rate'])}） | {cost} | {audit} |")
    lines.extend(["", "总运行延迟累计各次 run 调用的实测耗时，包含并行与恢复执行；批次累计时间另表报告。逐题延迟采用该题所在同步批次的推理完成时间，包含检索和失败重试，最终诊断审查耗时单独记录；P95 使用最近秩法。默认五组评测每批一题。批量运行时多题共享批次延迟；复用或恢复缓存会缩短实测耗时，应结合调用账本与复用数解读。历史批次缺少独立推理计时则该延迟记未知，计时覆盖题数见 performance.latency_questions。",
                  "在线调用与在线费用包含重排、回答、规划、修复和在线审查的失败尝试，单列最终评测审查费用；平均调用次数是模型尝试数/全部题数。调用恢复率以发生过失败尝试的逻辑调用为分母，以后续成功的逻辑调用为分子。",
                  "费用使用 policy.json 中指定币种及每百万 token 单价，按输入、缓存输入和输出分别计价；Codex 的输入总数包含缓存，Harness 的 inputTokens 为未缓存输入。没有价格或任一调用缺少必要用量时，费用为未知。Token 合计保持后端定义；缓存与推理 token 不重复加到输入/输出。"])
    return lines


def report(study, split):
    if split == "holdout":
        check_frozen(study)
    policy = read_json(study / "policy.json")
    arms = study_arms(policy)
    reservation = read_json(study / "reservation.json")
    evaluator = Path(reservation["data_dir"]) / "qasper_evaluator.py"
    runs, rows, checks, states = {}, {}, {}, {}
    for arm in arms:
        root = study / "runs" / split / arm
        config = read_json(root / "run-config.json")
        if (config["policy_sha256"] != digest((study / "policy.json").read_bytes())
                or config["implementation"] != implementation_hashes()
                or config["input_sha256"] != digest((study / split / "inputs.json").read_bytes())):
            raise ValueError("Report configuration or implementation changed")
        runs[arm], rows[arm] = read_json(root / "metrics.json"), read_json(root / "diagnostics.json")
        if runs[arm]["pending"] or runs[arm]["failed"]:
            raise ValueError("Comparison requires complete successful runs")
        states[arm] = {qid: state for p in sorted((root / "batches").glob("*/result.json"))
                       for qid, state in read_json(p)["states"].items()}
        output = subprocess.check_output([sys.executable, "-X", "utf8", str(evaluator),
                    "--gold", str(study / split / "gold.json"), "--predictions", str(root / "predictions.jsonl")],
                    text=True, encoding="utf-8")
        checks[arm] = json.loads(output) == runs[arm]["official"]
        if not checks[arm]:
            raise ValueError("Official CLI scores disagree")
    comparisons = [("rag20", "rag20_gate", "审查"), ("rag20_gate", "rag20_repair", "修复"),
                   ("rag20_repair", "agent", "补检索"), ("rag20", "agent", "整体")]
    five_arm = tuple(arms) == FIVE_ARMS
    if five_arm:
        comparisons = [("llm", "rag", "检索增强"), ("rag", "rerank", "重排"),
                       ("rerank", "dynamic", "主动补检索"), ("dynamic", "verified", "证据质量控制")]
    elif "rag5" in arms:
        comparisons = [("rag5", "rag20", "Top-K"), ("rag20", "agent", "整体"), ("rag5", "agent", "整体对 Top-5")]
    pairs, effects = {}, {}
    for reference, candidate, module in comparisons:
        name = f"{candidate}_minus_{reference}"
        pairs[name] = paired_delta(rows[reference], rows[candidate])
        effect = {"module": module, "answer_f1": pairs[name]}
        for measure in ("abstention", "supported_answer"):
            binary = {}
            for arm in (reference, candidate):
                binary[arm] = []
                for row in rows[arm]:
                    state = states[arm][row["question_id"]]
                    abstained = state["prediction"]["answer"] == "Unanswerable"
                    value = abstained if measure == "abstention" else not abstained and state["final_review"]["passed"]
                    binary[arm].append({"question_id": row["question_id"], "official_answer_f1": int(value)})
            effect[measure] = paired_delta(binary[reference], binary[candidate])
        effect["calls_delta"] = runs[candidate]["calls"] - runs[reference]["calls"]
        effect["usage_delta"] = {k: runs[candidate]["usage"].get(k, 0) - runs[reference]["usage"].get(k, 0)
                                 for k in set(runs[reference]["usage"]) | set(runs[candidate]["usage"])}
        effects[name] = effect
    integrity = all(r["completed"] == r["questions"] for r in runs.values())
    quality_pair = "verified_minus_dynamic" if five_arm else "agent_minus_rag20_repair" if "rag20_repair" in arms else "agent_minus_rag20"
    cost_complete = all(r.get("performance", {}).get("usage_complete", r["calls_with_usage"] == r["calls"]) for r in runs.values())
    acceptance = {"integrity_passed": integrity, "cost_usage_complete": cost_complete,
                  "quality_gain_demonstrated": integrity and pairs[quality_pair]["bootstrap_95_percentile_interval"][0] > 0,
                  "quality_rule": policy["acceptance"]["quality"],
                  "semantic_support": "same-model automatic diagnostic, not human-validated accuracy"}
    actual_usage = {}
    for r in runs.values():
        for k, v in r.get("actual_usage", r["usage"]).items():
            actual_usage[k] = actual_usage.get(k, 0) + v
    actual_cost = {"calls": sum(r.get("actual_calls", r["calls"]) for r in runs.values()),
                   "usage": actual_usage, "monetary_cost": sum_costs([r.get("monetary_cost") for r in runs.values()])}
    study_cost = {"calls": 0, "usage": {}, "by_split": {}, "monetary_cost": None,
                  "usage_complete": True}
    for metric_path in sorted((study / "runs").glob("*/*/metrics.json")):
        metric = read_json(metric_path)
        study_cost["usage_complete"] &= metric["calls_with_usage"] == metric["calls"]
        split_cost = study_cost["by_split"].setdefault(metric_path.parent.parent.name, {"calls": 0, "usage": {}})
        calls = metric.get("actual_calls", metric["calls"])
        study_cost["calls"] += calls
        split_cost["calls"] += calls
        for key, count in metric.get("actual_usage", metric["usage"]).items():
            study_cost["usage"][key] = study_cost["usage"].get(key, 0) + count
            split_cost["usage"][key] = split_cost["usage"].get(key, 0) + count
    study_cost["prior_attempts"] = []
    for prior in policy.get("prior_attempts", []):
        ledger_path = Path(prior["path"])
        if digest(ledger_path.read_bytes()) != prior["sha256"]:
            raise ValueError("Prior attempt cost ledger changed")
        ledger = read_json(ledger_path)
        summary = {key: ledger[key] for key in ("calls", "usage", "calls_with_usage")}
        summary["path"] = str(ledger_path)
        study_cost["prior_attempts"].append(summary)
        study_cost["calls"] += ledger["calls"]
        for key, count in ledger["usage"].items():
            study_cost["usage"][key] = study_cost["usage"].get(key, 0) + count
        study_cost["usage_complete"] &= ledger["calls_with_usage"] == ledger["calls"]
    if not study_cost["prior_attempts"]:
        study_cost["monetary_cost"] = sum_costs([read_json(p).get("monetary_cost") for p in sorted((study / "runs").glob("*/*/metrics.json"))])
    payload = {"split": split, "runs": runs, "paired": pairs, "module_effects": effects,
               "official_cli_checks": checks, "acceptance": acceptance, "actual_experiment_cost": actual_cost,
               "study_actual_cost": study_cost,
               "policy_sha256": digest((study / "policy.json").read_bytes()),
               "reservation_sha256": digest((study / "reservation.json").read_bytes())}
    target = study / "comparisons" / split
    write_json(target / "comparison.json", payload)
    percent = lambda v: "—" if v is None else f"{v:.2%}"
    lines = [f"# QASPER {split} 模块对照", "",
             "各组使用相同题目、模型、初始 Top-20 答案和上下文；动态组复用固定修复组在初始上下文中的修复和重审。审查组按支持检查拒答；修复组在已有上下文中修复后重新审查；动态组在修复后仍可规划补检索或补读邻段。最终支持审查使用独立调用。",
             "", "| 条件 | 完成/总数 | 官方 Answer F1 | 官方 Evidence F1 | 正面回答 | 支持回答数 | 支持率/正面回答 | 支持产出率/全部题目 | 拒答 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if five_arm:
        lines[2] = ("五组使用相同题目和模型。直接组只接收标题与问题；单轮组读取 BM25 Top-"
                    f"{policy['initial_k']['rag']}；重排、动态与完整组均从 BM25 Top-{policy['initial_k']['rerank']} 中重排保留 {policy['rerank_keep']} 段。"
                    "动态组由规划器自行决定查询改写、补读或停止；完整组增加在线证据审查、修复与审查失败后的拒答。各组独立推理，不跨组复用调用。最终支持诊断在推理结束后执行，不向动态组反馈。直接组未引用论文段落，论文证据支持诊断记为未通过。")
    for arm, r in runs.items():
        lines.append(f"| {LABELS[arm]} | {r['completed']}/{r['questions']} | {percent(r['official']['Answer F1'])} | {percent(r['official']['Evidence F1'])} | {r['positive_answers']} | {r['supported_positive_answers']} | {percent(r['support_pass_among_positive'])} | {percent(r['supported_answer_yield'])} | {r['abstentions']} |")
    lines.extend(["", "支持回答须为正面答案且最终审查通过；支持诊断由同模型完成，尚未经过独立人工核验。连续引文只允许空白规范化，数值、单位、否定及范围限定须被审查覆盖。",
                  "", "| 模块/配对比较 | Answer F1 差值（百分点） | 配对 95% 区间 | 拒答率差值（百分点） | 支持产出率差值（百分点） | 调用差值 | 输入 token 差值 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, effect in effects.items():
        pair = effect["answer_f1"]
        low, high = pair["bootstrap_95_percentile_interval"]
        lines.append(f"| {effect['module']}：{name} | {100*pair['mean_delta']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] | {100*effect['abstention']['mean_delta']:+.2f} | {100*effect['supported_answer']['mean_delta']:+.2f} | {effect['calls_delta']:+d} | {effect['usage_delta'].get('input_tokens', 0):+d} |")
    lines.extend(["", "按相同题目配对重采样 2000 次，固定随机种子；区间未覆盖模型重跑波动。拒答率与支持产出率的配对区间保存在 comparison.json。开发集结果用于选型，正式验收使用冻结后新预留的 holdout。",
                  "", "| 条件 | 全链路调用 | 本次实际调用 | 复用调用 | 输入 tokens | 输出 tokens | 失败调用 | 批次累计秒数 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm, r in runs.items():
        failures = sum(v["failed_calls"] for v in r["cost_by_stage"].values())
        lines.append(f"| {LABELS[arm]} | {r['calls']} | {r.get('actual_calls', r['calls'])} | {r.get('reused_calls', 0)} | {r['usage'].get('input_tokens', '未知')} | {r['usage'].get('output_tokens', '未知')} | {failures} | {r['batch_wall_seconds_total']:.1f} |")
    lines.extend(["", "| 条件/阶段 | 调用（含失败重试和复用） | 失败 | 输入 tokens | 输出 tokens |", "|---|---:|---:|---:|---:|"])
    for arm, r in runs.items():
        for stage, cost in r["cost_by_stage"].items():
            lines.append(f"| {LABELS[arm]}/{stage} | {cost['calls']} | {cost['failed_calls']} | {cost['usage'].get('input_tokens', '未知')} | {cost['usage'].get('output_tokens', '未知')} |")
    lines.extend(["", f"本次 {split} 实际调用共 {actual_cost['calls']} 次，用量：`{json.dumps(actual_usage)}`。各组全链路成本包含共享初始调用，实验实际成本对复用调用去重。回答、规划、修复、在线审查、修复审查、最终审查及失败尝试全部记录。货币费用尚无计价依据，记为未知。",
                  "用量记录完整性：" + ("完整。" if cost_complete else "部分调用缺少用量；已记录 token 数为下界。"),
                  "验收完整性：" + ("通过。" if integrity else "存在缺口。"),
                  "按预设区间标准，" + ("证据质量控制" if five_arm else "补检索") + "质量提升：" + ("已观察到。" if acceptance["quality_gain_demonstrated"] else "未证实。"),
                  f"上下文上限 {policy['max_context_characters']} 字符、{policy['max_paragraphs']} 段；回答最多 {policy['max_rounds']} 轮，修复每题最多 {policy.get('max_repair_rounds', 0)} 轮，每批最多 {policy.get('max_model_calls_per_batch', '未配置')} 次模型尝试，失败重试计入预算。",
                  "新题按本地已记录的论文、正文、标题及相同/高度近似问题排除；预训练或未记录曝光未知。官方 Answer/Evidence F1 使用 SHA-256 校验后的 QASPER v0.3 评分脚本，各组导出均经官方命令行复核。"])
    if policy.get("pricing"):
        lines = [line.replace("货币费用尚无计价依据，记为未知。", "货币费用按预设单价计算，详见补充指标与 comparison.json；缺失用量的费用记为未知。") for line in lines]
    lines.extend(extended_tables(runs, policy))
    if split == "holdout":
        frozen = read_json(study / "freeze.json")
        lines.extend(["", "开发集选定条件：" + LABELS.get(frozen.get("selected_arm"), "未记录") + "。",
                      f"本实验开发集与验收集累计实际调用 {study_cost['calls']} 次，累计用量：`{json.dumps(study_cost['usage'])}`。分集合及前期中断尝试明细见 comparison.json 的 study_actual_cost。"])
    if study_cost["prior_attempts"]:
        lines.append("前期中断尝试计入累计实际成本；缺失用量的调用以记录为准，累计 token 数为已记录下界。")
    atomic_text(target / "report.md", "\n".join(lines) + "\n")
    print(target / "report.md")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    args = parser.parse_args()
    report(args.study.resolve(), args.split)


if __name__ == "__main__":
    main()
