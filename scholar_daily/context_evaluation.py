"""Paired task comparison for full versus compact Planner context, with human quality ratings."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .pipeline import atomic_text
from .research_calls import read_json, write_json
from .research_state import fingerprint
from .research_verifier import approved_ids, delivery_view

CRITERIA = {"required_outputs": "所需产物完整；无法完成的部分有准确且具体的解释",
            "factual_support": "事实断言得到原文支持，推断和未知条件明确",
            "qualifier_fidelity": "对象、数字、单位、协议、适用范围和不确定性准确",
            "traceability": "主要结论能回到论文版本、证据标识和原文位置"}


def inspect_task(folder, mode):
    folder = Path(folder).resolve()
    meta = read_json(folder / "task.json")
    state = read_json(folder / "state.json")
    initial = meta["initial_state"]
    if not initial["papers"]:
        raise ValueError("配对上下文评测需要固定论文输入；请先冻结联网检索得到的材料")
    if initial["context_policy"]["planner_mode"] != mode:
        raise ValueError("运行的 planner_mode 与比较组不符：" + str(folder))
    if state["status"] not in {"complete", "partial"} or state["pending"]:
        raise ValueError("质量比较需要已结束的任务，不能把等待中的任务当作失败")
    if state["task_id"] != meta["task_id"]:
        raise ValueError("任务快照 ID 不一致")
    memory = read_json(folder / "memory.json")
    if memory["state_sha256"] != fingerprint(state):
        raise ValueError("状态快照与压缩记录指纹不一致")
    policy = {k: v for k, v in initial["context_policy"].items() if k != "planner_mode"}
    condition = {k: initial[k] for k in ("goal", "facts", "papers", "limits", "config")}
    condition.update(context_policy=policy, implementation=meta["implementation"],
                     required_outputs=sorted(state["deliverables"]),
                     loaded_skills=initial["loaded_skills"], final_facts=state["facts"])
    task_key = fingerprint(condition)
    safe = delivery_view(state)
    delivered = [name for name in state["deliverables"] if name in safe["artifacts"]]
    evidence = approved_ids(state)
    return task_key, {"task_dir": str(folder), "task_id": state["task_id"], "goal": state["goal"],
                      "status": state["status"], "requested_outputs": state["deliverables"],
                      "delivered_outputs": delivered,
                      "completion_fraction": len(delivered) / len(state["deliverables"]) if state["deliverables"] else 0,
                      "model_review_approved_evidence": len(evidence), "evidence_records": len(state["evidence"]),
                      "context_metrics": state["context_metrics"],
                      "report_path": str(folder / "report.md"), "state_path": str(folder / "state.json"),
                      "source_fingerprints": {pid: record["source_fingerprint"] for pid, record in state.get("paper_contexts", {}).items()}}


def _interval(deltas, seed):
    if len(deltas) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
    return [samples[49], samples[1949]]


def compare_tasks(full, compact, output, *, ratings=None, seed=20261014):
    full, compact = list(full), list(compact)
    output = Path(output).resolve()
    if not full or len(full) != len(compact):
        raise ValueError("两组必须包含数量相等且非空的配对任务")
    if any(output == Path(p).resolve() or Path(p).resolve() in output.parents for p in full + compact):
        raise ValueError("比较输出必须位于原任务目录外")
    arms = {}
    for mode, folders in (("full", full), ("compact", compact)):
        arms[mode] = {}
        for folder in folders:
            key, row = inspect_task(folder, mode)
            if key in arms[mode]:
                raise ValueError("同一组重复了相同任务条件；重复实验应分别进行配对比较")
            arms[mode][key] = row
    if set(arms["full"]) != set(arms["compact"]):
        raise ValueError("配对任务的目标、材料、配置、预算、代码或所需产物不一致")
    template = {"schema_version": 1, "rubric": CRITERIA, "rating": "每项填 0 或 1；未核验填 null。质量分数是四项均值。",
                "items": []}
    pairs, case_modes = [], {}
    for key in sorted(arms["full"]):
        pair = {"task_key": key, **{mode: arms[mode][key] for mode in ("full", "compact")}}
        source_a, source_b = pair["full"]["source_fingerprints"], pair["compact"]["source_fingerprints"]
        if any(source_a[pid] != source_b[pid] for pid in source_a.keys() & source_b.keys()):
            raise ValueError("配对任务读取的同一论文原文快照不一致")
        cases = []
        for mode in ("full", "compact"):
            row = pair[mode]
            case_id = fingerprint({"task_key": key, "task_id": row["task_id"]})[:24]
            case_modes[case_id] = (key, mode)
            cases.append({"case_id": case_id, "task_key": key, "goal": row["goal"],
                          "report_path": row["report_path"], "state_path": row["state_path"],
                          "criteria": dict.fromkeys(CRITERIA), "reviewer": "", "notes": ""})
        random.Random(seed + int(key[:8], 16)).shuffle(cases)
        template["items"].extend(cases)
        pairs.append(pair)
    labelled = {}
    if ratings is not None:
        submitted = read_json(ratings)
        if submitted.get("rubric") != CRITERIA or not isinstance(submitted.get("items"), list):
            raise ValueError("人工质量评分必须使用同一 rubric 与 items 模板")
        if any(not isinstance(row, dict) or not isinstance(row.get("case_id"), str) for row in submitted["items"]):
            raise ValueError("人工质量评分中的每项必须包含字符串 case_id")
        by_id = {row["case_id"]: row for row in submitted["items"]}
        if len(by_id) != len(submitted["items"]) or set(by_id) != set(case_modes):
            raise ValueError("人工质量评分 case_id 不能增加、遗漏或重复")
        for base in template["items"]:
            row = by_id[base["case_id"]]
            for field in ("case_id", "task_key", "goal", "report_path", "state_path"):
                if row.get(field) != base[field]:
                    raise ValueError("不能修改人工评分任务身份或来源")
            criteria = row.get("criteria")
            if not isinstance(criteria, dict) or set(criteria) != set(CRITERIA) or any(
                    v is not None and (type(v) is not int or v not in (0, 1)) for v in criteria.values()):
                raise ValueError("每项人工质量评分必须为 0、1 或 null")
            if not isinstance(row.get("reviewer"), str) or not isinstance(row.get("notes"), str):
                raise ValueError("人工评分需要 reviewer/notes 字符串")
            if all(v is not None for v in criteria.values()):
                if not row["reviewer"].strip():
                    raise ValueError("完整人工评分需要 reviewer")
                labelled[base["case_id"]] = {"score": sum(criteria.values()) / len(CRITERIA),
                                             "reviewer": row["reviewer"], "notes": row["notes"], "criteria": criteria}
    deltas, token_deltas, completion_deltas = [], [], []
    measured_pairs = 0
    for pair in pairs:
        key = pair["task_key"]
        quality = {}
        for case_id, (task_key, mode) in case_modes.items():
            if task_key == key and case_id in labelled:
                quality[mode] = labelled[case_id]
        pair["human_quality"] = quality
        if len(quality) == 2:
            pair["human_quality_delta"] = quality["compact"]["score"] - quality["full"]["score"]
            deltas.append(pair["human_quality_delta"])
        else:
            pair["human_quality_delta"] = None
        a, b = pair["full"]["context_metrics"], pair["compact"]["context_metrics"]
        measured = a["unmeasured_attempts"] == b["unmeasured_attempts"] == 0 and a["attempts"] > 0 and b["attempts"] > 0
        measured_pairs += measured
        metric = "measured_input_tokens" if measured else "estimated_input_tokens"
        if not measured and (a["estimate_missing_attempts"] or b["estimate_missing_attempts"]):
            raise ValueError("配对任务缺少完整本地上下文估计，不能计算 Token 差值")
        pair["input_token_comparison"] = {"basis": metric, "full": a[metric], "compact": b[metric],
                                           "delta": b[metric] - a[metric]}
        token_deltas.append(b[metric] - a[metric])
        completion_deltas.append(pair["compact"]["completion_fraction"] - pair["full"]["completion_fraction"])
    payload = {"schema_version": 1, "pairs": pairs, "summary": {
        "paired_tasks": len(pairs), "pairs_with_complete_measured_input": measured_pairs,
        "mean_completion_fraction_delta": sum(completion_deltas) / len(pairs),
        "input_token_deltas": token_deltas,
        "human_quality_pairs": len(deltas), "human_quality_status": "checked" if len(deltas) == len(pairs) else "awaiting_human_ratings",
        "mean_human_quality_delta": sum(deltas) / len(deltas) if deltas else None,
        "human_quality_bootstrap_95_interval": _interval(deltas, seed), "seed": seed},
        "limits": "Model gate/completion counts are process diagnostics. Human scores reflect the fixed rubric. Bootstrap resamples paired tasks and does not estimate model rerun variation."}
    if ratings is not None:
        write_json(output / "quality-ratings.json", submitted)
    write_json(output / "comparison.json", payload)
    write_json(output / "quality-template.json", template)
    lines = ["# 上下文压缩配对比较", "", "| 任务 | 完成比例 full/compact | 输入 Token full/compact | 计数依据 | 人工质量差值 |",
             "|---|---|---|---|---|"]
    for pair in pairs:
        u = pair["input_token_comparison"]
        lines.append("| %s | %.2f / %.2f | %d / %d | %s | %s |" % (
            pair["task_key"][:12], pair["full"]["completion_fraction"], pair["compact"]["completion_fraction"],
            u["full"], u["compact"], u["basis"], pair["human_quality_delta"] if pair["human_quality_delta"] is not None else "待人工评分"))
    lines += ["", "质量分数使用 quality-template.json 中的固定四项人工核验标准；模型通过率只作为流程诊断。",
              "输入用量缺失时采用本地估计；估计不代表供应商实际计费。差值均为 compact − full。", "",
              json.dumps(payload["summary"], ensure_ascii=False, indent=2), "", payload["limits"], ""]
    atomic_text(output / "comparison.md", "\n".join(lines))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, nargs="+", required=True)
    parser.add_argument("--compact", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratings", type=Path)
    args = parser.parse_args(argv)
    compare_tasks(args.full, args.compact, args.output, ratings=args.ratings)
    print(args.output / "comparison.md")


if __name__ == "__main__":
    main()
