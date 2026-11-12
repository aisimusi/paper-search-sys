"""Reproducible stratified human spot checks of provisional model support labels."""
from __future__ import annotations

import copy
import random
from pathlib import Path

from ._compat import FileLock
from .research_calls import read_json, write_json
from .research_state import fingerprint
from .research_verifier import evidence_signature, report_signature, review_inputs

LABELS = ("supported", "contradicted", "insufficient")


def audit_sample(state):
    policy = state["context_policy"]
    candidates = []
    for key, report in sorted(state.get("verification", {}).items()):
        if report.get("status") != "checked":
            continue
        request = {"phase": "evidence"} if key == "evidence" else (
            {"phase": "final"} if key == "final" else {"phase": "artifact", "artifacts": [key.split(":", 1)[1]]})
        signature = evidence_signature(state) if key == "evidence" else report_signature(state, request)
        if report.get("signature") != signature:
            continue
        payload = review_inputs(state, request)
        for row in report["items"]:
            if row["kind"] != "fact":
                continue
            predicted = row.get("support_status", "supported" if row["passed"] else "insufficient")
            identity = {"review_key": key, "input_hash": report["input_hash"], "item_id": row["item_id"]}
            candidates.append({"case_id": fingerprint(identity)[:24], **identity,
                "field": row["field"], "paper_id": row["paper_id"], "claim": row["text"],
                "evidence_ids": row["evidence_ids"], "model_label": predicted,
                "model_review": copy.deepcopy(row["claims"]), "structural_error": row["structural_error"],
                "sources": {sid: payload["sources"][sid] for sid in row["source_ids"] if sid in payload["sources"]},
                "human_label": None, "reviewer": "", "notes": ""})
    rng = random.Random(policy["audit_seed"])
    selected = []
    for label in LABELS:
        group = [c for c in candidates if c["model_label"] == label]
        if group and len(selected) < policy["audit_sample_size"]:
            selected.append(rng.choice(group))
    remaining = [c for c in candidates if c not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, policy["audit_sample_size"] - len(selected))])
    selected.sort(key=lambda c: c["case_id"])
    identity = {"task_id": state["task_id"], "seed": policy["audit_seed"],
                "sample_size": policy["audit_sample_size"], "cases": [c["case_id"] for c in selected]}
    sample_id = fingerprint({"identity": identity, "items": selected})
    return {"schema_version": 1, "sample_id": sample_id, **identity,
            "sampling": "seeded stratified sample; one per available model label, then random fill",
            "population": len(candidates),
            "population_by_model_label": {label: sum(c["model_label"] == label for c in candidates) for label in LABELS},
            "labels": list(LABELS), "items": selected,
            "instruction": "逐条核对断言、限定条件和提供的原文；仅填写 human_label、reviewer、notes。模型标签属于待核验判断。"}


def audit_metrics(sample):
    labelled = [row for row in sample["items"] if row["human_label"] is not None]
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    disagreements = []
    for row in labelled:
        confusion[row["model_label"]][row["human_label"]] += 1
        if row["model_label"] != row["human_label"]:
            disagreements.append({k: row[k] for k in ("case_id", "field", "model_label", "human_label", "notes")})
    return {"sample_id": sample["sample_id"], "sampled": len(sample["items"]), "labelled": len(labelled),
            "agreement_fraction": 1 - len(disagreements) / len(labelled) if labelled else None,
            "confusion_matrix_model_rows_human_columns": confusion, "disagreements": disagreements,
            "status": ("not_applicable" if not sample["items"] else
                       "checked" if len(labelled) == len(sample["items"]) else "awaiting_human_review"),
            "scope": "sample agreement only; stratified selection is not an unbiased population accuracy estimate",
            "effect": "annotations are audit records; model gate decisions are not relabelled as human ground truth"}


def validate_sample_labels(sample, submitted):
    if not isinstance(submitted, dict) or submitted.get("sample_id") != sample["sample_id"]:
        raise ValueError("人工标签必须对应当前抽样 sample_id，不能复用其他版本标签")
    items = submitted.get("items")
    if not isinstance(items, list):
        raise ValueError("人工核验 items 必须为数组")
    if any(not isinstance(row, dict) or not isinstance(row.get("case_id"), str) for row in items):
        raise ValueError("人工核验中的每项必须包含字符串 case_id")
    by_id = {row["case_id"]: row for row in items}
    if len(by_id) != len(items) or set(by_id) != {row["case_id"] for row in sample["items"]}:
        raise ValueError("人工核验必须覆盖全部抽样 case_id，不能增加、遗漏或重复")
    result = copy.deepcopy(sample)
    for row in result["items"]:
        supplied = by_id[row["case_id"]]
        for key, value in row.items():
            if key not in {"human_label", "reviewer", "notes"} and supplied.get(key) != value:
                raise ValueError("不能修改人工抽样的原文、断言或模型结果：" + key)
        label = supplied.get("human_label")
        if label is not None and label not in LABELS:
            raise ValueError("human_label 必须为 supported/contradicted/insufficient 或 null")
        if not isinstance(supplied.get("reviewer"), str) or not isinstance(supplied.get("notes"), str):
            raise ValueError("reviewer/notes 必须为字符串")
        if label is not None and not supplied["reviewer"].strip():
            raise ValueError("已填写人工标签时必须记录 reviewer")
        row.update({key: supplied[key] for key in ("human_label", "reviewer", "notes")})
    return result


def annotate_sample(folder: Path, labels_path: Path):
    folder = folder.resolve()
    submitted = read_json(labels_path)
    with FileLock(folder / ".lock"):
        current = read_json(folder / "audit-sample.json")
        sample_id = current.get("sample_id", "")
        if not isinstance(sample_id, str) or len(sample_id) != 64 or any(c not in "0123456789abcdef" for c in sample_id):
            raise ValueError("抽样 sample_id 无效")
        sample = read_json(folder / "human-audit" / sample_id / "sample.json")
        if sample["sample_id"] != sample_id:
            raise ValueError("冻结抽样与当前抽样标识不一致")
        result = validate_sample_labels(sample, submitted)
        target = folder / "human-audit" / sample["sample_id"]
        write_json(target / "labels.json", result)
        metrics = audit_metrics(result)
        write_json(target / "metrics.json", metrics)
        return dict(metrics, labels_path=str(target / "labels.json"), metrics_path=str(target / "metrics.json"))
