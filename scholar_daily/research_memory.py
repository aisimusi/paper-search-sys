"""Loss-aware, deterministic task memory; authoritative state and text remain on disk."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .paper_context import TOKEN_ESTIMATOR, estimated_tokens


def compact_view(state, view):
    result = copy.deepcopy(view)
    result["schema_version"] = 1
    result["task_id"] = state["task_id"]
    result["status"] = state["status"]
    result["pending"] = copy.deepcopy(state["pending"])
    result["next_action"] = copy.deepcopy(state["next_action"])
    result["unfinished_tasks"] = {
        "deliverables": [name for name in state["deliverables"] if name not in state["artifacts"]],
        "review_request": copy.deepcopy(state.get("review_request", {})),
        "failed_steps": [{k: row[k] for k in ("tool", "action_key", "error") if k in row}
                         for row in state["steps"] if row["status"] == "failed"],
    }
    labels = {}
    # Verification remains a model assessment, never a ground-truth label.
    from .research_verifier import approved_ids, evidence_signature
    allowed = approved_ids(state)
    report = state.get("verification", {}).get("evidence", {})
    current = report.get("status") == "checked" and report.get("signature") == evidence_signature(state)
    groups = {}
    for row in report.get("items", []):
        if row.get("section", "evidence") != "evidence":
            continue
        for eid in row["evidence_ids"]:
            groups.setdefault(eid, []).append(row.get("support_status", "supported" if row["passed"] else "insufficient"))
    for eid, decisions in groups.items():
        labels[eid] = ("stale" if not current else "contradicted" if "contradicted" in decisions else
                       "supported" if eid in allowed else "insufficient")
    result["evidence"] = [{"evidence_id": eid, "paper_id": row["paper_id"], "kind": row["kind"],
        "text": row["text"], "model_support_status": labels.get(eid, "unreviewed"),
        "review_current_and_passed": eid in allowed,
        "locator": {"index_path": row.get("index_path", ""), "location_count": len(row.get("locations", [])),
                    "locations": row.get("locations", [])[:1]}}
        for eid, row in state["evidence"].items()]
    for paper in result["papers"]:
        pid = paper["paper_id"]
        paper["methods"] = [{"evidence_id": c["evidence_id"], "text": c["text"]}
                            for c in state["analyses"].get(pid, {}).get("claims", []) if c["kind"] == "method"]
        ctx = state.get("paper_contexts", {}).get(pid, {})
        paper["context_index"] = {k: ctx[k] for k in ("index_path", "source_fingerprint", "sections", "selection") if k in ctx}
    from .mcp_client import result_preview
    result["external_tools"] = {name: {"catalog_fingerprint": row["catalog_fingerprint"], "tools": row["tools"]}
                                for name, row in state.get("external_tools", {}).items()}
    result["external_results"] = {key: {field: row[field] for field in ("server", "tool", "arguments", "result_path", "result_fingerprint", "external_data")}
                                  for key, row in state.get("external_results", {}).items()}
    for key, row in list(state.get("external_results", {}).items())[-4:]:
        result["external_results"][key]["preview"] = result_preview(row["result"], 2500) if "result" in row else {
            "text_preview": row.get("text_preview", "")[:2500], "truncated": True}
    result["recent_steps"] = [{k: row[k] for k in ("tool", "status", "reason", "action_key", "error") if k in row}
                              for row in state["steps"][-6:]]
    return result


def full_view(state, view):
    result = copy.deepcopy(view)
    result.update({key: copy.deepcopy(state[key]) for key in
                   ("papers", "analyses", "evidence", "artifacts", "steps", "review_history", "gate_history", "pending", "next_action")})
    result["paper_explanations"] = copy.deepcopy(state.get("paper_explanations", {}))
    return result


def bounded_view(state, view):
    policy = state["context_policy"]
    if policy["planner_mode"] == "full":
        return full_view(state, view)
    result = compact_view(state, view)
    target = policy["max_context_tokens"]
    # Keep all IDs, pointers, exact facts, active gaps, required outputs and pending tasks.
    # When the minimum safe memory is large, report overflow rather than silently losing facts.
    if estimated_tokens(json.dumps(result, ensure_ascii=False)) > target:
        for row in result["evidence"]:
            if row["kind"] != "method":
                row.pop("text", None)
        for paper in result["papers"]:
            paper["context_index"].pop("sections", None)
    if estimated_tokens(json.dumps(result, ensure_ascii=False)) > target:
        for paper in result["papers"]:
            paper["methods"] = [{"evidence_id": row["evidence_id"], "state_field": "analyses.%s.claims" % paper["paper_id"]}
                                for row in paper["methods"]]
        for row in result["evidence"]:
            row.pop("text", None)
    result["memory_budget"] = {"target_estimated_tokens": target, "overflow": False,
                              "token_estimator": TOKEN_ESTIMATOR}
    result["memory_budget"]["overflow"] = estimated_tokens(json.dumps(result, ensure_ascii=False)) > target
    return result


def memory_snapshot(state, state_path: Path):
    from .research_state import fingerprint, planner_view
    baseline = full_view(state, planner_view(state))
    compact_state = dict(state, context_policy=dict(state["context_policy"], planner_mode="compact"))
    memory = bounded_view(compact_state, planner_view(state))
    before = estimated_tokens(json.dumps(baseline, ensure_ascii=False))
    after = estimated_tokens(json.dumps(memory, ensure_ascii=False))
    return {"schema_version": 1, "task_id": state["task_id"], "state_path": str(state_path.resolve()),
            "state_sha256": fingerprint(state), "memory": memory,
            "compression": {"before_estimated_tokens": before, "after_estimated_tokens": after,
                            "reduction_fraction": 1 - after / before if before else 0,
                            "token_estimator": TOKEN_ESTIMATOR,
                            "scope": "serialized task state views; excludes prompt instructions, provider overhead and output",
                            "quality_comparison": "requires paired task runs and independent human ratings"},
            "resume": "Read pending/unfinished_tasks; use research-resume with the original task directory. Load state/source snapshots by recorded paths when needed."}
