"""Deterministic delivery policy over Verifier results and task/resource budgets."""
from __future__ import annotations

from .pipeline import cst_now
from .research_state import fingerprint
from .paper_explanations import content_gaps, empty_explanation


def review_key(request):
    return "artifact:" + request["artifacts"][0] if request["phase"] == "artifact" else request["phase"]


def evaluate_gate(state, request, report, budget):
    phase = request.get("phase", "terminal")
    issues = []
    for row in report.get("items", []):
        if row["passed"]:
            continue
        reasons = [claim["reason"] for claim in row["claims"] if not claim["passed"]]
        reasons += [failure for claim in row["claims"] for failure in claim.get("structural_failures", [])]
        if not row.get("complete"):
            reasons.append("审查未覆盖断言的全部实质文字")
        structural = row["structural_error"] or any(claim.get("structural_failures") for claim in row["claims"])
        issues.append({"item_id": row["item_id"], "section": row["section"], "field": row["field"],
                       "paper_id": row.get("paper_id", ""), "evidence_ids": row["evidence_ids"],
                       "reason": row["structural_error"] or "；".join(reasons) or "；".join(row["missing_information"]) or "语义支持检查未通过",
                       "missing_information": row["missing_information"], "action": "repair" if structural else row["recommended_action"],
                       "query": row["query"], "required_fields": row["required_fields"]})
    if phase != "terminal" and report.get("status") != "checked":
        issues.append({"section": phase, "reason": report.get("error", "Verifier 未完成"), "action": "repair"})

    desired = request.get("desired_status", "complete")
    explains_papers = "analysis" in state.get("artifacts", {}) or bool(state.get("paper_explanations"))
    if explains_papers and report.get("status") == "checked" and (phase == "evidence" or (phase == "final" and desired == "complete")):
        ids = request.get("paper_ids") or list(state.get("analyses", {}))
        for pid in ids:
            if pid not in state.get("analyses", {}):
                continue
            explanation = state.get("paper_explanations", {}).get(pid, empty_explanation(pid))
            for gap in content_gaps(explanation):
                issues.append({"section": "explanation", "paper_id": pid, "field": gap["field"],
                               "reason": gap["reason"], "action": "read_more", "query": gap["query"]})
    if phase == "final" and desired == "complete":
        for name in state["deliverables"]:
            if name not in state["artifacts"]:
                issues.append({"section": name, "reason": "尚未生成所需产物", "action": "retrieve_more"})
        for row in state["artifacts"].get("experiment", []):
            if row["feasibility"] == "blocked":
                issues.append({"section": "experiment", "reason": "实验估计资源需求超出已知预算，需要重新设计",
                               "missing_information": row["missing_resources"], "action": "repair"})
            elif row["feasibility"] != "candidate" or row["missing_resources"]:
                issues.append({"section": "experiment", "reason": "实验资源条件待补",
                               "missing_information": row["missing_resources"], "action": "needs_input"})
        deployment = state["artifacts"].get("deployment", {})
        if deployment.get("missing_information"):
            issues.append({"section": "deployment", "reason": "部署条件待补",
                           "missing_information": deployment["missing_information"], "action": "needs_input"})
        for row in deployment.get("assessments", []):
            if row["assessment"] == "needs_info" or row["missing_information"]:
                issues.append({"section": "deployment", "paper_id": row["paper_id"], "reason": "部署评估尚有未确认条件",
                               "missing_information": row["missing_information"], "action": "needs_input"})

    exhausted = (budget["model_calls"] >= budget["max_model_calls"] or budget["charged_tokens"] >= budget["token_budget"]
                 or len(state["steps"]) >= state["limits"]["max_steps"])
    key = review_key(request) if phase != "terminal" else phase
    failures = sum(row["review_key"] == key and row["decision"] not in {"pass", "partial"}
                   for row in state.get("gate_history", []))
    if phase == "terminal" or report.get("status") != "checked":
        decision = "partial"
        reason = state.get("stop_reason") or report.get("error") or "审查未完成，交付已确认的部分"
    elif phase == "final" and desired == "partial":
        decision, reason = "partial", request["reason"]
    elif not issues:
        decision, reason = "pass", "当前断言通过程序校验与独立语义审查"
    elif exhausted or failures + 1 >= state["limits"]["max_review_rounds"]:
        decision, reason = "partial", "质量问题在剩余预算或审查轮数内未能解决"
    else:
        actions = {issue["action"] for issue in issues}
        decision = next((action for action in ("repair", "needs_input", "read_more", "retrieve_more") if action in actions), "read_more")
        reason = "质量审查发现具体缺口，返回 Planner 处理"

    suggestions = []
    for issue in issues:
        action = issue["action"]
        tool = {"repair": "repair_result", "needs_input": "ask_user", "read_more": "read_papers", "retrieve_more": "search_papers"}.get(action, "read_papers")
        target = issue["section"]
        suggestions.append({"tool": tool, "paper_ids": [issue["paper_id"]] if issue.get("paper_id") else [],
                            "query": ("analysis" if target in {"evidence", "explanation"} else target) if tool == "repair_result" else issue.get("query", ""),
                            "required_fields": issue.get("required_fields", []), "reason": issue["reason"]})
    return {"gate_id": fingerprint({"review": report.get("input_hash"), "phase": phase, "issues": issues,
                                    "round": len(state.get("gate_history", []))})[:24],
            "review_key": key, "phase": phase, "decision": decision, "reason": reason,
            "issues": issues, "suggestions": suggestions, "at": cst_now()}
