"""LangGraph planner/tool loop with local durable checkpoints and file tasks."""
from __future__ import annotations

import copy
import importlib.metadata
import subprocess
import uuid
from pathlib import Path

from ._compat import FileLock
from .agents import TaskPending, get_backend
from .config import deep_merge, resolve_path
from .pipeline import atomic_text, cst_now
from .report import EvidenceLinks, cell, render_evidence, render_idea, render_paper
from .paper_explanations import content_gaps, empty_explanation
from .models import Paper
from .research_calls import BudgetExceeded, CallLedger, read_json, saved_budget, saved_context_metrics, write_json
from .research_state import PLANNER_SCHEMA, ResearchState, checked_limits, fingerprint, initial_state, planner_prompt
from .research_tools import ResearchTools
from .research_verifier import Verifier, delivery_view, evidence_signature
from .quality_gate import evaluate_gate, review_key


def graph_api():
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command, interrupt
    except ImportError as exc:
        raise RuntimeError('请在项目 Conda 环境安装编排依赖：python -m pip install -e ".[orchestrator]"') from exc
    return StateGraph, START, END, SqliteSaver, Command, interrupt


def implementation_stamp():
    files = ("orchestrator.py", "research_calls.py", "research_state.py", "research_tools.py",
             "research_verifier.py", "quality_gate.py", "skill_registry.py",
             "paper_context.py", "research_memory.py", "research_audit.py",
             "mcp_client.py", "zotero_mcp.py",
             "paper_explanations.py", "report.py", "agents.py", "contracts.py", "prompts.py", "sources.py", "fulltext.py", "ranking.py", "config.py", "models.py", "_compat.py")
    from .skill_registry import catalog
    dependencies = {name: importlib.metadata.version(name) for name in ("langgraph", "langgraph-checkpoint-sqlite")}
    try:
        dependencies["mcp"] = importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:
        dependencies["mcp"] = "not_installed"
    return {"source": {name: fingerprint((Path(__file__).parent / name).read_text(encoding="utf-8")) for name in files},
            "skills": fingerprint(catalog()),
            "dependencies": dependencies}


class ResearchOrchestrator:
    def __init__(self, ledger: CallLedger, search=None, reader=None):
        self.ledger = ledger
        self.tools = ResearchTools(ledger, search=search, reader=reader)
        self.reviewer = Verifier(ledger)

    def stop(self, reason):
        return {"status": "partial", "stop_reason": reason, "pending": {}, "budget": self.ledger.budget(),
                "review_request": {"phase": "terminal", "reason": reason, "desired_status": "partial"}}

    def planner(self, state: ResearchState):
        if len(state["steps"]) >= state["limits"]["max_steps"]:
            return self.stop("工具步骤已达上限")
        budget = self.ledger.budget()
        if budget["charged_tokens"] >= budget["token_budget"]:
            return self.stop("Token 预算已达上限")
        view = dict(state, budget=budget, context_metrics=self.ledger.context_metrics())
        def check(value):
            if not value["plan"] or not value["deliverables"]:
                raise ValueError("Planner 必须给出执行计划与所需产物")
            if len(value["deliverables"]) != len(set(value["deliverables"])):
                raise ValueError("deliverables 不能重复")
            if state["deliverables"] and set(value["deliverables"]) != set(state["deliverables"]):
                raise ValueError("重规划不能静默缩减用户所需产物")
        try:
            value = self.ledger.call("planner-%d" % state["planner_round"], "planner",
                                     planner_prompt(view, self.tools.catalog()), PLANNER_SCHEMA, check)
            return {"plan": value["plan"], "deliverables": value["deliverables"],
                    "next_action": value["next_action"], "planner_round": state["planner_round"] + 1,
                    "status": "running", "pending": {}, "budget": self.ledger.budget()}
        except TaskPending as exc:
            return {"status": "waiting", "pending": {"kind": "session", "resume_node": "planner", "task": exc.task},
                    "budget": self.ledger.budget()}
        except BudgetExceeded as exc:
            return self.stop(str(exc))
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            return {"steps": state["steps"] + [{"tool": "planner", "status": "failed", "error": str(exc), "at": cst_now()}],
                    "next_action": {}, "pending": {}, "budget": self.ledger.budget()}

    def execute(self, state: ResearchState):
        action = state["next_action"]
        tool = action["tool"]
        entry = {"tool": tool, "reason": action["reason"], "arguments": action, "at": cst_now()}
        key = ""
        try:
            if tool == "finish":
                missing = [name for name in state["deliverables"] if name not in state["artifacts"]]
                if action["finish_status"] == "complete" and missing:
                    raise ValueError("尚未生成所需产物：" + ", ".join(missing))
                entry["status"] = "reviewing"
                return {"status": "reviewing", "pending": {}, "steps": state["steps"] + [entry],
                        "review_request": {"phase": "final", "desired_status": action["finish_status"],
                                           "reason": action["reason"], "continuation": "finalize"},
                        "budget": self.ledger.budget()}
            if tool == "ask_user":
                if not action["question"].strip() or not action["required_fields"]:
                    raise ValueError("ask_user 需要明确问题及所需事实字段")
                entry["status"] = "waiting"
                issue = {"kind": "needs_info", "question": action["question"], "fields": action["required_fields"]}
                return {"status": "waiting_input", "pending": {**issue, "kind": "user", "resume_node": "planner"},
                        "steps": state["steps"] + [entry], "issues": state["issues"] + [issue], "budget": self.ledger.budget()}
            cap = self.tools.registry[tool]
            research = tool in {"compare_methods", "assess_deployment", "design_experiment"} or (tool == "repair_result" and action["query"] in {"deployment", "experiment"})
            ids = action["paper_ids"] or list(state["papers"])
            if research and ids and all(pid in state["analyses"] for pid in ids):
                reviewed = state.get("verification", {}).get("evidence", {})
                if reviewed.get("signature") != evidence_signature(state):
                    return {"status": "reviewing", "pending": {}, "review_request": {
                        "phase": "evidence", "continuation": "execute", "paper_ids": ids, "reason": "Research 执行前审查当前证据"}}
            cap.guard(state, action)
            key = self.tools.key(state, action)
            if any(step.get("action_key") == key and step["status"] == "completed" for step in state["steps"]):
                raise ValueError("该操作已完成且输入未变化，请选择补检索、补读或结束")
            patch = self.tools.execute(state, action)
            artifacts = dict(state["artifacts"])
            for name in patch.pop("invalidate_artifacts", []):
                artifacts.pop(name, None)
            if tool == "analyze_papers":
                for name in ("comparison", "deployment", "experiment"):
                    artifacts.pop(name, None)
            elif tool == "compare_methods":
                artifacts.pop("deployment", None)
                artifacts.pop("experiment", None)
            elif tool == "assess_deployment":
                artifacts.pop("experiment", None)
            artifacts.update(patch.pop("artifacts", {}))
            entry.update(status="completed", action_key=key, role=cap.role)
            issues = [issue for issue in state["issues"] if issue.get("action_key") != key]
            request = {}
            if tool == "analyze_papers" or (tool == "repair_result" and action["query"] in {"analysis", "comparison"}):
                request = {"phase": "evidence", "continuation": "planner", "paper_ids": ids, "reason": "Reader 断言的独立证据审查"}
            section = {"compare_methods": "comparison", "assess_deployment": "deployment", "design_experiment": "experiment"}.get(tool)
            if tool == "repair_result" and action["query"] in {"deployment", "experiment"}:
                section = action["query"]
            if section:
                request = {"phase": "artifact", "artifacts": [section], "continuation": "planner", "reason": "Research 产物的独立审查"}
            return dict(patch, artifacts=artifacts, issues=issues, steps=state["steps"] + [entry],
                        review_request=request, status="reviewing" if request else "running", pending={}, budget=self.ledger.budget())
        except TaskPending as exc:
            self.ledger.finish_operation(key, "waiting")
            return {"status": "waiting", "pending": {"kind": "session", "resume_node": "execute", "task": exc.task},
                    "budget": self.ledger.budget()}
        except BudgetExceeded as exc:
            return dict(self.stop(str(exc)), steps=state["steps"] + [dict(entry, status="failed", error=str(exc))])
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            if key:
                self.ledger.finish_operation(key, "failed", str(exc))
            issue = {"kind": "tool_error", "action_key": key, "tool": tool, "message": str(exc)}
            return {"steps": state["steps"] + [dict(entry, status="failed", action_key=key, error=str(exc))],
                    "issues": state["issues"] + [issue], "pending": {}, "status": "running", "budget": self.ledger.budget()}

    def verifier(self, state: ResearchState):
        request = state["review_request"]
        try:
            report = self.reviewer.review(state, request)
        except TaskPending as exc:
            return {"status": "waiting", "pending": {"kind": "session", "resume_node": "verifier", "task": exc.task},
                    "budget": self.ledger.budget()}
        except BudgetExceeded as exc:
            report = {"phase": request["phase"], "status": "budget_exhausted", "items": [], "error": str(exc), "at": cst_now()}
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            attempts = request.get("attempts", 0) + 1
            request = dict(request, attempts=attempts)
            report = {"phase": request["phase"], "status": "retry" if attempts < state["limits"]["max_attempts"] else "unavailable",
                      "items": [], "error": str(exc), "at": cst_now()}
        records = dict(state["verification"], **{review_key(request): report})
        history = state["review_history"] + [dict(report, review_key=review_key(request))]
        return {"verification": records, "review_history": history, "review_request": request,
                "status": "reviewing", "pending": {}, "budget": self.ledger.budget()}

    def quality_gate(self, state: ResearchState):
        request = state["review_request"]
        report = state["verification"].get(review_key(request), {}) if request["phase"] != "terminal" else {}
        budget = self.ledger.budget()
        gate = evaluate_gate(state, request, report, budget)
        final = request["phase"] in {"final", "terminal"}
        route = request.get("continuation", "planner") if gate["decision"] == "pass" else "planner"
        status = "running"
        stop_reason = ""
        if gate["decision"] == "partial" or (gate["decision"] == "pass" and final):
            route = "finalize"
            status = "partial" if gate["decision"] == "partial" else "complete"
            stop_reason = request.get("reason", "") if status == "complete" else gate["reason"]
        gate["route"] = route
        steps = copy.deepcopy(state["steps"])
        if final and steps and steps[-1]["tool"] == "finish" and steps[-1]["status"] == "reviewing":
            steps[-1]["status"] = "completed" if route == "finalize" else "rejected"
            steps[-1]["gate_id"] = gate["gate_id"]
        return {"gate": gate, "gate_history": state["gate_history"] + [gate], "review_request": {},
                "steps": steps, "status": status, "stop_reason": stop_reason, "pending": {}, "budget": budget}

    def wait(self, state: ResearchState):
        interrupt = graph_api()[-1]
        answer = interrupt(state["pending"])
        patch = {"status": "running", "budget": self.ledger.budget()}
        if state["pending"]["kind"] == "user":
            facts = deep_merge(state["facts"], answer.get("facts", {}))
            issues = [issue for issue in state["issues"] if issue.get("fields") != state["pending"]["fields"]]
            if answer.get("unavailable"):
                issues.append({"kind": "needs_info", "unavailable": answer["unavailable"]})
            steps = copy.deepcopy(state["steps"])
            steps[-1]["status"] = "completed"
            artifacts = dict(state["artifacts"])
            for name in ("deployment", "experiment"):
                artifacts.pop(name, None)
            patch.update(facts=facts, issues=issues, steps=steps, artifacts=artifacts)
        return patch

    def finalize(self, state: ResearchState):
        return {"finished_at": cst_now(), "pending": {}, "budget": self.ledger.budget()}

    def build(self, saver):
        StateGraph, START, END, *_ = graph_api()
        builder = StateGraph(ResearchState)
        for name in ("planner", "execute", "verifier", "quality_gate", "wait", "finalize"):
            builder.add_node(name, getattr(self, name))
        builder.add_edge(START, "planner")
        def after_plan(state):
            if state["status"] in {"partial", "complete"}:
                return "quality_gate"
            if state["pending"]:
                return "wait"
            return "execute" if state["next_action"] else "planner"
        def after_tool(state):
            if state["status"] in {"partial", "complete"}:
                return "quality_gate"
            if state["pending"]:
                return "wait"
            return "verifier" if state["review_request"] else "planner"
        def after_review(state):
            if state["pending"]:
                return "wait"
            return "verifier" if state["verification"][review_key(state["review_request"])]["status"] == "retry" else "quality_gate"
        builder.add_conditional_edges("planner", after_plan, ["execute", "wait", "planner", "quality_gate"])
        builder.add_conditional_edges("execute", after_tool, ["planner", "wait", "verifier", "quality_gate"])
        builder.add_conditional_edges("verifier", after_review, ["verifier", "wait", "quality_gate"])
        builder.add_conditional_edges("quality_gate", lambda s: s["gate"]["route"], ["planner", "execute", "finalize"])
        builder.add_conditional_edges("wait", lambda s: s["pending"]["resume_node"], ["planner", "execute", "verifier"])
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=saver, name="Research Orchestrator")


def render_task(state: ResearchState) -> str:
    raw = state
    state = delivery_view(state)
    links = EvidenceLinks(state["evidence"])
    statuses = {"complete": "已完成", "partial": "部分完成", "waiting": "等待会话任务", "waiting_input": "等待补充信息"}
    lines = ["# 研究任务报告", "", state["goal"], "", "状态：" + statuses.get(state["status"], "处理中"), "",
             "[原文证据](evidence.md) · [运行与审查记录](run-details.md)", ""]
    if state.get("stop_reason"):
        lines += ["交付范围：" + state["stop_reason"], ""]
    if state["analyses"]:
        lines += ["## 论文讲解", ""]
        for pid, analysis in state["analyses"].items():
            lines += render_paper(Paper(**state["papers"][pid]), analysis,
                state.get("paper_explanations", {}).get(pid), links, heading="###")
    candidates = [p for pid, p in state["papers"].items() if pid not in state["analyses"]]
    if candidates:
        lines += ["## 待读候选", ""]
        for paper in candidates:
            lines += ["### [%s](%s)" % (paper["title"], paper["url"]), "", "摘要线索：" + paper["abstract"], ""]
    comparison = state["artifacts"].get("comparison")
    if comparison:
        lines += ["## 方法比较", ""]
        rows = comparison["rows"]
        lines += ["| 比较维度 | " + " | ".join(cell(r["title"]) for r in rows) + " |",
                  "|---|" + "---|" * len(rows)]
        for kind, label, field in (("innovation", "创新变化", "novelty"), ("method", "处理流程", "pipeline"),
                                   ("result", "实验支持", "results"), ("limitation", "适用范围与局限", "limitations")):
            values = []
            for row in rows:
                part = state.get("paper_explanations", {}).get(row["paper_id"], {}).get(field)
                text = part["text"] + " " + links.refs(part["evidence_ids"]) if part and part["text"] else "；".join(
                    c["text"] + " " + links.refs([c["evidence_id"]]) for c in row[kind]) or "当前证据不足"
                values.append(cell(text))
            lines.append("| %s | %s |" % (label, " | ".join(values)))
        lines += ["", comparison["reason"], ""]
    deployment = state["artifacts"].get("deployment")
    if deployment:
        lines += ["## 部署评估", ""]
        for row in deployment["assessments"]:
            title = state["papers"][row["paper_id"]]["title"]
            lines += ["### " + title, "", row["reasoning"], "", "依据：" + links.refs(row["evidence_ids"]), "", "验证步骤：", ""]
            lines += ["- " + step for step in row["verification_steps"]] + [""]
            if row["missing_information"]:
                lines += ["待确认：" + "；".join(row["missing_information"]), ""]
    if state["artifacts"].get("experiment"):
        lines += ["## 实验方案", ""]
        for idea in state["artifacts"]["experiment"]:
            lines += render_idea(idea, links, heading="###")
    pending = raw.get("pending", {})
    if pending.get("question"):
        lines += ["## 待补充信息", "", pending["question"], ""]
    gaps = list(dict.fromkeys(issue["reason"] for issue in raw.get("gate", {}).get("issues", [])))
    if gaps:
        lines += ["## 影响结论的缺口", ""] + ["- " + gap for gap in gaps] + [""]
    return "\n".join(lines)


def render_task_details(state: ResearchState) -> str:
    import json
    raw = state
    state = delivery_view(state)
    lines = ["# 研究任务运行记录", "", "[返回报告](report.md)", "", "任务：%s · 状态：%s" % (state["task_id"], state["status"]),
             "", "## 执行计划", ""]
    lines += ["%d. %s" % (index + 1, item) for index, item in enumerate(state["plan"])]
    if raw.get("external_results"):
        lines += ["", "## 外部 MCP 调用", "", "外部结果作为待核验材料保存。", ""]
        for record in raw["external_results"].values():
            lines.append("- %s / %s：%s" % (record["server"], record["tool"], record["result_path"]))
            from .mcp_client import result_preview
            preview = result_preview(record["result"], 2000) if "result" in record else {"text_preview": record.get("text_preview", "")[:2000]}
            lines += ["", "```json", json.dumps(preview, ensure_ascii=False, indent=2), "```", ""]
    lines += ["", "## 步骤记录", ""]
    for step in state["steps"]:
        lines.append("- %s · %s：%s" % (step["tool"], step["status"], step.get("error") or step.get("reason", "")))
    lines += ["", "## Verifier 与 Quality Gate", ""]
    for record in raw.get("gate_history", []):
        lines.append("- %s · %s：%s" % (record["review_key"], record["decision"], record["reason"]))
        for issue in record["issues"]:
            lines.append("  审查问题 [%s/%s]：%s" % (issue["section"], issue.get("field", ""), issue["reason"]))
    latest = raw.get("gate", {})
    if latest.get("suggestions"):
        lines += ["", "建议操作：", "", "```json", json.dumps(latest["suggestions"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["", "## 信息缺口与预算", "", json.dumps(state["issues"], ensure_ascii=False, indent=2), "",
              json.dumps(state["budget"], ensure_ascii=False, indent=2), "",
              "measured_tokens 为已返回用量；charged_tokens 对未知调用使用预留估计，不能保证供应商实际 Token 的严格上限。", "",
              "结束原因：" + state["stop_reason"], ""]
    if raw.get("context_metrics"):
        lines += ["", "## 上下文与人工核验", "", json.dumps(raw["context_metrics"], ensure_ascii=False, indent=2), "",
                  "estimated_input_tokens 是本地估计；measured_input_tokens 仅累计供应商已返回的输入用量。", "",
                  "结构化恢复记录见 memory.json；人工抽样见 audit-sample.json，填写后运行 research-audit。", "",
                  json.dumps(raw.get("human_audit", {}), ensure_ascii=False, indent=2), ""]
    if state["pending"]:
        lines += ["待办：", "", "```json", json.dumps(state["pending"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def _snapshot(folder: Path):
    _, _, _, SqliteSaver, *_ = graph_api()
    with SqliteSaver.from_conn_string(str(folder / "graph.sqlite3")) as saver:
        meta = read_json(folder / "task.json")
        # get_state applies pending writes from completed nodes in an interrupted
        # superstep. Reading raw channel_values can return the pre-wait state.
        graph = ResearchOrchestrator(None).build(saver)
        snapshot = graph.get_state({"configurable": {"thread_id": meta["task_id"]}})
        if not snapshot.values:
            return meta["initial_state"]
        state = dict(snapshot.values)
        state["budget"] = saved_budget(folder, state["limits"])
        state["context_metrics"] = saved_context_metrics(folder)
        return state


def task_status(folder: Path):
    folder = Path(folder).resolve()
    if not (folder / "task.json").is_file():
        raise ValueError("找不到研究任务 task.json：" + str(folder))
    with FileLock(folder / ".lock"):
        return _snapshot(folder)


def validate_answer(state, answer):
    if not isinstance(answer, dict) or not isinstance(answer.get("facts", {}), dict) or not isinstance(answer.get("unavailable", []), list):
        raise ValueError("补充信息应为 {facts: {...}, unavailable: [...]} JSON 对象")
    facts = deep_merge(state["facts"], answer.get("facts", {}))
    for field in state["pending"]["fields"]:
        value = facts
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value in (None, "", [], {}) and field not in answer.get("unavailable", []):
            raise ValueError("请提供 facts.%s 或在 unavailable 中声明无法提供" % field)


def run_research(config: dict | None = None, goal: str = "", *, folder: Path | None = None,
                 inputs: dict | None = None, limits: dict | None = None, answer: dict | None = None,
                 backend=None, search=None, reader=None) -> dict:
    StateGraph, START, END, SqliteSaver, Command, interrupt = graph_api()
    fresh = folder is None
    if fresh:
        if not config:
            raise ValueError("新任务需要项目配置")
        task_id = str(uuid.uuid4())
        state = initial_state(task_id, goal, config, inputs if inputs is not None else {}, checked_limits(limits))
        skill_run = (inputs or {}).get("skill_run")
        if skill_run is not None:
            if not isinstance(skill_run, dict):
                raise ValueError("技能运行信息必须为对象")
            loaded = state["loaded_skills"].get(skill_run.get("skill"), {})
            if (loaded.get("version") != skill_run.get("version") or
                    skill_run.get("input_hash") != fingerprint(skill_run.get("input"))):
                raise ValueError("技能运行信息应绑定已加载技能的版本与输入")
        folder = resolve_path(config, "output_dir") / "research-tasks" / task_id
        folder.mkdir(parents=True, exist_ok=False)
    else:
        folder = Path(folder).resolve()
        if not (folder / "task.json").is_file():
            raise ValueError("找不到研究任务 task.json：" + str(folder))
    with FileLock(folder / ".lock"):
        if fresh:
            write_json(folder / "task.json", {"version": 1, "task_id": task_id, "implementation": implementation_stamp(), "initial_state": state})
            if skill_run is not None:
                write_json(folder / "skill-run.json", skill_run)
        else:
            meta = read_json(folder / "task.json")
            if meta.get("version") != 1:
                raise ValueError("不支持的研究任务版本")
            if meta["implementation"] != implementation_stamp():
                raise ValueError("任务代码或 LangGraph 版本已变化，请使用原版本恢复或创建新任务")
            state = _snapshot(folder)
        own_backend = backend is None
        if own_backend:
            backend = get_backend(state["config"]["agent"], Path(state["config"]["_root"]))
        if backend is None:
            raise ValueError("研究编排需要 codex-session、codex 或 harness 后端")
        reasoning_backend = backend
        if own_backend and state["config"]["agent"]["backend"] == "harness":
            settings = state["config"]["agent"].get("harness", {})
            model = settings.get("reasoning_model")
            if model and model != settings.get("model"):
                reasoning_backend = get_backend(state["config"]["agent"], Path(state["config"]["_root"]), model=model)
        ledger = CallLedger(folder, backend, state["limits"], reasoning_backend=reasoning_backend)
        try:
            with SqliteSaver.from_conn_string(str(folder / "graph.sqlite3")) as saver:
                runtime = ResearchOrchestrator(ledger, search=search, reader=reader)
                graph = runtime.build(saver)
                graph_config = {"configurable": {"thread_id": state["task_id"]},
                                "recursion_limit": 8 * state["limits"]["max_steps"] + 30}
                snapshot = graph.get_state(graph_config)
                if not fresh and state["status"] in {"complete", "partial"} and not snapshot.next:
                    pass
                else:
                    command = state if not snapshot.values else None
                    if snapshot.tasks and any(task.interrupts for task in snapshot.tasks):
                        if state["pending"]["kind"] == "user":
                            if answer is None:
                                return export_task(folder, state, ledger)
                            validate_answer(state, answer)
                        # An empty dict is treated as an empty interrupt-ID map
                        # by LangGraph, so use a scalar signal for file responses.
                        command = Command(resume=answer if state["pending"]["kind"] == "user" else True)
                    graph.invoke(command, graph_config, durability="sync")
                    state = dict(graph.get_state(graph_config).values)
                return export_task(folder, state, ledger)
        finally:
            ledger.close()
            if own_backend and hasattr(backend, "close"):
                backend.close()
            if own_backend and reasoning_backend is not backend and hasattr(reasoning_backend, "close"):
                reasoning_backend.close()


def export_task(folder: Path, state: dict, ledger: CallLedger):
    from .research_memory import memory_snapshot
    from .research_audit import audit_sample, audit_metrics, validate_sample_labels
    state = dict(state, budget=ledger.budget(), context_metrics=ledger.context_metrics())
    sample = audit_sample(state)
    labels_path = folder / "human-audit" / sample["sample_id"] / "labels.json"
    labels = validate_sample_labels(sample, read_json(labels_path)) if labels_path.exists() else sample
    state["human_audit"] = audit_metrics(labels)
    write_json(folder / "audit-sample.json", sample)
    write_json(folder / "human-audit" / sample["sample_id"] / "sample.json", sample)
    write_json(folder / "state.json", state)
    write_json(folder / "memory.json", memory_snapshot(state, folder / "state.json"))
    delivered = delivery_view(state)
    atomic_text(folder / "evidence.md", render_evidence(delivered["evidence"], reviewed=True))
    atomic_text(folder / "run-details.md", render_task_details(state))
    write_json(folder / "paper-explanations.json", {"schema_version": 1, "papers": delivered.get("paper_explanations", {}),
        "content_checks": {pid: content_gaps(delivered.get("paper_explanations", {}).get(pid, empty_explanation(pid)))
                           for pid in delivered["analyses"]}})
    atomic_text(folder / "report.md", render_task(state))
    return {"task_id": state["task_id"], "status": state["status"], "task_dir": str(folder),
            "state_path": str(folder / "state.json"), "report_path": str(folder / "report.md"),
            "evidence_path": str(folder / "evidence.md"), "run_details_path": str(folder / "run-details.md"),
            "explanations_path": str(folder / "paper-explanations.json"),
            "memory_path": str(folder / "memory.json"), "audit_sample_path": str(folder / "audit-sample.json"),
            "context_metrics": state["context_metrics"], "human_audit": state["human_audit"],
            "pending": state["pending"], "budget": state["budget"], "stop_reason": state["stop_reason"]}
