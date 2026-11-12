"""Reusable skill entrypoints over the existing screening, audit and graph runtime."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ._compat import FileLock
from .agents import TaskPending, get_backend
from .config import load_config, resolve_path
from .contracts import ANALYSIS_SCHEMA, validate_analysis, validate_shape
from .models import Paper
from .pipeline import cst_now
from .quality_gate import evaluate_gate, review_key
from .ranking import rank_papers
from .research_calls import BudgetExceeded, CallLedger, read_json, saved_budget, write_json
from .research_state import checked_limits, fingerprint, initial_state
from .research_tools import ResearchTools
from .research_verifier import Verifier, approved_ids, approved_state
from .skill_registry import contract, load_file, load_skill, spec
from .sources import deduplicate_papers


def validate_input(value, schema, location="input"):
    """The subset used by the bundled contracts, without a new runtime dependency."""
    kind = schema.get("type")
    accepted = {"object": isinstance(value, dict), "array": isinstance(value, list),
                "string": isinstance(value, str), "integer": type(value) is int,
                "boolean": type(value) is bool}.get(kind, False)
    if not accepted:
        raise ValueError(location + " 应为 " + str(kind))
    if kind == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ValueError(location + " 缺少字段：" + ", ".join(sorted(missing)))
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(location + " 包含未知字段：" + ", ".join(sorted(set(value) - set(properties))))
        for key, sub in properties.items():
            if key in value:
                validate_input(value[key], sub, location + "." + key)
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100000):
            raise ValueError(location + " 的条目数量超出范围")
        for index, item in enumerate(value):
            validate_input(item, schema["items"], location + "[%d]" % index)
    elif kind == "string":
        if not schema.get("minLength", 0) <= len(value.strip()) <= schema.get("maxLength", 10000):
            raise ValueError(location + " 的文本长度超出范围")
    elif kind == "integer" and not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
        raise ValueError(location + " 的数值超出范围")


def checked_input(name, payload):
    validate_input(payload, contract(name)["input_schema"])
    papers = []
    for row in payload.get("papers", []):
        try:
            paper = Paper(**row)
        except TypeError as exc:
            raise ValueError("papers 必须满足 Paper 字段契约") from exc
        for field in ("paper_id", "source", "title", "abstract", "published", "updated", "url", "pdf_url", "full_text"):
            if not isinstance(getattr(paper, field), str):
                raise ValueError("论文字段应为字符串：" + field)
        if not paper.paper_id or not paper.title.strip():
            raise ValueError("论文 ID 和标题不能为空")
        for field in ("authors", "categories", "figure_pages"):
            if not isinstance(getattr(paper, field), list) or any(not isinstance(v, str) for v in getattr(paper, field)):
                raise ValueError("论文字段应为字符串数组：" + field)
        if type(paper.attention) is not int or paper.attention < 0:
            raise ValueError("attention 应为非负整数")
        papers.append(paper)
    checked_limits(payload.get("limits"))
    if name != "paper-screening" and len({p.paper_id for p in papers}) != len(papers):
        raise ValueError("论文 ID 不能重复")
    if name == "evidence-verification":
        validate_shape({"analyses": payload["analyses"]}, ANALYSIS_SCHEMA)
        ids = [row["paper_id"] for row in payload["analyses"]]
        if len(ids) != len(set(ids)) or not set(ids) <= {p.paper_id for p in papers}:
            raise ValueError("analyses 的论文 ID 应存在于 papers 中且不重复")
        by_id = {p.paper_id: p for p in papers}
        for row in payload["analyses"]:
            validate_analysis(row, by_id[row["paper_id"]])
    return copy.deepcopy(payload)


def prepare(name, payload):
    checked_input(name, payload)
    row = spec(name)
    return {"skill": name, "version": row["version"], "input_hash": fingerprint(payload),
            "status": "ready", "resources": list(row["files"]),
            "mode": "offline" if name == "paper-screening" else "orchestrator" if name == "experiment-design" else "audit"}


def runtime_stamp(name):
    files = ("skill_runner.py", "skill_registry.py", "research_verifier.py", "quality_gate.py",
             "research_state.py", "research_calls.py", "research_tools.py", "paper_explanations.py", "contracts.py",
             "agents.py", "config.py", "models.py", "ranking.py", "sources.py", "_compat.py")
    row = spec(name)
    for resource in row["files"]:
        load_file(name, resource, expected=row)
    return {"skill": row, "source": {file: hashlib.sha256((Path(__file__).parent / file).read_bytes()).hexdigest() for file in files}}


def screening(payload, now=None):
    now = now or datetime.now(timezone.utc)
    raw = [Paper(**row) for row in payload["papers"]]
    unique = deduplicate_papers(raw)
    excluded = [{"paper_id": p.paper_id, "reason": "duplicate"} for p in raw if all(p is not kept for kept in unique)]
    eligible = []
    for paper in unique:
        if "days" in payload:
            try:
                date = datetime.fromisoformat((paper.updated or paper.published).replace("Z", "+00:00"))
                if date.tzinfo is None:
                    raise ValueError("timezone required")
                date = date.astimezone(timezone.utc)
            except ValueError:
                excluded.append({"paper_id": paper.paper_id, "reason": "unknown_date"})
                continue
            if not now - timedelta(days=payload["days"]) <= date <= now:
                excluded.append({"paper_id": paper.paper_id, "reason": "outside_window"})
                continue
        eligible.append(paper)
    ranked = rank_papers(eligible, payload["queries"], payload.get("limit", 10))
    excluded.extend({"paper_id": p.paper_id, "reason": "over_limit"} for p in eligible if all(p is not kept for kept in ranked))
    return {"status": "complete", "candidates": [p.as_dict() for p in ranked], "excluded": excluded,
            "score_kind": "metadata_priority", "screened_at": now.isoformat(),
            "scoring": "0.65 relevance + 0.25 freshness + 0.10 attention；筛选优先级，不是论文科学质量"}


def audit_state(name, payload, config, task_id):
    if name == "method-comparison":
        from .orchestrator import task_status
        source = task_status(Path(payload["task_dir"]).expanduser().resolve())
        ResearchTools.guard_compare(source, {"paper_ids": payload.get("paper_ids", [])})
        state = copy.deepcopy(source)
        state.update(task_id=task_id, goal=payload.get("goal", "对已有审查证据进行方法对比"),
                     config=copy.deepcopy(config or source["config"]), limits=checked_limits(payload.get("limits")),
                     status="running", pending={}, finished_at=None, budget={}, stop_reason="",
                     review_history=[], gate_history=[], gate={}, review_request={}, artifacts={}, deliverables=["comparison"])
        # Keep source reading steps: approved_ids binds them as part of its signature.
        state["origin_task_id"] = source["task_id"]
        state["selected_paper_ids"] = payload.get("paper_ids", []) or list(source["papers"])
        state["artifacts"] = ResearchTools(None).compare_methods(approved_state(state),
                               {"paper_ids": state["selected_paper_ids"]}, None, "")["artifacts"]
        return state, {"phase": "artifact", "artifacts": ["comparison"]}
    if config is None:
        raise ValueError("证据核查需要项目配置")
    state = initial_state(task_id, payload.get("goal", "核查所提供断言的引文及证据支持关系"), config,
                          {"papers": payload["papers"], "facts": payload.get("facts", {})}, checked_limits(payload.get("limits")))
    by_id = {row["paper_id"]: Paper(**row) for row in payload["papers"]}
    for row in payload["analyses"]:
        paper = by_id[row["paper_id"]]
        normalized = validate_analysis(row, paper)
        accepted_quotes = {claim["anchor_quote"] for claim in normalized["claims"]}
        normalized["claims"] = []
        for claim in row["claims"]:
            eid = "ER-" + hashlib.sha256((paper.paper_id + claim["anchor_quote"]).encode()).hexdigest()[:16]
            normalized["claims"].append(dict(claim, evidence_id=eid))
            if claim["anchor_quote"] in accepted_quotes:
                state["evidence"][eid] = dict(claim, evidence_id=eid, paper_id=paper.paper_id,
                                              source_url=paper.url, version=paper.updated)
        # Preserve rejected candidates so the independent audit records every failure.
        state["analyses"][paper.paper_id] = normalized
    state["deliverables"] = ["analysis"]
    return state, {"phase": "evidence", "artifacts": []}


def export(folder, meta, state, result):
    common = {"skill": meta["skill"], "version": meta["version"], "input_hash": meta["input_hash"],
              "run_dir": str(folder), "result_path": str(folder / "result.json")}
    output = dict(common, **result)
    write_json(folder / "state.json", state)
    write_json(folder / "result.json", output)
    return output


def run_audit(folder, meta, state, backend=None):
    own_backend = backend is None
    reasoning_backend = backend
    if own_backend:
        agent = state["config"]["agent"]
        workspace = Path(state["config"]["_root"])
        backend = get_backend(agent, workspace)
        reasoning_backend = backend
        model = agent.get("harness", {}).get("reasoning_model") if agent["backend"] == "harness" else None
        if model and model != agent["harness"].get("model"):
            reasoning_backend = get_backend(agent, workspace, model=model)
    if backend is None:
        raise ValueError("技能核查需要 codex-session、codex 或 harness 后端")
    ledger = CallLedger(folder, backend, state["limits"], reasoning_backend=reasoning_backend)
    try:
        request = meta["review_request"]
        try:
            if request["phase"] == "evidence" and not any(a["claims"] for a in state["analyses"].values()):
                raise ValueError("没有可核查的正文断言；请先取得正文并准备分析")
            report = Verifier(ledger).review(state, request)
        except TaskPending as exc:
            state.update(status="waiting", pending={"kind": "session", "task": exc.task}, budget=ledger.budget())
            return export(folder, meta, state, {"status": "waiting", "pending": state["pending"], "budget": state["budget"]})
        except (BudgetExceeded, ValueError, RuntimeError, OSError) as exc:
            report = {"status": "unavailable", "items": [], "error": str(exc)}
        state["verification"][review_key(request)] = report
        gate = evaluate_gate(state, request, report, ledger.budget())
        if gate["decision"] in {"repair", "read_more", "retrieve_more", "needs_input"}:
            gate["reason"] = "质量审查发现具体缺口，交由调用 Agent 决定补读、检索、修复或补充信息"
        state.update(status="complete" if gate["decision"] == "pass" else "partial", pending={},
                     budget=ledger.budget(), gate=gate, finished_at=cst_now())
        result = {"status": state["status"], "verification": report, "gate": gate,
                  "approved_evidence_ids": sorted(approved_ids(state)), "pending": {}, "budget": state["budget"]}
        if meta["skill"] == "method-comparison":
            result["comparison"] = state["artifacts"]["comparison"] if gate["decision"] == "pass" else None
            result["candidate_comparison_path"] = str(folder / "state.json")
            result["origin_task_id"] = state["origin_task_id"]
        return export(folder, meta, state, result)
    finally:
        ledger.close()
        if own_backend and hasattr(backend, "close"):
            backend.close()
        if own_backend and reasoning_backend is not backend and hasattr(reasoning_backend, "close"):
            reasoning_backend.close()


def run_skill(name, payload, *, config=None, folder=None, backend=None, search=None, reader=None):
    payload = checked_input(name, payload)
    load_skill(name)
    version = spec(name)["version"]
    if name == "experiment-design":
        if config is None or folder is not None:
            raise ValueError("实验设计需要项目配置，新任务目录由编排器生成")
        from .orchestrator import run_research
        meta = {"skill": name, "version": version, "input_hash": fingerprint(payload), "input": payload}
        inputs = {"skill_run": meta, "papers": payload.get("papers", []), "facts": payload.get("facts", {}), "skill_name": name,
                  "required_outputs": ["deployment", "experiment"] if payload.get("deployment") else ["experiment"]}
        result = run_research(config, payload["goal"], inputs=inputs, limits=payload.get("limits"),
                              backend=backend, search=search, reader=reader)
        folder = Path(result["task_dir"])
        output = dict(result, skill=name, version=version, input_hash=meta["input_hash"], run_dir=str(folder), result_path=str(folder / "result.json"))
        write_json(folder / "result.json", output)
        return output
    task_id = str(uuid.uuid4())
    state, request = ({}, {}) if name == "paper-screening" else audit_state(name, payload, config, task_id)
    config = state.get("config", config)
    base = resolve_path(config, "output_dir") if config else Path.cwd() / "reports"
    folder = Path(folder).resolve() if folder is not None else base / "skill-runs" / task_id
    folder.mkdir(parents=True, exist_ok=False)
    meta = {"skill": name, "version": version, "input_hash": fingerprint(payload), "input": payload,
            "implementation": runtime_stamp(name), "review_request": request}
    with FileLock(folder / ".lock"):
        write_json(folder / "skill-run.json", meta)
        write_json(folder / "state.json", state)
        if name == "paper-screening":
            return export(folder, meta, state, screening(payload))
        return run_audit(folder, meta, state, backend)


def resume_skill(name, folder, *, backend=None, answer=None, search=None, reader=None):
    folder = Path(folder).expanduser().resolve()
    meta = read_json(folder / "skill-run.json")
    if meta["skill"] != name:
        raise ValueError("运行目录不属于当前技能")
    if name == "experiment-design":
        from .orchestrator import run_research
        result = run_research(folder=folder, backend=backend, answer=answer, search=search, reader=reader)
        output = dict(result, skill=name, version=meta["version"], input_hash=meta["input_hash"],
                      run_dir=str(folder), result_path=str(folder / "result.json"))
        write_json(folder / "result.json", output)
        return output
    with FileLock(folder / ".lock"):
        if meta["implementation"] != runtime_stamp(name):
            raise ValueError("技能代码或资源版本已变化，请使用原版本恢复或创建新运行")
        if (folder / "result.json").exists():
            previous = read_json(folder / "result.json")
            if previous["status"] in {"complete", "partial"}:
                return previous
        return run_audit(folder, meta, read_json(folder / "state.json"), backend)


def skill_status(name, folder):
    folder = Path(folder).expanduser().resolve()
    meta = read_json(folder / "skill-run.json")
    if meta["skill"] != name:
        raise ValueError("运行目录不属于当前技能")
    if name == "experiment-design":
        from .orchestrator import task_status
        state = task_status(folder)
        return {"skill": name, "version": meta["version"], "run_dir": str(folder),
                **{key: state[key] for key in ("status", "budget", "pending", "gate")}}
    result = read_json(folder / "result.json")
    if name != "paper-screening":
        result["budget"] = saved_budget(folder, read_json(folder / "state.json")["limits"])
    return result


def main(name, argv=None):
    parser = argparse.ArgumentParser(description=name + "：独立技能入口")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "resume", "status"):
        item = sub.add_parser(command)
        if command in {"prepare", "run"}:
            item.add_argument("--input", required=True, type=Path)
        else:
            item.add_argument("--run-dir", required=True, type=Path)
        if command == "run":
            item.add_argument("--config", type=Path)
            item.add_argument("--backend", choices=["codex-session", "codex", "harness"])
        if command == "resume":
            item.add_argument("--answer", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(name, read_json(args.input))
        elif args.command == "run":
            config = load_config(args.config or "config.json") if name in {"evidence-verification", "experiment-design"} or args.config else None
            if args.backend and config is None:
                if name != "method-comparison":
                    raise ValueError("--backend 需要项目配置")
                from .orchestrator import task_status
                config = copy.deepcopy(task_status(Path(read_json(args.input)["task_dir"]))["config"])
            if args.backend:
                config["agent"]["backend"] = args.backend
            result = run_skill(name, read_json(args.input), config=config)
        elif args.command == "resume":
            result = resume_skill(name, args.run_dir, answer=read_json(args.answer) if args.answer else None)
        else:
            result = skill_status(name, args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if args.command in {"prepare", "status"} or result["status"] == "complete" else 2
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"skill": name, "status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1
