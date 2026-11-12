"""Small capability registry over the existing research/evidence contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import deep_merge, resolve_path, validate_config
from .contracts import ANALYSIS_SCHEMA, IDEA_SCHEMA, validate_analysis, validate_idea
from .fulltext import fetch_fulltext
from .models import Paper
from .paper_context import DEFAULT_POLICY, LEVELS, estimated_tokens, load_index, locate_quote, paper_sources, save_index, select_context
from .paper_explanations import EXPLANATION_SCHEMA, explanation_prompt, validate_explanation_batch
from .prompts import RULES, analysis_prompt, idea_prompt
from .ranking import rank_papers
from .research_calls import CallLedger, read_json, write_json
from .research_state import DEPLOYMENT_FIELDS, DEPLOYMENT_SCHEMA, ResearchState, fingerprint
from .research_verifier import approved_ids, approved_state, artifact_passed
from .sources import ArxivSource, deduplicate_papers
from .skill_registry import load_file, load_skill


@dataclass(frozen=True)
class Capability:
    name: str
    role: str
    description: str
    handler: Callable
    guard: Callable


def selected(state: ResearchState, action: dict) -> list[str]:
    ids = action["paper_ids"] or list(state["papers"])
    if not ids or len(ids) != len(set(ids)) or any(pid not in state["papers"] for pid in ids):
        raise ValueError("请选择本任务中存在且不重复的论文 ID")
    return ids


def require_analysis(state: ResearchState, action: dict) -> list[str]:
    ids = selected(state, action)
    if any(pid not in state["analyses"] for pid in ids):
        raise ValueError("需先调用 analyze_papers 分析所选论文")
    return ids


def require_evidence(state: ResearchState, action: dict) -> list[str]:
    ids = require_analysis(state, action)
    if any(not state["analyses"][pid]["claims"] for pid in ids):
        raise ValueError("所选论文缺少校验通过的全文证据，需补读/检索或交付部分结果")
    allowed = approved_ids(state)
    if any(not any(c["evidence_id"] in allowed for c in state["analyses"][pid]["claims"]) for pid in ids):
        raise ValueError("所选论文缺少通过独立 Verifier 审查的证据")
    return ids


def missing_deployment(state: ResearchState) -> list[str]:
    facts = state["facts"].get("deployment", {})
    if not isinstance(facts, dict):
        return list(DEPLOYMENT_FIELDS)
    return [key for key in DEPLOYMENT_FIELDS if facts.get(key) in (None, "", [], {})]


class ResearchTools:
    """One planner plus analyst/designer roles; all use the existing backend ABI.

    Roles specialize prompts/contracts. They are not autonomous agents and do
    not privately browse, spawn workers, or communicate outside this task.
    """
    def __init__(self, ledger: CallLedger, search=None, reader=None):
        self.ledger = ledger
        self.source = ArxivSource()
        self.search = search or (lambda query, categories, days, limit: self.source.search(query, categories, days, limit, "latest"))
        self.reader = reader or fetch_fulltext
        self.registry = {
            cap.name: cap for cap in (
                Capability("search_papers", "collector", "按 query/days 检索 arXiv；可改写查询补检索，每任务最多10篇", self.search_papers, self.guard_search),
                Capability("read_papers", "reader", "读取论文索引；query 指定主题，先读取相关片段，同一主题再次补读时逐级扩大至章节和原文，受上下文预算限制", self.read_papers, selected),
                Capability("analyze_papers", "literature_analyst", "分析 paper_ids，校验连续引文；摘要仅发现线索", self.analyze_papers, selected),
                Capability("compare_methods", "literature_analyst", "按已校验的创新/方法/结果/局限生成比较表；需至少2篇有证据的论文", self.compare_methods, self.guard_compare),
                Capability("assess_deployment", "experiment_designer", "根据全文证据与 facts.deployment 给出条件适用性和上机验证步骤", self.assess_deployment, self.guard_deployment),
                Capability("design_experiment", "experiment_designer", "生成引用Evidence ID的最小实验、baseline、指标及反证条件", self.design_experiment, self.guard_experiment),
                Capability("repair_result", "repair", "按 Quality Gate 的反馈修复 analysis/comparison/deployment/experiment，query 指定目标，修复后重新审查", self.repair_result, self.guard_repair),
                Capability("mcp_list_tools", "mcp_client", "初始化并发现配置中 server 的允许工具及输入输出 Schema；query 填 server 名称", self.mcp_list_tools, self.guard_mcp),
                Capability("mcp_call_tool", "mcp_client", "调用已发现的允许工具；query 填 {server,tool,arguments} JSON，保留标准结果和审计回执", self.mcp_call_tool, self.guard_mcp),
                Capability("mcp_import_paper", "reader", "从 Zotero MCP 的元数据和正文导入论文；query 填 {server,item_key} JSON，导入后需要分析和证据审查", self.mcp_import_paper, self.guard_mcp),
                Capability("load_skill", "knowledge", "按 query 中的技能名称加载 SKILL.md；只读取执行知识，不运行脚本", self.load_skill, self.guard_skill),
                Capability("load_skill_resource", "knowledge", "按 query 中的 name/资源路径加载已声明文件；需先载入该技能", self.load_skill_resource, self.guard_skill),
            )
        }

    def catalog(self):
        return [{"name": cap.name, "role": cap.role, "description": cap.description} for cap in self.registry.values()] + [
            {"name": "ask_user", "role": "orchestrator", "description": "通过 question/required_fields 请求必要事实；仅暂停此任务"},
            {"name": "finish", "role": "orchestrator", "description": "说明 reason，按 finish_status 交付 complete 或 partial"}]

    @staticmethod
    def profile(state):
        override = {k: state["facts"][k] for k in ("lab_context", "compute") if k in state["facts"]}
        if any(not isinstance(value, dict) for value in override.values()):
            raise ValueError("facts.lab_context 和 facts.compute 应为对象")
        profile = deep_merge(state["config"]["research_profile"], override)
        validate_config(dict(state["config"], research_profile=profile))
        return dict(profile, task_goal=state["goal"], task_facts=state["facts"])

    @staticmethod
    def guard_search(state, action):
        if not action["query"].strip() or type(action["days"]) is not int:
            raise ValueError("检索需要非空 query 和整数 days")
        if not state["config"]["sources"]["arxiv"]:
            raise ValueError("配置中 arxiv 来源未启用")

    @staticmethod
    def guard_compare(state, action):
        if len(require_evidence(state, action)) < 2:
            raise ValueError("方法对比需要至少两篇有全文证据的论文")

    @staticmethod
    def guard_deployment(state, action):
        require_evidence(state, action)
        missing = missing_deployment(state)
        if missing:
            raise ValueError("需 ask_user 补充 facts.deployment：" + ", ".join(missing))
        for key in ("memory_gb", "target_latency_ms"):
            value = state["facts"]["deployment"][key]
            if type(value) not in (int, float) or value <= 0:
                raise ValueError("facts.deployment.%s 应为正数" % key)

    @staticmethod
    def guard_experiment(state, action):
        require_evidence(state, action)
        if "deployment" in state["deliverables"] and "deployment" not in state["artifacts"]:
            raise ValueError("部署实验应先完成 assess_deployment")
        if "deployment" in state["deliverables"] and not artifact_passed(state, "deployment"):
            raise ValueError("部署评估需先通过 Verifier 审查")

    @staticmethod
    def guard_repair(state, action):
        if action["query"] not in {"analysis", "comparison", "deployment", "experiment"}:
            raise ValueError("repair_result 的 query 应为 analysis/comparison/deployment/experiment")
        if not state.get("gate", {}).get("issues"):
            raise ValueError("当前没有 Quality Gate 的修复反馈")
        if action["query"] in {"analysis", "comparison"}:
            selected(state, action)
        elif action["query"] == "deployment":
            ResearchTools.guard_deployment(state, action)
        else:
            ResearchTools.guard_experiment(state, action)

    def key(self, state, action):
        tool = action["tool"]
        if tool.startswith("mcp_"):
            inputs = {"query": action["query"], "mcp_config": state["config"].get("mcp", {}), "root": state["config"]["_root"]}
            if tool == "mcp_call_tool":
                request = self.mcp_arguments(action)
                inputs["discovery"] = state.get("external_tools", {}).get(request["server"])
            if tool == "mcp_import_paper":
                inputs.update(goal=state["goal"], paper_ids=sorted(state["papers"]), context_policy=state["context_policy"])
        elif tool in {"load_skill", "load_skill_resource"}:
            inputs = {"query": action["query"], "skill_catalog": state["skill_catalog"]}
        elif tool == "search_papers":
            inputs = {"query": action["query"], "days": action["days"]}
        else:
            ids = selected(state, action)
            inputs = {"papers": [state["papers"][pid] for pid in ids]}
            if tool in {"read_papers", "analyze_papers"}:
                inputs["paper_contexts"] = {pid: state.get("paper_contexts", {}).get(pid) for pid in ids}
                inputs["context_policy"] = state.get("context_policy", DEFAULT_POLICY)
            if tool == "read_papers":
                inputs["query"] = action["query"]
            if tool != "read_papers":
                inputs["facts"] = state["facts"]
            if tool not in {"read_papers", "analyze_papers"}:
                inputs.update(analyses={pid: state["analyses"].get(pid) for pid in ids}, facts=state["facts"])
            if tool == "design_experiment":
                inputs["deployment"] = state["artifacts"].get("deployment")
            if tool in {"compare_methods", "assess_deployment", "design_experiment", "repair_result"}:
                inputs["verification"] = state.get("verification", {}).get("evidence")
            if tool == "repair_result":
                inputs.update(target=action["query"], feedback=state["gate"]["issues"], artifacts=state["artifacts"])
        return fingerprint({"tool": tool, "inputs": inputs})

    @staticmethod
    def mcp_arguments(action):
        import json
        try:
            value = json.loads(action["query"], parse_constant=lambda s: (_ for _ in ()).throw(ValueError("非法 JSON 数字")))
        except ValueError as exc:
            raise ValueError("MCP query 必须为 JSON 对象") from exc
        if not isinstance(value, dict) or not isinstance(value.get("server"), str):
            raise ValueError("MCP query 必须包含 server")
        return value

    @staticmethod
    def guard_mcp(state, action):
        from .mcp_client import server_profile, validate_json_schema
        if action["tool"] == "mcp_list_tools":
            server_profile(state["config"], action["query"])
            return
        request = ResearchTools.mcp_arguments(action)
        profile = server_profile(state["config"], request["server"])
        if action["tool"] == "mcp_import_paper":
            import re
            if set(request) != {"server", "item_key"} or not isinstance(request["item_key"], str) or not re.fullmatch(r"[A-Za-z0-9]{8}", request["item_key"]):
                raise ValueError("导入论文 query 需要 server 与 8 位 item_key")
            if not {"zotero_get_item_metadata", "zotero_get_item_fulltext"} <= set(profile.get("allowed_tools", [])):
                raise ValueError("Zotero 导入需要允许 metadata 和 fulltext 两个工具")
            if len(state["papers"]) >= 10 and "zotero:%s:%s" % (request["server"], request["item_key"]) not in state["papers"]:
                raise ValueError("研究任务最多导入 10 篇论文")
            return
        if set(request) != {"server", "tool", "arguments"} or not isinstance(request.get("tool"), str) or not isinstance(request.get("arguments"), dict):
            raise ValueError("MCP query 需要 server、tool 和 arguments 对象")
        if request["tool"] not in profile.get("allowed_tools", []):
            raise ValueError("MCP tool 不在 allowed_tools 中")
        discovered = {row["name"]: row for row in state.get("external_tools", {}).get(request["server"], {}).get("tools", [])}
        if request["tool"] not in discovered:
            raise ValueError("需先 mcp_list_tools 发现此 server 的允许工具")
        validate_json_schema(request["arguments"], discovered[request["tool"]]["inputSchema"])

    def mcp_list_tools(self, state, action, target, key):
        from .mcp_client import function_catalog, request_sync
        server = action["query"]
        discovery = request_sync(state["config"], server, "list")
        discovery.update(function_calling=function_catalog(server, discovery["tools"]),
                         external_data=True, catalog_fingerprint=fingerprint(discovery["tools"]))
        return {"external_tools": dict(state.get("external_tools", {}), **{server: discovery})}

    def mcp_call_tool(self, state, action, target, key):
        from .mcp_client import MCPToolError, request_sync, result_preview
        request = self.mcp_arguments(action)
        try:
            result = request_sync(state["config"], request["server"], "call", tool=request["tool"], arguments=request["arguments"])
        except MCPToolError as exc:
            write_json(target / "mcp-error.json", {"request": request, "error": str(exc), "result": exc.result})
            raise
        result_path = target / "mcp-result.json"
        write_json(result_path, result)
        record = dict(request, **result_preview(result), result_path=str(result_path), result_fingerprint=fingerprint(result), external_data=True)
        discoveries = list(state["artifacts"].get("discovery", [])) + [{
            "source": "mcp", "server": request["server"], "tool": request["tool"],
            "result_path": str(result_path), "external_data": True}]
        return {"external_results": dict(state.get("external_results", {}), **{key: record}),
                "artifacts": {"discovery": discoveries}}

    def mcp_import_paper(self, state, action, target, key):
        from .zotero_mcp import fetch_zotero_paper_sync
        request = self.mcp_arguments(action)
        pid = "zotero:%s:%s" % (request["server"], request["item_key"])
        if pid in state["papers"]:
            raise ValueError("此 Zotero 文献已导入；请补读已冻结原文或创建新任务更新材料")
        receipt = target / "zotero-material.json"
        if receipt.exists():
            saved = read_json(receipt)
            paper = Paper(**saved["paper"])
            raw = saved["raw"]
        else:
            paper, raw = fetch_zotero_paper_sync(state["config"], request["server"], request["item_key"])
            write_json(receipt, {"paper": paper.as_dict(), "raw": raw})
        index, record = save_index(self.ledger.folder, paper, {"layout": paper.full_text}, ["layout"] if raw.get("local_attachment") else [])
        policy = state["context_policy"]
        selection = select_context(index, {"layout": paper.full_text}, state["goal"],
            max_chars=state["config"]["sources"]["max_text_chars"], max_tokens=policy["max_context_tokens"],
            max_fragments=policy["max_fragments"])
        paper.full_text = selection.pop("text")
        record["selection"] = selection
        record["external_source"] = dict(request, receipt_path=str(receipt))
        return {"papers": dict(state["papers"], **{paper.paper_id: paper.as_dict()}),
                "paper_contexts": dict(state.get("paper_contexts", {}), **{paper.paper_id: record})}

    @staticmethod
    def guard_skill(state, action):
        name = action["query"].split("/", 1)[0]
        if action["tool"] == "load_skill" and action["query"] != name:
            raise ValueError("load_skill 的 query 只填写技能名称；资源路径使用 load_skill_resource")
        if name not in {row["name"] for row in state.get("skill_catalog", [])}:
            raise ValueError("请选择目录中已声明的技能名称")
        if action["tool"] == "load_skill_resource" and (name not in state["loaded_skills"] or "/" not in action["query"]):
            raise ValueError("需先 load_skill，再指定 name/资源路径")

    def load_skill(self, state, action, target, key):
        name = action["query"]
        expected = next(row for row in state["skill_catalog"] if row["name"] == name)
        return {"loaded_skills": dict(state["loaded_skills"], **{name: load_skill(name, expected=expected)})}

    def load_skill_resource(self, state, action, target, key):
        name, resource = action["query"].split("/", 1)
        expected = next(row for row in state["skill_catalog"] if row["name"] == name)
        content = load_file(name, resource, expected=expected)
        return {"skill_resources": dict(state["skill_resources"], **{action["query"]: content})}

    def execute(self, state, action) -> dict:
        cap = self.registry[action["tool"]]
        cap.guard(state, action)
        key = self.key(state, action)
        target = self.ledger.operation(key)
        if (target / "result.json").exists():
            return read_json(target / "result.json")
        material = approved_state(state) if action["tool"] in {"compare_methods", "assess_deployment", "design_experiment"} else state
        result = cap.handler(material, action, target, key)
        write_json(target / "result.json", result)
        self.ledger.finish_operation(key, "completed")
        return result

    def search_papers(self, state, action, target, key):
        config = state["config"]
        found = self.search(action["query"], [], action["days"],
                            config["sources"]["max_results_per_query"])
        fresh = [p for p in deduplicate_papers(found) if p.paper_id not in state["papers"]]
        papers = dict(state["papers"])
        for paper in rank_papers(fresh, [action["query"]], max(0, 10 - len(papers))):
            papers[paper.paper_id] = paper.as_dict()
        discoveries = list(state["artifacts"].get("discovery", [])) + [
            {"query": action["query"], "days": action["days"], "found": len(found),
             "added": len(papers) - len(state["papers"]), "paper_ids": list(papers)}]
        return {"papers": papers, "artifacts": {"discovery": discoveries}}

    def prepare_context(self, state, paper, target, *, require_cache=False):
        previous = state.get("paper_contexts", {}).get(paper.paper_id, {})
        folder = self.ledger.folder if self.ledger is not None else target
        receipt = target / ("context-" + fingerprint(paper.paper_id)[:24] + ".json")
        if not previous and receipt.exists():
            previous = read_json(receipt)
        if previous:
            # Validate a supplied cache path even when the task already owns a snapshot.
            if require_cache and paper.full_text_path:
                path = Path(paper.full_text_path).resolve()
                allowed = (resolve_path(state["config"], "cache_dir") / "papers").resolve()
                if not path.is_relative_to(allowed) or path.suffix != ".txt":
                    raise ValueError("主题补读需要 cache/papers 下的完整正文，不能读取其他文件")
            index, texts = load_index(previous, folder=folder if self.ledger is not None else None)
            if index["paper_id"] != paper.paper_id or index["version"] != paper.updated:
                raise ValueError("原文索引与论文版本不一致")
            return index, texts, dict(previous)
        texts, page_sources = paper_sources(paper, resolve_path(state["config"], "cache_dir"),
                                            require_cache=require_cache)
        index, record = save_index(folder, paper, texts, page_sources)
        write_json(receipt, record)
        return index, texts, record

    def read_papers(self, state, action, target, key):
        papers = dict(state["papers"])
        contexts = dict(state.get("paper_contexts", {}))
        policy = state.get("context_policy", DEFAULT_POLICY)
        changed = set()
        for pid in selected(state, action):
            paper = Paper(**papers[pid])
            if not paper.full_text:
                receipt = target / (fingerprint(pid)[:24] + ".json")
                if receipt.exists():
                    paper = Paper(**read_json(receipt))
                else:
                    if not state["config"]["sources"]["full_text"]:
                        raise ValueError("配置中 full_text 未启用；可分析摘要或补充本地正文")
                    self.reader(paper, resolve_path(state["config"], "cache_dir"),
                                state["config"]["sources"]["max_text_chars"])
                    write_json(receipt, paper.as_dict())
            query = action["query"].strip()
            index, texts, record = self.prepare_context(state, paper, target, require_cache=bool(query))
            history = record.get("selections", [])
            matching = [row for row in history if row["query"].casefold() == query.casefold()]
            level = LEVELS[min(len(matching), len(LEVELS) - 1)] if query else "fragment"
            selection = select_context(index, texts, query or state["goal"], level=level,
                max_chars=state["config"]["sources"]["max_text_chars"],
                max_tokens=policy["max_context_tokens"], max_fragments=policy["max_fragments"],
                require_match=bool(query))
            if query and matching and selection["text"] == paper.full_text and level == "document":
                raise ValueError("当前主题在原文范围和上下文预算内没有新片段；请调整查询或交付证据缺口")
            paper.full_text = selection.pop("text")
            record.update(selection=selection, selections=history + [dict(selection, query=query)])
            if paper.full_text != state["papers"][pid]["full_text"] or (
                    contexts.get(pid) and contexts[pid]["source_fingerprint"] != record["source_fingerprint"]):
                changed.add(pid)
            contexts[pid] = record
            papers[pid] = paper.as_dict()
        if changed:
            analyses = {pid: row for pid, row in state["analyses"].items() if pid not in changed}
            evidence = {eid: row for eid, row in state["evidence"].items() if row["paper_id"] not in changed}
            return {"papers": papers, "paper_contexts": contexts, "analyses": analyses, "evidence": evidence,
                    "paper_explanations": {pid: row for pid, row in state.get("paper_explanations", {}).items() if pid not in changed},
                    "invalidate_artifacts": ["analysis", "comparison", "deployment", "experiment"],
                    "artifacts": {"analysis": {"paper_ids": list(analyses)}} if analyses else {}}
        return {"papers": papers, "paper_contexts": contexts}

    def analyze_papers(self, state, action, target, key):
        ids = selected(state, action)
        analyses, evidence = dict(state["analyses"]), dict(state["evidence"])
        profile = self.profile(state)
        papers, contexts = dict(state["papers"]), dict(state.get("paper_contexts", {}))
        indexes, source_texts = {}, {}
        explanations = dict(state.get("paper_explanations", {}))
        policy = state.get("context_policy", DEFAULT_POLICY)
        for pid in ids:
            paper = Paper(**papers[pid])
            index, texts, record = self.prepare_context(state, paper, target)
            if len(paper.full_text) > state["config"]["sources"]["max_text_chars"] or estimated_tokens(paper.full_text) > policy["max_context_tokens"]:
                selection = select_context(index, texts, state["goal"],
                    max_chars=state["config"]["sources"]["max_text_chars"], max_tokens=policy["max_context_tokens"],
                    max_fragments=policy["max_fragments"])
                paper.full_text = selection.pop("text")
                record["selection"] = selection
            contexts[pid], indexes[pid], source_texts[pid] = record, index, texts
            papers[pid] = paper.as_dict()
        for start in range(0, len(ids), 3):
            batch = [Paper(**papers[pid]) for pid in ids[start:start + 3]]
            def check(value):
                expected = {p.paper_id: p for p in batch}
                actual = [row["paper_id"] for row in value["analyses"]]
                if len(actual) != len(set(actual)) or set(actual) != set(expected):
                    raise ValueError("分析必须完整覆盖此批论文，ID不能重复")
                for row in value["analyses"]:
                    validate_analysis(row, expected[row["paper_id"]])
            prompt = analysis_prompt(batch, profile, contexts=contexts) + self.repair_feedback(state)
            value = self.ledger.call(key + "-analysis-%d" % start, "literature_analyst", prompt, ANALYSIS_SCHEMA, check)
            for row, paper in [(r, next(p for p in batch if p.paper_id == r["paper_id"])) for r in value["analyses"]]:
                analysis = validate_analysis(row, paper)
                analyses[paper.paper_id] = analysis
                # If an abstract analysis was replaced after reading, replace its ERs too.
                evidence = {eid: e for eid, e in evidence.items() if e["paper_id"] != paper.paper_id}
                for claim in analysis["claims"]:
                    evidence[claim["evidence_id"]] = dict(
                        claim, paper_id=paper.paper_id, source_url=paper.url, version=paper.updated,
                        index_path=contexts[paper.paper_id]["index_path"],
                        locations=locate_quote(indexes[paper.paper_id], source_texts[paper.paper_id], claim["anchor_quote"]))
            material = [analyses[paper.paper_id] for paper in batch if analyses[paper.paper_id]["claims"]]
            for paper in batch:
                explanations.pop(paper.paper_id, None)
            if material:
                value = self.ledger.call(key + "-explanation-%d" % start, "literature_analyst",
                    explanation_prompt(material, profile), EXPLANATION_SCHEMA,
                    lambda value: validate_explanation_batch(value, material))
                explanations.update(validate_explanation_batch(value, material))
        return {"papers": papers, "paper_contexts": contexts, "analyses": analyses, "evidence": evidence,
                "paper_explanations": explanations,
                "artifacts": {"analysis": {"paper_ids": list(analyses)}}}

    def compare_methods(self, state, action, target, key):
        rows = []
        for pid in selected(state, action):
            analysis = state["analyses"][pid]
            rows.append({"paper_id": pid, "title": state["papers"][pid]["title"],
                         "source_url": analysis["source_url"],
                         **{kind: [{"text": c["text"], "evidence_id": c["evidence_id"]}
                                   for c in analysis["claims"] if c["kind"] == kind]
                            for kind in ("innovation", "method", "result", "limitation")},
                         "missing_evidence": analysis["missing_evidence"]})
        return {"artifacts": {"comparison": {"rows": rows, "comparability": "needs_info",
                 "reason": "结果可直接比较前需核对数据划分、输入尺寸、硬件、精度及评测协议。"}}}

    def assess_deployment(self, state, action, target, key):
        import json
        ids = selected(state, action)
        allowed = {c["evidence_id"] for pid in ids for c in state["analyses"][pid]["claims"]}
        prompt = (RULES + "\n你承担实验设计角色。针对用户提供的设备条件，逐篇给出部署候选评估。"
                  "reasoning 区分论文观察和部署推断，evidence_ids 只引用本次证据。"
                  "设备事实只来自用户提供的 facts，禁止凭记忆补充设备参数或实测速度。"
                  "candidate 表示进入上机验证，不能声称已适配；未测试算子、转换、显存、延迟须列入 missing_information。"
                  "verification_steps 写清导出、算子检查、数值一致性、目标设备热身、固定功耗/输入尺寸/精度、延迟分位数与峰值内存测量。\n"
                  + json.dumps({"goal": state["goal"], "facts": state["facts"],
                                "analyses": [state["analyses"][pid] for pid in ids]}, ensure_ascii=False))
        def check(value):
            actual = [r["paper_id"] for r in value["assessments"]]
            if set(actual) != set(ids) or len(actual) != len(ids):
                raise ValueError("部署评估必须覆盖所选论文且不能重复")
            for row in value["assessments"]:
                paper_evidence = {c["evidence_id"] for c in state["analyses"][row["paper_id"]]["claims"]}
                if not row["evidence_ids"] or not set(row["evidence_ids"]) <= (allowed & paper_evidence):
                    raise ValueError("部署评估引用了未校验或其他论文的证据")
                if not row["reasoning"].strip() or not row["verification_steps"]:
                    raise ValueError("部署评估需要推理说明和验证步骤")
        value = self.ledger.call(key + "-deployment", "experiment_designer", prompt + self.repair_feedback(state), DEPLOYMENT_SCHEMA, check)
        return {"artifacts": {"deployment": value}}

    def design_experiment(self, state, action, target, key):
        ids = selected(state, action)
        analyses = [state["analyses"][pid] for pid in ids]
        allowed = {c["evidence_id"] for a in analyses for c in a["claims"]}
        profile = self.profile(state)
        profile.update(task_goal=state["goal"], task_facts=state["facts"],
                       deployment_assessment=state["artifacts"].get("deployment"))
        def check(value):
            if not value["ideas"]:
                raise ValueError("没有可执行的候选实验；请补充证据或交付部分结果")
            for idea in value["ideas"]:
                validate_idea(idea, allowed, profile)
        value = self.ledger.call(key + "-experiment", "experiment_designer",
                                 idea_prompt(analyses, profile, []) + self.repair_feedback(state), IDEA_SCHEMA, check)
        return {"artifacts": {"experiment": [validate_idea(idea, allowed, profile) for idea in value["ideas"]]}}

    @staticmethod
    def repair_feedback(state):
        import json
        if not state.get("_repair_feedback"):
            return ""
        return ("\n请按以下独立审查反馈修复产物。保留问题所需的事实、限定条件和反证标准；"
                "仅修正错误引用、无依据的附加判断或明确表达不确定性，不能把实测断言伪装成实验假设。\n"
                + json.dumps(state["_repair_feedback"], ensure_ascii=False))

    def repair_result(self, state, action, target, key):
        section = action["query"]
        material = state if section in {"analysis", "comparison"} else approved_state(state)
        material = dict(material, _repair_feedback=state["gate"]["issues"])
        handler = {"analysis": self.analyze_papers, "comparison": self.analyze_papers,
                   "deployment": self.assess_deployment, "experiment": self.design_experiment}[section]
        result = handler(material, action, target, key)
        result["invalidate_artifacts"] = ["comparison", "deployment", "experiment"] if section in {"analysis", "comparison"} else (["experiment"] if section == "deployment" else [])
        return result
