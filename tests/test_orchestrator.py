import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scholar_daily.agents import SessionBackend
from scholar_daily.config import DEFAULT_CONFIG, load_config
from scholar_daily.contracts import ANALYSIS_SCHEMA, IDEA_SCHEMA, validate_analysis
from scholar_daily.models import Paper
from scholar_daily.orchestrator import run_research, task_status
from scholar_daily.research_calls import BudgetExceeded, CallLedger, read_json, write_json
from scholar_daily.research_state import DEPLOYMENT_SCHEMA, PLANNER_SCHEMA, checked_limits, planner_prompt
from scholar_daily.research_tools import ResearchTools
from scholar_daily.research_verifier import VERIFICATION_SCHEMA
from test_research import analysis, idea, paper, reading_response

HAS_GRAPH = importlib.util.find_spec("langgraph") is not None


def action(tool, **kwargs):
    value = {"tool": tool, "reason": "验收夹具：" + tool, "query": "", "days": 7, "paper_ids": [],
             "question": "", "required_fields": [], "finish_status": "partial"}
    value.update(kwargs)
    return value


def planned(tool, outputs, **kwargs):
    return {"plan": ["验收夹具的受控计划"], "deliverables": outputs, "next_action": action(tool, **kwargs)}


def passing_reviews(prompt):
    """Controlled audit responses test graph mechanics, not semantic accuracy."""
    payload = json.loads(prompt.split("材料：\n")[1])
    reviews = []
    for item in payload["items"]:
        factual = item["kind"] == "fact"
        sid = next((sid for sid in item["source_ids"] if sid in payload["sources"] and sid != "USER_FACTS"), "USER_FACTS")
        reviews.append({"item_id": item["item_id"], "coverage_complete": "yes",
                        "claims": [{"span": item["text"], "verdict": "supported" if factual else "not_factual",
                                    "qualifiers_supported": "yes", "citations": [{"source_id": sid, "quote": payload["sources"][sid]["text"][:200]}] if factual else [],
                                    "reason": "受控审查夹具"}],
                        "missing_information": [], "recommended_action": "none", "query": "", "required_fields": []})
    return {"reviews": reviews}


class ScriptedBackend:
    """Controlled planner responses isolate routing/recovery from model quality."""
    def __init__(self, plans, papers=(), fail_analysis=False, write_usage=False):
        self.plans = list(plans)
        self.papers = {p.paper_id: p for p in papers}
        self.calls = []
        self.fail_analysis = fail_analysis
        self.write_usage = write_usage

    def run(self, prompt, schema, folder):
        self.calls.append(schema)
        if self.write_usage:
            write_json(folder / "events.jsonl", {"type": "turn.completed", "usage": {"input_tokens": 200, "output_tokens": 100}})
        if schema == PLANNER_SCHEMA:
            if not self.plans:
                raise AssertionError("Unexpected planner call")
            return self.plans.pop(0)
        if schema == VERIFICATION_SCHEMA:
            return passing_reviews(prompt)
        response = reading_response(prompt, schema)
        if response is not None:
            return response
        if schema == ANALYSIS_SCHEMA:
            if self.fail_analysis:
                self.fail_analysis = False
                raise RuntimeError("临时模型连接失败")
            payload, _ = json.JSONDecoder().raw_decode(prompt.split("论文材料：\n")[1])
            return {"analyses": [analysis(Paper(**self.papers[row["paper_id"]].as_dict())) for row in payload]}
        if schema == DEPLOYMENT_SCHEMA:
            rows = [dict(paper_id=pid, evidence_ids=[validate_analysis(analysis(p), p)["claims"][0]["evidence_id"]],
                         assessment="candidate", reasoning="测试夹具推断：需要上机验证", verification_steps=["测试夹具：固定配置测量延迟"], missing_information=[])
                    for pid, p in self.papers.items()]
            return {"assessments": rows, "missing_information": []}
        if schema == IDEA_SCHEMA:
            p = next(iter(self.papers.values()))
            return {"ideas": [idea(validate_analysis(analysis(p), p)["claims"][0]["evidence_id"])]}
        raise AssertionError("Unexpected schema")


@unittest.skipUnless(HAS_GRAPH, "安装 .[orchestrator] 后运行 LangGraph 验收测试")
class OrchestratorTests(unittest.TestCase):
    def config(self, root, backend="codex-session"):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["agent"]["backend"] = backend
        config["research_profile"]["lab_context"]["datasets"] = ["test-data"]
        config["research_profile"]["compute"].update(gpus=[{"model": "Fixture", "count": 1, "available_vram_gb": 24}], max_experiment_hours=8)
        path = root / "config.json"
        write_json(path, config)
        return load_config(path)

    def fixtures(self):
        a, b = paper(), paper()
        b.paper_id = "fixture-b"
        b.title = "Fictional second test paper"
        return [a, b]

    def test_path_discovery_does_not_read_or_analyze(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            backend = ScriptedBackend([planned("search_papers", ["discovery"], query="lightweight detection"),
                                       planned("finish", ["discovery"], finish_status="complete")])
            search, reader = Mock(return_value=[p]), Mock()
            result = run_research(self.config(Path(d)), "最近一周论文候选列表", backend=backend, search=search, reader=reader)
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual([s["tool"] for s in state["steps"]], ["search_papers", "finish"])
            reader.assert_not_called()
            self.assertEqual(len(backend.calls), 2)
            search.assert_called_once_with("lightweight detection", [], 7, 40)

    def test_path_supplied_papers_compare_without_search(self):
        with tempfile.TemporaryDirectory() as d:
            papers = self.fixtures()
            backend = ScriptedBackend([planned("analyze_papers", ["comparison"]), planned("compare_methods", ["comparison"]),
                                       planned("finish", ["comparison"], finish_status="complete")], papers)
            search, reader = Mock(), Mock()
            result = run_research(self.config(Path(d)), "比较所提供论文的创新点", inputs={"papers": [p.as_dict() for p in papers]},
                                  backend=backend, search=search, reader=reader)
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual([s["tool"] for s in state["steps"]], ["analyze_papers", "compare_methods", "finish"])
            self.assertEqual(len(state["artifacts"]["comparison"]["rows"]), 2)
            self.assertEqual(state["artifacts"]["comparison"]["comparability"], "needs_info")
            search.assert_not_called()
            reader.assert_not_called()

    def test_path_deployment_pauses_for_facts_then_designs(self):
        with tempfile.TemporaryDirectory() as d:
            papers = self.fixtures()
            outputs = ["comparison", "deployment", "experiment"]
            backend = ScriptedBackend([planned("search_papers", outputs, query="detector"),
                                       planned("read_papers", outputs), planned("analyze_papers", outputs),
                                       planned("compare_methods", outputs),
                                       planned("ask_user", outputs, question="请补充设备和实验条件", required_fields=["deployment.device_variant", "deployment.memory_gb", "deployment.software_stack", "deployment.power_mode", "deployment.dataset", "deployment.input_shape", "deployment.target_latency_ms"]),
                                       planned("assess_deployment", outputs), planned("design_experiment", outputs),
                                       planned("finish", outputs, finish_status="complete")], papers)
            search = Mock(return_value=papers)
            result = run_research(self.config(Path(d)), "比较检测方法并设计目标设备实验", backend=backend, search=search,
                                  limits={"token_budget": 1000000})
            folder = Path(result["task_dir"])
            self.assertEqual(result["status"], "waiting_input")
            calls = result["budget"]["model_calls"]
            poll = run_research(folder=folder, backend=backend, search=search)
            self.assertEqual(poll["budget"]["model_calls"], calls)
            with self.assertRaisesRegex(ValueError, "提供 facts"):
                run_research(folder=folder, backend=backend, answer={"facts": {}})
            answer = {"facts": {"deployment": {"device_variant": "Fictional fixture device", "memory_gb": 24,
                      "software_stack": "Fixture SDK", "power_mode": "Fixture mode", "dataset": "test-data",
                      "input_shape": "fixture shape", "target_latency_ms": 50}}}
            result = run_research(folder=folder, backend=backend, search=search, answer=answer)
            state = task_status(folder)
            self.assertEqual(result["status"], "complete")
            self.assertEqual([s["tool"] for s in state["steps"]], ["search_papers", "read_papers", "analyze_papers", "compare_methods", "ask_user", "assess_deployment", "design_experiment", "finish"])
            self.assertEqual(state["artifacts"]["experiment"][0]["feasibility"], "candidate")
            search.assert_called_once()

    def test_failed_tool_retry_and_replan(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("search_papers", ["discovery"], query="detector"),
                                       planned("search_papers", ["discovery"], query="detector"),
                                       planned("finish", ["discovery"], finish_status="complete")])
            search = Mock(side_effect=[OSError("临时检索断连"), [paper()]])
            result = run_research(self.config(Path(d)), "查找论文", backend=backend, search=search)
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual([s["status"] for s in state["steps"]], ["failed", "completed", "completed"])
            self.assertFalse(state["issues"])
            self.assertEqual(search.call_count, 2)

    def test_crash_after_completed_analysis_recovers_without_repeating_calls(self):
        with tempfile.TemporaryDirectory() as d:
            papers = self.fixtures()
            backend = ScriptedBackend([planned("analyze_papers", ["comparison"]), planned("compare_methods", ["comparison"]),
                                       planned("finish", ["comparison"], finish_status="complete")], papers)
            # Crash after analysis has a durable receipt, before the node checkpoint.
            from scholar_daily.research_tools import ResearchTools
            execute = ResearchTools.execute
            fired = []
            def crash(tools, state, act):
                result = execute(tools, state, act)
                if act["tool"] == "analyze_papers" and not fired:
                    fired.append(True)
                    raise KeyboardInterrupt("fixture crash")
                return result
            with patch.object(ResearchTools, "execute", crash), self.assertRaises(KeyboardInterrupt):
                run_research(self.config(Path(d)), "比较两篇论文", inputs={"papers": [p.as_dict() for p in papers]}, backend=backend)
            folder = next((Path(d) / "reports/research-tasks").iterdir())
            result = run_research(folder=folder, backend=backend)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(sum(schema == ANALYSIS_SCHEMA for schema in backend.calls), 1)
            self.assertEqual(len(task_status(folder)["steps"]), 3)

    def test_session_resume_in_new_process_and_invalid_response_repair(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_research(self.config(Path(d)), "只发现论文")
            folder = Path(result["task_dir"])
            write_json(Path(result["pending"]["task"]["response_path"]), planned("ask_user", ["discovery"], question="提供检索词", required_fields=["query"]))
            child = subprocess.run([sys.executable, "-X", "utf8", "-m", "scholar_daily", "research-resume", "--task-dir", str(folder)],
                                   capture_output=True, text=True, encoding="utf-8", cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(child.returncode, 2, child.stderr)
            state = task_status(folder)
            self.assertEqual(state["status"], "waiting_input")
            result = run_research(folder=folder, answer={"facts": {"query": "fixture"}})
            self.assertEqual(result["status"], "waiting")
            count = result["budget"]["model_calls"]
            path = Path(result["pending"]["task"]["response_path"])
            path.write_text("{broken", encoding="utf-8")
            result = run_research(folder=folder)
            self.assertIn("响应校验失败", result["pending"]["task"]["validation_error"])
            self.assertEqual(result["budget"]["model_calls"], count)
            write_json(path, planned("finish", ["discovery"], reason="无材料，部分交付"))
            result = run_research(folder=folder)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["budget"]["model_calls"], count)

    def test_guard_blocks_unsupported_routes_and_premature_completion(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("compare_methods", ["comparison"]),
                                       planned("finish", ["comparison"], finish_status="complete"),
                                       planned("finish", ["comparison"], reason="需要补充全文证据")])
            result = run_research(self.config(Path(d)), "比较创新", inputs={"papers": [paper().as_dict()]}, backend=backend)
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "partial")
            self.assertEqual([s["status"] for s in state["steps"]], ["failed", "failed", "completed"])
            self.assertIn("尚未生成", state["steps"][1]["error"])

    def test_model_attempt_limit_and_budget_survive_resume(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("analyze_papers", ["analysis"])] * 3, [paper()], fail_analysis=True)
            # Exhaust calls after failed analysis; no unbounded retry or implicit spending.
            result = run_research(self.config(Path(d)), "分析论文", inputs={"papers": [paper().as_dict()]},
                                  backend=backend, limits={"max_model_calls": 2})
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["budget"]["model_calls"], 2)
            count = len(backend.calls)
            again = run_research(folder=Path(result["task_dir"]), backend=backend)
            self.assertEqual(again["budget"]["model_calls"], 2)
            self.assertEqual(len(backend.calls), count)

    def test_token_budget_prevents_unreserved_call(self):
        with tempfile.TemporaryDirectory() as d:
            backend = Mock()
            result = run_research(self.config(Path(d)), "查找论文", backend=backend, limits={"token_budget": 1})
            self.assertEqual(result["status"], "partial")
            backend.run.assert_not_called()
            self.assertEqual(result["budget"]["model_calls"], 0)

    def test_retry_exhaustion_and_repeated_success_are_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("search_papers", ["discovery"], query="fixture")] * 3)
            search = Mock(side_effect=OSError("断连"))
            result = run_research(self.config(Path(d)), "发现论文", backend=backend, search=search)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(search.call_count, 2)
            self.assertIn("重试次数", result["stop_reason"])

    def test_failed_step_waits_and_resumes_same_task(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("search_papers", ["discovery"], query="fixture"),
                                       planned("ask_user", ["discovery"], question="检索源断连，请确认连接恢复", required_fields=["network_ready"]),
                                       planned("search_papers", ["discovery"], query="fixture"),
                                       planned("finish", ["discovery"], finish_status="complete")])
            search = Mock(side_effect=[OSError("断连"), [paper()]])
            result = run_research(self.config(Path(d)), "发现论文", backend=backend, search=search)
            self.assertEqual(result["status"], "waiting_input")
            original = result["task_id"]
            result = run_research(folder=Path(result["task_dir"]), backend=backend, search=search,
                                  answer={"facts": {"network_ready": True}})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["task_id"], original)
            self.assertEqual(search.call_count, 2)

    def test_repeated_completed_search_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("search_papers", ["discovery"], query="fixture")] * 2
                                      + [planned("finish", ["discovery"], finish_status="complete")])
            search = Mock(return_value=[paper()])
            result = run_research(self.config(Path(d)), "发现论文", backend=backend, search=search)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(search.call_count, 1)
            state = task_status(Path(result["task_dir"]))
            self.assertIn("输入未变化", state["steps"][1]["error"])

    def test_missing_device_facts_block_deployment_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("analyze_papers", ["deployment"]),
                                       planned("assess_deployment", ["deployment"]),
                                       planned("finish", ["deployment"])], [paper()])
            result = run_research(self.config(Path(d)), "评估 Orin 部署", inputs={"papers": [paper().as_dict()]}, backend=backend)
            self.assertEqual(result["status"], "partial")
            self.assertNotIn(DEPLOYMENT_SCHEMA, backend.calls)

    def test_planner_cannot_drop_required_output_on_replan(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("search_papers", ["discovery", "comparison"], query="fixture"),
                                       planned("finish", ["discovery"], finish_status="complete"),
                                       planned("finish", ["discovery"], finish_status="complete")])
            result = run_research(self.config(Path(d)), "发现并比较论文", backend=backend, search=Mock(return_value=[paper()]))
            self.assertEqual(result["status"], "partial")
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(state["deliverables"], ["discovery", "comparison"])
            self.assertIn("不能静默缩减", state["steps"][-1]["error"])

    def test_fabricated_quotes_cannot_enter_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            papers = self.fixtures()
            backend = ScriptedBackend([planned("analyze_papers", ["comparison"]), planned("compare_methods", ["comparison"]), planned("finish", ["comparison"])], papers)
            papers[0].full_text = "This real fixture sentence contains no matching quotation."
            result = run_research(self.config(Path(d)), "比较论文", inputs={"papers": [p.as_dict() for p in papers]}, backend=backend)
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(state["analyses"][papers[0].paper_id]["claims"], [])
            self.assertNotIn("comparison", state["artifacts"])

    def test_planner_sees_module_evidence_and_resource_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            backend = ScriptedBackend([planned("analyze_papers", ["analysis"]), planned("finish", ["analysis"])], [paper()])
            result = run_research(self.config(Path(d)), "分析并保留缺口", inputs={"papers": [paper().as_dict()]}, backend=backend)
            state = task_status(Path(result["task_dir"]))
            state["artifacts"]["experiment"] = [dict(title="fixture", feasibility="needs_info", missing_resources=["dataset unavailable"])]
            payload = json.loads(planner_prompt(state, []).split("状态：\n")[1])
            self.assertEqual(payload["module_feedback"]["analysis_gaps"][paper().paper_id]["missing_evidence"], ["原始图表未分析"])
            self.assertEqual(payload["module_feedback"]["experiments"][0]["missing_resources"], ["dataset unavailable"])

    def test_focused_rereading_invalidates_stale_analysis_and_rejects_other_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = paper()
            text = root / "cache/papers/fixture.txt"
            text.parent.mkdir(parents=True)
            text.write_text(p.full_text + "\n\nThe ablation protocol controls batch size and measures latency on a fixed device.", encoding="utf-8")
            p.full_text_path = str(text)
            backend = ScriptedBackend([planned("analyze_papers", ["analysis"]), planned("read_papers", ["analysis"], query="ablation latency"),
                                       planned("finish", ["analysis"])], [p])
            result = run_research(self.config(root), "补读实验协议", inputs={"papers": [p.as_dict()]}, backend=backend)
            state = task_status(Path(result["task_dir"]))
            self.assertIn("ablation protocol", state["papers"][p.paper_id]["full_text"])
            self.assertEqual(state["analyses"], {})
            self.assertEqual(state["evidence"], {})
            self.assertNotIn("analysis", state["artifacts"])
            state["papers"][p.paper_id]["full_text_path"] = str(root / "private.txt")
            with self.assertRaisesRegex(ValueError, "不能读取其他文件"):
                ResearchTools(None).read_papers(state, action("read_papers", query="private"), root, "key")


class BudgetReceiptTests(unittest.TestCase):
    def test_roles_select_backend_without_changing_backend_interface(self):
        with tempfile.TemporaryDirectory() as d:
            reading, reasoning = Mock(), Mock()
            reading.run.return_value = planned("finish", ["discovery"])
            reasoning.run.return_value = planned("finish", ["discovery"])
            ledger = CallLedger(Path(d), reading, checked_limits(), reasoning_backend=reasoning)
            ledger.call("read", "literature_analyst", "fixture", PLANNER_SCHEMA)
            ledger.call("plan", "planner", "fixture", PLANNER_SCHEMA)
            ledger.call("design", "experiment_designer", "fixture", PLANNER_SCHEMA)
            self.assertEqual(reading.run.call_count, 1)
            self.assertEqual(reasoning.run.call_count, 2)
            ledger.close()

    def test_failed_attempt_unknown_usage_and_success_receipt_survive_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            backend = Mock()
            backend.run.side_effect = [RuntimeError("断连"), planned("finish", ["discovery"])]
            ledger = CallLedger(root, backend, checked_limits())
            with self.assertRaises(RuntimeError):
                ledger.call("same", "planner", "fixture", PLANNER_SCHEMA)
            self.assertEqual(ledger.budget()["unmeasured_calls"], 1)
            ledger.close()
            ledger = CallLedger(root, backend, checked_limits())
            ledger.call("same", "planner", "fixture", PLANNER_SCHEMA)
            ledger.close()
            ledger = CallLedger(root, backend, checked_limits())
            ledger.call("same", "planner", "fixture", PLANNER_SCHEMA)
            self.assertEqual(backend.run.call_count, 2)
            self.assertEqual(ledger.budget()["model_calls"], 2)
            ledger.close()

    def test_known_usage_replaces_estimate_but_unknown_is_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            class Metered:
                def run(self, prompt, schema, folder):
                    (folder / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 6}}), encoding="utf-8")
                    return planned("finish", ["discovery"])
            ledger = CallLedger(root, Metered(), checked_limits())
            ledger.call("metered", "planner", "fixture", PLANNER_SCHEMA)
            self.assertEqual(ledger.budget()["measured_tokens"], 10)
            self.assertEqual(ledger.budget()["charged_tokens"], 10)
            self.assertEqual(ledger.budget()["unmeasured_calls"], 0)
            ledger.close()


if __name__ == "__main__":
    unittest.main()
