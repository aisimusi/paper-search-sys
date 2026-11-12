import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

from scholar_daily.agents import SessionBackend
from scholar_daily.research_calls import read_json, write_json
from scholar_daily.research_state import checked_limits, initial_state, planner_prompt
from scholar_daily.research_tools import ResearchTools
from scholar_daily.skill_registry import catalog, load_file, load_skill, metadata, spec
from scholar_daily.skill_runner import prepare, resume_skill, run_skill, screening, skill_status
import test_orchestrator as fixture
from test_research import QUOTE, analysis, paper

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("paper-screening", "evidence-verification", "method-comparison", "experiment-design")


class SkillTests(unittest.TestCase):
    def config(self, root):
        return fixture.OrchestratorTests().config(root)

    def payload(self):
        p = paper()
        return {"papers": [p.as_dict()], "analyses": [analysis(p)]}

    def reviewer(self):
        return fixture.ScriptedBackend([])

    def test_discovery_reads_only_metadata(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("正文不得在发现时读取")):
            rows = metadata()
        self.assertEqual({row["name"] for row in rows}, set(NAMES))
        self.assertTrue(all(set(row) == {"name", "description", "version"} for row in rows))

    def test_resource_path_and_digest_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = spec("paper-screening")
            write_json(root / "catalog.json", {"api_version": 1, "skills": [row]})
            target = root / row["name"] / "SKILL.md"
            target.parent.mkdir()
            target.write_bytes((ROOT / "skills" / row["name"] / "SKILL.md").read_bytes())
            loaded = load_skill(row["name"], root)
            self.assertIn("scripts/run.py", loaded["resources"])
            with self.assertRaisesRegex(ValueError, "未声明"):
                load_file(row["name"], "../../config.json", root)
            target.write_text("altered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "指纹"):
                load_skill(row["name"], root)

    def test_catalog_sync_requires_version_bump_after_content_changes(self):
        from scripts.sync_skill_catalog import sync
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for row in catalog():
                for resource in row["files"]:
                    source = ROOT / "skills" / row["name"] / resource
                    target = root / "skills" / row["name"] / resource
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            sync(root)
            body = root / "skills/paper-screening/SKILL.md"
            body.write_text(body.read_text(encoding="utf-8") + "\n新增筛选规则。\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "提高版本"):
                sync(root)
            body.write_text(body.read_text(encoding="utf-8").replace('version: "0.1.0"', 'version: "0.1.1"'), encoding="utf-8")
            contract_file = root / "skills/paper-screening/references/contract.json"
            contract_data = read_json(contract_file)
            contract_data["version"] = "0.1.1"
            write_json(contract_file, contract_data)
            sync(root)
            self.assertEqual(catalog(root / "skills")[0]["version"], "0.1.1")

    def test_skill_name_rejects_embedded_resource_path(self):
        state = {"skill_catalog": catalog(), "loaded_skills": {}}
        with self.assertRaisesRegex(ValueError, "只填写技能名称"):
            ResearchTools.guard_skill(state, {"tool": "load_skill", "query": "paper-screening/SKILL.md"})

    def test_contract_validation_rejects_unknown_and_wrong_types(self):
        sample = read_json(ROOT / "skills/paper-screening/assets/example.json")
        for extra in ({"days": True}, {"days": 0}, {"surprise": "yes"}, {"queries": []}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                prepare("paper-screening", dict(sample, **extra))
        with self.assertRaisesRegex(ValueError, "analyses"):
            prepare("evidence-verification", {"papers": [paper().as_dict()]})
        with self.assertRaises(ValueError):
            prepare("experiment-design", {"goal": ""})

    def test_all_entrypoints_prepare_examples(self):
        for name in NAMES:
            result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "skills" / name / "scripts/run.py"),
                                     "prepare", "--input", str(ROOT / "skills" / name / "assets/example.json")],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["skill"], name)

    def test_screening_filters_dates_duplicates_and_limits(self):
        now = datetime.now(timezone.utc)
        a = paper()
        b, c, future, unknown = (copy.deepcopy(a) for _ in range(4))
        b.paper_id, b.updated = "old", (now - timedelta(days=20)).isoformat()
        c.paper_id = "other"
        future.paper_id, future.updated = "future", (now + timedelta(days=2)).isoformat()
        unknown.paper_id, unknown.updated = "unknown", "not a date"
        payload = {"papers": [p.as_dict() for p in (a, copy.deepcopy(a), b, c, future, unknown)],
                   "queries": ["Fictional"], "days": 7, "limit": 1}
        result = screening(payload, now)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(sorted(row["reason"] for row in result["excluded"]),
                         ["duplicate", "outside_window", "outside_window", "over_limit", "unknown_date"])
        self.assertEqual(result["score_kind"], "metadata_priority")

    def test_screening_has_no_model_calls_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            payload = {"papers": [paper().as_dict()], "queries": ["test"]}
            backend = Mock()
            result = run_skill("paper-screening", payload, folder=Path(d) / "screen", backend=backend)
            backend.run.assert_not_called()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(resume_skill("paper-screening", result["run_dir"]), result)
            self.assertEqual(skill_status("paper-screening", result["run_dir"])["input_hash"], result["input_hash"])

    def test_empty_candidates_are_an_honest_success(self):
        result = screening({"papers": [], "queries": ["test"]})
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["status"], "complete")

    def test_audit_pass_reuses_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            backend = self.reviewer()
            result = run_skill("evidence-verification", self.payload(), config=self.config(Path(d)), backend=backend)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["approved_evidence_ids"]), 1)
            calls = len(backend.calls)
            resumed = resume_skill("evidence-verification", result["run_dir"], backend=backend)
            self.assertEqual(resumed["budget"], result["budget"])
            self.assertEqual(len(backend.calls), calls)
            self.assertTrue(Path(result["result_path"]).is_file())

    def test_audit_rejects_semantically_unsupported_claims(self):
        def denied(prompt, schema, folder):
            response = fixture.passing_reviews(prompt)
            for row in response["reviews"]:
                row.update(recommended_action="read_more", missing_information=["设备与数字依据待补"])
                for claim in row["claims"]:
                    claim.update(verdict="unsupported", qualifiers_supported="no", reason="现有原文缺少设备与数字依据")
            return response
        with tempfile.TemporaryDirectory() as d:
            result = run_skill("evidence-verification", self.payload(), config=self.config(Path(d)), backend=Mock(run=denied))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["approved_evidence_ids"], [])
            self.assertEqual(result["gate"]["decision"], "read_more")

    def test_audit_preserves_invalid_candidates_and_avoids_model(self):
        with tempfile.TemporaryDirectory() as d:
            payload = self.payload()
            for row in payload["analyses"][0]["claims"]:
                row["anchor_quote"] = "This quote does not occur in the provided paper."
            backend = Mock()
            result = run_skill("evidence-verification", payload, config=self.config(Path(d)), backend=backend)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(len(result["verification"]["items"]), 3)
            self.assertTrue(all(row["structural_error"] for row in result["verification"]["items"]))
            backend.run.assert_not_called()

    def test_short_quote_and_empty_analysis_cannot_pass(self):
        for claims in ([{"kind": "result", "text": "候选结果", "anchor_quote": "The"}], []):
            with self.subTest(claims=claims), tempfile.TemporaryDirectory() as d:
                payload = self.payload()
                payload["analyses"][0]["claims"] = claims
                result = run_skill("evidence-verification", payload, config=self.config(Path(d)), backend=Mock())
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["approved_evidence_ids"], [])

    def test_session_audit_repairs_response_and_resumes_same_call(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_skill("evidence-verification", self.payload(), config=self.config(Path(d)), backend=SessionBackend())
            self.assertEqual(result["status"], "waiting")
            pending = result["pending"]["task"]
            write_json(Path(pending["response_path"]), {"reviews": []})
            result = resume_skill("evidence-verification", result["run_dir"], backend=SessionBackend())
            self.assertEqual(result["status"], "waiting")
            self.assertIn("完整覆盖", result["pending"]["task"]["validation_error"])
            self.assertEqual(result["budget"]["model_calls"], 1)
            prompt = Path(pending["prompt_path"]).read_text(encoding="utf-8")
            write_json(Path(pending["response_path"]), fixture.passing_reviews(prompt))
            result = resume_skill("evidence-verification", result["run_dir"], backend=SessionBackend())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["budget"]["model_calls"], 1)

    def test_budget_exhaustion_keeps_audit_partial(self):
        with tempfile.TemporaryDirectory() as d:
            payload = dict(self.payload(), limits={"token_budget": 1})
            backend = Mock()
            result = run_skill("evidence-verification", payload, config=self.config(Path(d)), backend=backend)
            self.assertEqual(result["status"], "partial")
            self.assertIn("预算", result["verification"]["error"])
            backend.run.assert_not_called()

    def test_resume_rejects_changed_version(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_skill("evidence-verification", self.payload(), config=self.config(Path(d)), backend=SessionBackend())
            with patch("scholar_daily.skill_runner.runtime_stamp", return_value={"changed": True}):
                with self.assertRaisesRegex(ValueError, "版本"):
                    resume_skill("evidence-verification", result["run_dir"], backend=SessionBackend())
            with self.assertRaisesRegex(ValueError, "不属于"):
                resume_skill("method-comparison", result["run_dir"], backend=SessionBackend())

    def test_cli_errors_have_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.json"
            write_json(path, {"papers": [], "queries": [], "unexpected": 1})
            result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "skills/paper-screening/scripts/run.py"),
                                     "run", "--input", str(path)], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stdout)["status"], "error")


@unittest.skipUnless(fixture.HAS_GRAPH, "需要项目编排依赖")
class SkillGraphTests(unittest.TestCase):
    config = SkillTests.config
    def source(self, root):
        papers = fixture.OrchestratorTests().fixtures()
        backend = fixture.ScriptedBackend([fixture.planned("analyze_papers", ["comparison"]),
                  fixture.planned("compare_methods", ["comparison"]), fixture.planned("finish", ["comparison"], finish_status="complete")], papers)
        from scholar_daily.orchestrator import run_research
        return run_research(self.config(root), "比较两篇虚构论文", inputs={"papers": [p.as_dict() for p in papers]}, backend=backend)

    def test_skill_and_resource_load_are_separate_graph_steps(self):
        with tempfile.TemporaryDirectory() as d:
            config = self.config(Path(d))
            plans = [fixture.planned("load_skill", ["discovery"], query="paper-screening"),
                     fixture.planned("load_skill_resource", ["discovery"], query="paper-screening/references/contract.json"),
                     fixture.planned("search_papers", ["discovery"], query="test"),
                     fixture.planned("finish", ["discovery"], finish_status="complete")]
            from scholar_daily.orchestrator import run_research, task_status
            result = run_research(config, "筛选候选", backend=fixture.ScriptedBackend(plans), search=Mock(return_value=[paper()]))
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(set(state["loaded_skills"]), {"paper-screening"})
            self.assertEqual(set(state["skill_resources"]), {"paper-screening/references/contract.json"})
            self.assertEqual(len(state["steps"]), 4)

    def test_unknown_skill_failure_can_replan_and_recover(self):
        with tempfile.TemporaryDirectory() as d:
            plans = [fixture.planned("load_skill", ["discovery"], query="unknown"),
                     fixture.planned("load_skill", ["discovery"], query="paper-screening"),
                     fixture.planned("search_papers", ["discovery"], query="test"),
                     fixture.planned("finish", ["discovery"], finish_status="complete")]
            from scholar_daily.orchestrator import run_research, task_status
            result = run_research(self.config(Path(d)), "筛选候选", backend=fixture.ScriptedBackend(plans), search=Mock(return_value=[paper()]))
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(state["steps"][0]["status"], "failed")
            self.assertEqual(state["steps"][1]["status"], "completed")

    def test_comparison_uses_valid_evidence_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.source(Path(d))
            original = (Path(source["task_dir"]) / "state.json").read_bytes()
            result = run_skill("method-comparison", {"task_dir": source["task_dir"]}, backend=fixture.ScriptedBackend([]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["comparison"]["rows"]), 2)
            self.assertEqual(result["comparison"]["comparability"], "needs_info")
            self.assertEqual((Path(source["task_dir"]) / "state.json").read_bytes(), original)
            self.assertNotEqual(result["run_dir"], source["task_dir"])
            self.assertEqual(result["origin_task_id"], source["task_id"])

    def test_comparison_requires_two_evidenced_papers(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.source(Path(d))
            with self.assertRaisesRegex(ValueError, "条目数量"):
                run_skill("method-comparison", {"task_dir": source["task_dir"], "paper_ids": ["fixture"]})

    def test_comparison_session_resumes_without_changing_source(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.source(Path(d))
            result = run_skill("method-comparison", {"task_dir": source["task_dir"]}, backend=SessionBackend())
            self.assertEqual(result["status"], "waiting")
            pending = result["pending"]["task"]
            prompt = Path(pending["prompt_path"]).read_text(encoding="utf-8")
            write_json(Path(pending["response_path"]), fixture.passing_reviews(prompt))
            result = resume_skill("method-comparison", result["run_dir"], backend=SessionBackend())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["budget"]["model_calls"], 1)

    def test_rejected_comparison_keeps_candidate_outside_approved_result(self):
        def denied(prompt, schema, folder):
            response = fixture.passing_reviews(prompt)
            for row in response["reviews"]:
                row["recommended_action"] = "repair"
                for claim in row["claims"]:
                    claim.update(verdict="unsupported", reason="受控失败夹具")
            return response
        with tempfile.TemporaryDirectory() as d:
            source = self.source(Path(d))
            result = run_skill("method-comparison", {"task_dir": source["task_dir"]}, backend=Mock(run=denied))
            self.assertEqual(result["status"], "partial")
            self.assertIsNone(result["comparison"])
            self.assertEqual(len(read_json(Path(result["candidate_comparison_path"]))["artifacts"]["comparison"]["rows"]), 2)

    def test_comparison_rejects_stale_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.source(Path(d))
            from scholar_daily.orchestrator import task_status
            state = task_status(Path(source["task_dir"]))
            next(iter(state["papers"].values()))["full_text"] += " Additional source content."
            with patch("scholar_daily.orchestrator.task_status", return_value=state):
                with self.assertRaisesRegex(ValueError, "独立 Verifier"):
                    run_skill("method-comparison", {"task_dir": source["task_dir"]}, backend=Mock())

    def test_experiment_skill_fixes_outputs_and_loads_only_its_body(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            plans = [fixture.planned("analyze_papers", ["experiment"]), fixture.planned("design_experiment", ["experiment"]),
                     fixture.planned("finish", ["experiment"], finish_status="complete")]
            result = run_skill("experiment-design", {"goal": "设计最小实验", "papers": [p.as_dict()]},
                               config=self.config(Path(d)), backend=fixture.ScriptedBackend(plans, [p]))
            from scholar_daily.orchestrator import task_status
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(state["deliverables"], ["experiment"])
            self.assertEqual(set(state["loaded_skills"]), {"experiment-design"})
            self.assertEqual(state["skill_resources"], {})
            self.assertEqual(skill_status("experiment-design", result["run_dir"])["status"], "complete")

    def test_experiment_crash_can_resume_through_its_skill_entry(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            backend = fixture.ScriptedBackend([fixture.planned("analyze_papers", ["experiment"]),
                      fixture.planned("design_experiment", ["experiment"]),
                      fixture.planned("finish", ["experiment"], finish_status="complete")], [p])
            execute = ResearchTools.execute
            interrupted = []
            def crash(tools, state, action):
                patch_value = execute(tools, state, action)
                if action["tool"] == "analyze_papers" and not interrupted:
                    interrupted.append(True)
                    raise KeyboardInterrupt("隔离中断夹具")
                return patch_value
            with patch.object(ResearchTools, "execute", crash), self.assertRaises(KeyboardInterrupt):
                run_skill("experiment-design", {"goal": "设计实验", "papers": [p.as_dict()]},
                          config=self.config(Path(d)), backend=backend)
            meta = next(Path(d).rglob("skill-run.json"))
            calls_before = len(backend.calls)
            result = resume_skill("experiment-design", meta.parent, backend=backend)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(sum(schema == fixture.ANALYSIS_SCHEMA for schema in backend.calls), 1)
            self.assertGreater(len(backend.calls), calls_before)

    def test_deployment_skill_resumes_after_user_supplies_facts(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            outputs = ["deployment", "experiment"]
            fields = ["deployment." + name for name in ("device_variant", "memory_gb", "software_stack", "power_mode", "dataset", "input_shape", "target_latency_ms")]
            backend = fixture.ScriptedBackend([fixture.planned("analyze_papers", outputs),
                      fixture.planned("ask_user", outputs, question="补充设备条件", required_fields=fields),
                      fixture.planned("assess_deployment", outputs), fixture.planned("design_experiment", outputs),
                      fixture.planned("finish", outputs, finish_status="complete")], [p])
            result = run_skill("experiment-design", {"goal": "设计部署实验", "deployment": True, "papers": [p.as_dict()],
                               "limits": {"token_budget": 1000000}},
                               config=self.config(Path(d)), backend=backend)
            self.assertEqual(result["status"], "waiting_input")
            facts = {"facts": {"deployment": {"device_variant": "Fixture", "memory_gb": 24, "software_stack": "Fixture SDK",
                      "power_mode": "Fixture", "dataset": "test-data", "input_shape": "Fixture shape", "target_latency_ms": 50}}}
            result = resume_skill("experiment-design", result["run_dir"], backend=backend, answer=facts)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(read_json(Path(result["state_path"]))["deliverables"], outputs)

    def test_planner_cannot_silently_drop_required_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_skill("experiment-design", {"goal": "设计实验"}, config=self.config(Path(d)), backend=SessionBackend())
            self.assertEqual(result["status"], "waiting")
            path = Path(result["pending"]["task"]["response_path"])
            write_json(path, fixture.planned("finish", ["discovery"], finish_status="partial"))
            result = resume_skill("experiment-design", result["run_dir"], backend=SessionBackend())
            self.assertEqual(result["status"], "waiting")
            self.assertIn("不能静默缩减", result["pending"]["task"]["validation_error"])

    def test_planner_metadata_does_not_eagerly_include_bodies(self):
        with tempfile.TemporaryDirectory() as d:
            state = initial_state("fixture", "筛选论文", self.config(Path(d)), {}, checked_limits())
            prompt = planner_prompt(state, ResearchTools(None).catalog())
            view = json.loads(prompt.split("状态：\n")[1])
            self.assertEqual(view["loaded_skills"], {})
            self.assertEqual(view["skill_resources"], {})
            self.assertEqual(len(view["available_skills"]), 4)


if __name__ == "__main__":
    unittest.main()
