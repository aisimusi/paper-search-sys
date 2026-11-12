import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scholar_daily.contracts import ANALYSIS_SCHEMA
from scholar_daily.orchestrator import run_research, task_status
from scholar_daily.quality_gate import evaluate_gate
from scholar_daily.research_calls import read_json, write_json
from scholar_daily.research_tools import ResearchTools
from scholar_daily.research_verifier import (VERIFICATION_SCHEMA, Verifier, approved_ids, approved_state,
                                            delivery_view, evidence_signature, review_inputs, validate_review)
import test_orchestrator as fixtures
from test_orchestrator import ScriptedBackend, action, passing_reviews, planned
from test_research import QUOTE, analysis, paper


def reviewed(item_id="item", span="The latency is 5 ms.", verdict="supported", sid="ER-1", quote="The latency is 5 ms."):
    return {"item_id": item_id, "coverage_complete": "yes",
            "claims": [{"span": span, "verdict": verdict, "qualifiers_supported": "yes",
                        "citations": [{"source_id": sid, "quote": quote}] if quote else [], "reason": "fixture review"}],
            "missing_information": [], "recommended_action": "read_more" if verdict != "supported" else "none",
            "query": "latency hardware", "required_fields": []}


class ReviewValidationTests(unittest.TestCase):
    def setUp(self):
        self.item = {"item_id": "item", "section": "evidence", "field": "claims.0", "text": "The latency is 5 ms.",
                     "kind": "fact", "evidence_ids": ["ER-1"], "source_ids": ["ER-1", "USER_FACTS"],
                     "paper_id": "p", "structural_error": ""}
        self.sources = {"ER-1": {"text": "The latency is 5 ms."}, "USER_FACTS": {"text": "The latency is 5 ms."}}

    def test_valid_support_requires_full_span_qualifiers_and_exact_quote(self):
        self.assertTrue(validate_review(reviewed(), self.item, self.sources)["passed"])

    def test_unknown_and_uncited_source_ids_are_rejected(self):
        for sid in ("invented", "ER-2"):
            self.sources["ER-2"] = {"text": "The latency is 5 ms."}
            self.assertFalse(validate_review(reviewed(sid=sid), self.item, self.sources)["passed"])

    def test_changed_number_unit_negation_and_spliced_quote_are_rejected(self):
        for quote in ("The latency is 6 ms.", "The latency is 5 s.", "The latency is not 5 ms.", "latency 5 ms"):
            self.assertFalse(validate_review(reviewed(quote=quote), self.item, self.sources)["passed"])

    def test_whitespace_only_matching_preserves_minus_and_words(self):
        self.sources["ER-1"]["text"] = "The\nlatency is 5 ms."
        self.assertTrue(validate_review(reviewed(), self.item, self.sources)["passed"])
        self.sources["ER-1"]["text"] = "The latency change is -5 ms."
        self.assertFalse(validate_review(reviewed(quote="The latency change is 5 ms."), self.item, self.sources)["passed"])

    def test_model_yes_cannot_override_missing_factual_coverage(self):
        self.item["text"] = "The latency is 5 ms on Orin using FP16."
        result = validate_review(reviewed(span="The latency is 5 ms"), self.item, self.sources)
        self.assertFalse(result["complete"])
        self.assertFalse(result["passed"])

    def test_unsupported_and_unsupported_qualifiers_fail_despite_valid_citation(self):
        self.assertFalse(validate_review(reviewed(verdict="unsupported"), self.item, self.sources)["passed"])
        review = reviewed()
        review["claims"][0]["qualifiers_supported"] = "no"
        self.assertFalse(validate_review(review, self.item, self.sources)["passed"])

    def test_fact_cannot_be_exempted_as_hypothesis_or_user_statement(self):
        self.assertFalse(validate_review(reviewed(verdict="not_factual", quote=""), self.item, self.sources)["passed"])
        self.assertFalse(validate_review(reviewed(sid="USER_FACTS"), self.item, self.sources)["passed"])
        review = reviewed()
        review["claims"][0]["citations"].append({"source_id": "USER_FACTS", "quote": self.item["text"]})
        self.assertFalse(validate_review(review, self.item, self.sources)["passed"])

    def test_explicit_experiment_proposal_can_pass_without_paper_claim(self):
        self.item.update(kind="proposal", section="experiment", text="We will measure latency at batch size one.")
        review = reviewed(span=self.item["text"], verdict="not_factual", quote="")
        self.assertTrue(validate_review(review, self.item, self.sources)["passed"])


class GatePolicyTests(unittest.TestCase):
    def state(self):
        return {"steps": [], "limits": {"max_steps": 20, "max_review_rounds": 3}, "gate_history": [],
                "deliverables": [], "artifacts": {}, "stop_reason": ""}

    def budget(self):
        return {"model_calls": 1, "max_model_calls": 30, "charged_tokens": 100, "token_budget": 10000}

    def report(self, action):
        return {"status": "checked", "input_hash": "fixture", "items": [
            {"item_id": "item", "section": "evidence", "field": "p.claims.0", "paper_id": "p", "passed": False,
             "complete": True, "claims": [{"passed": False, "reason": "fixture gap"}], "evidence_ids": ["ER-1"],
             "structural_error": "", "missing_information": ["hardware unknown"], "recommended_action": action,
             "query": "latency hardware", "required_fields": ["deployment.device_variant"]}]}

    def test_distinct_feedback_decisions_and_tool_suggestions(self):
        for decision, tool in (("repair", "repair_result"), ("read_more", "read_papers"), ("retrieve_more", "search_papers"), ("needs_input", "ask_user")):
            gate = evaluate_gate(self.state(), {"phase": "evidence"}, self.report(decision), self.budget())
            self.assertEqual(gate["decision"], decision)
            self.assertEqual(gate["suggestions"][0]["tool"], tool)
            self.assertEqual(gate["issues"][0]["field"], "p.claims.0")

    def test_round_limit_and_budget_failure_force_partial(self):
        state = self.state()
        state["limits"]["max_review_rounds"] = 1
        self.assertEqual(evaluate_gate(state, {"phase": "evidence"}, self.report("repair"), self.budget())["decision"], "partial")
        budget = dict(self.budget(), model_calls=30)
        self.assertEqual(evaluate_gate(self.state(), {"phase": "evidence"}, self.report("read_more"), budget)["decision"], "partial")
        self.assertEqual(evaluate_gate(self.state(), {"phase": "evidence"}, {"status": "unavailable", "error": "timeout"}, self.budget())["decision"], "partial")
        self.assertEqual(evaluate_gate(self.state(), {"phase": "final", "desired_status": "complete"}, {}, self.budget())["decision"], "partial")

    def test_complete_delivery_checks_missing_artifact_and_resource_unknowns(self):
        state = self.state()
        state["deliverables"] = ["comparison"]
        gate = evaluate_gate(state, {"phase": "final", "desired_status": "complete"}, {"status": "checked", "items": []}, self.budget())
        self.assertEqual(gate["decision"], "retrieve_more")
        state["deliverables"] = ["experiment"]
        state["artifacts"]["experiment"] = [{"feasibility": "needs_info", "missing_resources": ["missing dataset"]}]
        gate = evaluate_gate(state, {"phase": "final", "desired_status": "complete"}, {"status": "checked", "items": []}, self.budget())
        self.assertEqual(gate["decision"], "needs_input")
        state["artifacts"]["experiment"][0]["feasibility"] = "blocked"
        gate = evaluate_gate(state, {"phase": "final", "desired_status": "complete"}, {"status": "checked", "items": []}, self.budget())
        self.assertEqual(gate["decision"], "repair")


class CorrectingBackend(ScriptedBackend):
    def __init__(self, plans, papers):
        super().__init__(plans, papers)
        self.analysis_count = 0
        self.audit_phases = []
        self.repair_prompts = []

    def run(self, prompt, schema, folder):
        if schema == ANALYSIS_SCHEMA:
            self.analysis_count += 1
            payload, _ = json.JSONDecoder().raw_decode(prompt.split("论文材料：\n")[1])
            rows = []
            for row in payload:
                p = self.papers[row["paper_id"]]
                value = analysis(p)
                value["claims"] = [dict(value["claims"][0], kind="method", text="该方法已经在 Orin 上达到 100 FPS" if self.analysis_count == 1 else "受控实验比较自适应选择方法和固定计算基线")]
                rows.append(value)
            if self.analysis_count > 1:
                self.repair_prompts.append(prompt)
            self.calls.append(schema)
            return {"analyses": rows}
        if schema == VERIFICATION_SCHEMA:
            payload = json.loads(prompt.split("材料：\n")[1])
            self.audit_phases.append(payload["phase"])
            response = passing_reviews(prompt)
            for item, review in zip(payload["items"], response["reviews"]):
                if "100 FPS" in item["text"]:
                    review["claims"][0].update(verdict="unsupported", qualifiers_supported="no", citations=[], reason="原文没有 Orin 上的帧率实测，硬件条件与数字没有依据")
                    review.update(recommended_action="repair", missing_information=["Orin 设备实测"])
            self.calls.append(schema)
            return response
        return super().run(prompt, schema, folder)


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "安装 .[orchestrator] 后运行")
class ReviewGraphTests(unittest.TestCase):
    def config(self, root):
        return fixtures.OrchestratorTests().config(root)

    def test_repair_loop_reaudits_and_does_not_reuse_rejected_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            papers = fixtures.OrchestratorTests().fixtures()
            backend = CorrectingBackend([planned("analyze_papers", ["comparison"]),
                                         planned("repair_result", ["comparison"], query="analysis"),
                                         planned("compare_methods", ["comparison"]),
                                         planned("finish", ["comparison"], finish_status="complete")], papers)
            result = run_research(self.config(Path(d)), "比较已提供方法证据", backend=backend,
                                  inputs={"papers": [p.as_dict() for p in papers]}, limits={"token_budget": 1000000})
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(state["gate_history"][0]["decision"], "repair")
            self.assertEqual(state["gate_history"][1]["decision"], "pass")
            self.assertEqual(backend.analysis_count, 2)
            self.assertEqual(sum(r["phase"] == "evidence" for r in state["review_history"]), 2)
            self.assertIn("原文没有 Orin", backend.repair_prompts[0])
            self.assertTrue(approved_ids(state))
            self.assertEqual([s["tool"] for s in state["steps"]], ["analyze_papers", "repair_result", "compare_methods", "finish"])
            self.assertNotIn("100 FPS", Path(result["report_path"]).read_text(encoding="utf-8"))

    def test_rejected_evidence_blocks_research_and_partial_report_omits_assertion(self):
        with tempfile.TemporaryDirectory() as d:
            papers = fixtures.OrchestratorTests().fixtures()
            backend = CorrectingBackend([planned("analyze_papers", ["comparison"]),
                                         planned("compare_methods", ["comparison"]), planned("finish", ["comparison"])], papers)
            result = run_research(self.config(Path(d)), "比较方法", backend=backend, inputs={"papers": [p.as_dict() for p in papers]})
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "partial")
            self.assertFalse(approved_ids(state))
            self.assertNotIn("comparison", state["artifacts"])
            self.assertIn("独立 Verifier", state["steps"][1]["error"])
            self.assertEqual(delivery_view(state)["evidence"], {})

    def test_final_review_can_reject_previously_accepted_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            papers = fixtures.OrchestratorTests().fixtures()
            class FinalReject(ScriptedBackend):
                def run(self, prompt, schema, folder):
                    if schema == VERIFICATION_SCHEMA:
                        payload = json.loads(prompt.split("材料：\n")[1])
                        result = passing_reviews(prompt)
                        if payload["phase"] == "final":
                            for item, review in zip(payload["items"], result["reviews"]):
                                review["claims"][0].update(verdict="unsupported", reason="最终比较缺少必要实验条件")
                                review.update(recommended_action="read_more")
                        return result
                    return super().run(prompt, schema, folder)
            backend = FinalReject([planned("analyze_papers", ["comparison"]), planned("compare_methods", ["comparison"]),
                                   planned("finish", ["comparison"], finish_status="complete"), planned("finish", ["comparison"])], papers)
            result = run_research(self.config(Path(d)), "比较方法", backend=backend, inputs={"papers": [p.as_dict() for p in papers]}, limits={"token_budget": 1000000})
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(state["steps"][-2]["status"], "rejected")
            self.assertNotIn("## 方法比较", Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(state["gate_history"][-2]["decision"], "read_more")

    def test_review_signature_is_invalidated_by_new_text_or_facts(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            backend = ScriptedBackend([planned("analyze_papers", ["analysis"]), planned("finish", ["analysis"], finish_status="complete")], [p])
            result = run_research(self.config(Path(d)), "分析材料", backend=backend, inputs={"papers": [p.as_dict()]})
            original = task_status(Path(result["task_dir"]))
            self.assertTrue(approved_ids(original))
            for mutation in ("text", "facts"):
                state = copy.deepcopy(original)
                if mutation == "text":
                    state["papers"][p.paper_id]["full_text"] += " Changed conditions."
                else:
                    state["facts"]["new_device"] = "other"
                self.assertFalse(approved_ids(state))
                with self.assertRaisesRegex(ValueError, "独立 Verifier"):
                    ResearchTools(None).guard_deployment(state, action("assess_deployment"))

    def test_crash_after_review_receipt_recovers_without_duplicate_verifier_calls(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            backend = CorrectingBackend([planned("analyze_papers", ["analysis"]), planned("finish", ["analysis"], finish_status="complete")], [p])
            backend.analysis_count = 1  # Start with a supported reading.
            original = Verifier.review
            crashed = []
            def interrupt(reviewer, state, request):
                result = original(reviewer, state, request)
                if request["phase"] == "evidence" and not crashed:
                    crashed.append(True)
                    raise KeyboardInterrupt("fixture crash after review receipt")
                return result
            with patch.object(Verifier, "review", interrupt), self.assertRaises(KeyboardInterrupt):
                run_research(self.config(Path(d)), "分析材料", backend=backend, inputs={"papers": [p.as_dict()]})
            folder = next((Path(d) / "reports/research-tasks").iterdir())
            result = run_research(folder=folder, backend=backend)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(backend.audit_phases.count("evidence"), 1)

    def test_session_verifier_response_repair_and_new_process_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            result = run_research(self.config(Path(d)), "分析材料", inputs={"papers": [p.as_dict()]})
            folder = Path(result["task_dir"])
            write_json(Path(result["pending"]["task"]["response_path"]), planned("analyze_papers", ["analysis"]))
            result = run_research(folder=folder)
            write_json(Path(result["pending"]["task"]["response_path"]), {"analyses": [analysis(p)]})
            result = run_research(folder=folder)
            from test_research import reading_response
            task = result["pending"]["task"]
            write_json(Path(task["response_path"]), reading_response(Path(task["prompt_path"]).read_text(encoding="utf-8"),
                                                                     read_json(Path(task["schema_path"]))))
            result = run_research(folder=folder)
            self.assertEqual(result["pending"]["resume_node"], "verifier")
            count = result["budget"]["model_calls"]
            path = Path(result["pending"]["task"]["response_path"])
            write_json(path, {"reviews": []})
            result = run_research(folder=folder)
            self.assertIn("完整覆盖", result["pending"]["task"]["validation_error"])
            self.assertEqual(result["budget"]["model_calls"], count)
            prompt = Path(result["pending"]["task"]["prompt_path"]).read_text(encoding="utf-8")
            write_json(path, passing_reviews(prompt))
            child = subprocess.run([sys.executable, "-X", "utf8", "-m", "scholar_daily", "research-resume", "--task-dir", str(folder)],
                                   capture_output=True, text=True, encoding="utf-8", cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(child.returncode, 2, child.stderr)
            state = task_status(folder)
            self.assertEqual(state["pending"]["resume_node"], "planner")
            self.assertTrue(approved_ids(state))

    def test_review_budget_exhaustion_cannot_publish_unreviewed_claims(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            backend = CorrectingBackend([planned("analyze_papers", ["analysis"])], [p])
            result = run_research(self.config(Path(d)), "分析材料", backend=backend, inputs={"papers": [p.as_dict()]}, limits={"max_model_calls": 2})
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["budget"]["model_calls"], 2)
            self.assertEqual(state["gate"]["decision"], "partial")
            self.assertFalse(approved_ids(state))
            self.assertNotIn("100 FPS", Path(result["report_path"]).read_text(encoding="utf-8"))

    def test_repair_action_creates_fresh_review_even_for_unchanged_reader_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = paper()
            class CitationFix(ScriptedBackend):
                attempts = 0
                def run(self, prompt, schema, folder):
                    if schema == VERIFICATION_SCHEMA:
                        payload = json.loads(prompt.split("材料：\n")[1])
                        result = passing_reviews(prompt)
                        if payload["phase"] == "evidence":
                            self.attempts += 1
                            if self.attempts == 1:
                                for review in result["reviews"]:
                                    review["claims"][0]["citations"][0]["source_id"] = "invented"
                        return result
                    return super().run(prompt, schema, folder)
            backend = CitationFix([planned("analyze_papers", ["analysis"]),
                                   planned("repair_result", ["analysis"], query="analysis"),
                                   planned("finish", ["analysis"], finish_status="complete")], [p])
            result = run_research(self.config(Path(d)), "分析材料", backend=backend, inputs={"papers": [p.as_dict()]})
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(backend.attempts, 2)
            self.assertEqual(state["gate_history"][0]["decision"], "repair")

    def test_research_receives_approved_subset_without_rejected_claims(self):
        with tempfile.TemporaryDirectory() as d:
            papers = fixtures.OrchestratorTests().fixtures()
            extra = "The evaluation records memory consumption using batch size one."
            for p in papers:
                p.full_text += "\n" + extra
            class Mixed(CorrectingBackend):
                def run(self, prompt, schema, folder):
                    if schema == ANALYSIS_SCHEMA:
                        result = super().run(prompt, schema, folder)
                        for row in result["analyses"]:
                            row["claims"].append({"kind": "result", "text": "评估使用 batch size one 记录内存消耗", "anchor_quote": extra})
                        return result
                    return super().run(prompt, schema, folder)
            backend = Mixed([planned("analyze_papers", ["comparison"]), planned("compare_methods", ["comparison"]),
                             planned("finish", ["comparison"], finish_status="complete")], papers)
            result = run_research(self.config(Path(d)), "比较已提供且通过审查的证据，保留其余缺口", backend=backend,
                                  inputs={"papers": [p.as_dict() for p in papers]})
            state = task_status(Path(result["task_dir"]))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(approved_ids(state)), 2)
            rows = state["artifacts"]["comparison"]["rows"]
            self.assertTrue(all(not row["method"] and row["result"] for row in rows))
            self.assertTrue(all("100 FPS" not in str(row) for row in rows))


if __name__ == "__main__":
    unittest.main()
