import copy
import os
import subprocess
import sys
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scholar_daily.agents import parse_response
from scholar_daily.benchmark import digest, read_json, write_json
from scholar_daily.benchmark_agent import (ARMS, Caller, REPAIR_INSTRUCTION, check_frozen,
    execute_batch, freeze, implementation_hashes, reuse_initial_calls, run, score)
from scholar_daily.benchmark_holdout import prepare_development, reserve
from scholar_daily.benchmark_support import validate_review
from tests.test_benchmark_agent import FakeBackend, policy, review
from tests import test_benchmark_agent as fixtures


def module_policy():
    p = policy()
    p.update({"arms": list(ARMS), "initial_k": {arm: 2 for arm in ARMS},
              "max_repair_rounds": 1, "max_model_calls_per_batch": 16})
    return p


class CoverageRegressionTests(unittest.TestCase):
    def test_three_recorded_development_false_rejections(self):
        cases = read_json(Path(__file__).parent / "fixtures/qasper_coverage_regressions.json")
        self.assertEqual(len(cases), 3)
        for case in cases:
            with self.subTest(question_id=case["question_id"]):
                self.assertTrue(validate_review(case["review"], case["answer"], case["cited"])["passed"])

    def test_bm25_scores_and_prompts_match_across_process_hash_seeds(self):
        script = ("import json; from scholar_daily.benchmark import retrieve; "
                  "chunks=[{'id':'p0','text':'alpha beta gamma alpha'}, {'id':'p1','text':'beta delta'}, "
                  "{'id':'p2','text':'gamma epsilon'}]; "
                  "print(json.dumps(retrieve('alpha beta gamma delta epsilon',chunks,3)))")
        outputs = [subprocess.check_output([sys.executable, "-c", script],
                                           env=dict(os.environ, PYTHONHASHSEED=str(seed)), text=True)
                   for seed in (1, 2, 3, 7, 19)]
        self.assertEqual(len(set(outputs)), 1)

    def test_material_facts_and_qualifiers_remain_required(self):
        cited = [{"id": "s0:p1", "text": "Target result: 42"}]
        for answer in ("42 and 17", "42%", "42 ms", "-42", "+42", "not 42", "only 42", "at least 42",
                       "at most 42", "up to 42", "approximately 42", "42 on test", "42 or 17", "42–50", "≤42", "42/s"):
            with self.subTest(answer=answer):
                self.assertFalse(validate_review(review(), answer, cited)["passed"])


class RepairBackend(FakeBackend):
    def __init__(self, extra_claim=False, fail_repair=False):
        super().__init__()
        self.extra_claim, self.fail_repair = extra_claim, fail_repair
        self.repairs = 0

    def run(self, prompt, schema, folder):
        result = super().run(prompt, schema, folder)
        if "answers" in result:
            repairing = prompt.startswith(REPAIR_INSTRUCTION)
            if repairing:
                self.repairs += 1
            for value in result["answers"]:
                value["evidence_ids"] = ["s0:p0" if not repairing or self.fail_repair else "s0:p1"]
                value["answer"] = "42 and 17" if self.extra_claim and not repairing else "42"
        return parse_response(json.dumps(result), schema)


class ModuleLoopTests(unittest.TestCase):
    def execute(self, arm, backend=None, p=None):
        p, backend = p or module_policy(), backend or RepairBackend()
        with tempfile.TemporaryDirectory() as directory:
            caller = Caller(backend, Path(directory), p)
            state = execute_batch([fixtures.LoopTests().item()], arm, p, caller, "b")["original-id"]
            records = [read_json(f) for f in Path(directory).glob("calls/**/record.json")]
            return state, backend, records

    def test_gate_abstains_without_repair_or_search(self):
        state, backend, _ = self.execute("rag20_gate")
        self.assertEqual(state["prediction"]["answer"], "Unanswerable")
        self.assertEqual(backend.calls, ["answers", "reviews"])
        self.assertEqual(state["repair_rounds"], 0)

    def test_repair_citation_without_new_context_then_reaudit(self):
        state, backend, _ = self.execute("rag20_repair")
        trace = state["history"][0]["repairs"][0]
        self.assertEqual(trace["before"]["answer"], trace["after"]["answer"])
        self.assertEqual(trace["before"]["evidence_ids"], ["s0:p0"])
        self.assertEqual(trace["after"]["evidence_ids"], ["s0:p1"])
        self.assertFalse(trace["before_review"]["passed"])
        self.assertTrue(trace["after_review"]["passed"])
        self.assertEqual(trace["context_ids"], state["history"][0]["context_ids"])
        self.assertEqual(state["rounds"], 1)
        self.assertTrue(state["final_review"]["passed"])
        self.assertEqual(backend.calls, ["answers", "reviews", "answers", "reviews", "reviews"])

    def test_repair_removes_optional_unsupported_claim(self):
        state, _, _ = self.execute("rag20_repair", RepairBackend(extra_claim=True))
        self.assertEqual(state["initial_proposal"]["answer"], "42 and 17")
        self.assertEqual(state["prediction"]["answer"], "42")
        self.assertTrue(state["final_review"]["passed"])

    def test_dynamic_agent_stops_after_supported_repair(self):
        state, backend, _ = self.execute("agent")
        self.assertEqual(state["stop_reason"], "support_passed")
        self.assertNotIn("actions", backend.calls)

    def test_unrepaired_answer_is_rejected_and_rounds_are_bounded(self):
        state, backend, records = self.execute("rag20_repair", RepairBackend(fail_repair=True))
        self.assertEqual(state["prediction"]["answer"], "Unanswerable")
        self.assertEqual(state["repair_rounds"], 1)
        self.assertEqual(backend.repairs, 1)
        self.assertLessEqual(len(records), 16)

    def test_call_budget_skips_repair_and_keeps_final_audit_capacity(self):
        p = module_policy()
        p["max_model_calls_per_batch"] = 6
        state, backend, records = self.execute("agent", p=p)
        self.assertEqual(state["prediction"]["answer"], "Unanswerable")
        self.assertEqual(state["repair_rounds"], 0)
        self.assertEqual(state["stop_reason"], "call_budget")
        self.assertEqual(state["history"][0]["repair_stop_reason"], "call_budget")
        self.assertLessEqual(len(records), 6)

    def test_repair_cache_survives_resume(self):
        backend, p = RepairBackend(), module_policy()
        with tempfile.TemporaryDirectory() as directory:
            caller = Caller(backend, Path(directory), p)
            batch = [fixtures.LoopTests().item()]
            first = execute_batch(batch, "rag20_repair", p, caller, "b")
            second = execute_batch(batch, "rag20_repair", p, caller, "b")
            self.assertEqual(first, second)
            self.assertEqual(backend.repairs, 1)

    def test_resume_at_exact_budget_does_not_change_answer(self):
        backend, p = RepairBackend(), module_policy()
        p["max_model_calls_per_batch"] = 5
        p["max_call_attempts"] = 1
        with tempfile.TemporaryDirectory() as directory:
            caller = Caller(backend, Path(directory), p)
            batch = [fixtures.LoopTests().item()]
            first = execute_batch(batch, "rag20_repair", p, caller, "b")
            second = execute_batch(batch, "rag20_repair", p, caller, "b")
            self.assertEqual(first, second)
            self.assertEqual(len(backend.calls), 5)
            self.assertTrue(first["original-id"]["final_review"]["passed"])

    def test_shared_proposal_is_reused_and_marked_in_cost_ledger(self):
        backend, p = RepairBackend(), module_policy()
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            write_json(study / "policy.json", p)
            for arm in ARMS:
                root = study / "runs/dev" / arm
                write_json(root / "run-config.json", {"policy_sha256": "p", "implementation": {}, "input_sha256": "i", "split": "dev"})
                reuse_initial_calls(study, "dev", arm, "b", root)
                states = execute_batch([fixtures.LoopTests().item()], arm, p, Caller(backend, root, p), "b")
                write_json(root / "batches/b/result.json", {"errors": {}, "states": states})
            self.assertEqual(backend.calls.count("answers"), 2)
            records = list((root / "calls/b/round-0/answer").glob("*/record.json"))
            self.assertIn("reused_from", read_json(records[0]))
            self.assertEqual(backend.repairs, 1)
            repair_records = list((root / "calls/b/round-0/repair-0/repair").glob("*/record.json"))
            self.assertIn("reused_from", read_json(repair_records[0]))


class ReportIntegrationTests(unittest.TestCase):
    def test_four_arms_report_module_effects_and_deduplicated_cost(self):
        from scholar_daily.benchmark_agent_report import report
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            p = module_policy()
            p.update({"model": {"backend": "fixture"}, "batch_size": 5,
                      "acceptance": {"quality": "agent minus repair paired lower bound > 0"}})
            item = fixtures.LoopTests().item()
            write_json(study / "prior-costs.json", {"calls": 2, "usage": {"input_tokens": 11}, "calls_with_usage": 1})
            p["prior_attempts"] = [{"path": str(study / "prior-costs.json"), "sha256": digest((study / "prior-costs.json").read_bytes())}]
            write_json(study / "policy.json", p)
            write_json(study / "dev/inputs.json", [item])
            write_json(study / "dev/gold.json", {})
            write_json(study / "reservation.json", {"data_dir": directory,
                       "hashes": {"dev": {k: digest((study / "dev" / f"{k}.json").read_bytes()) for k in ("gold", "inputs")}}})
            references = {"original-id": [{"answer": "42", "type": "abstractive", "evidence": ["Target result: 42"]}]}
            def evaluate(gold, predictions):
                prediction = predictions.get("original-id", {})
                return {"Answer F1": float(prediction.get("answer") == "42"),
                        "Evidence F1": float(prediction.get("evidence") == ["Target result: 42"])}
            evaluator = SimpleNamespace(get_answers_and_evidence=lambda *args: references,
                                        token_f1_score=lambda a, b: float(a == b), evaluate=evaluate)
            def cli_output(command, **kwargs):
                predictions = {v["question_id"]: {"answer": v["predicted_answer"], "evidence": v["predicted_evidence"]}
                               for v in (json.loads(line) for line in Path(command[-1]).read_text(encoding="utf-8").splitlines())}
                return json.dumps(evaluate(references, predictions))
            backend = RepairBackend()
            with patch("scholar_daily.benchmark_agent.get_backend", return_value=backend), \
                 patch("scholar_daily.benchmark_agent.load_evaluator", return_value=evaluator), \
                 patch("scholar_daily.benchmark_agent_report.subprocess.check_output", side_effect=cli_output):
                for arm in ARMS:
                    run(study, "dev", arm)
                result = report(study, "dev")
                self.assertEqual(result["actual_experiment_cost"]["calls"], len(backend.calls))
                self.assertEqual(result["actual_experiment_cost"]["usage"]["input_tokens"], 10 * len(backend.calls))
                self.assertEqual(len(result["module_effects"]), 4)
                self.assertEqual(result["study_actual_cost"]["calls"], len(backend.calls) + 2)
                self.assertFalse(result["study_actual_cost"]["usage_complete"])
                self.assertEqual(result["module_effects"]["rag20_repair_minus_rag20_gate"]["supported_answer"]["mean_delta"], 1)
                self.assertEqual(result["runs"]["rag20_repair"]["supported_positive_answers"], 1)
                self.assertEqual(result["runs"]["rag20_repair"]["reused_calls"], 2)
                self.assertTrue(all(result["official_cli_checks"].values()))
                # Missing usage in an earlier split must affect the cumulative ledger.
                p.pop("prior_attempts")
                write_json(study / "policy.json", p)
                for arm in ARMS:
                    configuration = read_json(study / "runs/dev" / arm / "run-config.json")
                    configuration["policy_sha256"] = digest((study / "policy.json").read_bytes())
                    write_json(study / "runs/dev" / arm / "run-config.json", configuration)
                previous = dict(result["runs"]["rag20"], calls_with_usage=0)
                write_json(study / "runs/previous/rag20/metrics.json", previous)
                cumulative = report(study, "dev")
                self.assertTrue(cumulative["acceptance"]["cost_usage_complete"])
                self.assertFalse(cumulative["study_actual_cost"]["usage_complete"])
                # Multiple completed turns in one call must all contribute to usage.
                record_path = study / "runs/dev/rag20/calls/0000/round-0/answer/attempt-0/record.json"
                record = read_json(record_path)
                record["usage"].append({"input_tokens": 7, "output_tokens": 3})
                write_json(record_path, record)
                metrics = score(study, "dev", "rag20")
                self.assertEqual(metrics["actual_usage"]["input_tokens"], 27)


class FreezeReservationTests(unittest.TestCase):
    def test_freeze_rejects_changed_cli_before_releasing_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory) / "study"
            cli = Path(directory) / ".codex/config.toml"
            cli.parent.mkdir()
            cli.write_text('model_reasoning_effort = "xhigh"', encoding="utf-8")
            write_json(study / "policy.json", {"model": {"backend": "codex",
                       "cli_config_sha256": digest(b'model_reasoning_effort = "high"')}})
            with patch("scholar_daily.benchmark_agent.Path.home", return_value=Path(directory)):
                with self.assertRaisesRegex(ValueError, "Model configuration changed"):
                    freeze(study)
            self.assertFalse((study / "freeze.json").exists())
            self.assertFalse((study / "development-reservation.json").exists())

    def dev_study(self, study, data):
        p = module_policy()
        p["acceptance"] = {"quality": "positive paired lower bound"}
        write_json(study / "policy.json", p)
        write_json(study / "dev/gold.json", {})
        write_json(study / "dev/inputs.json", [])
        reservation = {"data_dir": str(data), "hashes": {"dev": {key: digest((study / "dev" / f"{key}.json").read_bytes()) for key in ("gold", "inputs")}}}
        write_json(study / "reservation.json", reservation)
        for arm in ARMS:
            root = study / "runs/dev" / arm
            write_json(root / "metrics.json", {"failed": 0, "pending": 0})
            write_json(root / "diagnostics.json", [])
            (root / "predictions.jsonl").write_text("", encoding="utf-8")
            write_json(root / "run-config.json", {"policy_sha256": digest((study / "policy.json").read_bytes()), "implementation": implementation_hashes()})

    def test_prepare_dev_does_not_reserve_or_read_test(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory) / "study"
            args = SimpleNamespace(study=study, dev_source=Path(directory))
            with patch("scholar_daily.benchmark_holdout.checked_snapshot", return_value=({"data_dir": directory}, {}, [])):
                prepare_development(args)
            self.assertNotIn("holdout", read_json(study / "reservation.json")["hashes"])
            self.assertFalse((study / "holdout").exists())

    def test_reservation_requires_freeze_and_excludes_prior_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory) / "study"
            data = Path(directory) / "data"
            data.mkdir()
            self.dev_study(study, data)
            args = SimpleNamespace(study=study, exclude_root=Path(directory), count=1, seed=1)
            with self.assertRaises(FileNotFoundError):
                reserve(args)
            from tests.test_benchmark_agent import HoldoutTests
            paper = HoldoutTests().paper
            old = paper("Old", "old text", "old question", "q0")
            fresh = paper("New", "new text", "new question", "q1")
            write_json(Path(directory) / "old/holdout/gold.json", {"old": old})
            write_json(data / "qasper-test-v0.3.json", {"old": old, "fresh": fresh})
            freeze(study)
            reserve(args)
            self.assertEqual(read_json(study / "reservation.json")["holdout_question_ids"], ["q1"])
            check_frozen(study)
            with self.assertRaises(ValueError):
                reserve(args)
            write_json(study / "runs/dev/rag20/metrics.json", {"changed": True})
            with self.assertRaises(ValueError):
                check_frozen(study)

    def test_changed_reservation_seal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            self.dev_study(study, study)
            freeze(study)
            write_json(study / "reservation.json", {"tampered": True})
            with self.assertRaises(FileNotFoundError):
                check_frozen(study)


if __name__ == "__main__":
    unittest.main()
