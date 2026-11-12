import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scholar_daily.benchmark import digest, inspect_events, read_json, write_json
from scholar_daily.benchmark_agent import (Caller, FIVE_ARMS, execute_batch, init_policy, run, score, validate_policy)
from scholar_daily.benchmark_agent_report import report
from scholar_daily.benchmark_metrics import (canonical_usage, human_support, performance_metrics, prediction_scores,
                                            priced_cost, quality_metrics, ranked_scores)
from tests.test_benchmark_agent import FakeBackend, policy
from tests import test_benchmark_agent as fixtures


def five_policy():
    p = policy()
    p.update({"arms": list(FIVE_ARMS), "initial_k": {a: 1 for a in FIVE_ARMS}, "rerank_keep": 1,
              "batch_size": 1, "max_repair_rounds": 0, "max_model_calls_per_batch": 24,
              "model": {"backend": "fixture"}, "acceptance": {"quality": "positive paired interval"}})
    return p


class SuiteBackend(FakeBackend):
    def run(self, prompt, schema, folder):
        key = next(iter(schema["properties"]))
        payload = json.loads(prompt[prompt.index("\n[") + 1:])
        if key == "selections" or (key == "answers" and "paragraphs" not in payload[0]):
            self.calls.append(key)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "events.jsonl").write_text('{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}\n', encoding="utf-8")
            if key == "selections":
                return {key: [{"question_id": p["question_id"], "paragraph_ids": [c["id"] for c in p["paragraphs"][:p["select_count"]]]} for p in payload]}
            return {key: [{"question_id": p["question_id"], "answer": "42", "explanation": "known", "evidence_ids": []} for p in payload]}
        return super().run(prompt, schema, folder)


class FiveArmLoopTests(unittest.TestCase):
    def execute(self, arm, backend=None, p=None):
        backend, p = backend or SuiteBackend(), p or five_policy()
        with tempfile.TemporaryDirectory() as directory:
            result = execute_batch([fixtures.LoopTests().item()], arm, p, Caller(backend, Path(directory), p), "b")
        return result["original-id"], backend

    def test_closed_book_payload_has_no_paper_and_cannot_invent_citations(self):
        class InspectBackend(SuiteBackend):
            def run(self, prompt, schema, folder):
                payload = json.loads(prompt[prompt.index("\n[") + 1:])
                if "paragraphs" in payload[0] or "Target result" in prompt:
                    raise AssertionError("Paper text leaked")
                return super().run(prompt, schema, folder)
        state, backend = self.execute("llm", InspectBackend())
        self.assertEqual(state["prediction"]["answer"], "42")
        self.assertEqual(state["prediction"]["evidence"], [])
        self.assertEqual(backend.calls, ["answers"])
        self.assertFalse(state["final_review"]["passed"])

        class BadBackend(InspectBackend):
            def run(self, *args):
                response = super().run(*args)
                response["answers"][0]["evidence_ids"] = ["invented"]
                return response
        with self.assertRaisesRegex(RuntimeError, "bounded attempts"):
            self.execute("llm", BadBackend())

    def test_rerank_precedes_answer_and_preserves_candidate_ranking(self):
        state, backend = self.execute("rerank")
        self.assertEqual(backend.calls, ["selections", "answers", "reviews"])
        self.assertEqual(state["initial_context"], state["context"])
        self.assertEqual(state["rounds"], 1)

    def test_dynamic_searches_without_online_review_or_repair(self):
        state, backend = self.execute("dynamic")
        self.assertEqual(backend.calls, ["selections", "answers", "actions", "answers", "reviews"])
        self.assertNotIn("online_review", state)
        self.assertEqual(state["history"][0]["new_context_ids"], ["s0:p1"])
        self.assertEqual(state["prediction"]["evidence_ids"], ["s0:p1"])

    def test_dynamic_keeps_proposal_when_planner_stops_or_budget_runs_out(self):
        class StopBackend(SuiteBackend):
            def run(self, *args):
                result = super().run(*args)
                for action in result.get("actions", []):
                    action["action"] = "stop"
                return result
        state, _ = self.execute("dynamic", StopBackend())
        self.assertEqual(state["stop_reason"], "planner_stop")
        self.assertEqual(state["prediction"]["answer"], "42")
        p = five_policy()
        p.update(max_call_attempts=1, max_model_calls_per_batch=3)
        state, backend = self.execute("dynamic", p=p)
        self.assertEqual(state["stop_reason"], "call_budget")
        self.assertEqual(state["prediction"]["answer"], "42")
        self.assertEqual(len(backend.calls), 3)

    def test_verified_can_abstain_after_failed_online_checks(self):
        state, _ = self.execute("verified", SuiteBackend(never_supported=True))
        self.assertEqual(state["prediction"]["answer"], "Unanswerable")
        self.assertIn("online_review", state)
        self.assertEqual(state["rounds"], 2)

    def test_new_init_defaults_to_five_groups_and_keeps_legacy_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "model.json", {"model_configuration": {"backend": "fixture"}})
            init_policy(root / "new", root / "model.json")
            p = read_json(root / "new/policy.json")
            self.assertEqual(p["arms"], list(FIVE_ARMS))
            self.assertEqual(p["batch_size"], 1)
            init_policy(root / "old", root / "model.json", "legacy")
            self.assertEqual(len(read_json(root / "old/policy.json")["arms"]), 4)

    def test_mismatched_comparison_and_invalid_prices_fail_before_model_calls(self):
        p = five_policy()
        validate_policy(p)
        p["initial_k"]["dynamic"] = 2
        with self.assertRaisesRegex(ValueError, "Match candidate"):
            validate_policy(p)
        p = five_policy()
        p["pricing"] = {"currency": "CNY"}
        with self.assertRaisesRegex(ValueError, "input_per_million"):
            validate_policy(p)


class QualityTests(unittest.TestCase):
    def test_multireference_success_requires_one_consistent_reference(self):
        refs = [{"answer": "The 42.", "evidence": ["A"]}, {"answer": "43", "evidence": ["B"]}]
        result = prediction_scores({"answer": "42", "evidence": ["B"]}, refs)
        self.assertEqual(result["exact_match"], 1)
        self.assertEqual(result["evidence_precision"], 1)
        self.assertFalse(result["task_success"])
        self.assertTrue(prediction_scores({"answer": "42", "evidence": ["A"]}, refs)["task_success"])
        self.assertEqual(prediction_scores(None, refs)["evidence_recall"], 0)

    def test_ranking_uses_order_deduplication_and_nonempty_reference_denominator(self):
        context = [{"text": t} for t in ("X", "B", "B", "A")]
        refs = [{"evidence": []}, {"evidence": ["A", "B"]}]
        r = ranked_scores(context, refs, 3)
        self.assertEqual(r["recall"], 1)
        self.assertEqual(r["mrr"], .5)
        self.assertAlmostEqual(r["ndcg"], (1/math.log2(3) + .5)/(1 + 1/math.log2(3)))
        self.assertEqual(ranked_scores(context, refs, 1)["recall"], 0)
        self.assertIsNone(ranked_scores(context, [{"evidence": []}], 5))

    def test_missing_tasks_score_zero_and_recovery_has_explicit_denominator(self):
        refs = [{"answer": "42", "evidence": ["A"]}]
        gold = {"q": refs, "missing": refs}
        states = {"q": {"prediction": {"answer": "42", "evidence": ["A"]},
                        "initial_proposal": {"answer": "42", "evidence": ["wrong"]},
                        "initial_context": [{"text": "wrong"}], "context": [{"text": "A"}]}}
        rows = [{"question_id": qid} for qid in gold]
        result = quality_metrics(gold, states, rows, [1])
        self.assertEqual(result["task_success_rate"], .5)
        self.assertEqual(result["failure_recovery_rate"], 1)
        self.assertEqual(result["initial_task_failures"], 1)
        self.assertEqual(result["retrieval"]["final"]["1"]["recall"], .5)
        self.assertEqual(result["retrieval"]["final"]["1"]["questions"], 2)


class HumanAndCostTests(unittest.TestCase):
    def test_manual_reviews_require_identity_reason_and_matching_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = fixtures.LoopTests().item()
            states = {"original-id": {"prediction": {"answer": "42", "evidence_ids": ["s0:p1"]}, "context": item["chunks"]}}
            self.assertIsNone(human_support(root, [item], states)["support_rate"])
            document = read_json(root / "human-support.json")
            document["annotations"][0].update(verdict="supported", reviewer="reviewer-1", reason="Exact number and condition supported")
            write_json(root / "human-support.json", document)
            self.assertEqual(human_support(root, [item], states)["support_rate"], 1)
            packet_hash = digest((root / "human-support-packets.json").read_bytes())
            states["original-id"]["prediction"]["answer"] = "43"
            with self.assertRaisesRegex(ValueError, "different answer"):
                human_support(root, [item], states)
            self.assertEqual(digest((root / "human-support-packets.json").read_bytes()), packet_hash)

    def test_cached_pricing_and_missing_usage(self):
        prices = {"currency": "CNY", "input_per_million": 1., "cached_input_per_million": .1, "output_per_million": 2.}
        records = [{"usage": [{"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 50}]}]
        self.assertAlmostEqual(priced_cost(records, prices, "codex")["amount"], .000164)
        records[0]["usage"][0]["cache_read_tokens"] = records[0]["usage"][0].pop("cached_input_tokens")
        self.assertAlmostEqual(priced_cost(records, prices, "harness")["amount"], .000204)
        self.assertIsNone(priced_cost(records + [{"usage": []}], prices, "codex"))
        self.assertIsNone(priced_cost(records, None, "codex"))
        with self.assertRaises(ValueError):
            priced_cost(records, dict(prices, input_per_million=-1), "codex")

    def test_harness_camelcase_usage_and_retry_recovery_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            events.write_text('{"type":"assistant/message","data":{"usage":{"inputTokens":10,"outputTokens":5,"cacheReadTokens":2}}}', encoding="utf-8")
            usage, _ = inspect_events(events)
            usage = canonical_usage(usage)
            self.assertEqual(usage[0], {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 2})
            for attempt, status in enumerate(("error", "ok")):
                write_json(root / f"calls/0000/answer/attempt-{attempt}/record.json",
                           {"usage": usage, "status": status, "stage": "answer", "wall_seconds": 2})
            p = performance_metrics(root, [{"wall_seconds": 5, "inference_wall_seconds": 3, "question_ids": ["a", "b"]}], 2, {})
            self.assertEqual(p["call_failure_recovery_rate"], 1)
            self.assertEqual(p["question_latency_p95_seconds"], 3)
            self.assertEqual(p["evaluation_inclusive_question_latency_p95_seconds"], 5)
            self.assertEqual(p["mean_online_model_attempts"], 1)


class FiveArmIntegrationTests(unittest.TestCase):
    def test_five_arm_runs_scores_manual_review_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            p = five_policy()
            p["pricing"] = {"currency": "CNY", "input_per_million": 1., "cached_input_per_million": .1, "output_per_million": 2.}
            write_json(study / "policy.json", p)
            write_json(study / "dev/inputs.json", [fixtures.LoopTests().item()])
            write_json(study / "dev/gold.json", {})
            write_json(study / "reservation.json", {"data_dir": directory, "hashes": {"dev": {
                k: digest((study / "dev" / f"{k}.json").read_bytes()) for k in ("inputs", "gold")}}})
            refs = {"original-id": [{"answer": "42", "type": "abstractive", "evidence": ["Target result: 42"]}]}
            def evaluate(gold, predictions):
                pred = predictions.get("original-id", {})
                return {"Answer F1": float(pred.get("answer") == "42"), "Evidence F1": float(pred.get("evidence") == ["Target result: 42"])}
            evaluator = SimpleNamespace(get_answers_and_evidence=lambda *args: refs,
                                        token_f1_score=lambda a, b: float(a == b), evaluate=evaluate)
            def cli_output(command, **kwargs):
                predictions = {v["question_id"]: {"answer": v["predicted_answer"], "evidence": v["predicted_evidence"]}
                               for v in map(json.loads, Path(command[-1]).read_text(encoding="utf-8").splitlines())}
                return json.dumps(evaluate(refs, predictions))
            backend = SuiteBackend()
            with patch("scholar_daily.benchmark_agent.get_backend", return_value=backend), \
                 patch("scholar_daily.benchmark_agent.load_evaluator", return_value=evaluator), \
                 patch("scholar_daily.benchmark_agent_report.subprocess.check_output", side_effect=cli_output):
                for arm in FIVE_ARMS:
                    run(study, "dev", arm)
                document = read_json(study / "runs/dev/dynamic/human-support.json")
                document["annotations"][0].update(verdict="supported", reviewer="test-reviewer", reason="Target result matches")
                write_json(study / "runs/dev/dynamic/human-support.json", document)
                score(study, "dev", "dynamic")
                result = report(study, "dev")
            self.assertEqual(len(result["module_effects"]), 4)
            self.assertEqual(result["runs"]["dynamic"]["quality"]["failure_recovery_rate"], 1)
            self.assertEqual(result["runs"]["dynamic"]["human_support"]["support_rate"], 1)
            self.assertEqual(result["actual_experiment_cost"]["calls"], len(backend.calls))
            self.assertEqual(result["actual_experiment_cost"]["monetary_cost"]["currency"], "CNY")
            self.assertFalse(result["runs"]["llm"]["quality"]["retrieval_applicable"])
            self.assertEqual(result["runs"]["verified"]["reused_calls"], 0)
            self.assertTrue(all(result["official_cli_checks"].values()))
            content = (study / "comparisons/dev/report.md").read_text(encoding="utf-8")
            self.assertIn("P95", content)
            self.assertIn("人工支持率", content)
            self.assertIn("主动补检索", content)


if __name__ == "__main__":
    unittest.main()
