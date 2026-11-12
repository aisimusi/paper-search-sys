import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scholar_daily.benchmark import read_json, write_json
from scholar_daily.agents import parse_response
from scholar_daily.benchmark_agent import (Caller, ForbiddenToolUse, bounded_append,
    check_frozen, execute_batch, expand_context, implementation_hashes)
from scholar_daily.benchmark_holdout import select_holdout, similar_question
from scholar_daily.benchmark_support import validate_review


def policy():
    return {"initial_k": {"rag5": 1, "rag20": 1, "agent": 1}, "max_rounds": 2,
            "max_paragraphs": 3, "max_context_characters": 1000, "step_k": 1,
            "max_call_attempts": 2}


def review(span="42", quote="Target result: 42", verdict="supported", conditions=True):
    return {"all_claims_covered": "yes", "missing_information": "Need target result",
            "claims": [{"answer_span": span, "claim": "The result is 42", "verdict": verdict,
                        "conditions_supported": "yes" if conditions else "no", "reason": "check", "citations": [
                            {"paragraph_id": "s0:p1", "quote": quote}]}]}


class FakeBackend:
    def __init__(self, never_supported=False, repeated_query=False):
        self.never_supported, self.repeated_query = never_supported, repeated_query
        self.calls = []

    def run(self, prompt, schema, folder):
        payload = json.loads(prompt[prompt.index("\n[") + 1:])
        key = next(iter(schema["properties"]))
        self.calls.append(key)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "events.jsonl").write_text('{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}\n', encoding="utf-8")
        values = []
        for p in payload:
            if key == "answers":
                target = next((c for c in p["paragraphs"] if "Target" in c["text"]), p["paragraphs"][0])
                values.append({"question_id": p["question_id"], "answer": "42", "explanation": "result", "evidence_ids": [target["id"]]})
            elif key == "reviews":
                supported = any("Target" in c["text"] for c in p["cited_paragraphs"]) and not self.never_supported
                r = review(verdict="supported" if supported else "unsupported")
                if not supported:
                    r["claims"][0]["citations"] = []
                values.append(dict(r, question_id=p["question_id"]))
            else:
                values.append({"question_id": p["question_id"], "action": "search", "query": "baseline" if self.repeated_query else "target result", "anchor_ids": [], "reason": "gap"})
        return parse_response(json.dumps({key: values}), schema)


class SupportTests(unittest.TestCase):
    def test_supported_claim_requires_valid_quote_and_qualifiers(self):
        cited = [{"id": "s0:p1", "text": "Target result: 42 on the test split."}]
        self.assertTrue(validate_review(review(), "42", cited)["passed"])
        self.assertFalse(validate_review(review(conditions=False), "42", cited)["passed"])
        self.assertFalse(validate_review(review(verdict="partial"), "42", cited)["passed"])
        self.assertFalse(validate_review(review(verdict="contradicted"), "42", cited)["passed"])

    def test_fabricated_quote_or_uncited_id_is_rejected(self):
        cited = [{"id": "s0:p1", "text": "Target result: 42"}]
        checked = validate_review(review(quote="result: 99"), "42", cited)
        self.assertFalse(checked["passed"])
        self.assertEqual(checked["invalid_quotes"], 1)
        self.assertFalse(validate_review(review(), "42", [dict(cited[0], id="other")])["passed"])

    def test_uncovered_fact_or_unit_is_rejected(self):
        cited = [{"id": "s0:p1", "text": "Target result: 42"}]
        for answer in ("42 and 17", "42%", "-42"):
            self.assertFalse(validate_review(review(), answer, cited)["passed"])

    def test_connectors_between_atomic_claims_are_not_missing_facts(self):
        cited = [{"id": "s0:p1", "text": "Target result: 42 and 17"}]
        r = review()
        r["claims"].append(dict(r["claims"][0], answer_span="17", claim="The other result is 17"))
        self.assertTrue(validate_review(r, "42 and 17", cited)["passed"])
        self.assertFalse(validate_review(r, "42 and at least 17", cited)["passed"])
        self.assertFalse(validate_review(r, "42 or 17", cited)["passed"])

    def test_only_whitespace_normalization_is_allowed_in_quotes(self):
        cited = [{"id": "s0:p1", "text": "Target\nresult: 42"}]
        self.assertTrue(validate_review(review(), "42", cited)["passed"])
        self.assertFalse(validate_review(review(quote="target result: 42"), "42", cited)["passed"])


class LoopTests(unittest.TestCase):
    def item(self):
        return {"question_id": "original-id", "title": "Paper", "question": "baseline",
                "chunks": [{"id": "s0:p0", "section": "s", "text": "baseline baseline"},
                           {"id": "s0:p1", "section": "s", "text": "Target result: 42"}]}

    def execute(self, backend, arm="agent", item=None):
        with tempfile.TemporaryDirectory() as directory:
            caller = Caller(backend, Path(directory), policy())
            result = execute_batch([item or self.item()], arm, policy(), caller, "batch")
            return result["original-id"]

    def test_rewrite_retrieve_reanswer_and_independent_final_audit(self):
        backend = FakeBackend()
        state = self.execute(backend)
        self.assertEqual(state["rounds"], 2)
        self.assertEqual(state["stop_reason"], "support_passed")
        self.assertEqual(state["prediction"]["answer"], "42")
        self.assertTrue(state["final_review"]["passed"])
        self.assertEqual(state["history"][0]["new_context_ids"], ["s0:p1"])
        self.assertEqual(backend.calls, ["answers", "reviews", "actions", "answers", "reviews", "reviews"])

    def test_round_limit_abstains_and_keeps_last_proposal(self):
        state = self.execute(FakeBackend(never_supported=True))
        self.assertEqual(state["stop_reason"], "max_rounds")
        self.assertEqual(state["prediction"]["answer"], "Unanswerable")
        self.assertEqual(state["last_proposal"]["answer"], "42")
        self.assertFalse(state["final_review"]["passed"])
        self.assertEqual(state["rounds"], 2)

    def test_repeated_query_stops_before_another_answer(self):
        state = self.execute(FakeBackend(repeated_query=True))
        self.assertEqual(state["stop_reason"], "repeated_query")
        self.assertEqual(state["rounds"], 1)

    def test_static_rag_is_audited_without_rewriting_its_answer(self):
        backend = FakeBackend()
        state = self.execute(backend, "rag5")
        self.assertEqual(state["prediction"]["answer"], "42")
        self.assertFalse(state["final_review"]["passed"])
        self.assertEqual(state["stop_reason"], "fixed_context")
        self.assertEqual(backend.calls, ["answers", "reviews"])

    def test_supported_first_round_stops_early(self):
        item = self.item()
        item["question"] = "target result"
        state = self.execute(FakeBackend(), item=item)
        self.assertEqual(state["rounds"], 1)
        self.assertEqual(state["stop_reason"], "support_passed")

    def test_budgets_dedup_and_neighbor_boundaries(self):
        p = policy()
        p["max_context_characters"] = 10
        chunks = [{"id": "s0:p0", "text": "12345"}, {"id": "s0:p1", "text": "12345"}, {"id": "s1:p0", "text": "1"}]
        self.assertEqual(len(bounded_append([chunks[0]], chunks, p)), 2)
        state = {"context": [chunks[0]], "queries": ["q"]}
        context, stop = expand_context({"chunks": chunks}, state, {"action": "read_neighbors", "anchor_ids": ["s0:p0"]}, p)
        self.assertEqual([c["id"] for c in context], ["s0:p0", "s0:p1"])
        self.assertIsNone(stop)

    def test_no_new_evidence_and_invalid_anchor(self):
        item = self.item()
        state = {"context": item["chunks"], "queries": ["baseline"]}
        _, stop = expand_context(item, state, {"action": "search", "query": "missingxyz"}, policy())
        self.assertEqual(stop, "no_new_evidence")
        _, stop = expand_context(item, state, {"action": "read_neighbors", "anchor_ids": ["fake"]}, policy())
        self.assertEqual(stop, "invalid_anchor")


class CallerTests(unittest.TestCase):
    def test_failed_call_attempts_are_bounded_across_resume(self):
        class BrokenBackend:
            calls = 0
            def run(self, prompt, schema, folder):
                self.calls += 1
                raise RuntimeError("backend unavailable")
        backend = BrokenBackend()
        with tempfile.TemporaryDirectory() as directory:
            caller = Caller(backend, Path(directory), policy())
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    execute_batch([LoopTests().item()], "rag5", policy(), caller, "b")
            self.assertEqual(backend.calls, 2)

    def test_cached_calls_do_not_generate_again(self):
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            caller = Caller(backend, Path(directory), policy())
            batch = [LoopTests().item()]
            first = execute_batch(batch, "rag5", policy(), caller, "b")
            second = execute_batch(batch, "rag5", policy(), caller, "b")
            self.assertEqual(first, second)
            self.assertEqual(len(backend.calls), 2)

    def test_tool_use_is_rejected_without_retry(self):
        class ToolBackend(FakeBackend):
            def run(self, prompt, schema, folder):
                value = super().run(prompt, schema, folder)
                (folder / "events.jsonl").write_text('{"type":"item.completed","item":{"type":"command_execution"}}\n', encoding="utf-8")
                return value
        backend = ToolBackend()
        with self.assertRaises(ForbiddenToolUse):
            LoopTests().execute(backend, "rag5")
        self.assertEqual(len(backend.calls), 1)


class HoldoutTests(unittest.TestCase):
    def test_inference_cannot_read_holdout_inputs_before_freeze(self):
        from scholar_daily.benchmark_agent import run
        with tempfile.TemporaryDirectory() as directory:
            with patch("scholar_daily.benchmark_agent.checked_inputs") as inputs:
                with self.assertRaises(FileNotFoundError):
                    run(Path(directory), "holdout", "agent")
                inputs.assert_not_called()

    def paper(self, title, text, question, qid):
        return {"title": title, "full_text": [{"section_name": "s", "paragraphs": [text]}],
                "qas": [{"question_id": qid, "question": question, "answers": [{"answer": {"unanswerable": False, "evidence": [text]}}]}]}

    def test_excludes_prior_papers_titles_text_and_questions(self):
        old = self.paper("Old", "old text", "old question", "q0")
        papers = {"old": old, "same-title": self.paper("OLD", "new text", "new q1", "q1"),
                  "same-text": self.paper("Other", "old text", "new q2", "q2"),
                  "same-question": self.paper("Third", "third text", "Old question!", "q3"),
                  "fresh": self.paper("Fresh", "fresh text", "fresh question", "q4")}
        self.assertEqual(list(select_holdout(papers, [{"old": old}], 1, 9)), ["fresh"])
        with self.assertRaises(ValueError):
            select_holdout(papers, [{"old": old}], 2, 9)

    def test_near_duplicate_long_questions(self):
        seen = {"what are the main evaluation metrics used for these experiments"}
        self.assertTrue(similar_question("What are the main evaluation metrics used in these experiments?", seen))

    def test_holdout_requires_freeze_and_detects_changed_policy(self):
        from scholar_daily.benchmark import digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                check_frozen(root)
            write_json(root / "policy.json", {"limit": 2})
            write_json(root / "reservation.json", {})
            write_json(root / "freeze.json", {"policy_sha256": digest((root / "policy.json").read_bytes()),
                       "reservation_sha256": digest((root / "reservation.json").read_bytes()), "implementation": implementation_hashes()})
            check_frozen(root)
            write_json(root / "policy.json", {"limit": 3})
            with self.assertRaises(ValueError):
                check_frozen(root)


if __name__ == "__main__":
    unittest.main()
