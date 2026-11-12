import copy
import json
import math
import unittest

from scholar_daily.benchmark import paragraphs
from scripts.evaluate_retrieval import (ARMS, paired_delta, prepare_cases, rerank_payload,
                                        scores, select_reranked, summarize)


def fixture():
    paper = {"title": "Fictional evaluation fixture", "full_text": [
        {"section_name": "Results", "paragraphs": ["Evidence A", "Neutral text", "Evidence B"]}],
        "qas": [{"question_id": "q", "question": "Fixture question?", "answers": [
            {"answer": {"unanswerable": False, "evidence": ["Evidence A"]}},
            {"answer": {"unanswerable": False, "evidence": ["Evidence B"]}},
            {"answer": {"unanswerable": True, "evidence": ["Neutral text"]}}]}]}
    item = {"paper_id": "p", "question_id": "q", "title": paper["title"],
            "question": "Fixture question?", "chunks": paragraphs(paper)}
    return {"p": paper}, [item]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_fixed_union_does_not_choose_best_reference_for_each_method(self):
        gold, inputs = fixture()
        cases, exclusions = prepare_cases(gold, inputs)
        self.assertEqual(exclusions, [])
        self.assertEqual(cases[0]["relevant_texts"], ["Evidence A", "Evidence B"])
        result = scores([cases[0]["chunks"][0]], cases[0]["relevant_texts"])
        self.assertEqual(result["recall_at_5"], .5)
        self.assertAlmostEqual(result["ndcg_at_10"], 1 / (1 + 1 / math.log2(3)))

    def test_binary_ndcg_and_candidate_ceiling_use_full_qrels(self):
        ranking = [{"text": t} for t in ("unrelated", "A", "B")]
        result = scores(ranking, ["A", "B"])
        self.assertEqual(result["recall_at_5"], 1)
        self.assertAlmostEqual(result["ndcg_at_10"], (1 / math.log2(3) + .5) / (1 + 1 / math.log2(3)))
        self.assertEqual(scores([], ["A"]), {"recall_at_5": 0, "ndcg_at_10": 0})
        with self.assertRaises(ValueError):
            scores([], [])

    def test_exact_text_duplicates_have_one_canonical_paragraph(self):
        gold, inputs = fixture()
        gold["p"]["full_text"][0]["paragraphs"].insert(1, "Evidence A")
        inputs[0]["chunks"] = paragraphs(gold["p"])
        cases, _ = prepare_cases(gold, inputs)
        self.assertEqual(len(cases[0]["chunks"]), 3)
        self.assertEqual(cases[0]["chunks"][0]["id"], "s0:p0")
        self.assertTrue(all(c["source_kind"] == "qasper_native_paragraph" for c in cases[0]["chunks"]))

    def test_no_positive_or_unmapped_evidence_is_excluded_before_all_arms(self):
        for answers, reason in (([{"answer": {"unanswerable": True, "evidence": []}}], "no_positive_evidence"),
                                ([{"answer": {"unanswerable": False, "evidence": ["Missing"]}}], "unmapped_evidence")):
            gold, inputs = fixture()
            gold["p"]["qas"][0]["answers"] = answers
            cases, exclusions = prepare_cases(gold, inputs)
            self.assertEqual(cases, [])
            self.assertEqual(exclusions[0]["reason"], reason)

    def test_changed_candidate_scope_question_or_duplicate_id_is_rejected(self):
        gold, inputs = fixture()
        for mutation in ("scope", "question", "duplicate"):
            rows = copy.deepcopy(inputs)
            if mutation == "scope":
                rows[0]["chunks"].pop()
            elif mutation == "question":
                rows[0]["question"] = "Changed"
            else:
                rows.append(copy.deepcopy(rows[0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare_cases(gold, rows)

    def test_reranker_payload_never_contains_reference_or_relevance_fields(self):
        gold, inputs = fixture()
        case = prepare_cases(gold, inputs)[0][0]
        payload = rerank_payload(case, case["chunks"])
        self.assertEqual(set(payload[0]), {"question_id", "title", "question", "select_count", "paragraphs"})
        self.assertEqual(set(payload[0]["paragraphs"][0]), {"id", "section", "text"})
        self.assertNotIn("relevant_texts", json.dumps(payload))

    def test_rerank_requires_full_permutation_and_preserves_candidates(self):
        gold, inputs = fixture()
        chunks = prepare_cases(gold, inputs)[0][0]["chunks"]
        before = copy.deepcopy(chunks)
        output = {"paragraph_ids": [c["id"] for c in reversed(chunks)]}
        self.assertEqual(select_reranked(output, chunks), list(reversed(chunks)))
        with self.assertRaises(ValueError):
            select_reranked({"paragraph_ids": [chunks[0]["id"]]}, chunks)
        self.assertEqual(chunks, before)

    def test_failures_stay_in_common_denominator_and_latency_summary(self):
        rows = [{"arms": {arm: {"metrics": {"recall_at_5": score, "ndcg_at_10": score},
                                "seconds": .01 + score, "error": "failed" if score == 0 else None}
                          for arm in ARMS}} for score in (1, 0)]
        result = summarize(rows)
        self.assertTrue(all(v["questions"] == 2 and v["failures"] == 1 for v in result.values()))
        self.assertEqual(result["bm25"]["recall_at_5"], .5)
        self.assertEqual(result["bm25"]["query_ms_p95"], 1010)

    def test_paired_bootstrap_uses_same_questions(self):
        rows = [{"arms": {"bm25": {"metrics": {"recall_at_5": .5}},
                          "hybrid": {"metrics": {"recall_at_5": 1}}}}]
        result = paired_delta(rows, "hybrid", "recall_at_5")
        self.assertEqual(result["delta"], .5)
        self.assertEqual(result["bootstrap_95"], [.5, .5])


if __name__ == "__main__":
    unittest.main()
