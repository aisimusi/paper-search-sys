import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scholar_daily.benchmark import (add_neighbors, answer_diagnostics, character_normalize,
                                     checked_snapshot, digest, exclusion_reason, inspect_events,
                                     oracle_context, read_json, retrieve, score, validate_prediction,
                                     variant, write_json)


class BenchmarkTests(unittest.TestCase):
    def test_neighbors_stay_in_section_and_preserve_seed_order(self):
        chunks = [{"id": cid, "text": cid} for cid in ("s0:p0", "s0:p1", "s0:p2", "s1:p0")]
        result = add_neighbors([chunks[2], chunks[1]], chunks, 1)
        self.assertEqual([c["id"] for c in result], ["s0:p2", "s0:p1", "s0:p0"])
        self.assertEqual(add_neighbors([chunks[2]], chunks, 0), [chunks[2]])

    def test_oracle_union_ignores_unanswerable_annotations_and_deduplicates(self):
        qa = {"answers": [{"answer": {"unanswerable": False, "evidence": ["b", "a"]}},
                          {"answer": {"unanswerable": False, "evidence": ["a"]}},
                          {"answer": {"unanswerable": True, "evidence": ["c"]}}]}
        chunks = [{"id": str(i), "text": t} for i, t in enumerate(["a", "b", "a", "c"])]
        self.assertEqual([c["id"] for c in oracle_context(qa, chunks)], ["0", "1"])
        self.assertEqual(oracle_context({"answers": [qa["answers"][-1]]}, chunks), [])
        with self.assertRaises(ValueError):
            oracle_context(qa, chunks[:1])

    def test_character_normalization_is_explicit_and_preserves_numbers(self):
        self.assertEqual(character_normalize("‘Ｆ１’\u00a0–\u200b １２\u00ad"), "'F1' - 12")
        self.assertEqual(character_normalize("-1.5 vs 1.5"), "-1.5 vs 1.5")

    def test_diagnostics_keep_official_score_and_missing_predictions(self):
        evaluator = SimpleNamespace(token_f1_score=lambda a, b: float(a == b))
        gold = {"q": [{"answer": "F1", "type": "extractive", "evidence": ["text"]}],
                "missing": [{"answer": "Unanswerable", "type": "none", "evidence": []}]}
        tasks = [{"question_id": "q", "paragraphs": [{"text": "text"}]},
                 {"question_id": "missing", "paragraphs": []}]
        predictions = {"q": {"answer": "Ｆ１", "explanation": "F1", "evidence": ["text"]}}
        summary, rows = answer_diagnostics(evaluator, gold, predictions, tasks)
        self.assertEqual(rows[0]["official_answer_f1"], 0)
        self.assertEqual(rows[0]["character_normalized_answer_f1"], 1)
        self.assertEqual(summary["character_normalized_answer_f1"], .5)
        self.assertEqual(rows[1]["official_answer_f1"], 0)
        self.assertFalse(rows[1]["completed"])
        self.assertEqual(predictions["q"]["answer"], "Ｆ１")

    def test_short_answer_and_explanation_remain_separate(self):
        prediction = {"question_id": "q", "answer": "Yes", "explanation": "The paper provides support.", "evidence_ids": ["p"]}
        result = validate_prediction(prediction, {"question_id": "q", "paragraphs": [{"id": "p", "text": "text"}]})
        self.assertEqual(result["answer"], "Yes")
        self.assertEqual(result["explanation"], prediction["explanation"])

    def make_snapshot(self, root):
        chunks = [{"id": "s0:p0", "section": "section", "text": "text"}]
        task = {"question_id": "q", "paper_id": "paper", "title": "title", "question": "question",
                "paragraphs": chunks, "paper_text_sha256": digest(json.dumps(chunks, ensure_ascii=False).encode())}
        gold = {"paper": {"title": "title", "full_text": [{"section_name": "section", "paragraphs": ["text"]}],
                          "qas": [{"question_id": "q", "question": "question", "answers": [
                              {"answer": {"unanswerable": False, "evidence": ["text"]}}]}]}}
        write_json(root / "dev/gold.json", gold)
        write_json(root / "dev/tasks.json", [task])
        write_json(root / "manifest.json", {"data_dir": str(root), "top_k": 5, "splits": {"dev": {
            "question_ids": ["q"], "gold_sha256": digest((root / "dev/gold.json").read_bytes()),
            "tasks_sha256": digest((root / "dev/tasks.json").read_bytes())}}})

    def test_variant_keeps_exact_ids_and_rejects_overwrite_or_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "variant"
            self.make_snapshot(source)
            with patch("scholar_daily.benchmark.load_evaluator"), patch("scholar_daily.benchmark.retrieval_metrics", return_value={}):
                args = SimpleNamespace(source=source, output=output, top_k=10, neighbors=1, context="bm25", splits=["dev"])
                manifest = variant(args)
                self.assertEqual(list(manifest["splits"]), ["dev"])
                self.assertEqual(read_json(source / "dev/gold.json"), read_json(output / "dev/gold.json"))
                self.assertEqual(manifest["splits"]["dev"]["question_ids"], ["q"])
                with self.assertRaises(ValueError):
                    variant(args)
            (source / "dev/tasks.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_snapshot(source, "dev")

    def test_score_rejects_duplicate_batch_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_snapshot(root)
            result = {"question_ids": ["q"], "predictions": {"q": {"answer": "Yes", "evidence": ["text"]}}, "errors": {}}
            for folder in ("0000", "0001"):
                write_json(root / "dev/batches" / folder / "result.json", result)
            evaluator = SimpleNamespace(get_answers_and_evidence=lambda *a: {"q": []})
            with patch("scholar_daily.benchmark.load_evaluator", return_value=evaluator), self.assertRaises(ValueError):
                score(SimpleNamespace(output=root))

    def test_retrieval_uses_question_and_paragraphs(self):
        chunks = [{"id": "p0", "text": "A bird sings."},
                  {"id": "p1", "text": "The transformer uses attention."}]
        self.assertEqual(retrieve("transformer attention", chunks, 1)[0]["id"], "p1")
        self.assertEqual(retrieve("test", [], 5), [])

    def test_multireference_figure_cannot_pass_text_filter(self):
        qa = {"answers": [{"answer": {"unanswerable": False, "evidence": ["text"]}},
                          {"answer": {"unanswerable": False, "evidence": ["FLOAT SELECTED: table"]}}]}
        self.assertEqual(exclusion_reason(qa, [{"text": "text"}]), "figure_or_table_evidence")

    def test_unmapped_gold_is_excluded(self):
        qa = {"answers": [{"answer": {"unanswerable": False, "evidence": ["missing"]}}]}
        self.assertEqual(exclusion_reason(qa, [{"text": "text"}]), "evidence_not_exactly_mapped")

    def test_evidence_must_come_from_supplied_context(self):
        task = {"question_id": "q", "paragraphs": [{"id": "p", "text": "Exact evidence."}]}
        valid = {"question_id": "q", "answer": "Yes", "evidence_ids": ["p"]}
        self.assertEqual(validate_prediction(valid, task)["evidence"], ["Exact evidence."])
        for ids in (["unknown"], ["p", "p"], []):
            with self.assertRaises(ValueError):
                validate_prediction(dict(valid, evidence_ids=ids), task)

    def test_abstention_requires_empty_evidence(self):
        task = {"question_id": "q", "paragraphs": [{"id": "p", "text": "text"}]}
        prediction = {"question_id": "q", "answer": "Unanswerable", "evidence_ids": []}
        self.assertEqual(validate_prediction(prediction, task)["answer"], "Unanswerable")
        with self.assertRaises(ValueError):
            validate_prediction(dict(prediction, evidence_ids=["p"]), task)

    def test_tool_use_audit_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"type":"item.completed","item":{"type":"command_execution"}}\n'
                            '{"type":"turn.completed","usage":{"input_tokens":123}}\n', encoding="utf-8")
            usage, calls = inspect_events(path)
            self.assertEqual(calls, ["command_execution"])
            self.assertEqual(usage, [{"input_tokens": 123}])

    def test_harness_events_preserve_usage_and_detect_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"type":"assistant/message","data":{"usage":{"inputTokens":123}}}\n'
                            '{"type":"tool/start","data":{}}\n', encoding="utf-8")
            usage, calls = inspect_events(path)
            self.assertEqual(usage, [{"inputTokens": 123}])
            self.assertEqual(calls, ["tool/start"])


if __name__ == "__main__":
    unittest.main()
