import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scholar_daily import benchmark
from scholar_daily.models import Paper
from scholar_daily.paper_context import GAP, digest, load_index, save_index, select_context
from scholar_daily.retrieval import Retriever, bm25_retrieve, lexical_tokens


class BM25Tests(unittest.TestCase):
    def test_shared_formula_has_expected_scores_and_stable_ties(self):
        chunks = [{"text": "alpha alpha"}, {"text": "beta beta"}, {"text": "alpha beta"}]
        result = bm25_retrieve("alpha alpha", chunks, 10)
        idf = math.log(1 + 1.5 / 2.5)
        self.assertEqual([r["text"] for r in result], ["alpha alpha", "alpha beta"])
        self.assertEqual(result[0]["score"], idf * 2 * 2.5 / 3.5)
        self.assertAlmostEqual(result[1]["score"], idf)
        tied = [{"id": "b", "text": "alpha"}, {"id": "a", "text": "alpha"}]
        self.assertEqual([r["id"] for r in bm25_retrieve("alpha", tied, 2)], ["b", "a"])

    def test_no_zero_score_candidates_or_input_mutation(self):
        chunks = [{"id": "empty", "text": ""}, {"id": "a", "text": "alpha"}]
        original = copy.deepcopy(chunks)
        for query in ("", "  ", "?!", "absent"):
            self.assertEqual(bm25_retrieve(query, chunks, 4), [])
        self.assertEqual(bm25_retrieve("alpha", [], 5), [])
        self.assertEqual(chunks, original)

    def test_benchmark_delegates_with_historical_tokenizer_and_zero_fill(self):
        chunks = [{"id": "b", "text": "鸟"}, {"id": "a", "text": "Alpha"}]
        with patch("scholar_daily.benchmark.bm25_retrieve", wraps=bm25_retrieve) as shared:
            result = benchmark.retrieve("ALPHA", chunks, 10)
        shared.assert_called_once_with("ALPHA", chunks, 10, tokenizer=benchmark.tokens, include_zero=True)
        self.assertEqual([r["id"] for r in result], ["a", "b"])
        self.assertEqual(result[1]["score"], 0)
        self.assertEqual(benchmark.retrieve("unmatched", chunks, 2), [dict(c, score=0.0) for c in chunks])
        self.assertEqual(benchmark.tokens("ALPHA-7 中文"), ["alpha", "7"])

    def test_chinese_tokens_match_subphrases_without_common_character_noise(self):
        self.assertEqual(lexical_tokens("知识检索 BM25"), ["知识", "识检", "检索", "bm25"])
        chunks = [{"text": "本文使用知识检索增强模型"}, {"text": "本研究探讨索引"}]
        self.assertEqual([r["text"] for r in bm25_retrieve("检索增强", chunks, 5)], [chunks[0]["text"]])
        self.assertEqual(bm25_retrieve("完全缺失", chunks, 5), [])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.papers, self.contexts = {}, {}
        self.retriever = Retriever(self.contexts, self.papers, folder=self.root)

    def add_paper(self, pid, text, *, tables=None, pages=()):
        paper = Paper(paper_id=pid, source="fixture", title="Fictional retrieval fixture", authors=[],
                      abstract="Fixture abstract", published="v1", updated="v1", url="https://example.invalid/", pdf_url="")
        texts = {"layout": text}
        if tables is not None:
            texts["tables"] = tables
        index, record = save_index(self.root, paper, texts, pages)
        self.papers[pid], self.contexts[pid] = paper.as_dict(), record
        return index, record

    def rewrite_index(self, pid, index):
        record = self.contexts[pid]
        Path(record["index_path"]).write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        record["source_fingerprint"] = digest(json.dumps(index, ensure_ascii=False, sort_keys=True))[:24]

    def test_joint_corpus_scoring_and_paper_filter(self):
        self.add_paper("a", "alpha alpha")
        self.add_paper("b", "beta beta")
        self.add_paper("c", "alpha beta")
        result = self.retriever.retrieve("alpha", ["b", "c", "a"], 10, "bm25")
        self.assertEqual(result["issues"], [])
        self.assertEqual([r["paper_id"] for r in result["hits"]], ["a", "c"])
        self.assertEqual(result["hits"][0]["score"], math.log(1 + 1.5 / 2.5) * 5 / 3.5)
        self.assertEqual(len(self.retriever.retrieve("alpha", ["a", "c"], top_k=1)["hits"]), 1)
        self.assertEqual(self.retriever.retrieve("alpha", ["b"])["hits"], [])
        self.assertEqual(self.retriever.retrieve("alpha", [])["hits"], [])

    def test_offsets_pages_fingerprints_and_candidate_status_round_trip(self):
        text = "1 Introduction\n前文\n\f2 Results\nThe latency change is -5 ms."
        index, record = self.add_paper("a", text, pages=["layout"])
        row = self.retriever.retrieve("latency", ["a"])["hits"][0]
        self.assertEqual(row["text"], text[row["start"]:row["end"]])
        self.assertEqual((row["page_start"], row["page_end"], row["section"]), (2, 2, "Results"))
        self.assertEqual(row["source_fingerprint"], record["source_fingerprint"])
        self.assertEqual(row["source_sha256"], digest(text))
        self.assertEqual(row["offset_unit"], index["offset_unit"])
        self.assertEqual(row["verification_status"], "unverified")
        self.assertNotIn("evidence_id", row)
        self.assertEqual(json.loads(json.dumps(row)), row)

    def test_cross_paper_identical_fragments_retain_both_provenances(self):
        self.add_paper("a", "identical retrieval material")
        self.add_paper("b", "identical retrieval material")
        result = self.retriever.retrieve("retrieval", ["b", "a", "b", "a"])
        hits = result["hits"]
        self.assertEqual([r["paper_id"] for r in hits], ["a", "b"])
        self.assertEqual(hits[0]["fragment_id"], hits[1]["fragment_id"])
        self.assertNotEqual(hits[0]["source_fingerprint"], hits[1]["source_fingerprint"])
        self.assertEqual(result, self.retriever.retrieve("retrieval", ["a", "b"]))

    def test_chinese_query_and_empty_or_no_match_queries(self):
        self.add_paper("a", "2 方法\n本文使用知识检索增强模型。")
        self.assertEqual(len(self.retriever.retrieve("检索增强", ["a"])["hits"]), 1)
        for query in ("", " \n ", "?!", "absentterm", "完全缺失"):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.retrieve(query, ["a"])["hits"], [])

    def test_missing_index_unknown_paper_and_partial_success_are_explicit(self):
        self.add_paper("valid", "retrieval material")
        self.add_paper("no_record", "retrieval material")
        self.contexts.pop("no_record")
        _, record = self.add_paper("missing_file", "retrieval material")
        Path(record["index_path"]).unlink()
        result = self.retriever.retrieve("retrieval", ["valid", "no_record", "missing_file", "unknown"])
        self.assertEqual([r["paper_id"] for r in result["hits"]], ["valid"])
        self.assertEqual({r["paper_id"]: r["code"] for r in result["issues"]},
                         {"no_record": "index_missing", "missing_file": "material_missing", "unknown": "paper_missing"})

    def test_snapshot_change_after_first_query_is_not_cached(self):
        _, record = self.add_paper("a", "retrieval material")
        self.assertTrue(self.retriever.retrieve("retrieval", ["a"])["hits"])
        (Path(record["index_path"]).parent / "layout.txt").write_text("Changed retrieval material", encoding="utf-8")
        result = self.retriever.retrieve("retrieval", ["a"])
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["issues"][0]["code"], "invalid_material")

    def test_tampered_index_and_missing_source_fail_closed(self):
        index, record = self.add_paper("index", "retrieval material")
        index["title"] = "Changed"
        Path(record["index_path"]).write_text(json.dumps(index), encoding="utf-8")
        _, missing = self.add_paper("source", "retrieval material")
        (Path(missing["index_path"]).parent / "layout.txt").unlink()
        result = self.retriever.retrieve("retrieval", ["index", "source"])
        self.assertEqual(result["hits"], [])
        self.assertEqual([r["code"] for r in result["issues"]], ["invalid_material", "material_missing"])

    def test_stale_version_and_wrong_paper_cannot_return_candidates(self):
        self.add_paper("stale", "retrieval material")
        self.papers["stale"]["updated"] = "v2"
        self.add_paper("wrong", "retrieval material")
        self.contexts["wrong"] = self.contexts["stale"]
        result = self.retriever.retrieve("retrieval", ["stale", "wrong"])
        self.assertEqual(result["hits"], [])
        self.assertEqual([r["code"] for r in result["issues"]], ["stale_material", "invalid_material"])

    def test_task_directory_boundary_is_checked(self):
        self.add_paper("a", "retrieval material")
        other = Retriever(self.contexts, self.papers, folder=self.root / "other-task")
        result = other.retrieve("retrieval", ["a"])
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["issues"][0]["code"], "invalid_material")

    def test_table_pages_and_unknown_layout_pages(self):
        self.add_paper("a", "latency body", tables="[Table extraction: PDF page 7, table 1]\nlatency | 32.5")
        hits = self.retriever.retrieve("latency", ["a"])["hits"]
        by_kind = {r["source_kind"]: r for r in hits}
        self.assertIsNone(by_kind["layout"]["page_start"])
        self.assertEqual(by_kind["tables"]["page_start"], 7)
        self.assertNotIn("Table extraction", by_kind["tables"]["text"])

    def test_references_require_explicit_request_and_gaps_are_preserved(self):
        text = "2 Methods\nretrieval material" + GAP + "More retrieval details.\nReferences\nretrieval citation"
        self.add_paper("a", text)
        hits = self.retriever.retrieve("retrieval", ["a"])["hits"]
        self.assertEqual(len(hits), 2)
        self.assertTrue(all("EXCERPT GAP" not in r["text"] and "citation" not in r["text"] for r in hits))
        hits = self.retriever.retrieve("references retrieval", ["a"])["hits"]
        self.assertTrue(any(r["section"] == "References" for r in hits))

    def test_invalid_fragment_excludes_entire_paper_and_duplicate_row_is_not_counted(self):
        index, _ = self.add_paper("a", "retrieval material")
        source = index["sources"]["layout"]
        source["fragments"].append(dict(source["fragments"][0]))
        self.rewrite_index("a", index)
        self.assertEqual(len(self.retriever.retrieve("retrieval", ["a"])["hits"]), 1)
        source["fragments"][1]["end"] = 10000
        self.rewrite_index("a", index)
        result = self.retriever.retrieve("retrieval", ["a"])
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["issues"][0]["code"], "invalid_material")

    def test_retrieval_does_not_mutate_records_or_legacy_selection(self):
        _, record = self.add_paper("a", "2 Methods\nretrieval material\n3 Results\nother result")
        index, texts = load_index(record, folder=self.root)
        before_contexts, before_papers = copy.deepcopy(self.contexts), copy.deepcopy(self.papers)
        selection = select_context(index, texts, "missingterm")
        self.assertTrue(selection["text"])
        self.assertFalse(selection["matched"])
        self.assertEqual(self.retriever.retrieve("missingterm", ["a"])["hits"], [])
        self.assertEqual(select_context(index, texts, "missingterm"), selection)
        self.assertEqual(self.contexts, before_contexts)
        self.assertEqual(self.papers, before_papers)

    def test_invalid_arguments_fail_explicitly(self):
        for args in ((None, []), ("a", "paper-id"), ("a", [""]), ("a", [1]),
                     ("a", [], 0), ("a", [], -1), ("a", [], True), ("a", [], 1.5),
                     ("a", [], 1, "unsupported")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.retriever.retrieve(*args)


if __name__ == "__main__":
    unittest.main()
