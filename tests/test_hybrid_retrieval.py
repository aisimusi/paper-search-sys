"""RRF arithmetic, shared-corpus routing and deterministic FAISS integration."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scholar_daily.embeddings import EmbeddingConfig
from scholar_daily.models import Paper
from scholar_daily.paper_context import load_index, save_index
from scholar_daily.retrieval import Retriever, bm25_retrieve, rrf_fuse
from test_dense_retrieval import FakeEmbedder, HAS_DENSE


def hit(pid, fid="F0", score=1.0, fingerprint="snapshot"):
    return {"paper_id": pid, "source_fingerprint": fingerprint, "fragment_id": fid,
            "score": score, "text": "identical fixture text", "verification_status": "unverified"}


class RRFFusionTests(unittest.TestCase):
    def test_one_based_scores_overlap_and_route_diagnostics(self):
        bm25 = [hit("a", score=100), hit("b", score=3)]
        dense = [hit("b", score=.9), hit("c", score=-.2)]
        result = rrf_fuse(bm25, dense, 10)
        self.assertEqual([h["paper_id"] for h in result], ["b", "a", "c"])
        both, only_bm25, only_dense = result
        self.assertAlmostEqual(both["score"], 1 / 62 + 1 / 61)
        self.assertEqual(both["rrf_score"], both["score"])
        self.assertEqual((both["bm25_rank"], both["dense_rank"]), (2, 1))
        self.assertEqual((both["bm25_score"], both["dense_score"]), (3, .9))
        self.assertEqual(both["retrieval_sources"], ["bm25", "dense"])
        self.assertEqual((only_bm25["bm25_rank"], only_bm25["dense_rank"]), (1, None))
        self.assertIsNone(only_bm25["dense_score"])
        self.assertEqual(only_dense["dense_score"], -.2)
        self.assertEqual(only_dense["retrieval_sources"], ["dense"])
        self.assertEqual(only_dense["score"], 1 / 62)
        self.assertTrue(all(h["method"] == "hybrid" for h in result))

    def test_raw_score_scaling_cannot_change_fused_rank_or_score(self):
        bm25, dense = [hit("a"), hit("b")], [hit("b"), hit("a")]
        original = rrf_fuse(bm25, dense, 2)
        bm25[0]["score"], dense[0]["score"] = 1e30, -1
        scaled = rrf_fuse(bm25, dense, 2)
        self.assertEqual([(h["paper_id"], h["score"]) for h in original],
                         [(h["paper_id"], h["score"]) for h in scaled])

    def test_same_text_and_fragment_id_keep_paper_and_snapshot_provenance(self):
        rows = [hit("b"), hit("a", fingerprint="v2"), hit("a", fingerprint="v1")]
        result = rrf_fuse(rows, [rows[2]], 10)
        self.assertEqual(len(result), 3)
        self.assertEqual({(h["paper_id"], h["source_fingerprint"]) for h in result},
                         {("a", "v1"), ("a", "v2"), ("b", "snapshot")})
        self.assertEqual(result[0]["retrieval_sources"], ["bm25", "dense"])

    def test_ties_use_identity_and_top_k_is_deterministic(self):
        a, b = hit("a"), hit("b")
        self.assertEqual(rrf_fuse([b, a], [a, b], 1)[0]["paper_id"], "a")
        first = rrf_fuse([b, a], [a, b], 2)
        second = rrf_fuse([a, b], [b, a], 2)
        self.assertEqual([h["paper_id"] for h in first], [h["paper_id"] for h in second])
        self.assertEqual(first[0]["score"], first[1]["score"])

    def test_duplicate_route_rows_vote_once_and_keep_first_original_rank(self):
        a, b = hit("a"), hit("b")
        result = rrf_fuse([a, a, b], [a], 10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["score"], 2 / 61)
        self.assertEqual(result[1]["bm25_rank"], 3)
        self.assertEqual(result[1]["score"], 1 / 63)

    def test_empty_routes_custom_k_and_input_immutability(self):
        rows = [hit("a"), hit("b")]
        before = copy.deepcopy(rows)
        self.assertEqual(rrf_fuse([], [], 1), [])
        self.assertEqual(rrf_fuse(rows, [], 2, rrf_k=0)[0]["score"], 1)
        self.assertEqual(rrf_fuse([], rows, 2, rrf_k=10)[1]["score"], 1 / 12)
        self.assertEqual(rows, before)

    def test_invalid_parameters_raise_value_error(self):
        for top_k, rrf_k in ((0, 60), (True, 60), (1.5, 60), (1, -1),
                             (1, True), (1, 2.5), (1, None)):
            with self.subTest(top_k=top_k, rrf_k=rrf_k), self.assertRaises(ValueError):
                rrf_fuse([], [], top_k, rrf_k)


class HybridFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.papers, self.contexts = {}, {}
        self.encoder = FakeEmbedder()
        self.config = EmbeddingConfig(model_id="fixture", revision="fixture-v1", dimension=3)
        self.retriever = Retriever(self.contexts, self.papers, folder=self.root,
            dense_config=self.config, embedder=self.encoder, dense_cache_dir=self.root / "vectors")

    def add_paper(self, pid, text, pages=()):
        paper = Paper(paper_id=pid, source="fixture", title="Fictional hybrid fixture", authors=[],
                      abstract="fixture", published="v1", updated="v1", url="https://example.invalid/", pdf_url="")
        index, record = save_index(self.root, paper, {"layout": text}, pages)
        self.papers[pid], self.contexts[pid] = paper.as_dict(), record
        return index, record

    def search(self, query="alpha", ids=None, top_k=10, **kwargs):
        return self.retriever.retrieve(query, list(self.papers) if ids is None else ids,
                                       top_k, "hybrid", **kwargs)


class HybridAPITests(HybridFixture):
    def setUp(self):
        super().setUp()
        self.dense = Mock(last_stats={})
        self.dense.retrieve.side_effect = lambda q, chunks, k, issues: [
            dict(c, score=.9, method="dense") for c in chunks[:k]]
        self.retriever._dense = self.dense

    def test_default_windows_share_same_validated_chunks_and_load_once(self):
        self.add_paper("a", "alpha English body")
        self.add_paper("b", "beta English body")
        with patch("scholar_daily.retrieval.load_index", wraps=load_index) as loader, patch(
                "scholar_daily.retrieval.bm25_retrieve", wraps=bm25_retrieve) as bm25:
            result = self.search()
        self.assertEqual(loader.call_count, 2)
        self.assertIs(bm25.call_args.args[1], self.dense.retrieve.call_args.args[1])
        self.assertEqual(bm25.call_args.args[2], 20)
        self.assertEqual(self.dense.retrieve.call_args.args[2], 20)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["hits"][0]["score"], 2 / 61)

    def test_custom_window_top_k_and_rrf_k(self):
        for pid in ("a", "b", "c"):
            self.add_paper(pid, "alpha body")
        with patch("scholar_daily.retrieval.bm25_retrieve", wraps=bm25_retrieve) as bm25:
            result = self.search(top_k=1, candidate_k=2, rrf_k=0)
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["hits"][0]["score"], 2)
        self.assertEqual(bm25.call_args.args[2], 2)
        self.assertEqual(self.dense.retrieve.call_args.args[2], 2)
        self.assertNotIn("c", [h["paper_id"] for h in result["hits"]])

    def test_invalid_window_and_k_are_checked_before_empty_query(self):
        for kwargs in ({"candidate_k": 0}, {"candidate_k": True}, {"candidate_k": 2.5},
                       {"candidate_k": 9}, {"rrf_k": -1}, {"rrf_k": True}, {"rrf_k": .5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.search(query="", **kwargs)
        with self.assertRaises(ValueError):
            self.search(top_k=21)
        self.assertEqual(self.search(top_k=21, candidate_k=21)["hits"], [])
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.search(top_k=top_k)

    def test_diagnostics_and_all_original_locators_remain_unverified(self):
        text = "1 Introduction\npreamble\n\f2 Results\nalpha latency -5 ms."
        _, record = self.add_paper("a", text, pages=["layout"])
        result = self.search()
        selected = next(h for h in result["hits"] if h["section"] == "Results")
        baseline = next(h for h in self.retriever.retrieve("alpha", ["a"])["hits"] if h["section"] == "Results")
        for key, value in baseline.items():
            if key not in {"method", "score"}:
                self.assertEqual(selected[key], value)
        self.assertEqual(selected["verification_status"], "unverified")
        self.assertEqual(selected["source_fingerprint"], record["source_fingerprint"])
        self.assertEqual(selected["text"], text[selected["start"]:selected["end"]])
        self.assertEqual(selected["page_start"], 2)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_single_route_absence_is_not_reported_as_failure(self):
        self.add_paper("a", "alpha body")
        dense_only = self.search(query="unmatchedterm")
        self.assertEqual(dense_only["issues"], [])
        self.assertEqual(dense_only["hits"][0]["retrieval_sources"], ["dense"])
        self.dense.retrieve.return_value = []
        self.dense.retrieve.side_effect = None
        bm25_only = self.search()
        self.assertEqual(bm25_only["issues"], [])
        self.assertEqual(bm25_only["hits"][0]["retrieval_sources"], ["bm25"])
        self.assertEqual(self.search(query="unmatchedterm")["hits"], [])
        self.assertEqual(self.search(query="unmatchedterm")["issues"], [])

    def test_empty_query_range_or_invalid_material_does_not_run_dense(self):
        _, record = self.add_paper("a", "alpha body")
        self.assertEqual(self.search(query=" \n")["hits"], [])
        self.assertEqual(self.search(ids=[])["hits"], [])
        Path(record["index_path"]).unlink()
        failed = self.search()
        self.assertEqual(failed["hits"], [])
        self.assertEqual(failed["issues"][0]["code"], "material_missing")
        self.dense.retrieve.assert_not_called()

    def test_partial_material_failure_excludes_invalid_paper_from_both_routes(self):
        self.add_paper("valid", "alpha body")
        _, invalid = self.add_paper("invalid", "alpha body")
        (Path(invalid["index_path"]).parent / "layout.txt").write_text("tampered", encoding="utf-8")
        self.add_paper("stale", "alpha body")
        self.papers["stale"]["updated"] = "v2"
        result = self.search(ids=["unknown", "stale", "invalid", "valid"])
        self.assertEqual([h["paper_id"] for h in result["hits"]], ["valid"])
        self.assertEqual({r["code"] for r in result["issues"]},
                         {"paper_missing", "stale_material", "invalid_material"})
        self.assertEqual([c["paper_id"] for c in self.dense.retrieve.call_args.args[1]], ["valid"])

    def test_dense_initialization_failure_returns_explicit_partial_or_complete_failure(self):
        self.add_paper("a", "alpha body")
        self.retriever._dense = None
        with patch("scholar_daily.dense_retrieval.DenseRetriever", side_effect=RuntimeError("fixture init failure")):
            partial = self.search()
            failed = self.search(query="unmatchedterm")
        self.assertEqual(partial["hits"][0]["retrieval_sources"], ["bm25"])
        self.assertEqual(partial["hits"][0]["method"], "hybrid")
        self.assertEqual(partial["issues"][0]["code"], "dense_initialization_failed")
        self.assertIn("fixture init failure", partial["issues"][0]["message"])
        self.assertEqual(failed["hits"], [])
        self.assertEqual(failed["issues"][0]["code"], "dense_initialization_failed")

    def test_query_failure_retains_dense_error_instead_of_normal_empty_route(self):
        self.add_paper("a", "alpha body")
        def failed(query, chunks, k, issues):
            issues.append({"paper_id": "", "code": "dense_query_failed", "message": "fixture query failure"})
            return []
        self.dense.retrieve.side_effect = failed
        partial = self.search()
        self.assertEqual(partial["hits"][0]["dense_rank"], None)
        self.assertEqual(partial["issues"][0]["code"], "dense_query_failed")
        failed_result = self.search(query="unmatchedterm")
        self.assertEqual(failed_result["hits"], [])
        self.assertTrue(failed_result["issues"])


@unittest.skipUnless(HAS_DENSE, "安装 dense extra 后运行 Hybrid FAISS 测试")
class HybridDenseTests(HybridFixture):
    def test_actual_routes_overlap_and_candidate_window_keeps_disjoint_hits(self):
        for pid, text in (("a", "alpha body"), ("b", "beta body"), ("c", "mixed body"), ("d", "negative body")):
            self.add_paper(pid, text)
        both = self.search(top_k=1)
        self.assertEqual(both["hits"][0]["paper_id"], "a")
        self.assertEqual(both["hits"][0]["retrieval_sources"], ["bm25", "dense"])
        separate = self.search(query="beta", top_k=2, candidate_k=2)
        self.assertEqual([h["paper_id"] for h in separate["hits"]], ["a", "b"])
        self.assertEqual([h["retrieval_sources"] for h in separate["hits"]], [["dense"], ["bm25"]])
        self.assertEqual(separate["hits"][0]["score"], separate["hits"][1]["score"])

    def test_identical_cross_paper_fragments_keep_both_sources_and_stable_order(self):
        self.add_paper("a", "alpha identical body")
        self.add_paper("b", "alpha identical body")
        result = self.search(ids=["b", "a", "a"])
        self.assertEqual([h["paper_id"] for h in result["hits"]], ["a", "b"])
        self.assertEqual(result["hits"][0]["fragment_id"], result["hits"][1]["fragment_id"])
        self.assertEqual(result, self.search(ids=["a", "b"]))

    def test_cache_reuse_across_methods_queries_windows_and_new_retriever(self):
        self.add_paper("a", "alpha body")
        self.add_paper("b", "beta body")
        first = self.search()
        self.retriever.retrieve("beta", ["a", "b"], method="dense")
        self.search(query="beta", top_k=1, candidate_k=1, rrf_k=0)
        self.assertEqual(len(self.encoder.document_calls), 2)
        self.assertTrue(all(p["status"] == "hit" for p in self.retriever.dense_stats["papers"]))
        replacement = FakeEmbedder()
        self.retriever = Retriever(self.contexts, self.papers, folder=self.root, dense_config=self.config,
            embedder=replacement, dense_cache_dir=self.root / "vectors")
        self.assertEqual(first, self.search())
        self.assertEqual(replacement.document_calls, [])

    def test_corrupted_vector_cache_is_rebuilt_by_existing_dense_path(self):
        self.add_paper("a", "alpha body")
        first = self.search()
        folder = Path(self.retriever.dense_stats["papers"][0]["cache_path"])
        (folder / "index.faiss").write_bytes(b"corrupt")
        self.assertEqual(self.search(), first)
        self.assertEqual(self.retriever.dense_stats["papers"][0]["status"], "rebuilt")
        self.assertEqual(len(self.encoder.document_calls), 2)

    def test_real_dense_query_validation_failure_is_partial_then_complete(self):
        self.add_paper("a", "alpha body")
        self.encoder.query_vector = [[0, 0, 0]]
        partial = self.search()
        self.assertEqual(partial["issues"][0]["code"], "dense_query_failed")
        self.assertEqual(partial["hits"][0]["retrieval_sources"], ["bm25"])
        failed = self.search(query="unmatchedterm")
        self.assertEqual(failed["hits"], [])
        self.assertEqual(failed["issues"][0]["code"], "dense_query_failed")

    def test_one_dense_index_failure_preserves_valid_bm25_candidate_and_issue(self):
        self.add_paper("a", "alpha body")
        self.add_paper("b", "beta body")
        self.encoder.invalid_document = [[0, 0, 0]]
        result = self.search()
        self.assertEqual([h["paper_id"] for h in result["hits"]], ["a", "b"])
        self.assertEqual([h["retrieval_sources"] for h in result["hits"]], [["bm25"], ["dense"]])
        self.assertEqual(result["issues"][0]["paper_id"], "a")
        self.assertEqual(result["issues"][0]["code"], "dense_index_failed")

    def test_missing_faiss_reports_failure_and_never_claims_dense_contribution(self):
        self.add_paper("a", "alpha body")
        with patch.dict(sys.modules, {"faiss": None}):
            result = self.search()
        self.assertEqual(result["issues"][0]["code"], "dense_initialization_failed")
        self.assertEqual(result["hits"][0]["retrieval_sources"], ["bm25"])

    def test_original_methods_are_unchanged_after_hybrid(self):
        self.add_paper("a", "alpha body")
        self.add_paper("b", "beta body")
        papers, contexts = copy.deepcopy(self.papers), copy.deepcopy(self.contexts)
        before = {method: self.retriever.retrieve("alpha", ["a", "b"], method=method)
                  for method in ("bm25", "dense")}
        self.search()
        for method, original in before.items():
            result = self.retriever.retrieve("alpha", ["a", "b"], method=method)
            self.assertEqual(result, original)
            self.assertTrue(all("rrf_score" not in h for h in result["hits"]))
        self.assertEqual((self.papers, self.contexts), (papers, contexts))


if __name__ == "__main__":
    unittest.main()
