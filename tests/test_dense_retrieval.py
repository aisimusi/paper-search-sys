"""Deterministic embeddings with real FAISS; no model downloads or network calls."""
import builtins
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scholar_daily._compat import FileLock
from scholar_daily.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from scholar_daily.models import Paper
from scholar_daily.paper_context import save_index
from scholar_daily.retrieval import Retriever

HAS_DENSE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "faiss"))


class EmbeddingTests(unittest.TestCase):
    def test_config_rejects_invalid_values(self):
        for kwargs in ({"dimension": 0}, {"dimension": True}, {"max_length": -1},
                       {"batch_size": 1.5}, {"model_id": ""}, {"revision": ""},
                       {"dtype": "int8"}, {"query_prefix": None}, {"local_files_only": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EmbeddingConfig(**kwargs)

    def test_local_model_files_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            path = model / "model.safetensors"
            path.write_bytes(b"fixture-v1")
            config = EmbeddingConfig(model_id=str(model))
            first = config.signature()
            path.write_bytes(b"fixture-v2")
            self.assertNotEqual(first, config.signature())

    def test_lazy_loader_applies_prefixes_limits_and_no_hidden_prompt(self):
        model = Mock(max_seq_length=512)
        model.get_embedding_dimension.return_value = 384
        factory = Mock(return_value=model)
        encoder = SentenceTransformerEmbedder(EmbeddingConfig(max_length=128))
        self.assertIsNone(encoder._model)
        self.assertEqual(encoder.encode([]), [])
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
            encoder.encode(["中文问题"], query=True)
            self.assertEqual(model.encode.call_args.args[0], ["query: 中文问题"])
            self.assertEqual(model.encode.call_args.kwargs["prompt"], "")
            encoder.encode(["English passage"], query=False)
            self.assertEqual(model.encode.call_args.args[0], ["passage: English passage"])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(model.max_seq_length, 128)
        self.assertFalse(factory.call_args.kwargs["trust_remote_code"])

    def test_loader_rejects_wrong_dimension_long_inputs_and_floating_revision(self):
        for config, dimension, limit in ((EmbeddingConfig(), 42, 512),
                                         (EmbeddingConfig(max_length=513), 384, 512),
                                         (EmbeddingConfig(revision="main"), 384, 512)):
            model = Mock(max_seq_length=limit)
            model.get_embedding_dimension.return_value = dimension
            with self.subTest(config=config), patch.dict(sys.modules, {
                    "sentence_transformers": SimpleNamespace(SentenceTransformer=Mock(return_value=model))}):
                with self.assertRaises(ValueError):
                    SentenceTransformerEmbedder(config).encode(["text"])
                model.encode.assert_not_called()
        with self.assertRaises(ValueError):
            SentenceTransformerEmbedder(EmbeddingConfig()).encode([""])


class FakeEmbedder:
    def __init__(self):
        self.calls = []
        self.query_vector = [[3.0, 0.0, 0.0]]
        self.invalid_document = None

    def encode(self, texts, *, query=False):
        self.calls.append((query, list(texts)))
        if query:
            return self.query_vector
        if self.invalid_document is not None and any("alpha" in text for text in texts):
            return self.invalid_document
        return [[0, 4, 0] if "beta" in text else [-2, 0, 0] if "negative" in text
                else [3, 4, 0] if "mixed" in text else [3, 0, 0] if "alpha" in text
                else [.2, 0, 1] for text in texts]

    @property
    def document_calls(self):
        return [texts for query, texts in self.calls if not query]


@unittest.skipUnless(HAS_DENSE, "安装 dense extra 后运行 FAISS 测试")
class DenseRetrievalTests(unittest.TestCase):
    def setUp(self):
        import faiss
        import numpy as np
        self.faiss, self.np = faiss, np
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.papers, self.contexts = {}, {}
        self.config = EmbeddingConfig(model_id="fixture", revision="fixture-v1", dimension=3)
        self.encoder = FakeEmbedder()
        self.retriever = self.make_retriever()

    def make_retriever(self, *, config=None, encoder=None):
        return Retriever(self.contexts, self.papers, folder=self.root,
                         dense_config=config or self.config,
                         embedder=encoder if encoder is not None else self.encoder,
                         dense_cache_dir=self.root / "vectors")

    def add_paper(self, pid, text, *, version="v1", pages=()):
        paper = Paper(paper_id=pid, source="fixture", title="Fictional dense fixture", authors=[],
                      abstract="fixture", published="v1", updated=version, url="https://example.invalid/", pdf_url="")
        index, record = save_index(self.root, paper, {"layout": text}, pages)
        self.papers[pid], self.contexts[pid] = paper.as_dict(), record
        return index, record

    def search(self, ids=None, k=10, query="semantic question", retriever=None):
        return (retriever or self.retriever).retrieve(query, list(self.papers) if ids is None else ids, k, "dense")

    def cache_folder(self):
        return Path(self.retriever.dense_stats["papers"][0]["cache_path"])

    def test_single_paper_normalized_cosine_order_and_top_k(self):
        self.add_paper("a", "1 Alpha\nalpha\n2 Mixed\nmixed\n3 Beta\nbeta\n4 Negative\nnegative")
        result = self.search(k=20)
        self.assertEqual(result["issues"], [])
        self.assertEqual([h["section"] for h in result["hits"]], ["Alpha", "Mixed", "Beta", "Negative"])
        self.np.testing.assert_allclose([h["score"] for h in result["hits"]], [1, .6, 0, -1], atol=1e-6)
        self.assertEqual(len(self.search(k=2)["hits"]), 2)
        self.assertEqual(self.encoder.query_vector, [[3.0, 0.0, 0.0]])

    def test_cross_paper_scope_duplicates_and_stable_ties(self):
        self.add_paper("a", "alpha English body")
        self.add_paper("b", "beta English body")
        self.add_paper("c", "alpha English body")
        result = self.search(ids=["c", "b", "a", "a"], k=2)
        self.assertEqual([h["paper_id"] for h in result["hits"]], ["a", "c"])
        self.assertEqual(self.search(ids=["b"])["hits"][0]["paper_id"], "b")
        self.assertEqual(len(self.encoder.document_calls), 3)

    def test_chinese_and_non_ascii_query_preserve_english_offsets_and_pages(self):
        text = "1 Introduction\npreamble\n\f2 Results\nalpha English measurement -5 ms."
        index, record = self.add_paper("a", text, pages=["layout"])
        for query in ("中文语义问题", "مفهوم"):
            result = self.search(query=query)
            hit = result["hits"][0]
            self.assertEqual(hit["text"], text[hit["start"]:hit["end"]])
            self.assertEqual((hit["page_start"], hit["page_end"]), (2, 2))
            self.assertEqual(hit["source_fingerprint"], record["source_fingerprint"])
            self.assertEqual(hit["source_sha256"], index["sources"]["layout"]["sha256"])
            self.assertEqual(hit["verification_status"], "unverified")
            self.assertEqual(hit["method"], "dense")
            self.assertNotIn("evidence_id", hit)
            self.assertEqual(json.loads(json.dumps(hit)), hit)
            self.assertIn((True, [query]), self.encoder.calls)

    def test_empty_input_and_empty_sources_do_not_initialize_dense(self):
        self.add_paper("empty", "")
        self.assertEqual(self.search()["hits"], [])
        self.assertEqual(self.search(ids=[])["hits"], [])
        self.assertEqual(self.search(query=" \n ")["hits"], [])
        self.assertIsNone(self.retriever._dense)
        self.assertEqual(self.encoder.calls, [])

    def test_cache_is_reused_in_memory_and_after_new_retriever(self):
        self.add_paper("a", "alpha body")
        first = self.search()
        folder = self.cache_folder()
        self.assertTrue((folder / "index.faiss").is_file())
        self.assertTrue((folder / "manifest.json").is_file())
        second = self.search()
        self.assertEqual(first, second)
        self.assertEqual(self.retriever.dense_stats["papers"][0]["status"], "hit")
        replacement = FakeEmbedder()
        fresh = self.make_retriever(encoder=replacement)
        self.assertEqual(self.search(retriever=fresh), first)
        self.assertEqual(replacement.document_calls, [])
        self.assertEqual(len(self.encoder.document_calls), 1)

    def test_changed_material_rebuilds_only_affected_paper(self):
        self.add_paper("a", "alpha body")
        self.add_paper("b", "beta body")
        self.search()
        self.add_paper("a", "alpha updated body", version="v2")
        result = self.search()
        self.assertEqual(result["issues"], [])
        self.assertEqual({p["paper_id"]: p["status"] for p in self.retriever.dense_stats["papers"]},
                         {"a": "rebuilt", "b": "hit"})
        self.assertEqual(len(self.encoder.document_calls), 3)

    def test_embedding_model_revision_and_preprocessing_changes_invalidate_cache(self):
        self.add_paper("a", "alpha body")
        self.search()
        for changes in ({"model_id": "different"}, {"revision": "fixture-v2"},
                        {"document_prefix": "document: "}, {"max_length": 256}, {"dtype": "float16"}):
            encoder = FakeEmbedder()
            fresh = self.make_retriever(config=replace(self.config, **changes), encoder=encoder)
            self.assertTrue(self.search(retriever=fresh)["hits"])
            self.assertEqual(len(encoder.document_calls), 1)

    def test_malformed_query_vectors_fail_without_document_indexing(self):
        self.add_paper("a", "alpha body")
        for values in ([[1, 2]], [[0, 0, 0]], [[float("nan"), 0, 0]],
                       [[float("inf"), 0, 0]], [[1, 0, 0], [1, 0, 0]], [1, 0, 0],
                       [[complex(1, 2), 0, 0]], None):
            self.encoder.query_vector = values
            with self.subTest(values=values):
                result = self.search()
                self.assertEqual(result["hits"], [])
                self.assertEqual(result["issues"][0]["code"], "dense_query_failed")
        self.assertEqual(self.encoder.document_calls, [])

    def test_invalid_document_vectors_fail_per_paper_and_can_recover(self):
        self.add_paper("a", "alpha body")
        self.add_paper("b", "beta body")
        for values in ([[0, 0, 0]], [[1, 2]], [[float("nan"), 0, 1]], [[1, 0, 0], [1, 0, 0]]):
            self.encoder.invalid_document = values
            with self.subTest(values=values):
                result = self.search()
                self.assertEqual([h["paper_id"] for h in result["hits"]], ["b"])
                self.assertEqual(result["issues"][0]["code"], "dense_index_failed")
        self.encoder.invalid_document = None
        self.assertEqual(len(self.search()["hits"]), 2)

    def test_local_weight_change_invalidates_persistent_cache(self):
        model = self.root / "local-model"
        model.mkdir()
        weights = model / "model.safetensors"
        weights.write_bytes(b"fixture-v1")
        config = replace(self.config, model_id=str(model))
        self.add_paper("a", "alpha body")
        self.search(retriever=self.make_retriever(config=config))
        weights.write_bytes(b"fixture-v2")
        fresh = self.make_retriever(config=config)
        self.assertTrue(self.search(retriever=fresh)["hits"])
        self.assertEqual(fresh.dense_stats["papers"][0]["status"], "rebuilt")
        self.assertEqual(len(self.encoder.document_calls), 2)

    def test_busy_cache_lock_returns_explicit_issue_without_replacing_files(self):
        self.add_paper("a", "alpha body")
        expected = self.search()
        folder = self.cache_folder()
        binary = (folder / "index.faiss").read_bytes()
        with FileLock(folder / ".lock"):
            result = self.search()
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["issues"][0]["code"], "dense_index_failed")
        self.assertEqual((folder / "index.faiss").read_bytes(), binary)
        self.assertEqual(self.search(), expected)
        self.assertEqual(len(self.encoder.document_calls), 1)

    def test_index_and_manifest_corruption_rebuilds_without_stale_mapping(self):
        self.add_paper("a", "alpha body")
        expected = self.search()
        folder = self.cache_folder()
        (folder / "index.faiss").write_bytes(b"corrupt")
        self.assertEqual(self.search(), expected)
        (folder / "manifest.json").write_text("invalid JSON", encoding="utf-8")
        self.assertEqual(self.search(), expected)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        manifest["identity"]["fragments"][0]["fragment_id"] = "incorrect"
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.search(), expected)
        self.assertEqual(len(self.encoder.document_calls), 4)

    def test_rechecks_type_dimension_count_and_norm_even_with_matching_checksum(self):
        self.add_paper("a", "alpha body")
        expected = self.search()
        folder = self.cache_folder()
        indexes = [self.faiss.IndexFlatL2(3), self.faiss.IndexFlatIP(2),
                   self.faiss.IndexFlatIP(3), self.faiss.IndexFlatIP(3)]
        indexes[3].add(self.np.zeros((1, 3), dtype="float32"))
        for index in indexes:
            binary = self.faiss.serialize_index(index).tobytes()
            (folder / "index.faiss").write_bytes(binary)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            manifest["index_sha256"] = hashlib.sha256(binary).hexdigest()
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(index=index):
                self.assertEqual(self.search(), expected)

    def test_source_tampering_stale_material_and_missing_indexes_are_not_served_from_cache(self):
        self.add_paper("valid", "beta body")
        _, changed = self.add_paper("changed", "alpha body")
        _, missing = self.add_paper("missing", "alpha body")
        self.add_paper("stale", "alpha body")
        self.search()
        (Path(changed["index_path"]).parent / "layout.txt").write_text("tampered", encoding="utf-8")
        Path(missing["index_path"]).unlink()
        self.papers["stale"]["updated"] = "v2"
        result = self.search(ids=list(self.papers) + ["unknown"])
        self.assertEqual([h["paper_id"] for h in result["hits"]], ["valid"])
        self.assertEqual({r["code"] for r in result["issues"]},
                         {"invalid_material", "material_missing", "stale_material", "paper_missing"})
        self.assertEqual(len(self.encoder.document_calls), 4)

    def test_interrupted_manifest_write_returns_error_then_rebuilds(self):
        self.add_paper("a", "alpha body")
        with patch("scholar_daily.dense_retrieval.atomic_text", side_effect=OSError("fixture write failure")):
            result = self.search()
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["issues"][0]["code"], "dense_index_failed")
        self.assertIn("fixture write failure", result["issues"][0]["message"])
        self.assertTrue(self.search()["hits"])
        self.assertEqual(len(self.encoder.document_calls), 2)
        self.assertEqual(list((self.root / "vectors").rglob(".index-*")), [])

    def test_missing_optional_dependency_is_explicit_and_bm25_stays_usable(self):
        self.add_paper("a", "alpha body")
        with patch.dict(sys.modules, {"faiss": None}):
            result = self.search()
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["issues"][0]["code"], "dense_initialization_failed")
        self.assertTrue(self.retriever.retrieve("alpha", ["a"])["hits"])

    def test_bm25_never_imports_model_or_vector_dependencies(self):
        self.add_paper("a", "alpha body")
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "faiss", "numpy", "sentence_transformers"}:
                raise AssertionError("BM25 imported " + name)
            return original(name, *args, **kwargs)
        before = copy.deepcopy((self.papers, self.contexts))
        with patch("builtins.__import__", side_effect=guarded):
            retriever = Retriever(self.contexts, self.papers, folder=self.root, dense_config={"invalid": True})
            self.assertTrue(retriever.retrieve("alpha", ["a"])["hits"])
        self.assertEqual((self.papers, self.contexts), before)
        self.assertIsNone(retriever._dense)


if __name__ == "__main__":
    unittest.main()
