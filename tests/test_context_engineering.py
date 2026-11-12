import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scholar_daily.agents import SessionBackend, TaskPending
from scholar_daily.contracts import validate_analysis
from scholar_daily.context_evaluation import compare_tasks
from scholar_daily.orchestrator import run_research, task_status
from scholar_daily.paper_context import (GAP, build_index, checked_policy, estimated_tokens,
                                         load_index, locate_quote, save_index, select_context)
from scholar_daily.research_audit import annotate_sample, audit_metrics, audit_sample
from scholar_daily.research_calls import CallLedger, measured_input_usage, read_json, write_json
from scholar_daily.research_memory import bounded_view, memory_snapshot
from scholar_daily.research_state import PLANNER_SCHEMA, checked_limits, initial_state, planner_view
from scholar_daily.research_tools import ResearchTools
from scholar_daily.research_verifier import Verifier, evidence_signature, review_inputs, validate_review
import test_orchestrator as fixtures
from test_orchestrator import HAS_GRAPH, ScriptedBackend, action, planned
from test_research import QUOTE, analysis, paper
from test_verifier import reviewed


def metadata():
    return {"paper_id": "fictional-context-fixture", "title": "Fictional context fixture",
            "updated": "fixture-v1", "abstract": "Fixture abstract", "url": "https://example.invalid/fixture"}


class PaperContextTests(unittest.TestCase):
    def test_offsets_preserve_unicode_and_page_boundaries(self):
        text = "1 Introduction\n前文\n\f2 Results\nThe latency change is -5 ms.\n\nReferences\nFixture citation"
        texts = {"layout": text}
        index = build_index(metadata(), texts, page_sources=["layout"])
        locations = locate_quote(index, texts, "The latency change is -5 ms.")
        self.assertEqual(len(locations), 1)
        row = locations[0]
        self.assertEqual(text[row["start"]:row["end"]], "The latency change is -5 ms.")
        self.assertEqual((row["page_start"], row["page_end"]), (2, 2))
        self.assertEqual(row["sections"][0]["name"], "Results")
        self.assertEqual(locate_quote(index, texts, "The latency change is 5 ms."), [])
        self.assertTrue(row["fragment_ids"])

    def test_unknown_pages_duplicate_locations_and_gap_are_explicit(self):
        quote = "The controlled result applies to the fixed protocol."
        text = quote + "\n\n" + quote + GAP + "Another scope."
        texts = {"layout": text}
        index = build_index(metadata(), texts)
        locations = locate_quote(index, texts, quote)
        self.assertEqual(len(locations), 2)
        self.assertTrue(all(row["page_start"] is None for row in locations))
        self.assertEqual(locate_quote(index, texts, quote + GAP + "Another scope."), [])
        changed = {"layout": text + " Changed."}
        with self.assertRaisesRegex(ValueError, "指纹"):
            locate_quote(index, changed, quote)

    def test_table_page_markers_are_sources_not_quotes(self):
        tables = "[Table extraction: PDF page 7, table 1]\nMethod | AP\nFixture | 32.5\n"
        texts = {"tables": tables}
        index = build_index(metadata(), texts)
        rows = locate_quote(index, texts, "Fixture | 32.5")
        self.assertEqual(rows[0]["page_start"], 7)
        self.assertEqual(rows[0]["source_kind"], "tables")
        selected = select_context(index, texts, "Fixture", require_match=True)
        self.assertIn("Method | AP", selected["text"])
        self.assertNotIn("Table extraction", selected["text"])

    def test_retrieval_skips_bibliography_and_expands_within_budget(self):
        first = "2 Methods\n" + "A neutral sentence in the method section.\n" * 25 + "Latency uses a fixed protocol.\n" + "More neutral method details.\n" * 15
        second = "3 Results\n" + "Held-out result details.\n" * 20
        text = "1 Introduction\nUnrelated introduction.\n" + first + second + "References\n1. Latency citation fixture\n"
        texts = {"layout": text}
        index = build_index(metadata(), texts, fragment_chars=240)
        fragment = select_context(index, texts, "latency", max_fragments=1, max_tokens=2000)
        section = select_context(index, texts, "latency", level="section", max_fragments=1, max_tokens=2000)
        document = select_context(index, texts, "latency", level="document", max_fragments=1, max_tokens=2000)
        self.assertLess(len(fragment["text"]), len(section["text"]))
        self.assertLess(len(section["text"]), len(document["text"]))
        self.assertNotIn("citation fixture", document["text"])
        self.assertIn("citation fixture", select_context(index, texts, "references", level="document", max_tokens=2000)["text"])
        short = select_context(index, texts, "latency", level="document", max_chars=301, max_tokens=70)
        self.assertLessEqual(len(short["text"]), 301)
        self.assertLessEqual(estimated_tokens(short["text"]), 70)
        with self.assertRaisesRegex(ValueError, "未找到"):
            select_context(index, texts, "nonexistenttopic", require_match=True)

    def test_fallback_samples_methods_and_results_and_excludes_reference_continuation(self):
        text = "1 Introduction\n" + "Background text. " * 300 + "\n2 Methods\nFixture mechanism.\n3 Results\nFixture results.\nReferences\n1. Cited fixture\n2. Another cited fixture\n"
        index = build_index(metadata(), {"layout": text}, fragment_chars=200)
        selected = select_context(index, {"layout": text}, "unknown-topic", max_fragments=2)
        self.assertIn("Fixture mechanism", selected["text"])
        self.assertIn("Fixture results", selected["text"])
        self.assertNotIn("cited fixture", selected["text"])

    def test_snapshot_tampering_and_index_tampering_are_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = paper()
            index, record = save_index(root, p, {"layout": p.full_text}, [])
            self.assertEqual(load_index(record, folder=root)[0], index)
            source = Path(record["index_path"]).parent / "layout.txt"
            source.write_text("Tampered source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "指纹"):
                load_index(record, folder=root)
            source.write_text(p.full_text, encoding="utf-8")
            index["sources"]["layout"]["sections"][0]["start"] += 1
            write_json(Path(record["index_path"]), index)
            with self.assertRaisesRegex(ValueError, "指纹"):
                load_index(record, folder=root)

    def test_excerpt_markers_cannot_become_continuous_paper_evidence(self):
        p = paper()
        p.full_text = QUOTE + GAP + "The original text has a separate scope."
        value = analysis(p)
        for claim in value["claims"]:
            claim["anchor_quote"] = p.full_text
        checked = validate_analysis(value, p)
        self.assertEqual(checked["claims"], [])

    def test_policy_rejects_unknown_invalid_and_boolean_limits(self):
        for value in ({"max_context_tokens": True}, {"max_context_tokens": 200}, {"planner_mode": "other"}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                checked_policy(value)


class SemanticLabelTests(unittest.TestCase):
    def item(self):
        return {"item_id": "item", "section": "evidence", "field": "claims.0", "text": "The latency is 5 ms.",
                "kind": "fact", "evidence_ids": ["ER-1"], "source_ids": ["ER-1", "USER_FACTS"],
                "paper_id": "p", "structural_error": ""}

    def test_three_labels_do_not_bypass_structural_or_qualifier_checks(self):
        sources = {"ER-1": {"text": "The latency is 5 ms."}}
        for verdict, expected in (("supported", "supported"), ("contradicted", "contradicted"),
                                  ("partial", "insufficient"), ("unsupported", "insufficient")):
            result = validate_review(reviewed(verdict=verdict), self.item(), sources)
            self.assertEqual(result["support_status"], expected)
            self.assertFalse(result["human_verified"])
        review = reviewed()
        review["claims"][0]["qualifiers_supported"] = "no"
        self.assertEqual(validate_review(review, self.item(), sources)["support_status"], "insufficient")
        item = dict(self.item(), structural_error="Fixture source mismatch")
        self.assertEqual(validate_review(reviewed(verdict="contradicted"), item, sources)["support_status"], "insufficient")
        self.assertEqual(validate_review(reviewed(verdict="contradicted", quote="Fake quotation"), self.item(), sources)["support_status"], "insufficient")


class MemoryAndUsageTests(unittest.TestCase):
    def state(self, root):
        config = fixtures.OrchestratorTests().config(root)
        p = paper()
        p.full_text = QUOTE + "\n" + "Irrelevant fixture appendix. " * 1500
        return initial_state("fixture-memory", "保留方法、事实和待办", config,
                             {"papers": [p.as_dict()], "facts": {"precision": "FP16"},
                              "context_policy": {"max_context_tokens": 256}}, checked_limits())

    def test_compaction_preserves_pending_facts_ids_and_unfinished_work(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = self.state(root)
            pid = next(iter(state["papers"]))
            state["deliverables"] = ["comparison", "experiment"]
            state["pending"] = {"kind": "user", "fields": ["deployment.memory_gb"]}
            state["evidence"] = {"ER-fixture": {"paper_id": pid, "kind": "method", "text": "Fixed fixture mechanism",
                                                  "anchor_quote": QUOTE, "locations": [], "index_path": "fixture-index.json"}}
            state["steps"] = [{"tool": "read_papers", "status": "failed", "error": "Fixture gap", "action_key": "fixture-key"}]
            memory = memory_snapshot(state, root / "state.json")
            view = memory["memory"]
            self.assertEqual(view["facts"], state["facts"])
            self.assertEqual(view["pending"], state["pending"])
            self.assertEqual(view["unfinished_tasks"]["deliverables"], state["deliverables"])
            self.assertEqual(view["unfinished_tasks"]["failed_steps"][0]["action_key"], "fixture-key")
            self.assertEqual(view["evidence"][0]["evidence_id"], "ER-fixture")
            self.assertEqual(view["evidence"][0]["locator"]["index_path"], "fixture-index.json")
            self.assertTrue(view["memory_budget"]["overflow"])
            self.assertGreater(memory["compression"]["reduction_fraction"], 0.8)

    def test_memory_marks_reviews_stale_after_facts_change(self):
        with tempfile.TemporaryDirectory() as d:
            state = self.state(Path(d))
            pid = next(iter(state["papers"]))
            state["evidence"] = {"ER-fixture": {"paper_id": pid, "kind": "method", "text": "Fixture method", "anchor_quote": QUOTE}}
            state["verification"]["evidence"] = {"status": "checked", "signature": evidence_signature(state),
                "items": [{"evidence_ids": ["ER-fixture"], "passed": True, "support_status": "supported"}]}
            state["facts"]["precision"] = "FP32"
            memory = bounded_view(state, planner_view(state))
            self.assertEqual(memory["evidence"][0]["model_support_status"], "stale")
            self.assertFalse(memory["evidence"][0]["review_current_and_passed"])

    def test_input_usage_includes_cache_without_codex_double_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "events.jsonl"
            path.write_text(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 20, "cached_input_tokens": 9}}), encoding="utf-8")
            self.assertEqual(measured_input_usage(root), 20)
            path.write_text(json.dumps({"type": "assistant/message", "data": {"usage": {"inputTokens": 11, "cacheReadTokens": 9}}}), encoding="utf-8")
            self.assertEqual(measured_input_usage(root), 20)
            path.write_text(json.dumps({"type": "assistant/message", "data": {"usage": {"inputTokens": 11}}}), encoding="utf-8")
            self.assertIsNone(measured_input_usage(root))
            events = [{"type": "assistant/message", "data": {"usage": {"inputTokens": 11, "cacheReadTokens": 9}}},
                      {"type": "assistant/message", "data": {"usage": {"outputTokens": 4}}}]
            path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            self.assertIsNone(measured_input_usage(root))

    def test_resuming_pending_call_does_not_double_count_context(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = CallLedger(root, SessionBackend(), checked_limits())
            with self.assertRaises(TaskPending) as waiting:
                ledger.call("fixture", "planner", "Frozen prompt", PLANNER_SCHEMA)
            before = ledger.context_metrics()
            self.assertEqual(before["attempts"], 1)
            self.assertEqual(before["unmeasured_attempts"], 1)
            response = Path(waiting.exception.task["response_path"])
            write_json(response, planned("finish", ["discovery"]))
            ledger.call("fixture", "planner", "Changed replay prompt", PLANNER_SCHEMA)
            self.assertEqual(ledger.context_metrics(), before)
            ledger.close()
            ledger = CallLedger(root, Mock(), checked_limits())
            ledger.call("fixture", "planner", "Another replay prompt", PLANNER_SCHEMA)
            self.assertEqual(ledger.context_metrics(), before)
            ledger.backend.run.assert_not_called()
            ledger.close()

    def test_old_call_tables_add_context_columns_without_losing_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            connection = sqlite3.connect(str(root / "calls.sqlite3"))
            connection.execute("CREATE TABLE calls (name TEXT,attempt INTEGER,role TEXT,input_hash TEXT,status TEXT,reservation INTEGER,tokens INTEGER,folder TEXT,error TEXT, PRIMARY KEY(name,attempt))")
            connection.execute("INSERT INTO calls VALUES ('old',1,'planner','hash','failed',16000,NULL,'fixture','error')")
            connection.commit()
            connection.close()
            ledger = CallLedger(root, Mock(), checked_limits())
            self.assertEqual(ledger.budget()["model_calls"], 1)
            self.assertEqual(ledger.context_metrics()["estimate_missing_attempts"], 1)
            ledger.close()


@unittest.skipUnless(HAS_GRAPH, "需安装 orchestrator extra")
class ContextIntegrationTests(unittest.TestCase):
    def fixture_run(self, root, mode="compact", p=None, goal="分析虚构上下文夹具"):
        p = p or paper()
        backend = ScriptedBackend([planned("analyze_papers", ["analysis"]), planned("finish", ["analysis"], finish_status="complete")], [p])
        config = fixtures.OrchestratorTests().config(root)
        result = run_research(config, goal, inputs={"papers": [p.as_dict()], "context_policy": {"planner_mode": mode}}, backend=backend)
        return result, backend

    def test_exports_locators_memory_usage_and_provisional_human_sample(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = paper()
            result, backend = self.fixture_run(root, p=p)
            folder = Path(result["task_dir"])
            state = task_status(folder)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["context_metrics"]["attempts"], len(backend.calls))
            self.assertEqual(result["context_metrics"]["unmeasured_attempts"], len(backend.calls))
            self.assertGreater(result["context_metrics"]["estimated_input_tokens"], 0)
            evidence = next(iter(state["evidence"].values()))
            location = evidence["locations"][0]
            snapshot = Path(evidence["index_path"]).parent / (location["source_kind"] + ".txt")
            self.assertEqual(snapshot.read_text(encoding="utf-8")[location["start"]:location["end"]], QUOTE)
            self.assertIsNone(location["page_start"])
            self.assertEqual(read_json(folder / "memory.json")["memory"]["task_id"], state["task_id"])
            sample = read_json(folder / "audit-sample.json")
            self.assertEqual(sample, audit_sample(read_json(folder / "state.json")))
            self.assertTrue(sample["items"])
            self.assertIsNone(audit_metrics(sample)["agreement_fraction"])
            for row in sample["items"]:
                row.update(human_label="insufficient", reviewer="fixture human reviewer", notes="Fixture disagreement")
            labels = root / "labels.json"
            write_json(labels, sample)
            metrics = annotate_sample(folder, labels)
            self.assertEqual(metrics["agreement_fraction"], 0)
            self.assertTrue(metrics["disagreements"])
            unchanged = read_json(folder / "state.json")
            self.assertTrue(unchanged["verification"]["evidence"]["passed"])
            run_research(folder=folder, backend=backend)
            self.assertEqual(read_json(folder / "state.json")["human_audit"]["status"], "checked")

    def test_human_import_rejects_mutated_frozen_evidence_and_stale_sample(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result, _ = self.fixture_run(root)
            folder = Path(result["task_dir"])
            sample = read_json(folder / "audit-sample.json")
            sample["items"][0]["claim"] = "Tampered claim"
            write_json(folder / "audit-sample.json", sample)
            labels = root / "labels.json"
            write_json(labels, sample)
            with self.assertRaisesRegex(ValueError, "不能修改"):
                annotate_sample(folder, labels)
            sample["sample_id"] = "0" * 64
            write_json(labels, sample)
            with self.assertRaisesRegex(ValueError, "sample_id"):
                annotate_sample(folder, labels)

    def test_reading_escalation_reuses_original_snapshot_after_cache_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixtures.OrchestratorTests().config(root)
            p = paper()
            text = "1 Introduction\n" + QUOTE + "\n2 Methods\n" + "Neutral method details.\n" * 160 + "Latency uses a fixed batch size.\n3 Results\n" + "Controlled result details.\n" * 60
            cache = root / "cache/papers/fixture.txt"
            cache.parent.mkdir(parents=True)
            cache.write_text(text, encoding="utf-8")
            p.full_text_path = str(cache)
            state = initial_state("fixture-reading", "核对实验协议", config, {"papers": [p.as_dict()]}, checked_limits())
            ledger = CallLedger(root, Mock(), checked_limits())
            tools = ResearchTools(ledger)
            first = tools.read_papers(state, action("read_papers", query="latency"), root, "first")
            state.update({k: v for k, v in first.items() if k not in {"invalidate_artifacts", "artifacts"}})
            cache.write_text("Changed external cache that must not replace task material", encoding="utf-8")
            second = tools.read_papers(state, action("read_papers", query="latency"), root, "second")
            self.assertEqual(first["paper_contexts"][p.paper_id]["selection"]["level"], "fragment")
            self.assertEqual(second["paper_contexts"][p.paper_id]["selection"]["level"], "section")
            self.assertGreater(len(second["papers"][p.paper_id]["full_text"]), len(first["papers"][p.paper_id]["full_text"]))
            self.assertNotIn("Changed external cache", second["papers"][p.paper_id]["full_text"])
            self.assertEqual(first["paper_contexts"][p.paper_id]["source_fingerprint"], second["paper_contexts"][p.paper_id]["source_fingerprint"])
            ledger.close()

    def test_replayed_analysis_freezes_source_after_pending_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = fixtures.OrchestratorTests().config(root)
            p = paper()
            cache = root / "cache/papers/fixture.txt"
            cache.parent.mkdir(parents=True)
            cache.write_text(QUOTE, encoding="utf-8")
            p.full_text_path = str(cache)
            state = initial_state("fixture-frozen-analysis", "分析虚构材料", config, {"papers": [p.as_dict()]}, checked_limits())
            ledger = CallLedger(root, SessionBackend(), checked_limits())
            tools = ResearchTools(ledger)
            with self.assertRaises(TaskPending) as waiting:
                tools.analyze_papers(state, action("analyze_papers"), root, "fixture")
            cache.write_text("External cache changed while model call waits", encoding="utf-8")
            write_json(Path(waiting.exception.task["response_path"]), {"analyses": [analysis(p)]})
            with self.assertRaises(TaskPending) as explaining:
                tools.analyze_papers(state, action("analyze_papers"), root, "fixture")
            from test_research import reading_response
            task = explaining.exception.task
            write_json(Path(task["response_path"]), reading_response(Path(task["prompt_path"]).read_text(encoding="utf-8"),
                                                                     read_json(Path(task["schema_path"]))))
            result = tools.analyze_papers(state, action("analyze_papers"), root, "fixture")
            self.assertTrue(result["evidence"])
            row = next(iter(result["evidence"].values()))
            self.assertEqual(len(row["locations"]), 1)
            self.assertEqual(ledger.context_metrics()["attempts"], 2)
            ledger.close()

    def test_source_snapshot_changes_fail_review_before_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result, _ = self.fixture_run(root)
            state = read_json(Path(result["state_path"]))
            row = next(iter(state["evidence"].values()))
            source = Path(row["index_path"]).parent / "layout.txt"
            source.write_text("Tampered original fixture source", encoding="utf-8")
            payload = review_inputs(state, {"phase": "evidence"})
            self.assertTrue(all(item["structural_error"] for item in payload["items"]))
            ledger = Mock()
            reviewed_state = Verifier(ledger).review(state, {"phase": "evidence"})
            self.assertFalse(reviewed_state["passed"])
            ledger.call.assert_not_called()

    def test_quality_comparison_requires_matched_tasks_and_human_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = paper()
            full, _ = self.fixture_run(root, mode="full", p=p)
            compact, _ = self.fixture_run(root, mode="compact", p=p)
            output = root / "comparison"
            payload = compare_tasks([full["task_dir"]], [compact["task_dir"]], output)
            self.assertEqual(payload["summary"]["human_quality_status"], "awaiting_human_ratings")
            self.assertIsNone(payload["summary"]["mean_human_quality_delta"])
            template = read_json(output / "quality-template.json")
            for row in template["items"]:
                row.update(criteria={criterion: 1 for criterion in row["criteria"]}, reviewer="fixture human reviewer")
            labels = root / "quality-ratings.json"
            write_json(labels, template)
            checked = compare_tasks([full["task_dir"]], [compact["task_dir"]], output, ratings=labels)
            self.assertEqual(checked["summary"]["human_quality_status"], "checked")
            self.assertEqual(checked["summary"]["mean_human_quality_delta"], 0)
            self.assertIsNone(checked["summary"]["human_quality_bootstrap_95_interval"])
            other, _ = self.fixture_run(root, p=p, goal="Different goal")
            with self.assertRaisesRegex(ValueError, "不一致"):
                compare_tasks([full["task_dir"]], [other["task_dir"]], output)


if __name__ == "__main__":
    unittest.main()
