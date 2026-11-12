import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from scholar_daily.contracts import validate_analysis
from scholar_daily.agents import SessionBackend
from scholar_daily.fulltext import figure_page_numbers, make_excerpt
from scholar_daily.paper_context import build_index, locate_quote, select_reading_context
from scholar_daily.paper_explanations import (EXPLANATION_SCHEMA, content_gaps, empty_explanation,
    explanation_review_items, validate_explanation, validate_explanation_batch)
from scholar_daily.pipeline import run_pipeline
from scholar_daily.quality_gate import evaluate_gate
from scholar_daily.report import domain_evidence, render_evidence, render_report
from scholar_daily.models import DomainResult, Paper
from scholar_daily.research_verifier import VERIFICATION_SCHEMA, validate_review
from test_research import FakeBackend, QUOTE, analysis, explanation, paper
import test_research as fixtures


class ExplanationContractTests(unittest.TestCase):
    def setUp(self):
        self.analysis = validate_analysis(analysis(paper()), paper())
        self.value = explanation(self.analysis)

    def test_unknown_cross_paper_and_uncited_prose_are_rejected(self):
        for ids in (["ER-unknown"], [], self.value["novelty"]["evidence_ids"] * 2):
            value = copy.deepcopy(self.value)
            value["novelty"]["evidence_ids"] = ids
            with self.assertRaises(ValueError):
                validate_explanation(value, self.analysis)
        value = copy.deepcopy(self.value)
        value["paper_id"] = "other-paper"
        with self.assertRaisesRegex(ValueError, "其他论文"):
            validate_explanation(value, self.analysis)

    def test_missing_module_details_remain_gaps_and_do_not_become_facts(self):
        self.value["modules"][0]["inputs"] = {"text": "", "evidence_ids": []}
        self.value["modules"][0]["rationale"] = {"text": "", "evidence_ids": []}
        self.value["modules"][0]["rationale_basis"] = "unknown"
        result = validate_explanation(self.value, self.analysis)
        self.assertEqual({g["field"] for g in content_gaps(result)}, {"modules.0.inputs", "modules.0.rationale"})
        self.assertNotIn("设计理由", "\n".join(row["text"] for row in explanation_review_items(result)))

    def test_inference_is_explicit_and_exact_rendered_text_is_reviewed(self):
        self.value["modules"][0]["rationale_basis"] = "inference"
        rows = list(explanation_review_items(self.value))
        rationale = next(r for r in rows if r["field"].endswith("rationale"))
        self.assertEqual(rationale["kind"], "inference")
        self.assertIn("分析推断", rationale["text"])
        self.assertIn(self.value["modules"][0]["name"], rationale["text"])
        self.assertFalse(content_gaps(self.value))

    def test_batch_cannot_omit_or_duplicate_papers(self):
        for values in ([], [self.value, self.value]):
            with self.assertRaisesRegex(ValueError, "逐篇覆盖"):
                validate_explanation_batch({"explanations": values}, [self.analysis])

    def test_placeholder_text_and_empty_module_name_are_rejected(self):
        self.value["novelty"]["text"] = "待确认"
        with self.assertRaisesRegex(ValueError, "占位"):
            validate_explanation(self.value, self.analysis)
        self.value = explanation(self.analysis)
        self.value["modules"][0]["name"] = "  "
        with self.assertRaises(ValueError):
            validate_explanation(self.value, self.analysis)

    def test_fields_cannot_borrow_other_fields_citations_or_user_facts(self):
        self.value["baseline"]["evidence_ids"] = ["ER-second"]
        row = next(explanation_review_items(self.value))
        item = dict(row, item_id="fixture", section="explanation", paper_id=self.value["paper_id"],
                    source_ids=row["evidence_ids"] + ["USER_FACTS"], structural_error="")
        sources = {sid: {"text": QUOTE} for sid in item["source_ids"]}
        first = row["evidence_ids"][0]
        reviewed = {"item_id": "fixture", "coverage_complete": "yes", "claims": [{"span": item["text"],
                    "verdict": "supported", "qualifiers_supported": "yes", "citations": [{"source_id": first, "quote": QUOTE}],
                    "reason": "controlled fixture"}], "missing_information": [], "recommended_action": "none", "query": "", "required_fields": []}
        self.assertFalse(validate_review(reviewed, item, sources)["passed"])
        reviewed["claims"][0]["citations"].append({"source_id": "ER-second", "quote": QUOTE})
        self.assertTrue(validate_review(reviewed, item, sources)["passed"])
        item["citation_scopes"] = []
        reviewed["claims"][0]["citations"] = [{"source_id": "USER_FACTS", "quote": QUOTE}]
        self.assertFalse(validate_review(reviewed, item, sources)["passed"])

    def test_missing_baseline_routes_to_specific_reading_instead_of_completion(self):
        self.value["baseline"] = {"text": "", "evidence_ids": []}
        state = {"analyses": {self.analysis["paper_id"]: self.analysis}, "paper_explanations": {self.analysis["paper_id"]: self.value},
                 "steps": [], "limits": {"max_steps": 20, "max_review_rounds": 3}, "gate_history": [], "deliverables": ["analysis"],
                 "artifacts": {"analysis": {}}, "stop_reason": ""}
        budget = {"model_calls": 1, "max_model_calls": 30, "charged_tokens": 100, "token_budget": 10000}
        gate = evaluate_gate(state, {"phase": "final", "desired_status": "complete"}, {"status": "checked", "items": []}, budget)
        self.assertEqual(gate["decision"], "read_more")
        self.assertEqual(gate["issues"][0]["field"], "baseline")
        self.assertIn("related work", gate["suggestions"][0]["query"])

    def test_supported_facts_with_clarity_repair_cannot_pass_delivery(self):
        row = next(explanation_review_items(self.value))
        item = dict(row, item_id="clarity", section="explanation", paper_id=self.value["paper_id"],
                    source_ids=row["evidence_ids"], structural_error="")
        sources = {sid: {"text": QUOTE} for sid in item["source_ids"]}
        review = {"item_id": "clarity", "coverage_complete": "yes", "claims": [{"span": item["text"],
                  "verdict": "supported", "qualifiers_supported": "yes", "citations": [
                      {"source_id": sid, "quote": QUOTE} for sid in item["source_ids"]], "reason": "事实有依据"}],
                  "missing_information": ["‘对应结果’未说明对应哪组对照，须写出比较对象"],
                  "recommended_action": "repair", "query": "说明比较对象", "required_fields": []}
        checked = validate_review(review, item, sources)
        self.assertTrue(all(claim["passed"] for claim in checked["claims"]))
        self.assertEqual(checked["support_status"], "supported")
        self.assertFalse(checked["passed"])
        state = {"steps": [], "gate_history": [], "deliverables": [], "artifacts": {},
                 "limits": {"max_steps": 20, "max_review_rounds": 3}, "stop_reason": ""}
        gate = evaluate_gate(state, {"phase": "artifact", "artifacts": ["comparison"]},
                             {"status": "checked", "items": [checked]},
                             {"model_calls": 1, "max_model_calls": 30, "charged_tokens": 100, "token_budget": 10000})
        self.assertEqual(gate["decision"], "repair")
        self.assertIn("比较对象", gate["issues"][0]["reason"])
        review["recommended_action"] = "none"
        review["missing_information"] = []
        self.assertTrue(validate_review(review, item, sources)["passed"])


class ReadingSelectionTests(unittest.TestCase):
    def test_budget_reserves_method_and_ablation_and_preserves_source_slices(self):
        p = paper()
        text = "1 Introduction\n" + "Background introduction. " * 200 + "\n2 Method\n" + (
            "The module maps input features to selected indices.\n" * 160) + "3 Ablation\n" + (
            "The controlled ablation fixes the predictor and changes selection.\n" * 60) + "4 References\n" + "Unrelated citation\n" * 60
        index = build_index(p.as_dict(), {"layout": text})
        selection = select_reading_context(index, {"layout": text}, max_chars=4000, max_tokens=1200)
        self.assertIn("selected indices", selection["text"])
        self.assertIn("controlled ablation", selection["text"])
        self.assertNotIn("Unrelated citation", selection["text"])
        self.assertLessEqual(len(selection["text"]), 4000)
        self.assertLessEqual(selection["estimated_tokens"], 1200)
        for part in selection["text"].split("\n[EXCERPT GAP: omitted text; not read]\n"):
            self.assertTrue(locate_quote(index, {"layout": text}, part))
        excerpt = make_excerpt(text, 4000)
        self.assertIn("controlled ablation", excerpt)
        self.assertIn("selected indices", excerpt)


class ReportBundleTests(unittest.TestCase):
    def config(self, root):
        return fixtures.PipelineTest().config(root)

    def test_main_report_is_readable_and_all_short_references_resolve(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_pipeline(self.config(Path(d)), backend=FakeBackend(paper()), collector=lambda _: [paper()])
            folder = Path(result["report_path"]).parent
            main = Path(result["report_path"]).read_text(encoding="utf-8")
            evidence = Path(result["evidence_path"]).read_text(encoding="utf-8")
            self.assertIn("##### 核心创新", main)
            self.assertIn("**输入**", main)
            self.assertIn("**处理过程**", main)
            self.assertIn("**设计理由（作者陈述）**", main)
            for token in (QUOTE, "ER-", "原文锚点", "## Token 消耗", "input_tokens"):
                self.assertNotIn(token, main)
            self.assertIn(QUOTE, evidence)
            self.assertIn("ER-", evidence)
            self.assertIn("定位：", evidence)
            for anchor in re.findall(r"evidence\.md#(evidence-\d+)", main):
                self.assertIn('id="' + anchor + '"', evidence)
            latest = (Path(d) / "reports/latest.md").read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)]+\.md)(?:#[^)]+)?\)", latest):
                self.assertTrue((folder.parent / target).exists(), target)
            self.assertEqual(json.loads((folder / "paper-explanations.json").read_text(encoding="utf-8"))["schema_version"], 1)

    def test_rejected_explanation_does_not_revoke_valid_evidence_or_leak_into_body(self):
        class RejectExplanation(FakeBackend):
            def run(self, prompt, schema, folder):
                response = super().run(prompt, schema, folder)
                if schema == EXPLANATION_SCHEMA:
                    response["explanations"][0]["novelty"]["text"] = "UNSUPPORTED_EXPLANATION"
                if schema == VERIFICATION_SCHEMA:
                    payload = json.loads(prompt.split("材料：\n", 1)[1])
                    for item, review in zip(payload["items"], response["reviews"]):
                        if "UNSUPPORTED_EXPLANATION" in item["text"]:
                            review["claims"][0].update(verdict="unsupported", citations=[], reason="无原文支持")
                            review["recommended_action"] = "repair"
                return response
        with tempfile.TemporaryDirectory() as d:
            result = run_pipeline(self.config(Path(d)), backend=RejectExplanation(paper()), collector=lambda _: [paper()])
            self.assertEqual(result["status"], "partial")
            self.assertTrue(result["domains"][0]["analyses"][0]["claims"])
            self.assertFalse(result["domains"][0]["paper_explanations"])
            self.assertNotIn("UNSUPPORTED_EXPLANATION", Path(result["report_path"]).read_text(encoding="utf-8"))

    def test_unavailable_reviewer_prevents_downstream_factual_ideas(self):
        class FailedReview(FakeBackend):
            def run(self, prompt, schema, folder):
                if schema == VERIFICATION_SCHEMA:
                    raise RuntimeError("fixture unavailable reviewer")
                return super().run(prompt, schema, folder)
        with tempfile.TemporaryDirectory() as d:
            result = run_pipeline(self.config(Path(d)), backend=FailedReview(paper()), collector=lambda _: [paper()])
            self.assertEqual(result["status"], "partial")
            self.assertFalse(result["domains"][0]["analyses"][0]["claims"])
            self.assertFalse(result["domains"][0]["ideas"])
            self.assertNotIn(QUOTE, Path(result["report_path"]).read_text(encoding="utf-8"))

    def test_supplement_is_bounded_and_uses_frozen_sources_after_cache_change(self):
        for repaired in (True, False):
            with self.subTest(repaired=repaired), tempfile.TemporaryDirectory() as d:
                root, p = Path(d), paper()
                cached = root / "cache/papers/fixture.txt"
                cached.parent.mkdir(parents=True)
                cached.write_text("1 Introduction\n" + QUOTE + "\n2 Methods\n" + "Frozen module detail.\n" * 150 +
                                  "3 Ablation\nControlled evaluation.\n", encoding="utf-8")
                p.full_text_path = str(cached)
                class SupplementBackend(FakeBackend):
                    def run(self, prompt, schema, folder):
                        value = super().run(prompt, schema, folder)
                        if schema == EXPLANATION_SCHEMA:
                            if "initial" in folder.name:
                                cached.write_text("Changed external cache", encoding="utf-8")
                            if "initial" in folder.name or not repaired:
                                value["explanations"][0]["modules"][0]["inputs"] = {"text": "", "evidence_ids": []}
                        return value
                backend = SupplementBackend(p)
                result = run_pipeline(self.config(root), backend=backend, collector=lambda _: [p])
                from scholar_daily.contracts import ANALYSIS_SCHEMA
                self.assertEqual(sum(s == ANALYSIS_SCHEMA for s in backend.calls), 2 if repaired else 3)
                self.assertEqual(sum(s == EXPLANATION_SCHEMA for s in backend.calls), 2 if repaired else 3)
                self.assertIn("Frozen module detail", result["domains"][0]["papers"][0]["full_text"])
                self.assertNotIn("Changed external cache", result["domains"][0]["papers"][0]["full_text"])
                self.assertEqual(result["status"], "complete" if repaired else "partial")

    def test_waiting_report_hides_unreviewed_claims_and_preserves_latest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = self.config(root)
            latest = root / "reports/latest.md"
            latest.parent.mkdir(parents=True)
            latest.write_text("previous final report", encoding="utf-8")
            result = run_pipeline(config, backend=SessionBackend(), collector=lambda _: [paper()])
            Path(result["pending_task"]["response_path"]).write_text(json.dumps({"analyses": [analysis(paper())]}), encoding="utf-8")
            result = run_pipeline({}, session_folder=Path(result["session_path"]).parent)
            self.assertEqual(result["status"], "waiting")
            main = Path(result["report_path"]).read_text(encoding="utf-8")
            self.assertNotIn("测试断言", main)
            self.assertNotIn("ER-", main)
            self.assertEqual(latest.read_text(encoding="utf-8"), "previous final report")

    def test_legacy_analysis_can_render_without_new_artifacts(self):
        p = paper()
        a = validate_analysis(analysis(p), p)
        domain = DomainResult("fixture", [], [p], [p], [p], analyses=[a])
        main = render_report({"project": {"name": "fixture"}}, [domain], "fixture", "legacy", [])
        self.assertIn("核心创新", main)
        self.assertNotIn(QUOTE, main)
        self.assertIn(QUOTE, render_evidence(domain_evidence([domain])))

    def test_every_retained_paper_has_complete_reviewed_explanation(self):
        papers = [Paper(**dict(paper().as_dict(), paper_id="fixture-%d" % i, title="Paper %d" % i)) for i in range(10)]
        by_id = {p.paper_id: p for p in papers}

        class AllPapers(FakeBackend):
            def run(self, prompt, schema, folder):
                if schema == fixtures.ANALYSIS_SCHEMA:
                    self.calls.append(schema)
                    payload = json.loads(prompt.split("论文材料：\n", 1)[1])
                    return {"analyses": [analysis(by_id[p["paper_id"]]) for p in payload]}
                if schema == fixtures.IDEA_SCHEMA:
                    return {"ideas": []}
                return super().run(prompt, schema, folder)

        for limit in (10, 2):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as d:
                config = self.config(Path(d))
                config["sources"]["analysis_limit"] = limit
                result = run_pipeline(config, backend=AllPapers(papers[0]), collector=lambda _: papers)
                self.assertEqual(result["status"], "complete")
                domain = result["domains"][0]
                expected = {p["paper_id"] for p in domain["top10"]}
                self.assertEqual(len(expected), limit)
                self.assertEqual(set(domain["paper_explanations"]), expected)
                self.assertEqual(set(domain["reading_review"]["paper_reviews"]), expected)
                for row in domain["reading_review"]["paper_reviews"].values():
                    self.assertTrue(row["complete"])
                    self.assertTrue(row["rounds"][-1]["final"]["passed"])
                main = Path(result["report_path"]).read_text(encoding="utf-8")
                for label in ("研究问题", "已有方法与不足", "核心创新", "整体流程", "实验支持", "适用条件与局限"):
                    self.assertEqual(main.count("##### " + label), limit)
                self.assertEqual(main.count("###### 自适应选择"), limit)
                checks = json.loads(Path(result["explanations_path"]).read_text(encoding="utf-8"))["content_checks"][domain["name"]]
                self.assertEqual(set(checks), expected)
                self.assertTrue(all(not gaps for gaps in checks.values()))

    def test_semantic_and_final_rejections_trigger_feedback_repairs(self):
        for phase in ("evidence", "final"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as d:
                class RepairBackend(FakeBackend):
                    rejected = False
                    feedback = []

                    def run(self, prompt, schema, folder):
                        value = super().run(prompt, schema, folder)
                        if "上一轮审查" in prompt:
                            self.feedback.append(prompt)
                        if schema == VERIFICATION_SCHEMA:
                            payload = json.loads(prompt.split("材料：\n", 1)[1])
                            if payload["phase"] == phase and not self.rejected:
                                for item, row in zip(payload["items"], value["reviews"]):
                                    if item["section"] == "explanation":
                                        row["claims"][0].update(verdict="unsupported", reason="需要正确的模块比较证据")
                                        row["recommended_action"] = "repair"
                                        self.rejected = True
                                        break
                        return value

                backend = RepairBackend(paper())
                result = run_pipeline(self.config(Path(d)), backend=backend, collector=lambda _: [paper()])
                self.assertEqual(result["status"], "complete")
                self.assertTrue(backend.rejected)
                self.assertTrue(any("需要正确的模块比较证据" in prompt for prompt in backend.feedback))
                rounds = result["domains"][0]["reading_review"]["paper_reviews"][paper().paper_id]["rounds"]
                self.assertEqual(len(rounds), 2)
                self.assertTrue(rounds[-1]["final"]["passed"])

    def test_clarity_feedback_is_repaired_or_remains_partial_after_three_rounds(self):
        reason = "‘这一机制’未说明处理哪个对象，须展开模块操作"
        for persistent in (False, True):
            with self.subTest(persistent=persistent), tempfile.TemporaryDirectory() as d:
                class ClarityBackend(FakeBackend):
                    rejected = False
                    feedback = []

                    def run(self, prompt, schema, folder):
                        value = super().run(prompt, schema, folder)
                        if "上一轮审查" in prompt:
                            self.feedback.append(prompt)
                        if schema == VERIFICATION_SCHEMA:
                            payload = json.loads(prompt.split("材料：\n", 1)[1])
                            if payload["phase"] == "final" and (persistent or not self.rejected):
                                for item, row in zip(payload["items"], value["reviews"]):
                                    if item["section"] == "explanation":
                                        row.update(recommended_action="repair", missing_information=[reason], query="解释模块操作")
                                        self.rejected = True
                                        break
                        return value

                backend = ClarityBackend(paper())
                result = run_pipeline(self.config(Path(d)), backend=backend, collector=lambda _: [paper()])
                self.assertTrue(any(reason in prompt for prompt in backend.feedback))
                domain = result["domains"][0]
                rounds = domain["reading_review"]["paper_reviews"][paper().paper_id]["rounds"]
                self.assertEqual(result["status"], "partial" if persistent else "complete")
                self.assertEqual(len(rounds), 3 if persistent else 2)
                if persistent:
                    self.assertTrue(any(reason in error for error in domain["errors"]))
                    self.assertNotIn(paper().paper_id, domain["paper_explanations"])
                else:
                    self.assertTrue(rounds[-1]["final"]["passed"])

    def test_failed_paper_does_not_discard_other_complete_explanations(self):
        papers = [Paper(**dict(paper().as_dict(), paper_id="fixture-%d" % i)) for i in range(4)]
        by_id = {p.paper_id: p for p in papers}
        class IsolatedReview(FakeBackend):
            def run(self, prompt, schema, folder):
                if schema == fixtures.ANALYSIS_SCHEMA:
                    payload = json.loads(prompt.split("论文材料：\n", 1)[1])
                    return {"analyses": [analysis(by_id[p["paper_id"]]) for p in payload]}
                if schema == VERIFICATION_SCHEMA:
                    payload = json.loads(prompt.split("材料：\n", 1)[1])
                    if any(item["paper_id"] == "fixture-3" for item in payload["items"]):
                        raise RuntimeError("controlled reviewer failure")
                if schema == fixtures.IDEA_SCHEMA:
                    return {"ideas": []}
                return super().run(prompt, schema, folder)
        with tempfile.TemporaryDirectory() as d:
            result = run_pipeline(self.config(Path(d)), backend=IsolatedReview(papers[0]), collector=lambda _: papers)
            self.assertEqual(result["status"], "partial")
            domain = result["domains"][0]
            self.assertEqual(set(domain["paper_explanations"]), {"fixture-0", "fixture-1", "fixture-2"})
            self.assertFalse(next(a for a in domain["analyses"] if a["paper_id"] == "fixture-3")["claims"])
            self.assertTrue(any("fixture-3" in message for message in domain["errors"]))

    def test_disabled_backend_cannot_report_complete_full_reading(self):
        with tempfile.TemporaryDirectory() as d:
            config = self.config(Path(d))
            config["agent"]["backend"] = "none"
            result = run_pipeline(config, collector=lambda _: [paper()])
            self.assertEqual(result["status"], "partial")
            self.assertTrue(any("完整讲解未执行" in message for message in result["domains"][0]["errors"]))

    def test_session_repair_replays_frozen_source_and_keeps_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = paper()
            cached = root / "cache/papers/frozen.txt"
            cached.parent.mkdir(parents=True)
            cached.write_text(QUOTE + "\nAppendix\nFrozen appendix details.\n", encoding="utf-8")
            p.full_text_path = str(cached)
            result = run_pipeline(self.config(root), backend=SessionBackend(), collector=lambda _: [p])
            run_id = result["run_id"]
            Path(result["pending_task"]["response_path"]).write_text(json.dumps({"analyses": [analysis(p)]}), encoding="utf-8")
            result = run_pipeline({}, session_folder=Path(result["session_path"]).parent)
            task = result["pending_task"]
            prompt = Path(task["prompt_path"]).read_text(encoding="utf-8")
            value = fixtures.reading_response(prompt, EXPLANATION_SCHEMA)
            value["explanations"][0]["modules"][0]["inputs"] = {"text": "", "evidence_ids": []}
            Path(task["response_path"]).write_text(json.dumps(value), encoding="utf-8")
            cached.write_text("Changed external text", encoding="utf-8")
            result = run_pipeline({}, session_folder=Path(result["session_path"]).parent)
            for _ in range(20):
                self.assertEqual(result["run_id"], run_id)
                task = result["pending_task"]
                prompt = Path(task["prompt_path"]).read_text(encoding="utf-8")
                schema = json.loads(Path(task["schema_path"]).read_text(encoding="utf-8"))
                value = fixtures.reading_response(prompt, schema)
                if schema == fixtures.ANALYSIS_SCHEMA:
                    self.assertIn("Frozen appendix details", prompt)
                    self.assertNotIn("Changed external text", prompt)
                    value = {"analyses": [analysis(p)]}
                elif schema == fixtures.IDEA_SCHEMA:
                    value = {"ideas": []}
                self.assertIsNotNone(value)
                Path(task["response_path"]).write_text(json.dumps(value), encoding="utf-8")
                result = run_pipeline({}, session_folder=Path(result["session_path"]).parent)
                if result["status"] != "waiting":
                    break
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["run_id"], run_id)


class FullReadingTests(unittest.TestCase):
    def test_reference_filter_preserves_lettered_appendices_and_exact_slices(self):
        p = paper()
        text = ("1 Introduction\n" + QUOTE * 10 + "\n2 Method\nMain method description.\n3 References\n" +
                "[1] Bibliographic entry.\n" * 250 + "\fSupplementary Material\nA Training Details\n" +
                "Appendix training protocol.\nB.1 Ablation Results\nAppendix ablation results.\n")
        excerpt = make_excerpt(text, 4000)
        self.assertNotIn("Bibliographic entry", excerpt)
        self.assertIn("Appendix training protocol", excerpt)
        self.assertIn("Appendix ablation results", excerpt)
        self.assertLessEqual(len(excerpt), 4000)
        index = build_index(p.as_dict(), {"layout": text})
        for part in excerpt.split("\n[EXCERPT GAP: omitted text; not read]\n"):
            self.assertTrue(locate_quote(index, {"layout": text}, part))

    def test_unstructured_excerpt_respects_character_budget(self):
        self.assertLessEqual(len(make_excerpt("0123456789" * 1000, 2000)), 2000)

    def test_figure_selection_reaches_method_and_late_appendix_tables(self):
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "paper.pdf"
            pages = ["Background page" for _ in range(27)]
            pages[4] = "Figure 2: Framework overview."
            pages[6] = "Spatial equation (1)"
            pages[11] = "Table 5: Ablation results."
            pages[26] = "Table 9: Appendix ablation results."
            pdf.with_suffix(".txt").write_text("\f".join(pages), encoding="utf-8")
            self.assertEqual(figure_page_numbers(pdf, 12), [5, 7, 12, 27])
            self.assertIn(27, figure_page_numbers(pdf, 2))


if __name__ == "__main__":
    unittest.main()
