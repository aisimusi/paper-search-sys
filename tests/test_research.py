import copy
import json
import os
import re
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scholar_daily.agents import CodexBackend, HarnessBackend, parse_response, repair_json, usage_from_events
from scholar_daily.config import DEFAULT_CONFIG, load_config
from scholar_daily.fulltext import make_excerpt
from scholar_daily.contracts import ANALYSIS_SCHEMA, FIGURE_SCHEMA, IDEA_SCHEMA, validate_analysis, validate_idea, validate_shape
from scholar_daily.models import Paper
from scholar_daily.pipeline import run_pipeline
from scholar_daily.report import render_usage
from scholar_daily.paper_explanations import EXPLANATION_SCHEMA
from scholar_daily.schedule import unit_quote, write_systemd_schedule
from scholar_daily.sources import HuggingFaceSource, deduplicate_papers, parse_arxiv_xml
from scholar_daily.storage import Store

QUOTE = "The controlled experiment compares the adaptive selection method with a fixed compute baseline."


def paper():
    now = datetime.now(timezone.utc).isoformat()
    return Paper("2609.00001", "arxiv", "Fictional test paper", ["Fixture"], "Test abstract", now, now,
                 "https://example.invalid/test", "", full_text=QUOTE)


def analysis(p):
    return {"paper_id": p.paper_id, "reading_level": "paper_excerpt", "discovery_summary": "测试摘要概述",
            "claims": [{"kind": kind, "text": "测试断言", "anchor_quote": QUOTE} for kind in ["innovation", "method", "result"]],
            "scores": {"novelty": 3, "method": 3, "experiments": 3, "writing": 3, "relevance": 4},
            "fit_to_lab": "待确认", "resource_notes": "未知", "missing_evidence": ["原始图表未分析"]}


def idea(evidence_id):
    return {"title": "测试假设", "hypothesis": "在固定预算下改善选择", "evidence_ids": [evidence_id],
            "mechanism": "动态分配", "lab_fit": "已有 baseline", "closest_work_and_difference": "待查重",
            "novelty_check_queries": ["adaptive selection"], "minimal_experiment": "固定划分和随机种子，只改选择策略",
            "baseline": "固定策略", "metric": "固定预算准确率", "falsification": "收益低于种子方差则停止",
            "required_data": ["test-data"], "required_vram_gb": 20, "estimated_gpu_hours": 4,
            "estimate_basis": "测试估计", "missing_resources": [], "feasibility": "candidate"}


def explanation(analysis):
    """Fictional explanation for controlled contract/routing tests, not a semantic evaluation."""
    ids = [analysis["claims"][0]["evidence_id"]]
    def cited(text):
        return {"text": "虚构夹具：" + text, "evidence_ids": ids}
    return {"paper_id": analysis["paper_id"], "problem": cited("固定预算下选择输入"),
            "baseline": cited("固定选择策略无法按输入变化调整"), "novelty": cited("相对固定策略，按输入调整选择"),
            "pipeline": cited("输入进入选择模块，选中内容进入预测器得到预测"),
            "modules": [{"name": "自适应选择", "role": cited("本文改造选择模块，保留原预测器"),
                         "inputs": cited("输入特征与计算预算"), "operation": cited("按特征分配固定选择名额"),
                         "outputs": cited("选中输入的索引"), "rationale": cited("随输入调整名额分配以适应内容差异"),
                         "rationale_basis": "author", "validation": cited("固定预测器和预算对比两种选择策略")}],
            "results": cited("在固定预算和受控数据上比较准确率"), "limitations": cited("结论限于受控数据"),
            "missing_information": []}


def reading_response(prompt, schema):
    if schema == EXPLANATION_SCHEMA:
        payload = json.loads(prompt.split("讲解材料：\n", 1)[1])
        return {"explanations": [explanation(a) for a in payload]}
    from scholar_daily.research_verifier import VERIFICATION_SCHEMA
    if schema == VERIFICATION_SCHEMA:
        from test_orchestrator import passing_reviews
        return passing_reviews(prompt)
    return None


class FakeBackend:
    def __init__(self, p):
        self.p = p
        self.calls = []

    def run(self, prompt, schema, folder):
        self.calls.append(schema)
        response = reading_response(prompt, schema)
        if response is not None:
            return response
        if schema is ANALYSIS_SCHEMA:
            return {"analyses": [analysis(self.p)]}
        evidence_id = validate_analysis(analysis(self.p), self.p)["claims"][0]["evidence_id"]
        return {"ideas": [idea(evidence_id)]}


class ContractsTest(unittest.TestCase):
    def test_fabricated_quote_is_dropped_not_fatal(self):
        p = paper()
        data = analysis(p)
        data["claims"][0]["anchor_quote"] = "This sentence does not occur anywhere in the provided paper."
        result = validate_analysis(data, p)
        self.assertEqual(len(result["claims"]), 2)  # method + result 保留
        self.assertEqual(result["discarded_claims"], 1)
        self.assertIsNone(result["scores"]["novelty"])  # innovation 证据被丢弃，分数降级
        self.assertEqual(result["scores"]["method"], 3)
        self.assertIsNone(result["quality_score"])

    def test_overlong_quote_is_dropped_not_fatal(self):
        p = paper()
        p.full_text = QUOTE + " " + ("X" * 2500)
        data = analysis(p)
        data["claims"][0]["anchor_quote"] = "X" * 2500  # 超过 2000 字符上限
        result = validate_analysis(data, p)
        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(result["discarded_claims"], 1)
        self.assertIsNone(result["scores"]["novelty"])

    def test_abstract_cannot_be_promoted(self):
        p = paper()
        p.full_text = ""
        data = analysis(p)
        data["reading_level"] = "abstract_only"
        with self.assertRaisesRegex(ValueError, "摘要"):
            validate_analysis(data, p)

    def test_missing_result_demotes_experiment_score(self):
        p = paper()
        data = analysis(p)
        data["claims"] = [c for c in data["claims"] if c["kind"] != "result"]
        result = validate_analysis(data, p)
        self.assertIsNone(result["scores"]["experiments"])
        self.assertEqual(result["scores"]["novelty"], 3)
        self.assertIsNone(result["quality_score"])

    def test_multi_gpu_memory_is_not_summed(self):
        data = idea("ER-test")
        profile = {"compute": {"gpus": [{"available_vram_gb": 12, "count": 4}], "max_experiment_hours": 8},
                   "lab_context": {"datasets": ["test-data"]}}
        self.assertEqual(validate_idea(data, {"ER-test"}, profile)["feasibility"], "blocked")

    def test_unknown_conditions_require_more_info(self):
        self.assertEqual(validate_idea(idea("ER-test"), {"ER-test"}, {})["feasibility"], "needs_info")

    def test_unknown_evidence_rejected(self):
        with self.assertRaises(ValueError):
            validate_idea(idea("ER-invented"), {"ER-real"}, {})

    def test_nonfinite_json_rejected_and_extra_fields_dropped(self):
        # Non-finite numbers stay invalid. A model-added extra field is noise: drop
        # it instead of rejecting the whole batch, while required fields stay enforced.
        with self.assertRaises(ValueError):
            parse_response('{"ideas":NaN}', IDEA_SCHEMA)
        self.assertEqual(parse_response('{"ideas":[],"unexpected":true}', IDEA_SCHEMA), {"ideas": []})
        with self.assertRaises(ValueError):
            parse_response('{"unexpected":true}', IDEA_SCHEMA)


class ParserRelaxationTest(unittest.TestCase):
    def test_repair_json_completes_truncated_object(self):
        self.assertEqual(repair_json('{"figures": [{"page": 1}'),
                         '{"figures": [{"page": 1}]}')

    def test_repair_json_drops_trailing_commas(self):
        self.assertEqual(repair_json('{"a": [1, 2,]}'), '{"a": [1, 2]}')

    def test_parse_response_repairs_truncated_figure_batch(self):
        truncated = '{"figures": [{"page": 1, "kind": "figure", "caption": "c", "description": "d"}'
        value = parse_response(truncated, FIGURE_SCHEMA)
        self.assertEqual(value["figures"][0]["page"], 1)

    def test_parse_response_drops_unknown_fields(self):
        self.assertEqual(parse_response('{"ideas": [], "unexpected": true}', IDEA_SCHEMA), {"ideas": []})


class UsageReportTest(unittest.TestCase):
    def test_usage_from_events_sums_camel_case_counters(self):
        events = [
            {"type": "assistant/message", "data": {"usage": {"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 30,
                                                             "reasoningTokens": 5, "totalTokens": 150}}},
            {"type": "assistant/message", "data": {"usage": {"inputTokens": 10, "totalTokens": 12}}},
            {"type": "tool/result", "data": {"usage": {"inputTokens": 999}}},
            {"type": "assistant/message", "data": {}},
            "not-an-event",
        ]
        self.assertEqual(usage_from_events(events), {"input_tokens": 110, "output_tokens": 20, "cache_read_tokens": 30,
                                                     "reasoning_tokens": 5, "total_tokens": 162})
        self.assertEqual(usage_from_events(None), {field: 0 for field in usage_from_events([])})

    def test_render_usage_lists_roles_and_totals(self):
        text = "\n".join(render_usage([
            {"role": "text", "model": "deepseek-flash", "calls": 2, "input_tokens": 700, "output_tokens": 80,
             "cache_read_tokens": 100, "reasoning_tokens": 5, "total_tokens": 880},
            {"role": "vision", "model": "Qwen/Qwen3.8-27B", "calls": 1, "input_tokens": 1100, "output_tokens": 120,
             "cache_read_tokens": 400, "reasoning_tokens": 30, "total_tokens": 1620},
        ]))
        self.assertIn("## Token 消耗", text)
        self.assertIn("| text | deepseek-flash | 2 | 700 | 100 | 80 | 5 | 880 |", text)
        self.assertIn("| vision | Qwen/Qwen3.8-27B | 1 | 1100 | 400 | 120 | 30 | 1620 |", text)
        self.assertIn("| **合计** | — | — | 1800 | 500 | 200 | 35 | 2500 |", text)

    def test_report_without_usage_explains_why(self):
        self.assertIn("未记录到模型 token 消耗", "\n".join(render_usage([])))


class VisionUsageTest(unittest.TestCase):
    def test_vision_usage_excludes_cached_prompt_tokens_from_input(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "page-1.png"
            image.write_bytes(b"not-a-real-png")
            body = {"choices": [{"message": {"content": json.dumps({"figures": []})}}],
                    "usage": {"prompt_tokens": 1500, "completion_tokens": 120, "total_tokens": 1620,
                              "prompt_tokens_details": {"cached_tokens": 400},
                              "completion_tokens_details": {"reasoning_tokens": 30}}}
            backend = HarnessBackend({"backend": "harness", "timeout_seconds": 60,
                                      "harness": {"model": "deepseek-flash"},
                                      "vision": {"enabled": True, "model": "Qwen/Qwen3.8-27B",
                                                 "base_url": "https://api.siliconflow.cn/v1"}})
            with patch.dict(os.environ, {"VISION_API_KEY": "test-key"}), patch(
                    "scholar_daily.agents.urllib.request.urlopen",
                    return_value=FakeResponse(json.dumps(body).encode("utf-8"))):
                backend.describe_figures([str(image)], root / "figures")
        self.assertEqual(backend.usage, [{"role": "vision", "model": "Qwen/Qwen3.8-27B", "calls": 1,
                                          "input_tokens": 1100, "output_tokens": 120, "cache_read_tokens": 400,
                                          "reasoning_tokens": 30, "total_tokens": 1620}])

    def test_vision_empty_object_means_no_figures(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "page-1.png"
            image.write_bytes(b"not-a-real-png")
            body = {"choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102}}
            backend = HarnessBackend({"backend": "harness", "timeout_seconds": 60,
                                      "harness": {"model": "deepseek-flash"},
                                      "vision": {"enabled": True, "model": "Qwen/Qwen3.8-27B",
                                                 "base_url": "https://api.siliconflow.cn/v1"}})
            with patch.dict(os.environ, {"VISION_API_KEY": "test-key"}), patch(
                    "scholar_daily.agents.urllib.request.urlopen",
                    return_value=FakeResponse(json.dumps(body).encode("utf-8"))):
                result = backend.describe_figures([str(image)], root / "figures")
        self.assertEqual(result, {"figures": []})


class SourcesTest(unittest.TestCase):
    def test_excerpt_includes_middle_body_and_not_only_bibliography(self):
        text = 'A' * 8000 + 'METHODS' * 800 + 'RESULTS' * 800 + '\nReferences\n' + 'CITATION' * 1200
        excerpt = make_excerpt(text, 8000)
        self.assertIn('METHODS', excerpt)
        self.assertIn('RESULTS', excerpt)
        self.assertNotIn('CITATION', excerpt)
        self.assertIn('EXCERPT GAP', excerpt)
    def test_arxiv_version_dedup(self):
        xml = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.00001v2</id><title>Sample</title><summary>Sample abstract</summary><published>2024-01-01T00:00:00Z</published><updated>2024-02-01T00:00:00Z</updated></entry></feed>'''
        p = parse_arxiv_xml(xml)[0]
        self.assertEqual(p.paper_id, "2401.00001")
        hot = Paper(**asdict(p))
        hot.track, hot.attention, hot.hotness_basis = "hot", 20, "HF upvotes=20"
        merged = deduplicate_papers([p, hot])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].track, "latest+hot")
        self.assertEqual(merged[0].attention, 20)

    def test_hot_source_filters_future_and_outdated(self):
        now = datetime.now(timezone.utc)
        rows = [{"paper": {"id": "2610.0000%d" % i, "publishedAt": (now + timedelta(days=offset)).isoformat(),
                           "title": "Test", "summary": "Test", "upvotes": 5}} for i, offset in enumerate([-2, 2, -100])]
        papers = HuggingFaceSource.parse(rows, 30)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].attention, 5)

    def test_arxiv_error_feed_is_not_a_paper(self):
        xml = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_2401</id><title>Error</title><summary>Bad query</summary></entry></feed>'
        with self.assertRaisesRegex(ValueError, "查询错误"):
            parse_arxiv_xml(xml)


class PipelineTest(unittest.TestCase):
    def config(self, root):
        data = copy.deepcopy(DEFAULT_CONFIG)
        data["research_profile"]["domains"][0].update(name="Fixture", queries=["adaptive selection"])
        data["sources"]["full_text"] = False
        path = root / "config.json"
        path.write_text(json.dumps(data))
        return load_config(path)

    def test_cache_and_idea_history_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            p = paper()
            backend = FakeBackend(p)
            result1 = run_pipeline(config, backend=backend, collector=lambda d: [Paper(**asdict(p))])
            result2 = run_pipeline(config, backend=backend, collector=lambda d: [Paper(**asdict(p))])
            self.assertEqual(result1["status"], "complete")
            self.assertEqual(len(result1["domains"][0]["ideas"]), 1)
            self.assertEqual(len(result2["domains"][0]["ideas"]), 0)
            self.assertTrue(result2["domains"][0]["top10"][0]["previously_analyzed"])
            self.assertEqual(sum(s is ANALYSIS_SCHEMA for s in backend.calls), 1)
            self.assertIn("ER-", Path(result1["evidence_path"]).read_text(encoding="utf-8"))
            self.assertNotIn("ER-", Path(result1["report_path"]).read_text(encoding="utf-8"))

    def test_token_usage_is_written_into_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            backend = FakeBackend(paper())
            backend.usage = [{"role": "text", "model": "deepseek-flash", "calls": 2, "input_tokens": 700,
                              "output_tokens": 80, "cache_read_tokens": 100, "reasoning_tokens": 5,
                              "total_tokens": 880}]
            config["agent"]["backend"] = "harness"
            result = run_pipeline(config, backend=backend, collector=lambda d: [paper()])
            self.assertEqual(result["usage"], backend.usage)
            report = Path(result["run_details_path"]).read_text(encoding="utf-8")
            self.assertIn("## Token 消耗", report)
            self.assertIn("| text | deepseek-flash | 2 | 700 | 100 | 80 | 5 | 880 |", report)
            stored = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(stored["usage"], backend.usage)

    def test_backend_failure_still_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            backend = FakeBackend(paper())
            with patch.object(backend, "run", side_effect=RuntimeError("simulated timeout")):
                result = run_pipeline(config, backend=backend, collector=lambda d: [paper()])
            self.assertEqual(result["status"], "partial")
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertIn("simulated timeout", Path(result["report_path"]).read_text(encoding="utf-8"))

    def test_report_timestamps_share_local_timezone(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            backend = FakeBackend(paper())
            result = run_pipeline(config, backend=backend, collector=lambda d: [paper()])
            data = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertTrue(data["started_at"].endswith("+08:00"))
            self.assertTrue(data["finished_at"].endswith("+08:00"))

    def test_idea_generation_uses_reasoning_model(self):
        from scholar_daily import pipeline as pipeline_mod
        made = {}

        def fake_get_backend(agent, workspace=None, model=None):
            b = FakeBackend(paper())
            b.model = model
            b.usage = []
            b.closed = False

            def close():
                b.closed = True
            b.close = close
            made[model] = b
            return b

        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            config["agent"]["backend"] = "harness"
            config["agent"]["harness"]["model"] = "deepseek-flash"
            config["agent"]["harness"]["reasoning_model"] = "deepseek-v4-pro"
            with patch.object(pipeline_mod, "get_backend", side_effect=fake_get_backend):
                result = run_pipeline(config, collector=lambda d: [paper()])
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(made[None].model)
        self.assertEqual(made["deepseek-v4-pro"].model, "deepseek-v4-pro")
        self.assertIsNot(made[None], made["deepseek-v4-pro"])
        self.assertTrue(made[None].closed)
        self.assertTrue(made["deepseek-v4-pro"].closed)

    def test_analysis_is_per_paper_and_isolates_failures(self):
        papers = []
        for i in range(7):
            p = paper()
            p.paper_id = "2609.000%02d" % i
            papers.append(p)
        by_id = {p.paper_id: p for p in papers}

        class BatchingBackend:
            def __init__(self):
                self.analysis_calls = 0
                self.usage = []
                self.closed = False

            def run(self, prompt, schema, folder):
                response = reading_response(prompt, schema)
                if response is not None:
                    return response
                if schema is ANALYSIS_SCHEMA:
                    self.analysis_calls += 1
                    ids = re.findall(r'"paper_id":\s*"([^"]+)"', prompt)
                    if self.analysis_calls == 2:
                        raise RuntimeError("simulated truncated chunk")
                    return {"analyses": [analysis(by_id[pid]) for pid in ids]}
                evidence_id = validate_analysis(analysis(papers[0]), papers[0])["claims"][0]["evidence_id"]
                return {"ideas": [idea(evidence_id)]}

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            backend = BatchingBackend()
            result = run_pipeline(config, backend=backend, collector=lambda d: papers)
        self.assertEqual(backend.analysis_calls, 7)
        self.assertEqual(len(result["domains"][0]["analyses"]), 6)
        self.assertTrue(any("第 2 批" in error for error in result["domains"][0]["errors"]))
        self.assertTrue(backend.closed)

    def test_concurrent_run_is_rejected(self):
        from scholar_daily._compat import FileLock
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            lock = FileLock(Path(directory) / 'data' / 'daily.lock').acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, '已有每日任务'):
                    run_pipeline(config, collect_only=True, collector=lambda d: [])
            finally:
                lock.release()

    def test_interrupted_run_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            store = Store(Path(directory) / 'data/research.sqlite3')
            abandoned = store.create_run(config['_config_path'])
            store.close()
            run_pipeline(config, collect_only=True, collector=lambda d: [])
            store = Store(Path(directory) / 'data/research.sqlite3')
            try:
                old = next(run for run in store.latest_runs() if run['id'] == abandoned)
                self.assertEqual(old['status'], 'interrupted')
            finally:
                store.close()

    def test_timer_contains_explicit_timezone_and_absolute_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            service, timer = write_systemd_schedule(config, Path(directory) / "deploy")
            self.assertIn("08:00:00 Asia/Shanghai", timer.read_text(encoding="utf-8"))
            self.assertIn("Persistent=true", timer.read_text(encoding="utf-8"))
            # On Windows the absolute config path is backslash-escaped by unit_quote;
            # assert the exact quoted/escaped form so the test is cross-platform.
            self.assertIn(unit_quote(config["_config_path"]), service.read_text(encoding="utf-8"))
            self.assertNotIn('WorkingDirectory="', service.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
