import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from scholar_daily.__main__ import main
from scholar_daily.agents import SessionBackend, TaskPending
from scholar_daily.config import DEFAULT_CONFIG, load_config
from scholar_daily.contracts import IDEA_SCHEMA
from scholar_daily.onboarding import diagnose
from scholar_daily.models import Paper
from scholar_daily.pipeline import run_pipeline
from scholar_daily.schedule import write_schedule
from scholar_daily.storage import Store
from test_research import analysis, idea, paper, reading_response


class SessionTest(unittest.TestCase):
    def config(self, root, domains=1):
        config = copy.deepcopy(DEFAULT_CONFIG)
        domain = config["research_profile"]["domains"][0]
        domain.update(name="Fixture 0", queries=["adaptive selection"])
        config["research_profile"]["domains"] = [dict(domain, name="Fixture %d" % i) for i in range(domains)]
        config["sources"]["full_text"] = False
        # Existing independent-backend settings must not switch the session's idea model.
        config["agent"]["harness"].update(model="fixture-flash", reasoning_model="fixture-pro")
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return load_config(path)

    def respond(self, result, value):
        Path(result["pending_task"]["response_path"]).write_text(json.dumps(value), encoding="utf-8")

    def resume(self, result):
        return run_pipeline({}, session_folder=Path(result["session_path"]).parent)

    def complete_reading(self, result):
        while result["status"] == "waiting":
            task = result["pending_task"]
            prompt = Path(task["prompt_path"]).read_text(encoding="utf-8")
            schema = json.loads(Path(task["schema_path"]).read_text(encoding="utf-8"))
            response = reading_response(prompt, schema)
            if response is None:
                break
            self.respond(result, response)
            result = self.resume(result)
        return result

    def test_batched_multi_domain_lifecycle_preserves_run_and_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root, domains=2)
            fixtures = []
            for i in range(4):
                p = paper()
                p.paper_id = "fixture-%d" % i
                fixtures.append(p)
            by_id = {p.paper_id: p for p in fixtures}
            collector = Mock(side_effect=lambda domain: [Paper(**asdict(p)) for p in fixtures])
            with patch("scholar_daily.agents.subprocess.Popen") as process, patch(
                    "scholar_daily.agents.HarnessBackend") as harness:
                result = run_pipeline(config, collector=collector)
                original_run = result["run_id"]
                session_path = Path(result["session_path"])
                self.assertFalse((root / "reports/latest.md").exists())
                # A later config edit cannot change this run's research profile.
                (root / "config.json").write_text("{}", encoding="utf-8")
                tasks = []
                while result["status"] == "waiting":
                    prompt = Path(result["pending_task"]["prompt_path"]).read_text(encoding="utf-8")
                    schema = json.loads(Path(result["pending_task"]["schema_path"]).read_text(encoding="utf-8"))
                    response = reading_response(prompt, schema)
                    if response is not None:
                        self.respond(result, response)
                        tasks.append("explanation" if "explanations" in response else "verification")
                    elif "论文材料：\n" in prompt:
                        payload = json.loads(prompt.split("论文材料：\n", 1)[1])
                        self.respond(result, {"analyses": [analysis(by_id[p["paper_id"]]) for p in payload]})
                        tasks.append("analysis")
                    else:
                        evidence = json.loads(prompt.split("证据：\n", 1)[1].split("\n历史 idea：", 1)[0])
                        value = idea(evidence[0]["claims"][0]["evidence_id"])
                        value["title"] += " %d" % len(tasks)
                        self.respond(result, {"ideas": [value]})
                        tasks.append("ideas")
                    result = run_pipeline({}, session_folder=session_path.parent, collector=collector)
                    self.assertEqual(result["run_id"], original_run)
                    self.assertLessEqual(len(tasks), 80)
                process.assert_not_called()
                harness.assert_not_called()
            self.assertEqual(tasks, (["analysis"] * 4 +
                             ["explanation", "verification", "verification"] * 4 + ["ideas"]) * 2)
            self.assertEqual(collector.call_count, 2)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["domains"][0]["analyses"]), 4)
            self.assertEqual(len(result["domains"][1]["ideas"]), 1)
            latest = (root / "reports/latest.md").read_text(encoding="utf-8")
            folder_name = Path(result["report_path"]).parent.name
            self.assertEqual(latest.replace(folder_name + "/evidence.md", "evidence.md").replace(
                folder_name + "/run-details.md", "run-details.md"), Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(self.resume(result)["status"], "complete")
            store = Store(root / "data/research.sqlite3")
            try:
                self.assertEqual(len(store.latest_runs()), 1)
                self.assertEqual(store.latest_runs()[0]["status"], "complete")
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()[0], 0)
            finally:
                store.close()
            # A new run must ask the current session to analyze again.
            fresh = run_pipeline(config, collector=lambda d: [paper()])
            self.assertEqual(fresh["status"], "waiting")
            self.assertNotEqual(fresh["run_id"], original_run)

    def test_invalid_json_missing_papers_and_unknown_evidence_can_be_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            p = paper()
            result = run_pipeline(config, collector=lambda d: [p])
            response = Path(result["pending_task"]["response_path"])
            response.write_text('{"analyses":', encoding="utf-8")
            result = self.resume(result)
            self.assertEqual(result["status"], "waiting")
            self.assertIn("响应校验失败", result["pending_task"]["validation_error"])
            self.assertEqual(response.read_text(encoding="utf-8"), '{"analyses":')
            self.respond(result, {"analyses": []})
            result = self.resume(result)
            self.assertIn("逐篇覆盖", result["pending_task"]["validation_error"])
            invalid = analysis(p)
            invalid["reading_level"] = "abstract_only"
            self.respond(result, {"analyses": [invalid]})
            result = self.resume(result)
            self.assertIn("阅读范围", result["pending_task"]["validation_error"])
            self.respond(result, {"analyses": [analysis(p)]})
            result = self.resume(result)
            result = self.complete_reading(result)
            self.respond(result, {"ideas": [idea("ER-invented")]})
            result = self.resume(result)
            self.assertIn("已校验", result["pending_task"]["validation_error"])
            self.respond(result, {"ideas": []})
            result = self.resume(result)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["domains"][0]["ideas"])

    def test_fabricated_quotes_never_enter_idea_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            p = paper()
            result = run_pipeline(config, collector=lambda d: [p])
            value = analysis(p)
            value["claims"][0]["anchor_quote"] = "This fabricated sentence never occurs in the provided paper."
            self.respond(result, {"analyses": [value]})
            result = self.resume(result)
            prompt = Path(result["pending_task"]["prompt_path"]).read_text(encoding="utf-8")
            self.assertNotIn("fabricated sentence", prompt)
            self.assertEqual(result["domains"][0]["analyses"][0]["discarded_claims"], 1)

    def test_failed_fulltext_is_not_fetched_again_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            config["sources"]["full_text"] = True
            p = paper()
            p.full_text = ""
            with patch("scholar_daily.pipeline.fetch_fulltext", side_effect=RuntimeError("offline fixture")) as fetch:
                result = run_pipeline(config, collector=lambda d: [p])
                result = self.resume(result)
                value = analysis(p)
                value.update(reading_level="abstract_only", claims=[])
                value["scores"] = {key: None for key in value["scores"]}
                self.respond(result, {"analyses": [value]})
                result = self.resume(result)
                fetch.assert_called_once()
            self.assertEqual(result["status"], "partial")
            self.assertFalse(result["domains"][0]["ideas"])

    def test_vision_task_uses_local_pages_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            config["agent"]["vision"] = {"enabled": True, "max_pages": 1}
            p = paper()
            p.pdf_path = str(Path(directory) / "fixture.pdf")
            page = str(Path(directory) / "page-1.png")
            with patch("scholar_daily.pipeline.render_pages", return_value=[page]) as render:
                result = run_pipeline(config, collector=lambda d: [p])
                prompt = Path(result["pending_task"]["prompt_path"]).read_text(encoding="utf-8")
                self.assertIn(page.replace("\\", "\\\\"), prompt)
                self.respond(result, {"figures": [{"page": 1, "kind": "figure", "caption": "Fixture", "description": "Visible fixture"}]})
                result = self.resume(result)
                render.assert_called_once()
            prompt = Path(result["pending_task"]["prompt_path"]).read_text(encoding="utf-8")
            self.assertIn("Visible fixture", prompt)
            self.assertIn("论文材料", prompt)

    def test_changed_task_input_invalidates_existing_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            backend = SessionBackend()
            with self.assertRaises(TaskPending):
                backend.run("old", IDEA_SCHEMA, folder)
            (folder / "response.json").write_text('{"ideas":[]}', encoding="utf-8")
            with self.assertRaises(TaskPending) as pending:
                backend.run("new", IDEA_SCHEMA, folder)
            self.assertIn("输入已变化", pending.exception.task["validation_error"])
            self.assertEqual((folder / "prompt.txt").read_text(encoding="utf-8"), "new")
            self.assertTrue((folder / "response.previous.json").exists())
            self.assertFalse((folder / "response.json").exists())

    def test_cli_doctor_and_resume_need_no_independent_model_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            with patch("scholar_daily.onboarding.shutil.which", return_value=None):
                status = diagnose(config)
            self.assertFalse(status["errors"])
            self.assertEqual(status["model_selection"], "current_codex_session")
            result = run_pipeline(config, collector=lambda d: [paper()])
            (Path(directory) / "config.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["resume", "--session", result["session_path"]]), 2)
            self.assertEqual(json.loads(output.getvalue())["run_id"], result["run_id"])
            self.assertIn("Codex", Path(result["report_path"]).read_text(encoding="utf-8"))

    def test_session_cannot_be_scheduled_as_independent_process(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            with self.assertRaisesRegex(ValueError, "活跃"):
                write_schedule(config, Path(directory) / "deploy")
            self.assertFalse((Path(directory) / "deploy").exists())

    def test_resume_keeps_initial_candidate_order(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            result = run_pipeline(config, collector=lambda d: [paper()])
            with patch("scholar_daily.pipeline.rank_papers", side_effect=AssertionError("must use snapshot")):
                again = self.resume(result)
            self.assertEqual(again["pending_task"], result["pending_task"])

    def test_malformed_checkpoint_returns_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text('{"version":1}', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(main(["resume", "--session", str(path)]), 1)
            self.assertIn("检查点", output.getvalue())


if __name__ == "__main__":
    unittest.main()
