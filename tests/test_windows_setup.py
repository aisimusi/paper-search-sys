import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

from scholar_daily.__main__ import main
from scholar_daily.agents import HarnessBackend, harness_home
from scholar_daily.config import DEFAULT_CONFIG, load_config, resolve_path
from scholar_daily.contracts import IDEA_SCHEMA
from scholar_daily.onboarding import configure, create_config, diagnose
from scholar_daily.schedule import powershell_quote, write_windows_schedule
from scripts.package_skill import package


class SetupTest(unittest.TestCase):
    def test_defaults_and_config_survive_relocation_and_bom(self):
        with tempfile.TemporaryDirectory(prefix="scholar 空格 ") as folder:
            root = Path(folder)
            path = root / "settings" / "config.json"
            create_config(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["project"]["root"] = ".."
            path.write_text(json.dumps(data), encoding="utf-8-sig")
            loaded = load_config(path)
            self.assertEqual(loaded["_root"], str(root.resolve()))
            self.assertEqual(loaded["agent"]["backend"], "codex-session")
            self.assertEqual(resolve_path(loaded, "data_dir"), root / "data")
            self.assertEqual(harness_home(loaded["agent"]["harness"], root), root / ".harness-home")
            with patch.dict(os.environ, {"DSH_HOME": "relative-home"}):
                self.assertEqual(harness_home({"dsh_home": ""}, root), root / "relative-home")
            self.assertFalse(create_config(path))
            self.assertEqual(path.read_bytes()[:3], b"\xef\xbb\xbf")

    def test_wizard_preserves_other_domains_and_resources_and_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            data = copy.deepcopy(DEFAULT_CONFIG)
            data["research_profile"]["domains"].append(copy.deepcopy(data["research_profile"]["domains"][0]))
            data["research_profile"]["compute"]["gpus"] = [{"model": "fixture", "available_vram_gb": 12}]
            path.write_text(json.dumps(data), encoding="utf-8")
            before = path.read_bytes()
            answers = iter(["Test domain", "all:test AND all:method; baseline", "*", "harness", "", "fixture-model", "", "3", ""])
            with contextlib.redirect_stdout(io.StringIO()):
                configure(path, ask=lambda _: next(answers))
            actual = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(actual["research_profile"]["domains"][1], data["research_profile"]["domains"][1])
            self.assertEqual(actual["research_profile"]["compute"], data["research_profile"]["compute"])
            self.assertEqual(actual["research_profile"]["domains"][0]["queries"], ["all:test AND all:method", "baseline"])
            self.assertEqual(actual["agent"]["harness"]["model"], "fixture-model")
            self.assertEqual(list(Path(folder).glob("*.bak"))[0].read_bytes(), before)
            self.assertNotIn("_root", actual)

    def test_missing_config_stops_run_before_network_and_doctor_reports_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            create_config(path)
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()), patch("scholar_daily.__main__.run_pipeline") as run:
                self.assertEqual(main(["doctor", "--config", str(path)]), 1)
                status = json.loads(output.getvalue())
                self.assertEqual(status["provider_connection"], "not_tested")
                self.assertEqual(status["model_selection"], "current_codex_session")
                self.assertIn("真实方向", " ".join(status["errors"]))
                self.assertEqual(main(["run", "--config", str(path)]), 1)
                run.assert_not_called()

    def test_doctor_never_prints_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            create_config(path)
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-value"}):
                report = diagnose(load_config(path))
            self.assertTrue(report["credential_env_present"])
            self.assertNotIn("private-test-value", json.dumps(report))


class HarnessTest(unittest.TestCase):
    def backend(self, root):
        settings = copy.deepcopy(DEFAULT_CONFIG["agent"])
        settings["harness"]["model"] = "fixture-model"
        return HarnessBackend(settings, root)

    def test_reuses_runtime_and_independent_sessions(self):
        client = Mock()
        client.run.return_value = types.SimpleNamespace(finish_reason="completed", final_response='{"ideas":[]}')
        constructor = Mock(return_value=client)
        module = types.SimpleNamespace(DeepSeekHarness=constructor)
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {"deepseek_harness": module}):
            root = Path(folder)
            with self.backend(root) as backend:
                for i in range(2):
                    self.assertEqual(backend.run("prompt", IDEA_SCHEMA, root / str(i)), {"ideas": []})
            constructor.assert_called_once()
            self.assertEqual(constructor.call_args.kwargs["dsh_home"], str(root / ".harness-home"))
            self.assertEqual(constructor.call_args.kwargs["cwd"], str(root))
            self.assertEqual(constructor.call_args.kwargs["request_timeout_seconds"], 900)
            for call in client.run.call_args_list:
                self.assertNotIn("session_id", call.kwargs)
            client.close.assert_called_once()

    def test_unfinished_run_retains_raw_response(self):
        for reason in ["max-tokens", "error", None]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                backend = self.backend(root)
                backend._harness = Mock()
                backend._harness.run.return_value = types.SimpleNamespace(finish_reason=reason, final_response='{"ideas":[]}')
                with self.assertRaisesRegex(RuntimeError, "finish_reason"):
                    backend.run("prompt", IDEA_SCHEMA, root)
                self.assertEqual((root / "response.json").read_text(encoding="utf-8"), '{"ideas":[]}')
                backend.close()


class WindowsScheduleTest(unittest.TestCase):
    def test_xml_and_runner_preserve_paths_timezone_and_exit_code(self):
        with tempfile.TemporaryDirectory(prefix="scholar 空格 '") as folder:
            root = Path(folder)
            path = root / "config.json"
            create_config(path)
            config = load_config(path)
            runner, task = write_windows_schedule(config, root / "deploy")
            ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
            xml = ET.parse(task)
            self.assertTrue(xml.findtext("t:Triggers/t:CalendarTrigger/t:StartBoundary", namespaces=ns).endswith("08:00:00+08:00"))
            self.assertEqual(xml.findtext("t:Settings/t:MultipleInstancesPolicy", namespaces=ns), "IgnoreNew")
            self.assertEqual(xml.findtext("t:Principals/t:Principal/t:LogonType", namespaces=ns), "InteractiveToken")
            self.assertEqual(xml.findtext("t:Actions/t:Exec/t:WorkingDirectory", namespaces=ns), str(root))
            script = runner.read_text(encoding="utf-8-sig")
            self.assertIn(powershell_quote(str(path)), script)
            self.assertIn("Library", script)
            self.assertIn("exit $LASTEXITCODE", script)
            self.assertNotIn("Register-ScheduledTask", script)


class ShareTest(unittest.TestCase):
    def test_share_archive_contains_runnable_project_without_local_data(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as folder:
            target = package(root, Path(folder) / "skill.zip")
            with ZipFile(target) as archive:
                names = archive.namelist()
                self.assertIn("scholar-daily/SKILL.md", names)
                self.assertIn("scholar-daily/scripts/scholar-daily.ps1", names)
                self.assertIn("scholar-daily/scholar_daily/agents.py", names)
                self.assertIn("scholar-daily/.dsh/skills/scholar-daily/SKILL.md", names)
                self.assertIn("scholar-daily/.agents/skills/scholar-daily/SKILL.md", names)
                self.assertIn("scholar-daily/skills/catalog.json", names)
                for skill in ("paper-screening", "evidence-verification", "method-comparison", "experiment-design"):
                    self.assertIn("scholar-daily/skills/%s/scripts/run.py" % skill, names)
                    self.assertIn("scholar-daily/skills/%s/references/contract.json" % skill, names)
                    self.assertIn("scholar-daily/.agents/skills/%s/SKILL.md" % skill, names)
                forbidden = {".conda", ".harness-home", "data", "reports", "cache", "config.json", "config.smoke.json", ".env"}
                self.assertFalse(any(forbidden.intersection(Path(name).parts) for name in names))
                self.assertIsNone(archive.testzip())

    def test_relocated_share_runs_skill_without_absolute_source_paths(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as folder:
            target = package(root, Path(folder) / "skill.zip")
            with ZipFile(target) as archive:
                archive.extractall(Path(folder) / "relocated")
            project = Path(folder) / "relocated/scholar-daily"
            command = [sys.executable, "-X", "utf8", "skills/paper-screening/scripts/run.py", "prepare",
                       "--input", "skills/paper-screening/assets/example.json"]
            result = subprocess.run(command, cwd=project, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["skill"], "paper-screening")


if __name__ == "__main__":
    unittest.main()
