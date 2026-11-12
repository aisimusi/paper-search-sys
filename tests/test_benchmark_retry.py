import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scholar_daily.benchmark import read_json, write_json
from scholar_daily.benchmark_retry import retry


class RetryTests(unittest.TestCase):
    def test_archives_failure_and_retries_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "run-config.json", {"backend": "codex", "model": "test", "timeout_seconds": 30, "batch_size": 5})
            write_json(root / "metrics.json", {"dev": {"pending": 0}})
            path = root / "dev/batches/0000/result.json"
            failure = {"errors": {"q": "Batch has missing, duplicate or unexpected question IDs"}, "predictions": {}}
            write_json(path, failure)
            with patch("scholar_daily.benchmark_retry.run") as run:
                retry(Namespace(output=root))
                self.assertEqual(run.call_count, 1)
                self.assertFalse(path.exists())
                self.assertEqual(read_json(root / "attempts/dev/0000/result.json"), failure)
                write_json(path, failure)
                retry(Namespace(output=root))
                self.assertEqual(run.call_count, 1)

    def test_rejects_pending_first_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "run-config.json", {"backend": "codex"})
            write_json(root / "metrics.json", {"dev": {"pending": 5}})
            with self.assertRaises(ValueError):
                retry(Namespace(output=root))


if __name__ == "__main__":
    unittest.main()
