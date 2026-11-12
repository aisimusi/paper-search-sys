"""Retry invalid QASPER Codex response batches once, retaining first-pass artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .benchmark import read_json, run, write_json


def retry(args):
    root = args.output.resolve()
    config = read_json(root / "run-config.json")
    if config["backend"] != "codex":
        raise ValueError("This retry command currently supports Codex snapshots only")
    first_pass = read_json(root / "metrics.json")
    if first_pass["dev"]["pending"]:
        raise ValueError("Finish the first pass before retrying invalid batches")
    allowed = {"Batch has missing, duplicate or unexpected question IDs",
               "Unknown or duplicate evidence ID", "Answer/evidence inconsistency",
               "Wrong question ID or empty answer"}
    candidates = []
    for path in sorted((root / "dev/batches").glob("*/result.json")):
        result = read_json(path)
        archive = root / "attempts/dev" / path.parent.name
        if result["errors"] and set(result["errors"].values()) <= allowed and not archive.exists():
            if result["predictions"]:
                # Do not replace already valid answers by resampling a mixed batch.
                continue
            candidates.append((path, archive))
    if not candidates:
        print("No eligible invalid batches to retry")
        return
    if not (root / "first-pass-metrics.json").exists():
        write_json(root / "first-pass-metrics.json", first_pass)
    for path, archive in candidates:
        if root not in path.resolve().parents or root not in archive.resolve().parents:
            raise ValueError("Retry paths must stay inside the run directory")
        shutil.copytree(path.parent, archive)
        path.unlink()
    write_json(root / "retry-policy.json", {"max_retries": 1, "eligibility": sorted(allowed),
                                           "batches": [p.parent.name for p, _ in candidates],
                                           "selection": "all eligible fully invalid batches; no answer-quality selection"})
    run(argparse.Namespace(output=root, backend="codex", model=config["model"],
                           timeout=config["timeout_seconds"], batch_size=config["batch_size"],
                           config=Path("config.json"), splits=["dev"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    retry(parser.parse_args())


if __name__ == "__main__":
    main()
