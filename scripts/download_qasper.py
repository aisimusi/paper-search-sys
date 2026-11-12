"""Download the official QASPER v0.3 archives without third-party dependencies."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ARCHIVES = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for url in ARCHIVES:
        archive = root / url.rsplit("/", 1)[1]
        if not archive.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "scholar-daily-qasper/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz"):
                pass
            temporary = archive.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(archive)
        records.append({"url": url, "file": archive.name,
                        "bytes": archive.stat().st_size,
                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile():
                    continue
                target = (root / member.name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Unsafe archive member: " + member.name)
                content = bundle.extractfile(member).read()
                if target.exists() and target.read_bytes() != content:
                    raise ValueError("Refusing to overwrite a modified file: " + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        print(json.dumps(records[-1]), flush=True)
    extracted = [{"file": str(p.relative_to(root)), "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(root.rglob("*")) if p.is_file() and p.suffix != ".tgz"
                 and p.name != "download-manifest.json"]
    manifest = {"dataset": "QASPER", "version": "0.3", "license": "CC BY 4.0",
                "source": "https://huggingface.co/datasets/allenai/qasper/blob/main/qasper.py",
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "archives": records, "files": extracted}
    (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"destination": str(root), "files": extracted}), flush=True)


if __name__ == "__main__":
    main()
