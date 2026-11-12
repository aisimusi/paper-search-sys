"""Build a shareable source skill using an explicit file allowlist."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def package(root: Path, output: Path) -> Path:
    root = root.resolve()
    files = [root / name for name in (
        "SKILL.md", "README.md", "AGENTS.md", "TODO.md", "pyproject.toml", "environment.yml",
        "config.example.json", ".gitignore", ".dsh/skills/scholar-daily/SKILL.md",
        ".agents/skills/scholar-daily/SKILL.md",
        "docs/research-orchestrator.md", "docs/agent-skills.md", "docs/context-engineering.md", "docs/mcp.md", "skills/catalog.json",
    )]
    for folder, pattern in (("scholar_daily", "*.py"), ("scripts", "*.py"), ("scripts", "*.ps1"), ("tests", "*.py")):
        files.extend(sorted((root / folder).glob(pattern)))
    for name in ("paper-screening", "evidence-verification", "method-comparison", "experiment-design"):
        files.append(root / ".agents/skills" / name / "SKILL.md")
        files.extend(root / "skills" / name / resource for resource in
                     ("SKILL.md", "scripts/run.py", "references/contract.json", "assets/example.json"))
    for file in files:
        if not file.is_file() or file.is_symlink():
            raise ValueError("Missing or linked release file: " + str(file))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file, "scholar-daily/" + file.relative_to(root).as_posix())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/scholar-daily-windows.zip"))
    args = parser.parse_args()
    print(package(Path(__file__).resolve().parent.parent, args.output.resolve()))
