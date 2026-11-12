"""Update checksums and project discovery entries for the four bundled skills."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

NAMES = ("paper-screening", "evidence-verification", "method-comparison", "experiment-design")
FILES = ("SKILL.md", "scripts/run.py", "references/contract.json", "assets/example.json")


def sync(root, entries=False):
    root = Path(root).resolve()
    items = []
    previous_file = root / "skills/catalog.json"
    previous = {row["name"]: row for row in json.loads(previous_file.read_text(encoding="utf-8"))["skills"]} if previous_file.exists() else {}
    for name in NAMES:
        folder = root / "skills" / name
        text = (folder / "SKILL.md").read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1]
        actual = re.search(r"(?m)^name: (.+)$", frontmatter).group(1)
        description = re.search(r"(?m)^description: (.+)$", frontmatter).group(1)
        version = re.search(r'(?m)^  version: "(\d+\.\d+\.\d+)"$', frontmatter).group(1)
        if actual != name:
            raise ValueError("技能文件名与 frontmatter 不一致")
        contract = json.loads((folder / "references/contract.json").read_text(encoding="utf-8"))
        if contract["skill"] != name or contract["version"] != version:
            raise ValueError("技能契约与 frontmatter 版本不一致：" + name)
        item = {"name": name, "description": description, "version": version,
                "files": {file: hashlib.sha256((folder / file).read_bytes()).hexdigest() for file in FILES}}
        if name in previous and item != previous[name]:
            current_version = tuple(map(int, version.split(".")))
            previous_version = tuple(map(int, previous[name]["version"].split(".")))
            if current_version <= previous_version:
                raise ValueError("技能内容变化时必须提高版本并同步契约：" + name)
        items.append(item)
        if entries:
            target = root / ".agents/skills" / name / "SKILL.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('---\nname: %s\ndescription: %s\nmetadata:\n  version: "%s"\n---\n\n'
                              '# %s\n\n读取 [技能正文](../../../skills/%s/SKILL.md)，按需要读取其中声明的资源。\n' %
                              (name, description, version, name, name), encoding="utf-8")
    target = root / "skills/catalog.json"
    target.write_text(json.dumps({"api_version": 1, "skills": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--entries", action="store_true", help="同时生成项目 .agents/skills 的发现入口")
    args = parser.parse_args()
    print(sync(args.root, entries=args.entries))
