"""Versioned local skills: discover metadata, then load only selected resources."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def skills_root():
    root = Path(__file__).resolve().parents[1] / "skills"
    if not (root / "catalog.json").is_file():
        raise ValueError("找不到技能目录；请使用包含 skills/ 的完整项目或分享包")
    return root


def catalog(root=None):
    root = Path(root).resolve() if root is not None else skills_root()
    data = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    if data.get("api_version") != 1 or not isinstance(data.get("skills"), list):
        raise ValueError("不支持的技能目录契约")
    names = set()
    for item in data["skills"]:
        name = item["name"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or name in names:
            raise ValueError("技能名称非法或重复：" + name)
        if not re.fullmatch(r"\d+\.\d+\.\d+", item["version"]):
            raise ValueError("技能版本必须为 major.minor.patch")
        names.add(name)
    return data["skills"]


def metadata(root=None):
    return [{key: row[key] for key in ("name", "description", "version")} for row in catalog(root)]


def spec(name, root=None):
    if not isinstance(name, str):
        raise ValueError("技能名称必须为字符串")
    return next((row for row in catalog(root) if row["name"] == name), None) or _unknown(name)


def _unknown(name):
    raise ValueError("未知技能：" + name)


def load_file(name, resource, root=None, expected=None):
    root = Path(root).resolve() if root is not None else skills_root()
    row = spec(name, root)
    if expected is not None and row != expected:
        raise ValueError("技能版本或目录指纹已变化，请创建新任务")
    if resource not in row["files"]:
        raise ValueError("技能未声明此资源：" + resource)
    candidate = root / name / resource
    path = candidate.resolve()
    if not path.is_relative_to(root / name) or candidate.is_symlink():
        raise ValueError("技能资源路径超出技能目录")
    payload = path.read_bytes()
    if len(payload) > 65536 or hashlib.sha256(payload).hexdigest() != row["files"][resource]:
        raise ValueError("技能文件与目录指纹不符，请更新版本与 catalog.json")
    return {"skill": name, "version": row["version"], "resource": resource,
            "sha256": row["files"][resource], "text": payload.decode("utf-8")}


def load_skill(name, root=None, expected=None):
    row = spec(name, root)
    body = load_file(name, "SKILL.md", root, expected=expected)
    return dict(body, resources=[file for file in row["files"] if file != "SKILL.md"])


def contract(name, root=None):
    return json.loads(load_file(name, "references/contract.json", root)["text"])
