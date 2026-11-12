from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "root": ".",
        "name": "my-research",
        "timezone": "Asia/Shanghai",
        "data_dir": "data",
        "output_dir": "reports",
        "cache_dir": "cache",
    },
    "mcp": {"servers": {}, "zotero_server": "zotero"},
    "schedule": {"local_time": "08:00", "cron_timezone": "Asia/Shanghai"},
    "research_profile": {
        "domains": [
            {
                "name": "填写研究方向",
                "queries": ["填写英文关键词组合"],
                "arxiv_categories": ["cs.AI"],
                "recent_days": 7,
                "hot_days": 90,
            }
        ],
        "lab_context": {
            "summary": "",
            "methods": [],
            "datasets": [],
            "papers": [],
            "code_repositories": [],
            "current_questions": [],
        },
        "compute": {
            "gpus": [],
            "cpu_cores": None,
            "ram_gb": None,
            "storage_gb": None,
            "max_experiment_hours": None,
        },
        "constraints": [],
    },
    "sources": {
        "arxiv": True,
        "huggingface": True,
        "full_text": True,
        "max_results_per_query": 40,
        "analysis_limit": 10,
        "max_text_chars": 80000,
    },
    "agent": {
        "backend": "codex-session",
        "model": None,
        "timeout_seconds": 900,
        "harness": {
            "dsh_home": ".harness-home",
            "provider": "deepseek-official",
            "model": "",
            "reasoning_model": None,
            "profile": "",
            "reasoning_effort": None,
            "max_tokens": None,
            "initialize_timeout_seconds": None,
        },
        "vision": {
            "enabled": False,
            "provider": "deepseek-official",
            "model": "",
            "base_url": "",
            "max_pages": 12,
        },
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("配置文件必须是 JSON 对象")
    config = deep_merge(DEFAULT_CONFIG, loaded)
    config["_config_path"] = str(config_path)
    root = config["project"].get("root", ".")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("project.root 必须是非空路径；当前配置目录可填 .")
    root_path = Path(root).expanduser()
    config["_root"] = str((config_path.parent / root_path).resolve())
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    domains = config.get("research_profile", {}).get("domains", [])
    if not isinstance(domains, list) or not domains:
        raise ValueError("research_profile.domains 至少需要一个研究方向")
    for domain in domains:
        if not isinstance(domain, dict):
            raise ValueError("domain 必须是 JSON 对象")
        if not domain.get("name"):
            raise ValueError("每个 domain 必须有 name")
        queries = domain.get("queries", [])
        if not isinstance(queries, list) or not queries or not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("每个 domain 至少需要一个非空 query")
        for key in ("recent_days", "hot_days"):
            value = domain.get(key)
            if not isinstance(value, int) or value < 1:
                raise ValueError("%s 必须是正整数" % key)
    sources = config.get("sources", {})
    if not sources.get("arxiv") and not sources.get("huggingface"):
        raise ValueError("sources.arxiv 或 sources.huggingface 至少启用一个")
    for key, lower, upper in [("max_results_per_query", 1, 100), ("analysis_limit", 1, 10), ("max_text_chars", 2000, 80000)]:
        value = sources[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("sources.%s 必须在 %s 到 %s 之间" % (key, lower, upper))
    if type(config["agent"]["timeout_seconds"]) is not int or config["agent"]["timeout_seconds"] < 1:
        raise ValueError("agent.timeout_seconds 必须是正整数")
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", config["schedule"]["local_time"]):
        raise ValueError("schedule.local_time 应为 HH:MM")
    if config["project"]["timezone"] != "Asia/Shanghai":
        raise ValueError("第一版时间区域支持 Asia/Shanghai")
    compute = config["research_profile"]["compute"]
    if not isinstance(compute.get("gpus"), list):
        raise ValueError("compute.gpus 必须为数组")
    for gpu in compute["gpus"]:
        if not isinstance(gpu, dict):
            raise ValueError("每张 GPU 配置必须为对象")
        vram = gpu.get("available_vram_gb")
        if vram is not None and (type(vram) not in (int, float) or vram <= 0):
            raise ValueError("available_vram_gb 必须为正数或 null")
    hours = compute.get("max_experiment_hours")
    if hours is not None and (type(hours) not in (int, float) or hours <= 0):
        raise ValueError("max_experiment_hours 必须为正数或 null")
    backend = config.get("agent", {}).get("backend", "codex-session")
    if backend not in {"codex-session", "codex", "harness", "none"}:
        raise ValueError("agent.backend 必须是 codex-session、codex、harness 或 none")
    harness = config.get("agent", {}).get("harness", {})
    if not isinstance(harness, dict):
        raise ValueError("agent.harness 必须是 JSON 对象")
    for key in ("dsh_home", "provider", "model", "reasoning_model", "profile", "reasoning_effort"):
        value = harness.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError("agent.harness.%s 必须是字符串或 null" % key)
    for key in ("max_tokens", "initialize_timeout_seconds"):
        value = harness.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("agent.harness.%s 必须是正整数或 null" % key)
    vision = config.get("agent", {}).get("vision", {})
    if not isinstance(vision, dict):
        raise ValueError("agent.vision 必须是 JSON 对象")
    if vision.get("enabled"):
        for key in (() if backend == "codex-session" else ("provider", "model")):
            value = vision.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("agent.vision.%s 必须是该 provider 实际支持的非空字符串" % key)
        base_url = vision.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("agent.vision.base_url 必须是字符串或 null")
        max_pages = vision.get("max_pages")
        if type(max_pages) is not int or not 1 <= max_pages <= 50:
            raise ValueError("agent.vision.max_pages 必须在 1 到 50 之间")

    mcp = config.get("mcp", {})
    if not isinstance(mcp, dict) or not isinstance(mcp.get("servers", {}), dict):
        raise ValueError("mcp.servers 必须为 JSON 对象")
    from .mcp_client import checked_server
    for name, profile in mcp.get("servers", {}).items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", name):
            raise ValueError("MCP server 名称只允许 1～50 位字母数字、下划线或连字符")
        checked_server(profile, config.get("_root"))
    if mcp.get("zotero_storage_root") is not None and not isinstance(mcp["zotero_storage_root"], str):
        raise ValueError("mcp.zotero_storage_root 必须为本地 storage 路径字符串")
    if not isinstance(mcp.get("zotero_server", "zotero"), str):
        raise ValueError("mcp.zotero_server 必须为 server 名称")


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    value = config["project"][key]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_root"]) / path
    return path.resolve()


def domain_queries(domain: Dict[str, Any]) -> List[str]:
    return [str(query).strip() for query in domain.get("queries", []) if str(query).strip()]
