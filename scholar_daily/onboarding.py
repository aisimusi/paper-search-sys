"""Local first-run configuration and diagnostics; never calls a model."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .agents import harness_home
from .config import DEFAULT_CONFIG, load_config, resolve_path, validate_config


def create_config(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(DEFAULT_CONFIG, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        return False
    return True


def configure(path: Path, ask=input) -> None:
    create_config(path)
    # Keep user-only fields; do not persist internal resolved paths.
    data = {k: copy.deepcopy(v) for k, v in load_config(path).items() if not k.startswith("_")}
    print("回车保留当前值；英文检索式用分号分隔。codex-session 使用当前 Codex 会话的模型。")

    def prompt(label, current):
        value = ask(f"{label} [{current}]: ").strip()
        return value or current

    domain = data["research_profile"]["domains"][0]
    domain["name"] = prompt("首个研究方向", domain["name"])
    domain["queries"] = [v.strip() for v in prompt("英文检索式", "; ".join(domain["queries"])).split(";") if v.strip()]
    categories = prompt("arXiv 分类（逗号分隔，* 表示不限）", ",".join(domain["arxiv_categories"]) or "*")
    domain["arxiv_categories"] = [] if categories == "*" else [v.strip() for v in categories.split(",") if v.strip()]
    data["agent"]["backend"] = prompt("backend（codex-session/codex/harness/none）", data["agent"]["backend"])
    if data["agent"]["backend"] == "harness":
        harness = data["agent"]["harness"]
        harness["provider"] = prompt("Harness provider", harness["provider"])
        harness["model"] = prompt("该 provider 可用的模型 ID（必填）", harness["model"])
        harness["dsh_home"] = prompt("独立 Harness home（相对项目根）", harness["dsh_home"])
    data["sources"]["analysis_limit"] = int(prompt("每个领域精读上限（1–10）", str(data["sources"]["analysis_limit"])))
    context = data["research_profile"]["lab_context"]
    context["summary"] = prompt("已有工作与当前问题（可留空）", context["summary"])
    validate_config(data)
    backup = path.with_name(path.name + "." + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".bak")
    shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"配置已保存：{path}；原配置备份：{backup}")
    print("可继续编辑 research_profile：多个 domains、lab_context.datasets 和 compute 中的单卡显存/GPU 小时。未知资源保留 null/[]。")


def diagnose(config: dict) -> dict:
    errors, warnings = [], []
    for domain in config["research_profile"]["domains"]:
        if "填写" in domain["name"] or any("填写" in q for q in domain["queries"]):
            errors.append("请填写 research_profile.domains 中的真实方向和英文检索式")
            break
    sdk_version, runtime = None, None
    sdk_error = None
    backend = config["agent"]["backend"]
    if backend == "harness":
        try:
            import deepseek_harness  # noqa: F401
            from deepseek_harness_runtime import bundled_runtime_path
            sdk_version = importlib.metadata.version("deepseek-harness-sdk")
            runtime = str(bundled_runtime_path())
        except (ImportError, OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
            sdk_error = type(exc).__name__
    settings = config["agent"]["harness"]
    home = harness_home(settings, Path(config["_root"]))
    key_present = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
    if backend == "harness":
        if sdk_error:
            errors.append("Harness SDK/runtime 未就绪；请运行 scripts/scholar-daily.ps1 setup（" + sdk_error + "）")
        if not home:
            errors.append("请填写 agent.harness.dsh_home 或设置 DSH_HOME")
        if not (settings.get("model") or "").strip():
            errors.append("请填写 agent.harness.model：该 provider 实际可用的模型 ID")
        if not key_present:
            warnings.append("未检测到 DEEPSEEK_API_KEY；请在本机设置，或确保所选 Harness home/profile 已配置凭据")
    elif backend == "codex" and not shutil.which("codex"):
        errors.append("未找到 Codex CLI")
    mcp_version = None
    mcp_servers = config.get("mcp", {}).get("servers", {})
    if mcp_servers:
        try:
            mcp_version = importlib.metadata.version("mcp")
        except importlib.metadata.PackageNotFoundError:
            errors.append('已配置 MCP server，需在项目 Conda 环境安装 python -m pip install -e ".[mcp]"')
    pdf = shutil.which("pdftotext")
    if config["sources"]["full_text"] and not pdf:
        warnings.append("未找到 pdftotext；全文读取将降级为摘要线索，可用 Conda 安装 poppler")
    return {
        "python": sys.executable, "configuration": config["_config_path"],
        "project_root": config["_root"], "readiness": "needs_configuration" if errors else "local_ready",
        "backend": backend, "harness_sdk_version": sdk_version, "harness_runtime": runtime,
        "model_selection": "current_codex_session" if backend == "codex-session" else "backend_configuration",
        "dsh_home": str(home) if home else None, "harness_model": settings.get("model", ""),
        "mcp_sdk_version": mcp_version, "mcp_servers": list(mcp_servers), "mcp_connection": "not_tested",
        "credential_env_present": key_present, "provider_connection": "not_tested",
        "pdftotext": pdf, "output": str(resolve_path(config, "output_dir")),
        "daily_time": config["schedule"]["local_time"] + " Asia/Shanghai",
        "errors": errors, "warnings": warnings,
    }
