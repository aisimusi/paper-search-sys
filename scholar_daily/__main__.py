from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from .config import load_config, resolve_path
from .pipeline import run_pipeline
from .onboarding import configure, create_config, diagnose
from .schedule import write_schedule
from .storage import Store


def ready(config: dict) -> None:
    for domain in config["research_profile"]["domains"]:
        if "填写" in domain["name"] or any("填写" in q for q in domain["queries"]):
            raise ValueError("请先在配置中填写真实研究方向与英文检索词；试运行可用 demo")


def print_result(result: dict) -> int:
    keys = ["run_id", "status", "mode", "report_path", "json_path", "session_path", "pending_task"]
    print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, indent=2))
    for record in result.get("usage", []):
        print("token 消耗 [%s] %s：调用 %d 次，输入(未缓存) %d，缓存读 %d，输出 %d，推理 %d，合计 %d" % (
            record.get("role"), record.get("model"), int(record.get("calls") or 0),
            int(record.get("input_tokens") or 0), int(record.get("cache_read_tokens") or 0),
            int(record.get("output_tokens") or 0), int(record.get("reasoning_tokens") or 0),
            int(record.get("total_tokens") or 0)))
    return 0 if result["status"] == "complete" else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="个人每日论文调研：默认由当前 Codex 会话完成分析")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="创建可填写的 JSON 配置，保留已有文件")
    init.add_argument("--config", default="config.json")
    serve = sub.add_parser("mcp-serve", help="启动文献 MCP Server")
    serve.add_argument("--config", default="config.json")
    serve.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    for command_name in ("mcp-tools", "mcp-call"):
        command = sub.add_parser(command_name, help="发现或调用已配置的外部 MCP 工具")
        command.add_argument("--config", default="config.json")
        command.add_argument("--server", required=True)
        if command_name == "mcp-tools":
            command.add_argument("--all", action="store_true", help="同时显示未允许的工具，不能据此调用")
        else:
            command.add_argument("--tool", required=True)
            command.add_argument("--arguments", type=Path, required=True, help="JSON 参数文件")
    research = sub.add_parser("research", help="通过 LangGraph 动态规划研究任务")
    research.add_argument("--config", default="config.json")
    research.add_argument("--goal", required=True)
    research.add_argument("--input", type=Path, help="可选的 facts/papers JSON 输入")
    research.add_argument("--backend", choices=["codex-session", "codex", "harness"])
    research.add_argument("--context-mode", choices=["full", "compact"], help="Planner 原始上下文或结构化压缩上下文，用于配对评测")
    research.add_argument("--skill", help="预先载入一个项目技能；其余资源按需加载")
    sub.add_parser("skill-list", help="仅查看可用技能的名称、版本与适用条件")
    for flag in ("max-steps", "max-model-calls", "max-attempts", "token-budget", "max-review-rounds"):
        research.add_argument("--" + flag, type=int)
    audit = sub.add_parser("research-audit", help="导入抽样人工标签并统计与模型判断的分歧")
    audit.add_argument("--task-dir", type=Path, required=True)
    audit.add_argument("--labels", type=Path, required=True)
    for name in ("research-resume", "research-status"):
        command = sub.add_parser(name, help="恢复或查看持久化研究任务")
        command.add_argument("--config", default="config.json")
        command.add_argument("--task-dir", type=Path, required=True)
        if name == "research-resume":
            command.add_argument("--answer", type=Path, help="补充用户事实的 JSON 文件")
    for name, help_text in [("configure", "交互填写配置，保留备份"), ("doctor", "检查配置和本地运行条件"), ("collect", "仅联网检索初筛，不调用模型"),
                            ("run", "运行调研，会话模式生成待办文件"), ("resume", "校验当前会话响应并继续同一次调研"),
                            ("demo", "离线演示；使用明确标记的虚构样本"),
                            ("history", "查看运行记录"), ("schedule", "生成当前平台的定时任务文件")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", default="config.json")
        if name == "resume":
            command.add_argument("--session", type=Path, required=True, help="run 返回的 session.json 路径")
        if name in {"run", "doctor"}:
            command.add_argument("--backend", choices=["codex-session", "codex", "harness", "none"])
        if name == "schedule":
            command.add_argument("--output", default="deploy")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "mcp-serve":
            from .mcp_server import main as serve_mcp
            serve_mcp(["--config", args.config, "--transport", args.transport, "--host", args.host, "--port", str(args.port)])
            return 0
        if args.command in {"mcp-tools", "mcp-call"}:
            from .mcp_client import request_sync, function_catalog
            from .research_calls import read_json
            config = load_config(args.config)
            if args.command == "mcp-tools":
                result = request_sync(config, args.server, "list", allowed_only=not args.all)
                result["function_calling"] = function_catalog(args.server, [row for row in result["tools"] if row["allowed"]])
            else:
                result = request_sync(config, args.server, "call", tool=args.tool, arguments=read_json(args.arguments))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "research-audit":
            from .research_audit import annotate_sample
            print(json.dumps(annotate_sample(args.task_dir, args.labels), ensure_ascii=False, indent=2))
            return 0
        if args.command == "skill-list":
            from .skill_registry import metadata
            print(json.dumps(metadata(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "init":
            path = Path(args.config).expanduser().resolve()
            if not create_config(path):
                raise ValueError("配置已存在，使用 configure 编辑：" + str(path))
            print("配置模板：" + str(path))
            return 0
        if args.command == "configure":
            configure(Path(args.config).expanduser().resolve())
            return 0
        if args.command in {"research", "research-resume", "research-status"}:
            from .orchestrator import run_research, task_status
            from .research_calls import read_json
            if args.command == "research-status":
                state = task_status(args.task_dir)
                print(json.dumps({k: state.get(k) for k in ("task_id", "goal", "status", "plan", "deliverables", "issues", "steps", "budget", "pending", "stop_reason", "verification", "gate", "gate_history", "context_policy", "context_metrics", "human_audit")}, ensure_ascii=False, indent=2))
                return 0
            if args.command == "research-resume":
                result = run_research(folder=args.task_dir, answer=read_json(args.answer) if args.answer else None)
            else:
                config = load_config(args.config)
                if args.backend:
                    config["agent"]["backend"] = args.backend
                limits = {key: getattr(args, key) for key in ("max_steps", "max_model_calls", "max_attempts", "token_budget", "max_review_rounds")
                          if getattr(args, key) is not None}
                inputs = read_json(args.input) if args.input else {}
                if args.context_mode:
                    from .paper_context import checked_policy
                    if not isinstance(inputs, dict):
                        raise ValueError("任务输入必须为 JSON 对象")
                    policy = checked_policy(inputs.get("context_policy"))
                    inputs["context_policy"] = dict(policy, planner_mode=args.context_mode)
                if args.skill:
                    inputs["skill_name"] = args.skill
                result = run_research(config, args.goal, inputs=inputs, limits=limits)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "complete" else 2
        if args.command == "resume":
            if args.session.name != "session.json":
                raise ValueError("--session 应为 run 返回的 session.json 路径")
            return print_result(run_pipeline({}, session_folder=args.session.parent))
        config = load_config(args.config)
        if getattr(args, "backend", None):
            config["agent"]["backend"] = args.backend
        if args.command == "doctor":
            status = diagnose(config)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 1 if status["errors"] else 0
        if args.command == "history":
            db_path = resolve_path(config, "data_dir") / "research.sqlite3"
            if not db_path.exists():
                print("尚无运行记录")
                return 0
            store = Store(db_path)
            try:
                print(json.dumps(store.latest_runs(), ensure_ascii=False, indent=2))
            finally:
                store.close()
            return 0
        if args.command == "schedule":
            paths = write_schedule(config, Path(args.output).resolve())
            print("仅生成文件，尚未启用：\n" + "\n".join(str(p) for p in paths))
            print("启用前须填写真实方向、确认预算并成功运行一次 run。")
            return 0
        demo = args.command == "demo"
        if not demo:
            ready(config)
            if args.command == "run":
                status = diagnose(config)
                if status["errors"]:
                    raise ValueError("；".join(status["errors"]))
        else:
            # Demonstrations never contaminate real histories or the latest real report.
            config = copy.deepcopy(config)
            for key in ["data_dir", "cache_dir", "output_dir"]:
                config["project"][key] = str(Path(config["_root"]) / "demo-output" / key)
        result = run_pipeline(config, collect_only=args.command == "collect" or config["agent"]["backend"] == "none", demo=demo)
        return print_result(result)
    except (ValueError, RuntimeError, OSError, EOFError, KeyboardInterrupt) as exc:
        print("错误：" + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
