# AGENTS.md

个人每日文献调研工具 `scholar-daily`。核心是「文献收集 → 证据门槛分析 → 候选 idea → 报告」流水线，运行时可选择不同 agent backend；其中 DeepSeek Harness 通过官方 Python SDK 适配。本文件是给在此仓库工作的编码 agent（Codex / Claude / Harness 等）的项目约定。

## 命令速查

```sh
python -m scholar_daily doctor  --config config.json   # 体检：配置 + 解释器 + SDK + pdftotext
python -m scholar_daily configure --config config.json # 配置引导，保留备份
python -m scholar_daily collect --config config.json   # 只检索+初筛，不调用模型
python -m scholar_daily run     --config config.json   # 完整调研（分析 + idea + 报告）
python -m scholar_daily demo    --config config.json   # 离线演示（虚构样本）
python -m scholar_daily history --config config.json   # 历史运行记录
python -m unittest discover -s tests -v               # 单元测试
```

## 后端抽象（本项目如何适配 DeepSeek Harness）

核心模块（`sources` / `storage` / `contracts` / `ranking` / `report` / `pipeline`）**不依赖任何具体模型**。模型调用统一收敛为一个接口：

```
run(prompt: str, schema: dict, folder: Path) -> dict
```

实现在 `scholar_daily/agents.py`：

- `CodexBackend` —— 通过 `codex exec` CLI 调用。
- `SessionBackend` —— 与当前 Codex 会话交换文件任务；`TaskPending` 暂停流水线，`resume` 校验响应并继续同一次运行。
- `HarnessBackend` —— 通过 `deepseek-harness-sdk` 调用 DeepSeek Harness。
- `get_backend(config, workspace)` 工厂；`config["agent"]["backend"]` 决定用哪个。
- 默认 `codex-session`，沿用当前会话的模型和推理强度；Windows 首次运行 `scripts/scholar-daily.ps1 setup`，用本机 Anaconda 创建项目内 `.conda` 环境。非交互终端加 `-NonInteractive`。Harness 依赖通过 `[harness]` extra 安装。

**新增其他国产 agent 后端只需实现 `run(prompt, schema, folder) -> dict`，返回满足给定 schema 的 dict**；不要改动核心模块。

## DeepSeek Harness 适配约定（改动 HarnessBackend 前必读）

- SDK：`deepseek-harness-sdk`（当前 0.1.5rc1），`from deepseek_harness import DeepSeekHarness`；**要求 Python ≥ 3.10**。
- 复用同一个长生命周期 `DeepSeekHarness`（懒加载 + `close()`），每次 `run()` 用独立 session（不传 `session_id`）。
- `run()` 返回 `RunResult`；只认 `result.finish_reason == "completed"` 为成功，其余（`max-tokens` / `error` / `None`）一律按失败处理并保留 `final_response` 原文到 `response.json`。
- 配置字段：`agent.harness.{dsh_home, provider, model, profile, reasoning_effort, max_tokens, initialize_timeout_seconds}`。
- `dsh_home` 必须是**独立 home**（SDK 不会自动发现 `~/.dsh`），也可用环境变量 `DSH_HOME`；`model` 不能为空。
- `project.root` 默认 `.`，相对配置文件目录解析；相对 `dsh_home` 以项目根为基准，默认 `.harness-home`。
- 凭据：`DEEPSEEK_API_KEY`；可选 endpoint 为 `DEEPSEEK_BASE_URL`。不在配置中保存密钥。
- SDK 仍是 developer preview，接口可能变化；改动前先核对当前 SDK 版本，不要把旧接口假设硬编码进核心。

## 不可破坏的边界（迁移/重构时必须保留）

- `contracts.py` 的 JSON Schema（`ANALYSIS_SCHEMA` / `IDEA_SCHEMA`）与 `validate_shape` 是稳定契约。
- 证据门槛：摘要不冒充全文；每条贡献/结论需要一段原文连续引文且被程序核对；评分维度须有对应证据；idea 必须引用本次已校验的 Evidence Record。
- 稳定数据格式：JSON / SQLite / Markdown。替换模型层时只换适配器，不迁移历史知识格式。
- 报告原子写入（`atomic_text`）；返回码 `0`=完整成功，`1`=配置/致命失败，`2`=等待会话任务的阶段报告或部分失败报告。`waiting` 保留 run_id 与材料快照，最终报告才更新 latest.md。
- 资源诚实：不把多卡显存相加当可用预算；未知条件显式标 `needs_info`，不编造显存/GPU 小时/新颖性。

## 跨平台

- 文件锁：`scholar_daily/_compat.py`（Windows `msvcrt.locking`，其他 `fcntl.flock`）。
- 进程终止：Windows `terminate`/`kill`，POSIX `killpg`。
- `schedule` 在 Windows 生成 Task Scheduler XML 与 PowerShell runner，在其他平台保留 systemd 输出；生成过程不注册任务。
- 不要在模块顶层 `import fcntl` 等仅 Linux 的库；走 `_compat`。

## 目录与技能入口

- `scholar_daily/` 核心包；`tests/` 单元测试；`config.json` 配置。
- `.agents/skills/scholar-daily/SKILL.md` 是 Codex 技能入口，引用根目录 `SKILL.md`；会话模式通过文件任务驱动现有证据校验流程。
- `.dsh/skills/scholar-daily/SKILL.md` 是 **DeepSeek Harness 的技能入口**（DSH 约定），Codex 不读 `.dsh/`。
- 根目录 `SKILL.md` 保存完整技能说明。`scripts/package_skill.py` 用白名单生成分享 ZIP；个人配置、`.conda/`、`.harness-home/` 和运行产物留在本机。
- `reports/` `data/` `cache/` `demo-output/` `smoke-output/` `deploy/` 是运行时产物，勿提交。
