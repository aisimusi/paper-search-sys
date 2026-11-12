---
name: method-comparison
description: 使用已有任务中通过 Verifier 的证据，对比多篇论文的创新、方法、结果与局限，标注评测条件缺口。用于至少两篇论文的证据比较；未读材料先交给 Reader 分析与审查。
metadata:
  version: "0.1.0"
  runtime: "scholar_daily.skill_runner"
---

# 方法对比

依据已审查的 Evidence Record 生成多论文对比表。只有数据划分、输入尺寸、硬件、精度及评测协议一致或已有充分换算依据时，才能比较数值高低。

## 输入与运行

使用安装了本项目及 `[orchestrator]` extra 的 Anaconda 环境。按 [输入输出契约](references/contract.json) 提供原任务 `task_dir` 和可选 `paper_ids`；任务至少两篇论文具有当前有效的审查通过证据。

```powershell
.\.conda\python.exe -X utf8 skills/method-comparison/scripts/run.py prepare --input input.json
.\.conda\python.exe -X utf8 skills/method-comparison/scripts/run.py run --input input.json --backend codex-session
.\.conda\python.exe -X utf8 skills/method-comparison/scripts/run.py resume --run-dir '<返回的 run_dir>'
```

脚本读取原任务快照，复用 `compare_methods` 整理通过证据，然后独立审查比较产物。使用独立运行目录，保留来源任务标识及版本；原任务记录保持原样。

## 执行流程

1. 确定比较对象、问题和用户关心的条件。缺少正文时使用 `read_papers` 获取/补读，接着 `analyze_papers` 与 Verifier；需要动态安排这些步骤时使用 `research --goal ...`。
2. 验证所选论文的证据仍与当前文本、用户事实和审查范围匹配。失效或未通过证据的论文需要重新审查。
3. 按创新、方法、结果、局限整理已批准的断言，保留 Evidence ID 和来源。缺失类别明确列为证据待补；当前自动表格的 `comparability=needs_info` 表示尚未确认评测条件可直接比较。
4. Verifier 审查比较表中的事实与引用，Quality Gate 决定交付或返回缺口。需要跨论文推断或额外数值比较时，应由 Research 生成并接受独立审查。

## 失败处理与测试

少于两篇有效论文或证据失效时停止生成表格，返回明确原因；由 Agent 安排补读或重审。会话响应按 `pending.task` 完成后恢复本次运行。完整通过返回0，等待或部分结果返回2，输入/配置错误返回1。

[样例](assets/example.json) 给出任务快照路径的输入结构；运行前使用实际 `task_dir` 替换该路径。`tests/test_skills.py` 使用隔离的虚构任务测试有效证据筛选、来源保持及恢复，禁止通过手填 `passed=true` 替代真实任务审查记录。

只加载与当前比较有关的资源；更新版本后运行 `scripts/sync_skill_catalog.py --entries`。
