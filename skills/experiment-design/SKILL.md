---
name: experiment-design
description: 基于论文证据、已有数据和单卡资源约束设计可检验的最小实验，包含对照、指标、反证与停止条件。用于研究假设或部署实验方案；部署任务需明确目标设备与运行条件。
metadata:
  version: "0.1.0"
  runtime: "scholar_daily.skill_runner"
---

# 实验设计

把研究问题转成可执行、可证伪的候选实验。已提供论文可直接进入阅读分析；需要更多证据时由 Orchestrator 决定补读或检索。

## 输入与运行

使用安装了本项目 `[orchestrator]` extra 的 Anaconda 环境。按 [输入输出契约](references/contract.json) 提供 `goal`、可选 `papers/facts/limits`。`deployment=true` 表示需先评估目标设备，再设计实验；设备信息可以补充到 `facts.deployment`。

```powershell
.\.conda\python.exe -X utf8 skills/experiment-design/scripts/run.py prepare --input input.json
.\.conda\python.exe -X utf8 skills/experiment-design/scripts/run.py run --config config.json --input input.json --backend codex-session
.\.conda\python.exe -X utf8 skills/experiment-design/scripts/run.py resume --run-dir '<返回的 task_dir>'
```

脚本将当前技能的内容及版本绑定到 Research Orchestrator，复用 Planner、Reader、Verifier、Research 与 Quality Gate。等待期间按 `pending.task` 完成模型文件任务；`pending.kind=user` 时通过 `--answer answer.json` 提交 `facts` 和无法提供的 `unavailable` 字段。

## 执行流程

1. 明确假设、最接近工作、实际拥有的数据和预算。分别记录训练资源与部署设备；单卡可用显存逐卡判断，不能加总多卡显存。
2. 通过文献阅读、引文校验和独立审查取得证据。每个候选实验必须引用本次通过审查的 Evidence ID；新颖性保持待验证，按需进行独立检索。
3. 指定已有 baseline、数据划分、随机种子、单变量对照、指标、反证和停止条件。资源估计写明依据，没有依据的数值保留 null。
4. 部署任务先确认设备型号、内存预算、软件版本、功耗模式、输入与延迟目标；设计算子/导出检查、数值一致性与上机测量，不把论文训练硬件结果当作目标设备实测。
5. Research 产物和最终交付经过 Verifier/Quality Gate。条件不足时补充信息；已知预算不足时重设计，无法解决时交付部分方案与明确缺口。

## 失败处理与测试

保留原任务路径，通过 `resume` 继续同一次运行。修复使用 `repair_result` 并重新审查；模型调用及审查均受 `limits` 约束。返回0表示完整交付，2表示等待或部分交付，1表示输入/配置错误。

[离线样例](assets/example.json) 使用虚构设备与论文用于检查请求结构，不能作为实际部署依据。先执行 `prepare`；功能测试使用受控模型响应，不衡量科学结论或部署性能。

技能是可复用的知识与流程，Tool 负责具体操作，Agent 根据目标选择和组织工具。按需加载当前技能的契约与样例，版本更新后同步目录指纹和发现入口。
