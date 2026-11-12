---
name: evidence-verification
description: 核查论文分析中的原文引文、断言与证据支持关系，输出定位到断言的审查结果和补读或修复建议。用于审核已有分析；需要提供论文正文片段及待核查 analyses。
metadata:
  version: "0.1.0"
  runtime: "scholar_daily.skill_runner"
---

# 证据核查

审查已有科学断言。输入的分析都是待核查材料；通过引文存在性检查后仍需要独立的语义支持诊断。

## 输入与运行

使用安装了本项目的 Anaconda Python。读 [输入输出契约](references/contract.json) 构造 `papers` 与 `analyses`；analyses 沿用 `ANALYSIS_SCHEMA` 的条目，可提供用户明确声明的 `facts`。PDF 正文片段应保留提取来源和版本；摘要仅支持发现信息。

```powershell
.\.conda\python.exe -X utf8 skills/evidence-verification/scripts/run.py prepare --input input.json
.\.conda\python.exe -X utf8 skills/evidence-verification/scripts/run.py run --config config.json --input input.json --backend codex-session
.\.conda\python.exe -X utf8 skills/evidence-verification/scripts/run.py resume --run-dir '<返回的 run_dir>'
```

单次核查复用 Verifier、Quality Gate 及调用账本，不创建 Planner。`run` 返回会话待办时，当前 Agent 完整读取 `pending.task.prompt_path` 和 `schema_path`，按 schema 在 `response_path` 写 JSON，再执行 `resume`。独立后端使用项目配置并计入预算。

## 执行流程

1. 对照原始论文片段检查阅读范围、论文 ID 和候选引文。正文未取得时，事实判断保留未知，不能用摘要冒充正文。
2. Verifier 核对连续引文和来源，按断言拆分审查数字、单位、实验条件、否定、范围及因果强度。引用正确但不支持断言时，记录具体缺口。
3. Quality Gate 给出通过、补读、补检索、修复、补充信息或部分交付建议。不能把不受支持的既有结果改称未来假设来规避核查。
4. 交付审查 JSON、原文定位、通过的 Evidence ID、失败原因及建议。单次脚本会结束本轮核查；调用 Agent 决定是否修复材料并发起新一轮，动态闭环可交给 Research Orchestrator。

## 失败处理与测试

错误 JSON 或遗漏审查项时，按 `validation_error` 修正会话响应；模型重试、Token 预留和超时复用已有约束。预算不足或审查失败返回2并保留回执；格式/配置错误返回1；全部断言通过返回0。相同 `run_dir` 恢复不会重复已经完成的审查请求。

[离线样例](assets/example.json) 是虚构材料，其中候选断言声称 Orin 帧率，原文只描述对照实验；预期应指出硬件和数字缺少依据。样例验证流程，不构成模型审查准确率评测。语义审核默认沿用配置模型，仍需人工复核。

按需读取契约和样例；更新技能及契约版本后运行 `scripts/sync_skill_catalog.py --entries`。
