---
name: scholar-daily
description: 在当前 Codex 会话中检索和阅读论文，生成带原文证据的分析、可检验的研究 idea 和每日报告。用于配置研究方向、收集论文、运行调研或继续未完成的调研。
---

# Scholar Daily

本文件所在目录向上三级即项目根目录。先读取项目根的 [SKILL.md](../../../SKILL.md)，按其任务循环调用 Python 工具。

默认使用 `codex-session`：当前会话直接阅读论文、观察页图并生成 idea，模型和推理强度由用户在 Codex 中的选择决定。用 `run` 准备任务，在返回的 `response_path` 写入 JSON，再用 `resume --session <session_path>` 校验并继续，直到最终报告生成。
