# 每日研究助手：个人研究问题卡

Question: 待用户填写所选领域的具体研究问题。

Type: exploratory

Hypothesis: 暂未选择科学假设。工具将每天提出 0—3 个带来源的待验证候选。

Why it matters: 让文献阅读围绕课题组已有基础、可用算力和可执行实验形成积累。

Current evidence: 已验证工具的检索、PDF 提取、Codex 分析、证据检查和报告流程。机器人方向仅用来检验连通性，不代表用户研究方向。

Missing evidence:
- 研究领域和不考虑的相邻方向。
- 课题组已有论文、方法、数据集、代码，以及目前卡住的问题。
- 可用 GPU 型号/数量/单卡空闲显存、每个最小实验的 GPU 小时预算。
- 每日调研时间、每天可阅读数量、模型调用预算。

What would support it: 用户选中的 idea 具有可追溯论文依据、明确且未被已有方法覆盖的机制差异，能在资源预算内完成最小对照实验。

What would falsify it: 机制已经存在、原文证据不支持动机、数据不可获得、硬件预算不满足，或最小实验达不到预注册的实际收益。

Minimal next action: 填写 config.json 的 research_profile，在真实领域运行一次 collect 和 run，检查报告后启用定时任务。

Decision: 等待个人配置；工具第一版已经就绪。
