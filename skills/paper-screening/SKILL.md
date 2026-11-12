---
name: paper-screening
description: 按研究检索词、时间窗口和来源信息筛选论文候选，提供去重、排序依据及待精读列表。用于文献初筛；全文结论核查使用 evidence-verification，多论文比较使用 method-comparison。
metadata:
  version: "0.1.0"
  runtime: "scholar_daily.skill_runner"
---

# 论文筛选

把论文元数据整理为有来源、可解释排序的候选列表。初筛分数表示关键词相关性、时间和社区关注信号，不能用作论文方法或实验质量分数。

## 输入与运行

使用本项目的 Anaconda Python。需要联网发现论文时先用现有 `collect` 或 Orchestrator 的 `search_papers`；本技能脚本筛选已取得的候选，不调用模型。

需要构造输入时读 [输入输出契约](references/contract.json)。输入为 `papers`、英文 `queries`，可选 `days` 与 `limit`；Paper 字段沿用 `scholar_daily.models.Paper`。已有配置可直接使用项目内 `.conda/python.exe`。

```powershell
.\.conda\python.exe -X utf8 skills/paper-screening/scripts/run.py prepare --input input.json
.\.conda\python.exe -X utf8 skills/paper-screening/scripts/run.py run --input input.json
```

运行路径可随项目移动。其他项目使用这些脚本时，需要在其 Anaconda 环境安装本项目运行库，并保留本技能与 `skills/catalog.json`。

## 执行流程

1. 明确用户要找的主题与时间窗口，将中文主题转成用户可核查的英文检索词。只有需要收集新材料时才联网。
2. 检查输入来源与日期，按 arXiv 基础 ID 去重；保留实际版本与来源链接。指定窗口时剔除未知日期、未来日期和超出窗口的候选。
3. 用既有 `rank_papers` 排序，展示筛选依据。相关性、新近程度和社区信号权重分别为 0.65、0.25、0.10；质量评估需要另行阅读和证据审查。
4. 按实际数量交付候选、排除原因及输入/技能版本指纹。需要质量判断时，将候选交给 Reader，并使用 evidence-verification；技能可组合工具，不替代 Agent 的任务决策。

## 失败处理与测试

检索源失败时保留来源错误，由调用 Agent 决定重试、改写查询或部分交付。候选不足、零结果均按实际数量返回；不补造论文。输入字段错误返回1；完整候选筛选返回0。

离线样例 [assets/example.json](assets/example.json) 全部是虚构夹具，可运行 `prepare` 或 `run` 检查去重和排序。真实任务勿把夹具当文献。

只加载当前技能及本次任务需要的资源。维护时同步修改 frontmatter 与契约版本，再运行 `scripts/sync_skill_catalog.py --entries` 更新指纹和发现入口。
