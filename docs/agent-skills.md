# 可复用研究技能

项目提供四个 0.1.0 版本的技能。组织采用 Agent Skills 的 `SKILL.md`、YAML 名称与描述，以及按需读取的脚本、参考资料和样例（[官方规范](https://agentskills.io/specification)）。

| 技能 | 输入 | 执行方式 | 产物 |
| --- | --- | --- | --- |
| paper-screening | 论文元数据、检索词、可选时间窗口 | 本地去重和排序 | 候选列表、筛除理由、排序依据 |
| evidence-verification | 实际正文片段、候选 analyses、用户事实 | 独立 Verifier + Quality Gate | 断言审查、Evidence ID、失败原因和建议 |
| method-comparison | 已有编排任务目录、所选论文 ID | 有效证据投影 + 独立产物审查 | 创新、方法、结果、局限比较表 |
| experiment-design | 研究目标、可选论文和用户条件 | Research Orchestrator 动态执行 | 最小实验、baseline、指标、反证条件和资源缺口 |

`Tool` 是可以调用的具体操作，例如 `read_papers`；`Skill` 提供适用条件、执行知识、输入输出和失败处理，可以组合多个 Tool；`Agent` 根据目标和任务状态选择下一步。技能脚本复用 `scholar_daily.skill_runner` 和已有模块，各技能不维护一套独立的证据校验逻辑。PDF 提取与阅读由现有 Reader 提供。

## 目录与发现

```text
skills/
  catalog.json
  <skill-name>/
    SKILL.md
    scripts/run.py
    references/contract.json
    assets/example.json
.agents/skills/<skill-name>/SKILL.md
```

`skills/` 保存正文和资源，`.agents/skills/` 保存项目发现入口并引用正文；完整项目或分享 ZIP 是运行依赖的交付单位。脚本需要同一项目的 Python 包和 Anaconda 环境，仅拷贝单个技能文件夹不能获得运行器。发现时读取 `catalog.json` 的名称、描述和版本，不把四份正文同时加入 Planner 上下文。`load_skill` 读取选定正文，`load_skill_resource` 读取该技能已声明的单个资源。资源加载校验路径白名单、版本和 SHA-256。加载知识不会执行脚本。

## 调用

使用已有项目 Anaconda 环境，在项目根目录执行。动态实验设计和已有任务读取需要 `[orchestrator]` extra。

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily skill-list
.\.conda\python.exe -X utf8 skills/paper-screening/scripts/run.py prepare --input skills/paper-screening/assets/example.json
.\.conda\python.exe -X utf8 skills/paper-screening/scripts/run.py run --input skills/paper-screening/assets/example.json
.\.conda\python.exe -X utf8 skills/evidence-verification/scripts/run.py run --config config.json --input skills/evidence-verification/assets/example.json --backend codex-session
.\.conda\python.exe -X utf8 skills/evidence-verification/scripts/run.py resume --run-dir <返回的运行目录>
.\.conda\python.exe -X utf8 -m scholar_daily research --config config.json --goal "比较所提供论文并设计最小实验" --skill experiment-design --input input.json
```

`prepare` 校验输入契约，不调用模型或联网；`run` 创建运行目录；`resume` 使用原输入、配置与账本继续；`status` 查看状态。会话等待结果中的 `pending.task` 给出 prompt/schema/response 路径，当前 Agent 必须读完整请求并按 schema 写响应。字段错误保留原请求和预算，修正后继续。实验设计的 `waiting_input` 用 `resume --answer facts.json` 补充事实。

筛选默认本地执行；需要获取新论文时由 Agent 调用 `collect` 或编排器 `search_papers`，再将候选交给筛选技能。时间窗口检查采用 updated，缺失时采用 published，要求带时区；筛选结果记录执行时间。元数据分数表示检索优先级。

证据核查保留全部候选断言；引文不匹配和过短引文在程序审查中失败，连续引文存在仍需语义审查。审核为空或预算不足时交付缺口。方法对比从原任务读取审查快照，要求至少两篇论文有当前有效的通过证据；比较失败时结果文件不宣称候选表已批准，候选材料保留在 state.json。自动表格的 comparability 保留 needs_info，评测条件齐全前不能据此排名性能。

实验设计脚本预加载所选技能，并固定要求 experiment；deployment=true 时固定要求 deployment 和 experiment，Planner 重规划必须保留这些产物。方案中的事实前提与资源说明仍需 Verifier 和 Quality Gate；未知条件保留 needs_info，并支持暂停和补充事实。

返回码：0 表示准备/状态查询成功或完整交付；1 表示输入、配置或版本错误；2 表示等待或部分交付。结果 JSON 包含技能名称、版本、输入指纹、运行目录；审查和实验调用保留 Token 账本与模型回执。恢复不会重新请求已经记录为完成的逻辑调用。

## 版本与分享

版本写在每个 SKILL.md 的 metadata.version 及 contract.json 的 version。修改技能正文、契约、脚本或样例后同步提高版本，再生成目录指纹和发现入口：

```powershell
.\.conda\python.exe -X utf8 scripts/sync_skill_catalog.py --entries
.\.conda\python.exe -X utf8 scripts/package_skill.py --output dist/scholar-daily-windows.zip
```

内容与 catalog 指纹不符时拒绝加载。任务固定代码和资源版本，代码或目录版本变化后需要用原版本恢复或创建新任务。版本文件可纳入 Git；当前实现不自动提交。分享脚本使用白名单，包含四个技能、核心包和测试，不包含个人配置、密钥或运行产物。

## 验证范围

```powershell
.\.conda\python.exe -X utf8 -m unittest discover -s tests -p test_skills.py -v
.\.conda\python.exe -X utf8 -m unittest discover -s tests -v
```

各 assets/example.json 都是虚构材料。筛选与证据核查样例可以直接 prepare/run；方法对比样例的任务路径及论文 ID 应替换为实际任务数据。测试使用隔离输入和受控模型响应，覆盖契约、证据筛选、按需加载、失败恢复、版本变化和打包；它们不衡量真实模型的语义准确率、学术发现能力或协作收益。
