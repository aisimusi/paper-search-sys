# Context Engineering：长文献与任务上下文

本阶段能力接入动态研究任务 `research`：论文材料保存为任务内原文快照，正文按论文、章节和片段建立索引；Reader 根据主题选择原文，Verifier 审查断言与原文的支持关系；Planner 使用结构化任务记录继续规划。每日 `run/resume` 保持现有入口和证据契约，表格提取现在额外保存 PDF 页号。

## 运行与上下文策略

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily research --config config.json --goal '比较所提供论文的方法、实验结果与局限' --input research-input.json
```

`research-input.json` 继续使用 `papers/facts/required_outputs`。可加入可选对象：

```json
{
  "context_policy": {
    "planner_mode": "compact",
    "max_context_tokens": 6000,
    "max_fragments": 8,
    "audit_sample_size": 6,
    "audit_seed": 20261014
  }
}
```

这些值也是默认值。`max_context_tokens` 为每篇阅读文本及 Planner 状态视图的本地估计目标；论文片段同时受 `sources.max_text_chars` 限制。该值不包含提示词规则、响应 Schema、研究背景或后端额外输入，所以并不等于一次模型调用的实际总输入。Planner 最小必要状态仍超过目标时，保留事实、证据 ID、信息缺口与待办，并记录 `memory_budget.overflow=true`。

本地估计统一使用 `ceil(UTF-8 字节数 / 4)`，用于比较输入长度和控制选文规模；它没有采用具体模型的分词器，不能替代供应商用量或作为实际计费依据。

## 分层索引与补读

任务目录的 `context/papers/<材料指纹>/` 保存 `index.json`、`layout.txt` 和可选 `tables.txt`。快照使用 SHA-256 校验，后续补读复用同一次任务的材料；外部缓存更新不会改变已经冻结的材料，快照被修改会导致校验失败。

索引保留：

- 论文：`paper_id/title/abstract/url/version`。
- 章节：`section_id/name/start/end/is_reference`。
- 片段：`fragment_id/section_id/start/end/page_start/page_end`。
- 来源：`source_id/kind/sha256/characters/page_basis`；正文排版文本和表格提取文本分别存储。

首次发现可直接使用摘要和元数据。需要分析时，`read_papers.query` 指定英文主题，例如 `latency batch size ablation`。同一主题第一次读取相关片段，第二次扩大至相应章节，第三次在预算内读取更大原文范围。扩大范围仍受字符和估计 Token 限制，不代表全文已经读完；`selection.budget_exhausted` 记录截断情况。没有匹配时返回明确错误，Planner 可换查询、补检索或交付部分结果。

检索采用关键词频次和章节标题加权的确定性排序。无关键词匹配的初次阅读按章节分散选择方法、实验、结果等内容；默认跳过已识别的参考文献区域，查询明确包含 `references/bibliography/参考文献` 时可以读取。章节识别是标题规则，PDF 排版异常、跨行标题和表格结构损坏仍需核对。图像解读继续使用现有独立视觉来源，不能用于正文连续引文。

## 统一跨论文检索接口

`scholar_daily.retrieval.Retriever` 为生产流程、MCP 服务和评测代码提供独立的候选检索入口，保留原文快照和业务状态。调用者先绑定已获访问权限的论文元数据和快照记录，再调用 `retrieve(query, paper_ids, top_k=10, method="bm25")`，或显式选择 `method="dense"/"hybrid"`。Dense 分支生成和复用向量缓存。当前生产流程的 `select_context()`、`read_papers` 补读层级及 MCP `read_paper` 默认策略保持原有行为；本接口由调用者显式使用。

```python
from pathlib import Path
from scholar_daily.research_calls import read_json
from scholar_daily.retrieval import Retriever

task_dir = Path("reports/research-tasks/<task_id>").resolve()
state = read_json(task_dir / "state.json")
retriever = Retriever(state["paper_contexts"], state["papers"], folder=task_dir)
result = retriever.retrieve(
    query="retrieval latency",
    paper_ids=["<paper_a_id>", "<paper_b_id>"],
    top_k=5,
    method="bm25",
)
for hit in result["hits"]:
    print(hit["paper_id"], hit["section"], hit["score"], hit["text"])
for issue in result["issues"]:
    print(issue["paper_id"], issue["code"], issue["message"])
```

构造参数直接接受 `ResearchState.papers/paper_contexts` 的现有字典；每篇元数据至少包含 `paper_id/updated`，快照记录至少包含 `index_path/source_fingerprint`。MCP 调用者也可通过现有服务查找已授权记录，再构造这两个映射。传入 `folder` 会沿用 `load_index()` 的任务目录边界检查；省略时由调用者保证记录和路径的访问权限。接口不会自动扫描任务、下载全文或重建缺失索引。

返回值是可 JSON 序列化的 `RetrievalResult`，包含 `query/method/hits/issues`。每条 `RetrievalHit` 包含：

| 字段 | 含义 |
|---|---|
| `paper_id/version` | 论文 ID 与快照对应版本 |
| `fragment_id` | 现有三级索引的片段 ID |
| `source_fingerprint/index_path` | 原快照记录的 24 位索引指纹和索引路径 |
| `source_id/source_kind/source_sha256` | 来源 ID、`layout` 或 `tables`、完整原文 SHA-256 |
| `section_id/section` | 章节 ID 与名称 |
| `page_start/page_end/page_basis` | 索引记录的页码及页码依据，未知页码为 `null` |
| `start/end/offset_unit` | 原文快照中 Unicode 字符零起始半开区间 `[start,end)` 及单位说明 |
| `text/score/method` | 完整片段文本、BM25 分数/Dense 余弦相似度/Hybrid RRF 分数、检索方法 |
| `verification_status` | 固定为 `unverified`，候选仍须经过连续引文校验、Verifier 和 Quality Gate |

`paper_ids=[]` 表示空语料；重复论文 ID 仅加载一次。跨论文相同文本和相同 `fragment_id` 保留各自出处，定位时应联合使用 `paper_id/source_fingerprint/fragment_id`。同一来源同一片段的重复索引行只加入语料一次；相同文本位于不同字符范围时保留全部位置。相同文本来自多篇论文时，每个出处均参与语料统计，当前未对转载论文或重复内容降权。

BM25 在所选论文的有效片段集合上统一计算文档频率和平均长度，`k1=1.5, b=0.75`。默认跳过已识别的参考文献区域；查询明确包含英文 `reference/references/bibliography` 或中文 `参考文献` 时允许检索该区域。只返回分数大于零的片段，空白、无词项或无匹配查询返回 `hits=[]`；空白及无词项查询不加载材料。同分按论文 ID、来源类型、片段位置排序，与请求中论文排列顺序无关。分数用于当前语料内排序，不是证据支持概率，也不适合直接比较不同语料的分数。

`lexical_tokens()` 使用小写 ASCII 字母数字词项，以及汉字连续段内重叠的相邻双字词项；单独出现的一个汉字也保留为词项。无需中文分词依赖，但单汉字查询不会命中较长汉字段内部的字符，也不处理同义词、繁简转换或中英翻译。

共享评分函数 `bm25_retrieve()` 从 `benchmark.py` 抽取，保留原公式。`benchmark.retrieve()` 通过传入原英文 `tokens` 和 `include_zero=True` 保持既有评测的分词、排序、零分补齐及返回字段。统一检索器使用共享函数的默认正分过滤和中英文分词。评测框架可显式调用统一检索器，已有 QASPER 默认入口保持原实验口径。

每次有效查询都重新加载快照，复用索引指纹和原文 SHA-256 校验，并核对论文 ID、版本与片段范围。单篇失败记录在 `issues` 中，其他有效论文继续参与检索；调用者必须检查 `issues` 才能判断是否覆盖全部请求论文。问题代码为：

- `paper_missing`：缺少论文元数据。
- `index_missing`：缺少论文快照记录。
- `material_missing`：索引文件或原文快照文件缺失。
- `material_unavailable`：文件读取被拒绝等 I/O 错误。
- `stale_material`：快照版本与当前论文 `updated` 不一致。
- `invalid_material`：指纹、论文 ID、目录边界、索引结构或片段范围校验失败。

无效参数（包括非正整数 `top_k` 和未支持的 `method`）抛出 `ValueError`。有效论文集合变化会改变 BM25 语料统计及分数。任务冻结快照仍是检索来源，外部缓存更新不替换该快照。

BM25 每次在内存中加载片段、分词和评分，适合任务级语料。Dense 复用同一材料校验与片段构建过程，在独立模块中生成和缓存向量，所有候选仍保持 `unverified`；Planner 或 MCP 的生产策略接入应另行显式配置和评测。

## Dense Retrieval 与向量缓存

可选依赖通过项目现有 Anaconda 环境安装。当前验证的 CPU PyTorch 版本为 `2.14.1+cpu`，`dense` extra 固定 `faiss-cpu==1.15.1` 和 `sentence-transformers==6.1.0`。FAISS 使用 CPU `IndexFlatIP`，Embedding 设备可单独配置：

```powershell
.\.conda\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.conda\python.exe -m pip install -e '.[dense]'
```

`embeddings.py` 提供冻结的 `EmbeddingConfig` 和延迟加载的 `SentenceTransformerEmbedder`。本机默认选用 `intfloat/multilingual-e5-small`，固定版本 `614241f622f53c4eeff9890bdc4f31cfecc418b3`，384 维、512 token、batch size 4、CPU float32。检查时 RTX 2060 为 6 GiB、空闲约 3.36 GiB，系统内存共 15.91 GiB、空闲约 3.83 GiB；因此先使用较小模型验证本地检索。E5-small 的输入限制及检索前缀来自[官方模型说明](https://huggingface.co/intfloat/multilingual-e5-small)。BGE-M3 为 1024 维、最多 8192 token，查询可不加指令，配置依据见[官方说明](https://huggingface.co/BAAI/bge-m3)。

默认 `local_files_only=True`，不在检索请求中下载模型。可预先使用 Hugging Face 的 `snapshot_download` 下载固定 revision，并设置 `model_id` 为返回的本地快照目录；也可通过 `cache_folder` 指定已下载的 Hugging Face 缓存位置。显式设置 `local_files_only=False` 才允许模型加载器联网。远程模型 revision 必须为完整 commit SHA；本地目录中的配置、分词器和权重文件会另算 SHA-256，绑定到向量缓存。

```python
from pathlib import Path
from scholar_daily.embeddings import EmbeddingConfig
from scholar_daily.retrieval import Retriever

config = EmbeddingConfig(
    model_id=str(Path("cache/embedding-models/multilingual-e5-small-"
                      "614241f622f53c4eeff9890bdc4f31cfecc418b3").resolve()),
    # revision、dimension、max_length 和检索前缀沿用上述 E5 默认值。
)
retriever = Retriever(state["paper_contexts"], state["papers"], folder=task_dir,
                      dense_config=config)
bm25_result = retriever.retrieve("retrieval latency", paper_ids, top_k=5)
dense_result = retriever.retrieve("检索延迟如何测量？", paper_ids, top_k=5, method="dense")
print(dense_result["issues"])
print(retriever.dense_stats)
```

E5 对查询添加 `query: `，对片段添加 `passage: `；随后统一转为 float32 并执行 L2 归一化。FAISS 内积由此对应余弦相似度，分数约在 `[-1,1]`，保留零分及负分候选，不设语义相关性阈值。BM25 的正分过滤保持原有行为；两类分数的含义分别由 `method` 标识，不能直接比较。向量分数不表示引文真实性或断言支持程度。[FAISS 度量说明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

Embedding 最大长度包含前缀及特殊 token。超长片段按配置截断后编码，返回的 `text/start/end` 仍指向完整索引片段；本阶段保留原有分块和原文定位。需要核查结论时仍须阅读返回原文并走原有证据流程。更换模型时需一起配置其 `dimension/max_length/query_prefix/document_prefix`，例如 BGE-M3 使用空前缀；不会自动猜测不同模型的检索协议。

`dense_retrieval.py` 按论文保存缓存：

```text
context/papers/<source_fingerprint>/dense/<cache_key>/
    .lock
    index.faiss
    manifest.json
```

传入可选 `dense_cache_dir` 时，路径为 `<dense_cache_dir>/<paper_id 的 SHA-256>/<cache_key>/`，可复用到不同任务。调用者负责该目录的访问权限，目录中只应保存可信的本地生成缓存。

manifest 的 `identity.fragments` 顺序就是 FAISS 行号到片段的映射。缓存键覆盖论文 ID、版本、原索引指纹、来源 SHA-256、片段 ID、字符边界、文本 SHA-256，以及 Embedding 模型版本、维度、前缀、长度、精度、设备、批量设置、归一化版本和相关依赖版本；本地模型额外绑定文件指纹。有效分块边界的变化会产生新键，不需要修改原三级索引格式。模型对象与签名在检索器生命周期内固定；更换模型或配置应新建 `Retriever`。

每次检索先校验原文材料。缓存命中时，检查 manifest 身份、索引字节 SHA-256、FAISS 类型、维度、行数和向量归一化状态；只生成当前查询向量。按论文检索后合并排序，支持单篇及多篇范围。当前使用精确索引，取出各篇分数后统一处理同分排序，适合任务级语料。

材料或模型身份变化会使用新键，仅重建受影响论文。缓存文件缺失、损坏或映射不一致时重建当前键；旧键保留，不自动删除。读取和构建使用同一个按键文件锁；索引通过临时文件、flush/fsync 和原子替换写入，manifest 最后发布。进程中断导致文件组合不完整时，下一次校验会触发重建。并发占锁、权限不足或重建失败会返回明确问题。

Dense 继续返回 `query/method/hits/issues`，沿用全部定位字段和 `verification_status="unverified"`。模型或可选依赖仅在有效 Dense 请求时初始化。新增问题代码如下：

- `dense_initialization_failed`：FAISS/NumPy 导入、配置或 Dense 初始化失败。
- `dense_query_failed`：模型加载、查询编码或查询向量校验失败。
- `dense_index_failed`：某篇论文的向量生成、持久化、锁或 FAISS 检索失败；其他论文继续参与。

前两类全局问题使用 `paper_id=""`，单篇问题保留实际 ID。向量必须是数量、维度正确的有限实数非零矩阵，查询和文档采用相同归一化。单独调用 Dense 失败不会转用 BM25。空白查询、空论文范围或没有有效片段时不初始化模型。`Retriever.dense_stats` 单独记录最近一次 Dense 或 Hybrid 中 Dense 分支的缓存状态、每篇索引耗时、查询编码和 FAISS 搜索耗时，不改变结果契约。

真实模型冒烟测试使用已有索引，不访问 Planner 或修改证据状态：

```powershell
.\.conda\python.exe scripts/smoke_dense_retrieval.py --index '<现有 index.json>' --model '<本地模型目录>' --query '检索主题' --cache-dir 'smoke-output/dense-smoke-cache' --output 'smoke-output/dense-smoke.json'
```

`--index` 可重复传入不同论文，也可用 `--task-dir` 读取现有任务。脚本记录模型加载、首次检索、三次缓存查询和新检索器磁盘复用耗时，并检查每条命中的原文与字符范围一致。首次构建与缓存命中状态以 JSON 中的 `build_stats/reload_stats` 为准。

2026-10-19 本机真实验收使用 LW-DETR 与 MOTIP2 的已有快照，默认过滤参考文献后共 100 个片段。环境为 Windows x64、Anaconda Python 3.11.17、PyTorch 2.14.1+cpu、FAISS 1.15.1、NumPy 2.4.6、Sentence Transformers 6.1.0、Transformers 5.20.1，PyTorch/FAISS 均设为 4 线程。模型已下载并禁用网络；加载模型及首条查询共 14.219 秒，首次 Dense 检索共 10.063 秒（其中两篇论文编码、构建及持久化分别为 4.262、4.135 秒）。三次缓存查询为 80.45、34.86、32.04 毫秒，中位数 34.86 毫秒；新检索器复用磁盘缓存为 1.438 秒，包含一次本地模型文件指纹计算，复用了已加载的模型。原文定位检查通过、`issues=[]`；记录在运行产物 `smoke-output/dense-phase2-real-smoke.json`。这是本机链路与定位验收，未测检索准确率。

## Hybrid Search 与 Reciprocal Rank Fusion

`retrieval.py` 的 `rrf_fuse()` 实现等权 RRF，`Retriever.retrieve()` 的 `hybrid` 分支完成方法分派。一次 Hybrid 调用只加载和校验一批论文材料，将同一个有效片段列表传给现有 `bm25_retrieve()` 与 `DenseRetriever.retrieve()`，各取 Top-N 后融合。默认方法仍为 BM25；Dense、Embedding、QASPER 及生产编排保持原有实现。

```python
result = retriever.retrieve(
    query="inference latency", paper_ids=paper_ids, top_k=5,
    method="hybrid", candidate_k=20, rrf_k=60,
)
for hit in result["hits"]:
    print(hit["paper_id"], hit["fragment_id"], hit["rrf_score"],
          hit["bm25_rank"], hit["dense_rank"], hit["retrieval_sources"])
print(result["issues"])
```

新增参数为关键字参数，仅在 `method="hybrid"` 时使用和校验：`candidate_k` 是两路共同的候选窗口，默认 20，必须是正整数且不小于最终 `top_k`；`rrf_k` 默认 60，允许非负整数。布尔值不作为整数接受。`top_k=21` 时必须显式设置 `candidate_k>=21`，否则抛出 `ValueError`，不自动改变用户设置的窗口。

融合公式为：

```text
score(d) = Σ[r 为命中 d 的检索路] 1 / (rrf_k + rank_r(d))
```

每路在全部所选有效论文范围内排序，排名从 1 开始；不在某一路窗口内的片段，该路贡献为 0。原始分数只保留为诊断，不参与相加或归一化融合。算法出处：G. V. Cormack、C. L. A. Clarke、Stefan Büttcher（2009），*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*，SIGIR 2009，[作者提供的原论文](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)。本项目的候选窗口和 `rrf_k` 尚未在论文语料上调参验证。

合并键严格为 `(paper_id, source_fingerprint, fragment_id)`。两路命中同一键时保留一条结果；不同论文的相同正文保留独立出处。同分按这三个标识字段的字典序排序。每路重复键仅使用首次出现的原始排名，避免重复计票。融合过程不修改输入候选。

Hybrid 命中使用 `HybridHit`，继承全部原文定位字段和 `verification_status="unverified"`，并新增：

| 字段 | 含义 |
|---|---|
| `score/rrf_score` | 二者相同，均为 RRF 融合分数 |
| `bm25_score/bm25_rank` | BM25 原始分数与全局候选排名，未命中为 `null` |
| `dense_score/dense_rank` | Dense 原始余弦相似度与全局候选排名，未命中为 `null` |
| `retrieval_sources` | `["bm25"]`、`["dense"]` 或 `["bm25", "dense"]` |

单独调用 BM25 或 Dense 的命中结构不增加这些字段。顶层仍为 `query/method/hits/issues`。部分返回与失败约定如下：

| `hits` | `issues` | 解释 |
|---|---|---|
| 非空 | 空 | 正常返回；某一路正常无候选时可只由另一路贡献 |
| 非空 | 非空 | 部分返回，需检查失败论文或 Dense 分支问题后判断覆盖范围 |
| 空 | 空 | 正常空结果，例如空查询、空论文范围或两路都无候选 |
| 空 | 非空 | 未得到可返回结果，材料或 Dense 错误说明保留在 `issues` |

Dense 初始化或查询失败时，保留原有 `dense_initialization_failed/dense_query_failed`，BM25 的有效候选仍可按单路 RRF 返回；命中来源明确为 BM25，不能将这些结果解释为两路都成功。单篇向量索引失败时保留 `dense_index_failed`，该论文通过原文校验的 BM25 候选仍可贡献；原文材料校验失败的论文则在两路检索前排除。

BM25 沿用正分过滤。Dense 沿用现有 Top-N 余弦候选策略，包含零分或负分且未设相关性阈值，因此 Hybrid 返回片段不代表查询确有相关证据。中文查询可能只有 Dense 贡献，RRF 此时保持 Dense 候选的原有顺序。

RRF 参数和候选窗口不改变向量缓存身份。Hybrid 复用 Dense 缓存命中、指纹校验、失效重建及写入机制。独立自动化测试使用模拟 Embedding 和真实 FAISS；纯 RRF 与路由测试无需模型下载。

真实模型及同批查询对照可运行：

```powershell
.\.conda\python.exe scripts/smoke_hybrid_retrieval.py --index '<论文一 index.json>' --index '<论文二 index.json>' --model '<本地 E5-small 目录>' --cache-dir '<已有向量缓存目录>' --output 'smoke-output/hybrid-smoke.json'
```

脚本默认使用中英文四个查询，保留两路完整候选、三种方法的 Top-K、原始分数、排名、缓存状态和计时，另存 Markdown 对照表。它将 Hybrid 诊断与独立 BM25/Dense 调用逐项比对，核对 RRF 分数、最终顺序、原文回读与 `unverified` 状态。现有模型、本地材料和缓存即可离线运行，无需修改研究任务。

2026-10-20 验收：全量测试 321 项全部通过，其中新增 Hybrid/RRF 测试 24 项。真实 E5-small 在第二阶段相同的两篇论文、100 个有效片段上运行四个查询，每个查询返回 5 条 Hybrid 命中，全部通过原文回读和独立分路排名核对，`issues=[]`。Dense 与 Hybrid 分支全部复用已有向量缓存；四次 Hybrid 调用耗时约 42.57～47.68 毫秒，模型加载单独计时，不包含在该范围内。两条中文查询的 BM25 均无词项命中，Hybrid 保持 Dense 排序。完整结果保存在 `smoke-output/hybrid-phase3-real-smoke.json`，前三名对照见同名 Markdown；回归日志为 `smoke-output/hybrid-phase3-full-tests.log`。

下一阶段可在 Hybrid 候选生成后增加独立的 `rerank(query, hits, top_k)` 适配器，保持原有定位、分路诊断和未核查状态。检索评测先使用独立固定样例及片段相关性标注，冻结材料与查询划分，再比较各方法的 Recall@K、MRR、nDCG@K 和延迟；本阶段的样例对照只验证链路和排序计算，尚不能判断检索质量是否提高。现有 QASPER 入口保持原实验口径。

## 证据定位与支持性结果

Evidence Record 保留原有 `ER-*` 标识，并附加 `index_path/locations`。每个位置关联论文版本、来源指纹、章节、片段、页码和字符范围；重复引文保存全部匹配位置。Verifier 发送一个有界原文窗口，并提示剩余候选位置可能涉及范围歧义。

偏移是快照中 Unicode 字符的零起始半开区间 `[start,end)`，不是 PDF 字节偏移。PDF 文本提取的换页符用于记录物理页序号；表格页号来自提取页号。无法确认 PDF 页号的本地文本填写 `null`，不推测印刷页码。定位只规范化空白，保留负号、连字符、单位与数字。

Verifier 在原有连续引文、来源限制、限定条件和断言覆盖校验上增加统一结果：

| `support_status` | 含义 |
|---|---|
| `supported` | 模型认为必要断言受到支持，且程序校验全部通过 |
| `contradicted` | 模型认为至少一个覆盖的断言与来源矛盾，其来源和连续引文可核对 |
| `insufficient` | 证据、限定条件、覆盖或来源不足；也包含原文定位失败 |

原有细分 `verdict` 与理由保留在审查记录中。`assessment_origin/human_verified/ground_truth` 明确区分模型诊断、程序校验与人工核验。原文中的句子真实存在只是结构校验条件；语义支持性仍是模型判断，需抽样人工检查。

## 人工抽样核验

每次任务导出 `audit-sample.json`，固定 seed 按三类模型结果分层抽样：先从每个存在的类别抽一项，再随机补足，最多 `audit_sample_size` 项。每项包含原子断言、模型结果、原文窗口及定位；冻结副本保存在 `human-audit/<sample_id>/sample.json`。

复制抽样文件为自己的标签文件，仅填写 `human_label/reviewer/notes`；`human_label` 使用 `supported/contradicted/insufficient`，未核验保留 `null`。

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily research-audit --task-dir '<研究任务目录>' --labels '<已填写的人工标签.json>'
```

导入校验抽样版本、全部 case_id 和冻结原文；拒绝修改来源、断言或模型结果。输出混淆矩阵、已标注数量、样本一致比例和具体分歧。人工标签独立保存，不能覆盖模型审查历史或自动改写研究结论。分层样本的一致比例描述该样本，不直接当作总体准确率；未经核验时比例为 `null`。导入后再运行 `research-resume` 可更新任务报告中的人工核验统计。

## 结构化压缩与跨会话恢复

`memory.json` 保存论文 ID、关键方法、Evidence ID、原文索引、审查状态、必要事实、信息缺口、下一步操作和未完成任务，同时记录 `state.json` 的路径及指纹。它是供后续会话读取的任务摘要，完整状态和成功回执仍保存在原任务目录。

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily research-resume --task-dir '<同一任务目录>'
```

等待文件任务时先按 `memory.pending` 读取被冻结的 prompt/schema 并提交响应；等待用户输入时按原入口提供 `--answer`。恢复沿用原 task_id、检查点与调用回执，已完成调用不会重复计入。证据或事实变化时，原审查状态会过期，必须重新审查。代码或依赖指纹变化后，沿用既有恢复规则：使用原版本恢复，或新建任务。

`compression.before_estimated_tokens/after_estimated_tokens/reduction_fraction` 比较序列化任务状态的两种视图，只衡量记录体积；不能据此宣称真实模型任务质量未降低或实际调用节省了同样比例。

## 单任务输入用量与压缩质量评测

`context_metrics` 按任务和角色累计：

- `estimated_input_tokens`：每个模型尝试的 prompt 与响应 Schema 的本地估计，包括失败尝试。
- `measured_input_tokens`：事件返回的输入用量，包含缓存输入；部分事件缺少必要字段时该尝试标为未知。
- `attempts/unmeasured_attempts/estimate_missing_attempts`：调用尝试、未实测和历史估计缺失的数量。

等待中的文件任务和成功回执复用不增加尝试数。`research-status`、任务报告和返回 JSON 均可查看指标。原有总 Token 预算及保守预留统计继续单独保存。

固定同一份输入、配置、预算和代码，分别运行两种 Planner 上下文：

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily research --config config.json --goal '固定研究任务' --input research-input.json --context-mode full
.\.conda\python.exe -X utf8 -m scholar_daily research --config config.json --goal '固定研究任务' --input research-input.json --context-mode compact
.\.conda\python.exe -X utf8 -m scholar_daily.context_evaluation --full '<full任务目录>' --compact '<compact任务目录>' --output reports/context-comparison
```

`full` 是使用相同 Reader/Verifier 的完整任务状态视图；`compact` 使用结构化视图。这一比较检验 Planner 状态压缩，不检验分层检索相对于整篇 PDF 的效果。应使用固定本地论文输入，避免两组检索到不同材料；同一论文原文快照不同会拒绝比较。独立后端配置必须一致；会话后端的实际会话模型无法从项目配置核实，需要人工保持相同模型与推理强度。

比较输出 `comparison.json/comparison.md/quality-template.json`。自动统计已交付产物比例和输入用量差值；证据审查通过数仅作为流程诊断。人工质量评分使用四项固定标准：产物完整性、事实支持、限定条件保真和可追溯性，每项填 0 或 1，未核验保留 `null`。打乱同一任务两组评分顺序，保留评分人和逐项结果。

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily.context_evaluation --full '<full任务目录>' --compact '<compact任务目录>' --output reports/context-comparison --ratings '<已填写的质量评分.json>'
```

配对差值均为 `compact − full`。两组全部尝试具有完整供应商输入用量时比较实测值，否则两组都使用本地估计。至少两个已完整评分的任务才计算任务级 bootstrap 区间；它不估计模型重复运行波动。无人工评分时保持 `awaiting_human_ratings`，不输出质量保持或提升结论。

## 验证与参考

```powershell
.\.conda\python.exe -X utf8 -m unittest discover -s tests -p test_context_engineering.py -v
.\.conda\python.exe -X utf8 -m unittest discover -s tests -v
```

测试使用明确虚构材料与受控模型响应，覆盖 Unicode 偏移、页号、表格来源、重复句子、负号与间隔保真、参考文献跳过、预算、快照修改、缓存变化、待办恢复、用量完整性、人工标签版本及质量配对条件。它验证工程行为，不测量模型语义判断的实际准确率。

设计参考为 Anthropic Engineering 的两篇官方文章：

- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：讨论按需加载、上下文压缩与结构化笔记。
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)：讨论进度记录、逐步推进与跨会话连续性。

本项目使用确定性结构化记录与可校验原文快照；上述文章提供工程设计背景，不能作为本项目已经取得质量收益的证据。
