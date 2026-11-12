# MCP 文献工具

项目提供一个基于官方 MCP Python SDK 的本地 Server，以及可被 Planner 和其他代码调用的 Client。当前协议层使用 MCP SDK `1.26.0`；工具发现通过 `tools/list`，调用通过 `tools/call`，结果同时保留可读文本和 `structuredContent`。Server 给每个工具声明 `inputSchema` 和 `outputSchema`，Client 在调用前后做本地 JSON Schema 校验，并检查 `isError`。

## 安装

在项目 Anaconda 环境安装可选依赖：

```powershell
.\.conda\python.exe -m pip install -e ".[mcp]"
```

## Server 工具

启动本地 stdio Server：

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily mcp-serve --config config.json
```

启动只绑定回环地址的 Streamable HTTP Server：

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily mcp-serve --config config.json --transport streamable-http --port 8765
```

Server 暴露四个只读工具：

| 工具 | 作用 | 证据边界 |
|---|---|---|
| `search_papers` | 检索 arXiv 或 Zotero | 返回元数据；摘要不等于全文证据 |
| `read_paper` | 按主题读取论文片段、章节或文档范围 | 返回片段、来源指纹、章节和字符偏移 |
| `verify_citation` | 核对连续引文位置，并查询任务中已有语义审查 | `citation_exists` 与语义支持分开返回 |
| `query_history` | 搜索 SQLite 历史运行、idea、证据和任务报告 | 历史记录默认标记为需要当前任务重新核验 |

四个工具读取业务记录；检索和正文提取的缓存保存在项目内。

## 供其他 Agent 连接

stdio 客户端以本项目 Conda Python 启动服务即可；进程的 stdout 专用于 MCP 消息，日志写 stderr。Windows 服务入口自动补充当前 Conda 环境的 Poppler 路径，外部 Agent 无需先激活终端环境。外部客户端的配置片段可以使用：

```json
{
  "mcpServers": {
    "scholar-daily": {
      "command": "<项目目录>/.conda/python.exe",
      "args": ["-X", "utf8", "-m", "scholar_daily.mcp_server", "--config", "<项目目录>/config.json"]
    }
  }
}
```

本机可直接使用 `deploy/scholar-daily-mcp.json` 中的绝对路径配置片段。HTTP 客户端连接 `http://127.0.0.1:8765/mcp`，当前 HTTP 入口是本机服务。

## 外部 Server Client

配置放在 `config.json` 的 `mcp.servers`。每个 server 要明确配置 `transport`、命令或 URL、`allowed_tools` 和响应大小上限。stdio 命令以参数数组启动，不经过 shell；远程 HTTP 端点要求 HTTPS，只有本机回环地址允许 HTTP。`env_from` 是需要传入子进程的凭据环境变量名数组；`headers_from_env` 将 HTTP 头名称映射到本机环境变量名，不把密钥值写入配置。

本地 stdio 实例由 `MCPConnection` 的异步上下文管理，可在一次连接内复用多个调用；主 Agent 每个工具操作创建并清理会话，成功调用由任务回执复用。工具目录按游标分页，设置调用超时、目录上限和结果字符上限。`allowed_tools` 控制实际调用权限；`readOnlyHint` 等 annotation 仅描述工具行为。

Zotero 的本机配置示例：

```json
{
  "mcp": {
    "zotero_server": "zotero",
    "servers": {
      "zotero": {
        "transport": "stdio",
        "command": "zotero-mcp",
        "args": ["serve", "--transport", "stdio"],
        "env": {"ZOTERO_LOCAL": "true"},
        "allowed_tools": [
          "zotero_search_items",
          "zotero_get_item_metadata",
          "zotero_get_item_fulltext",
          "zotero_get_item_children",
          "zotero_get_recent",
          "zotero_get_annotations",
          "zotero_get_notes"
        ]
      }
    }
  }
}
```

如果 Zotero MCP 返回本地 `file://` 附件路径，可以额外填写 `mcp.zotero_storage_root`，例如 Zotero 的 `storage` 目录；相对路径以 `project.root` 为基准。适配器只在该显式目录内读取 PDF，使用 `pdftotext -layout`，并将 PDF SHA-256、附件 key 和提取器写入外部材料回执。没有显式 storage 根目录时，`file://` 附件不会被读取。

查看外部工具目录：

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily mcp-tools --config config.json --server zotero
```

参数文件 `recent.json` 内容为 `{"limit":5}`。调用已经发现且在 `allowed_tools` 中的外部工具：

```powershell
.\.conda\python.exe -X utf8 -m scholar_daily mcp-call --config config.json --server zotero --tool zotero_get_recent --arguments recent.json
```

## 主 Agent 路由

通过 `research --goal "从 Zotero 文献库查找目标论文并核查方法"` 启动后，Planner 可使用以下三个工具：

- `mcp_list_tools`：发现指定 server 的工具、输入 Schema、输出 Schema 和 Function Calling 映射。
- `mcp_call_tool`：调用已发现且允许的工具，按输入 Schema 校验参数，并把原始结果写入 `tool-calls/<fingerprint>/mcp-result.json`。
- `mcp_import_paper`：通过 Zotero 的 metadata/fulltext 工具导入论文，冻结任务内正文索引；随后仍需 `analyze_papers` 和 Verifier 才能形成项目证据。

Planner 的 `mcp_call_tool.query` 格式为：

```json
{"server":"zotero","tool":"zotero_get_recent","arguments":{"limit":5}}
```

外部 MCP 返回的内容标记为 `external_data`，会保存在 `state.json`、`report.md` 和 `memory.json` 中，但不会自动写入 Evidence Record，也不会绕过连续引文和语义支持审查。失败结果会保留 `mcp-error.json`，便于恢复和诊断。相同的调用输入会复用工具回执，任务恢复不会重复请求。

主 Agent 也会把每个已发现工具转换为 Function Calling 兼容定义：`type=function`、稳定的安全函数名、原始 MCP 工具名路由和原始 `inputSchema`。Function Calling 只负责选择工具；真正调用仍经过 MCP `tools/call`，并检查结构化结果。

## Zotero 联调边界

Zotero MCP 使用本地 `ZOTERO_LOCAL=true` 时仍要求 Zotero 桌面程序运行。本地 API 不可达时，MCP Server 可以初始化和发现工具，但文献调用会返回错误；项目会保留错误并停止当前导入，不生成虚构论文。

Zotero 条目没有可读取附件时，适配器返回 `abstract_only`。元数据、摘要和附件正文分开解析，摘要不会被当作全文证据。当前 `zotero-mcp 0.1.3` 在本机可能返回 `file://` 附件重定向；只有配置 `zotero_storage_root` 后，项目才会在该目录内做受限 PDF 回退读取。

## 测试

```powershell
.\.conda\python.exe -X utf8 -m unittest discover -s tests -p test_mcp.py -v
.\.conda\python.exe -X utf8 -m unittest discover -s tests -v
```

Function Calling 映射保留 MCP 参数 Schema；这里生成通用函数定义，具体模型 provider 的调用封装由对应后端处理。

测试覆盖 stdio 初始化、工具发现分页、输入和输出 Schema、`isError`、响应上限、Streamable HTTP、本地回执幂等、Planner 路由、Zotero 元数据／正文分离、PDF 附件回退边界和路径越界保护。

## 协议与实现参考

- [MCP Tools，2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)：工具 Schema、发现、调用、结构化内容和错误结果。
- [MCP Transports，2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)：stdio 与 Streamable HTTP 消息传输。
- [官方 Python SDK v1.26.0](https://github.com/modelcontextprotocol/python-sdk/tree/v1.26.0)：`ClientSession`、stdio/HTTP Client 及低层 Server。
- [Zotero MCP 项目](https://github.com/54yyyu/zotero-mcp)：本机 `zotero-mcp-server 0.1.3` 的工具实现；项目适配器核对并保留其实际返回格式。
