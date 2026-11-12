"""Official MCP SDK client: initialization, discovery, schema checks and bounded results."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
import hashlib
import json
import os
import re
from urllib.parse import urlparse


class MCPToolError(RuntimeError):
    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


def checked_server(profile, root=None):
    if not isinstance(profile, dict):
        raise ValueError("MCP server 配置必须为对象")
    profile = dict(profile)
    allowed_fields = {"transport", "command", "args", "env", "env_from", "cwd", "url", "headers_from_env",
                      "timeout_seconds", "max_response_chars", "allowed_tools"}
    if set(profile) - allowed_fields:
        raise ValueError("未知 MCP server 配置字段：" + ",".join(sorted(set(profile) - allowed_fields)))
    transport = profile.setdefault("transport", "stdio")
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("MCP transport 应为 stdio 或 streamable-http")
    if transport == "stdio":
        if not isinstance(profile.get("command"), str) or not profile["command"].strip():
            raise ValueError("stdio MCP server 需要 command")
        if not isinstance(profile.get("args", []), list) or not all(isinstance(x, str) for x in profile.get("args", [])):
            raise ValueError("MCP args 必须为字符串数组，不通过 shell 执行")
    else:
        parsed = urlparse(profile.get("url", ""))
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise ValueError("MCP URL 需要 HTTP(S) 端点，凭据通过环境变量传入")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("远程 MCP 端点需要 HTTPS")
    for field in ("env", "headers_from_env"):
        values = profile.get(field, {})
        if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
            raise ValueError("MCP %s 必须为字符串映射" % field)
    for field in ("allowed_tools", "env_from"):
        values = profile.get(field, [])
        if not isinstance(values, list) or not all(isinstance(x, str) and x.strip() for x in values) or len(values) != len(set(values)):
            raise ValueError("MCP %s 必须为不重复的字符串数组" % field)
    timeout = profile.setdefault("timeout_seconds", 45)
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ValueError("MCP timeout_seconds 应在 1～300 之间")
    maximum = profile.setdefault("max_response_chars", 1000000)
    if type(maximum) is not int or not 1000 <= maximum <= 2000000:
        raise ValueError("MCP max_response_chars 应在 1000～2000000 之间")
    if profile.get("cwd") is not None:
        from pathlib import Path
        if not isinstance(profile["cwd"], str):
            raise ValueError("MCP cwd 必须为路径字符串")
        profile["cwd"] = str(((Path(root) if root else Path.cwd()) / profile["cwd"]).resolve())
    return profile


def server_profile(config, name):
    servers = config.get("mcp", {}).get("servers", {})
    if name not in servers:
        raise ValueError("MCP server 尚未配置：" + name)
    return checked_server(servers[name], config.get("_root"))


def schema_validator(schema):
    from jsonschema.validators import validator_for
    def references(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if key in {"$ref", "$dynamicRef"} and (not isinstance(child, str) or not child.startswith("#")):
                    raise ValueError("MCP Schema 仅支持本地引用")
                references(child)
        elif isinstance(node, list):
            for child in node:
                references(child)
    references(schema)
    from jsonschema.exceptions import SchemaError
    try:
        validator = validator_for(schema)
        validator.check_schema(schema)
        return validator(schema)
    except SchemaError as exc:
        raise ValueError("MCP 工具提供了无效 JSON Schema：" + exc.message) from exc


def validate_json_schema(value, schema):
    errors = sorted(schema_validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        raise ValueError("MCP 参数或结果不满足 JSON Schema：" + errors[0].message)


def function_catalog(server, tools):
    """Provider-style Function Calling definitions plus an explicit MCP route table."""
    definitions, routes = [], {}
    for tool in tools:
        identity = server + ":" + tool["name"]
        label = re.sub(r"[^A-Za-z0-9_]", "_", "mcp_" + server + "_" + tool["name"])
        name = label[:51] + "_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        definitions.append({"type": "function", "function": {"name": name,
                            "description": tool.get("description", ""), "parameters": tool["inputSchema"]}})
        routes[name] = {"server": server, "tool": tool["name"]}
    return {"tools": definitions, "routes": routes}


def parse_function_call(catalog, name, arguments):
    if name not in catalog["routes"]:
        raise ValueError("Function Calling 返回了未知工具名称")
    if isinstance(arguments, str):
        arguments = json.loads(arguments, parse_constant=lambda s: (_ for _ in ()).throw(ValueError("非法 JSON 数字")))
    definition = next(row for row in catalog["tools"] if row["function"]["name"] == name)
    validate_json_schema(arguments, definition["function"]["parameters"])
    return dict(catalog["routes"][name], arguments=arguments)


class MCPConnection:
    """Multiple calls share one session while inside this asynchronous context."""
    def __init__(self, profile, root=None):
        self.profile = checked_server(profile, root)
        self.stack = AsyncExitStack()
        self.session = None
        self.tools = {}
        self.initialization = {}

    async def __aenter__(self):
        try:
            from mcp import ClientSession
            from mcp.types import Implementation
            if self.profile["transport"] == "stdio":
                from mcp.client.stdio import StdioServerParameters, stdio_client
                env = dict(self.profile.get("env", {}))
                for name in self.profile.get("env_from", []):
                    if name not in os.environ:
                        raise ValueError("MCP 需要环境变量：" + name)
                    env[name] = os.environ[name]
                params = StdioServerParameters(command=self.profile["command"], args=self.profile.get("args", []),
                                               env=env, cwd=self.profile.get("cwd"))
                streams = await self.stack.enter_async_context(stdio_client(params))
            else:
                import httpx
                from mcp.client.streamable_http import streamable_http_client
                headers = {}
                for header, name in self.profile.get("headers_from_env", {}).items():
                    if name not in os.environ:
                        raise ValueError("MCP HTTP 需要环境变量：" + name)
                    headers[header] = os.environ[name]
                client = await self.stack.enter_async_context(httpx.AsyncClient(headers=headers,
                    timeout=httpx.Timeout(self.profile["timeout_seconds"])))
                streams = await self.stack.enter_async_context(streamable_http_client(self.profile["url"], http_client=client))
            self.session = await self.stack.enter_async_context(ClientSession(streams[0], streams[1],
                read_timeout_seconds=timedelta(seconds=self.profile["timeout_seconds"]),
                client_info=Implementation(name="scholar-daily", version="0.1.0")))
            result = await self.session.initialize()
            self.initialization = result.model_dump(mode="json", by_alias=True, exclude_none=True)
            return self
        except ImportError as exc:
            await self.stack.aclose()
            raise RuntimeError('请在项目 Conda 环境安装 MCP：python -m pip install -e ".[mcp]"') from exc
        except BaseException:
            await self.stack.aclose()
            raise

    async def __aexit__(self, exc_type, exc, traceback):
        # Do not inject caller exceptions into the SDK's internal task groups.
        # This preserves MCPToolError and timeout identities after cleanup.
        await self.stack.aclose()
        return False

    async def list_tools(self, *, allowed_only=True):
        from mcp.types import PaginatedRequestParams
        collected, cursors, total_chars = {}, set(), 0
        cursor = None
        for _ in range(20):
            page = await self.session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
            for tool in page.tools:
                row = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
                if row["name"] in collected:
                    raise ValueError("MCP tools/list 重复了工具名称")
                size = len(json.dumps(row, ensure_ascii=False))
                total_chars += size
                if size > 100000 or total_chars > 1000000 or len(collected) >= 300:
                    raise ValueError("MCP 工具目录或 Schema 超过大小限制")
                row["allowed"] = row["name"] in self.profile.get("allowed_tools", [])
                collected[row["name"]] = row
            cursor = page.nextCursor
            if not cursor:
                break
            if cursor in cursors:
                raise ValueError("MCP tools/list 返回循环游标")
            cursors.add(cursor)
        else:
            raise ValueError("MCP 工具目录超过分页上限")
        self.tools = collected
        return [row for row in collected.values() if not allowed_only or row["allowed"]]

    async def call_tool(self, name, arguments):
        if name not in self.profile.get("allowed_tools", []):
            raise ValueError("MCP 工具不在配置的 allowed_tools 中：" + name)
        if not self.tools:
            await self.list_tools()
        if name not in self.tools:
            raise ValueError("MCP server 未暴露该工具：" + name)
        validate_json_schema(arguments, self.tools[name]["inputSchema"])
        output_schema = self.tools[name].get("outputSchema")
        if output_schema:
            schema_validator(output_schema)
        response = await self.session.call_tool(name, arguments=arguments)
        result = response.model_dump(mode="json", by_alias=True, exclude_none=True)
        if len(json.dumps(result, ensure_ascii=False)) > self.profile["max_response_chars"]:
            raise MCPToolError("MCP 返回内容超过 max_response_chars，需缩小查询范围")
        if response.isError:
            raise MCPToolError("MCP 工具执行失败：" + name, result)
        if response.structuredContent is not None and output_schema:
            validate_json_schema(response.structuredContent, output_schema)
        return result


async def client_request(profile, operation, *, tool=None, arguments=None, root=None, allowed_only=True):
    async with MCPConnection(profile, root) as client:
        tools = await client.list_tools(allowed_only=allowed_only)
        if operation == "list":
            return {"initialization": client.initialization, "tools": tools}
        if operation != "call":
            raise ValueError("未知 MCP Client 操作")
        return await client.call_tool(tool, {} if arguments is None else arguments)


def request_sync(config, server, operation, **kwargs):
    profile = server_profile(config, server)
    try:
        return asyncio.run(client_request(profile, operation, root=config.get("_root"), **kwargs))
    except (ValueError, MCPToolError):
        raise
    except Exception as exc:
        raise RuntimeError("MCP 连接或协议失败 [%s]：%s" % (server, exc)) from exc


def text_content(result):
    return "\n\n".join(block["text"] for block in result.get("content", []) if block.get("type") == "text")


def result_preview(result, max_chars=12000):
    text = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(text) <= max_chars:
        return {"result": result, "truncated": False, "characters": len(text)}
    return {"text_preview": (text_content(result) or json.dumps(result.get("structuredContent", result), ensure_ascii=False))[:max_chars], "truncated": True, "characters": len(text),
            "isError": bool(result.get("isError", False)), "full_result": "see recorded result_path"}
