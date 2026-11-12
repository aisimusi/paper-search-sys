import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import socket
import subprocess
import time
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scholar_daily.config import DEFAULT_CONFIG, load_config
from scholar_daily.literature_service import LiteratureService
from scholar_daily.mcp_client import (MCPConnection, MCPToolError, checked_server, client_request,
                                      function_catalog, parse_function_call, result_preview, validate_json_schema)
from scholar_daily.mcp_server import TOOL_DEFINITIONS
from scholar_daily.paper_context import GAP, build_index
from scholar_daily.research_calls import CallLedger, read_json, write_json
from scholar_daily.research_state import checked_limits, initial_state, planner_prompt
from scholar_daily.research_tools import ResearchTools
from scholar_daily.storage import Store
from scholar_daily.zotero_mcp import local_pdf_attachment, paper_from_material, search_records, zotero_text
import test_orchestrator as fixtures
from test_research import QUOTE, paper

HAS_MCP = importlib.util.find_spec("mcp") is not None
HAS_GRAPH = fixtures.HAS_GRAPH


def config(root):
    value = copy.deepcopy(DEFAULT_CONFIG)
    path = root / "config.json"
    write_json(path, value)
    return load_config(path)


def seed(root):
    cfg = config(root)
    p = paper()
    p.full_text = "1 Methods\n" + QUOTE + "\n\f2 Results\nThe latency change is -5 ms.\n"
    service = LiteratureService(cfg)
    service.register(p)
    store = Store(root / "data/research.sqlite3")
    rid = store.create_run(str(root / "config.json"))
    store.upsert_paper(p)
    report = root / "reports/report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("# Fictional report\nFixture long-context finding.", encoding="utf-8")
    store.save_idea(rid, "fixture-idea", {"title": "Fictional idea", "hypothesis": "Fixture testing"})
    store.save_analysis(rid, p.paper_id, {"claims": [{"text": "Fixture claim", "anchor_quote": QUOTE, "evidence_id": "ER-fixture"}]})
    store.commit()
    store.finish_run(rid, "complete", str(report))
    store.close()
    return cfg, p


@unittest.skipUnless(HAS_MCP, "安装 MCP extra 后运行")
class MCPClientTests(unittest.TestCase):
    def test_connection_profiles_are_explicit_and_do_not_use_shell(self):
        for value in ({"command": "python", "shell": True}, {"transport": "other"},
                      {"transport": "streamable-http", "url": "http://remote.invalid/mcp"},
                      {"transport": "streamable-http", "url": "https://user:secret@host.invalid/mcp"},
                      {"command": "python", "allowed_tools": ["a", "a"]},
                      {"command": "python", "timeout_seconds": True}):
            with self.assertRaises(ValueError):
                checked_server(value)
        self.assertEqual(checked_server({"command": "python"})["transport"], "stdio")

    def test_json_schema_preserves_integer_boolean_and_rejects_remote_refs(self):
        schema = TOOL_DEFINITIONS["search_papers"]["inputSchema"]
        validate_json_schema({"query": "fixture", "limit": 1}, schema)
        for value in ({"query": "fixture", "limit": True}, {"query": "fixture", "unknown": 1}, {"query": "fixture", "limit": 51}):
            with self.assertRaises(ValueError):
                validate_json_schema(value, schema)
        with self.assertRaisesRegex(ValueError, "本地引用"):
            validate_json_schema({}, {"$ref": "https://example.invalid/schema.json"})

    def test_function_call_mapping_keeps_original_mcp_route_and_schema(self):
        definitions = [{"name": "read-paper", "description": "Fixture", "inputSchema": {"type": "object", "properties": {"paper_id": {"type": "string"}}, "required": ["paper_id"]}},
                       {"name": "read_paper", "inputSchema": {"type": "object", "properties": {}}}]
        catalog = function_catalog("zotero-fixture", definitions)
        names = [row["function"]["name"] for row in catalog["tools"]]
        self.assertNotEqual(names[0], names[1])
        self.assertTrue(all(len(n) <= 64 for n in names))
        route = parse_function_call(catalog, names[0], '{"paper_id":"fixture"}')
        self.assertEqual(route, {"server": "zotero-fixture", "tool": "read-paper", "arguments": {"paper_id": "fixture"}})
        with self.assertRaises(ValueError):
            parse_function_call(catalog, names[0], '{}')
        with self.assertRaises(ValueError):
            parse_function_call(catalog, "invented", '{}')

    def test_client_rejects_disallowed_schema_errors_and_iserror_results(self):
        from mcp.types import CallToolResult, TextContent
        async def exercise():
            client = MCPConnection({"command": "fixture", "allowed_tools": ["read"], "max_response_chars": 1000})
            client.tools = {"read": {"inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}
            client.session = Mock()
            client.session.call_tool = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="Fixture result")]))
            with self.assertRaises(ValueError):
                await client.call_tool("write", {})
            with self.assertRaises(ValueError):
                await client.call_tool("read", {})
            self.assertFalse((await client.call_tool("read", {"id": "fixture"}))["isError"])
            client.session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="Fixture error")], isError=True)
            with self.assertRaises(MCPToolError) as failed:
                await client.call_tool("read", {"id": "fixture"})
            self.assertTrue(failed.exception.result["isError"])
            client.tools["read"]["outputSchema"] = {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}
            client.session.call_tool.return_value = CallToolResult(content=[], structuredContent={"status": 5})
            with self.assertRaises(ValueError):
                await client.call_tool("read", {"id": "fixture"})
            client.tools["read"].pop("outputSchema")
            client.session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="x" * 1100)])
            with self.assertRaises(MCPToolError):
                await client.call_tool("read", {"id": "fixture"})
        asyncio.run(exercise())

    def test_discovery_handles_pagination_and_rejects_cursor_loops(self):
        from mcp.types import ListToolsResult, Tool
        tool = Tool(name="read", inputSchema={"type": "object", "properties": {}})
        async def exercise():
            client = MCPConnection({"command": "fixture", "allowed_tools": ["read"]})
            client.session = Mock()
            client.session.list_tools = AsyncMock(side_effect=[ListToolsResult(tools=[tool], nextCursor="next"), ListToolsResult(tools=[])])
            rows = await client.list_tools()
            self.assertEqual(rows[0]["name"], "read")
            self.assertTrue(rows[0]["allowed"])
            client.session.list_tools = AsyncMock(side_effect=[ListToolsResult(tools=[], nextCursor="same"), ListToolsResult(tools=[], nextCursor="same")])
            with self.assertRaisesRegex(ValueError, "循环"):
                await client.list_tools()
        asyncio.run(exercise())


class LiteratureTests(unittest.TestCase):
    def test_search_registers_ids_and_calls_existing_source_signature(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config(Path(d))
            p = paper()
            search = Mock(return_value=[p, p])
            service = LiteratureService(cfg, search=search)
            result = service.search_papers("fixture", 7, 2)
            search.assert_called_once_with("fixture", [], 7, 2, "latest")
            self.assertEqual(len(result["papers"]), 1)
            self.assertEqual(service.lookup(p.paper_id)["paper"]["title"], p.title)

    def test_verify_uses_full_source_not_retrieved_excerpt_preserves_negation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, p = seed(root)
            service = LiteratureService(cfg)
            exists = service.verify_citation(p.paper_id, "The latency change is -5 ms.")
            self.assertTrue(exists["citation_exists"])
            self.assertEqual(exists["semantic_support"], "unreviewed")
            self.assertEqual(exists["locations"][0]["start"], p.full_text.index("The latency"))
            self.assertFalse(service.verify_citation(p.paper_id, "The latency change is 5 ms.")["citation_exists"])
            self.assertFalse(service.verify_citation(p.paper_id, QUOTE + GAP + "The latency change is -5 ms.")["citation_exists"])

    def test_read_paper_returns_bounded_traceable_sections(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, p = seed(Path(d))
            result = LiteratureService(cfg).read_paper(p.paper_id, "latency", max_chars=500)
            self.assertLessEqual(len(result["text"]), 500)
            self.assertIn("latency", result["text"])
            self.assertTrue(result["selection"]["segments"][0]["sections"])
            self.assertIsNone(result["selection"]["segments"][0]["page_start"])

    def test_partial_fragment_page_end_matches_returned_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config(Path(d))
            p = paper()
            p.full_text = "1 Methods\n" + "Method detail on first page. " * 25 + "\fSecond page detail."
            from scholar_daily.paper_context import save_index
            service = LiteratureService(cfg)
            index, context = save_index(service.folder, p, {"layout": p.full_text}, ["layout"])
            service.register(p, context)
            result = service.read_paper(p.paper_id, max_chars=500)
            segment = result["selection"]["segments"][0]
            self.assertLess(segment["end"], p.full_text.index("\f"))
            self.assertEqual((segment["page_start"], segment["page_end"]), (1, 1))

    def test_history_searches_report_contents_and_evidence_without_creating_store(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = config(root)
            service = LiteratureService(cfg)
            self.assertEqual(service.query_history()["records"], [])
            self.assertFalse((root / "data/research.sqlite3").exists())
            cfg, p = seed(root)
            self.assertEqual(len(service.query_history("long-context", kind="reports")["records"]), 1)
            evidence = service.query_history("ER-fixture", kind="evidence", paper_id=p.paper_id)
            self.assertEqual(evidence["records"][0]["record"]["evidence_id"], "ER-fixture")
            self.assertFalse(evidence["history_is_current_evidence"])

    def test_missing_quote_check_does_not_freeze_empty_body_before_fetch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = config(root)
            p = paper()
            body = p.full_text
            p.full_text = ""
            reader = Mock(side_effect=lambda value, *_: setattr(value, "full_text", body))
            service = LiteratureService(cfg, reader=reader)
            service.register(p)
            self.assertFalse(service.verify_citation(p.paper_id, QUOTE)["citation_exists"])
            self.assertIn(QUOTE, service.read_paper(p.paper_id)["text"])
            reader.assert_called_once()

    def test_paper_and_task_paths_cannot_escape_configured_roots(self):
        with tempfile.TemporaryDirectory() as d:
            service = LiteratureService(config(Path(d)))
            with self.assertRaises(ValueError):
                service.lookup("fixture", "../../private")
            with self.assertRaises(ValueError):
                service.lookup("../../private")


class ZoteroAdapterTests(unittest.TestCase):
    def metadata(self):
        return "# Fictional Zotero Paper\n\n**Item Key:** ABCD1234\n\n**Authors:** Fixture Author\n\n**Date:** 2020\n\n## Abstract\n\nThis abstract must never become paper body evidence."

    def test_metadata_and_attachment_body_are_separated(self):
        body = "The fixture attachment reports a controlled test."
        fulltext = self.metadata() + "\n\n---\n\n## Full Text\n\n" + body
        p = paper_from_material("zotero", "ABCD1234", self.metadata(), fulltext)
        self.assertEqual(p.full_text, body)
        self.assertNotIn(p.abstract, p.full_text)
        self.assertEqual(p.paper_id, "zotero:zotero:ABCD1234")
        self.assertTrue(p.updated.startswith("mcp-snapshot:"))
        absent = paper_from_material("zotero", "ABCD1234", self.metadata(), self.metadata() + "\nNo suitable attachment found for this item.")
        self.assertEqual(absent.full_text, "")
        with self.assertRaisesRegex(ValueError, "不一致"):
            paper_from_material("zotero", "WXYZ5678", self.metadata(), fulltext)

    def test_local_pdf_fallback_records_hash_and_normalizes_windows_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = config(root)
            storage = root / "zotero-storage"
            cfg["mcp"]["zotero_storage_root"] = str(storage)
            attachment = storage / "ABCD1234/fixture.pdf"
            attachment.parent.mkdir(parents=True)
            attachment.write_bytes(b"%PDF-Fictional fixture")
            children = "# Child Items\n## Attachments\n1. **Fixture**\n   - Key: ABCD1234\n   - Type: application/pdf\n   - Filename: fixture.pdf\n"
            decoded = (("Fictional result with a fixed protocol.\r\n" * 10) + "\fSecond page.\r\n").encode("utf-8")
            with patch("scholar_daily.zotero_mcp.shutil.which", return_value="pdftotext"), patch(
                    "scholar_daily.zotero_mcp.subprocess.run", return_value=Mock(stdout=decoded)) as run:
                text, source = local_pdf_attachment(cfg, children)
            self.assertNotIn("\r", text)
            self.assertIn("\f", text)
            self.assertEqual(source["attachment_key"], "ABCD1234")
            self.assertEqual(len(source["sha256"]), 64)
            run.assert_called_once()
            with self.assertRaisesRegex(ValueError, "文件名"):
                local_pdf_attachment(cfg, children.replace("fixture.pdf", "../../private.pdf"))
            cfg["mcp"].pop("zotero_storage_root")
            self.assertIsNone(local_pdf_attachment(cfg, children))

    def test_zotero_legacy_text_error_and_search_keys(self):
        with self.assertRaises(MCPToolError):
            zotero_text({"content": [{"type": "text", "text": "Error searching Zotero: fixture"}]})
        results = {"content": [{"type": "text", "text": "# Search\n\n## 1. Fictional title\n**Item Key:** ABCD1234\n"}]}
        self.assertEqual(search_records(results, "zotero")[0]["paper_id"], "zotero:zotero:ABCD1234")


@unittest.skipUnless(HAS_MCP, "安装 MCP extra 后运行")
class MCPProtocolTests(unittest.TestCase):
    def test_server_entrypoint_supplies_conda_poppler_path(self):
        from scholar_daily import mcp_server
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary = root / "Library/bin/pdftotext.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fictional-binary")
            with patch.object(mcp_server.sys, "executable", str(root / "python.exe")), patch.dict(mcp_server.os.environ, {"PATH": "fixture-path"}), \
                 patch.object(mcp_server, "create_server", return_value=Mock()), patch.object(mcp_server, "serve_http") as serve:
                mcp_server.main(["--config", "fixture.json", "--transport", "streamable-http"])
                self.assertTrue(mcp_server.os.environ["PATH"].startswith(str(binary.parent)))
                serve.assert_called_once()

    def test_official_stdio_client_initializes_discovers_and_calls_server(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, p = seed(root)
            profile = {"command": sys.executable, "args": ["-X", "utf8", "-m", "scholar_daily.mcp_server", "--config", str(root / "config.json")],
                       "cwd": str(Path(__file__).resolve().parents[1]), "allowed_tools": list(TOOL_DEFINITIONS), "timeout_seconds": 15}
            async def exercise():
                async with MCPConnection(profile) as client:
                    tools = await client.list_tools()
                    self.assertEqual({t["name"] for t in tools}, set(TOOL_DEFINITIONS))
                    self.assertIn("protocolVersion", client.initialization)
                    self.assertTrue(all("inputSchema" in t and "outputSchema" in t for t in tools))
                    result = await client.call_tool("verify_citation", {"paper_id": p.paper_id, "citation": QUOTE})
                    self.assertTrue(result["structuredContent"]["citation_exists"])
                    self.assertFalse(result["isError"])
                    self.assertEqual(json.loads(result["content"][0]["text"]), result["structuredContent"])
                    read = await client.call_tool("read_paper", {"paper_id": p.paper_id, "query": "latency", "max_chars": 500})
                    self.assertIn("latency", read["structuredContent"]["text"])
                    history = await client.call_tool("query_history", {"query": "long-context", "kind": "reports"})
                    self.assertTrue(history["structuredContent"]["records"])
                    # A valid call for a missing paper yields a tool execution error, not fabricated material.
                    with self.assertRaises(MCPToolError):
                        await client.call_tool("read_paper", {"paper_id": "unknown"})
            asyncio.run(exercise())


    def test_client_timeout_closes_unresponsive_stdio_process(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "unresponsive.py"
            target.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
            profile = {"command": sys.executable, "args": [str(target)], "timeout_seconds": 1}
            start = time.monotonic()
            with self.assertRaises(Exception) as failure:
                asyncio.run(client_request(profile, "list"))
            self.assertIn("timed out", str(failure.exception).casefold())
            self.assertLess(time.monotonic() - start, 10)

    def test_streamable_http_client_discovers_and_calls_local_server(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, p = seed(root)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [sys.executable, "-X", "utf8", "-m", "scholar_daily.mcp_server", "--config", str(root / "config.json"),
                       "--transport", "streamable-http", "--port", str(port)]
            with (root / "server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log, cwd=Path(__file__).resolve().parents[1])
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            self.fail("HTTP MCP server exited: " + (root / "server.log").read_text(encoding="utf-8"))
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                                break
                        except OSError:
                            time.sleep(.05)
                    else:
                        self.fail("HTTP MCP server startup timeout")
                    profile = {"transport": "streamable-http", "url": "http://127.0.0.1:%d/mcp" % port,
                               "allowed_tools": ["verify_citation"], "timeout_seconds": 5}
                    result = asyncio.run(client_request(profile, "call", tool="verify_citation",
                                                        arguments={"paper_id": p.paper_id, "citation": QUOTE}))
                    self.assertTrue(result["structuredContent"]["citation_exists"])
                finally:
                    process.terminate()
                    process.wait(timeout=10)

@unittest.skipUnless(HAS_MCP and HAS_GRAPH, "需安装 MCP 和 orchestrator extra")
class MCPResearchTests(unittest.TestCase):
    def state(self, root):
        cfg = config(root)
        cfg["mcp"]["servers"]["fixture"] = {"command": "fixture", "allowed_tools": ["read"]}
        return initial_state("fixture-mcp", "使用外部文献工具", cfg, {}, checked_limits())

    def test_default_research_search_adapter_passes_arxiv_track(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = self.state(root)
            tools = ResearchTools(None)
            tools.source.search = Mock(return_value=[paper()])
            result = tools.search_papers(state, fixtures.action("search_papers", query="fixture"), root, "fixture")
            tools.source.search.assert_called_once_with("fixture", [], 7, 40, "latest")
            self.assertTrue(result["papers"])

    def test_planner_receives_schema_and_bounded_untrusted_result_and_resume_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = self.state(root)
            ledger = CallLedger(root, Mock(), checked_limits())
            tools = ResearchTools(ledger)
            discovery = {"initialization": {"protocolVersion": "fixture"}, "tools": [{"name": "read", "inputSchema": {"type": "object", "properties": {}}, "allowed": True}]}
            call = fixtures.action("mcp_call_tool", query=json.dumps({"server": "fixture", "tool": "read", "arguments": {}}))
            with self.assertRaisesRegex(ValueError, "先"):
                tools.guard_mcp(state, call)
            with patch("scholar_daily.mcp_client.request_sync", return_value=discovery) as request:
                state.update(tools.execute(state, fixtures.action("mcp_list_tools", query="fixture")))
                self.assertEqual(request.call_count, 1)
            response = {"content": [{"type": "text", "text": "External fixture text"}], "isError": False}
            with patch("scholar_daily.mcp_client.request_sync", return_value=response) as request:
                patch_result = tools.execute(state, call)
                again = tools.execute(state, call)
                self.assertEqual(patch_result, again)
                self.assertEqual(request.call_count, 1)
            state.update(patch_result)
            view = json.loads(planner_prompt(state, []).split("状态：\n")[1])
            self.assertIn("fixture", view["external_tools"])
            record = next(iter(view["external_results"].values()))
            self.assertTrue(record["external_data"])
            self.assertIn("External fixture text", json.dumps(record))
            self.assertEqual(state["evidence"], {})
            ledger.close()

    def test_saved_semantic_review_is_separate_from_text_and_expires_with_facts(self):
        with tempfile.TemporaryDirectory() as d:
            from scholar_daily.orchestrator import run_research
            from scholar_daily.research_verifier import Verifier
            root = Path(d)
            cfg = config(root)
            p = paper()
            backend = fixtures.ScriptedBackend([fixtures.planned("analyze_papers", ["analysis"]),
                                               fixtures.planned("finish", ["analysis"], finish_status="complete")], [p])
            result = run_research(cfg, "核对虚构材料", inputs={"papers": [p.as_dict()]}, backend=backend)
            service = LiteratureService(cfg)
            checked = service.verify_citation(p.paper_id, QUOTE, task_id=result["task_id"])
            self.assertTrue(checked["citation_exists"])
            self.assertEqual(checked["semantic_support"], "saved_model_review")
            self.assertTrue(all(row["current"] and row["model_judgment"] for row in checked["semantic_reviews"]))
            state_path = Path(result["state_path"])
            state = read_json(state_path)
            state["facts"]["changed_protocol"] = "fixture changed"
            write_json(state_path, state)
            stale = service.verify_citation(p.paper_id, QUOTE, task_id=result["task_id"])
            self.assertTrue(stale["citation_exists"])
            self.assertTrue(all(row["support_status"] == "stale" for row in stale["semantic_reviews"]))

    def test_graph_routes_discovery_and_call_and_restores_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            from scholar_daily.orchestrator import run_research, task_status
            root = Path(d)
            cfg = config(root)
            cfg["mcp"]["servers"]["fixture"] = {"command": "fixture", "allowed_tools": ["read"]}
            backend = fixtures.ScriptedBackend([
                fixtures.planned("mcp_list_tools", ["discovery"], query="fixture"),
                fixtures.planned("mcp_call_tool", ["discovery"], query=json.dumps({"server": "fixture", "tool": "read", "arguments": {}})),
                fixtures.planned("finish", ["discovery"], finish_status="complete")])
            discovery = {"initialization": {"protocolVersion": "fixture"}, "tools": [{"name": "read", "allowed": True, "inputSchema": {"type": "object", "properties": {}}}]}
            result = {"content": [{"type": "text", "text": "Unreviewed external fixture"}], "isError": False}
            with patch("scholar_daily.mcp_client.request_sync", side_effect=[discovery, result]) as client:
                output = run_research(cfg, "通过 MCP 查询虚构材料", backend=backend)
                self.assertEqual(client.call_count, 2)
            state = task_status(Path(output["task_dir"]))
            self.assertEqual([s["status"] for s in state["steps"][:2]], ["completed", "completed"])
            self.assertTrue(state["external_results"])
            self.assertEqual(state["status"], "complete")
            with patch("scholar_daily.mcp_client.request_sync") as client:
                run_research(folder=Path(output["task_dir"]), backend=backend)
                client.assert_not_called()

    def test_zotero_import_freezes_full_source_and_invalidates_no_existing_papers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = self.state(root)
            state["config"]["mcp"]["servers"]["fixture"]["allowed_tools"] = ["zotero_get_item_metadata", "zotero_get_item_fulltext"]
            meta = ZoteroAdapterTests().metadata()
            p = paper_from_material("fixture", "ABCD1234", meta, meta + "\n## Full Text\n" + QUOTE)
            ledger = CallLedger(root, Mock(), checked_limits())
            tools = ResearchTools(ledger)
            action = fixtures.action("mcp_import_paper", query=json.dumps({"server": "fixture", "item_key": "ABCD1234"}))
            with patch("scholar_daily.zotero_mcp.fetch_zotero_paper_sync", return_value=(p, {"fixture": True})) as fetch:
                patch_result = tools.execute(state, action)
                self.assertEqual(fetch.call_count, 1)
            self.assertEqual(patch_result["papers"][p.paper_id]["full_text"], QUOTE)
            self.assertTrue(Path(patch_result["paper_contexts"][p.paper_id]["index_path"]).exists())
            self.assertEqual(state["evidence"], {})
            state.update(patch_result)
            with self.assertRaisesRegex(ValueError, "已导入"):
                tools.mcp_import_paper(state, action, root, "other")
            ledger.close()


if __name__ == "__main__":
    unittest.main()
