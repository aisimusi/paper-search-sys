"""Zotero MCP adapter: preserve metadata/body separation and source snapshots."""
from __future__ import annotations

import asyncio
import re
import hashlib
import shutil
import subprocess
from pathlib import Path

from .mcp_client import MCPConnection, MCPToolError, server_profile, text_content
from .models import Paper
from .paper_context import digest


def zotero_text(result):
    text = text_content(result)
    if not text.strip() or re.match(r"\s*(?:Error\b|Failed\b|No item found)", text, re.I):
        raise MCPToolError("Zotero 返回错误或空材料", result)
    return text


def search_records(result, server):
    text = zotero_text(result)
    rows = []
    for match in re.finditer(r"(?m)^## \d+\. (.+)\n([\s\S]*?)(?=^## \d+\. |\Z)", text):
        key = re.search(r"\*\*Item Key:\*\*\s*`?([A-Za-z0-9]{8})", match.group(2))
        if key:
            rows.append({"paper_id": "zotero:%s:%s" % (server, key.group(1)), "title": match.group(1).strip(),
                         "source": "zotero", "item_key": key.group(1), "mcp_server": server,
                         "url": "zotero://select/library/items/" + key.group(1), "reading_level": "metadata_only"})
    return rows


def paper_from_material(server, item_key, metadata, fulltext):
    if not re.fullmatch(r"[A-Za-z0-9]{8}", item_key):
        raise ValueError("Zotero item_key 应为 8 位字母数字标识")
    title = re.search(r"(?m)^# (.+)$", metadata)
    if not title:
        raise ValueError("Zotero 元数据未提供标题，需核对 server 输出格式")
    key = re.search(r"\*\*Item Key:\*\*\s*`?([A-Za-z0-9]{8})", metadata)
    if not key or key.group(1) != item_key:
        raise ValueError("Zotero 元数据的 item_key 与请求不一致")
    def field(name):
        m = re.search(r"(?m)^\*\*" + re.escape(name) + r":\*\*\s*(.+)$", metadata)
        return m.group(1).strip() if m else ""
    abstract = re.search(r"(?m)^## Abstract\s*\n([\s\S]*?)(?=\n\*\*|\n## |\Z)", metadata)
    marker = re.search(r"(?m)^## Full Text\s*\n", fulltext)
    body = fulltext[marker.end():].strip().replace("\r\n", "\n").replace("\r", "\n") if marker else ""
    # Metadata and errors preceding the body are never accepted as paper evidence.
    if marker and (not body or re.match(r"(?:Error\b|No suitable attachment|File download failed)", body, re.I)):
        body = ""
    pid = "zotero:%s:%s" % (server, item_key)
    version = "mcp-snapshot:" + digest(metadata + "\n" + fulltext)
    paper = Paper(pid, "zotero", title.group(1).strip(), field("Authors").split("; ") if field("Authors") else [],
                  abstract.group(1).strip() if abstract else "", field("Date"), version,
                  field("URL") or "zotero://select/library/items/" + item_key, "", full_text=body,
                  hotness_basis="Zotero MCP 材料快照；updated 是快照指纹，页码尚未核实")
    return paper


def local_pdf_attachment(config, children_text):
    root_setting = config.get("mcp", {}).get("zotero_storage_root")
    if not root_setting:
        return None
    root = (Path(config["_root"]) / Path(root_setting).expanduser()).resolve()
    for block in re.split(r"(?m)^\d+\. ", children_text):
        if not re.search(r"Type:\s*application/pdf", block):
            continue
        key = re.search(r"Key:\s*([A-Za-z0-9]{8})\b", block)
        filename = re.search(r"(?m)^\s*- Filename:\s*(.+)$", block)
        if not key or not filename:
            continue
        filename = filename.group(1).strip()
        if Path(filename).name != filename or "/" in filename or "\\" in filename or Path(filename).suffix.casefold() != ".pdf":
            raise ValueError("Zotero PDF 附件文件名无效")
        path = (root / key.group(1) / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Zotero PDF 附件超出配置的 storage 根目录")
        if not path.is_file():
            continue
        if path.stat().st_size > 30 * 1024 * 1024:
            raise ValueError("Zotero PDF 超过 30 MiB")
        executable = shutil.which("pdftotext")
        bundled = Path(config["_root"]) / ".conda" / "Library" / "bin" / "pdftotext.exe"
        if not executable and bundled.is_file():
            executable = str(bundled)
        if not executable:
            raise ValueError("本地 PDF 补读需要 Conda Poppler 的 pdftotext")
        pdf_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        result = subprocess.run([executable, "-layout", str(path), "-"], capture_output=True, timeout=45, check=True)
        text = result.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        if len(text.strip()) < 200 or len(text) > 2000000:
            raise ValueError("本地 PDF 可提取文本不足或超过上限")
        if hashlib.sha256(path.read_bytes()).hexdigest() != pdf_hash:
            raise ValueError("读取过程中 Zotero PDF 已变化")
        return text, {"path": str(path), "sha256": pdf_hash, "attachment_key": key.group(1),
                      "extractor": "pdftotext -layout", "page_basis": "physical_pdf_pages"}
    return None

async def fetch_zotero_paper(config, server, item_key):
    profile = server_profile(config, server)
    async with MCPConnection(profile, config.get("_root")) as client:
        await client.list_tools()
        metadata = await client.call_tool("zotero_get_item_metadata", {"item_key": item_key, "include_abstract": True})
        fulltext = await client.call_tool("zotero_get_item_fulltext", {"item_key": item_key})
        meta_text, full_text = zotero_text(metadata), zotero_text(fulltext)
        paper = paper_from_material(server, item_key, meta_text, full_text)
        raw = {"server": server, "item_key": item_key, "metadata": metadata, "fulltext": fulltext}
        # zotero-mcp 0.1.3 cannot follow a local file:// redirect in this installation.
        # Child metadata supplies the attachment key/name; filesystem reads stay in an explicit storage root.
        if not paper.full_text and "file://" in full_text and config.get("mcp", {}).get("zotero_storage_root"):
            children = await client.call_tool("zotero_get_item_children", {"item_key": item_key})
            local = await asyncio.to_thread(local_pdf_attachment, config, zotero_text(children))
            raw["children"] = children
            if local:
                paper.full_text, raw["local_attachment"] = local
                paper.pdf_path = raw["local_attachment"]["path"]
                paper.updated = "mcp-snapshot:" + digest(meta_text + paper.full_text + raw["local_attachment"]["sha256"])
    return paper, raw


def fetch_zotero_paper_sync(config, server, item_key):
    try:
        return asyncio.run(fetch_zotero_paper(config, server, item_key))
    except (ValueError, MCPToolError):
        raise
    except Exception as exc:
        raise RuntimeError("Zotero MCP 连接失败：" + str(exc)) from exc
