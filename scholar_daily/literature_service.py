"""Bounded literature API shared by MCP tools; independent of MCP transport and models."""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .config import resolve_path
from .fulltext import fetch_fulltext
from .models import Paper
from .paper_context import build_index, load_index, locate_quote, paper_sources, save_index, select_context
from .research_calls import read_json, write_json
from .research_state import fingerprint
from .research_verifier import evidence_signature, review_inputs
from .sources import ArxivSource, deduplicate_papers


class LiteratureService:
    def __init__(self, config, *, search=None, reader=None):
        self.config = config
        self.search = search or ArxivSource().search
        self.reader = reader or fetch_fulltext
        self.folder = resolve_path(config, "cache_dir") / "mcp"
        self.output = resolve_path(config, "output_dir")

    def _registry_path(self, pid):
        return self.folder / "papers" / (fingerprint(pid)[:24] + ".json")

    def register(self, paper, context=None):
        existing = self._registry_path(paper.paper_id)
        if existing.exists() and not context:
            saved = read_json(existing)
            if saved["paper"]["updated"] == paper.updated and saved.get("context"):
                return saved
        row = {"paper": paper.as_dict(), "context": context or {}}
        write_json(self._registry_path(paper.paper_id), row)
        return row

    def _connection(self):
        path = resolve_path(self.config, "data_dir") / "research.sqlite3"
        if not path.is_file():
            return None
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _task_states(self):
        paths = sorted((self.output / "research-tasks").glob("*/state.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for path in paths[:1000]:
            if not path.resolve().is_relative_to(self.output) or path.stat().st_size > 30 * 1024 * 1024:
                continue
            try:
                yield path, read_json(path)
            except (ValueError, OSError):
                continue

    def lookup(self, pid, task_id=""):
        if task_id:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", task_id):
                raise ValueError("task_id 格式无效")
            path = self.output / "research-tasks" / task_id / "state.json"
            if not path.resolve().is_relative_to(self.output):
                raise ValueError("任务路径越界")
            state = read_json(path)
            if pid not in state["papers"]:
                raise ValueError("任务中没有此论文 ID")
            return {"paper": state["papers"][pid], "context": state.get("paper_contexts", {}).get(pid, {}), "task_dir": path.parent}
        path = self._registry_path(pid)
        if path.is_file():
            row = read_json(path)
            if row["paper"]["paper_id"] != pid:
                raise ValueError("MCP 材料注册表论文 ID 不一致")
            return row
        for path, state in self._task_states():
            if pid in state.get("papers", {}):
                return {"paper": state["papers"][pid], "context": state.get("paper_contexts", {}).get(pid, {}), "task_dir": path.parent}
        connection = self._connection()
        if connection is not None:
            try:
                row = connection.execute("SELECT * FROM papers WHERE paper_id=?", (pid,)).fetchone()
                if row:
                    paper = Paper(row["paper_id"], row["source"], row["title"], json.loads(row["authors_json"]),
                                  row["abstract"], row["published"] or "", row["updated"] or "", row["url"] or "",
                                  row["pdf_url"] or "", categories=json.loads(row["categories_json"]))
                    return {"paper": paper.as_dict(), "context": {}}
            finally:
                connection.close()
        raise ValueError("未找到论文 ID；请先 search_papers 或指定已有任务 task_id")

    @staticmethod
    def metadata(paper):
        return {k: getattr(paper, k) for k in ("paper_id", "source", "title", "authors", "abstract", "published", "updated", "url", "pdf_url")}

    def search_papers(self, query, days=30, limit=10, source="arxiv"):
        errors, rows = [], []
        if source in {"arxiv", "all"}:
            try:
                papers = deduplicate_papers(self.search(query, [], days, limit, "latest"))
                for paper in papers[:limit]:
                    self.register(paper)
                    rows.append(dict(self.metadata(paper), reading_level="metadata_only"))
            except (ValueError, OSError, RuntimeError) as exc:
                if source != "all":
                    raise
                errors.append({"source": "arxiv", "message": str(exc)})
        if source in {"zotero", "all"}:
            try:
                from .mcp_client import request_sync
                from .zotero_mcp import search_records
                server = self.config.get("mcp", {}).get("zotero_server", "zotero")
                result = request_sync(self.config, server, "call", tool="zotero_search_items",
                                      arguments={"query": query, "qmode": "everything", "limit": limit})
                rows.extend(search_records(result, server)[:limit])
            except (ValueError, OSError, RuntimeError) as exc:
                if source != "all":
                    raise
                errors.append({"source": "zotero", "message": str(exc)})
        return {"query": query, "source": source, "papers": rows[:limit], "errors": errors,
                "truncated": len(rows) > limit, "time_filter": "arxiv submission date; Zotero searches saved library without date filter"}

    def material(self, pid, task_id="", fetch=False):
        if pid.startswith("zotero:") and not task_id and not self._registry_path(pid).exists():
            parts = pid.split(":")
            if len(parts) != 3:
                raise ValueError("Zotero paper_id 应为 zotero:<server>:<item_key>")
            from .zotero_mcp import fetch_zotero_paper_sync
            paper, receipt = fetch_zotero_paper_sync(self.config, parts[1], parts[2])
            index, context = save_index(self.folder, paper, {"layout": paper.full_text}, ["layout"] if receipt.get("local_attachment") else [])
            write_json(self.folder / "zotero" / (fingerprint(pid)[:24] + ".json"), receipt)
            self.register(paper, context)
        row = self.lookup(pid, task_id)
        paper = Paper(**row["paper"])
        if row.get("context"):
            index, texts = load_index(row["context"], folder=row.get("task_dir", self.folder))
            if index["paper_id"] != pid or index["version"] != paper.updated:
                raise ValueError("原文索引与论文版本不一致")
        else:
            if not paper.full_text and not paper.full_text_path:
                cache = resolve_path(self.config, "cache_dir")
                key = hashlib.sha256((paper.paper_id + paper.updated).encode()).hexdigest()[:24]
                path = cache / "papers" / (key + ".txt")
                if path.exists():
                    paper.full_text_path = str(path)
                    pdf = path.with_suffix(".pdf")
                    paper.pdf_path = str(pdf) if pdf.exists() else ""
                    table = path.with_suffix(".table.txt")
                    paper.tables_text = table.read_text(encoding="utf-8") if table.exists() else ""
                elif fetch and self.config["sources"]["full_text"]:
                    self.reader(paper, cache, self.config["sources"]["max_text_chars"])
            texts, pages = paper_sources(paper, resolve_path(self.config, "cache_dir"))
            if any(text.strip() for text in texts.values()):
                index, context = save_index(self.folder, paper, texts, pages)
                self.register(paper, context)
            else:
                index = build_index(paper.as_dict(), texts)
        return paper, index, texts

    def read_paper(self, paper_id, query="", level="fragment", max_chars=12000, task_id="", fetch=True):
        paper, index, texts = self.material(paper_id, task_id, fetch)
        selection = select_context(index, texts, query or paper.title, level=level, max_chars=max_chars,
                                   max_tokens=max_chars, max_fragments=8, require_match=bool(query))
        text = selection.pop("text")
        for segment in selection["segments"]:
            source = index["sources"][segment["source_kind"]]
            segment["sections"] = [{"section_id": section["section_id"], "name": section["name"]}
                                    for section in source["sections"] if section["start"] < segment["end"] and section["end"] > segment["start"]]
            markers = source["page_markers"]
            positions = [row[0] for row in markers]
            def page_at(offset):
                return markers[bisect.bisect_right(positions, offset) - 1][1] if markers and offset >= markers[0][0] else None
            segment["page_start"] = page_at(segment["start"])
            segment["page_end"] = page_at(segment["end"] - 1)
        return {"paper": self.metadata(paper), "reading_level": "paper_excerpt" if text else "abstract_only",
                "text": text, "selection": selection, "source_fingerprint": fingerprint(index),
                "sections": [{"name": s["name"], "section_id": s["section_id"]}
                             for source in index["sources"].values() for s in source["sections"]],
                "offset_unit": index["offset_unit"], "source_characters": sum(len(t) for t in texts.values())}

    def verify_citation(self, paper_id, citation, claim="", task_id="", fetch=False):
        paper, index, texts = self.material(paper_id, task_id, fetch)
        locations = locate_quote(index, texts, citation)
        semantic_reviews = []
        if task_id:
            row = self.lookup(paper_id, task_id)
            state = read_json(row["task_dir"] / "state.json")
            report = state.get("verification", {}).get("evidence", {})
            current = report.get("status") == "checked" and report.get("signature") == evidence_signature(state)
            payload = review_inputs(state, {"phase": "evidence"}) if current else {"items": []}
            errors = {r["item_id"]: r["structural_error"] for r in payload["items"]}
            for item in report.get("items", []):
                if item["paper_id"] != paper_id or (claim and item["text"] != claim):
                    continue
                records = [state["evidence"].get(eid, {}) for eid in item["evidence_ids"]]
                if not any(r.get("anchor_quote") == citation for r in records):
                    continue
                valid = current and not errors.get(item["item_id"], "missing")
                semantic_reviews.append({"item_id": item["item_id"], "claim": item["text"],
                    "support_status": item.get("support_status", "supported" if item["passed"] else "insufficient") if valid else "stale",
                    "evidence_ids": item["evidence_ids"], "model_judgment": True, "human_verified": False,
                    "current": valid, "reasons": [c["reason"] for c in item["claims"]]})
        return {"paper_id": paper_id, "version": paper.updated, "citation": citation, "claim": claim,
                "citation_exists": bool(locations), "locations": locations,
                "semantic_reviews": semantic_reviews, "semantic_support": "saved_model_review" if semantic_reviews else "unreviewed",
                "matching": "whitespace only; preserves numbers, units, minus signs and hyphens"}

    def query_history(self, query="", limit=10, kind="all", paper_id=""):
        records = []
        query = query.casefold()
        connection = self._connection()
        if connection is not None:
            try:
                if kind in {"all", "runs", "reports"} and not paper_id:
                    for row in connection.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1000"):
                        record = dict(row)
                        path = Path(record.get("report_path") or "").resolve()
                        snippet = ""
                        if path.is_relative_to(self.output) and path.is_file():
                            body = path.read_text(encoding="utf-8", errors="replace")[:200000]
                            start = max(0, body.casefold().find(query) - 200) if query else 0
                            if query and query not in body.casefold() and query not in json.dumps(record, ensure_ascii=False).casefold():
                                continue
                            snippet = body[start:start + 2000]
                        elif query and query not in json.dumps(record, ensure_ascii=False).casefold():
                            continue
                        records.append({"kind": "run", "record": record, "report_excerpt": snippet})
                if kind in {"all", "ideas", "evidence"}:
                    tables = [("ideas", "idea")] if kind == "ideas" else [("analyses", "evidence")] if kind == "evidence" else [("ideas", "idea"), ("analyses", "evidence")]
                    for table, label in tables:
                        sql = "SELECT run_id,payload_json" + (",paper_id" if table == "analyses" else "") + " FROM " + table
                        values = ()
                        if paper_id and table == "analyses":
                            sql += " WHERE paper_id=?"
                            values = (paper_id,)
                        for row in connection.execute(sql + " ORDER BY run_id DESC LIMIT 1000", values):
                            payload = json.loads(row["payload_json"])
                            items = payload.get("claims", []) if label == "evidence" else [payload]
                            for item in items:
                                if paper_id and label == "idea" and paper_id not in json.dumps(item, ensure_ascii=False):
                                    continue
                                if query and query not in json.dumps(item, ensure_ascii=False).casefold():
                                    continue
                                records.append({"kind": label, "run_id": row["run_id"], "record": item,
                                                "paper_id": row["paper_id"] if table == "analyses" else "",
                                                "review_status": "historical_record_unverified"})
            finally:
                connection.close()
        scanned = 0
        for path, state in self._task_states():
            scanned += 1
            if paper_id and paper_id not in state.get("papers", {}):
                continue
            if kind in {"all", "reports", "runs"}:
                report_path = path.parent / "report.md"
                body = report_path.read_text(encoding="utf-8", errors="replace")[:200000] if report_path.is_file() else ""
                if not query or query in body.casefold() or query in state.get("goal", "").casefold():
                    start = max(0, body.casefold().find(query) - 200) if query else 0
                    records.append({"kind": "research_report", "task_id": state["task_id"], "goal": state["goal"],
                                    "status": state["status"], "report_path": str(report_path), "report_excerpt": body[start:start + 2000]})
            if kind in {"all", "evidence"}:
                for eid, item in state.get("evidence", {}).items():
                    if paper_id and item["paper_id"] != paper_id:
                        continue
                    if not query or query in json.dumps(item, ensure_ascii=False).casefold():
                        records.append({"kind": "evidence", "task_id": state["task_id"], "record": item,
                                        "review_status": "see verify_citation with task_id"})
        return {"query": query, "kind": kind, "records": records[:limit], "matches_in_scanned_history": len(records),
                "truncated": len(records) > limit, "scanned_tasks": scanned, "scan_limit": 1000,
                "history_is_current_evidence": False, "scan_limit_reached": scanned >= 1000}
