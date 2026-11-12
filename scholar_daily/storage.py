from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .models import Paper


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                config_path TEXT,
                report_path TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                abstract TEXT NOT NULL,
                published TEXT,
                updated TEXT,
                url TEXT,
                pdf_url TEXT,
                categories_json TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_runs (
                run_id INTEGER NOT NULL,
                paper_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                track TEXT NOT NULL,
                rank INTEGER,
                selected INTEGER NOT NULL DEFAULT 0,
                score REAL,
                PRIMARY KEY (run_id, paper_id, domain, track)
            );
            CREATE TABLE IF NOT EXISTS analyses (
                run_id INTEGER NOT NULL,
                paper_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, paper_id)
            );
            CREATE TABLE IF NOT EXISTS ideas (
                run_id INTEGER NOT NULL,
                idea_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, idea_id)
            );
            CREATE TABLE IF NOT EXISTS analysis_cache (
                cache_key TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def create_run(self, config_path: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO runs(started_at, status, config_path) VALUES (?, ?, ?)",
            (utc_now(), "running", config_path),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def upsert_paper(self, paper: Paper) -> None:
        now = utc_now()
        self.connection.execute(
            """INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
              title=excluded.title, authors_json=excluded.authors_json,
              abstract=excluded.abstract, published=excluded.published,
              updated=excluded.updated, url=excluded.url, pdf_url=excluded.pdf_url,
              categories_json=excluded.categories_json, last_seen=excluded.last_seen""",
            (paper.paper_id, paper.source, paper.title, json.dumps(paper.authors),
             paper.abstract, paper.published, paper.updated, paper.url, paper.pdf_url,
             json.dumps(paper.categories), now, now),
        )

    def link_paper(self, run_id: int, paper: Paper, domain: str, rank: int, selected: bool) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO paper_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, paper.paper_id, domain, paper.track, rank, int(selected), paper.relevance_score),
        )

    def save_analysis(self, run_id: int, paper_id: str, payload: Dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO analyses VALUES (?, ?, ?)", (run_id, paper_id, json.dumps(payload, ensure_ascii=False))
        )

    def save_idea(self, run_id: int, idea_id: str, payload: Dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO ideas VALUES (?, ?, ?)", (run_id, idea_id, json.dumps(payload, ensure_ascii=False))
        )

    def cached_analysis(self, cache_key: str) -> Dict[str, Any] | None:
        row = self.connection.execute("SELECT payload_json FROM analysis_cache WHERE cache_key=?", (cache_key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_analysis(self, cache_key: str, paper_id: str, payload: Dict[str, Any]) -> None:
        self.connection.execute("INSERT OR REPLACE INTO analysis_cache VALUES (?, ?, ?)", (cache_key, paper_id, json.dumps(payload, ensure_ascii=False)))

    def recent_ideas(self, limit: int = 30) -> List[Dict[str, Any]]:
        rows = self.connection.execute("SELECT payload_json FROM ideas ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def latest_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))]

    def finish_run(self, run_id: int, status: str, report_path: str = "", error: str = "") -> None:
        self.connection.execute(
            "UPDATE runs SET finished_at=?, status=?, report_path=?, error=? WHERE id=?",
            (utc_now(), status, report_path, error, run_id),
        )
        self.connection.commit()

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
