"""Durable call receipts and conservative, cross-resume model budgets."""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from .agents import TaskPending
from .contracts import validate_shape
from .pipeline import atomic_text, cst_now
from .research_state import fingerprint
from .paper_context import TOKEN_ESTIMATOR, estimated_tokens


class BudgetExceeded(RuntimeError):
    pass


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError("非法 JSON 数字：" + s)))


def measured_usage(folder: Path) -> int | None:
    """Only returned counters count as measurements, including failed calls."""
    path = folder / "events.jsonl"
    values = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            usage = None
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
            elif event.get("type") == "assistant/message":
                usage = (event.get("data") or {}).get("usage")
            if isinstance(usage, dict):
                total = usage.get("totalTokens", usage.get("total_tokens"))
                if total is None:
                    incoming = usage.get("input_tokens", usage.get("inputTokens"))
                    outgoing = usage.get("output_tokens", usage.get("outputTokens"))
                    if type(incoming) is int and type(outgoing) is int:
                        total = incoming + outgoing + int(usage.get("cacheReadTokens", 0) or 0)
                if type(total) is int and total >= 0:
                    values.append(total)
    return sum(values) if values else None


def measured_input_usage(folder: Path) -> int | None:
    path = folder / "events.jsonl"
    values, missing = [], False
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("type") not in {"turn.completed", "assistant/message"}:
                continue
            data = event.get("data")
            usage = event.get("usage") if event["type"] == "turn.completed" else (
                data.get("usage") if isinstance(data, dict) else None)
            if not isinstance(usage, dict):
                missing = True
                continue
            # Codex includes cache hits in input_tokens. Harness reports them separately.
            value = usage.get("input_tokens")
            if value is None:
                incoming, cached = usage.get("inputTokens"), usage.get("cacheReadTokens")
                value = incoming + cached if type(incoming) is int and type(cached) is int and min(incoming, cached) >= 0 else None
            if type(value) is int and value >= 0:
                values.append(value)
            else:
                missing = True
    return sum(values) if values and not missing else None


def context_metrics(rows):
    rows = list(rows)
    roles = {}
    for row in rows:
        fields = set(row.keys())
        role = roles.setdefault(row["role"], {"attempts": 0, "estimated_input_tokens": 0,
                    "measured_input_tokens": 0, "unmeasured_attempts": 0})
        estimate = row["context_estimate"] if "context_estimate" in fields else None
        measurement = row["input_tokens"] if "input_tokens" in fields else None
        role["attempts"] += 1
        role["estimated_input_tokens"] += estimate or 0
        role["measured_input_tokens"] += measurement or 0
        role["unmeasured_attempts"] += measurement is None
    return {"attempts": len(rows), "estimated_input_tokens": sum(r["estimated_input_tokens"] for r in roles.values()),
            "measured_input_tokens": sum(r["measured_input_tokens"] for r in roles.values()),
            "unmeasured_attempts": sum(r["unmeasured_attempts"] for r in roles.values()),
            "estimate_missing_attempts": sum("context_estimate" not in row.keys() or row["context_estimate"] is None for row in rows),
            "roles": roles, "token_estimator": TOKEN_ESTIMATOR,
            "estimate_scope": "serialized prompt + response schema; excludes backend system instructions and transport",
            "measurement_scope": "provider input usage including cached input; totals only cover measured attempts"}


def call_budget(rows, limits):
    rows = list(rows)
    return {"model_calls": len(rows), "max_model_calls": limits["max_model_calls"],
            "measured_tokens": sum(r["tokens"] or 0 for r in rows),
            "unmeasured_calls": sum(r["tokens"] is None for r in rows),
            "charged_tokens": sum(r["tokens"] if r["tokens"] is not None else r["reservation"] for r in rows),
            "token_budget": limits["token_budget"]}


def saved_context_metrics(folder):
    path = folder / "calls.sqlite3"
    if not path.exists():
        return context_metrics([])
    connection = sqlite3.connect("file:" + path.as_posix() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return context_metrics(connection.execute("SELECT * FROM calls"))
    finally:
        connection.close()


def saved_budget(folder, limits):
    path = folder / "calls.sqlite3"
    if not path.exists():
        return call_budget([], limits)
    connection = sqlite3.connect("file:" + path.as_posix() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return call_budget(connection.execute("SELECT * FROM calls"), limits)
    finally:
        connection.close()


class CallLedger:
    def __init__(self, folder: Path, backend, limits: dict, reasoning_backend=None):
        self.folder, self.backend, self.limits = folder, backend, limits
        self.reasoning_backend = reasoning_backend or backend
        self.connection = sqlite3.connect(str(folder / "calls.sqlite3"))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
              name TEXT NOT NULL, attempt INTEGER NOT NULL, role TEXT NOT NULL,
              input_hash TEXT NOT NULL, status TEXT NOT NULL, reservation INTEGER NOT NULL,
              tokens INTEGER, folder TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
              PRIMARY KEY (name, attempt));
            CREATE TABLE IF NOT EXISTS operations (
              action_key TEXT PRIMARY KEY, attempts INTEGER NOT NULL, status TEXT NOT NULL,
              error TEXT NOT NULL DEFAULT '');
        """)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(calls)")}
        for name in ("context_estimate", "input_tokens"):
            if name not in columns:
                self.connection.execute("ALTER TABLE calls ADD COLUMN " + name + " INTEGER")
        self.connection.commit()

    def context_metrics(self):
        return context_metrics(self.connection.execute("SELECT * FROM calls"))

    def budget(self) -> dict:
        return call_budget(self.connection.execute("SELECT * FROM calls"), self.limits)

    def call(self, name: str, role: str, prompt: str, schema: dict, validator=None) -> dict:
        backend = self.reasoning_backend if role in {"planner", "experiment_designer", "verifier"} else self.backend
        parent = self.folder / "model-calls" / fingerprint(name)[:24]
        request_path = parent / "request.json"
        # The logical call ID binds a frozen graph input. Replay uses that input;
        # waiting-state bookkeeping must not rewrite a pending session's prompt.
        if request_path.exists():
            request = read_json(request_path)
            if request["schema"] != schema:
                raise ValueError("同一模型调用的契约已变化，请创建新任务")
            prompt = request["prompt"]
        else:
            write_json(request_path, {"name": name, "role": role, "prompt": prompt, "schema": schema})
        row = self.connection.execute("SELECT * FROM calls WHERE name=? ORDER BY attempt DESC LIMIT 1", (name,)).fetchone()
        if row:
            receipt = Path(row["folder"]) / "accepted.json"
            if receipt.exists():
                saved = read_json(receipt)
                validate_shape(saved["value"], schema)
                if validator:
                    validator(saved["value"])
                self._finish(name, row["attempt"], "completed", saved["tokens"])
                return saved["value"]
        if row and row["status"] == "waiting":
            attempt, target = row["attempt"], Path(row["folder"])
        else:
            attempt = 1 if row is None else row["attempt"] + 1
            if attempt > self.limits["max_attempts"]:
                raise BudgetExceeded("模型调用重试次数已达上限：" + role)
            budget = self.budget()
            reservation = max(self.limits["unknown_call_tokens"], len(prompt.encode("utf-8")) + 4096)
            if budget["model_calls"] >= self.limits["max_model_calls"]:
                raise BudgetExceeded("模型调用次数已达上限")
            if budget["charged_tokens"] + reservation > self.limits["token_budget"]:
                raise BudgetExceeded("剩余 Token 预算不足以预留下一次调用")
            target = parent / ("attempt-%d" % attempt)
            target.mkdir(parents=True, exist_ok=True)
            self.connection.execute("INSERT INTO calls(name,attempt,role,input_hash,status,reservation,folder) VALUES (?,?,?,?,?,?,?)",
                                    (name, attempt, role, fingerprint({"prompt": prompt, "schema": schema}),
                                     "started", reservation, str(target)))
            estimate = estimated_tokens(prompt + json.dumps(schema, ensure_ascii=False))
            self.connection.execute("UPDATE calls SET context_estimate=? WHERE name=? AND attempt=?", (estimate, name, attempt))
            self.connection.commit()  # Charge the attempt BEFORE a provider can be called.
        try:
            value = backend.run(prompt, schema, target)
            validate_shape(value, schema)
            if validator:
                validator(value)
            tokens = measured_usage(target)
            write_json(target / "accepted.json", {"value": value, "tokens": tokens, "accepted_at": cst_now(), "input_tokens": measured_input_usage(target)})
            self._finish(name, attempt, "completed", tokens)
            return value
        except TaskPending:
            self._finish(name, attempt, "waiting", None)
            raise
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            if getattr(backend, "resumable", False) and isinstance(exc, ValueError):
                self._finish(name, attempt, "waiting", None, str(exc))
                backend.reject(target, "任务结果校验失败：" + str(exc))
            self._finish(name, attempt, "failed", measured_usage(target), str(exc))
            raise

    def _finish(self, name: str, attempt: int, status: str, tokens: int | None, error: str = ""):
        row = self.connection.execute("SELECT folder,input_tokens FROM calls WHERE name=? AND attempt=?", (name, attempt)).fetchone()
        incoming = measured_input_usage(Path(row["folder"])) if row else None
        if incoming is None and row:
            incoming = row["input_tokens"]
        self.connection.execute("UPDATE calls SET status=?,tokens=?,input_tokens=?,error=? WHERE name=? AND attempt=?",
                                (status, tokens, incoming, error, name, attempt))
        self.connection.commit()

    def operation(self, key: str) -> Path:
        target = self.folder / "tool-calls" / key
        row = self.connection.execute("SELECT * FROM operations WHERE action_key=?", (key,)).fetchone()
        if row and (target / "result.json").exists():
            return target
        if not row or row["status"] != "waiting":
            attempts = 1 if row is None else row["attempts"] + 1
            if attempts > self.limits["max_attempts"]:
                raise BudgetExceeded("工具重试次数已达上限")
            self.connection.execute("INSERT INTO operations VALUES (?,?,?,?) ON CONFLICT(action_key) DO UPDATE SET attempts=excluded.attempts,status=excluded.status,error=''",
                                    (key, attempts, "started", ""))
            self.connection.commit()
        target.mkdir(parents=True, exist_ok=True)
        return target

    def finish_operation(self, key: str, status: str, error: str = ""):
        self.connection.execute("UPDATE operations SET status=?,error=? WHERE action_key=?", (status, error, key))
        self.connection.commit()

    def close(self):
        self.connection.close()
