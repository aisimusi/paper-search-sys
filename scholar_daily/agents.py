from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from ._compat import terminate_process
from .contracts import FIGURE_SCHEMA, validate_shape


def harness_home(settings: dict, workspace: Path) -> Optional[Path]:
    value = (settings.get("dsh_home") or "").strip() or os.environ.get("DSH_HOME", "").strip()
    if not value:
        return None
    return (workspace / Path(value).expanduser()).resolve()


USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens", "total_tokens")


def as_count(value: Any) -> Optional[int]:
    """Return an integer token count, or ``None`` when the provider omitted it."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def usage_from_events(events: list) -> dict:
    """Sum token usage from the ``assistant/message`` events of one harness run.

    Each event carries a ``data.usage`` object with camelCase counters. Missing or
    non-numeric values count as zero so one odd event never breaks the run.
    """
    totals = {field: 0 for field in USAGE_FIELDS}
    for event in events or []:
        if not isinstance(event, dict) or event.get("type") != "assistant/message":
            continue
        data = event.get("data")
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            continue
        for field, key in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens"),
                           ("cache_read_tokens", "cacheReadTokens"), ("reasoning_tokens", "reasoningTokens"),
                           ("total_tokens", "totalTokens")):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[field] += int(value)
    return totals


def drop_unknown_fields(value: Any, schema: dict) -> Any:
    """Recursively drop object keys the schema does not declare.

    Models occasionally add explanatory fields the schema does not define (for
    example ``reading_level_note``). Those are not evidence, so ignoring them keeps
    a whole batch from being rejected over model noise, while every required field
    and the schema's value constraints are still enforced by ``validate_shape``.
    """
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if "object" in kinds and isinstance(value, dict):
        properties = schema.get("properties", {})
        return {key: drop_unknown_fields(child, properties[key]) for key, child in value.items() if key in properties}
    if "array" in kinds and isinstance(value, list) and "items" in schema:
        return [drop_unknown_fields(child, schema["items"]) for child in value]
    return value


def repair_json(text: str) -> str:
    """Best-effort repair for JSON the model truncated mid-object.

    Handles the two most common LLM formatting failures: a trailing comma before
    ``]``/``}``, and an object/array that ends without its closing bracket(s)
    (the vision model often emits ``{"figures": [...]`` without the final ``}``).
    The repair is conservative: it never rewrites content, only drops trailing
    commas and appends missing closers at the very end.
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}":
            if stack and stack[-1] == "}":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "]":
                stack.pop()
    return s + "".join(reversed(stack))


def parse_response(text: str, schema: dict) -> dict:
    clean = text.strip()
    # Strip a single surrounding code fence when the model wrapped the JSON in one.
    if clean.startswith("```"):
        clean = re.sub(r"^```[^\n]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
    # The agent may append commentary after the JSON object; decode only the first
    # balanced object and ignore any trailing text instead of failing the whole turn.
    start = clean.find("{")
    if start == -1:
        raise ValueError("响应中未找到 JSON 对象")
    candidate = clean[start:]
    decoder = json.JSONDecoder(parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Invalid JSON number: " + s)))
    try:
        value, _end = decoder.raw_decode(candidate, 0)
    except ValueError:
        value, _end = decoder.raw_decode(repair_json(candidate), 0)
    value = drop_unknown_fields(value, schema)
    validate_shape(value, schema)
    return value


class CodexBackend:
    def __init__(self, config: dict, model: Optional[str] = None) -> None:
        self.config = config
        self._model_override = (model or "").strip() or None

    def run(self, prompt: str, schema: dict, folder: Path) -> dict:
        folder.mkdir(parents=True, exist_ok=True)
        prompt_file, schema_file, output = folder / "prompt.txt", folder / "schema.json", folder / "response.json"
        prompt_file.write_text(prompt, encoding="utf-8")
        schema_file.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        command = ["codex", "-a", "never", "exec", "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                   "--cd", str(folder.resolve()), "--json", "--output-schema", str(schema_file.resolve()),
                   "--output-last-message", str(output.resolve()), "-"]
        model = self._model_override or self.config.get("model")
        if model:
            command[4:4] = ["--model", model]
        # No shell expansion, no prompts for approval, and bounded unattended execution.
        with (folder / "events.jsonl").open("w") as events, (folder / "stderr.log").open("w") as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors, text=True, start_new_session=True)
            try:
                process.communicate(prompt, timeout=self.config["timeout_seconds"])
            except subprocess.TimeoutExpired:
                terminate_process(process)
                raise RuntimeError("Codex 超时；查看 " + str(folder / "stderr.log"))
        if process.returncode or not output.exists():
            raise RuntimeError("Codex 未产生有效结果；查看 " + str(folder / "stderr.log"))
        return parse_response(output.read_text(encoding="utf-8"), schema)


class TaskPending(Exception):
    """A file task needs a response from the agent's current conversation."""

    def __init__(self, task: dict) -> None:
        self.task = task
        super().__init__(task.get("validation_error") or "等待当前 Codex 会话完成任务")


class SessionBackend:
    """Exchange file tasks with the current Codex session, without model calls."""

    resumable = True

    def reject(self, folder: Path, error: str = "") -> None:
        task = {"prompt_path": str((folder / "prompt.txt").resolve()),
                "schema_path": str((folder / "schema.json").resolve()),
                "response_path": str((folder / "response.json").resolve()),
                "validation_error": error}
        raise TaskPending(task)

    def run(self, prompt: str, schema: dict, folder: Path) -> dict:
        folder.mkdir(parents=True, exist_ok=True)
        prompt_file, schema_file, output = folder / "prompt.txt", folder / "schema.json", folder / "response.json"
        changed = prompt_file.exists() and prompt_file.read_text(encoding="utf-8") != prompt
        if changed and output.exists():
            # An edited earlier answer must not silently reuse dependent answers.
            output.replace(folder / "response.previous.json")
        prompt_file.write_text(prompt, encoding="utf-8")
        schema_file.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        if not output.exists():
            self.reject(folder, "任务输入已变化；请根据更新后的 prompt.txt 重新生成 response.json" if changed else "")
        try:
            value = json.loads(output.read_text(encoding="utf-8-sig"),
                               parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Invalid JSON number: " + s)))
            value = drop_unknown_fields(value, schema)
            validate_shape(value, schema)
        except (ValueError, UnicodeError) as exc:
            self.reject(folder, "响应校验失败：" + str(exc))
        return value

    def describe_figures(self, images: list[str], folder: Path) -> dict:
        prompt = ("在当前 Codex 会话中打开以下论文 PDF 页图，逐页记录可见图表的页码、类型、图注和客观描述。"
                  "只记录实际看见的坐标轴、趋势和数字；视觉描述作为独立来源，不能用于逐字引文。"
                  "无法读取时返回 figures=[]。按 schema.json 将结果写入 response.json。\n"
                  + json.dumps(images, ensure_ascii=False))
        return self.run(prompt, FIGURE_SCHEMA, folder)


class HarnessBackend:
    """Adapter over the official ``deepseek-harness-sdk`` (Python ≥ 3.10).

    The SDK launches a bundled ``dsh`` runtime with ``--profile sdk`` and keeps
    it alive across ``run()`` calls, so one runtime subprocess serves every
    analysis and idea batch in a pipeline. Each ``run()`` uses a fresh session
    (the SDK default) so domain batches stay isolated.
    """

    def __init__(self, config: dict, workspace: Optional[Path] = None, model: Optional[str] = None) -> None:
        self.config = config
        self.workspace = Path(workspace).resolve() if workspace else Path.cwd()
        self._model_override = (model or "").strip() or None
        self._harness = None
        self._usage: Dict[tuple, dict] = {}

    def _settings(self) -> dict:
        settings = self.config.get("harness") or {}
        return settings if isinstance(settings, dict) else {}

    def _model(self) -> str:
        return (self._model_override or (self._settings().get("model") or "")).strip()

    def _add_usage(self, role: str, model: str, counts: dict) -> None:
        record = self._usage.get((role, model))
        if record is None:
            record = {"role": role, "model": model, "calls": 0}
            record.update({field: 0 for field in USAGE_FIELDS})
            self._usage[(role, model)] = record
        record["calls"] += 1
        for field in USAGE_FIELDS:
            value = counts.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                record[field] += int(value)

    @property
    def usage(self) -> list:
        """Per-role/model token totals accumulated during this backend's lifetime."""
        return [dict(record) for record in self._usage.values()]

    def _vision_settings(self) -> dict:
        settings = self.config.get("vision") or {}
        return settings if isinstance(settings, dict) else {}

    def _ensure_harness(self):
        if self._harness is not None:
            return self._harness
        try:
            from deepseek_harness import DeepSeekHarness
        except ImportError as exc:
            raise RuntimeError(
                "请运行 scripts/scholar-daily.ps1 setup，或在 Conda Python ≥ 3.10 环境安装本项目"
            ) from exc
        settings = self._settings()
        dsh_home = harness_home(settings, self.workspace)
        if not dsh_home:
            raise ValueError("Harness 需要独立的 dsh_home（agent.harness.dsh_home 或环境变量 DSH_HOME）")
        model = self._model()
        if not model:
            raise ValueError("Harness 需要 model（该 provider 实际支持的模型 ID）")
        kwargs: Dict[str, Any] = {
            "dsh_home": str(dsh_home),
            "cwd": str(self.workspace),
            "provider": settings.get("provider") or "deepseek-official",
            "model": model,
        }
        if settings.get("profile"):
            kwargs["profile"] = settings["profile"]
        if settings.get("reasoning_effort"):
            kwargs["reasoning_effort"] = settings["reasoning_effort"]
        if settings.get("max_tokens"):
            kwargs["max_tokens"] = int(settings["max_tokens"])
        if settings.get("initialize_timeout_seconds"):
            kwargs["initialize_timeout_seconds"] = int(settings["initialize_timeout_seconds"])
        if self.config.get("timeout_seconds"):
            kwargs["request_timeout_seconds"] = int(self.config["timeout_seconds"])
        self._harness = DeepSeekHarness(**kwargs)
        return self._harness

    def run(self, prompt: str, schema: dict, folder: Path) -> dict:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        full_prompt = prompt + "\n最终响应只返回满足以下 JSON Schema 的 JSON：\n" + json.dumps(schema, ensure_ascii=False)
        (folder / "prompt.txt").write_text(full_prompt, encoding="utf-8")
        (folder / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        harness = self._ensure_harness()
        result = harness.run(full_prompt)
        events = getattr(result, "events", None) or []
        (folder / "events.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")
        self._add_usage("text", self._model(),
                        usage_from_events(getattr(result, "events", None)))
        final = result.final_response or ""
        (folder / "response.json").write_text(final, encoding="utf-8")
        if result.finish_reason != "completed":
            reason = result.finish_reason if result.finish_reason is not None else "no-turn-ended"
            raise RuntimeError(
                "Harness 未完成本次分析（finish_reason=%s）；原始输出见 %s" % (reason, folder / "response.json")
            )
        if not final.strip():
            raise RuntimeError("Harness 返回了空的最终响应；原始输出见 " + str(folder / "response.json"))
        return parse_response(final, schema)

    def describe_figures(self, images: list[str], folder: Path) -> dict:
        """Describe rendered PDF pages via a direct OpenAI-compatible vision call.

        Uses the OpenAI ``image_url`` content format against ``agent.vision.base_url``
        so it works with any vision-capable OpenAI-compatible endpoint (e.g. Silicon
        Flow), independent of the harness runtime's model adapter. The returned text
        is a separate visual source and must not be used for verbatim evidence quotes.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        vision = self._vision_settings()
        model = (vision.get("model") or "").strip()
        base_url = ((vision.get("base_url") or "").strip()
                    or os.environ.get("VISION_BASE_URL", "").strip()
                    or os.environ.get("DEEPSEEK_BASE_URL", "").strip()
                    or "https://api.deepseek.com")
        api_key = (os.environ.get("VISION_API_KEY", "").strip()
                   or os.environ.get("SILICONFLOW_API_KEY", "").strip()
                   or os.environ.get("DEEPSEEK_API_KEY", "").strip())
        if not model:
            raise ValueError("视觉模型需要 agent.vision.model（该 provider 实际支持的视觉模型 ID）")
        if not api_key:
            raise ValueError("视觉模型需要 VISION_API_KEY / SILICONFLOW_API_KEY / DEEPSEEK_API_KEY")
        instruction = (
            "这是论文 PDF 的渲染页图。逐页识别其中的图（figure）和表格（table）。"
            "对每个图/表：给出所在页码 page、类型 kind（figure/table/other）、"
            "可见标题或图注 caption，以及客观描述 description（坐标轴、趋势、关键数字）。"
            "只描述可见内容，不要评价、解读或推断。"
            "最终响应只返回满足以下 JSON Schema 的 JSON：\n" + json.dumps(FIGURE_SCHEMA, ensure_ascii=False)
        )
        content: list[dict] = [{"type": "text", "text": instruction}]
        for path in images:
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}})
        payload = {"model": model, "messages": [{"role": "user", "content": content}]}
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=int(self.config.get("timeout_seconds") or 600)) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError("视觉模型 HTTP %s：%s" % (exc.code, exc.read().decode("utf-8", errors="replace")[:500])) from exc
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
        prompt_tokens = as_count(usage.get("prompt_tokens"))
        cache_read = as_count(prompt_details.get("cached_tokens"))
        # OpenAI-compatible ``prompt_tokens`` already includes cache hits, while
        # the report's 输入 column is defined as uncached input (the same meaning
        # the harness runtime gives ``inputTokens``). Subtract to avoid double
        # counting the same tokens in two columns.
        input_tokens = None if prompt_tokens is None else max(0, prompt_tokens - (cache_read or 0))
        self._add_usage("vision", model, {
            "input_tokens": input_tokens,
            "output_tokens": as_count(usage.get("completion_tokens")),
            "cache_read_tokens": cache_read,
            "reasoning_tokens": as_count(completion_details.get("reasoning_tokens")),
            "total_tokens": as_count(usage.get("total_tokens")),
        })
        final = (body.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        (folder / "figures-response.json").write_text(final, encoding="utf-8")
        if not final.strip():
            raise RuntimeError("视觉模型返回了空的最终响应；原始输出见 " + str(folder / "figures-response.json"))
        try:
            return parse_response(final, FIGURE_SCHEMA)
        except ValueError:
            # 视觉模型偶尔返回空对象 {}（表示未识别到图/表），按无图处理。
            if re.fullmatch(r"\{\s*\}", final.strip()):
                return {"figures": []}
            raise

    def close(self) -> None:
        harness = self._harness
        self._harness = None
        if harness is not None:
            try:
                harness.close()
            except Exception:
                pass

    def __enter__(self) -> "HarnessBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def get_backend(config: Dict[str, Any], workspace: Optional[Path] = None, model: Optional[str] = None):
    name = config["backend"]
    if name == "codex-session":
        return SessionBackend()
    if name == "codex":
        return CodexBackend(config, model=model)
    if name == "harness":
        return HarnessBackend(config, workspace, model=model)
    return None
