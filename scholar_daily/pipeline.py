from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from ._compat import FileLock
from .agents import TaskPending, get_backend
from .config import domain_queries, resolve_path, validate_config
from .contracts import ANALYSIS_SCHEMA, IDEA_SCHEMA, validate_analysis, validate_idea
from .daily_reading import review_daily_reading
from .fulltext import fetch_fulltext, make_excerpt, render_pages
from .paper_context import paper_sources
from .models import DomainResult, Paper
from .prompts import analysis_prompt, idea_prompt
from .ranking import rank_papers, relevance
from .report import domain_evidence, render_evidence, render_report, render_run_details
from .paper_explanations import content_gaps, empty_explanation
from .sources import ArxivSource, HuggingFaceSource, deduplicate_papers
from .storage import Store, utc_now

LOG = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))


def cst_now() -> str:
    """项目时区（Asia/Shanghai）的当前时间，用于报告与运行记录的时间戳。"""
    return datetime.now(CST).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def format_figure_context(figures: list[dict]) -> str:
    lines = []
    for item in figures:
        page = item.get("page")
        kind = item.get("kind", "other")
        caption = (item.get("caption") or "").strip()
        description = (item.get("description") or "").strip()
        if not description and not caption:
            continue
        header = "第%s页[%s]" % (page, kind)
        if caption:
            header += " 标题：" + caption
        lines.append(header + "｜" + description)
    return "\n".join(lines)


def analysis_key(paper: Paper, profile: dict, agent: dict) -> str:
    payload = {"contract": "v1", "reading_prompt": "complete-explanation-v3", "paper": paper.paper_id, "version": paper.updated,
               "text": paper.full_text, "abstract": paper.abstract, "profile": profile,
               "figure_context": paper.figure_context,
               "agent": {key: agent.get(key) for key in ["backend", "model", "harness", "vision"]}}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_report_bundle(folder, config, domains, started, mode, errors, usage=None, prefix=""):
    records = domain_evidence(domains)
    atomic_text(folder / "evidence.md", render_evidence(records,
        reviewed=all(d.reading_review.get("status") == "checked" for d in domains if d.analyses)))
    details = render_run_details(config, domains, started, mode, errors, usage)
    for domain in domains:
        if domain.reading_review:
            details += "\n\n## %s：语义核查\n\n状态：%s\n\n完整审查见本次目录的 domain-*-reading/reading-review.json。\n" % (
                domain.name, domain.reading_review["status"])
    atomic_text(folder / "run-details.md", details)
    explanations = {d.name: d.paper_explanations for d in domains}
    atomic_text(folder / "paper-explanations.json", json.dumps({"schema_version": 1, "domains": explanations,
        "content_checks": {d.name: {p.paper_id: content_gaps(d.paper_explanations.get(p.paper_id, empty_explanation(p.paper_id)))
                                   for p in d.top10} for d in domains}}, ensure_ascii=False, indent=2))
    atomic_text(folder / "report.md", prefix + render_report(config, domains, started, mode, errors, usage))


def select_top3(result: DomainResult) -> None:
    by_id = {item["paper_id"]: item for item in result.analyses}
    def key(paper: Paper) -> tuple:
        score = by_id.get(paper.paper_id, {}).get("quality_score")
        return (score is not None, score if score is not None else paper.relevance_score)
    result.top3 = sorted(result.top10, key=key, reverse=True)[:3]


def run_pipeline(config: dict, collect_only: bool = False, demo: bool = False, backend=None, collector: Optional[Callable] = None,
                 session_folder: Optional[Path] = None) -> dict:
    checkpoint = None
    if session_folder is not None:
        session_folder = Path(session_folder).resolve()
        checkpoint = json.loads((session_folder / "session.json").read_text(encoding="utf-8"))
        try:
            if checkpoint.get("version") != 1 or checkpoint["config"]["agent"]["backend"] != "codex-session":
                raise ValueError("无效的 Codex 会话检查点")
            config = checkpoint["config"]
            validate_config(config)
            required = {"run_id": int, "started_at": str, "collections": dict, "prepared_papers": list,
                        "histories": dict, "status": str}
            if any(type(checkpoint[key]) is not kind for key, kind in required.items()):
                raise ValueError("会话检查点字段类型不符")
            if not all(isinstance(config[key], str) for key in ("_root", "_config_path")):
                raise ValueError("会话检查点缺少配置路径")
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("会话检查点不完整或格式错误") from exc
    data = resolve_path(config, "data_dir")
    output = resolve_path(config, "output_dir")
    cache = resolve_path(config, "cache_dir")
    for folder in [data, output, cache]:
        folder.mkdir(parents=True, exist_ok=True)
    lock = FileLock(data / "daily.lock").acquire()
    store = Store(data / "research.sqlite3")
    # Holding this lock proves that runs left 'running' belong to interrupted processes.
    store.connection.execute("UPDATE runs SET status='interrupted', finished_at=?, error='Previous process exited before completion' WHERE status='running'", (utc_now(),))
    store.commit()
    if checkpoint:
        run_id, started, folder = checkpoint["run_id"], checkpoint["started_at"], session_folder
        row = store.connection.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            store.close()
            lock.release()
            raise ValueError("检查点对应的运行记录不存在")
        if checkpoint.get("status") in {"complete", "partial"}:
            store.close()
            lock.release()
            payload = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            payload.update(report_path=str(folder / "report.md"), json_path=str(folder / "report.json"),
                           session_path=str(folder / "session.json"), evidence_path=str(folder / "evidence.md"),
                           run_details_path=str(folder / "run-details.md"), explanations_path=str(folder / "paper-explanations.json"))
            return payload
    else:
        run_id = store.create_run(config["_config_path"])
        started = cst_now()
        stamp = datetime.now(CST).strftime("%Y-%m-%d-%H%M%S") + "-%d" % run_id
        folder = output / stamp
        folder.mkdir()
    mode = "demo" if demo else "collect" if collect_only else config["agent"]["backend"]
    domains, errors = [], []
    sources = config["sources"]
    profile = config["research_profile"]
    backend = backend if backend is not None else (None if collect_only or demo else get_backend(config["agent"], Path(config["_root"])))
    agent_cfg = config["agent"]
    reasoning_model = ((agent_cfg.get("harness") or {}).get("reasoning_model") or "").strip()
    default_model = ((agent_cfg.get("harness") or {}).get("model") or "").strip()
    # 论文阅读/分析用 harness.model（flash）；idea 等需要更强推理的环节用 reasoning_model（pro）。
    idea_backend = backend
    if backend is not None and agent_cfg["backend"] == "harness" and reasoning_model and reasoning_model != default_model:
        idea_backend = get_backend(agent_cfg, Path(config["_root"]), model=reasoning_model)
    resumable = bool(getattr(backend, "resumable", False))
    if resumable and checkpoint is None:
        checkpoint = {"version": 1, "config": config, "run_id": run_id, "started_at": started,
                      "status": "running", "collections": {}, "prepared_papers": [], "histories": {}}

    def save_checkpoint() -> None:
        if resumable:
            atomic_text(folder / "session.json", json.dumps(checkpoint, ensure_ascii=False, indent=2))

    def snapshot_domain(index: int, result: DomainResult) -> None:
        if resumable:
            checkpoint["collections"][str(index)] = {"papers": [asdict(p) for p in result.papers],
                                                    "top10_ids": [p.paper_id for p in result.top10],
                                                    "errors": list(result.errors)}
            save_checkpoint()

    atomic_text(folder / "profile.json", json.dumps(profile, ensure_ascii=False, indent=2))
    try:
        if resumable and session_folder is not None:
            # Replay rebuilds this run's validated records from its response files.
            # Keep historical runs untouched, including any interleaved sessions.
            for table in ("analyses", "ideas"):
                store.connection.execute("DELETE FROM " + table + " WHERE run_id=?", (run_id,))
        save_checkpoint()
        hot_papers = [Paper(**p) for p in checkpoint.get("hot_papers", [])] if resumable else []
        if not demo and collector is None and sources["huggingface"] and not (resumable and "hot_papers" in checkpoint):
            try:
                hot_papers = HuggingFaceSource().search(max(d["hot_days"] for d in profile["domains"]))
            except Exception as exc:
                errors.append("Hugging Face 热点抓取失败：%s" % exc)
        if resumable:
            if "hot_papers" in checkpoint:
                errors = list(checkpoint["errors"])
            else:
                checkpoint.update(hot_papers=[asdict(p) for p in hot_papers], errors=list(errors))
                save_checkpoint()
        arxiv = ArxivSource()
        for domain_index, domain in enumerate(profile["domains"]):
            LOG.info("研究方向：%s", domain["name"])
            result = DomainResult(domain["name"], domain_queries(domain), [], [], [])
            if backend is not None:
                result.reading_review = {"status": "pending"}
            domains.append(result)
            saved = checkpoint["collections"].get(str(domain_index)) if resumable else None
            if saved is not None:
                result.papers = [Paper(**p) for p in saved["papers"]]
                result.errors = list(saved["errors"])
            elif collector is not None:
                result.papers = collector(domain)
            elif demo:
                now = utc_now()
                result.papers = [Paper("demo-%d" % n, "demo", "[虚构演示] %s 样例 %d" % (domain["name"], n), ["Demo"],
                                       "This is a fictional fixture used only to exercise collection and reporting, not a scientific finding. " + " ".join(result.queries),
                                       now, now, "https://example.invalid/demo-%d" % n, "", hotness_basis="虚构测试数据，无真实热度") for n in range(1, 5)]
            else:
                if sources["arxiv"]:
                    for query in result.queries:
                        try:
                            LOG.info("arXiv 检索：%s", query)
                            result.papers.extend(arxiv.search(query, domain.get("arxiv_categories", []), domain["recent_days"], sources["max_results_per_query"], "latest"))
                        except Exception as exc:
                            result.errors.append("arXiv 检索失败 (%s)：%s" % (query, exc))
                cutoff = datetime.now(timezone.utc) - timedelta(days=domain["hot_days"])
                from .sources import _parse_time
                # Copy: the same paper may match multiple research domains.
                result.papers.extend(Paper(**asdict(p)) for p in hot_papers if _parse_time(p.published) >= cutoff and relevance(p, result.queries) >= .35)
            result.papers = deduplicate_papers(result.papers)
            if saved is not None:
                by_id = {p.paper_id: p for p in result.papers}
                result.top10 = [by_id[paper_id] for paper_id in saved["top10_ids"]]
            else:
                result.top10 = rank_papers(result.papers, result.queries)
            if backend is not None:
                result.top10 = result.top10[:sources["analysis_limit"]]
            result.top3 = result.top10[:3]
            if saved is None:
                snapshot_domain(domain_index, result)
            for paper in result.papers:
                store.upsert_paper(paper)
            for rank, paper in enumerate(result.top10, 1):
                store.link_paper(run_id, paper, result.name, rank, False)
            store.commit()
            analysis_papers = result.top10
            vision = config["agent"].get("vision") or {}
            if backend is not None:
                pending = []
                for paper in analysis_papers:
                    prepared_id = "%d:%s" % (domain_index, paper.paper_id)
                    prepared = resumable and prepared_id in checkpoint["prepared_papers"]
                    if sources["full_text"] and not paper.full_text and not prepared:
                        try:
                            LOG.info("获取全文：%s", paper.title)
                            fetch_fulltext(paper, cache, sources["max_text_chars"])
                        except Exception as exc:
                            result.errors.append("全文未取得 (%s)：%s" % (paper.paper_id, exc))
                    elif sources["full_text"] and paper.full_text_path and not prepared:
                        try:
                            texts, _ = paper_sources(paper, cache)
                            body = texts["layout"]
                            if texts.get("tables"):
                                body += "\n\n[表格文本 · table text]\n" + texts["tables"]
                            paper.full_text = make_excerpt(body, sources["max_text_chars"])
                        except (ValueError, OSError) as exc:
                            result.errors.append("缓存正文读取失败 (%s)：%s" % (paper.paper_id, exc))
                    if resumable and not prepared:
                        checkpoint["prepared_papers"].append(prepared_id)
                        snapshot_domain(domain_index, result)
                    if vision.get("enabled") and paper.pdf_path and hasattr(backend, "describe_figures"):
                        try:
                            pages = paper.figure_pages or render_pages(paper.pdf_path, vision["max_pages"])
                            paper.figure_pages = pages
                            snapshot_domain(domain_index, result)
                            figures = backend.describe_figures(pages, folder / ("domain-%d-figures" % domain_index) / paper.paper_id)
                            paper.figure_context = format_figure_context(figures.get("figures", []))
                        except TaskPending:
                            raise
                        except Exception as exc:
                            result.errors.append("图注分析失败 (%s)：%s" % (paper.paper_id, exc))
                    key = analysis_key(paper, profile, config["agent"])
                    prior = None if resumable else store.cached_analysis(key)
                    if prior:
                        paper.previously_analyzed = True
                        result.analyses.append(prior)
                        store.save_analysis(run_id, paper.paper_id, prior)
                    else:
                        pending.append(paper)
                if pending:
                    expected = {paper.paper_id: paper for paper in pending}
                    chunk_size = 1
                    LOG.info("调用 %s 分析 %d 篇候选（每批 %d 篇）", config["agent"]["backend"], len(pending), chunk_size)
                    for chunk_index, offset in enumerate(range(0, len(pending), chunk_size)):
                        chunk = pending[offset:offset + chunk_size]
                        task_folder = folder / ("domain-%d-analysis-%d" % (domain_index, chunk_index))
                        try:
                            response = backend.run(analysis_prompt(chunk, profile), ANALYSIS_SCHEMA,
                                                   task_folder)
                        except TaskPending:
                            raise
                        except Exception as exc:
                            result.errors.append("模型分析失败（第 %d 批）：%s" % (chunk_index + 1, exc))
                            continue
                        returned_ids = [item["paper_id"] for item in response["analyses"]]
                        chunk_ids = {p.paper_id for p in chunk}
                        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != chunk_ids:
                            if resumable:
                                backend.reject(task_folder, "分析必须逐篇覆盖本批论文 ID，且每篇仅返回一次：" + "、".join(sorted(chunk_ids)))
                            result.errors.append("第 %d 批分析遗漏或重复了候选论文" % (chunk_index + 1))
                        for item in response["analyses"]:
                            paper = expected.get(item["paper_id"])
                            if paper is None:
                                result.errors.append("拒绝未知论文的分析：%s" % item["paper_id"])
                                continue
                            try:
                                analysis = validate_analysis(item, paper)
                                result.analyses.append(analysis)
                                store.save_analysis(run_id, paper.paper_id, analysis)
                                if not resumable:
                                    store.cache_analysis(analysis_key(paper, profile, config["agent"]), paper.paper_id, analysis)
                            except ValueError as exc:
                                if resumable:
                                    backend.reject(task_folder, "分析证据校验失败 (%s)：%s" % (paper.paper_id, exc))
                                result.errors.append("拒绝证据不足的分析 (%s)：%s" % (paper.paper_id, exc))
                    covered = {a["paper_id"] for a in result.analyses}
                    missing = [paper_id for paper_id in expected if paper_id not in covered]
                    if missing:
                        result.errors.append("模型未覆盖候选论文：%s" % "、".join(missing))
                review_daily_reading(config, result, backend, folder, domain_index)
                select_top3(result)
                for analysis in result.analyses:
                    store.save_analysis(run_id, analysis["paper_id"], analysis)
                evidence = [a for a in result.analyses if a["claims"]]
                if evidence:
                    try:
                        if resumable:
                            if str(domain_index) not in checkpoint["histories"]:
                                checkpoint["histories"][str(domain_index)] = store.recent_ideas()
                                save_checkpoint()
                            history = checkpoint["histories"][str(domain_index)]
                        else:
                            history = store.recent_ideas()
                        previous_titles = {re.sub(r"\s+", "", item["title"]).casefold() for item in history}
                        response = (idea_backend or backend).run(idea_prompt(evidence, profile, history), IDEA_SCHEMA, folder / ("domain-%d-ideas" % domain_index))
                        ids = {c["evidence_id"] for a in evidence for c in a["claims"]}
                        for item in response["ideas"]:
                            try:
                                idea = validate_idea(item, ids, profile)
                                title = re.sub(r"\s+", "", idea["title"]).casefold()
                                if title in previous_titles:
                                    continue
                                previous_titles.add(title)
                                result.ideas.append(idea)
                                store.save_idea(run_id, idea["idea_id"], idea)
                            except ValueError as exc:
                                if resumable:
                                    backend.reject(folder / ("domain-%d-ideas" % domain_index), "Idea 证据校验失败：" + str(exc))
                                result.errors.append("拒绝无依据的 idea：%s" % exc)
                    except TaskPending:
                        raise
                    except Exception as exc:
                        result.errors.append("Idea 生成失败：%s" % exc)
            if backend is None and not collect_only and not demo and result.top10:
                result.errors.append("完整讲解未执行：请配置可用的 agent.backend；本次仅保留候选线索。")
            for rank, paper in enumerate(result.top10, 1):
                store.link_paper(run_id, paper, result.name, rank, paper in result.top3)
            store.commit()
        status = "partial" if errors or any(d.errors for d in domains) else "complete"
        usage, seen_backends = [], set()
        for b in (backend, idea_backend):
            if b is not None and id(b) not in seen_backends and hasattr(b, "usage"):
                seen_backends.add(id(b))
                usage.extend(b.usage)
        payload = {"run_id": run_id, "started_at": started, "finished_at": cst_now(), "mode": mode,
                   "status": status, "domains": [asdict(d) for d in domains], "errors": errors, "usage": usage}
        atomic_text(folder / "report.json", json.dumps(payload, ensure_ascii=False, indent=2))
        write_report_bundle(folder, config, domains, started, mode, errors, usage)
        atomic_text(output / "latest.md", render_report(config, domains, started, mode, errors, usage,
            evidence_href=folder.name + "/evidence.md", details_href=folder.name + "/run-details.md"))
        store.finish_run(run_id, status, str(folder / "report.md"), "; ".join(errors))
        payload["report_path"] = str(folder / "report.md")
        payload["json_path"] = str(folder / "report.json")
        payload.update(evidence_path=str(folder / "evidence.md"), run_details_path=str(folder / "run-details.md"),
                       explanations_path=str(folder / "paper-explanations.json"))
        if resumable:
            checkpoint.update(status=status, pending_task=None)
            save_checkpoint()
            payload["session_path"] = str(folder / "session.json")
        return payload
    except TaskPending as exc:
        checkpoint.update(status="waiting", pending_task=exc.task)
        save_checkpoint()
        payload = {"run_id": run_id, "started_at": started, "finished_at": None, "mode": mode,
                   "status": "waiting", "domains": [asdict(d) for d in domains], "errors": errors, "usage": [],
                   "session_path": str(folder / "session.json"), "pending_task": exc.task,
                   "report_path": str(folder / "report.md"), "json_path": str(folder / "report.json")}
        atomic_text(folder / "report.json", json.dumps(payload, ensure_ascii=False, indent=2))
        write_report_bundle(folder, config, domains, started, mode, errors,
            prefix="> 当前调研等待 Codex 会话完成文件任务；这是阶段报告。\n\n")
        store.connection.execute("UPDATE runs SET status='waiting', finished_at=NULL, report_path=?, error='' WHERE id=?",
                                 (payload["report_path"], run_id))
        store.commit()
        return payload
    except BaseException as exc:
        store.finish_run(run_id, "failed", error=str(exc))
        raise
    finally:
        closed = set()
        for b in (backend, idea_backend):
            if b is not None and id(b) not in closed and hasattr(b, "close"):
                closed.add(id(b))
                try:
                    b.close()
                except Exception:
                    pass
        store.close()
        lock.release()
