"""Complete per-paper explanations, bounded repairs and final semantic review."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict

from .agents import TaskPending
from .config import resolve_path
from .contracts import ANALYSIS_SCHEMA, validate_analysis, validate_shape
from .fulltext import make_excerpt
from .models import Paper
from .paper_context import load_index, locate_quote, paper_sources, save_index
from .paper_explanations import (EXPLANATION_SCHEMA, content_gaps, empty_explanation,
                                 explanation_prompt, validate_explanation_batch)
from .prompts import analysis_prompt

MAX_READING_ROUNDS = 3
READING_ERRORS = (ValueError, RuntimeError, OSError, subprocess.SubprocessError)


def review_daily_reading(config, result, backend, folder, domain_index):
    from .pipeline import atomic_text
    from .research_verifier import Verifier, approved_state, delivery_view

    root = folder / ("domain-%d-reading" % domain_index)
    root.mkdir(parents=True, exist_ok=True)
    analyses = {a["paper_id"]: a for a in result.analyses}
    papers = {p.paper_id: Paper(**asdict(p)) for p in result.papers}
    focused = [p.paper_id for p in result.top10]
    profile = config["research_profile"]
    result.reading_review = {"status": "pending"}
    indexes, texts, contexts, source_errors = {}, {}, {}, {}

    # Freeze all sources before any explanation call, including papers with no
    # usable first-pass claims. Resume cannot silently read a changed cache.
    for pid in focused:
        if not papers[pid].full_text:
            continue
        try:
            receipt = root / ("context-" + hashlib.sha256(pid.encode()).hexdigest()[:16] + ".json")
            if receipt.exists():
                record = json.loads(receipt.read_text(encoding="utf-8"))
                index, bodies = load_index(record, folder=root)
                if index["paper_id"] != pid or index["version"] != papers[pid].updated:
                    raise ValueError("每日阅读的原文快照与论文版本不一致")
            else:
                bodies, page_sources = paper_sources(papers[pid], resolve_path(config, "cache_dir"))
                normalized = [re.sub(r"\s+", " ", text) for text in bodies.values()]
                if not all(any(re.sub(r"\s+", " ", c["anchor_quote"]) in body for body in normalized)
                           for c in analyses.get(pid, {}).get("claims", [])):
                    bodies, page_sources = {"layout": papers[pid].full_text}, []
                index, record = save_index(root, papers[pid], bodies, page_sources)
                atomic_text(receipt, json.dumps(record, ensure_ascii=False, indent=2))
            indexes[pid], texts[pid], contexts[pid] = index, bodies, record
        except READING_ERRORS as exc:
            source_errors[pid] = str(exc)

    class Calls:
        stage = ""

        def call(self, key, role, prompt, schema, check=None):
            target = root / self.stage / key
            value = backend.run(prompt, schema, target)
            try:
                validate_shape(value, schema)
                if check:
                    check(value)
            except ValueError as exc:
                if getattr(backend, "resumable", False):
                    backend.reject(target, "阅读产物校验失败：" + str(exc))
                raise
            return value

    calls = Calls()
    final_analyses, explanations, records, paper_reviews = {}, {}, {}, {}
    candidates = {}
    for pid in focused:
        paper = papers[pid]
        analysis = analyses.get(pid)
        if not analysis or not paper.full_text or pid not in indexes:
            if analysis:
                analysis = dict(analysis, claims=[], quality_score=None,
                    scores={key: None if key != "relevance" else score for key, score in analysis["scores"].items()},
                    discovery_summary=paper.abstract)
                final_analyses[pid] = analysis
            reason = source_errors.get(pid) or ("未取得正文" if not paper.full_text else "初次分析未完成")
            result.errors.append("论文完整讲解未完成 (%s)：%s。" % (pid, reason))
            continue

        token = hashlib.sha256(pid.encode()).hexdigest()[:16]
        explanation = empty_explanation(pid)
        feedback, history = "", []
        state = None
        try:
            for round_index in range(MAX_READING_ROUNDS):
                calls.stage = "paper-%s/round-%d" % (token, round_index)
                if round_index:
                    # Reread the frozen main text and appendices. A feedback query
                    # guides bounded selection when the document exceeds the cap.
                    body = texts[pid]["layout"]
                    if texts[pid].get("tables"):
                        body += "\n\n[表格文本 · table text]\n" + texts[pid]["tables"]
                    query = " ".join(row.get("query", "") for row in issues)
                    receipt = root / calls.stage / "supplement.json"
                    if receipt.exists():
                        paper = Paper(**json.loads(receipt.read_text(encoding="utf-8")))
                    else:
                        paper = Paper(**paper.as_dict())
                        paper.full_text = make_excerpt(body, config["sources"]["max_text_chars"], query=query)
                        atomic_text(receipt, json.dumps(paper.as_dict(), ensure_ascii=False, indent=2))

                    def check_analysis(value):
                        if [a["paper_id"] for a in value["analyses"]] != [pid]:
                            raise ValueError("补读分析必须逐篇覆盖当前论文")
                        validate_analysis(value["analyses"][0], paper)

                    value = calls.call("supplement-analysis", "literature_analyst",
                        feedback + analysis_prompt([paper], profile), ANALYSIS_SCHEMA, check_analysis)
                    analysis = validate_analysis(value["analyses"][0], paper)

                if analysis["claims"]:
                    value = calls.call("%s-explanations-0" % ("initial" if not round_index else "supplement"),
                        "literature_analyst", feedback + explanation_prompt([analysis], profile), EXPLANATION_SCHEMA,
                        lambda value: validate_explanation_batch(value, [analysis]))
                    explanation = validate_explanation_batch(value, [analysis])[pid]
                else:
                    explanation = empty_explanation(pid)
                candidates[pid] = explanation
                evidence = {c["evidence_id"]: dict(c, paper_id=pid, source_url=paper.url, version=paper.updated,
                            index_path=contexts[pid]["index_path"], locations=locate_quote(indexes[pid], texts[pid], c["anchor_quote"]))
                            for c in analysis["claims"]}
                state = {"goal": "逐篇完整解释创新、训练与推理、模块输入/处理/输出、设计理由、实验条件及局限",
                         "papers": {pid: paper.as_dict()}, "analyses": {pid: analysis},
                         "paper_explanations": {pid: explanation} if analysis["claims"] else {},
                         "evidence": evidence, "paper_contexts": {pid: contexts[pid]}, "facts": {}, "config": config,
                         "steps": [], "verification": {}, "artifacts": {"analysis": {"paper_ids": [pid]}}}
                review = Verifier(calls).review(state, {"phase": "evidence"})
                state["verification"]["evidence"] = review
                gaps = content_gaps(explanation)
                issues = [{"field": g["field"], "reason": g["reason"], "query": g["query"]} for g in gaps]
                issues += [{"field": row["field"], "reason": "; ".join(
                                claim["reason"] for claim in row["claims"] if not claim["passed"]) or "; ".join(row["missing_information"]) or row["structural_error"] or "审查未完整覆盖当前断言",
                            "query": row["query"]} for row in review["items"] if not row["passed"]]
                # Final review checks the summary and the actual deliverable prose.
                if not issues or round_index == MAX_READING_ROUNDS - 1:
                    final = Verifier(calls).review(state, {"phase": "final"})
                    state["verification"]["final"] = final
                    issues += [{"field": row["field"], "reason": "; ".join(
                                    claim["reason"] for claim in row["claims"] if not claim["passed"]) or "; ".join(row["missing_information"]) or row["structural_error"] or "审查未完整覆盖当前断言",
                                "query": row["query"]} for row in final["items"] if not row["passed"]]
                history.append({"round": round_index + 1, "evidence": review,
                                "final": state["verification"].get("final"), "issues": issues})
                atomic_text(root / calls.stage / "review.json", json.dumps(history[-1], ensure_ascii=False, indent=2))
                if not issues:
                    break
                feedback = "根据本篇上一轮审查修复完整讲解，保留实验条件与有效结论；表达问题须展开具体对象、操作和条件，证据缺口须补充正确原文。\n" + json.dumps({
                    "issues": issues, "previous_analysis": {k: v for k, v in analysis.items() if k != "paper_id"},
                    "previous_explanation": {k: v for k, v in explanation.items() if k != "paper_id"}}, ensure_ascii=False) + "\n"

            safe = delivery_view(state) if "final" in state["verification"] else approved_state(state)
            final_analyses.update(safe["analyses"])
            explanations.update(safe["paper_explanations"])
            records.update(safe["evidence"])
            papers[pid] = paper
            paper_reviews[pid] = {"rounds": history, "complete": not issues}
            if issues:
                result.errors.append("论文讲解尚不完整 (%s)，已执行%d轮阅读与审查：%s。" % (
                    pid, len(history), "；".join(dict.fromkeys(row["reason"] for row in issues))[:800]))
        except TaskPending:
            raise
        except READING_ERRORS as exc:
            # A failed paper cannot leak unreviewed claims into reports or ideas.
            final_analyses[pid] = dict(analysis, claims=[], quality_score=None,
                scores={key: None if key != "relevance" else score for key, score in analysis["scores"].items()},
                discovery_summary=paper.abstract)
            paper_reviews[pid] = {"rounds": history, "complete": False, "error": str(exc)}
            result.errors.append("论文讲解与语义核查未完成 (%s)：%s" % (pid, exc))

    result.analyses = list(final_analyses.values())
    result.paper_explanations = explanations
    result.evidence_records = records
    for paper in result.papers:
        paper.full_text = papers[paper.paper_id].full_text
    review_items = [item for row in paper_reviews.values() for item in (
        row["rounds"][-1]["evidence"]["items"] if row["rounds"] else [])]
    result.reading_review = {"status": "checked", "items": review_items, "paper_reviews": paper_reviews,
                            "passed": all(paper_reviews.get(pid, {}).get("complete", False) for pid in focused)}
    atomic_text(root / "reading-review.json", json.dumps(result.reading_review, ensure_ascii=False, indent=2))
    atomic_text(root / "candidate-explanations.json", json.dumps(candidates, ensure_ascii=False, indent=2))
