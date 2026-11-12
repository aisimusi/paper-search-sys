from __future__ import annotations

from pathlib import Path

from .paper_explanations import FIELDS, MODULE_FIELDS, content_gaps


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def value_or_unknown(value: object) -> str:
    return "待确认" if value is None else str(value)


USAGE_TOTALS = ("input_tokens", "cache_read_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def render_usage(usage: list) -> list:
    lines = ["## Token 消耗", ""]
    if not usage:
        lines += ["本次未记录到模型 token 消耗（无模型调用，或 backend 未提供用量）。", ""]
        return lines
    lines += ["| 角色 | 模型 | 调用次数 | 输入(未缓存) | 缓存读 | 输出 | 推理 | 合计 |",
              "|---|---|---|---|---|---|---|---|"]
    totals = {field: 0 for field in USAGE_TOTALS}
    for record in usage:
        for field in totals:
            totals[field] += int(record.get(field) or 0)
        lines.append("| %s | %s | %d | %d | %d | %d | %d | %d |" % (
            cell(record.get("role", "")), cell(record.get("model", "")), int(record.get("calls") or 0),
            int(record.get("input_tokens") or 0), int(record.get("cache_read_tokens") or 0),
            int(record.get("output_tokens") or 0), int(record.get("reasoning_tokens") or 0),
            int(record.get("total_tokens") or 0)))
    lines += ["| **合计** | — | — | %d | %d | %d | %d | %d |" % (
        totals["input_tokens"], totals["cache_read_tokens"], totals["output_tokens"],
        totals["reasoning_tokens"], totals["total_tokens"]), "",
        "输入(未缓存)=未命中缓存的 prompt token；缓存读=命中缓存的 prompt token；合计=模型返回的 totalTokens 之和。",
        "数据来自模型返回的 usage 字段；未返回用量的调用（如失败调用）不计入。", ""]
    return lines


class EvidenceLinks:
    def __init__(self, records, href="evidence.md"):
        self.records = dict(records)
        self.numbers = {eid: index for index, eid in enumerate(self.records, 1)}
        self.href = href

    def refs(self, ids):
        return " ".join("[%d](%s#evidence-%d)" % (self.numbers[eid], self.href, self.numbers[eid])
                        for eid in dict.fromkeys(ids) if eid in self.numbers)


def domain_evidence(domains):
    records = {}
    for domain in domains:
        if getattr(domain, "reading_review", {}).get("status") in {"pending", "unavailable"}:
            continue
        papers = {p.paper_id: p for p in domain.papers}
        for analysis in domain.analyses:
            paper = papers[analysis["paper_id"]]
            for claim in analysis["claims"]:
                records[claim["evidence_id"]] = {**claim, **getattr(domain, "evidence_records", {}).get(claim["evidence_id"], {}),
                    "paper_id": paper.paper_id, "title": paper.title, "source_url": paper.url,
                    "version": paper.updated, "full_text_path": paper.full_text_path}
    return records


def render_evidence(records, *, reviewed=False):
    links = EvidenceLinks(records)
    lines = ["# 原文证据", "", "[返回报告](report.md)", "",
             "连续引文已核对。" + ("语义支持已接受独立模型诊断，仍需人工复核。" if reviewed else "语义支持状态见运行记录。"), ""]
    for eid, row in links.records.items():
        number = links.numbers[eid]
        lines += ['<a id="evidence-%d"></a>' % number, "", "## [%d] %s" % (number, row.get("title") or row["paper_id"]), "",
                  "证据编号：`%s` · 论文版本：%s" % (eid, row.get("version", "未知")), "",
                  "[论文来源](%s)" % row["source_url"], "", "**支持的断言**：" + row["text"], "", "**原文**：", ""]
        lines += ["> " + line for line in row["anchor_quote"].splitlines()] + [""]
        for location in row.get("locations", []):
            section = " / ".join(s["name"] for s in location["sections"])
            pages = "%s–%s" % (location["page_start"], location["page_end"]) if location["page_start"] is not None else "未知"
            lines += ["定位：%s · PDF 页码 %s · 字符 [%d, %d) · `%s`" %
                      (section, pages, location["start"], location["end"], location["source_id"]), ""]
        if row.get("full_text_path"):
            lines += ["[本地提取正文](<%s>)" % Path(row["full_text_path"]).as_posix(), ""]
    return "\n".join(lines)


def render_paper(paper, analysis, explanation, links, *, heading="####", detailed=True):
    lines = [heading + " [" + paper.title + "](" + paper.url + ")", ""]
    section_heading = "#" * min(len(heading) + 1, 6)
    module_heading = "#" * min(len(heading) + 2, 6)
    metadata = ["、".join(paper.authors), paper.updated[:10]]
    lines += [" · ".join(part for part in metadata if part), ""]
    if not analysis or not analysis["claims"]:
        lines += ["**摘要线索**：" + (analysis["discovery_summary"] if analysis else paper.abstract), "",
                  "阅读范围：尚无可用的正文结论。", ""]
        return lines
    lines += ["阅读范围：已取得的正文片段。", ""]
    if explanation:
        fields = FIELDS[:4] if detailed else (FIELDS[2], FIELDS[3])
        for key, label in fields:
            part = explanation[key]
            if part["text"].strip():
                lines += [section_heading + " " + label, "", part["text"] + " " + links.refs(part["evidence_ids"]), ""]
        if detailed:
            for module in explanation["modules"]:
                lines += [module_heading + " " + module["name"], ""]
                for key, label in MODULE_FIELDS:
                    part = module[key]
                    if key == "rationale":
                        label = "设计理由（分析推断）" if module["rationale_basis"] == "inference" else "设计理由（作者陈述）"
                    lines += ["- **%s**：%s%s" % (label, part["text"] or "当前材料缺少可核查的说明。",
                              " " + links.refs(part["evidence_ids"]) if part["text"] else "")]
                lines.append("")
            for key, label in FIELDS[4:]:
                part = explanation[key]
                if part["text"].strip():
                    lines += [section_heading + " " + label, "", part["text"] + " " + links.refs(part["evidence_ids"]), ""]
            gaps = [g["reason"] for g in content_gaps(explanation)] + explanation["missing_information"]
            if gaps:
                lines += ["**讲解缺口**：" + "；".join(g.rstrip("。；") for g in dict.fromkeys(gaps)), ""]
    else:
        kinds = (("innovation", "核心创新"), ("method", "方法要点"), ("result", "实验支持"), ("limitation", "局限"))
        for kind, label in kinds if detailed else kinds[:2]:
            claims = [c for c in analysis["claims"] if c["kind"] == kind]
            if not detailed:
                claims = claims[:1]
            if claims:
                lines += ["**" + label + "**", ""]
                for claim in claims:
                    lines += [claim["text"] + " " + links.refs([claim["evidence_id"]]), ""]
        if detailed:
            lines += ["**讲解状态**：完整设计说明尚未通过核查，以上保留可用的正文断言。", ""]
    if detailed and analysis["missing_evidence"]:
        lines += ["**待补证据**：" + "；".join(g.rstrip("。；") for g in analysis["missing_evidence"]), ""]
    return lines


def render_idea(idea, links, *, heading="####"):
    lines = [heading + " " + idea["title"], "", "状态：%s · 未验证假设，新颖性待独立查重" % idea["feasibility"], ""]
    for label, field in (("假设", "hypothesis"), ("机制", "mechanism"), ("与最接近工作的差异", "closest_work_and_difference"),
                         ("最小实验", "minimal_experiment"), ("对照", "baseline"), ("指标", "metric"), ("反证与停止条件", "falsification")):
        lines += ["**%s**：%s" % (label, idea[field]), ""]
    lines += ["依据：" + links.refs(idea["evidence_ids"]), "",
              "所需数据：" + "；".join(item.rstrip("。；") for item in idea["required_data"]), "",
              "资源估计：单卡显存 %s GiB，GPU 小时 %s。%s" %
              (value_or_unknown(idea["required_vram_gb"]), value_or_unknown(idea["estimated_gpu_hours"]), idea["estimate_basis"]), ""]
    if idea["missing_resources"]:
        lines += ["待补条件：" + "；".join(item.rstrip("。；") for item in idea["missing_resources"]), ""]
    return lines


def render_report(config: dict, domains: list, started: str, mode: str, errors: list, usage: list | None = None,
                  *, evidence_href="evidence.md", details_href="run-details.md") -> str:
    links = EvidenceLinks(domain_evidence(domains), evidence_href)
    lines = ["# 每日研究简报", "", "项目：%s · 时间：%s" % (config["project"]["name"], started), "",
             "[原文证据](%s) · [运行记录与评分](%s)" % (evidence_href, details_href), ""]
    if mode == "demo":
        lines += ["> **离线流程演示：以下样本全部为虚构测试数据。**", ""]
    elif mode == "collect":
        lines += ["> 本次仅完成收集和初筛，候选介绍来自摘要。", ""]
    else:
        lines += ["结论限于已取得材料；文中短编号链接到完整原文。研究 idea 属于未验证假设。", ""]
    for domain in domains:
        analyses = {a["paper_id"]: a for a in domain.analyses}
        if getattr(domain, "reading_review", {}).get("status") in {"pending", "unavailable"}:
            analyses = {pid: dict(a, claims=[]) for pid, a in analyses.items()}
        explanations = getattr(domain, "paper_explanations", {})
        lines += ["## " + domain.name, "", "找到 %d 篇去重论文，保留 %d 篇候选。" % (len(domain.papers), len(domain.top10)), ""]
        if domain.top3:
            top = domain.top3[0]
            evaluated = analyses.get(top.paper_id, {}).get("quality_score") is not None
            lines += ["**优先阅读**：[%s](%s)" % (top.title, top.url), "",
                      "在本次完成五维评分的候选中排序最高。" if evaluated else "当前按初筛相关性安排阅读，完整质量比较尚缺证据。", ""]
        if domain.top3:
            lines += ["### 重点论文", ""]
        for paper in domain.top3:
            lines += render_paper(paper, analyses.get(paper.paper_id), explanations.get(paper.paper_id), links)
        others = [p for p in domain.top10 if p not in domain.top3]
        if others:
            lines += ["### 其余论文", ""]
            for paper in others:
                lines += render_paper(paper, analyses.get(paper.paper_id), explanations.get(paper.paper_id), links)
        lines += ["### 可验证的研究假设", ""]
        if not domain.ideas:
            lines += ["本次尚无证据和条件充分的研究假设。", ""]
        for idea in domain.ideas:
            lines += render_idea(idea, links)
        if domain.errors:
            lines += ["### 影响交付的未完成项", ""] + ["- " + e for e in domain.errors] + [""]
    if errors:
        lines += ["## 影响交付的运行问题", ""] + ["- " + e for e in errors] + [""]
    return "\n".join(lines)


def render_run_details(config, domains, started, mode, errors, usage=None):
    lines = ["# 运行记录", "", "[返回报告](report.md)", "", "时间：%s · 运行方式：%s" % (started, mode), ""]
    if mode == "codex-session":
        lines += ["## Token 消耗", "", "当前 Codex 会话用量以 Codex 显示为准。", ""]
    else:
        lines += render_usage(usage or [])
    lines += ["初筛分衡量主题匹配、新近程度和社区关注度。社区热度不代表科研质量。", "",
              "质量权重：创新 30%、方法 25%、实验 25%、写作 10%、相关性 10%；证据不足的维度不评分。", ""]
    for domain in domains:
        analyses = {a["paper_id"]: a for a in domain.analyses}
        lines += ["## " + domain.name, "", "检索式：" + "；".join(domain.queries), "",
                  "| 论文 | 首发日期 | 来源轨道 | 初筛分 | 创新 | 方法 | 实验 | 写作 | 相关 | 加权分 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for paper in domain.top10:
            analysis = analyses.get(paper.paper_id, {})
            scores = analysis.get("scores", {})
            values = [scores.get(k) for k in ("novelty", "method", "experiments", "writing", "relevance")]
            lines.append("| %s | %s | %s | %.3f | %s | %s |" % (cell(paper.title), paper.published[:10], paper.track,
                paper.relevance_score, " | ".join("—" if v is None else str(v) for v in values), value_or_unknown(analysis.get("quality_score"))))
        lines.append("")
        for paper in domain.top10:
            analysis = analyses.get(paper.paper_id)
            if analysis:
                lines += ["### " + paper.title, "", "阅读范围：" + analysis["read_limit"], "",
                          "发现性概述（摘要层）：" + analysis["discovery_summary"], "",
                          "与已有基础：" + analysis["fit_to_lab"], "", "资源条件：" + analysis["resource_notes"], "",
                          "引文未通过连续匹配的条目数：%d" % analysis.get("discarded_claims", 0), ""]
            if paper.figure_context:
                lines += ["视觉描述（独立视觉来源）：", "", paper.figure_context, ""]
        for idea in domain.ideas:
            lines += ["### " + idea["title"], "", "已有基础：" + idea["lab_fit"], "",
                      "查重检索式：" + "；".join(idea["novelty_check_queries"]), ""]
        lines += ["- " + e for e in domain.errors] + [""]
    lines += ["- " + e for e in errors]
    return "\n".join(lines)
