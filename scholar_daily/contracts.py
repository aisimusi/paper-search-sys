from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict

from .models import Paper

TEXT = {"type": "string"}
NULL_NUMBER = {"type": ["number", "null"], "minimum": 1, "maximum": 5}
WEIGHTS = {"novelty": .30, "method": .25, "experiments": .25, "writing": .10, "relevance": .10}


def obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def arr(item: dict, maximum: int = 20) -> dict:
    return {"type": "array", "items": item, "maxItems": maximum}


ANALYSIS = obj({
    "paper_id": TEXT,
    "reading_level": {"type": "string", "enum": ["abstract_only", "paper_excerpt"]},
    "discovery_summary": TEXT,
    "claims": arr(obj({
        "kind": {"type": "string", "enum": ["innovation", "method", "result", "limitation"]},
        "text": TEXT,
        "anchor_quote": TEXT,
    }), 20),
    "scores": obj({key: NULL_NUMBER for key in WEIGHTS}),
    "fit_to_lab": TEXT,
    "resource_notes": TEXT,
    "missing_evidence": arr(TEXT),
})
ANALYSIS_SCHEMA = obj({"analyses": arr(ANALYSIS, 10)})
IDEA = obj({
    "title": TEXT,
    "hypothesis": TEXT,
    "evidence_ids": arr(TEXT, 10),
    "mechanism": TEXT,
    "lab_fit": TEXT,
    "closest_work_and_difference": TEXT,
    "novelty_check_queries": arr(TEXT, 5),
    "minimal_experiment": TEXT,
    "baseline": TEXT,
    "metric": TEXT,
    "falsification": TEXT,
    "required_data": arr(TEXT),
    "required_vram_gb": {"type": ["number", "null"], "minimum": 0},
    "estimated_gpu_hours": {"type": ["number", "null"], "minimum": 0},
    "estimate_basis": TEXT,
    "missing_resources": arr(TEXT),
    "feasibility": {"type": "string", "enum": ["candidate", "needs_info", "blocked"]},
})
IDEA_SCHEMA = obj({"ideas": arr(IDEA, 3)})
FIGURE = obj({
    "page": {"type": "number", "minimum": 1},
    "kind": {"type": "string", "enum": ["figure", "table", "other"]},
    "caption": TEXT,
    "description": TEXT,
})
FIGURE_SCHEMA = obj({"figures": arr(FIGURE, 40)})


def validate_shape(value: Any, schema: dict, path: str = "$", /) -> None:
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    checks = {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str),
              "number": type(value) in (int, float), "null": value is None}
    if not any(checks.get(kind, False) for kind in kinds):
        raise ValueError("%s 类型不符合 %s" % (path, kinds))
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("%s 不在允许的枚举值中" % path)
    if isinstance(value, dict):
        properties = schema["properties"]
        # 放宽解析器：允许模型多输出字段（忽略额外字段），但仍要求所有必需字段存在。
        missing = [key for key in properties if key not in value]
        if missing:
            raise ValueError("%s 缺少字段：%s" % (path, "、".join(missing)))
        for key, child in properties.items():
            validate_shape(value[key], child, path + "." + key)
    elif isinstance(value, list):
        if len(value) > schema.get("maxItems", 10**9):
            raise ValueError("%s 数组过长" % path)
        for index, child in enumerate(value):
            validate_shape(child, schema["items"], path + "[%d]" % index)
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 10**9):
            raise ValueError("%s 文本长度不符合约束" % path)
    elif value is not None:
        import math
        if not math.isfinite(value) or value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            raise ValueError("%s 数值超出范围" % path)


def normalized(text: str) -> str:
    """Canonicalize text for evidence-quote matching.

    The model is expected to copy quotes verbatim, but PDF extraction (pdftotext)
    and model output can differ in typography that does not change meaning:
    Unicode dashes/quotes, and line-break hyphenation (``real-\\ntime`` becomes
    ``real- time``). Normalize both sides so a faithful quote still matches while
    word order and wording remain exact.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Unify typographic dashes and quotes to ASCII.
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2043\ufe58\ufe63\uff0d]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)
    # Join hyphenated line breaks ("real-\ntime" -> "realtime").
    text = re.sub(r"-\s+", "", text)
    # Drop remaining hyphens so "multi-object" == "multiobject".
    text = re.sub(r"-", "", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_analysis(analysis: Dict[str, Any], paper: Paper) -> Dict[str, Any]:
    validate_shape(analysis, ANALYSIS)
    if analysis["paper_id"] != paper.paper_id:
        raise ValueError("分析引用了未知的论文 ID")
    expected = "paper_excerpt" if paper.full_text else "abstract_only"
    if analysis["reading_level"] != expected:
        raise ValueError("阅读范围与提供的证据不符")
    if not paper.full_text:
        if analysis["claims"] or any(score is not None for key, score in analysis["scores"].items() if key != "relevance"):
            raise ValueError("摘要不能生成已验证贡献或方法/实验/创新评分")
    # 放宽校验：只丢弃无法逐字核对的 claims，不再因一条坏引文整篇拒收。
    # 证据门槛不变：留下的每条 claim 仍必须有论文原文片段中的逐字引文。
    text = normalized(paper.full_text)
    kept, discarded = [], 0
    for claim in analysis["claims"]:
        quote = claim.get("anchor_quote") or ""
        # 长度也在逐条丢弃，而不是整篇拒收：过短/过长或无法逐字核对的引文一律剔除。
        if 16 <= len(quote) <= 2000 and "[EXCERPT GAP:" not in quote and normalized(quote) in text:
            kept.append(claim)
        else:
            discarded += 1
    scores = dict(analysis["scores"])
    for dimension, kind in [("novelty", "innovation"), ("method", "method"), ("experiments", "result")]:
        if scores[dimension] is not None and not any(claim["kind"] == kind for claim in kept):
            scores[dimension] = None
    result = dict(analysis)
    result["claims"] = []
    for claim in kept:
        evidence_id = "ER-" + hashlib.sha256((paper.paper_id + claim["anchor_quote"]).encode()).hexdigest()[:16]
        result["claims"].append(dict(claim, evidence_id=evidence_id))
    result["scores"] = scores
    result["discarded_claims"] = discarded
    result["quality_score"] = round(sum(scores[key] * weight for key, weight in WEIGHTS.items()), 3) if all(v is not None for v in scores.values()) else None
    result["source_url"] = paper.url
    result["read_limit"] = "只分析提供的论文文本片段；未验证未提供的图表、附录或完整相关工作" if paper.full_text else "摘要发现线索，未读全文"
    return result


def validate_idea(idea: dict, evidence_ids: set, profile: dict) -> dict:
    validate_shape(idea, IDEA)
    if not idea["evidence_ids"] or not set(idea["evidence_ids"]) <= evidence_ids:
        raise ValueError("idea 必须引用本次已校验的全文证据")
    for key in ["hypothesis", "mechanism", "minimal_experiment", "baseline", "metric", "falsification", "closest_work_and_difference"]:
        if not idea[key].strip():
            raise ValueError("idea 缺少 " + key)
    result = dict(idea)
    result["novelty_status"] = "unverified_hypothesis"
    compute = profile.get("compute", {})
    gpus = compute.get("gpus", [])
    capacities = [g.get("available_vram_gb") for g in gpus if isinstance(g.get("available_vram_gb"), (int, float))]
    needed = idea["required_vram_gb"]
    budget = compute.get("max_experiment_hours")
    hours = idea["estimated_gpu_hours"]
    notes = list(idea["missing_resources"])
    if needed is None or (needed > 0 and not capacities):
        notes.append("待确认所需显存与实际可用的单卡显存")
    if hours is None or budget is None:
        notes.append("待确认运行时间估算及实验时长预算")
    known_data = {str(d).casefold() for d in profile.get("lab_context", {}).get("datasets", [])}
    if any(d.casefold() not in known_data for d in idea["required_data"]):
        notes.append("至少一种实验数据尚未记录为课题组已有数据")
    blocked = (needed is not None and capacities and needed > max(capacities)) or (hours is not None and budget is not None and hours > budget)
    if blocked:
        result["feasibility"] = "blocked"
        notes.append("估计资源需求超出配置预算；不能把多卡显存简单相加")
    elif notes and result["feasibility"] != "blocked":
        result["feasibility"] = "needs_info"
    result["missing_resources"] = list(dict.fromkeys(notes))
    result["idea_id"] = "IDEA-" + hashlib.sha256(normalized(idea["title"]).casefold().encode()).hexdigest()[:16]
    return result
