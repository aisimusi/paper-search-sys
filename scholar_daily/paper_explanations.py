"""Reader-facing explanations linked to the unchanged Evidence Record contract."""
from __future__ import annotations

import copy
import json

from .contracts import TEXT, arr, obj, validate_shape

CITED_TEXT = obj({"text": dict(TEXT, maxLength=1200), "evidence_ids": arr(TEXT, 12)})
MODULE = obj({
    "name": dict(TEXT, minLength=1, maxLength=100),
    "role": CITED_TEXT,
    "inputs": CITED_TEXT,
    "operation": CITED_TEXT,
    "outputs": CITED_TEXT,
    "rationale": CITED_TEXT,
    "rationale_basis": {"type": "string", "enum": ["author", "inference", "unknown"]},
    "validation": CITED_TEXT,
})
EXPLANATION = obj({
    "paper_id": TEXT,
    "problem": CITED_TEXT,
    "baseline": CITED_TEXT,
    "novelty": CITED_TEXT,
    "pipeline": CITED_TEXT,
    "modules": arr(MODULE, 6),
    "results": CITED_TEXT,
    "limitations": CITED_TEXT,
    "missing_information": arr(TEXT, 20),
})
EXPLANATION_SCHEMA = obj({"explanations": arr(EXPLANATION, 10)})

FIELDS = (("problem", "研究问题"), ("baseline", "已有方法与不足"), ("novelty", "核心创新"),
          ("pipeline", "整体流程"), ("results", "实验支持"), ("limitations", "适用条件与局限"))
MODULE_FIELDS = (("role", "作用及组件来源"), ("inputs", "输入"), ("operation", "处理过程"),
                 ("outputs", "输出"), ("rationale", "设计理由"), ("validation", "模块实验"))
QUERIES = {"problem": "introduction motivation challenge", "baseline": "related work baseline limitations",
           "novelty": "contributions compared with prior methods", "pipeline": "method framework architecture",
           "modules": "method module architecture", "role": "method proposed existing components",
           "inputs": "method input notation", "operation": "method algorithm equations",
           "outputs": "method output", "rationale": "method motivation design rationale",
           "validation": "ablation component analysis", "results": "experiments evaluation protocol results",
           "limitations": "limitations failure cases discussion"}


def empty_explanation(paper_id, reason=""):
    value = {key: {"text": "", "evidence_ids": []} for key, _ in FIELDS}
    value.update(paper_id=paper_id, modules=[], missing_information=[reason] if reason else [])
    return value


def cited_parts(explanation):
    for key, _ in FIELDS:
        yield key, explanation[key]
    for index, module in enumerate(explanation["modules"]):
        for key, _ in MODULE_FIELDS:
            yield "modules.%d.%s" % (index, key), module[key]


def validate_explanation(value, analysis):
    validate_shape(value, EXPLANATION)
    if value["paper_id"] != analysis["paper_id"]:
        raise ValueError("讲解引用了其他论文")
    allowed = {c["evidence_id"] for c in analysis["claims"]}
    for field, part in cited_parts(value):
        ids = part["evidence_ids"]
        if len(ids) != len(set(ids)) or not set(ids) <= allowed:
            raise ValueError("讲解 %s 引用了未知、重复或其他论文的 Evidence Record" % field)
        if bool(part["text"].strip()) != bool(ids):
            raise ValueError("讲解 %s 的文字与证据必须同时提供；缺证据时留空" % field)
        if part["text"].strip().rstrip("。.") in {"待确认", "待补证据", "未知", "暂无", "未提供", "证据待补"}:
            raise ValueError("讲解 %s 使用了占位文字；缺证据时应留空并记录缺口" % field)
    names = [m["name"].strip().casefold() for m in value["modules"]]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("讲解的模块名称不能重复")
    for module in value["modules"]:
        if bool(module["rationale"]["text"].strip()) != (module["rationale_basis"] != "unknown"):
            raise ValueError("模块设计理由必须标明作者陈述或分析推断，未知时留空")
    return copy.deepcopy(value)


def validate_explanation_batch(value, analyses):
    validate_shape(value, EXPLANATION_SCHEMA)
    expected = {a["paper_id"]: a for a in analyses}
    actual = [e["paper_id"] for e in value["explanations"]]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("讲解必须逐篇覆盖本批论文，ID 不能遗漏、重复或增加")
    return {e["paper_id"]: validate_explanation(e, expected[e["paper_id"]]) for e in value["explanations"]}


def content_gaps(explanation):
    """Structural completeness only; semantic sufficiency is checked by Verifier."""
    gaps = []
    for key, label in FIELDS:
        if not explanation[key]["text"].strip():
            gaps.append({"field": key, "reason": "缺少" + label, "query": QUERIES[key]})
    if not explanation["modules"]:
        gaps.append({"field": "modules", "reason": "缺少主要模块的输入、处理、输出和设计理由", "query": QUERIES["modules"]})
    for index, module in enumerate(explanation["modules"]):
        for key, label in MODULE_FIELDS:
            if not module[key]["text"].strip():
                gaps.append({"field": "modules.%d.%s" % (index, key),
                             "reason": "模块“%s”缺少%s" % (module["name"], label),
                             "query": module["name"] + " " + QUERIES[key]})
    return gaps


def explanation_prompt(analyses, profile):
    from .prompts import RULES
    return RULES + """
将给定的已完成连续引文核对的断言整理成中文论文讲解。逐篇返回 explanations。
事实只能来自本篇给定的 Evidence Record；每个非空 text 必须引用支持全部实质内容的 evidence_ids。
引用存在尚不等于语义正确，讲解将另行接受独立审查。不要在 text 中重复证据编号或原文引文。
problem 说明具体任务和困难；baseline 点明已有方法及其不足；novelty 写清相对该方法改变了什么。
pipeline 按数据流说明输入如何经过主要步骤得到输出，不能只罗列模块名称。
modules 聚焦主要新模块或关键改造，每个模块说明 role（作用及新增/改造/沿用的来源）、inputs、operation、outputs。
完整讲解逐篇采用相同标准，不因论文排序省略模块。模块数量按论文实际主要组件确定，不为凑数量拆分或虚构组件。
训练与推理分别说明；首次使用的符号、缩写须定义，并说明模块输出如何被后续模块使用。
涉及公式时解释各项的对象、约束和梯度作用范围；原文提供概率归一化、停止梯度或特殊类别处理时保留这些细节。
rationale 说明该操作如何针对前述困难；作者明确陈述时 rationale_basis=author，条件性机制解释标为 inference，不能把推断写成实验结论。
validation 说明消融如何支持该模块，结果与对照条件必须同时交代；整体性能不能冒充单模块收益。
results 保留数据集、评测条件、对照、指标和收益范围；limitations 说明适用前提及失败条件。
实验区分型号、数据集划分、额外训练数据、硬件和数值精度。消融同时改变多个因素时指出归因限制；小幅点估计不等于统计显著。
某模块原文没有单独消融时，validation 可明确给出已有联合对照及其不能隔离该模块贡献的限制，并引用对应实验；不能将整体性能写成单模块收益。
任何字段缺少支持时返回 {"text":"","evidence_ids":[]}；rationale 缺证据时 rationale_basis=unknown。
不要把“待确认”“未提供”等占位话术填入有证据的字段，也不要为填满模板编造模块、动机、基线或实验。
missing_information 给出具体缺口。正文使用自然、完整的中文；缩写首次解释，避免空泛评价。
""" + "\n研究背景：\n" + json.dumps(profile, ensure_ascii=False) + "\n讲解材料：\n" + json.dumps(analyses, ensure_ascii=False)


def explanation_review_items(explanation):
    """Review the exact prose later rendered; module names are part of the text."""
    pid = explanation["paper_id"]
    def group_item(field, parts, prefix=""):
        text, scopes, ids = prefix, [], []
        for label, part in parts:
            if text and not text.endswith("\n"):
                text += "\n"
            text += label + "："
            start = len(text)
            text += part["text"]
            scopes.append({"start": start, "end": len(text), "evidence_ids": part["evidence_ids"]})
            ids.extend(part["evidence_ids"])
        return {"field": field, "text": text, "evidence_ids": list(dict.fromkeys(ids)),
                "kind": "fact", "citation_scopes": scopes}

    for group, fields in (("overview", FIELDS[:4]), ("assessment", FIELDS[4:])):
        parts = [(label, explanation[key]) for key, label in fields if explanation[key]["text"].strip()]
        if parts:
            yield group_item(pid + "." + group, parts)
    for index, module in enumerate(explanation["modules"]):
        parts = [(label, module[key]) for key, label in MODULE_FIELDS if key != "rationale" and module[key]["text"].strip()]
        item = group_item("%s.modules.%d" % (pid, index), parts, "模块“%s”\n" % module["name"])
        item["evidence_ids"] = list(dict.fromkeys(item["evidence_ids"] + module["rationale"]["evidence_ids"]))
        yield item
        if module["rationale"]["text"].strip():
            inference = module["rationale_basis"] == "inference"
            yield {"field": "%s.modules.%d.rationale" % (pid, index),
                   "text": "模块“%s”的%s：%s" % (module["name"], "分析推断" if inference else "作者设计理由", module["rationale"]["text"]),
                   "evidence_ids": module["rationale"]["evidence_ids"], "kind": "inference" if inference else "fact"}
    if explanation["missing_information"]:
        yield {"field": pid + ".missing_information", "text": "\n".join(explanation["missing_information"]),
               "evidence_ids": list(dict.fromkeys(eid for _, part in cited_parts(explanation) for eid in part["evidence_ids"])),
               "kind": "proposal"}
