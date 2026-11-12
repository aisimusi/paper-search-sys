"""Deterministic paper/section/fragment indexes over immutable Unicode text snapshots."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from pathlib import Path

GAP = "\n[EXCERPT GAP: omitted text; not read]\n"
TOKEN_ESTIMATOR = "ceil(UTF-8 bytes / 4); estimate, not provider usage"
LEVELS = ("fragment", "section", "document")
DEFAULT_POLICY = {"planner_mode": "compact", "max_context_tokens": 6000,
                  "max_fragments": 8, "audit_sample_size": 6, "audit_seed": 20261014}
HEADINGS = re.compile(
    r"(?mi)(?:^|(?<=\f))[ \t]*(?:(?:\d+(?:\.\d+)*\.?[ \t]+)([^\n\f]{2,90})|"
    r"(abstract|introduction|background|related work|methods?|methodology|"
    r"experiments?(?: and results)?|results?(?: and discussion)?|evaluation|"
    r"discussion|limitations?|conclusions?|references|bibliography|appendix[^\n\f]*|"
    r"supplementary (?:material|information)|(?-i:[A-Z](?:\.\d+)*\.?[ \t]+[A-Z][^\n\f]{1,80})|"
    r"摘要|引言|相关工作|方法|实验|结果|讨论|局限|结论|参考文献|附录[^\n\f]*))[ \t]*$"
)
MARKERS = re.compile(r"(?m)^\[(?:EXCERPT GAP:[^\n]*|Reference list omitted|表格文本[^\n]*|Table extraction:[^\n]*)\][ \t]*$")
TABLE_PAGE = re.compile(r"(?m)^\[Table extraction: PDF page (\d+), table (\d+)\]$")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimated_tokens(text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / 4)


def checked_policy(value=None):
    policy = dict(DEFAULT_POLICY)
    if value is not None and not isinstance(value, dict):
        raise ValueError("context_policy 必须为对象")
    for key, setting in (value or {}).items():
        if key not in policy:
            raise ValueError("未知上下文配置：" + key)
        if key == "planner_mode":
            if setting not in {"full", "compact"}:
                raise ValueError("planner_mode 应为 full 或 compact")
        elif type(setting) is not int or setting < (0 if key == "audit_seed" else 1):
            raise ValueError("上下文预算和抽样数量必须为正整数，seed 必须为非负整数")
        policy[key] = setting
    if not 256 <= policy["max_context_tokens"] <= 64000:
        raise ValueError("max_context_tokens 应在 256～64000 之间")
    if policy["max_fragments"] > 40 or policy["audit_sample_size"] > 100:
        raise ValueError("max_fragments/audit_sample_size 最多为 40/100")
    return policy


def _whitespace_map(text):
    # Unlike claim extraction, provenance preserves minus signs, hyphens and units.
    chars, starts, ends = [], [], []
    for match in re.finditer(r"\s+|\S", text):
        chars.append(" " if match.group().isspace() else match.group())
        starts.append(match.start())
        ends.append(match.end())
    first = 1 if chars and chars[0] == " " else 0
    last = len(chars) - 1 if chars and chars[-1] == " " else len(chars)
    return "".join(chars[first:last]), starts[first:last], ends[first:last]


def _sections(text, source_id):
    headings = [(m.start(), (m.group(1) or m.group(2)).strip()) for m in HEADINGS.finditer(text)
                if not re.fullmatch(r"[\d\W]+", (m.group(1) or m.group(2)))
                and len((m.group(1) or m.group(2)).split()) <= 14]
    if not headings or headings[0][0] > 0:
        headings.insert(0, (0, "Preamble"))
    rows = []
    in_references = False
    for number, (start, name) in enumerate(headings):
        end = headings[number + 1][0] if number + 1 < len(headings) else len(text)
        if re.match(r"references|bibliography|参考文献", name, re.I):
            in_references = True
        elif re.match(r"appendix|附录|supplementary|(?-i:[A-Z](?:\.\d+)*\.?[ \t]+[A-Z])", name, re.I):
            in_references = False
        rows.append({"section_id": source_id + ":S%d" % number, "name": name,
                     "start": start, "end": end,
                     "is_reference": in_references})
    return rows


def _page(index, offset):
    markers = index["page_markers"]
    if not markers or offset < markers[0][0]:
        return None
    return markers[max(0, bisect.bisect_right([row[0] for row in markers], offset) - 1)][1]


def build_index(paper: dict, texts: dict[str, str], *, page_sources=(), fragment_chars=1600):
    if fragment_chars < 64:
        raise ValueError("fragment_chars 过小")
    sources = {}
    for kind, text in texts.items():
        source_hash = digest(text)
        source_id = kind + ":" + source_hash[:16]
        pages = []
        if kind == "tables":
            pages = [[m.end(), int(m.group(1))] for m in TABLE_PAGE.finditer(text)]
        elif kind in page_sources:
            pages = [[0, 1]] + [[m.end(), i + 2] for i, m in enumerate(re.finditer("\f", text))]
        source = {"source_id": source_id, "kind": kind, "sha256": source_hash,
                  "characters": len(text), "page_markers": pages,
                  "page_basis": "pdf_extraction" if pages else "unknown",
                  "sections": _sections(text, source_id), "fragments": []}
        for section in source["sections"]:
            cuts = [section["start"]]
            for marker in MARKERS.finditer(text, section["start"], section["end"]):
                cuts.extend([marker.start(), marker.end()])
            cuts.append(section["end"])
            for part in range(0, len(cuts) - 1, 2):
                start, end = cuts[part], cuts[part + 1]
                while start < end:
                    stop = min(start + fragment_chars, end)
                    if stop < end:
                        boundary = text.rfind("\n", start + fragment_chars // 2, stop)
                        if boundary > start:
                            stop = boundary + 1
                    left, right = start, stop
                    while left < right and text[left].isspace():
                        left += 1
                    while right > left and text[right - 1].isspace():
                        right -= 1
                    if left < right:
                        source["fragments"].append({"fragment_id": source_id + ":F%d" % len(source["fragments"]),
                            "section_id": section["section_id"], "start": left, "end": right,
                            "page_start": _page(source, left), "page_end": _page(source, right - 1)})
                    start = stop
        sources[kind] = source
    return {"schema_version": 1, "paper_id": paper["paper_id"], "version": paper["updated"],
            "title": paper["title"], "abstract": paper["abstract"], "url": paper["url"],
            "offset_unit": "Unicode character; zero-based [start,end) in source snapshot",
            "sources": sources}


def locate_quote(index, texts, quote):
    needle, _, _ = _whitespace_map(quote)
    if not needle or MARKERS.search(quote):
        return []
    locations = []
    for kind, source in index["sources"].items():
        text = texts[kind]
        if digest(text) != source["sha256"]:
            raise ValueError("原文快照指纹不匹配")
        body, starts, ends = _whitespace_map(text)
        position = body.find(needle)
        while position >= 0:
            start, end = starts[position], ends[position + len(needle) - 1]
            if not MARKERS.search(text[start:end]):
                sections = [s for s in source["sections"] if s["start"] < end and s["end"] > start]
                locations.append({"paper_id": index["paper_id"], "version": index["version"],
                    "source_id": source["source_id"], "source_kind": kind, "source_sha256": source["sha256"],
                    "start": start, "end": end, "page_start": _page(source, start), "page_end": _page(source, end - 1),
                    "sections": [{"section_id": s["section_id"], "name": s["name"]} for s in sections],
                    "fragment_ids": [f["fragment_id"] for f in source["fragments"] if f["start"] < end and f["end"] > start]})
            position = body.find(needle, position + 1)
    return locations


def _terms(query):
    return set(re.findall(r"[a-z0-9_-]{3,}|[\u4e00-\u9fff]{2,}", query.casefold()))


def select_context(index, texts, query, *, level="fragment", max_chars=24000, max_tokens=6000,
                   max_fragments=8, require_match=False):
    if level not in LEVELS:
        raise ValueError("未知上下文层级")
    terms = _terms(query)
    candidates = []
    for kind, source in index["sources"].items():
        if digest(texts[kind]) != source["sha256"]:
            raise ValueError("原文快照指纹不匹配")
        sections = {s["section_id"]: s for s in source["sections"]}
        for f in source["fragments"]:
            section = sections[f["section_id"]]
            if section["is_reference"] and not any(t in {"reference", "references", "bibliography", "参考文献"} for t in terms):
                continue
            content = texts[kind][f["start"]:f["end"]].casefold()
            score = sum(content.count(t) + 3 * section["name"].casefold().count(t) for t in terms)
            candidates.append((score, kind, f, section))
    matched = [row for row in candidates if row[0] > 0]
    if require_match and not matched:
        raise ValueError("缓存全文未找到此主题，请调整 query 或补检索")
    def priority(row):
        name = row[3]["name"].casefold()
        preferred = 0 if re.match(r"abstract|methods?|methodology|experiments?|results?|evaluation|limitations?|摘要|方法|实验|结果|局限", name) else 1
        return (-row[0], preferred if not matched else 0, row[1], row[2]["start"])
    ranked = sorted(matched or candidates, key=priority)
    # Initial fallback samples sections rather than filling the budget with the introduction.
    chosen = []
    if not matched:
        seen_sections = set()
        for row in ranked:
            if row[2]["section_id"] not in seen_sections:
                chosen.append(row)
                seen_sections.add(row[2]["section_id"])
                if len(chosen) == max_fragments:
                    break
        chosen.extend(row for row in ranked if row not in chosen)
    else:
        chosen = ranked
    chosen = chosen[:max_fragments]
    if level != "fragment":
        section_ids = {row[2]["section_id"] for row in chosen}
        chosen = [row for row in candidates if level == "document" or row[2]["section_id"] in section_ids]
        chosen.sort(key=lambda row: (-row[0], row[1], row[2]["start"]))
    # Spend the budget in relevance order, then restore source reading order.
    ranges = []
    used_chars = used_tokens = 0
    for _, kind, fragment, _ in chosen:
        text = texts[kind][fragment["start"]:fragment["end"]]
        overhead = GAP if ranges else ""
        allowance = min(len(text), max_chars - used_chars - len(overhead))
        while allowance > 0 and used_tokens + estimated_tokens(overhead + text[:allowance]) > max_tokens:
            allowance = min(allowance - 1, max(0, (allowance * (max_tokens - used_tokens)) // max(1, estimated_tokens(overhead + text[:allowance]))))
        if allowance <= 0:
            continue
        ranges.append({"source_kind": kind, "source_id": index["sources"][kind]["source_id"],
                       "start": fragment["start"], "end": fragment["start"] + allowance,
                       "fragment_ids": [fragment["fragment_id"]]})
        used_chars += len(overhead) + allowance
        used_tokens += estimated_tokens(overhead + text[:allowance])
    ranges.sort(key=lambda row: (row["source_kind"], row["start"]))
    merged = []
    for row in ranges:
        if merged and row["source_kind"] == merged[-1]["source_kind"] and row["start"] >= merged[-1]["end"] and not texts[row["source_kind"]][merged[-1]["end"]:row["start"]].strip():
            # Merging is optional; keep it only if the source whitespace fits.
            between = texts[row["source_kind"]][merged[-1]["end"]:row["start"]]
            if len(between) <= len(GAP) and estimated_tokens(between) <= estimated_tokens(GAP):
                merged[-1]["end"] = row["end"]
                merged[-1]["fragment_ids"] += row["fragment_ids"]
                continue
        merged.append(dict(row))
    excerpt = GAP.join(texts[r["source_kind"]][r["start"]:r["end"]] for r in merged)
    if len(excerpt) > max_chars or estimated_tokens(excerpt) > max_tokens:
        raise ValueError("上下文选择超过本地字符或估计 Token 预算")
    return {"level": level, "query": query, "matched": bool(matched), "segments": merged,
            "text": excerpt, "characters": len(excerpt), "estimated_tokens": estimated_tokens(excerpt),
            "token_estimator": TOKEN_ESTIMATOR,
            "budget_exhausted": len(ranges) < len(chosen) or any(r["end"] < f["end"] for r in ranges for _, k, f, _ in chosen if k == r["source_kind"] and f["start"] == r["start"])}


def select_reading_context(index, texts, *, max_chars, max_tokens=6000, max_fragments=8, query=""):
    """Reserve space for motivation, method and validation instead of sampling positions."""
    groups = (
        (.2, r"introduction|background|related work|引言|相关工作", "motivation baseline contribution " + query),
        (.5, r"method|framework|architecture|approach|model|方法|框架", "method module design input output " + query),
        (.3, r"experiment|result|evaluation|ablation|limitation|discussion|实验|结果|局限", "ablation evaluation limitation " + query),
    )
    ranges = []
    # Leave room for separators between the independently selected groups.
    char_budget = max(1, max_chars - 2 * len(GAP))
    token_budget = max(1, max_tokens - 2 * estimated_tokens(GAP))
    for weight, pattern, terms in groups:
        subset = dict(index, sources={})
        for kind, source in index["sources"].items():
            ids = {s["section_id"] for s in source["sections"] if not s["is_reference"] and re.search(pattern, s["name"], re.I)}
            subset["sources"][kind] = dict(source, fragments=[f for f in source["fragments"] if f["section_id"] in ids])
        if not any(s["fragments"] for s in subset["sources"].values()):
            subset = index
        selection = select_context(subset, texts, terms, max_chars=int(char_budget * weight),
                                   max_tokens=int(token_budget * weight), max_fragments=max(1, round(max_fragments * weight)))
        ranges.extend(selection["segments"])
    ranges.sort(key=lambda row: (row["source_kind"], row["start"]))
    merged = []
    for row in ranges:
        if merged and row["source_kind"] == merged[-1]["source_kind"] and row["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            merged[-1]["fragment_ids"] = list(dict.fromkeys(merged[-1]["fragment_ids"] + row["fragment_ids"]))
        else:
            merged.append(dict(row))
    text = GAP.join(texts[r["source_kind"]][r["start"]:r["end"]] for r in merged)
    # Every emitted segment is still an exact slice of one source snapshot.
    while merged and (len(text) > max_chars or estimated_tokens(text) > max_tokens):
        merged.pop()
        text = GAP.join(texts[r["source_kind"]][r["start"]:r["end"]] for r in merged)
    return {"level": "fragment", "query": query, "selection_method": "motivation_method_validation",
            "segments": merged, "text": text, "characters": len(text), "estimated_tokens": estimated_tokens(text),
            "token_estimator": TOKEN_ESTIMATOR}


def paper_sources(paper, cache_dir: Path, *, require_cache=False):
    """Only the existing cache/papers boundary may supply a larger source."""
    texts = {"layout": paper.full_text.replace("\r\n", "\n").replace("\r", "\n")}
    page_sources = []
    if paper.full_text_path:
        path = Path(paper.full_text_path).resolve()
        allowed = (cache_dir / "papers").resolve()
        if not path.is_relative_to(allowed) or path.suffix != ".txt":
            if require_cache:
                raise ValueError("主题补读需要 cache/papers 下的完整正文，不能读取其他文件")
        else:
            texts["layout"] = path.read_text(encoding="utf-8")
            page_sources = ["layout"] if paper.pdf_path or "\f" in texts["layout"] else []
    elif require_cache:
        # Supplied local text can also be searched; it has unknown PDF page numbers.
        if not paper.full_text:
            raise ValueError("主题补读需要已取得的正文")
    if paper.tables_text:
        texts["tables"] = paper.tables_text.replace("\r\n", "\n").replace("\r", "\n")
    return texts, page_sources


def load_index(record, *, folder=None):
    from .research_calls import read_json
    path = Path(record["index_path"]).resolve()
    if folder is not None and not path.is_relative_to((folder / "context" / "papers").resolve()):
        raise ValueError("原文索引不在本任务快照目录中")
    index = read_json(path)
    if index.get("schema_version") != 1 or set(index.get("sources", {})) - {"layout", "tables"}:
        raise ValueError("原文索引版本或来源类型不受支持")
    key = digest(json.dumps(index, ensure_ascii=False, sort_keys=True))[:24]
    if record.get("source_fingerprint") != key:
        raise ValueError("原文索引指纹不匹配")
    texts = {}
    for kind, source in index["sources"].items():
        text = (path.parent / (kind + ".txt")).read_text(encoding="utf-8")
        if digest(text) != source["sha256"]:
            raise ValueError("原文快照指纹不匹配")
        texts[kind] = text
    return index, texts


def save_index(folder, paper, texts, page_sources):
    from .research_calls import write_json
    index = build_index(paper.as_dict(), texts, page_sources=page_sources)
    key = digest(json.dumps(index, ensure_ascii=False, sort_keys=True))[:24]
    target = folder / "context" / "papers" / key
    target.mkdir(parents=True, exist_ok=True)
    for kind, text in texts.items():
        path = target / (kind + ".txt")
        if path.exists():
            if digest(path.read_text(encoding="utf-8")) != index["sources"][kind]["sha256"]:
                raise ValueError("已有原文快照被修改")
        else:
            from .pipeline import atomic_text
            atomic_text(path, text)
    write_json(target / "index.json", index)
    return index, {"index_path": str(target / "index.json"), "source_fingerprint": key,
                   "sections": [{"section_id": s["section_id"], "name": s["name"]}
                                for source in index["sources"].values() for s in source["sections"]],
                   "source_characters": sum(len(t) for t in texts.values())}
