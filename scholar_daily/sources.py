from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List

from .models import Paper

ATOM = "http://www.w3.org/2005/Atom"
NS = {"a": ATOM}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _parse_time(value: str) -> datetime:
    text = (value or "").replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _paper_id(entry_id: str, title: str) -> str:
    if entry_id:
        return re.sub(r"v\d+$", "", entry_id.rstrip("/").split("/abs/")[-1])
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]


def parse_arxiv_xml(payload: bytes, track: str = "latest") -> List[Paper]:
    root = ET.fromstring(payload)
    papers: List[Paper] = []
    for entry in root.findall("a:entry", NS):
        if entry.findtext("a:id", default="", namespaces=NS).startswith("http://arxiv.org/api/errors"):
            raise ValueError("arXiv API 返回查询错误：" + entry.findtext("a:summary", default="", namespaces=NS))
        title = _clean(entry.findtext("a:title", default="", namespaces=NS))
        if not title:
            continue
        entry_id = _clean(entry.findtext("a:id", default="", namespaces=NS))
        links = entry.findall("a:link", NS)
        abs_url = next((link.attrib.get("href", "") for link in links if link.attrib.get("type") == "text/html"), entry_id)
        pdf_url = next((link.attrib.get("href", "") for link in links if link.attrib.get("title") == "pdf"), "")
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"
        authors = [_clean(node.findtext("a:name", default="", namespaces=NS)) for node in entry.findall("a:author", NS)]
        categories = [node.attrib.get("term", "") for node in entry.findall("a:category", NS)]
        published = _clean(entry.findtext("a:published", default="", namespaces=NS))
        updated = _clean(entry.findtext("a:updated", default=published, namespaces=NS))
        papers.append(Paper(
            paper_id=_paper_id(entry_id, title), source="arxiv", title=title,
            authors=[author for author in authors if author],
            abstract=_clean(entry.findtext("a:summary", default="", namespaces=NS)),
            published=published, updated=updated, url=abs_url, pdf_url=pdf_url,
            categories=categories, track=track,
            hotness_basis=("按最近更新排序；arXiv API 本身不提供引用/热度信号" if track == "hot" else "按提交时间排序"),
        ))
    return papers


class ArxivSource:
    endpoint = "https://export.arxiv.org/api/query"
    last_request = 0.0

    def search(self, query: str, categories: Iterable[str], days: int, max_results: int, track: str) -> List[Paper]:
        category_query = " OR ".join("cat:%s" % cat for cat in categories if cat)
        # Search each phrase as a phrase; advanced arXiv field queries are explicit.
        all_query = query if re.search(r"\b(?:all|ti|abs|au|cat):", query) else 'all:"%s"' % query.replace('"', '')
        search = "(%s) AND %s" % (category_query, all_query) if category_query else all_query
        params = {
            "search_query": search,
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = self.endpoint + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        request = urllib.request.Request(url, headers={"User-Agent": "scholar-daily/0.1"})
        time.sleep(max(0, 3.1 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        with urllib.request.urlopen(request, timeout=45) as response:
            papers = parse_arxiv_xml(response.read(), track=track)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        date_key = "updated" if track == "hot" else "published"
        now = datetime.now(timezone.utc)
        return [paper for paper in papers if cutoff <= _parse_time(getattr(paper, date_key)) <= now]


class HuggingFaceSource:
    """Community attention, explicitly separate from scientific quality."""

    def search(self, days: int, max_results: int = 100) -> List[Paper]:
        url = "https://huggingface.co/api/daily_papers?" + urllib.parse.urlencode({"sort": "trending", "limit": max_results})
        request = urllib.request.Request(url, headers={"User-Agent": "scholar-daily/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        return self.parse(payload, days)

    @staticmethod
    def parse(payload: list, days: int) -> List[Paper]:
        papers = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        for row in payload:
            item = row.get("paper", {})
            pid = item.get("id", "")
            if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", pid):
                continue
            published = item.get("publishedAt", "")
            if not cutoff <= _parse_time(published) <= now:
                continue
            votes = max(0, int(item.get("upvotes", 0) or 0))
            papers.append(Paper(
                paper_id=_paper_id(pid, ""), source="arxiv", title=_clean(item.get("title", "")),
                authors=[a["name"] for a in item.get("authors", []) if a.get("name")],
                abstract=_clean(item.get("summary", "")), published=published, updated=published,
                url="https://arxiv.org/abs/" + pid, pdf_url="https://arxiv.org/pdf/" + pid,
                track="hot", attention=votes,
                hotness_basis="Hugging Face trending 列表；社区 upvotes=%d；抓取=%s；不是引用数或质量评分" % (votes, now.isoformat()),
            ))
        return papers


def deduplicate_papers(papers: Iterable[Paper]) -> List[Paper]:
    merged: Dict[str, Paper] = {}
    for paper in papers:
        if paper.paper_id not in merged:
            merged[paper.paper_id] = paper
        else:
            current = merged[paper.paper_id]
            if current.track != paper.track:
                current.track = "latest+hot"
            if paper.updated > current.updated:
                current.updated = paper.updated
            if paper.attention > current.attention or paper.track == "hot":
                current.attention = paper.attention
                current.hotness_basis = paper.hotness_basis
    return list(merged.values())
