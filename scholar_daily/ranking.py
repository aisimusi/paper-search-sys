from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Iterable, List

from .models import Paper


def _tokens(query: str) -> List[str]:
    query = re.sub(r"\b(?:all|ti|abs|au|cat):", "", query, flags=re.IGNORECASE)
    return [token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", query)
            if len(token) > 2 and token.upper() not in {"AND", "OR", "NOT"}]


def relevance(paper: Paper, queries: Iterable[str]) -> float:
    tokens = {token for query in queries for token in _tokens(query)}
    text = (paper.title + " " + paper.abstract).lower()
    if not tokens:
        return 0.5
    matched = sum(1 for token in tokens if token in text)
    title_matched = sum(1 for token in tokens if token in paper.title.lower())
    return min(1.0, (matched / len(tokens)) * 0.7 + (title_matched / len(tokens)) * 0.3)


def _age_days(value: str) -> float:
    try:
        text = value.replace("Z", "+00:00")
        date = datetime.fromisoformat(text).astimezone(timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - date).total_seconds() / 86400)
    except ValueError:
        return 9999.0


def rank_papers(papers: Iterable[Paper], queries: List[str], limit: int = 10) -> List[Paper]:
    ranked = list(papers)
    for paper in ranked:
        rel = relevance(paper, queries)
        age = _age_days(paper.updated or paper.published)
        freshness = math.exp(-age / (30.0 if "hot" in paper.track else 7.0))
        # A popular but unrelated paper never displaces a topical one solely on votes.
        attention = min(1.0, math.log1p(paper.attention) / math.log(101))
        paper.relevance_score = rel * 0.65 + freshness * 0.25 + attention * 0.1
    ranked.sort(key=lambda paper: (paper.relevance_score, paper.updated, paper.title), reverse=True)
    return ranked[:limit]
