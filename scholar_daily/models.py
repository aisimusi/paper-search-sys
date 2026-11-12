from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Paper:
    paper_id: str
    source: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: str
    url: str
    pdf_url: str
    categories: List[str] = field(default_factory=list)
    track: str = "latest"
    relevance_score: float = 0.0
    hotness_basis: str = ""
    full_text: str = ""
    full_text_path: str = ""
    pdf_path: str = ""
    tables_text: str = ""
    figure_context: str = ""
    figure_pages: List[str] = field(default_factory=list)
    attention: int = 0
    previously_analyzed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "source": self.source,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "categories": self.categories,
            "track": self.track,
            "relevance_score": round(self.relevance_score, 4),
            "hotness_basis": self.hotness_basis,
            "full_text": self.full_text,
            "full_text_path": self.full_text_path,
            "pdf_path": self.pdf_path,
            "tables_text": self.tables_text,
            "figure_context": self.figure_context,
            "figure_pages": self.figure_pages,
            "attention": self.attention,
            "previously_analyzed": self.previously_analyzed,
        }


@dataclass
class DomainResult:
    name: str
    queries: List[str]
    papers: List[Paper]
    top10: List[Paper]
    top3: List[Paper]
    analyses: List[Dict[str, Any]] = field(default_factory=list)
    ideas: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    paper_explanations: Dict[str, Any] = field(default_factory=dict)
    reading_review: Dict[str, Any] = field(default_factory=dict)
    evidence_records: Dict[str, Any] = field(default_factory=dict)
