from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from .models import Paper


def make_excerpt(text: str, max_chars: int, *, query: str = "") -> str:
    if len(text) <= max_chars:
        return text
    from .paper_context import GAP, HEADINGS, build_index, select_reading_context
    if len(list(HEADINGS.finditer(text))) >= 2:
        metadata = dict(paper_id="", updated="", title="", abstract="", url="")
        texts = {"layout": text}
        index = build_index(metadata, texts)
        ranges = []
        for section in index["sources"]["layout"]["sections"]:
            if section["is_reference"]:
                continue
            if ranges and ranges[-1][1] == section["start"]:
                ranges[-1][1] = section["end"]
            else:
                ranges.append([section["start"], section["end"]])
        body = GAP.join(text[start:end] for start, end in ranges)
        if body and len(body) <= max_chars:
            return body
        selection = select_reading_context(index, texts, max_chars=max_chars,
                                           max_tokens=max_chars, max_fragments=40, query=query)
        if selection["text"]:
            return selection["text"]
    reference = re.search(r"(?mi)^\s*(?:\d+\.?\s+)?references\s*$", text)
    body = text[:reference.start()] if reference and reference.start() > len(text) // 3 else text
    if len(body) <= max_chars:
        return body
    width = max(1, (max_chars - 3 * len(GAP)) // 4)
    starts = [round(i * (len(body) - width) / 3) for i in range(4)]
    return GAP.join(body[start:start + width] for start in starts)


def _extract_pdftotext(pdf_path: Path, text_path: Path, extra_args: list[str]) -> None:
    """Run ``pdftotext`` with the given args, atomically writing the result.

    Raises ``ValueError`` when the PDF yields no usable text so callers can fall
    back (table extraction is best-effort, layout extraction is not).
    """
    temporary = text_path.with_suffix(".tmp")
    try:
        subprocess.run(["pdftotext", *extra_args, str(pdf_path), str(temporary)],
                       check=True, timeout=45, capture_output=True)
        text = temporary.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 200:
            raise ValueError("PDF 没有足够可提取文本")
        os.replace(str(temporary), str(text_path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _extract_tables_text(pdf_path: Path, max_chars: int, max_pages: int = 16, max_rows: int = 400) -> str:
    """Best-effort structured table extraction via ``pdfplumber`` (optional dependency).

    ``pdftotext -layout`` keeps reading order but interleaves table columns; this
    recovers cell-aligned rows so result tables stay readable as anchorable text.
    Returns "" when pdfplumber is unavailable or no table is detected, so callers
    fall back to the layout text only.
    """
    try:
        import pdfplumber
    except ImportError:
        return ""
    rows_out: list[str] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages[:max_pages], 1):
                for table_number, table in enumerate(page.extract_tables() or [], 1):
                    rows_out.append("[Table extraction: PDF page %d, table %d]" % (page_number, table_number))
                    for row in table:
                        cells = ["" if cell is None else re.sub(r"\s+", " ", str(cell)).strip() for cell in row]
                        if any(cells):
                            rows_out.append(" | ".join(cells))
                        if len(rows_out) >= max_rows:
                            return make_excerpt("\n".join(rows_out), max_chars)
                    rows_out.append("")
    except Exception:
        return ""
    return make_excerpt("\n".join(rows_out), max_chars)


def fetch_fulltext(paper: Paper, cache: Path, max_chars: int) -> None:
    """Keep the full local text; send explicitly marked excerpts to the agent.

    Extracts two text views of the PDF: ``-layout`` (reading flow, the anchorable
    evidence source) and ``-table`` (Poppler table analysis). The table text is
    merged as additional anchorable text so result tables stay readable instead of
    being flattened into garbled columns. Figure images are handled separately by
    ``render_pages`` + a vision model, never as text evidence.
    """
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext 不可用，保留为 abstract-only")
    parsed = urllib.parse.urlparse(paper.pdf_url)
    if parsed.scheme not in {"https", "http"} or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}:
        raise ValueError("仅获取 arXiv 的公开 PDF")
    key = hashlib.sha256((paper.paper_id + paper.updated).encode()).hexdigest()[:24]
    folder = cache / "papers"
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path, text_path = folder / (key + ".pdf"), folder / (key + ".txt")
    table_path = folder / (key + ".table.txt")
    if not text_path.exists():
        request = urllib.request.Request(paper.pdf_url.replace("http://", "https://", 1), headers={"User-Agent": "scholar-daily/0.1"})
        with urllib.request.urlopen(request, timeout=40) as response:
            payload = response.read(30 * 1024 * 1024 + 1)
        if len(payload) > 30 * 1024 * 1024 or not payload.startswith(b"%PDF"):
            raise ValueError("PDF 无效或超过 30 MiB")
        pdf_path.write_bytes(payload)
        _extract_pdftotext(pdf_path, text_path, ["-layout"])
    text = text_path.read_text(encoding="utf-8", errors="replace")
    paper.full_text_path = str(text_path)
    paper.pdf_path = str(pdf_path) if pdf_path.exists() else ""
    paper.full_text = make_excerpt(text, max_chars)
    if pdf_path.exists() and not table_path.exists():
        table_text = _extract_tables_text(pdf_path, max(max_chars // 3, 2000))
        if table_text.strip():
            temporary = table_path.with_suffix(".tmp")
            temporary.write_text(table_text, encoding="utf-8")
            os.replace(str(temporary), str(table_path))
    if table_path.exists():
        table_text = table_path.read_text(encoding="utf-8", errors="replace")
        if table_text.strip():
            paper.tables_text = table_text
            paper.full_text = make_excerpt(text + "\n\n[表格文本 · table text]\n" + table_text, max_chars)


def figure_page_numbers(pdf_path: Path, max_pages: int) -> list[int]:
    """Prefer method figures, formulas and experiment tables across the PDF."""
    path = pdf_path.with_suffix(".txt")
    if not path.exists():
        return []
    pages = path.read_text(encoding="utf-8").split("\f")
    candidates = []
    for number, text in enumerate(pages, 1):
        if not text.strip():
            continue
        captions = re.findall(r"(?mi)^\s*(?:Figure|Fig\.|Table)\s+\d+\s*[:.][^\n\f]*", text)
        tables = sum(caption.lstrip().startswith("Table") for caption in captions)
        equations = len(re.findall(r"\(\d+\)\s*$", text, re.M))
        if not captions and not equations:
            continue
        score = tables * 6 + (len(captions) - tables) * 3 + min(equations, 3) * 5
        if re.search(r"ablation|framework|overview|architecture|消融|框架", " ".join(captions), re.I):
            score += 6
        if re.search(r"appendix|supplementary|附录", text, re.I):
            score += 2
        candidates.append((score, number))
    if not candidates:
        return []
    chosen = sorted(candidates, key=lambda row: (-row[0], row[1]))[:max_pages]
    return sorted(number for _, number in chosen)


def render_pages(pdf_path: str, max_pages: int) -> list[str]:
    """Render selected method/result pages, falling back to the first pages.

    Renders once and caches under a directory keyed by ``max_pages`` so re-runs
    reuse the images. Returns absolute paths to the PNG files, oldest page first.
    """
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm 不可用，无法渲染 PDF 页图")
    pdf_path = Path(pdf_path)
    selected = figure_page_numbers(pdf_path, max_pages)
    suffix = "-" + hashlib.sha256(str(selected).encode()).hexdigest()[:10] if selected else ""
    out_dir = pdf_path.with_suffix("") / ("pages-%d%s" % (max_pages, suffix))
    done = out_dir / ".done"
    if done.exists():
        return [str(p) for p in sorted(out_dir.glob("page-*.png"))]
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("page-*.png"):
        stale.unlink()
    prefix = out_dir / "page"
    # Adjacent selections share one Poppler process; page numbers in file names
    # remain the PDF's actual numbers, including selected appendix pages.
    ranges = []
    for number in selected:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    for first, last in ranges or [[1, max_pages]]:
        subprocess.run(["pdftoppm", "-png", "-r", "120", "-f", str(first), "-l", str(last),
                        str(pdf_path), str(prefix)], check=True, timeout=120, capture_output=True)
    done.touch()
    return [str(p) for p in sorted(out_dir.glob("page-*.png"))]
