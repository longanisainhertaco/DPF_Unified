"""Minimal PyMuPDF-based extractor for KnowledgeReference paired .md + .json output.

Usage:
    python3 extract_papers.py path/to/file.pdf
    python3 extract_papers.py path/to/folder
    python3 extract_papers.py path/to/folder --recursive
    python3 extract_papers.py path/to/folder --rename
    python3 extract_papers.py path/to/file.pdf --out KnowledgeReference/

Schema (matches existing KR pairs as of 2026-04-30):
    JSON: source_pdf, page_count, file_size_bytes, pdf_version, citation,
          toc[level/title/start_page], sections[level/title/start_page/end_page/text],
          table_count
    MD:   H1 title, citation block, full text per section as H2 with page markers
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("ERROR: PyMuPDF not installed. Run: pip install pymupdf")


def slugify(text: str) -> str:
    """Lowercase, replace non-alphanumerics with hyphens, collapse runs."""
    text = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")[:80]


def detect_title(doc: fitz.Document) -> str:
    """Try metadata title; fall back to first non-empty line on page 1."""
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if meta_title and len(meta_title) > 5:
        return meta_title
    if doc.page_count == 0:
        return ""
    page1 = doc[0].get_text("text").strip().splitlines()
    for line in page1:
        line = line.strip()
        if len(line) > 10 and not line.lower().startswith(("doi", "abstract")):
            return line
    return ""


def build_sections(toc: list, doc: fitz.Document) -> list:
    """From TOC, build sections with start/end pages and extracted text.

    Falls back to a single 'Full Text' section if TOC is empty.
    """
    if not toc:
        full_text = "\n\n".join(doc[i].get_text("text") for i in range(doc.page_count))
        return [
            {
                "level": 1,
                "title": "Full Text",
                "start_page": 1,
                "end_page": doc.page_count,
                "text": full_text.strip(),
            }
        ]

    sections = []
    for idx, entry in enumerate(toc):
        level, title, start_page = entry[0], entry[1].strip(), entry[2]
        # End page = next TOC entry start - 1, or last page
        if idx + 1 < len(toc):
            end_page = max(start_page, toc[idx + 1][2] - 1)
        else:
            end_page = doc.page_count
        # Extract text from pages spanning this section (1-indexed -> 0-indexed)
        s_idx = max(0, start_page - 1)
        e_idx = min(doc.page_count, end_page)
        text = "\n\n".join(doc[i].get_text("text") for i in range(s_idx, e_idx))
        sections.append(
            {
                "level": level,
                "title": title,
                "start_page": start_page,
                "end_page": end_page,
                "text": text.strip(),
            }
        )
    return sections


def count_tables(doc: fitz.Document) -> int:
    """Best-effort table count via PyMuPDF's table finder."""
    total = 0
    for page in doc:
        try:
            total += len(page.find_tables().tables)
        except Exception:
            continue
    return total


def extract_paper(pdf_path: Path, out_dir: Path, rename: bool) -> tuple[Path, Path]:
    """Extract one PDF into <stem>.md and <stem>.json under out_dir."""
    doc = fitz.open(pdf_path)
    title = detect_title(doc)
    stem = slugify(title) if (rename and title) else pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    toc = doc.get_toc(simple=True) or []
    sections = build_sections(toc, doc)
    table_count = count_tables(doc)

    payload = {
        "source_pdf": pdf_path.name,
        "page_count": doc.page_count,
        "file_size_bytes": pdf_path.stat().st_size,
        "pdf_version": (doc.metadata or {}).get("format", "").replace("PDF ", "") or None,
        "citation": {
            "title": title or pdf_path.stem,
            "metadata": {k: v for k, v in (doc.metadata or {}).items() if v},
        },
        "table_count": table_count,
        "toc": [
            {"level": lvl, "title": t.strip(), "start_page": p}
            for (lvl, t, p) in toc
        ],
        "sections": [
            {k: v for k, v in s.items() if k != "text"}
            for s in sections
        ],
    }
    # Re-attach text in sections for completeness (matches KR full schema)
    for out_sec, in_sec in zip(payload["sections"], sections):
        out_sec["text"] = in_sec["text"]

    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    md_lines = [
        f"# {title or pdf_path.stem}",
        "",
        f"**Source PDF:** `{pdf_path.name}`  ",
        f"**Pages:** {doc.page_count}  ",
        f"**Tables (auto-detected):** {table_count}  ",
        "",
    ]
    meta = doc.metadata or {}
    if meta.get("author"):
        md_lines.append(f"**Author(s):** {meta['author']}  ")
    if meta.get("subject"):
        md_lines.append(f"**Subject:** {meta['subject']}  ")
    if meta.get("creationDate"):
        md_lines.append(f"**Created:** {meta['creationDate']}  ")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    for sec in sections:
        md_lines.append(f"## {sec['title']}")
        md_lines.append("")
        md_lines.append(f"_pp. {sec['start_page']}–{sec['end_page']}_")
        md_lines.append("")
        md_lines.append(sec["text"])
        md_lines.append("")
    md_path.write_text("\n".join(md_lines))
    doc.close()
    return md_path, json_path


def gather_pdfs(target: Path, recursive: bool) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    if target.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return sorted(target.glob(pattern))
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract PDF -> .md + .json (KR schema).")
    ap.add_argument("path", help="PDF file or directory")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirs")
    ap.add_argument("--rename", action="store_true", help="Rename output to slugified title")
    ap.add_argument("--out", help="Output directory (default: same dir as PDF)")
    args = ap.parse_args(argv)

    target = Path(args.path).expanduser().resolve()
    pdfs = gather_pdfs(target, args.recursive)
    if not pdfs:
        print(f"No PDFs found at {target}", file=sys.stderr)
        return 1

    for pdf in pdfs:
        out_dir = Path(args.out).expanduser().resolve() if args.out else pdf.parent
        try:
            md_path, json_path = extract_paper(pdf, out_dir, args.rename)
            print(f"OK  {pdf.name} -> {md_path.name} + {json_path.name}")
        except Exception as exc:
            print(f"FAIL {pdf.name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
