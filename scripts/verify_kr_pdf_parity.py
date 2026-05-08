"""Verify that a KnowledgeReference markdown/JSON pair preserves PDF text.

The check is intentionally local and deterministic:

- PDF page count must match the JSON `page_count`.
- Every PDF page's extracted text must match JSON `pages[].text` after
  normalization.
- Every PDF page's extracted text must be present in the markdown after
  normalization.
- The source PDF SHA-256 is reported for provenance.

This verifies text extraction parity. It does not claim that raster figures have
been digitized; figure data still needs the digitization provenance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    import fitz
except ImportError:  # pragma: no cover
    sys.exit("ERROR: PyMuPDF not installed. Run: pip install pymupdf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\uf0b4", "")
    text = text.replace("\uf0b5", "")
    return re.sub(r"\s+", "", text)


def pdf_pages(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    try:
        return [page.get_text("text").strip() for page in doc]
    finally:
        doc.close()


def verify_pair(pdf_path: Path, md_path: Path, json_path: Path) -> dict[str, object]:
    pages = pdf_pages(pdf_path)
    payload = json.loads(json_path.read_text())
    md_text = md_path.read_text()
    md_normalized = normalize_text(md_text)

    failures: list[str] = []
    json_page_count = int(payload.get("page_count", -1))
    if json_page_count != len(pages):
        failures.append("json_page_count_mismatch")

    json_pages = payload.get("pages", [])
    if not isinstance(json_pages, list) or len(json_pages) != len(pages):
        failures.append("json_pages_missing_or_count_mismatch")
        json_pages = []

    json_text_mismatches: list[int] = []
    md_missing_pages: list[int] = []
    for index, pdf_text in enumerate(pages, start=1):
        normalized_pdf = normalize_text(pdf_text)
        if not normalized_pdf:
            continue
        if len(json_pages) >= index:
            json_entry = json_pages[index - 1]
            json_text = json_entry.get("text", "") if isinstance(json_entry, dict) else ""
            if normalize_text(str(json_text)) != normalized_pdf:
                json_text_mismatches.append(index)
        if normalized_pdf not in md_normalized:
            md_missing_pages.append(index)

    if json_text_mismatches:
        failures.append("json_page_text_mismatch")
    if md_missing_pages:
        failures.append("markdown_missing_pdf_page_text")

    pdf_source_name = payload.get("source_pdf", "")
    if pdf_source_name and str(pdf_source_name) != pdf_path.name:
        failures.append("source_pdf_name_mismatch")

    return {
        "passed": not failures,
        "pdf_path": str(pdf_path),
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "source_pdf_sha256": sha256_file(pdf_path),
        "pdf_pages": len(pages),
        "json_page_count": json_page_count,
        "json_pages": len(json_pages),
        "json_page_text_mismatches": json_text_mismatches,
        "markdown_missing_pages": md_missing_pages,
        "failures": failures,
        "notes": (
            "Text parity only. Figure pixels and plotted curves require the "
            "digitization provenance gate before use as numeric evidence."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--md", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    result = verify_pair(args.pdf, args.md, args.json)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
