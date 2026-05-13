"""Verify and annotate KR records with source-critical PDF artifacts.

This is a second-pass ingestion verifier for the active research-paper intake.
It is not a scientific acceptance tool. It checks that figure/table captions,
formula-like lines, numeric target contexts, and uncertainty contexts are not
silently lost when PDFs are represented as KnowledgeReference Markdown/JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:
    import fitz
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: PyMuPDF is required for source fidelity checks") from exc

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: pdfplumber is required for source fidelity checks") from exc


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
KR_DIR = ROOT / "KnowledgeReference"
DOCS_DIR = ROOT / "docs"
REPORT_JSON = DOCS_DIR / "KR_SOURCE_FIDELITY_AUDIT_2026_05_11.json"
REPORT_MD = DOCS_DIR / "KR_SOURCE_FIDELITY_AUDIT_2026_05_11.md"
APPENDIX_MARKER = "<!-- source-fidelity-review:2026-05-11 -->"

sys.path.insert(0, str(SCRIPTS_DIR))
import promote_research_papers_to_kr as promote  # noqa: E402


FIGURE_RE = re.compile(r"^(?:fig(?:ure)?\.?\s*)\d+[a-z]?(?:[\s:.)-]|$)", re.IGNORECASE)
TABLE_RE = re.compile(r"^(?:table\s*)[ivxlcdm\d]+[a-z]?(?:[\s:.)-]|$)", re.IGNORECASE)
MATH_SYMBOL_RE = re.compile(
    r"(?:=|<=|>=|<|>|\+/-|\\pm|\^|_|\\frac|\\sum|\\int|\\sqrt|"
    r"\b(?:exp|ln|log|sin|cos|tan)\s*\(|"
    r"[\u00b1\u00d7\u2212\u2264\u2265\u2248\u221d\u2211\u222b\u221a"
    r"\u0394\u03b1-\u03c9])"
)
UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*(?:-|--|\u2013|\u2014|to)\s*\d+(?:[.,]\d+)?)?"
    r"\s*(?:"
    r"MA|kA|A|MV|kV|V|MJ|kJ|J|GW|MW|kW|W|"
    r"nH|uH|mH|pF|nF|uF|mF|F|"
    r"Torr|mbar|bar|Pa|atm|"
    r"cm-3|m-3|cm\\^-3|m\\^-3|cm3|m3|g/cm3|kg/m3|"
    r"mm|cm|m|um|\\u00b5m|nm|"
    r"ns|us|\\u00b5s|ms|s|Hz|kHz|MHz|GHz|"
    r"eV|keV|MeV|K|T|mT|G|"
    r"sr|rad|deg|%|percent"
    r")\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"(?:\d+(?:[.,]\d+)?\s*(?:\u00b1|\+/-|\\pm)\s*\d+(?:[.,]\d+)?)|"
    r"(?:\b(?:uncertaint|error|err\.|standard deviation|std\.?|sigma|"
    r"confidence interval|within|accuracy|resolution)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KRPair:
    source_path: Path
    source_rel: str
    sha256: str
    md_path: Path
    json_path: Path


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_match(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = normalize_text(item)
        if not value:
            continue
        key = normalize_match(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def parse_kr_path(reason: str) -> Path | None:
    match = re.search(r"(KnowledgeReference/[^\s,)]+)", reason)
    if not match:
        return None
    path = ROOT / match.group(1)
    if path.suffix.lower() == ".json":
        return path.with_suffix(".md") if path.with_suffix(".md").exists() else path
    if path.suffix.lower() == ".md":
        return path
    return None


def map_active_intake_to_kr() -> list[KRPair]:
    files = promote.scan_intake()
    groups = promote.group_by_sha(files)
    dry_run = promote.promotion_run(False)

    json_by_sha: dict[str, Path] = {}
    for json_path in KR_DIR.glob("*.json"):
        try:
            payload = json.loads(json_path.read_text(errors="ignore"))
        except Exception:
            continue
        sha = str(payload.get("source_pdf_sha256", "")).lower()
        if sha:
            json_by_sha.setdefault(sha, json_path)

    reason_path_by_sha: dict[str, Path] = {}
    for item in dry_run.get("skipped_existing", []):
        sha = str(item.get("sha256", "")).lower()
        path = parse_kr_path(str(item.get("reason", "")))
        if sha and path:
            reason_path_by_sha[sha] = path

    pairs: list[KRPair] = []
    for sha, group in sorted(groups.items(), key=lambda entry: sorted(i.relpath for i in entry[1])[0].lower()):
        canonical = sorted(group, key=promote.canonical_sort_key)[0]
        json_path = json_by_sha.get(sha.lower())
        md_path: Path | None = None
        if json_path:
            md_path = json_path.with_suffix(".md")
        elif sha.lower() in reason_path_by_sha:
            path = reason_path_by_sha[sha.lower()]
            if path.suffix.lower() == ".json":
                json_path = path
                md_path = path.with_suffix(".md")
            else:
                md_path = path
                json_path = path.with_suffix(".json")
        if not json_path or not md_path or not json_path.exists() or not md_path.exists():
            raise RuntimeError(f"Could not resolve KR md/json pair for {canonical.relpath} ({sha})")
        pairs.append(
            KRPair(
                source_path=canonical.path,
                source_rel=canonical.relpath,
                sha256=sha,
                md_path=md_path,
                json_path=json_path,
            )
        )
    return pairs


def fitz_page_lines(page: fitz.Page) -> list[str]:
    lines: list[str] = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
            if normalize_text(text):
                lines.append(text)
    return lines


def collect_caption(lines: list[str], index: int) -> str:
    caption = [lines[index]]
    for follow in lines[index + 1 : index + 5]:
        stripped = normalize_text(follow)
        if not stripped:
            break
        if FIGURE_RE.search(stripped) or TABLE_RE.search(stripped):
            break
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped):
            break
        caption.append(stripped)
        if stripped.endswith("."):
            break
    return normalize_text(" ".join(caption))


def formula_like_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        clean = normalize_text(line)
        if len(clean) < 3:
            continue
        symbol_hits = len(MATH_SYMBOL_RE.findall(clean))
        has_digit = bool(re.search(r"\d", clean))
        has_operator_expression = bool(re.search(r"[A-Za-z0-9]\s*[=<>]\s*[-+A-Za-z0-9(]", clean))
        if symbol_hits >= 2 or (has_digit and symbol_hits >= 1 and has_operator_expression):
            out.append(clean)
    return unique_preserve(out)


def target_context_lines(lines: list[str]) -> list[str]:
    return unique_preserve([line for line in lines if UNIT_RE.search(line)])


def uncertainty_context_lines(lines: list[str]) -> list[str]:
    return unique_preserve([line for line in lines if UNCERTAINTY_RE.search(line)])


def fitz_table_payloads(fitz_page: fitz.Page) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    if not hasattr(fitz_page, "find_tables"):
        return tables
    try:
        found = fitz_page.find_tables()
    except Exception:
        return tables
    for table_index, table in enumerate(getattr(found, "tables", []) or [], start=1):
        try:
            extracted = table.extract()
        except Exception:
            extracted = []
        cleaned_rows: list[list[str]] = []
        for row in extracted or []:
            cleaned = [normalize_text(str(cell or "")) for cell in row]
            if any(cleaned):
                cleaned_rows.append(cleaned)
        if cleaned_rows:
            tables.append({"method": "pymupdf.find_tables", "table_index": table_index, "rows": cleaned_rows})
    return tables


def pdfplumber_table_payloads(pdf_page: Any) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    try:
        extracted = pdf_page.extract_tables() or []
    except Exception:
        extracted = []
    for table_index, table in enumerate(extracted, start=1):
        cleaned_rows: list[list[str]] = []
        for row in table or []:
            cleaned = [normalize_text(str(cell or "")) for cell in row]
            if any(cleaned):
                cleaned_rows.append(cleaned)
        if cleaned_rows:
            tables.append({"method": "pdfplumber.extract_tables", "table_index": table_index, "rows": cleaned_rows})
    return tables


def page_artifacts(source_path: Path, deep_tables: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    artifacts: list[dict[str, Any]] = []
    counts = {
        "figure_captions": 0,
        "table_captions": 0,
        "extracted_tables": 0,
        "formula_like_lines": 0,
        "numeric_target_contexts": 0,
        "uncertainty_contexts": 0,
        "image_blocks": 0,
    }

    doc = fitz.open(source_path)
    try:
        use_deep_tables = deep_tables or doc.page_count < promote.BOOK_PAGE_THRESHOLD
        with pdfplumber.open(source_path) as pdf:
            for page_index, fitz_page in enumerate(doc, start=1):
                plumber_page = pdf.pages[page_index - 1] if page_index - 1 < len(pdf.pages) else None
                lines = fitz_page_lines(fitz_page)
                if plumber_page is not None:
                    plumber_text = plumber_page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                    lines.extend(plumber_text.splitlines())
                lines = unique_preserve(lines)
                figure_captions: list[str] = []
                table_captions: list[str] = []
                for idx, line in enumerate(lines):
                    clean = normalize_text(line)
                    if FIGURE_RE.search(clean):
                        figure_captions.append(collect_caption(lines, idx))
                    elif TABLE_RE.search(clean):
                        table_captions.append(collect_caption(lines, idx))
                tables = fitz_table_payloads(fitz_page) if use_deep_tables else []
                if use_deep_tables and plumber_page is not None:
                    tables.extend(pdfplumber_table_payloads(plumber_page))
                formulas = formula_like_lines(lines)
                targets = target_context_lines(lines)
                uncertainties = uncertainty_context_lines(lines)
                image_blocks = sum(1 for block in fitz_page.get_text("dict").get("blocks", []) if block.get("type") == 1)
                page_record = {
                    "page": page_index,
                    "figure_captions": unique_preserve(figure_captions),
                    "table_captions": unique_preserve(table_captions),
                    "extracted_tables": tables,
                    "formula_like_lines": formulas,
                    "numeric_target_contexts": targets,
                    "uncertainty_contexts": uncertainties,
                    "image_blocks": image_blocks,
                }
                for key in counts:
                    value = page_record[key]
                    counts[key] += int(value) if isinstance(value, int) else len(value)
                if (
                    page_record["figure_captions"]
                    or page_record["table_captions"]
                    or page_record["extracted_tables"]
                    or page_record["formula_like_lines"]
                    or page_record["numeric_target_contexts"]
                    or page_record["uncertainty_contexts"]
                    or page_record["image_blocks"]
                ):
                    artifacts.append(page_record)
    finally:
        doc.close()
    return artifacts, counts


def kr_text_for_pair(pair: KRPair, payload: dict[str, Any]) -> str:
    texts = [strip_existing_appendix(pair.md_path.read_text(errors="ignore"))]
    ingestion = payload.get("kr_ingestion", {})
    if isinstance(ingestion, dict):
        for chunk_ref in ingestion.get("markdown_chunks", []) or []:
            chunk_path = ROOT / str(chunk_ref)
            if chunk_path.exists():
                texts.append(chunk_path.read_text(errors="ignore"))
    texts.append(json.dumps(payload.get("pages", []), ensure_ascii=False))
    return "\n".join(texts)


def missing_from_kr(artifacts: list[dict[str, Any]], kr_text: str) -> list[dict[str, Any]]:
    normalized_kr = normalize_match(kr_text)
    missing: list[dict[str, Any]] = []
    fields = [
        "figure_captions",
        "table_captions",
        "formula_like_lines",
        "numeric_target_contexts",
        "uncertainty_contexts",
    ]
    for page in artifacts:
        page_number = int(page["page"])
        for field in fields:
            for value in page.get(field, []) or []:
                norm = normalize_match(str(value))
                if norm and norm not in normalized_kr:
                    missing.append({"page": page_number, "class": field, "text": normalize_text(str(value))})
        for table in page.get("extracted_tables", []) or []:
            flattened = " ".join(" ".join(row) for row in table.get("rows", []))
            norm = normalize_match(flattened)
            if norm and norm not in normalized_kr:
                missing.append(
                    {
                        "page": page_number,
                        "class": "extracted_tables",
                        "text": normalize_text(flattened),
                    }
                )
    return missing


def strip_existing_appendix(text: str) -> str:
    marker_index = text.find(APPENDIX_MARKER)
    if marker_index == -1:
        return text.rstrip()
    return text[:marker_index].rstrip()


def markdown_appendix(review: dict[str, Any]) -> str:
    counts = review["counts"]
    status = review["status"]
    lines = [
        APPENDIX_MARKER,
        "",
        "## Source-Critical Content Fidelity Review",
        "",
        f"**Review date:** {review['date']}  ",
        f"**Status:** `{status}`  ",
        f"**Figure captions detected:** {counts['figure_captions']}  ",
        f"**Table captions detected:** {counts['table_captions']}  ",
        f"**Extracted table matrices:** {counts['extracted_tables']}  ",
        f"**Formula-like lines detected:** {counts['formula_like_lines']}  ",
        f"**Numeric target contexts detected:** {counts['numeric_target_contexts']}  ",
        f"**Uncertainty contexts detected:** {counts['uncertainty_contexts']}  ",
        f"**PDF image blocks detected:** {counts['image_blocks']}  ",
        "",
        "Detailed per-page figure captions, table matrices, formula-like lines, "
        "numeric target contexts, and uncertainty contexts are stored in the "
        "same-stem JSON file under `source_fidelity_review`. Source PDFs remain "
        "the authority for plotted curves and visual geometry.",
        "",
    ]
    if review["recovered_missing_from_primary_text_count"]:
        lines.extend(
            [
                "### Recovered From Secondary Extraction",
                "",
                "These items were not found in the primary KR page text and were copied into "
                "`source_fidelity_review` from the second-pass extraction.",
                "",
                "| page | class | text |",
                "| ---: | --- | --- |",
            ]
        )
        for item in review["recovered_missing_from_primary_text"][:200]:
            text = str(item["text"]).replace("|", "\\|")
            lines.append(f"| {item['page']} | `{item['class']}` | {text} |")
        extra = review["recovered_missing_from_primary_text_count"] - 200
        if extra > 0:
            lines.append(f"| | | {extra} additional recovered items are stored in JSON. |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def apply_review(pair: KRPair, review: dict[str, Any], apply: bool) -> None:
    payload = json.loads(pair.json_path.read_text(errors="ignore"))
    payload["source_fidelity_review"] = review
    payload["figures"] = [
        {"page": page["page"], "caption": caption}
        for page in review["pages"]
        for caption in page.get("figure_captions", [])
    ]
    payload["table_count"] = int(review["counts"]["extracted_tables"])
    if apply:
        pair.json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
        existing = pair.md_path.read_text(errors="ignore")
        pair.md_path.write_text(strip_existing_appendix(existing) + "\n\n" + markdown_appendix(review))


def review_pair(pair: KRPair, apply: bool, deep_tables: bool) -> dict[str, Any]:
    payload = json.loads(pair.json_path.read_text(errors="ignore"))
    artifacts, counts = page_artifacts(pair.source_path, deep_tables=deep_tables)
    kr_text = kr_text_for_pair(pair, payload)
    missing = missing_from_kr(artifacts, kr_text)
    status = "reviewed_source_critical_content_copied"
    if missing:
        status = "reviewed_with_recovered_secondary_extraction_items"
    review = {
        "date": date.today().isoformat(),
        "status": status,
        "review_assertion": "User stated these intake documents are reviewed; this pass verifies extraction fidelity and copies recovered source-critical artifacts.",
        "methods": [
            "PyMuPDF page.get_text('dict') line extraction",
            "PyMuPDF page image-block count",
            "pdfplumber extract_text secondary line extraction",
            "PyMuPDF find_tables table matrix extraction for article-length sources unless --deep-tables is enabled",
            "pdfplumber extract_tables table matrix extraction for article-length sources unless --deep-tables is enabled",
            "regex classification for figure captions, table captions, formula-like lines, numeric target contexts, and uncertainty contexts",
        ],
        "source_pdf": pair.source_rel,
        "source_pdf_sha256": pair.sha256,
        "knowledge_reference_markdown": pair.md_path.relative_to(ROOT).as_posix(),
        "knowledge_reference_json": pair.json_path.relative_to(ROOT).as_posix(),
        "counts": counts,
        "recovered_missing_from_primary_text_count": len(missing),
        "recovered_missing_from_primary_text": missing,
        "pages": artifacts,
        "notes": (
            "The source PDF remains authoritative for plotted curves, visual geometry, "
            "and figure artwork. This pass copies captions, structured tables when "
            "extractable, formula-like lines, target-value contexts, and uncertainty "
            "contexts into JSON and records a Markdown summary."
        ),
    }
    apply_review(pair, review, apply)
    return {
        "source": pair.source_rel,
        "sha256": pair.sha256,
        "markdown": pair.md_path.relative_to(ROOT).as_posix(),
        "json": pair.json_path.relative_to(ROOT).as_posix(),
        "status": status,
        "counts": counts,
        "recovered_missing_from_primary_text_count": len(missing),
    }


def write_reports(result: dict[str, Any], apply: bool) -> None:
    if apply:
        REPORT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    lines = [
        "# KR Source Fidelity Audit",
        "",
        f"Generated: {result['date']}",
        "",
        "Guardrail: this audit checks source-copy fidelity only. It does not accept "
        "figures, tables, formulas, targets, uncertainty values, plotted curves, "
        "or scientific validation claims.",
        "",
        "## Summary",
        "",
        f"- Intake records checked: {result['records_checked']}",
        f"- Records updated: {result['records_updated']}",
        f"- Records with recovered secondary-extraction items: {result['records_with_recovered_items']}",
        f"- Recovered secondary-extraction items: {result['recovered_items_total']}",
        f"- Figure captions detected: {result['totals']['figure_captions']}",
        f"- Table captions detected: {result['totals']['table_captions']}",
        f"- Extracted table matrices: {result['totals']['extracted_tables']}",
        f"- Formula-like lines detected: {result['totals']['formula_like_lines']}",
        f"- Numeric target contexts detected: {result['totals']['numeric_target_contexts']}",
        f"- Uncertainty contexts detected: {result['totals']['uncertainty_contexts']}",
        f"- PDF image blocks detected: {result['totals']['image_blocks']}",
        "",
        "## Records",
        "",
        "| source | status | figures | table captions | tables | formulas | targets | uncertainties | recovered | KR json |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in result["records"]:
        counts = record["counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record['source']}`",
                    f"`{record['status']}`",
                    str(counts["figure_captions"]),
                    str(counts["table_captions"]),
                    str(counts["extracted_tables"]),
                    str(counts["formula_like_lines"]),
                    str(counts["numeric_target_contexts"]),
                    str(counts["uncertainty_contexts"]),
                    str(record["recovered_missing_from_primary_text_count"]),
                    f"`{record['json']}`",
                ]
            )
            + " |"
        )
    lines.append("")
    if apply:
        REPORT_MD.write_text("\n".join(lines).rstrip() + "\n")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False))


def run(apply: bool, deep_tables: bool) -> dict[str, Any]:
    pairs = map_active_intake_to_kr()
    records: list[dict[str, Any]] = []
    totals = {
        "figure_captions": 0,
        "table_captions": 0,
        "extracted_tables": 0,
        "formula_like_lines": 0,
        "numeric_target_contexts": 0,
        "uncertainty_contexts": 0,
        "image_blocks": 0,
    }
    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {pair.source_rel}", file=sys.stderr)
        record = review_pair(pair, apply=apply, deep_tables=deep_tables)
        records.append(record)
        for key in totals:
            totals[key] += int(record["counts"].get(key, 0))
    result = {
        "date": date.today().isoformat(),
        "applied": apply,
        "deep_tables": deep_tables,
        "records_checked": len(records),
        "records_updated": len(records) if apply else 0,
        "records_with_recovered_items": sum(1 for r in records if r["recovered_missing_from_primary_text_count"]),
        "recovered_items_total": sum(int(r["recovered_missing_from_primary_text_count"]) for r in records),
        "totals": totals,
        "records": records,
    }
    write_reports(result, apply=apply)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write KR JSON/Markdown annotations and reports")
    parser.add_argument(
        "--deep-tables",
        action="store_true",
        help="run slower pdfplumber table extraction on book-length sources too",
    )
    args = parser.parse_args()
    result = run(apply=args.apply, deep_tables=args.deep_tables)
    print(
        "records={records_checked} updated={records_updated} "
        "recovered_records={records_with_recovered_items} recovered_items={recovered_items_total}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
