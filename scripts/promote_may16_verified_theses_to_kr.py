"""Promote the May 16 user-verified thesis/PDF batch into KnowledgeReference.

This is a scoped source-ingestion utility. It does not accept figures, tables,
plotted curves, numeric targets, simulation outputs, or first-principles claims.
The Saw thesis is a scanned PDF, so its text record is explicitly OCR-derived
from the local OCRmyPDF/Tesseract sidecar while preserving the original PDF
hash as the source payload hash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from promote_research_papers_to_kr import IntakeFile
from promote_research_papers_to_kr import ROOT
from promote_research_papers_to_kr import already_represented
from promote_research_papers_to_kr import extract_pdf
from promote_research_papers_to_kr import load_existing_kr_index
from promote_research_papers_to_kr import normalize_spaces
from promote_research_papers_to_kr import parity_check
from promote_research_papers_to_kr import sha256_file
from promote_research_papers_to_kr import slugify
from promote_research_papers_to_kr import write_kr_pair

INTAKE_DIR = (
    ROOT
    / "downloaded_books_papers"
    / "Research Papers"
    / "2026-05-16-user-validated-theses"
)
PROMOTION_JSON = ROOT / "docs" / "USER_VALIDATED_THESES_KR_PROMOTION_2026_05_16.json"
PROMOTION_MD = ROOT / "docs" / "USER_VALIDATED_THESES_KR_PROMOTION_2026_05_16.md"
SAW_OCR_TEXT = ROOT / "tmp" / "pdfs" / "may16_verified_batch" / "sawsorheoh_ocr.txt"

SOURCES = (
    {
        "filename": "arwinderphdthesis.pdf",
        "title": "Comparative Study of Plasma Focus Machines",
        "status_note": "text_parity_extracted_review_needed",
    },
    {
        "filename": "PhD2012AlirezaTalebitaher.pdf",
        "title": "Coded Aperture Imaging of Nuclear Fusion in the Plasma Focus Device",
        "status_note": "text_parity_extracted_review_needed",
    },
    {
        "filename": "sawsorheoh.pdf",
        "title": "Experimental Studies of a Current-Stepped Z-Pinch",
        "status_note": "ocr_text_extracted_review_needed",
        "ocr_text": SAW_OCR_TEXT,
    },
    {
        "filename": "A SerbanPhD1995.pdf",
        "title": "Anode Geometry and Focus Characteristics",
        "status_note": "text_parity_extracted_review_needed",
    },
    {
        "filename": "MSR PhD thesis.pdf",
        "title": "Compression Dynamics and Radiation Emission from a Deuterium Plasma Focus",
        "status_note": "text_parity_extracted_review_needed",
    },
    {
        "filename": "PhD2010VermaRishi.pdf",
        "title": (
            "Construction and Optimization of Low Energy (<240J) Miniature "
            "Repetitive Plasma Focus Neutron Source"
        ),
        "status_note": "text_parity_extracted_review_needed",
    },
    {
        "filename": "s41598-022-19764-7.pdf",
        "title": (
            "Bayesian inference of spectrometric data and validation with "
            "numerical simulations of plasma sheath diagnostics of a plasma "
            "focus discharge"
        ),
        "status_note": "text_parity_extracted_review_needed",
        "allow_doi": True,
    },
)


def _intake_file(path: Path) -> IntakeFile:
    return IntakeFile(
        path=path,
        relpath=path.relative_to(INTAKE_DIR).as_posix(),
        sha256=sha256_file(path),
        size=path.stat().st_size,
        accession="",
        title_hint="",
        relevance="may16_user_verified_validated_first_principles_source_candidate",
    )


def _ocr_pages(ocr_text: Path, page_count: int) -> list[dict[str, object]]:
    raw_pages = ocr_text.read_text(errors="ignore").split("\f")
    pages = [normalize_spaces(page) for page in raw_pages]
    if pages and not pages[0]:
        pages = pages[1:]
    pages = pages[:page_count]
    while len(pages) < page_count:
        pages.append("")
    return [
        {"page": index, "text": text.strip(), "tables": []}
        for index, text in enumerate(pages, start=1)
    ]


def _apply_ocr_status(markdown: Path, json_path: Path) -> None:
    text = markdown.read_text(errors="ignore")
    text = text.replace(
        "**KR ingestion status:** `text_parity_extracted_review_needed`",
        "**KR ingestion status:** `ocr_text_extracted_review_needed`",
    )
    text = text.replace(
        "Text extraction is available for local source review.",
        "OCR-derived text is available for local source review.",
    )
    markdown.write_text(text)

    payload = json.loads(json_path.read_text())
    ingestion = dict(payload.get("kr_ingestion", {}))
    ingestion["method"] = "OCRmyPDF/Tesseract sidecar split by form-feed page breaks"
    ingestion["status"] = "ocr_text_extracted_review_needed"
    ingestion["notes"] = (
        "The original PDF is scanned. OCR-derived text was ingested for local "
        "source review against the original PDF hash. Figures, tables, plotted "
        "curves, and numeric validation targets are not accepted until "
        "separately reviewed and target-extracted."
    )
    payload["kr_ingestion"] = ingestion
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n")


def _promote_source(
    source: dict[str, object],
    kr_records: list[object],
    *,
    apply: bool,
) -> dict[str, object]:
    path = INTAKE_DIR / str(source["filename"])
    item = _intake_file(path)
    extracted = extract_pdf(path)
    if not bool(source.get("allow_doi", False)):
        extracted = dict(extracted)
        extracted["doi"] = ""
    if source.get("ocr_text"):
        ocr_path = Path(source["ocr_text"])
        extracted = dict(extracted)
        extracted["pages"] = _ocr_pages(ocr_path, int(extracted["page_count"]))
        extracted["nonempty_pages"] = sum(
            1 for page in extracted["pages"] if str(page["text"]).strip()
        )
        extracted["doi"] = ""
    title = str(source["title"])
    represented, reason = already_represented(item, title, str(extracted.get("doi", "")), kr_records)
    if represented:
        return {
            "path": item.relpath,
            "sha256": item.sha256,
            "title": title,
            "status": "skipped_existing",
            "reason": reason,
        }

    slug = f"{slugify(title, path.stem)}-{item.sha256[:8]}"
    md_path, json_path, chunk_paths = write_kr_pair(item, extracted, title, slug, apply=apply)
    parity = {"passed": None, "failures": [], "markdown_missing_pages": []}
    if apply:
        if source.get("ocr_text"):
            _apply_ocr_status(md_path, json_path)
        payload = json.loads(json_path.read_text())
        parity = parity_check(md_path, payload, extracted)
    return {
        "path": item.relpath,
        "sha256": item.sha256,
        "title": title,
        "doi": str(extracted.get("doi", "")),
        "pages": int(extracted["page_count"]),
        "nonempty_pages": int(extracted["nonempty_pages"]),
        "markdown": md_path.relative_to(ROOT).as_posix(),
        "markdown_chunks": [
            chunk_path.relative_to(ROOT).as_posix() for chunk_path in chunk_paths
        ],
        "json": json_path.relative_to(ROOT).as_posix(),
        "parity": parity,
        "status": str(source["status_note"]),
    }


def promote(apply: bool) -> dict[str, object]:
    kr_records = load_existing_kr_index()
    promoted: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for source in SOURCES:
        try:
            record = _promote_source(source, kr_records, apply=apply)
        except Exception as exc:  # pragma: no cover - source-file dependent.
            failed.append(
                {
                    "path": str(source.get("filename", "")),
                    "reason": repr(exc),
                }
            )
            continue
        if record.get("status") == "skipped_existing":
            skipped.append(record)
        else:
            promoted.append(record)

    return {
        "date": "2026-05-16",
        "applied": apply,
        "intake_dir": INTAKE_DIR.relative_to(ROOT).as_posix(),
        "files_scanned": len(SOURCES),
        "promoted_count": len(promoted),
        "skipped_existing_count": len(skipped),
        "failed_count": len(failed),
        "promoted": promoted,
        "skipped_existing": skipped,
        "failed": failed,
        "user_validation_status": "all_seven_user_verified_validated_documents",
        "notes": (
            "Promotion means local PDF/OCR text was extracted into "
            "KnowledgeReference markdown/JSON. It does not accept figures, "
            "tables, plotted curves, numeric targets, simulation outputs, or "
            "first-principles claims."
        ),
    }


def write_reports(result: dict[str, object], apply: bool) -> None:
    if not apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    PROMOTION_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    )
    lines = [
        "# May 16 Verified Theses KR Promotion",
        "",
        "Generated: 2026-05-16",
        "",
        "Source guardrail: this report records local source ingestion only. "
        "The user confirmed the seven documents are verified valid sources. "
        "Text or OCR parity does not accept figures, tables, plotted curves, "
        "numeric targets, simulation outputs, or first-principles claims.",
        "",
        "## Summary",
        "",
        f"- Files scanned: {result['files_scanned']}",
        f"- Promoted into `KnowledgeReference/`: {result['promoted_count']}",
        f"- Skipped because already represented: {result['skipped_existing_count']}",
        f"- Failed or not promoted: {result['failed_count']}",
        "",
        "## Promoted Sources",
        "",
        "| source | title | pages | nonempty pages | sha12 | KR markdown | chunks | KR json | status | parity |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for item in result["promoted"]:
        parity = item.get("parity", {})
        lines.append(
            "| {source} | {title} | {pages} | {nonempty} | {sha12} | {md} | {chunks} | {json} | {status} | {parity} |".format(
                source=item["path"],
                title=str(item["title"]).replace("|", "\\|"),
                pages=item["pages"],
                nonempty=item["nonempty_pages"],
                sha12=str(item["sha256"])[:12],
                md=item["markdown"],
                chunks=len(item.get("markdown_chunks", [])),
                json=item["json"],
                status=item["status"],
                parity=parity.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Existing KR Coverage",
            "",
            "| source | title | sha12 | reason |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in result["skipped_existing"]:
        lines.append(
            "| {source} | {title} | {sha12} | {reason} |".format(
                source=item["path"],
                title=str(item["title"]).replace("|", "\\|"),
                sha12=str(item["sha256"])[:12],
                reason=str(item["reason"]).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Failures / Not Promoted",
            "",
            "| source | reason |",
            "| --- | --- |",
        ]
    )
    for item in result["failed"]:
        lines.append(f"| {item['path']} | {item['reason']} |")
    PROMOTION_MD.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write KR records/reports")
    args = parser.parse_args()
    result = promote(apply=args.apply)
    write_reports(result, apply=args.apply)
    print(
        "files={files_scanned} promoted={promoted_count} "
        "skipped_existing={skipped_existing_count} failed={failed_count}".format(
            **result
        )
    )
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
