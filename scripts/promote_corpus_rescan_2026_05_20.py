"""Promote the 2026-05-20 PDF corpus-rescan P0 candidates into KR.

This is a scoped source-ingestion utility. It promotes only the two P0 raw-PDF
candidates identified in ``docs/FIRST_PRINCIPLES_PDF_CORPUS_RESCAN_2026_05_20.md``.
Promotion means local PDF text becomes searchable KnowledgeReference material;
it does not accept figures, tables, plotted curves, numeric targets, runtime
closures, or validation claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from promote_research_papers_to_kr import (
    ROOT,
    IntakeFile,
    already_represented,
    extract_pdf,
    load_existing_kr_index,
    parity_check,
    sha256_file,
    slugify,
    write_kr_pair,
)

INTAKE_DIR = (
    ROOT
    / "downloaded_books_papers"
    / "Research Papers"
    / "2026-05-20-corpus-rescan"
)
PROMOTION_JSON = ROOT / "docs" / "CORPUS_RESCAN_KR_PROMOTION_2026_05_20.json"
PROMOTION_MD = ROOT / "docs" / "CORPUS_RESCAN_KR_PROMOTION_2026_05_20.md"

SOURCES: tuple[dict[str, str], ...] = (
    {
        "filename": "plasma-04-00033.pdf",
        "title": "Update on the Scientific Status of the Plasma Focus",
        "status": "text_parity_extracted_review_needed",
        "priority": "P0",
        "scope": "review_source_dpf_phenomenology_models_diagnostics",
        "rescan_report": "docs/FIRST_PRINCIPLES_PDF_CORPUS_RESCAN_2026_05_20.md",
    },
    {
        "filename": "bernard1977.pdf",
        "title": "The Dense Plasma Focus - A High Intensity Neutron Source",
        "status": "text_parity_extracted_review_needed",
        "priority": "P0",
        "scope": "neutron_source_diagnostics_and_mechanism_review",
        "rescan_report": "docs/FIRST_PRINCIPLES_PDF_CORPUS_RESCAN_2026_05_20.md",
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
        relevance="corpus_rescan_p0_source_review_candidate",
    )


def _promote_source(
    source: dict[str, str],
    kr_records: list[Any],
    *,
    apply: bool,
) -> dict[str, Any]:
    path = INTAKE_DIR / source["filename"]
    item = _intake_file(path)
    extracted = extract_pdf(path)
    title = source["title"]

    represented, reason = already_represented(
        item,
        title,
        str(extracted.get("doi", "")),
        kr_records,
    )
    if represented:
        return {
            "path": item.relpath,
            "sha256": item.sha256,
            "title": title,
            "doi": str(extracted.get("doi", "")),
            "status": "skipped_existing",
            "reason": reason,
        }

    slug = f"{slugify(title, path.stem)}-{item.sha256[:8]}"
    md_path, json_path, chunk_paths = write_kr_pair(
        item,
        extracted,
        title,
        slug,
        apply=apply,
    )
    parity = {"passed": None, "failures": [], "markdown_missing_pages": []}
    if apply:
        payload = json.loads(json_path.read_text())
        ingestion = dict(payload.get("kr_ingestion", {}))
        ingestion["source"] = INTAKE_DIR.relative_to(ROOT).as_posix()
        ingestion["status"] = source["status"]
        ingestion["validation_status"] = "source_available_not_target_extracted"
        ingestion["priority"] = source["priority"]
        ingestion["scope"] = source["scope"]
        ingestion["promotion_report"] = PROMOTION_MD.relative_to(ROOT).as_posix()
        ingestion["rescan_report"] = source["rescan_report"]
        ingestion["notes"] = (
            "Promoted from the 2026-05-20 local PDF corpus rescan for "
            "KnowledgeReference search and source review. Figures, tables, "
            "plotted curves, numeric validation targets, runtime closures, and "
            "first-principles claims are not accepted until separately "
            "reviewed and target-extracted."
        )
        payload["kr_ingestion"] = ingestion
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
            + "\n"
        )
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
        "priority": source["priority"],
        "scope": source["scope"],
        "status": source["status"],
        "parity": parity,
    }


def promote(apply: bool) -> dict[str, Any]:
    kr_records = load_existing_kr_index()
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for source in SOURCES:
        try:
            record = _promote_source(source, kr_records, apply=apply)
        except Exception as exc:  # pragma: no cover - source-file dependent.
            failed.append({"path": source["filename"], "reason": repr(exc)})
            continue
        if record.get("status") == "skipped_existing":
            skipped.append(record)
        else:
            promoted.append(record)

    return {
        "date": "2026-05-20",
        "applied": apply,
        "intake_dir": INTAKE_DIR.relative_to(ROOT).as_posix(),
        "rescan_report": "docs/FIRST_PRINCIPLES_PDF_CORPUS_RESCAN_2026_05_20.md",
        "files_scanned": len(SOURCES),
        "promoted_count": len(promoted),
        "skipped_existing_count": len(skipped),
        "failed_count": len(failed),
        "promoted": promoted,
        "skipped_existing": skipped,
        "failed": failed,
        "guardrail": (
            "Promotion is local KnowledgeReference ingestion only. Raw PDFs and "
            "text-parity KR records remain source_available_not_target_extracted "
            "until typed target extraction and review are complete."
        ),
    }


def write_reports(result: dict[str, Any], apply: bool) -> None:
    if not apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    PROMOTION_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    )
    lines = [
        "# Corpus Rescan KR Promotion",
        "",
        "Generated: 2026-05-20",
        "",
        "Source guardrail: this report records local source ingestion only. Text "
        "parity does not accept figures, tables, plotted curves, numeric targets, "
        "runtime closures, simulation outputs, or whole-shot first-principles claims.",
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
    parser.add_argument("--apply", action="store_true", help="write KR records and reports")
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
