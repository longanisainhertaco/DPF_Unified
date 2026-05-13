"""Promote selected 2026-05-12 user PDF intake records into KR.

This wrapper intentionally limits promotion to this batch and keeps
review-only/non-physics PDFs staged but outside the physics source authority.
Promotion is text extraction only; it does not accept figures, tables, plotted
curves, numeric targets, or validation claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import promote_research_papers_to_kr as promote


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "downloaded_books_papers" / "Research Papers" / "2026-05-12-user-ingest"
DOCS_DIR = ROOT / "docs"
INTAKE_JSON = DOCS_DIR / "USER_PDF_INTAKE_2026_05_12.json"
AUDIT_CSV = DOCS_DIR / "USER_PDF_INTAKE_2026_05_12.csv"
PROMOTION_JSON = DOCS_DIR / "USER_PDF_KR_PROMOTION_2026_05_12.json"
PROMOTION_MD = DOCS_DIR / "USER_PDF_KR_PROMOTION_2026_05_12.md"
CHUNKING_JSON = DOCS_DIR / "USER_PDF_KR_TEXTBOOK_CHUNKING_2026_05_12.json"
CHUNKING_MD = DOCS_DIR / "USER_PDF_KR_TEXTBOOK_CHUNKING_2026_05_12.md"

PROMOTE_RELEVANCE = {
    "promote_to_kr_source_review",
    "promote_to_kr_method_review",
}

# These first-pass classifier misses are still plasma/numerics papers based on
# first-page local text. They are promoted as method/source-review records, not
# as accepted validation targets.
MANUAL_PROMOTE_FILES = {
    "10.1088@1742-6596@370@1@012059.pdf",
    "timofeev2011.pdf",
    "chen2019.pdf",
    "urano2018.pdf",
    "bilbao2006.pdf",
}

MANUAL_STAGE_ONLY_FILES = {
    "apostolou2020.pdf": "out_of_scope_social_science_not_dpf_or_simulation_source",
    "symons1994.pdf": "out_of_scope_jstor_social_science_review_not_dpf_or_simulation_source",
}


def load_intake_records() -> dict[str, dict[str, Any]]:
    payload = json.loads(INTAKE_JSON.read_text())
    records: dict[str, dict[str, Any]] = {}
    for record in payload["records"]:
        destination = str(record.get("destination", ""))
        if not destination:
            continue
        path = ROOT / destination
        if path.is_file():
            records[path.relative_to(BATCH_DIR).as_posix()] = record
    return records


def selected_relpaths(records: dict[str, dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
    selected: set[str] = set()
    excluded: list[dict[str, Any]] = []
    for relpath, record in records.items():
        name = Path(relpath).name
        relevance = str(record.get("relevance", ""))
        subject_class = str(record.get("subject_class", ""))
        if name in MANUAL_STAGE_ONLY_FILES:
            excluded.append(
                {
                    "path": relpath,
                    "title": record.get("title", ""),
                    "subject_class": subject_class,
                    "relevance": relevance,
                    "reason": MANUAL_STAGE_ONLY_FILES[name],
                }
            )
            continue
        if relevance in PROMOTE_RELEVANCE or name in MANUAL_PROMOTE_FILES:
            selected.add(relpath)
            continue
        excluded.append(
            {
                "path": relpath,
                "title": record.get("title", ""),
                "subject_class": subject_class,
                "relevance": relevance,
                "reason": MANUAL_STAGE_ONLY_FILES.get(
                    name,
                    "staged_for_review_not_physics_or_core_simulation_authority",
                ),
            }
        )
    return selected, excluded


def configure_promoter() -> None:
    promote.INTAKE_DIR = BATCH_DIR
    promote.AUDIT_CSV = AUDIT_CSV
    promote.PROMOTION_JSON = PROMOTION_JSON
    promote.PROMOTION_MD = PROMOTION_MD
    promote.CHUNKING_JSON = CHUNKING_JSON
    promote.CHUNKING_MD = CHUNKING_MD


def patch_scan_intake(selected: set[str]) -> None:
    original_scan = promote.scan_intake

    def scan_selected() -> list[promote.IntakeFile]:
        return [item for item in original_scan() if item.relpath in selected]

    promote.scan_intake = scan_selected


def append_exclusion_report(excluded: list[dict[str, Any]]) -> None:
    if not PROMOTION_MD.exists():
        return
    lines = [
        "",
        "## Staged But Not Promoted",
        "",
        "These local PDFs remain in the batch intake folder but were not promoted into KnowledgeReference in this physics/method pass.",
        "",
        "| source | title | class | relevance | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in excluded:
        lines.append(
            "| {path} | {title} | `{cls}` | `{relevance}` | {reason} |".format(
                path=str(item["path"]).replace("|", "\\|"),
                title=str(item["title"]).replace("|", "\\|"),
                cls=str(item["subject_class"]).replace("|", "\\|"),
                relevance=str(item["relevance"]).replace("|", "\\|"),
                reason=str(item["reason"]).replace("|", "\\|"),
            )
        )
    with PROMOTION_MD.open("a") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def write_selected_audit(selected: set[str], excluded: list[dict[str, Any]]) -> None:
    summary_path = DOCS_DIR / "USER_PDF_KR_PROMOTION_SELECTION_2026_05_12.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "decision", "reason"])
        writer.writeheader()
        for relpath in sorted(selected):
            writer.writerow({"path": relpath, "decision": "promote", "reason": "physics_or_method_source_review"})
        for item in excluded:
            writer.writerow({"path": item["path"], "decision": "stage_only", "reason": item["reason"]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write KR records and reports")
    args = parser.parse_args()

    configure_promoter()
    records = load_intake_records()
    selected, excluded = selected_relpaths(records)
    patch_scan_intake(selected)
    result = promote.promotion_run(apply=args.apply)
    result["source_intake_report"] = INTAKE_JSON.relative_to(ROOT).as_posix()
    result["selected_for_promotion_count"] = len(selected)
    result["staged_not_promoted_count"] = len(excluded)
    result["staged_not_promoted"] = excluded
    result["selection_guardrail"] = (
        "The May 12 batch promotion selected DPF/plasma/numerics/math-method "
        "sources for text-parity KR records. Stage-only records are local "
        "intake artifacts and must not be cited as physics authority."
    )
    if args.apply:
        write_selected_audit(selected, excluded)
    promote.write_reports(result, apply=args.apply)
    if args.apply:
        append_exclusion_report(excluded)
    print(
        "selected={selected_for_promotion_count} promoted={promoted_count} "
        "skipped_existing={skipped_existing_count} failed={failed_count} "
        "stage_only={staged_not_promoted_count}".format(**result)
    )
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
