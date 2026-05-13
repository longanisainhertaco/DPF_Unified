"""Run source-fidelity checks for the 2026-05-12 promoted PDF batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import promote_research_papers_to_kr as promote
import verify_kr_source_fidelity as fidelity


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "downloaded_books_papers" / "Research Papers" / "2026-05-12-user-ingest"
DOCS_DIR = ROOT / "docs"
PROMOTION_JSON = DOCS_DIR / "USER_PDF_KR_PROMOTION_2026_05_12.json"
AUDIT_CSV = DOCS_DIR / "USER_PDF_INTAKE_2026_05_12.csv"
REPORT_JSON = DOCS_DIR / "USER_PDF_KR_SOURCE_FIDELITY_AUDIT_2026_05_12.json"
REPORT_MD = DOCS_DIR / "USER_PDF_KR_SOURCE_FIDELITY_AUDIT_2026_05_12.md"
APPENDIX_MARKER = "<!-- source-fidelity-review:2026-05-12-user-pdf-batch -->"


def promoted_relpaths() -> set[str]:
    payload = json.loads(PROMOTION_JSON.read_text())
    selected = {str(item["path"]) for item in payload.get("promoted", [])}
    selected.update(str(item["path"]) for item in payload.get("skipped_existing", []))
    return selected


def configure(selected: set[str]) -> None:
    promote.INTAKE_DIR = BATCH_DIR
    promote.AUDIT_CSV = AUDIT_CSV
    fidelity.REPORT_JSON = REPORT_JSON
    fidelity.REPORT_MD = REPORT_MD
    fidelity.APPENDIX_MARKER = APPENDIX_MARKER
    original_scan = promote.scan_intake

    def scan_selected() -> list[promote.IntakeFile]:
        return [item for item in original_scan() if item.relpath in selected]

    promote.scan_intake = scan_selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write KR source-fidelity annotations")
    parser.add_argument(
        "--deep-tables",
        action="store_true",
        help="run slower table extraction on book-length sources too",
    )
    args = parser.parse_args()
    selected = promoted_relpaths()
    configure(selected)
    result = fidelity.run(apply=args.apply, deep_tables=args.deep_tables)
    result["promotion_report"] = PROMOTION_JSON.relative_to(ROOT).as_posix()
    result["batch_selected_count"] = len(selected)
    if args.apply:
        REPORT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    print(
        "selected={batch_selected_count} records={records_checked} updated={records_updated} "
        "recovered_records={records_with_recovered_items} recovered_items={recovered_items_total}".format(**result)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
