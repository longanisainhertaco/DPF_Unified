"""Repair Kortanek 2014 source-fidelity annotations after May 12 validation."""

from __future__ import annotations

import json
from pathlib import Path

import verify_kr_source_fidelity as fidelity


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "downloaded_books_papers" / "Research Papers" / "2026-05-11-user-ingest" / "kortanek2014.pdf"
MD = ROOT / "KnowledgeReference" / "this-content-has-been-downloaded-from-iopscience-please-scroll-down-to-see-the-full-text-7dbd9199.md"
JSON = MD.with_suffix(".json")


def main() -> int:
    payload = json.loads(JSON.read_text())
    sha = str(payload["source_pdf_sha256"])
    actual = fidelity.promote.sha256_file(SOURCE)
    if actual != sha:
        raise SystemExit(f"source hash mismatch for Kortanek repair: {actual} != {sha}")
    pair = fidelity.KRPair(
        source_path=SOURCE,
        source_rel=SOURCE.relative_to(fidelity.promote.INTAKE_DIR).as_posix(),
        sha256=sha,
        md_path=MD,
        json_path=JSON,
    )
    record = fidelity.review_pair(pair, apply=True, deep_tables=False)
    print(
        "repaired={source} recovered={recovered_missing_from_primary_text_count}".format(
            **record
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
