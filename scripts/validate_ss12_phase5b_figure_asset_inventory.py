#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "docs/SS12_P1_PHASE5B_FIGURE_ASSET_INVENTORY_2026_05_22.json"
DEFAULT_PHASE5_MANIFEST = ROOT / "docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json"

TOP_LEVEL_ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_asset_claim",
    "accepted_digitization_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)
ROW_ACCEPTANCE_FLAGS: tuple[str, ...] = TOP_LEVEL_ACCEPTANCE_FLAGS
REQUIRED_TOP_LEVEL: tuple[str, ...] = (
    "inventory_id",
    "generated_at",
    "validation_scope",
    "phase5_source_manifest",
    "acceptance_boundary",
    "figure_assets",
)
REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "id",
    "figure_source_id",
    "source_kind",
    "source_pdf_path",
    "source_pdf_sha256",
    "page",
    "figure_id",
    "figure_caption_hint",
    "asset_status",
    "region_status",
    "extraction_packet_status",
    "digitization_hash",
    "accepted_asset_claim",
    "accepted_digitization_claim",
    "accepted_runtime_claim",
    "promotes_acceptance",
    "can_support_first_principles_acceptance",
    "blocked_reason",
)
ALLOWED_SOURCE_KINDS = {"repo_pdf", "external_pdf"}
ALLOWED_ASSET_STATUS = {"asset_located_not_extracted"}
ALLOWED_REGION_STATUS = {"region_hint_only"}
ALLOWED_EXTRACTION_STATUS = {"not_extracted", "extracted"}
ALLOWED_EXTERNAL_PDF_ROOTS: tuple[Path, ...] = (Path("/Users/anthonyzamora/PDFs").resolve(),)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _issue(rule: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"rule": rule, "message": message, **detail}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_inventory(inventory: dict[str, Any], repo_root: Path = ROOT) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for field in REQUIRED_TOP_LEVEL:
        if field not in inventory:
            issues.append(_issue("missing_top_level_field", "required top-level field missing", field=field))

    boundary = inventory.get("acceptance_boundary", {})
    if not isinstance(boundary, dict):
        issues.append(_issue("acceptance_boundary_not_object", "acceptance_boundary must be an object"))
        boundary = {}
    for flag in TOP_LEVEL_ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            issues.append(
                _issue(
                    "top_level_acceptance_flag_not_false",
                    f"acceptance_boundary {flag} must be false",
                    flag=flag,
                    value=boundary.get(flag),
                )
            )

    rows = inventory.get("figure_assets")
    if not isinstance(rows, list):
        issues.append(_issue("figure_assets_not_list", "figure_assets must be a list"))
        return issues

    phase5_raw = Path(str(inventory.get("phase5_source_manifest", DEFAULT_PHASE5_MANIFEST)))
    phase5_path = phase5_raw if phase5_raw.is_absolute() else repo_root / phase5_raw
    phase5_path = phase5_path.resolve()
    if not _is_relative_to(phase5_path, repo_root):
        issues.append(
            _issue(
                "phase5_source_manifest_outside_repo",
                "phase5 source manifest path must stay inside the repository",
                path=str(phase5_path),
            )
        )
        source_ids: set[str] = set()
    elif not phase5_path.exists():
        issues.append(
            _issue(
                "phase5_source_manifest_missing",
                "phase5 source manifest referenced by asset inventory does not exist",
                path=str(phase5_path),
            )
        )
        source_ids = set()
    else:
        source_manifest = _load_json(phase5_path)
        source_ids = {str(row.get("id")) for row in source_manifest.get("figure_sources", [])}

    asset_source_ids = {str(row.get("figure_source_id")) for row in rows if isinstance(row, dict)}
    if source_ids and asset_source_ids != source_ids:
        issues.append(
            _issue(
                "asset_ids_do_not_match_phase5_manifest",
                "figure asset rows must map exactly one-to-one with Phase 5 figure source manifest rows",
                expected=sorted(source_ids),
                actual=sorted(asset_source_ids),
            )
        )

    seen_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(_issue("figure_asset_not_object", "figure asset row must be an object", index=index))
            continue
        _validate_row(row, index, issues, repo_root)
        row_id = str(row.get("id", f"row_{index}"))
        source_id = str(row.get("figure_source_id", ""))
        if row_id in seen_ids:
            issues.append(_issue("duplicate_asset_id", "duplicate asset id", row_id=row_id))
        seen_ids.add(row_id)
        if source_id in seen_source_ids:
            issues.append(_issue("duplicate_figure_source_id", "duplicate figure_source_id", figure_source_id=source_id))
        seen_source_ids.add(source_id)

    return issues


def _validate_row(row: dict[str, Any], index: int, issues: list[dict[str, Any]], repo_root: Path) -> None:
    row_id = str(row.get("id", f"row_{index}"))
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            issues.append(_issue("missing_asset_row_field", "required asset row field missing", row_id=row_id, field=field))

    for flag in ROW_ACCEPTANCE_FLAGS:
        if row.get(flag) is not False:
            issues.append(
                _issue(
                    "asset_acceptance_flag_not_false",
                    f"figure asset {flag} must be false",
                    row_id=row_id,
                    flag=flag,
                    value=row.get(flag),
                )
            )

    if row.get("source_kind") not in ALLOWED_SOURCE_KINDS:
        issues.append(_issue("invalid_source_kind", "source_kind is invalid", row_id=row_id, value=row.get("source_kind")))
    if row.get("asset_status") not in ALLOWED_ASSET_STATUS:
        issues.append(_issue("invalid_asset_status", "asset_status is invalid", row_id=row_id, value=row.get("asset_status")))
    if row.get("region_status") not in ALLOWED_REGION_STATUS:
        issues.append(_issue("invalid_region_status", "region_status is invalid", row_id=row_id, value=row.get("region_status")))
    if row.get("extraction_packet_status") not in ALLOWED_EXTRACTION_STATUS:
        issues.append(
            _issue(
                "invalid_extraction_packet_status",
                "extraction_packet_status is invalid",
                row_id=row_id,
                value=row.get("extraction_packet_status"),
            )
        )

    page = row.get("page")
    if not isinstance(page, int) or page < 1:
        issues.append(_issue("invalid_page", "page must be a positive integer", row_id=row_id, value=page))

    declared_pdf_path = Path(str(row.get("source_pdf_path", ""))).expanduser()
    if not declared_pdf_path.is_absolute():
        declared_pdf_path = repo_root / declared_pdf_path
    declared_pdf_path = Path(os.path.abspath(declared_pdf_path))
    pdf_path = declared_pdf_path.resolve()
    source_kind = row.get("source_kind")
    resolved_external_allowed = any(_is_relative_to(pdf_path, root) for root in ALLOWED_EXTERNAL_PDF_ROOTS)
    allowed_pdf_root = False
    if source_kind == "repo_pdf":
        # repo_pdf may be a repo-contained symlink into an explicitly allowed external PDF vault.
        allowed_pdf_root = _is_relative_to(declared_pdf_path, repo_root) and (
            _is_relative_to(pdf_path, repo_root) or resolved_external_allowed
        )
    elif source_kind == "external_pdf":
        allowed_pdf_root = resolved_external_allowed
    if not allowed_pdf_root:
        issues.append(
            _issue(
                "source_pdf_outside_allowed_roots",
                "source PDF path is outside allowed roots for its source_kind",
                row_id=row_id,
                source_kind=source_kind,
                path=str(pdf_path),
                allowed_external_roots=[str(root) for root in ALLOWED_EXTERNAL_PDF_ROOTS],
            )
        )
    if not pdf_path.exists():
        issues.append(_issue("source_pdf_missing", "source PDF does not exist", row_id=row_id, path=str(pdf_path)))
    elif not pdf_path.is_file():
        issues.append(_issue("source_pdf_not_file", "source PDF path is not a file", row_id=row_id, path=str(pdf_path)))
    else:
        expected_sha = row.get("source_pdf_sha256")
        actual_sha = _sha256(pdf_path)
        if expected_sha != actual_sha:
            issues.append(
                _issue(
                    "source_pdf_sha256_mismatch",
                    "source PDF SHA-256 does not match inventory",
                    row_id=row_id,
                    expected=expected_sha,
                    actual=actual_sha,
                )
            )

    if row.get("extraction_packet_status") == "extracted" and not row.get("digitization_hash"):
        issues.append(
            _issue(
                "digitization_hash_required_for_extracted_packet",
                "extracted packets require a digitization_hash",
                row_id=row_id,
            )
        )
    if row.get("extraction_packet_status") == "not_extracted" and row.get("digitization_hash") is not None:
        issues.append(
            _issue(
                "digitization_hash_for_unextracted_packet_forbidden",
                "not_extracted packets must not carry digitization_hash",
                row_id=row_id,
                value=row.get("digitization_hash"),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SS12 Phase 5-B figure asset inventory")
    parser.add_argument("inventory", nargs="?", default=str(DEFAULT_INVENTORY))
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    inventory = _load_json(inventory_path)
    issues = validate_inventory(inventory, ROOT)
    report = {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "inventory": str(inventory_path),
    }
    print(json.dumps(report, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
