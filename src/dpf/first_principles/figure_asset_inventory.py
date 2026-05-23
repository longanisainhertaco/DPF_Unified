"""Phase 5-B figure asset inventory and extraction packet planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.validate_ss12_phase5b_figure_asset_inventory import validate_inventory

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHASE5B_FIGURE_ASSET_INVENTORY_PATH = (
    ROOT / "docs/SS12_P1_PHASE5B_FIGURE_ASSET_INVENTORY_2026_05_22.json"
)


def load_phase5b_figure_asset_inventory(
    path: str | Path = DEFAULT_PHASE5B_FIGURE_ASSET_INVENTORY_PATH,
) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def build_phase5b_extraction_packet_plan(
    *,
    inventory_path: str | Path = DEFAULT_PHASE5B_FIGURE_ASSET_INVENTORY_PATH,
) -> dict[str, Any]:
    path = Path(inventory_path)
    if not path.exists():
        return _blocked_plan(
            "blocked_phase5b_asset_inventory_missing",
            "phase5b_asset_inventory_missing",
        )

    inventory = load_phase5b_figure_asset_inventory(path)
    validation_issues = validate_inventory(inventory, ROOT)
    if validation_issues:
        return _blocked_plan(
            "blocked_phase5b_asset_inventory_invalid",
            *sorted({str(issue.get("rule")) for issue in validation_issues}),
        )

    assets = inventory.get("figure_assets", [])
    extraction_tasks = [_task_from_asset(asset) for asset in assets]
    extracted_count = sum(
        1 for asset in assets if asset.get("extraction_packet_status") == "extracted"
    )
    blockers = sorted(
        {
            "figure_region_crop_missing",
            "digitization_hash_missing",
            "axis_calibration_missing",
            "uncertainty_budget_missing",
            "review_certificate_missing",
            "phase5b_no_acceptance_promotion",
        }
    )

    return {
        "status": "phase5b_extraction_assets_located_not_extracted",
        "accepted_asset_claim": False,
        "accepted_digitization_claim": False,
        "accepted_runtime_claim": False,
        "promotes_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "inventory_path": str(path),
        "summary": {
            "total_assets": len(assets),
            "assets_located": len(assets),
            "extracted_packets": extracted_count,
            "accepted_assets": 0,
        },
        "blocking_reasons": blockers,
        "next_required_actions": [
            "crop_figure_region",
            "compute_region_or_digitization_hash",
            "calibrate_axes",
            "assign_uncertainty_budget",
            "attach_review_certificate",
        ],
        "extraction_tasks": extraction_tasks,
    }


def _task_from_asset(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": asset["id"],
        "figure_source_id": asset["figure_source_id"],
        "source_pdf_path": asset["source_pdf_path"],
        "source_pdf_sha256": asset["source_pdf_sha256"],
        "page": asset["page"],
        "figure_id": asset["figure_id"],
        "figure_caption_hint": asset["figure_caption_hint"],
        "task_status": "planned_not_executed",
        "region_status": asset["region_status"],
        "digitization_hash": asset["digitization_hash"],
        "required_outputs": [
            "figure_region_crop",
            "region_hash",
            "axis_calibration",
            "digitized_curve_or_observable_candidate",
            "uncertainty_budget",
            "review_certificate_reference",
        ],
        "accepted_task_claim": False,
        "promotes_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "blocked_reason": asset["blocked_reason"],
    }


def _blocked_plan(status: str, *reasons: str) -> dict[str, Any]:
    return {
        "status": status,
        "accepted_asset_claim": False,
        "accepted_digitization_claim": False,
        "accepted_runtime_claim": False,
        "promotes_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "summary": {"total_assets": 0, "assets_located": 0, "extracted_packets": 0, "accepted_assets": 0},
        "blocking_reasons": list(reasons),
        "next_required_actions": [],
        "extraction_tasks": [],
    }
