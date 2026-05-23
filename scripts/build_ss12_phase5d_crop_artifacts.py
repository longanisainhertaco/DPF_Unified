#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGE_RENDER_MANIFEST = ROOT / "docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json"
DEFAULT_CROP_PLAN = ROOT / "docs/SS12_P1_PHASE5D_CROP_REGION_PLAN_2026_05_22.json"
DEFAULT_OUTPUT = ROOT / "docs/SS12_P1_PHASE5D_CROP_ARTIFACT_MANIFEST_2026_05_22.json"
DEFAULT_CROP_DIR = ROOT / "artifacts/ss12_phase5d/crops"
ACCEPTANCE_FLAGS: tuple[str, ...] = (
    "accepted_crop_claim",
    "accepted_digitization_claim",
    "accepted_runtime_claim",
    "can_support_first_principles_acceptance",
    "promotes_acceptance",
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")


def _blocked(status: str, rule: str, **detail: Any) -> dict[str, Any]:
    return {
        "passed": False,
        "status": status,
        "issue_count": 1,
        "issues": [{"rule": rule, **detail}],
    }


def build_crop_manifest(
    *,
    crop_plan_path: Path = DEFAULT_CROP_PLAN,
    page_render_manifest_path: Path = DEFAULT_PAGE_RENDER_MANIFEST,
    output_manifest_path: Path = DEFAULT_OUTPUT,
    crop_dir: Path = DEFAULT_CROP_DIR,
) -> dict[str, Any]:
    output_manifest_path = output_manifest_path.resolve()
    crop_dir = crop_dir.resolve()
    if not _is_relative_to(output_manifest_path, ROOT / "docs"):
        return _blocked("blocked_phase5d_output_outside_docs", "output_manifest_outside_docs", path=str(output_manifest_path))
    if not _is_relative_to(crop_dir, ROOT / "artifacts"):
        return _blocked("blocked_phase5d_crop_dir_outside_artifacts", "crop_dir_outside_artifacts", path=str(crop_dir))

    page_manifest = _load_json(page_render_manifest_path)
    crop_plan = _load_json(crop_plan_path)
    boundary = crop_plan.get("acceptance_boundary", {})
    for flag in ACCEPTANCE_FLAGS:
        if boundary.get(flag) is not False:
            return _blocked("blocked_phase5d_crop_plan_acceptance_flag", "crop_plan_acceptance_flag_not_false", flag=flag)

    page_artifacts = page_manifest.get("page_render_artifacts", [])
    page_source_ids = [str(row.get("figure_source_id")) for row in page_artifacts]
    page_ids = [str(row.get("id")) for row in page_artifacts]
    if len(page_ids) != len(set(page_ids)):
        return _blocked("blocked_phase5d_duplicate_page_render_id", "duplicate_page_render_id")
    if len(page_source_ids) != len(set(page_source_ids)):
        return _blocked("blocked_phase5d_duplicate_page_render_source_id", "duplicate_page_render_source_id")
    page_rows = {row["figure_source_id"]: row for row in page_artifacts}
    crop_rows = crop_plan.get("crop_regions", [])
    crop_source_ids = [str(row.get("figure_source_id")) for row in crop_rows]
    crop_ids = [str(row.get("id")) for row in crop_rows]
    if len(crop_source_ids) != len(set(crop_source_ids)) or len(crop_ids) != len(set(crop_ids)):
        return _blocked("blocked_phase5d_duplicate_crop_region", "duplicate_crop_region")
    if set(crop_source_ids) != set(page_rows):
        return _blocked(
            "blocked_phase5d_crop_plan_mismatch",
            "crop_plan_ids_do_not_match_page_render_manifest",
            expected=sorted(page_rows),
            actual=sorted(str(row.get("figure_source_id")) for row in crop_rows),
        )

    artifacts: list[dict[str, Any]] = []
    crop_dir.mkdir(parents=True, exist_ok=True)
    for crop in crop_rows:
        for flag in ACCEPTANCE_FLAGS:
            if crop.get(flag) is not False:
                return _blocked("blocked_phase5d_crop_row_acceptance_flag", "crop_row_acceptance_flag_not_false", row_id=crop.get("id"), flag=flag)
        source_id = crop["figure_source_id"]
        page_row = page_rows[source_id]
        page_path = Path(page_row["page_image_path"]).resolve()
        if not _is_relative_to(page_path, ROOT / "artifacts"):
            return _blocked(
                "blocked_phase5d_page_image_path_outside_artifacts",
                "page_image_path_outside_artifacts",
                row_id=page_row.get("id"),
                path=str(page_path),
            )
        if not page_path.exists():
            return _blocked(
                "blocked_phase5d_page_image_missing",
                "page_image_missing",
                row_id=page_row.get("id"),
                path=str(page_path),
            )
        actual_page_sha = _sha256(page_path)
        if page_row.get("page_image_sha256") != actual_page_sha:
            return _blocked(
                "blocked_phase5d_page_image_sha256_mismatch",
                "page_image_sha256_mismatch",
                row_id=page_row.get("id"),
                expected=page_row.get("page_image_sha256"),
                actual=actual_page_sha,
            )
        bbox = crop.get("crop_bbox_px")
        if not _valid_bbox(bbox):
            return _blocked("blocked_phase5d_invalid_crop_bbox", "invalid_crop_bbox", row_id=crop.get("id"), bbox=bbox)
        left, top, right, bottom = [int(v) for v in bbox]
        with Image.open(page_path) as page_image:
            if left < 0 or top < 0 or right > page_image.width or bottom > page_image.height:
                return _blocked(
                    "blocked_phase5d_crop_bbox_outside_page_bounds",
                    "crop_bbox_outside_page_bounds",
                    row_id=crop.get("id"),
                    bbox=bbox,
                    page_size=[page_image.width, page_image.height],
                )
            crop_image = page_image.crop((left, top, right, bottom))
            crop_path = crop_dir / f"{_safe_name(source_id)}_crop.png"
            crop_image.save(crop_path)
        artifacts.append(
            {
                "id": f"crop_{source_id}",
                "crop_plan_id": crop["id"],
                "figure_source_id": source_id,
                "page_render_id": page_row["id"],
                "page_image_path": page_row["page_image_path"],
                "page_image_sha256": page_row["page_image_sha256"],
                "crop_bbox_px": [left, top, right, bottom],
                "crop_basis": crop["crop_basis"],
                "crop_status": "crop_region_selected_not_digitized",
                "crop_image_path": str(crop_path),
                "crop_image_sha256": _sha256(crop_path),
                "digitization_hash": None,
                "axis_calibration_status": "missing",
                "uncertainty_budget_status": "missing",
                "review_certificate_status": "missing",
                "accepted_crop_claim": False,
                "accepted_digitization_claim": False,
                "accepted_runtime_claim": False,
                "promotes_acceptance": False,
                "can_support_first_principles_acceptance": False,
                "blocked_reason": "crop artifact only; digitization, axis calibration, uncertainty, and review certificate remain missing",
            }
        )

    manifest = {
        "manifest_id": "ss12_p1_phase5d_crop_artifact_manifest",
        "generated_at": "2026-05-22T09:35:00Z",
        "phase5c_page_render_manifest": str(page_render_manifest_path),
        "phase5d_crop_region_plan": str(crop_plan_path),
        "acceptance_boundary": {
            "accepted_crop_claim": False,
            "accepted_digitization_claim": False,
            "accepted_runtime_claim": False,
            "can_support_first_principles_acceptance": False,
            "promotes_acceptance": False,
            "note": "Phase 5-D crop artifacts are selected image regions only; they are not digitized or accepted observables."
        },
        "summary": {
            "total_crop_artifacts": len(artifacts),
            "digitized_packets": 0,
            "accepted_packets": 0,
        },
        "crop_artifacts": artifacts,
    }
    output_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "passed": True,
        "status": "phase5d_crop_artifacts_created_not_digitized",
        "issue_count": 0,
        "manifest": str(output_manifest_path),
        "crop_count": len(artifacts),
    }


def _valid_bbox(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(type(v) is int for v in value):
        return False
    left, top, right, bottom = value
    return left < right and top < bottom


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SS12 Phase 5-D crop artifacts")
    parser.add_argument("--crop-plan", default=str(DEFAULT_CROP_PLAN))
    parser.add_argument("--page-render-manifest", default=str(DEFAULT_PAGE_RENDER_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--crop-dir", default=str(DEFAULT_CROP_DIR))
    args = parser.parse_args()
    report = build_crop_manifest(
        crop_plan_path=Path(args.crop_plan),
        page_render_manifest_path=Path(args.page_render_manifest),
        output_manifest_path=Path(args.output),
        crop_dir=Path(args.crop_dir),
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
