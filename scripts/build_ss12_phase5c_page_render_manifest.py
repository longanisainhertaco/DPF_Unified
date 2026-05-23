#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_ss12_phase5b_figure_asset_inventory import validate_inventory  # noqa: E402

DEFAULT_ASSET_INVENTORY = ROOT / "docs/SS12_P1_PHASE5B_FIGURE_ASSET_INVENTORY_2026_05_22.json"
DEFAULT_OUTPUT = ROOT / "docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json"
DEFAULT_RENDER_DIR = ROOT / "artifacts/ss12_phase5c/page_renders"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def render_page(pdf_path: Path, page: int, output_path: Path, dpi: int = 150) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as document:
        if page < 1 or page > document.page_count:
            raise ValueError(f"page {page} outside PDF page_count {document.page_count}: {pdf_path}")
        pdf_page = document.load_page(page - 1)
        pixmap = pdf_page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(output_path)
    with Image.open(output_path) as image:
        return image.width, image.height


def build_manifest(
    *,
    asset_inventory_path: Path = DEFAULT_ASSET_INVENTORY,
    output_manifest_path: Path = DEFAULT_OUTPUT,
    render_dir: Path = DEFAULT_RENDER_DIR,
    dpi: int = 150,
) -> dict[str, Any]:
    output_manifest_path = output_manifest_path.resolve()
    render_dir = render_dir.resolve()
    if dpi < 72 or dpi > 300:
        return {
            "passed": False,
            "status": "blocked_phase5c_invalid_dpi",
            "issue_count": 1,
            "issues": [{"rule": "invalid_render_dpi", "dpi": dpi}],
            "manifest": str(output_manifest_path),
        }
    if not _is_relative_to(output_manifest_path, ROOT / "docs"):
        return {
            "passed": False,
            "status": "blocked_phase5c_output_outside_docs",
            "issue_count": 1,
            "issues": [{"rule": "output_manifest_outside_docs", "path": str(output_manifest_path)}],
            "manifest": str(output_manifest_path),
        }
    if not _is_relative_to(render_dir, ROOT / "artifacts"):
        return {
            "passed": False,
            "status": "blocked_phase5c_render_dir_outside_artifacts",
            "issue_count": 1,
            "issues": [{"rule": "render_dir_outside_artifacts", "path": str(render_dir)}],
            "manifest": str(output_manifest_path),
        }

    inventory = _load_json(asset_inventory_path)
    issues = validate_inventory(inventory, ROOT)
    if issues:
        return {
            "passed": False,
            "status": "blocked_phase5c_asset_inventory_invalid",
            "issue_count": len(issues),
            "issues": issues,
            "manifest": str(output_manifest_path),
        }

    artifacts: list[dict[str, Any]] = []
    for asset in inventory["figure_assets"]:
        pdf_path = Path(asset["source_pdf_path"])
        page = int(asset["page"])
        filename = f"{_safe_name(asset['figure_source_id'])}_page_{page:03d}.png"
        page_image_path = render_dir / filename
        width, height = render_page(pdf_path, page, page_image_path, dpi=dpi)
        artifacts.append(
            {
                "id": f"render_{asset['figure_source_id']}",
                "asset_id": asset["id"],
                "figure_source_id": asset["figure_source_id"],
                "source_pdf_path": asset["source_pdf_path"],
                "source_pdf_sha256": asset["source_pdf_sha256"],
                "page": page,
                "figure_id": asset["figure_id"],
                "render_dpi": dpi,
                "page_image_path": str(page_image_path),
                "page_image_sha256": _sha256(page_image_path),
                "page_width_px": width,
                "page_height_px": height,
                "crop_status": "crop_region_not_selected",
                "crop_bbox_pdf_points": None,
                "crop_image_path": None,
                "crop_image_sha256": None,
                "digitization_hash": None,
                "accepted_page_render_claim": False,
                "accepted_crop_claim": False,
                "accepted_digitization_claim": False,
                "accepted_runtime_claim": False,
                "promotes_acceptance": False,
                "can_support_first_principles_acceptance": False,
                "blocked_reason": "page rendered only; figure crop, axis calibration, uncertainty, and review certificate remain missing",
            }
        )

    manifest = {
        "manifest_id": "ss12_p1_phase5c_page_render_crop_manifest",
        "generated_at": "2026-05-22T08:15:00Z",
        "validation_scope": inventory["validation_scope"],
        "phase5b_asset_inventory": str(asset_inventory_path),
        "acceptance_boundary": {
            "accepted_page_render_claim": False,
            "accepted_crop_claim": False,
            "accepted_digitization_claim": False,
            "accepted_runtime_claim": False,
            "can_support_first_principles_acceptance": False,
            "promotes_acceptance": False,
            "note": "Phase 5-C page renders are source-location artifacts only. Crops and digitized values are not accepted."
        },
        "summary": {
            "total_render_artifacts": len(artifacts),
            "page_renders_created": len(artifacts),
            "crop_regions_selected": 0,
            "digitized_packets": 0,
            "accepted_packets": 0,
        },
        "page_render_artifacts": artifacts,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "passed": True,
        "status": "phase5c_page_renders_created_crop_regions_pending",
        "issue_count": 0,
        "manifest": str(output_manifest_path),
        "render_count": len(artifacts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SS12 Phase 5-C page render/crop manifest")
    parser.add_argument("--asset-inventory", default=str(DEFAULT_ASSET_INVENTORY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--render-dir", default=str(DEFAULT_RENDER_DIR))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    report = build_manifest(
        asset_inventory_path=Path(args.asset_inventory),
        output_manifest_path=Path(args.output),
        render_dir=Path(args.render_dir),
        dpi=args.dpi,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
