from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_ss12_phase5d_crop_artifacts.py"
CROP_PLAN = ROOT / "docs/SS12_P1_PHASE5D_CROP_REGION_PLAN_2026_05_22.json"
CROP_MANIFEST = ROOT / "docs/SS12_P1_PHASE5D_CROP_ARTIFACT_MANIFEST_2026_05_22.json"


def _run_builder(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(SCRIPT), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_manifest() -> dict:
    return json.loads(CROP_MANIFEST.read_text())


def test_phase5d_builder_creates_crop_artifact_manifest() -> None:
    completed = _run_builder()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert CROP_PLAN.exists()
    assert CROP_MANIFEST.exists()


def test_phase5d_crop_artifacts_exist_are_hashed_and_inside_page_bounds() -> None:
    manifest = _load_manifest()

    assert manifest["manifest_id"] == "ss12_p1_phase5d_crop_artifact_manifest"
    assert len(manifest["crop_artifacts"]) == 4
    assert manifest["acceptance_boundary"]["accepted_crop_claim"] is False
    assert manifest["acceptance_boundary"]["accepted_digitization_claim"] is False
    assert manifest["acceptance_boundary"]["promotes_acceptance"] is False

    for row in manifest["crop_artifacts"]:
        crop_path = Path(row["crop_image_path"])
        page_path = Path(row["page_image_path"])
        assert crop_path.exists(), row["id"]
        assert page_path.exists(), row["id"]
        assert row["crop_image_sha256"] == hashlib.sha256(crop_path.read_bytes()).hexdigest()
        with Image.open(crop_path) as crop, Image.open(page_path) as page:
            left, top, right, bottom = row["crop_bbox_px"]
            assert 0 <= left < right <= page.width
            assert 0 <= top < bottom <= page.height
            assert crop.width == right - left
            assert crop.height == bottom - top
        assert row["crop_status"] == "crop_region_selected_not_digitized"
        assert row["digitization_hash"] is None
        assert row["accepted_crop_claim"] is False
        assert row["accepted_digitization_claim"] is False
        assert row["accepted_runtime_claim"] is False
        assert row["promotes_acceptance"] is False
        assert row["can_support_first_principles_acceptance"] is False


def test_phase5d_rejects_crop_bbox_outside_page_bounds(tmp_path) -> None:
    plan = json.loads(CROP_PLAN.read_text())
    plan["crop_regions"][0]["crop_bbox_px"] = [-1, 10, 100, 100]
    bad_plan = tmp_path / "bad_crop_plan.json"
    bad_plan.write_text(json.dumps(plan))

    completed = _run_builder("--crop-plan", str(bad_plan))

    assert completed.returncode == 1
    assert "crop_bbox_outside_page_bounds" in completed.stdout


def test_phase5d_rejects_crop_output_outside_artifacts(tmp_path) -> None:
    completed = _run_builder("--crop-dir", str(tmp_path / "crops"))

    assert completed.returncode == 1
    assert "crop_dir_outside_artifacts" in completed.stdout


def test_phase5d_rejects_output_manifest_outside_docs(tmp_path) -> None:
    completed = _run_builder("--output", str(ROOT / "artifacts/ss12_phase5d/not_docs.json"))

    assert completed.returncode == 1
    assert "output_manifest_outside_docs" in completed.stdout


def test_phase5d_rejects_page_image_path_outside_artifacts(tmp_path) -> None:
    page_manifest = json.loads((ROOT / "docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json").read_text())
    page_manifest["page_render_artifacts"][0]["page_image_path"] = "/tmp/not_allowed.png"
    bad_page_manifest = tmp_path / "bad_page_manifest.json"
    bad_page_manifest.write_text(json.dumps(page_manifest))

    completed = _run_builder("--page-render-manifest", str(bad_page_manifest))

    assert completed.returncode == 1
    assert "page_image_path_outside_artifacts" in completed.stdout


def test_phase5d_rejects_duplicate_crop_source_ids(tmp_path) -> None:
    plan = json.loads(CROP_PLAN.read_text())
    plan["crop_regions"][1]["figure_source_id"] = plan["crop_regions"][0]["figure_source_id"]
    bad_plan = tmp_path / "duplicate_crop_source_plan.json"
    bad_plan.write_text(json.dumps(plan))

    completed = _run_builder("--crop-plan", str(bad_plan))

    assert completed.returncode == 1
    assert "duplicate_crop_region" in completed.stdout


def test_phase5d_rejects_duplicate_crop_ids(tmp_path) -> None:
    plan = json.loads(CROP_PLAN.read_text())
    plan["crop_regions"][1]["id"] = plan["crop_regions"][0]["id"]
    bad_plan = tmp_path / "duplicate_crop_id_plan.json"
    bad_plan.write_text(json.dumps(plan))

    completed = _run_builder("--crop-plan", str(bad_plan))

    assert completed.returncode == 1
    assert "duplicate_crop_region" in completed.stdout


def test_phase5d_rejects_duplicate_page_render_source_ids(tmp_path) -> None:
    page_manifest = json.loads((ROOT / "docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json").read_text())
    page_manifest["page_render_artifacts"][1]["figure_source_id"] = page_manifest["page_render_artifacts"][0]["figure_source_id"]
    bad_page_manifest = tmp_path / "duplicate_page_manifest.json"
    bad_page_manifest.write_text(json.dumps(page_manifest))

    completed = _run_builder("--page-render-manifest", str(bad_page_manifest))

    assert completed.returncode == 1
    assert "duplicate_page_render_source_id" in completed.stdout


def test_phase5d_rejects_duplicate_page_render_ids(tmp_path) -> None:
    page_manifest = json.loads((ROOT / "docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json").read_text())
    page_manifest["page_render_artifacts"][1]["id"] = page_manifest["page_render_artifacts"][0]["id"]
    bad_page_manifest = tmp_path / "duplicate_page_id_manifest.json"
    bad_page_manifest.write_text(json.dumps(page_manifest))

    completed = _run_builder("--page-render-manifest", str(bad_page_manifest))

    assert completed.returncode == 1
    assert "duplicate_page_render_id" in completed.stdout


def test_phase5d_rejects_page_image_hash_mismatch(tmp_path) -> None:
    page_manifest = json.loads((ROOT / "docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json").read_text())
    page_manifest["page_render_artifacts"][0]["page_image_sha256"] = "0" * 64
    bad_page_manifest = tmp_path / "bad_hash_page_manifest.json"
    bad_page_manifest.write_text(json.dumps(page_manifest))

    completed = _run_builder("--page-render-manifest", str(bad_page_manifest))

    assert completed.returncode == 1
    assert "page_image_sha256_mismatch" in completed.stdout


def test_phase5d_rejects_boolean_bbox_coordinates(tmp_path) -> None:
    plan = json.loads(CROP_PLAN.read_text())
    plan["crop_regions"][0]["crop_bbox_px"] = [True, 10, 100, 100]
    bad_plan = tmp_path / "boolean_bbox_plan.json"
    bad_plan.write_text(json.dumps(plan))

    completed = _run_builder("--crop-plan", str(bad_plan))

    assert completed.returncode == 1
    assert "invalid_crop_bbox" in completed.stdout
