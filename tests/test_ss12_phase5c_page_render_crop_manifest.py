from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_ss12_phase5c_page_render_manifest.py"
MANIFEST = ROOT / "docs/SS12_P1_PHASE5C_PAGE_RENDER_CROP_MANIFEST_2026_05_22.json"


def _run_builder() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_phase5c_builder_generates_page_render_manifest() -> None:
    completed = _run_builder()

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert MANIFEST.exists()


def test_phase5c_manifest_has_one_render_artifact_per_asset() -> None:
    manifest = _load_manifest()

    assert manifest["manifest_id"] == "ss12_p1_phase5c_page_render_crop_manifest"
    assert len(manifest["page_render_artifacts"]) == 4
    assert len({row["figure_source_id"] for row in manifest["page_render_artifacts"]}) == 4
    assert manifest["acceptance_boundary"]["accepted_page_render_claim"] is False
    assert manifest["acceptance_boundary"]["accepted_crop_claim"] is False
    assert manifest["acceptance_boundary"]["accepted_digitization_claim"] is False
    assert manifest["acceptance_boundary"]["promotes_acceptance"] is False


def test_phase5c_render_artifacts_exist_and_are_hashed() -> None:
    manifest = _load_manifest()

    for row in manifest["page_render_artifacts"]:
        image_path = Path(row["page_image_path"])
        assert image_path.exists(), row["id"]
        assert image_path.suffix == ".png"
        actual_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert row["page_image_sha256"] == actual_sha
        assert len(row["page_image_sha256"]) == 64
        assert row["page_width_px"] > 0
        assert row["page_height_px"] > 0
        assert row["crop_status"] == "crop_region_not_selected"
        assert row["crop_image_path"] is None
        assert row["crop_image_sha256"] is None
        assert row["digitization_hash"] is None
        assert row["accepted_page_render_claim"] is False
        assert row["accepted_crop_claim"] is False
        assert row["accepted_digitization_claim"] is False
        assert row["promotes_acceptance"] is False
        assert row["can_support_first_principles_acceptance"] is False


def test_phase5c_builder_rejects_invalid_asset_inventory(tmp_path) -> None:
    source_inventory = json.loads(
        (ROOT / "docs/SS12_P1_PHASE5B_FIGURE_ASSET_INVENTORY_2026_05_22.json").read_text()
    )
    source_inventory["figure_assets"][0]["source_pdf_sha256"] = "0" * 64
    bad_inventory = tmp_path / "bad_inventory.json"
    bad_inventory.write_text(json.dumps(source_inventory))

    completed = subprocess.run(
        [
            str(ROOT / ".venv312/bin/python"),
            str(SCRIPT),
            "--asset-inventory",
            str(bad_inventory),
            "--render-dir",
            str(ROOT / "artifacts/ss12_phase5c/bad_inventory_test"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "source_pdf_sha256_mismatch" in completed.stdout
    bad_render_dir = ROOT / "artifacts/ss12_phase5c/bad_inventory_test"
    assert not bad_render_dir.exists() or not any(bad_render_dir.iterdir())


def test_phase5c_builder_rejects_render_dir_outside_artifacts(tmp_path) -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(SCRIPT), "--render-dir", str(tmp_path / "renders")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "render_dir_outside_artifacts" in completed.stdout


def test_phase5c_builder_rejects_output_outside_docs() -> None:
    completed = subprocess.run(
        [
            str(ROOT / ".venv312/bin/python"),
            str(SCRIPT),
            "--output",
            str(ROOT / "artifacts/ss12_phase5c/not_docs_manifest.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "output_manifest_outside_docs" in completed.stdout


def test_phase5c_builder_allows_dpi_boundaries() -> None:
    for dpi in (72, 300):
        output_path = ROOT / "docs" / f"phase5c_dpi_{dpi}_test.json"
        render_dir = ROOT / "artifacts/ss12_phase5c" / f"dpi_{dpi}_test"
        completed = subprocess.run(
            [
                str(ROOT / ".venv312/bin/python"),
                str(SCRIPT),
                "--dpi",
                str(dpi),
                "--output",
                str(output_path),
                "--render-dir",
                str(render_dir),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        if output_path.exists():
            output_path.unlink()
        if render_dir.exists():
            shutil.rmtree(render_dir)


def test_phase5c_builder_rejects_unbounded_dpi() -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(SCRIPT), "--dpi", "1200"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "invalid_render_dpi" in completed.stdout
