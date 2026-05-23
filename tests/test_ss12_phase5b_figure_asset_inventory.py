from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_INVENTORY = ROOT / "docs/SS12_P1_PHASE5B_FIGURE_ASSET_INVENTORY_2026_05_22.json"
VALIDATOR = ROOT / "scripts/validate_ss12_phase5b_figure_asset_inventory.py"
SOURCE_MANIFEST = ROOT / "docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json"

REQUIRED_ROWS = {
    "pf1000_recent_progress_fig6_current_waveform",
    "pf1000_scholz_fig4_density_distribution",
    "pf1000_krauz_fig8_magnetic_probe_current",
    "pf1000_scholz_fig9_neutron_timing",
}


def _load_inventory() -> dict:
    return json.loads(ASSET_INVENTORY.read_text())


def test_phase5b_asset_inventory_links_all_phase5_rows() -> None:
    inventory = _load_inventory()
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_ids = {row["id"] for row in source_manifest["figure_sources"]}
    asset_ids = {row["figure_source_id"] for row in inventory["figure_assets"]}

    assert asset_ids == source_ids == REQUIRED_ROWS
    assert inventory["acceptance_boundary"]["accepted_asset_claim"] is False
    assert inventory["acceptance_boundary"]["promotes_acceptance"] is False
    assert inventory["acceptance_boundary"]["can_support_first_principles_acceptance"] is False


def test_phase5b_asset_rows_are_non_promoting_and_have_pdf_hashes() -> None:
    inventory = _load_inventory()

    for row in inventory["figure_assets"]:
        pdf_path = Path(row["source_pdf_path"])
        assert pdf_path.exists(), row["id"]
        assert len(row["source_pdf_sha256"]) == 64
        assert row["page"] >= 1
        assert row["asset_status"] == "asset_located_not_extracted"
        assert row["region_status"] == "region_hint_only"
        assert row["extraction_packet_status"] == "not_extracted"
        assert row["digitization_hash"] is None
        assert row["accepted_asset_claim"] is False
        assert row["accepted_digitization_claim"] is False
        assert row["accepted_runtime_claim"] is False
        assert row["promotes_acceptance"] is False
        assert row["can_support_first_principles_acceptance"] is False
        assert row["source_kind"] in {"repo_pdf", "external_pdf"}


def test_phase5b_asset_validator_accepts_inventory() -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(ASSET_INVENTORY)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["issue_count"] == 0


def test_phase5b_asset_validator_rejects_missing_phase5_row(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["figure_assets"] = inventory["figure_assets"][:-1]
    mutated = tmp_path / "missing_row.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "asset_ids_do_not_match_phase5_manifest" in completed.stdout


def test_phase5b_asset_validator_rejects_accepted_flags(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["figure_assets"][0]["accepted_digitization_claim"] = True
    mutated = tmp_path / "accepted_flag.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "asset_acceptance_flag_not_false" in completed.stdout


def test_phase5b_asset_validator_rejects_top_level_accepted_flags(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["acceptance_boundary"]["accepted_digitization_claim"] = True
    mutated = tmp_path / "top_level_accepted_flag.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "top_level_acceptance_flag_not_false" in completed.stdout


def test_phase5b_asset_validator_rejects_bad_hash(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["figure_assets"][0]["source_pdf_sha256"] = "0" * 64
    mutated = tmp_path / "bad_hash.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "source_pdf_sha256_mismatch" in completed.stdout


def test_phase5b_asset_validator_allows_repo_symlink_to_allowed_pdf_root() -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(ASSET_INVENTORY)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    inventory = _load_inventory()
    krauz = next(row for row in inventory["figure_assets"] if row["id"] == "asset_pf1000_krauz_fig8_magnetic_probe_current")
    assert krauz["source_kind"] == "repo_pdf"
    assert Path(krauz["source_pdf_path"]).is_symlink()
    assert str(Path(krauz["source_pdf_path"]).resolve()).startswith("/Users/anthonyzamora/PDFs/")


def test_phase5b_asset_validator_rejects_phase5_manifest_path_escape(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["phase5_source_manifest"] = "../outside_manifest.json"
    mutated = tmp_path / "manifest_escape.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "phase5_source_manifest_outside_repo" in completed.stdout


def test_phase5b_asset_validator_rejects_pdf_path_outside_allowed_roots(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["figure_assets"][0]["source_pdf_path"] = "/tmp/outside.pdf"
    mutated = tmp_path / "pdf_escape.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "source_pdf_outside_allowed_roots" in completed.stdout


def test_phase5b_asset_validator_rejects_extracted_without_hash(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["figure_assets"][0]["extraction_packet_status"] = "extracted"
    mutated = tmp_path / "extracted_without_hash.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "digitization_hash_required_for_extracted_packet" in completed.stdout


def test_phase5b_asset_validator_allows_extracted_with_hash_but_not_acceptance(tmp_path) -> None:
    inventory = _load_inventory()
    inventory["figure_assets"][0]["extraction_packet_status"] = "extracted"
    inventory["figure_assets"][0]["digitization_hash"] = "a" * 64
    mutated = tmp_path / "extracted_with_hash.json"
    mutated.write_text(json.dumps(inventory))

    completed = subprocess.run(
        [str(ROOT / ".venv312/bin/python"), str(VALIDATOR), str(mutated)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert inventory["figure_assets"][0]["accepted_digitization_claim"] is False
