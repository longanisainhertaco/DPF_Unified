"""Tests for 2026-05-20 user-supplied paper intake and target extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dpf.first_principles.sprint6_user_target_extractions import (
    BRUZZONE_BERNAL_2001_DUPLICATE_VERIFICATION,
    GRIBKOV_2007_PF1000_PART2_EXTRACTION,
    GRIBKOV_MALAQUIAS_2006_DMP_APPLICATIONS_EXTRACTION,
    HEROLD_1989_POSEIDON_PF360_EXTRACTION,
    LOARER_2007_GAS_BALANCE_EXTRACTION,
    SCHOLZ_1999_FOAM_LINER_EXTRACTION,
    SCHOLZ_2000_PF1000_DEVICE_EXTRACTION,
    SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_EXTRACTION,
    SHAKYA_2015_LEE_MODEL_EXTRACTION,
    SPRINT6_USER_SUPPLIED_TARGET_EXTRACTIONS,
    sprint6_user_supplied_target_extractions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INTAKE_REPORT = REPO_ROOT / "docs" / "USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.json"
TARGET_DOC = (
    REPO_ROOT
    / "docs"
    / "extractions"
    / "SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_TARGET_EXTRACTION_2026_05_20.md"
)


def test_user_supplied_intake_report_fails_closed() -> None:
    report = json.loads(INTAKE_REPORT.read_text())
    assert report["files_scanned"] == 9
    assert report["promoted_count"] == 0
    assert report["skipped_existing_count"] == 9
    assert report["failed_count"] == 0
    assert report["accepted_runtime_claim"] is False
    assert report["can_support_first_principles_acceptance"] is False


def _report_records() -> dict[str, dict[str, object]]:
    report = json.loads(INTAKE_REPORT.read_text())
    return {
        str(item["title"]): item
        for bucket in ("promoted", "skipped_existing")
        for item in report[bucket]
    }


def test_all_user_supplied_sources_are_represented_fail_closed() -> None:
    records = _report_records()
    assert set(records) == {
        "Recent progress in 1 MJ Plasma-Focus research",
        (
            "The need of using anomalous resistivity due to Lower Hybrid "
            "Instabilities in plasma-magnetic field interfaces"
        ),
        "PF-1000 device",
        (
            "Comparative analysis of large plasma focus experiments performed "
            "at IPF, Stuttgart, and IPJ, Swierk"
        ),
        "Foam liner driven by a plasma focus current sheath",
        "Gas balance and fuel retention in fusion devices",
        "Comparison of Plasma Dynamics in Plasma Focus Devices PF1000 and PF400",
        (
            "Plasma dynamics in the PF-1000 device under full-scale energy "
            "storage: II. Fast electron and ion characteristics versus "
            "neutron emission parameters and gun optimization perspectives"
        ),
        (
            "Dense magnetized plasma and its applications: review of the "
            "3-year activity of the IAEA Co-ordinated Research Programme"
        ),
    }
    for item in records.values():
        assert item["status"] == "skipped_existing_kr_source"
        assert item["accepted_runtime_claim"] is False
        assert item["can_support_first_principles_acceptance"] is False


def test_intake_sha_set_matches_target_packet_sources() -> None:
    report = json.loads(INTAKE_REPORT.read_text())
    intake_shas = {
        item["sha256"]
        for bucket in ("promoted", "skipped_existing")
        for item in report[bucket]
    }
    packet_shas = {
        packet["source_pdf_sha256"]
        for packet in SPRINT6_USER_SUPPLIED_TARGET_EXTRACTIONS.values()
    }
    assert packet_shas == intake_shas


def test_sprint6_artifacts_have_no_acceptance_true_flags() -> None:
    paths = [
        REPO_ROOT / "docs" / "USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.json",
        REPO_ROOT / "docs" / "USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.md",
        REPO_ROOT / "src" / "dpf" / "first_principles"
        / "sprint6_user_target_extractions.py",
    ]
    for path in paths:
        text = path.read_text()
        assert '"accepted_runtime_claim": true' not in text.lower()
        assert '"can_support_first_principles_acceptance": true' not in text.lower()
        assert "accepted_runtime_claim=True" not in text
        assert "can_support_first_principles_acceptance=True" not in text


def test_intake_report_records_exact_duplicates_and_doi_equivalence() -> None:
    records = _report_records()
    bruzzone = records[
        "The need of using anomalous resistivity due to Lower Hybrid "
        "Instabilities in plasma-magnetic field interfaces"
    ]
    assert bruzzone["sha256"] == (
        "73668d0e98604959a6fcd3e20adfd5d55d757dfad943972a2b56a9595f927112"
    )
    assert "source SHA already appears" in str(bruzzone["reason"])
    gribkov = records[
        "Plasma dynamics in the PF-1000 device under full-scale energy storage: "
        "II. Fast electron and ion characteristics versus neutron emission "
        "parameters and gun optimization perspectives"
    ]
    assert "10.1088/0022-3727/40/12/008" in str(gribkov["reason"])
    assert "scholz-2007-pf1000-part2-jphysd.md" in str(gribkov["reason"])


def test_target_manifest_contains_nine_fail_closed_packets() -> None:
    manifest = sprint6_user_supplied_target_extractions()
    assert manifest["packets_count"] == 9
    assert set(SPRINT6_USER_SUPPLIED_TARGET_EXTRACTIONS) == {
        "scholz_2001_recent_progress_pf1000_hardware",
        "bruzzone_bernal_2001_lhi_duplicate_verification",
        "scholz_2000_pf1000_device",
        "herold_1989_poseidon_pf360_comparative",
        "scholz_1999_foam_liner_current_sheath",
        "loarer_2007_tokamak_gas_balance_fuel_retention",
        "shakya_2015_pf1000_pf400_lee_model",
        "gribkov_2007_pf1000_part2_existing_kr_equivalent",
        "gribkov_malaquias_2006_dmp_applications",
    }
    assert manifest["accepted_runtime_claim"] is False
    assert manifest["can_support_first_principles_acceptance"] is False
    for packet in manifest["packets"].values():
        assert packet["accepted_runtime_claim"] is False
        assert packet["can_support_first_principles_acceptance"] is False


def test_packet_source_and_render_artifact_paths_exist() -> None:
    """Every packet with local path references points at an existing artifact."""
    for packet in SPRINT6_USER_SUPPLIED_TARGET_EXTRACTIONS.values():
        for key in ("kr_markdown", "kr_json", "existing_kr_markdown", "existing_kr_json"):
            path = packet.get(key)
            if path:
                assert (REPO_ROOT / path).is_file(), f"{packet['source_id']} {key}"
        report = packet.get("kr_promotion_report")
        if report:
            assert (REPO_ROOT / report).is_file(), packet["source_id"]

        if artifact := packet.get("render_artifact"):
            artifact_path = REPO_ROOT / artifact
            assert artifact_path.is_file(), packet["source_id"]
            expected_sha = packet.get("render_artifact_sha256")
            if expected_sha:
                actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                assert actual == expected_sha

        for artifact in packet.get("render_artifacts", ()):
            assert (REPO_ROOT / artifact).is_file(), packet["source_id"]


def test_scholz_hardware_targets_are_source_scoped() -> None:
    packet = SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_EXTRACTION
    assert packet["scope_tag"] == "pf1000_2001_24_rod_large_electrode_hardware"
    assert set(packet["resolves_blockers"]) == {"PF1000-BLK-004", "PF1000-BLK-015"}
    assert "PF1000-BLK-009" in packet["candidate_context_only"]
    assert "PF1000-BLK-016" in packet["still_blocked"]

    targets = packet["targets"]
    assert targets["cathode_rod_count"]["value"] == 24
    assert targets["cathode_rod_length_m"]["value"] == pytest.approx(0.600)
    assert targets["cathode_rod_diameter_m"]["value"] == pytest.approx(0.032)
    assert targets["outer_electrode_radius_m"]["value"] == pytest.approx(0.200)
    assert targets["inner_electrode_radius_m"]["value"] == pytest.approx(0.122)
    assert targets["insulator_outer_radius_m"]["value"] == pytest.approx(0.1145)
    assert targets["insulator_length_m"]["value"] == pytest.approx(0.113)


def test_scholz_2000_pf1000_device_targets_bank_and_geometry_context() -> None:
    packet = SCHOLZ_2000_PF1000_DEVICE_EXTRACTION
    assert packet["resolves_blockers"] == ("PF1000-BLK-004",)
    assert "PF1000-BLK-015" in packet["still_blocked"]
    targets = packet["targets"]
    assert targets["cathode_rod_count"]["value"] == 24
    assert targets["cathode_rod_length_m"]["value"] == pytest.approx(0.600)
    assert targets["capacitance_total_F"]["value"] == pytest.approx(1.332e-3)
    assert targets["nominal_inductance_H"]["value"] == pytest.approx(8.9e-9)
    assert targets["diagnostic_shot_current_A"]["value"] == pytest.approx(1.5e6)
    assert "ambiguous" in targets["reported_oe_ce_dimension_ambiguous_m"]["scope_note"]


def test_herold_cross_machine_packet_stays_context_only() -> None:
    packet = HEROLD_1989_POSEIDON_PF360_EXTRACTION
    assert packet["resolves_blockers"] == ()
    assert "not PF-1000 same-scope" in packet["scope_caveat"]
    targets = packet["targets"]
    assert targets["poseidon_ceramic_current_MA"]["value"] == pytest.approx(4.9)
    assert targets["good_shot_radial_compression_velocity_m_s"]["value"] == pytest.approx(
        1.5e5
    )
    assert targets["very_high_energy_ion_range_MeV"]["value"] == (1.0, 6.0)


def test_scholz_1999_foam_liner_packet_is_modified_load_context() -> None:
    packet = SCHOLZ_1999_FOAM_LINER_EXTRACTION
    assert packet["resolves_blockers"] == ()
    assert "modified foam-liner target" in packet["scope_caveat"]
    targets = packet["targets"]
    assert targets["modified_inner_electrode_diameter_m"]["value"] == pytest.approx(0.100)
    assert targets["fill_pressure_hPa"]["value"] == pytest.approx(4.7)
    assert targets["active_power_peak_W"]["value"] == pytest.approx(3.0e10)


def test_loarer_tokamak_gas_balance_cannot_close_dpf_blockers() -> None:
    packet = LOARER_2007_GAS_BALANCE_EXTRACTION
    assert packet["resolves_blockers"] == ()
    assert "not a DPF source" in packet["scope_caveat"]
    targets = packet["targets"]
    assert targets["long_term_retention_fraction_gas_balance"]["value"] == (0.10, 0.20)
    assert targets["post_mortem_retention_fraction"]["value"] == (0.03, 0.04)


def test_shakya_lee_model_packet_is_baseline_context_only() -> None:
    packet = SHAKYA_2015_LEE_MODEL_EXTRACTION
    assert packet["resolves_blockers"] == ()
    assert "not first-principles runtime evidence" in packet["scope_caveat"]
    assert "sanitized" in packet["source_doi_status"]
    targets = packet["targets"]
    assert targets["pf1000_lee_cathode_radius_m"]["value"] == pytest.approx(0.160)
    assert "not hardware metrology" in targets["pf1000_lee_cathode_radius_m"]["scope_note"]
    assert targets["pf1000_model_factors"]["value"]["fm"] == pytest.approx(0.13)


def test_gribkov_2007_existing_kr_packet_has_neutron_beam_context() -> None:
    packet = GRIBKOV_2007_PF1000_PART2_EXTRACTION
    assert packet["resolves_blockers"] == ()
    assert "existing KR" in packet["existing_kr_status"]
    targets = packet["targets"]
    assert targets["medium_deuteron_energy_keV"]["value"] == (50.0, 150.0)
    assert targets["typical_total_current_A"]["value"] == (2.5e6, 2.6e6)
    assert targets["shot_3121_y0_y90"]["value"] == pytest.approx(1.8)
    assert targets["neutron_pulse_fwhm_s"]["value"] == pytest.approx(150.0e-9)


def test_gribkov_malaquias_dmp_review_stays_applications_context() -> None:
    packet = GRIBKOV_MALAQUIAS_2006_DMP_APPLICATIONS_EXTRACTION
    assert packet["resolves_blockers"] == ()
    assert "does not close PF-1000" in packet["scope_caveat"]
    targets = packet["targets"]
    assert targets["pf1000_implosion_velocity_cm_s"]["value"] == pytest.approx(5.0e7)
    assert targets["radiation_material_flux_regimes_W_cm2"]["value"]["implantation"] == (
        1.0e5,
        1.0e7,
    )


def test_scholz_end_face_hole_does_not_close_full_hollow_bore() -> None:
    targets = SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_EXTRACTION["targets"]
    hole = targets["anode_end_face_hole_radius_m"]
    assert hole["value"] == pytest.approx(0.015)
    assert hole["resolves"] == ()
    assert hole["candidate_context_for"] == ("PF1000-BLK-009",)
    assert "PF1000-BLK-010" in (
        SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_EXTRACTION["still_blocked"]
    )


def test_render_artifact_manifest_matches_packet() -> None:
    packet = SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_EXTRACTION
    artifact = REPO_ROOT / packet["render_artifact"]
    manifest_path = artifact.parent / "render_manifest.json"
    assert artifact.is_file()
    manifest = json.loads(manifest_path.read_text())
    rendered = manifest["rendered_pages"][0]
    assert rendered["artifact"] == packet["render_artifact"]
    assert rendered["artifact_sha256"] == packet["render_artifact_sha256"]
    assert rendered["pdf_page"] == 2
    assert rendered["journal_page"] == 36


def test_bruzzone_duplicate_packet_preserves_anomalous_resistivity_blocker() -> None:
    packet = BRUZZONE_BERNAL_2001_DUPLICATE_VERIFICATION
    assert packet["status"] == "exact_sha_duplicate_existing_kr_source"
    assert packet["resolves_blockers"] == ()
    assert packet["candidate_context_only"] == ("CLOSURE-BLK-ANOM-001",)
    assert (REPO_ROOT / packet["existing_kr_json"]).is_file()
    assert (REPO_ROOT / packet["existing_kr_markdown"]).is_file()


def test_target_doc_preserves_fail_closed_language() -> None:
    text = TARGET_DOC.read_text()
    assert "accepted_runtime_claim=false" in text
    assert "can_support_first_principles_acceptance=false" in text
    assert "Still blocked" in text
    assert "PF1000-BLK-016" in text
