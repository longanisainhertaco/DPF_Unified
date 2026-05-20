"""Tests for Sprint 5 WS2 target-extraction packets.

These tests enforce structural invariants on the seven extraction packets:

- every packet carries ``accepted_runtime_claim = False``;
- every packet carries ``can_support_first_principles_acceptance = False``;
- the audit-row corrections (rows 6, 7, 8) and the negative findings
  (Bernard thermonuclear prefactor not found; 320/500 keV is FF-1 not
  PF-1000) are encoded in the packets and asserted here.
- Codex Sprint 5 WS2 audit acceptance tests:
    * A1 — per-target ``resolves`` is a subset of top-level
      ``resolves_blockers`` (no per-target claim that the top level does
      not endorse);
    * A2 — the Sprint 5 free-acquisition memo does not assert the
      broad "no DPF in any literature" Te/Ti wording.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dpf.first_principles.sprint5_target_extractions import (
    BENNETT_2017_STARTUP_EXTRACTION,
    BERNARD_1977_HISTORICAL_EXTRACTION,
    BRAGINSKII_1965_TRANSPORT_EXTRACTION,
    PLASMA_FOCUS_UPDATE_2021_EXTRACTION,
    SCHOLZ_GRIBKOV_2007_PARTII_EXTRACTION,
    SPRINT_5_TARGET_EXTRACTIONS,
    STEPNIEWSKI_2004_REVIEW_PACKET,
    UCSD_BEG_CURRENT_SHEATH_EXTRACTION,
    sprint5_local_target_extractions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FREE_ACQ_MEMO = (
    REPO_ROOT / "docs" / "SPRINT5_FREE_ACQUISITIONS_2026_05_20.md"
)


def test_sprint5_has_seven_packets() -> None:
    assert len(SPRINT_5_TARGET_EXTRACTIONS) == 7
    manifest = sprint5_local_target_extractions()
    assert manifest["packets_count"] == 7


def test_every_packet_fails_closed() -> None:
    for source_id, packet in SPRINT_5_TARGET_EXTRACTIONS.items():
        assert packet["accepted_runtime_claim"] is False, source_id
        assert packet["can_support_first_principles_acceptance"] is False, source_id


def test_manifest_fails_closed() -> None:
    manifest = sprint5_local_target_extractions()
    assert manifest["accepted_runtime_claim"] is False
    assert manifest["can_support_first_principles_acceptance"] is False


def test_bennett_row_7_correction_71pct_at_1us() -> None:
    target = BENNETT_2017_STARTUP_EXTRACTION["targets"]["sheath_current_fraction_1us"]
    assert target["audit_row_7_correction_confirmed"] is True
    assert target["value_percent"] == pytest.approx(71.0)
    assert target["at_time_us"] == pytest.approx(1.0)


def test_bennett_resolves_four_startup_channels() -> None:
    expected = {"STARTUP-BVP-CH03", "STARTUP-BVP-CH04", "STARTUP-BVP-CH07", "STARTUP-BVP-CH08"}
    assert set(BENNETT_2017_STARTUP_EXTRACTION["resolves_blockers"]) == expected


def test_bennett_explosive_emission_thresholds() -> None:
    bulk = BENNETT_2017_STARTUP_EXTRACTION["targets"]["explosive_emission_threshold_bulk"]
    knife = BENNETT_2017_STARTUP_EXTRACTION["targets"]["explosive_emission_threshold_knife_edge"]
    assert bulk["value"] == pytest.approx(250.0)
    assert bulk["units"] == "kV/cm"
    assert knife["value"] == pytest.approx(10.0)
    assert knife["units"] == "kV/cm"


def test_braginskii_render_verification_complete() -> None:
    p = BRAGINSKII_1965_TRANSPORT_EXTRACTION
    assert p["render_verification_status"] == "verified_via_read_tool_pdf_page_render"
    assert p["table_2_journal_page"] == 251
    assert p["table_2_pdf_page"] == 26
    assert p["equations_4_30_to_4_45_journal_pages"] == (249, 253)
    assert len(p["spot_checked_values"]) >= 12


def test_braginskii_z1_coefficients_present() -> None:
    p = BRAGINSKII_1965_TRANSPORT_EXTRACTION
    z1_entries = [v for v in p["spot_checked_values"] if v["Z"] == 1]
    assert len(z1_entries) >= 5
    coeff_names = {entry["coefficient"] for entry in z1_entries}
    assert "alpha_0" in coeff_names
    assert "beta_0" in coeff_names
    assert "gamma_0" in coeff_names


def test_scholz_gribkov_pf1000_full_energy_scope() -> None:
    p = SCHOLZ_GRIBKOV_2007_PARTII_EXTRACTION
    assert p["scope_tag"] == "pf1000_full_energy"
    assert p["device_operating_point"]["charge_voltage_kV"] == 35
    aniso = p["targets"]["5_counter_neutron_anisotropy"]
    assert aniso["Y0_over_Y90"] == pytest.approx(1.8)


def test_bernard_1977_thermonuclear_prefactor_not_found() -> None:
    p = BERNARD_1977_HISTORICAL_EXTRACTION
    assert p["thermonuclear_one_over_four_prefactor_found_here"] is False
    assert p["scope_tag"] == "historical_mather_wrong_scope"
    ti = p["targets"]["filament_phase_Ti_thomson"]
    assert ti["phase"] == "filament_NOT_pinch"
    assert ti["runtime_use_permitted"] is False


def test_ucsd_beg_row_6_corrected_line_ranges() -> None:
    p = UCSD_BEG_CURRENT_SHEATH_EXTRACTION
    assert p["audit_row_6_line_range_correction_confirmed"] is True
    assert p["targets"]["massf_formula"]["kr_lines"] == (597, 601)
    assert p["targets"]["pressure_regime_boundaries"]["kr_lines"] == (631, 640)
    assert p["targets"]["Te_4_eV_breakdown"]["kr_lines"] == (642, 644)
    assert p["targets"]["Liz_over_Li_ratio"]["kr_lines"] == (654, 660)


def test_ucsd_beg_massf_values_match_paper() -> None:
    massf = UCSD_BEG_CURRENT_SHEATH_EXTRACTION["targets"]["massf_formula"]
    assert massf["example_values_torr_to_massf"][2] == pytest.approx(0.28)
    assert massf["example_values_torr_to_massf"][4] == pytest.approx(0.20)
    assert massf["example_values_torr_to_massf"][6] == pytest.approx(0.16)


def test_stepniewski_stays_blocked() -> None:
    p = STEPNIEWSKI_2004_REVIEW_PACKET
    assert p["verdict"] == "stays_blocked_simulation_context_only"
    assert p["blocker_id"] == "PF1000-BLK-009"
    # Miklaszewski 2001 free-PDF URL must be present and HTTP-200-verified.
    candidates = p["external_acquisition_to_close"]
    miklaszewski = next(c for c in candidates if c["author"].startswith("Miklaszewski"))
    assert miklaszewski["free_pdf_url"] is not None
    assert miklaszewski["verified_http_200"] is True


def test_plasma_focus_update_audit_correction_ff1() -> None:
    p = PLASMA_FOCUS_UPDATE_2021_EXTRACTION
    assert p["audit_correction_320_500_kev_is_ff1_not_pf1000"] is True
    ff1 = p["targets"]["ff1_beam_target_NOT_pf1000"]
    assert ff1["scope"] == "cross_device_comparator_only"
    assert ff1["explicit_NOT_pf1000"] is True
    assert ff1["mean_deuteron_keV"] == 320
    assert ff1["max_deuteron_keV"] == 500


def test_plasma_focus_update_te_filter_ratio_text_only() -> None:
    te = PLASMA_FOCUS_UPDATE_2021_EXTRACTION["targets"]["te_filter_ratio_zaloga_2018"]
    assert te["scope"] == "TEXT_ONLY_METHOD_CONTEXT"
    assert te["accepted_runtime_claim"] is False
    caveats = te["method_caveats"]
    assert any("D2+Ne" in c for c in caveats)
    assert any("filter-ratio" in c for c in caveats)


def test_plasma_focus_update_third_cathode_cage_hardware_source() -> None:
    geom = PLASMA_FOCUS_UPDATE_2021_EXTRACTION["targets"]["pf1000_geometry_summary"]
    assert geom["cathode_cage_radius_mm"] == 200
    assert geom["cathode_count"] == 12
    assert geom["anode_diameter_mm"] == 230


def test_audit_corrections_folded_in() -> None:
    manifest = sprint5_local_target_extractions()
    folded = set(manifest["audit_corrections_folded_in"])
    required = {
        "audit_row_6_ucsd_beg_line_ranges",
        "audit_row_7_bennett_71pct_at_1us_not_500ns",
        "audit_row_8_braginskii_render_verified_via_read_tool_pdf_pages",
        "bernard_1977_thermonuclear_prefactor_verified_not_found",
        "plasma_focus_update_320_500_kev_is_ff1_not_pf1000",
    }
    assert required.issubset(folded)


def test_sprint5_audit_a1_per_target_resolves_subset_of_top_level() -> None:
    """Codex Sprint 5 WS2 audit A1: every per-target ``resolves`` tuple must
    be a subset of the packet's top-level ``resolves_blockers`` tuple.

    Corroborative-only targets must carry ``corroborative_only = True`` and
    their ``resolves`` tuple must be empty; their corroborative-for blockers
    are recorded in a separate ``corroborative_for`` field and are not
    counted against the top-level resolves set.
    """
    for source_id, packet in SPRINT_5_TARGET_EXTRACTIONS.items():
        top_level = set(packet.get("resolves_blockers", ()))
        targets = packet.get("targets", {})
        if not isinstance(targets, dict):
            continue
        for target_name, target in targets.items():
            if not isinstance(target, dict):
                continue
            per_target = set(target.get("resolves", ()))
            corroborative = target.get("corroborative_only", False)
            if corroborative:
                assert per_target == set(), (
                    f"{source_id}.targets[{target_name!r}] is "
                    f"corroborative_only but has non-empty resolves "
                    f"{per_target}; corroborative entries must use "
                    "corroborative_for and leave resolves empty so "
                    "downstream extractors cannot claim resolution."
                )
                continue
            # Non-corroborative per-target resolves must be a subset of
            # the top-level resolves_blockers set. Drop any qualifier
            # suffix "(historical context only)" before the comparison.
            top_level_normalized = {
                entry.split(" (")[0] for entry in top_level
            }
            per_target_normalized = {
                entry.split(" (")[0] for entry in per_target
            }
            extra = per_target_normalized - top_level_normalized
            assert not extra, (
                f"{source_id}.targets[{target_name!r}].resolves contains "
                f"blockers {extra} that are NOT in the packet's "
                f"top-level resolves_blockers {top_level}; either add "
                "them to the top level or mark the target corroborative_only."
            )


def test_sprint5_audit_a2_free_acquisitions_memo_no_broad_te_ti_wording() -> None:
    """Codex Sprint 5 WS2 audit A2: the Sprint 5 free-acquisition memo must
    NOT assert the broad "no DPF in any literature" wording about Te/Ti.

    The accepted narrow statement is "no accepted same-scope PF-1000 bulk
    pinch Te/Ti history exists for the selected certificate scope." Bernard
    1977 has filament-phase Ti (wrong-scope); Plasma Focus Update 2021 has
    PF-1000 local hot-spot Te method context (text-only). The broad
    "no DPF in any literature" phrasing overstates the field-wide claim.
    """
    text = FREE_ACQ_MEMO.read_text(encoding="utf-8")
    assert "no DPF in any literature" not in text, (
        "Sprint 5 free-acquisition memo contains the broad "
        "'no DPF in any literature' wording. Replace with the narrow "
        "Codex-accepted statement: 'no accepted same-scope PF-1000 bulk "
        "pinch Te/Ti history exists for the selected certificate scope'."
    )
    # Affirmative check: the narrow statement is present.
    narrow_present = (
        "no accepted same-scope PF-1000 bulk pinch" in text
        or "no accepted same-scope PF-1000 bulk\npinch" in text
    )
    assert narrow_present, (
        "Sprint 5 free-acquisition memo missing the Codex-accepted narrow "
        "Te/Ti statement"
    )


def test_sprint5_audit_a3_free_acquisitions_memo_softens_closes_language() -> None:
    """Codex Sprint 5 WS2 audit A3: the memo must not promise that a
    download by itself 'closes' a blocker.

    Acceptable closure language is "may close source availability after
    acquisition, KR ingestion, target extraction, and review." This test
    normalizes whitespace (markdown line wrapping) before checking that
    the explicit qualifier appears in the memo.
    """
    raw = FREE_ACQ_MEMO.read_text(encoding="utf-8")
    # Normalize whitespace so markdown line breaks don't fool the substring check.
    normalized = " ".join(raw.split())
    qualifier_hyphen = (
        "may close source-availability after acquisition, "
        "KR ingestion, target extraction, and review"
    )
    qualifier_space = (
        "may close source availability after acquisition, "
        "KR ingestion, target extraction, and review"
    )
    found = qualifier_hyphen in normalized or qualifier_space in normalized
    assert found, (
        "Sprint 5 free-acquisition memo is missing the narrow closure "
        "qualifier; add language stating that a download may close "
        "source availability only after acquisition, KR ingestion, "
        "target extraction, and review."
    )


def test_sprint5_audit_a1_bennett_corroborative_fill_pressure() -> None:
    """Bennett's fill_pressure_baseline target is explicitly corroborative
    for STARTUP-BVP-CH01 and must NOT claim to resolve any blocker.
    """
    target = BENNETT_2017_STARTUP_EXTRACTION["targets"]["fill_pressure_baseline"]
    assert target.get("corroborative_only") is True
    assert target.get("resolves") == ()
    assert "STARTUP-BVP-CH01" in target.get("corroborative_for", ())
    # And top-level resolves_blockers must NOT include CH01.
    assert "STARTUP-BVP-CH01" not in BENNETT_2017_STARTUP_EXTRACTION["resolves_blockers"]
