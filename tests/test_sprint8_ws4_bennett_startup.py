"""Sprint 8 WS4 — Bennett 2017 startup BVP candidate channel tests.

Controlling doc: docs/SPRINT8_SUPER_SPRINT_SOURCE_TO_RUNTIME_INSTRUCTIONS_2026_05_20.md §WS4
KR record: KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md
Source packet: src/dpf/first_principles/sprint8_bennett_startup_target_extraction.py

Exit criteria verified by this file:
  [x] CH03/CH04/CH07/CH08 are source-backed runtime candidate channels
  [x] whole-shot startup acceptance remains blocked
  [x] unit conversions enforced (SI <-> stated-unit round trips)
  [x] startup packet cites the KR record, not the misnamed raw PDF
  [x] Bennett-derived channels are CANDIDATE; same-scope status for
      pf1000_full_energy_27_to_40_kv is blocked_wrong_scope
  [x] PDF SHA-256 matches the value recorded in Sprint 7 WS-E
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dpf.first_principles.sprint8_bennett_startup_target_extraction import (
    BENNETT_2017_DEMONSTRATOR_SAME_SCOPE_STATUS,
    BENNETT_2017_DEMONSTRATOR_SCOPE,
    BENNETT_2017_KR_PATH,
    BENNETT_2017_PDF_SHA256,
    BENNETT_2017_SCOPE_TAG,
    BENNETT_CH03_SEED_DENSITY,
    BENNETT_CH04_BREAKDOWN_DELAY,
    BENNETT_CH07_EXPLOSIVE_EMISSION,
    BENNETT_CH08_SHEATH_IONIZATION,
    BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE,
    CH03_N_SEED_CM3,
    CH03_N_SEED_M3,
    CH04_T_BREAKDOWN_NS,
    CH04_T_BREAKDOWN_S,
    CH07_E_THRESHOLD_BULK_KV_CM,
    CH07_E_THRESHOLD_BULK_V_M,
    CH07_E_THRESHOLD_KNIFE_KV_CM,
    CH07_E_THRESHOLD_KNIFE_V_M,
    CH07_TE_EV_HIGH,
    CH07_TE_EV_LOW,
    CH07_TE_K_HIGH,
    CH07_TE_K_LOW,
    CH08_F_SHEATH,
    CH08_N_INSULATOR_400NS_CM3,
    CH08_N_INSULATOR_400NS_M3,
    CH08_N_IONIZ_100NS_CM3,
    CH08_N_IONIZ_100NS_M3,
    CH08_T_REF_S,
    K_to_ev,
    Pa_to_torr,
    cm3_to_m3,
    ev_to_K,
    kv_cm_to_v_m,
    m3_to_cm3,
    ns_to_s,
    s_to_ns,
    s_to_us,
    sprint8_ws4_bennett_startup_packet,
    torr_to_Pa,
    us_to_s,
    v_m_to_kv_cm,
    verify_pdf_sha256,
)

# Relative tolerance for floating-point round-trip tests.
_RTOL = 1.0e-9

# Repository root for file-existence tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Group 1: Source identity and KR citation
# ---------------------------------------------------------------------------


def test_packet_cites_kr_not_raw_pdf_filename() -> None:
    """The packet must cite the canonical KR record slug, not the mislabeled PDF name.

    The on-disk PDF is named 'schmidt-2017-kinetic-dpf-breakdown.pdf'; the actual
    first author is Bennett. The KR canonical path uses the correct slug
    'bennett-2017-kinetic-dpf-breakdown'. The packet must NOT contain the misnamed
    filename as its primary source reference.
    """
    packet = sprint8_ws4_bennett_startup_packet()
    assert "bennett-2017-kinetic-dpf-breakdown" in packet["kr_path"], (
        "packet kr_path must reference canonical KR slug (bennett-2017-...)"
    )
    assert "schmidt" not in packet["kr_path"].lower(), (
        "packet kr_path must not reference the mislabeled filename (schmidt-2017-...)"
    )
    # PDF path records the actual on-disk name for acquisition traceability.
    assert "schmidt-2017-kinetic-dpf-breakdown.pdf" in packet["pdf_path"], (
        "packet pdf_path must record the actual on-disk filename for acquisition traceability"
    )


def test_kr_record_exists_on_disk() -> None:
    """The KR canonical record must exist as a file in KnowledgeReference/."""
    kr_file = _REPO_ROOT / BENNETT_2017_KR_PATH
    assert kr_file.exists(), (
        f"KR canonical record not found: {kr_file}\n"
        "Sprint 8 WS4 requires KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md"
    )
    assert kr_file.stat().st_size > 0, "KR record file must not be empty"


def test_kr_record_path_matches_module_constant() -> None:
    """BENNETT_2017_KR_PATH must point to the actual KR file created by WS4."""
    assert BENNETT_2017_KR_PATH == "KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md"


def test_pdf_sha256_matches_sprint7_wse_record() -> None:
    """PDF SHA-256 must match the value recorded in Sprint 7 WS-E line/page verification.

    Sprint 7 WS-E computed: c5e6f5f1e2ca150a41c18c83f82a2fbaf35a2deb75d4a50b60cb7a45b0f0b92a
    (docs/extractions/SPRINT7_WSE_NEXT_PHYSICS_SOURCE_PACKETS_2026_05_20.md §Packet 2)
    """
    assert BENNETT_2017_PDF_SHA256 == (
        "c5e6f5f1e2ca150a41c18c83f82a2fbaf35a2deb75d4a50b60cb7a45b0f0b92a"
    ), "PDF SHA-256 must match Sprint 7 WS-E verified value"


def test_pdf_sha256_verified_against_on_disk_file() -> None:
    """Compute the SHA-256 of the on-disk PDF and verify it matches the KR record.

    This is a live verification (not just constant equality) — it fails if the file
    has been modified or corrupted since Sprint 7 WS-E ingestion.
    """
    result = verify_pdf_sha256(repo_root=_REPO_ROOT)
    assert result["match"], (
        f"Bennett 2017 PDF SHA-256 mismatch.\n"
        f"  File: {result['path']}\n"
        f"  Expected: {result['expected']}\n"
        f"  Computed: {result['computed']}\n"
        "The PDF may have been replaced or corrupted since Sprint 7 WS-E verification."
    )


def test_scope_tag_is_pf1000_generic() -> None:
    """Scope tag must be pf1000_generic (not Akel-specific, not pf1000_full_energy)."""
    assert BENNETT_2017_SCOPE_TAG == "pf1000_generic"
    packet = sprint8_ws4_bennett_startup_packet()
    assert packet["scope_tag"] == "pf1000_generic"


def test_demonstrator_scope_and_same_scope_status() -> None:
    """Demonstrator scope must be pf1000_full_energy_27_to_40_kv with blocked_wrong_scope."""
    assert BENNETT_2017_DEMONSTRATOR_SCOPE == "pf1000_full_energy_27_to_40_kv"
    assert BENNETT_2017_DEMONSTRATOR_SAME_SCOPE_STATUS == "blocked_wrong_scope"
    packet = sprint8_ws4_bennett_startup_packet()
    assert packet["demonstrator_scope"] == "pf1000_full_energy_27_to_40_kv"
    assert packet["demonstrator_same_scope_status"] == "blocked_wrong_scope"


# ---------------------------------------------------------------------------
# Group 2: Unit conversion round-trips (enforced, not just nominal)
# ---------------------------------------------------------------------------


def test_cm3_to_m3_round_trip() -> None:
    """cm⁻³ → m⁻³ → cm⁻³ must be exact (1 cm⁻³ = 1e6 m⁻³)."""
    for val_cm3 in [1.0, 1e7, 3.52e17]:
        val_m3 = cm3_to_m3(val_cm3)
        recovered = m3_to_cm3(val_m3)
        assert math.isclose(recovered, val_cm3, rel_tol=_RTOL), (
            f"cm3_to_m3 round trip failed for {val_cm3}: got {recovered}"
        )


def test_ns_to_s_round_trip() -> None:
    """ns → s → ns must be exact (1 ns = 1e-9 s)."""
    for val_ns in [1.0, 20.0, 125.0]:
        val_s = ns_to_s(val_ns)
        recovered = s_to_ns(val_s)
        assert math.isclose(recovered, val_ns, rel_tol=_RTOL), (
            f"ns_to_s round trip failed for {val_ns}: got {recovered}"
        )


def test_kv_cm_to_v_m_round_trip() -> None:
    """kV/cm → V/m → kV/cm must be exact (1 kV/cm = 1e5 V/m)."""
    for val_kv_cm in [1.0, 10.0, 250.0]:
        val_v_m = kv_cm_to_v_m(val_kv_cm)
        recovered = v_m_to_kv_cm(val_v_m)
        assert math.isclose(recovered, val_kv_cm, rel_tol=_RTOL), (
            f"kv_cm_to_v_m round trip failed for {val_kv_cm}: got {recovered}"
        )


def test_ev_to_K_round_trip() -> None:
    """eV → K → eV must be consistent (1 eV = 11604.52 K)."""
    for val_ev in [3.5, 4.0, 1.0]:
        val_K = ev_to_K(val_ev)
        recovered = K_to_ev(val_K)
        assert math.isclose(recovered, val_ev, rel_tol=_RTOL), (
            f"ev_to_K round trip failed for {val_ev}: got {recovered}"
        )


def test_torr_to_Pa_round_trip() -> None:
    """Torr → Pa → Torr must be consistent (1 Torr = 133.322 Pa)."""
    for val_torr in [1.0, 5.5, 15.0]:
        val_Pa = torr_to_Pa(val_torr)
        recovered = Pa_to_torr(val_Pa)
        assert math.isclose(recovered, val_torr, rel_tol=_RTOL), (
            f"torr_to_Pa round trip failed for {val_torr}: got {recovered}"
        )


def test_us_to_s_round_trip() -> None:
    """µs → s → µs must be exact (1 µs = 1e-6 s)."""
    for val_us in [0.1, 1.0, 10.0]:
        val_s = us_to_s(val_us)
        recovered = s_to_us(val_s)
        assert math.isclose(recovered, val_us, rel_tol=_RTOL), (
            f"us_to_s round trip failed for {val_us}: got {recovered}"
        )


def test_kv_cm_conversion_factor() -> None:
    """1 kV/cm = 1e5 V/m — verify the exact factor."""
    assert math.isclose(kv_cm_to_v_m(1.0), 1.0e5, rel_tol=_RTOL)
    assert math.isclose(kv_cm_to_v_m(250.0), 2.5e7, rel_tol=_RTOL)
    assert math.isclose(kv_cm_to_v_m(10.0), 1.0e6, rel_tol=_RTOL)


def test_cm3_conversion_factor() -> None:
    """1 cm⁻³ = 1e6 m⁻³ — verify the exact factor."""
    assert math.isclose(cm3_to_m3(1.0), 1.0e6, rel_tol=_RTOL)
    assert math.isclose(cm3_to_m3(1.0e7), 1.0e13, rel_tol=_RTOL)
    assert math.isclose(cm3_to_m3(1.0e15), 1.0e21, rel_tol=_RTOL)


# ---------------------------------------------------------------------------
# Group 3: CH03 seed density values and units
# ---------------------------------------------------------------------------


def test_ch03_seed_density_stated_value() -> None:
    """CH03: n_seed stated value must be 1e7 cm⁻³ (Bennett p.2 verbatim)."""
    assert CH03_N_SEED_CM3 == 1.0e7
    assert BENNETT_CH03_SEED_DENSITY["values"]["n_seed"]["stated_value"] == 1.0e7
    assert BENNETT_CH03_SEED_DENSITY["values"]["n_seed"]["stated_units"] == "cm^-3"


def test_ch03_seed_density_si_conversion() -> None:
    """CH03: n_seed SI form must be 1e13 m⁻³ (1e7 cm⁻³ × 1e6)."""
    assert math.isclose(CH03_N_SEED_M3, 1.0e13, rel_tol=_RTOL)
    assert math.isclose(
        BENNETT_CH03_SEED_DENSITY["values"]["n_seed"]["si_value"], 1.0e13, rel_tol=_RTOL
    )
    assert BENNETT_CH03_SEED_DENSITY["values"]["n_seed"]["si_units"] == "m^-3"


def test_ch03_channel_is_candidate_not_accepted() -> None:
    """CH03: must be source-backed CANDIDATE, not accepted."""
    assert BENNETT_CH03_SEED_DENSITY["source_backed_candidate"] is True
    assert BENNETT_CH03_SEED_DENSITY["accepted_runtime_claim"] is False
    assert BENNETT_CH03_SEED_DENSITY["can_support_first_principles_acceptance"] is False


def test_ch03_same_scope_status_blocked_wrong_scope() -> None:
    """CH03: same-scope status for pf1000_full_energy_27_to_40_kv must be blocked_wrong_scope."""
    assert (
        BENNETT_CH03_SEED_DENSITY["same_scope_status_for_demonstrator"]
        == "blocked_wrong_scope"
    )
    assert BENNETT_CH03_SEED_DENSITY["demonstrator_scope"] == "pf1000_full_energy_27_to_40_kv"


def test_ch03_cites_kr_record() -> None:
    """CH03: kr_path must reference the canonical KR markdown, not the raw PDF name."""
    assert "bennett-2017-kinetic-dpf-breakdown" in BENNETT_CH03_SEED_DENSITY["kr_path"]
    assert "schmidt" not in BENNETT_CH03_SEED_DENSITY["kr_path"]


# ---------------------------------------------------------------------------
# Group 4: CH04 breakdown delay values and units
# ---------------------------------------------------------------------------


def test_ch04_breakdown_delay_stated_value() -> None:
    """CH04: t_breakdown stated value must be 20 ns (Bennett p.4 verbatim)."""
    assert CH04_T_BREAKDOWN_NS == 20.0
    assert (
        BENNETT_CH04_BREAKDOWN_DELAY["values"]["t_breakdown"]["stated_value"] == 20.0
    )
    assert (
        BENNETT_CH04_BREAKDOWN_DELAY["values"]["t_breakdown"]["stated_units"] == "ns"
    )


def test_ch04_breakdown_delay_si_conversion() -> None:
    """CH04: t_breakdown SI form must be 2e-8 s (20 ns × 1e-9)."""
    assert math.isclose(CH04_T_BREAKDOWN_S, 2.0e-8, rel_tol=_RTOL)
    assert math.isclose(
        BENNETT_CH04_BREAKDOWN_DELAY["values"]["t_breakdown"]["si_value"],
        2.0e-8,
        rel_tol=_RTOL,
    )
    assert BENNETT_CH04_BREAKDOWN_DELAY["values"]["t_breakdown"]["si_units"] == "s"


def test_ch04_breakdown_delay_qualifier() -> None:
    """CH04: breakdown delay qualifier must be 'approximate' (source says 'approximately 20 ns')."""
    assert (
        BENNETT_CH04_BREAKDOWN_DELAY["values"]["t_breakdown"]["qualifier"] == "approximate"
    )


def test_ch04_pressure_regimes_present() -> None:
    """CH04: pressure_regimes must enumerate low/medium/high breakdown modes."""
    regimes = BENNETT_CH04_BREAKDOWN_DELAY["pressure_regimes"]
    assert "low" in regimes
    assert "medium" in regimes
    assert "high" in regimes
    assert "15 Torr" in regimes["high"]["criterion"]


def test_ch04_channel_is_candidate_blocked_wrong_scope() -> None:
    """CH04: source-backed candidate; same-scope blocked_wrong_scope."""
    assert BENNETT_CH04_BREAKDOWN_DELAY["source_backed_candidate"] is True
    assert BENNETT_CH04_BREAKDOWN_DELAY["accepted_runtime_claim"] is False
    assert BENNETT_CH04_BREAKDOWN_DELAY["same_scope_status_for_demonstrator"] == "blocked_wrong_scope"


# ---------------------------------------------------------------------------
# Group 5: CH07 explosive emission thresholds and Te
# ---------------------------------------------------------------------------


def test_ch07_bulk_threshold_stated_value() -> None:
    """CH07: E_threshold_bulk stated value must be 250 kV/cm (Bennett p.3 verbatim)."""
    assert CH07_E_THRESHOLD_BULK_KV_CM == 250.0
    val = BENNETT_CH07_EXPLOSIVE_EMISSION["values"]["E_threshold_bulk"]
    assert val["stated_value"] == 250.0
    assert val["stated_units"] == "kV/cm"


def test_ch07_bulk_threshold_si_conversion() -> None:
    """CH07: E_threshold_bulk SI form must be 2.5e7 V/m (250 kV/cm × 1e5)."""
    assert math.isclose(CH07_E_THRESHOLD_BULK_V_M, 2.5e7, rel_tol=_RTOL)
    val = BENNETT_CH07_EXPLOSIVE_EMISSION["values"]["E_threshold_bulk"]
    assert math.isclose(val["si_value"], 2.5e7, rel_tol=_RTOL)
    assert val["si_units"] == "V/m"


def test_ch07_knife_threshold_stated_value() -> None:
    """CH07: E_threshold_knife stated value must be 10 kV/cm (Bennett p.3 verbatim)."""
    assert CH07_E_THRESHOLD_KNIFE_KV_CM == 10.0
    val = BENNETT_CH07_EXPLOSIVE_EMISSION["values"]["E_threshold_knife"]
    assert val["stated_value"] == 10.0
    assert val["stated_units"] == "kV/cm"


def test_ch07_knife_threshold_si_conversion() -> None:
    """CH07: E_threshold_knife SI form must be 1e6 V/m (10 kV/cm × 1e5)."""
    assert math.isclose(CH07_E_THRESHOLD_KNIFE_V_M, 1.0e6, rel_tol=_RTOL)
    val = BENNETT_CH07_EXPLOSIVE_EMISSION["values"]["E_threshold_knife"]
    assert math.isclose(val["si_value"], 1.0e6, rel_tol=_RTOL)
    assert val["si_units"] == "V/m"


def test_ch07_te_range_lies_within_3p5_to_4p0_ev() -> None:
    """CH07: T_e initial condition must lie in [3.5, 4.0] eV (Bennett p.5 verbatim).

    Per Sprint 7 WS-E verification: Fig. 8 on p.5 plots 3.5-eV and 4-eV Maxwellian
    curves, confirming the (3.5, 4.0) eV range.
    """
    assert CH07_TE_EV_LOW == 3.5
    assert CH07_TE_EV_HIGH == 4.0
    # Range must be [3.5, 4.0] eV — not below 3.5, not above 4.0.
    assert CH07_TE_EV_LOW >= 3.5
    assert CH07_TE_EV_HIGH <= 4.0
    val = BENNETT_CH07_EXPLOSIVE_EMISSION["values"]["T_e"]
    low_ev, high_ev = val["stated_range_eV"]
    assert math.isclose(low_ev, 3.5, rel_tol=_RTOL)
    assert math.isclose(high_ev, 4.0, rel_tol=_RTOL)


def test_ch07_te_si_conversion_in_kelvin() -> None:
    """CH07: T_e SI values must be consistent with eV → K conversion (1 eV = 11604.52 K)."""
    expected_low_K = 3.5 * 11604.5221
    expected_high_K = 4.0 * 11604.5221
    assert math.isclose(CH07_TE_K_LOW, expected_low_K, rel_tol=1.0e-5)
    assert math.isclose(CH07_TE_K_HIGH, expected_high_K, rel_tol=1.0e-5)
    val = BENNETT_CH07_EXPLOSIVE_EMISSION["values"]["T_e"]
    assert math.isclose(val["si_range_K"][0], expected_low_K, rel_tol=1.0e-5)
    assert math.isclose(val["si_range_K"][1], expected_high_K, rel_tol=1.0e-5)
    assert val["si_units"] == "K"


def test_ch07_te_is_startup_model_context_only() -> None:
    """CH07: T_e scope caveat must indicate 'startup model context only'."""
    val = BENNETT_CH07_EXPLOSIVE_EMISSION["values"]["T_e"]
    assert "startup model context" in val["scope_caveat"].lower()
    assert "not an accepted runtime claim" in val["scope_caveat"].lower()


def test_ch07_channel_is_candidate_blocked_wrong_scope() -> None:
    """CH07: source-backed candidate; same-scope blocked_wrong_scope."""
    assert BENNETT_CH07_EXPLOSIVE_EMISSION["source_backed_candidate"] is True
    assert BENNETT_CH07_EXPLOSIVE_EMISSION["accepted_runtime_claim"] is False
    assert BENNETT_CH07_EXPLOSIVE_EMISSION["same_scope_status_for_demonstrator"] == "blocked_wrong_scope"


# ---------------------------------------------------------------------------
# Group 6: CH08 sheath current fraction and ionization landmarks
# ---------------------------------------------------------------------------


def test_ch08_sheath_current_fraction_value() -> None:
    """CH08: f_sheath must be 0.71 at 1 µs (Bennett p.3 verbatim, audit row-7 confirmed)."""
    assert math.isclose(CH08_F_SHEATH, 0.71, rel_tol=_RTOL)
    val = BENNETT_CH08_SHEATH_IONIZATION["values"]["f_sheath"]
    assert math.isclose(val["stated_value_percent"], 71.0, rel_tol=_RTOL)
    assert math.isclose(val["si_value_fraction"], 0.71, rel_tol=_RTOL)
    assert val["si_units"] == "dimensionless"


def test_ch08_sheath_current_at_1_us_not_500_ns() -> None:
    """CH08: f_sheath reference time must be 1 µs, NOT 500 ns (audit row-7 correction).

    The 500 ns entry (Fig. 4(b)) gives only channel formation; the 71% current
    fraction is unambiguously at t = 1 µs (Fig. 4(c)) per Sprint 7 WS-E verification.
    """
    val = BENNETT_CH08_SHEATH_IONIZATION["values"]["f_sheath"]
    assert math.isclose(val["at_time_us"], 1.0, rel_tol=_RTOL), (
        "f_sheath reference time must be 1 µs; 500 ns attribution is Codex "
        "audit row-7 error (confirmed corrected in Sprint 5 and Sprint 7 WS-E)"
    )
    assert math.isclose(CH08_T_REF_S, 1.0e-6, rel_tol=_RTOL)


def test_ch08_ionization_100ns_stated_and_si() -> None:
    """CH08: n_ioniz_100ns stated 1e13 cm⁻³ → SI 1e19 m⁻³."""
    assert CH08_N_IONIZ_100NS_CM3 == 1.0e13
    assert math.isclose(CH08_N_IONIZ_100NS_M3, 1.0e19, rel_tol=_RTOL)
    val = BENNETT_CH08_SHEATH_IONIZATION["values"]["n_ioniz_100ns"]
    assert val["stated_value"] == 1.0e13
    assert math.isclose(val["si_value"], 1.0e19, rel_tol=_RTOL)
    assert val["stated_units"] == "cm^-3"
    assert val["si_units"] == "m^-3"
    assert val["qualifier"] == "order of magnitude"


def test_ch08_insulator_400ns_stated_and_si() -> None:
    """CH08: n_insulator_400ns stated 1e15 cm⁻³ → SI 1e21 m⁻³."""
    assert CH08_N_INSULATOR_400NS_CM3 == 1.0e15
    assert math.isclose(CH08_N_INSULATOR_400NS_M3, 1.0e21, rel_tol=_RTOL)
    val = BENNETT_CH08_SHEATH_IONIZATION["values"]["n_insulator_400ns"]
    assert val["stated_value"] == 1.0e15
    assert math.isclose(val["si_value"], 1.0e21, rel_tol=_RTOL)


def test_ch08_audit_row7_correction_note_present() -> None:
    """CH08: audit row-7 correction note must be present in the channel record."""
    assert "audit_row7_note" in BENNETT_CH08_SHEATH_IONIZATION
    note = BENNETT_CH08_SHEATH_IONIZATION["audit_row7_note"]
    assert "1" in note and ("µs" in note or "us" in note.lower()), (
        "audit_row7_note must reference 1 µs timing"
    )
    assert "500" in note, "audit_row7_note must reference the corrected 500 ns error"


def test_ch08_channel_is_candidate_blocked_wrong_scope() -> None:
    """CH08: source-backed candidate; same-scope blocked_wrong_scope."""
    assert BENNETT_CH08_SHEATH_IONIZATION["source_backed_candidate"] is True
    assert BENNETT_CH08_SHEATH_IONIZATION["accepted_runtime_claim"] is False
    assert BENNETT_CH08_SHEATH_IONIZATION["same_scope_status_for_demonstrator"] == "blocked_wrong_scope"


# ---------------------------------------------------------------------------
# Group 7: Channels kept blocked/wrong-scope (CH01/02/05/06/09/10/11/12/13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ch_id", ["CH01", "CH02", "CH05", "CH06", "CH09", "CH10", "CH11", "CH12", "CH13"])
def test_non_candidate_channels_are_blocked(ch_id: str) -> None:
    """CH01/02/05/06/09/10/11/12/13 must be blocked or wrong-scope, not candidate."""
    assert ch_id in BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE, (
        f"{ch_id} not listed in BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE"
    )
    record = BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE[ch_id]
    status = record["status"]
    assert status in {"blocked_wrong_scope", "blocked_missing_source"}, (
        f"{ch_id} has unexpected status '{status}'; must be blocked"
    )


def test_ch01_is_wrong_scope() -> None:
    """CH01: fill pressure channel is blocked_wrong_scope (MA-scale 5.5 Torr vs Akel scope)."""
    assert BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE["CH01"]["status"] == "blocked_wrong_scope"


def test_ch06_is_blocked_photoemission_neglected() -> None:
    """CH06: photoemission is blocked_missing_source (Bennett explicitly neglects photons)."""
    assert BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE["CH06"]["status"] == "blocked_missing_source"
    reason = BENNETT_CHANNELS_BLOCKED_OR_WRONG_SCOPE["CH06"]["reason"].lower()
    assert "neglect" in reason or "neglected" in reason, (
        "CH06 reason must note that photons are neglected in Bennett 2017"
    )


# ---------------------------------------------------------------------------
# Group 8: Whole-shot startup remains blocked
# ---------------------------------------------------------------------------


def test_whole_shot_startup_blocked_in_packet() -> None:
    """Whole-shot startup acceptance must remain blocked in the WS4 packet."""
    packet = sprint8_ws4_bennett_startup_packet()
    assert packet["whole_shot_startup_blocked"] is True
    assert packet["accepted_runtime_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_whole_shot_startup_blocked_in_channel_records() -> None:
    """Every candidate channel record must carry whole_shot_startup_blocked=True."""
    for ch_id, record in [
        ("CH03", BENNETT_CH03_SEED_DENSITY),
        ("CH04", BENNETT_CH04_BREAKDOWN_DELAY),
        ("CH07", BENNETT_CH07_EXPLOSIVE_EMISSION),
        ("CH08", BENNETT_CH08_SHEATH_IONIZATION),
    ]:
        assert record["whole_shot_startup_blocked"] is True, (
            f"{ch_id}: whole_shot_startup_blocked must be True"
        )
        assert record["accepted_runtime_claim"] is False, (
            f"{ch_id}: accepted_runtime_claim must be False"
        )
        assert record["can_support_first_principles_acceptance"] is False, (
            f"{ch_id}: can_support_first_principles_acceptance must be False"
        )


def test_bennett_candidate_channels_do_not_lift_startup_bvp_acceptance() -> None:
    """Bennett candidate channels must NOT lift startup BVP acceptance via build_startup_bvp_packet.

    Even if CH03/CH04/CH07/CH08 are passed as 'accepted_channels' to the startup BVP
    packet builder, the typed StartupPacket (which has no computed channels) must block
    acceptance. This is the A1 guard: typed packet is the single acceptance authority.
    """
    from dpf.first_principles.startup_bvp import (
        REQUIRED_STARTUP_CHANNELS,
        build_startup_bvp_packet,
    )

    # Pass all required channels as "accepted" plus Bennett candidate IDs.
    bvp = build_startup_bvp_packet(
        {
            "mode": "surface_breakdown_bvp",
            "evidence_status": "reviewed",
            "source_scope": "pf1000_generic",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    assert bvp["can_support_first_principles_acceptance"] is False, (
        "Bennett candidate channels must not lift startup BVP acceptance; "
        "typed StartupPacket is the single acceptance authority (A1)"
    )
    assert bvp["whole_shot_startup_blocked"] is True
    assert bvp["status"] != "accepted_startup_bvp_packet"


# ---------------------------------------------------------------------------
# Group 9: Packet completeness and field presence
# ---------------------------------------------------------------------------


def test_packet_has_four_candidate_channels() -> None:
    """WS4 packet must carry exactly CH03/CH04/CH07/CH08 as candidate channels."""
    packet = sprint8_ws4_bennett_startup_packet()
    candidates = packet["candidate_channels"]
    assert set(candidates.keys()) == {"CH03", "CH04", "CH07", "CH08"}, (
        f"Expected candidate channels {{CH03, CH04, CH07, CH08}}, got {set(candidates.keys())}"
    )


def test_packet_has_nine_blocked_channels() -> None:
    """WS4 packet must carry CH01/02/05/06/09/10/11/12/13 as blocked/wrong-scope."""
    packet = sprint8_ws4_bennett_startup_packet()
    blocked = packet["blocked_or_wrong_scope_channels"]
    assert set(blocked.keys()) == {
        "CH01", "CH02", "CH05", "CH06", "CH09", "CH10", "CH11", "CH12", "CH13"
    }, (
        f"Expected 9 blocked channels, got {set(blocked.keys())}"
    )


def test_packet_already_in_kr_is_true() -> None:
    """already_in_kr must be True after Sprint 8 WS4 KR ingestion."""
    packet = sprint8_ws4_bennett_startup_packet()
    assert packet["already_in_kr"] is True, (
        "already_in_kr must be True after WS4 creates the KR markdown file"
    )


def test_packet_references_controlling_doc() -> None:
    """Packet must reference the Sprint 8 controlling document."""
    packet = sprint8_ws4_bennett_startup_packet()
    assert "SPRINT8_SUPER_SPRINT" in packet["controlling_doc"]
    assert "WS4" in packet["controlling_doc"]


def test_each_candidate_channel_has_required_fields() -> None:
    """Each candidate channel record must carry required traceability fields."""
    required_fields = {
        "channel_id",
        "source_id",
        "kr_path",
        "values",
        "source_backed_candidate",
        "same_scope_status_for_demonstrator",
        "demonstrator_scope",
        "accepted_runtime_claim",
        "can_support_first_principles_acceptance",
        "whole_shot_startup_blocked",
    }
    for ch_id, record in [
        ("CH03", BENNETT_CH03_SEED_DENSITY),
        ("CH04", BENNETT_CH04_BREAKDOWN_DELAY),
        ("CH07", BENNETT_CH07_EXPLOSIVE_EMISSION),
        ("CH08", BENNETT_CH08_SHEATH_IONIZATION),
    ]:
        missing = required_fields - set(record.keys())
        assert not missing, f"{ch_id} is missing required fields: {missing}"


def test_all_candidate_channel_values_have_si_and_stated_units() -> None:
    """Every value in every candidate channel must carry stated and SI unit forms."""
    for ch_id, record in [
        ("CH03", BENNETT_CH03_SEED_DENSITY),
        ("CH04", BENNETT_CH04_BREAKDOWN_DELAY),
        ("CH07", BENNETT_CH07_EXPLOSIVE_EMISSION),
        ("CH08", BENNETT_CH08_SHEATH_IONIZATION),
    ]:
        for val_key, val in record["values"].items():
            has_stated = "stated_value" in val or "stated_value_percent" in val or "stated_value_eV" in val
            has_si = any(k.startswith("si_") for k in val)
            assert has_stated, f"{ch_id}.{val_key}: no stated_value field"
            assert has_si, f"{ch_id}.{val_key}: no si_* field"


# ---------------------------------------------------------------------------
# Group 10: Sprint 9 WS9-4 — Bennett wired into startup runtime as wrong-scope
# candidate context (replaces the Sprint 8 WS4 "do not modify startup_bvp.py"
# guard, which the audit P1-2 explicitly retired).
# ---------------------------------------------------------------------------
#
# Audit P1-2 (CODEX_SUPER_SPRINT8_AUDIT_AND_SUPER_SPRINT9_INSTRUCTIONS_2026_05_20.md)
# asked the lead to apply a startup_bvp.py delta importing the Bennett packet as
# wrong-scope candidate context. The Sprint 8 guard test
# (test_startup_bvp_source_refs_do_not_include_bennett_kr) only proved WS4 did
# not overstep its file boundary; it is now obsolete. These tests assert the
# wiring IS present and that it is wrong-scope candidate context that cannot
# lift acceptance.


def test_startup_bvp_exposes_bennett_wrong_scope_candidate_context() -> None:
    """startup_bvp must expose Bennett CH03/04/07/08 as wrong-scope candidate context.

    WS9-4: build_startup_bvp_packet, when include_bennett_wrong_scope_context is
    requested, surfaces the Sprint 8 WS4 Bennett 2017 candidate channels
    (CH03/CH04/CH07/CH08) as blocked_wrong_scope candidate context for the
    selected full-energy demonstrator (pf1000_full_energy_27_to_40_kv).
    """
    from dpf.first_principles.startup_bvp import build_startup_bvp_packet

    packet = build_startup_bvp_packet(
        {
            "mode": "surface_breakdown_bvp",
            "evidence_status": "reviewed",
            "source_scope": "pf1000_full_energy_27_to_40_kv",
        },
        include_bennett_wrong_scope_context=True,
    )
    context = packet["bennett_wrong_scope_candidate_context"]
    assert context["status"] == "bennett_wrong_scope_candidate_context"
    assert context["source_id"] == "bennett_2017_kinetic_dpf_breakdown"
    assert "bennett-2017-kinetic-dpf-breakdown" in context["kr_path"]
    # Exactly the four Bennett candidate channels are surfaced.
    assert set(context["candidate_channels"].keys()) == {
        "CH03",
        "CH04",
        "CH07",
        "CH08",
    }
    # Each surfaced channel is source-backed AND wrong-scope for the demonstrator.
    for ch_id, record in context["candidate_channels"].items():
        assert record["source_backed_candidate"] is True, (
            f"{ch_id} must be source-backed candidate context"
        )
        assert record["same_scope_status_for_demonstrator"] == "blocked_wrong_scope", (
            f"{ch_id} must stay blocked_wrong_scope for the full-energy demonstrator"
        )
        assert record["demonstrator_scope"] == "pf1000_full_energy_27_to_40_kv"
    assert context["demonstrator_same_scope_status"] == "blocked_wrong_scope"
    assert context["transfer_rule_status"] == "no_reviewed_transfer_rule"


def test_bennett_wrong_scope_context_not_present_unless_requested() -> None:
    """The Bennett context block is opt-in: absent (empty) unless requested.

    Default callers (no include_bennett_wrong_scope_context flag) get an explicit
    'not_requested' marker with no candidate channels, so the demonstrator
    startup packet is unchanged unless the runtime asks for the context.
    """
    from dpf.first_principles.startup_bvp import build_startup_bvp_packet

    packet = build_startup_bvp_packet(
        {
            "mode": "surface_breakdown_bvp",
            "evidence_status": "reviewed",
            "source_scope": "pf1000_full_energy_27_to_40_kv",
        }
    )
    context = packet["bennett_wrong_scope_candidate_context"]
    assert context["status"] == "bennett_wrong_scope_candidate_context_not_requested"
    assert context["candidate_channels"] == {}
    assert context["can_close_startup_bvp_packet"] is False
    assert context["can_support_first_principles_acceptance"] is False


def test_bennett_wrong_scope_context_cannot_lift_startup_acceptance() -> None:
    """Bennett wrong-scope candidate context must NOT produce acceptance.

    WS9-4 non-negotiable: even with the Bennett context requested, all required
    startup channels declared, and an accepted mode + reviewed evidence, the
    startup BVP packet stays blocked. The Bennett context carries explicit
    non-acceptance flags and the typed StartupPacket remains the single
    acceptance authority (no computed channel).
    """
    from dpf.first_principles.startup_bvp import (
        REQUIRED_STARTUP_CHANNELS,
        build_startup_bvp_packet,
    )

    packet = build_startup_bvp_packet(
        {
            "mode": "surface_breakdown_bvp",
            "evidence_status": "accepted_same_scope_source",
            "source_scope": "pf1000_full_energy_27_to_40_kv",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        },
        include_bennett_wrong_scope_context=True,
    )
    # Headline packet stays blocked.
    assert packet["status"] != "accepted_startup_bvp_packet"
    assert packet["status"] == "blocked_startup_bvp_packet_not_available"
    assert packet["whole_shot_startup_blocked"] is True
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_support_whole_shot_acceptance"] is False
    # The Bennett context itself is non-acceptance.
    context = packet["bennett_wrong_scope_candidate_context"]
    assert context["can_close_startup_bvp_packet"] is False
    assert context["can_support_whole_shot_acceptance"] is False
    assert context["can_support_first_principles_acceptance"] is False
    assert context["whole_shot_startup_blocked"] is True
    for record in context["candidate_channels"].values():
        assert record["can_support_startup_bvp_acceptance"] is False
        assert record["accepted_runtime_claim"] is False
        assert record["can_support_first_principles_acceptance"] is False
    # The typed StartupPacket remains blocked (single acceptance authority).
    assert packet["startup_channel_packet"]["can_support_first_principles_acceptance"] is (
        False
    )


def test_startup_bvp_still_blocked_until_selected_scope_startup_packet() -> None:
    """Startup BVP stays blocked: no selected-scope startup packet exists.

    WS9-4 required test: passing the Bennett wrong-scope candidate channels (the
    only source-backed startup channels available) cannot produce a selected-scope
    startup packet. Until a same-scope startup packet exists for
    pf1000_full_energy_27_to_40_kv, the startup BVP remains blocked.
    """
    from dpf.first_principles.startup_bvp import build_startup_bvp_packet

    packet = build_startup_bvp_packet(
        {
            "mode": "imported_pic_sheath_state",
            "evidence_status": "reviewed",
            "source_scope": "pf1000_full_energy_27_to_40_kv",
            "can_support_whole_shot_acceptance": True,
        },
        include_bennett_wrong_scope_context=True,
    )
    assert packet["whole_shot_startup_blocked"] is True
    assert packet["can_support_first_principles_acceptance"] is False
    # Bennett context is present but wrong-scope for this demonstrator.
    context = packet["bennett_wrong_scope_candidate_context"]
    assert context["demonstrator_same_scope_status"] == "blocked_wrong_scope"
