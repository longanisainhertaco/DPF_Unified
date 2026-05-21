"""Sprint 8 WS2 — tests for the runtime demonstrator scope-lock packet.

Enforces:
1. Mixed-scope source sets (in-scope + wrong-scope) fail scope-consistency check.
2. Scope packet is NOT scientific authority (is_scientific_authority=False).
3. Scope packet does NOT carry accepted_runtime_claim=True.
4. Scope packet governance_class == "control_plane".
5. The canonical scope label is the single string used by all artifacts.
"""

from __future__ import annotations

from dpf.first_principles.runtime_demonstrator_scope import (
    CONTEXT_ONLY_SOURCES,
    IN_SCOPE_SOURCES,
    SELECTED_SCOPE_LABEL,
    WRONG_SCOPE_SOURCES,
    check_scope_consistency,
    runtime_demonstrator_scope_packet,
)

# ── canonical scope label ────────────────────────────────────────────────────

def test_selected_scope_label_is_canonical_string() -> None:
    assert SELECTED_SCOPE_LABEL == "pf1000_full_energy_27_to_40_kv"


def test_scope_packet_label_matches_module_constant() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert packet["selected_scope_label"] == SELECTED_SCOPE_LABEL


# ── governance flags ─────────────────────────────────────────────────────────

def test_scope_packet_is_not_scientific_authority() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert packet["is_scientific_authority"] is False


def test_scope_packet_accepted_runtime_claim_false() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert packet["accepted_runtime_claim"] is False


def test_scope_packet_can_support_acceptance_false() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert packet["can_support_first_principles_acceptance"] is False


def test_scope_packet_governance_class_is_control_plane() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert packet["governance_class"] == "control_plane"


# ── source classification — membership ──────────────────────────────────────

def test_in_scope_sources_non_empty() -> None:
    assert len(IN_SCOPE_SOURCES) > 0


def test_wrong_scope_sources_non_empty() -> None:
    assert len(WRONG_SCOPE_SOURCES) > 0


def test_context_only_sources_non_empty() -> None:
    assert len(CONTEXT_ONLY_SOURCES) > 0


def test_in_scope_and_wrong_scope_are_disjoint() -> None:
    overlap = set(IN_SCOPE_SOURCES) & set(WRONG_SCOPE_SOURCES)
    assert overlap == set(), f"Sources appear in both in-scope and wrong-scope: {overlap}"


def test_in_scope_and_context_only_are_disjoint() -> None:
    overlap = set(IN_SCOPE_SOURCES) & set(CONTEXT_ONLY_SOURCES)
    assert overlap == set(), f"Sources appear in both in-scope and context-only: {overlap}"


def test_wrong_scope_and_context_only_are_disjoint() -> None:
    overlap = set(WRONG_SCOPE_SOURCES) & set(CONTEXT_ONLY_SOURCES)
    assert overlap == set(), f"Sources appear in both wrong-scope and context-only: {overlap}"


# ── known source assignments ─────────────────────────────────────────────────

def test_nx2_talebitaher_is_wrong_scope() -> None:
    """NX2 (3 kJ NTU device) must be wrong-scope for full-energy PF-1000."""
    assert "talebitaher_2012_nx2_detector_anisotropy" in WRONG_SCOPE_SOURCES


def test_bernard_1977_is_wrong_scope() -> None:
    """Bernard 1977 historical Mather review must be wrong-scope."""
    assert "bernard_1977_dpf_high_intensity_neutron_source" in WRONG_SCOPE_SOURCES


def test_ucsd_beg_is_wrong_scope() -> None:
    """UCSD/Beg 10 kJ Mather device must be wrong-scope."""
    assert "ucsd_beg_current_sheath_initiation" in WRONG_SCOPE_SOURCES


def test_scholz_2007_partii_is_in_scope() -> None:
    """Gribkov/Scholz 2007 Part II is the primary full-energy PF-1000 source."""
    assert "scholz_gribkov_2007_partii" in IN_SCOPE_SOURCES


def test_scholz_2001_hardware_is_in_scope() -> None:
    """Scholz 2001 24-rod hardware paper is in-scope geometry source."""
    assert "scholz_2001_recent_progress_pf1000_hardware" in IN_SCOPE_SOURCES


def test_shakya_lee_model_is_context_only() -> None:
    """Reduced Lee-model comparison papers are context-only."""
    assert "shakya_2015_pf1000_pf400_lee_model" in CONTEXT_ONLY_SOURCES


def test_foam_liner_is_context_only() -> None:
    """Modified foam-liner PF-1000 configuration is context-only."""
    assert "scholz_1999_foam_liner_current_sheath" in CONTEXT_ONLY_SOURCES


def test_loarer_tokamak_is_context_only() -> None:
    """Tokamak gas-balance paper is context-only, not a DPF source."""
    assert "loarer_2007_tokamak_gas_balance_fuel_retention" in CONTEXT_ONLY_SOURCES


# ── scope consistency check ──────────────────────────────────────────────────

def test_pure_in_scope_set_is_consistent() -> None:
    result = check_scope_consistency(["scholz_gribkov_2007_partii", "malir_2024_interferometry_dpf"])
    assert result["consistent"] is True
    assert result["failure_reason"] is None


def test_pure_wrong_scope_set_is_consistent() -> None:
    """A set containing only wrong-scope sources has no mixing — still consistent."""
    result = check_scope_consistency(["talebitaher_2012_nx2_detector_anisotropy"])
    assert result["consistent"] is True


def test_mixed_in_scope_and_wrong_scope_fails() -> None:
    """Mixing in-scope and wrong-scope sources without a transfer rule must fail."""
    result = check_scope_consistency([
        "scholz_gribkov_2007_partii",
        "talebitaher_2012_nx2_detector_anisotropy",
    ])
    assert result["consistent"] is False
    assert result["failure_reason"] == "mixed_in_scope_and_wrong_scope_without_transfer_rule"


def test_mixed_with_bernard_fails() -> None:
    """PF-1000 full-energy + Bernard 1977 Mather review = mixed-scope failure."""
    result = check_scope_consistency([
        "scholz_2001_recent_progress_pf1000_hardware",
        "bernard_1977_dpf_high_intensity_neutron_source",
    ])
    assert result["consistent"] is False


def test_mixed_with_akel_16kv_fails() -> None:
    """PF-1000 full-energy + Akel 16 kV = wrong-scope mixing failure."""
    result = check_scope_consistency([
        "scholz_gribkov_2007_partii",
        "akel_2021_pf1000_neutron_yield_16kv",
    ])
    assert result["consistent"] is False


def test_mixed_with_ucsd_beg_fails() -> None:
    """PF-1000 full-energy + UCSD Beg = wrong-scope mixing failure."""
    result = check_scope_consistency([
        "malir_2024_interferometry_dpf",
        "ucsd_beg_current_sheath_initiation",
    ])
    assert result["consistent"] is False


def test_in_scope_plus_context_only_is_consistent() -> None:
    """Context-only sources may appear alongside in-scope sources."""
    result = check_scope_consistency([
        "scholz_gribkov_2007_partii",
        "shakya_2015_pf1000_pf400_lee_model",
    ])
    assert result["consistent"] is True


def test_empty_source_set_is_consistent() -> None:
    result = check_scope_consistency([])
    assert result["consistent"] is True


def test_unknown_source_does_not_cause_consistency_failure() -> None:
    """Unknown IDs are listed but do not by themselves trigger inconsistency."""
    result = check_scope_consistency(["some_future_source_id"])
    assert result["consistent"] is True
    assert "some_future_source_id" in result["unknown_sources"]


# ── scope packet completeness ────────────────────────────────────────────────

def test_scope_packet_contains_in_scope_sources() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert "in_scope_sources" in packet
    assert len(packet["in_scope_sources"]) > 0


def test_scope_packet_contains_wrong_scope_sources() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert "wrong_scope_sources" in packet
    assert len(packet["wrong_scope_sources"]) > 0


def test_scope_packet_contains_context_only_sources() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert "context_only_sources" in packet
    assert len(packet["context_only_sources"]) > 0


def test_scope_packet_references_decision_memo() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert "FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20" in packet["decision_memo"]


def test_scope_packet_references_governance_memo() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert "SPRINT8_WS2" in packet["governance_memo"]


def test_scope_packet_acknowledges_te_ti_gap() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert "te_ti_gap_acknowledgement" in packet
    assert "absent" in packet["te_ti_gap_acknowledgement"].lower()


def test_scope_packet_documents_scope_change() -> None:
    packet = runtime_demonstrator_scope_packet()
    assert "scope_change_note" in packet
    assert "16 kV" in packet["scope_change_note"] or "16kv" in packet["scope_change_note"].lower()
