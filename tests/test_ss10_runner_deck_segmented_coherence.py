"""Super-Sprint 10 SS10-1 / SS10-2 / SS10-3 coverage.

These tests close audit findings A1, A2, A3, and A4 for the PF-1000 full-energy
engineering-candidate path.  None of them promote any acceptance flag: the
runtime stays engineering-candidate only and every acceptance flag is false.

- SS10-1 (audit A1): the LLNL-like hybrid-PIC 3-D geometry evidence is emitted
  under an architecture-only key; no key containing ``same_scope`` carries the
  LLNL-like architecture scope for the PF-1000 full-energy preset.
- SS10-2 (audit A2 + A3): the runtime conductor-mask telemetry and the
  segmented manifest expose all five blocked geometry fields with blocker IDs;
  hollow-anode telemetry matches the deck (false / false).
- SS10-3 (audit A4): the six-step segmented manifest carries the four compact
  audit summary blocks.
"""

from __future__ import annotations

import json

from dpf.first_principles import pf1000_scholz_2001_24rod_full_energy_deck
from dpf.first_principles.deck import (
    PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE,
)
from dpf.first_principles.runner import run_first_principles_3d_deck
from dpf.first_principles.runtime_demonstrator_scope import SELECTED_SCOPE_LABEL
from dpf.first_principles.segmented_whole_shot import run_segmented_whole_shot

_LLNL_ARCHITECTURE_SCOPE = "llnl_like_180ka_axisymmetric_hybrid_pic"

_EXPECTED_BLOCKED_GEOMETRY_FIELDS = (
    "anode_hollow_bore_length_m",
    "insulator_wall_thickness_m",
    "backplate_radial_extent_m",
    "backplate_axial_thickness_m",
    "same_scope_reviewed_geometry_mask",
)


def _scan_same_scope_keys_for_scope(obj: object, scope: str) -> list[str]:
    """Recursively find paths of ``same_scope``-named keys carrying ``scope``."""

    hits: list[str] = []

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_path = f"{path}.{key}"
                if "same_scope" in str(key).lower():
                    blob = json.dumps(value, default=str)
                    if scope in blob:
                        hits.append(key_path)
                _walk(value, key_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, f"{path}[{index}]")

    _walk(obj, "")
    return hits


# ---------------------------------------------------------------------------
# SS10-1: architecture vs same-scope 3-D evidence
# ---------------------------------------------------------------------------
def test_pf1000_preset_no_same_scope_key_carries_llnl_like_scope() -> None:
    """No PF-1000 full-energy runtime key with ``same_scope`` carries LLNL-like.

    This walks the entire live runtime result recursively: every dict key that
    contains ``same_scope`` must not carry the LLNL-like architecture scope.
    """
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    leaks = _scan_same_scope_keys_for_scope(
        result.to_dict(), _LLNL_ARCHITECTURE_SCOPE
    )
    assert leaks == [], f"same_scope keys carry LLNL-like scope: {leaks}"


def test_pf1000_preset_emits_architecture_3d_geometry_under_named_key() -> None:
    """The LLNL-like 3-D geometry evidence is under an architecture-only key."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    evidence = result.telemetry["candidate_evidence"]

    # The architecture-only key is present; the old same-scope-named key is not.
    assert "architecture_3d_geometry_candidate_packet" in evidence
    assert "same_scope_3d_validation_packet" not in evidence

    packet = evidence["architecture_3d_geometry_candidate_packet"]
    assert packet["architecture_source_scope"] == _LLNL_ARCHITECTURE_SCOPE
    assert packet["is_same_scope_validation_evidence"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def _scan_object_keys(obj: object, target_key: str) -> list[str]:
    """Recursively find paths of dict keys named exactly ``target_key``."""

    hits: list[str] = []

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_path = f"{path}.{key}"
                if key == target_key:
                    hits.append(key_path)
                _walk(value, key_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, f"{path}[{index}]")

    _walk(obj, "")
    return hits


def test_pf1000_preset_emits_no_same_scope_3d_validation_evidence() -> None:
    """No ``same_scope_3d_validation_packet`` is emitted as candidate evidence.

    There is no genuine same-scope 3-D validation evidence for the current
    PF-1000 full-energy preset, so the runner fails closed: the candidate-
    evidence container carries the LLNL-like geometry under the architecture-
    only key, never under ``same_scope_3d_validation_packet``.

    ``same_scope_3d_validation_packet`` still appears as a *required-capability
    name* in the fail-closed ``missing_capabilities`` list and as a gate-roster
    status key (value ``missing_or_unaccepted``).  That is correct fail-closed
    behaviour: such entries never carry the LLNL-like architecture scope.
    """
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )

    # The candidate-evidence container must not emit it as an evidence packet.
    assert (
        "same_scope_3d_validation_packet"
        not in result.telemetry["candidate_evidence"]
    )
    assert (
        "same_scope_3d_validation_packet"
        not in result.manifest["candidate_evidence"]
    )

    # Wherever the name appears as an object key (the readiness-gate roster),
    # the value is a fail-closed status entry, never LLNL-like evidence.
    for path in _scan_object_keys(
        result.to_dict(), "same_scope_3d_validation_packet"
    ):
        assert "capabilities" in path, (
            "same_scope_3d_validation_packet appeared outside the readiness "
            f"gate roster: {path}"
        )

    # It is correctly carried as a fail-closed required-capability name.
    missing = result.validation_packet["hybrid_pic_3d_missing_capabilities"]
    assert "same_scope_3d_validation_packet" in missing


# ---------------------------------------------------------------------------
# SS10-2: geometry-mask runtime integrity (A2 + A3)
# ---------------------------------------------------------------------------
def test_pf1000_conductor_mask_exposes_five_blocked_geometry_fields() -> None:
    """The runtime conductor-mask telemetry exposes the five blocked fields."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    conductor_mask = result.telemetry["boundary_policy"]["conductor_mask"]
    blocked_fields = conductor_mask["blocked_geometry_fields"]

    by_name = {entry["field_name"]: entry for entry in blocked_fields}
    for field_name in _EXPECTED_BLOCKED_GEOMETRY_FIELDS:
        assert field_name in by_name, f"{field_name} missing from telemetry"
        entry = by_name[field_name]
        assert entry["blocked"] is True
        assert entry["blocker_id"], f"{field_name} must carry a blocker id"
        assert entry["source_scope_reason"]

    # The boundary policy also surfaces the blocked fields at its top level.
    boundary_policy = result.telemetry["boundary_policy"]
    bp_names = {
        entry["field_name"]
        for entry in boundary_policy["blocked_geometry_fields"]
    }
    assert bp_names == set(_EXPECTED_BLOCKED_GEOMETRY_FIELDS)


def test_pf1000_hollow_anode_telemetry_matches_deck_false_false() -> None:
    """Hollow-anode telemetry matches the deck: false / false (audit A3).

    The PF-1000 full-energy deck leaves ``anode_inner_radius_m=None`` so the
    anode is NOT declared hollow.  The runtime feature telemetry must report
    ``hollow_anode_declared_by_source=false`` and
    ``hollow_anode_inner_radius_supplied=false``.
    """
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    features = result.telemetry["boundary_policy"]["conductor_mask"][
        "pf1000_geometry_features"
    ]

    assert features["hollow_anode_declared_by_source"] is False
    assert features["hollow_anode_inner_radius_supplied"] is False
    # The missing bore length/radius is carried as a blocked-field entry.
    assert features["hollow_anode_bore_blocked"] is True


def test_pf1000_segmented_manifest_exposes_five_blocked_geometry_fields(
    tmp_path,
) -> None:
    """The segmented manifest exposes the five blocked geometry fields.

    This inspects the manifest produced from the actual runtime result of
    ``pf1000_scholz_2001_24rod_full_energy_deck()``, not just a deck helper.
    """
    deck = pf1000_scholz_2001_24rod_full_energy_deck().to_dict()
    manifest = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "pf1000_blocked_geometry",
        segment_steps=2,
        explicit_total_steps=6,
        verify_restart_equivalence=False,
    )
    summary = manifest["geometry_blocker_summary"]

    assert summary["blocked_geometry_field_count"] == 5
    assert set(summary["blocked_geometry_field_names"]) == set(
        _EXPECTED_BLOCKED_GEOMETRY_FIELDS
    )
    for entry in summary["blocked_geometry_fields"]:
        assert entry["blocked"] is True
        assert entry["blocker_id"]
    assert summary["hollow_anode_declared_by_source"] is False
    assert summary["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# SS10-3: segmented manifest audit summaries (A4)
# ---------------------------------------------------------------------------
def test_six_step_segmented_manifest_has_four_summary_blocks(
    tmp_path,
) -> None:
    """A six-step segmented probe emits the four required summary blocks."""
    deck = pf1000_scholz_2001_24rod_full_energy_deck().to_dict()
    manifest = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "pf1000_summary_blocks",
        segment_steps=2,
        explicit_total_steps=6,
        verify_restart_equivalence=False,
    )

    # ---- first_principles_scope_summary --------------------------------
    scope_summary = manifest["first_principles_scope_summary"]
    assert scope_summary["validation_scope"] == SELECTED_SCOPE_LABEL
    assert (
        scope_summary["selected_machine_source_scope"]
        == PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE
    )
    assert (
        scope_summary["architecture_source_scope"]
        == _LLNL_ARCHITECTURE_SCOPE
    )
    assert scope_summary["can_support_first_principles_acceptance"] is False
    assert scope_summary["accepted_runtime_claim"] is False

    # ---- same_scope_summary --------------------------------------------
    same_scope_summary = manifest["same_scope_summary"]
    assert same_scope_summary["declared_scope"] == SELECTED_SCOPE_LABEL
    assert isinstance(same_scope_summary["channel_states"], dict)
    assert same_scope_summary["channel_states"], "channel states must be present"
    assert (
        same_scope_summary["can_support_first_principles_acceptance"] is False
    )

    # ---- power_port_summary --------------------------------------------
    power_port_summary = manifest["power_port_summary"]
    blocked = power_port_summary["sigma_p_terms_ii_iv_v_vi_blocked"]
    for term in (
        "term_ii_motional_magnetic_sigma_p_J",
        "term_iv_motional_electric_sigma_p_J",
        "term_v_resistive_sigma_p_J",
        "term_vi_anomalous_poloidal_sigma_p_J",
    ):
        assert blocked[term] is True, f"{term} must be blocked"
    assert power_port_summary["all_sigma_p_terms_blocked"] is True
    assert (
        power_port_summary["can_support_first_principles_acceptance"] is False
    )

    # ---- geometry_blocker_summary --------------------------------------
    geometry_summary = manifest["geometry_blocker_summary"]
    assert geometry_summary["blocked_geometry_field_count"] == 5
    assert set(geometry_summary["blocked_geometry_field_names"]) == set(
        _EXPECTED_BLOCKED_GEOMETRY_FIELDS
    )


def test_six_step_segmented_manifest_summaries_keep_acceptance_false(
    tmp_path,
) -> None:
    """Every acceptance flag in the manifest stays false (SS10 guardrail 4)."""
    deck = pf1000_scholz_2001_24rod_full_energy_deck().to_dict()
    manifest = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "pf1000_acceptance_false",
        segment_steps=2,
        explicit_total_steps=6,
        verify_restart_equivalence=False,
    )

    assert manifest["can_support_first_principles_acceptance"] is False
    for block_name in (
        "first_principles_scope_summary",
        "same_scope_summary",
        "power_port_summary",
        "geometry_blocker_summary",
    ):
        block = manifest[block_name]
        assert block["can_support_first_principles_acceptance"] is False

    # No accepted_runtime_claim is ever promoted by the scope summary.
    assert (
        manifest["first_principles_scope_summary"]["accepted_runtime_claim"]
        is False
    )


def test_segmented_manifest_summaries_carry_no_llnl_like_same_scope_key(
    tmp_path,
) -> None:
    """The segmented manifest carries no ``same_scope`` key with LLNL-like."""
    deck = pf1000_scholz_2001_24rod_full_energy_deck().to_dict()
    manifest = run_segmented_whole_shot(
        deck=deck,
        run_dir=tmp_path / "pf1000_no_llnl_same_scope",
        segment_steps=2,
        explicit_total_steps=6,
        verify_restart_equivalence=False,
    )
    leaks = _scan_same_scope_keys_for_scope(manifest, _LLNL_ARCHITECTURE_SCOPE)
    assert leaks == [], f"manifest same_scope keys carry LLNL-like: {leaks}"
