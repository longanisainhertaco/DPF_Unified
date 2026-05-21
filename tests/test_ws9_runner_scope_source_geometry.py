"""Super-Sprint 9 WS9-1 / WS9-2 / WS9-6 coverage.

These tests enforce package-native runner scope/source/geometry coherence for
the PF-1000 full-energy engineering-candidate path.  None of them promote any
acceptance flag: the runtime stays engineering-candidate only.

- WS9-1 (audit P0-1): the selected runtime-demonstrator scope, never the deck
  id, is the declared validation scope in every runtime packet.
- WS9-2 (audit P0-2): the top-level ``validation_packet.source_scope`` carries
  a PF-1000 selected-machine source scope, never the LLNL-like architecture
  scope; the hybrid-PIC paper remains architecture/equation-method evidence.
- WS9-6: the conductor mask cites the selected deck geometry / PF-1000 source
  refs, and an under-resolved cathode rod cannot lift geometry acceptance.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from dpf.first_principles import (
    minimal_engineering_deck,
    pf1000_scholz_2001_24rod_full_energy_deck,
)
from dpf.first_principles.deck import (
    PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE,
    FirstPrinciplesInputDeck,
)
from dpf.first_principles.runner import (
    FirstPrinciples3DDeck,
    run_first_principles_3d_deck,
)
from dpf.first_principles.runtime_demonstrator_scope import SELECTED_SCOPE_LABEL
from dpf.first_principles.segmented_whole_shot import run_segmented_whole_shot

_LLNL_ARCHITECTURE_SCOPE = "llnl_like_180ka_axisymmetric_hybrid_pic"
_PF1000_DECK_ID = (
    "pf1000_scholz_2001_24rod_full_energy_27kv_3p5torr_engineering_candidate"
)


# --------------------------------------------------------------------------
# WS9-1: runtime scope propagation
# --------------------------------------------------------------------------
def test_pf1000_full_energy_deck_declares_selected_validation_scope() -> None:
    """The package deck and its 3-D conversion both carry the selected scope."""
    deck = pf1000_scholz_2001_24rod_full_energy_deck()
    assert isinstance(deck, FirstPrinciplesInputDeck)
    assert deck.validation_scope == SELECTED_SCOPE_LABEL

    converted = FirstPrinciples3DDeck.from_deck(deck)
    assert converted.validation_scope == SELECTED_SCOPE_LABEL
    # The deck id must never be used as the validation scope.
    assert converted.validation_scope != deck.deck_id
    assert converted.validation_scope != _PF1000_DECK_ID


def test_pf1000_full_energy_runtime_emits_selected_scope_into_every_sink() -> None:
    """Every runtime packet sink emits SELECTED_SCOPE_LABEL, not the deck id."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    telemetry = result.telemetry

    assert telemetry["same_scope_source"]["declared_scope"] == SELECTED_SCOPE_LABEL
    assert (
        telemetry["engineering_current_waveform_comparison"]["declared_scope"]
        == SELECTED_SCOPE_LABEL
    )
    assert telemetry["limiter_readiness"]["declared_scope"] == SELECTED_SCOPE_LABEL
    assert telemetry["waveform_phase"]["declared_scope"] == SELECTED_SCOPE_LABEL
    assert telemetry["numerical_fidelity"]["declared_scope"] == SELECTED_SCOPE_LABEL
    assert telemetry["comparator_uq"]["declared_scope"] == SELECTED_SCOPE_LABEL
    assert telemetry["certificate_gate"]["declared_scope"] == SELECTED_SCOPE_LABEL
    assert telemetry["generalization"]["declared_scope"] == SELECTED_SCOPE_LABEL

    # Regression guard: no packet substitutes the deck id for the scope.
    for packet_name in (
        "same_scope_source",
        "engineering_current_waveform_comparison",
        "limiter_readiness",
        "waveform_phase",
        "numerical_fidelity",
        "comparator_uq",
        "certificate_gate",
        "generalization",
    ):
        assert telemetry[packet_name]["declared_scope"] != _PF1000_DECK_ID

    assert telemetry["can_support_first_principles_acceptance"] is False


def test_pf1000_full_energy_preset_manifest_never_emits_deck_id_as_scope() -> None:
    """Regression: a PF-1000 full-energy manifest must not declare the deck id.

    This test FAILS if any PF-1000 full-energy preset manifest packet emits the
    deck id (``..._engineering_candidate``) as the declared validation scope.
    """
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    manifest_text = json.dumps(result.manifest, default=str)
    telemetry_text = json.dumps(result.telemetry, default=str)

    # The deck id may legitimately appear as a deck *identifier*; what must
    # never happen is the deck id appearing as a "declared_scope" value.
    for blob in (manifest_text, telemetry_text):
        assert f'"declared_scope": "{_PF1000_DECK_ID}"' not in blob
        assert f'"validation_scope": "{_PF1000_DECK_ID}"' not in blob

    # And the selected scope must be present.
    assert SELECTED_SCOPE_LABEL in telemetry_text


def test_undeclared_deck_does_not_borrow_deck_id_as_validation_scope() -> None:
    """A deck with validation targets but no declared scope stays undeclared.

    The pre-fix defect returned the deck id whenever validation targets existed.
    """
    minimal = minimal_engineering_deck()
    converted = FirstPrinciples3DDeck.from_deck(minimal)
    assert converted.validation_scope != minimal.deck_id
    assert converted.validation_scope == "not_declared_engineering_smoke"


def test_cli_pf1000_full_energy_preset_emits_selected_scope() -> None:
    """`first-principles-3d --deck-preset pf1000_scholz_2001_24rod_full_energy`."""
    from dpf.cli.main import cli

    result = CliRunner().invoke(
        cli,
        [
            "first-principles-3d",
            "--deck-preset",
            "pf1000_scholz_2001_24rod_full_energy",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    same_scope = payload["telemetry_packets"]["same_scope_source"]
    assert same_scope["declared_scope"] == SELECTED_SCOPE_LABEL
    assert same_scope["declared_scope"] != _PF1000_DECK_ID
    assert (
        payload["engineering_current_waveform_comparison"]["declared_scope"]
        == SELECTED_SCOPE_LABEL
    )
    assert payload["can_support_first_principles_acceptance"] is False


def test_segmented_whole_shot_manifest_emits_selected_validation_scope(
    tmp_path,
) -> None:
    """The segmented whole-shot run manifest carries the selected scope."""
    deck = pf1000_scholz_2001_24rod_full_energy_deck().to_dict()
    run_dir = tmp_path / "pf1000_full_energy_segmented"

    manifest = run_segmented_whole_shot(
        deck=deck,
        run_dir=run_dir,
        segment_steps=2,
        explicit_total_steps=6,
        verify_restart_equivalence=False,
    )

    assert manifest["deck"]["validation_scope"] == SELECTED_SCOPE_LABEL
    assert manifest["deck"]["validation_scope"] != _PF1000_DECK_ID
    assert manifest["deck_name"] == SELECTED_SCOPE_LABEL

    # The deck.json the run directory persists carries the same scope.
    deck_json = json.loads((run_dir / "deck.json").read_text())
    assert deck_json["validation_scope"] == SELECTED_SCOPE_LABEL
    assert manifest["can_support_first_principles_acceptance"] is False


# --------------------------------------------------------------------------
# WS9-2: runtime source evidence separation
# --------------------------------------------------------------------------
def test_pf1000_full_energy_validation_packet_carries_pf1000_source_scope() -> None:
    """Top-level validation_packet.source_scope is a PF-1000 source scope."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )

    source_scope = result.validation_packet["source_scope"]
    assert source_scope == PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE
    assert "pf1000" in source_scope
    # The defect: the LLNL-like architecture scope leaking in as source scope.
    assert source_scope != _LLNL_ARCHITECTURE_SCOPE

    # Telemetry agrees with the candidate packet.
    assert result.telemetry["source_scope"] == PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE
    assert (
        result.telemetry["selected_machine_source_scope"]
        == PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE
    )


def test_pf1000_preset_never_emits_llnl_like_as_validation_source_scope() -> None:
    """The PF-1000 preset must never emit the LLNL-like scope as source scope."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    assert result.validation_packet["source_scope"] != _LLNL_ARCHITECTURE_SCOPE
    assert result.telemetry["source_scope"] != _LLNL_ARCHITECTURE_SCOPE


def test_hybrid_pic_architecture_source_present_under_named_key() -> None:
    """The hybrid-PIC paper stays as architecture/equation-method evidence."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )

    packet = result.validation_packet
    assert packet["architecture_source_scope"] == _LLNL_ARCHITECTURE_SCOPE
    assert packet["architecture_evidence_role"] == (
        "equation_method_and_architecture_source"
    )
    assert "hybrid-pic-fluid" in packet["architecture_source"]

    telemetry = result.telemetry
    assert telemetry["architecture_source_scope"] == _LLNL_ARCHITECTURE_SCOPE
    assert telemetry["architecture_source"] == packet["architecture_source"]
    # Architecture evidence and selected-machine scope are distinct values.
    assert telemetry["architecture_source_scope"] != telemetry["source_scope"]


def test_pf1000_source_geometry_evidence_lists_kr_geometry_paths() -> None:
    """Source-geometry evidence cites the PF-1000 KR geometry paths."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )

    references = result.validation_packet["selected_machine_source_references"]
    assert references, "PF-1000 deck must expose KR geometry source references"
    assert all(ref.startswith("KnowledgeReference/") for ref in references)
    # The Scholz 2000/2001 KR geometry/facility sources back the 24-rod deck.
    joined = " ".join(references)
    assert "recent-progress-in-1-mj-plasma-focus-research" in joined
    assert "pf-1000-device" in joined


# --------------------------------------------------------------------------
# WS9-6: PF-1000 geometry mask runtime integrity
# --------------------------------------------------------------------------
def test_pf1000_24rod_deck_keeps_five_geometry_fields_blocked() -> None:
    """The five blocked geometry fields stay blocked with their blocker IDs."""
    from dpf.first_principles.deck import (
        _pf1000_scholz_2001_24rod_blocked_fields,
    )

    blocked = _pf1000_scholz_2001_24rod_blocked_fields()
    for field_name in (
        "anode_hollow_bore_length_m",
        "insulator_wall_thickness_m",
        "backplate_radial_extent_m",
        "backplate_axial_thickness_m",
        "same_scope_reviewed_geometry_mask",
    ):
        assert field_name in blocked
        assert blocked[field_name], f"{field_name} must carry a blocker id"


def test_pf1000_conductor_mask_references_selected_deck_geometry() -> None:
    """The conductor mask cites the selected deck geometry / PF-1000 refs."""
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    conductor_mask = result.telemetry["boundary_policy"]["conductor_mask"]

    assert conductor_mask["declared_scope"] == SELECTED_SCOPE_LABEL
    assert (
        conductor_mask["selected_machine_source_scope"]
        == PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE
    )
    refs = conductor_mask["selected_machine_source_references"]
    assert refs, "conductor mask must cite selected-machine KR geometry refs"
    assert all(ref.startswith("KnowledgeReference/") for ref in refs)
    # The mask must not borrow the LLNL-like architecture scope.
    assert conductor_mask["selected_machine_source_scope"] != _LLNL_ARCHITECTURE_SCOPE
    assert conductor_mask["can_support_first_principles_acceptance"] is False


def test_pf1000_under_resolved_rods_cannot_support_geometry_acceptance() -> None:
    """An under-resolved cathode rod raises a mesh warning and blocks acceptance."""
    # The default 5x5x5 PF-1000 grid under-resolves the 32 mm cathode rods.
    result = run_first_principles_3d_deck(
        pf1000_scholz_2001_24rod_full_energy_deck()
    )
    conductor_mask = result.telemetry["boundary_policy"]["conductor_mask"]
    warning = conductor_mask["mesh_resolution_warning"]

    assert warning["cathode_rod_under_resolved"] is True
    assert warning["status"] == "warning_cathode_rod_under_resolved_not_validation"
    assert warning["warning"] is not None
    assert "under-resolved" in warning["warning"]
    assert warning["cells_per_rod_diameter"] < (
        warning["reviewed_min_cells_per_rod_diameter"]
    )
    assert warning["can_support_geometry_acceptance"] is False
    assert warning["can_support_first_principles_acceptance"] is False
    # The resolution review propagates the under-resolved flag.
    assert (
        conductor_mask["resolution_review"]["cathode_rod_under_resolved"] is True
    )
    assert (
        conductor_mask["resolution_review"][
            "reviewed_status_resolution_gate_eligible"
        ]
        is False
    )


def test_pf1000_resolved_rod_grid_clears_mesh_resolution_warning() -> None:
    """A grid that resolves the rods does not raise the under-resolution warning."""
    # A fine radial grid (many cells across the cathode cage) resolves the rods.
    deck = pf1000_scholz_2001_24rod_full_energy_deck(shape=(121, 121, 9))
    result = run_first_principles_3d_deck(deck)
    conductor_mask = result.telemetry["boundary_policy"]["conductor_mask"]
    warning = conductor_mask["mesh_resolution_warning"]

    assert warning["cathode_rod_under_resolved"] is False
    assert warning["warning"] is None
    assert warning["cells_per_rod_diameter"] >= (
        warning["reviewed_min_cells_per_rod_diameter"]
    )
    # Resolving rods still does not promote acceptance.
    assert warning["can_support_geometry_acceptance"] is False
    assert conductor_mask["can_support_first_principles_acceptance"] is False
