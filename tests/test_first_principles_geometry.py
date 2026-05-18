"""WP-3 / SSR-005 negative controls for reviewed PF-1000 geometry masks.

Fail-closed discipline: coarse or under-resolved geometry must never be marked
accepted/reviewed, and the geometry packet must honestly report resolution
adequacy. Tests that assert unimplemented SSR-005 fields (mask hash,
projection-error block) are xfail(strict=False) until Patches 1 and 2 from the
WP-3 audit are applied.
"""

from __future__ import annotations

import pytest

from dpf.first_principles import (
    pf1000_akel_16kv_engineering_deck,
    run_first_principles_3d_deck,
)
from dpf.first_principles.deck import BoundaryPolicy


def _conductor_mask_packet(shape: tuple[int, int, int]) -> dict:
    deck = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=shape)
    result = run_first_principles_3d_deck(deck)
    return result.telemetry["boundary_policy"]["conductor_mask"]


# ---------------------------------------------------------------------------
# Tests that rely on SSR-005 fields NOT YET IMPLEMENTED (mask hash +
# projection-error block).  These are xfail until WP-3 Patch 1 lands.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="SSR-005 mask-hash / projection-error packet not yet implemented (WP-3 Patch 1)",
    strict=False,
)
def test_conductor_mask_packet_emits_mask_hash() -> None:
    """SSR-005: geometry packet must emit a mask SHA256 hash."""
    packet = _conductor_mask_packet((9, 9, 9))
    proj = packet.get("projection_error")
    assert proj is not None, "projection_error block missing"
    assert proj.get("mask_sha256"), "mask hash missing or empty"
    assert len(proj["mask_sha256"]) == 64


@pytest.mark.xfail(
    reason="SSR-005 mask-hash / projection-error packet not yet implemented (WP-3 Patch 1)",
    strict=False,
)
def test_conductor_mask_packet_emits_projection_error() -> None:
    """SSR-005: geometry packet must report error-from-source-dimensions."""
    packet = _conductor_mask_packet((9, 9, 9))
    proj = packet.get("projection_error")
    assert proj is not None, "projection_error block missing"
    assert proj["max_radial_discretization_error_m"] > 0.0
    assert proj["max_axial_discretization_error_m"] > 0.0
    assert proj["cells_per_rod_diameter"] is not None


# ---------------------------------------------------------------------------
# Tests for integrity-fix fields ALREADY APPLIED.
# These MUST pass against current code.
# ---------------------------------------------------------------------------

def test_cathode_rod_diameter_grid_cells_reported() -> None:
    """Integrity fix: cathode_rod_diameter_grid_cells must be present and > 0."""
    packet = _conductor_mask_packet((9, 9, 9))
    feats = packet["pf1000_geometry_features"]
    val = feats.get("cathode_rod_diameter_grid_cells")
    assert val is not None, "cathode_rod_diameter_grid_cells missing"
    assert val > 0.0


def test_cathode_rods_resolution_reviewed_is_false() -> None:
    """Integrity fix: cathode_rods_resolution_reviewed must be False (not yet reviewed)."""
    packet = _conductor_mask_packet((9, 9, 9))
    feats = packet["pf1000_geometry_features"]
    assert feats.get("cathode_rods_resolution_reviewed") is False


def test_coarse_grid_reports_low_cells_per_rod() -> None:
    """A 5^3 grid gives < 1 cell across a rod diameter — packet must report it.

    The integrity fix exposes cathode_rod_diameter_grid_cells so a reader can
    detect this; the exact value must be less than 4 (engineering minimum for
    resolving discrete rods).
    """
    packet = _conductor_mask_packet((5, 5, 5))
    feats = packet["pf1000_geometry_features"]
    val = feats.get("cathode_rod_diameter_grid_cells")
    assert val is not None, "cathode_rod_diameter_grid_cells missing at coarse grid"
    assert val < 4.0, (
        f"5^3 grid should give < 4 cells/rod-diameter, got {val}"
    )


def test_coarse_geometry_cannot_be_marked_accepted() -> None:
    """Geometry packets must never claim first-principles acceptance — at any grid."""
    for shape in [(5, 5, 5), (7, 7, 7), (9, 9, 9)]:
        packet = _conductor_mask_packet(shape)
        assert packet["can_support_first_principles_acceptance"] is False, (
            f"shape {shape}: can_support_first_principles_acceptance must be False"
        )
        assert packet["status"].startswith("candidate_"), (
            f"shape {shape}: status must start with 'candidate_', got {packet['status']!r}"
        )
        assert packet["conductor_mask_status"] != "reviewed_same_scope_geometry_mask", (
            f"shape {shape}: conductor_mask_status must not be reviewed_same_scope_geometry_mask"
        )


def test_insulator_is_declared_but_not_resolved() -> None:
    """Until an insulator material mask exists the packet must say so honestly."""
    packet = _conductor_mask_packet((9, 9, 9))
    feats = packet["pf1000_geometry_features"]
    assert feats["insulator_material_surface_declared"] is True
    assert feats["insulator_material_surface_resolved"] is False


# ---------------------------------------------------------------------------
# Resolution gate — currently NOT implemented (WP-3 Patch 2).
# The deck-level BoundaryPolicy.__post_init__ only checks source_references,
# not rod resolution.  This test is xfail until Patch 2 lands.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Resolution gate for reviewed_same_scope_geometry_mask not yet implemented (WP-3 Patch 2)",
    strict=False,
)
def test_reviewed_rod_mask_requires_resolved_rods() -> None:
    """A reviewed rod mask on an under-resolved grid must raise ValueError."""
    from dataclasses import replace

    from dpf.first_principles.deck import SourceReference

    base = pf1000_akel_16kv_engineering_deck(n_steps=1, shape=(5, 5, 5))
    dummy_source = SourceReference(
        path="KnowledgeReference/dummy.md",
        record_id="kr:dummy",
        capability_tags=("electrode_geometry",),
        role="test_source",
    )
    reviewed_boundaries = BoundaryPolicy(
        pml_cells=base.boundaries.pml_cells,
        pml_strength=base.boundaries.pml_strength,
        particle_absorption_enabled=base.boundaries.particle_absorption_enabled,
        conductor_mask_status="reviewed_same_scope_geometry_mask",
        conductor_mask_mode="pf1000_rod_hollow_projection",
        source_references=(dummy_source,),
    )
    reviewed_deck = replace(base, boundaries=reviewed_boundaries)
    with pytest.raises(ValueError, match="cells across a rod diameter"):
        run_first_principles_3d_deck(reviewed_deck)
