"""S3.2 WP-N3 PF-1000 / Akel source-tagged geometry packet and material masks.

Authority: docs/external_team_submissions/2026_05_18_three_sprint_blocker_
packet/sprint_3/WP_N3_GEOMETRY_SOURCE_PACKET.md and the Sprint 3 completion
handoff section "S3.2 PF-1000/Akel Geometry And Material Masks".

These tests prove the source-tagged geometry contract: conflicting source
dimensions (12 vs 24 rods, 460/480/600/450 mm anode length) are kept explicit
and never averaged; the WP-N3 missing dimensions stay typed `blocked` with
blocker IDs and never get a fabricated value; the 10 material/partition masks
are mutually disjoint where required and the Auluck partition stays
exhaustive; per-class SHA-256 hashes are deterministic; under-resolved rods
fail closed; a manifest missing a mask hash is rejected.

New S3.2/S3.3 structures are imported by full dotted path per the Sprint 3
file-scope rule (the package __init__ files are not edited).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dpf.fields.source_geometry as sg
from dpf.fields.maxwell_3d import Maxwell3DGrid

# Original source-geometry packet tests (unchanged behaviour).
from dpf.fields.source_geometry import (
    HybridPICSourceGeometry,
    source_geometry_candidate_evidence,
)
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pre-existing HybridPICSourceGeometry tests.
# ---------------------------------------------------------------------------

def test_source_geometry_packet_records_local_source_values() -> None:
    geometry = HybridPICSourceGeometry()

    assert geometry.source_scope == "llnl_like_180ka_axisymmetric_hybrid_pic"
    assert geometry.coordinate_system == "2d_axisymmetric_rz"
    assert geometry.anode_length_m == 0.05
    assert geometry.anode_radius_m == 0.01
    assert geometry.axial_pml_layers == 20
    assert geometry.background_particles == 500_000
    assert geometry.sheath_particles == 26_060
    assert geometry.can_support_first_principles_acceptance is False


def test_source_geometry_smoke_grid_is_cartesian_engineering_projection() -> None:
    geometry = HybridPICSourceGeometry()

    grid = geometry.smoke_grid(shape=(6, 6, 10))

    assert grid.shape == (6, 6, 10)
    assert grid.dx == grid.dy
    assert grid.dz == geometry.physical_length_m / 10


def test_source_geometry_candidate_evidence_does_not_satisfy_gate() -> None:
    evidence = source_geometry_candidate_evidence(HybridPICSourceGeometry())
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "same_scope_3d_validation_packet": evidence,
        },
    })

    assert evidence["status"] == "candidate"
    assert evidence["can_support_first_principles_acceptance"] is False
    assert "same_scope_3d_validation_packet" in status["missing_capabilities"]


# ---------------------------------------------------------------------------
# S3.2 fixtures: a grid that resolves the sourced rod diameter (80 mm) and a
# density field with an annular dense plasma ring (so wall cells exist in
# every material region for the partition tests).
# ---------------------------------------------------------------------------

_GRID_SHAPE = (80, 80, 20)
_GRID_SPACING = (0.0175, 0.0175, 0.05)  # dx*4 < 0.080 m rod diameter


def _resolved_grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=_GRID_SHAPE, spacing=_GRID_SPACING)


def _annular_density(shape: tuple[int, int, int], spacing: tuple[float, float, float],
                     *, floor: float = 1.0e15, dense: float = 1.0e23) -> np.ndarray:
    """Dense plasma in an annulus away from the axis; low density elsewhere."""
    nx, ny, nz = shape
    dx, dy, _ = spacing
    ci, cj = (nx - 1) / 2.0, (ny - 1) / 2.0
    ii = np.arange(nx, dtype=float) - ci
    jj = np.arange(ny, dtype=float) - cj
    radius = np.sqrt((ii[:, None] * dx) ** 2 + (jj[None, :] * dy) ** 2)
    radius3 = np.broadcast_to(radius[:, :, None], (nx, ny, nz))
    density = np.full((nx, ny, nz), floor)
    density[(radius3 >= 0.30) & (radius3 <= 0.45)] = dense
    return density


def _partition(packet: sg.PF1000GeometryPacket, *, pml_layers: int = 0,
                source_interface_z_index: int = 1,
                grid: Maxwell3DGrid | None = None,
                min_cells_per_feature: float = 4.0) -> dict:
    grid = grid if grid is not None else _resolved_grid()
    density = _annular_density(grid.shape, grid.spacing)
    current = np.full(grid.shape, 1.0e3)
    return sg.build_pf1000_material_partition(
        packet,
        grid=grid,
        electron_density_m3=density,
        current_density_norm_A_m2=current,
        source_interface_z_index=source_interface_z_index,
        pml_layers=pml_layers,
        electron_density_floor_m3=1.0e18,
        min_cells_per_feature=min_cells_per_feature,
    )


# ---------------------------------------------------------------------------
# Constructors preserve source conflicts without averaging.
# ---------------------------------------------------------------------------

def test_geometry_constructors_pin_one_source_revision() -> None:
    """Each constructor pins one self-consistent KR source set."""
    krauz = sg.PF1000GeometryPacket.krauz_2012()
    akel = sg.PF1000GeometryPacket.akel_shot_12581()
    scholz = sg.PF1000GeometryPacket.scholz_gribkov_revision()

    assert krauz.geometry_source_tag == "pf1000_krauz2012"
    assert akel.geometry_source_tag == "pf1000_akel_shot12581"
    assert scholz.geometry_source_tag == "pf1000_scholz_gribkov_revision"
    assert akel.scope_tag == "pf1000_akel_16kv_1p2torr_shot_12581"
    for packet in (krauz, akel, scholz):
        assert packet.can_support_first_principles_acceptance is False
        assert packet.geometry_review_status == "geometry_candidate_not_reviewed"


def test_conflicting_dimension_kept_explicit_not_averaged() -> None:
    """The 12-vs-24 rod count and 460/480/600/450 mm anode length stay explicit.

    [WP_N3_GEOMETRY_SOURCE_PACKET.md section 4] The conflicting candidate
    values are held verbatim in PF1000GeometryConflict records; their
    arithmetic mean never appears as a field value.
    """
    packet = sg.PF1000GeometryPacket.krauz_2012()

    rod_conflict = packet.conflicts["cathode_rod_count"]
    assert set(rod_conflict.candidate_values) == {12, 24}
    # the arithmetic mean (18) is never an emitted value.
    assert 18 not in rod_conflict.candidate_values

    z0_conflict = packet.conflicts["anode_length_z0"]
    assert set(z0_conflict.candidate_values) == {0.460, 0.480, 0.600, 0.450}
    mean_z0 = sum(z0_conflict.candidate_values) / len(z0_conflict.candidate_values)
    assert all(v != mean_z0 for v in z0_conflict.candidate_values)

    # the conflicting fields carry value None and a conflict group, never a
    # silently-chosen number.
    assert packet.get_field("cathode_rod_count").status == "conflict"
    assert packet.get_field("cathode_rod_count").value is None
    assert packet.get_field("anode_length_m").status == "conflict"
    assert packet.get_field("anode_length_m").value is None


def test_conflict_record_rejects_single_candidate() -> None:
    """A PF1000GeometryConflict needs at least two disagreeing candidates."""
    with pytest.raises(ValueError, match="at least two candidate"):
        sg.PF1000GeometryConflict(
            group="g", field_name="f", units="m",
            candidate_values=(0.1,), candidate_source_refs=("KR: x:1-2",),
            reason="single value is not a conflict",
        )


# ---------------------------------------------------------------------------
# Missing bore / insulator / backplate fields remain blocked.
# ---------------------------------------------------------------------------

def test_missing_bore_insulator_backplate_fields_stay_blocked() -> None:
    """The WP-N3 section-4 missing dimensions are typed `blocked`, never invented.

    [WP_N3_GEOMETRY_SOURCE_PACKET.md section 4] anode bore radius/length,
    anode end-cap, cathode rod length, insulator outer radius / wall, backplate
    radial extent / axial thickness, chamber wall material / thickness.
    """
    packet = sg.PF1000GeometryPacket.krauz_2012()
    expected_blocked = {
        "anode_hollow_bore_radius_m",
        "anode_hollow_bore_length_m",
        "anode_end_cap_diameter_m",
        "cathode_rod_length_m",
        "insulator_outer_radius_m",
        "insulator_wall_thickness_m",
        "backplate_radial_extent_m",
        "backplate_axial_thickness_m",
        "chamber_wall_material",
        "chamber_wall_thickness_m",
    }
    assert set(packet.blocked_field_names()) == expected_blocked
    for name in expected_blocked:
        fld = packet.get_field(name)
        assert fld.status == "blocked"
        assert fld.value is None, f"{name}: blocked field must not have a value"
        assert fld.blocker_id, f"{name}: blocked field must carry a blocker ID"
        assert fld.blocker_id.startswith("PF1000-BLK-")


def test_blocked_field_rejects_a_fabricated_value() -> None:
    """A blocked PF1000GeometryField with a numeric value is rejected."""
    with pytest.raises(ValueError, match="blocked field must have value None"):
        sg.PF1000GeometryField(
            name="anode_hollow_bore_radius_m", value=0.05, units="m",
            status="blocked", scope_tag="pf1000_krauz2012",
            blocker_id=(
                "PF1000-BLK-009-anode-bore-radius-"
                "source_available_not_target_extracted"
            ),
        )


def test_blocked_field_requires_a_blocker_id() -> None:
    """A blocked field with no blocker ID is rejected."""
    with pytest.raises(ValueError, match="needs a blocker_id"):
        sg.PF1000GeometryField(
            name="anode_hollow_bore_radius_m", value=None, units="m",
            status="blocked", scope_tag="pf1000_krauz2012",
        )


def test_source_supported_field_requires_a_source_ref() -> None:
    """A source_supported field with no KR source ref is rejected."""
    with pytest.raises(ValueError, match="needs source_ref"):
        sg.PF1000GeometryField(
            name="anode_radius_m", value=0.1155, units="m",
            status="source_supported", scope_tag="pf1000_krauz2012",
        )


# ---------------------------------------------------------------------------
# All source references exist locally.
# ---------------------------------------------------------------------------

def test_all_geometry_source_refs_exist_locally() -> None:
    """Every KR file in PF1000_GEOMETRY_SOURCE_REFS exists under the repo."""
    for ref in sg.PF1000_GEOMETRY_SOURCE_REFS:
        path = ref.split(":", 1)[0]
        assert path.startswith("KnowledgeReference/"), ref
        assert (_REPO_ROOT / path).is_file(), f"missing KR source: {path}"


def test_every_field_source_ref_points_at_an_existing_kr_file() -> None:
    """Each source_supported field cites a KR file that exists locally."""
    for ctor in (
        sg.PF1000GeometryPacket.krauz_2012,
        sg.PF1000GeometryPacket.akel_shot_12581,
        sg.PF1000GeometryPacket.scholz_gribkov_revision,
    ):
        packet = ctor()
        for name in packet.source_supported_field_names():
            ref = packet.get_field(name).source_ref
            assert ref is not None
            path = ref.split(":", 1)[0]
            assert (_REPO_ROOT / path).is_file(), f"{name}: missing KR {path}"


# ---------------------------------------------------------------------------
# The 10 masks; material sub-classes mutually disjoint; Auluck partition
# exhaustive; Omega and source interface disjoint.
# ---------------------------------------------------------------------------

def test_partition_emits_all_ten_mask_classes() -> None:
    """The partition emits all 10 handoff-required mask classes."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    assert set(partition["mask_packets"]) == set(sg.PF1000_MASK_CLASSES)
    assert len(sg.PF1000_MASK_CLASSES) == 10


def test_material_subclasses_are_mutually_disjoint() -> None:
    """The five material sub-classes never share a cell."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    masks = partition["_label_masks"]
    union = np.zeros(_GRID_SHAPE, dtype=bool)
    for name in sg.PF1000_MATERIAL_SUBCLASSES:
        mask = np.asarray(masks[name], dtype=bool)
        assert not np.any(union & mask), f"{name} overlaps another sub-class"
        union |= mask
    assert partition["partition_constraints"][
        "material_subclasses_mutually_disjoint"
    ] is True


def test_material_subclasses_exhaust_wall_material() -> None:
    """The five material sub-classes partition wall_material_faces exactly."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    masks = partition["_label_masks"]
    union = np.zeros(_GRID_SHAPE, dtype=bool)
    for name in sg.PF1000_MATERIAL_SUBCLASSES:
        union |= np.asarray(masks[name], dtype=bool)
    assert np.array_equal(union, np.asarray(masks["wall_material_faces"], bool))
    assert partition["partition_constraints"][
        "material_subclasses_exhaust_wall_material"
    ] is True


def test_auluck_top_level_partition_remains_exhaustive() -> None:
    """The four Auluck top-level labels still cover every cell exactly once."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    constraints = partition["partition_constraints"]
    assert constraints["auluck_top_level_mutually_disjoint"] is True
    assert constraints["auluck_top_level_exhaustive"] is True


def test_omega_and_source_interface_are_disjoint() -> None:
    """Omega never intersects the excluded terminal source interface."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    masks = partition["_label_masks"]
    overlap = np.asarray(masks["omega_volume_cells"], bool) & np.asarray(
        masks["terminal_source_interface_faces"], bool
    )
    assert not np.any(overlap)
    assert partition["partition_constraints"][
        "terminal_source_interface_disjoint_from_omega"
    ] is True


# ---------------------------------------------------------------------------
# Mask hashes are stable; each material sub-class has a distinct hash.
# ---------------------------------------------------------------------------

def test_mask_hashes_are_deterministic() -> None:
    """The same packet + grid always yields identical per-class hashes."""
    packet = sg.PF1000GeometryPacket.krauz_2012()
    first = _partition(packet)["manifest"]["mask_sha256_by_class"]
    second = _partition(packet)["manifest"]["mask_sha256_by_class"]
    assert first == second
    for digest in first.values():
        assert len(digest) == 64


def test_each_material_subclass_has_a_distinct_hash() -> None:
    """Anode/cathode/insulator/chamber/backplate masks each hash differently."""
    partition = _partition(
        sg.PF1000GeometryPacket.krauz_2012(),
        pml_layers=0,
        source_interface_z_index=1,
    )
    hashes = partition["manifest"]["mask_sha256_by_class"]
    sub = [hashes[name] for name in sg.PF1000_MATERIAL_SUBCLASSES]
    assert len(set(sub)) == len(sub), "material sub-class hashes collide"


def test_partition_never_promotes_first_principles_acceptance() -> None:
    """The partition packet and its manifest never claim acceptance."""
    partition = _partition(sg.PF1000GeometryPacket.akel_shot_12581())
    assert partition["can_support_first_principles_acceptance"] is False
    assert partition["can_support_power_port_acceptance"] is False
    assert partition["manifest"]["can_support_first_principles_acceptance"] is False
    assert partition["status"].startswith("candidate_")


# ---------------------------------------------------------------------------
# Under-resolution gate: under-resolved rods fail closed.
# ---------------------------------------------------------------------------

def test_under_resolved_rods_fail_closed() -> None:
    """A grid that cannot resolve the 80 mm rod diameter raises, not a mask.

    [WP_N3_GEOMETRY_SOURCE_PACKET.md section 6 item 8] Under-resolution gate.
    """
    coarse = Maxwell3DGrid(shape=(10, 10, 10), spacing=(0.14, 0.14, 0.25))
    with pytest.raises(ValueError, match="does not resolve"):
        _partition(sg.PF1000GeometryPacket.krauz_2012(), grid=coarse)


def test_resolved_grid_reports_no_under_resolution_flags() -> None:
    """A grid that resolves the sourced features sets no under-resolution flag."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    flags = partition["manifest"]["under_resolution_flags"]
    assert flags
    assert not any(flags.values())


# ---------------------------------------------------------------------------
# Manifest validation: a manifest missing a mask hash is rejected.
# ---------------------------------------------------------------------------

def test_manifest_missing_a_mask_hash_is_rejected() -> None:
    """PF1000MaskManifest construction fails closed when a class hash is absent."""
    full = {name: ("0" * 64) for name in sg.PF1000_MASK_CLASSES}
    partial = dict(full)
    del partial["cathode_rod_faces"]
    counts = {name: 0 for name in sg.PF1000_MASK_CLASSES}
    with pytest.raises(ValueError, match="missing per-class hashes"):
        sg.PF1000MaskManifest(
            geometry_packet_id="pf1000_geometry_packet_krauz2012",
            geometry_source_tag="pf1000_krauz2012",
            source_refs=sg.PF1000_GEOMETRY_SOURCE_REFS,
            conflict_groups=("cathode_rod_count",),
            blocked_fields=("anode_hollow_bore_radius_m",),
            grid_shape=(8, 8, 8),
            grid_spacing_m=(0.1, 0.1, 0.1),
            mask_sha256_by_class=partial,
            mask_cell_counts=counts,
            under_resolution_flags={},
        )


def test_manifest_rejects_a_first_principles_acceptance_claim() -> None:
    """A manifest that claims first-principles acceptance is rejected."""
    full = {name: ("0" * 64) for name in sg.PF1000_MASK_CLASSES}
    counts = {name: 0 for name in sg.PF1000_MASK_CLASSES}
    with pytest.raises(ValueError, match="must not claim first-principles"):
        sg.PF1000MaskManifest(
            geometry_packet_id="pf1000_geometry_packet_krauz2012",
            geometry_source_tag="pf1000_krauz2012",
            source_refs=sg.PF1000_GEOMETRY_SOURCE_REFS,
            conflict_groups=(),
            blocked_fields=(),
            grid_shape=(8, 8, 8),
            grid_spacing_m=(0.1, 0.1, 0.1),
            mask_sha256_by_class=full,
            mask_cell_counts=counts,
            under_resolution_flags={},
            can_support_first_principles_acceptance=True,
        )


def test_manifest_lists_all_mask_hashes_blocked_fields_and_source_refs() -> None:
    """The emitted manifest carries every per-class hash, blocker, and KR ref."""
    packet = sg.PF1000GeometryPacket.krauz_2012()
    partition = _partition(packet)
    manifest = partition["manifest"]
    assert set(manifest["mask_sha256_by_class"]) == set(sg.PF1000_MASK_CLASSES)
    assert set(manifest["mask_cell_counts"]) == set(sg.PF1000_MASK_CLASSES)
    assert set(manifest["blocked_fields"]) == set(packet.blocked_field_names())
    assert set(manifest["conflict_groups"]) == set(packet.conflicts)
    assert tuple(manifest["source_refs"]) == sg.PF1000_GEOMETRY_SOURCE_REFS
    assert manifest["geometry_source_tag"] == packet.geometry_source_tag


# ===========================================================================
# S3.3 -- WP-N3 SigmaPSurfacePacket structural tests.
#
# Authority: WP_N3_SIGMA_P_RUNTIME_INTERFACE_SPEC.md section 3 schema; handoff
# section "S3.3 Sigma_p Surface Packet Plumbing". S3.3 is plumbing only -- the
# packet carries face geometry and per-operand status, never a term value.
# ===========================================================================

def test_sigma_p_blocked_packet_is_fully_fail_closed() -> None:
    """The default blocked Sigma_p packet exposes no face set and no operands."""
    sp = sg.SigmaPSurfacePacket.blocked()
    assert sp.n_sigma_p_faces == 0
    assert sp.has_sigma_p() is False
    assert sp.has_velocity() is False
    assert sp.has_resistivity() is False
    assert sp.has_sign_convention() is False
    assert sp.can_support_power_port_acceptance is False
    assert sp.can_support_first_principles_acceptance is False
    # every absent operand carries a typed blocker.
    for operand in ("sigma_p", "v", "eta", "sign_convention"):
        assert sp.operand_blockers[operand]


def test_sigma_p_packet_rejects_a_power_port_acceptance_claim() -> None:
    """A Sigma_p packet that claims power-port acceptance is rejected."""
    blocked = sg.SigmaPSurfacePacket.blocked()
    with pytest.raises(ValueError, match="must not claim power-port"):
        replace_with_acceptance(blocked)


def replace_with_acceptance(packet: sg.SigmaPSurfacePacket) -> sg.SigmaPSurfacePacket:
    """Helper: rebuild a Sigma_p packet with the acceptance flag forced True."""
    from dataclasses import replace

    return replace(packet, can_support_power_port_acceptance=True)


def test_sigma_p_packet_rejects_faces_exceeding_total_sigma() -> None:
    """n_sigma_p_faces may never exceed face_count_total_sigma."""
    with pytest.raises(ValueError, match="must not exceed"):
        sg.SigmaPSurfacePacket(
            status="candidate_sigma_p_surface_packet_not_validation",
            source_refs=sg.SIGMA_P_SURFACE_SOURCE_REFS,
            source_geometry_packet_id="pid",
            source_geometry_hash="hash",
            n_sigma_p_faces=11,
            face_count_total_sigma=10,
            geometry_review_status="geometry_candidate_not_reviewed",
            face_ids=np.arange(11),
            dS_outward_m2=np.zeros((11, 3)),
            face_area_m2=np.ones(11),
            outward_normal=np.zeros((11, 3)),
            face_material_class=tuple("x" for _ in range(11)),
            is_moving=np.ones(11, dtype=bool),
            omega_side="omega_interior",
            excluded_interface_side="excluded",
            outward_normal_convention="outward_from_omega",
            field_sampler_status={"B": "blocked", "E": "blocked", "J": "blocked"},
            velocity_status="blocked",
            resistivity_status="blocked",
            centering={"time_centering": "candidate_step_consistent_not_accepted"},
            quadrature="not_available",
            sign_convention=None,
            operand_blockers={},
        )


def test_sigma_p_builder_with_no_runtime_returns_blocked_packet() -> None:
    """With no runtime Sigma_p face set the builder fails closed."""
    sp = sg.build_sigma_p_surface_packet(None)
    assert sp.status == "blocked_sigma_p_surface_packet_not_available"
    assert sp.has_sigma_p() is False


def test_sigma_p_builder_carries_geometry_packet_id_and_hash() -> None:
    """The Sigma_p packet records which S3.2 geometry it was built from."""
    packet = sg.PF1000GeometryPacket.krauz_2012()
    partition = _partition(packet)
    sp = sg.build_sigma_p_surface_packet(partition)
    assert sp.source_geometry_packet_id == packet.geometry_packet_id


def test_sigma_p_builder_does_not_fabricate_a_face_set() -> None:
    """Even given a non-None runtime hint the builder never invents Sigma_p.

    S3.3 is plumbing only; deriving the moving-boundary face set from reviewed
    material masks is Sprint 4 work.
    """
    packet = sg.PF1000GeometryPacket.krauz_2012()
    partition = _partition(packet)
    sp = sg.build_sigma_p_surface_packet(
        partition, sigma_p_runtime={"placeholder": True}
    )
    assert sp.has_sigma_p() is False
    assert sp.n_sigma_p_faces == 0
    assert "sprint4" in sp.status


def test_sigma_p_term_operand_map_covers_all_four_moving_terms() -> None:
    """SIGMA_P_TERM_OPERANDS lists exactly the four Auluck eq. (6) Sigma_p terms."""
    assert set(sg.SIGMA_P_TERM_OPERANDS) == {
        "term_ii_motional_magnetic_sigma_p_J",
        "term_iv_motional_electric_sigma_p_J",
        "term_v_resistive_sigma_p_J",
        "term_vi_anomalous_poloidal_sigma_p_J",
    }
    # term V is the only resistive term -- it alone needs eta.
    assert "eta" in sg.SIGMA_P_TERM_OPERANDS["term_v_resistive_sigma_p_J"]
    for key in (
        "term_ii_motional_magnetic_sigma_p_J",
        "term_iv_motional_electric_sigma_p_J",
        "term_vi_anomalous_poloidal_sigma_p_J",
    ):
        assert "eta" not in sg.SIGMA_P_TERM_OPERANDS[key]
        assert "v" in sg.SIGMA_P_TERM_OPERANDS[key]


def test_sigma_p_packet_operand_status_reports_per_operand_availability() -> None:
    """operand_status reflects each operand's availability flag."""
    blocked = sg.SigmaPSurfacePacket.blocked()
    assert blocked.operand_status("sigma_p") is False
    assert blocked.operand_status("v") is False
    assert blocked.operand_status("eta") is False
    with pytest.raises(ValueError, match="unknown Sigma_p operand"):
        blocked.operand_status("not_an_operand")


# ===========================================================================
# S3R.4 negative tests (A4, A5)
#
# Authority: Sprint 3R remediation handoff -- "Required negative tests / S3R.4"
# ===========================================================================

def test_s3r4_blocked_insulator_outer_radius_cannot_produce_source_backed_insulator_faces() -> None:
    """S3R.4 N1: insulator outer radius is blocked (PF1000-BLK-014). The
    insulator_material_faces mask is therefore candidate_projection not
    source_supported -- a blocked dimension may not back a source-backed mask.
    """
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    manifest = partition["manifest"]
    status = manifest["mask_class_status"]["insulator_material_faces"]
    assert status != "source_supported", (
        "insulator_material_faces must not be source_supported when "
        "insulator_outer_radius_m is blocked"
    )
    assert status in ("candidate_projection_not_source_mask", "blocked"), (
        f"unexpected status: {status!r}"
    )


def test_s3r4_conflict_cathode_cage_radius_cannot_produce_source_backed_cathode_shell() -> None:
    """S3R.4 N2: cathode_cage_radius_m is a conflict field (two KR values disagree).
    The cathode_rod_faces mask is therefore candidate_projection not source_supported.
    """
    packet = sg.PF1000GeometryPacket.krauz_2012()
    assert packet.fields["cathode_cage_radius_m"].status == "conflict", (
        "pre-condition: cathode_cage_radius_m must be conflict"
    )
    partition = _partition(packet)
    manifest = partition["manifest"]
    status = manifest["mask_class_status"]["cathode_rod_faces"]
    assert status != "source_supported", (
        "cathode_rod_faces must not be source_supported when "
        "cathode_cage_radius_m is a conflict field"
    )
    assert status in ("candidate_projection_not_source_mask", "blocked"), (
        f"unexpected status: {status!r}"
    )


def test_s3r4_mask_class_status_field_present_for_all_mask_classes() -> None:
    """S3R.4: every mask class in PF1000_MASK_CLASSES has an explicit status entry."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    manifest = partition["manifest"]
    for name in sg.PF1000_MASK_CLASSES:
        assert name in manifest["mask_class_status"], (
            f"mask_class_status missing entry for {name!r}"
        )
        status = manifest["mask_class_status"][name]
        assert status in (
            "source_supported",
            "candidate_projection_not_source_mask",
            "blocked",
        ), f"{name!r}: unexpected status {status!r}"


def test_s3r4_source_supported_masks_have_non_empty_sha256() -> None:
    """S3R.4: masks with status source_supported carry a 64-char SHA-256.
    Blocked masks must have an empty-string sentinel, not a real hash.
    """
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    manifest = partition["manifest"]
    for name in sg.PF1000_MASK_CLASSES:
        status = manifest["mask_class_status"][name]
        sha = manifest["mask_sha256_by_class"][name]
        if status == "blocked":
            assert sha == "", (
                f"{name!r}: blocked mask must have empty SHA-256 sentinel, got {sha!r}"
            )
        else:
            assert len(sha) == 64, (
                f"{name!r}: produced mask (status={status!r}) must have 64-char SHA-256"
            )


def test_s3r4_under_resolved_insulator_surface_fails_closed() -> None:
    """S3R.4/A5: when insulator_exposed_length_m is source_supported and the
    axial grid cannot resolve it, the partition build fails closed.

    This test uses a custom packet with a source-supported insulator length
    and a coarse axial grid (dz too large to resolve the feature).
    """
    krauz = sg.PF1000GeometryPacket.krauz_2012()
    # Override insulator_exposed_length_m to be source_supported with a small
    # value that a coarse axial grid won't resolve.  We patch the field dict.
    from dataclasses import replace as _dc_replace
    patched_fields = dict(krauz.fields)
    patched_fields["insulator_exposed_length_m"] = sg.PF1000GeometryField(
        name="insulator_exposed_length_m",
        value=0.010,  # 10 mm -- small enough to be under-resolved
        units="m",
        status="source_supported",
        scope_tag=krauz.scope_tag,
        source_ref=(
            "KnowledgeReference/experimental-study-of-the-structure-of-the-"
            "plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:349-350"
        ),
    )
    patched_packet = _dc_replace(krauz, fields=patched_fields)
    # dz = 0.10 m: 10 mm / 0.10 m = 0.1 cells << 4 -> under-resolved.
    coarse_axial = Maxwell3DGrid(shape=(80, 80, 20), spacing=(0.0175, 0.0175, 0.10))
    with pytest.raises(ValueError, match="insulator_exposed_length_m"):
        _partition(patched_packet, grid=coarse_axial)


def test_s3r4_no_fallback_to_generic_wall_material_class_when_masks_requested() -> None:
    """S3R.4 N4: the partition must emit the five distinct material sub-classes.
    There is no fallback to a single generic 'wall_material_faces' class even
    when a sub-class relies on a conflict/blocked dimension.
    """
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    masks = partition["_label_masks"]
    # All five named sub-classes must be present, never collapsed into one.
    for name in sg.PF1000_MATERIAL_SUBCLASSES:
        assert name in masks, f"material sub-class {name!r} is missing"
    # The generic wall_material_faces class is the union of sub-classes, not
    # a replacement for them.
    assert "wall_material_faces" in masks


def test_s3r4_12_rod_packet_rod_faces_status_is_candidate_projection() -> None:
    """S3R.4: with 12 rods (conflict) and conflict cage radius, cathode_rod_faces
    is candidate_projection_not_source_mask -- the heuristic cage geometry must
    not be promoted to source_supported.
    """
    packet = sg.PF1000GeometryPacket.akel_shot_12581()
    assert packet.fields["cathode_rod_count"].status == "conflict"
    assert packet.fields["cathode_cage_radius_m"].status == "conflict"
    partition = _partition(packet)
    status = partition["manifest"]["mask_class_status"]["cathode_rod_faces"]
    assert status == "candidate_projection_not_source_mask"


def test_s3r4_backplate_mask_is_source_supported() -> None:
    """S3R.4: backplate_source_interface_faces is built from the source-interface
    z-index (k_port) which is a solver parameter, not a heuristic. It must be
    reported as source_supported.
    """
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    manifest = partition["manifest"]
    assert (
        manifest["mask_class_status"]["backplate_source_interface_faces"]
        == "source_supported"
    )


def test_s3r4_chamber_wall_mask_is_candidate_until_target_extracted() -> None:
    """S3R.4: chamber_wall_faces must not be promoted to source_supported while
    chamber wall material/thickness are only KR text-parity and not target-extracted.
    """
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    manifest = partition["manifest"]
    assert (
        manifest["mask_class_status"]["chamber_wall_faces"]
        == "candidate_projection_not_source_mask"
    )


def test_s3r4_chamber_wall_candidate_even_if_cage_radius_is_sourced() -> None:
    """S3R.4: sourcing the cathode cage split is not enough to promote the
    chamber-wall mask while wall material/thickness remain blocked.
    """
    from dataclasses import replace as _dc_replace

    packet = sg.PF1000GeometryPacket.krauz_2012()
    fields = dict(packet.fields)
    fields["cathode_cage_radius_m"] = sg.PF1000GeometryField(
        name="cathode_cage_radius_m",
        value=0.16,
        units="m",
        status="source_supported",
        scope_tag=packet.scope_tag,
        source_ref=packet.conflicts[
            packet.fields["cathode_cage_radius_m"].conflict_group
        ].candidate_source_refs[0],
    )
    patched_packet = _dc_replace(packet, fields=fields)

    partition = _partition(patched_packet)
    assert (
        partition["manifest"]["mask_class_status"]["chamber_wall_faces"]
        == "candidate_projection_not_source_mask"
    )


def test_s3r4_source_available_blockers_use_honest_taxonomy() -> None:
    """S3R.4: source-available values stay blocked until target extraction, but
    their blocker IDs must not claim there is no KR source.
    """
    packet = sg.PF1000GeometryPacket.krauz_2012()
    assert (
        packet.fields["anode_hollow_bore_radius_m"].blocker_id
        == "PF1000-BLK-009-anode-bore-radius-source_available_not_target_extracted"
    )
    assert (
        packet.fields["chamber_wall_material"].blocker_id
        == "PF1000-BLK-021-chamber-wall-material-source_available_not_target_extracted"
    )
    assert (
        packet.fields["chamber_wall_thickness_m"].blocker_id
        == "PF1000-BLK-022-chamber-wall-thickness-source_available_not_target_extracted"
    )


def test_s3r4_mask_class_status_survives_to_dict_serialization() -> None:
    """S3R.4: mask_class_status must be present in the manifest's to_dict() output."""
    partition = _partition(sg.PF1000GeometryPacket.krauz_2012())
    manifest_dict = partition["manifest"]
    # The manifest is already a dict (to_dict() called by build_pf1000_material_partition)
    assert "mask_class_status" in manifest_dict
    assert isinstance(manifest_dict["mask_class_status"], dict)
    assert set(manifest_dict["mask_class_status"]) == set(sg.PF1000_MASK_CLASSES)


# ===========================================================================
# S3R.5 SigmaPSurfacePacket digest fields (A6)
# ===========================================================================

def test_s3r5_blocked_packet_carries_digest_field_sentinels() -> None:
    """S3R.5 A6: a blocked SigmaPSurfacePacket must carry the digest fields --
    even when blocked they must be present (empty-string sentinels, not absent).
    """
    sp = sg.SigmaPSurfacePacket.blocked()
    assert hasattr(sp, "sigma_p_face_set_sha256")
    assert hasattr(sp, "moving_classification_sha256")
    assert hasattr(sp, "omega_partition_sha256")
    assert hasattr(sp, "material_mask_sha256_by_class")
    assert hasattr(sp, "moving_classification_status")
    # blocked sentinel values.
    assert sp.sigma_p_face_set_sha256 == ""
    assert sp.moving_classification_sha256 == ""
    assert sp.omega_partition_sha256 == ""
    assert sp.material_mask_sha256_by_class == {}


def test_s3r5_build_sigma_p_blocked_path_preserves_geometry_hash() -> None:
    """S3R.5 A6: the blocked return path of build_sigma_p_surface_packet must
    preserve the source geometry hash so a reviewer can confirm which S3.2
    geometry drove the blocked packet.
    """
    packet = sg.PF1000GeometryPacket.krauz_2012()
    partition = _partition(packet)
    sp = sg.build_sigma_p_surface_packet(partition)
    # The blocked packet must carry the hash derived from the S3.2 manifest.
    assert sp.source_geometry_hash is not None, (
        "blocked build_sigma_p_surface_packet must preserve source geometry hash"
    )
    assert len(sp.source_geometry_hash) == 64, (
        f"expected 64-char hex SHA-256, got {sp.source_geometry_hash!r}"
    )


def test_s3r5_blocked_build_without_partition_still_has_sentinel_hash() -> None:
    """S3R.5: calling build_sigma_p_surface_packet(None) yields a blocked packet
    whose source_geometry_hash is None (no partition supplied -- that is fine,
    the field is present even if None).
    """
    sp = sg.build_sigma_p_surface_packet(None)
    assert sp.status == "blocked_sigma_p_surface_packet_not_available"
    # source_geometry_hash is an attribute even when no partition was supplied.
    assert hasattr(sp, "source_geometry_hash")


def test_s3r5_digest_fields_in_to_dict_output() -> None:
    """S3R.5 A6: digest fields must appear in to_dict() so they survive serialization."""
    sp = sg.SigmaPSurfacePacket.blocked()
    d = sp.to_dict()
    for key in (
        "sigma_p_face_set_sha256",
        "moving_classification_sha256",
        "omega_partition_sha256",
        "material_mask_sha256_by_class",
        "moving_classification_status",
    ):
        assert key in d, f"to_dict() missing S3R.5 field {key!r}"
    assert d["moving_classification_status"] == "not_classified"
