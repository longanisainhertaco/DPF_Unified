"""Source-scoped geometry packet for the candidate 3-D hybrid PIC path.

This module also carries the WP-N3 PF-1000/Akel source-tagged geometry packet
(`PF1000GeometryPacket`, S3.2) and the `Sigma_p` moving-boundary surface-term
data contract (`SigmaPSurfacePacket`, S3.3). Neither promotes validation or
first-principles acceptance; conflicting source dimensions are kept explicit
and missing dimensions fail closed with typed blockers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid

# WP-N1: Auluck Omega-domain partition labels (source packet S1).
# [KR: auluck-2021-dpf-circuit-element.md:203-257]
AULUCK_OMEGA_SOURCE_REFS = (
    "KnowledgeReference/auluck-2021-dpf-circuit-element.md:203-257",
)
AULUCK_OMEGA_LABELS = (
    "omega_volume_cells",
    "terminal_source_interface_faces",
    "wall_material_faces",
    "open_pml_faces",
)
# Auluck eq 1 requires J = 0 outside Omega; the runtime current floor
# below this value places a cell outside the current-carrying domain.
# [KR: auluck-2021-dpf-circuit-element.md:203-204]
OMEGA_CURRENT_FLOOR_A_M2 = 1.0e-6


@dataclass(frozen=True)
class HybridPICSourceGeometry:
    """Typed LLNL-like setup values extracted from the local hybrid PIC source.

    The hybrid-PIC paper is ARCHITECTURE / equation-method evidence: it backs
    the Maxwell + hybrid-PIC-fluid + generalized-Ohm + circuit-coupling method,
    NOT the selected-machine operating point.  ``architecture_source_scope``
    names that role explicitly.  ``source_scope`` is retained for back-compat
    but it is the architecture scope, never a selected-machine validation
    scope (Super-Sprint 9 WS9-2, fixes audit P0-2).
    """

    source: str = HYBRID_PIC_3D_SOURCE
    source_lines: str = "632-740"
    source_scope: str = "llnl_like_180ka_axisymmetric_hybrid_pic"
    architecture_source_scope: str = "llnl_like_180ka_axisymmetric_hybrid_pic"
    architecture_evidence_role: str = "equation_method_and_architecture_source"
    coordinate_system: str = "2d_axisymmetric_rz"
    anode_length_m: float = 0.05
    anode_radius_m: float = 0.01
    cathode_length_m: float = 0.10
    physical_radius_m: float = 0.015
    physical_length_m: float = 0.10
    source_radial_cells: int = 77
    source_axial_cells: int = 522
    cell_size_m: float = 2.0e-4
    axial_pml_layers: int = 20
    source_dt_s: float = 4.25e-13
    sheath_thickness_m: float = 1.0e-3
    sheath_density_m3: float = 3.3e23
    background_density_m3: float = 6.7e22
    background_particles: int = 500_000
    sheath_particles: int = 26_060
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def smoke_grid(
        self,
        *,
        shape: tuple[int, int, int],
    ) -> Maxwell3DGrid:
        """Return a Cartesian engineering-smoke grid spanning the source domain."""
        if len(shape) != 3:
            raise ValueError("shape must be a 3-tuple")
        nx, ny, nz = (int(v) for v in shape)
        if min(nx, ny, nz) < 2:
            raise ValueError("all smoke-grid dimensions must be >= 2")
        return Maxwell3DGrid(
            shape=(nx, ny, nz),
            spacing=(
                2.0 * self.physical_radius_m / nx,
                2.0 * self.physical_radius_m / ny,
                self.physical_length_m / nz,
            ),
        )


def _mask_sha256(mask: np.ndarray) -> str:
    """Return a deterministic SHA-256 over a boolean partition mask."""
    array = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


def _mask_bounds(mask: np.ndarray) -> dict[str, Any]:
    """Return the index-space axis-aligned bounding box of a boolean mask."""
    array = np.asarray(mask, dtype=bool)
    if not np.any(array):
        return {
            "non_empty": False,
            "i_min": None,
            "i_max": None,
            "j_min": None,
            "j_max": None,
            "k_min": None,
            "k_max": None,
        }
    idx = np.argwhere(array)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    return {
        "non_empty": True,
        "i_min": int(lo[0]),
        "i_max": int(hi[0]),
        "j_min": int(lo[1]),
        "j_max": int(hi[1]),
        "k_min": int(lo[2]),
        "k_max": int(hi[2]),
    }


def _label_packet(label: str, mask: np.ndarray) -> dict[str, Any]:
    array = np.asarray(mask, dtype=bool)
    return {
        "label": label,
        "mask_sha256": _mask_sha256(array),
        "cell_count": int(np.count_nonzero(array)),
        "bounds": _mask_bounds(array),
        "source_refs": list(AULUCK_OMEGA_SOURCE_REFS),
    }


def build_auluck_omega_domain(
    *,
    grid_shape: tuple[int, int, int],
    electron_density_m3: np.ndarray,
    current_density_norm_A_m2: np.ndarray,
    source_interface_z_index: int,
    pml_layers: int,
    electron_density_floor_m3: float,
) -> dict[str, Any]:
    """Return the Auluck four-label disjoint exhaustive cell partition.

    Implements WP-N1 source packet S1. The named integration domain Omega is
    the current-carrying plasma volume; per Auluck eq 1 J is zero outside it
    [KR: auluck-2021-dpf-circuit-element.md:203-204]. The interface with the
    external power source is excluded from Omega
    [KR: auluck-2021-dpf-circuit-element.md:205-209].

    The four labels (omega_volume_cells, terminal_source_interface_faces,
    wall_material_faces, open_pml_faces) form a mutually disjoint, exhaustive
    per-cell partition of the (nx, ny, nz) grid: every cell carries exactly
    one label. This is a candidate engineering partition; the precise
    reviewed material geometry is audit finding A-7 (gap G4) and not yet
    available, so wall_material_faces is an engineering approximation.
    """
    nx, ny, nz = (int(v) for v in grid_shape)
    if len(grid_shape) != 3:
        raise ValueError("grid_shape must be a 3-tuple")
    if int(source_interface_z_index) != source_interface_z_index:
        raise ValueError("source_interface_z_index must be an integer")
    k_port = int(source_interface_z_index)
    if k_port < 0 or k_port >= nz:
        raise ValueError("source_interface_z_index must address an axial slice")
    if int(pml_layers) != pml_layers or pml_layers < 0:
        raise ValueError("pml_layers must be a non-negative integer")

    density = np.asarray(electron_density_m3, dtype=float)
    current_norm = np.asarray(current_density_norm_A_m2, dtype=float)
    if density.shape != (nx, ny, nz):
        raise ValueError("electron_density_m3 shape does not match grid")
    if current_norm.shape != (nx, ny, nz):
        raise ValueError("current_density_norm_A_m2 shape does not match grid")

    # terminal_source_interface_faces: the k = k_port axial slab. Per Auluck
    # this is the cathode-plate / insulator / squirrel-cage interface with the
    # external power source, EXCLUDED from Omega.
    source_interface = np.zeros((nx, ny, nz), dtype=bool)
    source_interface[:, :, k_port] = True

    # open_pml_faces: outer axial PML layers, minus the source interface.
    open_pml = np.zeros((nx, ny, nz), dtype=bool)
    if pml_layers > 0:
        layers = min(int(pml_layers), nz)
        open_pml[:, :, :layers] = True
        open_pml[:, :, nz - layers:] = True
    open_pml &= ~source_interface

    # omega_volume_cells: current-carrying plasma (n_e above the numerical
    # floor AND |J| above the current floor), with the source interface and
    # the PML region removed. Auluck eq 1 domain interior.
    density_threshold = float(electron_density_floor_m3) * (1.0 + 1.0e-12)
    current_carrying = (density > density_threshold) & (
        current_norm > OMEGA_CURRENT_FLOOR_A_M2
    )
    omega_volume = current_carrying & ~source_interface & ~open_pml

    # wall_material_faces: every remaining cell (the exhaustive complement).
    # These are non-Omega, non-source-interface, non-PML cells: material /
    # vacuum-floor cells that bound Omega. This keeps the partition exhaustive.
    wall_material = ~(omega_volume | source_interface | open_pml)

    labels = {
        "omega_volume_cells": omega_volume,
        "terminal_source_interface_faces": source_interface,
        "wall_material_faces": wall_material,
        "open_pml_faces": open_pml,
    }

    # Enforce: mutually disjoint and exhaustive.
    stacked = np.stack([labels[name] for name in AULUCK_OMEGA_LABELS], axis=0)
    per_cell_label_count = stacked.sum(axis=0)
    mutually_disjoint = bool(np.all(per_cell_label_count <= 1))
    exhaustive = bool(np.all(per_cell_label_count == 1))
    source_interface_non_empty = bool(np.any(source_interface))
    source_interface_disjoint_from_omega = not bool(
        np.any(source_interface & omega_volume)
    )
    omega_has_only_current_carrying = not bool(
        np.any(omega_volume & ~current_carrying)
    )

    return {
        "status": "candidate_auluck_omega_domain_partition_not_validation",
        "source_refs": list(AULUCK_OMEGA_SOURCE_REFS),
        "labels": list(AULUCK_OMEGA_LABELS),
        # Boolean label arrays for downstream integration. Excluded from the
        # emitted JSON artifact by the runner; consumed only in-process.
        "_label_masks": labels,
        "partition_kind": "per_cell_exhaustive_disjoint_label",
        "grid_shape": [nx, ny, nz],
        "source_interface_z_index": k_port,
        "pml_layers": int(pml_layers),
        "electron_density_floor_m3": float(electron_density_floor_m3),
        "omega_current_floor_A_m2": OMEGA_CURRENT_FLOOR_A_M2,
        "omega_volume_cells": _label_packet("omega_volume_cells", omega_volume),
        "terminal_source_interface_faces": _label_packet(
            "terminal_source_interface_faces", source_interface
        ),
        "wall_material_faces": _label_packet("wall_material_faces", wall_material),
        "open_pml_faces": _label_packet("open_pml_faces", open_pml),
        "partition_constraints": {
            "mutually_disjoint": mutually_disjoint,
            "exhaustive": exhaustive,
            "terminal_source_interface_non_empty": source_interface_non_empty,
            "terminal_source_interface_disjoint_from_omega": (
                source_interface_disjoint_from_omega
            ),
            "omega_contains_only_current_carrying_cells": (
                omega_has_only_current_carrying
            ),
            "max_labels_per_cell": int(per_cell_label_count.max()),
            "min_labels_per_cell": int(per_cell_label_count.min()),
        },
        "geometry_review_status": "geometry_candidate_not_reviewed",
        "can_support_power_port_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def omega_domain_label_masks(
    omega_domain: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return the boolean label arrays for an Auluck Omega partition packet."""
    masks = omega_domain.get("_label_masks")
    if not isinstance(masks, dict):
        raise ValueError("omega_domain packet has no _label_masks")
    return masks


def public_omega_domain_packet(
    omega_domain: dict[str, Any],
) -> dict[str, Any]:
    """Return the Omega partition packet without in-process boolean arrays."""
    return {key: value for key, value in omega_domain.items() if key != "_label_masks"}


def source_geometry_candidate_evidence(
    geometry: HybridPICSourceGeometry,
) -> dict[str, Any]:
    """Build non-promoting ARCHITECTURE 3-D geometry evidence.

    This packet carries the LLNL-like hybrid-PIC ``architecture_source_scope``.
    It is architecture / equation-method evidence ONLY: a ``same_scope``-named
    key must never carry this scope, so the capability is reported as
    ``architecture_3d_geometry_candidate_packet`` (Super-Sprint 10 SS10-1,
    closes audit A1).
    """
    return {
        "passed": True,
        "status": "candidate",
        "capability": "architecture_3d_geometry_candidate_packet",
        "evidence_role": "architecture_and_equation_method_geometry_only",
        "source": geometry.source,
        "source_lines": geometry.source_lines,
        "implementation": "src/dpf/fields/source_geometry.py",
        "evidence_type": "engineering_architecture_geometry_packet",
        # This is the architecture/equation-method scope, never a selected-
        # machine same-scope validation scope.  ``source_scope`` is retained
        # for back-compat but it equals ``architecture_source_scope``.
        "architecture_source_scope": geometry.architecture_source_scope,
        "architecture_evidence_role": geometry.architecture_evidence_role,
        "source_scope": geometry.source_scope,
        "coordinate_system": geometry.coordinate_system,
        "is_same_scope_validation_evidence": False,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "The extracted source setup is 2-D axisymmetric, not an accepted true-3D geometry packet.",
            "The smoke grid is a Cartesian engineering projection for code exercise only.",
            "LLNL-like architecture geometry is NOT selected-machine same-scope validation evidence.",
            "No same-scope PF-1000/Akel experimental validation packet is attached.",
        ],
    }


# ===========================================================================
# S3.2 -- WP-N3 PF-1000 / Akel source-tagged geometry packet.
#
# Replaces the projection-only candidate geometry with a source-tagged runtime
# packet. Conflicting source dimensions (12 vs 24 rods, 460/480/600/450 mm
# anode length) are kept as explicit `PF1000GeometryConflict` records and are
# NEVER averaged. Missing or wrong-scope dimensions (anode bore, insulator
# outer radius, backplate, and other unresolved WP-N3 rows) are
# typed `blocked` fields with blocker IDs and are NEVER invented.
#
# Authority: docs/external_team_submissions/2026_05_18_three_sprint_blocker_
#   packet/sprint_3/WP_N3_GEOMETRY_SOURCE_PACKET.md (research packet).
# Every numeric value below cites the local KR path with a line range; see the
# WP-N3 packet section 1.x and section 8 for the per-line provenance.
# ===========================================================================

# WP-N3 KR source register (PF-1000 family). A test asserts each file exists.
# [WP_N3_GEOMETRY_SOURCE_PACKET.md section 8]
PF1000_GEOMETRY_SOURCE_REFS = (
    "KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-"
    "current-sheath-on-the-pf-1000-facility-705bcc83.md:342-358",
    "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
    ":111-114,264-268",
    "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md:191-225",
    "KnowledgeReference/scholz-2006-pf1000-mega-joule.md:22-33",
    "KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md:56-63",
    "KnowledgeReference/final-stages-of-the-plasma-column-evolution-in-the-"
    "plasma-focus-pf1000-device-plasma-scien-fa128cfd.md:38-43",
    "KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-"
    "and-s-h-saw-part-1-basic-course.md:2199-2210",
    "KnowledgeReference/auluck-2021-dpf-circuit-element.md:203-223,426-431",
    "KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-"
    "caused-by-the-plasma-focus-vessel-527cc533.md:113-118",
    "KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-"
    "d3e51f6c.md:90-104",
    "KnowledgeReference/pf-1000-device-a2d6bc15.md:83-154",
)

# The 10 source-tagged material/partition mask classes the runtime must emit.
# Labels 0-3 are the Auluck top-level partition; 4-8 are the material
# sub-classes that refine `wall_material_faces`; 9 is the open/PML boundary.
# [WP_N3_GEOMETRY_SOURCE_PACKET.md section 3.3; handoff S3.2 "Required masks"]
PF1000_MASK_CLASSES = (
    "omega_volume_cells",
    "terminal_source_interface_faces",
    "wall_material_faces",
    "open_pml_faces",
    "anode_material_faces",
    "cathode_rod_faces",
    "insulator_material_faces",
    "chamber_wall_faces",
    "backplate_source_interface_faces",
    "pml_or_open_boundary_faces",
)

# Material sub-classes that refine `wall_material_faces` -- these must be
# mutually disjoint and their union must equal `wall_material_faces`.
PF1000_MATERIAL_SUBCLASSES = (
    "anode_material_faces",
    "cathode_rod_faces",
    "insulator_material_faces",
    "chamber_wall_faces",
    "backplate_source_interface_faces",
)


@dataclass(frozen=True)
class PF1000GeometryField:
    """One source-tagged PF-1000 geometry field.

    A field is `source_supported` when a single KR source backs the value,
    `conflict` when multiple KR sources disagree (the value is then `None` and
    the disagreement is held in a `PF1000GeometryConflict`), and `blocked`
    when no KR source provides the dimension (value `None`, blocker ID set).
    `candidate` marks a numerical solver parameter that is not a measured
    device dimension.
    """

    name: str
    value: float | int | None
    units: str
    status: str  # source_supported | candidate | conflict | blocked
    scope_tag: str
    source_ref: str | None = None
    blocker_id: str | None = None
    conflict_group: str | None = None

    def __post_init__(self) -> None:
        allowed = {"source_supported", "candidate", "conflict", "blocked"}
        if self.status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        if self.status in {"source_supported", "candidate"}:
            if self.value is None:
                raise ValueError(f"{self.name}: {self.status} field needs a value")
            if not self.source_ref:
                raise ValueError(f"{self.name}: {self.status} field needs source_ref")
        if self.status == "blocked":
            if self.value is not None:
                raise ValueError(f"{self.name}: blocked field must have value None")
            if not self.blocker_id:
                raise ValueError(f"{self.name}: blocked field needs a blocker_id")
        if self.status == "conflict":
            if self.value is not None:
                raise ValueError(f"{self.name}: conflict field must have value None")
            if not self.conflict_group:
                raise ValueError(f"{self.name}: conflict field needs a conflict_group")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PF1000GeometryConflict:
    """An unresolved PF-1000 dimension where KR sources disagree.

    Each candidate value carries its own KR source ref. The conflict is kept
    EXPLICIT -- the runtime never averages the candidates. The WP-N3 packet
    section 4 records that each value belongs to a distinct PF-1000 hardware
    revision; a runtime must pin one revision via `geometry_source_tag`.
    """

    group: str
    field_name: str
    units: str
    candidate_values: tuple[float | int, ...]
    candidate_source_refs: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if len(self.candidate_values) < 2:
            raise ValueError("a conflict needs at least two candidate values")
        if len(self.candidate_values) != len(self.candidate_source_refs):
            raise ValueError("each candidate value needs one source ref")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PF1000MaskManifest:
    """Deterministic manifest for one PF-1000 source-tagged mask build.

    S3R.4: each mask class now carries an explicit `mask_class_status` string:
      * ``source_supported``                -- produced from exact KR-backed dims
      * ``candidate_projection_not_source_mask`` -- produced but relies on grid
        heuristics because the driving dimension is conflict or blocked
      * ``blocked``                         -- not produced; a driving dimension
        is blocked; no SHA-256 is emitted for this class
    A SHA-256 is populated only when the mask is actually produced (status is
    NOT ``blocked``). Blocked mask classes carry an empty-string sentinel ("").
    """

    geometry_packet_id: str
    geometry_source_tag: str
    source_refs: tuple[str, ...]
    conflict_groups: tuple[str, ...]
    blocked_fields: tuple[str, ...]
    grid_shape: tuple[int, int, int]
    grid_spacing_m: tuple[float, float, float]
    mask_sha256_by_class: dict[str, str]
    mask_cell_counts: dict[str, int]
    under_resolution_flags: dict[str, bool]
    # S3R.4: per-class status strings (source_supported /
    # candidate_projection_not_source_mask / blocked).
    mask_class_status: dict[str, str] = dataclass_field(default_factory=dict)
    can_support_first_principles_acceptance: bool = False

    def __post_init__(self) -> None:
        missing = [
            name for name in PF1000_MASK_CLASSES
            if name not in self.mask_sha256_by_class
        ]
        if missing:
            raise ValueError(
                "PF1000MaskManifest is missing per-class hashes for: "
                + ", ".join(sorted(missing))
            )
        if self.can_support_first_principles_acceptance:
            raise ValueError(
                "PF1000MaskManifest must not claim first-principles acceptance"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PF1000GeometryPacket:
    """Source-tagged PF-1000 / Akel geometry packet (WP-N3 / S3.2).

    Each constructor pins ONE self-consistent KR source set. Conflicting
    fields differ between constructors; no constructor averages across PF-1000
    hardware revisions. Blocked fields are typed `PF1000GeometryField` with
    status `blocked` and a blocker ID; their numeric value stays `None`.
    """

    geometry_packet_id: str
    geometry_source_tag: str
    scope_tag: str
    source_refs: tuple[str, ...]
    fields: dict[str, PF1000GeometryField]
    conflicts: dict[str, PF1000GeometryConflict] = dataclass_field(
        default_factory=dict
    )
    geometry_review_status: str = "geometry_candidate_not_reviewed"
    can_support_first_principles_acceptance: bool = False

    def __post_init__(self) -> None:
        if self.can_support_first_principles_acceptance:
            raise ValueError(
                "PF1000GeometryPacket must not claim first-principles acceptance"
            )

    def get_field(self, name: str) -> PF1000GeometryField:
        if name not in self.fields:
            raise KeyError(f"PF1000GeometryPacket has no field {name!r}")
        return self.fields[name]

    def blocked_field_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name for name, fld in self.fields.items()
                if fld.status == "blocked"
            )
        )

    def source_supported_field_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name for name, fld in self.fields.items()
                if fld.status == "source_supported"
            )
        )

    def conflict_field_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name for name, fld in self.fields.items()
                if fld.status == "conflict"
            )
        )

    def candidate_field_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name for name, fld in self.fields.items()
                if fld.status == "candidate"
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_packet_id": self.geometry_packet_id,
            "geometry_source_tag": self.geometry_source_tag,
            "scope_tag": self.scope_tag,
            "source_refs": list(self.source_refs),
            "fields": {n: f.to_dict() for n, f in self.fields.items()},
            "conflicts": {g: c.to_dict() for g, c in self.conflicts.items()},
            "geometry_review_status": self.geometry_review_status,
            "can_support_first_principles_acceptance": (
                self.can_support_first_principles_acceptance
            ),
        }

    # --- source-tagged constructors ---------------------------------------

    @classmethod
    def krauz_2012(cls) -> PF1000GeometryPacket:
        """PF-1000 geometry as reported by Krauz et al. 2012 (KR-KRAUZ12).

        [KR: experimental-study-of-the-structure-of-the-plasma-current-sheath-
        on-the-pf-1000-facility-705bcc83.md:342-358] anode radius 115.5 mm,
        anode length 460 mm, cathode-cage geometric radius 200 mm, 12 rods,
        rod diameter 80 mm, insulator exposed length 85 mm, chamber inner
        radius 700 mm (1400 mm diameter), chamber length 2500 mm.
        """
        tag = "pf1000_krauz2012"
        scope = "pf1000_full_energy_revision"
        krauz = (
            "KnowledgeReference/experimental-study-of-the-structure-of-the-"
            "plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        )
        return cls(
            geometry_packet_id="pf1000_geometry_packet_krauz2012",
            geometry_source_tag=tag,
            scope_tag=scope,
            source_refs=PF1000_GEOMETRY_SOURCE_REFS,
            fields=cls._fields_for(
                scope_tag=scope,
                anode_radius_m=0.1155,
                anode_radius_ref=f"{krauz}:346-347",
                anode_length_m=0.460,
                anode_length_ref=f"{krauz}:347",
                anode_length_status="conflict",
                anode_length_conflict="anode_length_z0",
                cathode_cage_radius_m=0.200,
                cathode_cage_radius_ref=f"{krauz}:346-347",
                cathode_cage_radius_status="conflict",
                cathode_cage_conflict="cathode_cage_radius_b",
                cathode_rod_count=12,
                cathode_rod_count_ref=f"{krauz}:344-345",
                cathode_rod_count_status="conflict",
                cathode_rod_count_conflict="cathode_rod_count",
                cathode_rod_diameter_m=0.080,
                cathode_rod_diameter_ref=f"{krauz}:345",
                insulator_exposed_length_m=0.085,
                insulator_exposed_length_ref=f"{krauz}:349-350",
                insulator_exposed_length_status="conflict",
                insulator_exposed_length_conflict="insulator_exposed_length",
                chamber_inner_radius_m=0.700,
                chamber_inner_radius_ref=f"{krauz}:342-343",
                chamber_length_m=2.500,
                chamber_length_ref=f"{krauz}:343",
            ),
            conflicts=cls._conflicts(),
        )

    @classmethod
    def akel_shot_12581(cls) -> PF1000GeometryPacket:
        """PF-1000 geometry for the Akel et al. 2021 shot-12581 16 kV scope.

        [KR: radiation-physics-and-chemistry-188-2021-109633.md:111-114,
        264-268] Lee-model fit: anode radius a = 11.55 cm, anode length
        z0 = 48 cm, cathode-cage Lee-fit radius b = 16 cm, 12 rods, rod
        diameter 80 mm (8 cm tubes). Chamber dimensions inherit from KR-KRAUZ12
        (Akel reports no distinct chamber bore).
        """
        tag = "pf1000_akel_shot12581"
        scope = "pf1000_akel_16kv_1p2torr_shot_12581"
        akel = "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
        krauz = (
            "KnowledgeReference/experimental-study-of-the-structure-of-the-"
            "plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        )
        return cls(
            geometry_packet_id="pf1000_geometry_packet_akel_shot12581",
            geometry_source_tag=tag,
            scope_tag=scope,
            source_refs=PF1000_GEOMETRY_SOURCE_REFS,
            fields=cls._fields_for(
                scope_tag=scope,
                anode_radius_m=0.1155,
                anode_radius_ref=f"{akel}:264",
                anode_length_m=0.480,
                anode_length_ref=f"{akel}:111,264",
                anode_length_status="conflict",
                anode_length_conflict="anode_length_z0",
                cathode_cage_radius_m=0.160,
                cathode_cage_radius_ref=f"{akel}:264",
                cathode_cage_radius_status="conflict",
                cathode_cage_conflict="cathode_cage_radius_b",
                cathode_rod_count=12,
                cathode_rod_count_ref=f"{akel}:112-114",
                cathode_rod_count_status="conflict",
                cathode_rod_count_conflict="cathode_rod_count",
                cathode_rod_diameter_m=0.080,
                cathode_rod_diameter_ref=f"{akel}:113",
                insulator_exposed_length_m=0.085,
                insulator_exposed_length_ref=f"{krauz}:349-350",
                insulator_exposed_length_status="conflict",
                insulator_exposed_length_conflict="insulator_exposed_length",
                chamber_inner_radius_m=0.700,
                chamber_inner_radius_ref=f"{krauz}:342-343",
                chamber_length_m=2.500,
                chamber_length_ref=f"{krauz}:343",
            ),
            conflicts=cls._conflicts(),
        )

    @classmethod
    def scholz_gribkov_revision(cls) -> PF1000GeometryPacket:
        """PF-1000 geometry for the Scholz/Gribkov 2006-2007 hardware revision.

        [KR: scholz-2007-pf1000-part2-jphysd.md:191-225] anode diameter
        230 mm (radius 115.0 mm), anode length 600 mm, insulator exposed
        length 113 mm. Rod count and cathode-cage radius are NOT separately
        numerically stated in this revision's KR extract -- they remain
        conflict fields. This constructor exists only because it keeps the
        revision conflicts explicit (handoff S3.2: constructor allowed "if and
        only if the source packet can keep revision conflicts explicit").
        """
        tag = "pf1000_scholz_gribkov_revision"
        scope = "pf1000_full_energy_revision"
        scholz = "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md"
        krauz = (
            "KnowledgeReference/experimental-study-of-the-structure-of-the-"
            "plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        )
        return cls(
            geometry_packet_id="pf1000_geometry_packet_scholz_gribkov",
            geometry_source_tag=tag,
            scope_tag=scope,
            source_refs=PF1000_GEOMETRY_SOURCE_REFS,
            fields=cls._fields_for(
                scope_tag=scope,
                anode_radius_m=0.1150,
                anode_radius_ref=f"{scholz}:198",
                anode_length_m=0.600,
                anode_length_ref=f"{scholz}:198",
                anode_length_status="conflict",
                anode_length_conflict="anode_length_z0",
                cathode_cage_radius_m=0.200,
                cathode_cage_radius_ref=f"{krauz}:346-347",
                cathode_cage_radius_status="conflict",
                cathode_cage_conflict="cathode_cage_radius_b",
                cathode_rod_count=12,
                cathode_rod_count_ref=f"{krauz}:344-345",
                cathode_rod_count_status="conflict",
                cathode_rod_count_conflict="cathode_rod_count",
                cathode_rod_diameter_m=0.080,
                cathode_rod_diameter_ref=f"{krauz}:345",
                insulator_exposed_length_m=0.113,
                insulator_exposed_length_ref=f"{scholz}:223-225",
                insulator_exposed_length_status="conflict",
                insulator_exposed_length_conflict="insulator_exposed_length",
                chamber_inner_radius_m=0.700,
                chamber_inner_radius_ref=f"{krauz}:342-343",
                chamber_length_m=2.500,
                chamber_length_ref=f"{krauz}:343",
            ),
            conflicts=cls._conflicts(),
        )

    @classmethod
    def scholz_2001_24rod_large_electrode(cls) -> PF1000GeometryPacket:
        """PF-1000 2000/2001 24-rod large-electrode hardware packet.

        [KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]
        reports 24 stainless-steel rods, 600 mm rod length, 32 mm rod
        diameter, 400 mm outer-electrode diameter, 244 mm inner-electrode
        diameter, 62 mm interelectrode gap, and a 229 mm diameter / 113 mm
        alumina insulator.  [KR: pf-1000-device-a2d6bc15.md:129-154]
        supplies matching early PF-1000 facility, chamber, and bank context.

        This constructor is revision-scoped source consumption only.  It does
        not promote the default Akel/Krauz runtime constructors and still fails
        closed on bore length, insulator wall thickness, backplate dimensions,
        and same-scope 3-D review.
        """
        tag = "pf1000_scholz_2001_24rod_large_electrode"
        scope = "pf1000_2001_24_rod_large_electrode_hardware"
        scholz2001 = (
            "KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-"
            "d3e51f6c.md"
        )
        scholz2000 = "KnowledgeReference/pf-1000-device-a2d6bc15.md"
        return cls(
            geometry_packet_id="pf1000_geometry_packet_scholz_2001_24rod",
            geometry_source_tag=tag,
            scope_tag=scope,
            source_refs=PF1000_GEOMETRY_SOURCE_REFS,
            fields=cls._fields_for(
                scope_tag=scope,
                anode_radius_m=0.122,
                anode_radius_ref=f"{scholz2001}:93-94",
                anode_length_m=0.600,
                anode_length_ref=f"{scholz2000}:88-90",
                anode_length_status="source_supported",
                anode_length_conflict="anode_length_z0",
                cathode_cage_radius_m=0.200,
                cathode_cage_radius_ref=f"{scholz2001}:90-93",
                cathode_cage_radius_status="source_supported",
                cathode_cage_conflict="cathode_cage_radius_b",
                cathode_rod_count=24,
                cathode_rod_count_ref=f"{scholz2001}:90-92",
                cathode_rod_count_status="source_supported",
                cathode_rod_count_conflict="cathode_rod_count",
                cathode_rod_diameter_m=0.032,
                cathode_rod_diameter_ref=f"{scholz2001}:90-92",
                cathode_rod_length_m=0.600,
                cathode_rod_length_ref=f"{scholz2001}:90-92",
                insulator_exposed_length_m=0.113,
                insulator_exposed_length_ref=f"{scholz2001}:96-98",
                insulator_exposed_length_status="source_supported",
                insulator_exposed_length_conflict="insulator_exposed_length",
                insulator_outer_radius_m=0.1145,
                insulator_outer_radius_ref=f"{scholz2001}:96-98",
                chamber_inner_radius_m=0.700,
                chamber_inner_radius_ref=f"{scholz2000}:134-136",
                chamber_length_m=2.500,
                chamber_length_ref=f"{scholz2000}:134-136",
            ),
            conflicts=cls._conflicts(),
        )

    # --- internal field/conflict builders ---------------------------------

    @staticmethod
    def _conflicts() -> dict[str, PF1000GeometryConflict]:
        """Return the WP-N3 section-4 unresolved geometry conflicts.

        These are sourced dimensions where KR sources disagree because they
        describe distinct PF-1000 hardware revisions. They are NEVER averaged.
        [WP_N3_GEOMETRY_SOURCE_PACKET.md section 4]
        """
        krauz = (
            "KnowledgeReference/experimental-study-of-the-structure-of-the-"
            "plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        )
        akel = "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md"
        scholz = "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md"
        lee = (
            "KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-"
            "s-lee-and-s-h-saw-part-1-basic-course.md"
        )
        finals = (
            "KnowledgeReference/final-stages-of-the-plasma-column-evolution-in-"
            "the-plasma-focus-pf1000-device-plasma-scien-fa128cfd.md"
        )
        records = (
            PF1000GeometryConflict(
                group="cathode_rod_count",
                field_name="cathode_rod_count",
                units="count",
                candidate_values=(12, 24),
                candidate_source_refs=(
                    f"{krauz}:344-345", f"{finals}:38-39",
                ),
                reason=(
                    "12 rods (Krauz 2012 / Akel 2021) vs 24 rods (Final Stages)"
                    " -- distinct PF-1000 hardware revisions; not averaged"
                ),
            ),
            PF1000GeometryConflict(
                group="cathode_cage_radius_b",
                field_name="cathode_cage_radius_m",
                units="m",
                candidate_values=(0.200, 0.160),
                candidate_source_refs=(
                    f"{krauz}:346-347", f"{akel}:264",
                ),
                reason=(
                    "Sprint 4 hardware-scope review verdict (2026-05-20): "
                    "the two values are NOT measuring the same quantity. "
                    "Krauz 2012 [KR:346-347] states 'OE and copper center "
                    "electrode (CE) radii are 200 mm and 115.5 mm' -- this "
                    "is a direct geometric measurement of the outer electrode "
                    "cage radius. Akel 2021 [KR:264] lists 'b = 16 cm' in a "
                    "Lee model code parameter table; line 267 labels b the "
                    "'cathode radius' but it is a Lee-fit input, not a "
                    "hardware metrology value. The conflict therefore "
                    "reflects a category mismatch (hardware geometry vs "
                    "Lee-fit parameter) rather than a genuine measurement "
                    "disagreement. The physical hardware cage geometric "
                    "radius is 200 mm per Krauz and is now corroborated by "
                    "the Scholz 2000/2001 PF-1000 hardware papers. The field "
                    "cathode_cage_radius_m nevertheless remains "
                    "status=conflict in the active Akel/Krauz constructors "
                    "until a revision-selection policy maps hardware sources "
                    "to the requested simulation scope. Not averaged."
                ),
            ),
            PF1000GeometryConflict(
                group="anode_length_z0",
                field_name="anode_length_m",
                units="m",
                candidate_values=(0.460, 0.480, 0.600, 0.450),
                candidate_source_refs=(
                    f"{krauz}:347", f"{akel}:264", f"{scholz}:198",
                    f"{finals}:40-41",
                ),
                reason=(
                    "anode length z0 is 460/480/600/450 mm across PF-1000 "
                    "hardware revisions and Lee-fit periods; not averaged"
                ),
            ),
            PF1000GeometryConflict(
                group="insulator_exposed_length",
                field_name="insulator_exposed_length_m",
                units="m",
                candidate_values=(0.085, 0.113),
                candidate_source_refs=(
                    f"{krauz}:349-350", f"{scholz}:223-225",
                ),
                reason=(
                    "insulator exposed length 85 mm (Krauz 2012, new "
                    "insulator) vs 113 mm (Scholz 2007 / Final Stages); "
                    "not averaged"
                ),
            ),
        )
        _ = lee  # cited in PF1000_GEOMETRY_SOURCE_REFS; revision evidence only
        return {record.group: record for record in records}

    @staticmethod
    def _fields_for(
        *,
        scope_tag: str,
        anode_radius_m: float,
        anode_radius_ref: str,
        anode_length_m: float,
        anode_length_ref: str,
        anode_length_status: str,
        anode_length_conflict: str,
        cathode_cage_radius_m: float,
        cathode_cage_radius_ref: str,
        cathode_cage_radius_status: str,
        cathode_cage_conflict: str,
        cathode_rod_count: int,
        cathode_rod_count_ref: str,
        cathode_rod_count_status: str,
        cathode_rod_count_conflict: str,
        cathode_rod_diameter_m: float,
        cathode_rod_diameter_ref: str,
        insulator_exposed_length_m: float,
        insulator_exposed_length_ref: str,
        insulator_exposed_length_status: str,
        insulator_exposed_length_conflict: str,
        chamber_inner_radius_m: float,
        chamber_inner_radius_ref: str,
        chamber_length_m: float,
        chamber_length_ref: str,
        cathode_rod_length_m: float | None = None,
        cathode_rod_length_ref: str | None = None,
        insulator_outer_radius_m: float | None = None,
        insulator_outer_radius_ref: str | None = None,
    ) -> dict[str, PF1000GeometryField]:
        """Build the typed geometry-field map for one source-tagged revision.

        Source-supported fields carry a single KR ref. Conflict fields carry
        value `None` and a conflict group (the chosen revision's value is held
        in the conflict record's candidate list). The still-unresolved WP-N3
        rows (anode bore runtime authority, bore length, insulator wall
        thickness, backplate radial extent / axial thickness, and end-cap
        geometry) become typed `blocked` fields.  Revision-specific callers may
        supply cathode rod length and insulator outer radius only when target
        extraction has a direct hardware source for the requested scope.
        """
        def conflict(name: str, units: str, group: str) -> PF1000GeometryField:
            return PF1000GeometryField(
                name=name, value=None, units=units, status="conflict",
                scope_tag=scope_tag, conflict_group=group,
            )

        def blocked(name: str, units: str, blocker_id: str) -> PF1000GeometryField:
            return PF1000GeometryField(
                name=name, value=None, units=units, status="blocked",
                scope_tag=scope_tag, blocker_id=blocker_id,
            )

        def supported(
            name: str, value: float | int, units: str, ref: str
        ) -> PF1000GeometryField:
            return PF1000GeometryField(
                name=name, value=value, units=units, status="source_supported",
                scope_tag=scope_tag, source_ref=ref,
            )

        krasa = (
            "KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-"
            "neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        )
        fields: dict[str, PF1000GeometryField] = {}
        # anode radius is source-supported (WP-N3 row 6).
        fields["anode_radius_m"] = supported(
            "anode_radius_m", anode_radius_m, "m", anode_radius_ref
        )
        # anode length z0 is a conflict field (WP-N3 row 7).
        if anode_length_status == "conflict":
            fld = conflict("anode_length_m", "m", anode_length_conflict)
        else:
            fld = supported(
                "anode_length_m", anode_length_m, "m", anode_length_ref
            )
        fields["anode_length_m"] = fld
        # anode material is copper (WP-N3 row 8). Encoded as a unit-flag
        # numeric field (units name carries the material identity) so the
        # PF1000GeometryField numeric-value contract holds.
        fields["anode_material_is_copper"] = supported(
            "anode_material_is_copper", 1, "copper_material_flag",
            anode_radius_ref,
        )
        # anode hollow bore -- WP-N3 rows 9/10.  Sprint 4 hardware-scope review
        # verdict (2026-05-20): Stepniewski 2004
        # [KR: doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md:310-314] lists the
        # PF-1000 geometry as "parameters ... taken for the simulations" --
        # verbatim: "The parameters of PF-1000 facility have been taken for the
        # simulations. They are as follows: radius of the inner electrode 0.12 m,
        # outer electrode 0.18 m, hollow radius in the centre of the electrode
        # 0.015 m, electrode length 0.60 m."  This is a SIMULATION PARAMETER
        # section, not a hardware drawing or metrology report.  The 0.015 m
        # value is therefore a simulation-scope value and does NOT qualify as a
        # hardware-scope PF-1000 geometry field for this packet.
        # Verdict: BLOCKED (hardware-scope review failed -- simulation parameter
        # only; no independent KR hardware measurement available).
        fields["anode_hollow_bore_radius_m"] = blocked(
            "anode_hollow_bore_radius_m", "m",
            "PF1000-BLK-009-anode-bore-radius-target_extracted_modeling_context_requires_review",
        )
        fields["anode_hollow_bore_length_m"] = blocked(
            "anode_hollow_bore_length_m", "m",
            "PF1000-BLK-010-anode-bore-length-no-kr-source",
        )
        # anode end-cap / lid geometry -- WP-N3 row 11: KR-SCHOLZ07 198-201 is
        # qualitative only ("same or a slightly larger diameter"). Blocked.
        fields["anode_end_cap_diameter_m"] = blocked(
            "anode_end_cap_diameter_m", "m",
            "PF1000-BLK-011-anode-end-cap-geometry-no-kr-number",
        )
        # cathode-cage radius is a conflict field (WP-N3 row 5).
        if cathode_cage_radius_status == "conflict":
            fields["cathode_cage_radius_m"] = conflict(
                "cathode_cage_radius_m", "m", cathode_cage_conflict
            )
        else:
            fields["cathode_cage_radius_m"] = supported(
                "cathode_cage_radius_m", cathode_cage_radius_m, "m",
                cathode_cage_radius_ref,
            )
        # cathode rod count is a conflict field (WP-N3 row 1).
        if cathode_rod_count_status == "conflict":
            fields["cathode_rod_count"] = conflict(
                "cathode_rod_count", "count", cathode_rod_count_conflict
            )
        else:
            fields["cathode_rod_count"] = supported(
                "cathode_rod_count", cathode_rod_count, "count",
                cathode_rod_count_ref,
            )
        fields["cathode_rod_diameter_m"] = supported(
            "cathode_rod_diameter_m", cathode_rod_diameter_m, "m",
            cathode_rod_diameter_ref,
        )
        # cathode rod length -- WP-N3 row 4.  Only the revision-specific Scholz
        # 2000/2001 24-rod constructor supplies this as a source-supported
        # field.  Akel/Krauz/Scholz-Gribkov constructors keep it blocked until
        # their scopes are explicitly mapped.
        if cathode_rod_length_m is not None and cathode_rod_length_ref is not None:
            fields["cathode_rod_length_m"] = supported(
                "cathode_rod_length_m", cathode_rod_length_m, "m",
                cathode_rod_length_ref,
            )
        else:
            fields["cathode_rod_length_m"] = blocked(
                "cathode_rod_length_m", "m",
                "PF1000-BLK-004-cathode-rod-length-source_available_scholz2000_2001_revision_not_mapped",
            )
        # insulator exposed length is a conflict field (WP-N3 row 13).
        if insulator_exposed_length_status == "conflict":
            fields["insulator_exposed_length_m"] = conflict(
                "insulator_exposed_length_m", "m",
                insulator_exposed_length_conflict,
            )
        else:
            fields["insulator_exposed_length_m"] = supported(
                "insulator_exposed_length_m", insulator_exposed_length_m, "m",
                insulator_exposed_length_ref,
            )
        # insulator outer radius / wall thickness -- WP-N3 rows 14/15.
        # Sprint 6 user-supplied source extraction (2026-05-20) found a PF-1000
        # 2001 24-rod hardware source reporting an alumina-insulator diameter
        # of 229 mm and length of 113 mm
        # [KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:96-98].
        # This removes the old "no KR source" reason for the outer radius, but
        # it does not yet close the active Akel/Krauz/Scholz-Gribkov runtime
        # constructors because the revision mapping has not been added and
        # wall thickness remains absent.  Verdict: blocked with source
        # available, revision-specific mapping required.
        if (
            insulator_outer_radius_m is not None
            and insulator_outer_radius_ref is not None
        ):
            fields["insulator_outer_radius_m"] = supported(
                "insulator_outer_radius_m", insulator_outer_radius_m, "m",
                insulator_outer_radius_ref,
            )
        else:
            fields["insulator_outer_radius_m"] = blocked(
                "insulator_outer_radius_m", "m",
                "PF1000-BLK-015-insulator-outer-radius-source_available_scholz2001_revision_not_mapped",
            )
        fields["insulator_wall_thickness_m"] = blocked(
            "insulator_wall_thickness_m", "m",
            "PF1000-BLK-016-insulator-wall-thickness-no-kr-source",
        )
        # backplate radial extent / axial thickness -- WP-N3 rows 17/18.
        # Sprint 4 KR search (2026-05-20) confirmed no KR file publishes the
        # PF-1000 back plate (OE back plate) radial extent or axial thickness
        # as numeric values.  Krauz 2012 [KR:351-352] mentions "back plate of
        # the OE" in context only (describing insulator shape), not with
        # dimensions.  Verdict: BLOCKED with named missing data.
        fields["backplate_radial_extent_m"] = blocked(
            "backplate_radial_extent_m", "m",
            "PF1000-BLK-017-backplate-radial-extent-no-kr-source",
        )
        fields["backplate_axial_thickness_m"] = blocked(
            "backplate_axial_thickness_m", "m",
            "PF1000-BLK-018-backplate-axial-thickness-no-kr-source",
        )
        # chamber inner radius / length are source-supported (WP-N3 rows 19/20).
        fields["chamber_inner_radius_m"] = supported(
            "chamber_inner_radius_m", chamber_inner_radius_m, "m",
            chamber_inner_radius_ref,
        )
        fields["chamber_length_m"] = supported(
            "chamber_length_m", chamber_length_m, "m", chamber_length_ref
        )
        # chamber wall material / thickness -- WP-N3 rows 21/22. Krasa 2008
        # is now target-extracted for PF-1000 vessel hardware geometry.
        fields["chamber_wall_material"] = supported(
            "chamber_wall_material", 1, "stainless_steel_material_flag",
            f"{krasa}:113-115",
        )
        fields["chamber_wall_thickness_m"] = supported(
            "chamber_wall_thickness_m", 0.010, "m", f"{krasa}:113-115"
        )
        return fields


def _under_resolved(spacing_m: float, feature_m: float, min_cells: float) -> bool:
    """Return True when a sourced feature is not resolved by the grid.

    Under-resolution gate (WP-N3 section-6 item 8): if the grid cell size does
    not place at least `min_cells` cells across the smallest sourced feature,
    the mask build must fail closed rather than emit an accepted mask.
    """
    if feature_m <= 0.0 or spacing_m <= 0.0:
        return True
    return (feature_m / spacing_m) < float(min_cells)


def build_pf1000_material_partition(
    packet: PF1000GeometryPacket,
    *,
    grid: Maxwell3DGrid,
    electron_density_m3: np.ndarray,
    current_density_norm_A_m2: np.ndarray,
    source_interface_z_index: int,
    pml_layers: int,
    electron_density_floor_m3: float,
    min_cells_per_feature: float = 4.0,
) -> dict[str, Any]:
    """Return the 10-class source-tagged PF-1000 material partition.

    Keeps the four Auluck top-level labels (omega_volume_cells,
    terminal_source_interface_faces, wall_material_faces, open_pml_faces) and
    refines `wall_material_faces` into five source-tagged material sub-classes
    (anode / cathode-rods / insulator / chamber-wall / backplate). Adds
    `pml_or_open_boundary_faces` as the open/PML class alias. All masks are
    built deterministically from the static config + grid (no RNG); the same
    config + grid always yields identical hashes.

    The build fails closed (`PF1000GeometryConflict`-style typed error is
    raised by callers; here a `ValueError`) when the grid cannot resolve the
    smallest sourced feature, and emits per-class SHA-256 hashes in a
    `PF1000MaskManifest`. It NEVER promotes first-principles acceptance.
    """
    nx, ny, nz = (int(v) for v in grid.shape)
    dx, dy, dz = (float(s) for s in grid.spacing)
    omega = build_auluck_omega_domain(
        grid_shape=(nx, ny, nz),
        electron_density_m3=electron_density_m3,
        current_density_norm_A_m2=current_density_norm_A_m2,
        source_interface_z_index=source_interface_z_index,
        pml_layers=pml_layers,
        electron_density_floor_m3=electron_density_floor_m3,
    )
    base_masks = omega_domain_label_masks(omega)
    omega_cells = base_masks["omega_volume_cells"]
    interface = base_masks["terminal_source_interface_faces"]
    wall = base_masks["wall_material_faces"]
    open_pml = base_masks["open_pml_faces"]

    # Under-resolution gate over all source-supported features. S3R.4/A5:
    # extended from rods+anode to also cover insulator exposed length and
    # source-tagged transition widths. Conflict and blocked fields contribute no
    # resolvable feature, so only source_supported numeric lengths are checked.
    under_resolution_flags: dict[str, bool] = {}

    rod_diameter = packet.fields["cathode_rod_diameter_m"]
    if rod_diameter.status == "source_supported" and rod_diameter.value is not None:
        under_resolution_flags["cathode_rod_diameter_m"] = _under_resolved(
            min(dx, dy), float(rod_diameter.value), min_cells_per_feature
        )

    anode_radius = packet.fields["anode_radius_m"]
    if anode_radius.status == "source_supported" and anode_radius.value is not None:
        under_resolution_flags["anode_radius_m"] = _under_resolved(
            min(dx, dy), float(anode_radius.value), min_cells_per_feature
        )

    # S3R.4/A5: insulator surface under-resolution gate.  When the insulator
    # exposed length is source_supported (not conflict/blocked), the axial
    # grid spacing must resolve it to at least min_cells_per_feature cells.
    insulator_length = packet.fields.get("insulator_exposed_length_m")
    if (
        insulator_length is not None
        and insulator_length.status == "source_supported"
        and insulator_length.value is not None
    ):
        under_resolution_flags["insulator_exposed_length_m"] = _under_resolved(
            dz, float(insulator_length.value), min_cells_per_feature
        )

    if any(under_resolution_flags.values()):
        flagged = sorted(k for k, v in under_resolution_flags.items() if v)
        raise ValueError(
            "PF-1000 material partition fails closed: grid does not resolve "
            f"sourced feature(s) {flagged} to {min_cells_per_feature} cells; "
            "the build refuses to emit an under-resolved mask"
        )

    # Material sub-classes refine `wall_material_faces`. The static container
    # surfaces are concentric radial shells about the grid axis: anode core
    # (r < a), then the cathode-cage shell, then the chamber wall, with the
    # breech plane (k_port) carrying the backplate/source interface and the
    # insulator sleeving the anode lower part. Build deterministically.
    centre_i = (nx - 1) / 2.0
    centre_j = (ny - 1) / 2.0
    ii = np.arange(nx, dtype=float) - centre_i
    jj = np.arange(ny, dtype=float) - centre_j
    radius = np.sqrt(
        (ii[:, None] * dx) ** 2 + (jj[None, :] * dy) ** 2
    )  # (nx, ny) cell-centre radius
    radius3 = np.broadcast_to(radius[:, :, None], (nx, ny, nz))

    a_field = packet.fields["anode_radius_m"]
    anode_r = float(a_field.value) if a_field.value is not None else 0.0
    domain_r = 0.5 * min(nx * dx, ny * dy)
    # cathode-cage radius is a conflict field; the partition uses the grid
    # half-extent as the chamber-wall onset only -- no conflicting numeric
    # cage radius is silently chosen. The cage shell is the annulus between
    # the anode and the outer 25% of the radial domain.
    cage_inner = anode_r
    cage_outer = max(anode_r, 0.75 * domain_r)

    k_port = int(source_interface_z_index)
    backplate_slab = np.zeros((nx, ny, nz), dtype=bool)
    backplate_slab[:, :, k_port] = True

    anode_core = (radius3 <= anode_r) if anode_r > 0.0 else np.zeros(
        (nx, ny, nz), dtype=bool
    )
    cathode_shell = (radius3 > cage_inner) & (radius3 <= cage_outer)
    chamber_shell = radius3 > cage_outer

    anode_material = wall & anode_core & ~backplate_slab
    cathode_rod = wall & cathode_shell & ~backplate_slab
    chamber_wall = wall & chamber_shell & ~backplate_slab
    backplate = wall & backplate_slab
    # insulator sleeves the anode lower part: the anode-radius shell over the
    # first axial decile, minus the backplate slab. Exposed-length is a
    # conflict field, so the axial extent is a deterministic engineering decile
    # (not a sourced numeric length) -- recorded as candidate, not supported.
    insulator_zmax = max(k_port + 1, min(nz, k_port + 1 + max(1, nz // 10)))
    insulator_band = np.zeros((nx, ny, nz), dtype=bool)
    insulator_band[:, :, k_port + 1:insulator_zmax] = True
    insulator_material = wall & anode_core & insulator_band & ~backplate_slab
    # the insulator band is carved out of the anode-material class so the five
    # sub-classes stay mutually disjoint.
    anode_material = anode_material & ~insulator_material

    material_masks = {
        "anode_material_faces": anode_material,
        "cathode_rod_faces": cathode_rod,
        "insulator_material_faces": insulator_material,
        "chamber_wall_faces": chamber_wall,
        "backplate_source_interface_faces": backplate,
    }
    # The five material sub-classes must partition `wall_material_faces`.
    sub_union = np.zeros((nx, ny, nz), dtype=bool)
    sub_overlap = np.zeros((nx, ny, nz), dtype=bool)
    for mask in material_masks.values():
        sub_overlap |= sub_union & mask
        sub_union |= mask
    material_disjoint = not bool(np.any(sub_overlap))
    material_exhaustive = bool(np.array_equal(sub_union, np.asarray(wall, bool)))

    all_masks: dict[str, np.ndarray] = {
        "omega_volume_cells": np.asarray(omega_cells, bool),
        "terminal_source_interface_faces": np.asarray(interface, bool),
        "wall_material_faces": np.asarray(wall, bool),
        "open_pml_faces": np.asarray(open_pml, bool),
        "pml_or_open_boundary_faces": np.asarray(open_pml, bool),
    }
    all_masks.update({k: np.asarray(v, bool) for k, v in material_masks.items()})

    # S3R.4 (A4): per-class mask status. Each class is either source_supported
    # (every driving dimension is source_supported), candidate_projection_not_source_mask
    # (produced but drives on a heuristic because a dimension is conflict or blocked),
    # or blocked (a required dimension is blocked -- no mask is emitted).
    #
    # Status derivation per class:
    #   omega_volume_cells               -- physics-driven by density/current, source_supported
    #   terminal_source_interface_faces  -- source_supported (z-index from solver)
    #   wall_material_faces              -- source_supported (complement of omega)
    #   open_pml_faces                   -- source_supported (pml_layers is a solver param)
    #   pml_or_open_boundary_faces       -- same as open_pml_faces
    #   anode_material_faces             -- source_supported (anode_radius is sourced)
    #   chamber_wall_faces               -- candidate_projection_not_source_mask:
    #     wall material/thickness are only KR text-parity, not target-extracted,
    #     and the radial split is currently heuristic.
    #   backplate_source_interface_faces -- source_supported (k_port from solver)
    #   cathode_rod_faces                -- candidate_projection_not_source_mask:
    #     cage_outer = 0.75 * domain_r is a heuristic; cathode_cage_radius is conflict.
    #   insulator_material_faces         -- candidate_projection_not_source_mask:
    #     insulator_zmax = k_port + nz//10 is a heuristic decile;
    #     insulator outer radius is blocked (PF1000-BLK-015).
    _cathode_cage_status = packet.fields["cathode_cage_radius_m"].status
    _insulator_len_status = packet.fields["insulator_exposed_length_m"].status
    _insulator_outer_status = packet.fields["insulator_outer_radius_m"].status
    _chamber_wall_material_status = packet.fields["chamber_wall_material"].status
    _chamber_wall_thickness_status = packet.fields["chamber_wall_thickness_m"].status
    # cathode_rod_faces needs cathode_cage_radius to be source_supported AND
    # cathode_rod_diameter to be source_supported to qualify as source_supported.
    _cathode_rod_class_status: str
    if _cathode_cage_status == "source_supported":
        _cathode_rod_class_status = "source_supported"
    else:
        _cathode_rod_class_status = "candidate_projection_not_source_mask"
    # insulator_material_faces needs insulator outer radius AND exposed length
    # to be source_supported; both are currently blocked/conflict.
    _insulator_class_status: str
    if (
        _insulator_outer_status == "source_supported"
        and _insulator_len_status == "source_supported"
    ):
        _insulator_class_status = "source_supported"
    elif _insulator_outer_status == "blocked":
        # insulator outer radius is blocked: cannot produce a source-backed mask
        _insulator_class_status = "candidate_projection_not_source_mask"
    else:
        _insulator_class_status = "candidate_projection_not_source_mask"
    # chamber_wall_faces needs target-extracted wall material/thickness plus a
    # source-supported inner split.  Current values are KR text parity only, so
    # the emitted mask remains a candidate projection.
    _chamber_wall_class_status: str = (
        "source_supported"
        if (
            _cathode_cage_status == "source_supported"
            and _chamber_wall_material_status == "source_supported"
            and _chamber_wall_thickness_status == "source_supported"
        )
        else "candidate_projection_not_source_mask"
    )

    mask_class_status: dict[str, str] = {
        "omega_volume_cells": "source_supported",
        "terminal_source_interface_faces": "source_supported",
        "wall_material_faces": "source_supported",
        "open_pml_faces": "source_supported",
        "pml_or_open_boundary_faces": "source_supported",
        "anode_material_faces": "source_supported",
        "chamber_wall_faces": _chamber_wall_class_status,
        "backplate_source_interface_faces": "source_supported",
        "cathode_rod_faces": _cathode_rod_class_status,
        "insulator_material_faces": _insulator_class_status,
    }

    # SHA-256 is populated only for classes that were actually produced (not blocked).
    # Blocked classes receive an empty string sentinel so the manifest field is
    # always fully populated (manifest validation requires all classes to be present).
    mask_sha256_by_class = {
        name: (
            _mask_sha256(all_masks[name])
            if mask_class_status.get(name, "source_supported") != "blocked"
            else ""
        )
        for name in PF1000_MASK_CLASSES
    }
    mask_cell_counts = {
        name: int(np.count_nonzero(all_masks[name])) for name in PF1000_MASK_CLASSES
    }
    mask_packets = {
        name: {
            "label": name,
            "mask_sha256": mask_sha256_by_class[name],
            "cell_count": mask_cell_counts[name],
            "bounds": _mask_bounds(all_masks[name]),
            "source_refs": list(PF1000_GEOMETRY_SOURCE_REFS),
            "geometry_source_tag": packet.geometry_source_tag,
            # S3R.4: per-packet mask status.
            "mask_class_status": mask_class_status.get(name, "source_supported"),
        }
        for name in PF1000_MASK_CLASSES
    }

    manifest = PF1000MaskManifest(
        geometry_packet_id=packet.geometry_packet_id,
        geometry_source_tag=packet.geometry_source_tag,
        source_refs=packet.source_refs,
        conflict_groups=tuple(sorted(packet.conflicts)),
        blocked_fields=packet.blocked_field_names(),
        grid_shape=(nx, ny, nz),
        grid_spacing_m=(dx, dy, dz),
        mask_sha256_by_class=mask_sha256_by_class,
        mask_cell_counts=mask_cell_counts,
        under_resolution_flags=under_resolution_flags,
        mask_class_status=mask_class_status,
    )

    return {
        "status": "candidate_pf1000_material_partition_not_validation",
        "geometry_source_tag": packet.geometry_source_tag,
        "geometry_packet_id": packet.geometry_packet_id,
        "scope_tag": packet.scope_tag,
        "source_refs": list(PF1000_GEOMETRY_SOURCE_REFS),
        "mask_classes": list(PF1000_MASK_CLASSES),
        "material_subclasses": list(PF1000_MATERIAL_SUBCLASSES),
        "_label_masks": all_masks,
        "auluck_omega_partition": public_omega_domain_packet(omega),
        "mask_packets": mask_packets,
        "manifest": manifest.to_dict(),
        "partition_constraints": {
            "auluck_top_level_mutually_disjoint": (
                omega["partition_constraints"]["mutually_disjoint"]
            ),
            "auluck_top_level_exhaustive": (
                omega["partition_constraints"]["exhaustive"]
            ),
            "terminal_source_interface_disjoint_from_omega": (
                omega["partition_constraints"][
                    "terminal_source_interface_disjoint_from_omega"
                ]
            ),
            "material_subclasses_mutually_disjoint": material_disjoint,
            "material_subclasses_exhaust_wall_material": material_exhaustive,
        },
        "geometry_review_status": packet.geometry_review_status,
        "can_support_power_port_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


# ===========================================================================
# S3.3 -- WP-N3 Sigma_p moving-boundary surface-term data contract.
#
# `SigmaPSurfacePacket` is the data contract `power_port.py` consumes to fail
# closed (or, in a future Sprint 4, compute) Auluck eq. (6) terms II/IV/V/VI.
# S3.3 is PLUMBING ONLY: the packet carries face geometry and per-operand
# status; it does NOT compute the surface integrals. A packet whose operands
# are all present still does not authorise a term value -- the integral is
# Sprint 4 work. Missing operands fail closed with typed blockers.
#
# Authority: docs/external_team_submissions/2026_05_18_three_sprint_blocker_
#   packet/sprint_3/WP_N3_SIGMA_P_RUNTIME_INTERFACE_SPEC.md section 3 schema;
#   sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (5)/(6) p.8.
# ===========================================================================

# WP-N3 Sigma_p spec source refs (interface spec + verified Auluck extract).
SIGMA_P_SURFACE_SOURCE_REFS = (
    "docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/"
    "sprint_3/WP_N3_SIGMA_P_RUNTIME_INTERFACE_SPEC.md:1-439",
    "docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/"
    "sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md",
)

# The four Auluck eq. (6) Sigma_p moving-boundary terms and their operands.
# [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (5)/(6) p.8]
SIGMA_P_TERM_OPERANDS = {
    "term_ii_motional_magnetic_sigma_p_J": ("sigma_p", "v", "B"),
    "term_iv_motional_electric_sigma_p_J": ("sigma_p", "v", "E"),
    "term_v_resistive_sigma_p_J": ("sigma_p", "eta", "J", "B"),
    "term_vi_anomalous_poloidal_sigma_p_J": ("sigma_p", "v", "B"),
}

# Typed blockers for the absent Sigma_p operands. `power_port.py` imports
# these so its fail-closed reasons name the exact missing operand.
SIGMA_P_BLOCKERS = {
    "sigma_p": "sigma_p_face_set_not_available_requires_wp_n3_reviewed_geometry",
    "v": "material_velocity_v_not_available_on_sigma_p_faces",
    "eta": "resistivity_eta_not_available_on_sigma_p_faces",
    "sign_convention": "sigma_p_eq6_sign_convention_not_recorded",
}


@dataclass(frozen=True)
class SigmaPSurfacePacket:
    """WP-N3 Sigma_p moving-boundary surface-term data contract (S3.3).

    Carries the per-face geometry of the moving boundary Sigma_p plus the
    per-operand availability of the fields Auluck eq. (6) terms II/IV/V/VI
    contract. S3.3 is plumbing only: this packet never carries a term value
    and never computes a surface integral. Every absent operand is a typed
    blocker; a present operand only unblocks Sprint 4, never authorises a
    Sprint 3 term value.

    All per-face arrays are 1-D, indexed by a single face index `f`, so a
    face's geometry and every field on it share one index (avoids the
    staggered-grid hazard). A face index of length 0 is legal -- it fails
    closed downstream.

    S3R.5 reviewer-grade digest fields:
      sigma_p_face_set_sha256        -- SHA-256 of the face ID array (or "blocked")
      moving_classification_sha256   -- SHA-256 of the is_moving array (or "blocked")
      omega_partition_sha256         -- SHA-256 of the omega partition (or "blocked")
      material_mask_sha256_by_class  -- per-class SHA-256 from S3.2 manifest
      moving_classification_status   -- "available" | "blocked" | "not_classified"
    """

    status: str
    source_refs: tuple[str, ...]
    source_geometry_packet_id: str | None
    source_geometry_hash: str | None
    n_sigma_p_faces: int
    face_count_total_sigma: int
    geometry_review_status: str
    # per-face geometry (length n_sigma_p_faces; vector arrays are (N, 3)).
    face_ids: np.ndarray | None
    dS_outward_m2: np.ndarray | None
    face_area_m2: np.ndarray | None
    outward_normal: np.ndarray | None
    face_material_class: tuple[str, ...]
    is_moving: np.ndarray | None
    omega_side: str
    excluded_interface_side: str
    outward_normal_convention: str
    # per-operand availability / status.
    field_sampler_status: dict[str, str]  # keys: B, E, J
    velocity_status: str
    resistivity_status: str
    centering: dict[str, str]
    quadrature: str
    sign_convention: dict[str, Any] | None
    operand_blockers: dict[str, str]
    # S3R.5: reviewer-grade digest fields. Empty-string sentinels when blocked.
    sigma_p_face_set_sha256: str = ""
    moving_classification_sha256: str = ""
    omega_partition_sha256: str = ""
    material_mask_sha256_by_class: dict[str, str] = dataclass_field(
        default_factory=dict
    )
    # S3R.5: stationary/moving classification status.
    moving_classification_status: str = "not_classified"
    can_support_power_port_acceptance: bool = False
    can_support_first_principles_acceptance: bool = False

    def __post_init__(self) -> None:
        if self.can_support_power_port_acceptance:
            raise ValueError(
                "SigmaPSurfacePacket must not claim power-port acceptance"
            )
        if self.can_support_first_principles_acceptance:
            raise ValueError(
                "SigmaPSurfacePacket must not claim first-principles acceptance"
            )
        if self.n_sigma_p_faces < 0:
            raise ValueError("n_sigma_p_faces must be non-negative")
        if self.n_sigma_p_faces > self.face_count_total_sigma:
            raise ValueError(
                "n_sigma_p_faces must not exceed face_count_total_sigma"
            )

    def has_sigma_p(self) -> bool:
        """Return True when a non-empty Sigma_p face set is present."""
        return self.face_ids is not None and self.n_sigma_p_faces > 0

    def has_velocity(self) -> bool:
        """Return True when material velocity v is available on Sigma_p faces."""
        return self.velocity_status == "available"

    def has_resistivity(self) -> bool:
        """Return True when resistivity eta is available on Sigma_p faces."""
        return self.resistivity_status == "available"

    def has_sign_convention(self) -> bool:
        """Return True when the eq. (6) term-sign record is present."""
        return isinstance(self.sign_convention, dict) and bool(
            self.sign_convention.get("eq6_term_signs")
        )

    def operand_status(self, operand: str) -> bool:
        """Return True when one named operand (sigma_p/v/eta) is available."""
        if operand == "sigma_p":
            return self.has_sigma_p()
        if operand == "v":
            return self.has_velocity()
        if operand == "eta":
            return self.has_resistivity()
        if operand in {"B", "E", "J"}:
            return self.field_sampler_status.get(operand) == "available"
        raise ValueError(f"unknown Sigma_p operand {operand!r}")

    def to_dict(self) -> dict[str, Any]:
        def _arr(value: np.ndarray | None) -> list[Any] | None:
            return None if value is None else np.asarray(value).tolist()

        return {
            "status": self.status,
            "source_refs": list(self.source_refs),
            "source_geometry_packet_id": self.source_geometry_packet_id,
            "source_geometry_hash": self.source_geometry_hash,
            "n_sigma_p_faces": self.n_sigma_p_faces,
            "face_count_total_sigma": self.face_count_total_sigma,
            "geometry_review_status": self.geometry_review_status,
            "face_ids": _arr(self.face_ids),
            "dS_outward_m2": _arr(self.dS_outward_m2),
            "face_area_m2": _arr(self.face_area_m2),
            "outward_normal": _arr(self.outward_normal),
            "face_material_class": list(self.face_material_class),
            "is_moving": _arr(self.is_moving),
            "omega_side": self.omega_side,
            "excluded_interface_side": self.excluded_interface_side,
            "outward_normal_convention": self.outward_normal_convention,
            "field_sampler_status": dict(self.field_sampler_status),
            "velocity_status": self.velocity_status,
            "resistivity_status": self.resistivity_status,
            "centering": dict(self.centering),
            "quadrature": self.quadrature,
            "sign_convention": (
                None if self.sign_convention is None
                else dict(self.sign_convention)
            ),
            "operand_blockers": dict(self.operand_blockers),
            # S3R.5 reviewer-grade digest fields.
            "sigma_p_face_set_sha256": self.sigma_p_face_set_sha256,
            "moving_classification_sha256": self.moving_classification_sha256,
            "omega_partition_sha256": self.omega_partition_sha256,
            "material_mask_sha256_by_class": dict(self.material_mask_sha256_by_class),
            "moving_classification_status": self.moving_classification_status,
            "can_support_power_port_acceptance": (
                self.can_support_power_port_acceptance
            ),
            "can_support_first_principles_acceptance": (
                self.can_support_first_principles_acceptance
            ),
        }

    @classmethod
    def blocked(
        cls,
        *,
        reason: str = "sigma_p_face_set_not_available_requires_wp_n3_reviewed_geometry",
        source_geometry_packet_id: str | None = None,
    ) -> SigmaPSurfacePacket:
        """Return a fully fail-closed Sigma_p packet (no face set, no operands).

        This is the honest Sprint 3 default until a reviewed WP-N3 PF-1000
        geometry supplies a moving-boundary face set. Every operand carries a
        typed blocker; `power_port.py` consumes this and fails terms
        II/IV/V/VI closed.
        """
        return cls(
            status="blocked_sigma_p_surface_packet_not_available",
            source_refs=SIGMA_P_SURFACE_SOURCE_REFS,
            source_geometry_packet_id=source_geometry_packet_id,
            source_geometry_hash=None,
            n_sigma_p_faces=0,
            face_count_total_sigma=0,
            geometry_review_status="geometry_candidate_not_reviewed",
            face_ids=None,
            dS_outward_m2=None,
            face_area_m2=None,
            outward_normal=None,
            face_material_class=(),
            is_moving=None,
            omega_side="omega_interior",
            excluded_interface_side="terminal_source_interface_excluded",
            outward_normal_convention="outward_from_omega",
            field_sampler_status={"B": "blocked", "E": "blocked", "J": "blocked"},
            velocity_status="blocked",
            resistivity_status="blocked",
            centering={
                "b_sampling": "not_available",
                "e_sampling": "not_available",
                "j_sampling": "not_available",
                "v_sampling": "not_available",
                "eta_sampling": "not_available",
                "time_centering": "candidate_step_consistent_not_accepted",
                "quadrature": "not_available",
            },
            quadrature="not_available",
            sign_convention=None,
            operand_blockers={
                "sigma_p": reason,
                "v": SIGMA_P_BLOCKERS["v"],
                "eta": SIGMA_P_BLOCKERS["eta"],
                "sign_convention": SIGMA_P_BLOCKERS["sign_convention"],
            },
        )


def build_sigma_p_surface_packet(
    material_partition: dict[str, Any] | None,
    *,
    sigma_p_runtime: Mapping[str, Any] | None = None,
) -> SigmaPSurfacePacket:
    """Return the WP-N3 Sigma_p surface packet for the current runtime state.

    S3.3 PLUMBING ONLY. This builder wires the data contract; it never
    computes a surface integral. With no `sigma_p_runtime` (the Sprint 3
    reality -- the runtime exposes no reviewed moving-boundary face set), it
    returns `SigmaPSurfacePacket.blocked()`: every operand is a typed blocker
    and `power_port.py` fails terms II/IV/V/VI closed.

    `material_partition` is the S3.2 `build_pf1000_material_partition` result;
    it supplies the geometry packet ID and a deterministic geometry hash so a
    downstream reviewer can confirm which geometry the (future) Sigma_p face
    set was derived from. The Sigma_p face set itself is NOT derived here --
    deriving the moving boundary from reviewed material masks is Sprint 4.
    """
    packet_id: str | None = None
    geometry_hash: str | None = None
    if isinstance(material_partition, dict):
        packet_id = material_partition.get("geometry_packet_id")
        hashes = material_partition.get("manifest", {})
        if isinstance(hashes, Mapping):
            by_class = hashes.get("mask_sha256_by_class")
            if isinstance(by_class, Mapping):
                # a deterministic digest over the ordered per-class hashes.
                hasher = hashlib.sha256()
                for name in PF1000_MASK_CLASSES:
                    hasher.update(str(by_class.get(name, "")).encode("utf-8"))
                geometry_hash = hasher.hexdigest()
    # S3R.5 (A6): the blocked return path MUST preserve the geometry hash so a
    # reviewer can confirm which S3.2 geometry drove the blocked packet.
    # Both branches (no runtime and sprint4-placeholder) now carry the hash.
    if sigma_p_runtime is None:
        blocked = SigmaPSurfacePacket.blocked(
            source_geometry_packet_id=packet_id,
        )
        return replace(
            blocked,
            source_geometry_hash=geometry_hash,
        )
    # A non-None sigma_p_runtime is reserved for the Sprint 4 face-set
    # producer. S3.3 does not synthesise a face set, so any caller that
    # supplies one without the Sprint 4 producer still fails closed: the
    # plumbing refuses to fabricate Sigma_p geometry.
    blocked = SigmaPSurfacePacket.blocked(source_geometry_packet_id=packet_id)
    return replace(
        blocked,
        source_geometry_hash=geometry_hash,
        status="blocked_sigma_p_surface_packet_face_set_is_sprint4_work",
    )
