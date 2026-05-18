"""Source-scoped geometry packet for the candidate 3-D hybrid PIC path."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
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
    """Typed LLNL-like setup values extracted from the local hybrid PIC source."""

    source: str = HYBRID_PIC_3D_SOURCE
    source_lines: str = "632-740"
    source_scope: str = "llnl_like_180ka_axisymmetric_hybrid_pic"
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
    """Build non-promoting evidence for source geometry extraction."""
    return {
        "passed": True,
        "status": "candidate",
        "capability": "same_scope_3d_validation_packet",
        "source": geometry.source,
        "source_lines": geometry.source_lines,
        "implementation": "src/dpf/fields/source_geometry.py",
        "evidence_type": "engineering_source_geometry_packet",
        "source_scope": geometry.source_scope,
        "coordinate_system": geometry.coordinate_system,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "The extracted source setup is 2-D axisymmetric, not an accepted true-3D geometry packet.",
            "The smoke grid is a Cartesian engineering projection for code exercise only.",
            "No same-scope PF-1000/Akel or LLNL-like experimental validation packet is attached.",
        ],
    }
