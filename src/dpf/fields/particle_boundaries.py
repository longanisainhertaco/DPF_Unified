"""Particle absorption for the 3-D hybrid PIC-fluid boundary contract.

The local hybrid PIC-fluid source states that particles entering conductor or
PML regions are absorbed and deleted.  This module provides that behavior as an
isolated, auditable engineering component for the candidate 3-D loop.  It does
not validate the DPF geometry or promote boundary evidence to first-principles
acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid


@dataclass(frozen=True)
class ParticleBoundaryTelemetry:
    """Particle absorption counts for one boundary application."""

    status: str
    source: str
    n_particles_before: int
    n_particles_after: int
    deleted_total: int
    deleted_conductor: int
    deleted_pml: int
    deleted_outside_domain: int
    pml_cells: int
    conductor_cells_active: int
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParticleAbsorbingBoundaries:
    """Delete PIC particles that enter conductor cells or PML cells."""

    capability_id = "pml_conductor_particle_boundaries"

    def __init__(
        self,
        grid: Maxwell3DGrid,
        *,
        conductor_cells: np.ndarray | None = None,
        pml_cells: int = 0,
    ) -> None:
        self.grid = grid
        if int(pml_cells) != pml_cells or pml_cells < 0:
            raise ValueError("pml_cells must be a non-negative integer")
        self.pml_cells = int(pml_cells)
        self.conductor_cells = None
        if conductor_cells is not None:
            mask = np.asarray(conductor_cells, dtype=bool)
            if mask.shape != grid.shape:
                raise ValueError("conductor_cells must match grid shape")
            self.conductor_cells = mask

    def apply(self, pic: HybridPIC) -> ParticleBoundaryTelemetry:
        """Delete particles in conductor/PML/outside regions from ``pic``."""
        if tuple(pic.grid_shape) != self.grid.shape:
            raise ValueError("PIC grid shape does not match Maxwell grid")
        if (float(pic.dx), float(pic.dy), float(pic.dz)) != self.grid.spacing:
            raise ValueError("PIC grid spacing does not match Maxwell grid")

        n_before = _particle_count(pic)
        deleted_conductor = 0
        deleted_pml = 0
        deleted_outside = 0
        for species in pic.species:
            n_species = species.n_particles()
            if n_species == 0:
                continue
            flags = self.classify_positions(species.positions)
            delete = flags.conductor | flags.pml | flags.outside_domain
            keep = ~delete
            deleted_conductor += int(np.count_nonzero(flags.conductor))
            deleted_pml += int(np.count_nonzero(flags.pml & ~flags.conductor))
            deleted_outside += int(
                np.count_nonzero(flags.outside_domain & ~flags.conductor & ~flags.pml)
            )
            if np.all(keep):
                continue
            species.positions = species.positions[keep]
            species.velocities = species.velocities[keep]
            species.weights = species.weights[keep]
            if species.positions_old.shape[0] == n_species:
                species.positions_old = species.positions_old[keep]
            else:
                species.positions_old = species.positions.copy()

        n_after = _particle_count(pic)
        return ParticleBoundaryTelemetry(
            status="candidate_engineering_particle_absorption",
            source=HYBRID_PIC_3D_SOURCE,
            n_particles_before=n_before,
            n_particles_after=n_after,
            deleted_total=n_before - n_after,
            deleted_conductor=deleted_conductor,
            deleted_pml=deleted_pml,
            deleted_outside_domain=deleted_outside,
            pml_cells=self.pml_cells,
            conductor_cells_active=(
                0
                if self.conductor_cells is None
                else int(np.count_nonzero(self.conductor_cells))
            ),
        )

    def classify_positions(self, positions: np.ndarray) -> _BoundaryFlags:
        """Classify particle positions against conductor, PML, and domain."""
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")

        nx, ny, nz = self.grid.shape
        dx, dy, dz = self.grid.spacing
        outside = (
            (pos[:, 0] < 0.0)
            | (pos[:, 1] < 0.0)
            | (pos[:, 2] < 0.0)
            | (pos[:, 0] >= nx * dx)
            | (pos[:, 1] >= ny * dy)
            | (pos[:, 2] >= nz * dz)
        )

        cell_i = np.floor(pos[:, 0] / dx).astype(int, copy=False)
        cell_j = np.floor(pos[:, 1] / dy).astype(int, copy=False)
        cell_k = np.floor(pos[:, 2] / dz).astype(int, copy=False)
        cell_i = np.clip(cell_i, 0, nx - 1)
        cell_j = np.clip(cell_j, 0, ny - 1)
        cell_k = np.clip(cell_k, 0, nz - 1)

        conductor = np.zeros(pos.shape[0], dtype=bool)
        if self.conductor_cells is not None:
            conductor = self.conductor_cells[cell_i, cell_j, cell_k] & ~outside

        pml = np.zeros(pos.shape[0], dtype=bool)
        if self.pml_cells > 0:
            p = self.pml_cells
            pml = (
                (cell_i < p)
                | (cell_i >= nx - p)
                | (cell_j < p)
                | (cell_j >= ny - p)
                | (cell_k < p)
                | (cell_k >= nz - p)
            ) & ~outside

        return _BoundaryFlags(
            conductor=conductor,
            pml=pml,
            outside_domain=outside,
        )


@dataclass(frozen=True)
class _BoundaryFlags:
    conductor: np.ndarray
    pml: np.ndarray
    outside_domain: np.ndarray


def particle_boundary_candidate_evidence(
    telemetry: ParticleBoundaryTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for candidate particle-boundary behavior."""
    return {
        "passed": telemetry.status == "candidate_engineering_particle_absorption",
        "status": "candidate",
        "capability": "pml_conductor_particle_boundaries",
        "source": telemetry.source,
        "source_lines": "613-619, 625-628",
        "implementation": "src/dpf/fields/particle_boundaries.py",
        "evidence_type": "engineering_particle_absorption_step",
        "deleted_total": telemetry.deleted_total,
        "deleted_conductor": telemetry.deleted_conductor,
        "deleted_pml": telemetry.deleted_pml,
        "deleted_outside_domain": telemetry.deleted_outside_domain,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Conductor and PML masks are supplied by engineering setup, not a reviewed DPF geometry packet.",
            "Existing HybridPIC push still applies reflecting edge handling before this deletion hook.",
            "No same-scope boundary-validation packet is attached.",
        ],
    }


def _particle_count(pic: HybridPIC) -> int:
    return int(sum(species.n_particles() for species in pic.species))
