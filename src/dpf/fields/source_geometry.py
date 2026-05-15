"""Source-scoped geometry packet for the candidate 3-D hybrid PIC path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid


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
