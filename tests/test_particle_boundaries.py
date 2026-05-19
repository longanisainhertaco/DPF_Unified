import numpy as np

from dpf.constants import e as ELEMENTARY_CHARGE  # noqa: N812
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.maxwell_3d import Maxwell3DGrid
from dpf.fields.particle_boundaries import (
    ParticleAbsorbingBoundaries,
    particle_boundary_candidate_evidence,
)
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

M_D = 3.344e-27


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def _pic(grid: Maxwell3DGrid, positions: np.ndarray) -> HybridPIC:
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=1.0e-12,
        use_esirkepov=False,
        use_binary_collisions=False,
    )
    pic.add_species(
        "d",
        M_D,
        ELEMENTARY_CHARGE,
        positions=positions,
        velocities=np.zeros_like(positions),
        weights=np.full(positions.shape[0], 1.0e8),
    )
    return pic


def test_particle_absorbing_boundaries_delete_conductor_particles() -> None:
    grid = _grid()
    conductor = np.zeros(grid.shape, dtype=bool)
    conductor[2, 2, 2] = True
    pic = _pic(
        grid,
        np.array([
            [2.2e-3, 2.2e-3, 2.2e-3],
            [1.5e-3, 1.5e-3, 1.5e-3],
        ]),
    )
    boundaries = ParticleAbsorbingBoundaries(grid, conductor_cells=conductor)

    telemetry = boundaries.apply(pic)

    assert telemetry.status == "candidate_engineering_particle_absorption"
    assert telemetry.n_particles_before == 2
    assert telemetry.n_particles_after == 1
    assert telemetry.deleted_conductor == 1
    assert telemetry.deleted_total == 1
    assert np.allclose(pic.species[0].positions, [[1.5e-3, 1.5e-3, 1.5e-3]])


def test_particle_absorbing_boundaries_delete_pml_and_outside_particles() -> None:
    grid = _grid()
    pic = _pic(
        grid,
        np.array([
            [0.2e-3, 1.5e-3, 1.5e-3],
            [4.2e-3, 1.5e-3, 1.5e-3],
            [1.5e-3, 1.5e-3, 1.5e-3],
        ]),
    )
    boundaries = ParticleAbsorbingBoundaries(grid, pml_cells=1)

    telemetry = boundaries.apply(pic)

    assert telemetry.n_particles_before == 3
    assert telemetry.n_particles_after == 1
    assert telemetry.deleted_pml == 1
    assert telemetry.deleted_outside_domain == 1
    assert telemetry.deleted_total == 2
    assert np.allclose(pic.species[0].positions, [[1.5e-3, 1.5e-3, 1.5e-3]])


def test_particle_boundary_candidate_evidence_does_not_satisfy_gate() -> None:
    grid = _grid()
    telemetry = ParticleAbsorbingBoundaries(grid, pml_cells=1).apply(
        _pic(grid, np.array([[0.2e-3, 1.5e-3, 1.5e-3]]))
    )

    candidate = particle_boundary_candidate_evidence(telemetry)
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "pml_conductor_particle_boundaries": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "pml_conductor_particle_boundaries" in status["missing_capabilities"]
