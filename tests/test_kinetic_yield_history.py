import numpy as np

from dpf.constants import e as ELEMENTARY_CHARGE
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.kinetic_yield import (
    KineticIonYieldHistory,
    kinetic_neutron_yield_authority_status,
    kinetic_yield_candidate_evidence,
)
from dpf.fields.maxwell_3d import Maxwell3DGrid
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

M_D = 3.344e-27


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def _pic(grid: Maxwell3DGrid) -> HybridPIC:
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
        positions=np.array([
            [1.5e-3, 1.5e-3, 1.5e-3],
            [2.0e-3, 1.5e-3, 1.5e-3],
        ]),
        velocities=np.array([
            [1.0e6, 0.0, 0.0],
            [-1.0e6, 0.0, 0.0],
        ]),
        weights=np.full(2, 1.0e8),
    )
    return pic


def test_kinetic_yield_history_accumulates_particle_distribution_rate() -> None:
    grid = _grid()
    history = KineticIonYieldHistory(grid)
    target = np.full(grid.shape, 1.0e24)

    first = history.step(_pic(grid), target_density_m3=target, dt_s=1.0e-9)
    second = history.step(_pic(grid), target_density_m3=target, dt_s=1.0e-9)

    assert first.status == "candidate_engineering_kinetic_yield_history"
    assert first.neutron_rate_per_s > 0.0
    assert first.neutron_increment > 0.0
    assert first.included_species == ("d",)
    assert first.mechanism_channels == ("dd_particle_distribution_total",)
    assert first.mechanism_separation_status == "not_mechanism_separated"
    assert second.cumulative_neutrons > first.cumulative_neutrons
    assert second.time_s == 2.0e-9


def test_kinetic_yield_candidate_evidence_does_not_satisfy_gate() -> None:
    grid = _grid()
    telemetry = KineticIonYieldHistory(grid).step(
        _pic(grid),
        target_density_m3=np.full(grid.shape, 1.0e24),
        dt_s=1.0e-9,
    )

    candidate = kinetic_yield_candidate_evidence(telemetry)
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "kinetic_ion_neutron_yield_history": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "kinetic_ion_neutron_yield_history" in status["missing_capabilities"]


def test_kinetic_yield_authority_blocks_scalar_cumulative_yield() -> None:
    grid = _grid()
    telemetry = KineticIonYieldHistory(grid).step(
        _pic(grid),
        target_density_m3=np.full(grid.shape, 1.0e24),
        dt_s=1.0e-9,
    )
    candidate = kinetic_yield_candidate_evidence(telemetry)

    status = kinetic_neutron_yield_authority_status(
        kinetic_yield_evidence=candidate,
        temperature_authority={
            "status": "candidate_separate_te_still_blocked",
            "can_support_pressure_hall_quantitative_claims": False,
        },
    )

    assert status["status"] == "blocked"
    assert status["can_support_total_yield_acceptance"] is False
    assert "accepted_kinetic_yield_history" in status["missing_evidence"]
    assert "mechanism_separated_yield_channels" in status["missing_evidence"]
    assert "same_scope_detector_response" in status["missing_evidence"]
    assert "yield_uncertainty_budget" in status["missing_evidence"]
    assert "electron_temperature_authority" in status["missing_evidence"]
