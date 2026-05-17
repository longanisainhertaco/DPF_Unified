import numpy as np
import pytest

from dpf.fields.conductivity import (
    PlasmaVacuumConductivityBlend,
    conductivity_blend_candidate_evidence,
    partial_ionized_conductivity,
)
from dpf.fields.maxwell_3d import EPSILON_0, Maxwell3DGrid
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(3, 1 + 1, 1 + 1), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def test_conductivity_blend_matches_source_piecewise_density_transition() -> None:
    grid = _grid()
    blend = PlasmaVacuumConductivityBlend(grid)
    n0 = 1.0e23
    ne = np.array([0.05, 0.5, 2.0])[:, np.newaxis, np.newaxis] * n0
    ne = np.broadcast_to(ne, grid.shape)

    sigma, telemetry = blend.effective_conductivity(
        sigma0_S_m=100.0,
        electron_density_m3=ne,
        background_density_m3=n0,
        dt_s=1.0e-15,
        ohmic_cfl_safety=1.0,
    )

    assert sigma[0, 0, 0] == 0.0
    assert sigma[1, 0, 0] == pytest.approx(0.5**3 * 100.0)
    assert sigma[2, 0, 0] == pytest.approx(100.0)
    assert telemetry.vacuum_fraction == pytest.approx(1.0 / 3.0)
    assert telemetry.transition_fraction == pytest.approx(1.0 / 3.0)
    assert telemetry.plasma_fraction == pytest.approx(1.0 / 3.0)
    assert telemetry.cfl_limited_fraction == 0.0


def test_conductivity_blend_applies_ohmic_cfl_cap() -> None:
    grid = _grid()
    blend = PlasmaVacuumConductivityBlend(grid)
    n0 = 1.0e23
    ne = np.full(grid.shape, 2.0 * n0)
    dt = 1.0e-12
    safety = 0.5

    sigma, telemetry = blend.effective_conductivity(
        sigma0_S_m=1.0e6,
        electron_density_m3=ne,
        background_density_m3=n0,
        dt_s=dt,
        ohmic_cfl_safety=safety,
    )

    expected_cap = safety * EPSILON_0 / dt
    assert np.all(sigma == pytest.approx(expected_cap))
    assert telemetry.sigma_cfl_S_m == pytest.approx(expected_cap)
    assert telemetry.cfl_limited_fraction == 1.0
    assert telemetry.ohmic_cfl_limit_applied is True


def test_conductivity_blend_can_report_unclamped_midpoint_ohm_path() -> None:
    grid = _grid()
    blend = PlasmaVacuumConductivityBlend(grid)
    n0 = 1.0e23
    ne = np.full(grid.shape, 2.0 * n0)

    sigma, telemetry = blend.effective_conductivity(
        sigma0_S_m=1.0e6,
        electron_density_m3=ne,
        background_density_m3=n0,
        dt_s=1.0e-12,
        ohmic_cfl_safety=0.5,
        apply_ohmic_cfl_limit=False,
    )

    assert np.all(sigma == pytest.approx(1.0e6))
    assert telemetry.cfl_limited_fraction == 1.0
    assert telemetry.ohmic_cfl_limit_applied is False


def test_conductivity_blend_rejects_invalid_inputs() -> None:
    grid = _grid()
    blend = PlasmaVacuumConductivityBlend(grid)
    ne = np.ones(grid.shape)

    with pytest.raises(ValueError, match="dt_s must be positive"):
        blend.effective_conductivity(
            sigma0_S_m=1.0,
            electron_density_m3=ne,
            background_density_m3=1.0,
            dt_s=0.0,
            ohmic_cfl_safety=1.0,
        )


def test_conductivity_blend_candidate_evidence_does_not_satisfy_hybrid_gate() -> None:
    grid = _grid()
    blend = PlasmaVacuumConductivityBlend(grid)
    sigma, telemetry = blend.effective_conductivity(
        sigma0_S_m=1.0,
        electron_density_m3=np.ones(grid.shape),
        background_density_m3=1.0,
        dt_s=1.0e-12,
        ohmic_cfl_safety=1.0,
    )
    candidate = conductivity_blend_candidate_evidence(telemetry)

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "plasma_vacuum_conductivity_blending": candidate,
        },
    })

    assert np.all(np.isfinite(sigma))
    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "plasma_vacuum_conductivity_blending" in status["missing_capabilities"]


def test_partial_ionized_conductivity_includes_neutral_drag() -> None:
    grid = _grid()
    ne = np.full(grid.shape, 1.0e20)
    Te = np.full(grid.shape, 2.0e5)
    sigma_ionized, telemetry_ionized = partial_ionized_conductivity(
        electron_density_m3=ne,
        neutral_density_m3=np.zeros(grid.shape),
        electron_temperature_K=Te,
    )
    sigma_neutral, telemetry_neutral = partial_ionized_conductivity(
        electron_density_m3=ne,
        neutral_density_m3=np.full(grid.shape, 1.0e22),
        electron_temperature_K=Te,
    )

    assert telemetry_neutral.status == (
        "candidate_source_backed_partial_ionized_conductivity"
    )
    assert telemetry_neutral.can_support_first_principles_acceptance is False
    assert np.max(sigma_ionized) == telemetry_ionized.max_sigma_S_m
    assert np.max(sigma_neutral) < np.max(sigma_ionized)
    assert telemetry_neutral.max_electron_neutral_resistivity_ohm_m > 0.0
