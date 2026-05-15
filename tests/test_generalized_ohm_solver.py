import numpy as np
import pytest

from dpf.fields.maxwell_3d import EPSILON_0, Maxwell3DGrid
from dpf.fields.ohm_solver import (
    GeneralizedOhmSolver,
    generalized_ohm_candidate_evidence,
)
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(3, 4, 5), spacing=(1.0e-3, 2.0e-3, 3.0e-3))


def _vector(grid: Maxwell3DGrid, values: tuple[float, float, float]) -> np.ndarray:
    out = np.zeros(grid.shape + (3,), dtype=float)
    out[..., 0] = values[0]
    out[..., 1] = values[1]
    out[..., 2] = values[2]
    return out


def test_hall_disabled_ohm_solver_reduces_to_A_over_D() -> None:
    grid = _grid()
    solver = GeneralizedOhmSolver(grid)
    E = _vector(grid, (2.0, -1.0, 0.5))
    zero = np.zeros_like(E)
    sigma = 50.0
    dt = 2.0e-13
    ne = np.full(grid.shape, 1.0e21)

    J, telemetry = solver.solve_current(
        electric_field_V_m=E,
        magnetic_field_T=zero,
        curl_B_T_m=zero,
        ion_current_A_m2=zero,
        conductivity_S_m=sigma,
        electron_density_m3=ne,
        dt_s=dt,
        include_hall=False,
    )

    D = 1.0 + sigma * dt / (2.0 * EPSILON_0)
    np.testing.assert_allclose(J, sigma * E / D)
    assert telemetry.include_hall is False
    assert telemetry.max_algebraic_residual_A_m2 < 1.0e-10
    assert telemetry.can_support_first_principles_acceptance is False


def test_hall_enabled_ohm_solver_satisfies_vector_algebraic_residual() -> None:
    grid = _grid()
    solver = GeneralizedOhmSolver(grid)
    E = _vector(grid, (3.0, 0.5, -0.25))
    B = _vector(grid, (0.0, 0.0, 0.8))
    curl_B = _vector(grid, (2.0e-2, -1.0e-2, 0.0))
    Ji = _vector(grid, (1.0e3, -2.0e3, 0.0))
    sigma = np.full(grid.shape, 20.0)
    ne = np.full(grid.shape, 2.0e21)

    J, telemetry = solver.solve_current(
        electric_field_V_m=E,
        magnetic_field_T=B,
        curl_B_T_m=curl_B,
        ion_current_A_m2=Ji,
        conductivity_S_m=sigma,
        electron_density_m3=ne,
        dt_s=1.0e-13,
        include_hall=True,
    )

    assert J.shape == grid.shape + (3,)
    assert telemetry.include_hall is True
    assert telemetry.max_current_A_m2 > 0.0
    assert telemetry.max_algebraic_residual_A_m2 < 1.0e-8


def test_pressure_gradient_term_can_be_density_thresholded() -> None:
    grid = _grid()
    solver = GeneralizedOhmSolver(grid)
    x = np.arange(grid.nx, dtype=float)[:, np.newaxis, np.newaxis]
    pe = np.broadcast_to(2.0e-3 * x, grid.shape)
    ne = np.full(grid.shape, 2.0e23)
    ne[0, :, :] = 5.0e22

    pressure, telemetry = solver.pressure_gradient_term(
        pe,
        ne,
        density_threshold_m3=1.0e23,
    )

    assert telemetry["status"] == "candidate_pressure_gradient_term"
    assert 0.0 < telemetry["active_fraction"] < 1.0
    assert np.count_nonzero(pressure[0]) == 0
    assert np.max(np.abs(pressure[1:, ..., 0])) > 0.0


def test_ohm_solver_rejects_nonpositive_electron_density() -> None:
    grid = _grid()
    solver = GeneralizedOhmSolver(grid)
    zero = np.zeros(grid.shape + (3,), dtype=float)
    ne = np.ones(grid.shape)
    ne[0, 0, 0] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        solver.solve_current(
            electric_field_V_m=zero,
            magnetic_field_T=zero,
            curl_B_T_m=zero,
            ion_current_A_m2=zero,
            conductivity_S_m=1.0,
            electron_density_m3=ne,
            dt_s=1.0e-13,
        )


def test_generalized_ohm_candidate_evidence_does_not_satisfy_hybrid_gate() -> None:
    grid = _grid()
    solver = GeneralizedOhmSolver(grid)
    zero = np.zeros(grid.shape + (3,), dtype=float)
    E = _vector(grid, (1.0, 0.0, 0.0))
    _, telemetry = solver.solve_current(
        electric_field_V_m=E,
        magnetic_field_T=zero,
        curl_B_T_m=zero,
        ion_current_A_m2=zero,
        conductivity_S_m=1.0,
        electron_density_m3=np.full(grid.shape, 1.0e20),
        dt_s=1.0e-13,
    )
    candidate = generalized_ohm_candidate_evidence(telemetry)

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "electron_fluid_generalized_ohm_solver": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "electron_fluid_generalized_ohm_solver" in status["missing_capabilities"]
