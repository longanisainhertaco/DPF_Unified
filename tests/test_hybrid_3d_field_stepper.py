import numpy as np

from dpf.fields.hybrid_stepper import (
    HybridPIC3DFieldStepper,
    hybrid_stepper_candidate_evidence,
)
from dpf.fields.maxwell_3d import Maxwell3DGrid


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def test_hybrid_stepper_zero_fields_and_zero_conductivity_remain_quiet() -> None:
    grid = _grid()
    stepper = HybridPIC3DFieldStepper(grid)
    state = stepper.maxwell.empty_state()
    zero_current = np.zeros(grid.shape + (3,), dtype=float)

    result = stepper.step(
        state,
        dt_s=1.0e-15,
        ion_current_A_m2=zero_current,
        electron_density_m3=np.full(grid.shape, 1.0e23),
        sigma0_S_m=0.0,
        background_density_m3=1.0e23,
        ohmic_cfl_safety=1.0,
        include_hall=False,
    )

    assert result.telemetry.status == "candidate_engineering_field_current_step"
    assert result.telemetry.diagnostics_after["total_energy_J"] == 0.0
    assert np.count_nonzero(result.total_current_A_m2) == 0


def test_hybrid_stepper_ohmic_current_reduces_uniform_electric_energy() -> None:
    grid = _grid()
    stepper = HybridPIC3DFieldStepper(grid)
    state = stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(10.0)
    zero_current = np.zeros(grid.shape + (3,), dtype=float)
    before = stepper.maxwell.electric_energy_J(state)

    result = stepper.step(
        state,
        dt_s=1.0e-15,
        ion_current_A_m2=zero_current,
        electron_density_m3=np.full(grid.shape, 1.0e23),
        sigma0_S_m=1.0,
        background_density_m3=1.0e23,
        ohmic_cfl_safety=1.0,
        include_hall=False,
    )
    after = stepper.maxwell.electric_energy_J(result.state)

    assert after < before
    assert result.telemetry.ohm_solver["max_current_A_m2"] > 0.0
    assert result.telemetry.current_port["deposition_method"] == (
        "generalized_ohm_total_current"
    )


def test_hybrid_stepper_can_run_candidate_predictor_corrector() -> None:
    grid = _grid()
    stepper = HybridPIC3DFieldStepper(grid)
    state = stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(25.0)
    zero_current = np.zeros(grid.shape + (3,), dtype=float)

    first = stepper.step(
        state,
        dt_s=1.0e-15,
        ion_current_A_m2=zero_current,
        electron_density_m3=np.full(grid.shape, 1.0e23),
        sigma0_S_m=1.0,
        background_density_m3=1.0e23,
        ohmic_cfl_safety=1.0,
        include_hall=False,
        use_predictor_corrector=True,
    )
    second = stepper.step(
        first.state,
        dt_s=1.0e-15,
        ion_current_A_m2=zero_current,
        electron_density_m3=np.full(grid.shape, 1.0e23),
        sigma0_S_m=1.0,
        background_density_m3=1.0e23,
        ohmic_cfl_safety=1.0,
        include_hall=False,
        use_predictor_corrector=True,
    )

    assert first.telemetry.predictor_corrector is not None
    assert first.telemetry.predictor_corrector["first_step_initialization"] is True
    assert second.telemetry.predictor_corrector is not None
    assert second.telemetry.predictor_corrector["first_step_initialization"] is False
    assert first.end_step_current_A_m2.shape == grid.shape + (3,)
    assert first.telemetry.predictor_corrector["corrected_max_residual_A_m2"] < 1.0e-8


def test_hybrid_stepper_can_apply_candidate_marder_correction() -> None:
    grid = Maxwell3DGrid(shape=(16, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))
    stepper = HybridPIC3DFieldStepper(grid)
    state = stepper.maxwell.empty_state()
    x = np.arange(grid.nx, dtype=float) * grid.dx
    state.E.Ex_edge[:, :, :] = np.sin(
        2.0 * np.pi * x[:, np.newaxis, np.newaxis] / (grid.nx * grid.dx)
    )

    result = stepper.step(
        state,
        dt_s=1.0e-15,
        ion_current_A_m2=np.zeros(grid.shape + (3,), dtype=float),
        electron_density_m3=np.full(grid.shape, 1.0e23),
        sigma0_S_m=0.0,
        background_density_m3=1.0e23,
        ohmic_cfl_safety=1.0,
        include_hall=False,
        marder_factor_m2=0.02 * grid.dx**2,
    )

    assert result.telemetry.marder is not None
    assert result.telemetry.marder["status"] == "candidate_engineering_marder_correction"
    assert result.telemetry.marder["residual_after_linf"] < (
        result.telemetry.marder["residual_before_linf"]
    )


def test_hybrid_stepper_candidate_evidence_is_non_promoting() -> None:
    grid = _grid()
    stepper = HybridPIC3DFieldStepper(grid)
    zero_current = np.zeros(grid.shape + (3,), dtype=float)
    result = stepper.step(
        stepper.maxwell.empty_state(),
        dt_s=1.0e-15,
        ion_current_A_m2=zero_current,
        electron_density_m3=np.full(grid.shape, 1.0e23),
        sigma0_S_m=0.0,
        background_density_m3=1.0e23,
        ohmic_cfl_safety=1.0,
        include_hall=False,
    )

    evidence = hybrid_stepper_candidate_evidence(result.telemetry)

    assert evidence["status"] == "candidate"
    assert evidence["can_support_first_principles_acceptance"] is False
