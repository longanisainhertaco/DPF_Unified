import numpy as np
import pytest

import dpf.fields.electron_energy as electron_energy_module
from dpf.constants import k_B
from dpf.fields.electron_energy import (
    ElectronEnergyClosure,
    electron_energy_candidate_evidence,
    extended_ohm_temperature_authority_status,
)
from dpf.fields.maxwell_3d import Maxwell3DGrid
from dpf.fluid.implicit_diffusion import diffuse_field_1d_backward_euler
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

M_D = 3.344e-27


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(3, 3, 3), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def test_electron_energy_closure_initializes_pressure_from_separate_te() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.full(grid.shape, 1.0e20)

    state = closure.initialize(
        electron_temperature_K=2.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )

    assert state.electron_energy_J_m3.shape == grid.shape
    assert np.all(state.electron_temperature_K == 2.0e5)
    assert np.all(state.ion_temperature_K == 1.0e5)
    np.testing.assert_allclose(
        state.electron_pressure_Pa(ne),
        ne * k_B * 2.0e5,
    )


def test_electron_energy_closure_heats_electrons_from_ohmic_source() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.full(grid.shape, 1.0e20)
    ni = np.full(grid.shape, 1.0e20)
    state = closure.initialize(
        electron_temperature_K=1.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )
    current = np.zeros(grid.shape + (3,))
    current[..., 0] = 1.0e8

    updated, telemetry = closure.step_sources(
        state,
        electron_density_m3=ne,
        ion_density_m3=ni,
        mass_density_kg_m3=ni * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=1.0e-6,
        current_A_m2=current,
        dt_s=1.0e-10,
        temperature_floor_K=10.0,
    )

    assert telemetry.status == "candidate_engineering_electron_energy_closure"
    assert telemetry.include_ohmic_heating is True
    assert telemetry.include_equilibration is True
    assert telemetry.include_bremsstrahlung_loss is True
    assert telemetry.max_abs_delta_electron_temperature_K > 0.0
    assert np.mean(updated.electron_temperature_K) > np.mean(state.electron_temperature_K)


def test_electron_energy_closure_reconciles_energy_after_density_change() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    initial_ne = np.full(grid.shape, 1.0e20)
    current_ne = np.ones(grid.shape)
    current_ni = np.ones(grid.shape)
    state = closure.initialize(
        electron_temperature_K=1.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=initial_ne,
    )

    updated, telemetry = closure.step_sources(
        state,
        electron_density_m3=current_ne,
        ion_density_m3=current_ni,
        mass_density_kg_m3=current_ni * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=0.0,
        current_A_m2=np.zeros(grid.shape + (3,)),
        dt_s=1.0e-10,
        temperature_floor_K=10.0,
    )

    assert telemetry.status == "candidate_engineering_electron_energy_closure"
    assert telemetry.closure_validity["status"] == (
        "candidate_nonrelativistic_electron_closure_in_range"
    )
    assert telemetry.density_reconciliation["status"] == (
        "candidate_density_temperature_energy_reconciled"
    )
    assert telemetry.density_reconciliation["preserves_temperature_field"] is True
    assert telemetry.density_reconciliation["max_relative_delta"] > 0.9
    assert np.max(updated.electron_temperature_K) < 1.0e8
    np.testing.assert_allclose(
        updated.electron_temperature_K,
        state.electron_temperature_K,
        rtol=1.0e-8,
    )


def test_electron_heat_flux_skips_all_numerical_floor_density_cells() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.ones(grid.shape)
    initial_te = np.full(grid.shape, 1.0e5)
    initial_te[1, 1, 1] = 1.0e6
    state = closure.initialize(
        electron_temperature_K=initial_te,
        ion_temperature_K=initial_te,
        electron_density_m3=ne,
    )

    updated, telemetry = closure.step_sources(
        state,
        electron_density_m3=ne,
        ion_density_m3=ne,
        mass_density_kg_m3=ne * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=0.0,
        current_A_m2=np.zeros(grid.shape + (3,)),
        dt_s=1.0e-10,
        temperature_floor_K=10.0,
        magnetic_field_T=np.full(grid.shape + (3,), [0.0, 0.0, 0.1]),
        heat_flux_subcycles_max=1,
    )

    assert telemetry.heat_flux["status"] == (
        "not_applied_no_resolved_plasma_electron_density"
    )
    assert telemetry.heat_flux["applied"] is False
    assert telemetry.heat_flux["density_gate"]["all_cells_at_numerical_floor"] is True
    assert telemetry.heat_flux["temperature_floor_contact_count"] == 0
    assert telemetry.include_heat_flux is False
    np.testing.assert_allclose(
        updated.electron_temperature_K,
        state.electron_temperature_K,
        rtol=1.0e-8,
    )


def test_electron_current_drift_validity_ignores_numerical_floor_cells() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.ones(grid.shape)
    ne[1, 1, 1] = 1.0e20
    state = closure.initialize(
        electron_temperature_K=1.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )
    current = np.zeros(grid.shape + (3,))
    current[0, 0, 0, 0] = 1.0e8
    current[1, 1, 1, 0] = 1.0e8

    _, telemetry = closure.step_sources(
        state,
        electron_density_m3=ne,
        ion_density_m3=ne,
        mass_density_kg_m3=ne * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=0.0,
        current_A_m2=current,
        dt_s=1.0e-10,
        temperature_floor_K=10.0,
    )

    domain = telemetry.closure_validity["electron_fluid_domain"]
    assert telemetry.status == "candidate_engineering_electron_energy_closure"
    assert telemetry.closure_validity["status"] == (
        "candidate_nonrelativistic_electron_closure_in_range"
    )
    assert domain["resolved_cell_count"] == 1
    assert domain["excluded_numerical_floor_cell_count"] == int(np.prod(grid.shape)) - 1
    assert domain["max_excluded_numerical_floor_current_A_m2"] == pytest.approx(
        1.0e8
    )
    assert telemetry.closure_validity["current_drift_to_c"] < 1.0


def test_electron_energy_closure_uses_diagonal_adi_when_tensor_solve_fails(
    monkeypatch,
) -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.full(grid.shape, 1.0e20)
    ni = np.full(grid.shape, 1.0e20)
    initial_te = np.full(grid.shape, 1.0e7)
    initial_te[0, :, :] = 1.1e7
    state = closure.initialize(
        electron_temperature_K=initial_te,
        ion_temperature_K=1.0e7,
        electron_density_m3=ne,
    )

    def _fail_gmres(operator, rhs, **kwargs):  # noqa: ANN001, ANN202
        return rhs.copy(), 1

    monkeypatch.setattr(electron_energy_module, "gmres", _fail_gmres)
    updated, telemetry = closure.step_sources(
        state,
        electron_density_m3=ne,
        ion_density_m3=ni,
        mass_density_kg_m3=ni * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=0.0,
        current_A_m2=np.zeros(grid.shape + (3,)),
        dt_s=1.0e-13,
        temperature_floor_K=10.0,
        magnetic_field_T=np.full(grid.shape + (3,), [0.0, 0.0, 0.1]),
        heat_flux_subcycles_max=1,
    )

    assert telemetry.heat_flux["status"] == (
        "candidate_braginskii_diagonal_adi_heat_flux_applied"
    )
    assert telemetry.heat_flux["fallback_after_tensor_solver_failure"] is True
    assert telemetry.heat_flux["implicit_scheme"] == (
        "diagonal_braginskii_backward_euler_adi_fallback"
    )
    assert telemetry.heat_flux[
        "positivity_preserving_for_nonnegative_temperature"
    ] is True
    assert telemetry.heat_flux["temperature_floor_contact_count"] == 0
    assert telemetry.heat_flux["omitted_cross_derivative_terms"] is True
    assert telemetry.heat_flux["nonlinear_kappa_frozen"] is True
    assert telemetry.heat_flux["required_subcycles"] > 1
    assert np.all(np.isfinite(updated.electron_temperature_K))


def test_backward_euler_diffusion_preserves_positive_temperature() -> None:
    field = np.array([10.0, 1.0e7, 10.0], dtype=float)
    coeff = np.full_like(field, 1.0e18)

    updated = diffuse_field_1d_backward_euler(
        field,
        coeff,
        dt=1.0e-9,
        dx=1.0e-3,
    )

    assert np.all(np.isfinite(updated))
    assert float(np.min(updated)) >= float(np.min(field))
    assert float(np.max(updated)) <= float(np.max(field))


def test_electron_energy_candidate_evidence_does_not_satisfy_gate() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.full(grid.shape, 1.0e20)
    state = closure.initialize(
        electron_temperature_K=1.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )
    _, telemetry = closure.step_sources(
        state,
        electron_density_m3=ne,
        ion_density_m3=ne,
        mass_density_kg_m3=ne * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=0.0,
        current_A_m2=np.zeros(grid.shape + (3,)),
        dt_s=1.0e-10,
        temperature_floor_K=10.0,
    )

    candidate = electron_energy_candidate_evidence(telemetry)
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "separate_electron_energy_closure": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert candidate["density_reconciliation"]["status"] == (
        "density_temperature_energy_already_consistent"
    )
    assert "separate_electron_energy_closure" in status["missing_capabilities"]


def test_extended_ohm_temperature_authority_blocks_te_equal_ti_closure() -> None:
    status = extended_ohm_temperature_authority_status(
        include_hall=True,
        include_pressure=True,
        electron_energy_evidence=None,
    )

    assert status["status"] == "blocked_te_equal_ti_or_missing_separate_te"
    assert status["requires_separate_te"] is True
    assert status["can_support_pressure_hall_quantitative_claims"] is False


def test_extended_ohm_temperature_authority_keeps_candidate_te_blocked() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.full(grid.shape, 1.0e20)
    state = closure.initialize(
        electron_temperature_K=1.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )
    _, telemetry = closure.step_sources(
        state,
        electron_density_m3=ne,
        ion_density_m3=ne,
        mass_density_kg_m3=ne * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=0.0,
        current_A_m2=np.zeros(grid.shape + (3,)),
        dt_s=1.0e-10,
        temperature_floor_K=10.0,
    )

    status = extended_ohm_temperature_authority_status(
        include_hall=True,
        include_pressure=False,
        electron_energy_evidence=telemetry,
    )

    assert status["status"] == "candidate_separate_te_still_blocked"
    assert status["can_support_pressure_hall_quantitative_claims"] is False


def test_extended_ohm_temperature_authority_allows_baseline_resistive_case() -> None:
    status = extended_ohm_temperature_authority_status(
        include_hall=False,
        include_pressure=False,
        electron_energy_evidence=None,
    )

    assert status["status"] == "not_required_for_baseline_resistive_ohm"
    assert status["requires_separate_te"] is False
    assert status["can_support_pressure_hall_quantitative_claims"] is True
