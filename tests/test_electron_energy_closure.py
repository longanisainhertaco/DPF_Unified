import numpy as np
from scipy.constants import Boltzmann as K_B

from dpf.fields.electron_energy import (
    ElectronEnergyClosure,
    electron_energy_candidate_evidence,
    extended_ohm_temperature_authority_status,
)
from dpf.fields.maxwell_3d import Maxwell3DGrid
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
        ne * K_B * 2.0e5,
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
