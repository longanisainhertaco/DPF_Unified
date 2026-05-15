import numpy as np

from dpf.fields.circuit_boundary import (
    CircuitMagneticBoundaryDrive,
    CircuitParameters,
    CircuitState,
    circuit_boundary_candidate_evidence,
)
from dpf.fields.maxwell_3d import MU_0, Maxwell3DGrid, Maxwell3DFieldCore
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(4, 4, 3), spacing=(1.0e-3, 1.0e-3, 2.0e-3))


def test_azimuthal_b_uses_source_eq_34() -> None:
    current_A = 2.0e4
    radius_m = 2.0e-3

    btheta = CircuitMagneticBoundaryDrive.azimuthal_B_T(current_A, radius_m)

    assert btheta == MU_0 * current_A / (2.0 * np.pi * radius_m)
    assert CircuitMagneticBoundaryDrive.azimuthal_B_T(2.0 * current_A, radius_m) == (
        2.0 * btheta
    )
    assert CircuitMagneticBoundaryDrive.azimuthal_B_T(current_A, 2.0 * radius_m) == (
        0.5 * btheta
    )


def test_explicit_rlc_update_matches_source_eq_37_38() -> None:
    grid = _grid()
    params = CircuitParameters(
        inductance_H=1.1e-7,
        voltage_V=1.5e4,
        resistance_ohm=1.2e-2,
        capacitance_F=2.0e-5,
    )
    drive = CircuitMagneticBoundaryDrive(grid, parameters=params)
    state = CircuitState(current_A=1.773e4, charge_C=0.218)

    next_state, telemetry = drive.step_circuit(state, dt_s=2.0e-13, udpf_V=50.0)

    expected_dI_dt = (
        params.voltage_V
        - params.resistance_ohm * state.current_A
        - state.charge_C / params.capacitance_F
        - 50.0
    ) / params.inductance_H
    assert telemetry.status == "candidate_engineering_rlc_circuit_step"
    assert telemetry.dI_dt_A_s == expected_dI_dt
    assert next_state.current_A == state.current_A + 2.0e-13 * expected_dI_dt
    assert next_state.charge_C == state.charge_C + 2.0e-13 * state.current_A


def test_cartesian_azimuthal_projection_has_expected_direction() -> None:
    grid = _grid()
    drive = CircuitMagneticBoundaryDrive(grid)

    field = drive.cell_centered_azimuthal_B_T(1.0e4)

    assert field.shape == grid.shape + (3,)
    assert np.all(np.isfinite(field))
    x_positive = field[3, 1, 0]
    y_positive = field[1, 3, 0]
    assert x_positive[1] > 0.0
    assert abs(x_positive[0]) < x_positive[1]
    assert y_positive[0] < 0.0
    assert abs(y_positive[1]) < abs(y_positive[0])
    assert np.all(field[..., 2] == 0.0)


def test_boundary_application_updates_injection_plane_only() -> None:
    grid = _grid()
    drive = CircuitMagneticBoundaryDrive(grid, radius_max_m=2.5e-3)
    state = Maxwell3DFieldCore(grid).empty_state()

    next_state, telemetry = drive.apply_injection_port_boundary(
        state,
        current_A=2.0e4,
        z_index=1,
    )

    assert telemetry.status == "candidate_engineering_magnetic_injection_boundary"
    assert telemetry.faces_updated > 0
    assert telemetry.btheta_abs_max_T > telemetry.btheta_abs_min_T >= 0.0
    assert np.count_nonzero(next_state.B.Bx_face[:, :, 1]) > 0
    assert np.count_nonzero(next_state.B.By_face[:, :, 1]) > 0
    assert np.count_nonzero(next_state.B.Bx_face[:, :, 0]) == 0
    assert np.count_nonzero(next_state.B.By_face[:, :, 2]) == 0


def test_circuit_boundary_candidate_evidence_does_not_satisfy_gate() -> None:
    grid = _grid()
    drive = CircuitMagneticBoundaryDrive(grid)
    state = Maxwell3DFieldCore(grid).empty_state()
    _, telemetry = drive.apply_injection_port_boundary(state, current_A=2.0e4)
    evidence = circuit_boundary_candidate_evidence(telemetry)

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "external_circuit_magnetic_boundary": evidence,
        },
    })

    assert evidence["status"] == "candidate"
    assert evidence["can_support_first_principles_acceptance"] is False
    assert "external_circuit_magnetic_boundary" in status["missing_capabilities"]
