import numpy as np
import pytest

from dpf import constants
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.circuit_boundary import CircuitMagneticBoundaryDrive, CircuitState
from dpf.fields.electron_energy import ElectronEnergyClosure
from dpf.fields.hybrid_loop import HybridPIC3DLoop
from dpf.fields.hybrid_simulator import (
    HybridPIC3DSimulator,
    hybrid_simulator_candidate_evidence,
)
from dpf.fields.ionization_transport import DeuteriumIonizationTransport
from dpf.fields.kinetic_yield import KineticIonYieldHistory
from dpf.fields.maxwell_3d import Maxwell3DGrid
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

M_D = 3.344e-27


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(5, 5, 5), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


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
        constants.e,
        positions=np.array([
            [2.0e-3, 2.0e-3, 2.0e-3],
            [2.4e-3, 2.0e-3, 2.0e-3],
        ]),
        velocities=np.array([
            [1.0e6, 0.0, 0.0],
            [-1.0e6, 0.0, 0.0],
        ]),
        weights=np.full(2, 1.0e8),
    )
    return pic


def test_hybrid_3d_simulator_runs_multiple_candidate_steps() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ionization_transport = DeuteriumIonizationTransport(grid)
    loop = HybridPIC3DLoop(
        grid,
        electron_energy_closure=closure,
        ionization_transport=ionization_transport,
        kinetic_yield_history=KineticIonYieldHistory(grid),
    )
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(1.0e5)
    state.B.Bz_face.fill(0.1)
    ne = np.full(grid.shape, 1.0e20)
    electron_state = closure.initialize(
        electron_temperature_K=1.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )
    ionization_state = ionization_transport.initialize(
        total_deuterium_density_m3=ne,
        ionization_fraction=0.01,
    )
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=state,
        pic=_pic(grid),
    )

    result = simulator.run(
        n_steps=3,
        dt_s=1.0e-13,
        sigma0_S_m=1.0e2,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        use_predictor_corrector=True,
        use_source_ordered_velocity_update=True,
        electron_energy_state=electron_state,
        ionization_state=ionization_state,
        use_source_backed_conductivity=True,
        mass_density_kg_m3=ne * M_D,
        plasma_velocity_m_s=np.zeros(grid.shape + (3,)),
        electron_temperature_floor_K=10.0,
    )

    assert result.telemetry.status == "candidate_engineering_3d_hybrid_pic_simulation"
    assert result.telemetry.n_steps_completed == 3
    np.testing.assert_allclose(result.telemetry.final_time_s, 3.0e-13)
    assert len(result.step_results) == 3
    assert result.electron_energy is not None
    assert result.ionization_charge_state is not None
    assert result.telemetry.last_step is not None
    assert result.telemetry.last_step["ionization_charge_state"]["status"] == (
        "candidate_deuterium_charge_state_transport"
    )
    assert result.telemetry.last_step["source_backed_transport"]["status"] == (
        "candidate_source_backed_partial_ionized_conductivity"
    )
    assert result.telemetry.last_step["kinetic_yield"]["cumulative_neutrons"] > 0.0
    assert result.telemetry.last_step["field_step"]["predictor_corrector"] is not None
    assert result.telemetry.last_step["source_workflow"]["status"] == (
        "candidate_engineering_source_ordered_loop"
    )
    assert result.telemetry.retained_step_result_count == 3
    assert result.telemetry.history_stride == 1
    assert result.telemetry.stop_reason == "completed_step_budget"
    assert result.telemetry.duration_request_satisfied is None
    limiter_summary = result.telemetry.limiter_activation_summary
    assert limiter_summary is not None
    assert limiter_summary["status"] == (
        "experimental_full_horizon_limiter_inventory_not_validation"
    )
    assert limiter_summary["steps_observed"] == 3
    assert set(limiter_summary["activation_counts"]) >= {
        "conductivity_ohmic_cfl_limited_steps",
        "marder_correction_steps",
        "electron_temperature_floor_contact_steps",
        "blocked_heat_flux_steps",
    }
    fingerprint = result.telemetry.state_fingerprint
    assert fingerprint is not None
    assert fingerprint["status"] == (
        "experimental_terminal_state_fingerprint_not_restart_acceptance"
    )
    assert len(fingerprint["sha256"]) == 64
    assert fingerprint["particle_count"] == sum(
        species.n_particles() for species in result.pic.species
    )
    assert [item["step_index"] for item in result.telemetry.history_summary] == [
        0,
        1,
        2,
    ]


def test_hybrid_simulator_can_thin_retained_history_for_long_runs() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=loop.field_stepper.maxwell.empty_state(),
        pic=_pic(grid),
    )

    result = simulator.run(
        n_steps=5,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        history_stride=2,
        max_step_results=2,
    )

    assert result.telemetry.n_steps_completed == 5
    np.testing.assert_allclose(result.telemetry.final_time_s, 5.0e-13)
    assert len(result.step_results) == 2
    assert result.telemetry.retained_step_result_count == 2
    assert result.telemetry.history_stride == 2
    assert result.telemetry.max_step_results == 2
    assert result.telemetry.last_completed_step_index == 4
    assert result.telemetry.last_step is not None
    assert result.telemetry.stop_reason == "completed_step_budget"
    assert [item["step_index"] for item in result.telemetry.history_summary] == [2, 4]


def test_hybrid_simulator_can_stop_at_target_time_without_history_miscount() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=loop.field_stepper.maxwell.empty_state(),
        pic=_pic(grid),
    )

    result = simulator.run(
        n_steps=10,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        history_stride=2,
        max_step_results=1,
        target_time_s=3.0e-13,
    )

    assert result.telemetry.n_steps_completed == 3
    np.testing.assert_allclose(result.telemetry.final_time_s, 3.0e-13)
    assert result.telemetry.stop_reason == "target_time_reached"
    assert result.telemetry.termination_reason == "target_time_reached"
    assert result.telemetry.duration_request_satisfied is True
    assert len(result.step_results) == 1
    assert result.telemetry.history_summary[-1]["step_index"] == 2


def test_hybrid_simulator_rejects_bad_long_run_controls() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=loop.field_stepper.maxwell.empty_state(),
        pic=_pic(grid),
    )
    kwargs = {
        "n_steps": 1,
        "dt_s": 1.0e-13,
        "sigma0_S_m": 0.0,
        "background_density_m3": 1.0e20,
        "ohmic_cfl_safety": 1.0,
        "density_floor_m3": 1.0e20,
        "include_hall": False,
    }

    with pytest.raises(ValueError, match="history_stride"):
        simulator.run(**kwargs, history_stride=0)
    with pytest.raises(ValueError, match="max_step_results"):
        simulator.run(**kwargs, max_step_results=-1)
    with pytest.raises(ValueError, match="target_time_s"):
        simulator.run(**kwargs, target_time_s=0.0)


def test_hybrid_simulator_candidate_evidence_does_not_satisfy_gate() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=loop.field_stepper.maxwell.empty_state(),
        pic=_pic(grid),
    )
    result = simulator.run(
        n_steps=1,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
    )

    candidate = hybrid_simulator_candidate_evidence(result.telemetry)
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "true_3d_dimensionality": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "true_3d_dimensionality" in status["missing_capabilities"]


def test_hybrid_simulator_can_apply_candidate_circuit_boundary() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=loop.field_stepper.maxwell.empty_state(),
        pic=_pic(grid),
        circuit_boundary=CircuitMagneticBoundaryDrive(grid),
    )

    result = simulator.run(
        n_steps=2,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        circuit_state=CircuitState(current_A=2.0e4, charge_C=0.20),
        apply_circuit_boundary=True,
        circuit_udpf_V=[0.0, 25.0],
    )

    assert result.circuit is not None
    assert result.circuit.current_A != 2.0e4
    assert result.telemetry.circuit is not None
    assert result.telemetry.circuit["n_steps"] == 2
    assert result.telemetry.circuit["current_history"][0]["sample"] == "initial"
    assert result.telemetry.circuit["current_history"][0]["current_A"] == 2.0e4
    assert result.telemetry.circuit["current_history"][-1]["sample"] == "post_step"
    assert len(result.telemetry.circuit["current_history"]) == 3
    assert result.telemetry.circuit["last"]["boundary"]["faces_updated"] > 0
    assert result.telemetry.circuit["last"]["udpf_source"] == "input_sequence"
    assert np.count_nonzero(result.state.B.Bx_face[:, :, 0]) > 0
    assert np.count_nonzero(result.state.B.By_face[:, :, 0]) > 0


def test_hybrid_simulator_keeps_circuit_history_denser_than_full_step_results() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=loop.field_stepper.maxwell.empty_state(),
        pic=_pic(grid),
        circuit_boundary=CircuitMagneticBoundaryDrive(grid),
    )

    result = simulator.run(
        n_steps=8,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        circuit_state=CircuitState(current_A=2.0e4, charge_C=0.20),
        apply_circuit_boundary=True,
        circuit_udpf_V=0.0,
        max_step_results=1,
    )

    assert result.telemetry.retained_step_result_count == 1
    assert result.telemetry.circuit is not None
    assert result.telemetry.circuit["current_history_cap"] == 64
    assert result.telemetry.circuit["retained_record_count"] == 8
    assert len(result.telemetry.circuit["current_history"]) == 9


def test_hybrid_simulator_can_use_lagged_volume_j_dot_e_circuit_feedback() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(1.0e5)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=state,
        pic=_pic(grid),
        circuit_boundary=CircuitMagneticBoundaryDrive(grid),
    )

    result = simulator.run(
        n_steps=2,
        dt_s=1.0e-13,
        sigma0_S_m=1.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        circuit_state=CircuitState(current_A=2.0e4, charge_C=0.20),
        apply_circuit_boundary=True,
        circuit_udpf_mode="lagged_volume_j_dot_e",
    )

    assert result.telemetry.circuit is not None
    assert result.telemetry.circuit["last"]["requested_udpf_mode"] == (
        "lagged_volume_j_dot_e"
    )
    assert result.telemetry.circuit["last"]["udpf_source"] == (
        "candidate_lagged_volume_j_dot_e"
    )
    assert result.telemetry.circuit["last"]["circuit_step"]["udpf_V"] != 0.0


def test_hybrid_simulator_blocks_j_dot_e_feedback_at_low_current() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(1.0e5)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=state,
        pic=_pic(grid),
        circuit_boundary=CircuitMagneticBoundaryDrive(grid),
    )

    result = simulator.run(
        n_steps=2,
        dt_s=1.0e-13,
        sigma0_S_m=1.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        circuit_state=CircuitState(current_A=0.01, charge_C=0.0),
        apply_circuit_boundary=True,
        circuit_udpf_mode="lagged_volume_j_dot_e",
        circuit_feedback_min_current_A=1.0,
    )

    assert result.telemetry.circuit is not None
    assert result.telemetry.circuit["last"]["udpf_source"] == (
        "input_sequence_fallback_low_current"
    )
    assert result.telemetry.circuit["last"]["circuit_step"]["udpf_V"] == 0.0
    assert result.telemetry.circuit["last"]["feedback_min_current_A"] == 1.0


def test_hybrid_simulator_requires_boundary_when_circuit_requested() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    simulator = HybridPIC3DSimulator(
        grid=grid,
        loop=loop,
        state=loop.field_stepper.maxwell.empty_state(),
        pic=_pic(grid),
    )

    try:
        simulator.run(
            n_steps=1,
            dt_s=1.0e-13,
            sigma0_S_m=0.0,
            background_density_m3=1.0e20,
            ohmic_cfl_safety=1.0,
            density_floor_m3=1.0e20,
            include_hall=False,
            apply_circuit_boundary=True,
        )
    except ValueError as exc:
        assert "circuit_boundary is required" in str(exc)
    else:
        raise AssertionError("expected missing circuit boundary to fail closed")
