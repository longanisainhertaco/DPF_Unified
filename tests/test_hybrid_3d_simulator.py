import numpy as np

from dpf.constants import e as ELEMENTARY_CHARGE
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.circuit_boundary import CircuitMagneticBoundaryDrive, CircuitState
from dpf.fields.electron_energy import ElectronEnergyClosure
from dpf.fields.hybrid_loop import HybridPIC3DLoop
from dpf.fields.hybrid_simulator import (
    HybridPIC3DSimulator,
    hybrid_simulator_candidate_evidence,
)
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
        ELEMENTARY_CHARGE,
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
    loop = HybridPIC3DLoop(
        grid,
        electron_energy_closure=closure,
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
        mass_density_kg_m3=ne * M_D,
        plasma_velocity_m_s=np.zeros(grid.shape + (3,)),
        electron_temperature_floor_K=10.0,
    )

    assert result.telemetry.status == "candidate_engineering_3d_hybrid_pic_simulation"
    assert result.telemetry.n_steps_completed == 3
    np.testing.assert_allclose(result.telemetry.final_time_s, 3.0e-13)
    assert len(result.step_results) == 3
    assert result.electron_energy is not None
    assert result.telemetry.last_step is not None
    assert result.telemetry.last_step["kinetic_yield"]["cumulative_neutrons"] > 0.0
    assert result.telemetry.last_step["field_step"]["predictor_corrector"] is not None
    assert result.telemetry.last_step["source_workflow"]["status"] == (
        "candidate_engineering_source_ordered_loop"
    )


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
    assert result.telemetry.circuit["last"]["boundary"]["faces_updated"] > 0
    assert np.count_nonzero(result.state.B.Bx_face[:, :, 0]) > 0
    assert np.count_nonzero(result.state.B.By_face[:, :, 0]) > 0


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
