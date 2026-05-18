import numpy as np

from dpf.constants import e
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.electron_energy import ElectronEnergyClosure
from dpf.fields.hybrid_loop import (
    HybridPIC3DLoop,
    hybrid_loop_candidate_evidence,
    ion_collision_loop_candidate_evidence,
    source_ordered_loop_candidate_evidence,
)
from dpf.fields.ionization_transport import DeuteriumIonizationTransport
from dpf.fields.kinetic_yield import KineticIonYieldHistory
from dpf.fields.maxwell_3d import Maxwell3DGrid
from dpf.fields.particle_boundaries import ParticleAbsorbingBoundaries
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

M_D = 3.344e-27


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(5, 5, 5), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def _pic(grid: Maxwell3DGrid, *, use_esirkepov: bool = True) -> HybridPIC:
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=1.0e-12,
        use_esirkepov=use_esirkepov,
        use_binary_collisions=False,
    )
    pic.add_species(
        "d",
        M_D,
        e,
        positions=np.array([
            [2.0e-3, 2.0e-3, 2.0e-3],
            [2.5e-3, 2.0e-3, 2.0e-3],
            [2.0e-3, 2.5e-3, 2.0e-3],
        ]),
        velocities=np.array([
            [1.0e5, 0.0, 0.0],
            [0.0, 1.0e5, 0.0],
            [0.0, 0.0, 1.0e5],
        ]),
        weights=np.full(3, 1.0e8),
    )
    return pic


def test_hybrid_loop_pushes_particles_deposits_current_and_advances_fields() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    pic = _pic(grid)
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(5.0)
    before_positions = pic.species[0].positions.copy()

    result = loop.step(
        state,
        pic,
        dt_s=1.0e-13,
        sigma0_S_m=1.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e12,
        include_hall=False,
    )

    assert result.telemetry.status == "candidate_engineering_particle_field_loop"
    assert result.telemetry.n_particles_before == 3
    assert result.telemetry.n_particles_after == 3
    assert result.telemetry.deposition_method == "esirkepov"
    assert result.telemetry.collision_operator["status"] == "disabled"
    assert result.ion_current_A_m2.shape == grid.shape + (3,)
    assert result.electron_density_m3.shape == grid.shape
    assert result.telemetry.ion_current_max_A_m2 > 0.0
    assert np.max(np.abs(pic.species[0].positions - before_positions)) > 0.0
    assert result.field_step.telemetry.ohm_solver["max_current_A_m2"] >= 0.0


def test_hybrid_loop_candidate_evidence_does_not_satisfy_hybrid_gate() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    result = loop.step(
        loop.field_stepper.maxwell.empty_state(),
        _pic(grid, use_esirkepov=False),
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e12,
        include_hall=False,
    )
    candidate = hybrid_loop_candidate_evidence(result.telemetry)

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "kinetic_ion_pic_push_deposition": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "kinetic_ion_pic_push_deposition" in status["missing_capabilities"]


def test_hybrid_loop_can_apply_candidate_particle_absorption_before_deposit() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(
        grid,
        particle_boundaries=ParticleAbsorbingBoundaries(grid, pml_cells=1),
    )
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
        e,
        positions=np.array([
            [0.2e-3, 2.0e-3, 2.0e-3],
            [2.0e-3, 2.0e-3, 2.0e-3],
        ]),
        velocities=np.zeros((2, 3)),
        weights=np.full(2, 1.0e8),
    )

    result = loop.step(
        loop.field_stepper.maxwell.empty_state(),
        pic,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e12,
        include_hall=False,
    )

    boundary = result.telemetry.particle_boundaries
    assert boundary is not None
    assert boundary["status"] == "candidate_engineering_particle_absorption"
    assert boundary["deleted_pml"] == 1
    assert result.telemetry.n_particles_before == 2
    assert result.telemetry.n_particles_after == 1
    assert pic.species[0].n_particles() == 1


def test_source_ordered_loop_reports_boundary_density_bypass() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(
        grid,
        particle_boundaries=ParticleAbsorbingBoundaries(grid, pml_cells=1),
    )
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
        e,
        positions=np.array([
            [0.2e-3, 2.0e-3, 2.0e-3],
            [2.0e-3, 2.0e-3, 2.0e-3],
        ]),
        velocities=np.zeros((2, 3)),
        weights=np.full(2, 1.0e8),
    )

    result = loop.step(
        loop.field_stepper.maxwell.empty_state(),
        pic,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e12,
        include_hall=False,
        use_source_ordered_velocity_update=True,
    )

    workflow = result.telemetry.source_workflow
    density_rebuild = workflow["density_rebuild"]
    assert workflow["status"] == "candidate_engineering_source_ordered_loop"
    assert density_rebuild["half_step_density_available"] is True
    assert density_rebuild["half_step_density_bypassed_by_particle_boundary"] is True
    assert density_rebuild["density_sample"] == "x_n_plus_1"
    assert density_rebuild["decision"] == "use_post_boundary_x_n_plus_1_density"


def test_hybrid_loop_reports_candidate_nanbu_collision_telemetry() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=1.0e-12,
        use_esirkepov=False,
        use_binary_collisions=True,
    )
    pic.enable_collisions(n_background=1.0e20, T_background_eV=500.0)
    pic.add_species(
        "d",
        M_D,
        e,
        positions=np.array([
            [2.0e-3, 2.0e-3, 2.0e-3],
            [2.4e-3, 2.0e-3, 2.0e-3],
        ]),
        velocities=np.array([
            [1.0e5, 0.0, 0.0],
            [-1.0e5, 0.0, 0.0],
        ]),
        weights=np.full(2, 1.0e8),
    )

    result = loop.step(
        loop.field_stepper.maxwell.empty_state(),
        pic,
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e12,
        include_hall=False,
    )
    candidate = ion_collision_loop_candidate_evidence(result.telemetry)
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "ion_collision_operator": candidate,
        },
    })

    assert result.telemetry.collision_operator["status"] == "candidate_enabled"
    assert result.telemetry.collision_operator["algorithm"] == "nanbu_perez_binary"
    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "ion_collision_operator" in status["missing_capabilities"]


def test_hybrid_loop_can_apply_candidate_electron_energy_update() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    loop = HybridPIC3DLoop(grid, electron_energy_closure=closure)
    ne = np.full(grid.shape, 1.0e20)
    electron_state = closure.initialize(
        electron_temperature_K=1.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(1.0e6)

    result = loop.step(
        state,
        _pic(grid, use_esirkepov=False),
        dt_s=1.0e-13,
        sigma0_S_m=1.0e3,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        electron_energy_state=electron_state,
        mass_density_kg_m3=ne * M_D,
        plasma_velocity_m_s=np.zeros(grid.shape + (3,)),
        electron_temperature_floor_K=10.0,
    )

    assert result.electron_energy is not None
    assert result.telemetry.electron_energy is not None
    assert result.telemetry.pressure_gradient is not None
    assert result.telemetry.electron_energy["status"] == (
        "candidate_engineering_electron_energy_closure"
    )
    assert result.telemetry.electron_energy["heat_flux"]["status"] == (
        "candidate_braginskii_anisotropic_heat_flux_applied"
    )
    assert result.telemetry.electron_energy["equilibration_audit"]["status"] == (
        "candidate_nrl_equal_temperature_equilibration_audit"
    )
    assert result.telemetry.electron_energy["include_heat_flux"] is True
    assert result.telemetry.temperature_authority["status"] == (
        "candidate_separate_te_still_blocked"
    )
    assert result.field_step.telemetry.ohm_solver["include_pressure"] is True
    assert np.mean(result.electron_energy.electron_temperature_K) > np.mean(
        electron_state.electron_temperature_K
    )


def test_hybrid_loop_can_apply_candidate_ionization_transport() -> None:
    grid = _grid()
    electron_closure = ElectronEnergyClosure(grid)
    ionization_transport = DeuteriumIonizationTransport(grid)
    loop = HybridPIC3DLoop(
        grid,
        electron_energy_closure=electron_closure,
        ionization_transport=ionization_transport,
    )
    ne = np.full(grid.shape, 1.0e20)
    electron_state = electron_closure.initialize(
        electron_temperature_K=4.0e5,
        ion_temperature_K=1.0e5,
        electron_density_m3=ne,
    )
    ionization_state = ionization_transport.initialize(
        total_deuterium_density_m3=ne,
        ionization_fraction=0.01,
    )

    result = loop.step(
        loop.field_stepper.maxwell.empty_state(),
        _pic(grid, use_esirkepov=False),
        dt_s=1.0e-13,
        sigma0_S_m=1.0e3,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        electron_energy_state=electron_state,
        ionization_state=ionization_state,
        use_source_backed_conductivity=True,
        mass_density_kg_m3=ne * M_D,
        plasma_velocity_m_s=np.zeros(grid.shape + (3,)),
        electron_temperature_floor_K=10.0,
    )

    assert result.ionization_charge_state is not None
    assert result.telemetry.ionization_charge_state is not None
    assert result.telemetry.ionization_charge_state["status"] == (
        "candidate_deuterium_charge_state_transport"
    )
    assert result.telemetry.ionization_charge_state["particle_source"]["status"] == (
        "candidate_ionization_pic_particle_source"
    )
    assert result.telemetry.source_backed_transport["status"] == (
        "candidate_source_backed_partial_ionized_conductivity"
    )
    assert result.telemetry.field_step["conductivity"][
        "density_blend_applied"
    ] is False
    assert result.telemetry.field_step["conductivity"][
        "ohmic_cfl_limit_applied"
    ] is False
    assert result.telemetry.field_step["ohm_solver"][
        "electric_update_scheme"
    ] == "backward_euler_resistive_ampere_ohm"
    assert result.telemetry.ionization_charge_state[
        "can_support_first_principles_acceptance"
    ] is False
    assert np.mean(result.ionization_charge_state.mean_charge_state) >= np.mean(
        ionization_state.mean_charge_state
    )


def test_electron_energy_closure_applies_candidate_braginskii_heat_flux() -> None:
    grid = _grid()
    closure = ElectronEnergyClosure(grid)
    ne = np.full(grid.shape, 1.0e20)
    Te = np.full(grid.shape, 1.0e5)
    Te[2, 2, 2] = 2.0e5
    electron_state = closure.initialize(
        electron_temperature_K=Te,
        ion_temperature_K=Te,
        electron_density_m3=ne,
    )

    next_state, telemetry = closure.step_sources(
        electron_state,
        electron_density_m3=ne,
        ion_density_m3=ne,
        mass_density_kg_m3=ne * M_D,
        velocity_m_s=np.zeros(grid.shape + (3,)),
        resistivity_ohm_m=0.0,
        current_A_m2=np.zeros(grid.shape + (3,)),
        dt_s=1.0e-10,
        gaunt_factor=0.0,
        temperature_floor_K=10.0,
        magnetic_field_T=np.zeros(grid.shape + (3,)),
    )

    assert telemetry.heat_flux["status"] == (
        "candidate_braginskii_anisotropic_heat_flux_applied"
    )
    assert telemetry.heat_flux["applied"] is True
    assert telemetry.include_heat_flux is True
    assert telemetry.can_support_first_principles_acceptance is False
    assert next_state.electron_temperature_K[2, 2, 2] < Te[2, 2, 2]
    assert next_state.electron_temperature_K[2, 2, 1] > Te[2, 2, 1]


def test_hybrid_loop_blocks_extended_ohm_temperature_authority_without_te() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)

    result = loop.step(
        loop.field_stepper.maxwell.empty_state(),
        _pic(grid, use_esirkepov=False),
        dt_s=1.0e-13,
        sigma0_S_m=1.0e2,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=True,
    )

    assert result.telemetry.temperature_authority["status"] == (
        "blocked_te_equal_ti_or_missing_separate_te"
    )
    assert result.telemetry.temperature_authority[
        "can_support_pressure_hall_quantitative_claims"
    ] is False


def test_hybrid_loop_can_accumulate_candidate_kinetic_yield_history() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid, kinetic_yield_history=KineticIonYieldHistory(grid))
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
        e,
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

    result = loop.step(
        loop.field_stepper.maxwell.empty_state(),
        pic,
        dt_s=1.0e-12,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e24,
        include_hall=False,
    )

    assert result.telemetry.kinetic_yield is not None
    assert result.telemetry.kinetic_yield["status"] == (
        "candidate_engineering_kinetic_yield_history"
    )
    assert result.telemetry.kinetic_yield["neutron_rate_per_s"] > 0.0
    assert result.telemetry.kinetic_yield["cumulative_neutrons"] > 0.0
    assert result.telemetry.neutron_yield_authority is not None
    assert result.telemetry.neutron_yield_authority["status"] == "blocked"
    assert "mechanism_separated_yield_channels" in (
        result.telemetry.neutron_yield_authority["missing_evidence"]
    )


def test_hybrid_loop_can_request_candidate_predictor_corrector() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(50.0)

    result = loop.step(
        state,
        _pic(grid, use_esirkepov=False),
        dt_s=1.0e-13,
        sigma0_S_m=1.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        use_predictor_corrector=True,
    )

    predictor = result.field_step.telemetry.predictor_corrector
    assert predictor is not None
    assert predictor["status"] == "candidate_engineering_predictor_corrector"
    assert result.field_step.end_step_current_A_m2.shape == grid.shape + (3,)


def test_hybrid_loop_can_request_candidate_marder_correction() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    state = loop.field_stepper.maxwell.empty_state()
    x = np.arange(grid.nx, dtype=float) * grid.dx
    state.E.Ex_edge[:, :, :] = np.sin(
        2.0 * np.pi * x[:, np.newaxis, np.newaxis] / (grid.nx * grid.dx)
    )

    result = loop.step(
        state,
        _pic(grid, use_esirkepov=False),
        dt_s=1.0e-13,
        sigma0_S_m=0.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        marder_factor_m2=0.001 * grid.dx**2,
        marder_nondominance_threshold=0.5,
    )

    assert result.field_step.telemetry.marder is not None
    assert result.field_step.telemetry.marder["status"] == (
        "candidate_engineering_marder_correction"
    )
    assert result.field_step.telemetry.marder["nondominance_status"] == (
        "candidate_dominant_correction"
    )


def test_hybrid_loop_can_run_candidate_source_ordered_eq7_update() -> None:
    grid = _grid()
    loop = HybridPIC3DLoop(grid)
    pic = _pic(grid, use_esirkepov=True)
    state = loop.field_stepper.maxwell.empty_state()
    state.E.Ex_edge.fill(25.0)
    state.B.Bz_face.fill(0.2)
    before_positions = pic.species[0].positions.copy()
    before_velocities = pic.species[0].velocities.copy()

    result = loop.step(
        state,
        pic,
        dt_s=1.0e-13,
        sigma0_S_m=50.0,
        background_density_m3=1.0e20,
        ohmic_cfl_safety=1.0,
        density_floor_m3=1.0e20,
        include_hall=False,
        use_predictor_corrector=True,
        marder_factor_m2=0.01 * grid.dx**2,
        marder_nondominance_threshold=0.5,
        use_source_ordered_velocity_update=True,
    )
    workflow = result.telemetry.source_workflow
    candidate = source_ordered_loop_candidate_evidence(result.telemetry)
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "source_ordered_time_loop": candidate,
        },
    })

    assert workflow["status"] == "candidate_engineering_source_ordered_loop"
    assert workflow["position_update"]["max_displacement_m"] > 0.0
    assert workflow["velocity_update"]["max_delta_v_m_s"] > 0.0
    assert workflow["predictor_particle_rebuild"]["status"] == (
        "candidate_engineering_predictor_particle_rebuild"
    )
    assert workflow["predictor_particle_rebuild"]["feeds_corrected_current"] is True
    assert (
        "runtime_predictor_particle_rebuild_executed_review_required"
        in workflow["stages_executed"]
    )
    assert (
        "long_run_source_ordered_stability_review_required"
        in workflow["review_required_stages"]
    )
    assert "same_scope_3d_validation" in workflow["acceptance_blocking_stages"]
    assert "electron_density_from_x_n_plus_half" in workflow["stages_executed"]
    assert result.telemetry.deposition_method == "esirkepov"
    assert np.max(np.abs(pic.species[0].positions - before_positions)) > 0.0
    assert np.max(np.abs(pic.species[0].velocities - before_velocities)) > 0.0
    assert result.field_step.telemetry.predictor_corrector is not None
    assert result.field_step.telemetry.marder is not None
    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "source_ordered_time_loop" in status["missing_capabilities"]
