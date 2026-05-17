"""Candidate particle-field loop for the 3-D hybrid PIC-fluid path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from dpf.constants import e
from dpf.experimental.pic.hybrid import (
    HybridPIC,
    deposit_current,
    deposit_density,
    interpolate_field_to_particles,
)
from dpf.fields.conductivity import (
    PartialIonizedConductivityTelemetry,
    partial_ionized_conductivity,
)
from dpf.fields.electron_energy import (
    ElectronEnergyClosure,
    ElectronEnergyState,
    ElectronEnergyTelemetry,
    extended_ohm_temperature_authority_status,
)
from dpf.fields.hybrid_stepper import (
    HybridPIC3DFieldStepper,
    HybridPIC3DStepResult,
)
from dpf.fields.ionization_transport import (
    DeuteriumIonizationState,
    DeuteriumIonizationTransport,
    IonizationParticleSourceTelemetry,
    IonizationTransportTelemetry,
    apply_ionization_particle_source,
)
from dpf.fields.kinetic_yield import (
    KineticIonYieldHistory,
    KineticYieldTelemetry,
    kinetic_neutron_yield_authority_status,
)
from dpf.fields.maxwell_3d import (
    HYBRID_PIC_3D_SOURCE,
    Maxwell3DBoundaries,
    Maxwell3DGrid,
    Maxwell3DState,
)
from dpf.fields.particle_boundaries import (
    ParticleAbsorbingBoundaries,
    ParticleBoundaryTelemetry,
)
from dpf.fluid.constrained_transport import face_to_cell_centered


@dataclass(frozen=True)
class HybridPIC3DLoopTelemetry:
    """Telemetry for one candidate particle-field loop step."""

    status: str
    source: str
    n_particles_before: int
    n_particles_after: int
    deposition_method: str
    particle_boundaries: dict[str, Any] | None
    collision_operator: dict[str, Any]
    pressure_gradient: dict[str, Any] | None
    source_backed_transport: dict[str, Any] | None
    electron_energy: dict[str, Any] | None
    ionization_charge_state: dict[str, Any] | None
    temperature_authority: dict[str, Any]
    kinetic_yield: dict[str, Any] | None
    neutron_yield_authority: dict[str, Any] | None
    source_workflow: dict[str, Any]
    electron_density_min_m3: float
    electron_density_max_m3: float
    ion_current_max_A_m2: float
    field_step: dict[str, Any]
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridPIC3DLoopResult:
    """Result from one candidate particle-field loop step."""

    state: Maxwell3DState
    rho_C_m3: np.ndarray
    ion_current_A_m2: np.ndarray
    electron_density_m3: np.ndarray
    electron_energy: ElectronEnergyState | None
    ionization_charge_state: DeuteriumIonizationState | None
    field_step: HybridPIC3DStepResult
    telemetry: HybridPIC3DLoopTelemetry


class HybridPIC3DLoop:
    """Push ions, deposit current, rebuild quasi-neutral density, advance fields."""

    def __init__(
        self,
        grid: Maxwell3DGrid,
        maxwell_boundaries: Maxwell3DBoundaries | None = None,
        particle_boundaries: ParticleAbsorbingBoundaries | None = None,
        electron_energy_closure: ElectronEnergyClosure | None = None,
        ionization_transport: DeuteriumIonizationTransport | None = None,
        kinetic_yield_history: KineticIonYieldHistory | None = None,
    ) -> None:
        self.grid = grid
        self.field_stepper = HybridPIC3DFieldStepper(
            grid,
            boundaries=maxwell_boundaries,
        )
        self.particle_boundaries = particle_boundaries
        self.electron_energy_closure = electron_energy_closure
        self.ionization_transport = ionization_transport
        self.kinetic_yield_history = kinetic_yield_history

    def cell_centered_fields(
        self,
        state: Maxwell3DState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cell-centered E and B fields for PIC interpolation."""
        E_cell = np.stack(
            self.field_stepper.maxwell.edge_E_to_cell_centered(state.E),
            axis=-1,
        )
        B_cell = np.stack(face_to_cell_centered(state.B), axis=-1)
        return E_cell, B_cell

    def step(
        self,
        state: Maxwell3DState,
        pic: HybridPIC,
        *,
        dt_s: float,
        sigma0_S_m: np.ndarray | float,
        background_density_m3: float,
        ohmic_cfl_safety: float,
        density_floor_m3: float = 1.0,
        pressure_term_V_m: np.ndarray | None = None,
        include_hall: bool = True,
        use_predictor_corrector: bool = False,
        marder_factor_m2: float = 0.0,
        marder_nondominance_threshold: float | None = None,
        electron_energy_state: ElectronEnergyState | None = None,
        ionization_state: DeuteriumIonizationState | None = None,
        use_source_backed_conductivity: bool = False,
        mass_density_kg_m3: np.ndarray | None = None,
        plasma_velocity_m_s: np.ndarray | None = None,
        charge_state_Z: float = 1.0,
        electron_temperature_floor_K: float = 1.0,
        heat_flux_subcycles_max: int = 1000,
        pressure_density_threshold_m3: float = 0.0,
        use_source_ordered_velocity_update: bool = False,
    ) -> HybridPIC3DLoopResult:
        if density_floor_m3 <= 0.0:
            raise ValueError("density_floor_m3 must be positive")
        if tuple(pic.grid_shape) != self.grid.shape:
            raise ValueError("PIC grid shape does not match Maxwell grid")
        n_before = _particle_count(pic)
        half_step_charge_density: np.ndarray | None = None
        position_telemetry: dict[str, Any] | None = None
        if use_source_ordered_velocity_update:
            position_telemetry, half_step_charge_density = (
                _advance_positions_from_half_step_velocities(pic, dt_s)
            )
        else:
            E_cell, B_cell = self.cell_centered_fields(state)
            pic.push_particles(E_cell, B_cell, dt=dt_s)
        boundary_telemetry: ParticleBoundaryTelemetry | None = None
        if self.particle_boundaries is not None:
            boundary_telemetry = self.particle_boundaries.apply(pic)
        n_after = _particle_count(pic)
        deposition_method = _deposition_method(pic)
        rho, Jx, Jy, Jz = pic.deposit()
        if half_step_charge_density is not None and boundary_telemetry is None:
            rho_for_density = half_step_charge_density
            density_sample = "x_n_plus_half"
        else:
            rho_for_density = rho
            density_sample = "x_n_plus_1"
        ion_current = np.stack((Jx, Jy, Jz), axis=-1)
        electron_density = np.maximum(
            np.abs(rho_for_density) / e,
            float(density_floor_m3),
        )
        sigma0_for_step: np.ndarray | float = sigma0_S_m
        apply_density_conductivity_blend = True
        source_backed_transport: PartialIonizedConductivityTelemetry | None = None
        if use_source_backed_conductivity:
            if ionization_state is None or electron_energy_state is None:
                raise ValueError(
                    "ionization_state and electron_energy_state are required "
                    "when use_source_backed_conductivity is True"
                )
            electron_density = np.maximum(ionization_state.electron_density_m3, 1.0)
            sigma0_for_step, source_backed_transport = partial_ionized_conductivity(
                electron_density_m3=electron_density,
                neutral_density_m3=ionization_state.neutral_density_m3,
                electron_temperature_K=electron_energy_state.electron_temperature_K,
            )
            apply_density_conductivity_blend = False
        kinetic_yield_telemetry: KineticYieldTelemetry | None = None
        if self.kinetic_yield_history is not None:
            kinetic_yield_telemetry = self.kinetic_yield_history.step(
                pic,
                target_density_m3=electron_density,
                dt_s=dt_s,
            )
        pressure_gradient_telemetry: dict[str, Any] | None = None
        if (
            pressure_term_V_m is None
            and self.electron_energy_closure is not None
            and electron_energy_state is not None
        ):
            electron_pressure = electron_energy_state.electron_pressure_Pa(
                electron_density
            )
            pressure_term_V_m, pressure_gradient_telemetry = (
                self.field_stepper.ohm_solver.pressure_gradient_term(
                    electron_pressure,
                    electron_density,
                    density_threshold_m3=pressure_density_threshold_m3,
                )
            )
        field_step = self.field_stepper.step(
            state,
            dt_s=dt_s,
            ion_current_A_m2=ion_current,
            electron_density_m3=electron_density,
            sigma0_S_m=sigma0_for_step,
            background_density_m3=background_density_m3,
            ohmic_cfl_safety=ohmic_cfl_safety,
            pressure_term_V_m=pressure_term_V_m,
            include_hall=include_hall,
            use_predictor_corrector=(
                use_predictor_corrector and not use_source_ordered_velocity_update
            ),
            marder_factor_m2=marder_factor_m2,
            marder_nondominance_threshold=marder_nondominance_threshold,
            charge_density_C_m3=rho,
            apply_density_conductivity_blend=apply_density_conductivity_blend,
            apply_ohmic_cfl_limit=True,
        )
        source_velocity_telemetry: dict[str, Any] | None = None
        provisional_rebuild_telemetry: dict[str, Any] | None = None
        if use_source_ordered_velocity_update:
            B_next_cell = np.stack(face_to_cell_centered(field_step.state.B), axis=-1)
            if use_predictor_corrector:
                provisional_ion_current, provisional_rebuild_telemetry = (
                    _provisional_particle_rebuild(
                        pic,
                        total_current_A_m2=field_step.total_current_A_m2,
                        magnetic_field_T=B_next_cell,
                        electron_density_m3=electron_density,
                        pressure_term_V_m=pressure_term_V_m,
                        dt_s=dt_s,
                        density_floor_m3=density_floor_m3,
                    )
                )
                E_next_cell = np.stack(
                    self.field_stepper.maxwell.edge_E_to_cell_centered(
                        field_step.state.E
                    ),
                    axis=-1,
                )
                _, corrected_current, pc_telemetry = (
                    self.field_stepper.predictor_corrector.correct_end_step_current(
                        midpoint_current_A_m2=field_step.total_current_A_m2,
                        previous_current_A_m2=(
                            self.field_stepper.previous_total_current_A_m2
                        ),
                        electric_field_next_V_m=E_next_cell,
                        magnetic_field_next_T=B_next_cell,
                        predicted_ion_current_A_m2=provisional_ion_current,
                        conductivity_S_m=field_step.conductivity_S_m,
                        electron_density_m3=electron_density,
                        pressure_term_V_m=pressure_term_V_m,
                        include_hall=include_hall,
                    )
                )
                self.field_stepper.previous_total_current_A_m2 = np.array(
                    corrected_current,
                    copy=True,
                )
                provisional_rebuild_telemetry = {
                    **provisional_rebuild_telemetry,
                    "feeds_corrected_current": True,
                }
                field_step.end_step_current_A_m2 = corrected_current
                field_step.telemetry = replace(
                    field_step.telemetry,
                    predictor_corrector=pc_telemetry.to_dict(),
                )
            source_velocity_telemetry = _apply_eq7_velocity_update(
                pic,
                total_current_A_m2=field_step.end_step_current_A_m2,
                magnetic_field_T=B_next_cell,
                electron_density_m3=electron_density,
                pressure_term_V_m=pressure_term_V_m,
                dt_s=dt_s,
                density_floor_m3=density_floor_m3,
            )
            if _collisions_enabled(pic):
                pic.apply_collisions(dt_s)
        next_electron_energy_state: ElectronEnergyState | None = None
        electron_energy_telemetry: ElectronEnergyTelemetry | None = None
        if self.electron_energy_closure is not None and electron_energy_state is not None:
            if mass_density_kg_m3 is None:
                raise ValueError(
                    "mass_density_kg_m3 is required when electron_energy_state is supplied"
                )
            if plasma_velocity_m_s is None:
                raise ValueError(
                    "plasma_velocity_m_s is required when electron_energy_state is supplied"
                )
            resistivity = np.divide(
                1.0,
                field_step.conductivity_S_m,
                out=np.zeros_like(field_step.conductivity_S_m),
                where=field_step.conductivity_S_m > 0.0,
            )
            magnetic_field_for_heat_flux = np.stack(
                face_to_cell_centered(field_step.state.B),
                axis=-1,
            )
            next_electron_energy_state, electron_energy_telemetry = (
                self.electron_energy_closure.step_sources(
                    electron_energy_state,
                    electron_density_m3=electron_density,
                    ion_density_m3=electron_density / charge_state_Z,
                    mass_density_kg_m3=mass_density_kg_m3,
                    velocity_m_s=plasma_velocity_m_s,
                    resistivity_ohm_m=resistivity,
                    current_A_m2=field_step.end_step_current_A_m2,
                    dt_s=dt_s,
                    charge_state_Z=charge_state_Z,
                    temperature_floor_K=electron_temperature_floor_K,
                    magnetic_field_T=magnetic_field_for_heat_flux,
                    heat_flux_subcycles_max=heat_flux_subcycles_max,
                )
            )
        next_ionization_state: DeuteriumIonizationState | None = None
        ionization_telemetry: IonizationTransportTelemetry | None = None
        ionization_particle_source_telemetry: (
            IonizationParticleSourceTelemetry | None
        ) = None
        if self.ionization_transport is not None and ionization_state is not None:
            temperature_state = next_electron_energy_state or electron_energy_state
            if temperature_state is None:
                raise ValueError(
                    "electron_energy_state is required when ionization_state is supplied"
                )
            next_ionization_state, ionization_telemetry = (
                self.ionization_transport.step(
                    ionization_state,
                    electron_temperature_K=temperature_state.electron_temperature_K,
                    dt_s=dt_s,
                )
            )
            ionization_particle_source_telemetry = apply_ionization_particle_source(
                pic,
                self.grid,
                previous_state=ionization_state,
                next_state=next_ionization_state,
                species_name=pic.species[0].name if pic.species else "d",
                ion_mass_kg=pic.species[0].mass if pic.species else None,
                ion_charge_C=pic.species[0].charge if pic.species else None,
                velocity_m_s=plasma_velocity_m_s,
            )
            n_after = _particle_count(pic)
        temperature_authority = extended_ohm_temperature_authority_status(
            include_hall=bool(field_step.telemetry.ohm_solver["include_hall"]),
            include_pressure=bool(field_step.telemetry.ohm_solver["include_pressure"]),
            electron_energy_evidence=electron_energy_telemetry,
        )
        neutron_yield_authority = (
            None
            if kinetic_yield_telemetry is None
            else kinetic_neutron_yield_authority_status(
                kinetic_yield_evidence=kinetic_yield_telemetry,
                temperature_authority=temperature_authority,
            )
        )
        telemetry = HybridPIC3DLoopTelemetry(
            status="candidate_engineering_particle_field_loop",
            source=HYBRID_PIC_3D_SOURCE,
            n_particles_before=n_before,
            n_particles_after=n_after,
            deposition_method=deposition_method,
            particle_boundaries=(
                None if boundary_telemetry is None else boundary_telemetry.to_dict()
            ),
            collision_operator=_collision_operator_telemetry(pic),
            pressure_gradient=pressure_gradient_telemetry,
            source_backed_transport=(
                None
                if source_backed_transport is None
                else source_backed_transport.to_dict()
            ),
            electron_energy=(
                None
                if electron_energy_telemetry is None
                else electron_energy_telemetry.to_dict()
            ),
            ionization_charge_state=(
                None
                if ionization_telemetry is None
                else {
                    **ionization_telemetry.to_dict(),
                    "particle_source": (
                        None
                        if ionization_particle_source_telemetry is None
                        else ionization_particle_source_telemetry.to_dict()
                    ),
                }
            ),
            temperature_authority=temperature_authority,
            kinetic_yield=(
                None
                if kinetic_yield_telemetry is None
                else kinetic_yield_telemetry.to_dict()
            ),
            neutron_yield_authority=neutron_yield_authority,
            source_workflow=_source_workflow_telemetry(
                use_source_ordered_velocity_update=use_source_ordered_velocity_update,
                position_telemetry=position_telemetry,
                velocity_telemetry=source_velocity_telemetry,
                provisional_rebuild_telemetry=provisional_rebuild_telemetry,
                electron_energy_telemetry=electron_energy_telemetry,
                density_sample=density_sample,
                collisions_enabled=_collisions_enabled(pic),
            ),
            electron_density_min_m3=float(np.min(electron_density)),
            electron_density_max_m3=float(np.max(electron_density)),
            ion_current_max_A_m2=float(np.max(np.linalg.norm(ion_current, axis=-1))),
            field_step=field_step.telemetry.to_dict(),
        )
        return HybridPIC3DLoopResult(
            state=field_step.state,
            rho_C_m3=rho,
            ion_current_A_m2=ion_current,
            electron_density_m3=electron_density,
            electron_energy=next_electron_energy_state,
            ionization_charge_state=next_ionization_state,
            field_step=field_step,
            telemetry=telemetry,
        )


def hybrid_loop_candidate_evidence(
    telemetry: HybridPIC3DLoopTelemetry,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build non-promoting evidence for one particle-field loop step."""
    return {
        "passed": telemetry.status == "candidate_engineering_particle_field_loop",
        "status": "candidate",
        "capability": "kinetic_ion_pic_push_deposition",
        "source": telemetry.source,
        "implementation": "src/dpf/fields/hybrid_loop.py",
        "evidence_type": "engineering_particle_field_loop_step",
        "deposition_method": telemetry.deposition_method,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Conductor/PML particle absorption is candidate-only when configured.",
            "Electron density is quasi-neutral rebuilt from deposited ion charge.",
            "Separate electron-energy coupling is optional candidate telemetry, not accepted Te authority.",
            "Charge-state transport and PIC particle source/sink coupling are candidate telemetry only.",
            "Source-backed partial-ionized conductivity is candidate-only when enabled.",
            "Kinetic ion yield history is optional candidate telemetry, not accepted neutron authority.",
            "Source-ordered Eq. 7 velocity update is candidate-only when enabled.",
            "No accepted DPF validation or detector packet is supplied.",
        ],
    }


def source_ordered_loop_candidate_evidence(
    telemetry: HybridPIC3DLoopTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for source-ordered loop sequencing."""
    workflow = telemetry.source_workflow
    return {
        "passed": workflow.get("status")
        == "candidate_engineering_source_ordered_loop",
        "status": "candidate",
        "capability": "source_ordered_time_loop",
        "source": telemetry.source,
        "source_lines": "246-315, 428-535",
        "implementation": "src/dpf/fields/hybrid_loop.py",
        "evidence_type": "engineering_source_ordered_loop_sequence",
        "stages_executed": workflow.get("stages_executed", []),
        "predictor_particle_rebuild": workflow.get("predictor_particle_rebuild"),
        "review_required_stages": workflow.get("review_required_stages", []),
        "acceptance_blocking_stages": workflow.get("acceptance_blocking_stages", []),
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Eq. 7 velocity update is exercised, but temperature source terms still require same-scope review.",
            "Predictor-corrector current rebuild is runtime telemetry, not reviewed acceptance evidence.",
            "No long-run DPF stability, nondominance, or same-scope validation packet is attached.",
        ],
    }


def ion_collision_loop_candidate_evidence(
    telemetry: HybridPIC3DLoopTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for ion-collision use in the loop."""
    collision = telemetry.collision_operator
    return {
        "passed": bool(collision.get("enabled")),
        "status": "candidate",
        "capability": "ion_collision_operator",
        "source": telemetry.source,
        "source_lines": "310-311",
        "implementation": "src/dpf/experimental/pic/hybrid.py via src/dpf/fields/hybrid_loop.py",
        "evidence_type": "engineering_loop_collision_telemetry",
        "algorithm": collision.get("algorithm"),
        "enabled": collision.get("enabled"),
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Collision parameters are configured as engineering inputs, not a reviewed DPF collision packet.",
            "Loop telemetry does not yet prove cell-local Nanbu pairing against same-scope DPF distributions.",
            "No same-scope 3D validation packet is attached.",
        ],
    }


def _particle_count(pic: HybridPIC) -> int:
    return int(sum(species.n_particles() for species in pic.species))


def _deposition_method(pic: HybridPIC) -> str:
    if not pic.use_esirkepov:
        return "cic_current"
    for species in pic.species:
        if species.n_particles() == 0:
            continue
        if not np.array_equal(species.positions_old, species.positions):
            return "esirkepov"
    return "cic_current_no_prior_push"


def _collision_operator_telemetry(pic: HybridPIC) -> dict[str, Any]:
    enabled = _collisions_enabled(pic)
    use_binary = bool(getattr(pic, "use_binary_collisions", False))
    if not enabled:
        algorithm = "disabled"
        status = "disabled"
    elif use_binary:
        algorithm = "nanbu_perez_binary"
        status = "candidate_enabled"
    else:
        algorithm = "takizuka_abe_fallback"
        status = "candidate_enabled"
    return {
        "status": status,
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "310-311",
        "enabled": enabled,
        "algorithm": algorithm,
        "n_background_m3": float(getattr(pic, "_n_background", 0.0)),
        "T_background_eV": float(getattr(pic, "_T_background_eV", 0.0)),
        "can_support_first_principles_acceptance": False,
    }


def _collisions_enabled(pic: HybridPIC) -> bool:
    return bool(getattr(pic, "_collision_enabled", False))


def _advance_positions_from_half_step_velocities(
    pic: HybridPIC,
    dt_s: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Advance x_n to x_{n+1} from stored half-step velocities."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    rho_half = np.zeros(pic.grid_shape, dtype=float)
    max_displacement = 0.0
    for species in pic.species:
        if species.n_particles() == 0:
            continue
        old_positions = species.positions.copy()
        half_positions = old_positions + 0.5 * dt_s * species.velocities
        full_positions = old_positions + dt_s * species.velocities
        full_positions, reflected_velocities = pic._apply_reflecting_bc(
            full_positions,
            species.velocities.copy(),
        )
        half_positions, _ = pic._apply_reflecting_bc(
            half_positions,
            species.velocities.copy(),
        )
        species.positions_old = old_positions
        species.positions = full_positions
        species.velocities = reflected_velocities
        rho_half += species.charge * deposit_density(
            half_positions,
            species.weights,
            pic.grid_shape,
            pic.dx,
            pic.dy,
            pic.dz,
        )
        if old_positions.size:
            max_displacement = max(
                max_displacement,
                float(np.max(np.linalg.norm(full_positions - old_positions, axis=1))),
            )
    return (
        {
            "status": "candidate_engineering_leapfrog_position_update",
            "source": HYBRID_PIC_3D_SOURCE,
            "source_lines": "246-261",
            "max_displacement_m": max_displacement,
            "half_step_charge_density_available": True,
            "can_support_first_principles_acceptance": False,
        },
        rho_half,
    )


def _apply_eq7_velocity_update(
    pic: HybridPIC,
    *,
    total_current_A_m2: np.ndarray,
    magnetic_field_T: np.ndarray,
    electron_density_m3: np.ndarray,
    pressure_term_V_m: np.ndarray | None,
    dt_s: float,
    density_floor_m3: float,
) -> dict[str, Any]:
    """Update ion velocities from source Eq. 7, m dv/dt = (JxB-grad pe)/ne."""
    density = np.maximum(electron_density_m3, density_floor_m3)
    grad_pe_N_m3 = (
        np.zeros_like(total_current_A_m2)
        if pressure_term_V_m is None
        else e * density[..., np.newaxis] * pressure_term_V_m
    )
    force_density_N_m3 = np.cross(total_current_A_m2, magnetic_field_T) - grad_pe_N_m3
    max_delta_v = 0.0
    updated_particles = 0
    for species in pic.species:
        if species.n_particles() == 0:
            continue
        acceleration_field = force_density_N_m3 / (
            density[..., np.newaxis] * species.mass
        )
        acceleration_at_particles = interpolate_field_to_particles(
            acceleration_field,
            species.positions,
            pic.dx,
            pic.dy,
            pic.dz,
        )
        delta_v = dt_s * acceleration_at_particles
        species.velocities = species.velocities + delta_v
        updated_particles += species.n_particles()
        if delta_v.size:
            max_delta_v = max(
                max_delta_v,
                float(np.max(np.linalg.norm(delta_v, axis=1))),
            )
    return {
        "status": "candidate_engineering_eq7_velocity_update",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "224-236, 286-294",
        "updated_particles": updated_particles,
        "max_delta_v_m_s": max_delta_v,
        "pressure_term_included": pressure_term_V_m is not None,
        "can_support_first_principles_acceptance": False,
    }


def _provisional_particle_rebuild(
    pic: HybridPIC,
    *,
    total_current_A_m2: np.ndarray,
    magnetic_field_T: np.ndarray,
    electron_density_m3: np.ndarray,
    pressure_term_V_m: np.ndarray | None,
    dt_s: float,
    density_floor_m3: float,
) -> dict[str, Any]:
    """Estimate provisional ion velocities/current for source predictor-corrector."""
    density = np.maximum(electron_density_m3, density_floor_m3)
    grad_pe_N_m3 = (
        np.zeros_like(total_current_A_m2)
        if pressure_term_V_m is None
        else e * density[..., np.newaxis] * pressure_term_V_m
    )
    force_density_N_m3 = np.cross(total_current_A_m2, magnetic_field_T) - grad_pe_N_m3
    Jx = np.zeros(pic.grid_shape, dtype=float)
    Jy = np.zeros(pic.grid_shape, dtype=float)
    Jz = np.zeros(pic.grid_shape, dtype=float)
    max_delta_v = 0.0
    n_particles = 0
    for species in pic.species:
        if species.n_particles() == 0:
            continue
        acceleration_field = force_density_N_m3 / (
            density[..., np.newaxis] * species.mass
        )
        acceleration_at_particles = interpolate_field_to_particles(
            acceleration_field,
            species.positions,
            pic.dx,
            pic.dy,
            pic.dz,
        )
        delta_v = dt_s * acceleration_at_particles
        provisional_velocity = species.velocities + delta_v
        jx, jy, jz = deposit_current(
            species.positions,
            provisional_velocity,
            species.weights,
            species.charge,
            pic.grid_shape,
            pic.dx,
            pic.dy,
            pic.dz,
        )
        Jx += jx
        Jy += jy
        Jz += jz
        n_particles += species.n_particles()
        if delta_v.size:
            max_delta_v = max(
                max_delta_v,
                float(np.max(np.linalg.norm(delta_v, axis=1))),
            )
    provisional_current = np.stack((Jx, Jy, Jz), axis=-1)
    return provisional_current, {
        "status": "candidate_engineering_predictor_particle_rebuild",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "428-535",
        "n_particles": n_particles,
        "max_delta_v_m_s": max_delta_v,
        "max_provisional_ion_current_A_m2": float(
            np.max(np.linalg.norm(provisional_current, axis=-1))
        ),
        "feeds_corrected_current": False,
        "can_support_first_principles_acceptance": False,
    }


def _source_workflow_telemetry(
    *,
    use_source_ordered_velocity_update: bool,
    position_telemetry: dict[str, Any] | None,
    velocity_telemetry: dict[str, Any] | None,
    provisional_rebuild_telemetry: dict[str, Any] | None,
    electron_energy_telemetry: ElectronEnergyTelemetry | None,
    density_sample: str,
    collisions_enabled: bool,
) -> dict[str, Any]:
    if not use_source_ordered_velocity_update:
        return {
            "status": "unsupported_boris_push_before_field_solve_sequence",
            "source": HYBRID_PIC_3D_SOURCE,
            "source_lines": "246-315",
            "stages_executed": [
                "boris_push_positions_and_velocities",
                "deposit_charge_current",
                "ohm_ampere_field_step",
            ],
            "required_source_ordered_stages": [
                "source_position_only_leapfrog",
                "eq7_end_step_velocity_update",
                "collisions_after_velocity_update",
                "reviewed_predictor_corrector_particle_rebuild",
            ],
            "can_support_first_principles_acceptance": False,
        }
    predictor_status = (
        "runtime_predictor_particle_rebuild_executed_review_required"
        if provisional_rebuild_telemetry is not None
        else "predictor_particle_rebuild_not_requested"
    )
    temperature_status = (
        "operator_split_temperature_source_terms_review_required"
        if electron_energy_telemetry is not None
        else "temperature_source_terms_not_configured"
    )
    return {
        "status": "candidate_engineering_source_ordered_loop",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "246-315, 428-535",
        "stages_executed": [
            "leapfrog_position_update_xn_to_xn_plus_1",
            "deposit_current_from_xn_to_xn_plus_1",
            f"electron_density_from_{density_sample}",
            "generalized_ohm_ampere_solve",
            "fdtd_maxwell_update",
            "optional_marder_correction",
            "optional_predictor_corrector_current",
            predictor_status,
            "eq7_velocity_update_v_half_to_v_three_half",
            "collisions_after_velocity_update"
            if collisions_enabled
            else "collisions_disabled",
            temperature_status,
        ],
        "position_update": position_telemetry,
        "velocity_update": velocity_telemetry,
        "predictor_particle_rebuild": provisional_rebuild_telemetry,
        "temperature_source_terms": (
            None if electron_energy_telemetry is None else electron_energy_telemetry.to_dict()
        ),
        "review_required_stages": [
            temperature_status,
            predictor_status,
            "long_run_source_ordered_stability_review_required",
            "same_scope_3d_review_required",
        ],
        "acceptance_blocking_stages": [
            "temperature_source_terms_same_scope_review",
            "predictor_corrector_particle_rebuild_review",
            "long_run_source_ordered_stability",
            "same_scope_3d_validation",
        ],
        "can_support_first_principles_acceptance": False,
    }
