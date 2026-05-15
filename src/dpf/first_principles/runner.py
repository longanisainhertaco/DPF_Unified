"""Package-native 3-D hybrid EM/PIC-fluid first-principles runner.

This module wires the existing ``dpf.fields`` 3-D Maxwell, HybridPIC loop,
electron-energy, kinetic-yield, and optional circuit-boundary components into
one minimal engineering-candidate run.  It is deliberately fail-closed:
results are marked as engineering candidates and cannot be used as validation
evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf import constants as dpf_constants
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields import (
    CircuitMagneticBoundaryDrive,
    CircuitParameters,
    CircuitState,
    ElectronEnergyClosure,
    ElectronEnergyState,
    HybridPIC3DLoop,
    HybridPIC3DSimulationResult,
    HybridPIC3DSimulator,
    HybridPICSourceGeometry,
    KineticIonYieldHistory,
    Maxwell3DGrid,
    hybrid_loop_candidate_evidence,
    hybrid_simulator_candidate_evidence,
    source_geometry_candidate_evidence,
    source_ordered_loop_candidate_evidence,
)
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE
from dpf.first_principles.certificate_gate import (
    build_first_principles_certificate_gate_packet,
)
from dpf.first_principles.closure_packet import build_physics_closure_packet
from dpf.first_principles.comparator_uq import build_comparator_uq_packet
from dpf.first_principles.deck import FirstPrinciplesInputDeck
from dpf.first_principles.dimensionality import build_dimensionality_handoff_packet
from dpf.first_principles.generalization import build_generalized_dpf_machine_packet
from dpf.first_principles.limiter_readiness import build_limiter_readiness_packet
from dpf.first_principles.manifest import (
    build_first_principles_manifest_from_hybrid_result,
)
from dpf.first_principles.neutron_authority import (
    build_mechanism_separated_neutron_packet,
)
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet
from dpf.first_principles.power_port import build_engineering_power_port_packet
from dpf.first_principles.same_scope import build_same_scope_source_packet
from dpf.first_principles.spatial_field_temperature import (
    build_spatial_field_temperature_packet,
)
from dpf.first_principles.startup_bvp import build_startup_bvp_packet
from dpf.first_principles.waveform_phase import build_waveform_phase_packet

ENGINEERING_CANDIDATE_STATUS = "engineering_candidate_not_validation"
RUN_MODE = "first_principles_3d_hybrid_em_pic_fluid"
ELEMENTARY_CHARGE = dpf_constants.e
DEUTERON_MASS_KG = dpf_constants.m_d


@dataclass(frozen=True)
class FirstPrinciples3DDeck:
    """Minimal input deck for a package-native 3-D hybrid EM/PIC-fluid run."""

    n_steps: int = 1
    grid_shape: tuple[int, int, int] = (5, 5, 5)
    grid_spacing_m: tuple[float, float, float] | None = None
    dt_s: float = 1.0e-13
    sigma0_S_m: float = 1.0e2
    background_density_m3: float = 1.0e20
    density_floor_m3: float = 1.0e20
    electron_temperature_K: float = 1.0e5
    ion_temperature_K: float = 1.0e5
    ion_species_name: str = "d"
    ion_mass_kg: float = DEUTERON_MASS_KG
    ion_charge_C: float = ELEMENTARY_CHARGE
    particle_weight: float = 1.0e8
    initial_E_x_V_m: float = 1.0e5
    initial_B_z_T: float = 0.0
    include_hall: bool = False
    use_predictor_corrector: bool = True
    use_source_ordered_velocity_update: bool = True
    marder_factor_scale: float = 1.0e-6
    marder_nondominance_threshold: float = 0.5
    ohmic_cfl_safety: float = 1.0
    apply_circuit_boundary: bool = True
    circuit_capacitance_F: float = 2.0e-5
    circuit_voltage_V: float = 1.5e4
    circuit_inductance_H: float = 1.1e-7
    circuit_resistance_ohm: float = 1.2e-2
    circuit_state: CircuitState | None = None
    circuit_udpf_V: float | tuple[float, ...] = 0.0
    circuit_z_index: int = 0
    circuit_blend: float = 1.0
    device_anode_radius_m: float | None = None
    device_cathode_radius_m: float | None = None
    device_anode_length_m: float | None = None
    device_insulator_length_m: float | None = None
    gas_pressure_Pa: float | None = None
    gas_temperature_K: float | None = None
    startup_mode: str = "source_backed_end_rundown_sheath"
    startup_evidence_status: str = "engineering_candidate_not_whole_shot"
    startup_source_scope: str = "end_of_rundown_or_engineering_startup"
    startup_can_support_whole_shot_acceptance: bool = False
    startup_accepted_channels: tuple[str, ...] = ()
    startup_required_channels: tuple[str, ...] = ()
    startup_missing_channels: tuple[str, ...] = ()
    device_name: str = "not_declared"
    validation_scope: str = "not_declared_engineering_smoke"
    validation_targets: tuple[dict[str, Any], ...] = ()
    limiter_readiness_accepted_channels: tuple[str, ...] = ()
    same_scope_accepted_channels: tuple[str, ...] = ()
    waveform_phase_accepted_channels: tuple[str, ...] = ()
    spatial_field_temperature_accepted_channels: tuple[str, ...] = ()
    neutron_authority_accepted_channels: tuple[str, ...] = ()
    comparator_uq_accepted_channels: tuple[str, ...] = ()
    numerical_fidelity_accepted_channels: tuple[str, ...] = ()
    certificate_accepted_channels: tuple[str, ...] = ()
    generalization_accepted_channels: tuple[str, ...] = ()
    seed: int | None = 0

    @classmethod
    def from_deck(
        cls,
        deck: Mapping[str, Any] | object | None = None,
        *,
        n_steps: int | None = None,
    ) -> FirstPrinciples3DDeck:
        if isinstance(deck, cls):
            values = asdict(deck)
        elif deck is None:
            values = {}
        elif isinstance(deck, FirstPrinciplesInputDeck):
            values = _values_from_package_deck(deck)
        elif isinstance(deck, Mapping):
            if {"device", "circuit", "grid"}.issubset(deck.keys()):
                values = _values_from_package_deck(
                    FirstPrinciplesInputDeck.from_mapping(dict(deck))
                )
            else:
                values = dict(deck)
        else:
            values = {
                field: getattr(deck, field)
                for field in cls.__dataclass_fields__
                if hasattr(deck, field)
            }
            if not values and all(hasattr(deck, name) for name in ("grid", "circuit")):
                values = _values_from_package_deck(deck)

        grid = values.pop("grid", None) or values.pop("maxwell_grid", None)
        geometry = values.pop("geometry", None)
        if grid is not None:
            values["grid_shape"] = tuple(int(v) for v in grid.shape)
            values["grid_spacing_m"] = tuple(float(v) for v in grid.spacing)
        elif values.get("grid_spacing_m") is None and values.get("grid_spacing") is not None:
            values["grid_spacing_m"] = values.pop("grid_spacing")
        elif values.get("spacing") is not None:
            values["grid_spacing_m"] = values.pop("spacing")

        if values.get("shape") is not None:
            values["grid_shape"] = values.pop("shape")

        if geometry is not None and grid is None and values.get("grid_spacing_m") is None:
            grid_from_geometry = geometry.smoke_grid(shape=tuple(values.get("grid_shape", cls.grid_shape)))
            values["grid_shape"] = grid_from_geometry.shape
            values["grid_spacing_m"] = grid_from_geometry.spacing

        if n_steps is not None:
            values["n_steps"] = n_steps
        if values.get("circuit_state") is not None and not isinstance(
            values["circuit_state"], CircuitState
        ):
            state = values["circuit_state"]
            values["circuit_state"] = CircuitState(
                current_A=float(_get(state, "current_A", CircuitState().current_A)),
                charge_C=float(_get(state, "charge_C", CircuitState().charge_C)),
            )
        return cls(**_coerce_deck_values(values))

    def grid(self) -> Maxwell3DGrid:
        if self.grid_spacing_m is not None:
            return Maxwell3DGrid(
                shape=tuple(int(v) for v in self.grid_shape),
                spacing=tuple(float(v) for v in self.grid_spacing_m),
            )
        return HybridPICSourceGeometry().smoke_grid(shape=self.grid_shape)

    def manifest_config(self) -> dict[str, Any]:
        return {
            "run_mode": RUN_MODE,
            "geometry": {
                "device_name": self.device_name,
                "type": "cartesian_3d_hybrid_pic",
                "grid_shape": list(self.grid_shape),
                "grid_spacing_m": (
                    None if self.grid_spacing_m is None else list(self.grid_spacing_m)
                ),
            },
            "fluid": {
                "backend": "hybrid",
                "precision": "float64",
                "sigma0_S_m": self.sigma0_S_m,
                "background_density_m3": self.background_density_m3,
                "density_floor_m3": self.density_floor_m3,
            },
            "diagnostics": {
                "artifact_classification": ENGINEERING_CANDIDATE_STATUS,
                "artifact_distribution": "local_engineering",
                "artifact_handling_notes": "candidate 3D hybrid EM/PIC-fluid run",
            },
            "first_principles_3d": {
                "n_steps": self.n_steps,
                "dt_s": self.dt_s,
                "apply_circuit_boundary": self.apply_circuit_boundary,
                "include_hall": self.include_hall,
                "use_predictor_corrector": self.use_predictor_corrector,
                "use_source_ordered_velocity_update": (
                    self.use_source_ordered_velocity_update
                ),
                "reduced_models_used": False,
            },
            "startup": self.startup_packet(),
            "limiter_readiness": {
                "accepted_channels": list(self.limiter_readiness_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "same_scope_source": {
                "declared_scope": self.validation_scope,
                "accepted_same_scope_channels": list(self.same_scope_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "waveform_phase": {
                "accepted_channels": list(self.waveform_phase_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "spatial_field_temperature": {
                "accepted_channels": list(
                    self.spatial_field_temperature_accepted_channels
                ),
                "validation_target_count": len(self.validation_targets),
            },
            "neutron_authority": {
                "accepted_channels": list(self.neutron_authority_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "comparator_uq": {
                "accepted_channels": list(self.comparator_uq_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "numerical_fidelity": {
                "accepted_channels": list(self.numerical_fidelity_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "certificate_gate": {
                "accepted_channels": list(self.certificate_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "generalization": {
                "accepted_channels": list(self.generalization_accepted_channels),
                "validation_target_count": len(self.validation_targets),
            },
            "circuit": {
                "capacitance_F": self.circuit_capacitance_F,
                "voltage_V": self.circuit_voltage_V,
                "inductance_H": self.circuit_inductance_H,
                "resistance_ohm": self.circuit_resistance_ohm,
            },
        }

    def startup_packet(self) -> dict[str, Any]:
        packet = build_startup_bvp_packet(
            {
                "mode": self.startup_mode,
                "evidence_status": self.startup_evidence_status,
                "source_scope": self.startup_source_scope,
                "can_support_whole_shot_acceptance": (
                    self.startup_can_support_whole_shot_acceptance
                ),
                "accepted_channels": list(self.startup_accepted_channels),
                "required_channels": list(self.startup_required_channels),
                "missing_channels": list(self.startup_missing_channels),
                "background_density_m3": self.background_density_m3,
                "electron_temperature_K": self.electron_temperature_K,
                "ion_temperature_K": self.ion_temperature_K,
                "initial_electric_field_V_m": (
                    self.initial_E_x_V_m,
                    0.0,
                    0.0,
                ),
                "initial_magnetic_field_T": (
                    0.0,
                    0.0,
                    self.initial_B_z_T,
                ),
            },
            device={
                "device_name": self.device_name,
                "anode_radius_m": self.device_anode_radius_m,
                "cathode_radius_m": self.device_cathode_radius_m,
                "anode_length_m": self.device_anode_length_m,
                "insulator_length_m": self.device_insulator_length_m,
            },
            gas={
                "species": self.ion_species_name,
                "pressure_Pa": self.gas_pressure_Pa,
                "temperature_K": self.gas_temperature_K,
            },
            circuit={
                "voltage_V": self.circuit_voltage_V,
                "initial_current_A": (
                    None if self.circuit_state is None else self.circuit_state.current_A
                ),
                "charge_C": (
                    None if self.circuit_state is None else self.circuit_state.charge_C
                ),
            },
            accepted_channels=self.startup_accepted_channels,
        )
        packet["declared_startup_required_channels"] = list(
            self.startup_required_channels
        )
        packet["declared_startup_missing_channels"] = list(
            self.startup_missing_channels
        )
        return packet


@dataclass
class HybridEMPicFluidRunResult:
    """Return object for the package-native engineering-candidate runner."""

    status: str
    result: HybridPIC3DSimulationResult
    manifest: dict[str, Any]
    conservation_telemetry: dict[str, Any]
    validation_packet: dict[str, Any]
    telemetry: dict[str, Any]
    reduced_models_used: bool = False
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_mode": RUN_MODE,
            "scientific_status": self.status,
            "reduced_models_used": self.reduced_models_used,
            "can_support_first_principles_acceptance": (
                self.can_support_first_principles_acceptance
            ),
            "manifest": self.manifest,
            "conservation_telemetry": self.conservation_telemetry,
            "validation_packet": self.validation_packet,
            "telemetry": self.telemetry,
        }


class HybridEMPicFluidRun:
    """Orchestrate a minimal whole-shot-candidate 3-D hybrid EM/PIC-fluid run."""

    def __init__(
        self,
        deck: Mapping[str, Any] | object | None = None,
        *,
        circuit_boundary: CircuitMagneticBoundaryDrive | None = None,
    ) -> None:
        self.deck = FirstPrinciples3DDeck.from_deck(deck)
        self.circuit_boundary = circuit_boundary

    def run(self, *, n_steps: int | None = None) -> HybridEMPicFluidRunResult:
        deck = FirstPrinciples3DDeck.from_deck(self.deck, n_steps=n_steps)
        _validate_deck(deck)
        if deck.seed is not None:
            np.random.seed(int(deck.seed))

        geometry = HybridPICSourceGeometry()
        grid = deck.grid()
        electron_density = np.full(grid.shape, deck.background_density_m3, dtype=float)
        electron_closure = ElectronEnergyClosure(grid)
        electron_state = electron_closure.initialize(
            electron_temperature_K=deck.electron_temperature_K,
            ion_temperature_K=deck.ion_temperature_K,
            electron_density_m3=electron_density,
        )
        loop = HybridPIC3DLoop(
            grid,
            electron_energy_closure=electron_closure,
            kinetic_yield_history=KineticIonYieldHistory(grid),
        )
        state = loop.field_stepper.maxwell.empty_state()
        state.E.Ex_edge.fill(deck.initial_E_x_V_m)
        state.B.Bz_face.fill(deck.initial_B_z_T)
        pic = _build_minimal_pic(deck, grid)
        initial_circuit_state = deck.circuit_state or CircuitState()
        circuit_boundary = self.circuit_boundary
        if deck.apply_circuit_boundary and circuit_boundary is None:
            circuit_boundary = CircuitMagneticBoundaryDrive(
                grid,
                parameters=CircuitParameters(
                    capacitance_F=deck.circuit_capacitance_F,
                    voltage_V=deck.circuit_voltage_V,
                    inductance_H=deck.circuit_inductance_H,
                    resistance_ohm=deck.circuit_resistance_ohm,
                ),
            )

        simulator = HybridPIC3DSimulator(
            grid=grid,
            loop=loop,
            state=state,
            pic=pic,
            circuit_boundary=circuit_boundary,
        )
        initial_energy = _energy_snapshot(
            loop=loop,
            state=state,
            pic=pic,
            electron_state=electron_state,
            circuit_boundary=circuit_boundary,
            circuit_state=initial_circuit_state if deck.apply_circuit_boundary else None,
        )
        simulation = simulator.run(
            n_steps=deck.n_steps,
            dt_s=deck.dt_s,
            sigma0_S_m=deck.sigma0_S_m,
            background_density_m3=deck.background_density_m3,
            ohmic_cfl_safety=deck.ohmic_cfl_safety,
            density_floor_m3=deck.density_floor_m3,
            include_hall=deck.include_hall,
            use_predictor_corrector=deck.use_predictor_corrector,
            marder_factor_m2=deck.marder_factor_scale * min(grid.spacing) ** 2,
            marder_nondominance_threshold=deck.marder_nondominance_threshold,
            electron_energy_state=electron_state,
            mass_density_kg_m3=electron_density * deck.ion_mass_kg,
            plasma_velocity_m_s=np.zeros(grid.shape + (3,), dtype=float),
            electron_temperature_floor_K=10.0,
            use_source_ordered_velocity_update=deck.use_source_ordered_velocity_update,
            circuit_state=initial_circuit_state,
            apply_circuit_boundary=deck.apply_circuit_boundary,
            circuit_udpf_V=deck.circuit_udpf_V,
            circuit_z_index=deck.circuit_z_index,
            circuit_blend=deck.circuit_blend,
        )
        final_energy = _energy_snapshot(
            loop=loop,
            state=simulation.state,
            pic=pic,
            electron_state=simulation.electron_energy,
            circuit_boundary=circuit_boundary,
            circuit_state=simulation.circuit,
        )
        conservation = _conservation_telemetry(
            grid=grid,
            n_steps=deck.n_steps,
            dt_s=deck.dt_s,
            initial=initial_energy,
            final=final_energy,
            final_diagnostics=loop.field_stepper.maxwell.diagnostics(simulation.state).to_dict(),
        )
        evidence = _candidate_evidence(
            geometry=geometry,
            simulation=simulation,
            conservation=conservation,
        )
        startup_packet = deck.startup_packet()
        power_port_packet = build_engineering_power_port_packet(
            simulation.telemetry.circuit,
            startup=startup_packet,
            conservation=conservation,
        )
        simulation_telemetry = simulation.telemetry.to_dict()
        limiter_readiness_packet = build_limiter_readiness_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            accepted_channels=deck.limiter_readiness_accepted_channels,
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
        )
        dimensionality_packet = build_dimensionality_handoff_packet(
            grid_shape=grid.shape,
            run_mode=RUN_MODE,
            startup=startup_packet,
            power_port=power_port_packet,
            simulation_telemetry=simulation_telemetry,
        )
        last_step = simulation.telemetry.last_step or {}
        physics_closure_packet = build_physics_closure_packet(
            include_hall=deck.include_hall,
            electron_energy_present=simulation.electron_energy is not None,
            kinetic_yield_present=_last_step_has_key(last_step, "kinetic_yield"),
            collisions_enabled=_collisions_enabled_in_telemetry(last_step),
            dimensionality=dimensionality_packet,
        )
        numerical_fidelity_packet = build_numerical_fidelity_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            accepted_channels=deck.numerical_fidelity_accepted_channels,
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
            upstream_packets={
                "startup_bvp": startup_packet,
                "limiter_readiness": limiter_readiness_packet,
                "power_port": power_port_packet,
                "dimensionality_handoff": dimensionality_packet,
                "physics_closure": physics_closure_packet,
            },
        )
        same_scope_source_packet = build_same_scope_source_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            validation_targets=deck.validation_targets,
            accepted_same_scope_channels=deck.same_scope_accepted_channels,
        )
        waveform_phase_packet = build_waveform_phase_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            validation_targets=deck.validation_targets,
            accepted_channels=deck.waveform_phase_accepted_channels,
            same_scope_source=same_scope_source_packet,
        )
        spatial_field_temperature_packet = build_spatial_field_temperature_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            validation_targets=deck.validation_targets,
            accepted_channels=deck.spatial_field_temperature_accepted_channels,
            same_scope_source=same_scope_source_packet,
        )
        kinetic_yield = last_step.get("kinetic_yield")
        if not isinstance(kinetic_yield, Mapping):
            kinetic_yield = None
        neutron_authority_packet = build_mechanism_separated_neutron_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            validation_targets=deck.validation_targets,
            accepted_channels=deck.neutron_authority_accepted_channels,
            kinetic_yield=kinetic_yield,
            same_scope_source=same_scope_source_packet,
            physics_closure=physics_closure_packet,
        )
        comparator_uq_packet = build_comparator_uq_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            validation_targets=deck.validation_targets,
            accepted_channels=deck.comparator_uq_accepted_channels,
            upstream_packets={
                "startup_bvp": startup_packet,
                "limiter_readiness": limiter_readiness_packet,
                "power_port": power_port_packet,
                "dimensionality_handoff": dimensionality_packet,
                "physics_closure": physics_closure_packet,
                "same_scope_source": same_scope_source_packet,
                "waveform_phase": waveform_phase_packet,
                "spatial_field_temperature": spatial_field_temperature_packet,
                "neutron_authority": neutron_authority_packet,
                "numerical_fidelity": numerical_fidelity_packet,
            },
        )
        certificate_gate_packet = build_first_principles_certificate_gate_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            accepted_channels=deck.certificate_accepted_channels,
            upstream_packets={
                "startup_bvp": startup_packet,
                "limiter_readiness": limiter_readiness_packet,
                "power_port": power_port_packet,
                "dimensionality_handoff": dimensionality_packet,
                "physics_closure": physics_closure_packet,
                "same_scope_source": same_scope_source_packet,
                "waveform_phase": waveform_phase_packet,
                "spatial_field_temperature": spatial_field_temperature_packet,
                "neutron_authority": neutron_authority_packet,
                "comparator_uq": comparator_uq_packet,
                "numerical_fidelity": numerical_fidelity_packet,
            },
        )
        generalization_packet = build_generalized_dpf_machine_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            accepted_channels=deck.generalization_accepted_channels,
            upstream_packets={
                "startup_bvp": startup_packet,
                "limiter_readiness": limiter_readiness_packet,
                "power_port": power_port_packet,
                "dimensionality_handoff": dimensionality_packet,
                "physics_closure": physics_closure_packet,
                "same_scope_source": same_scope_source_packet,
                "waveform_phase": waveform_phase_packet,
                "spatial_field_temperature": spatial_field_temperature_packet,
                "neutron_authority": neutron_authority_packet,
                "comparator_uq": comparator_uq_packet,
                "numerical_fidelity": numerical_fidelity_packet,
                "certificate_gate": certificate_gate_packet,
            },
        )
        validation_packet = _first_principles_candidate_packet(
            geometry_dimensionality="cartesian_3d",
            source_scope=geometry.source_scope,
            hybrid_pic_3d_evidence=evidence,
            conservation_evidence=conservation,
            startup_bvp=startup_packet,
            limiter_readiness=limiter_readiness_packet,
            dimensionality_handoff=dimensionality_packet,
            same_scope_source=same_scope_source_packet,
            waveform_phase=waveform_phase_packet,
            spatial_field_temperature=spatial_field_temperature_packet,
            neutron_authority=neutron_authority_packet,
            comparator_uq=comparator_uq_packet,
            numerical_fidelity=numerical_fidelity_packet,
            certificate_gate=certificate_gate_packet,
            generalization=generalization_packet,
        )
        telemetry = {
            "status": ENGINEERING_CANDIDATE_STATUS,
            "run_mode": RUN_MODE,
            "source": HYBRID_PIC_3D_SOURCE,
            "source_scope": geometry.source_scope,
            "startup": startup_packet,
            "limiter_readiness": limiter_readiness_packet,
            "power_port": power_port_packet,
            "dimensionality_handoff": dimensionality_packet,
            "physics_closure": physics_closure_packet,
            "same_scope_source": same_scope_source_packet,
            "waveform_phase": waveform_phase_packet,
            "spatial_field_temperature": spatial_field_temperature_packet,
            "neutron_authority": neutron_authority_packet,
            "comparator_uq": comparator_uq_packet,
            "numerical_fidelity": numerical_fidelity_packet,
            "certificate_gate": certificate_gate_packet,
            "generalization": generalization_packet,
            "grid_shape": list(grid.shape),
            "grid_spacing_m": list(grid.spacing),
            "n_steps": deck.n_steps,
            "dt_s": deck.dt_s,
            "simulation": simulation.telemetry.to_dict(),
            "candidate_evidence": evidence,
            "reduced_models_used": False,
            "can_support_first_principles_acceptance": False,
        }
        manifest = _build_manifest(
            deck=deck,
            grid=grid,
            simulation=simulation,
            telemetry=telemetry,
            conservation=conservation,
            validation_packet=validation_packet,
        )
        return HybridEMPicFluidRunResult(
            status=ENGINEERING_CANDIDATE_STATUS,
            result=simulation,
            manifest=manifest,
            conservation_telemetry=conservation,
            validation_packet=validation_packet,
            telemetry=telemetry,
        )


def run_first_principles_3d_deck(
    deck: Mapping[str, Any] | object | None = None,
    *,
    n_steps: int | None = None,
    circuit_boundary: CircuitMagneticBoundaryDrive | None = None,
) -> HybridEMPicFluidRunResult:
    """Execute a minimal package-native 3-D hybrid EM/PIC-fluid candidate deck."""

    return HybridEMPicFluidRun(deck, circuit_boundary=circuit_boundary).run(
        n_steps=n_steps
    )


def _validate_deck(deck: FirstPrinciples3DDeck) -> None:
    if int(deck.n_steps) != deck.n_steps or deck.n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if deck.dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if deck.sigma0_S_m < 0.0:
        raise ValueError("sigma0_S_m must be non-negative")
    if deck.background_density_m3 <= 0.0:
        raise ValueError("background_density_m3 must be positive")
    if deck.density_floor_m3 <= 0.0:
        raise ValueError("density_floor_m3 must be positive")
    if deck.particle_weight <= 0.0:
        raise ValueError("particle_weight must be positive")
    if deck.ion_mass_kg <= 0.0:
        raise ValueError("ion_mass_kg must be positive")
    if deck.marder_factor_scale < 0.0:
        raise ValueError("marder_factor_scale must be non-negative")
    if deck.circuit_capacitance_F <= 0.0:
        raise ValueError("circuit_capacitance_F must be positive")
    if deck.circuit_voltage_V <= 0.0:
        raise ValueError("circuit_voltage_V must be positive")
    if deck.circuit_inductance_H <= 0.0:
        raise ValueError("circuit_inductance_H must be positive")
    if deck.circuit_resistance_ohm < 0.0:
        raise ValueError("circuit_resistance_ohm must be non-negative")


def _build_minimal_pic(deck: FirstPrinciples3DDeck, grid: Maxwell3DGrid) -> HybridPIC:
    center = np.array(
        [0.5 * grid.nx * grid.dx, 0.5 * grid.ny * grid.dy, 0.5 * grid.nz * grid.dz],
        dtype=float,
    )
    offsets = np.array(
        [
            [-0.25 * grid.dx, 0.0, 0.0],
            [0.25 * grid.dx, 0.0, 0.0],
            [0.0, -0.25 * grid.dy, 0.0],
            [0.0, 0.25 * grid.dy, 0.0],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [8.0e5, 0.0, 0.0],
            [-8.0e5, 0.0, 0.0],
            [0.0, 8.0e5, 0.0],
            [0.0, -8.0e5, 0.0],
        ],
        dtype=float,
    )
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=deck.dt_s,
        use_esirkepov=True,
        use_binary_collisions=False,
    )
    pic.add_species(
        deck.ion_species_name,
        deck.ion_mass_kg,
        deck.ion_charge_C,
        positions=center[np.newaxis, :] + offsets,
        velocities=velocities,
        weights=np.full(4, deck.particle_weight, dtype=float),
    )
    return pic


def _candidate_evidence(
    *,
    geometry: HybridPICSourceGeometry,
    simulation: HybridPIC3DSimulationResult,
    conservation: dict[str, Any],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "true_3d_dimensionality": hybrid_simulator_candidate_evidence(
            simulation.telemetry
        ),
        "same_scope_3d_validation_packet": source_geometry_candidate_evidence(
            geometry
        ),
        "conservation_telemetry": conservation,
    }
    if simulation.step_results:
        last_step = simulation.step_results[-1]
        evidence["kinetic_ion_pic_push_deposition"] = hybrid_loop_candidate_evidence(
            last_step.telemetry
        )
        evidence["source_ordered_time_loop"] = source_ordered_loop_candidate_evidence(
            last_step.telemetry
        )
        if last_step.telemetry.electron_energy is not None:
            evidence["separate_electron_energy_closure"] = _candidate_record(
                capability="separate_electron_energy_closure",
                implementation="src/dpf/fields/electron_energy.py",
                telemetry=last_step.telemetry.electron_energy,
            )
        if last_step.telemetry.kinetic_yield is not None:
            evidence["kinetic_ion_neutron_yield_history"] = _candidate_record(
                capability="kinetic_ion_neutron_yield_history",
                implementation="src/dpf/fields/kinetic_yield.py",
                telemetry=last_step.telemetry.kinetic_yield,
            )
        if simulation.telemetry.circuit is not None:
            evidence["external_circuit_magnetic_boundary"] = {
                "passed": True,
                "status": "candidate",
                "capability": "external_circuit_magnetic_boundary",
                "source": HYBRID_PIC_3D_SOURCE,
                "implementation": "src/dpf/fields/circuit_boundary.py",
                "evidence_type": "engineering_circuit_boundary_coupled_run",
                "circuit": simulation.telemetry.circuit,
                "can_support_first_principles_acceptance": False,
            }
    return evidence


def _candidate_record(
    *,
    capability: str,
    implementation: str,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "passed": str(telemetry.get("status", "")).startswith("candidate_engineering_"),
        "status": "candidate",
        "capability": capability,
        "source": telemetry.get("source", HYBRID_PIC_3D_SOURCE),
        "source_lines": telemetry.get("source_lines"),
        "implementation": implementation,
        "evidence_type": "engineering_candidate_component_telemetry",
        "telemetry": dict(telemetry),
        "can_support_first_principles_acceptance": False,
    }


def _first_principles_candidate_packet(
    *,
    geometry_dimensionality: str,
    source_scope: str,
    hybrid_pic_3d_evidence: Mapping[str, Any],
    conservation_evidence: Mapping[str, Any],
    startup_bvp: Mapping[str, Any],
    limiter_readiness: Mapping[str, Any],
    dimensionality_handoff: Mapping[str, Any],
    same_scope_source: Mapping[str, Any],
    waveform_phase: Mapping[str, Any],
    spatial_field_temperature: Mapping[str, Any],
    neutron_authority: Mapping[str, Any],
    comparator_uq: Mapping[str, Any],
    numerical_fidelity: Mapping[str, Any],
    certificate_gate: Mapping[str, Any],
    generalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize candidate evidence without invoking the validation workflow."""

    evidence_keys = sorted(str(key) for key in hybrid_pic_3d_evidence)
    return {
        "status": "not_validation",
        "scientific_status": ENGINEERING_CANDIDATE_STATUS,
        "geometry_dimensionality": geometry_dimensionality,
        "source_scope": source_scope,
        "candidate_evidence_keys": evidence_keys,
        "conservation_status": conservation_evidence.get("status"),
        "startup_bvp_status": startup_bvp.get("status"),
        "startup_bvp_missing_acceptance_channels": startup_bvp.get(
            "missing_acceptance_channels",
            (),
        ),
        "limiter_readiness_status": limiter_readiness.get("status"),
        "limiter_readiness_missing_acceptance_channels": limiter_readiness.get(
            "missing_acceptance_channels",
            (),
        ),
        "dimensionality_handoff_status": dimensionality_handoff.get("status"),
        "dimensionality_handoff_missing_acceptance_channels": (
            dimensionality_handoff.get("missing_acceptance_channels", ())
        ),
        "dimensionality_handoff_blocked_observables": dimensionality_handoff.get(
            "blocked_observables",
            (),
        ),
        "dimensionality_handoff_can_support_first_principles_acceptance": (
            dimensionality_handoff.get("can_support_first_principles_acceptance")
        ),
        "same_scope_source_status": same_scope_source.get("status"),
        "same_scope_missing_acceptance_channels": same_scope_source.get(
            "missing_acceptance_channels",
            (),
        ),
        "waveform_phase_status": waveform_phase.get("status"),
        "waveform_phase_missing_acceptance_channels": waveform_phase.get(
            "missing_acceptance_channels",
            (),
        ),
        "spatial_field_temperature_status": spatial_field_temperature.get("status"),
        "spatial_field_temperature_missing_acceptance_channels": (
            spatial_field_temperature.get("missing_acceptance_channels", ())
        ),
        "neutron_authority_status": neutron_authority.get("status"),
        "neutron_authority_missing_acceptance_channels": neutron_authority.get(
            "missing_acceptance_channels",
            (),
        ),
        "comparator_uq_status": comparator_uq.get("status"),
        "comparator_uq_missing_acceptance_channels": comparator_uq.get(
            "missing_acceptance_channels",
            (),
        ),
        "numerical_fidelity_status": numerical_fidelity.get("status"),
        "numerical_fidelity_missing_acceptance_channels": numerical_fidelity.get(
            "missing_acceptance_channels",
            (),
        ),
        "certificate_gate_status": certificate_gate.get("status"),
        "certificate_gate_missing_acceptance_channels": certificate_gate.get(
            "missing_acceptance_channels",
            (),
        ),
        "generalization_status": generalization.get("status"),
        "generalization_missing_acceptance_channels": generalization.get(
            "missing_acceptance_channels",
            (),
        ),
        "reduced_models_used": False,
        "can_support_first_principles_acceptance": False,
        "blocking_reasons": [
            "engineering candidate only; no same-scope source-truth acceptance packet",
            "no accepted source-backed startup BVP or reviewed startup handoff packet attached",
            "no accepted limiter-readiness and limiter-zero packet attached",
            "no accepted dimensionality and MHD-to-kinetic handoff packet attached",
            "no independent backend convergence suite attached",
            "no mechanism-separated neutron authority packet attached",
            "no comparator and uncertainty matrix attached",
            "no accepted numerical-fidelity packet attached",
            "no accepted first-principles certificate gate attached",
            "no second-scope first-principles generalization packet attached",
            "no engineer-reviewed whole-shot validation target certificate attached",
        ],
    }


def _energy_snapshot(
    *,
    loop: HybridPIC3DLoop,
    state: Any,
    pic: HybridPIC,
    electron_state: ElectronEnergyState | None,
    circuit_boundary: CircuitMagneticBoundaryDrive | None,
    circuit_state: CircuitState | None,
) -> dict[str, float]:
    diagnostics = loop.field_stepper.maxwell.diagnostics(state)
    particle_kinetic = _particle_kinetic_energy_J(pic)
    electron_internal = (
        0.0
        if electron_state is None
        else float(np.sum(electron_state.electron_energy_J_m3) * loop.grid.cell_volume)
    )
    circuit = _circuit_energy_J(circuit_boundary, circuit_state)
    total = (
        diagnostics.total_energy_J
        + particle_kinetic
        + electron_internal
        + circuit
    )
    return {
        "field_energy_J": float(diagnostics.total_energy_J),
        "electric_energy_J": float(diagnostics.electric_energy_J),
        "magnetic_energy_J": float(diagnostics.magnetic_energy_J),
        "particle_kinetic_energy_J": particle_kinetic,
        "electron_internal_energy_J": electron_internal,
        "circuit_energy_J": circuit,
        "tracked_total_energy_J": float(total),
    }


def _particle_kinetic_energy_J(pic: HybridPIC) -> float:
    total = 0.0
    for species in pic.species:
        if species.n_particles() == 0:
            continue
        speed_sq = np.sum(species.velocities * species.velocities, axis=1)
        total += float(np.sum(0.5 * species.mass * species.weights * speed_sq))
    return total


def _circuit_energy_J(
    circuit_boundary: CircuitMagneticBoundaryDrive | None,
    circuit_state: CircuitState | None,
) -> float:
    if circuit_boundary is None or circuit_state is None:
        return 0.0
    params = circuit_boundary.parameters
    return float(
        0.5 * params.inductance_H * circuit_state.current_A**2
        + 0.5 * circuit_state.charge_C**2 / params.capacitance_F
    )


def _conservation_telemetry(
    *,
    grid: Maxwell3DGrid,
    n_steps: int,
    dt_s: float,
    initial: dict[str, float],
    final: dict[str, float],
    final_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    initial_total = initial["tracked_total_energy_J"]
    final_total = final["tracked_total_energy_J"]
    delta = final_total - initial_total
    relative = 0.0 if initial_total == 0.0 else delta / initial_total
    finite = bool(
        np.all(np.isfinite(list(initial.values())))
        and np.all(np.isfinite(list(final.values())))
        and np.isfinite(relative)
    )
    return {
        "passed": finite,
        "status": "engineering_candidate_conservation_telemetry_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "run_mode": RUN_MODE,
        "grid_shape": list(grid.shape),
        "n_steps": int(n_steps),
        "dt_s": float(dt_s),
        "initial": initial,
        "final": final,
        "delta_tracked_total_energy_J": float(delta),
        "relative_tracked_total_energy_change": float(relative),
        "final_max_abs_div_B_T_per_m": final_diagnostics["max_abs_div_B_T_per_m"],
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Tracks field, particle, electron-internal, and optional lumped-circuit energy only.",
            "Does not include accepted source-backed detector, UQ, nondominance, or backend-scaling evidence.",
            "Finite conservation telemetry is engineering evidence, not validation.",
        ],
    }


def _build_manifest(
    *,
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
    simulation: HybridPIC3DSimulationResult,
    telemetry: dict[str, Any],
    conservation: dict[str, Any],
    validation_packet: dict[str, Any],
) -> dict[str, Any]:
    package_manifest = build_first_principles_manifest_from_hybrid_result(
        simulation,
        backend="package_native",
        grid=grid,
        source_index_references=[{
            "source_id": "hybrid_pic_3d_architecture_source",
            "path": HYBRID_PIC_3D_SOURCE,
            "scope": RUN_MODE,
            "status": "source_reference_not_validation",
        }],
        metadata={
            "status": ENGINEERING_CANDIDATE_STATUS,
            "run_mode": RUN_MODE,
            "validation_packet_status": validation_packet["status"],
            "reduced_models_used": False,
            "deck": deck.manifest_config(),
        },
        notes="Package-native 3D hybrid EM/PIC-fluid engineering candidate.",
    ).to_dict()
    payload = dict(package_manifest)
    payload["status"] = ENGINEERING_CANDIDATE_STATUS
    payload["run_mode"] = RUN_MODE
    payload["scientific_status"] = ENGINEERING_CANDIDATE_STATUS
    payload["reduced_models_used"] = False
    payload["can_support_first_principles_acceptance"] = False
    payload["candidate_evidence"] = {
        "first_principles_3d_runner": {
            "passed": True,
            "status": ENGINEERING_CANDIDATE_STATUS,
            "run_mode": RUN_MODE,
            "can_support_first_principles_acceptance": False,
        },
        "first_principles_candidate_packet": validation_packet,
        "conservation_telemetry": conservation,
        "startup_bvp_packet": telemetry["startup"],
        "limiter_readiness_packet": telemetry["limiter_readiness"],
        "power_port_packet": telemetry["power_port"],
        "dimensionality_handoff_packet": telemetry["dimensionality_handoff"],
        "physics_closure_packet": telemetry["physics_closure"],
        "same_scope_source_packet": telemetry["same_scope_source"],
        "waveform_phase_packet": telemetry["waveform_phase"],
        "spatial_field_temperature_packet": telemetry["spatial_field_temperature"],
        "neutron_authority_packet": telemetry["neutron_authority"],
        "comparator_uq_packet": telemetry["comparator_uq"],
        "numerical_fidelity_packet": telemetry["numerical_fidelity"],
        "certificate_gate_packet": telemetry["certificate_gate"],
        "generalization_packet": telemetry["generalization"],
        "hybrid_pic_3d": telemetry["candidate_evidence"],
    }
    payload["first_principles_manifest"] = package_manifest
    return payload


def _coerce_deck_values(values: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(values)
    for key in ("grid_shape", "grid_spacing_m"):
        if coerced.get(key) is not None:
            coerced[key] = tuple(coerced[key])
    if isinstance(coerced.get("circuit_udpf_V"), list):
        coerced["circuit_udpf_V"] = tuple(float(v) for v in coerced["circuit_udpf_V"])
    for key in (
        "startup_accepted_channels",
        "startup_required_channels",
        "startup_missing_channels",
    ):
        if coerced.get(key) is not None:
            coerced[key] = tuple(str(v) for v in coerced[key])
    if coerced.get("limiter_readiness_accepted_channels") is not None:
        coerced["limiter_readiness_accepted_channels"] = tuple(
            str(v) for v in coerced["limiter_readiness_accepted_channels"]
        )
    if coerced.get("same_scope_accepted_channels") is not None:
        coerced["same_scope_accepted_channels"] = tuple(
            str(v) for v in coerced["same_scope_accepted_channels"]
        )
    if coerced.get("waveform_phase_accepted_channels") is not None:
        coerced["waveform_phase_accepted_channels"] = tuple(
            str(v) for v in coerced["waveform_phase_accepted_channels"]
        )
    if coerced.get("spatial_field_temperature_accepted_channels") is not None:
        coerced["spatial_field_temperature_accepted_channels"] = tuple(
            str(v) for v in coerced["spatial_field_temperature_accepted_channels"]
        )
    if coerced.get("neutron_authority_accepted_channels") is not None:
        coerced["neutron_authority_accepted_channels"] = tuple(
            str(v) for v in coerced["neutron_authority_accepted_channels"]
        )
    if coerced.get("comparator_uq_accepted_channels") is not None:
        coerced["comparator_uq_accepted_channels"] = tuple(
            str(v) for v in coerced["comparator_uq_accepted_channels"]
        )
    if coerced.get("numerical_fidelity_accepted_channels") is not None:
        coerced["numerical_fidelity_accepted_channels"] = tuple(
            str(v) for v in coerced["numerical_fidelity_accepted_channels"]
        )
    if coerced.get("certificate_accepted_channels") is not None:
        coerced["certificate_accepted_channels"] = tuple(
            str(v) for v in coerced["certificate_accepted_channels"]
        )
    if coerced.get("generalization_accepted_channels") is not None:
        coerced["generalization_accepted_channels"] = tuple(
            str(v) for v in coerced["generalization_accepted_channels"]
        )
    if coerced.get("validation_targets") is not None:
        coerced["validation_targets"] = tuple(
            dict(v) for v in coerced["validation_targets"]
        )
    return {
        key: value
        for key, value in coerced.items()
        if key in FirstPrinciples3DDeck.__dataclass_fields__
    }


def _get(value: Mapping[str, Any] | object, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _last_step_has_key(last_step: Mapping[str, Any], key: str) -> bool:
    return isinstance(last_step.get(key), Mapping)


def _collisions_enabled_in_telemetry(last_step: Mapping[str, Any]) -> bool:
    source_order = last_step.get("source_ordered_loop")
    if not isinstance(source_order, Mapping):
        return False
    return source_order.get("collision_stage") == "collisions_after_velocity_update"


def _values_from_package_deck(deck: Any) -> dict[str, Any]:
    grid = deck.grid
    diagnostics = deck.diagnostics
    startup = deck.startup
    closures = deck.closures
    circuit = deck.circuit
    gas = deck.gas
    electric = tuple(getattr(startup, "initial_electric_field_V_m", (1.0e5, 0.0, 0.0)))
    magnetic = tuple(getattr(startup, "initial_magnetic_field_T", (0.0, 0.0, 0.0)))
    return {
        "device_name": deck.device.name,
        "device_anode_radius_m": deck.device.anode_radius_m,
        "device_cathode_radius_m": deck.device.cathode_radius_m,
        "device_anode_length_m": deck.device.anode_length_m,
        "device_insulator_length_m": deck.device.insulator_length_m,
        "validation_scope": _validation_scope_from_package_deck(deck),
        "validation_targets": tuple(asdict(target) for target in deck.validation_targets),
        "n_steps": diagnostics.n_steps,
        "grid_shape": tuple(grid.shape),
        "grid_spacing_m": tuple(grid.spacing_m),
        "dt_s": diagnostics.dt_s,
        "sigma0_S_m": closures.sigma0_S_m,
        "background_density_m3": startup.background_density_m3,
        "density_floor_m3": closures.density_floor_m3,
        "electron_temperature_K": startup.electron_temperature_K,
        "ion_temperature_K": startup.ion_temperature_K,
        "ion_species_name": gas.species,
        "gas_pressure_Pa": gas.pressure_Pa,
        "gas_temperature_K": gas.temperature_K,
        "ion_mass_kg": gas.ion_mass_kg,
        "ion_charge_C": gas.ion_charge_C,
        "particle_weight": startup.particle_weight,
        "initial_E_x_V_m": electric[0],
        "initial_B_z_T": magnetic[2],
        "include_hall": closures.include_hall,
        "use_predictor_corrector": closures.use_predictor_corrector,
        "use_source_ordered_velocity_update": (
            closures.use_source_ordered_velocity_update
        ),
        "marder_factor_scale": closures.marder_factor_scale,
        "marder_nondominance_threshold": closures.marder_nondominance_threshold,
        "ohmic_cfl_safety": closures.ohmic_cfl_safety,
        "apply_circuit_boundary": closures.apply_circuit_boundary,
        "circuit_capacitance_F": circuit.capacitance_F,
        "circuit_voltage_V": circuit.voltage_V,
        "circuit_inductance_H": circuit.inductance_H,
        "circuit_resistance_ohm": circuit.resistance_ohm,
        "circuit_state": CircuitState(
            current_A=circuit.initial_current_A,
            charge_C=circuit.charge_C,
        ),
        "circuit_udpf_V": closures.circuit_udpf_V,
        "startup_mode": startup.mode,
        "startup_evidence_status": startup.evidence_status,
        "startup_source_scope": startup.source_scope,
        "startup_can_support_whole_shot_acceptance": (
            startup.can_support_whole_shot_acceptance
        ),
        "startup_required_channels": tuple(startup.required_channels),
        "startup_missing_channels": tuple(startup.missing_channels),
    }


def _validation_scope_from_package_deck(deck: Any) -> str:
    if getattr(deck, "validation_targets", ()):
        return str(getattr(deck, "deck_id", "declared_validation_target_scope"))
    return str(getattr(deck, "deck_id", getattr(deck.device, "name", "not_declared")))
