"""Package-native 3-D hybrid EM/PIC-fluid first-principles runner.

This module wires the existing ``dpf.fields`` 3-D Maxwell, HybridPIC loop,
electron-energy, kinetic-yield, and optional circuit-boundary components into
one minimal engineering-candidate run.  It is deliberately fail-closed:
results are marked as engineering candidates and cannot be used as validation
evidence.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from dpf import constants as dpf_constants
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields import (
    CircuitMagneticBoundaryDrive,
    CircuitParameters,
    CircuitState,
    DeuteriumIonizationTransport,
    ElectronEnergyClosure,
    ElectronEnergyState,
    HybridPIC3DLoop,
    HybridPIC3DSimulationResult,
    HybridPIC3DSimulator,
    HybridPICSourceGeometry,
    KineticIonYieldHistory,
    Maxwell3DBoundaries,
    Maxwell3DGrid,
    ParticleAbsorbingBoundaries,
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
from dpf.first_principles.current_waveform_comparator import (
    build_engineering_current_waveform_comparator,
)
from dpf.first_principles.deck import (
    FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES,
    FirstPrinciplesInputDeck,
)
from dpf.first_principles.dimensionality import build_dimensionality_handoff_packet
from dpf.first_principles.experimental_numerics import (
    build_experimental_numerical_runtime_audit_packet,
)
from dpf.first_principles.experimental_shot import (
    build_experimental_whole_shot_packet,
)
from dpf.first_principles.generalization import build_generalized_dpf_machine_packet
from dpf.first_principles.limiter_proof import (
    build_experimental_limiter_zero_probe_packet,
)
from dpf.first_principles.limiter_readiness import build_limiter_readiness_packet
from dpf.first_principles.manifest import (
    ARTIFACT_SCHEMA_VERSION,
    build_first_principles_manifest_from_hybrid_result,
    git_provenance,
    sha256_of_file_soft,
    sha256_of_text,
    source_packet_hashes_from_references,
)
from dpf.first_principles.neutron_authority import (
    build_mechanism_separated_neutron_packet,
)
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet
from dpf.first_principles.plasmapy_audit import build_plasmapy_formulary_audit_packet
from dpf.first_principles.power_port import build_engineering_power_port_packet
from dpf.first_principles.runtime_demonstrator_scope import SELECTED_SCOPE_LABEL
from dpf.first_principles.same_scope import (
    ARCHITECTURE_OR_SCHEMA_CONTEXT_SOURCES,
    build_same_scope_source_packet,
)
from dpf.first_principles.spatial_field_temperature import (
    build_spatial_field_temperature_packet,
)
from dpf.first_principles.startup_breakdown import (
    build_candidate_startup_breakdown_audit,
)
from dpf.first_principles.startup_bvp import build_startup_bvp_packet
from dpf.first_principles.waveform_phase import build_waveform_phase_packet
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

ENGINEERING_CANDIDATE_STATUS = "engineering_candidate_not_validation"
RUN_MODE = "first_principles_3d_hybrid_em_pic_fluid"
ELEMENTARY_CHARGE = dpf_constants.e
DEUTERON_MASS_KG = dpf_constants.m_d
K_B = dpf_constants.k_B
MU_0 = dpf_constants.mu_0

PF1000_AKEL_SOURCE_LOCKED_DECK = {
    "device_anode_radius_m": 0.1155,
    "device_cathode_radius_m": 0.16,
    "device_anode_length_m": 0.48,
    "device_insulator_length_m": 0.085,
    "device_cathode_rod_count": 12,
    "device_cathode_rod_diameter_m": 0.080,
    "circuit_capacitance_F": 1.332e-3,
    "circuit_voltage_V": 1.6e4,
    "circuit_inductance_H": 25.0e-9,
    "circuit_resistance_ohm": 6.1e-3,
    "gas_pressure_Pa": 1.2 * 133.32236842105263,
}

PF1000_AKEL_DECK_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "108-142,262-270",
        "role": "pf1000_akel_circuit_gas_geometry_scope",
    },
    {
        "path": (
            "KnowledgeReference/"
            "experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        ),
        "lines": "340-356",
        "role": "pf1000_electrode_rods_and_insulator_geometry",
    },
)


@dataclass(frozen=True)
class InitialPlasmaProfile:
    """Grid-shaped startup fields used by the candidate 3-D runtime."""

    total_deuterium_density_m3: np.ndarray
    ionization_fraction: np.ndarray
    electron_temperature_K: np.ndarray
    ion_temperature_K: np.ndarray
    plasma_velocity_m_s: np.ndarray
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class FirstPrinciples3DDeck:
    """Minimal input deck for a package-native 3-D hybrid EM/PIC-fluid run."""

    n_steps: int = 1
    history_stride: int = 1
    max_step_results: int | None = 256
    target_time_s: float | None = None
    grid_shape: tuple[int, int, int] = (5, 5, 5)
    grid_spacing_m: tuple[float, float, float] | None = None
    dt_s: float = 1.0e-13
    sigma0_S_m: float = 1.0e2
    background_density_m3: float = 1.0e20
    density_floor_m3: float = 1.0e20
    initial_ionization_fraction: float = 0.01
    pressure_density_threshold_m3: float | None = None
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
    marder_factor_scale: float = 0.0
    marder_nondominance_threshold: float = 0.5
    ohmic_cfl_safety: float = 1.0
    apply_circuit_boundary: bool = True
    circuit_capacitance_F: float = 2.0e-5
    circuit_voltage_V: float = 1.5e4
    circuit_inductance_H: float = 1.1e-7
    circuit_resistance_ohm: float = 1.2e-2
    circuit_state: CircuitState | None = None
    circuit_udpf_V: float | tuple[float, ...] = 0.0
    circuit_udpf_mode: str = "lagged_volume_j_dot_e"
    circuit_feedback_min_current_A: float = 1.0
    circuit_z_index: int = 0
    circuit_blend: float = 1.0
    pml_cells: int = 0
    pml_strength: float = 0.0
    particle_absorption_enabled: bool = False
    open_boundary: bool = True
    conductor_cells: Any | None = None
    conductor_mask_status: str = "not_supplied"
    conductor_mask_mode: str = "none"
    # SS10-2 (closes audit A2): blocked geometry fields carried from the package
    # deck's boundary policy.  Each entry is a plain mapping with field_name,
    # blocker_id, blocked, and source_scope_reason so the conductor-mask runtime
    # telemetry and the segmented manifest can expose every blocked field.
    blocked_geometry_fields: tuple[dict[str, Any], ...] = ()
    device_anode_radius_m: float | None = None
    device_cathode_radius_m: float | None = None
    device_anode_length_m: float | None = None
    device_insulator_length_m: float | None = None
    device_anode_inner_radius_m: float | None = None
    device_cathode_rod_count: int | None = None
    device_cathode_rod_diameter_m: float | None = None
    device_cathode_rod_length_m: float | None = None
    device_insulator_outer_radius_m: float | None = None
    device_insulator_material: str | None = None
    gas_pressure_Pa: float | None = None
    gas_temperature_K: float | None = None
    startup_mode: str = "source_backed_end_rundown_sheath"
    startup_evidence_status: str = "engineering_candidate_not_whole_shot"
    startup_source_scope: str = "end_of_rundown_or_engineering_startup"
    startup_can_support_whole_shot_acceptance: bool = False
    startup_accepted_channels: tuple[str, ...] = ()
    startup_required_channels: tuple[str, ...] = ()
    startup_missing_channels: tuple[str, ...] = ()
    startup_payload: dict[str, Any] = field(default_factory=dict)
    device_name: str = "not_declared"
    validation_scope: str = "not_declared_engineering_smoke"
    # Selected-machine source scope: device-and-operating-point source scope
    # derived from the package deck's KR geometry/circuit citations.  Distinct
    # from architecture/equation-method evidence (the hybrid-PIC paper).  A
    # deck id is NEVER a source scope.  Super-Sprint 9 WS9-2 (fixes P0-2).
    selected_machine_source_scope: str = (
        "not_declared_engineering_smoke_machine_source"
    )
    # KR geometry/circuit source-reference paths for the selected machine.
    # Populated from the package deck's device source references; used by the
    # candidate packet and conductor-mask telemetry instead of the LLNL-like
    # architecture geometry.  Super-Sprint 9 WS9-2 / WS9-6.
    selected_machine_source_references: tuple[str, ...] = ()
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

        boundary_policy = values.pop("boundary_policy", None) or values.pop(
            "boundaries",
            None,
        )
        if boundary_policy is not None:
            values.update(_boundary_values_from_policy(boundary_policy))

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
                "anode_radius_m": self.device_anode_radius_m,
                "anode_inner_radius_m": self.device_anode_inner_radius_m,
                "cathode_radius_m": self.device_cathode_radius_m,
                "cathode_rod_count": self.device_cathode_rod_count,
                "cathode_rod_diameter_m": self.device_cathode_rod_diameter_m,
                "cathode_rod_length_m": self.device_cathode_rod_length_m,
                "insulator_length_m": self.device_insulator_length_m,
                "insulator_outer_radius_m": self.device_insulator_outer_radius_m,
                "insulator_material": self.device_insulator_material,
            },
            "fluid": {
                "backend": "hybrid",
                "precision": "float64",
                "sigma0_S_m": self.sigma0_S_m,
                "background_density_m3": self.background_density_m3,
                "density_floor_m3": self.density_floor_m3,
                "initial_ionization_fraction": self.initial_ionization_fraction,
                "pressure_density_threshold_m3": _pressure_density_threshold_m3(self),
            },
            "diagnostics": {
                "artifact_classification": ENGINEERING_CANDIDATE_STATUS,
                "artifact_distribution": "local_engineering",
                "artifact_handling_notes": "candidate 3D hybrid EM/PIC-fluid run",
                "history_stride": self.history_stride,
                "max_step_results": self.max_step_results,
                "target_time_s": self.target_time_s,
            },
            "first_principles_3d": {
                "n_steps": self.n_steps,
                "dt_s": self.dt_s,
                "history_stride": self.history_stride,
                "max_step_results": self.max_step_results,
                "target_time_s": self.target_time_s,
                "apply_circuit_boundary": self.apply_circuit_boundary,
                "boundary_policy": _boundary_policy_manifest(self),
                "circuit_udpf_mode": self.circuit_udpf_mode,
                "circuit_feedback_min_current_A": self.circuit_feedback_min_current_A,
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
        device = {
            "device_name": self.device_name,
            "anode_radius_m": self.device_anode_radius_m,
            "cathode_radius_m": self.device_cathode_radius_m,
            "anode_length_m": self.device_anode_length_m,
            "insulator_length_m": self.device_insulator_length_m,
            "anode_inner_radius_m": self.device_anode_inner_radius_m,
            "cathode_rod_count": self.device_cathode_rod_count,
            "cathode_rod_diameter_m": self.device_cathode_rod_diameter_m,
            "cathode_rod_length_m": self.device_cathode_rod_length_m,
            "insulator_outer_radius_m": self.device_insulator_outer_radius_m,
            "insulator_material": self.device_insulator_material,
        }
        gas = {
            "species": self.ion_species_name,
            "pressure_Pa": self.gas_pressure_Pa,
            "temperature_K": self.gas_temperature_K,
        }
        circuit = {
            "voltage_V": self.circuit_voltage_V,
            "initial_current_A": (
                None if self.circuit_state is None else self.circuit_state.current_A
            ),
            "charge_C": (
                None if self.circuit_state is None else self.circuit_state.charge_C
            ),
        }
        startup = {
            "mode": self.startup_mode,
            "evidence_status": self.startup_evidence_status,
            "source_scope": self.startup_source_scope,
            "can_support_whole_shot_acceptance": (
                self.startup_can_support_whole_shot_acceptance
            ),
            "accepted_channels": list(self.startup_accepted_channels),
            "required_channels": list(self.startup_required_channels),
            "missing_channels": list(self.startup_missing_channels),
            "startup_payload": self.startup_payload,
            "background_density_m3": self.background_density_m3,
            "initial_ionization_fraction": self.initial_ionization_fraction,
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
        }
        candidate_breakdown_audit = build_candidate_startup_breakdown_audit(
            device=device,
            gas=gas,
            circuit=circuit,
            startup=startup,
        )
        packet = build_startup_bvp_packet(
            startup,
            device=device,
            gas=gas,
            circuit=circuit,
            candidate_breakdown_audit=candidate_breakdown_audit,
            accepted_channels=self.startup_accepted_channels,
            include_bennett_wrong_scope_context=(
                self.validation_scope == SELECTED_SCOPE_LABEL
            ),
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


@dataclass
class FirstPrinciples3DSession:
    """Reusable first-principles 3-D simulator session for split runs."""

    deck: FirstPrinciples3DDeck
    simulator: HybridPIC3DSimulator
    electron_state: ElectronEnergyState
    ionization_state: Any
    circuit_state: CircuitState | None
    total_deuterium_density_m3: np.ndarray
    plasma_velocity_m_s: np.ndarray
    completed_steps: int = 0
    lagged_field_work: dict[str, Any] | None = None

    @classmethod
    def from_deck(
        cls,
        deck: Mapping[str, Any] | object | None = None,
        *,
        circuit_boundary: CircuitMagneticBoundaryDrive | None = None,
    ) -> FirstPrinciples3DSession:
        resolved = FirstPrinciples3DDeck.from_deck(deck)
        _validate_deck(resolved)
        if resolved.seed is not None:
            np.random.seed(int(resolved.seed))

        grid = resolved.grid()
        conductor_cells, _ = _resolve_conductor_cells(resolved, grid)
        maxwell_boundaries = Maxwell3DBoundaries(
            conductor_cells=conductor_cells,
            pml_cells=resolved.pml_cells,
            pml_strength=resolved.pml_strength,
            open_boundary=resolved.open_boundary,
        )
        particle_boundaries = (
            ParticleAbsorbingBoundaries(
                grid,
                conductor_cells=conductor_cells,
                pml_cells=resolved.pml_cells,
            )
            if resolved.particle_absorption_enabled
            else None
        )
        plasma_profile = _build_initial_plasma_profile(
            resolved,
            grid,
            conductor_cells=conductor_cells,
        )
        total_deuterium_density = plasma_profile.total_deuterium_density_m3
        initial_electron_density = np.maximum(
            total_deuterium_density * plasma_profile.ionization_fraction,
            1.0,
        )
        electron_closure = ElectronEnergyClosure(grid)
        electron_state = electron_closure.initialize(
            electron_temperature_K=plasma_profile.electron_temperature_K,
            ion_temperature_K=plasma_profile.ion_temperature_K,
            electron_density_m3=initial_electron_density,
        )
        ionization_transport = DeuteriumIonizationTransport(grid)
        ionization_state = ionization_transport.initialize(
            total_deuterium_density_m3=total_deuterium_density,
            ionization_fraction=plasma_profile.ionization_fraction,
        )
        loop = HybridPIC3DLoop(
            grid,
            maxwell_boundaries=maxwell_boundaries,
            particle_boundaries=particle_boundaries,
            electron_energy_closure=electron_closure,
            ionization_transport=ionization_transport,
            kinetic_yield_history=KineticIonYieldHistory(grid),
        )
        state = loop.field_stepper.maxwell.empty_state()
        state.E.Ex_edge.fill(resolved.initial_E_x_V_m)
        state.B.Bz_face.fill(resolved.initial_B_z_T)
        pic, _ = _build_initial_pic(
            resolved,
            grid,
            conductor_cells,
            plasma_profile=plasma_profile,
        )
        active_circuit_boundary = circuit_boundary
        if resolved.apply_circuit_boundary and active_circuit_boundary is None:
            active_circuit_boundary = CircuitMagneticBoundaryDrive(
                grid,
                parameters=CircuitParameters(
                    capacitance_F=resolved.circuit_capacitance_F,
                    voltage_V=resolved.circuit_voltage_V,
                    inductance_H=resolved.circuit_inductance_H,
                    resistance_ohm=resolved.circuit_resistance_ohm,
                ),
            )
        simulator = HybridPIC3DSimulator(
            grid=grid,
            loop=loop,
            state=state,
            pic=pic,
            circuit_boundary=active_circuit_boundary,
        )
        return cls(
            deck=resolved,
            simulator=simulator,
            electron_state=electron_state,
            ionization_state=ionization_state,
            circuit_state=(
                (resolved.circuit_state or CircuitState())
                if resolved.apply_circuit_boundary
                else None
            ),
            total_deuterium_density_m3=total_deuterium_density,
            plasma_velocity_m_s=plasma_profile.plasma_velocity_m_s,
        )

    def run_segment(self, n_steps: int) -> HybridPIC3DSimulationResult:
        """Advance this live session by a fixed number of steps."""

        if int(n_steps) != n_steps or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        deck = self.deck
        result = self.simulator.run(
            n_steps=int(n_steps),
            dt_s=deck.dt_s,
            sigma0_S_m=deck.sigma0_S_m,
            background_density_m3=deck.background_density_m3,
            ohmic_cfl_safety=deck.ohmic_cfl_safety,
            density_floor_m3=deck.density_floor_m3,
            include_hall=deck.include_hall,
            use_predictor_corrector=deck.use_predictor_corrector,
            marder_factor_m2=deck.marder_factor_scale
            * min(self.simulator.grid.spacing) ** 2,
            marder_nondominance_threshold=deck.marder_nondominance_threshold,
            electron_energy_state=self.electron_state,
            ionization_state=self.ionization_state,
            use_source_backed_conductivity=True,
            mass_density_kg_m3=self.total_deuterium_density_m3 * deck.ion_mass_kg,
            plasma_velocity_m_s=self.plasma_velocity_m_s,
            electron_temperature_floor_K=10.0,
            heat_flux_subcycles_max=5000,
            pressure_density_threshold_m3=_pressure_density_threshold_m3(deck),
            use_source_ordered_velocity_update=deck.use_source_ordered_velocity_update,
            circuit_state=self.circuit_state,
            apply_circuit_boundary=deck.apply_circuit_boundary,
            circuit_udpf_V=_segment_circuit_udpf(
                deck.circuit_udpf_V,
                start=self.completed_steps,
                count=int(n_steps),
            ),
            circuit_udpf_mode=deck.circuit_udpf_mode,
            circuit_feedback_min_current_A=deck.circuit_feedback_min_current_A,
            circuit_z_index=deck.circuit_z_index,
            circuit_blend=deck.circuit_blend,
            history_stride=deck.history_stride,
            max_step_results=deck.max_step_results,
            target_time_s=None,
            initial_lagged_field_work=self.lagged_field_work,
            step_index_offset=self.completed_steps,
        )
        self.completed_steps += result.telemetry.n_steps_completed
        if result.electron_energy is not None:
            self.electron_state = result.electron_energy
        if result.ionization_charge_state is not None:
            self.ionization_state = result.ionization_charge_state
        self.circuit_state = result.circuit
        self.lagged_field_work = _last_field_work_from_simulation(result)
        return result

    def run_adaptive_validity(
        self,
        *,
        target_time_s: float,
        max_steps: int,
        min_dt_s: float | None = None,
        max_dt_s: float | None = None,
        shrink_factor: float = 0.5,
        growth_factor: float = 1.1,
    ) -> dict[str, Any]:
        """Advance with rollback/retry when a source validity gate rejects a step."""

        if target_time_s <= 0.0:
            raise ValueError("target_time_s must be positive")
        if int(max_steps) != max_steps or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if shrink_factor <= 0.0 or shrink_factor >= 1.0:
            raise ValueError("shrink_factor must satisfy 0 < shrink_factor < 1")
        if growth_factor < 1.0:
            raise ValueError("growth_factor must be >= 1")
        max_dt = float(max_dt_s if max_dt_s is not None else self.deck.dt_s)
        if max_dt <= 0.0:
            raise ValueError("max_dt_s must be positive")
        min_dt = float(
            min_dt_s if min_dt_s is not None else max(max_dt * 2.0**-24, 1.0e-18)
        )
        if min_dt <= 0.0 or min_dt > max_dt:
            raise ValueError("min_dt_s must be positive and <= max_dt_s")

        elapsed_s = 0.0
        accepted_steps = 0
        rejected_steps = 0
        dt_s = min(max_dt, float(target_time_s))
        last_result: HybridPIC3DSimulationResult | None = None
        rejection_records: list[dict[str, Any]] = []
        dt_history: list[dict[str, Any]] = []
        limiter_summary = _empty_adaptive_limiter_summary()
        termination_reason = "target_time_reached"

        while elapsed_s < float(target_time_s) and accepted_steps < int(max_steps):
            dt_s = min(float(dt_s), float(target_time_s) - elapsed_s, max_dt)
            snapshot = deepcopy(self)
            self.deck = replace(
                self.deck,
                dt_s=float(dt_s),
                n_steps=1,
                history_stride=1,
                max_step_results=1,
                target_time_s=None,
            )
            result = self.run_segment(1)
            stop_reason = str(result.telemetry.stop_reason)
            if _adaptive_retry_required(stop_reason):
                _restore_session_from_snapshot(self, snapshot)
                rejected_steps += 1
                rejection_records.append(
                    _adaptive_rejection_record(
                        result=result,
                        attempted_dt_s=dt_s,
                        elapsed_s=elapsed_s,
                        accepted_steps=accepted_steps,
                    )
                )
                dt_s *= float(shrink_factor)
                if dt_s < min_dt:
                    termination_reason = "adaptive_min_dt_exhausted"
                    break
                continue

            last_result = result
            accepted_steps += 1
            elapsed_s += float(dt_s)
            _merge_adaptive_limiter_summary(
                limiter_summary,
                result.telemetry.limiter_activation_summary,
            )
            dt_history.append({
                "accepted_step": accepted_steps,
                "time_s": elapsed_s,
                "dt_s": float(dt_s),
                "stop_reason": stop_reason,
            })
            dt_s = min(max_dt, float(dt_s) * float(growth_factor))

        if accepted_steps >= int(max_steps) and elapsed_s < float(target_time_s):
            termination_reason = "adaptive_step_budget_exhausted"
        return {
            "status": "candidate_adaptive_validity_run_not_validation",
            "source": HYBRID_PIC_3D_SOURCE,
            "source_lines": "740-792, 1074-1097, 1226-1240",
            "target_time_s": float(target_time_s),
            "final_time_s": float(elapsed_s),
            "duration_request_satisfied": elapsed_s >= float(target_time_s),
            "termination_reason": termination_reason,
            "accepted_step_count": accepted_steps,
            "rejected_step_count": rejected_steps,
            "max_steps": int(max_steps),
            "min_dt_s": min_dt,
            "max_dt_s": max_dt,
            "final_dt_s": float(dt_s),
            "dt_history": dt_history[-64:],
            "rejection_records": rejection_records[-64:],
            "limiter_activation_summary": limiter_summary,
            "last_step": (
                None if last_result is None else last_result.telemetry.last_step
            ),
            "circuit": None if last_result is None else last_result.telemetry.circuit,
            "state_fingerprint": (
                None if last_result is None else last_result.telemetry.state_fingerprint
            ),
            "finite_state": None if last_result is None else last_result.telemetry.finite_state,
            "can_support_first_principles_acceptance": False,
            "limitations": [
                "Candidate adaptive runtime controller only; it retries rejected steps but does not validate timestep convergence.",
                "Variable-dt power-port and restart equivalence still require accepted numerical-fidelity review.",
            ],
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
        conductor_cells, conductor_mask_packet = _resolve_conductor_cells(deck, grid)
        maxwell_boundaries = Maxwell3DBoundaries(
            conductor_cells=conductor_cells,
            pml_cells=deck.pml_cells,
            pml_strength=deck.pml_strength,
            open_boundary=deck.open_boundary,
        )
        particle_boundaries = (
            ParticleAbsorbingBoundaries(
                grid,
                conductor_cells=conductor_cells,
                pml_cells=deck.pml_cells,
            )
            if deck.particle_absorption_enabled
            else None
        )
        boundary_policy_packet = _boundary_policy_telemetry(
            deck=deck,
            grid=grid,
            conductor_cells=conductor_cells,
            conductor_mask=conductor_mask_packet,
        )
        plasma_profile = _build_initial_plasma_profile(
            deck,
            grid,
            conductor_cells=conductor_cells,
        )
        total_deuterium_density = plasma_profile.total_deuterium_density_m3
        initial_electron_density = np.maximum(
            total_deuterium_density * plasma_profile.ionization_fraction,
            1.0,
        )
        electron_closure = ElectronEnergyClosure(grid)
        electron_state = electron_closure.initialize(
            electron_temperature_K=plasma_profile.electron_temperature_K,
            ion_temperature_K=plasma_profile.ion_temperature_K,
            electron_density_m3=initial_electron_density,
        )
        ionization_transport = DeuteriumIonizationTransport(grid)
        ionization_state = ionization_transport.initialize(
            total_deuterium_density_m3=total_deuterium_density,
            ionization_fraction=plasma_profile.ionization_fraction,
        )
        loop = HybridPIC3DLoop(
            grid,
            maxwell_boundaries=maxwell_boundaries,
            particle_boundaries=particle_boundaries,
            electron_energy_closure=electron_closure,
            ionization_transport=ionization_transport,
            kinetic_yield_history=KineticIonYieldHistory(grid),
        )
        state = loop.field_stepper.maxwell.empty_state()
        state.E.Ex_edge.fill(deck.initial_E_x_V_m)
        state.B.Bz_face.fill(deck.initial_B_z_T)
        pic, pic_loading_packet = _build_initial_pic(
            deck,
            grid,
            conductor_cells,
            plasma_profile=plasma_profile,
        )
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
            ionization_state=ionization_state,
            use_source_backed_conductivity=True,
            mass_density_kg_m3=total_deuterium_density * deck.ion_mass_kg,
            plasma_velocity_m_s=plasma_profile.plasma_velocity_m_s,
            electron_temperature_floor_K=10.0,
            heat_flux_subcycles_max=5000,
            pressure_density_threshold_m3=_pressure_density_threshold_m3(deck),
            use_source_ordered_velocity_update=deck.use_source_ordered_velocity_update,
            circuit_state=initial_circuit_state,
            apply_circuit_boundary=deck.apply_circuit_boundary,
            circuit_udpf_V=deck.circuit_udpf_V,
            circuit_udpf_mode=deck.circuit_udpf_mode,
            circuit_feedback_min_current_A=deck.circuit_feedback_min_current_A,
            circuit_z_index=deck.circuit_z_index,
            circuit_blend=deck.circuit_blend,
            history_stride=deck.history_stride,
            max_step_results=deck.max_step_results,
            target_time_s=deck.target_time_s,
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
            n_steps=simulation.telemetry.n_steps_completed,
            dt_s=deck.dt_s,
            initial=initial_energy,
            final=final_energy,
            final_diagnostics=loop.field_stepper.maxwell.diagnostics(simulation.state).to_dict(),
        )
        evidence = _candidate_evidence(
            geometry=geometry,
            simulation=simulation,
            conservation=conservation,
            boundary_policy=boundary_policy_packet,
            pic_loading=pic_loading_packet,
        )
        startup_packet = deck.startup_packet()
        deck_diff_packet = _deck_source_diff_packet(deck)
        simulation_telemetry = simulation.telemetry.to_dict()
        current_waveform_comparison_packet = (
            build_engineering_current_waveform_comparator(
                declared_scope=deck.validation_scope,
                device_name=deck.device_name,
                validation_targets=deck.validation_targets,
                simulation_telemetry=simulation_telemetry,
            )
        )
        power_port_packet = build_engineering_power_port_packet(
            simulation.telemetry.circuit,
            startup=startup_packet,
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
        )
        limiter_readiness_packet = build_limiter_readiness_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            accepted_channels=deck.limiter_readiness_accepted_channels,
            conservation=conservation,
            simulation_telemetry=simulation_telemetry,
        )
        limiter_zero_probe_packet = build_experimental_limiter_zero_probe_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
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
        plasmapy_audit_packet = build_plasmapy_formulary_audit_packet(
            _plasmapy_reference_state(deck)
        )
        physics_closure_packet = build_physics_closure_packet(
            include_hall=deck.include_hall,
            electron_energy_present=simulation.electron_energy is not None,
            kinetic_yield_present=_last_step_has_key(last_step, "kinetic_yield"),
            collisions_enabled=_collisions_enabled_in_telemetry(last_step),
            electron_heat_flux_present=_last_step_has_applied_heat_flux(last_step),
            electron_equilibration_audit_present=(
                _last_step_has_equilibration_audit(last_step)
            ),
            ionization_charge_state_present=_last_step_has_key(
                last_step,
                "ionization_charge_state",
            ),
            source_backed_transport_present=_last_step_has_source_backed_transport(
                last_step
            ),
            dimensionality=dimensionality_packet,
            community_formula_audit=plasmapy_audit_packet,
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
                "experimental_limiter_zero_probe": limiter_zero_probe_packet,
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
                "experimental_limiter_zero_probe": limiter_zero_probe_packet,
                "power_port": power_port_packet,
                "dimensionality_handoff": dimensionality_packet,
                "physics_closure": physics_closure_packet,
                "same_scope_source": same_scope_source_packet,
                "waveform_phase": waveform_phase_packet,
                "spatial_field_temperature": spatial_field_temperature_packet,
                "neutron_authority": neutron_authority_packet,
                "numerical_fidelity": numerical_fidelity_packet,
                "engineering_current_waveform_comparison": (
                    current_waveform_comparison_packet
                ),
            },
        )
        certificate_gate_packet = build_first_principles_certificate_gate_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            accepted_channels=deck.certificate_accepted_channels,
            upstream_packets={
                "startup_bvp": startup_packet,
                "limiter_readiness": limiter_readiness_packet,
                "experimental_limiter_zero_probe": limiter_zero_probe_packet,
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
                "experimental_limiter_zero_probe": limiter_zero_probe_packet,
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
        experimental_whole_shot_packet = build_experimental_whole_shot_packet(
            declared_scope=deck.validation_scope,
            device_name=deck.device_name,
            requested_duration_s=deck.target_time_s,
            step_budget=deck.n_steps,
            simulation_telemetry=simulation_telemetry,
            upstream_packets={
                "startup_bvp": startup_packet,
                "limiter_readiness": limiter_readiness_packet,
                "experimental_limiter_zero_probe": limiter_zero_probe_packet,
                "deck_diff": deck_diff_packet,
                "boundary_policy": boundary_policy_packet,
                "pic_particle_loading": pic_loading_packet,
                "power_port": power_port_packet,
                "dimensionality_handoff": dimensionality_packet,
                "physics_closure": physics_closure_packet,
                "same_scope_source": same_scope_source_packet,
                "waveform_phase": waveform_phase_packet,
                "engineering_current_waveform_comparison": (
                    current_waveform_comparison_packet
                ),
                "spatial_field_temperature": spatial_field_temperature_packet,
                "neutron_authority": neutron_authority_packet,
                "comparator_uq": comparator_uq_packet,
                "numerical_fidelity": numerical_fidelity_packet,
                "certificate_gate": certificate_gate_packet,
                "generalization": generalization_packet,
                "plasmapy_audit": plasmapy_audit_packet,
            },
            grid_spacing_m=grid.spacing,
            dt_s=deck.dt_s,
        )
        experimental_numerics_packet = (
            build_experimental_numerical_runtime_audit_packet(
                declared_scope=deck.validation_scope,
                device_name=deck.device_name,
                simulation_telemetry=simulation_telemetry,
                conservation=conservation,
                duration_plan=experimental_whole_shot_packet.get("duration_plan"),
                limiter_readiness=limiter_readiness_packet,
                numerical_fidelity=numerical_fidelity_packet,
                grid_spacing_m=grid.spacing,
                dt_s=deck.dt_s,
            )
        )
        hybrid_pic_3d_readiness = hybrid_pic_3d_readiness_status(
            {
                "geometry_dimensionality": "cartesian_3d",
                "hybrid_pic_3d_evidence": evidence,
            }
        )
        # WS9-2: the candidate packet carries the SELECTED-MACHINE source scope
        # (device + operating point), never the LLNL-like architecture scope.
        # The hybrid-PIC paper stays separate architecture/equation-method
        # evidence under ``architecture_source`` / ``architecture_source_scope``.
        validation_packet = _first_principles_candidate_packet(
            geometry_dimensionality="cartesian_3d",
            source_scope=deck.selected_machine_source_scope,
            architecture_source=HYBRID_PIC_3D_SOURCE,
            architecture_source_scope=geometry.architecture_source_scope,
            selected_machine_source_references=(
                deck.selected_machine_source_references
            ),
            hybrid_pic_3d_evidence=evidence,
            hybrid_pic_3d_readiness=hybrid_pic_3d_readiness,
            conservation_evidence=conservation,
            startup_bvp=startup_packet,
            limiter_readiness=limiter_readiness_packet,
            dimensionality_handoff=dimensionality_packet,
            same_scope_source=same_scope_source_packet,
            waveform_phase=waveform_phase_packet,
            spatial_field_temperature=spatial_field_temperature_packet,
            current_waveform_comparison=current_waveform_comparison_packet,
            neutron_authority=neutron_authority_packet,
            comparator_uq=comparator_uq_packet,
            numerical_fidelity=numerical_fidelity_packet,
            certificate_gate=certificate_gate_packet,
            generalization=generalization_packet,
        )
        telemetry = {
            "status": ENGINEERING_CANDIDATE_STATUS,
            "run_mode": RUN_MODE,
            # ``source`` / ``architecture_source*`` are the equation-method
            # (hybrid-PIC) evidence; ``source_scope`` is the SELECTED-MACHINE
            # source scope.  They are deliberately separate (WS9-2, P0-2).
            "source": HYBRID_PIC_3D_SOURCE,
            "architecture_source": HYBRID_PIC_3D_SOURCE,
            "architecture_source_scope": geometry.architecture_source_scope,
            "architecture_evidence_role": geometry.architecture_evidence_role,
            "source_scope": deck.selected_machine_source_scope,
            "selected_machine_source_scope": deck.selected_machine_source_scope,
            "selected_machine_source_references": list(
                deck.selected_machine_source_references
            ),
            "startup": startup_packet,
            "deck_diff": deck_diff_packet,
            "limiter_readiness": limiter_readiness_packet,
            "experimental_limiter_zero_probe": limiter_zero_probe_packet,
            "boundary_policy": boundary_policy_packet,
            "pic_particle_loading": pic_loading_packet,
            "power_port": power_port_packet,
            "dimensionality_handoff": dimensionality_packet,
            "physics_closure": physics_closure_packet,
            "same_scope_source": same_scope_source_packet,
            # SS11-3 (audit S10-A3): other-scope hybrid-PIC architecture/schema
            # context lives under this clearly NON-``same_scope``-named sibling
            # field, never inside ``same_scope_source``.  It is architecture
            # context only and promotes nothing.
            "architecture_or_schema_context_sources": _architecture_context_sources(),
            "waveform_phase": waveform_phase_packet,
            "engineering_current_waveform_comparison": (
                current_waveform_comparison_packet
            ),
            "spatial_field_temperature": spatial_field_temperature_packet,
            "neutron_authority": neutron_authority_packet,
            "comparator_uq": comparator_uq_packet,
            "numerical_fidelity": numerical_fidelity_packet,
            "certificate_gate": certificate_gate_packet,
            "generalization": generalization_packet,
            "experimental_whole_shot": experimental_whole_shot_packet,
            "experimental_numerics": experimental_numerics_packet,
            "grid_shape": list(grid.shape),
            "grid_spacing_m": list(grid.spacing),
            "n_steps": deck.n_steps,
            "n_steps_completed": simulation.telemetry.n_steps_completed,
            "dt_s": deck.dt_s,
            "history_stride": deck.history_stride,
            "max_step_results": deck.max_step_results,
            "target_time_s": deck.target_time_s,
            "simulation": simulation.telemetry.to_dict(),
            "candidate_evidence": evidence,
            "hybrid_pic_3d_readiness": hybrid_pic_3d_readiness,
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


def build_first_principles_3d_session(
    deck: Mapping[str, Any] | object | None = None,
    *,
    circuit_boundary: CircuitMagneticBoundaryDrive | None = None,
) -> FirstPrinciples3DSession:
    """Build a reusable session for fixed-step split-continuation probes."""

    return FirstPrinciples3DSession.from_deck(
        deck,
        circuit_boundary=circuit_boundary,
    )


def _segment_circuit_udpf(
    values: float | tuple[float, ...],
    *,
    start: int,
    count: int,
) -> float | tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return float(array)
    segment = array[int(start) : int(start) + int(count)]
    if segment.shape != (int(count),):
        raise ValueError("circuit_udpf_V sequence does not cover segment")
    return tuple(float(item) for item in segment)


def _last_field_work_from_simulation(
    result: HybridPIC3DSimulationResult,
) -> dict[str, Any] | None:
    last_step = result.telemetry.last_step
    if not isinstance(last_step, Mapping):
        return None
    field_step = last_step.get("field_step")
    if not isinstance(field_step, Mapping):
        return None
    field_work = field_step.get("field_work")
    if not isinstance(field_work, Mapping):
        return None
    return dict(field_work)


def _restore_session_from_snapshot(
    session: FirstPrinciples3DSession,
    snapshot: FirstPrinciples3DSession,
) -> None:
    session.__dict__.clear()
    session.__dict__.update(deepcopy(snapshot.__dict__))


def _adaptive_retry_required(stop_reason: str) -> bool:
    return stop_reason in {
        "aborted_blocked_electron_energy_closure",
        "aborted_blocked_electron_heat_flux",
        "aborted_nonfinite_state",
    }


def _adaptive_rejection_record(
    *,
    result: HybridPIC3DSimulationResult,
    attempted_dt_s: float,
    elapsed_s: float,
    accepted_steps: int,
) -> dict[str, Any]:
    last_step = (
        result.telemetry.last_step
        if isinstance(result.telemetry.last_step, Mapping)
        else {}
    )
    electron_energy = (
        last_step.get("electron_energy") if isinstance(last_step, Mapping) else None
    )
    closure_validity = (
        electron_energy.get("closure_validity")
        if isinstance(electron_energy, Mapping)
        else None
    )
    return {
        "status": "candidate_adaptive_step_rejected_and_rolled_back",
        "attempted_dt_s": float(attempted_dt_s),
        "elapsed_s": float(elapsed_s),
        "accepted_steps_before_attempt": int(accepted_steps),
        "stop_reason": result.telemetry.stop_reason,
        "electron_energy_status": (
            None
            if not isinstance(electron_energy, Mapping)
            else electron_energy.get("status")
        ),
        "closure_validity": closure_validity,
        "finite_state": result.telemetry.finite_state,
        "can_support_first_principles_acceptance": False,
    }


def _empty_adaptive_limiter_summary() -> dict[str, Any]:
    return {
        "status": "candidate_adaptive_limiter_inventory_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "steps_observed": 0,
        "activation_counts": {
            "conductivity_ohmic_cfl_limited_steps": 0,
            "conductivity_density_blend_applied_steps": 0,
            "marder_dominant_correction_steps": 0,
            "electron_temperature_floor_contact_steps": 0,
            "blocked_heat_flux_steps": 0,
            "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps": 0,
            "marder_correction_steps": 0,
        },
        "max_observed": {
            "conductivity_cfl_limited_fraction": 0.0,
            "marder_relative_correction_linf": 0.0,
            "marder_residual_after_linf": 0.0,
            "marder_nondominance_threshold": None,
            "electron_temperature_min_K": None,
            "electron_temperature_max_K": None,
        },
        "acceptance_state": {
            "can_support_limiter_zero_acceptance": False,
            "can_support_first_principles_acceptance": False,
            "validated": False,
            "review_decision": "adaptive_runtime_inventory_only",
        },
    }


def _merge_adaptive_limiter_summary(
    accumulator: dict[str, Any],
    step_summary: Mapping[str, Any] | None,
) -> None:
    if not isinstance(step_summary, Mapping):
        return
    accumulator["steps_observed"] = int(accumulator.get("steps_observed", 0)) + int(
        step_summary.get("steps_observed", 0) or 0
    )
    counts = accumulator["activation_counts"]
    for name, value in _as_mapping(step_summary.get("activation_counts")).items():
        if name not in counts:
            counts[name] = 0
        counts[name] += int(value or 0)
    maxima = accumulator["max_observed"]
    observed = _as_mapping(step_summary.get("max_observed"))
    for name, value in observed.items():
        numeric = _optional_numeric(value)
        if numeric is None:
            continue
        if name in {"electron_temperature_min_K", "marder_nondominance_threshold"}:
            current = _optional_numeric(maxima.get(name))
            maxima[name] = numeric if current is None else min(current, numeric)
        else:
            current = _optional_numeric(maxima.get(name))
            maxima[name] = numeric if current is None else max(current, numeric)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _pressure_density_threshold_m3(deck: FirstPrinciples3DDeck) -> float:
    if deck.pressure_density_threshold_m3 is not None:
        return float(deck.pressure_density_threshold_m3)
    initial_electron_density = (
        deck.background_density_m3 * deck.initial_ionization_fraction
    )
    return max(1.0e12, initial_electron_density * 1.0e-6)


def _validate_deck(deck: FirstPrinciples3DDeck) -> None:
    if int(deck.n_steps) != deck.n_steps or deck.n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if int(deck.history_stride) != deck.history_stride or deck.history_stride <= 0:
        raise ValueError("history_stride must be a positive integer")
    if deck.max_step_results is not None and (
        int(deck.max_step_results) != deck.max_step_results
        or deck.max_step_results < 0
    ):
        raise ValueError("max_step_results must be a non-negative integer or None")
    if deck.target_time_s is not None and deck.target_time_s <= 0.0:
        raise ValueError("target_time_s must be positive")
    if deck.dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if deck.sigma0_S_m < 0.0:
        raise ValueError("sigma0_S_m must be non-negative")
    if deck.background_density_m3 <= 0.0:
        raise ValueError("background_density_m3 must be positive")
    if deck.density_floor_m3 <= 0.0:
        raise ValueError("density_floor_m3 must be positive")
    if not 0.0 <= deck.initial_ionization_fraction <= 1.0:
        raise ValueError("initial_ionization_fraction must be in [0, 1]")
    if (
        deck.pressure_density_threshold_m3 is not None
        and deck.pressure_density_threshold_m3 < 0.0
    ):
        raise ValueError("pressure_density_threshold_m3 must be non-negative")
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
    if deck.circuit_udpf_mode not in FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES:
        raise ValueError(
            "circuit_udpf_mode must be one of "
            f"{FIRST_PRINCIPLES_CIRCUIT_UDPF_MODES}"
        )
    if deck.circuit_feedback_min_current_A < 0.0:
        raise ValueError("circuit_feedback_min_current_A must be non-negative")
    if int(deck.pml_cells) != deck.pml_cells or deck.pml_cells < 0:
        raise ValueError("pml_cells must be a non-negative integer")
    if deck.pml_strength < 0.0:
        raise ValueError("pml_strength must be non-negative")
    if deck.conductor_mask_status not in {
        "not_supplied",
        "candidate_geometry_mask",
        "reviewed_same_scope_geometry_mask",
    }:
        raise ValueError("unknown conductor_mask_status")
    if deck.conductor_mask_mode not in {
        "none",
        "axisymmetric_coaxial_projection",
        "pf1000_rod_hollow_projection",
    }:
        raise ValueError("unknown conductor_mask_mode")
    if (
        deck.conductor_mask_mode
        in {"axisymmetric_coaxial_projection", "pf1000_rod_hollow_projection"}
        and deck.conductor_mask_status == "not_supplied"
    ):
        raise ValueError(
            f"{deck.conductor_mask_mode} requires conductor_mask_status"
        )
    _validate_reviewed_geometry_resolution(deck)


def _validate_reviewed_geometry_resolution(deck: FirstPrinciples3DDeck) -> None:
    """Reject a reviewed geometry-mask status on an under-resolved grid.

    A ``reviewed_same_scope_geometry_mask`` status asserts the projected mask
    resolves PF-1000 rods, the hollow anode, the insulator, and material
    surfaces.  Coarse Cartesian projections cannot honour that claim, so the
    status is refused before runtime when any required object is below its
    declared minimum resolution.  Coarse masks must instead stay at
    ``candidate_geometry_mask`` (see WP-N3, audit finding A-4).
    """

    if deck.conductor_mask_status != "reviewed_same_scope_geometry_mask":
        return

    grid = deck.grid()
    radial_cell = min(float(grid.dx), float(grid.dy))

    if (
        deck.conductor_mask_mode == "pf1000_rod_hollow_projection"
        or deck.device_cathode_rod_diameter_m is not None
    ):
        if deck.device_cathode_rod_diameter_m is None:
            raise ValueError(
                "reviewed_same_scope_geometry_mask requires "
                "device_cathode_rod_diameter_m to resolve cells across a rod "
                "diameter"
            )
        cells_per_rod_diameter = (
            float(deck.device_cathode_rod_diameter_m) / radial_cell
        )
        if cells_per_rod_diameter < _REVIEWED_MIN_CELLS_PER_ROD_DIAMETER:
            raise ValueError(
                "reviewed_same_scope_geometry_mask rejected: "
                f"{cells_per_rod_diameter:.3f} cells across a rod diameter is "
                f"below the declared minimum of "
                f"{_REVIEWED_MIN_CELLS_PER_ROD_DIAMETER:.1f}; the coarse rod "
                "projection cannot be a reviewed same-scope geometry mask"
            )

    if (
        deck.conductor_mask_mode == "pf1000_rod_hollow_projection"
        and deck.device_anode_inner_radius_m is None
    ):
        raise ValueError(
            "reviewed_same_scope_geometry_mask rejected: hollow-anode bore is "
            "unresolved because device_anode_inner_radius_m is not supplied"
        )

    if deck.device_insulator_material is not None and not (
        deck.device_insulator_length_m is not None
        and deck.device_insulator_outer_radius_m is not None
    ):
        raise ValueError(
            "reviewed_same_scope_geometry_mask rejected: insulator material "
            "surface is declared but not resolved as a material boundary "
            "region"
        )


def _build_initial_plasma_profile(
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
    *,
    conductor_cells: np.ndarray | None,
) -> InitialPlasmaProfile:
    """Build grid-shaped density, temperature, and drift startup fields."""

    total_density = np.full(grid.shape, float(deck.background_density_m3), dtype=float)
    ionization_fraction = np.full(
        grid.shape,
        float(deck.initial_ionization_fraction),
        dtype=float,
    )
    electron_temperature = np.full(
        grid.shape,
        float(deck.electron_temperature_K),
        dtype=float,
    )
    ion_temperature = np.full(
        grid.shape,
        float(deck.ion_temperature_K),
        dtype=float,
    )
    plasma_velocity = np.zeros(grid.shape + (3,), dtype=float)
    payload = deck.startup_payload if isinstance(deck.startup_payload, Mapping) else {}
    applied_regions: list[str] = []
    profile_status = "candidate_uniform_startup_profile_not_validation"

    if _payload_profile_type(payload) == "annular_axial_sheath":
        coordinates = _startup_profile_coordinates(deck, grid)
        radius = coordinates["radius"]
        axial = coordinates["z"]
        vacuum_density = _payload_float(
            payload,
            "vacuum_density_floor_m3",
            default=max(1.0, min(float(deck.background_density_m3), 1.0e12)),
        )
        total_density.fill(vacuum_density)
        ionization_fraction.fill(
            _payload_float(
                payload,
                "vacuum_ionization_fraction",
                default=0.0,
            )
        )
        electron_temperature.fill(
            _payload_float(
                payload,
                "vacuum_temperature_K",
                default=float(deck.electron_temperature_K),
            )
        )
        ion_temperature.fill(
            _payload_float(
                payload,
                "vacuum_temperature_K",
                default=float(deck.ion_temperature_K),
            )
        )

        if _payload_region_declared(payload, "background"):
            background_mask = _annular_axial_mask(
                radius=radius,
                axial=axial,
                r_min=_payload_float(payload, "background_radial_min_m", default=0.0),
                r_max=_payload_float(payload, "background_radial_max_m", default=None),
                z_min=_payload_float(payload, "background_z_min_m", default=None),
                z_max=_payload_float(payload, "background_z_max_m", default=None),
            )
            if np.any(background_mask):
                total_density[background_mask] = _payload_float(
                    payload,
                    "background_density_m3",
                    default=float(deck.background_density_m3),
                )
                ionization_fraction[background_mask] = _payload_float(
                    payload,
                    "background_ionization_fraction",
                    default=_payload_float(payload, "ionization_fraction", default=1.0),
                )
                background_temperature = _payload_float(
                    payload,
                    "background_temperature_K",
                    default=float(deck.ion_temperature_K),
                )
                electron_temperature[background_mask] = background_temperature
                ion_temperature[background_mask] = background_temperature
                applied_regions.append("background_prefill")

        sheath_mask = _annular_axial_mask(
            radius=radius,
            axial=axial,
            r_min=_payload_float(payload, "sheath_radial_min_m", default=None),
            r_max=_payload_float(payload, "sheath_radial_max_m", default=None),
            z_min=_payload_float(payload, "sheath_z_min_m", default=None),
            z_max=_payload_float(payload, "sheath_z_max_m", default=None),
        )
        if np.any(sheath_mask):
            total_density[sheath_mask] = _payload_float(
                payload,
                "sheath_density_m3",
                default=float(deck.background_density_m3),
            )
            ionization_fraction[sheath_mask] = _payload_float(
                payload,
                "sheath_ionization_fraction",
                default=_payload_float(payload, "ionization_fraction", default=1.0),
            )
            sheath_temperature = _payload_float(
                payload,
                "sheath_temperature_K",
                default=float(deck.ion_temperature_K),
            )
            electron_temperature[sheath_mask] = sheath_temperature
            ion_temperature[sheath_mask] = sheath_temperature
            drift = _payload_vector(
                payload,
                "sheath_drift_velocity_m_s",
                default=(0.0, 0.0, 0.0),
            )
            plasma_velocity[sheath_mask, :] = np.asarray(drift, dtype=float)
            applied_regions.append("preaccelerated_current_sheath")
        profile_status = "candidate_source_backed_annular_sheath_profile_not_validation"

    if conductor_cells is not None:
        conductor_mask = np.asarray(conductor_cells, dtype=bool)
        total_density = np.where(conductor_mask, 1.0, total_density)
        ionization_fraction = np.where(conductor_mask, 0.0, ionization_fraction)
        plasma_velocity = np.where(conductor_mask[..., np.newaxis], 0.0, plasma_velocity)

    ionization_fraction = np.clip(ionization_fraction, 0.0, 1.0)
    telemetry = {
        "status": profile_status,
        "startup_mode": deck.startup_mode,
        "startup_source_scope": deck.startup_source_scope,
        "profile_type": _payload_profile_type(payload) or "uniform",
        "applied_regions": applied_regions,
        "total_density_min_m3": float(np.min(total_density)),
        "total_density_max_m3": float(np.max(total_density)),
        "ionization_fraction_min": float(np.min(ionization_fraction)),
        "ionization_fraction_max": float(np.max(ionization_fraction)),
        "electron_temperature_min_K": float(np.min(electron_temperature)),
        "electron_temperature_max_K": float(np.max(electron_temperature)),
        "ion_temperature_min_K": float(np.min(ion_temperature)),
        "ion_temperature_max_K": float(np.max(ion_temperature)),
        "max_abs_plasma_drift_m_s": float(np.max(np.abs(plasma_velocity))),
        "source_references": payload.get("source_references", ()),
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Runtime startup profile only; not an accepted breakdown or liftoff BVP.",
            "Annular axial profile is a Cartesian projection of source geometry.",
            "Surface flashover, Debye sheaths, molecular D2, and electrode material ablation remain upstream blockers.",
        ],
    }
    return InitialPlasmaProfile(
        total_deuterium_density_m3=total_density,
        ionization_fraction=ionization_fraction,
        electron_temperature_K=electron_temperature,
        ion_temperature_K=ion_temperature,
        plasma_velocity_m_s=plasma_velocity,
        telemetry=telemetry,
    )


def _build_initial_pic(
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
    conductor_cells: np.ndarray | None,
    *,
    plasma_profile: InitialPlasmaProfile | None = None,
) -> tuple[HybridPIC, dict[str, Any]]:
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=deck.dt_s,
        use_esirkepov=True,
        use_binary_collisions=False,
    )
    active_cells = _initial_pic_active_cells(
        grid,
        conductor_cells=conductor_cells,
        pml_cells=deck.pml_cells,
    )
    conductor_mask = (
        np.zeros(grid.shape, dtype=bool)
        if conductor_cells is None
        else np.asarray(conductor_cells, dtype=bool)
    )
    pml_mask = _initial_pic_pml_mask(grid, deck.pml_cells)
    indices = np.argwhere(active_cells)
    spacings = np.array(grid.spacing, dtype=float)
    cell_centers = (indices.astype(float) + 0.5) * spacings
    if plasma_profile is None:
        plasma_profile = _build_initial_plasma_profile(
            deck,
            grid,
            conductor_cells=conductor_cells,
        )
    ion_density_m3 = (
        plasma_profile.total_deuterium_density_m3
        * plasma_profile.ionization_fraction
    )
    active_ion_density = ion_density_m3[active_cells]
    active_ion_temperature = plasma_profile.ion_temperature_K[active_cells]
    active_drift_velocity = plasma_profile.plasma_velocity_m_s[active_cells]
    physical_ions_per_cell = active_ion_density * grid.cell_volume
    thermal_speed_m_s = np.sqrt(
        3.0 * K_B * np.maximum(active_ion_temperature, 1.0) / float(deck.ion_mass_kg)
    )
    velocity_basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    quadrature_count = int(velocity_basis.shape[0])
    loaded = physical_ions_per_cell > 0.0
    if cell_centers.size and np.any(loaded):
        loaded_centers = cell_centers[loaded]
        loaded_speeds = thermal_speed_m_s[loaded]
        loaded_drift = active_drift_velocity[loaded]
        positions = np.repeat(loaded_centers, quadrature_count, axis=0)
        velocities = (
            np.repeat(loaded_drift, quadrature_count, axis=0)
            + np.vstack([
                velocity_basis * float(speed)
                for speed in loaded_speeds
            ])
        )
        weights = np.repeat(
            physical_ions_per_cell[loaded] / quadrature_count,
            quadrature_count,
        ).astype(float, copy=False)
    else:
        positions = np.empty((0, 3), dtype=float)
        velocities = np.empty((0, 3), dtype=float)
        weights = np.empty((0,), dtype=float)
    pic.add_species(
        deck.ion_species_name,
        deck.ion_mass_kg,
        deck.ion_charge_C,
        positions=positions,
        velocities=velocities,
        weights=weights,
    )
    represented_ions = float(np.sum(weights))
    active_count = int(indices.shape[0])
    total_cells = int(np.prod(grid.shape))
    loading_packet = {
        "status": "candidate_density_normalized_pic_loading_not_validation",
        "source": (
            "KnowledgeReference/"
            "the-vlasov-two-fluid-and-mhd-models-of-plasma-dynamics.md:1028-1060"
        ),
        "architecture_source": HYBRID_PIC_3D_SOURCE,
        "implementation": "src/dpf/first_principles/runner.py",
        "loading_policy": (
            "six_stream_zero_mean_thermal_moment_quadrature_per_active_cell"
        ),
        "grid_shape": list(grid.shape),
        "total_cells": total_cells,
        "active_loaded_cells": active_count,
        "active_cells_with_positive_ion_weight": int(np.count_nonzero(loaded)),
        "macroparticles_loaded": int(weights.size),
        "velocity_quadrature_directions_per_cell": quadrature_count,
        "ion_thermal_speed_m_s": (
            float(thermal_speed_m_s[0])
            if thermal_speed_m_s.size
            and float(np.min(thermal_speed_m_s)) == float(np.max(thermal_speed_m_s))
            else None
        ),
        "ion_thermal_speed_min_m_s": (
            float(np.min(thermal_speed_m_s)) if thermal_speed_m_s.size else 0.0
        ),
        "ion_thermal_speed_max_m_s": (
            float(np.max(thermal_speed_m_s)) if thermal_speed_m_s.size else 0.0
        ),
        "excluded_conductor_cells": int(np.count_nonzero(conductor_mask)),
        "excluded_pml_cells": int(np.count_nonzero(pml_mask & ~conductor_mask)),
        "initial_total_deuterium_density_m3": (
            float(plasma_profile.telemetry["total_density_max_m3"])
            if plasma_profile.telemetry["total_density_min_m3"]
            == plasma_profile.telemetry["total_density_max_m3"]
            else None
        ),
        "initial_total_deuterium_density_min_m3": plasma_profile.telemetry[
            "total_density_min_m3"
        ],
        "initial_total_deuterium_density_max_m3": plasma_profile.telemetry[
            "total_density_max_m3"
        ],
        "initial_ionization_fraction": (
            float(plasma_profile.telemetry["ionization_fraction_max"])
            if plasma_profile.telemetry["ionization_fraction_min"]
            == plasma_profile.telemetry["ionization_fraction_max"]
            else None
        ),
        "initial_ionization_fraction_min": plasma_profile.telemetry[
            "ionization_fraction_min"
        ],
        "initial_ionization_fraction_max": plasma_profile.telemetry[
            "ionization_fraction_max"
        ],
        "initial_ion_density_m3": (
            float(np.max(ion_density_m3))
            if float(np.min(ion_density_m3)) == float(np.max(ion_density_m3))
            else None
        ),
        "initial_ion_density_min_m3": float(np.min(ion_density_m3)),
        "initial_ion_density_max_m3": float(np.max(ion_density_m3)),
        "cell_volume_m3": float(grid.cell_volume),
        "macro_particle_weight_min": float(np.min(weights)) if weights.size else 0.0,
        "macro_particle_weight_max": float(np.max(weights)) if weights.size else 0.0,
        "represented_physical_ions": represented_ions,
        "initial_plasma_profile": plasma_profile.telemetry,
        "nominal_deck_particle_weight": float(deck.particle_weight),
        "particle_weight_policy": (
            "runtime weights are density times cell volume so the initial ion "
            "macroparticles conserve the deck ion density; six-stream velocity "
            "quadrature preserves the configured drift and isotropic thermal "
            "second moment per loaded cell"
        ),
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Cell-centered six-stream loading is a deterministic engineering PIC initialization, not an accepted startup BVP.",
            "The six-stream quadrature matches thermal moments but is not a sampled or reviewed DPF startup distribution.",
            "Molecular deuterium, preionization gradients, sheath liftoff, and surface flashover remain blocked upstream.",
        ],
    }
    return pic, loading_packet


def _payload_profile_type(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("profile_type", payload.get("initial_plasma_profile"))
    if value is None:
        return None
    return str(value)


def _payload_float(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: float | None,
) -> float | None:
    value = payload.get(key, default)
    if value is None:
        return None
    return float(value)


def _payload_vector(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    value = payload.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be a 3-vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _payload_region_declared(payload: Mapping[str, Any], prefix: str) -> bool:
    return any(str(key).startswith(f"{prefix}_") for key in payload)


def _startup_profile_coordinates(
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
) -> dict[str, np.ndarray]:
    z = (np.arange(grid.nz, dtype=float) + 0.5) * grid.dz
    axial = z[np.newaxis, np.newaxis, :]
    if deck.conductor_mask_mode == "pf1000_rod_hollow_projection":
        x = (np.arange(grid.nx, dtype=float) - 0.5 * (grid.nx - 1)) * grid.dx
        y = (np.arange(grid.ny, dtype=float) - 0.5 * (grid.ny - 1)) * grid.dy
    else:
        x = (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
        y = (np.arange(grid.ny, dtype=float) + 0.5) * grid.dy
    radius = np.sqrt(
        x[:, np.newaxis, np.newaxis] ** 2
        + y[np.newaxis, :, np.newaxis] ** 2
    )
    return {
        "radius": np.broadcast_to(radius, grid.shape),
        "z": np.broadcast_to(axial, grid.shape),
    }


def _annular_axial_mask(
    *,
    radius: np.ndarray,
    axial: np.ndarray,
    r_min: float | None,
    r_max: float | None,
    z_min: float | None,
    z_max: float | None,
) -> np.ndarray:
    mask = np.ones(radius.shape, dtype=bool)
    if r_min is not None:
        mask &= radius >= float(r_min)
    if r_max is not None:
        mask &= radius <= float(r_max)
    if z_min is not None:
        mask &= axial >= float(z_min)
    if z_max is not None:
        mask &= axial <= float(z_max)
    return mask


def _initial_pic_active_cells(
    grid: Maxwell3DGrid,
    *,
    conductor_cells: np.ndarray | None,
    pml_cells: int,
) -> np.ndarray:
    active = np.ones(grid.shape, dtype=bool)
    if conductor_cells is not None:
        active &= ~np.asarray(conductor_cells, dtype=bool)
    active &= ~_initial_pic_pml_mask(grid, pml_cells)
    return active


def _initial_pic_pml_mask(grid: Maxwell3DGrid, pml_cells: int) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    p = int(pml_cells)
    if p > 0:
        mask[:p, :, :] = True
        mask[-p:, :, :] = True
        mask[:, :p, :] = True
        mask[:, -p:, :] = True
        mask[:, :, :p] = True
        mask[:, :, -p:] = True
    return mask


def _architecture_context_sources() -> dict[str, Any]:
    """Return the non-same-scope architecture/schema-context source block.

    SS11-3 (audit S10-A3): the LLNL-like hybrid-PIC paper is other-scope
    architecture / equation-method / schema-context evidence.  It is emitted
    here under a clearly NON-``same_scope``-named runtime field so it never
    blurs selected-machine same-scope source evidence.  It promotes nothing.
    """
    return {
        "status": "architecture_or_schema_context_only_not_same_scope",
        "role": "other_scope_architecture_and_schema_context_sources",
        "usable_for": "architecture_and_closure_gap_requirements_or_schema_only",
        "is_same_scope_validation_evidence": False,
        "source_references": list(ARCHITECTURE_OR_SCHEMA_CONTEXT_SOURCES),
        "can_support_first_principles_acceptance": False,
    }


def _candidate_evidence(
    *,
    geometry: HybridPICSourceGeometry,
    simulation: HybridPIC3DSimulationResult,
    conservation: dict[str, Any],
    boundary_policy: Mapping[str, Any],
    pic_loading: Mapping[str, Any],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "true_3d_dimensionality": hybrid_simulator_candidate_evidence(
            simulation.telemetry
        ),
        # SS10-1 (closes audit A1): the LLNL-like hybrid-PIC 3-D geometry packet
        # carries ``architecture_source_scope=llnl_like_180ka_axisymmetric_
        # hybrid_pic``.  It is architecture / equation-method evidence only, so
        # it is emitted under an architecture-only key.  A ``same_scope``-named
        # key must NEVER carry the LLNL-like architecture scope.
        "architecture_3d_geometry_candidate_packet": (
            source_geometry_candidate_evidence(geometry)
        ),
        "conservation_telemetry": conservation,
        "field_particle_boundary_policy": _candidate_record(
            capability="field_particle_boundary_policy",
            implementation="src/dpf/first_principles/runner.py",
            telemetry=boundary_policy,
        ),
        "density_normalized_pic_particle_loading": _candidate_record(
            capability="density_normalized_pic_particle_loading",
            implementation="src/dpf/first_principles/runner.py",
            telemetry=pic_loading,
        ),
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
        if last_step.telemetry.particle_boundaries is not None:
            evidence["pml_conductor_particle_boundaries"] = _candidate_record(
                capability="pml_conductor_particle_boundaries",
                implementation="src/dpf/fields/particle_boundaries.py",
                telemetry=last_step.telemetry.particle_boundaries,
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
    architecture_source: str,
    architecture_source_scope: str,
    selected_machine_source_references: tuple[str, ...],
    hybrid_pic_3d_evidence: Mapping[str, Any],
    hybrid_pic_3d_readiness: Mapping[str, Any],
    conservation_evidence: Mapping[str, Any],
    startup_bvp: Mapping[str, Any],
    limiter_readiness: Mapping[str, Any],
    dimensionality_handoff: Mapping[str, Any],
    same_scope_source: Mapping[str, Any],
    waveform_phase: Mapping[str, Any],
    spatial_field_temperature: Mapping[str, Any],
    current_waveform_comparison: Mapping[str, Any],
    neutron_authority: Mapping[str, Any],
    comparator_uq: Mapping[str, Any],
    numerical_fidelity: Mapping[str, Any],
    certificate_gate: Mapping[str, Any],
    generalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize candidate evidence without invoking the validation workflow.

    ``source_scope`` is the SELECTED-MACHINE source scope (device + operating
    point).  ``architecture_source`` / ``architecture_source_scope`` carry the
    separate equation-method (hybrid-PIC) evidence and must NOT be conflated
    with the selected-machine scope.  Super-Sprint 9 WS9-2 (fixes audit P0-2).
    """

    evidence_keys = sorted(str(key) for key in hybrid_pic_3d_evidence)
    return {
        "status": "not_validation",
        "scientific_status": ENGINEERING_CANDIDATE_STATUS,
        "geometry_dimensionality": geometry_dimensionality,
        "source_scope": source_scope,
        "architecture_source": architecture_source,
        "architecture_source_scope": architecture_source_scope,
        "architecture_evidence_role": (
            "equation_method_and_architecture_source"
        ),
        "selected_machine_source_references": list(
            selected_machine_source_references
        ),
        "candidate_evidence_keys": evidence_keys,
        "hybrid_pic_3d_readiness_status": hybrid_pic_3d_readiness.get("status"),
        "hybrid_pic_3d_missing_capabilities": hybrid_pic_3d_readiness.get(
            "missing_capabilities",
            (),
        ),
        "hybrid_pic_3d_satisfied_capabilities": hybrid_pic_3d_readiness.get(
            "satisfied_capabilities",
            (),
        ),
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
        "engineering_current_waveform_comparison_status": (
            current_waveform_comparison.get("status")
        ),
        "engineering_current_waveform_comparison_missing_for_acceptance": (
            current_waveform_comparison.get("missing_for_acceptance", ())
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
    capacitor_voltage_V = params.voltage_V - circuit_state.charge_C / params.capacitance_F
    return float(
        0.5 * params.inductance_H * circuit_state.current_A**2
        + 0.5 * params.capacitance_F * capacitor_voltage_V**2
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
        "finite_state": finite,
        "energy_conservation_assessed": "not_assessed_no_accepted_tolerance",
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


def _deck_source_diff_packet(deck: FirstPrinciples3DDeck) -> dict[str, Any]:
    """Compare a source-locked demonstrator deck against local source values."""

    observed = {
        "device_anode_radius_m": deck.device_anode_radius_m,
        "device_cathode_radius_m": deck.device_cathode_radius_m,
        "device_anode_length_m": deck.device_anode_length_m,
        "device_insulator_length_m": deck.device_insulator_length_m,
        "device_cathode_rod_count": deck.device_cathode_rod_count,
        "device_cathode_rod_diameter_m": deck.device_cathode_rod_diameter_m,
        "circuit_capacitance_F": deck.circuit_capacitance_F,
        "circuit_voltage_V": deck.circuit_voltage_V,
        "circuit_inductance_H": deck.circuit_inductance_H,
        "circuit_resistance_ohm": deck.circuit_resistance_ohm,
        "gas_pressure_Pa": deck.gas_pressure_Pa,
    }
    is_pf1000_akel = (
        deck.validation_scope
        == "pf1000_akel_16kv_1p2torr_shot_12581_engineering_candidate"
        or deck.device_name.startswith("PF-1000/Akel")
    )
    if not is_pf1000_akel:
        return {
            "status": "candidate_deck_diff_not_applicable",
            "declared_scope": deck.validation_scope,
            "device_name": deck.device_name,
            "observed": observed,
            "source_references": [],
            "can_support_first_principles_acceptance": False,
        }

    comparisons = {
        key: _deck_value_comparison(
            observed=observed.get(key),
            expected=expected,
            tolerance=_deck_value_tolerance(expected),
        )
        for key, expected in PF1000_AKEL_SOURCE_LOCKED_DECK.items()
    }
    mismatch_keys = [
        key
        for key, comparison in comparisons.items()
        if comparison["status"] != "source_locked_match_not_validation"
    ]
    return {
        "status": (
            "candidate_source_locked_deck_match_not_validation"
            if not mismatch_keys
            else "blocked_source_deck_drift_not_validation"
        ),
        "declared_scope": deck.validation_scope,
        "device_name": deck.device_name,
        "deck_lock": "pf1000_akel_16kv_1p2torr_shot_12581",
        "observed": observed,
        "expected": dict(PF1000_AKEL_SOURCE_LOCKED_DECK),
        "comparisons": comparisons,
        "mismatch_keys": mismatch_keys,
        "source_references": list(PF1000_AKEL_DECK_SOURCE_REFS),
        "scope_policy": (
            "PF-1000/Akel 16 kV values are not interchangeable with PF-1000U "
            "or full-energy PF-1000 shots without an explicit transfer packet."
        ),
        "can_support_first_principles_acceptance": False,
    }


def _deck_value_comparison(
    *,
    observed: Any,
    expected: float | int,
    tolerance: float,
) -> dict[str, Any]:
    if observed is None:
        return {
            "observed": None,
            "expected": expected,
            "absolute_error": None,
            "tolerance": tolerance,
            "status": "missing_source_locked_value_not_validation",
        }
    if isinstance(expected, int):
        matches = int(observed) == int(expected)
        error = abs(int(observed) - int(expected))
    else:
        error = abs(float(observed) - float(expected))
        matches = error <= tolerance
    return {
        "observed": observed,
        "expected": expected,
        "absolute_error": error,
        "tolerance": tolerance,
        "status": (
            "source_locked_match_not_validation"
            if matches
            else "source_locked_mismatch_not_validation"
        ),
    }


def _deck_value_tolerance(expected: float | int) -> float:
    if isinstance(expected, int):
        return 0.0
    return max(abs(float(expected)) * 1.0e-12, 1.0e-15)


def _boundary_values_from_policy(value: Mapping[str, Any] | object) -> dict[str, Any]:
    return {
        "pml_cells": int(_get(value, "pml_cells", 0)),
        "pml_strength": float(_get(value, "pml_strength", 0.0)),
        "particle_absorption_enabled": bool(
            _get(value, "particle_absorption_enabled", False)
        ),
        "open_boundary": bool(_get(value, "open_boundary", True)),
        "conductor_cells": _get(value, "conductor_cells", None),
        "conductor_mask_status": str(_get(value, "conductor_mask_status", "not_supplied")),
        "conductor_mask_mode": str(_get(value, "conductor_mask_mode", "none")),
        # SS10-2 (closes audit A2): carry the blocked geometry fields onto the
        # 3-D deck so conductor-mask telemetry can expose every blocked field.
        "blocked_geometry_fields": _blocked_geometry_fields_to_mappings(
            _get(value, "blocked_geometry_fields", ())
        ),
    }


def _blocked_geometry_fields_to_mappings(
    value: Any,
) -> tuple[dict[str, Any], ...]:
    """Normalize blocked geometry fields to plain mappings.

    Accepts ``BlockedGeometryField`` dataclasses, plain mappings, or the
    ``asdict`` round-trip output, and returns a stable tuple of plain dicts so
    the 3-D deck and its telemetry never depend on the deck-module type.
    SS10-2 (closes audit A2).
    """

    mappings: list[dict[str, Any]] = []
    for item in value or ():
        if isinstance(item, Mapping):
            field_name = item.get("field_name")
            blocker_id = item.get("blocker_id", "")
            blocked = item.get("blocked", True)
            reason = item.get(
                "source_scope_reason",
                "no_same_scope_kr_source_for_selected_scope",
            )
        else:
            field_name = getattr(item, "field_name", None)
            blocker_id = getattr(item, "blocker_id", "")
            blocked = getattr(item, "blocked", True)
            reason = getattr(
                item,
                "source_scope_reason",
                "no_same_scope_kr_source_for_selected_scope",
            )
        if field_name is None:
            continue
        mappings.append(
            {
                "field_name": str(field_name),
                "blocker_id": str(blocker_id or ""),
                "blocked": bool(blocked),
                "source_scope_reason": str(reason),
            }
        )
    return tuple(mappings)


def _plasmapy_reference_state(deck: FirstPrinciples3DDeck) -> dict[str, float | str]:
    startup_B = abs(float(deck.initial_B_z_T))
    current_A = 0.0 if deck.circuit_state is None else float(deck.circuit_state.current_A)
    anode_radius_m = (
        float(deck.device_anode_radius_m)
        if deck.device_anode_radius_m is not None
        else float((deck.grid_spacing_m or (1.0e-3, 1.0e-3, 1.0e-3))[0])
    )
    circuit_B = (
        MU_0 * abs(current_A) / (2.0 * np.pi * max(anode_radius_m, 1.0e-12))
    )
    return {
        "electron_density_m3": float(deck.background_density_m3),
        "electron_temperature_K": float(deck.electron_temperature_K),
        "magnetic_field_T": startup_B if startup_B > 0.0 else max(circuit_B, 1.0e-9),
        "mass_density_kg_m3": float(deck.background_density_m3)
        * float(deck.ion_mass_kg),
        "ion": "D+",
    }


def _boundary_policy_manifest(deck: FirstPrinciples3DDeck) -> dict[str, Any]:
    conductor_cells = None
    if deck.conductor_cells is not None:
        conductor_cells = np.asarray(deck.conductor_cells, dtype=bool)
    return {
        "status": "candidate_engineering_boundary_policy_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "613-619, 625-628",
        "pml_cells": int(deck.pml_cells),
        "pml_strength": float(deck.pml_strength),
        "particle_absorption_enabled": bool(deck.particle_absorption_enabled),
        "open_boundary": bool(deck.open_boundary),
        "conductor_mask_status": deck.conductor_mask_status,
        "conductor_mask_mode": deck.conductor_mask_mode,
        "conductor_cells_active": (
            0 if conductor_cells is None else int(np.count_nonzero(conductor_cells))
        ),
        "conductor_mask_supplied": conductor_cells is not None,
        "can_support_first_principles_acceptance": False,
    }


def _coerce_conductor_cells(
    value: Any,
    grid: Maxwell3DGrid,
) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool)
    if mask.shape != grid.shape:
        raise ValueError("conductor_cells must match grid shape")
    return mask


def _resolve_conductor_cells(
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    supplied = _coerce_conductor_cells(deck.conductor_cells, grid)
    if supplied is not None:
        return supplied, _conductor_mask_packet(
            deck=deck,
            grid=grid,
            mask=supplied,
            source="supplied_conductor_cells",
        )
    if deck.conductor_mask_mode == "none":
        return None, _conductor_mask_packet(
            deck=deck,
            grid=grid,
            mask=None,
            source="not_supplied",
        )
    if deck.conductor_mask_mode == "axisymmetric_coaxial_projection":
        mask = _axisymmetric_coaxial_conductor_mask(deck, grid)
        return mask, _conductor_mask_packet(
            deck=deck,
            grid=grid,
            mask=mask,
            source="candidate_axisymmetric_coaxial_projection",
        )
    if deck.conductor_mask_mode == "pf1000_rod_hollow_projection":
        mask = _pf1000_rod_hollow_conductor_mask(deck, grid)
        return mask, _conductor_mask_packet(
            deck=deck,
            grid=grid,
            mask=mask,
            source="candidate_pf1000_rod_hollow_projection",
        )
    raise ValueError("unknown conductor_mask_mode")


def _axisymmetric_coaxial_conductor_mask(
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
) -> np.ndarray:
    if deck.device_anode_radius_m is None:
        raise ValueError("device_anode_radius_m is required for conductor mask")
    if deck.device_cathode_radius_m is None:
        raise ValueError("device_cathode_radius_m is required for conductor mask")
    if deck.device_anode_length_m is None:
        raise ValueError("device_anode_length_m is required for conductor mask")

    x = (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
    y = (np.arange(grid.ny, dtype=float) + 0.5) * grid.dy
    z = (np.arange(grid.nz, dtype=float) + 0.5) * grid.dz
    radius = np.sqrt(x[:, np.newaxis, np.newaxis] ** 2 + y[np.newaxis, :, np.newaxis] ** 2)
    axial = z[np.newaxis, np.newaxis, :]
    anode = (radius <= float(deck.device_anode_radius_m)) & (
        axial <= float(deck.device_anode_length_m)
    )
    cathode = radius >= float(deck.device_cathode_radius_m)
    return np.broadcast_to(anode | cathode, grid.shape).copy()


def _pf1000_rod_hollow_conductor_mask(
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
) -> np.ndarray:
    if deck.device_anode_radius_m is None:
        raise ValueError("device_anode_radius_m is required for conductor mask")
    if deck.device_cathode_radius_m is None:
        raise ValueError("device_cathode_radius_m is required for conductor mask")
    if deck.device_anode_length_m is None:
        raise ValueError("device_anode_length_m is required for conductor mask")
    if deck.device_cathode_rod_count is None:
        raise ValueError("device_cathode_rod_count is required for PF-1000 rod mask")
    if deck.device_cathode_rod_diameter_m is None:
        raise ValueError("device_cathode_rod_diameter_m is required for PF-1000 rod mask")

    x = (np.arange(grid.nx, dtype=float) - 0.5 * (grid.nx - 1)) * grid.dx
    y = (np.arange(grid.ny, dtype=float) - 0.5 * (grid.ny - 1)) * grid.dy
    z = (np.arange(grid.nz, dtype=float) + 0.5) * grid.dz
    xx = x[:, np.newaxis, np.newaxis]
    yy = y[np.newaxis, :, np.newaxis]
    axial = z[np.newaxis, np.newaxis, :]
    radius = np.sqrt(xx**2 + yy**2)

    anode_outer_radius = float(deck.device_anode_radius_m)
    anode_inner_radius = (
        0.0
        if deck.device_anode_inner_radius_m is None
        else float(deck.device_anode_inner_radius_m)
    )
    anode = (
        (radius <= anode_outer_radius)
        & (radius >= anode_inner_radius)
        & (axial <= float(deck.device_anode_length_m))
    )

    rod_radius = 0.5 * float(deck.device_cathode_rod_diameter_m)
    rod_center_radius = float(deck.device_cathode_radius_m) + rod_radius
    rod_length = (
        float(deck.device_cathode_rod_length_m)
        if deck.device_cathode_rod_length_m is not None
        else float(deck.device_anode_length_m)
    )
    cathode = np.zeros(grid.shape, dtype=bool)
    for index in range(int(deck.device_cathode_rod_count)):
        angle = 2.0 * np.pi * index / int(deck.device_cathode_rod_count)
        center_x = rod_center_radius * np.cos(angle)
        center_y = rod_center_radius * np.sin(angle)
        rod_radius_field = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        cathode |= (rod_radius_field <= rod_radius) & (axial <= rod_length)

    return np.broadcast_to(anode | cathode, grid.shape).copy()


# Declared minimum resolution criterion for a reviewed PF-1000 rod mask.
# Source dimensions: Krauz 2012 PF-1000 cathode rods are 80 mm diameter
# (KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-
# current-sheath-on-the-pf-1000-facility-705bcc83.md:344-347) and Akel 2021
# states twelve 8-cm-diameter stainless-steel cathode tubes
# (KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:111-117).
# A discrete circular rod cross-section cannot be represented on a Cartesian
# grid below about two cells across its diameter; a reviewed-status mask is
# rejected before runtime when any object falls below this.
_REVIEWED_MIN_CELLS_PER_ROD_DIAMETER = 2.0


def _conductor_mask_sha256(mask: np.ndarray | None, grid: Maxwell3DGrid) -> str:
    """Return a deterministic SHA256 over the conductor mask occupancy.

    The hash binds grid shape and the raw boolean occupancy bytes so two runs
    with the same projected geometry produce an identical, comparable digest.
    """

    digest = hashlib.sha256()
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    if mask is None:
        digest.update(b"no_conductor_mask")
    else:
        contiguous = np.ascontiguousarray(np.asarray(mask, dtype=bool))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _conductor_mask_projection_error(
    *,
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
    mask: np.ndarray | None,
    is_pf1000_rod_projection: bool,
) -> dict[str, Any]:
    """Projection-error packet: discretized mask deviation from source dimensions.

    Source dimensions are the KnowledgeReference PF-1000/Akel device values
    carried on the deck (anode/cathode radii, anode length, rod diameter from
    KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:111-117,
    262-268 and KnowledgeReference/experimental-study-of-the-structure-of-the-
    plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:344-351).  The
    discretization error is the half-cell quantization a Cartesian projection
    introduces relative to those continuous source surfaces.
    """

    dx = float(grid.dx)
    dy = float(grid.dy)
    dz = float(grid.dz)
    radial_cell = min(dx, dy)
    # A continuous radial surface is quantized to the nearest cell face: the
    # worst-case radial placement error of any electrode/rod surface is one
    # half-cell, and the diameter error (two surfaces) is one full cell.
    max_radial_discretization_error_m = radial_cell
    max_axial_discretization_error_m = dz

    cells_per_rod_diameter: float | None = None
    if is_pf1000_rod_projection and deck.device_cathode_rod_diameter_m is not None:
        cells_per_rod_diameter = (
            float(deck.device_cathode_rod_diameter_m) / radial_cell
        )

    cells_per_anode_radius: float | None = None
    if deck.device_anode_radius_m is not None:
        cells_per_anode_radius = float(deck.device_anode_radius_m) / radial_cell

    cells_per_anode_length: float | None = None
    if deck.device_anode_length_m is not None:
        cells_per_anode_length = float(deck.device_anode_length_m) / dz

    insulator_length_m = (
        deck.device_insulator_length_m
        if deck.device_insulator_length_m is not None
        else deck.device_insulator_outer_radius_m
    )
    cells_per_insulator_length: float | None = None
    if insulator_length_m is not None:
        cells_per_insulator_length = float(insulator_length_m) / dz

    return {
        "status": "candidate_projection_error_metrics_not_validation",
        "mask_sha256": _conductor_mask_sha256(mask, grid),
        "source_dimension_basis": (
            "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633"
            ".md:111-117,262-268; KnowledgeReference/experimental-study-of-the-"
            "structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-"
            "705bcc83.md:344-351"
        ),
        "radial_cell_size_m": radial_cell,
        "axial_cell_size_m": dz,
        "max_radial_discretization_error_m": max_radial_discretization_error_m,
        "max_axial_discretization_error_m": max_axial_discretization_error_m,
        "cells_per_rod_diameter": cells_per_rod_diameter,
        "cells_per_anode_radius": cells_per_anode_radius,
        "cells_per_anode_length": cells_per_anode_length,
        "cells_per_insulator_length": cells_per_insulator_length,
        "reviewed_min_cells_per_rod_diameter": _REVIEWED_MIN_CELLS_PER_ROD_DIAMETER,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Discretization error is a half-cell quantization estimate, not a "
            "reviewed convergence study against the source geometry.",
            "Cartesian projection of axisymmetric/rod surfaces is an "
            "engineering candidate, not an accepted same-scope geometry mask.",
        ],
    }


def _conductor_mask_mesh_resolution_warning(
    *,
    is_pf1000_rod_projection: bool,
    cells_per_rod_diameter: float | None,
) -> dict[str, Any]:
    """Build the explicit cathode-rod mesh-resolution warning packet.

    A discrete circular cathode rod cannot be represented on a Cartesian grid
    below about :data:`_REVIEWED_MIN_CELLS_PER_ROD_DIAMETER` cells across its
    diameter.  When a PF-1000 rod projection falls below that, the warning is
    raised and ``can_support_geometry_acceptance`` is False so an under-
    resolved rod can never lift geometry acceptance.  Super-Sprint 9 WS9-6.
    """

    under_resolved = bool(
        is_pf1000_rod_projection
        and cells_per_rod_diameter is not None
        and cells_per_rod_diameter < _REVIEWED_MIN_CELLS_PER_ROD_DIAMETER
    )
    if under_resolved:
        warning = (
            "cathode rod diameter is under-resolved by the grid: "
            f"{cells_per_rod_diameter:.3f} cells across a rod diameter is "
            f"below the declared minimum of "
            f"{_REVIEWED_MIN_CELLS_PER_ROD_DIAMETER:.1f}; refine the radial "
            "mesh before any reviewed PF-1000 geometry mask"
        )
    else:
        warning = None
    return {
        "status": (
            "warning_cathode_rod_under_resolved_not_validation"
            if under_resolved
            else "candidate_mesh_resolution_check_not_validation"
        ),
        "cathode_rod_under_resolved": under_resolved,
        "cells_per_rod_diameter": cells_per_rod_diameter,
        "reviewed_min_cells_per_rod_diameter": (
            _REVIEWED_MIN_CELLS_PER_ROD_DIAMETER
        ),
        "warning": warning,
        "can_support_geometry_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _blocked_geometry_field_entries(
    deck: FirstPrinciples3DDeck,
) -> tuple[dict[str, Any], ...]:
    """Return the deck's blocked geometry fields as stable telemetry entries.

    SS10-2 (closes audit A2).  Each entry carries the blocker id, blocked
    status, and source/scope reason.  A deck that does not declare blocked
    geometry fields yields an empty tuple (engineering-smoke decks).
    """

    entries: list[dict[str, Any]] = []
    for item in deck.blocked_geometry_fields or ():
        if not isinstance(item, Mapping):
            continue
        field_name = item.get("field_name")
        if field_name is None:
            continue
        entries.append(
            {
                "field_name": str(field_name),
                "blocker_id": str(item.get("blocker_id", "") or ""),
                "blocked": bool(item.get("blocked", True)),
                "source_scope_reason": str(
                    item.get(
                        "source_scope_reason",
                        "no_same_scope_kr_source_for_selected_scope",
                    )
                ),
            }
        )
    return tuple(entries)


def _has_blocked_geometry_field(
    deck: FirstPrinciples3DDeck,
    field_name: str,
) -> bool:
    """Return True when ``field_name`` is a blocked geometry field on the deck."""

    return any(
        entry["field_name"] == field_name and entry["blocked"] is True
        for entry in _blocked_geometry_field_entries(deck)
    )


def _conductor_mask_packet(
    *,
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
    mask: np.ndarray | None,
    source: str,
) -> dict[str, Any]:
    active = 0 if mask is None else int(np.count_nonzero(mask))
    total = int(np.prod(grid.shape))
    pf1000_rod_projection = source == "candidate_pf1000_rod_hollow_projection"
    projection_error = _conductor_mask_projection_error(
        deck=deck,
        grid=grid,
        mask=mask,
        is_pf1000_rod_projection=pf1000_rod_projection,
    )
    cells_per_rod = projection_error["cells_per_rod_diameter"]
    rods_resolved = (
        cells_per_rod is not None
        and cells_per_rod >= _REVIEWED_MIN_CELLS_PER_ROD_DIAMETER
    )
    # WS9-6: an under-resolved cathode rod cannot support a reviewed geometry
    # mask.  Emit an explicit mesh-resolution warning whenever a PF-1000 rod
    # projection places fewer than the declared minimum cells across a rod
    # diameter, and confirm it cannot lift geometry acceptance.
    mesh_resolution_warning = _conductor_mask_mesh_resolution_warning(
        is_pf1000_rod_projection=pf1000_rod_projection,
        cells_per_rod_diameter=cells_per_rod,
    )
    return {
        "status": "candidate_engineering_conductor_mask_not_validation",
        # ``source`` is the mask-PROJECTION algorithm (architecture) source;
        # the device geometry it projects is cited under
        # ``selected_machine_source_*`` from the selected deck (WS9-6).
        "source": HYBRID_PIC_3D_SOURCE,
        "architecture_source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "613-619, 640-641",
        "selected_machine_source_scope": deck.selected_machine_source_scope,
        "selected_machine_source_references": list(
            deck.selected_machine_source_references
        ),
        "declared_scope": deck.validation_scope,
        "mask_source": source,
        "conductor_mask_status": deck.conductor_mask_status,
        "conductor_mask_mode": deck.conductor_mask_mode,
        "grid_shape": list(grid.shape),
        "conductor_cells_active": active,
        "conductor_cell_fraction": 0.0 if total == 0 else active / total,
        "device_anode_radius_m": deck.device_anode_radius_m,
        "device_anode_inner_radius_m": deck.device_anode_inner_radius_m,
        "device_cathode_radius_m": deck.device_cathode_radius_m,
        "device_anode_length_m": deck.device_anode_length_m,
        "device_cathode_rod_count": deck.device_cathode_rod_count,
        "device_cathode_rod_diameter_m": deck.device_cathode_rod_diameter_m,
        "device_cathode_rod_length_m": deck.device_cathode_rod_length_m,
        "device_insulator_length_m": deck.device_insulator_length_m,
        "device_insulator_outer_radius_m": deck.device_insulator_outer_radius_m,
        "device_insulator_material": deck.device_insulator_material,
        "pf1000_geometry_features": {
            "cathode_rods_projected": bool(
                pf1000_rod_projection
                and deck.device_cathode_rod_count
                and deck.device_cathode_rod_diameter_m
            ),
            "cathode_rod_diameter_grid_cells": (
                float(deck.device_cathode_rod_diameter_m) / min(grid.dx, grid.dy)
                if pf1000_rod_projection and deck.device_cathode_rod_diameter_m
                else None
            ),
            "cathode_rods_resolution_reviewed": False,
            # SS10-2 / A3: the anode is "declared hollow by source" ONLY when a
            # KR-supported inner radius (or hollow-bore geometry field) is
            # actually present for the selected scope.  The PF-1000 full-energy
            # deck deliberately leaves ``anode_inner_radius_m=None`` — the
            # hollow bore is BLOCKED (PF1000-BLK-009/010) — so this is False,
            # never ``bool(pf1000_rod_projection)``.  A blocked-field entry for
            # the missing bore length/radius is carried below.
            "hollow_anode_declared_by_source": (
                deck.device_anode_inner_radius_m is not None
            ),
            "hollow_anode_inner_radius_supplied": (
                deck.device_anode_inner_radius_m is not None
            ),
            "hollow_anode_bore_blocked": _has_blocked_geometry_field(
                deck, "anode_hollow_bore_length_m"
            ),
            "insulator_material_surface_declared": bool(deck.device_insulator_material),
            "insulator_material_surface_resolved": False,
            "reviewed_same_scope_voxel_mask": (
                deck.conductor_mask_status == "reviewed_same_scope_geometry_mask"
            ),
        },
        # SS10-2 (closes audit A2): the deck blocked-field manifest threaded
        # into conductor-mask telemetry.  Each entry carries its blocker id,
        # blocked status, and source/scope reason so a reviewer sees every
        # blocked PF-1000 geometry field without re-reading the deck helper.
        "blocked_geometry_fields": list(_blocked_geometry_field_entries(deck)),
        "projection_error": projection_error,
        "resolution_review": {
            "status": "candidate_resolution_review_not_validation",
            "reviewed_min_cells_per_rod_diameter": (
                _REVIEWED_MIN_CELLS_PER_ROD_DIAMETER
            ),
            "cells_per_rod_diameter": projection_error["cells_per_rod_diameter"],
            "cathode_rods_resolved": bool(rods_resolved),
            "hollow_anode_resolved": deck.device_anode_inner_radius_m is not None,
            "insulator_material_surface_resolved": False,
            "material_surface_resolved": False,
            "reviewed_status_resolution_gate_eligible": bool(
                rods_resolved
                and deck.device_anode_inner_radius_m is not None
            ),
            "cathode_rod_under_resolved": bool(
                mesh_resolution_warning["cathode_rod_under_resolved"]
            ),
            "can_support_first_principles_acceptance": False,
        },
        "mesh_resolution_warning": mesh_resolution_warning,
        "coordinate_interpretation": (
            "centered_cartesian_full_azimuth_projection"
            if pf1000_rod_projection
            else "positive_quadrant_cartesian_projection_from_axis_r_equals_zero"
        ),
        "can_support_first_principles_acceptance": False,
        "limitations": [
            (
                "PF-1000 cathode rods are projected onto a Cartesian engineering grid."
                if pf1000_rod_projection
                else "Axisymmetric electrode dimensions are projected onto a Cartesian engineering grid."
            ),
            "Rod-diameter grid resolution is reported in cathode_rod_diameter_grid_cells; "
            "values below about two cells do not resolve discrete rods and rod-level "
            "fidelity is not resolution-reviewed.",
            "Hollow-anode bore is not resolved unless an accepted inner radius is supplied.",
            "Insulator material surfaces are declared but not resolved as material boundary regions.",
            "No reviewed same-scope electrode mask or boundary-validation packet is attached.",
        ],
    }


def _boundary_policy_telemetry(
    *,
    deck: FirstPrinciples3DDeck,
    grid: Maxwell3DGrid,
    conductor_cells: np.ndarray | None,
    conductor_mask: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "candidate_engineering_boundary_policy_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "613-619, 625-628",
        "capability": "pml_conductor_field_particle_boundaries",
        "implementation": [
            "src/dpf/fields/maxwell_3d.py",
            "src/dpf/fields/particle_boundaries.py",
            "src/dpf/fields/hybrid_loop.py",
            "src/dpf/first_principles/runner.py",
        ],
        "grid_shape": list(grid.shape),
        "pml_cells": int(deck.pml_cells),
        "pml_strength": float(deck.pml_strength),
        "particle_absorption_enabled": bool(deck.particle_absorption_enabled),
        "open_boundary": bool(deck.open_boundary),
        "conductor_mask_status": deck.conductor_mask_status,
        "conductor_mask_mode": deck.conductor_mask_mode,
        "conductor_mask": dict(conductor_mask),
        # SS10-2 (closes audit A2): also surface the blocked geometry fields at
        # the boundary-policy top level so a reviewer does not have to descend
        # into ``conductor_mask`` to enumerate them.
        "blocked_geometry_fields": list(
            conductor_mask.get("blocked_geometry_fields", ())
        ),
        "conductor_cells_active": (
            0 if conductor_cells is None else int(np.count_nonzero(conductor_cells))
        ),
        "field_boundary_runtime": {
            "maxwell_core_receives_boundary_policy": True,
            "pml_damping_candidate": deck.pml_cells > 0 and deck.pml_strength > 0.0,
            "conductor_e_zero_candidate": conductor_cells is not None,
        },
        "particle_boundary_runtime": {
            "absorbing_boundary_enabled": bool(deck.particle_absorption_enabled),
            "deletes_pml_conductor_or_outside_particles": bool(
                deck.particle_absorption_enabled
            ),
        },
        "acceptance_gate": (
            "pml_conductor_boundary_runtime_is_candidate_only_until_same_scope_"
            "geometry_masks_boundary_validation_and_particle_boundary_order_are_accepted"
        ),
        "negative_test_policy": {
            "raw_boundary_runtime_promotion_rejection_required": True,
            "raw_conductor_mask_without_same_scope_review_rejection_required": True,
            "particle_deletion_after_reflecting_pic_push_rejection_required": True,
        },
        "can_support_first_principles_acceptance": False,
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
    _commit, _dirty = git_provenance()
    _deck_sha = sha256_of_text(
        json.dumps(deck.manifest_config(), sort_keys=True, default=str)
    )
    _source_refs = [{
        "source_id": "hybrid_pic_3d_architecture_source",
        "path": HYBRID_PIC_3D_SOURCE,
        "scope": RUN_MODE,
        "status": "source_reference_not_validation",
    }]
    _source_truth_index_sha = sha256_of_file_soft(
        "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json"
    )
    _source_packet_hashes = source_packet_hashes_from_references(_source_refs)
    package_manifest = build_first_principles_manifest_from_hybrid_result(
        simulation,
        backend="package_native",
        grid=grid,
        source_index_references=_source_refs,
        metadata={
            "status": ENGINEERING_CANDIDATE_STATUS,
            "run_mode": RUN_MODE,
            "validation_packet_status": validation_packet["status"],
            "reduced_models_used": False,
            "deck": deck.manifest_config(),
        },
        command_argv=tuple(sys.argv),
        git_commit=_commit,
        dirty_worktree=_dirty,
        source_truth_index_sha256=_source_truth_index_sha,
        source_packet_hashes=_source_packet_hashes,
        input_deck_sha256=_deck_sha,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        artifact_generation_commit=_commit,
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
        "deck_diff_packet": telemetry["deck_diff"],
        "limiter_readiness_packet": telemetry["limiter_readiness"],
        "experimental_limiter_zero_probe_packet": telemetry[
            "experimental_limiter_zero_probe"
        ],
        "boundary_policy_packet": telemetry["boundary_policy"],
        "pic_particle_loading_packet": telemetry["pic_particle_loading"],
        "power_port_packet": telemetry["power_port"],
        "dimensionality_handoff_packet": telemetry["dimensionality_handoff"],
        "physics_closure_packet": telemetry["physics_closure"],
        "same_scope_source_packet": telemetry["same_scope_source"],
        # SS11-3 (audit S10-A3): other-scope architecture/schema context under
        # a non-``same_scope``-named manifest key.
        "architecture_or_schema_context_sources": telemetry[
            "architecture_or_schema_context_sources"
        ],
        "waveform_phase_packet": telemetry["waveform_phase"],
        "spatial_field_temperature_packet": telemetry["spatial_field_temperature"],
        "neutron_authority_packet": telemetry["neutron_authority"],
        "comparator_uq_packet": telemetry["comparator_uq"],
        "numerical_fidelity_packet": telemetry["numerical_fidelity"],
        "certificate_gate_packet": telemetry["certificate_gate"],
        "generalization_packet": telemetry["generalization"],
        "experimental_whole_shot_packet": telemetry["experimental_whole_shot"],
        "experimental_numerical_runtime_audit_packet": telemetry[
            "experimental_numerics"
        ],
        "hybrid_pic_3d_readiness_packet": telemetry["hybrid_pic_3d_readiness"],
        "hybrid_pic_3d": telemetry["candidate_evidence"],
    }
    payload["first_principles_manifest"] = package_manifest
    return payload


def _coerce_deck_values(values: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(values)
    for key in ("grid_shape", "grid_spacing_m"):
        if coerced.get(key) is not None:
            coerced[key] = tuple(coerced[key])
    if coerced.get("history_stride") is not None:
        coerced["history_stride"] = int(coerced["history_stride"])
    if coerced.get("max_step_results") is not None:
        coerced["max_step_results"] = int(coerced["max_step_results"])
    if coerced.get("target_time_s") is not None:
        coerced["target_time_s"] = float(coerced["target_time_s"])
    if isinstance(coerced.get("circuit_udpf_V"), list):
        coerced["circuit_udpf_V"] = tuple(float(v) for v in coerced["circuit_udpf_V"])
    if coerced.get("circuit_udpf_mode") is not None:
        coerced["circuit_udpf_mode"] = str(coerced["circuit_udpf_mode"])
    if coerced.get("pml_cells") is not None:
        coerced["pml_cells"] = int(coerced["pml_cells"])
    if coerced.get("pml_strength") is not None:
        coerced["pml_strength"] = float(coerced["pml_strength"])
    if coerced.get("particle_absorption_enabled") is not None:
        coerced["particle_absorption_enabled"] = bool(
            coerced["particle_absorption_enabled"]
        )
    if coerced.get("open_boundary") is not None:
        coerced["open_boundary"] = bool(coerced["open_boundary"])
    if coerced.get("conductor_mask_status") is not None:
        coerced["conductor_mask_status"] = str(coerced["conductor_mask_status"])
    if coerced.get("conductor_mask_mode") is not None:
        coerced["conductor_mask_mode"] = str(coerced["conductor_mask_mode"])
    if coerced.get("blocked_geometry_fields") is not None:
        coerced["blocked_geometry_fields"] = _blocked_geometry_fields_to_mappings(
            coerced["blocked_geometry_fields"]
        )
    if coerced.get("device_cathode_rod_count") is not None:
        coerced["device_cathode_rod_count"] = int(coerced["device_cathode_rod_count"])
    for key in (
        "device_anode_inner_radius_m",
        "device_cathode_rod_diameter_m",
        "device_cathode_rod_length_m",
        "device_insulator_outer_radius_m",
    ):
        if coerced.get(key) is not None:
            coerced[key] = float(coerced[key])
    if coerced.get("device_insulator_material") is not None:
        coerced["device_insulator_material"] = str(coerced["device_insulator_material"])
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
    for key in ("validation_scope", "selected_machine_source_scope"):
        if coerced.get(key) is not None:
            coerced[key] = str(coerced[key])
    if coerced.get("selected_machine_source_references") is not None:
        coerced["selected_machine_source_references"] = tuple(
            str(v) for v in coerced["selected_machine_source_references"]
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


def _last_step_has_applied_heat_flux(last_step: Mapping[str, Any]) -> bool:
    electron_energy = last_step.get("electron_energy")
    if not isinstance(electron_energy, Mapping):
        return False
    heat_flux = electron_energy.get("heat_flux")
    return isinstance(heat_flux, Mapping) and heat_flux.get("applied") is True


def _last_step_has_equilibration_audit(last_step: Mapping[str, Any]) -> bool:
    electron_energy = last_step.get("electron_energy")
    if not isinstance(electron_energy, Mapping):
        return False
    audit = electron_energy.get("equilibration_audit")
    return (
        isinstance(audit, Mapping)
        and audit.get("status") == "candidate_nrl_equal_temperature_equilibration_audit"
    )


def _last_step_has_source_backed_transport(last_step: Mapping[str, Any]) -> bool:
    transport = last_step.get("source_backed_transport")
    return (
        isinstance(transport, Mapping)
        and transport.get("status")
        == "candidate_source_backed_partial_ionized_conductivity"
    )


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
    boundaries = getattr(deck, "boundaries", None)
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
        "device_anode_inner_radius_m": _get(
            deck.device, "anode_inner_radius_m", None
        ),
        "device_cathode_rod_count": _get(deck.device, "cathode_rod_count", None),
        "device_cathode_rod_diameter_m": _get(
            deck.device, "cathode_rod_diameter_m", None
        ),
        "device_cathode_rod_length_m": _get(
            deck.device, "cathode_rod_length_m", None
        ),
        "device_insulator_outer_radius_m": _get(
            deck.device, "insulator_outer_radius_m", None
        ),
        "device_insulator_material": _get(deck.device, "insulator_material", None),
        "validation_scope": _validation_scope_from_package_deck(deck),
        "selected_machine_source_scope": _selected_machine_source_scope_from_package_deck(
            deck
        ),
        "selected_machine_source_references": (
            _selected_machine_source_references_from_package_deck(deck)
        ),
        "validation_targets": tuple(asdict(target) for target in deck.validation_targets),
        "n_steps": diagnostics.n_steps,
        "history_stride": _get(diagnostics, "history_stride", 1),
        "max_step_results": _get(diagnostics, "max_step_results", 256),
        "target_time_s": _get(diagnostics, "target_time_s", None),
        "grid_shape": tuple(grid.shape),
        "grid_spacing_m": tuple(grid.spacing_m),
        "dt_s": diagnostics.dt_s,
        "sigma0_S_m": closures.sigma0_S_m,
        "background_density_m3": startup.background_density_m3,
        "density_floor_m3": closures.density_floor_m3,
        "initial_ionization_fraction": startup.initial_ionization_fraction,
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
        "pml_cells": int(_get(boundaries, "pml_cells", 0)),
        "pml_strength": float(_get(boundaries, "pml_strength", 0.0)),
        "particle_absorption_enabled": bool(
            _get(boundaries, "particle_absorption_enabled", False)
        ),
        "open_boundary": bool(_get(boundaries, "open_boundary", True)),
        "conductor_mask_status": str(
            _get(boundaries, "conductor_mask_status", "not_supplied")
        ),
        "conductor_mask_mode": str(
            _get(boundaries, "conductor_mask_mode", "none")
        ),
        # SS10-2 (closes audit A2): the package deck boundary policy carries the
        # blocked geometry fields; thread them onto the 3-D deck so the
        # conductor-mask telemetry exposes every blocked field with its blocker.
        "blocked_geometry_fields": _blocked_geometry_fields_to_mappings(
            _get(boundaries, "blocked_geometry_fields", ())
        ),
        "circuit_capacitance_F": circuit.capacitance_F,
        "circuit_voltage_V": circuit.voltage_V,
        "circuit_inductance_H": circuit.inductance_H,
        "circuit_resistance_ohm": circuit.resistance_ohm,
        "circuit_udpf_mode": str(
            _get(closures, "circuit_udpf_mode", "lagged_volume_j_dot_e")
        ),
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
        "startup_accepted_channels": tuple(startup.accepted_channels),
        "startup_required_channels": tuple(startup.required_channels),
        "startup_missing_channels": tuple(startup.missing_channels),
        "startup_payload": dict(startup.startup_payload),
    }


_UNDECLARED_PACKAGE_DECK_VALIDATION_SCOPE = "not_declared_engineering_smoke"
_UNDECLARED_PACKAGE_DECK_SOURCE_SCOPE = (
    "not_declared_engineering_smoke_machine_source"
)


def _validation_scope_from_package_deck(deck: Any) -> str:
    """Resolve the package-native deck's declared validation scope.

    A deck id is a runtime artifact identifier, NOT a validation scope, and is
    never substituted here.  The deck's explicit ``validation_scope`` field is
    authoritative; a deck that does not declare one stays at the engineering-
    smoke placeholder so no downstream packet mistakes the deck id for the
    selected scope.  Super-Sprint 9 WS9-1 (fixes audit P0-1).
    """

    declared = str(
        getattr(deck, "validation_scope", _UNDECLARED_PACKAGE_DECK_VALIDATION_SCOPE)
    )
    if declared and declared != _UNDECLARED_PACKAGE_DECK_VALIDATION_SCOPE:
        return declared
    return _UNDECLARED_PACKAGE_DECK_VALIDATION_SCOPE


def _selected_machine_source_scope_from_package_deck(deck: Any) -> str:
    """Resolve the package-native deck's selected-machine source scope.

    This is a device/operating-point source scope, distinct from the
    architecture/equation-method source (the hybrid-PIC paper).  A deck id is
    never substituted.  Super-Sprint 9 WS9-2 (fixes audit P0-2).
    """

    declared = str(
        getattr(
            deck,
            "selected_machine_source_scope",
            _UNDECLARED_PACKAGE_DECK_SOURCE_SCOPE,
        )
    )
    if declared and declared != _UNDECLARED_PACKAGE_DECK_SOURCE_SCOPE:
        return declared
    return _UNDECLARED_PACKAGE_DECK_SOURCE_SCOPE


def _selected_machine_source_references_from_package_deck(
    deck: Any,
) -> tuple[str, ...]:
    """Collect KR geometry/circuit source-reference paths for the deck machine.

    Pulls the source-reference paths declared on the deck device (and circuit),
    so the candidate packet and conductor-mask telemetry can cite the selected
    deck's KR geometry rather than the LLNL-like architecture geometry.
    Super-Sprint 9 WS9-2 / WS9-6.
    """

    paths: list[str] = []
    for holder_name in ("device", "circuit"):
        holder = getattr(deck, holder_name, None)
        if holder is None:
            continue
        references = getattr(holder, "source_references", ())
        for reference in references:
            path = _get(reference, "path", None)
            if path:
                text = str(path)
                if text not in paths:
                    paths.append(text)
    return tuple(paths)
