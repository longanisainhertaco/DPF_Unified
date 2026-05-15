"""Candidate multi-step 3-D hybrid PIC-fluid simulation driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.circuit_boundary import (
    CircuitMagneticBoundaryDrive,
    CircuitState,
)
from dpf.fields.electron_energy import ElectronEnergyState
from dpf.fields.hybrid_loop import HybridPIC3DLoop, HybridPIC3DLoopResult
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid, Maxwell3DState


@dataclass(frozen=True)
class HybridPIC3DSimulationTelemetry:
    """Compact telemetry for a candidate multi-step 3-D hybrid run."""

    status: str
    source: str
    n_steps_requested: int
    n_steps_completed: int
    final_time_s: float
    n_particles_initial: int
    n_particles_final: int
    initial_field_energy_J: float
    final_field_energy_J: float
    last_step: dict[str, Any] | None
    circuit: dict[str, Any] | None
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridPIC3DSimulationResult:
    """Result from a candidate multi-step 3-D hybrid run."""

    state: Maxwell3DState
    electron_energy: ElectronEnergyState | None
    circuit: CircuitState | None
    step_results: list[HybridPIC3DLoopResult]
    telemetry: HybridPIC3DSimulationTelemetry


class HybridPIC3DSimulator:
    """Advance the candidate 3-D hybrid PIC-fluid loop for multiple steps."""

    def __init__(
        self,
        *,
        grid: Maxwell3DGrid,
        loop: HybridPIC3DLoop,
        state: Maxwell3DState,
        pic: HybridPIC,
        circuit_boundary: CircuitMagneticBoundaryDrive | None = None,
    ) -> None:
        if tuple(pic.grid_shape) != grid.shape:
            raise ValueError("PIC grid shape does not match Maxwell grid")
        if circuit_boundary is not None and circuit_boundary.grid != grid:
            raise ValueError("circuit_boundary grid does not match Maxwell grid")
        self.grid = grid
        self.loop = loop
        self.state = state
        self.pic = pic
        self.circuit_boundary = circuit_boundary

    def run(
        self,
        *,
        n_steps: int,
        dt_s: float,
        sigma0_S_m: np.ndarray | float,
        background_density_m3: float,
        ohmic_cfl_safety: float,
        density_floor_m3: float = 1.0,
        include_hall: bool = True,
        use_predictor_corrector: bool = False,
        marder_factor_m2: float = 0.0,
        marder_nondominance_threshold: float | None = None,
        electron_energy_state: ElectronEnergyState | None = None,
        mass_density_kg_m3: np.ndarray | None = None,
        plasma_velocity_m_s: np.ndarray | None = None,
        charge_state_Z: float = 1.0,
        electron_temperature_floor_K: float = 1.0,
        use_source_ordered_velocity_update: bool = False,
        circuit_state: CircuitState | None = None,
        apply_circuit_boundary: bool = False,
        circuit_udpf_V: float | list[float] | tuple[float, ...] | np.ndarray = 0.0,
        circuit_z_index: int = 0,
        circuit_blend: float = 1.0,
    ) -> HybridPIC3DSimulationResult:
        if int(n_steps) != n_steps or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if apply_circuit_boundary and self.circuit_boundary is None:
            raise ValueError(
                "circuit_boundary is required when apply_circuit_boundary is True"
            )

        initial_particles = _particle_count(self.pic)
        initial_energy = self.loop.field_stepper.maxwell.diagnostics(
            self.state
        ).total_energy_J
        step_results: list[HybridPIC3DLoopResult] = []
        circuit_records: list[dict[str, Any]] = []
        current_electron_state = electron_energy_state
        current_circuit_state = circuit_state
        if apply_circuit_boundary and current_circuit_state is None:
            current_circuit_state = CircuitState()
        udpf_values = _udpf_sequence(circuit_udpf_V, int(n_steps))
        for step_index in range(int(n_steps)):
            if (
                apply_circuit_boundary
                and self.circuit_boundary is not None
                and current_circuit_state is not None
            ):
                self.state, boundary_telemetry = (
                    self.circuit_boundary.apply_injection_port_boundary(
                        self.state,
                        current_A=current_circuit_state.current_A,
                        z_index=circuit_z_index,
                        blend=circuit_blend,
                    )
                )
                next_circuit_state, circuit_step_telemetry = (
                    self.circuit_boundary.step_circuit(
                        current_circuit_state,
                        dt_s=dt_s,
                        udpf_V=udpf_values[step_index],
                    )
                )
                circuit_records.append({
                    "step_index": step_index,
                    "boundary": boundary_telemetry.to_dict(),
                    "circuit_step": circuit_step_telemetry.to_dict(),
                })
                current_circuit_state = next_circuit_state
            step = self.loop.step(
                self.state,
                self.pic,
                dt_s=dt_s,
                sigma0_S_m=sigma0_S_m,
                background_density_m3=background_density_m3,
                ohmic_cfl_safety=ohmic_cfl_safety,
                density_floor_m3=density_floor_m3,
                include_hall=include_hall,
                use_predictor_corrector=use_predictor_corrector,
                marder_factor_m2=marder_factor_m2,
                marder_nondominance_threshold=marder_nondominance_threshold,
                electron_energy_state=current_electron_state,
                mass_density_kg_m3=mass_density_kg_m3,
                plasma_velocity_m_s=plasma_velocity_m_s,
                charge_state_Z=charge_state_Z,
                electron_temperature_floor_K=electron_temperature_floor_K,
                use_source_ordered_velocity_update=use_source_ordered_velocity_update,
            )
            self.state = step.state
            if step.electron_energy is not None:
                current_electron_state = step.electron_energy
            step_results.append(step)

        final_energy = self.loop.field_stepper.maxwell.diagnostics(
            self.state
        ).total_energy_J
        telemetry = HybridPIC3DSimulationTelemetry(
            status="candidate_engineering_3d_hybrid_pic_simulation",
            source=HYBRID_PIC_3D_SOURCE,
            n_steps_requested=int(n_steps),
            n_steps_completed=len(step_results),
            final_time_s=float(len(step_results) * dt_s),
            n_particles_initial=initial_particles,
            n_particles_final=_particle_count(self.pic),
            initial_field_energy_J=float(initial_energy),
            final_field_energy_J=float(final_energy),
            last_step=(
                None if not step_results else step_results[-1].telemetry.to_dict()
            ),
            circuit=(
                None
                if not circuit_records
                else {
                    "status": "candidate_engineering_circuit_boundary_coupled",
                    "source": HYBRID_PIC_3D_SOURCE,
                    "source_lines": "740-792",
                    "n_steps": len(circuit_records),
                    "final_current_A": (
                        None
                        if current_circuit_state is None
                        else current_circuit_state.current_A
                    ),
                    "final_charge_C": (
                        None
                        if current_circuit_state is None
                        else current_circuit_state.charge_C
                    ),
                    "last": circuit_records[-1],
                    "can_support_first_principles_acceptance": False,
                }
            ),
        )
        return HybridPIC3DSimulationResult(
            state=self.state,
            electron_energy=current_electron_state,
            circuit=current_circuit_state if apply_circuit_boundary else None,
            step_results=step_results,
            telemetry=telemetry,
        )


def hybrid_simulator_candidate_evidence(
    telemetry: HybridPIC3DSimulationTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for a multi-step 3-D hybrid run."""
    return {
        "passed": telemetry.status == "candidate_engineering_3d_hybrid_pic_simulation",
        "status": "candidate",
        "capability": "true_3d_dimensionality",
        "source": telemetry.source,
        "source_lines": "246-311, 1215-1225, 1274-1278",
        "implementation": "src/dpf/fields/hybrid_simulator.py",
        "evidence_type": "engineering_multi_step_3d_hybrid_pic_run",
        "n_steps_completed": telemetry.n_steps_completed,
        "final_time_s": telemetry.final_time_s,
        "circuit": telemetry.circuit,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Runs the candidate Cartesian 3-D loop but does not validate 3-D DPF instabilities.",
            "Optional circuit-boundary coupling uses an input UDPF placeholder, not the accepted flux-derivative closure.",
            "No same-scope geometry, detector, neutron, UQ, or backend-scaling packet is attached.",
            "The source's full provisional ion-push/rebuild ordering remains candidate-only.",
        ],
    }


def _particle_count(pic: HybridPIC) -> int:
    return int(sum(species.n_particles() for species in pic.species))


def _udpf_sequence(
    values: float | list[float] | tuple[float, ...] | np.ndarray,
    n_steps: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(n_steps, float(array), dtype=float)
    if array.shape != (n_steps,):
        raise ValueError("circuit_udpf_V must be scalar or length n_steps")
    return array
