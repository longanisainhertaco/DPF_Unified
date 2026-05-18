"""Candidate multi-step 3-D hybrid PIC-fluid simulation driver."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.circuit_boundary import (
    CircuitMagneticBoundaryDrive,
    CircuitState,
)
from dpf.fields.electron_energy import ElectronEnergyState
from dpf.fields.hybrid_loop import HybridPIC3DLoop, HybridPIC3DLoopResult
from dpf.fields.ionization_transport import DeuteriumIonizationState
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
    stop_reason: str = "completed_step_budget"
    termination_reason: str = "completed_step_budget"
    target_time_s: float | None = None
    duration_request_satisfied: bool | None = None
    last_completed_step_index: int | None = None
    retained_step_result_count: int = 0
    history_stride: int = 1
    max_step_results: int | None = None
    history_summary: list[dict[str, Any]] = field(default_factory=list)
    cumulative_j_dot_e_work_J: float | None = None
    cumulative_j_dot_e_step_count: int = 0
    cumulative_j_dot_e_status: str = "candidate_cumulative_volume_j_dot_e_not_validation"
    cumulative_active_port_work_J: float | None = None
    cumulative_active_port_step_count: int = 0
    cumulative_active_port_status: str = (
        "candidate_cumulative_terminal_i_udpf_not_validation"
    )
    udpf_source_counts: dict[str, int] = field(default_factory=dict)
    limiter_activation_summary: dict[str, Any] | None = None
    state_fingerprint: dict[str, Any] | None = None
    continuation_state: dict[str, Any] | None = None
    finite_state: dict[str, Any] | None = None
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridPIC3DSimulationResult:
    """Result from a candidate multi-step 3-D hybrid run."""

    state: Maxwell3DState
    electron_energy: ElectronEnergyState | None
    ionization_charge_state: DeuteriumIonizationState | None
    circuit: CircuitState | None
    pic: HybridPIC
    previous_total_current_A_m2: np.ndarray | None
    kinetic_yield_state: dict[str, Any] | None
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
        ionization_state: DeuteriumIonizationState | None = None,
        use_source_backed_conductivity: bool = False,
        mass_density_kg_m3: np.ndarray | None = None,
        plasma_velocity_m_s: np.ndarray | None = None,
        charge_state_Z: float = 1.0,
        electron_temperature_floor_K: float = 1.0,
        heat_flux_subcycles_max: int = 1000,
        pressure_density_threshold_m3: float = 0.0,
        use_source_ordered_velocity_update: bool = False,
        circuit_state: CircuitState | None = None,
        apply_circuit_boundary: bool = False,
        circuit_udpf_V: float | list[float] | tuple[float, ...] | np.ndarray = 0.0,
        circuit_udpf_mode: str = "input_sequence",
        circuit_feedback_min_current_A: float = 1.0,
        circuit_z_index: int = 0,
        circuit_blend: float = 1.0,
        history_stride: int = 1,
        max_step_results: int | None = None,
        target_time_s: float | None = None,
        abort_on_nonfinite: bool = True,
        initial_lagged_field_work: dict[str, Any] | None = None,
        step_index_offset: int = 0,
    ) -> HybridPIC3DSimulationResult:
        if int(n_steps) != n_steps or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if apply_circuit_boundary and self.circuit_boundary is None:
            raise ValueError(
                "circuit_boundary is required when apply_circuit_boundary is True"
            )
        if circuit_udpf_mode not in {"input_sequence", "lagged_volume_j_dot_e"}:
            raise ValueError(
                "circuit_udpf_mode must be 'input_sequence' or 'lagged_volume_j_dot_e'"
            )
        if circuit_feedback_min_current_A < 0.0:
            raise ValueError("circuit_feedback_min_current_A must be non-negative")
        if int(history_stride) != history_stride or history_stride <= 0:
            raise ValueError("history_stride must be a positive integer")
        if max_step_results is not None and (
            int(max_step_results) != max_step_results or max_step_results < 0
        ):
            raise ValueError("max_step_results must be a non-negative integer or None")
        if target_time_s is not None and target_time_s <= 0.0:
            raise ValueError("target_time_s must be positive when supplied")
        if int(step_index_offset) != step_index_offset or step_index_offset < 0:
            raise ValueError("step_index_offset must be a non-negative integer")

        initial_particles = _particle_count(self.pic)
        initial_energy = self.loop.field_stepper.maxwell.diagnostics(
            self.state
        ).total_energy_J
        step_results: list[HybridPIC3DLoopResult] = []
        history_summary: list[dict[str, Any]] = []
        cumulative_j_dot_e_work_J = 0.0
        cumulative_j_dot_e_step_count = 0
        cumulative_active_port_work_J = 0.0
        cumulative_active_port_step_count = 0
        udpf_source_counts: dict[str, int] = {}
        limiter_activation_summary = _empty_limiter_activation_summary()
        circuit_records: list[dict[str, Any]] = []
        circuit_history_cap = _circuit_history_record_cap(max_step_results)
        first_circuit_record: dict[str, Any] | None = None
        last_circuit_record: dict[str, Any] | None = None
        current_electron_state = electron_energy_state
        current_ionization_state = ionization_state
        current_circuit_state = circuit_state
        if apply_circuit_boundary and current_circuit_state is None:
            current_circuit_state = CircuitState()
        udpf_values = _udpf_sequence(circuit_udpf_V, int(n_steps))
        lagged_field_work: dict[str, Any] | None = (
            None
            if initial_lagged_field_work is None
            else dict(initial_lagged_field_work)
        )
        n_steps_completed = 0
        last_step_telemetry: dict[str, Any] | None = None
        stop_reason = "completed_step_budget"
        finite_state = _finite_state_packet(
            self.state,
            self.pic,
            current_electron_state,
            current_ionization_state,
            current_circuit_state if apply_circuit_boundary else None,
        )
        for step_index in range(int(n_steps)):
            absolute_step_index = int(step_index_offset) + step_index
            if (
                apply_circuit_boundary
                and self.circuit_boundary is not None
                and current_circuit_state is not None
            ):
                lagged_j_dot_e_power_W = _optional_float(
                    None
                    if lagged_field_work is None
                    else lagged_field_work.get("j_dot_e_power_W")
                )
                udpf_value, udpf_source = _circuit_udpf_for_step(
                    mode=circuit_udpf_mode,
                    input_udpf_V=float(udpf_values[step_index]),
                    lagged_field_work=lagged_field_work,
                    current_A=current_circuit_state.current_A,
                    min_current_A=circuit_feedback_min_current_A,
                )
                active_port_power_W = float(
                    current_circuit_state.current_A * udpf_value
                )
                cumulative_active_port_work_J += active_port_power_W * float(dt_s)
                cumulative_active_port_step_count += 1
                udpf_source_counts[udpf_source] = (
                    int(udpf_source_counts.get(udpf_source, 0)) + 1
                )
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
                        udpf_V=udpf_value,
                    )
                )
                voltage_balance = _circuit_voltage_balance(
                    self.circuit_boundary,
                    current_circuit_state,
                    udpf_V=udpf_value,
                    dI_dt_A_s=circuit_step_telemetry.dI_dt_A_s,
                    lagged_j_dot_e_power_W=lagged_j_dot_e_power_W,
                )
                circuit_record = {
                    "step_index": absolute_step_index,
                    "udpf_source": udpf_source,
                    "requested_udpf_mode": circuit_udpf_mode,
                    "feedback_min_current_A": float(circuit_feedback_min_current_A),
                    "active_port_power_W": active_port_power_W,
                    "active_port_work_J": active_port_power_W * float(dt_s),
                    "active_port_power_sign": (
                        "positive_I_udpf_is_power_drawn_from_generator_by_DPF"
                    ),
                    "active_port_time_centering": (
                        "begin_step_current_times_begin_step_udpf_candidate"
                    ),
                    "voltage_balance": voltage_balance,
                    "boundary": boundary_telemetry.to_dict(),
                    "circuit_step": circuit_step_telemetry.to_dict(),
                }
                if first_circuit_record is None:
                    first_circuit_record = circuit_record
                last_circuit_record = circuit_record
                if _retain_step(step_index, history_stride):
                    _append_capped(circuit_records, circuit_record, circuit_history_cap)
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
                ionization_state=current_ionization_state,
                use_source_backed_conductivity=use_source_backed_conductivity,
                mass_density_kg_m3=mass_density_kg_m3,
                plasma_velocity_m_s=plasma_velocity_m_s,
                charge_state_Z=charge_state_Z,
                electron_temperature_floor_K=electron_temperature_floor_K,
                heat_flux_subcycles_max=heat_flux_subcycles_max,
                pressure_density_threshold_m3=pressure_density_threshold_m3,
                use_source_ordered_velocity_update=use_source_ordered_velocity_update,
            )
            self.state = step.state
            if step.electron_energy is not None:
                current_electron_state = step.electron_energy
            if step.ionization_charge_state is not None:
                current_ionization_state = step.ionization_charge_state
            lagged_field_work = step.field_step.telemetry.field_work
            j_dot_e_power_W = _optional_float(
                None if lagged_field_work is None else lagged_field_work.get(
                    "j_dot_e_power_W"
                )
            )
            if j_dot_e_power_W is not None:
                cumulative_j_dot_e_work_J += j_dot_e_power_W * float(dt_s)
                cumulative_j_dot_e_step_count += 1
            n_steps_completed = step_index + 1
            last_step_telemetry = step.telemetry.to_dict()
            _record_limiter_activation(
                limiter_activation_summary,
                last_step_telemetry,
                electron_temperature_floor_K=electron_temperature_floor_K,
            )
            step_diagnostics = self.loop.field_stepper.maxwell.diagnostics(self.state)
            finite_state = _finite_state_packet(
                self.state,
                self.pic,
                current_electron_state,
                current_ionization_state,
                current_circuit_state if apply_circuit_boundary else None,
            )
            if _retain_step(step_index, history_stride):
                _append_capped(step_results, step, max_step_results)
                _append_capped(
                    history_summary,
                        _step_history_summary(
                            step_index=absolute_step_index,
                            dt_s=dt_s,
                            telemetry=last_step_telemetry,
                            diagnostics=step_diagnostics.to_dict(),
                            circuit_record=last_circuit_record,
                        ),
                        max_step_results,
                    )
            blocked_source_reason = _blocked_source_term_reason(last_step_telemetry)
            if blocked_source_reason is not None:
                stop_reason = blocked_source_reason
                break
            if abort_on_nonfinite and not finite_state["all_finite"]:
                stop_reason = "aborted_nonfinite_state"
                break
            if target_time_s is not None and float(n_steps_completed * dt_s) >= float(
                target_time_s
            ):
                stop_reason = "target_time_reached"
                break

        final_energy = self.loop.field_stepper.maxwell.diagnostics(
            self.state
        ).total_energy_J
        telemetry = HybridPIC3DSimulationTelemetry(
            status="candidate_engineering_3d_hybrid_pic_simulation",
            source=HYBRID_PIC_3D_SOURCE,
            n_steps_requested=int(n_steps),
            n_steps_completed=n_steps_completed,
            final_time_s=float(n_steps_completed * dt_s),
            n_particles_initial=initial_particles,
            n_particles_final=_particle_count(self.pic),
            initial_field_energy_J=float(initial_energy),
            final_field_energy_J=float(final_energy),
            last_step=last_step_telemetry,
            circuit=(
                None
                if last_circuit_record is None
                else {
                    "status": "candidate_engineering_circuit_boundary_coupled",
                    "source": HYBRID_PIC_3D_SOURCE,
                    "source_lines": "740-792",
                    "n_steps": n_steps_completed,
                    "retained_record_count": len(circuit_records),
                    "current_history_cap": circuit_history_cap,
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
                    "current_history": _circuit_current_history(
                        circuit_records,
                        dt_s=dt_s,
                        initial_record=first_circuit_record,
                        final_record=last_circuit_record,
                    ),
                    "last": last_circuit_record,
                    "can_support_first_principles_acceptance": False,
                }
            ),
            stop_reason=stop_reason,
            termination_reason=stop_reason,
            target_time_s=None if target_time_s is None else float(target_time_s),
            duration_request_satisfied=(
                None
                if target_time_s is None
                else float(n_steps_completed * dt_s) >= float(target_time_s)
                and not stop_reason.startswith("aborted_")
            ),
            last_completed_step_index=(
                None
                if n_steps_completed <= 0
                else int(step_index_offset) + n_steps_completed - 1
            ),
            retained_step_result_count=len(step_results),
            history_stride=int(history_stride),
            max_step_results=(
                None if max_step_results is None else int(max_step_results)
            ),
            history_summary=history_summary,
            cumulative_j_dot_e_work_J=(
                cumulative_j_dot_e_work_J
                if cumulative_j_dot_e_step_count > 0
                else None
            ),
            cumulative_j_dot_e_step_count=cumulative_j_dot_e_step_count,
            cumulative_active_port_work_J=(
                cumulative_active_port_work_J
                if cumulative_active_port_step_count > 0
                else None
            ),
            cumulative_active_port_step_count=cumulative_active_port_step_count,
            udpf_source_counts=dict(sorted(udpf_source_counts.items())),
            limiter_activation_summary=limiter_activation_summary,
            state_fingerprint=_state_fingerprint(
                self.state,
                self.pic,
                current_electron_state,
                current_ionization_state,
                current_circuit_state if apply_circuit_boundary else None,
            ),
            continuation_state=_continuation_state_packet(
                step_index_offset=int(step_index_offset),
                n_steps_completed=n_steps_completed,
                dt_s=dt_s,
                lagged_field_work=lagged_field_work,
            ),
            finite_state=finite_state,
        )
        return HybridPIC3DSimulationResult(
            state=self.state,
            electron_energy=current_electron_state,
            ionization_charge_state=current_ionization_state,
            circuit=current_circuit_state if apply_circuit_boundary else None,
            pic=self.pic,
            previous_total_current_A_m2=(
                None
                if self.loop.field_stepper.previous_total_current_A_m2 is None
                else np.array(
                    self.loop.field_stepper.previous_total_current_A_m2,
                    copy=True,
                )
            ),
            kinetic_yield_state=_kinetic_yield_state(
                self.loop.kinetic_yield_history
            ),
            step_results=step_results,
            telemetry=telemetry,
        )


def hybrid_simulator_candidate_evidence(
    telemetry: HybridPIC3DSimulationTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for a multi-step 3-D hybrid run."""
    return {
        "passed": (
            telemetry.status == "candidate_engineering_3d_hybrid_pic_simulation"
            and not telemetry.stop_reason.startswith("aborted_")
        ),
        "status": "candidate",
        "capability": "true_3d_dimensionality",
        "source": telemetry.source,
        "source_lines": "246-311, 1215-1225, 1274-1278",
        "implementation": "src/dpf/fields/hybrid_simulator.py",
        "evidence_type": "engineering_multi_step_3d_hybrid_pic_run",
        "n_steps_completed": telemetry.n_steps_completed,
        "final_time_s": telemetry.final_time_s,
        "stop_reason": telemetry.stop_reason,
        "termination_reason": telemetry.termination_reason,
        "target_time_s": telemetry.target_time_s,
        "duration_request_satisfied": telemetry.duration_request_satisfied,
        "retained_step_result_count": telemetry.retained_step_result_count,
        "history_stride": telemetry.history_stride,
        "max_step_results": telemetry.max_step_results,
        "limiter_activation_summary": telemetry.limiter_activation_summary,
        "state_fingerprint": telemetry.state_fingerprint,
        "continuation_state": telemetry.continuation_state,
        "finite_state": telemetry.finite_state,
        "circuit": telemetry.circuit,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Runs the candidate Cartesian 3-D loop but does not validate 3-D DPF instabilities.",
            "Long-run history may be stride-thinned and capped; completed-step counters remain authoritative for elapsed time.",
            "Optional circuit-boundary coupling can use input UDPF or lagged candidate volume J.E feedback, not an accepted centered field-power closure.",
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


def _circuit_udpf_for_step(
    *,
    mode: str,
    input_udpf_V: float,
    lagged_field_work: dict[str, Any] | None,
    current_A: float,
    min_current_A: float = 0.0,
) -> tuple[float, str]:
    if mode == "input_sequence":
        return float(input_udpf_V), "input_sequence"
    if lagged_field_work is None:
        return float(input_udpf_V), "input_sequence_fallback_first_step"
    if abs(float(current_A)) <= float(min_current_A):
        return float(input_udpf_V), "input_sequence_fallback_low_current"
    power_W = float(lagged_field_work.get("j_dot_e_power_W", 0.0))
    if power_W < 0.0:
        return (
            float(input_udpf_V),
            "input_sequence_fallback_negative_j_dot_e_active_port_blocked",
        )
    return float(power_W / float(current_A)), "candidate_lagged_volume_j_dot_e"


def _blocked_source_term_reason(telemetry: dict[str, Any] | None) -> str | None:
    if telemetry is None:
        return None
    electron_energy = telemetry.get("electron_energy")
    if not isinstance(electron_energy, dict):
        return None
    electron_status = str(electron_energy.get("status", ""))
    if electron_status.startswith("blocked_"):
        return "aborted_blocked_electron_energy_closure"
    heat_flux = electron_energy.get("heat_flux")
    if not isinstance(heat_flux, dict):
        return None
    status = str(heat_flux.get("status", ""))
    if status.startswith("blocked_"):
        return "aborted_blocked_electron_heat_flux"
    return None


def _circuit_current_history(
    records: list[dict[str, Any]],
    *,
    dt_s: float,
    initial_record: dict[str, Any] | None = None,
    final_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return compact current-vs-time samples from circuit-step telemetry."""

    if not records and initial_record is None:
        return []

    history: list[dict[str, Any]] = []
    first_record = initial_record or records[0]
    first_step = first_record.get("circuit_step", {})
    first_index = int(first_record.get("step_index", 0))
    history.append({
        "sample": "initial",
        "step_index": first_index,
        "time_s": float(first_index * dt_s),
        "time_us": float(first_index * dt_s * 1.0e6),
        "current_A": float(first_step.get("current_A", 0.0)),
        "source": "candidate_engineering_rlc_circuit_step",
        "voltage_balance": first_record.get("voltage_balance"),
    })

    for record in records:
        step = record.get("circuit_step", {})
        step_index = int(record.get("step_index", 0))
        time_s = float((step_index + 1) * dt_s)
        history.append({
            "sample": "post_step",
            "step_index": step_index,
            "time_s": time_s,
            "time_us": float(time_s * 1.0e6),
            "current_A": float(step.get("next_current_A", 0.0)),
            "source": str(step.get("status", "candidate_engineering_rlc_circuit_step")),
            "udpf_source": str(record.get("udpf_source", "not_recorded")),
            "voltage_balance": record.get("voltage_balance"),
        })

    if final_record is not None and (
        not records
        or int(final_record.get("step_index", -1))
        != int(records[-1].get("step_index", -2))
    ):
        step = final_record.get("circuit_step", {})
        step_index = int(final_record.get("step_index", 0))
        time_s = float((step_index + 1) * dt_s)
        history.append({
            "sample": "post_step",
            "step_index": step_index,
            "time_s": time_s,
            "time_us": float(time_s * 1.0e6),
            "current_A": float(step.get("next_current_A", 0.0)),
            "source": str(step.get("status", "candidate_engineering_rlc_circuit_step")),
            "udpf_source": str(final_record.get("udpf_source", "not_recorded")),
            "voltage_balance": final_record.get("voltage_balance"),
        })

    return history


def _circuit_voltage_balance(
    boundary: CircuitMagneticBoundaryDrive,
    state: CircuitState,
    *,
    udpf_V: float,
    dI_dt_A_s: float,
    lagged_j_dot_e_power_W: float | None,
) -> dict[str, Any]:
    params = boundary.parameters
    resistive_drop_V = float(params.resistance_ohm * state.current_A)
    charge_drop_V = float(state.charge_C / params.capacitance_F)
    net_drive_voltage_V = float(
        params.voltage_V - resistive_drop_V - charge_drop_V - float(udpf_V)
    )
    return {
        "status": "candidate_circuit_voltage_balance_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "740-792",
        "bank_voltage_V": float(params.voltage_V),
        "resistive_drop_V": resistive_drop_V,
        "charge_drop_V": charge_drop_V,
        "udpf_V": float(udpf_V),
        "net_drive_voltage_V": net_drive_voltage_V,
        "current_A": float(state.current_A),
        "charge_C": float(state.charge_C),
        "inductance_H": float(params.inductance_H),
        "dI_dt_A_s": float(dI_dt_A_s),
        "L_dI_dt_V": float(params.inductance_H * dI_dt_A_s),
        "lagged_j_dot_e_power_W": lagged_j_dot_e_power_W,
        "active_port_power_W": float(state.current_A * float(udpf_V)),
        "active_port_power_sign": (
            "positive_I_udpf_is_power_drawn_from_generator_by_DPF"
        ),
        "can_support_first_principles_acceptance": False,
    }


def _retain_step(step_index: int, history_stride: int) -> bool:
    return int(step_index) % int(history_stride) == 0


def _append_capped(items: list[Any], item: Any, max_items: int | None) -> None:
    if max_items == 0:
        return
    if max_items is not None and len(items) >= int(max_items):
        items.pop(0)
    items.append(item)


def _circuit_history_record_cap(max_step_results: int | None) -> int | None:
    """Keep circuit-current samples denser than full field/PIC step payloads."""

    if max_step_results is None:
        return None
    return max(64, int(max_step_results))


def _empty_limiter_activation_summary() -> dict[str, Any]:
    return {
        "status": "experimental_full_horizon_limiter_inventory_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "steps_observed": 0,
        "activation_counts": {
            "conductivity_ohmic_cfl_limited_steps": 0,
            "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps": 0,
            "conductivity_density_blend_applied_steps": 0,
            "marder_correction_steps": 0,
            "marder_dominant_correction_steps": 0,
            "electron_temperature_floor_contact_steps": 0,
            "blocked_heat_flux_steps": 0,
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
            "review_decision": "runtime_inventory_only",
        },
    }


def _record_limiter_activation(
    summary: dict[str, Any],
    telemetry: dict[str, Any],
    *,
    electron_temperature_floor_K: float,
) -> None:
    summary["steps_observed"] = int(summary.get("steps_observed", 0)) + 1
    counts = summary["activation_counts"]
    maxima = summary["max_observed"]

    field_step = telemetry.get("field_step")
    if not isinstance(field_step, dict):
        field_step = {}
    conductivity = field_step.get("conductivity")
    if not isinstance(conductivity, dict):
        conductivity = {}
    cfl_limited_fraction = _optional_float(
        conductivity.get("cfl_limited_fraction")
    )
    if cfl_limited_fraction is not None:
        maxima["conductivity_cfl_limited_fraction"] = max(
            float(maxima["conductivity_cfl_limited_fraction"]),
            cfl_limited_fraction,
        )
        if cfl_limited_fraction > 0.0:
            counts[
                "conductivity_ohmic_cfl_raw_exceeds_explicit_limit_steps"
            ] += 1
        if (
            cfl_limited_fraction > 0.0
            and conductivity.get("ohmic_cfl_limit_applied") is True
        ):
            counts["conductivity_ohmic_cfl_limited_steps"] += 1
    if conductivity.get("density_blend_applied") is True:
        counts["conductivity_density_blend_applied_steps"] += 1

    marder = field_step.get("marder")
    if isinstance(marder, dict):
        counts["marder_correction_steps"] += 1
        relative = _optional_float(marder.get("relative_correction_linf"))
        residual_after = _optional_float(marder.get("residual_after_linf"))
        if relative is not None:
            maxima["marder_relative_correction_linf"] = max(
                float(maxima["marder_relative_correction_linf"]),
                relative,
            )
        if residual_after is not None:
            maxima["marder_residual_after_linf"] = max(
                float(maxima["marder_residual_after_linf"]),
                residual_after,
            )
        threshold = _optional_float(marder.get("nondominance_threshold"))
        maxima["marder_nondominance_threshold"] = _min_optional(
            maxima.get("marder_nondominance_threshold"),
            threshold,
        )
        if marder.get("nondominance_status") == "candidate_dominant_correction":
            counts["marder_dominant_correction_steps"] += 1

    electron_energy = telemetry.get("electron_energy")
    if isinstance(electron_energy, dict):
        min_te = _optional_float(electron_energy.get("min_electron_temperature_K"))
        max_te = _optional_float(electron_energy.get("max_electron_temperature_K"))
        maxima["electron_temperature_min_K"] = _min_optional(
            maxima.get("electron_temperature_min_K"),
            min_te,
        )
        maxima["electron_temperature_max_K"] = _max_optional(
            maxima.get("electron_temperature_max_K"),
            max_te,
        )
        if min_te is not None and min_te <= float(electron_temperature_floor_K):
            counts["electron_temperature_floor_contact_steps"] += 1
        heat_flux = electron_energy.get("heat_flux")
        if isinstance(heat_flux, dict) and str(heat_flux.get("status", "")).startswith(
            "blocked"
        ):
            counts["blocked_heat_flux_steps"] += 1


def _continuation_state_packet(
    *,
    step_index_offset: int,
    n_steps_completed: int,
    dt_s: float,
    lagged_field_work: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "experimental_live_continuation_state_not_restart_acceptance",
        "step_index_offset": int(step_index_offset),
        "segment_steps_completed": int(n_steps_completed),
        "total_steps_completed": int(step_index_offset) + int(n_steps_completed),
        "total_time_s": float((int(step_index_offset) + int(n_steps_completed)) * dt_s),
        "has_lagged_field_work": lagged_field_work is not None,
        "lagged_field_work": lagged_field_work,
        "can_support_checkpoint_restart_acceptance": False,
    }


def _kinetic_yield_state(history: Any) -> dict[str, Any] | None:
    if history is None:
        return None
    return {
        "cumulative_neutrons": float(
            getattr(history, "cumulative_neutrons", 0.0)
        ),
        "time_s": float(getattr(history, "time_s", 0.0)),
    }


def _state_fingerprint(
    state: Maxwell3DState,
    pic: HybridPIC,
    electron_energy: ElectronEnergyState | None,
    ionization_state: DeuteriumIonizationState | None,
    circuit_state: CircuitState | None,
) -> dict[str, Any]:
    hasher = hashlib.sha256()
    names: list[str] = []
    _hash_array(hasher, names, "E.Ex_edge", state.E.Ex_edge)
    _hash_array(hasher, names, "E.Ey_edge", state.E.Ey_edge)
    _hash_array(hasher, names, "E.Ez_edge", state.E.Ez_edge)
    _hash_array(hasher, names, "B.Bx_face", state.B.Bx_face)
    _hash_array(hasher, names, "B.By_face", state.B.By_face)
    _hash_array(hasher, names, "B.Bz_face", state.B.Bz_face)
    if electron_energy is not None:
        _hash_array(
            hasher,
            names,
            "electron_energy.electron_energy_J_m3",
            electron_energy.electron_energy_J_m3,
        )
        _hash_array(
            hasher,
            names,
            "electron_energy.electron_temperature_K",
            electron_energy.electron_temperature_K,
        )
        _hash_array(
            hasher,
            names,
            "electron_energy.ion_temperature_K",
            electron_energy.ion_temperature_K,
        )
    if ionization_state is not None:
        _hash_array(
            hasher,
            names,
            "ionization.neutral_density_m3",
            ionization_state.neutral_density_m3,
        )
        _hash_array(
            hasher,
            names,
            "ionization.ion_density_m3",
            ionization_state.ion_density_m3,
        )
        _hash_array(
            hasher,
            names,
            "ionization.electron_density_m3",
            ionization_state.electron_density_m3,
        )
    for species_index, species in enumerate(pic.species):
        prefix = f"pic.species.{species_index}.{species.name}"
        _hash_scalar(hasher, names, f"{prefix}.mass", species.mass)
        _hash_scalar(hasher, names, f"{prefix}.charge", species.charge)
        _hash_array(hasher, names, f"{prefix}.positions", species.positions)
        _hash_array(hasher, names, f"{prefix}.positions_old", species.positions_old)
        _hash_array(hasher, names, f"{prefix}.velocities", species.velocities)
        _hash_array(hasher, names, f"{prefix}.weights", species.weights)
    if circuit_state is not None:
        _hash_scalar(hasher, names, "circuit.current_A", circuit_state.current_A)
        _hash_scalar(hasher, names, "circuit.charge_C", circuit_state.charge_C)
    return {
        "status": "experimental_terminal_state_fingerprint_not_restart_acceptance",
        "source": HYBRID_PIC_3D_SOURCE,
        "hash_algorithm": "sha256",
        "sha256": hasher.hexdigest(),
        "included_state_count": len(names),
        "included_state_names": names,
        "particle_species_count": len(pic.species),
        "particle_count": _particle_count(pic),
        "can_support_restart_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _hash_array(
    hasher: Any,
    names: list[str],
    name: str,
    value: np.ndarray,
) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    names.append(name)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.view(np.uint8))


def _hash_scalar(
    hasher: Any,
    names: list[str],
    name: str,
    value: float,
) -> None:
    names.append(name)
    hasher.update(name.encode("utf-8"))
    hasher.update(np.asarray([float(value)], dtype=np.float64).view(np.uint8))


def _step_history_summary(
    *,
    step_index: int,
    dt_s: float,
    telemetry: dict[str, Any],
    diagnostics: dict[str, Any],
    circuit_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    electron_energy = telemetry.get("electron_energy")
    if not isinstance(electron_energy, dict):
        electron_energy = {}
    heat_flux = (
        electron_energy.get("heat_flux")
        if isinstance(electron_energy.get("heat_flux"), dict)
        else {}
    )
    closure_validity = (
        electron_energy.get("closure_validity")
        if isinstance(electron_energy.get("closure_validity"), dict)
        else {}
    )
    kinetic_yield = telemetry.get("kinetic_yield")
    if not isinstance(kinetic_yield, dict):
        kinetic_yield = {}
    field_work = (
        telemetry.get("field_step", {})
        .get("field_work", {})
        if isinstance(telemetry.get("field_step"), dict)
        else {}
    )
    field_step = (
        telemetry.get("field_step")
        if isinstance(telemetry.get("field_step"), dict)
        else {}
    )
    conductivity = (
        field_step.get("conductivity", {})
        if isinstance(field_step, dict)
        else {}
    )
    source_backed_transport = (
        telemetry.get("source_backed_transport")
        if isinstance(telemetry.get("source_backed_transport"), dict)
        else {}
    )
    ionization = (
        telemetry.get("ionization_charge_state")
        if isinstance(telemetry.get("ionization_charge_state"), dict)
        else {}
    )
    circuit_step = (
        circuit_record.get("circuit_step")
        if isinstance(circuit_record, dict)
        and isinstance(circuit_record.get("circuit_step"), dict)
        else {}
    )
    voltage_balance = (
        circuit_record.get("voltage_balance")
        if isinstance(circuit_record, dict)
        and isinstance(circuit_record.get("voltage_balance"), dict)
        else None
    )
    return {
        "step_index": int(step_index),
        "time_s": float((step_index + 1) * dt_s),
        "time_us": float((step_index + 1) * dt_s * 1.0e6),
        "n_particles": int(telemetry.get("n_particles_after", 0)),
        "electron_density_min_m3": _optional_float(
            telemetry.get("electron_density_min_m3")
        ),
        "electron_density_max_m3": _optional_float(
            telemetry.get("electron_density_max_m3")
        ),
        "field_energy_J": float(diagnostics.get("total_energy_J", 0.0)),
        "electric_energy_J": float(diagnostics.get("electric_energy_J", 0.0)),
        "magnetic_energy_J": float(diagnostics.get("magnetic_energy_J", 0.0)),
        "max_abs_div_B_T_per_m": float(
            diagnostics.get("max_abs_div_B_T_per_m", 0.0)
        ),
        "electron_temperature_min_K": _optional_float(
            electron_energy.get("min_electron_temperature_K")
        ),
        "electron_temperature_max_K": _optional_float(
            electron_energy.get("max_electron_temperature_K")
        ),
        "electron_energy_status": electron_energy.get("status"),
        "electron_closure_validity_status": closure_validity.get("status"),
        "electron_current_drift_to_c": _optional_float(
            closure_validity.get("current_drift_to_c")
        ),
        "electron_thermal_speed_to_c": _optional_float(
            closure_validity.get("thermal_speed_to_c")
        ),
        "electron_heat_flux_status": heat_flux.get("status"),
        "electron_heat_flux_required_subcycles": _optional_float(
            heat_flux.get("required_subcycles")
        ),
        "electron_heat_flux_dt_stable_s": _optional_float(
            heat_flux.get("dt_stable_s")
        ),
        "cumulative_neutrons": _optional_float(
            kinetic_yield.get("cumulative_neutrons")
        ),
        "j_dot_e_power_W": _optional_float(field_work.get("j_dot_e_power_W")),
        "terminal_current_A": _optional_float(circuit_step.get("current_A")),
        "terminal_udpf_V": _optional_float(circuit_step.get("udpf_V")),
        "terminal_dI_dt_A_s": _optional_float(circuit_step.get("dI_dt_A_s")),
        "circuit_udpf_source": (
            None
            if not isinstance(circuit_record, dict)
            else circuit_record.get("udpf_source")
        ),
        "circuit_voltage_balance": voltage_balance,
        "source_backed_sigma_min_S_m": _optional_float(
            source_backed_transport.get("min_sigma_S_m")
        ),
        "source_backed_sigma_max_S_m": _optional_float(
            source_backed_transport.get("max_sigma_S_m")
        ),
        "source_backed_resistivity_max_ohm_m": _optional_float(
            source_backed_transport.get("max_resistivity_ohm_m")
        ),
        "source_backed_neutral_density_min_m3": _optional_float(
            source_backed_transport.get("min_neutral_density_m3")
        ),
        "source_backed_neutral_density_max_m3": _optional_float(
            source_backed_transport.get("max_neutral_density_m3")
        ),
        "conductivity_effective_max_S_m": _optional_float(
            conductivity.get("max_sigma_effective_S_m")
        ),
        "conductivity_cfl_limited_fraction": _optional_float(
            conductivity.get("cfl_limited_fraction")
        ),
        "conductivity_ohmic_cfl_limit_applied": (
            None
            if conductivity.get("ohmic_cfl_limit_applied") is None
            else bool(conductivity.get("ohmic_cfl_limit_applied"))
        ),
        "ohm_time_centering_theta": _optional_float(
            field_step.get("ohm_solver", {}).get("ohm_time_centering_theta")
            if isinstance(field_step.get("ohm_solver"), dict)
            else None
        ),
        "electric_update_scheme": (
            field_step.get("ohm_solver", {}).get("electric_update_scheme")
            if isinstance(field_step.get("ohm_solver"), dict)
            else None
        ),
        "ionization_fraction_min": _optional_float(
            ionization.get("min_ionization_fraction")
        ),
        "ionization_fraction_max": _optional_float(
            ionization.get("max_ionization_fraction")
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _min_optional(left: Any, right: float | None) -> float | None:
    left_float = _optional_float(left)
    if left_float is None:
        return right
    if right is None:
        return left_float
    return min(left_float, right)


def _max_optional(left: Any, right: float | None) -> float | None:
    left_float = _optional_float(left)
    if left_float is None:
        return right
    if right is None:
        return left_float
    return max(left_float, right)


def _finite_state_packet(
    state: Maxwell3DState,
    pic: HybridPIC,
    electron_energy: ElectronEnergyState | None,
    ionization_state: DeuteriumIonizationState | None,
    circuit_state: CircuitState | None,
) -> dict[str, Any]:
    checks = {
        "electric_field": _finite_arrays(
            state.E.Ex_edge,
            state.E.Ey_edge,
            state.E.Ez_edge,
        ),
        "magnetic_field": _finite_arrays(
            state.B.Bx_face,
            state.B.By_face,
            state.B.Bz_face,
        ),
        "particles": _finite_particles(pic),
        "electron_energy": (
            True
            if electron_energy is None
            else _finite_arrays(
                electron_energy.electron_energy_J_m3,
                electron_energy.electron_temperature_K,
                electron_energy.ion_temperature_K,
            )
        ),
        "ionization_state": (
            True
            if ionization_state is None
            else _finite_arrays(
                ionization_state.neutral_density_m3,
                ionization_state.ion_density_m3,
                ionization_state.electron_density_m3,
                ionization_state.mean_charge_state,
            )
        ),
        "circuit_state": (
            True
            if circuit_state is None
            else bool(
                np.isfinite(circuit_state.current_A)
                and np.isfinite(circuit_state.charge_C)
            )
        ),
    }
    return {
        "status": (
            "finite_candidate_state"
            if all(checks.values())
            else "nonfinite_candidate_state"
        ),
        "all_finite": all(checks.values()),
        "checks": checks,
    }


def _finite_arrays(*arrays: np.ndarray) -> bool:
    return bool(all(np.all(np.isfinite(np.asarray(array))) for array in arrays))


def _finite_particles(pic: HybridPIC) -> bool:
    for species in pic.species:
        if not _finite_arrays(species.positions, species.velocities, species.weights):
            return False
    return True
