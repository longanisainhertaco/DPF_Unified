"""Experimental whole-shot packet for source-grounded candidate DPF runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, sqrt
from typing import Any

from dpf.fields.maxwell_3d import EPSILON_0, SPEED_OF_LIGHT

EXPERIMENTAL_WHOLE_SHOT_STATUS = (
    "experimental_whole_shot_candidate_not_validation"
)

EXPERIMENTAL_WHOLE_SHOT_SOURCE_REFS = (
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "role": "hybrid_em_pic_fluid_runtime_architecture",
    },
    {
        "path": "KnowledgeReference/auluck-2021-dpf-circuit-element.md",
        "role": "field_power_and_circuit_coupling_contract",
    },
    {
        "path": (
            "KnowledgeReference/"
            "experimental-study-of-the-structure-of-the-plasma-current-sheath-on-"
            "the-pf-1000-facility-705bcc83.md"
        ),
        "role": "pf1000_candidate_geometry_features",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md",
        "role": "local_fail_closed_execution_spec",
    },
)

ENGINEERING_REVIEW_FOCUS = (
    "startup_breakdown_flashover_preionization_and_sheath_liftoff",
    "reviewed_pf1000_rods_hollow_anode_and_insulator_material_geometry",
    "microsecond_whole_shot_duration_with_convergence_and_restart_evidence",
    "time_centered_terminal_or_poynting_j_dot_e_power_port_with_residual_budget",
    "accepted_eos_ionization_transport_radiation_ablation_and_anomalous_resistance",
    "accepted_two_temperature_and_pic_collision_stopping_closures",
    "mechanism_separated_thermonuclear_and_beam_target_neutron_authority",
    "same_scope_waveform_field_density_temperature_history_comparison_and_uq",
    "numerical_fidelity_limiter_zero_backend_parity_and_certificate_gate",
)


def build_experimental_whole_shot_packet(
    *,
    declared_scope: str,
    device_name: str,
    requested_duration_s: float | None,
    step_budget: int | None,
    simulation_telemetry: Mapping[str, Any] | None,
    upstream_packets: Mapping[str, Any] | None,
    grid_spacing_m: Sequence[float] | None = None,
    dt_s: float | None = None,
    vacuum_cfl: float = 0.95,
) -> dict[str, Any]:
    """Build the explicit engineer-review packet for an experimental shot run.

    The packet is intentionally non-promoting.  It records which source-backed
    candidate modules were exercised and which whole-shot requirements still
    need engineering review before any validation or certificate claim.
    """

    simulation = _mapping(simulation_telemetry)
    upstream = _mapping(upstream_packets)
    target_time_s = _first_float(
        simulation.get("target_time_s"),
        requested_duration_s,
    )
    final_time_s = _optional_float(simulation.get("final_time_s"))
    duration_request_satisfied = _duration_request_satisfied(
        explicit=simulation.get("duration_request_satisfied"),
        final_time_s=final_time_s,
        target_time_s=target_time_s,
    )
    duration_gap_s = (
        None
        if target_time_s is None or final_time_s is None
        else max(float(target_time_s) - float(final_time_s), 0.0)
    )
    duration_plan = build_whole_shot_duration_plan(
        grid_spacing_m=grid_spacing_m,
        dt_s=_first_float(dt_s, simulation.get("dt_s")),
        target_time_s=target_time_s,
        step_budget=step_budget,
        vacuum_cfl=vacuum_cfl,
        n_particles_initial=_optional_int(simulation.get("n_particles_initial")),
        n_particles_final=_optional_int(simulation.get("n_particles_final")),
        n_steps_completed=_optional_int(simulation.get("n_steps_completed")),
    )
    candidate_modules = _candidate_modules(
        simulation=simulation,
        upstream=upstream,
    )
    upstream_statuses = _upstream_statuses(upstream)
    blockers = _active_blockers(
        duration_request_satisfied=duration_request_satisfied,
        duration_gap_s=duration_gap_s,
        duration_plan=duration_plan,
        upstream_statuses=upstream_statuses,
    )
    completed_steps = _optional_int(simulation.get("n_steps_completed"))

    return {
        "status": EXPERIMENTAL_WHOLE_SHOT_STATUS,
        "run_intent": "experimental_whole_shot_engineering_review",
        "declared_scope": declared_scope,
        "device_name": device_name,
        "source_truth_policy": {
            "physics_claim_authority": "local_knowledge_reference_only",
            "candidate_runtime_modules_may_execute": True,
            "external_library_policy": (
                "optional_formula_audit_only_not_source_authority"
            ),
            "reduced_model_promotion_allowed": False,
            "validation_promotion_allowed": False,
        },
        "execution_policy": {
            "first_principles_only_requested": True,
            "reduced_models_used": False,
            "candidate_physics_can_run_engineering_cases": True,
            "candidate_physics_can_support_acceptance": False,
            "same_scope_engineer_review_required": True,
        },
        "duration_request": {
            "requested_duration_s": target_time_s,
            "final_time_s": final_time_s,
            "duration_gap_s": duration_gap_s,
            "duration_request_satisfied": duration_request_satisfied,
            "step_budget": step_budget,
            "steps_completed": completed_steps,
            "termination_reason": simulation.get("termination_reason"),
            "duration_status": _duration_status(
                target_time_s=target_time_s,
                duration_request_satisfied=duration_request_satisfied,
            ),
        },
        "duration_plan": duration_plan,
        "simulation_status": {
            "status": simulation.get("status"),
            "finite_state": simulation.get("finite_state"),
            "retained_step_result_count": simulation.get(
                "retained_step_result_count"
            ),
            "history_stride": simulation.get("history_stride"),
            "max_step_results": simulation.get("max_step_results"),
        },
        "can_run_experimental_shot_attempt": (
            completed_steps is not None
            and completed_steps > 0
            and str(simulation.get("status", "")).startswith(
                "candidate_engineering_"
            )
        ),
        "whole_shot_duration_reached": bool(duration_request_satisfied),
        "candidate_modules_executed": candidate_modules,
        "candidate_module_names": _module_names(candidate_modules),
        "candidate_module_count": len(candidate_modules),
        "upstream_packet_statuses": upstream_statuses,
        "active_blockers": blockers,
        "remaining_engineering_review_focus": list(ENGINEERING_REVIEW_FOCUS),
        "engineer_review_packet_required": True,
        "acceptance_state": {
            "can_support_first_principles_acceptance": False,
            "can_support_validation_claims": False,
            "can_support_certificate": False,
            "validated": False,
            "review_decision": "engineering_candidate_only",
            "promotion_path": (
                "replace_candidate_packets_with_reviewed_same_scope_packets_and_"
                "rerun_convergence_uq_certificate_gates"
            ),
        },
        "source_references": list(EXPERIMENTAL_WHOLE_SHOT_SOURCE_REFS),
        "can_support_first_principles_acceptance": False,
}


def build_whole_shot_duration_plan(
    *,
    grid_spacing_m: Sequence[float] | None,
    dt_s: float | None,
    target_time_s: float | None,
    step_budget: int | None,
    vacuum_cfl: float = 0.95,
    n_particles_initial: int | None = None,
    n_particles_final: int | None = None,
    n_steps_completed: int | None = None,
) -> dict[str, Any]:
    """Return a non-promoting whole-shot duration feasibility plan."""

    spacing = _spacing_tuple(grid_spacing_m)
    stable_dt = (
        stable_vacuum_cfl_dt_s(spacing, cfl=vacuum_cfl)
        if spacing is not None
        else None
    )
    required_steps_current_dt = _steps_required(target_time_s, dt_s)
    required_steps_vacuum_cfl = _steps_required(target_time_s, stable_dt)
    max_time_current_dt = (
        None if step_budget is None or dt_s is None else int(step_budget) * dt_s
    )
    max_time_vacuum_cfl = (
        None
        if step_budget is None or stable_dt is None
        else int(step_budget) * stable_dt
    )
    return {
        "status": "experimental_duration_plan_not_validation",
        "time_step_policy": "explicit_maxwell_yee_vacuum_cfl_budget_check",
        "grid_spacing_m": None if spacing is None else list(spacing),
        "dt_s": dt_s,
        "vacuum_cfl": float(vacuum_cfl),
        "stable_vacuum_dt_s": stable_dt,
        "dt_within_vacuum_cfl": (
            None if dt_s is None or stable_dt is None else dt_s <= stable_dt
        ),
        "target_time_s": target_time_s,
        "step_budget": step_budget,
        "steps_required_current_dt": required_steps_current_dt,
        "steps_required_vacuum_cfl_dt": required_steps_vacuum_cfl,
        "max_simulated_time_current_dt_s": max_time_current_dt,
        "max_simulated_time_vacuum_cfl_s": max_time_vacuum_cfl,
        "current_step_budget_satisfies_target": _budget_satisfies(
            step_budget,
            required_steps_current_dt,
        ),
        "vacuum_cfl_step_budget_satisfies_target": _budget_satisfies(
            step_budget,
            required_steps_vacuum_cfl,
        ),
        "ohmic_cfl_limiter_risk": (
            "increasing_dt_lowers_sigma_cfl_and_requires_limiter_zero_proof"
        ),
        "particle_scaling_projection": _particle_scaling_projection(
            required_steps_current_dt=required_steps_current_dt,
            required_steps_vacuum_cfl=required_steps_vacuum_cfl,
            n_particles_initial=n_particles_initial,
            n_particles_final=n_particles_final,
            n_steps_completed=n_steps_completed,
        ),
        "claim_limit": (
            "duration planning is engineering telemetry only and cannot validate "
            "a whole-shot result"
        ),
        "can_support_first_principles_acceptance": False,
    }


def stable_vacuum_cfl_dt_s(
    grid_spacing_m: Sequence[float],
    *,
    cfl: float = 0.95,
) -> float:
    """Return the explicit 3-D Yee vacuum-CFL timestep for grid spacing."""

    spacing = _spacing_tuple(grid_spacing_m)
    if spacing is None:
        raise ValueError("grid_spacing_m must contain three positive spacings")
    if cfl <= 0.0 or cfl > 1.0:
        raise ValueError("cfl must satisfy 0 < cfl <= 1")
    inverse_length = sqrt(sum(1.0 / float(dx) ** 2 for dx in spacing))
    return float(cfl) / (SPEED_OF_LIGHT * inverse_length)


def stable_ohmic_cfl_dt_s(
    max_conductivity_S_m: float,
    *,
    ohmic_cfl_safety: float = 1.0,
    cfl: float = 0.95,
) -> float:
    """Return an explicit Ohmic relaxation timestep for scalar conductivity."""

    sigma = float(max_conductivity_S_m)
    if sigma <= 0.0:
        raise ValueError("max_conductivity_S_m must be positive")
    if ohmic_cfl_safety <= 0.0:
        raise ValueError("ohmic_cfl_safety must be positive")
    if cfl <= 0.0 or cfl > 1.0:
        raise ValueError("cfl must satisfy 0 < cfl <= 1")
    return float(cfl) * float(ohmic_cfl_safety) * EPSILON_0 / sigma


def _candidate_modules(
    *,
    simulation: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    simulation_status = str(simulation.get("status", ""))
    if simulation_status.startswith("candidate_engineering_"):
        modules.append(_module(
            name="package_native_3d_hybrid_em_pic_fluid_loop",
            packet="simulation",
            status=simulation_status,
            role="full_state_advance_candidate_runtime",
        ))

    startup = _packet(upstream, "startup_bvp")
    startup_audit = _mapping(startup.get("candidate_breakdown_audit"))
    if str(startup_audit.get("status", "")).startswith("candidate_"):
        modules.append(_module(
            name="startup_breakdown_liftoff_audit",
            packet="startup_bvp.candidate_breakdown_audit",
            status=startup_audit.get("status"),
            role="breakdown_and_liftoff_telemetry_not_startup_bvp_authority",
        ))

    boundary = _packet(upstream, "boundary_policy")
    conductor_mask = _mapping(boundary.get("conductor_mask"))
    if conductor_mask.get("mask_source") == "candidate_pf1000_rod_hollow_projection":
        modules.append(_module(
            name="pf1000_rod_hollow_conductor_projection",
            packet="boundary_policy.conductor_mask",
            status=conductor_mask.get("status"),
            role="pf1000_rods_hollow_anode_geometry_projection_candidate",
        ))
    if str(boundary.get("status", "")).startswith("candidate_"):
        modules.append(_module(
            name="pml_conductor_particle_boundary_policy",
            packet="boundary_policy",
            status=boundary.get("status"),
            role="field_particle_boundary_candidate_runtime",
        ))

    pic_loading = _packet(upstream, "pic_particle_loading")
    if str(pic_loading.get("status", "")).startswith("candidate"):
        modules.append(_module(
            name="source_geometry_pic_particle_loading",
            packet="pic_particle_loading",
            status=pic_loading.get("status"),
            role="macro_particle_loading_candidate_runtime",
        ))

    power_port = _packet(upstream, "power_port")
    power_channels = _strings(power_port.get("candidate_runtime_channels"))
    if "candidate_volume_j_dot_e_power" in power_channels:
        modules.append(_module(
            name="volume_j_dot_e_power_accounting",
            packet="power_port",
            status=power_port.get("status"),
            role="field_power_diagnostic_candidate_runtime",
        ))
    active_load = _mapping(power_port.get("active_load_decision"))
    if active_load.get("candidate_lagged_volume_j_dot_e_is_active_load") is True:
        modules.append(_module(
            name="lagged_volume_j_dot_e_power_feedback",
            packet="power_port.active_load_decision",
            status=active_load.get("decision"),
            role="active_load_candidate_not_accepted_power_authority",
        ))

    dimensionality = _packet(upstream, "dimensionality_handoff")
    for channel in _strings(dimensionality.get("candidate_runtime_channels")):
        if channel in {
            "candidate_true_3d_grid",
            "candidate_hybrid_pic_fluid_solver_family",
            "candidate_separate_electron_energy_source_terms",
            "candidate_kinetic_yield_history",
            "candidate_source_ordered_loop",
        }:
            modules.append(_module(
                name=channel.replace("candidate_", ""),
                packet="dimensionality_handoff",
                status=dimensionality.get("status"),
                role="dimensionality_or_handoff_candidate_runtime",
            ))

    closure = _packet(upstream, "physics_closure")
    for channel in _strings(closure.get("candidate_runtime_channels")):
        if channel in {
            "candidate_electrical_transport_source_terms",
            "candidate_ionization_charge_state_transport",
            "candidate_source_backed_partial_ionized_conductivity",
            "candidate_electron_energy_source_terms",
            "candidate_braginskii_electron_heat_flux",
            "candidate_electron_ion_equilibration_audit",
            "candidate_kinetic_yield_history",
            "candidate_collision_stage_enabled",
            "candidate_plasmapy_community_formula_audit",
        }:
            modules.append(_module(
                name=channel.replace("candidate_", ""),
                packet="physics_closure",
                status=closure.get("status"),
                role="physics_closure_candidate_runtime",
            ))

    neutron = _packet(upstream, "neutron_authority")
    neutron_channels = _strings(neutron.get("candidate_runtime_channels"))
    if neutron_channels:
        modules.append(_module(
            name="kinetic_neutron_yield_history",
            packet="neutron_authority",
            status=neutron.get("status"),
            role="neutron_yield_candidate_not_mechanism_separated_authority",
        ))

    return _dedupe_modules(modules)


def _module(
    *,
    name: str,
    packet: str,
    status: Any,
    role: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "packet": packet,
        "status": str(status) if status is not None else "unknown",
        "role": role,
        "can_support_first_principles_acceptance": False,
    }


def _dedupe_modules(modules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for module in modules:
        key = (str(module.get("name")), str(module.get("packet")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(module))
    return deduped


def _module_names(modules: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for module in modules:
        name = str(module.get("name"))
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _upstream_statuses(upstream: Mapping[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for key, value in upstream.items():
        packet = _mapping(value)
        if "status" in packet:
            statuses[str(key)] = str(packet["status"])
    return statuses


def _active_blockers(
    *,
    duration_request_satisfied: bool | None,
    duration_gap_s: float | None,
    duration_plan: Mapping[str, Any],
    upstream_statuses: Mapping[str, str],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if duration_request_satisfied is False:
        blockers.append({
            "blocker": "requested_whole_shot_duration_not_reached",
            "duration_gap_s": duration_gap_s,
        })
    if duration_plan.get("current_step_budget_satisfies_target") is False:
        blockers.append({
            "blocker": "step_budget_below_required_current_dt_steps",
            "required_steps": duration_plan.get("steps_required_current_dt"),
            "step_budget": duration_plan.get("step_budget"),
        })
    if duration_plan.get("vacuum_cfl_step_budget_satisfies_target") is False:
        blockers.append({
            "blocker": "step_budget_below_vacuum_cfl_minimum_steps",
            "required_steps": duration_plan.get("steps_required_vacuum_cfl_dt"),
            "step_budget": duration_plan.get("step_budget"),
        })
    particle_projection = _mapping(duration_plan.get("particle_scaling_projection"))
    if particle_projection.get("projection_status") == "particle_growth_projection_high":
        blockers.append({
            "blocker": "particle_population_growth_for_target_duration",
            "projected_particles_current_dt": particle_projection.get(
                "projected_particles_current_dt"
            ),
            "projected_particles_vacuum_cfl_dt": particle_projection.get(
                "projected_particles_vacuum_cfl_dt"
            ),
        })
    for key, status in upstream_statuses.items():
        if status.startswith("blocked") or "not_available" in status:
            blockers.append({
                "blocker": f"{key}_blocked",
                "status": status,
            })
    blockers.extend(
        {"blocker": focus, "status": "requires_engineering_review"}
        for focus in ENGINEERING_REVIEW_FOCUS
    )
    return blockers


def _duration_status(
    *,
    target_time_s: float | None,
    duration_request_satisfied: bool | None,
) -> str:
    if target_time_s is None:
        return "no_requested_duration_attached"
    if duration_request_satisfied is True:
        return "requested_duration_reached"
    if duration_request_satisfied is False:
        return "requested_duration_not_reached"
    return "requested_duration_status_unknown"


def _duration_request_satisfied(
    *,
    explicit: Any,
    final_time_s: float | None,
    target_time_s: float | None,
) -> bool | None:
    if isinstance(explicit, bool):
        return explicit
    if explicit is not None:
        return str(explicit).lower() == "true"
    if final_time_s is None or target_time_s is None:
        return None
    return final_time_s >= target_time_s


def _steps_required(target_time_s: float | None, dt_s: float | None) -> int | None:
    if target_time_s is None or dt_s is None:
        return None
    if target_time_s <= 0.0 or dt_s <= 0.0:
        return None
    return int(ceil(target_time_s / dt_s))


def _budget_satisfies(step_budget: int | None, required_steps: int | None) -> bool | None:
    if step_budget is None or required_steps is None:
        return None
    return int(step_budget) >= int(required_steps)


def _particle_scaling_projection(
    *,
    required_steps_current_dt: int | None,
    required_steps_vacuum_cfl: int | None,
    n_particles_initial: int | None,
    n_particles_final: int | None,
    n_steps_completed: int | None,
) -> dict[str, Any]:
    if (
        n_particles_initial is None
        or n_particles_final is None
        or n_steps_completed is None
        or n_steps_completed <= 0
    ):
        return {
            "projection_status": "particle_growth_projection_unavailable",
            "can_support_first_principles_acceptance": False,
        }
    growth_per_step = (n_particles_final - n_particles_initial) / n_steps_completed
    projected_current = _project_particles(
        n_particles_initial,
        growth_per_step,
        required_steps_current_dt,
    )
    projected_cfl = _project_particles(
        n_particles_initial,
        growth_per_step,
        required_steps_vacuum_cfl,
    )
    high_projection = any(
        value is not None and value > 1_000_000
        for value in (projected_current, projected_cfl)
    )
    return {
        "projection_status": (
            "particle_growth_projection_high"
            if high_projection
            else "particle_growth_projection_engineering_estimate"
        ),
        "n_particles_initial": n_particles_initial,
        "n_particles_final": n_particles_final,
        "n_steps_completed": n_steps_completed,
        "observed_particle_growth_per_step": growth_per_step,
        "projected_particles_current_dt": projected_current,
        "projected_particles_vacuum_cfl_dt": projected_cfl,
        "projection_limit": (
            "linear extrapolation from this run only; not a physical population "
            "control model"
        ),
        "can_support_first_principles_acceptance": False,
    }


def _project_particles(
    n_particles_initial: int,
    growth_per_step: float,
    required_steps: int | None,
) -> int | None:
    if required_steps is None:
        return None
    return int(round(n_particles_initial + growth_per_step * required_steps))


def _spacing_tuple(value: Sequence[float] | None) -> tuple[float, float, float] | None:
    if value is None or len(value) != 3:
        return None
    spacing = tuple(float(item) for item in value)
    if any(item <= 0.0 for item in spacing):
        return None
    return spacing


def _packet(upstream: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _mapping(upstream.get(key))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence):
        return {str(item) for item in value}
    return set()


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = _optional_float(value)
        if number is not None:
            return number
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
