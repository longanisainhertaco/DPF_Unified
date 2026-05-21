"""Fail-closed first-principles MHD run-mode metadata.

The first-principles path is initially a scoped PF-1000/Akel readiness shell.
It labels the execution path, separates reduced-model baselines from predictive
evidence, and blocks acceptance until field-derived coupling, same-scope
KnowledgeReference evidence, numerical verification, and physics coverage are
attached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from dpf.validation.circuit_field_coupling import field_coupling_evidence_from_result
from dpf.validation.first_principles_limiters import first_principles_limiter_status
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

FIRST_PRINCIPLES_MHD_MODE = "first_principles_mhd"
FIRST_PRINCIPLES_MHD_EXECUTION_MODE = "mhd"
PF1000_AKEL_SOURCE_SCOPE = "pf1000_16kv_2021_akel_shot12581"
PF1000_AKEL_VALIDATION_SCOPE = "pf1000_16kv_2021_akel"

# The package-native 3-D hybrid EM/PIC-fluid runner labels its results with
# this run-mode.  Such a run is NOT a cylindrical MHD run; it must be judged by
# ``dpf.validation.hybrid_pic_3d`` / the package-native readiness packet, never
# by this legacy cylindrical gate's cylindrical key expectations (Codex S7-A7,
# Sprint 8 WS1).
PACKAGE_NATIVE_3D_RUN_MODE = "first_principles_3d_hybrid_em_pic_fluid"
_THREE_D_DIMENSIONALITIES = {"3d", "three_dimensional", "cartesian_3d"}


def is_package_native_3d_result(result: Mapping[str, object]) -> bool:
    """True when ``result`` came from the package-native 3-D hybrid runner.

    The legacy cylindrical gate must defer to the 3-D readiness packet for
    these runs rather than silently scoring them against cylindrical output
    keys it never produced.
    """

    run_mode = str(result.get("run_mode") or "").strip().lower()
    if run_mode == PACKAGE_NATIVE_3D_RUN_MODE:
        return True
    dimensionality = str(
        result.get("geometry_dimensionality")
        or result.get("dimensionality")
        or ""
    ).strip().lower()
    return dimensionality in _THREE_D_DIMENSIONALITIES

_BASELINE_OUTPUT_KEYS = (
    "Lp_snowplow_nH",
    "phase_model_authority",
    "snowplow_validation",
    "snowplow_phase_validation_status",
    "fc",
    "fm",
)
_CLOSURE_FACTOR_KEYS = (
    "fc",
    "fm",
    "fcr",
    "fmr",
    "current_fraction",
    "mass_fraction",
    "radial_current_fraction",
    "radial_mass_fraction",
)

_REQUIRED_OUTPUTS: dict[str, tuple[str, ...]] = {
    "current_waveform": ("I_MA", "current_A", "current"),
    "voltage_waveform": ("V_kV", "voltage_V", "voltage"),
    "field_derived_plasma_inductance": (
        "field_derived_inductance",
        "magnetic_energy_inductance",
        "Lp_mhd_nH",
    ),
    "dLdt_or_back_emf": ("dL_dt", "dLdt", "dLp_dt", "back_emf_V", "back_emf"),
    "sheath_position": (
        "sheath_position",
        "sheath_position_m",
        "z_sheath",
        "r_sheath",
    ),
    "density_field": ("rho", "density", "final_state", "mhd_snapshots"),
    "magnetic_field": ("B", "B_field", "final_state", "mhd_snapshots"),
    "temperature_field": ("T", "Te", "Ti", "final_state", "mhd_snapshots"),
    "phase_labels": ("phases",),
    "energy_balance": (
        "circuit_coupled_energy_verification",
        "energy_balance",
        "E_cap_kJ",
        "E_ind_kJ",
        "E_res_kJ",
        "poynting_balance",
        "dynamic_inductance_power_balance",
    ),
}


@dataclass
class FirstPrinciplesMHDReadiness:
    """Readiness gate for first-principles MHD acceptance."""

    ready: bool
    status: str
    run_mode: str
    execution_mode: str
    validation_scope: str
    source_scope: str
    source_scope_status: str
    satisfied_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    output_status: dict[str, dict[str, object]] = field(default_factory=dict)
    closure_factor_status: dict[str, object] = field(default_factory=dict)
    reduced_model_baselines: dict[str, dict[str, object]] = field(default_factory=dict)
    coupling_interval_authority: list[dict[str, object]] = field(default_factory=list)
    energy_accounting_status: dict[str, object] = field(default_factory=dict)
    startup_initialization_status: dict[str, object] = field(default_factory=dict)
    neutron_yield_authority_status: dict[str, object] = field(default_factory=dict)
    limiter_ledger_status: dict[str, object] = field(default_factory=dict)
    backend_scope_status: dict[str, object] = field(default_factory=dict)
    hybrid_pic_3d_status: dict[str, object] = field(default_factory=dict)
    validity_notes: dict[str, str] = field(default_factory=dict)


def normalize_first_principles_run_mode(mode: str) -> tuple[str, bool]:
    """Map the public first-principles mode onto the current MHD execution path."""
    requested = str(mode or "lee").strip().lower()
    if requested == FIRST_PRINCIPLES_MHD_MODE:
        return FIRST_PRINCIPLES_MHD_EXECUTION_MODE, True
    return requested, False


def _sequence_present(value: Sequence[object]) -> bool:
    if len(value) == 0:
        return False
    return any(item is not None for item in value)


def _value_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return _sequence_present(value)
    return True


def _present_keys(result: Mapping[str, object], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key in result and _value_present(result.get(key))]


def _nested_closure_keys(result: Mapping[str, object]) -> list[str]:
    found = set(_present_keys(result, _CLOSURE_FACTOR_KEYS))
    snowplow_cfg = result.get("snowplow_cfg")
    if isinstance(snowplow_cfg, Mapping):
        for key in _CLOSURE_FACTOR_KEYS:
            if key in snowplow_cfg and snowplow_cfg.get(key) is not None:
                found.add(f"snowplow_cfg.{key}")
    return sorted(found)


def first_principles_output_status(
    result: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Report required first-principles outputs without accepting them."""
    status: dict[str, dict[str, object]] = {}
    for name, keys in _REQUIRED_OUTPUTS.items():
        present = _present_keys(result, keys)
        if not present:
            state = "missing"
        elif name == "field_derived_plasma_inductance" and present == ["Lp_mhd_nH"]:
            state = "candidate_not_validated"
        elif name == "energy_balance" and present == ["dynamic_inductance_power_balance"]:
            state = "diagnostic_not_validated"
        else:
            state = "present_not_validated"
        status[name] = {
            "required": True,
            "status": state,
            "present": bool(present),
            "evidence_keys": present,
        }
    return status


def reduced_model_baseline_authority(
    result: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Classify Lee/snowplow outputs as baselines only."""
    present = _present_keys(result, _BASELINE_OUTPUT_KEYS)
    closure_keys = _nested_closure_keys(result)
    return {
        "lee_snowplow": {
            "output_role": "baseline_reduced_model",
            "validation_status": "not_first_principles_evidence",
            "can_support_first_principles_acceptance": False,
            "result_keys": present,
            "closure_factor_keys": closure_keys,
            "notes": (
                "Lee/RADPF snowplow outputs may support comparison, "
                "initialization, and regression only. They are excluded from "
                "first_principles_mhd predictive scoring."
            ),
        }
    }


def _channel_status(
    result: Mapping[str, object],
    keys: Sequence[str],
    *,
    missing_status: str = "missing",
    present_status: str = "present_not_validated",
) -> dict[str, object]:
    present = _present_keys(result, keys)
    return {
        "status": present_status if present else missing_status,
        "present": bool(present),
        "evidence_keys": present,
        "validated": False,
    }


def first_principles_energy_accounting_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Report field-coupled energy-accounting channels fail-closed."""
    channels = {
        "magnetic_energy_or_field_inductance": _channel_status(
            result,
            ("magnetic_energy_inductance", "field_derived_inductance", "Lp_mhd_nH"),
            present_status="candidate_not_validated",
        ),
        "dLdt_or_back_emf": _channel_status(
            result,
            ("dL_dt", "dLdt", "dLp_dt", "back_emf_V", "back_emf"),
        ),
        "field_poynting_power": _channel_status(
            result,
            ("poynting_power_W", "poynting_power", "poynting_flux", "poynting_balance"),
            present_status="diagnostic_not_validated",
        ),
        "circuit_energy_channels": _channel_status(
            result,
            ("E_cap_kJ", "E_ind_kJ", "E_res_kJ", "energy_balance"),
            present_status="diagnostic_not_validated",
        ),
        "circuit_energy_residual": _channel_status(
            result,
            (
                "circuit_energy_residual_kJ",
                "circuit_energy_residual_fraction",
                "circuit_coupled_energy_verification",
            ),
            present_status="diagnostic_not_validated",
        ),
    }
    missing = [
        name for name, channel in channels.items()
        if channel["present"] is not True
    ]
    field_coupling = result.get("field_coupling_validation")
    validated = isinstance(field_coupling, Mapping) and field_coupling.get("passed") is True
    if missing:
        status = "incomplete"
    elif validated:
        status = "validated"
    else:
        status = "complete_candidate_not_validated"
    return {
        "status": status,
        "required_channels": channels,
        "missing_channels": missing,
        "validated": validated,
        "can_support_first_principles_acceptance": status == "validated",
        "validity_notes": {
            "claim_scope": (
                "Circuit energy channels and density-weighted MHD inductance "
                "are diagnostics until field Poynting power and same-scope "
                "field-coupling evidence pass."
            ),
        },
    }


def first_principles_startup_initialization_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Report startup/sheath initialization readiness fail-closed."""
    channels = {
        "breakdown_model": _channel_status(
            result,
            ("breakdown_model", "breakdown_evidence", "flashover_model"),
        ),
        "preionization_state": _channel_status(
            result,
            ("preionization", "initial_preionization", "preionization_settings"),
        ),
        "initial_plasma_distribution": _channel_status(
            result,
            ("initial_plasma_distribution", "startup_sheath_initialization"),
            present_status="scaffold_not_validated",
        ),
        "electrode_boundary_conditions": _channel_status(
            result,
            (
                "electrode_boundary_conditions",
                "electrode_bc",
                "boundary_conditions",
            ),
            present_status="implemented_not_validated",
        ),
        "resolved_sheath_position": _channel_status(
            result,
            ("sheath_position", "z_sheath", "r_sheath", "z_sheath_cm"),
            present_status="candidate_not_validated",
        ),
    }
    missing = [
        name for name, channel in channels.items()
        if channel["present"] is not True
    ]
    startup_record = result.get("startup_sheath_initialization")
    accepted = (
        isinstance(startup_record, Mapping)
        and startup_record.get("can_support_first_principles_startup") is True
        and not missing
    )
    if accepted:
        status = "accepted"
    elif missing:
        status = "incomplete"
    else:
        status = "scaffold_not_validated"
    return {
        "status": status,
        "required_channels": channels,
        "missing_channels": missing,
        "can_support_first_principles_acceptance": accepted,
        "validity_notes": {
            "claim_scope": (
                "First-principles startup requires source-backed breakdown, "
                "preionization, initial plasma distribution, electrode boundary "
                "conditions, and resolved sheath-position evidence. A seeded "
                "thin layer or snowplow sheath position is initialization "
                "scaffold only until validated."
            ),
        },
    }


def first_principles_intervals_from_sources(
    times_us: Sequence[object] | None,
    coupling_sources: Sequence[object] | None,
) -> list[dict[str, object]]:
    """Build compact first-principles interval labels from coupling-source history."""
    if coupling_sources is None:
        return []
    sources = [str(item) for item in coupling_sources]
    if not sources:
        return []
    times = list(times_us) if times_us is not None else []
    intervals: list[dict[str, object]] = []
    start = 0

    def record(end: int) -> None:
        raw = sources[start]
        if raw == "snowplow":
            label = "snowplow_loaded"
            authority = "baseline_reduced_model"
        elif raw == "mhd_blend":
            label = "handoff"
            authority = "engineering_blend"
        else:
            label = "field_coupled"
            authority = "field_derived_candidate"
        item: dict[str, object] = {
            "interval_label": label,
            "authority": authority,
            "coupling_source": raw,
            "validated": False,
            "can_support_first_principles_acceptance": False,
        }
        if start < len(times):
            item["t_start_us"] = times[start]
        if end - 1 < len(times):
            item["t_end_us"] = times[end - 1]
        intervals.append(item)

    for idx in range(1, len(sources)):
        if sources[idx] != sources[start]:
            record(idx)
            start = idx
    record(len(sources))
    return intervals


def _field_coupling_evidence(result: Mapping[str, object]) -> Mapping[str, object]:
    existing = result.get("field_coupling_validation")
    if isinstance(existing, Mapping):
        return existing
    return field_coupling_evidence_from_result(result)


def _closure_factor_status(result: Mapping[str, object]) -> dict[str, object]:
    keys = _nested_closure_keys(result)
    return {
        "present": bool(keys),
        "keys": keys,
        "allowed_role": "baseline_reduced_model_only",
        "can_support_first_principles_acceptance": False,
    }


def first_principles_backend_scope_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Classify backend eligibility for current first-principles acceptance."""
    backend = str(result.get("backend") or "").strip()
    requested_backend = str(result.get("requested_backend") or "").strip()
    requested_run_mode = str(result.get("requested_run_mode") or "").strip()
    normalized = backend.lower()
    requested = requested_backend.lower() or requested_run_mode.lower()

    if not normalized:
        return {
            "status": "backend_missing",
            "backend": backend,
            "requested_backend": requested_backend,
            "requested_run_mode": requested_run_mode,
            "can_support_first_principles_acceptance": False,
            "blocked_backends": ["metal", "mlx", "athena", "athenak", "hybrid"],
            "reason": "No effective backend label is attached to the run result.",
        }

    python_instrumented = normalized.startswith("python") and "fallback" not in normalized
    if python_instrumented:
        return {
            "status": "python_cylindrical_instrumented",
            "backend": backend,
            "requested_backend": requested_backend,
            "requested_run_mode": requested_run_mode,
            "can_support_first_principles_acceptance": True,
            "required_limiter_telemetry": "first_principles_limiter_ledger",
            "reason": (
                "Current first-principles acceptance scope is limited to the "
                "Python cylindrical MHD path with result-bound limiter telemetry."
            ),
        }

    backend_tokens = ("athenak", "athena", "metal", "mlx", "hybrid")
    blocked_token = next(
        (
            token
            for token in backend_tokens
            if token in requested
        ),
        None,
    )
    if blocked_token is None:
        blocked_token = next(
            (
                token
                for token in backend_tokens
                if token in normalized
            ),
            normalized,
        )
    return {
        "status": "backend_scope_blocked",
        "backend": backend,
        "requested_backend": requested_backend,
        "requested_run_mode": requested_run_mode,
        "blocked_backend": blocked_token,
        "can_support_first_principles_acceptance": False,
        "required_limiter_telemetry": "backend_native_first_principles_limiter_ledger",
        "reason": (
            "This backend is runnable engineering infrastructure, but it is "
            "outside first-principles acceptance scope until backend-native "
            "limiter/fallback telemetry and parity evidence are attached."
        ),
    }


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _mapping_passed(result: Mapping[str, object], key: str) -> bool:
    value = result.get(key)
    return isinstance(value, Mapping) and value.get("passed") is True


def _neutron_history_has(result: Mapping[str, object], *keys: str) -> bool:
    history = result.get("yield_time_resolved")
    if not isinstance(history, Mapping):
        return False
    if not (
        _value_present(history.get("t_s"))
        or _value_present(history.get("times_us"))
    ):
        return False
    return any(_value_present(history.get(key)) for key in keys)


def first_principles_neutron_yield_authority_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Classify neutron-yield outputs for first-principles authority.

    The gate is intentionally stricter than "a number exists". A total neutron
    yield can support first-principles acceptance only when thermonuclear yield
    is integrated from a resolved field history, beam-target yield comes from a
    kinetic/hybrid beam model rather than Lee/Saw calibration, and same-scope
    scalar yield, timing, spectrum, anisotropy, detector response, and UQ
    evidence all pass together.
    """
    yield_block = result.get("neutron_yield_details")
    if not isinstance(yield_block, Mapping):
        yield_block = result.get("neutron_yield")
    if not isinstance(yield_block, Mapping):
        yield_block = {}

    y_thermo = _finite_number(yield_block.get("Y_thermonuclear"))
    y_beam = _finite_number(yield_block.get("Y_beam_target"))
    y_total = _finite_number(yield_block.get("Y_neutron"))
    if y_total is None and y_thermo is not None and y_beam is not None:
        y_total = y_thermo + y_beam

    thermo_history = _neutron_history_has(
        result,
        "dY_th",
        "dY_thermo",
        "thermonuclear_rate",
        "thermonuclear_yield_rate",
    )
    beam_history = _neutron_history_has(
        result,
        "dY_bt",
        "beam_target_rate",
        "beam_target_yield_rate",
    )
    final_state_only = (
        y_thermo is not None
        and not thermo_history
        and _value_present(result.get("final_state"))
    )

    limiter_status = first_principles_limiter_status(result)
    limiter_active = (
        int(limiter_status.get("acceptance_blocking_activation_count") or 0) > 0
    )

    kinetic_beam = result.get("kinetic_beam_neutron_model")
    if not isinstance(kinetic_beam, Mapping):
        kinetic_beam = result.get("beam_tracker")
    kinetic_beam_accepted = (
        isinstance(kinetic_beam, Mapping)
        and kinetic_beam.get("can_support_first_principles_neutron_yield") is True
        and kinetic_beam.get("validation_status") in {"accepted", "validated"}
    )

    scalar_yield_passed = _mapping_passed(result, "neutron_yield_validation")
    timing_passed = _mapping_passed(result, "neutron_mechanism_timing_validation")
    spectrum_passed = _mapping_passed(result, "neutron_spectrum_validation")
    anisotropy_passed = _mapping_passed(result, "neutron_anisotropy_validation")
    detector_passed = _mapping_passed(result, "neutron_detector_response_validation")
    uq_passed = _mapping_passed(result, "neutron_uncertainty_validation")
    numerical_passed = _mapping_passed(result, "mhd_numerical_fidelity")
    physics_passed = _mapping_passed(result, "physics_fidelity_evidence")

    produced = y_total is not None or y_thermo is not None or y_beam is not None
    beam_reduced = (y_beam or 0.0) > 0.0 and not kinetic_beam_accepted

    blockers: list[str] = []
    if not produced:
        blockers.append("first_principles_neutron_yield_not_computed")
    if y_thermo is not None and not thermo_history:
        blockers.append("thermonuclear_yield_not_integrated_from_field_history")
    if final_state_only:
        blockers.append("thermonuclear_yield_uses_final_state_duration_approximation")
    if beam_reduced:
        blockers.append("beam_target_yield_uses_reduced_or_calibrated_model")
    if (y_beam or 0.0) > 0.0 and not beam_history:
        blockers.append("beam_target_time_history_missing")
    if limiter_active:
        blockers.append("engineering_limiter_active")
    if not scalar_yield_passed:
        blockers.append("same_scope_scalar_neutron_yield_validation_missing")
    if not timing_passed:
        blockers.append("same_scope_neutron_timing_validation_missing")
    if not spectrum_passed:
        blockers.append("same_scope_neutron_spectrum_validation_missing")
    if not anisotropy_passed:
        blockers.append("same_scope_neutron_anisotropy_validation_missing")
    if not detector_passed:
        blockers.append("same_scope_detector_activation_response_validation_missing")
    if not uq_passed:
        blockers.append("same_scope_neutron_uncertainty_validation_missing")
    if not kinetic_beam_accepted:
        blockers.append("kinetic_or_hybrid_beam_target_model_missing")
    if not numerical_passed:
        blockers.append("mhd_numerical_fidelity_packet_missing")
    if not physics_passed:
        blockers.append("physics_fidelity_packet_missing")

    passed = produced and not blockers
    return {
        "passed": passed,
        "status": (
            "first_principles_neutron_yield_ready"
            if passed
            else ("blocked" if produced else "not_produced")
        ),
        "validation_tier": 5,
        "total_yield_n": y_total,
        "can_support_first_principles_acceptance": passed,
        "ten_percent_paper_yield_accuracy": {
            "target_relative_error": 0.10,
            "status": "eligible" if passed else "blocked",
            "claim": (
                "A 10 percent paper-yield comparison is an acceptance "
                "criterion only after same-scope scalar yield, mechanism "
                "timing, spectrum, anisotropy, detector response, UQ, and "
                "kinetic beam-target evidence pass together."
            ),
        },
        "mechanisms": {
            "thermonuclear": {
                "yield_n": y_thermo,
                "authority": (
                    "resolved_field_history_candidate"
                    if thermo_history
                    else (
                        "final_state_duration_approximation"
                        if y_thermo is not None else "not_produced"
                    )
                ),
                "can_support_component_acceptance": (
                    bool(thermo_history)
                    and not limiter_active
                    and numerical_passed
                    and physics_passed
                ),
            },
            "beam_target": {
                "yield_n": y_beam,
                "authority": (
                    "kinetic_hybrid_candidate"
                    if kinetic_beam_accepted
                    else (
                        "baseline_reduced_model"
                        if (y_beam or 0.0) > 0.0 else "not_produced"
                    )
                ),
                "can_support_component_acceptance": kinetic_beam_accepted,
            },
        },
        "required_validation": {
            "same_scope_scalar_yield": scalar_yield_passed,
            "same_scope_mechanism_timing": timing_passed,
            "same_scope_spectrum": spectrum_passed,
            "same_scope_anisotropy": anisotropy_passed,
            "same_scope_detector_response": detector_passed,
            "same_scope_uncertainty": uq_passed,
            "mhd_numerical_fidelity": numerical_passed,
            "physics_fidelity": physics_passed,
            "kinetic_or_hybrid_beam_target_model": kinetic_beam_accepted,
        },
        "blockers": blockers,
        "validity_notes": {
            "source_scope": (
                "Local KnowledgeReference evidence must be same-scope with "
                "the paper/device/shot being compared; cross-device neutron "
                "yield packets cannot pass this authority gate."
            ),
            "reduced_model_boundary": (
                "Lee/Saw beam-target yield and empirical beam fractions are "
                "baseline reduced-model estimates, not first-principles "
                "predictive authority."
            ),
        },
    }


def _package_native_3d_deferral_readiness(
    result: Mapping[str, object],
    *,
    run_mode: str,
    execution_mode: str,
    validation_scope: str,
    source_scope: str,
    source_scope_status: str,
) -> FirstPrinciplesMHDReadiness:
    """Return a blocked readiness that defers a 3-D run to its own gate.

    The legacy cylindrical gate cannot accept or reject a package-native 3-D
    hybrid run.  It returns ``ready=False`` with a single explicit blocker
    pointing at ``hybrid_pic_3d_readiness`` so no caller can mistake the
    silence-on-cylindrical-keys for either acceptance or a cylindrical
    rejection (Codex S7-A7, Sprint 8 WS1).
    """

    hybrid_pic_3d_status = hybrid_pic_3d_readiness_status(result)
    blocker = (
        "package_native_3d_run_detected: this run is judged by the "
        "package-native hybrid_pic_3d readiness packet, not the legacy "
        "cylindrical first_principles_mhd gate. The cylindrical gate does "
        "not accept or reject 3-D runs."
    )
    return FirstPrinciplesMHDReadiness(
        ready=False,
        status="blocked_package_native_3d_run_uses_hybrid_pic_3d_gate",
        run_mode=run_mode,
        execution_mode=execution_mode,
        validation_scope=validation_scope,
        source_scope=source_scope,
        source_scope_status=source_scope_status,
        satisfied_evidence=[],
        missing_evidence=["package_native_3d_uses_hybrid_pic_3d_gate"],
        blockers=[blocker],
        hybrid_pic_3d_status=hybrid_pic_3d_status,
        validity_notes={
            "gate_separation": (
                "Package-native 3-D hybrid EM/PIC-fluid runs are gated by "
                "dpf.validation.hybrid_pic_3d.hybrid_pic_3d_readiness_status. "
                "The legacy cylindrical MHD gate does not apply cylindrical "
                "key expectations (current/density/sheath_position) to 3-D "
                "runs and cannot grant or deny their acceptance."
            ),
            "authority_packet": (
                "hybrid_pic_3d_readiness is the authoritative 3-D gate; its "
                "missing_capabilities list governs 3-D acceptance."
            ),
        },
    )


def first_principles_mhd_readiness_report(
    result: Mapping[str, object],
    *,
    preset_name: str = "",
    validation_scope: str = "",
    source_scope: str = "",
    source_scope_status: str = "",
    run_mode: str = FIRST_PRINCIPLES_MHD_MODE,
    execution_mode: str = FIRST_PRINCIPLES_MHD_EXECUTION_MODE,
) -> FirstPrinciplesMHDReadiness:
    """Gate first-principles MHD acceptance on fail-closed evidence."""
    validation_scope = str(validation_scope or "")
    source_scope = str(source_scope or "")
    source_scope_status = str(source_scope_status or "")
    preset_name = str(preset_name or "")
    run_mode = str(run_mode or FIRST_PRINCIPLES_MHD_MODE)
    execution_mode = str(execution_mode or FIRST_PRINCIPLES_MHD_EXECUTION_MODE)

    # Codex S7-A7 / Sprint 8 WS1: a package-native 3-D hybrid run is outside
    # this legacy cylindrical gate's authority.  Do NOT score it against
    # cylindrical output keys (I_MA, rho, sheath_position, ...) -- that would
    # silently mark unrelated channels missing.  Defer to the package-native
    # 3-D readiness packet and stay blocked here.
    if is_package_native_3d_result(result):
        return _package_native_3d_deferral_readiness(
            result,
            run_mode=run_mode,
            execution_mode=execution_mode,
            validation_scope=validation_scope,
            source_scope=source_scope,
            source_scope_status=source_scope_status,
        )

    output_status = first_principles_output_status(result)
    closure_status = _closure_factor_status(result)
    baselines = reduced_model_baseline_authority(result)
    energy_status = first_principles_energy_accounting_status(result)
    startup_status = first_principles_startup_initialization_status(result)
    neutron_status = first_principles_neutron_yield_authority_status(result)
    limiter_status = first_principles_limiter_status(result)
    backend_status = first_principles_backend_scope_status(result)
    hybrid_pic_3d_status = hybrid_pic_3d_readiness_status(result)
    intervals = list(result.get("first_principles_intervals") or [])
    coupling_evidence = _field_coupling_evidence(result)

    satisfied: list[str] = []
    missing: list[str] = []
    blockers: list[str] = []

    for name, item in output_status.items():
        if item["present"]:
            satisfied.append(name)
        else:
            missing.append(name)
            blockers.append(f"{name}: missing required first-principles output")

    if source_scope != PF1000_AKEL_SOURCE_SCOPE or validation_scope != PF1000_AKEL_VALIDATION_SCOPE:
        missing.append("pf1000_akel_same_scope")
        blockers.append(
            "first_principles_mhd is initially scoped to PF-1000/Akel "
            f"({PF1000_AKEL_SOURCE_SCOPE}, {PF1000_AKEL_VALIDATION_SCOPE}); "
            f"got preset={preset_name or 'unknown'}, source_scope={source_scope or 'unset'}, "
            f"validation_scope={validation_scope or 'unset'}."
        )

    if "blocked_by_review" in source_scope_status:
        missing.append("accepted_same_scope_akel_digitization")
        blockers.append(
            "PF-1000/Akel source scope is blocked_by_review; accepted "
            "same-scope waveform/current-dip evidence is required before "
            "first-principles acceptance."
        )

    if closure_status["present"]:
        missing.append("reduced_model_active_closure_rejected")
        blockers.append(
            "Lee/RADPF closure factors are present in the run metadata; "
            "first_principles_mhd must keep them baseline-only and out of "
            "predictive scoring."
        )

    if limiter_status["can_support_first_principles_acceptance"] is True:
        satisfied.append("limiter_zero_acceptance")
    else:
        missing.append("acceptance_blocking_limiter_activation")
        active_ids = ", ".join(
            str(item) for item in limiter_status.get("activated_acceptance_blockers", [])
        )
        if not active_ids:
            active_ids = str(limiter_status.get("validation_status") or "missing")
        blockers.append(
            "First-principles run has no accepted zero-limiter ledger; "
            f"acceptance-blocking limiter status: {active_ids}."
        )

    if backend_status["can_support_first_principles_acceptance"] is True:
        satisfied.append("instrumented_backend_scope")
    else:
        missing.append("instrumented_backend_scope")
        blockers.append(
            "First-principles backend scope is not accepted; "
            f"{backend_status.get('reason', backend_status.get('status'))}"
        )

    if hybrid_pic_3d_status["can_support_first_principles_acceptance"] is True:
        satisfied.append("hybrid_pic_3d_first_principles_core")
    else:
        missing.append("hybrid_pic_3d_first_principles_core")
        missing_capabilities = ", ".join(
            str(item)
            for item in hybrid_pic_3d_status.get("missing_capabilities", [])
        )
        blockers.append(
            "Full first-principles DPF acceptance requires a reviewed 3-D "
            "hybrid PIC-fluid core informed by the local 2604.09032v1 source; "
            f"missing capabilities: {missing_capabilities or 'unknown'}."
        )

    if energy_status["can_support_first_principles_acceptance"] is True:
        satisfied.append("field_coupled_energy_accounting")
    else:
        missing.append("field_coupled_energy_accounting")
        missing_channels = ", ".join(energy_status["missing_channels"]) or (
            f"status={energy_status['status']}"
        )
        blockers.append(
            "Field-coupled energy accounting is not accepted for "
            f"first_principles_mhd; missing or unvalidated channels: {missing_channels}."
        )

    if startup_status["can_support_first_principles_acceptance"] is True:
        satisfied.append("first_principles_startup_initialization")
    else:
        missing.append("first_principles_startup_initialization")
        missing_channels = ", ".join(startup_status["missing_channels"]) or (
            f"status={startup_status['status']}"
        )
        blockers.append(
            "Startup/sheath initialization is not accepted for "
            f"first_principles_mhd; missing or unvalidated channels: {missing_channels}."
        )

    if coupling_evidence.get("passed") is True:
        satisfied.append("validated_field_coupling_packet")
    else:
        missing.append("validated_field_coupling_packet")
        blockers.append(
            "Field-derived coupling is not validated for one same-scope packet; "
            "inductance, dL/dt/back-EMF, Poynting power, energy balance, "
            "handoff metadata, and KR comparison must all pass."
        )

    if neutron_status["can_support_first_principles_acceptance"] is True:
        satisfied.append("first_principles_neutron_yield_authority")
    else:
        missing.append("first_principles_neutron_yield_authority")
        neutron_blockers = ", ".join(
            str(item) for item in neutron_status.get("blockers", [])
        ) or f"status={neutron_status['status']}"
        blockers.append(
            "First-principles neutron-yield authority is not accepted; "
            f"remaining blockers: {neutron_blockers}."
        )

    physics = result.get("physics_fidelity_evidence")
    if isinstance(physics, Mapping) and physics.get("passed") is True:
        satisfied.append("physics_fidelity_packet")
    else:
        missing.append("physics_fidelity_packet")
        blockers.append(
            "Physics-fidelity packet is missing or not accepted for startup, "
            "ionization/EOS/two-temperature/transport/radiation, and late-pinch scope."
        )

    numerical = result.get("mhd_numerical_fidelity")
    if isinstance(numerical, Mapping) and numerical.get("passed") is True:
        satisfied.append("mhd_numerical_fidelity_packet")
    else:
        missing.append("mhd_numerical_fidelity_packet")
        blockers.append(
            "MHD numerical-fidelity packet is missing or incomplete for "
            "cylindrical terms, diffusion/heating, circuit energy, backend "
            "parity, and restart/reproducibility."
        )

    ready = not missing
    return FirstPrinciplesMHDReadiness(
        ready=ready,
        status="first_principles_ready" if ready else "blocked",
        run_mode=run_mode,
        execution_mode=execution_mode,
        validation_scope=validation_scope,
        source_scope=source_scope,
        source_scope_status=source_scope_status,
        satisfied_evidence=sorted(set(satisfied)),
        missing_evidence=sorted(set(missing)),
        blockers=blockers,
        output_status=output_status,
        closure_factor_status=closure_status,
        reduced_model_baselines=baselines,
        coupling_interval_authority=intervals,
        energy_accounting_status=energy_status,
        startup_initialization_status=startup_status,
        neutron_yield_authority_status=neutron_status,
        limiter_ledger_status=limiter_status,
        backend_scope_status=backend_status,
        hybrid_pic_3d_status=hybrid_pic_3d_status,
        validity_notes={
            "scope": (
                "Initial first_principles_mhd acceptance is limited to the "
                "PF-1000/Akel same-scope path while the full 3-D hybrid "
                "PIC-fluid finish line is implemented."
            ),
            "baseline_boundary": (
                "Lee/snowplow outputs are retained for initialization, "
                "comparison, and regression, not predictive authority."
            ),
        },
    )


def annotate_first_principles_mhd_result(
    result: dict[str, Any],
    *,
    preset_name: str,
    validation_scope: str,
    source_scope: str,
    source_scope_status: str,
    requested_mode: str = FIRST_PRINCIPLES_MHD_MODE,
    execution_mode: str = FIRST_PRINCIPLES_MHD_EXECUTION_MODE,
) -> dict[str, Any]:
    """Attach first-principles mode metadata and fail-closed readiness."""
    intervals = first_principles_intervals_from_sources(
        result.get("t_us"),
        result.get("coupling_source"),
    )
    result["run_mode"] = requested_mode
    result["execution_mode"] = execution_mode
    result["validation_scope"] = validation_scope
    result["source_scope"] = source_scope
    result["source_scope_status"] = source_scope_status
    result["first_principles_intervals"] = intervals
    result["reduced_model_baselines"] = reduced_model_baseline_authority(result)
    if "field_coupling_validation" not in result:
        result["field_coupling_validation"] = field_coupling_evidence_from_result(result)
    result["first_principles_energy_accounting"] = (
        first_principles_energy_accounting_status(result)
    )
    result["first_principles_startup_initialization"] = (
        first_principles_startup_initialization_status(result)
    )
    result["first_principles_neutron_yield_authority"] = (
        first_principles_neutron_yield_authority_status(result)
    )
    result["first_principles_backend_scope"] = (
        first_principles_backend_scope_status(result)
    )
    result["hybrid_pic_3d_readiness"] = hybrid_pic_3d_readiness_status(result)
    readiness = first_principles_mhd_readiness_report(
        result,
        preset_name=preset_name,
        validation_scope=validation_scope,
        source_scope=source_scope,
        source_scope_status=source_scope_status,
        run_mode=requested_mode,
        execution_mode=execution_mode,
    )
    result["first_principles_mhd_readiness"] = asdict(readiness)
    return result
