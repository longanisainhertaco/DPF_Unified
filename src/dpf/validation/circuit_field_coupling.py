"""Circuit/field coupling audit for DPF predictive-readiness claims.

This module is intentionally conservative. It records whether a result carries
the signals needed to audit circuit-to-plasma coupling, but it does not treat
those signals as validated field-derived coupling unless KR-backed validation
evidence is attached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np


_KR_SOURCE_BASIS = {
    "auluck_circuit_element": "KnowledgeReference/auluck-2021-dpf-circuit-element.md",
    "beresnyak_mhd_coupling": (
        "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md"
    ),
    "lee_dynamic_resistance": (
        "KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-"
        "s-lee-and-s-h-saw-part-1-basic-course.md"
    ),
}


_REQUIRED_EVIDENCE = {
    "plasma_inductance_series": {
        "source_key": "lee_dynamic_resistance",
        "source_lines": "12098-12128",
        "requirement": (
            "A time-resolved plasma inductance or equivalent coupling signal is "
            "needed before dL/dt, motional impedance, and dynamic resistance can "
            "be audited."
        ),
    },
    "field_derived_inductance": {
        "source_key": "auluck_circuit_element",
        "source_lines": "35-38, 957-991",
        "requirement": (
            "The coupling signal must be distinguished from a reduced "
            "time-varying circuit inductance because the KR source identifies "
            "conceptual differences from a first-principles Poynting-theorem "
            "description."
        ),
    },
    "dLdt_or_back_emf": {
        "source_key": "lee_dynamic_resistance",
        "source_lines": "12098-12128",
        "requirement": (
            "The run needs dL/dt or back-EMF evidence because a changing "
            "inductance introduces motional/dynamic-resistance power that is "
            "not the same as stored inductive energy."
        ),
    },
    "poynting_power_balance": {
        "source_key": "auluck_circuit_element",
        "source_lines": "435-455",
        "requirement": (
            "A field-coupled circuit element must account for Poynting-flux or "
            "equivalent power input at the circuit/plasma interface."
        ),
    },
    "circuit_energy_balance": {
        "source_key": "lee_dynamic_resistance",
        "source_lines": "12098-12128",
        "requirement": (
            "Circuit energy accounting must separate stored inductive energy "
            "from the power transferred through changing plasma inductance."
        ),
    },
    "handoff_transition_metadata": {
        "source_key": "auluck_circuit_element",
        "source_lines": "35-38, 957-991",
        "requirement": (
            "Hybrid snowplow/MHD runs need explicit transition metadata so "
            "snowplow-loaded, blended, and field-coupled intervals are not "
            "collapsed into one unqualified current prediction."
        ),
    },
    "kr_experimental_comparison": {
        "source_key": "auluck_circuit_element",
        "source_lines": "35-38",
        "requirement": (
            "Implemented coupling signals must be compared against a "
            "KR-backed waveform or field-coupling target before they can "
            "support predictive current claims."
        ),
    },
}

_COMPONENT_BY_NORMALIZED = {
    key.lower(): key for key in _REQUIRED_EVIDENCE
}


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _finite_values(value: object, *, limit: int = 16) -> list[float]:
    """Return a bounded list of finite numeric values from a scalar or sequence."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return []
    if hasattr(value, "flat"):
        try:
            values: list[float] = []
            for item in value.flat:
                if len(values) >= limit:
                    break
                values.extend(_finite_values(item, limit=limit - len(values)))
            return values
        except Exception:
            return []
    if hasattr(value, "ravel"):
        try:
            value = value.ravel()
        except Exception:
            return []
    if _is_sequence(value):
        values: list[float] = []
        for item in value:
            if len(values) >= limit:
                break
            values.extend(_finite_values(item, limit=limit - len(values)))
        return values
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    return [number] if isfinite(number) else []


def _series_info(
    result: Mapping[str, object],
    *keys: str,
) -> tuple[str | None, list[float]]:
    for key in keys:
        if key not in result:
            continue
        values = _finite_values(result.get(key))
        if values:
            return key, values
    return None, []


def _record(
    name: str,
    *,
    status: str,
    present: bool,
    validated: bool = False,
    evidence_keys: Sequence[str] = (),
    notes: str,
) -> dict[str, object]:
    meta = _REQUIRED_EVIDENCE[name]
    return {
        "status": status,
        "present": present,
        "validated": validated,
        "source": _KR_SOURCE_BASIS[str(meta["source_key"])],
        "source_lines": meta["source_lines"],
        "requirement": meta["requirement"],
        "evidence_keys": list(evidence_keys),
        "notes": notes,
    }


def field_coupling_component_evidence(
    component: str,
    *,
    validation_scope: str,
    source: str | None = None,
    source_lines: str | None = None,
    notes: str = "",
) -> dict[str, object]:
    """Build line-referenced evidence for one field-coupling component."""
    component_key = _COMPONENT_BY_NORMALIZED.get(
        str(component).strip().lower(),
        str(component).strip(),
    )
    known_component = component_key in _REQUIRED_EVIDENCE
    if known_component:
        meta = _REQUIRED_EVIDENCE[component_key]
        default_source = _KR_SOURCE_BASIS[str(meta["source_key"])]
        default_lines = str(meta["source_lines"])
    else:
        default_source = ""
        default_lines = ""
    source_value = source or default_source
    line_value = source_lines or default_lines
    passed = (
        known_component
        and bool(validation_scope)
        and str(source_value).startswith("KnowledgeReference/")
        and bool(line_value)
    )
    return {
        "passed": passed,
        "validation_tier": "field_coupling",
        "model_role": "field_coupling_component_validation",
        "component": component_key,
        "validation_scope": validation_scope,
        "source": source_value,
        "source_lines": line_value,
        "details": {
            "known_component": known_component,
            "notes": notes,
        },
        "validity_notes": {
            "claim_scope": (
                "This evidence supports one field-coupling component for the "
                "stated validation scope; it does not validate other coupling "
                "components or experimental observables."
            ),
        },
    }


def _valid_component_evidence(
    evidence: object,
    component: str,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "field_coupling_component_validation":
        return None
    if evidence.get("validation_tier") != "field_coupling":
        return None
    evidence_component = _COMPONENT_BY_NORMALIZED.get(
        str(evidence.get("component", "")).strip().lower(),
        str(evidence.get("component", "")).strip(),
    )
    if evidence_component != component:
        return None
    if not str(evidence.get("source", "")).startswith("KnowledgeReference/"):
        return None
    if not evidence.get("validation_scope"):
        return None
    return evidence


def _validated_components(
    result: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], str]]:
    found: dict[str, tuple[Mapping[str, object], str]] = {}

    for container_key in (
        "field_coupling_component_validation",
        "field_coupling_component_validations",
    ):
        container = result.get(container_key)
        if isinstance(container, Mapping):
            for component, candidate in container.items():
                component_key = _COMPONENT_BY_NORMALIZED.get(
                    str(component).strip().lower(),
                    str(component).strip(),
                )
                evidence = _valid_component_evidence(candidate, component_key)
                if evidence is not None:
                    found[component_key] = (evidence, container_key)
        elif isinstance(container, Sequence) and not isinstance(
            container, (str, bytes, bytearray)
        ):
            for candidate in container:
                if not isinstance(candidate, Mapping):
                    continue
                component_key = _COMPONENT_BY_NORMALIZED.get(
                    str(candidate.get("component", "")).strip().lower(),
                    str(candidate.get("component", "")).strip(),
                )
                evidence = _valid_component_evidence(candidate, component_key)
                if evidence is not None:
                    found[component_key] = (evidence, container_key)

    for component in _REQUIRED_EVIDENCE:
        key = f"{component}_validation"
        evidence = _valid_component_evidence(result.get(key), component)
        if evidence is not None:
            found[component] = (evidence, key)

    return found


def _kr_sourced_evidence_passed(evidence: object) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    if evidence.get("passed") is not True:
        return False
    return str(evidence.get("source", "")).startswith("KnowledgeReference/")


def dynamic_inductance_power_balance_from_waveforms(
    times_s: Sequence[float],
    current_A: Sequence[float],
    inductance_H: Sequence[float],
    *,
    relative_tolerance: float = 1.0e-9,
) -> dict[str, object]:
    """Compute Lee dynamic-inductance power-accounting residuals."""
    times = np.asarray(times_s, dtype=float)
    current = np.asarray(current_A, dtype=float)
    inductance = np.asarray(inductance_H, dtype=float)
    n = min(times.size, current.size, inductance.size)
    times = times[:n]
    current = current[:n]
    inductance = inductance[:n]
    finite = np.isfinite(times) & np.isfinite(current) & np.isfinite(inductance)
    times = times[finite]
    current = current[finite]
    inductance = inductance[finite]

    if times.size < 3 or np.any(np.diff(times) <= 0.0):
        return {
            "passed": False,
            "validation_tier": "field_coupling_diagnostic",
            "model_role": "lee_dynamic_inductance_power_accounting",
            "source": _KR_SOURCE_BASIS["lee_dynamic_resistance"],
            "source_lines": "12103-12127",
            "details": {
                "n_samples": int(times.size),
                "limitation": "need at least three strictly increasing finite samples",
            },
        }

    dLdt = np.gradient(inductance, times)
    dIdt = np.gradient(current, times)
    induced_voltage = current * dLdt + inductance * dIdt
    interface_power = induced_voltage * current
    magnetic_energy_derivative = 0.5 * current * current * dLdt + inductance * current * dIdt
    dynamic_resistance_power = 0.5 * current * current * dLdt
    residual = interface_power - (
        magnetic_energy_derivative + dynamic_resistance_power
    )
    scale = np.maximum(
        np.abs(interface_power),
        np.abs(magnetic_energy_derivative) + np.abs(dynamic_resistance_power),
    )
    finite_scale = scale[np.isfinite(scale)]
    max_scale = float(np.max(finite_scale)) if finite_scale.size else 0.0
    max_abs_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    relative_residual = (
        max_abs_residual / max_scale if max_scale > 0.0 else max_abs_residual
    )

    return {
        "passed": relative_residual <= relative_tolerance,
        "validation_tier": "field_coupling_diagnostic",
        "model_role": "lee_dynamic_inductance_power_accounting",
        "source": _KR_SOURCE_BASIS["lee_dynamic_resistance"],
        "source_lines": "12103-12127",
        "details": {
            "n_samples": int(times.size),
            "max_abs_residual_W": max_abs_residual,
            "max_power_scale_W": max_scale,
            "max_relative_residual": relative_residual,
            "relative_tolerance": relative_tolerance,
            "induced_voltage_V_range": [
                float(np.min(induced_voltage)),
                float(np.max(induced_voltage)),
            ],
            "interface_power_W_range": [
                float(np.min(interface_power)),
                float(np.max(interface_power)),
            ],
            "dynamic_resistance_power_W_range": [
                float(np.min(dynamic_resistance_power)),
                float(np.max(dynamic_resistance_power)),
            ],
        },
        "validity_notes": {
            "diagnostic_scope": (
                "This is an internal Lee dynamic-inductance power-accounting "
                "identity check. It does not validate first-principles Poynting "
                "coupling or experimental current/voltage agreement."
            ),
        },
    }


def circuit_coupled_energy_evidence_from_history(
    times_s: Sequence[float],
    current_A: Sequence[float],
    voltage_V: Sequence[float],
    poynting_power_W: Sequence[float],
    stored_energy_J: Sequence[float],
    dissipated_energy_J: Sequence[float] | None = None,
    *,
    verification_scope: str = "",
    relative_tolerance: float = 0.05,
) -> dict[str, object]:
    """Build KR-scoped evidence for circuit/MHD power and energy coupling."""
    times = np.asarray(times_s, dtype=float)
    current = np.asarray(current_A, dtype=float)
    voltage = np.asarray(voltage_V, dtype=float)
    poynting_power = np.asarray(poynting_power_W, dtype=float)
    stored_energy = np.asarray(stored_energy_J, dtype=float)
    if dissipated_energy_J is None:
        dissipated_energy = np.zeros_like(stored_energy)
    else:
        dissipated_energy = np.asarray(dissipated_energy_J, dtype=float)

    n = min(
        times.size,
        current.size,
        voltage.size,
        poynting_power.size,
        stored_energy.size,
        dissipated_energy.size,
    )
    times = times[:n]
    current = current[:n]
    voltage = voltage[:n]
    poynting_power = poynting_power[:n]
    stored_energy = stored_energy[:n]
    dissipated_energy = dissipated_energy[:n]
    finite = (
        np.isfinite(times)
        & np.isfinite(current)
        & np.isfinite(voltage)
        & np.isfinite(poynting_power)
        & np.isfinite(stored_energy)
        & np.isfinite(dissipated_energy)
    )
    times = times[finite]
    current = current[finite]
    voltage = voltage[finite]
    poynting_power = poynting_power[finite]
    stored_energy = stored_energy[finite]
    dissipated_energy = dissipated_energy[finite]

    has_samples = times.size >= 3 and np.all(np.diff(times) > 0.0)
    if has_samples:
        circuit_power = voltage * current
        power_scale = np.maximum(np.abs(circuit_power), np.abs(poynting_power))
        power_scale = np.maximum(power_scale, 1.0e-300)
        max_relative_power_residual = float(
            np.max(np.abs(circuit_power - poynting_power) / power_scale)
        )
        input_energy = float(np.trapezoid(poynting_power, times))
        accounted_energy = float(
            stored_energy[-1]
            - stored_energy[0]
            + dissipated_energy[-1]
            - dissipated_energy[0]
        )
        energy_scale = max(abs(input_energy), abs(accounted_energy), 1.0e-300)
        relative_energy_residual = abs(input_energy - accounted_energy) / energy_scale
    else:
        circuit_power = np.asarray([], dtype=float)
        max_relative_power_residual = float("inf")
        input_energy = 0.0
        accounted_energy = 0.0
        relative_energy_residual = float("inf")

    metrics = {
        "finite_monotonic_samples": bool(has_samples),
        "circuit_power_matches_poynting": (
            max_relative_power_residual <= relative_tolerance
        ),
        "integrated_energy_accounted": (
            relative_energy_residual <= relative_tolerance
        ),
    }
    passed = all(metrics.values())
    return {
        "passed": passed,
        "validation_tier": 3,
        "model_role": "code_verification_circuit_coupled_energy_balance",
        "verification_scope": verification_scope,
        "source": _KR_SOURCE_BASIS["auluck_circuit_element"],
        "source_lines": "1026-1031",
        "source_basis": {
            "poynting_theorem_circuit_element": (
                _KR_SOURCE_BASIS["auluck_circuit_element"]
            ),
            "mhd_voltage_circuit_feedback": (
                _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
            ),
        },
        "source_line_basis": {
            "poynting_theorem_circuit_element": "1026-1031",
            "mhd_voltage_circuit_feedback": "383-414",
        },
        "metrics": metrics,
        "missing_or_failed_metrics": [
            name for name, ok in metrics.items() if not ok
        ],
        "details": {
            "n_samples": int(times.size),
            "relative_tolerance": relative_tolerance,
            "max_relative_power_residual": max_relative_power_residual,
            "input_energy_J": input_energy,
            "accounted_energy_J": accounted_energy,
            "relative_energy_residual": relative_energy_residual,
            "circuit_power_W_range": [
                float(np.min(circuit_power)) if circuit_power.size else 0.0,
                float(np.max(circuit_power)) if circuit_power.size else 0.0,
            ],
        },
        "validity_notes": {
            "claim_scope": (
                "Supports circuit/MHD interface power and integrated energy "
                "accounting for the supplied history only. It does not validate "
                "experimental current agreement, field topology, or late-pinch "
                "MHD activity."
            ),
        },
    }


def _valid_circuit_coupled_energy_evidence(
    evidence: object,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "code_verification_circuit_coupled_energy_balance":
        return None
    if evidence.get("validation_tier") != 3:
        return None
    source_basis = evidence.get("source_basis")
    source_ok = evidence.get("source") == _KR_SOURCE_BASIS["auluck_circuit_element"]
    if isinstance(source_basis, Mapping):
        source_ok = source_ok or (
            source_basis.get("mhd_voltage_circuit_feedback")
            == _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
        )
    return evidence if source_ok else None


def _circuit_coupled_energy_evidence(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[str]]:
    for key in (
        "circuit_coupled_energy_verification",
        "circuit_coupled_energy_validation",
    ):
        evidence = _valid_circuit_coupled_energy_evidence(result.get(key))
        if evidence is not None:
            return evidence, [key]
    return None, []


def _has_any(result: Mapping[str, object], *keys: str) -> tuple[bool, list[str]]:
    found = [key for key in keys if key in result and _finite_values(result.get(key))]
    return bool(found), found


def _has_signal(result: Mapping[str, object], *keys: str) -> tuple[bool, list[str]]:
    found = []
    for key in keys:
        if key not in result:
            continue
        value = result.get(key)
        if _finite_values(value) or (isinstance(value, Mapping) and bool(value)):
            found.append(key)
    return bool(found), found


def _has_metadata(result: Mapping[str, object], *keys: str) -> tuple[bool, list[str]]:
    found = [key for key in keys if key in result and result.get(key) is not None]
    return bool(found), found


def _derived_dldt_available(result: Mapping[str, object]) -> tuple[bool, str | None]:
    time_key, times = _series_info(result, "t_s", "time_s", "t_us", "time_us")
    inductance_key, inductance = _series_info(
        result,
        "L_p_nH",
        "Lp_nH",
        "L_plasma",
        "Lp_mhd_nH",
        "field_inductance",
    )
    if time_key is None or inductance_key is None:
        return False, None
    if len(times) < 2 or len(inductance) < 2:
        return False, None
    if times[0] == times[1] or inductance[0] == inductance[1]:
        return False, None
    return True, f"{inductance_key}_from_{time_key}"


def field_coupling_evidence_from_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Build a conservative circuit/field coupling evidence record for a run."""
    evidence: dict[str, dict[str, object]] = {}
    validated_component_scopes: dict[str, str] = {}
    circuit_energy_evidence, circuit_energy_evidence_keys = (
        _circuit_coupled_energy_evidence(result)
    )
    circuit_energy_validated = circuit_energy_evidence is not None
    circuit_energy_scope = (
        str(circuit_energy_evidence.get("validation_scope", ""))
        if circuit_energy_evidence is not None
        else ""
    )

    inductance_key, _ = _series_info(
        result,
        "L_p_nH",
        "Lp_nH",
        "L_plasma",
        "Lp_mhd_nH",
        "field_inductance",
    )
    evidence["plasma_inductance_series"] = _record(
        "plasma_inductance_series",
        status="implemented_not_validated" if inductance_key else "absent",
        present=inductance_key is not None,
        evidence_keys=[inductance_key] if inductance_key else [],
        notes=(
            "A plasma-inductance-like series is exported, but no KR-backed "
            "field-coupling validation is attached."
            if inductance_key
            else "No time-resolved plasma inductance or equivalent coupling signal is exported."
        ),
    )

    field_like, field_keys = _has_any(
        result,
        "Lp_mhd_nH",
        "field_inductance",
        "field_derived_inductance",
        "magnetic_energy_inductance",
    )
    reduced_like = inductance_key in {"L_p_nH", "Lp_nH", "L_plasma"}
    evidence["field_derived_inductance"] = _record(
        "field_derived_inductance",
        status=(
            "implemented_not_validated"
            if field_like else
            "reduced_or_unknown_source"
            if reduced_like else
            "absent"
        ),
        present=field_like,
        evidence_keys=field_keys or ([inductance_key] if inductance_key else []),
        notes=(
            "A field-derived or MHD inductance series is exported, but it has "
            "not been validated against a KR field-coupling target."
            if field_like
            else "Only a reduced or source-ambiguous inductance signal is present."
            if reduced_like
            else "No field-derived inductance evidence is exported."
        ),
    )

    dldt_like, dldt_keys = _has_any(
        result,
        "back_emf",
        "back_emf_V",
        "dL_dt",
        "dLdt",
        "dLp_dt",
    )
    derived_dldt, derived_key = _derived_dldt_available(result)
    evidence["dLdt_or_back_emf"] = _record(
        "dLdt_or_back_emf",
        status=(
            "implemented_not_validated"
            if dldt_like else
            "candidate_from_inductance_derivative"
            if derived_dldt else
            "absent"
        ),
        present=dldt_like or derived_dldt,
        evidence_keys=dldt_keys or ([derived_key] if derived_key else []),
        notes=(
            "Back-EMF or dL/dt is exported, but no KR validation evidence is attached."
            if dldt_like
            else "A finite inductance derivative can be inferred from exported "
            "time and inductance arrays, but it is not an explicit validated "
            "coupling signal."
            if derived_dldt
            else "No dL/dt or back-EMF evidence is exported or inferable."
        ),
    )

    poynting_like, poynting_keys = _has_signal(
        result,
        "poynting_power",
        "poynting_flux",
        "poynting_voltage",
        "V_poynting",
        "poynting_balance",
    )
    if circuit_energy_validated:
        poynting_like = True
        poynting_keys = sorted(set(poynting_keys + circuit_energy_evidence_keys))
        validated_component_scopes["poynting_power_balance"] = circuit_energy_scope
    evidence["poynting_power_balance"] = _record(
        "poynting_power_balance",
        status=(
            "supported"
            if circuit_energy_validated else
            "diagnostic_not_validated"
            if poynting_like else
            "absent"
        ),
        present=poynting_like,
        validated=circuit_energy_validated,
        evidence_keys=poynting_keys,
        notes=(
            "KR-scoped circuit/MHD power evidence is attached for the supplied "
            "history."
            if circuit_energy_validated
            else
            "Poynting-related diagnostics are exported, but no KR-backed "
            "power-balance validation is attached."
            if poynting_like
            else "No Poynting-flux or equivalent interface power balance is exported."
        ),
    )

    energy_like, energy_keys = _has_signal(
        result,
        "E_cap",
        "E_ind",
        "E_res",
        "E_cap_kJ",
        "E_ind_kJ",
        "E_res_kJ",
        "energy_balance",
        "circuit_energy_balance",
        "dynamic_inductance_power_balance",
    )
    if circuit_energy_validated:
        energy_like = True
        energy_keys = sorted(set(energy_keys + circuit_energy_evidence_keys))
        validated_component_scopes["circuit_energy_balance"] = circuit_energy_scope
    evidence["circuit_energy_balance"] = _record(
        "circuit_energy_balance",
        status=(
            "supported"
            if circuit_energy_validated else
            "diagnostic_not_validated"
            if energy_like else
            "absent"
        ),
        present=energy_like,
        validated=circuit_energy_validated,
        evidence_keys=energy_keys,
        notes=(
            "KR-scoped circuit/MHD integrated energy evidence is attached for "
            "the supplied history."
            if circuit_energy_validated
            else
            "Circuit energy channels are exported, but the changing-Lp power "
            "partition has not been validated against a KR target."
            if energy_like
            else "No circuit energy balance channels are exported."
        ),
    )

    transition_like, transition_keys = _has_metadata(
        result,
        "coupling_transition",
        "mhd_handoff",
        "handoff_mode",
        "coupling_alpha",
        "Lp_snowplow_nH",
    )
    if "Lp_snowplow_nH" in result and "Lp_mhd_nH" in result:
        transition_like = True
        transition_keys = sorted(set(transition_keys + ["Lp_snowplow_nH", "Lp_mhd_nH"]))
    evidence["handoff_transition_metadata"] = _record(
        "handoff_transition_metadata",
        status="diagnostic_not_validated" if transition_like else "absent",
        present=transition_like,
        evidence_keys=transition_keys,
        notes=(
            "Handoff or blended-coupling metadata is present, but the transition "
            "timing is not validated against KR evidence."
            if transition_like
            else "No explicit snowplow-to-field-coupling transition metadata is exported."
        ),
    )

    kr_evidence = result.get("field_coupling_experimental_validation")
    kr_validated = _kr_sourced_evidence_passed(kr_evidence)
    if kr_validated and isinstance(kr_evidence, Mapping):
        validated_component_scopes["kr_experimental_comparison"] = str(
            kr_evidence.get("validation_scope", "")
        )
    evidence["kr_experimental_comparison"] = _record(
        "kr_experimental_comparison",
        status="supported" if kr_validated else "validation_absent",
        present=isinstance(kr_evidence, Mapping),
        validated=kr_validated,
        evidence_keys=["field_coupling_experimental_validation"] if kr_evidence else [],
        notes=(
            "KR-backed field-coupling comparison evidence is attached."
            if kr_validated
            else "No passing KR-backed field-coupling comparison evidence is attached."
        ),
    )

    for component, (component_evidence, evidence_key) in _validated_components(
        result
    ).items():
        if component not in evidence:
            continue
        validated_component_scopes[component] = str(
            component_evidence.get("validation_scope", "")
        )
        evidence[component] = _record(
            component,
            status="supported",
            present=True,
            validated=True,
            evidence_keys=[evidence_key],
            notes=(
                "KR-backed field-coupling component evidence is attached for "
                "the stated validation scope."
            ),
        )

    missing = [
        name for name, item in evidence.items()
        if item.get("validated") is not True
    ]
    scope_values = {
        scope for scope in validated_component_scopes.values()
        if scope
    }
    same_scope_passed = (
        not missing
        and all(validated_component_scopes.get(name) for name in evidence)
        and len(scope_values) == 1
    )
    if not missing and not same_scope_passed:
        missing.append("same_scope_field_coupling_packet")
    passed = not missing
    return {
        "passed": passed,
        "validation_tier": "field_coupling",
        "model_role": "field_coupling_audit",
        "source": _KR_SOURCE_BASIS["auluck_circuit_element"],
        "source_basis": _KR_SOURCE_BASIS,
        "required_evidence": evidence,
        "component_validation_scopes": validated_component_scopes,
        "same_scope_passed": same_scope_passed,
        "missing_or_unvalidated_evidence": missing,
        "validity_notes": {
            "claim_scope": (
                "MHD-mode current prediction is not field-coupling validated "
                "unless inductance, dL/dt or back-EMF, interface power, energy "
                "partition, transition timing, and KR comparison evidence are "
                "all validated for one claimed scope."
            ),
            "audit_role": (
                "This audit separates exported coupling signals from validated "
                "first-principles circuit/field coupling evidence."
            ),
        },
    }
