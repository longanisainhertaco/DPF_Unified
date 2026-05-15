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

from dpf.constants import mu_0


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
    "coupling_interval_authority": {
        "source_key": "auluck_circuit_element",
        "source_lines": "35-38, 957-991",
        "requirement": (
            "Circuit-field coupling evidence must distinguish snowplow-loaded, "
            "blended, field-derived candidate, and validated field-coupled "
            "intervals before supporting MHD current-prediction claims."
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

_INTERVAL_AUTHORITY_LABELS = (
    "snowplow_loaded",
    "blended",
    "field_derived_candidate",
    "validated_field_coupled",
)


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


def _trapezoid_integral(values: np.ndarray, times: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(values, times))


def _axisym_scalar(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"expected scalar field shape (nr, nz) or (nr, 1, nz), got {arr.shape}")


def _axisym_vector(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 4 and arr.shape[0] == 3 and arr.shape[2] == 1:
        return arr[:, :, 0, :]
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr
    raise ValueError(
        f"expected vector field shape (3, nr, nz) or (3, nr, 1, nz), got {arr.shape}"
    )


def _safe_gradient(values: np.ndarray, spacing: float, *, axis: int) -> np.ndarray:
    if values.shape[axis] < 2:
        return np.zeros_like(values)
    return np.gradient(values, spacing, axis=axis)


def _eta_array(eta: object | None, shape: tuple[int, int]) -> np.ndarray:
    if eta is None:
        return np.zeros(shape, dtype=float)
    arr = np.asarray(eta, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    return np.broadcast_to(arr, shape).astype(float, copy=False)


def field_power_diagnostics_from_cylindrical_state(
    state: Mapping[str, object],
    *,
    dr: float,
    dz: float,
    current_A: float,
    r_cell_m: Sequence[float] | None = None,
    r_min_m: float = 0.0,
    eta_ohm_m: object | None = None,
    previous_inductance_H: float | None = None,
    dt_s: float | None = None,
    current_floor_A: float = 5.0e4,
) -> dict[str, object]:
    """Compute field-derived circuit diagnostics from an axisymmetric MHD state.

    This is an engineering diagnostic, not accepted validation evidence. It
    uses resolved fields to compute magnetic energy, the corresponding
    energy-derived inductance, and the generalized-Ohm-law volume integral
    ``integral(J dot E) dV`` on a cylindrical grid.
    """
    B = np.nan_to_num(_axisym_vector(state["B"]))
    velocity = np.nan_to_num(
        _axisym_vector(state.get("velocity", np.zeros_like(B)))
    )
    nr, nz = B.shape[1], B.shape[2]
    if r_cell_m is None:
        r = r_min_m + (np.arange(nr, dtype=float) + 0.5) * dr
    else:
        r = np.asarray(r_cell_m, dtype=float)
        if r.size != nr:
            raise ValueError(f"r_cell_m length {r.size} does not match nr={nr}")
    r_safe = np.maximum(r, 1.0e-12)
    volume = 2.0 * np.pi * r_safe[:, None] * dr * dz

    Br, Bt, Bz = B[0], B[1], B[2]
    vr, vt, vz = velocity[0], velocity[1], velocity[2]
    B_sq = Br * Br + Bt * Bt + Bz * Bz
    magnetic_energy_J = float(np.sum(0.5 * B_sq / mu_0 * volume))

    current_value = float(current_A)
    current_sq = current_value * current_value
    threshold = max(float(current_floor_A), 0.0)
    has_terminal_current = (
        abs(current_value) >= threshold
        if threshold > 0.0
        else current_value != 0.0
    )
    if has_terminal_current:
        field_inductance_H = 2.0 * magnetic_energy_J / current_sq
    else:
        field_inductance_H = 0.0 if magnetic_energy_J == 0.0 else float("inf")
    if (
        previous_inductance_H is not None
        and dt_s is not None
        and dt_s > 0.0
        and np.isfinite(previous_inductance_H)
    ):
        dL_field_dt_H_s = (field_inductance_H - float(previous_inductance_H)) / dt_s
    else:
        dL_field_dt_H_s = 0.0

    dBt_dz = _safe_gradient(Bt, dz, axis=1)
    rBt = r_safe[:, None] * Bt
    d_rBt_dr = _safe_gradient(rBt, dr, axis=0)
    dBr_dz = _safe_gradient(Br, dz, axis=1)
    dBz_dr = _safe_gradient(Bz, dr, axis=0)

    inv_mu0 = 1.0 / mu_0
    Jr = -dBt_dz * inv_mu0
    Jt = (dBr_dz - dBz_dr) * inv_mu0
    Jz = (d_rBt_dr / r_safe[:, None]) * inv_mu0
    eta_arr = _eta_array(eta_ohm_m, (nr, nz))

    # Generalized Ohm law in cylindrical components: E = -v x B + eta J.
    Er = vz * Bt - vt * Bz + eta_arr * Jr
    Et = vr * Bz - vz * Br + eta_arr * Jt
    Ez = vt * Br - vr * Bt + eta_arr * Jz
    j_dot_e = Jr * Er + Jt * Et + Jz * Ez
    j_dot_e_power_W = float(np.sum(j_dot_e * volume))
    J_sq = Jr * Jr + Jt * Jt + Jz * Jz
    joule_power_W = float(np.sum(eta_arr * J_sq * volume))

    if has_terminal_current:
        load_voltage_V = j_dot_e_power_W / current_value
        source_orientation_voltage_V = -load_voltage_V
    elif j_dot_e_power_W == 0.0:
        load_voltage_V = 0.0
        source_orientation_voltage_V = 0.0
    else:
        load_voltage_V = float("nan")
        source_orientation_voltage_V = float("nan")
    field_interface_power_W = current_value * load_voltage_V

    return {
        "classification": "engineering_field_coupling_diagnostic",
        "validation_status": "not_validation_evidence",
        "magnetic_energy_J": magnetic_energy_J,
        "field_derived_inductance_H": field_inductance_H,
        "dL_field_dt_H_s": dL_field_dt_H_s,
        "j_dot_e_power_W": j_dot_e_power_W,
        "poynting_power_W": field_interface_power_W,
        "joule_power_W": joule_power_W,
        "field_terminal_voltage_V": load_voltage_V,
        "poynting_voltage_source_orientation_V": source_orientation_voltage_V,
        "back_emf_V": load_voltage_V,
        "current_A": float(current_A),
        "current_floor_A": current_floor_A,
        "n_cells": int(nr * nz),
        "sign_convention": (
            "Positive j_dot_e_power_W is power absorbed by the resolved plasma. "
            "field_terminal_voltage_V = integral(J dot E)dV / current_A is "
            "passed to RLCSolver as opposing back_emf; "
            "poynting_voltage_source_orientation_V stores the opposite terminal "
            "orientation."
        ),
    }


def implicit_midpoint_power_port_back_emf(
    *,
    current_A: float,
    capacitor_voltage_V: float,
    L_total_H: float,
    resistance_ohm: float,
    capacitance_F: float,
    dL_dt_H_s: float,
    dt_s: float,
    power_W: float,
    crowbar_fired: bool = False,
) -> dict[str, object]:
    """Convert field load power to an RLC back-EMF without a current floor.

    The app-level field-coupled candidate computes the resolved-field load as
    a power.  The circuit solver accepts a terminal voltage.  This helper
    enforces the same implicit-midpoint relation used by ``RLCSolver.step`` and
    chooses the root continuous with the zero-load circuit update:

    ``power_W = I_mid * back_emf_V``.

    It is a numerical power-port closure, not validation evidence.
    """
    I_n = float(current_A)
    V_n = float(capacitor_voltage_V)
    L_total = float(L_total_H)
    R_star = float(resistance_ohm) + float(dL_dt_H_s)
    C = float(capacitance_F)
    dt = float(dt_s)
    power = float(power_W)
    if not all(
        np.isfinite(value)
        for value in (I_n, V_n, L_total, R_star, C, dt, power)
    ):
        return {
            "passed": False,
            "reason": "nonfinite_input",
            "back_emf_V": 0.0,
            "power_W": power,
        }
    if L_total <= 0.0 or dt <= 0.0 or (not crowbar_fired and C <= 0.0):
        return {
            "passed": False,
            "reason": "invalid_circuit_domain",
            "back_emf_V": 0.0,
            "power_W": power,
            "L_total_H": L_total,
            "dt_s": dt,
            "capacitance_F": C,
        }

    alpha = dt / (2.0 * L_total)
    beta = 0.0 if crowbar_fired else alpha * dt / (2.0 * C)
    denom = 1.0 + alpha * R_star + beta
    A = I_n * (1.0 - alpha * R_star - beta)
    if not crowbar_fired:
        A += 2.0 * alpha * V_n
    if not np.isfinite(denom) or denom == 0.0:
        return {
            "passed": False,
            "reason": "singular_midpoint_denominator",
            "back_emf_V": 0.0,
            "power_W": power,
            "denominator": denom,
        }

    I_no_load_new = A / denom
    if power == 0.0:
        I_mid = 0.5 * (I_n + I_no_load_new)
        return {
            "passed": True,
            "method": "implicit_midpoint_power_port",
            "back_emf_V": 0.0,
            "current_mid_A": I_mid,
            "current_new_A": I_no_load_new,
            "current_new_no_load_A": I_no_load_new,
            "power_W": power,
            "power_residual_W": 0.0,
        }

    b = denom * I_n - A
    c = 4.0 * alpha * power - A * I_n
    discriminant = b * b - 4.0 * denom * c
    if not np.isfinite(discriminant) or discriminant < 0.0:
        return {
            "passed": False,
            "reason": "no_real_midpoint_power_port_root",
            "back_emf_V": 0.0,
            "power_W": power,
            "discriminant": discriminant,
            "current_new_no_load_A": I_no_load_new,
        }

    sqrt_discriminant = float(np.sqrt(discriminant))
    roots = (
        (-b + sqrt_discriminant) / (2.0 * denom),
        (-b - sqrt_discriminant) / (2.0 * denom),
    )
    I_new = min(roots, key=lambda root: abs(root - I_no_load_new))
    I_mid = 0.5 * (I_n + I_new)
    if not np.isfinite(I_mid) or I_mid == 0.0:
        return {
            "passed": False,
            "reason": "zero_midpoint_current_for_nonzero_power",
            "back_emf_V": 0.0,
            "power_W": power,
            "current_mid_A": I_mid,
            "current_new_A": I_new,
            "current_new_no_load_A": I_no_load_new,
        }
    back_emf = power / I_mid
    return {
        "passed": bool(np.isfinite(back_emf)),
        "method": "implicit_midpoint_power_port",
        "back_emf_V": float(back_emf) if np.isfinite(back_emf) else 0.0,
        "current_mid_A": float(I_mid),
        "current_new_A": float(I_new),
        "current_new_no_load_A": float(I_no_load_new),
        "power_W": power,
        "power_residual_W": float(back_emf * I_mid - power),
        "discriminant": float(discriminant),
    }


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
        input_energy = _trapezoid_integral(poynting_power, times)
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


def _normalize_interval_authority(value: object) -> str | None:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if "blend" in text or "handoff" in text or "transition" in text:
        return "blended"
    if "snowplow" in text or "lee" in text or "density_weight" in text:
        return "snowplow_loaded"
    if "validated" in text and "field" in text:
        return "validated_field_coupled"
    if "field" in text or "mhd" in text or "poynting" in text:
        return "field_derived_candidate"
    return None


def _add_interval_authority_labels(
    value: object,
    labels: set[str],
    keys: set[str],
    *,
    key_hint: str,
) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            label = _normalize_interval_authority(raw_key)
            if label is None and isinstance(raw_value, Mapping):
                label = _normalize_interval_authority(
                    raw_value.get("authority")
                    or raw_value.get("label")
                    or raw_value.get("status")
                    or raw_value.get("mode")
                )
            if label is None:
                label = _normalize_interval_authority(raw_value)
            if label is not None:
                labels.add(label)
                keys.add(key_hint)
        return

    if _is_sequence(value):
        for item in value:
            if isinstance(item, Mapping):
                label = _normalize_interval_authority(
                    item.get("authority")
                    or item.get("label")
                    or item.get("status")
                    or item.get("mode")
                )
            else:
                label = _normalize_interval_authority(item)
            if label is not None:
                labels.add(label)
                keys.add(key_hint)
        return

    label = _normalize_interval_authority(value)
    if label is not None:
        labels.add(label)
        keys.add(key_hint)


def _coupling_interval_authority(
    result: Mapping[str, object],
) -> tuple[set[str], list[str]]:
    labels: set[str] = set()
    keys: set[str] = set()
    for key in (
        "field_coupling_intervals",
        "coupling_intervals",
        "coupling_interval_authority",
        "interval_authority",
    ):
        if key in result:
            _add_interval_authority_labels(result.get(key), labels, keys, key_hint=key)

    field_coupled_declared = result.get("field_coupled_candidate") is True or any(
        key in result
        for key in (
            "field_inductance",
            "field_derived_inductance",
            "magnetic_energy_inductance",
        )
    )
    if "Lp_snowplow_nH" in result or (
        not field_coupled_declared
        and any(key in result for key in ("L_p_nH", "Lp_nH", "L_plasma"))
    ):
        labels.add("snowplow_loaded")
    if "coupling_alpha" in result or (
        "Lp_snowplow_nH" in result and "Lp_mhd_nH" in result
    ):
        labels.add("blended")
    if any(
        key in result
        for key in (
            "Lp_mhd_nH",
            "field_inductance",
            "field_derived_inductance",
            "magnetic_energy_inductance",
            "poynting_power",
            "poynting_power_W",
            "poynting_balance",
        )
    ):
        labels.add("field_derived_candidate")

    if labels:
        keys.update(
            key for key in (
                "Lp_snowplow_nH",
                "L_p_nH",
                "Lp_nH",
                "L_plasma",
                "Lp_mhd_nH",
                "coupling_alpha",
                "field_inductance",
                "field_derived_inductance",
                "magnetic_energy_inductance",
                "poynting_power",
                "poynting_power_W",
                "poynting_balance",
            )
            if key in result
        )
    return labels, sorted(keys)


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
    density_weighted_candidate = "Lp_mhd_nH" in field_keys and not any(
        key in field_keys
        for key in ("field_inductance", "field_derived_inductance", "magnetic_energy_inductance")
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
            "An MHD/density-weighted inductance series is exported. It is "
            "candidate coupling evidence only, not fully field-derived or "
            "validated field-coupled authority."
            if density_weighted_candidate
            else "A field-derived inductance series is exported, but it has "
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
        "poynting_power_W",
        "poynting_flux",
        "poynting_voltage",
        "V_poynting",
        "field_terminal_voltage_V",
        "j_dot_e_power_W",
        "field_interface_power_W",
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

    interval_labels, interval_keys = _coupling_interval_authority(result)
    missing_interval_labels = [
        label for label in _INTERVAL_AUTHORITY_LABELS if label not in interval_labels
    ]
    evidence["coupling_interval_authority"] = _record(
        "coupling_interval_authority",
        status=(
            "staged_not_validated"
            if interval_labels and not missing_interval_labels else
            "incomplete_interval_authority"
            if interval_labels else
            "absent"
        ),
        present=bool(interval_labels),
        evidence_keys=interval_keys,
        notes=(
            "Interval labels are exported, but they are not KR-validated "
            "field-coupling authority; missing labels: "
            + ", ".join(missing_interval_labels)
            if missing_interval_labels
            else "All staged interval labels are visible, but interval "
            "authority still needs KR-backed component validation before it "
            "can support predictive field-coupled claims."
            if interval_labels
            else "No staged coupling interval authority labels are exported."
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
        "coupling_interval_authority": {
            "labels": sorted(interval_labels),
            "missing_labels": missing_interval_labels,
        },
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
