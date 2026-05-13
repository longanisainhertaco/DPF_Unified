"""Automated simulation quality assessment.

Evaluates a simulation result against physics expectations and
assigns a quality grade (A-F) with specific feedback.

Checks:
    1. Current waveform shape (rise, peak, dip)
    2. Pinch compression ratio
    3. Bennett equilibrium consistency
    4. Energy conservation
    5. Grid resolution adequacy
    6. Neutron yield plausibility
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re

_REQUIRED_CIRCUIT_METRICS = {"peak_current", "peak_time", "waveform_shape"}
_REQUIRED_SNOWPLOW_PHASES = {"axial", "radial", "pinch"}
_REQUIRED_MHD_VERIFICATION_TESTS = {"sod", "brio_wu"}
_REQUIRED_SPATIAL_QUANTITIES = {"density", "magnetic_field", "temperature"}
_REQUIRED_NEUTRON_MECHANISMS = {"thermonuclear", "beam_target"}
_REQUIRED_NEUTRON_YIELD_FEATURES = {"yield"}
_REQUIRED_NEUTRON_SPECTRAL_FEATURES = {"spectrum"}
_REQUIRED_NEUTRON_ANISOTROPY_FEATURES = {"anisotropy"}
_REQUIRED_NEUTRON_DETECTOR_FEATURES = {
    "activation_response",
    "detector_response",
}
_REQUIRED_NEUTRON_UNCERTAINTY_FEATURES = {"uncertainty"}
_RADIAL_PHASE_LABELS = {
    "radial",
    "mhd_radial",
    "radial_implosion",
    "reflected",
    "pinch",
    "post_pinch",
}
_PINCH_PHASE_LABELS = {"reflected", "pinch", "post_pinch", "stagnation"}
_SOURCE_AUTHORITY_VALIDATION_KEYS = (
    "circuit_validation",
    "snowplow_validation",
    "mhd_verification",
    "mhd_numerical_fidelity",
    "field_coupling_validation",
    "physics_fidelity_evidence",
    "uncertainty_validation",
    "neutron_yield_validation",
    "neutron_mechanism_timing_validation",
    "neutron_spectrum_validation",
    "neutron_anisotropy_validation",
    "neutron_detector_response_validation",
)
_LINE_RANGE_RE = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))?\s*$")


def _evidence_passed(evidence: object) -> bool:
    """Return True only for explicit pass markers."""
    if evidence is True:
        return True
    if isinstance(evidence, dict):
        return evidence.get("passed") is True
    return False


def _normalize_evidence_label(value: object) -> str:
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "b": "magnetic_field",
        "b_field": "magnetic_field",
        "magnetic": "magnetic_field",
        "magneticfield": "magnetic_field",
        "rho": "density",
        "mass_density": "density",
        "ne": "density",
        "electron_density": "density",
        "temp": "temperature",
        "te": "temperature",
        "ti": "temperature",
        "electron_temperature": "temperature",
        "ion_temperature": "temperature",
    }
    return aliases.get(label, label)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _has_kr_source_authority(evidence: object, validation_tier: int) -> bool:
    if not isinstance(evidence, dict):
        return False
    return (
        str(evidence.get("source", "")).startswith("KnowledgeReference/")
        and _safe_int(evidence.get("validation_tier", 0)) == validation_tier
        and str(evidence.get("model_role", "")).startswith("simulation_to_kr_")
    )


def _has_circuit_source_authority(evidence: object) -> bool:
    if not isinstance(evidence, dict):
        return False
    details = evidence.get("details", {})
    authority = evidence.get("source_authority")
    if authority is None and isinstance(details, dict):
        authority = details.get("source_authority")
    if not isinstance(authority, dict):
        return False
    if authority.get("passed") is True or authority.get("validation_ready") is True:
        return True
    return (
        authority.get("kr_status") == "verified"
        and authority.get("reliability") == "measured"
        and authority.get("waveform_provenance") == "measured"
        and authority.get("waveform_kr_status") == "verified"
    )


def _has_snowplow_source_authority(evidence: object) -> bool:
    if not isinstance(evidence, dict):
        return False
    details = evidence.get("details", {})
    authority = evidence.get("source_authority")
    if authority is None and isinstance(details, dict):
        authority = details.get("source_authority")
    if isinstance(authority, dict) and authority.get("passed") is True:
        return True
    return (
        str(evidence.get("reference_source", "")).startswith("KnowledgeReference/")
        and evidence.get("reference_kr_status") == "verified"
    )


def _has_spatial_source_authority(evidence: object) -> bool:
    if not isinstance(evidence, dict):
        return False
    if _has_kr_source_authority(evidence, 4):
        return True
    details = evidence.get("details", {})
    if not isinstance(details, dict):
        return False
    components = details.get("component_evidence", [])
    if not isinstance(components, list) or not components:
        return False
    return all(
        isinstance(component, dict)
        and component.get("source_authority_passed") is True
        for component in components
    )


def _evidence_items(evidence: object, *field_names: str) -> set[str]:
    """Extract normalized evidence item labels from a metadata dictionary."""
    if not isinstance(evidence, dict):
        return set()
    items: set[str] = set()
    for field_name in field_names:
        value = evidence.get(field_name)
        if isinstance(value, dict):
            items.update(_normalize_evidence_label(k) for k, v in value.items() if v)
        elif isinstance(value, (list, tuple, set)):
            items.update(_normalize_evidence_label(item) for item in value)
        elif isinstance(value, str):
            items.add(_normalize_evidence_label(value))
    return items


def _covers_required(evidence: object, required: set[str], *field_names: str) -> bool:
    return required.issubset(_evidence_items(evidence, *field_names))


def circuit_validation_evidence_from_result(
    validation_result: object,
    *,
    waveform_shape_passed: bool | None = None,
    waveform_nrmse: float | None = None,
    waveform_tolerance: float | None = None,
    source: str = "ValidationSuite circuit metrics plus waveform comparison",
) -> dict[str, object]:
    """Build strict circuit-validation evidence from validation outputs."""
    metric_passes: dict[str, bool] = {}
    details: dict[str, dict[str, object]] = {}
    for metric in getattr(validation_result, "metrics", []):
        name = str(getattr(metric, "name", ""))
        passed = bool(getattr(metric, "passed", False))
        canonical = {
            "peak_current": "peak_current",
            "peak_current_time": "peak_time",
            "peak_time": "peak_time",
            "waveform_shape": "waveform_shape",
            "waveform_nrmse": "waveform_shape",
        }.get(name)
        if canonical:
            metric_passes[canonical] = passed
            details[canonical] = {
                "source_metric": name,
                "passed": passed,
                "relative_error": getattr(metric, "relative_error", None),
                "tolerance": getattr(metric, "tolerance", None),
            }

    if waveform_shape_passed is None and waveform_nrmse is not None and waveform_tolerance is not None:
        waveform_shape_passed = waveform_nrmse <= waveform_tolerance

    if waveform_shape_passed is not None:
        metric_passes["waveform_shape"] = bool(waveform_shape_passed)
        details["waveform_shape"] = {
            "source_metric": "waveform_nrmse",
            "passed": bool(waveform_shape_passed),
            "nrmse": waveform_nrmse,
            "tolerance": waveform_tolerance,
        }

    passed = _REQUIRED_CIRCUIT_METRICS.issubset(
        {name for name, ok in metric_passes.items() if ok}
    )
    return {
        "passed": passed,
        "metrics": metric_passes,
        "device": getattr(validation_result, "device", ""),
        "overall_score": getattr(validation_result, "overall_score", 0.0),
        "source": source,
        "details": details,
    }


def circuit_validation_evidence_from_waveform(
    times_s: object,
    current_A: object,
    device_name: str,
    *,
    peak_current_tolerance: float = 0.15,
    timing_tolerance: float = 0.10,
    waveform_tolerance: float = 0.20,
    truncate_at_dip: bool = False,
    require_kr_verified: bool = True,
    source: str = "Experimental current waveform comparison",
) -> dict[str, object]:
    """Build strict circuit evidence from a simulated current waveform."""
    from dpf.validation.experimental import DEVICES, validate_current_waveform

    metrics = validate_current_waveform(
        times_s,
        current_A,
        device_name,
        truncate_at_dip=truncate_at_dip,
    )
    device = DEVICES.get(device_name)
    source_authority = {
        "kr_status": getattr(device, "kr_status", "") if device is not None else "",
        "reliability": getattr(device, "reliability", "") if device is not None else "",
        "waveform_provenance": (
            getattr(device, "waveform_provenance", "") if device is not None else ""
        ),
        "waveform_kr_status": (
            getattr(device, "waveform_kr_status", "") if device is not None else ""
        ),
        "require_kr_verified": require_kr_verified,
    }
    source_authority_passed = (
        not require_kr_verified
        or (
            source_authority["kr_status"] == "verified"
            and source_authority["reliability"] == "measured"
            and source_authority["waveform_provenance"] == "measured"
            and source_authority["waveform_kr_status"] == "verified"
        )
    )
    waveform_nrmse = metrics.get("waveform_nrmse")
    waveform_shape_passed = (
        metrics.get("waveform_available") is True
        and waveform_nrmse is not None
        and float(waveform_nrmse) <= waveform_tolerance
    )
    metric_passes = {
        "peak_current": float(metrics["peak_current_error"]) <= peak_current_tolerance,
        "peak_time": (
            bool(metrics["timing_ok"])
            and float(metrics["timing_error"]) <= timing_tolerance
        ),
        "waveform_shape": waveform_shape_passed,
    }
    passed = all(metric_passes.values()) and source_authority_passed
    return {
        "passed": passed,
        "metrics": metric_passes,
        "device": device_name,
        "source": source,
        "details": {
            "peak_current_error": metrics["peak_current_error"],
            "timing_error": metrics["timing_error"],
            "waveform_nrmse": waveform_nrmse,
            "peak_current_tolerance": peak_current_tolerance,
            "timing_tolerance": timing_tolerance,
            "waveform_tolerance": waveform_tolerance,
            "measurement_notes": metrics.get("measurement_notes", ""),
            "uncertainty": metrics.get("uncertainty", {}),
            "source_authority": {
                **source_authority,
                "passed": source_authority_passed,
            },
        },
        "validity_notes": {
            "tier_scope": (
                "Circuit evidence validates the current waveform only. It does "
                "not validate snowplow, MHD spatial structure, or neutron timing."
            ),
            "source_authority": (
                "With require_kr_verified=True, only measured waveforms from "
                "KR-verified device records can support tier 1."
            ),
        },
    }


def mhd_verification_evidence_from_tests(
    test_results: dict[str, bool],
    *,
    source: str = "MHD analytic verification test results",
) -> dict[str, object]:
    """Build strict MHD verification evidence from named analytic tests."""
    normalized = {
        _normalize_evidence_label(name): bool(passed)
        for name, passed in test_results.items()
    }
    passed = _REQUIRED_MHD_VERIFICATION_TESTS.issubset(
        {name for name, ok in normalized.items() if ok}
    )
    return {
        "passed": passed,
        "analytic_tests": normalized,
        "required_tests": sorted(_REQUIRED_MHD_VERIFICATION_TESTS),
        "validation_tier": 3,
        "model_role": "code_verification_analytic_tests",
        "source": source,
    }


def snowplow_validation_evidence_from_phase_errors(
    phase_relative_errors: dict[str, float],
    *,
    tolerance: float = 0.20,
    source: str = "Snowplow phase/timing validation metrics",
) -> dict[str, object]:
    """Build strict snowplow validation evidence from phase timing errors."""
    normalized_errors = {
        _normalize_evidence_label(phase): float(error)
        for phase, error in phase_relative_errors.items()
    }
    phase_passes = {
        phase: (phase in normalized_errors and normalized_errors[phase] <= tolerance)
        for phase in _REQUIRED_SNOWPLOW_PHASES
    }
    passed = all(phase_passes.values())
    return {
        "passed": passed,
        "phases": phase_passes,
        "phase_relative_errors": normalized_errors,
        "tolerance": tolerance,
        "source": source,
    }


def mhd_verification_evidence_from_shock_tube_results(
    sod_result: object,
    brio_wu_result: object,
    *,
    sod_l1_tolerances: dict[str, float] | None = None,
    required_brio_wu_checks: set[str] | None = None,
    source: str = "MHD Sod and Brio-Wu shock-tube verification results",
) -> dict[str, object]:
    """Build tier-3 MHD evidence from shock-tube verification outputs."""
    tolerances = sod_l1_tolerances or {"rho": 0.08, "u": 0.10, "p": 0.08}
    brio_required = required_brio_wu_checks or {
        "no_nan",
        "rho_positive",
        "p_positive",
        "Bx_preserved",
        "has_wave_structure",
        "By_sign_change",
    }

    sod_errors = dict(getattr(sod_result, "errors", {}) or {})
    sod_checks = dict(getattr(sod_result, "checks", {}) or {})
    sod_error_passes = {
        field: (
            field in sod_errors
            and float(sod_errors[field]) <= float(tolerance)
        )
        for field, tolerance in tolerances.items()
    }
    sod_sanity_checks = {
        check: bool(sod_checks.get(check, False))
        for check in ("no_nan", "rho_positive", "p_positive")
    }
    sod_passed = all(sod_error_passes.values()) and all(sod_sanity_checks.values())

    brio_checks = dict(getattr(brio_wu_result, "checks", {}) or {})
    brio_passes = {
        check: bool(brio_checks.get(check, False))
        for check in sorted(brio_required)
    }
    brio_passed = all(brio_passes.values())

    evidence = mhd_verification_evidence_from_tests(
        {"sod": sod_passed, "brio_wu": brio_passed},
        source=source,
    )
    evidence["details"] = {
        "sod": {
            "l1_errors": sod_errors,
            "l1_tolerances": tolerances,
            "error_passes": sod_error_passes,
            "sanity_checks": sod_sanity_checks,
        },
        "brio_wu": {
            "checks": brio_checks,
            "required_checks": sorted(brio_required),
            "check_passes": brio_passes,
        },
    }
    evidence["validity_notes"] = {
        "tier_scope": (
            "Sod and Brio-Wu shock tubes are code-verification evidence for "
            "the MHD solver. They do not validate a DPF discharge against "
            "spatial density, magnetic-field, or temperature measurements."
        ),
    }
    return evidence


def _phase_target_key(name: object) -> str:
    label = _normalize_evidence_label(name)
    aliases = {
        "rundown": "axial",
        "axial_end": "axial",
        "axial_duration": "axial",
        "radial_start": "axial",
        "radial": "radial",
        "radial_duration": "radial",
        "radial_transit": "radial",
        "pinch": "pinch",
        "pinch_time": "pinch",
        "stagnation": "pinch",
        "stagnation_time": "pinch",
    }
    return aliases.get(label, label)


def _observed_snowplow_phase_times(
    times_s: object,
    phases: object,
) -> tuple[dict[str, float | None], dict[str, object]]:
    """Extract axial-end, radial-duration, and pinch times from phase labels."""
    times = [float(t) for t in times_s]
    labels = [_normalize_evidence_label(p) for p in phases]
    n = min(len(times), len(labels))
    times = times[:n]
    labels = labels[:n]

    radial_start = None
    pinch_time = None
    phases_seen: set[str] = set()
    for time_s, label in zip(times, labels):
        phases_seen.add(label)
        if radial_start is None and label in _RADIAL_PHASE_LABELS:
            radial_start = time_s
        if pinch_time is None and label in _PINCH_PHASE_LABELS:
            pinch_time = time_s

    radial_duration = None
    if radial_start is not None and pinch_time is not None:
        radial_duration = max(pinch_time - radial_start, 0.0)

    observed = {
        "axial": radial_start,
        "radial": radial_duration,
        "pinch": pinch_time,
    }
    details = {
        "radial_start_time_s": radial_start,
        "pinch_time_s": pinch_time,
        "phases_seen": sorted(phases_seen),
        "n_samples": n,
    }
    return observed, details


def snowplow_phase_observation_from_history(
    times_s: object,
    phases: object,
    *,
    source: str = (
        "Snowplow phase history observation; not validation without "
        "reference phase timing targets"
    ),
) -> dict[str, object]:
    """Summarize snowplow phase coverage without promoting it to validation."""
    observed, details = _observed_snowplow_phase_times(times_s, phases)
    phase_present = {
        phase: observed[phase] is not None
        for phase in _REQUIRED_SNOWPLOW_PHASES
    }
    return {
        "passed": False,
        "phases": phase_present,
        "observed_phase_times_s": observed,
        "source": source,
        "details": details,
        "validity_notes": {
            "tier_scope": (
                "Observed phase labels are useful diagnostics, but tier-2 "
                "support requires comparison with reference phase timing."
            ),
        },
    }


def snowplow_validation_evidence_from_phase_history(
    times_s: object,
    phases: object,
    reference_phase_times_s: dict[str, float],
    *,
    tolerance: float = 0.20,
    reference_source: str = "",
    reference_kr_status: str = "",
    require_kr_verified: bool = True,
    source: str = (
        "Snowplow phase/timing validation from phase history and reference "
        "targets"
    ),
) -> dict[str, object]:
    """Build strict snowplow validation evidence from phase-label history.

    Reference targets use:
        ``axial``: time when axial rundown ends / radial phase starts.
        ``radial``: radial transit duration from radial start to stagnation.
        ``pinch``: absolute stagnation or pinch time.
    """
    observed, details = _observed_snowplow_phase_times(times_s, phases)
    targets = {
        _phase_target_key(phase): float(value)
        for phase, value in reference_phase_times_s.items()
        if _phase_target_key(phase) in _REQUIRED_SNOWPLOW_PHASES
    }
    errors: dict[str, float] = {}
    for phase in _REQUIRED_SNOWPLOW_PHASES:
        observed_value = observed.get(phase)
        target_value = targets.get(phase)
        if observed_value is None or target_value is None or abs(target_value) == 0.0:
            errors[phase] = float("inf")
        else:
            errors[phase] = abs(observed_value - target_value) / abs(target_value)

    evidence = snowplow_validation_evidence_from_phase_errors(
        errors,
        tolerance=tolerance,
        source=source,
    )
    source_authority_passed = (
        not require_kr_verified
        or (
            str(reference_source).startswith("KnowledgeReference/")
            and reference_kr_status == "verified"
        )
    )
    evidence["passed"] = bool(evidence["passed"] and source_authority_passed)
    evidence["observed_phase_times_s"] = observed
    evidence["reference_phase_times_s"] = targets
    evidence["details"] = {
        **details,
        "source_authority": {
            "reference_source": reference_source,
            "reference_kr_status": reference_kr_status,
            "require_kr_verified": require_kr_verified,
            "passed": source_authority_passed,
        },
        "target_semantics": {
            "axial": "absolute axial-rundown end / radial-start time",
            "radial": "radial-transit duration from radial start to pinch",
            "pinch": "absolute stagnation / pinch time",
        },
    }
    evidence["validity_notes"] = {
        "tier_scope": (
            "Tier-2 support requires all three snowplow phase timings to "
            "match reference targets within tolerance. This remains a "
            "reduced-order Lee/snowplow phase validation, not spatial MHD "
            "validation."
        ),
        "source_authority": (
            "With require_kr_verified=True, reference phase targets must "
            "come from KnowledgeReference and be marked verified."
        ),
    }
    return evidence


def spatial_validation_evidence_from_quantity_errors(
    quantity_relative_errors: dict[str, float],
    *,
    tolerance: float = 0.50,
    source: str = "Spatial DPF diagnostic validation metrics",
) -> dict[str, object]:
    """Build strict spatial DPF validation evidence from diagnostic errors."""
    normalized_errors = {
        _normalize_evidence_label(quantity): float(error)
        for quantity, error in quantity_relative_errors.items()
    }
    diagnostics = {
        quantity: (quantity in normalized_errors and normalized_errors[quantity] <= tolerance)
        for quantity in _REQUIRED_SPATIAL_QUANTITIES
    }
    passed = all(diagnostics.values())
    return {
        "passed": passed,
        "diagnostics": diagnostics,
        "quantity_relative_errors": normalized_errors,
        "tolerance": tolerance,
        "source": source,
    }


def combine_spatial_validation_evidence(
    evidence_items: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    validation_scope: str | None = None,
    require_source_authority: bool = True,
    source: str = "Combined spatial validation evidence",
) -> dict[str, object]:
    """Combine partial spatial evidence only when validation scope is consistent."""
    diagnostics = {quantity: False for quantity in _REQUIRED_SPATIAL_QUANTITIES}
    component_summaries: list[dict[str, object]] = []
    scopes: list[str] = []

    for idx, evidence in enumerate(evidence_items):
        if not isinstance(evidence, dict):
            continue
        scope = str(
            evidence.get("validation_scope")
            or evidence.get("target")
            or evidence.get("source")
            or f"component_{idx}"
        )
        scopes.append(scope)
        covered = _evidence_items(
            evidence,
            "diagnostics",
            "validated_quantities",
            "quantities",
        )
        component_source = str(evidence.get("source", ""))
        source_authority_passed = (
            not require_source_authority
            or (
                component_source.startswith("KnowledgeReference/")
                and _safe_int(evidence.get("validation_tier", 0)) == 4
                and str(evidence.get("model_role", "")).startswith("simulation_to_kr_")
            )
        )
        if _evidence_passed(evidence) and source_authority_passed:
            for quantity in _REQUIRED_SPATIAL_QUANTITIES:
                if quantity in covered:
                    diagnostics[quantity] = True
        component_summaries.append({
            "scope": scope,
            "passed": _evidence_passed(evidence),
            "source_authority_passed": source_authority_passed,
            "covered_quantities": sorted(covered),
            "source": component_source,
            "target": evidence.get("target", ""),
        })

    unique_scopes = sorted(set(scopes))
    if validation_scope is not None:
        scope_consistent = all(scope == validation_scope for scope in scopes)
        combined_scope = validation_scope
    else:
        scope_consistent = len(unique_scopes) == 1 and bool(unique_scopes)
        combined_scope = unique_scopes[0] if scope_consistent else ""

    passed = scope_consistent and all(diagnostics.values())
    return {
        "passed": passed,
        "diagnostics": diagnostics,
        "validated_quantities": diagnostics,
        "validation_scope": combined_scope,
        "scope_consistent": scope_consistent,
        "component_scopes": unique_scopes,
        "source": source,
        "details": {
            "component_evidence": component_summaries,
            "required_quantities": sorted(_REQUIRED_SPATIAL_QUANTITIES),
            "missing_quantities": [
                quantity for quantity, ok in diagnostics.items() if not ok
            ],
        },
        "validity_notes": {
            "scope_rule": (
                "Partial spatial evidence is combined only when all components "
                "share the same validation scope."
            ),
            "source_authority": (
                "With require_source_authority=True, only KR-sourced tier-4 "
                "component evidence contributes to combined spatial validation."
            ),
        },
    }


def spatial_validation_scope_closure_report(
    evidence_items: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    require_source_authority: bool = True,
) -> dict[str, object]:
    """Report same-scope Tier-4 closure status for partial spatial evidence."""
    scopes: dict[str, dict[str, object]] = {}

    for idx, evidence in enumerate(evidence_items):
        if not isinstance(evidence, dict):
            continue
        scope = str(
            evidence.get("validation_scope")
            or evidence.get("target")
            or evidence.get("source")
            or f"component_{idx}"
        )
        scope_record = scopes.setdefault(
            scope,
            {
                "validation_scope": scope,
                "covered_quantities": {q: False for q in _REQUIRED_SPATIAL_QUANTITIES},
                "components": [],
            },
        )
        covered = _evidence_items(
            evidence,
            "diagnostics",
            "validated_quantities",
            "quantities",
        )
        component_source = str(evidence.get("source", ""))
        source_authority_passed = (
            not require_source_authority
            or (
                component_source.startswith("KnowledgeReference/")
                and _safe_int(evidence.get("validation_tier", 0)) == 4
                and str(evidence.get("model_role", "")).startswith("simulation_to_kr_")
            )
        )
        contributes = _evidence_passed(evidence) and source_authority_passed
        if contributes:
            coverage = scope_record["covered_quantities"]
            if isinstance(coverage, dict):
                for quantity in _REQUIRED_SPATIAL_QUANTITIES:
                    if quantity in covered:
                        coverage[quantity] = True
        scope_record["components"].append({
            "target": evidence.get("target", ""),
            "source": component_source,
            "passed": _evidence_passed(evidence),
            "source_authority_passed": source_authority_passed,
            "contributes": contributes,
            "covered_quantities": sorted(covered),
        })

    scope_reports = []
    closed_scopes = []
    for scope, record in sorted(scopes.items()):
        coverage = record["covered_quantities"]
        covered_quantities = (
            dict(coverage) if isinstance(coverage, dict)
            else {q: False for q in _REQUIRED_SPATIAL_QUANTITIES}
        )
        missing = [
            quantity for quantity in sorted(_REQUIRED_SPATIAL_QUANTITIES)
            if covered_quantities.get(quantity) is not True
        ]
        scope_closed = not missing
        if scope_closed:
            closed_scopes.append(scope)
        scope_reports.append({
            "validation_scope": scope,
            "closed": scope_closed,
            "covered_quantities": covered_quantities,
            "missing_quantities": missing,
            "components": record["components"],
        })

    return {
        "passed": bool(closed_scopes),
        "validation_tier": 4,
        "model_role": "spatial_same_scope_closure_report",
        "required_quantities": sorted(_REQUIRED_SPATIAL_QUANTITIES),
        "closed_scopes": closed_scopes,
        "scopes": scope_reports,
        "validity_notes": {
            "same_scope_rule": (
                "Tier-4 spatial validation requires density, magnetic-field, "
                "and temperature components that share the same validation scope."
            ),
            "audit_role": (
                "This report identifies which partial spatial components can "
                "combine and which quantities remain missing per scope."
            ),
        },
    }


def neutron_timing_validation_evidence_from_errors(
    mechanism_timing_errors: dict[str, float],
    *,
    tolerance: float = 0.50,
    source: str = "Neutron mechanism/timing validation metrics",
) -> dict[str, object]:
    """Build strict neutron mechanism/timing validation evidence."""
    normalized_errors = {
        _normalize_evidence_label(mechanism): float(error)
        for mechanism, error in mechanism_timing_errors.items()
    }
    mechanisms = {
        mechanism: (
            mechanism in normalized_errors
            and normalized_errors[mechanism] <= tolerance
        )
        for mechanism in _REQUIRED_NEUTRON_MECHANISMS
    }
    passed = all(mechanisms.values())
    return {
        "passed": passed,
        "mechanisms": mechanisms,
        "timing_relative_errors": normalized_errors,
        "tolerance": tolerance,
        "source": source,
    }


def _neutron_validation_scope(evidence: object, fallback: str) -> str:
    if not isinstance(evidence, dict):
        return fallback
    details = evidence.get("details", {})
    target_id = details.get("target_id") if isinstance(details, dict) else None
    return str(
        evidence.get("validation_scope")
        or target_id
        or evidence.get("target")
        or evidence.get("source")
        or fallback
    )


def _neutron_uncertainty_source_values_present(evidence: object) -> bool:
    """Return True when neutron UQ evidence carries explicit source values."""
    if not isinstance(evidence, dict):
        return False
    for key in (
        "source_uncertainty_values",
        "source_uncertainty",
        "uncertainty_values",
        "source_error_bars",
        "source_standard_deviation",
    ):
        if evidence.get(key):
            return True
    details = evidence.get("details")
    if isinstance(details, dict):
        return any(
            details.get(key)
            for key in (
                "source_uncertainty_values",
                "source_uncertainty",
                "uncertainty_values",
                "source_error_bars",
                "source_standard_deviation",
            )
        )
    return False


def neutron_validation_scope_closure_report(result: dict) -> dict[str, object]:
    """Report same-scope Tier-5 closure for neutron validation evidence."""
    components = {
        "yield": (
            result.get("neutron_yield_validation"),
            _REQUIRED_NEUTRON_YIELD_FEATURES,
            ("validated_features", "diagnostics"),
        ),
        "timing": (
            result.get("neutron_mechanism_timing_validation"),
            _REQUIRED_NEUTRON_MECHANISMS,
            ("mechanisms", "validated_mechanisms"),
        ),
        "spectrum": (
            result.get("neutron_spectrum_validation"),
            _REQUIRED_NEUTRON_SPECTRAL_FEATURES,
            ("validated_features", "diagnostics"),
        ),
        "anisotropy": (
            result.get("neutron_anisotropy_validation"),
            _REQUIRED_NEUTRON_ANISOTROPY_FEATURES,
            ("validated_features", "diagnostics"),
        ),
        "detector_response": (
            result.get("neutron_detector_response_validation"),
            _REQUIRED_NEUTRON_DETECTOR_FEATURES,
            ("validated_features", "diagnostics"),
        ),
        "uncertainty": (
            result.get("neutron_uncertainty_validation")
            or result.get("same_scope_neutron_uncertainty"),
            _REQUIRED_NEUTRON_UNCERTAINTY_FEATURES,
            ("validated_features", "diagnostics"),
        ),
    }
    scopes: dict[str, dict[str, object]] = {}

    for feature, (evidence, required, fields) in components.items():
        if not isinstance(evidence, dict):
            continue
        scope = _neutron_validation_scope(evidence, feature)
        record = scopes.setdefault(
            scope,
            {
                "validation_scope": scope,
                "covered_features": {
                    "yield": False,
                    "timing": False,
                    "spectrum": False,
                    "anisotropy": False,
                    "detector_response": False,
                    "uncertainty": False,
                },
                "components": [],
            },
        )
        source_authority_passed = _has_kr_source_authority(evidence, 5)
        required_covered = _covers_required(evidence, required, *fields)
        source_uncertainty_values_passed = (
            feature != "uncertainty"
            or _neutron_uncertainty_source_values_present(evidence)
        )
        contributes = (
            _evidence_passed(evidence)
            and source_authority_passed
            and required_covered
            and source_uncertainty_values_passed
        )
        if contributes:
            covered = record["covered_features"]
            if isinstance(covered, dict):
                covered[feature] = True
        record["components"].append({
            "feature": feature,
            "target": evidence.get("target", ""),
            "source": evidence.get("source", ""),
            "passed": _evidence_passed(evidence),
            "source_authority_passed": source_authority_passed,
            "required_covered": required_covered,
            "source_uncertainty_values_passed": source_uncertainty_values_passed,
            "contributes": contributes,
        })

    scope_reports = []
    closed_scopes = []
    required_features = [
        "anisotropy",
        "detector_response",
        "spectrum",
        "timing",
        "uncertainty",
        "yield",
    ]
    for scope, record in sorted(scopes.items()):
        covered = record["covered_features"]
        covered_features = (
            dict(covered) if isinstance(covered, dict)
            else {feature: False for feature in required_features}
        )
        missing = [
            feature for feature in required_features
            if covered_features.get(feature) is not True
        ]
        closed = not missing
        if closed:
            closed_scopes.append(scope)
        scope_reports.append({
            "validation_scope": scope,
            "closed": closed,
            "covered_features": covered_features,
            "missing_features": missing,
            "components": record["components"],
        })

    return {
        "passed": bool(closed_scopes),
        "validation_tier": 5,
        "model_role": "neutron_same_scope_closure_report",
        "required_features": required_features,
        "closed_scopes": closed_scopes,
        "scopes": scope_reports,
        "validity_notes": {
            "same_scope_rule": (
                "Tier-5 neutron validation requires yield, timing/mechanism, "
                "spectrum, anisotropy, detector/activation response, and "
                "uncertainty evidence from the same validation scope."
            ),
            "audit_role": (
                "This report prevents independently sourced neutron evidence "
                "from being combined into a single predictive-readiness claim."
            ),
        },
    }


@dataclass
class QualityCheck:
    """Single quality check result."""

    name: str
    passed: bool
    score: float       # 0-1
    message: str
    severity: str      # "critical", "warning", "info"


@dataclass
class ValidationTier:
    """Validation tier status for a simulation result."""

    level: int
    name: str
    status: str
    validation_role: str
    evidence: str
    limitation: str


@dataclass
class PredictiveReadiness:
    """End-to-end predictive-readiness gate for a simulation result."""

    ready: bool
    status: str
    claim_scope: str
    satisfied_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class ScientificAccuracyGap:
    """Remaining work item before high-fidelity predictive DPF claims."""

    area: str
    status: str
    blocker: str
    next_ratcheting_step: str
    done_condition: str


@dataclass
class HighFidelityReadiness:
    """Strict readiness gate for high-fidelity scientific DPF claims."""

    ready: bool
    status: str
    claim_scope: str
    predictive_status: str
    supported_areas: list[str] = field(default_factory=list)
    remaining_areas: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class QualityAssessment:
    """Overall simulation quality assessment."""

    checks: list[QualityCheck] = field(default_factory=list)
    grade: str = "F"
    score: float = 0.0
    summary: str = ""
    validation_tiers: list[ValidationTier] = field(default_factory=list)
    predictive_readiness: PredictiveReadiness | None = None

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_critical_failures(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "critical")


def validation_tier_report(result: dict) -> list[ValidationTier]:
    """Classify validation evidence without promoting estimates to validation."""
    has_current = result.get("I_peak", 0) > 0 and result.get("n_steps", 0) > 10
    circuit_evidence = result.get("circuit_validation")
    has_circuit_validation = (
        _evidence_passed(circuit_evidence)
        and _has_circuit_source_authority(circuit_evidence)
        and _covers_required(circuit_evidence, _REQUIRED_CIRCUIT_METRICS, "metrics")
    )
    has_snowplow = bool(result.get("has_snowplow"))
    snowplow_evidence = result.get("snowplow_validation")
    has_snowplow_validation = (
        _evidence_passed(snowplow_evidence)
        and _has_snowplow_source_authority(snowplow_evidence)
        and _covers_required(snowplow_evidence, _REQUIRED_SNOWPLOW_PHASES, "phases")
    )
    has_mhd = bool(result.get("has_mhd"))
    mhd_evidence = result.get("mhd_verification")
    has_mhd_verification = (
        _evidence_passed(mhd_evidence)
        and isinstance(mhd_evidence, dict)
        and _safe_int(mhd_evidence.get("validation_tier", 0)) == 3
        and str(mhd_evidence.get("model_role", "")).startswith("code_verification")
        and _covers_required(
            mhd_evidence,
            _REQUIRED_MHD_VERIFICATION_TESTS,
            "analytic_tests",
            "tests",
        )
    )
    spatial_evidence = result.get("spatial_validation")
    has_spatial_validation = (
        _evidence_passed(spatial_evidence)
        and _has_spatial_source_authority(spatial_evidence)
        and _covers_required(
            spatial_evidence,
            _REQUIRED_SPATIAL_QUANTITIES,
            "validated_quantities",
            "diagnostics",
        )
    )
    has_neutron = bool(
        result.get("neutron_yield")
        or result.get("neutron_yield_details")
        or result.get("neutron_yield_validation")
        or result.get("neutron_mechanism_timing_validation")
        or result.get("neutron_spectrum_validation")
        or result.get("neutron_anisotropy_validation")
        or result.get("neutron_detector_response_validation")
        or result.get("neutron_detector_response_validation_candidate")
        or result.get("neutron_uncertainty_validation")
        or result.get("same_scope_neutron_uncertainty")
    )
    neutron_yield_evidence = result.get("neutron_yield_validation")
    has_neutron_yield_validation = (
        _evidence_passed(neutron_yield_evidence)
        and _has_kr_source_authority(neutron_yield_evidence, 5)
        and _covers_required(
            neutron_yield_evidence,
            _REQUIRED_NEUTRON_YIELD_FEATURES,
            "validated_features",
            "diagnostics",
        )
    )
    neutron_evidence = result.get("neutron_mechanism_timing_validation")
    has_neutron_timing = (
        _evidence_passed(neutron_evidence)
        and _has_kr_source_authority(neutron_evidence, 5)
        and _covers_required(
            neutron_evidence,
            _REQUIRED_NEUTRON_MECHANISMS,
            "mechanisms",
            "validated_mechanisms",
        )
    )
    neutron_spectrum_evidence = result.get("neutron_spectrum_validation")
    has_neutron_spectrum = (
        _evidence_passed(neutron_spectrum_evidence)
        and _has_kr_source_authority(neutron_spectrum_evidence, 5)
        and _covers_required(
            neutron_spectrum_evidence,
            _REQUIRED_NEUTRON_SPECTRAL_FEATURES,
            "validated_features",
            "diagnostics",
        )
    )
    neutron_anisotropy_evidence = result.get("neutron_anisotropy_validation")
    has_neutron_anisotropy = (
        _evidence_passed(neutron_anisotropy_evidence)
        and _has_kr_source_authority(neutron_anisotropy_evidence, 5)
        and _covers_required(
            neutron_anisotropy_evidence,
            _REQUIRED_NEUTRON_ANISOTROPY_FEATURES,
            "validated_features",
            "diagnostics",
        )
    )
    neutron_scope_closure = neutron_validation_scope_closure_report(result)
    has_neutron_validation = bool(neutron_scope_closure.get("passed"))

    return [
        ValidationTier(
            level=1,
            name="Circuit/Lee waveform",
            status=(
                "supported" if has_circuit_validation else
                "diagnostic_present" if has_current else
                "missing"
            ),
            validation_role="strongest_current_evidence",
            evidence=(
                "Circuit waveform validation evidence is attached."
                if has_circuit_validation else
                "Current waveform scalar checks are present."
                if has_current else
                "No usable current waveform scalar checks are present."
            ),
            limitation="This does not validate spatial MHD pinch structure.",
        ),
        ValidationTier(
            level=2,
            name="Snowplow phase/timing",
            status=(
                "supported" if has_snowplow_validation else
                "partial" if has_snowplow else
                "not_assessed"
            ),
            validation_role="reduced_order_phase_check",
            evidence=(
                "Snowplow phase/timing validation evidence is attached."
                if has_snowplow_validation else
                "Snowplow/current-dip diagnostics are present."
                if has_snowplow else
                "No snowplow phase diagnostics are present."
            ),
            limitation="Reduced Lee/RADPF phase checks are not spatial MHD validation.",
        ),
        ValidationTier(
            level=3,
            name="Spatial MHD verification",
            status=(
                "supported" if has_mhd_verification else
                "verification_only" if has_mhd else
                "not_assessed"
            ),
            validation_role="code_verification_not_dpf_validation",
            evidence=(
                "MHD analytic/code-verification evidence is attached."
                if has_mhd_verification else
                "MHD backend flag is present; rely on analytic test-problem verification."
                if has_mhd else
                "No MHD backend evidence is present in this result."
            ),
            limitation="Verification tests do not validate DPF density, B-field, or temperature diagnostics.",
        ),
        ValidationTier(
            level=4,
            name="Spatial DPF experimental validation",
            status="supported" if has_spatial_validation else "not_validated",
            validation_role="highest_required_spatial_validation",
            evidence=(
                "Result declares spatial experimental validation evidence."
                if has_spatial_validation else
                "No spatially resolved DPF experimental validation is attached."
            ),
            limitation="Needed for predictive claims about sheath morphology and pinch state.",
        ),
        ValidationTier(
            level=5,
            name="Neutron yield/mechanism/timing/spectrum/anisotropy",
            status=(
                "supported" if has_neutron_validation else
                "decomposed_estimate" if has_neutron else
                "not_assessed"
            ),
            validation_role="estimate_until_neutron_yield_mechanism_spectrum_anisotropy_validated",
            evidence=(
                "Same-scope neutron yield, mechanism/timing, spectrum, anisotropy, detector/activation response, and uncertainty validation evidence is attached."
                if has_neutron_validation else
                "Neutron yield, mechanism/timing, spectrum, anisotropy, detector/activation response, and uncertainty evidence are present but not same-scope complete."
                if (
                    has_neutron_yield_validation
                    and has_neutron_timing
                    and has_neutron_spectrum
                    and has_neutron_anisotropy
                ) else
                "Neutron timing, spectrum, and anisotropy evidence are attached, but KR-backed scalar yield validation is missing."
                if has_neutron_timing and has_neutron_spectrum and has_neutron_anisotropy else
                "Neutron timing and spectrum evidence are attached, but anisotropy validation is missing."
                if has_neutron_timing and has_neutron_spectrum else
                "Neutron timing evidence is attached, but spectrum or anisotropy validation is missing."
                if has_neutron_timing else
                "Neutron spectrum or anisotropy evidence is attached, but timing validation is missing."
                if has_neutron_spectrum or has_neutron_anisotropy else
                "Neutron-yield estimates are present."
                if has_neutron else
                "No neutron-yield estimate is present."
            ),
            limitation=(
                "A total yield estimate is not validation of kinetic beam "
                "formation, same-scope scalar yield, neutron timing, spectrum, "
                "or angular anisotropy."
            ),
        ),
    ]


def predictive_readiness_report(
    result: dict,
    claim_scope: str = "end_to_end_deuterium_dpf",
) -> PredictiveReadiness:
    """Gate predictive DPF claims on required validation evidence."""
    tiers = validation_tier_report(result)
    tier_by_level = {tier.level: tier for tier in tiers}
    required = {
        1: "Circuit/Lee waveform validation",
        2: "Snowplow phase/timing validation",
        3: "Spatial MHD code verification",
        4: "Spatial DPF experimental validation",
        5: "Neutron yield/mechanism/timing/spectrum/anisotropy validation",
    }

    satisfied: list[str] = []
    missing: list[str] = []
    blockers: list[str] = []

    for level, label in required.items():
        tier = tier_by_level[level]
        if tier.status == "supported":
            satisfied.append(label)
        else:
            missing.append(label)
            blockers.append(f"T{level} {tier.name}: {tier.status} - {tier.limitation}")

    if not missing:
        neutron_scope_report = neutron_validation_scope_closure_report(result)
        neutron_scopes = (
            _scope_set_from_sequence(neutron_scope_report.get("closed_scopes", []))
            if neutron_scope_report.get("passed") is True
            else set()
        )
        evidence_scopes = {
            "circuit_validation": _passed_evidence_scope_set(
                result.get("circuit_validation")
            ),
            "snowplow_validation": _passed_evidence_scope_set(
                result.get("snowplow_validation")
            ),
            "spatial_validation": _spatial_same_scope_set(result),
            "neutron_validation": neutron_scopes,
        }
        missing_scopes = [
            name for name, scopes in evidence_scopes.items()
            if not scopes
        ]
        common_scopes = (
            set.intersection(*evidence_scopes.values())
            if not missing_scopes
            else set()
        )
        if missing_scopes:
            missing.append("Predictive validation scope alignment")
            blockers.append(
                "Predictive tiers are supported but missing validation_scope "
                f"metadata for {', '.join(missing_scopes)}."
            )
        elif not common_scopes:
            missing.append("Predictive validation scope alignment")
            blockers.append(
                "Predictive tiers are supported but do not share one "
                "validation_scope across circuit, snowplow, spatial, and "
                "neutron evidence."
            )

    validation_errors = result.get("validation_errors", [])
    if isinstance(validation_errors, list) and validation_errors:
        missing.append("Validation pipeline health")
        for err in validation_errors:
            if isinstance(err, dict):
                stage = err.get("stage", "unknown")
                error_type = err.get("error_type", "Exception")
                message = err.get("message", "")
                blockers.append(
                    f"Validation pipeline error in {stage}: {error_type} - {message}"
                )
            else:
                blockers.append(f"Validation pipeline error: {err}")

    ready = not missing
    if ready:
        status = "predictive_ready"
    elif isinstance(validation_errors, list) and validation_errors:
        status = "validation_pipeline_error"
    else:
        status = "not_predictive_ready"
    return PredictiveReadiness(
        ready=ready,
        status=status,
        claim_scope=claim_scope,
        satisfied_evidence=satisfied,
        missing_evidence=missing,
        blockers=blockers,
    )


def _gap_status_from_tier_status(tier_status: str) -> str:
    if tier_status == "supported":
        return "supported"
    if tier_status in {"diagnostic_present", "partial", "verification_only", "decomposed_estimate"}:
        return "partial"
    return "blocked"


def _kr_sourced_evidence_passed(evidence: object) -> bool:
    if not _evidence_passed(evidence) or not isinstance(evidence, dict):
        return False
    return str(evidence.get("source", "")).startswith("KnowledgeReference/")


def _source_line_items(
    source_lines: object,
    *,
    prefix: str = "",
) -> list[dict[str, str]]:
    """Flatten a source_lines scalar/list/dict into line-range records."""
    if isinstance(source_lines, str):
        line_range = source_lines.strip()
        if not line_range:
            return []
        return [{
            "source_line_key": prefix,
            "source_lines": line_range,
        }]
    if isinstance(source_lines, Mapping):
        items: list[dict[str, str]] = []
        for key, value in source_lines.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_source_line_items(value, prefix=child_prefix))
        return items
    if isinstance(source_lines, Sequence) and not isinstance(
        source_lines,
        (str, bytes, bytearray),
    ):
        items = []
        for idx, value in enumerate(source_lines):
            child_prefix = f"{prefix}[{idx}]" if prefix else str(idx)
            items.extend(_source_line_items(value, prefix=child_prefix))
        return items
    return []


def _resolve_kr_source_path(source: str) -> Path | None:
    if not source.startswith("KnowledgeReference/"):
        return None
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path if source_path.is_file() else None
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        candidate = base / source_path
        if candidate.is_file():
            return candidate
    return None


def _kr_source_file_exists(source: str) -> bool:
    return _resolve_kr_source_path(source) is not None


@lru_cache(maxsize=256)
def _kr_source_line_count(source: str) -> int:
    source_path = _resolve_kr_source_path(source)
    if source_path is None:
        return 0
    with source_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _line_range_within_source(source: str, line_range: str) -> bool:
    line_count = _kr_source_line_count(source)
    if line_count <= 0:
        return False
    ranges: list[tuple[int, int]] = []
    for part in line_range.split(","):
        match = _LINE_RANGE_RE.match(part)
        if match is None:
            return False
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        ranges.append((start, end))
    return bool(ranges) and all(
        1 <= start <= end <= line_count
        for start, end in ranges
    )


def _source_records_from_mapping(
    evidence_key: str,
    evidence: Mapping[str, object],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return KR source-line records and missing-source blockers for evidence."""
    records: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    source = str(evidence.get("source", ""))
    for line_item in _source_line_items(evidence.get("source_lines")):
        if _line_range_within_source(source, line_item["source_lines"]):
            records.append({
                "evidence_key": evidence_key,
                "source": source,
                **line_item,
            })

    source_basis = evidence.get("source_basis")
    line_basis = evidence.get("source_line_basis")
    if isinstance(source_basis, Mapping) and isinstance(line_basis, Mapping):
        for basis_key, basis_source in source_basis.items():
            basis_source_str = str(basis_source)
            for line_item in _source_line_items(
                line_basis.get(basis_key),
                prefix=str(basis_key),
            ):
                if _line_range_within_source(
                    basis_source_str,
                    line_item["source_lines"],
                ):
                    records.append({
                        "evidence_key": f"{evidence_key}.source_basis",
                        "source": basis_source_str,
                        **line_item,
                    })

    for container_key in (
        "required_evidence",
        "required_effects",
        "required_components",
    ):
        container = evidence.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for name, item in container.items():
            if not isinstance(item, Mapping):
                missing.append({
                    "evidence_key": f"{evidence_key}.{container_key}.{name}",
                    "reason": "missing_kr_source_or_line_range",
                })
                continue
            nested_records, nested_missing = _source_records_from_mapping(
                f"{evidence_key}.{container_key}.{name}",
                item,
            )
            records.extend(nested_records)
            missing.extend(nested_missing)

    if not records:
        missing.append({
            "evidence_key": evidence_key,
            "reason": "missing_kr_source_or_line_range",
        })
    return records, missing


def _source_authority_records_from_value(
    evidence_key: str,
    value: object,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    if isinstance(value, Mapping):
        if _evidence_passed(value):
            return _source_records_from_mapping(evidence_key, value)
        return records, missing

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value):
            item_key = f"{evidence_key}[{idx}]"
            item_records, item_missing = _source_authority_records_from_value(
                item_key,
                item,
            )
            records.extend(item_records)
            missing.extend(item_missing)
    return records, missing


def _dedupe_source_records(records: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        key = (
            record.get("evidence_key", ""),
            record.get("source", ""),
            record.get("source_line_key", ""),
            record.get("source_lines", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(record))
    return unique


def source_authority_evidence(
    *,
    validation_scope: str,
    sources: Sequence[str],
    source_lines: Sequence[str],
    provenance: str = "measured",
) -> dict[str, object]:
    """Build result-level KR source-authority evidence for a validation scope."""
    source_list = [str(source) for source in sources]
    line_list = [str(line_range) for line_range in source_lines]
    source_authority_passed = (
        bool(validation_scope)
        and bool(source_list)
        and len(source_list) == len(line_list)
        and all(
            _line_range_within_source(source, line_range)
            for source, line_range in zip(source_list, line_list, strict=True)
        )
        and provenance in {"measured", "kr_extracted", "published_table"}
    )
    return {
        "passed": source_authority_passed,
        "validation_tier": "source_authority",
        "model_role": "source_authority_validation",
        "validation_scope": validation_scope,
        "source": source_list[0] if source_list else "",
        "sources": source_list,
        "source_lines": line_list,
        "details": {
            "provenance": provenance,
            "n_sources": len(source_list),
        },
        "validity_notes": {
            "claim_scope": (
                "This evidence supports source authority for the stated "
                "validation scope only; it does not make unrelated registered "
                "devices validation-ready."
            ),
        },
    }


def source_authority_evidence_from_result(
    result: Mapping[str, object],
    *,
    validation_keys: Sequence[str] | None = None,
    validation_scope: str = "result_validation_evidence",
) -> dict[str, object]:
    """Audit KR source paths and line ranges for passed validation evidence."""
    keys = tuple(validation_keys or _SOURCE_AUTHORITY_VALIDATION_KEYS)
    records: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for key in keys:
        item_records, item_missing = _source_authority_records_from_value(
            key,
            result.get(key),
        )
        records.extend(item_records)
        missing.extend(item_missing)

    if validation_keys is None and _evidence_passed(result.get("spatial_validation")):
        components = result.get("spatial_validation_components")
        if isinstance(components, Sequence) and not isinstance(
            components,
            (str, bytes, bytearray),
        ):
            item_records, item_missing = _source_authority_records_from_value(
                "spatial_validation_components",
                components,
            )
        else:
            item_records, item_missing = _source_authority_records_from_value(
                "spatial_validation",
                result.get("spatial_validation"),
            )
        records.extend(item_records)
        missing.extend(item_missing)

    source_records = _dedupe_source_records(records)
    if not source_records and not missing:
        missing.append({
            "evidence_key": "result",
            "reason": "no_passed_validation_evidence",
        })

    passed = bool(source_records) and not missing
    sources = [record["source"] for record in source_records]
    line_ranges = [record["source_lines"] for record in source_records]
    claimed_evidence = {
        record["evidence_key"].split(".", 1)[0].split("[", 1)[0]
        for record in source_records
    }
    claimed_evidence.update(
        item["evidence_key"].split(".", 1)[0].split("[", 1)[0]
        for item in missing
        if item.get("evidence_key") != "result"
    )
    return {
        "passed": passed,
        "validation_tier": "source_authority",
        "model_role": "source_authority_validation",
        "validation_scope": validation_scope,
        "source": sources[0] if sources else "",
        "sources": sources,
        "source_lines": line_ranges,
        "details": {
            "claimed_evidence": sorted(claimed_evidence),
            "source_records": source_records,
            "missing_source_authority": missing,
            "n_sources": len(source_records),
        },
        "validity_notes": {
            "claim_scope": (
                "This audit covers only passed validation evidence records in "
                "the result. Candidate, failed, or absent evidence remains "
                "outside the result-level source-authority claim."
            ),
        },
    }


def _missing_source_authority_keys(evidence: object) -> list[str]:
    if not isinstance(evidence, Mapping):
        return []
    details = evidence.get("details", {})
    missing_authority = (
        details.get("missing_source_authority", [])
        if isinstance(details, Mapping)
        else []
    )
    missing_keys: list[str] = []
    if isinstance(missing_authority, Sequence) and not isinstance(
        missing_authority,
        (str, bytes, bytearray),
    ):
        for item in missing_authority:
            if isinstance(item, Mapping):
                key = str(item.get("evidence_key", ""))
                if key:
                    missing_keys.append(key)
    return missing_keys


def _kr_target_coverage_report_from_result(result: Mapping[str, object]) -> Mapping[str, object]:
    report = result.get("kr_validation_target_coverage")
    if (
        isinstance(report, Mapping)
        and report.get("model_role") == "kr_validation_target_coverage_report"
    ):
        return report
    try:
        from dpf.validation.kr_targets import kr_validation_target_coverage_report

        return kr_validation_target_coverage_report()
    except Exception:
        return {}


def _kr_target_semantic_audit_from_result(result: Mapping[str, object]) -> Mapping[str, object]:
    report = result.get("kr_validation_target_semantic_audit")
    if (
        isinstance(report, Mapping)
        and report.get("model_role") == "kr_validation_target_semantic_audit"
    ):
        return report
    try:
        from dpf.validation.kr_targets import kr_validation_target_semantic_audit

        return kr_validation_target_semantic_audit()
    except Exception:
        return {}


def _kr_same_scope_target_report_from_result(
    result: Mapping[str, object],
) -> Mapping[str, object]:
    report = result.get("kr_validation_same_scope_targets")
    if (
        isinstance(report, Mapping)
        and report.get("model_role") == "kr_validation_same_scope_target_report"
    ):
        return report
    try:
        from dpf.validation.kr_targets import kr_validation_same_scope_target_report

        return kr_validation_same_scope_target_report()
    except Exception:
        return {}


def _kr_corpus_review_status_from_result(result: Mapping[str, object]) -> Mapping[str, object]:
    report = result.get("kr_corpus_review_status")
    if (
        isinstance(report, Mapping)
        and report.get("model_role") == "kr_corpus_review_status"
    ):
        return report
    try:
        from dpf.validation.kr_corpus import kr_corpus_review_status

        return kr_corpus_review_status()
    except Exception:
        return {}


def _scope_set_from_sequence(value: object) -> set[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {str(item) for item in value if str(item)}
    return set()


def _scope_set_from_packet(
    evidence: object,
    scope_map_key: str,
) -> set[str]:
    if not isinstance(evidence, Mapping):
        return set()
    if evidence.get("passed") is not True:
        return set()
    if evidence.get("same_scope_passed") is not True:
        return set()
    scopes = evidence.get(scope_map_key)
    if isinstance(scopes, Mapping):
        return {str(scope) for scope in scopes.values() if str(scope)}
    scope = str(evidence.get("validation_scope", ""))
    return {scope} if scope else set()


def _target_same_scope_set(report: Mapping[str, object]) -> set[str]:
    if report.get("passed") is not True:
        return set()
    passed_scopes = _scope_set_from_sequence(report.get("passed_scopes", []))
    if passed_scopes:
        return passed_scopes
    best_scope = report.get("best_available_scope")
    if isinstance(best_scope, Mapping):
        scope = str(best_scope.get("validation_scope", ""))
        return {scope} if scope else set()
    return set()


def _passed_evidence_scope_set(evidence: object) -> set[str]:
    if not isinstance(evidence, Mapping):
        return set()
    if evidence.get("passed") is not True:
        return set()
    scope = str(evidence.get("validation_scope", ""))
    return {scope} if scope else set()


def _spatial_same_scope_set(result: Mapping[str, object]) -> set[str]:
    report = result.get("spatial_validation_scope_closure")
    if not isinstance(report, Mapping):
        components = result.get("spatial_validation_components")
        if isinstance(components, Sequence) and not isinstance(
            components,
            (str, bytes, bytearray),
        ):
            report = spatial_validation_scope_closure_report(list(components))
        else:
            spatial = result.get("spatial_validation")
            report = spatial_validation_scope_closure_report(
                [spatial] if isinstance(spatial, dict) else []
            )
    if report.get("passed") is not True:
        return set()
    return _scope_set_from_sequence(report.get("closed_scopes", []))


def _neutron_same_scope_set(result: Mapping[str, object]) -> set[str]:
    report = result.get("neutron_validation_scope_closure")
    if not isinstance(report, Mapping):
        report = neutron_validation_scope_closure_report(dict(result))
    if report.get("passed") is not True:
        return set()
    return _scope_set_from_sequence(report.get("closed_scopes", []))


def _high_fidelity_scope_alignment_report(
    result: Mapping[str, object],
    kr_same_scope_targets: Mapping[str, object],
) -> dict[str, object]:
    uq_evidence = result.get("uncertainty_validation") or result.get("uq_validation")
    evidence_scopes = {
        "source_authority_validation": _passed_evidence_scope_set(
            result.get("source_authority_validation")
        ),
        "kr_validation_same_scope_targets": _target_same_scope_set(
            kr_same_scope_targets
        ),
        "circuit_validation": _passed_evidence_scope_set(
            result.get("circuit_validation")
        ),
        "snowplow_validation": _passed_evidence_scope_set(
            result.get("snowplow_validation")
        ),
        "spatial_validation": _spatial_same_scope_set(result),
        "neutron_validation": _neutron_same_scope_set(result),
        "field_coupling_validation": _scope_set_from_packet(
            result.get("field_coupling_validation"),
            "component_validation_scopes",
        ),
        "physics_fidelity_evidence": _scope_set_from_packet(
            result.get("physics_fidelity_evidence"),
            "effect_validation_scopes",
        ),
        "uncertainty_validation": _scope_set_from_packet(
            uq_evidence,
            "component_validation_scopes",
        ),
    }
    missing_packets = [
        name for name, scopes in evidence_scopes.items()
        if not scopes
    ]
    if missing_packets:
        return {
            "status": "partial",
            "common_scopes": [],
            "evidence_scopes": {
                name: sorted(scopes) for name, scopes in evidence_scopes.items()
            },
            "missing_packets": missing_packets,
            "blocker": (
                "High-fidelity scope alignment is waiting on same-scope "
                f"evidence for {', '.join(missing_packets)}."
            ),
        }

    scope_sets = list(evidence_scopes.values())
    common_scopes = set.intersection(*scope_sets) if scope_sets else set()
    if common_scopes:
        return {
            "status": "supported",
            "common_scopes": sorted(common_scopes),
            "evidence_scopes": {
                name: sorted(scopes) for name, scopes in evidence_scopes.items()
            },
            "missing_packets": [],
            "blocker": (
                "Target, field-coupling, physics-fidelity, and uncertainty "
                "packets and tier evidence share one high-fidelity validation "
                "scope."
            ),
        }
    return {
        "status": "blocked",
        "common_scopes": [],
        "evidence_scopes": {
            name: sorted(scopes) for name, scopes in evidence_scopes.items()
        },
        "missing_packets": [],
        "blocker": (
            "Supported high-fidelity packets do not share one validation_scope."
        ),
    }


def scientific_accuracy_gap_report(result: dict | None = None) -> list[ScientificAccuracyGap]:
    """Return the remaining KR-gated work needed for high-fidelity DPF claims."""
    result = result or {}
    tiers = validation_tier_report(result)
    tier_status = {tier.level: tier.status for tier in tiers}
    readiness = predictive_readiness_report(result)

    ready_devices = 0
    total_devices = 0
    try:
        from dpf.validation.experimental import DEVICES, get_validation_ready_devices

        total_devices = len(DEVICES)
        ready_devices = len(get_validation_ready_devices())
    except Exception:
        total_devices = 0
        ready_devices = 0

    result_source_evidence = result.get("source_authority_validation")
    source_blocker = (
        f"{ready_devices}/{total_devices} registered devices have KR-verified "
        "measured waveform authority."
        if total_devices
        else "Validation-ready device registry could not be inspected."
    )
    if (
        isinstance(result_source_evidence, dict)
        and result_source_evidence.get("model_role") == "source_authority_validation"
    ):
        derived_source_evidence = source_authority_evidence_from_result(result)
        derived_missing_keys = _missing_source_authority_keys(derived_source_evidence)
        derived_has_claimed_evidence = bool(
            derived_source_evidence.get("details", {}).get("claimed_evidence", [])
            if isinstance(derived_source_evidence.get("details", {}), dict)
            else []
        )
        if (
            _kr_sourced_evidence_passed(result_source_evidence)
            and (
                derived_source_evidence.get("passed") is True
                or not derived_has_claimed_evidence
            )
        ):
            source_status = "supported"
        else:
            source_status = "blocked"
            missing_keys = (
                derived_missing_keys
                if derived_has_claimed_evidence
                else _missing_source_authority_keys(result_source_evidence)
            )
            source_blocker = (
                "Result-level source authority failed for "
                f"{', '.join(key for key in missing_keys if key) or 'passed evidence'}."
            )
    elif total_devices and ready_devices == total_devices:
        source_status = "supported"
    elif ready_devices > 0:
        source_status = "partial"
    else:
        source_status = "blocked"

    kr_target_coverage = _kr_target_coverage_report_from_result(result)
    kr_target_semantic = _kr_target_semantic_audit_from_result(result)
    kr_same_scope_targets = _kr_same_scope_target_report_from_result(result)
    kr_corpus_review = _kr_corpus_review_status_from_result(result)
    missing_target_groups = [
        str(group)
        for group in kr_target_coverage.get("missing_or_partial_groups", [])
        if str(group)
    ] if isinstance(kr_target_coverage, Mapping) else []
    failed_semantic_targets = [
        str(target)
        for target in kr_target_semantic.get("missing_or_failed_targets", [])
        if str(target)
    ] if isinstance(kr_target_semantic, Mapping) else []
    best_scope = (
        kr_same_scope_targets.get("best_available_scope", {})
        if isinstance(kr_same_scope_targets, Mapping)
        else {}
    )
    best_scope_missing = (
        best_scope.get("missing_groups", [])
        if isinstance(best_scope, Mapping)
        else []
    )
    best_scope_partial = (
        best_scope.get("partial_groups", [])
        if isinstance(best_scope, Mapping)
        else []
    )
    widest_scope = (
        kr_same_scope_targets.get("widest_available_scope", {})
        if isinstance(kr_same_scope_targets, Mapping)
        else {}
    )
    widest_scope_name = (
        str(widest_scope.get("validation_scope", ""))
        if isinstance(widest_scope, Mapping)
        else ""
    )
    widest_closure_groups = (
        widest_scope.get("closure_blocker_groups", [])
        if isinstance(widest_scope, Mapping)
        else []
    )
    unreviewed_relevant_sources = (
        kr_corpus_review.get("unreviewed_dpf_relevant_md_files", [])
        if isinstance(kr_corpus_review, Mapping)
        else []
    )
    if isinstance(unreviewed_relevant_sources, Sequence) and not isinstance(
        unreviewed_relevant_sources,
        (str, bytes, bytearray),
    ):
        unreviewed_relevant_count = len(unreviewed_relevant_sources)
    else:
        unreviewed_relevant_count = 0
    reviewed_relevant = (
        kr_corpus_review.get("reviewed_dpf_relevant_md_files", 0)
        if isinstance(kr_corpus_review, Mapping)
        else 0
    )
    relevant_total = 0
    if isinstance(kr_corpus_review, Mapping):
        counts = kr_corpus_review.get("corpus_counts", {})
        if isinstance(counts, Mapping):
            relevant_total = _safe_int(counts.get("dpf_relevant_md_files", 0))
    if unreviewed_relevant_count == 0 and relevant_total:
        corpus_status = "supported"
        corpus_blocker = (
            f"DPF-relevant KnowledgeReference markdown review is closed "
            f"({reviewed_relevant}/{relevant_total} files)."
        )
    elif unreviewed_relevant_count > 0:
        corpus_status = "partial"
        corpus_blocker = (
            f"{unreviewed_relevant_count} DPF-relevant KnowledgeReference "
            "markdown files still need coded targets or explicit review decisions."
        )
    else:
        corpus_status = "blocked"
        corpus_blocker = "KR corpus review status is unavailable."
    if (
        kr_target_coverage.get("passed") is True
        and kr_target_semantic.get("passed") is True
        and kr_same_scope_targets.get("passed") is True
    ):
        kr_target_status = "supported"
        kr_target_blocker = (
            "KR target coverage, same-scope coverage, and semantic "
            "source-window audits pass."
        )
    elif failed_semantic_targets:
        kr_target_status = "blocked"
        kr_target_blocker = (
            "KR target semantic source-window audit failed for "
            f"{', '.join(failed_semantic_targets)}."
        )
    elif missing_target_groups:
        kr_target_status = "partial"
        kr_target_blocker = (
            "KR validation target coverage is missing or partial for "
            f"{', '.join(missing_target_groups)}."
        )
        if widest_scope_name and widest_closure_groups:
            kr_target_blocker += (
                f" Widest same-scope closure path is {widest_scope_name}, "
                "with closure blockers for "
                f"{', '.join(str(group) for group in widest_closure_groups)}."
            )
    elif kr_same_scope_targets.get("passed") is not True:
        kr_target_status = "partial"
        kr_target_blocker = (
            "No single KR validation scope covers all target groups; best scope "
            f"is missing {', '.join(str(item) for item in best_scope_missing) or 'none'} "
            f"and partial for {', '.join(str(item) for item in best_scope_partial) or 'none'}."
        )
    else:
        kr_target_status = "blocked"
        kr_target_blocker = "KR validation target coverage or semantic audit is unavailable."
    high_fidelity_scope_alignment = _high_fidelity_scope_alignment_report(
        result,
        kr_same_scope_targets,
    )
    digitization_status_report = result.get("scientific_closure_digitization_status")
    if not isinstance(digitization_status_report, Mapping):
        try:
            from dpf.validation.digitization import (
                scientific_closure_digitization_status,
            )

            digitization_status_report = scientific_closure_digitization_status()
        except Exception:
            digitization_status_report = {}
    accepted_digitization_tasks = (
        _safe_int(digitization_status_report.get("accepted_task_count", 0))
        if isinstance(digitization_status_report, Mapping)
        else 0
    )
    failed_digitization_tasks = (
        _safe_int(digitization_status_report.get("failed_task_count", 0))
        if isinstance(digitization_status_report, Mapping)
        else 0
    )
    open_digitization_tasks = (
        _safe_int(digitization_status_report.get("open_task_count", 0))
        if isinstance(digitization_status_report, Mapping)
        else 0
    )
    total_digitization_tasks = (
        _safe_int(digitization_status_report.get("task_count", 0))
        if isinstance(digitization_status_report, Mapping)
        else 0
    )
    digitization_complete = bool(
        isinstance(digitization_status_report, Mapping)
        and digitization_status_report.get("queue_complete") is True
    )
    if digitization_complete:
        digitization_status = "supported"
        digitization_blocker = (
            "All local scientific-closure figure digitization tasks are accepted."
        )
    elif accepted_digitization_tasks:
        digitization_status = "partial"
        digitization_blocker = (
            f"{accepted_digitization_tasks}/{total_digitization_tasks} local "
            "scientific-closure figure digitization tasks are accepted; "
            f"{failed_digitization_tasks} failed and {open_digitization_tasks} "
            "remain open."
        )
    elif failed_digitization_tasks:
        digitization_status = "blocked"
        digitization_blocker = (
            f"0/{total_digitization_tasks} local scientific-closure figure "
            "digitization tasks are accepted; "
            f"{failed_digitization_tasks} draft/failed packet(s) need review "
            f"or correction and {open_digitization_tasks} remain open."
        )
    elif total_digitization_tasks:
        digitization_status = "blocked"
        digitization_blocker = (
            f"0/{total_digitization_tasks} local scientific-closure figure "
            "digitization tasks are accepted."
        )
    else:
        digitization_status = "blocked"
        digitization_blocker = "Local figure digitization status is unavailable."

    tier2_status = _gap_status_from_tier_status(tier_status.get(2, "not_assessed"))
    tier4_status = _gap_status_from_tier_status(tier_status.get(4, "not_assessed"))
    tier5_base_status = _gap_status_from_tier_status(
        tier_status.get(5, "not_assessed")
    )
    detector_response_evidence = result.get("neutron_detector_response_validation")
    detector_response_candidate = result.get(
        "neutron_detector_response_validation_candidate"
    )
    detector_response_supported = _kr_sourced_evidence_passed(
        detector_response_evidence
    )
    if tier5_base_status == "supported":
        tier5_status = "supported" if detector_response_supported else "partial"
    elif tier5_base_status == "partial" or isinstance(
        detector_response_candidate, dict
    ):
        tier5_status = "partial"
    else:
        tier5_status = "blocked"

    tier3_raw_status = tier_status.get(3, "not_assessed")
    mhd_numerical_evidence = result.get("mhd_numerical_fidelity")
    if _kr_sourced_evidence_passed(mhd_numerical_evidence):
        tier3_status = "supported"
    elif (
        isinstance(mhd_numerical_evidence, dict)
        and "same_scope_mhd_numerical_packet"
        in mhd_numerical_evidence.get("missing_or_unvalidated_evidence", [])
    ):
        tier3_status = "blocked"
    elif isinstance(mhd_numerical_evidence, dict):
        tier3_status = "partial"
    elif tier3_raw_status in {"supported", "verification_only"}:
        tier3_status = "partial"
    else:
        tier3_status = "blocked"

    physics_evidence = result.get("physics_fidelity_evidence")
    physics_status = (
        "supported" if _kr_sourced_evidence_passed(physics_evidence) else "blocked"
    )
    coupling_evidence = result.get("field_coupling_validation")
    if _kr_sourced_evidence_passed(coupling_evidence):
        coupling_status = "supported"
    elif (
        isinstance(coupling_evidence, dict)
        and "same_scope_field_coupling_packet"
        in coupling_evidence.get("missing_or_unvalidated_evidence", [])
    ):
        coupling_status = "blocked"
    elif isinstance(coupling_evidence, dict) or any(
        key in result
        for key in (
            "Lp_mhd_nH",
            "Lp_nH",
            "L_p_nH",
            "L_plasma",
            "back_emf",
            "back_emf_V",
        )
    ):
        coupling_status = "partial"
    else:
        coupling_status = "blocked"

    uq_evidence = result.get("uncertainty_validation") or result.get("uq_validation")
    if _kr_sourced_evidence_passed(uq_evidence):
        uq_status = "supported"
    elif isinstance(uq_evidence, dict) or "uncertainty" in result:
        uq_status = "partial"
    else:
        uq_status = "blocked"

    export_status = (
        "supported"
        if "validation_tiers" in result and "predictive_readiness" in result
        else "partial"
    )

    return [
        ScientificAccuracyGap(
            area="source_authority_data",
            status=source_status,
            blocker=source_blocker,
            next_ratcheting_step=(
                "Extract line-referenced waveform, geometry, bank, fill, timing, "
                "uncertainty, and provenance records from KnowledgeReference."
            ),
            done_condition=(
                "Every validation device has KR source path, line range, "
                "waveform authority, uncertainty class, and pass/fail reason."
            ),
        ),
        ScientificAccuracyGap(
            area="kr_source_review",
            status=corpus_status,
            blocker=corpus_blocker,
            next_ratcheting_step=(
                "Review any newly added DPF-relevant KnowledgeReference "
                "markdown into a coded target or explicit decision before "
                "using it for claims."
            ),
            done_condition=(
                "The DPF-relevant KnowledgeReference markdown queue is empty "
                "and every source is represented by a coded target or explicit "
                "review decision."
            ),
        ),
        ScientificAccuracyGap(
            area="kr_target_coverage",
            status=kr_target_status,
            blocker=kr_target_blocker,
            next_ratcheting_step=(
                "Convert the missing or partial KR target groups into typed, "
                "same-scope, uncertainty-bearing validation target packets."
            ),
            done_condition=(
                "The KR target coverage report passes and every coded target "
                "also passes semantic source-window and same-scope target audits."
            ),
        ),
        ScientificAccuracyGap(
            area="figure_digitization",
            status=digitization_status,
            blocker=digitization_blocker,
            next_ratcheting_step=(
                "Render, crop, calibrate, digitize, hash, and independently "
                "review each open local figure task, then evaluate the packets "
                "with scientific_closure_digitization_status()."
            ),
            done_condition=(
                "Every local scientific-closure figure task has an accepted "
                "digitization packet tied to the exact KR source, source line "
                "window, figure ID, page, PDF hash, image hash, axis "
                "calibration, and required series."
            ),
        ),
        ScientificAccuracyGap(
            area="same_scope_high_fidelity_claim",
            status=str(high_fidelity_scope_alignment["status"]),
            blocker=str(high_fidelity_scope_alignment["blocker"]),
            next_ratcheting_step=(
                "Tie source authority, target coverage, circuit, snowplow, "
                "spatial, neutron, detector-response, field-coupling, "
                "physics-fidelity, and uncertainty evidence to one shared "
                "validation_scope before promoting high-fidelity readiness."
            ),
            done_condition=(
                "The high-fidelity target packet, tier evidence, and all "
                "required supporting evidence packets share at least one "
                "KR-backed validation_scope."
            ),
        ),
        ScientificAccuracyGap(
            area="snowplow_phase_validation",
            status=tier2_status,
            blocker=(
                f"Tier 2 snowplow phase/timing status is "
                f"{tier_status.get(2, 'not_assessed')}."
            ),
            next_ratcheting_step=(
                "Attach KR-backed axial, radial, and pinch timing targets to "
                "ordinary device/shot simulation runs."
            ),
            done_condition=(
                "Production runs emit tier-2 snowplow_validation only from "
                "same-device KR phase targets; targetless labels remain candidates."
            ),
        ),
        ScientificAccuracyGap(
            area="mhd_numerical_fidelity",
            status=tier3_status,
            blocker=(
                "Current tier-3 evidence must provide finite-volume, "
                "cylindrical, resistive, circuit-energy, backend-parity, "
                "restart, convergence, and scope-limit packets for the "
                "claimed MHD numerical scope."
            ),
            next_ratcheting_step=(
                "Add cylindrical, resistive, circuit-energy, convergence, "
                "Orszag-Tang/rotor, backend-parity, and restart "
                "reproducibility verification evidence."
            ),
            done_condition=(
                "Tier-3 evidence names all required ideal/resistive MHD, "
                "cylindrical geometry, circuit-coupling, convergence, "
                "backend, restart, and scope-limit tests with tolerances."
            ),
        ),
        ScientificAccuracyGap(
            area="spatial_dpf_validation",
            status=tier4_status,
            blocker=(
                f"Tier 4 spatial DPF validation status is "
                f"{tier_status.get(4, 'not_assessed')}."
            ),
            next_ratcheting_step=(
                "Ingest same-scope density, magnetic-field or EM, and "
                "temperature diagnostics with timing and uncertainty metadata."
            ),
            done_condition=(
                "Tier-4 support requires density, magnetic-field, and temperature "
                "evidence from one KR-backed device/shot/scope."
            ),
        ),
        ScientificAccuracyGap(
            area="neutron_validation",
            status=tier5_status,
            blocker=(
                f"Tier 5 neutron yield/mechanism/timing/spectrum/anisotropy/"
                f"detector/uncertainty status is {tier_status.get(5, 'not_assessed')}; "
                f"detector/activation "
                f"response status is "
                f"{'supported' if detector_response_supported else 'not_supported'}."
            ),
            next_ratcheting_step=(
                "Generate or ingest scalar yield, mechanism-separated neutron "
                "histories, spectra, anisotropy, detector response, and "
                "activation/yield uncertainty for one KR-backed scope."
            ),
            done_condition=(
                "Tier-5 support compares scalar yield, timing, spectrum, "
                "anisotropy, detector/activation observables, and uncertainty "
                "against one KR-backed neutron validation scope."
            ),
        ),
        ScientificAccuracyGap(
            area="missing_physics_fidelity",
            status=physics_status,
            blocker=(
                "Predictive late-pinch claims still need EOS, ionization, "
                "two-temperature, radiation transport/opacities, ablation or "
                "impurities, kinetic/Hall/FLR/PIC effects, 3D instabilities, "
                "flashover, restrike, and anomalous-resistance scope control."
            ),
            next_ratcheting_step=(
                "Create a physics-fidelity evidence record that marks each "
                "required effect as implemented, verified, validated, empirical, "
                "or absent for each run."
            ),
            done_condition=(
                "Predictive claims are limited to scopes whose required physics "
                "effects are implemented and validated, or shown not to control "
                "the target observable under KR-backed conditions."
            ),
        ),
        ScientificAccuracyGap(
            area="circuit_field_coupling",
            status=coupling_status,
            blocker=(
                "MHD current prediction still needs validated field-derived "
                "inductance, dL/dt, back-EMF, Poynting flux, and transition "
                "timing from snowplow to resolved MHD."
            ),
            next_ratcheting_step=(
                "Add field_coupling_validation evidence for inductance, dL/dt, "
                "back-EMF, Poynting flux, and circuit-energy balance."
            ),
            done_condition=(
                "MHD-mode circuit current is driven by validated field-derived "
                "coupling, with snowplow-loaded, blended, and field-coupled "
                "intervals exported separately."
            ),
        ),
        ScientificAccuracyGap(
            area="uncertainty_quantification",
            status=uq_status,
            blocker=(
                "High-fidelity readiness is not yet tied to full experimental, "
                "numerical, input, model-form, and shot-to-shot uncertainty budgets."
            ),
            next_ratcheting_step=(
                "Extend GUM/ASME-style uncertainty from circuit waveform "
                "comparisons into phase, spatial, neutron, and numerical evidence."
            ),
            done_condition=(
                "Every supported validation tier reports uncertainty components "
                "and the acceptance rule used for pass/fail."
            ),
        ),
        ScientificAccuracyGap(
            area="export_claim_hygiene",
            status=export_status,
            blocker=(
                f"Predictive-readiness status is {readiness.status}; downstream "
                "outputs must also carry the scientific-accuracy blocker list."
            ),
            next_ratcheting_step=(
                "Export this gap report beside validation_tiers and "
                "predictive_readiness in app/API results."
            ),
            done_condition=(
                "Every result exposes blockers, next ratchet, and done condition "
                "for each major scientific-accuracy area."
            ),
        ),
    ]


def high_fidelity_readiness_report(
    result: dict,
    claim_scope: str = "high_fidelity_end_to_end_deuterium_dpf",
) -> HighFidelityReadiness:
    """Gate high-fidelity scientific claims on tiers plus gap closure."""
    predictive = predictive_readiness_report(result)
    gaps = scientific_accuracy_gap_report(result)

    supported = [gap.area for gap in gaps if gap.status == "supported"]
    remaining = [gap.area for gap in gaps if gap.status != "supported"]
    blockers = [
        f"{gap.area}: {gap.status} - {gap.blocker}"
        for gap in gaps
        if gap.status != "supported"
    ]

    if not predictive.ready:
        status = "not_predictive_ready"
        blockers = [*predictive.blockers, *blockers]
    elif remaining:
        status = "scientific_accuracy_gaps_open"
    else:
        status = "high_fidelity_ready"

    return HighFidelityReadiness(
        ready=status == "high_fidelity_ready",
        status=status,
        claim_scope=claim_scope,
        predictive_status=predictive.status,
        supported_areas=supported,
        remaining_areas=remaining,
        blockers=blockers,
    )


def assess_quality(result: dict) -> QualityAssessment:
    """Assess simulation quality from result dict.

    Args:
        result: Simulation result from run_mhd_simulation or run_simulation_core.

    Returns:
        QualityAssessment with grade and detailed checks.
    """
    checks: list[QualityCheck] = []

    # 1. Current waveform — does it have a peak?
    I_peak = result.get("I_peak", 0)
    if I_peak > 0.01:
        checks.append(QualityCheck(
            "Current peak", True, 1.0,
            f"I_peak = {I_peak:.3f} MA", "critical",
        ))
    else:
        checks.append(QualityCheck(
            "Current peak", False, 0.0,
            f"No significant current peak (I_peak = {I_peak:.4f} MA)", "critical",
        ))

    # 2. Current dip (if snowplow present) — indicates radial compression
    if result.get("has_snowplow"):
        dip = result.get("dip_pct", 0)
        if dip > 1:
            score = min(dip / 20.0, 1.0)  # 20% dip = perfect
            checks.append(QualityCheck(
                "Current dip", True, score,
                f"Dip = {dip:.0f}% (indicates radial compression)", "warning",
            ))
        else:
            checks.append(QualityCheck(
                "Current dip", False, 0.0,
                "No current dip — radial compression may not have occurred", "warning",
            ))

    # 3. Simulation completed (enough steps)
    n_steps = result.get("n_steps", 0)
    if n_steps > 10:
        checks.append(QualityCheck(
            "Simulation length", True, min(n_steps / 100, 1.0),
            f"{n_steps} timesteps completed", "critical",
        ))
    else:
        checks.append(QualityCheck(
            "Simulation length", False, 0.0,
            f"Only {n_steps} steps — simulation may have failed early", "critical",
        ))

    # 4. Bennett equilibrium (if available)
    bennett = result.get("bennett")
    if bennett and bennett.get("T_bennett_keV", 0) > 0.01:
        T_B = bennett["T_bennett_keV"]
        checks.append(QualityCheck(
            "Bennett equilibrium", True, min(T_B / 5.0, 1.0),
            f"T_Bennett = {T_B:.2f} keV", "info",
        ))

    # 5. Neutron yield (for deuterium)
    ny = result.get("neutron_yield")
    if ny and ny.get("Y_neutron", 0) > 0:
        Yn = ny["Y_neutron"]
        bt = ny.get("bt_fraction", 0) * 100
        checks.append(QualityCheck(
            "Neutron yield", True, min(Yn / 1e8, 1.0),
            f"Yn = {Yn:.2e} ({bt:.0f}% beam-target)", "info",
        ))

    # 6. MHD density compression (if MHD backend)
    if result.get("has_mhd") and not result.get("has_snowplow"):
        import numpy as np
        rho_max = result.get("rho_max", [])
        rho0 = result.get("rho0", 1)
        if len(rho_max) > 0 and rho0 > 0:
            comp = float(np.max(rho_max)) / rho0
            if comp > 2.0:
                checks.append(QualityCheck(
                    "Density compression", True, min(comp / 10, 1.0),
                    f"Peak compression: {comp:.1f}x", "warning",
                ))
            else:
                checks.append(QualityCheck(
                    "Density compression", False, comp / 10,
                    f"Low compression ({comp:.1f}x) — grid may be too coarse", "warning",
                ))

    # 7. Breakdown mechanism (if available)
    bd = result.get("breakdown")
    if bd:
        checks.append(QualityCheck(
            "Breakdown model", True, 1.0,
            f"{bd['mechanism']} (CIV ratio {bd.get('civ_ratio', 0):.1f})", "info",
        ))

    # 8. Plasma regime (if available)
    regime = result.get("plasma_regime")
    if regime:
        Kn = regime.get("knudsen", 0)
        checks.append(QualityCheck(
            "Regime validity", regime.get("mhd_valid", False), 1.0 if Kn < 0.01 else 0.5,
            regime.get("summary", ""), "info",
        ))

    # Compute overall grade
    if not checks:
        readiness = predictive_readiness_report(result)
        return QualityAssessment(checks=checks, grade="F", score=0.0,
                                  summary="No data to assess",
                                  validation_tiers=validation_tier_report(result),
                                  predictive_readiness=readiness)

    total_score = sum(c.score for c in checks) / len(checks)
    n_critical_fail = sum(1 for c in checks if not c.passed and c.severity == "critical")

    if n_critical_fail > 0:
        grade = "F" if n_critical_fail > 1 else "D"
    elif total_score > 0.8:
        grade = "A"
    elif total_score > 0.6:
        grade = "B"
    elif total_score > 0.4:
        grade = "C"
    else:
        grade = "D"

    summary_parts = [f"Grade: {grade} ({total_score*100:.0f}%)"]
    for c in checks:
        icon = "PASS" if c.passed else "FAIL"
        summary_parts.append(f"  [{icon}] {c.name}: {c.message}")
    tiers = validation_tier_report(result)
    tier_summary = ", ".join(f"T{t.level}:{t.status}" for t in tiers)
    summary_parts.append(f"Validation tiers: {tier_summary}")
    readiness = predictive_readiness_report(result)
    summary_parts.append(
        f"Predictive readiness: {readiness.status} "
        f"({len(readiness.missing_evidence)} missing)"
    )

    return QualityAssessment(
        checks=checks,
        grade=grade,
        score=total_score,
        summary="\n".join(summary_parts),
        validation_tiers=tiers,
        predictive_readiness=readiness,
    )
