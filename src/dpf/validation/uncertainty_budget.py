"""Uncertainty-budget audit for high-fidelity DPF validation claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_KR_SOURCE_BASIS = {
    "plasma_uq_review": (
        "KnowledgeReference/2022-review-of-data-driven-plasma-science.md"
    ),
    "dpf_shot_variation": (
        "KnowledgeReference/paper-open-access-dense-plasma-focus-from-"
        "alternative-fusion-source-to-versatile-high-energy.md"
    ),
    "mjolnir_error_bars": (
        "KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-"
        "dense-plasma-focus-z-pinch-5.md"
    ),
    "beresnyak_voltage_uncertainty": (
        "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md"
    ),
    "malir_density_uncertainty": "KnowledgeReference/malir-2024-interferometry-dpf.md",
}


_REQUIRED_COMPONENTS = {
    "experimental_measurement_uncertainty": {
        "source_key": "beresnyak_voltage_uncertainty",
        "source_lines": "2689-2695",
        "requirement": (
            "Validation comparisons must carry measurement uncertainty for "
            "electrical and diagnostic data."
        ),
    },
    "input_parameter_uncertainty": {
        "source_key": "plasma_uq_review",
        "source_lines": "1118-1138",
        "requirement": (
            "Predictive computations require uncertainty in model inputs and "
            "parameters, not only nominal bank, geometry, and fill values."
        ),
    },
    "numerical_discretization_uncertainty": {
        "source_key": "malir_density_uncertainty",
        "source_lines": "381-390, 950-969",
        "requirement": (
            "Spatial comparisons must account for discretization, geometry, "
            "resolution, and setup limitations that affect quantitative outputs."
        ),
    },
    "model_form_uncertainty": {
        "source_key": "plasma_uq_review",
        "source_lines": "1118-1138",
        "requirement": (
            "Predictive claims require uncertainty in model structure and "
            "physics closures, including reduced MHD, coupling, and neutron "
            "mechanism assumptions."
        ),
    },
    "shot_to_shot_variability": {
        "source_key": "dpf_shot_variation",
        "source_lines": "399-406",
        "requirement": (
            "DPF validation must include shot-to-shot variation because the "
            "KR corpus identifies reproducibility as a device limitation."
        ),
    },
    "uncertainty_propagation_to_observables": {
        "source_key": "plasma_uq_review",
        "source_lines": "1118-1138",
        "requirement": (
            "Uncertainty must be propagated from inputs and model assumptions "
            "to output observables used for validation decisions."
        ),
    },
    "validation_acceptance_rule": {
        "source_key": "mjolnir_error_bars",
        "source_lines": "565-604",
        "requirement": (
            "Validation pass/fail decisions need explicit acceptance rules "
            "that use uncertainty, error bars, or standard deviations."
        ),
    },
    "kr_uncertainty_targets": {
        "source_key": "malir_density_uncertainty",
        "source_lines": "825-831, 983-986",
        "requirement": (
            "High-fidelity validation needs KR-backed uncertainty targets for "
            "the same observable, device, shot scope, and diagnostic."
        ),
    },
}

_VALIDATION_OBSERVABLES = (
    ("circuit_validation", "circuit_waveform"),
    ("snowplow_validation", "snowplow_phase"),
    ("snowplow_validation_candidate", "snowplow_phase_candidate"),
    ("snowplow_dynamics_validation_candidate", "snowplow_dynamics_candidate"),
    ("mhd_verification", "mhd_verification"),
    ("mhd_numerical_fidelity", "mhd_numerical_fidelity"),
    ("spatial_validation", "spatial_validation"),
    ("spatial_validation_candidate", "spatial_validation_candidate"),
    ("neutron_yield_validation", "neutron_yield"),
    ("neutron_mechanism_timing_validation", "neutron_timing"),
    ("neutron_spectrum_validation", "neutron_spectrum"),
    ("neutron_anisotropy_validation", "neutron_anisotropy"),
    ("neutron_detector_response_validation", "neutron_detector_response"),
    (
        "neutron_detector_response_validation_candidate",
        "neutron_detector_response_candidate",
    ),
)

_VALIDATION_TIER_AREAS = {
    1: {"circuit_waveform"},
    2: {
        "snowplow_phase",
        "snowplow_phase_candidate",
        "snowplow_dynamics_candidate",
    },
    3: {"mhd_verification", "mhd_numerical_fidelity"},
    4: {
        "spatial_validation",
        "spatial_validation_candidate",
        "spatial_component",
    },
    5: {
        "neutron_yield",
        "neutron_timing",
        "neutron_spectrum",
        "neutron_anisotropy",
        "neutron_detector_response",
        "neutron_detector_response_candidate",
    },
}

_UNCERTAINTY_FIELDS = {
    "uncertainty",
    "uncertainty_budget",
    "measurement_uncertainty",
    "diagnostic_uncertainty",
    "experimental_uncertainty",
    "error_bars",
    "standard_deviation",
    "relative_sigma",
    "sigma",
    "confidence_interval",
}


def _is_nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    if isinstance(value, Sequence):
        return bool(value)
    return True


def _nested_get(mapping: Mapping[str, object], path: Sequence[str]) -> object:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _has_any(result: Mapping[str, object], *keys: str) -> tuple[bool, list[str]]:
    found = [key for key in keys if key in result and _is_nonempty(result.get(key))]
    return bool(found), found


def _has_nested(
    result: Mapping[str, object],
    paths: Mapping[str, Sequence[str]],
) -> tuple[bool, list[str]]:
    found = [
        label for label, path in paths.items()
        if _is_nonempty(_nested_get(result, path))
    ]
    return bool(found), found


def _uncertainty_paths_in_evidence(
    evidence: Mapping[str, object],
    prefix: str,
) -> list[str]:
    paths: list[str] = []
    for field in sorted(_UNCERTAINTY_FIELDS):
        if _is_nonempty(evidence.get(field)):
            paths.append(f"{prefix}.{field}")
    details = evidence.get("details")
    if isinstance(details, Mapping):
        for field in sorted(_UNCERTAINTY_FIELDS):
            if _is_nonempty(details.get(field)):
                paths.append(f"{prefix}.details.{field}")
    return paths


def _result_uncertainty_paths(
    result: Mapping[str, object],
    observable_key: str,
    area: str,
) -> list[str]:
    paths: list[str] = []
    tokens = {observable_key, area}
    for root in (
        "validation_uncertainty",
        "uncertainty_budget",
        "uq_summary",
        "uncertainty",
    ):
        value = result.get(root)
        if not isinstance(value, Mapping):
            continue
        for token in tokens:
            if _is_nonempty(value.get(token)):
                paths.append(f"{root}.{token}")
        observables = value.get("observables")
        if isinstance(observables, Mapping):
            for token in tokens:
                if _is_nonempty(observables.get(token)):
                    paths.append(f"{root}.observables.{token}")
    return paths


def _tier_uncertainty_status(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    status: dict[str, dict[str, object]] = {}
    for tier, areas in _VALIDATION_TIER_AREAS.items():
        present = [
            record for record in records
            if str(record.get("area", "")) in areas
        ]
        missing = [
            str(record.get("observable", ""))
            for record in present
            if record.get("has_uncertainty") is not True
        ]
        if not present:
            tier_status = "not_present"
        elif missing:
            tier_status = "missing_uncertainty"
        else:
            tier_status = "complete_for_present_observables"
        status[f"tier_{tier}"] = {
            "validation_tier": tier,
            "status": tier_status,
            "present_observables": [
                str(record.get("observable", ""))
                for record in present
            ],
            "missing_uncertainty_observables": missing,
        }
    return status


def validation_uncertainty_coverage_from_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Report which validation observables carry uncertainty metadata."""
    records: list[dict[str, object]] = []

    for observable_key, area in _VALIDATION_OBSERVABLES:
        evidence = result.get(observable_key)
        if not isinstance(evidence, Mapping):
            continue
        paths = _uncertainty_paths_in_evidence(evidence, observable_key)
        paths.extend(_result_uncertainty_paths(result, observable_key, area))
        records.append({
            "observable": observable_key,
            "area": area,
            "present": True,
            "has_uncertainty": bool(paths),
            "uncertainty_paths": sorted(set(paths)),
            "validation_scope": evidence.get("validation_scope"),
            "source": evidence.get("source", ""),
        })

    spatial_components = result.get("spatial_validation_components")
    if isinstance(spatial_components, Sequence) and not isinstance(
        spatial_components, (str, bytes, bytearray)
    ):
        for idx, evidence in enumerate(spatial_components):
            if not isinstance(evidence, Mapping):
                continue
            key = f"spatial_validation_components[{idx}]"
            paths = _uncertainty_paths_in_evidence(evidence, key)
            paths.extend(_result_uncertainty_paths(result, key, "spatial_component"))
            records.append({
                "observable": key,
                "area": "spatial_component",
                "present": True,
                "has_uncertainty": bool(paths),
                "uncertainty_paths": sorted(set(paths)),
                "validation_scope": evidence.get("validation_scope"),
                "source": evidence.get("source", ""),
            })

    missing = [
        str(record["observable"])
        for record in records
        if record.get("has_uncertainty") is not True
    ]
    tier_status = _tier_uncertainty_status(records)
    return {
        "passed": bool(records) and not missing,
        "validation_tier": "uncertainty_quantification",
        "model_role": "validation_uncertainty_coverage_audit",
        "source": _KR_SOURCE_BASIS["plasma_uq_review"],
        "source_lines": "1118-1138, 6889-6892",
        "observables": records,
        "tier_uncertainty_status": tier_status,
        "missing_uncertainty_observables": missing,
        "validity_notes": {
            "coverage_scope": (
                "This report checks whether each present validation evidence "
                "record carries uncertainty metadata. It does not validate the "
                "uncertainty model or the KR target uncertainty by itself."
            ),
        },
    }


def _kr_sourced_evidence_passed(evidence: object) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    if evidence.get("passed") is not True:
        return False
    return (
        str(evidence.get("source", "")).startswith("KnowledgeReference/")
        and _has_source_uncertainty_values(evidence)
    )


def _has_source_uncertainty_values(evidence: Mapping[str, object]) -> bool:
    """Return True only when evidence carries explicit source uncertainty data."""
    for key in (
        "source_uncertainty_values",
        "source_uncertainty",
        "uncertainty_values",
        "source_error_bars",
        "source_standard_deviation",
    ):
        if _is_nonempty(evidence.get(key)):
            return True
    details = evidence.get("details")
    if isinstance(details, Mapping):
        return any(
            _is_nonempty(details.get(key))
            for key in (
                "source_uncertainty_values",
                "source_uncertainty",
                "uncertainty_values",
                "source_error_bars",
                "source_standard_deviation",
            )
        )
    return False


def _record(
    name: str,
    *,
    status: str,
    present: bool,
    validated: bool = False,
    evidence_keys: Sequence[str] = (),
    notes: str,
) -> dict[str, object]:
    meta = _REQUIRED_COMPONENTS[name]
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


def uncertainty_component_evidence(
    component: str,
    *,
    validation_scope: str,
    source: str | None = None,
    source_lines: str | None = None,
    source_uncertainty_values: Mapping[str, object] | None = None,
    notes: str = "",
) -> dict[str, object]:
    """Build line-referenced evidence for one uncertainty-budget component."""
    component_key = str(component).strip().lower()
    known_component = component_key in _REQUIRED_COMPONENTS
    if known_component:
        meta = _REQUIRED_COMPONENTS[component_key]
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
        and _is_nonempty(source_uncertainty_values)
    )
    return {
        "passed": passed,
        "validation_tier": "uncertainty_quantification",
        "model_role": "uncertainty_component_validation",
        "component": component_key,
        "validation_scope": validation_scope,
        "source": source_value,
        "source_lines": line_value,
        "source_uncertainty_values": dict(source_uncertainty_values or {}),
        "details": {
            "known_component": known_component,
            "source_uncertainty_values_present": _is_nonempty(
                source_uncertainty_values
            ),
            "notes": notes,
        },
        "validity_notes": {
            "claim_scope": (
                "This evidence supports one uncertainty-budget component for "
                "the stated validation scope; it does not validate the other "
                "uncertainty components."
            ),
            "source_uncertainty_rule": (
                "Passing uncertainty-component evidence must carry explicit "
                "source uncertainty values; KR citations alone are not enough."
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
    if evidence.get("model_role") != "uncertainty_component_validation":
        return None
    if evidence.get("validation_tier") != "uncertainty_quantification":
        return None
    if str(evidence.get("component", "")).strip().lower() != component:
        return None
    if not str(evidence.get("source", "")).startswith("KnowledgeReference/"):
        return None
    if not evidence.get("validation_scope"):
        return None
    if not _has_source_uncertainty_values(evidence):
        return None
    return evidence


def _validated_components(
    result: Mapping[str, object],
) -> dict[str, tuple[Mapping[str, object], str]]:
    found: dict[str, tuple[Mapping[str, object], str]] = {}

    for container_key in (
        "uncertainty_component_validation",
        "uncertainty_component_validations",
    ):
        container = result.get(container_key)
        if isinstance(container, Mapping):
            for component, candidate in container.items():
                component_key = str(component).strip().lower()
                evidence = _valid_component_evidence(candidate, component_key)
                if evidence is not None:
                    found[component_key] = (evidence, container_key)
        elif isinstance(container, Sequence) and not isinstance(
            container, (str, bytes, bytearray)
        ):
            for candidate in container:
                if not isinstance(candidate, Mapping):
                    continue
                component_key = str(candidate.get("component", "")).strip().lower()
                evidence = _valid_component_evidence(candidate, component_key)
                if evidence is not None:
                    found[component_key] = (evidence, container_key)

    for component in _REQUIRED_COMPONENTS:
        key = f"{component}_validation"
        evidence = _valid_component_evidence(result.get(key), component)
        if evidence is not None:
            found[component] = (evidence, key)

    return found


def uncertainty_evidence_from_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Build a conservative uncertainty-budget evidence record for a run."""
    components: dict[str, dict[str, object]] = {}

    direct_measurement, direct_measurement_keys = _has_any(
        result,
        "measurement_uncertainty",
        "diagnostic_uncertainty",
        "experimental_uncertainty",
    )
    nested_measurement, nested_measurement_keys = _has_nested(
        result,
        {
            "circuit_validation.details.uncertainty": (
                "circuit_validation",
                "details",
                "uncertainty",
            ),
            "spatial_validation.uncertainty": ("spatial_validation", "uncertainty"),
            "neutron_validation.uncertainty": ("neutron_validation", "uncertainty"),
        },
    )
    measurement_present = direct_measurement or nested_measurement
    components["experimental_measurement_uncertainty"] = _record(
        "experimental_measurement_uncertainty",
        status="diagnostic_not_validated" if measurement_present else "absent",
        present=measurement_present,
        evidence_keys=direct_measurement_keys + nested_measurement_keys,
        notes=(
            "Measurement uncertainty is attached to at least one validation "
            "observable, but the full high-fidelity budget is not validated."
            if measurement_present
            else "No experimental measurement uncertainty is attached."
        ),
    )

    input_present, input_keys = _has_any(
        result,
        "input_uncertainty",
        "parameter_uncertainty",
        "bank_uncertainty",
        "geometry_uncertainty",
        "fill_uncertainty",
    )
    components["input_parameter_uncertainty"] = _record(
        "input_parameter_uncertainty",
        status="diagnostic_not_validated" if input_present else "absent",
        present=input_present,
        evidence_keys=input_keys,
        notes=(
            "Input or parameter uncertainty is present, but it is not yet "
            "propagated through all claimed validation observables."
            if input_present
            else "No bank, geometry, fill, or model-parameter uncertainty is attached."
        ),
    )

    numerical_present, numerical_keys = _has_any(
        result,
        "numerical_uncertainty",
        "grid_convergence",
        "convergence",
        "resolution_study",
        "discretization_error",
    )
    components["numerical_discretization_uncertainty"] = _record(
        "numerical_discretization_uncertainty",
        status="diagnostic_not_validated" if numerical_present else "absent",
        present=numerical_present,
        evidence_keys=numerical_keys,
        notes=(
            "Numerical uncertainty or convergence evidence is present, but no "
            "KR-scoped acceptance rule closes this component."
            if numerical_present
            else "No numerical discretization or convergence uncertainty is attached."
        ),
    )

    model_present, model_keys = _has_any(
        result,
        "model_form_uncertainty",
        "physics_fidelity_evidence",
        "field_coupling_validation",
        "plasma_regime",
    )
    components["model_form_uncertainty"] = _record(
        "model_form_uncertainty",
        status="blocker_reported" if model_present else "absent",
        present=model_present,
        evidence_keys=model_keys,
        notes=(
            "Model-form limitations are reported, but they are not quantified "
            "as uncertainty on the claimed observables."
            if model_present
            else "No model-form uncertainty or physics-scope blocker is attached."
        ),
    )

    shot_present, shot_keys = _has_any(
        result,
        "shot_to_shot_uncertainty",
        "shot_to_shot_variability",
        "multi_shot_uncertainty",
        "neutron_yield_uncertainty",
        "shot_statistics",
    )
    components["shot_to_shot_variability"] = _record(
        "shot_to_shot_variability",
        status="diagnostic_not_validated" if shot_present else "absent",
        present=shot_present,
        evidence_keys=shot_keys,
        notes=(
            "Shot-to-shot variability is represented, but it is not yet part "
            "of every validation acceptance rule."
            if shot_present
            else "No shot-to-shot variability budget is attached."
        ),
    )

    coverage = result.get("validation_uncertainty_coverage")
    if not isinstance(coverage, Mapping):
        coverage = validation_uncertainty_coverage_from_result(result)
    coverage_records = coverage.get("observables") if isinstance(coverage, Mapping) else []
    coverage_present = bool(coverage_records)
    coverage_complete = bool(
        isinstance(coverage, Mapping) and coverage.get("passed") is True
    )
    propagation_present, propagation_keys = _has_any(
        result,
        "uncertainty",
        "uncertainty_budget",
        "uq_samples",
        "uq_summary",
        "ensemble_uncertainty",
        "monte_carlo_uncertainty",
    )
    if coverage_present:
        propagation_keys.append("validation_uncertainty_coverage")
    propagation_present = propagation_present or coverage_present
    components["uncertainty_propagation_to_observables"] = _record(
        "uncertainty_propagation_to_observables",
        status=(
            "observable_coverage_present"
            if coverage_complete else
            "diagnostic_not_validated"
            if propagation_present else
            "absent"
        ),
        present=propagation_present,
        evidence_keys=propagation_keys,
        notes=(
            "Every present validation observable carries uncertainty metadata, "
            "but KR uncertainty targets and propagation validation are still open."
            if coverage_complete else
            "Some propagated uncertainty output is present, but high-fidelity "
            "readiness requires all validation observables to carry it."
            if propagation_present
            else "No propagated uncertainty output is attached."
        ),
    )

    rule_present, rule_keys = _has_any(
        result,
        "validation_tiers",
        "predictive_readiness",
        "high_fidelity_readiness",
    )
    nested_rule, nested_rule_keys = _has_nested(
        result,
        {
            "circuit_validation.details.peak_current_tolerance": (
                "circuit_validation",
                "details",
                "peak_current_tolerance",
            ),
            "circuit_validation.details.timing_tolerance": (
                "circuit_validation",
                "details",
                "timing_tolerance",
            ),
            "circuit_validation.details.waveform_tolerance": (
                "circuit_validation",
                "details",
                "waveform_tolerance",
            ),
        },
    )
    acceptance_present = rule_present or nested_rule
    components["validation_acceptance_rule"] = _record(
        "validation_acceptance_rule",
        status="rule_present_not_full_budget" if acceptance_present else "absent",
        present=acceptance_present,
        evidence_keys=rule_keys + nested_rule_keys,
        notes=(
            "Pass/fail rules are exported, but they are not yet driven by full "
            "uncertainty budgets for every required validation tier."
            if acceptance_present
            else "No uncertainty-aware validation acceptance rule is attached."
        ),
    )

    kr_evidence = result.get("kr_uncertainty_evidence")
    kr_validated = _kr_sourced_evidence_passed(kr_evidence)
    components["kr_uncertainty_targets"] = _record(
        "kr_uncertainty_targets",
        status="supported" if kr_validated else "validation_absent",
        present=isinstance(kr_evidence, Mapping),
        validated=kr_validated,
        evidence_keys=["kr_uncertainty_evidence"] if kr_evidence else [],
        notes=(
            "KR-backed uncertainty target evidence is attached."
            if kr_validated
            else "No passing same-scope KR-backed uncertainty target evidence is attached."
        ),
    )

    validated_component_scopes: dict[str, str] = {}
    for component, (component_evidence, evidence_key) in _validated_components(
        result
    ).items():
        if component not in components:
            continue
        validated_component_scopes[component] = str(
            component_evidence.get("validation_scope", "")
        )
        components[component] = _record(
            component,
            status="supported",
            present=True,
            validated=True,
            evidence_keys=[evidence_key],
            notes=(
                "KR-backed uncertainty component evidence is attached for the "
                "stated validation scope."
            ),
        )

    missing = [
        name for name, item in components.items()
        if item.get("validated") is not True
    ]
    scope_values = {
        scope for scope in validated_component_scopes.values()
        if scope
    }
    same_scope_passed = (
        not missing
        and bool(scope_values)
        and len(scope_values) == 1
    )
    if not missing and not same_scope_passed:
        missing.append("same_scope_uncertainty_packet")
    passed = not missing
    return {
        "passed": passed,
        "validation_tier": "uncertainty_quantification",
        "model_role": "uncertainty_budget_audit",
        "source": _KR_SOURCE_BASIS["plasma_uq_review"],
        "source_basis": _KR_SOURCE_BASIS,
        "required_components": components,
        "tier_uncertainty_status": (
            coverage.get("tier_uncertainty_status")
            if isinstance(coverage, Mapping)
            else {}
        ),
        "component_validation_scopes": validated_component_scopes,
        "same_scope_passed": same_scope_passed,
        "missing_or_unvalidated_components": missing,
        "validity_notes": {
            "claim_scope": (
                "A run is not high-fidelity predictive unless experimental, "
                "input, numerical, model-form, shot-to-shot, and propagated "
                "uncertainty are tied to the validation acceptance rule for "
                "one validation scope."
            ),
            "audit_role": (
                "This audit reports uncertainty-budget blockers; it does not "
                "turn nominal validation tolerances into validated UQ."
            ),
        },
    }
