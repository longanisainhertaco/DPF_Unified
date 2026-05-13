"""MHD numerical-fidelity audit for DPF validation claims."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real


_KR_SOURCE_BASIS = {
    "auluck_circuit_element": "KnowledgeReference/auluck-2021-dpf-circuit-element.md",
    "beresnyak_mhd_coupling": (
        "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md"
    ),
    "leveque_finite_volume_methods": (
        "KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md"
    ),
    "bennett_zpinch_equilibrium": (
        "KnowledgeReference/"
        "bennett-vorticity-analytic-solutions-to-a-flowing-nonlinear-"
        "shear-flow-stabilized-z-pinch.md"
    ),
    "mhd_resistive_diffusion": (
        "KnowledgeReference/"
        "modeling-and-simulation-in-science-engineering-and-technology-"
        "mathematical-models-and.md"
    ),
    "hall_anomalous_resistivity": (
        "KnowledgeReference/"
        "the-hall-term-and-anomalous-resistivity-effects-in-neon-gas-puff-"
        "z-pinches.md"
    ),
    "malir_resistive_dpf": "KnowledgeReference/malir-2024-interferometry-dpf.md",
}


_REQUIRED_EVIDENCE = {
    "finite_volume_mhd_verification": {
        "source_key": "leveque_finite_volume_methods",
        "source_lines": "1319-1467, 5239-5376, 5901-6038, 8061-8127, 8471-8584",
        "requirement": (
            "The MHD backend must have named finite-volume/ideal-MHD analytic "
            "verification evidence, not just a backend flag."
        ),
    },
    "cylindrical_geometry_verification": {
        "source_key": "beresnyak_mhd_coupling",
        "source_lines": "383-398, 1900-1955",
        "requirement": (
            "DPF MHD verification must cover cylindrical geometry, boundary "
            "conditions, and convergence against a cylindrical test solution."
        ),
    },
    "circuit_coupled_energy_verification": {
        "source_key": "beresnyak_mhd_coupling",
        "source_lines": "399-414",
        "requirement": (
            "The circuit equation and MHD voltage/current coupling must be "
            "verified as an energy-coupled numerical system."
        ),
    },
    "resistive_or_nonideal_verification": {
        "source_key": "malir_resistive_dpf",
        "source_lines": "511-541, 912-930",
        "requirement": (
            "Resistive or non-ideal terms used in DPF simulations need explicit "
            "verification because resistivity choices affect current-density "
            "and implosion structure."
        ),
    },
    "convergence_study": {
        "source_key": "beresnyak_mhd_coupling",
        "source_lines": "1939-1955",
        "requirement": (
            "Numerical fidelity requires convergence evidence with tolerances "
            "for the claimed DPF observables."
        ),
    },
    "backend_parity": {
        "source_key": "beresnyak_mhd_coupling",
        "source_lines": "336-347",
        "requirement": (
            "Multiple production backends need parity evidence so numerical "
            "results are not backend-specific artifacts."
        ),
    },
    "restart_reproducibility": {
        "source_key": "beresnyak_mhd_coupling",
        "source_lines": "336-347, 1900-1955",
        "requirement": (
            "Production MHD verification needs checkpoint/restart "
            "reproducibility evidence so long runs and backend comparisons are "
            "not artifacts of one uninterrupted execution path."
        ),
    },
    "dpf_scope_limit": {
        "source_key": "beresnyak_mhd_coupling",
        "source_lines": "2506-2519, 2690-2711",
        "requirement": (
            "Numerical evidence must state the DPF phase/scope where ideal or "
            "resistive MHD remains applicable."
        ),
    },
}

_FINITE_VOLUME_ANALYTIC_TESTS = {"sod", "brio_wu"}
_MHD_APPLICABLE_PHASES = {
    "formation",
    "plasma_column_formation",
    "pre_disruption",
    "before_disruption",
    "before_first_collapse",
    "first_collapse",
    "during_first_collapse",
    "rundown",
}
_MHD_INVALID_PHASES = {
    "after_disruption",
    "post_disruption",
    "after_first_collapse",
    "post_first_collapse",
    "secondary_collapse",
    "tertiary_collapse",
    "late_pinch",
}
_MHD_LIMIT_REASON_TERMS = {
    "nonideal",
    "non_ideal",
    "instability",
    "rayleigh_taylor",
    "disruption",
    "beyond_ideal_mhd",
}
_NUMERICAL_VERIFICATION_CLAIM_BOUNDARY = {
    "evidence_class": "code_numerical_verification",
    "experimental_dpf_validation": False,
    "supports_predictive_scientific_claims": False,
    "supports_high_fidelity_scientific_claims": False,
    "supports_validation_tiers": [3],
    "cannot_substitute_for_validation_tiers": [4, 5],
    "cannot_substitute_for": [
        "same_scope_spatial_dpf_validation",
        "neutron_timing_spectrum_anisotropy_validation",
        "reference_scientific_authority",
    ],
}


def _numerical_verification_claim_boundary() -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in _NUMERICAL_VERIFICATION_CLAIM_BOUNDARY.items()
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


def _has_any(result: Mapping[str, object], *keys: str) -> tuple[bool, list[str]]:
    found = [key for key in keys if key in result and _is_nonempty(result.get(key))]
    return bool(found), found


def _nested_get(mapping: Mapping[str, object], path: Sequence[str]) -> object:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _has_nested(
    result: Mapping[str, object],
    paths: Mapping[str, Sequence[str]],
) -> tuple[bool, list[str]]:
    found = [
        label for label, path in paths.items()
        if _is_nonempty(_nested_get(result, path))
    ]
    return bool(found), found


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
        **_numerical_verification_claim_boundary(),
        "source": _KR_SOURCE_BASIS[str(meta["source_key"])],
        "source_lines": meta["source_lines"],
        "requirement": meta["requirement"],
        "evidence_keys": list(evidence_keys),
        "notes": notes,
    }


def _mhd_verification_passed(result: Mapping[str, object]) -> bool:
    evidence = result.get("mhd_verification")
    return isinstance(evidence, Mapping) and evidence.get("passed") is True


def _valid_finite_volume_mhd_verification(
    result: Mapping[str, object],
    *,
    method_declares_finite_volume: bool,
    method_has_solver: bool,
) -> bool:
    evidence = result.get("mhd_verification")
    if not isinstance(evidence, Mapping):
        return False
    if evidence.get("passed") is not True:
        return False
    if _as_finite_float(evidence.get("validation_tier")) != 3.0:
        return False
    if not str(evidence.get("model_role", "")).startswith("code_verification"):
        return False
    tests = evidence.get("analytic_tests")
    if not isinstance(tests, Mapping):
        tests = evidence.get("tests")
    if not isinstance(tests, Mapping):
        return False
    passed_tests = {str(name).lower() for name, ok in tests.items() if bool(ok)}
    return (
        method_declares_finite_volume
        and method_has_solver
        and _FINITE_VOLUME_ANALYTIC_TESTS.issubset(passed_tests)
    )


def _mhd_method_metadata(result: Mapping[str, object]) -> Mapping[str, object]:
    method = result.get("mhd_numerical_method")
    return method if isinstance(method, Mapping) else {}


def _as_finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_sequence(value: object) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return []
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError:
        return []
    numbers: list[float] = []
    for item in items:
        number = _as_finite_float(item)
        if number is None:
            return []
        numbers.append(number)
    return numbers


def _field_value(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _numeric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    numbers: dict[str, float] = {}
    for key, item in value.items():
        number = _as_finite_float(item)
        if number is None:
            continue
        numbers[str(key)] = number
    return numbers


def _normalized_labels(value: object) -> set[str]:
    if isinstance(value, (str, bytes, bytearray)):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        return set()
    labels = set()
    for item in items:
        label = str(item).strip().lower().replace("-", "_").replace(" ", "_")
        if label:
            labels.add(label)
    return labels


def _strictly_decreasing(values: Sequence[float]) -> bool:
    return len(values) >= 2 and all(
        later < earlier for earlier, later in zip(values, values[1:])
    )


def cylindrical_convergence_evidence_from_results(
    results: Mapping[str, object],
    *,
    verification_scope: str = "",
    min_convergence_order: float = 1.0,
) -> dict[str, object]:
    """Build Tier-3 evidence from a cylindrical z-pinch convergence run.

    The local convergence runner is a manufactured smooth cylindrical
    z-pinch equilibrium preservation test.  It supports only the narrow
    numerical claim checked here: cylindrical source terms and B-theta
    evolution converge for this analytic equilibrium.  It does not validate
    DPF phase dynamics, circuit coupling, or late-pinch physics.
    """
    resolutions = _numeric_sequence(results.get("resolutions"))
    btheta_errors = _numeric_sequence(results.get("Btheta_errors"))
    pressure_errors = _numeric_sequence(results.get("pressure_errors"))
    velocity_errors = _numeric_sequence(results.get("velocity_errors"))
    convergence_order = _as_finite_float(results.get("convergence_order"))

    metrics = {
        "three_or_more_resolutions": len(resolutions) >= 3,
        "btheta_errors_match_resolutions": (
            len(btheta_errors) == len(resolutions) and len(btheta_errors) > 0
        ),
        "finite_positive_btheta_errors": all(error > 0.0 for error in btheta_errors),
        "btheta_errors_decrease": _strictly_decreasing(btheta_errors),
        "convergence_order_passed": (
            convergence_order is not None
            and convergence_order >= float(min_convergence_order)
        ),
    }
    passed = all(metrics.values())
    missing = [name for name, ok in metrics.items() if not ok]

    return {
        "passed": passed,
        "validation_tier": 3,
        "model_role": "code_verification_cylindrical_convergence",
        **_numerical_verification_claim_boundary(),
        "verification_scope": verification_scope
        or str(results.get("verification_scope", "")),
        "source": _KR_SOURCE_BASIS["beresnyak_mhd_coupling"],
        "source_lines": "1900-1955",
        "source_basis": {
            "cylindrical_mhd_convergence": (
                _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
            ),
            "zpinch_equilibrium_force_balance": (
                _KR_SOURCE_BASIS["bennett_zpinch_equilibrium"]
            ),
        },
        "source_line_basis": {
            "cylindrical_mhd_convergence": "1900-1955",
            "zpinch_equilibrium_force_balance": "285-299, 386-397, 452-485",
        },
        "analytic_tests": {
            "cylindrical_zpinch_equilibrium_convergence": passed,
        },
        "metrics": metrics,
        "missing_or_failed_metrics": missing,
        "details": {
            "resolutions": [int(value) for value in resolutions],
            "Btheta_errors": btheta_errors,
            "pressure_errors": pressure_errors,
            "velocity_errors": velocity_errors,
            "convergence_order": convergence_order,
            "minimum_convergence_order": float(min_convergence_order),
        },
        "validity_notes": {
            "claim_scope": (
                "Supports cylindrical z-pinch equilibrium convergence only; "
                "it is not a DPF shot validation or a circuit-coupled "
                "verification."
            ),
            "kr_basis": (
                "Beresnyak reports cylindrical MHD verification and convergence "
                "against a theoretical self-similar solution; the Bennett "
                "reference supplies the local z-pinch force-balance basis."
            ),
        },
    }


def _valid_cylindrical_convergence_evidence(
    evidence: object,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "code_verification_cylindrical_convergence":
        return None
    if _as_finite_float(evidence.get("validation_tier")) != 3.0:
        return None
    source_basis = evidence.get("source_basis")
    source_ok = evidence.get("source") == _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
    if isinstance(source_basis, Mapping):
        source_ok = source_ok or (
            source_basis.get("cylindrical_mhd_convergence")
            == _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
        )
    return evidence if source_ok else None


def _cylindrical_convergence_evidence(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[str]]:
    for key in (
        "cylindrical_convergence_verification",
        "cylindrical_convergence_evidence",
    ):
        evidence = _valid_cylindrical_convergence_evidence(result.get(key))
        if evidence is not None:
            return evidence, [key]

    for key in ("cylindrical_convergence", "cylindrical_convergence_results"):
        raw = result.get(key)
        if not isinstance(raw, Mapping):
            continue
        evidence = cylindrical_convergence_evidence_from_results(raw)
        if evidence["passed"] is True:
            return evidence, [key]

    return None, []


def resistive_diffusion_convergence_evidence_from_results(
    results: object,
    *,
    verification_scope: str = "",
    min_convergence_order: float = 1.0,
) -> dict[str, object]:
    """Build Tier-3 evidence for the resistive magnetic-diffusion operator."""
    method = str(_field_value(results, "method") or "").lower()
    resolutions = _numeric_sequence(_field_value(results, "resolutions"))
    errors = _numeric_sequence(_field_value(results, "errors"))
    convergence_order = _as_finite_float(_field_value(results, "convergence_order"))
    eta = _as_finite_float(_field_value(results, "eta"))
    sigma0 = _as_finite_float(_field_value(results, "sigma0"))
    t_end = _as_finite_float(_field_value(results, "t_end"))

    metrics = {
        "recognized_diffusion_method": method in {"explicit", "implicit", "sts"},
        "three_or_more_resolutions": len(resolutions) >= 3,
        "errors_match_resolutions": len(errors) == len(resolutions) and len(errors) > 0,
        "finite_positive_errors": all(error > 0.0 for error in errors),
        "errors_decrease": _strictly_decreasing(errors),
        "convergence_order_passed": (
            convergence_order is not None
            and convergence_order >= float(min_convergence_order)
        ),
        "positive_resistivity": eta is not None and eta > 0.0,
    }
    passed = all(metrics.values())
    missing = [name for name, ok in metrics.items() if not ok]

    return {
        "passed": passed,
        "validation_tier": 3,
        "model_role": "code_verification_resistive_diffusion_convergence",
        **_numerical_verification_claim_boundary(),
        "verification_scope": verification_scope
        or str(_field_value(results, "verification_scope") or ""),
        "source": _KR_SOURCE_BASIS["mhd_resistive_diffusion"],
        "source_lines": "1288-1295, 1341-1358",
        "source_basis": {
            "resistive_magnetic_diffusion_operator": (
                _KR_SOURCE_BASIS["mhd_resistive_diffusion"]
            ),
            "dpf_resistivity_sensitivity": (
                _KR_SOURCE_BASIS["malir_resistive_dpf"]
            ),
            "anomalous_resistivity_scope_limit": (
                _KR_SOURCE_BASIS["hall_anomalous_resistivity"]
            ),
        },
        "source_line_basis": {
            "resistive_magnetic_diffusion_operator": "1288-1295, 1341-1358",
            "dpf_resistivity_sensitivity": "511-545, 908-930",
            "anomalous_resistivity_scope_limit": "17-38, 402-410",
        },
        "analytic_tests": {
            "resistive_magnetic_diffusion_convergence": passed,
        },
        "metrics": metrics,
        "missing_or_failed_metrics": missing,
        "details": {
            "method": method,
            "resolutions": [int(value) for value in resolutions],
            "errors": errors,
            "convergence_order": convergence_order,
            "minimum_convergence_order": float(min_convergence_order),
            "eta_ohm_m": eta,
            "sigma0_m": sigma0,
            "t_end_s": t_end,
        },
        "validity_notes": {
            "claim_scope": (
                "Supports the numerical resistive magnetic-diffusion operator "
                "only; it does not validate Spitzer closure, anomalous "
                "resistivity, Hall physics, or DPF current redistribution."
            ),
            "kr_basis": (
                "The KR corpus identifies resistive magnetic diffusion as a "
                "term in the magnetic-field evolution equation and separately "
                "shows DPF current-density structure is sensitive to the "
                "chosen resistivity model."
            ),
        },
    }


def _valid_resistive_diffusion_evidence(
    evidence: object,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "code_verification_resistive_diffusion_convergence":
        return None
    if _as_finite_float(evidence.get("validation_tier")) != 3.0:
        return None
    source_basis = evidence.get("source_basis")
    source_ok = evidence.get("source") == _KR_SOURCE_BASIS["mhd_resistive_diffusion"]
    if isinstance(source_basis, Mapping):
        source_ok = source_ok or (
            source_basis.get("resistive_magnetic_diffusion_operator")
            == _KR_SOURCE_BASIS["mhd_resistive_diffusion"]
        )
    return evidence if source_ok else None


def _resistive_diffusion_evidence(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[str]]:
    for key in (
        "resistive_diffusion_verification",
        "magnetic_diffusion_verification",
        "diffusion_convergence_verification",
    ):
        evidence = _valid_resistive_diffusion_evidence(result.get(key))
        if evidence is not None:
            return evidence, [key]

    for key in (
        "resistive_diffusion_convergence",
        "magnetic_diffusion_convergence",
        "diffusion_convergence",
    ):
        raw = result.get(key)
        if raw is None:
            continue
        evidence = resistive_diffusion_convergence_evidence_from_results(raw)
        if evidence["passed"] is True:
            return evidence, [key]

    return None, []


def _valid_circuit_coupled_energy_evidence(
    evidence: object,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "code_verification_circuit_coupled_energy_balance":
        return None
    if _as_finite_float(evidence.get("validation_tier")) != 3.0:
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


def backend_parity_evidence_from_results(
    results: Mapping[str, object],
    *,
    verification_scope: str = "",
    relative_tolerance: float = 0.05,
    required_observables: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build Tier-3 backend parity evidence from per-backend observables."""
    backend_block = results.get("backends")
    if not isinstance(backend_block, Mapping):
        backend_block = results.get("results_by_backend")
    if not isinstance(backend_block, Mapping):
        backend_block = {
            key: value for key, value in results.items() if isinstance(value, Mapping)
        }

    backend_outputs = {
        str(name): _numeric_mapping(values)
        for name, values in backend_block.items()
        if isinstance(values, Mapping)
    }
    backend_outputs = {
        name: values for name, values in backend_outputs.items() if values
    }
    backend_names = list(backend_outputs)

    if required_observables is None:
        common_observables = (
            sorted(set.intersection(*(set(values) for values in backend_outputs.values())))
            if backend_outputs
            else []
        )
    else:
        common_observables = [str(observable) for observable in required_observables]

    tolerance_block = results.get("relative_tolerances")
    tolerances = _numeric_mapping(tolerance_block)
    reference_backend = str(results.get("reference_backend") or "")
    if reference_backend not in backend_outputs and backend_names:
        reference_backend = backend_names[0]

    comparisons: dict[str, dict[str, float]] = {}
    max_relative_error = 0.0
    all_within_tolerance = True
    for backend in backend_names:
        if backend == reference_backend:
            continue
        backend_comparison: dict[str, float] = {}
        for observable in common_observables:
            ref_value = backend_outputs.get(reference_backend, {}).get(observable)
            value = backend_outputs[backend].get(observable)
            if ref_value is None or value is None:
                all_within_tolerance = False
                continue
            scale = max(abs(ref_value), abs(value), 1.0e-300)
            relative_error = abs(value - ref_value) / scale
            backend_comparison[observable] = relative_error
            max_relative_error = max(max_relative_error, relative_error)
            if relative_error > tolerances.get(observable, relative_tolerance):
                all_within_tolerance = False
        comparisons[backend] = backend_comparison

    metrics = {
        "two_or_more_backends": len(backend_outputs) >= 2,
        "common_observables_present": len(common_observables) > 0,
        "required_observables_present": all(
            all(observable in values for observable in common_observables)
            for values in backend_outputs.values()
        ),
        "relative_errors_within_tolerance": bool(
            all_within_tolerance and comparisons
        ),
    }
    passed = all(metrics.values())
    return {
        "passed": passed,
        "validation_tier": 3,
        "model_role": "code_verification_backend_parity",
        **_numerical_verification_claim_boundary(),
        "authority_label": "BackendParityVerification",
        "verification_scope": verification_scope
        or str(results.get("verification_scope", "")),
        "source": _KR_SOURCE_BASIS["beresnyak_mhd_coupling"],
        "source_lines": "1900-1903, 1939-1955",
        "source_basis": {
            "multi_code_verification_context": (
                _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
            ),
        },
        "source_line_basis": {
            "multi_code_verification_context": "1900-1903, 1939-1955",
        },
        "metrics": metrics,
        "missing_or_failed_metrics": [
            name for name, ok in metrics.items() if not ok
        ],
        "details": {
            "reference_backend": reference_backend,
            "backends": backend_names,
            "observables": common_observables,
            "relative_tolerance": relative_tolerance,
            "relative_tolerances": tolerances,
            "max_relative_error": max_relative_error,
            "comparisons": comparisons,
        },
        "validity_notes": {
            "claim_scope": (
                "Supports backend parity only for the supplied observables and "
                "tolerances. It does not validate those observables against "
                "experiment or prove convergence of the individual backends."
            ),
            "authority_boundary": (
                "Backend parity is a numerical consistency label, not "
                "Reference scientific authority."
            ),
        },
    }


def _valid_backend_parity_evidence(
    evidence: object,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "code_verification_backend_parity":
        return None
    if _as_finite_float(evidence.get("validation_tier")) != 3.0:
        return None
    return (
        evidence
        if evidence.get("source") == _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
        else None
    )


def _backend_parity_evidence(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[str]]:
    for key in ("backend_parity_verification", "backend_parity_evidence"):
        evidence = _valid_backend_parity_evidence(result.get(key))
        if evidence is not None:
            return evidence, [key]

    for key in ("backend_parity_results", "backend_comparison_results"):
        raw = result.get(key)
        if not isinstance(raw, Mapping):
            continue
        evidence = backend_parity_evidence_from_results(raw)
        if evidence["passed"] is True:
            return evidence, [key]

    return None, []


def restart_reproducibility_evidence_from_results(
    results: Mapping[str, object],
    *,
    verification_scope: str = "",
    relative_tolerance: float = 1.0e-9,
    required_observables: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build Tier-3 evidence for checkpoint/restart reproducibility."""
    continuous = results.get("continuous")
    if not isinstance(continuous, Mapping):
        continuous = results.get("uninterrupted")
    baseline_values = _numeric_mapping(continuous)

    restarted = results.get("restarted")
    if not isinstance(restarted, Mapping):
        restarted = results.get("restart")
    restarted_values = _numeric_mapping(restarted)

    if required_observables is None:
        common_observables = (
            sorted(set(baseline_values) & set(restarted_values))
            if baseline_values and restarted_values
            else []
        )
    else:
        common_observables = [str(observable) for observable in required_observables]

    tolerance_block = results.get("relative_tolerances")
    tolerances = _numeric_mapping(tolerance_block)
    comparisons: dict[str, float] = {}
    all_within_tolerance = True
    max_relative_error = 0.0
    for observable in common_observables:
        baseline = baseline_values.get(observable)
        restart = restarted_values.get(observable)
        if baseline is None or restart is None:
            all_within_tolerance = False
            continue
        scale = max(abs(baseline), abs(restart), 1.0e-300)
        relative_error = abs(restart - baseline) / scale
        comparisons[observable] = relative_error
        max_relative_error = max(max_relative_error, relative_error)
        if relative_error > tolerances.get(observable, relative_tolerance):
            all_within_tolerance = False

    config_hash = str(results.get("config_hash") or "")
    restart_config_hash = str(
        results.get("restart_config_hash")
        or results.get("checkpoint_config_hash")
        or ""
    )
    restart_marker = (
        results.get("restart_step")
        or results.get("checkpoint_step")
        or results.get("restart_time_s")
        or results.get("checkpoint_time_s")
    )
    metrics = {
        "continuous_run_observables_present": bool(baseline_values),
        "restarted_run_observables_present": bool(restarted_values),
        "restart_marker_present": _is_nonempty(restart_marker),
        "config_identity_declared": bool(config_hash and restart_config_hash),
        "config_hashes_match": bool(
            config_hash and restart_config_hash and config_hash == restart_config_hash
        ),
        "common_observables_present": len(common_observables) > 0,
        "required_observables_present": all(
            observable in baseline_values and observable in restarted_values
            for observable in common_observables
        ),
        "relative_errors_within_tolerance": bool(
            all_within_tolerance and comparisons
        ),
    }
    passed = all(metrics.values())
    return {
        "passed": passed,
        "validation_tier": 3,
        "model_role": "code_verification_restart_reproducibility",
        **_numerical_verification_claim_boundary(),
        "verification_scope": verification_scope
        or str(results.get("verification_scope", "")),
        "source": _KR_SOURCE_BASIS["beresnyak_mhd_coupling"],
        "source_lines": "336-347, 1900-1955",
        "source_basis": {
            "finite_volume_mhd_code_context": (
                _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
            ),
            "multi_code_verification_context": (
                _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
            ),
        },
        "source_line_basis": {
            "finite_volume_mhd_code_context": "336-347",
            "multi_code_verification_context": "1900-1955",
        },
        "metrics": metrics,
        "missing_or_failed_metrics": [
            name for name, ok in metrics.items() if not ok
        ],
        "details": {
            "restart_marker": restart_marker,
            "observables": common_observables,
            "relative_tolerance": relative_tolerance,
            "relative_tolerances": tolerances,
            "max_relative_error": max_relative_error,
            "comparisons": comparisons,
        },
        "validity_notes": {
            "claim_scope": (
                "Supports checkpoint/restart reproducibility only for the "
                "supplied observables, config identity, restart marker, and "
                "tolerances. It does not validate the physics model or the "
                "observables against experiment."
            ),
            "authority_boundary": (
                "Restart reproducibility is Tier-3 code verification, not "
                "Reference scientific authority or high-fidelity DPF validation."
            ),
        },
    }


def _valid_restart_reproducibility_evidence(
    evidence: object,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "code_verification_restart_reproducibility":
        return None
    if _as_finite_float(evidence.get("validation_tier")) != 3.0:
        return None
    return (
        evidence
        if evidence.get("source") == _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
        else None
    )


def _restart_reproducibility_evidence(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[str]]:
    for key in (
        "restart_reproducibility_verification",
        "checkpoint_restart_verification",
        "restart_reproducibility_evidence",
    ):
        evidence = _valid_restart_reproducibility_evidence(result.get(key))
        if evidence is not None:
            return evidence, [key]

    for key in (
        "restart_reproducibility_results",
        "checkpoint_restart_results",
    ):
        raw = result.get(key)
        if not isinstance(raw, Mapping):
            continue
        evidence = restart_reproducibility_evidence_from_results(raw)
        if evidence["passed"] is True:
            return evidence, [key]

    return None, []


def mhd_scope_limit_evidence_from_phases(
    applicable_phases: Sequence[str],
    invalid_phases: Sequence[str],
    *,
    verification_scope: str = "",
    limit_reasons: Sequence[str] = (),
) -> dict[str, object]:
    """Build evidence that an MHD claim is bounded by DPF phase."""
    applicable = _normalized_labels(applicable_phases)
    invalid = _normalized_labels(invalid_phases)
    reasons = _normalized_labels(limit_reasons)
    reason_blob = "_".join(sorted(reasons))
    metrics = {
        "pre_disruption_scope_declared": bool(
            applicable & _MHD_APPLICABLE_PHASES
        ),
        "post_disruption_or_post_collapse_excluded": bool(
            invalid & _MHD_INVALID_PHASES
        ),
        "limit_reason_declared": any(
            term in reason_blob for term in _MHD_LIMIT_REASON_TERMS
        ),
    }
    passed = all(metrics.values())
    return {
        "passed": passed,
        "validation_tier": 3,
        "model_role": "mhd_phase_scope_limit",
        **_numerical_verification_claim_boundary(),
        "verification_scope": verification_scope,
        "source": _KR_SOURCE_BASIS["beresnyak_mhd_coupling"],
        "source_lines": "2506-2519, 2689-2711",
        "source_basis": {
            "dpf_mhd_applicability_limit": (
                _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
            ),
        },
        "source_line_basis": {
            "dpf_mhd_applicability_limit": "2506-2519, 2689-2711",
        },
        "metrics": metrics,
        "missing_or_failed_metrics": [
            name for name, ok in metrics.items() if not ok
        ],
        "details": {
            "applicable_phases": sorted(applicable),
            "invalid_phases": sorted(invalid),
            "limit_reasons": sorted(reasons),
        },
        "validity_notes": {
            "claim_scope": (
                "Supports only an explicit phase boundary for MHD numerical "
                "claims. It does not validate the solver inside that boundary."
            ),
        },
    }


def _valid_mhd_scope_limit_evidence(
    evidence: object,
) -> Mapping[str, object] | None:
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("passed") is not True:
        return None
    if evidence.get("model_role") != "mhd_phase_scope_limit":
        return None
    if _as_finite_float(evidence.get("validation_tier")) != 3.0:
        return None
    return (
        evidence
        if evidence.get("source") == _KR_SOURCE_BASIS["beresnyak_mhd_coupling"]
        else None
    )


def _mhd_scope_limit_evidence(
    result: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[str]]:
    for key in (
        "mhd_scope_limit",
        "mhd_scope_limit_evidence",
        "mhd_scope_limit_validation",
    ):
        evidence = _valid_mhd_scope_limit_evidence(result.get(key))
        if evidence is not None:
            return evidence, [key]
    return None, []


def _verification_scope(evidence: object) -> str:
    if not isinstance(evidence, Mapping):
        return ""
    return str(
        evidence.get("verification_scope")
        or evidence.get("validation_scope")
        or ""
    )


def mhd_numerical_fidelity_evidence_from_result(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Build a conservative MHD numerical-fidelity evidence record for a run."""
    evidence: dict[str, dict[str, object]] = {}
    validated_evidence_scopes: dict[str, str] = {}

    method = _mhd_method_metadata(result)
    method_declares_finite_volume = method.get("finite_volume") is True
    method_has_solver = all(
        _is_nonempty(method.get(key))
        and str(method.get(key)).lower() != "unknown"
        for key in ("reconstruction", "riemann_solver", "time_integrator")
    )
    mhd_passed = _mhd_verification_passed(result)
    mhd_present, mhd_keys = _has_any(result, "mhd_verification")
    method_keys = ["mhd_numerical_method"] if method else []
    finite_volume_validated = _valid_finite_volume_mhd_verification(
        result,
        method_declares_finite_volume=method_declares_finite_volume,
        method_has_solver=method_has_solver,
    )
    if finite_volume_validated:
        validated_evidence_scopes["finite_volume_mhd_verification"] = (
            _verification_scope(result.get("mhd_verification"))
            or str(method.get("verification_scope") or "")
        )
    evidence["finite_volume_mhd_verification"] = _record(
        "finite_volume_mhd_verification",
        status=(
            "supported"
            if finite_volume_validated else
            "implemented_not_complete"
            if mhd_passed else
            "method_metadata_only"
            if method_declares_finite_volume and method_has_solver else
            "absent"
        ),
        present=mhd_present or bool(method),
        validated=finite_volume_validated,
        evidence_keys=mhd_keys + method_keys,
        notes=(
            "Finite-volume method metadata and required Sod/Brio-Wu "
            "code-verification evidence are attached. This supports the "
            "generic finite-volume MHD channel only; DPF-specific cylindrical, "
            "circuit, resistive, convergence, backend, and scope evidence are "
            "separate audit channels."
            if finite_volume_validated
            else
            "Named MHD verification evidence is attached, but high-fidelity "
            "DPF numerical fidelity requires the additional cylindrical, "
            "circuit, convergence, and backend checks in this audit."
            if mhd_present
            else "Finite-volume MHD method metadata is exported, but analytic "
            "verification evidence is not attached."
            if method_declares_finite_volume and method_has_solver
            else "No named MHD analytic verification evidence is attached."
        ),
    )

    cylindrical_evidence, cylindrical_evidence_keys = (
        _cylindrical_convergence_evidence(result)
    )
    cylindrical_validated = cylindrical_evidence is not None
    if cylindrical_validated:
        validated_evidence_scopes["cylindrical_geometry_verification"] = (
            _verification_scope(cylindrical_evidence)
        )
        validated_evidence_scopes["convergence_study"] = (
            _verification_scope(cylindrical_evidence)
        )

    geometry_present, geometry_keys = _has_any(
        result,
        "cylindrical_verification",
        "magnetized_noh_verification",
        "cylindrical_mhd_verification",
        "cylindrical_convergence_verification",
        "cylindrical_convergence_evidence",
        "cylindrical_convergence",
        "cylindrical_convergence_results",
    )
    backend = str(result.get("backend", "")).lower()
    method_coordinates = str(method.get("coordinates", "")).lower()
    if "cylindrical" in backend or method_coordinates == "cylindrical":
        geometry_present = True
        geometry_keys = sorted(set(geometry_keys + ["backend"] + method_keys))
    if cylindrical_validated:
        geometry_present = True
        geometry_keys = sorted(set(geometry_keys + cylindrical_evidence_keys))
    evidence["cylindrical_geometry_verification"] = _record(
        "cylindrical_geometry_verification",
        status=(
            "supported"
            if cylindrical_validated else
            "diagnostic_not_validated"
            if geometry_present else
            "backend_scope_only"
            if "cylindrical" in backend else
            "absent"
        ),
        present=geometry_present,
        validated=cylindrical_validated,
        evidence_keys=geometry_keys,
        notes=(
            "KR-scoped cylindrical convergence evidence is attached. This "
            "supports cylindrical source-term verification, but not the "
            "remaining DPF numerical-fidelity channels."
            if cylindrical_validated
            else
            "Cylindrical MHD scope or verification evidence is present, but "
            "no complete KR-scoped cylindrical convergence evidence is attached."
            if geometry_present
            else "No cylindrical DPF MHD verification evidence is attached."
        ),
    )

    circuit_energy_evidence, circuit_energy_evidence_keys = (
        _circuit_coupled_energy_evidence(result)
    )
    circuit_energy_validated = circuit_energy_evidence is not None
    if circuit_energy_validated:
        validated_evidence_scopes["circuit_coupled_energy_verification"] = (
            _verification_scope(circuit_energy_evidence)
        )

    circuit_present, circuit_keys = _has_any(
        result,
        "circuit_coupled_energy_verification",
        "circuit_coupled_energy_validation",
        "field_coupling_validation",
        "circuit_energy_verification",
        "E_cap_kJ",
        "E_ind_kJ",
        "E_res_kJ",
        "L_p_nH",
        "back_emf_V",
    )
    if circuit_energy_validated:
        circuit_present = True
        circuit_keys = sorted(set(circuit_keys + circuit_energy_evidence_keys))
    evidence["circuit_coupled_energy_verification"] = _record(
        "circuit_coupled_energy_verification",
        status=(
            "supported"
            if circuit_energy_validated else
            "diagnostic_not_validated"
            if circuit_present else
            "absent"
        ),
        present=circuit_present,
        validated=circuit_energy_validated,
        evidence_keys=circuit_keys,
        notes=(
            "KR-scoped circuit/MHD power and integrated-energy evidence is "
            "attached for the supplied history. Full DPF numerical fidelity "
            "still requires the remaining audit channels."
            if circuit_energy_validated
            else
            "Circuit/MHD energy-coupling channels are present, but no complete "
            "energy-coupled MHD verification evidence is attached."
            if circuit_present
            else "No circuit-coupled MHD energy verification evidence is attached."
        ),
    )

    resistive_evidence, resistive_evidence_keys = _resistive_diffusion_evidence(result)
    resistive_validated = resistive_evidence is not None
    if resistive_validated:
        validated_evidence_scopes["resistive_or_nonideal_verification"] = (
            _verification_scope(resistive_evidence)
        )

    nonideal_present, nonideal_keys = _has_any(
        result,
        "resistive_verification",
        "resistive_diffusion_verification",
        "magnetic_diffusion_verification",
        "diffusion_convergence_verification",
        "resistive_diffusion_convergence",
        "magnetic_diffusion_convergence",
        "diffusion_convergence",
        "resistivity",
        "eta",
        "ohmic_heating",
        "R_anom",
        "post_pinch_empirical_resistance",
    )
    if resistive_validated:
        nonideal_present = True
        nonideal_keys = sorted(set(nonideal_keys + resistive_evidence_keys))
    evidence["resistive_or_nonideal_verification"] = _record(
        "resistive_or_nonideal_verification",
        status=(
            "supported"
            if resistive_validated else
            "implemented_not_validated"
            if nonideal_present else
            "absent"
        ),
        present=nonideal_present,
        validated=resistive_validated,
        evidence_keys=nonideal_keys,
        notes=(
            "KR-scoped resistive magnetic-diffusion convergence evidence is "
            "attached. This supports only the resistive operator; anomalous "
            "resistivity, Hall terms, and DPF current-density redistribution "
            "remain unvalidated."
            if resistive_validated
            else
            "Resistive or non-ideal channels are present, but no KR-scoped "
            "verification for their DPF impact is attached."
            if nonideal_present
            else "No resistive/non-ideal MHD verification evidence is attached."
        ),
    )

    convergence_present, convergence_keys = _has_any(
        result,
        "grid_convergence",
        "convergence",
        "resolution_study",
        "convergence_study",
        "cylindrical_convergence_verification",
        "cylindrical_convergence_evidence",
        "cylindrical_convergence",
        "cylindrical_convergence_results",
    )
    nested_convergence, nested_convergence_keys = _has_nested(
        result,
        {
            "mhd_verification.convergence": ("mhd_verification", "convergence"),
            "mhd_verification.tolerances": ("mhd_verification", "tolerances"),
        },
    )
    convergence_present = convergence_present or nested_convergence
    if cylindrical_validated:
        convergence_present = True
        convergence_keys = sorted(set(convergence_keys + cylindrical_evidence_keys))
    evidence["convergence_study"] = _record(
        "convergence_study",
        status=(
            "supported"
            if cylindrical_validated else
            "diagnostic_not_validated"
            if convergence_present else
            "absent"
        ),
        present=convergence_present,
        validated=cylindrical_validated,
        evidence_keys=convergence_keys + nested_convergence_keys,
        notes=(
            "Cylindrical z-pinch convergence evidence is attached. It closes "
            "only the cylindrical convergence slice; convergence remains "
            "needed for DPF observables, circuit coupling, non-ideal physics, "
            "and production backends."
            if cylindrical_validated
            else
            "Convergence or tolerance evidence is present, but it is not yet "
            "complete for all claimed DPF observables and backends."
            if convergence_present
            else "No convergence study evidence is attached."
        ),
    )

    backend_evidence, backend_evidence_keys = _backend_parity_evidence(result)
    backend_validated = backend_evidence is not None
    if backend_validated:
        validated_evidence_scopes["backend_parity"] = (
            _verification_scope(backend_evidence)
        )

    parity_present, parity_keys = _has_any(
        result,
        "backend_parity",
        "backend_parity_verification",
        "backend_parity_evidence",
        "backend_parity_results",
        "backend_comparison_results",
        "backend_comparison",
        "backend_validation",
    )
    if backend_validated:
        parity_present = True
        parity_keys = sorted(set(parity_keys + backend_evidence_keys))
    evidence["backend_parity"] = _record(
        "backend_parity",
        status=(
            "supported"
            if backend_validated else
            "diagnostic_not_validated"
            if parity_present else
            "single_backend_only"
            if result.get("backend") else
            "absent"
        ),
        present=parity_present,
        validated=backend_validated,
        evidence_keys=parity_keys or (["backend"] if result.get("backend") else []),
        notes=(
            "KR-scoped backend parity evidence is attached for the supplied "
            "observables and tolerances."
            if backend_validated
            else
            "Backend comparison evidence is present, but parity is not yet "
            "validated for all production backends."
            if parity_present
            else "Only one backend label is exported; no backend parity evidence is attached."
            if result.get("backend")
            else "No backend evidence is attached."
        ),
    )

    restart_evidence, restart_evidence_keys = _restart_reproducibility_evidence(result)
    restart_validated = restart_evidence is not None
    if restart_validated:
        validated_evidence_scopes["restart_reproducibility"] = (
            _verification_scope(restart_evidence)
        )

    restart_present, restart_keys = _has_any(
        result,
        "restart_reproducibility_verification",
        "checkpoint_restart_verification",
        "restart_reproducibility_evidence",
        "restart_reproducibility_results",
        "checkpoint_restart_results",
        "restart_metadata",
        "checkpoint_metadata",
        "restart_step",
        "checkpoint_step",
    )
    if restart_validated:
        restart_present = True
        restart_keys = sorted(set(restart_keys + restart_evidence_keys))
    evidence["restart_reproducibility"] = _record(
        "restart_reproducibility",
        status=(
            "supported"
            if restart_validated else
            "diagnostic_not_validated"
            if restart_present else
            "absent"
        ),
        present=restart_present,
        validated=restart_validated,
        evidence_keys=restart_keys,
        notes=(
            "KR-scoped checkpoint/restart reproducibility evidence is attached "
            "for the supplied observables and tolerances."
            if restart_validated
            else
            "Checkpoint or restart metadata is present, but no complete "
            "restart reproducibility evidence is attached."
            if restart_present
            else "No checkpoint/restart reproducibility evidence is attached."
        ),
    )

    scope_evidence, scope_evidence_keys = _mhd_scope_limit_evidence(result)
    scope_validated = scope_evidence is not None
    if scope_validated:
        validated_evidence_scopes["dpf_scope_limit"] = (
            _verification_scope(scope_evidence)
        )

    scope_present, scope_keys = _has_any(
        result,
        "plasma_regime",
        "physics_fidelity_evidence",
        "mhd_scope_limit",
        "validity_notes",
    )
    if scope_validated:
        scope_present = True
        scope_keys = sorted(set(scope_keys + scope_evidence_keys))
    evidence["dpf_scope_limit"] = _record(
        "dpf_scope_limit",
        status=(
            "supported"
            if scope_validated else
            "scope_limiter_reported"
            if scope_present else
            "absent"
        ),
        present=scope_present,
        validated=scope_validated,
        evidence_keys=scope_keys,
        notes=(
            "KR-scoped MHD phase/scope limit evidence is attached."
            if scope_validated
            else
            "MHD scope limitations are reported, but not yet tied to a "
            "validated phase-by-phase DPF applicability boundary."
            if scope_present
            else "No DPF phase/scope limit for the MHD model is attached."
        ),
    )

    missing = [
        name for name, item in evidence.items()
        if item.get("validated") is not True
    ]
    scope_values = {
        scope for scope in validated_evidence_scopes.values()
        if scope
    }
    same_scope_passed = (
        not missing
        and all(validated_evidence_scopes.get(name) for name in evidence)
        and len(scope_values) == 1
    )
    if not missing and not same_scope_passed:
        missing.append("same_scope_mhd_numerical_packet")
    passed = not missing
    return {
        "passed": passed,
        "validation_tier": "mhd_numerical_fidelity",
        "model_role": "mhd_numerical_fidelity_audit",
        **_numerical_verification_claim_boundary(),
        "source": _KR_SOURCE_BASIS["beresnyak_mhd_coupling"],
        "source_basis": _KR_SOURCE_BASIS,
        "required_evidence": evidence,
        "evidence_verification_scopes": validated_evidence_scopes,
        "same_scope_passed": same_scope_passed,
        "missing_or_unvalidated_evidence": missing,
        "validity_notes": {
            "claim_scope": (
                "Tier-3 MHD evidence is not high-fidelity complete unless "
                "finite-volume, cylindrical, circuit-coupled, resistive, "
                "convergence, backend-parity, restart reproducibility, and "
                "scope-limit evidence are validated for the claimed DPF phase."
            ),
            "audit_role": (
                "This audit separates generic MHD backend presence from "
                "DPF-specific numerical-fidelity evidence."
            ),
        },
    }


def _copy_mapping_with_scope(value: object, verification_scope: str) -> object:
    if not isinstance(value, Mapping):
        return value
    copied = dict(value)
    copied.setdefault("verification_scope", verification_scope)
    return copied


def build_mhd_numerical_verification_packet(
    result: Mapping[str, object] | None = None,
    *,
    verification_scope: str,
    mhd_numerical_method: Mapping[str, object] | None = None,
    mhd_verification: Mapping[str, object] | None = None,
    cylindrical_convergence: Mapping[str, object] | None = None,
    circuit_coupled_energy_verification: Mapping[str, object] | None = None,
    resistive_diffusion_convergence: object | None = None,
    backend_parity_results: Mapping[str, object] | None = None,
    restart_reproducibility_results: Mapping[str, object] | None = None,
    mhd_scope_limit: Mapping[str, object] | None = None,
    applicable_phases: Sequence[str] = (),
    invalid_phases: Sequence[str] = (),
    limit_reasons: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble a reviewer-visible Tier-3 MHD numerical packet.

    The function builds evidence records from explicit verification-run
    outputs, then immediately runs the same conservative audit used by
    production summaries.  It is intentionally fail-closed: partial packets
    remain ``production_packet_status="blocked"`` and cannot substitute for
    DPF experimental validation.
    """
    if not verification_scope:
        raise ValueError("verification_scope is required for Tier-3 packet assembly")

    assembled: dict[str, object] = dict(result or {})

    if mhd_numerical_method is not None:
        assembled["mhd_numerical_method"] = _copy_mapping_with_scope(
            mhd_numerical_method,
            verification_scope,
        )
    if mhd_verification is not None:
        assembled["mhd_verification"] = _copy_mapping_with_scope(
            mhd_verification,
            verification_scope,
        )
    if cylindrical_convergence is not None:
        assembled["cylindrical_convergence_verification"] = (
            cylindrical_convergence_evidence_from_results(
                cylindrical_convergence,
                verification_scope=verification_scope,
            )
        )
    if circuit_coupled_energy_verification is not None:
        assembled["circuit_coupled_energy_verification"] = (
            _copy_mapping_with_scope(
                circuit_coupled_energy_verification,
                verification_scope,
            )
        )
    if resistive_diffusion_convergence is not None:
        assembled["resistive_diffusion_verification"] = (
            resistive_diffusion_convergence_evidence_from_results(
                resistive_diffusion_convergence,
                verification_scope=verification_scope,
            )
        )
    if backend_parity_results is not None:
        assembled["backend_parity_verification"] = (
            backend_parity_evidence_from_results(
                backend_parity_results,
                verification_scope=verification_scope,
            )
        )
    if restart_reproducibility_results is not None:
        assembled["restart_reproducibility_verification"] = (
            restart_reproducibility_evidence_from_results(
                restart_reproducibility_results,
                verification_scope=verification_scope,
            )
        )
    if mhd_scope_limit is not None:
        assembled["mhd_scope_limit"] = _copy_mapping_with_scope(
            mhd_scope_limit,
            verification_scope,
        )
    elif applicable_phases or invalid_phases or limit_reasons:
        assembled["mhd_scope_limit"] = mhd_scope_limit_evidence_from_phases(
            applicable_phases,
            invalid_phases,
            verification_scope=verification_scope,
            limit_reasons=limit_reasons,
        )

    audit = mhd_numerical_fidelity_evidence_from_result(assembled)
    status = mhd_numerical_verification_packet_status({
        **assembled,
        "mhd_numerical_fidelity": audit,
    })
    return {
        "packet_version": "1.0",
        "validation_tier": 3,
        "model_role": "mhd_numerical_verification_packet",
        **_numerical_verification_claim_boundary(),
        "passed": status["passed"],
        "production_packet_status": status["production_packet_status"],
        "verification_scope": verification_scope,
        "metadata": dict(metadata or {}),
        "result": assembled,
        "mhd_numerical_fidelity": audit,
        "mhd_numerical_verification_packet_status": status,
        "validity_notes": {
            "claim_scope": (
                "This packet is Tier-3 code numerical verification only. It "
                "does not validate DPF spatial observables, neutron outputs, "
                "or Reference scientific authority."
            ),
            "evidence_rule": (
                "The packet passes only when all required evidence channels "
                "are attached, validated, and share this verification scope."
            ),
        },
    }


def mhd_numerical_verification_packet_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Report production-visible Tier-3 verification packet status.

    This is a scheduling/reporting helper. It does not run verification jobs and
    it does not promote method metadata or diagnostic outputs into evidence.
    """
    audit = result.get("mhd_numerical_fidelity")
    if not isinstance(audit, Mapping):
        audit = mhd_numerical_fidelity_evidence_from_result(result)
    required = audit.get("required_evidence")
    required_evidence = required if isinstance(required, Mapping) else {}

    packet_next_actions = {
        "finite_volume_mhd_verification": (
            "Attach Tier-3 Sod and Brio-Wu finite-volume analytic-test evidence "
            "for the claimed method scope."
        ),
        "cylindrical_geometry_verification": (
            "Attach cylindrical z-pinch convergence evidence for the claimed "
            "geometry/backend scope."
        ),
        "circuit_coupled_energy_verification": (
            "Attach circuit/MHD power and integrated-energy balance evidence "
            "for the claimed coupling scope."
        ),
        "resistive_or_nonideal_verification": (
            "Attach resistive magnetic-diffusion convergence evidence for each "
            "enabled non-ideal operator."
        ),
        "convergence_study": (
            "Attach convergence evidence with grid sequence, observables, "
            "tolerances, and observed order."
        ),
        "backend_parity": (
            "Attach parity evidence across the production backends used for "
            "the claim observables."
        ),
        "restart_reproducibility": (
            "Attach checkpoint/restart evidence with matching config hashes, "
            "restart marker, and tolerance-bounded observables."
        ),
        "dpf_scope_limit": (
            "Attach explicit DPF phase/scope-limit evidence for where MHD "
            "verification is applicable."
        ),
    }

    packet_status: dict[str, dict[str, object]] = {}
    attached_validated: list[str] = []
    attached_diagnostic: list[str] = []
    missing_required: list[str] = []
    for name in _REQUIRED_EVIDENCE:
        item = required_evidence.get(name)
        if not isinstance(item, Mapping):
            packet_status[name] = {
                "status": "missing_required",
                "validated": False,
                "present": False,
                "evidence_keys": [],
                "next_action": packet_next_actions[name],
            }
            missing_required.append(name)
            continue
        validated = item.get("validated") is True
        present = item.get("present") is True
        if validated:
            status = "attached_validated"
            attached_validated.append(name)
        elif present:
            status = "attached_non_validating"
            attached_diagnostic.append(name)
            missing_required.append(name)
        else:
            status = "missing_required"
            missing_required.append(name)
        packet_status[name] = {
            "status": status,
            "validated": validated,
            "present": present,
            "audit_status": item.get("status"),
            "source": item.get("source"),
            "source_lines": item.get("source_lines"),
            "evidence_keys": list(item.get("evidence_keys", [])),
            "next_action": packet_next_actions[name],
        }

    complete = not missing_required and audit.get("same_scope_passed") is True
    return {
        "passed": complete,
        "validation_tier": 3,
        "model_role": "mhd_numerical_verification_packet_status",
        **_numerical_verification_claim_boundary(),
        "production_packet_status": "complete" if complete else "blocked",
        "same_scope_passed": audit.get("same_scope_passed") is True,
        "attached_validated_packets": attached_validated,
        "attached_non_validating_packets": attached_diagnostic,
        "missing_required_packets": missing_required,
        "packet_status": packet_status,
        "validity_notes": {
            "claim_scope": (
                "This status names the Tier-3 verification packets needed for "
                "a production MHD numerical-fidelity claim. Missing packets "
                "keep the claim blocked; diagnostic metadata does not pass."
            ),
        },
    }
