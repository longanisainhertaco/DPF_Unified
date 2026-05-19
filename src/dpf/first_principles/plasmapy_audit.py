"""Optional PlasmaPy community-formulary cross-checks.

PlasmaPy is useful as an independently maintained, community package for
formula and unit sanity checks.  This module deliberately keeps PlasmaPy out of
the accepted source-of-truth path: local ``KnowledgeReference`` evidence remains
the scientific authority for first-principles claims.

S3.5 PlasmaPy rule (WP-N5 closure-registry source audit, section 5.3):
PlasmaPy is a cross-check ONLY.  A missing PlasmaPy audit cannot promote or
reject a local-source closure.  A disagreement outside tolerance sets
review-required telemetry.  The strong-coupling Coulomb-logarithm regime --
where PlasmaPy raises a ``CouplingWarning`` -- is captured as a SURFACED
``bounded_out_with_source`` signal, never swallowed silently.  The NRL Plasma
Formulary confirms this validity edge: classical transport needs ``lambda >> 1``
and the Coulomb-log theory fails when ``lambda ~ 1``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from io import StringIO
from typing import Any

import numpy as np

from dpf.collision.spitzer import coulomb_log
from dpf.constants import e, epsilon_0, k_B, m_d, m_e, mu_0

PLASMAPY_AUDIT_ROLE = "optional_community_formula_cross_check_not_source_authority"
PLASMAPY_OPTIONAL_EXTRA = "dpf-unified[audit]"
PLASMAPY_CURRENT_PYTHON_REQUIRES = ">=3.12"
PLASMAPY_AUDIT_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "role": "local_formula_authority_for_transport_and_basic_plasma_scales",
    },
    {
        "path": "KnowledgeReference/plasma-formulary.md",
        "role": "local_formula_authority_for_radiation_and_plasma_parameter_checks",
    },
)
PLASMAPY_DOC_REFS = (
    {
        "url": "https://docs.plasmapy.org/en/stable/api_static/plasmapy.html",
        "role": "plasmapy_package_scope",
    },
    {
        "url": "https://docs.plasmapy.org/en/stable/formulary/index.html",
        "role": "plasmapy_formulary_scope",
    },
)

# NRL Plasma Formulary citations for the classical-transport validity edge.
# These back the strong-coupling regime gate (S3.5 / WP-N5 section 5.3): the
# classical Coulomb-logarithm transport closure is out of validity where the
# Coulomb logarithm is small / the plasma is strongly coupled.
STRONG_COUPLING_REGIME_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "lines": "3036-3038",
        "equation": "coulomb-log-validity-edge",
        "role": "coulomb_log_theory_good_to_10_percent_and_fails_when_lambda_near_1",
    },
    {
        "path": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "lines": "3379-3383",
        "equation": "classical-transport-validity-criteria-3-5-6",
        "role": "classical_transport_valid_only_when_coulomb_log_lambda_much_gt_1",
    },
)

# WP-N5 section 5.3: treat ln Lambda <= ~2 as the out-of-weak-coupling trigger.
# This mirrors the in-code coulomb_log floor at 2.0 ("Spitzer theory invalid
# below this", spitzer.py) -- but here it is a SURFACED flag, not a floor.
WEAK_COUPLING_COULOMB_LOG_THRESHOLD = 2.0


def build_plasmapy_formulary_audit_packet(
    reference_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-promoting PlasmaPy formula cross-check packet.

    The packet is allowed to be absent or partial.  It is engineering telemetry
    only, intended to catch convention/unit mistakes in our local implementations.
    """

    state = _reference_state(reference_state)
    try:
        formulary, units = _import_plasmapy_dependencies()
    except ImportError as exc:
        # A missing PlasmaPy audit cannot promote or reject a closure. The
        # strong-coupling regime is still surfaced from the LOCAL Coulomb log
        # so a strongly coupled reference state is never silently accepted.
        local_only = _local_quantities(state)
        local_coulomb = {
            "local_value": local_only["coulomb_log"],
            "plasmapy_value": None,
        }
        return {
            "status": "community_formula_audit_unavailable_optional_dependency",
            "role": PLASMAPY_AUDIT_ROLE,
            "dependency": "plasmapy",
            "install_extra": PLASMAPY_OPTIONAL_EXTRA,
            "python_requires": PLASMAPY_CURRENT_PYTHON_REQUIRES,
            "reason": str(exc),
            "reference_state": state,
            "quantities": {},
            "strong_coupling_regime": _strong_coupling_regime(
                state,
                local_coulomb,
                coupling_warning_raised=False,
            ),
            "source_references": list(PLASMAPY_AUDIT_SOURCE_REFS),
            "source_truth_policy": _source_truth_policy(),
            "docs": list(PLASMAPY_DOC_REFS),
            "can_support_first_principles_acceptance": False,
        }

    local = _local_quantities(state)
    community: dict[str, dict[str, Any]] = {}
    # Capture any PlasmaPy CouplingWarning rather than swallowing it: the
    # strong-coupling warning is physically meaningful for the dense DPF pinch
    # core and is the trigger for the bounded-out-with-source regime gate.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        community["coulomb_log"] = _plasmapy_coulomb_log(formulary, units, state)
        community["debye_length_m"] = _plasmapy_debye_length(formulary, units, state)
        community["alfven_speed_m_s"] = _plasmapy_alfven_speed(
            formulary,
            units,
            state,
        )
        community["electron_gyrofrequency_rad_s"] = _plasmapy_gyrofrequency(
            formulary,
            units,
            state,
        )
    coupling_warning_raised = _coupling_warning_raised(caught_warnings)

    quantities: dict[str, dict[str, Any]] = {}
    error_count = 0
    checked_count = 0
    for name, local_value in local.items():
        record = community[name]
        if record.get("status") != "computed":
            error_count += 1
            quantities[name] = {
                "status": "plasmapy_quantity_unavailable",
                "local_value": local_value,
                "plasmapy_value": None,
                "relative_difference": None,
                "error": record.get("error"),
            }
            continue
        checked_count += 1
        plasma_value = float(record["value"])
        quantities[name] = {
            "status": _agreement_status(name, local_value, plasma_value),
            "local_value": local_value,
            "plasmapy_value": plasma_value,
            "relative_difference": _relative_difference(local_value, plasma_value),
        }

    status = (
        "community_formula_audit_executed_not_authority"
        if checked_count and not error_count
        else "community_formula_audit_partial_not_authority"
        if checked_count
        else "community_formula_audit_failed_not_authority"
    )
    return {
        "status": status,
        "role": PLASMAPY_AUDIT_ROLE,
        "dependency": "plasmapy",
        "install_extra": PLASMAPY_OPTIONAL_EXTRA,
        "python_requires": PLASMAPY_CURRENT_PYTHON_REQUIRES,
        "reference_state": state,
        "quantities": quantities,
        "checked_quantity_count": checked_count,
        "error_count": error_count,
        "strong_coupling_regime": _strong_coupling_regime(
            state,
            quantities.get("coulomb_log", {}),
            coupling_warning_raised=coupling_warning_raised,
        ),
        "source_references": list(PLASMAPY_AUDIT_SOURCE_REFS),
        "source_truth_policy": _source_truth_policy(),
        "docs": list(PLASMAPY_DOC_REFS),
        "can_support_first_principles_acceptance": False,
    }


def _import_plasmapy_dependencies() -> tuple[Any, Any]:
    try:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            return import_module("plasmapy.formulary"), import_module("astropy.units")
    except ImportError as exc:
        raise ImportError("install optional extra dpf-unified[audit] to enable") from exc


def _reference_state(reference_state: Mapping[str, Any] | None) -> dict[str, float | str]:
    value = dict(reference_state or {})
    ne = float(value.get("electron_density_m3", 1.0e22))
    Te = float(value.get("electron_temperature_K", 1.0e6))
    B = float(value.get("magnetic_field_T", 10.0))
    rho = float(value.get("mass_density_kg_m3", ne * m_d))
    ion = str(value.get("ion", "D+"))
    if ne <= 0.0:
        raise ValueError("electron_density_m3 must be positive")
    if Te <= 0.0:
        raise ValueError("electron_temperature_K must be positive")
    if rho <= 0.0:
        raise ValueError("mass_density_kg_m3 must be positive")
    return {
        "electron_density_m3": ne,
        "electron_temperature_K": Te,
        "electron_temperature_eV": float(k_B * Te / e),
        "magnetic_field_T": abs(B),
        "mass_density_kg_m3": rho,
        "ion": ion,
    }


def _local_quantities(state: Mapping[str, float | str]) -> dict[str, float]:
    ne = float(state["electron_density_m3"])
    Te = float(state["electron_temperature_K"])
    B = float(state["magnetic_field_T"])
    rho = float(state["mass_density_kg_m3"])
    local_ln = float(coulomb_log(np.array([ne]), np.array([Te]))[0])
    return {
        "coulomb_log": local_ln,
        "debye_length_m": float(np.sqrt(epsilon_0 * k_B * Te / (ne * e * e))),
        "alfven_speed_m_s": float(B / np.sqrt(mu_0 * rho)),
        "electron_gyrofrequency_rad_s": float(e * B / m_e),
    }


def _plasmapy_coulomb_log(formulary: Any, units: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = formulary.Coulomb_logarithm(
            float(state["electron_temperature_eV"]) * units.eV,
            float(state["electron_density_m3"]) / units.m**3,
            ("e-", str(state["ion"])),
        )
        return {"status": "computed", "value": _quantity_value(value)}
    except Exception as exc:  # pragma: no cover - API/version compatibility guard.
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _plasmapy_debye_length(formulary: Any, units: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = formulary.Debye_length(
            float(state["electron_temperature_eV"]) * units.eV,
            float(state["electron_density_m3"]) / units.m**3,
        ).to(units.m)
        return {"status": "computed", "value": _quantity_value(value)}
    except Exception as exc:  # pragma: no cover - API/version compatibility guard.
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _plasmapy_alfven_speed(formulary: Any, units: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = formulary.Alfven_speed(
            float(state["magnetic_field_T"]) * units.T,
            float(state["mass_density_kg_m3"]) * units.kg / units.m**3,
            ion=str(state["ion"]),
        ).to(units.m / units.s)
        return {"status": "computed", "value": _quantity_value(value)}
    except Exception as exc:  # pragma: no cover - API/version compatibility guard.
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _plasmapy_gyrofrequency(
    formulary: Any,
    units: Any,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = abs(
            formulary.gyrofrequency(
                float(state["magnetic_field_T"]) * units.T,
                "e-",
                signed=True,
            ).to(1 / units.s)
        )
        return {"status": "computed", "value": _quantity_value(value)}
    except Exception as exc:  # pragma: no cover - API/version compatibility guard.
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _quantity_value(value: Any) -> float:
    if hasattr(value, "value"):
        return float(value.value)
    return float(value)


def _agreement_status(name: str, local_value: float, plasma_value: float) -> str:
    rel = _relative_difference(local_value, plasma_value)
    tolerance = 0.30 if name == "coulomb_log" else 1.0e-8
    return (
        "community_formula_cross_check_within_tolerance_not_authority"
        if rel <= tolerance
        else "community_formula_cross_check_outside_tolerance_not_authority"
    )


def _relative_difference(local_value: float, plasma_value: float) -> float:
    denom = max(abs(local_value), abs(plasma_value), 1.0e-300)
    return float(abs(local_value - plasma_value) / denom)


def _coupling_warning_raised(caught_warnings: list[warnings.WarningMessage]) -> bool:
    """True if PlasmaPy raised a CouplingWarning while computing a quantity.

    PlasmaPy emits ``CouplingWarning`` when the plasma coupling parameter shows
    a strongly coupled / non-ideal regime where the weak-coupling Coulomb-log
    expansion is invalid. Detection is by class name so the audit does not hard
    depend on PlasmaPy's exception module layout across versions.
    """
    for entry in caught_warnings:
        category = getattr(entry, "category", None)
        names = {getattr(category, "__name__", "")}
        names.update(
            base.__name__ for base in getattr(category, "__mro__", ())
        )
        if "CouplingWarning" in names:
            return True
    return False


def _strong_coupling_regime(
    state: Mapping[str, Any],
    coulomb_log_quantity: Mapping[str, Any],
    *,
    coupling_warning_raised: bool,
) -> dict[str, Any]:
    """Build the surfaced strong-coupling regime signal (S3.5 / WP-N5 5.3).

    The classical Spitzer/collision transport closure is out of its validity
    range when the Coulomb logarithm is small or PlasmaPy raises a
    ``CouplingWarning``. This signal is consumed by
    :func:`dpf.first_principles.closure_packet.build_plasmapy_closure_regime_gate`
    to emit the ``bounded_out_with_source`` gate. It is SURFACED telemetry, not
    a silent floor, and it can never promote or reject a local-source closure.
    """
    local_ln = coulomb_log_quantity.get("local_value")
    plasmapy_ln = coulomb_log_quantity.get("plasmapy_value")
    coulomb_log_value = (
        float(plasmapy_ln)
        if plasmapy_ln is not None
        else float(local_ln)
        if local_ln is not None
        else None
    )
    low_coulomb_log = (
        coulomb_log_value is not None
        and coulomb_log_value <= WEAK_COUPLING_COULOMB_LOG_THRESHOLD
    )
    out_of_validity = bool(coupling_warning_raised or low_coulomb_log)
    return {
        "coupling_warning_raised": bool(coupling_warning_raised),
        "coulomb_log_value": coulomb_log_value,
        "weak_coupling_coulomb_log_threshold": WEAK_COUPLING_COULOMB_LOG_THRESHOLD,
        "low_coulomb_log": low_coulomb_log,
        "strong_coupling_out_of_validity": out_of_validity,
        "classification": (
            "bounded_out_with_source"
            if out_of_validity
            else "weak_coupling_within_classical_transport_validity"
        ),
        "regime_flag_field": "strong_coupling_out_of_validity",
        "validity_edge_source_references": list(
            STRONG_COUPLING_REGIME_SOURCE_REFS
        ),
        "warning_swallowed_silently": False,
        "is_silent_floor": False,
        "can_support_first_principles_acceptance": False,
    }


def _source_truth_policy() -> dict[str, Any]:
    return {
        "local_knowledge_reference_remains_authority": True,
        "plasmapy_can_promote_claims": False,
        "plasmapy_use": (
            "optional formula, unit, and convention cross-checks for developer "
            "audit and engineering telemetry"
        ),
        "runtime_solver_dependency": False,
    }
