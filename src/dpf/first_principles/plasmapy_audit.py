"""Optional PlasmaPy community-formulary cross-checks.

PlasmaPy is useful as an independently maintained, community package for
formula and unit sanity checks.  This module deliberately keeps PlasmaPy out of
the accepted source-of-truth path: local ``KnowledgeReference`` evidence remains
the scientific authority for first-principles claims.
"""

from __future__ import annotations

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
        return {
            "status": "community_formula_audit_unavailable_optional_dependency",
            "role": PLASMAPY_AUDIT_ROLE,
            "dependency": "plasmapy",
            "install_extra": PLASMAPY_OPTIONAL_EXTRA,
            "python_requires": PLASMAPY_CURRENT_PYTHON_REQUIRES,
            "reason": str(exc),
            "reference_state": state,
            "quantities": {},
            "source_references": list(PLASMAPY_AUDIT_SOURCE_REFS),
            "source_truth_policy": _source_truth_policy(),
            "docs": list(PLASMAPY_DOC_REFS),
            "can_support_first_principles_acceptance": False,
        }

    local = _local_quantities(state)
    community: dict[str, dict[str, Any]] = {}
    community["coulomb_log"] = _plasmapy_coulomb_log(formulary, units, state)
    community["debye_length_m"] = _plasmapy_debye_length(formulary, units, state)
    community["alfven_speed_m_s"] = _plasmapy_alfven_speed(formulary, units, state)
    community["electron_gyrofrequency_rad_s"] = _plasmapy_gyrofrequency(
        formulary,
        units,
        state,
    )

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
