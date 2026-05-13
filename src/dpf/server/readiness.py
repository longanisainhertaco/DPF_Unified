"""Server-facing readiness and authority payloads."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dpf.validation import (
    FIRST_PRINCIPLES_MHD_EXECUTION_MODE,
    FIRST_PRINCIPLES_MHD_MODE,
    akel_fig1_draft_digitization_packet,
    first_principles_mhd_readiness_report,
    high_fidelity_readiness_report,
    predictive_readiness_report,
    scientific_closure_digitization_status,
)
from dpf.validation.artifacts import ValidationStatus, classify_result


def api_readiness_payload(
    *,
    backend: str,
    result: dict[str, Any] | None = None,
    validation_status: str = "not_evaluated",
    validation_scope: str | None = None,
) -> dict[str, Any]:
    """Build the API-visible authority/readiness payload for a simulation."""

    status = ValidationStatus(validation_status)
    classification = classify_result(
        backend=backend,
        validation_status=status,
        reason="API status has not been promoted by accepted validation evidence",
    )
    readiness_input = dict(result or {})
    readiness_input.setdefault("backend", backend)
    readiness_input.setdefault("validation_status", status.value)
    readiness_input.setdefault(
        "result_classification",
        classification.model_dump(mode="json"),
    )
    declared_run_mode = str(
        readiness_input.get("run_mode")
        or readiness_input.get("requested_run_mode")
        or backend
    )

    predictive = asdict(predictive_readiness_report(readiness_input))
    high_fidelity = asdict(high_fidelity_readiness_report(readiness_input))
    first_principles = {}
    first_principles_energy = {}
    first_principles_startup = {}
    first_principles_neutron = {}
    if declared_run_mode == FIRST_PRINCIPLES_MHD_MODE or backend == FIRST_PRINCIPLES_MHD_MODE:
        first_principles = asdict(
            first_principles_mhd_readiness_report(
                readiness_input,
                preset_name=str(readiness_input.get("preset_name", "")),
                validation_scope=str(
                    validation_scope or readiness_input.get("validation_scope") or ""
                ),
                source_scope=str(readiness_input.get("source_scope", "")),
                source_scope_status=str(
                    readiness_input.get("source_scope_status", "")
                ),
                run_mode=FIRST_PRINCIPLES_MHD_MODE,
                execution_mode=str(
                    readiness_input.get("execution_mode")
                    or readiness_input.get("backend")
                    or FIRST_PRINCIPLES_MHD_EXECUTION_MODE
                ),
            )
        )
        first_principles_energy = dict(
            first_principles.get("energy_accounting_status", {})
        )
        first_principles_startup = dict(
            first_principles.get("startup_initialization_status", {})
        )
        first_principles_neutron = dict(
            first_principles.get("neutron_yield_authority_status", {})
        )
    digitization = scientific_closure_digitization_status([akel_fig1_draft_digitization_packet()])
    digitization_scope = str(digitization.get("validation_scope", "not_declared"))
    run_scope = str(validation_scope or readiness_input.get("validation_scope") or "not_declared")
    digitization_applies = run_scope != "not_declared" and run_scope == digitization_scope

    blockers: list[str] = []
    blockers.extend(str(item) for item in predictive.get("blockers", []))
    blockers.extend(str(item) for item in high_fidelity.get("blockers", []))
    blockers.extend(str(item) for item in first_principles.get("blockers", []))
    blockers.extend(str(item) for item in digitization.get("missing_or_failed_tasks", []))
    for task in digitization.get("task_statuses", []):
        blockers.extend(str(item) for item in task.get("missing_or_failed_checks", []))

    return {
        "validation_status": status.value,
        "result_classification": classification.model_dump(mode="json"),
        "predictive_readiness": predictive,
        "high_fidelity_readiness": high_fidelity,
        "first_principles_mhd_readiness": first_principles,
        "first_principles_energy_accounting": first_principles_energy,
        "first_principles_startup_initialization": first_principles_startup,
        "first_principles_neutron_yield_authority": first_principles_neutron,
        "digitization_status": digitization,
        "readiness_scope": {
            "scope_type": "run_status_with_global_source_queue",
            "run_validation_scope": run_scope,
            "digitization_validation_scope": digitization_scope,
            "digitization_applies_to_run": digitization_applies,
            "source_blocker_scope": (
                "run_scope" if digitization_applies else "global_source_queue"
            ),
            "source_blocker_scope_note": (
                "Akel digitization blockers apply to this declared run scope."
                if digitization_applies
                else (
                    "Akel digitization blockers are global source-closure "
                    "blockers unless the run declares the same validation scope."
                )
            ),
        },
        "source_blockers": sorted(set(blockers)),
    }
