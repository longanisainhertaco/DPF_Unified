"""Server-facing readiness and authority payloads."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dpf.validation import (
    akel_fig1_draft_digitization_packet,
    high_fidelity_readiness_report,
    predictive_readiness_report,
    scientific_closure_digitization_status,
)
from dpf.validation.artifacts import ValidationStatus, classify_result

FIRST_PRINCIPLES_MHD_MODE = "first_principles_mhd"


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
        first_principles, first_principles_energy, first_principles_startup, first_principles_neutron = (
            _package_native_first_principles_readiness(
                validation_scope=str(
                    validation_scope
                    or readiness_input.get("validation_scope")
                    or "not_declared"
                ),
                source_scope=str(
                    readiness_input.get("source_scope")
                    or validation_scope
                    or readiness_input.get("validation_scope")
                    or "not_declared"
                ),
            )
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


# Declared-scope -> runtime deck resolution.
#
# S10-A2 (closes the Akel/full-energy API mixing path): the readiness layer
# previously ALWAYS ran the Akel 16 kV seed-layer deck and then stamped the
# caller-supplied validation/source scope onto the resulting payload.  A
# PF-1000 full-energy request could therefore be labelled full-energy while the
# startup packet came from the Akel deck.  The map below pins each declared
# scope to its own runtime deck.  A scope that is not in this map (including
# ``not_declared``/unknown) MUST NOT silently default to any deck — it maps to
# a fail-closed blocked readiness.
#
# SS11-A2 (closes the residual partial-pair mixing surface): the SS11 resolver
# matched on a TOKEN SET — ``{validation_scope, source_scope} & deck_tokens`` —
# so a full-energy validation scope paired with an Akel-like (or unknown, or
# ``not_declared``) source scope still overlapped one full-energy token,
# resolved to the full-energy deck, echoed the foreign source label, and
# stamped ``scope_match=True``.  The resolver below instead matches the ORDERED
# (validation_scope, source_scope) pair against the canonical accepted pairs.
# Only an EXACT pair resolves to a deck; every partial pair, unknown pair,
# undeclared pair, startup-source label, or mixed pair resolves to ``None`` and
# fails closed.  ``scope_match`` is then computed from exact requested-vs-actual
# pair equality, never from "a resolved family exists".

# Akel PF-1000 16 kV shot-12581 engineering scope labels.
_AKEL_VALIDATION_SCOPE = "pf1000_16kv_2021_akel"
_AKEL_SOURCE_SCOPE = "pf1000_16kv_2021_akel_shot12581"

# PF-1000 Scholz 2000/2001 24-rod large-electrode full-energy scope labels.
_FULL_ENERGY_VALIDATION_SCOPE = "pf1000_full_energy_27_to_40_kv"
_FULL_ENERGY_SOURCE_SCOPE = (
    "pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source"
)


# Canonical accepted (validation_scope, source_scope) pairs.  A runtime deck is
# resolved ONLY when the ordered requested pair is identical to one of these.
_AKEL_SCOPE_PAIR = (_AKEL_VALIDATION_SCOPE, _AKEL_SOURCE_SCOPE)
_FULL_ENERGY_SCOPE_PAIR = (
    _FULL_ENERGY_VALIDATION_SCOPE,
    _FULL_ENERGY_SOURCE_SCOPE,
)


def _resolve_runtime_deck_scope(
    *,
    validation_scope: str,
    source_scope: str,
) -> str | None:
    """Resolve a declared scope/source PAIR to a runtime deck identity.

    Returns ``"akel"`` only when the ordered ``(validation_scope,
    source_scope)`` pair is EXACTLY the canonical Akel 16 kV pair, and
    ``"full_energy"`` only when it is EXACTLY the canonical PF-1000 full-energy
    pair.  Every other case — a partial pair (one scope canonical, the other
    not), an unknown pair, an undeclared/``not_declared`` pair, a startup-source
    label, or a mixed pair (validation from one family, source from another) —
    returns ``None``.

    ``None`` MUST map to a fail-closed blocked readiness (SS11-A2): the
    readiness layer never resolves a deck from a single overlapping token, so a
    foreign or undeclared source label can no longer ride a deck whose scope
    contradicts the request.
    """

    requested_pair = (str(validation_scope), str(source_scope))
    if requested_pair == _AKEL_SCOPE_PAIR:
        return "akel"
    if requested_pair == _FULL_ENERGY_SCOPE_PAIR:
        return "full_energy"
    return None


def _scope_blocked_first_principles_readiness(
    *,
    validation_scope: str,
    source_scope: str,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return a fail-closed blocked readiness with no runtime deck executed.

    Used when the declared scope/source cannot be resolved to a runtime deck.
    No deck is run, so no deck's scope can be mis-stamped onto the request.
    """

    from dpf.first_principles.runner import RUN_MODE

    readiness = {
        "ready": False,
        "status": "blocked",
        "run_mode": FIRST_PRINCIPLES_MHD_MODE,
        "execution_mode": RUN_MODE,
        "requested_validation_scope": validation_scope,
        "requested_source_scope": source_scope,
        "actual_runtime_validation_scope": "not_run",
        "actual_runtime_source_scope": "not_run",
        "runtime_deck_id": "not_run",
        "scope_match": False,
        # Legacy keys retained for payload-shape stability; they echo the
        # request and never a runtime deck's own scope.
        "source_scope": source_scope,
        "validation_scope": validation_scope,
        "package_native_runner": RUN_MODE,
        "scientific_status": "blocked_unresolved_runtime_scope",
        "missing_evidence": ["resolved_runtime_deck_scope"],
        "blockers": [reason],
        "can_support_first_principles_acceptance": False,
    }
    return readiness, {}, {}, {}


def _package_native_first_principles_readiness(
    *,
    validation_scope: str,
    source_scope: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from dpf.first_principles import (
        pf1000_akel_16kv_engineering_deck,
        pf1000_scholz_2001_24rod_full_energy_deck,
    )
    from dpf.first_principles.runner import RUN_MODE, run_first_principles_3d_deck

    resolved = _resolve_runtime_deck_scope(
        validation_scope=validation_scope,
        source_scope=source_scope,
    )
    if resolved is None:
        return _scope_blocked_first_principles_readiness(
            validation_scope=validation_scope,
            source_scope=source_scope,
            reason=(
                "first-principles readiness blocked: declared validation/"
                f"source scope ('{validation_scope}'/'{source_scope}') does "
                "not resolve to a known runtime deck; readiness fails closed "
                "rather than run a deck whose scope contradicts the request"
            ),
        )

    # Resolve the runtime deck strictly from the declared scope.  A PF-1000
    # full-energy request runs the full-energy deck; an Akel request runs the
    # Akel deck.  ``n_steps=1`` — readiness needs only the validation_packet /
    # telemetry structure, not a physics run.
    if resolved == "full_energy":
        deck = pf1000_scholz_2001_24rod_full_energy_deck(n_steps=1)
    else:
        deck = pf1000_akel_16kv_engineering_deck(n_steps=1)

    run = run_first_principles_3d_deck(deck)
    validation_packet = dict(run.validation_packet)
    telemetry = run.telemetry
    startup = dict(telemetry.get("startup", {}))
    neutron = dict(telemetry.get("neutron_authority", {}))
    energy = dict(run.conservation_telemetry)
    missing_evidence = [
        key
        for key in (
            "startup_bvp",
            "power_port",
            "dimensionality_handoff",
            "physics_closure",
            "limiter_readiness",
            "same_scope_source",
            "waveform_phase",
            "spatial_field_temperature",
            "neutron_authority",
            "comparator_uq",
            "numerical_fidelity",
            "certificate_gate",
            "generalization",
        )
        if validation_packet.get(f"{key}_status")
    ]

    # ``actual_runtime_*`` report the runtime DECK PAIR identity — the deck that
    # was actually executed (Akel vs full-energy), derived from ``resolved``,
    # never from the caller-supplied request.
    actual_runtime_validation_scope = (
        _FULL_ENERGY_VALIDATION_SCOPE
        if resolved == "full_energy"
        else _AKEL_VALIDATION_SCOPE
    )
    actual_runtime_source_scope = (
        _FULL_ENERGY_SOURCE_SCOPE
        if resolved == "full_energy"
        else _AKEL_SOURCE_SCOPE
    )
    # SS11-A2: ``scope_match`` is COMPUTED from exact requested-vs-actual PAIR
    # equality — both the validation scope AND the source scope must be
    # identical — never hard-coded ``True`` and never "a resolved family
    # exists".  ``resolved`` already required the requested pair to be an exact
    # canonical pair, so a request reaching this deck-execution path matches by
    # construction; computing the flag here keeps the contract explicit and
    # fails closed if either runtime-vs-request scope ever diverges.
    scope_match = (
        str(validation_scope) == actual_runtime_validation_scope
        and str(source_scope) == actual_runtime_source_scope
    )
    # The deck-internal runtime scope as the runner reports it — surfaced for
    # traceability so a reviewer can confirm which deck telemetry was consumed.
    runtime_deck_internal_source_scope = str(
        validation_packet.get("source_scope")
        or startup.get("source_scope")
        or "unknown_runtime_source_scope"
    )

    readiness = {
        "ready": False,
        "status": "blocked",
        "run_mode": FIRST_PRINCIPLES_MHD_MODE,
        "execution_mode": RUN_MODE,
        # Both the requested scope and the actual runtime deck scope are
        # exposed (S10-A2): a reviewer can always see which deck ran versus
        # what the caller asked for.
        "requested_validation_scope": validation_scope,
        "requested_source_scope": source_scope,
        "actual_runtime_validation_scope": actual_runtime_validation_scope,
        "actual_runtime_source_scope": actual_runtime_source_scope,
        "runtime_deck_id": deck.deck_id,
        "runtime_deck_internal_source_scope": runtime_deck_internal_source_scope,
        # Computed from exact requested-vs-actual PAIR equality (SS11-A2).
        "scope_match": scope_match,
        # Legacy keys preserved for the existing Akel readiness contract.  They
        # echo the requested scope, which is now SAFE because the runtime deck
        # was selected from that same request — a requested label can no longer
        # ride a different runtime deck.
        "source_scope": source_scope,
        "validation_scope": validation_scope,
        "package_native_runner": RUN_MODE,
        "scientific_status": validation_packet.get("scientific_status"),
        "missing_evidence": missing_evidence,
        "blockers": list(validation_packet.get("blocking_reasons", ())),
        "can_support_first_principles_acceptance": False,
    }
    return readiness, energy, startup, neutron
