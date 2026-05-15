"""First-principles limiter ledger helpers.

The first-principles acceptance path must make numerical repairs, engineering
guards, and fallback closures explicit. These helpers keep run artifacts small
while preserving enough provenance for readiness gates and later review.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


ACCEPTANCE_BLOCKING_CLASSIFICATIONS = {
    "acceptance_blocker",
    "debug_repair",
    "engineering_guard",
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


def finite_stats(value: object) -> dict[str, object]:
    """Return compact finite-value statistics for a scalar or array."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        number = _finite_number(value)
        if number is None:
            return {
                "count": 0,
                "finite_count": 0,
                "nonfinite_count": 0,
                "min": None,
                "max": None,
            }
        arr = np.asarray([number], dtype=float)
    flat = arr.ravel()
    if flat.size == 0:
        return {
            "count": 0,
            "finite_count": 0,
            "nonfinite_count": 0,
            "min": None,
            "max": None,
        }
    finite = np.isfinite(flat)
    finite_values = flat[finite]
    return {
        "count": int(flat.size),
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(flat.size - np.count_nonzero(finite)),
        "min": float(np.min(finite_values)) if finite_values.size else None,
        "max": float(np.max(finite_values)) if finite_values.size else None,
    }


def _default_acceptance_blocking(classification: str) -> bool:
    return classification in ACCEPTANCE_BLOCKING_CLASSIFICATIONS


def limiter_event(
    *,
    limiter_id: str,
    code_path: str,
    affected_field: str,
    classification: str,
    activation_count: int,
    before: object | None = None,
    after: object | None = None,
    threshold: object | None = None,
    acceptance_blocking: bool | None = None,
    justification: str = "",
) -> dict[str, object]:
    """Create one limiter event with JSON-safe metadata."""
    classification = str(classification or "engineering_guard")
    event: dict[str, object] = {
        "limiter_id": str(limiter_id),
        "code_path": str(code_path),
        "affected_field": str(affected_field),
        "classification": classification,
        "activation_count": int(max(activation_count, 0)),
        "acceptance_blocking": (
            _default_acceptance_blocking(classification)
            if acceptance_blocking is None
            else bool(acceptance_blocking)
        ),
        "justification": str(justification),
    }
    if before is not None:
        event["before"] = finite_stats(before)
    if after is not None:
        event["after"] = finite_stats(after)
    if threshold is not None:
        event["threshold"] = threshold
    return event


def _merge_stats(existing: object, incoming: object) -> dict[str, object]:
    if not isinstance(existing, Mapping):
        existing = {}
    if not isinstance(incoming, Mapping):
        incoming = {}
    count = int(existing.get("count") or 0) + int(incoming.get("count") or 0)
    finite_count = int(existing.get("finite_count") or 0) + int(
        incoming.get("finite_count") or 0
    )
    nonfinite_count = int(existing.get("nonfinite_count") or 0) + int(
        incoming.get("nonfinite_count") or 0
    )
    mins = [
        value
        for value in (
            _finite_number(existing.get("min")),
            _finite_number(incoming.get("min")),
        )
        if value is not None
    ]
    maxes = [
        value
        for value in (
            _finite_number(existing.get("max")),
            _finite_number(incoming.get("max")),
        )
        if value is not None
    ]
    return {
        "count": count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "min": min(mins) if mins else None,
        "max": max(maxes) if maxes else None,
    }


def summarize_limiter_ledger(
    events: Iterable[Mapping[str, object]],
    *,
    source: str = "run_result",
) -> dict[str, object]:
    """Aggregate raw limiter events into a compact run-scoped ledger."""
    by_id: dict[str, dict[str, object]] = {}
    raw_count = 0
    for event in events:
        raw_count += 1
        limiter_id = str(event.get("limiter_id") or "unknown_limiter")
        record = by_id.get(limiter_id)
        if record is None:
            record = {
                "limiter_id": limiter_id,
                "code_path": str(event.get("code_path") or ""),
                "affected_field": str(event.get("affected_field") or ""),
                "classification": str(event.get("classification") or "engineering_guard"),
                "activation_count": 0,
                "activation_events": 0,
                "acceptance_blocking": bool(event.get("acceptance_blocking", True)),
                "justification": str(event.get("justification") or ""),
            }
            if "threshold" in event:
                record["threshold"] = event["threshold"]
            by_id[limiter_id] = record
        activation_count = int(_finite_number(event.get("activation_count")) or 0)
        record["activation_count"] = int(record["activation_count"]) + activation_count
        if activation_count > 0:
            record["activation_events"] = int(record["activation_events"]) + 1
        if "before" in event:
            record["before"] = _merge_stats(record.get("before"), event.get("before"))
        if "after" in event:
            record["after"] = _merge_stats(record.get("after"), event.get("after"))

    entries = sorted(by_id.values(), key=lambda item: str(item.get("limiter_id")))
    activation_count = sum(int(item.get("activation_count") or 0) for item in entries)
    active_acceptance_blockers = [
        str(item.get("limiter_id"))
        for item in entries
        if bool(item.get("acceptance_blocking"))
        and int(item.get("activation_count") or 0) > 0
    ]
    acceptance_blocking_activation_count = sum(
        int(item.get("activation_count") or 0)
        for item in entries
        if bool(item.get("acceptance_blocking"))
    )
    by_classification: dict[str, int] = {}
    for item in entries:
        key = str(item.get("classification") or "unknown")
        by_classification[key] = by_classification.get(key, 0) + int(
            item.get("activation_count") or 0
        )
    status = "blocked" if active_acceptance_blockers else "clear"
    return {
        "schema": "dpf.first_principles.limiter_ledger.v1",
        "source": source,
        "status": status,
        "validation_status": status,
        "event_count": raw_count,
        "entry_count": len(entries),
        "activation_count": activation_count,
        "acceptance_blocking_activation_count": acceptance_blocking_activation_count,
        "activated_acceptance_blockers": active_acceptance_blockers,
        "by_classification": by_classification,
        "can_support_first_principles_acceptance": not active_acceptance_blockers,
        "entries": entries,
    }


def _legacy_engineering_limiter_events(
    limiter: Mapping[str, object],
) -> list[dict[str, object]]:
    counts = limiter.get("counts")
    if not isinstance(counts, Mapping):
        return []
    events: list[dict[str, object]] = []
    for key, value in counts.items():
        count = int(_finite_number(value) or 0)
        if count <= 0:
            continue
        events.append(
            limiter_event(
                limiter_id=f"legacy.first_principles_engineering_limiter.{key}",
                code_path="app_mhd._apply_first_principles_engineering_bounds",
                affected_field=str(key),
                classification="acceptance_blocker",
                activation_count=count,
                acceptance_blocking=True,
                justification=(
                    "Legacy app-level first-principles engineering limiter "
                    "activated before FP-2 ledger instrumentation."
                ),
            )
        )
    return events


def first_principles_limiter_status(
    result: Mapping[str, object],
) -> dict[str, object]:
    """Return fail-closed limiter status from a run result."""
    ledger = result.get("first_principles_limiter_ledger")
    if isinstance(ledger, Mapping):
        entries = ledger.get("entries")
        active = list(ledger.get("activated_acceptance_blockers") or [])
        return {
            "status": str(
                ledger.get("status")
                or ledger.get("validation_status")
                or "blocked"
            ),
            "validation_status": str(
                ledger.get("validation_status") or ledger.get("status") or "blocked"
            ),
            "ledger_present": True,
            "can_support_first_principles_acceptance": bool(
                ledger.get("can_support_first_principles_acceptance")
            ),
            "activation_count": int(_finite_number(ledger.get("activation_count")) or 0),
            "acceptance_blocking_activation_count": int(
                _finite_number(ledger.get("acceptance_blocking_activation_count")) or 0
            ),
            "activated_acceptance_blockers": [str(item) for item in active],
            "entry_count": int(_finite_number(ledger.get("entry_count")) or 0),
            "entries": list(entries) if isinstance(entries, list) else [],
        }
    legacy = result.get("first_principles_engineering_limiter")
    if isinstance(legacy, Mapping):
        legacy_events = _legacy_engineering_limiter_events(legacy)
        if not legacy_events:
            return {
                "status": "missing",
                "validation_status": "missing_limiter_ledger",
                "ledger_present": False,
                "legacy_limiter_counts_present": isinstance(
                    legacy.get("counts"),
                    Mapping,
                ),
                "can_support_first_principles_acceptance": False,
                "activation_count": 0,
                "acceptance_blocking_activation_count": 0,
                "activated_acceptance_blockers": [],
                "entry_count": 0,
                "entries": [],
            }
        legacy_ledger = summarize_limiter_ledger(
            legacy_events,
            source="legacy_first_principles_engineering_limiter",
        )
        active = list(legacy_ledger.get("activated_acceptance_blockers") or [])
        return {
            "status": legacy_ledger["status"],
            "validation_status": legacy_ledger["validation_status"],
            "ledger_present": False,
            "legacy_limiter_counts_present": True,
            "can_support_first_principles_acceptance": bool(
                legacy_ledger["can_support_first_principles_acceptance"]
            ),
            "activation_count": int(legacy_ledger["activation_count"]),
            "acceptance_blocking_activation_count": int(
                legacy_ledger["acceptance_blocking_activation_count"]
            ),
            "activated_acceptance_blockers": [str(item) for item in active],
            "entry_count": int(legacy_ledger["entry_count"]),
            "entries": list(legacy_ledger.get("entries") or []),
        }
    return {
        "status": "missing",
        "validation_status": "missing_limiter_ledger",
        "ledger_present": False,
        "can_support_first_principles_acceptance": False,
        "activation_count": 0,
        "acceptance_blocking_activation_count": 0,
        "activated_acceptance_blockers": [],
        "entry_count": 0,
        "entries": [],
    }
