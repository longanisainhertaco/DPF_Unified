"""Canonical per-channel acceptance states for first-principles packets.

Sprint 8 Workstream 1 (Codex audit S7-A7/S7-A8).  Before this module the
same-scope, numerical-fidelity, and certificate packet builders all used a
contradictory pattern: a channel could be reported as ``accepted`` while the
same channel was unconditionally re-added to ``missing_acceptance_channels``::

    missing = set(REQUIRED_CHANNELS) - accepted
    missing.update(REQUIRED_CHANNELS)   # re-adds EVERY channel, even accepted

This is safe today only because top-level acceptance is hard-coded False, but
it makes coherent acceptance impossible and misleads any reviewer or audit
tool that reads the packet.

This module replaces that pattern with a single canonical vocabulary of seven
mutually exclusive per-channel states, shared by ``same_scope``,
``numerical_fidelity``, ``certificate_gate``, ``runner``, and ``manifest`` so
every packet agrees on what a channel's state means.

State meanings:

- ``accepted`` -- channel has reviewed, same-scope, uncertainty-bounded
  evidence.  This is the ONLY state that may count toward acceptance.
- ``blocked_missing_source`` -- no source packet / evidence is available.
- ``blocked_wrong_scope`` -- evidence exists but for a different
  device/shot/configuration scope (no reviewed transfer rule).
- ``blocked_missing_review`` -- evidence exists but lacks an independent
  review certificate.
- ``blocked_missing_uncertainty`` -- evidence exists and is reviewed but has
  no uncertainty budget.
- ``excluded_not_validated`` -- channel is deliberately excluded from the
  claim (e.g. reduced-model output, candidate telemetry, manual request not
  backed by a reviewed target).  An excluded channel NEVER counts as
  comparator evidence.
- ``not_claimed`` -- channel is not part of the current claim and no evidence
  was offered.

A channel is in exactly one state.  ``accepted`` and any ``blocked_*`` /
``excluded_*`` / ``not_claimed`` state are mutually exclusive -- this is what
makes "a channel cannot be both accepted and missing" structurally true.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

__all__ = [
    "ChannelState",
    "ACCEPTED",
    "BLOCKED_MISSING_SOURCE",
    "BLOCKED_WRONG_SCOPE",
    "BLOCKED_MISSING_REVIEW",
    "BLOCKED_MISSING_UNCERTAINTY",
    "EXCLUDED_NOT_VALIDATED",
    "NOT_CLAIMED",
    "CHANNEL_STATE_VALUES",
    "ACCEPTING_CHANNEL_STATES",
    "is_accepted",
    "blocks_acceptance",
    "counts_as_comparator_evidence",
    "channel_state_map",
    "missing_channels",
    "accepted_channels",
    "channel_state_summary",
    "all_states_canonical",
]


class ChannelState(StrEnum):
    """The seven canonical per-channel acceptance states.

    A :class:`~enum.StrEnum`, so a state serializes to its plain string value
    in JSON payloads without a custom encoder, and ``packet["state"] ==
    "accepted"`` style assertions keep working.
    """

    ACCEPTED = "accepted"
    BLOCKED_MISSING_SOURCE = "blocked_missing_source"
    BLOCKED_WRONG_SCOPE = "blocked_wrong_scope"
    BLOCKED_MISSING_REVIEW = "blocked_missing_review"
    BLOCKED_MISSING_UNCERTAINTY = "blocked_missing_uncertainty"
    EXCLUDED_NOT_VALIDATED = "excluded_not_validated"
    NOT_CLAIMED = "not_claimed"


# Module-level constants for ergonomic, import-once use by packet builders.
ACCEPTED = ChannelState.ACCEPTED
BLOCKED_MISSING_SOURCE = ChannelState.BLOCKED_MISSING_SOURCE
BLOCKED_WRONG_SCOPE = ChannelState.BLOCKED_WRONG_SCOPE
BLOCKED_MISSING_REVIEW = ChannelState.BLOCKED_MISSING_REVIEW
BLOCKED_MISSING_UNCERTAINTY = ChannelState.BLOCKED_MISSING_UNCERTAINTY
EXCLUDED_NOT_VALIDATED = ChannelState.EXCLUDED_NOT_VALIDATED
NOT_CLAIMED = ChannelState.NOT_CLAIMED

# Ordered tuple of the seven canonical string values -- the shared vocabulary
# that contract tests assert runner / CLI / manifest / certificate agree on.
CHANNEL_STATE_VALUES: tuple[str, ...] = tuple(state.value for state in ChannelState)

# Only ``accepted`` may count toward acceptance.  Kept as a set so future
# states (if any) can be added here deliberately rather than by accident.
ACCEPTING_CHANNEL_STATES: frozenset[ChannelState] = frozenset({ChannelState.ACCEPTED})

# Blocked-family states: evidence is missing or insufficient.
_BLOCKED_STATES: frozenset[ChannelState] = frozenset(
    {
        ChannelState.BLOCKED_MISSING_SOURCE,
        ChannelState.BLOCKED_WRONG_SCOPE,
        ChannelState.BLOCKED_MISSING_REVIEW,
        ChannelState.BLOCKED_MISSING_UNCERTAINTY,
    }
)


def _coerce(state: ChannelState | str) -> ChannelState:
    """Return ``state`` as a :class:`ChannelState`, raising on unknown values."""

    if isinstance(state, ChannelState):
        return state
    try:
        return ChannelState(str(state))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown channel state {state!r}; expected one of {CHANNEL_STATE_VALUES}"
        ) from exc


def is_accepted(state: ChannelState | str) -> bool:
    """True only for the ``accepted`` state.

    This is the single decision point: every packet builder asks this -- not a
    string prefix check -- so "accepted" cannot be confused with
    "accepted_engineering_review" or any near-miss label.
    """

    return _coerce(state) in ACCEPTING_CHANNEL_STATES


def blocks_acceptance(state: ChannelState | str) -> bool:
    """True for every state that is not ``accepted``.

    A claimed channel that is not ``accepted`` blocks acceptance.  Excluded and
    not-claimed channels also return True here -- a packet may only accept when
    every *claimed* channel is ``accepted`` and every *excluded* channel is
    explicitly excluded (see :func:`channel_state_summary`).
    """

    return _coerce(state) is not ChannelState.ACCEPTED


def counts_as_comparator_evidence(state: ChannelState | str) -> bool:
    """True only when a channel may be used as comparator evidence.

    ``excluded_not_validated`` and ``not_claimed`` channels NEVER count as
    comparator evidence, and neither does any ``blocked_*`` channel.  Only an
    ``accepted`` channel does.  Exit-criteria invariant for S8-WS1.
    """

    return _coerce(state) is ChannelState.ACCEPTED


def channel_state_map(
    states: Mapping[str, ChannelState | str],
) -> dict[str, str]:
    """Return a JSON-safe ``{channel: state_value}`` mapping.

    Validates every value against the canonical vocabulary.  Use this when a
    packet builder publishes its per-channel states so the published map can
    never contain a non-canonical state string.
    """

    return {str(name): _coerce(state).value for name, state in states.items()}


def missing_channels(
    states: Mapping[str, ChannelState | str],
) -> list[str]:
    """Return the sorted channels that are not ``accepted``.

    This is the canonical, contradiction-free replacement for the old
    ``missing_acceptance_channels`` computation: a channel appears here only
    when its state is not ``accepted``, so an ``accepted`` channel can never
    also be reported missing.
    """

    return sorted(
        name for name, state in states.items() if blocks_acceptance(state)
    )


def accepted_channels(
    states: Mapping[str, ChannelState | str],
) -> list[str]:
    """Return the sorted channels whose state is ``accepted``."""

    return sorted(name for name, state in states.items() if is_accepted(state))


def channel_state_summary(
    states: Mapping[str, ChannelState | str],
    *,
    claimed_channels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a coherence summary over a per-channel state map.

    ``claimed_channels`` is the set of channels the packet asserts as part of
    its claim.  When omitted, every channel that is not ``excluded_not_validated``
    or ``not_claimed`` is treated as claimed.

    The summary's ``all_claimed_channels_accepted`` is True only when every
    claimed channel is ``accepted``.  ``contradictions`` is always empty by
    construction -- a channel has exactly one state -- and is reported
    explicitly so downstream packets and audits can assert it.
    """

    normalized = {str(name): _coerce(state) for name, state in states.items()}
    if claimed_channels is None:
        claimed = {
            name
            for name, state in normalized.items()
            if state
            not in {
                ChannelState.EXCLUDED_NOT_VALIDATED,
                ChannelState.NOT_CLAIMED,
            }
        }
    else:
        claimed = {str(name) for name in claimed_channels}

    excluded = sorted(
        name
        for name, state in normalized.items()
        if state is ChannelState.EXCLUDED_NOT_VALIDATED
    )
    not_claimed = sorted(
        name
        for name, state in normalized.items()
        if state is ChannelState.NOT_CLAIMED
    )
    accepted = accepted_channels(normalized)
    blocked = sorted(
        name
        for name, state in normalized.items()
        if state in _BLOCKED_STATES
    )
    claimed_not_accepted = sorted(
        name for name in claimed if not is_accepted(normalized.get(name, NOT_CLAIMED))
    )
    counts_by_state = {
        value: sum(1 for state in normalized.values() if state.value == value)
        for value in CHANNEL_STATE_VALUES
    }
    return {
        "channel_states": channel_state_map(normalized),
        "accepted_channels": accepted,
        "blocked_channels": blocked,
        "excluded_channels": excluded,
        "not_claimed_channels": not_claimed,
        "missing_acceptance_channels": missing_channels(normalized),
        "claimed_channels": sorted(claimed),
        "claimed_not_accepted_channels": claimed_not_accepted,
        "all_claimed_channels_accepted": not claimed_not_accepted,
        "counts_by_state": counts_by_state,
        # By construction a channel maps to exactly one state, so an
        # accepted+missing contradiction is structurally impossible.  Reported
        # explicitly so a contract test can assert it stays empty.
        "contradictions": [],
        "schema": "first_principles_channel_state_v1",
    }


def all_states_canonical(values: Iterable[str]) -> bool:
    """True when every value in ``values`` is a canonical channel-state string.

    Used by contract tests to prove that runner, CLI, manifest, and certificate
    packets only ever emit states from the shared vocabulary.
    """

    return all(str(value) in CHANNEL_STATE_VALUES for value in values)
