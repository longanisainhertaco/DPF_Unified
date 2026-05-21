"""Report-only acceptance-gate dry run for the PF-1000 full-energy probe.

Super-Sprint 10 SS10-7.  This module is non-promoting report-only
infrastructure.  It NEVER computes physics and NEVER promotes an acceptance
flag.  It only reads the eight first-principles gate packets already emitted by
the package-native 3-D runner and re-presents them as a single fail-closed
ledger so the next physics sprint is measurable.

What it does
------------
1. Runs (or accepts a pre-built result of) the six-step PF-1000 full-energy
   engineering probe ``pf1000_scholz_2001_24rod_full_energy``.
2. Reads each gate packet's existing ``status`` / ``missing`` / blocker fields.
   It does not re-evaluate any gate -- the gate modules
   (``numerical_fidelity``, ``same_scope``, ``comparator_uq``,
   ``certificate_gate``, the conductor-mask geometry packet, ``startup_bvp``,
   ``power_port``, ``neutron_authority``) remain the sole authority.
3. Emits a typed :class:`AcceptanceGateDryRunLedger`: per gate a
   :class:`GateDryRunResult` with ``status`` (``pass``/``blocked``), a
   non-empty ``missing`` list for every blocked gate, and a short
   ``next_action``.

Fail-closed contract
--------------------
The ledger carries ``report_only=True``, ``promotes_acceptance=False``,
``accepted_runtime_claim=False`` and
``can_support_first_principles_acceptance=False``.  These are hard-coded; this
module has no code path that can set any of them True.  It is the SS10-7
report-only workstream: it runs gates and REPORTS, it never accepts.

This module is deliberately NOT exported from ``dpf.first_principles`` -- it is
a leaf reporting tool, not part of the active first-principles closure.  It is
candidate/blocked-aware report infrastructure with no KR physics authority,
mirroring ``channel_state.py``'s ``nonphysics_infrastructure`` registration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PF1000_FULL_ENERGY_DECK_PRESET = "pf1000_scholz_2001_24rod_full_energy"

# Per-gate plumbing: which telemetry packet backs the gate and the short
# next-action a reader follows when the gate is blocked.  The next-action text
# is reporting metadata only; it never relaxes the gate's own acceptance logic.
_GATE_PLAN: tuple[dict[str, str], ...] = (
    {
        "gate": "numerical_fidelity",
        "packet_key": "numerical_fidelity",
        "next_action": (
            "Build the numerical-fidelity acceptance suite: source-backed "
            "reference solutions, norms/tolerances, mesh+timestep convergence, "
            "restart reproducibility, backend/precision parity, and a limiter-"
            "zero proof for the PF-1000 full-energy scope."
        ),
    },
    {
        "gate": "same_scope_comparator",
        "packet_key": "same_scope_source",
        "next_action": (
            "Extract a same-scope PF-1000 full-energy (27-40 kV, 24-rod) "
            "source packet from KnowledgeReference: digitized current "
            "waveform, startup, density/field/temperature histories, neutron "
            "timing/spectrum/anisotropy, detector calibration, an uncertainty "
            "budget, and an independent review certificate."
        ),
    },
    {
        "gate": "uq",
        "packet_key": "comparator_uq",
        "next_action": (
            "Construct the comparator/UQ matrix once same-scope targets exist: "
            "per-observable output mapping, metrics, tolerances, measurement/"
            "model/numerical uncertainty, a UQ propagation method, negative "
            "controls, and an independent review certificate."
        ),
    },
    {
        "gate": "certificate",
        "packet_key": "certificate_gate",
        "next_action": (
            "Assemble the first-principles certificate only after every "
            "upstream gate is accepted: run/evidence hashes, reviewer "
            "metadata, the full negative-test matrix, and a release decision."
        ),
    },
    {
        "gate": "geometry",
        "packet_key": None,  # resolved from boundary_policy.conductor_mask
        "next_action": (
            "Request the four absent PF-1000 hollow-bore geometry dimensions "
            "(anode hollow-bore length, insulator wall thickness, backplate "
            "radial extent and axial thickness) from IPPLM and supply a "
            "same-scope reviewed conductor-mask geometry packet."
        ),
    },
    {
        "gate": "startup",
        "packet_key": "startup",
        "next_action": (
            "Author the startup BVP source packet for D2 breakdown / "
            "flashover / liftoff handoff with same-scope geometry, insulator, "
            "and early-circuit evidence; imported-PIC startup stays context-"
            "only and cannot satisfy this gate."
        ),
    },
    {
        "gate": "power_port",
        "packet_key": "power_port",
        "next_action": (
            "Construct the reviewed sigma-p face set for power-port terms "
            "II/IV/V/VI, supply terminal current/voltage and the active-load "
            "relation, fix sign convention and time centering, and close the "
            "Auluck Eq.6 six-term energy ledger with a residual tolerance."
        ),
    },
    {
        "gate": "neutron",
        "packet_key": "neutron_authority",
        "next_action": (
            "Provide mechanism-separated beam-target and thermonuclear yield "
            "histories with a DD cross-section source, a deuteron transport/"
            "stopping model, an activation-counter response model, and a same-"
            "scope reviewed neutron source packet."
        ),
    },
)

# Status prefixes that mean a gate packet is accepted.  Every PF-1000 full-
# energy gate packet is fail-closed today, so the dry run reports them all as
# ``blocked`` -- this keeps the ledger honest if a gate is ever accepted.
_ACCEPTED_STATUS_PREFIXES = ("accepted", "passed", "ready")


@dataclass(frozen=True)
class GateDryRunResult:
    """One gate's report-only dry-run result.

    ``status`` is ``pass`` or ``blocked``.  A ``blocked`` gate always carries a
    non-empty ``missing`` list naming the exact missing source packets, runtime
    fields, or numerical checks, plus a short ``next_action``.
    """

    gate: str
    status: str
    packet_status: str
    missing: tuple[str, ...] = ()
    next_action: str = ""
    promotes_acceptance: bool = False
    can_support_first_principles_acceptance: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceGateDryRunLedger:
    """Fail-closed report-only ledger for the eight acceptance gates.

    The ledger is a report.  Every acceptance-bearing flag is hard-coded False
    and there is no code path that can flip one.
    """

    deck_preset: str
    runtime_status: str
    gates: tuple[GateDryRunResult, ...]
    report_only: bool = True
    promotes_acceptance: bool = False
    accepted_runtime_claim: bool = False
    can_support_first_principles_acceptance: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gates"] = [gate.as_dict() for gate in self.gates]
        return payload

    @property
    def all_gates_reported(self) -> bool:
        return len(self.gates) == len(_GATE_PLAN)

    @property
    def is_fail_closed(self) -> bool:
        """True when the ledger is report-only and no flag promotes acceptance.

        The dry run is designed to eventually report ``pass`` gates, so a
        passing gate does NOT by itself break fail-closed.  ``is_fail_closed``
        is True when ALL of the following hold:

        - no ledger-level flag promotes acceptance
          (``promotes_acceptance``, ``accepted_runtime_claim``,
          ``can_support_first_principles_acceptance`` all False);
        - every gate has a recognized status (``pass`` or ``blocked``);
        - every ``blocked`` gate names at least one missing input;
        - no gate-level flag promotes acceptance.

        It does NOT mean "no gate passes" -- a gate may legitimately pass once
        its backing packet authorizes first-principles acceptance (see
        ``_build_gate_result``).
        """

        if self.promotes_acceptance or self.accepted_runtime_claim:
            return False
        if self.can_support_first_principles_acceptance:
            return False
        for gate in self.gates:
            if gate.status not in {"pass", "blocked"}:
                return False
            if gate.status == "blocked" and not gate.missing:
                return False
            if gate.promotes_acceptance or gate.can_support_first_principles_acceptance:
                return False
        return True


def _packet_status(packet: Mapping[str, Any] | None) -> str:
    if not isinstance(packet, Mapping):
        return "missing_packet"
    status = packet.get("status")
    return str(status) if status is not None else "missing_status"


def _status_is_accepted(status: str) -> bool:
    normalized = status.strip().lower()
    return normalized.startswith(_ACCEPTED_STATUS_PREFIXES)


def _named_missing(packet: Mapping[str, Any] | None) -> list[str]:
    """Collect the named missing inputs a gate packet already reports.

    Reads the gate's own ``missing_acceptance_channels`` and any blocker
    fields.  Nothing is invented -- the names come straight from the packet.
    """

    if not isinstance(packet, Mapping):
        return ["gate_packet_absent_from_runtime_telemetry"]

    missing: list[str] = []
    channels = packet.get("missing_acceptance_channels")
    if isinstance(channels, (list, tuple)):
        missing.extend(str(item) for item in channels)

    reasons = packet.get("blocking_reasons")
    if isinstance(reasons, (list, tuple)):
        missing.extend(str(item) for item in reasons)

    blocker = packet.get("blocker")
    if isinstance(blocker, str) and blocker:
        missing.append(blocker)

    # Deduplicate while preserving first-seen order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in missing:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _geometry_packet(telemetry_packets: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Resolve the geometry gate packet: boundary_policy.conductor_mask."""

    boundary = telemetry_packets.get("boundary_policy")
    if not isinstance(boundary, Mapping):
        return None
    conductor_mask = boundary.get("conductor_mask")
    return conductor_mask if isinstance(conductor_mask, Mapping) else None


def _geometry_missing(packet: Mapping[str, Any] | None) -> list[str]:
    """Named missing inputs for the geometry gate.

    The conductor-mask packet reports blocked geometry via
    ``blocked_geometry_fields`` (a list of typed field records), not
    ``missing_acceptance_channels``.
    """

    if not isinstance(packet, Mapping):
        return ["conductor_mask_geometry_packet_absent_from_runtime_telemetry"]

    missing: list[str] = []
    blocked_fields = packet.get("blocked_geometry_fields")
    if isinstance(blocked_fields, (list, tuple)):
        for entry in blocked_fields:
            if isinstance(entry, Mapping):
                name = entry.get("field_name") or entry.get("blocker_id")
                if name is not None:
                    missing.append(str(name))
            else:
                missing.append(str(entry))
    if not missing:
        # Fail closed: an unreviewed conductor-mask packet with no explicit
        # blocked-field list is still not an accepted geometry source.
        missing.append("blocked_geometry_fields_not_reported_by_conductor_mask")
    return missing


def _build_gate_result(
    plan: Mapping[str, Any],
    telemetry_packets: Mapping[str, Any],
) -> GateDryRunResult:
    gate_name = str(plan["gate"])
    next_action = str(plan["next_action"])

    if gate_name == "geometry":
        packet = _geometry_packet(telemetry_packets)
        packet_status = _packet_status(packet)
        missing = _geometry_missing(packet)
    else:
        packet = telemetry_packets.get(str(plan["packet_key"]))
        if not isinstance(packet, Mapping):
            packet = None
        packet_status = _packet_status(packet)
        missing = _named_missing(packet)

    accepted = (
        isinstance(packet, Mapping)
        and _status_is_accepted(packet_status)
        and not missing
        # Fail-closed (S10-A5): a gate reports ``pass`` ONLY when the backing
        # packet explicitly authorizes first-principles acceptance.  Strict
        # identity -- an absent key or ``None`` is NOT authorization; only a
        # literal ``True`` is.  ``is not False`` was too permissive: it let an
        # accepted-status packet with the flag missing or ``None`` pass.
        and packet.get("can_support_first_principles_acceptance") is True
    )

    if accepted:
        return GateDryRunResult(
            gate=gate_name,
            status="pass",
            packet_status=packet_status,
            missing=(),
            next_action="",
        )

    # Fail-closed: a blocked gate must always name at least one blocker.
    if not missing:
        missing = ["gate_packet_blocked_without_named_missing_input"]
    return GateDryRunResult(
        gate=gate_name,
        status="blocked",
        packet_status=packet_status,
        missing=tuple(missing),
        next_action=next_action,
    )


def _runtime_payload(
    runtime_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a PF-1000 full-energy runtime payload, running the probe if needed.

    ``runtime_result`` may be a payload already produced by
    ``dpf first-principles-3d``; if ``None`` the probe is executed here.
    """

    if runtime_result is not None:
        if not isinstance(runtime_result, Mapping):
            raise TypeError("runtime_result must be a mapping payload")
        return dict(runtime_result)

    # Imported lazily: this keeps the module a leaf reporting tool and avoids a
    # heavy import when a caller supplies a pre-built runtime payload.
    from dpf.cli.main import (
        _first_principles_3d_deck_preset,
        _first_principles_3d_payload,
    )

    deck = _first_principles_3d_deck_preset(PF1000_FULL_ENERGY_DECK_PRESET)
    return _first_principles_3d_payload(deck)


def run_acceptance_gate_dry_run(
    runtime_result: Mapping[str, Any] | None = None,
) -> AcceptanceGateDryRunLedger:
    """Run the report-only acceptance-gate dry run for the PF-1000 full-energy probe.

    Reads the eight gate packets from the PF-1000 full-energy runtime result
    (running the probe if ``runtime_result`` is not supplied) and returns a
    fail-closed :class:`AcceptanceGateDryRunLedger`.  Promotes nothing.
    """

    payload = _runtime_payload(runtime_result)
    telemetry_packets = payload.get("telemetry_packets")
    if not isinstance(telemetry_packets, Mapping):
        telemetry_packets = {}

    gates = tuple(
        _build_gate_result(plan, telemetry_packets) for plan in _GATE_PLAN
    )

    blocked = [gate for gate in gates if gate.status == "blocked"]
    passed = [gate for gate in gates if gate.status == "pass"]
    summary = {
        "deck_preset": PF1000_FULL_ENERGY_DECK_PRESET,
        "gate_count": len(gates),
        "blocked_count": len(blocked),
        "pass_count": len(passed),
        "blocked_gates": [gate.gate for gate in blocked],
        "pass_gates": [gate.gate for gate in passed],
        "runtime_can_support_first_principles_acceptance": bool(
            payload.get("can_support_first_principles_acceptance", False)
        ),
    }

    return AcceptanceGateDryRunLedger(
        deck_preset=str(payload.get("deck", {}).get("source", PF1000_FULL_ENERGY_DECK_PRESET))
        if isinstance(payload.get("deck"), Mapping)
        else PF1000_FULL_ENERGY_DECK_PRESET,
        runtime_status=str(payload.get("scientific_status", "unknown")),
        gates=gates,
        summary=summary,
    )


def write_ledger_json(
    ledger: AcceptanceGateDryRunLedger,
    path: str | Path,
) -> Path:
    """Write the ledger to ``path`` as JSON and return the resolved path.

    Callers choose the path.  The CLI entry point defaults this to a temp path
    so a transient ledger never dirties ``results/`` or the artifact linter.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(ledger.as_dict(), indent=2, sort_keys=True)
    )
    return destination
