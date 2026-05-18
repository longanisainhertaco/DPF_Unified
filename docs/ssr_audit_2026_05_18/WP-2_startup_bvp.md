# WP-2 / SSR-004 Startup BVP — SSR Audit

Date: 2026-05-18
Auditor scope: WP-2 Startup BVP (SSR-004)
Repo: `/Users/anthonyzamora/dpf-unified`, branch `codex/corpus`, runtime `.venv312/bin/python` (Python 3.12.13, confirmed)
Source authority: `KnowledgeReference/` only.

Audited files (read-only):
- `src/dpf/first_principles/startup_bvp.py` (unmodified on branch)
- `src/dpf/first_principles/startup_breakdown.py` (unmodified on branch)
- `src/dpf/first_principles/deck.py` (modified on branch)
- `src/dpf/first_principles/runner.py` (modified on branch)
- `src/dpf/experimental/civ_breakdown.py` (breakdown physics, read-only)
- `src/dpf/first_principles/certificate_gate.py` (read-only)
- `tests/test_first_principles_runner.py`, `tests/test_first_principles_input_deck.py`, `tests/test_startup_breakdown_audit.py`

---

## (a) Verdict And Reasoning

Verdict: **request_changes**.

The startup BVP work is honest at the *default-deck* and *legacy-mode* level: the PF-1000/Akel
demonstrator deck uses `mode="seeded_layer"`, which the `startup_bvp` packet correctly classifies
as `rejected_startup_mode_for_first_principles`, and `can_support_first_principles_acceptance`
is `False`. Adversarial forcing of `seeded_layer` to accepted status fails closed. The certificate
gate carries `rejected_startup_mode_for_first_principles` in `BLOCKING_UPSTREAM_STATUSES`, so a
seeded-layer startup cannot be laundered into an accepted certificate. SSR-004's hard requirement
("`seeded_layer` startup is accepted" -> reject) is satisfied for the seeded path.

However the submission **cannot be accepted** for three reasons:

1. **Packet-honesty defect (blocking).** For the two *accepted* modes (`surface_breakdown_bvp`,
   `imported_pic_sheath_state`), `build_startup_bvp_packet` grants
   `status="accepted_startup_bvp_packet"` and `can_support_first_principles_acceptance=True`
   purely on caller-declared `accepted_channels`, with **no startup_payload supplied and no
   mode-required payload channels present**. The packet simultaneously reports
   `startup_payload_review.status="startup_payload_not_supplied"` and a `mode_payload_status`
   showing nearly every payload channel `missing_or_unreviewed_payload`. A packet that says
   "accepted" while its own payload review says "not supplied" is internally contradictory and
   violates SSR-004's intent and Rejection Criterion "Runtime artifacts omit packet statuses" in
   spirit. `accepted_channels` is reachable from user config (`runner` field
   `startup_accepted_channels`, `deck.py:199`; `StartupPolicy.accepted_channels`, `deck.py:299/406`),
   so this is an exploitable text-only acceptance path. Verified empirically (see section c).

2. **No source-backed breakdown model exists; the only computational breakdown model is
   training-data-sourced.** The single breakdown computation wired into startup
   (`startup_breakdown.py` -> `dpf.experimental.civ_breakdown`) is a Critical Ionization
   Velocity / Paschen scaffold whose docstring cites Alfven 1954, Brenning 1992, Danielsson 1970,
   Haerendel 1982 — none in `KnowledgeReference/`. The module self-flags
   `source_status="civ_paschen_gas_coefficients_source_packets_missing"` and
   `validation_status="not_validation_evidence"`, and `startup_breakdown.py` forces
   `can_support_first_principles_acceptance: False` on all outputs. This honesty keeps the run
   safe, but it means **zero of the 9 SSR-004 startup channels are source-backed-and-implemented**;
   the breakdown channel in particular has no KR closure. WP-2 deliverable "Implemented startup
   state generator ... Startup fields, particles, current density, ionization, temperatures, and
   sheath liftoff" is not met — only fail-closed packetization exists.

3. **No dedicated negative-test file; handoff interval is a tracked blocker, not implemented.**
   SSR-004 and WP-2 require "negative tests proving seeded/text-only startup remains rejected" and
   "Handoff interval into the field/PIC loop." There is no `tests/test_first_principles_startup_bvp.py`.
   Negative coverage is partial and scattered across two files; critically there is **no test that
   catches defect (1)** — i.e., no test asserts that an accepted mode with all channels declared
   but an empty payload is still rejected. The handoff interval appears only as a *missing*
   acceptance channel (`startup_handoff_interval` in `power_port.missing_acceptance_channels`,
   asserted at `test_first_principles_runner.py:564-567`); no explicit liftoff/handoff interval
   is computed and passed into the field/PIC loop.

Net: the simulator is honestly blocked at the seeded/default level (good), but the accepted-mode
acceptance gate is unsound, the breakdown physics is uncited, and the required negative tests are
incomplete. Fix the acceptance gate and add the negative-test file before this WP can advance.

---

## (b) Source Evidence Table

All cited KR files opened at the cited lines and verified in this audit.

| Cited source : lines | Claim it must support | Verified? | Notes |
| --- | --- | --- | --- |
| `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md:55-80` | DPF phase structure: insulator gas breakdown, kinetic surface discharge (ns-100ns), MHD inverse pinch to cathode bars, microsecond axial acceleration | TRUE | Lines 60-75 read verbatim: "first stage is gas breakdown developing along the exterior of a cylindrical insulator ... surface discharge ... a few to a hundred nanoseconds ... non-equilibrium kinetic (K) character ... avalanche"; "second stage ... MHD ... inverse pinch ... expands from the insulator to the cathode bars"; "third stage ... several microseconds ... supersonic plasma acceleration by an azimuth magnetic field". Fully supports the four-phase claim. |
| `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md:616-642` | Paschen-style pressure regimes are guidelines only; Paschen<->DPF breakdown link is fragile | TRUE | Lines 627-636 read verbatim: "the connection between Paschen-type physics and DPFs is fragile - the Paschen curve refers to a specific situation where ions ... release sufficient secondaries ... Notably, this is not the type of breakdown that occurs in DPFs ... these pressure boundaries are variable guidelines proportional to the geometry of a specific DPF". Directly supports keeping CIV/Paschen scaffold as candidate-only. |
| `startup_bvp.py:50-53` cites `gribkov-2007-pf1000-jphysd-part2.md:56-74` | "pf1000 surface discharge avalanche streamer context" | TRUE | Lines 56-74 contain the surface-discharge/avalanche/streamer text. Citation accurate. |
| `startup_bvp.py:42-48` cites `effect-of-current-sheath...b2e95b88.md:452-670` | "pressure regime, insulator length, sheath mass and velocity context" | TRUE (broad) | File has 1076 lines; range 452-670 is valid and contains the Paschen-regime / pressure-boundary discussion (the spec's 616-642 target sits inside it). Range is broad but not fabricated. |
| `startup_bvp.py` `STARTUP_BVP_SOURCE_REFS` — 6 other KR files (`...acb71fa9.md`, `...alegra-hedp...md`, `sand2009-6373-b93aec67.md`, `...versatile-dense-pinch...md`, `...12205ba4.md`, `...d1758d55.md`) | Startup/breakdown/liftoff/handoff context | FILE-EXISTS verified; line-content NOT exhaustively re-verified | All 6 files exist in `KnowledgeReference/`. Per audit scope only the two spec-named sources were content-verified line-by-line. No fabricated paths found. |
| `startup_breakdown.py:15-39` `STARTUP_BREAKDOWN_AUDIT_SOURCE_REFS` — incl. `alfven-ionization-in-an-mhd-gas-interactions-code.md:420-447`, `gribkov...:56-74`, `alegra-hedp...:245-392,555-585` | CIV candidate closure / surface discharge / breakdown-liftoff-handoff context | gribkov:56-74 TRUE; others file-level only | The Alfven-ionization KR file is cited as "candidate closure context" (not as acceptance authority), consistent with fail-closed status. |
| `civ_breakdown.py:19-24` docstring references: Alfven 1954, Brenning 1992, Danielsson 1970, Haerendel 1982 | Breakdown physics (v_crit, ExB, Townsend, Paschen, sheath, Te) | FALSE as source authority | NONE of these are in `KnowledgeReference/`. They are training-data / external-literature citations in a docstring. The module correctly self-flags `not_validation_evidence`; it is NOT promoted, so this is a documented gap rather than a hidden overclaim — but it means the breakdown channel has no local source. |

Citations verified TRUE (content-level): 4. Citations verified TRUE (file-existence only, content not re-verified): 8 KR files across the two source-ref tuples. Citations FALSE as physics authority: 1 (the `civ_breakdown.py` docstring reference set — 4 external papers, none in KR; not promoted, so honest-gap not fabrication). No fabricated KR paths or non-existent KR line ranges were found.

---

## (c) Packet-Status Honesty Check

Method: `.venv312/bin/python -c` direct calls to `build_startup_bvp_packet` (no pytest, no CLI).

PASS — seeded_layer fails closed even under adversarial forcing:

```
build_startup_bvp_packet({
    'mode': 'seeded_layer',
    'evidence_status': 'accepted_same_scope_source',
    'can_support_whole_shot_acceptance': True,
    'accepted_channels': <all 18 REQUIRED_STARTUP_CHANNELS>,
})
-> status = rejected_startup_mode_for_first_principles
-> can_support_first_principles_acceptance = False
-> whole_shot_startup_blocked = True
-> startup_mode_class = rejected_for_accepted_claims
```

PASS — default PF-1000/Akel deck (`mode="seeded_layer"`): packet status
`rejected_startup_mode_for_first_principles`; covered by
`test_pf1000_candidate_breakdown_profile_seeds_insulator_layer_only` and
`test_first_principles_runner_rejects_seeded_startup_for_acceptance`. Certificate gate receives
`rejected_startup_mode_for_first_principles` (asserted `test_first_principles_runner.py:1568-1571`).

PASS — `not_declared` / unknown mode: `status=blocked_startup_bvp_packet_not_available`,
`startup_mode_class=unknown`, acceptance `False`.

FAIL (BLOCKING) — accepted modes accept on self-declaration with empty payload:

```
build_startup_bvp_packet({
    'mode': 'surface_breakdown_bvp',          # also reproduces with imported_pic_sheath_state
    'evidence_status': 'reviewed',
    'can_support_whole_shot_acceptance': True,
    'accepted_channels': <all 18 REQUIRED_STARTUP_CHANNELS>,
})
-> status = accepted_startup_bvp_packet
-> can_support_first_principles_acceptance = True
-> startup_payload_review.status = startup_payload_not_supplied      # CONTRADICTION
-> mode_payload_status = 6/7 channels 'missing_or_unreviewed_payload'
```

Same result for `mode='imported_pic_sheath_state'` with `evidence_status='accepted_same_scope_source'`:
`status=accepted_startup_bvp_packet`, `can_support_first_principles_acceptance=True`, while
`startup_payload_review.status=startup_payload_not_supplied` and all 16 mode-required payload
channels report `missing_or_unreviewed_payload`.

Root cause: `build_startup_bvp_packet` lines 211-219 compute `can_support` as

```
can_support = whole_shot_requested and mode_is_accepted and reviewed and not missing
```

where `missing = set(REQUIRED_STARTUP_CHANNELS) - accepted` and `accepted` is built *only* from
caller-supplied `accepted_channels` / `startup["accepted_channels"]` (plus payload-review channels
*only if* `channel_acceptance_eligible` — which itself requires a payload). The gate never
consults `startup_payload_review["channel_acceptance_eligible"]` nor a "no missing mode-required
payload" condition. So a caller can declare all 18 channels and skip the payload entirely.
`startup_payload_review` and `mode_payload_status` are computed correctly and tell the truth — but
they are *advisory*; they do not feed the `can_support` decision. This is the defect.

Honest-status conclusion: seeded/legacy/unknown paths are honest. Accepted-mode acceptance is
**not** honest — the headline `status` / `can_support_first_principles_acceptance` can disagree
with the same packet's `startup_payload_review`.

`startup_breakdown.py` honesty check: PASS. Every code path (`build_candidate_startup_breakdown_audit`
and `_blocked_packet`) hard-sets `can_support_validation_claims=False`,
`can_support_whole_shot_acceptance=False`, `can_support_first_principles_acceptance=False`, and
`decision="do_not_promote_civ_paschen_audit_to_startup_bvp"`. `_candidate_breakdown_audit_packet`
(`startup_bvp.py:559-571`) re-forces those flags False even if a caller passes a doctored audit.

---

## (d) Startup-Channel Coverage Table

SSR-004 names 9 required startup channels. `startup_bvp.py` declares 18
`REQUIRED_STARTUP_CHANNELS`; the table maps the 9 SSR-004 channels to the implementation and to KR
source backing. "Present" means a runtime value / model is actually produced and wired; "Packet
slot only" means the channel exists as a fail-closed packet field but no source-backed runtime
model fills it.

| # | SSR-004 channel | Implementation state | Source-cited? | Status |
| --- | --- | --- | --- | --- |
| 1 | Gas breakdown model | `civ_breakdown.compute_breakdown` (CIV/Paschen scaffold) wired via `startup_breakdown.py`; packet channel `breakdown_or_flashover_model` | NO local KR source — docstring cites Alfven/Brenning/Danielsson/Haerendel (training data); module self-flags `civ_paschen_gas_coefficients_source_packets_missing` | MISSING (engineering scaffold only; no KR closure) |
| 2 | Preionization state | Packet channel `preionization_state`; `StartupPolicy.initial_ionization_fraction` carries a CIV-derived number | Partial — CIV ionization fraction is uncited; channel listed in `missing_channels` for the PF-1000 deck | MISSING (candidate input only) |
| 3 | Insulator surface flashover | Packet channel `breakdown_or_flashover_model` + mode-required payload `surface_flashover_equations`; no implementation | NO — no flashover equations exist; `surface_flashover_closure` is in `missing_channels` | MISSING |
| 4 | Electrode/insulator boundary conditions | Packet channel `surface_material_secondary_emission` + payload `electrode_insulator_boundary_data`; device geometry (insulator length/material) is carried | Geometry values are KR-cited (PF-1000 deck sources); BC *physics* is not | MISSING (geometry present, BC closure absent) |
| 5 | Initial current-density distribution | Packet channel `initial_current_density_distribution`; listed in PF-1000 deck `missing_channels` | NO source-backed J distribution | MISSING |
| 6 | Electron + ion temperatures | Channels `electron_temperature_initial`, `ion_temperature_initial`; `StartupPolicy` carries `electron_temperature_K` (CIV-derived) and `ion_temperature_K` (= gas temperature) | Partial — values exist but Te is CIV-derived (uncited) and Ti is just fill-gas temperature | Present-as-candidate-input (`candidate_input_only_not_acceptance`) |
| 7 | Ionization / species state | Channel `initial_density_ionization_charge_state`; `background_density_m3` from ideal-gas law, `initial_ionization_fraction` from CIV | Ideal-gas density is defensible; ionization/charge state uncited | MISSING (candidate input only) |
| 8 | Electric + magnetic fields | Channels `initial_electric_field`, `initial_magnetic_field`; PF-1000 deck sets E=(0,0,0) and relies on the circuit boundary | Field handling is documented (`initial_electric_field_note`) but there is no resolved chamber-field BVP | MISSING (no resolved startup field solve) |
| 9 | Sheath liftoff / handoff interval | Channel `sheath_liftoff_and_handoff_interval`; `compute_liftoff_delay` produces a candidate delay; handoff appears as `startup_handoff_interval` in `power_port.missing_acceptance_channels` | NO — liftoff delay is an "engineering_estimate_not_reviewed_startup_bvp"; no explicit handoff interval wired into the field/PIC loop | MISSING |

Coverage summary: 0 of 9 SSR-004 channels are source-backed-and-implemented. ~2 (temperatures,
partially density) exist as engineering candidate inputs and are correctly labelled
`candidate_input_only_not_acceptance`. The remaining channels are fail-closed packet slots with no
runtime model. The packetization infrastructure (mode classes, required-payload maps,
candidate-input mapping, negative-test policy declaration) is well-structured and honest about
this — but WP-2's *implementation* deliverables are not met.

---

## (e) Proposed Patch Text — Negative Tests File

Author note: this file is provided as TEXT only and has NOT been written to the repo. It is the
proposed full content of a new `tests/test_first_principles_startup_bvp.py`. It is designed to
(i) lock SSR-004's seeded-layer rejection, (ii) catch the packet-honesty defect from section (c),
(iii) prove text-only / self-declared acceptance cannot pass, and (iv) prove the certificate gate
blocks on a non-accepted startup. Tests in group (ii) are written so they FAIL against the current
`startup_bvp.py` (they assert the contradiction must not occur); they pass only once the
acceptance gate is fixed to require `startup_payload_review["channel_acceptance_eligible"]`.

```python
"""Negative tests for the first-principles startup BVP (SSR-004 / WP-2).

These tests prove that seeded, uniform/profile, text-only, and self-declared
startup states cannot pass a first-principles acceptance gate. They are
intentionally adversarial: several tests assert that the packet must NOT grant
acceptance on caller-declared channels alone. Such tests are expected to fail
against a startup_bvp module whose acceptance gate ignores the payload review,
and to pass once the gate also requires startup_payload_review channel
acceptance eligibility.

Source basis (KnowledgeReference only):
- gribkov-2007-pf1000-jphysd-part2.md:55-80 -- DPF phase structure: insulator
  gas breakdown, kinetic surface discharge, MHD inverse pinch, microsecond
  axial acceleration. Establishes that startup is a multi-stage breakdown
  problem, not a seeded layer.
- effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-
  particle-accelera-b2e95b88.md:616-642 -- Paschen-style pressure regimes are
  variable guidelines only and the Paschen<->DPF breakdown link is fragile.
  Establishes that a Paschen/CIV scaffold cannot be promoted to an accepted
  startup BVP without local closure.
"""

from __future__ import annotations

import pytest

from dpf.first_principles.startup_bvp import (
    ACCEPTED_STARTUP_MODES,
    REJECTED_STARTUP_MODES,
    REQUIRED_STARTUP_CHANNELS,
    build_startup_bvp_packet,
)


# --- Group 1: seeded_layer and legacy modes must fail closed (SSR-004) -------

@pytest.mark.parametrize("mode", sorted(REJECTED_STARTUP_MODES))
def test_rejected_startup_modes_cannot_support_acceptance(mode: str) -> None:
    """seeded_layer, uniform, and profile startup must never be accepted."""
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    assert packet["status"] == "rejected_startup_mode_for_first_principles"
    assert packet["startup_mode_class"] == "rejected_for_accepted_claims"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_support_whole_shot_acceptance"] is False
    assert packet["whole_shot_startup_blocked"] is True
    assert packet["startup_mode_status"][mode]["decision"] == (
        "must_fail_acceptance_gate"
    )


def test_seeded_layer_rejection_is_immune_to_declared_channels() -> None:
    """Declaring every required channel must not rescue a seeded layer."""
    packet = build_startup_bvp_packet(
        {
            "mode": "seeded_layer",
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "missing_channels": (),
        }
    )
    assert packet["status"] == "rejected_startup_mode_for_first_principles"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["negative_test_policy"]["seeded_layer_rejection_required"] is True


# --- Group 2: accepted modes must NOT accept on declaration alone -----------
#   These catch the packet-honesty defect in section (c) of the WP-2 audit.

@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_accepted_mode_without_payload_cannot_support_acceptance(
    mode: str,
) -> None:
    """An accepted mode with no startup_payload must stay blocked.

    Declaring all required channels but supplying no payload is a text-only
    acceptance attempt. The headline status must agree with the packet's own
    startup_payload_review: if the payload is not supplied, acceptance must be
    False.
    """
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    review = packet["startup_payload_review"]
    assert review["status"] == "startup_payload_not_supplied"
    # The headline decision must not contradict the payload review.
    assert packet["can_support_first_principles_acceptance"] is False, (
        "accepted-mode startup granted acceptance with no payload supplied; "
        "headline status contradicts startup_payload_review"
    )
    assert packet["status"] != "accepted_startup_bvp_packet"
    assert packet["whole_shot_startup_blocked"] is True


@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_accepted_mode_headline_status_matches_payload_review(
    mode: str,
) -> None:
    """Headline acceptance must never exceed payload-review eligibility."""
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    review = packet["startup_payload_review"]
    if not review["channel_acceptance_eligible"]:
        assert packet["can_support_first_principles_acceptance"] is False
        assert packet["status"] != "accepted_startup_bvp_packet"


@pytest.mark.parametrize("mode", sorted(ACCEPTED_STARTUP_MODES))
def test_accepted_mode_with_incomplete_payload_stays_blocked(
    mode: str,
) -> None:
    """A partial payload (one channel) must not pass the acceptance gate."""
    packet = build_startup_bvp_packet(
        {
            "mode": mode,
            "evidence_status": "reviewed",
            "source_scope": "pf1000_akel_16kv_shot_12581",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_payload": {
                "mode": mode,
                "evidence_status": "reviewed",
                "source_scope": "pf1000_akel_16kv_shot_12581",
                "can_support_whole_shot_acceptance": True,
                # Only one payload channel present; the rest are missing.
                "magnetic_field": {"value": "placeholder"},
                "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            },
        }
    )
    review = packet["startup_payload_review"]
    assert review["status"] in {
        "startup_payload_incomplete",
        "startup_payload_blocked",
    }
    assert review["missing_payload_fields"], "incomplete payload not detected"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["status"] != "accepted_startup_bvp_packet"


# --- Group 3: unknown / undeclared modes fail closed ------------------------

def test_undeclared_startup_mode_is_blocked() -> None:
    """A startup packet with no declared mode must block, not accept."""
    packet = build_startup_bvp_packet({})
    assert packet["status"] == "blocked_startup_bvp_packet_not_available"
    assert packet["startup_mode_class"] == "unknown"
    assert packet["can_support_first_principles_acceptance"] is False


def test_unknown_startup_mode_blocks_acceptance() -> None:
    """An invented mode name must be classed unknown and fail the gate."""
    packet = build_startup_bvp_packet(
        {
            "mode": "definitely_not_a_real_startup_mode",
            "evidence_status": "accepted_same_scope_source",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    assert packet["startup_mode_class"] == "unknown"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["startup_mode_status"][
        "definitely_not_a_real_startup_mode"
    ]["status"] == "unknown_startup_mode_blocks_acceptance"


# --- Group 4: engineering-only modes cannot reach whole-shot acceptance -----

def test_engineering_only_end_rundown_mode_cannot_support_whole_shot() -> None:
    """source_backed_end_rundown_sheath is engineering-only, never whole-shot."""
    packet = build_startup_bvp_packet(
        {
            "mode": "source_backed_end_rundown_sheath",
            "evidence_status": "reviewed",
            "can_support_whole_shot_acceptance": True,
            "accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
        }
    )
    assert packet["startup_mode_class"] == "engineering_only"
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["startup_mode_status"][
        "source_backed_end_rundown_sheath"
    ]["status"] == "engineering_candidate_not_whole_shot"


# --- Group 5: CIV/Paschen breakdown audit cannot be promoted ----------------

def test_candidate_breakdown_audit_cannot_promote_startup() -> None:
    """A CIV/Paschen breakdown audit is engineering-only (Paschen<->DPF link
    is fragile per the current-sheath-initiation source) and must never lift
    the startup packet to acceptance."""
    doctored_audit = {
        "status": "candidate_civ_paschen_breakdown_audit_engineering_only",
        # Adversarial: caller tries to assert acceptance on the audit.
        "can_support_first_principles_acceptance": True,
        "breakdown": {"initial_ionization_fraction": 0.1},
        "liftoff": {"candidate_liftoff_delay_s": 1.0e-8},
    }
    packet = build_startup_bvp_packet(
        {
            "mode": "seeded_layer",
            "evidence_status": "reviewed",
        },
        candidate_breakdown_audit=doctored_audit,
    )
    audit = packet["candidate_breakdown_audit"]
    assert audit["can_support_first_principles_acceptance"] is False
    assert audit["can_support_whole_shot_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


# --- Group 6: runner + certificate-gate integration -------------------------

def test_runner_seeded_startup_blocks_certificate_gate() -> None:
    """A seeded-layer startup must propagate a blocking status into the
    certificate gate, so no accepted certificate can be written."""
    from dpf.first_principles.runner import run_first_principles_3d_deck

    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "startup_mode": "seeded_layer",
            "startup_evidence_status": "reviewed",
            "startup_can_support_whole_shot_acceptance": False,
            "startup_missing_channels": (),
        }
    )
    startup = result.telemetry["startup"]
    assert startup["status"] == "rejected_startup_mode_for_first_principles"
    assert startup["can_support_first_principles_acceptance"] is False
    gate = result.telemetry["certificate_gate"]
    assert (
        gate["upstream_packet_statuses"]["startup_bvp"]
        == "rejected_startup_mode_for_first_principles"
    )
    assert gate["can_support_first_principles_acceptance"] is False


def test_runner_text_declared_accepted_startup_does_not_pass_certificate() -> (
    None
):
    """Even if a deck declares an accepted startup mode with all channels but
    no payload, the certificate gate must not be accepted. This is the
    runner-level guard for the section (c) packet-honesty defect."""
    from dpf.first_principles.runner import run_first_principles_3d_deck

    result = run_first_principles_3d_deck(
        {
            "n_steps": 1,
            "grid_shape": (4, 4, 4),
            "dt_s": 1.0e-13,
            "startup_mode": "surface_breakdown_bvp",
            "startup_evidence_status": "reviewed",
            "startup_can_support_whole_shot_acceptance": True,
            "startup_accepted_channels": list(REQUIRED_STARTUP_CHANNELS),
            "startup_missing_channels": (),
        }
    )
    startup = result.telemetry["startup"]
    # The startup payload was never supplied; acceptance must not be granted.
    assert startup["startup_payload_review"]["status"] == (
        "startup_payload_not_supplied"
    )
    assert startup["can_support_first_principles_acceptance"] is False, (
        "text-declared accepted startup with no payload reached acceptance"
    )
    gate = result.telemetry["certificate_gate"]
    assert gate["can_support_first_principles_acceptance"] is False
```

Proposed companion fix (TEXT ONLY — not applied) for the section (c) defect, in
`build_startup_bvp_packet` (`startup_bvp.py`, the `can_support` block at lines 211-219):

```python
# CURRENT (unsound): acceptance depends only on caller-declared channels.
can_support = (
    whole_shot_requested
    and mode_is_accepted
    and reviewed
    and not missing
)

# PROPOSED: also require the payload review to be channel-acceptance eligible,
# and require no missing mode-required payload channel. This makes the headline
# status agree with startup_payload_review and mode_payload_status.
payload_eligible = bool(
    startup_payload_review.get("channel_acceptance_eligible")
)
mode_payload_complete = all(
    state == "accepted_payload_channel_declared"
    for state in _mode_payload_status(mode, accepted).values()
) if MODE_REQUIRED_PAYLOADS.get(mode) else False
can_support = (
    whole_shot_requested
    and mode_is_accepted
    and reviewed
    and not missing
    and payload_eligible
    and mode_payload_complete
)
```

Rationale: `startup_payload_review` already computes `channel_acceptance_eligible` honestly
(it requires a supplied, reviewed, same-scope, complete payload). The fix simply makes the
headline `can_support` consume that signal instead of trusting `accepted_channels` alone. With
the fix, the two adversarial cases in section (c) return `blocked_*` instead of
`accepted_startup_bvp_packet`, and the Group-2 negative tests above pass.

---

## (f) Negative Tests — Present vs Missing

Present (existing repo tests, verified by reading the files):

| Test | File:line | Covers |
| --- | --- | --- |
| `test_first_principles_runner_rejects_seeded_startup_for_acceptance` | `test_first_principles_runner.py:1540` | seeded_layer -> `rejected_startup_mode_for_first_principles`, certificate gate carries the rejection |
| `test_pf1000_candidate_breakdown_profile_seeds_insulator_layer_only` | `test_first_principles_runner.py:1466` | default PF-1000 deck startup is `rejected_startup_mode_for_first_principles` |
| `test_first_principles_3d_runner_carries_startup_policy_from_package_deck` | `test_first_principles_runner.py:1508` | end-rundown mode -> `blocked_startup_bvp_packet_not_available` |
| `test_seeded_layer_startup_cannot_claim_whole_shot_acceptance` | `test_first_principles_input_deck.py:257` | deck-level seeded_layer cannot claim acceptance |
| `test_imported_pic_startup_requires_review_before_acceptance` | `test_first_principles_input_deck.py:269` | imported PIC mode needs review |
| `test_surface_breakdown_bvp_with_missing_channels_remains_blocked` | `test_first_principles_input_deck.py:282` | surface BVP with missing channels stays blocked |
| `test_startup_breakdown_audit.py` (whole file) | `tests/test_startup_breakdown_audit.py` | CIV/Paschen breakdown audit fail-closed behavior |

Missing (required by SSR-004 / WP-2, not present):

1. No `tests/test_first_principles_startup_bvp.py` dedicated file (SSR-004 audit expectation:
   "tests in `tests/test_first_principles_runner.py` or a dedicated startup test file").
2. **No test catches the section (c) defect** — no existing test asserts that an accepted mode
   with all `accepted_channels` declared but an empty `startup_payload` must still be blocked.
   This is the most important gap; the current test
   `test_surface_breakdown_bvp_with_missing_channels_remains_blocked` only checks the case where
   `missing_channels` is non-empty, never the case where the caller declares everything but
   supplies no payload.
3. No negative test asserts the headline `status` / `can_support_first_principles_acceptance`
   must agree with `startup_payload_review["channel_acceptance_eligible"]`.
4. No runner-level test exercises a text-declared accepted startup mode reaching (and being
   rejected by) the certificate gate.
5. No test for `imported_pic_sheath_state` self-declared acceptance with empty payload at the
   `startup_bvp` packet level (the deck-level `__post_init__` raises on empty payload, but the
   `build_startup_bvp_packet` path is independently reachable and is the unsound one).

Section (e) provides all five as concrete pytest functions.

---

## (g) Remaining Blockers

1. **Acceptance-gate honesty defect (BLOCKING).** `build_startup_bvp_packet` grants
   `accepted_startup_bvp_packet` / `can_support_first_principles_acceptance=True` for accepted
   modes on caller-declared `accepted_channels` alone, contradicting its own
   `startup_payload_review`. Fix proposed in section (e). Until fixed, the startup packet's
   acceptance verdict is not trustworthy and SSR-004 cannot be considered structurally sound.

2. **No source-backed breakdown / flashover model.** The only computational breakdown model is
   the CIV/Paschen scaffold in `civ_breakdown.py`, sourced from training-data papers (Alfven 1954
   et al.), not `KnowledgeReference/`. The current-sheath-initiation KR source
   (`b2e95b88.md:616-642`) explicitly states the Paschen<->DPF link is fragile, so this scaffold
   cannot be promoted. WP-2 needs either (a) a reviewed `surface_breakdown_bvp` implementation
   with KR-cited flashover/avalanche/secondary-emission closures, or (b) a reviewed imported PIC
   sheath state with a full payload. Neither exists.

3. **0 of 9 SSR-004 startup channels source-backed-and-implemented.** All channels are
   fail-closed packet slots; ~2 (temperatures, density) are engineering candidate inputs only.
   The breakdown model, preionization, flashover, electrode/insulator BC physics, initial
   current-density distribution, resolved E/B startup fields, and a reviewed liftoff are all
   absent.

4. **No explicit handoff interval into the field/PIC loop.** `sheath_liftoff_and_handoff_interval`
   is a required channel; `compute_liftoff_delay` yields only an
   `engineering_estimate_not_reviewed_startup_bvp`. `startup_handoff_interval` appears as a
   *missing* acceptance channel in the `power_port` packet — i.e., the handoff is tracked as a
   blocker, not implemented. WP-2 deliverable "Handoff interval into the field/PIC loop" is not met.

5. **Missing negative tests** (section f) — no dedicated startup BVP test file and, critically,
   no test that catches blocker 1.

What is NOT a blocker (credit where due): the default PF-1000/Akel deck honestly uses
`seeded_layer` and is correctly rejected; `seeded_layer` cannot be forced to accepted under
adversarial input; the certificate gate correctly treats
`rejected_startup_mode_for_first_principles` as a blocking upstream status; `startup_breakdown.py`
hard-pins `can_support_first_principles_acceptance=False` on every path; KR source citations in
`startup_bvp.py` / `startup_breakdown.py` are accurate (no fabricated paths or non-existent line
ranges found); the packet's `negative_test_policy` correctly *declares* the eight negative-test
requirements (even though five are not yet implemented).

WP-2 status: explicitly blocked, as SSR-004 requires. Verdict `request_changes` — fix blocker 1
(acceptance gate), then deliver a KR-backed breakdown model or reviewed imported PIC payload, add
the negative-test file, and wire an explicit handoff interval before WP-2 can advance toward
`accept_engineering_progress` or higher.
