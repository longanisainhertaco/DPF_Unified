# WP-6 / SSR-009 Audit — Neutron Mechanism And Detector Authority

Date: 2026-05-18
Auditor scope: WP-6 / SSR-009 (neutron mechanism + detector authority)
Repo: `/Users/anthonyzamora/dpf-unified`  Branch: `codex/corpus`
Runtime: `.venv312/bin/python` (Python 3.12)
Mode: READ-ONLY on all `*.py` and existing docs/tests. Only this file was created.

Files audited:
- `src/dpf/first_principles/neutron_authority.py` (408 lines)
- `src/dpf/fields/kinetic_yield.py` (214 lines)
- `src/dpf/first_principles/runner.py` (neutron wiring; read-only)

---

## (a) Verdict

`accept_engineering_progress` for WP-6 / SSR-009.

The neutron-authority packet is **mechanism-separated, fail-closed, and source-cited
correctly**. A scalar-total-yield-only result provably **cannot** accept neutron
authority, and Lee / hybrid-PIC reduced outputs are correctly held as
comparator/runtime-diagnostic baselines only. SSR-009 and the Rejection Criteria
("Codex will reject total-yield-only claims") are satisfied at the packet level.

It is **not** `accept_certificate_candidate` because two test gaps remain:
1. No dedicated `tests/test_first_principles_neutron_authority.py` exists. Existing
   coverage of `build_mechanism_separated_neutron_packet` is **existence-only**
   (`tests/test_first_principles_source_targets.py:233-270`) — it never exercises the
   scalar-yield-only rejection, the Lee-reduced rejection, the cross-scope rejection,
   or the accepted-channel discrimination logic. The negative tests SSR-009 requires
   ("Tests do not include negative controls" is a Rejection Criterion) are absent for
   this specific function.
2. `build_mechanism_separated_neutron_packet` is **structurally unable to ever emit an
   accepting verdict** (see §d). This is maximally safe, but it means the "accepted"
   branch is dead code: the detector-response / UQ gating that SSR-009 demands "before
   any accepted neutron claim" is asserted by hardcoded `False` literals, not by a
   reviewable gate that consumes a detector-response packet and a UQ packet. This is
   honest (fails closed) but is not yet a *promotable* gate. Flagged as a blocker for
   WP-6 completion, not as an overclaim.

No overclaim was found. No hidden floors, no reduced-model authority, no fabricated
citations.

---

## (b) Source Evidence Table

Every citation below was opened at the cited lines in this audit session and confirmed.

| Local source path:lines | Claim / role asserted by code | Verified? |
| --- | --- | --- |
| `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:121-137` | PF-1000 DD-neutron anisotropy; vessel/wall scattering affects spectra; MCNP separates scattered vs direct neutrons | TRUE — lines 121-130 state the paper studies PF-1000 anisotropy, scattered vs direct group spectra computed by MCNP 4C; vessel-wall scattering elucidates TOF shape |
| `…anisotropy…527cc533.md:175-204` | Silver activation detectors + TOF probes at multiple distances (7, 16.3, 58.3 m); direct/scattered must be distinguished | TRUE — lines 176-180 give four silver activation detectors and TOF probes at 7/16.3/58.3 m; lines 181-204 require distinguishing scattered vs direct |
| `…anisotropy…527cc533.md:269-288` | Separate scattered from direct neutrons in TOF spectrum before transforming to velocity/energy distribution | TRUE — lines 280-283 verbatim: "it is very important to separate scattered neutrons from direct ones in the TOF spectrum before the transformation … Without their separation the direct transformation … is disputable" |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:120-131` | PF-1000/Akel scintillators at 0/90/180 deg; TOF mean neutron/deuteron energy; two calibrated silver activation counters; Am-Be calibration; +-0.2 yield uncertainty; t=0 at current-derivative dip | TRUE — lines 120-133 confirm all listed items verbatim |
| `…radiation-physics-and-chemistry…109633.md:190-215` | Lee model neutron emission via thermonuclear AND beam-target models; beam-target Eq.(1) with calibrated global constant Cn | TRUE — lines 194-216: Lee model has thermonuclear + beam-target; `Yn=Yb-t` Eq.(1); Cn = 8.54e8 "calibrated at an experimental point of 0.5 MA … (global constant)" |
| `…radiation-physics-and-chemistry…109633.md:282-288` | Shot-12581-like scalar yield: Yn = 6.14e9 computed vs measured (6.1+-0.2)e9 | TRUE — lines 286-288 verbatim |
| `…radiation-physics-and-chemistry…109633.md:862-889` | Lee model is a fitted reduced model (current-waveform fitting; fc held constant 0.7) | TRUE — lines 862-889: "fitted to the computed currents", "by fitting the computed current waveform to the measured current waveform" |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:952-970` | Hybrid-PIC yield history (dN/dt, cumulative N(t)); total yield 0.296e7; order-of-magnitude agreement only | TRUE — lines 952-973 confirm yield history + "order-of-magnitude agreement rather than … strict one-to-one validation" (lines 990-991) |
| `…hybrid-pic-fluid…acb71fa9.md:1214-1266` | Model limitations: Te=Ti closure; "factor of a few uncertainty in the absolute neutron yield"; "order-of-magnitude validation rather than … precise prediction" | TRUE — lines 1238-1240 and 1259-1266 confirm verbatim |
| `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:34-43` | Fluid predicts no neutrons; hybrid under-predicts ~100x with ion tail <200 keV; only fully kinetic reaches MeV ions + experimental yields | TRUE — lines 34-44 confirm verbatim; supports kinetic MeV-ion / beam-target requirement |
| `KnowledgeReference/sand2009-6373-b93aec67.md:346-352` | DPF densities/temperatures insufficient for observed thermonuclear yield; anisotropy/spectra indicate non-thermonuclear mechanisms | TRUE — lines 345-352 confirm verbatim |
| `…sand2009-6373…b93aec67.md:511-512` | MHD cannot model non-thermonuclear neutron production | TRUE — lines 509-512: "inability of MHD to model non-thermonuclear production mechanisms" |
| `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:39-44` | Two mechanisms (thermonuclear + beam-target) from kinetic simulations on MJ-class MJOLNIR | TRUE — lines 39-44 confirm verbatim |
| `…neutron-generation-dynamics…z-pinch-5.md:409-448` | Mechanism-separated neutron pulse shape: thermonuclear at stagnation + beam-target at pinch disruptions; synthetic-vs-experiment comparison | TRUE — lines 409-448 confirm successive peaks, peak 1 thermonuclear at stagnation, peaks 2/3/4 beam-target disruptions |
| `KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md:32-53` | Time-energy spectrum validates multiphysics codes; TOF tomography at multiple distances with scatter background subtraction | TRUE — lines 32-53 confirm verbatim |
| `…tomographic-reconstruction…b78f1154.md:390-427` | Shadow-bar detector system: foreground vs shadowed detector for direct-vs-scattered separation; SB scaling factor | TRUE — lines 388-427 confirm verbatim |
| `KnowledgeReference/original-research-f7894f85.md:269-288` | IR-MPF-100: neutron activation counter (Ag foil), Am-Be calibration, ~1.5e9 n/shot, double pinch | TRUE — lines 269-288 confirm verbatim |
| `KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md:180-200` | Compact DPF; S.Lee model used as design tool; TOF detector at 90 deg; pressure-dependent yield | TRUE — lines 183-200 confirm S.Lee simulation model, neutron detector at 90 deg, yield-vs-pressure, max ~6.45e8 n/pulse |
| `KnowledgeReference/open-access-proceedings-journal-of-physics-conference-series-ed196711.md:93-141,152-190,680-697,782-805` | Current abruption / plasma diode / MITL / neutron application context | NOT INDIVIDUALLY OPENED in this audit (10th of 10 sources). File exists (1146 lines); cited ranges are in-bounds. Role claim ("user_validated … context") is consistent with the corpus and is a *context* citation, not an equation. Recommend a follow-up spot-check; not flagged as a defect. |

**Citation accuracy: 17 of 18 line-ranges opened and confirmed TRUE. 0 fabricated. 0
wrong. 1 not individually opened (in-bounds, context-only role).** All line ranges
fall within the actual file lengths. No citation to external AI/web material as source
truth. No PF-1000U / full-energy PF-1000 values mixed into the Akel scope — note the
anisotropy paper is *correctly self-labeled* in `OTHER_SCOPE_SOURCE_GROUPS` as
`pf1000_full_energy_anisotropy` ("PF-1000 operated at 450-500 kJ and 3.5 Torr, not Akel
16 kV"), which matches the paper's own §2 (450-500 kJ, 3.5 Torr). This is the correct
SSR-003 / Rule 9 handling.

---

## (c) Mechanism-Separation Coverage Table

SSR-009 requires separation of: thermonuclear D-D, beam-target, beam formation, beam
transport, spectrum, anisotropy, direct-vs-scattered detector contribution, activation
counter response, TOF response. Mapped to `REQUIRED_NEUTRON_AUTHORITY_CHANNELS` and
`BLOCKING_NEUTRON_AUTHORITY_CHANNELS` in `neutron_authority.py:76-128`.

| SSR-009 mechanism | Packet channel(s) | Present as required channel? | Source-cited (verified)? | Runtime status |
| --- | --- | --- | --- | --- |
| Thermonuclear D-D production | `accepted_thermonuclear_yield_history` | YES (required + blocking) | YES — SAND2009 346-352, MJOLNIR 409-448 | `missing_or_blocked` (correct) |
| Beam-target production | `accepted_beam_target_yield_history` | YES (required + blocking) | YES — fully-kinetic 34-43, MJOLNIR 39-44, Lee context 190-215 | `missing_or_blocked` (correct) |
| Mechanism separation itself | `mechanism_separated_yield_channels` | YES (required + blocking) | YES — MJOLNIR 39-44, 409-448 | `missing_or_blocked` (correct) |
| Beam formation (ion energy distribution) | `ion_energy_distribution_history` | YES (required + blocking) | YES — fully-kinetic 34-43 (MeV ion tail) | `missing_or_blocked` (correct) |
| Beam formation (angular) | `beam_angular_distribution_history` | YES (required + blocking) | YES — fully-kinetic 126-161 (role-cited) | `missing_or_blocked` (correct) |
| Beam transport / stopping | `beam_transport_stopping_model` | YES (required + blocking) | YES — fully-kinetic 68-78, Lee 196-199 (beam traverses pinch column) | `missing_or_blocked` (correct) |
| Target density / path length | `target_density_path_length_history` | YES (required + blocking) | YES — Lee 200-211 (ni, b/rp, zp) | `missing_or_blocked` (correct) |
| D-D cross-section + units | `dd_cross_section_source_and_units` | YES (required; NOT blocking) | YES — Lee Eq.(1) 200-211 | `missing_or_blocked` (correct) |
| Spectrum | `neutron_energy_spectrum` | YES (required + blocking) | YES — anisotropy 269-288, TOF-tomography 32-53 | `missing_or_blocked` (correct) |
| Anisotropy | `neutron_anisotropy_angular_yield` | YES (required + blocking) | YES — anisotropy 121-137, 175-204 | `missing_or_blocked` (correct) |
| TOF response | `neutron_timing_history` | YES (required; NOT blocking) | YES — radiation-phys 120-131 (t=0 dip), MJOLNIR 409-448 | `missing_or_blocked` (correct) |
| Direct vs scattered detector contribution | `direct_scattered_neutron_transport` | YES (required + blocking) | YES — anisotropy 269-288, TOF-tomography 390-427 (shadow bar) | `missing_or_blocked` (correct) |
| Detector response | `detector_response_model` | YES (required + blocking) | YES — TOF-tomography 390-427, anisotropy 175-204 | `missing_or_blocked` (correct) |
| Activation counter response | `activation_counter_response_model` | YES (required + blocking) | YES — radiation-phys 120-131 (Ag counters, Am-Be), original-research 269-288 | `missing_or_blocked` (correct) |
| Scalar yield (comparator only) | `same_scope_scalar_yield` | YES (required; NOT blocking — comparator) | YES — radiation-phys 282-288 | accepted only as `baseline_comparison_only` |
| Yield UQ budget | `yield_uncertainty_budget` | YES (required + blocking) | YES — radiation-phys 130-131 (+-0.2), hybrid-PIC 1238-1240 | `missing_or_blocked` (correct) |
| Te-sensitivity UQ | `electron_temperature_yield_sensitivity_uq` | YES (required + blocking) | YES — hybrid-PIC 1226-1240 (factor-of-a-few Te uncertainty) | `missing_or_blocked` (correct) |
| Output mapping / comparator | `output_mapping_and_comparator` | YES (required + blocking) | YES — TOF-tomography 32-53 | `missing_or_blocked` (correct) |
| Source review certificate | `source_review_certificate` | YES (required + blocking) | n/a (process gate) | `missing_or_blocked` (correct) |

**Coverage assessment: COMPLETE.** All nine SSR-009-mandated mechanisms are represented
as explicit required channels, 15 of them as *blocking* channels. Beam formation is
correctly split into energy-distribution + angular-distribution + transport. The
detector side is correctly split into detector-response + activation-counter-response +
direct/scattered-transport + TOF-timing. `kinetic_yield.py` independently emits
`mechanism_separation_status="not_mechanism_separated"` and
`mechanism_channels=("dd_particle_distribution_total",)` (lines 116-117) — i.e. the PIC
yield diagnostic *self-declares* it is not mechanism-separated. Honest.

---

## (d) Scalar-Yield-Only / Reduced-Model Honesty Check

**Result: PASS. The packet provably cannot accept on scalar yield alone, and reduced
models stay baselines.**

Behavioral probes run this session with `.venv312/bin/python`:

1. **Scalar-yield-only target accepted** — input: one `neutron_scalar_yield` validation
   target with `status=accepted_same_scope_source` and matching PF-1000/Akel scope.
   Output: `status=blocked_mechanism_separated_neutron_authority_not_available`,
   `can_support_total_yield_acceptance=False`, `can_support_first_principles_acceptance=
   False`, 18 channels still missing. The only accepted channel is
   `same_scope_scalar_yield`, and `mechanism_separation_policy.scalar_yield_agreement_
   usable_for="baseline_comparison_only"`. **Total-yield-only cannot accept.**

2. **All 19 required channels declared accepted** — output: `status=blocked`,
   `can_support_first_principles_acceptance=False`, **16 blocking channels still
   reported missing.** Root cause: `neutron_authority.py:199-200` does
   `missing = set(REQUIRED…) - accepted` then unconditionally
   `missing.update(BLOCKING_NEUTRON_AUTHORITY_CHANNELS)`. And lines 258-259 return
   `can_support_total_yield_acceptance: False` / `can_support_first_principles_
   acceptance: False` as **hardcoded literals**.

Reduced-model handling (all correct per Non-Negotiable Rule 4 and SSR-009 "Reduced Lee
neutron outputs may remain comparator baselines only"):
- Lee thermonuclear+beam-target model: lives only in
  `PF1000_AKEL_TEXT_SUPPORTED_CHANNELS` (`lee_thermonuclear_and_beam_target_model_text`,
  `lee_beam_target_formula_context`). These are surfaced as
  `text_supported_reference_channels` with channel status
  `text_supported_reference_only_not_acceptance` (`_channel_statuses`, lines 338-355).
  They are **never** added to `accepted`.
- `mechanism_separation_policy` (lines 231-239): `scalar_yield_agreement_usable_for=
  "baseline_comparison_only"`, `candidate_pic_yield_usable_for="runtime_diagnostic_
  only"`.
- Hybrid-PIC kinetic yield: `kinetic_yield.py` `KineticYieldTelemetry.status=
  "candidate_engineering_kinetic_yield_history"`, `can_support_first_principles_
  acceptance=False` (dataclass default, line 34). `kinetic_yield_candidate_evidence`
  hardcodes `"status": "candidate"` and `can_support_first_principles_acceptance:
  False` (lines 129, 140). `kinetic_neutron_yield_authority_status` fails closed unless
  kinetic + mechanism + detector-response + UQ evidence all carry
  `passed is True and status in {accepted, validated}` (`_accepted`, lines 209-213).
- `acceptance_gate` string (lines 207-211) explicitly says
  "scalar_yield_reduced_model_text_and_other_scope_neutron_diagnostics_cannot_support_
  total_yield_authority_until_same_scope_mechanism_separated_histories_detector_
  transport_comparator_uq_and_review_pass".

Cross-scope honesty (Rule 9 / SSR-003): the PF-1000 full-energy anisotropy paper, the
MJ-class MJOLNIR paper, the NNSS TOF-tomography paper, the LLNL fully-kinetic paper,
and the axisymmetric hybrid paper are all listed in `OTHER_SCOPE_SOURCE_GROUPS` with an
explicit `scope_mismatch` string and `usable_for="requirements_or_schema_only"`.
`cross_scope_policy.can_use_other_scope_for_acceptance=False` and a 9-item
`TRANSFER_RULE_REQUIRED_CHANNELS` list is required before any cross-scope use. Correct.

**Honest weakness (blocker, not overclaim):** because acceptance is two hardcoded
`False` literals plus an unconditional `missing.update(BLOCKING…)`, the detector-
response packet and UQ packet that SSR-009 demands "before any accepted neutron claim"
are **not actually consumed** by `build_mechanism_separated_neutron_packet`. The
function receives `kinetic_yield`, `same_scope_source`, `physics_closure` but no
`detector_response` packet and no `comparator_uq` packet, and its accept path is
unreachable regardless. The sibling function `kinetic_neutron_yield_authority_status`
(in `kinetic_yield.py`) *does* consume `detector_response_evidence` and
`uncertainty_evidence` and *can* return `accepted` — but the runner
(`runner.py:1043-1051`) wires only `build_mechanism_separated_neutron_packet`, not
`kinetic_neutron_yield_authority_status`. So the runtime neutron packet's accept gate
is presently inert. This fails closed (good) but is not a promotable gate (WP-6 not
complete).

---

## (e) Proposed Patch Text — Negative Tests

Author the following as a NEW file `tests/test_first_principles_neutron_authority.py`.
This is proposed text only — DO NOT APPLY as part of this audit. All expected values
were confirmed against live behavior this session.

```python
"""SSR-009 / WP-6 negative controls: neutron authority is mechanism-separated and
cannot accept on scalar total yield or reduced-model outputs.

Verified against KnowledgeReference:
- radiation-physics-and-chemistry-188-2021-109633.md:282-288 (scalar yield 6.14e9)
- radiation-physics-and-chemistry-188-2021-109633.md:862-889 (Lee is a fitted model)
- sand2009-6373-b93aec67.md:511-512 (MHD cannot model non-thermonuclear production)
- fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:34-43 (hybrid under-
  predicts ~100x; only fully kinetic reaches experimental yield)
- anisotropy-...-527cc533.md:269-288 (separate scattered vs direct before TOF inversion)
"""

from __future__ import annotations

from dpf.first_principles.neutron_authority import (
    BLOCKING_NEUTRON_AUTHORITY_CHANNELS,
    REQUIRED_NEUTRON_AUTHORITY_CHANNELS,
    build_mechanism_separated_neutron_packet,
)
from dpf.fields.kinetic_yield import (
    kinetic_neutron_yield_authority_status,
)

_AKEL = "pf1000_akel_16kv_1p2torr_shot_12581"


def test_scalar_total_yield_only_cannot_accept_neutron_authority() -> None:
    """A same-scope, accepted scalar yield target must NOT grant authority."""
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        validation_targets=[
            {
                "name": "pf1000_akel_total_yield",
                "observable": "neutron_scalar_yield",
                "status": "accepted_same_scope_source",
                "declared_scope": _AKEL,
            }
        ],
    )
    assert packet["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert packet["can_support_total_yield_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    # The scalar channel may be the ONLY accepted channel, and it is comparator-only.
    assert packet["accepted_channels"] == ["same_scope_scalar_yield"]
    assert packet["mechanism_separation_policy"][
        "scalar_yield_agreement_usable_for"
    ] == "baseline_comparison_only"
    # Every blocking mechanism channel is still missing.
    missing = set(packet["missing_acceptance_channels"])
    assert set(BLOCKING_NEUTRON_AUTHORITY_CHANNELS).issubset(missing)


def test_packet_fails_closed_even_if_every_channel_declared_accepted() -> None:
    """Declaring all required channels accepted must still not accept: the blocking
    channels are re-asserted and acceptance is fail-closed by construction."""
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        accepted_channels=tuple(REQUIRED_NEUTRON_AUTHORITY_CHANNELS),
    )
    assert packet["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["can_support_total_yield_acceptance"] is False
    # Blocking channels are re-injected into "missing" regardless of input.
    missing = set(packet["missing_acceptance_channels"])
    assert set(BLOCKING_NEUTRON_AUTHORITY_CHANNELS).issubset(missing)


def test_lee_reduced_beam_target_model_stays_text_reference_not_acceptance() -> None:
    """Lee thermonuclear+beam-target model text is comparator/reference only and must
    never appear as an accepted authority channel (KR 109633:862-889 = fitted model)."""
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
    )
    text_refs = set(packet["text_supported_reference_channels"])
    assert "lee_thermonuclear_and_beam_target_model_text" in text_refs
    assert "lee_beam_target_formula_context" in text_refs
    # Lee channels are reference-only, never accepted.
    assert set(packet["accepted_channels"]).isdisjoint(text_refs)
    statuses = packet["neutron_authority_channel_status"]
    for ch in ("accepted_thermonuclear_yield_history",
               "accepted_beam_target_yield_history"):
        assert statuses[ch] == "missing_or_blocked"
    # Scalar measured-yield text is also reference-only.
    assert "measured_scalar_yield_shot_12581" in text_refs


def test_candidate_pic_yield_is_runtime_diagnostic_not_authority() -> None:
    """A candidate PIC kinetic-yield history must not promote the packet."""
    kinetic_yield = {
        "status": "candidate_engineering_kinetic_yield_history",
        "mechanism_separation_status": "not_mechanism_separated",
        "mechanism_channels": ["dd_particle_distribution_total"],
        "cumulative_neutrons": 4.2e9,
    }
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        kinetic_yield=kinetic_yield,
    )
    assert packet["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["mechanism_separation_policy"][
        "candidate_pic_yield_usable_for"
    ] == "runtime_diagnostic_only"
    assert packet["kinetic_yield_mechanism_separation_status"] == (
        "not_mechanism_separated"
    )
    # PIC channels are surfaced only as candidate runtime channels.
    for ch in packet["candidate_runtime_channels"]:
        assert ch.startswith("candidate_")


def test_cross_scope_target_cannot_accept_pf1000_akel_neutron_authority() -> None:
    """An accepted target from a different scope must be rejected for Akel authority."""
    packet = build_mechanism_separated_neutron_packet(
        declared_scope=_AKEL,
        device_name="PF-1000/Akel",
        validation_targets=[
            {
                "name": "pf1000_full_energy_anisotropy",
                "observable": "neutron_anisotropy",
                "status": "accepted_same_scope_source",
                "declared_scope": "pf1000_full_energy_450kj_3p5torr",
            }
        ],
    )
    decisions = packet["validation_target_scope_decisions"]
    assert any(
        d["decision"] == "rejected_missing_or_mismatched_scope_metadata"
        for d in decisions
    )
    assert "neutron_anisotropy_angular_yield" not in packet["accepted_channels"]
    assert packet["cross_scope_policy"]["can_use_other_scope_for_acceptance"] is False
    assert packet["can_support_first_principles_acceptance"] is False


def test_detector_response_and_uq_required_before_kinetic_yield_authority() -> None:
    """kinetic_neutron_yield_authority_status must block when detector-response or UQ
    evidence is missing, even if a kinetic-yield history is attached."""
    accepted_kinetic = {"passed": True, "status": "accepted"}
    accepted_mech = {"passed": True, "status": "accepted"}

    # Missing detector response -> blocked.
    no_detector = kinetic_neutron_yield_authority_status(
        kinetic_yield_evidence=accepted_kinetic,
        mechanism_evidence=accepted_mech,
        detector_response_evidence=None,
        uncertainty_evidence={"passed": True, "status": "accepted"},
    )
    assert no_detector["status"] == "blocked"
    assert "same_scope_detector_response" in no_detector["missing_evidence"]
    assert no_detector["can_support_first_principles_acceptance"] is False

    # Missing UQ -> blocked.
    no_uq = kinetic_neutron_yield_authority_status(
        kinetic_yield_evidence=accepted_kinetic,
        mechanism_evidence=accepted_mech,
        detector_response_evidence={"passed": True, "status": "accepted"},
        uncertainty_evidence=None,
    )
    assert no_uq["status"] == "blocked"
    assert "yield_uncertainty_budget" in no_uq["missing_evidence"]


def test_electron_temperature_authority_gates_kinetic_yield() -> None:
    """A Te closure that cannot support quantitative claims must block yield authority
    (hybrid-PIC KR acb71fa9:1226-1240: factor-of-a-few yield uncertainty from Te)."""
    status = kinetic_neutron_yield_authority_status(
        kinetic_yield_evidence={"passed": True, "status": "accepted"},
        mechanism_evidence={"passed": True, "status": "accepted"},
        detector_response_evidence={"passed": True, "status": "accepted"},
        uncertainty_evidence={"passed": True, "status": "accepted"},
        temperature_authority={
            "status": "blocked",
            "can_support_pressure_hall_quantitative_claims": False,
        },
    )
    assert status["status"] == "blocked"
    assert "electron_temperature_authority" in status["missing_evidence"]
    assert status["can_support_first_principles_acceptance"] is False
```

Additionally recommended (not authored here — would require non-trivial fixture
construction): a runner-level negative test asserting
`result.telemetry["neutron_authority"]["status"]` stays blocked when a deck supplies
`neutron_authority_accepted_channels` equal to all required channels. The existing
`tests/test_first_principles_runner.py:868-899` asserts the blocked status for the
default deck but does not attempt the "deck declares everything accepted" attack.

---

## (f) Negative Tests — Present vs Missing

| Negative control | Present today? | Where |
| --- | --- | --- |
| Default-deck neutron packet is blocked | PRESENT | `test_first_principles_runner.py:868-899` |
| `build_mechanism_separated_neutron_packet` source refs include required KR files | PRESENT (existence-only) | `test_first_principles_source_targets.py:259-270` |
| Reduced beam-target total yield blocked (sibling fn `first_principles_neutron_yield_authority_status`) | PRESENT | `test_first_principles_mhd.py:284-316` |
| Final-state duration approximation for thermonuclear yield blocked (sibling fn) | PRESENT | `test_first_principles_mhd.py:318-336` |
| **Scalar-total-yield-only target cannot accept `build_mechanism_separated_neutron_packet`** | **MISSING** | proposed §e test 1 |
| **All-channels-declared-accepted still fails closed** | **MISSING** | proposed §e test 2 |
| **Lee model stays text-reference, never accepted channel** | **MISSING** | proposed §e test 3 |
| **Candidate PIC yield is runtime-diagnostic only** | **MISSING** | proposed §e test 4 |
| **Cross-scope accepted target rejected for Akel neutron authority** | **MISSING** | proposed §e test 5 |
| **Detector-response packet required before kinetic yield authority** | **MISSING** | proposed §e test 6 |
| **UQ packet required before kinetic yield authority** | **MISSING** | proposed §e test 6 |
| **Te-authority gates kinetic yield authority** | **MISSING** | proposed §e test 7 |

No dedicated `tests/test_first_principles_neutron_authority.py` exists. The
mechanism-separation discrimination logic of `build_mechanism_separated_neutron_packet`
(scalar rejection, Lee rejection, cross-scope rejection, accepted-channel aliasing) has
**zero behavioral test coverage** — only existence assertions. Per the Rejection
Criteria ("Tests do not include negative controls"), WP-6 cannot be promoted past
`accept_engineering_progress` until §e is added.

---

## (g) Remaining Blockers

1. **No `tests/test_first_principles_neutron_authority.py`.** The scalar-yield-only
   rejection, Lee-reduced rejection, cross-scope rejection, and candidate-PIC
   non-promotion are untested at the function level. Add the §e suite. (Highest
   priority — required for any status above `accept_engineering_progress`.)

2. **Accept path of `build_mechanism_separated_neutron_packet` is unreachable / inert.**
   `can_support_total_yield_acceptance` and `can_support_first_principles_acceptance`
   are hardcoded `False` (lines 258-259) and `BLOCKING_NEUTRON_AUTHORITY_CHANNELS` is
   unconditionally re-added to `missing` (lines 199-200). This is maximally fail-closed
   (correct for now) but means the detector-response + UQ gate SSR-009 requires is not
   actually a *consuming* gate. Two options for WP-6 closure: (a) keep it permanently
   inert and document it as a deliberate hard block until the full mechanism stack
   exists; or (b) refactor so the packet consumes a `detector_response` packet and a
   `comparator_uq` packet and computes acceptance from declared accepted channels minus
   genuinely-missing ones — mirroring how `kinetic_neutron_yield_authority_status`
   already works. Until one is chosen and documented, WP-6 "Expected result:
   `neutron_authority` does not depend on scalar yield alone" is met, but the gate is
   not promotable.

3. **`kinetic_neutron_yield_authority_status` is not wired into the runner.** It is the
   only neutron function that actually consumes `detector_response_evidence` and
   `uncertainty_evidence` and can return `accepted`. `runner.py:1043-1051` wires only
   `build_mechanism_separated_neutron_packet`. The runtime neutron packet therefore has
   no path that exercises detector/UQ gating. Decide whether
   `kinetic_neutron_yield_authority_status` should feed the runner packet, or be
   removed/merged to avoid two divergent neutron-authority code paths.

4. **No mechanism-separated neutron *history* is produced.** `kinetic_yield.py` emits a
   single total D-D rate (`mechanism_channels=("dd_particle_distribution_total",)`,
   `mechanism_separation_status="not_mechanism_separated"`). WP-6 deliverable
   "Mechanism-separated neutron production histories" and "Beam/ion distribution
   packet" and "Spectrum and anisotropy packet" and "Detector/activation/TOF response
   packet" are **not implemented** — only the *requirement channels* exist. This is
   honestly reported (every channel is `missing_or_blocked`), so it is a scope blocker,
   not an overclaim. The simulator correctly stays fail-closed.

5. **One context citation not individually verified.** `open-access-proceedings-…-
   ed196711.md:93-141,152-190,680-697,782-805` — file exists, ranges in-bounds, role
   is a context citation only. Spot-check recommended for completeness; not a defect.

---

## Audit Conclusion

SSR-009 mechanism separation and the WP-6 "no scalar-yield-only acceptance" requirement
are **satisfied at the packet level**. All physics citations trace to local
`KnowledgeReference/` evidence and 17/18 opened ranges verified TRUE with zero
fabrications. Reduced Lee and hybrid-PIC outputs are correctly confined to
comparator/runtime-diagnostic baselines. The packet provably cannot accept on scalar
total yield. No overclaim, no hidden floors, no reduced-model authority.

The blocking gaps are test coverage (no dedicated negative-test file for the
mechanism-separation discrimination logic) and an inert/duplicated acceptance gate.
Verdict: **`accept_engineering_progress`** — honest, source-disciplined, fail-closed
engineering progress; not yet a promotable WP-6 neutron-authority gate.
