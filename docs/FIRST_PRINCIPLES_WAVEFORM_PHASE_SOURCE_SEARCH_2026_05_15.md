# First-Principles Blocker Source Search - Waveform And Phase Evidence

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` and source-truth index artifacts already in the repo.

Blocker: `FP-10`, accepted waveform and phase evidence.

Question: can PF-1000/Akel current-waveform, current-dip, and phase-timing
evidence support accepted first-principles comparators today?

## Verdict

No. The Akel/PF-1000 source supports waveform and phase context, but accepted
waveform and phase comparators remain blocked.

The current accepted-contract answer is:

- Akel 2021 states that voltage, current-derivative waveforms, and total
  current traces were measured at the main collector plate.
- It defines the maximum current-derivative dip as `t = 0`, gives the
  breakdown-to-dip interval, gives constriction and secondary-plasmoid timing
  context, and reports channel timing uncertainty.
- For shot 12581, the paper gives bank/tube/operating parameters, uses the
  measured current waveform as the basis for a Lee fit, says Fig. 1 is good
  through the end of the current dip, and reports peak current, pinch current,
  pinch dimensions, and pinch duration.
- The local queue says Akel Fig. 1 is cropped and has a draft packet, but the
  current state remains `blocked_by_review`; Akel Fig. 2-4 waveform packets and
  phase targets are not accepted.
- The draft Fig. 1 packet contains candidate measured/computed current arrays
  and internal overlay residual metadata, but its own acceptance boundary says
  it is not accepted waveform evidence until overlay residuals and independent
  review are supplied.

Therefore `FP-10` remains blocked for accepted whole-shot authority. The runner
must expose the supported text context while preventing any first-principles
certificate from using waveform, current-dip, or phase-timing comparators until
accepted digitized traces, phase targets, uncertainty, and review metadata are
attached.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:108-142` | Confirms measured voltage/current-derivative/current traces, derivative-dip time origin, breakdown-to-dip interval, constriction timing, secondary plasmoid timing, and timing uncertainty. | It does not provide accepted digitized waveform arrays, per-point current uncertainty, or current-dip depth. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:218-295` | Explains Lee fitting against measured current traces, axial/radial current-fit context, shot 12581 inputs, fit through current dip, peak current, pinch current, pinch dimensions, and pinch duration. | The phase information is Lee-fit/scalar context, not an accepted first-principles phase target packet. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:318-333` | Confirms computed and measured current waveforms are in Figs. 1-4 and Table 1 lists peak current and pinch current. | Figure curves are not accepted target-extracted comparator arrays. |
| `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md:146-172` | Tracks Akel Fig. 1 as the priority current-waveform digitization task, identifies the local crop/draft packet path, and reports candidate point counts plus internal overlay residuals. | The packet is `draft_unreviewed` and currently not accepted for validation use. |
| `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md:198-215` | States the Akel Fig. 1 waveform digitization candidate reports `blocked_by_review` and lists review blockers. | No simulation waveform may be compared against the draft trace for accepted claims. |
| `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md:224-249` | Identifies current-waveform and phase-timing evidence needs: digitized current and derivative traces plus per-point current/timing uncertainty and per-shot phase timing uncertainty. | Same-device phase targets remain absent. |
| `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json:710-787` | Records draft Fig. 1 extraction metadata: candidate point counts, source hashes, overlay residuals, independent review count, and `review_status="draft"`. | Its acceptance boundary explicitly says it is draft output only and not accepted digitized waveform evidence. |
| `docs/DPF_REQUIREMENTS_BASELINE.md:75-79` | Requires accepted digitization review and blocks Akel Fig. 1 until independent review and `review_status="accepted"`. | Akel Fig. 2-4 and phase packets are still planned or blocked. |

## Required Waveform And Phase Packet

An accepted first-principles current/phase comparator needs these channels:

| Channel | PF-1000/Akel current state |
| --- | --- |
| Accepted digitized current waveform | Blocked by review. |
| Accepted current-derivative or dip trace | Blocked. |
| Time-axis calibration | Draft/queue only; not accepted. |
| Current-axis calibration | Draft/queue only; not accepted. |
| Per-point waveform uncertainty | Blocked. |
| Figure/source hashes | Present in draft packet; not accepted. |
| Independent review accepted | Blocked. |
| Breakdown-to-derivative-dip timing | Text-supported. |
| Derivative-dip `t = 0` definition | Text-supported. |
| Current-dip timing/depth | Timing context text-supported; depth blocked without traces. |
| Axial/radial/pinch phase timing | Partial Lee-fit/scalar context only; accepted phase packet blocked. |
| Phase semantics | Partial; needs typed packet. |
| Production output mapping | Blocked for accepted claim. |
| Comparator metric and tolerance | Blocked until accepted evidence/UQ exists. |
| UQ budget | Blocked beyond limited scalar/timing text. |

## Implementation Impact

Immediate implementation requirements:

- Emit a `waveform_phase` packet from every package-native first-principles run
  and manifest.
- Keep status `blocked_waveform_phase_packet_not_available` until accepted
  Akel waveform/derivative/phase packets exist.
- Allow Akel text-supported timing and scalar current context to guide
  engineering probes, but never use draft or reconstructed traces for accepted
  first-principles validation.
- Require accepted comparator mapping and UQ before waveform, current-dip, or
  phase-timing evidence can affect a certificate.

Next blocker to search after this one: `FP-11`, accepted spatial, field, and
temperature evidence.

## Current Implementation Ratchet

Implemented after this source search:

- `src/dpf/first_principles/waveform_phase.py` now emits a fail-closed
  waveform/phase packet with per-channel status, text-supported-but-not-accepted
  fields, required review channels, draft Akel Fig. 1 digitization packet
  status, and validation-target scope decisions.
- The packet now emits `waveform_phase_target_policy` and
  `negative_test_policy` so draft digitization, text timing/scalars, missing
  per-point UQ, missing review, mismatched scope, and missing output
  mapping/tolerances cannot be promoted.
- Accepted target channels require matching declared-scope metadata; draft,
  text, or mismatched-scope waveform evidence is not promoted.
- `tests/test_first_principles_runner.py` proves the package-native runner keeps
  Akel Fig. 1 draft evidence non-accepting and blocks current waveform,
  derivative/current-dip, phase, comparator, and UQ authority.

Verified command:

- `python3 -m pytest tests/test_first_principles_runner.py` -> `8 passed`.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_cli_backend_options.py tests/test_server_readiness.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_pic_3d_validation_packet.py`
  -> `60 passed`.

Remaining blocker:

- No accepted digitized current waveform, current-derivative trace, per-point
  waveform uncertainty, independent review, current-dip depth, typed phase
  packet, output mapping, comparator tolerance, or waveform/phase UQ packet
  exists for the declared PF-1000/Akel scope.
