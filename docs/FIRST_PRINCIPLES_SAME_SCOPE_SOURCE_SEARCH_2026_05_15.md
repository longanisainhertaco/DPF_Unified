# First-Principles Blocker Source Search - Same-Scope Source Availability

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/`, including the ingested
`KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`,
and source-truth index artifacts already in the repo.

Blocker: `FP-9`, same-scope source availability decision.

Question: does the local source of truth contain enough same-scope material to
support an accepted whole-shot first-principles PF-1000/Akel demonstrator?

## Verdict

No. PF-1000/Akel remains the right engineering reference candidate, but the
local source set does not yet contain a complete accepted same-scope packet for
a whole DPF shot.

The current accepted-contract answer is:

- `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`
  supports PF-1000/Akel 16 kV reference-candidate inputs and scalar outputs:
  geometry, bank, operating pressure/gas, measured-current-shape discussion,
  shot 12581 peak current/pinch current, scalar neutron yield, neutron detector
  layout, activation counters, and timing uncertainty.
- Those Akel sources do not provide accepted digitized current traces or
  same-shot density, EM-field, electron/ion temperature, neutron spectrum,
  anisotropy, detector-response, and full uncertainty packets.
- The PF-1000 interferometry, plasma-current-sheath, and neutron-anisotropy
  papers are valuable requirement sources, but they are other-scope campaigns
  relative to Akel 16 kV shots.
- The 2026 fully electromagnetic hybrid PIC-fluid source is useful for
  architecture and closure-gap requirements. It is not PF-1000/Akel
  validation evidence and explicitly frames its 2-D axisymmetric hybrid result
  as order-of-magnitude rather than precise validation.

Therefore `FP-9` remains blocked for accepted whole-shot authority. The plan
must keep PF-1000/Akel as an engineering/reference candidate until same-scope
evidence is acquired and reviewed, or the first accepted demonstrator must be
narrowed or switched to a better-diagnosed local source scope.

## Source Answers

| Source | What it answers | Why it does not close FP-9 |
| --- | --- | --- |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:108-142` | PF-1000 electrode geometry, 16 kV / 170.5 kJ bank conditions, pressure range, current/voltage diagnostics, scintillator directions, silver activation counters, yield uncertainty, and timing uncertainty. | It is not a full accepted packet; plotted current traces and detector/response details are not target-extracted and reviewed for comparator use. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:256-333` | Shot 12581 bank/tube/operational parameters, current-fit context, peak current, pinch current, pinch dimensions, and measured neutron yield. | The current trace is figure-based and Lee-fit context; same-shot density, fields, temperatures, neutron spectrum/anisotropy, detector response, and full UQ are missing. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:862-889` | Shot-series average computed/measured yield comparison and measured-yield ranges. | This supports scalar yield comparison only; it does not provide a first-principles whole-shot validation packet. |
| `KnowledgeReference/sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md:129-131,162-176` | PF-1000 interferometry can record time-resolved density/geometry information during one discharge and correlate with x-ray/neutron diagnostics. | The cited example shot has different pressure/yield/scope from Akel 16 kV shot 12581 and is not target-extracted into accepted density arrays. |
| `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:1459-1517` | PF-1000 plasma-current-sheath structure, magnetic probes, interferograms, current-density structure, and neutron-yield context. | It is a different PF-1000 campaign with different voltage/pressure/shot conditions; it cannot be mixed into Akel same-scope validation. |
| `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:121-137,175-177,269-275` | PF-1000 vessel-scattering, Bonner/TLD/activation detector approach, anisotropy and energy-spectrum transport considerations. | It is full-scale PF-1000 at 450-500 kJ and 3.5 Torr, not the Akel 16 kV shot set. It defines detector requirements but is not same-scope evidence. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1220-1266` | Full-Maxwell hybrid PIC-fluid architecture, closure limitations, and order-of-magnitude comparison against fully kinetic results. | It is a 2-D axisymmetric, non-PF-1000/Akel source and explicitly does not supply precise same-scope validation. |

## Required Same-Scope Packet

An accepted whole-shot demonstrator needs one source scope containing these
channels:

| Channel | PF-1000/Akel current state |
| --- | --- |
| Declared validation scope | Candidate: shot 12581 / 16 kV / 1.2 Torr can be named. |
| Device geometry and electrode dimensions | Text-supported from Akel source. |
| Bank/circuit drive | Text-supported from Akel source. |
| Gas species, pressure, temperature | Species/pressure text-supported; temperature needs explicit packet. |
| Accepted digitized current waveform | Blocked. Figures are not accepted target traces. |
| Startup/breakdown/preionization | Blocked. |
| Spatial density history | Blocked for same scope. |
| EM field history | Blocked for same scope. |
| Electron temperature history | Blocked. |
| Ion temperature or ion distribution history | Blocked. |
| Neutron scalar yield | Text-supported. |
| Neutron timing history | Partially described; accepted packet blocked. |
| Neutron spectrum | Blocked for same scope. |
| Neutron anisotropy | Blocked for same scope. |
| Detector response/calibration | Partial detector description; accepted response packet blocked. |
| Uncertainty budget | Partial scalar/timing uncertainty; full propagated UQ blocked. |
| Source review certificate | Blocked. |

## Implementation Impact

Immediate implementation requirements:

- Emit a `same_scope_source` packet from every package-native first-principles
  run and manifest.
- Mark PF-1000/Akel as `blocked_same_scope_source_packet_not_available` until
  accepted same-scope evidence exists.
- Allow PF-1000/Akel scalar/circuit/current/yield text evidence to guide
  engineering probes, but never use it to promote a whole-shot claim.
- Keep other-scope PF-1000 diagnostics in requirement and schema roles unless a
  reviewer explicitly narrows the claim to their exact scope.

Next blocker to search after this one: `FP-10`, accepted waveform and phase
evidence.

## Current Implementation Ratchet

Implemented after this source search:

- `src/dpf/first_principles/same_scope.py` now emits a fail-closed same-scope
  packet with a per-channel status matrix, text-supported-but-not-accepted
  channels, cross-scope policy, reviewed-transfer-rule requirements, and
  per-target scope decisions.
- Accepted validation targets without matching declared-scope metadata are
  rejected as same-scope channels rather than silently accepted.
- `tests/test_first_principles_runner.py` proves PF-1000/Akel scalar/text
  channels remain reference-only and that cross-scope transfer evidence remains
  blocked.

Verified command:

- `python3 -m pytest tests/test_first_principles_runner.py` -> `8 passed`.

Remaining blocker:

- No accepted one-scope PF-1000/Akel packet exists for current waveform,
  startup, density, fields, temperatures, neutron timing/spectrum/anisotropy,
  detector response, propagated UQ, source review certificate, or reviewed
  transfer rule for any other-scope material.
