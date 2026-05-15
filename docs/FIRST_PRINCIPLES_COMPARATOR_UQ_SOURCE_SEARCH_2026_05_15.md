# First-Principles Blocker Source Search - Comparator And UQ Matrix

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` and source-truth index artifacts already in the repo.

Blocker: `FP-13`, comparator and UQ matrix.

Question: can PF-1000/Akel first-principles outputs be compared with accepted
same-scope evidence using complete metrics, tolerances, pass/fail rules, and
uncertainty today?

## Verdict

No. The local source truth supplies useful scalar uncertainty text, timing
uncertainty text, detector-response schema, numerical sensitivity examples, and
required-data queues, but it does not supply a complete accepted comparator and
UQ matrix for PF-1000/Akel.

The current accepted-contract answer is:

- Akel 2021 provides scalar neutron-yield uncertainty, channel timing
  uncertainty, detector layout, activation calibration text, and shot-series
  yield ranges.
- The 2026 hybrid PIC-fluid source provides useful examples of resolution,
  parameter, and electron-temperature sensitivity studies, but it is other-scope
  and not a PF-1000/Akel comparator packet.
- TOF tomography and PF-1000 anisotropy sources show that neutron spectrum and
  angular comparisons must be detector-forward and scatter-aware.
- The local closure queue lists required current, phase, neutron-yield,
  temperature, and uncertainty data, but those entries are partial and not
  accepted comparator-bound packets.

Therefore `FP-13` remains blocked. Every package-native first-principles run
must expose a fail-closed `comparator_uq` packet until each observable has an
accepted target, output mapping, units/coordinates, metric, tolerance,
measurement/model/numerical uncertainty, propagation method, pass/fail rule,
artifact hash, requirement link, and independent review.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:120-139` | Detector geometry, activation calibration text, scalar yield uncertainty, derivative-dip time origin, and 3-5 ns channel timing uncertainty. | No accepted target arrays, detector response function, comparator metric, or propagated uncertainty matrix. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:862-889` | Shot-series computed/measured yield comparisons, average yields, measured-yield ranges, and outlier mismatch context. | Lee-fit agreement is baseline comparison, not a first-principles comparator/UQ matrix. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1018-1040` | Example numerical resolution sensitivity: sheath trajectory spread and neutron-yield change under time/space refinement. | Other-scope 2D hybrid source; does not set PF-1000/Akel tolerances. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1042-1089,1214-1266` | Example parameter and closure sensitivity for conductivity threshold, Ohmic CFL, Marder cleaning, electron-temperature closure, and order-of-magnitude yield interpretation. | Sensitivity examples do not close accepted UQ or pass/fail rules for the package-native 3D PF-1000/Akel path. |
| `KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md:32-53` | Time-energy neutron spectrum is a strong multiphysics validation observable and requires quality measurements, TOF inversion, source time profile, detector pairs, and scatter subtraction. | Not same-scope Akel; no accepted spectrum target packet. |
| `KnowledgeReference/tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md:337-351,390-427,518-526` | Detector-forward comparison requires distance, neutron count, detector efficiency, surface area, scatter subtraction, calibration scaling, sampling rate, bandwidth, and impulse response. | No Akel detector-response packet or uncertainty propagation is accepted. |
| `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:175-204,277-284` | PF-1000 anisotropy/spectrum interpretation requires direct/scattered neutron separation and vessel/room transport. | Full-energy PF-1000 other-scope source; not accepted Akel comparator evidence. |
| `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md:529-604` | Mechanism timing, spectrum, anisotropy, and error bars are part of mechanism-sensitive comparisons. | MJOLNIR source is not Akel same-scope evidence. |
| `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md:224-289` | Local queue lists required current trace uncertainty, phase timing uncertainty, neutron-yield uncertainty, detector/transport uncertainty, temperature diagnostics, and same-shot density uncertainty. | Queue status is partial; these are not reviewed comparator/UQ artifacts. |

## Required Comparator And UQ Packet

An accepted comparator/UQ matrix needs these channels:

| Channel | PF-1000/Akel current state |
| --- | --- |
| Accepted same-scope target registry | Blocked. |
| Source hashes and review status | Partial for some draft/digitization artifacts; not complete. |
| Output field mapping by observable | Blocked. |
| Unit conversion and coordinate mapping | Blocked. |
| Time alignment policy | Partial timing text only. |
| Comparator metric by observable | Blocked. |
| Comparator tolerance by observable | Blocked. |
| Measurement uncertainty by observable | Partial scalar/timing text only. |
| Model uncertainty by observable | Blocked. |
| Numerical uncertainty by observable | Other-scope examples only. |
| Closure sensitivity uncertainty | Other-scope examples only. |
| Detector response uncertainty | Blocked for same scope. |
| Shot-to-shot uncertainty or scope rule | Partial yield-series text only. |
| UQ propagation method | Blocked. |
| Pass/fail rule by observable | Blocked. |
| Negative control cases | Blocked. |
| Requirement links | Partial in plan/RTM, not complete packet. |
| Artifact links and hashes | Partial; not complete. |
| Independent review certificate | Blocked. |

## Implementation Impact

Immediate implementation requirements:

- Emit a `comparator_uq` packet from every package-native first-principles run
  and manifest.
- Mark status `blocked_comparator_uq_matrix_not_available` until every
  observable group has accepted source evidence, output mapping, metric,
  tolerance, uncertainty, propagation, and review.
- Keep Akel scalar yield and timing uncertainty as text-supported context only.
- Keep hybrid resolution/sensitivity and neutron-detector methodology sources as
  schema and requirement material until their exact source scope is selected.
- Require negative controls proving draft, missing-UQ, cross-scope, missing
  review, and reduced-model comparison artifacts cannot enter a certificate.

Next blocker to search after this one: `FP-14`, validation certificate and
release decision.
