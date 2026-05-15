# First-Principles Blocker Source Search - Physics Closures

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` and source-truth index artifacts already in the repo.

Blocker: `FP-8`, physics closure packets.

Question: can the source of truth close the required physics closures for a
whole-shot first-principles DPF simulator?

## Verdict

The source of truth provides enough material to define the required closure
packets and several engineering-candidate closures, but it does not close an
accepted whole-shot physics-fidelity packet.

The current accepted-contract answer is:

- Conductivity, generalized Ohm, predictor-corrector current, Marder cleaning,
  ion PIC push/deposition, and candidate electron-energy plumbing can be
  source-backed as engineering closures.
- EOS/thermodynamics, ionization and charge-state kinetics, electron heat flux,
  collisional coupling, radiation losses, impurity/electrode ablation,
  anomalous resistivity, restrike, and beam-target coupling still require
  source equations, validity regimes, tests, sensitivity/UQ, and claim-impact
  decisions before an accepted PF-1000/Akel certificate.
- The hybrid PIC-fluid source explicitly says its scalar electron pressure,
  simplified collisional conductivity, quasineutrality, two-dimensional
  axisymmetric geometry, and `Te = Ti` electron-temperature approximation limit
  quantitative Hall/pressure/yield authority.
- ALEGRA material shows that high-fidelity DPF modeling needs careful EOS,
  transport, two-temperature, radiation, material, and diagnostic choices; it
  also shows deuterium EOS can be a stability blocker in low-density DPF gas.

Therefore `FP-8` remains blocked for accepted whole-shot first-principles
authority. The package-native runner must emit closure packets and fail closed
until every active or bounded-out effect has source, units, validity,
verification, sensitivity/UQ, and claim-impact metadata.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:431-619` | Provides source equations for current predictor-corrector, conductivity, Coulomb-log collision frequency, plasma-vacuum conductivity blending, Ohmic CFL limiting, Yee grid, PML/conductor, and particle absorption. | Accepted proof that these closures are valid and nondominant in the local 3D runner and PF-1000/Akel scope. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1210-1280` | States limitations: scalar electron pressure, simplified collisional conductivity, no pressure anisotropy, no anomalous/turbulent transport, quasineutrality, no Debye/near-wall sheaths, axisymmetry, and `Te = Ti` limitations for Hall/pressure/yield. | Separate electron-energy equation, heat flux, collisional coupling, kinetic electron effects, 3D modes, and near-wall physics remain blocked. |
| `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:333-369` | Lists high-fidelity DPF modeling capabilities: EOS, transport/conductivity, two-temperature physics, emission/radiation, material strength/fracture, neutron diagnostics, and output diagnostics. It also documents low-density deuterium EOS consistency problems and QEOS use. | The local runner has no accepted QEOS/tabular EOS, radiation transport, material model, or PF-1000/Akel closure packet. |
| `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:277-293` | Shows why pinch physics and neutron production need nonthermal/beam-target and instability treatment beyond MHD. | Closure packets for anomalous resistivity, beam formation, beam-target yield, hot spots, and post-pinch physics remain missing. |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md` | Provides formulary support for plasma parameters, collisions, transport, radiation constants, fusion cross sections, and units. | It is general formulary support, not same-scope DPF closure validation. Equations still need symbol mapping, implementation tests, and validity checks. |

## Required Closure Packet Matrix

| Effect | Current answer | Accepted packet requirement |
| --- | --- | --- |
| EOS and thermodynamics | Missing or ideal-gas candidate only | Source equations or tables, density/temperature validity, low-density behavior, units, and tests. |
| Ionization and charge state | Startup-dependent, not accepted | Ionization/recombination or bounded charge-state model, effect on pressure and conductivity, tests and UQ. |
| Single/two-temperature energy | Candidate separate electron-energy wrapper | Electron heat flux, electron-ion exchange, pressure/Hall coupling, diagnostics, sensitivity/UQ. |
| Electrical/thermal transport | Candidate conductivity blend and Ohm solver | Resistivity/conductivity validity, limiter nondominance, convergence, transport uncertainty. |
| Radiation losses | Not accepted | Loss model or bounded contribution in energy ledger; opacity/diffusion decision by gas/material regime. |
| Impurity/electrode ablation | Not accepted | Material source model or explicit bound for waveform, pinch, radiation, and neutron observables. |
| Hall/FLR/kinetic scope | Candidate Hall/pressure path; kinetic ions present | Validity regime, electron-temperature authority, FLR/kinetic handoff, sensitivity/UQ. |
| 3D instabilities | Geometry exists; m-mode evidence not accepted | Claim interval excludes them or 3D instability evidence is accepted. |
| Restrike/anomalous resistance | Not accepted | Post-pinch/restrike/anomalous-resistivity model or claim exclusion. |
| Beam-target coupling | Candidate ion distribution/yield history only | Mechanism-separated production, stopping, spectrum/anisotropy, detector response, UQ. |

## Implementation Impact

Immediate implementation requirements:

- Emit a package-native physics-closure packet from the runner and manifest.
- Mark every effect as `candidate`, `blocked`, `bounded_out`, or `not_present`;
  do not let absent closures disappear from artifacts.
- Keep Hall/pressure and total-yield claims blocked without accepted electron
  temperature and kinetic mechanism packets.
- Require closure packets before any run can be promoted beyond engineering
  candidate.

Next blocker to search after this one: `FP-9`, same-scope source availability
decision.

## Current Implementation Ratchet

Implemented after this source search:

- `src/dpf/first_principles/closure_packet.py` now emits a fail-closed closure
  matrix with required packet channels for every effect, per-effect
  classifications, missing channels, review status, claim impact, active
  candidate closures, source-model limitations, and candidate runtime channels.
- `src/dpf/first_principles/runner.py` already places the closure packet in run
  telemetry, manifest candidate evidence, neutron authority, numerical,
  comparator/UQ, certificate, and generalization upstream maps.
- `tests/test_first_principles_runner.py` proves the packet keeps EOS,
  radiation, accepted electron-energy authority, and total-yield closure
  blocked while exposing electron-energy and kinetic-yield scaffolds as
  candidate-only runtime channels.

Verified command:

- `python3 -m pytest tests/test_first_principles_runner.py` -> `8 passed`.

Remaining blocker:

- No accepted closure packet exists for EOS/tabular thermodynamics, ionization
  and charge-state transport, electron heat flux/collisional exchange,
  radiation, impurity/electrode ablation, anomalous resistance/restrike,
  Hall/FLR/kinetic scope, 3D-instability evidence, or beam-target coupling.
