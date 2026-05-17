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
  ion PIC push/deposition, candidate electron-energy plumbing, candidate
  deuterium ionization/recombination transport, and candidate weakly ionized
  conductivity can be source-backed as engineering closures.
- EOS/thermodynamics, accepted ionization/charge-state authority, accepted
  electron heat-flux authority, collisional coupling, radiation losses,
  impurity/electrode ablation,
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
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1210-1280` | States limitations: scalar electron pressure, simplified collisional conductivity, no pressure anisotropy, no anomalous/turbulent transport, quasineutrality, no Debye/near-wall sheaths, axisymmetry, and `Te = Ti` limitations for Hall/pressure/yield. | Accepted separate electron-energy, heat-flux, collisional-coupling, kinetic-electron, 3D-mode, and near-wall authority remains blocked. |
| `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json:57-62` | PF-1000 MHD source evolves separate `Te`/`Ti`, includes electron and ion heat-flux terms, electron-ion exchange, Joule heating, ionization/radiation losses, and states no-normal heat-flux/temperature-gradient boundary conditions for pinch/electrode/axis surfaces. | It is 2D non-ideal MHD evidence, not an accepted 3D hybrid PIC closure packet; coefficients, implementation mapping, diagnostics, UQ, and review still gate promotion. |
| `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md:252-259` | Provides the PF-1000 ionization equation structure: ground-state electron-impact ionization, radiative recombination, and three-body recombination. | It does not by itself provide an accepted 3D hybrid charge-state packet, neutral-particle coupling review, or conductivity/EOS feedback authority. |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:4572-4648` | Provides the NRL charge-state rate equation form, ground-state ionization rate, radiative recombination, and three-body recombination support. | The local implementation is single-stage deuterium candidate transport only; molecular D2, excited states, impurities, review, and UQ remain blocked. |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3379-3425` | Provides weakly ionized collision and conductivity structure, including electron-neutral scattering and `sigma = n e mu`. | The runtime uses a candidate scalar conductivity; magnetized tensor transport and reviewed deuterium cross-section data remain blocked. |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2996-3020` | Provides the NRL thermal-equilibration form for different-temperature plasma components and the equal-`Te/Ti` electron-ion special case. | The runtime now has an audit/reference frequency, but accepted arbitrary-`Te/Ti` collisional coupling still needs a line-by-line symbol map, regime validity, and same-scope diagnostics/UQ. |
| `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:333-369` | Lists high-fidelity DPF modeling capabilities: EOS, transport/conductivity, two-temperature physics, emission/radiation, material strength/fracture, neutron diagnostics, and output diagnostics. It also documents low-density deuterium EOS consistency problems and QEOS use. | The local runner has no accepted QEOS/tabular EOS, radiation transport, material model, or PF-1000/Akel closure packet. |
| `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md:277-293` | Shows why pinch physics and neutron production need nonthermal/beam-target and instability treatment beyond MHD. | Closure packets for anomalous resistivity, beam formation, beam-target yield, hot spots, and post-pinch physics remain missing. |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md` | Provides formulary support for plasma parameters, collisions, transport, radiation constants, fusion cross sections, and units. | It is general formulary support, not same-scope DPF closure validation. Equations still need symbol mapping, implementation tests, and validity checks. |

## Required Closure Packet Matrix

| Effect | Current answer | Accepted packet requirement |
| --- | --- | --- |
| EOS and thermodynamics | Missing or ideal-gas candidate only | Source equations or tables, density/temperature validity, low-density behavior, units, and tests. |
| Ionization and charge state | Candidate D/D+ ionization, recombination, charge-state transport, and PIC particle source/sink are wired; not accepted | Accepted ionization/recombination packet, startup link, particle-source review, conductivity/EOS feedback authority, tests and UQ. |
| Single/two-temperature energy | Candidate separate electron-energy wrapper plus candidate Braginskii heat-flux channel | Accepted electron heat flux, electron-ion exchange audit, pressure/Hall coupling, diagnostics, sensitivity/UQ. |
| Electrical/thermal transport | Candidate weakly ionized scalar conductivity, candidate conductivity blend fallback, Ohm solver, and heat flux | Resistivity/conductivity validity, magnetized tensor decision, limiter nondominance, convergence, transport uncertainty. |
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
- The closure packet now emits `closure_effect_status`, per-effect
  `channel_status`, `active_closure_policy`, `dimensionality_acceptance_gate`,
  `acceptance_gate`, and `negative_test_policy`.
- Candidate Ohm/transport/electron-energy/Hall/instability/yield closures are
  explicitly engineering-only until every active or bounded-out effect has
  source equations, symbol maps, units, validity regimes, verification tests,
  sensitivity/UQ, claim impact, artifact hashes, and review.
- `src/dpf/fields/electron_energy.py` now applies a candidate Braginskii
  anisotropic electron heat-flux update when the 3-D loop supplies
  cell-centered magnetic field. The finite-volume update uses zero-normal-flux
  boundary handling to match the PF-1000 MHD source boundary language and emits
  `candidate_braginskii_anisotropic_heat_flux_applied` telemetry. It remains
  non-promoting.
- `src/dpf/fluid/two_temperature.py` now exposes an NRL equal-temperature
  electron-ion thermal-equilibration reference frequency and a
  `candidate_nrl_equal_temperature_equilibration_audit` telemetry packet. This
  compares the active arbitrary-`Te/Ti` Spitzer mass-ratio relaxation rate to
  the local NRL special case without promoting collisional coupling.
- `src/dpf/first_principles/closure_packet.py` now reports
  `candidate_braginskii_electron_heat_flux` as a runtime channel when applied,
  and `candidate_electron_ion_equilibration_audit` when the NRL audit is
  present, while still listing `accepted_electron_heat_flux` and
  `accepted_electron_ion_collisional_coupling` as missing for FP-8 acceptance.
- `src/dpf/fields/ionization_transport.py` now provides a source-backed
  candidate single-stage deuterium chemistry state. It advances neutral density,
  D+ density, electron density, and mean charge state using local PF-1000/NRL
  ionization, radiative recombination, and three-body recombination structure.
  The 3-D loop can also convert ionization/recombination deltas into candidate
  PIC macroparticle source/sink weight for the next deposit.
- `src/dpf/fields/conductivity.py` now provides candidate partial-ionized
  scalar conductivity using Spitzer electron-ion resistivity plus NRL
  weakly-ionized electron-neutral drag. The first-principles runner enables
  this path and bypasses the older density-threshold plasma-vacuum blend for
  the source-backed transport route.
- The ionization/transport runtime remains non-promoting and does not use the
  empirical `src/dpf/fluid/ionization.py` coronal-fit helper as authority.
- `src/dpf/first_principles/runner.py` already places the closure packet in run
  telemetry, manifest candidate evidence, neutron authority, numerical,
  comparator/UQ, certificate, and generalization upstream maps.
- `tests/test_first_principles_runner.py` proves the packet keeps EOS,
  radiation, accepted electron-energy authority, and total-yield closure
  blocked while exposing electron-energy and kinetic-yield scaffolds as
  candidate-only runtime channels.

Verified command:

- `python3 -m pytest tests/test_first_principles_runner.py` -> `8 passed`.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_cli_backend_options.py tests/test_server_readiness.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_pic_3d_validation_packet.py`
  -> `60 passed`.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_cli_backend_options.py tests/test_server_readiness.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_3d_simulator.py tests/test_hybrid_pic_3d_validation_packet.py tests/test_maxwell_3d_field_core.py tests/test_particle_boundaries.py tests/test_unreviewed_physics_metadata.py tests/test_two_temperature.py tests/test_formulary_transport_audit.py tests/test_physics.py::TestBraginskiiKappaZDependent`
  -> `137 passed`.
- `python3 -m pytest tests/test_first_principles_input_deck.py tests/test_first_principles_runner.py tests/test_first_principles_manifest.py tests/test_cli_first_principles_3d.py tests/test_cli_backend_options.py tests/test_server_readiness.py tests/test_kinetic_yield_history.py tests/test_hybrid_3d_loop.py tests/test_hybrid_3d_simulator.py tests/test_hybrid_pic_3d_validation_packet.py tests/test_maxwell_3d_field_core.py tests/test_particle_boundaries.py tests/test_unreviewed_physics_metadata.py tests/test_two_temperature.py tests/test_formulary_transport_audit.py tests/test_conductivity_blend.py tests/test_ionization_transport.py tests/test_physics.py::TestBraginskiiKappaZDependent`
  -> `147 passed`.

Remaining blocker:

- No accepted closure packet exists for EOS/tabular thermodynamics, accepted
  ionization/charge-state transport, accepted weakly ionized/tensor transport,
  accepted electron heat flux/collisional exchange, radiation,
  impurity/electrode ablation, anomalous resistance/restrike, Hall/FLR/kinetic
  scope, 3D-instability evidence, or beam-target coupling.
