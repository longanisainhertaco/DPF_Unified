# First-Principles Blocker Source Search - Dimensionality And Handoff

Date: 2026-05-15

Scope: local source of truth only. Scientific claims in this note are limited to
`KnowledgeReference/` and source-truth index artifacts already in the repo.

Blocker: `FP-7`, 3D hybrid PIC-fluid dimensionality and MHD-to-kinetic handoff.

Question: can the source of truth support a full first-principles 3D DPF
simulation path, and where must MHD, hybrid PIC-fluid, or fully kinetic physics
be used?

## Verdict

The source of truth supports the finish-line direction but does not close
accepted whole-shot authority.

The corpus requires a dimensionality decision:

- A 2D or axisymmetric path cannot claim full DPF authority for intervals or
  observables controlled by cathode-bar geometry, azimuthal flow, kink,
  fragmentation, or beam formation.
- 3D MHD is source-supported for macroscopic electrode geometry and rundown
  timing, but MHD becomes physically insufficient near pinch when kinetic
  instabilities, anomalous resistivity, nonthermal ions, and beam-target
  neutron production matter.
- The newly ingested hybrid PIC-fluid source supports a target architecture:
  Maxwell fields, kinetic ions, electron-fluid Ohm closure, current
  predictor-corrector, conductivity blending, Marder correction, collisions,
  and neutron-yield history. However, the source implementation is explicitly
  axisymmetric and quasineutral, with no resolved Debye sheath microphysics,
  no separate electron-energy equation, and no 3D azimuthal modes.
- Fully kinetic DPF sources show that fluid and hybrid models can miss
  nonthermal ion tails and beam-target neutron yield, especially for lower
  current beam-dominated devices. Therefore total neutron authority requires
  either a fully kinetic interval or a reviewed kinetic/hybrid handoff with
  mechanism separation, stopping, spectrum/anisotropy, detector response, and
  UQ.

Therefore `FP-7` remains blocked for accepted full-shot first-principles
authority. The implementation can continue as a 3D hybrid EM/PIC-fluid
engineering candidate, but accepted claims must be narrowed until dimensionality
and kinetic handoff packets are complete.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:246-320` | Defines the hybrid PIC-fluid solver loop: Boris ion push, charge/current deposition, generalized Ohm/Ampere current solve, FDTD fields, Marder correction, Faraday update, predictor-corrector current, and ion-ion collisions. | The paper's implementation is not a complete 3D accepted whole-shot implementation, and source-to-code equivalence remains unreviewed. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:468-606` | Defines predictor-corrector current update and plasma-vacuum conductivity blending with Ohmic CFL limiting. | Acceptance still needs numerical-fidelity tolerances, limiter classification, and proof that the limiter does not control claimed observables. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1030-1068` | Provides refinement and sensitivity evidence for sheath motion and neutron production in the source model. | Evidence applies to the source's model scope, not automatically to the local 3D implementation or PF-1000/Akel. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1210-1230` and `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:1270-1280` | States core limitations: quasineutral electron fluid, no Debye/near-wall sheath microphysics, axisymmetric geometry with only `m = 0`, no m=1 kink or higher azimuthal modes, and future work to extend to 3D and improve electron temperature treatment. | This blocks treating the source model as already full 3D whole-shot authority. |
| `KnowledgeReference/fully-three-dimensional-simulation-and-modeling-of-a-dense-plasma-focus.md:352-380` | States MHD becomes unphysical near pinch and can transfer state to PIC; also states gas breakdown is not covered by MHD. | Requires an explicit handoff boundary, transferred fields/state, and kinetic/PIC acceptance packet. |
| `KnowledgeReference/fully-three-dimensional-simulation-and-modeling-of-a-dense-plasma-focus.md:474-546` and `KnowledgeReference/fully-three-dimensional-simulation-and-modeling-of-a-dense-plasma-focus.md:614-624` | Shows 3D MHD can represent cathode-bar geometry and improves rundown-time prediction compared with 2D/axisymmetric models. | 3D MHD alone does not close pinch, beam, spectrum, anisotropy, or total neutron authority. |
| `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:34-80` | Shows fully kinetic simulations reproduce high-energy ion beams and experimental neutron yields where fluid and hybrid models underpredict or miss nonthermal ions. | The local implementation is hybrid PIC-fluid, not fully kinetic electron-ion PIC, so beam-target authority remains blocked unless claim scope is narrowed or a kinetic handoff is added. |
| `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md:152-168` | Separates thermonuclear and beam-target neutron mechanisms and shows fully kinetic simulations reproduce yield/beam behavior not captured by fluid and hybrid models. | Requires mechanism-separated neutron history, ion distribution transport/stopping, detector response, and UQ. |
| `KnowledgeReference/comparisons-of-dense-plasma-focus-kinetic-simulations-with-experimental-measurements.md:126-200` | Supports comparing fully kinetic simulations to measured yields, ion energy distributions, RF fluctuations, driver effects, and electrode geometry. | This is a different LLNL-scale device scope, not direct PF-1000/Akel same-scope validation. |

## Accepted Dimensionality Contract

The simulator needs an explicit dimensionality/handoff packet with one of these
claim modes:

| Mode | Status | Allowed claim |
| --- | --- | --- |
| `bounded_axisymmetric_mhd_claim` | Interim only | Stops before MHD breakdown and excludes 3D/kinetic observables. |
| `validated_3d_mhd_rundown_claim` | Requires review | Can claim macroscopic 3D rundown/electrode-geometry behavior only, not kinetic pinch or beam-target yield. |
| `mhd_to_kinetic_handoff_claim` | Required if MHD covers early shot and PIC covers pinch | Needs transferred density, velocity, pressure/temperature, `E`, `B`, current, charge, species, boundaries, and conservation checks. |
| `validated_3d_hybrid_pic_fluid_claim` | Target engineering path, still blocked for acceptance | Needs full Maxwell plasma/vacuum fields, kinetic ion PIC push/deposition, generalized Ohm current closure, predictor-corrector, divergence control, conductivity blend, boundaries, collisions, electron-energy closure, and kinetic yield history. |
| `fully_kinetic_pinch_claim` | Required for unrestricted beam-target authority if hybrid limitations dominate | Needs kinetic electrons and ions, resolved instability/beam formation, stopping, spectrum/anisotropy, detector response, and UQ. |

## Implementation Impact

Immediate implementation requirements:

- Keep the package-native 3D hybrid EM/PIC-fluid runner as an engineering
  candidate until it emits a dimensionality/handoff packet.
- Record whether the active claim is axisymmetric, full 3D MHD, 3D hybrid
  PIC-fluid, MHD-to-kinetic handoff, or fully kinetic pinch.
- Require explicit blocking fields for source-model limitations: no Debye
  sheath, quasineutral electron fluid, no separate electron energy, no kinetic
  electrons, no accepted m-mode evidence, no same-scope 3D validation packet.
- Keep total neutron-yield authority blocked unless thermonuclear and
  beam-target mechanisms are separated and the kinetic interval has accepted
  distribution/stopping/detector/UQ evidence.
- Add tests proving a 3D runner can be present while whole-shot authority still
  remains blocked.

Next blocker to search after this one: `FP-8`, physics closure packets.

## Current Implementation Ratchet

Implemented after this source search:

- `src/dpf/first_principles/dimensionality.py` now emits a fail-closed
  dimensionality/handoff packet with explicit claim modes, allowed-claim text,
  source-model limitations, handoff-required observables, candidate runtime
  channels, and missing acceptance channels.
- `src/dpf/first_principles/runner.py` now feeds package-native simulation
  telemetry into that packet so 3D grid, hybrid PIC-fluid runtime, electron
  energy scaffold, source-ordered loop, and kinetic-yield history remain
  visible as candidate-only channels.
- `tests/test_first_principles_runner.py` proves the runner can expose a 3D
  hybrid candidate while still blocking unrestricted whole-shot dimensionality
  authority.

Verified command:

- `python3 -m pytest tests/test_first_principles_runner.py` -> `8 passed`.

Remaining blocker:

- No accepted source-equivalence review, same-scope 3D validation packet,
  MHD-to-kinetic handoff state, kinetic-electron or bounded-out electron
  kinetics packet, fully kinetic/reviewed beam-target interval, or
  mechanism-separated detector/UQ packet exists.
