# First-Principles 3D Hybrid PIC Review

Date: 2026-05-14

Source: `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`

Status: source-architecture review only. The source is locally ingested and
user-validated for source-of-truth use, but its LLNL-like geometry,
sheath-front comparison, cross-section fit, and neutron-yield values are not
accepted validation targets until typed same-scope target packets are extracted
and reviewed.

## What The Source Changes

The first-principles finish line is not "better 2D MHD." The source describes a
fully electromagnetic hybrid ion-PIC/electron-fluid architecture: ions are
macroparticles, electrons are a quasi-neutral fluid, current is closed through a
generalized Ohm law, and Maxwell fields are evolved in both plasma and vacuum.
The source also explicitly says its demonstrated implementation is 2D
axisymmetric and misses 3D kink and higher azimuthal modes.

This makes the project target sharper:

- keep the current Python cylindrical MHD path as an engineering ratchet and
  benchmark scaffold;
- build the accepted finish-line core as a 3D hybrid PIC-fluid field-particle
  loop;
- use reduced Lee/snowplow and 2D MHD outputs only as baseline/comparison or
  startup scaffolding until the 3D hybrid core has its own evidence packets.

## Source-Derived Capabilities

| Capability | Local source lines | Application to this repo |
| --- | --- | --- |
| Full Maxwell fields in plasma and vacuum | 150-173, 186-207, 609-618, 1243-1247 | Replace first-principles acceptance dependence on 2D resistive MHD with a 3D Yee/FDTD or equivalent full-EM field solve, including conductor and PML boundaries. |
| Kinetic ion PIC push/deposition | 210-236, 246-311, 633-639 | Promote `src/dpf/experimental/pic/hybrid.py` from utility/beam infrastructure into a self-consistent ion-current authority for DPF runs. |
| Electron-fluid generalized Ohm solver | 200-209, 325-408, 1107-1185 | Implement resistive, electron-pressure, and Hall terms as gated closure components with per-term diagnostics and source-backed validity flags. |
| End-of-step current predictor-corrector | 431-532, 561-562 | Add a current predictor-corrector to avoid pushing particles against stale current/electric-field closure. |
| Gauss-law or Marder-style divergence control | 410-425, 1067-1073 | Add a Maxwell/PIC divergence-control packet, with tests showing it does not dominate sheath motion or energy accounting. |
| Plasma-vacuum conductivity blending | 563-606, 1050-1066 | Replace ad hoc vacuum/MHD conductivity behavior with a 3D hybrid conductivity model and explicit stability diagnostics. |
| PML, conductor, and particle-boundary semantics | 613-619, 625-628 | Make field boundaries and particle deletion/absorption part of the geometry contract, not hidden solver behavior. |
| Ion collision operator | 310-311 | Verify Nanbu/Perez-style ion collisions in the accepted DPF particle loop, not only in isolated PIC utilities. |
| True 3D dimensionality | 1215-1225, 1274-1278 | Require 3D azimuthal modes before any final "full first-principles DPF" claim. Current 2D axisymmetric paths cannot capture m=1 kink or fragmentation. |
| Separate electron-energy closure | 1074-1097, 1226-1240 | Treat `Te = Ti` as a blocker for quantitative Hall/pressure and neutron-yield authority; add a separate electron energy equation and heat-flux/collisional coupling. |
| Kinetic ion neutron-yield history | 952-963, 1083-1089, 1259-1266 | Compute time-resolved yield from resolved ion distributions and mechanism-separated histories; do not accept terminal scalar yield fits. |
| Same-scope 3D validation packet | 942-951, 974-991, 1215-1225, 1259-1266 | The source's comparison is order-of-magnitude and 2D/non-hollow; use it for architecture, not as validation closure. |

## Code Mapping

Existing useful hooks:

- `src/dpf/fields/maxwell_3d.py`: first isolated 3D Yee/CT full-Maxwell field
  component with edge-centered `E`, face-centered `B`, Ampere/Faraday updates,
  conductor masks, deterministic PML damping metadata, EM energy diagnostics,
  and `div B` diagnostics. This is engineering component evidence, not a
  complete accepted DPF field-particle-current loop.
- `src/dpf/fields/pic_coupling.py`: candidate bridge from cell-centered PIC
  current deposition to Yee edge current density for the Maxwell Ampere update,
  with continuity telemetry that remains nonaccepting until a reviewed
  charge-conservation packet exists.
- `src/dpf/fields/ohm_solver.py`: candidate generalized Ohm-Ampere cell solver
  using the source algebraic form for resistive/Hall current and
  density-thresholded electron-pressure contribution. It is not yet accepted
  because it is not integrated into the full field-particle loop and pressure/
  Hall terms still require separate electron-energy closure.
- `src/dpf/fields/predictor_corrector.py`: candidate current extrapolation and
  end-step Ohm correction primitive. It follows the source predictor-corrector
  algebra but does not yet perform the provisional ion push or rebuild
  density/temperature/current from particle state.
- `src/dpf/fields/marder.py`: candidate Marder/Gauss-law electric-field
  correction with residual telemetry. It is not yet accepted because it is
  only candidate-coupled back to Yee edges and has not been bounded as
  nondominant in DPF runs.
- `src/dpf/fields/conductivity.py`: candidate source-derived
  plasma-vacuum conductivity transition with Ohmic CFL limiting. It is not yet
  accepted because it has only candidate loop integration and lacks a
  DPF sensitivity packet showing the limiter is weakly active.
- `src/dpf/fields/hybrid_stepper.py`: candidate one-step field-current
  integration tying the Yee Maxwell state, conductivity blend, generalized Ohm
  current solve, current edge mapping, Maxwell advance, and optional end-step
  predictor-corrector current solve together. It still lacks the source's full
  provisional ion-push/rebuild sequence and same-scope DPF validation.
- `src/dpf/fields/hybrid_loop.py`: candidate particle-field loop step tying the
  Maxwell state to HybridPIC push/deposit, quasi-neutral electron-density
  rebuild, ion-current deposition, and field-current advance. It is not yet an
  accepted DPF loop because nondominance gates, long-run conservation,
  electron energy, kinetic yield, and same-scope validation remain missing.
- `src/dpf/fields/particle_boundaries.py`: candidate conductor/PML particle
  absorption hook. It deletes particles entering conductor/PML regions before
  deposition when configured in `HybridPIC3DLoop`, but remains nonaccepting
  because electrode geometry, face-specific boundary metadata, and same-scope
  boundary validation are not closed.
- `src/dpf/experimental/pic/hybrid.py`: 3D particle push, CIC/Esirkepov
  deposition, interpolation, collisions, beam injection. This is the closest
  local foundation for the kinetic-ion part. `HybridPIC3DLoop` now reports
  whether this path is using disabled, fallback, or Nanbu/Perez-enabled
  collisions.
- `src/dpf/fields/electron_energy.py`: candidate separate electron-energy
  source update using the repo two-temperature scaffold. `HybridPIC3DLoop` can
  use a supplied electron-energy state to build the pressure-gradient term and
  then update `Te` from the solved current, but this remains nonaccepting until
  heat-flux/collisional coupling, diagnostics, UQ, and same-scope validation
  are closed.
- `src/dpf/fields/kinetic_yield.py`: candidate time-history accumulator for
  D-D neutron yield from PIC ion distributions. `HybridPIC3DLoop` can attach
  instantaneous rate and cumulative neutrons to loop telemetry, but detector
  response, mechanism separation, angular/spectral diagnostics, and UQ remain
  unclosed.
- `src/dpf/fields/hybrid_simulator.py`: candidate multi-step 3D hybrid
  PIC-fluid driver. It repeatedly advances the loop while carrying field,
  particle, optional electron-energy, and yield state forward, but it is still
  an engineering smoke surface rather than accepted DPF predictive authority.
- `src/dpf/fields/source_geometry.py`: typed source-geometry packet for the
  local LLNL-like axisymmetric setup. It records the source values and exposes a
  Cartesian smoke-grid projection, but it is blocked as same-scope true-3D
  validation evidence.
- `src/dpf/diagnostics/pic_yield.py`: particle-distribution D-D yield-rate
  integration candidate.
- `src/dpf/validation/first_principles_mhd.py`: fail-closed readiness surface
  that now carries `hybrid_pic_3d_first_principles_core`.
- `src/dpf/fluid/cylindrical_mhd.py`: current 2D MHD ratchet and numerical
  methods; keep as scaffold/comparator, not final 3D authority.
- `src/dpf/metal/mlx_engine.py` and related Metal/MLX paths: possible future 3D
  performance backend, but still outside first-principles acceptance until
  backend-native telemetry and parity evidence exist.

New gate added:

- `src/dpf/validation/hybrid_pic_3d.py` defines the source-derived 3D hybrid
  PIC-fluid capability list and returns `blocked` unless every capability has
  accepted evidence and the run declares explicit 3D geometry.

## Execution Order

1. Build a minimal 3D full-EM field object: Yee layout, curl operators,
   conductor masks, PML/open boundary metadata, and energy diagnostics.
   Started as `src/dpf/fields/maxwell_3d.py`; remaining work is accepted
   plasma/electron closure, validated boundary coefficients, and integration
   into a full DPF run loop.
2. Bind the existing 3D PIC particle utilities to that field object:
   interpolation, Boris push, charge-conserving current deposition, density
   deposition, and particle-boundary handling. Started as the nonaccepting
   `PICCurrentSourcePort`, `hybrid_loop.py`, and `particle_boundaries.py`;
   remaining work is self-consistent long-run push/deposition sequencing,
   continuity control, face-specific boundary semantics, and validation.
3. Add the electron-fluid Ohm-Ampere closure as an explicit solver stage:
   resistive term first, then pressure-gradient and Hall terms behind
   independent gates. Started as `src/dpf/fields/ohm_solver.py`; remaining work
   is Yee-loop integration, predictor-corrector coupling, electron-temperature
   closure, and same-scope validation.
4. Add current predictor-corrector and divergence-control tests.
   Predictor-corrector algebra has started in
   `src/dpf/fields/predictor_corrector.py`, and candidate end-step current
   correction is reachable through `hybrid_stepper.py`/`hybrid_loop.py`;
   candidate source-ordered loop execution can now update ion velocities from
   source Eq. 7 after the field step and build a candidate provisional
   particle-current rebuild that feeds the corrected current; remaining work is
   accepted Te/Ti rebuild, conservation/nondominance evidence, and accepted
   full-loop coupling. Marder correction algebra has started in
   `src/dpf/fields/marder.py`, and candidate Yee-edge coupling is reachable
   through the field stepper with correction-size telemetry; remaining work is
   accepted nondominance against sheath/current observables in full DPF runs.
5. Add plasma-vacuum conductivity blending with limiter telemetry and
   sensitivity tests. Component blending has started in
   `src/dpf/fields/conductivity.py` and candidate field-current loop
   integration has started in `src/dpf/fields/hybrid_stepper.py`; remaining
   work is full-loop sensitivity evidence.
6. Add separate electron-temperature evolution before treating pressure/Hall
   runs as quantitative. Started as `src/dpf/fields/electron_energy.py` and
   optional `HybridPIC3DLoop` coupling; extended-Ohm temperature authority now
   blocks Hall/pressure claims when separate Te evidence is missing or only
   candidate. Remaining work is source-closed heat flux, collisional coupling
   audit, accepted diagnostics, and UQ.
7. Attach and validate loop-local ion collisions with reviewed density,
   temperature, Coulomb-log, timestep, and cell-pairing metadata before using
   collisional kinetic histories as accepted DPF evidence.
8. Attach the source external-circuit magnetic injection boundary. Started as
   `src/dpf/fields/circuit_boundary.py`, which implements Eq. 34 and Eq. 37-38
   as candidate engineering code and can be invoked by the multi-step simulator
   as an optional per-step boundary drive. Remaining work is `U_DPF`
   magnetic-flux derivative closure, validated injection-port geometry, and
   same-scope circuit evidence.
9. Attach kinetic ion neutron-yield histories from particle distributions,
   mechanism-separated from thermonuclear fluid estimates and reduced
   beam-target baselines. Started as `src/dpf/fields/kinetic_yield.py` and
   optional loop telemetry; scalar cumulative-yield authority now blocks unless
   mechanism separation, detector response, UQ, and Te authority are attached.
   Remaining work is detector response, angular/spectral diagnostics,
   mechanism separation, and UQ.
10. Run a small 3D LLNL-like or PF-1000/Akel architecture smoke only as
   engineering evidence. Started as `src/dpf/fields/hybrid_simulator.py`;
   now exposed through `dpf hybrid-3d-smoke`; remaining work is a reviewed
   geometry packet, production-scale backend scope, conservation/nondominance
   tolerances, and same-scope validation.
11. Extract typed same-scope targets and produce numerical, physics, detector,
   neutron, and UQ packets before accepting predictive authority. Started as
   `src/dpf/validation/hybrid_pic_3d_validation_packet.py`, which blocks final
   acceptance unless capability, target, detector, UQ, conservation,
   nondominance, and backend-scaling packets are all accepted.

## Current Verdict

The repo is moving in the right direction on fail-closed metadata and 2D MHD
field coupling. The first 3D full-Maxwell component, PIC current-source port,
generalized Ohm-Ampere algebraic component, predictor-corrector primitive, and
Marder/conductivity controls, a one-step field-current integrator, and a
candidate particle-field loop step with conductor/PML particle absorption and
collision/electron-energy/yield telemetry, Marder nondominance flagging,
extended-Ohm Te authority blocking, kinetic-yield authority blocking, a
candidate source-ordered Eq. 7 velocity-update loop mode with provisional
predictor particle-current rebuild telemetry, a multi-step driver, and a
blocked source-geometry packet plus a source-scoped candidate RLC/magnetic
injection boundary wired into the multi-step simulator now exist as engineering
ratchets, `dpf hybrid-3d-smoke` can produce a blocked JSON engineering
artifact, and a final same-scope validation-packet gate now exists, but the
full goal still requires an accepted 3D hybrid
PIC-fluid core. The current Python cylindrical MHD path can support near-term
engineering progress, source-scoped startup work, and comparator evidence; it
cannot be the finish-line first-principles simulator.
