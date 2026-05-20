# CortexFindings

This file records the execution plan requested on 2026-05-05. `CodexFindings.md`
remains the running findings log; this file is the detailed plan and review
artifact named in the latest request.

## Goal

Reach a validated end-to-end predictive Dense Plasma Focus simulation tool using
only local `KnowledgeReference/` documents as scientific source authority.

## Plan Review

The current codebase has useful validation gates, but gates are not the same as
validated physics. The plan must therefore prioritize real target extraction,
production evidence generation, and scope control over adding more nominal
metadata. A result should become predictive only when it carries line-referenced
KR targets, same-scope comparisons, numerical verification, physics-fidelity
coverage, detector response, and propagated uncertainty for the claimed device
and observables.

The plan below is intentionally ordered. Each step should leave the tool more
truthful even if later steps are incomplete.

## First-Principles Finish-Line Baseline

2026-05-13 update: `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md` is now the active
execution specification for the first-principles priority. The older validation
plan below remains useful for evidence gates, but the first-principles critical
path is now ordered around these executable gates:

1. consolidate the first-principles candidate into a package-native `src/dpf`
   runner shared by CLI, API, config, and app surfaces;
2. build a global limiter/repair registry across `app_mhd.py`, solver internals,
   backend adapters, circuit coupling, and post-processing;
3. make first-principles readiness fail on any acceptance-blocking limiter
   activation and replace blocker limiters with verified numerical methods or
   source-backed physical bounds;
4. close numerical-fidelity tests with named norms, mesh families, tolerances,
   precision/backend scope, and limiter-zero acceptance;
5. build a source-backed startup boundary-value problem for breakdown,
   preionization, electrode/insulator boundaries, current density, fields,
   ionization, temperature, and sheath lift-off;
6. validate resolved power-port circuit coupling with Poynting or `J.E`,
   electrode work, time centering, sign convention, and residual tolerance;
7. decide dimensionality and any MHD-to-kinetic handoff for the claimed interval
   and observables;
8. close physics-fidelity packets for each active or bounded-out effect;
9. independently accept same-scope waveform, phase, spatial, neutron, detector,
   field-coupling, and UQ evidence, or narrow/switch the accepted demonstrator
   scope if PF-1000/Akel lacks the required evidence;
10. generate a validation certificate only after same-scope packet, comparator,
    UQ, review, physics-fidelity, numerical-fidelity, limiter, dimensionality,
    and package-native gates pass;
11. repeat the full evidence path on a second device or shot before claiming a
    generalized first-principles DPF-machine tool.

No scientific acceptance is promoted by this update. Akel Fig. 1 remains
`blocked_by_review`, Lee/snowplow remains baseline-only, and the current
`dpf first-principles` run remains engineering-probe evidence.

2026-05-14 FP-2 status: the first limiter-ledger slice is implemented for the
app-level PF-1000/Akel first-principles path. Run results now expose
`first_principles_limiter_ledger`, readiness blocks on missing or active
acceptance-blocking limiter evidence, and CLI/manifest artifacts carry compact
ledger summaries. This does not finish FP-2: solver-internal Python/Metal/MLX
limiters still need result-bound activation telemetry or verified-method
classification before limiter-zero acceptance is possible.

2026-05-14 FP-2 continuation: the Python cylindrical solver now emits
result-bound limiter events for state-mutating floors/clamps/repairs, and the
app-level first-principles path merges those events into
`first_principles_limiter_ledger`. Remaining FP-2 scope is narrower but still
blocking: classify flux-local positivity controls and PLM/HLL limiters with
verification evidence, then add or exclude Metal/MLX repair/fallback telemetry.

2026-05-14 FP-2 backend scope update: PLM/minmod, HLL flux,
reconstructed-state positivity floors, and CFL timestep control are now
nonblocking `verified_numerical_method` ledger entries for the Python
cylindrical path. Readiness also has an explicit backend-scope gate: Metal,
MLX, Athena, AthenaK, and hybrid paths remain outside first-principles
acceptance until backend-native limiter/fallback telemetry and parity evidence
are attached. Negative coverage now checks the advertised non-Python backend
labels and preserves requested backend identity across fallback labels.

2026-05-14 FP-2 blocker reduction: the active Python PF-1000/Akel
field-coupled path now removes the app-level resistivity floor/cap, temperature
floor/cap, hard field-coupled timestep cap, low-current voltage floor, and
back-EMF clip from bounded probes. The replacements are still candidate
engineering evidence, not validation: an uncapped partial-ionization
Spitzer/Braginskii resistivity path initialized from the local PF-1000
post-breakdown source state, public solver `compute_dt()` routing for
CFL/resistive diffusion control, and an implicit-midpoint circuit power port
that enforces `P_load = I_mid * V_load`. Bounded probes through `0.05 us`
now complete with clear limiter ledgers, but the full-horizon limiter-zero run,
startup BVP, same-scope field-coupling packet, numerical-fidelity packet,
physics-fidelity packet, Akel evidence review, and neutron authority gates
remain blocking.

2026-05-14 FP-2 timestep diagnosis update: the Python cylindrical path now uses
`Z_bar` in partial-ionization pressure recovery, electron-density bookkeeping,
and Ohmic electron heating. This prevents the false `Te=1 K` floor collapse
that previously drove uncapped resistivity into a prohibitive explicit
resistive timestep. Per-step timestep diagnostics are now exported as
`dt_s`, `dt_adv_s`, `dt_diff_s`, and `dt_controller`. Bounded probes through
`1.0 us` now complete with clear limiter ledgers, but they remain
resistive-diffusion controlled. The next numerics target is therefore a
verified implicit or STS resistive operator, followed by source-scoped
ionization/recombination evolution instead of constant `Z_bar`.

2026-05-14 FP-2 implicit resistive update: the active Python first-principles
path now uses a Crank-Nicolson ADI split for the cylindrical axisymmetric
`B_theta` resistive-induction operator and reports the old explicit
diffusion-CFL value as stiffness evidence instead of timestep authority. A
separate LC phase timestep controller keeps the field boundary and
implicit-midpoint circuit power port time-resolved after removing the accidental
explicit-diffusion clamp. Bounded probes through `1.0 us` now clear with
nonzero field feedback and no active acceptance blockers. Remaining FP-2/FP-3
work is now source-scoped ionization/recombination, electron-neutral transport,
anomalous resistivity, full-vector resistive operator scope if `B_r/B_z`
becomes material, and the same-scope validation packets.

2026-05-14 source-of-truth ingestion update: the user-validated arXiv
`2604.09032v1` paper, "A Fully Electromagnetic Hybrid PIC-Fluid Model for
Predictive Fusion Neutron Yield in Dense Plasma Focus", is now represented in
`KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`
and `.json`, with the staged PDF hash recorded in
`docs/USER_PDF_INTAKE_2026_05_14.json`. Treat it as source authority for the
first-principles architecture review queue, especially kinetic-ion/fluid-electron
handoff, fully electromagnetic field evolution, Hall/resistive/electron-pressure
Ohm-law terms, vacuum-field handling, and mechanism-separated neutron-yield
review. Do not promote its LLNL-like geometry, sheath-front comparison,
cross-section fit, or total neutron-yield number into simulator acceptance until
Step 2 creates typed KR target packets with same-scope traceability and review.

2026-05-20 source-target extraction update: the Sprint 4 extraction pass has
converted six already-local KR/source-available items into typed, line-referenced
target records in
`sprint4_source_available_target_extractions()` and
`docs/FIRST_PRINCIPLES_TARGET_EXTRACTIONS_2026_05_20.md`. Krasa 2008 now
supports PF-1000 chamber wall material/thickness as geometry context and has a
new KR validation target for vessel scatter/direct-vs-scattered neutron
requirements. Stepniewski 2004 now has a target-extracted `0.015 m` hollow-bore
candidate, but runtime geometry still blocks it as simulation-context evidence
pending hardware-scope review. UCSD/Beg startup, neon gas-puff Hall/LHDI
anomalous resistivity, NRL 2019 transport formulary, Talebitaher NX2
detector/anisotropy, and Klir ToF detector-response records are now explicit
extraction packets. None of these
records promotes Akel 16 kV same-scope validation, a startup BVP, accepted
transport closure, neutron authority, or a whole-shot first-principles
certificate.

2026-05-14 3D finish-line gate update: the new source has been converted into a
concrete architecture gate rather than a loose literature note.
`docs/FIRST_PRINCIPLES_3D_HYBRID_PIC_REVIEW_2026_05_14.md` now records the
source-derived application map. `src/dpf/validation/hybrid_pic_3d.py` defines
the required 3D hybrid PIC-fluid capability set, and
`first_principles_mhd_readiness_report()` now blocks on
`hybrid_pic_3d_first_principles_core` until a run has accepted evidence for
full Maxwell plasma/vacuum fields, kinetic ion PIC push/deposition,
electron-fluid generalized Ohm closure, current predictor-corrector,
divergence control, plasma-vacuum conductivity blending,
PML/conductor/particle-boundary semantics, ion collisions, true 3D
dimensionality, separate electron-energy closure, kinetic ion yield histories,
and same-scope 3D validation. Current 2D/cylindrical work remains valuable as
an engineering ratchet and comparator path, but it is no longer framed as the
finish line for the `/goal` simulator.

2026-05-15 FP-7 field/current/Ohm/predictor/Marder/conductivity/loop/boundary/collision/electron-energy/yield/multi-step/geometry implementation ratchet: the first isolated
3D Maxwell component, PIC current-source bridge, generalized Ohm component,
predictor-corrector primitive, Marder correction, conductivity blend,
particle-field loop step, particle-boundary hook, collision telemetry,
electron-energy hook, kinetic-yield history, multi-step driver, and
source-geometry packet now exist under
`src/dpf/fields/`. `maxwell_3d.py` uses the existing Yee/CT convention for
edge-centered `E` and face-centered `B`, provides Ampere/Faraday stepping,
conductor masks, deterministic PML damping metadata, Courant timestep, `div B`,
and EM energy diagnostics. `pic_coupling.py` maps cell-centered PIC deposition
to Yee edge currents for Ampere's law and reports continuity status as
nonaccepting telemetry. `ohm_solver.py` implements the source-derived
Ohm-Ampere midpoint algebraic current solve with Hall, Hall-disabled, and
density-thresholded pressure-gradient paths. `predictor_corrector.py` implements
the source current extrapolation and end-step Ohm correction around a supplied
provisional ion current. `marder.py` implements the source electric-field
correction for Gauss-law residual control with residual telemetry, and the
candidate stepper can map that correction back to Yee electric edges.
`conductivity.py` implements the source plasma-vacuum conductivity transition
and Ohmic CFL cap with active-fraction telemetry. `hybrid_stepper.py` ties the
Yee Maxwell state, conductivity blend, generalized Ohm solve, current edge
mapping, Maxwell advance, and optional predictor-corrector end-step current
solve/Marder correction into one candidate field-current step.
`hybrid_loop.py` pushes HybridPIC ions, deposits current, rebuilds
quasi-neutral electron density, and advances the field-current stepper in one
candidate particle-field loop step. `particle_boundaries.py` implements
candidate deletion of particles entering conductor/PML regions and the loop can
apply it before deposition. The loop also reports disabled versus
Nanbu/Perez-enabled ion-collision status from the existing HybridPIC kernel.
`electron_energy.py` wraps the repo two-temperature scaffold as candidate
source-term telemetry, and the loop can use a supplied electron-energy state to
build the pressure-gradient term and then update `Te` from the solved current.
`kinetic_yield.py` accumulates candidate D-D yield history from PIC ion
distributions and can attach instantaneous rate plus cumulative neutrons to the
loop telemetry. `hybrid_simulator.py` runs the candidate 3D loop for repeated
steps while carrying field, particle, optional `Te`, and yield state forward.
`source_geometry.py` captures the local source's LLNL-like axisymmetric setup
as blocked candidate geometry, not same-scope true-3D validation.
`circuit_boundary.py` now implements the local source's explicit RLC
current/charge update and `B_theta = mu0 I/(2 pi r)` injection-boundary formula
as a Cartesian engineering projection onto the 3D Maxwell grid; it remains
blocked because `U_DPF` flux-derivative closure, true injection-port geometry,
and same-scope circuit validation are missing. `HybridPIC3DSimulator` can now
optionally apply that boundary and advance the RLC state each step, recording
candidate telemetry without promoting the run.
`HybridPIC3DLoop` now also has a candidate source-ordered mode that advances
positions from stored half-step velocities, deposits current from old/new
positions, can rebuild density from half-step charge deposition, advances the
Ohm/Maxwell/Marder/predictor path, and then updates ion velocities with source
Eq. 7 before applying collisions. The simulator can request this mode for
multi-step runs. When predictor-corrector is requested, the loop now also
builds candidate provisional ion velocities and provisional ion current from
the particle state and feeds that provisional ion current into the candidate
end-step Ohm correction.
Marder telemetry now records correction magnitude, relative correction, an
explicit nondominance threshold, and whether the correction is within bound or
dominant. The current coupled particle-loop test deliberately preserves a
`candidate_dominant_correction` result for explicit charge density, so the gate
cannot overclaim divergence-control authority.
Extended-Ohm temperature authority is now explicit: Hall/pressure-gradient runs
without separate electron-temperature evidence are blocked as
`blocked_te_equal_ti_or_missing_separate_te`, and the candidate Te scaffold
still reports `candidate_separate_te_still_blocked` rather than promoting
quantitative extended-Ohm or neutron-yield claims.
Kinetic neutron-yield telemetry now records its mechanism status as
`not_mechanism_separated`, and a total-yield authority check blocks cumulative
scalar yield unless accepted kinetic history, mechanism-separated channels,
same-scope detector response, UQ, and electron-temperature authority are all
attached.
`hybrid_pic_3d_validation_packet.py` now provides the public same-scope
validation-packet evaluator: even if all 3D hybrid capabilities are accepted,
the final packet remains blocked without accepted targets, detector response,
UQ, conservation, nondominance, and backend-scaling evidence.
The CLI now exposes `dpf hybrid-3d-smoke`, a runnable engineering candidate
that exercises source-ordered 3D hybrid PIC-fluid stepping, circuit boundary
coupling, Te/yield telemetry, and the blocked validation packet in one JSON
artifact.
Focused tests pass for field
stepping, CT `div B`, conductor/PML behavior, HybridPIC current mapping,
Ampere current sourcing, Ohm residual closure, predictor-corrector residual
closure, Marder residual correction, conductivity blending, and gate
non-promotion; the integrated stepper test verifies Ohmic current reduces a
uniform electric field energy, and the loop test verifies particle motion plus
Esirkepov deposition after push plus optional particle absorption before
deposition, collision telemetry, optional Marder correction, optional
electron-energy coupling, candidate kinetic-yield history, and a three-step
candidate simulator run; the circuit-boundary tests verify Eq. 34 scaling, Eq.
37-38 current/charge stepping, azimuthal Cartesian direction, injection-plane
application, simulator coupling, and fail-closed gate behavior. Source-ordered
loop tests verify Eq. 7 velocity update telemetry, old/new current deposition,
half-step density use, optional predictor-corrector/Marder hooks, simulator
pass-through, candidate predictor-particle rebuild telemetry, and fail-closed
gate behavior. Marder tests now verify bounded smooth residual cleanup and
dominant-correction flagging. Extended-Ohm Te tests
verify missing separate-Te evidence and candidate Te evidence both remain
blocked for Hall/pressure claims. Kinetic-yield authority tests verify scalar
cumulative yield remains blocked without mechanism/detector/UQ/Te authority.
Validation-packet tests verify the current 2D source-geometry packet remains
blocked while a complete synthetic same-scope 3D packet can pass. CLI tests
verify the 3D smoke command writes a blocked engineering-candidate artifact.
FP-7 remains
blocked until the rest of the 3D hybrid
PIC-fluid source-derived capability set is implemented and accepted:
self-consistent long-run source-ordered ion PIC sequencing, accepted Ohm-loop
integration, accepted Te/Ti rebuild, accepted provisional ion-push/rebuild
predictor-corrector conservation/stability evidence, accepted nondominant
divergence-control evidence
against sheath/current observables,
accepted weakly active plasma-vacuum conductivity, external-circuit `U_DPF`
closure, electrode and boundary-validation packets, accepted collision
parameterization, accepted electron-energy
heat-flux/collisional coupling, mechanism-separated kinetic neutron-yield
authority with detector/UQ closure, and
same-scope 3D validation.

## Detailed Plan

0. KR Corpus Inventory And Exhaustive Review Control

   - Inventory every local `KnowledgeReference/` file.
   - Track DPF-named markdown files separately from the full broader plasma
     corpus.
   - Count a file as review-closed only when it contributes to a coded,
     line-referenced KR validation target or is explicitly rejected as
     non-extractable with reason.
   - Done when the unreviewed DPF-named source list is empty and target
     extraction or rejection status is auditable.

1. KR Target Authority Manifest

   - Enumerate every KR-backed validation target currently embedded in code.
   - Audit each target for local source file existence and in-bounds line ranges.
   - Expose a machine-readable manifest for app/API/report layers.
   - Done when every target used by validation helpers can be listed and source
     audited from local files.

2. KR Semantic Target Extraction

   - Expand target records from bibliographic citations into typed observable
     targets: device, shot, bank, fill, geometry, waveform, phase timing,
     density, magnetic/EM, temperature, neutron, detector, and uncertainty.
   - Add tests that assert source path, source lines, units, and validation
     scope for each extracted target.
   - Done when each predictive tier has at least one same-scope KR target packet
     with typed observables.

3. Tier 2 Phase Validation

   - Extract full axial, radial, and pinch timing targets for at least one
     KR-backed device/shot.
   - Promote ordinary production runs only when simulated phase history matches
     the same-scope target within explicit tolerance and uncertainty.
   - Done when tier 2 can pass from a real run without candidate-only fallbacks.

4. Tier 3 Production MHD Fidelity

   - Make production runs emit finite-volume method verification, cylindrical
     convergence, resistive diffusion convergence, circuit-coupled energy,
     backend parity, and MHD phase/scope-limit evidence.
   - Keep MLX/Metal runtime failures as blockers until resolved or bypassed by
     a validated non-MLX backend path.
   - Done when a production MHD result passes the numerical-fidelity audit.

5. Tier 4 Same-Scope Spatial Validation

   - Build one same-scope spatial validation packet with density, magnetic/EM,
     and temperature evidence from KR-backed diagnostics.
   - Reject cross-device or review-only component mixing.
   - Done when tier 4 can pass from a real same-scope target packet.

6. Tier 5 Neutron Validation

   - Generate or ingest mechanism-separated neutron timing, spectrum,
     anisotropy, detector/activation response, and yield uncertainty for one
     KR-backed scope.
   - Done when tier 5 and the neutron high-fidelity gap both pass from the same
     validation scope.

7. Physics-Fidelity Closure

   - Implement or explicitly bound EOS/conductivity, ionization,
     two-temperature physics, radiation transport/opacities, ablation/impurity
     mixing, Hall/FLR/kinetic/PIC effects, 3D instabilities, flashover/sheath
     initiation, restrike/anomalous resistance, and beam-target coupling.
   - Done when every required effect is validated or bounded out for the
     claimed observable scope.

8. Uncertainty Propagation

   - Propagate experimental, input, numerical, model-form, shot-to-shot, and
     acceptance-rule uncertainties into every supported validation tier.
   - Done when high-fidelity readiness is blocked by no UQ component.

9. End-to-End Predictive Demonstration

   - Run a complete same-device/same-scope DPF case through all gates.
   - Export reproducibility, source authority, predictive readiness,
     high-fidelity readiness, and scientific-accuracy gaps.
   - Done when an ordinary production run, not a synthetic packet, reports
     `high_fidelity_ready`.

## Current Execution Position

Step 0 source-review control is complete for the DPF-relevant local markdown
corpus. Later execution log entries supersede the older partial-source position:
96 of 96 DPF-relevant markdown files are review-closed through coded target
records or explicit non-target/duplicate/context decisions. The current blocker
is validation evidence quality, not unreviewed DPF-named sources.

The active scientific ratchet is now Akel/PF-1000 same-scope source closure.
Standalone source-scoped no-crowbar `pf1000_akel` M2/M6 evidence is current
after the Lee current-factor circuit-loading fix: the 12 us probe reached
`peak_I_MA=1.150507` inside the shot-12581 M2 band `1.0485-1.2815 MA`. S1/S2
remain blocked because accepted same-scope digitized current waveform/current-dip
evidence with uncertainty does not yet exist. The Akel Fig. 1 draft packet is no
longer missing waveform data; it has 294 measured-current candidate points, 34
computed-current candidate points, and internal overlay RMS `0.213455189 px`,
but it remains `blocked_by_review` until independent review is completed and
`review_status="accepted"`.

Parallel scientific-closure work started on 2026-05-11 for newly promoted local
PF-1000/diagnostic KR sources. Cikhardtova 2015, Szydlowski 2004, Klir 2011,
Springham 2021, and Catenacci 2020 now have typed target records and a dated
target-extraction workbench report. The report contains 23 rendered pages and
36 unreviewed crop candidates, all with `accepted_for_validation=false`; this
is preparation for digitization, not accepted validation evidence.

Supplemental user intake on 2026-05-12 staged 35 unique local PDFs from 39
supplied paths, promoted 28 new DPF/plasma/numerics/math-method records into
`KnowledgeReference/`, repaired a false Trunk 1975/Kortanek 2014 title match, and
kept seven non-physics/AI-only support PDFs staged but outside physics authority.
Six promoted book-length sources were chunked into 126 Markdown chunks, and 28
promoted records received a second-pass source-fidelity review. This
adds local source text and source-critical artifact indexes only; every new
record remains `source_available_not_target_extracted` and cannot validate
physics until typed targets or reviewed digitization packets are created.

The SRS/productization ratchet is separate from the scientific ratchet. The
draft SRS now identifies missing product controls that are not closed by science
evidence alone: formal requirement traceability, compute-authority labels
(`Reference`/`Preview` or equivalent), run manifests, validation certificates,
memory preflight, unsupported-backend-feature warnings, local-first/security
audits, export acceptance decisions, and a refreshed current TODO audit.

First-principles MHD execution now has a fail-closed mode contract but not a
validated physics stack. `first_principles_mhd` is initially scoped to the
PF-1000/Akel path, maps onto the current MHD execution path, and exports
readiness metadata that blocks on accepted Akel evidence, field-derived
coupling, numerical-fidelity packets, physics-fidelity coverage, and removal of
Lee/RADPF closure factors from acceptance scoring. Lee/snowplow remains allowed
only as `baseline_reduced_model` for initialization, comparison, and regression.
The first production energy-accounting slice exposes capacitor, inductive,
resistive, residual, and dynamic-inductance power channels, but it remains
blocked until field Poynting power and same-scope field-coupling validation are
attached. Startup/sheath initialization is also surfaced as scaffolded metadata:
the current seeded inlet layer, electrode boundary condition, and snowplow
sheath-position outputs remain blocked until source-backed breakdown,
preionization, initial plasma, and same-scope sheath evidence exist. No new
scientific evidence is promoted by this contract. The app/post-processing path
now honors the same `first_principles_mhd` fail-closed metadata, so UI/API-style
MHD runs cannot skip the Akel review or first-principles evidence blockers.
The server/API response model now exposes the same first-principles readiness,
energy-accounting, and startup-initialization blocker packets for declared
`first_principles_mhd` runs, and the legacy Gradio selector includes the mode
with explicit `baseline_reduced_model` language for Lee/snowplow outputs.
Config/CLI/engine summaries now carry the same `run_mode` authority label, so
`dpf simulate --run-mode=first_principles_mhd` can write fail-closed readiness
metadata into ordinary run summaries and manifests without changing backend
selection or promoting evidence.

Plan audit 2026-05-13: `docs/FIRST_PRINCIPLES_EXECUTION_AUDIT_2026_05_13.md`
supersedes the broad blocker plan as the immediate execution guide. The
metadata/readiness/reporting layer is good enough for now. Stop adding
additional readiness surfaces and move the critical path to one working
field-coupled PF-1000/Akel candidate run. The next code target is resolved-field
coupling: magnetic field energy, field-derived inductance, Poynting or `J.E`
power, nonzero field-derived back-EMF/terminal voltage, and an inspectable
energy ledger wired into the circuit after startup/handoff. Broad tests and
validation gates resume after that candidate run exists.

Execution update 2026-05-13: the first field-coupled Python MHD candidate now
runs as an engineering probe for `first_principles_mhd`. The cylindrical grid
now supports an annular radial offset for PF-1000/Akel, and the candidate path
computes magnetic energy, `L_field`, `dL_field/dt`, `integral(J dot E)dV`,
field terminal voltage/back-EMF, interface power, Joule-power history, and
energy residual from resolved fields. Coarse PF-1000/Akel probes completed
finite 0.2 us and 1.0 us intervals with nonzero `B_max`, `L_field`, `J.E`
power, and back-EMF histories. Follow-on ratcheting added capped
Spitzer/Braginskii resistivity, nonzero Joule heating, corrected field-power
circuit authority, and a recorded engineering limiter. A coarse PF-1000/Akel
12 us `first_principles_mhd` shot now completes with `nan_detected=False` over
24,000 steps and `I_peak_MA=1.1930438248311477`. This is not scientific
validation; the limiter remains an engineering blocker, and same-scope Akel
evidence remains blocked by review.

Execution update 2026-05-13, neutron-yield authority: first-principles
acceptance now includes a fail-closed neutron-yield gate. A total neutron yield
cannot support first-principles predictive claims unless thermonuclear yield is
integrated from resolved field history, beam-target yield is supplied by an
accepted kinetic/hybrid beam model rather than Lee/Saw calibration, and
same-scope scalar yield, mechanism timing, spectrum, anisotropy,
detector/activation response, uncertainty, numerical-fidelity, and
physics-fidelity evidence all pass together. Existing app-level neutron totals
are therefore explicitly `blocked` for first-principles total-yield authority
when they depend on final-state duration approximation or reduced beam-target
estimators. The 10% paper-yield recreation target is an acceptance criterion
for future same-scope packets, not a current validated capability.

Execution update 2026-05-13, thermonuclear history: the Python MHD candidate
now performs the thermonuclear DD part as a resolved-field history integral over
cylindrical cell volumes rather than a final-state-times-duration shortcut.
Post-processing preserves that field-history result and keeps total-yield
authority blocked until an accepted kinetic/hybrid beam-target model and the
same-scope neutron evidence packet exist. This moves the thermonuclear component
toward first principles without relaxing the total-yield validation gate.

Tool update 2026-05-13: the first-principles PF-1000/Akel engineering candidate
now has a narrow user-facing entrypoint. `run_pf1000_akel_first_principles()`
locks the app path to `backend="first_principles_mhd"` and
`preset_name="pf1000_akel"`, and `dpf first-principles` calls that helper,
enforces `field_coupled_candidate=True` plus `has_snowplow=False`, and can write
a compact JSON artifact. A 0.2 us smoke wrote
`results/first_principles_pf1000_akel_smoke.json` with 400 finite steps,
`I_peak_MA=0.1128993`, nonzero field back-EMF, and `readiness=blocked`. This is
still engineering-probe evidence only: Akel S1/S2 remain blocked, the Fig. 1
packet remains `blocked_by_review`, and first-principles readiness/neutron-yield
authority remain blocked until the same-scope evidence gates pass.

Tooling activation 2026-05-08: the Codex environment now has curated `pdf`,
`playwright`, `security-best-practices`, `security-threat-model`, and
`security-ownership-map` skills installed, plus local `dpf-validation` and
`srs-traceability` skills under `~/.codex/skills`. The repo now exposes a
`traceability` optional dependency group with Doorstop and documents the
first-pass SRS/RTM tooling path in `docs/SRS_TRACEABILITY_TOOLING.md`. Doorstop
is tooling only; it does not change scientific acceptance status.

## Blocker Closure Plan

This plan tracks every remaining scientific blocker and every new SRS/product
blocker identified from the draft SRS review. Each task must leave evidence in
code, tests, docs, or findings before it can be marked complete. If a task cannot
be closed with current local evidence, it must preserve a blocker rather than
create a passing placeholder.

### Track A: Scientific Closure

| ID | Task | Goals and objectives | Guardrails | Skills and methods | Exit evidence |
| --- | --- | --- | --- | --- | --- |
| A1 | Findings/status hygiene | Keep `CortexFindings.md`, `CodexFindings.md`, SRS draft, and source queue mutually consistent. Correct stale status before implementing downstream work. | Do not delete historical log entries; supersede them with dated addenda. Preserve exact probe values and failure strings. | Technical writing, git diff review, evidence reconciliation. | Current-position sections name source review as closed and identify Akel S1/S2 plus SRS product controls as active blockers. |
| A2 | Akel Fig. 1 independent review gate | Move the Fig. 1 draft packet from `blocked_by_review` only if an independent review accepts the digitization. | Do not self-promote draft arrays. Do not treat internal overlay residual as independent review. Keep `passed=False` until review metadata is valid. | Digitization QA, figure/axis audit, JSON packet review, hash verification. | `digitization_verification_evidence()` fails only before review and passes only after valid reviewer count and `review_status="accepted"`. |
| A3 | S1/S2 waveform comparator | Add a source-scoped comparator for accepted Akel current waveform and current-dip evidence. | Use Akel 16 kV shot-12581 scope only. Do not mix Scholz/Gribkov 27 kV/full-energy PF-1000 evidence. Do not compare against draft data. | Signal processing, NRMSE/dip metrics, uncertainty-aware acceptance, pytest. | Tests show draft packet reports blocked-by-review; accepted packet enables S1/S2 candidate evidence with explicit uncertainty and source scope. |
| A4 | Remaining Akel figure digitization | Process Fig. 2-4 current waveforms and Fig. 5-6 yield plots through the same queue/gate. | Each packet needs source path/hash, figure path/hash, page/figure ID, axis calibration, arrays, overlay residual, and review acceptance. | PDF/SVG extraction, axis calibration, JSON schema, residual analysis. | `scientific_closure_digitization_status()` reports each processed task as accepted or explicitly blocked with reason. |
| A5 | Source acquisition and KR ingestion | Acquire missing exact source documents for detector response, anisotropy, spectra, neutron timing, and same-scope diagnostics. | External links are not evidence. User must acquire the document, it must be placed under `KnowledgeReference/`, and local review must pass before use. | Literature triage, local PDF parity, KR markdown ingestion, source-line extraction. | Source queue entries move from candidate acquisition to local KR-reviewed status with hashes and target decisions. |
| A6 | Tier 2 phase validation | Build same-device KR-backed axial, radial, pinch/stagnation timing targets and attach them to production runs. | Targetless phase labels stay candidates. Phase evidence must name diagnostic source and uncertainty. | Lee/RADPF semantics, phase-history extraction, event detection, tolerance design. | Tier 2 passes only for same-scope targets; partial/cross-scope phase evidence remains blocked. |
| A7 | Tier 3 numerical fidelity | Expand production MHD evidence beyond generic Sod/Brio-Wu tests. | Verification is not DPF validation. Do not let generic shock tests substitute for spatial experiment evidence. | Numerical methods, convergence studies, backend parity, restart tests, energy accounting. | Tier 3 packet includes named tests for cylindrical source terms, resistive diffusion/heating, circuit energy, backend parity, convergence, restart/reproducibility. |
| A8 | Tier 4 spatial validation | Build one same-scope density, magnetic/EM, and temperature validation packet. | Reject cross-device mixing and review-only context. Temperature-only or density-only packets stay partial. | Diagnostic interpretation, spatial comparison metrics, source authority, uncertainty tracking. | Tier 4 support requires density, magnetic/EM, and temperature evidence from one KR-backed validation scope. |
| A9 | Tier 5 neutron validation | Generate or ingest same-scope neutron timing, spectrum, anisotropy, detector/activation response, scalar yield, and uncertainty. | Do not count helper comparisons on supplied arrays as production neutron validation. Do not treat scalar yield alone as tier 5. | Neutron diagnostics, detector response, TOF/spectrum analysis, mechanism-separated histories. | Tier 5 passes only when all neutron components share one KR-backed validation scope and uncertainty model. |
| A10 | Physics-fidelity closure | Make missing/empirical physics status explicit for each claimed scope. | Keep reduced/empirical models, but label them. Do not claim predictive high-Z, p-B11, late-pinch, or neutron behavior unless required physics is implemented/validated or bounded out. | Physics model audit, evidence schema design, source-backed scope limits. | Run results report implemented, verified, validated, empirical, absent, or bounded-out status for each required physics effect. |
| A11 | Circuit-field coupling fidelity | Define and validate evidence for field-derived inductance, dL/dt/back-EMF, Poynting flux, circuit energy, and handoff timing. | Do not call density-weighted or Lee-style coupling fully field-derived. Preserve snowplow/blended/field-coupled interval labels. | Circuit/MHD coupling, energy balance, Poynting theorem, result metadata. | `field_coupling_validation` evidence distinguishes snowplow-loaded, blended, and validated field-coupled intervals. |
| A12 | UQ propagation | Extend uncertainty beyond circuit waveform comparisons into phase, spatial, neutron, numerical, model-form, and shot-to-shot evidence. | Avoid point-tolerance-only pass/fail for high-fidelity claims. Missing uncertainty keeps readiness blocked. | ASME/GUM UQ, error propagation, statistical validation, acceptance-rule design. | Every supported validation tier reports experimental, input, numerical, model-form, shot-to-shot, propagated-observable, and acceptance-rule uncertainty status. |
| A13 | Long PF-1000 fixture policy | Decide whether the long PF-1000 xfailed classes stay scientific gates or become opt-in endurance/regression tests. | Do not re-enable as passing scientific gates until same-scope S1/S2 evidence exists. Do not hide long-run cap exhaustion. | Test architecture, pytest markers, MLX process isolation, runtime budgeting. | `tests/test_mlx_pf1000.py` clearly separates scientific xfail gates from opt-in endurance/regression paths. |
| A14 | Newly promoted KR target extraction and digitization | Move Cikhardtova 2015, Szydlowski 2004, Klir 2011, Springham 2021, and Catenacci 2020 from reviewed source text into reviewed target/digitization packets. | Use only local KR markdown records and hashed local intake source PDFs. Rendered pages and crop candidates are workbench artifacts, not validation evidence. Do not promote any packet before axis/table calibration, numeric arrays, residual checks, and independent review. | PDF rendering, crop QA, axis calibration, table extraction, figure digitization, JSON packet schema, independent review gating. | A14 crop-boundary QA report exists with 36 crops, 21 ready-for-draft-extraction figure crops, 9 manual-review crops, 0 crop-adjustment-needed crops, 6 review-blocked table drafts, and 0 accepted validation packets; all accepted packets must pass `digitization_verification_evidence()` before validation use. |

#### Track A Detailed Simulation/Physics Breakdown

This section breaks the remaining simulation and physics plan into smaller
work units. Status terms are intentionally conservative:

- `ready-to-code` means the repo has enough local context to implement or test
  the guardrail now.
- `evidence-blocked` means implementation must keep reporting a blocker until
  local `KnowledgeReference/` evidence, digitization, or review metadata exists.
- `policy-decision` means the code path exists or can be built, but the release
  posture must be chosen before it is treated as a gate.

| Parent | Work unit | Current state | Guardrails | Concrete objective | Methods and skills | Verification / exit |
| --- | --- | --- | --- | --- | --- | --- |
| A2 | Review packet intake | evidence-blocked | Only an independent accepted review can promote Akel Fig. 1. Internal overlay residuals and local helper output are not review acceptance. | Define the minimum accepted-review packet fields: reviewer identity or role, review date, reviewed source hash, reviewed packet hash, accepted status, and reviewer notes. | Digitization QA, provenance hashing, review metadata schema, negative tests. | Draft packet remains `blocked_by_review`; accepted status is honored only when packet hash and reviewer metadata match the current artifact. |
| A2 | Review-gate regression hardening | completed guardrail; evidence-blocked for real acceptance | Do not allow manual field flips to bypass packet-bound review checks. | Add or keep tests for stale packet hash, missing reviewer fields, mismatched figure hash, and non-accepted review states. | Pytest, JSON fixture mutation, failure-mode assertions. | `digitization_verification_evidence()` fails closed for every malformed or stale review fixture. |
| A3 | Accepted waveform comparator path | evidence-blocked | Compare only accepted Akel 16 kV shot-12581 current traces; no draft packet, cross-scope PF-1000, or scalar-only substitution. | Finish the S1/S2 comparison path for NRMSE, peak-current error, current-dip depth, current-dip timing, and uncertainty-aware acceptance. | Signal processing, interpolation, uncertainty bands, source-scope validation records. | Draft data reports `blocked_by_review`; an accepted same-scope fixture produces explicit metrics and requirement IDs without promoting unrelated scopes. |
| A3 | Production run attachment | completed guardrail; evidence-blocked for accepted metrics | Do not let helper-only comparisons become production validation unless run metadata names the source scope and accepted evidence. | Attach S1/S2 waveform comparison output to production summaries, manifests, readiness reports, and certificate inputs. | Engine metadata wiring, manifest/certificate tests, readiness schema checks. | A normal run can carry blocked S1/S2 evidence today and accepted S1/S2 evidence only after accepted waveform data exists. |
| A4 | Akel Fig. 2-4 waveform queue | evidence-blocked | Each figure must keep its own source path/hash, page/figure ID, axis calibration, series arrays, overlay residual, and review status. | Extract, digitize, review-gate, and status-report each remaining Akel waveform figure. | PDF/SVG extraction, manual/vector digitization, axis calibration, JSON packet tests. | Queue status distinguishes `not_extracted`, `draft_unreviewed`, `blocked_by_review`, and `accepted` per figure. |
| A4 | Akel Fig. 5-6 yield queue | evidence-blocked | Yield figures cannot substitute for waveform validation; they need their own observables and uncertainty. | Digitize yield plots into source-scoped packets usable for scalar-yield or trend checks only. | Figure digitization, yield-unit audit, uncertainty capture, source-scope labels. | Yield packets can support only their mapped requirements and cannot close S1/S2 waveform gates. |
| A5 | Source acquisition queue by physics need | completed queue guardrail; evidence-blocked for source ingestion | External links and remembered citations are not evidence. Local files under `KnowledgeReference/` plus review/hash metadata are required. | Maintain an acquisition queue for current traces, phase timing, density, magnetic/EM, temperature, neutron timing/spectrum/anisotropy, detector response, and UQ sources. | Literature triage, local PDF parity, source-line extraction, queue/status schema. | Every missing validation input is represented as an acquisition item with required data, candidate source, local-evidence status, and done condition. |
| A5 | KR ingestion and source authority | evidence-blocked until documents exist | Do not mark a source usable from title/abstract or DOI alone. | Convert acquired local sources into reviewed KR entries with exact line ranges or figure/table references. | OCR/PDF review, hash checks, source authority audit, source-backed target records. | Source status moves to KR-reviewed only when local artifact hashes and extracted support lines are recorded. |
| A5 | May 12 user intake promotion and fidelity | completed text-parity/source-fidelity intake; evidence-blocked for target use | Promotion is not validation. Stage-only AI/non-physics PDFs must not be cited as DPF physics authority. Newly promoted records remain unusable for thresholds until target extraction or reviewed digitization is done. | Local PDF staging, SHA-256 de-duplication, PyMuPDF text extraction, book chunking, source-critical fidelity audit. | `docs/USER_PDF_INTAKE_2026_05_12.*`, `docs/USER_PDF_KR_PROMOTION_2026_05_12.*`, and `docs/USER_PDF_KR_SOURCE_FIDELITY_AUDIT_2026_05_12.*` record 35 unique staged PDFs, 28 new KR pairs, 126 book chunks, and 28 source-fidelity-reviewed records. |
| A14 | All-source crop-boundary review | completed QA inventory and crop rectification; evidence-blocked for acceptance | The 36 crop candidates are not evidence. Review only crop completeness: axes, captions, legends, units, trace visibility, and any missing panels. A visual QA pass is not independent scientific review. | Boundary-review six Cikhardtova 2015 crops, five Szydlowski 2004 crops, four Klir 2011 crops, nine Springham 2021 crops, and twelve Catenacci 2020 crops, adjusting crop rectangles only when axes/captions/traces are incomplete. | Visual crop QA, PDF coordinate review, hash-recorded artifact regeneration, report invariant checks. | `docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.json` records 36 crop entries, 21 boundary-ready figure crops, 9 manual-review crops, 0 crop-adjustment-needed crops, 6 review-blocked table drafts, and 0 accepted packets. |
| A14 | Klir/Springham/Catenacci crop generation | completed workbench crop generation; evidence-blocked for acceptance | Crops cannot be treated as digitized data. Rendered pages and crop images are only provenance-preserving workbench artifacts until calibrated/extracted/reviewed. | Keep crop hashes and report invariants current while moving Klir detector figures, Springham activation figures/tables, and Catenacci tomography figures/tables into numeric extraction packets. | Page-image review, PyMuPDF crop rectangles, figure/table task inventory, report regeneration. | The target-extraction report records crop candidates for all cited figures/tables: Klir Figs. 1-4, Springham Figs. 1-7 and Tables 1-2, and Catenacci Figs. 1-8 and Tables I-IV. |
| A14 | Axis/table calibration and numeric extraction | draft table extraction complete for 6 tables; draft axis-calibration scaffolds complete for 3 priority figures; Springham Fig. 5 and Klir Fig. 2 numeric drafts extracted; evidence-blocked for acceptance | Do not invent values from OCR-suspect text. Every extracted value needs source path/hash, local PDF hash, figure/table ID, units, calibration points or table structure, and residual evidence. Axis scaffolds without series arrays are not digitization packets. | Build draft extraction packets for timing traces, density profiles, spectra, anisotropy, detector response, activation response, and tomography tables. | Figure digitization, table parsing, unit normalization, uncertainty capture, overlay residual analysis. | `a14-2026-05-11-table-draft-packets.json` contains 6 review-blocked table drafts with loader-provided per-table item hashes plus enforced local-PDF and crop-hash checks; Springham Fig. 5 now has mono-energetic and Gaussian-curve draft packets; Klir Fig. 2 now has an FWHM/rise-time timing-response draft packet; all remain blocked on independent review/status acceptance. |
| A14 | Independent review gate for new packets | review handoff, local-PDF hash checks, and table hash guardrails complete; evidence-blocked for real acceptance | Codex cannot self-accept target/digitization packets. Review metadata must bind to the current packet/source/local-PDF/figure-or-crop hashes. | Apply the same review-gate model used for Akel to every newly extracted packet and provide a reviewer-facing manifest. | Review workflow, packet-hash binding, source-PDF binding, crop-hash binding, negative tests for stale or incomplete review metadata. | `docs/A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.json` lists 9 reviewable draft packets and 3 context-only axis scaffolds; packets pass `digitization_verification_evidence()` only with current hashes, required arrays, residuals, and independent accepted review. |
| A6 | Tier 2 target selection | evidence-blocked | Targetless phase labels are candidates, not validation. Do not mix Akel shot-12581 with full-energy PF-1000. | Choose one same-device validation scope and define axial end, radial start/end, pinch/stagnation, and current-derivative timing targets with uncertainty. | Lee/RADPF phase semantics, event detection, target schema design. | Phase targets are typed, source-backed, same-scope, and uncertainty-bearing before Tier 2 can pass. |
| A6 | Phase-history comparator | completed comparator/status guardrail; evidence-blocked for targets | Simulation event detection must not invent target tolerances. | Compare production phase histories to verified targets and attach pass/blocked/failed Tier 2 evidence. | Time-series event detection, tolerance checks, readiness/certificate integration. | Tier 2 remains blocked without verified targets and passes only on same-scope comparison. |
| A7 | Cylindrical MHD source-term verification | scheduled Tier-3 packet complete for code verification | Numerical verification is not experimental DPF validation. | Add named cylindrical source-term checks with expected invariants and convergence behavior. | Numerical methods, manufactured/analytic checks, convergence studies. | `results/mhd_tier3_numerical_packet.json` carries cylindrical z-pinch convergence evidence for the scheduled CPU scope and now closes the same-scope Tier-3 packet. |
| A7 | Resistive diffusion/heating verification | scheduled Tier-3 magnetic-diffusion packet complete; real heating validation still required outside Tier-3 code verification | Passing generic shock tubes cannot stand in for resistive DPF numerics. | Verify magnetic diffusion and Joule/resistive heating against analytic or reference checks. | PDE verification, energy accounting, tolerance design. | `results/mhd_tier3_numerical_packet.json` carries resistive magnetic-diffusion convergence evidence; DPF Joule/heating validation remains a higher-scope evidence need. |
| A7 | Circuit-coupled energy, backend parity, and restart evidence | scheduled Tier-3 packet complete for same-scope code verification | MLX/Metal preview behavior must stay labeled until parity and evidence support promotion. | Check circuit energy balance, restart reproducibility, CPU/MLX parity where comparable, and finite-volume MHD channel behavior. | Backend parity tests, restart tests, energy-budget tests, deterministic fixtures. | `results/mhd_tier3_numerical_packet.json` now attaches finite-volume, circuit-energy, backend-parity, and restart reproducibility evidence with no missing Tier-3 packet channels. |
| A8 | Same-scope spatial packet selection | evidence-blocked | Density-only, temperature-only, or EM-only evidence is partial; cross-device merging is rejected. | Select one validation scope that can supply density/proxy, magnetic/EM, and temperature observables with uncertainty. | Diagnostic interpretation, source acquisition, spatial comparison schema. | Tier 4 combiner sees all three components from the same scope before it can support high-fidelity readiness. |
| A8 | Spatial comparison implementation | completed closure/status guardrail; evidence-blocked for packet selected | Derived or proxy diagnostics must be labeled as such and cannot masquerade as direct field validation. | Implement component metrics for density/proxy geometry, magnetic/EM signal, and temperature comparison. | Image/signal comparison, uncertainty propagation, component evidence records. | Partial components remain visible but non-promoting; complete same-scope packets can support Tier 4. |
| A9 | Neutron evidence packet selection | completed same-scope closure guardrail; evidence-blocked for real packet | Scalar yield alone is not Tier 5. Timing-only evidence is not detector/anisotropy/spectrum validation. Missing detector/activation response or explicit uncertainty keeps the gate blocked. | Select or build a same-scope packet for neutron pulse timing, spectrum, anisotropy, detector/activation response, scalar yield, and uncertainty. | Neutron diagnostics, TOF/spectrum review, detector response modeling, source authority. | Tier 5 cannot pass until scalar yield, timing, spectrum, anisotropy, detector/activation response, and uncertainty all share one KR-backed validation scope. |
| A9 | Mechanism-separated production output | completed reporting guardrail; evidence-blocked for validation | MHD-side helper arrays are not proof of beam-target physics. Mechanism labels must remain explicit. | Ensure production runs expose thermonuclear, beam-target/fast-ion, detector/activation, and timing histories separately enough for comparison. | Diagnostics integration, yield decomposition, result metadata, tests. | Neutron readiness reports blocked components instead of collapsing them into a single total yield. |
| A10 | Per-run physics-fidelity matrix | completed reporting guardrail; evidence-blocked for real validation | Reduced or empirical models are allowed only when labeled. Do not imply predictive late-pinch, high-Z, p-B11, or neutron behavior. | Attach per-run status for EOS, ionization, two-temperature, radiation transport, impurities/ablation, Hall/FLR/kinetic/PIC, 3D, flashover/startup, restrike/anomalous resistance, and beam-target coupling. | Physics model audit, metadata schema, warning/readiness tests. | Run summaries and readiness reports state `implemented`, `verified`, `validated`, `empirical`, `absent`, or `bounded_out` for each effect. |
| A10 | Scope-bounding rules | completed reporting guardrail | Missing physics should block only claims that depend on it, not every engineering run. | Map each physics effect to the claims it blocks: circuit waveform, phase dynamics, spatial MHD, neutron, high-Z, p-B11, or late-pinch prediction. | Requirements mapping, source-backed claim limits, negative tests. | Predictive/high-fidelity readiness identifies exactly which claims remain blocked and why. |
| A11 | Field-derived coupling evidence design | completed reporting guardrail; evidence-blocked for validated coupling | Density-weighted or Lee-style inductance is not fully field-derived coupling. | Define evidence fields for field-derived inductance, `dL/dt`, back-EMF, Poynting power, circuit energy, handoff timing, and interval label. | Circuit/MHD coupling, energy balance, Poynting theorem, evidence schema. | `field_coupling_validation` distinguishes snowplow-loaded, blended, and validated field-coupled intervals. |
| A11 | Minimal staged coupling acceptance | completed reporting guardrail; evidence-blocked for validated intervals | Do not jump directly from blended coupling to full MHD circuit authority. | Stage acceptance as: snowplow-loaded baseline, blended interval accounting, field-derived candidate interval, then validated field-coupled interval. | Incremental tests, energy closure metrics, backend-specific labels. | Each interval reports its authority and blockers; unsupported intervals cannot support field-coupled MHD claims. |
| A12 | UQ schema per validation tier | completed tier/status/source-value guardrail; evidence-blocked for real source values | Point tolerances alone are insufficient for high-fidelity claims. Missing uncertainty keeps tiers blocked. | Define required uncertainty fields for circuit waveform, phase, spatial, neutron, numerical, model-form, and shot-to-shot evidence. | UQ modeling, evidence schema, acceptance-rule design. | Every tier can report whether experimental, input, numerical, model-form, shot-to-shot, propagated-observable, and acceptance-rule uncertainty exists. |
| A12 | Propagation and acceptance rules | evidence-blocked for real sources | Do not invent uncertainty values where source packets lack them. | Propagate available uncertainties into comparison metrics and mark missing terms as blockers. | Error propagation, interval comparisons, statistical validation. | Readiness uses interval-aware pass/blocked/fail decisions and names missing UQ components. |
| A13 | Long fixture classification | completed policy; scientific gates remain evidence-blocked | Do not re-enable long PF-1000 xfails as scientific gates until S1/S2 source closure exists. | Decide whether long PF-1000 classes are scientific xfails, opt-in endurance checks, or scheduled regression jobs. | Pytest marker design, runtime budgeting, MLX process isolation. | `tests/test_mlx_pf1000.py` documents the policy and separates scientific blockers from endurance/runtime evidence. |
| A13 | Endurance/run-budget implementation | completed implementation; scientific gates remain evidence-blocked | Cap exhaustion must be explicit; long-run tests must not hide process aborts or skipped source closure. | Add marker/env controls for target time, cap, memory telemetry, and expected blocker classification. | Test architecture, standalone runner controls, CI/runtime policy. | Endurance jobs report target, cap, final time, memory telemetry, and scientific/non-scientific status. |

Near-term simulation/physics execution order:

1. Advance A14 through all-source crop-boundary review, axis/table calibration,
   numeric extraction, and residual checks, while preserving 0 accepted packets.
2. Preserve the A2/A3 blocker while preparing the accepted-review intake path.
3. Finish the S1/S2 production-attachment path so blocked waveform evidence
   appears consistently in run summaries, manifests, readiness reports, and
   certificates.
4. Use A4/A5/A14 source queues for every missing same-scope observable before
   attempting Tier 2, Tier 4, or Tier 5 acceptance.
5. Continue A7, A10, A11, and A12 as code-ready guardrail work that can proceed
   without pretending the missing experimental packets exist.
6. Treat A6, A8, and A9 acceptance as evidence-blocked until verified local
   source packets provide the same-scope targets.
7. Keep A13 scientific gates blocked until S1/S2 source closure exists; use the
   opt-in endurance path only as non-scientific regression evidence.

Current Track A closure status after the 2026-05-12 source-intake pass:

- Code-ready guardrails and production status surfaces are complete for A2/A3,
  A5, A6, A7, A8, A9, A10, A11, A12, and A13.
- A5 gained a new local-source intake batch: 39 supplied PDF paths were all
  readable; 35 unique SHA-256 payloads were staged; 28 new KR Markdown/JSON
  records were promoted; a false Trunk 1975/Kortanek 2014 generic-title match
  was repaired; and 7 stage-only records remain outside
  physics authority. Source-fidelity review updated 28 records and recovered
  source-critical secondary-extraction items. A follow-on triage report now
  ranks 5 target-extraction candidates and 23 method/context references, but
  no validation targets were accepted.
- A14 is active: typed target records exist for five newly promoted local KR
  sources, 23 pages were rendered, and 36 crop candidates were generated. A
  crop-boundary QA report now classifies 21 figure crops as ready for draft
  axis/numeric extraction, 9 as manual-review diagram/image crops, and 0 as
  needing crop adjustment before extraction. Six table draft packets now exist
  for Springham 2021 Tables 1-2 and Catenacci 2020 Tables I-IV, but every
  packet remains review-blocked and `accepted_for_validation=false`. Three
  priority figure crops now have axis-calibration draft scaffolds: Cikhardtova
  2015 Fig. 6, Klir 2011 Fig. 2, and Springham 2021 Fig. 5. These scaffolds
  are not validation evidence. Springham 2021 Fig. 5 now also has a 14-point
  mono-energetic curve draft packet with draft round-trip RMS residual
  `0.002049609754498783 px` and max residual `0.0031865149536866814 px`; it
  remains blocked on independent review and accepted review status only. The
  companion Gaussian-curve draft packet extracts the visible 200 keV and
  400 keV FWHM curves as review-blocked draft data. Klir 2011 Fig. 2 now has
  a timing-response draft packet for the visible FWHM and rise-time curves;
  error-bar magnitudes remain open. Cikhardtova 2015 Fig. 6 is explicitly
  blocked for manual/vector curve separation because five monochrome line
  styles overlap and merge. The A14 independent-review handoff now lists 9
  reviewable draft packets and 3 context-only axis
  scaffolds while preserving `accepted_for_validation=false` for every item.
  The A14 remaining-extraction backlog now reports 18 ready-not-started crops,
  9 manual-review crops, 1 blocked crop, and 8 distinct crops with reviewable
  drafts.
- Scientific acceptance remains blocked, not incomplete-by-plumbing, for the
  tasks that require new evidence: independent Akel review acceptance, accepted
  S1/S2 waveform/current-dip data with uncertainty, remaining Akel Fig. 2-6
  digitization/review, A14 numeric extraction, same-device
  phase targets, same-scope Tier-4 spatial packets, same-scope Tier-5 neutron packets,
  and real same-scope physics/coupling/UQ validation evidence.
- The correct current product behavior is therefore explicit blockers and
  non-promoting candidates, not predictive/high-fidelity support.

### Track B: SRS/Productization Closure

| ID | Task | Goals and objectives | Guardrails | Skills and methods | Exit evidence |
| --- | --- | --- | --- | --- | --- |
| B1 | Formal SRS baseline | Convert `docs/DPF_UNIFIED_SRS_DRAFT.md` into a baselineable SRS with stable IDs, owners, priorities, status, and verification mappings. | Do not baseline speculative capabilities as implemented. Keep scientific-source claims tied to KR evidence. | Requirements engineering, IEEE/ISO SRS structure, traceability design, Doorstop candidate requirement tree. | SRS table links every P0/P1 requirement to test/inspection/analysis/demo and an owner/status. |
| B2 | Compute-authority model | Decide whether to adopt T0/T2 from the template or define DPF-Unified-specific authority labels. | Do not call MLX float32 a certification backend unless validated for the claim. Do not demote existing useful MLX evidence; label it correctly. | Architecture decision record, backend audit, precision/risk analysis. | ADR and SRS update define Reference/Preview or equivalent labels and promotion rules. |
| B3 | Result classification labels | Add output labels such as `Reference`, `Preview`, `Derived Diagnostic`, `Exploratory`, `Superseded`, and `Invalid`. | Labels must fail closed. Preview or draft evidence cannot be promoted by UI/API convenience. | Data modeling, schema design, regression tests. | Result metadata and tests prove unsupported/draft/preview outputs cannot masquerade as reference evidence. |
| B4 | Run manifest schema | Emit a manifest for each run with input hashes, backend, solver mode, hardware profile, dependency hashes, seed, output list, and validation status. | Manifest must be generated for failed and blocked runs too. Do not rely on prose logs as manifest substitutes. | Schema design, hashing, runtime metadata collection, tests. | `run_manifest.json` or equivalent validates against schema in unit/integration tests. |
| B5 | Validation certificate schema | Define certificates with evidence links, requirement IDs, reviewer fields, pass/fail status, and supersession status. | Certificate creation must be impossible when gates are partial, draft, cross-scope, or failed. | V&V process design, schema validation, negative tests. | Certificate artifact is emitted only when linked gates pass; negative tests cover draft Akel and cross-scope packets. |
| B6 | Project lifecycle | Define create/load/duplicate/archive project behavior and preserve inputs, outputs, manifests, validation status, and logs. | Project operations must not mutate physics results silently. Archived projects remain reproducible or explicitly stale. | Product/API design, file layout, schema migration. | Project lifecycle tests cover create/load/duplicate/archive and metadata preservation. |
| B7 | Memory preflight and telemetry | Add projected memory budget and runtime telemetry rules for solver starts and long runs. | Unsafe memory estimates must block or require explicit override. Do not silently swap or downcast to make a run fit. | Performance modeling, MLX/Python memory telemetry, failure-code design. | Memory-exceeding test refuses launch; accepted runs record projected and peak memory telemetry. |
| B8 | Backend unsupported-feature warnings | Prevent selected physics flags from silently skipping on unsupported backends. | Unsupported requested physics should warn or fail according to severity. Do not break intentionally unavailable optional dependencies. | Backend capability matrix, config validation, warning/error tests. | Tests prove unsupported backend/physics combinations produce explicit diagnostics. |
| B9 | CLI/backend consistency | Align CLI backend choices with `SimulationConfig` and engine support, including `mlx` if supported. | If `mlx` is intentionally excluded from CLI, document why and fail with a clear message. | CLI/API review, Click tests, backend availability guards. | CLI tests cover `--backend mlx` or its explicit rejection path. |
| B10 | UI/API readiness surfacing | Expose result classification, predictive readiness, high-fidelity gaps, digitization status, and source blockers in UI/API. | Do not hide blockers behind summary quality scores. Draft packets must display draft/review-blocked state. | API schema, frontend review, status UX, snapshot tests. | API/UI tests show blockers for Akel draft, missing spatial/neutron/UQ, and preview outputs. |
| B11 | Export bridge scope and acceptance | Decide v1.0 export scope for HDF5, Well, VTK/VTU, CGNS, OpenFOAM, and Ansys/PyMAPDL. | If external tool support is not tested, mark it deferred. Do not imply export correctness from file creation alone. | Data exchange, schema tests, external smoke tests, product scoping. | SRS marks each export required/deferred/rejected and tests accepted exports for units/provenance/readability. |
| B12 | Local-first/security controls | Formalize no hardware control, local-first network default, classification metadata, runtime AI boundaries, and audit logs. | Do not add hidden network calls or hardware-control endpoints. AI agents must not mutate active simulation state at runtime. | Security review, network audit, metadata schema, process audit. | Security tests/inspection show local-only default, no hardware drivers, classification fields, and runtime AI absence. |
| B13 | Air-gap build and release gate | Define offline install/test path and pinned dependency/hash expectations where licensing allows. | Do not promise air-gap support for dependencies that cannot be legally vendored. | Release engineering, dependency locking, CI design. | Air-gap runbook and reproducible baseline test logs exist. |
| B14 | Current TODO audit refresh | Replace historical `docs/todo_audit.md` with a current audit against decomposed engine paths and active source tree. | Do not carry stale `src/dpf/engine.py` references forward as live blockers. Classify comments as bug, deferred, benign, or obsolete. | Static search, source inspection, issue triage, documentation. | New TODO audit lists current blockers and maps any real bugs into SRS or engineering backlog. |

### Execution Order

1. Close A1 immediately so the plan artifacts stop contradicting the latest log.
2. Run A2 and A3 next because they control S1/S2 and prevent false waveform
   acceptance.
3. Start B1, B2, B3, B4, and B5 in parallel with A2/A3 as the minimum SRS
   productization spine.
4. Use A4/A5/A14 to expand source coverage only after the review gate pattern is
   stable.
5. Continue A6-A12 as the high-fidelity scientific closure path; each task must
   leave readiness blocked until real same-scope evidence exists.
6. Use B6-B14 to turn the workbench into a releasable, auditable product rather
   than only a scientifically honest research code.

### Task Completion Log

- 2026-05-12 user PDF intake, KR promotion, and source-fidelity review:
  completed staging for the new supplied local batch. Evidence:
  `scripts/stage_user_pdf_batch_2026_05_12.py` generated
  `docs/USER_PDF_INTAKE_2026_05_12.json`, `.md`, and `.csv` with 39 readable
  inputs, 35 unique SHA-256 payloads, 4 duplicate input paths, and 0
  missing/read failures. `scripts/promote_user_pdf_batch_2026_05_12.py`
  promoted 28 new selected physics/method records into `KnowledgeReference/`,
  repaired the false Trunk 1975/Kortanek 2014 generic-title match, left 7 non-physics/AI-only
  support PDFs staged but not promoted, and chunked 6 book-length sources into
  126 page-range Markdown chunks. `scripts/verify_user_pdf_batch_source_fidelity_2026_05_12.py`
  updated 28 KR records with source-fidelity reviews, detecting 1,698 figure
  captions, 293 table-caption hits, 68 extracted table matrices, 25,298
  formula-like lines, 4,423 numeric target contexts, 1,666 uncertainty
  contexts, 1,433 image blocks, and 11,376 recovered secondary-extraction
  items. Checks run: `py_compile` passed for the three intake scripts; the
  promotion report shows 28/28 parity checks passed. Boundary: this is source
  availability and copy-fidelity only, not accepted target/digitization
  evidence.
- 2026-05-12 May batch target triage: completed the first backlog split for
  newly local May 12 records. Evidence:
  `scripts/create_user_pdf_may12_target_triage.py` generated
  `docs/USER_PDF_MAY12_TARGET_TRIAGE_2026_05_12.json` and `.md` with 28
  entries, 5 target-extraction candidates, 4 P1 source-review candidates, 1 P2
  source-review candidate, 3 P3 context/materials records, and 20 method
  references. Boundary: this is a planning report only; every target candidate
  still requires source-line review, unit normalization, typed target records,
  uncertainty handling, and reviewed digitization for figure/table values.
- 2026-05-12 May batch source validation: completed source-level validation
  for the corrected May 12 batch. Evidence:
  `scripts/validate_user_pdf_may12_sources.py` generated
  `docs/USER_PDF_MAY12_SOURCE_VALIDATION_2026_05_12.json` and `.md` with 28
  promoted source records checked, 7 stage-only records checked, 5
  source-validated target-extraction candidates, 23 source-validated
  method/context records, and 0 validation failures. The report also records
  the Trunk 1975 false-match repair and confirms the Kortanek 2014
  source-fidelity annotation is repaired and validated. Boundary: source-level
  validation does not accept target values, plotted curves, tables, formula
  thresholds, uncertainty values, or simulation validation criteria.
- 2026-05-11 A14 remaining-extraction backlog: completed the generated backlog
  tying the crop review, draft packets, and Cikhardtova blocker into one
  status artifact. Evidence: `scripts/create_a14_remaining_extraction_backlog.py`
  generated `docs/A14_REMAINING_EXTRACTION_BACKLOG_2026_05_11.json` and `.md`
  with 36 crop candidates, 9 reviewable draft packets across 8 distinct crops,
  18 ready-not-started crops, 9 manual-review crops, 1 blocked crop, and 0
  accepted validation items. Checks run: `py_compile` passed; A14 backlog gate
  passed; `python3 -m pytest tests/test_digitization.py -q` passed
  (`37 passed`).
- 2026-05-11 A14 Cikhardtova Fig. 6 extraction blocker: completed the safe
  non-extraction decision for the remaining priority axis scaffold. Evidence:
  `scripts/create_a14_cikhardtova_fig6_extraction_blocker.py` generated
  `docs/A14_CIKHARDTOVA_FIG6_EXTRACTION_BLOCKER_2026_05_11.json` and `.md`
  with source/PDF/figure hashes, five visible series labels, blocker reason,
  and required next steps for manual or vector-assisted curve separation.
  Checks run: `py_compile` passed; Cikhardtova blocker gate passed; `python3 -m
  pytest tests/test_digitization.py -q` passed (`36 passed`).
- 2026-05-11 A14 Klir Fig. 2 timing-response draft extraction: completed a
  review-blocked numeric draft for the visible PMT response curves. Evidence:
  `scripts/create_a14_klir_fig2_timing_response_draft.py` generated
  `KnowledgeReference/digitization/a14-2026-05-11-klir-fig2-timing-response-draft-packet.json`
  and `docs/A14_KLIR_FIG2_TIMING_RESPONSE_DRAFT_2026_05_11.md` with two draft
  series: FWHM and rise time versus PMT voltage. Error-bar magnitudes are
  explicitly not digitized in this packet. The handoff now lists 9 reviewable
  draft packets and 3 context-only axis scaffolds. Checks run: `py_compile`
  passed; Klir Fig. 2 draft gate passed; `python3 -m pytest
  tests/test_digitization.py -q` passed (`35 passed`).
- 2026-05-11 A14 Springham Fig. 5 Gaussian-curve draft extraction: completed a
  second Springham Fig. 5 numeric draft packet for the visible Gaussian
  response curves without modifying the mono-energetic packet. Evidence:
  `scripts/create_a14_springham_fig5_gaussian_curve_drafts.py` generated
  `KnowledgeReference/digitization/a14-2026-05-11-springham-fig5-gaussian-curves-draft-packet.json`
  and `docs/A14_SPRINGHAM_FIG5_GAUSSIAN_CURVES_DRAFT_2026_05_11.md` with two
  draft series: Gaussian peak neutrons at 200 keV FWHM and 400 keV FWHM. The
  handoff now lists 9 reviewable draft packets and 3 context-only axis
  scaffolds. Checks run: `py_compile` passed; Gaussian draft gate passed;
  `python3 -m pytest tests/test_digitization.py -q` passed (`34 passed`).
- 2026-05-11 A14 source-PDF review-gate hardening: completed the next hash
  binding guardrail for A14 digitization packets. Evidence:
  `digitization_verification_evidence()` now verifies a declared
  `source_pdf_path`/`source_pdf_sha256` pair and accepted review metadata must
  match `reviewed_source_pdf_sha256` when a packet declares a local PDF.
  `scripts/create_a14_independent_review_handoff.py` regenerated the handoff
  template with `reviewed_source_pdf_sha256` in review metadata. Checks run:
  `py_compile` passed; A14 source-PDF/table gate check passed; `python3 -m
  pytest tests/test_digitization.py -q` passed (`30 passed`).
- 2026-05-11 A14 Springham Fig. 5 accepted-review fixture hardening: added the
  figure-specific future acceptance path and negative review checks without
  accepting the packet. Evidence: `tests/test_digitization.py` now proves the
  Springham Fig. 5 packet can pass only when synthetic accepted review metadata
  binds to the current packet, source, local PDF, and figure hashes; stale
  source-PDF or figure-image review hashes fail. Checks run: `py_compile`
  passed; Springham accepted-review fixture gate passed; `python3 -m pytest
  tests/test_digitization.py -q` passed (`33 passed`).
- 2026-05-11 A14 independent-review handoff and table review hardening:
  completed the reviewer-facing handoff bundle and tightened table review
  gates without accepting any packet. Evidence:
  `scripts/create_a14_independent_review_handoff.py` generated
  `docs/A14_INDEPENDENT_REVIEW_HANDOFF_2026_05_11.json` and `.md` with 7
  reviewable draft packets, 3 context-only axis scaffolds, required review
  fields/checklist, source/crop hashes, and zero accepted validation packets.
  `a14_table_extraction_draft_packets()` now supplies per-table item hashes,
  and `digitization_verification_evidence()` verifies table crop hashes and
  accepted-review crop-hash binding. Checks run: `py_compile` passed; handoff
  invariant check passed; `python3 -m pytest tests/test_digitization.py -q`
  passed (`28 passed`).
- 2026-05-11 A14 Springham Fig. 5 draft residual check: closed the stale
  residual blocker on the first A14 figure numeric draft without promoting it
  to validation evidence. Evidence:
  `scripts/create_a14_springham_fig5_digitization_draft.py` regenerated
  `KnowledgeReference/digitization/a14-2026-05-11-springham-fig5-monoenergetic-draft-packet.json`
  and `docs/A14_SPRINGHAM_FIG5_DIGITIZATION_DRAFT_2026_05_11.md` with draft
  round-trip RMS residual `0.002049609754498783 px` and max residual
  `0.0031865149536866814 px`. `digitization_verification_evidence()` now
  intentionally fails only on `independent_review_missing` and
  `review_status_not_accepted`. Checks run: `py_compile` passed; residual gate
  check passed; `python3 -m pytest tests/test_digitization.py -q` passed
  (`24 passed`).
- 2026-05-11 A14 Springham Fig. 5 mono-energetic draft extraction: completed
  the first A14 figure numeric draft packet. Evidence:
  `scripts/create_a14_springham_fig5_digitization_draft.py` generated
  `KnowledgeReference/digitization/a14-2026-05-11-springham-fig5-monoenergetic-draft-packet.json`
  and `docs/A14_SPRINGHAM_FIG5_DIGITIZATION_DRAFT_2026_05_11.md` with 14
  candidate mono-energetic Zr/Be count-ratio versus effective-energy points,
  source hash, local PDF hash, crop hash, axis calibration, and pixel-pick
  metadata. The follow-on residual check measured draft round-trip residuals
  and left the packet blocked only on independent review/status. Checks run:
  `py_compile` passed; draft gate check passed; `python3 -m pytest
  tests/test_digitization.py -q` passed (`24 passed`).
- 2026-05-11 A14 axis-calibration draft scaffolds: completed the first
  source-bound figure calibration scaffolds for Cikhardtova 2015 Fig. 6, Klir
  2011 Fig. 2, and Springham 2021 Fig. 5. Evidence:
  `scripts/create_a14_axis_calibration_drafts.py` generated
  `KnowledgeReference/digitization/a14-2026-05-11-axis-calibration-draft-packets.json`
  and `docs/A14_AXIS_CALIBRATION_DRAFTS_2026_05_11.md` with 3 packets, 0
  accepted validation packets, source hashes, local PDF hashes, crop hashes,
  visible axis ranges, visible series names, and extraction notes. The packets
  intentionally contain no digitized arrays or residuals. Checks run:
  `py_compile` passed; axis draft invariant check passed; `python3 -m pytest
  tests/test_digitization.py -q` passed (`23 passed`).
- 2026-05-11 A14 crop-boundary rectification: completed the six flagged crop
  rectangle fixes and regenerated the target-extraction and crop-boundary
  reports. Evidence: `scripts/start_target_extraction_digitization.py` now
  produces corrected crops for Cikhardtova 2015 Fig. 5, Klir 2011 Figs. 1/3/4,
  and Catenacci 2020 Figs. 1/2. `docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.json`
  now records 21 `boundary_ready_for_draft_extraction` figure crops, 9
  `manual_review_required` crops, 6 `draft_extracted_review_blocked` table
  crops, 0 `crop_adjustment_needed` crops, and 0 accepted validation packets.
  Checks run: `py_compile` passed; A14 invariant check passed; `python3 -m
  pytest tests/test_digitization.py -q` passed (`22 passed`).
- 2026-05-11 A14 crop-boundary QA inventory: completed the all-source crop
  boundary status artifact without accepting any validation evidence. Evidence:
  `scripts/create_a14_crop_boundary_review.py` generated
  `docs/A14_CROP_BOUNDARY_REVIEW_2026_05_11.json` and `.md`. The report records
  36 crop entries, 30 figure crops, 6 table crops, 21
  `boundary_ready_for_draft_extraction` figure crops, 9 `manual_review_required`
  crops, 0 `crop_adjustment_needed` crops, 6 `draft_extracted_review_blocked`
  table crops, and 0 accepted validation packets after the follow-on crop
  rectification. The recommended next axis calibration crops are Cikhardtova
  2015 Fig. 6, Klir 2011 Fig. 2, and Springham 2021 Fig. 5. Checks run:
  `py_compile` passed and
  `python3 -m pytest tests/test_digitization.py -q` passed (`22 passed`).
- 2026-05-11 A14 table draft extraction: completed the first source-bound
  table extraction pass for Springham 2021 Tables 1-2 and Catenacci 2020
  Tables I-IV. Evidence:
  `scripts/create_a14_table_extraction_drafts.py` generated
  `KnowledgeReference/digitization/a14-2026-05-11-table-draft-packets.json`
  and `docs/A14_TABLE_EXTRACTION_DRAFTS_2026_05_11.md` with 6 draft packets,
  0 accepted validation packets, local KR source hashes, local PDF hashes,
  crop-image hashes, source line windows, table rows, and numeric series.
  `digitization_verification_evidence()` fails every packet only on
  `independent_review_missing` and `review_status_not_accepted`. Checks run:
  `py_compile` passed; report invariant check passed with 5/23/36/0 for the
  crop workbench; `python3 -m pytest tests/test_digitization.py -q` passed
  (`21 passed`).
- 2026-05-11 A14 crop-generation expansion: completed the all-source
  workbench crop pass. Evidence:
  `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.json` now reports 5 local
  KR-backed source tasks, 23 rendered pages, 36 unreviewed crop candidates, and
  0 accepted validation packets. The 36 crops cover Cikhardtova 2015 Figs. 1-6,
  Szydlowski 2004 Figs. 1-5, Klir 2011 Figs. 1-4, Springham 2021 Figs. 1-7 and
  Tables 1-2, and Catenacci 2020 Figs. 1-8 and Tables I-IV. Representative
  Klir, Springham, and Catenacci crop boundaries were visually spot-checked and
  adjusted where captions, rotated axes, or table boundaries were clipped.
  Remaining A14 work is crop-boundary review notes, axis/table calibration,
  numeric extraction, residual checks, and independent review.
- 2026-05-11 A14 target-extraction/digitization lane: started and now tracked
  as a first-class Track A work item. Evidence:
  `scripts/start_target_extraction_digitization.py` generated
  `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md` / `.json` with 5 local
  KR-backed source tasks, 23 rendered pages, 36 unreviewed crop candidates, and
  0 accepted validation packets. Crop candidates now cover Cikhardtova 2015
  Figs. 1-6, Szydlowski 2004 Figs. 1-5, Klir 2011 Figs. 1-4, Springham 2021
  Figs. 1-7 and Tables 1-2, and Catenacci 2020 Figs. 1-8 and Tables I-IV.
  Remaining A14 work is crop-boundary review, axis/table calibration, numeric
  extraction, residual checks, and independent review. These artifacts are
  workbench material only and cannot support validation until accepted packets pass
  `digitization_verification_evidence()`.
- 2026-05-08 A2/A3 digitization and waveform-comparator guardrail expansion:
  completed. Evidence: accepted review metadata must now bind to packet hash,
  source hash, figure hash, task ID, validation scope, reviewer, review date,
  review notes, and accepted decision. The S1/S2 comparator continues to refuse
  metrics for draft, stale-review, malformed-review, cross-scope, and
  missing-uncertainty states. Additional integration tightening requires the
  packet itself to carry a packet hash before review metadata can accept it.
  Checks run: `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/kr_targets.py src/dpf/validation/source_acquisition.py tests/test_digitization.py tests/test_kr_targets.py tests/test_source_acquisition.py`
  and `python3 -m pytest tests/test_digitization.py tests/test_kr_targets.py tests/test_source_acquisition.py -q`
  passed (`109 passed`). Remaining scientific blockers: no real independent
  accepted Akel Fig. 1 review packet exists, and same-scope per-point
  current/timing uncertainty remains absent.
- 2026-05-08 A3 production waveform-comparison attachment: completed for app
  result surfaces. Evidence: `_apply_post_processing()` now attaches
  `pf1000_16kv_current_waveform_comparison_candidate` to production-style app
  results using the current Akel Fig. 1 draft packet when no accepted packet is
  supplied. Current app results therefore carry the S1/S2 blocker as
  `waveform_comparison_status="blocked_by_review"` with
  `metrics_computed=False`, rather than leaving the comparator as a standalone
  helper. Checks run: `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  and `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q`
  passed (`1 passed`). Scientific acceptance remains blocked.
- 2026-05-08 A3 run-manifest waveform blocker propagation: completed for
  artifact traceability. Evidence: `RunManifest` now carries compact
  `validation_evidence`, and `build_run_manifest()` copies blocker-oriented
  fields from known validation summary packets, including
  `pf1000_16kv_current_waveform_comparison_candidate` and nested
  digitization-readiness status. Bulk candidate trace arrays are not copied into
  the manifest. Checks run:
  `python3 -m py_compile src/dpf/validation/artifacts.py tests/test_validation_artifacts.py`
  and the focused validation-artifact slice passed (`3 passed`). This makes
  blocked S1/S2 evidence auditable in run artifacts without promoting draft
  Akel data.
- 2026-05-08 A5 source-acquisition queue matrix: completed for machine-readable
  blocker triage. Evidence: `scientific_closure_source_acquisition_queue()` now
  reports summary counts, same-scope group statuses, `source_action`, and
  blocked validation tiers for each blocker. Current PF-1000 full-energy queue
  state is 10 blockers, 5 priority-1 items, 5 priority-2 items, 7 local
  digitization/target-extraction actions, 5 items with user-acquisition
  requirements, 2 complete same-scope groups, 10 partial same-scope groups, and
  0 missing same-scope groups. `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` records
  the updated queue semantics. This closes queue detail, not the underlying
  evidence gaps.
- 2026-05-08 A7 numerical-fidelity claim-boundary guardrail: completed.
  Evidence: MHD numerical-fidelity evidence now carries
  `evidence_class="code_numerical_verification"`, explicitly marks itself as
  not experimental DPF validation, not predictive scientific support, not
  high-fidelity scientific support, and not a substitute for Tier 4 spatial or
  Tier 5 neutron validation. Backend parity evidence is labeled
  `BackendParityVerification`, distinct from Reference scientific authority.
  Checks run: `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py tests/test_mhd_numerical_fidelity.py`
  and `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed
  (`22 passed`). Tier 4 and Tier 5 remain blocked on same-scope experimental
  evidence.
- 2026-05-08 A7 restart/reproducibility evidence guardrail: completed.
  Evidence: MHD numerical-fidelity evidence now requires
  `restart_reproducibility` as a Tier-3 code-verification channel. The new
  `restart_reproducibility_evidence_from_results()` helper requires continuous
  and restarted observables, a restart/checkpoint marker, matching config
  hashes, and tolerance-bounded relative errors before a restart packet can
  support the audit. Checks run:
  `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py tests/test_mhd_numerical_fidelity.py src/dpf/validation/__init__.py`
  and `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed
  (`25 passed`). This remains code reproducibility evidence only, not DPF
  experimental validation or Reference scientific authority.
- 2026-05-08 A7 production Tier-3 verification packet status: completed as a
  production-visible blocker map. Evidence:
  `mhd_numerical_verification_packet_status()` now reports every required
  Tier-3 packet as `attached_validated`, `attached_non_validating`, or
  `missing_required`, and app post-processing exports
  `mhd_numerical_verification_packet_status` beside the main
  `mhd_numerical_fidelity` audit. Checks run:
  `python3 -m py_compile app_mhd.py src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/__init__.py src/dpf/validation/quality_assessment.py tests/test_mhd_numerical_fidelity.py tests/test_mhd_physics_integration.py`
  and `python3 -m pytest tests/test_mhd_numerical_fidelity.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q`
  passed (`28 passed`). Remaining A7 work is to generate or attach real
  scheduled verification packets for the claimed backend/scope; packet status
  reporting itself is now wired.
- 2026-05-09 A7 scheduled Tier-3 packet artifact path: completed for the first
  partial local packet. Evidence:
  `build_mhd_numerical_verification_packet()` assembles same-scope Tier-3
  packets from explicit verification-run outputs and immediately applies the
  production fail-closed packet status. The new
  `scripts/build_mhd_tier3_numerical_packet.py` ran the local cylindrical
  z-pinch convergence and implicit resistive-diffusion convergence studies and
  wrote `results/mhd_tier3_numerical_packet.json`. The packet is correctly
  `production_packet_status="blocked"` with attached validated packets for
  `finite_volume_mhd_verification`, `cylindrical_geometry_verification`,
  `circuit_coupled_energy_verification`,
  `resistive_or_nonideal_verification`, `convergence_study`, and
  `backend_parity`, and `dpf_scope_limit`; it still names
  `restart_reproducibility` as the missing same-scope packet. Finite-volume
  evidence is preview-backend MLX pytest/JUnit evidence, and backend parity is
  the existing Python-cylindrical vs MLX current-NRMSE gate.
  Checks run:
  `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/__init__.py scripts/build_mhd_tier3_numerical_packet.py tests/test_mhd_numerical_fidelity.py`,
  `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed
  (`29 passed`), and
  `python3 -m pytest tests/test_mlx_acceptance.py::TestStandardShockTubes::test_s5_sod_cross_backend_parity tests/test_mlx_acceptance.py::TestStandardShockTubes::test_s6_briowu_compound_waves tests/test_mlx_acceptance.py::TestStandardShockTubes::test_s7_sod_convergence -q --junitxml=results/mhd_finite_volume_mlx_shock_tubes.junit.xml`
  passed (`3 passed`); `python3 scripts/record_mhd_finite_volume_pytest_evidence.py --junitxml results/mhd_finite_volume_mlx_shock_tubes.junit.xml --output results/mhd_finite_volume_mlx_shock_tubes_evidence.json`
  produced passing finite-volume evidence;
  `python3 -m pytest tests/test_cross_backend_parity.py -q --junitxml=results/mhd_backend_parity_cross_backend_current.junit.xml`
  passed (`1 passed`); `python3 scripts/record_mhd_backend_parity_pytest_evidence.py --junitxml results/mhd_backend_parity_cross_backend_current.junit.xml --output results/mhd_backend_parity_cross_backend_current_evidence.json`
  produced passing backend-parity evidence; and
  `python3 scripts/build_mhd_tier3_numerical_packet.py --mhd-verification-file results/mhd_finite_volume_mlx_shock_tubes_evidence.json --backend-parity-file results/mhd_backend_parity_cross_backend_current_evidence.json --output results/mhd_tier3_numerical_packet.json`
  completed. This is Tier-3 code numerical verification only, not experimental
  DPF validation or Reference scientific authority.
- 2026-05-09 A7 same-scope restart evidence and complete Tier-3 packet:
  completed for code numerical verification. Evidence:
  `scripts/build_mhd_restart_reproducibility_evidence.py` runs a deterministic
  CPU checkpoint/restart fixture, compares uninterrupted and restarted circuit
  plus field-norm observables, and writes
  `results/mhd_restart_reproducibility_evidence.json`. The restart packet
  passed with `max_relative_error=0.0`, matching config hashes, checkpoint
  marker present, and no missing metrics. The updated
  `scripts/build_mhd_tier3_numerical_packet.py` accepts
  `--restart-reproducibility-file`; rebuilding
  `results/mhd_tier3_numerical_packet.json` with finite-volume, backend-parity,
  and restart evidence now reports `production_packet_status="complete"`,
  `missing_required_packets=[]`, and attached validated packets for all eight
  required Tier-3 channels. Checks run:
  `python3 -m py_compile scripts/build_mhd_restart_reproducibility_evidence.py scripts/build_mhd_tier3_numerical_packet.py`,
  `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed
  (`30 passed`),
  `python3 scripts/build_mhd_restart_reproducibility_evidence.py --output results/mhd_restart_reproducibility_evidence.json`
  produced passing restart evidence, and
  `python3 scripts/build_mhd_tier3_numerical_packet.py --mhd-verification-file results/mhd_finite_volume_mlx_shock_tubes_evidence.json --backend-parity-file results/mhd_backend_parity_cross_backend_current_evidence.json --restart-reproducibility-file results/mhd_restart_reproducibility_evidence.json --output results/mhd_tier3_numerical_packet.json`
  completed the packet. This remains Tier-3 code numerical verification only;
  it does not close Tier-4 spatial validation, Tier-5 neutron validation, or
  Reference scientific authority.
- 2026-05-08 A6/A8 production blocker status surfaces: completed as
  non-promoting status reporting. Evidence: app post-processing now emits
  `snowplow_phase_validation_status` for phase-history output even when
  verified phase targets are absent, and `spatial_validation_scope_closure` is
  emitted even when no spatial components are supplied. Checks run:
  `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` and
  the focused app integration slice passed (`3 passed`). Tier 2 and Tier 4
  remain evidence-blocked until same-device KR phase targets and same-scope
  density/magnetic-field/temperature packets exist.
- 2026-05-08 A12 tier-grouped uncertainty reporting: completed as reporting
  polish. Evidence: `validation_uncertainty_coverage_from_result()` now reports
  `tier_uncertainty_status` for T1-T5, including present observables and
  missing uncertainty per tier, and `uncertainty_evidence_from_result()` carries
  that tier map forward. Checks run:
  `python3 -m py_compile src/dpf/validation/uncertainty_budget.py tests/test_uncertainty_budget.py`
  and `python3 -m pytest tests/test_uncertainty_budget.py -q` passed
  (`13 passed`). Real UQ support still requires same-scope KR uncertainty
  values and full component validation.
- 2026-05-08 Track A code-ready closure sweep: completed. Evidence: the
  multi-agent audit found the remaining small code-ready gaps in A3, A6, A8,
  A9, and A12; the implementation sweep closed them with run-manifest blocker
  propagation, phase/spatial blocker status packets, stricter neutron
  detector/UQ same-scope closure, and tier-grouped UQ reporting. Verification:
  `python3 -m py_compile app_mhd.py src/dpf/validation/quality_assessment.py src/dpf/validation/uncertainty_budget.py src/dpf/validation/artifacts.py src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/physics_fidelity.py src/dpf/validation/circuit_field_coupling.py tests/test_mhd_physics_integration.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_uncertainty_budget.py tests/test_validation_artifacts.py tests/test_mhd_numerical_fidelity.py tests/test_physics_fidelity.py tests/test_circuit_field_coupling.py`
  passed; `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_uncertainty_budget.py tests/test_validation_artifacts.py tests/test_mhd_numerical_fidelity.py tests/test_physics_fidelity.py tests/test_circuit_field_coupling.py -q`
  passed (`263 passed, 3 skipped`). Remaining Track A items are evidence or
  independent-review campaigns, not open guardrail plumbing.
- 2026-05-08 A10 per-run physics-fidelity claim matrix: completed as a
  fail-closed reporting guardrail. Evidence:
  `physics_fidelity_evidence_from_result()` now preserves detailed effect
  statuses while adding canonical `fidelity_status` values
  (`implemented`, `verified`, `validated`, `empirical`, `absent`, or
  `bounded_out`), per-effect `blocks_claims`, a top-level `claim_blockers`
  matrix, and `blocked_claims`. Missing or empirical physics now blocks only
  the mapped predictive claims, not non-predictive engineering runs. Checks run:
  `python3 -m py_compile src/dpf/validation/physics_fidelity.py tests/test_physics_fidelity.py`
  and `python3 -m pytest tests/test_physics_fidelity.py -q` passed
  (`9 passed`). Remaining blocker: real KR-backed same-scope effect evidence is
  still required before late-pinch, neutron, high-Z, p-B11, or high-fidelity
  MHD claims can pass.
- 2026-05-08 A11 staged circuit-field coupling authority: completed as a
  fail-closed guardrail. Evidence: field-coupling evidence now requires
  `coupling_interval_authority` and reports staged labels for
  `snowplow_loaded`, `blended`, `field_derived_candidate`, and
  `validated_field_coupled`. Density-weighted/`Lp_mhd_nH` coupling remains
  candidate-only and cannot support validated field-coupled authority without
  same-scope KR-backed component validation. Checks run:
  `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py tests/test_circuit_field_coupling.py`
  and `python3 -m pytest tests/test_physics_fidelity.py tests/test_circuit_field_coupling.py -q`
  passed (`21 passed`). Predictive/high-fidelity field-coupling claims remain
  blocked until same-scope field-coupling evidence exists.
- 2026-05-08 A9 mechanism-separated neutron output reporting: completed as a
  non-promoting production guardrail. Evidence: app post-processing now attaches
  `neutron_mechanism_outputs` when neutron-yield estimates or time-resolved
  neutron histories are present. The summary separates thermonuclear and
  beam-target yields/histories, reports detector/activation, timing, spectrum,
  anisotropy, and UQ blockers, and sets `validation_status` to
  `estimate_not_validation`. Checks run:
  `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py` and
  focused app tests passed (`3 passed`). Tier 5 remains blocked until one
  same-scope KR-backed packet supplies scalar yield, mechanism timing,
  spectrum, anisotropy, detector/activation response, and uncertainty.
- 2026-05-08 A9 same-scope neutron closure gate: completed as a fail-closed
  Tier-5 guardrail. Evidence: `neutron_validation_scope_closure_report()` now
  requires scalar yield, mechanism timing, spectrum, anisotropy,
  detector/activation response, and explicit source uncertainty to contribute
  within one validation scope before Tier 5 can report support. Missing
  detector-response evidence or uncertainty source values are now named in the
  closure packet instead of being implied by timing/spectrum/anisotropy
  support. Checks run:
  `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py`
  and the focused A9 quality/KR/app slice passed (`8 passed`). Real neutron
  validation remains blocked until a same-scope KR-backed packet supplies those
  observables and uncertainty values.
- 2026-05-08 A12 UQ source-value guardrail: completed as fail-closed
  implementation. Evidence: uncertainty component evidence and KR uncertainty
  evidence now require explicit source uncertainty values before supporting UQ
  targets. A KnowledgeReference citation plus validation scope is no longer
  enough to create passing UQ evidence. Checks run: combined focused suite
  `python3 -m pytest tests/test_digitization.py tests/test_kr_targets.py tests/test_source_acquisition.py tests/test_uncertainty_budget.py tests/test_quality_assessment.py -q`
  passed (`176 passed`). Remaining blocker: real same-scope KR uncertainty
  packets with source values are still required for supported validation tiers.
- 2026-05-08 A13 long PF-1000 fixture policy: completed. Evidence:
  `tests/test_mlx_pf1000.py` keeps scientific long-fixture gates
  `xfail(run=False)` and source-blocked on S1/S2 closure. Endurance/regression
  probe paths in `tests/test_mlx_pf1000_probe.py` and
  `scripts/run_mlx_pf1000_probe.py` now require `DPF_MLX_RUN_ENDURANCE=1` and
  report `scientific_status=non_scientific`, target, cap, final time,
  cap-exhaustion status, and memory telemetry/unavailable marker.
  `docs/PF1000_LONG_FIXTURE_POLICY.md` records the policy. Checks run:
  `python3 scripts/run_mlx_pf1000_probe.py` refused with
  `ENDURANCE_NOT_OPTED_IN` and exit `3`; focused policy tests passed
  (`1 passed, 1 skipped`); PF-1000 gate tests remained blocked as intended
  (`5 passed, 14 xfailed`).
- 2026-05-08 Track A detailed simulation/physics breakdown: completed as a
  planning update. Evidence: Track A now has a subtask-level breakdown with
  current state, guardrails, objectives, methods, and exit evidence for A2-A13.
  The added breakdown distinguishes ready-to-code guardrail work from
  evidence-blocked scientific acceptance and the A13 policy decision. No
  scientific evidence was promoted.
- 2026-05-08 A1 findings/status hygiene: completed. Evidence: the current
  execution position now names source review closure, Akel S1/S2 review
  blockers, and the separate SRS/productization ratchet; the SRS draft,
  traceability tooling note, and both findings docs agree on the next work.
  Checks run: `dpf_skill_preflight.py /Users/anthonyzamora/dpf-unified`,
  `srs_trace_audit.py /Users/anthonyzamora/dpf-unified`, `git diff --check`,
  and TOML parse of `pyproject.toml`. No scientific evidence was promoted.
- 2026-05-08 B1 formal SRS baseline: completed for candidate-baseline stage.
  Evidence: `docs/DPF_REQUIREMENTS_BASELINE.md` now contains the first
  stable-ID P0/P1 requirements table with 47 unique `DPF-*` IDs, owner roles,
  current status, verification methods, and evidence/blocker links. The SRS
  draft and traceability tooling note now point to that candidate baseline.
  Doorstop tree import remains a follow-on after review; this completion does
  not mark blocked/scaffolded requirements as implemented.
- 2026-05-08 B2 compute-authority model: completed. Evidence:
  `docs/ADR_COMPUTE_AUTHORITY.md` now defines Reference/Preview/Derived
  Diagnostic/Exploratory/Superseded/Invalid labels and backend authority
  defaults. `src/dpf/validation/artifacts.py` implements fail-closed
  classification rules. `tests/test_validation_artifacts.py` proves MLX remains
  Preview even with accepted validation status unless future promotion rules are
  added, and proves Reference requires accepted evidence on a reference-candidate
  backend. Later 2026-05-08 entries supersede the original schema-stage note
  by wiring B3-B5 runtime output and certificate persistence guards.
- 2026-05-08 A2 review-gate hardening: completed as an implementation subtask;
  scientific acceptance remains blocked. Evidence:
  `digitization_verification_evidence()` now requires packet-tied review
  metadata when review count/status are accepting, so simply flipping
  `independent_review_count=1` and `review_status="accepted"` cannot promote a
  draft packet. Checks run: targeted digitization/KR tests passed (`4 passed`).
  Remaining A2 blocker: no real independent accepted review for the Akel Fig. 1
  packet exists.
- 2026-05-08 A3 waveform comparator scaffold: completed as guarded comparator
  implementation; S1/S2 validation remains blocked. Evidence:
  `pf1000_16kv_current_waveform_comparison_candidate_evidence()` now refuses to
  compute metrics for draft/review-blocked digitization, rejects cross-scope
  packets, requires current/time uncertainty metadata, and computes NRMSE plus
  current-dip depth/timing only for accepted same-scope packets. Targeted KR
  tests passed (`7 passed`). Remaining A3 blocker: real accepted Akel Fig. 1
  waveform packet and uncertainty are still absent.
- 2026-05-08 B3 result classification labels: completed for engine/runtime
  output. Evidence: `ResultClassification` is exported from
  `dpf.validation`, `SimulationEngine.run()` now attaches
  `validation_status`, `result_classification`, and `run_manifest` metadata to
  run summaries, and MLX remains Preview/non-certifying even if a caller passes
  accepted validation status. Checks run: `tests/test_validation_artifacts.py`
  passed (`13 passed`).
- 2026-05-08 B4 run manifest schema/runtime emission: completed for normal and
  failed engine runs. Evidence: file-backed engine runs now write a
  `*.run_manifest.json` sidecar with config hash, backend, solver mode,
  hardware profile, output hashes, validation status, and result
  classification. Failed runs attempt manifest emission before re-raising.
  Checks run: focused artifact tests passed (`13 passed`) and runtime smoke
  tests passed.
- 2026-05-08 B5 validation certificate schema/writer: completed for
  fail-closed artifact creation. Evidence: `ValidationCertificate`,
  `build_validation_certificate()`, and `write_validation_certificate()` reject
  accepted certificates with blocked, failed, draft, or cross-scope evidence
  before persistence. This does not create an Akel certificate because Akel Fig.
  1 remains `blocked_by_review`.
- 2026-05-08 B9 CLI/backend consistency: completed. Evidence:
  `dpf simulate --backend mlx` is now accepted by Click and passed through to
  the engine config, and `dpf backends` lists MLX availability. Checks run:
  focused CLI/backend tests passed (`11 passed`).
- 2026-05-08 B8 backend unsupported-feature diagnostics: completed. Evidence:
  `backend_feature_diagnostics()` now returns explicit warning/info records for
  skipped Athena/AthenaK/hybrid physics and GPU diffusion fallbacks,
  `SimulationEngine` logs those diagnostics and includes them in run summaries,
  and MLX now receives requested Hall, Braginskii conduction/viscosity, Nernst,
  and precision flags instead of silently dropping them. Checks run:
  `tests/test_backend_capabilities.py` passed (`3 passed`) and existing backend
  warning tests passed (`4 passed`).
- 2026-05-08 B7 memory preflight: completed for launch-time safety. Evidence:
  `run_memory_preflight()` estimates projected memory from grid/backend
  configuration before solver allocation, blocks unsafe launches above
  `diagnostics.memory_limit_fraction`, allows only explicit
  `diagnostics.allow_memory_overcommit`, and attaches the preflight record to
  run summaries. Checks run: `tests/test_memory_preflight.py` passed
  (`5 passed`).
- 2026-05-08 B14 current TODO audit refresh: completed. Evidence:
  `docs/todo_audit.md` now audits active `TODO`/`FIXME`/`XXX` markers against
  the decomposed source tree, excludes vendored/hidden/archive paths from live
  blocker status, classifies current findings as bug/deferred/benign/obsolete,
  and retires stale `src/dpf/engine.py` blockers because that file is absent.
  Checks run: active source marker scan, engine path check, docs/tooling marker
  scan, excluded-scope scan, and `git diff --check -- docs/todo_audit.md`.
- 2026-05-08 B12 local-first/security controls: completed for current release
  defaults and manifest metadata. Evidence: `dpf ui` and root Gradio launch now
  default to `127.0.0.1`, public Gradio share remains opt-in, FastAPI CORS
  defaults to localhost origins and rejects wildcard CORS unless
  `DPF_ALLOW_WILDCARD_CORS=1`, `local_first_security_audit()` scans active
  source for direct hardware-control imports and runtime-AI mutation paths, and
  `RunManifest` carries owner-supplied artifact classification/distribution
  metadata. Checks run: `tests/test_local_first_security.py` passed
  (`6 passed`) and `tests/test_validation_artifacts.py` passed (`14 passed`).
  Remaining related work: propagate classification metadata into non-manifest
  export schemas and decide audit-log depth.
- 2026-05-08 DPF-OPS-004 runtime memory telemetry: completed. Evidence:
  `RuntimeMemoryTelemetry` records process start/end/peak RSS, sample count,
  backend, and optional MLX active/peak backend memory when MLX exposes it.
  `SimulationEngine.run()` samples during normal, failed, and hybrid runs and
  attaches `runtime_memory_telemetry` to summaries before run manifests are
  built. Checks run: `tests/test_memory_preflight.py` passed (`8 passed`) and
  the artifact/backend slice passed (`17 passed`).
- 2026-05-08 B6 project lifecycle: completed for local project helpers.
  Evidence: `dpf.project.lifecycle` now provides create/load/duplicate/archive
  operations around a `project_manifest.json` and preserved `config.json`.
  The manifest tracks config hash, output paths, run-manifest paths,
  validation status, result classification, logs, archive metadata, and
  duplicate provenance. Loading rejects silent config mutation, duplicate
  preserves result files, and archive marks status without changing outputs.
  Checks run: `tests/test_project_lifecycle.py` passed (`4 passed`) and the
  memory/artifact regression slice passed (`22 passed`). Remaining related
  work: expose these lifecycle helpers through UI/API if v1.0 product scope
  requires it.
- 2026-05-08 B10 UI/API readiness surfacing: completed for authority and
  readiness visibility. Evidence: FastAPI `SimulationInfo` now exposes
  `validation_status`, fail-closed `result_classification`,
  `predictive_readiness`, `high_fidelity_readiness`, Akel
  `digitization_status`, and `source_blockers`. The GUI wire type mirrors
  those fields, the simulation store preserves the latest response, and the
  TopBar displays a Preview/Reference badge plus blocker count. Checks run:
  `tests/test_server_readiness.py` passed (`3 passed`), focused server
  lifecycle tests passed (`2 passed`), and `npm --prefix gui run typecheck`
  passed. Remaining related work: explicit units/dimensions API schema and
  broader UI mode requirements.
- 2026-05-08 B11 export bridge scope and acceptance: completed for v1 scope.
  Evidence: `docs/EXPORT_SCOPE_V1.md` and `export_scope_decisions()` accept
  DPF HDF5 diagnostics and Well HDF5, while explicitly deferring VTK/VTU,
  CGNS/HDF5, OpenFOAM, and Ansys/PyMAPDL until writer/readability or
  license-aware tests exist. HDF5 diagnostics now write schema/time-base root
  attributes and dataset units. The engine Well adapter now passes grid
  spacing, geometry, and simulation provenance instead of hardcoding `dx=1.0`.
  Checks run: `tests/test_export_scope.py` passed (`3 passed`) and the
  validation/project regression slice passed (`18 passed`). Remaining related
  work: fully embed run-manifest provenance inside HDF5 if v1 requires
  single-file artifacts.
- 2026-05-08 B13 air-gap build and release gate: completed for fail-closed
  gate definition; actual air-gap release remains blocked. Evidence:
  `docs/AIR_GAP_RELEASE_GATE.md` defines required artifacts and offline
  commands, while `airgap_release_gate()` checks for `dist/wheelhouse`,
  `dist/wheelhouse/SHA256SUMS`, and offline smoke/typecheck logs. The current
  repo correctly reports `passed=false` until those release artifacts exist.
  Checks run: `tests/test_airgap_gate.py` passed (`2 passed`) and the
  export/server readiness slice passed (`6 passed`). Remaining related work:
  produce license-reviewed wheelhouse artifacts and real offline logs.
- 2026-05-08 HDF5 embedded run metadata: completed for accepted HDF5
  diagnostics. Evidence: `_attach_run_artifacts()` embeds backend, solver mode,
  validation status, result label, validation-claim capability,
  classification JSON, artifact classification JSON, and KR source-authority
  text into HDF5 before the sidecar manifest hashes the file. Well exports now
  carry fail-closed `validation_status`/`result_label` defaults through
  `sim_params`. Checks run: `tests/test_validation_artifacts.py` passed
  (`14 passed`) and `tests/test_export_scope.py` passed (`3 passed`).
- 2026-05-08 project owner classification metadata: completed. Evidence:
  `ProjectManifest` now carries owner-supplied `artifact_classification`
  metadata using the same schema as run manifests. Create/load/duplicate/archive
  tests preserve classification/distribution fields. Checks run:
  `tests/test_project_lifecycle.py` passed (`4 passed`) and
  `tests/test_validation_artifacts.py` passed (`14 passed`).
- 2026-05-08 API units/dimensions schema: completed. Evidence:
  `/api/metadata/units` now returns canonical scalar, field, time-base, and
  authority metadata with units and dimensions, and the GUI client has a typed
  `UnitsMetadata` response. Checks run: `tests/test_server_metadata.py` passed
  (`2 passed`) and `npm --prefix gui run typecheck` passed.

## Execution Log

### 2026-05-05: KR Target Authority Manifest

- Added a machine-readable manifest for every coded KR validation target.
- Added a source audit for the manifest that checks local `KnowledgeReference/`
  files and line ranges through the existing source-authority helper.
- Exported the manifest and audit from `dpf.validation`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_target_manifest_lists_coded_targets tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`2 passed in 0.89s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`86 passed in 0.50s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.29s`)
  - `git diff --check` passed.
- Remaining limit: this audit proves target source authority, not semantic
  completeness. The next step is typed KR target extraction hardening.

### 2026-05-05: Typed KR Target Coverage Report

- Added a target coverage report that maps coded KR targets to the end-to-end
  observable groups needed for predictive validation.
- The report intentionally fails today because the target set is incomplete:
  `circuit_waveform` is missing, and `phase_timing` is partial.
- Exported the coverage report from `dpf.validation`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`2 passed in 0.80s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`87 passed in 0.55s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.36s`)
  - `git diff --check` passed.
- Remaining limit: coverage presence does not mean same-scope closure or
  simulation agreement. It only makes the target extraction backlog explicit.

### 2026-05-05: PF-1000 Partial Circuit Waveform Target

- Added a typed PF-1000 16 kV current-waveform target from the Akel 2021 KR
  source.
- The target records measured-current context, shot/fill/bank context,
  peak-current range, shot-12581 peak/pinch current, and the fact that the fit
  is valid only until the end of the current dip.
- The target is intentionally partial because digitized current trace points and
  per-point timing/current uncertainty are not extracted into the target packet.
- The coverage report now marks `circuit_waveform` as `partial` instead of
  `missing`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_pf1000_current_waveform_target_metadata tests/test_kr_targets.py::test_kr_validation_target_manifest_lists_coded_targets tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`4 passed in 2.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`88 passed in 0.52s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.43s`)
  - `git diff --check` passed.
- Remaining limit: this is not yet a full waveform target because the actual
  digitized current series and per-point uncertainty are not extracted.

### 2026-05-05: Lee-Course Full Phase-Timing Example Target

- Added a typed Lee/RADPF course example target for NX2 neon phase timing.
- The target records numeric axial end, radial start/end, radial duration,
  pinch start/end, pinch duration, radial shock axis time, and reflected-shock
  piston timing.
- The target is marked example-only for predictive purposes because it is a
  fitted worksheet example, not a same-shot deuterium experimental target with
  uncertainty.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_lee_course_nx2_phase_timing_example_metadata tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`3 passed in 2.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`89 passed in 0.66s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.44s`)
  - `git diff --check` passed.
- Remaining limit: full predictive tier 2 still needs same-device/same-shot
  deuterium phase timing targets with experimental uncertainty.

### 2026-05-05: App Exports KR Target Source And Coverage Reports

- App post-processing now exports `kr_validation_target_source_audit` and
  `kr_validation_target_coverage` beside predictive and high-fidelity readiness.
- The target source audit passes for the coded local targets.
- The target coverage report intentionally fails because target extraction is
  still incomplete, including partial `phase_timing`.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_target_source_audit_passes_for_local_targets -q` passed (`3 passed in 0.73s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`89 passed in 0.55s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.24s`)
  - `git diff --check` passed.
- Remaining limit: app export does not close target extraction. It makes the
  missing target groups visible on ordinary result payloads.

### 2026-05-05: KR Target Semantic Source-Window Audit

- Added `kr_validation_target_semantic_audit()` to check that every coded KR
  target's cited line windows contain expected domain markers for the extracted
  observable.
- Exported the semantic audit from `dpf.validation` and app post-processing, so
  ordinary result payloads now include source-file validity, semantic
  source-window plausibility, and target coverage.
- Adjusted the PF-1000 Malir density marker to match the cited-window language:
  the target line windows use `interferometer` / `interferometric` diagnostic
  language rather than the title-form word `interferometry`.
- Verification:
  - `python3 -m py_compile app_mhd.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims tests/test_kr_targets.py::test_kr_validation_target_semantic_audit_passes_for_coded_targets -q` passed (`2 passed in 1.19s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`90 passed in 0.50s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.68s`)
  - `git diff --check` passed.
- Remaining limit: this is a lightweight semantic guard. It does not replace
  human review of extracted values, same-shot completeness, digitized waveform
  data, or uncertainty-bearing simulation-to-experiment validation.

### 2026-05-05: KR Target Coverage Becomes A High-Fidelity Gap

- Added a `kr_target_coverage` area to `scientific_accuracy_gap_report()`.
- High-fidelity readiness now requires both a passing KR target coverage report
  and a passing KR semantic source-window audit.
- The default app result still reports this gap as `partial` because the coded
  target set has partial `circuit_waveform`, `phase_timing`, and
  `spatial_temperature` coverage.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 0.73s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`90 passed in 0.46s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.25s`)
  - `git diff --check` passed.
- Remaining limit: this closes a readiness-gate loophole. It does not yet
  supply the missing digitized waveform, same-shot phase timing, or same-device
  spatial temperature targets.

### 2026-05-05: Same-Scope KR Target Coverage Audit

- Added `kr_validation_same_scope_target_report()` to distinguish cross-device
  target availability from one compatible validation scope.
- App post-processing now exports `kr_validation_same_scope_targets`.
- High-fidelity readiness now requires target coverage, semantic source-window
  audit, and same-scope target coverage to pass.
- The best available scope is currently MJOLNIR neutron timing/detector response,
  but it is still missing circuit waveform, phase timing, spatial density,
  spatial magnetic/EM, spatial temperature, and uncertainty target groups.
- Verification:
  - `python3 -m py_compile app_mhd.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py src/dpf/validation/quality_assessment.py tests/test_kr_targets.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 3.03s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`91 passed in 0.49s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.32s`)
  - `git diff --check` passed.
- Remaining limit: this audit tightens the definition of an end-to-end target
  set. It does not create the missing same-scope experimental waveform, phase,
  spatial, or uncertainty targets.

### 2026-05-05: MJOLNIR Stagnation Temperature Target Context

- Added `mjolnir_stagnation_temperature_targets()` from the MJOLNIR KR paper.
- The target is tied to `mjolnir_neutron_timing_2025_goyon`, so same-scope
  audits now see MJOLNIR neutron timing/detector response plus partial
  spatial-temperature context.
- The target records the KR stagnation-temperature scaling reference, the
  `(Te + Ti) / 2` definition, the several-keV context, and explicit missing
  items for full tier 4: direct experimental temperature diagnostic,
  experimental uncertainty, and same-scope density/magnetic-field targets.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_mjolnir_stagnation_temperature_target_is_partial_context tests/test_kr_targets.py::test_kr_validation_target_semantic_audit_passes_for_coded_targets tests/test_kr_targets.py::test_kr_validation_target_coverage_report_lists_remaining_groups tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q` passed (`4 passed in 0.50s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py -q` passed (`92 passed in 0.58s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.46s`)
  - `git diff --check` passed.
- Remaining limit: this is temperature context from shock-theory/MHD-kinetic
  analysis, not a direct experimental temperature diagnostic. It cannot close
  tier 4 by itself.

### 2026-05-05: Corpus Review Status Saved And Audited

- Saved the explicit status that the complete `KnowledgeReference/` corpus has
  not yet been line-by-line review-closed.
- Added `kr_corpus_inventory()` and `kr_corpus_review_status()`.
- Current local inventory:
  - total files: 827
  - markdown files: 398
  - JSON files: 396
  - DPF-named markdown files: 54
- Current review-closed status under the coded-target rule at initial creation:
  - coded KR target records: 11
  - unique coded KR target source files: 7
  - DPF-named markdown files represented by coded targets: 6 of 54
  - unreviewed DPF-named markdown files: 48
- App post-processing now exports `kr_corpus_review_status`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py app_mhd.py tests/test_kr_corpus.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_kr_corpus.py tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q` passed (`3 passed in 1.14s`)
- Remaining limit: this is an inventory and progress audit. It does not itself
  extract the remaining data. The next extraction ratchet is to review the
  unreviewed DPF-named markdown files for waveform, phase, spatial, neutron,
  and uncertainty targets or mark them explicitly non-extractable.

### 2026-05-05: Unreviewed DPF Source Triage Queue

- Added `kr_unreviewed_dpf_source_triage()` to rank the 48 unreviewed DPF-named
  markdown files by observable keyword categories.
- Triage category counts among the 48 unreviewed files:
  - circuit waveform candidates: 30
  - phase timing candidates: 31
  - spatial density candidates: 17
  - spatial magnetic/EM candidates: 33
  - spatial temperature candidates: 42
  - neutron validation candidates: 42
  - uncertainty candidates: 18
- Current highest-priority sources by category breadth:
  - `KnowledgeReference/focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md`
  - `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
  - `KnowledgeReference/regular-article-deuterium-argon-admixture-for-plasma-focus-neutron-generation-muhammad-luqman.md`
  - `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
  - `KnowledgeReference/characterising-the-plasma-focus-pinch-and-speed-enhancing-the-neutron-yield.md`
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`3 passed in 0.53s`)
- Remaining limit: keyword triage is not extraction. Each candidate still needs
  line-by-line review, typed targets, source lines, units, and either an
  extracted validation target or an explicit non-extractable reason.

### 2026-05-05: PF-1000 Full-Energy Target Bundle From 2007 Papers

- Reviewed and extracted two high-priority PF-1000 full-energy papers from the
  local source of truth:
  - `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`
  - `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`
- Added `pf1000_full_energy_phase_context_targets()` for paper I. It records
  phase semantics, the 2-4 Torr / up to 850 kJ / 2.5-3 MA operating regime,
  maximum compression about 100 ns before the current dip, maximum compression
  about 2 us after current maximum, about 150 ns confinement/neutron-pulse
  timing, and the missing digitized endpoints needed for full tier 2.
- Added `pf1000_full_energy_neutron_spatial_targets()` for paper II. It records
  810 kJ operation, shot 3121 at 465 Pa and 35 kV, typical total current
  2.5-2.6 MA, best current near 3 MA, estimated average pinch current about
  2 MA, neutron anisotropy ratios, 5e10-2e11 n/shot yield range with 6e11
  maximum, 7 m TOF correction, 2.45 MeV first-pulse context, density and
  magnetic-field estimates, temperature estimates, and detector/temperature
  limitations.
- The two targets share validation scope
  `pf1000_full_energy_2007_gribkov_scholz`.
- Current corpus status after this ratchet:
  - coded KR target records: 13 at this point in the sequence
  - unique coded KR target source files: 9
  - DPF-named markdown files represented by coded targets: 8 of 54
  - unreviewed DPF-named markdown files: 46
- Current target status remains intentionally blocked:
  - target coverage does not pass: `circuit_waveform`, `phase_timing`, and
    `spatial_temperature` remain partial.
  - same-scope coverage does not pass.
  - PF-1000 full-energy scope now has every required group present, including
    neutron detector response, but current waveform, phase timing, neutron
    timing/spectrum/anisotropy/detector response, spatial magnetic/EM,
    spatial temperature, and uncertainty are still partial.
- Updated triage counts after removing the two newly coded sources:
  - circuit waveform candidates: 28
  - phase timing candidates: 29
  - spatial density candidates: 15
  - spatial magnetic/EM candidates: 31
  - spatial temperature candidates: 40
  - neutron validation candidates: 40
  - uncertainty candidates: 16
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`43 passed in 0.45s`)
- Remaining limit: this is a major target-authority improvement, not a
  validation closure. The PF-1000 full-energy source itself says ion
  temperatures were not directly measured, pinch current was not directly
  measured, neutron pulse tails are affected by scatter, and current/neutron
  traces still need digitization and uncertainty for quantitative validation.

### 2026-05-05: PF-1000 Same-Scope Detector-Response Context

- Extended the PF-1000 full-energy paper II target with activation-counter,
  indium/bubble-detector cross-check, AmBe calibration, scintillator-PM,
  time-of-flight, and room-scatter response requirements.
- The PF-1000 full-energy scope now has every required end-to-end target group
  present in one validation scope.
- The same scope still fails by design because detector response is partial:
  the KR lines identify calibration and TOF context plus scatter limitations,
  but do not provide a complete neutron-field transport, detector-response, or
  room-scatter model.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`43 passed in 0.50s`)
- Remaining limit: group presence is not validation closure. PF-1000 still
  needs digitized current and neutron traces, quantitative detector response,
  direct/uncertainty-bearing plasma state diagnostics, and simulation outputs
  compared against those targets.

### 2026-05-05: Deuterium-Argon Admixture Neutron Target

- Reviewed
  `KnowledgeReference/regular-article-deuterium-argon-admixture-for-plasma-focus-neutron-generation-muhammad-luqman.md`.
- Added `deuterium_argon_admixture_neutron_targets()` for the 2.7 kJ
  Mather-type PF gas-mixture experiment.
- The target records 30 uF / 14 kV / 4 mbar operation, 10-70% argon mass
  mixtures, measured current/voltage waveform availability, Rogowski and
  voltage-probe calibration, Lee-model current fitting, focus-time shift from
  2.7 to 3.3 us, voltage-spike FWHM values, indium activation calibration,
  pure-D2 and 50% argon neutron yields with standard deviations, energy into
  pinch, computed pinch current, computed ion-temperature context, and
  shot-to-shot uncertainty.
- Current corpus status after this ratchet:
  - coded KR target records: 14
  - unique coded KR target source files: 10
  - DPF-named markdown files represented by coded targets: 9 of 54
  - unreviewed DPF-named markdown files: 45
- Updated triage counts after removing this source:
  - circuit waveform candidates: 27
  - phase timing candidates: 28
  - spatial density candidates: 14
  - spatial magnetic/EM candidates: 30
  - spatial temperature candidates: 39
  - neutron validation candidates: 39
  - uncertainty candidates: 15
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`44 passed in 0.85s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`98 passed in 0.67s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q` passed (`87 passed, 3 skipped in 1.84s`)
  - `git diff --check` passed.
- Remaining limit: this source is useful for admixture-yield and activation
  validation, but not for full end-to-end closure. It is time-integrated for
  neutron yield, uses Lee-fitted computed temperature, lacks digitized
  waveform points, and does not provide spatial density/magnetic/temperature
  fields for simulation comparison.

### 2026-05-05: FF-1 Focus Fusion Plasmoid And p-B11 Context Target

- Reviewed
  `KnowledgeReference/focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the-dense-plasma-focus.md`.
- Added `ff1_focus_fusion_plasmoid_targets()` for FF-1 / FF-2B plasmoid,
  neutron, ion-energy, density, impurity, and p-B11 context.
- The target records FF-1 device parameters, diagnostic suite, main and beam
  Rogowski context, ion-beam energy-transfer measurements, confined-ion
  energy by neutron TOF, isotropy support from bubble detectors, best 2016
  neutron yield, wall-plug efficiency, estimated density, n-tau-T product,
  beryllium impurity/deposition measurements, QMF/p-B11 magnetic-field
  constraints, and current oscillation/yield-plateau limitations.
- Current corpus status after this ratchet:
  - coded KR target records: 15
  - unique coded KR target source files: 11
  - DPF-named markdown files represented by coded targets: 10 of 54
  - unreviewed DPF-named markdown files: 44
- Updated triage counts after removing this source:
  - circuit waveform candidates: 26
  - phase timing candidates: 27
  - spatial density candidates: 13
  - spatial magnetic/EM candidates: 29
  - spatial temperature candidates: 38
  - neutron validation candidates: 38
  - uncertainty candidates: 14
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`45 passed in 0.80s`)
- Remaining limit: this target is explicitly not p-B11 net-energy validation.
  The source includes measured deuterium FF-1 values, but the p-B11 and QMF
  parts are constraints, projections, or reduced simulations. Full validation
  still needs digitized waveforms, detector response, shot-series uncertainty,
  and direct advanced-fuel measurements.

### 2026-05-05: Lee Drive-Parameter Speed-Enhancement Target

- Reviewed
  `KnowledgeReference/characterising-the-plasma-focus-pinch-and-speed-enhancing-the-neutron-yield.md`.
- Added `lee_drive_parameter_speed_enhancement_targets()` as a generic
  scaling/regime target, not a same-device validation packet.
- The target records Lee axial snowplow/radial slug phase semantics,
  deuterium and neon pinch radius/length/lifetime scaling with anode radius,
  the neutron-optimized drive parameter `Ip/a/sqrt(p_D2) = 89.0 +/- 7.7`
  kA/cm/sqrt(torr), typical axial and radial speeds, constant-speed `Y ~ I^4`
  scaling, speed-enhanced thermonuclear and beam-target scaling, and
  operational speed limits where focus quality deteriorates.
- Current corpus status after this ratchet:
  - coded KR target records: 16
  - unique coded KR target source files: 12
  - DPF-named markdown files represented by coded targets: 11 of 54
  - unreviewed DPF-named markdown files: 43
- Updated triage counts after removing this source:
  - circuit waveform candidates: 25
  - phase timing candidates: 26
  - spatial density candidates: 13
  - spatial magnetic/EM candidates: 28
  - spatial temperature candidates: 37
  - neutron validation candidates: 37
  - uncertainty candidates: 13
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`46 passed in 0.49s`)
- Remaining limit: this source supports regime checks and scaling law hygiene,
  but it is generic. It cannot close any same-shot validation gate without a
  device-specific waveform, pressure, geometry, phase timing, neutron history,
  and detector-response packet.

### 2026-05-05: PFZ-200 Hybrid X-Pinch Proton/Neutron Target

- Reviewed
  `KnowledgeReference/deuterium-hybrid-x-pinch-driven-by-small-dense-plasma-focus-2.md`.
- Added `pfz200_hybrid_xpinch_proton_neutron_targets()` for the 3 kJ PFZ-200
  DPF-driven deuterium hybrid X-pinch.
- The target records PFZ-200 current/geometry/gas context, Rogowski current
  diagnostics, silver activation and nTOF detector setup, schlieren and CR-39
  diagnostic details, neutron FWHM timing for 3 mm and 5 mm A-K gaps versus
  unmodified DPF operation, neutron-yield ranges, localized proton-source
  dimensions, proton spectrum/yield values, and anisotropy/shot-to-shot
  interpretation limits.
- Current corpus status after this ratchet:
  - coded KR target records: 17
  - unique coded KR target source files: 13
  - DPF-named markdown files represented by coded targets: 12 of 54
  - unreviewed DPF-named markdown files: 42
- Updated triage counts after removing this source:
  - circuit waveform candidates: 24
  - phase timing candidates: 26
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 27
  - spatial temperature candidates: 36
  - neutron validation candidates: 36
  - uncertainty candidates: 12
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`47 passed in 0.48s`)
- Remaining limit: this is a modified hybrid X-pinch load. It is useful for
  localized DD particle-source and detector-response validation, but it is not
  an ordinary DPF end-to-end target and does not provide density, magnetic, or
  temperature validation for a standard DPF pinch.

### 2026-05-05: LLNL Fully Kinetic DPF Benchmark And Duplicate Review Decisions

- Reviewed the three local copies of the Schmidt/Tang/Welch fully kinetic DPF
  paper:
  `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch-8.md`,
  `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch-9.md`,
  and
  `KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md`.
- Added explicit corpus review decisions marking the base and `-9` files as
  duplicate local copies represented by the canonical `-8` coded target.
- Added `llnl_fully_kinetic_dpf_targets()` for the LLNL 180 kA, 1 torr
  deuterium fully kinetic benchmark.
- The target records the LSP implicit-PIC setup, 2D cylindrical geometry,
  322-by-151 grid, 5 cm anode, 1.5 cm cathode radius, 10 cm domain length,
  1 mm initial sheath, neutral and sheath density, 4 kV initial voltage drop,
  180 kA current, current dip/impedance context, lower-hybrid-frequency
  fluctuation context, hot-pinch temperatures, MeV-ion spectrum context, and
  fluid/hybrid/fully kinetic neutron-yield comparison.
- Current corpus status after this ratchet:
  - coded KR target records: 18
  - unique coded KR target source files: 14
  - DPF-named markdown files represented by coded targets: 13 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 15 of 54
  - unreviewed DPF-named markdown files: 39
- Updated triage counts after removing the three fully kinetic local copies:
  - circuit waveform candidates: 21
  - phase timing candidates: 23
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 24
  - spatial temperature candidates: 33
  - neutron validation candidates: 33
  - uncertainty candidates: 9
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`52 passed in 0.91s`)
- Remaining limit: this source is a simulation benchmark showing that fully
  kinetic physics is needed for MeV ions and approximate neutron yield in the
  LLNL low-current DPF. It is not a direct experimental data packet, does not
  provide detector response or shot-ensemble uncertainty, and cannot close
  same-scope predictive readiness by itself.

### 2026-05-05: NSTec/Gemini Fully 3D MHD Rundown Benchmark

- Reviewed
  `KnowledgeReference/fully-three-dimensional-simulation-and-modeling-of-a-dense-plasma-focus.md`.
- Added `nstec_3d_mhd_rundown_targets()` as a partial 3D-MHD current/rundown
  benchmark for the NSTec/Gemini DPF.
- The target records device geometry, bank/circuit context, Faraday rotator
  current diagnostic setup, 37-shot waveform repeatability at 37.5 kV and
  7.28 Torr, 2D/3D ALEGRA current comparisons, rundown-time comparisons, 3D
  cathode-bar flow/inductance context, density-floor and artificial hot-start
  limits, and the source's explicit statement that MHD becomes unphysical near
  Z-pinch without kinetic/PIC closure.
- Current corpus status after this ratchet:
  - coded KR target records: 19
  - unique coded KR target source files: 15
  - DPF-named markdown files represented by coded targets: 14 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 16 of 54
  - unreviewed DPF-named markdown files: 38
- Updated triage counts after removing this source:
  - circuit waveform candidates: 20
  - phase timing candidates: 22
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 23
  - spatial temperature candidates: 32
  - neutron validation candidates: 32
  - uncertainty candidates: 8
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`53 passed in 0.56s`)
- Remaining limit: this target supports current/rundown benchmarking and 3D MHD
  scope control, not neutron-yield validation. It lacks digitized Faraday trace
  points, per-shot uncertainty, direct density/temperature/field diagnostics,
  detector response, and any validated late-pinch kinetic closure.

### 2026-05-05: MJOLNIR High/Low-Yield Parasitic-Current Target

- Reviewed `KnowledgeReference/goyon-2022-mjolnir-high-low.md`.
- Added `mjolnir_high_low_parasitic_current_targets()` as a partial
  MJOLNIR same-device mechanism target for variable yield, parasitic current
  paths, current dips, voltage spikes, conditioning, run-down/run-in velocity,
  pressure effects, and PIC/snowplow interpretation.
- The target records the 1-MJ and 2-MJ MJOLNIR pulsed-power configurations,
  highest reported neutron yields, Rogowski/voltage/photodiode/framing-camera
  diagnostics, CHICAGO/BERTHA/PIC setup, snow-plow alternate-current-path model,
  sheath phase sequence, current-dip and voltage-yield correlations, rBtheta
  parasitic-path interpretation, beam-energy mechanism, pressure degradation,
  and the remaining detector/trace/uncertainty gaps.
- Current corpus status after this ratchet:
  - coded KR target records: 20
  - unique coded KR target source files: 16
  - DPF-named markdown files represented by coded targets: 15 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 17 of 54
  - unreviewed DPF-named markdown files: 37
- Updated triage counts after removing this source:
  - circuit waveform candidates: 19
  - phase timing candidates: 21
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 22
  - spatial temperature candidates: 31
  - neutron validation candidates: 31
  - uncertainty candidates: 7
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`54 passed in 0.53s`)
- Remaining limit: this source adds strong MJOLNIR mechanism constraints, but
  it still lacks digitized traces, shot-resolved uncertainty, activation
  detector response details, neutron timing/spectrum/anisotropy, and direct
  spatial density/temperature/field validation.

### 2026-05-05: PF-400J X-Ray Diagnostic Inference Target

- Reviewed
  `KnowledgeReference/inference-of-x-ray-emission-from-a-plasma-focus-discharge-comparison-between-characteristic.md`.
- Added `pf400j_xray_inference_targets()` as a PF-400J hydrogen x-ray
  diagnostic inference target, explicitly not a neutron validation packet.
- The target records PF-400J bank, geometry, fill, and discharge conditions;
  Rogowski, ILS, voltage-divider, Vivaldi, and scintillator-PMT diagnostics;
  x-ray detector response context; 959-shot campaign size; breakdown/pinch
  feature definitions; machine-learning feature-selection results; and the
  limits of using electrical/EM signals to infer x-ray emission.
- Current corpus status after this ratchet:
  - coded KR target records: 21
  - unique coded KR target source files: 17
  - DPF-named markdown files represented by coded targets: 16 of 54
  - DPF-named markdown files closed by explicit duplicate decisions: 2 of 54
  - total DPF-named markdown files review-closed: 18 of 54
  - unreviewed DPF-named markdown files: 36
- Updated triage counts after removing this source:
  - circuit waveform candidates: 18
  - phase timing candidates: 20
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 21
  - spatial temperature candidates: 30
  - neutron validation candidates: 30
  - uncertainty candidates: 6
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q` passed (`55 passed in 0.55s`)
- Remaining limit: this target supports x-ray diagnostic and feature-selection
  hygiene for a hundreds-of-joules hydrogen PF device. It does not provide
  deuterium neutron validation, same-scope spatial validation, absolute x-ray
  spectrum/response, or production solver closure.

### 2026-05-05: Reuben 2024 Thesis Review Decision

- Reviewed
  `KnowledgeReference/modification-and-numerical-modelling-of-dense-plasma-focus.md`.
- Added an explicit `insufficient_extractable_validation_data` corpus review
  decision instead of a coded target.
- Reason: the local markdown contains useful abstract, introduction, table, and
  figure-caption context for a 1 kJ / 1.3 uF / 40 kV modified DPF thesis, but
  the Experimental System, Numerical Modelling, Results and Discussion, and
  Conclusion sections are empty page stubs in this text extraction. Result
  values such as current waveform, radial trajectories, neutron production,
  pinch temperature, and scaling appear only as figure-list captions rather
  than source-line data suitable for validation targets.
- Current corpus status after this decision:
  - coded KR target records: 21
  - unique coded KR target source files: 17
  - DPF-named markdown files represented by coded targets: 16 of 54
  - DPF-named markdown files closed by explicit review decisions: 3 of 54
  - total DPF-named markdown files review-closed: 19 of 54
  - unreviewed DPF-named markdown files: 35
- Updated triage counts after removing this source:
  - circuit waveform candidates: 17
  - phase timing candidates: 19
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 20
  - spatial temperature candidates: 29
  - neutron validation candidates: 29
  - uncertainty candidates: 5
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py src/dpf/validation/__init__.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.59s`)
- Remaining limit: this file should be re-ingested from the original PDF if the
  thesis is needed for validation. The current markdown is not reliable enough
  for line-referenced current waveform, radial trajectory, or neutron-yield
  targets.

### 2026-05-05: Goyon 2025 Neutron-Generation Duplicate Decision

- Reviewed
  `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md`.
- Added an explicit duplicate review decision pointing to the canonical
  `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`.
- Reason: the canonical `-5` source already backs the coded MJOLNIR neutron
  timing, stagnation-temperature, and neutron detector-response targets. This
  local copy is the same Phys. Plasmas 2025 Goyon MA-class MJOLNIR
  neutron-generation paper and should not produce duplicate target records.
- Current corpus status after this decision:
  - coded KR target records: 21
  - unique coded KR target source files: 17
  - DPF-named markdown files represented by coded targets: 16 of 54
  - DPF-named markdown files closed by explicit review decisions: 4 of 54
  - total DPF-named markdown files review-closed: 20 of 54
  - unreviewed DPF-named markdown files: 34
- Updated triage counts after removing this source:
  - circuit waveform candidates: 16
  - phase timing candidates: 18
  - spatial density candidates: 12
  - spatial magnetic/EM candidates: 19
  - spatial temperature candidates: 28
  - neutron validation candidates: 28
  - uncertainty candidates: 4
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.55s`)
- Remaining limit: duplicate closure avoids double-counting; it does not add
  new validation coverage beyond the existing MJOLNIR coded targets.

### 2026-05-05: Rawat 2015 Generic DPF Operating-Envelope Target

- Reviewed the duplicate Rawat 2015 review pair:
  `KnowledgeReference/paper-open-access-dense-plasma-focus-from-alternative-fusion-source-to-versatile-high-energy-4.md`
  and
  `KnowledgeReference/paper-open-access-dense-plasma-focus-from-alternative-fusion-source-to-versatile-high-energy.md`.
- Added `rawat_dpf_operating_envelope_targets()` from the canonical `-4`
  source and added an explicit duplicate decision for the header/PDF-name
  variant without the `-4` suffix.
- Encoded the source as a generic DPF operating-envelope target, not as a
  same-device benchmark. Extracted constraints include 100-500 ns current
  sheath formation, 500-3000 ns quarter period, 2-10 cm/us optimized axial
  sheath speed, radial speed 2-2.5 times axial speed, pinch density
  `5e24-1e26 m^-3`, DPF energy density `1.2e10-9.5e10 J/m^3`, pinch
  temperatures `0.2-2 keV`, ion temperatures `0.3-1.5 keV`, 10-30 kV typical
  charge voltage, efficient operation at a few mbar, and explicit shot-to-shot
  repeatability/conditioning limits.
- Current corpus status after this ratchet:
  - coded KR target records: 22
  - unique coded KR target source files: 18
  - DPF-named markdown files represented by coded targets: 17 of 54
  - DPF-named markdown files closed by explicit review decisions: 5 of 54
  - total DPF-named markdown files review-closed: 22 of 54
  - unreviewed DPF-named markdown files: 32
- Updated triage counts after removing this pair:
  - circuit waveform candidates: 16
  - phase timing candidates: 16
  - spatial density candidates: 10
  - spatial magnetic/EM candidates: 17
  - spatial temperature candidates: 26
  - neutron validation candidates: 26
  - uncertainty candidates: 2
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`56 passed in 0.63s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`107 passed in 0.63s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.54s`); `git diff --check` clean.
- Remaining limit: this is a review-derived sanity envelope. It can catch
  simulations outside basic DPF scale, but it cannot close predictive
  validation without same-scope measured current, phase, spatial, neutron, and
  uncertainty data.

### 2026-05-05: Petrov/LLNL 2022 MJOLNIR Duplicate Decision

- Reviewed `KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md`.
- Added an explicit duplicate review decision pointing to
  `KnowledgeReference/goyon-2022-mjolnir-high-low.md`.
- Reason: the Petrov/LLNL report extraction is the same Schmidt/Goyon 2022
  MJOLNIR high/low-performing discharge paper already represented by the coded
  `mjolnir_high_low_parasitic_current_2022_goyon` target. Differences are
  header, page-stamp, and line-wrap extraction differences, not separate
  validation evidence.
- Current corpus status after this decision:
  - coded KR target records: 22
  - unique coded KR target source files: 18
  - DPF-named markdown files represented by coded targets: 17 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 23 of 54
  - unreviewed DPF-named markdown files: 31
- Updated triage counts after removing this source:
  - circuit waveform candidates: 15
  - phase timing candidates: 15
  - spatial density candidates: 10
  - spatial magnetic/EM candidates: 16
  - spatial temperature candidates: 25
  - neutron validation candidates: 25
  - uncertainty candidates: 1
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 0.51s`)
  - Broad post-decision verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`107 passed in 0.59s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.49s`); `git diff --check` clean.
- Remaining limit: duplicate closure avoids double-counting the same MJOLNIR
  parasitic-current evidence. It does not add new same-scope coverage.

### 2026-05-05: Auluck 2023 Generalized Plasma Focus Scaling Target

- Reviewed
  `KnowledgeReference/the-generalized-plasma-focus-problem-and-its-application-to-space-propulsion-s-k-h-auluck.md`.
- Added `auluck_gpf_scaling_theory_targets()` as a KR-backed theory/scaling
  target, not as an experimental validation pass.
- Encoded the paper's key scientific warning: conventional DPF fusion energy
  output involves a complex, not fully understood process; neutron-yield scaling
  failure is experimentally observed; no theoretical understanding or empirical
  workaround exists for conventional DPF propulsion claims in this source.
- Encoded model-scope constraints: Lee and RGV-type reduced models require
  experimental current waveform fitting and compensate for neglected formation,
  propagation-delay, and sheath-geometry physics. The GPF treatment explicitly
  addresses lift-off/propagation delay and treats the moving sheath as a
  power-density-amplifying plasma flow switch.
- Extracted laboratory example values include 20 kV charge voltage, `43 uF`
  capacitance, `160 kA` current scale, `8.6 kJ` stored energy, `8.45 us`
  quarter period, hydrogen density `0.00342 kg/m^3` or about `43 mbar`,
  example power-density amplification about `9000`, magnetic field rising from
  `20 T` to about `200 T` in about `40 ns`, wire current about `80 kA`, current
  density `1.8e12 A/m^2`, radial Alfven transit time about `17 ns`, wire travel
  time about `8.4 ns`, explosion timescale about `3 ps`, jet Alfven velocity
  about `1450 m/s`, and impulse about `0.002 kg m/s`.
- Encoded validation requirements from the source: measure plasma voltage and
  current, compute and compare inductance variation, repeat across profile
  parameters, measure jet momentum and velocity, verify energy deposition in
  dynamic-hohlraum variants, validate gas-distribution/breakdown strategy, and
  separately test deuterium-filled tube neutron emission.
- Current corpus status after this ratchet:
  - coded KR target records: 23
  - unique coded KR target source files: 19
  - DPF-named markdown files represented by coded targets: 18 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 24 of 54
  - unreviewed DPF-named markdown files: 30
- Updated triage counts after removing this source:
  - circuit waveform candidates: 14
  - phase timing candidates: 14
  - spatial density candidates: 9
  - spatial magnetic/EM candidates: 15
  - spatial temperature candidates: 24
  - neutron validation candidates: 24
  - uncertainty candidates: 1
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`57 passed in 0.60s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`108 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.54s`); `git diff --check` clean.
- Remaining limit: this source is important for scope control and scaling
  requirements, but it is not a same-shot DPF benchmark. It does not provide
  measured waveforms, phase endpoints, spatial profiles, neutron data, or
  uncertainty for a completed predictive validation case.

### 2026-05-05: Sandia 2009 ALEGRA-HEDP DPF MHD Target

- Reviewed
  `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md`.
- Added `alegra_hedp_dpf_mhd_validation_targets()` as a partial MHD/circuit
  benchmark and scope-limit target.
- Encoded the source's central scientific limit: 2D ALEGRA-HEDP can reproduce
  early DPF current, timing, sheath speed, density, and temperature behavior for
  Bernard-class devices, but MHD only predicts the thermonuclear neutron
  component and must stop when charge separation and instabilities make the MHD
  approximation invalid.
- Extracted benchmark values include Bernard Long `135 uF`, `20 kV`, `27 kJ`,
  `27 nH` estimated stray inductance, `3.3 mOhm` estimated resistance, `3 Torr`,
  experiment/simulation peak current `0.6 MA`/`0.5-0.6 MA`, and neutron yield
  `1.5e9` experiment vs `1.2e5` ALEGRA thermonuclear. Bernard Short includes
  `120 uF`, `40 kV`, `96 kJ`, `10 Torr`, peak current `1.5 MA` experiment and
  ALEGRA, and neutron yield `3e10` experiment vs `1.5e6` ALEGRA. Tallboy
  includes `216 uF`, `50 kV`, `270 kJ`, `50 nH`, peak current `2.3 MA`
  experiment vs `1.8 MA` ALEGRA, and neutron yield `3.5e11` experiment vs
  `3.7e7` ALEGRA.
- Encoded spatial/temperature context: generic pinch width about `1 mm`, length
  of a few mm, density `1e19-1e20 cm^-3`, Bernard Long measured pinch density
  `1e18-5e19 cm^-3`, simulated density `1.4e19 cm^-3`, experimental pre-pinch
  ion temperature `300 eV`, simulated pre-pinch ion/electron temperatures
  `250-650 eV` and `200-360 eV`, experimental pinch ion temperature about
  `700 eV`, and unresolved simulated pinch ion temperature `9 keV`.
- Encoded numerical limits: Sesame EOS density floor `0.01 kg/m^3` is
  inconsistent for initial DPF gas, QEOS deuterium was used, approximate cell
  size was `0.5 mm`, the ionized seed layer was arbitrary at `1 eV`, cathode
  bars require 3D modeling, and PIC-to-MHD sheath import is needed.
- Current corpus status after this ratchet:
  - coded KR target records: 24
  - unique coded KR target source files: 20
  - DPF-named markdown files represented by coded targets: 19 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 25 of 54
  - unreviewed DPF-named markdown files: 29
- Updated triage counts after removing this source:
  - circuit waveform candidates: 13
  - phase timing candidates: 13
  - spatial density candidates: 8
  - spatial magnetic/EM candidates: 14
  - spatial temperature candidates: 23
  - neutron validation candidates: 23
  - uncertainty candidates: 1
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`58 passed in 0.62s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`109 passed in 0.60s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.51s`); `git diff --check` clean.
- Remaining limit: this target supports early MHD/circuit validation only. It
  does not validate total neutron yield, beam-target production, neutron timing,
  neutron spectrum, neutron anisotropy, detector response, or post-MHD kinetic
  pinch evolution.

### 2026-05-05: Auluck 2021 Circuit-Element/Poynting Target

- Reviewed `KnowledgeReference/auluck-2021-dpf-circuit-element.md`.
- Added `auluck_circuit_element_poynting_targets()` as a circuit-field
  coupling target.
- Encoded the paper's core constraint: representing DPF post-stagnation behavior
  as a scalar time-varying inductance is incomplete. The terminal voltage must
  account for the volume-integrated field power, and the difference between the
  Poynting-theorem term and the motional impedance implied by a time-varying
  inductance appears as anomalous impedance.
- Extracted diagnostic context includes standard `dI/dt` and voltage
  diagnostics, current derivative dip and voltage spike as proper-operation
  indicators, their correlation with neutron yield, and the note that voltage
  spike and current derivative minimum are time-correlated but not simultaneous.
- Extracted PF-1000 context includes magnetic probe radii `40`, `13`, and
  `0 mm`, probe height `10 mm` above the anode, interferogram intervals
  `10-15 ns`, current-carrying layer thickness `1.6-2.6 cm`, sheath velocity
  about `2.1e5 m/s` with `25%` shot-to-shot variation, density fall by at least
  two orders of magnitude within less than `1 mm`, illustrative probe times
  `-68`, `-38`, and `22 ns`, and a `10-20 ns` diagnostic propagation delay over
  about `2 m`.
- Encoded field-coupling requirements: 3D magnetic and velocity structures,
  motional dynamo amplification of seed fields, poloidal magnetic fields, all
  three magnetic-field components contributing to plasma inductance, and
  quasi-closed post-breakup current streamlines that still draw energy from the
  external circuit.
- Current corpus status after this ratchet:
  - coded KR target records: 25
  - unique coded KR target source files: 21
  - DPF-named markdown files represented by coded targets: 20 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 26 of 54
  - unreviewed DPF-named markdown files: 28
- Updated triage counts after removing this source:
  - circuit waveform candidates: 13
  - phase timing candidates: 12
  - spatial density candidates: 7
  - spatial magnetic/EM candidates: 13
  - spatial temperature candidates: 23
  - neutron validation candidates: 22
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`59 passed in 0.56s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`110 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.53s`); `git diff --check` clean.
- Remaining limit: this target gives theory and diagnostic interpretation, not
  a complete same-shot validation dataset. It requires digitized `dI/dt` and
  voltage traces, 3D field/velocity measurements, volume-integrated `J dot E`,
  and neutron-response coupling before it can close field-circuit predictive
  validation.

### 2026-05-05: Esaulov 2003 2D MHRDR DPF Target

- Reviewed `KnowledgeReference/esaulov_2003_2d_mhd_dpf.md`.
- Added `esaulov_2d_mhrdr_dpf_targets()` as a partial 2D multi-temperature MHD
  and thermal neutron-rate context target for the LANL Begay DPF.
- Encoded device parameters: Mather-type Begay device, inner electrode radius
  `1.18 cm`, outer electrode radius `3.65 cm`, inner electrode length `15.7 cm`,
  deuterium fill `1 Torr`, capacitance `36.4 uF`, charging voltage `14 kV`, and
  series inductance `178 nH`.
- Encoded model physics: MHRDR uses multi-temperature ion/electron/radiation
  MHD, electron and ion thermal conduction, resistive diffusion, radiation
  diffusion, Lorentz force, shock hydrodynamics, self-consistent external
  circuit coupling, and Maxwell-averaged D-D cross sections for neutron-rate
  computation.
- Extracted phase/context values: current-sheath formation examples around
  `0.9` and `2.0 us`, acceleration slices at `1.0` and `2.0 us`, collapse
  pressure contours around `2.6` and `2.65 us`, local neutron-rate peaks at
  `2.74` and `2.92 us`, radial slices at `2.72` and `2.90 us`, focus duration
  `100-150 ns`, current during acceleration about `50-100 kA`, electrode
  voltage drop about `1-2 kV`, abstract density above `1e19 cm^-3`, and
  axis-history temperature scale to `5 keV`.
- Encoded scope limits: the target assumes a high-pressure thermal-MHD regime,
  treats beam-target mechanisms as outside the primary target, uses figure-scale
  quantities as context only, and lacks digitized traces, error bars, detector
  response, and same-shot experimental profiles.
- Current corpus status after this ratchet:
  - coded KR target records: 26
  - unique coded KR target source files: 22
  - DPF-named markdown files represented by coded targets: 21 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 27 of 54
  - unreviewed DPF-named markdown files: 27
- Updated triage counts after removing this source:
  - circuit waveform candidates: 12
  - phase timing candidates: 12
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 12
  - spatial temperature candidates: 22
  - neutron validation candidates: 21
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`60 passed in 0.58s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`111 passed in 0.57s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.43s`); `git diff --check` clean.
- Remaining limit: this source strengthens the thermal-MHD branch of the
  validation plan but cannot validate end-to-end DPF neutron prediction. It
  lacks absolute neutron yield, neutron timing, spectra, anisotropy, detector
  response, kinetic beam-target physics, and uncertainty.

### 2026-05-05: FAETON-I 2025 High-Voltage DPF Target

- Reviewed
  `KnowledgeReference/faeton-i-investigation-of-plasma-dynamics-and-radiation-output-of-a-100-kv-plasma-focus-device.md`.
- Added `faeton_i_high_voltage_dpf_targets()` as a partial high-voltage DPF
  validation target. The local markdown extraction contains the references,
  conclusion, and Table 3 region, not the full paper body, so the target is
  intentionally marked partial.
- Encoded Table 3 shot values: shot `1062` with `fcr=0.4`, `fcr2=0.35`,
  `Vp=37.3 kV`, code yield `2.77e9`, measured yield `3e9`; shot `1036`
  with `fcr=0.72`, `Vp=101.4 kV`, code yield `2.54e10`, measured yield
  `2.21e10`; shot `1027` with `fcr=0.8`, `Vp=160.5 kV`, code yield
  `5.5e10`, measured yield `5.44e10`; and shot `895` with `fcr=0.9`,
  `Vp=194 kV`, code yield `4.1e10`, measured yield `6e10`.
- Encoded interpretation limits from the source: `fcr=0.7` marks good current
  sheath formation, exceptional shots use `fcr=0.8-0.9`, peak inductive
  voltage `Vmax` is a better high-voltage PF indicator than current-dip
  severity when restrikes truncate the dip, and the voltage spike is reported
  pre-stagnation and dynamics-induced.
- Encoded neutron/radiation diagnostics: consistent D-D yield `2.5e10` over
  five shots without gas refill, exceptional D-D yield up to `8e10`, forward
  anisotropy factor `1.6`, neutron energy peak `2.5 MeV` with `0.3 MeV`
  uncertainty, PMT scintillators at `5`, `10`, `20`, and `40 m`, `40 m` nTOF,
  `30 cm` lead shielding for gamma measurements above `3 MeV`, and Faraday-cup
  deuteron energy about `350 keV`.
- Recorded D-T Faeton-X values only under `projections_not_validation_targets`:
  `2e14` neutrons for `65 kV`, `1 MJ`, `4 MA`, and `2e15` neutrons for
  `150 kV`, `5 MJ`, `7 MA`. These are not treated as validated FAETON-I D-D
  evidence.
- Current corpus status after this ratchet:
  - coded KR target records: 27
  - unique coded KR target source files: 23
  - DPF-named markdown files represented by coded targets: 22 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 28 of 54
  - unreviewed DPF-named markdown files: 26
- Updated triage counts after removing this source:
  - circuit waveform candidates: 11
  - phase timing candidates: 11
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 11
  - spatial temperature candidates: 21
  - neutron validation candidates: 20
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`61 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`112 passed in 0.62s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.51s`); `git diff --check` clean.
- Remaining limit: this source strengthens high-voltage DPF waveform/yield and
  detector-response requirements, but it does not close predictive validation.
  The local extract lacks digitized current/voltage traces, absolute phase
  times, spatial density/temperature/magnetic-field profiles, full detector
  response and calibration uncertainty, full neutron histories/spectra, and
  the complete shot dataset.

### 2026-05-05: Lee/RADPF Theoretical Model-Scope Target

- Reviewed `KnowledgeReference/lee_radpf_theory.md`.
- Added `lee_radpf_theory_model_scope_targets()` as a reduced-model scope
  target, not as experimental validation evidence.
- Encoded model structure: external circuit and sheath motion are coupled; the
  equation of motion is affected by current and the circuit equation is
  affected by sheath motion/position; plasma resistance is ignored for the
  electromagnetic-drive approximation; and axial/radial tube voltage is treated
  as inductive in the reduced model.
- Encoded phase assumptions: axial phase uses a snowplow current sheath for
  trajectory, speed, and current-profile fitting; radial phase replaces the
  singular thin-snowplow limit with a slug model where the magnetic piston
  follows a shock front; reflected shock begins when the radial shock reaches
  the axis; and pinch breakup is modeled as an expanded uniform current column.
- Extracted timing/scale constraints: `alpha` is electrical time over axial
  transit time, `alpha1` is axial transit over radial transit time, axial
  transit time is characteristically about `20` times radial shock transit time,
  the typical axial/radial characteristic time ratio is about `40`, reflected
  shock speed is `0.3` of the on-axis inward radial shock speed, and the
  communication delay expression is `(rp - rs) / SDS`.
- Encoded radiation/temperature constraints: shocked-plasma temperature is
  computed from shock speed, slow-compression temperature from energy balance,
  Spitzer resistivity is used, bremsstrahlung/recombination/line losses are
  represented, self-absorption drives volumetric-to-surface emission transition,
  deuterium radiation collapse critical current is `1.6 MA`, and neon/argon
  line radiation can reduce the critical current below `100 kA`.
- Encoded neutron-model limits: thermonuclear yield uses density, volume,
  thermal `sigma v`, and time; beam-target yield is phenomenological; beam
  deuterons are produced by diode action near the anode; beam voltage is tied
  to `Vmax`; the code uses beam energy `3 * Vmax` for the cross section; the
  source reports code `Vmax` of order `20-50 kV`, experimental beam-energy
  relevance `50-150 keV`, and lower-voltage machine range `30-60 keV`; the
  empirical fit is `Yn = 9e10 * Ipinch^3.8` for `0.1-1 MA`; and the calibration
  point is `0.5 MA`, `7e9` neutrons.
- Current corpus status after this ratchet:
  - coded KR target records: 28
  - unique coded KR target source files: 24
  - DPF-named markdown files represented by coded targets: 23 of 54
  - DPF-named markdown files closed by explicit review decisions: 6 of 54
  - total DPF-named markdown files review-closed: 29 of 54
  - unreviewed DPF-named markdown files: 25
- Updated triage counts after removing this source:
  - circuit waveform candidates: 10
  - phase timing candidates: 10
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 10
  - spatial temperature candidates: 20
  - neutron validation candidates: 19
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`62 passed in 0.59s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`113 passed in 0.63s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.58s`); `git diff --check` clean.
- Remaining limit: this source defines Lee/RADPF reduced-model assumptions and
  calibrations. It does not validate the code against same-shot experimental
  current traces, phase endpoints, spatial profiles, detector response,
  neutron spectra/anisotropy, or independent beam-target calibration.

### 2026-05-05: Blagoev 2025 Electric-Flux Formation Diagnostic Target

- Reviewed
  `KnowledgeReference/measurement-of-electric-flux-emission-a-new-diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v.md`.
- Closed
  `KnowledgeReference/measurement-of-electric-flux-emission-a-new-diagnostic-for-the-dense-plasma-focus-a-b-blagoev12aa-v-4.md`
  as a duplicate header/PDF-name variant of the same paper.
- Added `blagoev_electric_flux_diagnostic_targets()` as a formation-symmetry
  and electric-flux diagnostic target, not a neutron-yield validation target.
- Encoded the University of Sofia plasma focus context: `3 kJ` Mather device,
  `20 uF`, up to `40 kV`, hollow copper tube anode diameter `2 cm`, anode
  length `14.5 cm`, six cathode rods of `0.8 cm` diameter and `16 cm` length,
  cathode radius `3.5 cm`, chamber inner diameter `15.5 cm`, chamber height
  `35 cm`, and operation with air, argon, or deuterium.
- Encoded shot examples: shot `665`, argon `0.95 Torr`, `19.0 kV`; shot `668`,
  argon `0.83 Torr`, `19.1 kV`; and shot `667`, argon `0.77 Torr`, `19.0 kV`,
  with a reference singularity time `3.03 us`.
- Encoded diagnostic requirements: three symmetric identical D-dot probes
  placed through a hexagonal support, SMA central pins as floating conductors,
  `50 ohm` coax termination at both ends, CH2/CH3/CH4 probe channels, `1 ns`
  sampling, `10` point smoothing, and integration after baseline correction.
- Encoded calibration constraints: central-conductor symmetry test, voltage
  divider resistances `1306 ohm` and `13.2 ohm`, applied voltage `5.34 kV`,
  integrated D-dot maxima within `3%` of their mean, and `C1` capacitance
  ballpark `0.006 pF`.
- Encoded phase/symmetry interpretation: current maximum marks end of rundown,
  the interval from current maximum to current-derivative singularity is radial
  phase, lower pressure produces earlier singularity, similar D-dot shape and
  magnitude in formation/rundown indicate adequate symmetry, radial-phase
  divergence indicates changing azimuthal behavior, and Rogowski `dI/dt` can be
  contaminated by electric-flux pickup.
- Current corpus status after this ratchet:
  - coded KR target records: 29
  - unique coded KR target source files: 25
  - DPF-named markdown files represented by coded targets: 24 of 54
  - DPF-named markdown files closed by explicit review decisions: 7 of 54
  - total DPF-named markdown files review-closed: 31 of 54
  - unreviewed DPF-named markdown files: 23
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 8
  - phase timing candidates: 8
  - spatial density candidates: 6
  - spatial magnetic/EM candidates: 8
  - spatial temperature candidates: 18
  - neutron validation candidates: 17
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`63 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`114 passed in 0.59s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.48s`); `git diff --check` clean.
- Remaining limit: this source improves startup/formation and diagnostic
  symmetry constraints. It still lacks digitized probe/current traces, per-point
  waveform uncertainty, independent phase endpoint diagnostics, calibrated
  electric-field reconstruction, same-shot density/temperature/magnetic-field
  profiles, and same-shot neutron outputs.

### 2026-05-05: Auluck 2024 Poloidal Magnetic-Field Dynamo Target

- Reviewed
  `KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus.md`.
- Closed `KnowledgeReference/poloidal-magnetic-field-in-the-dense-plasma-focus-5.md`
  as a duplicate header/PDF-name variant of the same Physics of Plasmas letter.
- Added `auluck_poloidal_magnetic_field_targets()` as a poloidal/axial
  magnetic-field scope and proposed-test target.
- Encoded the source's diagnostic warning: point measurement of axial magnetic
  field inside the plasma with a magnetic probe is treated as meaningless
  because the probe has finite `1-2 mm` spatial resolution, perturbs plasma
  flow/current, and forms a Langmuir sheath; Faraday-rotation Abel inversion is
  available for the azimuthal component but not for the axial component.
- Encoded the simple dynamo hypothesis: a curved plasma armature in the
  geomagnetic seed field generates azimuthal electric field through generalized
  Ohm's law; the Hall term is neglected as a model assumption; a zero-resistivity
  limit is used; and the magnetic Reynolds number is assumed much greater than
  one for ballpark plasma-focus values.
- Encoded GPF/GV context: coordinates scale by anode radius, density by fill-gas
  mass density, magnetic field by `B0 = mu0 * I(t) / (2*pi*a*r_tilde)`, velocity
  by `B0` and fill density, Mather-type GV surfaces resemble experimental
  plasma shapes, and the flux function evolves in Hamilton-Jacobi form.
- Encoded circuit implications: MHD codes that neglect the dynamo may
  overestimate observed current, apparent current loss may be azimuthal
  circulating current, Lee radial current fraction should vary under an external
  axial-field sweep, and equivalent loop voltage may include a geomagnetic term
  independent of charging voltage.
- Encoded proposed experiment: use a Helmholtz coil with DC variable polarity,
  a uniform axial field over the whole small DPF, maximum field not more than
  `2` times the local geomagnetic field, monitor current derivative/integrated
  current/poloidal flux emission, and look for variation near the geomagnetic
  null. Nonuniform or excessively high applied fields are explicitly not valid
  tests.
- Encoded supporting Nikulin observation: a cone-shaped copper foil on a
  `2.5 kJ` plasma focus was twisted rather than radially imploded; the source
  argues a purely azimuthal field cannot produce that torque.
- Current corpus status after this ratchet:
  - coded KR target records: 30
  - unique coded KR target source files: 26
  - DPF-named markdown files represented by coded targets: 25 of 54
  - DPF-named markdown files closed by explicit review decisions: 8 of 54
  - total DPF-named markdown files review-closed: 33 of 54
  - unreviewed DPF-named markdown files: 21
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 6
  - phase timing candidates: 8
  - spatial density candidates: 4
  - spatial magnetic/EM candidates: 6
  - spatial temperature candidates: 16
  - neutron validation candidates: 15
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`64 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`115 passed in 0.58s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.47s`); `git diff --check` clean.
- Remaining limit: this source is a model-scope and proposed-test constraint.
  It lacks the completed external-field sweep dataset, calibrated poloidal flux
  signals, radial-current-fraction response, 3D magnetic reconstruction, and
  same-shot neutron yield/anisotropy response.

### 2026-05-05: Wante 2025 UNU/ICTP Nitrogen-Ion Irradiation Target

- Reviewed
  `KnowledgeReference/regular-article-nitrogen-ion-irradiation-of-carbon-thin-lms-using-a-dense-plasma-focus-enhanced.md`.
- Added `wante_nitrogen_ion_irradiation_targets()` as an ion-beam and
  material-processing target, not a neutron or end-to-end DPF validation target.
- Encoded UNU/ICTP PF configuration: nominal `3.0 kJ` device operated at
  `2.54 kJ`, `30 uF`, `13 kV`, `156 nH`, `21.4 mOhm`, anode radius `0.95 cm`,
  cathode radius `3.2 cm`, anode length `16 cm`, anode diameter `1.9 cm`, six
  copper cathode rods, Pyrex insulator, nitrogen purity `99.999%`, optimal
  pressure `1.5 mbar`, initial vacuum `5e-3 mbar`, four preliminary shots for
  stable pinch, sample distance `38 cm`, and irradiation sequences of `6`, `12`,
  and `24` shots at `5 min` intervals.
- Encoded diagnostics and Lee fit: Yokogawa `DL7480` captures current, voltage,
  and ion signals; Faraday cup biased ion collector uses `-45 V`; ion TOF is
  defined from X-ray peak to ion peak; X-ray peak aligns with voltage peak; Lee
  current-fit parameters are `fm=0.03`, `fc=0.7`, `fmr=0.18`, and `fcr=0.85`.
- Extracted ion-beam outputs: measured nitrogen ion energy `72.40 keV`, Lee
  model ion energy `71.0 keV`, ion flux `7.2e27 ions m^-2 s^-1`, and ion
  fluence `6.4e19 ions m^-2`.
- Encoded contextual plasma scales from the source: pinch temperature order
  `1e6 K` and particle-density range `1e18-1e20 m^-3`, explicitly marked as
  contextual rather than same-shot profile validation.
- Encoded material-response constraints: nitrogen doping `7.06%`, `5.96%`, and
  `7.93%` for `6`, `12`, and `24` shots; deposition rates `1.18%`, `0.50%`,
  and `0.33%` per shot; copper impurity from anode ablation increasing to
  `2.11%` at `24` shots; fluorine falling from `12.06%` to `4.94%`; crystallite
  size increasing from `6.27 nm` to `11.16 nm`; new XRD peaks at `52` and
  `76` degrees; and interlayer spacing decreasing from `0.37 nm` to `0.340 nm`.
- Current corpus status after this ratchet:
  - coded KR target records: 31
  - unique coded KR target source files: 27
  - DPF-named markdown files represented by coded targets: 26 of 54
  - DPF-named markdown files closed by explicit review decisions: 8 of 54
  - total DPF-named markdown files review-closed: 34 of 54
  - unreviewed DPF-named markdown files: 20
- Updated triage counts after removing this source:
  - circuit waveform candidates: 5
  - phase timing candidates: 7
  - spatial density candidates: 3
  - spatial magnetic/EM candidates: 6
  - spatial temperature candidates: 15
  - neutron validation candidates: 14
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`65 passed in 0.49s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`116 passed in 0.62s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.54s`); `git diff --check` clean.
- Remaining limit: this target validates only a bounded ion-beam/material
  processing use case. It lacks digitized current/voltage/ion waveforms,
  Faraday-cup response uncertainty, absolute X-ray/ion peak times, same-shot
  density/temperature/magnetic profiles, and any neutron output.

### 2026-05-05: Kiai 2025 Double 3 MJ DPF/ICF Concept Target

- Reviewed
  `KnowledgeReference/2025-double-3mj-dense-plasma-focus-thermonuclear-icf.md`.
- Closed both local duplicates as reviewed:
  `KnowledgeReference/double-3-mj-dense-plasma-focus-for-thermonuclear-drive-inertial-confinement-fusion-5.md`
  and
  `KnowledgeReference/double-3-mj-dense-plasma-focus-for-thermonuclear-drive-inertial-confinement-fusion.md`.
- Added `kiai_double_dpf_icf_concept_targets()` as a concept and experimental
  roadmap target, not as experimental validation evidence.
- Encoded the full-scale design table: deuterium at `10 torr`, impedance
  `12.5 mOhm`, peak circuit current `20 MA`, charging voltage `200 kV`,
  capacitance `150 uF`, stored bank energy `6 MJ` total with `3 MJ` per DPF,
  inductance `35 nH`, circuit period `17.5 us`, anode radius `15 cm`, anode
  length `80 cm`, cathode radius `22.5 cm`, axial speed `29.5 cm/us`, radial
  speed `42.4 cm/us`, pinch radius `1.8 cm`, pinch lifetime `300 ns` for each
  DPF, pinch length `12 cm`, current loss factor `0.7`, mass sweep factor
  `0.13`, and induced voltage `20 MV`.
- Encoded the proposed `30 kJ` prototype table: operating voltage `50-60 kV`,
  capacitance `500 uF`, plasma/deuteron density `6e25 ions/m^3`, projected
  fusion neutron yield `1e10 neutrons/shot`, pinch efficiency `20-30%`, peak
  current `3.54-4.24 MA`, maximum pinch current `0.71-1.06 MA`, pinch radius
  `3.0 mm`, pinch length `2.0 cm`, and pinch lifetime `50 ns`.
- Encoded the HTS and pellet projections as model outputs only: HTS field
  `10-15 T`, pellet ignition temperature `10-20 keV`, simplified with-HTS
  comparison `75 MW` fusion and `30 MW` electric power, without-HTS comparison
  `25 MW` fusion and `10 MW` electric power, and an explicitly flagged extreme
  pellet power projection of `3.61 PW` fusion and `613 TW` electric.
- Encoded the proposed validation roadmap: single `30 kJ` DPF prototype,
  synchronized double `30 kJ` DPF, and full-scale fusion testing with plasma
  diagnostics, neutron-yield measurements, and high-speed imaging.
- Current corpus status after this ratchet:
  - coded KR target records: 32
  - unique coded KR target source files: 28
  - DPF-named markdown files represented by coded targets: 27 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 37 of 54
  - unreviewed DPF-named markdown files: 17
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 2
  - phase timing candidates: 7
  - spatial density candidates: 3
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 12
  - neutron validation candidates: 11
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`66 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`117 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.55s`); `git diff --check` clean.
- Remaining limit: this source is theoretical and explicitly points to future
  laboratory validation. It does not provide measured current/voltage traces,
  synchronized double-DPF timing, same-shot density/temperature/HTS-field
  profiles, DT pellet coupling diagnostics, measured neutron yield/timing/
  spectrum/anisotropy, detector response, full energy accounting, or validated
  scale-up from `30 kJ` to `6 MJ`.

### 2026-05-05: Beresnyak 2018 HAWK 3D MHD Model-Scope Target

- Reviewed `KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md`.
- Added `beresnyak_hawk_3d_mhd_targets()` as a HAWK-specific 3D MHD
  model-scope target, not as an experimental validation packet.
- Encoded HAWK setup: `665 kA` generator, `1.2 us` rise time, `720 nH`
  high-impedance generator inductance, local plasma injection by plasma guns,
  evacuated interelectrode space, and fully ionized deuterium assumption.
- Encoded circuit coupling: `720 nH`, `0.15 ohm`, `1.07 uF`, initial
  capacitor voltage `640 kV`, zero initial current, current and `dI/dt` as
  simulation inputs, azimuthal magnetic boundary from current, velocity-gradient
  boundary from `dI/dt`, and device voltage from integrated electric field.
- Encoded HAWK geometry and injected-plasma setup: anode radius `6.33 cm`,
  anode length `4 cm`, cathode radius `8.57 cm`, high-to-low injected-density
  ratio `2`, background density `1/4 rho0`, azimuthal modes `m=0`, `m=3`, and
  `m=6`, and characteristic density `1e-7 g/cc` or `3e16 cm^-3`.
- Encoded phase/current behavior: Lee-estimated density gives pinch time
  `0.95 us`, near the current peak; device voltage is typically below `10 kV`
  at the target density; short-circuit sine period is `5.2 us`; grid resolution
  examples are `480 x 480 x 288`.
- Encoded model outputs and limits: total thermal-yield metric peaks at
  `9e15 cm^-3`, thermal fusion is explicitly subdominant and not a projected
  HAWK yield, Hall-MHD positive-polarity runs give faster/tighter pinch near
  the anode, Spitzer resistivity does not qualitatively change dynamics, and
  stochastic ion acceleration gives a mostly isotropic power-law tail to about
  `200 keV`.
- Current corpus status after this ratchet:
  - coded KR target records: 33
  - unique coded KR target source files: 29
  - DPF-named markdown files represented by coded targets: 28 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 38 of 54
  - unreviewed DPF-named markdown files: 16
- Updated triage counts after removing this source:
  - circuit waveform candidates: 2
  - phase timing candidates: 6
  - spatial density candidates: 2
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 11
  - neutron validation candidates: 10
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`67 passed in 0.61s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`118 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.62s`); `git diff --check` clean.
- Remaining limit: HAWK experiments were planned in this paper, current
  disruption was not modeled, and the local extract lacks measured HAWK
  current/voltage traces, measured phase endpoints, spatial profile
  diagnostics, measured neutron yield/timing/spectrum/anisotropy, detector
  response, and uncertainty.

### 2026-05-05: Wang/Yang 1999 DPF-16 Metallic-Vapor Interferometry Target

- Reviewed
  `KnowledgeReference/observation-of-the-metallic-vapor-from-a-plasma-focus-wang-xinxin-3-yang-jinji-department-of.md`.
- Added `wang_metallic_vapor_interferometry_targets()` as an interferometry
  and anode-material-vapor target, not as a neutron or complete DPF validation
  target.
- Encoded DPF-16 setup: `16 kJ`, `20 kV`, `380 kA`, Mather type, hydrogen
  fill pressure `70-650 Pa`, typical interferograms at `200 Pa`, and
  higher-pressure vapor-development images at `330 Pa`.
- Encoded geometry: oxygen-free copper anode, anode diameter `66 mm`, anode
  and cathode length `265 mm`, tungsten target `10 mm` diameter and `6 mm`
  high, and interferometer field of view about `60 mm`.
- Encoded phase timing: `t=0` is the pinch spike in the `dI/dt` waveform and
  maximum compression above the anode; compression frames at `-200`, `-140`,
  and `-60 ns`; expansion beginning at `40 ns`; post-focus expansion at
  `200 ns`; metallic vapor visible at `280 ns`; and higher-pressure vapor
  frames at `220` and `300 ns`.
- Encoded evidence interpretation: laser differential interferometry records
  plasma-sheath evolution; a high-density volume emerges from the anode target
  after the focus is over; target erosion after many shots supports material
  evaporation; the high-density volume disappears when a hollow anode replaces
  the target; and the source links the delayed metallic plasma to hard X-ray
  emission several hundred nanoseconds after focus.
- Current corpus status after this ratchet:
  - coded KR target records: 34
  - unique coded KR target source files: 30
  - DPF-named markdown files represented by coded targets: 29 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 39 of 54
  - unreviewed DPF-named markdown files: 15
- Updated triage counts after removing this source:
  - circuit waveform candidates: 2
  - phase timing candidates: 5
  - spatial density candidates: 1
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 10
  - neutron validation candidates: 9
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`68 passed in 0.56s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`119 passed in 0.60s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.55s`); `git diff --check` clean.
- Remaining limit: this source is qualitative visual evidence. It lacks
  digitized `dI/dt`, current, voltage, interferogram phase shift, density
  inversion, vapor-species spectroscopy, X-ray time history/spectrum, electron
  beam energy/current, neutron diagnostics, detector response, and uncertainty.

### 2026-05-05: Altarabulsi 2024 Deuteron-Beam Fluence Target

- Reviewed
  `KnowledgeReference/original-deuteron-beam-fluence-emitted-from-dense-plasma-focus.md`.
- Added `altarabulsi_deuteron_beam_fluence_targets()` as a Lee-code
  deuteron-beam fluence target, not as neutron validation.
- Encoded three fitted devices: PF-1000 (`863.1 kJ`), MPEF-12 kJ (`9.7 kJ`),
  and PF-2.7 kJ (`2.7 kJ`) operated in deuterium using `RADPFV6.16FIB`.
- Encoded Table 1 device parameters, including PF-1000 `L0=33.5 nH`,
  `C0=1332 uF`, `r0=6.3 mOhm`, `a=11.5 cm`, `b=16 cm`, `V0=36 kV`,
  `p0=3.5 Torr`; MPEF-12 kJ `L0=65 nH`, `C0=40 uF`, `r0=1 mOhm`,
  `a=3 cm`, `b=5.5 cm`, `V0=22 kV`, `p0=3 Torr`; and PF-2.7 kJ
  `L0=110 nH`, `C0=30 uF`, `r0=22 mOhm`, `a=0.95 cm`, `b=3.2 cm`,
  `V0=13.5 kV`, `p0=0.15 Torr`.
- Encoded current-waveform fitting requirements: computed current is fitted to
  measured discharge current by adjusting Lee mass/current factors and
  sometimes `L0`/`r0`; the example MPEF-12 fit is to the end of pinch at about
  `2.08 us`; after that point divergence is not considered important for ion
  acceleration in this model.
- Encoded Table 3 fluence comparisons: PF-1000 at `14 cm`, `0.5 Torr`,
  simulated `7.3e19 ions/m^2` versus measured about `7.5e19`; MPEF-12 kJ at
  `14 cm`, pressures `0.76-7.5 Torr`, simulated `5.5e18-7.5e18` versus
  measured values with errors; and PF-2.7 kJ at `40 cm`, pressures
  `0.075-0.6 Torr`, simulated `1.77e15-4.94e15` versus measured values with
  errors.
- Encoded distance/application scaling: pinch-exit fluence order `1e20
  ions/m^2`, `14 cm` fluence order `1e19 ions/m^2`, PF-24 at `11 Torr` with
  pinch-exit fluence `3.87e20 ions/m^2`, flux dropping from `8.7e27` at the
  pinch exit to `2.61e26 ions/m^2/s` at `26 cm`, and energy flux dropping from
  `1.37e14` to `4.09e12 W/m^2`.
- Current corpus status after this ratchet:
  - coded KR target records: 35
  - unique coded KR target source files: 31
  - DPF-named markdown files represented by coded targets: 30 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 40 of 54
  - unreviewed DPF-named markdown files: 14
- Updated triage counts after removing this source:
  - circuit waveform candidates: 1
  - phase timing candidates: 4
  - spatial density candidates: 1
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 9
  - neutron validation candidates: 8
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`69 passed in 0.57s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`120 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.58s`); `git diff --check` clean.
- Remaining limit: this source validates a bounded ion-beam fluence workflow
  after current-waveform fitting. It lacks raw digitized current/voltage
  waveforms, raw fluence detector response, raw detector calibration,
  same-shot density/temperature/beam divergence diagnostics, complete
  uncertainty propagation, and neutron timing/spectrum/anisotropy validation.

### 2026-05-05: Narkis/Hahn 2021 Kr-Doped Gemini-Like DPF MHD Target

- Reviewed `KnowledgeReference/seyler-2021-kr-doped-dpf-mhd.md`.
- Added `narkis_kr_doped_dpf_mhd_targets()` as a 2D radiation-MHD
  model-scope target for Kr-doped, Gemini-like DPF simulations.
- Encoded the core warning from the source: fully kinetic simulations are
  required for pinch stagnation and total neutron yield; MHD cannot capture
  kinetic effects or beam-target neutron production.
- Encoded setup: HYDRA quasi-2D `R-Z` geometry with one azimuthal cell, current
  levels `2-3 MA`, Kr volume fractions `0`, `0.1%`, and `1%`, charging
  voltages `35`, `40`, `45`, and `50 kV`, experimental current data only for
  `35` and `40 kV`, anode radius `7.62 cm`, cathode radius `10.16 cm`, anode
  length `43.18 cm`, cathode length `59.18 cm`, and near-cap mesh resolution
  `200 x 200 um`.
- Encoded circuit and initial-condition limits: RLC circuit with `R=1.4 mOhm`,
  `L=40 nH`, `C=432 uF`; resistance treated as a free parameter; fill pressure
  scaled by `0.75`; matching implosion times and peak currents is described as
  a sanity check rather than strict quantitative comparison; breakdown physics
  is neglected.
- Encoded Table I: sheath-radius `5 mm` timing, ion/electron temperatures, and
  ion densities for all dopant/voltage cases. Example high-density case: `1%`
  Kr, `50 kV`, `ni=15.87e18 cm^-3`, `Ti=156 eV`, `Te=98.5 eV`, `t=6.525 us`.
- Encoded radiation and temperature results: Kr increases radiative losses,
  narrows the sheath, gives approximate peak temperatures `6.7`, `8.3`, and
  `12.6 keV` for `0%`, `0.1%`, and `1%` Kr, and leaves two-temperature
  behavior throughout radial implosion for `0.1%` and `1%` Kr.
- Encoded neutron outputs and caveats: thermonuclear yield order `1e9-1e10`,
  yield increases with Kr dopant in 2D MHD, all-point scaling exponents
  `5.726`, `4.643`, and `4.859`, and `35 kV` maximum `dN/dt` values
  `1.1e9`, `2.4e9`, and `1.8e9 neutrons/ns` for `0%`, `0.1%`, and `1%` Kr.
- Current corpus status after this ratchet:
  - coded KR target records: 36
  - unique coded KR target source files: 32
  - DPF-named markdown files represented by coded targets: 31 of 54
  - DPF-named markdown files closed by explicit review decisions: 10 of 54
  - total DPF-named markdown files review-closed: 41 of 54
  - unreviewed DPF-named markdown files: 13
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 3
  - spatial density candidates: 1
  - spatial magnetic/EM candidates: 3
  - spatial temperature candidates: 8
  - neutron validation candidates: 7
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`70 passed in 0.83s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`121 passed in 0.63s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.76s`); `git diff --check` clean.
- Remaining limit: this is not a predictive total-yield target. It lacks strict
  digitized current/voltage trace fitting, measured phase endpoints for every
  voltage/dopant case, breakdown physics, 3D instability growth, species
  separation, fully kinetic stagnation, beam-target neutron production,
  detector response, and neutron spectrum/anisotropy validation.

### 2026-05-05: Auluck 2022 DPF Theory Part 1 Extraction Decision

- Reviewed `KnowledgeReference/auluck-2022-dpf-theory-part1.md`.
- The local markdown is not usable as a line-referenced scientific target:
  despite metadata indicating a 74-page PDF with tables and figures, the
  extracted markdown contains only the final references page.
- Added an explicit `insufficient_extractable_validation_data` corpus decision
  instead of inferring theory content from the title or references.
- Current corpus status after this ratchet:
  - coded KR target records: 36
  - unique coded KR target source files: 32
  - DPF-named markdown files represented by coded targets: 31 of 54
  - DPF-named markdown files closed by explicit review decisions: 11 of 54
  - total DPF-named markdown files review-closed: 42 of 54
  - unreviewed DPF-named markdown files: 12
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 3
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 2
  - spatial temperature candidates: 8
  - neutron validation candidates: 6
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`70 passed in 0.49s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`121 passed in 0.57s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.55s`); `git diff --check` clean.
- Remaining limit: the original PDF would need to be re-ingested before any
  KR-only theory target can be extracted from this source.

### 2026-05-05: Auluck 2023 Neutron-Yield Scaling Failure Target

- Reviewed
  `KnowledgeReference/on-the-failure-of-neutron-yield-scaling-in-the-dense-plasma-focus-s-k-h-auluck-international.md`.
- Added `auluck_neutron_yield_scaling_failure_targets()` as a narrow
  theory/test target. Only the exposed conclusion and references were used.
- Encoded the source's core claim: large plasma-focus devices can abruptly stop
  following expected neutron-yield scaling above some voltage because the device
  must satisfy drive-parameter limits and generalized optimization criteria.
- Encoded the insulator-radius scaling claim: reaction yield should vary as the
  inverse fifth power of the outer-insulator-radius to anode-radius ratio; the
  source proposes reducing the ratio from typical `~1` to `~0.4` by placing the
  insulator in the shadow of the anode, with a possible two-order yield increase
  only if all optimization conditions are satisfied simultaneously.
- Encoded the proposed inexpensive tests: measure lift-off time and correlate it
  with drive parameter and insulator radius; change the operating pressure range
  by increasing insulator radius with an add-on insulator; and test insulators
  with outer radius less than the anode radius.
- Encoded the source's warning that small devices should study this scaling
  failure through lift-off-time measurements, not by using neutron measurements
  as the primary test.
- Current corpus status after this ratchet:
  - coded KR target records: 37
  - unique coded KR target source files: 33
  - DPF-named markdown files represented by coded targets: 32 of 54
  - DPF-named markdown files closed by explicit review decisions: 11 of 54
  - total DPF-named markdown files review-closed: 43 of 54
  - unreviewed DPF-named markdown files: 11
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 3
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 7
  - neutron validation candidates: 5
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`71 passed in 0.58s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`122 passed in 0.59s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.57s`); `git diff --check` clean.
- Remaining limit: equations `12` and `17`, the derivation, and the actual
  validation data are not in the markdown. Full use requires PDF re-ingestion or
  a new KR source exposing liftoff-time, pressure-range, drive-parameter, and
  neutron-yield sweeps.

### 2026-05-05: Ou/FOI 2D Dense Plasma Focus Simulation Target

- Reviewed `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus.md`.
- Closed `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md`
  as a duplicate header/PDF-name variant of the same source.
- Added `ou_foi_2d_dpf_simulation_targets()` as a 2D MHD parameter-sweep target.
- Encoded FOI model scope: electron inertia ignored, simplified Ohm law closes
  Maxwell equations, electromagnetic solver `TVD-CP`, fluid solver `RTVD`,
  adiabatic single-phase ideal gas, high-resistivity swept/vacuum region,
  low-resistivity plasma region, fixed electrodes, Courant number `0.5`, and
  sine-current boundary `Imax * sin(2*pi*f*t)`.
- Encoded LLNL reference case: anode diameter `15.2 cm`, cathode-anode gap
  `4.3 cm`, peak current `2.5 MA`, fill pressure `2926 Pa`, sheath images at
  `3.9 us`, `6.2 us`, `7.4 us`, and breakup at `7.4 us`. The source says
  simulated morphology agrees with LLNL optical framing images but timing
  differs greatly.
- Encoded current sweep: amplitudes `1.5`, `2.0`, `2.5`, `3.0`, `3.5 MA`;
  pinch times `188.99`, `155.08`, `135.65`, `123.40`, `114.29 ns`; quarter
  period `135 ns`; and corresponding pinch currents `1.213`, `1.946`, `2.500`,
  `2.973`, `3.399 MA`.
- Encoded pressure/anode/gap trends: pressure sweep `133-2660 Pa`; sheath speed
  above `1e5 m/s`; sheath speed decreases with square root of pressure,
  increases with current, and decreases with anode radius; anode radii
  `30-50 mm`; gaps `15-35 mm`; gap has little effect on near-anode axial motion.
- Current corpus status after this ratchet:
  - coded KR target records: 38
  - unique coded KR target source files: 34
  - DPF-named markdown files represented by coded targets: 33 of 54
  - DPF-named markdown files closed by explicit review decisions: 12 of 54
  - total DPF-named markdown files review-closed: 45 of 54
  - unreviewed DPF-named markdown files: 9
- Updated triage counts after removing these sources:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 5
  - neutron validation candidates: 3
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`72 passed in 0.49s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`123 passed in 0.64s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.65s`); `git diff --check` clean.
- Remaining limit: this is design guidance, not a complete validation packet. It
  lacks measured current/voltage traces, timing uncertainty, quantitative LLNL
  frame alignment, density/temperature/magnetic-field diagnostics, and neutron
  outputs.

### 2026-05-05: Sun 2025 Two-Temperature MHD Motion Target

- Reviewed
  `KnowledgeReference/2025-theoretical-and-numerical-studies-on-motion-process-of-dense-plasma-focus.md`.
- Added `sun_two_temperature_mhd_motion_targets()` as a two-temperature MHD
  motion and design-scaling target for UNU / UDMPF1 / PF-1000 studies.
- Encoded model scope: nonideal two-temperature MHD coupled to an external RLC
  circuit, electron-ion thermal nonequilibrium, Braginskii transport
  coefficients, resistive effects, and qualitative/plot-based benchmark
  comparisons against UNU current/voltage and UDMPF1 radial trajectory.
- Encoded UNU circuit and geometry: charging voltage `15 kV`, capacitance
  `30 uF`, inductance `110 nH`, resistance `12 mOhm`, anode radius `0.95 cm`,
  cathode radius `3.2 cm`, cathode-anode gap `2.25 cm`, anode length `16 cm`,
  and cathode length `25 cm`.
- Encoded motion targets: axial phase `0-2.5 us`, radial implosion
  `2.78-2.90 us`, pinch around `2.8 us`, background density `2.4e23 m^-3`,
  background pressure about `3.5 Torr`, axial sheath speed up to `90 km/s`,
  axial ion-temperature rise from `1` to `100 eV`, radial density about
  `1e24 m^-3`, and radial ion temperature about `1 keV`.
- Encoded parameter-law guidance: for large DPF devices, current saturates when
  increasing capacitance or decreasing inductance; increasing circuit voltage is
  more effective; and the anode-to-cathode radius ratio should be small. The
  PF-1000 `c` cases in the source are `1.4`, `1.8`, `2.2`, and `2.6`.
- Current corpus status after this ratchet:
  - coded KR target records: 39
  - unique coded KR target source files: 35
  - DPF-named markdown files represented by coded targets: 34 of 54
  - DPF-named markdown files closed by explicit review decisions: 12 of 54
  - total DPF-named markdown files review-closed: 46 of 54
  - unreviewed DPF-named markdown files: 8
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 4
  - neutron validation candidates: 2
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`73 passed in 0.55s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`124 passed in 0.66s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.83s`); `git diff --check` clean.
- Remaining limit: the source strengthens macroscopic MHD motion, phase,
  temperature, and design-scaling targets, but it explicitly states that MHD
  cannot self-consistently resolve high-energy particle beams or neutron
  production. It also lacks digitized current/voltage traces, quantified error
  bars, density/temperature profile uncertainty, and neutron validation outputs.

### 2026-05-05: Demina/Gribkov DPF Material-Damage Irradiation Target

- Reviewed
  `KnowledgeReference/application-of-a-plasma-accelerator-of-the-dense-plasma-focus-type-in-simulation-of-radiation.md`.
- Added `demina_dpf_material_damage_targets()` as an application-response target.
  It is not core DPF machine validation.
- Encoded device and irradiation context: PF-5M bank energy `5 kJ`, PF-6 bank
  energy `7 kJ`, PF-1000 bank energy `1.2 MJ`, PF-1000 experimental stored
  energy about `600 kJ`, deuterium working gas at `470 Pa`, sample exposure
  power flux `1e7-1e10 W/cm2`, pulse duration `0.2-1 us`, `10` W/W-CFC pulses,
  and `5` CFC/SiC pulses.
- Encoded tungsten response: melting, evaporation, wavelike relief, nanoscale
  cellular structure at `1e10 W/cm2`, intergranular/transgranular microcracks
  above `1e8 W/cm2`, bubble size around `1 um`, microcrack penetration around
  `10 um`, and table-derived erosion depths including about `2.05 um` per pulse
  for the highest ion/plasma-stream condition.
- Encoded CFC/CFC-SiC response: W droplets/ridges on CFC, stronger evaporation
  when fibers are normal to the irradiated surface, lower erosion when fibers
  are parallel to the surface, CFC-8SiC evaporated layer `2.6 um` per shot at
  `1e9 W/cm2`, and CFC-40SiC `1.9 um` per shot.
- Encoded redeposition observations: Cu/O/Fe/Cr on W, Fe/Cr/Si/Cu on CFC-SiC,
  steel-holder sources for Fe/Cr, copper-anode source for Cu, and possible
  compounds `Fe2C`, `Fe5C2`, `Cu4Si`, and `(Cr,Fe)7C3`.
- Current corpus status after this ratchet:
  - coded KR target records: 40
  - unique coded KR target source files: 36
  - DPF-named markdown files represented by coded targets: 35 of 54
  - DPF-named markdown files closed by explicit review decisions: 12 of 54
  - total DPF-named markdown files review-closed: 47 of 54
  - unreviewed DPF-named markdown files: 7
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 1
  - spatial temperature candidates: 3
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`74 passed in 0.69s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`125 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.63s`); `git diff --check` clean.
- Remaining limit: the source can bound DPF-driven material erosion and
  redeposition, but it does not provide current/voltage waveforms, incident
  particle spectra, sample-distance tables by condition, same-shot plasma
  profiles, neutron observables, or uncertainty budgets.

### 2026-05-05: Unity Front-End Guide Review Decision

- Reviewed
  `KnowledgeReference/building-a-sci-fi-themed-dense-plasma-focus-simulation-front-end-in-unity.md`.
- Classified the source as `non_scientific_frontend_guide`.
- No validation target was added. The document is a Unity/URP/VFX Graph/UI,
  raymarching, data-ingestion, and WebSocket display tutorial. It is not a
  verified DPF physics source and does not provide KR-backed equations,
  experimental targets, diagnostics, or model-validation data.
- Current corpus status after this ratchet:
  - coded KR target records: 40
  - unique coded KR target source files: 36
  - DPF-named markdown files represented by coded targets: 35 of 54
  - DPF-named markdown files closed by explicit review decisions: 13 of 54
  - total DPF-named markdown files review-closed: 48 of 54
  - unreviewed DPF-named markdown files: 6
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 1
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 2
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`74 passed in 0.48s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`125 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.57s`); `git diff --check` clean.
- Remaining limit: this source is review-closed only to keep the KR scientific
  queue accurate. It is not evidence for DPF scientific accuracy.

### 2026-05-05: Lee 2014 Radiative Lee-Model Review Target

- Reviewed `KnowledgeReference/lee-2014-plasma-focus-radiative-model.md`.
- Added `lee_2014_radiative_model_review_targets()` as a peer-reviewed
  equation/scope target for the radiative Lee model.
- Encoded 5-phase scope: axial snowplow, radial inward shock slug model,
  radial reflected shock, slow compression/pinch, expanded column, plus optional
  Type-2 `Phase 4a` anomalous-resistance extension.
- Encoded timing/model constraints: radial inward phase equation set
  `14,15,17,19`; reflected-shock equation set `34,35,36,37`; reflected-shock
  speed fraction `0.3`; axial phase ends when the current sheath reaches the
  anode end; radial inward phase ends when the shock reaches axis; pinch phase
  ends after one small-disturbance transit time.
- Encoded radiative-pinch terms: Joule heating, Spitzer resistance, Bennett
  temperature, Bremsstrahlung, line radiation, total `dQ/dt`, self-absorption,
  surface-emission transition, radiation collapse, deuterium critical current
  `1.6 MA`, and Ne/Ar critical current below `100 kA`.
- Current corpus status after this ratchet:
  - coded KR target records: 41
  - unique coded KR target source files: 37
  - DPF-named markdown files represented by coded targets: 36 of 54
  - DPF-named markdown files closed by explicit review decisions: 13 of 54
  - total DPF-named markdown files review-closed: 49 of 54
  - unreviewed DPF-named markdown files: 5
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 1
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`75 passed in 0.74s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`126 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.66s`); `git diff --check` clean.
- Remaining limit: this is an equation/scope source, not an experimental
  validation packet. It lacks measured waveforms, shock/piston trajectories,
  radiated-power traces, profile diagnostics, neutron observables, and
  uncertainty budgets; the local extract also omits equations `51`, `52`, and
  `53`.

### 2026-05-05: Focus Fusion p-B11 Correction-Only Decision

- Reviewed
  `KnowledgeReference/2023-correction-to-focus-fusion-overview-of-progress-towards-p-b11-fusion-with-the.md`.
- Classified the source as `correction_only`.
- The one-page correction fixes the original Focus Fusion abstract's highest
  `n tau T` product to `3.4e20 keV-s/m3`.
- No new target was added. The corrected value is already encoded in
  `ff1_focus_fusion_plasmoid_targets()` from the canonical original Focus
  Fusion source.
- Current corpus status after this ratchet:
  - coded KR target records: 41
  - unique coded KR target source files: 37
  - DPF-named markdown files represented by coded targets: 36 of 54
  - DPF-named markdown files closed by explicit review decisions: 14 of 54
  - total DPF-named markdown files review-closed: 50 of 54
  - unreviewed DPF-named markdown files: 4
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 1
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`75 passed in 0.49s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`126 passed in 0.61s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.60s`); `git diff --check` clean.
- Remaining limit: the correction notice adds no independent DPF validation
  data. It only fixes a scalar abstract value in a target already represented by
  the canonical source.

### 2026-05-05: McAlpine 2014 DPF/NRTA MCNP Application Target

- Reviewed
  `KnowledgeReference/monte-carlo-simulations-of-neutron-resonance-transmission-analysis-with-the-dense-plasma-focus.md`.
- Added `mcalpine_dpf_nrta_mcnp_targets()` as a downstream neutron-resonance
  transmission analysis application target, not a DPF plasma-validation target.
- Encoded DPF source context: LLNL DPF D-D `2.45 MeV` neutrons, yield about
  `1e7`, simulated pulse duration `20-60 ns`, generic DPF yield `1e4-1e13`
  neutrons in `10-100 ns`, deuterium working gas, optional DT context, and
  kinetic simulations used to inform desired yield/pinch length.
- Encoded MCNP/NRTA setup: monoenergetic isotropic point source, `3 cm`
  polyethylene moderator, detector volume `2 m` away, assumed `3He` detector
  with `1/v` absorption postprocessing, inspection object about `180 cm3`,
  Gaussian DPF pulse FWHM `20 ns`, conventional ENG trapezoid `4 us`, and
  `1e10` source particles per simulation.
- Encoded application results: TOF slightly broadens resonances but preserves
  locations; DPF resolves resonances not detectable with ENG; an ENG would take
  about a day for comparable resolvable measurement while DPF can do it in one
  pulse; depleted uranium, highly enriched uranium, plutonium, and lead were
  compared and distinguished.
- Updated the corpus triage test because the remaining unreviewed DPF-named
  files have no scientific category-marker hits.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 14 of 54
  - total DPF-named markdown files review-closed: 51 of 54
  - unreviewed DPF-named markdown files: 3
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 0
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.49s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.62s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.65s`); `git diff --check` clean.
- Remaining limit: this report models DPF-enabled NRTA, not the DPF plasma. It
  assumes a monoenergetic isotropic point source, postprocesses detector
  response, ignores room scatter/passive background, and explicitly calls for
  experiments, minimum-yield analysis, room geometry, and direct detector-
  response modeling.

### 2026-05-05: DimLifePF96 Empty Extraction Decision

- Reviewed
  `KnowledgeReference/dimensions-and-lifetime-of-the-plasma-focus-pinch-plasma-science-ieee-transactions-on-2.md`.
- Classified the source as `insufficient_extractable_validation_data`.
- No validation target was added. The local markdown contains only a
  title/source header and page stub, so pinch dimensions and lifetime cannot be
  extracted under the KR-only rule.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 15 of 54
  - total DPF-named markdown files review-closed: 52 of 54
  - unreviewed DPF-named markdown files: 2
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 0
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.59s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.73s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.96s`); `git diff --check` clean.
- Remaining limit: this source needs re-ingestion from the original PDF before
  any KR-only pinch dimension, lifetime, or diagnostic target can be extracted.

### 2026-05-05: DPF-Bi-RRT Acronym-Collision Decision

- Reviewed
  `KnowledgeReference/dpf-bi-rrt-an-improved-path-planning-algorithm-for-complex-3d-environments-with-adaptive-sampling.md`.
- Classified the source as `non_dpf_acronym_collision`.
- In this IEEE Access path-planning paper, DPF means Dual Potential Field in
  the `DPF-Bi-RRT*` algorithm for autonomous aerial vehicle navigation. It is
  unrelated to Dense Plasma Focus physics.
- No validation target was added.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 16 of 54
  - total DPF-named markdown files review-closed: 53 of 54
  - unreviewed DPF-named markdown files: 1
- Updated triage counts after removing this source:
  - circuit waveform candidates: 0
  - phase timing candidates: 0
  - spatial density candidates: 0
  - spatial magnetic/EM candidates: 0
  - spatial temperature candidates: 0
  - neutron validation candidates: 0
  - uncertainty candidates: 0
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.54s`)
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.66s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.73s`); `git diff --check` clean.
- Remaining limit: none for Dense Plasma Focus; this file is outside the
  scientific domain.

### 2026-05-05: DPF Simulator Software-Performance Summary Decision

- Reviewed
  `KnowledgeReference/optimization-and-development-of-a-dense-plasma-focus-simulator.md`.
- Classified the source as `non_scientific_software_performance_summary`.
- No validation target was added. The two-page local source summarizes DPF
  simulator software architecture and performance claims: GUI, solvers,
  ML-control, visualization, Metal GPU acceleration, CPU utilization, memory,
  and FPS. It provides no verified DPF physics equations, diagnostics,
  calibration data, validation targets, or uncertainty data.
- Updated corpus tests for the completed review state: all DPF-named markdown
  files are now reviewed, but `kr_corpus_review_status()["passed"]` remains
  false because validation coverage and same-scope predictive evidence remain
  incomplete.
- Current corpus status after this ratchet:
  - coded KR target records: 42
  - unique coded KR target source files: 38
  - DPF-named markdown files represented by coded targets: 37 of 54
  - DPF-named markdown files closed by explicit review decisions: 17 of 54
  - total DPF-named markdown files review-closed: 54 of 54
  - unreviewed DPF-named markdown files: 0
- Updated triage status:
  - `kr_unreviewed_dpf_source_triage()` passes.
  - unreviewed DPF-named markdown files: 0
  - all tracked scientific category counts: 0
- Remaining KR target coverage blockers after full DPF-named corpus review:
  `circuit_waveform`, `phase_timing`, and `spatial_temperature`; same-scope
  predictive readiness remains false.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/kr_corpus.py src/dpf/validation/__init__.py tests/test_kr_targets.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.54s`)
  - Semantic/source audits passed:
    `source True`, `semantic True []`
  - Broad post-ratchet verification passed:
    `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    (`127 passed in 0.65s`);
    `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    (`87 passed, 3 skipped in 1.80s`); `git diff --check` clean.
- Remaining limit: all DPF-named local markdown files have now been reviewed or
  target-extracted, but the repository is still not a validated end-to-end
  predictive DPF simulation tool. The remaining blockers are validation depth
  and implementation fidelity, not unreviewed DPF-named KR files.

### 2026-05-05: Corpus-Review Completion Plan Update

- Reviewed the post-corpus status after all DPF-named markdown files reached
  closure.
- Updated `kr_corpus_review_status()["next_ratcheting_steps"]` so it no longer
  asks for unreviewed-source extraction after the queue is empty.
- The code now reports this local plan:
  - DPF-named KnowledgeReference markdown review is complete.
  - Close remaining target coverage blockers: `circuit_waveform`,
    `phase_timing`, and `spatial_temperature`.
  - Promote one same-scope validation packet by adding KR-backed circuit,
    phase, spatial, neutron, and uncertainty evidence for a single
    device/shot/scope, or keep readiness blocked when KR lacks those
    observables.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.50s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`127 passed in 0.61s`)
  - `git diff --check` clean.
- Remaining limit: source review is no longer the ratchet. The next ratchet
  must improve validation evidence or explicitly preserve readiness blockers
  where KR data is absent.

### 2026-05-05: Same-Scope Closure-Path Report

- Reviewed same-scope target status after full DPF-named corpus closure.
- Added `widest_available_scope` and `next_same_scope_steps` to
  `kr_validation_same_scope_target_report()`.
- The report now distinguishes:
  - `best_available_scope`: MJOLNIR currently has fewer total blockers but is
    missing several required groups.
  - `widest_available_scope`: PF-1000 full-energy
    `pf1000_full_energy_2007_gribkov_scholz` has all required groups present
    but remains incomplete because most groups are partial.
- PF-1000 full-energy partial blockers: `circuit_waveform`,
  `neutron_anisotropy`, `neutron_detector_response`, `neutron_spectrum`,
  `neutron_timing`, `phase_timing`, `spatial_magnetic_or_em`,
  `spatial_temperature`, and `uncertainty`.
- The code-level next step now says to use the widest same-scope packet as the
  closure path and keep predictive readiness blocked until those partial groups
  have digitized traces, uncertainty, and same-shot diagnostic support.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`127 passed in 0.61s`)
  - `git diff --check` clean.
- Remaining limit: the KR corpus, as currently extracted, gives a broad
  PF-1000 packet but not a complete predictive validation packet. Closing it
  requires digitized waveform, phase, spatial, neutron, and uncertainty evidence
  for the same PF-1000 scope, or explicit permanent blockers where KR lacks
  those observables.

### 2026-05-05: PF-1000 Closure-Blocker Checklist

- Added `closure_blockers` and `closure_blocker_groups` to
  `kr_validation_same_scope_target_report()`.
- The PF-1000 full-energy scope now reports the exact blocker checklist for
  each partial group rather than only listing group names.
- Encoded blocker examples:
  - `circuit_waveform`: `digitized_current_trace_points`
  - `phase_timing`: `radial_transit_start_and_end_times`
  - `spatial_temperature`: `direct_experimental_temperature_diagnostic`
  - `neutron_detector_response`:
    `neutron_field_transport_or_room_scatter_response_model`
  - `uncertainty`: `fast_ion_distribution_uncertainty`
- Review outcome: the current closure path is not more validated than before;
  it is more auditable. The plan now has code-level blockers that can be used as
  the extraction checklist or as explicit KR-absence reasons.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q`
    passed (`1 passed in 0.53s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 0.49s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`127 passed in 0.64s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`87 passed, 3 skipped in 1.64s`)
  - `git diff --check` clean.
- Remaining limit: the missing observables are still missing from the coded
  target packet. The next ratchet must either extract more same-scope PF-1000
  evidence from already reviewed KR line windows or mark each blocker as a
  KR-absence gate that predictive readiness cannot pass.

### 2026-05-05: Broad DPF-Content Corpus Queue

- Updated corpus review accounting from filename-only DPF relevance to
  filename-or-strong-content DPF relevance.
- Strong content markers: `dense plasma focus`, `plasma focus`, `PF-1000`,
  `PF1000`, `PF 1000`, `MJOLNIR`, `Mather-type`, and `Filippov`.
- Current inventory:
  - 827 total local source files
  - 398 markdown files
  - 396 JSON files
  - 54 DPF-named markdown files
  - 94 DPF-content markdown files
  - 96 DPF-relevant markdown files by filename or content
- Current review status:
  - 55 of 96 DPF-relevant markdown files are review-closed by coded target or
    explicit decision.
  - 41 DPF-relevant markdown files remain open.
- Review outcome: the earlier DPF-named queue is complete, but that is not the
  same as complete DPF-relevant corpus review. The plan is now corrected to
  process the remaining content-hit files before claiming source closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py tests/test_kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.37s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`76 passed in 1.24s`)
- Remaining limit: the 41 broad content-hit files still need review. Each must
  become either a coded target or an explicit decision before source closure is
  honest.

### 2026-05-05: Broad DPF-Content Review Wave 1

- Added explicit review decisions for 20 of the 41 newly exposed broad
  DPF-content markdown files.
- Closed categories:
  - duplicate FAETON-I and hybrid X-pinch extractions
  - non-DPF/reference-only Z-pinch papers
  - general Z-pinch snowplow/scaling papers without DPF device targets
  - educational, software-manual, image-index, and application/materials files
    that do not contain DPF machine-validation observables
- Current status:
  - 75 of 96 DPF-relevant markdown files are review-closed.
  - 21 DPF-relevant markdown files remain open.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.09s`)
- Remaining limit: the remaining 21 files are more likely to contain useful DPF
  science or diagnostics, so the next wave must review them individually rather
  than bulk-closing them as reference-only.

### 2026-05-05: Broad DPF-Content Review Closure

- Added two coded targets from the broad content queue:
  - `mjolnir_first_experiments_2021_offermann`
  - `uofsi_argon_temperature_thesis_2020`
- Added explicit review decisions for the remaining broad content-hit files.
- Current source-review status:
  - 96 of 96 DPF-relevant markdown files are review-closed.
  - The unreviewed DPF-relevant queue is empty.
- Remaining code-reported validation blockers:
  - `circuit_waveform`
  - `phase_timing`
  - `spatial_temperature`
  - `uncertainty`
- Review outcome: the source-review question is now answered more honestly than
  the prior filename-only pass. The DPF-relevant markdown corpus has been
  reviewed into either coded target records or explicit non-target/duplicate/
  context decisions. The project remains blocked by validation evidence quality,
  not by unreviewed DPF-relevant markdown sources.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_corpus.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_corpus.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_corpus.py -q` passed (`4 passed in 1.13s`)
  - `python3 -m pytest tests/test_kr_targets.py -q` passed (`74 passed in 0.46s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`129 passed in 1.29s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`87 passed, 3 skipped in 5.88s`)
  - `git diff --check` clean.
- Remaining limit: no same-scope validation packet passes. The next phase must
  either improve same-scope evidence or make the final product explicitly block
  high-fidelity claims when KR lacks digitized traces, phase endpoints, spatial
  temperature/density/B validation, detector response, and propagated
  uncertainty.

### 2026-05-05: Source-Review Gap Closure In Readiness Reports

- Added a `kr_source_review` entry to `scientific_accuracy_gap_report()`.
- Current status: `kr_source_review` is `supported` because the
  DPF-relevant markdown queue is empty.
- Updated the `kr_target_coverage` blocker so it points at the widest
  same-scope closure path when target evidence remains partial.
- Review outcome: the app/readiness layer now distinguishes the closed source
  review from the still-open validation-evidence problem. This prevents future
  status summaries from saying "source review remains" when the real blockers
  are digitized traces, same-scope phase/spatial evidence, detector response,
  and UQ.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_mhd_physics_integration.py::test_predictive_readiness_exported_and_blocks_unvalidated_claims -q`
    passed (`2 passed in 1.16s`)
  - `python3 -m pytest tests/test_quality_assessment.py -q` passed
    (`51 passed in 2.38s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`129 passed in 3.13s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`87 passed, 3 skipped in 7.50s`)
  - `git diff --check` clean.
- Remaining limit: high-fidelity readiness is still false. This is now clearly
  because target evidence is incomplete, not because the KR source queue is
  unreviewed.

### 2026-05-05: Same-Scope Uncertainty Packet Gate

- Tightened `uncertainty_evidence_from_result()` so a complete UQ component set
  must share one `validation_scope`.
- Cross-scope uncertainty component packets now fail with
  `same_scope_uncertainty_packet`.
- Review outcome: UQ can no longer be assembled from unrelated scopes to satisfy
  high-fidelity readiness.
- Verification:
  - `python3 -m py_compile src/dpf/validation/uncertainty_budget.py tests/test_uncertainty_budget.py`
  - `python3 -m pytest tests/test_uncertainty_budget.py -q` passed
    (`10 passed in 0.91s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q`
    passed (`2 passed in 0.65s`)
- Remaining limit: real runs still need a same-scope uncertainty packet with
  experimental, input, numerical, model-form, shot-to-shot, propagated,
  acceptance-rule, and KR-target components.

### 2026-05-05: Same-Scope Physics-Fidelity Packet Gate

- Tightened `physics_fidelity_evidence_from_result()` so a complete
  high-fidelity physics-effect packet must share one `validation_scope`.
- Cross-scope physics-effect packets now fail with
  `same_scope_physics_packet`.
- Review outcome: high-fidelity physics readiness can no longer be assembled
  from unrelated validation scopes for EOS/conductivity, ionization,
  two-temperature partition, radiation transport, ablation/impurity mixing,
  Hall/FLR/kinetic effects, 3D instabilities, flashover, restrike, and
  beam-target coupling.
- Verification:
  - `python3 -m py_compile src/dpf/validation/physics_fidelity.py tests/test_physics_fidelity.py`
  - `python3 -m pytest tests/test_physics_fidelity.py -q` passed
    (`7 passed in 1.32s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_physics_fidelity.py::test_complete_physics_effects_must_share_validation_scope tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q`
    passed (`3 passed in 0.85s`)
  - `python3 -m pytest tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`89 passed, 3 skipped in 7.71s`)
- Remaining limit: real runs still need one KR-backed claim scope whose
  required high-fidelity physics effects are implemented and validated, or
  explicitly bounded out for that same scope.

### 2026-05-05: Same-Scope Circuit/Field-Coupling Packet Gate

- Tightened `field_coupling_evidence_from_result()` so a complete
  field-coupling component packet must share one `validation_scope`.
- Cross-scope field-coupling packets now fail with
  `same_scope_field_coupling_packet`.
- Updated `scientific_accuracy_gap_report()` so a complete-but-cross-scope
  field-coupling packet is `blocked`, not merely `partial`.
- Review outcome: MHD-mode current prediction cannot be promoted by combining
  inductance, dL/dt/back-EMF, Poynting power, circuit energy, transition
  metadata, and KR experimental comparison evidence from unrelated scopes.
- Verification:
  - `python3 -m py_compile src/dpf/validation/circuit_field_coupling.py src/dpf/validation/quality_assessment.py tests/test_circuit_field_coupling.py`
  - `python3 -m pytest tests/test_circuit_field_coupling.py -q` passed
    (`12 passed in 0.85s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_circuit_field_coupling.py::test_complete_field_coupling_components_must_share_validation_scope tests/test_physics_fidelity.py::test_complete_physics_effects_must_share_validation_scope tests/test_uncertainty_budget.py::test_complete_uncertainty_components_must_share_validation_scope -q`
    passed (`4 passed in 1.05s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`141 passed, 3 skipped in 9.54s`)
- Remaining limit: real runs still need same-scope validated circuit/field
  coupling, not only exported coupling signals or code-verification energy
  identities.

### 2026-05-05: Global High-Fidelity Scope-Alignment Gate

- Added `same_scope_high_fidelity_claim` to
  `scientific_accuracy_gap_report()`.
- The new gap requires KR target coverage, field-coupling,
  physics-fidelity, and uncertainty packets to share at least one
  `validation_scope`.
- Complete same-scope synthetic evidence remains high-fidelity ready.
- Complete but cross-scope support packets are now blocked at the
  high-fidelity claim level even when each packet passes internally.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_scope_alignment_blocks_cross_scope_packets -q`
    passed (`3 passed in 1.30s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`142 passed, 3 skipped in 9.86s`)
- Remaining limit: this enforces claim consistency. It does not create the
  missing same-scope experimental data needed for a real DPF validation packet.

### 2026-05-05: Global Scope Gate Extended To Tier Evidence

- Extended `same_scope_high_fidelity_claim` so it also requires source
  authority, circuit validation, snowplow validation, spatial validation,
  neutron validation, and neutron detector-response evidence to share the same
  `validation_scope`.
- The global gate no longer checks only the support packets. It now aligns the
  actual tier evidence used for the predictive claim with the KR target packet,
  field-coupling packet, physics-fidelity packet, and uncertainty packet.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_scope_alignment_blocks_cross_scope_packets -q`
    passed (`3 passed in 1.31s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`142 passed, 3 skipped in 9.78s`)
- Remaining limit: this is still a gate, not data. The live gap report remains
  blocked because no real KR-backed DPF run provides the full same-scope packet.

### 2026-05-05: Same-Scope MHD Numerical-Fidelity Packet Gate

- Added `verification_scope` metadata to the MHD numerical evidence builders
  for cylindrical convergence, resistive diffusion, backend parity, MHD phase
  scope limits, and circuit-coupled energy verification.
- Tightened `mhd_numerical_fidelity_evidence_from_result()` so a complete
  Tier-3 numerical-fidelity packet must share one verification scope.
- Cross-scope numerical verification bundles now fail with
  `same_scope_mhd_numerical_packet`.
- Updated `scientific_accuracy_gap_report()` so a complete-but-cross-scope MHD
  numerical packet is `blocked`, not `partial`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/mhd_numerical_fidelity.py src/dpf/validation/circuit_field_coupling.py src/dpf/validation/quality_assessment.py tests/test_mhd_numerical_fidelity.py`
  - `python3 -m pytest tests/test_mhd_numerical_fidelity.py -q` passed
    (`21 passed in 1.09s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`143 passed, 3 skipped in 10.25s`)
- Remaining limit: this verifies numerical-packet consistency only. It does not
  validate DPF late-pinch physics or provide same-scope experimental closure.

### 2026-05-05: Same-Scope Predictive-Readiness Tier Gate

- Tightened `predictive_readiness_report()` so tiers 1, 2, 4, and 5 must share
  one `validation_scope` before the lower `predictive_ready` label can pass.
- The guard covers circuit waveform validation, snowplow phase/timing
  validation, spatial DPF validation, and same-scope neutron validation.
- Cross-scope tier evidence now fails with `Predictive validation scope
  alignment` in `missing_evidence`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py`
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_passes_only_with_all_required_tiers tests/test_quality_assessment.py::TestQualityAssessment::test_predictive_readiness_requires_one_validation_scope tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_requires_gap_closure tests/test_quality_assessment.py::TestQualityAssessment::test_high_fidelity_readiness_can_pass_with_complete_evidence_packet -q`
    passed (`4 passed in 1.11s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py -q`
    passed (`144 passed, 3 skipped in 10.45s`)
- Remaining limit: no real run currently has same-scope tier evidence across
  circuit, snowplow, spatial, and neutron validation.

### 2026-05-05: Machine-Readable KR Data-Availability Blockers

- Added `data_availability` and `required_data_to_complete` to each
  `closure_blockers` record emitted by
  `kr_validation_same_scope_target_report()`.
- Missing same-scope groups are now marked
  `absent_from_same_scope_targets`.
- Partial PF-1000 closure groups are now marked
  `partial_only_in_same_scope_targets` with the exact required data list copied
  into `required_data_to_complete`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - `python3 -m pytest tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q`
    passed (`1 passed in 0.55s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`78 passed in 1.13s`)
- Remaining limit: the report now says more clearly what is absent or partial
  in the reviewed KR targets. It does not manufacture the absent digitized
  traces, uncertainties, detector response, or same-shot diagnostics.

### 2026-05-05: Verification Sweep Checkpoint

- Validation/KR/readiness regression sweep passed:
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py tests/test_kr_targets.py tests/test_kr_corpus.py -q`
  - Result: `222 passed, 3 skipped in 11.85s`
- `git diff --check` is clean.
- Full `python3 -m pytest -q` was attempted and aborted during collection while
  importing `dpf.metal.mlx_device` from `tests/test_amr_mlx.py`. The traceback
  indicates a fatal Python abort in the MLX import path before assertions ran.
- Claim-surface scan found current blocker language in README/SCOPE/JOSS-style
  files and tests; remaining positive hits are historical planning/debate docs
  or non-DPF backend production notes.
- Current live gap report:
  - KR source review: supported, 96/96 DPF-relevant markdown files closed.
  - KR target coverage: partial; PF-1000 full-energy remains the widest closure
    path.
  - Predictive and high-fidelity readiness: blocked by missing same-scope
    validation evidence, physics-fidelity evidence, field coupling, UQ, and
    Tier-3/Tier-4/Tier-5 validation evidence.

### 2026-05-06: User Decisions And Next Scientific-Closure Plan

- User decisions captured:
  - New source-of-truth material is allowed only after an AI researches and
    provides a link/source document and the user acquires the correct document.
  - Manual digitization from existing or newly acquired source documents is
    allowed, but the project needs a reproducible one-for-one verification
    method before using digitized data for validation claims.
  - Device choice is secondary; the physics closure matters.
  - Product target is a full high-fidelity neutron-predictive DPF simulator.
  - Scientific closure is priority 1; product hardening is priority 2.
- Next plan:
  1. Build a digitization provenance and verification workflow: source file
     hash, figure/page/axis metadata, calibration points, extracted point
     arrays, reviewer check, and residual/error report against the source
     image or table.
  2. Turn the current closure blockers into a source-acquisition queue grouped
     by physics need: circuit waveform, phase timing, spatial density/B/T,
     neutron timing/spectrum/anisotropy, detector response, and uncertainty.
  3. Research candidate source documents for each queue item and provide links
     for user acquisition before adding anything to `KnowledgeReference`.
  4. After user acquisition, ingest the document locally, review it under the
     KR-only rule, extract typed targets, and update the same-scope closure
     report.
  5. Only after same-scope data exists, implement or validate the required
     physics closures: EOS/conductivity, ionization, two-temperature energy
     partition, radiation transport/opacities, impurity/ablation, Hall/FLR/PIC
     or bounded kinetic treatment, 3D instability scope, flashover/startup,
     restrike/anomalous resistance, and beam-target neutron coupling.
- Working assumption: until new acquired documents or verified digitized data
  close the evidence gaps, the code must keep both predictive and high-fidelity
  readiness blocked.

### 2026-05-06: Digitization Gate And Source-Acquisition Queue

- Added a reusable one-for-one digitization audit in
  `src/dpf/validation/digitization.py`.
- The audit requires:
  - local `KnowledgeReference/` source path
  - matching source SHA-256 hash
  - figure image path and hash for figure extractions
  - figure/table ID and page
  - x/y axis calibration with units and residual limits
  - extracted series arrays with units
  - overlay residual evidence for figure data
  - at least one accepted independent review
- The audit fails closed on `KnowledgeReference` path traversal and malformed
  review-count metadata.
- Added `scientific_closure_source_acquisition_queue()` in
  `src/dpf/validation/source_acquisition.py`.
- The queue is built from live blockers in
  `kr_validation_same_scope_target_report()` and keeps candidate sources
  separate from source-of-truth evidence.
- Added `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` as the actionable acquisition
  list for the user. It records candidate DOI links for:
  - PF-1000 circuit waveform and phase timing
  - direct or bounded spatial temperature
  - same-scope uncertainty
  - neutron anisotropy
  - neutron detector response
  - neutron spectrum
  - neutron timing
  - spatial magnetic/EM validation
- Corrected the Zr/Be activation-detector candidate DOI to
  `10.1016/j.nima.2020.164830`.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/source_acquisition.py src/dpf/validation/__init__.py tests/test_digitization.py`
  - `python3 -m pytest tests/test_digitization.py -q` passed
    (`5 passed in 0.75s`)
  - `python3 -m pytest tests/test_digitization.py tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
    passed (`84 passed in 1.62s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_mhd_physics_integration.py tests/test_mhd_numerical_fidelity.py tests/test_circuit_field_coupling.py tests/test_physics_fidelity.py tests/test_uncertainty_budget.py tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_digitization.py -q`
    passed (`227 passed, 3 skipped in 12.22s`)
  - `git diff --check` is clean.
- Remaining limit: this closes the process and provenance gap, not the
  scientific evidence gap. The project still needs user-acquired local sources
  and/or verified digitization for same-scope current traces, phase timing,
  temperature, magnetic/EM fields, neutron timing/spectra/anisotropy,
  detector response, fast-ion distribution uncertainty, density uncertainty,
  and propagated UQ before predictive or high-fidelity readiness can pass.

### 2026-05-06: Local PDF Source Audit

- Checked local PDFs under DPF-Unified by filename, PDF metadata, DOI/title
  text extraction, and SHA-256 duplicate checks.
- Added the audit to `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`.
- Added `docs/LOCAL_PDF_SOURCE_AUDIT_2026_05_06.md`.
- Exact local matches found:
  - Akel et al. 2021, DOI `10.1016/j.radphyschem.2021.109633`, outside KR
    at `archive_reference_OLD/references/papers/core-dpf/akel-2021-pf1000-neutron-yield.pdf`
    with an identical duplicate under `archive_reference_OLD/references/papers/archive/`.
  - Gribkov et al. 2007 Part I, DOI `10.1088/0022-3727/40/7/021`.
  - Gribkov et al. 2007 Part II, DOI `10.1088/0022-3727/40/12/008`.
  - Schmidt et al. 2022 MJOLNIR high-low, DOI `10.1063/5.0089121`.
  - Malir et al. 2024 interferometry, DOI `10.1063/5.0193268`.
  - Goyon et al. 2025 neutron-generation dynamics, DOI `10.1063/5.0253547`.
- Filename problems found:
  - `gribkov-2007-pf1000-jphysd-part2` is actually Part I.
  - `scholz-2007-pf1000-part2-jphysd` is actually Gribkov et al. Part II.
  - `goyon-2022-mjolnir-high-low` is the Schmidt et al. article named by a
    non-first author.
  - `petrov-2022-mjolnir-high-low-discharges` appears to be an LLNL
    accepted-manuscript/preprint copy of the Schmidt/Goyon article.
- No exact local PDF match found for:
  - Cikhardtova et al. 2015 linear densities
  - Sadowski/Scholz/PF-1000 team 2004 fast ions/neutrons
  - Catenacci et al. 2020 neutron time-energy tomography
  - Springham et al. 2021 Zr/Be activation detectors
  - Klir et al. 2011 TOF detector calibration
  - Jednorog et al. 2017 PF-1000 activation monitor
- Correction from the subsequent parity pass: Akel et al. 2021 was already in
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
  The next action is verified digitization of waveform/yield figures and
  tables, not paper ingestion.
- Verification: `git diff --check` is clean after the audit documentation
  update.

### 2026-05-06: KR PDF Parity Verification

- Added `scripts/verify_kr_pdf_parity.py`.
- The verifier checks:
  - PDF page count equals KR JSON `page_count`
  - every PDF page's extracted text matches KR JSON `pages[].text`
  - every PDF page's extracted text is present in the KR markdown after
    normalization
  - source PDF SHA-256 is reported for provenance
- Corrected the prior audit: Akel et al. 2021 was already represented in
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
  The earlier check missed it because the KR filename is generic.
- No new `KnowledgeReference` markdown file was created. All exact local PDF
  matches already had KR markdown/JSON pairs and passed text parity:
  - Akel et al. 2021: 6/6 pages
  - Gribkov et al. 2007 Part I: 13/13 pages
  - Gribkov et al. 2007 Part II: 16/16 pages
  - Schmidt et al. 2022 MJOLNIR article: 29/29 pages
  - Schmidt/Goyon accepted-manuscript copy: 16/16 pages
  - Malir et al. 2024: 14/14 pages
  - Goyon et al. 2025 canonical KR record: 10/10 pages
  - Goyon et al. 2025 short-name KR duplicate: 10/10 pages
- Boundary: this verifies text parity only. Figure pixels and plotted curves
  are not numeric validation evidence until they pass the digitization
  provenance gate.
- Immediate next action: verified digitization of Akel 2021 waveform/yield
  figures and tables, not paper ingestion.
- Verification:
  - `python3 -m py_compile scripts/verify_kr_pdf_parity.py`
  - `python3 -m pytest tests/test_digitization.py -q` passed
    (`5 passed in 0.85s`)
  - `git diff --check` is clean.

### 2026-05-06: Akel 2021 Typed Table Target

- Added `pf1000_16kv_shot_table_2021_akel` from
  `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`.
- The new target encodes all 24 merged Akel Table 1/Table 2 shot rows:
  pressure, L0, r0, peak current, pinch current, fitted Lee factors, axial
  speed, shock speed, piston speed, pinch density, pinch radius/length,
  computed neutron yield, measured neutron yield, and measured-yield
  uncertainty.
- Added table provenance metadata:
  - Table 1 rows: `330-583`
  - Table 2 rows: `584-837`
  - merged rows: `24`
  - table shot IDs match
  - KR markdown/PDF text parity verified
  - Akel PDF SHA-256:
    `9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b`
- Corrected the existing shot-12581 phase target `fmr` from `0.25` to `0.26`
  for table-row consistency. The prose gives `0.25`; Table 1 gives `0.26`.
  The table-backed row now preserves the table value explicitly.
- Grouped the Akel PF-1000 16 kV waveform, phase, and scalar/yield table
  targets under validation scope `pf1000_16kv_2021_akel`.
- Scientific status:
  - Closed: row-level scalar current, fitted-parameter, pinch-geometry, and
    neutron-yield targets are now available from KR without manual paper
    rereading.
  - Still open: waveform curves, phase timing curves, neutron timing,
    neutron spectrum, neutron anisotropy, detector response, and blind
    predictive acceptance criteria. This is not high-fidelity neutron
    predictive closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - focused Akel/KR target tests passed (`6 passed in 0.92s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`79 passed in 1.29s`)
  - `python3 -m pytest tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work tests/test_kr_targets.py::test_kr_validation_same_scope_target_report_requires_one_scope -q`
    passed (`2 passed in 0.68s`)
  - `git diff --check` is clean.

### 2026-05-06: Akel 2021 Scalar-Table Evidence Comparator

- Added `pf1000_16kv_akel_table_candidate_evidence()`.
- The comparator accepts either:
  - a mapping keyed by shot number
  - a list of row mappings with `shot`
  - a mapping containing `shot_rows`
- Default required fields:
  - `peak_current_kA`
  - `pinch_current_kA`
  - `axial_speed_cm_per_us`
  - `shock_speed_cm_per_us`
  - `piston_speed_cm_per_us`
  - `pinch_density_1e23_per_m3`
  - `pinch_radius_cm`
  - `pinch_length_cm`
  - `neutron_yield_n`
- The neutron-yield target is the measured yield from Akel Table 2. The
  article's computed Lee yield remains available in the table rows as source
  context, but the validation comparison defaults to measured yield.
- Evidence output includes:
  - required/provided row counts
  - missing shots
  - extra shots
  - missing fields
  - field pass/fail flags
  - maximum relative errors
  - per-shot/per-field errors
- Boundary: this can pass only scalar table agreement. It does not close
  waveform, phase timing, neutron timing, spectrum, anisotropy, detector
  response, or blind predictive acceptance.
- Updated the source-queue/audit docs so the remaining Akel work is figure
  digitization; table rows are no longer listed as uningested.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_kr_targets.py`
  - focused comparator tests passed (`3 passed in 0.70s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py -q`
    passed (`81 passed in 1.24s`)
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
    passed (`82 passed in 1.47s`)
  - broader validation slice passed (`230 passed, 3 skipped in 12.36s`)
  - `git diff --check` is clean.

### 2026-05-06: Tier-5 Scalar-Yield Closure Gate

- Tightened `neutron_validation_scope_closure_report()`.
- Tier 5 now requires same-scope evidence for:
  - scalar neutron yield
  - neutron mechanism/timing
  - neutron spectrum
  - neutron anisotropy
- Added `neutron_yield_validation` to source-authority auditing.
- Updated `validation_tier_report()` and predictive-readiness wording to say
  `Neutron yield/mechanism/timing/spectrum/anisotropy validation`.
- Updated `pf1000_16kv_akel_table_candidate_evidence()` so passing scalar yield
  table comparison exposes `validated_features={"yield": True}`.
- Updated docs that still described neutron validation as
  timing/spectrum/anisotropy-only.
- Resulting scientific status:
  - A timing/spectrum/anisotropy packet alone is no longer Tier-5 supported.
  - App-level MJOLNIR helper evidence now remains `decomposed_estimate` until
    a same-scope scalar-yield validation packet is attached.
  - The Akel table comparator can supply scalar-yield evidence for the
    PF-1000 16 kV Akel scope, but that scope still lacks waveform, neutron
    timing, spectrum, anisotropy, and detector-response closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py`
  - focused Tier-5/yield tests passed (`6 passed in 0.90s`)
  - `python3 -m pytest tests/test_quality_assessment.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py -q`
    passed (`171 passed, 3 skipped in 8.84s`)
  - broader validation slice passed (`230 passed, 3 skipped in 12.49s`)
  - `git diff --check` is clean.

### 2026-05-06: Neutron-Yield KR Target Group

- Added `neutron_yield` to the end-to-end KR target groups.
- `_typed_observable_groups()` now treats `neutron_yield_targets` as a first-
  class observable group.
- Same-scope closure blockers now report neutron-yield missing/partial status.
- `scientific_closure_source_acquisition_queue()` now has priority-1
  `neutron_yield` items.
- PF-1000 full-energy target now explicitly records scalar-yield context:
  - yield range `5.0e10` to `2.0e11` neutrons/shot
  - maximum yield `6.0e11` neutrons/shot
  - shot-3121 activation anisotropy availability
  - 90 degree bubble-detector cross-check angle
  - same-scope detector response required for predictive yield
- Current widest scope remains `pf1000_full_energy_2007_gribkov_scholz`, but
  neutron yield is now a partial group with blockers:
  - `yield_calibration_uncertainty`
  - `neutron_field_transport_or_room_scatter_response_model`
  - `fast_ion_distribution_uncertainty`
- Updated `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` with a priority-1
  `neutron_yield` section.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py src/dpf/validation/source_acquisition.py tests/test_kr_targets.py tests/test_digitization.py`
  - `python3 -m pytest tests/test_kr_targets.py tests/test_kr_corpus.py tests/test_digitization.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
    passed (`87 passed in 1.90s`)
  - broader validation slice passed (`230 passed, 3 skipped in 12.30s`)
  - `git diff --check` is clean.

### 2026-05-06: App-Level Akel Scalar-Yield Validation Hook

- Added a PF-1000 16 kV Akel table hook in `_apply_post_processing()`.
- Accepted input keys:
  - `pf1000_16kv_akel_table_predictions`
  - `akel_2021_table_predictions`
  - `neutron_yield_validation_rows`
- Guardrails:
  - device/preset must identify PF-1000
  - circuit voltage must be within 5 percent of 16 kV
  - row comparison must pass `pf1000_16kv_akel_table_candidate_evidence()`
- Passing rows are promoted to `neutron_yield_validation`.
- Failing/incomplete rows remain `neutron_yield_validation_candidate`.
- App neutron scope closure now runs when only scalar-yield validation is
  present, so timing/spectrum/anisotropy blockers are visible.
- Scientific status:
  - Closed: production result dictionaries can now carry KR-backed scalar-
    yield validation for the Akel PF-1000 16 kV scope.
  - Still open: the hook needs the full 24-shot table. A single run yield does
    not validate predictive neutron performance.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - focused app Akel/Tier-5 tests passed (`3 passed in 1.72s`)
  - broader validation slice passed (`232 passed, 3 skipped in 13.01s`)
  - `git diff --check` is clean.

### 2026-05-06: Akel 2021 Figure Digitization Queue

- Added `scientific_closure_digitization_queue()`.
- Exported it from `dpf.validation`.
- The queue now tracks the six remaining Akel 2021 figure tasks:
  - Fig. 1 current waveform, shot 12581, 1.2 Torr, source lines 294-295.
  - Fig. 2 current waveform, shot 12584, 1.2 Torr, source lines 296-297.
  - Fig. 3 current waveform, shot 12592, 1.05 Torr, source lines 298-299.
  - Fig. 4 current waveform, shot 12604, 1.05 Torr, source lines 300-301.
  - Fig. 5 neutron-yield plot, 1.2 Torr, source line 916.
  - Fig. 6 neutron-yield plot, 1.05 Torr, source line 917.
- Each task records:
  - KR markdown path and SHA-256
  - parity-verified Akel PDF SHA-256
  - local PDF candidates
  - required series
  - page hint
  - required digitization packet fields
  - `digitization_verification_evidence()` as the gate
  - `figure_image_status="not_extracted"`
- Updated `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` so the local Akel figure
  work is explicit and tested.
- Scientific status:
  - Closed: figure digitization is now represented as machine-readable local
    closure work instead of an unstructured note.
  - Still open: no extracted figure image, axis calibration, digitized series,
    overlay residual, or independent review exists yet.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/__init__.py tests/test_digitization.py`
  - focused digitization/KR/quality tests passed (`89 passed in 1.88s`)
  - broader validation slice passed (`234 passed, 3 skipped in 13.47s`)
  - `git diff --check` is clean

### 2026-05-06: Digitization Queue Acceptance Status

- Added `scientific_closure_digitization_status()`.
- Exported it from `dpf.validation`.
- The status function evaluates future digitization packets against both:
  - `digitization_verification_evidence()`
  - the exact local queue task metadata
- Additional task-level checks require matching:
  - task ID
  - KR source path
  - KR source SHA-256
  - local PDF SHA-256
  - source line window
  - figure ID
  - page
  - required series names
- The report separates accepted, failed, open, invalid, and extra packets.
- Scientific status:
  - Closed: the Akel digitization workflow now has a tested one-for-one
    acceptance method.
  - Still open: no real Akel figure packet has been accepted; all six figure
    tasks remain open unless packets are supplied.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/__init__.py tests/test_digitization.py`
  - `tests/test_digitization.py` passed (`10 passed in 0.44s`)
  - focused digitization/KR/quality tests passed (`92 passed in 1.48s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.80s`)
  - `git diff --check` is clean

### 2026-05-06: App-Level Digitization Closure Export

- App post-processing now exports:
  - `scientific_closure_digitization_queue`
  - `scientific_closure_digitization_status`
- If a caller supplies `scientific_closure_digitization_packets` or
  `digitization_packets`, the app evaluates them through
  `scientific_closure_digitization_status()`.
- Default production runs now show the Akel figure queue as open instead of
  leaving figure digitization outside the result metadata.
- Scientific status:
  - Closed: app results now carry local figure-digitization blockers alongside
    KR target coverage, KR corpus review status, predictive readiness, and
    high-fidelity readiness.
  - Still open: no Akel figure data has been rendered, cropped, calibrated,
    digitized, reviewed, or accepted.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - focused app/digitization tests passed (`11 passed in 1.02s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.67s`)
  - `git diff --check` is clean

### 2026-05-06: Figure-Digitization Scientific-Accuracy Gap

- Added `figure_digitization` to `scientific_accuracy_gap_report()`.
- The gap reads `scientific_closure_digitization_status` from the result when
  present, or computes the default open queue status when absent.
- Status rules:
  - `supported`: the local digitization queue is complete.
  - `partial`: at least one task is accepted, with the rest failed or open.
  - `blocked`: no task is accepted, or status is unavailable.
- App results now show `figure_digitization` blocked by `0/6` accepted local
  scientific-closure figure tasks.
- Scientific status:
  - Closed: open Akel figure digitization is now a first-class high-fidelity
    blocker.
  - Still open: no digitized figure data has been created or accepted.
- Verification:
  - `python3 -m py_compile src/dpf/validation/quality_assessment.py tests/test_quality_assessment.py tests/test_mhd_physics_integration.py`
  - targeted readiness/gap tests passed (`3 passed in 1.52s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.60s`)
  - `git diff --check` is clean

### 2026-05-06: App-Level Source-Acquisition Queue Export

- App post-processing now exports `scientific_closure_source_acquisition_queue`.
- Result payloads now carry candidate DOI links and required local-ingestion
  steps beside KR target coverage, corpus review status, digitization status,
  predictive readiness, and high-fidelity readiness.
- Scientific status:
  - Closed: app results expose the user-requested source acquisition workflow.
  - Still open: acquisition candidates are not evidence until the correct
    document is acquired, added locally under `KnowledgeReference`, reviewed,
    and digitized if required.
- Verification:
  - `python3 -m py_compile app_mhd.py tests/test_mhd_physics_integration.py`
  - focused app/source-queue tests passed (`3 passed in 1.44s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.10s`)
  - `git diff --check` is clean

### 2026-05-06: Local-vs-Acquisition Source Queue Split

- Source-acquisition queue entries now annotate DOI leads with local status
  from the local PDF parity audit.
- Queue items now separate:
  - `local_sources_available`
  - `candidate_sources_for_acquisition`
  - `candidate_sources`, retained for compatibility and annotated per source
- Tagged as `parity_verified_knowledge_reference`:
  - Akel 2021
  - Gribkov 2007 Parts I/II
  - Schmidt 2022
  - Malir 2024
  - Goyon 2025
- Tagged as `not_found_as_exact_local_pdf`:
  - Cikhardtova 2015
  - Sadowski/Scholz 2004
  - Catenacci 2020
  - Springham 2021
  - Klir 2011
  - Jednorog 2017
- Updated `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` so local sources are not
  framed as user-acquisition requests.
- Scientific status:
  - Closed: the queue no longer tells the user to acquire sources already
    verified locally.
  - Still open: local sources still need typed targets or verified digitized
    data for any missing observable before they close validation groups.
- Verification:
  - `python3 -m py_compile src/dpf/validation/source_acquisition.py tests/test_digitization.py`
  - focused source-queue/app tests passed (`3 passed in 1.38s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.87s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Figure Render Page Correction

- Rendered the parity-verified Akel PDF pages into the temporary workbench
  `/private/tmp/dpf_akel_digitization` using `pdftoppm`.
- Corrected digitization queue page hints:
  - Figs. 1-4 render on PDF page 3, not page 4.
  - Figs. 5-6 render on PDF page 5, not page 6.
- Page 4 is the typed table page, and page 6 is references, so the previous
  queue hints would have sent digitization to the wrong page renders.
- Scientific status:
  - Closed: the queue now points to the rendered pages that actually contain
    the cited Akel plots.
  - Still open: the temporary renders are not KR evidence, accepted packets, or
    digitized arrays.
- Verification:
  - `python3 -m py_compile src/dpf/validation/digitization.py tests/test_digitization.py`
  - `tests/test_digitization.py` passed (`10 passed in 0.76s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.36s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Scalar-Yield Uncertainty Diagnostics

- `pf1000_16kv_akel_table_candidate_evidence()` now reports:
  - neutron-yield absolute error
  - source-reported measured-yield uncertainty per row
  - measurement-uncertainty-normalized error per row
  - `max_measurement_uncertainty_normalized_error`
- Scientific status:
  - Closed: scalar-yield comparison now exposes the uncertainty scale printed
    in Akel Table 2.
  - Still open: this remains scalar table comparison. It does not provide a
    blind-prediction acceptance criterion, detector response, neutron timing,
    spectrum, or anisotropy closure.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - focused comparator/app tests passed (`3 passed in 0.94s`)
  - broader validation slice passed (`237 passed, 3 skipped in 13.30s`)
  - `git diff --check` is clean

### 2026-05-06: PF-1000 16 kV Candidate Scope Consistency

- Fixed Akel candidate evidence scope reporting in:
  - `pf1000_16kv_phase_candidate_evidence_from_history()`
  - `pf1000_16kv_derived_output_candidate_evidence()`
- Both now report `validation_scope="pf1000_16kv_2021_akel"` instead of the
  individual target ID.
- App-level PF-1000 16 kV phase and derived-output candidates now share the
  same Akel validation scope as the waveform and scalar-yield targets.
- Scientific status:
  - Closed: Akel candidate evidence has consistent scope identity for
    same-scope accounting.
  - Still open: phase and derived-output packets remain partial candidates
    because the KR record lacks complete measured axial, radial, and pinch
    phase endpoints.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py tests/test_mhd_physics_integration.py`
  - focused scope tests passed (`3 passed in 0.84s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.92s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Phase-Semantics Target

- Added `phase_semantics` to `pf1000_16kv_shot12581_phase_2021_akel`.
- The target now records that Akel's fitted Lee factors map to:
  - axial phase mass/current semantics: `fm`, `fc`
  - radial phase mass/current semantics: `fmr`, `fcr`
- Same-scope target reporting now marks `phase_semantics` present for
  `pf1000_16kv_2021_akel`.
- Scientific status:
  - Closed: the Akel 16 kV scope no longer has a false missing
    phase-semantics blocker.
  - Still open: phase timing remains partial because complete measured axial,
    radial, and pinch endpoint timings with uncertainty are not available.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - focused target/same-scope tests passed (`2 passed in 0.60s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.65s`)
  - `git diff --check` is clean

### 2026-05-06: Akel Table Uncertainty Target

- Added an explicit `uncertainty` block to
  `pf1000_16kv_shot_table_2021_akel`.
- The target now records:
  - measured neutron-yield uncertainty is available per row
  - row uncertainty range is `2.0e7` to `2.0e8` neutrons/shot
  - missing uncertainty components for waveform uncertainty, detector-response
    uncertainty, model-form uncertainty, input-parameter covariance, and a
    blind-prediction acceptance rule
- Same-scope reporting now marks `uncertainty` present but partial for
  `pf1000_16kv_2021_akel`.
- Scientific status:
  - Closed: Akel scalar yield uncertainty is now typed KR target data.
  - Still open: this is not a full uncertainty budget.
- Verification:
  - `python3 -m py_compile src/dpf/validation/kr_targets.py tests/test_kr_targets.py`
  - focused table/same-scope tests passed (`2 passed in 0.43s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.67s`)
  - `git diff --check` is clean

### 2026-05-06: MLX Collection Abort Hardening

- Issue:
  - The local MLX install/path is not a normal missing optional dependency.
  - `import mlx.core` aborts the Python interpreter, so pytest cannot catch it
    as `ImportError` and cannot safely apply `pytest.importorskip`.
  - Local metadata: `mlx==0.31.0`, Python `3.11.9`, and
    `macOS-26.3.1-arm64-arm-64bit`; the project safe detector reports
    `HAS_MLX=False` because the child-process import probe fails.
  - Initial collection abort path was `tests/test_amr_mlx.py` ->
    `dpf.metal.__init__` -> `dpf.metal.mlx_device` -> eager `mlx.core`
    import. Full collection then exposed the same eager-import hazard in
    `dpf.metal.mlx_kernels`.
- Fix:
  - `src/dpf/metal/mlx_device.py` now probes MLX in a child process and sets
    `HAS_MLX` from the child exit code.
  - `require_mlx()` imports MLX in-process only after the safe probe succeeds.
  - `src/dpf/metal/device.py`, `src/dpf/metal/mlx_amr.py`, and
    `src/dpf/metal/mlx_kernels.py` now use the safe detector path.
  - MLX tests and mixed CPU/MLX tests now use safe `HAS_MLX` gates, and
    `tests/conftest.py` guards legacy `pytest.importorskip("mlx.core")` calls.
- Scientific status:
  - Closed: broken MLX can no longer abort collection, so KR validation and
    CPU fallback tests remain runnable.
  - Still open: this does not validate MLX/Metal physics. In this environment
    MLX-specific tests skip until `mlx.core` imports cleanly.
- Verification:
  - targeted MLX/mixed tests passed or skipped cleanly
    (`15 passed, 28 skipped in 0.30s`)
  - full pytest collection completed without abort
    (`3657/3775 tests collected, 118 deselected in 3.11s`)
  - broader validation slice passed (`237 passed, 3 skipped in 12.97s`)
  - `git diff --check` is clean

### 2026-05-06: MLX Runtime Triage And Scientific Gate Correction

- Touched:
  - `tests/test_mlx_circuit_coupling.py`
  - `tests/test_mlx_pf1000.py`
  - `CodexFindings.md`
  - `CortexFindings.md`
- MLX collection issue:
  - `mlx==0.31.0` and `mlx-metal==0.31.0` are installed.
  - Outside the sandbox, `mlx.core` imports and reports `Device(gpu, 0)`.
  - Inside the sandbox, Metal enumeration returns no device, and `mlx.core`
    aborts natively during Metal device construction.
  - CPU/disable-Metal environment attempts did not prevent the import abort:
    `MLX_DEFAULT_DEVICE=cpu`, `MLX_DEVICE=cpu`, `MLX_DISABLE_METAL=1`, and
    `MLX_DISABLE_COMPILE=1`.
- Fix status:
  - Sandbox collection remains protected by the safe MLX child-process probe.
  - Real MLX validation is now explicitly a Metal-visible/outside-sandbox
    execution path.
- Scientific correction:
  - `test_btheta_increases_inward` now expects stronger `B_theta` at smaller
    radius.
  - Source-of-truth basis:
    `KnowledgeReference/plasma-formulary.md:2470-2473` gives
    `B_theta = mu I / (2*pi*r)`, and
    `KnowledgeReference/two-dimensional-simulation-of-dense-plasma-focus-5.md:78-84`
    gives the DPF boundary relation `Bphi = mu I / (2*pi*r)`.
- PF-1000 gate correction:
  - `TestMLXPF1000MustHave` and `TestMLXPF1000ShouldHave` are
    `xfail(run=False)`.
  - Reason: M6/full-discharge stability is still blocked by the documented
    CFL/full-duration issue. A run that stops before the required post-peak
    interval is not five-phase validation.
  - Source/project gate basis:
    `docs/SPRINT4_VALIDATION_REVIEW.md:105-113` and
    `docs/METAL_V2_DOD.md:330-337`.
- Remaining plan:
  - Close PF-1000 M6/CFL duration stability.
  - Re-enable the PF-1000 full-discharge MLX classes only after the fixture
    reaches the required post-peak/full-discharge duration without hitting the
    fixed step cap.
  - Keep fast config-level MLX checks runnable.
  - Continue same-scope KR-backed closure for spatial state, neutron timing,
    spectrum, anisotropy, detector response, uncertainty, and numerical
    convergence before any high-fidelity neutron-predictive claim.
- Verification:
  - Python syntax compilation passed for touched MLX hardening/test files.
  - `git diff --check` is clean.
  - sandbox collection passed:
    `4228/4346 tests collected, 118 deselected in 9.91s`
  - standing KR validation slice passed:
    `237 passed, 3 skipped in 13.40s`
  - outside-sandbox targeted MLX tests passed:
    `139 passed in 0.97s`
  - outside-sandbox full MLX glob passed with blocked PF-1000 gates as xfail:
    `553 passed, 19 xfailed in 50.43s`
  - `tests/test_mlx_pf1000.py`:
    `4 passed, 14 xfailed in 0.97s`

### 2026-05-06: PF-1000 MLX Probe Stability Through 10000 Steps

- Touched:
  - `src/dpf/metal/mlx_primitives.py`
  - `src/dpf/metal/mlx_state.py`
  - `src/dpf/metal/mlx_solver.py`
  - `tests/test_mlx_primitives.py`
  - `tests/test_mlx_state.py`
- Issue:
  - The interrupted 3000-step PF-1000 probe was rerun outside the sandbox.
  - The dense 1900-step probe passed, but the 2200-step probe exposed a
    deterministic `pressure` NaN at engine step `1985`, `t=0.690156 us`.
  - After pressure unpack hardening, the next run exposed the underlying
    conservative-state overflow as non-finite `B` at step `1986`.
- Fix:
  - Dual-energy pressure recovery now sanitizes the total-energy and entropy
    pressure candidates before blending, so an unused `inf` candidate cannot
    produce `NaN` through `0*inf`.
  - `MLXState.to_state_dict()` uses the same finite dual-energy blend for
    cylindrical and Cartesian unpacking.
  - The MLX solver's CPU-side post-hyperbolic floor now rebuilds momentum from
    bounded velocity after density flooring instead of multiplying momentum by
    `_rho_floor/rho` in vacuum cells.
  - CPU-side energy and vacuum `B_theta` prescription bookkeeping now uses
    finite float64 intermediates before returning to MLX float32.
- Operational MLX float32 rule:
  - For future MLX float32 nonfinite/overflow issues, first test the same
    pattern before adding narrower clamps: move CPU-side repair bookkeeping to
    float64, recover finite primitive-like quantities, rebuild conserved
    fields from bounded finite values, clip only for representability, and only
    then return to MLX float32.
  - Do not multiply conserved components by huge floor ratios in vacuum cells;
    rebuild from finite velocity, pressure/energy, and magnetic components
    instead.
- Verification:
  - Python syntax compilation passed for touched MLX files and tests.
  - Focused MLX regressions passed:
    `2 passed in 0.73s`.
  - Short PF-1000 MLX probe passed:
    `DPF_MLX_PROBE_STEPS=600`, `1 passed in 16.79s`.
  - Former failure-window PF-1000 MLX probe passed:
    `DPF_MLX_PROBE_STEPS=2200`, `1 passed in 48.60s`.
  - Original long PF-1000 MLX probe passed:
    `DPF_MLX_PROBE_STEPS=3000`, reaching `t=0.868989 us`,
    `I=0.671730 MA`, `max_B=1.164430`, `1 passed in 65.62s`.
  - Extended PF-1000 MLX probe passed:
    `PYTHONFAULTHANDLER=1 DPF_MLX_PROBE_STEPS=5000`, reaching
    `t=1.141062 us`, `I=0.872277 MA`, `max_B=1.512073`,
    `1 passed in 110.73s`.
  - Longer PF-1000 MLX probe passed:
    `PYTHONFAULTHANDLER=1 DPF_MLX_PROBE_STEPS=10000`, reaching
    `t=1.584247 us`, `I=1.187937 MA`, `max_B=2.059263`,
    `1 passed in 218.22s`.
- Late-window native-abort triage:
  - A first 20000-step probe attempt exited at native/process level with code
    `-1` before the first 2000-step checkpoint, with no Python faulthandler
    traceback and no probe assertion.
  - A dense 2000-step rerun passed with `DPF_MLX_PROBE_PRINT_INTERVAL=100`,
    reaching `t=0.747684 us`, `I=0.580770 MA`, and `max_B=1.006752`.
  - An exact 20000-step rerun then advanced past the historical stall and
    reached step 18000 (`t=2.075847 us`, `I=1.521106 MA`,
    `max_B=2.636804`) before another native/process-level `-1` exit before
    step 20000.
  - The Python nonfinite state checks did not fire in either native exit, so
    the current working hypothesis is late-window MLX/Metal runtime stability
    or cache/resource pressure rather than a caught Python-level NaN.
  - The probe now has optional MLX memory/cache controls:
    `DPF_MLX_PROBE_MEMORY=1` prints `mlx_active_MB`, `mlx_cache_MB`, and
    `mlx_peak_MB`; `DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL=N` calls
    `mlx.clear_cache()` every `N` completed steps.
  - A cache-clearing 20000-step run with
    `DPF_MLX_PROBE_CLEAR_CACHE_INTERVAL=1000` and
    `DPF_MLX_PROBE_MEMORY=1` reached step 12000 (`t=1.709192 us`,
    `I=1.274353 MA`, `max_B=2.209062`, `mlx_active_MB=0.288`,
    `mlx_cache_MB=10.770`, `mlx_peak_MB=9.801`) and then exited with native
    code `-1` before step 14000. Periodic `mlx.clear_cache()` did not remove
    the late-window native abort.
  - A bounded dense-window run with `DPF_MLX_PROBE_PRINT_START=12000` and
    `DPF_MLX_PROBE_PRINT_START_INTERVAL=25` exited natively after the first
    printed step, reinforcing that the current abort behavior is intermittent
    process/runtime stability and not a deterministic field value at a fixed
    step.
  - Fresh macOS crash reports were found under
    `~/Library/Logs/DiagnosticReports/Python-2026-05-06-*.ips`. The latest
    MLX-related report shows `SIGABRT`, `NSRangeException`,
    `-[__NSArray0 objectAtIndex:]: index 0 beyond bounds for empty array`, with
    the backtrace in `mlx::core::metal::Device::Device()`. A separate report
    shows `crashed on child side of fork pre-exec`. These reports support a
    native MLX/Metal/subprocess-environment issue around device discovery or
    process spawning; they do not identify a Python-level state NaN.
  - A fresh direct MLX initialization check after those reports still passed:
    `python3 -X faulthandler -c "import mlx.core as mx; print(mx.default_device())"`
    reported `Device(gpu, 0)`.
  - Added `scripts/run_mlx_pf1000_probe.py`, a standalone probe runner that
    bypasses pytest/conftest and sets `DPF_MLX_ASSUME_AVAILABLE=1` only after
    importing `mlx.core` directly in the Metal-visible process. This separates
    solver/runtime failures from pytest plugin, safe import-probe, and
    subprocess behavior.
  - Added `DPF_MLX_ASSUME_AVAILABLE=1` support in `mlx_device.py` as an
    explicit opt-in for already-validated Metal-visible processes. Do not use
    this in sandboxed collection because it bypasses the protective child
    import probe.
  - The standalone probe passed 2000 steps with memory telemetry, reaching
    `t=0.747684 us`, `I=0.580770 MA`, `max_B=1.006752`,
    `mlx_active_MB=0.288`, `mlx_cache_MB=10.525`, and
    `mlx_peak_MB=9.801`.
  - The standalone 20000-step cap run passed, reaching `t=2.200558 us`,
    `I=1.602652 MA`, `max_B=2.778162`, `mlx_active_MB=0.288`,
    `mlx_cache_MB=10.238`, and `mlx_peak_MB=9.801`. This indicates the prior
    `-1` exits were tied to the pytest/conftest/subprocess path or local
    process-spawn/device-discovery instability rather than a deterministic
    solver-state failure in the MLX MHD step.
  - Focused verification after standalone isolation:
    `python3 -m py_compile src/dpf/metal/mlx_device.py
    tests/test_mlx_pf1000_probe.py scripts/run_mlx_pf1000_probe.py` passed;
    `python3 -m pytest tests/test_mlx_device.py -q` passed (`21 passed`);
    `python3 -m pytest tests/test_mlx_primitives.py tests/test_mlx_state.py -q`
    passed (`61 passed`); `git diff --check` passed.
  - Target-time gate update:
    `tests/test_mlx_pf1000.py` now uses named PF-1000 cap/target controls
    instead of a hidden `range(20000)`: `DPF_MLX_PF1000_STEP_CAP` and
    `DPF_MLX_PF1000_TARGET_US`. The target is increase-only and cannot be set
    below the M6 `6 us` requirement. The fixture records target, cap, and
    cap-exhaustion metadata on the engine so M6 reports `step cap reached
    before target` explicitly.
  - Probe target-time update:
    both `tests/test_mlx_pf1000_probe.py` and
    `scripts/run_mlx_pf1000_probe.py` now accept `DPF_MLX_PROBE_TARGET_US`.
    The pytest probe asserts if the target is not reached within
    `DPF_MLX_PROBE_STEPS`; the standalone runner prints `CAP_EXHAUSTED` and
    returns exit code `2`.
  - Verification after target-time update:
    `python3 -m pytest tests/test_mlx_pf1000.py -q` passed with blocked gates
    preserved (`4 passed, 14 xfailed`); standalone target success smoke passed
    with `DPF_MLX_PROBE_TARGET_US=0.00005`; standalone cap-exhaustion smoke
    returned code `2` with `CAP_EXHAUSTED steps=5 target_us=1.000000
    final_t_us=0.243416`; focused MLX detector/pressure/state regressions
    passed (`82 passed`); `git diff --check` passed.
  - M6 target-time probe:
    standalone `DPF_MLX_PROBE_TARGET_US=6` with
    `DPF_MLX_PROBE_STEPS=80000` reached the M6 target and exited `PASSED`.
    Checkpoints: step 10000 `t=1.584247 us`, step 20000
    `t=2.200558 us`, step 30000 `t=2.812534 us`, step 40000
    `t=3.427694 us`, step 50000 `t=4.066377 us`, step 60000
    `t=4.711821 us`, and step 70000 `t=5.354566 us`. The pre-fix runner did
    not print the final target-hit step/time; the probe now prints final
    `PASSED steps=... final_t_us=...` and includes target-hit in the telemetry
    print condition.
- Remaining limit:
  - This closes the observed early PF-1000 MLX probe instability through the
    M6 `6 us` target on the standalone path, provided the cap is raised above
    the old 20000-step fixture limit. The blocked PF-1000 full-discharge
    classes should remain disabled because the current waveform is not
    accepted: by step 70000 the current was still rising at `3.215728 MA`,
    far above the M2 nominal upper band, and S2 current-dip behavior is not
    demonstrated.

### 2026-05-06: DoD Source-Of-Truth Audit

- Reviewed the Metal v2 DoD surface against the local KR source rule:
  `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, and
  `docs/SPRINT4_VALIDATION_REVIEW.md`, with matching comment cleanup in
  `tests/test_mlx_pf1000.py`.
- Added source-audit/superseded-status addenda instead of rewriting the
  historical March 2026 documents.
- Corrected PF-1000 scope discipline:
  - Akel 2021 16 kV (`pf1000_akel`) is the current same-scope MLX acceptance
    target.
  - Scholz/Gribkov full-energy PF-1000 remains a separate 27 kV/full-energy
    target packet and must not be mixed into Akel M2/S1/S2 gates.
- Corrected M2 target:
  - Akel shot 12581 uses `Ipeak = 1.165 MA +/- 10%`, i.e.
    `1.0485-1.2815 MA`.
  - The previous unspecified `1.2 MA` target and `1.87 MA` spec gate were
    marked/replaced as mixed-scope or stale for the Akel 16 kV gate.
- Corrected waveform/dip status:
  - Akel establishes measured current waveform figures and derivative/dip
    timing context.
  - NRMSE and dip-depth gates remain blocked until same-scope digitized current
    trace points and per-point uncertainty are accepted.
- Corrected mass and duration language:
  - M3 distinguishes closed-domain conservation from open-discharge
    outflow/density-floor accounting.
  - M6 `12 us` is documented as a conservative engineering endurance gate, not
    a direct measured Akel source value.
- Recorded the latest Akel MLX probe result:
  - Standalone `pf1000_akel` 40000-step run passed to `t = 3.238777 us`.
  - Peak current was `1.685154 MA`, above the Akel shot-12581 M2 upper bound
    `1.2815 MA`, and was still rising.
- Remaining limit:
  - The DoD is now more truthful about source scope, but full-discharge MLX
    scientific acceptance is still blocked by current-waveform mismatch,
    missing digitized Akel waveform evidence, and incomplete same-scope
    duration/dip closure.

### 2026-05-06: Akel Preset And Axial Pressure Coupling

- Corrected `pf1000_akel` from an average/nominal Akel 24-shot preset to the
  current same-scope shot-12581 preset.
- The preset now follows the typed KR shot target:
  `p0=1.2 Torr`, `rho0=2.583e-4 kg/m^3`, `C0=1332 uF`, `V0=16 kV`,
  `L0=25 nH`, `r0=6.1 mOhm`, `fm=0.17`, `fc=0.70`, `fmr=0.26`,
  and `fcr=0.75`.
- Added `tests/test_pf1000_akel_preset.py` to ratchet the preset against
  `pf1000_16kv_shot12581_phase_targets()`.
- Added `radial_current_fraction` support to the reduced MLX snowplow and
  forwarded it from `run_mlx_discharge()`, so reduced MLX Lee/RADPF runs can
  use the same Akel `fcr` scalar as the CPU snowplow.
- Fixed the remaining overshoot mechanism in the full `SimulationEngine`
  path: during axial rundown, `_dynamic_sheath_pressure()` now returns the
  configured cold fill pressure instead of feeding MHD total plasma pressure
  into the Lee/RADPF snowplow. The old path gave Akel shot 12581 about
  `640 Pa` axial back-pressure at step 1 instead of the source `160 Pa` cold
  fill, delaying rundown and under-loading the circuit.
- Probe telemetry now prints phase, voltage, sheath position, shock radius,
  `Lp`, `dL/dt`, plasma resistance, and sheath pressure.
- Evidence before the pressure fix but after the preset fix:
  standalone 40000-step `pf1000_akel` probe passed to `t=3.316852 us`, but
  current was still rising at `peak_I=1.367902 MA`, above the shot-12581 M2
  upper bound `1.2815 MA`.
- Evidence after the pressure fix:
  standalone 32000-step `pf1000_akel` probe passed to `t=2.971234 us` with
  `peak_I=0.977154 MA`.
  - step 10000: `t=1.389409 us`, `I=0.678638 MA`, `phase=rundown`,
    `Lp=2.762352 nH`, `sheath_p=160 Pa`
  - step 20000: `t=2.028429 us`, `I=0.844696 MA`, `Lp=5.431736 nH`
  - step 30000: `t=2.806466 us`, `I=0.961079 MA`, `Lp=9.382222 nH`
- Reduced reference check:
  `run_mlx_discharge(preset_name="pf1000_akel", mode="lee", max_steps=80000)`
  now peaks at `1.150685 MA` at `5.250577 us`, inside the Akel M2 band.
- Verification:
  `py_compile` passed for touched preset/coupling/probe/test files;
  `tests/test_mlx_snowplow.py tests/test_pf1000_akel_preset.py` passed
  (`6 passed`);
  `tests/test_snowplow_consolidated.py::TestDynamicPressureFallback` passed
  (`9 passed`);
  focused Akel KR target checks passed (`2 passed`);
  `tests/test_mlx_pf1000.py -q` remains `4 passed, 14 xfailed`.
- Remaining limit:
  the 32k/2.97 us run shows the full path is back on the reference trajectory.
- Post-fix M6 probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=6` and
  `DPF_MLX_PROBE_STEPS=90000` exited `PASSED`, reaching `t=6.000007 us`
  in `76948` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 40000: `t=3.576539 us`, `I=1.018345 MA`
  - step 50000: `t=4.283569 us`, `I=1.041204 MA`
  - step 60000: `t=4.927035 us`, `I=1.047142 MA`
  - step 70000: `t=5.567840 us`, `I=1.044211 MA`
- Updated remaining limit:
  standalone M6 `6 us` is now closed post-fix, but strict M2 is a low-side
  near miss (`1.047183 MA` vs lower bound `1.0485 MA`). Keep the PF-1000
  full-discharge gates blocked until M2 is confirmed inside the strict band,
  S1/S2 have accepted same-scope digitized waveform evidence, and the
  conservative 12 us engineering endurance gate is proven post-fix.
- Post-fix 8 us radial/pinch probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=8` and
  `DPF_MLX_PROBE_STEPS=130000` exited `PASSED`, reaching `t=8.000045 us`
  in `107566` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 90000: `t=6.809701 us`, `phase=radial`, `I=1.015650 MA`,
    `r=12.387635 cm`, `Lp=34.725876 nH`
  - step 100000: `t=7.435049 us`, `phase=radial`, `I=0.923846 MA`,
    `r=6.068274 cm`, `Lp=44.316989 nH`, `dLdt=26.719455 nH/us`
  - final step 107566: `phase=pinch`, `I=0.739814 MA`,
    `r=2.863039 cm`, `Lp=54.412990 nH`, `dLdt=-15.836659 nH/us`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.47 MB`, peak about `9.80 MB`).
- Next checks:
  the 8 us result closes the immediate radial-to-pinch stability question on
  the standalone path.
- Post-fix 12 us endurance probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=12` and
  `DPF_MLX_PROBE_STEPS=220000` exited `PASSED`, reaching `t=12.000000 us`
  in `160418` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 110000: `t=8.171744 us`, `phase=pinch`, `I=0.716376 MA`,
    `r=3.442282 cm`, `Lp=51.936660 nH`, `dLdt=-13.171781 nH/us`
  - step 120000: `t=9.000231 us`, `I=0.643373 MA`, `r=6.237254 cm`,
    `Lp=43.947850 nH`
  - step 140000: `t=10.458063 us`, `I=0.641996 MA`, `r=11.155373 cm`,
    `Lp=36.134086 nH`
  - step 160000: `t=11.965820 us`, `I=0.520224 MA`, `r=15.200000 cm`,
    `Lp=31.976097 nH`
  - final step 160418: `I=0.517539 MA`, `r=15.200000 cm`,
    `Lp=31.976097 nH`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.489 MB`, peak about `9.801 MB`).
- Updated remaining limit before source-scope cleanup:
  standalone `6 us`, `8 us`, and conservative `12 us` engineering endurance
  targets were closed post-fix through post-pinch expansion. Full-discharge
  acceptance remained blocked because strict M2 was still a low-side near miss
  (`1.047183 MA` vs lower bound `1.0485 MA`), S1/S2 still needed accepted
  same-scope digitized waveform evidence, and the fixed-time crowbar was not
  Akel shot-scope sourced.
- Late-voltage telemetry explanation:
  `V_kV=0.000000` after about `11.19 us` was explained by the inherited
  fixed-time crowbar previously present in `pf1000_akel` (`crowbar_enabled=True`,
  `crowbar_mode="fixed_time"`, `crowbar_time=10.5e-6`). The local Akel source
  search did not find shot-scope crowbar timing support, so post-10.5 us
  voltage/current behavior was engineering crowbar behavior,
  not same-scope Akel waveform evidence.
- Probe telemetry update:
  `scripts/run_mlx_pf1000_probe.py` and `tests/test_mlx_pf1000_probe.py` now
  print `crowbar` and `crowbar_t_us` fields. Verification: `py_compile` passed;
  5-step standalone Akel smoke passed and printed
  `crowbar=0 crowbar_t_us=-1.000000`.
- Next checks before source-scope cleanup:
  inspect the strict M2 low-side near miss without arbitrary tuning, review
  whether `pf1000_akel` should keep the unsourced fixed-time crowbar or move it
  behind an engineering preset/override, and continue same-scope Akel current
  trace digitization for S1/S2 waveform and dip acceptance.
- Final verification snapshot after 12 us/crowbar telemetry update:
  `git diff --check` clean; trailing-whitespace scan clean for touched
  notes/docs/probe files; targeted preset/PF-1000 gate slice passed
  (`5 passed, 14 xfailed in 1.72s`).

### 2026-05-06: Akel Source-Scoped Crowbar Cleanup

- Source audit result:
  the typed Akel shot-12581 target records circuit, geometry, Lee factors,
  waveform availability, and phase/dip context, but no crowbar enablement,
  crowbar time, crowbar resistance, or crowbar inductance. Local search in the
  Akel KR source found no shot-scope support for the inherited `10.5 us`
  fixed-time crowbar.
- Fix implemented:
  removed the unsourced inherited fixed-time crowbar from the source-scoped
  `pf1000_akel` preset. The preset now keeps `crowbar_enabled=False` and no
  longer carries `crowbar_time`, `crowbar_resistance`, or `crowbar_inductance`.
  `tests/test_pf1000_akel_preset.py` ratchets this source boundary.
- Probe telemetry retained:
  `scripts/run_mlx_pf1000_probe.py` and `tests/test_mlx_pf1000_probe.py` still
  print `crowbar` and `crowbar_t_us`, so engineering crowbar overrides remain
  visible in future logs.
- Verification after cleanup:
  `py_compile` passed for the touched preset/probe/test files; focused
  Akel/PF-1000 gate slice passed (`5 passed, 14 xfailed in 1.83s`); 5-step
  standalone Akel smoke passed with `crowbar=0 crowbar_t_us=-1.000000`.
- Source-scoped no-crowbar 12 us probe:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=12`,
  `DPF_MLX_PROBE_STEPS=220000`, and `DPF_MLX_PROBE_PRINT_INTERVAL=20000`
  exited `PASSED`, reaching `t=12.000000 us` in `161659` steps.
  - Final reported `peak_I=1.047183 MA` at `t=4.990339 us`.
  - step 120000: `t=9.000231 us`, `phase=pinch`, `I=0.643373 MA`,
    `V=10.293450 kV`, `r=6.237254 cm`, `Lp=43.947850 nH`
  - step 140000: `t=10.458063 us`, `I=0.641996 MA`, `V=9.600860 kV`,
    `r=11.155373 cm`
  - step 160000: `t=11.879750 us`, `I=0.704460 MA`, `V=8.881655 kV`,
    `r=15.200000 cm`
  - final step 161659: `I=0.707858 MA`, `V=8.817907 kV`, `crowbar=0`,
    `r=15.200000 cm`, `Lp=31.976097 nH`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.529 MB`, peak about `9.793 MB`).
- Updated remaining limit:
  standalone `6 us`, `8 us`, and conservative `12 us` source-scoped endurance
  are now closed without a crowbar. Scientific acceptance remains blocked
  because strict M2 is still a low-side near miss (`1.047183 MA` vs lower bound
  `1.0485 MA`) and S1/S2 still require accepted same-scope digitized Akel
  waveform evidence and uncertainty.
- Next checks:
  troubleshoot the M2 low-side near miss by comparing full-engine
  `Lp/dLdt/phase/current` against the reduced Lee path that peaks inside band;
  continue Akel current trace digitization to turn S1/S2 into source-backed
  gates.

### 2026-05-07: CPU Snowplow Lee Current-Factor Circuit Loading

- Source-of-truth basis:
  the Lee course describes the axial `fm`/`fc` equation of motion as coupled to
  a circuit equation, defines `fc` as the current fraction effectively flowing
  in/driving the axial moving structure, and states that radial `fmr`/`fcr`
  factors are incorporated in all three radial phases. The same course gives an
  axial dynamic-resistance example where `0.5*dL/dt` drops from about `5 mOhm`
  to `3.5 mOhm` when the current factor is considered. That supports
  current-factor scaling of circuit-facing `Lp`/`dLdt`.
- Issue isolated:
  the full `SimulationEngine` path used CPU `SnowplowModel`, whose magnetic
  force already used `(fc*I)^2`, but whose axial `plasma_inductance`, axial
  `dL_dt`, and frozen axial inductance used unscaled `L_coeff`. The reduced
  `MLXSnowplow` path already scaled axial circuit inductance by `fc` and radial
  circuit inductance/back-EMF by `fcr`, and it peaked inside the Akel M2 band.
  This explained why the full-engine path stayed just below strict M2.
- Fix implemented:
  `SnowplowModel` now keeps `L_coeff` as the unscaled coaxial geometry
  coefficient, but uses explicit circuit-facing helpers:
  - axial `fc * L_coeff * z`
  - radial `fcr_eff * (mu0/2pi) * z_f * ln(b/r)`
  - matching current-factor-scaled axial/radial/reflected/post-pinch `dL/dt`
  Tests now assert this convention while preserving the unscaled geometry
  coefficient checks.
- Verification:
  `py_compile` passed for `src/dpf/fluid/snowplow.py` and
  `tests/test_snowplow_consolidated.py`; focused snowplow formula slice passed
  (`35 passed in 1.30s`); full consolidated snowplow suite passed
  (`417 passed, 1 xfailed, 5 xpassed in 11.93s`); focused Akel/PF-1000 gate
  slice passed (`5 passed, 14 xfailed in 1.40s`).
- Source-scoped no-crowbar 6 us probe after the fix:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=6` exited `PASSED`,
  reaching `t=6.000050 us` in `75181` steps.
  - Final reported `peak_I=1.150507 MA` at `t=5.250198 us`, inside the Akel
    shot-12581 M2 band `1.0485-1.2815 MA`.
  - step 40000: `t=3.597952 us`, `I=1.103840 MA`, `Lp=10.142424 nH`
  - step 50000: `t=4.301412 us`, `I=1.137455 MA`, `Lp=13.357013 nH`
  - step 60000: `t=5.023197 us`, `I=1.149869 MA`, `Lp=16.792884 nH`
  - final step 75181: `I=1.144742 MA`, `V=11.965346 kV`, `crowbar=0`,
    `z=47.273209 cm`, `Lp=21.569092 nH`
- Source-scoped no-crowbar 8 us radial/pinch probe after the fix:
  standalone `pf1000_akel` with `DPF_MLX_PROBE_TARGET_US=8` exited `PASSED`,
  reaching `t=8.000071 us` in `105978` steps with the same final peak
  `1.150507 MA` at `t=5.250198 us`.
  - step 80000: `t=6.295662 us`, `phase=radial`, `I=1.136085 MA`,
    `r=14.036811 cm`, `Lp=23.220228 nH`, `dLdt=6.427329 nH/us`
  - step 90000: `t=6.949331 us`, `phase=radial`, `I=1.053240 MA`,
    `r=7.214705 cm`, `Lp=29.929093 nH`, `dLdt=17.453964 nH/us`
  - step 100000: `t=7.599487 us`, `phase=pinch`, `I=0.825187 MA`,
    `r=3.078976 cm`, `Lp=38.512458 nH`, `dLdt=-12.436875 nH/us`
  - final step 105978: `phase=pinch`, `I=0.767598 MA`, `V=10.479495 kV`,
    `r=4.600753 cm`, `Lp=34.464097 nH`, `dLdt=-8.323167 nH/us`
- Updated status:
  standalone source-scoped no-crowbar M2 is now closed for the `6 us` and
  `8 us` probes. The previous no-crowbar `12 us` endurance evidence was
  generated before the radial/reflected `fcr_eff` circuit-loading correction,
  so rerun the conservative `12 us` probe before claiming post-8us endurance
  is current. S1/S2 remain blocked until accepted same-scope digitized Akel
  waveform and dip evidence with uncertainty exist.
- Next checks:
  rerun the source-scoped no-crowbar `12 us` probe with the current
  circuit-loading fix; update the Metal v2 DoD/spec/review docs so they no
  longer call M2 a low-side near miss once the rerun evidence is complete; then
  continue Akel current-trace digitization for S1/S2.

### 2026-05-07: Current-Factor-Corrected 12 us Akel Probe And Doc Cleanup

- Rerun:
  standalone `pf1000_akel` no-crowbar probe with `DPF_MLX_PROBE_TARGET_US=12`,
  `DPF_MLX_PROBE_STEPS=220000`, `DPF_MLX_PROBE_PRINT_INTERVAL=20000`, and
  `DPF_MLX_PROBE_MEMORY=1` exited `PASSED`.
- Current-factor-corrected 12 us evidence:
  the run reached `t=12.000000 us` in `159912` steps.
  - Final reported `peak_I_MA=1.150507` at `peak_t_us=5.250198`, inside the
    Akel shot-12581 M2 band `1.0485-1.2815 MA`.
  - step 20000: `t=2.012285 us`, `I=0.880408 MA`, `phase=rundown`,
    `Lp=3.827323 nH`
  - step 40000: `t=3.597952 us`, `I=1.103840 MA`, `Lp=10.142424 nH`
  - step 60000: `t=5.023197 us`, `I=1.149869 MA`, `Lp=16.792884 nH`
  - step 80000: `t=6.295662 us`, `phase=radial`, `I=1.136085 MA`,
    `r=14.036811 cm`
  - step 100000: `t=7.599487 us`, `phase=pinch`, `I=0.825187 MA`,
    `r=3.078976 cm`
  - step 120000: `t=9.147131 us`, `I=0.709868 MA`, `r=8.958310 cm`
  - step 140000: `t=10.597788 us`, `I=0.763376 MA`, `r=14.469201 cm`
  - final step 159912: `I=0.811876 MA`, `V=8.228613 kV`, `crowbar=0`,
    `r=15.200000 cm`, `Lp=22.417737 nH`, `dLdt=0.000000 nH/us`
  - MLX memory telemetry stayed flat (`active=0.288 MB`, cache about
    `10.333 MB`, peak about `9.794 MB`).
- Documentation cleanup:
  `docs/METAL_V2_DOD.md`, `docs/METAL_V2_SPEC.md`, and
  `docs/SPRINT4_VALIDATION_REVIEW.md` now record that standalone
  source-scoped no-crowbar M2 and conservative M6 endurance are current after
  the Lee current-factor circuit-loading fix. They no longer call M2 a
  low-side near miss.
- Gate wording cleanup:
  `tests/test_mlx_pf1000.py` no longer says the long xfailed gate is blocked by
  M6/CFL duration stability. The remaining blocker is source closure for S1/S2:
  accepted same-scope digitized Akel current waveform and current-dip evidence
  with uncertainty.
- Next checks:
  continue Akel current-trace digitization for S1/S2, then decide whether to
  convert the long PF-1000 fixture from an `xfail(run=False)` scientific gate
  into an opt-in endurance/regression path with a large enough step cap.

### 2026-05-07: Akel Fig. 1 Extraction Status

- Progress made:
  promoted the local Akel 2021 Fig. 1 page-3 crop into
  `KnowledgeReference/figures/akel-2021-fig1-current-waveform-shot-12581.png`.
- Figure provenance:
  the crop was made from the parity-verified local Akel PDF page-3 render at
  300 dpi. Its SHA-256 is
  `4c574525f1de413e54cd02bd06aa35d549db700270281310a3809edc54ab255e`.
- OCR/axis check:
  the extracted panel preserves the `0-10 us` x-axis, `0-1400 kA` y-axis, and
  legend entries for measured `PF1000 D2 Meas. curr. kA 1.2 Torr shot 12581`
  and computed `PF1000 D2 comp. curr. kA 1.2 Torr`.
- Draft vector extraction route:
  the current `pdftocairo` page-3 SVG separates a measured-current candidate
  as filled black paths `1987-2280` (`294` compact path elements,
  approximately `0.02-9.98 us`) and a computed-current candidate as black
  stroke paths `1942-1975` (`34` path elements, approximately `0.01-10.01 us`).
  Filled black paths `2345-2411` are legend glyphs in the white legend box and
  must be excluded. This is extraction metadata only.
- Queue update:
  `scientific_closure_digitization_queue()` now reports
  `akel_2021_fig1_current_waveform_shot_12581` as
  `extracted_not_digitized`, with figure path/hash, candidate axis calibration
  points, and draft vector path-separation metadata. The other Akel figure
  tasks remain `not_extracted`.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py tests/test_digitization.py`
  passed; `python3 -m pytest tests/test_digitization.py -q` passed
  (`11 passed`);
  `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py::TestQualityAssessment::test_scientific_accuracy_gap_report_lists_remaining_work -q`
  passed (`12 passed`); the Fig. 1 PNG SHA-256 matched the queue hash;
  `git diff --check` passed; trailing-whitespace scan over touched text files
  found no matches.
- Scientific boundary:
  this does not close S1/S2. The extracted figure is a provenance artifact
  only; accepted waveform evidence still requires measured/computed series
  arrays, overlay residuals, and independent review through
  `digitization_verification_evidence()`.
- Next checks:
  export the separated measured/computed candidate paths into arrays, keep the
  packet draft until overlay residuals and independent review exist, then add a
  comparator that can report S1/S2 as blocked-by-review rather than
  blocked-by-missing-figure.

### 2026-05-07: Akel Fig. 1 Draft Arrays

- Progress made:
  exported the separated Fig. 1 measured/computed vector candidates into a
  draft packet at
  `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`.
- Draft packet provenance:
  SHA-256 is
  `0b8fae6147480392fcbe77eabeebc915a6a9561ec994daec32dea22859878017`.
  The loader `akel_fig1_draft_digitization_packet()` attaches the expected and
  actual packet hash and reports `draft_packet_hash_verified=True` when the
  artifact matches.
- Candidate arrays:
  measured current has `294` points from filled black paths `1987-2280`;
  computed current has `34` points from black stroke paths `1942-1975`.
  Legend glyphs `2345-2411` remain excluded (`67` filled path elements).
- Gate result:
  `digitization_verification_evidence(akel_fig1_draft_digitization_packet())`
  fails on exactly `independent_review_missing`,
  `overlay_residual_too_large`, and `review_status_not_accepted`. It does not
  fail on source, figure image, axis calibration, or required-series checks.
- Status update:
  `scientific_closure_digitization_status([akel_fig1_draft_digitization_packet()])`
  now reports `failed_task_count=1`, `open_task_count=5`, and
  `accepted_task_count=0`, so Fig. 1 is no longer reported as a missing packet.
  Gap reporting distinguishes this as a draft/failed packet needing review or
  correction.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py`
  passed; focused digitization/gap-report pytest slice passed (`16 passed`);
  full touched-file pytest slice passed (`68 passed`); `git diff --check`
  passed; trailing-whitespace scan over touched files found no matches; draft
  packet `shasum -a 256` matched
  `0b8fae6147480392fcbe77eabeebc915a6a9561ec994daec32dea22859878017`.
- Scientific boundary:
  this remains `draft_unreviewed` evidence only. S1/S2 stay blocked until
  overlay residuals are measured, independent review is completed, and the
  review status is accepted.

### 2026-05-07: Akel Fig. 1 Internal Overlay Residual

- Progress made:
  archived the page-3 SVG used for vector extraction at
  `KnowledgeReference/digitization/akel-2021-page3.svg` and measured an
  internal round-trip residual for the Fig. 1 draft arrays.
- Source SVG provenance:
  SVG SHA-256 is
  `b045c3b7033e50bd355e025ecf7c40d96edc1ffc7fcb6ef26832fe065fe99d3f`.
- Draft packet hash:
  adding overlay metadata changed
  `KnowledgeReference/digitization/akel-2021-fig1-current-waveform-shot-12581-draft-packet.json`
  to SHA-256
  `abe4a283ee154f84f6061da8ea508d3871faf3b14dddb2d1cfc8a7a0a5f8e0e7`.
- Overlay residual method:
  reprojected draft data arrays through the Fig. 1 axis calibration and
  compared them with transformed `pdftocairo` SVG path bounding-box centers.
  This is internal vector round-trip evidence, not an independent review.
- Overlay residual result:
  combined `328` candidate points had RMS residual `0.213455189 px` and max
  residual `2.733560259 px`. Computed-current RMS was `0.000027947 px` over
  `34` points; measured-current RMS was `0.225460245 px` over `294` points.
- Gate result:
  `digitization_verification_evidence(akel_fig1_draft_digitization_packet())`
  now fails only on `independent_review_missing` and
  `review_status_not_accepted`; `overlay_residual_too_large` is no longer
  present.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py`
  passed; `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py -q`
  passed (`68 passed`); `git diff --check` passed; trailing-whitespace scan
  over touched files found no matches; packet and SVG SHA-256 checks matched
  the values above.
- Scientific boundary:
  S1/S2 remain blocked. The packet is still `draft_unreviewed` until
  independent review accepts it.

### 2026-05-07: Akel Waveform Digitization Readiness Status

- Progress made:
  added `pf1000_16kv_current_waveform_digitization_candidate_evidence()` as a
  data-readiness helper for the Akel Fig. 1 waveform packet.
- Current helper result:
  with the local draft packet, the helper returns `passed=False`,
  `waveform_digitization_status="blocked_by_review"`, required series present,
  overlay RMS `0.213455189 px`, and missing checks
  `["independent_review_missing", "review_status_not_accepted"]`.
- Verification status:
  `python3 -m py_compile src/dpf/validation/digitization.py src/dpf/validation/quality_assessment.py src/dpf/validation/kr_targets.py src/dpf/validation/__init__.py tests/test_digitization.py tests/test_quality_assessment.py tests/test_kr_targets.py`
  passed; `python3 -m pytest tests/test_digitization.py tests/test_quality_assessment.py tests/test_kr_targets.py -q`
  passed (`146 passed`); `git diff --check` passed; trailing-whitespace scan
  over touched files found no matches.
- Status boundary:
  downstream code can now distinguish draft waveform data blocked by review
  from missing waveform data. This is not a simulation-vs-trace comparator and
  does not close S1/S2 or tier-1 waveform validation.

### 2026-05-09: Track A/B Change-Set Verification Consolidation

- Progress made:
  verified the current large Track A/B implementation set before taking on the
  next work-ready item. This is an engineering/productization checkpoint, not a
  scientific-evidence promotion.
- Verification status:
  `python3 ~/.codex/skills/srs-traceability/scripts/srs_trace_audit.py /Users/anthonyzamora/dpf-unified`
  passed and reported `48` unique requirement IDs across the SRS baseline and
  findings docs; `npm --prefix gui run typecheck` passed; `git diff --check`
  passed; and the focused Python regression suite
  `python3 -m pytest tests/test_digitization.py tests/test_kr_targets.py tests/test_source_acquisition.py tests/test_uncertainty_budget.py tests/test_quality_assessment.py tests/test_validation_artifacts.py tests/test_mhd_numerical_fidelity.py tests/test_physics_fidelity.py tests/test_circuit_field_coupling.py tests/test_mhd_physics_integration.py tests/test_memory_preflight.py tests/test_backend_capabilities.py tests/test_cli_backend_options.py tests/test_export_scope.py tests/test_local_first_security.py tests/test_project_lifecycle.py tests/test_server_metadata.py tests/test_server_readiness.py tests/test_airgap_gate.py -q`
  passed (`323 passed, 3 skipped`).
- Current boundary:
  this confirms the current guardrail/productization implementation remains
  internally consistent. It does not unblock Akel S1/S2, Tier 2, Tier 4, Tier
  5, predictive readiness, or high-fidelity readiness, all of which still need
  accepted same-scope evidence.
- Next work-ready item:
  continue with SRS traceability/Doorstop import preparation, then the remaining
  productization follow-ons that do not require new science sources: UI/API
  units schema, lifecycle API exposure, export provenance completion, and
  air-gap release artifacts.

### 2026-05-09: SRS Traceability Matrix Export Staged

- Progress made:
  advanced the Doorstop/traceability path without requiring a network install
  in the active shell. Doorstop remains the planned requirements-management
  tool, but the candidate baseline now has import-ready JSON/CSV traceability
  artifacts that can be reviewed and later imported.
- Modules/docs/tests/artifacts touched:
  `scripts/export_srs_traceability.py`,
  `tests/test_srs_traceability_export.py`,
  `docs/SRS_TRACEABILITY_MATRIX.json`,
  `docs/SRS_TRACEABILITY_MATRIX.csv`,
  `docs/DPF_REQUIREMENTS_BASELINE.md`,
  `docs/SRS_TRACEABILITY_TOOLING.md`, and
  `docs/DPF_UNIFIED_SRS_DRAFT.md`.
- Current artifact status:
  `scripts/export_srs_traceability.py` parses
  `docs/DPF_REQUIREMENTS_BASELINE.md`, validates duplicate IDs, known
  statuses/priorities, P0/P1 verification methods, and implemented-row
  evidence, then writes `48` requirements into staged RTM JSON/CSV exports.
- Verification status:
  `python3 scripts/export_srs_traceability.py` exported `48` requirements;
  `python3 -m py_compile scripts/export_srs_traceability.py tests/test_srs_traceability_export.py`
  passed; `python3 -m pytest tests/test_srs_traceability_export.py -q` passed
  (`2 passed`); `python3 ~/.codex/skills/srs-traceability/scripts/srs_trace_audit.py /Users/anthonyzamora/dpf-unified`
  still reports `48` unique requirement IDs; and `git diff --check` passed for
  the traceability files.
- Remaining boundary:
  this is an RTM staging step, not formal Doorstop validation. A real Doorstop
  tree still requires the optional `dpf-unified[traceability]` dependency in
  the active environment and review acceptance of the candidate baseline.
- Post-change regression status:
  the combined focused suite including `tests/test_srs_traceability_export.py`
  passed (`325 passed, 3 skipped`); `npm --prefix gui run typecheck` passed;
  the SRS trace audit still reports `48` unique requirement IDs; and
  `git diff --check` passed.

### 2026-05-09: Project Lifecycle API Surface

- Progress made:
  exposed the existing local project lifecycle helpers through bounded FastAPI
  endpoints and GUI wire client types. The API uses `DPF_PROJECTS_ROOT`
  (`./projects` by default) as the local project boundary, so create/load/
  duplicate/archive requests cannot write arbitrary paths outside the configured
  project root.
- Modules/docs/tests touched:
  `src/dpf/server/app.py`, `src/dpf/server/models.py`,
  `gui/src/renderer/api/client.ts`, `gui/src/renderer/api/types.ts`,
  `tests/test_server_projects.py`, `docs/DPF_REQUIREMENTS_BASELINE.md`,
  `docs/DPF_UNIFIED_SRS_DRAFT.md`, and regenerated
  `docs/SRS_TRACEABILITY_MATRIX.json` / `docs/SRS_TRACEABILITY_MATRIX.csv`.
- API surface added:
  `GET /api/projects/root`, `POST /api/projects`,
  `POST /api/projects/load`, `POST /api/projects/duplicate`, and
  `POST /api/projects/archive`.
- Verification status:
  `python3 -m py_compile src/dpf/server/app.py src/dpf/server/models.py tests/test_server_projects.py`
  passed; `python3 -m pytest tests/test_server_projects.py tests/test_project_lifecycle.py tests/test_server_readiness.py tests/test_server_metadata.py tests/test_local_first_security.py -q`
  passed (`18 passed`); `npm --prefix gui run typecheck` passed;
  `python3 scripts/export_srs_traceability.py` regenerated the staged RTM with
  `48` requirements; and `git diff --check` passed for the touched API/client
  files.
- Remaining boundary:
  this exposes lifecycle operations but does not add a full GUI project browser
  or user workflow view. It also does not change scientific readiness.

### 2026-05-09: Doorstop Installed And Verified

- Progress made:
  installed the repository traceability extra with
  `python3 -m pip install -e '.[traceability]'`.
- Installed tool status:
  `doorstop --version` reports `Doorstop v3.1`, and `doorstop --help` shows the
  expected create/import/export/publish commands. The package does not provide
  a `python3 -m doorstop` entrypoint, so the console script is the supported
  invocation.
- Verification status:
  `python3 -m pytest tests/test_srs_traceability_export.py tests/test_server_projects.py tests/test_project_lifecycle.py -q`
  passed (`9 passed`); the SRS trace audit still reports `48` unique
  requirement IDs; and `git diff --check` passed.
- Environment caveat:
  `python3 -m pip check` still reports global Python environment conflicts,
  including `letta` requiring `typer<0.10.0` while the active environment now
  has `typer 0.25.1`, plus several unrelated pre-existing dependency conflicts.
  Use a dedicated virtual environment before treating a release or air-gap
  build as clean.
- Remaining boundary:
  Doorstop is installed, but no Doorstop requirements tree has been initialized
  yet. The current guardrail still says to review/accept the candidate baseline
  and staged RTM before importing it into Doorstop.

### 2026-05-09: Extended Local Source Search Outside KnowledgeReference

- Progress made:
  searched likely local machine source pools outside `KnowledgeReference/` for
  the currently needed paper and textbook/method references, then captured the
  result in `docs/LOCAL_SOURCE_SEARCH_2026_05_09.md`.
- Search coverage:
  checked top-level Downloads, the two OneDrive paper drops, GPT paper
  downloads, DPF-U2 paper and converted-text pools, old project paper archives,
  `downloaded_books_papers`, and the Claude memory-stage DPF paper text store.
  Build/cache trees, `node_modules`, `.web`, `.next`, `.git`, and
  `KnowledgeReference/` were excluded.
- Result:
  no exact local copy was found for the six blocking/secondary paper
  acquisitions: Klir 2011, Sadowski/Scholz/PF-1000 2004, Catenacci 2020,
  Springham 2021, Jednorog 2017, or Cikhardtova 2015. Klir appeared only as a
  citation in a 2026 hybrid X-pinch paper, not as the target source.
- Method-source candidates found:
  LeVeque 2002 appears as a 580-page local PDF outside KR with SHA-256
  `b3adec0d3616dbde57a5522cfce1861890887d7c03a2232d2136cb94c9bac1d5`;
  Toro 2009 appears only as a 47-page reading sample/excerpt with SHA-256
  `78144939eadb0f7382c222f49a9a11ce9bae3e19c4f866b94e4aa6de1f39d73f`; and
  Rybicki-Lightman appears only as a 63-page partial/frontmatter candidate with
  SHA-256 `fcff04d2c6c1c77855192cd107ad144497cc7637706a66278658af1a5f23a08d`.
- Boundary:
  these local method files are not source-of-truth evidence yet. They need KR
  ingestion/review before they can support method authority, verification, or
  readiness claims.

### 2026-05-09: Local Method Source Candidate Review

- Progress made:
  reviewed the local method-source candidates from the extended source search
  and added `docs/LOCAL_METHOD_SOURCE_REVIEW_2026_05_09.md`.
- LeVeque review result:
  `archive_reference_OLD/references/papers/textbooks/leveque-2002-finite-volume-hyperbolic.pdf`
  is a likely full local candidate: metadata title is `Finite Volume Methods
  for Hyperbolic Problems`, author is `RANDALL J.LEVEQUE`, page count is `580`,
  and SHA-256 is
  `b3adec0d3616dbde57a5522cfce1861890887d7c03a2232d2136cb94c9bac1d5`.
  It covers finite-volume conservation laws, CFL, Godunov/Riemann methods,
  high-resolution/TVD methods, convergence, source terms, nonlinear systems,
  Euler equations, shock tubes, multidimensional finite-volume methods, and
  quadrilateral grids.
- Toro review result:
  `toro-2009-riemann-solvers-excerpt.pdf` is only a 47-page reading sample. It
  can support terminology checks but cannot close Riemann-solver method
  authority; at that checkpoint, the full Toro source was still marked needed.
  Superseded 2026-05-11: the full Toro source has now been promoted to KR and
  chunked for readable Markdown review; Toro remains a method-target extraction
  task, not an acquisition blocker.
- Rybicki-Lightman review result:
  `rybicki-lightman-1979-radiative-processes.pdf` is only a 63-page partial
  candidate with frontmatter and Chapter 1 radiative-transfer material. It does
  not include the full radiation-process coverage needed for bremsstrahlung or
  radiation-loss closure.
- Boundary:
  LeVeque is review-ready for KR method-source ingestion, but no method-source
  authority has been promoted yet. At that checkpoint, Toro and
  Rybicki-Lightman were treated as acquisition blockers for their broader
  method areas.
  Superseded 2026-05-11: LeVeque and full Toro have now been promoted to KR;
  Rybicki-Lightman remains only a partial local candidate.

### 2026-05-09: LeVeque Method Source Promotion

- Progress made:
  converted the local LeVeque 2002 PDF into paired `KnowledgeReference/`
  records:
  `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md` and
  `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.json`.
- Provenance:
  original local PDF path is
  `archive_reference_OLD/references/papers/textbooks/leveque-2002-finite-volume-hyperbolic.pdf`;
  original PDF SHA-256 is
  `b3adec0d3616dbde57a5522cfce1861890887d7c03a2232d2136cb94c9bac1d5`.
- Validation:
  `scripts/validate_kr_schema.py` passed for the promoted JSON record, and
  `scripts/verify_kr_pdf_parity.py` passed across all `580` pages with no JSON
  page-text mismatches and no Markdown missing pages.
- Source-list update:
  `docs/SOURCE_ACQUISITION_NEEDED.md` now moves LeVeque out of the needed
  acquisition list and into the promoted local method-source section.
- Mapping update:
  code references that previously pointed to the old archive location now point
  to `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md`.
  The structured MHD numerical-fidelity audit now uses the promoted LeVeque KR
  record for the generic `finite_volume_mhd_verification` method authority,
  while Beresnyak remains the DPF-specific source for cylindrical, circuit, and
  phase-scope evidence.
- Boundary:
  LeVeque is now usable as local KR method authority for finite-volume and
  hyperbolic-conservation-law numerical verification only. It is not DPF
  experimental evidence and does not close PF-1000 same-scope Tier 4 spatial
  validation, Tier 5 neutron validation, or predictive scientific readiness.

### 2026-05-09: Compact Restart Handoff And Remaining Work Summary

- Current source-of-truth state:
  `CortexFindings.md` and `CodexFindings.md` are synchronized through the
  LeVeque promotion. Scientific claims remain limited to local
  `KnowledgeReference/` artifacts. The promoted LeVeque 2002 records exist at
  `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md` and
  `.json`; `KnowledgeReference/` is git-ignored, so those local files do not
  appear in ordinary `git status`.
- Track A coding/simulation state:
  the code-ready guardrails, reporting surfaces, fail-closed blockers, and
  Tier-3 numerical verification packet are complete for the current plan.
  Remaining Track A work is not more blocker plumbing; it is source evidence,
  accepted review metadata, same-scope target packets, and validation/UQ values.
- Track A scientific blockers still open:
  Akel Fig. 1 needs a real independent accepted review; Akel Figs. 2-6 still
  need verified digitization/review; S1/S2 waveform/current-dip validation
  remains `blocked_by_review`; Tier 2 phase, Tier 4 spatial, Tier 5 neutron,
  field-coupling, physics-fidelity, and UQ acceptance remain blocked until
  same-scope KR-backed targets and uncertainty values exist.
- Source acquisition still open:
  the exact local search did not find Klir 2011, Sadowski/Scholz/PF-1000 2004,
  Catenacci 2020, Springham 2021, Jednorog 2017, or Cikhardtova 2015. Method
  sources still needing full acquisition include Hutchinson diagnostics, full
  Toro, Freidberg or Goedbloed depending on the next MHD scope, Birdsall and
  Langdon for PIC/kinetic work, Griem for spectroscopy, and full
  Rybicki-Lightman for radiation-process closure. LeVeque no longer needs
  acquisition.
- Track B product/SRS state:
  the candidate SRS baseline, staged RTM JSON/CSV, compute-authority labels,
  result classification, run manifests, validation certificates, memory
  preflight/telemetry, unsupported-backend diagnostics, project lifecycle API,
  units metadata, local-first controls, export v1 scope, and air-gap gate
  definition are implemented or staged. Doorstop is installed and usable, but
  no Doorstop requirements tree has been initialized/imported yet.
- Track B remaining work:
  review and accept the candidate requirements baseline, initialize/import a
  Doorstop tree, build the full GUI project browser/workflow if required,
  decide audit-log depth, propagate classification metadata into any remaining
  non-manifest exports, produce license-reviewed wheelhouse/SHA256/offline
  smoke logs for the air-gap gate, and clean dependency conflicts in a dedicated
  release virtual environment.
- Recommended restart order after compaction:
  1. Re-anchor with `dpf_skill_preflight.py`, `git status --short`, and the
     tail of both findings docs.
  2. Pick one lane: evidence lane (`Akel Fig. 1 review` or `Figs. 2-6
     digitization`), requirements lane (`Doorstop import`), source-acquisition
     lane, or release lane (`air-gap artifacts / clean venv`).
  3. Preserve all scientific blockers until same-scope KR evidence and review
     metadata pass their gates.

### 2026-05-09: Web And Google Scholar Source-Acquisition Review

- Scope:
  reviewed the six open paper acquisitions in
  `docs/SOURCE_ACQUISITION_NEEDED.md` through Google Scholar title-query URLs
  and reachable scholarly pages. Direct automated access to
  `scholar.google.com` returned HTTP 403, so the durable record now stores
  reproducible Scholar queries plus publisher, institutional, repository, and
  scholarly-index pages instead of scraped citation counts.
- Existing blockers:
  Klir 2011, Sadowski/Scholz/PF-1000 2004, Catenacci 2020, Springham 2021,
  Jednorog 2017, and Cikhardtova 2015 remain acquisition targets only. Their
  blocker status is unchanged until exact files are acquired, hashed, reviewed
  into `KnowledgeReference/`, and mapped into typed targets or digitization
  packets.
- Fastest apparent paper acquisitions:
  Jednorog 2017 and Cikhardtova 2015 have open Sciendo/Nukleonika routes and
  should be the quickest paper-PDF intake candidates. Klir 2011, Sadowski 2004,
  Catenacci 2020, and Springham 2021 appear to require publisher/licensed or
  author/institutional access.
- New candidate acquisitions added:
  Rezac et al. 2026 silver activation counter, Rezac/Klir/Kubes/Kravarik 2012
  TOF reconstruction, Klir/Kubes/PF-1000 2012 thermonuclear-neutron search,
  Krauz et al. 2012 PF-1000 plasma-current sheath structure, Kubes/Klir/PF-1000
  2013 pinch-evolution scenario, Kortanek/Kubes/PF-1000 2014 current-flow and
  energy-balance paper, Scholz et al. 2012 IPPLM MJ plasma-focus progress,
  Auluck et al. 2021 DPF review, and Bernard et al. 1998 DPF status review.
- Guardrail:
  ResearchGate/Academia-style pages were treated only as discovery leads.
  CTU FEE, IPPLM, PNNL, OSTI, IAEA/INIS, PubMed, Sciendo, ScienceDirect, MDPI,
  Nukleonika, J-GLOBAL, and ICDMP pages were recorded as verified metadata or
  acquisition leads, not scientific evidence.

### 2026-05-09: Physics-Gap-Driven Source Search

- Scope:
  reran source discovery from the remaining validation physics instead of from
  the existing acquisition titles alone. The search targeted the missing
  physics behind Akel S1/S2, Tier 2 phase timing, Tier 4 density/field/temperature
  spatial closure, Tier 5 neutron timing/spectrum/anisotropy/detector closure,
  circuit-field energy coupling, and physics-fidelity/model-form limits.
- Highest-value new or re-ranked acquisition leads:
  Zielinska/Paduch/Scholz 2011 sixteen-frame interferometer
  (`10.1002/ctpp.201000047`), Kubes et al. 2009 interferometric pinch/neutron
  timing (`10.1109/TPS.2009.2030576`), Kubes et al. 2012 magnetic-probe/neutron/
  interferometry correlation (`10.1088/0741-3335/54/10/105023`), Krauz et al.
  2012 current-sheath structure (`10.1088/0741-3335/54/2/025010`), Mitrofanov
  et al. 2014 fine current-sheath/magnetic-field structure
  (`10.1134/S1063780X14070071`), and Malir et al. 2022 implosion dynamics
  (`10.1063/5.0098124`).
- Tier 5 neutron additions:
  Krasa et al. 2008 vessel-caused DD neutron anisotropy
  (`10.1088/0741-3335/50/12/125006`), Jednorog et al. 2015 radioindium radial
  asymmetry (`10.1007/s10967-014-3444-z`), Klir et al. 2011 thermonuclear-neutron
  evidence (`10.1063/1.3555447`), and Kubes et al. 2009 deuteron energy
  distribution from neutron diagnostics were added as neutron mechanism,
  anisotropy, activation, and spectrum/TOF leads.
- Tier 4 temperature and model-form additions:
  Jakubowska et al. 2011 optical emission spectroscopy public Nukleonika PDF,
  Skladnik-Sadowska et al. 2011 optical spectroscopy in PF-1000
  (`10.1002/ctpp.201000046`), Stepniewski 2004 PF-1000 MHD modelling
  (`10.1016/j.vacuum.2004.05.019`), Schmidt et al. 2014 fully kinetic MJ DPF
  (`10.1063/1.4897192`), Munzar et al. 2021 azimuthal B-field mapping
  (`10.1063/5.0040515`), and Lee/Saw/Akel/Kubes/Paduch 2016 radiative-cooling
  limits (`10.1109/TPS.2015.2497269`) were added or re-ranked.
- Local extraction leads:
  the search also flagged already-local KR records that should be mined before
  acquiring adjacent material: PF-1000 pinch-column evolution/fast particle
  acceleration, DPF-1000U optical spectra with gas puffing, and Malir 2024
  interferometry-vs-MHD. These are not automatic same-scope closures; each needs
  extraction, scope matching, and uncertainty review.
- Status:
  no validation blocker changed state. `docs/SOURCE_ACQUISITION_NEEDED.md` now
  records this as a physics-gap-ranked acquisition and local-review shortlist.

### 2026-05-09: Module-Coverage Source Search

- Scope:
  ran another acquisition search from code surfaces that were not fully covered
  by the PF-1000/Track A physics-gap queue. The pass covered atomic/CR rates,
  line radiation, electrode ablation, anomalous resistivity, scaling-law
  diagnostics, p-B11 reactivity, Thomson and X-ray synthetic diagnostics,
  instability/shear diagnostics, CIV/Paschen breakdown, PIC/hybrid kinetics,
  Bohm/sheath support, Sedov verification, Athena/AthenaK backend wrappers, and
  AI/surrogate provenance.
- Highest-priority module gaps:
  `src/dpf/radiation/line_radiation.py` still needs Post/ADAS/CHIANTI-compatible
  cooling tables before it can move beyond `empirical_cooling_estimate`;
  `src/dpf/atomic/ionization.py` needs exact Lotz/Seaton/Burgess/NIST evidence
  before stronger CR/impurity-charge-state claims; `src/dpf/experimental/pic`
  needs Nanbu/Perez plus DPF kinetic validation papers; and
  `src/dpf/diagnostics/pb11_yield.py` needs p-B11 reactivity/cross-section
  tables before feasibility claims.
- Additional module guardrails:
  ablation, anomalous resistivity, CIV breakdown, scaling laws, X-ray images,
  instability/shear margins, Athena/AthenaK comparison, and ML/surrogate outputs
  remain method or diagnostic scaffolds until exact source records, dataset
  hashes, validity ranges, and tests/certificates are added.
- Search leads added:
  Lotz 1967/ApJS ionization, Post 1977 cooling, ADAS/Summers, CHIANTI,
  Puetterich 2019, Seaton/Burgess recombination, NIST ASD, Buneman 1959,
  Davidson/Gladd LHDI, Nevins/Swain and Sikora/Weller p-B11, Salpeter/Hutchinson
  /Sheffield scattering, Danielsson/Brenning CIV, Nanbu/Perez collisions,
  Schmidt DPF kinetic simulations, Shumlak-Hartman shear stabilization, Taylor
  1950, and Athena/Athena++/AthenaK method papers.
- Status:
  no scientific blocker changed state. This pass updates
  `docs/SOURCE_ACQUISITION_NEEDED.md` so future work does not accidentally
  promote scaffolded modules without local KR evidence and module-specific
  verification.

### 2026-05-09: WALRUS / MHD Training Data Review

- Scope:
  searched external WALRUS, The Well, and public MHD dataset sources, then
  audited local WALRUS/DPF training artifacts under `docs/`, `training_data/`,
  `models/`, and the WALRUS integration code.
- New review artifact:
  added `docs/WALRUS_MHD_TRAINING_DATA_REVIEW_2026_05_09.md` and linked it
  from `docs/SOURCE_ACQUISITION_NEEDED.md`.
- External leads added:
  WALRUS arXiv/GitHub/model-card sources, The Well NeurIPS 2024 paper and docs,
  The Well `MHD_64`/`MHD_256` pages, the CATS astrophysical turbulence paper
  (`10.3847/1538-4357/abc484`), and NASA/ASME/FDA credibility-method leads.
- Local data assessment:
  tracked `docs/walrus_training_*.json` files are Lee-model current/yield
  waveform sweeps, not volumetric MHD and not experimental validation. Ignored
  HDF5 training sets under `training_data/` are not defensible as-is: the audit
  found missing manifests, missing energy-conservation fields, non-finite
  circuit scalars, suspicious float32-limit field values, metadata/geometry
  mismatches, and sampled all-zero magnetic fields.
- Decision:
  the current WALRUS/DPF data can support pipeline development, schema tests,
  negative tests, and exploratory ML only. It cannot support scientific
  validation, high-fidelity readiness, or publication claims unless regenerated
  with strict validation, manifests, accepted source-backed solver evidence,
  and clear context-of-use limits.

### 2026-05-09: Module-By-Module Suspect-Code Audit Notes

- Scope:
  performed the requested module-by-module audit as notes only. No source code
  or `KnowledgeReference/` files were edited. All non-`KnowledgeReference/`
  code, tests, comments, docs, generated data, and training artifacts were
  treated as suspect until backed by local reviewed evidence.
- New audit packet:
  added `docs/MODULE_AUDIT/INDEX.md`, `docs/MODULE_AUDIT/BACKLOG.md`, and one
  module note each for validation, engine/core, Metal/MLX, circuit/snowplow,
  diagnostics, radiation/atomic/neutrons, IO/export, AI/WALRUS, and
  server/GUI/CLI.
- Module findings:
  validation has good blocker-preserving guardrails but still mixes authority,
  target extraction, calibration, diagnostics, and verification roles; engine/
  app surfaces can hide failures through fallback or mismatched feature labels;
  MLX and circuit/snowplow contain useful engineering paths but need source-
  status labels around coupling, floor, radius, and current-factor assumptions;
  diagnostics and radiation contain several source-backed pieces but many
  synthetic or empirical outputs remain non-authoritative; IO/export and
  AI/WALRUS are useful scaffolds but not validation evidence; and server/GUI/CLI
  surfaces need tighter backend, unit, readiness, and claim-label consistency.
- Backlog update:
  `docs/MODULE_AUDIT/BACKLOG.md` now contains module-specific task IDs for all
  nine modules. These are advisory future-work entries, not implementation
  authority, and each still requires current-code review, source review, and
  task sizing before work begins.
- Verification:
  `git diff --check -- docs/MODULE_AUDIT` passed, and the module-audit folder
  has no pending markers or trailing whitespace.

### 2026-05-09: Engine/Core MHD Wrapper Guardrails

- Scope:
  started engine/core implementation work while validation remains blocked on
  independent review. This closes the first two code-ready items from
  `docs/MODULE_AUDIT/BACKLOG.md`: `ENG-001` and `ENG-002`.
- Changes made:
  `app_engine.run_mhd_simulation_core()` now honors the requested `n_steps` by
  calling `SimulationEngine.run(max_steps=...)`; validates `n_steps`; exposes
  `requested_max_steps` and `terminated_by_max_steps`; and no longer silently
  falls back to Lee-only output after a full-engine failure.
- Explicit fallback path:
  callers that intentionally want Lee fallback must pass
  `allow_engine_fallback=True`. Those fallback results are labeled with
  `engine_status="failed"`, `engine_fallback="lee"`,
  `engine_fallback_allowed=True`, and the engine error type/message.
- Tests:
  added `tests/test_app_engine_core_guardrails.py` for max-step forwarding,
  invalid step rejection, default fail-visible failure, and explicit fallback
  metadata. Focused test result:
  `python3 -m pytest tests/test_app_engine_core_guardrails.py -q` passed
  (`4 passed`). `git diff --check` also passed for the touched files.
- Boundary:
  this improves auditability and prevents an MHD failure from masquerading as a
  successful engine result. It does not promote any validation evidence or close
  scientific blockers.
- Follow-on `ENG-007` closure:
  separated backend implementation maturity from validation status by adding
  `backend_authority_labels()` and attaching backend authority metadata to
  `SimulationEngine.run()` summaries. `engine_tier` remains backward-compatible,
  but the new summary fields mark backend tier as `not_validation_evidence`
  unless real readiness artifacts say otherwise.
- Additional verification:
  extended `tests/test_backend_capabilities.py`; focused run
  `python3 -m pytest tests/test_backend_capabilities.py
  tests/test_app_engine_core_guardrails.py -q` passed (`9 passed`). Compile and
  `git diff --check` checks also passed for the touched engine/app/test files.
- `ENG-004` closure:
  since breakdown is not yet wired into the engine initial-state path, the run
  summary now reports `breakdown_authority` with
  `status="config_only_not_applied"`, `applied_to_initial_state=False`, and
  `validation_status="not_validation_evidence"`. The focused engine/core suite
  now passes with `10 passed`.
- `ENG-006` closure:
  `src/dpf/constants.py` now states that these are standards-scoped
  implementation constants, not KR-scoped scientific validation inputs. The
  code adds `CONSTANTS_SCOPE` and `CONSTANTS_AUTHORITY`, derives `m_d` from
  SciPy's deuteron-mass constant, and adds direct constants authority tests.
- `ENG-008` closure:
  state sanitation now preserves first-failure evidence before repair and
  exposes it through `SimulationEngine.nonfinite_state_evidence` and run
  summaries. Probe/audit runs can set `fail_fast_on_nonfinite=True` with
  `nan_check_stride=1` to fail before repair while retaining the first event.
- Probe wiring:
  the opt-in PF-1000 MLX pytest probe and standalone probe script now use the
  built-in fail-fast sanitation path instead of monkeypatching `_sanitize_state`.
- Verification:
  focused run `python3 -m pytest tests/test_constants_authority.py
  tests/test_backend_capabilities.py tests/test_app_engine_core_guardrails.py -q`
  passed (`15 passed`). Compile checks passed for touched constants, config,
  engine, app, test, and probe files.

### 2026-05-09: Metal/MLX Engineering Guardrails

- Scope:
  continued with simulation-side engineering work after engine/core. Closed the
  unblocked MLX backlog items `MLX-001`, `MLX-002`, `MLX-003`, `MLX-005`,
  `MLX-006`, `MLX-007`, and `MLX-008`. Source-dependent item `MLX-004`
  remains blocked.
- Coupling authority cleanup:
  MLX coupling comments and docstrings no longer describe voltage-flux or
  Poynting-voltage coupling as "correct" or "first-principles" authority.
  `coupling_method_authority()` now labels density-weighted `Lp`,
  voltage-flux, and Poynting-voltage paths as
  `validation_status="not_validation_evidence"` with
  `can_support_scientific_claims=False`; `run_mlx_discharge()` returns
  `mhd_coupling_authority`, and claim-guard tests pin those labels.
- No-density-injection guardrail:
  `_apply_floors()` no longer raises density using the old `B^2/va_max^2`
  vacuum floor. New tests cover the direct helper and zero-`dt` RK2/RK3 paths
  so the full timestepper cannot add fake mass through floor logic alone.
- Coupling cleanup:
  removed the dead radial-coordinate expression in `compute_upf_voltage_flux()`.
  This does not source-close the voltage-flux method; it only removes stale code.
- Probe policy:
  added a non-slow policy test for the standalone PF-1000 MLX probe so it stays
  classified as engineering endurance regression, not scientific acceptance.
- MLX result metadata:
  `run_mlx_discharge()` now emits `back_emf_V`, `back_emf_authority`, and
  `phase_model_authority`. These fields make clear that separate motional
  back-EMF is not applied and that pure MLX snowplow output is reduced
  axial/radial/pinch coverage, not full Lee five-phase coverage.
- MHD coupling gate:
  `evaluate_mhd_coupling_gate()` now requires phase eligibility,
  finite/positive/comparable `Lp`, finite `dLdt`, and finite/nonnegative
  resistance before MHD-derived coupling can enter the engineering blend.
  `run_mlx_discharge()` emits `mhd_coupling_gate`, which remains
  `not_validation_evidence` and cannot support scientific claims without
  same-scope validation packets.
- Verification:
  MLX timestepper/boris slice passed (`24 passed`), focused discharge metadata
  tests passed (`2 passed`), probe policy tests passed (`3 passed`), the
  combined MLX claim/gate guard and discharge authority slice passed (`7
  passed`), and compile checks passed for the touched MLX code/tests.

### 2026-05-09: Circuit/Snowplow Engineering Guardrails

- Scope:
  closed the unblocked circuit/snowplow items `CIR-001`, `CIR-002`, `CIR-003`,
  `CIR-004`, `CIR-006`, `CIR-007`, and `CIR-008`. The remaining item stays
  blocked on Akel waveform review.
- CircuitCoupler authority:
  density-weighted MHD feedback is now labeled as engineering scaffolding in
  `src/dpf/circuit/coupler.py`. `circuit_coupler_authority()` and
  `CircuitCoupler.authority` report `validation_status="not_validation_evidence"`
  and `can_support_scientific_claims=False`, and engine summaries include the
  same authority record.
- Current-factor boundary:
  `src/dpf/fluid/snowplow.py` now documents `L_coeff` as the unscaled geometric
  coefficient while circuit-facing helpers apply `f_c` and `f_cr_eff`. Tests now
  assert that `L_coeff` does not vary with `current_fraction`.
- Radius convention boundary:
  CPU and reduced-MLX snowplows now expose `radius_convention` metadata. CPU
  radial loading is labeled as shock-front-radius `r_s` with PF-1000/0.14-0.17
  `r_min` scope. MLX radial loading is labeled as piston-radius `r_p` with a
  reduced deuterium gross `0.13a` termination and no full Lee five-phase
  coverage. The records explicitly reject cross-backend equivalence as
  validation evidence.
- Post-pinch resistance provenance:
  CPU post-pinch resistance multipliers now expose
  `post_pinch_resistance_authority`, labeling them as empirical engineering
  continuity knobs with missing multiplier source provenance and
  `validation_status="not_validation_evidence"`.
- Auto-coupler trust gate:
  `_should_use_coupler()` no longer treats any positive density as enough for
  auto MHD circuit loading. It requires a resolved MHD signal such as nonzero
  `B`, nonzero velocity, or dynamic density. Explicit `density_weighted` remains
  caller-controlled.
- Summary metadata:
  `SimulationEngine.run()` now includes `circuit_coupler_trust_status` so runs
  can report whether auto coupling was trusted and why. The trust-status record
  is still explicitly non-validation evidence.
- Verification:
  focused circuit/snowplow tests passed (`15 passed`), and compile checks passed
  for the touched circuit, engine, snowplow, config, and test files. The focused
  CircuitCoupler authority slice passed (`4 passed`), and focused CPU/MLX
  radius-convention tests passed (`3 passed`). The post-pinch resistance
  authority test passed (`1 passed`).

### 2026-05-09: Diagnostics Engineering Guardrails

- Scope:
  closed the unblocked diagnostics items `DIA-001`, `DIA-002`, and `DIA-007`.
  Source-dependent diagnostics items remain blocked until same-scope local KR
  evidence, source-status manifests, and validation packets exist.
- BeamTracker yield guardrail:
  `BeamTracker.get_result()` now uses the beam-target helper's voltage-equivalent
  contract instead of passing kinetic energy in joules as `V_pinch`. The result
  exposes `equivalent_V_pinch`, `yield_status`, `yield_model_role`, and
  `yield_warning`.
- BeamTracker authority:
  the yield path is explicitly labeled `engineering_estimate_not_validation`.
  Helper failures are surfaced as `yield_status="failed"` with the exception
  type/message, rather than being silently swallowed.
- HDF5 divergence guardrail:
  exported `max_div_B` is now labeled as
  `rough_array_metric_not_physical_divergence` with `T/cell` units and
  `validation_status="not_validation_evidence"`. This preserves compatibility
  while preventing the scalar from being mistaken for geometry-aware divergence
  evidence.
- Stale diagnostics notes:
  `src/dpf/diagnostics/Troubleshooting.md` now has a current audit preface, and
  `docs/MODULE_AUDIT/diagnostics.md` records the new BeamTracker/HDF5 status.
- Verification:
  BeamTracker tests passed (`10 passed`) and export-scope/HDF5 tests passed
  (`4 passed`) before the combined diagnostics verification.

### 2026-05-10: Diagnostics Evidence Manifest Guardrail

- Scope:
  closed `DIA-005`. The diagnostics package now has a single fail-closed
  evidence manifest for every diagnostics module/public symbol.
- Manifest behavior:
  `src/dpf/diagnostics/evidence_manifest.py` classifies diagnostics outputs as
  `blocked-by-review`, `missing`, `engineering-probe`, or `synthetic-only`.
  It intentionally contains no accepted validation entries.
- Guardrails:
  every manifest entry reports `validation_status="not_validation_evidence"`
  and `can_support_validation_claims=False`. Same-scope KR/source closure,
  detector response, uncertainty, and independent review remain prerequisites
  for future validation promotion.
- Tests:
  `tests/test_diagnostics_evidence_manifest.py` parses the diagnostics source
  tree and requires every public class/function in each diagnostics module to
  appear in the manifest.
- Verification:
  manifest compile checks passed, and the focused diagnostics evidence manifest
  test passed (`4 passed`). The combined BeamTracker/export-scope/manifest
  diagnostics slice also passed (`22 passed`).
- Remaining diagnostics work:
  `DIA-003`, `DIA-004`, and `DIA-008` remain open for local formula/source
  closure, anisotropy/beam-target assumptions, and same-scope diagnostic
  validation packets.

### 2026-05-10: Diagnostics Test-Lane Guardrail

- Scope:
  closed `DIA-006`. Diagnostics tests now have an explicit non-validation lane
  manifest and collection-time pytest markers.
- Test-lane behavior:
  `src/dpf/diagnostics/test_lanes.py` classifies diagnostics-oriented test
  files as `engineering-smoke`, `source-component-check`, `source-blocked`, or
  `synthetic-only`. No diagnostics test is currently classified as
  `source-backed-validation`.
- Pytest wiring:
  `tests/conftest.py` applies diagnostics markers during collection, and
  `pyproject.toml` registers the marker names, including a reserved future
  `diagnostics_validation` marker.
- Guardrails:
  source-component tests such as DD reactivity checks remain separate from
  total DPF neutron validation. Synthetic diagnostics tests remain separate
  from detector-response validation.
- Verification:
  manifest compile checks passed, and the focused diagnostics test-lane test
  passed (`5 passed`). The combined diagnostics manifest/test-lane/BeamTracker/
  export slice also passed (`27 passed`).
- Remaining diagnostics work:
  `DIA-003`, `DIA-004`, and `DIA-008` remain open for local formula/source
  closure, anisotropy/beam-target assumption review, and same-scope diagnostic
  validation packets.

### 2026-05-10: Preset Value Authority Guardrail

- Scope:
  closed `ENG-005`. Preset source-scope labeling now extends from the preset
  summary level down to every runtime config leaf.
- Authority behavior:
  `src/dpf/presets.py` exposes `preset_value_authority()` and
  `preset_authority_manifest()`. These produce one fail-closed authority record
  per preset value path.
- Guardrails:
  every preset value record reports `validation_status="not_validation_evidence"`
  and `can_support_validation_claims=False`. Broad PF-1000, derived 20 kV,
  tutorial/custom/demo, and other narrative/empirical presets remain scaffolds
  until exact KR line references or accepted source packets are added.
- Product surface:
  `list_presets()` now carries compact value-source labels for API/UI display,
  while `get_preset()` still strips `_meta` and returns only simulation config.
- Verification:
  preset compile checks passed, and the focused preset source-scope test passed
  (`7 passed`). The combined diagnostics/preset guardrail slice also passed
  (`34 passed`).

### 2026-05-09: Radiation/Atomic/Neutrons Metadata Guardrails

- Scope:
  closed `RAD-006` and `RAD-008`. The source-dependent radiation/neutron tasks
  remain blocked by missing local tables, same-scope neutron validation packets,
  p-B11/QMF source closure, and ionization/ablation provenance work.
- Conservative metadata:
  line-radiation metadata now carries `source_status`,
  `validation_status="not_validation_evidence"`, and
  `claim_scope="engineering_cooling_estimate"`.
- QMF label:
  QMF suppression now exposes `qmf_model_metadata()` and labels the suppression
  formula as `free_free_suppression_source_missing` and
  `unverified_not_design_evidence`.
- CPU/MLX parity:
  MLX line-radiation provenance wording now matches the CPU surface:
  unknown-provenance empirical fits, not direct CHIANTI/ADAS/Post source tables.
- Verification:
  focused radiation metadata tests passed (`4 passed`) before the combined
  radiation verification.

### 2026-05-10: QMF Diagnostic-Only Quarantine

- Scope:
  closed `RAD-005` by quarantine. No QMF derivation/source packet was added or
  implied.
- Output authority:
  `QMFDiag` now carries the same fail-closed status as `qmf_model_metadata()`:
  heuristic diagnostic role, missing free-free suppression source, not
  validation evidence, and no validation/design-claim support.
- Guardrails:
  QMF suppression outputs can remain useful as regime diagnostics, but they
  cannot support p-B11 feasibility, high-field radiation, or design claims until
  a primary local source packet is acquired and reviewed.
- Verification:
  QMF/radiation metadata compile checks passed, and the focused QMF/radiation
  metadata tests passed (`17 passed`). The combined diagnostics/preset/QMF
  guardrail slice also passed (`51 passed`).

### 2026-05-09: IO/Export Well Guardrails

- Scope:
  closed `IO-001`, `IO-006`, `IO-007`, and `IO-008`. The remaining IO/export
  tasks remain blocked by local Well-schema source review, strict validator
  work, deferred bridge classification propagation, and training-data
  quarantine/regeneration decisions.
- Well flush:
  `SimulationEngine.run()` now flushes Well output on normal completion and
  attempts the same after run errors. A short-run regression confirms a Well file
  is emitted without manual `engine.close()`.
- Circuit scalars:
  the buffered `src/dpf/io/well_exporter.py` adapter now forwards circuit
  scalars to the full AI Well exporter, and engine/Athena export calls provide
  current, voltage, circuit energies, and total circuit energy.
- Grid metadata:
  the full AI Well exporter now writes cylindrical root `grid_type` as
  `cylindrical` instead of always writing `cartesian`.
- Scope/SRS sync:
  the export scope, SRS draft, and candidate requirements baseline now state
  that accepted HDF5/Well paths carry fail-closed classification/provenance
  labels; deferred external bridges still need their own classification
  propagation before acceptance.

### 2026-05-09: Source-Truth Verification Boundary

- Answer:
  no current work should be read as verification of all modules against
  `KnowledgeReference/`. The closed items are engineering, metadata, lifecycle,
  or claim-boundary guardrails unless an accepted source packet says otherwise.
- Current matrix:
  `docs/MODULE_AUDIT/INDEX.md` now records module-level source-truth status.
  All modules remain blocked, partial, engineering-guarded, or product/export
  guarded; none is globally source-verified.
- Practical effect:
  future work must continue to treat older formulas, generated data, broad
  tests, and provenance comments as suspect until local KR evidence and
  same-scope validation packets promote them.

### 2026-05-09: AI/WALRUS Guardrails

- Scope:
  closed `AI-003`, `AI-004`, `AI-006`, and `AI-008` from
  `docs/MODULE_AUDIT/BACKLOG.md`. The remaining AI/WALRUS blockers are source
  acquisition/review for WALRUS/The Well/CATS, checkpoint/license provenance,
  local-data quarantine, and real formatter/checkpoint verification.
- Strict dataset validation:
  `DatasetValidator(strict=True)` now checks scalar finite values, required
  energy/time datasets, monotonic time, geometry/root consistency,
  provenance/classification attrs, non-finite sanitation labels, saturation
  thresholds, and all-zero magnetic fields.
- Export metadata:
  the full AI Well exporter now labels preview/source status and records
  non-finite sanitation counts at dataset/root level, so sanitized output cannot
  masquerade as validation evidence.
- Model reporting:
  `DPFSurrogate` and the AI status API now distinguish placeholder, real model,
  and source-backed model states. `source_backed_model_loaded` remains false
  until a reviewed source packet records checkpoint hash/version/license/source
  and accepted validation scope.
- Stale-doc cleanup:
  AI/WALRUS audit notes, AI troubleshooting notes, and the WALRUS data generator
  script now describe JSON exploratory candidates and non-validation status
  instead of Well HDF5 or identity-placeholder behavior.
- Verification:
  focused AI/WALRUS pytest slices passed. Treat those passes as implementation
  guardrails only; they do not validate WALRUS physics against
  `KnowledgeReference/`.

### 2026-05-09: Server/GUI/CLI Time Display Guardrail

- Scope:
  closed `SGC-002`.
- Fix:
  `TopBar` now formats simulation time from seconds, matching API/store units,
  and displays ns/us/ms/s based on magnitude. This removes the prior
  seconds-as-nanoseconds display bug.
- Verification:
  `npm run typecheck` passed in `gui/`.
- Boundary:
  this is a renderer display guardrail only, not a physics-validation change.

### 2026-05-09: Server/GUI/CLI Version Display Guardrail

- Scope:
  closed `SGC-007`.
- Fix:
  the renderer TopBar version label now uses a Vite-injected value from
  `gui/package.json`, replacing the stale hardcoded `v1.0.0` display.
- Verification:
  `npm run typecheck` and `npm run build:renderer` passed in `gui/`. Vite still
  reports non-fatal chunk-size and Node module-type warnings.
- Boundary:
  this is product-label hygiene only, not source-truth verification.

### 2026-05-09: Server/GUI/CLI Local-First Renderer Guardrail

- Scope:
  closed `SGC-006`.
- Fix:
  renderer HTML no longer loads remote Google font assets. The renderer CSP now
  permits self/local style/script/font sources and localhost/127.0.0.1 API and
  WebSocket connections only.
- Audit coverage:
  local-first security audit now includes `DPF-SEC-005` for non-local renderer
  HTTP asset references.
- Verification:
  local-first security tests passed (`8 passed`), Python compile checks passed,
  `npm run typecheck` passed in `gui/`, and `npm run build:renderer` passed with
  non-fatal Vite/Node warnings.
- Boundary:
  this is a local-first product/security guardrail only, not source-backed
  scientific validation.

### 2026-05-09: Server/GUI/CLI Validation Authority Display

- Scope:
  closed `SGC-003` for the CLI validation path.
- Fix:
  `dpf validate` now shows source-authority status and blocker count alongside
  the peak-current PASS/FAIR/POOR grade, plus a note that those grades are
  engineering comparisons until accepted KR/same-scope source gates promote the
  result.
- Verification:
  CLI backend/validation tests passed (`4 passed`) and Python compile checks
  passed.
- Boundary:
  no validation result was promoted to Reference; this only prevents the CLI
  from presenting peak-current grade as source-backed validation.

### 2026-05-09: Server/GUI/CLI Backend Contract Alignment

- Scope:
  closed `SGC-001`.
- Fix:
  backend names now align across server health, CLI `simulate`, CLI
  `export-well`, renderer/Electron status types, TopBar badges, and the backend
  selector for `mlx` and `hybrid`.
- Verification:
  backend contract tests passed (`6 passed`), Python compile checks passed,
  `npm run typecheck` passed in `gui/`, and `npm run build:renderer` passed with
  non-fatal Vite/Node warnings.
- Boundary:
  availability/status wiring is not validation authority. `mlx` and `hybrid`
  remain subject to the same source-truth and readiness gates as other backends.

### 2026-05-09: Server/GUI/CLI Gradio Claim Hygiene

- Scope:
  closed `SGC-005`.
- Fix:
  legacy Gradio copy now labels backends and high-resolution output as
  Preview/source-gated instead of using validated, publication-grade, WORKING,
  or 97x-demonstrated language. Backend availability is presented as product
  readiness only.
- Validation markdown:
  Gradio validation output now reports an engineering comparison and states
  that Reference validation requires accepted local `KnowledgeReference/`
  evidence plus same-scope validation packets.
- Verification:
  Gradio claim-hygiene tests passed and reject the old overclaim phrases.
- Follow-on regression cleanup:
  the focused combined suite exposed stale slow-test assumptions rather than a
  source-backed validation issue. The hybrid linked-Athena test now uses a
  minimal valid cylindrical/PLM config, and real WALRUS tests now skip unless
  `dpf.ai.HAS_WALRUS` is true.
- Combined verification:
  the focused Server/GUI/CLI + AI/WALRUS suite passed (`494 passed`, `12
  skipped`, `1 xfailed`). Skips remain dependency/source availability states,
  not validation success.
- Boundary:
  this is claim-boundary enforcement, not physics validation. At this checkpoint
  the remaining Server/GUI/CLI work was readiness-scope semantics (`SGC-004`)
  and PF-1000 source-scope labeling (`SGC-008`).

### 2026-05-09: Server/GUI/CLI Readiness Scope Metadata

- Scope:
  closed `SGC-004`.
- Fix:
  API readiness now reports explicit `readiness_scope` metadata so Akel Fig. 1
  digitization blockers are not implied to be per-run blockers for undeclared or
  unrelated tutorial runs. Same-scope runs can still show the blocker as
  applying to the run.
- Propagation:
  simulation managers store an optional declared validation scope; REST creation
  reads a raw `validation_scope` or the source-scoped `pf1000_akel` preset; the
  renderer blocker badge uses the scope note as tooltip context.
- Verification:
  server readiness tests cover undeclared/global source queue behavior and
  declared PF-1000 Akel same-scope behavior. The combined focused
  Server/GUI/CLI + AI/WALRUS suite passed (`500 passed`, `12 skipped`, `1
  xfailed`).
- Boundary:
  this adds scope clarity only. Akel evidence remains blocked pending accepted
  independent review. At this checkpoint, broader PF-1000 preset source-scope
  labeling was still tracked separately as `SGC-008`.

### 2026-05-09: Server/GUI/CLI PF-1000 Preset Source-Scope Labels

- Scope:
  closed `SGC-008`.
- Fix:
  preset listings and the REST preset endpoint now carry explicit
  source-scope labels. The broad PF-1000 engineering preset, Akel shot-12581
  preset, and 20 kV derived trend preset are separated at the product/API
  metadata layer.
- UI/API behavior:
  renderer preset selection can show the source-scope status and note, while
  `get_preset()` continues to return only simulation config values.
- Verification:
  preset source-scope and server readiness tests passed (`10 passed`), and GUI
  typecheck passed.
- Boundary:
  no PF-1000 preset was promoted to validation evidence. The labels prevent
  scope confusion while preserving the remaining source-closure blockers.

### 2026-05-09: Validation Calibration Provenance Labels

- Scope:
  closed `VAL-007` for active calibration outputs.
- Fix:
  added a shared calibration provenance helper and attached its metadata to
  Lee/MLX calibration result dictionaries. Calibration fits now report
  `optimized_parameter_fit`, `Calibration Fit`, `not_validation_evidence`, and
  `can_support_validation_claims=false`.
- UI behavior:
  calibration markdown now says optimized fits are not validation evidence and
  that Reference validation still requires accepted local `KnowledgeReference/`
  evidence plus same-scope validation packets.
- Verification:
  calibration provenance tests passed (`3 passed`).
- Boundary:
  this is claim hygiene for fitted parameters, not source closure for the device
  registry or reconstructed waveforms.

### 2026-05-09: IO/Export Well Artifact Classification Propagation

- Scope:
  partially closed `IO-004` for the Well HDF5 and CLI export path.
- Fix:
  Well HDF5 exports now carry fail-closed artifact classification metadata
  (`artifact_classification`, `artifact_distribution`,
  `artifact_classification_json`, and `dpf_artifact_classification_json`) while
  retaining `validation_status="not_validation_evidence"` and Preview labels.
  `dpf export-well` now exposes owner/classification/distribution/handling-note
  flags, and engine-flushed Well artifacts keep training-data-interchange status
  instead of generic run `not_evaluated`.
- Plan status:
  `docs/MODULE_AUDIT/BACKLOG.md` now marks `IO-004` as partial. Remaining work
  is config/API-level classification propagation and dataset-manifest linkage;
  Well/WALRUS/The Well source authority is still blocked by local source review.
- Verification:
  `python3 -m py_compile src/dpf/ai/well_exporter.py
  src/dpf/io/well_exporter.py src/dpf/engine/core.py src/dpf/cli/main.py
  tests/test_export_scope.py tests/test_cli_backend_options.py` passed;
  `python3 -m pytest tests/test_export_scope.py tests/test_cli_backend_options.py
  -q` passed (`15 passed`); targeted WALRUS metadata/strict-validator tests
  passed (`3 passed`); and the engine manifest regression passed (`1 passed`).
- Boundary:
  this does not validate Well compatibility, WALRUS data, or physics outputs. It
  prevents exported artifacts from losing governance metadata while remaining
  non-validation evidence.

### 2026-05-09: IO/Export Config-Driven Artifact Classification

- Scope:
  extended the `IO-004` partial closure into the normal engine configuration
  path.
- Fix:
  `SimulationConfig.diagnostics` now has artifact owner, classification,
  distribution, and handling-note fields. `build_run_manifest()` extracts those
  fields by default, and `SimulationEngine` uses the same metadata for HDF5
  governance attributes, run manifests, and engine-flushed Well output.
- Plan status:
  `IO-004` remains partial, not complete. The closed slice is config-driven
  propagation for engine HDF5/Well/run-manifest artifacts; the remaining slices
  are batch-generated Well trajectories, dataset manifests, checkpoint HDF5
  labeling, and certificate/readiness context.
- Verification:
  `python3 -m py_compile src/dpf/config.py src/dpf/validation/artifacts.py
  src/dpf/engine/core.py tests/test_validation_artifacts.py` passed, and
  `python3 -m pytest tests/test_validation_artifacts.py tests/test_export_scope.py
  tests/test_cli_backend_options.py -q` passed (`32 passed`).
- Boundary:
  this is export governance metadata only. It does not promote any artifact to
  Reference, and it does not source-close The Well/WALRUS compatibility.

### 2026-05-09: IO/Export Batch Well Classification Propagation

- Scope:
  extended `IO-004` partial closure to batch-generated Well trajectories.
- Fix:
  `BatchRunner.run_single()` now uses config-derived artifact classification
  metadata when constructing `WellExporter`, so sweep-generated training
  artifacts inherit owner, classification, distribution, and handling-note
  labels.
- Plan status:
  batch Well propagation is closed. `IO-004` remains partial because dataset
  manifests and certificate readiness/context propagation still need separate
  implementation.
- Verification:
  `python3 -m py_compile src/dpf/ai/batch_runner.py
  tests/test_walrus_consolidated.py` passed, and the focused
  BatchRunner/export/artifact pytest slice passed (`27 passed`).
- Boundary:
  batch artifacts remain training-data interchange and non-validation evidence.

### 2026-05-09: IO/Export Checkpoint Artifact Classification

- Scope:
  extended `IO-004` partial closure to checkpoint/restart HDF5 artifacts.
- Fix:
  checkpoint files now carry fail-closed artifact role, Preview/non-validation
  labels, source-authority text, and config-derived artifact classification
  metadata. This closes the unclassified checkpoint HDF5 surface identified in
  the IO/export propagation audit.
- Plan status:
  checkpoint labeling is closed. `IO-004` remains partial because dataset
  manifests and certificate readiness/context propagation still need separate
  implementation.
- Verification:
  `python3 -m py_compile src/dpf/diagnostics/checkpoint.py
  tests/test_infrastructure_consolidated.py` passed, and the focused checkpoint
  artifact pytest slice passed (`6 passed`).
- Boundary:
  checkpoint metadata is governance labeling only, not scientific restart
  validation.

### 2026-05-09: IO/Export Dataset Manifest And API Classification Closure

- Scope:
  closed `IO-004` for artifact-classification propagation.
- Fix:
  Batch runs now write a fail-closed `dataset_manifest.json` with artifact
  classification, config hash, parameter ranges, output hashes, counts, and a
  training-candidate guardrail. The REST create-simulation path preserves
  artifact classification fields supplied in the config payload, completing the
  config/CLI/API propagation path for HDF5, Well, manifests, checkpoint HDF5,
  and dataset manifests.
- Plan status:
  `IO-004` is marked complete in `docs/MODULE_AUDIT/BACKLOG.md`. Certificate
  readiness/context propagation remains under `VAL-010`.
- Verification:
  `python3 -m py_compile src/dpf/ai/batch_runner.py
  tests/test_walrus_consolidated.py tests/test_server_readiness.py` passed,
  focused dataset/API tests passed (`4 passed`), and the broader
  BatchRunner/export/artifact slice passed (`31 passed`).
- Boundary:
  dataset manifests are provenance/guardrail artifacts. They do not promote
  WALRUS/The Well data to validation evidence.

### 2026-05-09: Validation Certificate Readiness Context

- Scope:
  partially closed `VAL-010` for validation certificate artifacts.
- Fix:
  certificates now carry result classification, artifact classification,
  readiness summaries, and blocker lists. Accepted certificates fail closed when
  blockers are present or a supplied result classification cannot support
  validation claims.
- Plan status:
  `VAL-010` is now partial in `docs/MODULE_AUDIT/BACKLOG.md`. Certificate
  context is closed; embedded HDF5 readiness-summary propagation remains open.
- Verification:
  `python3 -m py_compile src/dpf/validation/artifacts.py
  tests/test_validation_artifacts.py` passed, and
  `python3 -m pytest tests/test_validation_artifacts.py -q` passed (`19
  passed`).
- Boundary:
  this is artifact governance. No validation evidence was promoted.

### 2026-05-09: Validation HDF5 Readiness Metadata

- Scope:
  completed `VAL-010` propagation guardrails.
- Fix:
  HDF5 run metadata now embeds compact readiness/source-blocker evidence when a
  run summary provides it, using the same blocker-oriented evidence compaction
  used for run manifests. Oversized payloads are not copied into attributes.
- Plan status:
  `VAL-010` is marked complete in `docs/MODULE_AUDIT/BACKLOG.md`. Remaining
  validation work is source/evidence closure, not readiness surfacing.
- Verification:
  `python3 -m py_compile src/dpf/validation/artifacts.py src/dpf/engine/core.py
  tests/test_validation_artifacts.py` passed, and
  `python3 -m pytest tests/test_validation_artifacts.py tests/test_export_scope.py
  tests/test_server_readiness.py -q` passed (`35 passed`).
- Boundary:
  no source blocker was cleared.

### 2026-05-09: IO/Export Strict Well Validator Closure

- Scope:
  closed `IO-003`.
- Fix/verification:
  no code change was needed for this task in the current pass. Existing strict
  validator behavior already covers scalar-history finiteness, required
  provenance/classification attrs, energy evidence, monotonic time, geometry
  consistency, sanitized-dataset rejection, saturation-scale values, and all-zero
  magnetic-field rejection. Focused validator tests passed (`7 passed`).
- Boundary:
  this is local dataset integrity checking, not source-backed Well/WALRUS
  validation.

### 2026-05-09: Engine/Core GPU Operator Ownership Guardrail

- Scope:
  closed `ENG-003`.
- Fix:
  backend diagnostics now distinguish backend-owned, explicit-fallback, and
  Python-operator-owned physics paths for GPU backends. Python-side Nernst and
  implicit/STS diffusion are skipped for `metal`/`mlx` so those requested
  operators cannot be applied twice.
- Verification:
  `python3 -m py_compile src/dpf/engine/backend_capabilities.py
  src/dpf/engine/physics_operators.py src/dpf/engine/core.py
  tests/test_backend_capabilities.py` passed, and the backend capability suite
  passed (`11 passed`).
- Boundary:
  this closes an engineering ownership blocker only. It does not source-verify
  the GPU physics models.

### 2026-05-11: Root Agent Operating Contract

- Scope:
  added root `AGENTS.md` as the project-level operating contract for future
  Codex/Cortex/sub-agent work.
- Plan effect:
  this supports A1 findings/status hygiene by making the first-read sequence,
  source hierarchy, evidence-state vocabulary, blocker preservation,
  verification commands, module routing, and delegation rules explicit for every
  future task.
- Scientific boundary:
  this is not validation evidence and does not change the state of Akel
  review, S1/S2 source closure, Tier 2/4/5 readiness, diagnostics formulas,
  radiation/QMF/p-B11 closure, or WALRUS/The Well provenance.
- Maintenance:
  `AGENTS.md` must be updated whenever the source-of-truth policy, evidence
  states, verification matrix, module routing, hard blockers, or multi-agent
  expectations change.

### 2026-05-11: Akel Digitization Source-Integrity Verifier

- Scope:
  added `scripts/verify_akel_digitization_source_integrity.py` as the review
  preflight for Akel Fig. 1 digitization.
- Plan effect:
  this strengthens A2 review packet intake and A3 guarded waveform comparison
  by proving the local document, PDF text parity, figure crop, SVG overlay
  source, draft packet hash, and series counts have not drifted before any
  independent reviewer decision is considered.
- Current status:
  live source-integrity verification passes all non-review checks and reports
  `validation_status="blocked_by_review"` with
  `accepted_for_validation=false`. The accepted-review gate remains blocked by
  `independent_review_missing` and `review_status_not_accepted`.
- Scientific boundary:
  this is not accepted digitization evidence and does not close S1/S2. It only
  protects the input artifact chain so a future independent review can be tied
  to the exact local source and packet.

### 2026-05-11: Source Acquisition Team Handoff Workbook

- Scope:
  exported the current source-acquisition queue to
  `docs/SOURCE_ACQUISITION_TEAM_HANDOFF_2026_05_11.xlsx` for handoff to a team
  that can acquire missing papers, books, data sheets, datasets, and supporting
  process references.
- Plan effect:
  this supports A5 source acquisition by turning the current queue into an
  email-ready workbook with 91 actionable acquisition rows, 10 already-local
  rows to avoid duplicate requests, and an intake checklist for post-acquisition
  hash/review/KR promotion steps.
- Scientific boundary:
  the workbook is not source evidence. External links remain acquisition leads
  only; validation or method support still requires local acquisition, hashing,
  `KnowledgeReference/` review, and independent review where the gate requires
  it.

### 2026-05-11: Research Papers KR Promotion And Deduplication

- Scope:
  advanced A5 source acquisition/KR ingestion for the local
  `downloaded_books_papers/Research Papers` intake folder.
- Plan effect:
  54 unique PDFs are now local `KnowledgeReference/` markdown/JSON text records,
  and 7 unique PDFs were skipped because they were already represented at
  source level. The promotion report path was
  `docs/RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.md` / `.json`; that path was
  later refreshed by the supplemental user-intake run below, while this
  initial-run count remains preserved here and in
  `docs/RESEARCH_PAPERS_INTAKE_AUDIT_2026_05_11.md`.
- Deduplication:
  16 exact byte-for-byte duplicate intake files were removed, leaving 61
  PDF-like files with 61 unique SHA-256 payloads in the intake folder.
- Newly unblocked source availability:
  Schmidt et al. 2014, "Fully Kinetic Simulations of MegaJoule-Scale Dense
  Plasma Focus" is no longer merely a citation in other KR files; the exact
  `1169854.pdf` source is now promoted as
  `KnowledgeReference/fully-kinetic-simulations-of-megajoule-scale-dense-plasma-focus-3f439245.md`
  / `.json`.
- Remaining A5 boundary:
  the new KR records are `text_parity_extracted_review_needed`. They improve
  local source availability and searchability, but they do not by themselves
  close typed target extraction, figure/table review, same-scope comparison, or
  validation-tier acceptance.

### 2026-05-11: Formulary And Local-KR Formula Audit

- Scope:
  added `docs/FORMULARY_CODE_AUDIT_2026_05_11.md` and patched confirmed
  formula mismatches found while comparing coded module families to the local
  NRL formulary and local-KR MHD/circuit identities.
- Plan effect:
  this advances A7 numerical/formula correctness and A10 physics-fidelity
  closure by removing known incorrect formula implementations while preserving
  blocker status for empirical or source-missing physics.
- Completed fixes:
  NRL Eq. 30 bremsstrahlung, Eq. 33 recombination radiation, Eq. 34 cyclotron,
  Eq. 13 radiative recombination, Braginskii perpendicular conductivity,
  electron-ion Coulomb-log/resistivity diagnostics, electron-ion mfp Coulomb log,
  SI MHD energy flux, cylindrical conservative source terms, circuit inductive
  EMF ownership, and Lee axial `fc` circuit loading.
- Remaining blockers:
  `nu_ee` needs an explicit collision/relaxation-rate convention before edit;
  empirical line cooling, QMF, p-B11, opacity/FLD, detector response, high-Z
  EOS/radiation, ablation/impurity mixing, and kinetic/neutron production paths
  still require separate source packets. A same-day physics follow-up closed
  the Lee radial `fcr` helper-level audit and replaced MLX field-aware
  fixed-ratio cross-field conduction with the local-NRL Braginskii path.
- Verification:
  the focused formulary/MHD/circuit regression suite passed (`202 passed`).
- Scientific boundary:
  this work improves coded formula correctness. It is not end-to-end DPF
  validation and does not close same-scope Akel review, S1/S2, Tier 2, Tier 4,
  Tier 5, or high-fidelity readiness.

### 2026-05-11: Physics Focus - Transport, Lee fcr, Radiation Provenance

- Scope:
  advanced the physics side of A7/A10 using only local `KnowledgeReference/`
  authority, with transport, Lee/circuit, and radiation/atomic sub-agent review
  feeding into local final checks.
- Plan effect:
  `MLX-010` is now complete: MLX field-aware thermal conduction computes the
  NRL electron-ion Coulomb log and Braginskii high-field perpendicular
  conductivity instead of using a fixed cross-field ratio when field components
  are available.
- Plan effect:
  `CIR-010` is now complete for the validation Lee helper: axial `fc` and
  radial `fcr` are separated, device `lee_fcr` overrides are applied per run,
  and radial/frozen circuit-facing inductance, `dLp/dt`, force, and metadata
  use the radial factor.
- Plan effect:
  `RAD-010` remains blocked but guarded; radiation transport now exposes
  fail-closed metadata showing the FLD/Rosseland/Kramers source packet is
  missing. `RAD-011` is complete for p-B11 metadata separation, but p-B11
  reactivity/yield remain non-validation evidence.
- Verification:
  py-compile for touched modules/tests passed, and the focused physics
  regression suite passed (`80 passed`).
- Remaining plan work:
  keep `nu_ee` convention-blocked until the public API names the intended
  collision/relaxation-rate convention. Continue source-packet work for line
  cooling, QMF derivation, p-B11 reactivity/yield, opacity/FLD, detector
  response, high-Z EOS/radiation, ablation/impurity mixing, kinetic/neutron
  production, Akel S1/S2, and validation tier closure.
- Scientific boundary:
  this makes several physics formulas and metadata surfaces more correct. It
  does not make the simulator globally verified against source of truth, does
  not close same-scope waveform validation, and does not promote scaffolded
  physics to predictive evidence.

### 2026-05-11: File-Level Supplemental Physics Guardrail Pass

- Scope:
  continued A10 physics-fidelity cleanup by checking files that were not
  individually represented in the initial module notes: ablation,
  two-temperature energy, viscosity, Nernst/Ettingshausen, sheath utilities,
  anomalous resistivity, CIV/Paschen startup, turbulence, Auluck/GV poloidal
  field utilities, and Sedov verification.
- Plan effect:
  added `PHX-001` through `PHX-006` to the module backlog and created
  `docs/MODULE_AUDIT/supplemental_physics_helpers.md`.
- Plan effect:
  `PHX-001` is complete. Seven uncovered physics helper surfaces now expose
  fail-closed metadata with `validation_status="not_validation_evidence"` and
  `can_support_validation_claims=False`.
- Remaining plan work:
  keep `PHX-002` through `PHX-006` blocked until local source packets close
  ablation constants, 2T equilibration, ion-viscosity collision conventions,
  Nernst/Ettingshausen coefficients, anomalous/CIV/Paschen startup rules,
  sheath validation, Sedov normalization, and method-only bounds for Auluck/GV
  and verification helpers.
- Verification:
  py-compile for touched helper modules/tests passed, and the focused
  supplemental-physics metadata/readiness suite passed (`16 passed`).
- Scientific boundary:
  this improves status hygiene for previously under-reviewed physics helpers.
  It does not source-verify those helpers and does not promote high-fidelity,
  startup, late-pinch, neutron, high-Z, or p-B11 claims.

### 2026-05-11: Supplemental User PDF Intake Promotion

- Scope:
  advanced A5 source acquisition/KR ingestion with the 30 PDFs supplied from
  `/Users/anthonyzamora/Downloads`, copied into
  `downloaded_books_papers/Research Papers/2026-05-11-user-ingest/`.
- Plan effect:
  the current promotion run scanned 91 unique intake PDFs, promoted 32 new
  `KnowledgeReference/` Markdown/JSON records, skipped 59 already represented
  records, failed 0 extractions, and deleted 0 duplicates. The two promoted
  records beyond the supplied batch were existing intake copies of Schmidt et
  al. 2014 (`1169854.pdf`) and the 2019 NRL Plasma Formulary.
- Source-queue effect:
  `docs/SOURCE_ACQUISITION_NEEDED.md` now distinguishes newly local text
  records from still-missing acquisitions. Many PF-1000 phase/spatial/neutron/
  energy/model-form leads now move from acquisition blockers to title-cleanup,
  figure/table review, typed target extraction, and uncertainty extraction.
  The code-backed `scientific_closure_source_acquisition_queue()` and
  `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md` were updated to match that state.
- Tooling effect:
  `scripts/promote_research_papers_to_kr.py` now treats an existing source
  SHA-256 in KR metadata as authoritative before accession/title heuristics.
  The follow-up dry-run is idempotent: `files=91 unique=91 promoted=0
  skipped_existing=91 failed=0 deleted_duplicates=0`.
- Remaining plan work:
  prioritize metadata cleanup and target extraction for the newly local PF-1000
  diagnostics papers, Toro method mapping, Lotz/Seaton atomic source packets,
  CHIANTI/Puetterich radiation review, Shumlak/Buneman instability guardrails,
  and Stepniewski/Lee/Schmidt model-form bounds. Jednorog 2017, Rezac 2012,
  Malir 2022, Mitrofanov 2014, Jakubowska 2011, Post/ADAS/Dere, Burgess/NIST,
  Nanbu/Perez, p-B11 reactivity/cross-section sources, Hutchinson, Freidberg/
  Goedbloed, and Rybicki-Lightman still remain acquisition or extraction gaps.
- Scientific boundary:
  these records are `text_parity_extracted_review_needed` and
  `source_available_not_target_extracted`. This pass improves local source
  availability only; it does not accept formulas, figures, tables, plotted
  curves, waveform points, validation targets, or any pass/fail physics claim.

### 2026-05-11: Broader PDF Inventory And Textbook Chunking

- Scope:
  corrected the source-inventory scope after the active intake reported only 91
  unique payloads.
- Plan effect:
  generated `docs/PDF_SOURCE_INVENTORY_2026_05_11.md` / `.json` with a broader
  count: 1,159 project PDF-like files outside `KnowledgeReference/`, 583
  unique project payloads, 139 Downloads PDF-like files at depth 2, 130 unique
  Downloads payloads, and 651 unique payloads across project plus Downloads.
- Intake policy:
  do not bulk-promote the 651-unique inventory. Triage
  `archive_reference_OLD/references/papers` into the active intake only when a
  source directly closes a named source-packet, module, or validation blocker.
- Textbook policy:
  large/book-length sources should use chunked Markdown. The promotion utility
  now writes sources over the page threshold as a top-level Markdown index plus
  `KnowledgeReference/chunks/` page-range files, with full page text preserved
  in JSON.
- Completed chunking:
  Toro 2009 now has a readable index at
  `KnowledgeReference/toro2009-433cd861.md` and 30 page-range chunks under
  `KnowledgeReference/chunks/toro2009-433cd861/`. The chunking report is
  `docs/KR_TEXTBOOK_CHUNKING_2026_05_11.md` / `.json`.
- Remaining plan work:
  avoid rechunking older large KR records unless actively reviewing them,
  because older docs/findings may cite top-level line numbers. Continue source
  triage from the broader inventory before any further KR promotion.

### 2026-05-11: Kepler Read-Only Formulary Audit Backlog

- Audit result:
  Kepler completed an independent read-only pass over formula-bearing
  collision, radiation, diagnostics, sheath, atomic, and fluid modules using
  only `KnowledgeReference/plasma-formulary.md` for directly comparable
  formulas.
- New suspected physics tasks:
  check and patch `src/dpf/fluid/ionization.py` bremsstrahlung coefficient/unit
  conversion; check and patch recombination radiation scaling in
  `src/dpf/radiation/improved_radiation.py` and
  `src/dpf/radiation/line_radiation.py`; replace or explicitly label
  `src/dpf/collision/spitzer.py` perpendicular conductivity interpolation; and
  resolve the `nu_ee = sqrt(2) * nu_ei(..., Z=1)` convention before changing
  behavior.
- Confirmed direct-formulary matches:
  keep the audit notes for bremsstrahlung helper, cyclotron radiation, e-i/i-i
  collision rates, Spitzer resistivity, Debye length, cold-ion Bohm speed, EOS,
  Saha ratio, beta/Alfven/fast speeds, Bennett diagnostics, and the
  direction/dimensional form of two-temperature equilibration.
- Plan boundary:
  no code was changed by this audit. These items are queued for focused patch
  and test work, with each formula still requiring source-line traceability
  before being called validated.

### 2026-05-11: Active Intake Source Fidelity Review Applied

- Work completed:
  added and ran `scripts/verify_kr_source_fidelity.py --apply` across the 91
  active-intake KR records. Each matching KR JSON now has a
  `source_fidelity_review` section, and each same-stem Markdown file has a
  source-fidelity summary marker.
- Audit result:
  `docs/KR_SOURCE_FIDELITY_AUDIT_2026_05_11.md` / `.json` reports 91 checked
  and 91 updated records, with 90 records containing recovered secondary
  extraction items and 10,767 recovered items copied into KR JSON.
- Source-critical coverage:
  the pass copied or indexed 2,012 figure captions, 255 table captions,
  345 extracted table matrices, 14,554 formula-like lines, 9,533 numeric target
  contexts, 2,143 uncertainty contexts, and 19,784 PDF image-block counts.
- Plan effect:
  the active intake is no longer text-parity-only for the reviewed source
  records. Remaining scientific work is typed target extraction, plotted-curve
  digitization where needed, uncertainty normalization, and module/test
  traceability.
- Boundary:
  source PDFs remain authoritative for visual geometry and plotted curves. This
  pass prevents source-critical text artifacts from being dropped, but it does
  not by itself accept validation thresholds.

### 2026-05-11: Target Extraction And Digitization Start

- Work completed:
  started typed target extraction for Cikhardtova 2015, Szydlowski 2004, Klir
  2011, Springham 2021, and Catenacci 2020, using only local
  `KnowledgeReference/` records and their local source PDFs.
- Implementation status:
  `src/dpf/validation/kr_targets.py` now exposes five new target records for
  PF-1000 linear-density motion, PF-1000 fast-ion/neutron spectrum and
  anisotropy, ToF detector response, NX3 Zr/Be activation anisotropy, and NNSS
  DPF neutron time-energy tomography.
- Digitization status:
  `scripts/start_target_extraction_digitization.py` generated
  `docs/TARGET_EXTRACTION_DIGITIZATION_2026_05_11.md` / `.json` and rendered
  23 crop-pending workbench pages under
  `KnowledgeReference/figures/target-extraction/2026-05-11/`. It also created
  36 hash-recorded crop candidates: Cikhardtova 2015 Figs. 1-6, Szydlowski
  2004 Figs. 1-5, Klir 2011 Figs. 1-4, Springham 2021 Figs. 1-7 and
  Tables 1-2, and Catenacci 2020 Figs. 1-8 and Tables I-IV.
- Boundary:
  no new figure/table digitization packet is accepted. Crop candidates remain
  `crop_candidate_unreviewed` with `accepted_for_validation=false`. OCR-suspect
  values, plotted curves, table matrices, and visual geometry remain
  review-blocked until extracted packets pass
  `digitization_verification_evidence()` and independent review.
- Verification:
  focused target/digitization/source/quality tests passed (`169 passed`);
  `py_compile`, `git diff --check`, and the generated report invariant check
  passed with 5 sources, 23 rendered pages, 36 unreviewed crop candidates, and
  0 accepted validation packets.

### 2026-05-12: Validated Physics Pipeline Plan

- Work completed:
  added `docs/VALIDATED_PHYSICS_PIPELINE_PLAN.md` as the planning baseline for
  promoting local sources into validated physics evidence.
- Plan effect:
  the scientific closure plan is now explicitly staged from
  `source_validated` through source-line review, typed target extraction,
  figure/table digitization, formula evidence, uncertainty propagation,
  comparator binding, same-scope packet assembly, and validation certificate
  release gates.
- Guardrails:
  no state may skip independent review or same-scope checks. Source validation,
  recovered source-fidelity text, crop candidates, axis scaffolds, formula
  audits, and typed target drafts remain non-accepting until their specific
  gates pass.
- Immediate execution order:
  finish independent review handoff for existing A14/Akel draft packets, start
  source-line review for the five May 12 target candidates, build canonical
  evidence schemas and typed target validators, then wire UQ, comparator, and
  same-scope packet gates before any validation certificate can be written.
- SRS traceability:
  the plan proposes candidate requirements `DPF-VV-011` through `DPF-VV-016`
  for typed evidence, independent digitization review, formula packets, UQ
  propagation, comparator binding, and same-scope packet rejection of
  cross-scope evidence. These are not Doorstop-imported until reviewed.

### 2026-05-12: Source-Truth Pipeline Validation Pass

- Work completed:
  ran the May 12 source-validation, target-triage, A14 handoff/backlog, Akel
  source-integrity, source-fidelity, and pytest validation lanes against the
  current workspace state.
- Source pipeline evidence:
  `scripts/validate_user_pdf_may12_sources.py` completed with
  `promoted=28 stage_only=7 target_candidates=5 failures=0`;
  `scripts/create_user_pdf_may12_target_triage.py` completed with
  `entries=28 target_candidates=5`.
- Review-gated digitization evidence:
  `scripts/create_a14_independent_review_handoff.py` reported
  `review_item_count=9`, `axis_context_item_count=3`, and
  `accepted_for_validation_count=0`;
  `scripts/create_a14_remaining_extraction_backlog.py` reported
  `total_crop_count=36`, `reviewable_draft_packet_count=9`, and
  `accepted_for_validation_count=0`.
- Akel integrity evidence:
  `scripts/verify_akel_digitization_source_integrity.py` passed its
  pre-review integrity checks but still reports
  `validation_status=blocked_by_review` and `accepted_for_validation=false`.
  Remaining blockers are independent review missing and review status not
  accepted.
- Source-fidelity evidence:
  `scripts/verify_user_pdf_batch_source_fidelity_2026_05_12.py` was run
  without `--apply` and reported `selected=28`, `records=28`, `updated=0`,
  `recovered_records=27`, and `recovered_items=11376`.
- Pytest evidence:
  the top-level non-slow/non-Athena pipeline completed:
  `python3 -m pytest tests/ -q -m "not slow and not athena"` ->
  `4151 passed, 7 skipped, 362 deselected, 48 xfailed, 14 xpassed,
  25 warnings in 445.00s`.
- Focused source/guardrail evidence:
  source, digitization, acquisition, diagnostics, readiness, calibration,
  Akel integrity, preset scope, MHD physics, MLX guardrail, and unreviewed
  physics lanes also passed in focused runs before the full pass. The largest
  split lanes completed as `878 passed, 3 skipped, 242 deselected, 30 xfailed,
  9 xpassed`, `380 passed, 3 skipped, 6 deselected, 1 xfailed`,
  `161 passed, 22 deselected, 4 xfailed`, and
  `2343 passed, 1 skipped, 69 deselected, 10 xfailed, 5 xpassed`.
- Validation blockers preserved:
  passing pytest does not mean all physics evidence is accepted. Explicit
  blocked gates remain for MHD/RADPF acceptance angles 1, 3, and 5; PF-1000
  circuit validation (`I_peak=2.277 MA`, `21.8%` high, and PF-1000
  `NRMSE=0.370` above the `0.35` fence); PF-1000 model-validity fraction
  (`0.3077` below the `0.40` gate); and all A14/Akel draft digitizations until
  independent review accepts them.
- Boundary:
  no plotted curve, table extraction, formula threshold, uncertainty value, or
  simulation validation criterion was newly accepted. The current state is
  source-validated and pipeline-clean with explicit review/acceptance blockers,
  not a release of new validated physics claims.

### 2026-05-12: PF-1000 Standard Circuit Source-Scope Repair

- Work completed:
  repaired the standard 27 kV PF-1000 production validation path so it no
  longer mixes Akel 16 kV/shot-series bank and geometry values into the
  Scholz/Gribkov 27 kV validation scope.
- Source basis:
  local `KnowledgeReference/plasma-physics-and-technology-1211-9-2025.md`
  supports the standard PF-1000 bank/model fit (`L0=33.5 nH`,
  `C0=1332 uF`, `r0=6.1 mOhm`, `fc=0.7`, `fm=0.13`, `fmr=0.35`,
  `fcr=0.65`) at 3.5 Torr D2; local Lee-course KR lines support the
  same standard geometry (`a=11.55 cm`, `b=16 cm`, `z0=60 cm`) and
  `L0/r0` range. Akel 16 kV/shot-series `25 nH` / `48 cm` values remain
  isolated in `PF-1000-16kV`.
- Implementation:
  `run_rlc_snowplow_pf1000()` now uses the Lee/Malek standard PF-1000 bank
  and passes `radial_current_fraction=0.65` into `SnowplowModel`.
  `PF1000_DATA`, `PF1000_GRIBKOV_DATA`, and the estimated 20 kV variant now
  share the standard 27 kV bank/geometry scope where applicable.
- Validation result:
  the previous blocked PF-1000 circuit evidence is superseded. The current
  production RLC+snowplow run gives `I_peak=1.826508 MA`,
  `t_peak=7.041 us`, Scholz peak error `2.33%`, Scholz `NRMSE=0.181734`,
  Gribkov peak error `1.06%`, and Gribkov `NRMSE=0.153639`.
- Pipeline result:
  `python3 -m pytest tests/test_validation_ci.py -q -o addopts=""` completed
  as `28 passed`; `TestModelValidityWindow` completed as `2 passed` with
  PF-1000 20% point-wise validity fraction `0.692308`; `TestBlindPrediction16kV`
  completed as `4 passed`; quality/readiness guardrails completed as
  `2 passed`.
- Remaining blockers:
  MHD/RADPF acceptance is still blocked on angles 1, 3, and 5
  (`tests/test_mhd_acceptance.py`: `2 passed, 3 xfailed`). Reflected-shock
  dip/peak acceptance remains partly blocked (`3 passed, 3 xfailed`) pending
  source-scoped threshold recalibration. A14/Akel digitization packets remain
  `blocked_by_review` until independent review accepts them.
- Boundary:
  this repair validates the standard PF-1000 circuit regression path and
  clears the stale PF-1000 circuit/model-validity blockers. It does not issue
  a new validation certificate and does not accept cross-scope Akel 16 kV
  data for 27 kV PF-1000 scoring.

### 2026-05-12: Source-Truth Simulation Monitor And Preset Repair

- Work completed:
  added and ran `scripts/run_source_truth_simulation_monitor.py` to monitor the
  full local app-engine preset set plus all source-registered waveform devices.
  The monitor writes auditable JSON/Markdown and classifies broken runs,
  nonfinite arrays, runtime warnings, source authority, and accuracy-review
  flags.
- Generated evidence:
  `docs/SOURCE_TRUTH_SIMULATION_MONITOR_2026_05_12.json` and
  `docs/SOURCE_TRUTH_SIMULATION_MONITOR_2026_05_12.md`.
- Source boundary:
  monitor science labels come only from local `KnowledgeReference/`-backed
  registry metadata. Reconstructed waveforms, unverified waveforms,
  reference-only devices, and Akel draft digitization remain nonaccepting.
- Repair finding:
  the monitor caught the stale user-facing `pf1000` app preset: it completed
  but peaked at `2.249 MA` against the standard PF-1000 reference, while the
  source-scoped production circuit path had already cleared the current
  pipeline fence. The preset now uses the Lee/Malek standard PF-1000 values
  (`R0=6.1 mOhm`, `a=0.1155 m`, `fcr=0.65`) and is labeled
  `same_scope_source_reviewed_not_certificate`.
- Runtime repair:
  `app_engine.py` now delegates DD reactivity to
  `dpf.diagnostics.neutron_yield.dd_reactivity`, eliminating the monitor
  runtime warning from the duplicate Bosch-Hale helper.
- Current result:
  `python3 scripts/run_source_truth_simulation_monitor.py --include-pytest-lanes`
  completed with 16/16 presets operational, `broken_preset_count=0`,
  `warning_preset_count=0`, and `pytest_failed_lane_count=0`.
- Current PF-1000 preset:
  `pf1000` now reports `I_peak=1.826 MA`, `t_peak=6.346 us`, and
  `2.337%` peak error against the PF-1000 registry reference. It remains
  non-certifying until accepted run-level validation evidence exists.
- Remaining monitor findings:
  `nx2` preset timing and `poseidon_60kv` preset peak/timing need accuracy
  review. Nonaccepting device-level review flags remain for `MJOLNIR`, `NX2`,
  and `PF-1000-16kV`; those are tracked as source/provenance or review-gated
  work rather than accepted validation failures.
- Verification:
  `python3 -m pytest tests/test_neutron_yield.py tests/test_preset_source_scope.py tests/test_validation_ci.py -q -o addopts=""`
  passed as `103 passed`; `git diff --check` passed. The monitor's MHD
  acceptance lane skipped all 5 tests because MLX was unavailable in this
  shell, so it does not refresh MHD/RADPF acceptance evidence.

### 2026-05-12: Source-Config Monitor Ratchet

- Work completed:
  extended the full simulator monitor to compare source-facing preset
  configuration against the local device registry, so operational runs now
  expose source-config mismatches as explicit `source_config_flags` instead of
  relying only on output peak/timing errors.
- Source-scope repairs:
  `poseidon_60kv` now uses the local POSEIDON-60kV registry fit
  (`fc=0.60`, `fm=0.275`, `fmr=0.45`, `fcr=0.44`) and is labeled
  `same_scope_source_reviewed_waveform_unverified_not_certificate`.
  Current monitor result: `I_peak=3.155 MA`, `t_peak=1.990 us`, `1.102%`
  peak error, no warnings, and no source-config flags.
- Akel registry repair:
  `PF-1000-16kV` now matches the local Akel shot-12581 source scope:
  `p0=1.20 Torr`, `r0=6.1 mOhm`, `Yn=6.1e9`, `fm=0.17`, `fc=0.70`,
  `fmr=0.26`, `fcr=0.75`. The direct device monitor now reports
  `Ipeak Err=1.613%`, `Timing Err=12.667%`, `NRMSE=0.167`; this remains
  nonaccepting because the waveform is reconstructed and unverified.
- Additional source-alignment repairs:
  `unu_ictp` now uses the local Lee/Saw table p.152 registry conditions
  (`15 kV`, `4 Torr`) and reports `I_peak=0.181 MA`, `0.502%` peak error.
  FAETON fill density now matches its 12 Torr source scope; its remaining
  source-config flag is the two-step radial-current model versus the registry
  single `fcr`.
- Final monitor evidence:
  `python3 scripts/run_source_truth_simulation_monitor.py --include-pytest-lanes`
  completed with `device_count=9`, `preset_count=16`,
  `broken_preset_count=0`, `warning_preset_count=0`,
  `accuracy_review_preset_count=1`, `source_config_review_preset_count=3`,
  `accuracy_review_device_count=2`, and `pytest_failed_lane_count=0`.
- Remaining monitor findings:
  `nx2` remains accuracy/source-config review only; the registry marks it
  `reference_only` with no waveform. `MJOLNIR` remains nonaccepting and source
  config review needed. `FAETON-I` remains nonaccepting pending source review
  of the two-step radial current convention.
- Test evidence:
  `python3 -m pytest tests/test_validation_ci.py tests/test_neutron_yield.py tests/test_preset_source_scope.py -q -o addopts=`
  passed as `106 passed`; `python3 -m pytest tests/test_akel_digitization_source_integrity.py tests/test_unreviewed_physics_metadata.py -q -o addopts=`
  passed as `9 passed, 5 warnings`. The monitor lanes passed, with MHD
  acceptance skipped because MLX was unavailable in this shell.
- Boundary:
  this is an operational/source-truth audit improvement, not a new validation
  certificate. Nonaccepting waveform provenance and review gates remain
  preserved.

### 2026-05-12: Full Source-Truth Simulator Monitor Closure

- Work completed:
  finished the current simulator-monitor ratchet for NX2, MJOLNIR, and FAETON.
  NX2 is now reference-only instead of silently empirical, MJOLNIR follows the
  local Schmidt 2021 1 MJ registry values, and FAETON declares a noncertifying
  Damideh 2025 Table 3 two-step radial-current source scope.
- FAETON source-config result:
  FAETON now uses the KR-backed Table 3 shot-1027 factors `fcr=0.8` and
  `fcr2=0.58`. The remaining monitor flag is limited to
  `snowplow.radial_transition_time_not_in_faeton_kr_extract_observed=7e-06`,
  which means the transition time still needs accepted digitization/source
  closure.
- Full monitor evidence:
  `python3 scripts/run_source_truth_simulation_monitor.py --include-pytest-lanes`
  completed with `device_count=9`, `validation_ready_device_count=1`,
  `preset_count=16`, `broken_preset_count=0`, `warning_preset_count=0`,
  `accuracy_review_preset_count=2`, `source_config_review_preset_count=1`,
  `accuracy_review_device_count=2`, `pytest_lane_count=3`, and
  `pytest_failed_lane_count=0`.
- Remaining monitor findings:
  all presets completed without nonfinite arrays. `nx2` still needs peak and
  timing accuracy review; `mjolnir` still needs peak-current accuracy review.
  Those remain nonaccepting source/provenance issues, not accepted validation
  failures.
- Test evidence:
  `tests/test_preset_source_scope.py` passed as `13 passed`;
  `tests/test_validation_ci.py tests/test_neutron_yield.py tests/test_preset_source_scope.py`
  passed as `109 passed`;
  `tests/test_akel_digitization_source_integrity.py tests/test_unreviewed_physics_metadata.py`
  passed as `9 passed, 5 warnings`; and the FAETON preset-specific snowplow
  regression passed as `1 passed, 425 deselected`. The full monitor lanes also
  passed, with MHD acceptance skipped because MLX is unavailable in this shell.
- Boundary:
  this pass makes the simulator workflow operationally auditable against local
  source truth. It does not issue validation certificates or accept any
  reconstructed, unverified, reference-only, or review-blocked evidence.

### 2026-05-12: Source-Gap And Model-Coverage Classification

- Work completed:
  updated the source-truth monitor to distinguish actual validation-ready
  accuracy failures from nonaccepting source gaps, source-config gaps, and
  missing model coverage.
- Current result:
  `python3 scripts/run_source_truth_simulation_monitor.py --include-pytest-lanes`
  completed with `preset_count=16`, `broken_preset_count=0`,
  `warning_preset_count=0`, `accuracy_review_preset_count=0`,
  `source_gap_review_preset_count=1`,
  `model_coverage_review_preset_count=1`,
  `source_config_review_preset_count=1`,
  `accuracy_review_device_count=0`,
  `source_gap_review_device_count=7`,
  `model_coverage_review_device_count=1`, and
  `pytest_failed_lane_count=0`.
- What is now pinpointed:
  `nx2` is a source-gap case, not an accepted physics failure: the local source
  state is reference-only and lacks same-shot deuterium waveform evidence.
  `mjolnir` is a model-coverage case: the local MJOLNIR source says current
  traces require restrike timing/magnitude variation, which is not yet present
  as accepted structured simulator input. `faeton` remains a source-config case
  because the 7 us radial transition time is not in the current KR extract.
- Test evidence:
  `tests/test_preset_source_scope.py` passed as `14 passed`;
  `tests/test_validation_ci.py tests/test_neutron_yield.py tests/test_preset_source_scope.py`
  passed as `110 passed`; `tests/test_akel_digitization_source_integrity.py tests/test_unreviewed_physics_metadata.py`
  passed as `9 passed, 5 warnings`; and `git diff --check` passed.
- Boundary:
  this is a troubleshooting classification improvement, not a validation
  certificate. It makes the next science work concrete: add accepted MJOLNIR
  restrike/current-diversion parameters or a supported model path, and acquire
  or digitize same-scope waveforms for the source-gap rows.

### 2026-05-19: Sprint 3 Completion Audit Rejected; Sprint 3R Required

- Work completed:
  recorded the governance outcome from the Sprint 3 completion audit at HEAD
  `269d7d1`. The durable audit is
  `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_COMPLETION_2026_05_19.md`; the
  next-team instruction packet is
  `docs/FIRST_PRINCIPLES_SPRINT3R_REMEDIATION_HANDOFF_2026_05_19.md`.
- Planning impact:
  Sprint 4 is blocked until Sprint 3R closes the audit findings. The next
  sprint must repair control-plane ledgers, startup fail-closed acceptance,
  neutron authority status, NumPy 2 beam-target integration, PF-1000 mask
  source status, `Sigma_p` packet completeness, closure matrix completeness,
  restart ledger merge coverage, and traceability path drift.
- Boundary:
  the program remains on the first-principles PF-1000/Akel path with reduced
  models kept as baselines only. This is an audit/control-plane and runtime
  fail-closed remediation step, not a validation certificate or whole-shot
  acceptance step.

## Sprint 3R Status (2026-05-19)

Sprint 3 completion audit (docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_COMPLETION_2026_05_19.md)
identified findings A1–A12. Sprint 3R remediation is in progress per
docs/FIRST_PRINCIPLES_SPRINT3R_REMEDIATION_HANDOFF_2026_05_19.md.

Findings summary (A1–A12):
- A1: startup BVP acceptance can be spoofed by caller-declared payloads.
- A2: scalar neutron yield can be promoted to mechanism authority.
- A3: NumPy 2 breaks beam_target._trapezoid_integral().
- A4: blocked insulator dimensions produce source-backed geometry masks.
- A5: under-resolution gate does not cover all source-supported features.
- A6: SigmaPSurfacePacket schema incomplete (missing SHA-256 fields and
  operand arrays).
- A7: power-port consumes dict-form Sigma_p packet without reconstruction or
  named blocker.
- A8: closure matrix omits electron_inertia and stopping_collisions from
  REQUIRED_EFFECTS.
- A9: merged restart ledger drops extended cumulative channels.
- A10: packet ledgers contradict final submission (split-brain delivery state).
- A11: shorthand [KR: ...] citations remain in WP-N5 source audit.
- A12: traceability paths reference closures.py and certificate.py instead of
  closure_packet.py and certificate_gate.py.

S3R.1 (this pass) closes A10, A11, A12:
- 4-boolean delivery state (research_packet_delivered, runtime_foundation_delivered,
  accepted_physics_delivered, validation_delivered) replaces the stale 3-boolean scheme.
- All Sprint 3 BLOCKER_MATRIX rows updated; sprint_3/PENDING.md references removed.
- S3.1 and S3.9 rows added to CLAIMS_LEDGER.csv and TEST_MAP.csv.
- All shorthand [KR: ...] citations in WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md
  expanded to full KnowledgeReference/ paths with line ranges.
- closures.py → closure_packet.py and certificate.py → certificate_gate.py
  corrected in all packet docs and SRS/RTM artifacts.
- Test suite extended to enforce: 4-boolean delivery state, no PENDING.md
  references, no shorthand citations, no bad module paths.

S3R.2–S3R.7 are assigned to parallel agents and are in progress.

### 2026-05-20: PDF Corpus Rescan Added A Source-Extraction Queue

- Work completed:
  rescanned the local PDF corpus for first-principles blocker leads and added
  `docs/FIRST_PRINCIPLES_PDF_CORPUS_RESCAN_2026_05_20.md` as the durable
  discovery and extraction-priority record.
- Findings:
  the strongest raw-PDF promotion candidates are Auluck et al. 2021
  (`/Users/anthonyzamora/Downloads/plasma-04-00033.pdf`) and Bernard et al.
  1977 (`/Users/anthonyzamora/Downloads/bernard1977.pdf`). The rescan also
  identified already-KR records that should be target-extracted instead of
  reingested, including Krishnan 2012, Malir 2024, UCSD/Beg current-sheath
  initiation, Blagoev electric-flux formation diagnostics, and Beresnyak
  HAWK/ideal-MHD method records.
- Boundary:
  no raw PDF was promoted to authority and no validation state changed.
  Same-scope PF-1000/Akel 16 kV `V(t)`, `T_e/T_i`, X-ray, neutron spectrum, and
  anisotropy remain blocked. The next parallel source work is KR promotion,
  target extraction, and source-index alias reconciliation.

### 2026-05-20: P0 Corpus-Rescan PDFs Promoted To KR

- Work completed:
  promoted the two P0 raw-PDF candidates from the corpus rescan into
  fail-closed `KnowledgeReference/` text-parity records using
  `scripts/promote_corpus_rescan_2026_05_20.py`. The durable promotion ledger is
  `docs/CORPUS_RESCAN_KR_PROMOTION_2026_05_20.md`.
- Promoted records:
  `KnowledgeReference/update-on-the-scientific-status-of-the-plasma-focus-1385adeb.md`
  with 9 page-range chunks, and
  `KnowledgeReference/the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md`.
- Boundary:
  promotion did not change validation or runtime acceptance state. The promoted
  records remain `source_available_not_target_extracted` until source-fidelity
  review and typed target extraction are complete.
