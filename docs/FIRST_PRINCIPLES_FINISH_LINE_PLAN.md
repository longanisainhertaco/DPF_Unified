# First-Principles DPF Finish-Line Plan

Date: 2026-05-15

Status: active execution specification for first-principles development. This
plan does not promote any current run, digitization packet, source target, or
readiness status to accepted scientific evidence.

## Purpose

This document is the execution specification for getting DPF-Unified from the
current PF-1000/Akel engineering probe to a true first-principles Dense Plasma
Focus simulator.

The first demonstrator remains PF-1000/Akel shot-12581 at 16 kV. Lee/RADPF and
snowplow paths stay in the project as baselines, regression fixtures, and
comparison models only. Predictive authority moves to resolved fields,
conservation laws, source-backed physical closures, same-scope validation
packets, reviewed uncertainty, and validation certificates.

## Local Source Basis

Scientific claims in this plan are scoped to local `KnowledgeReference/` files.
The immediate source-routing set is:

- `KnowledgeReference/auluck-2021-dpf-circuit-element.md` for the
  first-principles circuit-element and Poynting-theorem boundary.
- `KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md` for
  circuit/MHD boundary-condition style and generator coupling context.
- `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md`
  for reduced-model dynamic-resistance comparison boundaries.
- `KnowledgeReference/unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md`
  for DPF MHD, 3D, EOS/conductivity, radiation, two-temperature, and
  beam-target limitations.
- `KnowledgeReference/the-dense-plasma-focus-a-versatile-dense-pinch-for-diverse-applications.md`
  for the need to complement late-pinch MHD with particle/PIC-type physics.
- `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md`
  for mechanism-separated thermonuclear and beam-target neutron evidence.
- `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md`
  for the newly ingested arXiv:2604.09032v1 hybrid PIC-fluid DPF source,
  routed to first-principles architecture review for kinetic-ion/fluid-electron
  handoff, fully electromagnetic field evolution, generalized Ohm-law terms,
  vacuum-field handling, and neutron-yield authority boundaries.

These files guide scope and blocker classification. They do not, by themselves,
accept any current simulation output.

## Definition Of First Principles

For this project, a first-principles DPF simulation is one whose accepted claim
is produced by evolved physical state variables, conservation laws, source-backed
closures, explicit approximations, and reviewed same-scope evidence.

It must:

- evolve plasma mass, momentum, energy, magnetic field, and circuit state through
  resolved equations and explicit boundary conditions;
- drive the circuit-plasma power exchange through a resolved power port, not a
  fitted Lee/RADPF inductance, current fraction, snowplow mass fraction, fixed
  crowbar timing, or empirical beam fraction;
- expose every numerical limiter, floor, cap, repair, fallback, source-term
  split, and backend precision decision as either verified numerics,
  source-backed physical bound, or acceptance blocker;
- attach a numerical-fidelity packet and a physics-closure packet before any
  first-principles claim can leave `engineering_probe`;
- define the dimensionality and handoff boundary: either the accepted claim stops
  before the local sources say MHD breaks down, or the path includes validated 3D
  MHD plus kinetic/hybrid particle physics for the post-MHD interval;
- separate thermonuclear and beam-target neutron mechanisms before total
  neutron-yield authority can pass;
- validate only against accepted same-scope local `KnowledgeReference/` evidence.

It cannot:

- hide state repair behind normal-looking waveforms;
- use reduced-model closure factors as predictive authority;
- treat draft digitization, source validation, figure crops, formula audits, or
  engineering probes as scientific validation;
- mix Akel 16 kV shot-12581 evidence with full-energy PF-1000 evidence unless a
  reviewed transfer rule exists.

## Claim Levels

| Label | Meaning | Minimum gate |
| --- | --- | --- |
| Engineering Probe | Runnable finite candidate; no scientific validation claim. | `FP-1` plus fail-closed readiness metadata. |
| Reference Candidate | Candidate has no hidden acceptance-blocking limiters and carries numerical plus physics-closure packets, but some same-scope evidence can remain pending. | `FP-1` through `FP-8`; certificate remains blocked. |
| Accepted First-Principles PF-1000/Akel | Same-scope PF-1000/Akel certificate passes with numerical, physics, comparator, UQ, and review evidence. | `FP-1` through `FP-14`. |
| General First-Principles DPF Tool | A second device or shot repeats the full evidence path without hidden PF-1000/Akel assumptions. | `FP-15`. |

Unknown, draft, blocked, or partial states fail closed. Only
`validated_physics_evidence` can support an accepted scientific claim.

## Finish Line

The PF-1000/Akel milestone is complete when one ordinary package-native run can
satisfy all of these conditions in the same validation scope:

- It runs through one supported `src/dpf` execution path, not a repo-root app
  special case.
- It starts from a source-backed startup boundary-value problem: breakdown or
  preionization, electrode and insulator boundary conditions, current-density
  distribution, ionization state, electron/ion temperature state, magnetic field,
  velocity field, and sheath-liftoff metadata.
- It evolves with resolved fields and explicit conservation ledgers, not
  Lee/RADPF closure factors.
- It drives circuit feedback through a resolved power port with recorded
  Poynting-surface or `J.E` accounting, electrode work, sign convention, time
  centering, and residual tolerance.
- It completes the accepted interval without acceptance-blocking density,
  pressure, temperature, velocity, magnetic-field, magnetic-energy, resistivity,
  timestep, current-floor, or back-EMF caps/clips/repairs.
- It carries accepted numerical-fidelity evidence for the active solver,
  geometry, cylindrical terms, `div B`, time integration, shock behavior,
  resistive diffusion, Joule heating, energy update, restart reproducibility,
  backend or precision choice, and package-native entrypoint.
- It carries accepted physics-closure packets for every active or bounded-out
  effect: EOS, ionization, two-temperature physics, transport, radiation,
  impurity/ablation, Hall/FLR/kinetic scope, 3D scope, startup, restrike,
  anomalous resistance, and beam-target coupling.
- It includes a dimensionality decision. A 2D/axisymmetric MHD claim must stop
  before the local source-backed breakdown of MHD applicability, or the accepted
  path must include validated 3D MHD plus kinetic/hybrid handoff.
- It compares against accepted same-scope local evidence: current waveform,
  current dip, phase timing, spatial density, magnetic/EM field, temperature,
  neutron yield, neutron timing, spectrum, anisotropy, detector or activation
  response, and propagated uncertainty. If PF-1000/Akel lacks enough same-scope
  evidence, the accepted claim must be narrowed or the first accepted scope must
  move to a better-diagnosed local source target.
- It writes a validation certificate that rejects draft, cross-scope, blocked,
  missing-UQ, missing-review, hidden-limiter, or app-only evidence.

The broader DPF-machine tool is finished only after this same pattern is applied
to at least one additional device or shot scope without changing the scientific
source rules.

## Evidence State Contract

| State | Can support accepted first-principles claim | Required handling |
| --- | --- | --- |
| `engineering_probe` | No | May guide implementation only. Must show blockers. |
| `candidate_not_validated` | No | Requires numerical, physics, and evidence review before promotion. |
| `blocked` or `blocked_by_review` | No | Blocks certificate and readiness. |
| `source_validated` | No | Confirms source identity only. |
| `accepted_source_extraction` | No by itself | Can feed comparator after UQ and binding. |
| `comparator_bound` | Only for that observable | Requires same-scope packet before run-level validation. |
| `same_scope_packet_ready` | Conditional | Can feed certificate after all required groups and UQ pass. |
| `validated_physics_evidence` | Yes | Only state allowed in accepted certificate. |

Every FP milestone must report one of these states or explicitly map its local
state to one of these states. Unknown states fail closed.

## Completion Gates

| ID | Milestone | Current state | Exit evidence |
| --- | --- | --- | --- |
| FP-0 | Execution focus freeze | Partial | New work changes solver, startup, power-port coupling, source extraction, comparators, UQ, or certificate gates. UI/reporting work is allowed only when required by these gates. |
| FP-1 | Package-native PF-1000/Akel engineering probe | Source-scoped PF-1000/Akel 3D deck added; broader routing still partial | `docs/FIRST_PRINCIPLES_PF1000_PACKAGE_DECK_SOURCE_SEARCH_2026_05_15.md` records the built-in package-native PF-1000/Akel shot-12581 engineering deck. `dpf first-principles-3d` now defaults to that source-scoped deck and remains non-promoting; `dpf first-principles`, `dpf simulate --run-mode=first_principles_mhd`, API, and app surfaces still need unified package-native routing. |
| FP-2 | Global limiter registry and readiness gate | Source search complete for first pass; package-native packet remains blocked | `docs/FIRST_PRINCIPLES_LIMITER_READINESS_SOURCE_SEARCH_2026_05_15.md` finds that every floor, cap, clip, repair, fallback, timestep/current/back-EMF cap, precision fallback, and solver-layer limiter must be inventoried and classified before acceptance. The package-native runner now emits a fail-closed `limiter_readiness` packet. |
| FP-3 | Limiter-free or physically bounded candidate | Source search complete for first pass; full-horizon limiter-zero proof remains blocked | The same limiter-readiness packet blocks until a full-horizon run proves zero acceptance-blocking limiter activations and every remaining bound is either source-backed or a verified numerical method. Current bounded short probes remain engineering evidence only. |
| FP-4 | Numerical reference workflow | Source search complete for first pass; accepted numerical-fidelity packet remains blocked | `docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_2026_05_15.md` finds source support for the required numerical test surfaces and existing candidate component tests, but no accepted packet with complete tolerances, convergence evidence, limiter-zero proof, backend/precision scope, artifact hashes, and review. The package-native runner now emits a fail-closed `numerical_fidelity` packet. |
| FP-5 | Source-backed startup BVP | Source search complete for first pass; package-native packet remains blocked for accepted whole-shot BVP | `docs/FIRST_PRINCIPLES_BLOCKER_SOURCE_SEARCH_2026_05_15.md` finds that the corpus supports explicit startup packet modes and an end-of-rundown sheath engineering candidate, but not a complete neutral-gas breakdown/flashover/preionization/liftoff BVP. The package-native runner now emits a fail-closed `startup_bvp` packet, rejects seeded/legacy startup modes for accepted claims, and blocks acceptance unless reviewed imported PIC sheath fields/particles or a source-backed surface-breakdown BVP is attached. |
| FP-6 | Resolved power-port circuit coupling | Source search complete for first pass; candidate packet remains blocked for accepted power authority | `docs/FIRST_PRINCIPLES_POWER_PORT_SOURCE_SEARCH_2026_05_15.md` finds that the corpus supports field-power authority through Poynting flux or `J.E`, but accepted whole-shot authority still requires a reviewed packet for the implemented geometry: named interface/domain, terminal voltage/current, sign convention, time centering, electrode work, active load, diagnostic alternatives, and residual budget. The package-native packet now separates the active placeholder load from diagnostic `L_field = 2 E_B/I^2` and keeps both non-promoting. |
| FP-7 | 3D hybrid PIC-fluid dimensionality and handoff | Source search complete for first pass; package-native packet remains blocked for accepted whole-shot authority | `docs/FIRST_PRINCIPLES_DIMENSIONALITY_SOURCE_SEARCH_2026_05_15.md` finds that true 3D and kinetic handoff are source-required for full claims. The runner now emits a dimensionality/handoff packet with claim modes, source-model limitations, candidate runtime channels, and handoff-required observables, but accepted authority still requires source-equivalence evidence, same-scope 3D evidence, electron-energy/kinetic limitations, MHD-to-kinetic state transfer if used, and mechanism-separated neutron authority. |
| FP-8 | Physics closure packets | Source search complete for first pass; package-native closure matrix remains blocked for acceptance | `docs/FIRST_PRINCIPLES_CLOSURE_SOURCE_SEARCH_2026_05_15.md` finds engineering-candidate source support for conductivity, generalized Ohm, predictor-corrector current, Marder cleaning, ion PIC plumbing, and electron-energy scaffolding. The runner now emits a closure matrix with per-effect required packet channels, classifications, missing channels, claim impacts, review status, and candidate runtime channels. Accepted whole-shot closure authority remains blocked for EOS, ionization, electron heat flux/collisional coupling, radiation, impurity/electrode ablation, anomalous resistivity, restrike, and beam-target coupling. |
| FP-9 | Same-scope source availability decision | Source search complete for first pass; PF-1000/Akel remains engineering/reference candidate only | `docs/FIRST_PRINCIPLES_SAME_SCOPE_SOURCE_SEARCH_2026_05_15.md` finds that Akel/PF-1000 supports geometry, bank/drive, pressure, scalar current/yield, detector layout, and timing text for the 16 kV shot set, but does not provide accepted digitized current, same-shot density, fields, temperatures, neutron spectrum/anisotropy, detector-response, and propagated-UQ packets. The package-native runner now emits a fail-closed `same_scope_source` packet with channel-status, text-reference-only, cross-scope policy, and validation-target scope-decision fields. |
| FP-10 | Accepted waveform and phase evidence | Source search complete for first pass; accepted waveform/phase packet remains blocked by review | `docs/FIRST_PRINCIPLES_WAVEFORM_PHASE_SOURCE_SEARCH_2026_05_15.md` finds that Akel/PF-1000 supports measured waveform context, current-derivative dip time origin, breakdown-to-dip timing, constriction timing, timing uncertainty, shot 12581 current/pinch scalars, and Fig. 1-4 waveform existence, but accepted digitized current/derivative traces, per-point UQ, independent review, current-dip depth, typed phase targets, output mapping, and comparator tolerances remain blocked. The package-native runner now emits a fail-closed `waveform_phase` packet with draft Fig. 1 status, required review channels, per-channel status, and target-scope decisions. |
| FP-11 | Accepted spatial, field, and temperature evidence | Source search complete for first pass; accepted spatial/field/temperature packet remains blocked | `docs/FIRST_PRINCIPLES_SPATIAL_FIELD_TEMPERATURE_SOURCE_SEARCH_2026_05_15.md` finds that Akel/PF-1000 supports Lee-output density and pinch-geometry scalars for shot 12581, but accepted same-scope density histories, EM-field histories, electron/ion temperatures, diagnostic geometry/calibration, output mappings, comparator tolerances, and UQ remain blocked. Broader PF-1000 interferometry, magnetic-probe, imaging, and spectroscopy sources are requirement material unless their exact scope is selected as the demonstrator. The package-native runner now emits a fail-closed `spatial_field_temperature` packet. |
| FP-12 | Mechanism-separated neutron authority | Source search complete for first pass; accepted neutron-authority packet remains blocked | `docs/FIRST_PRINCIPLES_NEUTRON_AUTHORITY_SOURCE_SEARCH_2026_05_15.md` finds that PF-1000/Akel supports scalar yield, detector-layout text, activation calibration context, and Lee baseline mechanism text, while the new hybrid and fully kinetic sources define the resolved-ion and mechanism-separation requirements. Accepted total-yield authority remains blocked until thermonuclear and beam-target histories, ion/beam distributions, spectrum, anisotropy, detector/activation response, direct/scattered transport, comparator mapping, and UQ are all same-scope and reviewed. The package-native runner now emits a fail-closed `neutron_authority` packet. |
| FP-13 | Comparator and UQ matrix complete | Source search complete for first pass; accepted comparator/UQ matrix remains blocked | `docs/FIRST_PRINCIPLES_COMPARATOR_UQ_SOURCE_SEARCH_2026_05_15.md` finds that PF-1000/Akel supports scalar yield uncertainty, timing uncertainty, detector-layout text, activation-calibration text, and shot-series range context, while other local sources define numerical-sensitivity, detector-forward spectrum, direct/scattered neutron, and mechanism-error requirements. Accepted comparator authority remains blocked until every observable has accepted same-scope target evidence, output mapping, units/coordinates, metric, tolerance, measurement/model/numerical UQ, propagation, pass/fail rule, artifact hashes, requirement links, and independent review. The package-native runner now emits a fail-closed `comparator_uq` packet. |
| FP-14 | Validation certificate and release decision | Source search complete for first pass; accepted certificate remains blocked | `docs/FIRST_PRINCIPLES_CERTIFICATE_SOURCE_SEARCH_2026_05_15.md` finds that the project has a clear fail-closed certificate contract, but the accepted PF-1000/Akel first-principles certificate cannot be written while upstream packets remain candidate or blocked. The certificate path still requires manifest and packet hashes, accepted upstream same-scope/waveform/spatial/neutron/comparator/numerical/closure/power/startup/dimensionality evidence, reviewer metadata, metrics/UQ IDs, requirement links, command provenance, release label, release decision, and negative-test proof. The package-native runner now emits a fail-closed `certificate_gate` packet. |
| FP-15 | Generalized DPF-machine path | Source search complete for first pass; generalized claim remains blocked | `docs/FIRST_PRINCIPLES_GENERALIZATION_SOURCE_SEARCH_2026_05_15.md` finds candidate second scopes in PF-1000 full-energy diagnostics, FAETON-I, LLNL-like kinetic/hybrid references, MJOLNIR, and Akel shot/pressure series. None is accepted. A general DPF-machine claim remains blocked until one second device or shot repeats `FP-1` through `FP-14` with no hidden PF-1000/Akel assumptions. The package-native runner now emits a fail-closed `generalization` packet. |

The initial package-native FP-7 implementation surface is
`src/dpf/first_principles/{deck,runner,manifest,conservation}.py`, called by
`dpf first-principles-3d`. It produces `not_validation` engineering artifacts
until the same-scope acceptance packet exists. The detailed source-derived
FP-7 application map is
`docs/FIRST_PRINCIPLES_3D_HYBRID_PIC_REVIEW_2026_05_14.md`. Existing
readiness helpers can still report blockers, but the implementation authority
for the new 3D execution path is the package-native `dpf.first_principles`
runner and its fail-closed manifest.

## Workstream A: Package-Native Execution Path

Objective: make the first-principles simulator a real package capability, not an
app-only helper.

Current package-deck source search: see
`docs/FIRST_PRINCIPLES_PF1000_PACKAGE_DECK_SOURCE_SEARCH_2026_05_15.md`.
`dpf first-principles-3d` now defaults to a PF-1000/Akel 16 kV shot-12581
engineering deck built from the local source truth. This deck is a demonstrator
input surface only; it remains `engineering_candidate_not_validation` and
cannot promote while startup, limiter, numerical, power-port, dimensionality,
closure, same-scope, waveform, spatial/field/temperature, neutron,
comparator/UQ, certificate, and generalization packets remain blocked.

1. Move the accepted execution path into `src/dpf`, with `app_mhd.py` becoming a
   caller rather than the authority.
   - Initial 3D engineering implementation landed in
     `src/dpf/first_principles/runner.py` and is callable through
     `dpf first-principles-3d`.
2. Ensure CLI, API, config, and app all select the same package-native
   `first_principles_mhd` execution mode.
3. Separate active-model provenance from baseline reference metadata. Reduced
   Lee/snowplow factors may appear in comparison metadata, but must not be
   present as active-model closure inputs in an accepted run.
4. Emit one manifest schema for successful, blocked, and failed first-principles
   runs.

Acceptance evidence:

- Package-native runner and tests proving CLI/app/config select the same path.
- Run manifest showing active model inputs separately from baseline comparison
  inputs.
- Negative test proving a repo-root app-only run cannot receive an accepted
  certificate.

## Workstream B: Global Limiter And Repair Registry

Objective: replace hidden stability scaffolding with verified numerics or
source-backed physical bounds.

Current limiter-readiness blocker search: see
`docs/FIRST_PRINCIPLES_LIMITER_READINESS_SOURCE_SEARCH_2026_05_15.md`. Every
package-native first-principles run now must emit a `limiter_readiness` packet.
It may list finite runtime and conservation telemetry as candidate evidence, but
it must return `blocked_limiter_readiness_packet_not_available` until the active
path has a complete limiter inventory, classifications, activation counts,
before/after min/max, nonfinite counts, source or method justification,
readiness effects, source-backed physical bounds or verified numerical method
bounds, full-horizon zero-acceptance-blocker proof, fallback rejection tests,
artifact hashes, and independent review.

The registry must include every active-path limiter source:

- `app_mhd.py` field/state repair, density, pressure, temperature, velocity,
  magnetic field, magnetic energy, resistivity closure, timestep control,
  field power-port coupling, and any reintroduced floor/cap/clip path.
- `src/dpf/fluid/cylindrical_mhd.py` positivity floors, inter-stage velocity
  clamps, final velocity limiting, temperature caps, and any solver-internal
  nonfinite repair.
- backend-specific precision, sanitization, or fallback behavior.
- source-term subcycling, operator splitting, or positivity fixes.

For each limiter or bound, the artifact must include:

- limiter ID and code path;
- classification: `physical_bound`, `verified_numerical_method`,
  `debug_repair`, `engineering_guard`, or `acceptance_blocker`;
- before/after min, max, finite count, activation count, and affected field;
- justification source: local `KnowledgeReference/`, numerical verification
  packet, or blocker;
- readiness effect.

Acceptance evidence:

- Full-run limiter ledger with zero `acceptance_blocker` activations.
- Readiness gate test where a synthetic limiter activation blocks
  first-principles acceptance.
- Replacement plan for each remaining engineering guard.

## Workstream C: Numerical Verification Specification

Objective: make solver correctness measurable before physics validation.

Current numerical-fidelity blocker search: see
`docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_2026_05_15.md`.
Every package-native first-principles run now must emit a `numerical_fidelity`
packet. It may list runtime conservation, `div B`, and hybrid-loop diagnostics
as candidate telemetry, but it must return
`blocked_numerical_fidelity_packet_not_available` until the active solver path
has named test surfaces, source-backed methods, analytic/manufactured
references, mesh/time families, norms, tolerances, convergence evidence,
limiter-zero proof, artifact hashes, backend/precision scope, negative tests,
and independent review.

| Test surface | Required method | Acceptance fields |
| --- | --- | --- |
| Finite-volume shock behavior | 1D/2D analytic or manufactured shock fixtures using active reconstruction/Riemann path. | Mesh family, norm, observed order or monotonic convergence, tolerance, limiter ledger. |
| Cylindrical source terms | Cylindrical conservation and geometry fixtures, including annular PF-1000/Akel grid. | Mass, momentum, energy residuals, coordinate convention, cell-volume proof. |
| `div B` control | Analytic field and evolved-field checks. | Initial and final `div B` norms, cleaning method, tolerance, failure mode. |
| Resistive diffusion | Analytic diffusion or manufactured-solution fixture with active resistivity path. | Diffusion error norms, timestep sensitivity, source-term split metadata. |
| Joule heating and total energy | Closed-box and circuit-coupled energy ledgers. | Component energy histories, residual budget, sign convention, no hidden repair. |
| Circuit power-port coupling | Standalone power-port tests plus integrated plasma/circuit tests. | Poynting or `J.E`, electrode work, time-centering, residual tolerance. |
| Restart and reproducibility | Restart at multiple times and rerun. | Bitwise or tolerance-bounded field/history differences, manifest hash. |
| Backend and precision parity | Reference workflow versus GPU/MLX/Metal preview path. | Norms by observable, unsupported physics list, promotion decision. |
| Limiter-zero acceptance | Run with limiter registry active. | Zero acceptance-blocking limiter activations. |

No numerical packet can pass with unspecified tolerances. If a tolerance is not
defended by an analytic fixture, convergence study, or source-backed UQ
allocation, the numerical packet remains blocked.

## Workstream D: Startup Boundary-Value Problem

Objective: replace seeded-layer startup with a source-backed initial and
boundary state.

Current blocker search: see
`docs/FIRST_PRINCIPLES_BLOCKER_SOURCE_SEARCH_2026_05_15.md`. The local corpus
supports a four-way startup distinction:

- `imported_pic_sheath_state`: accepted only after reviewed local field,
  particle, current, temperature, boundary, unit, and conservation evidence is
  present.
- `source_backed_end_rundown_sheath`: engineering candidate supported by the
  hybrid PIC-fluid source, but not a full breakdown-to-liftoff startup model.
- `surface_breakdown_bvp`: the required whole-shot path, currently blocked until
  equations, material/secondary-emission, avalanche/streamer, preionization,
  pressure-regime, and electrode/insulator boundary evidence are present.
- `seeded_layer`: rejected for accepted first-principles claims.

The startup generator must produce:

- density and species or ionization state;
- electron and ion temperatures or a stated single-temperature limit;
- pressure, velocity, magnetic field, current density, and resistivity;
- electrode, cathode, anode, and insulator boundary metadata;
- circuit initial derivative or boundary drive consistency;
- `div B` and unit checks;
- sheath-liftoff and early axial-run evidence status.

Acceptance evidence:

- Startup evidence packet with explicit startup mode, local source paths,
  hashes, pages/lines/figures,
  units, assumptions, and review status.
- Package-native `startup_bvp` packet in run telemetry and manifest, with
  upstream propagation into numerical, comparator, certificate, and
  generalization gates.
- BVP consistency tests for current, magnetic field, boundary conditions, units,
  and `div B`.
- Negative test proving seeded-layer startup blocks accepted first-principles
  claims.

## Workstream E: Resolved Power-Port Coupling

Objective: make circuit-plasma feedback a conservation-law power exchange.

Current blocker search: see
`docs/FIRST_PRINCIPLES_POWER_PORT_SOURCE_SEARCH_2026_05_15.md`. The local
source-truth contract is that terminal voltage/load authority comes from field
power, through a named Poynting surface flux or equivalent volume `J.E` relation.
`L_field = 2 E_B/I^2` remains diagnostic-only unless a reviewed packet proves
load equivalence for the claimed interval.

The accepted circuit update must include:

- a named interface surface or volume relation;
- Poynting flux or equivalent `J.E` power;
- electrode work and external circuit energy;
- Joule heating, magnetic energy, kinetic/thermal energy, radiation losses, and
  residual;
- time-centering or subcycling metadata;
- sign convention;
- no clipped back-EMF or hidden current-floor behavior for acceptance;
- handoff metadata if any engineering startup interval remains.
- active-load metadata separating the load used by the circuit update from
  diagnostic alternatives.

`L_field = 2 E_B/I^2` may remain a diagnostic. It is not the circuit load unless
a reviewed power-port packet proves that use for the claimed interval.

Acceptance evidence:

- Standalone component tests for power-port sign and energy conservation.
- Package-native power-port packet with per-step current/voltage, active-load
  relation, candidate energy ledger terms, diagnostic-only field inductance,
  startup-handoff blocker, source references, and upstream certificate linkage.
- Integrated PF-1000/Akel run with bounded residual and no fallback to
  snowplow/Lee load after startup.
- Same-scope waveform comparator only after evidence and UQ pass.

## Workstream F: Dimensionality And Kinetic Handoff

Objective: prevent a 2D MHD engineering model from being overstated as a
complete DPF simulation, and define the full 3D hybrid PIC-fluid finish-line
core.

Current blocker search: see
`docs/FIRST_PRINCIPLES_DIMENSIONALITY_SOURCE_SEARCH_2026_05_15.md`. The source
truth requires a claim-mode decision: axisymmetric/MHD paths are narrowed
interim scopes, 3D MHD can support macroscopic rundown/electrode-geometry
claims after review, MHD must hand off before kinetic pinch authority, and
unrestricted beam-target neutron authority requires a kinetic interval or a
reviewed hybrid/kinetic handoff with mechanism separation.

The plan must make one of these accepted decisions:

- `bounded_axisymmetric_mhd_claim`: the accepted claim stops before local
  source-backed MHD breakdown and excludes observables requiring 3D/kinetic
  dynamics. This is allowed only for interim comparator/scaffold releases, not
  for the `/goal` full first-principles DPF simulator.
- `validated_3d_mhd_claim`: the accepted claim uses a 3D MHD path for observables
  requiring non-axisymmetric structure.
- `mhd_kinetic_handoff_claim`: the accepted claim hands resolved MHD output to a
  kinetic/hybrid model for late-pinch, beam, spectrum, anisotropy, and total
  neutron-yield observables.
- `validated_3d_hybrid_pic_fluid_claim`: the accepted claim is produced by a
  3D ion-PIC/electron-fluid/full-Maxwell loop with reviewed evidence for every
  capability in `hybrid_pic_3d_first_principles_core`.

Acceptance evidence:

- Dimensionality decision packet with local source links and claimed interval.
- Tests proving out-of-scope observables cannot be marked accepted.
- Handoff packet with fields, particle source terms, energy/momentum transfer,
  and conservation checks when kinetic/hybrid physics is used.
- Source-derived 3D hybrid PIC-fluid capability packet from
  `src/dpf/validation/hybrid_pic_3d.py`, including accepted evidence for
  Maxwell vacuum/plasma fields, ion PIC push/deposition, electron-fluid Ohm
  closure, current predictor-corrector, source-ordered loop execution,
  divergence control, plasma-vacuum conductivity, PML/conductor/particle
  boundaries, external-circuit magnetic boundary drive, collisions, 3D
  dimensionality, electron-energy closure, kinetic yield history, and same-scope
  validation.

Current implementation ratchet, 2026-05-15:

- `src/dpf/fields/maxwell_3d.py` now provides the first isolated 3D full-Maxwell
  field component on the repo's Yee/CT layout: edge-centered electric fields,
  face-centered magnetic fields, Ampere/Faraday updates, conductor electric
  masks, deterministic PML damping metadata, divergence diagnostics, Courant
  timestep, and EM energy accounting.
- `src/dpf/fields/pic_coupling.py` now provides a candidate
  `PICCurrentSourcePort` that maps cell-centered PIC current deposition onto
  Yee edge currents for Ampere's law, with continuity telemetry kept
  `measured_not_accepted` or blocked when required inputs are absent.
- `src/dpf/fields/ohm_solver.py` now provides the first cell-centered
  generalized Ohm-Ampere algebraic current solve with resistive, Hall, and
  density-thresholded pressure-gradient terms traced to the new source.
- `src/dpf/fields/predictor_corrector.py` now provides the source linear
  current extrapolation `J*_{n+1}=2J_{n+1/2}-J_n` and the end-step generalized
  Ohm correction around a supplied provisional ion current.
- `src/dpf/fields/marder.py` now provides a candidate Marder/Gauss-law
  correction `E <- E + d grad(div E - rho/epsilon0)` with residual and
  nondominance telemetry. The candidate field stepper can map this
  cell-centered correction back to Yee electric edges and reapply field
  boundaries.
- `src/dpf/fields/conductivity.py` now provides the source plasma-vacuum
  conductivity transition and Ohmic CFL cap with active-fraction telemetry.
- `src/dpf/fields/hybrid_stepper.py` now integrates the field-current slice for
  one candidate step: cell-centered fields are derived from the Yee state,
  source conductivity is blended, generalized Ohm current is solved, current is
  mapped back to Yee edges, Maxwell is advanced, and optional end-step
  predictor-corrector current telemetry can be produced from the next fields.
- `src/dpf/fields/hybrid_loop.py` now performs the first candidate
  particle-field loop step: convert Yee fields to cell centers, push HybridPIC
  ions, deposit Esirkepov/CIC current, rebuild quasi-neutral electron density
  from deposited ion charge, and feed that current into the field-current
  stepper.
- `HybridPIC3DLoop` also now has a candidate source-ordered update mode:
  advance positions from stored half-step velocities, deposit current from
  `x_n` to `x_{n+1}`, optionally use half-step charge deposition for density,
  run the Ohm/Maxwell/Marder/predictor field-current path, update ion
  velocities from source Eq. 7, and apply configured collisions only after that
  velocity update. It also emits candidate predictor-corrector
  particle-rebuild telemetry by estimating provisional ion velocities and
  provisional ion current from the particle state, then feeds that provisional
  ion current into the candidate end-step Ohm correction.
- `src/dpf/fields/particle_boundaries.py` now provides candidate particle
  absorption for the source rule that particles entering conductor or PML
  regions are absorbed and deleted. `HybridPIC3DLoop` can invoke it before
  deposition so removed particles do not contribute charge/current to the
  field-current step.
- `HybridPIC3DLoop` now reports disabled versus Nanbu/Perez-enabled ion
  collision status from the existing `HybridPIC` collision kernel as
  source-traced candidate telemetry. Collision parameterization and cell-local
  same-scope DPF validation remain unclosed.
- `src/dpf/fields/electron_energy.py` now wraps the repo two-temperature
  scaffold as candidate 3D electron-energy telemetry. `HybridPIC3DLoop` can use
  a supplied separate electron-energy state to build the electron pressure
  gradient and then update `Te` from the solved current, resistivity,
  collisional equilibration, and bremsstrahlung terms. Extended-Ohm runs now
  also carry fail-closed temperature-authority telemetry: Hall/pressure claims
  remain blocked without accepted separate-Te evidence, and the current
  candidate Te scaffold does not promote.
- `src/dpf/fields/kinetic_yield.py` now accumulates candidate D-D neutron-yield
  history from PIC ion distributions. `HybridPIC3DLoop` can attach the
  instantaneous particle-distribution yield rate and cumulative neutron count
  to loop telemetry. A separate authority check blocks total-yield acceptance
  unless accepted kinetic history, mechanism separation, detector response, UQ,
  and electron-temperature authority are all attached.
- `src/dpf/first_principles/neutron_authority.py` now wraps the package-native
  runner's neutron status into a source-traced FP-12 packet. It exposes
  candidate PIC ion-yield telemetry and Akel scalar/detector text context, but
  blocks total-yield authority until mechanism-separated, same-scope neutron,
  detector-response, direct/scattered transport, comparator, and UQ evidence are
  attached.
- `src/dpf/first_principles/comparator_uq.py` now wraps FP-13 as a source-traced
  comparator/UQ matrix packet. It exposes text-supported Akel yield/timing
  uncertainty context and upstream packet statuses, but blocks comparator
  authority until every observable has accepted targets, output mapping,
  metrics, tolerances, UQ propagation, pass/fail rules, artifact hashes,
  requirement links, and independent review.
- `src/dpf/first_principles/numerical_fidelity.py` now wraps FP-4 as a
  source-traced numerical-fidelity gate. It exposes candidate conservation,
  `div B`, and hybrid-loop runtime channels plus existing candidate test
  surfaces, but blocks numerical acceptance until all named test surfaces carry
  tolerances, convergence, limiter-zero proof, backend/precision scope,
  artifact hashes, negative tests, and independent review.
- `src/dpf/first_principles/limiter_readiness.py` now wraps FP-2/FP-3 as a
  source-traced limiter-readiness gate. It exposes finite conservation and
  runtime telemetry as candidate evidence only, and blocks accepted
  limiter-zero or physical-bound claims until the active-path inventory,
  classifications, full-horizon proof, fallback rejection tests, hashes, and
  review are attached.
- `src/dpf/first_principles/startup_bvp.py` now wraps FP-5 as a source-traced
  startup gate. It emits accepted-mode, engineering-only-mode, rejected-mode,
  required-channel, candidate-input, and source-reference fields into the
  package-native runner and blocks whole-shot claims for end-of-rundown,
  seeded, legacy uniform/profile, or unreviewed startup states.
- `src/dpf/first_principles/power_port.py` now wraps FP-6 as a source-traced
  power-port gate. It records candidate terminal current/voltage, active-load
  placeholder metadata, energy-ledger terms, startup-handoff blocker, and
  diagnostic-only `L_field = 2 E_B/I^2`; accepted power authority still blocks
  until named Poynting or `J.E`, sign, centering, electrode-work, residual, and
  review packets exist.
- `src/dpf/first_principles/dimensionality.py` now wraps FP-7 as a source-traced
  dimensionality/handoff gate. It records the allowed claim modes, active 3D
  hybrid candidate status, source-model limitations, candidate runtime channels,
  and observables that require kinetic handoff or fully kinetic authority.
- `src/dpf/first_principles/closure_packet.py` now wraps FP-8 as a source-traced
  closure matrix. It records required packet channels, per-effect
  classifications, active candidate closures, review status, claim impact, and
  candidate-only runtime closure channels while blocking every unaccepted effect.
- `src/dpf/first_principles/same_scope.py` now wraps FP-9 as a source-traced
  same-scope gate. It records reference-only text channels, per-channel status,
  other-scope requirement sources, reviewed-transfer-rule requirements, and
  validation-target scope decisions.
- `src/dpf/first_principles/waveform_phase.py` now wraps FP-10 as a
  source-traced waveform/phase gate. It records reference-only text support,
  Akel Fig. 1 draft digitization status, required review channels, per-channel
  status, and target-scope decisions while blocking waveform/phase comparators.
- `src/dpf/first_principles/certificate_gate.py` now wraps FP-14 as a
  source-traced release gate. It carries upstream packet statuses and blocks
  accepted certificate writing until manifest hashes, evidence packet hashes,
  accepted upstream packets, reviewer metadata, metrics/UQ IDs, requirement
  links, command provenance, release decision, and negative-test proof are all
  attached.
- `src/dpf/first_principles/generalization.py` now wraps FP-15 as a
  source-traced generalized DPF-machine gate. It lists candidate second scopes
  from the local source truth, carries upstream packet statuses, and blocks any
  generalized DPF-machine claim until a second device or shot repeats every
  evidence packet and certificate gate without hidden PF-1000/Akel assumptions.
- `src/dpf/fields/hybrid_simulator.py` now provides a compact candidate
  multi-step 3D hybrid PIC-fluid driver that carries Maxwell state, PIC state,
  optional electron-energy state, optional external-circuit boundary state, and
  telemetry forward across repeated steps.
- `src/dpf/fields/source_geometry.py` now records the local source's LLNL-like
  axisymmetric setup values as a typed blocked geometry packet and can derive a
  Cartesian smoke grid for engineering exercise only.
- `src/dpf/validation/hybrid_pic_3d_validation_packet.py` now provides the
  final same-scope validation packet gate. It wraps
  `hybrid_pic_3d_first_principles_core` and additionally requires accepted
  target, detector-response, UQ, conservation, nondominance, and backend-scaling
  packets.
- `dpf hybrid-3d-smoke` now exposes the candidate 3D hybrid PIC-fluid path as a
  runnable CLI tool. It writes a JSON engineering artifact with source-ordered
  loop telemetry, circuit-boundary telemetry, Te/yield authority status, and
  the blocked same-scope validation packet.
- `src/dpf/fields/circuit_boundary.py` now provides a source-scoped candidate
  external-circuit magnetic boundary slice: explicit RLC current/charge
  stepping from Eq. 37-38 and `B_theta = mu0 I/(2 pi r)` from Eq. 34 projected
  onto the Cartesian 3D Maxwell grid at an injection plane. The multi-step
  simulator can apply this boundary and advance the circuit state when
  explicitly requested.
- The new component tests verify zero-field stability, CT preservation of
  `div B`, Ampere response to `curl B`, conductor suppression of adjacent
  electric edges, PML energy removal, HybridPIC-deposit current mapping,
  generalized Ohm residual closure, predictor-corrector current extrapolation,
  end-step Ohm residual closure, smooth Marder residual reduction, Marder
  nondominance/dominance flagging, and source conductivity blending/CFL
  limiting, one integrated Ohmic field-current
  step, one particle-field push/deposit/field step, candidate conductor/PML
  particle deletion before deposition, collision telemetry, optional
  electron-energy coupling, kinetic-yield accumulation, a three-step candidate
  simulator run, source-geometry packet blocking, RLC/magnetic-boundary formula
  and injection-plane/simulator-coupling behavior, source-ordered Eq. 7 loop
  behavior, extended-Ohm Te authority blocking, and fail-closed readiness
  behavior, kinetic total-yield authority blocking, and same-scope validation
  packet gating, candidate predictor-particle rebuild telemetry, including CLI
  artifact generation for the 3D smoke tool.
- This ratchet may satisfy engineering evidence for individual components only
  when explicitly attached through `hybrid_pic_3d_evidence`. It does not
  complete FP-7, because accepted long-run ion PIC self-consistency, accepted Ohm-loop
  integration, accepted provisional particle-push/rebuild coupling,
  accepted source-ordered Te/Ti and predictor-corrector
  conservation/stability evidence,
  accepted nondominant Gauss-law/Marder control against sheath/current
  observables,
  accepted weakly active plasma-vacuum conductivity, accepted electrode and boundary-validation packets,
  accepted external-circuit `U_DPF` closure and injection-port validation,
  accepted provisional ion-push/rebuild predictor-corrector coupling, accepted collision parameterization, accepted electron-energy heat-flux/collisional coupling and temperature diagnostics, accepted mechanism-separated kinetic yield authority with detector response and UQ, and same-scope 3D
  validation remain missing. The Ohm component is especially nonaccepting until
  it is part of a particle-coupled Yee/PIC loop and a separate
  electron-temperature closure supports pressure/Hall claims.

## Workstream G: Physics Closure Packets

Objective: turn named physics features into auditable closures or explicit
bounds.

Current blocker search: see
`docs/FIRST_PRINCIPLES_CLOSURE_SOURCE_SEARCH_2026_05_15.md`. The local corpus
supports several engineering-candidate closures, but the accepted whole-shot
packet remains blocked until every active or bounded-out effect carries source
equations, symbol map, units, validity regime, verification, sensitivity/UQ, and
claim impact.

Each row below must have source equations or bounded-out rationale, symbol map,
units, validity regime, verification test, validation or bound evidence,
sensitivity/UQ, and claim impact.

| Effect | Accepted status required before PF-1000/Akel certificate |
| --- | --- |
| EOS and thermodynamics | Implemented and verified, or bounded for the accepted interval. |
| Ionization and charge state | Implemented or bounded with source-backed startup and resistivity impact. |
| Single-fluid/two-temperature energy | Explicit model decision with electron/ion energy exchange status. |
| Electrical and thermal transport | Source-backed conductivity/resistivity and conduction regime. |
| Radiation losses | Implemented or bounded with loss contribution in energy ledger. |
| Impurity and electrode ablation | Implemented or bounded for waveform/pinch/neutron observables. |
| Hall, FLR, kinetic scope | Implemented or bounded; if required by observables, handoff is mandatory. |
| 3D instabilities | Implemented or claim interval/observables exclude them. |
| Restrike and anomalous resistance | Implemented or bounded for current dip/post-pinch claims. |
| Beam-target coupling | Kinetic/hybrid implementation required for total neutron-yield authority. |

## Workstream H: Neutron And Beam-Target Authority

Objective: move total neutron yield from reduced estimates to
mechanism-separated first-principles prediction.

Current blocker search: see
`docs/FIRST_PRINCIPLES_NEUTRON_AUTHORITY_SOURCE_SEARCH_2026_05_15.md`. The local
source truth supports the requirement that neutron authority be
mechanism-separated, spectrum-aware, detector-forward, and UQ-bearing. It does
not yet support an accepted PF-1000/Akel total-yield claim. Every
package-native first-principles run now must emit a `neutron_authority` packet,
and that packet must remain
`blocked_mechanism_separated_neutron_authority_not_available` until all required
same-scope channels pass review.

Required gates:

1. Thermonuclear history from resolved density, temperature, volume, and time
   histories, with reactivity validity and units verified.
2. Kinetic or hybrid beam generation with source-backed acceleration mechanism,
   energy distribution, angular distribution, current/particle normalization, and
   time history.
3. Beam transport/stopping and target coupling with density, path length,
   cross-section, and energy loss.
4. Spectrum and anisotropy calculation.
5. Detector or activation response model, calibration, and UQ.
6. Same-scope scalar yield, timing, spectrum, anisotropy, detector response, and
   UQ comparator pass.
7. Direct/scattered neutron transport and detector impulse/efficiency/geometry
   response must be attached before spectrum or angular evidence can enter a
   certificate.

Lee/Saw beam-target estimates, empirical beam fractions, and final-state
thermonuclear duration approximations are baseline comparisons only.

## Workstream I: Same-Scope Evidence And UQ

Objective: turn local sources into accepted comparison data for the accepted run.

Current blocker search: see
`docs/FIRST_PRINCIPLES_SAME_SCOPE_SOURCE_SEARCH_2026_05_15.md`. The local
source set does not yet contain a complete accepted PF-1000/Akel same-scope
packet. PF-1000/Akel remains the engineering/reference candidate, not an
accepted whole-shot demonstrator, until accepted evidence is acquired for the
missing waveform, startup, density, field, temperature, neutron, detector, and
UQ channels.

Every package-native first-principles run now must emit a `same_scope_source`
packet. The packet can list text-supported Akel/PF-1000 channels, but it must
return `blocked_same_scope_source_packet_not_available` unless the same source
scope contains accepted evidence for all required channels below.

Current waveform/phase blocker search: see
`docs/FIRST_PRINCIPLES_WAVEFORM_PHASE_SOURCE_SEARCH_2026_05_15.md`. Every
package-native run now must also emit a `waveform_phase` packet. It may list
Akel text-supported timing and scalar current context, but it must return
`blocked_waveform_phase_packet_not_available` until accepted digitized current
and derivative traces, phase targets, output mappings, comparator tolerances,
and UQ are attached.

Current spatial/field/temperature blocker search: see
`docs/FIRST_PRINCIPLES_SPATIAL_FIELD_TEMPERATURE_SOURCE_SEARCH_2026_05_15.md`.
Every package-native run now must emit a `spatial_field_temperature` packet. It
may list Akel Lee-output density and pinch-geometry scalars, but it must return
`blocked_spatial_field_temperature_packet_not_available` until accepted
same-scope density, EM-field, electron/ion temperature, output-mapping,
comparator, and UQ packets are attached.

Current neutron-authority blocker search: see
`docs/FIRST_PRINCIPLES_NEUTRON_AUTHORITY_SOURCE_SEARCH_2026_05_15.md`. Every
package-native run now must emit a `neutron_authority` packet. It may list Akel
scalar yield and detector-layout text plus candidate runtime PIC ion-yield
diagnostics, but it must return
`blocked_mechanism_separated_neutron_authority_not_available` until accepted
same-scope mechanism-separated yield, spectrum, anisotropy, detector/activation
response, transport, comparator, and UQ packets are attached.

Current comparator/UQ blocker search: see
`docs/FIRST_PRINCIPLES_COMPARATOR_UQ_SOURCE_SEARCH_2026_05_15.md`. Every
package-native run now must emit a `comparator_uq` packet. It may list Akel
scalar-yield and timing uncertainty text plus other-scope numerical and
detector-response methodology, but it must return
`blocked_comparator_uq_matrix_not_available` until accepted target evidence,
output mapping, metric, tolerance, measurement/model/numerical UQ, propagation,
pass/fail, artifact-hash, requirement-link, negative-control, and independent
review channels are attached for every observable group.

| Observable group | Output field or artifact | Evidence state required | Comparator and UQ requirement |
| --- | --- | --- | --- |
| Current waveform | `I_MA`/current history | Accepted waveform packet | Time alignment, amplitude metric, NRMSE or uncertainty-aware residual. |
| Current dip | waveform/dip diagnostics | Accepted current-dip packet | Dip timing and magnitude with propagated waveform UQ. |
| Phase timing | phase/sheath history | Accepted phase packet | Axial/radial/pinch timing metrics and uncertainty. |
| Spatial density | field snapshots | Accepted spatial density packet | Spatial registration, interpolation rule, diagnostic UQ. |
| Magnetic/EM field | `B` snapshots and diagnostics | Accepted field packet | Field component mapping, geometry, calibration UQ. |
| Temperature | `Te`/`Ti` or model output | Accepted temperature packet | Species/model mapping and measurement UQ. |
| Field coupling | power-port ledger | Accepted coupling packet | Poynting or `J.E`, energy residual, waveform relation. |
| Neutron scalar yield | mechanism-separated neutron output | Accepted yield packet | Relative error including detector and shot/UQ. |
| Neutron timing | neutron history | Accepted timing packet | Pulse timing/width and mechanism separation. |
| Spectrum | spectrum artifact | Accepted spectrum packet | Energy-bin metric and detector response. |
| Anisotropy | angular yield artifact | Accepted anisotropy packet | Angular metric and calibration UQ. |
| Detector/activation | detector response artifact | Accepted detector packet | Forward response, calibration, uncertainty. |
| Numerical fidelity | numerical packet | Accepted numerical packet | All numerical tests pass with declared tolerances. |
| Physics fidelity | closure packet matrix | Accepted closure packet | All effects implemented, validated, or bounded out. |

If PF-1000/Akel cannot supply the required same-scope evidence, the plan must
choose one of these outcomes before certificate work continues:

- narrow the accepted claim to the evidence-supported interval and observables;
- keep PF-1000/Akel as engineering/reference-candidate only;
- switch the first accepted demonstrator to a better-diagnosed local source
  scope.

## Workstream J: Traceability, Certificate, And Release Gate

Objective: make the final claim auditable from requirement to source to run
artifact.

Current certificate blocker search: see
`docs/FIRST_PRINCIPLES_CERTIFICATE_SOURCE_SEARCH_2026_05_15.md`. Every
package-native first-principles run now must emit a `certificate_gate` packet.
It must return `blocked_first_principles_certificate_not_available` until every
upstream packet is accepted and every certificate payload, review, release, and
negative-test channel is attached. While blocked, the release label remains
`engineering_candidate_not_releasable_for_first_principles_claim`.

Certificate payload must include:

- run manifest path and hash;
- package-native runner version and command;
- validation scope and source scope;
- evidence packet IDs and hashes;
- reviewer identities, review dates, decisions, and defect resolutions;
- comparator metrics, UQ packet IDs, and pass/fail rules;
- numerical-fidelity packet ID and tolerances;
- physics-closure packet IDs and statuses;
- limiter ledger summary and zero acceptance-blocker proof;
- dimensionality/handoff decision;
- neutron mechanism authority packet;
- linked requirement IDs;
- release label;
- negative-test evidence for draft, blocked, cross-scope, missing-UQ,
  missing-review, hidden-limiter, app-only, and reduced-model fallback cases.

## Workstream K: Generalized DPF-Machine Path

Objective: prove the tool is a first-principles DPF simulator, not a
PF-1000/Akel-only engineering exercise.

Current generalization blocker search: see
`docs/FIRST_PRINCIPLES_GENERALIZATION_SOURCE_SEARCH_2026_05_15.md`. Every
package-native first-principles run now must emit a `generalization` packet. It
must return `blocked_generalized_dpf_machine_path_not_available` until a second
device or shot repeats `FP-1` through `FP-14` with typed evidence, review, UQ,
and a separate certificate. While blocked, the release label remains
`single_scope_engineering_candidate_not_generalized`.

Candidate second scopes from the local source truth:

- PF-1000 full-energy anisotropy/interferometry scope
  (`450-500 kJ`, `3.5 Torr`): strong density, pinch, TOF, direct/scattered, and
  anisotropy context; not Akel shot 12581.
- FAETON-I 100 kV scope: distinct high-voltage DPF with current sheath,
  voltage, neutron-yield, anisotropy, PMT-scintillator, and Faraday-cup context;
  must be separated from Lee-model baseline use.
- LLNL 180 kA kinetic/hybrid reference: strong for kinetic beam-target and
  architecture requirements; current same-scope public experimental packet is
  incomplete.
- MJOLNIR 60 kV, 735 kJ, 9 Torr mechanism scope: strong for mechanism timing,
  spectrum, anisotropy, activation, and MHD-to-kinetic context; requires full
  packet extraction and review.
- PF-1000/Akel second shot or pressure series: useful for reproducibility, but
  insufficient for cross-device generality by itself.

Generalization certificate must include:

- accepted primary-scope certificate;
- declared second scope;
- second-scope geometry, drive waveform, startup, power port, dimensionality,
  physics closure, density/field/temperature, neutron authority, detector/UQ,
  comparator/UQ, numerical fidelity, and certificate packets;
- proof that no PF-1000/Akel constants, geometry assumptions, calibration,
  closure tolerances, limiter rules, or comparator tolerances are hidden in the
  generic path;
- device parameterization schema;
- scale-transition or nondimensionalization review;
- regression against the first accepted scope;
- source review certificate;
- cross-scope negative tests.

## Requirements Map

| ID | Priority | Owner | Requirement | Status | Verification | Acceptance evidence or blocker |
| --- | --- | --- | --- | --- | --- | --- |
| DPF-PHYS-008 | P0 | Physics/Engine | Circuit feedback for first-principles mode shall be driven by resolved field power and conservation ledgers, not Lee/RADPF closure factors. | partial | test, analysis | The Python first-principles candidate now drives circuit feedback through resolved field-load power and an implicit-midpoint power port; accepted same-scope field-coupling validation remains blocked. |
| DPF-PHYS-009 | P0 | Physics/Engine | First-principles PF-1000/Akel runs shall reject hidden engineering limiters for accepted claims. | partial | test, inspection | First limiter-readiness source search is recorded in `docs/FIRST_PRINCIPLES_LIMITER_READINESS_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed `limiter_readiness` packet. App-level ledger, Python cylindrical state-mutating limiter telemetry, nonblocking Python PLM/HLL/CFL/implicit-resistive/circuit-resolution method records, uncapped source-traced resistivity, partial-ionization pressure/electron-density bookkeeping, per-step timestep-controller diagnostics, bounded Python limiter-clear probes through 1.0 us, CLI/manifest summaries, and backend-scope rejection are wired; accepted package-native inventory, full-horizon zero-acceptance-blocker proof, source-backed bounds, fallback rejection tests, hashes, and review remain blocked. |
| DPF-PHYS-010 | P0 | Physics/V&V | First-principles startup shall use source-backed breakdown, preionization, electrode boundary, and initial plasma evidence. | blocked | review, analysis, test | First blocker source search is recorded in `docs/FIRST_PRINCIPLES_BLOCKER_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed `startup_bvp` packet. End-of-rundown sheath initialization can be an engineering candidate, but accepted whole-shot startup still requires reviewed imported PIC startup state or a surface-breakdown BVP. |
| DPF-PHYS-011 | P0 | Physics/V&V | Field-circuit coupling shall carry validated inductance, terminal voltage/back-EMF, Poynting or `J.E` power, handoff intervals, and energy residual evidence. | partial | analysis, test | First power-port source search is recorded in `docs/FIRST_PRINCIPLES_POWER_PORT_SOURCE_SEARCH_2026_05_15.md`; field-power authority is source-supported, but accepted packet evidence remains blocked. |
| DPF-PHYS-012 | P0 | Physics/V&V | First-principles physics fidelity shall classify each required effect as implemented, validated, bounded out, blocked, or missing for the claimed scope. | partial | inspection, test | First closure source search is recorded in `docs/FIRST_PRINCIPLES_CLOSURE_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed closure matrix with per-effect classifications and required packet channels, but accepted physics-fidelity authority remains blocked. |
| DPF-PHYS-013 | P0 | Physics/V&V | Total neutron-yield authority shall require resolved thermonuclear history, accepted kinetic/hybrid beam-target production, same-scope neutron evidence, and UQ. | blocked | analysis, review, test | First neutron-authority source search is recorded in `docs/FIRST_PRINCIPLES_NEUTRON_AUTHORITY_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed `neutron_authority` packet, but accepted mechanism-separated histories, detector response, direct/scattered transport, spectrum, anisotropy, comparator mapping, and UQ remain blocked. |
| DPF-PHYS-014 | P0 | Physics/Architecture | First-principles acceptance shall define dimensionality and any MHD-to-kinetic handoff for the claimed interval and observables. | blocked | analysis, test, review | First dimensionality source search is recorded in `docs/FIRST_PRINCIPLES_DIMENSIONALITY_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed dimensionality/handoff packet with claim modes and source-model limitations, but accepted claims still require source-equivalence review, same-scope 3D validation, MHD-to-kinetic transfer if used, electron-kinetic scope disposition, and mechanism-separated detector/UQ evidence. |
| DPF-PHYS-015 | P0 | Numerics/V&V | First-principles numerical-fidelity packets shall define named tests, norms, mesh families, tolerances, precision/backend scope, and limiter-zero acceptance. | blocked | test, analysis | First numerical-fidelity source search is recorded in `docs/FIRST_PRINCIPLES_NUMERICAL_FIDELITY_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed `numerical_fidelity` packet, but accepted tolerances, convergence evidence, limiter-zero proof, backend/precision scope, artifact hashes, negative tests, and review remain blocked. |
| DPF-PHYS-016 | P0 | Physics/Engine | The active first-principles circuit power port shall pass Poynting or `J.E`, electrode-work, time-centering, sign, and residual tests without clipped back-EMF for acceptance. | partial | test, analysis | The active Python path now emits a fail-closed power-port packet with terminal current/voltage, active placeholder load, candidate energy ledger, startup-handoff blocker, and diagnostic-only field inductance. Accepted authority still requires named Poynting or `J.E`, electrode-work partition, sign convention, centering, residual tests, artifact hashes, and review. |
| DPF-PHYS-017 | P0 | Physics/Engine | First-principles startup shall be generated as a source-backed boundary-value problem with current-density, field, ionization, temperature, and sheath-liftoff consistency checks. | blocked | test, review, analysis | Startup modes now distinguish `imported_pic_sheath_state`, `source_backed_end_rundown_sheath`, blocked `surface_breakdown_bvp`, and rejected `seeded_layer` in package-native runner telemetry and manifest. Accepted BVP evidence remains blocked until the complete reviewed startup payload is attached. |
| DPF-PHYS-018 | P0 | Physics/V&V | Every active or bounded-out physical closure shall have a packet with source equations, symbol map, units, validity regime, verification, sensitivity/UQ, and claim impact. | blocked | inspection, analysis, test | Closure search now defines the required effect matrix and the runner emits a fail-closed closure packet; accepted authority remains blocked until EOS, ionization, electron-energy, transport, radiation, impurity/material, Hall/FLR/kinetic, 3D-instability, restrike, and beam-target records are complete and reviewed. |
| DPF-PHYS-019 | P1 | Architecture/Engine | Accepted first-principles execution shall run through one package-native `src/dpf` path shared by CLI, API, config, and app surfaces. | planned | test, inspection | Current runnable tool is app-backed. |
| DPF-VV-011 | P0 | V&V/Data | Every target, curve, table, formula, uncertainty value, comparator, and same-scope packet shall be typed evidence with local source provenance. | planned | test, inspection | Canonical evidence schema remains planned. |
| DPF-VV-012 | P0 | V&V/Data | Digitized figure or table evidence shall require independent accepted review before validation use. | partial | test, review | Akel Fig. 1 remains `blocked_by_review`. |
| DPF-VV-013 | P0 | V&V/Physics | Source-closed coded formulas shall carry formula evidence packets. | planned | analysis, test | Formula registry remains planned. |
| DPF-VV-014 | P0 | V&V/UQ | Quantitative validation shall require uncertainty extraction and propagation. | blocked | analysis, test | First comparator/UQ source search is recorded in `docs/FIRST_PRINCIPLES_COMPARATOR_UQ_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed `comparator_uq` packet, but accepted measurement/model/numerical uncertainty and propagation remain missing. |
| DPF-VV-015 | P0 | V&V/Physics | Accepted targets shall be bound to simulation outputs through tested comparators. | blocked | test, analysis | The runner now blocks without output field mapping, units/coordinates, comparator metric, comparator tolerance, pass/fail rule, artifact hashes, requirement links, and review for every observable group. |
| DPF-VV-016 | P0 | V&V/Physics | Same-scope packet assembly shall reject cross-device, cross-shot, or cross-configuration evidence unless a reviewed transfer rule exists. | planned | test, inspection | Same-scope assembler remains planned. |
| DPF-VV-017 | P0 | V&V/Data | A first-principles validation certificate shall require same-scope waveform, phase, spatial, neutron, detector, field-coupling, physics-fidelity, numerical-fidelity, and UQ evidence. | blocked | review, test, inspection | First certificate source search is recorded in `docs/FIRST_PRINCIPLES_CERTIFICATE_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed `certificate_gate` packet, but accepted upstream packets, hashes, reviewers, metrics/UQ IDs, requirement links, and negative-test proof remain blocked. |
| DPF-VV-018 | P1 | V&V/Physics | A generalized DPF-machine first-principles claim shall require repeating the full evidence path on at least one additional device or shot scope. | blocked | review, analysis, test | First generalization source search is recorded in `docs/FIRST_PRINCIPLES_GENERALIZATION_SOURCE_SEARCH_2026_05_15.md`; the runner emits a fail-closed `generalization` packet, but accepted primary certificate, declared second-scope packet chain, no-hidden-assumption proof, device parameterization, scale-transition review, regression, source review, and cross-scope negative tests remain blocked. |
| DPF-DATA-001 | P0 | Data/V&V | Every first-principles solver execution shall produce a run manifest for successful, blocked, and failed runs. | implemented for `first-principles-3d`; broader routing partial | test, inspection | `dpf first-principles-3d` now emits a `FirstPrinciplesRunManifest`; PF-1000/app/general-engine paths still need unification. |
| DPF-DATA-002 | P0 | Data/Product | Every first-principles result shall carry a fail-closed classification label. | implemented | test, inspection | Existing labels remain non-promoting until certificate passes. |
| DPF-DATA-004 | P1 | V&V/Data | Validation certificates shall write only when all linked gates pass. | partial | test, inspection | Existing certificate artifacts reject blocked/cross-scope evidence; the package-native first-principles runner now also emits `certificate_gate`, but first-principles-specific negative tests for hidden-limiter, app-only, missing-UQ, missing-review, and reduced-model fallback evidence remain incomplete. |
| DPF-REL-002 | P0 | Release/V&V | Every P0 first-principles requirement shall map to verification evidence or an explicit blocker. | partial | inspection | RTM export is staged; Doorstop import remains deferred. |

## Execution Order

1. Extend the new package-native `src/dpf.first_principles` runner from compact
   engineering candidate to the shared PF-1000/Akel and whole-shot execution
   contract.
2. Build the global limiter/repair registry across app, solver, backend, and
   circuit layers.
3. Add a top-level first-principles readiness blocker for any
   acceptance-blocking limiter activation.
4. Replace engineering repairs with verified numerical methods or source-backed
   physical bounds until a full-run limiter-zero candidate exists.
5. Build the numerical-fidelity matrix and toleranced tests.
6. Implement the source-backed startup BVP.
7. Extend the current implicit-midpoint power-port candidate into an accepted
   coupling packet with electrode-work, field-load partition, sign, residual,
   and same-scope validation evidence.
8. Decide dimensionality and MHD/kinetic handoff for the claimed interval.
9. Build closure packets for every active or bounded-out physics effect.
10. Complete Akel Fig. 1 independent review and waveform/current-dip UQ.
11. Build same-scope phase, spatial, field, temperature, neutron, detector, and
    UQ packets, or narrow/switch the accepted demonstrator scope.
12. Implement kinetic/hybrid beam-target production and detector response.
13. Assemble the same-scope packet and generate the validation certificate.
14. Repeat the full evidence path on a second DPF scope.

## Current First Critical Path

FP-2 now has five implementation slices. The app-level PF-1000/Akel
first-principles path emits `first_principles_limiter_ledger`; Python
cylindrical state-mutating floors/clamps/repairs are merged into that ledger;
Python PLM/HLL/reconstructed-state positivity/CFL controls are classified as
nonblocking `verified_numerical_method` records; and the active Python
field-coupled probe now removes the resistivity floor/cap, temperature floor/cap,
hard field-coupled timestep cap, current floor, and back-EMF clip from bounded
short probes. The replacement resistivity path is a source-traced,
uncapped partial-ionization Spitzer/Braginskii candidate initialized from the
local PF-1000 post-breakdown source state, and the field feedback now uses an
implicit-midpoint power port instead of suppressing voltage below a current
threshold. The active Python path also now uses an operator-split
Crank-Nicolson ADI update for the local-source axisymmetric `B_theta`
resistive-induction scope, while continuing to report the explicit
resistive-diffusion timestep as stiffness evidence.

Current bounded evidence:

- `run_pf1000_akel_first_principles(sim_time_us=0.002)` completed with
  `nan_detected=False`, `n_steps=56`, and
  `first_principles_limiter_ledger.status=clear`.
- `run_pf1000_akel_first_principles(sim_time_us=0.01)` completed with
  `nan_detected=False`, `n_steps=287`, and a clear limiter ledger.
- `run_pf1000_akel_first_principles(sim_time_us=0.05)` completed with
  `nan_detected=False`, `n_steps=1415`, a clear limiter ledger, maximum
  field-power back-EMF about `1.19e4 V`, and power-port residual below
  `1.5e-8 W`.
- After partial-ionization thermodynamics correction, the explicit resistive
  path completed `run_pf1000_akel_first_principles(sim_time_us=0.1)` with
  `nan_detected=False`, `n_steps=422`,
  `first_principles_limiter_ledger.status=clear`, `Te_min=1122 K`, and
  `eta_max=0.0496 ohm m`; this is superseded as the active path by the
  implicit cylindrical `B_theta` operator below.
- With the implicit cylindrical `B_theta` resistive operator and reported LC
  phase timestep control,
  `run_pf1000_akel_first_principles(sim_time_us=0.1)` completed with
  `nan_detected=False`, `n_steps=91`, a clear limiter ledger,
  `Te_min=296.7 K`, nonzero field feedback, and peak field-power back-EMF about
  `8.05 kV`.
- `run_pf1000_akel_first_principles(sim_time_us=1.0)` completed with
  `nan_detected=False`, `n_steps=904`, a clear limiter ledger,
  `Te_min=193.7 K`, and nonzero field feedback.
- The new timestep diagnostics show `dt_diff_s` remains below the actual
  coupled timestep, but it is now reported as `resistive_stiffness_ratio`
  evidence instead of controlling the accepted app timestep. The active
  controller for these bounded probes is the reported LC phase resolution,
  not an eta cap or hidden hard timestep cap.

Readiness still blocks on missing same-scope evidence and full-run proof:
`accepted_same_scope_akel_digitization`, `validated_field_coupling_packet`,
`field_coupled_energy_accounting`,
`first_principles_startup_initialization`,
`first_principles_neutron_yield_authority`, `mhd_numerical_fidelity_packet`,
`physics_fidelity_packet`, `reduced_model_active_closure_rejected`, and
`sheath_position`.

The remaining FP-2/FP-3 work is the full active-path registry and full-horizon
limiter-zero candidate:

- keep the app-level ledger wired through `first_principles_limiter_ledger`, CLI
  payloads, and manifests;
- add equivalent result-bound telemetry for Metal/MLX/Athena/AthenaK/hybrid
  repair/fallback paths before allowing those backends into first-principles
  acceptance scope;
- maintain negative tests for every non-Python backend/fallback surface that
  remains excluded from acceptance as new backend surfaces are added;
- continue recording type, code path, count, before/after min/max, finite count,
  affected field, justification, and acceptance-blocking status for every active
  path intervention;
- add tests for limiter-free engineering candidate classification, app-only
  runner rejection, solver-internal limiter propagation, and backend fallback
  rejection;
- begin replacing each blocker with a verified numerical method or source-backed
  physical bound.
- extend bounded clear probes to the required PF-1000/Akel horizon without
  timestep runaway, nonfinite state, or acceptance-blocking limiter activation.
- extend the implicit resistive operator beyond the current axisymmetric
  `B_theta` scope or keep material `B_r/B_z` as an acceptance blocker.
- implement source-scoped ionization/recombination and electron-neutral
  transport so constant `Z_bar` no longer carries the post-breakdown plasma.

This keeps the plan focused on first principles: the next milestone is a
PF-1000/Akel run whose fields are evolved by verified numerics and whose
remaining approximations are explicit, reviewed, and non-promoting until the
certificate path passes.
