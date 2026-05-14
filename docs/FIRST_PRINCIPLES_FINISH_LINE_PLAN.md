# First-Principles DPF Finish-Line Plan

Date: 2026-05-13

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
| FP-1 | Package-native PF-1000/Akel engineering probe | App-backed probe exists | `dpf first-principles` and `dpf simulate --run-mode=first_principles_mhd` route to one package-native solver path and produce the same run manifest schema. |
| FP-2 | Global limiter registry and readiness gate | Blocked | Every floor, cap, clip, repair, fallback, timestep/current/back-EMF cap, precision fallback, and solver-layer limiter in the active path is recorded with type, count, before/after min/max, source, justification, and acceptance-blocking status. First-principles readiness fails when an acceptance blocker activates. |
| FP-3 | Limiter-free or physically bounded candidate | Blocked | Full 12 us PF-1000/Akel engineering run completes with zero acceptance-blocking limiter activations. Any remaining bound is either a source-backed physical bound or a verified numerical method. |
| FP-4 | Numerical reference workflow | Partial | Numerical-fidelity packet passes named tests for finite-volume shocks, cylindrical source terms, `div B`, resistive diffusion, Joule heating, circuit-coupled energy, restart, precision, and backend parity. |
| FP-5 | Source-backed startup BVP | Scaffolded | Startup state generator solves or constructs source-backed breakdown/preionization/electrode/insulator/current-density/ionization/temperature/field/sheath-liftoff state with units, source links, and tests. Seeded-layer startup remains engineering-only. |
| FP-6 | Resolved power-port circuit coupling | Candidate | Circuit update is driven by a tested power-port relation with Poynting or `J.E`, electrode work, time centering, sign convention, no clipped back-EMF for acceptance, and residual tolerance. |
| FP-7 | Dimensionality and handoff decision | Missing | Claim interval is explicitly bounded to valid 2D/axisymmetric MHD, or a 3D MHD plus kinetic/hybrid handoff path is implemented and validated for the claimed observables. |
| FP-8 | Physics closure packets | Partial | Each active or bounded-out physics effect has source equations, symbol mapping, units, validity regime, verification test, validation/bounding evidence, sensitivity/UQ, and claim impact. |
| FP-9 | Same-scope source availability decision | Blocked | PF-1000/Akel has accepted same-scope evidence for required observables, or the claim is narrowed, or the first accepted demonstrator is switched to a better-diagnosed local source scope. |
| FP-10 | Accepted waveform and phase evidence | Blocked by review | Akel current waveform/current dip and phase targets are accepted, UQ-bearing, comparator-bound, and tested against production outputs. |
| FP-11 | Accepted spatial, field, and temperature evidence | Blocked | Density, magnetic/EM field, temperature, and relevant diagnostic response targets are accepted, UQ-bearing, comparator-bound, and same-scope. |
| FP-12 | Mechanism-separated neutron authority | Blocked | Thermonuclear history comes from resolved fields; beam-target yield comes from accepted kinetic/hybrid production, transport/stopping, spectrum, anisotropy, detector response, and UQ. |
| FP-13 | Comparator and UQ matrix complete | Blocked | Every required observable has accepted evidence state, output field mapping, comparator metric, UQ components, pass/fail rule, linked requirement, and linked test/artifact. |
| FP-14 | Validation certificate and release decision | Blocked | Certificate passes and includes run manifest hash, evidence packet hashes, reviewer metadata, comparator metrics, UQ packets, requirement IDs, commands, release label, and negative-test proof. |
| FP-15 | Generalized DPF-machine path | Planned | A second device or shot repeats `FP-1` through `FP-14` without hidden PF-1000/Akel assumptions. |

## Workstream A: Package-Native Execution Path

Objective: make the first-principles simulator a real package capability, not an
app-only helper.

1. Move the accepted execution path into `src/dpf`, with `app_mhd.py` becoming a
   caller rather than the authority.
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

The registry must include every active-path limiter source:

- `app_mhd.py` field/state repair, density, pressure, temperature, velocity,
  magnetic field, magnetic energy, resistivity floor/cap, timestep cap, current
  floor, and back-EMF cap.
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

The startup generator must produce:

- density and species or ionization state;
- electron and ion temperatures or a stated single-temperature limit;
- pressure, velocity, magnetic field, current density, and resistivity;
- electrode, cathode, anode, and insulator boundary metadata;
- circuit initial derivative or boundary drive consistency;
- `div B` and unit checks;
- sheath-liftoff and early axial-run evidence status.

Acceptance evidence:

- Startup evidence packet with local source paths, hashes, pages/lines/figures,
  units, assumptions, and review status.
- BVP consistency tests for current, magnetic field, boundary conditions, units,
  and `div B`.
- Negative test proving seeded-layer startup blocks accepted first-principles
  claims.

## Workstream E: Resolved Power-Port Coupling

Objective: make circuit-plasma feedback a conservation-law power exchange.

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

`L_field = 2 E_B/I^2` may remain a diagnostic. It is not the circuit load unless
a reviewed power-port packet proves that use for the claimed interval.

Acceptance evidence:

- Standalone component tests for power-port sign and energy conservation.
- Integrated PF-1000/Akel run with bounded residual and no fallback to
  snowplow/Lee load after startup.
- Same-scope waveform comparator only after evidence and UQ pass.

## Workstream F: Dimensionality And Kinetic Handoff

Objective: prevent a 2D MHD engineering model from being overstated as a
complete DPF simulation.

The plan must make one of these accepted decisions:

- `bounded_axisymmetric_mhd_claim`: the accepted claim stops before local
  source-backed MHD breakdown and excludes observables requiring 3D/kinetic
  dynamics.
- `validated_3d_mhd_claim`: the accepted claim uses a 3D MHD path for observables
  requiring non-axisymmetric structure.
- `mhd_kinetic_handoff_claim`: the accepted claim hands resolved MHD output to a
  kinetic/hybrid model for late-pinch, beam, spectrum, anisotropy, and total
  neutron-yield observables.

Acceptance evidence:

- Dimensionality decision packet with local source links and claimed interval.
- Tests proving out-of-scope observables cannot be marked accepted.
- Handoff packet with fields, particle source terms, energy/momentum transfer,
  and conservation checks when kinetic/hybrid physics is used.

## Workstream G: Physics Closure Packets

Objective: turn named physics features into auditable closures or explicit
bounds.

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

Lee/Saw beam-target estimates, empirical beam fractions, and final-state
thermonuclear duration approximations are baseline comparisons only.

## Workstream I: Same-Scope Evidence And UQ

Objective: turn local sources into accepted comparison data for the accepted run.

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

## Requirements Map

| ID | Priority | Owner | Requirement | Status | Verification | Acceptance evidence or blocker |
| --- | --- | --- | --- | --- | --- | --- |
| DPF-PHYS-008 | P0 | Physics/Engine | Circuit feedback for first-principles mode shall be driven by resolved field power and conservation ledgers, not Lee/RADPF closure factors. | partial | test, analysis | Field-coupled engineering evidence exists; accepted power-port packet remains blocked. |
| DPF-PHYS-009 | P0 | Physics/Engine | First-principles PF-1000/Akel runs shall reject hidden engineering limiters for accepted claims. | blocked | test, inspection | Global limiter registry and top-level readiness blocker are not complete. |
| DPF-PHYS-010 | P0 | Physics/V&V | First-principles startup shall use source-backed breakdown, preionization, electrode boundary, and initial plasma evidence. | blocked | review, analysis, test | Startup BVP packet and tests do not exist. |
| DPF-PHYS-011 | P0 | Physics/V&V | Field-circuit coupling shall carry validated inductance, terminal voltage/back-EMF, Poynting or `J.E` power, handoff intervals, and energy residual evidence. | partial | analysis, test | Diagnostics exist; accepted power-port evidence remains blocked. |
| DPF-PHYS-012 | P0 | Physics/V&V | First-principles physics fidelity shall classify each required effect as implemented, validated, bounded out, blocked, or missing for the claimed scope. | partial | inspection, test | Closure packets remain missing or partial. |
| DPF-PHYS-013 | P0 | Physics/V&V | Total neutron-yield authority shall require resolved thermonuclear history, accepted kinetic/hybrid beam-target production, same-scope neutron evidence, and UQ. | blocked | analysis, review, test | Kinetic/hybrid beam-target authority and same-scope neutron packet remain blocked. |
| DPF-PHYS-014 | P0 | Physics/Architecture | First-principles acceptance shall define dimensionality and any MHD-to-kinetic handoff for the claimed interval and observables. | blocked | analysis, test, review | No accepted dimensionality/handoff packet exists. |
| DPF-PHYS-015 | P0 | Numerics/V&V | First-principles numerical-fidelity packets shall define named tests, norms, mesh families, tolerances, precision/backend scope, and limiter-zero acceptance. | blocked | test, analysis | Numerical-fidelity tolerances and limiter-zero packet are not complete. |
| DPF-PHYS-016 | P0 | Physics/Engine | The active first-principles circuit power port shall pass Poynting or `J.E`, electrode-work, time-centering, sign, and residual tests without clipped back-EMF for acceptance. | blocked | test, analysis | Current field-coupled probe still uses engineering caps. |
| DPF-PHYS-017 | P0 | Physics/Engine | First-principles startup shall be generated as a source-backed boundary-value problem with current-density, field, ionization, temperature, and sheath-liftoff consistency checks. | blocked | test, review, analysis | Current startup is scaffolded engineering initialization. |
| DPF-PHYS-018 | P0 | Physics/V&V | Every active or bounded-out physical closure shall have a packet with source equations, symbol map, units, validity regime, verification, sensitivity/UQ, and claim impact. | blocked | inspection, analysis, test | Closure packet registry is not complete. |
| DPF-PHYS-019 | P1 | Architecture/Engine | Accepted first-principles execution shall run through one package-native `src/dpf` path shared by CLI, API, config, and app surfaces. | planned | test, inspection | Current runnable tool is app-backed. |
| DPF-VV-011 | P0 | V&V/Data | Every target, curve, table, formula, uncertainty value, comparator, and same-scope packet shall be typed evidence with local source provenance. | planned | test, inspection | Canonical evidence schema remains planned. |
| DPF-VV-012 | P0 | V&V/Data | Digitized figure or table evidence shall require independent accepted review before validation use. | partial | test, review | Akel Fig. 1 remains `blocked_by_review`. |
| DPF-VV-013 | P0 | V&V/Physics | Source-closed coded formulas shall carry formula evidence packets. | planned | analysis, test | Formula registry remains planned. |
| DPF-VV-014 | P0 | V&V/UQ | Quantitative validation shall require uncertainty extraction and propagation. | planned | analysis, test | UQ packets are not comparator-bound. |
| DPF-VV-015 | P0 | V&V/Physics | Accepted targets shall be bound to simulation outputs through tested comparators. | planned | test, analysis | General comparator registry remains incomplete. |
| DPF-VV-016 | P0 | V&V/Physics | Same-scope packet assembly shall reject cross-device, cross-shot, or cross-configuration evidence unless a reviewed transfer rule exists. | planned | test, inspection | Same-scope assembler remains planned. |
| DPF-VV-017 | P0 | V&V/Data | A first-principles validation certificate shall require same-scope waveform, phase, spatial, neutron, detector, field-coupling, physics-fidelity, numerical-fidelity, and UQ evidence. | blocked | review, test, inspection | No accepted first-principles PF-1000/Akel packet exists. |
| DPF-VV-018 | P1 | V&V/Physics | A generalized DPF-machine first-principles claim shall require repeating the full evidence path on at least one additional device or shot scope. | planned | review, analysis | Second-scope generalization waits on PF-1000/Akel closure. |
| DPF-DATA-001 | P0 | Data/V&V | Every first-principles solver execution shall produce a run manifest for successful, blocked, and failed runs. | implemented | test, inspection | Existing manifest support must be bound to package-native first-principles path. |
| DPF-DATA-002 | P0 | Data/Product | Every first-principles result shall carry a fail-closed classification label. | implemented | test, inspection | Existing labels remain non-promoting until certificate passes. |
| DPF-DATA-004 | P1 | V&V/Data | Validation certificates shall write only when all linked gates pass. | implemented | test, inspection | Certificate negative tests must cover the new hidden-limiter and package-native blockers. |
| DPF-REL-002 | P0 | Release/V&V | Every P0 first-principles requirement shall map to verification evidence or an explicit blocker. | partial | inspection | RTM export is staged; Doorstop import remains deferred. |

## Execution Order

1. Move the first-principles candidate toward a package-native `src/dpf` runner.
2. Build the global limiter/repair registry across app, solver, backend, and
   circuit layers.
3. Add a top-level first-principles readiness blocker for any
   acceptance-blocking limiter activation.
4. Replace engineering repairs with verified numerical methods or source-backed
   physical bounds until a full-run limiter-zero candidate exists.
5. Build the numerical-fidelity matrix and toleranced tests.
6. Implement the source-backed startup BVP.
7. Replace current field-load engineering feedback with an accepted power-port
   coupling packet.
8. Decide dimensionality and MHD/kinetic handoff for the claimed interval.
9. Build closure packets for every active or bounded-out physics effect.
10. Complete Akel Fig. 1 independent review and waveform/current-dip UQ.
11. Build same-scope phase, spatial, field, temperature, neutron, detector, and
    UQ packets, or narrow/switch the accepted demonstrator scope.
12. Implement kinetic/hybrid beam-target production and detector response.
13. Assemble the same-scope packet and generate the validation certificate.
14. Repeat the full evidence path on a second DPF scope.

## Current First Critical Path

The immediate code target is no longer just "enumerate active limiter fields in
`app_mhd.py`." The correct FP-2 target is the full active-path limiter registry:

- inventory limiters in `app_mhd.py`, `src/dpf/fluid/cylindrical_mhd.py`, circuit
  coupling, backend adapters, and first-principles post-processing;
- record type, code path, count, before/after min/max, finite count, affected
  field, justification, and acceptance-blocking status;
- expose the registry in the run artifact and manifest;
- make `first_principles_mhd_readiness_report()` fail on any
  `acceptance_blocker` activation, not only neutron authority;
- add tests for synthetic limiter activation, limiter-free engineering
  candidate classification, app-only runner rejection, and reduced-model active
  closure rejection;
- begin replacing each blocker with a verified numerical method or
  source-backed physical bound.

This keeps the plan focused on first principles: the next milestone is a
PF-1000/Akel run whose fields are evolved by verified numerics and whose
remaining approximations are explicit, reviewed, and non-promoting until the
certificate path passes.
