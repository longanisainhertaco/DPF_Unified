# First-Principles DPF Simulator SSR And Implementation Audit Spec

Date: 2026-05-18

Status: execution specification for outside implementation teams. This is a
source-scoped engineering instruction document, not a validation certificate.

Primary repo: `/Users/anthonyzamora/dpf-unified`

Primary source authority: `KnowledgeReference/`

Primary demonstrator: PF-1000/Akel 16 kV, 1.2 Torr, shot-12581 engineering
scope.

## Purpose

This document defines the work required to turn the current DPF-Unified
first-principles path into a full 3-D whole-shot Dense Plasma Focus simulator.
It also defines how Codex will audit another team's work after they implement
the instructions.

The instructions are intentionally strict. The team may use other AI tools,
external research assistants, generated code, notebooks, and architecture
guidance, but those tools do not become scientific authority. Every physics
claim, equation, closure, parameter, geometry value, diagnostic target, and
acceptance statement must trace back to local `KnowledgeReference/` evidence or
remain explicitly blocked.

## Non-Negotiable Rules

1. `KnowledgeReference/` is the scientific source of truth.
2. External AI output is never source truth.
3. External web or paper material is never source truth until it is explicitly
   ingested or promoted into `KnowledgeReference/` with path, hash, scope, and
   review status.
4. Reduced models, Lee/RADPF, snowplow, scaling laws, fitted current fractions,
   empirical beam fractions, and GV executable outputs may be baselines or
   comparators only. They may not drive first-principles predictive authority.
5. Unknown, partial, draft, candidate, blocked, or unreviewed evidence fails
   closed.
6. Engineering artifacts may run and may be useful. They must remain
   `engineering_candidate_not_validation` or equivalent until all certificate
   gates pass.
7. Do not remove blocker packets to make the simulator look complete.
8. Do not rename a candidate packet to accepted status without adding the
   source packet, implementation proof, negative tests, convergence/restart
   evidence, comparator/UQ binding, and review metadata.
9. Do not mix PF-1000/Akel 16 kV shot-12581 values with PF-1000U or full-energy
   PF-1000 shots unless a reviewed transfer packet exists.
10. If an AI tool proposes an equation, parameter, boundary condition, or
    closure, the submitted implementation must include a local source citation
    proving it or must mark it `candidate_not_validation`.

## Current System Baseline

The current first-principles path is a package-native 3-D hybrid
electromagnetic PIC/fluid engineering runtime. It is centered on:

- `src/dpf/first_principles/deck.py`
- `src/dpf/first_principles/runner.py`
- `src/dpf/first_principles/power_port.py`
- `src/dpf/fields/hybrid_simulator.py`
- `src/dpf/fields/hybrid_loop.py`
- `src/dpf/fields/hybrid_stepper.py`
- `src/dpf/fields/circuit_boundary.py`

The current runtime can:

- build a source-scoped PF-1000/Akel engineering deck;
- project candidate PF-1000 conductor masks;
- initialize a seeded-layer candidate startup state;
- advance a Cartesian 3-D Maxwell state;
- push and deposit PIC ion particles;
- apply generalized-Ohm and electron-energy candidate closures;
- evolve a lumped external circuit state;
- apply a magnetic injection boundary from terminal current;
- emit fail-closed packets for startup, power port, dimensionality, closures,
  limiter readiness, numerical fidelity, same-scope source state, waveform
  phase, spatial/field/temperature evidence, neutron authority, comparator/UQ,
  certificate gate, and generalization.

The current runtime cannot yet claim a full first-principles whole-shot
simulation because startup, power-port authority, reviewed geometry, closure
authority, neutron authority, same-scope comparison, numerical fidelity, and
certificate release remain blocked.

## Source Basis

The following local sources are mandatory starting points. The team must add
more local source references when implementing any additional module or closure.

| Source | Required use |
| --- | --- |
| `KnowledgeReference/auluck-2021-dpf-circuit-element.md:151-209` | DPF voltage as field-power relation over declared domain, source-interface exclusion, and the requirement that all chamber phenomena draw power from the external circuit. |
| `KnowledgeReference/auluck-2021-dpf-circuit-element.md:235-262` | Poynting-flux relation at the source interface. |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1880-1888` | Poynting theorem and signed `J.E` energy accounting. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:741-789` | External-circuit update, current-derived magnetic boundary condition, and source-derived `U_DPF` pattern. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:108-142` | PF-1000/Akel geometry, bank, pressure, diagnostics, timing, and neutron measurement context. |
| `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:262-270` | PF-1000/Akel shot-12581-like deck values: `L0`, `C0`, `r0`, `b`, `a`, `z0`, `V0`, `p0`. |
| `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:340-356` | PF-1000 rod cathode, center electrode, insulator, and capacitor-bank geometry context. |
| `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md:55-80` | DPF phase structure: insulator breakdown, kinetic surface discharge, MHD inverse pinch, and microsecond acceleration. |

The existing source-truth audits must remain clean:

- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md`
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json`
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_16.md`
- `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_16.md`

## System Goal

Build a full 3-D first-principles DPF simulator that can run a whole machine
shot from startup through rundown, pinch, post-pinch, and diagnostic/neutron
outputs using evolved state variables and source-backed closures.

The accepted final system must:

- solve startup from source-backed breakdown, preionization, insulator
  flashover, sheath formation, and liftoff conditions;
- represent the reviewed machine geometry and material boundaries;
- couple the pulsed-power circuit through a reviewed electromagnetic power
  port;
- advance 3-D electromagnetic fields, particles, charge/current, electron and
  ion energy, ionization, radiation, ablation/impurities, collisions/stopping,
  and transport;
- emit mechanism-separated neutron histories and detector-response artifacts;
- compare to same-scope source targets with explicit units, metrics, tolerance,
  uncertainty, and review state;
- prove numerical fidelity through convergence, restart reproducibility,
  backend/precision scope, limiter-zero or physical-bound packets, and
  artifact hashes;
- write an engineering review/certificate packet that shows what is accepted,
  what is blocked, and why.

## Target Runtime Wire Diagram

The implementation must converge on this package-native workflow:

```text
KnowledgeReference source packet(s)
  -> FirstPrinciplesInputDeck
     -> device geometry/material/boundary packet
     -> gas/species packet
     -> circuit packet
     -> startup BVP packet
     -> closure policy packet
     -> diagnostics/comparator target packet
  -> FirstPrinciples3DSession or FirstPrinciplesRun
     -> 3-D grid and reviewed conductor/material masks
     -> startup fields/particles/species/temperatures/current density
     -> external circuit state
     -> electromagnetic power port
     -> Maxwell field update
     -> PIC ion push/deposition
     -> generalized Ohm/electron closure
     -> ion/electron energy and chemistry update
     -> radiation/material/collision/stopping update
     -> neutron mechanism update
     -> diagnostics/UQ update
  -> FirstPrinciplesManifest
     -> conservation ledger
     -> limiter ledger
     -> power-port ledger
     -> numerical-fidelity packet
     -> same-scope comparator/UQ packet
     -> certificate gate
  -> JSON/HDF5 artifact for engineering review
```

No UI, app, notebook, external binary, or reduced-model path may bypass this
package-native flow for a first-principles claim.

## Required Repo Structure

Implement within the existing structure unless a change is justified in a
design note.

| Area | Existing files | Required responsibility |
| --- | --- | --- |
| Decks and source inputs | `src/dpf/first_principles/deck.py` | Source-scoped input contract, PF-1000/Akel lock, user-verified non-promoting decks, source references, units, deck hash. |
| Runtime orchestration | `src/dpf/first_principles/runner.py` | Package-native 3-D execution, packet assembly, manifest creation, session continuation. |
| Power port | `src/dpf/first_principles/power_port.py`, `src/dpf/fields/circuit_boundary.py`, `src/dpf/fields/hybrid_simulator.py` | Source-backed circuit/plasma power exchange, low-current `P/I` handling, sign/domain/time-centering review, residual ledger. |
| Startup | `src/dpf/first_principles/startup_bvp.py`, `src/dpf/first_principles/startup_breakdown.py` | Breakdown/flashover/preionization/liftoff packet and implementation. |
| Geometry/boundaries | `src/dpf/fields/source_geometry.py`, `src/dpf/fields/particle_boundaries.py`, runner boundary helpers | Reviewed rods, hollow anode, insulator, material surfaces, PML/open/conductor semantics. |
| Fields/PIC loop | `src/dpf/fields/maxwell_3d.py`, `hybrid_stepper.py`, `hybrid_loop.py`, `hybrid_simulator.py`, `pic_coupling.py` | Long-run 3-D EM/PIC/fluid engine with source-ordered updates and conservation ledgers. |
| Closures | `src/dpf/first_principles/closure_packet.py`, `src/dpf/fields/electron_energy.py`, `src/dpf/fields/ionization_transport.py`, `src/dpf/fluid/two_temperature.py`, `src/dpf/atomic`, `src/dpf/radiation`, `src/dpf/collision` | EOS, ionization, transport, electron/ion energy, radiation, ablation/impurities, anomalous resistance, collisions/stopping. |
| Numerics | `src/dpf/first_principles/numerical_fidelity.py`, `experimental_numerics.py`, `checkpoint_restart.py`, `split_continuation.py`, `limiter_readiness.py`, `limiter_proof.py` | Convergence, restart, limiter, precision/backend, timestep, and artifact-hash proof. |
| Diagnostics/comparison | `same_scope.py`, `waveform_phase.py`, `spatial_field_temperature.py`, `neutron_authority.py`, `current_waveform_comparator.py`, `comparator_uq.py` | Same-scope target binding, metrics, uncertainty, neutron mechanism separation, detector response. |
| Release gate | `certificate_gate.py`, `manifest.py`, docs | Fail-closed certificate and engineering review packet. |
| CLI/API | `src/dpf/cli/main.py`, server surfaces if touched | Must call package-native first-principles runner, not app-only or reduced paths. |

## SSR Requirements

### SSR-001 Source-Truth Traceability

Every physics-facing module or packet must declare local source references.

Expected implementation:

- Add or update `source_references` fields for every new closure/operator.
- Include file path and line range.
- Include source scope: device, shot, geometry, formula family, or numerical
  method scope.
- If source support is partial, status must be `candidate` or `blocked`.

Audit expectation:

- Codex will run `scripts/verify_first_principles_source_truth_exhaustion.py`.
- Codex will run `scripts/verify_first_principles_module_source_vetting.py`.
- Codex will reject uncited formulas or line ranges that do not exist locally.

### SSR-002 Package-Native Runtime Authority

All first-principles runtime claims must route through
`dpf.first_principles`.

Expected implementation:

- Use `run_first_principles_3d_deck()` or `FirstPrinciples3DSession`.
- CLI commands must use package-native deck/runtime/manifest paths.
- App-backed or legacy engine paths may call the package-native runner but may
  not define separate scientific authority.

Audit expectation:

- Codex will inspect `src/dpf/cli/main.py`, `src/dpf/server`, and any touched
  app surfaces.
- Codex will reject first-principles claims produced by repo-root app code,
  notebooks, or reduced-model wrappers.

### SSR-003 PF-1000/Akel Deck Lock

The default demonstrator must remain PF-1000/Akel 16 kV, 1.2 Torr, shot-12581
engineering scope unless explicitly changed.

Expected implementation:

- Preserve source-locked values:
  - `C0 = 1332 uF`
  - `V0 = 16 kV`
  - `L0 = 25 nH`
  - `r0 = 6.1 mOhm`
  - `a = 11.55 cm`
  - `b = 16 cm`
  - `z0 = 48 cm`
  - `p0 = 1.2 Torr`
  - 12 cathode rods, 80 mm diameter
  - alumina insulator, 85 mm length
- Emit a deck-diff packet for every PF-1000/Akel run.
- Any deviation must be reported as drift, not silently accepted.

Audit expectation:

- Codex will inspect `telemetry["deck_diff"]`.
- Codex will reject mixed PF-1000/PF-1000U/full-energy values without a transfer
  packet.

### SSR-004 Startup BVP

The simulator must replace seeded-layer startup with a source-backed startup
boundary-value problem before any whole-shot first-principles claim can pass.

Expected implementation:

- Implement or import a reviewed startup packet containing:
  - gas breakdown model;
  - preionization state;
  - insulator surface flashover;
  - electrode/insulator boundary conditions;
  - initial current-density distribution;
  - electron and ion temperatures;
  - ionization and species state;
  - electric and magnetic fields;
  - sheath liftoff/handoff interval.
- Existing seeded-layer mode must remain rejected for accepted claims.
- Startup packet must include negative tests proving seeded/text-only startup
  cannot pass.

Audit expectation:

- Codex will inspect `startup_bvp` packet status.
- Codex will reject any `seeded_layer` or text-only startup marked accepted.
- Codex will require tests in `tests/test_first_principles_runner.py` or a
  dedicated startup test file.

### SSR-005 Reviewed Geometry And Material Boundaries

The solver must represent PF-1000 geometry as reviewed 3-D masks and material
interfaces.

Expected implementation:

- Implement masks for:
  - 12 cathode rods;
  - hollow/copper anode geometry if source-supported;
  - alumina insulator;
  - electrode backplate/source interface;
  - vacuum chamber/wall boundary if active;
  - material surfaces for ablation/electrode work if modeled.
- Emit geometry packet with mask hash, grid spacing, projected dimensions,
  source values, and error from source dimensions.
- Candidate coarse projections must remain candidate.

Audit expectation:

- Codex will inspect `boundary_policy`, conductor mask packets, and manifest
  metadata.
- Codex will reject geometry that looks axisymmetric while claiming rod-level
  PF-1000 authority.

### SSR-006 Circuit/Power-Port Authority

The power port is the highest-priority physics blocker.

Expected implementation:

- Keep these candidate alternatives explicit:
  - Auluck volume `J.E`: `U_DPF = - integral_Omega(J.E)dV / I`
  - Poynting surface power through a declared source interface
  - hybrid-PIC source-derived `U_DPF` architecture pattern
  - Sigma/quasi-TEM line voltage as exploratory diagnostic only
- Do not promote Sigma/quasi-TEM line voltage as a primary driver unless a
  local source packet defines the DPF port plane, path-independence evidence,
  current/power closure, and wall/electrode work accounting.
- Implement accepted or candidate packets for:
  - named domain or interface surface;
  - terminal current and voltage;
  - sign convention;
  - time-centering;
  - electrode/interface work;
  - wall Poynting flux excluding the declared port;
  - volume `J.E` work;
  - stored EM energy delta;
  - external circuit work;
  - low-current `P/I` singularity handling;
  - residual budget.
- Negative local `J.E` must not be clipped just to stabilize a run.
- Low-current `P/I` fallback must be reported as a blocker unless a
  source-backed handoff/regularization packet exists.

Audit expectation:

- Codex will inspect `power_port.stage0_packet_scaffolds`,
  `candidate_stage0_energy_ledger`, `power_port_operator_comparison`, and
  `low_current_p_over_i_singularity`.
- Codex will reject a solution that hides `1/I`, current-floor, back-EMF, or
  negative `J.E` fallback behavior.
- Codex will run sign-reversal/domain-corruption/time-centering/low-current
  negative tests when present, or require them if absent.

### SSR-007 3-D Field/PIC/Electron Runtime

The target loop must advance fields, particles, current, charge, and electron
state as one coherent first-principles runtime.

Expected implementation:

- Use the existing `src/dpf/fields` nucleus unless a design note justifies a
  replacement.
- Preserve Maxwell evolution in plasma and vacuum.
- Preserve charge/current deposition and continuity telemetry.
- Preserve source-ordered velocity/current/field updates.
- Preserve electron energy state and ionization state across continuation and
  restart.
- Emit per-step conservation, limiter, and residual summaries.

Audit expectation:

- Codex will compare uninterrupted, split-continuation, checkpoint-restart, and
  rerun fingerprints.
- Codex will reject runs that pass only because state repairs, floors, or
  fallback paths are hidden.

### SSR-008 Physics Closures

All active physics closures must be source-backed, unit-checked, and scoped.

Expected implementation:

- Close or explicitly block:
  - EOS;
  - Spitzer/Braginskii/conductivity and anomalous resistance;
  - Hall/pressure/electron inertia terms if active;
  - ionization/recombination;
  - separate electron and ion energy coupling;
  - heat flux;
  - radiation;
  - electrode ablation and impurities;
  - restrike;
  - collisions;
  - stopping and beam-target coupling.
- Each closure must report:
  - source references;
  - validity regime;
  - active/inactive status;
  - numerical limiter or physical bound;
  - coupling order;
  - energy accounting;
  - negative tests.

Audit expectation:

- Codex will inspect `physics_closure` packet and active runtime telemetry.
- Codex will reject closures that are implemented as constants, fitted factors,
  or hidden empirical knobs without source scope and fail-closed status.

### SSR-009 Neutron Mechanism And Detector Authority

Neutron authority must be mechanism-separated.

Expected implementation:

- Separate:
  - thermonuclear D-D production;
  - beam-target production;
  - beam formation and transport;
  - spectrum;
  - anisotropy;
  - direct versus scattered detector contribution;
  - activation counter response;
  - TOF response.
- Scalar total yield alone cannot accept neutron authority.
- Reduced Lee neutron outputs may remain comparator baselines only.

Audit expectation:

- Codex will inspect `neutron_authority` packet.
- Codex will reject total-yield-only claims.
- Codex will require detector-response and UQ packets before any accepted
  neutron claim.

### SSR-010 Same-Scope Comparator And UQ

Every accepted observable must have a same-scope source target, output mapping,
metric, tolerance, and uncertainty.

Expected implementation:

- Bind current waveform, phase timing, field, density, temperature, neutron
  yield, neutron timing, spectrum, anisotropy, and detector response only when
  local source packets exist.
- Record units, coordinate system, time origin, interpolation method, metric,
  tolerance, measurement uncertainty, model uncertainty, numerical uncertainty,
  and pass/fail rule.
- Cross-scope material may inform requirements but cannot pass PF-1000/Akel
  acceptance without a transfer rule.

Audit expectation:

- Codex will inspect `same_scope_source`, `waveform_phase`,
  `spatial_field_temperature`, `engineering_current_waveform_comparison`,
  `comparator_uq`, and `certificate_gate`.
- Codex will reject comparisons that use an experimental waveform as a drive or
  fit while claiming prediction.

### SSR-011 Numerical Fidelity

The simulator must prove that results are not artifacts of a grid, timestep,
backend, limiter, or restart path.

Expected implementation:

- Provide:
  - timestep convergence family;
  - mesh convergence family;
  - restart reproducibility;
  - split-continuation equivalence;
  - checkpoint/restart equivalence;
  - backend/precision declaration;
  - divergence budgets;
  - limiter-zero or physically bounded limiter proof;
  - artifact hashes.
- Long runs must preserve cumulative histories even when full step payloads are
  capped.

Audit expectation:

- Codex will run the existing experimental numerics/checkpoint/limiter CLI
  probes where practical.
- Codex will reject unbounded memory growth, history truncation that hides
  cumulative ledgers, or convergence claims without families.

### SSR-012 Certificate Gate

The simulator is accepted only when the certificate gate says it can be
accepted.

Expected implementation:

- `certificate_gate` must list every upstream packet and whether it is accepted.
- Acceptance requires all upstream packets to be accepted.
- Required negative tests must be present.
- Manifest must include command provenance, package/runtime versions, artifact
  hashes, source packet hashes, and review metadata.

Audit expectation:

- Codex will reject any final answer, README, UI, or release artifact that
  claims full first-principles readiness while `certificate_gate` remains
  blocked.

### SSR-013 Generalization

General DPF-machine authority requires at least one second device/shot to repeat
the full evidence path.

Expected implementation:

- Keep PF-1000/Akel assumptions explicit.
- Any second-scope deck must have independent source references, geometry,
  startup, power port, closure, comparator/UQ, and certificate packets.
- GV, May 15, May 16, Soto, and IPFS machines are candidate requirement/deck
  material until independently packetized and reviewed.

Audit expectation:

- Codex will reject a general DPF claim if only PF-1000/Akel has been exercised.

## Required Work Packages

### WP-0 Repo And Source Baseline

Deliverables:

- Confirm Python 3.12 runtime.
- Confirm source-truth index is current.
- Confirm module source-vetting is clean.
- Add a short implementation note listing changed files and the source packets
  used.

Required commands:

```bash
.venv312/bin/python --version
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py
git diff --check
```

Expected result:

- Source-truth exhaustion passes.
- Module source-vetting passes.
- No active physics module lacks source references.

### WP-1 Power Port Closure

Deliverables:

- Accepted or still-fail-closed implementation of the active power-port
  operator.
- Domain/interface packet.
- Sign packet.
- Time-centering packet.
- Electrode/interface work packet.
- Four-term or five-term energy ledger packet.
- Low-current `P/I` handoff/regularization packet.
- Negative tests.
- 100 ns, 1 us, and 12 us source-sign run attempts with artifacts.

Expected result:

- If unsolved: runtime explicitly reports the blocker and remains candidate.
- If solved: `power_port` packet can be promoted only after all source,
  residual, negative-test, and review gates are present.

Audit focus:

- `src/dpf/first_principles/power_port.py`
- `src/dpf/fields/hybrid_simulator.py`
- `src/dpf/fields/circuit_boundary.py`
- `tests/test_first_principles_runner.py`
- `tests/test_hybrid_3d_simulator.py`

### WP-2 Startup BVP

Deliverables:

- Implemented startup state generator or reviewed imported startup packet.
- Startup fields, particles, current density, ionization, temperatures, and
  sheath liftoff.
- Handoff interval into the field/PIC loop.
- Negative tests proving seeded/text-only startup remains rejected.

Expected result:

- `startup_bvp` no longer blocks only when all required channels are present.
- Until then, startup remains explicitly blocked.

Audit focus:

- `startup_bvp.py`
- `startup_breakdown.py`
- `deck.py`
- runner startup packet propagation.

### WP-3 Reviewed Geometry And Boundaries

Deliverables:

- Reviewed PF-1000 geometry mask packet.
- Material boundary packet.
- Electrode/source-interface labels.
- Particle and field boundary semantics.
- Grid/projection error packet.

Expected result:

- `boundary_policy` and deck-diff packets make geometric mismatch visible.
- Coarse geometry remains candidate unless resolution/projection review passes.

### WP-4 Long-Run Field/PIC/Electron Stability

Deliverables:

- Source-ordered predictor-corrector integration over long horizons.
- Cumulative histories independent of retained step payload count.
- Split-continuation and checkpoint/restart equivalence.
- No hidden state repair.

Expected result:

- 12 us run can be attempted and inspected without losing conservation and
  power-port ledgers.
- Failing horizons return clear blocker telemetry, not silent success.

### WP-5 Closure Completion

Deliverables:

- Closure packets for EOS, ionization, transport, radiation, ablation,
  impurities, anomalous resistance, restrike, collisions, stopping, and
  beam-target coupling.
- Unit tests and negative tests for each active closure.
- Runtime coupling into `physics_closure`.

Expected result:

- Active closures are source-scoped and energy-accounted.
- Missing closures remain visible blockers.

### WP-6 Neutron And Detector Authority

Deliverables:

- Mechanism-separated neutron production histories.
- Beam/ion distribution packet.
- Spectrum and anisotropy packet.
- Detector/activation/TOF response packet.
- UQ packet.

Expected result:

- `neutron_authority` does not depend on scalar yield alone.

### WP-7 Comparator, UQ, And Certificate

Deliverables:

- Same-scope targets with hashes, units, time origin, coordinate mapping,
  metrics, tolerances, and uncertainty.
- Comparator/UQ packet.
- Certificate packet with upstream acceptance matrix.
- Negative controls for draft, blocked, cross-scope, missing-UQ, missing-review,
  hidden-limiter, reduced-model fallback, and app-only evidence.

Expected result:

- Full first-principles readiness is claimed only if the certificate gate
  passes.

## Required Submission Format

Every outside-team submission must include:

1. Summary of intent.
2. Changed file list.
3. Requirement IDs touched: `SSR-001` through `SSR-013`, plus work package IDs.
4. Source evidence table with local file, line range, equation/claim, and
   implementation file.
5. Runtime command list.
6. Test command list.
7. Artifact list with paths and hashes.
8. Negative tests added or updated.
9. Remaining blockers.
10. AI/tool usage disclosure:
    - tool name;
    - whether it produced code, formulas, tests, or prose;
    - which local sources were used to verify the output;
    - any AI suggestion rejected because it lacked local source support.

Submission template:

```text
Implementation Summary:

Changed Files:

Requirements Touched:

Source Evidence:
| Local source path:lines | Claim/equation | Implemented in | Status |

Commands Run:

Artifacts:
| Path | Purpose | SHA256 |

Negative Tests:

Remaining Blockers:

AI/Tool Disclosure:
```

## Codex Audit Methodology

Codex will audit submissions in this order.

### Audit Phase 1: Worktree And Diff Hygiene

Commands:

```bash
git status --short
git diff --check
git diff --stat
```

Audit decisions:

- Reject unrelated file churn.
- Reject generated artifacts that are not referenced in the submission.
- Reject hidden changes to source-truth files unless explicitly requested.

### Audit Phase 2: Source-Truth Verification

Commands:

```bash
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py
```

Manual checks:

- For every source claim, Codex will open the cited local file and line range.
- Codex will verify that the line range supports the claim made.
- Codex will reject fabricated line ranges, broad unspecific citations, and
  citations to external AI/web material as source truth.

### Audit Phase 3: Status And Claim Safety

Searches:

```bash
rg -n "accepted|validated|first-principles ready|full first-principles|can_support_first_principles_acceptance" src docs tests
rg -n "lee|snowplow|fcr|empirical|fit|scaling" src/dpf/first_principles src/dpf/fields
```

Audit decisions:

- Reject promotion of candidate packets without evidence.
- Reject reduced-model authority in first-principles paths.
- Reject UI/doc claims that exceed packet status.

### Audit Phase 4: Focused Runtime Tests

Baseline commands:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_runner.py -q
.venv312/bin/python -m pytest tests/test_hybrid_3d_simulator.py -q
.venv312/bin/python -m pytest tests/test_first_principles_input_deck.py -q
.venv312/bin/python -m pytest tests/test_first_principles_*.py -q
.venv312/bin/python -m ruff check src/dpf/first_principles src/dpf/fields tests
```

Additional tests will be selected based on touched files.

Audit decisions:

- Reject if focused tests fail.
- Reject if tests only assert existence and not fail-closed behavior.
- Reject if new physics has no negative tests.

### Audit Phase 5: CLI And Artifact Inspection

Representative commands:

```bash
.venv312/bin/dpf first-principles-3d \
  --deck-preset pf1000_akel_16kv \
  --steps 2 \
  --output results/audit_first_principles_3d_smoke.json

.venv312/bin/dpf experimental-whole-shot \
  --deck-preset pf1000_akel_16kv \
  --steps 20 \
  --target-time-s 1.0e-10 \
  --dt-policy combined-cfl \
  --output results/audit_experimental_whole_shot_smoke.json
```

Codex will inspect:

- `scientific_status`
- `validation_packet`
- `telemetry_packets`
- `power_port`
- `startup`
- `deck_diff`
- `limiter_readiness`
- `numerical_fidelity`
- `neutron_authority`
- `comparator_uq`
- `certificate_gate`
- `can_support_first_principles_acceptance`

Audit decisions:

- Reject if a run exits successfully while hiding failed duration,
  nonfinite-state, limiter, source, or packet blockers.
- Reject if artifacts are not reproducible from submitted commands.

### Audit Phase 6: Physics-Specific Review

Codex will review each touched work package.

Power port:

- Verify domain/interface labels.
- Verify sign convention.
- Verify time-centering.
- Verify low-current `P/I` behavior.
- Verify negative `J.E` is signed and not silently clipped.
- Verify residual budget terms.

Startup:

- Verify source-backed BVP channels.
- Verify seeded/text-only paths remain rejected.
- Verify handoff into runtime is explicit.

Geometry:

- Verify source values.
- Verify mask dimensions and projection error.
- Verify material/surface labels.

Closures:

- Verify validity regimes.
- Verify units.
- Verify energy accounting.
- Verify missing effects are blocked.

Neutrons:

- Verify mechanism separation.
- Verify detector response.
- Verify UQ.

Numerics:

- Verify convergence/restart/continuation evidence.
- Verify backend/precision claims.
- Verify limiter accounting.

### Audit Phase 7: Final Verdict

Codex will return one of these verdicts:

- `accept_engineering_progress`: implementation is useful and honest, but not
  accepted first-principles.
- `request_changes`: implementation has code/test/source gaps.
- `reject_overclaim`: implementation promotes unsupported physics or hides
  blockers.
- `accept_certificate_candidate`: all packets needed for an engineering firm
  review package are present, but final engineering sign-off is still external.
- `accepted_first_principles_ready_for_external_review`: only possible when
  certificate gate and all upstream packets are accepted.

## Rejection Criteria

Codex will reject the submission if any of these are true:

- Physics claims cite no local `KnowledgeReference/` source.
- AI-generated formulas are implemented without local source verification.
- A reduced model drives a first-principles result.
- Sigma/quasi-TEM line voltage is promoted without a local source packet.
- `seeded_layer` startup is accepted.
- PF-1000/Akel values are mixed with another shot or machine without transfer
  packet.
- Negative `J.E`, current floors, back-EMF clips, pressure floors, density
  floors, timestep caps, or state repairs are hidden.
- Runtime artifacts omit packet statuses.
- Tests do not include negative controls.
- Certificate or docs claim readiness while upstream packets remain blocked.

## Minimum Expected Results Before External Engineering Review

Before the project is handed to an engineering firm for serious testing, the
repo must produce a single review packet containing:

1. Package-native command provenance.
2. Source-locked deck and deck-diff packet.
3. Startup packet.
4. Reviewed or candidate geometry/material packet.
5. Power-port packet with source/domain/sign/time-centering/residual status.
6. Conservation ledger.
7. Limiter ledger.
8. Numerical-fidelity packet.
9. Physics-closure packet.
10. Same-scope comparator/UQ packet.
11. Mechanism-separated neutron packet.
12. Certificate gate.
13. Artifact hashes.
14. Explicit list of remaining blockers.

If any packet is candidate or blocked, the engineering firm can still test the
experimental simulator, but the artifact must not be labeled as accepted or
validated first-principles.

## Immediate Priority Order

1. Finish source-sign power-port closure and long-run segmented proof.
2. Build or import source-backed startup BVP.
3. Replace coarse PF-1000 geometry projection with reviewed masks and material
   surfaces.
4. Complete physics closure packets and runtime coupling.
5. Complete neutron mechanism and detector-response authority.
6. Complete same-scope comparator/UQ binding.
7. Complete convergence/restart/backend/limiter proof.
8. Write certificate packet for engineering review.

## Current Expected State After This Spec

The correct near-term state is not "finished simulator." The correct near-term
state is:

```text
experimental 3-D first-principles-path simulator
  + source-truth citations
  + honest blocker packets
  + power-port/source-sign work progressing
  + whole-shot experimental artifacts
  + no unsupported acceptance claim
```

The project reaches the finish line only when the certificate gate and all
upstream packets stop failing closed for the same machine/shot scope.
