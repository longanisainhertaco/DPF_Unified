# First-Principles Physics Acceptance Promotion Protocol (2026-05-20)

Purpose: convert the V2 blocker ledgers into a controlled promotion path where
individual physics modules may become accepted only after independent source
review, implementation review, and executable verification all agree. This
protocol does not accept any physics today; it defines the gate that must be
passed before a future acceptance can be made.

Controlling inputs:

- `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`
- `docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv`
- `docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv`
- repo-local `KnowledgeReference/` records and promoted KR target packets
- current source code, tests, run manifests, conservation ledgers, and
  certificate gates

## Acceptance States

Every first-principles module must be in exactly one state:

1. `blocked_or_absent`: a source, equation, parameter, or same-scope datum is
   missing. Runtime must fail closed or label output as candidate-only.
2. `source_available`: a KR text record or source packet exists, but typed
   target extraction is not complete.
3. `target_extracted`: exact equations, constants, tables, assumptions, scope,
   uncertainty, and line/page citations are extracted into typed evidence.
4. `implemented_candidate`: code consumes the typed evidence and has focused
   tests, but independent verification is incomplete.
5. `verified_implementation`: source, code, tests, and numerical checks pass
   all three verification lanes below.
6. `accepted_physics_module`: the module can be used as accepted
   first-principles physics for its declared scope.
7. `validated_scope_certificate`: a separate, higher state requiring
   same-scope experimental comparison, uncertainty propagation, comparator
   review, and certificate-gate approval.

The whole DPF shot cannot be called accepted or validated until every required
module for the selected shot scope reaches at least `accepted_physics_module`
and the shot-level comparator reaches `validated_scope_certificate`.

## Triple Verification Rule

No row may set `accepted_physics_allowed=true`, `accepted_runtime_claim=true`,
or `can_support_first_principles_acceptance=true` unless all three lanes pass
for the same source scope and code revision.

### Lane 1 - Other-Team Evidence And Implementation Packet

The other team must deliver one packet per physics item:

- exact KR source paths or promoted source paths;
- exact lines, pages, figure/table identifiers, equations, constants, units,
  assumptions, and scope tags;
- typed target extraction output and hashes;
- code paths changed and call graph surface;
- tests added or changed;
- run artifacts, manifests, and residual/conservation outputs;
- a written statement of what remains blocked, candidate-only, or accepted.

The packet fails if it uses external material that has not been promoted into
KR, merges PF-1000 16 kV evidence with full-energy evidence without an explicit
scope tag, or relies on tuned fit constants where first-principles equations
are required.

### Lane 2 - Codex Independent Audit

Codex must independently re-audit the same item from the local source of truth:

- re-read the cited KR lines/pages and verify the extracted target values;
- verify units, dimensions, assumptions, and device/shot scope;
- inspect implementation for hard-coded unsupported constants, hidden reduced
  models, empirical fits, or fail-open defaults;
- rerun the focused tests and review the artifact manifests;
- check that source-supported, candidate, external-blocked, and absent states
  remain distinct.

The audit fails if any accepted claim rests on wrong-scope evidence,
non-reproducible extraction, missing uncertainty, missing conservation budget,
or code that can silently fall back to a placeholder path.

### Lane 3 - Executable Reproducibility Gate

The repository must enforce the promotion mechanically:

- ledger parser confirms all required rows and field counts;
- tests verify all accepted rows have three passed verification records;
- unit and property tests cover the implemented equations and limits;
- dimensional and units checks pass;
- conservation and power-port residual budgets pass;
- convergence, restart reproducibility, and backend-parity checks pass where
  the module affects time evolution;
- certificate gate rejects missing same-scope comparator evidence;
- artifact hashes reproduce from the committed inputs.

The executable gate is not optional. A human review cannot override a failing
test or missing artifact; it can only explain the failure and decide whether to
change the scope or implementation.

## Promotion Requirements By Module Type

### Source Extraction Modules

Required for Bennett startup, UCSD/Beg liftoff context, Stepniewski geometry,
Scholz/Gribkov fast-ion evidence, Bernard thermonuclear context, and
Braginskii rendered/OCR transport pages.

Acceptance requires:

- target packet with exact source citations and scope tag;
- independent source reread by Codex;
- one automated test that fails if the packet is removed, renamed, or loses its
  blocked/candidate state.

Source extraction alone can reach `target_extracted`; it cannot reach
`accepted_physics_module` without implementation tests.

### Runtime Physics Modules

Required for startup BVP, power port, Braginskii/electrical transport, D2
atomic/molecular kinetics, two-temperature coupling, EOS/radiation,
ablation/impurity surfaces, and neutron source terms.

Acceptance requires:

- typed source packet consumed directly by code;
- no measured-or-placeholder path in accepted mode;
- finite units and dimensional checks;
- equation-level tests against source examples or independently computed
  reference values;
- conservation/residual budget if the module exchanges mass, charge,
  momentum, or energy;
- explicit fail-closed output if a dependency is missing.

Runtime modules can reach `accepted_physics_module`; they still do not validate
the whole shot by themselves.

### Numerical Acceptance Modules

Required for solver stepping, field/current coupling, power accounting,
restart, backend parity, limiter-zero proof, and long-duration whole-shot runs.

Acceptance requires:

- deterministic restart reproduction;
- convergence or refinement trend appropriate to the method;
- backend parity when more than one backend claims the feature;
- residual ledger proving energy, charge, and relevant source/sink accounting;
- 12 us or selected-shot duration only after the startup and transport gates
  are source-backed.

### Comparator And Certificate Modules

Required for any engineering-firm-facing validation claim.

Acceptance requires:

- locked shot scope;
- same-scope current, voltage or terminal power, density/temperature,
  radiation, neutron yield, neutron spectrum, anisotropy, and detector-response
  evidence as applicable to the claim;
- uncertainty propagation;
- comparator report;
- certificate gate that rejects cross-scope substitutions.

Comparator/certificate work is the only path to
`validated_scope_certificate`.

### Package-Native 3-D Acceptance Contract

The package-native 3-D runner cannot be promoted by virtue of running
end-to-end. It must emit the same acceptance contract consumed by the legacy
`first_principles_mhd` readiness gate and by the certificate gate.

Required fields:

- effective backend and requested backend;
- requested run mode and execution mode;
- limiter ledger and backend-scope status;
- manifest hashes and source-packet hashes;
- conservation and power-port residual budgets;
- telemetry-packet hashes for startup, geometry, closure, neutron, comparator,
  numerical, and certificate packets;
- explicit candidate/blocked/accepted state for every packet.

The contract fails if package-native evidence is available only through a CLI
engineering wrapper or if any acceptance gate has to infer field meaning from
runner-specific names.

### Claim-Limited Certificates And Excluded Observables

Some selected scopes may lack same-scope measurement for a requested observable.
That does not permit a fake acceptance state. The only allowed exclusion state
is `observable_excluded_not_validated`.

Rules:

- excluded observables do not count as accepted comparator evidence;
- excluded observables must appear in the certificate claim text;
- the simulator must still provide model initialization, source basis,
  sensitivity/UQ, and negative tests for excluded physics if that physics
  affects other validated observables;
- any certificate that claims validation of an excluded observable must fail.

The rejected state name is `caveat_accepted`; it is too easy to confuse with
actual accepted evidence.

## Sprint 5/6 Execution Plan

### Sprint 5A - Source Packets

Promote and target-extract the local/high-priority packets first:

1. Bennett 2017 startup channels CH03, CH04, CH07, CH08.
2. Braginskii 1965 rendered/OCR coefficient and equation pages.
3. Scholz/Gribkov 2007 Part II fast-ion and anisotropy context.
4. Bernard 1977 historical neutron and thermonuclear-prefactor context.
5. UCSD/Beg current-sheath initiation lines 597-601, 631-660, and 642-660.
6. Stepniewski 2004 hardware-scope review for hollow-anode context.

Exit condition: target packets exist, are hash-stable, and still do not accept
runtime physics unless code and verification are complete.

### Sprint 5B - Runtime Candidate Integration

Wire source packets into fail-closed registries:

1. startup BVP channel registry;
2. geometry source-scope registry;
3. transport closure registry;
4. power-port/conservation registry;
5. neutron mechanism registry.

Exit condition: every registry can report `source_supported`,
`implemented_candidate`, `external_blocked`, or `absent` without using
placeholders in accepted mode.

### Sprint 5B-0 - Acceptance Contract Plumbing

Before promoting any package-native 3-D physics, implement the
`package_native_3d_acceptance_contract` gate:

1. add shared package-native acceptance fields to runner, CLI, server readiness,
   and certificate packets;
2. map legacy `first_principles_mhd` backend-scope checks onto the
   package-native contract without making the package-native path accepted by
   default;
3. add negative tests for missing backend labels, missing limiter ledger,
   missing backend-scope status, mismatched manifest hashes, and hidden
   engineering-only telemetry.

Exit condition: a package-native run can explain exactly why it is accepted,
candidate-only, or blocked using the same schema as the legacy readiness gate.

### Sprint 5C - Triple Verification Pass

For each item in
`docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv`:

1. other team submits an evidence/implementation packet;
2. Codex performs a source-and-code audit from local KR;
3. executable gates reproduce source hashes, tests, residuals, and manifests.

Exit condition: any row that remains incomplete stays
`accepted_physics_allowed=false`; rows that pass all gates may be proposed for
`accepted_physics_module` in a new commit with the three verification artifacts.

### Sprint 6 - Shot-Level Assembly

Only after module-level accepted physics exists:

1. lock PF-1000 full-energy or Akel 16 kV as the comparator scope;
2. assemble the whole-shot deck from accepted module packets;
3. run segmented duration tests to the selected shot time;
4. run same-scope comparator and UQ;
5. issue or reject the validation certificate.

## Audit Instructions For Future Team Submissions

Codex will audit each submitted packet in this order:

1. Check the packet names the exact ledger row and blocker IDs.
2. Re-read every cited `KnowledgeReference/` path and verify line/page targets.
3. Confirm source scope matches runtime scope.
4. Confirm extracted equations/constants/tables have units and uncertainty.
5. Inspect code paths for placeholder, empirical, measured, or reduced-model
   fallbacks in accepted mode.
6. Rerun the exact tests and any required convergence/restart/backend parity
   checks.
7. Verify artifact hashes and manifests.
8. Confirm all three verification lanes passed at the same commit.
9. Confirm findings docs and ledgers are updated.
10. Reject promotion if any acceptance field is true before all gates pass.

## Current Verdict

As of this protocol, every physics item remains unaccepted. The project now has
a defined route to acceptance, but acceptance must be earned one module at a
time through the triple verification rule above.
