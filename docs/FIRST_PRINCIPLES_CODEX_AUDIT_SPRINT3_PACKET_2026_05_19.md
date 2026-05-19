# First-Principles DPF Sprint 3 Packet Audit And Next Instructions

Date: 2026-05-19

Repository: `/Users/anthonyzamora/dpf-unified`

Branch reviewed: `codex/corpus`

HEAD reviewed: `173e10da16beb5f6422d0a40656b99e70eb867bb`

Audit log: `/private/tmp/dpf-unified-audit-logs/20260519T183301Z/summary.md`

Scope: audit of the other team's Sprint 3 research/spec packet submission, audit
of the Codex Sprint 2.2 closure work, and explicit next-sprint instructions for
work that moves the project toward an experimental full-shot, first-principles,
3D DPF simulator.

## Verdict

Decision: accept as research/spec progress only; request changes before any
Sprint 3 implementation claims are made.

The current repo is clean and the package-native audit gate passed at HEAD
`173e10d`. Codex Sprint 2.2 is accepted as a control-plane/runtime cleanup:
the segmented whole-shot combiner and traceability fixes are now green under the
periodic audit.

The other team's Sprint 3 packets are useful and mostly fail closed. They
correctly avoid claiming validation, acceptance, or implemented runtime physics.
They are not enough to close any first-principles blocker yet. The next sprint
must first clean packet state and citation precision, then implement the first
source-tagged runtime slice: PF-1000/Akel geometry packets and the `Sigma_p`
surface inventory needed by the Auluck power-port terms II, IV, V, and VI.

## Audit Method

1. Treated `KnowledgeReference/` and tracked verified extracts as the only
   scientific authority.
2. Read the Sprint 3 packet files under
   `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/`.
3. Checked packet claims for:
   - validation or acceptance overclaiming;
   - stale Sprint 2.2 status;
   - source-reference precision;
   - implemented-code claims without code and tests;
   - readiness for the next implementation sprint.
4. Ran the repo periodic audit:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Result: PASS, including `git_status_clean`, `source_truth_exhaustion`,
`module_source_vetting`, `focused_pytest`, and `broad_first_principles_pytest`.

## Codex Work Audit

Accepted for the current sprint:

- Current HEAD `173e10d` passed the full periodic audit.
- Sprint 2.2 combiner/traceability closure is green.
- `merge_cumulative_ledgers()` and `combine_whole_run_artifacts()` are no
  longer the immediate stop-the-line blocker for Sprint 3 research.
- The current acceptance state remains honest: the simulator still does not
  have an accepted 12 us full-shot run, accepted PF-1000 geometry, accepted
  startup packet, accepted closure registry, accepted neutron mechanism packet,
  or same-scope comparator/certificate packet.

Remaining Codex-side obligations:

- Keep reduced Lee/snowplow paths as baselines only.
- Keep first-principles runtime claims fail-closed until source packets, code,
  tests, and audit evidence land together.
- Do not promote packet text into runtime defaults unless every numeric field
  has a local-source reference and a conflict policy.
- Continue using `.venv312` as the active Python 3.12 lane.

## Other-Team Work Audit

Accepted as research/spec progress:

- `WP_N3_GEOMETRY_SOURCE_PACKET.md` correctly identifies PF-1000 geometry
  conflicts and blocks missing dimensions instead of inventing them. The packet
  also gives concrete implementation recommendations for a
  `PF1000GeometryPacket`, source-tagged constructors, material sub-masks, and
  per-mask hashes.
- `WP_N3_SIGMA_P_RUNTIME_INTERFACE_SPEC.md` correctly keeps the Auluck
  `Sigma_p` moving-boundary terms independent and rejects closure substitution.
  Its negative controls are directly useful for the next implementation sprint.
- `WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md` correctly says no closure can yet
  support first-principles acceptance and treats PlasmaPy as a candidate
  cross-check rather than authority.
- `WP_N6_NEUTRON_AUTHORITY_PACKET.md` correctly keeps scalar neutron yield out
  of mechanism authority and identifies the missing kinetic/detector-response
  chain.
- `WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md` correctly stays fail-closed at the
  certificate level and requires same-scope packets before reviewable
  engineering claims.
- `WP_N4_PERFORMANCE_AND_RUN_PLAN.md` correctly forbids a 12 us completion claim
  until the manifest actually says `horizon_complete=true`.

Not accepted as implemented physics:

- No Sprint 3 packet closes a blocker by itself.
- No packet provides accepted PF-1000 material masks or source-tagged runtime
  geometry.
- No packet provides a runtime `Sigma_p` face inventory.
- No packet provides a startup BVP solver, accepted closure registry, kinetic
  neutron mechanism authority, same-scope comparator, UQ packet, or certificate.

## Findings

### F1 - Sprint 3 packet state is stale after Sprint 2.2 closure

Severity: high for packet control, not a physics failure.

Evidence:

- `sprint_3/PENDING.md:1-30` still says Sprint 3 is deferred and that no Sprint
  3 work has started.
- Seven Sprint 3 research/spec packets now exist in the same folder.
- `WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md:212-215` says WP-N5 and WP-N2 are
  not delivered and that WP-N1B terms II/IV/V/VI are blocked because "Sprint
  2.2 open".
- `WP_N4_PERFORMANCE_AND_RUN_PLAN.md:155` and `207` still describe the
  three-restart combiner as a known Sprint 2.2 bug, even though current HEAD
  passed the Sprint 2.2 closure audit.

Required correction:

- Rewrite `sprint_3/PENDING.md` as a Sprint 3 status ledger:
  - research/spec packets delivered;
  - runtime implementation still pending;
  - no accepted physics blockers closed.
- Update WP-N7 and WP-N4 stale wording:
  - WP-N2 research delivered, startup BVP runtime/accepted packet not delivered;
  - WP-N5 research delivered, closure registry runtime/accepted packet not
    delivered;
  - WP-N1B Sprint 2.2 control work closed, but `Sigma_p` terms II/IV/V/VI still
    blocked pending WP-N3 runtime `Sigma_p` inventory and Sprint 4 computation.

### F2 - Some citations are not implementation-grade

Severity: high before implementation.

Evidence:

- `WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md` uses shorthand citations such as
  `[KR: same file L201-203]` and similar lines throughout the packet.
- `WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md:418-426` uses shorthand references
  such as `[KR: ...formulary...]` and `[KR: ...lee-saw-part-1...]`.

Required correction:

- In every actionable table, negative-control test, runtime field definition,
  and proposed implementation rule, replace shorthand citations with exact
  repo-relative references:
  - `KnowledgeReference/<file>.md:Lx-Ly`, or
  - `docs/external_team_submissions/.../<verified_extract>.md:Lx-Ly`.
- Shorthand citations are allowed only in clearly marked narrative examples,
  not in anything a coder could treat as an implementation source.

### F3 - Packet hygiene tests do not yet guard Sprint 3

Severity: medium now, high before accepting the next packet.

The current `tests/test_external_team_submission_package.py` protects Sprint 2
package consistency, but it does not yet enforce Sprint 3 packet hygiene.

Required correction:

- Add Sprint 3 package tests that fail on:
  - live `sprint_3/PENDING.md` text claiming Sprint 3 has not started after
    packet files exist;
  - stale phrases: `Sprint 2.2 open`, `WP-N2 not delivered`, `WP-N5 closure
    registry not delivered`;
  - shorthand `[KR: same file ...]` or `[KR: ...]` in actionable rows;
  - ambiguous acceptance wording such as `accepted as comparator only` when the
    real status is `candidate_comparator_only`;
  - any Sprint 3 packet claiming `implemented` without a same-diff code and test
    change.

### F4 - The next implementation must start with geometry and `Sigma_p`

Severity: blocking for power-port acceptance.

Reason:

- Auluck terms II, IV, V, and VI require the moving `Sigma_p` boundary surface,
  oriented face areas, face-centered field samples, material velocity, and
  resistivity.
- Without the PF-1000/Akel geometry partition, the runtime cannot know which
  faces are `Omega`, `Sigma`, `Sigma_p`, electrode/source interface, chamber
  wall, cathode rods, insulator, or PML/open boundary.
- Therefore power-port acceptance cannot be solved by editing
  `power_port.py` alone. The geometry and surface packet must exist first.

Required correction:

- Implement the source-tagged PF-1000 geometry packet and material partition
  before attempting to compute the Auluck `Sigma_p` terms.

### F5 - PlasmaPy must remain a cross-check, not authority

Severity: medium.

`WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md` correctly states that PlasmaPy is an
external candidate cross-check only. The next implementation must preserve that
policy in code and tests.

Required correction:

- A missing, failed, or disagreeing PlasmaPy audit must not promote or reject a
  local-source closure by itself.
- A PlasmaPy disagreement outside tolerance must set a review-required telemetry
  field and keep acceptance false until a local-source review resolves it.

## Next Sprint: Sprint 3.1 To Sprint 3.3

Overall sprint goal:

Convert the Sprint 3 research packets into audit-grade, source-tagged runtime
foundations for an experimental first-principles full-shot simulator, without
claiming validation or acceptance.

Hard exit rule:

The sprint is complete only when packet hygiene tests pass, source-tagged
PF-1000/Akel geometry exists in runtime code with masks and hashes, and a
`Sigma_p` surface packet can be built or fail closed with precise blockers.

### Sprint 3.1 - Packet Hygiene And Status Correction

Owner: other engineering team.

Allowed files:

- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/*.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/README.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/THREE_SPRINT_FINAL_SUMMARY.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/CHANGELOG.md`
- `tests/test_external_team_submission_package.py`

Required work:

1. Replace `sprint_3/PENDING.md` with a status ledger:
   - `research_packets_delivered=true`;
   - `runtime_implementation_delivered=false`;
   - `first_principles_acceptance=false`;
   - list each delivered Sprint 3 packet path;
   - list each still-blocked runtime artifact.
2. Normalize citations in WP-N2 and WP-N5:
   - no actionable row may use `[KR: same file ...]`;
   - no actionable row may use `[KR: ...formulary...]`;
   - every equation, coefficient, numeric parameter, source phrase, and
     proposed runtime default must cite exact local paths and line ranges.
3. Update stale Sprint 2.2 references in WP-N4 and WP-N7:
   - identify Sprint 2.2 as closed at Codex HEAD `173e10d`;
   - keep Auluck `Sigma_p` runtime computation blocked pending WP-N3/Sprint 4.
4. Add Sprint 3 hygiene tests to `tests/test_external_team_submission_package.py`.
5. Update `CHANGELOG.md` with every non-HEAD commit covered by the packet.

Acceptance tests:

```bash
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py -q -rx
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Expected output:

- No stale Sprint 2.2 wording.
- No live Sprint 3 "pending/no work started" contradiction.
- No shorthand citations in actionable packet content.
- Periodic audit PASS.

### Sprint 3.2 - Implement Source-Tagged PF-1000/Akel Geometry Packet

Owner: other engineering team.

Allowed files:

- `src/dpf/fields/source_geometry.py`
- `tests/test_source_geometry_packet.py`
- `tests/test_first_principles_geometry.py`
- `docs/DPF_REQUIREMENTS_BASELINE.md`
- `docs/SRS_TRACEABILITY_MATRIX.csv`
- `docs/SRS_TRACEABILITY_MATRIX.json`
- packet docs under the Sprint 3 folder, only for traceability updates.

Required work:

1. Add a frozen `PF1000GeometryPacket` dataclass.
2. Every field must carry:
   - value;
   - unit;
   - source status: `source_supported`, `candidate`, `conflict`, or `blocked`;
   - source reference path and line range when value is present;
   - conflict group ID when multiple source revisions disagree.
3. Add source-tagged constructors, at minimum:
   - `PF1000GeometryPacket.krauz_2012()`;
   - `PF1000GeometryPacket.akel_shot_12581()`;
   - optional `PF1000GeometryPacket.scholz_gribkov_revision()` only if the
     packet can keep conflicts explicit.
4. Do not average conflicting geometry values.
5. Leave missing fields as `None` with a `blocked_*` reason:
   - anode bore radius;
   - anode bore length;
   - insulator outer radius;
   - insulator wall thickness;
   - any backplate/source-interface dimensions not locally sourced.
6. Build deterministic material partitions:
   - `anode_material_faces`;
   - `cathode_rod_faces`;
   - `insulator_material_faces`;
   - `chamber_wall_faces`;
   - `backplate_source_interface_faces`;
   - `pml_or_open_boundary_faces`.
7. Preserve the Auluck top-level partition:
   - `Omega`;
   - `Sigma`;
   - `Sigma_p`;
   - excluded electrode/source interface.
8. Emit SHA-256 hashes for every mask and include those hashes in the geometry
   manifest.
9. Fail closed if masks overlap, if a partition is not exhaustive where it must
   be exhaustive, or if a source-supported field lacks a source reference.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_source_geometry_packet.py tests/test_first_principles_geometry.py -q -rx
```

Minimum test cases:

- source constructors preserve conflicting values without averaging;
- blocked fields remain `None` and carry blocker IDs;
- all source refs point to existing local files;
- material masks do not overlap;
- `Omega` and excluded source interface do not overlap;
- PML/open boundary is never tagged as a sourced PF-1000 machine dimension;
- per-mask hashes are stable across repeated builds;
- under-resolved rods/insulator features fail closed with a review-required
  blocker.

### Sprint 3.3 - Implement `Sigma_p` Surface Packet Plumbing

Owner: other engineering team.

Allowed files:

- `src/dpf/fields/source_geometry.py`
- `src/dpf/first_principles/power_port.py`
- `tests/test_first_principles_power_port.py`
- `tests/test_source_geometry_packet.py`
- `docs/DPF_REQUIREMENTS_BASELINE.md`
- `docs/SRS_TRACEABILITY_MATRIX.csv`
- `docs/SRS_TRACEABILITY_MATRIX.json`
- Sprint 3 packet docs, only for traceability updates.

Required work:

1. Add a `SigmaPSurfacePacket` or equivalent frozen structure that records:
   - face IDs;
   - oriented `dS` vectors;
   - face area;
   - outward normal convention;
   - moving/stationary classification;
   - `Omega` side and excluded-interface side;
   - material class;
   - mask/source hashes;
   - source geometry packet ID;
   - centering/quadrature metadata as engineering metadata, not source-backed
     physics.
2. Build `Sigma_p` from the reviewed geometry/material partition.
3. Classify stationary boundaries explicitly. Do not infer stationarity inside
   `power_port.py`.
4. Thread face-centered field-sampling placeholders only as blocked operands:
   - `B`;
   - `E`;
   - `J`;
   - material velocity `v`;
   - resistivity `eta`.
5. Keep Auluck terms II/IV/V/VI blocked until all operands exist. Do not compute
   any term by closure substitution.
6. Add negative controls:
   - missing `Sigma_p` blocks II/IV/V/VI;
   - missing `v` blocks II/IV/VI;
   - missing `eta` blocks V;
   - stationary faces contribute exactly zero to motional terms;
   - any closure-derived term is rejected;
   - missing sign convention blocks the residual.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_power_port.py tests/test_source_geometry_packet.py -q -rx
```

Expected output:

- `power_port.py` can consume a `Sigma_p` packet shape.
- Terms II/IV/V/VI remain blocked if operands are missing.
- No `I*V - other_terms` substitution is accepted.
- The code is ready for Sprint 4 term computation once field samplers and
  time-centering are available.

## Sprint 3 Parallel Research Track

The team may continue research outside Sprint 3.2/3.3 only if it does not edit
the geometry or power-port implementation files.

Allowed parallel packets:

1. WP-N2 startup BVP implementation plan:
   - one channel per source-backed equation;
   - exact references only;
   - blocked fields for missing secondary emission/photoemission/surface
     coefficients;
   - no inferred startup density as a default.
2. WP-N5 closure registry implementation plan:
   - exact local source packet per closure;
   - validity regime gates;
   - PlasmaPy disagreement telemetry;
   - no acceptance from PlasmaPy alone.
3. WP-N6 neutron authority implementation plan:
   - mechanism-separated yield interface;
   - ion distribution requirements;
   - detector response/scatter packet blockers;
   - scalar yield allowed only as comparator, not mechanism authority.
4. WP-N7 certificate/comparator wording cleanup:
   - replace any `accepted as comparator only` wording with
     `candidate_comparator_only`;
   - distinguish research packet delivered from runtime/accepted packet
     delivered.

## Audit Criteria For The Next Submission

I will audit the next submission using these gates:

1. `git status --short --branch` must be clean before the audit run.
2. Packet claims must match file reality. If packet files exist, no packet may
   say the sprint has not started.
3. Any `implemented` claim must have code and tests in the same submission.
4. Any local-source physics claim must cite exact paths and line ranges.
5. No reduced model, training-data value, outside web source, or PlasmaPy result
   may become a source of truth.
6. Missing physics must fail closed with a named blocker, not a guessed default.
7. The package tests below must pass:

```bash
.venv312/bin/python -m pytest \
  tests/test_external_team_submission_package.py \
  tests/test_source_geometry_packet.py \
  tests/test_first_principles_geometry.py \
  tests/test_first_principles_power_port.py \
  tests/test_srs_traceability_export.py \
  -q -rx
```

8. The periodic audit must pass:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

## Next-Team Prompt

Use this exact goal statement for the next engineering sprint:

> Continue the DPF-Unified first-principles simulator build. Treat
> `KnowledgeReference/` and tracked verified extracts as the only physics source
> of truth. First fix Sprint 3 packet hygiene and stale Sprint 2.2 wording, then
> implement source-tagged PF-1000/Akel geometry and `Sigma_p` surface packet
> plumbing. Do not claim validation, acceptance, or full-shot completion. Every
> implemented field must have code, tests, source references, and fail-closed
> blockers for missing inputs. The sprint exit condition is passing packet
> hygiene tests, source-tagged geometry/mask/hash tests, `Sigma_p` fail-closed
> power-port tests, traceability tests, and the full Codex periodic audit.

