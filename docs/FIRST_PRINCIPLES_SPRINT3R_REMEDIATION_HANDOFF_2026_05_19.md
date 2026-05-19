# First-Principles DPF Sprint 3R Remediation Handoff

Date: 2026-05-19

Repository: `/Users/anthonyzamora/dpf-unified`

Branch: `codex/corpus`

Controlling audit:
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_COMPLETION_2026_05_19.md`

Previous completion handoff:
`docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md`

## Sprint Goal

Close Sprint 3 for real.

Sprint 3R is complete only when every Sprint 3 completion-audit finding is
fixed or explicitly converted into a typed, source-backed, fail-closed blocker
with regression tests and traceability updates. Sprint 3R is not allowed to
claim validation, engineering acceptance, first-principles predictive authority,
or a complete 12 us full shot.

The desired end state is:

- the packet ledgers agree with the code and tests;
- runtime fail-closed packets cannot be promoted by caller-declared metadata;
- PF-1000 geometry and `Sigma_p` packets either use source-backed masks or
  clearly block source-missing masks;
- neutron authority cannot be satisfied by scalar yield or target-only metadata;
- numerical merged ledgers preserve the same channels emitted by live manifests;
- traceability paths point to real files;
- the periodic audit and all Sprint 3R focused tests pass.

## Non-Negotiable Rules

1. `KnowledgeReference/` and tracked verified extracts are the only scientific
   authorities.
2. PlasmaPy, online references, AI tools, and open-source code may be used only
   as engineering aids or cross-checks. They cannot promote a physics claim.
3. Do not infer missing geometry or material values into accepted defaults.
   Inferred values must be labeled `inferred_candidate` or
   `candidate_projection_not_source_mask` and must block acceptance.
4. Do not use scalar neutron yield as mechanism authority.
5. Do not compute or claim Auluck `Sigma_p` terms II/IV/V/VI from residual
   closure or terminal `I*V` substitution.
6. Do not claim startup authority until the typed startup packet has computed,
   reviewed, source-backed channels.
7. Do not promote any requirement to `implemented` or `accepted` unless the
   code, tests, local source references, and traceability artifacts all support
   that status.
8. If a test passes by enforcing stale or weak behavior, change the test.

## Required Work Sequence

The team must work through these packages in order. Later packages may begin
read-only research in parallel, but code merges should preserve this order to
avoid hiding upstream contradictions.

## S3R.1 - Control Plane, Ledgers, and Citation Hygiene

Audit findings covered: A10, A11, A12.

Allowed files:

- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/README.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/THREE_SPRINT_FINAL_SUMMARY.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/CHANGELOG.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/CLAIMS_LEDGER.csv`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/BLOCKER_MATRIX.csv`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/TEST_MAP.csv`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/*.md`
- `tests/test_external_team_submission_package.py`
- `docs/DPF_REQUIREMENTS_BASELINE.md`
- `docs/SRS_TRACEABILITY_MATRIX.csv`
- `docs/SRS_TRACEABILITY_MATRIX.json`
- `CodexFindings.md`
- `CortexFindings.md`

Required implementation:

1. Replace the split-brain delivery state with four explicit booleans:
   - `research_packet_delivered`;
   - `runtime_foundation_delivered`;
   - `accepted_physics_delivered`;
   - `validation_delivered`.
2. Update every Sprint 3 row in `BLOCKER_MATRIX.csv` so it no longer points to
   deleted `sprint_3/PENDING.md`.
3. Add S3.1 and S3.9 rows to `CLAIMS_LEDGER.csv` and `TEST_MAP.csv`.
4. Replace every actionable shorthand citation in WP-N5 with exact
   repo-relative paths and line ranges.
5. Update `tests/test_external_team_submission_package.py` so it rejects:
   - stale `runtime_implementation_delivered=false` after runtime commits;
   - live `sprint_3/PENDING.md` references;
   - `[KR: ...]` shorthand without a local path and line range;
   - bad module paths such as `closures.py` and `certificate.py`.
6. Replace wrong traceability paths:
   - `src/dpf/first_principles/closures.py` ->
     `src/dpf/first_principles/closure_packet.py`;
   - `src/dpf/first_principles/certificate.py` ->
     `src/dpf/first_principles/certificate_gate.py`.
7. Regenerate SRS traceability exports using the existing project tooling.
8. Update `CodexFindings.md` and `CortexFindings.md` with a dated Sprint 3R
   status entry.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py tests/test_srs_traceability_export.py -q -rx
python3 /Users/anthonyzamora/.codex/skills/srs-traceability/scripts/srs_trace_audit.py /Users/anthonyzamora/dpf-unified
```

Done criteria:

- All packet docs agree on Sprint 3 status.
- No live packet doc references deleted `sprint_3/PENDING.md`.
- No actionable shorthand citations remain.
- RTM and packet ledgers point only to real files.

## S3R.2 - Startup BVP Fail-Closed Acceptance

Audit finding covered: A1.

Allowed files:

- `src/dpf/first_principles/startup_bvp.py`
- `tests/test_first_principles_startup_bvp.py`
- `tests/test_cli_first_principles_3d.py` if CLI status output changes
- traceability docs only if requirement status text changes

Required implementation:

1. Bind `build_startup_bvp_packet()` acceptance to the typed `StartupPacket`.
2. Reject caller-declared accepted channels unless they correspond to typed
   channels with computed source-backed status, review metadata, source hashes,
   and required payload fields.
3. Until such computed channels exist, force:
   - `status != accepted_startup_bvp_packet`;
   - `whole_shot_startup_blocked=true`;
   - `can_support_whole_shot_acceptance=false`;
   - `can_support_first_principles_acceptance=false`.
4. Keep candidate CIV/Paschen/preionization telemetry available for engineering
   runs, but never as acceptance support.

Required negative tests:

- accepted-mode spoof payload with all channel names must remain blocked;
- reviewed evidence status without source hashes must remain blocked;
- candidate seeded layer must remain blocked for whole-shot startup;
- CLI output must report the typed startup packet blocker.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_startup_bvp.py tests/test_cli_first_principles_3d.py -q -rx
```

Done criteria:

- No caller-controlled payload can promote startup authority.
- Typed startup packet remains the single acceptance source.

## S3R.3 - Neutron Authority and NumPy 2 Beam-Target Fix

Audit findings covered: A2, A3.

Allowed files:

- `src/dpf/first_principles/neutron_authority.py`
- `src/dpf/diagnostics/beam_target.py`
- `tests/test_first_principles_neutron_authority.py`
- a narrow beam-target diagnostic test file if needed
- traceability docs only if status text changes

Required implementation:

1. Keep scalar yield as `candidate_comparator_only`.
2. Prevent `accepted_neutron_authority` from appearing unless mechanism
   channels and source-review packets are both complete.
3. Split status words:
   - scalar/target-only evidence -> `candidate_comparator_only` or
     `target_metadata_only_not_authority`;
   - text support -> `text_supported_reference_only_not_acceptance`;
   - complete mechanism packet -> only then can `accepted_neutron_authority`
     appear.
4. Fix `_trapezoid_integral()` for NumPy 2 using lazy fallback.

Required negative tests:

- same-scope scalar yield cannot create authority;
- accepted target metadata without runtime mechanism histories cannot create
  authority;
- missing stopping or detector response keeps total-yield authority blocked;
- `_trapezoid_integral()` works under the active NumPy 2 lane.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_neutron_authority.py -q -rx
```

Done criteria:

- Channel-level status cannot contradict top-level fail-closed status.
- Beam-target diagnostic paths do not crash before fail-closed gating.

## S3R.4 - PF-1000 Geometry Masks and Resolution Gates

Audit findings covered: A4, A5.

Allowed files:

- `src/dpf/fields/source_geometry.py`
- `tests/test_source_geometry_packet.py`
- `tests/test_first_principles_geometry.py`
- traceability docs only if status text changes

Required implementation:

1. Separate source-backed material masks from candidate projection masks.
2. If a mask needs a blocked dimension, do not emit it as source-backed.
3. For insulator and cathode cage fields:
   - use only exact local-source values;
   - otherwise emit explicit blockers and no accepted/source mask.
4. Add a manifest field that states each mask class status:
   `source_supported`, `candidate_projection_not_source_mask`, or `blocked`.
5. Preserve per-mask SHA-256 only with the status that generated the mask.
6. Extend under-resolution checks to every source-supported feature used by a
   mask.

Required negative tests:

- blocked insulator thickness cannot produce source-backed
  `insulator_material_faces`;
- conflict cathode cage radius cannot produce source-backed cathode shell
  geometry;
- under-resolved source-supported insulator feature fails closed;
- no fallback to a single generic `wall_material_faces` class when material
  masks are requested.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_source_geometry_packet.py tests/test_first_principles_geometry.py -q -rx
```

Done criteria:

- Every material mask has a status and source/blocker explanation.
- No blocked dimension is silently consumed as geometry.

## S3R.5 - `Sigma_p` Packet Schema and Power-Port Consumption

Audit findings covered: A6, A7.

Allowed files:

- `src/dpf/fields/source_geometry.py`
- `src/dpf/first_principles/power_port.py`
- `tests/test_source_geometry_packet.py`
- `tests/test_first_principles_power_port.py`
- traceability docs only if status text changes

Required implementation:

1. Extend `SigmaPSurfacePacket` with:
   - `sigma_p_face_set_sha256`;
   - `moving_classification_sha256`;
   - `omega_partition_sha256`;
   - `material_mask_sha256_by_class`;
   - source geometry hash retained on blocked packets;
   - face operand availability and raw operand arrays or explicit blockers for
     `B`, `E`, `J`, `v`, and `eta`;
   - sign convention and quadrature/centering metadata.
2. Implement dict-form packet reconstruction or fail closed with a named
   `serialized_sigma_p_packet_not_supported` blocker.
3. Add stationary/moving controls:
   - stationary faces must contribute zero when the future integral exists;
   - until integration exists, stationary/moving classification must be
     recorded and tested.
4. Add a missing-sign-convention negative control for terms II/IV/V/VI.
5. Keep terms II/IV/V/VI blocked until a reviewed surface integral exists.

Required negative tests:

- dict-form packet is not silently ignored;
- missing sign convention blocks Sigma_p terms;
- absent moving classification blocks Sigma_p terms;
- full operand presence still does not compute terms before the Sprint 4
  integrator exists.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_source_geometry_packet.py tests/test_first_principles_power_port.py -q -rx
```

Done criteria:

- `Sigma_p` packet is reviewer-grade even when blocked.
- Power-port ledger consumes or explicitly rejects serialized packets.

## S3R.6 - Closure Matrix Completeness

Audit finding covered: A8.

Allowed files:

- `src/dpf/first_principles/closure_packet.py`
- `tests/test_first_principles_closures.py`
- traceability docs only if status text changes

Required implementation:

1. Ensure every `REQUIRED_EFFECTS` entry appears in top-level `effects`.
2. Ensure every top-level effect appears in:
   - `closure_matrix_status_by_effect`;
   - `closure_effect_status`;
   - `missing_or_unaccepted_effects` when not accepted.
3. Preserve PlasmaPy as cross-check only.

Required negative tests:

- `REQUIRED_EFFECTS - effects.keys()` must be empty.
- `electron_inertia` and `stopping_collisions` remain blockers.
- PlasmaPy cannot promote or reject a local-source closure by itself.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_closures.py -q -rx
```

Done criteria:

- No closure can be silently omitted from the matrix.

## S3R.7 - Restart Ledger Completeness and Certificate Paths

Audit findings covered: A9, A12.

Allowed files:

- `src/dpf/first_principles/segmented_whole_shot.py`
- `src/dpf/first_principles/segmented_whole_shot_combine.py`
- `src/dpf/first_principles/certificate_gate.py`
- `tests/test_first_principles_segmented_whole_shot.py`
- `tests/test_first_principles_segmented_whole_shot_combine.py`
- `tests/test_first_principles_long_run_integrity.py`
- `tests/test_first_principles_certificate_negative_controls.py`
- traceability docs and packet ledgers

Required implementation:

1. Preserve these fields through merged ledgers:
   - `cumulative_field_energy_delta_J`;
   - `cumulative_pml_removed_energy_J`;
   - `cumulative_power_port_work_J`;
   - `cumulative_ionization_step_count`.
2. Define aggregation rules for each field in the merged ledger docstring or
   manifest schema.
3. Correct certificate path references to
   `src/dpf/first_principles/certificate_gate.py`.
4. Keep certificate acceptance fail-closed.

Required negative tests:

- three-segment merge preserves every extended field;
- missing field from any segment emits a named blocker or deterministic default
  according to the schema;
- certificate fixture cannot promote validation status.

Required tests:

```bash
.venv312/bin/python -m pytest \
  tests/test_first_principles_segmented_whole_shot.py \
  tests/test_first_principles_segmented_whole_shot_combine.py \
  tests/test_first_principles_long_run_integrity.py \
  tests/test_first_principles_certificate_negative_controls.py \
  -q -rx
```

Done criteria:

- Live manifests and merged manifests expose the same audited ledger channel
  family.
- Certificate docs and code paths agree.

## S3R.8 - Final Traceability, Audit, and Submission Packet

Audit findings covered: all.

Allowed files:

- all docs and tests touched above;
- no new physics implementation files unless required by S3R.1-S3R.7.

Required implementation:

1. Update `CHANGELOG.md` with every non-HEAD commit hash in the packet range.
2. Update `CLAIMS_LEDGER.csv`, `BLOCKER_MATRIX.csv`, and `TEST_MAP.csv`.
3. Regenerate `docs/SRS_TRACEABILITY_MATRIX.csv` and `.json`.
4. Add a final Sprint 3R submission file under:
   `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/SPRINT_3R_FINAL_SUBMISSION.md`
5. The final submission must include:
   - commit list;
   - mapping from A1-A12 to fixes/tests;
   - remaining blockers;
   - no validation claim;
   - periodic audit log path;
   - exact test command transcript.

Required final commands:

```bash
git diff --check
.venv312/bin/python -m ruff check src/ tests/
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py tests/test_srs_traceability_export.py -q -rx
.venv312/bin/python -m pytest \
  tests/test_source_geometry_packet.py \
  tests/test_first_principles_geometry.py \
  tests/test_first_principles_power_port.py \
  tests/test_first_principles_startup_bvp.py \
  tests/test_first_principles_closures.py \
  tests/test_first_principles_neutron_authority.py \
  tests/test_first_principles_segmented_whole_shot.py \
  tests/test_first_principles_segmented_whole_shot_combine.py \
  tests/test_first_principles_long_run_integrity.py \
  tests/test_first_principles_certificate_negative_controls.py \
  tests/test_cli_first_principles_3d.py \
  -q -rx
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Done criteria:

- All A1-A12 findings are closed or have typed fail-closed blockers.
- All required tests pass.
- Periodic audit passes from a clean tree.
- Handoff packet is internally consistent.
- No accepted physics or validation claim appears.

## Required Handoff Format For The Team

The team must return a single completion packet with this exact structure:

1. `Sprint 3R Final Submission`
2. `Commit List`
3. `Finding Closure Matrix`
   - one row per A1-A12;
   - status `fixed`, `blocked_fail_closed`, or `not_fixed`;
   - code paths;
   - test paths;
   - remaining blocker ID if not fixed.
4. `Runtime Behavior Summary`
   - what the simulator now does;
   - what remains blocked;
   - what cannot be claimed.
5. `Source Authority Summary`
   - local source files used;
   - missing local source fields;
   - candidate-only cross-checks.
6. `Test Evidence`
   - exact commands;
   - pass/fail counts;
   - periodic audit log path.
7. `No-Validation Statement`
   - explicit statement that Sprint 3R does not validate a full shot.

## How Codex Will Audit The Sprint 3R Submission

Codex will not rely on the team's prose. The audit will:

1. Start from `git status --short --branch` and reject dirty, unexplained
   worktrees.
2. Compare `git log 0b8fa97..HEAD` to packet `CHANGELOG.md`.
3. Run the focused tests listed above.
4. Run the full periodic audit.
5. Inspect every A1-A12 fix at line level.
6. Run negative probes for:
   - startup acceptance spoofing;
   - scalar neutron-yield authority spoofing;
   - NumPy 2 beam-target integration;
   - blocked insulator-mask synthesis;
   - dict-form `Sigma_p` packet ingestion;
   - omitted closure matrix effects;
   - merged-ledger extended-channel loss.
7. Verify that all source-backed claims cite exact local paths and line ranges.
8. Verify that `CodexFindings.md`, `CortexFindings.md`, SRS/RTM, packet ledgers,
   and final submission agree.
9. Reject any validation, engineering acceptance, or whole-shot completion
   claim unless same-scope source packets and certificate gates support it.

If any one of these checks fails, Sprint 3R is not complete.
