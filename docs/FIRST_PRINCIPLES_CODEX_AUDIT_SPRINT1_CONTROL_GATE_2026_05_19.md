# Codex Audit: Sprint 1 Control-Gate Submission

Date: 2026-05-19

Audited HEAD: `fa9088e`

Branch: `codex/corpus`, ahead of `origin/codex/corpus` by 10 commits.

Audited submission:
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/`

Prior audit:
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md`

Audit method: local command audit plus six read-only agent lanes:
provenance/CI, runtime, physics/source authority, SRS/package structure,
QA/test, and next-level work synthesis.

## Verdict

`accept_sprint_1_engineering_progress_request_changes_before_sprint_2`

The team successfully completed the narrow Sprint 1 control-gate hardening
scope. The core A-1 through A-7 control-plane fixes are real and pass local
verification.

This is not a completed three-sprint package. Sprints 2 and 3 are explicitly
deferred in the submission. No physics blocker is solved. No accepted or
validated first-principles claim may be made.

The next action is not to start broad physics work. First fix the remaining
submission/process defects listed below, then proceed to Sprint 2 only:
WP-N1B power-port burn-down and WP-N4B 12 us orchestration.

## Allowed Claim

`package-native 3-D first-principles engineering candidate with fail-closed source-truth, artifact, power-port, and segmented-run controls`

## Forbidden Claims

- Accepted first-principles simulation.
- Validated PF-1000/Akel prediction.
- Accepted power-port authority.
- Solved breakdown/startup/sheath-liftoff BVP.
- Reviewed PF-1000 geometry/material masks.
- Accepted EOS, radiation, ablation, restrike, anomalous resistance, 2T, or
  neutron closures.
- Completed 12 us source-sign whole-shot run.

## Local Verification

Commands run locally at current HEAD:

| Gate | Result |
| --- | --- |
| `verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_18` | pass: `exhausted=true`, `open_issue_count=0` |
| `verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_18` | pass: `strict_passed=true`, `total_modules=289`, `active_physics_unvetted_count=0` |
| `audit_first_principles_artifacts.py 'results/*.json'` | pass: `42 scanned`, `0 first-principles`, `31 skipped`, `11 exempt`, `0 failed` |
| `audit_first_principles_artifacts.py 'results/**/*.json'` | pass: `81 scanned`, `39 first-principles`, `50 exempt`, `0 failed` |
| Sprint 1 focused suite | `62 passed` |
| Broad first-principles/hybrid lane | `275 passed`, `9` PlasmaPy `CouplingWarning` warnings |
| Direct resume-ledger probe | pass: `resume_started_at_step=2`, `total_steps_completed=4`, `ledger_steps=4`, `covers_executed_horizon=true` |
| First-principles ruff audit slice | pass: `All checks passed` |
| `git diff --check` | pass |
| `git diff --exit-code` | pass |
| `git status --short --branch` | clean, ahead 10 |

Important: the CI-level broad lint command still fails. See finding RC-2.

## Accepted Sprint 1 Engineering Progress

### A-1 Manifest Provenance

Accepted as engineering progress.

`source_packet_hashes` is now required for complete provenance, empty mappings
are treated as missing, and artifact-linter C7 re-derives manifest completeness
from raw fields instead of trusting `manifest.provenance_complete`.

Evidence:

- `src/dpf/first_principles/manifest.py`
- `scripts/audit_first_principles_artifacts.py`
- `tests/test_first_principles_manifest.py`
- `tests/test_first_principles_artifact_linter.py`

### A-2 Stale Active Artifacts

Accepted as engineering progress with scope limit.

The stale active first-principles artifacts were quarantined, not regenerated.
That is allowed for Sprint 1 hygiene. It means the active root currently has no
authority-scope first-principles result artifact proving a current positive
runtime path.

Next physics submissions must generate fresh current-HEAD artifacts or keep all
old artifacts quarantined.

### A-3 Read-Only Gates

Accepted as engineering progress.

The source-truth and module-vetting scripts expose read-only `--check` modes,
pin the date in CI, and no longer churn timestamps during verification.

### A-4 Archive Policy CI Coverage

Accepted as engineering progress with additional next gate required.

CI now runs both active-root and recursive artifact scans. Archive entries are
reported `EXEMPT` with reasons.

Missing next-level control: active artifacts are not yet required to have
generation commits matching current HEAD. See RC-5.

### A-5 First-Principles Ruff Slice

Accepted only for the scoped first-principles slice.

The audit-named first-principles/fields slice passes. The repository-wide CI
lint job still fails. See RC-2.

### A-6 Resume Ledger Continuity

Accepted as engineering progress.

The prior bug is fixed for checkpoints produced by the current segmented
runner. Resumed runs now rehydrate cumulative ledger sidecars and report full
executed-horizon coverage.

Remaining runtime work:

- production 12 us orchestrator;
- artifact combiner;
- hash-chain across prior and resumed run directories;
- explicit fail-closed behavior for legacy checkpoints without ledger sidecars;
- CLI resume/output tests;
- PML removed-energy ledger or explicit inactive-zero telemetry.

### A-7 SRS/RTM Status

Accepted as engineering progress.

`DPF-PHYS-020` and `DPF-PHYS-023` moved to `partial`, not `implemented`, with
acceptance blockers preserved. RTM exports were regenerated and Doorstop remains
explicitly deferred.

## Request Changes Before Sprint 2

### RC-1: This Is Not A Complete Three-Sprint Submission

The package states that it delivers Sprint 1 only. Sprint 2 and Sprint 3 are
placeholders:

- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_2/PENDING.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/PENDING.md`

This is acceptable because the prior audit gated Sprint 2 on a clean Sprint 1.
It is not acceptable to call this a completed three-sprint package.

Required before claiming three-sprint completion:

- all six Sprint 2 proposal/status docs;
- all ten Sprint 3 proposal/matrix docs;
- Sprint 2/3 rows in `CLAIMS_LEDGER.csv`;
- real `KnowledgeReference/` rows in `SOURCE_PACKET_INDEX.csv`;
- updated `BLOCKER_MATRIX.csv` statuses.

### RC-2: CI Broad Ruff Is Still Red

CI runs:

```bash
ruff check src/ tests/
```

Local result at current HEAD:

```text
Found 69 errors.
36 I001, 7 F401, 4 N812, 4 UP042, 3 E402, 3 F841, 2 B905,
2 UP017, 1 F821, 1 SIM102, 1 SIM114, 1 SIM300, 1 UP012,
1 UP034, 1 UP035, 1 UP041
```

This is a release/merge blocker unless the workflow is intentionally narrowed
or the failures are fixed/allowlisted. The submission's "ruff clean" claim is
true only for the scoped first-principles audit slice.

Required fix:

- either fix the broad `src/ tests/` ruff failures; or
- change CI to the intended scoped lint command and document why the broader
  debt is out of scope; or
- add a dated, explicit allowlist with owners and removal criteria.

Do not leave CI implicitly red.

### RC-3: `BLOCKER_MATRIX.csv` Is Malformed CSV

CSV parse audit:

```text
BLOCKER_MATRIX.csv rows 18 header 6 widths [6, 7, 9]
bad [(5, 9), (8, 7)]
```

Rows with comma-containing evidence fields are not quoted. This breaks the
machine-readable review contract.

Required fix:

- quote comma-containing fields;
- add a package schema test that verifies all rows have the header width;
- include `BLOCKER_MATRIX.csv` in the read-only package audit gate.

### RC-4: Verification Transcript Does Not Match Current HEAD

`AUDIT_COMMANDS.md` reports verification at `fe038f7`, while current audited
HEAD is `fa9088e`. The final commit changed only the transcript, but the package
still needs a post-packet current-HEAD verification line.

Required fix:

- append a final verification block at current HEAD;
- include `git rev-parse HEAD`, `git status --short`, and `git diff --exit-code`;
- make the package schema gate reject stale transcript HEADs.

### RC-5: Add Artifact Commit-Match Gate C8

The artifact linter verifies provenance presence, but not that active artifacts
were generated from current HEAD.

The current tree passes because all authority-scope first-principles artifacts
are either absent from active root or quarantined. That is fine for Sprint 1,
but Sprint 2 will need fresh active artifacts.

Required next gate:

- C8 active artifact `artifact_generation_commit == HEAD`;
- nested `manifest.git_commit == HEAD`;
- nested `manifest.artifact_generation_commit == HEAD`;
- `dirty_worktree is False`;
- tests for stale active commit, nested/top-level mismatch, and dirty artifacts.

### RC-6: Add Positive Current-Schema Artifact Fixture

The linter currently passes mostly through `SKIP` and `EXEMPT`. Add a small
positive current-schema first-principles artifact fixture so CI proves `PASS`,
not only "nothing active failed."

### RC-7: Linter Required-Field Tuple Drift Risk

The linter carries a local copy of required provenance fields rather than
importing `REQUIRED_PROVENANCE_FIELDS`. This is documented in
`UNKNOWN_AND_INFERENCE_LOG.md` as I-6.

Required fix:

- add a unit test proving the linter C7 required-field tuple equals
  `dpf.first_principles.manifest.REQUIRED_PROVENANCE_FIELDS`; or
- import the tuple directly and handle CLI/package import constraints.

## Physics Status

No physics blocker was closed in this submission.

| Area | Status | Audit conclusion |
| --- | --- | --- |
| WP-N1B power-port proposals | blocked | Sprint 2 proposal files are absent. |
| Auluck eq. 5/6 | blocked | Exact electrode/moving-boundary integrand remains OCR/source blocked. |
| Power residual tolerance | blocked | No local KR source defines an accepted tolerance. |
| Time-centering | implemented candidate | Step-consistent metadata only; no accepted quadrature. |
| Startup BVP | implemented candidate | Fail-closed packet exists; accepted channel payloads absent. |
| PF-1000 geometry masks | implemented candidate | Rod/hollow projection and hashes exist; reviewed material masks absent. |
| Closures | implemented candidate | Some source-backed pieces exist; EOS/radiation/ablation/restrike/anomalous resistance remain blocked. |
| Neutron authority | implemented candidate | Scalar/candidate yield only; mechanism-separated detector/UQ authority absent. |

All external sources remain non-authority until ingested into
`KnowledgeReference/`.

## Agent-Lane Synthesis

| Lane | Result |
| --- | --- |
| Provenance/CI | A-1 through A-3 fixed; archive scan operational; add C8 current-HEAD artifact gate. |
| Runtime | Resume ledger bug fixed; 12 us orchestrator, artifact combiner, PML removed-energy ledger still missing. |
| Physics/source authority | Sprint 2/3 physics proposals absent; all physics acceptance blockers remain. |
| SRS/package | Sprint 1 package skeleton good; full three-sprint package incomplete; transcript HEAD stale. |
| QA/test | Focused and first-principles tests pass; CI broad ruff remains red. |
| Next-level planning | Proceed to Sprint 2 only after RC items are fixed; keep startup/geometry before closures/neutrons. |

## Next-Level Work Order

### Sprint 1.1: Pre-Sprint-2 Hygiene Fixes

This small cleanup must happen before Sprint 2 is accepted for audit:

1. Fix or scope CI broad ruff.
2. Fix `BLOCKER_MATRIX.csv` quoting and add package CSV schema tests.
3. Add current-HEAD transcript verification.
4. Add artifact C8 current-HEAD/clean-worktree gate.
5. Add positive current-schema first-principles artifact fixture.
6. Add required-field tuple drift test or remove duplication.

### Sprint 2: WP-N1B And WP-N4B Only

Deliver these required files:

- `WP_N1B_AULUCK_EQ_5_6_SOURCE_STATUS.md`
- `WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md`
- `WP_N1B_RESIDUAL_TOLERANCE_SOURCE_STATUS.md`
- `WP_N1B_TIME_CENTERING_PROPOSAL.md`
- `WP_N4B_12US_ORCHESTRATION_PROPOSAL.md`
- `WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`

Acceptance criteria:

- every physics assertion has `KnowledgeReference/` path, line range, and hash;
- Auluck eq. 5/6 is either transcribed with symbol/unit/sign map or explicitly
  blocked;
- electrode/interface work is independent, not a residual closure;
- residual tolerance is source-backed or review-packet-defined;
- time-centering is one consistent discrete quadrature across all terms;
- 12 us plan has explicit steps, segments, checkpoints, wall-clock estimate,
  partial-run labels, artifact combiner, and production ladder;
- any generated active artifacts pass C7 and C8 from clean current HEAD;
- no accepted/validated language is introduced.

### Sprint 3: Startup, Geometry, Closures, Neutrons, Certificate

Do not start Sprint 3 until Sprint 2 is clean.

Order:

1. WP-N2 startup BVP.
2. WP-N3 PF-1000 geometry/material masks.
3. WP-N5 closure registry and PlasmaPy/regime-warning policy.
4. WP-N6 neutron mechanism authority.
5. WP-N7 comparator/UQ/certificate.
6. Numerical acceptance: convergence, limiter-zero proof, backend parity,
   restart reproducibility.

## Final Audit State

Sprint 1 control-gate work is accepted as engineering progress.

The repository is not merge-ready while the broad CI lint job is red.

The simulator is still an engineering candidate, not a full first-principles
DPF simulator and not a validated PF-1000/Akel tool.

