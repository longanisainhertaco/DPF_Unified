# Codex Audit - Sprint 2 Packet Pass

Date: 2026-05-19
Branch: `codex/corpus`
Audited HEAD: `c52bed3`
Verdict: `accept_sprint_1_1_control_gate_fixes_request_packet_hygiene_before_sprint_2_clean`

## Scope

This audit reviews the post-Sprint-1.1 work after
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT1_CONTROL_GATE_2026_05_19.md`.
It checks whether the prior request-change items were closed, whether the
Sprint 2 packet makes only source-supported claims, and whether the package can
be audited repeatedly while the external team keeps working.

## Gate Results

Commands run from repository root:

| Gate | Result |
| --- | --- |
| `verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_18` | PASS (`exhausted=true`, `open_issue_count=0`) |
| `verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_18` | PASS (`strict_passed=true`, `total_modules=289`) |
| `audit_first_principles_artifacts.py 'results/*.json'` | PASS (`42 scanned`, `0 failed`) |
| `audit_first_principles_artifacts.py 'results/**/*.json'` | PASS (`81 scanned`, `0 failed`) |
| `ruff check src/ tests/` | PASS |
| Focused package/control tests | PASS (`75 passed`) |
| Broad first-principles/hybrid tests | PASS (`283 passed`, `9 warnings`) |
| CSV package shape check | PASS (`CLAIMS_LEDGER`, `BLOCKER_MATRIX`, `SOURCE_PACKET_INDEX`, `TEST_MAP`, `ARTIFACT_HASHES`) |
| `git diff --exit-code` | PASS |

The PlasmaPy `CouplingWarning` warnings remain physics-regime warnings, not test
failures. They still belong to WP-N5 collision/closure policy.

## Accepted Progress

1. RC-2 is closed: repository-wide `ruff check src/ tests/` passes.
2. RC-3 is closed structurally: the submission CSV files now parse with stable
   row widths, and `tests/test_external_team_submission_package.py` exists.
3. RC-5 is closed as an engineering gate: artifact linter C8 now compares the
   active artifact commit fields against live HEAD and rejects dirty-worktree
   artifacts.
4. RC-6 is closed: the artifact-linter suite includes a live-HEAD positive
   first-principles fixture, so the linter proves real PASS, not only SKIP or
   EXEMPT.
5. RC-7 is closed: the C7 required-provenance tuple has a drift test against
   `dpf.first_principles.manifest.REQUIRED_PROVENANCE_FIELDS`.
6. The Auluck equation extract is useful and source-grounded: the team
   correctly identified the missing minus sign in eq. (1), the six-term eq. (6)
   decomposition, and the fact that "electrode/interface work" is not an
   Auluck balance term.

## Findings

### F1 - Sprint 2 is proposal-only, not an implementation close

Severity: High

`sprint_2/WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md` and
`sprint_2/WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md` are accurate
about their own status: no six-term Auluck ledger has been implemented, no
`Sigma_p` moving-boundary face set exists, no residual tolerance is attached,
no accepted time-centering is attached, and the cross-restart merge/combiner is
not built. Therefore WP-N1B and WP-N4B remain open engineering/physics blockers.

Required action: next implementation sprint must produce code and tests for the
six-term Auluck ledger and the cross-restart combiner, while keeping
`can_support_first_principles_acceptance=false`.

### F2 - `THREE_SPRINT_FINAL_SUMMARY.md` contradicts the current packet

Severity: Medium

`THREE_SPRINT_FINAL_SUMMARY.md:5-8` says only Sprint 1 is complete and Sprints 2
and 3 are deferred. Lines 45-49 say Sprint 2 is pending and point to
`sprint_2/PENDING.md`, but `sprint_2/PENDING.md` has been removed and Sprint 2
proposal files now exist. This contradicts `README.md:24-31` and
`README.md:60-81`.

Required action: update the final summary to match the current packet:
Sprint 1.1 hygiene closed, Sprint 2 proposals delivered, Sprint 2 implementation
not yet clean, Sprint 3 pending.

### F3 - Packet metadata still describes Submission 1 only

Severity: Medium

`CHANGELOG.md:1-18`, `PATCH_SCOPE.md:1-93`, and
`UNKNOWN_AND_INFERENCE_LOG.md:1-82` remain Submission-1/Sprint-1 documents. They
do not cover commits `9df8d3b`, `49e80ee`, `bd840f4`, `0585eec`, `0bef78c`, or
`c52bed3`, nor do they record the Sprint 2 inferences such as the Auluck
OCR/verified-extract decision, the electrode-work retraction, the residual
tolerance no-source verdict, or the 12 us compute-wall measurement.

Required action: update these top-level control documents before calling the
packet clean. The README is current enough to understand the submission, but
the packet is not internally synchronized.

### F4 - Verification transcript is acceptable but structurally fragile

Severity: Low

`AUDIT_COMMANDS.md` includes a current-HEAD block for `0bef78c` and explains why
the containing documentation commit cannot name itself. The audited HEAD is
`c52bed3`, whose changes are limited to `AUDIT_COMMANDS.md` and
`BLOCKER_MATRIX.csv`. This is acceptable for this pass, but the next packet
should include an explicit "final documentation-only wrapper commit" note in
the changelog to avoid repeating the stale-transcript ambiguity.

Required action: keep the explanation, and add the final wrapper commit to the
changelog with changed paths.

## Next Instructions For The Team

1. Fix packet synchronization before more research: update
   `THREE_SPRINT_FINAL_SUMMARY.md`, `CHANGELOG.md`, `PATCH_SCOPE.md`, and
   `UNKNOWN_AND_INFERENCE_LOG.md`.
2. Add a package-consistency test that fails when a doc references a deleted
   `PENDING.md`, when README says Sprint 2 proposals exist but the final summary
   says Sprint 2 is pending, or when the changelog omits commits since the last
   audit.
3. Implement WP-N1B as code, not prose:
   - replace the current non-independent power-port residual path with Auluck
     terms I-VI;
   - compute terms II, IV, V, and VI from independent `Sigma_p` surface
     integrals;
   - fail closed if `Sigma_p` is missing, if any term is closure-derived, or if
     the eq. (1) sign convention is not recorded in the manifest;
   - leave acceptance false until residual tolerance, time-centering, and
     reviewed geometry are supplied.
4. Implement WP-N4B as code:
   - add cross-process restart ledger merge;
   - add whole-run artifact combiner;
   - prove contiguous tiling, no gaps, no overlaps, and merged-ledger equality
     to an uninterrupted short run.
5. Keep Sprint 3 research pending until WP-N1B implementation produces a
   reviewable runtime artifact.

## Periodic Audit

Added `scripts/run_codex_periodic_audit.py`.

One cycle:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py
```

Loop every 30 minutes in the foreground:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --loop --interval-minutes 30
```

Bounded loop, useful overnight:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --loop --cycles 16 --interval-minutes 30
```

The runner writes logs outside the git worktree by default:
`/private/tmp/dpf-unified-audit-logs/`. That avoids the audit dirtying the same
repository state it is checking. It records `summary.json`, `summary.md`, and
per-gate stdout/stderr for each cycle.
