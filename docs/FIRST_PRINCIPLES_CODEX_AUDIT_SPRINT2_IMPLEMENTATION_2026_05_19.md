# Codex Audit - Sprint 2 Implementation Pass

Date: 2026-05-19
Branch: `codex/corpus`
Audited HEAD: `092871b`
Verdict: `accept_engineering_progress_request_changes_before_wp_n1b_wp_n4b_clean`

## Scope

This audit reviews the implementation commits after
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_PACKET_2026_05_19.md`:

- `4b080eb` - WP-N1B six-term Auluck power-port ledger, fail-closed.
- `4c8dac1` - WP-N4B cross-restart ledger merge and artifact combiner.
- `93620ba` / `092871b` - packet hygiene and changelog catch-up.

The goal is still the same: a full-fidelity, first-principles, true-physics 3D
DPF shot simulator. This audit checks engineering truthfulness and next
blockers only. It does not validate the simulator.

## Gate Results

Full periodic audit runner:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Latest passing log:

`/private/tmp/dpf-unified-audit-logs/20260519T152601Z/summary.md`

Summary:

| Gate | Result |
| --- | --- |
| clean worktree | PASS |
| source-truth exhaustion | PASS |
| module-source vetting | PASS (`total_modules=290`) |
| artifact linter active root | PASS |
| artifact linter recursive | PASS |
| `ruff check src/ tests/` | PASS |
| focused package/control pytest | PASS |
| broad first-principles/hybrid pytest | PASS |

Direct test checks during this audit:

- Focused package/control set: `78 passed`.
- New WP-N1B/WP-N4B implementation tests:
  `tests/test_first_principles_power_port.py`
  `tests/test_first_principles_segmented_whole_shot_combine.py`:
  `27 passed`.
- Broad first-principles/hybrid set: `298 passed`, 9 PlasmaPy
  `CouplingWarning` warnings.

One earlier periodic-run attempt exited with signal 15 during pytest and no
assertion traceback. The same gates passed when rerun directly and then through
the full periodic runner. Treat that attempt as transient process termination,
not a confirmed code regression, but keep watching the loop logs.

## Accepted Engineering Progress

### WP-N1B

`src/dpf/first_principles/power_port.py` now exposes an Auluck eq. (6)
six-term ledger structure:

- I: stored magnetic energy rate.
- II: motional magnetic `Sigma_p` surface integral.
- III: stored electric energy rate.
- IV: motional electric `Sigma_p` surface integral.
- V: resistive `Sigma_p` surface integral.
- VI: anomalous/poloidal `Sigma_p` surface integral.

The code correctly removes the old `electrode_interface_work_J` closure term
from the authoritative WP-N1B ledger. It records Auluck eq. (1)'s load-bearing
leading minus and refuses to emit a residual while any Auluck term is missing.
This is the correct fail-closed behavior.

Important boundary: this is not yet a computed power-port closure. All six
terms currently fail closed because the runtime does not expose the required
magnetic/electric stored-energy split or the reviewed `Sigma_p` moving-boundary
face data.

### WP-N4B

`src/dpf/first_principles/segmented_whole_shot_combine.py` now provides:

- `merge_cumulative_ledgers()`.
- `combine_whole_run_artifacts()`.
- gap, overlap, missing-manifest, and empty-input fail-closed tests.
- a small two-restart positive test against an uninterrupted short run.

This is real engineering progress toward the 12 us orchestration path. It does
not make a completed 12 us source-sign run claim.

## Findings

### F1 - WP-N1B is a fail-closed contract, not a computed six-term power port

Severity: High

The new Auluck ledger is structurally correct and fail-closed, but it computes
zero of the six terms today. Terms I/III are blocked on magnetic/electric
stored-energy split; terms II/IV/V/VI are blocked on reviewed `Sigma_p`
geometry plus face-centered `v`, `B`, `J`, and `eta`.

Required action:

1. Add runtime diagnostics for separate magnetic and electric energy over
   Auluck `Omega`.
2. Implement reviewed `Sigma_p` face inventory from WP-N3 geometry, including
   moving vs stationary boundary classification.
3. Expose `v`, `B`, `J`, `eta`, outward `dS`, and centering metadata on
   `Sigma_p` faces.
4. Compute each Auluck term independently. Do not compute any term by residual
   closure.
5. Only then compute `residual_J = I*V - (I + II + III + IV + V + VI)`.

Sprint status should be worded as:
`WP-N1B six-term fail-closed ledger contract implemented; computed six-term
power-port closure still blocked`.

### F2 - Requirements and packet traceability are stale after implementation

Severity: High

`docs/DPF_REQUIREMENTS_BASELINE.md`, `docs/SRS_TRACEABILITY_MATRIX.csv`,
`docs/SRS_TRACEABILITY_MATRIX.json`, and
`docs/external_team_submissions/.../RTM_DELTA.md` still describe DPF-PHYS-020 as
a five-term ledger with `electrode/interface work`, and DPF-PHYS-023 as missing
the cross-restart ledger merge/artifact combiner.

The packet ledgers are also stale:

- `BLOCKER_MATRIX.csv` still marks WP-N1B/WP-N4B as `proposal_delivered`.
- `CLAIMS_LEDGER.csv` still says no code changed for the Sprint 2 claims.
- `TEST_MAP.csv` still reports the old 275 broad-test count and does not include
  the new WP-N1B/WP-N4B tests.

Required action:

1. Rewrite DPF-PHYS-020 around Auluck eq. (6), no electrode-work term, and
   current blocker set: stored EM split, `Sigma_p`, residual tolerance,
   centering, reviewed geometry, same-scope review.
2. Rewrite DPF-PHYS-023 to mention the candidate combiner now exists, but 12 us
   completion, production-grid wall-clock, and long-run restart evidence remain
   blocked.
3. Regenerate `docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.
4. Update `RTM_DELTA.md`, `BLOCKER_MATRIX.csv`, `CLAIMS_LEDGER.csv`, and
   `TEST_MAP.csv` to reflect implementation-candidate status and the new tests.
5. Add a package-consistency test that fails on any surviving "five-term
   Auluck ledger" or "electrode/interface work" authority wording in SRS/RTM
   artifacts.

### F3 - Legacy `electrode_work` authority wording remains in runtime packets

Severity: Medium

The authoritative WP-N1B ledger correctly removes
`electrode_interface_work_J`, but the surrounding power-port packet still
contains legacy authority fields:

- `REQUIRED_POWER_PORT_CHANNELS` includes `electrode_work`.
- `ACCEPTANCE_BLOCKING_CHANNELS` includes `electrode_work_partition`.
- The top-level `acceptance_gate` string mentions
  `centering_electrode_work_residual_tolerance_hashes_and_review_pass`.
- `negative_test_policy` still includes
  `electrode_work_omission_required`.

This creates ambiguity for downstream artifact consumers because the source
verdict is now explicit: Auluck has no electrode-contact-work balance term.

Required action:

1. Replace `electrode_work` acceptance language with Auluck eq. (6) term
   completeness: `term_i` through `term_vi`.
2. Keep source-interface and wall/electrode labels only as geometry/domain
   labels, not as an independent work term.
3. Add tests proving no authoritative WP-N1B required channel, acceptance gate,
   or negative-test policy requires `electrode_work`.

### F4 - WP-N4B combiner needs stronger manifest invariants before long runs

Severity: Medium

The combiner positive test is useful, but `merge_cumulative_ledgers()` assumes
the terminal restart manifest already carries cumulative ledgers rehydrated
from the sidecar. That is true for the tested runner path, but the combiner
does not independently prove that every input manifest satisfies the invariant.

Required action:

1. Add a negative test where a later restart manifest is contiguous by step
   index but its cumulative ledger counters are not cumulative. The combiner
   must fail closed.
2. Add a negative test where the first restart starts after step 0. Either fail
   closed for whole-run mode or emit an explicit `suffix_run_not_whole_run`
   status.
3. Check monotonicity of cumulative counters across restarts.
4. Reconcile the proposal text, which says "sum additive counters", with the
   implementation, which correctly takes the terminal rehydrated cumulative
   ledger to avoid double-counting sidecar prefixes.

### F5 - Source references should name the verified Auluck extract at top level

Severity: Low

The WP-N1B subledger cites `AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md`,
but the top-level `POWER_PORT_SOURCE_REFS` still points primarily at the OCR
extract and older support references. The artifact-level `source_references`
should include the verified Auluck extract explicitly so reviewers do not chase
the garbled equation source first.

Required action:

Add the verified extract to `POWER_PORT_SOURCE_REFS` with the role
`auluck_eq1_eq5_eq6_verified_power_balance`, and add a test that the emitted
power-port packet references it.

## Directions For The Team

### Sprint 2.1 - Traceability and wording cleanup

Objective: make the repo's control plane match the new implementation and the
Auluck source verdict.

Deliverables:

- Updated `docs/DPF_REQUIREMENTS_BASELINE.md`.
- Regenerated `docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.
- Updated packet `RTM_DELTA.md`, `BLOCKER_MATRIX.csv`, `CLAIMS_LEDGER.csv`,
  `TEST_MAP.csv`, `README.md`, and `THREE_SPRINT_FINAL_SUMMARY.md`.
- Tests that fail on stale five-term/electrode-work authority wording.

Acceptance:

```bash
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py tests/test_srs_traceability_export.py -q -rx
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

### Sprint 2.2 - Complete WP-N4B combiner hardening

Objective: make combined whole-run artifacts fail closed on malformed
restart-ledger invariants before any long run uses them.

Deliverables:

- Combiner input-invariant checks.
- Negative tests for non-cumulative terminal ledgers, nonzero first restart,
  non-monotonic cumulative counters, and mismatched planned horizon.
- Updated WP-N4B proposal text explaining terminal-ledger use vs additive
  summing.

Acceptance:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_segmented_whole_shot_combine.py -q -rx
```

### Sprint 2.3 - Start real WP-N1B runtime terms

Objective: move from a fail-closed Auluck ledger contract to at least one
independently computed Auluck term.

Order:

1. Add split magnetic/electric stored energy over `Omega` and compute terms I
   and III.
2. Implement reviewed `Sigma_p` face inventory from WP-N3 geometry.
3. Add `Sigma_p` face field sampling for `v`, `B`, `J`, `eta`, and outward
   `dS`.
4. Implement terms II/IV/V/VI independently.
5. Only after all six terms are independent, emit a genuine residual.

Acceptance for this sprint is not power-port acceptance. It is narrower:
the implementation must prove each newly computed term is independent and fail
closed when its source field or domain is missing.

### Sprint 3 Unlock Rule

Do not start broad Sprint 3 physics implementation until the traceability
cleanup is done. After that, prioritize WP-N3 geometry because WP-N1B cannot
compute `Sigma_p` terms without it. The correct next physics order is:

1. WP-N3 reviewed PF-1000 geometry/material masks and `Sigma_p`.
2. WP-N1B computed Auluck terms.
3. WP-N2 startup BVP.
4. WP-N5 closure registry.
5. WP-N6 neutron authority.
6. Numerical acceptance and WP-N7 comparator/UQ/certificate.

Every item remains non-accepting until its source-backed acceptance criteria
exist and pass.
