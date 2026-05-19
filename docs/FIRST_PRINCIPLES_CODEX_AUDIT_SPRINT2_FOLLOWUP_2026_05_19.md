# Codex Audit - Sprint 2 Follow-up Implementation Pass

Date: 2026-05-19
Branch: `codex/corpus`
Audited HEAD: `07fe76a`
Verdict: `request_changes_before_sprint_3_execution`

## Scope

This audit reviews the engineering team's follow-up commits after
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_IMPLEMENTATION_2026_05_19.md`.
The reviewed commits are:

- `65c477f` - compute Auluck power-port terms I and III; drop
  `electrode_work` wording from the power-port channels.
- `21d4e07` - harden WP-N4B combiner with cross-restart ledger invariants.
- `07fe76a` - synchronize SRS/RTM and packet traceability after WP-N1B/WP-N4B
  implementation.

The audit standard is unchanged: first-principles DPF claims must be grounded
in the local source-of-truth corpus and must remain fail-closed when a runtime
channel, geometry packet, source equation, tolerance, or comparator is missing.
This audit does not validate the simulator.

## Gate Results

Full periodic audit runner:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Passing log:

`/private/tmp/dpf-unified-audit-logs/20260519T173535Z/summary.md`

Summary:

| Gate | Result |
| --- | --- |
| clean worktree | PASS |
| `git diff --check` | PASS |
| source-truth exhaustion | PASS |
| module-source vetting | PASS |
| active artifact linter | PASS |
| recursive artifact linter | PASS |
| `ruff check src/ tests/` | PASS |
| focused pytest | PASS |
| broad first-principles/hybrid pytest | PASS |

Additional manual audit check:

```text
Three contiguous cumulative restart manifests:
  r0: start 0, total_steps_completed 2
  r1: start 2, total_steps_completed 4
  r2: start 4, total_steps_completed 6

merge_cumulative_ledgers(...) result:
  LedgerMergeError: step overlap detected before restart 2:
  expected start at step 6, but restart begins at step 4
```

That manual check exposes a WP-N4B combiner correctness bug that the current
two-restart positive test does not cover.

## Accepted Engineering Progress

### WP-N1B Power Port

The follow-up work materially improves WP-N1B.

- The runtime now emits separate stored magnetic and stored electric energy
  deltas over Auluck `Omega`.
- `build_wp_n1_auluck_power_port_ledger()` computes term I from
  `stored_magnetic_energy_delta_J`.
- `build_wp_n1_auluck_power_port_ledger()` computes term III from
  `stored_electric_energy_delta_J`.
- Terms I and III are marked `computed_independently`, not
  closure-derived.
- Terms II, IV, V, and VI correctly remain fail-closed on missing reviewed
  `Sigma_p` moving-boundary geometry and face quantities.
- The authoritative runtime channels no longer require a fake
  `electrode_work` term.

This is the correct direction. It does not complete the power-port closure
because the four `Sigma_p` surface terms remain blocked.

### WP-N4B Restart Combiner

The combiner now checks several necessary invariants:

- first restart must begin at step 0;
- cumulative counters must be non-decreasing across restarts;
- missing or malformed manifests fail closed;
- gap and overlap checks exist;
- terminal rehydrated ledgers are used instead of summing sidecar prefixes.

These are useful changes, but the combiner is not yet clean because it rejects
valid chains with more than two restarts.

## Findings

### F1 - WP-N4B combiner rejects valid three-restart whole-run chains

Severity: High

`merge_cumulative_ledgers()` uses inconsistent semantics for
`total_steps_completed`.

The manifest writer records `total_steps_completed` as the cumulative terminal
step after a run invocation. `combine_whole_run_artifacts()` also treats it that
way when checking `curr_start == prev_completed`.

However, `merge_cumulative_ledgers()` updates the next expected start as:

```python
expected_next_step = actual_start + steps_completed
```

For cumulative manifests this is wrong after the second restart. A valid chain
`0->2`, `2->4`, `4->6` becomes:

- after r0: expected start 2;
- after r1: expected start 6;
- r2 starts at 4, so the combiner falsely reports an overlap.

Evidence:

- [src/dpf/first_principles/segmented_whole_shot_combine.py](/Users/anthonyzamora/dpf-unified/src/dpf/first_principles/segmented_whole_shot_combine.py:120)
- [src/dpf/first_principles/segmented_whole_shot_combine.py](/Users/anthonyzamora/dpf-unified/src/dpf/first_principles/segmented_whole_shot_combine.py:137)
- [src/dpf/first_principles/segmented_whole_shot_combine.py](/Users/anthonyzamora/dpf-unified/src/dpf/first_principles/segmented_whole_shot_combine.py:211)
- [tests/test_first_principles_segmented_whole_shot_combine.py](/Users/anthonyzamora/dpf-unified/tests/test_first_principles_segmented_whole_shot_combine.py:133)

Required correction:

1. Treat `total_steps_completed` as the cumulative terminal step everywhere.
2. In `merge_cumulative_ledgers()`, after each manifest set:
   `expected_next_step = int(manifest["total_steps_completed"])`.
3. Add a positive three-restart test using synthetic manifests:
   `0->2`, `2->4`, `4->6`.
4. Add a live runner three-restart test if runtime cost stays acceptable.
5. Add a malformed manifest negative test where `total_steps_completed <
   resume_started_at_step`; fail closed with an attributable error.
6. Re-run:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_segmented_whole_shot_combine.py -q -rx
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Acceptance state: WP-N4B remains `implemented_candidate_with_changes_required`.

### F2 - SRS/RTM traceability is stale after the stored-EM split implementation

Severity: High

The follow-up code now exposes separate magnetic and electric stored-energy
deltas, and terms I/III are independently computed when those channels exist.
The SRS and packet still say the opposite.

Stale examples:

- `DPF-PHYS-020` says "magnetic/electric stored-EM split not exposed by runtime
  (blocks terms I and III)".
- `RTM_DELTA.md` says all six terms return `None` because the runtime does not
  expose the split.
- `BLOCKER_MATRIX.csv` says terms I/III remain blocked on the split.

Evidence:

- [docs/DPF_REQUIREMENTS_BASELINE.md](/Users/anthonyzamora/dpf-unified/docs/DPF_REQUIREMENTS_BASELINE.md:99)
- [docs/SRS_TRACEABILITY_MATRIX.csv](/Users/anthonyzamora/dpf-unified/docs/SRS_TRACEABILITY_MATRIX.csv:63)
- [docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/RTM_DELTA.md](/Users/anthonyzamora/dpf-unified/docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/RTM_DELTA.md:20)
- [docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/BLOCKER_MATRIX.csv](/Users/anthonyzamora/dpf-unified/docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/BLOCKER_MATRIX.csv:16)

Required correction:

1. Rewrite DPF-PHYS-020 to state that terms I and III are now computed
   independently from runtime split-energy telemetry.
2. Keep DPF-PHYS-020 `partial`, because terms II/IV/V/VI, residual tolerance,
   accepted centering, reviewed PF-1000 masks, and same-scope review remain
   open.
3. Regenerate `docs/SRS_TRACEABILITY_MATRIX.csv` and
   `docs/SRS_TRACEABILITY_MATRIX.json`.
4. Update `RTM_DELTA.md`, `BLOCKER_MATRIX.csv`, `CLAIMS_LEDGER.csv`,
   `TEST_MAP.csv`, `README.md`, `THREE_SPRINT_FINAL_SUMMARY.md`, and
   `CHANGELOG.md` so the packet has one consistent state.
5. Add a package consistency test that fails when SRS/RTM artifacts say
   `stored-EM split not exposed by runtime` after the runtime emits
   `stored_magnetic_energy_delta_J` and `stored_electric_energy_delta_J`.

Acceptance state: the traceability update is not accepted until a generated
RTM diff and package-consistency test are present.

### F3 - WP-N1B source-faithful runtime is ahead of its docstrings

Severity: Low

The implementation body correctly computes terms I/III from split telemetry,
but stale comments still say terms I/III fail closed because the runtime only
emits combined stored EM energy. The helper docstring also still calls the
ledger "WP-N1 five-term".

Evidence:

- [src/dpf/first_principles/power_port.py](/Users/anthonyzamora/dpf-unified/src/dpf/first_principles/power_port.py:324)
- [src/dpf/first_principles/power_port.py](/Users/anthonyzamora/dpf-unified/src/dpf/first_principles/power_port.py:467)

Required correction:

1. Update the stale docstrings.
2. Do not change runtime behavior while doing this cleanup.
3. Keep the compatibility alias comments if needed, but mark them explicitly
   historical.

Acceptance state: low-risk cleanup, but required before the packet is called
clean.

## Next Directions For The Engineering Team

### Stop-the-line Sprint 2.2 - correctness and traceability cleanup

Objective: make the completed Sprint 2 implementation internally consistent
before any Sprint 3 physics work proceeds.

Required deliverables:

1. Fix `merge_cumulative_ledgers()` for N restart chains.
2. Add three-restart positive coverage and malformed-manifest negative
   coverage.
3. Update stale WP-N1B docstrings.
4. Update DPF-PHYS-020 and all generated RTM artifacts.
5. Update external-team packet ledgers and changelog.
6. Add consistency tests that catch stale stored-split blocker wording.
7. Run the full periodic audit.

Exit criteria:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_segmented_whole_shot_combine.py tests/test_first_principles_power_port.py tests/test_external_team_submission_package.py tests/test_srs_traceability_export.py -q -rx
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Do not claim WP-N4B clean until a three-restart chain passes.
Do not claim WP-N1B traceability clean until the SRS/RTM no longer says the
stored split is missing.

### Sprint 3 - reviewed PF-1000 geometry and `Sigma_p`

Objective: unblock Auluck terms II, IV, V, and VI without inventing closure
terms.

Required research and implementation outputs:

1. PF-1000/Akel geometry packet with reviewed dimensions for:
   cathode rods, anode, hollow bore, alumina insulator, backplate/source
   interface, chamber wall, open/PML boundary, and plasma domain.
2. Deterministic masks with separate hashes for each material/domain class.
3. `Omega` and `Sigma_p` derivation from those masks.
4. Moving vs stationary boundary classification. Auluck stationary boundaries
   do not contribute to the `dS.v` terms.
5. Face-centered sampling for `B`, `E`, `J`, `v`, `eta`, outward `dS`, face
   area, and centering metadata on `Sigma_p`.
6. Under-resolution gate for rods, bore, insulator surfaces, sheath thickness,
   and moving-boundary features.
7. Tests:
   - mask disjointness and exhaustiveness;
   - `Omega` excludes the source interface;
   - stationary boundaries contribute zero to motional terms;
   - moving-boundary faces are non-empty in a controlled synthetic case;
   - all mask hashes and source references appear in the manifest.

Exit criteria:

`build_wp_n1_auluck_power_port_ledger()` may still leave terms II/IV/V/VI
blocked until numerical sampling is connected, but the runtime must expose a
reviewed `Sigma_p` packet with enough fields to compute them next.

### Sprint 4 - full Auluck six-term power-port computation

Objective: compute all six Auluck eq. (6) terms independently from runtime
fields.

Required implementation outputs:

1. Compute term II from the motional magnetic `Sigma_p` surface integral.
2. Compute term IV from the motional electric `Sigma_p` surface integral.
3. Compute term V from the resistive `eta J x B` `Sigma_p` surface integral.
4. Compute term VI from the anomalous/poloidal `B(B.v)` `Sigma_p` surface
   integral.
5. Preserve term I and term III independent stored-energy deltas.
6. Compute `residual_J` only when all six terms are present and the sign
   convention is recorded.
7. Keep `can_support_first_principles_acceptance=false` unless a source-backed
   residual tolerance, accepted time-centering policy, reviewed geometry, and
   same-scope packet are attached.

Required negative tests:

1. Missing `Sigma_p` faces blocks II/IV/V/VI.
2. Missing `v` on `Sigma_p` blocks II/IV/VI.
3. Missing `eta` on `Sigma_p` blocks V.
4. Any term derived by closure is rejected.
5. Wrong or missing Auluck eq. (1) sign convention blocks residual.
6. Residual remains `None` when only five of six terms are present.

### Sprint 5 - whole-shot runtime scaling and restart evidence

Objective: turn WP-N4B from a small-horizon orchestrator into a credible
experimental whole-shot execution path.

Required outputs:

1. N-restart combiner proven for at least three live restart invocations.
2. Production-grid planning tied to the Sprint 3 reviewed grid size.
3. Wall-clock forecast with measured per-step cost on the selected backend.
4. Restart reproducibility evidence at staged increasing horizons.
5. PML removed-energy ledger, if PML/open boundaries are active.
6. Artifact manifest that reports whether the requested `12 us` horizon was
   completed, partially completed, or compute-wall blocked.

No 12 us claim is allowed until the run actually reaches 12 us and the artifact
states `horizon_complete=true`.

### Sprint 6 - startup, closures, neutron authority, and comparator packets

Objective: continue after the power port and geometry are coherent.

Required work lanes:

1. WP-N2 startup BVP: breakdown, preionization, flashover, secondary emission,
   surface plasma, initial E/B/J, density/species, ionization, Te/Ti, sheath
   surface, and handoff interval.
2. WP-N5 closure packets: EOS, radiation, ablation/impurity, anomalous
   resistance, restrike, collision/stopping, electron inertia, and beam-target
   coupling.
3. WP-N6 neutron authority: thermonuclear vs beam-target split, spectra,
   anisotropy, detector response, and UQ.
4. WP-N7 comparator/certificate: same-scope PF-1000/Akel waveforms, field and
   density histories, configuration hashes, environment hashes, and explicit
   engineering review gates.

Each lane must include source packet, equations, symbol map, units, validity
range, implementation path, tests, blocker state, and acceptance impact.

## How Codex Will Audit The Next Submission

The next audit will fail the submission if any of these are true:

1. The worktree is dirty before the audit starts.
2. The periodic audit runner fails.
3. A valid three-restart chain is rejected by `merge_cumulative_ledgers()`.
4. A stale SRS/RTM artifact says terms I/III are still blocked on missing split
   telemetry.
5. Any `electrode_work` wording reappears as an Auluck balance term.
6. Any Auluck term is computed by closure from the residual.
7. Any source-truth claim cites outside material instead of the local
   `KnowledgeReference`/tracked verified extract packet.
8. Any artifact promotes validation, acceptance, or first-principles authority
   while `can_support_first_principles_acceptance=false`.

Expected next audit command:

```bash
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Expected targeted commands:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_segmented_whole_shot_combine.py -q -rx
.venv312/bin/python -m pytest tests/test_first_principles_power_port.py -q -rx
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py tests/test_srs_traceability_export.py -q -rx
```

The next packet must include the command output summary, the commit SHAs, and
the exact files changed for each finding.
