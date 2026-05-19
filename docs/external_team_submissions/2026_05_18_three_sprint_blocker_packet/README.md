# Codex Audit Response — Three-Sprint Blocker Packet

- Dates: 2026-05-18 (Sprint 1), 2026-05-19 (Sprint 2 proposals)
- Repository: DPF-Unified
- Branch: `codex/corpus`
- Responds to: `docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md`
  and the Sprint 1 audit `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT1_CONTROL_GATE_2026_05_19.md`
- See `CHANGELOG.md` for the commit list and `AUDIT_COMMANDS.md` for the
  verification transcript with the current-HEAD block.

## Allowed claim for this submission

> package-native 3-D first-principles engineering candidate with fail-closed
> source-truth, artifact, power-port, and segmented-run controls

## Forbidden claims — NOT made anywhere in this packet

Accepted first-principles simulation; validated PF-1000/Akel prediction;
accepted power-port authority; solved breakdown/startup/sheath-liftoff BVP;
reviewed PF-1000 geometry/material masks; accepted EOS/radiation/ablation/
restrike/anomalous-resistance/2T/neutron closures; completed 12 us source-sign
whole-shot run.

## Scope of this submission

This packet delivers **Sprint 1 (Control Gate Hardening)** and the **Sprint 2
proposal set** (WP-N1B power port, WP-N4B 12 us orchestration). The Codex Sprint 1
audit accepted Sprint 1 engineering progress and requested seven Sprint 1.1
hygiene fixes (RC-1…RC-7) before Sprint 2 acceptance. Sprint 3
(WP-N2/N3/N5/N6/N7 physics, numerical acceptance) research packets have been
delivered — see `sprint_3/SPRINT_3_STATUS_LEDGER.md`. Runtime implementation
and acceptance remain not delivered.

No artifact, manifest, requirement, equation extract, or document in this packet
promotes anything to accepted or validated status.
`can_support_first_principles_acceptance` is `false` everywhere.

## Sprint 1 outcome — all seven control-gate blockers closed

| Blocker | Result | Commit |
| --- | --- | --- |
| A-1 manifest provenance | Closed — `source_packet_hashes` required; linter C7 re-derives independently | `4424785` |
| A-2 stale active artifacts | Closed — 3 artifacts quarantined; no active artifact predates HEAD | `1abe15a` |
| A-3 read-only CI gates | Closed — `--check` mode + deterministic output | `80654b9` |
| A-4 archive policy CI | Closed — recursive `results/**/*.json` scan in CI | `bf22c33` |
| A-5 broad ruff failures | Closed for the first-principles slice — see RC-2 for the repo-wide CI job | `b626ad9` |
| A-6 resume ledger continuity | Closed — cumulative ledger rehydrated on resume | `55e3f94` |
| A-7 stale SRS/RTM | Closed — `DPF-PHYS-020/023` → `partial`; RTM regenerated | `fe038f7` |

## Sprint 1 audit result and Sprint 1.1 hygiene

The Codex Sprint 1 audit
(`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT1_CONTROL_GATE_2026_05_19.md`) verdict:
`accept_sprint_1_engineering_progress_request_changes_before_sprint_2`. A-1…A-7
were each accepted as engineering progress. Seven RC items — Sprint 1.1 hygiene —
gate Sprint 2 acceptance and are carried as rows `RC-1`…`RC-7` in
`BLOCKER_MATRIX.csv`. RC-1 (deliver the Sprint 2 docs) and RC-3 (CSV quoting) are
addressed in this packet revision; RC-2/RC-4/RC-5/RC-6/RC-7 are the Sprint 1.1
fix set.

## Sprint 2 outcome — WP-N1B and WP-N4B implementation delivered as fail-closed runtime code

- **WP-N1B power port (commits `4b080eb`, `65c477f`).** The six-term Auluck eq. (6) ledger
  contract is now implemented as fail-closed runtime code in
  `src/dpf/first_principles/power_port.py`. The prior "electrode/interface work"
  term is absent — it was a category error; Auluck excludes the electrode
  interface from the integration domain Omega. All six terms (I: stored magnetic
  rate; II: motional magnetic Sigma_p; III: stored electric rate; IV: motional
  electric Sigma_p; V: resistive Sigma_p; VI: anomalous/poloidal Sigma_p)
  remain fail-closed for acceptance. Terms I and III are now independently
  computed from runtime split telemetry (`stored_magnetic_energy_delta_J` and
  `stored_electric_energy_delta_J`); terms II, IV, V, VI remain blocked on the
  reviewed Sigma_p moving-boundary geometry (requires WP-N3). No invented
  values or closure substitutions are used. 27 new tests pass
  (`tests/test_first_principles_power_port.py`), with follow-up coverage for
  the split-telemetry path.
  Power-port acceptance remains blocked.
- **WP-N4B 12 us orchestration (commit `4c8dac1` plus Sprint 2.2 follow-up).** A candidate cross-restart
  ledger-merge and artifact-combiner is now implemented in
  `src/dpf/first_principles/segmented_whole_shot_combine.py`, providing
  `merge_cumulative_ledgers()` and `combine_whole_run_artifacts()` with
  fail-closed gap/overlap/missing-manifest/empty-input/malformed-step checks.
  The combiner treats `total_steps_completed` as the cumulative terminal step
  and includes synthetic plus live three-restart coverage. 12 focused tests pass
  (`tests/test_first_principles_segmented_whole_shot_combine.py`). The 12 us
  compute-wall, production-grid wall-clock, and long-run restart evidence remain
  blocked on WP-N3 grid size.

Sprint 2 delivers implementation as engineering candidates. No physics blocker
is closed; no power-port or whole-shot acceptance is claimed.

## Verification headlines

All audit Submission-1 commands pass (full transcript: `AUDIT_COMMANDS.md`).

- Broad first-principles/hybrid suite: **298 passed**, 9 warnings.
- Focused Submission-1 suite: 62 passed.
- Artifact linter: active root 0 failed; recursive `results/**/*.json` 0 failed.
- Read-only verification gates: exit 0, zero worktree writes.
- `git status --short` and `git diff --exit-code`: clean.

## Packet contents

Top-level: `README.md`, `CLAIMS_LEDGER.csv`, `BLOCKER_MATRIX.csv`,
`SOURCE_PACKET_INDEX.csv`, `EXTERNAL_LEADS_NOT_AUTHORITY.md`,
`UNKNOWN_AND_INFERENCE_LOG.md`, `AUDIT_COMMANDS.md`, `CHANGELOG.md`,
`THREE_SPRINT_FINAL_SUMMARY.md`, `PATCH_SCOPE.md`, `TEST_MAP.csv`,
`ARTIFACT_HASHES.csv`, `RTM_DELTA.md`.

`sprint_1/`: `CONTROL_GATE_PROPOSAL.md`, `RESUME_LEDGER_CONTINUITY_PROPOSAL.md`,
`ARTIFACT_REGENERATION_OR_QUARANTINE_PLAN.md`, `SRS_RTM_BASELINE_DECISION.md`.

`sprint_2/`: `WP_N1B_AULUCK_EQ_5_6_SOURCE_STATUS.md`,
`WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md`,
`WP_N1B_RESIDUAL_TOLERANCE_SOURCE_STATUS.md`,
`WP_N1B_TIME_CENTERING_PROPOSAL.md`, `WP_N4B_12US_ORCHESTRATION_PROPOSAL.md`,
`WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`.

`sprint_3/SPRINT_3_STATUS_LEDGER.md` — Sprint 3 status ledger (supersedes the deleted `sprint_3/PENDING.md`; research_packets_delivered=true; runtime_implementation_delivered=false; first_principles_acceptance=false). See `sprint_3/` for the seven delivered research packets.
