# Changelog — Submission 1 + Sprint 1.1 + Sprint 2

Base commit (audited state): `76480b0`
Branch: `codex/corpus`

## Commits

### Sprint 1 — Control Gate Hardening (`76480b0`..`fe038f7`)

| Commit | Subject |
| --- | --- |
| `3dc4c11` | docs: add Codex WP-N1/WP-N4 first-principles audit |
| `4424785` | fix(first-principles): require non-empty source_packet_hashes provenance (A-1) |
| `80654b9` | fix(scripts): deterministic output and read-only check mode for first-principles gates (A-3) |
| `b626ad9` | style(first-principles): clear broad ruff lint slice (A-5) |
| `55e3f94` | fix(first-principles): rehydrate cumulative ledger on segmented resume (A-6) |
| `1abe15a` | chore(results): quarantine stale pre-HEAD audit artifacts (A-2) |
| `bf22c33` | ci(first-principles): read-only audit gates, recursive archive scan, dirty-tree assertion (A-3/A-4) |
| `fe038f7` | docs(srs): mark DPF-PHYS-020/023 partial, regenerate RTM, close Sprint 0 with debt (A-7) |

The submission packet commit for Sprint 1 (`fa9088e` — docs: add Sprint 1
control-gate submission packet) is a documentation-only wrapper: changed paths
are all under `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/`.

### Sprint 1 Wrappers + Sprint 1.1 + Sprint 2 — Hygiene and Proposals (`fe038f7`..`c52bed3`)

| Commit | Subject |
| --- | --- |
| `f07d4ba` | docs: add Sprint 1 control-gate submission packet *(documentation-only wrapper)* |
| `fa9088e` | docs: pin --date in submission audit-command transcript *(documentation-only)* |
| `06cfd64` | docs: add Codex Sprint 1 control-gate audit |
| `9df8d3b` | style: clear repository-wide ruff debt so the CI lint job passes (RC-2) |
| `49e80ee` | docs(kr): ingest verified Auluck 2021 power-balance equations; refresh source-truth index |
| `bd840f4` | docs: add Sprint 2 WP-N1B and WP-N4B proposal set |
| `0585eec` | docs: add tracked copy of verified Auluck equations to Sprint 2 packet |
| `0bef78c` | fix(first-principles): add artifact linter check C8 and package schema tests (RC-3/5/6/7) |
| `c52bed3` | docs: finalize Sprint 1.1 — current-HEAD transcript block and RC statuses (RC-4) *(documentation-only wrapper)* |

### Sprint 2 Audit and Periodic Runner (`c52bed3`..`6af4454`)

| Commit | Subject |
| --- | --- |
| `6af4454` | docs: add sprint 2 audit and periodic runner |

### Sprint 2 Implementation — WP-N1B / WP-N4B (`6af4454`..HEAD)

| Commit | Subject |
| --- | --- |
| `4b080eb` | feat(first-principles): implement WP-N1B six-term Auluck power-port ledger (fail-closed) |
| `4c8dac1` | feat(first-principles): implement WP-N4B cross-restart ledger merge and artifact combiner |
| `93620ba` | docs: synchronize Sprint 2 packet hygiene and add consistency tests (F2/F3/F4) |

The final commit of this packet revision is a documentation-only changelog
wrapper: it updates this `CHANGELOG.md` and `tests/test_external_team_submission_package.py`
and, by construction, cannot list its own hash. The package-consistency test
`test_changelog_covers_all_commits_since_base` exempts HEAD for exactly this
reason (it checks `76480b0..HEAD~1`).

## Changed paths

### Sprint 1

`3dc4c11` — added: `docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md`.

`4424785` — modified: `src/dpf/first_principles/manifest.py`,
`scripts/audit_first_principles_artifacts.py`,
`tests/test_first_principles_manifest.py`,
`tests/test_first_principles_artifact_linter.py`.

`80654b9` — modified: `scripts/verify_first_principles_source_truth_exhaustion.py`,
`scripts/verify_first_principles_module_source_vetting.py`,
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}`,
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`; added:
`tests/test_first_principles_verification_check_mode.py`.

`b626ad9` — modified: `src/dpf/fields/__init__.py`,
`src/dpf/fields/maxwell_3d.py`, `src/dpf/fields/particle_boundaries.py`,
`src/dpf/first_principles/gv_waveforms.py`,
`tests/test_first_principles_mhd.py`.

`55e3f94` — modified: `src/dpf/first_principles/segmented_whole_shot.py`,
`tests/test_first_principles_segmented_whole_shot.py`.

`1abe15a` — renamed (quarantine): `results/audit_first_principles_3d_smoke.json`,
`results/audit_experimental_whole_shot_smoke.json`,
`results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json` →
`results/archive_stale_pre_ssr_codex_a2_2026_05_18/`; added:
`results/archive_stale_pre_ssr_codex_a2_2026_05_18/QUARANTINE_NOTICE.md`.

`bf22c33` — modified: `.github/workflows/ci.yml`.

`fe038f7` — modified: `docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.csv`, `docs/SRS_TRACEABILITY_MATRIX.json`,
`docs/FIRST_PRINCIPLES_EXPERIMENTAL_SIMULATOR_SPRINT_PLAN_2026_05_18.md`,
`tests/test_srs_traceability_export.py`.

`f07d4ba` — added all files under
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/`
*(documentation-only wrapper commit for Sprint 1 packet)*.

`fa9088e` — modified:
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/AUDIT_COMMANDS.md`
(pinned `--date 2026_05_18` on the two verification-gate commands so the
read-only check resolves to the committed baseline when re-run on any future
date) *(documentation-only)*.

`06cfd64` — added:
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT1_CONTROL_GATE_2026_05_19.md`.

### Sprint 1.1 — Hygiene

`9df8d3b` — style: repository-wide ruff cleanup (RC-2); 52 files changed across
`src/dpf/` and `tests/`; also regenerated
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`.

`49e80ee` — docs(kr): ingested verified Auluck 2021 equations; modified:
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}`,
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.{json,md}`.

`bd840f4` — docs: Sprint 2 proposals; modified:
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/BLOCKER_MATRIX.csv`,
`CLAIMS_LEDGER.csv`, `README.md`, `SOURCE_PACKET_INDEX.csv`;
removed `sprint_2/PENDING.md` (placeholder superseded by proposal files);
added six `sprint_2/WP_N1B_*` and `sprint_2/WP_N4B_*` proposal docs.

`0585eec` — docs: tracked copy of verified Auluck extract; modified:
`SOURCE_PACKET_INDEX.csv`; added:
`sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md`.

`0bef78c` — fix(first-principles): artifact linter C8 + package schema tests
(RC-3/5/6/7); modified: `scripts/audit_first_principles_artifacts.py`,
`tests/test_first_principles_artifact_linter.py`; added:
`tests/test_external_team_submission_package.py`.

`c52bed3` — docs: finalized Sprint 1.1 (RC-4); modified:
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/AUDIT_COMMANDS.md`,
`BLOCKER_MATRIX.csv` *(documentation-only wrapper commit; RC-2..RC-7 rows closed)*.

### Sprint 2 Audit and Periodic Runner

`6af4454` — docs: added
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_PACKET_2026_05_19.md`,
`scripts/run_codex_periodic_audit.py`.

### Sprint 2 Implementation — WP-N1B / WP-N4B

`4b080eb` — feat: WP-N1B six-term Auluck power-port ledger; modified
`src/dpf/first_principles/power_port.py`,
`tests/test_first_principles_power_port.py`, and
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`
(module-source-vetting regenerated after the first-principles module edit).

`4c8dac1` — feat: WP-N4B cross-restart ledger merge and artifact combiner;
added `src/dpf/first_principles/segmented_whole_shot_combine.py` and
`tests/test_first_principles_segmented_whole_shot_combine.py`.

`93620ba` — docs: Sprint 2 packet hygiene (F2/F3/F4); modified
`THREE_SPRINT_FINAL_SUMMARY.md`, `CHANGELOG.md`, `PATCH_SCOPE.md`,
`UNKNOWN_AND_INFERENCE_LOG.md` under the packet, and
`tests/test_external_team_submission_package.py`.

`092871b` — docs: changelog catch-up for WP-N1B/WP-N4B and HEAD-exempt
changelog test; modified `CHANGELOG.md` and
`tests/test_external_team_submission_package.py`.

`ada801c` — docs: add Sprint 2 implementation audit and next directions
(`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_IMPLEMENTATION_2026_05_19.md`);
documentation-only.

### Sprint 2.1 Implementation Pass — F1/F3/F4/F5

`65c477f` — feat: compute Auluck power-port terms I and III, drop electrode_work
wording (F1/F3/F5); modified `src/dpf/first_principles/power_port.py`,
`src/dpf/fields/hybrid_simulator.py`, `src/dpf/fields/hybrid_stepper.py`,
`tests/test_first_principles_power_port.py`, and
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`.

`21d4e07` — fix: harden the WP-N4B combiner with cross-restart ledger
invariants (F4); modified
`src/dpf/first_principles/segmented_whole_shot_combine.py`,
`tests/test_first_principles_segmented_whole_shot_combine.py`, and
`sprint_2/WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`.

The final commit of this pass — the F2 SRS/RTM traceability sync — also updates
this `CHANGELOG.md`; by construction it cannot list its own hash and is
HEAD-exempt from the `test_changelog_covers_all_commits_since_base` check
(`76480b0..HEAD~1`), the same constraint Codex audit F4 accepted for
`AUDIT_COMMANDS.md`.

## Deleted / quarantined artifacts

Three audit smoke artifacts were moved (not deleted) to
`results/archive_stale_pre_ssr_codex_a2_2026_05_18/` and marked quarantined. They
embed `artifact_generation_commit 466a0a5` and `dirty_worktree true`. See
`sprint_1/ARTIFACT_REGENERATION_OR_QUARANTINE_PLAN.md`.

## Generated artifacts

No new runtime result artifacts were generated. There are no active
first-principles result artifacts at HEAD `fe038f7`; the audit step "active
artifact generation commit equals HEAD" is satisfied because the active set is
empty.

The four dated source-truth docs
(`FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}`,
`FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`) were regenerated
once into deterministic form (A-3); the SRS RTM exports
(`SRS_TRACEABILITY_MATRIX.{csv,json}`) were regenerated from the baseline (A-7).
