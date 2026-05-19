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

### S3.1 Packet Hygiene — stale-language fixes (documentation-only, in-progress)

S3.1 scoped changes to the packet documents (no SRS/RTM edits, no src/ edits):

- `sprint_3/PENDING.md` deleted; superseded by `sprint_3/SPRINT_3_STATUS_LEDGER.md`
  (research_packets_delivered=true; runtime_implementation_delivered=false;
  first_principles_acceptance=false).
- `sprint_3/WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md`: stale "Sprint 2.2 open",
  "WP-N2 not delivered", "WP-N5 closure registry not delivered" language replaced
  with `research_packet_delivered`, `runtime_packet_not_delivered`,
  `accepted_packet_not_delivered` distinctions.
- `sprint_3/WP_N4_PERFORMANCE_AND_RUN_PLAN.md`: "unknown" grid size replaced
  with `blocked_by_missing_local_source`.
- `sprint_3/WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md`: all `[KR: same file ...]`
  shorthand citations expanded to exact local paths with line ranges.
- `README.md`, `THREE_SPRINT_FINAL_SUMMARY.md`: `sprint_3/PENDING.md` nav
  references updated to `sprint_3/SPRINT_3_STATUS_LEDGER.md`; Sprint 3 status
  updated to reflect research packets delivered.
- `tests/test_external_team_submission_package.py`: new S3.1 consistency tests
  added (stale-state rejection, shorthand-citation rejection).

### Sprint 3 Implementation And Traceability (`0b8fa97`..HEAD)

Sprint 3 packages S3.1–S3.9 are committed: `100d87d` (S3.1 packet hygiene),
`0e91f08` (S3.2/S3.3 PF-1000 geometry packet + Sigma_p plumbing), `06744fd`
(S3.4 typed startup BVP packet), `7dd1994` (S3.5 closure registry + PlasmaPy
regime gate), `d1dc17c` (S3.6 mechanism-separated neutron authority packet),
`6660eb9` (S3.7/S3.8 numerical acceptance gates + certificate scaffold),
`f7bb9f8` (module-source-vetting regeneration), `e9b3c20` (S3.9 SRS/RTM and
packet-ledger traceability). The Sprint 3 final submission is
`sprint_3/SPRINT_3_FINAL_SUBMISSION.md`.

`269d7d1` is the Sprint 3 final-submission wrapper commit: it adds
`SPRINT_3_FINAL_SUBMISSION.md` and updates this `CHANGELOG.md`. The current
package-consistency test `test_changelog_covers_all_commits_since_base` exempts
only HEAD because the changelog is committed in the same commit that changes it
(it checks `76480b0..HEAD~1`).

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

`07fe76a` — docs: synchronize SRS/RTM and packet traceability after
WP-N1B/WP-N4B implementation (F2); modified
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`, and packet traceability files.

`9e1407d` — docs: add Codex Sprint 2 follow-up implementation audit and next
directions; added
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`.

`e984176` — docs: update Sprint 2 audit changelog after the follow-up audit;
modified this `CHANGELOG.md` so the packet consistency gate covered the Sprint
2 traceability and audit commits.

`4e3987b` — docs: add parallel work directions to the Sprint 2 follow-up audit;
modified
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md` with
allowed non-conflicting Sprint 3 research/spec lanes and audit acceptance
rules.

`6d015c4` — docs: update changelog for parallel directions; modified this
`CHANGELOG.md` so the packet consistency gate covered the parallel-directions
documentation commit.

`df71d99` — docs: add Sprint 3 WP-N2/N3/N4/N5/N6/N7 research and spec packets;
added source-backed packet drafts under
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/`.

`173e10d` — fix(first-principles): close Sprint 2.2 combiner and traceability
gaps; modified `src/dpf/first_principles/segmented_whole_shot_combine.py`,
`tests/test_first_principles_segmented_whole_shot_combine.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`, `docs/SRS_TRACEABILITY_MATRIX.{csv,json}`,
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_2/WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`,
and `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`.

`d44601f` — docs: audit Sprint 3 research packets; added
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_PACKET_2026_05_19.md` and updated
this `CHANGELOG.md` with the Sprint 2.2 closure commit.

`0b8fa97` — docs: add sprint 3 completion handoff; added
`docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md`.

### Sprint 3 Implementation — S3.1–S3.8

| Commit | Subject |
| --- | --- |
| `100d87d` | docs: S3.1 packet hygiene — status ledger, citation normalization, consistency tests |
| `0e91f08` | feat(first-principles): S3.2/S3.3 source-tagged PF-1000 geometry packet and Sigma_p plumbing |
| `06744fd` | feat(first-principles): S3.4 typed startup BVP packet (fail-closed) |
| `7dd1994` | feat(first-principles): S3.5 closure registry and PlasmaPy regime gate |
| `d1dc17c` | feat(first-principles): S3.6 mechanism-separated neutron authority packet |
| `6660eb9` | feat(first-principles): S3.7/S3.8 numerical acceptance gates and certificate scaffold |
| `f7bb9f8` | chore(first-principles): regenerate module-source-vetting after Sprint 3 implementation |

### S3.1 Implementation — packet hygiene (`100d87d`)

`100d87d` — docs: S3.1 packet hygiene (status ledger, citation normalization, consistency
tests); modified
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/SPRINT_3_STATUS_LEDGER.md`
and related sprint_3 packet docs; added or updated S3.1 consistency tests in
`tests/test_external_team_submission_package.py`.

### S3.2/S3.3 Implementation — geometry packet and Sigma_p plumbing (`0e91f08`)

`0e91f08` — feat: S3.2/S3.3 source-tagged PF-1000 geometry packet and Sigma_p plumbing;
added/modified `src/dpf/fields/source_geometry.py` (`PF1000GeometryPacket`,
`PF1000GeometryField`, `PF1000GeometryConflict`, `PF1000MaskManifest`,
`SigmaPSurfacePacket`; geometry: 5 source-supported, 4 conflict, 10 blocked fields;
Sigma_p packet fail-closed; Auluck terms II/IV/V/VI stay fail-closed;
I/III computed); modified `tests/test_source_geometry_packet.py`,
`tests/test_first_principles_geometry.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.4 Implementation — startup BVP packet (`06744fd`)

`06744fd` — feat: S3.4 typed startup BVP packet (fail-closed); added
`src/dpf/first_principles/startup_bvp.py` (13-channel `StartupBVPPacket`,
0 source-supported, all candidate/blocked; startup authority blocked);
modified `tests/test_first_principles_startup_bvp.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.5 Implementation — closure registry and PlasmaPy regime gate (`7dd1994`)

`7dd1994` — feat: S3.5 closure registry and PlasmaPy regime gate; added/modified
`src/dpf/first_principles/closures.py` (12 closures + 2 sub-closures;
1 active_source_backed_candidate, rest blocked/not-simulated;
`PhysicsClosurePacket` embedding); modified `tests/test_first_principles_closures.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.6 Implementation — neutron authority packet (`d1dc17c`)

`d1dc17c` — feat: S3.6 mechanism-separated neutron authority packet; added/modified
`src/dpf/first_principles/neutron_authority.py` (`NeutronAuthorityPacket`,
10 channels; 5 candidate, 4-5 blocked; scalar yield comparator-only);
modified `tests/test_first_principles_neutron_authority.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.7/S3.8 Implementation — numerical acceptance gates and certificate scaffold (`6660eb9`)

`6660eb9` — feat: S3.7/S3.8 numerical acceptance gates and certificate scaffold;
modified `src/dpf/first_principles/segmented_whole_shot.py` (extended cumulative
ledgers), `src/dpf/first_principles/certificate.py` (certificate scaffold with
all channels missing — no accepted certificate); added/modified
`tests/test_first_principles_long_run_integrity.py`,
`tests/test_first_principles_certificate_negative_controls.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.9 Traceability — module-source-vetting regeneration (`f7bb9f8`)

`f7bb9f8` — chore: regenerate module-source-vetting after Sprint 3 implementation;
modified `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`.

The final commit of any pass that updates this `CHANGELOG.md` is structurally
HEAD-exempt from the `test_changelog_covers_all_commits_since_base` check
(`76480b0..HEAD~1`), because a commit cannot list its own hash before it exists.
Every non-HEAD commit in the range must be listed here.

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
