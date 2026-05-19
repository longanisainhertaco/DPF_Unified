# Patch Scope — Submission 1 + Sprint 1.1 + Sprint 2

Why each changed file was necessary. Grouped by audit finding.

## A-1 — manifest provenance (`4424785`)

- `src/dpf/first_principles/manifest.py` — `source_packet_hashes` was absent from
  `REQUIRED_PROVENANCE_FIELDS`, and `missing_provenance_fields()` tested
  emptiness only for `str | tuple`. Both were necessary changes: without them a
  manifest with an empty `source_packet_hashes` dict reported
  `provenance_complete=true`. Added the field to the required tuple; generalised
  the emptiness test to any empty sized container.
- `scripts/audit_first_principles_artifacts.py` — linter check C7 trusted the
  self-reported `manifest.provenance_complete` boolean. Necessary: a stale or
  hand-edited manifest can carry `provenance_complete: true` while omitting
  `source_packet_hashes`. C7 now re-derives completeness from the raw manifest
  fields.
- `tests/test_first_principles_manifest.py` — added the negative test the audit
  required (empty `source_packet_hashes` ⇒ incomplete) plus a positive control.
- `tests/test_first_principles_artifact_linter.py` — added a C7 test for a lying
  manifest and one for a genuinely complete manifest; the shared fixture needed
  full provenance so it would not trip the strengthened C7.

## A-3 — read-only gates (`80654b9`)

- `scripts/verify_first_principles_source_truth_exhaustion.py` and
  `scripts/verify_first_principles_module_source_vetting.py` — both wrote dated
  docs unconditionally and embedded a sub-second wall-clock `generated_at_utc`.
  Necessary: in CI this dirties the worktree on every run. Added a `--check`
  read-only mode and replaced the wall-clock field with the date slug so output
  is deterministic.
- `tests/test_first_principles_verification_check_mode.py` (new) — proves
  `--check` exits 0 when in sync, fails on drift, and writes nothing.
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}` and
  `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}` — refreshed
  once into the deterministic form so `--check` has a matching baseline. This is
  the single intentional write.

## A-5 — ruff slice (`b626ad9`)

Each file carried exactly one class of ruff finding from the audit-named slice:

- `src/dpf/fields/__init__.py` — I001 unsorted import block.
- `src/dpf/fields/maxwell_3d.py` — UP037 quoted type annotations (×2).
- `src/dpf/fields/particle_boundaries.py` — UP037 quoted type annotation.
- `src/dpf/first_principles/gv_waveforms.py` — SIM117 nested `with`, and B905
  `zip()` without an explicit `strict=` (set `strict=False`: the two iterables
  differ in length by one element by design).
- `tests/test_first_principles_mhd.py` — I001 unsorted imports.

## A-6 — resume ledger (`55e3f94`)

- `src/dpf/first_principles/segmented_whole_shot.py` — necessary: a resumed run
  rebuilt `_CumulativeLedgers` from zero, so skipped segments never accumulated.
  Added `to_state_dict`/`from_state_dict`/`_STATE_FIELDS`, a per-checkpoint
  cumulative-ledger sidecar write, and resume-time rehydration. The checkpoint
  `.npz` schema and `state_checkpoint.py` fail-closed loader were deliberately
  left untouched.
- `tests/test_first_principles_segmented_whole_shot.py` — added the audit's
  required regression test (resumed cumulative ledger equals an uninterrupted
  run's, every counter field).

## A-2 — quarantine (`1abe15a`)

- Three `results/audit_*.json` smoke artifacts moved into
  `results/archive_stale_pre_ssr_codex_a2_2026_05_18/` with a `QUARANTINE_NOTICE.md`.
  Necessary: they embed a stale generation commit and `dirty_worktree=true`. See
  `sprint_1/ARTIFACT_REGENERATION_OR_QUARANTINE_PLAN.md`.

## A-3 / A-4 — CI (`bf22c33`)

- `.github/workflows/ci.yml` — the `first-principles-audit` job ran the two
  verification gates in writing mode and linted only `results/*.json`. Necessary:
  switch the gates to `--check` (read-only, `--date` pinned), add a recursive
  `results/**/*.json` scan so archive-exemption policy is exercised, and add a
  final `git diff --exit-code` assertion.

## A-7 — SRS / RTM (`fe038f7`)

- `docs/DPF_REQUIREMENTS_BASELINE.md` — `DPF-PHYS-020` and `DPF-PHYS-023` were
  `blocked` though candidate implementations now exist. Status and blocker cells
  rewritten to `partial` with explicit remaining acceptance blockers.
- `docs/SRS_TRACEABILITY_MATRIX.{csv,json}` — regenerated from the baseline by
  `scripts/export_srs_traceability.py`.
- `docs/FIRST_PRINCIPLES_EXPERIMENTAL_SIMULATOR_SPRINT_PLAN_2026_05_18.md` —
  Sprint 0 marked closed-with-debt.
- `tests/test_srs_traceability_export.py` — pinned the new statuses with
  assertions.

## A-0 — audit document (`3dc4c11`)

- `docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md` (new) — the audit
  this submission answers; committed as the authority document.

---

## Sprint 1.1 — Hygiene (RC-2 through RC-7)

### RC-2 — repository-wide ruff cleanup (`9df8d3b`)

- 52 files across `src/dpf/` and `tests/` — necessary: `ruff check src/ tests/`
  was failing with 69 errors (I001 import order ×36, UP012/017/034/035/041/042
  modernizations, F401 unused imports, F841 unused vars, B905 zip strict=, E402,
  N812, SIM102, F821 missing import). All resolved behavior-preservingly. The CI
  lint job was red; RC-2 makes it green.
- `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}` — regenerated
  once to reflect the cleaned modules; `strict_passed` stays `true`.

### RC-4 / RC-1 — Sprint 1 audit and packet documents

- `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT1_CONTROL_GATE_2026_05_19.md` (added,
  `06cfd64`) — the Sprint 1 control-gate audit authority document.
- Sprint 1 submission packet (`fa9088e`) — documentation-only wrapper; all
  changed paths under
  `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/`.

### RC-3 / RC-5 / RC-6 / RC-7 — artifact linter C8 and package schema tests (`0bef78c`)

- `scripts/audit_first_principles_artifacts.py` — added linter check C8: an
  active artifact must carry `artifact_generation_commit`,
  `manifest.git_commit`, and `manifest.artifact_generation_commit` equal to
  live HEAD with `dirty_worktree False`. C8 degrades to a `SKIPPED` warning
  when git is unavailable. Necessary per RC-5: previously committed artifacts
  with stale commits could pass C1–C7.
- `tests/test_first_principles_artifact_linter.py` — added RC-6 positive
  current-HEAD fixture (proves real PASS, not only SKIP/EXEMPT across C1–C8);
  added RC-7 drift test asserting the C7 required-provenance tuple equals
  `manifest.REQUIRED_PROVENANCE_FIELDS`.
- `tests/test_external_team_submission_package.py` (new) — RC-3 test half:
  asserts every submission CSV row matches its header field count. 26
  linter/package tests pass.

### RC-4 — Sprint 1.1 finalization wrapper (`c52bed3`)

- `AUDIT_COMMANDS.md` — current-HEAD verification block appended for `0bef78c`;
  explains that the containing documentation commit cannot name itself.
- `BLOCKER_MATRIX.csv` — RC-2 through RC-7 rows marked closed.
  *(documentation-only wrapper commit)*

---

## Sprint 2 — Proposals and Source-Truth

### Auluck KR extract (`49e80ee`)

- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.{json,md}` — updated to include the
  verified Auluck 2021 extract; necessary because the auto-extracted KR markdown
  renders eqs. (2)–(14) as OCR-garbled tokens.
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}` —
  refreshed; `exhausted=true`, `open_issue_count=0` preserved.

### Sprint 2 proposal docs and tracked Auluck extract (`bd840f4`, `0585eec`)

- `docs/external_team_submissions/.../sprint_2/WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md`,
  `WP_N1B_AULUCK_EQ_5_6_SOURCE_STATUS.md`,
  `WP_N1B_RESIDUAL_TOLERANCE_SOURCE_STATUS.md`,
  `WP_N1B_TIME_CENTERING_PROPOSAL.md`,
  `WP_N4B_12US_ORCHESTRATION_PROPOSAL.md`,
  `WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md` (all new) — necessary
  to close RC-1; these are the six Sprint 2 proposal/status documents.
- `sprint_2/PENDING.md` — removed; superseded by the six proposal files.
- `sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md` (added, `0585eec`)
  — necessary because `KnowledgeReference/` is gitignored (local-only); the
  tracked copy makes the WP-N1B audit evidence self-contained and reviewable
  without local KR access.
- `BLOCKER_MATRIX.csv`, `CLAIMS_LEDGER.csv`, `SOURCE_PACKET_INDEX.csv`,
  `README.md` — updated to reflect RC rows and Sprint 2 coverage; CSV quoting
  fixed per RC-3.

### Sprint 2 audit and periodic runner (`6af4454`)

- `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_PACKET_2026_05_19.md` (new) — the
  Sprint 2 packet audit authority document.
- `scripts/run_codex_periodic_audit.py` (new) — autonomous periodic audit runner
  (single-cycle and looped modes); writes logs outside the worktree to avoid
  dirtying the repository state it checks.
