# Patch Scope — Submission 1

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
