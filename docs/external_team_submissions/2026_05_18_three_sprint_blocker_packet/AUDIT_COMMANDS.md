# Audit Commands — Submission 1

## Environment

- Interpreter: `.venv312/bin/python` — Python 3.12.13
- Repository: DPF-Unified, branch `codex/corpus`
- HEAD at verification: `fe038f70cb454505fe710aee23f9d0d35fb267ec`
- Date: 2026-05-18

## Codex Submission-1 command list and results

All commands run from the repository root. Results are post-commit.

| # | Command | Result |
| --- | --- | --- |
| 1 | `git status --short` | clean |
| 2 | `git diff --check` | clean |
| 3 | `.venv312/bin/python -m ruff check scripts/verify_first_principles_source_truth_exhaustion.py scripts/audit_first_principles_artifacts.py src/dpf/first_principles src/dpf/fields tests/test_first_principles_*.py tests/test_hybrid_3d_*.py tests/test_cli_first_principles_3d.py` | `All checks passed!` |
| 4 | `.venv312/bin/python scripts/audit_first_principles_artifacts.py 'results/*.json'` | `42 scanned -- 0 first-principles, 31 skipped, 11 exempt, 0 passed, 0 failed` — PASS |
| 5 | `.venv312/bin/python scripts/audit_first_principles_artifacts.py 'results/**/*.json'` | `81 scanned -- 39 first-principles, 31 skipped, 50 exempt, 0 passed, 0 failed` — PASS |
| 6 | `.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_18` | exit 0 — `exhausted: true`, `open_issue_count: 0`; no worktree write |
| 7 | `.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_18` | exit 0 — `strict_passed: true`, `total_modules: 289`, `missing_source_reference_count: 0`; no worktree write |
| 8 | `.venv312/bin/python -m pytest tests/test_first_principles_manifest.py tests/test_first_principles_artifact_linter.py tests/test_first_principles_segmented_whole_shot.py tests/test_srs_traceability_export.py tests/test_first_principles_verification_check_mode.py -q -rx` | `62 passed` |
| 9 | `.venv312/bin/python -m pytest tests/test_first_principles_*.py tests/test_hybrid_3d_*.py tests/test_cli_first_principles_3d.py -q -rx` | `275 passed, 9 warnings` |
| 10 | `git diff --exit-code` | clean |

Command 8 adds `tests/test_first_principles_verification_check_mode.py` (the new A-3
test file) to the audit's four-file focused set; the four-file set alone is a
subset and also passes.

The 9 warnings in command 9 are PlasmaPy `CouplingWarning` strong-coupling
Coulomb-logarithm warnings. They are not test failures. They are physics-regime
warnings and are routed to WP-N5 closure policy (`BLOCKER_MATRIX.csv` rows
`WP-N5-COL`); they are not in Sprint 1 scope.

## CI command equivalence

`.github/workflows/ci.yml` job `first-principles-audit` runs commands 6, 7, 4,
5, and a final `git diff --exit-code`, in that order.

Commands 6 and 7 pass `--date 2026_05_18` explicitly. `--date` otherwise
defaults to UTC-today, and the committed baseline dated docs are
`FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}` and
`FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`; pinning the date
makes the read-only check resolve to the committed baseline when re-run on any
date. CI uses the same pinned form. See `UNKNOWN_AND_INFERENCE_LOG.md` entry I-1.

## Current-HEAD verification — Sprint 1.1 (2026-05-19)

Codex Sprint 1 audit RC-4 requires a verification block at the current HEAD.
Recorded after the Sprint 1.1 hygiene fixes (RC-2…RC-7) and the Sprint 2
proposal set were committed.

- `git rev-parse HEAD`: `0bef78cdd04e933d62e404e24943f2bf28606be4`
- `git status --short`: clean
- `git diff --check`: clean
- `git diff --exit-code`: clean
- `ruff check src/ tests/`: `All checks passed!` (RC-2 — repository-wide ruff debt cleared)
- `audit_first_principles_artifacts.py 'results/*.json'`: `PASS -- all first-principles artifacts pass C1-C8`
- `audit_first_principles_artifacts.py 'results/**/*.json'`: `PASS -- all first-principles artifacts pass C1-C8` (C8 added per RC-5)
- `verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_18`: exit 0
- `verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_18`: exit 0
- focused Submission set (manifest, artifact-linter, segmented-whole-shot, srs-traceability, verification-check-mode, external-team-submission-package): `75 passed`
- broad first-principles/hybrid suite (`tests/test_first_principles_*.py tests/test_hybrid_3d_*.py tests/test_cli_first_principles_3d.py`): `275 passed`

A committed transcript cannot name the commit that contains it: this
`AUDIT_COMMANDS.md` update is committed as the single documentation-only commit
on top of `0bef78cdd04e933d62e404e24943f2bf28606be4`. The verified state above
is that commit's tree. This is the same structural constraint documented for
stale active artifacts in `sprint_1/ARTIFACT_REGENERATION_OR_QUARANTINE_PLAN.md`.
