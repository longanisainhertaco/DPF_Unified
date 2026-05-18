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
| 6 | `.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check` | exit 0 — `exhausted: true`, `open_issue_count: 0`; no worktree write |
| 7 | `.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check` | exit 0 — `strict_passed: true`, `total_modules: 289`, `missing_source_reference_count: 0`; no worktree write |
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
5, and a final `git diff --exit-code`, in that order. CI pins `--date 2026_05_18`
on commands 6 and 7 so the read-only check compares against the committed
baseline dated docs regardless of the CI run date; the manual audit commands
above use the default date, which resolves to `2026_05_18` on the audit date.
See `UNKNOWN_AND_INFERENCE_LOG.md` entry I-1.
