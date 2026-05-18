# Quarantine Notice — Codex Audit A-2

Date: 2026-05-18
Authority: `docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md`, finding A-2.

## Quarantined artifacts

- `audit_first_principles_3d_smoke.json`
- `audit_experimental_whole_shot_smoke.json`
- `audit_limiter_proof_auluck_power_port_1us_2026_05_18.json`

## Why

All three embed `artifact_generation_commit: 466a0a54e992acf61a9dd0f2d12e7e15fd23e9af`
and `dirty_worktree: true`. They were generated from a stale, uncommitted
worktree that predates the current branch HEAD. An artifact generated from a
dirty worktree has no reproducible provenance and cannot be regenerated to embed
its own committed-state commit, so it cannot be a trustworthy *active* result
artifact.

Per Codex audit A-2 they are quarantined here rather than regenerated.

## Status

These artifacts **cannot support any active first-principles claim**. They are
retained only as historical telemetry. They already carry
`can_support_first_principles_acceptance: false`.

The artifact linter (`scripts/audit_first_principles_artifacts.py`) treats every
path containing the substring `archive_stale_pre_ssr` as `EXEMPT`: quarantined
artifacts are visibly excluded from the C1–C7 first-principles checks and are
recorded as exempt by the recursive (`results/**/*.json`) CI scan.
