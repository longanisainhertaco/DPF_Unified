# Artifact Regeneration or Quarantine Plan (A-2)

Audit finding: A-2. Commit: `1abe15a`.

## Problem

Three active first-principles result artifacts embedded
`artifact_generation_commit: 466a0a54e992acf61a9dd0f2d12e7e15fd23e9af` (older
than HEAD) and `dirty_worktree: true`:

- `results/audit_first_principles_3d_smoke.json`
- `results/audit_experimental_whole_shot_smoke.json`
- `results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json`

The audit allows two corrections: regenerate from a clean HEAD, or quarantine.

## Decision: quarantine all three

Moved to `results/archive_stale_pre_ssr_codex_a2_2026_05_18/` with a
`QUARANTINE_NOTICE.md`. The artifact linter exempts any path containing the
substring `archive_stale_pre_ssr`.

## Rationale

1. **Dirty-worktree provenance is unrecoverable.** All three carry
   `dirty_worktree: true` — they were generated from an uncommitted tree. The
   exact code and inputs that produced them cannot be reconstructed from any
   commit. Regenerating would not repair the existing files; it would replace
   them, and the audit reject condition is about what the *committed* artifacts
   embed.

2. **A committed active artifact cannot embed its own commit.** If an artifact
   is regenerated at HEAD `X` and then committed, the commit that contains it is
   a child of `X`; the artifact's `artifact_generation_commit` is `X`, one
   commit behind the new HEAD. The audit's reject condition — "active artifacts
   embed an `artifact_generation_commit` older than HEAD" — would then be tripped
   by construction. Quarantine removes the artifacts from the *active* set
   entirely, so the audit step "active artifact generation commit equals HEAD"
   is satisfied because the active set is empty.

3. **Zero claim cost.** All three already carry
   `can_support_first_principles_acceptance: false`. They are smoke telemetry,
   not evidence. Quarantining them removes no claim capability. The runtime is
   exercised at HEAD by the 275-test suite, not by these files.

4. **It makes A-4 testable.** Quarantining produces real archived artifacts for
   the recursive CI scan (`results/**/*.json`) to report as `EXEMPT`, which is
   exactly the archive-policy coverage A-4 asks for.

5. **The expensive one stays expensive.** `audit_limiter_proof_auluck_power_port`
   is a 4632-step run. Regenerating it spends wall-clock time for an artifact
   that supports no claim.

## Result

- Active scan `results/*.json`: 0 first-principles artifacts, 0 failed.
- Recursive scan `results/**/*.json`: 39 first-principles, 50 exempt (47 prior
  + 3 quarantined here), 0 failed.
- Hashes and embedded provenance: `../ARTIFACT_HASHES.csv`.

## If active artifacts are wanted later

A future sprint may regenerate smoke artifacts from a clean committed HEAD and
accept that `artifact_generation_commit` equals the last code commit (the parent
of the artifact-only commit). That is a deliberate, documented choice; until
then the active first-principles result set is intentionally empty and the
project relies on the test suite as the runtime-exercise evidence.
