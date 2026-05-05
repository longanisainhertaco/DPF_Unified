# PR Queue Health — 2026-05-04

Generated: 2026-04-30 (snapshot taken 2026-05-04/05)

---

## PR Status Table

| # | Title | Branch | Status | CI | Mergeable |
|---|-------|--------|--------|----|-----------|
| #6 | fix(lambda): hygiene cleanup — fabrications removed, post-rebase | `fix/lambda-hygiene` | RED | PASS (stale) | CONFLICTING |
| #7 | chore: W4 dead-code purge — remove 3,166 LOC | `chore/w4-dead-code-purge` | YELLOW | FAIL (lint only; test/validate SKIPPED) | MERGEABLE |
| #8 | chore(lambda): cherry-pick docs hygiene + apply_floor migration + EMPIRICAL markers | `chore/lambda-hygiene` | YELLOW | FAIL (lint only; test/validate SKIPPED) | MERGEABLE |
| #9 | fix: resolve experimental_waveforms conflict + post-merge doc sync + Toh 2025 ψ(n_i) limiter | `chore/post-merge-doc-sync` | YELLOW | FAIL (lint only; test/validate SKIPPED) | MERGEABLE |
| #10 | fix(ci): ruff lint hotfix — unstick main CI | `fix/ci-ruff-hotfix` | YELLOW* | lint PASS; tests IN_PROGRESS | MERGEABLE |
| #11 | fix(mjolnir): correct 2-MJ L0 to 46.7 nH per Petrov 2022 §II.A | `fix/mjolnir-l0-petrov-2022` | YELLOW | FAIL (lint only; test/validate SKIPPED) | MERGEABLE |

*#10 CI was actively running at snapshot time — lint passed, tests in progress.

---

## Classification

### RED (1)
- **#6** — `mergeStateStatus: DIRTY`, `mergeable: CONFLICTING`. CI checks passed on the original commit (2026-04-27) but main has advanced since; the branch now has conflicts. Tests cannot run until resolved.

### YELLOW (5)
- **#7, #8, #9, #11** — `mergeStateStatus: UNSTABLE`. Lint (`ruff`) fails; downstream jobs (test, validation, athena-test, smoke-test) are all SKIPPED because CI gates on lint. The lint failure is a pre-existing ruff config issue on main — not a code defect in these PRs.
- **#10** — lint already passes (it *is* the ruff fix); tests were IN_PROGRESS at snapshot time. Likely GREEN once tests complete.

### GREEN (0)
None currently — #10 is the closest and should flip GREEN on test completion.

---

## Root-Cause Analysis

### Why do #7–#9 and #11 all fail lint?
All four PRs were opened on 2026-05-01 or earlier, before the ruff hotfix (PR #10, opened 2026-05-05). The lint failure is inherited from the base branch state at that time. The fix is already queued in #10 — merging #10 first will unblock re-running CI on the others.

### Why is #6 CONFLICTING?
PR #6 (`fix/lambda-hygiene`) was rebased and pushed 2026-04-27. In the ~3 days since, other PRs touching overlapping lambda/hygiene files landed on main, creating conflicts. Needs rebase onto current main before it can be merged.

---

## Recommended Merge Order

1. **#10** (fix/ci-ruff-hotfix) — merge first. Fixes the ruff config; unblocks CI for all other PRs. Wait for in-progress tests to pass.
2. **#7** (W4 dead-code purge) — safe structural cleanup, no physics logic changes. Low conflict surface. Re-trigger CI after #10 lands.
3. **#8** (lambda hygiene + EMPIRICAL markers) — depends on dead code being gone; cleaner to apply after #7.
4. **#9** (experimental_waveforms + Toh 2025 limiter) — physics change; merge after hygiene PRs are in so diff is minimal.
5. **#11** (Mjolnir L0 correction) — physics parameter fix, independent of #7–#9 but benefits from clean CI baseline.
6. **#6** (lambda hygiene, post-rebase) — **requires manual rebase first**. Superseded in part by #8; author should confirm there is no duplication before merging.

---

## Blockers Per PR

| # | Blocker | Action Required |
|---|---------|-----------------|
| #6 | Merge conflict — DIRTY | Rebase onto main; verify no overlap with #8 |
| #7 | Lint CI failure (inherited) | Wait for #10 to merge, then re-trigger CI |
| #8 | Lint CI failure (inherited) | Wait for #10 + #7, then re-trigger CI |
| #9 | Lint CI failure (inherited) | Wait for #10 + #7 + #8, then re-trigger CI |
| #10 | Tests IN_PROGRESS | Wait for test jobs to complete |
| #11 | Lint CI failure (inherited) | Wait for #10, then re-trigger CI; can merge independently of #7–#9 |

---

## Key Observations

- The single root cause for 4 of 5 YELLOW PRs is the ruff config on main — PR #10 is the unlock.
- All SKIPPED downstream jobs (test, validation, athena-test, smoke-test) are gated on lint; there is no evidence of actual test failures.
- PR #6 is the only true conflict requiring developer action (rebase).
- After #10 lands and CI is re-triggered, the queue could clear in a single session if no test failures surface.
