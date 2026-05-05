# Submodule Hygiene

## Overview

Two external submodules with distinct ownership and fix strategies.

| Submodule | Remote | Pinned Commit | Status |
|---|---|---|---|
| `external/athena` | PrincetonUniversity/athena | `5ea9ab7c` (v24.0-148) | dirty — HEAD drift |
| `external/athenak` | IAS-Astrophysics/athenak | `d2421b12` (heads/main) | clean |

---

## external/athenak — bin/ Artifacts

**Problem:** `athenak/bin/` contains build artifacts (`athena`, `athena_briowu`, `athena_cartesian`, `athena_cylindrical`, `athena_sod`, `athenak`) that pollute submodule status.

**Root cause:** Upstream `.gitignore` covers `build*/` and `*.bin` but not `bin/` as a directory.

**Upstream status:** `IAS-Astrophysics/athenak` — no push access. No upstream PR filed (out of scope).

**Fix applied:** Local-only exclude at `.git/modules/external/athenak/info/exclude`:
```
bin/
```

**Result:** `git status` shows `athenak` as clean. This exclude persists in the local clone but is not committed to the repo — it must be re-applied after fresh clones.

**Re-apply after fresh clone:**
```bash
echo 'bin/' >> .git/modules/external/athenak/info/exclude
```

Add this to `scripts/setup.sh` or the onboarding checklist if contributors build athenak locally.

---

## external/athena — HEAD Drift

**Problem:** `git status` shows `M external/athena` because the submodule HEAD (`5ea9ab7c`, v24.0-148) is ahead of the pinned commit recorded in the parent repo.

**Root cause:** Commits were made inside the submodule (DPF-specific physics fixes: Braginskii transport, Spitzer resistivity, Bremsstrahlung coefficient) without updating the parent repo's submodule pointer.

**Upstream status:** `PrincetonUniversity/athena` — this is a fork with local commits not present upstream. No push access to upstream.

**Fix required:** Update the parent repo's submodule pointer to the current HEAD:
```bash
git -C /path/to/dpf-unified add external/athena
git commit -m "chore: advance athena submodule pointer to 5ea9ab7c"
```

This is not done automatically — confirm the `athena` HEAD is stable before committing the pointer advance.

---

## Decision: Local Exclude vs Upstream PR

Both submodules are upstream-pinned (no push access). Local exclude is the correct long-term strategy for `athenak/bin/`. Document the exclude in setup scripts so fresh clones are not silently dirty.

For `athena`, the dirty state is a submodule pointer drift, not a `.gitignore` gap — fix by committing the updated pointer when the local commits are finalized.
