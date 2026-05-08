# PR Queue Health — 2026-05-05 (v2)

Generated: 2026-05-05 | Snapshot of open PR queue against `main`.

---

## Summary

**Total open PRs: 25** (numbers #6–#30; note #6 is CONFLICTING)

| Status | Count | Notes |
|--------|-------|-------|
| GREEN | 0 | No PR has all checks passing |
| YELLOW | 3 | Mergeable; CI in-progress or partial-pass (#10, #27, #6*) |
| RED | 22 | lint FAILURE on all; test/validation SKIPPED downstream |

*#6 is DIRTY (CONFLICTING) — superseded by #8 cherry-pick; CI was green on its own run but merge conflict makes it unmergeable.

**Root cause of universal YELLOW/RED**: `fix/ci-ruff-hotfix` (#10) is the CI infrastructure fix — lint passes on #10, but all other branches were cut before or after that fix landed on main, and none rebased. Every non-#10 branch shows `lint: FAILURE` with test/validation SKIPPED. Once lint is fixed on main (via #10), rebased branches will unlock their test gates.

---

## Per-Category Status

### CI Infrastructure (#10) — YELLOW
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #10 | fix(ci): ruff lint hotfix | lint:PASS, test(3.11):FAIL, validation:PASS, athena:PASS | MERGEABLE |

lint passes; test(3.11) fails (the actual test suite bug this queue is queued behind). This is the gate PR.

### Big Cleanup (#7) — RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #7 | chore: W4 dead-code purge (-3,166 LOC) | lint:FAIL (skips all) | MERGEABLE |

### Hygiene Cherry-Picks (#8, #9) — RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #8 | chore(lambda): cherry-pick hygiene | lint:FAIL | MERGEABLE |
| #9 | fix: experimental_waveforms conflict + doc sync | lint:FAIL | MERGEABLE |

#8 supersedes #6 (which is CONFLICTING). Merge #8 before closing #6.

### Architecture Refactors (#17, #24) — RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #17 | refactor(validation): D1-A engine_validation SSoT | lint:FAIL | MERGEABLE |
| #24 | feat(mjolnir): config split MJOLNIR-1MJ + MJOLNIR-2MJ | lint:FAIL | MERGEABLE |

### Physics Fixes / KR-Anchored (#11, #16, #20, #21, #23, #28, #29) — RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #11 | fix(mjolnir): L0 → 46.7 nH per Petrov 2022 | lint:FAIL | MERGEABLE |
| #16 | fix: UNU-ICTP V0 → KR-canonical 15 kV | lint:FAIL | MERGEABLE |
| #20 | fix(nx2): mass_fraction 1.0→0.10, UNVERIFIED tag | lint:FAIL | MERGEABLE |
| #21 | fix(unu): fill_pressure 3→4 Torr per Lee & Saw 2014 | lint:FAIL | MERGEABLE |
| #23 | fix(pf1000_akel): KR-anchor fmr+fcr, fcr typo | lint:FAIL | MERGEABLE |
| #28 | docs(pf1000): promote n_cathode_rods=12 → KR-anchored | lint:FAIL | MERGEABLE |
| #29 | fix(poseidon_60kv): KR-anchor Lee fits, Lee & Saw 2014 | lint:FAIL | MERGEABLE |

### Test xfails / Engine Defects (#13, #18, #22, #27) — YELLOW/RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #13 | fix(validation): drop POSEIDON 40kV from targets | lint:FAIL | MERGEABLE |
| #18 | test(sod): xfail Sod L1 convergence (non-conservative) | lint:FAIL (2 runs) | MERGEABLE |
| #22 | test(mlx): xfail m6 discharge dt-collapse | lint:FAIL | MERGEABLE |
| #27 | test: xfail test_pcf_dominant | lint:PASS, tests IN_PROGRESS | MERGEABLE |

#27 is the only PR with lint passing + tests running (it rebased onto the lint fix). YELLOW.

### Submodule Operations (#19) — RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #19 | fix(athena): fork+pin BREM_COEFF SI + ion_mass 9-sig-fig | lint:FAIL | MERGEABLE |

### Documentation / Cleanup (#12, #25, #26, #30) — RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #12 | chore(validation): document experimental.py facade | lint:FAIL | MERGEABLE |
| #25 | docs(wave-7): 15 audit + design + backlog docs | lint:FAIL | MERGEABLE |
| #26 | test(toh): pinch convergence sweep | lint:FAIL | MERGEABLE |
| #30 | docs(poseidon): tag 40kV UNVERIFIED | lint:FAIL | MERGEABLE |

### New Physics Feature (#14, #15) — RED
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #14 | feat(kr): ingest Lee & Saw 2008 JoFE, NX2_DATA reference | lint:FAIL | MERGEABLE |
| #15 | feat(mhd): Sun 2025 §2.4 Eq.18 wall BCs (opt-in) | lint:FAIL | MERGEABLE |

### Conflicting (superseded) (#6) — RED/CONFLICT
| PR | Title | CI | Mergeable |
|----|-------|----|-----------|
| #6 | fix(lambda): hygiene cleanup (post-rebase) | lint:PASS (old run) | CONFLICTING |

Close #6 — superseded by #8.

---

## Dependency Graph

```
#10 (ci-ruff-hotfix) ──┐
                        ├─► ALL other PRs unblock for lint
#6 (CLOSE — superseded by #8)

Ordering within unblocked set:
  #7 (dead-code purge) ──► reduces merge surface for all subsequent PRs
  #8 (lambda hygiene) ──► prereq cleanup before #9 doc sync
  #9 (post-merge doc sync) ──► consolidates after #7+#8
  #11 (mjolnir L0) ──► prereq for #24 (mjolnir config split uses corrected L0)
  #16 (UNU-ICTP V0) ──► prereq for #21 (fill_pressure fix on same device)
  #23 (pf1000 fmr+fcr) ──► prereq for #28 (pf1000 n_cathode_rods promo same SSoT)
  #14 (kr ingest NX2) ──► prereq for #20 (nx2 mass_fraction fix references KR)
  #13 (drop POSEIDON 40kV) ──► prereq for #30 (tag POSEIDON 40kV UNVERIFIED in docs)
  #17 (validation SSoT) ──► prereq for #18 (sod xfail references validation refactor)
  #19 (athena fork+pin) ──► prereq for #22 (m6 xfail touches athena-test gate)
  #27 (pcf xfail) — independent (lint already passes)
  #15 (wall BCs) — independent feature, no blockers
  #12 (docs) — independent
  #25, #26 (wave-7 docs) — independent
```

---

## Recommended Merge Order (fastest path to all-green main)

| Step | PR | Category | Rationale |
|------|----|----------|-----------|
| 1 | **#10** | CI infra | Unblocks lint for all 22 red PRs |
| 2 | **#7** | Cleanup | -3,166 LOC reduces conflict surface for everything after |
| 3 | **#8** | Hygiene | Supersedes #6; close #6 after |
| 4 | **#9** | Hygiene | Doc sync depends on #7+#8 merged |
| 5 | **#11** | Physics | Mjolnir L0 prerequisite for #24 |
| 6 | **#24** | Arch | Mjolnir config split (needs corrected L0 from #11) |
| 7 | **#16** | Physics | UNU-ICTP V0 prereq for #21 |
| 8 | **#21** | Physics | Fill pressure fix (same device as #16) |
| 9 | **#23** | Physics | PF1000 fmr+fcr prereq for #28 |
| 10 | **#28** | Physics | PF1000 n_cathode_rods promo (same SSoT as #23) |

**Remaining after top-10** (no hard blockers, any order):
#14 → #20, #13 → #30, #17 → #18, #19 → #22, #27, #29, #15, #12, #25, #26

**Close without merging:** #6 (CONFLICTING, superseded)

---

## Critical Finding

Every PR in this queue fails CI for the same reason: ruff lint (introduced when a line-length or import-order rule was added to CI). #10 is the targeted fix. Merging #10 first, then rebasing all branches, would turn 22 RED PRs to YELLOW instantly — at which point actual test failures (#10's test(3.11) failure) become the new gate to investigate.

#27 already has lint passing (it must have rebased) and tests are IN_PROGRESS — watch it first for a signal on whether test(3.11) failures are per-PR bugs or a shared engine regression.
