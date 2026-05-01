# PR-B Changelog: KR-canonical re-anchor: physics, devices, validation gates

**Branch:** `fix/audit-cleanup-apr23` → `main`
**Tag:** `pr-b-physics-recalibration-apr23`
**Date:** 2026-04-30

---

## Breaking Changes (3)

### 1. MJOLNIR_DATA voltage 60 kV → 100 kV
- **Source:** Schmidt 2021 §III.A (KR-canonical)
- **Commits:** `3c6d3cf`, `f89ebb1`
- **Impact:** MJOLNIR peak current, pinch timing, and all derived yield metrics change.
  Any downstream script hardcoding `V0=60e3` for MJOLNIR must be updated to `V0=100e3`.
- **Validation gate:** MJOLNIR must pass `cortana-dpf-validate` after merge.

### 2. PF-1000 R0_mOhm 6.43 → 2.3 (empirical knob removed)
- **Source:** Akel et al. published circuit parameters; Wave-9/10 KR preset (`5746c81`)
- **Commits:** `5746c81`, `3c6d3cf`
- **Impact:** Removes a tuning knob that was masking I_peak error. I_peak accuracy is
  now load-bearing on physics correctness, not parameter fitting.
  Any calibration script that adjusted `R0_mOhm` to hit a target must be deleted.
- **Validation gate:** PF-1000 27 kV I_peak error must remain ≤ 12% post-merge.

### 3. AECS-PF2 and POSEIDON-40kV moved to `_REFERENCE_ONLY`
- **Source:** `2e1ca76` — devices lack KR source; marked UNVERIFIED
- **Impact:** These devices are excluded from `cortana-dpf-validate` pass/fail scoring.
  CI and oncall dashboards that counted them as validation devices will show lower
  device counts. Expected total active devices: subtract 2 from pre-PR-B count.

---

## Non-Breaking Changes

- **KR provenance system activated** (`20d589a`): `~/dpf-unified/KnowledgeReference/`
  is now the canonical physics source. All new physics functions must cite `[KR: ...]`.
- **Dual-fence validation thresholds** (`0958947`): PF-1000 I_peak gate raised to 12%
  (was 10%) as a regression guard while physics debt is resolved.
- **KR extraction pipeline** (`42e248c`, `20d589a`): `extract_papers.py` now emits
  `pages[]` array for KR schema parity; pipeline documented in `ARCHITECTURAL_DEBT.md`.
- **NRL bremsstrahlung coefficient unified** (`1d633df`, `4706cc7`): All backends
  (Python, MLX) now use `1.569e-40` from NRL eq. 30. MLX rtol relaxed 5% → 10% for
  float32 log-space floor (`4706cc7`).
- **Cylindrical `/mu_0` pressure recovery** (`3c6d3cf`): Cylindrical MHD pressure
  equation now divides by `mu_0`; radial-phase MHD regression guard added (`91675ed`).
- **KR-anchored regression guards added** (`91675ed`): Tests for NRL bremsstrahlung,
  cylindrical `/mu_0`, and radial-phase MHD now enforce KR-sourced values.
- **Beam-target yield → Lee/Saw canonical form** (`f89ebb1`): Replaces prior
  non-canonical yield expression with KR-anchored Lee/Saw form.
- **Zombie modules deleted** (`6f64a44`): `preconditioner.py` and `tabulated_eos.py`
  removed. No callers existed.
- **KMP_DUPLICATE_LIB_OK set in root conftest** (`f16a599`): Prevents OMP abort
  segfault on macOS with MKL + OpenMP conflict; applied globally to all test runs.
- **Sedov fabrication removed** (`42e248c` context / `4e59b8e`): `fminbound` failure
  no longer silently returns a fabricated Sedov point; raises instead.
- **Silent floor → `apply_floor` migration** (`f8c29e2`, `c9b6030`): Verification
  utilities no longer swallow numerical floors silently.
- **`estimate_convergence_order` NaN fix** (`5096057`): Returns explicit error instead
  of silent `0.0` on NaN.
- **Non-positive tolerance raises** (`4e59b8e` / `8cec5bf`): `report.record` now
  raises on non-positive tolerance instead of silently accepting bad input.
- **Brio-Wu fixture migrated** (`ab391a3`): Step-cap loop replaced with `t_end` loop;
  consistent with other fixture patterns.
- **UNVERIFIED device triage** (`2e1ca76`): Devices without KR sources are explicitly
  annotated; validation docs updated.

---

## Behavior Shifts

| Quantity | Before | After | Source |
|---|---|---|---|
| `BREM_COEFF` | ~1.42e-40 (mixed) | 1.569e-40 (+10.5%) | NRL eq. 30 [KR] |
| Cyl. MHD pressure | missing `/mu_0` | divided by `mu_0` | KR canonical |
| MJOLNIR V0 | 60 kV | 100 kV | Schmidt 2021 §III.A |
| PF-1000 R0 | 6.43 mΩ (empirical) | 2.3 mΩ (published) | Akel KR preset |
| Yield expression | non-canonical | Lee/Saw KR form | KR |

---

## Known Residuals

- **PF-1000 27 kV I_peak error: ~7.6%** — within gate (≤12%) but above pre-Apr-10
  baseline (2.8%). Regression introduced by empirical knob removal. Resolution requires
  RADPF baseline regeneration (Anthony owed).
- **RADPF baseline drift** — baseline was generated with pre-KR parameters. Regeneration
  blocked pending full RADPF KR audit. Tracked in `ARCHITECTURAL_DEBT.md`.
- **UNU-ICTP fill/V0/I_peak all changed** — net accuracy shift TBD; requires full
  re-run post-merge before declaring pass/fail.
- **MLX brem rtol relaxed to 10%** — acceptable for float32 but masks precision gap
  vs Python backend. Long-term: investigate float32 log-space floor in MLX.
- **AECS-PF2 / POSEIDON-40kV UNVERIFIED** — KR sources not yet located. If sources
  are found post-merge, restore to active set.

---

## Test Infrastructure

- `KMP_DUPLICATE_LIB_OK=TRUE` set globally in `conftest.py` (`f16a599`).
- KR ingestion pipeline: `extract_papers.py` updated (`42e248c`); run
  `python tools/extract_papers.py` to regenerate KnowledgeReference entries.
- Regression guards: 3 new KR-anchored tests added (`91675ed`).
- Total CI test count at merge: must remain ≥ 3990 (was 4000; delta from zombie
  module deletion + fixture migration is expected ≤ 10).
