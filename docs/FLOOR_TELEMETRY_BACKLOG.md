# Floor Telemetry Migration Backlog

**Audit date:** 2026-04-30  
**Wave:** 7 (enumeration only — no migration performed)  
**Scope:** `src/dpf/` excluding `engine_archive/` and `.md` files  

---

## Summary

| Category | Count |
|---|---|
| Total `np.maximum`/`mx.maximum` hits | 291 |
| VIOLATION (bare numeric literal on state variable) | ~116 |
| DENOMINATOR (div-by-zero / magnitude guard, leave as-is) | ~140 |
| SAFE (uses named constant `_RHO_FLOOR` / `_P_FLOOR`) | ~35 |

The telemetry infrastructure exists at `src/dpf/metal/floor_telemetry.py` (`apply_floor(arr, floor_val, name, step)`). Several files in `src/dpf/metal/` already import `_RHO_FLOOR` / `_P_FLOOR` from `dpf.metal.constants` — those usages are SAFE and do not need migration.

---

## Classification Guide

**VIOLATION** — bare numeric literal applied directly to a state array:
```python
rho_new = np.maximum(rho_new, 1e-20)          # VIOLATION
rho_L = np.maximum(rho_L, 1e-30)              # VIOLATION
```

**DENOMINATOR** — guard against division by zero; not a state floor:
```python
a2 = gamma * p / np.maximum(rho, 1e-30)       # DENOMINATOR — leave
denom = np.maximum(S_R - S_L, 1e-30)          # DENOMINATOR — leave
inv_r = 1.0 / mx.maximum(r, 1e-30)            # DENOMINATOR — leave
```

**SAFE** — uses named constant already:
```python
rho = mx.maximum(U[IDN], _RHO_FLOOR)          # SAFE — already correct
```

---

## Per-File Violation Counts (all files with hits)

| File | Total hits | State violations | Denom guards | Status |
|---|---|---|---|---|
| `fluid/cylindrical_mhd.py` | 58 | ~23 | ~26 | **VIOLATION** |
| `fluid/mhd_solver.py` | 39 | ~14 | ~21 | **VIOLATION** |
| `metal/mlx_amr.py` | 18 | ~3 | ~1 | **VIOLATION** |
| `metal/mlx_transport.py` | 12 | ~7 | ~10 | **VIOLATION** |
| `metal/mlx_solver.py` | 11 | ~4 | ~6 | **VIOLATION** |
| `jax/lee_model.py` | 9 | TBD | TBD | **VIOLATION** |
| `fluid/nernst.py` | 8 | TBD | TBD | **VIOLATION** |
| `fluid/anisotropic_conduction.py` | 8 | TBD | TBD | **VIOLATION** |
| `metal/mlx_sources.py` | 7 | 0 (already uses `_RHO_FLOOR`/`_P_FLOOR`) | 7 | **SAFE** |
| `metal/mlx_operator_split.py` | 7 | ~5 | ~2 | **VIOLATION** |
| `fluid/viscosity.py` | 6 | TBD | TBD | **VIOLATION** |
| `diagnostics/derived.py` | 6 | TBD | TBD | **VIOLATION** |
| `turbulence/anomalous.py` | 5 | 0 (all denominator context) | 5 | **DENOMINATOR** |
| `engine/physics_operators.py` | 5 | TBD | TBD | **VIOLATION** |
| `athena_wrapper/athena_engine.py` | 5 | ~3 | ~2 | **VIOLATION** |
| `radiation/transport.py` | 4 | ~2 | ~2 | **VIOLATION** |
| `metal/mlx_kernels.py` | 4 | ~2 | ~2 | **VIOLATION** |
| `metal/mlx_eos.py` | 4 | ~3 | ~1 | **VIOLATION** |
| `fluid/eos.py` | 4 | 0 (all denominator context) | 4 | **DENOMINATOR** |
| `collision/spitzer.py` | 4 | TBD | TBD | **VIOLATION** |
| `metal/mlx_viscosity.py` | 3 | 0 (omega_ci, B_mag guards) | 3 | **DENOMINATOR** |
| `metal/mlx_riemann.py` | 3 | 0 (gm1 denominator) | 3 | **DENOMINATOR** |
| `metal/mlx_primitives.py` | 3 | 0 (E_abs magnitude) | 3 | **DENOMINATOR** |
| `metal/mlx_line_radiation.py` | 3 | 0 (uses `_RHO_FLOOR`/`_P_FLOOR`) | 0 | **SAFE** |
| `metal/mlx_coupling.py` | 3 | ~2 | ~1 | **VIOLATION** |
| `metal/mlx_bc.py` | 3 | ~1 | ~2 | **VIOLATION** |
| `experimental/pic/hybrid.py` | 3 | 0 (vector normalization) | 3 | **DENOMINATOR** |
| `diagnostics/interferometry.py` | 3 | TBD | TBD | TBD |
| *(remaining 29 files, 1–2 hits each)* | ~55 | ~30 | ~25 | mixed |

---

## Top 5 by Violation Count

| Rank | File | State violations | Floor values seen |
|---|---|---|---|
| 1 | `fluid/cylindrical_mhd.py` | ~23 | 1e-10, 1e-20, 1e-30 (inconsistent) |
| 2 | `fluid/mhd_solver.py` | ~14 | 1e-10, 1e-20, 1e-30 (inconsistent) |
| 3 | `metal/mlx_transport.py` | ~7 | 1e-10, 1e-12, 1e-20, 1e-30 |
| 4 | `metal/mlx_operator_split.py` | ~5 | 1e-10, 1e-30 |
| 5 | `metal/mlx_solver.py` | ~4 | 1e-4, 1e-12, 1e-30 |

---

## PR-Floor-1 Recommendation: `fluid/cylindrical_mhd.py`

**Rationale:**

- Highest state violation count (~23), all numpy — no MLX mixed backend complexity.
- Violations are concentrated in Riemann reconstruction (lines 570–573, 711–714, 739–742) and time integration (lines 897, 1138, 1437+). Localized clusters reduce diff scatter.
- File is numpy-only in the floor sections (no `mx.maximum` state ops), so `apply_floor` from `dpf.metal.floor_telemetry` plugs in with a single import.
- No existing `_RHO_FLOOR`/`_P_FLOOR` imports — clean slate, no import conflicts.
- Floor values are only 1e-10, 1e-20, 1e-30: maps cleanly to `RHO_FLOOR` / `P_FLOOR` constants from `dpf.metal.constants` (verify values match before wiring).
- 1,678 lines — large but bounded; violations are in contiguous solver blocks, not scattered helpers.

**Lower priority next:** `fluid/mhd_solver.py` (2,412 lines, same pattern, do after cylindrical is proven).  
**Avoid first:** `metal/mlx_amr.py` — violations are inside float-cast loops (`float(np.maximum(...))`) requiring different migration pattern.

---

## BLOCKERS

1. **Floor value mismatch risk.** `cylindrical_mhd.py` uses 1e-10, 1e-20, and 1e-30 inconsistently for rho and pressure — must verify which maps to `RHO_FLOOR` and `P_FLOOR` in `dpf.metal.constants` before substitution. Wrong constant = physics change masked as refactor.

2. **`apply_floor` is numpy-only.** `floor_telemetry.apply_floor` calls `np.maximum` internally and takes `np.ndarray`. Any site where the array is an MLX tensor requires a separate MLX-aware wrapper (or conversion). `cylindrical_mhd.py` is numpy throughout its floor sites — not blocked, but must verify before touching `metal/` files.

3. **No `step` propagation.** `apply_floor(arr, floor_val, name, step)` requires a timestep argument. `cylindrical_mhd.py` does not thread a `step` counter through all floor sites; callers will need to pass `-1` (unknown step) or the method signature needs updating. Confirm convention before starting.

4. **Missing test coverage.** No floor-activation tests exist in `tests/` for `cylindrical_mhd`. A regression suite detecting when floors mask physics (floor activation count rises unexpectedly) should be written as part of PR-Floor-1, not after.

---

## Migration Pattern (for PR-Floor-1)

```python
# Before
rho_L = np.maximum(rho_L, 1e-20)

# After
from dpf.metal.floor_telemetry import apply_floor
rho_L = apply_floor(rho_L, _RHO_FLOOR, "rho_L", step=-1)
```

Import `_RHO_FLOOR`, `_P_FLOOR` from `dpf.metal.constants` at top of file. Confirm constant values match the literals currently in place before committing.
