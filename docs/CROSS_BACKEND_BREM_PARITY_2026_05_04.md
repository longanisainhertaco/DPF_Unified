# Cross-Backend Bremsstrahlung Parity Report
Date: 2026-05-04
Context: Wave-6 S18, post-PR #9 merge (Metal import conflict resolved)

## Test Suite Results

Command:
```
pytest tests/test_metal_gpu_consolidated.py::TestBremsstrahlungConsistency \
       tests/test_mlx_sources.py::TestBremsstrahlung \
       tests/test_bremsstrahlung_nrl.py \
       tests/test_verification_consolidated.py -k brem -v --no-header --tb=line
```

Result: 24 passed, 0 failed (4.93 s)

| Test File | Tests | Result |
|-----------|-------|--------|
| test_metal_gpu_consolidated.py::TestBremsstrahlungConsistency | 1 | PASS |
| test_mlx_sources.py::TestBremsstrahlung | 4 | PASS |
| test_bremsstrahlung_nrl.py | 4 | PASS |
| test_verification_consolidated.py (brem filter) | 15 | PASS |

## Fixed-Input Canonical Case

Input: ne = 1.00e+25 m^-3, Te = 1.00e+07 K, Z = 1.0, g_ff = 1.2

Formula: P_ff = 1.569e-40 * g_ff * Z * ne^2 * sqrt(Te)   [W/m^3]
Source: NRL Plasma Formulary (2019) eq.(30), p.58

| Backend | Precision | Method | P_brem [W/m^3] |
|---------|-----------|--------|----------------|
| Python  | float64   | numba @njit | 5.95393638e+13 |
| MLX     | float32   | log-space MLX | 5.95391477e+13 |
| Metal   | float64   | numba @njit (CPU path) | 5.95393638e+13 |

Note: Metal backend wraps the same `bremsstrahlung_power` numba kernel as Python;
the distinction is confirmed by TestBremsstrahlungConsistency passing with rel_err < 5%.
Athena C++ excluded — O3 fork in progress.

## Pairwise Relative Error Matrix

|            | Python      | MLX         | Metal       |
|------------|-------------|-------------|-------------|
| Python     | —           | 3.629e-06   | 2.624e-16   |
| MLX        | 3.629e-06   | —           | 3.629e-06   |
| Metal      | 2.624e-16   | 3.629e-06   | —           |

All errors well within 1% float32 tolerance threshold.
MLX deviation (~3.6e-6) is float32 rounding in log-space arithmetic — expected and acceptable.
Python-Metal deviation (~2.6e-16) is machine epsilon (float64 identity path).

## VERDICT

Parity OK. All three active backends agree within 3.6e-6 relative error on the canonical
bremsstrahlung case. No regression introduced by PR #9.
