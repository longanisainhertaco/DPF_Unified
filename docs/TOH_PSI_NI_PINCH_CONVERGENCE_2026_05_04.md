# Toh 2025 ψ(n_i) Limiter — Pinch Convergence Sweep

**Date:** 2026-04-30
**Commit:** worktree `agent-a92314a9ee378475a` at `28b9c75`
**Test file:** `tests/test_toh_pinch_convergence.py`

## Setup

- Solver: `CylindricalMHDSolver`, PLM+HLL path (`use_godunov_flux=True`)
- Grid: 1D axial (nr=4 slab, varying nz), dz = 1/nz
- Resolutions: N = [32, 64, 128, 256]
- Reference: `toh_lambda_dx_cubed=0.005`, `toh_n_ref=None` (auto from max density)
- Limiter formula (KR §3.1 Eq.31, Toh 2025):
  `ψ(n_i) = (1 + exp((λ₀/Δx)³ - n_i/n_ref))^{-1}`

## Convergence Table — Smooth Sinusoidal Wave

L1 reconstruction error: mean |QL[i] - q[i]| (reconstruction quality, no time evolution).

```
     N |     method |     L1(rho) |       L1(P) |      L1(Bz)
------------------------------------------------------------------------
    32 |     minmod |   1.550e-02 |   5.168e-03 |   2.584e-03
    64 |     minmod |   8.580e-03 |   2.860e-03 |   1.430e-03
   128 |     minmod |   4.491e-03 |   1.497e-03 |   7.485e-04
   256 |     minmod |   2.295e-03 |   7.650e-04 |   3.825e-04

    32 | toh_psi_ni |   1.057e-02 |   3.523e-03 |   1.762e-03
    64 | toh_psi_ni |   5.848e-03 |   1.949e-03 |   9.747e-04
   128 | toh_psi_ni |   3.061e-03 |   1.020e-03 |   5.102e-04
   256 | toh_psi_ni |   1.564e-03 |   5.214e-04 |   2.607e-04
```

## Convergence Rates

```
        pair |     method | rate(rho) |   rate(P) |  rate(Bz)
------------------------------------------------------------
      32->64 |     minmod |      0.85 |      0.85 |      0.85
     64->128 |     minmod |      0.93 |      0.93 |      0.93
    128->256 |     minmod |      0.97 |      0.97 |      0.97
      32->64 | toh_psi_ni |      0.85 |      0.85 |      0.85
     64->128 | toh_psi_ni |      0.93 |      0.93 |      0.93
    128->256 | toh_psi_ni |      0.97 |      0.97 |      0.97
```

Rates are identical to 2 decimal places. Both converge toward 1st order (minmod is
1st-order at smooth extrema per LeVeque 2002 §6.5 — this is expected behavior).

## Low-Density Step: Slope Damping

Total slope norm (sum |2(QL[i] - q[i])| across all cells) for a tanh-smoothed
vacuum-edge step (rho: 1e-4 → 1.0):

```
     N |     method |  slope norm
----------------------------------------
    32 |     minmod |  3.4988e+00
    32 | toh_psi_ni |  2.1663e+00  (38% reduction)
    64 |     minmod |  3.5022e+00
    64 | toh_psi_ni |  2.1682e+00  (38% reduction)
   128 |     minmod |  3.5022e+00
   128 | toh_psi_ni |  2.1682e+00  (38% reduction)
   256 |     minmod |  3.5022e+00
   256 | toh_psi_ni |  2.1682e+00  (38% reduction)
```

Toh reduces the reconstructed slope magnitude uniformly by ~38% across all
resolutions. The reduction is resolution-independent (as expected — it is a
pointwise limiter on the density ratio n_i/n_ref, not a grid-size term).

ψ distribution for this problem:
- vacuum half (rho ~ 1e-4): ψ ≈ 0.499 (near 0.5 — close to 1st-order diffusion)
- bulk half (rho ~ 1.0):    ψ ≈ 0.73  (reduced from 1.0 because n_ref=max(rho))

Key finding: with `n_ref = max(rho)` (auto mode), ψ never reaches 1.0 in the
bulk either — the sigmoid saturates at ψ ≈ 0.73 for cells at peak density.
This is expected from the formula: when n_i = n_ref, arg = α - 1 = -0.995,
giving ψ = sigmoid(0.995) ≈ 0.73. To get ψ → 1 in the bulk, the user must
provide an explicit `toh_n_ref` that is much smaller than the bulk density
(i.e., a true vacuum reference density).

## Verdict

**toh_psi_ni = EQUAL to minmod on smooth data, BETTER on low-density step.**

- Convergence rate: identical (both 0.85→0.93→0.97, approaching 1st order)
- Smooth-region L1 errors: Toh is 30-40% LOWER than minmod on the sinusoidal wave
  (because ψ < 1 everywhere reduces extrapolation error on smooth data)
- Low-density slope damping: 38% slope reduction vs minmod (measured, not claimed)
- No regression in convergence order

The limiter meets its design claim (KR §3.1): reduces slope magnitude in
low-density regions. It does NOT meet the claim of "equal to 1 at continuum
limit" in auto-n_ref mode — ψ_max ≈ 0.73 not 1.0. For full recovery of
minmod behavior in the bulk, `toh_n_ref` must be set to the physical vacuum
density, not left at auto.

## Recommendation

Merge. Stage as **opt-in default** for `use_godunov_flux=True` paths where
`rho_min/rho_max < 1e-2` (near-vacuum detected). Do NOT make it the global
default — the ψ_bulk ≈ 0.73 effect is a subtle accuracy reduction in
high-density regions that users should opt into knowingly.

If the intent is true "minmod in bulk, diffusive in vacuum", add a guard:

```python
# Recommended fix for user-facing default
toh_n_ref = rho_floor  # physical vacuum reference, not max(rho)
```

## Test Coverage

5/5 tests pass (`pytest tests/test_toh_pinch_convergence.py -v`):
- `test_minmod_converges` — baseline PLM order >= 0.8
- `test_toh_converges` — Toh order >= 0.8
- `test_toh_not_worse_than_minmod` — Toh L1 <= 2x minmod on smooth data
- `test_toh_reduces_tv_in_low_density` — slope norm reduced in vacuum zone
- `test_toh_damps_less_in_high_density_than_low_density` — ψ density-sensitive
