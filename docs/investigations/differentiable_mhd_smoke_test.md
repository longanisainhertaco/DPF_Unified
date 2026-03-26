# Differentiable MHD Smoke Test

**Date**: 2026-03-26  
**MLX version**: 0.31.0  
**Grid**: (10, 8, 8) — non-uniform density gradient  
**Test**: `compute_fluxes(dim=0, method='plm')` with sinusoidal rho modulation

## Result

**BREAKTHROUGH — mx.grad works and agrees with finite differences.**

Accurate gradients confirmed for: `hlls, hll`  
AD vs FD relative error < 5% on non-trivial (non-uniform) state.

Gradient-based calibration through the MHD flux is **feasible**.

## Per-Solver Results

| Riemann | AD works? | AD grad | FD grad | Rel error | NaN? | Notes |
|---------|-----------|---------|---------|-----------|------|-------|
| `hlls` | YES | -0.0575781 | -0.0572205 | 6.25e-03 | no | accurate |
| `hll` | YES | -0.0575764 | -0.0572205 | 6.22e-03 | no | accurate |
| `hlls_cpu` | YES | 0 | -0.0572205 | 1.00e+00 | no | disagrees with FD |

## Analysis

### `hlls` and `hll` (pure MLX GPU paths)

Both route through `_get_hlls_compiled(dim)` / `_get_hll_compiled(dim)`,
which wrap `_hlls_flux_gpu` / `_hll_flux_gpu` with `mx.compile()`.
These contain **only pure MLX ops** — no `np.asarray`, no CPU materialisation.
`mx.compile` is compatible with `mx.grad` in MLX >= 0.4: the VJP is traced
through the compiled graph at the same time as the forward pass.

### `hlls_cpu` (CPU numpy fallback)

`hlls_cpu` routes to `_hlls_flux()` which calls `np.asarray(QL)` immediately.
This materialises the MLX lazy array into numpy, breaking the computation graph.
Surprisingly, `mx.grad` may still 'work' here because MLX can treat the
numpy call as a constant (zero gradient through it), not raise an error.
The zero AD gradient confirms the chain is severed at the numpy boundary.

### Gradient magnitude analysis

The test uses `loss = sum(F)` over all flux components. For an 8x8 uniform-
pressure state with sinusoidal density modulation, the flux sum can be near-zero
due to left-right flux cancellation across symmetric interfaces. A near-zero
FD gradient with a non-zero AD gradient indicates float32 noise dominates
the numerical gradient — NOT a sign that AD is wrong.

### Implication for gradient-based calibration

mx.grad propagates through `compute_fluxes` with `riemann='hlls'` or `'hll'`.
This opens the door to:
1. Single-step sensitivity analysis: `d(flux_sum)/d(fc)` at pinch conditions
2. Gradient-based tuning of scalar parameters (fc, fm) via a single-step proxy loss
3. Adjoint-state sensitivity for multi-step integration (requires unrolling or
   checkpointing — not trivial at 20K steps)

**Caveat**: differentiability is through a single RHS call.
Full-discharge gradient optimization requires adjoint ODE solver or
short-horizon sensitivity (last N steps before pinch).

## Next Steps

- [x] Smoke test confirms mx.grad propagates through pure-MLX paths
- [x] Add `test_hlls_is_differentiable` to `tests/test_mlx_riemann.py` — PASSES in 0.28s
- [ ] Prototype single-step sensitivity: `d(sum(F))/d(rho)` with non-zero FD grad
- [ ] Investigate short-horizon (last 50 steps) gradient for fc/fm sensitivity
- [ ] Evaluate vs Optuna TPE: are gradient directions informative for calibration?
