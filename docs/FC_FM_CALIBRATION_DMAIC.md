# Six Sigma DMAIC: fc/fm Calibration Sweeps for MLX MHD Solver

**Date**: 2026-03-25
**Author**: dpf-mhd-physicist
**Status**: Research complete, implementation ready
**Estimated Implementation**: 4-6 hours
**Estimated Compute**: 6-18 hours for full PF-1000 sweep (M3 Pro)

---

## 1. DEFINE

### 1.1 Problem Statement

The MLX cylindrical MHD solver completes full PF-1000 discharges (19,186 steps,
1.807 MA, 3.4% I_peak error) using published Lee & Saw (2014) parameters
fc=0.7, fm=0.08. These parameters were calibrated for the *Lee circuit ODE model*,
not for a spatially-resolved MHD solver. The MHD solver resolves physics that the
Lee model lumps into fc/fm (sheath structure, current redistribution, magnetic
pressure profiles), so the optimal fc/fm for MHD will differ from the Lee values.

**Goal**: Find (fc, fm) that minimize waveform NRMSE when the *MLX MHD solver*
(not the Lee ODE) is used as the forward model.

### 1.2 Objective Function

Use a weighted composite matching the existing `LeeModelCalibrator`:

```
J(fc, fm) = w_peak * |I_peak_sim - I_peak_exp| / I_peak_exp
           + w_timing * |t_peak_sim - t_peak_exp| / t_peak_exp
           + w_nrmse * NRMSE(I_sim(t), I_exp(t))
```

Weights: `w_peak=0.4, w_timing=0.3, w_nrmse=0.3` (existing defaults).

**Rationale**: Three metrics vs two parameters gives the system 1+ DOF for
overdetermined fitting, avoiding parameter degeneracy. NRMSE captures waveform
*shape*, not just scalars.

### 1.3 Parameter Bounds

From Lee & Saw (2014) Table 1 and `_PUBLISHED_FC_FM_RANGES` in
`src/dpf/validation/_calibration_data.py`:

| Parameter | Published Range (PF-1000) | Sweep Range | Justification |
|-----------|--------------------------|-------------|---------------|
| fc | 0.6 -- 0.8 | 0.50 -- 0.85 | Wider by 0.1 each side for MHD exploration |
| fm | 0.05 -- 0.20 | 0.03 -- 0.30 | MHD resolves sheath mass loading differently |

### 1.4 Compute Budget

| Configuration | Time/eval | Grid | Steps |
|---------------|-----------|------|-------|
| Coarse (32x1x64) | ~140s | 2048 cells | ~19K |
| Medium (48x1x96) | ~350s | 4608 cells | ~28K |
| Fine (64x1x128) | ~600s | 8192 cells | ~38K |

Budget allocation (M3 Pro, single GPU):
- Grid sweep (coarse, 5x5 = 25 evals): 25 * 140s = **58 min**
- Nelder-Mead refinement (coarse, ~40 evals): 40 * 140s = **93 min**
- Top-3 candidates at medium grid (3 evals): 3 * 350s = **17 min**
- Winner at fine grid (1 eval): 1 * 600s = **10 min**
- **Total: ~3 hours** for a single device

### 1.5 Success Criteria (CTQ - Critical to Quality)

| Metric | Target | Stretch |
|--------|--------|---------|
| Waveform NRMSE (full) | < 0.10 (10%) | < 0.07 |
| I_peak error | < 5% | < 3% |
| t_peak error | < 10% | < 5% |
| Current dip captured | Qualitative | Depth within 50% of exp |

### 1.6 Out of Scope

- Calibrating fmr (radial mass fraction) and fcr (radial current fraction) --
  these affect radial phase only, which is a 2nd-order correction.
- Multi-device calibration -- start with PF-1000 (best experimental data).
- Neutron yield optimization (Yn is dominated by fill pressure per Sobol
  analysis, ST=0.90; fc/fm contribute ST<0.08).

---

## 2. MEASURE

### 2.1 Existing Infrastructure Audit

**`src/dpf/validation/_calibration_core.py`** -- `LeeModelCalibrator`
- Nelder-Mead via `scipy.optimize.minimize` with bounds
- Objective: weighted sum of peak error + timing error + NRMSE
- Forward model: `LeeModel.compare_with_experiment()` (circuit ODE, ~0.1s/eval)
- Convergence: `xatol=0.005, fatol=0.001`
- Failure penalty: returns 10.0

**`src/dpf/validation/_calibration_data.py`**
- `_PUBLISHED_FC_FM_RANGES`: 7 devices with published (fc, fm) bounds
- `_DEFAULT_DEVICE_PCF`: pinch column fraction per device
- `_DEFAULT_CROWBAR_R`: crowbar resistance per device
- `CalibrationResult` dataclass

**`app_calibrate.py`** -- CLI auto-calibration
- Calls `LeeModelCalibrator.calibrate()` with fc_bounds=(0.4, 0.80), fm_bounds=(0.02, 0.6)
- Compares against published Lee params
- No MLX/MHD path exists

**`scripts/validate_waveform_nrmse.py`** -- NRMSE harness
- Uses `run_simulation_core()` (Lee snowplow + circuit, NOT MHD solver)
- Computes: full NRMSE, rise-phase NRMSE, dI/dt metrics, dip analysis
- Has device-to-preset mapping

**`tests/test_mlx_pf1000.py`** -- MLX full-discharge test
- Uses `SimulationEngine` with `backend="mlx"` preset
- Grid: 32x1x64, 12 us, ~20K steps
- Validates: no negative pressure, I_peak within 10%, mass/energy conservation
- Uses fc=0.7, fm=0.08 (hardcoded in pf1000 preset)

**`src/dpf/engine/core.py`** -- `SimulationEngine`
- MLX backend wired at line 182-208
- Step loop returns `StepResult` with current, voltage, coupling state
- State dict has `rho`, `velocity`, `pressure`, `B`, `Te`, `Ti`

### 2.2 Gap Analysis

| Component | Exists? | Gap |
|-----------|---------|-----|
| MLX forward model | Yes | Not wired into calibration objective |
| Objective function | Yes (Lee ODE) | Needs MLX variant |
| Optimizer | Yes (Nelder-Mead) | Should add Optuna/TPE for efficiency |
| Grid sweep | No | Need structured (fc, fm) grid evaluation |
| Multi-fidelity | No | Need coarse->fine grid refinement |
| Waveform comparison | Yes (`nrmse_peak`) | Reusable as-is |
| fc/fm injection | Partial | Preset overrides exist; need to propagate to engine |

### 2.3 Data Flow for MLX Calibration

```
(fc, fm) --> preset["snowplow"]["current_fraction"] = fc
             preset["snowplow"]["mass_fraction"] = fm
         --> SimulationConfig(**preset)
         --> SimulationEngine(config)
         --> engine.step() x N  (MLX MHD solver)
         --> extract I(t) from engine.circuit.current per step
         --> nrmse_peak(I_sim, I_exp)
         --> J(fc, fm)
```

The snowplow model inside the engine uses fc and fm to compute:
- Swept mass: `M_swept = fm * rho0 * pi * (b^2 - a^2) * z_sheath`
- Effective current: `I_eff = fc * I_circuit`
- These feed into Lp and dL/dt which drive the circuit-plasma coupling

---

## 3. ANALYZE

### 3.1 Literature Review (2024-2026)

**Multi-fidelity Bayesian optimization for plasma simulations**

Luo et al. (2026, Phys. Plasmas 33:012702) demonstrate automated simulation-based
design via multi-fidelity active learning for laser direct drive implosions. Their
approach uses neural network ensembles with transfer learning to bridge 1D (cheap)
and 2D (expensive) simulations. Key insight: a GP surrogate built on 50-100
low-fidelity evaluations can guide 10-20 high-fidelity evaluations to find optima
that would otherwise require 500+ high-fidelity runs.

Smaeyama et al. (2024, arXiv:2404.11965) apply multi-fidelity Gaussian process
regression to plasma microturbulence, demonstrating that fusing data from multiple
simulation fidelities improves prediction accuracy by 3-5x over single-fidelity GP
at equal computational cost.

Ferran Pousa et al. (2023, Phys. Rev. Accel. Beams 26:084601; updated 2024)
show Bayesian optimization of laser-plasma accelerators assisted by reduced
physical models achieves an order-of-magnitude speedup by using cheap Wake-T
evaluations to guide expensive FBPIC PIC simulations.

**Tokamak control via multi-scale BO**

Char et al. (2025, arXiv:2506.10287) present multi-timescale Bayesian optimization
for plasma stabilization in DIII-D, integrating data-driven dynamics with GP
surrogates and rapidly adapting between shots. Validated on tearing instability
control.

**DPF-specific calibration**

Auluck (2022, arXiv:2211.16775) develops a kinematic DPF framework with
propagation delay and nonzero sheath thickness. This provides analytical
predictions for sheath dynamics that could serve as a cross-check for MHD
calibration results.

Gratton & Vargas (2014, arXiv:1407.8271) demonstrate global parameter
optimization of Mather-type DPF using the 2D snowplow model, establishing
that dynamic inductance from first principles can replace empirical fitting.

No published work exists on MHD-based (as opposed to circuit-ODE-based)
calibration of DPF fc/fm parameters. This would be a novel contribution.

### 3.2 Parameter Degeneracy Analysis

From the existing Sobol sensitivity analysis (documented in MEMORY.md):
- Neutron yield Yn: dominated by fill pressure (ST=0.90), fc/fm barely matter (ST<0.08)
- I_peak: primarily set by circuit parameters (C, V0, L0, R0)
- Waveform shape (timing, dip depth): sensitive to fc and fm

**Degeneracy structure**: fc and fm are partially degenerate for I_peak
(increasing fc while decreasing fm can produce similar peak currents). The
waveform NRMSE breaks this degeneracy because:
- fc affects the rise rate (how much circuit current couples to plasma)
- fm affects the timing (how much mass loads the sheath, slowing it)
- Different (fc, fm) combinations produce different rise shapes even if I_peak matches

This is why the 3-metric objective (peak + timing + NRMSE) with 2 parameters
is critical -- it provides 1 DOF of overdetermination.

### 3.3 Optimizer Selection

| Optimizer | Evals to converge | Handles noise? | Parallel? | Library |
|-----------|-------------------|----------------|-----------|---------|
| Nelder-Mead | 40-80 | Poor | No | scipy |
| TPE (Optuna) | 30-60 | Yes | Yes | optuna |
| GP-BO (Optuna) | 20-40 | Yes | Yes | optuna |
| CMA-ES | 50-100 | Good | No | scipy/cmaes |
| Grid search | N_fc * N_fm | N/A | Yes | manual |

**Recommendation**: Optuna TPE (Tree-structured Parzen Estimator).

Rationale:
1. **Noise tolerance**: MHD simulations have float32 stochastic noise from GPU
   reductions. TPE handles this natively.
2. **Evaluation efficiency**: TPE typically converges in 30-60 evaluations vs
   40-80 for Nelder-Mead, and each evaluation costs 140-600s.
3. **Pruning**: Optuna's median pruning can abort unpromising trials early by
   monitoring I(t) during the simulation and pruning if the rise rate is
   clearly wrong (saves ~40% compute on bad trials).
4. **Visualization**: Optuna provides built-in parameter importance, contour
   plots, and optimization history -- all useful for diagnosing degeneracy.
5. **Already a dependency**: Optuna is in pyproject.toml (used by WALRUS).

**Fallback**: If Optuna is not available, use scipy Nelder-Mead (already
implemented) with the MLX forward model swapped in.

### 3.4 Multi-Fidelity Strategy

Inspired by Ferran Pousa et al. (2023/2024) and Luo et al. (2026):

```
Phase 1: COARSE GRID SWEEP (32x1x64, ~140s/eval)
  - 5x5 structured grid: fc in [0.50, 0.575, 0.65, 0.725, 0.80]
                          fm in [0.03, 0.10, 0.17, 0.24, 0.30]
  - 25 evaluations, ~58 min
  - Purpose: map the objective landscape, identify promising region

Phase 2: COARSE GRID OPTIMIZATION (32x1x64, ~140s/eval)
  - Optuna TPE within narrowed bounds from Phase 1
  - 40 evaluations, ~93 min
  - Purpose: find approximate optimum at coarse resolution

Phase 3: MEDIUM GRID VERIFICATION (48x1x96, ~350s/eval)
  - Top 3 candidates from Phase 2
  - 3 evaluations, ~17 min
  - Purpose: verify ranking holds at higher resolution

Phase 4: FINE GRID VALIDATION (64x1x128, ~600s/eval)
  - Winner from Phase 3
  - 1 evaluation, ~10 min
  - Purpose: production-quality NRMSE number
```

Total: ~68 evaluations, ~3 hours on M3 Pro.

If Phase 3 reranks candidates (coarse optimum != medium optimum), run 5 more
Optuna trials at medium resolution to refine. This adds ~30 min.

---

## 4. IMPROVE

### 4.1 Implementation Plan

#### 4.1.1 New File: `src/dpf/validation/mlx_calibration.py`

```python
"""MLX MHD-based calibration of fc/fm parameters.

Unlike LeeModelCalibrator (circuit ODE forward model), this uses the full
MLX cylindrical MHD solver as the forward model. Each evaluation runs a
complete PF-1000 discharge (~19K steps, ~140s at 32x1x64).
"""

from __future__ import annotations

import logging
import time as wall_time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MLXCalibrationResult:
    best_fc: float
    best_fm: float
    nrmse_full: float
    nrmse_rise: float
    peak_current_error: float
    timing_error: float
    objective_value: float
    n_evals: int
    grid_shape: tuple[int, int, int]
    wall_time_s: float
    all_trials: list[dict[str, Any]]


def mlx_objective(
    fc: float,
    fm: float,
    device_name: str = "PF-1000",
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    sim_time_us: float = 12.0,
    peak_weight: float = 0.4,
    timing_weight: float = 0.3,
    nrmse_weight: float = 0.3,
) -> dict[str, float]:
    """Run MLX MHD solver with given (fc, fm) and return metrics.

    Returns dict with keys: objective, nrmse_full, nrmse_rise,
    peak_error, timing_error, I_peak_MA, t_peak_us, wall_time_s.
    """
    from dpf.config import SimulationConfig
    from dpf.engine import SimulationEngine
    from dpf.presets import get_preset
    from dpf.validation.experimental import DEVICES
    from dpf.validation.experimental_comparison import nrmse_peak

    t0 = wall_time.perf_counter()

    dev = DEVICES[device_name]
    preset = get_preset("pf1000")

    # Inject fc/fm
    preset["snowplow"] = preset.get("snowplow", {})
    preset["snowplow"]["current_fraction"] = fc
    preset["snowplow"]["mass_fraction"] = fm

    # MLX backend config
    preset["fluid"] = {
        "backend": "mlx",
        "riemann_solver": "hll",
        "reconstruction": "plm",
        "time_integrator": "ssp_rk2",
        "precision": "float32",
        "use_ct": False,
    }
    preset["grid_shape"] = list(grid_shape)
    preset["sim_time"] = sim_time_us * 1e-6
    preset["radiation"] = {"bremsstrahlung_enabled": False}
    preset["collision"] = {"enabled": False}

    config = SimulationConfig(**preset)
    engine = SimulationEngine(config)

    times: list[float] = []
    currents: list[float] = []
    max_steps = 25000

    for _ in range(max_steps):
        try:
            engine.step()
        except (RuntimeError, ValueError, FloatingPointError):
            break
        times.append(engine.time)
        currents.append(abs(engine.circuit.current))
        if engine.time >= sim_time_us * 1e-6:
            break

    if len(times) < 100:
        return {"objective": 10.0, "error": "simulation_failed"}

    t_sim = np.array(times)
    I_sim = np.array(currents)

    # Metrics
    I_peak_sim = float(np.max(I_sim))
    idx_peak = int(np.argmax(I_sim))
    t_peak_sim = float(t_sim[idx_peak])

    I_peak_exp = dev.peak_current
    t_peak_exp = dev.current_rise_time

    peak_err = abs(I_peak_sim - I_peak_exp) / I_peak_exp
    timing_err = abs(t_peak_sim - t_peak_exp) / t_peak_exp

    nrmse = nrmse_peak(t_sim, I_sim, dev.waveform_t, dev.waveform_I)
    nrmse_rise_val = nrmse_peak(
        t_sim, I_sim, dev.waveform_t, dev.waveform_I,
        max_time=t_peak_exp,
    )

    obj = peak_weight * peak_err + timing_weight * timing_err + nrmse_weight * nrmse

    elapsed = wall_time.perf_counter() - t0

    return {
        "objective": float(obj),
        "nrmse_full": float(nrmse),
        "nrmse_rise": float(nrmse_rise_val),
        "peak_error": float(peak_err),
        "timing_error": float(timing_err),
        "I_peak_MA": I_peak_sim / 1e6,
        "t_peak_us": t_peak_sim * 1e6,
        "wall_time_s": elapsed,
        "n_steps": len(times),
    }


def run_grid_sweep(
    fc_values: list[float] | None = None,
    fm_values: list[float] | None = None,
    grid_shape: tuple[int, int, int] = (32, 1, 64),
) -> list[dict[str, Any]]:
    """Phase 1: Structured grid sweep over (fc, fm) space."""
    if fc_values is None:
        fc_values = [0.50, 0.575, 0.65, 0.725, 0.80]
    if fm_values is None:
        fm_values = [0.03, 0.10, 0.17, 0.24, 0.30]

    results = []
    total = len(fc_values) * len(fm_values)
    for i, fc in enumerate(fc_values):
        for j, fm in enumerate(fm_values):
            idx = i * len(fm_values) + j + 1
            logger.info("Grid sweep %d/%d: fc=%.3f, fm=%.3f", idx, total, fc, fm)
            r = mlx_objective(fc, fm, grid_shape=grid_shape)
            r["fc"] = fc
            r["fm"] = fm
            results.append(r)
            logger.info(
                "  -> obj=%.4f, NRMSE=%.3f, I_pk_err=%.1f%%, wall=%.0fs",
                r["objective"], r.get("nrmse_full", -1),
                r.get("peak_error", -1) * 100, r.get("wall_time_s", 0),
            )
    return results


def run_optuna_optimization(
    fc_bounds: tuple[float, float] = (0.50, 0.85),
    fm_bounds: tuple[float, float] = (0.03, 0.30),
    n_trials: int = 40,
    grid_shape: tuple[int, int, int] = (32, 1, 64),
    seed: int = 42,
) -> MLXCalibrationResult:
    """Phase 2: Optuna TPE optimization."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_trials: list[dict[str, Any]] = []
    t0 = wall_time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        fc = trial.suggest_float("fc", fc_bounds[0], fc_bounds[1])
        fm = trial.suggest_float("fm", fm_bounds[0], fm_bounds[1])
        r = mlx_objective(fc, fm, grid_shape=grid_shape)
        r["fc"] = fc
        r["fm"] = fm
        all_trials.append(r)
        return r["objective"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_fc = best.params["fc"]
    best_fm = best.params["fm"]

    # Re-evaluate best to get full metrics
    best_metrics = mlx_objective(best_fc, best_fm, grid_shape=grid_shape)

    elapsed = wall_time.perf_counter() - t0

    return MLXCalibrationResult(
        best_fc=best_fc,
        best_fm=best_fm,
        nrmse_full=best_metrics["nrmse_full"],
        nrmse_rise=best_metrics["nrmse_rise"],
        peak_current_error=best_metrics["peak_error"],
        timing_error=best_metrics["timing_error"],
        objective_value=best_metrics["objective"],
        n_evals=n_trials + 1,
        grid_shape=grid_shape,
        wall_time_s=elapsed,
        all_trials=all_trials,
    )


def run_full_calibration_pipeline(
    device_name: str = "PF-1000",
) -> dict[str, Any]:
    """Run the complete 4-phase multi-fidelity calibration pipeline."""
    logger.info("=== Phase 1: Coarse grid sweep (32x1x64) ===")
    grid_results = run_grid_sweep(grid_shape=(32, 1, 64))

    # Find promising region from grid sweep
    valid = [r for r in grid_results if r.get("objective", 10) < 5.0]
    if not valid:
        valid = sorted(grid_results, key=lambda r: r.get("objective", 10))[:5]

    fc_vals = [r["fc"] for r in valid]
    fm_vals = [r["fm"] for r in valid]
    fc_lo = max(0.3, min(fc_vals) - 0.1)
    fc_hi = min(0.95, max(fc_vals) + 0.1)
    fm_lo = max(0.01, min(fm_vals) - 0.05)
    fm_hi = min(0.50, max(fm_vals) + 0.05)

    logger.info("=== Phase 2: Optuna optimization (32x1x64) ===")
    logger.info("Narrowed bounds: fc=[%.2f, %.2f], fm=[%.2f, %.2f]",
                fc_lo, fc_hi, fm_lo, fm_hi)
    optuna_result = run_optuna_optimization(
        fc_bounds=(fc_lo, fc_hi),
        fm_bounds=(fm_lo, fm_hi),
        n_trials=40,
        grid_shape=(32, 1, 64),
    )

    # Top 3 candidates for medium-grid verification
    all_sorted = sorted(
        optuna_result.all_trials,
        key=lambda r: r.get("objective", 10),
    )[:3]

    logger.info("=== Phase 3: Medium grid verification (48x1x96) ===")
    medium_results = []
    for r in all_sorted:
        m = mlx_objective(r["fc"], r["fm"], grid_shape=(48, 1, 96))
        m["fc"] = r["fc"]
        m["fm"] = r["fm"]
        medium_results.append(m)
        logger.info("  fc=%.3f fm=%.3f -> obj=%.4f NRMSE=%.3f",
                     r["fc"], r["fm"], m["objective"], m["nrmse_full"])

    best_medium = min(medium_results, key=lambda r: r["objective"])

    logger.info("=== Phase 4: Fine grid validation (64x1x128) ===")
    fine = mlx_objective(
        best_medium["fc"], best_medium["fm"],
        grid_shape=(64, 1, 128),
    )

    return {
        "phase1_grid": grid_results,
        "phase2_optuna": {
            "best_fc": optuna_result.best_fc,
            "best_fm": optuna_result.best_fm,
            "n_evals": optuna_result.n_evals,
            "objective": optuna_result.objective_value,
        },
        "phase3_medium": medium_results,
        "phase4_fine": {
            "fc": best_medium["fc"],
            "fm": best_medium["fm"],
            **fine,
        },
        "final_fc": best_medium["fc"],
        "final_fm": best_medium["fm"],
        "final_nrmse": fine["nrmse_full"],
        "final_I_peak_err": fine["peak_error"],
    }
```

#### 4.1.2 CLI Entry Point Addition to `app_calibrate.py`

Add a function `auto_calibrate_mlx()` that calls `run_full_calibration_pipeline()`.

#### 4.1.3 Key Wiring Detail: fc/fm Propagation

The fc and fm parameters reach the plasma through the snowplow model in the
engine step loop. In `engine/core.py` lines 367-379, the snowplow model uses:
- `mass_fraction` (fm) to compute swept mass
- `current_fraction` (fc) to compute effective current I_eff = fc * I_circuit

These propagate to Lp and dL/dt, which drive the circuit response. The MHD
solver itself does not use fc/fm directly -- they modify the circuit-plasma
coupling. This means the calibration is tuning how much of the circuit current
and gas mass participates in the MHD dynamics.

#### 4.1.4 Optuna Early Pruning (Compute Savings)

```python
def objective_with_pruning(trial: optuna.Trial) -> float:
    fc = trial.suggest_float("fc", *fc_bounds)
    fm = trial.suggest_float("fm", *fm_bounds)

    # Run simulation with step-by-step monitoring
    engine = create_engine(fc, fm, grid_shape)
    for step_idx in range(max_steps):
        engine.step()
        if step_idx % 1000 == 0 and step_idx > 2000:
            # Compute intermediate I_peak error
            I_so_far = abs(engine.circuit.current)
            # If current is way off expected trajectory, prune
            trial.report(intermediate_metric, step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return final_objective
```

This can save ~40% of compute on clearly bad (fc, fm) combinations by aborting
simulations where the current trace diverges early from the experimental waveform.

### 4.2 Implementation Sequence

| Step | File | Change | Time |
|------|------|--------|------|
| 1 | `src/dpf/validation/mlx_calibration.py` | New file: `mlx_objective`, `run_grid_sweep`, `run_optuna_optimization`, `run_full_calibration_pipeline` | 2h |
| 2 | `app_calibrate.py` | Add `auto_calibrate_mlx()` CLI entry point | 30m |
| 3 | `tests/test_mlx_calibration.py` | Unit tests: objective function (mocked engine), grid sweep structure, Optuna integration | 1h |
| 4 | `scripts/run_mlx_calibration.py` | Standalone script for unattended execution | 30m |
| 5 | Run Phase 1-4 | Execute full pipeline on M3 Pro | 3-6h (compute) |
| 6 | Analysis | Generate contour plots, compare vs Lee-model fc/fm | 1h |

### 4.3 Fallback: Scipy Nelder-Mead (No Optuna)

If Optuna is unavailable, the same pipeline works with
`scipy.optimize.minimize(method="nelder-mead")` using the existing
`LeeModelCalibrator` pattern but with `mlx_objective` as the forward model.

### 4.4 Expected Results

Based on the current 3.4% I_peak error with published fc=0.7, fm=0.08:
- The MHD solver resolves sheath structure that the Lee model lumps into fm.
  Expect optimal fm_MHD < fm_Lee (0.08) because the MHD solver captures mass
  loading that fm is a proxy for.
- fc should stay near 0.7 because current fraction is a genuine circuit
  property (skin depth, contact resistance) that the MHD solver cannot replace.
- Predicted improvement: NRMSE from ~0.14 (current) to <0.10 (target).

---

## 5. CONTROL

### 5.1 Validation Criteria

| Check | Method | Pass |
|-------|--------|------|
| NRMSE < 0.10 | `nrmse_peak()` at fine grid | Hard gate |
| I_peak within 5% | `abs(I_sim - I_exp) / I_exp` | Hard gate |
| fc within published range | Compare vs Lee & Saw (2014) 0.6-0.8 | Soft (warning if outside) |
| fm within published range | Compare vs Lee & Saw (2014) 0.05-0.20 | Soft (MHD may differ) |
| No negative pressure | `min(pressure) > 0` throughout | Hard gate |
| Mass conservation < 5% | `abs(M_final - M_init) / M_init` | Hard gate |

### 5.2 Overfitting Prevention

1. **Cross-device validation**: After calibrating on PF-1000 (Scholz 27 kV),
   run the same (fc, fm) on PF-1000-Gribkov (independent shot, same device)
   and PF-1000-16kV (different operating point). If NRMSE degrades by >50%,
   the calibration overfit to Scholz waveform details.

2. **Lee-model consistency**: The calibrated MHD fc/fm should be within ~20%
   of Lee-model fc/fm. If they differ by more, it indicates the MHD solver
   is compensating for a physics bug (e.g., wrong Lp formula, missing back-EMF).

3. **Grid convergence**: The optimal (fc, fm) at 32x1x64 vs 64x1x128 should
   agree within 0.05 for fc and 0.03 for fm. Larger differences indicate the
   coarse grid is not resolving the relevant physics.

### 5.3 Monitoring and Maintenance

- Store calibration results in `docs/calibration_results/` as JSON
- Track NRMSE regression: if a solver change worsens NRMSE by >10%, re-run calibration
- Re-calibrate after any change to: Lp formula, back-EMF wiring, snowplow model,
  electrode BCs, or circuit solver

### 5.4 Known Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| fc/fm optimal values are grid-dependent | Calibration not transferable | Grid convergence study in Phase 3-4 |
| Simulation crashes for extreme (fc, fm) | Optimizer gets stuck | Penalty return (obj=10.0) for failed runs |
| Optuna finds local minimum | Suboptimal result | Grid sweep in Phase 1 maps global landscape |
| Compensating errors | Calibrated values mask bugs | Lee-model consistency check + cross-device validation |
| Compute exceeds budget | Incomplete calibration | Early pruning saves ~40%; can reduce grid sweep to 3x3 |

---

## Appendix A: Literature References (2024-2026 Only)

1. Luo et al., "Automated simulation-based design via multi-fidelity active
   learning and optimization for laser direct drive implosions," Phys. Plasmas
   33, 012702 (2026). Multi-fidelity NN surrogate + Bayesian optimization for
   plasma simulation calibration.

2. Smaeyama et al., "Multi-fidelity Gaussian process surrogate modeling for
   regression problems in physics," arXiv:2404.11965 (2024). GP regression
   fusing multiple simulation fidelities for plasma microturbulence.

3. Ferran Pousa et al., "Bayesian optimization of laser-plasma accelerators
   assisted by reduced physical models," Phys. Rev. Accel. Beams 26, 084601
   (2023; updated 2024). Multi-fidelity BO: cheap reduced model guides
   expensive PIC simulations, 10x speedup.

4. Char et al., "Multi-Timescale Dynamics Model Bayesian Optimization for
   Plasma Stabilization in Tokamaks," arXiv:2506.10287 (2025). Multi-scale
   BO integrating data-driven dynamics with GP for tokamak control.

5. Auluck, "First steps towards a theory of the Dense Plasma Focus: Part I,"
   arXiv:2211.16775 (2022; cited in 2024 DPF reviews). Kinematic DPF
   framework with propagation delay for cross-validation.

6. Nature Scientific Reports, "Double 3 MJ dense plasma focus for
   thermonuclear drive inertial confinement fusion," (2025). Recent DPF
   design optimization context.

---

## Appendix B: File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `src/dpf/validation/mlx_calibration.py` | TO CREATE | MLX calibration pipeline |
| `src/dpf/validation/_calibration_core.py` | EXISTS | Lee ODE calibrator (reference) |
| `src/dpf/validation/_calibration_data.py` | EXISTS | Published fc/fm ranges |
| `src/dpf/validation/experimental_waveforms.py` | EXISTS | 7 devices, digitized I(t) |
| `src/dpf/validation/experimental_comparison.py` | EXISTS | `nrmse_peak()` function |
| `src/dpf/engine/core.py` | EXISTS | Engine with MLX backend wired |
| `src/dpf/metal/mlx_solver.py` | EXISTS | MLX MHD solver |
| `tests/test_mlx_pf1000.py` | EXISTS | Full-discharge validation |
| `tests/test_mlx_calibration.py` | TO CREATE | Calibration unit tests |
| `scripts/run_mlx_calibration.py` | TO CREATE | Standalone execution script |
| `app_calibrate.py` | TO MODIFY | Add MLX calibration entry point |
| `docs/calibration_results/` | TO CREATE | Result storage directory |
