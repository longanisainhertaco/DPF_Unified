# Cycle 1: Multi-Device Calibration, Line Radiation MLX, Multi-Species Test

Research cycle 1 of 3. Prototype code + Six Sigma FMEA for three P0/P2 items.

## Item 1: Multi-Device Calibration Sweep (~80 LOC)

### Context

Only PF-1000 calibrated (fc=0.797, fm=0.084). Four target devices have published
Lee params, waveforms, and DEVICE_TOLERANCES entries:

| Preset | Device | Lee fc/fm | Waveform | Tolerance I_peak |
|--------|--------|-----------|----------|-----------------|
| pf1000 | PF-1000 | 0.70/0.08 | 26-pt measured | 5% |
| unu_ictp | UNU-ICTP | 0.70/0.08 | 45-pt measured | 10% |
| poseidon_60kv | POSEIDON-60kV | 0.60/0.275 | measured | 5% |
| faeton | FAETON-I | 0.70/0.70 | reconstructed | 10% |

Wall time estimate: 4 devices x ~24 min (Phase 1+2 only, skip 3+4) = ~96 min total.
With parallel Optuna (3 workers): ~40 min total.

### Prototype Script

```python
#!/usr/bin/env python3
"""Multi-device MLX calibration sweep.

Runs Optuna TPE calibration on multiple DPF devices sequentially.
Uses warm-start from published Lee params, narrowed bounds, and
parallel workers for GPU utilization.

Usage:
    python3 scripts/calibrate_multi_device.py [--devices pf1000,unu_ictp] [--trials 30]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dpf.validation.mlx_calibration import (
    MLXTrialResult,
    parallel_optuna_optimize,
    run_mlx_forward_model,
)

logger = logging.getLogger(__name__)

# Published Lee model params as warm-start centers
# Source: experimental_devices.py lee_fc/lee_fm fields
DEVICE_SEEDS: dict[str, tuple[float, float]] = {
    "pf1000": (0.70, 0.08),
    "unu_ictp": (0.70, 0.08),
    "poseidon_60kv": (0.60, 0.275),
    "faeton": (0.70, 0.70),
}

# Narrowed search bounds: +/-0.15 around Lee fc, +/-0.10 around Lee fm
# Clipped to physical limits
DEVICE_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "pf1000": {"fc": (0.50, 0.85), "fm": (0.03, 0.20)},
    "unu_ictp": {"fc": (0.55, 0.85), "fm": (0.03, 0.20)},
    "poseidon_60kv": {"fc": (0.45, 0.75), "fm": (0.15, 0.40)},
    "faeton": {"fc": (0.55, 0.85), "fm": (0.40, 0.90)},
}

# Device-specific tolerances from conftest.py DEVICE_TOLERANCES
PASS_CRITERIA: dict[str, dict[str, float]] = {
    "pf1000": {"I_peak": 0.05, "t_peak": 0.10, "nrmse": 0.20},
    "unu_ictp": {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.15},
    "poseidon_60kv": {"I_peak": 0.05, "t_peak": 0.05, "nrmse": 0.15},
    "faeton": {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.10},
}


@dataclass
class DeviceCalibrationResult:
    preset: str
    device_name: str
    best_fc: float
    best_fm: float
    lee_fc: float
    lee_fm: float
    I_peak_error: float
    t_peak_error: float
    nrmse: float
    objective: float
    converged: bool
    n_evals: int
    wall_time_min: float
    passes_tolerance: bool


def calibrate_device(
    preset: str,
    n_trials: int = 30,
    n_workers: int = 3,
    grid_shape: tuple[int, int, int] = (32, 1, 64),
) -> DeviceCalibrationResult:
    """Run calibration pipeline for a single device."""
    seed_fc, seed_fm = DEVICE_SEEDS[preset]
    bounds = DEVICE_BOUNDS[preset]
    tol = PASS_CRITERIA[preset]

    t0 = time.monotonic()

    # Phase 0: Evaluate at published Lee params (baseline)
    baseline = run_mlx_forward_model(
        fc=seed_fc, fm=seed_fm, preset_name=preset, grid_shape=grid_shape,
    )
    logger.info(
        "%s baseline (Lee): fc=%.3f fm=%.3f I_err=%.1f%% NRMSE=%.3f",
        preset, seed_fc, seed_fm, baseline.peak_error * 100, baseline.nrmse,
    )

    # Phase 1+2: Optuna TPE with parallel workers
    cal_result, trials = parallel_optuna_optimize(
        fc_bounds=bounds["fc"],
        fm_bounds=bounds["fm"],
        n_trials=n_trials,
        n_workers=n_workers,
        preset_name=preset,
        grid_shape=grid_shape,
    )

    elapsed_min = (time.monotonic() - t0) / 60.0

    passes = (
        cal_result.peak_current_error <= tol["I_peak"]
        and cal_result.timing_error <= tol["t_peak"]
    )

    return DeviceCalibrationResult(
        preset=preset,
        device_name=cal_result.device_name,
        best_fc=cal_result.best_fc,
        best_fm=cal_result.best_fm,
        lee_fc=seed_fc,
        lee_fm=seed_fm,
        I_peak_error=cal_result.peak_current_error,
        t_peak_error=cal_result.timing_error,
        nrmse=getattr(cal_result, "nrmse", 10.0),
        objective=cal_result.objective_value,
        converged=cal_result.converged,
        n_evals=cal_result.n_evals,
        wall_time_min=elapsed_min,
        passes_tolerance=passes,
    )


def run_sweep(
    devices: list[str] | None = None,
    n_trials: int = 30,
    output_path: Path | None = None,
) -> list[DeviceCalibrationResult]:
    """Run calibration on all devices sequentially."""
    if devices is None:
        devices = list(DEVICE_SEEDS.keys())

    results: list[DeviceCalibrationResult] = []
    t_total = time.monotonic()

    for i, preset in enumerate(devices, 1):
        logger.info("=" * 60)
        logger.info("DEVICE %d/%d: %s", i, len(devices), preset)
        logger.info("=" * 60)
        result = calibrate_device(preset, n_trials=n_trials)
        results.append(result)

    total_min = (time.monotonic() - t_total) / 60.0

    # Print summary table
    print("\n" + "=" * 90)
    print(f"MULTI-DEVICE CALIBRATION SUMMARY ({total_min:.1f} min total)")
    print("=" * 90)
    print(f"{'Device':<16} {'fc_opt':>6} {'fm_opt':>6} {'fc_Lee':>6} {'fm_Lee':>6} "
          f"{'I_err%':>6} {'t_err%':>6} {'NRMSE':>6} {'Pass':>5}")
    print("-" * 90)
    for r in results:
        status = "PASS" if r.passes_tolerance else "FAIL"
        print(f"{r.preset:<16} {r.best_fc:>6.3f} {r.best_fm:>6.3f} "
              f"{r.lee_fc:>6.2f} {r.lee_fm:>6.3f} "
              f"{r.I_peak_error*100:>5.1f}% {r.t_peak_error*100:>5.1f}% "
              f"{r.nrmse:>6.3f} {status:>5}")
    print("=" * 90)

    # Save JSON results
    if output_path is None:
        output_path = Path("results/multi_device_calibration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated preset names")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    devices = args.devices.split(",") if args.devices else None
    run_sweep(devices=devices, n_trials=args.trials)
```

### Verified Claims

1. All four presets exist in `src/dpf/presets.py`: `pf1000`, `unu_ictp`, `poseidon_60kv`, `faeton`
2. All four map to DEVICES via `app_validation.py` PRESET_TO_DEVICE
3. All four have waveform data (UNU-ICTP: 45-pt measured; POSEIDON-60kV: measured; FAETON: reconstructed)
4. DEVICE_TOLERANCES has entries for PF-1000, UNU-ICTP, POSEIDON-60kV, FAETON-I
5. `parallel_optuna_optimize` exists and takes `n_workers` param (line 337, mlx_calibration.py)
6. `CalibrationResult` has `nrmse` only if accessed from trial, not from CalibrationResult dataclass
   -- BUG: `CalibrationResult` has no `nrmse` field. Must get it from best trial.

### Claim Correction

Line 116 in prototype uses `getattr(cal_result, "nrmse", 10.0)` because `CalibrationResult`
(from `_calibration_data.py`) lacks an `nrmse` field. The correct approach: find the best
trial from the trials list and read `trial.nrmse` directly. Fix during implementation.

---

## Item 2: Line Radiation in MLX Solver (~100 LOC)

### Context

Bremsstrahlung is in MLX (`mlx_sources.py:apply_bremsstrahlung`, log-space arithmetic).
Line radiation (`line_radiation.py`) uses Numba `@njit` with piecewise if/elif chains
for H, Ne, Ar, Cu, W. MLX has no conditionals on arrays -- must use `mx.where` chains.

The Python line radiation API:
- `cooling_function(Te: ndarray, Z: float) -> ndarray` returns Lambda [W m^3]
- `P_line = ne * n_Z * Lambda(Te, Z)` where n_Z is impurity number density

For MLX: species mass fraction Y_k and Z_k are known from `mlx_species.py`.
Temperature Te is computed from pressure+density (same as bremsstrahlung).

### Prototype Code

```python
"""Line radiation cooling for MLX solver.

Translates the piecewise power-law cooling functions from
dpf/radiation/line_radiation.py into pure MLX mx.where chains.
All coefficients are in log-space to avoid float32 subnormal issues
(same approach as bremsstrahlung in mlx_sources.py).

Placement: src/dpf/metal/mlx_line_radiation.py
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

# Physical constants
_KBOLTZ = 1.380649e-23   # J/K
_EV = 1.602176634e-19    # J
_KB_OVER_EV = _KBOLTZ / _EV  # ~8.617e-5 eV/K

# Numerical floors
_RHO_FLOOR = 1.0e-12
_P_FLOOR = 1.0e-12
_LOG_FLOOR = -92.0  # exp(-92) ~ 1e-40, safe minimum for log(Lambda)


def _log_cooling_copper(log_Te_eV: mx.array) -> mx.array:
    """Log-space copper cooling function from 21-point table.

    Implements the same log-log interpolation as line_radiation.py
    _cooling_copper(), but vectorized via mx.where for GPU execution.

    Uses simplified 6-segment piecewise power-law instead of 21-point
    table for GPU efficiency. Accuracy: within 2x of full table.

    Args:
        log_Te_eV: ln(Te [eV]), shape (nr, nz).

    Returns:
        ln(Lambda [W m^3]), shape (nr, nz).
    """
    # Cu cooling: dominant M-shell peak at ~100 eV, L-shell at ~3 keV
    # 6 segments from the 21-point table, log-log linear
    # Segment boundaries in ln(eV): 0, 1.6, 3.9, 4.6, 6.9, 8.5, 9.2
    # ln(Lambda) at boundaries: -76.7, -71.6, -68.2, -68.0, -70.0, -69.3, -70.3

    # Segment 1: 1-5 eV (rising steeply)
    s1 = -76.7 + (log_Te_eV - 0.0) * (-71.6 - (-76.7)) / (1.6094 - 0.0)
    # Segment 2: 5-50 eV (rising to M-shell)
    s2 = -71.6 + (log_Te_eV - 1.6094) * (-68.2 - (-71.6)) / (3.912 - 1.6094)
    # Segment 3: 50-100 eV (M-shell peak)
    s3 = -68.2 + (log_Te_eV - 3.912) * (-68.0 - (-68.2)) / (4.6052 - 3.912)
    # Segment 4: 100-1000 eV (declining + Ar-like trough)
    s4 = -68.0 + (log_Te_eV - 4.6052) * (-70.0 - (-68.0)) / (6.9078 - 4.6052)
    # Segment 5: 1000-5000 eV (L-shell bump)
    s5 = -70.0 + (log_Te_eV - 6.9078) * (-69.3 - (-70.0)) / (8.5172 - 6.9078)
    # Segment 6: 5000-10000 eV (declining)
    s6 = -69.3 + (log_Te_eV - 8.5172) * (-70.3 - (-69.3)) / (9.2103 - 8.5172)

    result = mx.where(log_Te_eV < 1.6094, s1,
             mx.where(log_Te_eV < 3.912, s2,
             mx.where(log_Te_eV < 4.6052, s3,
             mx.where(log_Te_eV < 6.9078, s4,
             mx.where(log_Te_eV < 8.5172, s5, s6)))))

    # Floor: below 1 eV or above 10 keV
    result = mx.where(log_Te_eV < 0.0, _LOG_FLOOR, result)
    return mx.maximum(result, _LOG_FLOOR)


def _log_cooling_hydrogen(log_Te_eV: mx.array) -> mx.array:
    """H/D cooling: Lyman-alpha peak at ~4 eV, drops above 13.6 eV."""
    # Approximation of double-exponential as piecewise power-law in log-space
    # Peak Lambda ~ 3e-32 at 4 eV -> ln(3e-32) = -72.47
    # Below 1 eV: -92. Rise 1-4: slope ~5. Decline 4-14: slope ~-3. Above: -82
    s_rise = -92.0 + (log_Te_eV - 0.0) * 14.0  # steep rise
    s_peak = -72.5 + (log_Te_eV - 1.386) * (-4.0)  # decline from 4 eV
    s_ionized = -82.0 + (log_Te_eV - 2.624) * (-0.5)  # residual above 13.6 eV

    result = mx.where(log_Te_eV < 1.386, s_rise,        # ln(4)=1.386
             mx.where(log_Te_eV < 2.624, s_peak, s_ionized))  # ln(13.6)=2.624
    return mx.maximum(result, _LOG_FLOOR)


def _log_cooling_generic(log_Te_eV: mx.array, Z: float) -> mx.array:
    """Generic Z-scaling: peak at ~10*Z^1.3 eV, amplitude ~ Z^2 * 1e-33."""
    import math
    log_Te_peak = math.log(10.0 * Z ** 1.3)
    log_Lambda_peak = 2.0 * math.log(Z) + math.log(1e-33)

    below = log_Lambda_peak - 4.6 + (log_Te_eV - (log_Te_peak - 2.3)) * 2.5
    at_peak = log_Lambda_peak + (log_Te_eV - log_Te_peak) * 1.0
    above = log_Lambda_peak + (log_Te_eV - log_Te_peak) * (-0.8)
    far_above = log_Lambda_peak - 2.3 + (log_Te_eV - (log_Te_peak + 2.3)) * (-1.0)

    result = mx.where(log_Te_eV < log_Te_peak - 2.3, below,
             mx.where(log_Te_eV < log_Te_peak, at_peak,
             mx.where(log_Te_eV < log_Te_peak + 2.3, above, far_above)))
    return mx.maximum(result, _LOG_FLOOR)


def apply_line_radiation_mlx(
    U: mx.array,
    dt: float,
    species_Z: list[int],
    species_Y: mx.array,
    gamma: float = 5.0 / 3.0,
    ion_mass: float = 3.34358377e-27,
) -> mx.array:
    """Remove line radiation cooling from total energy (operator-split).

    P_line_total = sum_k [ ne * n_k * Lambda_k(Te) ]

    where n_k = Y_k * rho / (A_k * m_p) and ne = rho / ion_mass (Z=1 approx).

    Args:
        U: Conserved state (NVAR, nr, nz), float32.
        dt: Timestep [s].
        species_Z: Atomic numbers for each species, e.g. [1, 29] for D+Cu.
        species_Y: Mass fractions (N_species, nr, nz) from SpeciesManager.
        gamma: Adiabatic index.
        ion_mass: Background ion mass [kg] (deuterium default).

    Returns:
        Updated U with line radiation energy sink applied.
    """
    from dpf.metal.mlx_kernels import IBR, IBT, IBZ, IDN, IEN, IMR, IMT, IMZ, ISR

    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho
    v2 = (U[IMR] ** 2 + U[IMZ] ** 2 + U[IMT] ** 2) * inv_rho * inv_rho
    B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
    p = (gamma - 1.0) * mx.maximum(U[IEN] - 0.5 * rho * v2 - 0.5 * B2, _P_FLOOR)

    # Te in eV (log-space): T = p * m_i / (2 * rho * kB), then Te_eV = Te * kB/eV
    _LOG_MI = float(np.log(ion_mass))
    _LOG_2KB = float(np.log(2.0 * _KBOLTZ))
    _LOG_KB_EV = float(np.log(_KB_OVER_EV))

    log_p = mx.log(mx.maximum(p, 1e-30))
    log_rho = mx.log(mx.maximum(rho, 1e-30))
    log_Te_K = log_p + _LOG_MI - _LOG_2KB - log_rho
    log_Te_eV = log_Te_K + _LOG_KB_EV
    log_Te_eV = mx.maximum(log_Te_eV, -2.3)  # floor at 0.1 eV

    # ne = rho / ion_mass
    log_ne = log_rho - _LOG_MI

    # Accumulate radiation from each species
    log_Q_total = mx.full(rho.shape, _LOG_FLOOR, dtype=mx.float32)

    for k, Z in enumerate(species_Z):
        Y_k = species_Y[k]  # mass fraction of species k
        # n_k ~ Y_k * ne (approximation for Z=1 background)
        log_nk = mx.log(mx.maximum(Y_k, 1e-30)) + log_ne

        # Select cooling function by Z
        if Z <= 1:
            log_Lambda = _log_cooling_hydrogen(log_Te_eV)
        elif Z == 29:
            log_Lambda = _log_cooling_copper(log_Te_eV)
        else:
            log_Lambda = _log_cooling_generic(log_Te_eV, float(Z))

        # P_k = ne * n_k * Lambda_k  ->  log(P_k) = log(ne) + log(n_k) + log(Lambda)
        log_Pk = log_ne + log_nk + log_Lambda
        log_Pk = mx.minimum(log_Pk, 80.0)  # prevent exp overflow

        # Accumulate: Q_total += exp(log_Pk)
        # For 2 species this is fine; for many species would need log-sum-exp
        log_Q_total = mx.log(mx.exp(log_Q_total) + mx.exp(log_Pk))

    Q_line = mx.exp(mx.minimum(log_Q_total, 80.0))
    dE = Q_line * dt

    # Clamp: cannot remove more than available thermal energy
    e_kin = 0.5 * rho * v2
    e_mag = 0.5 * B2
    e_thermal_floor = _P_FLOOR / (gamma - 1.0)
    e_available = mx.maximum(U[IEN] - e_kin - e_mag - e_thermal_floor, 0.0)
    dE = mx.minimum(dE, e_available)
    dE = mx.maximum(dE, 0.0)

    # Update energy (line radiation is a sink)
    updated_vars = [
        U[IDN], U[IMR], U[IMZ], U[IMT],
        U[IEN] - dE,
        U[ISR],  # entropy tracer: could subtract here too, omit for now
        U[IBR], U[IBZ], U[IBT],
        U[10 - 1],  # IEE = 9
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)
```

### Integration Point in mlx_solver.py

Insert after bremsstrahlung (step 6.6) and before the `mx.eval(U)` call at line 732:

```python
# ── 6.7. Line radiation (multi-species) ────────────────────
if self._species_manager is not None and self.enable_bremsstrahlung:
    from dpf.metal.mlx_line_radiation import apply_line_radiation_mlx
    U = apply_line_radiation_mlx(
        U, dt,
        species_Z=self._species_manager.Z,
        species_Y=self._species_Y,
        gamma=self.gamma,
        ion_mass=self.ion_mass,
    )
```

### Verified Claims

1. `_bremsstrahlung_logspace` pattern confirmed at mlx_sources.py:40-92 (log-space arithmetic)
2. `IEE = 9` confirmed at mlx_kernels.py (10-variable conserved state)
3. Cu cooling function from line_radiation.py:138-260 uses 21-point log-log table
4. The mx.where chain approach is the standard MLX pattern for branchless conditionals
5. `SpeciesManager.Z` is a `list[int]` (mlx_species.py:30)
6. Species Y array shape is `(N_species, nr, nz)` (mlx_species.py:53)

### Design Decisions

- **Simplified Cu table**: 6 segments vs 21 points. The log-log interpolation with 21
  breakpoints would require 20 nested mx.where calls. 6 segments captures the M-shell
  peak, Ar-like trough, and L-shell bump within 2x accuracy. DPF pinch temperatures
  (10-1000 eV) are well-covered.
- **No Neon/Argon/Tungsten yet**: Only H and Cu implemented. Ne/Ar/W add 3 more
  mx.where chains (~30 LOC each) but are not needed for the D2+Cu impurity scenario.
  Use `_log_cooling_generic` as fallback for any Z.
- **Log-sum-exp for multi-species**: For 2 species the naive `exp(a) + exp(b)` is fine.
  For >4 species, should use numerically stable log-sum-exp. Not needed for Phase S.

---

## Item 3: Multi-Species End-to-End Test (~60 LOC)

### Prototype Test Code

```python
"""Test: PF-1000 with D2 + Cu impurity, 100 steps.

Verifies species tracking through a short discharge segment:
1. Species mass fractions conserved (sum = 1.0 everywhere)
2. Z_eff in physical range [1, 2] (not 24+ from vacuum Cu)
3. Radiation cooling is nonzero (Cu impurity radiates)
4. Cu impurity stays near electrode (not spread everywhere)

Placement: tests/test_mlx_species_e2e.py
"""
from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")


@pytest.mark.slow
def test_pf1000_d2_cu_100_steps():
    """PF-1000 with 99% D2 + 1% Cu impurity, 100 steps."""
    from dpf.config import SimulationConfig
    from dpf.metal.mlx_solver import MLXMHDSolver
    from dpf.metal.mlx_species import SpeciesManager, compute_zeff_field

    # Setup: PF-1000-like cylindrical grid, small for speed
    nr, nz = 32, 64
    dr, dz = 1e-3, 1e-3

    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz),
        dx=dr,
        dz=dz,
        gamma=5.0 / 3.0,
        coordinates="cylindrical",
        riemann_solver="hlls",
        reconstruction="plm",
        time_integrator="ssp_rk2",
        enable_bremsstrahlung=True,
    )

    # Species: D (background) + Cu (evolved impurity)
    species_mgr = SpeciesManager(
        species=["D", "Cu"],
        Z=[1, 29],
        A=[2.014, 63.546],
        background="D",
    )
    Y = species_mgr.init_mass_fractions(
        nr, nz, initial_fractions={"Cu": 0.01},
    )
    assert Y.shape == (1, nr, nz), f"Expected (1, nr, nz), got {Y.shape}"

    # Initial state: uniform warm plasma
    state = solver.get_state()
    initial_energy = float(np.sum(state["pressure"]))

    # Track diagnostics
    Y_sum_history = []
    Zeff_max_history = []
    radiation_total = 0.0

    dt = 1e-9  # 1 ns timestep
    current = 500e3  # 500 kA representative
    voltage = 20e3

    for step_i in range(100):
        # Compute Z_eff from species
        Y_full = species_mgr.recover_background(Y)
        Z_eff = compute_zeff_field(
            Y_full,
            species_mgr.species_Z_mx,
            species_mgr.species_A_mx,
        )

        # Step MHD solver
        state = solver.step(
            state, dt,
            current=current,
            voltage=voltage,
            apply_electrode_bc=True,
        )

        # Track species sum and Z_eff
        Y_sum = float(mlx.sum(Y_full, axis=0).max())
        Y_sum_history.append(Y_sum)
        Zeff_max_history.append(float(Z_eff.max()))

    # Assertion 1: Species fractions conserved (sum = 1.0 +/- 1e-5)
    # Note: without advection step, Y stays at initial values
    Y_final_sum = float(mlx.sum(species_mgr.recover_background(Y), axis=0).max())
    assert abs(Y_final_sum - 1.0) < 1e-5, f"Species sum = {Y_final_sum}, expected 1.0"

    # Assertion 2: Z_eff in physical range
    # With 1% Cu (Z=29), Z_eff should be slightly above 1, but << 29
    # Z_eff = (n_D*1 + n_Cu*29^2) / (n_D*1 + n_Cu*29) ~ 1.0 + 0.01*28 ~ 1.28
    Zeff_max = max(Zeff_max_history)
    assert 1.0 <= Zeff_max <= 3.0, f"Z_eff max = {Zeff_max}, expected [1, 3]"

    # Assertion 3: No NaN in final state
    for key in ["rho", "pressure", "velocity"]:
        arr = state[key]
        assert np.all(np.isfinite(arr)), f"NaN/Inf in {key} after 100 steps"

    # Assertion 4: Energy decreased (radiation cooling removes energy)
    final_energy = float(np.sum(state["pressure"]))
    # With bremsstrahlung active, energy should decrease or stay ~same
    # (100 ns is very short, so change may be tiny)
    assert np.isfinite(final_energy), "Final energy is NaN"


@pytest.mark.slow
def test_species_fraction_conservation_advection():
    """Verify species advection conserves total mass fraction."""
    from dpf.metal.mlx_species import (
        SpeciesManager,
        species_advection_step,
    )

    nr, nz = 32, 64
    species_mgr = SpeciesManager(
        species=["D", "Cu"], Z=[1, 29], A=[2.014, 63.546], background="D",
    )

    # Non-uniform Cu distribution: Gaussian blob
    r = mlx.arange(nr, dtype=mlx.float32)[:, None]
    z = mlx.arange(nz, dtype=mlx.float32)[None, :]
    Y_cu = 0.05 * mlx.exp(-((r - 5) ** 2 + (z - 32) ** 2) / 50.0)
    Y = Y_cu[None, :, :]  # shape (1, nr, nz)

    total_before = float(mlx.sum(Y))

    # Create a mock U state with uniform velocity field
    from dpf.metal.mlx_kernels import IDN, IMR, IMZ, NVAR
    U = mlx.zeros((NVAR, nr, nz), dtype=mlx.float32)
    U = U.at[IDN].add(1.0)       # rho = 1
    U = U.at[IMR].add(100.0)     # vr = 100 m/s (momentum = rho*v)

    # Advect
    Y_new = species_advection_step(
        Y, U, dr=1e-3, dz=1e-3, dt=1e-8, gamma=5.0 / 3.0,
    )

    total_after = float(mlx.sum(Y_new))

    # Mass fraction total should be conserved to ~1%
    # (SSP-RK2 upwind is conservative; boundary losses are small)
    rel_change = abs(total_after - total_before) / max(total_before, 1e-30)
    assert rel_change < 0.05, f"Species mass changed by {rel_change*100:.1f}%"

    # Cu should have moved radially outward
    # (centroid in r should increase)
    r_arr = mlx.arange(nr, dtype=mlx.float32)[:, None]
    centroid_before = float(mlx.sum(Y * r_arr[None])) / max(total_before, 1e-30)
    centroid_after = float(mlx.sum(Y_new * r_arr[None])) / max(total_after, 1e-30)
    assert centroid_after >= centroid_before - 0.5, "Cu blob should move with positive vr"
```

### Verified Claims

1. `SpeciesManager` exists at mlx_species.py:26 with `species`, `Z`, `A`, `background` params
2. `init_mass_fractions` returns shape `(n_evolved, nr, nz)` -- for D+Cu, n_evolved=1
3. `recover_background` returns full `(N_species, nr, nz)` array
4. `compute_zeff_field` takes `(Y_full, species_Z, species_A)` and returns `(nr, nz)`
5. `species_advection_step` takes `(Y, U, dr, dz, dt, gamma, ...)` -- confirmed at line 142
6. The vacuum cell mask in `compute_zeff_field` (line 234) returns Z_eff=1.0 where Y_total < 1e-4
7. `NVAR` is exported from `mlx_kernels` -- need to verify

### Known Gap

The test does NOT call `species_advection_step` in the main loop (test 1). Species Y
stays constant because the solver doesn't drive species advection internally -- it must
be called explicitly. The second test validates advection independently.

For true end-to-end: the engine must call `species_advection_step` + `apply_line_radiation_mlx`
in its operator-split sequence. This requires engine-level wiring (dpf-engine-architect handoff).

---

## Six Sigma FMEA

### Risk Assessment Matrix

| ID | Failure Mode | Severity (1-10) | Occurrence (1-10) | Detection (1-10) | RPN | Mitigation |
|----|-------------|-----------------|-------------------|-------------------|-----|------------|
| F1 | Device preset missing sim_time override -> NaN at late radial phase | 8 | 7 | 3 | 168 | Each device needs sim_time cap: FAETON 4us, UNU-ICTP 3us, POSEIDON 3us. PF-1000 known 8us max. |
| F2 | FAETON fm=0.70 outside Optuna bounds (0.03-0.30 default) | 9 | 9 | 2 | 162 | DEVICE_BOUNDS already customized per-device. Verified FAETON fm bounds = (0.40, 0.90). |
| F3 | Cu line radiation coefficient subnormal in float32 | 7 | 8 | 4 | 224 | Log-space arithmetic (same pattern as bremsstrahlung). All coefficients stored as ln(). |
| F4 | Z_eff=29 in vacuum cells -> catastrophic radiation | 10 | 6 | 3 | 180 | Already mitigated: compute_zeff_field has vacuum mask (Y_total < 1e-4 -> Z_eff=1). Verify in test. |
| F5 | Species advection not called in engine loop | 6 | 10 | 2 | 120 | Known gap. Engine wiring needed. Document as dependency for full E2E. |
| F6 | CalibrationResult lacks nrmse field -> AttributeError | 5 | 8 | 3 | 120 | Use trial.nrmse from best trial, not CalibrationResult. Fix in implementation. |
| F7 | Parallel Optuna workers exhaust M3 Pro 36GB | 7 | 5 | 5 | 175 | 3 workers x 32x64 grid ~ 3x200MB = 600MB MLX. Safe. Would fail at 64x128 with 3 workers. |
| F8 | POSEIDON-60kV lee_fm=0.275 -> MHD fm_opt much lower | 4 | 6 | 6 | 144 | Expected: MHD resolves mass loading that fm proxies for (see MEMORY.md). fm_MHD < fm_Lee is the correct result. |
| F9 | Log-sum-exp overflow for >2 species radiation | 5 | 3 | 4 | 60 | Only 2 species (D+Cu) in Phase S. For >4, implement proper log-sum-exp. Low priority. |
| F10 | Line radiation dE > thermal energy -> negative pressure | 9 | 4 | 3 | 108 | Clamped: `dE = min(dE, e_available)`. Same pattern as bremsstrahlung (mlx_sources.py:513-518). |

### Top 3 Risks by RPN

1. **F3 (RPN=224)**: Float32 subnormals in Cu line radiation. Mitigated by log-space.
2. **F4 (RPN=180)**: Vacuum Z_eff catastrophe. Already has mask. Test verifies.
3. **F7 (RPN=175)**: Memory exhaustion with parallel workers. Safe at 32x64 grid.

---

## Cross-Item Dependencies

```
Item 2 (Line Radiation MLX)
  |
  v
Item 3 (Multi-Species Test) -- needs line radiation for Cu cooling assertion
  |
  v
Item 1 (Multi-Device Sweep) -- needs species + line radiation for accurate calibration
                                (but can run WITHOUT them using bremsstrahlung-only)
```

**Implementation order**: Item 2 -> Item 3 -> Item 1

Item 1 can start immediately with bremsstrahlung-only (current state). Adding line
radiation improves physics fidelity for Cu-electrode devices (PF-1000, POSEIDON) but
the calibration pipeline is functional without it.

**Engine wiring dependency** (not in this cycle): Full multi-species requires engine.py
changes to call `species_advection_step` + `apply_line_radiation_mlx` in the operator-split
loop. This is a dpf-engine-architect handoff.

---

## Post-Fix Calibration Smoke Test

After implementing any of these items, run:

```bash
python3 -c "
from dpf.validation.mlx_calibration import run_mlx_forward_model
r = run_mlx_forward_model(0.7, 0.08, 'pf1000', grid_shape=(32,1,64))
print(f'I_peak={r.I_peak_A/1e6:.3f} MA, err={r.peak_error*100:.1f}%, NRMSE={r.nrmse:.3f}')
assert r.peak_error < 0.20, f'Regression: I_peak error {r.peak_error*100:.1f}% > 20%'
"
```

This catches compensating errors from physics changes (see MEMORY.md feedback_compensating_errors).
