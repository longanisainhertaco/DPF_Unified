"""Grid convergence study for DPF MHD simulations.

Runs the same physics at multiple grid resolutions to verify that
the solution converges as expected. Reports convergence order and
Richardson extrapolation for the grid-independent solution.

Usage:
    study = ConvergenceStudy(preset_name="pf1000")
    results = study.run()
    print(results.summary)

References:
    Roache, P.J., "Verification and Validation in Computational Science
    and Engineering" (1998) — GCI method.
    Richardson, L.F., Phil. Trans. R. Soc. A 226:299 (1927).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceResult:
    """Result of a grid convergence study."""

    resolutions: list[tuple[int, int, int]]  # Grid shapes tested
    dx_values: list[float]  # Cell sizes [m]
    I_peak_values: list[float]  # Peak current [MA] at each resolution
    t_peak_values: list[float]  # Time of peak [us] at each resolution
    B_max_values: list[float]  # Peak B-field [T] at each resolution
    rho_max_values: list[float]  # Peak density at each resolution
    wall_times: list[float]  # Wall-clock time [s] per run
    convergence_order: float  # Measured convergence order p
    richardson_I_peak: float  # Richardson-extrapolated I_peak
    gci_fine: float  # Grid Convergence Index (%) for finest grid
    is_converged: bool  # True if GCI < 5%
    preset: str
    summary: str


def compute_convergence_order(
    f1: float, f2: float, f3: float,
    r: float = 2.0,
) -> float:
    """Compute observed convergence order from three grid levels.

    Uses the formula: p = ln((f3-f2)/(f2-f1)) / ln(r)

    where f1 = finest, f2 = medium, f3 = coarsest, r = refinement ratio.

    Args:
        f1: Solution on finest grid.
        f2: Solution on medium grid.
        f3: Solution on coarsest grid.
        r: Grid refinement ratio (default 2.0).

    Returns:
        Observed convergence order p. Returns 0 if oscillatory.
    """
    num = f3 - f2
    den = f2 - f1
    if abs(den) < 1e-15 or abs(num) < 1e-15:
        return 0.0  # No difference — already converged or oscillatory
    ratio = num / den
    if ratio <= 0:
        return 0.0  # Oscillatory convergence
    return float(np.log(abs(ratio)) / np.log(r))


def richardson_extrapolation(
    f1: float, f2: float,
    p: float, r: float = 2.0,
) -> float:
    """Richardson extrapolation to estimate grid-independent solution.

    f_exact ~ f1 + (f1 - f2) / (r^p - 1)

    Args:
        f1: Solution on finest grid.
        f2: Solution on next coarser grid.
        p: Convergence order.
        r: Refinement ratio.

    Returns:
        Estimated grid-independent solution.
    """
    if p <= 0 or abs(r**p - 1) < 1e-15:
        return f1  # Can't extrapolate
    return f1 + (f1 - f2) / (r**p - 1)


def grid_convergence_index(
    f1: float, f2: float,
    p: float, r: float = 2.0,
    Fs: float = 1.25,
) -> float:
    """Compute Grid Convergence Index (GCI) for the fine grid.

    GCI = Fs * |epsilon| / (r^p - 1)

    where epsilon = (f2 - f1) / f1 is the relative error.

    Args:
        f1: Solution on finest grid.
        f2: Solution on next coarser grid.
        p: Convergence order.
        r: Refinement ratio.
        Fs: Safety factor (1.25 for 3+ grids, 3.0 for 2 grids).

    Returns:
        GCI as a fraction (multiply by 100 for percentage).
    """
    if abs(f1) < 1e-30 or p <= 0:
        return 1.0  # 100% uncertainty
    epsilon = abs((f2 - f1) / f1)
    rp = r**p
    if abs(rp - 1) < 1e-15:
        return 1.0
    return Fs * epsilon / (rp - 1)


def run_convergence_study(
    preset_name: str = "tutorial",
    resolutions: list[tuple[int, int, int]] | None = None,
    sim_time_us: float = 5.0,
    gas_key: str = "D2",
    backend: str = "hybrid",
) -> ConvergenceResult:
    """Run a grid convergence study.

    Executes the simulation at 3+ grid resolutions and computes
    convergence order, Richardson extrapolation, and GCI.

    Args:
        preset_name: Device preset.
        resolutions: List of (nr, ny, nz) grid shapes.
            Default: 3 levels with 2x refinement.
        sim_time_us: Simulation time per run [us].
        gas_key: Fill gas species.
        backend: Simulation backend.

    Returns:
        ConvergenceResult with all metrics.
    """
    import time

    if resolutions is None:
        resolutions = [
            (16, 1, 32),   # Coarse
            (32, 1, 64),   # Medium
            (64, 1, 128),  # Fine
        ]

    dx_values = []
    I_peak_values = []
    t_peak_values = []
    B_max_values = []
    rho_max_values = []
    wall_times = []

    for grid_shape in resolutions:
        nr, ny, nz = grid_shape
        logger.info("Convergence study: running %s at %dx%dx%d", preset_name, nr, ny, nz)

        t0 = time.perf_counter()
        try:
            from app_mhd import MHD_GRID_PRESETS, run_mhd_simulation
            # Find matching grid preset or use custom
            grid_key = f"{nr}x{ny}x{nz}"
            for k, v in MHD_GRID_PRESETS.items():
                if v == grid_shape:
                    grid_key = k
                    break

            result = run_mhd_simulation(
                backend=backend,
                grid_preset=grid_key,
                preset_name=preset_name,
                sim_time_us=sim_time_us,
                gas_key=gas_key,
            )

            elapsed = time.perf_counter() - t0

            I_peak = result.get("I_peak", 0.0)
            t_us = result.get("t_us", np.array([0]))
            I_MA = result.get("I_MA", np.array([0]))
            t_peak = float(t_us[np.argmax(np.abs(I_MA))]) if len(I_MA) > 0 else 0.0

            final_state = result.get("final_state", {})
            B = final_state.get("B")
            B_max = float(np.max(np.sqrt(np.sum(B**2, axis=0)))) if B is not None else 0.0
            rho_max = float(np.max(final_state.get("rho", np.array([0])))) if final_state else 0.0

            # Compute dx from geometry
            from dpf.presets import get_preset
            preset = get_preset(preset_name)
            cc = preset["circuit"]
            a, b_r = cc["anode_radius"], cc["cathode_radius"]
            dx = (b_r - a) / nr

        except Exception as exc:
            logger.warning("Convergence run at %s failed: %s", grid_shape, exc)
            elapsed = time.perf_counter() - t0
            I_peak = 0.0
            t_peak = 0.0
            B_max = 0.0
            rho_max = 0.0
            dx = 0.001

        dx_values.append(dx)
        I_peak_values.append(I_peak)
        t_peak_values.append(t_peak)
        B_max_values.append(B_max)
        rho_max_values.append(rho_max)
        wall_times.append(elapsed)

        logger.info(
            "  dx=%.3f mm, I_peak=%.3f MA, B_max=%.1f T, %.1f s",
            dx * 1e3, I_peak, B_max, elapsed,
        )

    # Compute convergence metrics from I_peak (most physically meaningful)
    n = len(I_peak_values)
    if n >= 3 and all(v > 0 for v in I_peak_values[-3:]):
        f1, f2, f3 = I_peak_values[-1], I_peak_values[-2], I_peak_values[-3]
        r = dx_values[-2] / dx_values[-1] if dx_values[-1] > 0 else 2.0
        p = compute_convergence_order(f1, f2, f3, r)
        rich = richardson_extrapolation(f1, f2, p, r)
        gci = grid_convergence_index(f1, f2, p, r)
    elif n >= 2 and all(v > 0 for v in I_peak_values[-2:]):
        f1, f2 = I_peak_values[-1], I_peak_values[-2]
        r = dx_values[-2] / dx_values[-1] if dx_values[-1] > 0 else 2.0
        p = 1.0  # Assume first order with only 2 grids
        rich = richardson_extrapolation(f1, f2, p, r)
        gci = grid_convergence_index(f1, f2, p, r, Fs=3.0)
    else:
        p = 0.0
        rich = I_peak_values[-1] if I_peak_values else 0.0
        gci = 1.0

    is_converged = gci < 0.05  # 5% threshold

    summary_lines = [
        f"Grid Convergence Study: {preset_name}",
        f"Backend: {backend}, Gas: {gas_key}, sim_time: {sim_time_us} us",
        "",
        "| Grid | dx [mm] | I_peak [MA] | B_max [T] | Time [s] |",
        "|------|---------|-------------|-----------|----------|",
    ]
    for i, res in enumerate(resolutions):
        summary_lines.append(
            f"| {res[0]}x{res[2]} | {dx_values[i]*1e3:.2f} | "
            f"{I_peak_values[i]:.3f} | {B_max_values[i]:.1f} | "
            f"{wall_times[i]:.1f} |"
        )
    summary_lines.extend([
        "",
        f"Convergence order p = {p:.2f}",
        f"Richardson-extrapolated I_peak = {rich:.4f} MA",
        f"GCI (fine grid) = {gci*100:.1f}%",
        f"Converged: {'YES' if is_converged else 'NO (GCI > 5%)'}",
    ])

    return ConvergenceResult(
        resolutions=resolutions,
        dx_values=dx_values,
        I_peak_values=I_peak_values,
        t_peak_values=t_peak_values,
        B_max_values=B_max_values,
        rho_max_values=rho_max_values,
        wall_times=wall_times,
        convergence_order=p,
        richardson_I_peak=rich,
        gci_fine=gci,
        is_converged=is_converged,
        preset=preset_name,
        summary="\n".join(summary_lines),
    )
