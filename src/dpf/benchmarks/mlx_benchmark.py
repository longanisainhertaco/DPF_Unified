"""MLX MHD solver performance benchmarks.

Measures wall-clock time per step for:
  1. 64x256   (small — GPU overhead visible)
  2. 128x512  (production — target grid)
  3. 256x1024 (large — bandwidth-bound)

Each benchmark runs 20 steps of a uniform-state advection problem
(no circuit coupling, no resistivity — pure MHD throughput).

Usage:
    python3 -m dpf.benchmarks.mlx_benchmark
    python3 -m dpf.benchmarks.mlx_benchmark --output results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

import numpy as np

HAS_MLX = False
HAS_MPS = False

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    pass

try:
    import torch

    HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except ImportError:
    pass

# Uniform state parameters — chosen to give a stable CFL timestep on all grids
_RHO0: float = 1e-3       # kg/m^3
_P0: float = 1000.0       # Pa
_B_THETA0: float = 1.0    # T (SI, converted to HL inside MLX solver)
_GAMMA: float = 5.0 / 3.0


# ------------------------------------------------------------------
# State builders
# ------------------------------------------------------------------


def _uniform_state_numpy(nr: int, nz: int) -> dict[str, np.ndarray]:
    """Return a uniform plasma state dict on a (nr, 1, nz) grid."""
    rho = np.full((nr, 1, nz), _RHO0, dtype=np.float32)
    vel = np.zeros((3, nr, 1, nz), dtype=np.float32)
    p = np.full((nr, 1, nz), _P0, dtype=np.float32)
    # B: set B_theta (index 2 = toroidal) to B_THETA0; Br=Bz=0
    B = np.zeros((3, nr, 1, nz), dtype=np.float32)
    B[2] = _B_THETA0
    m_D = 3.34358377e-27
    k_B = 1.380649e-23
    T = _P0 * m_D / (_RHO0 * k_B)
    Te = np.full((nr, 1, nz), T, dtype=np.float32)
    return {
        "rho": rho,
        "velocity": vel,
        "pressure": p,
        "B": B,
        "Te": Te,
        "Ti": Te.copy(),
        "psi": np.zeros((nr, 1, nz), dtype=np.float32),
    }


def _fixed_dt(nr: int, nz: int, dr: float, dz: float) -> float:
    """Conservative CFL timestep for the uniform state."""
    cs = float(np.sqrt(_GAMMA * _P0 / _RHO0))
    v_A = float(_B_THETA0 / np.sqrt(4.0 * np.pi * 1e-7 * _RHO0))
    cf = float(np.sqrt(cs**2 + v_A**2))
    return 0.3 * min(dr, dz) / (cf + 1e-30)


# ------------------------------------------------------------------
# MLX benchmark
# ------------------------------------------------------------------


def benchmark_mlx(nr: int, nz: int, n_steps: int = 20) -> dict[str, Any]:
    """Run n_steps of the MLX cylindrical MHD solver.

    Returns ms/step and zone-cycles/sec.  First step is excluded (JIT warmup).

    Parameters
    ----------
    nr, nz : int
        Grid dimensions (radial, axial).
    n_steps : int
        Number of timed steps (excluding 1 warmup step).

    Returns
    -------
    dict with keys: ms_per_step, zone_cycles_per_sec, grid, backend.
    """
    if not HAS_MLX:
        return {"grid": f"{nr}x{nz}", "backend": "mlx", "error": "MLX not available"}

    from dpf.metal.mlx_solver import MLXMHDSolver

    dr = 0.10 / nr   # outer radius 0.1 m
    dz = 0.50 / nz   # electrode length 0.5 m
    dt = _fixed_dt(nr, nz, dr, dz)

    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz),
        dx=dr,
        dz=dz,
        gamma=_GAMMA,
        reconstruction="plm",
        riemann_solver="hll",
        time_integrator="ssp_rk2",
        coordinates="cylindrical",
        convert_b_si_to_hl=True,
    )
    state = _uniform_state_numpy(nr, nz)

    # Warmup step — triggers MLX JIT compilation
    _ = solver.step(state, dt, current=0.0, voltage=0.0)
    mx.eval()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = solver.step(state, dt, current=0.0, voltage=0.0)
        mx.eval()
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed * 1000.0 / n_steps
    zones = nr * nz
    zone_cycles_per_sec = zones * n_steps / elapsed

    return {
        "grid": f"{nr}x{nz}",
        "backend": "mlx",
        "ms_per_step": ms_per_step,
        "zone_cycles_per_sec": zone_cycles_per_sec,
    }


# ------------------------------------------------------------------
# PyTorch Metal benchmark
# ------------------------------------------------------------------


def benchmark_pytorch_metal(nr: int, nz: int, n_steps: int = 20) -> dict[str, Any]:
    """Run n_steps of the PyTorch Metal (MPS) cylindrical MHD solver.

    Operates on a (nr, nz, nz) grid — Metal solver is Cartesian/3D.
    Uses nz for the third dimension to keep zone count comparable.

    Parameters
    ----------
    nr, nz : int
        Grid dimensions; Metal solver gets (nr, nz, nz) 3D grid.
    n_steps : int
        Number of timed steps (excluding 1 warmup step).

    Returns
    -------
    dict with keys: ms_per_step, zone_cycles_per_sec, grid, backend.
    """
    if not HAS_MPS:
        return {"grid": f"{nr}x{nz}", "backend": "metal", "error": "MPS not available"}

    from dpf.metal.metal_solver import MetalMHDSolver

    # Use a slim (nr, 4, nz) grid so zone count ≈ MLX (nr*nz)
    ny_metal = 4
    dx = 0.10 / nr
    dz = 0.50 / nz
    dt = _fixed_dt(nr, nz, dx, dz)

    solver = MetalMHDSolver(
        grid_shape=(nr, ny_metal, nz),
        dx=dx,
        dz=dz,
        gamma=_GAMMA,
        device="mps",
        reconstruction="plm",
        riemann_solver="hll",
        use_ct=False,
    )

    rho = np.full((nr, ny_metal, nz), _RHO0, dtype=np.float32)
    vel = np.zeros((3, nr, ny_metal, nz), dtype=np.float32)
    p = np.full((nr, ny_metal, nz), _P0, dtype=np.float32)
    B = np.zeros((3, nr, ny_metal, nz), dtype=np.float32)
    m_D = 3.34358377e-27
    k_B = 1.380649e-23
    T_val = _P0 * m_D / (_RHO0 * k_B)
    Te = np.full((nr, ny_metal, nz), T_val, dtype=np.float32)
    state = {
        "rho": rho, "velocity": vel, "pressure": p, "B": B,
        "Te": Te, "Ti": Te.copy(), "psi": np.zeros_like(rho),
    }

    # Warmup
    _ = solver.step(state, dt, current=0.0, voltage=0.0)
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = solver.step(state, dt, current=0.0, voltage=0.0)
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed * 1000.0 / n_steps
    zones = nr * nz  # report per-(r,z) slice for fair comparison
    zone_cycles_per_sec = zones * n_steps / elapsed

    return {
        "grid": f"{nr}x{nz}",
        "backend": "metal",
        "ms_per_step": ms_per_step,
        "zone_cycles_per_sec": zone_cycles_per_sec,
    }


# ------------------------------------------------------------------
# NumPy (Python engine) benchmark
# ------------------------------------------------------------------


def benchmark_numpy(nr: int, nz: int, n_steps: int = 20) -> dict[str, Any]:
    """Run n_steps of the Python/NumPy cylindrical MHD solver.

    Uses CylindricalMHDSolver with PLM+HLL for a fair comparison.

    Parameters
    ----------
    nr, nz : int
        Grid dimensions.
    n_steps : int
        Number of timed steps (excluding 1 warmup step).

    Returns
    -------
    dict with keys: ms_per_step, zone_cycles_per_sec, grid, backend.
    """
    try:
        from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver
    except ImportError:
        try:
            from dpf.fluid.mhd_solver import MHDSolver as _Fallback

            # Fallback: use Cartesian solver with a thin (nr,1,nz) grid
            dr = 0.10 / nr
            dz = 0.50 / nz
            dt = _fixed_dt(nr, nz, dr, dz)
            solver = _Fallback(
                grid_shape=(nr, 1, nz),
                dx=dr,
                gamma=_GAMMA,
                enable_hall=False,
                enable_braginskii=False,
                enable_resistive=False,
            )
            state = _uniform_state_numpy(nr, nz)

            _ = solver.step(state, dt, current=0.0, voltage=0.0)
            t0 = time.perf_counter()
            for _ in range(n_steps):
                state = solver.step(state, dt, current=0.0, voltage=0.0)
            elapsed = time.perf_counter() - t0

            ms_per_step = elapsed * 1000.0 / n_steps
            zones = nr * nz
            return {
                "grid": f"{nr}x{nz}",
                "backend": "numpy",
                "ms_per_step": ms_per_step,
                "zone_cycles_per_sec": zones * n_steps / elapsed,
            }
        except Exception as exc:
            return {"grid": f"{nr}x{nz}", "backend": "numpy", "error": str(exc)}

    dr = 0.10 / nr
    dz = 0.50 / nz
    dt = _fixed_dt(nr, nz, dr, dz)

    solver = CylindricalMHDSolver(
        nr=nr,
        nz=nz,
        dr=dr,
        dz=dz,
        gamma=_GAMMA,
    )
    state = _uniform_state_numpy(nr, nz)

    _ = solver.step(state, dt, current=0.0, voltage=0.0)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = solver.step(state, dt, current=0.0, voltage=0.0)
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed * 1000.0 / n_steps
    zones = nr * nz
    return {
        "grid": f"{nr}x{nz}",
        "backend": "numpy",
        "ms_per_step": ms_per_step,
        "zone_cycles_per_sec": zones * n_steps / elapsed,
    }


# ------------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------------


def run_all_benchmarks(
    n_steps: int = 20,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run MLX, Metal, and NumPy benchmarks at three grid sizes.

    Parameters
    ----------
    n_steps : int
        Steps per benchmark (excluding 1 JIT warmup step).
    output_path : str | None
        Optional path to write JSON results.

    Returns
    -------
    dict with system info and per-grid benchmark results.
    """
    grids = [(64, 256), (128, 512), (256, 1024)]

    print("MLX MHD Solver Benchmarks (M3 Pro, 36GB)")
    print("=========================================")
    print(f"  Platform : {platform.platform()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  MLX      : {HAS_MLX}")
    print(f"  MPS      : {HAS_MPS}")
    print(f"  Steps    : {n_steps} (+ 1 warmup)")
    print()

    results: dict[str, Any] = {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "mlx_available": HAS_MLX,
            "mps_available": HAS_MPS,
            "n_steps": n_steps,
        },
        "benchmarks": [],
    }

    rows: list[dict[str, Any]] = []

    for nr, nz in grids:
        label = f"{nr}x{nz}"
        print(f"Grid {label}:")

        print("  MLX ...   ", end="", flush=True)
        mlx_res = benchmark_mlx(nr, nz, n_steps)
        _print_result(mlx_res)

        print("  Metal ... ", end="", flush=True)
        metal_res = benchmark_pytorch_metal(nr, nz, n_steps)
        _print_result(metal_res)

        print("  NumPy ... ", end="", flush=True)
        numpy_res = benchmark_numpy(nr, nz, n_steps)
        _print_result(numpy_res)

        row: dict[str, Any] = {"grid": label}
        row["mlx"] = mlx_res
        row["metal"] = metal_res
        row["numpy"] = numpy_res

        # Compute speedups vs NumPy baseline
        numpy_ms = numpy_res.get("ms_per_step")
        mlx_ms = mlx_res.get("ms_per_step")
        metal_ms = metal_res.get("ms_per_step")

        if numpy_ms and mlx_ms:
            row["mlx_speedup_vs_numpy"] = numpy_ms / mlx_ms
        if numpy_ms and metal_ms:
            row["metal_speedup_vs_numpy"] = numpy_ms / metal_ms
        if mlx_ms and metal_ms:
            row["mlx_speedup_vs_metal"] = metal_ms / mlx_ms

        rows.append(row)
        results["benchmarks"].append(row)

    # Print comparison table
    _print_table(rows)

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults written to {output_path}")

    return results


def _print_result(res: dict[str, Any]) -> None:
    if "error" in res:
        print(f"SKIP ({res['error']})")
    else:
        ms = res["ms_per_step"]
        zcs = res["zone_cycles_per_sec"] / 1e6
        print(f"{ms:8.2f} ms/step  ({zcs:.2f} Mzc/s)")


def _fmt_ms(res: dict[str, Any]) -> str:
    if "error" in res:
        return f"{'N/A':>14}"
    return f"{res['ms_per_step']:>14.2f}"


def _fmt_speedup(val: float | None) -> str:
    if val is None:
        return f"{'N/A':>12}"
    return f"{val:>11.2f}x"


def _print_table(rows: list[dict[str, Any]]) -> None:
    hdr = (
        f"\n{'Grid':<12} | {'MLX (ms/step)':>14} | {'Metal (ms/step)':>15} | "
        f"{'NumPy (ms/step)':>15} | {'MLX Speedup':>12}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for row in rows:
        mlx_s = _fmt_ms(row["mlx"])
        metal_s = _fmt_ms(row["metal"])
        numpy_s = _fmt_ms(row["numpy"])
        speedup = _fmt_speedup(row.get("mlx_speedup_vs_numpy"))
        print(f"{row['grid']:<12} | {mlx_s} | {metal_s:>15} | {numpy_s:>15} | {speedup}")
    print(sep)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLX MHD solver performance benchmarks (M3 Pro)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Timed steps per grid (default: 20, +1 warmup excluded)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()
    run_all_benchmarks(n_steps=args.steps, output_path=args.output)
