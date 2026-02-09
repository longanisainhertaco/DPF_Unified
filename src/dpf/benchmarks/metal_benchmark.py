"""Metal GPU benchmark suite comparing CPU vs Metal GPU performance for DPF operations.

Production benchmark suite exercising the full Metal acceleration stack:
  - PyTorch MPS element-wise operations and reductions
  - Metal stencil kernels (Laplacian, constrained transport)
  - Metal Riemann solver (HLL flux computation)
  - MetalMHDSolver full MHD timesteps
  - Memory transfer overhead (NumPy ↔ MPS vs MLX)
  - WALRUS surrogate inference (when checkpoint available)

Usage:
    python -m dpf.benchmarks.metal_benchmark --grid 32 --output results.json
    # or
    from dpf.benchmarks.metal_benchmark import run_all_benchmarks
    run_all_benchmarks(grid_size=32)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

# Check Metal/MPS availability
HAS_MPS = False
HAS_MLX = False

try:
    import torch

    HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except ImportError:
    pass

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    pass


# ------------------------------------------------------------------
# Timing utility
# ------------------------------------------------------------------


def _bench(
    fn: callable,
    n_warmup: int = 3,
    n_iter: int = 10,
    device_sync: bool = False,
) -> dict[str, float]:
    """Run fn n_warmup times (discard), then n_iter times, return timing stats.

    Args:
        fn: Callable to benchmark (no arguments).
        n_warmup: Number of warmup iterations (results discarded).
        n_iter: Number of timed iterations.
        device_sync: If True, synchronize MPS device after each iteration.

    Returns:
        Dict with timing statistics in milliseconds:
            - mean_ms: Mean execution time
            - std_ms: Standard deviation
            - p50_ms: Median
            - p95_ms: 95th percentile
            - min_ms: Minimum time
    """
    for _ in range(n_warmup):
        fn()
    if device_sync and HAS_MPS:
        torch.mps.synchronize()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        if device_sync and HAS_MPS:
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return {
        "mean_ms": float(np.mean(times_arr) * 1000),
        "std_ms": float(np.std(times_arr) * 1000),
        "p50_ms": float(np.median(times_arr) * 1000),
        "p95_ms": float(np.percentile(times_arr, 95) * 1000),
        "min_ms": float(np.min(times_arr) * 1000),
    }


# ------------------------------------------------------------------
# Benchmark 1: Element-wise operations
# ------------------------------------------------------------------


def bench_elementwise(n: int = 64, device: str = "mps", n_iter: int = 100) -> dict:
    """Compare NumPy element-wise ops vs PyTorch MPS on (n,n,n) arrays.

    Operations: maximum, +, *, /

    Args:
        n: Grid size (cube n×n×n).
        device: PyTorch device ("mps" or "cpu").
        n_iter: Number of iterations.

    Returns:
        Dict with cpu_ms, mps_ms, speedup.
    """
    if not HAS_MPS and device == "mps":
        return {"error": "MPS not available"}

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((n, n, n)).astype(np.float32)
    b_np = rng.standard_normal((n, n, n)).astype(np.float32)

    # CPU (NumPy)
    def cpu_fn():
        c = np.maximum(a_np, b_np)
        d = a_np + b_np
        e = a_np * b_np
        f = a_np / (b_np + 1.0)
        return c + d + e + f

    cpu_stats = _bench(cpu_fn, n_warmup=3, n_iter=n_iter)

    # MPS (PyTorch)
    if HAS_MPS:
        import torch

        a_mps = torch.as_tensor(a_np, device=device)
        b_mps = torch.as_tensor(b_np, device=device)

        def mps_fn():
            c = torch.maximum(a_mps, b_mps)
            d = a_mps + b_mps
            e = a_mps * b_mps
            f = a_mps / (b_mps + 1.0)
            return c + d + e + f

        mps_stats = _bench(mps_fn, n_warmup=3, n_iter=n_iter, device_sync=True)
        speedup = cpu_stats["mean_ms"] / max(mps_stats["mean_ms"], 1e-9)
    else:
        mps_stats = {"mean_ms": 0.0}
        speedup = 0.0

    return {
        "operation": f"Elementwise ({n}³)",
        "cpu_ms": cpu_stats["mean_ms"],
        "mps_ms": mps_stats["mean_ms"],
        "speedup": speedup,
    }


# ------------------------------------------------------------------
# Benchmark 2: Reduction operations
# ------------------------------------------------------------------


def bench_reduction(n: int = 128, device: str = "mps", n_iter: int = 100) -> dict:
    """Compare NumPy reductions vs PyTorch MPS on (n,n,n) arrays.

    Operations: sum, max

    Args:
        n: Grid size (cube n×n×n).
        device: PyTorch device ("mps" or "cpu").
        n_iter: Number of iterations.

    Returns:
        Dict with cpu_ms, mps_ms, speedup.
    """
    if not HAS_MPS and device == "mps":
        return {"error": "MPS not available"}

    rng = np.random.default_rng(123)
    a_np = rng.standard_normal((n, n, n)).astype(np.float32)

    # CPU (NumPy)
    def cpu_fn():
        s = np.sum(a_np)
        m = np.max(a_np)
        return s + m

    cpu_stats = _bench(cpu_fn, n_warmup=3, n_iter=n_iter)

    # MPS (PyTorch)
    if HAS_MPS:
        import torch

        a_mps = torch.as_tensor(a_np, device=device)

        def mps_fn():
            s = torch.sum(a_mps)
            m = torch.max(a_mps)
            return s + m

        mps_stats = _bench(mps_fn, n_warmup=3, n_iter=n_iter, device_sync=True)
        speedup = cpu_stats["mean_ms"] / max(mps_stats["mean_ms"], 1e-9)
    else:
        mps_stats = {"mean_ms": 0.0}
        speedup = 0.0

    return {
        "operation": f"Reduction ({n}³)",
        "cpu_ms": cpu_stats["mean_ms"],
        "mps_ms": mps_stats["mean_ms"],
        "speedup": speedup,
    }


# ------------------------------------------------------------------
# Benchmark 3: 7-point Laplacian stencil
# ------------------------------------------------------------------


def bench_stencil_laplacian(n: int = 64, device: str = "mps", n_iter: int = 50) -> dict:
    """Compare NumPy Laplacian vs metal_stencil.laplacian_3d_mps.

    Uses 7-point stencil: d²f/dx² + d²f/dy² + d²f/dz².

    Args:
        n: Grid size (cube n×n×n).
        device: PyTorch device ("mps" or "cpu").
        n_iter: Number of iterations.

    Returns:
        Dict with cpu_ms, mps_ms, speedup.
    """
    if not HAS_MPS and device == "mps":
        return {"error": "MPS not available"}

    rng = np.random.default_rng(456)
    field_np = rng.standard_normal((n, n, n)).astype(np.float32)
    dx = dy = dz = 1e-3

    # CPU (NumPy finite differences)
    def laplacian_np(f, h):
        lap = np.zeros_like(f)
        # d²f/dx²
        lap[1:-1, :, :] += (f[2:, :, :] - 2 * f[1:-1, :, :] + f[:-2, :, :]) / h**2
        # d²f/dy²
        lap[:, 1:-1, :] += (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / h**2
        # d²f/dz²
        lap[:, :, 1:-1] += (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / h**2
        return lap

    def cpu_fn():
        return laplacian_np(field_np, dx)

    cpu_stats = _bench(cpu_fn, n_warmup=3, n_iter=n_iter)

    # MPS (metal_stencil)
    if HAS_MPS:
        import torch

        from dpf.metal.metal_stencil import laplacian_3d_mps

        field_mps = torch.as_tensor(field_np, device=device)

        def mps_fn():
            return laplacian_3d_mps(field_mps, dx, dy, dz)

        mps_stats = _bench(mps_fn, n_warmup=3, n_iter=n_iter, device_sync=True)
        speedup = cpu_stats["mean_ms"] / max(mps_stats["mean_ms"], 1e-9)
    else:
        mps_stats = {"mean_ms": 0.0}
        speedup = 0.0

    return {
        "operation": f"Laplacian ({n}³)",
        "cpu_ms": cpu_stats["mean_ms"],
        "mps_ms": mps_stats["mean_ms"],
        "speedup": speedup,
    }


# ------------------------------------------------------------------
# Benchmark 4: Constrained transport update
# ------------------------------------------------------------------


def bench_ct_update(n: int = 32, device: str = "mps", n_iter: int = 50) -> dict:
    """Compare Numba ct_update vs metal_stencil.ct_update_mps.

    Faraday's law update on staggered grid.

    Args:
        n: Grid size (cube n×n×n).
        device: PyTorch device ("mps" or "cpu").
        n_iter: Number of iterations.

    Returns:
        Dict with cpu_ms, mps_ms, speedup.
    """
    if not HAS_MPS and device == "mps":
        return {"error": "MPS not available"}

    from dpf.fluid.constrained_transport import (
        StaggeredBField,
        ct_update,
        emf_from_fluxes,
    )

    rng = np.random.default_rng(789)
    dx = dy = dz = 1e-3
    dt = 1e-9

    Bx_face_np = rng.standard_normal((n + 1, n, n)).astype(np.float32) * 0.1
    By_face_np = rng.standard_normal((n, n + 1, n)).astype(np.float32) * 0.1
    Bz_face_np = rng.standard_normal((n, n, n + 1)).astype(np.float32) * 0.1

    Fx_np = rng.standard_normal((n + 1, n, n)).astype(np.float32) * 1e4
    Fy_np = rng.standard_normal((n, n + 1, n)).astype(np.float32) * 1e4
    Fz_np = rng.standard_normal((n, n, n + 1)).astype(np.float32) * 1e4

    # CPU (Numba)
    Ex_np, Ey_np, Ez_np = emf_from_fluxes(Fx_np, Fy_np, Fz_np, dx, dy, dz)

    def cpu_fn():
        stag = StaggeredBField(Bx_face_np, By_face_np, Bz_face_np, dx, dy, dz)
        ct_update(stag, Ex_np, Ey_np, Ez_np, dt)
        return stag

    cpu_stats = _bench(cpu_fn, n_warmup=2, n_iter=n_iter)

    # MPS (metal_stencil)
    if HAS_MPS:
        import torch

        from dpf.metal.metal_stencil import ct_update_mps, emf_from_fluxes_mps

        Bx_mps = torch.as_tensor(Bx_face_np, device=device)
        By_mps = torch.as_tensor(By_face_np, device=device)
        Bz_mps = torch.as_tensor(Bz_face_np, device=device)

        Fx_mps = torch.as_tensor(Fx_np, device=device)
        Fy_mps = torch.as_tensor(Fy_np, device=device)
        Fz_mps = torch.as_tensor(Fz_np, device=device)

        Ex_mps, Ey_mps, Ez_mps = emf_from_fluxes_mps(Fx_mps, Fy_mps, Fz_mps)

        def mps_fn():
            return ct_update_mps(
                Bx_mps, By_mps, Bz_mps, Ex_mps, Ey_mps, Ez_mps, dx, dy, dz, dt
            )

        mps_stats = _bench(mps_fn, n_warmup=2, n_iter=n_iter, device_sync=True)
        speedup = cpu_stats["mean_ms"] / max(mps_stats["mean_ms"], 1e-9)
    else:
        mps_stats = {"mean_ms": 0.0}
        speedup = 0.0

    return {
        "operation": f"CT update ({n}³)",
        "cpu_ms": cpu_stats["mean_ms"],
        "mps_ms": mps_stats["mean_ms"],
        "speedup": speedup,
    }


# ------------------------------------------------------------------
# Benchmark 5: HLL Riemann solver flux computation
# ------------------------------------------------------------------


def bench_hll_flux(n: int = 32, device: str = "mps", n_iter: int = 50) -> dict:
    """Compare Numba HLL flux vs metal_riemann.compute_fluxes_mps.

    Uses full MHD HLL Riemann solver on (n,n,n) grid.

    Args:
        n: Grid size (cube n×n×n).
        device: PyTorch device ("mps" or "cpu").
        n_iter: Number of iterations.

    Returns:
        Dict with cpu_ms, mps_ms, speedup.
    """
    if not HAS_MPS and device == "mps":
        return {"error": "MPS not available"}

    from dpf.fluid.mhd_solver import MHDSolver

    rng = np.random.default_rng(101112)
    dx = 1e-3
    gamma = 5.0 / 3.0

    # Create solver for CPU test
    solver_cpu = MHDSolver(
        grid_shape=(n, n, n),
        dx=dx,
        gamma=gamma,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
    )

    # Create random state
    rho_np = rng.uniform(0.1, 1.0, (n, n, n)).astype(np.float32)
    p_np = rng.uniform(0.1, 1.0, (n, n, n)).astype(np.float32)
    vel_np = rng.standard_normal((3, n, n, n)).astype(np.float32) * 1e3
    B_np = rng.standard_normal((3, n, n, n)).astype(np.float32) * 0.1

    state_np = {
        "rho": rho_np,
        "velocity": vel_np,
        "pressure": p_np,
        "B": B_np,
        "Te": np.full((n, n, n), 1e6, dtype=np.float32),
        "Ti": np.full((n, n, n), 1e6, dtype=np.float32),
        "psi": np.zeros((n, n, n), dtype=np.float32),
    }

    # CPU: benchmark a full MHDSolver step (includes flux computation)
    state_cpu_copy = {k: v.copy() for k, v in state_np.items()}
    dt_cpu = solver_cpu._compute_dt(state_cpu_copy)

    def cpu_fn():
        s = {k: v.copy() for k, v in state_np.items()}
        return solver_cpu.step(s, dt_cpu, current=0.0, voltage=0.0)

    cpu_stats = _bench(cpu_fn, n_warmup=2, n_iter=n_iter)

    # MPS (metal_riemann)
    if HAS_MPS:
        import torch

        from dpf.metal.metal_riemann import mhd_rhs_mps

        state_mps = {
            "rho": torch.as_tensor(rho_np, device=device),
            "velocity": torch.as_tensor(vel_np, device=device),
            "pressure": torch.as_tensor(p_np, device=device),
            "B": torch.as_tensor(B_np, device=device),
        }

        def mps_fn():
            return mhd_rhs_mps(state_mps, gamma, dx, dx, dx, limiter="minmod")

        mps_stats = _bench(mps_fn, n_warmup=2, n_iter=n_iter, device_sync=True)
        speedup = cpu_stats["mean_ms"] / max(mps_stats["mean_ms"], 1e-9)
    else:
        mps_stats = {"mean_ms": 0.0}
        speedup = 0.0

    return {
        "operation": f"HLL flux ({n}³)",
        "cpu_ms": cpu_stats["mean_ms"],
        "mps_ms": mps_stats["mean_ms"],
        "speedup": speedup,
    }


# ------------------------------------------------------------------
# Benchmark 6: Full MHD timestep (Sod problem)
# ------------------------------------------------------------------


def bench_full_mhd_step(n: int = 32, device: str = "mps", n_iter: int = 20) -> dict:
    """Compare MHDSolver.step() vs MetalMHDSolver.step() on Sod problem.

    Single timestep of SSP-RK2 MHD integration.

    Args:
        n: Grid size in x (ny=nz=1 for 1D Sod).
        device: PyTorch device ("mps" or "cpu").
        n_iter: Number of iterations.

    Returns:
        Dict with cpu_ms, mps_ms, speedup, max_rho_diff (accuracy).
    """
    if not HAS_MPS and device == "mps":
        return {"error": "MPS not available"}

    from dpf.constants import k_B, m_d
    from dpf.fluid.mhd_solver import MHDSolver

    dx = 1.0 / n
    gamma = 1.4

    solver_cpu = MHDSolver(
        grid_shape=(n, n, n),
        dx=dx,
        gamma=gamma,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
    )

    # Sod initial conditions (3D, discontinuity along x)
    rho_np = np.where(
        np.linspace(0, 1, n).reshape(n, 1, 1) < 0.5, 1.0, 0.125
    ) * np.ones((n, n, n))
    rho_np = rho_np.astype(np.float64)
    p_np = np.where(
        np.linspace(0, 1, n).reshape(n, 1, 1) < 0.5, 1.0, 0.1
    ) * np.ones((n, n, n))
    p_np = p_np.astype(np.float64)
    vel_np = np.zeros((3, n, n, n), dtype=np.float64)
    B_np = np.zeros((3, n, n, n), dtype=np.float64)
    Te_np = (p_np / (2.0 * rho_np * k_B) * m_d).astype(np.float64)
    Ti_np = Te_np.copy()

    state_np = {
        "rho": rho_np,
        "velocity": vel_np,
        "pressure": p_np,
        "B": B_np,
        "Te": Te_np,
        "Ti": Ti_np,
        "psi": np.zeros_like(rho_np),
    }

    dt = 0.2 * dx

    # CPU
    def cpu_fn():
        s = {k: v.copy() for k, v in state_np.items()}
        return solver_cpu.step(s, dt, current=0.0, voltage=0.0)

    cpu_stats = _bench(cpu_fn, n_warmup=1, n_iter=n_iter)
    result_cpu = cpu_fn()

    # MPS (MetalMHDSolver)
    if HAS_MPS:
        from dpf.metal.metal_solver import MetalMHDSolver

        solver_mps = MetalMHDSolver(
            grid_shape=(n, n, n),
            dx=dx,
            gamma=gamma,
            device=device,
            use_ct=False,
        )

        def mps_fn():
            s = {k: v.copy() for k, v in state_np.items()}
            return solver_mps.step(s, dt, current=0.0, voltage=0.0)

        mps_stats = _bench(mps_fn, n_warmup=1, n_iter=n_iter, device_sync=True)
        result_mps = mps_fn()

        # Compute accuracy: max absolute difference in rho
        max_rho_diff = float(np.max(np.abs(result_cpu["rho"] - result_mps["rho"])))
        speedup = cpu_stats["mean_ms"] / max(mps_stats["mean_ms"], 1e-9)
    else:
        mps_stats = {"mean_ms": 0.0}
        speedup = 0.0
        max_rho_diff = 0.0

    return {
        "operation": f"Full MHD step ({n}³)",
        "cpu_ms": cpu_stats["mean_ms"],
        "mps_ms": mps_stats["mean_ms"],
        "speedup": speedup,
        "note": f"Δρ<{max_rho_diff:.1e}",
    }


# ------------------------------------------------------------------
# Benchmark 7: Memory transfer overhead (NumPy ↔ GPU)
# ------------------------------------------------------------------


def bench_memory_transfer(n: int = 64, n_iter: int = 100) -> dict:
    """Time NumPy ↔ GPU transfer overhead (MPS vs MLX).

    MPS: np → torch.float32 → mps → cpu → np
    MLX: np → mlx (zero-copy on Apple Silicon)

    Args:
        n: Grid size (cube n×n×n).
        n_iter: Number of iterations.

    Returns:
        Dict with mps_roundtrip_ms, mlx_roundtrip_ms.
    """
    rng = np.random.default_rng(131415)
    data_np = rng.standard_normal((n, n, n)).astype(np.float32)

    # MPS roundtrip
    if HAS_MPS:
        import torch

        def mps_fn():
            t = torch.as_tensor(data_np, dtype=torch.float32).to("mps")
            torch.mps.synchronize()
            result = t.cpu().numpy()
            return result

        mps_stats = _bench(mps_fn, n_warmup=3, n_iter=n_iter)
        mps_roundtrip_ms = mps_stats["mean_ms"]
    else:
        mps_roundtrip_ms = 0.0

    # MLX roundtrip (zero-copy on Apple Silicon)
    if HAS_MLX:

        def mlx_fn():
            t = mx.array(data_np)
            mx.eval(t)
            result = np.array(t)
            return result

        mlx_stats = _bench(mlx_fn, n_warmup=3, n_iter=n_iter)
        mlx_roundtrip_ms = mlx_stats["mean_ms"]
    else:
        mlx_roundtrip_ms = 0.0

    return {
        "operation": f"Memory np↔MPS ({n}³)",
        "mps_roundtrip_ms": mps_roundtrip_ms,
        "mlx_roundtrip_ms": mlx_roundtrip_ms,
        "note": "zero-copy" if mlx_roundtrip_ms < 0.01 else "",
    }


# ------------------------------------------------------------------
# Benchmark 8: WALRUS inference (surrogate model)
# ------------------------------------------------------------------


def bench_walrus_inference(device: str = "mps", n_iter: int = 5) -> dict:
    """Time DPFSurrogate.predict_next_step on CPU vs MPS.

    Only runs if WALRUS checkpoint exists at models/walrus-pretrained/walrus.pt.

    Args:
        device: PyTorch device ("mps" or "cpu").
        n_iter: Number of iterations (WALRUS is slow, use small n_iter).

    Returns:
        Dict with cpu_ms, mps_ms, speedup (or error if checkpoint missing).
    """
    checkpoint_path = Path("models/walrus-pretrained/walrus.pt")
    if not checkpoint_path.exists():
        return {
            "operation": "WALRUS inference",
            "error": "Checkpoint not found (models/walrus-pretrained/walrus.pt)",
        }

    try:
        from dpf.ai.surrogate import DPFSurrogate
    except ImportError:
        return {
            "operation": "WALRUS inference",
            "error": "DPFSurrogate not available (install walrus package)",
        }

    # Create test state (16×16×16 minimum grid for WALRUS)
    n = 16
    rng = np.random.default_rng(161718)
    state = {
        "rho": rng.uniform(0.1, 1.0, (n, n, n)).astype(np.float32),
        "velocity": rng.standard_normal((3, n, n, n)).astype(np.float32) * 1e3,
        "pressure": rng.uniform(0.1, 1.0, (n, n, n)).astype(np.float32),
        "B": rng.standard_normal((3, n, n, n)).astype(np.float32) * 0.1,
        "Te": np.full((n, n, n), 1e6, dtype=np.float32),
        "Ti": np.full((n, n, n), 1e6, dtype=np.float32),
        "psi": np.zeros((n, n, n), dtype=np.float32),
    }

    history = [state.copy() for _ in range(4)]

    # CPU inference
    surrogate_cpu = DPFSurrogate(str(checkpoint_path), device="cpu", history_length=4)

    def cpu_fn():
        return surrogate_cpu.predict_next_step(history)

    cpu_stats = _bench(cpu_fn, n_warmup=1, n_iter=n_iter)

    # MPS inference
    if HAS_MPS and device == "mps":
        surrogate_mps = DPFSurrogate(
            str(checkpoint_path), device="mps", history_length=4
        )

        def mps_fn():
            return surrogate_mps.predict_next_step(history)

        mps_stats = _bench(mps_fn, n_warmup=1, n_iter=n_iter, device_sync=True)
        speedup = cpu_stats["mean_ms"] / max(mps_stats["mean_ms"], 1e-9)
    else:
        mps_stats = {"mean_ms": 0.0}
        speedup = 0.0

    return {
        "operation": "WALRUS inference",
        "cpu_ms": cpu_stats["mean_ms"],
        "mps_ms": mps_stats["mean_ms"],
        "speedup": speedup,
    }


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def run_all_benchmarks(grid_size: int = 32, output_path: str | None = None) -> dict:
    """Run all Metal benchmarks and return combined results dict.

    Args:
        grid_size: Default grid size for spatial benchmarks.
        output_path: Optional path to write JSON results.

    Returns:
        Dict with benchmark results and system info.
    """
    print("=" * 70)
    print("DPF Metal Benchmark Suite — M3 Pro (36 GB)")
    print(f"  Platform : {platform.platform()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Machine  : {platform.machine()}")
    print(f"  MPS      : {HAS_MPS}")
    print(f"  MLX      : {HAS_MLX}")
    print("=" * 70)
    print()

    results: dict[str, Any] = {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "mps_available": HAS_MPS,
            "mlx_available": HAS_MLX,
        },
        "benchmarks": [],
    }

    # Run benchmarks
    benchmarks = [
        ("Elementwise", lambda: bench_elementwise(n=grid_size)),
        ("Reduction", lambda: bench_reduction(n=128)),
        ("Laplacian", lambda: bench_stencil_laplacian(n=grid_size)),
        ("CT update", lambda: bench_ct_update(n=grid_size)),
        ("HLL flux", lambda: bench_hll_flux(n=grid_size)),
        ("Full MHD step", lambda: bench_full_mhd_step(n=grid_size)),
        ("Memory transfer", lambda: bench_memory_transfer(n=grid_size)),
        ("WALRUS inference", bench_walrus_inference),
    ]

    for name, bench_fn in benchmarks:
        print(f"Running: {name}...", end=" ", flush=True)
        try:
            result = bench_fn()
            results["benchmarks"].append(result)
            print("✓")
        except Exception as exc:
            print(f"✗ ({exc})")
            results["benchmarks"].append({"operation": name, "error": str(exc)})

    # Print formatted table
    print()
    print("=" * 70)
    print(
        f"{'Operation':<24} {'CPU (ms)':>10} {'MPS (ms)':>10} {'Speedup':>8} {'Note':<10}"
    )
    print("-" * 70)

    for res in results["benchmarks"]:
        if "error" in res:
            op = res.get("operation", "Unknown")
            err = res["error"]
            print(f"{op:<24} {'—':>10} {'—':>10} {'—':>8} {err:<10}")
        elif "mps_roundtrip_ms" in res:
            # Memory transfer special case
            op = res["operation"]
            mps = res.get("mps_roundtrip_ms", 0.0)
            mlx = res.get("mlx_roundtrip_ms", 0.0)
            note = res.get("note", "")
            print(f"{op:<24} {'—':>10} {mps:>10.3f} {'—':>8} {note:<10}")
            if mlx > 0:
                print(
                    f"{'Memory np↔MLX':<24} {'—':>10} {mlx:>10.3f} {'—':>8} {'zero-copy':<10}"
                )
        else:
            op = res["operation"]
            cpu = res.get("cpu_ms", 0.0)
            mps = res.get("mps_ms", 0.0)
            speedup = res.get("speedup", 0.0)
            note = res.get("note", "")
            print(
                f"{op:<24} {cpu:>10.2f} {mps:>10.2f} {speedup:>7.1f}× {note:<10}"
            )

    print("=" * 70)
    print()

    # Optionally write JSON
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output_path}")

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPF Metal GPU benchmark suite (CPU vs Metal GPU)"
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=32,
        help="Grid size for spatial benchmarks (default: 32)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path (optional)"
    )
    args = parser.parse_args()

    run_all_benchmarks(grid_size=args.grid, output_path=args.output)
