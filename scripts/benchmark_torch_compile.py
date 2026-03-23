"""Benchmark torch.compile acceleration for MetalMHDSolver.

Runs the Sod shock tube problem with compile_mode=True and compile_mode=False,
reporting wall time, speedup, and max numerical difference between the two paths.

Usage
-----
    python3 scripts/benchmark_torch_compile.py [--grid N] [--steps S] [--device D]

Defaults: --grid 64 --steps 20 --device cpu
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from dpf.metal.metal_solver import MetalMHDSolver


def _sod_state(nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
    """1D Sod shock initial conditions mapped to 3D array."""
    rho = np.ones((nx, ny, nz), dtype=np.float32)
    vel = np.zeros((3, nx, ny, nz), dtype=np.float32)
    p = np.ones((nx, ny, nz), dtype=np.float32)
    B = np.zeros((3, nx, ny, nz), dtype=np.float32)
    B[2] = 1e-4  # small Bz to keep MHD well-posed

    mid = nx // 2
    rho[:mid] = 1.0
    rho[mid:] = 0.125
    p[:mid] = 1.0
    p[mid:] = 0.1

    Te = np.full((nx, ny, nz), 1e4, dtype=np.float32)
    Ti = np.full((nx, ny, nz), 1e4, dtype=np.float32)
    psi = np.zeros((nx, ny, nz), dtype=np.float32)

    return {"rho": rho, "velocity": vel, "pressure": p, "B": B,
            "Te": Te, "Ti": Ti, "psi": psi}


def _run(
    compile_mode: bool,
    grid_n: int,
    n_steps: int,
    device: str,
    dt: float,
) -> tuple[float, dict[str, np.ndarray]]:
    """Run n_steps of the Sod problem and return (wall_time, final_state)."""
    solver = MetalMHDSolver(
        grid_shape=(grid_n, grid_n, grid_n),
        dx=1.0 / grid_n,
        gamma=1.4,
        cfl=0.3,
        device=device,
        use_ct=False,  # CT needs MPS; avoid device constraint
        compile_mode=compile_mode,
        time_integrator="ssp_rk2",
    )

    state = _sod_state(grid_n, grid_n, grid_n)

    # Warmup step (exclude from timing — first call does JIT/trace)
    _ = solver.step(state, dt=dt, current=0.0, voltage=0.0)

    state = _sod_state(grid_n, grid_n, grid_n)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
    elapsed = time.perf_counter() - t0

    return elapsed, state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64,
                        help="Grid size (NxNxN). Default: 64")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of steps to time. Default: 20")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device (cpu or mps). Default: cpu")
    args = parser.parse_args()

    n = args.grid
    steps = args.steps
    device = args.device
    dt = 1e-4 / n  # scale dt with grid so CFL stays safe

    print(f"Grid: {n}^3  |  Steps: {steps}  |  Device: {device}")
    print("-" * 50)

    t_eager, state_eager = _run(
        compile_mode=False, grid_n=n, n_steps=steps, device=device, dt=dt,
    )
    print(f"Eager  path: {t_eager:.3f}s  ({t_eager/steps*1000:.2f} ms/step)")

    t_compiled, state_compiled = _run(
        compile_mode=True, grid_n=n, n_steps=steps, device=device, dt=dt,
    )
    print(f"Compiled path: {t_compiled:.3f}s  ({t_compiled/steps*1000:.2f} ms/step)")

    speedup = t_eager / t_compiled if t_compiled > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")

    # Numerical diff (compiled result vs eager result)
    max_diff: float = 0.0
    for key in ("rho", "pressure", "B"):
        diff = np.max(np.abs(state_compiled[key] - state_eager[key]))
        max_diff = max(max_diff, float(diff))
        print(f"  max|diff[{key}]|  = {diff:.3e}")

    print(f"\nMax numerical difference (compiled vs eager): {max_diff:.3e}")
    if max_diff < 1e-3:
        print("PASS: compiled and eager paths agree to <1e-3")
    else:
        print("WARN: larger-than-expected difference — review compiled path")


if __name__ == "__main__":
    main()
