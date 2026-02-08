"""Apple Silicon performance benchmarks for DPF solvers.

Exercises the Numba-parallelised hot loops (viscosity, constrained transport,
Nernst advection) and the MHD solver to measure throughput on Apple Silicon
(M-series) hardware.

Usage:
    python -m dpf.benchmarks.apple_silicon
    # or
    from dpf.benchmarks.apple_silicon import run_all; run_all()
"""

from __future__ import annotations

import platform
import time

import numpy as np

from dpf.constants import k_B, m_d, mu_0
from dpf.fluid.viscosity import (
    braginskii_eta0,
    ion_collision_time,
    viscous_stress_rate,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _timer(func, warmup: int = 2, iterations: int = 10):
    """Run *func()* with warmup then return (mean_time, std_time) in seconds."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return float(np.mean(arr)), float(np.std(arr))


# ------------------------------------------------------------------
# Viscosity benchmark
# ------------------------------------------------------------------

def benchmark_viscosity(n: int = 64, iterations: int = 10) -> dict:
    """Benchmark viscous_stress_rate on an n^3 grid.

    Returns dict with grid size, mean time, and std.
    """
    rng = np.random.default_rng(42)
    velocity = rng.standard_normal((3, n, n, n)) * 1e3
    rho = np.full((n, n, n), 1e-4)
    ni = rho / m_d
    Ti = np.full((n, n, n), 1e7)
    tau_i = ion_collision_time(ni, Ti)
    eta0 = braginskii_eta0(ni, Ti, tau_i)
    dx = dy = dz = 1e-3

    def _run():
        viscous_stress_rate(velocity, rho, eta0, dx, dy, dz)

    mean_t, std_t = _timer(_run, warmup=2, iterations=iterations)
    return {"grid": f"{n}^3", "mean_s": mean_t, "std_s": std_t}


# ------------------------------------------------------------------
# Constrained-transport benchmark
# ------------------------------------------------------------------

def benchmark_ct(n: int = 32, iterations: int = 10) -> dict:
    """Benchmark constrained transport update on an n^3 grid."""
    from dpf.fluid.constrained_transport import (
        StaggeredBField,
        ct_update,
        emf_from_fluxes,
    )

    rng = np.random.default_rng(123)
    dx = dy = dz = 1e-3
    dt = 1e-9

    Bx_face = rng.standard_normal((n + 1, n, n)) * 0.1
    By_face = rng.standard_normal((n, n + 1, n)) * 0.1
    Bz_face = rng.standard_normal((n, n, n + 1)) * 0.1
    stag = StaggeredBField(Bx_face, By_face, Bz_face, dx, dy, dz)

    Fx = rng.standard_normal((n + 1, n, n)) * 1e4
    Fy = rng.standard_normal((n, n + 1, n)) * 1e4
    Fz = rng.standard_normal((n, n, n + 1)) * 1e4

    Ex, Ey, Ez = emf_from_fluxes(Fx, Fy, Fz, dx, dy, dz)

    def _run():
        ct_update(stag, Ex, Ey, Ez, dt)

    mean_t, std_t = _timer(_run, warmup=2, iterations=iterations)
    return {"grid": f"{n}^3", "mean_s": mean_t, "std_s": std_t}


# ------------------------------------------------------------------
# Sod shock-tube benchmark
# ------------------------------------------------------------------

def benchmark_sod(nx: int = 256, iterations: int = 10) -> dict:
    """Benchmark a short Sod shock-tube run with MHDSolver."""
    from dpf.fluid.mhd_solver import MHDSolver

    ny = nz = 1
    dx = 1.0 / nx
    gamma = 1.4
    solver = MHDSolver(
        grid_shape=(nx, ny, nz),
        dx=dx,
        gamma=gamma,
        enable_hall=False,
        enable_braginskii=False,
        enable_resistive=False,
    )

    # Sod initial conditions
    rho = np.where(
        np.linspace(0, 1, nx).reshape(nx, 1, 1) < 0.5,
        1.0,
        0.125,
    )
    p = np.where(
        np.linspace(0, 1, nx).reshape(nx, 1, 1) < 0.5,
        1.0,
        0.1,
    )
    vel = np.zeros((3, nx, ny, nz))
    B = np.zeros((3, nx, ny, nz))
    Te = p / (2.0 * rho * k_B) * m_d
    Ti = Te.copy()

    state = {
        "rho": rho,
        "velocity": vel,
        "pressure": p,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": np.zeros_like(rho),
    }

    dt = 0.2 * dx

    def _run():
        s = {k: v.copy() for k, v in state.items()}
        for _ in range(5):
            s = solver.step(s, dt, current=0.0, voltage=0.0)

    mean_t, std_t = _timer(_run, warmup=1, iterations=iterations)
    return {"nx": nx, "steps": 5, "mean_s": mean_t, "std_s": std_t}


# ------------------------------------------------------------------
# Cylindrical MHD step benchmark
# ------------------------------------------------------------------

def benchmark_cylindrical_mhd(nr: int = 64, nz: int = 64, iterations: int = 10) -> dict:
    """Benchmark a single CylindricalMHDSolver step."""
    from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver

    dr = 1e-3
    dz_val = 1e-3
    solver = CylindricalMHDSolver(
        nr=nr,
        nz=nz,
        dr=dr,
        dz=dz_val,
        enable_hall=False,
        enable_resistive=False,
    )

    rho = np.full((nr, 1, nz), 1e-4)
    vel = np.zeros((3, nr, 1, nz))
    p = np.full((nr, 1, nz), 1e3)
    B = np.zeros((3, nr, 1, nz))
    B[2, :, :, :] = 0.5  # axial Bz
    Te = np.full((nr, 1, nz), 1e6)
    Ti = Te.copy()

    state = {
        "rho": rho,
        "velocity": vel,
        "pressure": p,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": np.zeros_like(rho),
    }

    dt = 1e-10

    def _run():
        s = {k: v.copy() for k, v in state.items()}
        solver.step(s, dt, current=1e5, voltage=2e4)

    mean_t, std_t = _timer(_run, warmup=1, iterations=iterations)
    return {"grid": f"{nr}x{nz}", "mean_s": mean_t, "std_s": std_t}


# ------------------------------------------------------------------
# BLAS check
# ------------------------------------------------------------------

def check_blas():
    """Print NumPy BLAS/LAPACK linkage information."""
    print("NumPy BLAS/LAPACK configuration:")
    print("-" * 50)
    try:
        np.show_config()
    except Exception as exc:
        print(f"  (could not retrieve config: {exc})")
    print()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def run_all():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("DPF Apple Silicon Performance Benchmarks")
    print(f"  Platform : {platform.platform()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Machine  : {platform.machine()}")
    print("=" * 60)
    print()

    check_blas()

    print("-" * 60)
    print("Benchmark 1: viscous_stress_rate  (64^3)")
    r = benchmark_viscosity(n=64)
    print(f"  {r['grid']:>8s}  {r['mean_s']*1e3:8.2f} ms  +/- {r['std_s']*1e3:.2f} ms")
    print()

    print("Benchmark 2: viscous_stress_rate  (128^3)")
    r = benchmark_viscosity(n=128, iterations=5)
    print(f"  {r['grid']:>8s}  {r['mean_s']*1e3:8.2f} ms  +/- {r['std_s']*1e3:.2f} ms")
    print()

    print("Benchmark 3: CT update  (32^3)")
    r = benchmark_ct(n=32)
    print(f"  {r['grid']:>8s}  {r['mean_s']*1e3:8.2f} ms  +/- {r['std_s']*1e3:.2f} ms")
    print()

    print("Benchmark 4: Sod shock tube  (nx=256, 5 steps)")
    r = benchmark_sod(nx=256)
    print(f"  nx={r['nx']}, {r['steps']} steps  {r['mean_s']*1e3:8.2f} ms  +/- {r['std_s']*1e3:.2f} ms")
    print()

    print("Benchmark 5: Cylindrical MHD step  (64x64)")
    r = benchmark_cylindrical_mhd(nr=64, nz=64)
    print(f"  {r['grid']:>8s}  {r['mean_s']*1e3:8.2f} ms  +/- {r['std_s']*1e3:.2f} ms")
    print()

    print("=" * 60)
    print("Benchmarks complete.")


if __name__ == "__main__":
    run_all()
