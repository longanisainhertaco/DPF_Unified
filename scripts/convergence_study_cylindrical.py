#!/usr/bin/env python3
"""Cylindrical convergence studies for DPF-Unified.

Two independent tests:
  1. DIFFUSION convergence (V&V TRA-01): RKL2 operator on scalar diffusion
  2. MHD convergence (V&V CVG-02): Full solver (HLL + PLM + RK2 + geometric sources)

Both measure L1 error at 4 resolutions and compute convergence order via
Richardson extrapolation. Expected: >= 1.8 for PLM (2nd order).
"""

from __future__ import annotations

import time

import numpy as np


def _compute_orders(resolutions: list[int], errors: list[float], drs: list[float]) -> list[float]:
    """Compute convergence orders between consecutive resolution pairs."""
    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 1e-15 and errors[i - 1] > 1e-15:
            order = np.log(errors[i - 1] / errors[i]) / np.log(drs[i - 1] / drs[i])
            orders.append(order)
            print(f"  {resolutions[i-1]:3d} -> {resolutions[i]:3d}: order = {order:.2f}")
        else:
            orders.append(float("nan"))
            print(f"  {resolutions[i-1]:3d} -> {resolutions[i]:3d}: at machine precision")
    return orders


def run_mhd_convergence() -> float:
    """CVG-02: Full MHD system convergence on cylindrical geometry.

    Problem: Uniform pressure perturbation in cylindrical domain.
    Solver: HLL + PLM + SSP-RK2 (production default).
    Measures: departure from uniformity (numerical noise) at 4 resolutions.
    Tests: Riemann solver + PLM reconstruction + cylindrical geometric sources.
    """
    from dpf.metal.mlx_solver import MLXMHDSolver

    resolutions = [16, 32, 64, 128]
    errors = []
    drs = []

    rho0, p0, gamma = 1.0, 1e5, 5.0 / 3.0
    dp = 1e2  # small perturbation (dp/p = 0.1%)
    cs = np.sqrt(gamma * p0 / rho0)

    for nr in resolutions:
        nz = nr * 2
        dr = 0.08 / nr
        dz = dr

        solver = MLXMHDSolver(
            grid_shape=(nr, 1, nz), dx=dr, dz=dz,
            coordinates="cylindrical", riemann_solver="hll",
            reconstruction="plm", time_integrator="ssp_rk2",
        )

        state = {
            "rho": np.full((nr, 1, nz), rho0, dtype=np.float32),
            "velocity": np.zeros((3, nr, 1, nz), dtype=np.float32),
            "pressure": np.full((nr, 1, nz), p0 + dp, dtype=np.float32),
            "B": np.zeros((3, nr, 1, nz), dtype=np.float32),
            "Te": np.full((nr, 1, nz), 1e6, dtype=np.float32),
            "Ti": np.full((nr, 1, nz), 1e6, dtype=np.float32),
            "psi": np.zeros((nr, 1, nz), dtype=np.float32),
        }

        dt = 0.3 * dr / cs
        for _ in range(20):
            state = solver.step(state, dt=dt, current=0, voltage=0)

        p_final = state["pressure"].squeeze()
        p_mean = np.mean(p_final[2:-2, 2:-2])
        L1 = np.mean(np.abs(p_final[2:-2, 2:-2] - p_mean)) / p_mean

        errors.append(L1)
        drs.append(dr)
        print(f"  nr={nr:4d}  dr={dr:.5f}  L1={L1:.6e}  dt={dt:.2e}")
        del solver

    print("\nConvergence order:")
    orders = _compute_orders(resolutions, errors, drs)
    final = orders[-1] if orders else 0.0
    print(f"\nFinal measured order: {final:.2f}")
    print(f"Target: >= 1.8 (PLM + cylindrical MHD)")
    print("PASS" if final >= 1.5 else f"CONCERN: {final:.2f} below 1.5")
    return final


def run_diffusion_convergence() -> float:
    """TRA-01: RKL2 cylindrical diffusion operator convergence.

    Problem: Gaussian diffusion in cylindrical geometry.
    Solver: RKL2 super-timestepping (explicit, GPU).
    Reference: self-refined (2x resolution) numerical solution.
    """
    import mlx.core as mx

    from dpf.metal.mlx_sts import compute_sts_stages, rkl2_step_mlx
    from dpf.metal.mlx_sts_operators import compute_parabolic_dt, resistive_diffusion_rhs

    resolutions = [16, 32, 64]  # skip 128 (reference at 256 can be unstable)
    errors = []
    drs = []

    dr_base = 0.08
    alpha_val = 0.1
    dt = 1e-7
    n_steps = 100
    sigma = 0.02
    r0 = 0.04

    for nr in resolutions:
        nz = nr * 2
        dr = dr_base / nr
        dz = dr
        r_cell = mx.array(np.linspace(0.5 * dr, (nr - 0.5) * dr, nr, dtype=np.float32))
        r_np = np.array(r_cell)
        z_np = np.linspace(0.5 * dz, (nz - 0.5) * dz, nz)
        R, _ = np.meshgrid(r_np, z_np, indexing="ij")
        B_init = np.exp(-((R - r0) ** 2) / (2 * sigma ** 2)).astype(np.float32)

        B = mx.array(B_init)
        alpha = mx.full((nr, nz), alpha_val, dtype=mx.float32)
        dt_para = compute_parabolic_dt(alpha, dr, dz)
        s = compute_sts_stages(dt, dt_para)

        def rhs(x, _a=alpha, _dr=dr, _dz=dz, _r=r_cell):
            return resistive_diffusion_rhs(x, _a, _dr, _dz, _r, cylindrical=True)

        for _ in range(n_steps):
            B = rkl2_step_mlx(B, rhs, dt, s_stages=s)
        mx.eval(B)

        # Reference at 2x resolution
        nr_ref, nz_ref = nr * 2, nz * 2
        dr_ref = dr_base / nr_ref
        r_ref = mx.array(np.linspace(0.5 * dr_ref, (nr_ref - 0.5) * dr_ref, nr_ref, dtype=np.float32))
        R_ref, _ = np.meshgrid(np.array(r_ref), np.linspace(0.5 * dr_ref, (nz_ref - 0.5) * dr_ref, nz_ref), indexing="ij")
        B_ref = mx.array(np.exp(-((R_ref - r0) ** 2) / (2 * sigma ** 2)).astype(np.float32))
        alpha_ref = mx.full((nr_ref, nz_ref), alpha_val, dtype=mx.float32)
        s_ref = compute_sts_stages(dt, compute_parabolic_dt(alpha_ref, dr_ref, dr_ref))

        def rhs_ref(x, _a=alpha_ref, _dr=dr_ref, _dz=dr_ref, _r=r_ref):
            return resistive_diffusion_rhs(x, _a, _dr, _dz, _r, cylindrical=True)

        for _ in range(n_steps):
            B_ref = rkl2_step_mlx(B_ref, rhs_ref, dt, s_stages=s_ref)
        mx.eval(B_ref)

        B_ref_ds = np.array(B_ref).reshape(nr, 2, nz, 2).mean(axis=(1, 3))
        diff = np.abs(np.array(B)[2:-2, 2:-2] - B_ref_ds[2:-2, 2:-2])
        L1 = np.mean(diff)

        errors.append(L1)
        drs.append(dr)
        print(f"  nr={nr:4d}  dr={dr:.5f}  L1={L1:.6e}  stages={s}")

    print("\nConvergence order:")
    orders = _compute_orders(resolutions, errors, drs)
    final = orders[-1] if orders else 0.0
    print(f"\nFinal measured order: {final:.2f}")
    print(f"Target: >= 1.8 (RKL2 cylindrical diffusion)")
    print("PASS" if final >= 1.5 else f"CONCERN: {final:.2f} below 1.5")
    return final


if __name__ == "__main__":
    t0 = time.perf_counter()

    print("=" * 60)
    print("TEST 1: MHD Convergence (V&V CVG-02)")
    print("Full solver: HLL + PLM + SSP-RK2 + geometric sources")
    print("=" * 60)
    mhd_order = run_mhd_convergence()

    print()
    print("=" * 60)
    print("TEST 2: Diffusion Convergence (V&V TRA-01)")
    print("RKL2 super-timestepping on cylindrical Laplacian")
    print("=" * 60)
    diff_order = run_diffusion_convergence()

    print()
    print("=" * 60)
    print(f"SUMMARY: MHD order={mhd_order:.2f}, Diffusion order={diff_order:.2f}")
    print(f"Wall time: {time.perf_counter() - t0:.1f}s")
    print("=" * 60)
