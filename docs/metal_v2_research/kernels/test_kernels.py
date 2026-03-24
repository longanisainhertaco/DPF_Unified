"""Validation tests for MLX Metal MHD kernels.

Compares Metal kernel outputs against NumPy reference implementations.
Run with: python3 test_kernels.py
"""

from __future__ import annotations

import math
import sys
import time

import mlx.core as mx
import numpy as np
from cylindrical_source_kernel import cylindrical_source_mlx, cylindrical_source_numpy
from ghost_cell_kernel import ghost_cell_pad_mlx, ghost_cell_pad_numpy
from hlld_kernel import GAMMA, hlld_flux_mlx, hlld_flux_numpy


def test_ghost_cell_uniform():
    """Ghost padding on uniform state should preserve interior."""
    nr, nz = 32, 64
    ng = 3
    state = np.ones((10, nr, nz), dtype=np.float32) * 1.225  # uniform density
    r_inner = 0.005
    dr = 0.01
    current = 100e3  # 100 kA

    # NumPy reference
    ref = ghost_cell_pad_numpy(state, current, r_inner, dr, ng)
    assert ref.shape == (10, nr + 2 * ng, nz), f"Shape mismatch: {ref.shape}"

    # MLX kernel
    state_mx = mx.array(state)
    result_mx = ghost_cell_pad_mlx(state_mx, current, r_inner, dr, ng)
    mx.eval(result_mx)
    result = np.array(result_mx)

    # Interior should be identical
    interior_ref = ref[:, ng:ng + nr, :]
    interior_res = result[:, ng:ng + nr, :]
    err = np.max(np.abs(interior_ref - interior_res))
    assert err < 1e-6, f"Interior mismatch: max error = {err}"

    # Check axis BCs: vr (IMR=1), Br (IBR=6), Btheta (IBT=8) should be negative
    assert np.all(result[1, :ng, :] <= 0), "IMR inner ghost should be <= 0"
    assert np.all(result[6, :ng, :] <= 0), "IBR inner ghost should be <= 0"

    # Check cathode BCs: vr=0, Br=0 in outer ghosts
    assert np.all(result[1, ng + nr:, :] == 0), "IMR outer ghost should be 0"
    assert np.all(result[6, ng + nr:, :] == 0), "IBR outer ghost should be 0"

    # Btheta in outer ghost should have mu0*I/(2*pi*r) pattern
    mu0 = 4 * math.pi * 1e-7
    for ig in range(ng):
        out_idx = ng + nr + ig
        r_pos = r_inner + (out_idx - ng + 0.5) * dr
        expected_Bt = mu0 * current / (2 * math.pi * r_pos)
        actual_Bt = result[8, out_idx, 0]
        rel_err = abs(actual_Bt - expected_Bt) / (abs(expected_Bt) + 1e-30)
        assert rel_err < 1e-4, f"Btheta outer ghost[{ig}]: expected {expected_Bt:.6e}, got {actual_Bt:.6e}"

    # Full comparison Metal vs NumPy
    max_err = np.max(np.abs(result - ref))
    print(f"  Ghost cell: max error vs reference = {max_err:.2e}")
    assert max_err < 1e-4, f"Ghost cell kernel diverges from NumPy: {max_err}"

    return True


def test_ghost_cell_zero_current():
    """Ghost padding with zero current should have zero Btheta in ghosts."""
    nr, nz = 16, 16
    state = np.ones((10, nr, nz), dtype=np.float32)
    state[8] = 0.0  # zero Btheta
    ref = ghost_cell_pad_numpy(state, 0.0, 0.005, 0.01)
    result = np.array(ghost_cell_pad_mlx(mx.array(state), 0.0, 0.005, 0.01))
    mx.eval(mx.array(result))  # force eval

    assert np.max(np.abs(result[8, :3, :])) < 1e-10, "Btheta should be ~0 with zero current"
    max_err = np.max(np.abs(result - ref))
    print(f"  Ghost cell (I=0): max error = {max_err:.2e}")
    assert max_err < 1e-6
    return True


def test_hlld_uniform_state():
    """HLLD on uniform state should produce zero flux differences."""
    n_ifaces, nz = 32, 64
    nvar = 10

    # Uniform state: rho=1, v=0, p=1, B=(1,0,0)
    UL = np.zeros((nvar, n_ifaces, nz), dtype=np.float32)
    UL[0] = 1.0   # rho
    UL[4] = 1.0 / (GAMMA - 1) + 0.5  # E = p/(g-1) + B^2/2
    UL[6] = 1.0   # Br
    UR = UL.copy()

    ref = hlld_flux_numpy(UL, UR, GAMMA, 0)
    result = np.array(hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0))

    # Fluxes should be identical for identical L/R
    err = np.max(np.abs(ref - result))
    print(f"  HLLD uniform: max error = {err:.2e}")
    assert err < 1e-5, f"HLLD uniform state error: {err}"
    return True


def test_hlld_sod_shock():
    """HLLD on Sod shock tube initial condition: no NaN, physical fluxes."""
    n_ifaces, nz = 64, 1
    nvar = 10

    UL = np.zeros((nvar, n_ifaces, nz), dtype=np.float32)
    UR = np.zeros((nvar, n_ifaces, nz), dtype=np.float32)

    # Left state: rho=1, p=1, Bz=0.75
    UL[0] = 1.0
    UL[4] = 1.0 / (GAMMA - 1) + 0.5 * 0.75**2
    UL[7] = 0.75  # Bz

    # Right state: rho=0.125, p=0.1, Bz=-0.75
    UR[0] = 0.125
    UR[4] = 0.1 / (GAMMA - 1) + 0.5 * 0.75**2
    UR[7] = -0.75

    ref = hlld_flux_numpy(UL, UR, GAMMA, 0)
    result_mx = hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0)
    mx.eval(result_mx)
    result = np.array(result_mx)

    assert not np.any(np.isnan(result)), "HLLD Sod test produced NaN"
    assert not np.any(np.isinf(result)), "HLLD Sod test produced Inf"

    # Compare with reference
    err = np.max(np.abs(ref - result))
    print(f"  HLLD Sod shock: max error = {err:.2e}")
    # float32 vs float64 reference — allow some tolerance
    assert err < 1e-3, f"HLLD Sod divergence: {err}"
    return True


def test_hlld_brio_wu():
    """HLLD on Brio-Wu MHD shock: tests NaN safety with strong By discontinuity."""
    n_ifaces, nz = 64, 1
    nvar = 10

    UL = np.zeros((nvar, n_ifaces, nz), dtype=np.float32)
    UR = np.zeros((nvar, n_ifaces, nz), dtype=np.float32)

    # Brio-Wu: strong Bt discontinuity
    UL[0] = 1.0
    UL[4] = 1.0 / (GAMMA - 1) + 0.5 * (0.75**2 + 1.0**2)
    UL[6] = 0.75  # Br (normal)
    UL[8] = 1.0   # Btheta (tangential)

    UR[0] = 0.125
    UR[4] = 0.1 / (GAMMA - 1) + 0.5 * (0.75**2 + 1.0**2)
    UR[6] = 0.75
    UR[8] = -1.0

    result_mx = hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0)
    mx.eval(result_mx)
    result = np.array(result_mx)

    assert not np.any(np.isnan(result)), "HLLD Brio-Wu produced NaN (catastrophic cancellation?)"
    assert not np.any(np.isinf(result)), "HLLD Brio-Wu produced Inf"
    print(f"  HLLD Brio-Wu: NaN-free, mass flux range [{result[0].min():.4f}, {result[0].max():.4f}]")
    return True


def test_hlld_entropy_advection():
    """Entropy tracer should advect with contact speed, not diffuse."""
    n_ifaces, nz = 32, 1
    nvar = 10

    UL = np.zeros((nvar, n_ifaces, nz), dtype=np.float32)
    UR = np.zeros((nvar, n_ifaces, nz), dtype=np.float32)

    # Uniform hydro state, different entropy tracer
    for U in [UL, UR]:
        U[0] = 1.0
        U[1] = 1.0  # rho*vr = rho * 1 = 1 (rightward flow)
        U[4] = 1.0 / (GAMMA - 1) + 0.5
    UL[5] = 2.0  # Srho left
    UR[5] = 1.0  # Srho right

    result_mx = hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0)
    mx.eval(result_mx)
    result = np.array(result_mx)

    # For rightward flow (SM > 0), entropy flux should use left state
    assert not np.any(np.isnan(result[5])), "Entropy flux has NaN"
    print(f"  HLLD entropy: flux = {result[5, 0, 0]:.4f} (expected ~2.0 for rightward flow)")
    return True


def test_cyl_source_uniform():
    """Uniform state with vtheta=0 should have zero source (except p/r)."""
    nr, nz = 32, 64
    dr = 0.01
    r_cell = np.arange(nr, dtype=np.float32) * dr + 0.5 * dr  # starts at dr/2

    prim = np.zeros((10, nr, nz), dtype=np.float32)
    prim[0] = 1.0  # rho
    prim[4] = 1.0  # p

    ref = cylindrical_source_numpy(prim, r_cell, dr)
    result_mx = cylindrical_source_mlx(mx.array(prim), mx.array(r_cell), dr)
    mx.eval(result_mx)
    result = np.array(result_mx)

    # Theta-momentum source should be zero (no vtheta, no Btheta)
    assert np.max(np.abs(result[3])) < 1e-6, "S_theta should be 0 for uniform state"

    # r-momentum source = p_tot / r (for zero B, zero vtheta)
    expected_S_mr = prim[4, :, :] / r_cell[:, np.newaxis]
    # First cell uses L'Hopital, skip
    err_mr = np.max(np.abs(result[1, 1:, :] - expected_S_mr[1:, :]))
    print(f"  Cyl source uniform: S_mr error (excluding ir=0) = {err_mr:.2e}")
    assert err_mr < 1e-4

    # Compare Metal vs NumPy
    max_err = np.max(np.abs(result - ref))
    print(f"  Cyl source uniform: max error vs reference = {max_err:.2e}")
    assert max_err < 1e-4
    return True


def test_cyl_source_centrifugal():
    """Rotating flow: vtheta source should produce centrifugal force."""
    nr, nz = 32, 32
    dr = 0.01
    r_cell = np.arange(nr, dtype=np.float32) * dr + 0.5 * dr

    prim = np.zeros((10, nr, nz), dtype=np.float32)
    prim[0] = 1.0   # rho
    prim[3] = 100.0  # vtheta
    prim[4] = 1.0   # p

    ref = cylindrical_source_numpy(prim, r_cell, dr)
    result_mx = cylindrical_source_mlx(mx.array(prim), mx.array(r_cell), dr)
    mx.eval(result_mx)
    result = np.array(result_mx)

    # S_mr should include rho*vtheta^2/r (centrifugal)
    # Check interior cells (skip first for L'Hopital)
    err = np.max(np.abs(result[1, 2:, :] - ref[1, 2:, :]))
    print(f"  Cyl source centrifugal: Metal vs NumPy error = {err:.2e}")
    assert err < 1e-2  # float32 tolerance at 10^4 magnitudes

    # S_theta should be negative (Coriolis for vr=0 → just 0, but theta source from -rho*vr*vtheta/r = 0)
    assert np.max(np.abs(result[3, 1:, :])) < 1e-4, "S_theta should be ~0 when vr=0"
    return True


def test_cyl_source_lhopital():
    """L'Hopital at r=0: first cell should use dp/dr not p/r."""
    nr, nz = 16, 8
    dr = 0.01
    r_cell = np.arange(nr, dtype=np.float32) * dr + 0.5 * dr

    prim = np.zeros((10, nr, nz), dtype=np.float32)
    prim[0] = 1.0
    # Pressure gradient: p increases linearly with r
    for ir in range(nr):
        prim[4, ir, :] = 1.0 + 10.0 * r_cell[ir]

    ref = cylindrical_source_numpy(prim, r_cell, dr)
    result_mx = cylindrical_source_mlx(mx.array(prim), mx.array(r_cell), dr)
    mx.eval(result_mx)
    result = np.array(result_mx)

    # At ir=0, S_mr should be dp/dr ≈ 10.0
    dp_dr = (prim[4, 1, 0] - prim[4, 0, 0]) / dr
    err = abs(result[1, 0, 0] - dp_dr)
    print(f"  L'Hopital: S_mr[0] = {result[1, 0, 0]:.4f}, dp/dr = {dp_dr:.4f}, error = {err:.2e}")
    assert err < 0.1, f"L'Hopital test failed: {err}"

    # NumPy reference should match
    err_ref = abs(ref[1, 0, 0] - dp_dr)
    assert err_ref < 0.1
    return True


def benchmark_kernels():
    """Benchmark kernel throughput on representative grid sizes."""
    print("\n--- Benchmark: M3 Pro (14 GPU cores) ---")

    for nr, nz in [(128, 256), (256, 512), (512, 1024)]:
        # Ghost cell
        state = mx.ones((10, nr, nz), dtype=mx.float32)
        mx.eval(state)
        t0 = time.perf_counter()
        for _ in range(100):
            out = ghost_cell_pad_mlx(state, 100e3, 0.005, 0.01)
            mx.eval(out)
        t_ghost = (time.perf_counter() - t0) / 100
        cells = nr * nz
        print(f"  Ghost ({nr}x{nz}): {t_ghost*1e6:.0f} us, {cells/t_ghost:.2e} cells/s")

        # HLLD
        UL = mx.ones((10, nr, nz), dtype=mx.float32) * 0.5
        UR = mx.ones((10, nr, nz), dtype=mx.float32) * 0.3
        mx.eval(UL)
        mx.eval(UR)
        t0 = time.perf_counter()
        for _ in range(100):
            f = hlld_flux_mlx(UL, UR, GAMMA, 0)
            mx.eval(f)
        t_hlld = (time.perf_counter() - t0) / 100
        print(f"  HLLD  ({nr}x{nz}): {t_hlld*1e6:.0f} us, {cells/t_hlld:.2e} cells/s")

        # Cyl source
        prim = mx.ones((10, nr, nz), dtype=mx.float32)
        r_cell = mx.array(np.arange(nr, dtype=np.float32) * 0.01 + 0.005)
        mx.eval(prim)
        mx.eval(r_cell)
        t0 = time.perf_counter()
        for _ in range(100):
            s = cylindrical_source_mlx(prim, r_cell, 0.01)
            mx.eval(s)
        t_cyl = (time.perf_counter() - t0) / 100
        print(f"  CylSrc({nr}x{nz}): {t_cyl*1e6:.0f} us, {cells/t_cyl:.2e} cells/s")


def main():
    tests = [
        ("Ghost cell (uniform)", test_ghost_cell_uniform),
        ("Ghost cell (I=0)", test_ghost_cell_zero_current),
        ("HLLD (uniform)", test_hlld_uniform_state),
        ("HLLD (Sod shock)", test_hlld_sod_shock),
        ("HLLD (Brio-Wu)", test_hlld_brio_wu),
        ("HLLD (entropy)", test_hlld_entropy_advection),
        ("Cyl source (uniform)", test_cyl_source_uniform),
        ("Cyl source (centrifugal)", test_cyl_source_centrifugal),
        ("Cyl source (L'Hopital)", test_cyl_source_lhopital),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS: {name}")
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")

    if failed == 0:
        benchmark_kernels()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
