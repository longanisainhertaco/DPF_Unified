"""Tests for Metal compute kernels (mlx_kernels.py).

Covers NumPy reference implementations unconditionally, and MLX Metal kernel
parity tests when MLX is available. All tests run at three representative
grid sizes: (16, 32), (32, 64), (64, 128).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.metal.mlx_kernels import (
    GAMMA,
    IBR,
    IBT,
    IBZ,
    IDN,
    IEN,
    IMR,
    IMT,
    ISR,
    MU0,
    NVAR,
    cylindrical_source_mlx,
    cylindrical_source_numpy,
    ghost_pad_mlx,
    ghost_pad_numpy,
    hlld_flux_mlx,
    hlld_flux_numpy,
)

# ── MLX availability ──────────────────────────────────────────
mx = pytest.importorskip("mlx.core", reason="mlx not installed")

from dpf.metal.mlx_kernels import HAS_MLX_KERNELS  # noqa: E402

# Grid sizes to exercise
GRID_SIZES = [(16, 32), (32, 64), (64, 128)]

# float32 tolerance for Metal vs NumPy comparisons
FLOAT32_TOL = 1e-5


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_uniform_state(nr: int, nz: int) -> np.ndarray:
    """Uniform non-trivial conserved state."""
    Q = np.zeros((NVAR, nr, nz), dtype=np.float32)
    Q[IDN] = 1.225
    Q[IEN] = 2.5  # p/(g-1) = 1.0 * 1.5 + ke=0
    return Q


def _make_sod_states(n_ifaces: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Sod shock tube L/R states in conserved form."""
    UL = np.zeros((NVAR, n_ifaces, nz), dtype=np.float32)
    UR = np.zeros((NVAR, n_ifaces, nz), dtype=np.float32)
    UL[IDN] = 1.0
    UL[IEN] = 1.0 / (GAMMA - 1.0) + 0.5 * 0.75**2
    UL[IBZ] = 0.75
    UR[IDN] = 0.125
    UR[IEN] = 0.1 / (GAMMA - 1.0) + 0.5 * 0.75**2
    UR[IBZ] = -0.75
    return UL, UR


def _make_brio_wu_states(n_ifaces: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Brio-Wu strong Bt discontinuity states."""
    UL = np.zeros((NVAR, n_ifaces, nz), dtype=np.float32)
    UR = np.zeros((NVAR, n_ifaces, nz), dtype=np.float32)
    UL[IDN] = 1.0
    UL[IEN] = 1.0 / (GAMMA - 1.0) + 0.5 * (0.75**2 + 1.0**2)
    UL[IMR] = 0.75
    UL[IBT] = 1.0
    UR[IDN] = 0.125
    UR[IEN] = 0.1 / (GAMMA - 1.0) + 0.5 * (0.75**2 + 1.0**2)
    UR[IMR] = 0.75
    UR[IBT] = -1.0
    return UL, UR


def _make_uniform_hlld(n_ifaces: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Identical L/R states (zero net flux expected for symmetric configs)."""
    U = np.zeros((NVAR, n_ifaces, nz), dtype=np.float32)
    U[IDN] = 1.0
    U[IEN] = 1.0 / (GAMMA - 1.0) + 0.5
    U[IMR] = 0.75  # Br
    return U.copy(), U.copy()


def _make_rotation_prim(nr: int, nz: int, vtheta: float = 100.0) -> np.ndarray:
    """Pure rotating flow: primitive state."""
    prim = np.zeros((NVAR, nr, nz), dtype=np.float32)
    prim[IDN] = 1.0
    prim[3] = vtheta  # vtheta (primitive slot 3)
    prim[4] = 1.0     # pressure
    return prim


def _cell_radii(nr: int, dr: float = 0.01) -> np.ndarray:
    return (np.arange(nr, dtype=np.float32) + 0.5) * dr


# ══════════════════════════════════════════════════════════════
# 1. Ghost Cell Padding — NumPy reference
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_numpy_outflow_copies_interior(nr: int, nz: int) -> None:
    """Outflow BC: interior should be copied unchanged."""
    Q = _make_uniform_state(nr, nz)
    ng = 2
    padded = ghost_pad_numpy(Q, ng=ng, bc_type="outflow")

    assert padded.shape == (NVAR, nr + 2 * ng, nz)
    np.testing.assert_array_equal(padded[:, ng : ng + nr, :], Q)


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_numpy_inner_reflecting(nr: int, nz: int) -> None:
    """Reflecting inner BC: radial quantities flip sign at axis."""
    Q = np.ones((NVAR, nr, nz), dtype=np.float32)
    Q[IMR] = 2.0
    Q[IBR] = 3.0
    Q[IBT] = 4.0
    Q[IMT] = 5.0
    ng = 3
    padded = ghost_pad_numpy(Q, ng=ng, bc_type="outflow")

    for ig in range(ng):
        assert np.all(padded[IMR, ig, :] <= 0), f"IMR inner ghost[{ig}] should be <= 0"
        assert np.all(padded[IBR, ig, :] <= 0), f"IBR inner ghost[{ig}] should be <= 0"
        assert np.all(padded[IBT, ig, :] <= 0), f"IBT inner ghost[{ig}] should be <= 0"
        assert np.all(padded[IMT, ig, :] <= 0), f"IMT inner ghost[{ig}] should be <= 0"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_numpy_electrode_btheta(nr: int, nz: int) -> None:
    """Electrode BC: outer ghost Btheta = mu0*I/(2*pi*r)."""
    Q = np.ones((NVAR, nr, nz), dtype=np.float32)
    ng = 3
    current = 100e3
    dr = 0.005
    r_face = _cell_radii(nr + 2 * ng, dr=dr)

    padded = ghost_pad_numpy(Q, ng=ng, bc_type="electrode", current=current, r_face=r_face)

    for ig in range(ng):
        out_idx = ng + nr + ig
        r_pos = float(r_face[out_idx])
        expected = MU0 * current / (2.0 * math.pi * r_pos)
        actual = float(padded[IBT, out_idx, 0])
        rel_err = abs(actual - expected) / (abs(expected) + 1e-30)
        assert rel_err < 1e-4, f"Btheta outer[{ig}]: expected {expected:.4e}, got {actual:.4e}"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_numpy_zero_current(nr: int, nz: int) -> None:
    """Zero current: Btheta in outer ghosts inherits zero from interior."""
    Q = np.ones((NVAR, nr, nz), dtype=np.float32)
    Q[IBT] = 0.0
    ng = 2
    padded = ghost_pad_numpy(Q, ng=ng, bc_type="electrode", current=0.0)
    assert np.max(np.abs(padded[IBT, ng + nr :, :])) < 1e-10


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_numpy_outer_wall_zeros(nr: int, nz: int) -> None:
    """Outflow BC: IMR and IBR should be zero in outer ghosts."""
    Q = np.ones((NVAR, nr, nz), dtype=np.float32)
    ng = 2
    padded = ghost_pad_numpy(Q, ng=ng, bc_type="outflow")
    np.testing.assert_array_equal(padded[IMR, ng + nr :, :], 0.0)
    np.testing.assert_array_equal(padded[IBR, ng + nr :, :], 0.0)


# ══════════════════════════════════════════════════════════════
# 2. Ghost Cell Padding — MLX vs NumPy parity
# ══════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_mlx_outflow_matches_numpy(nr: int, nz: int) -> None:
    Q = _make_uniform_state(nr, nz)
    ng = 2
    ref = ghost_pad_numpy(Q, ng=ng, bc_type="outflow")
    result = np.asarray(ghost_pad_mlx(mx.array(Q), ng=ng, bc_type="outflow"))
    mx.eval(mx.array(result))
    err = np.max(np.abs(result - ref))
    assert err < FLOAT32_TOL, f"Ghost outflow MLX vs NumPy max error = {err:.2e} for ({nr},{nz})"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_mlx_electrode_matches_numpy(nr: int, nz: int) -> None:
    Q = np.ones((NVAR, nr, nz), dtype=np.float32)
    ng = 3
    current = 100e3
    dr = 0.005
    r_face = _cell_radii(nr + 2 * ng, dr=dr)

    ref = ghost_pad_numpy(Q, ng=ng, bc_type="electrode", current=current, r_face=r_face)
    result = np.asarray(ghost_pad_mlx(mx.array(Q), ng=ng, bc_type="electrode",
                                      current=current, r_face=r_face))
    mx.eval(mx.array(result))
    err = np.max(np.abs(result - ref))
    assert err < 1e-3, f"Ghost electrode MLX vs NumPy max error = {err:.2e} for ({nr},{nz})"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_electrode_energy_consistency(nr: int, nz: int) -> None:
    """Electrode ghost cells must have positive pressure after B_theta injection.

    Root cause of the HLLD NaN blocker: ghost_pad copies conserved energy E
    from the last interior cell but injects electrode B_theta without updating
    E.  This makes p = (gamma-1)(E - KE - 0.5*B^2) go negative.  The fix
    adds the magnetic energy difference to E and enforces a minimum beta.
    """
    Q = np.zeros((NVAR, nr, nz), dtype=np.float32)
    rho_fill = 2e-5  # deuterium fill gas
    p_fill = 3.0
    Q[IDN, :, :] = rho_fill
    Q[IEN, :, :] = p_fill / (GAMMA - 1.0)

    ng = 3
    dr = 0.001
    r_face = _cell_radii(nr + 2 * ng, dr=dr)

    for current in [200e3, 500e3, 1e6]:
        padded = ghost_pad_numpy(Q, ng=ng, bc_type="electrode",
                                 current=current, r_face=r_face)
        for ig in range(ng):
            idx = ng + nr + ig
            rho = padded[IDN, idx, 0]
            E = padded[IEN, idx, 0]
            B2 = (padded[IBR, idx, 0] ** 2
                  + padded[IBZ, idx, 0] ** 2
                  + padded[IBT, idx, 0] ** 2)
            p = (GAMMA - 1.0) * (E - 0.5 * B2)
            assert rho > 0, f"Ghost rho <= 0 at I={current:.0e}A, ig={ig}"
            assert p > 0, (
                f"Ghost pressure <= 0 at I={current:.0e}A, ig={ig}: "
                f"E={E:.4e}, B^2/2={0.5*B2:.4e}, p={p:.4e}"
            )
            assert not np.isnan(p), f"Ghost pressure NaN at I={current:.0e}A"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_mlx_electrode_energy_consistency(nr: int, nz: int) -> None:
    """MLX Metal kernel: electrode ghost cells must also have positive pressure."""
    Q = np.zeros((NVAR, nr, nz), dtype=np.float32)
    rho_fill = 2e-5
    p_fill = 3.0
    Q[IDN, :, :] = rho_fill
    Q[IEN, :, :] = p_fill / (GAMMA - 1.0)

    ng = 3
    dr = 0.001
    r_face = _cell_radii(nr + 2 * ng, dr=dr)
    current = 500e3

    padded = np.asarray(ghost_pad_mlx(mx.array(Q), ng=ng, bc_type="electrode",
                                       current=current, r_face=r_face))
    for ig in range(ng):
        idx = ng + nr + ig
        E = padded[IEN, idx, 0]
        B2 = (padded[IBR, idx, 0] ** 2
              + padded[IBZ, idx, 0] ** 2
              + padded[IBT, idx, 0] ** 2)
        p = (GAMMA - 1.0) * (E - 0.5 * B2)
        assert padded[IDN, idx, 0] > 0, f"MLX ghost rho <= 0 at ig={ig}"
        assert p > 0, (
            f"MLX ghost pressure <= 0 at ig={ig}: "
            f"E={E:.4e}, B^2/2={0.5*B2:.4e}, p={p:.4e}"
        )


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_mlx_inner_bc_sign_flip(nr: int, nz: int) -> None:
    """MLX result: inner ghost IMR/IBR should be <= 0."""
    Q = np.ones((NVAR, nr, nz), dtype=np.float32)
    ng = 2
    result = np.asarray(ghost_pad_mlx(mx.array(Q), ng=ng, bc_type="outflow"))
    assert np.all(result[IMR, :ng, :] <= 0)
    assert np.all(result[IBR, :ng, :] <= 0)


# ══════════════════════════════════════════════════════════════
# 3. HLLD Flux — NumPy reference
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_numpy_uniform_no_nan(nr: int, nz: int) -> None:
    UL, UR = _make_uniform_hlld(nr, nz)
    flux = hlld_flux_numpy(UL, UR, GAMMA, 0)
    assert not np.any(np.isnan(flux)), "Uniform HLLD produced NaN"
    assert not np.any(np.isinf(flux)), "Uniform HLLD produced Inf"
    assert flux.shape == (NVAR, nr, nz)


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_numpy_sod_no_nan(nr: int, nz: int) -> None:
    UL, UR = _make_sod_states(nr, nz)
    flux = hlld_flux_numpy(UL, UR, GAMMA, 0)
    assert not np.any(np.isnan(flux)), "Sod HLLD produced NaN"
    assert not np.any(np.isinf(flux)), "Sod HLLD produced Inf"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_numpy_brio_wu_no_nan(nr: int, nz: int) -> None:
    """Brio-Wu with strong Bt discontinuity: NaN-safe discriminant required."""
    UL, UR = _make_brio_wu_states(nr, nz)
    flux = hlld_flux_numpy(UL, UR, GAMMA, 0)
    assert not np.any(np.isnan(flux)), "Brio-Wu HLLD produced NaN"
    assert not np.any(np.isinf(flux)), "Brio-Wu HLLD produced Inf"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_numpy_entropy_tracer_advection(nr: int, nz: int) -> None:
    """Entropy tracer (ISR=5) should advect with the contact wave."""
    UL = np.zeros((NVAR, nr, nz), dtype=np.float32)
    UR = np.zeros((NVAR, nr, nz), dtype=np.float32)
    for U in [UL, UR]:
        U[IDN] = 1.0
        U[IMR] = 1.0  # rho*vr = 1 (rightward)
        U[IEN] = 1.0 / (GAMMA - 1.0) + 0.5
    UL[ISR] = 2.0
    UR[ISR] = 1.0

    flux = hlld_flux_numpy(UL, UR, GAMMA, 0)
    assert not np.any(np.isnan(flux[ISR])), "Entropy flux has NaN"
    # Rightward flow: entropy flux should use left state ≈ 2*vn=2
    assert np.all(flux[ISR] > 0), "Entropy flux should be positive for rightward flow"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_numpy_axial_dim(nr: int, nz: int) -> None:
    """dim=1 (axial): solver should run without error."""
    UL, UR = _make_sod_states(nr, nz)
    flux = hlld_flux_numpy(UL, UR, GAMMA, 1)
    assert not np.any(np.isnan(flux))


# ══════════════════════════════════════════════════════════════
# 4. HLLD Flux — MLX vs NumPy parity
# ══════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_mlx_uniform_matches_numpy(nr: int, nz: int) -> None:
    UL, UR = _make_uniform_hlld(nr, nz)
    ref = hlld_flux_numpy(UL, UR, GAMMA, 0)
    result = np.asarray(hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0))
    mx.eval(mx.array(result))
    err = np.max(np.abs(result - ref))
    assert err < FLOAT32_TOL, f"HLLD uniform MLX vs NumPy max error = {err:.2e}"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_mlx_sod_no_nan(nr: int, nz: int) -> None:
    UL, UR = _make_sod_states(nr, nz)
    result = np.asarray(hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0))
    mx.eval(mx.array(result))
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_mlx_sod_matches_numpy(nr: int, nz: int) -> None:
    UL, UR = _make_sod_states(nr, nz)
    ref = hlld_flux_numpy(UL, UR, GAMMA, 0)
    result = np.asarray(hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0))
    mx.eval(mx.array(result))
    err = np.max(np.abs(result - ref))
    # float32 Metal vs float64 NumPy reference — allow wider tolerance
    assert err < 1e-3, f"HLLD Sod MLX vs NumPy max error = {err:.2e}"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_mlx_brio_wu_no_nan(nr: int, nz: int) -> None:
    UL, UR = _make_brio_wu_states(nr, nz)
    result = np.asarray(hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0))
    mx.eval(mx.array(result))
    assert not np.any(np.isnan(result)), "Brio-Wu MLX produced NaN"
    assert not np.any(np.isinf(result)), "Brio-Wu MLX produced Inf"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_mlx_entropy_tracer_no_nan(nr: int, nz: int) -> None:
    UL = np.zeros((NVAR, nr, nz), dtype=np.float32)
    UR = np.zeros((NVAR, nr, nz), dtype=np.float32)
    for U in [UL, UR]:
        U[IDN] = 1.0
        U[IMR] = 1.0
        U[IEN] = 1.0 / (GAMMA - 1.0) + 0.5
    UL[ISR] = 2.0
    UR[ISR] = 1.0
    result = np.asarray(hlld_flux_mlx(mx.array(UL), mx.array(UR), GAMMA, 0))
    mx.eval(mx.array(result))
    assert not np.any(np.isnan(result[ISR])), "MLX entropy flux has NaN"


# ══════════════════════════════════════════════════════════════
# 5. Cylindrical Source — NumPy reference
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_numpy_zero_vtheta_theta_mom(nr: int, nz: int) -> None:
    """Zero vtheta and B: S_theta = 0 everywhere."""
    prim = np.zeros((NVAR, nr, nz), dtype=np.float32)
    prim[IDN] = 1.0
    prim[4] = 1.0
    r_cell = _cell_radii(nr)
    src = cylindrical_source_numpy(prim, r_cell)
    assert np.max(np.abs(src[IMT])) < 1e-6, "S_theta should be 0 (no vtheta, no B)"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_numpy_pressure_gradient(nr: int, nz: int) -> None:
    """Uniform pressure: S_mr[r>0] = p/r, S_mr[r=0] uses L'Hopital."""
    prim = np.zeros((NVAR, nr, nz), dtype=np.float32)
    prim[IDN] = 1.0
    prim[4] = 2.0
    dr = 0.01
    r_cell = _cell_radii(nr, dr=dr)
    src = cylindrical_source_numpy(prim, r_cell)

    # Interior cells (ir >= 1): S_mr = p/r
    expected = 2.0 / r_cell[1:]  # (nr-1,)
    err = np.max(np.abs(src[_IVR := 1][1:, :] - expected[:, np.newaxis]))
    assert err < 1e-4, f"S_mr pressure term error = {err:.2e}"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_numpy_centrifugal(nr: int, nz: int) -> None:
    """Uniform rotation: S_mr includes rho*vtheta^2/r centrifugal term."""
    vtheta = 100.0
    prim = _make_rotation_prim(nr, nz, vtheta=vtheta)
    dr = 0.01
    r_cell = _cell_radii(nr, dr=dr)
    src = cylindrical_source_numpy(prim, r_cell)

    # S_mr interior: (p + rho*vtheta^2) / r (no B)
    expected = (prim[4, 2:, 0] + prim[IDN, 2:, 0] * vtheta**2) / r_cell[2:]
    err = np.max(np.abs(src[1][2:, 0] - expected))
    assert err < 1.0, f"Centrifugal S_mr error = {err:.2e}"  # float32 at 1e4 magnitudes


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_numpy_lhopital_finite(nr: int, nz: int) -> None:
    """L'Hopital at r=0: first cell uses dp/dr instead of p/r."""
    prim = np.zeros((NVAR, nr, nz), dtype=np.float32)
    prim[IDN] = 1.0
    dr = 0.01
    r_cell = _cell_radii(nr, dr=dr)
    # Linear pressure gradient
    for ir in range(nr):
        prim[4, ir, :] = 1.0 + 10.0 * r_cell[ir]

    src = cylindrical_source_numpy(prim, r_cell)
    # L'Hopital: S_mr[0] ≈ dp/dr = 10.0
    dp_dr = (float(prim[4, 1, 0]) - float(prim[4, 0, 0])) / dr
    err = abs(float(src[1][0, 0]) - dp_dr)
    assert err < 0.1, f"L'Hopital S_mr[0] = {src[1][0,0]:.4f}, dp/dr = {dp_dr:.4f}, err = {err:.2e}"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_numpy_lhopital_no_inf(nr: int, nz: int) -> None:
    """Source terms at axis must be finite (no division by r=0)."""
    prim = np.ones((NVAR, nr, nz), dtype=np.float32)
    dr = 0.01
    r_cell = _cell_radii(nr, dr=dr)
    src = cylindrical_source_numpy(prim, r_cell)
    assert np.all(np.isfinite(src)), "Cylindrical source has non-finite values at axis"


# ══════════════════════════════════════════════════════════════
# 6. Cylindrical Source — MLX vs NumPy parity
# ══════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_mlx_matches_numpy_uniform_pressure(nr: int, nz: int) -> None:
    prim = np.zeros((NVAR, nr, nz), dtype=np.float32)
    prim[IDN] = 1.0
    prim[4] = 1.0
    r_cell = _cell_radii(nr)

    ref = cylindrical_source_numpy(prim, r_cell)
    result = np.asarray(cylindrical_source_mlx(mx.array(prim), mx.array(r_cell)))
    mx.eval(mx.array(result))
    err = np.max(np.abs(result - ref))
    assert err < FLOAT32_TOL, f"CylSrc uniform MLX vs NumPy max error = {err:.2e}"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_mlx_matches_numpy_centrifugal(nr: int, nz: int) -> None:
    prim = _make_rotation_prim(nr, nz, vtheta=50.0)
    r_cell = _cell_radii(nr)

    ref = cylindrical_source_numpy(prim, r_cell)
    result = np.asarray(cylindrical_source_mlx(mx.array(prim), mx.array(r_cell)))
    mx.eval(mx.array(result))

    # Interior cells (skip first for L'Hopital differences)
    err = np.max(np.abs(result[:, 1:, :] - ref[:, 1:, :]))
    assert err < 1e-2, f"CylSrc centrifugal MLX vs NumPy error (interior) = {err:.2e}"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_mlx_lhopital_matches_numpy(nr: int, nz: int) -> None:
    prim = np.zeros((NVAR, nr, nz), dtype=np.float32)
    prim[IDN] = 1.0
    dr = 0.01
    r_cell = _cell_radii(nr, dr=dr)
    for ir in range(nr):
        prim[4, ir, :] = 1.0 + 10.0 * r_cell[ir]

    result = np.asarray(cylindrical_source_mlx(mx.array(prim), mx.array(r_cell)))
    mx.eval(mx.array(result))

    dp_dr = (float(prim[4, 1, 0]) - float(prim[4, 0, 0])) / dr
    err_axis = abs(float(result[1][0, 0]) - dp_dr)
    assert err_axis < 0.1, f"MLX L'Hopital S_mr[0] error = {err_axis:.2e}"


@pytest.mark.skipif(not HAS_MLX_KERNELS, reason="MLX Metal kernels unavailable")
@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_mlx_finite(nr: int, nz: int) -> None:
    prim = np.ones((NVAR, nr, nz), dtype=np.float32)
    r_cell = _cell_radii(nr)
    result = np.asarray(cylindrical_source_mlx(mx.array(prim), mx.array(r_cell)))
    mx.eval(mx.array(result))
    assert np.all(np.isfinite(result)), "MLX cylindrical source has non-finite values"


# ══════════════════════════════════════════════════════════════
# 7. Cross-kernel consistency
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_hlld_numpy_zero_mass_flux_at_rest(nr: int, nz: int) -> None:
    """At rest (v=0): mass flux must be zero."""
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    U[IDN] = 1.0
    U[IEN] = 2.5
    flux = hlld_flux_numpy(U, U.copy(), GAMMA, 0)
    assert np.max(np.abs(flux[IDN])) < 1e-6, "Mass flux non-zero at rest"


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_ghost_pad_numpy_shape(nr: int, nz: int) -> None:
    Q = _make_uniform_state(nr, nz)
    for ng in [2, 3]:
        out = ghost_pad_numpy(Q, ng=ng, bc_type="outflow")
        assert out.shape == (NVAR, nr + 2 * ng, nz)


@pytest.mark.parametrize("nr,nz", GRID_SIZES)
def test_cyl_source_numpy_zero_for_no_vtheta_no_B(nr: int, nz: int) -> None:
    """No vtheta, no B: only S_mr = p/r survives; S_theta=S_Bt=0."""
    prim = np.zeros((NVAR, nr, nz), dtype=np.float32)
    prim[IDN] = 1.0
    prim[4] = 0.0  # p=0 → S_mr=0 too
    r_cell = _cell_radii(nr)
    src = cylindrical_source_numpy(prim, r_cell)
    assert np.max(np.abs(src)) < 1e-10
