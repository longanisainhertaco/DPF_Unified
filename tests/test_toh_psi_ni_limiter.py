"""Tests for Toh 2025 §3.1 Eqs. (30)-(31) density-dependent slope limiter ψ(n_i).

Reference:
    Toh Y.H., Dolence J., Duraisamy K. — Asymptotic-preserving semi-implicit
    finite volume scheme for extended magnetohydrodynamics.
    [KR: asymptotic-preserving-semi-implicit-finite-volume-scheme-for-
     extended-magnetohydrodynamics-yi-han.md §3.1 Eqs. (30)-(31) lines 638-684]

Verbatim Eq. (31):
    psi(n_i) = ( 1 + exp( (lambda_0 / Delta x)^3 - n_i / L_0 ) )^(-1)

Paper-specified shock-tube parameter (line 675): (lambda_0 / Delta x)^3 = 0.005.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from dpf.metal.mlx_reconstruction import (
    _toh_psi_ni_limiter,
    plm_reconstruct,
    plm_reconstruct_toh,
    reconstruct,
)


# ------------------------------------------------------------------
# Eq. (31) sigmoid shape & limit cases
# ------------------------------------------------------------------


def test_psi_sigmoid_continuum_limit_approaches_one():
    """Toh §3.1: psi -> 1 at continuum limit (n_i / L_0 >> (lambda_0/dx)^3)."""
    n_i = mx.array([1.0e3, 1.0e6, 1.0e9])
    psi = _toh_psi_ni_limiter(n_i, lambda0_over_dx_cubed=0.005, L0=1.0)
    psi_np = np.asarray(psi)
    assert np.all(psi_np > 0.9999), f"psi should approach 1 at continuum, got {psi_np}"
    assert np.all(psi_np <= 1.0)


def test_psi_sigmoid_kinetic_limit_approaches_zero():
    """Toh §3.1: psi -> 0 at kinetic limit (arg = (lambda_0/dx)^3 - n_i/L_0 large)."""
    n_i = mx.array([0.0, 1.0e-6, 1.0e-3])
    psi = _toh_psi_ni_limiter(n_i, lambda0_over_dx_cubed=50.0, L0=1.0)
    psi_np = np.asarray(psi)
    assert np.all(psi_np < 1e-10), f"psi should approach 0 at kinetic limit, got {psi_np}"
    assert np.all(psi_np >= 0.0)


def test_psi_sigmoid_midpoint_is_half():
    """Eq. (31): when n_i / L_0 == (lambda_0/dx)^3, psi == 1/2 exactly."""
    threshold = 0.005
    n_i = mx.array([threshold])
    psi = _toh_psi_ni_limiter(n_i, lambda0_over_dx_cubed=threshold, L0=1.0)
    np.testing.assert_allclose(np.asarray(psi), 0.5, rtol=1e-6, atol=1e-7)


def test_psi_sigmoid_monotonic_increasing_in_density():
    """Eq. (31): psi is monotonically increasing in n_i."""
    n_i = mx.array(np.linspace(0.0, 10.0, 64).astype(np.float32))
    psi_np = np.asarray(_toh_psi_ni_limiter(n_i, lambda0_over_dx_cubed=1.0, L0=1.0))
    diffs = np.diff(psi_np)
    assert np.all(diffs >= -1e-7), f"psi must be non-decreasing in n_i, diffs={diffs}"
    assert np.max(diffs) > 0.0


def test_psi_sigmoid_bounded_unit_interval():
    """Eq. (31): psi is bounded in [0, 1] for all finite inputs.

    Mathematically psi is in the open interval (0, 1), but in float32 the
    sigmoid underflows/saturates to 0 or 1 at the extremes — which is
    physically the desired Toh §3.1 behavior ("approaches zero ... 1st-order
    diffusive flux"). We test the closed interval to match implementation.
    """
    n_i = mx.array(np.linspace(-100.0, 100.0, 1024).astype(np.float32))
    psi_np = np.asarray(_toh_psi_ni_limiter(n_i, lambda0_over_dx_cubed=10.0, L0=1.0))
    assert np.all(psi_np >= 0.0)
    assert np.all(psi_np <= 1.0 + 1e-7)
    assert np.all(np.isfinite(psi_np))


def test_psi_sigmoid_matches_reference_formula_pointwise():
    """Verbatim numerical match against Eq. (31) numpy reference."""
    rng = np.random.default_rng(0)
    n_arr = rng.uniform(0.0, 5.0, size=(8, 8)).astype(np.float32)
    lam = 0.005
    L0 = 1.0
    expected = 1.0 / (1.0 + np.exp(lam - n_arr / L0))
    got = np.asarray(_toh_psi_ni_limiter(mx.array(n_arr), lam, L0))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_psi_paper_shock_tube_parameter():
    """Toh §3.1 line 675: (lambda_0/Delta x)^3 = 0.005 is the shock-tube value."""
    n_i = mx.array([0.0, 0.005, 1.0, 10.0])
    got = np.asarray(_toh_psi_ni_limiter(n_i, lambda0_over_dx_cubed=0.005, L0=1.0))
    expected = 1.0 / (1.0 + np.exp(0.005 - np.array([0.0, 0.005, 1.0, 10.0])))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_psi_L0_normalization_invariant_under_joint_scaling():
    """Toh §3.1 lines 680-683: only the ratio n_i/L_0 enters Eq. (31)."""
    psi_ref = _toh_psi_ni_limiter(
        mx.array([1.0, 2.0, 5.0]), lambda0_over_dx_cubed=0.005, L0=1.0
    )
    psi_scaled = _toh_psi_ni_limiter(
        mx.array([100.0, 200.0, 500.0]), lambda0_over_dx_cubed=0.005, L0=100.0
    )
    np.testing.assert_allclose(np.asarray(psi_ref), np.asarray(psi_scaled), rtol=1e-6)


# ------------------------------------------------------------------
# Eq. (30) reconstruction wiring
# ------------------------------------------------------------------


def _make_smooth_state(nvar: int = 8, nr: int = 16, nz: int = 4) -> mx.array:
    rng = np.random.default_rng(42)
    Q = rng.uniform(0.5, 1.5, size=(nvar, nr, nz)).astype(np.float32)
    Q[0] = 1.0e6  # Density at continuum scale -> psi ~ 1.
    return mx.array(Q)


def test_eq30_recovers_plain_plm_at_continuum_limit():
    """Toh Eq. (30): when psi(n_i) -> 1, reconstruction reduces to standard PLM."""
    Q = _make_smooth_state()
    QL_plain, QR_plain = plm_reconstruct(Q, dim=0, limiter="mc")
    QL_toh, QR_toh = plm_reconstruct_toh(
        Q, dim=0, density_index=0, limiter="mc",
        lambda0_over_dx_cubed=0.005, L0=1.0,
    )
    np.testing.assert_allclose(np.asarray(QL_toh), np.asarray(QL_plain), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(QR_toh), np.asarray(QR_plain), rtol=1e-5, atol=1e-6)


def test_eq30_falls_back_to_first_order_at_kinetic_limit():
    """Toh Eq. (30): when psi -> 0, faces equal cell-center values (1st-order)."""
    Q = _make_smooth_state()
    Q_np = np.asarray(Q).copy()
    Q_np[0] = 1.0e-12
    Q_kinetic = mx.array(Q_np)

    QL_toh, QR_toh = plm_reconstruct_toh(
        Q_kinetic, dim=0, density_index=0, limiter="mc",
        lambda0_over_dx_cubed=50.0, L0=1.0,
    )
    Q_left = Q_np[:, :-1, :]
    Q_right = Q_np[:, 1:, :]
    np.testing.assert_allclose(np.asarray(QL_toh), Q_left, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(QR_toh), Q_right, rtol=1e-5, atol=1e-7)


def test_dispatch_opt_in_only_not_default():
    """toh_psi_ni must be opt-in. Default `reconstruct` is weno5z, not Toh."""
    Q = _make_smooth_state(nr=8)
    QL_default, QR_default = reconstruct(Q, dim=0)
    assert QL_default.shape[1] >= 1
    assert QR_default.shape[1] >= 1

    QL_toh, QR_toh = reconstruct(
        Q, dim=0, method="toh_psi_ni", density_index=0, limiter="mc",
        lambda0_over_dx_cubed=0.005, L0=1.0,
    )
    assert QL_toh.shape == QR_toh.shape


def test_dispatch_unknown_method_raises():
    Q = _make_smooth_state()
    with pytest.raises(ValueError, match="Unknown reconstruction method"):
        reconstruct(Q, dim=0, method="not_a_method")


def test_plm_toh_shape_contract_radial():
    Q = _make_smooth_state(nr=16, nz=4)
    QL, QR = plm_reconstruct_toh(Q, dim=0, density_index=0)
    assert QL.shape == (8, 15, 4)
    assert QR.shape == (8, 15, 4)


def test_plm_toh_shape_contract_axial():
    Q = _make_smooth_state(nr=4, nz=16)
    QL, QR = plm_reconstruct_toh(Q, dim=1, density_index=0)
    assert QL.shape == (8, 4, 15)
    assert QR.shape == (8, 4, 15)
