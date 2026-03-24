"""Tests for MLX WENO5-Z and PLM reconstruction module.

All tests are skipped when MLX is not installed (Apple Silicon only).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mlx = pytest.importorskip("mlx")  # noqa: E402
mx = pytest.importorskip("mlx.core")  # noqa: E402

from dpf.metal.mlx_reconstruction import (  # noqa: E402, I001
    plm_reconstruct,
    reconstruct,
    weno5z_reconstruct,
)

# ============================================================
# Helpers
# ============================================================

NVAR = 5  # number of state variables in tests


def _make_uniform(nr: int, nz: int, value: float = 1.0) -> mx.array:
    """Uniform state Q[v, r, z] = value."""
    return mx.ones((NVAR, nr, nz), dtype=mx.float32) * value


def _make_linear(nr: int, nz: int, dim: int) -> mx.array:
    """Linear profile along the specified dim."""
    if dim == 0:
        coords = mx.arange(nr, dtype=mx.float32)  # shape (nr,)
        profile = coords[None, :, None]  # (1, nr, 1)
    else:
        coords = mx.arange(nz, dtype=mx.float32)
        profile = coords[None, None, :]  # (1, 1, nz)
    return mx.broadcast_to(profile, (NVAR, nr, nz))


def _sine_wave(n: int) -> np.ndarray:
    """One period of a smooth sine over n cells."""
    x = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.sin(x)


def _l1_error(pred: mx.array, ref: mx.array) -> float:
    """Mean absolute error between two arrays."""
    mx.eval(pred, ref)
    return float(mx.mean(mx.abs(pred - ref)).item())


# ============================================================
# Output shape tests
# ============================================================


class TestOutputShapes:
    def test_plm_dim0_shape(self) -> None:
        nr, nz = 16, 12
        Q = _make_uniform(nr, nz)
        QL, QR = plm_reconstruct(Q, dim=0)
        assert QL.shape == (NVAR, nr - 1, nz), f"QL shape {QL.shape}"
        assert QR.shape == (NVAR, nr - 1, nz), f"QR shape {QR.shape}"

    def test_plm_dim1_shape(self) -> None:
        nr, nz = 12, 20
        Q = _make_uniform(nr, nz)
        QL, QR = plm_reconstruct(Q, dim=1)
        assert QL.shape == (NVAR, nr, nz - 1)
        assert QR.shape == (NVAR, nr, nz - 1)

    def test_weno5z_dim0_shape(self) -> None:
        nr, nz = 16, 12
        Q = _make_uniform(nr, nz)
        QL, QR = weno5z_reconstruct(Q, dim=0)
        # Full WENO5 interior only: n-5 interfaces (both QL and QR need full stencil)
        assert QL.shape == (NVAR, nr - 5, nz), f"QL shape {QL.shape}"
        assert QR.shape == (NVAR, nr - 5, nz)

    def test_weno5z_dim1_shape(self) -> None:
        nr, nz = 12, 20
        Q = _make_uniform(nr, nz)
        QL, QR = weno5z_reconstruct(Q, dim=1)
        assert QL.shape == (NVAR, nr, nz - 5)
        assert QR.shape == (NVAR, nr, nz - 5)

    def test_weno5z_fallback_to_plm_small_grid(self) -> None:
        """Fewer than 6 cells → PLM fallback, n-1 interfaces."""
        nr, nz = 5, 10
        Q = _make_uniform(nr, nz)
        QL, QR = weno5z_reconstruct(Q, dim=0)
        assert QL.shape == (NVAR, nr - 1, nz)

    def test_dispatch_weno5z(self) -> None:
        Q = _make_uniform(10, 10)
        QL1, QR1 = reconstruct(Q, dim=0, method="weno5z")
        QL2, QR2 = weno5z_reconstruct(Q, dim=0)
        mx.eval(QL1, QL2, QR1, QR2)
        np.testing.assert_allclose(np.array(QL1.tolist()), np.array(QL2.tolist()), atol=1e-6)

    def test_dispatch_plm(self) -> None:
        Q = _make_uniform(10, 10)
        QL1, QR1 = reconstruct(Q, dim=0, method="plm")
        QL2, QR2 = plm_reconstruct(Q, dim=0)
        mx.eval(QL1, QL2, QR1, QR2)
        np.testing.assert_allclose(np.array(QL1.tolist()), np.array(QL2.tolist()), atol=1e-6)

    def test_dispatch_unknown_raises(self) -> None:
        Q = _make_uniform(10, 10)
        with pytest.raises(ValueError, match="Unknown reconstruction"):
            reconstruct(Q, dim=0, method="ppm")


# ============================================================
# Uniform state preservation
# ============================================================


class TestUniformState:
    """A constant field must reconstruct exactly to that constant."""

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("method", ["plm", "weno5z"])
    def test_uniform_preserved(self, dim: int, method: str) -> None:
        nr, nz = 16, 16
        val = 3.14
        Q = _make_uniform(nr, nz, val)
        QL, QR = reconstruct(Q, dim=dim, method=method)
        mx.eval(QL, QR)
        QL_np = np.array(QL.tolist())
        QR_np = np.array(QR.tolist())
        np.testing.assert_allclose(QL_np, val, atol=1e-5, err_msg="QL != val for uniform")
        np.testing.assert_allclose(QR_np, val, atol=1e-5, err_msg="QR != val for uniform")


# ============================================================
# Linear profile exactness
# ============================================================


class TestLinearProfile:
    """Both PLM and WENO5-Z are exact on linear profiles at interior interfaces.

    PLM has zero slope at boundary cells (i=0 and i=n-1), so only interfaces
    [1 .. n-3] are exactly reconstructed. WENO5-Z returns n-4 interfaces, all
    of which are interior and therefore exact.
    """

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("method", ["plm", "weno5z"])
    def test_linear_exact(self, dim: int, method: str) -> None:
        nr, nz = 16, 16
        Q = _make_linear(nr, nz, dim)
        QL, QR = reconstruct(Q, dim=dim, method=method)
        mx.eval(QL, QR)

        axis = dim + 1
        n = Q.shape[axis]

        QL_np = np.array(QL.tolist())
        QR_np = np.array(QR.tolist())

        if method == "weno5z" and n >= 6:
            # All n-5 output interfaces are interior → exact
            # Interface j (0-indexed) is between cells j+2 and j+3 → position j+2.5
            n_iface = QL.shape[axis]
            expected = np.arange(n_iface, dtype=np.float32) + 2.5  # j+2 + 0.5
            if dim == 0:
                expected_bc = np.broadcast_to(
                    expected[np.newaxis, :, np.newaxis], QL_np.shape
                )
            else:
                expected_bc = np.broadcast_to(
                    expected[np.newaxis, np.newaxis, :], QL_np.shape
                )
            np.testing.assert_allclose(QL_np, expected_bc, atol=1e-3,
                                       err_msg=f"QL linear weno5z dim={dim}")
            np.testing.assert_allclose(QR_np, expected_bc, atol=1e-3,
                                       err_msg=f"QR linear weno5z dim={dim}")
        else:
            # PLM: only interior interfaces [1 .. n-3] are exact (skip boundary)
            n_iface = QL.shape[axis]
            if dim == 0:
                ql_interior = QL_np[:, 1:-1, :]
                qr_interior = QR_np[:, 1:-1, :]
            else:
                ql_interior = QL_np[:, :, 1:-1]
                qr_interior = QR_np[:, :, 1:-1]
            n_int = ql_interior.shape[axis]
            expected_int = np.arange(1, 1 + n_int, dtype=np.float32) + 0.5
            if dim == 0:
                expected_bc = np.broadcast_to(
                    expected_int[np.newaxis, :, np.newaxis], ql_interior.shape
                )
            else:
                expected_bc = np.broadcast_to(
                    expected_int[np.newaxis, np.newaxis, :], ql_interior.shape
                )
            np.testing.assert_allclose(ql_interior, expected_bc, atol=1e-3,
                                       err_msg=f"QL interior linear plm dim={dim}")
            np.testing.assert_allclose(qr_interior, expected_bc, atol=1e-3,
                                       err_msg=f"QR interior linear plm dim={dim}")


# ============================================================
# Smooth sine wave convergence order
# ============================================================


class TestConvergenceOrder:
    """Verify reconstruction order on smooth data.

    PLM → ~2nd order, WENO5-Z → ~5th order.
    We measure error at the reconstructed left-face value vs the exact
    interface value (midpoint between cells).
    """

    def _run_convergence(
        self, method: str, dim: int, ns: list[int]
    ) -> list[float]:
        errors = []
        for n in ns:
            # Build 1D sine over n cells, embedded in 2D (NVAR, n, 8) or (NVAR, 8, n)
            x = np.linspace(0, 2 * math.pi, n, endpoint=False)
            q1d = np.sin(x).astype(np.float32)

            if dim == 0:
                Q_np = np.tile(q1d[np.newaxis, :, np.newaxis], (NVAR, 1, 4))
            else:
                Q_np = np.tile(q1d[np.newaxis, np.newaxis, :], (NVAR, 4, 1))

            Q = mx.array(Q_np)
            QL, _ = reconstruct(Q, dim=dim, method=method)
            mx.eval(QL)

            # Exact interface values at x_i + dx/2
            dx = 2 * math.pi / n
            if method == "weno5z" and n >= 6:
                offset = 2
            else:
                offset = 0
            n_iface = QL.shape[dim + 1]
            x_iface = (np.arange(offset, offset + n_iface) + 0.5) * dx
            exact = np.sin(x_iface).astype(np.float32)

            QL_np = np.array(QL.tolist())
            if dim == 0:
                pred_1d = QL_np[0, :, 0]
            else:
                pred_1d = QL_np[0, 0, :]

            err = float(np.mean(np.abs(pred_1d - exact)))
            errors.append(err)
        return errors

    @pytest.mark.parametrize("dim", [0, 1])
    def test_plm_order_approx_2(self, dim: int) -> None:
        ns = [32, 64, 128]
        errors = self._run_convergence("plm", dim, ns)
        orders = [
            math.log(errors[i] / errors[i + 1]) / math.log(ns[i + 1] / ns[i])
            for i in range(len(ns) - 1)
        ]
        for order in orders:
            assert order > 1.5, f"PLM order {order:.2f} < 1.5 (expected ~2nd)"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_weno5z_order_approx_5(self, dim: int) -> None:
        # Use n=16,32,64 to stay above the float32 saturation floor (~3e-8).
        # At n=128 the FD WENO5-Z error is already at float32 machine epsilon,
        # so convergence rate collapses to ~1 there.
        ns = [16, 32, 64]
        errors = self._run_convergence("weno5z", dim, ns)
        orders = [
            math.log(errors[i] / errors[i + 1]) / math.log(ns[i + 1] / ns[i])
            for i in range(len(ns) - 1)
        ]
        for order in orders:
            assert order > 3.5, f"WENO5-Z order {order:.2f} < 3.5 (expected ~5th)"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_weno5z_lower_error_than_plm(self, dim: int) -> None:
        """WENO5-Z must have lower L1 error than PLM at the same resolution."""
        n = 32
        errs_weno = self._run_convergence("weno5z", dim, [n])
        errs_plm = self._run_convergence("plm", dim, [n])
        assert errs_weno[0] < errs_plm[0], (
            f"WENO5-Z error {errs_weno[0]:.2e} >= PLM error {errs_plm[0]:.2e} at n={n}"
        )


# ============================================================
# Discontinuity — no oscillation
# ============================================================


class TestDiscontinuity:
    """Step function: reconstructed values must stay monotone between 0 and 1."""

    def _step_Q(self, n: int, dim: int) -> mx.array:
        if dim == 0:
            q1d = np.where(np.arange(n) < n // 2, 0.0, 1.0).astype(np.float32)
            Q_np = np.tile(q1d[np.newaxis, :, np.newaxis], (NVAR, 1, 4))
        else:
            q1d = np.where(np.arange(n) < n // 2, 0.0, 1.0).astype(np.float32)
            Q_np = np.tile(q1d[np.newaxis, np.newaxis, :], (NVAR, 4, 1))
        return mx.array(Q_np)

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("method", ["plm", "weno5z"])
    def test_step_no_oscillation(self, dim: int, method: str) -> None:
        Q = self._step_Q(32, dim)
        QL, QR = reconstruct(Q, dim=dim, method=method)
        mx.eval(QL, QR)
        QL_np = np.array(QL.tolist())
        QR_np = np.array(QR.tolist())

        assert np.all(QL_np >= -1e-5), f"QL has undershoot: min={QL_np.min():.3e}"
        assert np.all(QL_np <= 1.0 + 1e-5), f"QL has overshoot: max={QL_np.max():.3e}"
        assert np.all(QR_np >= -1e-5), f"QR has undershoot: min={QR_np.min():.3e}"
        assert np.all(QR_np <= 1.0 + 1e-5), f"QR has overshoot: max={QR_np.max():.3e}"


# ============================================================
# PLM limiter comparison
# ============================================================


class TestPLMLimiters:
    def test_minmod_vs_mc_uniform(self) -> None:
        Q = _make_uniform(16, 16, 2.0)
        QL_mc, QR_mc = plm_reconstruct(Q, dim=0, limiter="mc")
        QL_mm, QR_mm = plm_reconstruct(Q, dim=0, limiter="minmod")
        mx.eval(QL_mc, QR_mc, QL_mm, QR_mm)
        np.testing.assert_allclose(
            np.array(QL_mc.tolist()), np.array(QL_mm.tolist()), atol=1e-6
        )

    def test_mc_less_diffusive_than_minmod(self) -> None:
        """MC limiter should have lower L1 error on smooth data than minmod."""
        n = 32
        x = np.linspace(0, 2 * math.pi, n, endpoint=False)
        q1d = np.sin(x).astype(np.float32)
        Q_np = np.tile(q1d[np.newaxis, :, np.newaxis], (NVAR, 1, 4))
        Q = mx.array(Q_np)

        QL_mc, _ = plm_reconstruct(Q, dim=0, limiter="mc")
        QL_mm, _ = plm_reconstruct(Q, dim=0, limiter="minmod")
        mx.eval(QL_mc, QL_mm)

        dx = 2 * math.pi / n
        x_iface = (np.arange(n - 1) + 0.5) * dx
        exact = np.sin(x_iface).astype(np.float32)

        err_mc = float(np.mean(np.abs(np.array(QL_mc.tolist())[0, :, 0] - exact)))
        err_mm = float(np.mean(np.abs(np.array(QL_mm.tolist())[0, :, 0] - exact)))

        assert err_mc < err_mm, f"MC err={err_mc:.3e} not < minmod err={err_mm:.3e}"

    def test_plm_too_few_cells_raises(self) -> None:
        Q = _make_uniform(1, 10)
        with pytest.raises(ValueError, match="PLM requires"):
            plm_reconstruct(Q, dim=0)


# ============================================================
# Multi-variable independence
# ============================================================


class TestMultiVar:
    """Each variable component should be reconstructed independently."""

    def test_different_vars_reconstructed_independently(self) -> None:
        nr, nz = 16, 8
        # Give each variable a different amplitude
        Q_np = np.zeros((NVAR, nr, nz), dtype=np.float32)
        for v in range(NVAR):
            x = np.linspace(0, 2 * math.pi, nr, endpoint=False)
            Q_np[v, :, :] = (np.sin(x) * (v + 1))[:, np.newaxis]

        Q = mx.array(Q_np)
        QL, QR = weno5z_reconstruct(Q, dim=0)
        mx.eval(QL, QR)

        QL_np = np.array(QL.tolist())
        QR_np = np.array(QR.tolist())

        # Variable 1 should be ~2x variable 0 (ratio of amplitudes)
        ratio_L = QL_np[1] / (QL_np[0] + 1e-10)
        ratio_R = QR_np[1] / (QR_np[0] + 1e-10)

        np.testing.assert_allclose(
            ratio_L, 2.0, atol=0.05, err_msg="Variable 1 ≠ 2×Variable 0 in QL"
        )
        np.testing.assert_allclose(
            ratio_R, 2.0, atol=0.05, err_msg="Variable 1 ≠ 2×Variable 0 in QR"
        )


# ============================================================
# WENO-Z weight properties
# ============================================================


class TestWenoZWeights:
    """On smooth data, WENO-Z weights should approach ideal weights."""

    def test_smooth_weights_near_ideal(self) -> None:
        """On a slowly varying sine, reconstructed value ~ linear combination
        with weights close to ideal (d0=0.1, d1=0.6, d2=0.3)."""
        n = 128
        x = np.linspace(0, 2 * math.pi, n, endpoint=False)
        q1d = np.sin(x).astype(np.float32)
        Q_np = np.tile(q1d[np.newaxis, :, np.newaxis], (1, 1, 4))
        Q = mx.array(Q_np)

        QL, QR = weno5z_reconstruct(Q, dim=0)
        mx.eval(QL, QR)

        # On smooth data, error should be tiny (< PLM error at same n)
        n_iface = QL.shape[1]
        dx = 2 * math.pi / n
        x_iface = (np.arange(2, 2 + n_iface) + 0.5) * dx
        exact = np.sin(x_iface).astype(np.float32)

        QL_np = np.array(QL.tolist())[0, :, 0]
        err = float(np.mean(np.abs(QL_np - exact)))
        assert err < 1e-5, f"WENO5-Z smooth error too large: {err:.2e}"
