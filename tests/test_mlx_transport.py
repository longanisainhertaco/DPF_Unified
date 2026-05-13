"""Tests for mlx_transport: Thomas solver, resistive diffusion, thermal conduction."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="mlx not installed")

from dpf.metal.mlx_transport import (  # noqa: E402, I001
    _braginskii_kappa_perp_nrl,
    apply_resistive_diffusion,
    apply_thermal_conduction,
    thomas_solve,
)

MU_0 = 4.0 * math.pi * 1e-7
K_B = 1.380649e-23
M_D = 3.34358377e-27
M_E = 9.1093837015e-31
E_CHARGE = 1.602176634e-19


# ── Thomas Solver ───────────────────────────────────────────────


class TestThomasSolve:
    def test_identity_system(self) -> None:
        n = 8
        b = np.ones(n)
        a = np.zeros(n - 1)
        c = np.zeros(n - 1)
        d = np.arange(1.0, n + 1.0)
        x = thomas_solve(a, b, c, d)
        np.testing.assert_allclose(x, d, rtol=1e-12)

    def test_known_3x3(self) -> None:
        # [2 -1  0] [x0]   [1]
        # [-1  2 -1] [x1] = [0]
        # [0 -1  2] [x2]   [1]
        # Solution: [1, 1, 1]
        a = np.array([-1.0, -1.0])
        b = np.array([2.0, 2.0, 2.0])
        c = np.array([-1.0, -1.0])
        d = np.array([1.0, 0.0, 1.0])
        x = thomas_solve(a, b, c, d)
        np.testing.assert_allclose(x, [1.0, 1.0, 1.0], atol=1e-13)

    def test_diagonal_dominant(self) -> None:
        rng = np.random.default_rng(42)
        n = 32
        a = -rng.uniform(0.1, 0.3, n - 1)
        c = -rng.uniform(0.1, 0.3, n - 1)
        b = np.concatenate([[0.0], np.abs(a)]) + np.concatenate([np.abs(c), [0.0]]) + 1.0
        d = rng.standard_normal(n)
        x = thomas_solve(a, b, c, d)
        # Verify A*x = d
        Ax = np.zeros(n)
        Ax[0] = b[0] * x[0] + c[0] * x[1]
        for i in range(1, n - 1):
            Ax[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
        Ax[-1] = a[-1] * x[-2] + b[-1] * x[-1]
        np.testing.assert_allclose(Ax, d, rtol=1e-10)

    def test_single_element(self) -> None:
        x = thomas_solve(np.array([]), np.array([3.0]), np.array([]), np.array([9.0]))
        np.testing.assert_allclose(x, [3.0], atol=1e-14)

    def test_two_elements(self) -> None:
        # [2 -1] [x0]   [3]
        # [-1  2] [x1] = [3]  => x = [3, 3]
        a = np.array([-1.0])
        b = np.array([2.0, 2.0])
        c = np.array([-1.0])
        d = np.array([3.0, 3.0])
        x = thomas_solve(a, b, c, d)
        np.testing.assert_allclose(x, [3.0, 3.0], atol=1e-13)


# ── Resistive Diffusion ─────────────────────────────────────────


def _make_field(nr: int, nz: int, val: float = 0.0) -> mx.array:
    return mx.array(np.full((nr, nz), val, dtype=np.float32))


def _make_gaussian_Bz(nr: int, nz: int, dz: float, k: float = 4.0) -> mx.array:
    """Sinusoidal Bz profile for analytic diffusion test."""
    z = np.arange(nz) * dz
    B = np.zeros((nr, nz), dtype=np.float32)
    for ir in range(nr):
        B[ir, :] = np.cos(k * z)
    return mx.array(B)


class TestResistiveDiffusion:
    def _r_cell(self, nr: int, dr: float = 0.01) -> mx.array:
        return mx.array(np.arange(1, nr + 1, dtype=np.float32) * dr)

    def test_zero_eta_unchanged(self) -> None:
        nr, nz = 4, 16
        dr, dz = 0.01, 0.005
        r_cell = self._r_cell(nr, dr)
        Br = mx.array(np.random.default_rng(0).standard_normal((nr, nz)).astype(np.float32))
        Bz = mx.array(np.random.default_rng(1).standard_normal((nr, nz)).astype(np.float32))
        Bt = mx.array(np.random.default_rng(2).standard_normal((nr, nz)).astype(np.float32))
        rho = _make_field(nr, nz, 1e-3)
        p = _make_field(nr, nz, 1e3)

        Br_new, Bz_new, Bt_new, p_new = apply_resistive_diffusion(
            Br, Bz, Bt, rho, p, eta=0.0, dt=1e-7, dr=dr, dz=dz, r_cell=r_cell
        )
        np.testing.assert_allclose(np.asarray(Br_new), np.asarray(Br), atol=1e-6)
        np.testing.assert_allclose(np.asarray(Bz_new), np.asarray(Bz), atol=1e-6)
        np.testing.assert_allclose(np.asarray(Bt_new), np.asarray(Bt), atol=1e-6)

    def test_diffusion_reduces_gradient(self) -> None:
        nr, nz = 2, 32
        dr, dz = 0.01, 0.002
        eta = 1e-4
        dt = 1e-6
        r_cell = self._r_cell(nr, dr)
        rho = _make_field(nr, nz, 1e-3)
        p = _make_field(nr, nz, 1e3)

        k = 2.0 * math.pi / (nz * dz)
        z = np.arange(nz, dtype=np.float64) * dz
        B0_np = np.cos(k * z)
        Bz = mx.array(np.tile(B0_np, (nr, 1)).astype(np.float32))
        Br = _make_field(nr, nz, 0.0)
        Bt = _make_field(nr, nz, 0.0)

        Br_new, Bz_new, Bt_new, p_new = apply_resistive_diffusion(
            Br, Bz, Bt, rho, p, eta=eta, dt=dt, dr=dr, dz=dz, r_cell=r_cell
        )
        # Field amplitude must decrease with diffusion
        amp_before = float(mx.abs(Bz).mean())
        amp_after = float(mx.abs(Bz_new).mean())
        assert amp_after < amp_before, f"Expected diffusion: {amp_before:.4f} -> {amp_after:.4f}"

    def test_analytic_decay_z(self) -> None:
        """B(z,t) = cos(k*z) * exp(-k^2 * alpha * t) should hold for z-only diffusion."""
        nz = 64
        L = 1.0
        dz = L / nz
        dr = 0.01
        eta = 1e-5
        alpha = eta / MU_0
        dt = 1e-6
        k = 2.0 * math.pi / L

        r_cell = mx.array(np.array([0.005], dtype=np.float32))
        z = np.arange(nz, dtype=np.float64) * dz
        B0_np = np.cos(k * z)
        Bz = mx.array(B0_np.reshape(1, nz).astype(np.float32))
        Br = mx.array(np.zeros((1, nz), dtype=np.float32))
        Bt = mx.array(np.zeros((1, nz), dtype=np.float32))
        rho = mx.array(np.full((1, nz), 1e-3, dtype=np.float32))
        p = mx.array(np.full((1, nz), 1e3, dtype=np.float32))

        _, Bz_new, _, _ = apply_resistive_diffusion(
            Br, Bz, Bt, rho, p, eta=eta, dt=dt, dr=dr, dz=dz, r_cell=r_cell
        )
        decay = math.exp(-k**2 * alpha * dt)
        Bz_analytic = B0_np * decay
        Bz_out = np.asarray(Bz_new)[0, :]
        l2_err = np.sqrt(np.mean((Bz_out - Bz_analytic) ** 2))
        assert l2_err < 0.02, f"Analytic decay L2 error too large: {l2_err:.4e}"

    def test_second_order_convergence(self) -> None:
        """Halving dt should halve the L2 error (implicit Euler ~ 1st-order in time)."""
        nz = 64
        L = 1.0
        dz = L / nz
        dr = 0.01
        eta = 1e-5
        alpha = eta / MU_0
        k = 2.0 * math.pi / L

        r_cell = mx.array(np.array([0.005], dtype=np.float32))
        z = np.arange(nz, dtype=np.float64) * dz
        B0_np = np.cos(k * z)
        rho = mx.array(np.full((1, nz), 1e-3, dtype=np.float32))
        p = mx.array(np.full((1, nz), 1e3, dtype=np.float32))
        Br0 = mx.array(np.zeros((1, nz), dtype=np.float32))

        errors = []
        for dt in [2e-6, 1e-6]:
            Bz = mx.array(B0_np.reshape(1, nz).astype(np.float32))
            _, Bz_new, _, _ = apply_resistive_diffusion(
                Br0, Bz, Br0, rho, p, eta=eta, dt=dt, dr=dr, dz=dz, r_cell=r_cell
            )
            decay = math.exp(-k**2 * alpha * dt)
            analytic = B0_np * decay
            l2 = np.sqrt(np.mean((np.asarray(Bz_new)[0] - analytic) ** 2))
            errors.append(l2)

        ratio = errors[0] / errors[1]
        # Implicit Euler is 1st-order: expect ratio ~ 2 (dt halved)
        assert 1.5 < ratio < 3.5, f"Convergence ratio {ratio:.2f} outside [1.5, 3.5]"

    def test_ohmic_heating(self) -> None:
        """Total energy must not decrease: p_new >= p_old when diffusion removes B energy."""
        nr, nz = 4, 16
        dr, dz = 0.01, 0.005
        eta = 1e-3
        dt = 1e-7
        r_cell = self._r_cell(nr, dr)

        rng = np.random.default_rng(7)
        Bz_np = rng.standard_normal((nr, nz)).astype(np.float32)
        Bz = mx.array(Bz_np)
        Br = _make_field(nr, nz, 0.0)
        Bt = _make_field(nr, nz, 0.0)
        p0 = _make_field(nr, nz, 1e3)
        rho = _make_field(nr, nz, 1e-3)

        _, _, _, p_new = apply_resistive_diffusion(
            Br, Bz, Bt, rho, p0, eta=eta, dt=dt, dr=dr, dz=dz, r_cell=r_cell
        )
        # Pressure must not decrease (Ohmic heating adds energy)
        assert float(mx.min(p_new)) >= float(mx.min(p0)) - 1e-10

    def test_output_shapes(self) -> None:
        nr, nz = 6, 10
        dr, dz = 0.01, 0.005
        r_cell = self._r_cell(nr, dr)
        Br = _make_field(nr, nz)
        Bz = _make_field(nr, nz)
        Bt = _make_field(nr, nz)
        p = _make_field(nr, nz, 1e3)
        rho = _make_field(nr, nz, 1e-3)
        Br_new, Bz_new, Bt_new, p_new = apply_resistive_diffusion(
            Br, Bz, Bt, rho, p, eta=1e-5, dt=1e-7, dr=dr, dz=dz, r_cell=r_cell
        )
        for arr in (Br_new, Bz_new, Bt_new, p_new):
            assert arr.shape == (nr, nz)

    def test_spatially_varying_eta(self) -> None:
        nr, nz = 4, 16
        dr, dz = 0.01, 0.005
        r_cell = self._r_cell(nr, dr)
        eta_np = np.linspace(1e-5, 1e-4, nr * nz).reshape(nr, nz).astype(np.float32)
        eta = mx.array(eta_np)
        Bz = mx.array(np.ones((nr, nz), dtype=np.float32))
        Br = _make_field(nr, nz)
        Bt = _make_field(nr, nz)
        rho = _make_field(nr, nz, 1e-3)
        p = _make_field(nr, nz, 1e3)
        Br_new, Bz_new, Bt_new, p_new = apply_resistive_diffusion(
            Br, Bz, Bt, rho, p, eta=eta, dt=1e-7, dr=dr, dz=dz, r_cell=r_cell
        )
        assert np.all(np.isfinite(np.asarray(Bz_new)))


# ── Thermal Conduction ──────────────────────────────────────────


class TestThermalConduction:
    def _make_temps(self, nr: int, nz: int, val: float = 1e4) -> mx.array:
        return mx.array(np.full((nr, nz), val, dtype=np.float32))

    def test_uniform_temperature_unchanged(self) -> None:
        nr, nz = 4, 16
        Te = self._make_temps(nr, nz, 1e4)
        Ti = self._make_temps(nr, nz, 5e3)
        rho = mx.array(np.full((nr, nz), 1e-3, dtype=np.float32))
        B = mx.array(np.full((nr, nz), 1.0, dtype=np.float32))
        Te_new, Ti_new = apply_thermal_conduction(Te, Ti, rho, B, 1e2, dt=1e-7, dz=0.005)
        np.testing.assert_allclose(np.asarray(Te_new), np.asarray(Te), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(Ti_new), np.asarray(Ti), rtol=1e-5)

    def test_gradient_smooths(self) -> None:
        nr, nz = 2, 32
        dz = 0.002
        z = np.arange(nz, dtype=np.float32)
        Te_np = (1e4 + 5e3 * np.sin(2 * math.pi * z / nz)).reshape(1, nz)
        Te_np = np.tile(Te_np, (nr, 1))
        Te = mx.array(Te_np)
        Ti = mx.array(Te_np.copy())
        rho = mx.array(np.full((nr, nz), 1e-3, dtype=np.float32))
        B = mx.array(np.full((nr, nz), 1.0, dtype=np.float32))

        kappa = 1e3
        Te_new, _ = apply_thermal_conduction(Te, Ti, rho, B, kappa, dt=1e-7, dz=dz)
        # Peak-to-peak variation should decrease
        te_out = np.asarray(Te_new)
        te_in = np.asarray(Te)
        var_before = te_in.max() - te_in.min()
        var_after = te_out.max() - te_out.min()
        assert var_after < var_before, f"Expected smoothing: {var_before:.1f} -> {var_after:.1f}"

    def test_temperature_floor_enforced(self) -> None:
        nr, nz = 2, 8
        Te = mx.array(np.zeros((nr, nz), dtype=np.float32))  # below floor
        Ti = mx.array(np.zeros((nr, nz), dtype=np.float32))
        rho = mx.array(np.full((nr, nz), 1e-3, dtype=np.float32))
        B = mx.array(np.full((nr, nz), 1.0, dtype=np.float32))
        Te_new, Ti_new = apply_thermal_conduction(Te, Ti, rho, B, 1e2, dt=1e-6, dz=0.005)
        assert float(mx.min(Te_new)) >= 1.0
        assert float(mx.min(Ti_new)) >= 1.0

    def test_zero_kappa_unchanged(self) -> None:
        nr, nz = 4, 16
        rng = np.random.default_rng(3)
        Te_np = (1e4 + 1e3 * rng.standard_normal((nr, nz))).astype(np.float32)
        Te = mx.array(Te_np)
        Ti = mx.array(Te_np.copy())
        rho = mx.array(np.full((nr, nz), 1e-3, dtype=np.float32))
        B = mx.array(np.full((nr, nz), 1.0, dtype=np.float32))
        Te_new, Ti_new = apply_thermal_conduction(Te, Ti, rho, B, 0.0, dt=1e-7, dz=0.005)
        np.testing.assert_allclose(np.asarray(Te_new), Te_np, atol=1.0)

    def test_braginskii_kappa_perp_uses_nrl_high_field_coefficient(self) -> None:
        ne = 1.0e22
        Te_K = 1.0e6
        B_T = 50.0
        Te = np.full((2, 2), Te_K)
        rho = np.full((2, 2), ne * M_D)
        B_mag = np.full((2, 2), B_T)

        actual = _braginskii_kappa_perp_nrl(Te, rho, B_mag)
        Te_eV = Te_K * K_B / E_CHARGE
        ne_cm3 = ne * 1.0e-6
        lnL = 24.0 - math.log(math.sqrt(ne_cm3) * Te_eV**-1.0)
        tau_e = 3.44e5 * Te_eV**1.5 / (ne_cm3 * lnL)
        omega_ce = E_CHARGE * B_T / M_E
        expected = 4.7 * ne * K_B**2 * Te_K / (M_E * omega_ce**2 * tau_e)

        np.testing.assert_allclose(actual, expected, rtol=1.0e-6)

    def test_anisotropic_conduction_uses_nrl_perpendicular_path(self) -> None:
        nr, nz = 8, 1
        dr = 0.01
        ne = 1.0e22
        Te_np = np.linspace(8.0e5, 1.2e6, nr, dtype=np.float32).reshape(nr, 1)
        Te = mx.array(Te_np)
        Ti = mx.array(Te_np.copy())
        rho = mx.array(np.full((nr, nz), ne * M_D, dtype=np.float32))
        B = mx.array(np.full((nr, nz), 50.0, dtype=np.float32))
        Br = mx.array(np.zeros((nr, nz), dtype=np.float32))
        Bz = mx.array(np.zeros((nr, nz), dtype=np.float32))
        Bt = B
        kappa_parallel = mx.array(np.full((nr, nz), 1.0e7, dtype=np.float32))

        expected_kperp = _braginskii_kappa_perp_nrl(
            Te_np.astype(np.float64),
            np.full((nr, nz), ne * M_D),
            np.full((nr, nz), 50.0),
        )
        fixed_ratio_kperp = np.asarray(kappa_parallel) * 1.0e-6
        assert not np.allclose(expected_kperp, fixed_ratio_kperp)

        Te_auto, _ = apply_thermal_conduction(
            Te, Ti, rho, B, kappa_parallel,
            dt=1.0e-7, dz=0.005, dr=dr, Br=Br, Bz=Bz, Bt=Bt,
        )
        Te_explicit, _ = apply_thermal_conduction(
            Te, Ti, rho, B, kappa_parallel,
            dt=1.0e-7, dz=0.005, dr=dr, Br=Br, Bz=Bz, Bt=Bt,
            kappa_perpendicular=mx.array(expected_kperp.astype(np.float32)),
        )

        np.testing.assert_allclose(
            np.asarray(Te_auto), np.asarray(Te_explicit), rtol=1.0e-6, atol=1.0e-3
        )

    def test_analytic_conduction_z(self) -> None:
        """T(z,t) = T_mean + dT * exp(-k^2*chi*t) * cos(k*z) should hold."""
        nz = 64
        L = 1.0
        dz = L / nz
        kappa = 1e2
        dt = 1e-7
        k = 2.0 * math.pi / L

        z = np.arange(nz, dtype=np.float64) * dz
        T_mean = 1e4
        dT_amp = 500.0
        T0_np = T_mean + dT_amp * np.cos(k * z)

        rho_val = 1e-3
        n = rho_val / M_D
        chi = kappa / (n * K_B)
        decay = math.exp(-k**2 * chi * dt)
        T_analytic = T_mean + dT_amp * decay * np.cos(k * z)

        Te = mx.array(T0_np.reshape(1, nz).astype(np.float32))
        Ti = mx.array(T0_np.reshape(1, nz).astype(np.float32))
        rho = mx.array(np.full((1, nz), rho_val, dtype=np.float32))
        B = mx.array(np.ones((1, nz), dtype=np.float32))

        Te_new, _ = apply_thermal_conduction(Te, Ti, rho, B, kappa, dt=dt, dz=dz)
        l2 = np.sqrt(np.mean((np.asarray(Te_new)[0] - T_analytic) ** 2))
        # Allow ~5% relative amplitude error for implicit Euler
        assert l2 < 0.05 * dT_amp, f"L2 error {l2:.2f} K exceeds 5% of amplitude {dT_amp}"

    def test_output_shapes(self) -> None:
        nr, nz = 5, 12
        Te = self._make_temps(nr, nz)
        Ti = self._make_temps(nr, nz, 5e3)
        rho = mx.array(np.full((nr, nz), 1e-3, dtype=np.float32))
        B = mx.array(np.ones((nr, nz), dtype=np.float32))
        Te_new, Ti_new = apply_thermal_conduction(Te, Ti, rho, B, 1e2, dt=1e-7, dz=0.005)
        assert Te_new.shape == (nr, nz)
        assert Ti_new.shape == (nr, nz)

    def test_single_z_cell_noop(self) -> None:
        nr, nz = 4, 1
        Te = self._make_temps(nr, nz, 1e4)
        Ti = self._make_temps(nr, nz, 5e3)
        rho = mx.array(np.full((nr, nz), 1e-3, dtype=np.float32))
        B = mx.array(np.ones((nr, nz), dtype=np.float32))
        Te_new, Ti_new = apply_thermal_conduction(Te, Ti, rho, B, 1e3, dt=1e-6, dz=0.005)
        np.testing.assert_allclose(np.asarray(Te_new), np.asarray(Te), atol=1e-6)
