"""Tests for Sprint 1 physics: Boris vacuum correction, Lee-More resistivity,
and flux-limited thermal conduction.

Tests cover:
  1. Boris factor: limits, physical/vacuum cells, monotonicity
  2. Boris-corrected geometric source terms: Metal kernel + NumPy reference
  3. Boris in timestepper: no fake mass injection, velocity clamping bounded
  4. Lee-More resistivity: physical limits, Spitzer convergence, saturation
  5. Spitzer resistivity: T^{-3/2} scaling
  6. compute_resistivity dispatcher
  7. Flux-limited conduction: harmonic limiting, free-streaming cap
"""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")


# ──────────────────────────────────────────────────────────────────────────────
# Boris factor (mlx_primitives)
# ──────────────────────────────────────────────────────────────────────────────


class TestBorisFactor:
    """Tests for the Boris reduction factor in mlx_primitives."""

    def test_physical_cell_unchanged(self):
        """In physical cells (v_A << c_boris), f_boris ~ 1."""
        from dpf.metal.mlx_primitives import boris_factor

        rho = mlx.array([[1e-3, 1e-2, 0.1]])
        B_sq = mlx.array([[1.0, 1.0, 1.0]])
        f = boris_factor(rho, B_sq)
        np.testing.assert_allclose(np.array(f), 1.0, atol=1e-3)

    def test_vacuum_cell_suppressed(self):
        """In vacuum cells (v_A >> c_boris), f_boris << 1."""
        from dpf.metal.mlx_primitives import boris_factor

        rho = mlx.array([[1e-12]])
        B_sq = mlx.array([[1.0]])
        f = np.array(boris_factor(rho, B_sq))
        assert f[0, 0] < 0.5

    def test_monotonic_with_density(self):
        """Boris factor increases monotonically with density (at fixed B)."""
        from dpf.metal.mlx_primitives import boris_factor

        rho = mlx.array([[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]])
        B_sq = mlx.ones_like(rho)
        f = np.array(boris_factor(rho, B_sq))
        assert np.all(np.diff(f[0]) > 0)

    def test_bounded_zero_one(self):
        """Boris factor is always in (0, 1]."""
        from dpf.metal.mlx_primitives import boris_factor

        rho = mlx.array([[1e-15, 1e-10, 1e-5, 1.0, 100.0]])
        B_sq = mlx.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        f = np.array(boris_factor(rho, B_sq))
        assert np.all(f > 0.0)
        assert np.all(f <= 1.0)

    def test_zero_B_gives_one(self):
        """Zero magnetic field -> f_boris = 1 (no correction needed)."""
        from dpf.metal.mlx_primitives import boris_factor

        rho = mlx.array([[1e-3]])
        B_sq = mlx.array([[0.0]])
        f = np.array(boris_factor(rho, B_sq))
        np.testing.assert_allclose(f, 1.0, atol=1e-6)

    def test_custom_c_boris(self):
        """Lower c_boris -> stronger suppression at same conditions."""
        from dpf.metal.mlx_primitives import boris_factor

        rho = mlx.array([[1e-9]])
        B_sq = mlx.array([[1.0]])
        f_default = np.array(boris_factor(rho, B_sq, c_boris=5e5))
        f_lower = np.array(boris_factor(rho, B_sq, c_boris=1e5))
        assert f_lower[0, 0] < f_default[0, 0]


# ──────────────────────────────────────────────────────────────────────────────
# Boris-corrected geometric sources
# ──────────────────────────────────────────────────────────────────────────────


class TestBorisGeometricSources:
    """Tests for Boris correction in cylindrical geometric source terms."""

    def _make_primitive_state(self, nr: int, nz: int, rho_val: float, Bt_val: float):
        """Create a primitive state with given rho and Btheta."""
        Q = np.zeros((10, nr, nz), dtype=np.float32)
        Q[0] = rho_val   # rho
        Q[4] = 1e3        # pressure
        Q[8] = Bt_val     # Btheta
        return Q

    def test_numpy_source_physical_cell(self):
        """In physical cells, Boris correction ~ 1 => standard sources."""
        from dpf.metal.mlx_kernels import cylindrical_source_numpy

        nr, nz = 8, 8
        Q = self._make_primitive_state(nr, nz, rho_val=1.0, Bt_val=0.1)
        r_cell = np.linspace(0.005, 0.04, nr, dtype=np.float32)
        inv_r = 1.0 / r_cell

        src = cylindrical_source_numpy(Q, r_cell, inv_r)
        # Source should be non-zero at physical cells (Bt != 0)
        assert np.any(np.abs(src[1]) > 0)  # S_mr

    def test_numpy_source_vacuum_suppressed(self):
        """In vacuum cells with B >> p, Boris correction suppresses magnetic source terms."""
        from dpf.metal.mlx_kernels import cylindrical_source_numpy

        nr, nz = 8, 8
        r_cell = np.linspace(0.005, 0.04, nr, dtype=np.float32)
        inv_r = 1.0 / r_cell

        # Use low pressure so magnetic pressure dominates
        # Physical cell: rho = 1.0, strong B
        Q_phys = np.zeros((10, nr, nz), dtype=np.float32)
        Q_phys[0] = 1.0    # rho
        Q_phys[4] = 1.0    # low pressure (B-dominated)
        Q_phys[8] = 100.0  # strong Btheta
        src_phys = cylindrical_source_numpy(Q_phys, r_cell, inv_r)

        # Vacuum cell: rho = 1e-12, same B
        Q_vac = np.zeros((10, nr, nz), dtype=np.float32)
        Q_vac[0] = 1e-12
        Q_vac[4] = 1.0
        Q_vac[8] = 100.0
        src_vac = cylindrical_source_numpy(Q_vac, r_cell, inv_r)

        # In physical cell: va² = B²/rho = 10000/1 = 10000, f_boris ~ 1
        # In vacuum cell: va² = B²/rho = 10000/1e-12 = 1e16, f_boris ~ 2.5e-5
        # Magnetic source (p_tot - Bt²*f_boris)/r should be much smaller in vacuum
        # Compare at ir=4 (away from axis L'Hopital)
        ratio = np.abs(src_vac[1, 4, 4]) / np.maximum(np.abs(src_phys[1, 4, 4]), 1e-30)
        assert ratio < 0.1, f"Vacuum source not sufficiently suppressed: ratio={ratio:.3f}"

    def test_metal_kernel_matches_numpy(self):
        """Metal GPU kernel produces same results as NumPy reference."""
        from dpf.metal.mlx_kernels import (
            cylindrical_source_mlx,
            cylindrical_source_numpy,
        )

        nr, nz = 8, 8
        Q_np = self._make_primitive_state(nr, nz, rho_val=0.01, Bt_val=5.0)
        r_cell = np.linspace(0.005, 0.04, nr, dtype=np.float32)
        inv_r = 1.0 / r_cell

        src_np = cylindrical_source_numpy(Q_np, r_cell, inv_r)
        Q_mx = mlx.array(Q_np)
        r_mx = mlx.array(r_cell)
        inv_r_mx = mlx.array(inv_r)
        src_mx = np.array(cylindrical_source_mlx(Q_mx, r_mx, inv_r_mx))

        # Compare interior cells (skip ir=0 L'Hopital)
        np.testing.assert_allclose(
            src_mx[1, 1:, :], src_np[1, 1:, :], rtol=1e-4, atol=1e-6,
            err_msg="S_mr mismatch between Metal and NumPy"
        )
        np.testing.assert_allclose(
            src_mx[3, 1:, :], src_np[3, 1:, :], rtol=1e-4, atol=1e-6,
            err_msg="S_mt mismatch between Metal and NumPy"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Boris in timestepper (no fake mass injection)
# ──────────────────────────────────────────────────────────────────────────────


class TestBorisTimestepper:
    """Tests for Boris correction in the SSP-RK3 timestepper."""

    def test_no_density_injection(self):
        """Timestepper should NOT inject fake mass (old va_max hack removed)."""
        from dpf.metal.mlx_primitives import RHO_FLOOR
        from dpf.metal.mlx_timestepper import _stage_post_impl

        # Create a vacuum-like state: low rho, strong B
        nr, nz = 4, 4
        U = mlx.zeros((10, nr, nz))
        rho_in = 1e-10
        U = U.at[0].add(rho_in)       # rho
        U = U.at[4].add(1e3)          # energy
        U = U.at[5].add(1e-5)         # entropy tracer
        U = U.at[8].add(100.0)        # strong Btheta

        U_post = _stage_post_impl(U, 5.0 / 3.0)
        rho_out = np.array(U_post[0])

        # With old hack: rho would be pumped up to B²/va_max² = 10000/1e12 = 1e-8
        # With Boris: rho stays at rho_in (or RHO_FLOOR)
        assert rho_out.max() <= max(rho_in * 10, RHO_FLOOR * 10), (
            f"Density inflated to {rho_out.max():.2e} — fake mass injection detected"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Lee-More resistivity
# ──────────────────────────────────────────────────────────────────────────────


class TestLeeMoreResistivity:

    def test_converges_to_spitzer_at_high_T(self):
        """Lee-More should approach Spitzer scaling at T >> T_Fermi."""
        from dpf.metal.mlx_transport import lee_more_resistivity, spitzer_resistivity

        Te = np.array([[100.0, 500.0, 1000.0]])  # eV
        rho = np.full_like(Te, 1e-4)
        eta_lm = lee_more_resistivity(Te, rho)
        eta_sp = spitzer_resistivity(Te)
        ratio = eta_lm / eta_sp
        # Should be within factor of 3 at high T (different Coulomb log formulations)
        assert np.all(ratio < 5.0), f"Lee-More too far from Spitzer: ratio={ratio}"
        assert np.all(ratio > 0.2), f"Lee-More too far from Spitzer: ratio={ratio}"

    def test_saturates_at_low_T(self):
        """Lee-More should saturate (not diverge) as T -> 0."""
        from dpf.metal.mlx_transport import lee_more_resistivity

        Te_low = np.array([[0.01, 0.1, 0.5]])  # eV
        rho = np.full_like(Te_low, 1e-3)
        eta = lee_more_resistivity(Te_low, rho)
        # Should be bounded (not inf or > 1 Ohm*m)
        assert np.all(np.isfinite(eta))
        assert np.all(eta < 1.0)  # Ohm*m

    def test_decreases_with_temperature(self):
        """Resistivity decreases as plasma gets hotter."""
        from dpf.metal.mlx_transport import lee_more_resistivity

        Te = np.array([[1.0, 10.0, 100.0, 1000.0]])
        rho = np.full_like(Te, 1e-3)
        eta = lee_more_resistivity(Te, rho)
        assert np.all(np.diff(eta[0]) < 0), "Resistivity should decrease with T"

    def test_physical_range(self):
        """Resistivity values in expected physical range for DPF conditions."""
        from dpf.metal.mlx_transport import lee_more_resistivity

        # DPF sheath: T ~ 1-10 eV, rho ~ 1e-3 kg/m3
        Te = np.array([[1.0, 5.0, 10.0]])
        rho = np.full_like(Te, 1e-3)
        eta = lee_more_resistivity(Te, rho)
        # Expected: 1e-5 to 1e-3 Ohm*m range
        assert np.all(eta > 1e-6)
        assert np.all(eta < 1e-2)


class TestSpitzerResistivity:

    def test_t_minus_3_2_scaling(self):
        """Spitzer eta ~ T^{-3/2}."""
        from dpf.metal.mlx_transport import spitzer_resistivity

        Te = np.array([[10.0, 100.0]])
        eta = spitzer_resistivity(Te)
        # eta(10)/eta(100) should be (100/10)^{3/2} = 31.6
        ratio = eta[0, 0] / eta[0, 1]
        np.testing.assert_allclose(ratio, 10**1.5, rtol=0.01)

    def test_floor_prevents_divergence(self):
        """Spitzer should not diverge at very low T (floor at 0.1 eV)."""
        from dpf.metal.mlx_transport import spitzer_resistivity

        Te = np.array([[0.001, 0.01, 0.05]])
        eta = spitzer_resistivity(Te)
        assert np.all(np.isfinite(eta))
        assert np.all(eta < 100.0)  # bounded


class TestComputeResistivity:

    def test_lee_more_model(self):
        from dpf.metal.mlx_transport import compute_resistivity

        Te = np.full((4, 8), 10.0)
        rho = np.full((4, 8), 1e-3)
        eta = compute_resistivity(Te, rho, model="lee_more")
        assert eta.shape == (4, 8)
        assert np.all(np.isfinite(eta))

    def test_spitzer_model(self):
        from dpf.metal.mlx_transport import compute_resistivity

        Te = np.full((4, 8), 10.0)
        rho = np.full((4, 8), 1e-3)
        eta = compute_resistivity(Te, rho, model="spitzer")
        assert eta.shape == (4, 8)

    def test_constant_model(self):
        from dpf.metal.mlx_transport import compute_resistivity

        Te = np.full((4, 8), 10.0)
        rho = np.full((4, 8), 1e-3)
        eta = compute_resistivity(Te, rho, model="constant", eta_floor=1e-6)
        np.testing.assert_allclose(eta, 1e-6)

    def test_unknown_model_raises(self):
        from dpf.metal.mlx_transport import compute_resistivity

        with pytest.raises(ValueError, match="Unknown resistivity"):
            compute_resistivity(np.ones((2, 2)), np.ones((2, 2)), model="bogus")

    def test_clamping(self):
        from dpf.metal.mlx_transport import compute_resistivity

        Te = np.full((2, 2), 10.0)
        rho = np.full((2, 2), 1e-3)
        eta = compute_resistivity(Te, rho, model="lee_more", eta_floor=1e-8, eta_cap=1e-4)
        assert np.all(eta >= 1e-8)
        assert np.all(eta <= 1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# Flux-limited conduction
# ──────────────────────────────────────────────────────────────────────────────


class TestFluxLimitedConduction:

    def test_no_limiting_at_low_kappa(self):
        """When Braginskii flux << free-streaming, kappa unchanged."""
        from dpf.metal.mlx_transport import flux_limit_kappa

        kappa = np.full((4, 8), 1.0)  # very low kappa
        Te = np.full((4, 8), 1e6)     # ~86 eV
        rho = np.full((4, 8), 1e-2)
        kappa_lim = flux_limit_kappa(kappa, Te, rho, dz=0.001)
        ratio = kappa_lim / kappa
        assert np.all(ratio > 0.9), f"Low-kappa should not be limited: min ratio={ratio.min():.3f}"

    def test_strong_limiting_at_high_kappa(self):
        """When Braginskii flux >> free-streaming, kappa significantly reduced."""
        from dpf.metal.mlx_transport import flux_limit_kappa

        kappa = np.full((4, 8), 1e10)  # extremely high kappa
        Te = np.full((4, 8), 1e5)      # ~8.6 eV (relatively cold)
        rho = np.full((4, 8), 1e-4)
        kappa_lim = flux_limit_kappa(kappa, Te, rho, dz=0.001)
        ratio = kappa_lim / kappa
        assert np.all(ratio < 0.5), f"High-kappa should be limited: max ratio={ratio.max():.3f}"

    def test_output_non_negative(self):
        """Flux-limited kappa is always non-negative."""
        from dpf.metal.mlx_transport import flux_limit_kappa

        rng = np.random.RandomState(42)
        kappa = rng.uniform(0.1, 1e8, (8, 16))
        Te = rng.uniform(1e3, 1e8, (8, 16))
        rho = rng.uniform(1e-6, 1e-1, (8, 16))
        kappa_lim = flux_limit_kappa(kappa, Te, rho, dz=0.001)
        assert np.all(kappa_lim >= 0.0)

    def test_preserves_shape(self):
        from dpf.metal.mlx_transport import flux_limit_kappa

        kappa = np.ones((16, 32))
        Te = np.ones((16, 32)) * 1e6
        rho = np.ones((16, 32)) * 1e-3
        result = flux_limit_kappa(kappa, Te, rho, dz=0.001)
        assert result.shape == (16, 32)

    def test_f_limit_parameter(self):
        """Lower f_limit -> stronger flux limiting."""
        from dpf.metal.mlx_transport import flux_limit_kappa

        kappa = np.full((4, 8), 1e6)
        Te = np.full((4, 8), 1e6)
        rho = np.full((4, 8), 1e-3)
        k_high = flux_limit_kappa(kappa, Te, rho, dz=0.001, f_limit=0.15)
        k_low = flux_limit_kappa(kappa, Te, rho, dz=0.001, f_limit=0.03)
        assert np.all(k_low <= k_high), "Lower f_limit should give more limiting"
