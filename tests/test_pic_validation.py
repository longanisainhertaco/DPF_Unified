"""PIC validation tests for hybrid.py Boris pusher, deposition, and yield.

Tests 1-4 are unit tests verifying individual PIC components.
Tests 5+ verify integration with the DPF engine.
"""

from __future__ import annotations

import numpy as np
import pytest

# Boris pusher and deposition are Numba JIT — import may trigger compilation
try:
    from dpf.experimental.pic.hybrid import (
        HybridPIC,
        boris_push,
        deposit_density,
        interpolate_field_to_particles,
    )

    HAS_PIC = True
except ImportError:
    HAS_PIC = False

pytestmark = pytest.mark.skipif(not HAS_PIC, reason="PIC module not available")

_M_D = 3.34e-27  # deuterium mass [kg]
_Q_E = 1.602e-19  # elementary charge [C]


class TestBorisGyration:
    """Test 1: Boris push produces correct Larmor gyration."""

    def test_gyration_radius(self):
        """Deuteron in uniform Bz gyrates with r_L = m*v_perp/(q*B)."""
        B0 = 1.0
        v_perp = 1e6
        dt = 1e-10
        r_L = _M_D * v_perp / (_Q_E * B0)
        omega_c = _Q_E * B0 / _M_D

        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[v_perp, 0.0, 0.0]])
        E = np.zeros((1, 3))
        B = np.array([[0.0, 0.0, B0]])

        T_c = 2 * np.pi / omega_c
        n_steps = int(T_c / dt)

        max_r = 0.0
        for _ in range(n_steps):
            pos, vel = boris_push(pos, vel, E, B, _Q_E, _M_D, dt)
            r = np.sqrt(pos[0, 0] ** 2 + pos[0, 1] ** 2)
            max_r = max(max_r, r)

        # Boris half-step offset gives orbit diameter ~2*r_L for max displacement
        assert max_r == pytest.approx(2 * r_L, rel=0.2)

    def test_returns_to_origin(self):
        """After one full gyration, particle returns near starting point."""
        B0 = 1.0
        v_perp = 1e5
        dt = 1e-10
        omega_c = _Q_E * B0 / _M_D
        T_c = 2 * np.pi / omega_c

        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[v_perp, 0.0, 0.0]])
        E = np.zeros((1, 3))
        B = np.array([[0.0, 0.0, B0]])

        n_steps = int(T_c / dt)
        for _ in range(n_steps):
            pos, vel = boris_push(pos, vel, E, B, _Q_E, _M_D, dt)

        r_L = _M_D * v_perp / (_Q_E * B0)
        assert abs(pos[0, 0]) < r_L * 0.1
        assert abs(pos[0, 1]) < r_L * 0.1


class TestCICDeposition:
    """Test 2: CIC deposition conserves charge."""

    def test_total_charge_conservation(self):
        """Deposited density integrates to total weight."""
        grid = (8, 8, 8)
        dx = dy = dz = 0.01
        pos = np.array([[0.035, 0.035, 0.035]])
        weights = np.array([1e12])

        n = deposit_density(pos, weights, grid, dx, dy, dz)
        total = np.sum(n) * dx * dy * dz
        assert total == pytest.approx(1e12, rel=1e-6)

    def test_multiple_particles(self):
        """Multiple particles: total density = sum of weights."""
        grid = (8, 8, 8)
        dx = dy = dz = 0.01
        pos = np.array([
            [0.02, 0.02, 0.02],
            [0.05, 0.05, 0.05],
            [0.03, 0.04, 0.06],
        ])
        weights = np.array([1e10, 2e10, 3e10])

        n = deposit_density(pos, weights, grid, dx, dy, dz)
        total = np.sum(n) * dx * dy * dz
        assert total == pytest.approx(6e10, rel=1e-4)


class TestInterpolation:
    """Test 3: Field interpolation returns correct values."""

    def test_uniform_field(self):
        """Uniform field returns same value at any position."""
        field = np.full((8, 8, 8, 3), [0.0, 0.0, 1.5])
        pos = np.array([[0.035, 0.035, 0.035]])
        B = interpolate_field_to_particles(field, pos, 0.01, 0.01, 0.01)
        np.testing.assert_allclose(B[0], [0.0, 0.0, 1.5], atol=1e-10)


class TestDDCrossSection:
    """Test 4: Bosch-Hale DD cross section sanity checks."""

    def test_nonzero_at_50kev(self):
        from dpf.diagnostics.beam_target import dd_cross_section
        sigma = dd_cross_section(50.0)
        assert sigma > 0

    def test_zero_below_threshold(self):
        from dpf.diagnostics.beam_target import dd_cross_section
        sigma = dd_cross_section(0.01)
        assert sigma == 0.0

    def test_peak_near_100kev(self):
        from dpf.diagnostics.beam_target import dd_cross_section
        s50 = dd_cross_section(50.0)
        s100 = dd_cross_section(100.0)
        assert s100 > s50  # rising toward peak


class TestPICYield:
    """Test 5: PIC neutron yield produces nonzero rate."""

    def test_beam_produces_neutrons(self):
        from dpf.diagnostics.pic_yield import pic_neutron_yield_rate

        n = 100
        v_beam = np.sqrt(2 * 100e3 * _Q_E / _M_D)
        pos = np.random.uniform(0.001, 0.007, (n, 3))
        vel = np.zeros((n, 3))
        vel[:, 2] = v_beam
        weights = np.full(n, 1e13)
        n_target = np.full((8, 8, 8), 1e25)

        dY = pic_neutron_yield_rate(pos, vel, weights, n_target, 0.001, 0.001, 0.001)
        assert dY > 0

    def test_stationary_particles_no_yield(self):
        from dpf.diagnostics.pic_yield import pic_neutron_yield_rate

        pos = np.array([[0.004, 0.004, 0.004]])
        vel = np.array([[0.0, 0.0, 0.0]])
        weights = np.array([1e15])
        n_target = np.full((8, 8, 8), 1e25)

        dY = pic_neutron_yield_rate(pos, vel, weights, n_target, 0.001, 0.001, 0.001)
        assert dY == 0.0


class TestParticleInit:
    """Test 6: MHD-to-PIC initialization."""

    def test_maxwellian_velocity(self):
        from dpf.kinetic.initialize import initialize_particles_from_mhd

        rho = np.full((8, 1, 16), 1e-4)
        Te = np.full((8, 1, 16), 1e6)
        result = initialize_particles_from_mhd(rho, Te, n_particles=5000)

        v2_mean = np.mean(np.sum(result["velocities"] ** 2, axis=1))
        v2_thermal = 3 * 1.38e-23 * 1e6 / 3.34e-27
        assert v2_mean == pytest.approx(v2_thermal, rel=0.3)

    def test_correct_shapes(self):
        from dpf.kinetic.initialize import initialize_particles_from_mhd

        rho = np.full((8, 1, 16), 1e-4)
        Te = np.full((8, 1, 16), 1e6)
        result = initialize_particles_from_mhd(rho, Te, n_particles=1000)

        assert result["positions"].shape == (1000, 3)
        assert result["velocities"].shape == (1000, 3)
        assert result["weights"].shape == (1000,)


class TestGhostNaNGuard:
    """Test Fix 1: interpolation survives NaN in ghost cells."""

    def test_interpolation_survives_nan_field(self):
        """Field with NaN in ghost cells returns finite values at particle positions."""
        field = np.full((8, 8, 8, 3), 1.0)
        # Poison ghost cells (x=0 face) with NaN — as MHD solver produces at pinch
        field[0, :, :, :] = np.nan
        field[7, :, :, :] = np.nan

        # Particle near the ghost boundary
        pos = np.array([[0.005, 0.04, 0.04]])  # 0.005 / 0.01 = 0.5 cells in
        result = interpolate_field_to_particles(field, pos, 0.01, 0.01, 0.01)

        assert result.shape == (1, 3)
        assert np.all(np.isfinite(result)), (
            f"Expected finite interpolated values, got {result}"
        )

    def test_interior_field_unaffected_by_nan_guard(self):
        """NaN guard does not corrupt values when field is clean."""
        field = np.zeros((8, 8, 8, 3))
        field[:, :, :, 2] = 5.0  # Bz = 5 T everywhere

        pos = np.array([[0.04, 0.04, 0.04]])
        result = interpolate_field_to_particles(field, pos, 0.01, 0.01, 0.01)

        np.testing.assert_allclose(result[0], [0.0, 0.0, 5.0], atol=1e-12)


class TestEsirkepovDtConsistency:
    """Test Fix 2: Esirkepov deposit uses the same dt as the push."""

    def test_esirkepov_dt_consistency(self):
        """Pushing with dt=0.5*default_dt then depositing uses the correct J scaling.

        The Esirkepov J prefactor is q*(x_new-x_old)/(cell_vol*dt).
        If deposit uses self.dt instead of the push dt, J is off by a factor
        of push_dt / self.dt.  After fix, J should scale linearly with dt.
        """
        grid_shape = (8, 8, 8)
        dx = dy = dz = 0.01
        default_dt = 1e-9

        # Build two identical PIC instances with different effective push dts
        def _make_pic_with_push(push_dt: float) -> tuple[np.ndarray, ...]:
            pic = HybridPIC(grid_shape, dx, dy, dz, default_dt)
            pos0 = np.array([[0.04, 0.04, 0.04]])
            vel = np.array([[1e4, 0.0, 0.0]])
            weights = np.array([1e15])
            pic.add_species("d", 3.34e-27, 1.602e-19, pos0, vel, weights)
            E = np.zeros((8, 8, 8, 3))
            B = np.zeros((8, 8, 8, 3))
            pic.push_particles(E, B, dt=push_dt)
            _, Jx, _, _ = pic.deposit()
            return Jx

        Jx_full = _make_pic_with_push(default_dt)
        Jx_half = _make_pic_with_push(default_dt / 2)

        # With correct dt tracking: J is proportional to displacement / dt.
        # Displacement scales with dt (Boris push), so J ~ (v*dt)/dt = v.
        # Both runs should give the same J magnitude (velocity is identical).
        # Without the fix: Jx_half would use self.dt (=1e-9), giving
        # J proportional to (v * dt/2) / 1e-9 = half the correct value.
        jx_sum_full = float(np.sum(np.abs(Jx_full)))
        jx_sum_half = float(np.sum(np.abs(Jx_half)))

        assert jx_sum_full > 0, "No current deposited at full dt"
        assert jx_sum_half > 0, "No current deposited at half dt"

        # After fix: J ~ v (same for both). Before fix: ratio would be 2.
        ratio = jx_sum_full / jx_sum_half
        assert ratio == pytest.approx(1.0, rel=0.05), (
            f"J ratio full_dt/half_dt = {ratio:.3f}, expected ~1.0. "
            f"Esirkepov is using wrong dt (expected fix: _last_push_dt)."
        )
