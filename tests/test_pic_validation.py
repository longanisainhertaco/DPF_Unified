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
        subcycle_pic,  # noqa: F401
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


class TestRelativisticBoris:
    """Tests for the relativistic Boris push (Vay 2008)."""

    def test_boris_relativistic_speed_bounded(self):
        """Relativistic Boris never exceeds c in strong DPF E-fields.

        Classical pusher reaches v > c in ~6000 steps at 1e7 V/m.
        10000 steps in this field must all satisfy |v| < c with the
        relativistic kernel.
        """
        c = 2.998e8
        E_strong = 1e7  # V/m — typical DPF accelerating gradient
        dt = 1e-10
        n_steps = 10000

        pos = np.array([[0.05, 0.05, 0.05]])
        vel = np.zeros((1, 3))
        E_p = np.array([[E_strong, 0.0, 0.0]])
        B_p = np.zeros((1, 3))

        for step in range(n_steps):
            pos, vel = boris_push(pos, vel, E_p, B_p, _Q_E, _M_D, dt, relativistic=True)
            speed = float(np.sqrt(np.sum(vel ** 2)))
            assert speed < c, (
                f"Speed exceeded c at step {step}: |v|={speed:.4e} m/s, c={c:.4e} m/s"
            )

    def test_boris_relativistic_reduces_to_classical(self):
        """At v << c, relativistic and classical Boris agree to rtol=1e-3.

        Deuteron in 1 T field with v_perp = 1e4 m/s (beta ~ 3e-5).
        One gyroperiod with 100 steps.
        """
        B0 = 1.0
        v_perp = 1e4  # << c
        dt = 1e-10
        n_steps = 100

        pos_r = np.array([[0.0, 0.0, 0.0]])
        vel_r = np.array([[v_perp, 0.0, 0.0]])
        pos_c = pos_r.copy()
        vel_c = vel_r.copy()
        E_p = np.zeros((1, 3))
        B_p = np.array([[0.0, 0.0, B0]])

        for _ in range(n_steps):
            pos_r, vel_r = boris_push(pos_r, vel_r, E_p, B_p, _Q_E, _M_D, dt, relativistic=True)
            pos_c, vel_c = boris_push(pos_c, vel_c, E_p, B_p, _Q_E, _M_D, dt, relativistic=False)

        np.testing.assert_allclose(
            vel_r, vel_c, rtol=1e-3,
            err_msg="Relativistic and classical velocities diverge at v << c"
        )
        np.testing.assert_allclose(
            pos_r, pos_c, rtol=1e-3,
            err_msg="Relativistic and classical positions diverge at v << c"
        )

    def test_subcycle_improves_gyration(self):
        """Sub-cycled push tracks analytical Larmor orbit more closely.

        Deuteron in 10 T field. MHD dt ~ 0.77 gyroperiods (under-resolved).
        10 sub-steps resolve the gyration at ~0.077 gyroperiods each.
        Sub-cycled position should be closer to the analytical arc endpoint.
        """
        B0 = 10.0
        v_perp = 1e6
        charge = _Q_E
        mass = _M_D
        omega_c = charge * B0 / mass
        T_cyc = 2.0 * np.pi / omega_c
        mhd_dt = 0.77 * T_cyc  # single step spans 0.77 gyroperiods

        pos0 = np.array([[0.0, 0.0, 0.0]])
        vel0 = np.array([[v_perp, 0.0, 0.0]])
        E_p = np.zeros((1, 3))
        B_p = np.array([[0.0, 0.0, B0]])

        # Analytical arc endpoint: gyration in x-y plane
        theta = omega_c * mhd_dt
        r_L = mass * v_perp / (charge * B0)
        x_exact = r_L * np.sin(theta) + pos0[0, 0]
        y_exact = r_L * (1.0 - np.cos(theta)) + pos0[0, 1]

        # Single-step push
        pos_1, _ = boris_push(pos0.copy(), vel0.copy(), E_p, B_p, charge, mass, mhd_dt)
        err_single = np.sqrt((pos_1[0, 0] - x_exact) ** 2 + (pos_1[0, 1] - y_exact) ** 2)

        # Sub-cycled push (10 sub-steps)
        n_sub = 10
        dt_sub = mhd_dt / n_sub
        pos_n = pos0.copy()
        vel_n = vel0.copy()
        for _ in range(n_sub):
            pos_n, vel_n = boris_push(pos_n, vel_n, E_p, B_p, charge, mass, dt_sub)
        err_sub = np.sqrt((pos_n[0, 0] - x_exact) ** 2 + (pos_n[0, 1] - y_exact) ** 2)

        assert err_sub < err_single, (
            f"Sub-cycling not more accurate: "
            f"single={err_single:.3e} m, sub={err_sub:.3e} m (r_L={r_L:.3e} m)"
        )


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
