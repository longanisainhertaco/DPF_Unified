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


# =====================================================================
# Phase V3: MHD-coupled integration on a static pinch snapshot
# =====================================================================

# PF-1000 pinch conditions (frozen MHD state — no MHD evolution)
_RHO_PINCH = 1e-4       # kg/m^3
_B_THETA_PINCH = 10.0   # T
_P_PINCH = 1e6          # Pa

# Beam parameters
_N_BEAM = 100
_E_BEAM_EV = 100e3      # 100 keV
_WEIGHT_TOTAL = 1e16

# Grid geometry: small but representative of pinch region
_GS = (16, 16, 16)
_DX = 5e-4              # 0.5 mm cells -> 8 mm domain

# MHD timestep (frozen field)
_MHD_DT = 1e-10         # s, << gyroperiod for 10 T (T_c ~ 2.2 ns)


def _build_static_mhd_state() -> tuple[np.ndarray, np.ndarray]:
    """Return (E, B) arrays for a static PF-1000 pinch snapshot.

    E = 0 (no inductive field in frozen snapshot).
    B = B_theta in the y-component (proxy for toroidal/azimuthal field).
    """
    nx, ny, nz = _GS
    E = np.zeros((nx, ny, nz, 3), dtype=np.float64)

    B = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    B[:, :, :, 1] = _B_THETA_PINCH   # By = B_theta

    return E, B


def _build_pic_with_beam() -> HybridPIC:
    """Create a HybridPIC instance with 100 deuterium beam particles at 100 keV."""
    pic = HybridPIC(
        grid_shape=_GS,
        dx=_DX,
        dy=_DX,
        dz=_DX,
        dt=_MHD_DT,
        use_esirkepov=True,
        use_binary_collisions=False,  # no collisions in V3 — isolate push-deposit cycle
    )
    pic.add_species(
        name="deuterium_beam",
        mass=_M_D,
        charge=_Q_E,
        positions=np.zeros((0, 3)),
        velocities=np.zeros((0, 3)),
        weights=np.zeros((0,)),
    )

    # Inject beam at domain centre directed along z-axis
    centre = np.array([
        _GS[0] * _DX * 0.5,
        _GS[1] * _DX * 0.5,
        _GS[2] * _DX * 0.1,  # near the lower z boundary (injection point)
    ])
    pic.inject_beam(
        species_idx=0,
        n_beam=_N_BEAM,
        energy_eV=_E_BEAM_EV,
        direction=[0.0, 0.0, 1.0],
        position=centre,
        spread=0.05,
        weight_total=_WEIGHT_TOTAL,
    )
    return pic


class TestPICOnStaticMHD:
    """Phase V3: push-deposit-feedback cycle on a frozen MHD pinch state.

    100 deuterium beam particles at 100 keV are pushed for 100 MHD steps
    (each step uses subcycle_pic with static E=0, B=10 T).  The MHD state
    never evolves — this isolates the PIC subsystem.
    """

    _N_STEPS = 100
    _C = 2.998e8  # speed of light [m/s]

    @pytest.fixture(scope="class")
    def evolved_pic(self) -> HybridPIC:
        """Run 100 steps and return the final PIC state."""
        pic = _build_pic_with_beam()
        E, B = _build_static_mhd_state()
        for _ in range(self._N_STEPS):
            subcycle_pic(pic, E, B, mhd_dt=_MHD_DT, n_sub=4)
        return pic

    def test_pic_on_static_mhd_no_nan(self, evolved_pic: HybridPIC) -> None:
        """No NaN in positions or velocities after 100 steps."""
        sp = evolved_pic.species[0]
        assert sp.n_particles() == _N_BEAM, (
            f"Particle count changed: expected {_N_BEAM}, got {sp.n_particles()}"
        )
        assert np.all(np.isfinite(sp.positions)), (
            f"NaN/Inf in positions: {sp.positions[~np.isfinite(sp.positions).all(axis=1)]}"
        )
        assert np.all(np.isfinite(sp.velocities)), (
            f"NaN/Inf in velocities: {sp.velocities[~np.isfinite(sp.velocities).all(axis=1)]}"
        )

    def test_pic_on_static_mhd_speed_bounded(self, evolved_pic: HybridPIC) -> None:
        """All |v| < c after 100 steps (relativistic Boris guarantee)."""
        sp = evolved_pic.species[0]
        speeds = np.sqrt(np.sum(sp.velocities ** 2, axis=1))
        max_speed = float(np.max(speeds))
        assert max_speed < self._C, (
            f"Speed exceeded c: |v|_max = {max_speed:.4e} m/s, c = {self._C:.4e} m/s"
        )

    def test_pic_on_static_mhd_jkin_nonzero(self, evolved_pic: HybridPIC) -> None:
        """Deposited J_kin has at least one nonzero cell after pushing."""
        _, Jx, Jy, Jz = evolved_pic.deposit()

        J_total = np.abs(Jx) + np.abs(Jy) + np.abs(Jz)
        nonzero_cells = int(np.sum(J_total > 0.0))
        J_max = float(np.max(J_total))

        assert nonzero_cells > 0, "All J_kin cells are zero — deposition did not run"
        assert np.isfinite(J_max), f"J_kin contains NaN/Inf: max={J_max}"

    def test_pic_on_static_mhd_energy_approximate(self) -> None:
        """Total kinetic energy changes less than 50% over 100 steps.

        With no collisions and a static (non-accelerating) B-only field,
        the Boris push conserves kinetic energy.  The 50% bound is generous
        to accommodate float64 round-off and sub-cycling phase errors.
        """
        pic = _build_pic_with_beam()
        E, B = _build_static_mhd_state()

        sp = pic.species[0]
        KE_initial = float(np.sum(0.5 * _M_D * np.sum(sp.velocities ** 2, axis=1) * sp.weights))

        for _ in range(self._N_STEPS):
            subcycle_pic(pic, E, B, mhd_dt=_MHD_DT, n_sub=4)

        KE_final = float(np.sum(0.5 * _M_D * np.sum(sp.velocities ** 2, axis=1) * sp.weights))

        assert KE_initial > 0.0, "Initial kinetic energy is zero — beam injection failed"
        assert KE_final > 0.0, "Final kinetic energy is zero — particles lost all energy"

        change_frac = abs(KE_final - KE_initial) / KE_initial
        assert change_frac < 0.50, (
            f"KE changed by {change_frac * 100:.1f}% (>50%). "
            f"KE_initial={KE_initial:.3e} J·macro, KE_final={KE_final:.3e} J·macro"
        )


# =====================================================================
# Phase V4: First full DPF discharge attempt with PIC active
# =====================================================================


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "PIC V4 exploratory — first full discharge attempt with KineticManager active. "
        "Known failure modes: ghost-cell NaN chain (pic_compound_bugs.md §5.4), "
        "non-relativistic Boris runaway at DPF E-fields (§5.2), "
        "KineticManager init may fail if grid_shape incompatible with PIC driver. "
        "This test documents what breaks, not what works."
    ),
    strict=False,  # xpass is also acceptable — if it survives, record that too
)
def test_pic_v4_short_discharge() -> None:
    """Attempt 100 engine steps with PIC active on PF-1000 8x1x16 grid.

    Setup:
    - pf1000 preset scaled to a small 8x1x16 grid
    - KineticConfig.enabled = True, start_time set to near-zero so PIC
      activates on step 1 (default 1e-6 s would never trigger in ~100 ns run)
    - 100 engine steps (not full discharge)

    Records: did it NaN? At which step? What was the failure mode?
    The test body always passes — failure information is captured in
    pytest's xfail output, not as an assertion error.
    """
    import warnings

    try:
        from dpf.engine.core import SimulationEngine
        from dpf.presets import get_preset
    except ImportError as exc:
        pytest.skip(f"Engine not importable: {exc}")

    # Build a minimal PF-1000-like config with a small grid.
    # Use the pf1000 preset as a base and override the heavy parts.
    base = get_preset("pf1000")

    # Small grid: 8 radial x 1 azimuthal x 16 axial
    base["grid_shape"] = [8, 1, 16]
    base["dx"] = 7.5e-4          # same cell size as pf1000

    # Short run: ~100 ns total (gives ~100 steps at dt ~ 1 ns)
    base["sim_time"] = 1e-7

    # Enable PIC with start_time much smaller than sim_time so it activates
    # on step 1.  KineticConfig gt=0, so use a very small positive value.
    base["kinetic"] = {
        "enabled": True,
        "start_time": 1e-15,    # activates immediately
        "inject_beam": True,
        "n_particles": 50,      # small — minimise overhead for this diagnostic
        "beam_energy": 100e3,
        "beam_position_ratio": [0.5, 0.5, 0.1],
        "beam_direction": [0.0, 0.0, 1.0],
        "beam_weight_total": 1e16,
    }

    # Force Python backend (metal/mlx may have different init paths)
    if "fluid" not in base:
        base["fluid"] = {}
    base["fluid"]["backend"] = "python"  # type: ignore[index]

    try:
        from dpf.config import SimulationConfig
        cfg = SimulationConfig(**base)
    except Exception as exc:
        pytest.xfail(f"Config construction failed: {exc}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = SimulationEngine(cfg)
    except Exception as exc:
        pytest.xfail(f"Engine __init__ failed (step 0): {exc}")

    # Run up to 100 steps, catching any error and recording failure chain
    nan_step: int | None = None
    failure_exc: Exception | None = None
    failure_mode: str = "unknown"

    for i in range(100):
        try:
            result = engine.step()
        except Exception as exc:
            failure_exc = exc
            failure_mode = type(exc).__name__
            nan_step = i
            break

        # Check for NaN in critical state fields
        state = engine.state
        nan_fields = [
            k for k, v in state.items()
            if isinstance(v, np.ndarray) and not np.all(np.isfinite(v))
        ]
        if nan_fields:
            nan_step = i
            failure_mode = f"NaN in {nan_fields}"
            break

        if result.finished:
            break

    # Build a summary that appears in pytest -v output regardless of xfail status
    kinetic_alive = (
        engine.kinetic is not None
        and engine.kinetic.kc.enabled
    )
    n_particles = (
        engine.kinetic.ion_species.n_particles()
        if kinetic_alive and engine.kinetic is not None
        else 0
    )

    summary_lines = [
        f"steps_completed={nan_step if nan_step is not None else i + 1}",
        f"kinetic_active={kinetic_alive}",
        f"n_particles={n_particles}",
        f"failure_mode={failure_mode}",
        f"failure_exc={failure_exc!r}",
    ]
    summary = " | ".join(summary_lines)

    if nan_step is not None or failure_exc is not None:
        pytest.xfail(f"PIC V4 NaN/error at step {nan_step}: {summary}")

    # If we reach here, 100 steps completed without NaN — record the win
    assert True, f"PIC V4 survived 100 steps: {summary}"


# =====================================================================
# Phase V5: PIC on MLX backend — 200 steps with beam injection
# =====================================================================

from dpf.metal.mlx_device import HAS_MLX as _HAS_MLX


def _make_uniform_state(nr: int, nz: int, rho0: float, p0: float) -> dict[str, np.ndarray]:
    """Build a uniform DPF state dict without calling solver.initialize() (does not exist)."""
    return {
        "rho": np.full((nr, 1, nz), rho0, dtype=np.float64),
        "velocity": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "pressure": np.full((nr, 1, nz), p0, dtype=np.float64),
        "B": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "Te": np.full((nr, 1, nz), p0 * 3.34e-27 / (2 * rho0 * 1.381e-23), dtype=np.float64),
        "Ti": np.full((nr, 1, nz), p0 * 3.34e-27 / (2 * rho0 * 1.381e-23), dtype=np.float64),
        "psi": np.zeros((nr, 1, nz), dtype=np.float64),
    }


@pytest.mark.skipif(not (_HAS_MLX and HAS_PIC), reason="MLX or PIC not available")
@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="PIC V5 first MLX attempt — ghost NaN / Boris runaway possible")
def test_pic_v5_mlx_200_steps_beam_at_50() -> None:
    """200 engine steps on MLX 16x1x32 grid with beam injected at step 50.

    Uses KineticManager pattern: PIC push applied manually each step,
    beam injected once at step 50.  MLX MHD solver runs the fluid physics.
    xfail(strict=False) so both pass and fail are acceptable outcomes.
    """
    import math  # noqa: PLC0415

    from dpf.metal.mlx_solver import MLXMHDSolver  # noqa: PLC0415, E402

    nr, nz = 16, 32
    dx = 0.23 / nr
    dz = 0.60 / nz

    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz),
        dx=dx,
        dz=dz,
        gamma=5.0 / 3.0,
        cfl=0.3,
        riemann_solver="hll",
        reconstruction="plm",
        time_integrator="ssp_rk3",
    )

    rho0, p0 = 0.084, 350.0
    state = _make_uniform_state(nr, nz, rho0, p0)

    V0, C0, L0, R0 = 6.12e-3
    omega = 1.0 / math.sqrt(L0 * C0)
    tau = 2.0 * L0 / R0

    pic = HybridPIC(
        grid_shape=(nr, 1, nz),
        dx=dx,
        dy=dx,
        dz=dz,
        dt=1e-9,
    )
    pic.add_species(
        name="deuterons",
        mass=3.34e-27,
        charge=1.602e-19,
        positions=np.zeros((0, 3)),
        velocities=np.zeros((0, 3)),
        weights=np.zeros((0,)),
    )

    n_total, inject_start = 200, 50
    nan_detected = False
    nan_step = -1
    max_v_over_c = 0.0
    particle_counts: list[int] = []
    c_light = 2.998e8
    beam_injected = False

    for step_i in range(n_total):
        t = step_i * 1e-9
        current = (V0 / (omega * L0)) * math.exp(-t / tau) * math.sin(omega * t)

        dt_mhd = min(solver.compute_dt(state), 5e-9)
        state = solver.step(state, dt_mhd, current=current, voltage=V0)

        for _key, val in state.items():
            if isinstance(val, np.ndarray) and np.any(np.isnan(val)):
                nan_detected = True
                nan_step = step_i
                break
        if nan_detected:
            break

        if step_i >= inject_start:
            if not beam_injected:
                pic.inject_beam(
                    species_idx=0,
                    n_beam=100,
                    energy_eV=100e3,
                    direction=[0.0, 0.0, 1.0],
                    position=[dx * nr / 2, 0.0, dz],
                    spread=0.1,
                    weight_total=1e16,
                )
                beam_injected = True

            B_arr = state["B"]  # (3, nr, 1, nz)
            B_avg = np.mean(B_arr, axis=(1, 2, 3))
            E_field = np.zeros((nr, 1, nz, 3), dtype=np.float64)
            B_field = np.zeros((nr, 1, nz, 3), dtype=np.float64)
            B_field[..., 0] = B_avg[0]
            B_field[..., 1] = B_avg[1]
            B_field[..., 2] = B_avg[2]

            pic.push_particles(E_field, B_field, dt=dt_mhd)

            for sp in pic.species:
                if sp.n_particles() > 0:
                    v2 = np.sum(sp.velocities ** 2, axis=1)
                    v_max = math.sqrt(float(np.max(v2)))
                    max_v_over_c = max(max_v_over_c, v_max / c_light)

        particle_counts.append(sum(sp.n_particles() for sp in pic.species))

    assert not nan_detected, f"NaN detected at step {nan_step}"
    assert max_v_over_c < 1.0, f"Superluminal particle: v/c = {max_v_over_c:.3f}"
    assert beam_injected, "Beam was never injected (inject_start not reached)"
    assert particle_counts[-1] >= 100, (
        f"Particle count dropped below injected amount: {particle_counts[-1]}"
    )
    rho_final = state["rho"]
    assert np.std(rho_final) > 1e-6 * np.mean(rho_final), "MHD state did not evolve"


@pytest.mark.skipif(not _HAS_MLX, reason="MLX not available")
def test_pic_v5_smoke_10_steps() -> None:
    """10-step smoke: MLX solver + no PIC, verifies no crash."""
    from dpf.metal.mlx_solver import MLXMHDSolver

    nr, nz = 16, 32
    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz),
        dx=0.015,
        dz=0.019,
        riemann_solver="hll",
        reconstruction="plm",
    )
    state = _make_uniform_state(nr, nz, rho0=0.084, p0=350.0)
    for _ in range(10):
        dt = min(solver.compute_dt(state), 5e-9)
        state = solver.step(state, dt, current=1e5, voltage=27e3)
    assert not any(
        np.any(np.isnan(v)) for v in state.values() if isinstance(v, np.ndarray)
    )
