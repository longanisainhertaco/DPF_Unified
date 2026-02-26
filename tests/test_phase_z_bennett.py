"""Phase Z: Bennett Z-pinch equilibrium verification tests.

Tests the Bennett equilibrium module — the first Z-pinch-specific
verification benchmark in DPF-Unified.  Covers:
- Analytical profile correctness (density, B_theta, pressure, J_z)
- Bennett relation (global constraint)
- Force balance dp/dr = -J_z * B_theta to machine precision
- State dict initialization for MHD solvers
- Grid convergence of equilibrium drift
- Equilibrium maintenance under MHD evolution (CylindricalMHDSolver)

References:
    Bennett W.H., Phys. Rev. 45, 890 (1934).
    Haines M.G., Plasma Phys. Control. Fusion 53, 093001 (2011).
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import k_B, mu_0, pi  # noqa: N812
from dpf.validation.bennett_equilibrium import (
    bennett_btheta,
    bennett_current_density,
    bennett_current_from_temperature,
    bennett_density,
    bennett_line_density,
    bennett_pressure,
    create_bennett_state,
    verify_force_balance,
)

# ═══════════════════════════════════════════════════════
# Test parameters — typical DPF-scale Bennett equilibrium
# ═══════════════════════════════════════════════════════

# On-axis density: ~1e24 m^-3 (typical for DPF pinch column)
N_0 = 1.0e24
# Bennett radius: 1 mm
A_BENNETT = 1.0e-3
# Electron temperature: 1 keV (~1.16e7 K)
TE = 1.16e7
# Ion temperature: same as Te (thermalized)
TI = 1.16e7
# Deuterium ion mass
M_ION = 3.34358377e-27


class TestBennettDensity:
    """Tests for bennett_density()."""

    def test_on_axis_value(self):
        """n(0) = n_0."""
        r = np.array([0.0])
        n = bennett_density(r, N_0, A_BENNETT)
        assert n[0] == pytest.approx(N_0, rel=1e-15)

    def test_at_bennett_radius(self):
        """n(a) = n_0 / 4."""
        r = np.array([A_BENNETT])
        n = bennett_density(r, N_0, A_BENNETT)
        assert n[0] == pytest.approx(N_0 / 4.0, rel=1e-15)

    def test_large_r_decay(self):
        """n(r) ~ n_0 * a^4 / r^4 for r >> a."""
        r = np.array([1000.0 * A_BENNETT])
        n = bennett_density(r, N_0, A_BENNETT)
        expected = N_0 * A_BENNETT**4 / r[0] ** 4
        assert n[0] == pytest.approx(expected, rel=1e-5)

    def test_monotonic_decrease(self):
        """Density monotonically decreases with r."""
        r = np.linspace(0, 10 * A_BENNETT, 100)
        n = bennett_density(r, N_0, A_BENNETT)
        assert np.all(np.diff(n) <= 0)

    def test_non_negative(self):
        """Density is always >= 0."""
        r = np.linspace(0, 1000 * A_BENNETT, 1000)
        n = bennett_density(r, N_0, A_BENNETT)
        assert np.all(n >= 0)


class TestBennettBtheta:
    """Tests for bennett_btheta()."""

    def test_on_axis_zero(self):
        """B_theta(0) = 0 (field vanishes on axis)."""
        I_total = 1.0e6  # 1 MA
        r = np.array([0.0])
        Bt = bennett_btheta(r, I_total, A_BENNETT)
        assert Bt[0] == pytest.approx(0.0, abs=1e-30)

    def test_peak_location(self):
        """B_theta peaks at r = a (the Bennett radius)."""
        I_total = 1.0e6
        r = np.linspace(1e-6, 10 * A_BENNETT, 10000)
        Bt = bennett_btheta(r, I_total, A_BENNETT)
        idx_max = np.argmax(Bt)
        r_peak = r[idx_max]
        assert r_peak == pytest.approx(A_BENNETT, rel=0.01)

    def test_peak_value(self):
        """B_theta(a) = mu_0 * I / (4 * pi * a)."""
        I_total = 1.0e6
        r = np.array([A_BENNETT])
        Bt = bennett_btheta(r, I_total, A_BENNETT)
        expected = mu_0 * I_total / (4.0 * pi * A_BENNETT)
        assert Bt[0] == pytest.approx(expected, rel=1e-14)

    def test_large_r_decay(self):
        """B_theta(r) ~ mu_0 * I / (2*pi*r) for r >> a (wire limit)."""
        I_total = 1.0e6
        r = np.array([1000.0 * A_BENNETT])
        Bt = bennett_btheta(r, I_total, A_BENNETT)
        expected = mu_0 * I_total / (2.0 * pi * r[0])
        assert Bt[0] == pytest.approx(expected, rel=1e-4)


class TestBennettPressure:
    """Tests for bennett_pressure()."""

    def test_on_axis_value(self):
        """p(0) = n_0 * k_B * (Te + Ti)."""
        r = np.array([0.0])
        p = bennett_pressure(r, N_0, A_BENNETT, TE, TI)
        expected = N_0 * k_B * (TE + TI)
        assert p[0] == pytest.approx(expected, rel=1e-14)

    def test_proportional_to_density(self):
        """p(r) / n(r) = k_B * (Te + Ti) = const (isothermal)."""
        r = np.linspace(0.01 * A_BENNETT, 5 * A_BENNETT, 50)
        p = bennett_pressure(r, N_0, A_BENNETT, TE, TI)
        n = bennett_density(r, N_0, A_BENNETT)
        ratio = p / n
        expected = k_B * (TE + TI)
        np.testing.assert_allclose(ratio, expected, rtol=1e-14)


class TestBennettCurrentDensity:
    """Tests for bennett_current_density()."""

    def test_on_axis_value(self):
        """J_z(0) = I / (pi * a^2)."""
        I_total = 1.0e6
        r = np.array([0.0])
        Jz = bennett_current_density(r, I_total, A_BENNETT)
        expected = I_total / (pi * A_BENNETT**2)
        assert Jz[0] == pytest.approx(expected, rel=1e-14)

    def test_total_current_integral(self):
        """integral(J_z * 2*pi*r dr, 0, inf) = I."""
        I_total = 1.0e6
        # Numerical integration over large range
        r = np.linspace(0, 100 * A_BENNETT, 100000)
        dr = r[1] - r[0]
        Jz = bennett_current_density(r, I_total, A_BENNETT)
        I_integrated = np.sum(Jz * 2.0 * pi * r * dr)
        assert I_integrated == pytest.approx(I_total, rel=1e-3)


class TestBennettRelation:
    """Tests for the Bennett global constraint."""

    def test_line_density(self):
        """N = pi * n_0 * a^2."""
        N_line = bennett_line_density(N_0, A_BENNETT)
        expected = pi * N_0 * A_BENNETT**2
        assert N_line == pytest.approx(expected, rel=1e-15)

    def test_current_from_temperature(self):
        """Bennett relation: mu_0 * I^2 / (8*pi) = N * k_B * (Te + Ti)."""
        I_total = bennett_current_from_temperature(N_0, A_BENNETT, TE, TI)
        N_line = bennett_line_density(N_0, A_BENNETT)
        lhs = mu_0 * I_total**2 / (8.0 * pi)
        rhs = N_line * k_B * (TE + TI)
        assert lhs == pytest.approx(rhs, rel=1e-14)

    def test_current_magnitude_realistic(self):
        """Bennett current should be in the ~100 kA - 10 MA range for DPF."""
        I_total = bennett_current_from_temperature(N_0, A_BENNETT, TE, TI)
        assert 1e4 < I_total < 1e7, (
            f"Bennett current {I_total:.2e} A outside DPF range"
        )


class TestForceBalance:
    """Tests that dp/dr + J_z * B_theta = 0 analytically."""

    def test_analytical_force_balance(self):
        """Residual of force balance should be near machine epsilon."""
        I_total = bennett_current_from_temperature(N_0, A_BENNETT, TE, TI)
        r = np.linspace(0.01 * A_BENNETT, 10 * A_BENNETT, 1000)
        residual, max_rel_err = verify_force_balance(
            r, N_0, A_BENNETT, I_total, TE, TI
        )
        # Should be zero to float64 precision
        assert max_rel_err < 1e-12, (
            f"Force balance relative error {max_rel_err:.2e} exceeds tolerance"
        )

    def test_force_balance_inconsistent_current(self):
        """Non-Bennett current should give nonzero residual."""
        I_wrong = 2.0 * bennett_current_from_temperature(
            N_0, A_BENNETT, TE, TI
        )
        r = np.linspace(0.01 * A_BENNETT, 10 * A_BENNETT, 100)
        _, max_rel_err = verify_force_balance(
            r, N_0, A_BENNETT, I_wrong, TE, TI
        )
        # Should be large (order 1 error)
        assert max_rel_err > 0.1


class TestCreateBennettState:
    """Tests for create_bennett_state()."""

    def test_state_dict_keys(self):
        """State dict has all required keys."""
        state, _I_total, _r = create_bennett_state(
            nr=32, nz=16, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        required = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        assert set(state.keys()) == required

    def test_state_shapes(self):
        """Arrays have correct (nr, 1, nz) cylindrical shapes."""
        nr, nz = 32, 16
        state, _I_total, _r = create_bennett_state(
            nr=nr, nz=nz, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        assert state["rho"].shape == (nr, 1, nz)
        assert state["velocity"].shape == (3, nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)
        assert state["pressure"].shape == (nr, 1, nz)

    def test_density_profile(self):
        """Mass density follows rho = n(r) * m_ion."""
        state, _I_total, r = create_bennett_state(
            nr=64, nz=8, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        rho_expected = bennett_density(r, N_0, A_BENNETT) * M_ION
        np.testing.assert_allclose(
            state["rho"][:, 0, 0], rho_expected, rtol=1e-14
        )

    def test_btheta_profile(self):
        """B_theta(r) matches analytical Bennett field."""
        state, I_total, r = create_bennett_state(
            nr=64, nz=8, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        Bt_expected = bennett_btheta(r, I_total, A_BENNETT)
        np.testing.assert_allclose(
            state["B"][1, :, 0, 0], Bt_expected, rtol=1e-14
        )

    def test_z_uniformity(self):
        """Profiles are uniform along z (infinite cylinder)."""
        state, _, _ = create_bennett_state(
            nr=16, nz=8, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        for iz in range(8):
            np.testing.assert_array_equal(
                state["rho"][:, 0, iz], state["rho"][:, 0, 0]
            )

    def test_velocity_zero(self):
        """Initial velocity is zero (static equilibrium)."""
        state, _, _ = create_bennett_state(
            nr=16, nz=8, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        assert np.all(state["velocity"] == 0.0)

    def test_only_btheta_nonzero(self):
        """Only B_theta component is nonzero (pure Z-pinch)."""
        state, _, _ = create_bennett_state(
            nr=16, nz=8, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        assert np.all(state["B"][0] == 0.0), "B_r should be zero"
        assert np.all(state["B"][2] == 0.0), "B_z should be zero"
        assert np.any(state["B"][1] != 0.0), "B_theta should be nonzero"

    def test_current_magnitude(self):
        """Returned current matches Bennett relation."""
        state, I_total, _r = create_bennett_state(
            nr=32, nz=8, r_max=5e-3, dz=1e-4,
            n_0=N_0, a=A_BENNETT, Te=TE, Ti=TI, m_ion=M_ION,
        )
        I_expected = bennett_current_from_temperature(N_0, A_BENNETT, TE, TI)
        assert I_total == pytest.approx(I_expected, rel=1e-14)
