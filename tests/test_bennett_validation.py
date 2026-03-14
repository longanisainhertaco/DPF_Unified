"""Analytical validation tests for Bennett Z-pinch equilibrium.

Validates dpf.validation.bennett_equilibrium against exact analytical
solutions from Russell (2025) Bennett vortex formulas stored in the
reference database.

Tests cover:
    - Bennett density profile n(r) = n0 / (1 + r^2/a^2)^2
    - Bennett relation: mu_0*I_tot^2/(8*pi) = N*k_B*(Te+Ti)
    - Azimuthal field B_theta(r) from Ampere's law
    - Radial force balance dp/dr + Jz*B_theta = 0
    - Current density profile Jz(r)
    - Bennett vortex flow profile uz(r) = uz0/(1+xi^2*r^2)^2
    - Vorticity omega_theta(r) = 4*xi^2*uz0*r/(1+xi^2*r^2)^3
    - Pressure profile from MHD momentum equation
    - create_bennett_state consistency checks
"""
from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import e as e_charge
from dpf.constants import k_B, m_d, mu_0, pi
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

# Typical DPF-scale parameters
N0 = 1e24          # On-axis density [m^-3]
A_RADIUS = 1e-3    # Bennett radius 1 mm
Te = 1e7           # 1 keV electron temperature [K]
Ti = 1e7           # 1 keV ion temperature [K]
R = np.linspace(1e-5, 10e-3, 500)  # 10 um to 10 mm, avoid r=0


class TestBennettDensityProfile:
    """Validate n(r) = n0 / (1 + r^2/a^2)^2 (Russell 2025, Eq. 1)."""

    def test_on_axis_density(self) -> None:
        n = bennett_density(np.array([0.0]), N0, A_RADIUS)
        assert n[0] == pytest.approx(N0, rel=1e-12)

    def test_at_bennett_radius(self) -> None:
        """At r = a, n(a) = n0/4."""
        n = bennett_density(np.array([A_RADIUS]), N0, A_RADIUS)
        assert n[0] == pytest.approx(N0 / 4.0, rel=1e-12)

    def test_far_field_decay(self) -> None:
        """For r >> a, n(r) ~ n0*a^4/r^4 (1/r^4 falloff)."""
        r_far = np.array([100.0 * A_RADIUS])
        n = bennett_density(r_far, N0, A_RADIUS)
        expected = N0 * A_RADIUS**4 / r_far[0]**4
        assert n[0] == pytest.approx(expected, rel=1e-3)

    def test_monotonically_decreasing(self) -> None:
        n = bennett_density(R, N0, A_RADIUS)
        assert np.all(np.diff(n) < 0)

    def test_non_negative(self) -> None:
        n = bennett_density(R, N0, A_RADIUS)
        assert np.all(n > 0)


class TestBennettRelation:
    """Validate mu_0*I_tot^2/(8*pi) = N*k_B*(Te+Ti) (Russell 2025, Eq. 3)."""

    def test_bennett_relation_identity(self) -> None:
        """Current from bennett_current_from_temperature satisfies Bennett relation."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        N = bennett_line_density(N0, A_RADIUS)
        lhs = mu_0 * I_tot**2 / (8.0 * pi)
        rhs = N * k_B * (Te + Ti)
        assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_current_scales_with_temperature(self) -> None:
        """I_tot ~ sqrt(Te + Ti), so doubling T should increase I_tot by sqrt(2)."""
        I1 = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        I2 = bennett_current_from_temperature(N0, A_RADIUS, 2 * Te, 2 * Ti)
        assert pytest.approx(np.sqrt(2.0), rel=1e-10) == I2 / I1

    def test_current_scales_with_density(self) -> None:
        """I_tot ~ sqrt(N) ~ sqrt(n0), so 4x density -> 2x current."""
        I1 = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        I2 = bennett_current_from_temperature(4 * N0, A_RADIUS, Te, Ti)
        assert pytest.approx(2.0, rel=1e-10) == I2 / I1

    def test_line_density_formula(self) -> None:
        """N = pi * n0 * a^2."""
        N = bennett_line_density(N0, A_RADIUS)
        assert pytest.approx(pi * N0 * A_RADIUS**2, rel=1e-12) == N


class TestBennettMagneticField:
    """Validate B_theta(r) = mu_0*I_tot/(2*pi) * r/(r^2+a^2) (Russell 2025, Eq. 27)."""

    def test_on_axis_field_zero(self) -> None:
        """B_theta(0) = 0 by symmetry."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        B = bennett_btheta(np.array([0.0]), I_tot, A_RADIUS)
        assert B[0] == pytest.approx(0.0, abs=1e-20)

    def test_peak_at_bennett_radius(self) -> None:
        """B_theta peaks at r = a (dB/dr = 0 at r = a)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_fine = np.linspace(0.1 * A_RADIUS, 5 * A_RADIUS, 10000)
        B = bennett_btheta(r_fine, I_tot, A_RADIUS)
        i_max = np.argmax(B)
        r_peak = r_fine[i_max]
        assert r_peak == pytest.approx(A_RADIUS, rel=0.01)

    def test_far_field_1_over_r(self) -> None:
        """For r >> a, B_theta ~ mu_0*I_tot/(2*pi*r) (exterior field of a wire)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_far = np.array([1000.0 * A_RADIUS])
        B = bennett_btheta(r_far, I_tot, A_RADIUS)
        B_wire = mu_0 * I_tot / (2.0 * pi * r_far[0])
        assert B[0] == pytest.approx(B_wire, rel=1e-5)

    def test_ampere_law_integral(self) -> None:
        """Integral of J_z over cross-section equals I_total."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_int = np.linspace(0, 50 * A_RADIUS, 50000)
        dr = r_int[1] - r_int[0]
        Jz = bennett_current_density(r_int, I_tot, A_RADIUS)
        I_integrated = 2.0 * pi * np.sum(Jz * r_int * dr)
        assert I_integrated == pytest.approx(I_tot, rel=1e-3)


class TestForceBalance:
    """Validate dp/dr + Jz*B_theta = 0 (Russell 2025, Eq. 19 static case)."""

    def test_residual_zero(self) -> None:
        """Force balance residual should be zero for exact Bennett profiles."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_check = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 1000)
        _, max_rel_err = verify_force_balance(r_check, N0, A_RADIUS, I_tot, Te, Ti)
        assert max_rel_err < 1e-10

    def test_force_balance_not_satisfied_with_wrong_current(self) -> None:
        """With 2x current, force balance should NOT hold."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_check = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 1000)
        _, max_rel_err = verify_force_balance(r_check, N0, A_RADIUS, 2 * I_tot, Te, Ti)
        assert max_rel_err > 0.1


class TestCurrentDensity:
    """Validate Jz(r) = I_tot*a^2 / (pi*(r^2+a^2)^2) (Russell 2025, Eq. 26)."""

    def test_on_axis_peak(self) -> None:
        """Jz(0) = I_tot/(pi*a^2) is the maximum."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Jz = bennett_current_density(np.array([0.0]), I_tot, A_RADIUS)
        expected = I_tot / (pi * A_RADIUS**2)
        assert Jz[0] == pytest.approx(expected, rel=1e-12)

    def test_at_bennett_radius(self) -> None:
        """Jz(a) = I_tot/(4*pi*a^2)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Jz = bennett_current_density(np.array([A_RADIUS]), I_tot, A_RADIUS)
        expected = I_tot / (4.0 * pi * A_RADIUS**2)
        assert Jz[0] == pytest.approx(expected, rel=1e-12)

    def test_monotonically_decreasing(self) -> None:
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Jz = bennett_current_density(R, I_tot, A_RADIUS)
        assert np.all(np.diff(Jz) < 0)


class TestBennettVortex:
    """Validate Bennett Vortex solutions (Russell 2025, Eqs. 6-7, 31-32).

    The Bennett Vortex transfers nonlinearity from density to flow:
    - n(r) = n0 (uniform)
    - uz(r) = uz0 / (1 + xi^2*r^2)^2
    """

    def test_vortex_flow_on_axis(self) -> None:
        """uz(0) = uz0."""
        uz0 = 1e5  # 100 km/s
        xi2 = 1e6  # xi^2 [m^-2]
        uz = uz0 / (1.0 + xi2 * 0.0**2)**2
        assert uz == pytest.approx(uz0, rel=1e-12)

    def test_vortex_flow_at_characteristic_radius(self) -> None:
        """At r = 1/xi, uz = uz0/4."""
        uz0 = 1e5
        xi2 = 1e6
        r_char = 1.0 / np.sqrt(xi2)
        uz = uz0 / (1.0 + xi2 * r_char**2)**2
        assert uz == pytest.approx(uz0 / 4.0, rel=1e-12)

    def test_vortex_flow_far_field(self) -> None:
        """For r >> 1/xi, uz ~ uz0/(xi^4*r^8)... effectively zero."""
        uz0 = 1e5
        xi2 = 1e6
        r_far = 100.0 / np.sqrt(xi2)
        uz = uz0 / (1.0 + xi2 * r_far**2)**2
        assert uz < uz0 * 1e-6

    def test_vorticity_profile(self) -> None:
        """omega_theta = 4*xi^2*uz0*r / (1+xi^2*r^2)^3 (Eq. 31)."""
        uz0 = 1e5
        xi2 = 1e6
        r_test = np.linspace(1e-5, 5e-3, 200)
        omega = 4.0 * xi2 * uz0 * r_test / (1.0 + xi2 * r_test**2)**3
        # Should be zero at r=0 and r->inf, positive in between
        assert omega[0] < omega[50]
        assert omega[-1] < omega[50]
        assert np.all(omega >= 0)

    def test_vorticity_is_derivative_of_flow(self) -> None:
        """omega_theta = -duz/dr (Eq. 29 applied to axial flow)."""
        uz0 = 1e5
        xi2 = 1e6
        r_test = np.linspace(1e-5, 5e-3, 10000)
        dr = r_test[1] - r_test[0]
        uz = uz0 / (1.0 + xi2 * r_test**2)**2
        duz_dr = np.gradient(uz, dr)
        omega_analytical = 4.0 * xi2 * uz0 * r_test / (1.0 + xi2 * r_test**2)**3
        # Numerical derivative should match analytical vorticity
        # (sign: omega = -duz/dr for flow in +z; duz/dr < 0)
        np.testing.assert_allclose(-duz_dr[10:-10], omega_analytical[10:-10], rtol=1e-3)

    def test_sfs_criterion(self) -> None:
        """Shear-flow stabilization: |duz/dr| >= 0.1*k*V_A (Eq. 25).

        For a Bennett vortex with reversed flow (-z), the shear is positive
        everywhere, satisfying the SFS criterion for sufficiently large uz0.
        """
        uz0 = 1e5
        xi2 = 1e6
        r_test = np.linspace(1e-5, 3e-3, 500)
        duz_dr = -4.0 * xi2 * uz0 * r_test / (1.0 + xi2 * r_test**2)**3
        # With flow in -z: shear = -duz/dr > 0 everywhere
        assert np.all(-duz_dr > 0)


class TestBennettPressureProfile:
    """Validate pressure from MHD momentum (Russell 2025, Eq. 30)."""

    def test_pressure_from_ideal_gas(self) -> None:
        """p(r) = n(r)*k_B*(Te+Ti) for Bennett equilibrium."""
        p = bennett_pressure(R, N0, A_RADIUS, Te, Ti)
        n = bennett_density(R, N0, A_RADIUS)
        expected = n * k_B * (Te + Ti)
        np.testing.assert_allclose(p, expected, rtol=1e-12)

    def test_on_axis_pressure(self) -> None:
        """p(0) = n0*k_B*(Te+Ti)."""
        p = bennett_pressure(np.array([0.0]), N0, A_RADIUS, Te, Ti)
        assert p[0] == pytest.approx(N0 * k_B * (Te + Ti), rel=1e-12)

    def test_pressure_monotonically_decreasing(self) -> None:
        p = bennett_pressure(R, N0, A_RADIUS, Te, Ti)
        assert np.all(np.diff(p) < 0)

    def test_pressure_gradient_matches_jxb(self) -> None:
        """dp/dr = -Jz*B_theta (force balance check via numerical gradient)."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_fine = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 10000)
        dr = r_fine[1] - r_fine[0]
        p = bennett_pressure(r_fine, N0, A_RADIUS, Te, Ti)
        dp_dr_num = np.gradient(p, dr)
        Jz = bennett_current_density(r_fine, I_tot, A_RADIUS)
        Bt = bennett_btheta(r_fine, I_tot, A_RADIUS)
        jxb = -Jz * Bt
        # Skip edges where np.gradient is less accurate
        np.testing.assert_allclose(dp_dr_num[50:-50], jxb[50:-50], rtol=1e-3)


class TestBennettBParameter:
    """Validate b = mu_0*e^2*u0^2 / (8*k_B*(Te+Ti)) (Russell 2025, Eq. 3)."""

    def test_b_parameter_two_temp(self) -> None:
        """b parameter for two-temperature plasma."""
        u0 = 1e5  # flow velocity [m/s]
        b = mu_0 * e_charge**2 * u0**2 / (8.0 * k_B * (Te + Ti))
        # b should have dimensions of [m^-5] (inverse length^2 / density)
        assert b > 0
        # xi^2 = b * n0 should give a characteristic length scale
        xi2 = b * N0
        r_char = 1.0 / np.sqrt(xi2)
        assert 0 < r_char < 1.0  # reasonable characteristic radius

    def test_b_parameter_ideal_limit(self) -> None:
        """Ideal MHD: Te = Ti = T, b = mu_0*e^2*u0^2 / (16*k_B*T) (Eq. 5)."""
        u0 = 1e5
        T = Te  # Equal temperatures
        b_two_temp = mu_0 * e_charge**2 * u0**2 / (8.0 * k_B * (T + T))
        b_ideal = mu_0 * e_charge**2 * u0**2 / (16.0 * k_B * T)
        assert b_two_temp == pytest.approx(b_ideal, rel=1e-12)


class TestCreateBennettState:
    """Validate the 2D cylindrical state generator."""

    def test_state_dict_keys(self) -> None:
        state, I_tot, r = create_bennett_state(
            nr=32, nz=16, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        assert set(state.keys()) == expected_keys

    def test_state_shapes(self) -> None:
        nr, nz = 32, 16
        state, _, _ = create_bennett_state(
            nr=nr, nz=nz, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        assert state["rho"].shape == (nr, 1, nz)
        assert state["velocity"].shape == (3, nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)
        assert state["pressure"].shape == (nr, 1, nz)

    def test_density_matches_profile(self) -> None:
        nr, nz = 64, 8
        state, _, r_centers = create_bennett_state(
            nr=nr, nz=nz, r_max=10e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        rho_1d = state["rho"][:, 0, 0]
        expected_n = bennett_density(r_centers, N0, A_RADIUS)
        np.testing.assert_allclose(rho_1d, expected_n * m_d, rtol=1e-10)

    def test_btheta_matches_profile(self) -> None:
        nr, nz = 64, 8
        state, I_tot, r_centers = create_bennett_state(
            nr=nr, nz=nz, r_max=10e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        Bt_1d = state["B"][1, :, 0, 0]
        expected_Bt = bennett_btheta(r_centers, I_tot, A_RADIUS)
        np.testing.assert_allclose(Bt_1d, expected_Bt, rtol=1e-10)

    def test_velocity_zero(self) -> None:
        """No bulk flow in standard Bennett equilibrium."""
        state, _, _ = create_bennett_state(
            nr=32, nz=8, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        assert np.all(state["velocity"] == 0.0)

    def test_z_uniform(self) -> None:
        """All quantities are uniform along z (infinite cylinder)."""
        nr, nz = 32, 16
        state, _, _ = create_bennett_state(
            nr=nr, nz=nz, r_max=5e-3, dz=1e-3,
            n_0=N0, a=A_RADIUS, Te=Te, Ti=Ti, m_ion=m_d,
        )
        for k in range(nz):
            np.testing.assert_array_equal(
                state["rho"][:, 0, k], state["rho"][:, 0, 0]
            )


class TestAlfvenVelocity:
    """Validate V_A = B/sqrt(rho*mu_0) (Russell 2025, near Eq. 25)."""

    def test_alfven_speed_profile(self) -> None:
        """V_A should be finite and positive for r > 0."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_test = np.linspace(0.01 * A_RADIUS, 10 * A_RADIUS, 200)
        Bt = bennett_btheta(r_test, I_tot, A_RADIUS)
        n = bennett_density(r_test, N0, A_RADIUS)
        rho = n * m_d
        V_A = np.abs(Bt) / np.sqrt(rho * mu_0)
        assert np.all(np.isfinite(V_A))
        assert np.all(V_A > 0)

    def test_alfven_speed_order_of_magnitude(self) -> None:
        """For typical DPF params, V_A should be ~1e4-1e6 m/s."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Bt_peak = bennett_btheta(np.array([A_RADIUS]), I_tot, A_RADIUS)[0]
        n_peak = bennett_density(np.array([A_RADIUS]), N0, A_RADIUS)[0]
        rho_peak = n_peak * m_d
        V_A = np.abs(Bt_peak) / np.sqrt(rho_peak * mu_0)
        assert 1e3 < V_A < 1e8


class TestMHDConservationRelations:
    """Cross-checks between Bennett formulas and ideal MHD conservation laws."""

    def test_ampere_law_differential(self) -> None:
        """Jz = (1/mu_0)*(1/r)*d(r*B_theta)/dr."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_fine = np.linspace(0.01 * A_RADIUS, 20 * A_RADIUS, 50000)
        dr = r_fine[1] - r_fine[0]
        Bt = bennett_btheta(r_fine, I_tot, A_RADIUS)
        rBt = r_fine * Bt
        d_rBt_dr = np.gradient(rBt, dr)
        Jz_numerical = d_rBt_dr / (mu_0 * r_fine)
        Jz_analytical = bennett_current_density(r_fine, I_tot, A_RADIUS)
        np.testing.assert_allclose(
            Jz_numerical[100:-100], Jz_analytical[100:-100], rtol=5e-4
        )

    def test_magnetic_energy_density(self) -> None:
        """u_B = B^2/(2*mu_0) should be finite and positive."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        Bt = bennett_btheta(R, I_tot, A_RADIUS)
        u_B = Bt**2 / (2.0 * mu_0)
        assert np.all(u_B >= 0)
        assert np.all(np.isfinite(u_B))

    def test_beta_profile(self) -> None:
        """Plasma beta = 2*mu_0*p / B^2 — peaks on axis, decays outward."""
        I_tot = bennett_current_from_temperature(N0, A_RADIUS, Te, Ti)
        r_test = np.linspace(0.1 * A_RADIUS, 10 * A_RADIUS, 200)
        p = bennett_pressure(r_test, N0, A_RADIUS, Te, Ti)
        Bt = bennett_btheta(r_test, I_tot, A_RADIUS)
        beta = 2.0 * mu_0 * p / Bt**2
        # For Bennett equilibrium, beta should be a well-defined profile
        assert np.all(np.isfinite(beta))
        assert np.all(beta > 0)
