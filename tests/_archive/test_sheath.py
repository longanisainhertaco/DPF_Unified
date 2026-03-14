"""Tests for sheath module: Bohm velocity, Child-Langmuir, Debye, Poisson.

Test categories:
1. Bohm velocity matches analytic formula
2. Child-Langmuir current matches analytic formula
3. Debye length matches analytic formula
4. Sheath thickness reasonable for DPF conditions
5. Floating potential is negative
6. Poisson solver boundary conditions and charge neutrality
7. Sheath BC application
"""

from __future__ import annotations

import numpy as np

from dpf.constants import e, epsilon_0, k_B, m_e, m_p

# ═══════════════════════════════════════════════════════
# Bohm Velocity Tests
# ═══════════════════════════════════════════════════════

class TestBohmVelocity:
    """Tests for Bohm velocity computation."""

    def test_analytic_match(self):
        """v_B = sqrt(k_B * Te / m_i) for hydrogen."""
        from dpf.sheath.bohm import bohm_velocity

        Te = 1e6  # K
        v_B = bohm_velocity(Te, m_p)
        v_expected = np.sqrt(k_B * Te / m_p)

        np.testing.assert_allclose(v_B, v_expected, rtol=1e-10)

    def test_scales_with_sqrt_Te(self):
        """v_B should scale as sqrt(Te)."""
        from dpf.sheath.bohm import bohm_velocity

        v1 = bohm_velocity(1e6, m_p)
        v2 = bohm_velocity(4e6, m_p)

        np.testing.assert_allclose(v2 / v1, 2.0, rtol=1e-10)

    def test_heavier_ion_slower(self):
        """Heavier ions should have lower Bohm velocity."""
        from dpf.sheath.bohm import bohm_velocity

        v_H = bohm_velocity(1e6, m_p)        # Hydrogen
        v_D = bohm_velocity(1e6, 2.0 * m_p)  # Deuterium

        assert v_D < v_H
        np.testing.assert_allclose(v_H / v_D, np.sqrt(2.0), rtol=1e-6)

    def test_zero_temperature(self):
        """v_B = 0 at zero temperature."""
        from dpf.sheath.bohm import bohm_velocity

        assert bohm_velocity(0.0, m_p) == 0.0

    def test_typical_dpf_value(self):
        """At Te ~ 1 keV = 1.16e7 K, v_B ~ 1e5 m/s for hydrogen."""
        from dpf.sheath.bohm import bohm_velocity

        Te_keV = 1.16e7  # 1 keV in Kelvin
        v_B = bohm_velocity(Te_keV, m_p)

        # v_B ~ sqrt(1.38e-23 * 1.16e7 / 1.67e-27) ~ sqrt(9.6e10) ~ 3.1e5 m/s
        assert 1e5 < v_B < 1e6, f"v_B = {v_B:.2e} should be ~ 3e5 m/s"


# ═══════════════════════════════════════════════════════
# Child-Langmuir Tests
# ═══════════════════════════════════════════════════════

class TestChildLangmuir:
    """Tests for Child-Langmuir space-charge-limited current."""

    def test_analytic_match(self):
        """J_CL = (4/9) * epsilon_0 * sqrt(2e/mi) * V^{3/2} / d^2."""
        from dpf.sheath.bohm import child_langmuir_current

        V = 100.0  # V
        d = 1e-3   # m
        J = child_langmuir_current(V, d, m_p)
        J_expected = (4.0 / 9.0) * epsilon_0 * np.sqrt(2.0 * e / m_p) * V**1.5 / d**2

        np.testing.assert_allclose(J, J_expected, rtol=1e-10)

    def test_scales_with_V_three_halves(self):
        """J_CL ~ V^{3/2}."""
        from dpf.sheath.bohm import child_langmuir_current

        d = 1e-3
        J1 = child_langmuir_current(100.0, d, m_p)
        J2 = child_langmuir_current(400.0, d, m_p)

        np.testing.assert_allclose(J2 / J1, (4.0)**1.5, rtol=1e-10)

    def test_scales_with_d_inv_squared(self):
        """J_CL ~ 1/d^2."""
        from dpf.sheath.bohm import child_langmuir_current

        V = 100.0
        J1 = child_langmuir_current(V, 1e-3, m_p)
        J2 = child_langmuir_current(V, 2e-3, m_p)

        np.testing.assert_allclose(J1 / J2, 4.0, rtol=1e-10)

    def test_positive_current(self):
        """Current should always be positive."""
        from dpf.sheath.bohm import child_langmuir_current

        assert child_langmuir_current(100.0, 1e-3) > 0
        assert child_langmuir_current(-100.0, 1e-3) > 0  # Uses abs(V)


# ═══════════════════════════════════════════════════════
# Debye Length Tests
# ═══════════════════════════════════════════════════════

class TestDebyeLength:
    """Tests for Debye length."""

    def test_analytic_match(self):
        """lambda_D = sqrt(epsilon_0 * k_B * Te / (ne * e^2))."""
        from dpf.sheath.bohm import debye_length

        ne = 1e20
        Te = 1e6
        lam = debye_length(ne, Te)
        lam_expected = np.sqrt(epsilon_0 * k_B * Te / (ne * e * e))

        np.testing.assert_allclose(lam, lam_expected, rtol=1e-10)

    def test_increases_with_temperature(self):
        """Hotter plasma has larger Debye length."""
        from dpf.sheath.bohm import debye_length

        lam1 = debye_length(1e20, 1e6)
        lam2 = debye_length(1e20, 4e6)

        np.testing.assert_allclose(lam2 / lam1, 2.0, rtol=1e-10)

    def test_decreases_with_density(self):
        """Denser plasma has smaller Debye length."""
        from dpf.sheath.bohm import debye_length

        lam1 = debye_length(1e20, 1e6)
        lam2 = debye_length(4e20, 1e6)

        np.testing.assert_allclose(lam1 / lam2, 2.0, rtol=1e-10)


# ═══════════════════════════════════════════════════════
# Sheath Thickness Tests
# ═══════════════════════════════════════════════════════

class TestSheathThickness:
    """Tests for sheath thickness estimation."""

    def test_positive(self):
        """Sheath thickness should be positive."""
        from dpf.sheath.bohm import sheath_thickness

        d = sheath_thickness(ne=1e20, Te=1e6, V_sheath=100.0)
        assert d > 0

    def test_larger_than_debye(self):
        """Sheath should be at least a few Debye lengths."""
        from dpf.sheath.bohm import debye_length, sheath_thickness

        ne = 1e20
        Te = 1e6
        d = sheath_thickness(ne, Te, V_sheath=100.0)
        lam_D = debye_length(ne, Te)

        assert d >= 3.0 * lam_D

    def test_increases_with_voltage(self):
        """Higher voltage gives thicker sheath."""
        from dpf.sheath.bohm import sheath_thickness

        d1 = sheath_thickness(1e20, 1e6, V_sheath=10.0)
        d2 = sheath_thickness(1e20, 1e6, V_sheath=1000.0)

        assert d2 > d1


# ═══════════════════════════════════════════════════════
# Floating Potential Tests
# ═══════════════════════════════════════════════════════

class TestFloatingPotential:
    """Tests for floating potential."""

    def test_negative(self):
        """Floating potential should be negative (repels electrons)."""
        from dpf.sheath.bohm import floating_potential

        V_f = floating_potential(1e6)
        assert V_f < 0

    def test_scales_with_Te(self):
        """V_f should be proportional to Te."""
        from dpf.sheath.bohm import floating_potential

        V1 = floating_potential(1e6)
        V2 = floating_potential(2e6)

        np.testing.assert_allclose(V2 / V1, 2.0, rtol=1e-10)

    def test_typical_value(self):
        """V_f ~ -2.84 * Te_eV for hydrogen."""
        from dpf.sheath.bohm import floating_potential

        Te_eV = 1.0  # 1 eV
        Te_K = Te_eV * e / k_B  # Convert to Kelvin
        V_f = floating_potential(Te_K)

        # For hydrogen: V_f ~ -2.84 Te_eV
        # More precisely: V_f = -(Te_eV/2) * ln(mp/(2*pi*me)) = -(Te_eV/2) * 5.68 = -2.84
        expected = -(Te_eV / 2.0) * np.log(m_p / (2.0 * np.pi * m_e))
        np.testing.assert_allclose(V_f, expected, rtol=1e-3)


# ═══════════════════════════════════════════════════════
# Poisson Solver Tests
# ═══════════════════════════════════════════════════════

class TestPoisson1D:
    """Tests for 1D Poisson solver."""

    def test_boundary_conditions(self):
        """Solution should match boundary values at endpoints."""
        from dpf.sheath.bohm import poisson_1d

        N = 100
        ne = np.full(N, 1e20)
        ni = np.full(N, 1e20)  # Neutral
        dx = 1e-4

        phi, E = poisson_1d(ne, ni, dx, V_left=0.0, V_right=-100.0)

        np.testing.assert_allclose(phi[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(phi[-1], -100.0, atol=1e-10)

    def test_neutral_plasma_linear(self):
        """With neutral plasma (ne = ni), potential is linear between BCs."""
        from dpf.sheath.bohm import poisson_1d

        N = 100
        ne = np.full(N, 1e20)
        ni = np.full(N, 1e20)  # Neutral: rho = 0
        dx = 1e-4

        phi, E = poisson_1d(ne, ni, dx, V_left=0.0, V_right=-100.0)

        # With zero charge density, phi should be linear
        x = np.linspace(0, (N - 1) * dx, N)
        phi_expected = -100.0 * x / x[-1]
        np.testing.assert_allclose(phi, phi_expected, atol=0.1)

    def test_positive_charge_concave_down(self):
        """Net positive charge should make potential curve concave down (positive peak).

        d²phi/dx² = -e*(Z*ni - ne)/eps0 < 0 when ni >> ne
        With both BCs at 0, this gives a positive potential peak in the middle.
        """
        from dpf.sheath.bohm import poisson_1d

        N = 50
        ne = np.full(N, 1e18)
        ni = np.full(N, 1e20)  # ni >> ne: positive space charge
        dx = 1e-5

        phi, E = poisson_1d(ne, ni, dx, V_left=0.0, V_right=0.0)

        # With positive space charge, phi > 0 inside (potential hill)
        assert np.max(phi[1:-1]) > 0, "Positive charge should create positive potential hill"
        # Peak should be in the middle (symmetric charge distribution)
        mid = N // 2
        assert phi[mid] > phi[1], "Potential should peak near the center"


# ═══════════════════════════════════════════════════════
# Sheath BC Application Tests
# ═══════════════════════════════════════════════════════

class TestSheathBC:
    """Tests for sheath boundary condition application."""

    def test_applies_bohm_velocity(self):
        """Should set boundary velocity to Bohm speed."""
        from dpf.sheath.bohm import apply_sheath_bc, bohm_velocity

        nx, ny, nz = 8, 8, 8
        state = {
            "velocity": np.zeros((3, nx, ny, nz)),
            "rho": np.full((nx, ny, nz), 1e-2),
            "Te": np.full((nx, ny, nz), 1e6),
        }

        Te_boundary = 1e6
        state = apply_sheath_bc(
            state, ne_boundary=1e20, Te_boundary=Te_boundary,
            V_sheath=100.0, boundary="z_high"
        )

        v_B = bohm_velocity(Te_boundary, m_p)
        # z-component of velocity at z_high boundary
        np.testing.assert_allclose(state["velocity"][2, :, :, -1], v_B, rtol=1e-10)

    def test_z_low_boundary(self):
        """Should set negative velocity at z_low (into the wall)."""
        from dpf.sheath.bohm import apply_sheath_bc, bohm_velocity

        nx, ny, nz = 8, 8, 8
        state = {
            "velocity": np.zeros((3, nx, ny, nz)),
            "rho": np.full((nx, ny, nz), 1e-2),
            "Te": np.full((nx, ny, nz), 1e6),
        }

        state = apply_sheath_bc(
            state, ne_boundary=1e20, Te_boundary=1e6,
            V_sheath=100.0, boundary="z_low"
        )

        v_B = bohm_velocity(1e6, m_p)
        # z_low: velocity should be negative (pointing into wall at z=0)
        np.testing.assert_allclose(state["velocity"][2, :, :, 0], -v_B, rtol=1e-10)
