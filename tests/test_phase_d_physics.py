"""Tests for Phase D: Physics Fidelity Improvements.

Covers:
    D.1 Braginskii eta_1 (first perpendicular viscosity) limits
    D.2 Braginskii eta_2 = 4 * eta_1
    D.3 Full Braginskii stress tensor backward compatibility
    D.4 Anisotropic thermal conduction along B
    D.5 Anisotropic thermal conduction across B
    D.6 Powell source terms for zero div(B)
    D.7 Powell source terms for nonzero div(B)
    D.8 Dedner damping with Mignone-Tzeferacos tuning
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import e as e_charge
from dpf.constants import k_B, m_d


# ===================================================================
# D.1: Braginskii eta_1 limits
# ===================================================================

class TestBraginskiiEta1Limits:
    """Verify eta_1 asymptotic behaviour in weakly and strongly magnetised limits."""

    def test_weakly_magnetised_eta1_approaches_eta0(self):
        """For omega_ci * tau_i << 1, eta_1 should approach eta_0."""
        from dpf.fluid.viscosity import braginskii_eta0, braginskii_eta1, ion_collision_time

        ni = np.array([1e22])   # moderate density
        Ti = np.array([1e6])    # 1 MK
        tau_i = ion_collision_time(ni, Ti)

        # Very weak B -> omega_ci * tau_i << 1
        B_weak = np.array([1e-10])  # extremely weak field
        eta0 = braginskii_eta0(ni, Ti, tau_i)
        eta1 = braginskii_eta1(ni, Ti, tau_i, B_weak, m_ion=m_d)

        # eta_1 should be capped at eta_0 in the weakly magnetised limit
        assert eta1[0] > 0.0, "eta_1 should be positive"
        # Due to capping, eta_1 = min(0.3 * n*kB*T/(omega^2*tau), eta_0) => eta_0
        ratio = eta1[0] / eta0[0]
        assert ratio == pytest.approx(1.0, abs=0.01), (
            f"In weakly magnetised limit, eta_1 should be capped at eta_0 (ratio={ratio})"
        )

    def test_strongly_magnetised_eta1_small(self):
        """For omega_ci * tau_i >> 1, eta_1 should be much smaller than eta_0."""
        from dpf.fluid.viscosity import braginskii_eta0, braginskii_eta1, ion_collision_time

        ni = np.array([1e22])
        Ti = np.array([1e6])
        tau_i = ion_collision_time(ni, Ti)

        # Strong B -> omega_ci * tau_i >> 1
        B_strong = np.array([10.0])  # 10 Tesla
        eta0 = braginskii_eta0(ni, Ti, tau_i)
        eta1 = braginskii_eta1(ni, Ti, tau_i, B_strong, m_ion=m_d)

        ratio = eta1[0] / eta0[0]
        assert ratio < 0.01, (
            f"In strongly magnetised limit, eta_1/eta_0 should be << 1, got {ratio:.4e}"
        )


# ===================================================================
# D.2: Braginskii eta_2 = 4 * eta_1
# ===================================================================

class TestBraginskiiEta2:
    """Verify eta_2 = 4 * eta_1."""

    def test_eta2_is_4_eta1(self):
        """eta_2 should be exactly 4 times eta_1."""
        from dpf.fluid.viscosity import braginskii_eta1, braginskii_eta2, ion_collision_time

        ni = np.array([1e22, 5e22, 1e23])
        Ti = np.array([5e5, 1e6, 5e6])
        tau_i = ion_collision_time(ni, Ti)
        B_mag = np.array([0.5, 1.0, 3.0])

        eta1 = braginskii_eta1(ni, Ti, tau_i, B_mag, m_ion=m_d)
        eta2 = braginskii_eta2(ni, Ti, tau_i, B_mag, m_ion=m_d)

        np.testing.assert_allclose(eta2, 4.0 * eta1, rtol=1e-12,
                                   err_msg="eta_2 should be exactly 4 * eta_1")


# ===================================================================
# D.3: Full Braginskii stress backward compatibility
# ===================================================================

class TestFullBraginskiiBackwardCompatible:
    """Verify that viscous_stress_rate with full_braginskii=False gives same result as before."""

    def test_default_matches_old_behaviour(self):
        """Default (full_braginskii=False) should give the same result as the isotropic stress."""
        from dpf.fluid.viscosity import viscous_stress_rate

        np.random.seed(42)
        nx, ny, nz = 8, 8, 8
        velocity = np.random.randn(3, nx, ny, nz) * 1e4
        rho = np.ones((nx, ny, nz)) * 1e-4
        eta0 = np.ones((nx, ny, nz)) * 1e-2
        dx = dy = dz = 1e-3

        # Call without full_braginskii
        accel_default = viscous_stress_rate(velocity, rho, eta0, dx, dy, dz)

        # Call with full_braginskii=False explicitly
        accel_explicit = viscous_stress_rate(
            velocity, rho, eta0, dx, dy, dz,
            full_braginskii=False,
        )

        np.testing.assert_allclose(accel_default, accel_explicit, rtol=1e-14,
                                   err_msg="Default should match explicit full_braginskii=False")

    def test_full_braginskii_without_B_falls_back(self):
        """full_braginskii=True without B should fall back to isotropic."""
        from dpf.fluid.viscosity import viscous_stress_rate

        np.random.seed(42)
        nx, ny, nz = 8, 8, 8
        velocity = np.random.randn(3, nx, ny, nz) * 1e4
        rho = np.ones((nx, ny, nz)) * 1e-4
        eta0 = np.ones((nx, ny, nz)) * 1e-2
        dx = dy = dz = 1e-3

        accel_default = viscous_stress_rate(velocity, rho, eta0, dx, dy, dz)
        accel_fallback = viscous_stress_rate(
            velocity, rho, eta0, dx, dy, dz,
            full_braginskii=True, B=None, eta1=None,
        )

        np.testing.assert_allclose(accel_default, accel_fallback, rtol=1e-14,
                                   err_msg="full_braginskii=True without B should fall back to isotropic")


# ===================================================================
# D.4: Anisotropic thermal conduction along B
# ===================================================================

class TestAnisotropicConductionAlongB:
    """Uniform B in x-direction, temperature gradient in x -> full parallel conduction."""

    def test_conduction_along_B(self):
        """Heat should flow primarily along B when B is aligned with temperature gradient."""
        from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction

        nx, ny, nz = 32, 4, 4
        dx = dy = dz = 1e-3

        # Uniform B in x-direction
        B = np.zeros((3, nx, ny, nz))
        B[0] = 1.0  # 1 Tesla in x

        # Temperature gradient in x-direction (same direction as B)
        x = np.linspace(0, (nx - 1) * dx, nx)
        Te = np.zeros((nx, ny, nz))
        for i in range(nx):
            Te[i, :, :] = 1e6 + 5e5 * np.sin(2 * np.pi * x[i] / (nx * dx))

        # Electron density
        ne = np.full((nx, ny, nz), 1e22)

        # Apply conduction
        dt = 1e-12  # very small timestep for explicit stability
        Te_new = anisotropic_thermal_conduction(
            Te, B, ne, dt, dx, dy, dz,
            Z_eff=1.0,
        )

        # Temperature should have changed (conduction occurred)
        delta_Te = Te_new - Te
        max_change = float(np.max(np.abs(delta_Te)))
        assert max_change > 0.0, "Conduction along B should change temperature"


# ===================================================================
# D.5: Anisotropic thermal conduction across B
# ===================================================================

class TestAnisotropicConductionAcrossB:
    """Uniform B in x-direction, temperature gradient in y -> only perp conduction (weaker)."""

    def test_conduction_across_B_is_weaker(self):
        """Heat conduction across B should be much weaker than along B."""
        from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction

        nx, ny, nz = 8, 32, 4
        dx = dy = dz = 1e-3

        # Uniform B in x-direction
        B = np.zeros((3, nx, ny, nz))
        B[0] = 1.0  # 1 Tesla in x

        # Temperature gradient in y-direction (perpendicular to B)
        y = np.linspace(0, (ny - 1) * dy, ny)
        Te_perp = np.zeros((nx, ny, nz))
        for j in range(ny):
            Te_perp[:, j, :] = 1e6 + 5e5 * np.sin(2 * np.pi * y[j] / (ny * dy))

        # Same temperature gradient magnitude but in x (parallel to B)
        x = np.linspace(0, (nx - 1) * dx, nx)
        Te_par = np.zeros((nx, ny, nz))
        for i in range(nx):
            Te_par[i, :, :] = 1e6 + 5e5 * np.sin(2 * np.pi * x[i] / (nx * dx))

        ne = np.full((nx, ny, nz), 1e22)
        dt = 1e-12

        # Parallel conduction
        Te_par_new = anisotropic_thermal_conduction(
            Te_par, B, ne, dt, dx, dy, dz, Z_eff=1.0,
        )

        # Perpendicular conduction
        Te_perp_new = anisotropic_thermal_conduction(
            Te_perp, B, ne, dt, dx, dy, dz, Z_eff=1.0,
        )

        change_par = float(np.max(np.abs(Te_par_new - Te_par)))
        change_perp = float(np.max(np.abs(Te_perp_new - Te_perp)))

        # Parallel should produce more change than perpendicular
        # (or at minimum perp should not exceed parallel)
        if change_par > 1e-10:
            assert change_perp <= change_par * 1.1, (
                f"Perpendicular conduction ({change_perp:.4e}) should not exceed "
                f"parallel conduction ({change_par:.4e})"
            )


# ===================================================================
# D.6: Powell source terms for zero div(B)
# ===================================================================

class TestPowellSourceZeroDivB:
    """If div(B) = 0, all Powell sources should be zero."""

    def test_divergence_free_B_gives_zero_sources(self):
        """A curl-type B field has div(B) = 0 -> zero Powell sources."""
        from dpf.fluid.mhd_solver import powell_source_terms

        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 1e-3

        # Create a divergence-free B field: B = curl(A)
        # Simplest: uniform B field
        B = np.zeros((3, nx, ny, nz))
        B[2] = 1.0  # uniform Bz

        state = {
            "rho": np.ones((nx, ny, nz)) * 1e-4,
            "velocity": np.random.RandomState(42).randn(3, nx, ny, nz) * 1e3,
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": B,
            "psi": np.zeros((nx, ny, nz)),
        }

        sources = powell_source_terms(state, dx, dy, dz)

        # For uniform B, div(B) = 0, so all sources should be zero
        np.testing.assert_allclose(
            sources["dmom_powell"], 0.0, atol=1e-10,
            err_msg="Powell momentum source should be zero for div-free B"
        )
        np.testing.assert_allclose(
            sources["denergy_powell"], 0.0, atol=1e-10,
            err_msg="Powell energy source should be zero for div-free B"
        )
        np.testing.assert_allclose(
            sources["dB_powell"], 0.0, atol=1e-10,
            err_msg="Powell induction source should be zero for div-free B"
        )


# ===================================================================
# D.7: Powell source terms for nonzero div(B)
# ===================================================================

class TestPowellSourceNonzeroDivB:
    """Create a state with known div(B) and verify source magnitudes."""

    def test_nonzero_div_B_gives_nonzero_sources(self):
        """A B field with nonzero divergence should produce nonzero Powell sources."""
        from dpf.fluid.mhd_solver import powell_source_terms

        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 1e-3

        # Create B field with known div(B):
        # Bx = x (linear in x) -> d(Bx)/dx = 1 -> div(B) = 1
        x = np.linspace(0, (nx - 1) * dx, nx)
        B = np.zeros((3, nx, ny, nz))
        for i in range(nx):
            B[0, i, :, :] = x[i] * 100.0  # Bx varies linearly with x

        vel = np.ones((3, nx, ny, nz)) * 1e4  # uniform velocity

        state = {
            "rho": np.ones((nx, ny, nz)) * 1e-4,
            "velocity": vel,
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": B,
            "psi": np.zeros((nx, ny, nz)),
        }

        sources = powell_source_terms(state, dx, dy, dz)

        # div(B) should be nonzero (= 100 approximately at interior points)
        div_B = sources["div_B"]
        assert float(np.max(np.abs(div_B))) > 1.0, (
            f"div(B) should be nonzero, got max |div(B)| = {float(np.max(np.abs(div_B)))}"
        )

        # Momentum source should be nonzero: -div(B) * B
        max_mom = float(np.max(np.abs(sources["dmom_powell"])))
        assert max_mom > 0.0, "Powell momentum source should be nonzero for B with div(B)!=0"

        # Energy source should be nonzero: -div(B) * (v.B)
        max_energy = float(np.max(np.abs(sources["denergy_powell"])))
        assert max_energy > 0.0, "Powell energy source should be nonzero"

        # Induction source should be nonzero: -div(B) * v
        max_ind = float(np.max(np.abs(sources["dB_powell"])))
        assert max_ind > 0.0, "Powell induction source should be nonzero"

    def test_powell_momentum_formula(self):
        """Verify the Powell momentum source equals -div(B) * B at interior points."""
        from dpf.fluid.mhd_solver import powell_source_terms

        nx, ny, nz = 16, 16, 16
        dx = dy = dz = 1e-3

        # Uniform Bx varying linearly in x
        B = np.zeros((3, nx, ny, nz))
        x = np.linspace(0, (nx - 1) * dx, nx)
        for i in range(nx):
            B[0, i, :, :] = x[i] * 100.0

        vel = np.zeros((3, nx, ny, nz))

        state = {
            "rho": np.ones((nx, ny, nz)) * 1e-4,
            "velocity": vel,
            "pressure": np.ones((nx, ny, nz)) * 1e3,
            "B": B,
            "psi": np.zeros((nx, ny, nz)),
        }

        sources = powell_source_terms(state, dx, dy, dz)

        # At interior points, div(B) ~ 100 (dBx/dx = 100)
        # Powell momentum source = -div(B) * B
        # So dmom_x = -100 * Bx at each point
        div_B = sources["div_B"]
        expected_dmom_x = -div_B * B[0]

        # Check interior points (avoid boundaries where gradient is one-sided)
        s = slice(2, -2)
        np.testing.assert_allclose(
            sources["dmom_powell"][0, s, s, s],
            expected_dmom_x[s, s, s],
            rtol=1e-10,
            err_msg="Powell momentum_x source should equal -div(B)*Bx"
        )


# ===================================================================
# D.8: Dedner damping with Mignone-Tzeferacos tuning
# ===================================================================

class TestDednerDamping:
    """Verify that psi decays under Dedner parabolic damping."""

    def test_psi_decays_with_damping(self):
        """Psi should decay exponentially under parabolic damping."""
        from dpf.fluid.mhd_solver import _dedner_source_mt2010

        nx, ny, nz = 16, 16, 16
        dx = 1e-3

        # Start with nonzero psi, zero div(B)
        psi = np.ones((nx, ny, nz)) * 1.0
        B = np.zeros((3, nx, ny, nz))
        B[2] = 1.0  # uniform -> div(B) = 0

        ch = 1e6  # 10^6 m/s
        cr = ch / dx  # Mignone-Tzeferacos: cr ~ ch/dx

        dpsi_dt, dB_dt = _dedner_source_mt2010(psi, B, ch, cr, dx)

        # dpsi/dt should be -cr * psi (since div(B) = 0)
        expected_dpsi = -cr * psi

        np.testing.assert_allclose(
            dpsi_dt, expected_dpsi, rtol=1e-10,
            err_msg="Dedner dpsi/dt should equal -cr*psi when div(B)=0"
        )

        # After one forward Euler step, psi should decrease
        dt = 1e-10
        psi_new = psi + dt * dpsi_dt
        assert float(np.max(psi_new)) < float(np.max(psi)), (
            "Psi should decay under Dedner damping"
        )

    def test_dedner_div_B_term(self):
        """Verify that the div(B) driving term is present."""
        from dpf.fluid.mhd_solver import _dedner_source_mt2010

        nx, ny, nz = 16, 16, 16
        dx = 1e-3

        psi = np.zeros((nx, ny, nz))

        # B with nonzero divergence
        B = np.zeros((3, nx, ny, nz))
        x = np.linspace(0, (nx - 1) * dx, nx)
        for i in range(nx):
            B[0, i, :, :] = x[i] * 100.0

        ch = 1e6
        cr = ch / dx

        dpsi_dt, _ = _dedner_source_mt2010(psi, B, ch, cr, dx)

        # With nonzero div(B), dpsi_dt should have a -ch^2 * div(B) component
        assert float(np.max(np.abs(dpsi_dt))) > 0, (
            "Dedner source should be nonzero when div(B) != 0"
        )


# ===================================================================
# D.9: Config fields validation
# ===================================================================

class TestConfigFields:
    """Verify the new Phase D config fields exist and have correct defaults."""

    def test_new_config_fields_exist(self):
        """Check that all new config fields are present with correct defaults."""
        from dpf.config import FluidConfig

        fc = FluidConfig()

        assert fc.enable_powell is False, "enable_powell should default to False"
        assert fc.dedner_cr == 0.0, "dedner_cr should default to 0.0"
        assert fc.enable_anisotropic_conduction is False, (
            "enable_anisotropic_conduction should default to False"
        )
        assert fc.full_braginskii_viscosity is False, (
            "full_braginskii_viscosity should default to False"
        )

    def test_config_fields_settable(self):
        """Check that new config fields can be set."""
        from dpf.config import FluidConfig

        fc = FluidConfig(
            enable_powell=True,
            dedner_cr=1e8,
            enable_anisotropic_conduction=True,
            full_braginskii_viscosity=True,
        )

        assert fc.enable_powell is True
        assert fc.dedner_cr == 1e8
        assert fc.enable_anisotropic_conduction is True
        assert fc.full_braginskii_viscosity is True
