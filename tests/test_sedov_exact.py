"""Tests for the exact Sedov-Taylor blast wave solution.

Validates the SedovExact class against known analytical results from
Kamm & Timmes (2007), LA-UR-07-2849 and Sedov (1959).

Key checks:
- Dimensionless energy integral alpha matches published values
- Shock radius follows R ~ t^(2/(n+2)) scaling
- Rankine-Hugoniot jump conditions at the shock
- Self-similarity (solutions at different times collapse when rescaled)
- Energy conservation (integrated energy = E_blast)
"""

import numpy as np
import pytest

from dpf.validation.sedov_exact import SedovExact

# --- Alpha values from Kamm & Timmes (2007), Table 1 ---


class TestAlphaValues:
    """Verify dimensionless energy integral alpha against published values."""

    def test_spherical_gamma_5_3(self):
        """Spherical (n=3), gamma=5/3: alpha ~ 0.4936 (Sedov 1959)."""
        sol = SedovExact(geometry=3, gamma=5.0 / 3.0)
        assert sol.get_alpha() == pytest.approx(0.4936, rel=5e-3)

    def test_spherical_gamma_7_5(self):
        """Spherical (n=3), gamma=7/5: alpha ~ 0.8510 (Kamm & Timmes)."""
        sol = SedovExact(geometry=3, gamma=7.0 / 5.0)
        assert sol.get_alpha() == pytest.approx(0.8510, rel=5e-3)

    def test_cylindrical_gamma_5_3(self):
        """Cylindrical (n=2), gamma=5/3: alpha ~ 0.5643."""
        sol = SedovExact(geometry=2, gamma=5.0 / 3.0)
        assert sol.get_alpha() == pytest.approx(0.5643, rel=5e-3)

    def test_cylindrical_gamma_7_5(self):
        """Cylindrical (n=2), gamma=7/5: alpha ~ 0.9841."""
        sol = SedovExact(geometry=2, gamma=7.0 / 5.0)
        assert sol.get_alpha() == pytest.approx(0.9841, rel=5e-3)

    def test_planar_gamma_5_3(self):
        """Planar (n=1), gamma=5/3: alpha ~ 0.3015."""
        sol = SedovExact(geometry=1, gamma=5.0 / 3.0)
        assert sol.get_alpha() == pytest.approx(0.3015, rel=5e-3)

    def test_planar_gamma_7_5(self):
        """Planar (n=1), gamma=7/5: alpha ~ 0.5387."""
        sol = SedovExact(geometry=1, gamma=7.0 / 5.0)
        assert sol.get_alpha() == pytest.approx(0.5387, rel=5e-3)


class TestShockRadius:
    """Verify shock radius scaling R ~ t^(2/(n+2))."""

    @pytest.mark.parametrize("geometry,exponent", [(1, 2.0 / 3.0), (2, 0.5), (3, 0.4)])
    def test_shock_radius_power_law(self, geometry, exponent):
        """R_shock should scale as t^(2/(n+2)) for uniform medium."""
        sol = SedovExact(geometry=geometry, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)

        t1, t2 = 0.01, 0.04
        r1 = sol.shock_radius(t1)
        r2 = sol.shock_radius(t2)

        # R2/R1 = (t2/t1)^exponent
        ratio = r2 / r1
        expected = (t2 / t1) ** exponent
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_shock_radius_energy_scaling(self):
        """Doubling energy should increase R by 2^(1/5) for spherical."""
        sol1 = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)
        sol2 = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=2.0, rho0=1.0)
        t = 0.01
        ratio = sol2.shock_radius(t) / sol1.shock_radius(t)
        assert ratio == pytest.approx(2.0 ** (1.0 / 5.0), rel=1e-6)

    def test_shock_radius_density_scaling(self):
        """Doubling density should decrease R by 2^(-1/5) for spherical."""
        sol1 = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)
        sol2 = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=2.0)
        t = 0.01
        ratio = sol2.shock_radius(t) / sol1.shock_radius(t)
        assert ratio == pytest.approx(2.0 ** (-1.0 / 5.0), rel=1e-6)


class TestRankineHugoniot:
    """Verify Rankine-Hugoniot jump conditions at the shock front."""

    @pytest.mark.parametrize("geometry", [1, 2, 3])
    def test_density_jump(self, geometry):
        """Post-shock density = (gamma+1)/(gamma-1) * pre-shock density."""
        gamma = 5.0 / 3.0
        sol = SedovExact(geometry=geometry, gamma=gamma, eblast=1.0, rho0=1.0)
        info = sol.get_shock_info(t=0.01)

        compression = info["rho_post"] / info["rho_pre"]
        expected = (gamma + 1) / (gamma - 1)
        assert compression == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("geometry", [1, 2, 3])
    def test_velocity_jump(self, geometry):
        """Post-shock velocity = 2*U_s/(gamma+1) (strong shock limit)."""
        gamma = 5.0 / 3.0
        sol = SedovExact(geometry=geometry, gamma=gamma, eblast=1.0, rho0=1.0)
        info = sol.get_shock_info(t=0.01)

        assert info["u_post"] == pytest.approx(
            2.0 * info["U_shock"] / (gamma + 1), rel=1e-10
        )

    @pytest.mark.parametrize("geometry", [1, 2, 3])
    def test_pressure_jump(self, geometry):
        """Post-shock pressure = 2*rho1*Us^2/(gamma+1) (strong shock)."""
        gamma = 5.0 / 3.0
        sol = SedovExact(geometry=geometry, gamma=gamma, eblast=1.0, rho0=1.0)
        info = sol.get_shock_info(t=0.01)

        expected_p = 2.0 * info["rho_pre"] * info["U_shock"] ** 2 / (gamma + 1)
        assert info["p_post"] == pytest.approx(expected_p, rel=1e-10)


class TestSelfSimilarity:
    """Verify that solutions at different times collapse under rescaling."""

    def test_spherical_self_similarity(self):
        """Solutions at t=0.01 and t=0.04 should match when rescaled by R_shock."""
        sol = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)

        t1, t2 = 0.01, 0.04
        r1_s = sol.shock_radius(t1)
        r2_s = sol.shock_radius(t2)

        # Evaluate at same lambda = r/R_shock
        lam_pts = np.linspace(0.1, 0.95, 20)

        r_pts_1 = lam_pts * r1_s
        r_pts_2 = lam_pts * r2_s

        _, rho1, p1, v1, _ = sol.evaluate(r_pts_1, t1)
        _, rho2, p2, v2, _ = sol.evaluate(r_pts_2, t2)

        # Rescale to dimensionless: rho/rho_post, p/p_post, v/u_post
        info1 = sol.get_shock_info(t1)
        info2 = sol.get_shock_info(t2)

        rho1_dim = rho1 / info1["rho_post"]
        rho2_dim = rho2 / info2["rho_post"]

        # Allow some interpolation tolerance
        np.testing.assert_allclose(rho1_dim, rho2_dim, rtol=0.05, atol=1e-6)


class TestSolutionProfiles:
    """Test basic properties of the solution profiles."""

    def test_density_positive(self):
        """Density should be positive everywhere."""
        sol = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)
        r_pts = np.linspace(0.01, 0.5, 200)
        _, rho, _, _, _ = sol.evaluate(r_pts, t=0.01)
        assert np.all(rho > 0)

    def test_pressure_non_negative(self):
        """Pressure should be non-negative everywhere."""
        sol = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)
        r_pts = np.linspace(0.01, 0.5, 200)
        _, _, p, _, _ = sol.evaluate(r_pts, t=0.01)
        assert np.all(p >= 0)

    def test_velocity_decreases_toward_origin(self):
        """Velocity should decrease toward the origin (interior of blast)."""
        sol = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)
        r_shock = sol.shock_radius(0.01)
        # Sample at 10%, 30%, 60% of shock radius — well within interpolation range
        r_pts = np.array([0.1, 0.3, 0.6]) * r_shock
        _, _, _, v, _ = sol.evaluate(r_pts, t=0.01)
        # Velocity should be monotonically increasing with r (v=0 at origin, max at shock)
        assert v[0] < v[1] < v[2]
        # All should be less than post-shock velocity
        info = sol.get_shock_info(0.01)
        assert np.all(v < info["u_post"] * 1.05)

    def test_ambient_outside_shock(self):
        """Outside the shock, density = rho0, velocity = 0, pressure = 0."""
        sol = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)
        r_shock = sol.shock_radius(0.01)
        r_pts = np.linspace(r_shock * 1.1, r_shock * 2.0, 50)
        _, rho, p, v, _ = sol.evaluate(r_pts, t=0.01)
        np.testing.assert_allclose(rho, 1.0, rtol=1e-10)
        np.testing.assert_allclose(v, 0.0, atol=1e-10)
        np.testing.assert_allclose(p, 0.0, atol=1e-10)

    def test_density_peak_near_shock(self):
        """Peak density should be near the shock front."""
        sol = SedovExact(geometry=3, gamma=5.0 / 3.0, eblast=1.0, rho0=1.0)
        r_shock = sol.shock_radius(0.01)
        r_pts = np.linspace(0.01, r_shock * 1.1, 500)
        _, rho, _, _, _ = sol.evaluate(r_pts, t=0.01)

        # Find peak density location
        i_max = np.argmax(rho)
        r_peak = r_pts[i_max]
        # Peak should be within 5% of shock radius
        assert abs(r_peak - r_shock) / r_shock < 0.05


class TestEnergyConservation:
    """Verify that the total energy integrates to E_blast."""

    @pytest.mark.parametrize("geometry", [2, 3])
    def test_energy_integral(self, geometry):
        """Integrated KE + IE should equal E_blast for the standard problem."""
        gamma = 5.0 / 3.0
        eblast = 1.0
        rho0 = 1.0
        sol = SedovExact(geometry=geometry, gamma=gamma, eblast=eblast, rho0=rho0)

        t = 0.01
        r_shock = sol.shock_radius(t)

        # Fine grid for integration — use 5000 points for better quadrature
        npts = 5000
        r_pts = np.linspace(1e-6, r_shock * 0.999, npts)
        _, rho, p, v, _ = sol.evaluate(r_pts, t, npts=10001)

        # KE = 0.5 * rho * v^2, IE = p / (gamma - 1)
        ke_density = 0.5 * rho * v**2
        ie_density = p / (gamma - 1.0)
        e_density = ke_density + ie_density

        if geometry == 3:
            # Volume element: 4*pi*r^2*dr
            E_total = np.trapezoid(4.0 * np.pi * r_pts**2 * e_density, r_pts)
        elif geometry == 2:
            # Volume element: 2*pi*r*dr (per unit length)
            E_total = np.trapezoid(2.0 * np.pi * r_pts * e_density, r_pts)

        # Allow 10% tolerance — the sharp density/pressure peak at the shock
        # is hard to integrate accurately with trapezoidal quadrature on
        # the interpolated profile.
        assert E_total == pytest.approx(eblast, rel=0.10)


class TestInputValidation:
    """Test input validation."""

    def test_invalid_geometry(self):
        with pytest.raises(ValueError, match="geometry"):
            SedovExact(geometry=4)

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            SedovExact(gamma=0.5)

    def test_invalid_rho0(self):
        with pytest.raises(ValueError, match="rho0"):
            SedovExact(rho0=-1.0)

    def test_invalid_eblast(self):
        with pytest.raises(ValueError, match="eblast"):
            SedovExact(eblast=0.0)

    def test_invalid_omega(self):
        with pytest.raises(ValueError, match="omega"):
            SedovExact(geometry=3, omega=3.0)

    def test_invalid_time(self):
        sol = SedovExact()
        with pytest.raises(ValueError, match="t must be > 0"):
            sol.evaluate(np.linspace(0, 1, 10), t=0.0)


class TestShockInfo:
    """Test the get_shock_info method."""

    def test_shock_info_keys(self):
        sol = SedovExact()
        info = sol.get_shock_info(t=0.01)
        expected_keys = {"R_shock", "U_shock", "rho_pre", "rho_post", "u_post", "p_post", "alpha"}
        assert set(info.keys()) == expected_keys

    def test_shock_info_positive_values(self):
        sol = SedovExact()
        info = sol.get_shock_info(t=0.01)
        for key, val in info.items():
            assert val > 0, f"{key} should be positive"

    def test_shock_info_consistency(self):
        """shock_info R_shock should match shock_radius()."""
        sol = SedovExact()
        t = 0.01
        info = sol.get_shock_info(t)
        assert info["R_shock"] == pytest.approx(sol.shock_radius(t), rel=1e-12)
        assert info["U_shock"] == pytest.approx(sol.shock_velocity(t), rel=1e-12)
        assert info["alpha"] == pytest.approx(sol.get_alpha(), rel=1e-12)
