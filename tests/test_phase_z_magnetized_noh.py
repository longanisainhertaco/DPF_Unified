"""Phase Z: Magnetized Noh Z-pinch verification benchmark.

Tests the exact self-similar solution for the magnetized Noh problem
(Velikovich & Giuliani, Phys. Plasmas 19, 012707, 2012).  This is the
gold-standard Z-pinch benchmark used by MACH2 and Athena.

Test categories:
    - Compression ratio computation
    - Upstream/downstream profiles
    - Rankine-Hugoniot jump conditions
    - Self-similarity
    - DPF state dict creation
    - Solver convergence (when backend available)
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.constants import mu_0
from dpf.validation.magnetized_noh import (
    compression_ratio,
    create_noh_state,
    noh_downstream,
    noh_exact_solution,
    noh_upstream,
    shock_velocity,
    verify_rankine_hugoniot,
)

# ═══════════════════════════════════════════════════════
# Compression ratio tests
# ═══════════════════════════════════════════════════════


class TestCompressionRatio:
    """Test the shock compression ratio solver."""

    def test_hydro_gamma_5_3(self) -> None:
        """Unmagnetized (B=0) with gamma=5/3 should give X=4."""
        X = compression_ratio(5.0 / 3.0, 0.0)
        assert pytest.approx(4.0, rel=1e-10) == X

    def test_hydro_gamma_7_5(self) -> None:
        """Unmagnetized with gamma=7/5 should give X=6."""
        X = compression_ratio(7.0 / 5.0, 0.0)
        assert pytest.approx(6.0, rel=1e-10) == X

    def test_hydro_gamma_2(self) -> None:
        """Unmagnetized with gamma=2 should give X=3."""
        X = compression_ratio(2.0, 0.0)
        assert pytest.approx(3.0, rel=1e-10) == X

    def test_magnetized_reduces_compression(self) -> None:
        """Magnetic pressure should reduce the compression ratio."""
        X_hydro = compression_ratio(5.0 / 3.0, 0.0)
        X_mag = compression_ratio(5.0 / 3.0, 0.5)
        assert X_mag < X_hydro
        assert X_mag > 1.0

    def test_strong_field_limit(self) -> None:
        """Very strong B-field (large beta_A) -> X approaches 1."""
        X = compression_ratio(5.0 / 3.0, 100.0)
        assert X < 1.5
        assert X > 1.0

    def test_monotonic_in_beta_A(self) -> None:
        """Compression ratio decreases monotonically with beta_A."""
        betas = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
        Xs = [compression_ratio(5.0 / 3.0, b) for b in betas]
        for i in range(len(Xs) - 1):
            assert Xs[i] > Xs[i + 1]

    def test_negative_beta_A_raises(self) -> None:
        """Negative magnetization parameter should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            compression_ratio(5.0 / 3.0, -0.1)

    def test_compression_ratio_various_gamma(self) -> None:
        """Check compression ratio for several gamma values with moderate B."""
        for gamma in [1.2, 4.0 / 3.0, 5.0 / 3.0, 2.0, 3.0]:
            X = compression_ratio(gamma, 0.3)
            X_hydro = (gamma + 1.0) / (gamma - 1.0)
            assert 1.0 < X < X_hydro


# ═══════════════════════════════════════════════════════
# Shock velocity tests
# ═══════════════════════════════════════════════════════


class TestShockVelocity:
    """Test shock velocity computation."""

    def test_hydro_shock_speed(self) -> None:
        """V_s = V_0/(X-1) for unmagnetized gamma=5/3: V_s = V_0/3."""
        X = compression_ratio(5.0 / 3.0, 0.0)
        V_s = shock_velocity(1.0, X)
        assert V_s == pytest.approx(1.0 / 3.0, rel=1e-10)

    def test_magnetized_faster_shock(self) -> None:
        """Magnetic pressure drives a faster shock (smaller X -> larger V_s)."""
        X_hydro = compression_ratio(5.0 / 3.0, 0.0)
        X_mag = compression_ratio(5.0 / 3.0, 1.0)
        V_s_hydro = shock_velocity(1.0, X_hydro)
        V_s_mag = shock_velocity(1.0, X_mag)
        assert V_s_mag > V_s_hydro

    def test_velocity_scales_with_V0(self) -> None:
        """Shock velocity should scale linearly with V_0."""
        X = compression_ratio(5.0 / 3.0, 0.5)
        V_s_1 = shock_velocity(1.0, X)
        V_s_2 = shock_velocity(2.0, X)
        assert V_s_2 == pytest.approx(2.0 * V_s_1, rel=1e-10)


# ═══════════════════════════════════════════════════════
# Upstream profile tests
# ═══════════════════════════════════════════════════════


class TestUpstream:
    """Test upstream (pre-shock) profiles."""

    def test_initial_conditions(self) -> None:
        """At t=0+epsilon, upstream should be close to initial values."""
        r = np.linspace(0.1, 1.0, 50)
        rho, vr, Bt, p = noh_upstream(r, 1e-10, rho_0=1.0, V_0=1.0, B_0=0.5)
        np.testing.assert_allclose(rho, 1.0, atol=1e-8)
        np.testing.assert_allclose(vr, -1.0)
        np.testing.assert_allclose(Bt, 0.5, atol=1e-8)
        np.testing.assert_allclose(p, 0.0)

    def test_cylindrical_compression(self) -> None:
        """Density should increase as 1 + V_0*t/r (cylindrical convergence)."""
        r = np.array([0.5, 1.0, 2.0])
        t = 0.5
        rho, _, _, _ = noh_upstream(r, t, rho_0=1.0, V_0=1.0, B_0=0.0)
        expected = 1.0 + 0.5 / r
        np.testing.assert_allclose(rho, expected)

    def test_B_field_compression(self) -> None:
        """B_theta compresses same as density: B_0 * (1 + V_0*t/r)."""
        r = np.array([0.5, 1.0, 2.0])
        t = 0.3
        B_0 = 0.1
        _, _, Bt, _ = noh_upstream(r, t, rho_0=1.0, V_0=1.0, B_0=B_0)
        expected = B_0 * (1.0 + 0.3 / r)
        np.testing.assert_allclose(Bt, expected)

    def test_upstream_velocity_uniform(self) -> None:
        """Velocity should be uniform at -V_0 everywhere upstream."""
        r = np.linspace(0.1, 5.0, 100)
        _, vr, _, _ = noh_upstream(r, 1.0, rho_0=1.0, V_0=3.0, B_0=0.0)
        np.testing.assert_allclose(vr, -3.0)

    def test_upstream_cold(self) -> None:
        """Pressure should be zero everywhere upstream."""
        r = np.linspace(0.1, 5.0, 100)
        _, _, _, p = noh_upstream(r, 1.0, rho_0=1.0, V_0=1.0, B_0=1.0)
        np.testing.assert_allclose(p, 0.0)


# ═══════════════════════════════════════════════════════
# Downstream profile tests
# ═══════════════════════════════════════════════════════


class TestDownstream:
    """Test downstream (post-shock) uniform state."""

    def test_hydro_density(self) -> None:
        """Post-shock density for unmagnetized gamma=5/3: rho = 16*rho_0."""
        X = compression_ratio(5.0 / 3.0, 0.0)
        rho, _, _, _ = noh_downstream(1.0, 1.0, 0.0, 5.0 / 3.0, X)
        assert rho == pytest.approx(16.0, rel=1e-10)

    def test_hydro_stagnation(self) -> None:
        """Post-shock velocity should be zero (stagnation)."""
        X = compression_ratio(5.0 / 3.0, 0.0)
        _, vr, _, _ = noh_downstream(1.0, 1.0, 0.0, 5.0 / 3.0, X)
        assert vr == 0.0

    def test_hydro_pressure_positive(self) -> None:
        """Post-shock pressure should be positive."""
        X = compression_ratio(5.0 / 3.0, 0.0)
        _, _, _, p = noh_downstream(1.0, 1.0, 0.0, 5.0 / 3.0, X)
        assert p > 0

    def test_magnetized_density(self) -> None:
        """Post-shock density rho = X^2 * rho_0."""
        beta_A = 0.5
        X = compression_ratio(5.0 / 3.0, beta_A)
        rho, _, _, _ = noh_downstream(1.0, 1.0, 0.0, 5.0 / 3.0, X)
        assert rho == pytest.approx(X**2, rel=1e-10)

    def test_magnetized_B_field(self) -> None:
        """Post-shock B_theta = X^2 * B_0."""
        B_0 = 0.3
        rho_0 = 1.0
        V_0 = 1.0
        beta_A = B_0**2 / (mu_0 * rho_0 * V_0**2)
        X = compression_ratio(5.0 / 3.0, beta_A)
        _, _, Bt, _ = noh_downstream(rho_0, V_0, B_0, 5.0 / 3.0, X)
        assert Bt == pytest.approx(X**2 * B_0, rel=1e-10)

    def test_magnetized_pressure_positive(self) -> None:
        """Post-shock pressure should remain positive with magnetic field."""
        B_0 = 0.1
        rho_0 = 1.0
        V_0 = 1.0
        beta_A = B_0**2 / (mu_0 * rho_0 * V_0**2)
        X = compression_ratio(5.0 / 3.0, beta_A)
        _, _, _, p = noh_downstream(rho_0, V_0, B_0, 5.0 / 3.0, X)
        assert p > 0

    def test_magnetic_pressure_reduces_thermal(self) -> None:
        """Adding B-field should reduce post-shock thermal pressure."""
        X_hydro = compression_ratio(5.0 / 3.0, 0.0)
        _, _, _, p_hydro = noh_downstream(1.0, 1.0, 0.0, 5.0 / 3.0, X_hydro)

        B_0 = 0.1
        beta_A = B_0**2 / (mu_0 * 1.0 * 1.0**2)
        X_mag = compression_ratio(5.0 / 3.0, beta_A)
        _, _, _, p_mag = noh_downstream(1.0, 1.0, B_0, 5.0 / 3.0, X_mag)

        assert p_mag < p_hydro


# ═══════════════════════════════════════════════════════
# Rankine-Hugoniot verification
# ═══════════════════════════════════════════════════════


class TestRankineHugoniot:
    """Verify that the solution satisfies RH jump conditions."""

    @pytest.mark.parametrize("beta_A", [0.0, 0.1, 0.5, 1.0, 5.0])
    def test_rh_mass_flux(self, beta_A: float) -> None:
        """Mass flux should be continuous across the shock."""
        B_0 = np.sqrt(beta_A * mu_0 * 1.0 * 1.0**2)
        report = verify_rankine_hugoniot(1.0, 1.0, B_0, 5.0 / 3.0)
        assert abs(report["mass_residual"]) < 1e-10

    @pytest.mark.parametrize("beta_A", [0.0, 0.1, 0.5, 1.0, 5.0])
    def test_rh_momentum_flux(self, beta_A: float) -> None:
        """Momentum flux should be continuous across the shock."""
        B_0 = np.sqrt(beta_A * mu_0 * 1.0 * 1.0**2)
        report = verify_rankine_hugoniot(1.0, 1.0, B_0, 5.0 / 3.0)
        assert abs(report["momentum_residual"]) < 1e-10

    @pytest.mark.parametrize("beta_A", [0.0, 0.1, 0.5, 1.0, 5.0])
    def test_rh_energy_flux(self, beta_A: float) -> None:
        """Energy flux should be continuous across the shock."""
        B_0 = np.sqrt(beta_A * mu_0 * 1.0 * 1.0**2)
        report = verify_rankine_hugoniot(1.0, 1.0, B_0, 5.0 / 3.0)
        assert abs(report["energy_residual"]) < 1e-10

    @pytest.mark.parametrize("beta_A", [0.1, 0.5, 1.0, 5.0])
    def test_rh_induction(self, beta_A: float) -> None:
        """Tangential B flux should be continuous across the shock."""
        B_0 = np.sqrt(beta_A * mu_0 * 1.0 * 1.0**2)
        report = verify_rankine_hugoniot(1.0, 1.0, B_0, 5.0 / 3.0)
        assert abs(report["induction_residual"]) < 1e-10

    def test_rh_all_residuals_small(self) -> None:
        """All relative residuals should be < 1e-10."""
        B_0 = np.sqrt(0.3 * mu_0)
        report = verify_rankine_hugoniot(1.0, 1.0, B_0, 5.0 / 3.0)
        assert report["max_relative_residual"] < 1e-10

    @pytest.mark.parametrize("gamma", [1.2, 4.0 / 3.0, 5.0 / 3.0, 2.0])
    def test_rh_various_gamma(self, gamma: float) -> None:
        """RH should hold for various gamma values."""
        B_0 = np.sqrt(0.5 * mu_0)
        report = verify_rankine_hugoniot(1.0, 1.0, B_0, gamma)
        assert report["max_relative_residual"] < 1e-10


# ═══════════════════════════════════════════════════════
# Full exact solution tests
# ═══════════════════════════════════════════════════════


class TestExactSolution:
    """Test the full piecewise exact solution."""

    def test_discontinuity_at_shock(self) -> None:
        """Solution should be discontinuous at r = V_s * t."""
        t = 1.0
        exact = noh_exact_solution(np.array([0.1]), t, rho_0=1.0, V_0=1.0, B_0=0.0)
        r_shock = exact["r_shock"]

        r = np.array([r_shock - 0.001, r_shock + 0.001])
        sol = noh_exact_solution(r, t, rho_0=1.0, V_0=1.0, B_0=0.0)

        # Density should jump across shock
        assert sol["rho"][0] > sol["rho"][1]

    def test_negative_time_raises(self) -> None:
        """t <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            noh_exact_solution(np.array([1.0]), 0.0)

    def test_self_similarity(self) -> None:
        """Solution at (r, t) should equal solution at (alpha*r, alpha*t)."""
        r = np.linspace(0.1, 2.0, 50)
        t = 1.0
        alpha = 2.5

        sol1 = noh_exact_solution(r, t, rho_0=1.0, V_0=1.0, B_0=0.0)
        sol2 = noh_exact_solution(alpha * r, alpha * t, rho_0=1.0, V_0=1.0, B_0=0.0)

        # Density depends on r/t which is the same for both
        np.testing.assert_allclose(sol1["rho"], sol2["rho"], rtol=1e-10)
        np.testing.assert_allclose(sol1["vr"], sol2["vr"], rtol=1e-10)
        np.testing.assert_allclose(sol1["pressure"], sol2["pressure"], rtol=1e-10)

    def test_self_similarity_magnetized(self) -> None:
        """Self-similarity should hold for magnetized case too."""
        r = np.linspace(0.1, 2.0, 50)
        t = 1.0
        alpha = 3.0
        B_0 = 0.1

        sol1 = noh_exact_solution(r, t, B_0=B_0)
        sol2 = noh_exact_solution(alpha * r, alpha * t, B_0=B_0)

        np.testing.assert_allclose(sol1["rho"], sol2["rho"], rtol=1e-10)
        np.testing.assert_allclose(sol1["vr"], sol2["vr"], rtol=1e-10)
        np.testing.assert_allclose(sol1["B_theta"], sol2["B_theta"], rtol=1e-10)

    def test_shock_position(self) -> None:
        """Shock should be at r = V_s * t."""
        t = 2.0
        V_0 = 1.0
        sol = noh_exact_solution(np.array([0.1]), t, V_0=V_0)
        X = sol["X"]
        V_s = sol["V_s"]
        assert sol["r_shock"] == pytest.approx(V_s * t, rel=1e-10)
        assert V_s == pytest.approx(V_0 / (X - 1.0), rel=1e-10)

    def test_hydro_downstream_density(self) -> None:
        """Downstream density for B=0, gamma=5/3 should be 16*rho_0."""
        r = np.array([0.01, 0.05, 0.1])
        sol = noh_exact_solution(r, 1.0, rho_0=1.0, V_0=1.0, B_0=0.0)
        # All these points are downstream (r < V_s*t = 1/3)
        np.testing.assert_allclose(sol["rho"], 16.0, rtol=1e-10)

    def test_downstream_stagnation(self) -> None:
        """Downstream velocity should be zero."""
        r = np.array([0.01, 0.05, 0.1])
        sol = noh_exact_solution(r, 1.0, rho_0=1.0, V_0=1.0, B_0=0.0)
        np.testing.assert_allclose(sol["vr"], 0.0)

    def test_upstream_beyond_shock(self) -> None:
        """Points beyond the shock should have upstream profiles."""
        r = np.array([0.5, 1.0, 2.0])  # All > V_s*t = 1/3 for t=1
        sol = noh_exact_solution(r, 1.0, rho_0=1.0, V_0=1.0, B_0=0.0)
        expected_rho = 1.0 + 1.0 / r
        np.testing.assert_allclose(sol["rho"], expected_rho, rtol=1e-10)
        np.testing.assert_allclose(sol["vr"], -1.0)

    def test_returns_metadata(self) -> None:
        """Solution dict should include shock metadata."""
        sol = noh_exact_solution(np.array([0.1]), 1.0, B_0=0.1)
        assert "r_shock" in sol
        assert "V_s" in sol
        assert "X" in sol
        assert "beta_A" in sol

    def test_magnetized_solution_consistent(self) -> None:
        """Magnetized solution should have higher downstream B_theta."""
        B_0 = 0.1
        r = np.array([0.01])
        sol = noh_exact_solution(r, 1.0, B_0=B_0)
        X = sol["X"]
        assert sol["B_theta"][0] == pytest.approx(X**2 * B_0, rel=1e-10)


# ═══════════════════════════════════════════════════════
# DPF state dict creation
# ═══════════════════════════════════════════════════════


class TestCreateNohState:
    """Test creation of DPF-compatible state dict."""

    def test_state_dict_keys(self) -> None:
        """State dict should have all required DPF keys."""
        state, info = create_noh_state(32, 4, 1.0, 1.0)
        for key in ("rho", "velocity", "pressure", "B", "Te", "Ti", "psi"):
            assert key in state

    def test_state_dict_shapes(self) -> None:
        """Arrays should have correct (nr, 1, nz) cylindrical shapes."""
        nr, nz = 32, 8
        state, _ = create_noh_state(nr, nz, 1.0, 1.0)
        assert state["rho"].shape == (nr, 1, nz)
        assert state["pressure"].shape == (nr, 1, nz)
        assert state["velocity"].shape == (3, nr, 1, nz)
        assert state["B"].shape == (3, nr, 1, nz)

    def test_info_metadata(self) -> None:
        """Info dict should contain shock metadata."""
        _, info = create_noh_state(32, 4, 1.0, 1.0)
        assert "r_shock" in info
        assert "V_s" in info
        assert "X" in info
        assert "r_centers" in info

    def test_hydro_density_profile(self) -> None:
        """Radial density profile should match exact solution."""
        nr = 64
        state, info = create_noh_state(nr, 4, 1.0, 1.0)
        r = info["r_centers"]
        rho_1d = state["rho"][:, 0, 0]

        exact = noh_exact_solution(r, 1.0, rho_0=1.0, V_0=1.0, B_0=0.0)
        np.testing.assert_allclose(rho_1d, exact["rho"], rtol=1e-10)

    def test_B_theta_in_B_array(self) -> None:
        """B_theta should be in B[1] (theta component)."""
        B_0 = 0.1
        state, _ = create_noh_state(32, 4, 1.0, 1.0, B_0=B_0)
        # B_r = B[0] should be zero
        np.testing.assert_allclose(state["B"][0], 0.0)
        # B_theta = B[1] should have the Noh profile
        assert np.any(state["B"][1] != 0.0)
        # B_z = B[2] should be zero
        np.testing.assert_allclose(state["B"][2], 0.0)

    def test_v_r_in_velocity(self) -> None:
        """Radial velocity should be in velocity[0]."""
        state, info = create_noh_state(32, 4, 1.0, 1.0)
        r_shock = info["r_shock"]
        r = info["r_centers"]

        # Upstream cells should have v_r = -V_0
        upstream_mask = r > r_shock
        if np.any(upstream_mask):
            vr_upstream = state["velocity"][0, upstream_mask, 0, 0]
            np.testing.assert_allclose(vr_upstream, -1.0)

    def test_axial_uniformity(self) -> None:
        """Profiles should be uniform along z-axis."""
        state, _ = create_noh_state(32, 8, 1.0, 1.0)
        for iz in range(8):
            np.testing.assert_allclose(
                state["rho"][:, 0, iz], state["rho"][:, 0, 0]
            )


# ═══════════════════════════════════════════════════════
# Mass / momentum / energy conservation checks
# ═══════════════════════════════════════════════════════


class TestConservation:
    """Test conservation properties of the exact solution."""

    def test_mass_conservation(self) -> None:
        """Total mass per unit length should grow from swept-up material."""
        # At time t, total mass = integral of rho * 2*pi*r*dr
        t = 1.0
        r = np.linspace(0.001, 3.0, 10000)
        dr = r[1] - r[0]

        sol = noh_exact_solution(r, t, rho_0=1.0, V_0=1.0, B_0=0.0)

        # Integrated mass per unit length
        mass = np.sum(sol["rho"] * 2 * np.pi * r * dr)

        # Expected: mass at t=0 within r < r_max + V_0*t was in shell
        # [r, r+V_0*t] initially, contributing rho_0 * 2*pi*r_0*dr_0
        # Total initial mass within r_max = 3: pi*rho_0*r_max^2
        # Mass that was at r_0 = r + V_0*t at t=0 and has since moved
        # to r at time t.  Everything from r_0 in [V_0*t, r_max+V_0*t]
        # maps to r in [0, r_max].
        r_max = 3.0
        r_0_max = r_max + 1.0 * t
        total_initial_mass = np.pi * 1.0 * r_0_max**2
        assert mass == pytest.approx(total_initial_mass, rel=0.02)

    def test_momentum_downstream_zero(self) -> None:
        """Total radial momentum in downstream should be ~zero (stagnant)."""
        t = 1.0
        r = np.linspace(0.001, 0.3, 1000)  # All downstream
        dr = r[1] - r[0]

        sol = noh_exact_solution(r, t, rho_0=1.0, V_0=1.0, B_0=0.0)
        momentum = np.sum(sol["rho"] * sol["vr"] * 2 * np.pi * r * dr)
        assert abs(momentum) < 1e-10

    def test_total_pressure_positive(self) -> None:
        """Total pressure (thermal + magnetic) should be positive everywhere."""
        r = np.linspace(0.01, 2.0, 500)
        sol = noh_exact_solution(r, 1.0, B_0=0.1)

        p_thermal = sol["pressure"]
        p_magnetic = sol["B_theta"] ** 2 / (2.0 * mu_0)
        p_total = p_thermal + p_magnetic

        assert np.all(p_total >= 0)
        assert np.all(p_thermal >= 0)


# ═══════════════════════════════════════════════════════
# Physical parameter scaling tests
# ═══════════════════════════════════════════════════════


class TestScaling:
    """Test dimensional analysis and scaling properties."""

    def test_density_scales_with_rho_0(self) -> None:
        """All densities should scale linearly with rho_0."""
        r = np.linspace(0.1, 2.0, 50)
        sol1 = noh_exact_solution(r, 1.0, rho_0=1.0, V_0=1.0, B_0=0.0)
        sol2 = noh_exact_solution(r, 1.0, rho_0=3.0, V_0=1.0, B_0=0.0)
        np.testing.assert_allclose(sol2["rho"], 3.0 * sol1["rho"], rtol=1e-10)

    def test_pressure_scales(self) -> None:
        """Pressure should scale as rho_0 * V_0^2."""
        r = np.array([0.01])  # downstream
        sol1 = noh_exact_solution(r, 1.0, rho_0=1.0, V_0=1.0)
        sol2 = noh_exact_solution(r, 1.0, rho_0=2.0, V_0=3.0)
        # p ~ rho_0 * V_0^2 * X^2/(X-1)
        ratio = (2.0 * 9.0) / (1.0 * 1.0)
        np.testing.assert_allclose(
            sol2["pressure"], ratio * sol1["pressure"], rtol=1e-10
        )
