"""Tests for tearing mode growth rate diagnostic (Challenge 7).

Covers:
- Return dict structure and key presence
- Physical scaling laws (S^{-3/5}, tau_A^{-1})
- Boundary conditions: zero-B, zero-eta, low-S
- Vector vs scalar B input
- Stochastic IC properties (Challenge 9): density floor, perturbation amplitude
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.diagnostics.instability import tearing_mode_growth_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_B_rho(n: int = 16, B_val: float = 1.0, rho_val: float = 1e-6) -> tuple:
    """Return (B_vec, rho) arrays for a uniform n-cell 1-D domain."""
    B = np.full((3, n), 0.0)
    B[1] = B_val  # B_theta component
    rho = np.full(n, rho_val)
    return B, rho


# ---------------------------------------------------------------------------
# Feature 1 — tearing_mode_growth_rate
# ---------------------------------------------------------------------------


class TestTearingModeReturnStructure:
    """Return dict always has the required keys with correct types."""

    def test_keys_present(self):
        B, rho = _uniform_B_rho()
        result = tearing_mode_growth_rate(B, rho, eta=1e-6, dx=1e-3)
        required = {
            "growth_rate", "growth_time", "lundquist_number",
            "alfven_speed", "alfven_time", "system_scale", "is_tearing",
        }
        assert required.issubset(result.keys())

    def test_growth_rate_nonnegative(self):
        B, rho = _uniform_B_rho()
        result = tearing_mode_growth_rate(B, rho, eta=1e-6, dx=1e-3)
        assert result["growth_rate"] >= 0.0

    def test_growth_time_positive_or_inf(self):
        B, rho = _uniform_B_rho()
        result = tearing_mode_growth_rate(B, rho, eta=1e-6, dx=1e-3)
        assert result["growth_time"] > 0.0

    def test_is_tearing_bool(self):
        B, rho = _uniform_B_rho()
        result = tearing_mode_growth_rate(B, rho, eta=1e-6, dx=1e-3)
        assert isinstance(result["is_tearing"], (bool, np.bool_))


class TestTearingModePhysics:
    """FKR scaling: gamma ~ S^{-3/5} / tau_A and related relations."""

    def test_higher_resistivity_raises_growth_rate(self):
        """Higher eta → lower S → faster tearing (FKR: gamma ~ S^{-3/5})."""
        B, rho = _uniform_B_rho(n=32, B_val=1.0, rho_val=1e-6)
        dx = 1e-3
        r_lo = tearing_mode_growth_rate(B, rho, eta=1e-7, dx=dx)["growth_rate"]
        r_hi = tearing_mode_growth_rate(B, rho, eta=1e-4, dx=dx)["growth_rate"]
        assert r_hi > r_lo, "Higher eta should give faster tearing growth"

    def test_stronger_B_raises_alfven_speed(self):
        """Doubling B should roughly double v_A."""
        rho = np.full(16, 1e-6)
        B1, _ = _uniform_B_rho(B_val=1.0)
        B2, _ = _uniform_B_rho(B_val=2.0)
        vA1 = tearing_mode_growth_rate(B1, rho, eta=1e-6, dx=1e-3)["alfven_speed"]
        vA2 = tearing_mode_growth_rate(B2, rho, eta=1e-6, dx=1e-3)["alfven_speed"]
        assert pytest.approx(vA2 / vA1, rel=1e-6) == 2.0

    def test_lundquist_number_formula(self):
        """S = mu_0 * v_A * L / eta should match analytic value."""
        mu_0 = 4 * np.pi * 1e-7
        B_val = 1.0
        rho_val = 1e-6
        n = 16
        dx = 1e-3
        eta = 1e-5
        v_A = B_val / np.sqrt(mu_0 * rho_val)
        L = n * dx
        S_expected = mu_0 * v_A * L / eta

        B, rho = _uniform_B_rho(n=n, B_val=B_val, rho_val=rho_val)
        result = tearing_mode_growth_rate(B, rho, eta=eta, dx=dx)
        assert pytest.approx(result["lundquist_number"], rel=1e-5) == S_expected

    def test_alfven_time_formula(self):
        """tau_A = L / v_A should match the returned alfven_time."""
        B, rho = _uniform_B_rho(n=16, B_val=2.0, rho_val=1e-5)
        dx = 5e-4
        result = tearing_mode_growth_rate(B, rho, eta=1e-6, dx=dx)
        tau_A_check = result["system_scale"] / result["alfven_speed"]
        assert pytest.approx(result["alfven_time"], rel=1e-6) == tau_A_check

    def test_growth_rate_scales_with_eta_power(self):
        """gamma ~ eta^{3/5} — check exponent from two eta values."""
        B, rho = _uniform_B_rho(n=64, B_val=1.0, rho_val=1e-6)
        dx = 1e-3
        eta1, eta2 = 1e-6, 1e-5
        g1 = tearing_mode_growth_rate(B, rho, eta=eta1, dx=dx)["growth_rate"]
        g2 = tearing_mode_growth_rate(B, rho, eta=eta2, dx=dx)["growth_rate"]
        # Expected ratio: (eta2/eta1)^{3/5}
        ratio_expected = (eta2 / eta1) ** 0.6
        ratio_actual = g2 / g1
        # Allow 5% tolerance — discrete grid effects perturb the exponent slightly
        assert abs(ratio_actual / ratio_expected - 1.0) < 0.05

    def test_scalar_B_input(self):
        """Scalar (non-vector) B array should be accepted and give same v_A."""
        n = 16
        B_val = 1.0
        rho_val = 1e-6
        B_scalar = np.full(n, B_val)
        rho = np.full(n, rho_val)
        B_vec, _ = _uniform_B_rho(n=n, B_val=B_val, rho_val=rho_val)

        r_scalar = tearing_mode_growth_rate(B_scalar, rho, eta=1e-6, dx=1e-3)
        r_vec = tearing_mode_growth_rate(B_vec, rho, eta=1e-6, dx=1e-3)
        # v_A and hence gamma should agree (both see same mean |B|)
        assert pytest.approx(r_scalar["alfven_speed"], rel=1e-6) == r_vec["alfven_speed"]


class TestTearingModeEdgeCases:
    """Degenerate inputs should not raise and should return sensible sentinels."""

    def test_zero_B_no_tearing(self):
        """Zero field → no Alfven speed → is_tearing should be False."""
        B = np.zeros((3, 16))
        rho = np.full(16, 1e-6)
        result = tearing_mode_growth_rate(B, rho, eta=1e-6, dx=1e-3)
        assert not result["is_tearing"]
        assert result["growth_rate"] == 0.0

    def test_low_lundquist_no_tearing(self):
        """Very large eta (S << 1) → is_tearing=False, growth_rate=0."""
        B, rho = _uniform_B_rho(B_val=0.01, rho_val=1e-2)
        # With tiny B and very large eta, S will be << 1
        result = tearing_mode_growth_rate(B, rho, eta=1.0, dx=1e-3)
        assert not result["is_tearing"]
        assert result["growth_rate"] == 0.0
        assert result["growth_time"] == float("inf")

    def test_3d_B_rho_arrays(self):
        """3-D spatial arrays (nr, ny, nz) should not raise."""
        nr, ny, nz = 8, 4, 8
        B = np.zeros((3, nr, ny, nz))
        B[1] = 1.0
        rho = np.full((nr, ny, nz), 1e-6)
        result = tearing_mode_growth_rate(B, rho, eta=1e-6, dx=1e-3)
        assert "growth_rate" in result
        assert np.isfinite(result["growth_rate"])


# ---------------------------------------------------------------------------
# Feature 2 — Stochastic IC properties (Challenge 9)
# ---------------------------------------------------------------------------


class TestStochasticIC:
    """Validate the stochastic density perturbation properties."""

    def _build_rho_3d(self, rng_seed=None):
        """Replicate the stochastic IC logic from app_mhd.py for testing."""
        nr, nz = 16, 32
        rho0 = 1e-4
        L_anode = 0.1
        rng = np.random.default_rng(rng_seed)
        delta_rho = 0.01
        noise = rng.normal(0, delta_rho, size=(nr, 1, nz))
        z_arr = np.linspace(0, L_anode, nz)
        m0_pert = 0.005 * np.sin(4 * np.pi * z_arr / L_anode)
        rho_3d = rho0 * (1.0 + noise + m0_pert[np.newaxis, np.newaxis, :])
        rho_3d = np.maximum(rho_3d, rho0 * 0.01)
        return rho_3d, rho0

    def test_density_floor_enforced(self):
        """All cells must be >= 1% of background density."""
        rho_3d, rho0 = self._build_rho_3d(rng_seed=0)
        assert np.all(rho_3d >= rho0 * 0.01), "Floor at 1% rho0 violated"

    def test_density_shape(self):
        """Output array has shape (nr, 1, nz)."""
        rho_3d, _ = self._build_rho_3d(rng_seed=1)
        assert rho_3d.shape == (16, 1, 32)

    def test_perturbation_amplitude_order_of_magnitude(self):
        """Mean density should be within ~5% of rho0 (perturbations cancel)."""
        rho_3d, rho0 = self._build_rho_3d(rng_seed=42)
        mean_ratio = float(np.mean(rho_3d)) / rho0
        assert 0.95 <= mean_ratio <= 1.10, f"Mean density ratio {mean_ratio:.3f} out of range"

    def test_two_shots_differ(self):
        """Without a fixed seed, two shots should produce different ICs."""
        # Build two ICs with unseeded RNG (different every call by design)
        rng_a = np.random.default_rng()
        rng_b = np.random.default_rng()
        nr, nz = 16, 32
        rho0 = 1e-4
        noise_a = rng_a.normal(0, 0.01, size=(nr, 1, nz))
        noise_b = rng_b.normal(0, 0.01, size=(nr, 1, nz))
        # Probability of identical draws from two independent RNGs is negligible
        assert not np.allclose(noise_a, noise_b), "Independent RNG draws should differ"

    def test_reproducible_with_fixed_seed(self):
        """Same seed → identical IC (for regression/reproduce studies)."""
        rho_a, _ = self._build_rho_3d(rng_seed=7)
        rho_b, _ = self._build_rho_3d(rng_seed=7)
        assert np.array_equal(rho_a, rho_b), "Fixed seed must give identical IC"

    def test_m0_component_present(self):
        """The structured m=0 sinusoidal component should be detectable in the mean
        over the radial axis — the azimuthal average of random noise is ~zero,
        leaving the m=0 signal."""
        nr, nz = 64, 32
        rho0 = 1e-4
        L_anode = 0.1
        # Use many radial cells so noise averages down
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, size=(nr, 1, nz))
        z_arr = np.linspace(0, L_anode, nz)
        m0_pert = 0.005 * np.sin(4 * np.pi * z_arr / L_anode)
        rho_3d = rho0 * (1.0 + noise + m0_pert[np.newaxis, np.newaxis, :])
        rho_3d = np.maximum(rho_3d, rho0 * 0.01)

        # Radial mean normalised by rho0 — should track m0_pert
        radial_mean = rho_3d[:, 0, :].mean(axis=0) / rho0 - 1.0
        correlation = float(np.corrcoef(radial_mean, m0_pert)[0, 1])
        assert correlation > 0.5, f"m=0 signature too weak (r={correlation:.3f})"
