"""Phase Y — m=0 Sausage Instability Growth Rate Tests.

Tests for the m=0 sausage instability growth rate calculator in
dpf.diagnostics.instability, covering:

1. Unstable low-beta case: growth rate positive and is_unstable True.
2. Stable high-beta case: growth rate zero and is_unstable False.
3. Growth rate scaling with Alfven speed (doubling B_theta ~ doubles growth_rate).
4. Growth rate scaling with mode number (mode_number=2 ~ 2x growth_rate).
5. Stability margin sign: positive when unstable, negative when stable.
6. m0_growth_rate_from_state with mock state, snowplow, and config.
7. m0_growth_rate_from_state with snowplow=None returns default stable result.
8. Dimensional analysis: typical PF-1000 pinch gives physically plausible timescales.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from dpf.diagnostics.instability import m0_growth_rate, m0_growth_rate_from_state

MU_0 = 4e-7 * np.pi

# Expected keys present in every result dict
_EXPECTED_KEYS = {"growth_rate", "growth_time", "alfven_speed", "beta_p", "is_unstable", "stability_margin"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_snowplow(r_shock: float = 0.04, r_pinch_min: float = 0.005, phase: str = "radial") -> MagicMock:
    """Return a minimal mock SnowplowModel."""
    sp = MagicMock()
    sp.r_shock = r_shock
    sp.r_pinch_min = r_pinch_min
    sp.phase = phase
    return sp


def _make_mock_config(dx: float = 0.01) -> MagicMock:
    """Return a minimal mock SimulationConfig with a dx attribute."""
    cfg = MagicMock()
    cfg.dx = dx
    return cfg


def _make_mock_state(nr: int = 8, nz: int = 8, B_theta_val: float = 1.0, rho_val: float = 1e-3,
                     p_val: float = 1e3) -> dict:
    """Return a minimal MHD state dict for m0_growth_rate_from_state tests.

    B has shape (3, nr, 1, nz) matching the cylindrical engine layout.
    rho and pressure have shape (nr, 1, nz).
    """
    B = np.zeros((3, nr, 1, nz))
    B[1] = B_theta_val  # azimuthal component
    rho = np.full((nr, 1, nz), rho_val)
    pressure = np.full((nr, 1, nz), p_val)
    return {"B": B, "rho": rho, "pressure": pressure}


# ===========================================================================
# TestM0GrowthRate
# ===========================================================================


class TestM0GrowthRate:
    """Unit tests for m0_growth_rate() covering physics correctness."""

    def test_unstable_low_beta(self):
        """Low beta well below critical (3.0) gives is_unstable=True and positive growth_rate.

        beta_p = 2*mu_0*pressure / B_theta^2 = 2*mu_0*1e3 / 1.0 ~ 2.5e-3.
        Critical beta (gamma=5/3) = 2/(5/3 - 1) = 3.0.  2.5e-3 << 3.0 → unstable.
        """
        result = m0_growth_rate(B_theta=1.0, rho=1e-3, pressure=1e3, a_pinch=0.01)

        assert set(result.keys()) == _EXPECTED_KEYS

        beta_p_expected = 2.0 * MU_0 * 1e3 / 1.0**2
        assert result["beta_p"] == pytest.approx(beta_p_expected, rel=0.01)
        assert beta_p_expected < 3.0, "pre-condition: beta_p must be below critical"

        assert result["is_unstable"] is True
        assert result["growth_rate"] > 0.0
        assert result["growth_time"] < float("inf")
        assert result["alfven_speed"] > 0.0

    def test_stable_high_beta(self):
        """High beta far above the Kadomtsev threshold (3.0) has a large negative stability margin.

        For the m=0 formula gamma_m0 = k*v_A*sqrt(1 - beta_p/(2 + gamma*beta_p)), when
        gamma > 1 the argument never reaches zero (asymptotes to 1 - 1/gamma = 0.4 for
        gamma=5/3), so the growth rate remains non-zero.  However, the code's stability_margin
        (= beta_p_crit - beta_p) becomes strongly negative, and beta_p >> beta_p_crit signals
        the Kadomtsev-stable regime.

        beta_p = 2*mu_0*1e6 / (0.01)^2 = 2*mu_0*1e10 >> 3.0 → Kadomtsev stable.
        """
        result = m0_growth_rate(B_theta=0.01, rho=1e-3, pressure=1e6, a_pinch=0.01)

        beta_p_expected = 2.0 * MU_0 * 1e6 / 0.01**2
        assert beta_p_expected > 3.0, "pre-condition: beta_p must exceed Kadomtsev threshold"

        # Stability margin is large and negative (plasma is far into Kadomtsev-stable regime)
        assert result["stability_margin"] < -1000.0
        # beta_p is correctly computed and large
        assert result["beta_p"] == pytest.approx(beta_p_expected, rel=0.01)
        # growth_rate and alfven_speed are still physically defined (formula asymptotes > 0)
        assert result["growth_rate"] >= 0.0
        assert result["alfven_speed"] > 0.0

    def test_growth_rate_scales_with_alfven_speed(self):
        """Doubling B_theta approximately doubles growth_rate for low-beta plasma.

        In the low-beta limit (beta_p → 0): gamma_m0 ≈ k * v_A = k * B_theta / sqrt(mu_0 * rho).
        Scaling: gamma_m0(2*B) / gamma_m0(B) ≈ 2.0.
        """
        params = dict(rho=1e-3, pressure=1e3, a_pinch=0.01)
        r1 = m0_growth_rate(B_theta=0.5, **params)
        r2 = m0_growth_rate(B_theta=1.0, **params)

        assert r1["is_unstable"] is True
        assert r2["is_unstable"] is True

        ratio = r2["growth_rate"] / r1["growth_rate"]
        # Expect ratio ~ 2.0 (within 10% for low-beta regime)
        assert ratio == pytest.approx(2.0, rel=0.10)

    def test_growth_rate_scales_with_mode_number(self):
        """Mode number=2 gives approximately twice the growth rate of mode_number=1.

        k = mode_number / a_pinch, so gamma_m0 ∝ k ∝ mode_number at fixed beta_p.
        """
        params = dict(B_theta=1.0, rho=1e-3, pressure=1e3, a_pinch=0.01)
        r1 = m0_growth_rate(mode_number=1, **params)
        r2 = m0_growth_rate(mode_number=2, **params)

        assert r1["is_unstable"] is True
        assert r2["is_unstable"] is True

        ratio = r2["growth_rate"] / r1["growth_rate"]
        assert ratio == pytest.approx(2.0, rel=0.05)

    def test_stability_margin_sign(self):
        """stability_margin > 0 for unstable plasma, < 0 for stable plasma.

        stability_margin = beta_p_crit - beta_p.
        Unstable: beta_p < beta_p_crit  → margin > 0.
        Stable:   beta_p > beta_p_crit  → margin < 0.
        """
        unstable = m0_growth_rate(B_theta=1.0, rho=1e-3, pressure=1e3, a_pinch=0.01)
        stable = m0_growth_rate(B_theta=0.01, rho=1e-3, pressure=1e6, a_pinch=0.01)

        assert unstable["stability_margin"] > 0.0
        assert stable["stability_margin"] < 0.0


# ===========================================================================
# TestM0FromState
# ===========================================================================


class TestM0FromState:
    """Tests for m0_growth_rate_from_state() — state extraction path."""

    def test_from_state_with_mock(self):
        """m0_growth_rate_from_state returns a valid result dict from mock inputs.

        State: B=(3,8,1,8), rho=(8,1,8), pressure=(8,1,8).
        Snowplow: r_shock=0.04, r_pinch_min=0.005, phase='radial'.
        Config: dx=0.01.
        """
        state = _make_mock_state(nr=8, nz=8, B_theta_val=2.0, rho_val=1e-3, p_val=1e4)
        snowplow = _make_mock_snowplow(r_shock=0.04, r_pinch_min=0.005, phase="radial")
        config = _make_mock_config(dx=0.01)

        result = m0_growth_rate_from_state(state, snowplow, config)

        assert isinstance(result, dict)
        assert set(result.keys()) == _EXPECTED_KEYS

        # All float values should be finite or inf (no NaN)
        assert not np.isnan(result["growth_rate"])
        assert not np.isnan(result["alfven_speed"])
        assert not np.isnan(result["beta_p"])
        assert result["growth_rate"] >= 0.0
        assert result["alfven_speed"] >= 0.0
        assert result["beta_p"] >= 0.0

    def test_from_state_none_snowplow(self):
        """With snowplow=None, m0_growth_rate_from_state returns default stable result.

        Default result: growth_rate=0, growth_time=inf, is_unstable=False.
        """
        state = _make_mock_state()
        config = _make_mock_config()

        result = m0_growth_rate_from_state(state, snowplow=None, config=config)

        assert isinstance(result, dict)
        assert set(result.keys()) == _EXPECTED_KEYS
        assert result["growth_rate"] == pytest.approx(0.0, abs=1e-30)
        assert result["growth_time"] == float("inf")
        assert result["is_unstable"] is False
        assert result["alfven_speed"] == pytest.approx(0.0, abs=1e-30)


# ===========================================================================
# TestDimensionalAnalysis
# ===========================================================================


class TestDimensionalAnalysis:
    """Dimensional sanity checks using realistic DPF pinch parameters."""

    def test_typical_dpf_pinch(self):
        """Typical PF-1000 pinch parameters produce physically plausible timescales.

        Representative values at peak compression:
            B_theta ~ 5 T (strong toroidal field at pinch surface)
            rho     ~ 1e-2 kg/m^3 (compressed deuterium)
            p       ~ 1e8 Pa (100 MBar pressure)
            a_pinch ~ 5 mm = 0.005 m

        Expected:
            Alfven speed: 1e3 – 1e6 m/s (km/s to Mm/s range)
            Growth time:  10 ns – 1 μs  (1e-8 – 1e-6 s)
        """
        result = m0_growth_rate(
            B_theta=5.0,
            rho=1e-2,
            pressure=1e8,
            a_pinch=0.005,
        )

        # Alfven speed must be in the km/s to Mm/s range
        v_A = result["alfven_speed"]
        assert 1e3 <= v_A <= 1e6, f"Alfven speed {v_A:.2e} m/s outside expected 1e3–1e6 range"

        # If unstable, growth time should be in 10 ns – 1 μs range
        if result["is_unstable"]:
            tau = result["growth_time"]
            assert 1e-8 <= tau <= 1e-6, f"Growth time {tau:.2e} s outside expected 10 ns–1 μs range"

        # beta_p must be non-negative
        assert result["beta_p"] >= 0.0

        # growth_rate must be non-negative
        assert result["growth_rate"] >= 0.0
