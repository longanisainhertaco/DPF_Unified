"""Tests for Pease-Braginskii current diagnostic.

Verifies the I_PB calculation against Haines (2011) and checks
the radiative collapse regime detection for PF-1000 conditions.

References:
    Haines (2011) Plasma Phys. Control. Fusion 53, 093001
    Pease (1957) Proc. Phys. Soc. 70, 11
"""

import math

import numpy as np
import pytest

from dpf.diagnostics.pease_braginskii import (
    check_pease_braginskii,
    pease_braginskii_current,
)


class TestPeaseBraginskiiCurrent:
    """Test the Pease-Braginskii current calculation."""

    def test_returns_expected_keys(self):
        """Result dict has all expected fields."""
        result = pease_braginskii_current()
        assert "I_PB" in result
        assert "I_PB_MA" in result
        assert "Z" in result
        assert "gaunt_factor" in result
        assert "ln_Lambda" in result

    def test_default_deuterium_in_ma_range(self):
        """I_PB for deuterium (Z=1) is in the 0.5-2.0 MA range."""
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        assert 0.5 < result["I_PB_MA"] < 2.0, (
            f"I_PB = {result['I_PB_MA']:.3f} MA outside expected [0.5, 2.0] MA"
        )

    def test_haines_formula_z1_g1_ln10(self):
        """Haines (2011): I_PB = 0.433 * sqrt(6/2) = 0.750 MA for Z=1, g=1, lnL=10."""
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        expected = 0.433 * math.sqrt(3.0)  # sqrt(6/2) = sqrt(3)
        assert result["I_PB_MA"] == pytest.approx(expected, rel=1e-3)

    def test_haines_formula_z1_g12_ln10(self):
        """I_PB ~ 0.685 MA for Z=1, g_ff=1.2, lnL=10."""
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        expected = 0.433 * math.sqrt(6.0 / 2.4)  # sqrt(6/(2*1.2))
        assert result["I_PB_MA"] == pytest.approx(expected, rel=1e-3)

    def test_gaunt_factor_1_gives_higher_ipb(self):
        """g_ff=1 (less radiation) -> higher I_PB than g_ff=1.2."""
        pb_g1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        pb_g12 = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        assert pb_g1["I_PB"] > pb_g12["I_PB"]

    def test_higher_z_lowers_ipb(self):
        """Higher Z -> more radiation -> lower I_PB."""
        pb_z1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        pb_z2 = pease_braginskii_current(Z=2.0, gaunt_factor=1.2, ln_Lambda=10.0)
        assert pb_z2["I_PB"] < pb_z1["I_PB"]

    def test_higher_ln_lambda_raises_ipb(self):
        """Higher ln_Lambda -> more Ohmic heating -> higher I_PB."""
        pb_low = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=5.0)
        pb_high = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=20.0)
        assert pb_high["I_PB"] > pb_low["I_PB"]

    def test_ipb_scales_as_sqrt_ln_lambda(self):
        """I_PB is proportional to sqrt(ln_Lambda)."""
        pb_10 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        pb_40 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=40.0)
        ratio = pb_40["I_PB"] / pb_10["I_PB"]
        expected = math.sqrt(40.0 / 10.0)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_ipb_scales_as_inv_sqrt_gaunt(self):
        """I_PB is proportional to 1/sqrt(g_ff)."""
        pb_1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.0, ln_Lambda=10.0)
        pb_4 = pease_braginskii_current(Z=1.0, gaunt_factor=4.0, ln_Lambda=10.0)
        ratio = pb_4["I_PB"] / pb_1["I_PB"]
        expected = math.sqrt(1.0 / 4.0)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_ma_conversion(self):
        """I_PB_MA = I_PB * 1e-6."""
        result = pease_braginskii_current()
        assert result["I_PB_MA"] == pytest.approx(result["I_PB"] * 1e-6, rel=1e-12)

    def test_positive_result(self):
        """I_PB is always positive."""
        result = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=2.0)
        assert result["I_PB"] > 0

    def test_z2_value(self):
        """I_PB for Z=2 (He-like) should be lower than Z=1."""
        pb_z2 = pease_braginskii_current(Z=2.0, gaunt_factor=1.2, ln_Lambda=10.0)
        # Z*(1+Z) = 6 for Z=2 vs 2 for Z=1, so I_PB(Z=2)/I_PB(Z=1) = sqrt(2/6)
        pb_z1 = pease_braginskii_current(Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0)
        ratio = pb_z2["I_PB"] / pb_z1["I_PB"]
        assert ratio == pytest.approx(math.sqrt(2.0 / 6.0), rel=1e-10)


class TestCheckPeaseBraginskii:
    """Test the regime detection function."""

    def test_pf1000_exceeds_pb(self):
        """PF-1000 at 1.87 MA exceeds I_PB (radiative collapse regime).

        Even with generous ln_Lambda=20, I_PB ~ 0.97 MA < 1.87 MA.
        """
        result = check_pease_braginskii(
            I_current=1.87e6, Z=1.0, gaunt_factor=1.2, ln_Lambda=20.0
        )
        assert result["exceeds_PB"], (
            f"PF-1000 at 1.87 MA should exceed I_PB = {result['I_PB_MA']:.3f} MA"
        )
        assert result["regime"] == "radiative_collapse"
        assert result["ratio"] > 1.0

    def test_small_device_below_pb(self):
        """A small DPF at 100 kA should be below I_PB (stable regime)."""
        result = check_pease_braginskii(
            I_current=100e3, Z=1.0, gaunt_factor=1.2, ln_Lambda=10.0
        )
        assert not result["exceeds_PB"]
        assert result["regime"] == "stable"
        assert result["ratio"] < 1.0

    def test_returns_both_currents(self):
        """Result includes both I_current and I_PB."""
        result = check_pease_braginskii(I_current=1e6)
        assert "I_current" in result
        assert "I_current_MA" in result
        assert "I_PB" in result
        assert "I_PB_MA" in result
        assert "ratio" in result
        assert "exceeds_PB" in result
        assert "regime" in result

    def test_ratio_is_correct(self):
        """ratio = I_current / I_PB."""
        result = check_pease_braginskii(I_current=2e6)
        assert result["ratio"] == pytest.approx(
            result["I_current"] / result["I_PB"], rel=1e-10
        )

    def test_negative_current_uses_abs(self):
        """Negative current should use absolute value."""
        result_pos = check_pease_braginskii(I_current=1.87e6)
        result_neg = check_pease_braginskii(I_current=-1.87e6)
        assert result_pos["ratio"] == pytest.approx(result_neg["ratio"], rel=1e-10)

    def test_zero_current_is_stable(self):
        """Zero current is always in stable regime."""
        result = check_pease_braginskii(I_current=0.0)
        assert not result["exceeds_PB"]
        assert result["regime"] == "stable"
        assert result["ratio"] == 0.0


class TestDerivedDiagnostics:
    """Test fast_magnetosonic_speed and bennett_radius from derived.py."""

    def test_fast_magnetosonic_pure_hydro(self):
        """With B=0, fast magnetosonic speed = sound speed."""
        from dpf.diagnostics.derived import fast_magnetosonic_speed

        B = np.zeros((3, 4, 4, 4))
        rho = np.ones((4, 4, 4)) * 1e-3
        p = np.ones((4, 4, 4)) * 1e5
        gamma = 5.0 / 3.0

        c_f = fast_magnetosonic_speed(B, p, rho, gamma)
        c_s = np.sqrt(gamma * p / rho)
        np.testing.assert_allclose(c_f, c_s, rtol=1e-10)

    def test_fast_magnetosonic_pure_magnetic(self):
        """With p=0, fast magnetosonic speed = Alfven speed."""
        from dpf.diagnostics.derived import alfven_speed, fast_magnetosonic_speed

        B = np.zeros((3, 4, 4, 4))
        B[2] = 1.0  # Bz = 1 T
        rho = np.ones((4, 4, 4)) * 1e-3
        p = np.zeros((4, 4, 4))

        c_f = fast_magnetosonic_speed(B, p, rho, gamma=5.0 / 3.0)
        v_A = alfven_speed(B, rho)
        np.testing.assert_allclose(c_f, v_A, rtol=1e-10)

    def test_fast_magnetosonic_positive(self):
        """Fast magnetosonic speed is always positive."""
        from dpf.diagnostics.derived import fast_magnetosonic_speed

        B = np.random.randn(3, 4, 4, 4) * 0.1
        rho = np.ones((4, 4, 4)) * 1e-3
        p = np.ones((4, 4, 4)) * 1e5
        assert np.all(fast_magnetosonic_speed(B, p, rho) > 0)

    def test_bennett_radius_scales_with_current(self):
        """Bennett radius ~ I (doubles when current doubles)."""
        from dpf.diagnostics.derived import bennett_radius

        a1 = bennett_radius(current=1e6, Te=1e7, ne=1e24)
        a2 = bennett_radius(current=2e6, Te=1e7, ne=1e24)
        assert a2 / a1 == pytest.approx(2.0, rel=1e-10)

    def test_bennett_radius_pf1000(self):
        """Bennett radius for PF-1000 conditions is mm-scale.

        PF-1000: I ~ 1.87 MA, Te ~ 1 keV = 1.16e7 K, ne ~ 1e25 m^-3.
        a_B = I * sqrt(mu_0 / (8*pi^2*ne*kB*2*Te))
        """
        from dpf.diagnostics.derived import bennett_radius

        a_B = bennett_radius(current=1.87e6, Te=1.16e7, ne=1e25)
        # Bennett radius should be ~1-10 mm for PF-1000 at high compression
        assert 1e-4 < a_B < 0.05, f"Bennett radius {a_B:.4e} m outside [0.1mm, 50mm]"

    def test_bennett_radius_positive(self):
        """Bennett radius is always positive."""
        from dpf.diagnostics.derived import bennett_radius

        assert bennett_radius(current=1e6, Te=1e7, ne=1e24) > 0
        assert bennett_radius(current=0, Te=1e7, ne=1e24) >= 0


class TestEngineIntegration:
    """Test that PB diagnostic is wired into the engine."""

    def test_engine_has_pb_result_after_step(self, small_config):
        """Engine stores _last_pb_result after a step."""
        from dpf.engine import SimulationEngine

        small_config.diagnostics.hdf5_filename = ":memory:"
        engine = SimulationEngine(small_config)
        engine.step()
        assert hasattr(engine, "_last_pb_result")
        pb = engine._last_pb_result
        assert "I_PB_MA" in pb
        assert "ratio" in pb
        assert "regime" in pb
        assert pb["I_PB_MA"] > 0
