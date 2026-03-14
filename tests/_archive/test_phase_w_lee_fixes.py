"""Phase W: Lee Model Fixes & Validation Tightening.

Tests for:
1. Lee Model radial mass correction (dynamic mass sweep).
2. Lee Model adiabatic back-pressure (missing term).
3. Lee Model docstring swap fix (fm vs fc).
4. Validation timing tolerance tightening (50% -> 10%).
5. Validation uncertainty budget placeholder.
"""

from __future__ import annotations

import numpy as np

from dpf.validation.experimental import DEVICES, validate_current_waveform
from dpf.validation.lee_model_comparison import LeeModel


class TestLeeModelRadialPhysicsFixes:
    """Tests for corrections to Lee model radial phase physics."""

    def test_radial_mass_is_dynamic(self):
        """Mass should accumulate as the shock moves inward.

        Old buggy behavior: M_radial was constant (annulus mass).
        New correct behavior: M_radial(r_s) = fm * rho0 * pi * (b^2 - r^2) * L.

        We can't inspect M_radial directly as it's internal, but we can verify
        that dynamics change when f_m is changed. If math was fixed, f_m should
        strongly affect the implosion time.
        """
        # Run with low mass fraction
        model_low = LeeModel(mass_fraction=0.1)
        res_low = model_low.run("PF-1000")

        # Run with high mass fraction
        model_high = LeeModel(mass_fraction=0.9)
        res_high = model_high.run("PF-1000")

        # Higher mass -> slower implosion -> later pinch time
        # The difference should be significant if the radial mass term is correct
        diff_time = res_high.pinch_time - res_low.pinch_time

        assert diff_time > 0, "Higher mass fraction should lead to slower implosion"
        # Ideally check for significant difference (e.g. > 100 ns)
        assert diff_time > 1e-7, (
            f"Time difference {diff_time:.2e} s too small for 0.1->0.9 fm change. "
            "Suggests M_radial might still be constant/incorrect."
        )

    def test_adiabatic_back_pressure_opposes_compression(self):
        """Back-pressure should slow down the compression.

        We can't easily turn it off without code modification, but we can
        compare runs with different initial pressures.
        Higher pressure -> Higher back-pressure -> Slower/stalled compression
        (though also higher mass, so we need to disentangle).

        Better approach: Use a very high pressure where back-pressure dominates.
        """
        # Nominal pressure
        model_nom = LeeModel()
        # Custom params: same as PF-1000 but very high pressure
        params = self._get_default_params()
        params["fill_pressure_torr"] = 100.0  # 30x nominal

        res_high_p = model_nom.run(device_params=params)

        # It should still pinch, but much slower
        assert res_high_p.pinch_time > 0

        # Check if it reached small radius. If back-pressure is working,
        # it might bounce or stagnate at a larger radius than without it.
        min_r = np.min(res_high_p.r_shock)

        # With adiabatic back-pressure, r_min is limited by pressure balance.
        # Without it, it would crash to 0 or numerical floor.
        assert min_r > 0.001, (
            f"Minimum radius {min_r:.4e} m too small for high pressure. "
            "Suggests missing back-pressure term."
        )

    def _get_default_params(self):
        return {
            "C": 1.332e-3, "V0": 27e3, "L0": 33.5e-9, "R0": 2.3e-3,
            "anode_radius": 0.0575, "cathode_radius": 0.08,
            "anode_length": 0.16, "fill_pressure_torr": 3.5,
            "peak_current_exp": 1.87e6, "current_rise_time_exp": 5.8e-6,
        }


class TestValidationRefinements:
    """Tests for stricter validation logic."""

    def test_timing_tolerance_tightened(self):
        """Validation should reject 20% timing error (was 50% allowed)."""
        # Mock device with rise time known
        device = DEVICES["NX2"]
        t_target = device.current_rise_time
        t_bad = t_target * 1.25  # 25% error (simulate slower rise)

        t_arr = np.linspace(0, 3*t_target, 200)
        # Create sine wave peaking at t_bad
        I_arr = device.peak_current * np.sin(np.pi * t_arr / (2 * t_bad)) * np.exp(-t_arr/t_bad)

        result = validate_current_waveform(t_arr, I_arr, "NX2")

        # Should be False now (would have been True with 0.5 tolerance)
        assert not result["timing_ok"], (
            f"Timing error {abs(result['peak_time_sim']-t_target)/t_target:.1%} accepted, "
            "but should be rejected (threshold 10%)."
        )

    def test_uncertainty_field_present(self):
        """Validation result should include uncertainty budget."""
        t_arr = np.linspace(0, 1e-5, 100)
        I_arr = np.zeros_like(t_arr)

        # We need a valid call that doesn't crash on metrics
        # Use a dummy waveform
        t_arr = np.linspace(0, 1e-5, 100)
        I_arr = np.sin(t_arr * 1e6) * 1e6

        result = validate_current_waveform(t_arr, I_arr, "PF-1000")
        assert "uncertainty" in result, "Missing 'uncertainty' field"
        unc = result["uncertainty"]
        assert "peak_current_exp_1sigma" in unc
        assert "peak_current_combined_1sigma" in unc
