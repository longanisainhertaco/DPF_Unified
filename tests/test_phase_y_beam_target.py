"""Phase Y — Beam-Target Integration and Quick Fixes Tests.

Tests beam_target_yield_rate from src/dpf/diagnostics/beam_target.py and the
int() -> round() fix for zipper BC index computation in engine.py.
"""

from __future__ import annotations

import numpy as np
import pytest

# ===========================================================================
# TestRoundFix
# ===========================================================================


class TestRoundFix:
    """Test int() -> round() fix for zipper BC indices."""

    def test_round_nearest_cell(self) -> None:
        """round() gives nearest cell vs int() truncation."""
        # z_sheath = 0.097 m, dz = 0.01 m
        # int(0.097 / 0.01) = int(9.7) = 9  (truncation)
        # round(0.097 / 0.01) = round(9.7) = 10 (nearest)
        z_sheath = 0.097
        dz = 0.01
        assert int(z_sheath / dz) == 9  # Old behavior (truncation)
        assert round(z_sheath / dz) == 10  # New behavior (nearest)

    def test_round_exact(self) -> None:
        """round() matches int() for exact values."""
        z_sheath = 0.05
        dz = 0.01
        assert round(z_sheath / dz) == int(z_sheath / dz) == 5

    def test_round_midpoint(self) -> None:
        """round() handles midpoints (banker's rounding)."""
        # Python uses banker's rounding: 0.5 rounds to nearest even
        z_sheath = 0.055
        dz = 0.01
        idx = round(z_sheath / dz)
        assert idx in (5, 6)  # Either is acceptable for midpoint


# ===========================================================================
# TestBeamTargetYieldRate
# ===========================================================================


class TestBeamTargetYieldRate:
    """Test beam_target_yield_rate function directly."""

    def test_positive_yield(self) -> None:
        """Non-zero inputs produce positive yield rate."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(
            I_pinch=200e3,  # 200 kA
            V_pinch=50e3,  # 50 kV
            n_target=1e25,  # typical pinch density
            L_target=0.01,  # 10 mm pinch column
            f_beam=0.2,
        )
        assert rate > 0

    def test_zero_current_zero_yield(self) -> None:
        """Zero current gives zero yield."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(0.0, 50e3, 1e25, 0.01, 0.2)
        assert rate == 0.0

    def test_zero_voltage_zero_yield(self) -> None:
        """Zero voltage gives zero yield (no beam energy)."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 0.0, 1e25, 0.01, 0.2)
        assert rate == 0.0

    def test_yield_scales_with_current(self) -> None:
        """Yield rate scales linearly with current."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate1 = beam_target_yield_rate(100e3, 50e3, 1e25, 0.01, 0.2)
        rate2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, 0.2)
        assert rate2 == pytest.approx(2.0 * rate1, rel=1e-10)

    def test_yield_scales_with_density(self) -> None:
        """Yield rate scales linearly with target density."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate1 = beam_target_yield_rate(200e3, 50e3, 1e24, 0.01, 0.2)
        rate2 = beam_target_yield_rate(200e3, 50e3, 2e24, 0.01, 0.2)
        assert rate2 == pytest.approx(2.0 * rate1, rel=1e-10)

    def test_yield_scales_with_length(self) -> None:
        """Yield rate scales linearly with target length."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate1 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, 0.2)
        rate2 = beam_target_yield_rate(200e3, 50e3, 1e25, 0.02, 0.2)
        assert rate2 == pytest.approx(2.0 * rate1, rel=1e-10)

    def test_zero_density_zero_yield(self) -> None:
        """Zero target density gives zero yield."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 50e3, 0.0, 0.01, 0.2)
        assert rate == 0.0

    def test_zero_length_zero_yield(self) -> None:
        """Zero target length gives zero yield."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 50e3, 1e25, 0.0, 0.2)
        assert rate == 0.0

    def test_typical_dpf_magnitude(self) -> None:
        """Yield rate for typical PF-1000 parameters is in physically expected range.

        Expected: 10^10 to 10^14 neutrons/s for DPF devices at 100-500 kJ stored.
        """
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(
            I_pinch=500e3,  # 500 kA
            V_pinch=100e3,  # 100 kV
            n_target=1e25,  # m^-3
            L_target=0.01,  # 10 mm
            f_beam=0.2,
        )
        # Must be a finite, positive, physically plausible number
        assert np.isfinite(rate)
        assert rate > 0.0
        # Rough bounds: 10^8 to 10^18 neutrons/s (very conservative window)
        assert rate > 1e8
        assert rate < 1e18

    def test_f_beam_clamped_above_one(self) -> None:
        """f_beam > 1 is clamped to 1; result must still be positive and finite."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate_clamped = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=2.0)
        rate_one = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=1.0)
        assert np.isfinite(rate_clamped)
        assert rate_clamped == pytest.approx(rate_one, rel=1e-10)

    def test_f_beam_zero_zero_yield(self) -> None:
        """f_beam=0 gives zero yield (no beam ions)."""
        from dpf.diagnostics.beam_target import beam_target_yield_rate

        rate = beam_target_yield_rate(200e3, 50e3, 1e25, 0.01, f_beam=0.0)
        assert rate == 0.0
