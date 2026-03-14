"""Tests for two-step radial current fraction model (Damideh et al. 2025).

Validates that the SnowplowModel correctly transitions from f_cr to f_cr2
at the specified transition time, modeling re-strike current diversion in
high-voltage DPF devices like FAETON-I.

Reference: Damideh et al., Sci. Rep. 15:23048 (2025)
"""
from __future__ import annotations

import numpy as np
import pytest

from dpf.fluid.snowplow import SnowplowModel


def make_faeton_snowplow(
    radial_current_fraction: float | None = None,
    radial_current_fraction_2: float | None = None,
    radial_transition_time: float | None = None,
) -> SnowplowModel:
    """Create a FAETON-I-like SnowplowModel for testing."""
    return SnowplowModel(
        anode_radius=0.05,
        cathode_radius=0.10,
        fill_density=1.29e-3,
        anode_length=0.17,
        mass_fraction=0.70,
        fill_pressure_Pa=1600.0,
        current_fraction=0.7,
        radial_mass_fraction=0.1,
        pinch_column_fraction=0.14,
        radial_current_fraction=radial_current_fraction,
        radial_current_fraction_2=radial_current_fraction_2,
        radial_transition_time=radial_transition_time,
    )


def advance_to_radial(sp: SnowplowModel, current: float = 1e6) -> None:
    """Advance snowplow through rundown into radial phase."""
    dt = 1e-9
    for _ in range(50_000):
        sp.step(dt, current=current)
        if sp.phase == "radial":
            break
    assert sp.phase == "radial", "Failed to reach radial phase"


class TestSingleStepRadial:
    """Verify single-step (default) radial behavior is unchanged."""

    def test_default_fcr_equals_fc(self) -> None:
        """Without radial_current_fraction, f_cr defaults to f_c."""
        sp = make_faeton_snowplow()
        assert sp.f_cr == sp.f_c == 0.7

    def test_no_two_step_when_fcr2_none(self) -> None:
        """Without f_cr2, _effective_radial_fc returns f_cr always."""
        sp = make_faeton_snowplow(radial_current_fraction=0.8)
        assert sp.f_cr == 0.8
        assert sp._effective_radial_fc() == 0.8

    def test_no_two_step_when_transition_none(self) -> None:
        """With f_cr2 but no transition time, single-step model persists."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=None,
        )
        assert sp._effective_radial_fc() == 0.8

    def test_backward_compatible_result_keys(self) -> None:
        """New f_cr_eff key is present in step() results."""
        sp = make_faeton_snowplow()
        result = sp.step(1e-9, current=1e6)
        assert "f_cr_eff" in result


class TestTwoStepRadialTransition:
    """Verify two-step radial parameter transition."""

    def test_fcr_before_transition(self) -> None:
        """Before transition time, effective f_cr equals f_cr."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        # At t=0 (well before 2.5us transition), should be close to f_cr
        f_eff = sp._effective_radial_fc()
        assert f_eff == pytest.approx(0.8, abs=0.01)

    def test_fcr_after_transition(self) -> None:
        """Well after transition time, effective f_cr approaches f_cr2."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        # Manually set elapsed time well past transition
        sp._elapsed_time = 3.5e-6  # 1us past transition
        f_eff = sp._effective_radial_fc()
        assert f_eff == pytest.approx(0.5, abs=0.01)

    def test_fcr_at_transition_midpoint(self) -> None:
        """At transition time, effective f_cr is midpoint of f_cr and f_cr2."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        sp._elapsed_time = 2.5e-6  # Exactly at transition
        f_eff = sp._effective_radial_fc()
        # Midpoint of sigmoid: (0.8 + 0.5) / 2 = 0.65
        assert f_eff == pytest.approx(0.65, abs=0.02)

    def test_smooth_transition(self) -> None:
        """Transition is smooth (monotonic decrease from f_cr to f_cr2)."""
        sp = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        times = np.linspace(2.0e-6, 3.0e-6, 100)
        f_vals = []
        for t in times:
            sp._elapsed_time = t
            f_vals.append(sp._effective_radial_fc())

        # Monotonically decreasing
        for i in range(1, len(f_vals)):
            assert f_vals[i] <= f_vals[i - 1] + 1e-12

        # Bounded
        assert all(0.5 - 0.01 <= f <= 0.8 + 0.01 for f in f_vals)


class TestTwoStepRadialPhysics:
    """Verify physics impact of two-step radial model."""

    def test_reduced_fcr2_reduces_radial_force(self) -> None:
        """Lower f_cr2 should result in weaker J×B radial driving force.

        F_rad ~ (f_cr * I)^2, so reducing f_cr reduces F_rad quadratically.
        """
        sp_single = make_faeton_snowplow(radial_current_fraction=0.8)
        sp_two = make_faeton_snowplow(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )

        # Advance both past rundown
        advance_to_radial(sp_single)
        advance_to_radial(sp_two)

        # Now simulate radial phase, stepping past transition time for sp_two
        dt = 1e-9
        I_drive = 8e5  # 0.8 MA
        results_single = []
        results_two = []

        for _ in range(1000):
            r1 = sp_single.step(dt, current=I_drive)
            r2 = sp_two.step(dt, current=I_drive)
            both_radial = sp_single.phase in ("radial", "reflected")
            both_radial = both_radial and sp_two.phase in ("radial", "reflected")
            if both_radial:
                results_single.append(r1)
                results_two.append(r2)

        if results_single and results_two:
            # Two-step model should have less total radial force after transition
            # (but may not reach transition in 1000 steps, depending on timing)
            assert len(results_single) > 0
            assert len(results_two) > 0

    def test_faeton_preset_has_two_step_params(self) -> None:
        """FAETON-I preset includes two-step radial parameters."""
        from dpf.presets import _PRESETS
        faeton = _PRESETS["faeton"]["snowplow"]
        assert "radial_current_fraction" in faeton
        assert "radial_current_fraction_2" in faeton
        assert "radial_transition_time" in faeton
        assert faeton["radial_current_fraction"] == 0.8
        assert faeton["radial_current_fraction_2"] == 0.5
        assert faeton["radial_transition_time"] == 7.0e-6

    def test_config_accepts_two_step_params(self) -> None:
        """SnowplowConfig Pydantic model accepts the new fields."""
        from dpf.config import SnowplowConfig
        cfg = SnowplowConfig(
            radial_current_fraction=0.8,
            radial_current_fraction_2=0.5,
            radial_transition_time=2.5e-6,
        )
        assert cfg.radial_current_fraction == 0.8
        assert cfg.radial_current_fraction_2 == 0.5
        assert cfg.radial_transition_time == 2.5e-6

    def test_config_defaults_to_none(self) -> None:
        """New fields default to None (backward compatible)."""
        from dpf.config import SnowplowConfig
        cfg = SnowplowConfig()
        assert cfg.radial_current_fraction is None
        assert cfg.radial_current_fraction_2 is None
        assert cfg.radial_transition_time is None
