"""Audit tests for SnowplowModel post-pinch empirical closures."""

from __future__ import annotations

import pytest

import numpy as np

from dpf.constants import k_B, m_D2
from dpf.fluid.snowplow import SnowplowModel


def _make_snowplow() -> SnowplowModel:
    return SnowplowModel(
        anode_radius=0.0575,
        cathode_radius=0.080,
        fill_density=1.0e-4,
        anode_length=0.16,
        mass_fraction=0.3,
        fill_pressure_Pa=400.0,
        current_fraction=0.7,
        pinch_column_fraction=0.14,
    )


def test_reflected_density_ratio_is_kr_rankine_hugoniot_value() -> None:
    sp = _make_snowplow()

    assert sp._reflected_shock_density_ratio == pytest.approx(4.0)


def test_post_pinch_expansion_multiplier_is_not_hidden_factor_three() -> None:
    sp = _make_snowplow()

    assert sp._post_pinch_expansion_velocity_multiplier == pytest.approx(1.0)


def test_post_pinch_result_reports_empirical_resistance_components() -> None:
    sp = _make_snowplow()
    sp._pinch_complete = True
    sp.phase = "pinch"
    sp.r_shock = sp.r_pinch_min
    sp._r_pinch_at_stagnation = sp.r_pinch_min
    sp._pinch_time = 0.0
    sp._elapsed_time = 20.0e-9
    sp._tau_m0 = 50.0e-9
    sp._v_expand = 1.0e5
    sp._I_at_pinch = 1.0e6

    result = sp.step(1.0e-9, 1.0e6)

    assert result["post_pinch_empirical_resistance"] == 1.0
    assert result["post_pinch_resistance_multiplier"] == pytest.approx(2.0)
    assert result["post_pinch_expansion_velocity_multiplier"] == pytest.approx(1.0)
    assert result["R_spitzer"] >= 0.0
    assert result["R_anom"] >= 0.0
    assert result["R_plasma"] == pytest.approx(result["R_spitzer"] + result["R_anom"])


def test_radial_profile_cold_fill_pressure_uses_d2_molecular_mass() -> None:
    sp = _make_snowplow()
    sp.vr = 0.0
    r_grid = np.linspace(sp.a, sp.b, 8)

    profiles = sp.export_radial_profiles(r_grid, current=1.0e6)

    expected_pressure = sp.rho0 * k_B * 300.0 / m_D2
    assert np.all(profiles["pressure"] == pytest.approx(expected_pressure))
