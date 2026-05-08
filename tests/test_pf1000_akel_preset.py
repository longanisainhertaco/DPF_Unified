"""PF-1000 Akel preset source-scope ratchets."""

from __future__ import annotations

import pytest

from dpf.constants import k_B, m_D2
from dpf.presets import get_preset
from dpf.validation.kr_targets import pf1000_16kv_shot12581_phase_targets


def test_pf1000_akel_preset_matches_shot12581_kr_target() -> None:
    """The MLX Akel preset must not mix nominal PF-1000 and shot-12581 inputs."""
    preset = get_preset("pf1000_akel")
    target = pf1000_16kv_shot12581_phase_targets()
    context = target["shot_context"]
    lee = target["lee_fit_parameters"]

    circuit = preset["circuit"]
    snowplow = preset["snowplow"]
    fill_pressure_pa = float(context["fill_pressure_torr"]) * 133.322
    fill_density = fill_pressure_pa * m_D2 / (k_B * float(preset["T0"]))

    assert circuit["C"] == pytest.approx(float(context["capacitance_uF"]) * 1e-6)
    assert circuit["V0"] == pytest.approx(float(context["voltage_kV"]) * 1e3)
    assert circuit["L0"] == pytest.approx(float(context["static_inductance_nH"]) * 1e-9)
    assert circuit["R0"] == pytest.approx(
        float(context["short_circuit_resistance_mohm"]) * 1e-3
    )
    assert circuit["anode_radius"] == pytest.approx(
        float(context["anode_radius_cm"]) * 1e-2
    )
    assert circuit["cathode_radius"] == pytest.approx(
        float(context["cathode_radius_cm"]) * 1e-2
    )
    assert circuit.get("crowbar_enabled", False) is False
    assert "crowbar_time" not in circuit
    assert "crowbar_resistance" not in circuit
    assert "crowbar_inductance" not in circuit

    assert preset["rho0"] == pytest.approx(fill_density, rel=1e-4)
    assert snowplow["fill_pressure_Pa"] == pytest.approx(fill_pressure_pa, rel=1e-3)
    assert snowplow["anode_length"] == pytest.approx(
        float(context["anode_length_cm"]) * 1e-2
    )
    assert snowplow["mass_fraction"] == pytest.approx(float(lee["fm"]))
    assert snowplow["current_fraction"] == pytest.approx(float(lee["fc"]))
    assert snowplow["radial_mass_fraction"] == pytest.approx(float(lee["fmr"]))
    assert snowplow["radial_current_fraction"] == pytest.approx(float(lee["fcr"]))
