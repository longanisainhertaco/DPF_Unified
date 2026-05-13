"""Lee radial current-factor audit tests."""

from __future__ import annotations

import math

import pytest

from dpf.constants import mu_0, pi
from dpf.validation.lee_model_comparison import LeeModel


def test_lee_model_radial_inductance_scales_with_fcr() -> None:
    low = LeeModel(radial_current_fraction=0.4)
    high = LeeModel(radial_current_fraction=0.8)

    args = dict(L_pinch=0.12, cathode_radius=0.032, radius=0.008)
    assert high._radial_circuit_inductance(**args) == pytest.approx(
        2.0 * low._radial_circuit_inductance(**args)
    )


def test_lee_model_radial_dlpdt_scales_with_fcr() -> None:
    low = LeeModel(radial_current_fraction=0.35)
    high = LeeModel(radial_current_fraction=0.70)

    args = dict(L_pinch=0.12, radius=0.008, vr=-1.0e5)
    assert high._radial_circuit_dLp_dt(**args) == pytest.approx(
        2.0 * low._radial_circuit_dLp_dt(**args)
    )


def test_lee_model_radial_force_uses_fcr_not_axial_fc() -> None:
    model = LeeModel(current_fraction=0.7, radial_current_fraction=0.5)
    current = 1.0e6
    L_pinch = 0.12
    radius = 0.008

    expected = (mu_0 / (4.0 * pi)) * (0.5 * current) ** 2 * L_pinch / radius
    axial_fc_value = (mu_0 / (4.0 * pi)) * (0.7 * current) ** 2 * L_pinch / radius

    assert model._radial_magnetic_force(current, L_pinch, radius) == pytest.approx(expected)
    assert model._radial_magnetic_force(current, L_pinch, radius) != pytest.approx(
        axial_fc_value
    )


def test_lee_model_frozen_inductance_preserves_fc_fcr_scaling() -> None:
    model = LeeModel(current_fraction=0.6, radial_current_fraction=0.3)
    L_per_length = 1.0e-6
    z_max = 0.16
    L_pinch = 0.10
    b = 0.032
    r = 0.008

    expected = (
        0.6 * L_per_length * z_max
        + 0.3 * (mu_0 / (2.0 * pi)) * L_pinch * math.log(b / r)
    )
    actual = model._frozen_circuit_inductance(L_per_length, z_max, L_pinch, b, r)

    assert actual == pytest.approx(expected)


def test_lee_model_device_params_apply_lee_fcr_and_restore_default() -> None:
    model = LeeModel()
    params = {
        "C": 1.0e-6,
        "V0": 1.0e3,
        "L0": 1.0e-7,
        "R0": 1.0e-2,
        "anode_radius": 0.008,
        "cathode_radius": 0.032,
        "anode_length": 0.01,
        "fill_pressure_torr": 1.0,
        "lee_fc": 0.7,
        "lee_fcr": 0.42,
    }

    result = model.run(device_params=params)

    assert result.metadata["fcr"] == pytest.approx(0.42)
    assert model.f_cr == pytest.approx(model.fc)
