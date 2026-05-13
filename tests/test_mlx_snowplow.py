"""Unit tests for the scalar MLX snowplow model.

These tests do not import MLX. They cover the reduced Lee/RADPF snowplow
used by the pure-MLX discharge driver.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


def _load_mlx_snowplow_module():
    """Load the pure-Python module without importing dpf.metal/__init__.py."""
    module_path = Path(__file__).resolve().parents[1] / "src/dpf/metal/mlx_snowplow.py"
    spec = importlib.util.spec_from_file_location("_mlx_snowplow_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MLX_SNOWPLOW_MODULE = _load_mlx_snowplow_module()
MLXSnowplow = MLX_SNOWPLOW_MODULE.MLXSnowplow


def _make_radial_snowplow(
    *,
    anode_radius: float = 0.10,
    cathode_radius: float = 0.20,
    anode_length: float = 1.0,
    pinch_column_fraction: float = 0.05,
    current_fraction: float = 0.7,
    radial_current_fraction: float | None = None,
) -> MLXSnowplow:
    sp = MLXSnowplow(
        anode_radius=anode_radius,
        cathode_radius=cathode_radius,
        fill_density=1.0e-4,
        anode_length=anode_length,
        mass_fraction=0.2,
        current_fraction=current_fraction,
        radial_mass_fraction=0.1,
        radial_current_fraction=radial_current_fraction,
        pinch_column_fraction=pinch_column_fraction,
    )
    sp._phase = "radial"
    sp._z = anode_length
    sp._r_s = anode_radius
    sp._r_p = anode_radius
    sp._z_f = 0.9 * sp._z_pinch_limit
    sp._L_axial = sp._axial_inductance(anode_length)
    sp._L_plasma = sp._L_axial + sp._radial_inductance(sp._r_p, sp._z_f)
    return sp


def test_pinch_column_fraction_caps_radial_column_length() -> None:
    """The pcf input must cap the reduced radial focus length."""
    sp = _make_radial_snowplow(pinch_column_fraction=0.05)
    cap = sp._z_pinch_limit

    result = sp.step(dt=2.0e-8, current=1.0e6, voltage=2.0e4, R0=0.0, L0=20e-9)

    assert math.isclose(cap, 0.05, rel_tol=0.0, abs_tol=1.0e-12)
    assert result["z_focus"] <= cap
    assert result["z_pinch_limit"] == cap


def test_deuterium_radial_stop_uses_gross_rmin_not_axis() -> None:
    """Reduced MLX snowplow should stop at KR r_min=0.13a, not 0.01a."""
    a = 0.10
    sp = _make_radial_snowplow(anode_radius=a, cathode_radius=0.20)
    sp._r_s = 0.014
    sp._r_p = 0.014
    sp._L_plasma = sp._L_axial + sp._radial_inductance(sp._r_p, sp._z_f)

    result = sp.step(dt=2.0e-8, current=1.0e6, voltage=2.0e4, R0=0.0, L0=20e-9)

    expected_rmin = 0.13 * a
    assert result["phase"] == "pinch"
    assert sp.is_active is False
    assert math.isclose(result["r_min"], expected_rmin)
    assert math.isclose(result["r_piston"], expected_rmin)
    assert math.isclose(result["r_shock"], expected_rmin)
    assert result["r_shock"] > 0.01 * a


def test_reduced_mlx_radius_convention_is_scope_separated() -> None:
    a = 0.10
    sp = _make_radial_snowplow(anode_radius=a, cathode_radius=0.20)
    convention = sp.radius_convention

    assert convention == MLX_SNOWPLOW_MODULE.mlx_snowplow_radius_convention()
    assert convention["radial_inductance_radius"] == "r_p"
    assert convention["radial_inductance_radius_meaning"] == "piston_radius"
    assert convention["r_min_over_a"] == 0.13
    assert convention["full_lee_five_phase_coverage"] is False
    assert convention["cross_backend_equivalent_to_cpu"] is False
    assert convention["validation_status"] == "not_validation_evidence"


def test_invalid_geometry_rejected() -> None:
    with pytest.raises(ValueError, match="cathode_radius"):
        MLXSnowplow(
            anode_radius=0.02,
            cathode_radius=0.02,
            fill_density=1.0e-4,
            anode_length=0.16,
        )


def test_radial_current_fraction_defaults_to_current_fraction() -> None:
    sp = _make_radial_snowplow(current_fraction=0.72)

    result = sp.step(dt=2.0e-8, current=1.0e6, voltage=2.0e4, R0=0.0, L0=20e-9)

    assert math.isclose(sp._fcr, 0.72)
    assert result["f_cr_eff"] == 0.72


def test_radial_current_fraction_is_separate_from_axial_fraction() -> None:
    sp = _make_radial_snowplow(
        current_fraction=0.70,
        radial_current_fraction=0.75,
    )

    result = sp.step(dt=2.0e-8, current=1.0e6, voltage=2.0e4, R0=0.0, L0=20e-9)

    assert math.isclose(sp._fc, 0.70)
    assert math.isclose(sp._fcr, 0.75)
    assert result["f_cr_eff"] == 0.75
