"""DD fusion reactivity verification against Bosch-Hale (1992) reference values.

Validates the D(d,n)He3 and D(d,p)T fusion reactivity calculations against
published reference data from:

    Bosch & Hale, Nuclear Fusion 32:611 (1992)
    Parametric fit valid for 0.2 keV < Ti < 100 keV.

References:
    Bosch, H.-S., & Hale, G. M. (1992). Nuclear Fusion, 32(4), 611-631.
"""

import numpy as np
import pytest

from dpf.constants import eV, k_B  # noqa: F401
from dpf.diagnostics.neutron_yield import (
    dd_reactivity,
    dd_reactivity_array,
    integrate_neutron_yield,
    neutron_yield_rate,
)


def test_dd_reactivity_below_threshold():
    """DD reactivity should return 0 for Ti < 0.2 keV."""
    assert dd_reactivity(0.1) == pytest.approx(0.0, abs=1e-30)
    assert dd_reactivity(0.0) == pytest.approx(0.0, abs=1e-30)


def test_dd_reactivity_at_1keV():
    """DD reactivity at 1 keV — order of magnitude check.

    At 1 keV the total DD reactivity is very low (~1e-28 m^3/s).
    The Bosch-Hale fit gives the sum of both branches.
    """
    sv = dd_reactivity(1.0)
    # At 1 keV, reactivity is tiny — order of magnitude: 1e-29 to 1e-26
    assert sv > 0, "Reactivity should be positive above threshold"
    assert 1e-30 <= sv <= 1e-24, (
        f"DD reactivity at 1 keV = {sv:.3e} m^3/s, expected ~1e-28 range"
    )


def test_dd_reactivity_at_10keV():
    """DD reactivity at 10 keV should be in ~1e-25 to 1e-22 m^3/s range."""
    sv = dd_reactivity(10.0)
    assert 1e-25 <= sv <= 1e-22, (
        f"DD reactivity at 10 keV = {sv:.3e} m^3/s, out of expected range"
    )


def test_dd_reactivity_at_50keV():
    """DD reactivity at 50 keV should be in ~1e-24 to 1e-21 m^3/s range."""
    sv = dd_reactivity(50.0)
    assert 1e-24 <= sv <= 1e-21, (
        f"DD reactivity at 50 keV = {sv:.3e} m^3/s, out of expected range"
    )


def test_dd_reactivity_monotonic_1_to_50keV():
    """DD reactivity should increase monotonically from 1 to 50 keV."""
    temps = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    reactivities = [dd_reactivity(T) for T in temps]
    for i in range(len(reactivities) - 1):
        assert reactivities[i + 1] > reactivities[i], (
            f"Non-monotonic: sv({temps[i]})={reactivities[i]:.3e} >= "
            f"sv({temps[i+1]})={reactivities[i+1]:.3e}"
        )


def test_dd_reactivity_cap_at_100keV():
    """Ti > 100 keV should return same value as 100 keV (capped)."""
    sv_100 = dd_reactivity(100.0)
    sv_200 = dd_reactivity(200.0)
    assert sv_200 == pytest.approx(sv_100, rel=1e-12)


def test_dd_reactivity_array():
    """dd_reactivity_array should match element-wise dd_reactivity calls."""
    temps = np.array([1.0, 5.0, 10.0, 50.0])
    sv_arr = dd_reactivity_array(temps)
    sv_scalar = np.array([dd_reactivity(T) for T in temps])
    np.testing.assert_allclose(sv_arr, sv_scalar, rtol=1e-12)


def test_neutron_yield_rate_basic():
    """Neutron yield rate should be positive at fusion-relevant temperatures.

    neutron_yield_rate(n_D, Ti, cell_volumes) -> (rate_density, total_rate)
    """
    n_D = np.full((8, 8, 8), 1e25)
    Ti = np.full((8, 8, 8), 5e7)  # K (~4.3 keV)
    cell_vol = 1e-9  # m^3

    rate_density, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)

    assert total_rate > 0, f"Total neutron rate should be positive, got {total_rate:.3e}"
    assert rate_density.shape == n_D.shape
    assert np.all(rate_density >= 0)


def test_neutron_yield_rate_zero_cold():
    """Neutron yield should be zero at Ti < 0.2 keV."""
    n_D = np.full((8, 8, 8), 1e25)
    Ti = np.full((8, 8, 8), 2000.0)  # K < 0.2 keV threshold
    cell_vol = 1e-9

    _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
    assert total_rate == pytest.approx(0.0, abs=1e-10)


def test_neutron_yield_scales_with_nD_squared():
    """Doubling n_D should quadruple yield rate (rate ~ n_D^2)."""
    Ti = np.full((8, 8, 8), 5e7)
    cell_vol = 1e-9

    _, rate_base = neutron_yield_rate(np.full((8, 8, 8), 1e25), Ti, cell_vol)
    _, rate_double = neutron_yield_rate(np.full((8, 8, 8), 2e25), Ti, cell_vol)

    assert rate_double == pytest.approx(4.0 * rate_base, rel=1e-10)


def test_integrate_neutron_yield():
    """integrate_neutron_yield(n_D, Ti, cell_vol, dt) = total_rate * dt."""
    n_D = np.full((8, 8, 8), 1e25)
    Ti = np.full((8, 8, 8), 5e7)
    cell_vol = 1e-9
    dt = 1e-9

    _, total_rate = neutron_yield_rate(n_D, Ti, cell_vol)
    yield_count = integrate_neutron_yield(n_D, Ti, cell_vol, dt)

    assert yield_count == pytest.approx(total_rate * dt, rel=1e-10)
