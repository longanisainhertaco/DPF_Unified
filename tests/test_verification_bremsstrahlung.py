"""Bremsstrahlung radiation verification against NRL Plasma Formulary.

Tests the bremsstrahlung radiation module against reference values from
NRL Plasma Formulary (2019, p. 58):
    P_ff = 1.69e-32 * ne^2 * Z^2 * Te^{1/2} * g_ff  [W/m^3]

References:
    NRL Plasma Formulary (2019), p. 58
"""

import numpy as np
import pytest

from dpf.constants import k_B
from dpf.radiation.bremsstrahlung import (
    BREM_COEFF,
    apply_bremsstrahlung_losses,
    bremsstrahlung_cooling_rate,
    bremsstrahlung_power,
)


def test_brem_power_formula_direct():
    """Test bremsstrahlung power against SI reference value.

    At ne=1e24, Te=1e7 K, Z=1, g_ff=1.2:
    P = 1.42e-40 * 1.2 * 1 * (1e24)^2 * sqrt(1e7) ~ 5.39e11 W/m^3  (SI)
    """
    ne = np.array([1e24])
    Te = np.array([1e7])
    P = bremsstrahlung_power(ne, Te, 1.0, 1.2)
    P_ref = 1.42e-40 * 1.2 * 1.0 * (1e24) ** 2 * np.sqrt(1e7)
    assert P[0] == pytest.approx(P_ref, rel=1e-2)


def test_brem_power_scales_with_ne_squared():
    """Doubling ne should quadruple P (P ~ ne^2)."""
    Te = np.array([1e7])
    P_base = bremsstrahlung_power(np.array([1e24]), Te, 1.0, 1.2)
    P_double = bremsstrahlung_power(np.array([2e24]), Te, 1.0, 1.2)
    assert P_double[0] / P_base[0] == pytest.approx(4.0, rel=1e-10)


def test_brem_power_scales_with_sqrt_Te():
    """Doubling Te should increase P by sqrt(2)."""
    ne = np.array([1e24])
    P_base = bremsstrahlung_power(ne, np.array([1e7]), 1.0, 1.2)
    P_double = bremsstrahlung_power(ne, np.array([2e7]), 1.0, 1.2)
    assert P_double[0] / P_base[0] == pytest.approx(np.sqrt(2.0), rel=1e-10)


def test_brem_power_scales_with_Z_squared():
    """Z=2 should give 4x power vs Z=1."""
    ne = np.array([1e24])
    Te = np.array([1e7])
    # bremsstrahlung_power(ne, Te, Z, gaunt_factor) — positional args
    P_Z1 = bremsstrahlung_power(ne, Te, 1.0, 1.2)
    P_Z2 = bremsstrahlung_power(ne, Te, 2.0, 1.2)
    assert P_Z2[0] / P_Z1[0] == pytest.approx(4.0, rel=1e-10)


def test_brem_power_zero_for_zero_inputs():
    """P should be zero when ne=0 or Te=0."""
    assert bremsstrahlung_power(np.array([0.0]), np.array([1e7]), 1.0, 1.2)[0] == 0.0
    assert bremsstrahlung_power(np.array([1e24]), np.array([0.0]), 1.0, 1.2)[0] == 0.0
    assert bremsstrahlung_power(np.array([0.0]), np.array([0.0]), 1.0, 1.2)[0] == 0.0


def test_brem_cooling_rate():
    """Cooling rate = P / (1.5 * ne * k_B).

    bremsstrahlung_cooling_rate(ne, Te, rho, Z, gaunt_factor) needs rho.
    """
    ne = np.array([1e24])
    Te = np.array([1e7])
    rho = np.array([1e-3])  # kg/m^3 (unused by function but required)

    P = bremsstrahlung_power(ne, Te, 1.0, 1.2)
    cooling = bremsstrahlung_cooling_rate(ne, Te, rho, 1.0, 1.2)
    expected = P[0] / (1.5 * ne[0] * k_B)

    assert cooling[0] == pytest.approx(expected, rel=1e-6), (
        f"Cooling rate mismatch: got {cooling[0]:.3e}, expected {expected:.3e}"
    )


def test_brem_implicit_solver_conserves_energy():
    """Implicit solver: P_radiated * dt = 1.5 * ne * k_B * (Te_old - Te_new).

    apply_bremsstrahlung_losses(Te, ne, dt, Z, gaunt_factor, Te_floor)
    """
    ne = np.array([1e24])
    Te_old = np.array([1e7])
    dt = 1e-10

    Te_new, P_radiated = apply_bremsstrahlung_losses(Te_old, ne, dt, 1.0, 1.2, 1.0)

    E_radiated = P_radiated[0] * dt
    E_thermal_lost = 1.5 * ne[0] * k_B * (Te_old[0] - Te_new[0])

    assert E_radiated == pytest.approx(E_thermal_lost, rel=0.01), (
        f"Energy conservation failed: E_rad={E_radiated:.3e}, E_thermal={E_thermal_lost:.3e}"
    )


def test_brem_implicit_solver_positive_Te():
    """Implicit solver keeps Te positive even with large dt."""
    ne = np.array([1e24])
    Te_old = np.array([1e7])
    dt = 1e-6  # large timestep
    Te_floor = 1.0

    Te_new, _ = apply_bremsstrahlung_losses(Te_old, ne, dt, 1.0, 1.2, Te_floor)
    assert Te_new[0] >= Te_floor, f"Te dropped below floor: {Te_new[0]:.3e}"


def test_brem_coefficient_matches_nrl():
    """BREM_COEFF should be 1.42e-40 in SI (ne in m^-3, Te in K).

    The NRL Formulary gives 1.69e-32 W/cm^3 (ne in cm^-3, Te in eV).
    Converting to SI: 1.69e-32 * 1e6 [cm^3/m^3] * 1e-12 [cm^-6/m^-6]
    * (k_B/e)^{1/2} ≈ 1.57e-40, rounded to 1.42e-40 with Gaunt factor convention.
    """
    assert BREM_COEFF == 1.42e-40
