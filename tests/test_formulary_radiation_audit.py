"""Formula audit checks against the local NRL Plasma Formulary.

These tests cover formulas that are directly source-checkable in
``KnowledgeReference/plasma-formulary.md``:

- Eq. 30: free-free bremsstrahlung, lines 5099-5110.
- Eq. 33: free-bound recombination radiation, lines 5118-5127.
- Eq. 34: cyclotron radiation, lines 5128-5131.
- Eq. 13: radiative recombination rate, lines 4984-4997.
"""

from __future__ import annotations

import numpy as np

from dpf.atomic.ionization import radiative_recombination_rate
from dpf.fluid.ionization import coronal_radiation_power
from dpf.radiation.improved_radiation import (
    cyclotron_power,
)
from dpf.radiation.improved_radiation import (
    recombination_power as improved_recombination_power,
)
from dpf.radiation.line_radiation import recombination_power as line_recombination_power

_EV_TO_K = 11604.518


def _nrl_eq30_bremsstrahlung(ne_m3: float, Te_eV: float, Z_eff: float) -> float:
    ne_cm3 = ne_m3 * 1.0e-6
    p_w_cm3 = 1.69e-32 * ne_cm3 * np.sqrt(Te_eV) * (Z_eff * ne_cm3)
    return float(p_w_cm3 * 1.0e6)


def _nrl_eq33_recombination(ne_m3: float, Te_K: float, Z: float) -> float:
    Te_eV = Te_K / _EV_TO_K
    ne_cm3 = ne_m3 * 1.0e-6
    chi_eV = 13.6 * Z * Z
    # Single hydrogenic species: sum[Z^2 N_i (chi / Te)] = Z * ne * chi / Te.
    p_w_cm3 = 1.69e-32 * ne_cm3 * np.sqrt(Te_eV) * (Z * ne_cm3 * chi_eV / Te_eV)
    return float(p_w_cm3 * 1.0e6)


def _nrl_eq34_cyclotron(ne_m3: float, Te_K: float, B_T: float) -> float:
    B_G = abs(B_T) * 1.0e4
    ne_cm3 = ne_m3 * 1.0e-6
    Te_eV = Te_K / _EV_TO_K
    p_w_cm3 = 6.21e-28 * B_G**2 * ne_cm3 * Te_eV
    return float(p_w_cm3 * 1.0e6)


def _nrl_eq13_radiative_recombination_rate(Te_eV: float, Z: int) -> float:
    chi = 13.6 * Z * Z / Te_eV
    bracket = 0.43 + 0.5 * np.log(chi) + 0.469 * chi ** (-1.0 / 3.0)
    alpha_cgs = 5.2e-14 * Z * np.sqrt(chi) * bracket
    return float(alpha_cgs * 1.0e-6)


def test_coronal_bremsstrahlung_term_matches_nrl_eq30_at_high_temperature() -> None:
    """The coronal helper's free-free base term must use W/cm^3, not erg/cm^3/s."""
    ne = np.array([1.0e24])
    Te_eV = np.array([1.0e6])
    Z_eff = np.array([3.0])

    actual = coronal_radiation_power(Te_eV, ne, Z_eff)[0]
    expected = _nrl_eq30_bremsstrahlung(float(ne[0]), float(Te_eV[0]), float(Z_eff[0]))

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12)


def test_coronal_bremsstrahlung_charge_scaling_is_linear_for_single_species() -> None:
    ne = np.array([1.0e24])
    Te_eV = np.array([1.0e6])

    p_z2 = coronal_radiation_power(Te_eV, ne, np.array([2.0]))[0]
    p_z4 = coronal_radiation_power(Te_eV, ne, np.array([4.0]))[0]

    np.testing.assert_allclose(p_z4 / p_z2, 2.0, rtol=1.0e-12)


def test_recombination_helpers_match_nrl_eq33_hydrogenic_form() -> None:
    ne = np.array([1.0e24])
    Te_K = np.array([1.0e7])
    Z = 2.0
    expected = _nrl_eq33_recombination(float(ne[0]), float(Te_K[0]), Z)

    line_actual = line_recombination_power(ne, Te_K, Z)[0]
    improved_actual = improved_recombination_power(ne, Te_K, Z)[0]

    np.testing.assert_allclose(line_actual, expected, rtol=5.0e-4)
    np.testing.assert_allclose(improved_actual, expected, rtol=5.0e-4)


def test_cyclotron_power_matches_nrl_eq34_and_uses_field_magnitude() -> None:
    ne = np.array([1.0e24])
    Te_K = np.array([1.0e7])
    B_T = np.array([50.0])
    expected = _nrl_eq34_cyclotron(float(ne[0]), float(Te_K[0]), float(B_T[0]))

    actual_pos = cyclotron_power(ne, Te_K, B_T)[0]
    actual_neg = cyclotron_power(ne, Te_K, -B_T)[0]

    np.testing.assert_allclose(actual_pos, expected, rtol=5.0e-4)
    np.testing.assert_allclose(actual_neg, expected, rtol=5.0e-4)


def test_atomic_radiative_recombination_rate_matches_nrl_eq13() -> None:
    Te_eV = 100.0
    Z = 2

    actual = radiative_recombination_rate(Te_eV, Z)
    expected = _nrl_eq13_radiative_recombination_rate(Te_eV, Z)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12)
