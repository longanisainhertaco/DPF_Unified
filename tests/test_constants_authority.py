from __future__ import annotations

import scipy.constants as scipy_constants

from dpf import constants


def test_constants_are_standards_scoped_implementation_values() -> None:
    assert constants.CONSTANTS_SCOPE == "standards_scoped_implementation_constants"
    assert constants.CONSTANTS_AUTHORITY == "scipy.constants"


def test_core_constants_match_scipy_constants() -> None:
    assert constants.e == scipy_constants.e
    assert constants.epsilon_0 == scipy_constants.epsilon_0
    assert constants.mu_0 == scipy_constants.mu_0
    assert constants.c == scipy_constants.c
    assert constants.m_e == scipy_constants.m_e
    assert constants.m_p == scipy_constants.m_p
    assert constants.m_n == scipy_constants.m_n
    assert constants.k_B == scipy_constants.k
    assert constants.h == scipy_constants.h
    assert constants.hbar == scipy_constants.hbar
    assert constants.pi == scipy_constants.pi
    assert constants.eV == scipy_constants.eV


def test_deuterium_masses_are_derived_from_deuteron_mass() -> None:
    deuteron_mass = scipy_constants.physical_constants["deuteron mass"][0]

    assert constants.m_d == deuteron_mass
    assert constants.m_D2 == 2 * deuteron_mass
