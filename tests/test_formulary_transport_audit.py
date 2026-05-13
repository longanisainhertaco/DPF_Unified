"""Formula audit checks for NRL/Braginskii transport coefficients."""

from __future__ import annotations

import numpy as np

from dpf.collision.spitzer import braginskii_kappa, coulomb_log, nu_ei, spitzer_resistivity
from dpf.constants import e, k_B, m_e, mu_0
from dpf.diagnostics.plasma_regime import magnetic_reynolds_number
from dpf.diagnostics.regime_classifier import classify_regime
from dpf.fluid.anisotropic_conduction import braginskii_kappa_perp
from dpf.validation.pinch_physics import coulomb_mean_free_path


def test_braginskii_kappa_perp_preserves_unmagnetized_limit() -> None:
    ne = np.array([1.0e22])
    Te = np.array([1.0e6])
    B = np.array([0.0])

    kpar, kperp = braginskii_kappa(ne, Te, B, Z=1.0)

    np.testing.assert_allclose(kperp, kpar, rtol=1.0e-12)


def test_braginskii_kappa_perp_matches_nrl_high_field_limit() -> None:
    """NRL formulary gives kappa_e_perp = 4.7 n kT tau / (m omega_c^2 tau^2)."""
    ne = np.array([1.0e22])
    Te = np.array([1.0e6])
    B = np.array([50.0])

    _, kperp = braginskii_kappa(ne, Te, B, Z=1.0)
    lnL = coulomb_log(ne, Te)
    freq = nu_ei(ne, Te, lnL, Z=1.0)
    omega_ce = e * B / m_e
    expected = 4.7 * ne * k_B**2 * Te * freq / (m_e * omega_ce**2)

    np.testing.assert_allclose(kperp, expected, rtol=1.0e-6)


def test_anisotropic_conduction_perp_uses_nrl_high_field_coefficient() -> None:
    ne = np.array([1.0e22])
    Te = np.array([1.0e6])
    B = np.array([50.0])

    kperp = braginskii_kappa_perp(ne, Te, B, Z_eff=1.0)
    lnL = coulomb_log(ne, Te)
    Te_safe = np.maximum(Te, 1.0)
    ne_safe = np.maximum(ne, 1.0e-10)
    tau_e = (
        3.0
        * np.sqrt(2.0 * np.pi)
        * (8.8541878128e-12) ** 2
        * np.sqrt(m_e)
        * (k_B * Te_safe) ** 1.5
        / np.maximum(ne_safe * e**4 * np.maximum(lnL, 2.0), 1.0e-300)
    )
    omega_ce = e * B / m_e
    expected = 4.7 * ne * k_B**2 * Te / (m_e * omega_ce**2 * tau_e)

    np.testing.assert_allclose(kperp, expected, rtol=1.0e-6)


def test_magnetic_reynolds_uses_corrected_nrl_spitzer_resistivity() -> None:
    ne = np.array([1.0e22])
    Te = np.array([1.0e6])
    velocity = np.array([2.0e5])
    length = 1.0e-2
    lnL = 10.0

    actual = magnetic_reynolds_number(ne, Te, velocity, length, lnL=lnL)
    eta = spitzer_resistivity(ne, Te, lnL=lnL, Z=1.0)
    expected = mu_0 * velocity * length / eta

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12)


def test_regime_classifier_uses_nrl_high_temperature_coulomb_log_branch() -> None:
    ne = 1.0e23
    Te_eV = 100.0
    velocity = 2.0e5
    length = 1.0e-2
    Z = 1.0

    result = classify_regime(ne, Te_eV, B_T=10.0, L_m=length, v_m_s=velocity, Z=Z)
    ne_cm3 = ne * 1.0e-6
    lnL = 24.0 - np.log(np.sqrt(ne_cm3) * Te_eV ** -1.0)
    eta = 5.2e-5 * Z * lnL / Te_eV**1.5
    expected_rm = mu_0 * velocity * length / eta

    np.testing.assert_allclose(result.magnetic_reynolds, expected_rm, rtol=1.0e-9)


def test_pinch_physics_mfp_uses_nrl_electron_ion_coulomb_log_branch() -> None:
    ne = 1.0e23
    Te_eV = 100.0
    Z = 1.0
    ne_cm3 = ne * 1.0e-6
    lnL_ei_high = 24.0 - np.log(np.sqrt(ne_cm3) * Te_eV ** -1.0)

    actual = coulomb_mean_free_path(Te_eV, ne, Z=Z)
    expected = coulomb_mean_free_path(Te_eV, ne, Z=Z, ln_lambda=lnL_ei_high)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12)
