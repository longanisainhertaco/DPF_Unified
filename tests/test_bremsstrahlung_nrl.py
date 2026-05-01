"""Regression test: bremsstrahlung coefficient matches NRL Plasma Formulary eq.(30).

Source of truth (verbatim from NRL):
    [KR: plasma-formulary.md L5099-5105 eq.(30)]

    P_Br = 1.69e-32 * Ne * Te^(1/2) * sum_Z [Z^2 * N(Z)]   [W/cm^3]
        Ne in cm^-3, Te in eV

For hydrogen-like quasi-neutral plasma (single species, ni = Ne/Z):
    sum_Z [Z^2 * N(Z)] = Z * Ne
    => P_Br = 1.69e-32 * Z * Ne^2 * sqrt(Te_eV)   [W/cm^3]

Test plasma: Ne=1e25 m^-3, Te=1 keV, Z=1, no Gaunt factor.
Acceptance: SI coefficient must reproduce NRL prediction within 1%.
"""

from __future__ import annotations

import numpy as np

from dpf.diagnostics.xray_imaging import bremsstrahlung_emissivity
from dpf.radiation.bremsstrahlung import BREM_COEFF, bremsstrahlung_power

# NRL CGS coefficient (eq. 30, hydrogen-like Z=1)
NRL_CGS_COEFF = 1.69e-32  # W/cm^3, Ne in cm^-3, Te in eV
EV_TO_K = 11604.518  # 1 eV in kelvin


def _nrl_truth_W_per_m3(ne_m3: float, Te_eV: float, Z: float = 1.0) -> float:
    """Return NRL eq.(30) prediction in SI for hydrogen-like plasma (no Gaunt)."""
    ne_cm3 = ne_m3 * 1e-6
    P_W_cm3 = NRL_CGS_COEFF * Z * ne_cm3**2 * np.sqrt(Te_eV)
    return P_W_cm3 * 1e6  # W/cm^3 -> W/m^3


def test_brem_coeff_K_form_matches_nrl_within_1pct() -> None:
    """K-form (radiation/bremsstrahlung.py): coefficient must satisfy NRL eq.(30)."""
    ne = 1e25  # m^-3
    Te_eV = 1000.0  # 1 keV
    Te_K = Te_eV * EV_TO_K
    Z = 1.0

    P_truth = _nrl_truth_W_per_m3(ne, Te_eV, Z)

    # Compute with module under test, no Gaunt factor (gff=1.0) so we test
    # only the prefactor against NRL's bare coefficient.
    P_test = bremsstrahlung_power(
        np.array([ne], dtype=float),
        np.array([Te_K], dtype=float),
        Z=Z,
        gaunt_factor=1.0,
    )[0]

    rel_err = abs(P_test - P_truth) / P_truth
    assert rel_err < 0.01, (
        f"K-form bremsstrahlung off NRL by {rel_err*100:.2f}% "
        f"(P_test={P_test:.4e}, P_truth={P_truth:.4e}, "
        f"BREM_COEFF={BREM_COEFF:.4e})"
    )


def test_brem_coeff_eV_form_matches_nrl_within_1pct() -> None:
    """eV-form (diagnostics/xray_imaging.py): emissivity * 4pi must match NRL eq.(30)."""
    ne = 1e25  # m^-3
    Te_eV = 1000.0  # 1 keV
    Z = 1.0

    P_truth = _nrl_truth_W_per_m3(ne, Te_eV, Z)

    # bremsstrahlung_emissivity returns per-steradian; multiply by 4pi for total.
    eps_per_sr = bremsstrahlung_emissivity(
        np.array([ne], dtype=float),
        np.array([Te_eV], dtype=float),
        Z_eff=Z,
    )[0]
    P_test = eps_per_sr * 4.0 * np.pi

    rel_err = abs(P_test - P_truth) / P_truth
    assert rel_err < 0.01, (
        f"eV-form bremsstrahlung off NRL by {rel_err*100:.2f}% "
        f"(P_test={P_test:.4e}, P_truth={P_truth:.4e})"
    )


def test_brem_coeff_value_explicit() -> None:
    """The K-form coefficient must equal 1.569e-40 within rounding."""
    expected = 1.569e-40
    rel_err = abs(BREM_COEFF - expected) / expected
    assert rel_err < 0.005, (
        f"BREM_COEFF={BREM_COEFF:.4e} differs from NRL-derived 1.569e-40"
    )


def test_brem_K_and_eV_forms_agree() -> None:
    """K-form and eV-form must agree across multiple temperatures."""
    ne = 5e24
    for Te_eV in [10.0, 100.0, 1000.0, 10000.0]:
        Te_K = Te_eV * EV_TO_K

        P_K = bremsstrahlung_power(
            np.array([ne]), np.array([Te_K]), Z=1.0, gaunt_factor=1.0,
        )[0]
        P_eV = bremsstrahlung_emissivity(
            np.array([ne]), np.array([Te_eV]), Z_eff=1.0,
        )[0] * 4.0 * np.pi

        rel_err = abs(P_K - P_eV) / max(P_K, 1e-30)
        assert rel_err < 0.01, (
            f"K-form ({P_K:.4e}) vs eV-form ({P_eV:.4e}) disagree "
            f"at Te={Te_eV} eV: {rel_err*100:.2f}%"
        )
