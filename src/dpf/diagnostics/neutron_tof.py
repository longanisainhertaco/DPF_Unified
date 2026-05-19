"""Synthetic neutron time-of-flight (nToF) spectrum diagnostic.

Computes the energy-angle distribution of D-D fusion neutrons from
beam-target and thermonuclear production channels. The spectrum
distinguishes the two channels by their energy distributions:

- Thermonuclear: isotropic, E_n ~ 2.45 MeV +/- thermal broadening
- Beam-target: anisotropic, E_n shifted by beam kinetic energy

This is a synthetic generator only. The thermonuclear Doppler-broadening
width and the beam-target kinematic shift are inferred laws with no
KnowledgeReference source (see ``THERMONUCLEAR_DOPPLER_WIDTH_STATUS`` and
WP-N6 §4). It applies no detector response and no scatter handling. It
cannot support neutron authority: the neutron-spectrum channel in
``dpf.first_principles.neutron_authority`` stays ``missing_or_blocked``.

Source-backed constant only:
    KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3802-3814 —
    D-D reaction (1b) 2.45 MeV neutron birth energy.
"""

from __future__ import annotations

import numpy as np

# D-D fusion Q-value
_Q_DD = 3.269e6 * 1.602e-19  # 3.269 MeV in Joules
_M_N = 1.6749e-27  # neutron mass [kg]
_M_D = 3.3436e-27  # deuteron mass [kg]
_M_HE3 = 5.0082e-27  # He-3 mass [kg]
# Center-of-mass neutron birth energy [eV]. Source-backed:
# KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3802-3814 reaction
# (1b) D + D -> He3(0.82 MeV) + n(2.45 MeV).
_E_N_CENTER = 2.45e6

# S3.6 / WP-N6 §4 uncited-coefficient isolation.
# The thermonuclear Doppler-broadening width used below,
# sigma = 82.5*sqrt(Ti_keV) keV (FWHM 177*sqrt(Ti_keV) keV, attributed in
# legacy comments to "Brysk 1973"), has NO KnowledgeReference source. No KR
# file in this corpus carries the 82.5 / 177 coefficient or a Brysk 1973
# extract. The KR-cited alternative is the shifted-Maxwell TOF form
# (KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-
# by-the-plasma-focus-vessel-527cc533.md:366-378, eq. 4). The coefficient is
# therefore labelled inferred_candidate and isolated from neutron authority:
# the neutron-spectrum channel in dpf.first_principles.neutron_authority stays
# missing_or_blocked until doppler_width_law_ref is a reviewed KR source. This
# constant is the explicit flag; 82.5 is NOT a source-backed default.
THERMONUCLEAR_DOPPLER_WIDTH_STATUS = {
    "coefficient": "thermonuclear_doppler_width_82p5_sqrt_Ti",
    "law": "sigma_keV = 82.5*sqrt(Ti_keV) (FWHM 177*sqrt(Ti_keV) keV)",
    "legacy_attribution": "brysk_1973_not_in_knowledgereference",
    "status": "inferred_candidate",
    "kr_source": "none",
    "kr_cited_alternative": (
        "KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-"
        "caused-by-the-plasma-focus-vessel-527cc533.md:366-378 (shifted-"
        "Maxwell TOF form, eq. 4)"
    ),
    "isolation": (
        "neutron-spectrum authority stays blocked until the Doppler-width law "
        "has a reviewed KR citation (WP-N6 §4); not a source-backed default"
    ),
    "can_support_first_principles_acceptance": False,
}


def thermonuclear_spectrum(
    n_neutrons: int = 10000,
    Ti_eV: float = 1000.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate thermonuclear D-D neutron energy spectrum.

    Isotropic emission with Doppler broadening from ion thermal motion. The
    width sigma = 82.5*sqrt(Ti_keV) keV (FWHM 177*sqrt(Ti_keV) keV) is an
    INFERRED law with no KnowledgeReference source — see
    ``THERMONUCLEAR_DOPPLER_WIDTH_STATUS``. Synthetic diagnostic only; cannot
    support neutron authority (WP-N6 §4).

    Args:
        n_neutrons: Number of neutrons to sample.
        Ti_eV: Ion temperature [eV].
        seed: Random seed.

    Returns:
        Neutron energies [eV], shape (n_neutrons,).
    """
    rng = np.random.default_rng(seed)
    Ti_keV = Ti_eV / 1000.0
    # Doppler width sigma = 82.5*sqrt(Ti_keV) keV — inferred_candidate, no KR
    # source (THERMONUCLEAR_DOPPLER_WIDTH_STATUS, WP-N6 §4).
    sigma_eV = 82500.0 * np.sqrt(max(Ti_keV, 0.001))
    return rng.normal(_E_N_CENTER, sigma_eV, size=n_neutrons)


def beam_target_spectrum(
    n_neutrons: int = 10000,
    E_beam_eV: float = 50000.0,
    theta_det: float = 0.0,
    seed: int = 43,
) -> np.ndarray:
    """Generate beam-target D-D neutron energy spectrum.

    Anisotropic: neutron energy depends on beam direction relative
    to detector. Forward neutrons get energy boost, backward get
    energy reduction.

    E_n(theta) ~ E_n_cm + 2 * sqrt(E_beam * E_n_cm * m_n / m_total) * cos(theta)

    Args:
        n_neutrons: Number of neutrons.
        E_beam_eV: Beam deuteron kinetic energy [eV].
        theta_det: Detector angle from beam axis [rad].
        seed: Random seed.

    Returns:
        Neutron energies [eV], shape (n_neutrons,).
    """
    rng = np.random.default_rng(seed)
    E_cm = _E_N_CENTER
    m_total = _M_N + _M_HE3
    # Kinematic shift
    shift = 2.0 * np.sqrt(
        E_beam_eV * E_cm * _M_N / m_total
    ) * np.cos(theta_det)
    # Spread from beam energy distribution (assume 10% spread)
    sigma = 0.1 * abs(shift) + 50000.0  # minimum 50 keV spread
    return rng.normal(E_cm + shift, sigma, size=n_neutrons)


def combined_tof_spectrum(
    Y_thermo: float,
    Y_bt: float,
    Ti_eV: float = 1000.0,
    E_beam_eV: float = 50000.0,
    theta_det: float = 0.0,
    n_samples: int = 10000,
    E_bins: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combined thermonuclear + beam-target nToF spectrum.

    Args:
        Y_thermo: Thermonuclear yield (number).
        Y_bt: Beam-target yield (number).
        Ti_eV: Ion temperature [eV].
        E_beam_eV: Beam energy [eV].
        theta_det: Detector angle [rad].
        n_samples: Total Monte Carlo samples.
        E_bins: Energy bins [eV]. Default: 2.0-3.0 MeV in 100 bins.

    Returns:
        (E_centers, counts): Energy bin centers [eV] and normalized counts.
    """
    Y_total = max(Y_thermo + Y_bt, 1.0)
    f_thermo = Y_thermo / Y_total
    n_thermo = int(n_samples * f_thermo)
    n_bt = n_samples - n_thermo

    energies = []
    if n_thermo > 0:
        energies.append(thermonuclear_spectrum(n_thermo, Ti_eV))
    if n_bt > 0:
        energies.append(beam_target_spectrum(n_bt, E_beam_eV, theta_det))

    if not energies:
        if E_bins is None:
            E_bins = np.linspace(2.0e6, 3.0e6, 101)
        centers = 0.5 * (E_bins[:-1] + E_bins[1:])
        return centers, np.zeros(len(centers))

    all_E = np.concatenate(energies)

    if E_bins is None:
        E_bins = np.linspace(2.0e6, 3.0e6, 101)

    counts, _ = np.histogram(all_E, bins=E_bins, density=True)
    centers = 0.5 * (E_bins[:-1] + E_bins[1:])
    return centers, counts
