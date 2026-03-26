"""Synthetic neutron time-of-flight (nToF) spectrum diagnostic.

Computes the energy-angle distribution of D-D fusion neutrons from
beam-target and thermonuclear production channels. The spectrum
distinguishes the two channels by their energy distributions:

- Thermonuclear: isotropic, E_n ~ 2.45 MeV ± thermal broadening
- Beam-target: anisotropic, E_n shifted by beam kinetic energy

References:
    Bernstein et al., Phys. Rev. 94:1515 (1954) — D-D kinematics.
    Jager & Herold, NIM A 271:495 (1988) — DPF nToF diagnostics.
"""

from __future__ import annotations

import numpy as np

# D-D fusion Q-value
_Q_DD = 3.269e6 * 1.602e-19  # 3.269 MeV in Joules
_M_N = 1.6749e-27  # neutron mass [kg]
_M_D = 3.3436e-27  # deuteron mass [kg]
_M_HE3 = 5.0082e-27  # He-3 mass [kg]
_E_N_CENTER = 2.45e6  # Center-of-mass neutron energy [eV]


def thermonuclear_spectrum(
    n_neutrons: int = 10000,
    Ti_eV: float = 1000.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate thermonuclear D-D neutron energy spectrum.

    Isotropic emission with Doppler broadening from ion thermal motion.
    FWHM ~ 177 * sqrt(Ti_keV) keV (Brysk 1973).

    Args:
        n_neutrons: Number of neutrons to sample.
        Ti_eV: Ion temperature [eV].
        seed: Random seed.

    Returns:
        Neutron energies [eV], shape (n_neutrons,).
    """
    rng = np.random.default_rng(seed)
    Ti_keV = Ti_eV / 1000.0
    # Doppler broadening: sigma = 82.5 * sqrt(Ti_keV) keV
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
