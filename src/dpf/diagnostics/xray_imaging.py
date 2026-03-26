"""Synthetic X-ray pinhole imaging diagnostic.

Computes line-of-sight integrated X-ray emission from MHD state,
mimicking experimental gated X-ray cameras and filtered pinhole images.

The emission is computed as:
    I(y, z) = integral along x of epsilon(n_e, T_e, Z) dx

where epsilon is the volumetric emissivity from Bremsstrahlung + line
radiation, filtered by the detector response function.

For axisymmetric geometry, the line-of-sight integration reduces to an
Abel transform (same as interferometry.py).

References:
    Shan et al., Rev. Sci. Instrum. 75:3497 (2004) — DPF X-ray diagnostics.
    GORGON: synthetic gated X-ray images for z-pinch validation.
"""

from __future__ import annotations

import numpy as np


def bremsstrahlung_emissivity(
    ne: np.ndarray,
    Te_eV: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    """Bremsstrahlung volumetric emissivity [W/m^3/sr].

    epsilon_ff = 1.42e-40 * Z_eff * ne^2 * sqrt(Te_eV)  [W/m^3]
    Per steradian: divide by 4*pi (isotropic).

    Args:
        ne: Electron density [m^-3].
        Te_eV: Electron temperature [eV].
        Z_eff: Effective charge state.

    Returns:
        Emissivity [W/m^3/sr].
    """
    return 1.42e-40 * Z_eff * ne**2 * np.sqrt(np.maximum(Te_eV, 0.01)) / (4.0 * np.pi)


def filtered_emissivity(
    ne: np.ndarray,
    Te_eV: np.ndarray,
    Z_eff: float = 1.0,
    E_min_keV: float = 0.1,
    E_max_keV: float = 10.0,
) -> np.ndarray:
    """Bremsstrahlung emissivity filtered by photon energy window.

    Approximates a Be-filtered X-ray detector response by applying
    an exponential energy cutoff:
        I_filtered = I_total * exp(-E_min/Te) * (1 - exp(-E_max/Te))

    Args:
        ne: Electron density [m^-3].
        Te_eV: Electron temperature [eV].
        Z_eff: Effective charge state.
        E_min_keV: Low-energy cutoff [keV] (Be filter absorption).
        E_max_keV: High-energy cutoff [keV] (detector sensitivity).

    Returns:
        Filtered emissivity [W/m^3/sr].
    """
    eps = bremsstrahlung_emissivity(ne, Te_eV, Z_eff)
    Te_keV = Te_eV / 1000.0
    Te_keV_safe = np.maximum(Te_keV, 0.001)
    # Spectral filter: fraction of emission in [E_min, E_max]
    f_low = np.exp(-E_min_keV / Te_keV_safe)
    f_high = np.exp(-E_max_keV / Te_keV_safe)
    return eps * (f_low - f_high)


def synthetic_xray_image(
    ne_2d: np.ndarray,
    Te_2d: np.ndarray,
    r_cell: np.ndarray,
    dr: float,
    Z_eff: float = 1.0,
    E_min_keV: float = 0.1,
    E_max_keV: float = 10.0,
) -> np.ndarray:
    """Generate synthetic X-ray pinhole image from axisymmetric MHD state.

    For axisymmetric (r,z) data, computes line-of-sight integrated
    emission along chords perpendicular to the z-axis (Abel transform).

    The image axes are:
        y (impact parameter) — perpendicular to both r and z
        z (axial position) — along the anode

    Args:
        ne_2d: Electron density [m^-3], shape (nr, nz).
        Te_2d: Electron temperature [eV], shape (nr, nz).
        r_cell: Cell-center radii [m], shape (nr,).
        dr: Radial cell spacing [m].
        Z_eff: Effective charge.
        E_min_keV: Filter low cutoff [keV].
        E_max_keV: Filter high cutoff [keV].

    Returns:
        X-ray image [W/m^2/sr], shape (nr, nz). Each row is a chord
        at impact parameter y = r_cell[i].
    """
    nr, nz = ne_2d.shape
    eps = filtered_emissivity(ne_2d, Te_2d, Z_eff, E_min_keV, E_max_keV)

    # Abel transform: integrate emissivity along chords
    # For impact parameter y, the chord length through a shell at r is:
    #   dl = 2 * r * dr / sqrt(r^2 - y^2)  for r > y
    image = np.zeros((nr, nz))
    for iy in range(nr):
        y = r_cell[iy]
        for ir in range(iy, nr):
            r = r_cell[ir]
            if r <= y:
                continue
            dl = 2.0 * r * dr / np.sqrt(max(r**2 - y**2, 1e-30))
            image[iy, :] += eps[ir, :] * dl

    return image


def synthetic_bdot_probe(
    B_field: np.ndarray,
    probe_r: float,
    probe_z: float,
    r_cell: np.ndarray,
    z_cell: np.ndarray,
) -> dict[str, float]:
    """Synthetic magnetic probe (B-dot) signal at a specific location.

    Interpolates B-field components to the probe position using
    bilinear interpolation on the (r,z) grid.

    Args:
        B_field: Magnetic field [T], shape (3, nr, nz).
        probe_r: Probe radial position [m].
        probe_z: Probe axial position [m].
        r_cell: Cell-center radii [m], shape (nr,).
        z_cell: Cell-center z positions [m], shape (nz,).

    Returns:
        Dict with Br, Bz, Bt at probe location.
    """
    from scipy.interpolate import RegularGridInterpolator

    result = {}
    for comp, name in enumerate(["Br", "Bz", "Bt"]):
        interp = RegularGridInterpolator(
            (r_cell, z_cell), B_field[comp],
            method="linear", bounds_error=False, fill_value=0.0,
        )
        result[name] = float(interp((probe_r, probe_z)))

    return result
