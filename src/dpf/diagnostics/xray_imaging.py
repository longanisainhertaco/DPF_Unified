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

from dpf.radiation.bremsstrahlung import BREM_COEFF

_EV_TO_K = 11604.518


def bremsstrahlung_emissivity(
    ne: np.ndarray,
    Te_eV: np.ndarray,
    Z_eff: float = 1.0,
) -> np.ndarray:
    """Bremsstrahlung volumetric emissivity [W/m^3/sr].

    NRL eq.(30) SI K-form:
        P_total = BREM_COEFF * Z * ne^2 * sqrt(Te_K)   [W/m^3]
    Per steradian (assume isotropic): divide by 4*pi.

    [KR: plasma-formulary.md L5101 eq.(30)]

    Args:
        ne: Electron density [m^-3].
        Te_eV: Electron temperature [eV].
        Z_eff: Effective charge state.

    Returns:
        Emissivity [W/m^3/sr].
    """
    Te_K = np.maximum(Te_eV * _EV_TO_K, 1.0)
    return BREM_COEFF * Z_eff * ne**2 * np.sqrt(Te_K) / (4.0 * np.pi)


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


def radiating_pinch_geometry_from_image(
    image: np.ndarray,
    y_cell: np.ndarray,
    z_cell: np.ndarray,
    *,
    threshold_fraction: float = 0.5,
) -> dict[str, object]:
    """Estimate radiating pinch geometry from a synthetic image.

    The geometry is a density proxy when the image emissivity is dominated by
    bremsstrahlung, because the local emission scales with electron density
    squared. It should be compared with matching gated-image diagnostics, not
    treated as a direct density measurement.
    """
    image = np.asarray(image, dtype=float)
    y_cell = np.asarray(y_cell, dtype=float)
    z_cell = np.asarray(z_cell, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if image.shape != (len(y_cell), len(z_cell)):
        raise ValueError("image shape must match y_cell and z_cell lengths")
    if not (0.0 < threshold_fraction <= 1.0):
        raise ValueError("threshold_fraction must be in (0, 1]")

    finite = np.where(np.isfinite(image), image, 0.0)
    peak = float(np.max(finite)) if finite.size else 0.0
    if peak <= 0.0:
        return {
            "has_radiating_region": False,
            "threshold_fraction": threshold_fraction,
            "peak_intensity": peak,
            "diagnostic_role": "density_proxy_bremsstrahlung_spatial_geometry",
        }

    mask = finite >= threshold_fraction * peak
    if not np.any(mask):
        return {
            "has_radiating_region": False,
            "threshold_fraction": threshold_fraction,
            "peak_intensity": peak,
            "diagnostic_role": "density_proxy_bremsstrahlung_spatial_geometry",
        }

    y_idx, z_idx = np.where(mask)
    y_extent = float(np.max(np.abs(y_cell[y_idx])))
    z_min = float(np.min(z_cell[z_idx]))
    z_max = float(np.max(z_cell[z_idx]))
    weights = finite[mask]
    z_centroid = float(np.average(z_cell[z_idx], weights=weights))

    peak_idx = np.unravel_index(int(np.argmax(finite)), finite.shape)
    return {
        "has_radiating_region": True,
        "threshold_fraction": threshold_fraction,
        "peak_intensity": peak,
        "diameter_mm": 2.0 * y_extent * 1.0e3,
        "length_cm": (z_max - z_min) * 100.0,
        "z_min_cm": z_min * 100.0,
        "z_max_cm": z_max * 100.0,
        "z_centroid_cm": z_centroid * 100.0,
        "peak_y_mm": float(y_cell[peak_idx[0]] * 1.0e3),
        "peak_z_cm": float(z_cell[peak_idx[1]] * 100.0),
        "diagnostic_role": "density_proxy_bremsstrahlung_spatial_geometry",
        "validity_notes": {
            "density_proxy": (
                "Bremsstrahlung-dominated image geometry tracks dense plasma "
                "emission but is not a calibrated electron-density field."
            ),
        },
    }


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
