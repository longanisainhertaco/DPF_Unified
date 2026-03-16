"""Filamentation diagnostic for 3D MHD output (Challenge 8).

Detects density filaments in 3D MHD state by analyzing azimuthal
structure in cylindrical coordinates.

Physics:
    Filamentation in DPF occurs during the radial implosion phase
    when the current sheath breaks up into discrete filamentary
    channels. The m-number of the dominant filaments is related to
    the balance between magnetic pressure and kinetic energy.

References:
    Schmidt et al., Phys. Rev. Lett. 109:205003 (2012) — kinetic MJ DPF
    Kaselouris et al., Phys. Plasmas 24:012704 (2017) — solid-state DPF
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FilamentResult:
    """Result of filament detection."""

    n_filaments: int  # Number of detected filaments
    dominant_m: int  # Dominant azimuthal mode number
    m_spectrum: np.ndarray  # Fourier amplitude of each m mode
    density_contrast: float  # max(rho)/min(rho) in azimuthal ring
    filament_width_mm: float  # Estimated filament width
    is_filamented: bool  # True if density contrast > threshold


def detect_filaments(
    rho: np.ndarray,
    dx: float,
    threshold: float = 1.5,
) -> FilamentResult:
    """Detect density filaments in 3D MHD output.

    Analyzes the azimuthal density structure at the midplane (z=nz/2)
    by computing the Fourier transform along circular rings.

    Args:
        rho: Density field (nx, ny, nz) [kg/m^3].
        dx: Grid spacing [m].
        threshold: Density contrast threshold for filamentation.

    Returns:
        FilamentResult with mode analysis.
    """
    nx, ny, nz = rho.shape
    mid_z = nz // 2
    rho_mid = rho[:, :, mid_z]

    # Build cylindrical coordinates centered on domain
    x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
    y = (np.arange(ny) - ny / 2.0 + 0.5) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Sample density along a ring at r ~ 70% of domain radius
    r_target = 0.7 * (nx / 2.0) * dx
    dr_ring = 2.0 * dx  # ring thickness
    ring_mask = np.abs(R - r_target) < dr_ring

    if np.sum(ring_mask) < 8:
        return FilamentResult(
            n_filaments=0, dominant_m=0, m_spectrum=np.array([]),
            density_contrast=1.0, filament_width_mm=0, is_filamented=False,
        )

    # Extract density and theta on the ring
    rho_ring = rho_mid[ring_mask]
    theta_ring = theta[ring_mask]

    # Sort by angle for FFT
    sort_idx = np.argsort(theta_ring)
    rho_sorted = rho_ring[sort_idx]

    # Density contrast
    rho_max = float(np.max(rho_sorted))
    rho_min = float(np.min(rho_sorted))
    contrast = rho_max / max(rho_min, 1e-30)

    # FFT for azimuthal mode spectrum
    n_pts = len(rho_sorted)
    fft_result = np.fft.rfft(rho_sorted - np.mean(rho_sorted))
    m_amplitudes = np.abs(fft_result) / max(n_pts, 1)

    # Dominant mode (skip m=0 DC component)
    if len(m_amplitudes) > 1:
        dominant_m = int(np.argmax(m_amplitudes[1:])) + 1
    else:
        dominant_m = 0

    # Filament count: peaks in the azimuthal density profile
    n_filaments = 0
    if len(rho_sorted) >= 4:
        mean_rho = np.mean(rho_sorted)
        above = rho_sorted > mean_rho * 1.05
        # Count transitions from below to above threshold
        for i in range(1, len(above)):
            if above[i] and not above[i - 1]:
                n_filaments += 1

    # Filament width estimate: 2*pi*r / (2*m)
    filament_width = 2 * np.pi * r_target / max(2 * dominant_m, 1)

    is_filamented = contrast > threshold and n_filaments >= 2

    return FilamentResult(
        n_filaments=n_filaments,
        dominant_m=dominant_m,
        m_spectrum=m_amplitudes,
        density_contrast=float(contrast),
        filament_width_mm=float(filament_width * 1e3),
        is_filamented=is_filamented,
    )
