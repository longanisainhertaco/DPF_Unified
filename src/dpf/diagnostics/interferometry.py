"""Synthetic interferometry diagnostics for DPF simulations.

Implements Abel transform for axisymmetric plasma density profiles to
compute line-integrated electron density (as measured by interferometers).

For an axisymmetric plasma with electron density ne(r), the line-integrated
density along a chord at impact parameter y is:

    N_L(y) = 2 * integral_{y}^{R} ne(r) * r / sqrt(r^2 - y^2) dr

This is the Abel transform of ne(r).

The corresponding phase shift measured by a Mach-Zehnder interferometer:

    delta_phi(y) = (e^2 / (2 * me * c * epsilon_0 * omega^2)) * N_L(y)
                 = (r_e * lambda) * N_L(y)

where r_e = e^2 / (4*pi*epsilon_0*me*c^2) is the classical electron radius
and lambda is the probe laser wavelength.

Reference:
    Bracewell, "The Fourier Transform and Its Applications" (2000)
    Hutchinson, "Principles of Plasma Diagnostics" (2002)
"""

from __future__ import annotations

import numpy as np

from dpf.constants import c, e, epsilon_0, m_e, pi

# Classical electron radius [m]
r_e = e**2 / (4.0 * pi * epsilon_0 * m_e * c**2)


def abel_transform(
    ne_r: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Compute the Abel transform of an axisymmetric density profile.

    N_L(y_j) = 2 * integral_{y_j}^{R} ne(r) * r / sqrt(r^2 - y_j^2) dr

    Uses vectorized trapezoidal quadrature with singularity handling at r = y.

    Args:
        ne_r: Electron density profile ne(r), shape (nr,).
        r: Radial coordinate array [m], shape (nr,).
            Should be cell-centered: r[0] > 0.

    Returns:
        Line-integrated density N_L(y), shape (nr,).
        y[j] = r[j] (impact parameters coincide with radial grid).
    """
    nr = len(r)
    if nr == 0:
        return np.zeros(nr)

    # Quadrature weights: forward differences, last cell uses backward
    dr = np.empty(nr)
    dr[:-1] = np.diff(r)
    dr[-1] = dr[-2] if nr > 1 else r[0]

    # Build kernel matrix K[j, i] via broadcasting
    ri = r[np.newaxis, :]  # (1, nr) — integration variable
    yj = r[:, np.newaxis]  # (nr, 1) — impact parameter
    r2_y2 = ri**2 - yj**2  # (nr, nr)

    # Upper triangular mask (i >= j) with non-singular points
    mask_upper = np.triu(np.ones((nr, nr), dtype=bool))
    valid = mask_upper & (r2_y2 > 1e-30)

    # Standard contribution: r_i / sqrt(r_i^2 - y_j^2) * dr_i
    K = np.where(valid, ri / np.sqrt(np.maximum(r2_y2, 1e-30)) * dr[np.newaxis, :], 0.0)

    # Diagonal regularization (i = j, j < nr-1): L'Hopital approximation
    # At singularity r = y: use sqrt(2*r*dr) instead of sqrt(r^2 - y^2)
    if nr > 1:
        diag = np.arange(nr - 1)
        r_sq_reg = np.maximum(2.0 * r[diag] * dr[diag], 1e-30)
        K[diag, diag] = r[diag] / np.sqrt(r_sq_reg) * dr[diag]

    return 2.0 * (K @ ne_r)


def abel_inversion(
    N_L: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Compute the inverse Abel transform to recover ne(r) from N_L(y).

    ne(r_i) = -(1/pi) * integral_{r_i}^{R} dN_L/dy / sqrt(y^2 - r_i^2) dy

    Uses vectorized numerical differentiation and trapezoidal quadrature.

    Args:
        N_L: Line-integrated density, shape (nr,).
        r: Radial coordinate array [m], shape (nr,).

    Returns:
        Recovered density profile ne(r), shape (nr,).
    """
    nr = len(r)
    if nr == 0:
        return np.zeros(nr)

    # Numerical derivative of N_L with respect to y (= r)
    dr = np.gradient(r)
    dNL_dy = np.gradient(N_L, r)

    # Build kernel K[i, j] via broadcasting
    yj = r[np.newaxis, :]  # (1, nr) — y values (columns)
    ri = r[:, np.newaxis]  # (nr, 1) — r values (rows)
    y2_r2 = yj**2 - ri**2  # (nr, nr)

    # Upper triangular mask (j >= i) with non-singular points
    mask_upper = np.triu(np.ones((nr, nr), dtype=bool))
    valid = mask_upper & (y2_r2 > 1e-30)

    K = np.where(valid, dr[np.newaxis, :] / np.sqrt(np.maximum(y2_r2, 1e-30)), 0.0)

    ne_recovered = -(K @ dNL_dy) / pi

    # Enforce non-negative density
    return np.maximum(ne_recovered, 0.0)


def phase_shift(
    N_L: np.ndarray,
    wavelength: float = 632.8e-9,
) -> np.ndarray:
    """Compute interferometric phase shift from line-integrated density.

    delta_phi = r_e * lambda * N_L

    Args:
        N_L: Line-integrated density [m^-2], shape (nr,).
        wavelength: Probe laser wavelength [m]. Default: 632.8 nm (HeNe).

    Returns:
        Phase shift [radians], shape (nr,).
    """
    return r_e * wavelength * N_L


def fringe_shift(
    N_L: np.ndarray,
    wavelength: float = 632.8e-9,
) -> np.ndarray:
    """Compute interferometric fringe shift from line-integrated density.

    N_fringes = delta_phi / (2*pi) = r_e * lambda * N_L / (2*pi)

    Args:
        N_L: Line-integrated density [m^-2], shape (nr,).
        wavelength: Probe laser wavelength [m]. Default: 632.8 nm (HeNe).

    Returns:
        Number of fringe shifts, shape (nr,).
    """
    return phase_shift(N_L, wavelength) / (2.0 * pi)
