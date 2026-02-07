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

    Uses trapezoidal quadrature with singularity handling at r = y.

    Args:
        ne_r: Electron density profile ne(r), shape (nr,).
        r: Radial coordinate array [m], shape (nr,).
            Should be cell-centered: r[0] > 0.

    Returns:
        Line-integrated density N_L(y), shape (nr,).
        y[j] = r[j] (impact parameters coincide with radial grid).
    """
    nr = len(r)
    N_L = np.zeros(nr)

    for j in range(nr):
        y = r[j]
        # Integrate from r = y to r = R (index j to nr-1)
        integral = 0.0
        for i in range(j, nr):
            r_i = r[i]
            r_sq_diff = r_i**2 - y**2
            if r_sq_diff < 1e-30:
                # At the singularity (r = y), use L'Hopital approximation
                # The integrand ~ ne(r) * sqrt(2*r) / sqrt(r - y) near r = y
                # For the first cell, approximate with midpoint rule
                if i + 1 < nr:
                    dr = r[i + 1] - r[i]
                    # Regularized: replace sqrt(r^2 - y^2) with sqrt(2*r*dr)
                    r_sq_diff_reg = 2.0 * r_i * dr
                    integral += ne_r[i] * r_i / np.sqrt(max(r_sq_diff_reg, 1e-30)) * dr
                continue
            # Standard trapezoidal contribution
            integrand = ne_r[i] * r_i / np.sqrt(r_sq_diff)
            if i + 1 < nr:
                dr = r[i + 1] - r[i]
            elif i > 0:
                dr = r[i] - r[i - 1]
            else:
                dr = r[0]
            integral += integrand * dr

        N_L[j] = 2.0 * integral

    return N_L


def abel_inversion(
    N_L: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Compute the inverse Abel transform to recover ne(r) from N_L(y).

    ne(r_i) = -(1/pi) * integral_{r_i}^{R} dN_L/dy / sqrt(y^2 - r_i^2) dy

    Uses numerical differentiation of N_L and trapezoidal quadrature.

    Args:
        N_L: Line-integrated density, shape (nr,).
        r: Radial coordinate array [m], shape (nr,).

    Returns:
        Recovered density profile ne(r), shape (nr,).
    """
    nr = len(r)
    ne_recovered = np.zeros(nr)

    # Numerical derivative of N_L with respect to y (= r)
    dr = np.gradient(r)
    dNL_dy = np.gradient(N_L, r)

    for i in range(nr):
        r_i = r[i]
        integral = 0.0
        for j in range(i, nr):
            y = r[j]
            y_sq_diff = y**2 - r_i**2
            if y_sq_diff < 1e-30:
                continue
            integrand = dNL_dy[j] / np.sqrt(y_sq_diff)
            integral += integrand * dr[j]

        ne_recovered[i] = -integral / pi

    # Enforce non-negative density
    ne_recovered = np.maximum(ne_recovered, 0.0)
    return ne_recovered


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
