"""Line radiation cooling for MLX solver.

Translates the piecewise power-law cooling functions from
dpf/radiation/line_radiation.py into pure MLX mx.where chains.
All coefficients are in log-space to avoid float32 subnormal issues
(same approach as bremsstrahlung in mlx_sources.py).

Placement: src/dpf/metal/mlx_line_radiation.py
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from dpf.metal.mlx_kernels import IBR, IBT, IBZ, IDN, IEE, IEN, IMR, IMT, IMZ, ISR

# Physical constants
_KBOLTZ = 1.380649e-23   # J/K
_EV = 1.602176634e-19    # J
_KB_OVER_EV = _KBOLTZ / _EV  # ~8.617e-5 eV/K

# Numerical floors
_RHO_FLOOR = 1.0e-12
_P_FLOOR = 1.0e-12
_LOG_FLOOR = -92.0  # exp(-92) ~ 1e-40, safe minimum for log(Lambda)


def _log_cooling_copper(log_Te_eV: mx.array) -> mx.array:
    """Log-space copper cooling function (6-segment piecewise power-law).

    Simplified from the 21-point log-log table in line_radiation.py.
    Captures M-shell peak (~100 eV), Ar-like trough, and L-shell bump (~3 keV).
    Accuracy: within 2x of full table. DPF pinch temperatures (10-1000 eV) well-covered.

    Args:
        log_Te_eV: ln(Te [eV]), shape (nr, nz).

    Returns:
        ln(Lambda [W m^3]), shape (nr, nz).
    """
    # Segment boundaries in ln(eV): 0, 1.6094, 3.912, 4.6052, 6.9078, 8.5172, 9.2103
    # ln(Lambda) at boundaries: -76.7, -71.6, -68.2, -68.0, -70.0, -69.3, -70.3

    # Segment 1: 1-5 eV (rising steeply)
    s1 = -76.7 + (log_Te_eV - 0.0) * (-71.6 - (-76.7)) / (1.6094 - 0.0)
    # Segment 2: 5-50 eV (rising to M-shell)
    s2 = -71.6 + (log_Te_eV - 1.6094) * (-68.2 - (-71.6)) / (3.912 - 1.6094)
    # Segment 3: 50-100 eV (M-shell peak)
    s3 = -68.2 + (log_Te_eV - 3.912) * (-68.0 - (-68.2)) / (4.6052 - 3.912)
    # Segment 4: 100-1000 eV (declining + Ar-like trough)
    s4 = -68.0 + (log_Te_eV - 4.6052) * (-70.0 - (-68.0)) / (6.9078 - 4.6052)
    # Segment 5: 1000-5000 eV (L-shell bump)
    s5 = -70.0 + (log_Te_eV - 6.9078) * (-69.3 - (-70.0)) / (8.5172 - 6.9078)
    # Segment 6: 5000-10000 eV (declining)
    s6 = -69.3 + (log_Te_eV - 8.5172) * (-70.3 - (-69.3)) / (9.2103 - 8.5172)

    result = mx.where(log_Te_eV < 1.6094, s1,
             mx.where(log_Te_eV < 3.912, s2,
             mx.where(log_Te_eV < 4.6052, s3,
             mx.where(log_Te_eV < 6.9078, s4,
             mx.where(log_Te_eV < 8.5172, s5, s6)))))

    # Floor: below 1 eV or above 10 keV
    result = mx.where(log_Te_eV < 0.0, _LOG_FLOOR, result)
    return mx.maximum(result, _LOG_FLOOR)


def _log_cooling_hydrogen(log_Te_eV: mx.array) -> mx.array:
    """H/D cooling: Lyman-alpha peak at ~4 eV, drops above 13.6 eV (3-segment).

    Approximation of double-exponential as piecewise power-law in log-space.
    Peak Lambda ~ 3e-32 at 4 eV -> ln(3e-32) = -72.47.

    Args:
        log_Te_eV: ln(Te [eV]), shape (nr, nz).

    Returns:
        ln(Lambda [W m^3]), shape (nr, nz).
    """
    # Below 4 eV: steep rise from floor
    s_rise = -92.0 + (log_Te_eV - 0.0) * 14.0
    # 4-14 eV: decline from Lyman-alpha peak
    s_peak = -72.5 + (log_Te_eV - 1.386) * (-4.0)
    # Above 13.6 eV: residual free-free + collisional excitation
    s_ionized = -82.0 + (log_Te_eV - 2.624) * (-0.5)

    result = mx.where(log_Te_eV < 1.386, s_rise,          # ln(4) = 1.386
             mx.where(log_Te_eV < 2.624, s_peak, s_ionized))  # ln(13.6) = 2.624
    return mx.maximum(result, _LOG_FLOOR)


def _log_cooling_generic(log_Te_eV: mx.array, Z: float) -> mx.array:
    """Generic Z-scaling fallback: peak at ~10*Z^1.3 eV, amplitude ~ Z^2 * 1e-33.

    Args:
        log_Te_eV: ln(Te [eV]), shape (nr, nz).
        Z: Atomic number.

    Returns:
        ln(Lambda [W m^3]), shape (nr, nz).
    """
    import math
    log_Te_peak = math.log(10.0 * Z ** 1.3)
    log_Lambda_peak = 2.0 * math.log(Z) + math.log(1e-33)

    below = log_Lambda_peak - 4.6 + (log_Te_eV - (log_Te_peak - 2.3)) * 2.5
    at_peak = log_Lambda_peak + (log_Te_eV - log_Te_peak) * 1.0
    above = log_Lambda_peak + (log_Te_eV - log_Te_peak) * (-0.8)
    far_above = log_Lambda_peak - 2.3 + (log_Te_eV - (log_Te_peak + 2.3)) * (-1.0)

    result = mx.where(log_Te_eV < log_Te_peak - 2.3, below,
             mx.where(log_Te_eV < log_Te_peak, at_peak,
             mx.where(log_Te_eV < log_Te_peak + 2.3, above, far_above)))
    return mx.maximum(result, _LOG_FLOOR)


def apply_line_radiation_mlx(
    U: mx.array,
    dt: float,
    species_Z: list[int],
    species_Y: mx.array,
    gamma: float = 5.0 / 3.0,
    ion_mass: float = 3.34358377e-27,
) -> mx.array:
    """Remove line radiation cooling from total energy (operator-split).

    P_line_total = sum_k [ ne * n_k * Lambda_k(Te) ]

    where n_k = Y_k * ne (approximation: same mass as background ion)
    and ne = rho / ion_mass (Z=1 background).

    All arithmetic in log-space to handle the wide dynamic range of Lambda(Te).

    Args:
        U: Conserved state (NVAR, nr, nz), float32.
        dt: Timestep [s].
        species_Z: Atomic numbers for each species, e.g. [1, 29] for D+Cu.
        species_Y: Mass fractions (N_species, nr, nz) from SpeciesManager.
        gamma: Adiabatic index.
        ion_mass: Background ion mass [kg] (deuterium default).

    Returns:
        Updated U with line radiation energy sink applied, shape (NVAR, nr, nz), float32.
    """
    rho = mx.maximum(U[IDN], _RHO_FLOOR)
    inv_rho = 1.0 / rho
    v2 = (U[IMR] ** 2 + U[IMZ] ** 2 + U[IMT] ** 2) * inv_rho * inv_rho
    B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
    p = (gamma - 1.0) * mx.maximum(U[IEN] - 0.5 * rho * v2 - 0.5 * B2, _P_FLOOR)

    # Te in eV, log-space: Te = p * m_i / (2 * rho * kB), then Te_eV = Te * kB/eV
    _LOG_MI = float(np.log(ion_mass))
    _LOG_2KB = float(np.log(2.0 * _KBOLTZ))
    _LOG_KB_EV = float(np.log(_KB_OVER_EV))

    log_p = mx.log(mx.maximum(p, 1e-30))
    log_rho = mx.log(mx.maximum(rho, 1e-30))
    log_Te_K = log_p + _LOG_MI - _LOG_2KB - log_rho
    log_Te_eV = log_Te_K + _LOG_KB_EV
    log_Te_eV = mx.maximum(log_Te_eV, -2.3)  # floor at 0.1 eV in log-space

    # ne = rho / ion_mass
    log_ne = log_rho - _LOG_MI

    # Accumulate radiation from each species
    # Initialize to floor so first log-sum-exp is stable
    Q_total = mx.zeros(rho.shape, dtype=mx.float32)

    for k, Z in enumerate(species_Z):
        Y_k = species_Y[k]  # mass fraction of species k, shape (nr, nz)
        # n_k ~ Y_k * ne (approximation for Z=1 background)
        log_nk = mx.log(mx.maximum(Y_k, 1e-30)) + log_ne

        # Select cooling function by Z
        if Z <= 1:
            log_Lambda = _log_cooling_hydrogen(log_Te_eV)
        elif Z == 29:
            log_Lambda = _log_cooling_copper(log_Te_eV)
        else:
            log_Lambda = _log_cooling_generic(log_Te_eV, float(Z))

        # P_k = ne * n_k * Lambda_k  ->  log(P_k) = log(ne) + log(n_k) + log(Lambda)
        log_Pk = log_ne + log_nk + log_Lambda
        log_Pk = mx.minimum(log_Pk, 80.0)  # prevent exp overflow before accumulation

        # Accumulate: Q_total += exp(log_Pk)
        Q_total = Q_total + mx.exp(log_Pk)

    dE = Q_total * dt

    # Clamp: cannot remove more than available thermal energy above floor
    e_kin = 0.5 * rho * v2
    e_mag = 0.5 * B2
    e_thermal_floor = _P_FLOOR / (gamma - 1.0)
    e_available = mx.maximum(U[IEN] - e_kin - e_mag - e_thermal_floor, 0.0)
    dE = mx.minimum(dE, e_available)
    dE = mx.maximum(dE, 0.0)

    # Update: line radiation is an energy sink
    # Entropy tracer omitted (same convention as apply_bremsstrahlung comment)
    updated_vars = [
        U[IDN], U[IMR], U[IMZ], U[IMT],
        U[IEN] - dE,
        U[ISR],
        U[IBR], U[IBZ], U[IBT],
        U[IEE],  # IEE = 9 (use constant, not hardcoded index)
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)
