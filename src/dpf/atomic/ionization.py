"""Ionization models: Saha equilibrium and collisional-radiative (CR).

Provides two levels of ionization modelling:

1. **Saha equilibrium** (fast, LTE valid at high density):
   For hydrogen-like species (H, D, T), the average ionization state Z_bar
   is 0 (neutral) or 1 (fully ionized).  Valid when collisional rates
   dominate radiative rates: n_e >> 10^{20} m^{-3} for typical DPF.

2. **Collisional-radiative model** (non-LTE, for impurities and low density):
   Solves rate equations dn_Z/dt = ionization - recombination per charge
   state using Lotz ionization rates and radiative+dielectronic recombination.
   Returns spatially-resolved Z_bar(r,z) for multi-species tracking.

References:
    NRL Plasma Formulary (2019)
    Griem, "Principles of Plasma Spectroscopy" (1997)
    Lotz, Z. Phys. 206:205 (1967)  — empirical ionization rate
    Burgess, ApJ 139:776 (1964)    — dielectronic recombination
    Post et al., At. Data Nucl. Data Tables 20:397 (1977)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import eV, h, k_B, m_e, pi

# ---------------------------------------------------------------------------
# Physical constants used locally
# ---------------------------------------------------------------------------
E_ION_H = 13.6 * eV  # Hydrogen ionization energy [J]

# Ionization potentials I_Z (eV) for each charge state Z → Z+1.
# Index = charge state *before* ionization.
# Hydrogen / Deuterium: only Z=0→1
_IP_H = np.array([13.6])

# Copper (Z=29): ground-state ionization potentials from NIST ASD.
# Full 29 values: Cu⁰→Cu¹ … Cu²⁸→Cu²⁹.
_IP_CU = np.array([
    7.726, 20.292, 36.841, 57.38, 79.8,
    103.0, 139.0, 166.0, 199.0, 232.0,
    265.3, 369.0, 401.0, 435.0, 484.0,
    520.0, 557.0, 633.0, 671.0, 1690.0,
    1800.0, 1918.0, 2044.0, 2179.0, 2307.0,
    2479.0, 2587.0, 11062.0, 11567.8,
])

# Tungsten (Z=74): first 10 ionization potentials (enough for DPF
# conditions where T_e < 10 keV and W rarely exceeds W^{10+}).
_IP_W = np.array([
    7.864, 16.1, 24.0, 35.0, 48.0,
    61.0, 75.0, 91.0, 108.0, 126.0,
])

# Map Z_nuclear → ionization-potential array
IONIZATION_POTENTIALS: dict[int, np.ndarray] = {
    1: _IP_H,
    29: _IP_CU,
    74: _IP_W,
}

# Lotz fitting parameter a_Z (cm² eV²) — universal value for most ions.
# Lotz, Z. Phys. 206:205 (1967), Table I.
_LOTZ_A = 4.5e-14  # cm² eV²  (converted to SI below)
_LOTZ_A_SI = _LOTZ_A * 1e-4  # m² eV²


@njit(cache=True)
def _saha_parameter(Te: float, ne: float) -> float:
    """Compute the Saha parameter S for hydrogen.

    S = (1/ne) * (2*pi*me*kB*T/h^2)^{3/2} * 2 * exp(-E_ion/(kB*T))

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].

    Returns:
        Saha parameter S (dimensionless).
    """
    if Te < 1.0 or ne < 1.0:
        return 0.0

    kT = k_B * Te
    # Thermal de Broglie: (2*pi*me*kB*T/h^2)^{3/2}
    thermal_factor = (2.0 * pi * m_e * kT / (h * h)) ** 1.5

    # Exponential: exp(-E_ion / kT)
    # Guard against overflow for very low temperatures
    exponent = -E_ION_H / kT
    if exponent < -500.0:
        return 0.0

    exp_factor = np.exp(exponent)

    # Statistical weight ratio g_ion/g_neutral = 2/1 for hydrogen
    g_ratio = 2.0

    # S = (thermal_factor * g_ratio * exp_factor) / ne
    S = thermal_factor * g_ratio * exp_factor / ne

    return S


@njit(cache=True)
def saha_ionization_fraction(Te: float, ne: float) -> float:
    """Compute the average ionization state Z_bar for hydrogen.

    Uses the Saha equation:
        Z_bar = S / (1 + S)

    where S is the Saha parameter.

    For cold gas (Te << 1 eV): Z_bar ~ 0 (neutral)
    For hot plasma (Te >> 1 eV): Z_bar ~ 1 (fully ionized)

    Args:
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].

    Returns:
        Average ionization state Z_bar in [0, 1].
    """
    S = _saha_parameter(Te, ne)
    Z_bar = S / (1.0 + S)
    # Clamp to [0, 1] for safety
    return min(max(Z_bar, 0.0), 1.0)


@njit(cache=True)
def saha_ionization_fraction_array(
    Te: np.ndarray, ne: np.ndarray
) -> np.ndarray:
    """Vectorized Saha ionization fraction for arrays.

    Computes Z_bar element-wise for temperature and density arrays.

    Args:
        Te: Electron temperature array [K].
        ne: Electron number density array [m^-3].

    Returns:
        Z_bar array with same shape as input.
    """
    Z_bar = np.empty_like(Te)
    for i in range(Te.size):
        Z_bar.flat[i] = saha_ionization_fraction(Te.flat[i], ne.flat[i])
    return Z_bar


# ===================================================================
# Collisional-Radiative (CR) Model
# ===================================================================


@njit(cache=True)
def lotz_ionization_rate(Te_eV: float, I_Z_eV: float) -> float:
    """Lotz empirical electron-impact ionization rate coefficient.

    S_Z(Te) = a * P / I_Z² * exp(-I_Z / Te) * f(u)

    where u = I_Z / Te, P = number of equivalent electrons in outer
    shell (taken as 1 for simplicity — accurate within factor ~2),
    and f(u) ≈ 1 - (1 - u/(1+u))·exp(-u) is a slowly varying function.

    Reference: Lotz, Z. Phys. 206:205 (1967).

    Args:
        Te_eV: Electron temperature [eV].
        I_Z_eV: Ionization potential of charge state Z [eV].

    Returns:
        Ionization rate coefficient S_Z [m³/s].
    """
    if Te_eV < 0.01 or I_Z_eV < 0.1:
        return 0.0

    u = I_Z_eV / Te_eV
    if u > 500.0:
        return 0.0

    # Lotz formula: S = a / I_Z² * exp(-u) * Ei-like correction
    # Using the simplified 1-subshell version with P=1.
    # a = 4.5e-14 cm² eV² = 4.5e-18 m² eV²
    a_SI = 4.5e-18  # m² eV²

    # f(u) accounts for the Ei integral; simplified Lotz approx:
    # S = a * P * exp(-u) / (I_Z² * u) * [1 - b*exp(-c*(u-1))]
    # For most ions, just using the leading term with P=1:
    exp_neg_u = np.exp(-u)

    # Rate = a / I_Z² * (1/u) * exp(-u) integrated over Maxwellian
    # Lotz gives S_Z ≈ a * P * f(u) / I_Z² where
    # f(u) = exp(-u) * [ln(1 + 1/u) + E_1(u)] but a good approximation
    # that captures the scaling is:
    #   S_Z ≈ (a / I_Z^2) * sqrt(u) * exp(-u) * (1 + 0.5/u)
    # This gives correct high-T (exp(-u)→1) and low-T (exponential cutoff).
    rate = a_SI / (I_Z_eV * I_Z_eV) * np.sqrt(u) * exp_neg_u * (1.0 + 0.5 / u)

    return max(rate, 0.0)


@njit(cache=True)
def radiative_recombination_rate(Te_eV: float, Z_ion: int) -> float:
    """Radiative recombination rate coefficient.

    Uses Seaton's formula (hydrogenic scaling):
        α_rr = 5.2e-14 * Z * (13.6 * Z² / Te_eV)^{0.5} [cm³/s]

    Converted to SI: multiply by 1e-6.

    Reference: Seaton, MNRAS 119:81 (1959).

    Args:
        Te_eV: Electron temperature [eV].
        Z_ion: Charge of the recombining ion (Z → Z-1).

    Returns:
        Radiative recombination rate coefficient α_rr [m³/s].
    """
    if Te_eV < 0.01 or Z_ion < 1:
        return 0.0

    # Seaton formula in CGS
    chi = 13.6 * Z_ion * Z_ion / Te_eV
    alpha_cgs = 5.2e-14 * Z_ion * np.sqrt(chi)  # cm³/s

    return alpha_cgs * 1.0e-6  # m³/s


@njit(cache=True)
def dielectronic_recombination_rate(Te_eV: float, Z_ion: int) -> float:
    """Dielectronic recombination rate coefficient (Burgess formula).

    Simplified Burgess general formula:
        α_dr ≈ 4.8e-12 * Z^{1/2} * (13.6*Z²/Te)^{3/2} * exp(-0.25*13.6*Z²/Te)
                / (Te_eV^{1/2})  [cm³/s]

    This captures the resonance peak at Te ≈ I_Z / 4.

    Reference: Burgess, ApJ 139:776 (1964).

    Args:
        Te_eV: Electron temperature [eV].
        Z_ion: Charge of the recombining ion.

    Returns:
        Dielectronic recombination rate coefficient α_dr [m³/s].
    """
    if Te_eV < 0.01 or Z_ion < 1:
        return 0.0

    chi = 13.6 * Z_ion * Z_ion  # hydrogenic IP in eV
    x = chi / Te_eV

    if x > 500.0:
        return 0.0

    # Simplified Burgess formula
    alpha_cgs = (
        4.8e-12
        * np.sqrt(float(Z_ion))
        * x**1.5
        * np.exp(-0.25 * x)
        / np.sqrt(Te_eV)
    )  # cm³/s

    return max(alpha_cgs * 1.0e-6, 0.0)  # m³/s


@njit(cache=True)
def total_recombination_rate(Te_eV: float, Z_ion: int) -> float:
    """Total recombination rate: radiative + dielectronic.

    Args:
        Te_eV: Electron temperature [eV].
        Z_ion: Charge of the recombining ion (Z → Z-1).

    Returns:
        Total recombination rate coefficient α [m³/s].
    """
    return radiative_recombination_rate(Te_eV, Z_ion) + dielectronic_recombination_rate(
        Te_eV, Z_ion
    )


@njit(cache=True)
def cr_solve_charge_states(
    ne: float,
    Te_eV: float,
    Z_max: int,
    dt: float,
    frac_in: np.ndarray,
    ip_eV: np.ndarray,
) -> np.ndarray:
    """Advance charge-state fractions by dt using implicit backward Euler.

    Solves the rate equations:
        df_Z/dt = ne * [S_{Z-1} f_{Z-1} - (S_Z + α_Z) f_Z + α_{Z+1} f_{Z+1}]

    where S_Z = ionization rate from Z→Z+1, α_Z = recombination rate Z→Z-1.

    Uses implicit backward Euler (tridiagonal system) for stiff rate equations.

    Args:
        ne: Electron density [m^-3].
        Te_eV: Electron temperature [eV].
        Z_max: Maximum charge state (nuclear charge).
        dt: Time step [s].
        frac_in: Input charge-state fractions, shape (Z_max+1,).
            frac_in[Z] = fraction of ions in charge state Z.
        ip_eV: Ionization potentials [eV], shape (Z_max,).
            ip_eV[Z] = energy to ionize from Z to Z+1.

    Returns:
        Updated charge-state fractions, shape (Z_max+1,).
    """
    n_states = Z_max + 1
    frac = frac_in.copy()

    # If ne or Te too low, no ionization/recombination occurs
    if ne < 1.0 or Te_eV < 0.01:
        return frac

    # Compute ionization and recombination rate coefficients
    S = np.zeros(Z_max)  # S[Z] = ionization rate Z→Z+1
    alpha = np.zeros(Z_max + 1)  # alpha[Z] = recomb rate Z→Z-1

    n_ip = min(len(ip_eV), Z_max)
    for Z in range(n_ip):
        S[Z] = lotz_ionization_rate(Te_eV, ip_eV[Z])
    for Z in range(1, n_states):
        alpha[Z] = total_recombination_rate(Te_eV, Z)

    # Build tridiagonal system: (I - dt*ne*A) * f_new = f_old
    # A is the rate matrix: A[Z,Z-1] = S[Z-1], A[Z,Z] = -(S[Z]+alpha[Z]),
    #                        A[Z,Z+1] = alpha[Z+1]
    # Tridiagonal: lower[Z] = -dt*ne*S[Z-1]
    #              diag[Z]  = 1 + dt*ne*(S[Z] + alpha[Z])
    #              upper[Z] = -dt*ne*alpha[Z+1]

    lower = np.zeros(n_states)
    diag = np.zeros(n_states)
    upper = np.zeros(n_states)
    rhs = frac.copy()

    for Z in range(n_states):
        s_z = S[Z] if Z_max > Z else 0.0
        a_z = alpha[Z]
        diag[Z] = 1.0 + dt * ne * (s_z + a_z)
        if Z > 0:
            lower[Z] = -dt * ne * S[Z - 1]
        if Z_max > Z:
            upper[Z] = -dt * ne * alpha[Z + 1]

    # Thomas algorithm for tridiagonal solve
    # Forward sweep
    for i in range(1, n_states):
        if abs(diag[i - 1]) < 1e-300:
            continue
        w = lower[i] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]

    # Back substitution
    if abs(diag[n_states - 1]) > 1e-300:
        frac[n_states - 1] = rhs[n_states - 1] / diag[n_states - 1]
    for i in range(n_states - 2, -1, -1):
        if abs(diag[i]) > 1e-300:
            frac[i] = (rhs[i] - upper[i] * frac[i + 1]) / diag[i]

    # Ensure non-negative and normalize
    for Z in range(n_states):
        if frac[Z] < 0.0:
            frac[Z] = 0.0
    total = 0.0
    for Z in range(n_states):
        total += frac[Z]
    if total > 0.0:
        for Z in range(n_states):
            frac[Z] /= total

    return frac


@njit(cache=True)
def cr_average_charge(
    ne: float,
    Te_eV: float,
    Z_max: int,
    ip_eV: np.ndarray,
    n_iter: int = 20,
) -> float:
    """Compute Z_bar from coronal/CR equilibrium (steady-state).

    Iterates the CR rate equations to steady state from an initial guess.

    Args:
        ne: Electron density [m^-3].
        Te_eV: Electron temperature [eV].
        Z_max: Nuclear charge (maximum ionization state).
        ip_eV: Ionization potentials [eV], shape (Z_max,).
        n_iter: Number of implicit iterations toward equilibrium.

    Returns:
        Average charge state Z_bar.
    """
    n_states = Z_max + 1
    # Initial guess: neutral
    frac = np.zeros(n_states)
    frac[0] = 1.0

    # Use large pseudo-dt to reach steady state quickly
    dt_pseudo = 1.0e-6  # 1 μs — much longer than CR timescales at DPF ne

    for _ in range(n_iter):
        frac = cr_solve_charge_states(ne, Te_eV, Z_max, dt_pseudo, frac, ip_eV)

    # Compute Z_bar = Σ Z * f_Z
    Z_bar = 0.0
    for Z in range(n_states):
        Z_bar += Z * frac[Z]

    return Z_bar


@njit(cache=True)
def cr_zbar_field(
    ne_field: np.ndarray,
    Te_field: np.ndarray,
    Z_max: int,
    ip_eV: np.ndarray,
) -> np.ndarray:
    """Compute spatially-resolved Z_bar(r,z) from CR equilibrium.

    For each cell, computes the steady-state average charge from the
    collisional-radiative model.

    Args:
        ne_field: Electron density field [m^-3], arbitrary shape.
        Te_field: Electron temperature field [K] (will be converted to eV).
        Z_max: Nuclear charge.
        ip_eV: Ionization potentials [eV], shape (Z_max,).

    Returns:
        Z_bar field with same shape as input.
    """
    eV_per_K = 1.0 / 11604.5  # k_B/e ≈ 1/11604.5 eV/K
    Z_bar = np.empty_like(ne_field)
    for i in range(ne_field.size):
        ne_i = ne_field.flat[i]
        Te_eV_i = Te_field.flat[i] * eV_per_K
        Z_bar.flat[i] = cr_average_charge(ne_i, Te_eV_i, Z_max, ip_eV)
    return Z_bar


@njit(cache=True)
def cr_evolve_field(
    ne_field: np.ndarray,
    Te_field: np.ndarray,
    Z_max: int,
    dt: float,
    frac_field: np.ndarray,
    ip_eV: np.ndarray,
) -> np.ndarray:
    """Evolve charge-state fractions for an entire field by dt.

    Args:
        ne_field: Electron density field [m^-3], shape (Nr, Nz) or flat.
        Te_field: Electron temperature field [K].
        Z_max: Nuclear charge.
        dt: Time step [s].
        frac_field: Charge-state fractions, shape (*spatial, Z_max+1).
            Last axis is the charge-state index.
        ip_eV: Ionization potentials [eV], shape (Z_max,).

    Returns:
        Updated frac_field, same shape.
    """
    eV_per_K = 1.0 / 11604.5
    n_cells = ne_field.size
    n_states = Z_max + 1
    out = frac_field.copy()

    for i in range(n_cells):
        ne_i = ne_field.flat[i]
        Te_eV_i = Te_field.flat[i] * eV_per_K
        frac_i = np.empty(n_states)
        for Z in range(n_states):
            frac_i[Z] = frac_field.flat[i * n_states + Z]
        frac_new = cr_solve_charge_states(ne_i, Te_eV_i, Z_max, dt, frac_i, ip_eV)
        for Z in range(n_states):
            out.flat[i * n_states + Z] = frac_new[Z]

    return out
