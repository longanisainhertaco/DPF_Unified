"""Two-temperature electron energy evolution for MHD plasma.

Evolves electron internal energy density as a separate conserved variable,
enabling physically correct Te/Ti separation without the fraction-preserving
hack currently used in mhd_solver.py and cylindrical_mhd.py.

Governing equation (Braginskii 1965):

    d(rho * e_e)/dt + div(rho * e_e * v) = -p_e * div(v) + Q_ohm + Q_ei + Q_rad

where:
    e_e   = (3/2) * n_e * k_B * Te / rho   [J/kg]  (specific electron internal energy)
    p_e   = n_e * k_B * Te                  [Pa]     (electron partial pressure)
    Q_ohm = eta * J^2                       [W/m^3]  (Ohmic heating -> electrons only)
    Q_ei  = electron-ion equilibration      [W/m^3]  (from Spitzer collisions)
    Q_rad = bremsstrahlung + line cooling    [W/m^3]  (radiation losses from electrons)

The ion energy is derived: e_i = e_total - e_e (never separately evolved).

References:
    Braginskii, S.I. (1965) "Transport Processes in a Plasma",
        Reviews of Plasma Physics, Vol. 1, pp. 205-311.
    NRL Plasma Formulary (2019), pp. 34, 58.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann as k_B

from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures
from dpf.constants import eV, m_d, m_e, m_p
from dpf.radiation.bremsstrahlung import bremsstrahlung_power

TWO_TEMPERATURE_MODEL_ROLE = "operator_split_two_temperature_source_terms"
TWO_TEMPERATURE_SOURCE_STATUS = "equilibration_convention_source_audit_incomplete"
TWO_TEMPERATURE_VALIDATION_STATUS = "not_validation_evidence"


def two_temperature_model_metadata() -> dict[str, object]:
    """Return fail-closed source-status metadata for 2T source terms."""
    return {
        "model_role": TWO_TEMPERATURE_MODEL_ROLE,
        "source_status": TWO_TEMPERATURE_SOURCE_STATUS,
        "validation_status": TWO_TEMPERATURE_VALIDATION_STATUS,
        "can_support_validation_claims": False,
        "components": {
            "electron_energy": (
                "Ideal electron internal-energy bookkeeping is implemented."
            ),
            "ohmic_heating": "Ohmic heating is deposited into electron energy.",
            "equilibration": (
                "Electron-ion relaxation uses the shared Spitzer helper; the "
                "active arbitrary-Te/Ti convention is cross-checked against "
                "the local NRL equal-temperature equilibration formula but "
                "still needs same-scope review before validation use."
            ),
            "radiation_loss": (
                "Bremsstrahlung loss is delegated to the radiation helper; line "
                "cooling and opacity remain separately source-blocked."
            ),
        },
        "validity_notes": {
            "claim_limit": (
                "Use as engineering 2T energy bookkeeping only until same-scope "
                "temperature diagnostics and source-closed relaxation evidence exist."
            ),
        },
    }


def electron_energy_from_temperature(
    Te: np.ndarray,
    n_e: np.ndarray,
) -> np.ndarray:
    """Convert electron temperature to volumetric electron energy density.

    Parameters
    ----------
    Te : np.ndarray
        Electron temperature [K].
    n_e : np.ndarray
        Electron number density [m^-3].

    Returns
    -------
    np.ndarray
        Electron energy density rho_e_e = (3/2) * n_e * k_B * Te  [J/m^3].
    """
    return 1.5 * n_e * k_B * Te


def temperature_from_electron_energy(
    rho_e_e: np.ndarray,
    n_e: np.ndarray,
    Te_floor: float = 1.0,
) -> np.ndarray:
    """Recover electron temperature from volumetric energy density.

    Parameters
    ----------
    rho_e_e : np.ndarray
        Electron energy density [J/m^3].
    n_e : np.ndarray
        Electron number density [m^-3].
    Te_floor : float
        Minimum electron temperature [K].

    Returns
    -------
    np.ndarray
        Electron temperature [K], floored at Te_floor.
    """
    Te = (2.0 / 3.0) * rho_e_e / np.maximum(n_e * k_B, 1e-300)
    return np.maximum(Te, Te_floor)


def ion_temperature_from_total(
    e_total: np.ndarray,
    rho_e_e: np.ndarray,
    n_i: np.ndarray,
    Ti_floor: float = 1.0,
) -> np.ndarray:
    """Derive ion temperature from total internal energy minus electron energy.

    Parameters
    ----------
    e_total : np.ndarray
        Total internal energy density [J/m^3]:  p / (gamma - 1).
    rho_e_e : np.ndarray
        Electron energy density [J/m^3].
    n_i : np.ndarray
        Ion number density [m^-3].
    Ti_floor : float
        Minimum ion temperature [K].

    Returns
    -------
    np.ndarray
        Ion temperature [K].
    """
    e_ion = np.maximum(e_total - rho_e_e, 0.0)
    Ti = (2.0 / 3.0) * e_ion / np.maximum(n_i * k_B, 1e-300)
    return np.maximum(Ti, Ti_floor)


def compute_ohmic_heating(
    eta: np.ndarray,
    J_sq: np.ndarray,
) -> np.ndarray:
    """Ohmic (Joule) heating rate deposited into electrons.

    Parameters
    ----------
    eta : np.ndarray
        Resistivity [Ohm*m].
    J_sq : np.ndarray
        Current density magnitude squared |J|^2  [A^2/m^4].

    Returns
    -------
    np.ndarray
        Volumetric heating rate Q_ohm = eta * |J|^2  [W/m^3].
    """
    return eta * J_sq


def compute_equilibration_source(
    Te: np.ndarray,
    Ti: np.ndarray,
    n_e: np.ndarray,
    Z: float = 1.0,
) -> np.ndarray:
    """Electron-ion equilibration energy transfer rate.

    Positive Q_ei means energy flows INTO electrons (Ti > Te).
    Uses Spitzer collision frequency from the collision module.

    Parameters
    ----------
    Te : np.ndarray
        Electron temperature [K].
    Ti : np.ndarray
        Ion temperature [K].
    n_e : np.ndarray
        Electron number density [m^-3].
    Z : float
        Ion charge state.

    Returns
    -------
    np.ndarray
        Volumetric energy transfer rate Q_ei [W/m^3].
        Positive = energy into electrons, negative = energy out of electrons.

    Notes
    -----
    Q_ei = 3 * n_e * k_B * (Ti - Te) * m_e / (m_i * tau_eq)

    where tau_eq = 1 / nu_ei. This is the standard Braginskii (1965)
    equilibration rate with mass ratio correction.
    """
    Te_safe = np.maximum(Te, 1.0)
    lnL = coulomb_log(n_e, Te_safe)
    freq_ei = nu_ei(n_e, Te_safe, lnL, Z)

    # Energy transfer: Q = 3 * n_e * k_B * (Ti - Te) * (m_e / m_i) * nu_ei
    # Factor of 3 from (3/2) * 2 * (m_e/m_i) in Braginskii's formulation
    Q_ei = 3.0 * n_e * k_B * (Ti - Te) * (m_e / m_d) * freq_ei
    return Q_ei


def nrl_equal_temperature_ei_equilibration_frequency(
    ion_density_m3: np.ndarray,
    temperature_K: np.ndarray,
    *,
    Z: float = 1.0,
    ion_mass_kg: float = m_d,
    coulomb_log_value: np.ndarray | float | None = None,
) -> np.ndarray:
    """NRL equal-temperature electron-ion thermal equilibration frequency.

    Local source:
    ``KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2996-3020``.
    The cited NRL special case is for ``Te ~= Ti == T`` and gives
    ``nu_e|i^epsilon / n_i = 3.2e-9 Z^2 lambda / (mu T_eV^1.5)``
    in ``cm^3/s``.  This helper is therefore an audit/reference channel for
    the active arbitrary-Te/Ti relaxation, not a whole-shot validation claim.
    """
    ni = np.asarray(ion_density_m3, dtype=float)
    T = np.asarray(temperature_K, dtype=float)
    if np.any(ni <= 0.0):
        raise ValueError("ion_density_m3 must be positive")
    if np.any(T <= 0.0):
        raise ValueError("temperature_K must be positive")
    if Z <= 0.0:
        raise ValueError("Z must be positive")
    if ion_mass_kg <= 0.0:
        raise ValueError("ion_mass_kg must be positive")

    T_eV = np.maximum(k_B * T / eV, 1.0e-300)
    ni_cm3 = ni * 1.0e-6
    mu = ion_mass_kg / m_p
    if coulomb_log_value is None:
        lnL = coulomb_log(np.maximum(Z * ni, 1.0), T)
    else:
        lnL = np.asarray(coulomb_log_value, dtype=float)
    frequency = 3.2e-9 * ni_cm3 * Z * Z * lnL / (mu * T_eV**1.5)
    return np.where(np.isfinite(frequency), frequency, 0.0)


def equilibration_convention_audit(
    *,
    electron_temperature_K: np.ndarray,
    ion_temperature_K: np.ndarray,
    electron_density_m3: np.ndarray,
    ion_density_m3: np.ndarray,
    Z: float = 1.0,
    ion_mass_kg: float = m_d,
) -> dict[str, float | str | bool]:
    """Compare active relaxation rate to the local NRL equal-T reference."""
    Te = np.asarray(electron_temperature_K, dtype=float)
    Ti = np.asarray(ion_temperature_K, dtype=float)
    ne = np.asarray(electron_density_m3, dtype=float)
    ni = np.asarray(ion_density_m3, dtype=float)
    T_ref = np.maximum(0.5 * (Te + Ti), 1.0)
    lnL = coulomb_log(ne, np.maximum(Te, 1.0))
    active_rate = 2.0 * (m_e / ion_mass_kg) * nu_ei(ne, np.maximum(Te, 1.0), lnL, Z)
    nrl_rate = nrl_equal_temperature_ei_equilibration_frequency(
        ni,
        T_ref,
        Z=Z,
        ion_mass_kg=ion_mass_kg,
    )
    ratio = np.divide(
        active_rate,
        nrl_rate,
        out=np.full_like(active_rate, np.nan, dtype=float),
        where=nrl_rate > 0.0,
    )
    finite_ratio = ratio[np.isfinite(ratio)]
    max_ratio = float(np.max(finite_ratio)) if finite_ratio.size else float("nan")
    min_ratio = float(np.min(finite_ratio)) if finite_ratio.size else float("nan")
    return {
        "status": "candidate_nrl_equal_temperature_equilibration_audit",
        "source": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "source_lines": "2996-3020",
        "active_model": "spitzer_nu_ei_mass_ratio_temperature_relaxation",
        "reference_model": "nrl_equal_temperature_thermal_equilibration_frequency",
        "min_active_to_nrl_rate_ratio": min_ratio,
        "max_active_to_nrl_rate_ratio": max_ratio,
        "can_support_first_principles_acceptance": False,
    }


def compute_radiation_loss(
    Te: np.ndarray,
    n_e: np.ndarray,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
) -> np.ndarray:
    """Radiation loss rate from electrons (bremsstrahlung).

    Parameters
    ----------
    Te : np.ndarray
        Electron temperature [K].
    n_e : np.ndarray
        Electron number density [m^-3].
    Z : float
        Ion charge state.
    gaunt_factor : float
        Gaunt factor (default 1.2 for DPF conditions).

    Returns
    -------
    np.ndarray
        Volumetric radiation loss Q_rad [W/m^3].
        Always non-negative (energy leaves electrons).
    """
    return bremsstrahlung_power(n_e, Te, Z, gaunt_factor)


def electron_energy_rhs(
    rho_e_e: np.ndarray,
    rho: np.ndarray,
    velocity: np.ndarray,
    eta: np.ndarray,
    J_sq: np.ndarray,
    Te: np.ndarray,
    Ti: np.ndarray,
    n_e: np.ndarray,
    n_i: np.ndarray,
    dx: float,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
    gamma: float = 5.0 / 3.0,
) -> np.ndarray:
    """Compute the RHS of the electron energy equation (source terms only).

    Computes:
        d(rho_e_e)/dt = -p_e * div(v) + Q_ohm + Q_ei - Q_rad

    The advection term div(rho_e_e * v) is handled by the MHD solver's
    transport step, not here. This function returns only the source terms
    that must be operator-split from the MHD advection.

    Parameters
    ----------
    rho_e_e : np.ndarray
        Electron energy density [J/m^3].
    rho : np.ndarray
        Mass density [kg/m^3].
    velocity : np.ndarray
        Velocity field, shape (3, *spatial_dims) [m/s].
    eta : np.ndarray
        Resistivity field [Ohm*m].
    J_sq : np.ndarray
        Current density squared |J|^2 [A^2/m^4].
    Te : np.ndarray
        Electron temperature [K].
    Ti : np.ndarray
        Ion temperature [K].
    n_e : np.ndarray
        Electron number density [m^-3].
    n_i : np.ndarray
        Ion number density [m^-3].
    dx : float
        Grid spacing [m] (uniform, isotropic).
    Z : float
        Ion charge state.
    gaunt_factor : float
        Gaunt factor for bremsstrahlung.
    gamma : float
        Adiabatic index.

    Returns
    -------
    np.ndarray
        Source term d(rho_e_e)/dt [W/m^3].
    """
    p_e = n_e * k_B * Te

    # Velocity divergence: div(v)
    ndim = velocity.shape[0]
    div_v = np.zeros_like(rho)
    for d in range(min(ndim, len(rho.shape))):
        div_v += np.gradient(velocity[d], dx, axis=d)

    # Compressional work: -p_e * div(v)
    compression = -p_e * div_v

    # Ohmic heating
    Q_ohm = compute_ohmic_heating(eta, J_sq)

    # Electron-ion equilibration
    Q_ei = compute_equilibration_source(Te, Ti, n_e, Z)

    # Radiation loss
    Q_rad = compute_radiation_loss(Te, n_e, Z, gaunt_factor)

    return compression + Q_ohm + Q_ei - Q_rad


def step_electron_energy(
    rho_e_e: np.ndarray,
    rho: np.ndarray,
    velocity: np.ndarray,
    eta: np.ndarray,
    J_sq: np.ndarray,
    Te: np.ndarray,
    Ti: np.ndarray,
    n_e: np.ndarray,
    n_i: np.ndarray,
    dx: float,
    dt: float,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
    gamma: float = 5.0 / 3.0,
    Te_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance electron energy density by one timestep (source terms only).

    Uses forward Euler for the source term update. The advection of
    rho_e_e is handled by the MHD transport step (not here).

    For the electron-ion equilibration sub-step, delegates to the
    exact implicit solver in spitzer.py for unconditional stability.

    Parameters
    ----------
    rho_e_e : np.ndarray
        Electron energy density [J/m^3].
    rho : np.ndarray
        Mass density [kg/m^3].
    velocity : np.ndarray
        Velocity field, shape (3, *spatial_dims) [m/s].
    eta : np.ndarray
        Resistivity field [Ohm*m].
    J_sq : np.ndarray
        Current density squared |J|^2 [A^2/m^4].
    Te : np.ndarray
        Electron temperature [K].
    Ti : np.ndarray
        Ion temperature [K].
    n_e : np.ndarray
        Electron number density [m^-3].
    n_i : np.ndarray
        Ion number density [m^-3].
    dx : float
        Grid spacing [m].
    dt : float
        Timestep [s].
    Z : float
        Ion charge state.
    gaunt_factor : float
        Gaunt factor for bremsstrahlung.
    gamma : float
        Adiabatic index.
    Te_floor : float
        Minimum electron temperature [K].

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (rho_e_e_new, Te_new, Ti_new) after source term update.
    """
    p_e = n_e * k_B * Te

    # --- Step 1: Compressional work (explicit) ---
    ndim = velocity.shape[0]
    div_v = np.zeros_like(rho)
    for d in range(min(ndim, len(rho.shape))):
        div_v += np.gradient(velocity[d], dx, axis=d)
    rho_e_e = rho_e_e - dt * p_e * div_v

    # --- Step 2: Ohmic heating (explicit, electrons only) ---
    Q_ohm = compute_ohmic_heating(eta, J_sq)
    rho_e_e = rho_e_e + dt * Q_ohm

    # --- Step 3: Radiation losses (explicit, from electrons) ---
    Q_rad = compute_radiation_loss(Te, n_e, Z, gaunt_factor)
    rho_e_e = rho_e_e - dt * Q_rad

    # Floor electron energy
    e_e_floor = 1.5 * n_e * k_B * Te_floor
    rho_e_e = np.maximum(rho_e_e, e_e_floor)

    # --- Step 4: Electron-ion equilibration (implicit, via Spitzer) ---
    Te_mid = temperature_from_electron_energy(rho_e_e, n_e, Te_floor)
    Te_safe = np.maximum(Te_mid, 1.0)
    lnL = coulomb_log(n_e, Te_safe)
    freq_ei = nu_ei(n_e, Te_safe, lnL, Z)
    Te_new, Ti_new = relax_temperatures(Te_mid, Ti, freq_ei, dt, Z)

    # Recover electron energy from relaxed temperature
    Te_new = np.maximum(Te_new, Te_floor)
    Ti_new = np.maximum(Ti_new, Te_floor)
    rho_e_e_new = electron_energy_from_temperature(Te_new, n_e)

    return rho_e_e_new, Te_new, Ti_new


def initialize_electron_energy(
    Te: np.ndarray,
    Ti: np.ndarray,
    pressure: np.ndarray,
    rho: np.ndarray,
    ion_mass: float,
    Z: float = 1.0,
) -> np.ndarray:
    """Initialize electron energy density from existing Te field.

    Used when transitioning from the old fraction-preserving hack
    to the conserved electron energy variable.

    Parameters
    ----------
    Te : np.ndarray
        Electron temperature [K].
    Ti : np.ndarray
        Ion temperature [K] (unused, for interface consistency).
    pressure : np.ndarray
        Total pressure [Pa] (unused, for cross-check only).
    rho : np.ndarray
        Mass density [kg/m^3].
    ion_mass : float
        Ion mass [kg] (e.g., m_d for deuterium).
    Z : float
        Ion charge state.

    Returns
    -------
    np.ndarray
        Electron energy density rho_e_e [J/m^3].
    """
    n_i = rho / ion_mass
    n_e = Z * n_i
    return electron_energy_from_temperature(Te, n_e)
