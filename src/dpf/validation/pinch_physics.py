"""Post-pinch physics diagnostics for dense plasma focus.

Implements DPF-specific pinch physics models and diagnostics from recent
literature:

- **Implosion/stagnation**: 1D shock theory (Angus 2021, Goyon et al. 2025)
- **Magneto-Rayleigh-Taylor**: Growth rates with B-field suppression
  (Bian et al. 2026)
- **Neutron yield scaling**: I^4 empirical law (Lee & Saw 2008)
- **Collisionality**: ND parameter for MHD validity (Kindi et al. 2026)
- **Post-pinch expansion**: Timescales for m=0 disruption and expansion

These diagnostics connect the 0D Lee model to spatially-resolved physics,
providing quantitative predictions testable against experimental data.

References
----------
- Goyon et al., Phys. Plasmas 32, 033105 (2025). DOI: 10.1063/5.0253547
- Bian et al., Phys. Plasmas 33, 012303 (2026). DOI: 10.1063/5.0305344
- Kindi et al., Phys. Plasmas (2026). DOI: 10.1063/5.0294460
- Lee & Saw, J. Fusion Energy 27, 292-295 (2008).
- Angus et al., Phys. Plasmas 28, 012705 (2021).
- Auluck, Phys. Plasmas 31, 010704 (2024). DOI: 10.1063/5.0189593
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants (SI)
MU_0 = 4 * np.pi * 1e-7       # Vacuum permeability [H/m]
K_B = 1.380649e-23             # Boltzmann constant [J/K]
E_CHARGE = 1.602176634e-19     # Elementary charge [C]
EPSILON_0 = 8.854187817e-12    # Vacuum permittivity [F/m]
M_PROTON = 1.672621898e-27     # Proton mass [kg]
M_DEUTERON = 2 * M_PROTON      # Deuteron mass [kg] (approx)
KEV_TO_JOULE = 1.602176634e-16  # 1 keV in Joules


# =====================================================================
# 1D Shock Theory — Implosion and Stagnation (Angus 2021 / Goyon 2025)
# =====================================================================


def implosion_velocity(
    I_MA: float,
    R_cm: float,
    P_Torr: float,
) -> float:
    """Estimate sheath implosion velocity from 1D shock theory.

    From Angus (2021) / Goyon et al. (2025) Eq. 1.  Empirical fit
    calibrated against Rankine-Hugoniot simulations of magnetically-driven
    cylindrical implosion in a uniform fill gas.  The coefficient 950 is
    a dimensional fit constant, not derived analytically from conservation
    laws.

    Args:
        I_MA: Drive current at implosion [MA].
        R_cm: Implosion radius (anode inner radius) [cm].
        P_Torr: Fill gas pressure [Torr].

    Returns:
        Implosion velocity [km/s].

    Notes:
        Verified value: 993 km/s from full R-H analysis vs 950 from
        the approximate formula (4.5% difference).
        Formula assumes cylindrical Z-pinch geometry with D2 fill.
    """
    if R_cm <= 0 or P_Torr <= 0 or I_MA <= 0:
        raise ValueError("All parameters must be positive")
    return 950.0 * I_MA / (R_cm * np.sqrt(P_Torr))


def stagnation_temperature(
    I_MA: float,
    R_cm: float,
    P_Torr: float,
) -> float:
    """Estimate post-shock ion temperature at stagnation.

    From Goyon et al. (2025) Eq. 2 / Angus (2021), derived from
    strong-shock Rankine-Hugoniot jump conditions with v_imp from Eq. 1.

    Args:
        I_MA: Drive current at implosion [MA].
        R_cm: Implosion radius [cm].
        P_Torr: Fill gas pressure [Torr].

    Returns:
        Stagnation ion temperature [keV].

    Notes:
        This is the thermalization temperature, not the beam-target
        temperature which can be much higher (up to ~100 keV).
    """
    if R_cm <= 0 or P_Torr <= 0 or I_MA <= 0:
        raise ValueError("All parameters must be positive")
    return 21.0 * I_MA**2 / (R_cm**2 * P_Torr)


def expansion_timescale(
    R_cm: float,
    P_Torr: float,
    CR: float,
    I_MA: float,
) -> float:
    """Estimate post-pinch expansion timescale.

    From Goyon et al. (2025) Eq. 3, the characteristic time for the
    stagnated plasma column to expand back to 2x the stagnation radius.

    Args:
        R_cm: Implosion radius [cm].
        P_Torr: Fill gas pressure [Torr].
        CR: Compression ratio (R_implosion / R_stagnation). Typical ~10.
        I_MA: Drive current at implosion [MA].

    Returns:
        Expansion timescale [ns].

    Notes:
        v_exp ~ v_imp / 3, r_exp ~ 2 * r_stag.
        For MJOLNIR at 60 kV: tau_exp ~ 30-50 ns.
    """
    if R_cm <= 0 or P_Torr <= 0 or I_MA <= 0 or CR <= 0:
        raise ValueError("All parameters must be positive")
    return 31.5 * R_cm**2 * np.sqrt(P_Torr) / (CR * I_MA)


def m0_instability_timescale(
    R_cm: float,
    P_Torr: float,
    CR: float,
    I_MA: float,
) -> float:
    """Estimate m=0 sausage instability growth timescale.

    From Goyon et al. (2025) Eq. 4. The characteristic growth time
    for the fastest-growing m=0 MHD mode with wavelength ~ r_stag.

    Args:
        R_cm: Implosion radius [cm].
        P_Torr: Fill gas pressure [Torr].
        CR: Compression ratio. Typical ~10.
        I_MA: Drive current at implosion [MA].

    Returns:
        m=0 instability timescale [ns].

    Notes:
        When tau_m0 < tau_exp, the pinch disrupts before significant
        expansion. Multiple m=0 disruptions are observed experimentally
        in MA-class devices (Goyon et al. 2025, Fig. 8).
    """
    if R_cm <= 0 or P_Torr <= 0 or I_MA <= 0 or CR <= 0:
        raise ValueError("All parameters must be positive")
    return 31.0 * R_cm**2 * np.sqrt(P_Torr) / (CR * I_MA)


@dataclass
class StagnationDiagnostics:
    """Collection of stagnation diagnostics for a given DPF configuration.

    Attributes:
        v_imp: Implosion velocity [km/s].
        T_stag: Stagnation temperature [keV].
        tau_exp: Expansion timescale [ns].
        tau_m0: m=0 instability timescale [ns].
        disruption_ratio: tau_m0 / tau_exp. < 1 means m=0 disrupts
            before expansion.
        I_MA: Drive current used [MA].
        R_cm: Radius used [cm].
        P_Torr: Pressure used [Torr].
        CR: Compression ratio used.
    """

    v_imp: float
    T_stag: float
    tau_exp: float
    tau_m0: float
    disruption_ratio: float
    I_MA: float
    R_cm: float
    P_Torr: float
    CR: float


def stagnation_diagnostics(
    I_MA: float,
    R_cm: float,
    P_Torr: float,
    CR: float = 10.0,
) -> StagnationDiagnostics:
    """Compute all stagnation diagnostics for a DPF configuration.

    Args:
        I_MA: Drive current at implosion [MA].
        R_cm: Implosion radius [cm].
        P_Torr: Fill gas pressure [Torr].
        CR: Compression ratio (default 10).

    Returns:
        :class:`StagnationDiagnostics` with all computed quantities.
    """
    v_imp = implosion_velocity(I_MA, R_cm, P_Torr)
    T_stag = stagnation_temperature(I_MA, R_cm, P_Torr)
    tau_exp = expansion_timescale(R_cm, P_Torr, CR, I_MA)
    tau_m0 = m0_instability_timescale(R_cm, P_Torr, CR, I_MA)

    return StagnationDiagnostics(
        v_imp=v_imp,
        T_stag=T_stag,
        tau_exp=tau_exp,
        tau_m0=tau_m0,
        disruption_ratio=tau_m0 / tau_exp,
        I_MA=I_MA,
        R_cm=R_cm,
        P_Torr=P_Torr,
        CR=CR,
    )


# =====================================================================
# Magneto-Rayleigh-Taylor Instability (Bian et al. 2026)
# =====================================================================


def mrti_growth_rate(
    g: float,
    k: float,
    A: float,
    B: float = 0.0,
    theta: float = 0.0,
    rho_h: float = 1.0,
    rho_l: float = 0.0,
) -> float:
    """Compute magneto-Rayleigh-Taylor instability growth rate.

    From Bian et al. (2026) Eq. 3, the linear MRT growth rate
    including magnetic field stabilization for a sharp interface.

    Args:
        g: Effective gravity (deceleration) [m/s^2].
        k: Wavenumber of perturbation [1/m].
        A: Atwood number = (rho_h - rho_l) / (rho_h + rho_l).
            Must be in [0, 1].
        B: Magnetic field strength [T]. Stabilizes if parallel to k.
        theta: Angle between B and k [rad].
            theta=0: B parallel to k (maximum stabilization).
            theta=pi/2: B perpendicular to k (no stabilization).
        rho_h: Heavy fluid density [kg/m^3].
        rho_l: Light fluid density [kg/m^3].

    Returns:
        Growth rate gamma [1/s]. Returns 0 if stabilized (gamma^2 < 0).

    Notes:
        For DPF: B_theta is the azimuthal pinch field, g is the
        deceleration during radial compression. Perturbations along
        the z-axis (axial) have theta ~ 0 when B = B_theta = 0
        (no stabilization). Axial B_z provides stabilization.

        Classical RT (B=0): gamma = sqrt(g * k * A)
        Magnetic RT: gamma = sqrt(g*k*A - 2*B^2*k^2*cos^2(theta)/(rho_h+rho_l))
    """
    if A < 0 or A > 1:
        raise ValueError(f"Atwood number must be in [0, 1], got {A}")
    if k < 0:
        raise ValueError(f"Wavenumber must be non-negative, got {k}")

    rho_sum = rho_h + rho_l
    if rho_sum <= 0:
        raise ValueError("Sum of densities must be positive")

    # Classical RT term
    rt_term = g * k * A

    # Magnetic stabilization term
    mag_term = 2 * B**2 * k**2 * np.cos(theta)**2 / (MU_0 * rho_sum)

    gamma_sq = rt_term - mag_term

    if gamma_sq <= 0:
        return 0.0  # Stabilized
    return float(np.sqrt(gamma_sq))


def classical_rt_growth_rate(g: float, k: float, A: float) -> float:
    """Classical Rayleigh-Taylor growth rate (no magnetic field).

    Args:
        g: Effective gravity [m/s^2].
        k: Wavenumber [1/m].
        A: Atwood number (0 to 1).

    Returns:
        Growth rate gamma [1/s].
    """
    return mrti_growth_rate(g, k, A, B=0.0)


def critical_magnetic_field(
    rho_h: float,
    rho_l: float,
    g: float,
    wavelength: float,
) -> float:
    """Critical magnetic field for mRT stabilization.

    From Bian et al. (2026) Eq. 5. Above this B, the mode with
    given wavelength is stabilized (gamma = 0).

    Args:
        rho_h: Heavy fluid density [kg/m^3].
        rho_l: Light fluid density [kg/m^3].
        g: Effective gravity [m/s^2].
        wavelength: Perturbation wavelength [m].

    Returns:
        Critical magnetic field B_c [T].

    Notes:
        Assumes B parallel to k (maximum stabilization, theta=0).
        B_c = sqrt(mu_0 * (rho_h - rho_l) * g * lambda / (4*pi))
    """
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive")
    drho = rho_h - rho_l
    if drho < 0:
        raise ValueError("rho_h must be >= rho_l")

    return float(np.sqrt(MU_0 * drho * g * wavelength / (4 * np.pi)))


def mrti_saturated_growth_rate(
    A: float,
    g: float,
    V_A: float,
    a1: float = 1.0,
    a2: float = 1.0,
) -> float:
    """Maximum mRT growth rate in the strong-B limit.

    From Bian et al. (2026) Eq. 7. When B_z is strong, the growth
    rate saturates at a maximum value independent of wavenumber.

    Args:
        A: Atwood number.
        g: Effective gravity [m/s^2].
        V_A: Alfven velocity [m/s].
        a1: Density ratio parameter (rho_h / rho_ref).
        a2: Density ratio parameter (rho_l / rho_ref).

    Returns:
        Saturated maximum growth rate [1/s].

    Notes:
        gamma_max = 2*A*g / (V_A * (sqrt(a1) + sqrt(a2)))
        This applies when B is parallel to k. For perpendicular B,
        classical RT is recovered.
    """
    if V_A <= 0:
        raise ValueError("Alfven velocity must be positive")
    return 2 * A * g / (V_A * (np.sqrt(a1) + np.sqrt(a2)))


@dataclass
class MRTIDiagnostics:
    """Magneto-RT diagnostics for a DPF pinch configuration.

    Attributes:
        gamma_classical: Classical RT growth rate (no B) [1/s].
        gamma_with_B: MRT growth rate with given B [1/s].
        B_critical: Critical B for stabilization [T].
        suppression_factor: gamma_with_B / gamma_classical.
            0 = fully stabilized, 1 = no suppression.
        e_foldings: Number of e-foldings during pinch lifetime.
        wavelength: Perturbation wavelength used [m].
        B_applied: Magnetic field used [T].
    """

    gamma_classical: float
    gamma_with_B: float
    B_critical: float
    suppression_factor: float
    e_foldings: float
    wavelength: float
    B_applied: float


def mrti_diagnostics(
    g: float,
    rho_h: float,
    rho_l: float,
    wavelength: float,
    B: float = 0.0,
    theta: float = 0.0,
    pinch_lifetime_ns: float = 30.0,
) -> MRTIDiagnostics:
    """Compute comprehensive mRT diagnostics for a DPF pinch.

    Args:
        g: Effective gravity (deceleration) [m/s^2].
        rho_h: Heavy fluid density [kg/m^3].
        rho_l: Light fluid density [kg/m^3].
        wavelength: Perturbation wavelength [m].
        B: Applied magnetic field [T].
        theta: Angle between B and k [rad].
        pinch_lifetime_ns: Pinch lifetime [ns] for e-folding calculation.

    Returns:
        :class:`MRTIDiagnostics` with all computed quantities.
    """
    rho_sum = rho_h + rho_l
    A = (rho_h - rho_l) / rho_sum if rho_sum > 0 else 0.0
    k = 2 * np.pi / wavelength

    gamma_cl = classical_rt_growth_rate(g, k, A)
    gamma_B = mrti_growth_rate(g, k, A, B, theta, rho_h, rho_l)
    B_c = critical_magnetic_field(rho_h, rho_l, g, wavelength)

    suppression = gamma_B / gamma_cl if gamma_cl > 0 else 0.0
    tau = pinch_lifetime_ns * 1e-9  # Convert to seconds
    e_folds = gamma_B * tau

    return MRTIDiagnostics(
        gamma_classical=gamma_cl,
        gamma_with_B=gamma_B,
        B_critical=B_c,
        suppression_factor=suppression,
        e_foldings=e_folds,
        wavelength=wavelength,
        B_applied=B,
    )


# =====================================================================
# Neutron Yield Scaling (Lee & Saw 2008)
# =====================================================================


def neutron_yield_I4(
    I_peak_MA: float,
    coefficient: float = 9e10,
) -> float:
    """Estimate DD neutron yield from I^4 scaling law.

    From Lee & Saw (2008), the DD neutron yield scales as Y_n ~ C * I^4
    where I is in MA. The coefficient C depends on the device class
    and operating conditions.

    Args:
        I_peak_MA: Peak current [MA].
        coefficient: Scaling coefficient. Default 9e10 from
            Lee & Saw (2008) for conventional DPF devices.
            Herold (1989): ~1e11 for MA-class devices.

    Returns:
        Estimated DD neutron yield [neutrons/shot].

    Notes:
        The I^4 law holds over ~5 orders of magnitude (10 kA to 5 MA).
        Deviations occur for:
        - Very high I (> 3 MA): radiative collapse, beam-target dominates
        - Very low I (< 50 kA): threshold effects
        - Non-optimal fill pressure

        From Goyon et al. (2025) Table I:
        Device      I_peak (MA)   Y_n (DD)
        PF-1000     2.0           2e11
        POSEIDON    4.6           4.6e11
        MJOLNIR     2.8           8e11
        Gemini      6.0           1-2e12
    """
    return coefficient * I_peak_MA**4


@dataclass
class NeutronYieldComparison:
    """Comparison of predicted vs measured neutron yield.

    Attributes:
        I_peak_MA: Peak current [MA].
        Y_predicted: Predicted yield from I^4 law.
        Y_measured: Measured yield (if available).
        ratio: Y_predicted / Y_measured (if available).
        device_name: Device name.
    """

    I_peak_MA: float
    Y_predicted: float
    Y_measured: float | None = None
    ratio: float | None = None
    device_name: str = ""


# Published MA-class device data from Goyon et al. (2025) Table I
_MA_CLASS_DEVICES: dict[str, dict[str, float]] = {
    "LANL-DPF6": {"I_peak_MA": 2.3, "Y_n": 1.5e12, "E_MJ": 0.42},
    "POSEIDON": {"I_peak_MA": 4.6, "Y_n": 4.6e11, "E_MJ": 0.50},
    "Verus": {"I_peak_MA": 2.0, "Y_n": 4.6e11, "E_MJ": 0.75},
    "PF-1000": {"I_peak_MA": 2.0, "Y_n": 2e11, "E_MJ": 1.0},
    "Gemini": {"I_peak_MA": 6.0, "Y_n": 1.5e12, "E_MJ": 2.0},
    "MJOLNIR": {"I_peak_MA": 2.8, "Y_n": 8e11, "E_MJ": 2.0},
}


@dataclass
class I4FitResult:
    """Result of fitting Y_n = C * I^n to multi-device data.

    Attributes:
        coefficient: Best-fit C.
        exponent: Best-fit n (4.0 if forced, free otherwise).
        r_squared: Coefficient of determination R^2.
        n_devices: Number of devices used in fit.
        forced_exponent: Whether exponent was fixed at 4.0.
    """

    coefficient: float
    exponent: float
    r_squared: float
    n_devices: int
    forced_exponent: bool


def fit_I4_coefficient(
    device_data: dict[str, dict[str, float]] | None = None,
    *,
    free_exponent: bool = False,
) -> float | I4FitResult:
    """Fit the I^n scaling coefficient from multi-device data.

    Uses least-squares fit of log(Y_n) = log(C) + n*log(I_peak)
    to published device data.

    Args:
        device_data: Dict of device data, each with 'I_peak_MA'
            and 'Y_n' keys. Default: MA-class devices from Goyon (2025).
        free_exponent: If True, fit both C and n freely and return
            an :class:`I4FitResult`. If False, fix n=4 and return
            just the coefficient C (backward compatible).

    Returns:
        If ``free_exponent=False``: best-fit coefficient C (float).
        If ``free_exponent=True``: :class:`I4FitResult` with C, n, R^2.

    Notes:
        The I^4 law (Lee & Saw 2008) is derived under assumptions of
        beam-target dominance, optimal fill pressure, and similar
        geometry class.  The free-exponent fit tests whether these
        assumptions hold for the given dataset.
    """
    if device_data is None:
        device_data = _MA_CLASS_DEVICES

    log_I = []
    log_Y = []
    for data in device_data.values():
        log_I.append(np.log(data["I_peak_MA"]))
        log_Y.append(np.log(data["Y_n"]))

    log_I_arr = np.array(log_I)
    log_Y_arr = np.array(log_Y)

    if not free_exponent:
        # Fix n=4, solve for log(C) = mean(log(Y) - 4*log(I))
        log_C = np.mean(log_Y_arr - 4 * log_I_arr)
        return float(np.exp(log_C))

    # Free-exponent fit: log(Y) = log(C) + n * log(I)
    coeffs = np.polyfit(log_I_arr, log_Y_arr, 1)
    n_fit = float(coeffs[0])
    log_C_fit = float(coeffs[1])

    # R^2
    y_pred = coeffs[0] * log_I_arr + coeffs[1]
    ss_res = float(np.sum((log_Y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((log_Y_arr - np.mean(log_Y_arr)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    return I4FitResult(
        coefficient=float(np.exp(log_C_fit)),
        exponent=n_fit,
        r_squared=r_squared,
        n_devices=len(device_data),
        forced_exponent=False,
    )


# =====================================================================
# Collisionality Diagnostics (Kindi et al. 2026)
# =====================================================================


def debye_length(n_e: float, T_e_eV: float) -> float:
    """Compute electron Debye length.

    Args:
        n_e: Electron density [m^-3].
        T_e_eV: Electron temperature [eV].

    Returns:
        Debye length [m].
    """
    if n_e <= 0 or T_e_eV <= 0:
        raise ValueError("Density and temperature must be positive")
    T_e_J = T_e_eV * E_CHARGE
    return float(np.sqrt(EPSILON_0 * T_e_J / (n_e * E_CHARGE**2)))


def ND_parameter(n_e: float, T_e_eV: float) -> float:
    """Compute ND = number of particles in a Debye sphere.

    From Kindi et al. (2026). The ND parameter determines the
    validity of the MHD fluid approximation. MHD requires ND >> 1.

    Args:
        n_e: Electron density [m^-3].
        T_e_eV: Electron temperature [eV].

    Returns:
        ND (dimensionless). Typical DPF values:
        - Rundown phase: ND ~ 1e7 (strongly collisional)
        - Pinch: ND ~ 1e4-1e6 (collisional, MHD valid)
        - Disruption: ND ~ 1e3-1e5 (transition regime)
    """
    lam_D = debye_length(n_e, T_e_eV)
    return float((4 / 3) * np.pi * n_e * lam_D**3)


def coulomb_mean_free_path(
    T_e_eV: float,
    n_e: float,
    Z: float = 1.0,
    ln_lambda: float | None = None,
) -> float:
    """Compute electron-ion Coulomb mean free path.

    Uses the Daligault formula from Kindi et al. (2026) for
    the collisional mean free path.

    Args:
        T_e_eV: Electron temperature [eV].
        n_e: Electron density [m^-3].
        Z: Ion charge state (default 1 for D2).
        ln_lambda: Coulomb logarithm. If None, computed from NRL formula.

    Returns:
        Mean free path [m].
    """
    if T_e_eV <= 0 or n_e <= 0:
        raise ValueError("Temperature and density must be positive")

    if ln_lambda is None:
        # NRL formula for Te > 10 eV
        ln_lambda = max(23.5 - np.log(np.sqrt(n_e * 1e-6) / T_e_eV**(5/4))
                        - np.sqrt(1e-5 + (np.log(T_e_eV) - 2)**2 / 16), 2.0)

    T_e_J = T_e_eV * E_CHARGE
    # lambda_mfp = (4*pi*eps0)^2 * (m_e*v_th^2)^2 / (n_e * Z^2 * e^4 * ln_lambda)
    # Simplified: lambda_mfp ~ 7.4e-4 * T_e_eV^2 / (n_e * 1e-20 * Z^2 * ln_lambda) [m]
    # Using the standard NRL formula:
    m_e = 9.10938e-31
    v_th = np.sqrt(2 * T_e_J / m_e)
    tau_ei = 3 * np.sqrt(2 * np.pi) * EPSILON_0**2 * m_e**2 * v_th**3 / (
        n_e * Z**2 * E_CHARGE**4 * ln_lambda
    )
    return float(v_th * tau_ei)


def collisionality_regime(
    n_e: float,
    T_e_eV: float,
    L: float,
    Z: float = 1.0,
) -> str:
    """Classify the collisionality regime.

    Args:
        n_e: Electron density [m^-3].
        T_e_eV: Electron temperature [eV].
        L: Characteristic length scale [m] (e.g., pinch radius).
        Z: Ion charge state.

    Returns:
        One of: "collisional", "weakly_collisional", "collisionless".
        "collisional": MHD fully valid (lambda_mfp << L)
        "weakly_collisional": Extended MHD needed (lambda_mfp ~ L)
        "collisionless": Kinetic treatment needed (lambda_mfp >> L)
    """
    mfp = coulomb_mean_free_path(T_e_eV, n_e, Z)
    ratio = mfp / L

    if ratio < 0.01:
        return "collisional"
    elif ratio < 1.0:
        return "weakly_collisional"
    else:
        return "collisionless"


@dataclass
class CollisionalityDiagnostics:
    """Collisionality diagnostics for MHD validity assessment.

    Attributes:
        ND: Particles in Debye sphere.
        debye_length_m: Debye length [m].
        mfp_m: Coulomb mean free path [m].
        knudsen: Knudsen number (mfp / L).
        regime: Collisionality regime string.
        mhd_valid: Whether standard MHD is valid.
    """

    ND: float
    debye_length_m: float
    mfp_m: float
    knudsen: float
    regime: str
    mhd_valid: bool


def collisionality_diagnostics(
    n_e: float,
    T_e_eV: float,
    L: float,
    Z: float = 1.0,
) -> CollisionalityDiagnostics:
    """Compute comprehensive collisionality diagnostics.

    Args:
        n_e: Electron density [m^-3].
        T_e_eV: Electron temperature [eV].
        L: Characteristic length scale [m].
        Z: Ion charge state.

    Returns:
        :class:`CollisionalityDiagnostics`.
    """
    lam_D = debye_length(n_e, T_e_eV)
    nd = ND_parameter(n_e, T_e_eV)
    mfp = coulomb_mean_free_path(T_e_eV, n_e, Z)
    kn = mfp / L
    regime = collisionality_regime(n_e, T_e_eV, L, Z)

    return CollisionalityDiagnostics(
        ND=nd,
        debye_length_m=lam_D,
        mfp_m=mfp,
        knudsen=kn,
        regime=regime,
        mhd_valid=(kn < 0.01 and nd > 10),
    )


# =====================================================================
# DPF-Specific Diagnostics
# =====================================================================


def dpf_deceleration(
    I_MA: float,
    r_pinch_m: float,
    rho: float,
    dr: float,
) -> float:
    """Estimate effective gravitational acceleration during DPF compression.

    The magnetic pressure gradient acts as an effective gravity for
    Rayleigh-Taylor analysis:
        g_eff = (B_theta^2 / (2*mu_0)) / (rho * dr)

    where B_theta = mu_0 * I / (2*pi*r) is the azimuthal pinch field.

    Args:
        I_MA: Drive current [MA].
        r_pinch_m: Pinch radius [m].
        rho: Mass density [kg/m^3].
        dr: Sheath thickness [m].

    Returns:
        Effective deceleration [m/s^2].
    """
    I_A = I_MA * 1e6  # Convert to Amperes
    B_theta = MU_0 * I_A / (2 * np.pi * r_pinch_m)
    P_mag = B_theta**2 / (2 * MU_0)
    return P_mag / (rho * dr)


def alfven_velocity(B: float, rho: float) -> float:
    """Compute Alfven velocity.

    Args:
        B: Magnetic field [T].
        rho: Mass density [kg/m^3].

    Returns:
        Alfven velocity [m/s].
    """
    if rho <= 0:
        raise ValueError("Density must be positive")
    return float(B / np.sqrt(MU_0 * rho))


@dataclass
class DPFPinchDiagnostics:
    """Comprehensive diagnostics for a DPF pinch.

    Combines stagnation, mRT, collisionality, and yield diagnostics.

    Attributes:
        stagnation: Stagnation diagnostics.
        mrti: mRT diagnostics (if computed).
        collisionality: Collisionality diagnostics (if computed).
        Y_predicted: Predicted neutron yield from I^4.
        B_theta: Azimuthal magnetic field at pinch radius [T].
    """

    stagnation: StagnationDiagnostics
    mrti: MRTIDiagnostics | None = None
    collisionality: CollisionalityDiagnostics | None = None
    Y_predicted: float = 0.0
    B_theta: float = 0.0


def dpf_pinch_diagnostics(
    I_peak_MA: float,
    anode_radius_cm: float,
    fill_pressure_Torr: float,
    CR: float = 10.0,
    n_e_pinch: float | None = None,
    T_e_pinch_eV: float | None = None,
    pinch_lifetime_ns: float = 30.0,
    yield_coefficient: float = 9e10,
) -> DPFPinchDiagnostics:
    """Compute comprehensive pinch diagnostics for a DPF device.

    This is the top-level diagnostic function that combines all
    sub-diagnostics into a single assessment.

    Args:
        I_peak_MA: Peak current [MA].
        anode_radius_cm: Anode inner radius [cm].
        fill_pressure_Torr: Fill gas pressure [Torr].
        CR: Compression ratio (default 10).
        n_e_pinch: Electron density at pinch [m^-3]. If None,
            estimated from fill gas and compression ratio.
        T_e_pinch_eV: Electron temperature at pinch [eV]. If None,
            estimated from stagnation temperature.
        pinch_lifetime_ns: Pinch lifetime for e-folding calc [ns].
        yield_coefficient: I^4 scaling coefficient.

    Returns:
        :class:`DPFPinchDiagnostics` with all sub-diagnostics.
    """
    # Stagnation diagnostics
    stag = stagnation_diagnostics(I_peak_MA, anode_radius_cm, fill_pressure_Torr, CR)

    # Neutron yield
    Y_pred = neutron_yield_I4(I_peak_MA, yield_coefficient)

    # B_theta at pinch radius
    r_pinch_m = anode_radius_cm * 1e-2 / CR
    I_A = I_peak_MA * 1e6
    B_theta = MU_0 * I_A / (2 * np.pi * r_pinch_m)

    # Estimate pinch conditions if not provided
    if n_e_pinch is None:
        # n_fill from ideal gas at room temperature
        P_Pa = fill_pressure_Torr * 133.322
        n_fill = P_Pa / (K_B * 300)  # room temp fill
        # Molecular D2 → 2 atoms per molecule, fully ionized at pinch
        n_e_pinch = 2 * n_fill * CR**2  # Cylindrical compression

    if T_e_pinch_eV is None:
        T_e_pinch_eV = stag.T_stag * 1e3  # Convert keV to eV

    # mRT diagnostics (use pinch radius as wavelength scale)
    pinch_radius_m = anode_radius_cm * 1e-2 / CR
    wavelength = 2 * pinch_radius_m  # Dominant mode ~ diameter
    rho_h = n_e_pinch * M_DEUTERON  # Pinch density
    rho_l = (fill_pressure_Torr * 133.322) / (K_B * 300) * M_DEUTERON * 2  # Ambient
    g_eff = dpf_deceleration(I_peak_MA, pinch_radius_m, rho_h, pinch_radius_m * 0.1)

    mrti = mrti_diagnostics(
        g=g_eff,
        rho_h=rho_h,
        rho_l=rho_l,
        wavelength=wavelength,
        B=B_theta,
        theta=np.pi / 2,  # B_theta perp to z-axis perturbations
        pinch_lifetime_ns=pinch_lifetime_ns,
    )

    # Collisionality diagnostics
    coll = collisionality_diagnostics(
        n_e=n_e_pinch,
        T_e_eV=T_e_pinch_eV,
        L=pinch_radius_m,
    )

    return DPFPinchDiagnostics(
        stagnation=stag,
        mrti=mrti,
        collisionality=coll,
        Y_predicted=Y_pred,
        B_theta=B_theta,
    )
