"""Plasma regime classifier for DPF simulations.

Evaluates dimensionless parameters to determine which physics
regime the plasma is in and which simulation backend is appropriate.

Key parameters:
    - Lundquist number S = tau_R / tau_A (resistive vs Alfven)
    - Magnetic Reynolds Rm = v*L/eta (advection vs diffusion)
    - Beta = p_thermal / p_magnetic
    - Knudsen Kn = lambda_mfp / L (continuum vs kinetic)
    - Hall parameter omega_ci * tau_i (magnetized vs unmagnetized)

References:
    Haines M.G., PPCF 53:093001 (2011) — DPF plasma parameters.
    NRL Plasma Formulary (2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants
MU_0 = 4.0 * math.pi * 1e-7
K_B = 1.380649e-23
E_CHARGE = 1.602e-19
M_E = 9.109e-31
M_P = 1.673e-27
EPSILON_0 = 8.854e-12


@dataclass
class RegimeResult:
    """Plasma regime classification."""

    # Dimensionless parameters
    lundquist_S: float      # tau_R / tau_A
    magnetic_reynolds: float # v*L/eta
    beta: float             # p_thermal / p_magnetic
    knudsen: float          # lambda_mfp / L
    hall_parameter: float   # omega_ci * tau_i
    debye_cells: float      # L / lambda_D (cells per Debye length)

    # Regime classification
    mhd_valid: bool         # S >> 1 and Kn << 1
    resistive_important: bool  # Rm ~ 1
    kinetic_needed: bool    # Kn > 0.01
    hall_important: bool    # Hall param < 10
    ideal_mhd: bool         # S >> 1 and Rm >> 1 and beta ~ 1

    # Recommended backend
    recommended_backend: str
    regime_summary: str


def classify_regime(
    n_e: float,
    T_e_eV: float,
    B_T: float,
    L_m: float,
    v_m_s: float = 1e5,
    Z: float = 1.0,
    ion_mass_kg: float = 3.34e-27,
) -> RegimeResult:
    """Classify the plasma regime from local parameters.

    Args:
        n_e: Electron density [m^-3].
        T_e_eV: Electron temperature [eV].
        B_T: Magnetic field magnitude [T].
        L_m: Characteristic length scale [m] (e.g., pinch radius).
        v_m_s: Characteristic velocity [m/s] (e.g., Alfven speed).
        Z: Ion charge state.
        ion_mass_kg: Ion mass [kg].

    Returns:
        RegimeResult with all parameters and classification.
    """
    T_K = T_e_eV * E_CHARGE / K_B
    T_J = T_e_eV * E_CHARGE

    # Debye length
    if n_e > 0:
        lambda_D = math.sqrt(EPSILON_0 * K_B * T_K / (n_e * E_CHARGE**2))
    else:
        lambda_D = 1.0

    # Coulomb logarithm
    if n_e > 0 and T_e_eV > 0:
        ln_Lambda = max(1.0, 23.0 - 0.5 * math.log(n_e / 1e6) + 1.5 * math.log(T_e_eV))
    else:
        ln_Lambda = 10.0

    # Spitzer resistivity: eta = 0.51 * m_e * nu_ei / (n_e * e^2)
    # nu_ei = n_e * e^4 * ln_Lambda / (3 * (2*pi)^1.5 * epsilon_0^2 * m_e^0.5 * (k_B*T)^1.5)
    if T_K > 0 and n_e > 0:
        nu_ei = (n_e * E_CHARGE**4 * ln_Lambda /
                 (3.0 * (2 * math.pi)**1.5 * EPSILON_0**2 *
                  M_E**0.5 * (K_B * T_K)**1.5))
        eta = 0.51 * M_E * nu_ei / (n_e * E_CHARGE**2)
    else:
        eta = 1e-4

    # Alfven speed
    rho = n_e * ion_mass_kg / Z
    if rho > 0 and B_T > 0:
        v_A = B_T / math.sqrt(MU_0 * rho)
    else:
        v_A = v_m_s

    # Alfven transit time
    tau_A = L_m / max(v_A, 1.0)

    # Resistive diffusion time
    tau_R = MU_0 * L_m**2 / max(eta, 1e-30)

    # Lundquist number
    S = tau_R / max(tau_A, 1e-30)

    # Magnetic Reynolds number
    Rm = v_m_s * L_m * MU_0 / max(eta, 1e-30)

    # Plasma beta
    p_thermal = n_e * K_B * T_K * (1.0 + 1.0 / Z)  # electron + ion
    p_magnetic = B_T**2 / (2.0 * MU_0) if B_T > 0 else 1.0
    beta = p_thermal / max(p_magnetic, 1e-30)

    # Ion mean free path (Coulomb collisions)
    if n_e > 0 and T_e_eV > 0:
        # lambda_ii ~ v_ti * tau_ii
        v_ti = math.sqrt(2.0 * T_J / ion_mass_kg)
        # tau_ii ~ 12 * pi^1.5 * epsilon_0^2 * m_i^0.5 * (kT)^1.5 / (n_i * Z^4 * e^4 * ln_Lambda)
        n_i = n_e / Z
        tau_ii = (12.0 * math.pi**1.5 * EPSILON_0**2 * ion_mass_kg**0.5 *
                  (K_B * T_K)**1.5 / (max(n_i, 1.0) * Z**4 * E_CHARGE**4 * ln_Lambda))
        lambda_mfp = v_ti * tau_ii
    else:
        lambda_mfp = 1.0
        tau_ii = 1.0

    # Knudsen number
    Kn = lambda_mfp / max(L_m, 1e-10)

    # Hall parameter: omega_ci * tau_i
    omega_ci = Z * E_CHARGE * B_T / ion_mass_kg if B_T > 0 else 0.0
    hall_param = omega_ci * tau_ii

    # Debye cells
    debye_cells = L_m / max(lambda_D, 1e-30)

    # --- Classification ---
    mhd_valid = S > 100 and Kn < 0.01
    resistive_important = Rm < 100
    kinetic_needed = Kn > 0.01
    hall_important = hall_param < 10
    ideal_mhd = S > 1000 and Rm > 100 and 0.01 < beta < 100

    # Backend recommendation
    if kinetic_needed:
        backend = "pic_hybrid"
        summary = "Kinetic regime (Kn > 0.01) — PIC or hybrid PIC-MHD needed"
    elif hall_important and resistive_important:
        backend = "hall_mhd"
        summary = "Hall-resistive MHD — Hall term and resistivity both important"
    elif resistive_important:
        backend = "resistive_mhd"
        summary = "Resistive MHD — magnetic diffusion matters (Rm ~ 1)"
    elif ideal_mhd:
        backend = "ideal_mhd"
        summary = "Ideal MHD regime — fast, accurate with PLM+HLL"
    else:
        backend = "resistive_mhd"
        summary = "General MHD — use Metal solver with WENO5+HLLD"

    return RegimeResult(
        lundquist_S=S,
        magnetic_reynolds=Rm,
        beta=beta,
        knudsen=Kn,
        hall_parameter=hall_param,
        debye_cells=debye_cells,
        mhd_valid=mhd_valid,
        resistive_important=resistive_important,
        kinetic_needed=kinetic_needed,
        hall_important=hall_important,
        ideal_mhd=ideal_mhd,
        recommended_backend=backend,
        regime_summary=summary,
    )
