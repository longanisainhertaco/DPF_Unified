"""Pease-Braginskii current for Z-pinch radiative collapse assessment.

The Pease-Braginskii current I_PB is the threshold current above which
bremsstrahlung radiation losses exceed Ohmic heating in a Bennett-equilibrium
Z-pinch.  When I > I_PB, the pinch undergoes radiative collapse.

Physics (Haines 2011, Sec. 3.3):
    In Bennett equilibrium, the balance between Ohmic heating (from Spitzer
    resistivity) and bremsstrahlung cooling leads to a critical current that
    depends only on Z, the Gaunt factor, and the Coulomb logarithm — all
    other parameters (T, n, a) cancel due to the self-similar Bennett profile.

    Haines (2011) Eq. on p. 51:
        I_PB = 0.433 * sqrt(6 / (Z*(1+Z)*g_ff)) MA     [for ln_Lambda=10]

    With explicit ln_Lambda dependence:
        I_PB = 0.433 * sqrt(6 * ln_Lambda / (10 * Z*(1+Z)*g_ff)) MA

    For deuterium (Z=1, g_ff=1.2, ln_Lambda=10):
        I_PB ~ 0.68 MA

    For the common approximation (Z=1, g_ff=1, ln_Lambda=10):
        I_PB ~ 0.75 MA

    The frequently-cited ~1.4 MA corresponds to ln_Lambda ~ 35, which
    can occur at DPF pinch conditions (Te ~ 5-10 keV, ne ~ 10^24 m^-3).

Significance for DPF:
    PF-1000 operates at I_peak ~ 1.87 MA.  Even with generous ln_Lambda ~ 20,
    I_PB ~ 1.0 MA < I_peak, confirming the radiative collapse regime.
    Beam-target neutron production dominates over thermonuclear in this regime.

References:
    Pease, R.S., Proc. Phys. Soc. 70, 11 (1957).
    Braginskii, S.I., Sov. Phys. JETP 6, 494 (1958).
    Haines, M.G., Plasma Phys. Control. Fusion 53, 093001 (2011).
"""

from __future__ import annotations

import math
from typing import Any

# Haines (2011) numerical constant: I_PB = K_HAINES * sqrt(6*lnL/(10*Z*(1+Z)*g_ff))
# where K_HAINES = 0.433 MA = 433,000 A.
# Derived from balancing bremsstrahlung cooling against Spitzer Ohmic heating
# in Bennett equilibrium, with uniform density profile.
_K_HAINES_A = 433_000.0  # [A] from Haines (2011) p. 51


def pease_braginskii_current(
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
    ln_Lambda: float = 10.0,
) -> dict[str, Any]:
    """Compute the Pease-Braginskii current for a Z-pinch.

    Uses the Haines (2011) formula with explicit ln_Lambda scaling:

        I_PB = 0.433 MA * sqrt(6 * ln_Lambda / (10 * Z * (1+Z) * g_ff))

    Args:
        Z: Ion charge state (default 1 for deuterium).
        gaunt_factor: Gaunt factor g_ff (default 1.2 for DPF conditions).
        ln_Lambda: Coulomb logarithm (default 10).

    Returns:
        Dictionary with:
            I_PB: Pease-Braginskii current [A].
            I_PB_MA: Same in megaamperes.
            Z: Ion charge state used.
            gaunt_factor: Gaunt factor used.
            ln_Lambda: Coulomb logarithm used.
    """
    Z = max(float(Z), 0.1)
    gaunt_factor = max(float(gaunt_factor), 0.1)
    ln_Lambda = max(float(ln_Lambda), 1.0)

    # Haines (2011) p. 51:
    #   I_PB = 0.433 * sqrt(6/(Z*(1+Z)*g_ff)) MA  [at ln_Lambda=10]
    #
    # The ln_Lambda dependence enters through the Spitzer resistivity:
    #   eta ~ ln_Lambda / T^(3/2)
    # Higher ln_Lambda means more Ohmic heating, requiring higher I to
    # reach radiative collapse.  I_PB scales as sqrt(ln_Lambda).
    arg = 6.0 * ln_Lambda / (10.0 * Z * (1.0 + Z) * gaunt_factor)
    I_PB = _K_HAINES_A * math.sqrt(arg)

    return {
        "I_PB": I_PB,
        "I_PB_MA": I_PB * 1e-6,
        "Z": Z,
        "gaunt_factor": gaunt_factor,
        "ln_Lambda": ln_Lambda,
    }


def check_pease_braginskii(
    I_current: float,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
    ln_Lambda: float = 10.0,
) -> dict[str, Any]:
    """Check whether the current exceeds the Pease-Braginskii limit.

    Args:
        I_current: Discharge current [A].
        Z: Ion charge state.
        gaunt_factor: Gaunt factor g_ff.
        ln_Lambda: Coulomb logarithm.

    Returns:
        Dictionary with:
            I_current: Input current [A].
            I_PB: Pease-Braginskii current [A].
            ratio: I_current / I_PB (>1 means radiative collapse regime).
            exceeds_PB: True if I_current > I_PB.
            regime: "stable" if I < I_PB, "radiative_collapse" if I > I_PB.
    """
    pb = pease_braginskii_current(Z=Z, gaunt_factor=gaunt_factor, ln_Lambda=ln_Lambda)
    I_PB = pb["I_PB"]

    ratio = abs(float(I_current)) / max(I_PB, 1e-30)

    return {
        "I_current": float(I_current),
        "I_current_MA": float(I_current) * 1e-6,
        "I_PB": I_PB,
        "I_PB_MA": I_PB * 1e-6,
        "ratio": ratio,
        "exceeds_PB": ratio > 1.0,
        "regime": "radiative_collapse" if ratio > 1.0 else "stable",
        **{k: v for k, v in pb.items() if k not in ("I_PB", "I_PB_MA")},
    }
