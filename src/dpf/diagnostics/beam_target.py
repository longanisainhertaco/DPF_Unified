"""Beam-target neutron yield model for Dense Plasma Focus.

Implements the Lee model beam-target mechanism: when the m=0 instability
disrupts the pinch column, a fraction of the pinch current is converted
into a fast deuteron beam that traverses the dense plasma target.  The
beam-target yield typically dominates over thermonuclear yield in DPF
devices operating below ~1 MJ stored energy.

Physics:
    1. Pinch disruption is detected by a sudden pressure/density spike
       (m=0 sausage instability).
    2. A fraction f_beam of the pinch current becomes a deuteron beam
       with energy E_beam ~ e * V_pinch (pinch voltage from circuit).
    3. The beam traverses the pinch column of length L_target through
       a deuterium target of density n_target.
    4. Neutron yield rate:
           dY/dt = f_beam * (I_pinch / e) * n_target * sigma_DD(E_beam) * L_target

DD cross section uses the Bosch-Hale (1992) parametric fit:
    sigma(E) = S(E) / (E * exp(B_G / sqrt(E)))

where S(E) is the astrophysical S-factor with a 5th-order rational
polynomial fit valid for 0.5 keV < E < 5000 keV.

Neutron anisotropy distinguishes beam-target (forward-peaked) from
thermonuclear (isotropic) contributions -- an important experimental
diagnostic signature.

References:
    Bosch & Hale, Nuclear Fusion 32:611 (1992)
    Lee, S., J Fusion Energy 33:319 (2014)
    NRL Plasma Formulary (2019)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dpf.constants import e as e_charge
from dpf.constants import eV

# ---------------------------------------------------------------------------
# Bosch-Hale DD fusion cross section (D(d,n)He3 branch)
# ---------------------------------------------------------------------------

# Gamow constant for DD [keV^{1/2}]
_BG = 31.3970

# Reduced mass * c^2 for DD [keV]
_MU_C2 = 937814.0

# Fit coefficients for astrophysical S-factor rational polynomial
# Table IV of Bosch & Hale (1992), D(d,n)He-3 branch
# S(E) in units of [keV * millibarn]; sigma comes out in millibarns
_A1 = 5.3701e4
_A2 = 3.3027e2
_A3 = -1.2706e-1
_A4 = 2.9327e-5
_A5 = -2.5151e-9

_B1 = 0.0
_B2 = 0.0
_B3 = 0.0
_B4 = 0.0

# 1 millibarn = 1e-3 barn = 1e-31 m^2
_MBARN_TO_M2 = 1.0e-31


@njit(cache=True)
def dd_cross_section(E_keV: float) -> float:
    """DD fusion cross section sigma(E) for D(d,n)He-3 [m^2].

    Uses the Bosch-Hale (1992) parametric fit:
        sigma(E) = S(E) / (E * exp(B_G / sqrt(E)))

    where S(E) is a rational polynomial (astrophysical S-factor) and
    E is the centre-of-mass energy.

    The Bosch-Hale Table IV coefficients give S in keV*millibarn, so
    sigma comes out in millibarns.

    This returns the cross section for the neutron-producing branch only:
        D + D -> He-3 + n  (2.45 MeV neutron)

    Valid range: 0.5 keV < E_cm < 5000 keV.  Returns 0 outside this range.

    For beam-target calculations (beam on stationary target), convert the
    lab-frame beam energy to CM energy first: E_cm = E_lab / 2 (equal
    mass DD system).

    Reference values (D(d,n)He-3, CM frame):
        E_cm =  10 keV: sigma ~ 0.28  mbarn  (2.8e-31 m^2)
        E_cm =  50 keV: sigma ~  16   mbarn  (1.6e-29 m^2)
        E_cm = 100 keV: sigma ~  37   mbarn  (3.7e-29 m^2)
        E_cm = 200 keV: sigma ~  62   mbarn  (6.2e-29 m^2)

    Args:
        E_keV: Centre-of-mass energy [keV].  For beam-target use,
            pass E_lab / 2.

    Returns:
        Fusion cross section [m^2].
    """
    if E_keV < 0.5 or E_keV > 5000.0:
        return 0.0

    # Astrophysical S-factor [keV * millibarn]
    # S(E) = (A1 + E*(A2 + E*(A3 + E*(A4 + E*A5))))
    #       / (1 + E*(B1 + E*(B2 + E*(B3 + E*B4))))
    S_numer = _A1 + E_keV * (_A2 + E_keV * (_A3 + E_keV * (_A4 + E_keV * _A5)))
    S_denom = 1.0 + E_keV * (_B1 + E_keV * (_B2 + E_keV * (_B3 + E_keV * _B4)))

    if abs(S_denom) < 1e-30:
        return 0.0

    S = S_numer / S_denom  # [keV * millibarn]

    # sigma(E) = S(E) / (E * exp(B_G / sqrt(E)))
    exponent = _BG / np.sqrt(E_keV)

    # Guard against overflow: exp(x) overflows for x > ~700
    if exponent > 700.0:
        return 0.0

    sigma_mbarn = S / (E_keV * np.exp(exponent))  # [millibarn]

    return max(sigma_mbarn * _MBARN_TO_M2, 0.0)


@njit(cache=True)
def dd_cross_section_array(E_keV: np.ndarray) -> np.ndarray:
    """Vectorized DD fusion cross section for energy arrays.

    Args:
        E_keV: Deuteron energies [keV], arbitrary shape.

    Returns:
        Cross section array [m^2], same shape as input.
    """
    result = np.empty_like(E_keV)
    for i in range(E_keV.size):
        result.flat[i] = dd_cross_section(E_keV.flat[i])
    return result


# ---------------------------------------------------------------------------
# Beam-target neutron yield rate
# ---------------------------------------------------------------------------


@njit(cache=True)
def beam_target_yield_rate(
    I_pinch: float,
    V_pinch: float,
    n_target: float,
    L_target: float,
    f_beam: float = 0.2,
) -> float:
    """Beam-target DD neutron production rate [1/s].

    When the m=0 instability disrupts the pinch, a fraction f_beam of the
    pinch current is converted into a fast deuteron beam with energy
    E_beam = e * V_pinch.  This beam traverses the target plasma column.

    The yield rate is:
        dY/dt = f_beam * (I_pinch / e) * n_target * sigma_DD(E_beam) * L_target

    where:
        f_beam * (I_pinch / e)  = deuteron beam flux [1/s]
        n_target * sigma_DD * L_target = reaction probability per beam ion

    Typical DPF values: I_pinch ~ 100-500 kA, V_pinch ~ 20-200 kV,
    n_target ~ 10^24-10^26 m^-3, L_target ~ 5-20 mm, f_beam ~ 0.1-0.3.

    Args:
        I_pinch: Pinch current [A].
        V_pinch: Pinch voltage [V].  Determines beam energy via E_beam = e*V.
        n_target: Target deuterium number density [m^-3].
        L_target: Effective target length (pinch column) [m].
        f_beam: Fraction of pinch current converted to beam (default 0.2).
                 Typical range: 0.1 to 0.3 (Lee model).

    Returns:
        Neutron production rate dY/dt [1/s].
        Returns 0.0 if any input is non-positive.
    """
    if I_pinch <= 0.0 or V_pinch <= 0.0 or n_target <= 0.0 or L_target <= 0.0:
        return 0.0

    # Clamp f_beam to physical range
    fb = max(min(f_beam, 1.0), 0.0)

    # Lab-frame beam energy in keV: E_lab = e * V_pinch [J] -> keV
    E_lab_keV = V_pinch * e_charge / (1.0e3 * eV)
    # Simplifies to V_pinch / 1000.0 since e_charge / eV = 1, but keep
    # explicit for clarity and unit safety.

    # Convert to centre-of-mass energy for equal-mass DD system:
    # E_cm = E_lab * m_target / (m_beam + m_target) = E_lab / 2
    E_cm_keV = E_lab_keV / 2.0

    # DD cross section at CM energy
    sigma = dd_cross_section(E_cm_keV)

    # Beam deuteron flux: f_beam * I_pinch / e  [deuterons/s]
    beam_flux = fb * I_pinch / e_charge

    # Yield rate: beam_flux * n_target * sigma * L_target
    dY_dt = beam_flux * n_target * sigma * L_target

    return dY_dt


# ---------------------------------------------------------------------------
# Pinch disruption detector
# ---------------------------------------------------------------------------


def detect_pinch_disruption(
    pressure_history: np.ndarray | list[float],
    threshold_ratio: float = 5.0,
) -> bool:
    """Detect m=0 pinch disruption from pressure history.

    The m=0 sausage instability causes a sudden spike in plasma pressure
    (and density) when the pinch column disrupts.  This function detects
    such a spike by comparing the most recent pressure value to a baseline
    computed from the earlier history.

    The baseline is the median of the first half of the history (robust to
    outliers).  Disruption is flagged when the latest value exceeds the
    baseline by the threshold ratio.

    Args:
        pressure_history: Array or list of peak pressure values [Pa] over
            recent timesteps.  Needs at least 4 entries for a meaningful
            baseline.
        threshold_ratio: Ratio of current pressure to baseline that triggers
            disruption detection (default 5.0).

    Returns:
        True if a disruption (pressure spike) is detected.
    """
    p = np.asarray(pressure_history, dtype=np.float64)

    if p.size < 4:
        return False

    # Baseline: median of first half of history (robust estimator)
    half = p.size // 2
    baseline = np.median(p[:half])

    if baseline <= 0.0:
        return False

    # Current pressure: last entry
    current = p[-1]

    return bool(current / baseline >= threshold_ratio)


# ---------------------------------------------------------------------------
# Neutron anisotropy
# ---------------------------------------------------------------------------


@njit(cache=True)
def neutron_anisotropy(
    Y_beam: float,
    Y_thermal: float,
    E_beam_keV: float = 100.0,
) -> float:
    """Compute forward/sideways neutron anisotropy ratio Y(0 deg)/Y(90 deg).

    In DPF devices, beam-target neutrons are emitted preferentially along
    the beam direction (forward-peaked), while thermonuclear neutrons are
    isotropic.  The measured anisotropy is a weighted average and serves
    as a diagnostic for the dominant yield mechanism.

    Beam-target anisotropy model:
        For DD at typical DPF beam energies (50-500 keV), the CM-frame
        angular distribution is nearly isotropic, but the lab-frame
        kinematics (beam into stationary target) produce forward peaking.
        The anisotropy ratio scales approximately as:
            A_bt ~ 1 + alpha * sqrt(E_beam / E_ref)
        where alpha ~ 0.3 and E_ref = 100 keV, giving A_bt ~ 1.3 to 2.5
        for typical beam energies.

    The total anisotropy is the yield-weighted average:
        A_total = (Y_beam * A_bt + Y_thermal * 1.0) / (Y_beam + Y_thermal)

    Args:
        Y_beam: Beam-target neutron yield (count or rate).
        Y_thermal: Thermonuclear neutron yield (count or rate).
        E_beam_keV: Beam deuteron energy [keV] (default 100 keV).

    Returns:
        Anisotropy ratio Y(0 deg) / Y(90 deg).
        Returns 1.0 if total yield is zero (isotropic by convention).
    """
    Y_total = Y_beam + Y_thermal
    if Y_total <= 0.0:
        return 1.0

    # Beam-target anisotropy from lab-frame kinematics
    # A_bt ~ 1 + 0.3 * sqrt(E_beam / 100 keV)
    # Clamped to physical range [1.0, 4.0]
    E_safe = max(E_beam_keV, 0.0)
    A_bt = 1.0 + 0.3 * np.sqrt(E_safe / 100.0)
    A_bt = min(max(A_bt, 1.0), 4.0)

    # Thermonuclear contribution is isotropic
    A_th = 1.0

    # Yield-weighted average
    A_total = (Y_beam * A_bt + Y_thermal * A_th) / Y_total

    return A_total
