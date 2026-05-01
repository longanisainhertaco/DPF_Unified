"""Physics diagnostics for DPF device characterisation.

Functions for computing L_p/L0 ratio, bare RLC timing, and Lee speed factor.
These are independent of simulation results and operate on device parameters.
"""

from __future__ import annotations

import numpy as np

from dpf.constants import mu_0
from dpf.validation.experimental_devices import DEVICES

# =====================================================================
# L_p / L0 diagnostic (Debate #29)
# =====================================================================


def compute_lp_l0_ratio(
    L0: float,
    anode_radius: float,
    cathode_radius: float,
    anode_length: float,
) -> dict[str, float]:
    """Compute the plasma-to-circuit inductance ratio L_p/L0.

    This diagnostic determines whether a DPF device's validation is
    informative (plasma-significant) or vacuously true (circuit-dominated).

    The axial plasma inductance at the end of the anode is::

        L_p = (mu_0 / 2pi) * ln(b/a) * z_max

    where *a* is anode radius, *b* is cathode radius, *z_max* is anode
    length.

    PhD Debate #29 classification:
        - L_p/L0 > 1.0: **Plasma-significant** — physics fundamentally
          alters the waveform.  Bare RLC gives large timing error.
        - L_p/L0 < 0.5: **Circuit-dominated** — bare damped RLC gives
          reasonable timing.  Validation is vacuously true.

    Parameters
    ----------
    L0 : float
        External (circuit) inductance [H].
    anode_radius : float
        Anode radius [m].
    cathode_radius : float
        Cathode radius [m].
    anode_length : float
        Anode length [m].

    Returns
    -------
    dict
        ``L_p_axial`` : float
            Axial plasma inductance at end of anode [H].
        ``L_p_over_L0`` : float
            Ratio L_p / L0 (dimensionless).
        ``regime`` : str
            "plasma-significant" if ratio > 1.0,
            "transitional" if 0.5 <= ratio <= 1.0,
            "circuit-dominated" if ratio < 0.5.
        ``L_per_length`` : float
            Inductance per unit length [H/m].

    References
    ----------
    PhD Debate #29 (2026-02-28): L_p/L0 diagnostic for validation
    informativeness.
    """
    L_per_length = (mu_0 / (2.0 * np.pi)) * np.log(cathode_radius / anode_radius)
    L_p_axial = L_per_length * anode_length
    ratio = L_p_axial / max(L0, 1e-15)

    if ratio > 1.0:
        regime = "plasma-significant"
    elif ratio >= 0.5:
        regime = "transitional"
    else:
        regime = "circuit-dominated"

    return {
        "L_p_axial": L_p_axial,
        "L_p_over_L0": ratio,
        "regime": regime,
        "L_per_length": L_per_length,
    }


def lp_l0_for_device(device_name: str) -> float:
    """Return L_p/L0 ratio for a named device.

    Convenience wrapper around :func:`compute_lp_l0_ratio`.

    Args:
        device_name: Key in ``DEVICES`` dict.

    Returns:
        L_p/L0 ratio (dimensionless).
    """
    dev = DEVICES[device_name]
    result = compute_lp_l0_ratio(
        dev.inductance, dev.anode_radius, dev.cathode_radius, dev.anode_length,
    )
    return result["L_p_over_L0"]


def compute_bare_rlc_timing(
    C: float,
    L0: float,
    R0: float,
) -> float:
    """Compute the quarter-period of a bare damped RLC circuit.

    For a lossless RLC, the quarter-period is T/4 = pi * sqrt(L0 * C).
    With damping, the underdamped period is
    T = 2*pi / sqrt(1/(L0*C) - (R0/(2*L0))^2) and T/4 is one quarter.

    Parameters
    ----------
    C : float
        Capacitance [F].
    L0 : float
        External inductance [H].
    R0 : float
        External resistance [Ohm].

    Returns
    -------
    float
        Quarter-period [s].
    """
    omega_0_sq = 1.0 / (L0 * C)
    gamma_sq = (R0 / (2.0 * L0)) ** 2
    if omega_0_sq <= gamma_sq:
        # Overdamped — no oscillation, return RC timescale
        return np.pi * np.sqrt(L0 * C)
    omega_d = np.sqrt(omega_0_sq - gamma_sq)
    return np.pi / (2.0 * omega_d)


# =====================================================================
# Speed factor diagnostic (Debate #36)
# =====================================================================

# Typical speed factor for most deuterium Mather-type DPFs (incl. PF-1000):
# S ~ 89 kA/(cm * sqrt(Torr)).  This is the TYPICAL value reported by
# Lee & Saw (2008) for the catalogued machines (PF400, UNU/ICTP, NX2,
# DPF78, PF1000) -- it is NOT an "optimum" in the sense of maximising
# neutron yield.  Lee & Saw 2008, §"Results" (around Table 1), states
# explicitly: "All devices except Poseidon have typical S values.
# Poseidon is the exceptional high speed device in this respect."  The
# paper does not claim 89 is optimal; it is simply the population mean
# for well-characterised Mather-geometry DPFs.
# Kept as a reference point for the speed-factor diagnostic regime
# classification, with the ratio S/S_typical interpreted as
# "how close is this device to the standard PF-1000-class regime?"
_S_TYPICAL_PF1000 = 89.0

# Deprecated alias (retained for backward compatibility with external
# callers); new code should use _S_TYPICAL_PF1000.  The name
# _S_OPTIMAL_KA_CM_TORR was misleading -- see comment above.
_S_OPTIMAL_KA_CM_TORR = _S_TYPICAL_PF1000


def compute_speed_factor(
    peak_current: float,
    anode_radius: float,
    fill_pressure_torr: float,
) -> dict[str, float]:
    """Compute the Lee speed factor S = I_peak / (a * sqrt(p)).

    The speed factor is a dimensionless scaling parameter that
    characterizes the drive condition of a DPF device.  Lee & Saw
    (2008) report that most catalogued Mather-geometry deuterium DPFs
    (PF400, UNU/ICTP, NX2, DPF78, PF1000) cluster around a typical
    value S_typical ~ 89 kA/(cm * sqrt(Torr)); Poseidon is called out
    as the one device significantly above this value.  The 89
    reference is a TYPICAL population value, NOT a tabulated neutron-
    yield optimum.

    Classification (PhD Debate #36):

    - S/S_typical ~ 0.8-1.2: **PF1000-class** — thin-sheath snowplow
      valid, Lee model fc/fm most transferable.
    - S/S_typical < 0.8: **Sub-driven** — slow sheath, thick and
      diffuse, under-compressed pinch.
    - S/S_typical > 1.2: **Super-driven** — sheath outruns fill gas,
      snowplow approximation breaks down, fc/fm become strongly
      device-dependent (e.g. Poseidon).

    Parameters
    ----------
    peak_current : float
        Peak discharge current [A].
    anode_radius : float
        Anode radius [m].
    fill_pressure_torr : float
        Fill gas pressure [Torr].

    Returns
    -------
    dict
        ``S`` : float
            Speed factor [kA / (cm * sqrt(Torr))].
        ``S_over_S_typical`` : float
            Ratio S / S_typical (dimensionless).  Primary key.
        ``S_over_S_opt`` : float
            Legacy alias identical to ``S_over_S_typical``, retained
            for backward compatibility; prefer ``S_over_S_typical`` in
            new code.
        ``regime`` : str
            "PF1000-class", "sub-driven", or "super-driven".
            The legacy string "optimal" is mapped to "PF1000-class".

    References
    ----------
    S. Lee & S. H. Saw, J. Fusion Energy 27:292-295 (2008).
      Paper on disk:
      references/papers/nuclear-radiation/
      PP2 with Erratum JoFE NeutronScalingLawsFromNumericalExperiments.pdf
      Table 1 and surrounding discussion.
    S. Lee, J. Fusion Energy 33:319-335 (2014).
    """
    # Convert to kA/(cm * sqrt(Torr))
    I_kA = peak_current / 1e3
    a_cm = anode_radius * 100.0
    p_torr = max(fill_pressure_torr, 1e-10)

    S = I_kA / (a_cm * np.sqrt(p_torr))
    S_ratio = S / _S_TYPICAL_PF1000

    if 0.8 <= S_ratio <= 1.2:
        regime = "PF1000-class"
    elif S_ratio < 0.8:
        regime = "sub-driven"
    else:
        regime = "super-driven"

    return {
        "S": S,
        "S_over_S_typical": S_ratio,
        "S_over_S_opt": S_ratio,  # deprecated alias, retained for compatibility
        "regime": regime,
    }
