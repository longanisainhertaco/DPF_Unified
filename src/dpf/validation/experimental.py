"""Multi-device experimental validation data for DPF simulations.

Provides published experimental parameters and measured observables for
well-characterised Dense Plasma Focus devices.  These can be used to:

    1. Configure a simulation to match a real device geometry and
       electrical parameters.
    2. Validate simulated current waveforms and neutron yields against
       published measurements.

Devices included:
    - **PF-1000** (IPPLM Warsaw, Poland) -- the largest DPF in Europe.
    - **NX2** (NIE Singapore) -- compact Mather-type DPF.
    - **UNU-ICTP PFF** -- the widely-replicated training device.

Usage::

    from dpf.validation.experimental import (
        PF1000_DATA, NX2_DATA, UNU_ICTP_DATA,
        validate_current_waveform,
        validate_neutron_yield,
        device_to_config_dict,
    )

    metrics = validate_current_waveform(t_sim, I_sim, "PF-1000")
    print(f"Peak current error: {metrics['peak_current_error']:.1%}")

Units: SI throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dpf.constants import k_B, m_d, m_D2

# =====================================================================
# Experimental device dataclass
# =====================================================================

@dataclass
class ExperimentalDevice:
    """Published experimental data for a Dense Plasma Focus device.

    Attributes
    ----------
    name : str
        Device name.
    institution : str
        Host institution / laboratory.
    capacitance : float
        Bank capacitance [F].
    voltage : float
        Charging voltage [V].
    inductance : float
        External (stray) inductance [H].
    resistance : float
        External (stray) resistance [Ohm].
    anode_radius : float
        Anode radius [m].
    cathode_radius : float
        Cathode radius [m].
    anode_length : float
        Anode length [m].
    fill_pressure_torr : float
        Fill gas pressure [Torr].
    fill_gas : str
        Fill gas species (e.g. ``"deuterium"``).
    peak_current : float
        Measured peak discharge current [A].
    neutron_yield : float
        Measured total DD neutron yield per shot.
    current_rise_time : float
        Measured current quarter-period (time to first peak) [s].
    reference : str
        Publication reference string.
    """

    name: str
    institution: str
    capacitance: float        # [F]
    voltage: float            # [V]
    inductance: float         # [H]
    resistance: float         # [Ohm]
    anode_radius: float       # [m]
    cathode_radius: float     # [m]
    anode_length: float       # [m]
    fill_pressure_torr: float
    fill_gas: str
    peak_current: float       # [A]
    neutron_yield: float
    current_rise_time: float  # [s]
    reference: str
    # Experimental uncertainties (1-sigma, relative)
    # Following GUM (JCGM 100:2008) and ASME V&V 20-2009 uncertainty framework.
    peak_current_uncertainty: float = 0.0   # Relative uncertainty on peak current
    rise_time_uncertainty: float = 0.0      # Relative uncertainty on rise time
    neutron_yield_uncertainty: float = 0.0  # Relative uncertainty on neutron yield
    # Digitized waveform data (optional)
    waveform_t: np.ndarray | None = None    # Time array [s]
    waveform_I: np.ndarray | None = None    # Current array [A]
    # Waveform digitization uncertainty (1-sigma, relative)
    waveform_digitization_uncertainty: float = 0.0  # Amplitude digitization error
    waveform_time_uncertainty: float = 0.0          # Temporal digitization error
    # Measurement provenance note
    measurement_notes: str = ""


# =====================================================================
# Device data
# =====================================================================

# PF-1000 digitized I(t) from Scholz et al., Nukleonika 51(1), 2006, Fig. 2
# 26 points covering 0-10 us, interpolated from published waveform
# Characteristic features: rise to ~1.87 MA at ~5.8 us, current dip at ~7 us
_PF1000_WAVEFORM_T_US = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.3, 5.6, 5.8, 6.0, 6.3, 6.5, 6.8, 7.0, 7.3,
    7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
])
_PF1000_WAVEFORM_I_MA = np.array([
    0.00, 0.15, 0.35, 0.58, 0.82, 1.05, 1.25, 1.42, 1.56, 1.67,
    1.76, 1.81, 1.85, 1.87, 1.86, 1.82, 1.75, 1.55, 1.40, 1.30,
    1.25, 1.15, 1.05, 0.95, 0.85, 0.75,
])

PF1000_DATA = ExperimentalDevice(
    name="PF-1000",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # 1.332 mF
    voltage=27e3,                  # 27 kV
    inductance=33.5e-9,            # 33.5 nH (short-circuit calibration, Scholz 2006)
    resistance=2.3e-3,             # 2.3 mOhm (short-circuit discharge, Scholz 2006 Table 1)
    anode_radius=0.115,            # 115 mm outer radius (IPPLM: anode OD 230mm)
    cathode_radius=0.16,           # 160 mm effective (Lee & Saw 2014; rods at 200mm)
    anode_length=0.60,             # 600 mm (IPPLM: anode length 60 cm)
    fill_pressure_torr=3.5,
    fill_gas="deuterium",
    peak_current=1.87e6,           # 1.87 MA
    neutron_yield=1e11,
    current_rise_time=5.8e-6,      # 5.8 us
    reference="Scholz et al., Nukleonika 51(1), 2006",
    peak_current_uncertainty=0.05,     # 5% (Rogowski coil + calibration)
    rise_time_uncertainty=0.10,        # 10% (quarter-period timing)
    neutron_yield_uncertainty=0.50,    # 50% (shot-to-shot variability)
    waveform_t=_PF1000_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_PF1000_WAVEFORM_I_MA * 1e6,        # Convert MA -> A
    # Digitization uncertainty for hand-digitized Fig. 2 of Scholz et al. (2006).
    # Amplitude: ±3% (trace width ~0.06 MA on ~2 MA full scale).
    # Time: ±0.5% of full scale (~0.05 us on 10 us trace).
    # Combined current uncertainty: sqrt(5%^2 + 3%^2) = 5.8% (1-sigma).
    waveform_digitization_uncertainty=0.03,  # 3% amplitude from trace reading
    waveform_time_uncertainty=0.005,         # 0.5% of full scale (~0.05 us)
    measurement_notes=(
        "26 points hand-digitized from Scholz et al., Nukleonika 51(1), 2006, Fig. 2. "
        "Rogowski coil uncertainty ~5% (Type B, estimated — not stated in source). "
        "Digitization amplitude uncertainty ~3% (Type B, trace width / full scale). "
        "Combined waveform uncertainty: u_I = sqrt(0.05^2 + 0.03^2) = 5.8% (1-sigma). "
        "Temporal uncertainty ~0.05 us (Type B, 0.5% of 10 us trace). "
        "Effective independent data points ~5 (autocorrelation time ~1-2 us on 10 us trace). "
        "Scholz (2006) does not state measurement uncertainty; values above are estimates. "
        "Framework: ASME V&V 20-2009 for validation, GUM (JCGM 100:2008) for measurement."
    ),
)

NX2_DATA = ExperimentalDevice(
    name="NX2",
    institution="NIE Singapore",
    capacitance=28e-6,             # 28 uF
    voltage=11.5e3,                # 11.5 kV operating voltage (Lee & Saw 2008)
    inductance=20e-9,              # 20 nH (RADPF Module 1)
    resistance=2.3e-3,             # 2.3 mOhm (RADPF; RESF=0.1)
    anode_radius=0.019,            # 19 mm
    cathode_radius=0.041,          # 41 mm
    anode_length=0.05,             # 50 mm
    fill_pressure_torr=3.0,        # 3 Torr D2 (Lee & Saw 2008)
    fill_gas="deuterium",
    peak_current=400e3,            # 400 kA (Lee & Saw 2008)
    neutron_yield=1e8,
    current_rise_time=1.8e-6,      # 1.8 us
    reference="Lee & Saw, J. Fusion Energy 27:292, 2008; RADPF Module 1",
    peak_current_uncertainty=0.08,     # 8% (compact device, lower SNR)
    rise_time_uncertainty=0.12,        # 12%
    neutron_yield_uncertainty=0.60,    # 60% (shot-to-shot)
    measurement_notes=(
        "No digitized waveform available. Peak current and rise time from "
        "Lee & Saw, J. Fusion Energy 27:292, 2008. R0=2.3 mOhm from RADPF "
        "Module 1 preset (plasmafocus.net); actual RESF=R0/sqrt(L0/C)=0.086 "
        "(not 0.1 as sometimes stated). "
        "Fill pressure 3 Torr D2 for neutron operation. "
        "WARNING: The 400 kA peak current is likely a RADPF model output, "
        "not a direct Rogowski coil measurement. The unloaded RLC peak is "
        "402.5 kA (implying only 0.6% plasma loading, which is physically "
        "implausible for any DPF discharge). Treat as 'reference' quality, "
        "not 'experimental measurement' quality. "
        "L0 uncertainty: literature reports 15-20 nH (Sahyouni et al. 2021 "
        "DOI:10.1155/2021/6611925 vs RADPF preset). "
        "Uncertainties are Type B estimates (not stated in source)."
    ),
)

UNU_ICTP_DATA = ExperimentalDevice(
    name="UNU-ICTP",
    institution="UNU-ICTP PFF",
    capacitance=30e-6,             # 30 uF
    voltage=14e3,                  # 14 kV
    inductance=110e-9,             # 110 nH
    resistance=12e-3,              # 12 mOhm
    anode_radius=0.0095,           # 9.5 mm
    cathode_radius=0.032,          # 32 mm
    anode_length=0.16,             # 160 mm
    fill_pressure_torr=3.0,
    fill_gas="deuterium",
    peak_current=170e3,            # 170 kA
    neutron_yield=1e8,
    current_rise_time=2.8e-6,      # 2.8 us
    reference="Lee et al., Am. J. Phys. 56, 1988",
    peak_current_uncertainty=0.10,     # 10% (training device, less precise)
    rise_time_uncertainty=0.15,        # 15%
    neutron_yield_uncertainty=0.70,    # 70% (shot-to-shot)
    measurement_notes=(
        "No digitized waveform available. Parameters from Lee et al., Am. J. Phys. 56, 1988. "
        "Uncertainties are Type B estimates (not stated in source)."
    ),
)


# PF-1000 at 16 kV — Akel et al., Radiat. Phys. Chem. 188:109638, 2021
# Same device (IPPLM Warsaw), different operating conditions:
#   V0 = 16 kV (vs 27 kV), fill pressure = 1.05 Torr D2 (vs 3.5 Torr)
# Peak current measured at 1.1-1.3 MA for multiple shots.
# Digitized I(t) waveform available in paper (Fig. 3).
PF1000_16KV_DATA = ExperimentalDevice(
    name="PF-1000-16kV",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # Same bank
    voltage=16e3,                  # 16 kV (reduced from 27 kV)
    inductance=33.5e-9,            # Same circuit
    resistance=2.3e-3,             # Same circuit
    anode_radius=0.115,            # Same geometry
    cathode_radius=0.16,           # Same geometry
    anode_length=0.60,             # Same geometry
    fill_pressure_torr=1.05,       # 1.05 Torr D2 (Akel 2021)
    fill_gas="deuterium",
    peak_current=1.2e6,            # 1.2 MA (midpoint of 1.1-1.3 MA range)
    neutron_yield=2.33e9,          # 2.33e9 n/shot at 1.05 Torr (average of 16 shots)
    current_rise_time=6.0e-6,      # ~6 us (estimated from Lee model fit in paper)
    reference="Akel et al., Radiat. Phys. Chem. 188:109638, 2021",
    peak_current_uncertainty=0.10,     # 10% (range 1.1-1.3 MA = ±8.3%)
    rise_time_uncertainty=0.15,        # 15% (no explicit timing stated)
    neutron_yield_uncertainty=0.40,    # 40% (shot-to-shot, Akel Table 1)
    measurement_notes=(
        "PF-1000 operated at 16 kV (170.5 kJ) with 1.05 Torr D2 fill. "
        "Measured I(t) waveform in Fig. 3 of Akel et al. (2021). "
        "Peak current 1.1-1.3 MA across multiple shots (16 shots at 1.05 Torr). "
        "Lee model fitted with good agreement to measured traces. "
        "Neutron yield 2.33e9 ± 40% shot-to-shot. "
        "DOI: 10.1016/j.radphyschem.2021.109638"
    ),
)


# Registry mapping device name -> ExperimentalDevice
DEVICES: dict[str, ExperimentalDevice] = {
    "PF-1000": PF1000_DATA,
    "PF-1000-16kV": PF1000_16KV_DATA,
    "NX2": NX2_DATA,
    "UNU-ICTP": UNU_ICTP_DATA,
}


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
    mu_0 = 4.0 * np.pi * 1e-7  # [H/m]
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
# Helpers
# =====================================================================

def _find_first_peak(signal: np.ndarray, min_prominence: float = 0.05) -> int:
    """Find the index of the first local maximum (first peak) in *signal*.

    The algorithm identifies the first point where the signal transitions
    from rising to falling, provided the peak is at least *min_prominence*
    times the global maximum.  This avoids picking up early noise spikes.

    Falls back to ``np.argmax(signal)`` if no qualifying local peak is
    found (e.g. a monotonically rising signal).

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Non-negative signal (typically ``np.abs(I)``).
    min_prominence : float
        Minimum fraction of the global max that a local peak must reach
        to qualify.  Default 0.05 (5 %).

    Returns
    -------
    int
        Index of the first qualifying local peak.
    """
    if len(signal) < 3:
        return int(np.argmax(signal))

    global_max = float(np.max(signal))
    threshold = min_prominence * global_max
    n_decline = 3  # Points ahead to verify sustained decline

    # Walk through signal and find first point where signal shows a
    # sustained decline, confirming a true local maximum rather than a
    # phase-transition plateau (common in DPF current waveforms).
    # A candidate peak at index i is confirmed if:
    # 1. Signal has been rising above threshold
    # 2. signal[i] >= signal[i-1] (local max candidate)
    # 3. All n_decline points after i are strictly below signal[i]
    #    AND the signal does not recover above signal[i] within that window
    rising = False
    for i in range(1, len(signal) - n_decline):
        if signal[i] >= threshold:
            rising = True
        if rising and signal[i] >= threshold and signal[i] >= signal[i - 1]:
            # Confirm: all following n_decline points are below peak value
            is_peak = all(
                signal[i + k + 1] < signal[i]
                for k in range(n_decline)
            )
            if is_peak:
                return i

    # Fallback: global maximum
    return int(np.argmax(signal))


# =====================================================================
# Waveform comparison
# =====================================================================

def nrmse_peak(
    t_sim: np.ndarray,
    I_sim: np.ndarray,
    t_exp: np.ndarray,
    I_exp: np.ndarray,
    truncate_at_dip: bool = False,
) -> float:
    """Compute peak-normalized RMSE between simulated and experimental waveforms.

    Resamples the simulated waveform onto the experimental time grid via
    linear interpolation, then computes NRMSE = RMSE / |I_peak_exp|.

    Parameters
    ----------
    t_sim : ndarray
        Simulated time array [s].
    I_sim : ndarray
        Simulated current waveform [A].
    t_exp : ndarray
        Experimental time array [s].
    I_exp : ndarray
        Experimental current waveform [A].
    truncate_at_dip : bool, optional
        If True, truncate comparison at the current dip (first minimum
        of |I| after peak).  This excludes the post-pinch region where
        frozen-L_plasma makes the model invalid.  Default False.

    Returns
    -------
    float
        Peak-normalized RMSE (dimensionless).  0.0 for a perfect match.
    """
    t_e = np.asarray(t_exp, dtype=np.float64)
    I_e = np.asarray(I_exp, dtype=np.float64)

    if truncate_at_dip:
        # Find the current dip in the SIMULATED waveform — the model is
        # invalid after the dip (frozen L_plasma region).  Truncate the
        # experimental time grid to only include times up to the sim dip.
        abs_I_sim = np.abs(np.asarray(I_sim, dtype=np.float64))
        sim_peak_idx = int(np.argmax(abs_I_sim))
        post_peak_sim = abs_I_sim[sim_peak_idx:]
        if len(post_peak_sim) > 2:
            dip_offset = int(np.argmin(post_peak_sim))
            if dip_offset > 1:
                t_sim_arr = np.asarray(t_sim, dtype=np.float64)
                t_dip = t_sim_arr[sim_peak_idx + dip_offset]
                # Truncate experimental grid at sim dip time
                mask = t_e <= t_dip
                if np.sum(mask) > 2:
                    t_e = t_e[mask]
                    I_e = I_e[mask]

    I_sim_resampled = np.interp(t_e, t_sim, I_sim)
    residuals = I_sim_resampled - I_e
    rmse = float(np.sqrt(np.mean(residuals**2)))
    I_peak_exp = float(np.max(np.abs(I_e)))
    return rmse / max(I_peak_exp, 1e-300)


# Backward-compatible alias (CRIT-1: prefer nrmse_peak for clarity)
normalized_rmse = nrmse_peak


# =====================================================================
# Validation functions
# =====================================================================

def validate_current_waveform(
    t_sim: np.ndarray,
    I_sim: np.ndarray,
    device_name: str,
    truncate_at_dip: bool = False,
) -> dict[str, Any]:
    """Validate a simulated current waveform against experimental data.

    Compares the peak current magnitude and its timing against published
    measurements for the specified device.

    Parameters
    ----------
    t_sim : ndarray, shape (M,)
        Simulated time array [s].
    I_sim : ndarray, shape (M,)
        Simulated current waveform [A].
    device_name : str
        Key into ``DEVICES`` (e.g. ``"PF-1000"``, ``"NX2"``).

    Returns
    -------
    dict
        ``peak_current_error`` : float
            Relative error |I_peak_sim - I_peak_exp| / I_peak_exp.
        ``peak_current_sim`` : float
            Peak of simulated current [A].
        ``peak_current_exp`` : float
            Experimental peak current [A].
        ``timing_ok`` : bool
            True if simulated peak time is within 10 % of experimental
            rise time.

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    t_arr = np.asarray(t_sim, dtype=np.float64)
    I_arr = np.asarray(I_sim, dtype=np.float64)

    # Peak current: find the FIRST local maximum of |I(t)|.
    # For DPF waveforms, the first peak (before the current dip) is the
    # physically meaningful one.  Post-pinch oscillation peaks can exceed
    # the first peak and must not be mistaken for the primary peak.
    abs_I = np.abs(I_arr)
    peak_idx = _find_first_peak(abs_I)
    peak_current_sim = float(abs_I[peak_idx])
    peak_time_sim = float(t_arr[peak_idx])

    peak_current_exp = device.peak_current
    rise_time_exp = device.current_rise_time

    # Relative error on peak current
    peak_current_error = abs(peak_current_sim - peak_current_exp) / max(
        abs(peak_current_exp), 1e-300
    )

    # Timing check: peak time within 10% of experimental rise time
    timing_error = abs(peak_time_sim - rise_time_exp) / max(rise_time_exp, 1e-300)
    timing_ok = timing_error < 0.10

    # Uncertainty budget following GUM (JCGM 100:2008) and ASME V&V 20-2009.
    # Components: Rogowski coil (Type B), digitization (Type B), simulation error.
    # u_combined = sqrt(u_rogowski^2 + u_digitization^2 + u_sim^2)
    u_exp_peak = device.peak_current_uncertainty
    u_exp_timing = device.rise_time_uncertainty
    u_digitization = device.waveform_digitization_uncertainty
    # Total experimental uncertainty (Rogowski + digitization in quadrature)
    u_exp_total = np.sqrt(u_exp_peak**2 + u_digitization**2)
    # Combined uncertainty (experimental + simulation error)
    u_combined_peak = np.sqrt(u_exp_total**2 + peak_current_error**2)
    u_combined_timing = np.sqrt(u_exp_timing**2 + timing_error**2)
    # Agreement check: simulation within 2-sigma of total experimental uncertainty
    agreement_within_2sigma = peak_current_error <= 2.0 * max(u_exp_total, 0.01)

    # Waveform NRMSE: compare full I(t) trace if digitized waveform available
    waveform_available = (
        device.waveform_t is not None and device.waveform_I is not None
    )
    waveform_nrmse = float("nan")
    if waveform_available:
        waveform_nrmse = normalized_rmse(
            t_arr, I_arr, device.waveform_t, device.waveform_I,
            truncate_at_dip=truncate_at_dip,
        )

    return {
        "peak_current_error": peak_current_error,
        "peak_current_sim": peak_current_sim,
        "peak_current_exp": peak_current_exp,
        "peak_time_sim": peak_time_sim,
        "timing_ok": timing_ok,
        "timing_error": timing_error,
        "waveform_available": waveform_available,
        "waveform_nrmse": waveform_nrmse,
        "uncertainty": {
            "peak_current_exp_1sigma": u_exp_peak,
            "digitization_1sigma": u_digitization,
            "peak_current_total_exp_1sigma": float(u_exp_total),
            "rise_time_exp_1sigma": u_exp_timing,
            "peak_current_combined_1sigma": float(u_combined_peak),
            "timing_combined_1sigma": float(u_combined_timing),
            "agreement_within_2sigma": bool(agreement_within_2sigma),
        },
        "measurement_notes": device.measurement_notes,
    }


def validate_neutron_yield(
    Y_sim: float,
    device_name: str,
) -> dict[str, Any]:
    """Validate simulated neutron yield against experimental data.

    Parameters
    ----------
    Y_sim : float
        Simulated total neutron yield.
    device_name : str
        Key into ``DEVICES``.

    Returns
    -------
    dict
        ``yield_ratio`` : float
            Y_sim / Y_exp.
        ``within_order_magnitude`` : bool
            True if 0.1 < ratio < 10.
        ``yield_sim`` : float
        ``yield_exp`` : float

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    yield_exp = device.neutron_yield
    yield_ratio = Y_sim / max(yield_exp, 1e-300)

    u_exp_yield = device.neutron_yield_uncertainty
    return {
        "yield_ratio": yield_ratio,
        "within_order_magnitude": 0.1 < yield_ratio < 10.0,
        "yield_sim": float(Y_sim),
        "yield_exp": yield_exp,
        "uncertainty": {
            "neutron_yield_exp_1sigma": u_exp_yield,
        },
    }


def device_to_config_dict(device_name: str) -> dict[str, Any]:
    """Convert device parameters to a ``SimulationConfig``-compatible dict.

    Produces a configuration dictionary that can be passed directly to
    ``SimulationConfig(**device_to_config_dict("PF-1000"))`` or written
    to a JSON config file.

    The grid shape is chosen to give approximately 1 mm axial resolution
    (capped at 256 cells per dimension for tractability).

    The initial fill gas density is computed from the ideal gas law at
    room temperature (300 K)::

        n = p / (k_B * T)
        rho = n * m_D2  # D2 molecular mass at room temperature

    where *p* is the fill pressure converted from Torr to Pa.

    Parameters
    ----------
    device_name : str
        Key into ``DEVICES``.

    Returns
    -------
    dict
        Configuration dictionary with keys matching ``SimulationConfig``
        fields (``grid_shape``, ``dx``, ``sim_time``, ``circuit``, etc.).

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    # --- Grid resolution ------------------------------------------------
    # Target ~1 mm resolution; cap at 256 cells per dimension
    target_dx = 1e-3  # 1 mm

    # Domain size: slightly larger than cathode diameter x anode length
    domain_r = device.cathode_radius * 1.5
    domain_z = device.anode_length * 1.5

    nx = min(int(np.ceil(2.0 * domain_r / target_dx)), 256)
    ny = min(int(np.ceil(2.0 * domain_r / target_dx)), 256)
    nz = min(int(np.ceil(domain_z / target_dx)), 256)

    # Ensure at least 8 cells per dimension
    nx = max(nx, 8)
    ny = max(ny, 8)
    nz = max(nz, 8)

    # Actual grid spacing from chosen cell count
    dx = 2.0 * domain_r / nx

    # --- Fill gas density from ideal gas law ----------------------------
    # Convert Torr to Pa: 1 Torr = 133.322 Pa
    pressure_Pa = device.fill_pressure_torr * 133.322
    T_room = 300.0  # K

    n_fill = pressure_Pa / (k_B * T_room)  # number density [m^-3]
    rho_fill = n_fill * m_D2                # mass density [kg/m^3] — D2 molecular

    # --- Simulation time ------------------------------------------------
    # A few quarter-periods is usually enough to capture peak current + pinch
    sim_time = 4.0 * device.current_rise_time

    return {
        "grid_shape": [nx, ny, nz],
        "dx": dx,
        "sim_time": sim_time,
        "rho0": rho_fill,
        "T0": T_room,
        "ion_mass": m_d,
        "circuit": {
            "C": device.capacitance,
            "V0": device.voltage,
            "L0": device.inductance,
            "R0": device.resistance,
            "anode_radius": device.anode_radius,
            "cathode_radius": device.cathode_radius,
        },
    }
