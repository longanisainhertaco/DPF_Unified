"""Waveform comparison and validation functions for DPF simulations.

Provides NRMSE computation, waveform comparison, and scalar validation
functions against experimental device data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dpf.constants import k_B, m_d, m_D2
from dpf.validation.experimental_devices import DEVICES

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
    max_time: float | None = None,
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
        If True, truncate comparison at the current dip (first local
        minimum of |I| after peak, searched within a limited window of
        2× the peak time).  This excludes the post-pinch region where
        frozen-L_plasma makes the model invalid.  Default False.
    max_time : float or None, optional
        If given, truncate comparison at this time [s].  Only
        experimental points with t <= max_time are included.
        Useful for windowed validation (e.g. rise-phase only).

    Returns
    -------
    float
        Peak-normalized RMSE (dimensionless).  0.0 for a perfect match.
    """
    t_e = np.asarray(t_exp, dtype=np.float64)
    I_e = np.asarray(I_exp, dtype=np.float64)

    # Explicit time window truncation
    if max_time is not None:
        mask = t_e <= max_time
        if np.sum(mask) > 2:
            t_e = t_e[mask]
            I_e = I_e[mask]

    if truncate_at_dip:
        # Find the current dip in the SIMULATED waveform — the model is
        # invalid after the dip (frozen L_plasma region).  Search only
        # within a limited window (2× peak time) to avoid picking up
        # the L-R crowbar decay tail at late times.
        abs_I_sim = np.abs(np.asarray(I_sim, dtype=np.float64))
        t_sim_arr = np.asarray(t_sim, dtype=np.float64)
        sim_peak_idx = int(np.argmax(abs_I_sim))
        t_peak = t_sim_arr[sim_peak_idx]

        # Search window: peak to 2× peak time (captures the dip but not
        # the crowbar L-R decay which can extend to 10× peak time)
        t_search_end = 2.0 * t_peak
        search_end_idx = int(np.searchsorted(t_sim_arr, t_search_end))
        search_end_idx = min(search_end_idx, len(abs_I_sim))

        post_peak_sim = abs_I_sim[sim_peak_idx:search_end_idx]
        if len(post_peak_sim) > 2:
            dip_offset = int(np.argmin(post_peak_sim))
            if dip_offset > 1:
                t_dip = t_sim_arr[sim_peak_idx + dip_offset]
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
    u_digitization = device.waveform_amplitude_uncertainty
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

    source_authority = {
        "kr_status": device.kr_status,
        "reliability": device.reliability,
        "waveform_provenance": device.waveform_provenance,
        "waveform_kr_status": device.waveform_kr_status,
    }
    validation_ready = (
        source_authority["kr_status"] == "verified"
        and source_authority["reliability"] == "measured"
        and source_authority["waveform_provenance"] == "measured"
        and source_authority["waveform_kr_status"] == "verified"
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
        "source_authority": {
            **source_authority,
            "validation_ready": validation_ready,
            "validation_role": (
                "tier1_circuit_evidence_candidate"
                if validation_ready else "numeric_comparison_only"
            ),
        },
        "validity_notes": {
            "numeric_metrics": (
                "Peak, timing, and NRMSE metrics are numeric comparisons. "
                "They support validation claims only when source_authority."
                "validation_ready is true and the strict evidence helper passes."
            ),
        },
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
        "source_authority": {
            "kr_status": device.kr_status,
            "reliability": device.reliability,
            "waveform_kr_status": device.waveform_kr_status,
            "validation_ready": False,
            "validation_role": "numeric_yield_comparison_only",
        },
        "validity_notes": {
            "not_neutron_physics_validation": (
                "Total-yield order checks do not validate neutron mechanism, "
                "timing, spectrum, anisotropy, detector response, or beam-target "
                "physics. Tier-5 validation requires separate KR-sourced evidence."
            ),
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
