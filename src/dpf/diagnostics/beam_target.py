"""Beam-target neutron yield model for Dense Plasma Focus.

Canonical model: Lee & Saw (KR §5109-5145) phenomenological beam-target
form with a single calibrated proportionality constant Cn.

KR formula (eq. 1, verbatim):
    Yb-t = Cn * n_i * I_pinch^2 * z_p^2 * ln(b/r_p) * sigma(E_beam) / V_max^(1/2)

with E_beam = 3 * V_max (KR L5133-5139, motivated by experimental
observations that the relevant ion energy is 30-150 keV while the
code computes V_max in the 20-50 kV range).

Cn calibration (KR L5141-5144): Yn = 7e9 at I_pinch = 0.5 MA, using
canonical PF-1000 geometry (a=0.115 m, b=0.16 m, r_p=0.1*a=0.0115 m,
z_p=0.06 m) and typical inputs (n_i=1e25 m^-3, V_max=1e5 V). This
yields Cn = 1.810e7 in SI units consistent with the formula.

DD cross section uses the Bosch-Hale (1992) parametric fit:
    sigma(E) = S(E) / (E * exp(B_G / sqrt(E)))

where S(E) is the astrophysical S-factor with a 5th-order rational
polynomial fit valid for 0.5 keV < E < 5000 keV.

The legacy `beam_target_yield_rate(I_pinch, V_pinch, n_target, L_target,
f_beam)` function with f_beam-style scaling is retained as
`_legacy_beam_target_yield_rate` for backward-compat consumers.

References:
    Lee S. & Saw S.H., A Course on Plasma Focus Numerical Experiments
        Part 1 (Basic Course), §5109-5145 (KR).
    Bosch & Hale, Nuclear Fusion 32:611 (1992)
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
# Lee/Saw KR-canonical beam-target neutron yield (KR §5109-5145)
# ---------------------------------------------------------------------------

# Cn calibrated against KR L5141-5144 datum: Yn = 7e9 at I_pinch = 0.5 MA,
# using canonical PF-1000 geometry (a=0.115 m, b=0.16 m, r_p=0.1*a,
# z_p=0.06 m) and typical inputs (n_i=1e25 m^-3, V_max=1e5 V).
# [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md L5141-5144]
_LEE_SAW_CN: float = 1.810426e7

# Standard Lee/Saw cathode-to-pinch radius ratio for legacy-signature wrapper:
# b/r_p = 0.16 / 0.0115 = 13.91, ln(b/r_p) = 2.633 (PF-1000 canonical)
_LEE_SAW_LN_BRP_DEFAULT: float = 2.6328


@njit(cache=True)
def beam_target_yield_lee_saw(
    n_i: float,
    I_pinch: float,
    z_p: float,
    b: float,
    r_p: float,
    V_max: float,
) -> float:
    """Lee/Saw KR-canonical beam-target neutron yield (total per shot).

    Implements KR eq. 1 [KR: a-course-on-plasma-focus-numerical-experiments-
    s-lee-and-s-h-saw-part-1-basic-course.md L5125-5128] verbatim:

        Yb-t = Cn * n_i * I_pinch^2 * z_p^2 * ln(b/r_p) * sigma(E_beam) / sqrt(V_max)

    with E_beam = 3 * V_max in lab frame [KR L5133-5139], converted to CM
    energy (E_cm = E_lab / 2) for the Bosch-Hale DD cross section.

    Cn = 1.810e7 calibrated to give Yn = 7e9 at I_pinch = 0.5 MA per
    KR L5141-5144 with canonical PF-1000 geometry.

    Returns the total beam-target neutron yield per shot, NOT a rate.
    To convert to a rate for time-integration, divide by the pinch dwell
    time tau ~ r_p / v_beam ~ r_p / sqrt(2*E_beam/m_d).

    Args:
        n_i: Ion (deuterium) number density in pinch [m^-3].
        I_pinch: Pinch current [A].
        z_p: Pinch column length [m].
        b: Cathode radius [m].
        r_p: Pinch (collapsed plasma) radius [m].
        V_max: Maximum induced voltage from current sheet collapse [V].

    Returns:
        Total beam-target neutron yield [neutrons]. Zero if any input
        is non-positive or if b <= r_p (invalid geometry).
    """
    if (
        n_i <= 0.0
        or I_pinch <= 0.0
        or z_p <= 0.0
        or b <= 0.0
        or r_p <= 0.0
        or V_max <= 0.0
        or b <= r_p
    ):
        return 0.0

    ln_brp = np.log(b / r_p)

    # E_beam = 3 * V_max [KR L5135-5139]. KR notes V_max is "of order 20-50 kV"
    # and the 3x scales it into the experimentally observed 50-150 keV beam ion
    # energy range. Cap E_beam at 500 keV (lab) to stay within Bosch-Hale fit
    # validity and to avoid unphysical extrapolation when upstream callers pass
    # an inflated V_max (e.g. inductive back-EMF estimates exceeding 1 MV).
    E_lab_keV = 3.0 * V_max / 1000.0
    if E_lab_keV > 500.0:
        E_lab_keV = 500.0
    E_cm_keV = E_lab_keV / 2.0

    sigma = dd_cross_section(E_cm_keV)
    if sigma <= 0.0:
        return 0.0

    # KR eq. 1: Yb-t = Cn * n_i * I_pinch^2 * z_p^2 * ln(b/r_p) * sigma / sqrt(V_max)
    Yn = (
        _LEE_SAW_CN
        * n_i
        * I_pinch * I_pinch
        * z_p * z_p
        * ln_brp
        * sigma
        / np.sqrt(V_max)
    )

    return max(Yn, 0.0)


# ---------------------------------------------------------------------------
# Beam-target neutron yield rate (legacy-signature wrapper, Lee/Saw default)
# ---------------------------------------------------------------------------


@njit(cache=True)
def beam_target_yield_rate(
    I_pinch: float,
    V_pinch: float,
    n_target: float,
    L_target: float,
    f_beam: float = 0.14,
    tau_dwell: float = 0.0,
) -> float:
    """Beam-target DD neutron production rate [1/s].

    Lee/Saw KR-canonical form (default since 2026-04-27). See
    `beam_target_yield_lee_saw` for the unwrapped per-shot yield form.

    KR eq. 1 [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-
    s-h-saw-part-1-basic-course.md §"beam-target yield" L4080-4087 p.18]
    is verbatim:

        Yb-t = Cn * n_i * I_pinch^2 * z_p^2 * ln(b/r_p) * sigma / V_max^(1/2)   (1)

    This is a PER-SHOT total yield [neutrons], NOT a rate. The published
    calibration (Yn = 7e9 at I_pinch = 0.5 MA, KR L4103-4104) is a
    single-shot scalar. The derivation at KR L4064-4078 (p.18) carries
    the beam-target interaction time tau through the proportionality
    Yb-t ~ nb*ni*(rp^2*zp)*(sigma*vb)*tau, and the substitutions
    nb ~ Lp*I^2/vb^2, tau ~ rp ~ zp, vb ~ U^(1/2) collapse tau into the
    geometric and voltage factors of eq. (1). The tau is already absorbed.

    BUG HISTORY (fixed 2026-05-04, MJOLNIR-2MJ Yn 41x over-prediction):
    The previous wrapper divided Yn_total by a beam transit time
    tau_transit = L_target/v_beam ~ 1 ns, then callers integrated
    dY_dt * dt over a ~30-50 ns pinch dwell, double-counting tau and
    over-predicting by 30-50x. Exact match to the 41x MJOLNIR anomaly.

    CORRECT NORMALIZATION: divide the per-shot Yn by the pinch dwell
    time tau_dwell (the duration over which the beam-target reaction
    occurs). When the caller integrates dY_dt * dt over a window of
    length tau_dwell, the time-integral recovers Yn_total exactly.
    Per KR L4068-4069 (p.18): "tau is the beam-target interaction time
    assumed proportional to the confinement time of the plasma column."

    The caller MUST pass tau_dwell explicitly (the pinch dwell time
    over which it intends to integrate). If tau_dwell <= 0 the function
    returns 0.0 and emits no yield; callers that cannot supply a dwell
    time should use `beam_target_yield_lee_saw` directly and gate the
    per-shot yield to a single MHD timestep at peak compression.

    Args:
        I_pinch: Pinch current [A].
        V_pinch: Pinch voltage [V] (interpreted as V_max in KR eq. 1).
        n_target: Target deuterium number density [m^-3] (KR n_i).
        L_target: Pinch column length [m] (KR z_p).
        f_beam: Multiplicative scaling around 0.14 baseline. Range [0,1].
        tau_dwell: Pinch dwell time [s] over which the caller integrates
            dY/dt to recover the per-shot total. MUST be > 0; if 0 or
            negative, returns 0.0 (the caller should switch to a one-shot
            yield gate via `beam_target_yield_lee_saw`).

    Returns:
        Beam-target neutron production rate dY/dt [1/s]. The time-integral
        of dY/dt over tau_dwell equals f_beam_scale * Yn_total per KR eq. 1.
        Zero if any input is non-positive or tau_dwell is non-positive.
    """
    if I_pinch <= 0.0 or V_pinch <= 0.0 or n_target <= 0.0 or L_target <= 0.0:
        return 0.0
    if tau_dwell <= 0.0:
        # Refuse to emit a rate without an explicit dwell time. This
        # prevents the historical bug where tau_transit ~ 1 ns was used
        # implicitly while callers integrated over ~30-50 ns pinch dwells.
        return 0.0

    # Clamp f_beam to physical range and scale around 0.14 baseline so
    # that f_beam=0.14 reproduces the unscaled Lee/Saw KR formula
    fb = max(min(f_beam, 1.0), 0.0)
    fb_scale = fb / 0.14

    # E_beam = 3 * V_max [KR L5135-5139], capped at 500 keV (lab) for
    # Bosch-Hale fit validity (see beam_target_yield_lee_saw notes).
    E_lab_keV = 3.0 * V_pinch / 1000.0
    if E_lab_keV > 500.0:
        E_lab_keV = 500.0
    E_cm_keV = E_lab_keV / 2.0

    sigma = dd_cross_section(E_cm_keV)
    if sigma <= 0.0:
        return 0.0

    # KR eq. 1 with canonical ln(b/r_p) = 2.633 (PF-1000 default)
    Yn_total = (
        _LEE_SAW_CN
        * n_target
        * I_pinch * I_pinch
        * L_target * L_target
        * _LEE_SAW_LN_BRP_DEFAULT
        * sigma
        / np.sqrt(V_pinch)
    )

    # Spread the per-shot total uniformly over the explicit dwell time so
    # that the caller's time-integral of dY_dt * dt over [t_pinch_start,
    # t_pinch_start + tau_dwell] recovers Yn_total. Any integration
    # window mismatch is the caller's responsibility, not the wrapper's.
    dY_dt = fb_scale * Yn_total / tau_dwell

    return max(dY_dt, 0.0)


@njit(cache=True)
def _legacy_beam_target_yield_rate(
    I_pinch: float,
    V_pinch: float,
    n_target: float,
    L_target: float,
    f_beam: float = 0.14,
) -> float:
    """Legacy non-canonical beam-target rate: dY/dt = f_beam*(I/e)*n*sigma*L.

    [EMPIRICAL — uncalibrated, retained for backward compat with consumers
    that intentionally need the linear-in-current form. Documented as
    superseded by the Lee/Saw KR form (`beam_target_yield_rate`,
    `beam_target_yield_lee_saw`) on 2026-04-27.]

    This form is dimensionally a per-beam-ion reaction probability times
    a beam flux f_beam*I/e, but it lacks the Lee/Saw I_pinch^2 scaling
    that arises from beam-energy ~ Lp*I^2 (KR L5115-5119). It under-predicts
    by 4-60x at PF-1000 27 kV.
    """
    if I_pinch <= 0.0 or V_pinch <= 0.0 or n_target <= 0.0 or L_target <= 0.0:
        return 0.0

    fb = max(min(f_beam, 1.0), 0.0)

    # Lab-frame beam energy: E_lab = e * V_pinch [J] -> keV
    E_lab_keV = V_pinch * e_charge / (1.0e3 * eV)
    E_cm_keV = E_lab_keV / 2.0

    sigma = dd_cross_section(E_cm_keV)

    beam_flux = fb * I_pinch / e_charge
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


# ---------------------------------------------------------------------------
# Multi-event neutron decomposition (Goyon et al. 2025)
# ---------------------------------------------------------------------------


def decompose_neutron_events(
    times: np.ndarray,
    rates: np.ndarray,
    threshold_fraction: float = 0.1,
    min_separation_ns: float = 10.0,
) -> dict:
    """Decompose time-resolved neutron signal into distinct emission events.

    MJOLNIR and other large DPF devices can produce multiple distinct neutron
    emission events during a single discharge, corresponding to separate pinch
    compressions or instability-driven re-pinches.  This function identifies
    and characterizes each event from the time-resolved neutron yield rate.

    Algorithm:
        1. Smooth the signal (3-point moving average) to suppress noise.
        2. Identify peaks above a threshold (fraction of global maximum).
        3. Merge peaks closer than min_separation_ns into a single event.
        4. For each event, compute: peak time, peak rate, FWHM duration,
           and integrated yield (trapezoidal rule over the event).

    Args:
        times: Time array [s].
        rates: Neutron yield rate array [1/s], same length as times.
        threshold_fraction: Minimum peak height as fraction of global max
            to count as an event (default 0.1 = 10%).
        min_separation_ns: Minimum time between distinct events [ns]
            (default 10 ns).  Peaks closer than this are merged.

    Returns:
        Dictionary with:
            n_events: Number of distinct neutron events.
            events: List of dicts, each with keys:
                peak_time: Time of event peak [s].
                peak_rate: Peak neutron rate [1/s].
                fwhm_ns: Full-width at half-maximum [ns].
                yield_count: Integrated yield for this event.
                start_time: Event start time [s].
                end_time: Event end time [s].
            total_yield: Total integrated yield across all events.
            primary_fraction: Fraction of yield in the largest event.

    References:
        Goyon et al., Phys. Plasmas 32:033105 (2025) — multi-event
        neutron dynamics in MJOLNIR DPF.
    """
    times = np.asarray(times, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)

    if len(times) < 3 or np.max(rates) <= 0:
        return {
            "n_events": 0,
            "events": [],
            "total_yield": 0.0,
            "primary_fraction": 0.0,
        }

    # Step 1: Smooth with 3-point moving average
    smoothed = np.copy(rates)
    for i in range(1, len(smoothed) - 1):
        smoothed[i] = (rates[i - 1] + rates[i] + rates[i + 1]) / 3.0

    # Step 2: Find local maxima above threshold
    rate_max = float(np.max(smoothed))
    threshold = threshold_fraction * rate_max
    min_sep_s = min_separation_ns * 1e-9

    peaks: list[int] = []
    for i in range(1, len(smoothed) - 1):
        if (smoothed[i] > smoothed[i - 1]
                and smoothed[i] >= smoothed[i + 1]
                and smoothed[i] >= threshold):
            peaks.append(i)

    if not peaks:
        return {
            "n_events": 0,
            "events": [],
            "total_yield": float(np.trapezoid(rates, times)),
            "primary_fraction": 0.0,
        }

    # Step 3: Merge peaks closer than min_separation
    merged: list[list[int]] = [[peaks[0]]]
    for pk in peaks[1:]:
        if times[pk] - times[merged[-1][-1]] < min_sep_s:
            merged[-1].append(pk)
        else:
            merged.append([pk])

    # Step 4: Characterize each event
    events = []
    for group in merged:
        # Pick the highest peak in the group
        best_idx = max(group, key=lambda i: smoothed[i])
        peak_rate = float(smoothed[best_idx])
        peak_time = float(times[best_idx])

        # Find FWHM: half-max level
        half_max = peak_rate / 2.0

        # Search left for half-max crossing
        left_idx = best_idx
        while left_idx > 0 and smoothed[left_idx] > half_max:
            left_idx -= 1

        # Search right for half-max crossing
        right_idx = best_idx
        while right_idx < len(smoothed) - 1 and smoothed[right_idx] > half_max:
            right_idx += 1

        start_time = float(times[left_idx])
        end_time = float(times[right_idx])
        fwhm_ns = (end_time - start_time) * 1e9

        # Integrated yield over this event window
        event_yield = float(np.trapezoid(rates[left_idx:right_idx + 1],
                                     times[left_idx:right_idx + 1]))

        events.append({
            "peak_time": peak_time,
            "peak_rate": peak_rate,
            "fwhm_ns": fwhm_ns,
            "yield_count": event_yield,
            "start_time": start_time,
            "end_time": end_time,
        })

    total_yield = float(np.trapezoid(rates, times))
    primary_yield = max(ev["yield_count"] for ev in events) if events else 0.0
    primary_fraction = primary_yield / max(total_yield, 1e-300)

    return {
        "n_events": len(events),
        "events": events,
        "total_yield": total_yield,
        "primary_fraction": min(primary_fraction, 1.0),
    }
