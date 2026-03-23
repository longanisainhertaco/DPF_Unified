"""ASME V&V 20-2009 validation assessment and multi-shot uncertainty."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dpf.validation._calibration_data import _SHOT_TO_SHOT_DATA
from dpf.validation._calibration_stats import MonteCarloNRMSEResult

logger = logging.getLogger(__name__)


@dataclass
class ASMEValidationResult:
    """ASME V&V 20-2009 formal validation assessment.

    Follows ASME V&V 20-2009 Section 5: comparison error E, validation
    standard uncertainty u_val, and the ratio E/u_val.  Validation passes
    when |E| <= u_val (ratio <= 1.0).

    Attributes:
        E: Comparison error (model error metric, e.g. NRMSE).
        u_exp: Experimental measurement uncertainty (1-sigma).
        u_input: Input parameter uncertainty (from Monte Carlo, 1-sigma).
        u_num: Numerical solution uncertainty (1-sigma).
        u_val: Validation standard uncertainty = sqrt(u_exp² + u_input² + u_num²).
        ratio: E / u_val.  Pass if <= 1.0.
        passes: True if ratio <= 1.0.
        metric_name: Name of the error metric used for E.
        device_name: Device assessed.
        time_window: Description of the time window used.
        waveform_provenance: "measured" or "reconstructed".
        qualified: True if comparison uses reconstructed (model-derived)
            waveform — result is model-vs-model, not model-vs-experiment.
            Per ASME V&V 20 §4.1, validation data should be independent
            of the computational model.
    """

    E: float
    u_exp: float
    u_input: float
    u_num: float
    u_val: float
    ratio: float
    passes: bool
    delta_model: float = 0.0
    metric_name: str = "NRMSE"
    device_name: str = ""
    time_window: str = "full"
    waveform_provenance: str = ""
    qualified: bool = False


@dataclass
class ASMEStratifiedSummary:
    """Stratified ASME summary splitting measured vs reconstructed waveforms.

    Per ASME V&V 20 §4.1, validation data should be independent of the
    computational model.  Reconstructed waveforms are model-derived and
    therefore "qualified" — their PASS/FAIL status is informative but
    should not be combined with measured-waveform results.

    Attributes:
        all_results: All ASME results.
        measured_results: Results using measured (independent) waveforms.
        reconstructed_results: Results using reconstructed (model-derived)
            waveforms — qualified, model-vs-model.
        n_measured_pass: Number of measured-waveform PASSes.
        n_measured_total: Total measured-waveform assessments.
        n_reconstructed_pass: Number of reconstructed-waveform PASSes.
        n_reconstructed_total: Total reconstructed-waveform assessments.
    """

    all_results: list[ASMEValidationResult]
    measured_results: list[ASMEValidationResult]
    reconstructed_results: list[ASMEValidationResult]
    n_measured_pass: int
    n_measured_total: int
    n_reconstructed_pass: int
    n_reconstructed_total: int

    @property
    def n_total_pass(self) -> int:
        return self.n_measured_pass + self.n_reconstructed_pass

    @property
    def n_total(self) -> int:
        return self.n_measured_total + self.n_reconstructed_total


@dataclass
class MultiShotUncertainty:
    """Estimated experimental uncertainty from shot-to-shot variability.

    PF-1000 shot-to-shot variability is well-documented in the literature:
    - Scholz et al. (2006): sigma_I/I ~ 5% for peak current
    - Lee & Saw (2014): reproducibility to ~5-8% for well-conditioned shots

    Attributes:
        u_shot_to_shot: Shot-to-shot relative uncertainty (1-sigma).
        u_rogowski: Rogowski coil calibration uncertainty (1-sigma).
        u_amplitude: Waveform amplitude uncertainty (1-sigma). Per GUM,
            this is "digitization" for measured waveforms or "reconstruction"
            for model-generated waveforms.
        u_exp_combined: Combined experimental uncertainty (RSS).
        n_shots_typical: Typical number of shots for the estimate.
        u_exp_with_averaging: u_exp after averaging n_shots.
        reference: Literature reference.
    """

    u_shot_to_shot: float
    u_rogowski: float
    u_amplitude: float
    u_exp_combined: float
    n_shots_typical: int
    u_exp_with_averaging: float
    reference: str


def asme_vv20_assessment(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.094,
    f_mr: float = 0.1,
    pinch_column_fraction: float = 0.14,
    crowbar_enabled: bool = True,
    crowbar_resistance: float = 1.5e-3,
    liftoff_delay: float = 0.0,
    max_time: float | None = None,
    u_num: float = 0.001,
    mc_result: MonteCarloNRMSEResult | None = None,
    include_shot_to_shot: bool = True,
) -> ASMEValidationResult:
    """Compute formal ASME V&V 20-2009 validation assessment.

    Computes comparison error E (NRMSE), experimental uncertainty u_exp,
    input parameter uncertainty u_input (from Monte Carlo), and numerical
    uncertainty u_num.  The validation standard uncertainty is:

        u_val = sqrt(u_exp² + u_input² + u_num²)

    Validation passes when |E| <= u_val.

    Args:
        device_name: Device to assess.
        fc: Current fraction.
        fm: Mass fraction.
        f_mr: Radial mass fraction.
        pinch_column_fraction: Pinch column fraction.
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        max_time: If given, compute NRMSE only up to this time [s].
        u_num: Numerical uncertainty (1-sigma, relative).  Default 0.001
            (0.1%) for ODE solver with rtol=1e-8.
        mc_result: Pre-computed Monte Carlo result for u_input.
            If None, uses NRMSE_std = 0.027 as default.
        include_shot_to_shot: Whether to include shot-to-shot variability
            in u_exp (from multi_shot_uncertainty data).  Default True.

    Returns:
        :class:`ASMEValidationResult` with pass/fail assessment.
    """
    from dpf.validation.experimental import DEVICES, nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    # Run the model
    model = LeeModel(
        current_fraction=fc,
        mass_fraction=fm,
        radial_mass_fraction=f_mr,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
        liftoff_delay=liftoff_delay,
    )
    result = model.run(device_name)

    # Compute NRMSE (comparison error E)
    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        raise ValueError(f"No digitized waveform for {device_name}")

    E = nrmse_peak(
        result.t, result.I, device.waveform_t, device.waveform_I,
        max_time=max_time,
    )

    # Experimental uncertainty: combine Rogowski + amplitude + shot-to-shot
    # Per GUM (JCGM 100:2008), each component identified by physical source.
    u_exp_sq = (
        device.peak_current_uncertainty**2
        + device.waveform_amplitude_uncertainty**2
    )
    # Skip shot-to-shot if peak_current_uncertainty already incorporates it
    # (e.g. PF-1000-16kV: 10% from 1.1-1.3 MA range). Per GUM, components
    # must be independent — adding both would be double-counting.
    _skip_shot = getattr(device, "peak_current_from_shot_spread", False)
    if include_shot_to_shot and device_name in _SHOT_TO_SHOT_DATA and not _skip_shot:
        u_shot = _SHOT_TO_SHOT_DATA[device_name]["u_shot_to_shot"]
        n_shots = _SHOT_TO_SHOT_DATA[device_name]["n_shots_typical"]
        # Shot-to-shot component reduces with averaging
        u_shot_avg = u_shot / np.sqrt(n_shots)
        u_exp_sq += u_shot_avg**2
    u_exp = float(np.sqrt(u_exp_sq))

    # Input parameter uncertainty from Monte Carlo
    if mc_result is not None:
        u_input = mc_result.nrmse_std
    else:
        u_input = 0.027  # Default from Phase AS Monte Carlo

    # Validation standard uncertainty
    u_val = float(np.sqrt(u_exp**2 + u_input**2 + u_num**2))

    ratio = E / max(u_val, 1e-15)
    passes = ratio <= 1.0

    # Model-form error per ASME V&V 20-2009 Section 5.3
    delta_model = float(np.sqrt(max(0.0, E**2 - u_val**2)))

    time_desc = f"0-{max_time*1e6:.1f} us" if max_time else "full waveform"

    # Provenance: flag model-vs-model comparisons as qualified
    provenance = device.waveform_provenance
    is_qualified = provenance == "reconstructed"

    status_str = "PASS" if passes else "FAIL"
    if is_qualified:
        status_str += " (qualified: reconstructed waveform)"

    logger.info(
        "ASME V&V 20: %s (%s) — E=%.3f, u_exp=%.3f, u_input=%.3f, "
        "u_num=%.4f, u_val=%.3f, delta_model=%.3f, ratio=%.2f → %s",
        device_name, time_desc, E, u_exp, u_input, u_num, u_val,
        delta_model, ratio, status_str,
    )

    return ASMEValidationResult(
        E=E,
        u_exp=u_exp,
        u_input=u_input,
        u_num=u_num,
        u_val=u_val,
        ratio=ratio,
        passes=passes,
        delta_model=delta_model,
        metric_name="NRMSE",
        device_name=device_name,
        time_window=time_desc,
        waveform_provenance=provenance,
        qualified=is_qualified,
    )


def asme_stratified_summary(
    results: list[ASMEValidationResult],
) -> ASMEStratifiedSummary:
    """Stratify ASME results by waveform provenance.

    Separates measured (genuine validation) from reconstructed
    (model-vs-model, qualified) results.

    Args:
        results: List of ASME validation results.

    Returns:
        :class:`ASMEStratifiedSummary` with per-provenance breakdowns.
    """
    measured = [r for r in results if r.waveform_provenance == "measured"]
    reconstructed = [r for r in results if r.qualified]
    return ASMEStratifiedSummary(
        all_results=results,
        measured_results=measured,
        reconstructed_results=reconstructed,
        n_measured_pass=sum(1 for r in measured if r.passes),
        n_measured_total=len(measured),
        n_reconstructed_pass=sum(1 for r in reconstructed if r.passes),
        n_reconstructed_total=len(reconstructed),
    )


def _pinch_phase_asme(
    device_name: str,
    fc: float,
    fm: float,
    f_mr: float,
    pinch_column_fraction: float,
    crowbar_enabled: bool,
    crowbar_resistance: float,
    t_start_us: float,
) -> ASMEValidationResult | None:
    """Compute ASME assessment for the pinch phase only.

    Computes NRMSE only for waveform data after t_start_us.
    """
    from dpf.validation.experimental import DEVICES
    from dpf.validation.lee_model_comparison import LeeModel

    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        return None

    t_start = t_start_us * 1e-6
    mask = device.waveform_t >= t_start
    if np.sum(mask) < 3:
        return None

    t_exp_pinch = device.waveform_t[mask]
    I_exp_pinch = device.waveform_I[mask]

    model = LeeModel(
        current_fraction=fc,
        mass_fraction=fm,
        radial_mass_fraction=f_mr,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
    )
    result = model.run(device_name)

    # Interpolate simulation to experimental time points
    I_sim_interp = np.interp(t_exp_pinch, result.t, result.I)
    I_peak = float(np.max(np.abs(device.waveform_I)))

    # NRMSE (peak-normalized)
    E = float(np.sqrt(np.mean((I_sim_interp - I_exp_pinch) ** 2)) / I_peak)

    # Experimental uncertainty with shot-to-shot (same as asme_vv20_assessment)
    u_exp_sq = (
        device.peak_current_uncertainty**2
        + device.waveform_amplitude_uncertainty**2
    )
    _skip_shot = getattr(device, "peak_current_from_shot_spread", False)
    if device_name in _SHOT_TO_SHOT_DATA and not _skip_shot:
        u_shot = _SHOT_TO_SHOT_DATA[device_name]["u_shot_to_shot"]
        n_shots = _SHOT_TO_SHOT_DATA[device_name]["n_shots_typical"]
        u_shot_avg = u_shot / np.sqrt(n_shots)
        u_exp_sq += u_shot_avg**2
    u_exp = float(np.sqrt(u_exp_sq))
    u_input = 0.027
    u_num = 0.001
    u_val = float(np.sqrt(u_exp**2 + u_input**2 + u_num**2))
    ratio = E / max(u_val, 1e-15)

    delta_model = float(np.sqrt(max(0.0, E**2 - u_val**2)))

    provenance = device.waveform_provenance
    is_qualified = provenance == "reconstructed"

    return ASMEValidationResult(
        E=E,
        u_exp=u_exp,
        u_input=u_input,
        u_num=u_num,
        u_val=u_val,
        ratio=ratio,
        passes=ratio <= 1.0,
        delta_model=delta_model,
        metric_name="NRMSE",
        device_name=device_name,
        time_window=f"{t_start_us:.0f}-end us",
        waveform_provenance=provenance,
        qualified=is_qualified,
    )


def multi_shot_uncertainty(
    device_name: str = "PF-1000",
) -> MultiShotUncertainty:
    """Estimate experimental uncertainty from published shot-to-shot data.

    Combines three independent uncertainty sources in quadrature:
    1. Shot-to-shot variability (from published data)
    2. Rogowski coil calibration uncertainty
    3. Waveform digitization uncertainty

    Also computes the reduced uncertainty from averaging multiple shots.

    Args:
        device_name: Registered device name.

    Returns:
        :class:`MultiShotUncertainty`.

    Raises:
        KeyError: If no shot-to-shot data available for device.
    """
    if device_name not in _SHOT_TO_SHOT_DATA:
        raise KeyError(
            f"No shot-to-shot data for '{device_name}'. "
            f"Available: {list(_SHOT_TO_SHOT_DATA.keys())}"
        )

    data = _SHOT_TO_SHOT_DATA[device_name]
    u_shot = data["u_shot_to_shot"]
    u_rog = data["u_rogowski"]
    u_dig = data["u_amplitude"]
    n_shots = data["n_shots_typical"]

    # Combined uncertainty (RSS)
    u_combined = float(np.sqrt(u_shot**2 + u_rog**2 + u_dig**2))

    # Reduced by averaging n_shots (only shot-to-shot component reduces)
    u_shot_avg = u_shot / np.sqrt(n_shots)
    u_with_avg = float(np.sqrt(u_shot_avg**2 + u_rog**2 + u_dig**2))

    logger.info(
        "Multi-shot %s: u_shot=%.1f%%, u_rog=%.1f%%, u_dig=%.1f%%, "
        "u_combined=%.1f%%, u_avg(%d shots)=%.1f%%",
        device_name, u_shot * 100, u_rog * 100, u_dig * 100,
        u_combined * 100, n_shots, u_with_avg * 100,
    )

    return MultiShotUncertainty(
        u_shot_to_shot=u_shot,
        u_rogowski=u_rog,
        u_amplitude=u_dig,
        u_exp_combined=u_combined,
        n_shots_typical=n_shots,
        u_exp_with_averaging=u_with_avg,
        reference=data["reference"],
    )
