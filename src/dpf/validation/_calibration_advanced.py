"""Advanced calibration: circuit-only, liftoff delay, and blind prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dpf.validation._calibration_asme import (
    ASMEValidationResult,
    _pinch_phase_asme,
    asme_vv20_assessment,
)
from dpf.validation._calibration_core import LeeModelCalibrator
from dpf.validation._calibration_data import _DEFAULT_CROWBAR_R, _DEFAULT_DEVICE_PCF
from dpf.validation._calibration_stats import MonteCarloNRMSEResult

logger = logging.getLogger(__name__)


@dataclass
class CircuitOnlyCalibrationResult:
    """Result of circuit-window-only calibration with blind pinch prediction.

    Calibrates fc/fm using only the circuit phase (0 to circuit_window_us),
    then evaluates the pinch phase as a genuine blind prediction.  This
    converts the ASME assessment from "calibration residual" to true
    validation for the pinch phase.

    Attributes:
        device_name: Device name.
        circuit_window_us: End of circuit window [us].
        best_fc: Optimal current fraction (from circuit-only calibration).
        best_fm: Optimal mass fraction (from circuit-only calibration).
        n_evals: Number of objective evaluations.
        converged: Whether the optimizer converged.
        circuit_asme: ASME assessment for the circuit window (calibration).
        pinch_asme: ASME assessment for the pinch phase (blind prediction).
        full_asme: ASME assessment for the full waveform.
        circuit_nrmse: NRMSE in the circuit window.
        pinch_nrmse: NRMSE in the pinch phase (blind prediction).
        full_nrmse: NRMSE for the full waveform.
        nrmse_ratio: pinch_nrmse / circuit_nrmse — amplification factor.
        standard_fc: fc from standard full-waveform calibration (for comparison).
        standard_fm: fm from standard full-waveform calibration.
    """

    device_name: str
    circuit_window_us: float
    best_fc: float
    best_fm: float
    n_evals: int
    converged: bool
    circuit_asme: ASMEValidationResult
    pinch_asme: ASMEValidationResult | None
    full_asme: ASMEValidationResult
    circuit_nrmse: float
    pinch_nrmse: float | None
    full_nrmse: float
    nrmse_ratio: float | None
    standard_fc: float
    standard_fm: float


@dataclass
class LiftoffCalibrationResult:
    """Result of 3-parameter (fc, fm, liftoff_delay) calibration.

    Extends standard 2-parameter calibration by optimizing the insulator
    flashover delay, which shifts the simulation time origin.  This separates
    timing error from amplitude error, often reducing NRMSE by 30-50%.

    The liftoff delay represents the time between capacitor bank discharge
    and insulator flashover (breakdown across the insulator surface).
    For MJ-class devices this is typically 0.5-1.5 us (Lee 2005).

    Attributes:
        device_name: Device name.
        best_fc: Optimal current fraction.
        best_fm: Optimal mass fraction.
        best_delay_us: Optimal liftoff delay [us].
        nrmse: Full waveform NRMSE at optimum.
        asme: ASME V&V 20 assessment at optimum.
        n_evals: Number of objective evaluations.
        converged: Whether the optimizer converged.
        standard_fc: fc from standard 2-parameter calibration.
        standard_fm: fm from standard 2-parameter calibration.
        standard_nrmse: NRMSE from standard 2-parameter calibration.
        standard_asme: ASME from standard 2-parameter calibration.
        nrmse_improvement: Fractional NRMSE reduction vs standard.
        delta_model: Model-form error sqrt(E^2 - u_val^2).
    """

    device_name: str
    best_fc: float
    best_fm: float
    best_delay_us: float
    nrmse: float
    asme: ASMEValidationResult
    n_evals: int
    converged: bool
    standard_fc: float
    standard_fm: float
    standard_nrmse: float
    standard_asme: ASMEValidationResult
    nrmse_improvement: float
    delta_model: float


@dataclass
class BlindPredictionResult:
    """Result of a blind prediction: calibrate on one condition, predict another.

    Attributes
    ----------
    train_device : str
        Device/condition used for calibration.
    test_device : str
        Device/condition used for blind prediction.
    train_fc, train_fm, train_delay_us : float
        Calibrated parameters from training device.
    train_nrmse : float
        NRMSE on training device (calibration residual).
    test_asme : ASMEValidationResult
        ASME assessment on test device (blind prediction).
    test_nrmse : float
        NRMSE on test device (blind prediction error).
    peak_current_error : float
        Relative error in predicted vs measured peak current.
    """

    train_device: str
    test_device: str
    train_fc: float
    train_fm: float
    train_delay_us: float
    train_nrmse: float
    test_asme: ASMEValidationResult
    test_nrmse: float
    peak_current_error: float


def circuit_only_calibration(
    device_name: str = "PF-1000",
    circuit_window_us: float = 6.0,
    fc_bounds: tuple[float, float] = (0.6, 0.8),
    fm_bounds: tuple[float, float] = (0.05, 0.25),
    maxiter: int = 100,
    pinch_column_fraction: float | None = None,
    crowbar_enabled: bool | None = None,
    crowbar_resistance: float | None = None,
) -> CircuitOnlyCalibrationResult:
    """Calibrate fc/fm on circuit window only, blind-predict pinch phase.

    This is the key insight from PhD Debate #38 path-to-7.0: if we calibrate
    fc/fm using only the 0-6 us circuit phase, then the pinch-phase NRMSE
    becomes a genuine blind prediction rather than a calibration residual.
    This transforms the ASME assessment from Section 5.1 (calibration) to
    Section 5.3 (validation) compliance.

    Args:
        device_name: Registered device name.
        circuit_window_us: End of circuit calibration window [us].
        fc_bounds: Bounds for current fraction.
        fm_bounds: Bounds for mass fraction.
        maxiter: Maximum optimizer iterations.
        pinch_column_fraction: Pinch column fraction.  Uses device default
            if None.
        crowbar_enabled: Whether crowbar is enabled.  Auto-detected if None.
        crowbar_resistance: Crowbar resistance [Ohm].  Auto-detected if None.

    Returns:
        :class:`CircuitOnlyCalibrationResult` with calibration and blind
        prediction metrics.
    """
    from scipy.optimize import Bounds, minimize

    from dpf.validation.experimental import DEVICES, nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    # Device defaults
    pcf = pinch_column_fraction
    if pcf is None:
        pcf = _DEFAULT_DEVICE_PCF.get(device_name, 1.0)
    if crowbar_enabled is None:
        cr = _DEFAULT_CROWBAR_R.get(device_name, 0.0)
        crowbar_enabled = cr > 0
        if crowbar_resistance is None:
            crowbar_resistance = cr
    if crowbar_resistance is None:
        crowbar_resistance = 0.0

    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        raise ValueError(f"No digitized waveform for {device_name}")

    circuit_max_time = circuit_window_us * 1e-6
    n_evals = 0

    def _circuit_objective(params: np.ndarray) -> float:
        """Objective: NRMSE in circuit window only."""
        nonlocal n_evals
        n_evals += 1

        fc = float(np.clip(params[0], *fc_bounds))
        fm = float(np.clip(params[1], *fm_bounds))

        try:
            model = LeeModel(
                current_fraction=fc,
                mass_fraction=fm,
                radial_mass_fraction=fm,
                pinch_column_fraction=pcf,
                crowbar_enabled=crowbar_enabled,
                crowbar_resistance=crowbar_resistance,
            )
            result = model.run(device_name)

            # NRMSE in circuit window only
            nrmse = nrmse_peak(
                result.t, result.I,
                device.waveform_t, device.waveform_I,
                max_time=circuit_max_time,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            return 10.0

        return nrmse

    # Run circuit-only optimization
    x0 = np.array([
        0.5 * (fc_bounds[0] + fc_bounds[1]),
        0.5 * (fm_bounds[0] + fm_bounds[1]),
    ])

    opt_result = minimize(
        _circuit_objective,
        x0,
        method="nelder-mead",
        bounds=Bounds(
            [fc_bounds[0], fm_bounds[0]],
            [fc_bounds[1], fm_bounds[1]],
        ),
        options={"maxiter": maxiter, "xatol": 0.005, "fatol": 0.001},
    )

    fc_cir = float(np.clip(opt_result.x[0], *fc_bounds))
    fm_cir = float(np.clip(opt_result.x[1], *fm_bounds))

    logger.info(
        "Circuit-only calibration %s (0-%.0f us): fc=%.3f, fm=%.3f, "
        "NRMSE_circuit=%.4f, n_evals=%d, converged=%s",
        device_name, circuit_window_us, fc_cir, fm_cir,
        float(opt_result.fun), n_evals, opt_result.success,
    )

    # --- Evaluate at circuit-only optimum ---

    # Circuit-window ASME
    circuit_asme = asme_vv20_assessment(
        device_name=device_name, fc=fc_cir, fm=fm_cir,
        f_mr=fm_cir, pinch_column_fraction=pcf,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
        max_time=circuit_max_time,
    )

    # Pinch-phase ASME (blind prediction)
    pinch_asme = _pinch_phase_asme(
        device_name=device_name, fc=fc_cir, fm=fm_cir,
        f_mr=fm_cir, pinch_column_fraction=pcf,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
        t_start_us=circuit_window_us,
    )

    # Full-waveform ASME (for comparison)
    full_asme = asme_vv20_assessment(
        device_name=device_name, fc=fc_cir, fm=fm_cir,
        f_mr=fm_cir, pinch_column_fraction=pcf,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
    )

    # Also run standard full-waveform calibration for comparison
    std_cal = LeeModelCalibrator(
        device_name,
        pinch_column_fraction=pcf,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
    )
    std_result = std_cal.calibrate(
        fc_bounds=fc_bounds, fm_bounds=fm_bounds, maxiter=maxiter,
    )

    pinch_nrmse = pinch_asme.E if pinch_asme else None
    circuit_nrmse = circuit_asme.E
    full_nrmse = full_asme.E
    nrmse_ratio = (pinch_nrmse / circuit_nrmse) if (
        pinch_nrmse is not None and circuit_nrmse > 0
    ) else None

    logger.info(
        "Circuit-only result %s: circuit_NRMSE=%.3f, pinch_NRMSE=%s, "
        "full_NRMSE=%.3f, ratio=%.2f, standard fc=%.3f/fm=%.3f",
        device_name, circuit_nrmse,
        f"{pinch_nrmse:.3f}" if pinch_nrmse is not None else "N/A",
        full_nrmse,
        nrmse_ratio if nrmse_ratio is not None else float("nan"),
        std_result.best_fc, std_result.best_fm,
    )

    return CircuitOnlyCalibrationResult(
        device_name=device_name,
        circuit_window_us=circuit_window_us,
        best_fc=fc_cir,
        best_fm=fm_cir,
        n_evals=n_evals,
        converged=bool(opt_result.success),
        circuit_asme=circuit_asme,
        pinch_asme=pinch_asme,
        full_asme=full_asme,
        circuit_nrmse=circuit_nrmse,
        pinch_nrmse=pinch_nrmse,
        full_nrmse=full_nrmse,
        nrmse_ratio=nrmse_ratio,
        standard_fc=std_result.best_fc,
        standard_fm=std_result.best_fm,
    )


def calibrate_with_liftoff(
    device_name: str = "PF-1000",
    fc_bounds: tuple[float, float] = (0.5, 0.95),
    fm_bounds: tuple[float, float] = (0.01, 0.3),
    delay_bounds_us: tuple[float, float] = (0.0, 2.0),
    pinch_column_fraction: float | None = None,
    crowbar_enabled: bool | None = None,
    crowbar_resistance: float | None = None,
    maxiter: int = 200,
    include_shot_to_shot: bool = True,
    mc_result: MonteCarloNRMSEResult | None = None,
    seed: int = 42,
) -> LiftoffCalibrationResult:
    """Three-parameter calibration: fc, fm, and insulator liftoff delay.

    Optimizes (fc, fm, liftoff_delay) jointly by minimizing NRMSE against
    experimental I(t) data.  The liftoff delay shifts the simulation time
    origin to account for insulator flashover delay.

    This typically reduces NRMSE by 30-50% compared to 2-parameter
    calibration because it separates timing error from amplitude error.

    Args:
        device_name: Device to calibrate against.
        fc_bounds: Bounds for current fraction (fc).
        fm_bounds: Bounds for mass fraction (fm).
        delay_bounds_us: Bounds for liftoff delay [us].
        pinch_column_fraction: Pinch column fraction.  If None, uses
            device-specific default from ``_DEFAULT_DEVICE_PCF``.
        crowbar_enabled: Whether crowbar is enabled.  If None, auto-detects
            from ``_DEFAULT_CROWBAR_R`` (enabled only for devices with a
            known crowbar resistance).
        crowbar_resistance: Crowbar resistance [Ohm].  If None, uses
            device-specific default from ``_DEFAULT_CROWBAR_R``.
        maxiter: Maximum optimizer iterations.
        include_shot_to_shot: Include shot-to-shot uncertainty in ASME.
        mc_result: Pre-computed Monte Carlo result for u_input.  If None,
            uses default u_input=0.027 from Phase AS.  Pass result from
            ``monte_carlo_nrmse(liftoff_delay=...)`` to include delay
            uncertainty in u_val (PhD Debate #40 recommendation).
        seed: Random seed for differential evolution optimizer.

    Returns:
        :class:`LiftoffCalibrationResult` with optimized parameters and
        comparison against standard 2-parameter calibration.
    """
    from dpf.validation.experimental import DEVICES, nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    # Resolve device-specific defaults (matching calibrate_default_params)
    if pinch_column_fraction is None:
        pinch_column_fraction = _DEFAULT_DEVICE_PCF.get(device_name, 0.14)
    if crowbar_resistance is None:
        crowbar_resistance = _DEFAULT_CROWBAR_R.get(device_name, 0.0)
    if crowbar_enabled is None:
        crowbar_enabled = crowbar_resistance > 0

    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        raise ValueError(f"No digitized waveform for {device_name}")

    n_evals = 0

    def _objective(x: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        fc_try, fm_try, delay_us = float(x[0]), float(x[1]), float(x[2])
        delay_s = delay_us * 1e-6
        try:
            model = LeeModel(
                current_fraction=fc_try,
                mass_fraction=fm_try,
                pinch_column_fraction=pinch_column_fraction,
                crowbar_enabled=crowbar_enabled,
                crowbar_resistance=crowbar_resistance,
                liftoff_delay=delay_s,
            )
            result = model.run(device_name)
            return float(nrmse_peak(
                result.t, result.I, device.waveform_t, device.waveform_I,
            ))
        except Exception:
            return 1.0

    # Use differential evolution for global optimization over 3-parameter
    # space.  The landscape has ridges (fc^2/fm degeneracy) that trap local
    # optimizers.  DE explores the full bounds.
    # NOTE: polish=False because scipy's built-in L-BFGS-B polish uses
    # default maxiter=15000*ndim which can take hours on noisy Lee model
    # objectives (~0.2s/eval).  Instead we do a bounded manual polish.
    from scipy.optimize import differential_evolution, minimize

    de_bounds = [fc_bounds, fm_bounds, delay_bounds_us]
    opt = differential_evolution(
        _objective, de_bounds, maxiter=maxiter, seed=seed,
        tol=1e-5, atol=1e-5, polish=False, workers=1,
    )

    # Bounded L-BFGS-B polish: cap at 50 iterations to avoid runaway
    # convergence on noisy/flat objectives (e.g. UNU-ICTP with wrong pcf).
    polish = minimize(
        _objective, opt.x, method="L-BFGS-B",
        bounds=de_bounds, options={"maxiter": 50},
    )
    if polish.fun <= opt.fun:
        opt_x = polish.x
    else:
        opt_x = opt.x

    fc_opt, fm_opt, delay_opt_us = (
        float(opt_x[0]), float(opt_x[1]), float(opt_x[2])
    )
    # Clamp to bounds
    fc_opt = float(np.clip(fc_opt, *fc_bounds))
    fm_opt = float(np.clip(fm_opt, *fm_bounds))
    delay_opt_us = float(np.clip(delay_opt_us, *delay_bounds_us))
    delay_opt_s = delay_opt_us * 1e-6

    # ASME assessment with optimized liftoff delay
    asme_opt = asme_vv20_assessment(
        device_name, fc=fc_opt, fm=fm_opt,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
        liftoff_delay=delay_opt_s,
        include_shot_to_shot=include_shot_to_shot,
        mc_result=mc_result,
    )

    # Standard 2-parameter calibration for comparison.
    # Uses the SAME fc_bounds as the 3-param optimization to avoid
    # the bound asymmetry confound identified in PhD Debate #40.
    std_cal = LeeModelCalibrator(
        device_name,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
    )
    std_result = std_cal.calibrate(
        fc_bounds=fc_bounds,
        fm_bounds=fm_bounds,
        maxiter=maxiter,
    )
    std_asme = asme_vv20_assessment(
        device_name, fc=std_result.best_fc, fm=std_result.best_fm,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
        include_shot_to_shot=include_shot_to_shot,
        mc_result=mc_result,
    )

    nrmse_opt = asme_opt.E
    nrmse_std = std_asme.E
    improvement = (nrmse_std - nrmse_opt) / nrmse_std if nrmse_std > 0 else 0.0

    # Model-form error: delta_model = sqrt(E^2 - u_val^2) if E > u_val
    if nrmse_opt > asme_opt.u_val:
        delta = float(np.sqrt(nrmse_opt**2 - asme_opt.u_val**2))
    else:
        delta = 0.0

    logger.info(
        "3-param calibration %s: fc=%.4f, fm=%.4f, delay=%.3f us, "
        "NRMSE=%.4f (was %.4f, improvement=%.1f%%), "
        "ASME ratio=%.3f, delta_model=%.4f",
        device_name, fc_opt, fm_opt, delay_opt_us,
        nrmse_opt, nrmse_std, improvement * 100,
        asme_opt.ratio, delta,
    )

    return LiftoffCalibrationResult(
        device_name=device_name,
        best_fc=fc_opt,
        best_fm=fm_opt,
        best_delay_us=delay_opt_us,
        nrmse=nrmse_opt,
        asme=asme_opt,
        n_evals=n_evals,
        converged=bool(opt.success),
        standard_fc=std_result.best_fc,
        standard_fm=std_result.best_fm,
        standard_nrmse=nrmse_std,
        standard_asme=std_asme,
        nrmse_improvement=improvement,
        delta_model=delta,
    )


def blind_predict(
    train_device: str = "PF-1000",
    test_device: str = "PF-1000-16kV",
    fc_bounds: tuple[float, float] = (0.6, 0.80),
    fm_bounds: tuple[float, float] = (0.10, 0.30),
    delay_bounds_us: tuple[float, float] = (0.0, 2.0),
    pinch_column_fraction: float | None = None,
    crowbar_enabled: bool | None = None,
    crowbar_resistance: float | None = None,
    maxiter: int = 200,
) -> BlindPredictionResult:
    """Calibrate on train_device, blind-predict on test_device.

    This satisfies ASME V&V 20-2009 Section 5.3: the test_device waveform
    is NEVER used during calibration.  The prediction is genuinely blind.

    Args:
        train_device: Device for calibration (provides fc, fm, delay).
        test_device: Device for blind prediction (provides reference waveform).
        fc_bounds: Bounds for current fraction.
        fm_bounds: Bounds for mass fraction (physical range).
        delay_bounds_us: Bounds for liftoff delay [us].
        pinch_column_fraction: Pinch column fraction.
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        maxiter: Maximum optimizer iterations.

    Returns:
        :class:`BlindPredictionResult` with calibration and prediction metrics.
    """
    from dpf.validation.experimental import DEVICES
    from dpf.validation.lee_model_comparison import LeeModel

    # Resolve device-specific defaults for test device
    test_pcf = _DEFAULT_DEVICE_PCF.get(test_device, 0.14)
    if pinch_column_fraction is not None:
        test_pcf = pinch_column_fraction
    test_cr = _DEFAULT_CROWBAR_R.get(test_device, 0.0)
    if crowbar_resistance is not None:
        test_cr = crowbar_resistance
    test_cb = test_cr > 0
    if crowbar_enabled is not None:
        test_cb = crowbar_enabled

    # Step 1: Calibrate on training device (uses its own device-specific defaults)
    cal = calibrate_with_liftoff(
        device_name=train_device,
        fc_bounds=fc_bounds,
        fm_bounds=fm_bounds,
        delay_bounds_us=delay_bounds_us,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
        maxiter=maxiter,
    )

    # Step 2: Blind prediction on test device (NO re-fitting)
    test_asme = asme_vv20_assessment(
        device_name=test_device,
        fc=cal.best_fc,
        fm=cal.best_fm,
        pinch_column_fraction=test_pcf,
        crowbar_enabled=test_cb,
        crowbar_resistance=test_cr,
        liftoff_delay=cal.best_delay_us * 1e-6,
    )

    # Step 3: Peak current comparison
    test_dev = DEVICES[test_device]
    model = LeeModel(
        current_fraction=cal.best_fc,
        mass_fraction=cal.best_fm,
        pinch_column_fraction=test_pcf,
        crowbar_enabled=test_cb,
        crowbar_resistance=test_cr,
        liftoff_delay=cal.best_delay_us * 1e-6,
    )
    pred = model.run(test_device)
    predicted_peak = float(np.max(pred.I))
    measured_peak = float(test_dev.peak_current)
    peak_error = abs(predicted_peak - measured_peak) / measured_peak

    logger.info(
        "Blind prediction %s → %s: NRMSE=%.4f, peak error=%.1f%%, "
        "ASME ratio=%.3f (train: fc=%.3f, fm=%.3f, delay=%.3f us)",
        train_device, test_device, test_asme.E, peak_error * 100,
        test_asme.ratio, cal.best_fc, cal.best_fm, cal.best_delay_us,
    )

    return BlindPredictionResult(
        train_device=train_device,
        test_device=test_device,
        train_fc=cal.best_fc,
        train_fm=cal.best_fm,
        train_delay_us=cal.best_delay_us,
        train_nrmse=cal.nrmse,
        test_asme=test_asme,
        test_nrmse=test_asme.E,
        peak_current_error=peak_error,
    )
