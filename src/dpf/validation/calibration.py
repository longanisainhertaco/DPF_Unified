"""Automated calibration of Lee model fc/fm against experimental data.

Uses scipy.optimize to find the (fc, fm) pair that minimizes the combined
relative error in peak current and timing against published experimental
measurements for a given DPF device.

Usage::

    from dpf.validation.calibration import LeeModelCalibrator

    cal = LeeModelCalibrator("PF-1000")
    result = cal.calibrate()
    print(f"Best fc={result.best_fc:.3f}, fm={result.best_fm:.3f}")

References
----------
- S. Lee & S. H. Saw, J. Fusion Energy **27**, 292-295 (2008).
- S. Lee, J. Fusion Energy **33**, 319-335 (2014).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of an fc/fm calibration run.

    Attributes:
        best_fc: Optimal current fraction.
        best_fm: Optimal mass fraction.
        peak_current_error: Relative error in peak current at optimum.
        timing_error: Relative error in peak timing at optimum.
        objective_value: Final objective function value.
        n_evals: Number of objective evaluations.
        converged: Whether the optimizer reported convergence.
        device_name: Device name that was calibrated.
    """

    best_fc: float
    best_fm: float
    peak_current_error: float
    timing_error: float
    objective_value: float
    n_evals: int
    converged: bool
    device_name: str = ""


class LeeModelCalibrator:
    """Automated calibration of Lee model fc/fm against experimental data.

    Args:
        device_name: Registered device name (e.g. ``"PF-1000"``, ``"NX2"``).
        method: Optimization method for ``scipy.optimize.minimize``.
            Default: ``"nelder-mead"`` (derivative-free, robust for noisy
            objective).
        peak_weight: Weight for peak current error in objective (default 0.4).
        timing_weight: Weight for timing error in objective (default 0.3).
        waveform_weight: Weight for waveform NRMSE in objective (default 0.3).
            This provides 1+ DOF when a digitized waveform is available
            (3 metrics vs 2 parameters).
    """

    def __init__(
        self,
        device_name: str,
        method: str = "nelder-mead",
        peak_weight: float = 0.4,
        timing_weight: float = 0.3,
        waveform_weight: float = 0.3,
    ) -> None:
        self.device_name = device_name
        self.method = method
        self.peak_weight = peak_weight
        self.timing_weight = timing_weight
        self.waveform_weight = waveform_weight
        self._n_evals = 0

    def calibrate(
        self,
        fc_bounds: tuple[float, float] = (0.5, 0.95),
        fm_bounds: tuple[float, float] = (0.05, 0.95),
        maxiter: int = 100,
        x0: tuple[float, float] | None = None,
    ) -> CalibrationResult:
        """Find optimal fc, fm that minimize validation error.

        Args:
            fc_bounds: Bounds for current fraction (fc).
            fm_bounds: Bounds for mass fraction (fm).
            maxiter: Maximum optimizer iterations.
            x0: Initial guess ``(fc, fm)``. Default: midpoint of bounds.

        Returns:
            :class:`CalibrationResult` with optimized parameters.
        """
        from scipy.optimize import minimize

        if x0 is None:
            x0_arr = np.array([
                0.5 * (fc_bounds[0] + fc_bounds[1]),
                0.5 * (fm_bounds[0] + fm_bounds[1]),
            ])
        else:
            x0_arr = np.array(x0)

        self._n_evals = 0

        # Nelder-Mead doesn't support bounds directly; clamp inside objective
        self._fc_bounds = fc_bounds
        self._fm_bounds = fm_bounds

        result = minimize(
            self._objective,
            x0_arr,
            method=self.method,
            options={"maxiter": maxiter, "xatol": 0.005, "fatol": 0.001},
        )

        # Evaluate at the optimum to get error components
        fc_opt = float(np.clip(result.x[0], *fc_bounds))
        fm_opt = float(np.clip(result.x[1], *fm_bounds))
        comparison = self._run_comparison(fc_opt, fm_opt)

        logger.info(
            "Calibration %s: fc=%.3f, fm=%.3f, peak_err=%.1f%%, "
            "timing_err=%.1f%%, n_evals=%d, converged=%s",
            self.device_name, fc_opt, fm_opt,
            comparison.peak_current_error * 100,
            comparison.timing_error * 100,
            self._n_evals, result.success,
        )

        return CalibrationResult(
            best_fc=fc_opt,
            best_fm=fm_opt,
            peak_current_error=comparison.peak_current_error,
            timing_error=comparison.timing_error,
            objective_value=float(result.fun),
            n_evals=self._n_evals,
            converged=bool(result.success),
            device_name=self.device_name,
        )

    def _objective(self, params: np.ndarray) -> float:
        """Run LeeModel with (fc, fm), return weighted error.

        Args:
            params: Array ``[fc, fm]``.

        Returns:
            Weighted sum of relative errors.
        """
        self._n_evals += 1

        fc = float(np.clip(params[0], *self._fc_bounds))
        fm = float(np.clip(params[1], *self._fm_bounds))

        try:
            comparison = self._run_comparison(fc, fm)
        except Exception:
            logger.debug("Objective evaluation failed for fc=%.3f, fm=%.3f", fc, fm)
            return 10.0  # Large penalty for failed runs

        obj = (
            self.peak_weight * comparison.peak_current_error
            + self.timing_weight * comparison.timing_error
        )
        # Add waveform NRMSE term if available (provides 1+ DOF)
        if self.waveform_weight > 0:
            nrmse = getattr(comparison, "waveform_nrmse", float("nan"))
            if isinstance(nrmse, (int, float)) and np.isfinite(nrmse):
                obj += self.waveform_weight * nrmse
        return obj

    def _run_comparison(self, fc: float, fm: float) -> Any:
        """Run LeeModel and compare against experiment.

        Args:
            fc: Current fraction.
            fm: Mass fraction.

        Returns:
            :class:`LeeModelComparison` instance.
        """
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=fc, mass_fraction=fm)
        return model.compare_with_experiment(self.device_name)


def calibrate_default_params(
    devices: list[str] | None = None,
    maxiter: int = 100,
) -> dict[str, CalibrationResult]:
    """Run calibration for multiple devices with default settings.

    Args:
        devices: Device names to calibrate. Default: ``["PF-1000", "NX2"]``.
        maxiter: Maximum optimizer iterations per device.

    Returns:
        Dict mapping device name to :class:`CalibrationResult`.
    """
    if devices is None:
        devices = ["PF-1000", "NX2"]

    results: dict[str, CalibrationResult] = {}
    for dev in devices:
        try:
            cal = LeeModelCalibrator(dev)
            results[dev] = cal.calibrate(maxiter=maxiter)
        except Exception as exc:
            logger.warning("Calibration failed for %s: %s", dev, exc)

    return results


# =====================================================================
# Cross-validation framework
# =====================================================================


@dataclass
class CrossValidationResult:
    """Result of cross-device validation.

    Calibrate fc/fm on *train_device*, then predict on *test_device*
    and measure generalization error.

    Attributes:
        train_device: Device used for calibration.
        test_device: Device used for prediction.
        calibration: Calibration result from train_device.
        prediction_peak_error: Relative peak current error on test_device.
        prediction_timing_error: Relative timing error on test_device.
        generalization_score: 1 - average prediction error (higher = better).
    """

    train_device: str
    test_device: str
    calibration: CalibrationResult
    prediction_peak_error: float
    prediction_timing_error: float
    generalization_score: float


class CrossValidator:
    """Cross-validate Lee model calibration across devices.

    Calibrates fc/fm on one device, then evaluates prediction quality
    on a different device.  This tests whether the calibrated parameters
    generalize across different DPF geometries and operating conditions.
    """

    def validate(
        self,
        train_device: str,
        test_device: str,
        maxiter: int = 100,
    ) -> CrossValidationResult:
        """Calibrate on train_device, predict on test_device.

        Args:
            train_device: Device name for calibration.
            test_device: Device name for prediction evaluation.
            maxiter: Maximum optimizer iterations.

        Returns:
            :class:`CrossValidationResult` with generalization metrics.
        """
        from dpf.validation.lee_model_comparison import LeeModel

        # Step 1: Calibrate on train device
        cal = LeeModelCalibrator(train_device)
        cal_result = cal.calibrate(maxiter=maxiter)

        # Step 2: Run prediction on test device with calibrated params
        model = LeeModel(
            current_fraction=cal_result.best_fc,
            mass_fraction=cal_result.best_fm,
        )
        comparison = model.compare_with_experiment(test_device)

        # Step 3: Compute generalization score
        avg_error = 0.5 * (
            comparison.peak_current_error + comparison.timing_error
        )
        generalization_score = max(1.0 - avg_error, 0.0)

        logger.info(
            "Cross-validation %s→%s: fc=%.3f, fm=%.3f, "
            "pred_peak_err=%.1f%%, pred_timing_err=%.1f%%, "
            "generalization=%.2f",
            train_device, test_device,
            cal_result.best_fc, cal_result.best_fm,
            comparison.peak_current_error * 100,
            comparison.timing_error * 100,
            generalization_score,
        )

        return CrossValidationResult(
            train_device=train_device,
            test_device=test_device,
            calibration=cal_result,
            prediction_peak_error=comparison.peak_current_error,
            prediction_timing_error=comparison.timing_error,
            generalization_score=generalization_score,
        )
