"""Core LeeModelCalibrator class and calibrate_default_params utility."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dpf.validation._calibration_data import (
    _DEFAULT_CROWBAR_R,
    _DEFAULT_DEVICE_PCF,
    _PUBLISHED_FC_FM_RANGES,
    CalibrationResult,
)

logger = logging.getLogger(__name__)


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
        f_mr: float | None = None,
        pinch_column_fraction: float = 1.0,
        crowbar_enabled: bool = False,
        crowbar_resistance: float = 0.0,
    ) -> None:
        self.device_name = device_name
        self.method = method
        self.peak_weight = peak_weight
        self.timing_weight = timing_weight
        self.waveform_weight = waveform_weight
        self.f_mr = f_mr
        self.pinch_column_fraction = pinch_column_fraction
        self.crowbar_enabled = crowbar_enabled
        self.crowbar_resistance = crowbar_resistance
        self._n_evals = 0

    def calibrate(
        self,
        fc_bounds: tuple[float, float] = (0.6, 0.8),
        fm_bounds: tuple[float, float] = (0.05, 0.25),
        maxiter: int = 100,
        x0: tuple[float, float] | None = None,
    ) -> CalibrationResult:
        """Find optimal fc, fm that minimize validation error.

        Args:
            fc_bounds: Bounds for current fraction (fc).
            fm_bounds: Bounds for mass fraction (fm). Default tightened
                to ``(0.05, 0.25)`` per Lee & Saw (2014) published range.
            maxiter: Maximum optimizer iterations.
            x0: Initial guess ``(fc, fm)``. Default: midpoint of bounds.

        Returns:
            :class:`CalibrationResult` with optimized parameters.
        """
        from scipy.optimize import Bounds, minimize

        if x0 is None:
            x0_arr = np.array([
                0.5 * (fc_bounds[0] + fc_bounds[1]),
                0.5 * (fm_bounds[0] + fm_bounds[1]),
            ])
        else:
            x0_arr = np.array(x0)

        self._n_evals = 0
        self._fc_bounds = fc_bounds
        self._fm_bounds = fm_bounds

        result = minimize(
            self._objective,
            x0_arr,
            method=self.method,
            bounds=Bounds(
                [fc_bounds[0], fm_bounds[0]],
                [fc_bounds[1], fm_bounds[1]],
            ),
            options={"maxiter": maxiter, "xatol": 0.005, "fatol": 0.001},
        )

        # Evaluate at the optimum to get error components
        fc_opt = float(np.clip(result.x[0], *fc_bounds))
        fm_opt = float(np.clip(result.x[1], *fm_bounds))
        comparison = self._run_comparison(fc_opt, fm_opt, f_mr=self.f_mr)

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
            comparison = self._run_comparison(fc, fm, f_mr=self.f_mr)
        except (RuntimeError, ValueError, FloatingPointError):
            logger.debug("Objective evaluation failed for fc=%.3f, fm=%.3f", fc, fm)
            return 10.0  # Large penalty for failed runs

        # Check if waveform NRMSE is available
        nrmse = getattr(comparison, "waveform_nrmse", float("nan"))
        has_waveform = isinstance(nrmse, (int, float)) and np.isfinite(nrmse)

        if has_waveform and self.waveform_weight > 0:
            obj = (
                self.peak_weight * comparison.peak_current_error
                + self.timing_weight * comparison.timing_error
                + self.waveform_weight * nrmse
            )
        else:
            # Renormalize weights when waveform unavailable (MED-3)
            total = self.peak_weight + self.timing_weight
            if total > 0:
                obj = (
                    (self.peak_weight / total) * comparison.peak_current_error
                    + (self.timing_weight / total) * comparison.timing_error
                )
            else:
                obj = comparison.peak_current_error
        return obj

    def benchmark_against_published(
        self,
        calibration_result: CalibrationResult | None = None,
        maxiter: int = 100,
    ) -> dict[str, object]:
        """Compare calibrated fc/fm against published Lee & Saw (2014) values.

        Runs calibration (or accepts a pre-run result) and checks whether the
        optimal fc/fm fall within the published ranges from Lee & Saw (2014)
        Table 1.  These published values were obtained by fitting Lee model
        simulations to experimental I(t) waveforms for each device.

        Args:
            calibration_result: Pre-computed :class:`CalibrationResult` to
                benchmark. If ``None``, runs calibration with *maxiter*.
            maxiter: Optimizer iterations if *calibration_result* is ``None``.

        Returns:
            Dict with keys:

            ``"fc_calibrated"``
                Calibrated current fraction.
            ``"fm_calibrated"``
                Calibrated mass fraction.
            ``"fc_published_range"``
                Tuple ``(lo, hi)`` from Lee & Saw (2014).
            ``"fm_published_range"``
                Tuple ``(lo, hi)`` from Lee & Saw (2014).
            ``"fc_in_range"``
                ``True`` if ``fc_calibrated`` is within the published range.
            ``"fm_in_range"``
                ``True`` if ``fm_calibrated`` is within the published range.
            ``"both_in_range"``
                ``True`` if both fc and fm are within published ranges.
            ``"reference"``
                Citation string for the published values.

        Raises:
            KeyError: If device has no published fc/fm range.
        """
        if self.device_name not in _PUBLISHED_FC_FM_RANGES:
            raise KeyError(
                f"No published fc/fm range for device '{self.device_name}'. "
                f"Available: {list(_PUBLISHED_FC_FM_RANGES.keys())}"
            )

        if calibration_result is None:
            calibration_result = self.calibrate(maxiter=maxiter)

        ranges = _PUBLISHED_FC_FM_RANGES[self.device_name]
        fc_lo, fc_hi = ranges["fc"]
        fm_lo, fm_hi = ranges["fm"]

        fc_cal = calibration_result.best_fc
        fm_cal = calibration_result.best_fm

        fc_in = fc_lo <= fc_cal <= fc_hi
        fm_in = fm_lo <= fm_cal <= fm_hi

        logger.info(
            "Benchmark %s: fc=%.3f (published [%.2f, %.2f] → %s), "
            "fm=%.3f (published [%.2f, %.2f] → %s)",
            self.device_name,
            fc_cal, fc_lo, fc_hi, "IN" if fc_in else "OUT",
            fm_cal, fm_lo, fm_hi, "IN" if fm_in else "OUT",
        )

        return {
            "fc_calibrated": fc_cal,
            "fm_calibrated": fm_cal,
            "fc_published_range": (fc_lo, fc_hi),
            "fm_published_range": (fm_lo, fm_hi),
            "fc_in_range": fc_in,
            "fm_in_range": fm_in,
            "both_in_range": fc_in and fm_in,
            "reference": "S. Lee & S.H. Saw, J. Fusion Energy 33:319-335 (2014)",
        }

    def _run_comparison(self, fc: float, fm: float, f_mr: float | None = None) -> Any:
        """Run LeeModel and compare against experiment.

        Args:
            fc: Current fraction.
            fm: Mass fraction.
            f_mr: Radial mass fraction (defaults to fm if None).

        Returns:
            :class:`LeeModelComparison` instance.
        """
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(
            current_fraction=fc,
            mass_fraction=fm,
            radial_mass_fraction=f_mr,
            pinch_column_fraction=self.pinch_column_fraction,
            crowbar_enabled=self.crowbar_enabled,
            crowbar_resistance=self.crowbar_resistance,
        )
        return model.compare_with_experiment(self.device_name)


def calibrate_default_params(
    devices: list[str] | None = None,
    maxiter: int = 100,
) -> dict[str, CalibrationResult]:
    """Run calibration for multiple devices with default settings.

    Uses device-specific pinch_column_fraction values where known.

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
            pcf = _DEFAULT_DEVICE_PCF.get(dev, 1.0)
            # Enable crowbar with device-specific resistance for PF-1000
            # PhD Debate #30: R_crowbar=0 biases fc upward during calibration
            crowbar_r = _DEFAULT_CROWBAR_R.get(dev, 0.0)
            cal = LeeModelCalibrator(
                dev,
                pinch_column_fraction=pcf,
                crowbar_enabled=crowbar_r > 0,
                crowbar_resistance=crowbar_r,
            )
            results[dev] = cal.calibrate(maxiter=maxiter)
        except Exception as exc:
            logger.warning("Calibration failed for %s: %s", dev, exc)

    return results
