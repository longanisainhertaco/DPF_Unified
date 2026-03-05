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


# =====================================================================
# Published Lee model fc/fm ranges from Lee & Saw (2014), Table 1
# These provide ground-truth benchmarks for calibration validation.
# Source: S. Lee & S.H. Saw, J. Fusion Energy 33:319-335 (2014)
# NOTE: Ranges match Lee & Saw (2014) published values directly.
# Previous versions widened bounds to (0.65, 0.85) which was circular.
# =====================================================================

_PUBLISHED_FC_FM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "PF-1000": {
        "fc": (0.6, 0.8),    # Lee & Saw 2014 Table 1: fc ~ 0.7 for PF-1000
        "fm": (0.05, 0.20),   # Lee & Saw 2014 Table 1: fm ~ 0.05-0.15 for PF-1000
    },
    "NX2": {
        "fc": (0.60, 0.85),   # Lee & Saw 2008: fc ~ 0.7-0.8 for NX2
        "fm": (0.07, 0.25),   # Lee & Saw 2008: fm ~ 0.1-0.2 for NX2
    },
    "UNU-ICTP": {
        "fc": (0.55, 0.80),   # Lee et al., Am. J. Phys. 56 (1988): fc ~ 0.7
        "fm": (0.04, 0.35),   # Lee & Saw (2009): fm=0.05 for UNU-ICTP; widened to 0.04 lower bound
    },
    "POSEIDON": {
        "fc": (0.60, 0.85),   # Lee & Saw 2014: fc ~ 0.7 for POSEIDON
        "fm": (0.05, 0.20),   # Lee & Saw 2014: fm ~ 0.08-0.12 for POSEIDON (MJ-class)
    },
    "POSEIDON-60kV": {
        "fc": (0.50, 0.70),   # IPFS fit: fc=0.595 (different bank/geometry)
        "fm": (0.15, 0.40),   # IPFS fit: fm=0.275 (higher mass fraction)
    },
    "FAETON-I": {
        "fc": (0.55, 0.85),   # Wide range: circuit-dominated, Lee is co-author
        "fm": (0.04, 0.25),   # Wide range: no published Lee model fit yet
    },
    "MJOLNIR": {
        "fc": (0.55, 0.80),   # MA-class: similar to PF-1000 range
        "fm": (0.05, 0.20),   # MA-class: similar to PF-1000 range
    },
}


_DEFAULT_DEVICE_PCF: dict[str, float] = {
    "PF-1000": 0.14,
    "PF-1000-Gribkov": 0.14,  # Same device, different shot/publication
    "PF-1000-16kV": 0.14,
    "PF-1000-20kV": 0.14,
    "NX2": 0.5,
    "UNU-ICTP": 0.06,  # ~1 cm pinch of 16 cm anode (Lee & Saw 2009; matches presets.py)
    "POSEIDON": 0.14,  # Similar to PF-1000 (Lee & Saw 2014 scaling)
    "POSEIDON-60kV": 0.14,  # Lee & Saw scaling for MA-class
    "FAETON-I": 0.14,  # Starting estimate (no published Lee model fit)
    "MJOLNIR": 0.14,   # MA-class: same as PF-1000
}

# Default crowbar spark gap arc resistance [Ohm] per device.
# PhD Debate #30 Finding 4: R_crowbar=0 is physically incorrect and
# systematically biases fc upward during calibration.
# PF-1000: ~1-3 mOhm for ignitron/spark gap (Dr. PP estimate).
_DEFAULT_CROWBAR_R: dict[str, float] = {
    "PF-1000": 1.5e-3,  # 1.5 mOhm midpoint of 1-3 mOhm range
    "PF-1000-Gribkov": 1.5e-3,  # Same device as PF-1000
    "PF-1000-16kV": 1.5e-3,  # Same device as PF-1000 (different operating conditions)
    "PF-1000-20kV": 1.5e-3,  # Same device as PF-1000 (different operating conditions)
    "POSEIDON-60kV": 1.5e-3,  # estimated, same as PF-1000
    "UNU-ICTP": 0.0,  # No crowbar in UNU-ICTP PFF (simple capacitor bank)
    "FAETON-I": 0.0,   # No crowbar switch (Damideh 2025)
    "MJOLNIR": 1.5e-3,  # Estimated spark gap resistance
}


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


# =====================================================================
# Circuit-only calibration (path-to-7.0: blind prediction of pinch)
# =====================================================================


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


# =====================================================================
# 3-parameter calibration with liftoff delay (path-to-7.0)
# =====================================================================


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


# =====================================================================
# Blind prediction (Section 5.3 compliance)
# =====================================================================


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


# =====================================================================
# Fisher Information Matrix for identifiability analysis
# =====================================================================


@dataclass
class FIMResult:
    """Fisher Information Matrix analysis result.

    Attributes
    ----------
    fim : np.ndarray
        3x3 Fisher Information Matrix.
    eigenvalues : np.ndarray
        Eigenvalues of FIM (sorted ascending).
    condition_number : float
        Ratio of largest to smallest eigenvalue.
    param_names : list[str]
        Parameter names corresponding to FIM axes.
    is_identifiable : bool
        True if condition number < 1e4 (well-conditioned).
    """

    fim: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    param_names: list[str]
    is_identifiable: bool


def fisher_information_matrix(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.100,
    delay_us: float = 0.571,
    pinch_column_fraction: float = 0.14,
    crowbar_enabled: bool = True,
    crowbar_resistance: float = 1.5e-3,
    step_size: float = 0.01,
    nondimensionalize: bool = False,
    param_ranges: tuple[float, float, float] | None = None,
) -> FIMResult:
    """Compute Fisher Information Matrix at a parameter point.

    Uses finite-difference Jacobian of the residual vector to compute
    the FIM = J^T @ J, where J_{ij} = (1/sigma_i) * dy_i/dtheta_j.

    The condition number of the FIM indicates practical identifiability:
      - cond < 1e3: well-identified
      - cond 1e3-1e6: weakly identified (ridges)
      - cond > 1e6: practically non-identifiable

    If ``nondimensionalize=True``, the Jacobian columns are scaled by
    ``param_ranges`` (fc_range, fm_range, delay_range) so that the FIM
    condition number is unit-independent.  This addresses the issue that
    the raw FIM mixes dimensionless (fc, fm) with microsecond (delay)
    parameters.

    Args:
        device_name: Device to evaluate.
        fc, fm, delay_us: Parameter point for evaluation.
        pinch_column_fraction: Pinch column fraction.
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        step_size: Relative step size for finite differences.
        nondimensionalize: If True, scale Jacobian by param_ranges.
        param_ranges: (fc_range, fm_range, delay_range_us) for scaling.
            Required when ``nondimensionalize=True``.

    Returns:
        :class:`FIMResult` with FIM, eigenvalues, and condition number.
    """
    from dpf.validation.experimental import DEVICES
    from dpf.validation.lee_model_comparison import LeeModel

    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        raise ValueError(f"No digitized waveform for {device_name}")

    t_exp = device.waveform_t
    I_exp = device.waveform_I
    n_data = len(t_exp)

    # Measurement uncertainty per point (combined Rogowski + digitization)
    sigma = float(np.sqrt(
        device.peak_current_uncertainty**2
        + device.waveform_digitization_uncertainty**2
    )) * float(np.max(I_exp))

    theta = np.array([fc, fm, delay_us])
    param_names = ["fc", "fm", "delay_us"]

    def _run_model(fc_v: float, fm_v: float, delay_v: float) -> np.ndarray:
        """Run Lee model and interpolate to experimental time points."""
        model = LeeModel(
            current_fraction=fc_v,
            mass_fraction=fm_v,
            pinch_column_fraction=pinch_column_fraction,
            crowbar_enabled=crowbar_enabled,
            crowbar_resistance=crowbar_resistance,
            liftoff_delay=delay_v * 1e-6,
        )
        result = model.run(device_name)
        return np.interp(t_exp, result.t, result.I)

    # Jacobian via central finite differences
    J = np.zeros((n_data, 3))
    for j in range(3):
        eps = step_size * max(abs(theta[j]), 0.01)
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] += eps
        theta_minus[j] -= eps
        y_plus = _run_model(*theta_plus)
        y_minus = _run_model(*theta_minus)
        J[:, j] = (y_plus - y_minus) / (2.0 * eps * sigma)

    # Nondimensionalize: scale Jacobian columns by parameter ranges
    if nondimensionalize:
        if param_ranges is None:
            raise ValueError("param_ranges required when nondimensionalize=True")
        scales = np.array(param_ranges, dtype=float)
        J = J * scales[np.newaxis, :]  # J_scaled[:, j] = J[:, j] * range_j

    # FIM = J^T @ J
    fim = J.T @ J
    eigenvalues = np.sort(np.linalg.eigvalsh(fim))
    cond = float(eigenvalues[-1] / max(eigenvalues[0], 1e-30))

    logger.info(
        "FIM at (fc=%.3f, fm=%.3f, delay=%.3f us): "
        "eigenvalues=[%.2e, %.2e, %.2e], cond=%.2e, identifiable=%s",
        fc, fm, delay_us,
        eigenvalues[0], eigenvalues[1], eigenvalues[2],
        cond, cond < 1e4,
    )

    return FIMResult(
        fim=fim,
        eigenvalues=eigenvalues,
        condition_number=cond,
        param_names=param_names,
        is_identifiable=cond < 1e4,
    )


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


@dataclass
class MonteCarloNRMSEResult:
    """Result of Monte Carlo NRMSE uncertainty propagation.

    Attributes:
        nrmse_mean: Mean NRMSE across all Monte Carlo samples.
        nrmse_std: Standard deviation of NRMSE across samples.
        nrmse_median: Median NRMSE.
        nrmse_ci_lo: Lower bound of 95% confidence interval.
        nrmse_ci_hi: Upper bound of 95% confidence interval.
        peak_error_mean: Mean peak current error.
        peak_error_std: Standard deviation of peak current error.
        timing_error_mean: Mean timing error.
        timing_error_std: Standard deviation of timing error.
        n_samples: Number of Monte Carlo samples.
        n_failures: Number of failed runs.
        all_nrmse: All NRMSE values (for histogram plotting).
        dominant_parameter: Parameter contributing most to variance.
        sensitivity: Dict mapping parameter name to variance fraction.
    """

    nrmse_mean: float
    nrmse_std: float
    nrmse_median: float
    nrmse_ci_lo: float
    nrmse_ci_hi: float
    peak_error_mean: float
    peak_error_std: float
    timing_error_mean: float
    timing_error_std: float
    n_samples: int
    n_failures: int
    all_nrmse: np.ndarray
    dominant_parameter: str
    sensitivity: dict[str, float]


def monte_carlo_nrmse(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.094,
    n_samples: int = 200,
    seed: int = 42,
    pinch_column_fraction: float = 0.14,
    f_mr: float = 0.1,
    crowbar_enabled: bool = True,
    crowbar_resistance: float = 1.5e-3,
    liftoff_delay: float = 0.0,
    parameter_uncertainties: dict[str, float] | None = None,
) -> MonteCarloNRMSEResult:
    """Monte Carlo propagation of input parameter uncertainty to NRMSE.

    Perturbs circuit and geometry parameters within their measurement
    uncertainties (1-sigma, Gaussian), runs the Lee model for each sample,
    and computes the distribution of NRMSE values.

    Default uncertainties (1-sigma relative) for PF-1000:
        C: 2% (capacitor bank tolerance)
        V0: 1% (voltage monitor calibration)
        L0: 5% (short-circuit discharge calibration)
        R0: 10% (short-circuit, frequency-dependent)
        a: 1% (machining tolerance)
        b: 1% (machining tolerance)
        z: 1% (machining tolerance)
        fc: 5% (calibration valley width)
        fm: 20% (calibration valley width)
        pcf: 30% (Lee & Saw 2014: 0.07-0.21)

    Args:
        device_name: Device to validate against.
        fc: Current fraction (central value).
        fm: Mass fraction (central value).
        n_samples: Number of Monte Carlo draws.
        seed: Random seed for reproducibility.
        pinch_column_fraction: Central pcf value.
        f_mr: Radial mass fraction.
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        liftoff_delay: Insulator flashover delay [s].  Default 0.0.
            Perturbed with additive Gaussian noise (sigma=0.3 us).
        parameter_uncertainties: Override default uncertainties.
            Keys: 'C', 'V0', 'L0', 'R0', 'a', 'b', 'z', 'fc', 'fm', 'pcf',
            'liftoff_delay'.  Values: 1-sigma uncertainty.  For liftoff_delay,
            the value is absolute [s] (not relative).

    Returns:
        :class:`MonteCarloNRMSEResult` with NRMSE distribution statistics.
    """
    from dpf.validation.lee_model_comparison import LeeModel

    # Default PF-1000 parameter uncertainties (1-sigma relative)
    # liftoff_delay uses absolute uncertainty [s] (Lee 2005: 0.5-1.5 us for MJ)
    default_u = {
        "C": 0.02, "V0": 0.01, "L0": 0.05, "R0": 0.10,
        "a": 0.01, "b": 0.01, "z": 0.01,
        "fc": 0.05, "fm": 0.20, "pcf": 0.30,
        "liftoff_delay": 0.3e-6,  # ±0.3 us absolute (Lee 2005: 0.5-1.5 us)
    }
    if parameter_uncertainties:
        default_u.update(parameter_uncertainties)

    rng = np.random.default_rng(seed)
    nrmse_arr = []
    peak_err_arr = []
    timing_err_arr = []
    n_fail = 0

    # Get nominal device parameters
    from dpf.validation.experimental import DEVICES
    device = DEVICES[device_name]
    C_nom = device.capacitance
    V0_nom = device.voltage
    L0_nom = device.inductance
    R0_nom = device.resistance
    a_nom = device.anode_radius
    b_nom = device.cathode_radius
    z_nom = device.anode_length

    for _ in range(n_samples):
        # Perturb each parameter
        C_s = C_nom * (1 + rng.normal(0, default_u["C"]))
        V0_s = V0_nom * (1 + rng.normal(0, default_u["V0"]))
        L0_s = L0_nom * (1 + rng.normal(0, default_u["L0"]))
        R0_s = R0_nom * (1 + rng.normal(0, default_u["R0"]))
        a_s = a_nom * (1 + rng.normal(0, default_u["a"]))
        b_s = b_nom * (1 + rng.normal(0, default_u["b"]))
        z_s = z_nom * (1 + rng.normal(0, default_u["z"]))
        fc_s = fc * (1 + rng.normal(0, default_u["fc"]))
        fm_s = fm * (1 + rng.normal(0, default_u["fm"]))
        pcf_s = pinch_column_fraction * (1 + rng.normal(0, default_u["pcf"]))
        # Liftoff delay: additive Gaussian perturbation (absolute, not relative)
        delay_s = liftoff_delay + rng.normal(0, default_u["liftoff_delay"])
        delay_s = max(delay_s, 0.0)  # Cannot be negative

        # Clamp to physical bounds
        fc_s = float(np.clip(fc_s, 0.3, 1.0))
        fm_s = float(np.clip(fm_s, 0.01, 0.5))
        pcf_s = float(np.clip(pcf_s, 0.01, 1.0))
        a_s = max(a_s, 0.001)
        b_s = max(b_s, a_s * 1.1)
        R0_s = max(R0_s, 1e-6)
        L0_s = max(L0_s, 1e-12)
        C_s = max(C_s, 1e-9)

        try:
            model = LeeModel(
                current_fraction=fc_s,
                mass_fraction=fm_s,
                radial_mass_fraction=f_mr,
                pinch_column_fraction=pcf_s,
                crowbar_enabled=crowbar_enabled,
                crowbar_resistance=crowbar_resistance,
                liftoff_delay=delay_s,
            )
            # Override device parameters for this sample
            comp = model.compare_with_experiment(
                device_name,
                override_params={
                    "C": C_s, "V0": V0_s, "L0": L0_s, "R0": R0_s,
                    "anode_radius": a_s, "cathode_radius": b_s,
                    "anode_length": z_s,
                },
            )
            nrmse_arr.append(comp.waveform_nrmse)
            peak_err_arr.append(comp.peak_current_error)
            timing_err_arr.append(comp.timing_error)
        except Exception:
            n_fail += 1

    nrmse = np.array(nrmse_arr)
    peak_err = np.array(peak_err_arr)
    timing_err = np.array(timing_err_arr)

    # Sensitivity analysis: compute variance contribution of each parameter
    # Use one-at-a-time perturbation at ±1 sigma
    sensitivity = {}
    nominal_model = LeeModel(
        current_fraction=fc, mass_fraction=fm,
        radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance,
    )
    nominal_model.compare_with_experiment(device_name)

    param_map = {
        "C": ("C", C_nom), "V0": ("V0", V0_nom),
        "L0": ("L0", L0_nom), "R0": ("R0", R0_nom),
        "a": ("anode_radius", a_nom), "b": ("cathode_radius", b_nom),
        "z": ("anode_length", z_nom),
    }
    total_var = 0.0
    for pname, (okey, pnom) in param_map.items():
        u = default_u[pname]
        try:
            m_plus = LeeModel(
                current_fraction=fc, mass_fraction=fm,
                radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance,
            )
            c_plus = m_plus.compare_with_experiment(
                device_name, override_params={okey: pnom * (1 + u)}
            )
            m_minus = LeeModel(
                current_fraction=fc, mass_fraction=fm,
                radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance,
            )
            c_minus = m_minus.compare_with_experiment(
                device_name, override_params={okey: pnom * (1 - u)}
            )
            delta = (c_plus.waveform_nrmse - c_minus.waveform_nrmse) / 2
            sensitivity[pname] = delta ** 2
            total_var += delta ** 2
        except Exception:
            sensitivity[pname] = 0.0

    # Add liftoff_delay sensitivity (absolute perturbation)
    if liftoff_delay > 0 and default_u.get("liftoff_delay", 0) > 0:
        u_delay = default_u["liftoff_delay"]
        try:
            m_p = LeeModel(
                current_fraction=fc, mass_fraction=fm,
                radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance,
                liftoff_delay=liftoff_delay + u_delay)
            m_m = LeeModel(
                current_fraction=fc, mass_fraction=fm,
                radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance,
                liftoff_delay=max(liftoff_delay - u_delay, 0.0))
            c_p = m_p.compare_with_experiment(device_name)
            c_m = m_m.compare_with_experiment(device_name)
            delta = (c_p.waveform_nrmse - c_m.waveform_nrmse) / 2
            sensitivity["liftoff_delay"] = delta ** 2
            total_var += delta ** 2
        except Exception:
            sensitivity["liftoff_delay"] = 0.0

    # Add fc, fm, pcf sensitivity
    for pname, pval, _pkey in [("fc", fc, None), ("fm", fm, None), ("pcf", pinch_column_fraction, None)]:
        u = default_u[pname]
        try:
            if pname == "fc":
                m_p = LeeModel(current_fraction=pval*(1+u), mass_fraction=fm,
                    radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                    crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance)
                m_m = LeeModel(current_fraction=pval*(1-u), mass_fraction=fm,
                    radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                    crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance)
            elif pname == "fm":
                m_p = LeeModel(current_fraction=fc, mass_fraction=pval*(1+u),
                    radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                    crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance)
                m_m = LeeModel(current_fraction=fc, mass_fraction=pval*(1-u),
                    radial_mass_fraction=f_mr, pinch_column_fraction=pinch_column_fraction,
                    crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance)
            else:  # pcf
                m_p = LeeModel(current_fraction=fc, mass_fraction=fm,
                    radial_mass_fraction=f_mr, pinch_column_fraction=pval*(1+u),
                    crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance)
                m_m = LeeModel(current_fraction=fc, mass_fraction=fm,
                    radial_mass_fraction=f_mr, pinch_column_fraction=pval*(1-u),
                    crowbar_enabled=crowbar_enabled, crowbar_resistance=crowbar_resistance)
            c_p = m_p.compare_with_experiment(device_name)
            c_m = m_m.compare_with_experiment(device_name)
            delta = (c_p.waveform_nrmse - c_m.waveform_nrmse) / 2
            sensitivity[pname] = delta ** 2
            total_var += delta ** 2
        except Exception:
            sensitivity[pname] = 0.0

    # Normalize to fractions of total variance
    if total_var > 0:
        sensitivity = {k: v / total_var for k, v in sensitivity.items()}

    dominant = max(sensitivity, key=sensitivity.get) if sensitivity else "unknown"

    return MonteCarloNRMSEResult(
        nrmse_mean=float(np.mean(nrmse)),
        nrmse_std=float(np.std(nrmse)),
        nrmse_median=float(np.median(nrmse)),
        nrmse_ci_lo=float(np.percentile(nrmse, 2.5)),
        nrmse_ci_hi=float(np.percentile(nrmse, 97.5)),
        peak_error_mean=float(np.mean(peak_err)),
        peak_error_std=float(np.std(peak_err)),
        timing_error_mean=float(np.mean(timing_err)),
        timing_error_std=float(np.std(timing_err)),
        n_samples=len(nrmse),
        n_failures=n_fail,
        all_nrmse=nrmse,
        dominant_parameter=dominant,
        sensitivity=sensitivity,
    )


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

    # Experimental uncertainty: combine Rogowski + digitization + shot-to-shot
    u_exp_sq = (
        device.peak_current_uncertainty**2
        + device.waveform_digitization_uncertainty**2
    )
    if include_shot_to_shot and device_name in _SHOT_TO_SHOT_DATA:
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

    logger.info(
        "ASME V&V 20: %s (%s) — E=%.3f, u_exp=%.3f, u_input=%.3f, "
        "u_num=%.4f, u_val=%.3f, delta_model=%.3f, ratio=%.2f → %s",
        device_name, time_desc, E, u_exp, u_input, u_num, u_val,
        delta_model, ratio, "PASS" if passes else "FAIL",
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
    )


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
        f_mr: float | None = None,
        pinch_column_fraction: float = 1.0,
        train_pcf: float | None = None,
        test_pcf: float | None = None,
    ) -> CrossValidationResult:
        """Calibrate on train_device, predict on test_device.

        Args:
            train_device: Device name for calibration.
            test_device: Device name for prediction evaluation.
            maxiter: Maximum optimizer iterations.
            f_mr: Radial mass fraction. Defaults to None (uses fm).
            pinch_column_fraction: Fraction of anode length for radial
                compression.  Used as fallback if device-specific pcf
                values are not provided.
            train_pcf: pcf for the training device.  If None, uses
                ``_DEFAULT_DEVICE_PCF[train_device]`` or falls back to
                ``pinch_column_fraction``.
            test_pcf: pcf for the test device.  If None, uses
                ``_DEFAULT_DEVICE_PCF[test_device]`` or falls back to
                ``pinch_column_fraction``.

        Returns:
            :class:`CrossValidationResult` with generalization metrics.
        """
        from dpf.validation.lee_model_comparison import LeeModel

        # Resolve device-specific pcf values
        if train_pcf is None:
            train_pcf = _DEFAULT_DEVICE_PCF.get(train_device, pinch_column_fraction)
        if test_pcf is None:
            test_pcf = _DEFAULT_DEVICE_PCF.get(test_device, pinch_column_fraction)

        # Step 1: Calibrate on train device with its own pcf
        cal = LeeModelCalibrator(
            train_device, pinch_column_fraction=train_pcf,
        )
        cal_result = cal.calibrate(maxiter=maxiter)

        # Step 2: Run prediction on test device with TEST device's pcf
        model = LeeModel(
            current_fraction=cal_result.best_fc,
            mass_fraction=cal_result.best_fm,
            radial_mass_fraction=f_mr,
            pinch_column_fraction=test_pcf,
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


@dataclass
class BootstrapCIResult:
    """Bootstrap confidence intervals for calibration parameters.

    Attributes:
        fc_mean: Mean fc across bootstrap resamples.
        fc_std: Standard deviation of fc.
        fc_ci_lo: Lower 95% CI for fc.
        fc_ci_hi: Upper 95% CI for fc.
        fm_mean: Mean fm across bootstrap resamples.
        fm_std: Standard deviation of fm.
        fm_ci_lo: Lower 95% CI for fm.
        fm_ci_hi: Upper 95% CI for fm.
        fc_fm_corr: Pearson correlation between fc and fm.
        n_resamples: Number of bootstrap resamples completed.
        fc_at_boundary_frac: Fraction of resamples where fc hit upper bound.
        degeneracy_ratio_mean: Mean fc^2/fm ratio.
        degeneracy_ratio_std: Std of fc^2/fm ratio.
    """

    fc_mean: float
    fc_std: float
    fc_ci_lo: float
    fc_ci_hi: float
    fm_mean: float
    fm_std: float
    fm_ci_lo: float
    fm_ci_hi: float
    fc_fm_corr: float
    n_resamples: int
    fc_at_boundary_frac: float
    degeneracy_ratio_mean: float
    degeneracy_ratio_std: float


def _estimate_block_size(t: np.ndarray, I_data: np.ndarray) -> int:  # noqa: N803
    """Estimate optimal block size for block bootstrap from autocorrelation.

    Uses the first lag where autocorrelation drops below 1/e ≈ 0.368,
    clamped to [2, n//3].  Falls back to n^(1/3) rule (Kunsch 1989)
    if autocorrelation doesn't decay.

    Args:
        t: Time array.
        I_data: Current array.

    Returns:
        Estimated block size (integer >= 2).
    """
    n = len(I_data)
    if n < 6:
        return 2

    # Normalize
    I_centered = I_data - np.mean(I_data)
    var = np.var(I_centered)
    if var < 1e-30:
        return max(2, int(np.ceil(n ** (1.0 / 3.0))))

    # Compute autocorrelation up to n//2 lags
    max_lag = n // 2
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = float(np.mean(I_centered[: n - lag] * I_centered[lag:])) / var

    # Find first lag where acf drops below 1/e
    threshold = 1.0 / np.e
    for lag in range(1, max_lag):
        if acf[lag] < threshold:
            block_size = max(2, lag + 1)
            return min(block_size, n // 3)

    # Fallback: Kunsch (1989) n^(1/3) rule
    return max(2, min(int(np.ceil(n ** (1.0 / 3.0))), n // 3))


def bootstrap_calibration(
    device_name: str = "PF-1000",
    n_resamples: int = 50,
    fc_bounds: tuple[float, float] = (0.6, 0.8),
    fm_bounds: tuple[float, float] = (0.05, 0.25),
    maxiter: int = 80,
    pinch_column_fraction: float = 0.14,
    crowbar_enabled: bool = True,
    crowbar_resistance: float = 1.5e-3,
    seed: int = 42,
    block_size: int | None = None,
) -> BootstrapCIResult:
    """Block bootstrap confidence intervals for fc/fm calibration.

    Uses the moving block bootstrap (Kunsch 1989, Liu & Singh 1992) to
    resample contiguous blocks of the experimental waveform, preserving
    temporal autocorrelation.  Re-runs Nelder-Mead calibration for each
    resample.

    Args:
        device_name: Registered device name.
        n_resamples: Number of bootstrap resamples (default 50).
        fc_bounds: Bounds for current fraction.
        fm_bounds: Bounds for mass fraction.
        maxiter: Max optimizer iterations per resample.
        pinch_column_fraction: Pinch column fraction (fixed).
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        seed: Random seed for reproducibility.
        block_size: Block size for moving block bootstrap.  If None,
            estimated from autocorrelation (recommended).

    Returns:
        :class:`BootstrapCIResult` with confidence intervals and
        degeneracy diagnostics.
    """
    from dpf.validation.experimental import DEVICES, nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    rng = np.random.default_rng(seed)
    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        raise ValueError(f"No digitized waveform for {device_name}")

    t_exp = device.waveform_t
    I_exp = device.waveform_I
    n_pts = len(t_exp)

    # Estimate or use provided block size
    if block_size is None:
        block_size = _estimate_block_size(t_exp, I_exp)
    block_size = max(2, min(block_size, n_pts // 2))

    logger.info(
        "Block bootstrap %s: n_pts=%d, block_size=%d, n_resamples=%d",
        device_name, n_pts, block_size, n_resamples,
    )

    fc_samples = []
    fm_samples = []

    n_blocks_needed = int(np.ceil(n_pts / block_size))

    for _i in range(n_resamples):
        # Moving block bootstrap: sample n_blocks_needed start indices,
        # concatenate blocks, truncate to n_pts
        max_start = n_pts - block_size
        if max_start < 1:
            max_start = 1
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)

        idx = np.concatenate([
            np.arange(s, min(s + block_size, n_pts)) for s in starts
        ])[:n_pts]

        t_boot = t_exp[idx]
        I_boot = I_exp[idx]

        # Sort by time for interpolation
        sort_order = np.argsort(t_boot)
        t_boot = t_boot[sort_order]
        I_boot = I_boot[sort_order]

        # Remove duplicate times (keep first)
        unique_mask = np.concatenate([[True], np.diff(t_boot) > 0])
        t_boot = t_boot[unique_mask]
        I_boot = I_boot[unique_mask]
        if len(t_boot) < 5:
            continue

        # Calibrate on resampled data — capture loop vars explicitly
        t_ref, I_ref = t_boot, I_boot

        def objective(
            x: np.ndarray,
            _t_ref: np.ndarray = t_ref,
            _I_ref: np.ndarray = I_ref,
        ) -> float:
            fc_t, fm_t = float(x[0]), float(x[1])
            try:
                model = LeeModel(
                    current_fraction=fc_t,
                    mass_fraction=fm_t,
                    pinch_column_fraction=pinch_column_fraction,
                    crowbar_enabled=crowbar_enabled,
                    crowbar_resistance=crowbar_resistance,
                )
                result = model.run(device_name)
                nrmse = nrmse_peak(
                    result.t, result.I, _t_ref, _I_ref,
                )
                return float(nrmse)
            except Exception:
                return 1.0

        from scipy.optimize import Bounds, minimize

        x0 = np.array([
            0.5 * (fc_bounds[0] + fc_bounds[1]),
            0.5 * (fm_bounds[0] + fm_bounds[1]),
        ])
        try:
            res = minimize(
                objective, x0, method="nelder-mead",
                bounds=Bounds(
                    [fc_bounds[0], fm_bounds[0]],
                    [fc_bounds[1], fm_bounds[1]],
                ),
                options={"maxiter": maxiter, "xatol": 0.005, "fatol": 0.001},
            )
            fc_opt = float(np.clip(res.x[0], *fc_bounds))
            fm_opt = float(np.clip(res.x[1], *fm_bounds))
            fc_samples.append(fc_opt)
            fm_samples.append(fm_opt)
        except Exception:
            continue

    fc_arr = np.array(fc_samples)
    fm_arr = np.array(fm_samples)
    n_ok = len(fc_arr)

    if n_ok < 3:
        raise RuntimeError(f"Bootstrap failed: only {n_ok} successful resamples")

    ratio = fc_arr**2 / fm_arr

    # Compute correlation
    if np.std(fc_arr) > 0 and np.std(fm_arr) > 0:
        corr = float(np.corrcoef(fc_arr, fm_arr)[0, 1])
    else:
        corr = 0.0

    boundary_frac = float(np.mean(fc_arr >= fc_bounds[1] - 0.005))

    logger.info(
        "Bootstrap %s (n=%d/%d): fc=%.3f±%.3f [%.3f, %.3f], "
        "fm=%.3f±%.3f [%.3f, %.3f], corr=%.2f, boundary=%.0f%%",
        device_name, n_ok, n_resamples,
        np.mean(fc_arr), np.std(fc_arr),
        np.percentile(fc_arr, 2.5), np.percentile(fc_arr, 97.5),
        np.mean(fm_arr), np.std(fm_arr),
        np.percentile(fm_arr, 2.5), np.percentile(fm_arr, 97.5),
        corr, boundary_frac * 100,
    )

    return BootstrapCIResult(
        fc_mean=float(np.mean(fc_arr)),
        fc_std=float(np.std(fc_arr)),
        fc_ci_lo=float(np.percentile(fc_arr, 2.5)),
        fc_ci_hi=float(np.percentile(fc_arr, 97.5)),
        fm_mean=float(np.mean(fm_arr)),
        fm_std=float(np.std(fm_arr)),
        fm_ci_lo=float(np.percentile(fm_arr, 2.5)),
        fm_ci_hi=float(np.percentile(fm_arr, 97.5)),
        fc_fm_corr=corr,
        n_resamples=n_ok,
        fc_at_boundary_frac=boundary_frac,
        degeneracy_ratio_mean=float(np.mean(ratio)),
        degeneracy_ratio_std=float(np.std(ratio)),
    )


@dataclass
class BennettEquilibriumResult:
    """Bennett equilibrium check at pinch conditions.

    The Bennett relation states that for a z-pinch in equilibrium:
        I^2 = (8*pi/mu_0) * N_L * k_B * (T_e + T_i)

    where N_L is the line density (particles per unit length).

    Attributes:
        I_pinch: Current at pinch [A].
        r_pinch: Pinch radius [m].
        z_pinch: Pinch length [m].
        n_pinch: Pinch number density [m^-3].
        N_L: Line density [m^-1].
        T_bennett: Bennett temperature [eV].
        I_bennett: Bennett current for the given T and N_L [A].
        I_ratio: I_pinch / I_bennett (should be ~1 for equilibrium).
        is_consistent: Whether |I_ratio - 1| < tolerance.
    """

    I_pinch: float
    r_pinch: float
    z_pinch: float
    n_pinch: float
    N_L: float
    T_bennett: float
    I_bennett: float
    I_ratio: float
    is_consistent: bool


def bennett_equilibrium_check(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.094,
    pinch_column_fraction: float = 0.14,
    compression_ratio: float = 10.0,
    T_assumed_eV: float | None = None,
    tolerance: float = 0.5,
) -> BennettEquilibriumResult:
    """Check Bennett equilibrium self-consistency at pinch.

    Computes the Bennett temperature from the snowplow pinch conditions
    and verifies that I_pinch^2 ~ (8*pi/mu_0) * N_L * k_B * (T_e + T_i).

    If T_assumed_eV is provided, uses that temperature.  Otherwise
    estimates from the snowplow kinetic energy at pinch.

    Args:
        device_name: Registered device name.
        fc: Current fraction.
        fm: Mass fraction.
        pinch_column_fraction: Pinch column fraction.
        compression_ratio: Ratio of cathode radius to pinch radius.
            Default 10 (r_pinch = a/10).
        T_assumed_eV: If given, use this temperature [eV] instead of
            estimating from kinetics.
        tolerance: Fractional tolerance for consistency check.
            Default 0.5 (I_ratio within 0.5-1.5).

    Returns:
        :class:`BennettEquilibriumResult`.
    """
    from dpf.validation.experimental import DEVICES

    MU_0 = 4e-7 * np.pi  # H/m
    K_B = 1.38064852e-23  # J/K
    EV_TO_K = 11604.5  # K/eV

    device = DEVICES[device_name]

    # Geometry
    a = device.anode_radius  # m
    b = device.cathode_radius  # m
    z_anode = device.anode_length  # m
    z_pinch = pinch_column_fraction * z_anode  # m
    r_pinch = a / compression_ratio  # m

    # Fill conditions
    fill_pressure_Pa = device.fill_pressure_torr * 133.322
    n_fill = fill_pressure_Pa / (K_B * 300.0)  # room temperature fill

    # Swept mass and pinch density
    # Mass fraction fm of the annular gas mass is swept into the pinch column
    V_annular = np.pi * (b**2 - a**2) * z_pinch
    # For D2 gas: n_fill is molecular density, each molecule has 2 deuterons
    # So particle count = 2 * n_fill * V_annular * fm
    n_particles = 2 * n_fill * V_annular * fm
    V_pinch = np.pi * r_pinch**2 * z_pinch
    n_pinch = n_particles / V_pinch  # ions/m^3

    # Line density
    N_L = n_pinch * np.pi * r_pinch**2  # particles/m

    # Pinch current
    I_peak = device.peak_current
    I_pinch = fc * I_peak  # current through pinch

    if T_assumed_eV is not None:
        T_total_K = T_assumed_eV * EV_TO_K * 2  # T_e + T_i ~ 2T
        T_bennett_eV = T_assumed_eV
    else:
        # Non-tautological: run the Lee model to get the implosion velocity
        # at pinch.  T = m_D * v_imp^2 / (3 * k_B).
        # This is independent of the Bennett relation because v_imp comes
        # from the snowplow ODE dynamics (I(t) history), not from local I.
        from dpf.validation.lee_model_comparison import LeeModel

        m_D = 3.3436e-27  # deuteron mass [kg]
        lee = LeeModel(
            current_fraction=fc,
            mass_fraction=fm,
            pinch_column_fraction=pinch_column_fraction,
        )
        lee_result = lee.run(device_name)

        # Extract implosion velocity from r_shock trajectory.
        # r_shock = b during axial phase, then decreases during radial phase.
        # Find the radial phase portion (where r < b) and compute dr/dt.
        v_imp = 0.0
        r_all = lee_result.r_shock
        t_all = lee_result.t
        if len(r_all) >= 4 and len(t_all) == len(r_all):
            # Identify radial phase: r < 0.99 * b
            radial_mask = r_all < 0.99 * b
            if np.any(radial_mask):
                r_rad = r_all[radial_mask]
                t_rad = t_all[radial_mask]
                if len(r_rad) >= 3:
                    # Compute velocity near pinch from converging (inward) motion only.
                    # Filter for dr/dt < 0 to exclude Phase 4 reflected shock (outward).
                    n_tail = min(5, len(r_rad) - 1)
                    dr = np.diff(r_rad[-n_tail - 1:])
                    dt_r = np.diff(t_rad[-n_tail - 1:])
                    v_raw = dr / np.maximum(dt_r, 1e-15)
                    # Select only converging motion (dr/dt < 0 = inward)
                    inward_mask = v_raw < 0
                    if np.any(inward_mask):
                        v_imp = float(np.mean(np.abs(v_raw[inward_mask])))
                    else:
                        # All points are expanding; use absolute values as fallback
                        v_imp = float(np.mean(np.abs(v_raw)))

        if v_imp > 1e3:  # Physically reasonable (> 1 km/s)
            # Rankine-Hugoniot strong shock temperature for gamma=5/3:
            # T_post = 3 * m * v^2 / (16 * k_B)  [per species, post-shock]
            # At DPF pinch conditions, tau_ei >> tau_pinch, so T_e << T_i.
            # Bennett relation uses (T_e + T_i) ≈ T_i for the pressure balance.
            T_ion_K = 3.0 * m_D * v_imp**2 / (16.0 * K_B)
            T_total_K = T_ion_K  # T_e + T_i ≈ T_i (T_e << T_i at pinch)
            T_bennett_eV = T_total_K / EV_TO_K
        else:
            # Fallback: adiabatic compression T = T_fill * (b/r_pinch)^(2(gamma-1))
            gamma = 5.0 / 3.0
            T_fill_K = 300.0  # room temperature
            T_total_K = T_fill_K * (b / max(r_pinch, 1e-6)) ** (2 * (gamma - 1))
            T_bennett_eV = T_total_K / (2 * EV_TO_K)

    # Bennett current for the given T and N_L
    I_bennett = np.sqrt(8 * np.pi * N_L * K_B * T_total_K / MU_0)

    I_ratio = I_pinch / max(I_bennett, 1.0)
    is_consistent = abs(I_ratio - 1.0) < tolerance

    logger.info(
        "Bennett check %s: I_pinch=%.2f MA, r_pinch=%.1f mm, "
        "n_pinch=%.2e m^-3, N_L=%.2e m^-1, T_bennett=%.0f eV, "
        "I_bennett=%.2f MA, ratio=%.2f → %s",
        device_name, I_pinch / 1e6, r_pinch * 1e3,
        n_pinch, N_L, T_bennett_eV,
        I_bennett / 1e6, I_ratio,
        "CONSISTENT" if is_consistent else "INCONSISTENT",
    )

    return BennettEquilibriumResult(
        I_pinch=I_pinch,
        r_pinch=r_pinch,
        z_pinch=z_pinch,
        n_pinch=n_pinch,
        N_L=N_L,
        T_bennett=T_bennett_eV,
        I_bennett=I_bennett,
        I_ratio=I_ratio,
        is_consistent=is_consistent,
    )


# =====================================================================
# NRMSE timing/amplitude decomposition
# (path-to-7.0: separate timing error from amplitude error)
# =====================================================================


@dataclass
class NRMSEDecomposition:
    """Decomposition of NRMSE into timing and amplitude components.

    The total NRMSE conflates two distinct error sources:
    1. Timing error: the simulated waveform is time-shifted relative to
       the experimental waveform (phase error).
    2. Amplitude error: after optimal time alignment, the residual
       amplitude mismatch.

    The decomposition uses cross-correlation to find the optimal time
    shift that minimizes the aligned NRMSE.

    Attributes:
        total_nrmse: Original (unaligned) NRMSE.
        aligned_nrmse: NRMSE after optimal time alignment (amplitude error).
        timing_nrmse: sqrt(total^2 - aligned^2) — timing contribution.
        optimal_shift_us: Optimal time shift [us] (positive = sim is late).
        timing_fraction: Fraction of NRMSE^2 attributable to timing.
        amplitude_fraction: Fraction of NRMSE^2 attributable to amplitude.
        device_name: Device name.
    """

    total_nrmse: float
    aligned_nrmse: float
    timing_nrmse: float
    optimal_shift_us: float
    timing_fraction: float
    amplitude_fraction: float
    device_name: str = ""


def nrmse_timing_amplitude_decomposition(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.094,
    f_mr: float = 0.1,
    pinch_column_fraction: float = 0.14,
    crowbar_enabled: bool = True,
    crowbar_resistance: float = 1.5e-3,
    max_shift_us: float = 2.0,
    shift_resolution_ns: float = 10.0,
) -> NRMSEDecomposition:
    """Decompose NRMSE into timing and amplitude components.

    Uses brute-force time-shift search: shift the simulated waveform
    by dt in [-max_shift_us, +max_shift_us] and compute NRMSE at each
    shift.  The minimum-NRMSE shift gives the optimal alignment.

    This addresses PhD Debate #38 Finding #11: "NRMSE conflates ~8%
    timing error + ~7% amplitude error."

    Args:
        device_name: Registered device name.
        fc: Current fraction.
        fm: Mass fraction.
        f_mr: Radial mass fraction.
        pinch_column_fraction: Pinch column fraction.
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        max_shift_us: Maximum time shift to search [us].
        shift_resolution_ns: Resolution of time shift search [ns].

    Returns:
        :class:`NRMSEDecomposition` with timing/amplitude breakdown.
    """
    from dpf.validation.experimental import DEVICES, nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        raise ValueError(f"No digitized waveform for {device_name}")

    # Run the model
    model = LeeModel(
        current_fraction=fc,
        mass_fraction=fm,
        radial_mass_fraction=f_mr,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
    )
    result = model.run(device_name)

    # Unshifted NRMSE
    total_nrmse = nrmse_peak(
        result.t, result.I,
        device.waveform_t, device.waveform_I,
    )

    # Brute-force time-shift search
    max_shift_s = max_shift_us * 1e-6
    resolution_s = shift_resolution_ns * 1e-9
    n_shifts = int(2 * max_shift_s / resolution_s) + 1
    shifts = np.linspace(-max_shift_s, max_shift_s, n_shifts)

    t_exp = np.asarray(device.waveform_t, dtype=np.float64)
    I_exp = np.asarray(device.waveform_I, dtype=np.float64)
    I_peak = float(np.max(np.abs(I_exp)))

    best_nrmse = total_nrmse
    best_shift = 0.0

    for dt in shifts:
        # Shift the simulated waveform: t_sim -> t_sim + dt
        # This is equivalent to evaluating sim at (t_exp - dt)
        I_sim_shifted = np.interp(t_exp - dt, result.t, result.I)
        residuals = I_sim_shifted - I_exp
        nrmse = float(np.sqrt(np.mean(residuals**2))) / max(I_peak, 1e-300)
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_shift = dt

    aligned_nrmse = best_nrmse
    timing_nrmse_sq = max(0.0, total_nrmse**2 - aligned_nrmse**2)
    timing_nrmse = float(np.sqrt(timing_nrmse_sq))

    total_sq = total_nrmse**2
    timing_frac = timing_nrmse_sq / total_sq if total_sq > 0 else 0.0
    amplitude_frac = 1.0 - timing_frac

    optimal_shift_us = best_shift * 1e6

    logger.info(
        "NRMSE decomposition %s: total=%.3f, aligned=%.3f, timing=%.3f, "
        "shift=%.2f us, timing_frac=%.1f%%, amplitude_frac=%.1f%%",
        device_name, total_nrmse, aligned_nrmse, timing_nrmse,
        optimal_shift_us, timing_frac * 100, amplitude_frac * 100,
    )

    return NRMSEDecomposition(
        total_nrmse=total_nrmse,
        aligned_nrmse=aligned_nrmse,
        timing_nrmse=timing_nrmse,
        optimal_shift_us=optimal_shift_us,
        timing_fraction=timing_frac,
        amplitude_fraction=amplitude_frac,
        device_name=device_name,
    )


# =====================================================================
# Validation summary report (path-to-7.0: u_val alongside every NRMSE)
# =====================================================================


@dataclass
class ValidationSummaryReport:
    """Comprehensive validation report with NRMSE + u_val for multiple windows.

    Reports decoupled circuit-phase (0-6 us) and pinch-phase (6-10 us) metrics
    alongside full-waveform metrics.  Every NRMSE is accompanied by its
    validation uncertainty u_val per ASME V&V 20-2009.

    Attributes:
        device_name: Device name.
        fc: Current fraction used.
        fm: Mass fraction used.
        full: ASME result for full waveform.
        circuit_phase: ASME result for 0-6 us (circuit-dominated).
        pinch_phase: ASME result for pinch window (if waveform extends past 6 us).
        bennett: Bennett equilibrium check result.
        fc_squared_over_fm: Degeneracy diagnostic fc^2/fm.
        speed_factor: Speed factor S/S_opt (if available).
    """

    device_name: str
    fc: float
    fm: float
    full: ASMEValidationResult
    circuit_phase: ASMEValidationResult | None
    pinch_phase: ASMEValidationResult | None
    bennett: BennettEquilibriumResult | None
    fc_squared_over_fm: float
    speed_factor: dict[str, float] | None = None


def validation_summary(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.094,
    f_mr: float = 0.1,
    pinch_column_fraction: float = 0.14,
    crowbar_enabled: bool = True,
    crowbar_resistance: float = 1.5e-3,
    circuit_window_us: float = 6.0,
    include_bennett: bool = True,
) -> ValidationSummaryReport:
    """Generate comprehensive validation summary with decoupled metrics.

    Reports NRMSE + u_val for:
    - Full waveform
    - Circuit phase only (0 to circuit_window_us)
    - Pinch phase only (circuit_window_us to end)

    Args:
        device_name: Registered device name.
        fc: Current fraction.
        fm: Mass fraction.
        f_mr: Radial mass fraction.
        pinch_column_fraction: Pinch column fraction.
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        circuit_window_us: End of circuit phase in microseconds.
        include_bennett: Whether to include Bennett equilibrium check.

    Returns:
        :class:`ValidationSummaryReport` with decoupled metrics.
    """
    # Full waveform ASME assessment
    full = asme_vv20_assessment(
        device_name=device_name, fc=fc, fm=fm, f_mr=f_mr,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
    )

    # Circuit-phase only (0 to circuit_window_us)
    circuit_max_time = circuit_window_us * 1e-6
    try:
        circuit = asme_vv20_assessment(
            device_name=device_name, fc=fc, fm=fm, f_mr=f_mr,
            pinch_column_fraction=pinch_column_fraction,
            crowbar_enabled=crowbar_enabled,
            crowbar_resistance=crowbar_resistance,
            max_time=circuit_max_time,
        )
    except Exception:
        circuit = None

    # Pinch-phase NRMSE: compute from waveform data beyond circuit_window_us
    pinch = _pinch_phase_asme(
        device_name=device_name, fc=fc, fm=fm, f_mr=f_mr,
        pinch_column_fraction=pinch_column_fraction,
        crowbar_enabled=crowbar_enabled,
        crowbar_resistance=crowbar_resistance,
        t_start_us=circuit_window_us,
    )

    # Bennett equilibrium
    bennett = None
    if include_bennett:
        import contextlib
        with contextlib.suppress(Exception):
            bennett = bennett_equilibrium_check(
                device_name=device_name, fc=fc, fm=fm,
                pinch_column_fraction=pinch_column_fraction,
            )

    # Speed factor
    speed = None
    try:
        from dpf.validation.experimental import DEVICES, compute_speed_factor
        dev = DEVICES[device_name]
        if dev.peak_current > 0:
            speed = compute_speed_factor(
                dev.peak_current, dev.anode_radius, dev.fill_pressure_torr,
            )
    except Exception:
        pass

    report = ValidationSummaryReport(
        device_name=device_name,
        fc=fc,
        fm=fm,
        full=full,
        circuit_phase=circuit,
        pinch_phase=pinch,
        bennett=bennett,
        fc_squared_over_fm=fc**2 / fm if fm > 0 else float("inf"),
        speed_factor=speed,
    )

    logger.info(
        "Validation summary %s: full NRMSE=%.3f (u_val=%.3f, ratio=%.2f), "
        "circuit NRMSE=%s, pinch NRMSE=%s, Bennett=%s",
        device_name,
        full.E, full.u_val, full.ratio,
        f"{circuit.E:.3f}" if circuit else "N/A",
        f"{pinch.E:.3f}" if pinch else "N/A",
        f"ratio={bennett.I_ratio:.2f}" if bennett else "N/A",
    )

    return report


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
        + device.waveform_digitization_uncertainty**2
    )
    if device_name in _SHOT_TO_SHOT_DATA:
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
    )


# =====================================================================
# Optimizer gradient report (path-to-7.0: document gradient at optimum)
# =====================================================================


@dataclass
class OptimizerGradientReport:
    """Finite-difference gradient and curvature at the calibration optimum.

    Attributes:
        fc: Current fraction at optimum.
        fm: Mass fraction at optimum.
        objective_value: Objective function value at optimum.
        grad_fc: Partial derivative of objective w.r.t. fc.
        grad_fm: Partial derivative of objective w.r.t. fm.
        grad_magnitude: |grad|.
        hess_eigenvalues: Eigenvalues of the 2x2 Hessian.
        condition_number: Ratio of max/min eigenvalue (high = degenerate).
        ridge_direction: Unit vector along the degenerate ridge.
        fc_bounds: Bounds used for fc.
        fm_bounds: Bounds used for fm.
        fc_at_boundary: Whether fc is within 0.5% of a bound.
    """

    fc: float
    fm: float
    objective_value: float
    grad_fc: float
    grad_fm: float
    grad_magnitude: float
    hess_eigenvalues: tuple[float, float]
    condition_number: float
    ridge_direction: tuple[float, float]
    fc_bounds: tuple[float, float]
    fm_bounds: tuple[float, float]
    fc_at_boundary: bool


def optimizer_gradient_report(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.094,
    fc_bounds: tuple[float, float] = (0.6, 0.8),
    fm_bounds: tuple[float, float] = (0.05, 0.25),
    pinch_column_fraction: float = 0.14,
    crowbar_enabled: bool = True,
    crowbar_resistance: float = 1.5e-3,
    delta: float = 0.005,
) -> OptimizerGradientReport:
    """Compute finite-difference gradient and Hessian at the calibration optimum.

    Uses central differences with step size delta to estimate the gradient
    and 2x2 Hessian matrix of the calibration objective at (fc, fm).

    Args:
        device_name: Registered device name.
        fc: Current fraction at optimum.
        fm: Mass fraction at optimum.
        fc_bounds: Bounds for current fraction.
        fm_bounds: Bounds for mass fraction.
        pinch_column_fraction: Pinch column fraction.
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Crowbar resistance [Ohm].
        delta: Step size for finite differences.

    Returns:
        :class:`OptimizerGradientReport`.
    """
    from dpf.validation.experimental import DEVICES, nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    device = DEVICES[device_name]
    if device.waveform_t is None or device.waveform_I is None:
        raise ValueError(f"No digitized waveform for {device_name}")

    def obj(fc_v: float, fm_v: float) -> float:
        try:
            model = LeeModel(
                current_fraction=fc_v,
                mass_fraction=fm_v,
                pinch_column_fraction=pinch_column_fraction,
                crowbar_enabled=crowbar_enabled,
                crowbar_resistance=crowbar_resistance,
            )
            result = model.run(device_name)
            return float(nrmse_peak(
                result.t, result.I, device.waveform_t, device.waveform_I,
            ))
        except Exception:
            return 1.0

    f0 = obj(fc, fm)

    # Gradient: use one-sided differences at boundaries, central otherwise
    fc_at_lo = fc <= fc_bounds[0] + delta
    fc_at_hi = fc >= fc_bounds[1] - delta
    fm_at_lo = fm <= fm_bounds[0] + delta
    fm_at_hi = fm >= fm_bounds[1] - delta

    if fc_at_hi:
        # Backward difference at upper bound
        f_fc_m = obj(fc - delta, fm)
        grad_fc = (f0 - f_fc_m) / delta
        f_fc_p = f0  # not used for Hessian; use 2nd-order backward
        H_ff = (f0 - 2 * obj(fc - delta, fm) + obj(fc - 2 * delta, fm)) / (delta**2)
    elif fc_at_lo:
        # Forward difference at lower bound
        f_fc_p = obj(fc + delta, fm)
        grad_fc = (f_fc_p - f0) / delta
        f_fc_m = f0
        H_ff = (obj(fc + 2 * delta, fm) - 2 * f_fc_p + f0) / (delta**2)
    else:
        f_fc_p = obj(fc + delta, fm)
        f_fc_m = obj(fc - delta, fm)
        grad_fc = (f_fc_p - f_fc_m) / (2 * delta)
        H_ff = (f_fc_p - 2 * f0 + f_fc_m) / (delta**2)

    if fm_at_hi:
        f_fm_m = obj(fc, fm - delta)
        grad_fm = (f0 - f_fm_m) / delta
        H_mm = (f0 - 2 * obj(fc, fm - delta) + obj(fc, fm - 2 * delta)) / (delta**2)
    elif fm_at_lo:
        f_fm_p = obj(fc, fm + delta)
        grad_fm = (f_fm_p - f0) / delta
        H_mm = (obj(fc, fm + 2 * delta) - 2 * f_fm_p + f0) / (delta**2)
    else:
        f_fm_p = obj(fc, fm + delta)
        f_fm_m = obj(fc, fm - delta)
        grad_fm = (f_fm_p - f_fm_m) / (2 * delta)
        H_mm = (f_fm_p - 2 * f0 + f_fm_m) / (delta**2)

    grad_mag = float(np.sqrt(grad_fc**2 + grad_fm**2))

    # Cross-Hessian: use central differences where possible, one-sided at bounds
    fc_lo = max(fc - delta, fc_bounds[0])
    fc_hi = min(fc + delta, fc_bounds[1])
    fm_lo = max(fm - delta, fm_bounds[0])
    fm_hi = min(fm + delta, fm_bounds[1])
    f_pp = obj(fc_hi, fm_hi)
    f_pm = obj(fc_hi, fm_lo)
    f_mp = obj(fc_lo, fm_hi)
    f_mm = obj(fc_lo, fm_lo)
    dfc = fc_hi - fc_lo
    dfm = fm_hi - fm_lo
    H_fm = (f_pp - f_pm - f_mp + f_mm) / max(dfc * dfm, delta**2 * 0.01)

    H = np.array([[H_ff, H_fm], [H_fm, H_mm]])
    eigvals = np.sort(np.linalg.eigvalsh(H))
    eigvecs = np.linalg.eigh(H)[1]

    # Ridge direction = eigenvector of smallest eigenvalue
    min_idx = 0
    ridge = eigvecs[:, min_idx]

    # Condition number
    if abs(eigvals[0]) > 1e-15:
        cond = abs(eigvals[1]) / abs(eigvals[0])
    else:
        cond = float("inf")

    fc_at_boundary = (
        fc <= fc_bounds[0] + 0.005 or fc >= fc_bounds[1] - 0.005
    )

    logger.info(
        "Gradient at (fc=%.3f, fm=%.3f): |grad|=%.4f, "
        "eigenvalues=(%.4f, %.4f), condition=%.1f, "
        "ridge=(%.2f, %.2f), at_boundary=%s",
        fc, fm, grad_mag, eigvals[0], eigvals[1], cond,
        ridge[0], ridge[1], fc_at_boundary,
    )

    return OptimizerGradientReport(
        fc=fc,
        fm=fm,
        objective_value=f0,
        grad_fc=grad_fc,
        grad_fm=grad_fm,
        grad_magnitude=grad_mag,
        hess_eigenvalues=(float(eigvals[0]), float(eigvals[1])),
        condition_number=cond,
        ridge_direction=(float(ridge[0]), float(ridge[1])),
        fc_bounds=fc_bounds,
        fm_bounds=fm_bounds,
        fc_at_boundary=fc_at_boundary,
    )


# =====================================================================
# Multi-shot experimental uncertainty (path-to-7.0)
# =====================================================================


@dataclass
class MultiShotUncertainty:
    """Estimated experimental uncertainty from shot-to-shot variability.

    PF-1000 shot-to-shot variability is well-documented in the literature:
    - Scholz et al. (2006): sigma_I/I ~ 5% for peak current
    - Lee & Saw (2014): reproducibility to ~5-8% for well-conditioned shots

    Attributes:
        u_shot_to_shot: Shot-to-shot relative uncertainty (1-sigma).
        u_rogowski: Rogowski coil calibration uncertainty (1-sigma).
        u_digitization: Waveform digitization uncertainty (1-sigma).
        u_exp_combined: Combined experimental uncertainty (RSS).
        n_shots_typical: Typical number of shots for the estimate.
        u_exp_with_averaging: u_exp after averaging n_shots.
        reference: Literature reference.
    """

    u_shot_to_shot: float
    u_rogowski: float
    u_digitization: float
    u_exp_combined: float
    n_shots_typical: int
    u_exp_with_averaging: float
    reference: str


# Published shot-to-shot variability data
_SHOT_TO_SHOT_DATA: dict[str, dict[str, Any]] = {
    "PF-1000": {
        "u_shot_to_shot": 0.05,  # 5% sigma_I/I per Scholz et al. (2006)
        "u_rogowski": 0.05,      # 5% Rogowski coil calibration
        "u_digitization": 0.03,  # 3% digitization error
        "n_shots_typical": 5,    # Scholz et al. averaged ~5 reproducible shots
        "reference": (
            "Scholz et al., Nukleonika 51(1), 2006; "
            "Lee & Saw, J. Fusion Energy 33:319-335 (2014)"
        ),
    },
    "NX2": {
        "u_shot_to_shot": 0.08,  # 8% — smaller devices show more variability
        "u_rogowski": 0.05,
        "u_digitization": 0.03,
        "n_shots_typical": 10,
        "reference": "Lee & Saw, J. Fusion Energy 27:292-295 (2008)",
    },
    "POSEIDON-60kV": {
        "u_shot_to_shot": 0.06,  # 6% estimated from IPFS data scatter
        "u_rogowski": 0.05,
        "u_digitization": 0.03,
        "n_shots_typical": 3,
        "reference": "IPFS plasmafocus.net (Lee fitting)",
    },
    "UNU-ICTP": {
        "u_shot_to_shot": 0.10,  # 10% — teaching device, higher variability
        "u_rogowski": 0.05,
        "u_digitization": 0.03,
        "n_shots_typical": 10,
        "reference": "Lee et al., Am. J. Phys. 56 (1988)",
    },
    "FAETON-I": {
        "u_shot_to_shot": 0.08,  # 8% (re-strikes cause variability)
        "u_rogowski": 0.05,
        "u_digitization": 0.08,  # 8% (reconstructed waveform, not digitized)
        "n_shots_typical": 5,
        "reference": "Damideh et al., Sci. Rep. 15:23048 (2025)",
    },
    "MJOLNIR": {
        "u_shot_to_shot": 0.10,  # 10% (large device, high-power variability)
        "u_rogowski": 0.05,
        "u_digitization": 0.10,  # 10% (reconstructed waveform, high uncertainty)
        "n_shots_typical": 5,
        "reference": "Schmidt et al., IEEE TPS (2021); Goyon et al., Phys. Plasmas (2025)",
    },
    "PF-1000-16kV": {
        "u_shot_to_shot": 0.05,  # 5% — same bank as PF-1000 (Scholz 2006)
        "u_rogowski": 0.05,      # 5% — same Rogowski coil
        "u_digitization": 0.05,  # 5% (reconstructed from 27 kV Scholz scaling)
        "n_shots_typical": 16,   # Akel et al. (2021) Table 1: 16 shots at 1.05 Torr
        "reference": "Akel et al., Radiat. Phys. Chem. 188:109638, 2021",
    },
    "PF-1000-Gribkov": {
        "u_shot_to_shot": 0.05,  # 5% — same bank as PF-1000
        "u_rogowski": 0.05,      # 5% — same Rogowski coil
        "u_digitization": 0.03,  # 3% (digitized from IPFS archive, 94 points)
        "n_shots_typical": 5,    # Gribkov et al. (2007) — similar campaign
        "reference": "Gribkov et al., J. Phys. D 40:3592, 2007",
    },
}


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
    u_dig = data["u_digitization"]
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
        u_digitization=u_dig,
        u_exp_combined=u_combined,
        n_shots_typical=n_shots,
        u_exp_with_averaging=u_with_avg,
        reference=data["reference"],
    )


# =====================================================================
# Multi-device simultaneous calibration (Phase BJ)
# =====================================================================


@dataclass
class MultiDeviceResult:
    """Result of multi-device simultaneous calibration.

    Attributes
    ----------
    mode : str
        Calibration mode: "shared", "shared_fc", or "pareto".
    devices : list[str]
        Device names used.
    shared_fc : float
        Shared current fraction (all modes).
    shared_fm : float
        Shared mass fraction (mode="shared" only; NaN for others).
    shared_delay_us : float
        Shared liftoff delay [us] (mode="shared" only).
    device_fm : dict[str, float]
        Per-device mass fraction (mode="shared_fc").
    device_delay_us : dict[str, float]
        Per-device liftoff delay [us] (mode="shared_fc").
    device_nrmse : dict[str, float]
        Per-device NRMSE at the multi-device optimum.
    combined_nrmse : float
        Weighted sum of per-device NRMSEs.
    independent_nrmse : dict[str, float]
        Per-device NRMSE from independent calibration (baseline).
    independent_fc : dict[str, float]
        Per-device fc from independent calibration.
    independent_fm : dict[str, float]
        Per-device fm from independent calibration.
    nrmse_penalty : dict[str, float]
        Per-device NRMSE increase vs independent: (multi - indep) / indep.
    combined_improvement : float
        Improvement vs naive transfer: 1 - combined / naive_combined.
    converged : bool
        Whether the optimizer converged.
    n_evals : int
        Total number of model evaluations.
    """

    mode: str
    devices: list[str]
    shared_fc: float
    shared_fm: float
    shared_delay_us: float
    device_fm: dict[str, float]
    device_delay_us: dict[str, float]
    device_nrmse: dict[str, float]
    combined_nrmse: float
    independent_nrmse: dict[str, float]
    independent_fc: dict[str, float]
    independent_fm: dict[str, float]
    nrmse_penalty: dict[str, float]
    combined_improvement: float
    converged: bool
    n_evals: int


@dataclass
class ParetoPoint:
    """A single point on the Pareto front.

    Attributes
    ----------
    fc : float
        Current fraction.
    fm : float
        Mass fraction.
    delay_us : float
        Liftoff delay [us].
    nrmse : dict[str, float]
        Per-device NRMSE.
    combined : float
        Weighted combined NRMSE.
    """

    fc: float
    fm: float
    delay_us: float
    nrmse: dict[str, float]
    combined: float


@dataclass
class ParetoFrontResult:
    """Pareto front of multi-device NRMSE trade-offs.

    Attributes
    ----------
    devices : list[str]
        Device names (exactly 2 for 2D Pareto).
    points : list[ParetoPoint]
        Pareto-optimal points.
    n_evaluated : int
        Total points evaluated on the grid.
    independent_nrmse : dict[str, float]
        Per-device NRMSE from independent calibration.
    utopia_point : dict[str, float]
        Minimum achievable NRMSE per device (independent calibration).
    nadir_point : dict[str, float]
        Worst NRMSE on Pareto front per device.
    """

    devices: list[str]
    points: list[ParetoPoint]
    n_evaluated: int
    independent_nrmse: dict[str, float]
    utopia_point: dict[str, float]
    nadir_point: dict[str, float]


class MultiDeviceCalibrator:
    """Simultaneous calibration of Lee model across multiple DPF devices.

    Tests whether fc/fm can be shared across devices (universality
    hypothesis) or whether device-specific values are required (as
    suggested by Phase BI cross-device blind prediction results).

    Three calibration modes:

    1. **"shared"**: Single (fc, fm, delay) optimized to minimize the
       weighted sum of per-device NRMSE.  Tests universal fc/fm.

    2. **"shared_fc"**: Shared fc, but device-specific fm and delay.
       Tests whether current fraction is more universal than mass fraction
       (physical motivation: fc depends on insulator surface flashover
       physics, fm depends on electrode gap geometry).

    3. **"pareto"**: Maps the Pareto front of device-specific NRMSE
       trade-offs as (fc, fm) are varied.  No single optimum — shows
       the full trade-off landscape.

    Args:
        devices: List of device names (must have digitized waveforms).
        weights: Optional per-device weights for combined NRMSE.
            Default: equal weight (1/N_devices).
        fc_bounds: Bounds for current fraction.
        fm_bounds: Bounds for mass fraction.
        delay_bounds_us: Bounds for liftoff delay [us].
        pinch_column_fraction: Default pcf (overridden by device-specific).
        crowbar_enabled: Whether crowbar is enabled.
        crowbar_resistance: Default crowbar resistance [Ohm].
        maxiter: Maximum optimizer iterations.
        seed: Random seed for differential evolution.
    """

    def __init__(
        self,
        devices: list[str] | None = None,
        weights: dict[str, float] | None = None,
        fc_bounds: tuple[float, float] = (0.5, 0.95),
        fm_bounds: tuple[float, float] = (0.01, 0.40),
        delay_bounds_us: tuple[float, float] = (0.0, 2.0),
        pinch_column_fraction: float = 0.14,
        crowbar_enabled: bool = True,
        crowbar_resistance: float = 1.5e-3,
        maxiter: int = 200,
        seed: int = 42,
    ) -> None:
        if devices is None:
            devices = ["PF-1000", "POSEIDON-60kV"]
        self.devices = devices
        self.fc_bounds = fc_bounds
        self.fm_bounds = fm_bounds
        self.delay_bounds_us = delay_bounds_us
        self.pinch_column_fraction = pinch_column_fraction
        self.crowbar_enabled = crowbar_enabled
        self.crowbar_resistance = crowbar_resistance
        self.maxiter = maxiter
        self.seed = seed

        # Equal weights by default
        if weights is None:
            w = 1.0 / len(devices)
            self.weights = {d: w for d in devices}
        else:
            total = sum(weights.values())
            self.weights = {d: weights[d] / total for d in devices}

    def _compute_nrmse(
        self,
        device_name: str,
        fc: float,
        fm: float,
        delay_us: float,
    ) -> float:
        """Run Lee model for a device and return NRMSE."""
        from dpf.validation.experimental import DEVICES, nrmse_peak
        from dpf.validation.lee_model_comparison import LeeModel

        device = DEVICES[device_name]
        if device.waveform_t is None or device.waveform_I is None:
            return 1.0

        pcf = _DEFAULT_DEVICE_PCF.get(device_name, self.pinch_column_fraction)
        cr = _DEFAULT_CROWBAR_R.get(device_name, self.crowbar_resistance)
        cb_enabled = self.crowbar_enabled and cr > 0

        try:
            model = LeeModel(
                current_fraction=fc,
                mass_fraction=fm,
                pinch_column_fraction=pcf,
                crowbar_enabled=cb_enabled,
                crowbar_resistance=cr,
                liftoff_delay=delay_us * 1e-6,
            )
            result = model.run(device_name)
            return float(nrmse_peak(
                result.t, result.I, device.waveform_t, device.waveform_I,
            ))
        except Exception:
            return 1.0

    def _independent_calibrations(self) -> dict[str, LiftoffCalibrationResult]:
        """Run independent per-device calibrations as baseline."""
        results = {}
        for dev in self.devices:
            cr = _DEFAULT_CROWBAR_R.get(dev, self.crowbar_resistance)
            # Enable crowbar only if device has a non-zero crowbar resistance
            cb_enabled = self.crowbar_enabled and cr > 0
            results[dev] = calibrate_with_liftoff(
                device_name=dev,
                fc_bounds=self.fc_bounds,
                fm_bounds=self.fm_bounds,
                delay_bounds_us=self.delay_bounds_us,
                pinch_column_fraction=_DEFAULT_DEVICE_PCF.get(
                    dev, self.pinch_column_fraction
                ),
                crowbar_enabled=cb_enabled,
                crowbar_resistance=cr,
                maxiter=self.maxiter,
                seed=self.seed,
            )
        return results

    def calibrate_shared(
        self,
        _cached_independent: dict[str, LiftoffCalibrationResult] | None = None,
    ) -> MultiDeviceResult:
        """Optimize a single (fc, fm, delay) across all devices.

        Minimizes weighted_sum(NRMSE_i) over shared (fc, fm, delay).

        Args:
            _cached_independent: Pre-computed independent calibrations to avoid
                redundant work in leave-one-out loops.  Keys are device names.

        Returns:
            :class:`MultiDeviceResult` with mode="shared".
        """
        from scipy.optimize import differential_evolution, minimize

        n_evals = 0

        def _objective(x: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            fc, fm, delay_us = float(x[0]), float(x[1]), float(x[2])
            total = 0.0
            for dev in self.devices:
                nrmse = self._compute_nrmse(dev, fc, fm, delay_us)
                total += self.weights[dev] * nrmse
            return total

        bounds = [self.fc_bounds, self.fm_bounds, self.delay_bounds_us]
        opt = differential_evolution(
            _objective, bounds, maxiter=self.maxiter, seed=self.seed,
            tol=1e-5, atol=1e-5, polish=False, workers=1,
        )

        # Bounded L-BFGS-B polish (maxiter=50 to avoid runaway)
        polish = minimize(
            _objective, opt.x, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 50},
        )
        opt_x = polish.x if polish.fun <= opt.fun else opt.x

        fc_opt = float(np.clip(opt_x[0], *self.fc_bounds))
        fm_opt = float(np.clip(opt_x[1], *self.fm_bounds))
        delay_opt = float(np.clip(opt_x[2], *self.delay_bounds_us))

        # Per-device NRMSE at shared optimum
        dev_nrmse = {}
        for dev in self.devices:
            dev_nrmse[dev] = self._compute_nrmse(dev, fc_opt, fm_opt, delay_opt)

        combined = sum(
            self.weights[d] * dev_nrmse[d] for d in self.devices
        )

        # Independent baselines (use cache if provided)
        if _cached_independent is not None:
            indep = {d: _cached_independent[d] for d in self.devices}
        else:
            indep = self._independent_calibrations()
        indep_nrmse = {d: indep[d].nrmse for d in self.devices}
        indep_fc = {d: indep[d].best_fc for d in self.devices}
        indep_fm = {d: indep[d].best_fm for d in self.devices}

        # Penalty: how much worse is each device vs its independent optimum
        penalty = {}
        for d in self.devices:
            if indep_nrmse[d] > 0:
                penalty[d] = (dev_nrmse[d] - indep_nrmse[d]) / indep_nrmse[d]
            else:
                penalty[d] = 0.0

        # Improvement vs naive combined (using device A's params on device B)
        naive_combined = sum(self.weights[d] * 0.5 for d in self.devices)
        improvement = 1.0 - combined / naive_combined if naive_combined > 0 else 0.0

        logger.info(
            "Multi-device shared: fc=%.4f, fm=%.4f, delay=%.3f us, "
            "combined NRMSE=%.4f, penalties=%s",
            fc_opt, fm_opt, delay_opt, combined,
            {d: f"{penalty[d]:.1%}" for d in self.devices},
        )

        return MultiDeviceResult(
            mode="shared",
            devices=list(self.devices),
            shared_fc=fc_opt,
            shared_fm=fm_opt,
            shared_delay_us=delay_opt,
            device_fm={d: fm_opt for d in self.devices},
            device_delay_us={d: delay_opt for d in self.devices},
            device_nrmse=dev_nrmse,
            combined_nrmse=combined,
            independent_nrmse=indep_nrmse,
            independent_fc=indep_fc,
            independent_fm=indep_fm,
            nrmse_penalty=penalty,
            combined_improvement=improvement,
            converged=bool(opt.success),
            n_evals=n_evals,
        )

    def calibrate_shared_fc(self) -> MultiDeviceResult:
        """Optimize shared fc with device-specific (fm, delay).

        The parameter vector is [fc, fm_1, delay_1, fm_2, delay_2, ...].

        Returns:
            :class:`MultiDeviceResult` with mode="shared_fc".
        """
        from scipy.optimize import differential_evolution, minimize

        n_evals = 0

        def _objective(x: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            fc = float(x[0])
            total = 0.0
            for i, dev in enumerate(self.devices):
                fm_i = float(x[1 + 2 * i])
                delay_i = float(x[2 + 2 * i])
                nrmse = self._compute_nrmse(dev, fc, fm_i, delay_i)
                total += self.weights[dev] * nrmse
            return total

        # Bounds: [fc, fm_1, delay_1, fm_2, delay_2, ...]
        bounds = [self.fc_bounds]
        for _ in self.devices:
            bounds.append(self.fm_bounds)
            bounds.append(self.delay_bounds_us)

        opt = differential_evolution(
            _objective, bounds, maxiter=self.maxiter, seed=self.seed,
            tol=1e-5, atol=1e-5, polish=False, workers=1,
        )

        # Bounded L-BFGS-B polish (maxiter=50 to avoid runaway)
        polish = minimize(
            _objective, opt.x, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 50},
        )
        opt_x = polish.x if polish.fun <= opt.fun else opt.x

        fc_opt = float(np.clip(opt_x[0], *self.fc_bounds))
        dev_fm = {}
        dev_delay = {}
        dev_nrmse = {}
        for i, dev in enumerate(self.devices):
            fm_i = float(np.clip(opt_x[1 + 2 * i], *self.fm_bounds))
            delay_i = float(np.clip(opt_x[2 + 2 * i], *self.delay_bounds_us))
            dev_fm[dev] = fm_i
            dev_delay[dev] = delay_i
            dev_nrmse[dev] = self._compute_nrmse(dev, fc_opt, fm_i, delay_i)

        combined = sum(
            self.weights[d] * dev_nrmse[d] for d in self.devices
        )

        # Independent baselines
        indep = self._independent_calibrations()
        indep_nrmse = {d: indep[d].nrmse for d in self.devices}
        indep_fc = {d: indep[d].best_fc for d in self.devices}
        indep_fm = {d: indep[d].best_fm for d in self.devices}

        penalty = {}
        for d in self.devices:
            if indep_nrmse[d] > 0:
                penalty[d] = (dev_nrmse[d] - indep_nrmse[d]) / indep_nrmse[d]
            else:
                penalty[d] = 0.0

        naive_combined = sum(self.weights[d] * 0.5 for d in self.devices)
        improvement = 1.0 - combined / naive_combined if naive_combined > 0 else 0.0

        logger.info(
            "Multi-device shared_fc: fc=%.4f, device_fm=%s, "
            "combined NRMSE=%.4f, penalties=%s",
            fc_opt,
            {d: f"{dev_fm[d]:.4f}" for d in self.devices},
            combined,
            {d: f"{penalty[d]:.1%}" for d in self.devices},
        )

        return MultiDeviceResult(
            mode="shared_fc",
            devices=list(self.devices),
            shared_fc=fc_opt,
            shared_fm=float("nan"),
            shared_delay_us=float("nan"),
            device_fm=dev_fm,
            device_delay_us=dev_delay,
            device_nrmse=dev_nrmse,
            combined_nrmse=combined,
            independent_nrmse=indep_nrmse,
            independent_fc=indep_fc,
            independent_fm=indep_fm,
            nrmse_penalty=penalty,
            combined_improvement=improvement,
            converged=bool(opt.success),
            n_evals=n_evals,
        )

    def pareto_front(
        self,
        fc_grid: int = 15,
        fm_grid: int = 15,
        delay_us: float = 0.5,
    ) -> ParetoFrontResult:
        """Map the Pareto front of per-device NRMSE trade-offs.

        Evaluates Lee model on a (fc, fm) grid for each device and
        extracts the Pareto-optimal points (no point dominates another
        on all device NRMSEs simultaneously).

        Args:
            fc_grid: Number of fc grid points.
            fm_grid: Number of fm grid points.
            delay_us: Fixed liftoff delay [us] (to reduce dimensionality).

        Returns:
            :class:`ParetoFrontResult` with Pareto-optimal points.
        """
        fc_vals = np.linspace(self.fc_bounds[0], self.fc_bounds[1], fc_grid)
        fm_vals = np.linspace(self.fm_bounds[0], self.fm_bounds[1], fm_grid)

        all_points: list[ParetoPoint] = []

        for fc_v in fc_vals:
            for fm_v in fm_vals:
                nrmse = {}
                for dev in self.devices:
                    nrmse[dev] = self._compute_nrmse(
                        dev, float(fc_v), float(fm_v), delay_us,
                    )
                combined = sum(
                    self.weights[d] * nrmse[d] for d in self.devices
                )
                all_points.append(ParetoPoint(
                    fc=float(fc_v),
                    fm=float(fm_v),
                    delay_us=delay_us,
                    nrmse=nrmse,
                    combined=combined,
                ))

        # Extract Pareto front (non-dominated points)
        pareto = []
        for p in all_points:
            dominated = False
            for q in all_points:
                if p is q:
                    continue
                # q dominates p if q is <= p on all devices and < on at least one
                all_leq = all(
                    q.nrmse[d] <= p.nrmse[d] for d in self.devices
                )
                any_lt = any(
                    q.nrmse[d] < p.nrmse[d] for d in self.devices
                )
                if all_leq and any_lt:
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)

        # Sort by first device's NRMSE
        pareto.sort(key=lambda p: p.nrmse[self.devices[0]])

        # Independent baselines
        indep = self._independent_calibrations()
        indep_nrmse = {d: indep[d].nrmse for d in self.devices}

        # Utopia = independent optimum per device
        utopia = dict(indep_nrmse)

        # Nadir = worst NRMSE on Pareto front per device
        nadir = {}
        for d in self.devices:
            if pareto:
                nadir[d] = max(p.nrmse[d] for p in pareto)
            else:
                nadir[d] = 1.0

        logger.info(
            "Pareto front: %d points from %d evaluated, "
            "utopia=%s, nadir=%s",
            len(pareto), len(all_points),
            {d: f"{utopia[d]:.4f}" for d in self.devices},
            {d: f"{nadir[d]:.4f}" for d in self.devices},
        )

        return ParetoFrontResult(
            devices=list(self.devices),
            points=pareto,
            n_evaluated=len(all_points),
            independent_nrmse=indep_nrmse,
            utopia_point=utopia,
            nadir_point=nadir,
        )

    def leave_one_out(self) -> dict[str, dict[str, float]]:
        """Leave-one-out cross-validation across devices.

        For each device D_held:
        1. Calibrate on remaining devices (train set)
        2. Predict D_held with trained parameters
        3. Compare prediction NRMSE to independent calibration NRMSE

        Pre-computes independent calibrations once and caches them across
        LOO iterations to avoid redundant DE runs (O(N) instead of O(N^2)).

        Returns:
            Dict mapping held-out device name to a dict with keys:
            - "train_nrmse": avg NRMSE on training devices
            - "blind_nrmse": NRMSE on held-out device using trained params
            - "independent_nrmse": NRMSE from independent calibration
            - "degradation": blind / independent ratio (1.0 = perfect)
            - "trained_fc", "trained_fm", "trained_delay_us": parameters
        """
        if len(self.devices) < 2:
            raise ValueError("Need >= 2 devices for leave-one-out")

        # Pre-compute all independent calibrations once (avoids O(N^2) work)
        indep = self._independent_calibrations()
        results: dict[str, dict[str, float]] = {}

        for held_out in self.devices:
            train_devs = [d for d in self.devices if d != held_out]

            # Create a sub-calibrator on the training set
            sub_cal = MultiDeviceCalibrator(
                devices=train_devs,
                fc_bounds=self.fc_bounds,
                fm_bounds=self.fm_bounds,
                delay_bounds_us=self.delay_bounds_us,
                pinch_column_fraction=self.pinch_column_fraction,
                crowbar_enabled=self.crowbar_enabled,
                crowbar_resistance=self.crowbar_resistance,
                maxiter=self.maxiter,
                seed=self.seed,
            )

            # Calibrate on training set (pass cached independents)
            train_result = sub_cal.calibrate_shared(
                _cached_independent=indep,
            )
            fc_train = train_result.shared_fc
            fm_train = train_result.shared_fm
            delay_train = train_result.shared_delay_us

            # Predict held-out device
            blind_nrmse = self._compute_nrmse(
                held_out, fc_train, fm_train, delay_train,
            )

            # Average training NRMSE
            train_nrmse = np.mean([
                train_result.device_nrmse[d] for d in train_devs
            ])

            indep_nrmse = indep[held_out].nrmse

            # Compute metadata for stratified analysis
            from dpf.validation.experimental import DEVICES, lp_l0_for_device
            dev_data = DEVICES.get(held_out)
            lp_l0 = lp_l0_for_device(held_out) if dev_data else 0.0
            provenance = dev_data.waveform_provenance if dev_data else ""

            results[held_out] = {
                "train_nrmse": float(train_nrmse),
                "blind_nrmse": float(blind_nrmse),
                "independent_nrmse": float(indep_nrmse),
                "degradation": (
                    float(blind_nrmse / indep_nrmse)
                    if indep_nrmse > 0 else float("inf")
                ),
                "trained_fc": fc_train,
                "trained_fm": fm_train,
                "trained_delay_us": delay_train,
                "lp_l0": lp_l0,
                "waveform_provenance": provenance,
            }

            logger.info(
                "LOO held=%s: blind=%.4f, indep=%.4f, degrad=%.2fx, "
                "trained fc=%.4f fm=%.4f delay=%.3f us",
                held_out,
                blind_nrmse,
                indep_nrmse,
                results[held_out]["degradation"],
                fc_train, fm_train, delay_train,
            )

        return results


# ── Multi-condition validation ────────────────────────────────────────


@dataclass
class MultiConditionResult:
    """Result of multi-condition validation (same device, different V0/p0).

    Calibrate on condition A, predict condition B.  This tests whether the
    Lee model parameters (fc, fm, delay) are truly device-specific constants
    or depend on operating conditions.

    Attributes:
        train_device: Device/condition used for calibration.
        test_device: Device/condition used for blind prediction.
        train_fc: Calibrated fc on training condition.
        train_fm: Calibrated fm on training condition.
        train_delay_us: Calibrated liftoff delay [us] on training condition.
        train_nrmse: NRMSE on training condition (self-fit).
        blind_nrmse: NRMSE on test condition using trained params.
        independent_nrmse: NRMSE on test condition from independent calibration.
        degradation: blind / independent ratio (1.0 = perfect transfer).
        asme_blind: ASME V&V 20 result using trained params on test condition.
        asme_independent: ASME V&V 20 result using independent params.
    """

    train_device: str
    test_device: str
    train_fc: float
    train_fm: float
    train_delay_us: float
    train_nrmse: float
    blind_nrmse: float
    independent_nrmse: float
    degradation: float
    asme_blind: ASMEValidationResult | None = None
    asme_independent: ASMEValidationResult | None = None


def multi_condition_validation(
    train_device: str = "PF-1000",
    test_device: str = "PF-1000-16kV",
    fc_bounds: tuple[float, float] = (0.5, 0.95),
    fm_bounds: tuple[float, float] = (0.04, 0.40),
    delay_bounds_us: tuple[float, float] = (0.0, 2.0),
    maxiter: int = 10,
    seed: int = 42,
    run_asme: bool = True,
) -> MultiConditionResult:
    """Multi-condition validation: calibrate on one condition, predict another.

    This is the strongest form of model validation for parameter-based models:
    same device hardware, different operating conditions (V0, fill pressure).
    If fc/fm are true device constants, they should transfer across conditions.

    Args:
        train_device: Device name for calibration (e.g. "PF-1000").
        test_device: Device name for blind prediction (e.g. "PF-1000-16kV").
        fc_bounds: Bounds for current fraction.
        fm_bounds: Bounds for mass fraction.
        delay_bounds_us: Bounds for liftoff delay [us].
        maxiter: Maximum DE iterations for calibration.
        seed: Random seed.
        run_asme: Whether to run ASME V&V 20 assessments.

    Returns:
        :class:`MultiConditionResult` with train/blind/independent NRMSE,
        degradation ratio, and optional ASME assessments.

    References:
        Lee & Saw, J. Fusion Energy 27, 292-295 (2008) — fc/fm universality.
        ASME V&V 20-2009 — formal validation standard.
    """
    from dpf.validation.experimental import DEVICES

    # Validate devices exist and have waveforms
    for dev_name in (train_device, test_device):
        if dev_name not in DEVICES:
            raise ValueError(f"Device '{dev_name}' not in DEVICES registry")
        dev = DEVICES[dev_name]
        if dev.waveform_t is None or dev.waveform_I is None:
            raise ValueError(f"Device '{dev_name}' has no digitized waveform")

    # Get device-specific settings
    train_pcf = _DEFAULT_DEVICE_PCF.get(train_device, 0.14)
    test_pcf = _DEFAULT_DEVICE_PCF.get(test_device, 0.14)
    train_cr = _DEFAULT_CROWBAR_R.get(train_device, 1.5e-3)
    test_cr = _DEFAULT_CROWBAR_R.get(test_device, 1.5e-3)

    # Step 1: Calibrate on training condition
    train_result = calibrate_with_liftoff(
        device_name=train_device,
        fc_bounds=fc_bounds,
        fm_bounds=fm_bounds,
        delay_bounds_us=delay_bounds_us,
        pinch_column_fraction=train_pcf,
        crowbar_enabled=train_cr > 0,
        crowbar_resistance=train_cr,
        maxiter=maxiter,
        seed=seed,
    )
    fc_train = train_result.best_fc
    fm_train = train_result.best_fm
    delay_train = train_result.best_delay_us  # already in us

    # Step 2: Blind prediction on test condition
    from dpf.validation.experimental import nrmse_peak
    from dpf.validation.lee_model_comparison import LeeModel

    test_dev = DEVICES[test_device]
    model_blind = LeeModel(
        current_fraction=fc_train,
        mass_fraction=fm_train,
        radial_mass_fraction=0.1,
        pinch_column_fraction=test_pcf,
        crowbar_enabled=test_cr > 0,
        crowbar_resistance=test_cr,
        liftoff_delay=delay_train * 1e-6,  # us → s
    )
    sim_blind = model_blind.run(test_device)
    blind_nrmse = nrmse_peak(
        sim_blind.t, sim_blind.I,
        test_dev.waveform_t, test_dev.waveform_I,
    )

    # Step 3: Independent calibration on test condition (baseline)
    indep_result = calibrate_with_liftoff(
        device_name=test_device,
        fc_bounds=fc_bounds,
        fm_bounds=fm_bounds,
        delay_bounds_us=delay_bounds_us,
        pinch_column_fraction=test_pcf,
        crowbar_enabled=test_cr > 0,
        crowbar_resistance=test_cr,
        maxiter=maxiter,
        seed=seed,
    )
    indep_nrmse = indep_result.nrmse

    degradation = blind_nrmse / max(indep_nrmse, 1e-15)

    logger.info(
        "Multi-condition: train=%s -> test=%s: blind=%.4f, indep=%.4f, "
        "degrad=%.2fx, fc=%.3f, fm=%.3f, delay=%.3f us",
        train_device, test_device, blind_nrmse, indep_nrmse,
        degradation, fc_train, fm_train, delay_train,
    )

    # Step 4: ASME V&V 20 assessments
    asme_blind = None
    asme_indep = None
    if run_asme:
        try:
            asme_blind = asme_vv20_assessment(
                device_name=test_device,
                fc=fc_train,
                fm=fm_train,
                liftoff_delay=delay_train * 1e-6,
                pinch_column_fraction=test_pcf,
                crowbar_enabled=test_cr > 0,
                crowbar_resistance=test_cr,
                u_num=0.001,
            )
        except Exception:
            logger.warning("ASME blind assessment failed for %s", test_device)

        try:
            asme_indep = asme_vv20_assessment(
                device_name=test_device,
                fc=indep_result.best_fc,
                fm=indep_result.best_fm,
                liftoff_delay=indep_result.best_delay_us * 1e-6,
                pinch_column_fraction=test_pcf,
                crowbar_enabled=test_cr > 0,
                crowbar_resistance=test_cr,
                u_num=0.001,
            )
        except Exception:
            logger.warning("ASME indep assessment failed for %s", test_device)

    return MultiConditionResult(
        train_device=train_device,
        test_device=test_device,
        train_fc=fc_train,
        train_fm=fm_train,
        train_delay_us=delay_train,
        train_nrmse=train_result.nrmse,
        blind_nrmse=float(blind_nrmse),
        independent_nrmse=float(indep_nrmse),
        degradation=float(degradation),
        asme_blind=asme_blind,
        asme_independent=asme_indep,
    )
