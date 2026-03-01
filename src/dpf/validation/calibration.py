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
}


_DEFAULT_DEVICE_PCF: dict[str, float] = {
    "PF-1000": 0.14,
    "PF-1000-16kV": 0.14,
    "PF-1000-20kV": 0.14,
    "NX2": 0.5,
    "POSEIDON": 0.14,  # Similar to PF-1000 (Lee & Saw 2014 scaling)
    "POSEIDON-60kV": 0.14,  # Lee & Saw scaling for MA-class
}

# Default crowbar spark gap arc resistance [Ohm] per device.
# PhD Debate #30 Finding 4: R_crowbar=0 is physically incorrect and
# systematically biases fc upward during calibration.
# PF-1000: ~1-3 mOhm for ignitron/spark gap (Dr. PP estimate).
_DEFAULT_CROWBAR_R: dict[str, float] = {
    "PF-1000": 1.5e-3,  # 1.5 mOhm midpoint of 1-3 mOhm range
    "POSEIDON-60kV": 1.5e-3,  # estimated, same as PF-1000
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
