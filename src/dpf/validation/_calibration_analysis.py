"""Analysis tools: FIM, cross-validation, NRMSE decomposition, summary reports."""

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
from dpf.validation._calibration_data import _DEFAULT_DEVICE_PCF
from dpf.validation._calibration_stats import (
    BennettEquilibriumResult,
    bennett_equilibrium_check,
)

logger = logging.getLogger(__name__)


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
    calibration: object
    prediction_peak_error: float
    prediction_timing_error: float
    generalization_score: float


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
        + device.waveform_amplitude_uncertainty**2
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
