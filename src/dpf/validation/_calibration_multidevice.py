"""Multi-device and multi-condition calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dpf.validation._calibration_advanced import (
    LiftoffCalibrationResult,
    calibrate_with_liftoff,
)
from dpf.validation._calibration_asme import ASMEValidationResult, asme_vv20_assessment
from dpf.validation._calibration_data import _DEFAULT_CROWBAR_R, _DEFAULT_DEVICE_PCF

logger = logging.getLogger(__name__)


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
