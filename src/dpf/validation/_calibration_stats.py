"""Monte Carlo uncertainty propagation, bootstrap CI, and Bennett equilibrium."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


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


_ATOMS_PER_MOLECULE = {
    # Lee 2014 JFE 33:319 Eq. 32 "DN = dissociation number, e.g. for
    # Deuterium DN = 2, whereas for argon DN = 1".
    "deuterium": 2,
    "d2": 2,
    "hydrogen": 2,
    "h2": 2,
    "tritium": 2,
    "t2": 2,
    "dt": 2,
    "nitrogen": 2,
    "n2": 2,
    "oxygen": 2,
    "o2": 2,
    "argon": 1,
    "ar": 1,
    "neon": 1,
    "ne": 1,
    "helium": 1,
    "he": 1,
    "xenon": 1,
    "xe": 1,
    "krypton": 1,
    "kr": 1,
}


def _atoms_per_molecule(fill_gas: str) -> int:
    """Return number of atoms per fill-gas molecule (Lee 2014 Eq. 32 DN)."""
    key = fill_gas.strip().lower()
    if key in _ATOMS_PER_MOLECULE:
        return _ATOMS_PER_MOLECULE[key]
    # Unknown gas: fall back to monatomic (conservative for noble gases).
    # Caller can override by passing atoms_per_molecule explicitly.
    import warnings
    warnings.warn(
        f"Unknown fill_gas={fill_gas!r}; defaulting to atoms_per_molecule=1. "
        "Pass atoms_per_molecule explicitly for H2/D2/T2/DT/N2/O2.",
        stacklevel=3,
    )
    return 1


# EMPIRICAL: _DEFAULT_COMPRESSION_RATIO = 10 is a round-number estimate
# matching the r_min = 0.1 * a clamp used in lee_model_comparison.py::pinch_event
# (terminal event at r_shock = 0.1 * anode_radius).  Lee & Saw 2008 (JoFE 27:292,
# Table 1) reports empirical k_min = r_p/a in [0.14, 0.21] across PF400, UNU,
# NX2, PF1000 and Poseidon -- i.e. actual simulated compression ratios of
# 4.8-7.1 rather than 10.  Callers who need a device-specific value should
# pass compression_ratio explicitly; the default 10 is held for back-
# compatibility with existing Bennett-equilibrium checks.
_DEFAULT_COMPRESSION_RATIO = 10.0


def bennett_equilibrium_check(
    device_name: str = "PF-1000",
    fc: float = 0.800,
    fm: float = 0.094,
    pinch_column_fraction: float = 0.14,
    compression_ratio: float = _DEFAULT_COMPRESSION_RATIO,
    T_assumed_eV: float | None = None,
    tolerance: float = 0.5,
    atoms_per_molecule: int | None = None,
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
        compression_ratio: Ratio of anode radius to pinch radius (a / r_p).
            Default 10 (r_pinch = a/10), matching the r_min clamp in
            ``lee_model_comparison.py``.  See module-level EMPIRICAL note:
            Lee & Saw 2008 Table 1 observes k_min = r_p/a in [0.14, 0.21]
            across PF400 -> Poseidon, i.e. actual CR of 4.8-7.1.
        T_assumed_eV: If given, use this temperature [eV] instead of
            estimating from kinetics.
        tolerance: Fractional tolerance for consistency check.
            Default 0.5 (I_ratio within 0.5-1.5).
        atoms_per_molecule: Number of atoms per fill-gas molecule (Lee
            2014 JFE 33:319 Eq. 32 "DN").  If None (default), looked up
            from ``device.fill_gas``.  DN=2 for H2/D2/T2/N2/O2; DN=1
            for monatomic gases (He/Ne/Ar/Kr/Xe).

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

    # Resolve atoms-per-molecule (Lee 2014 Eq. 32 dissociation number DN)
    if atoms_per_molecule is None:
        atoms_per_molecule = _atoms_per_molecule(getattr(device, "fill_gas", "deuterium"))

    # Swept mass and pinch density.
    # n_fill is the molecular density.  A diatomic fill (e.g. D2) contributes
    # two ions per molecule upon full dissociation; a noble gas (Ar/Ne/He)
    # contributes one.  See Lee 2014 p. 327 Eq. 32.
    V_annular = np.pi * (b**2 - a**2) * z_pinch
    n_particles = atoms_per_molecule * n_fill * V_annular * fm
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
