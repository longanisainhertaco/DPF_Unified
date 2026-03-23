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

# Re-export everything from the split sub-modules so that all existing
# ``from dpf.validation.calibration import X`` statements continue to work.
from dpf.validation._calibration_advanced import (
    BlindPredictionResult,
    CircuitOnlyCalibrationResult,
    LiftoffCalibrationResult,
    blind_predict,
    calibrate_with_liftoff,
    circuit_only_calibration,
)
from dpf.validation._calibration_analysis import (
    CrossValidationResult,
    CrossValidator,
    FIMResult,
    NRMSEDecomposition,
    OptimizerGradientReport,
    ValidationSummaryReport,
    fisher_information_matrix,
    nrmse_timing_amplitude_decomposition,
    optimizer_gradient_report,
    validation_summary,
)
from dpf.validation._calibration_asme import (
    ASMEStratifiedSummary,
    ASMEValidationResult,
    MultiShotUncertainty,
    _pinch_phase_asme,
    asme_stratified_summary,
    asme_vv20_assessment,
    multi_shot_uncertainty,
)
from dpf.validation._calibration_core import (
    LeeModelCalibrator,
    calibrate_default_params,
)
from dpf.validation._calibration_data import (
    _DEFAULT_CROWBAR_R,
    _DEFAULT_DEVICE_PCF,
    _PUBLISHED_FC_FM_RANGES,
    _SHOT_TO_SHOT_DATA,
    CalibrationResult,
)
from dpf.validation._calibration_multidevice import (
    MultiConditionResult,
    MultiDeviceCalibrator,
    MultiDeviceResult,
    ParetoFrontResult,
    ParetoPoint,
    multi_condition_validation,
)
from dpf.validation._calibration_stats import (
    BennettEquilibriumResult,
    BootstrapCIResult,
    MonteCarloNRMSEResult,
    _estimate_block_size,
    bennett_equilibrium_check,
    bootstrap_calibration,
    monte_carlo_nrmse,
)

__all__ = [
    # Core
    "CalibrationResult",
    "LeeModelCalibrator",
    "calibrate_default_params",
    # Device data tables
    "_PUBLISHED_FC_FM_RANGES",
    "_DEFAULT_DEVICE_PCF",
    "_DEFAULT_CROWBAR_R",
    "_SHOT_TO_SHOT_DATA",
    # ASME
    "ASMEValidationResult",
    "ASMEStratifiedSummary",
    "MultiShotUncertainty",
    "asme_vv20_assessment",
    "asme_stratified_summary",
    "_pinch_phase_asme",
    "multi_shot_uncertainty",
    # Advanced calibration
    "CircuitOnlyCalibrationResult",
    "LiftoffCalibrationResult",
    "BlindPredictionResult",
    "circuit_only_calibration",
    "calibrate_with_liftoff",
    "blind_predict",
    # Statistical tools
    "MonteCarloNRMSEResult",
    "BootstrapCIResult",
    "BennettEquilibriumResult",
    "monte_carlo_nrmse",
    "_estimate_block_size",
    "bootstrap_calibration",
    "bennett_equilibrium_check",
    # Analysis tools
    "FIMResult",
    "CrossValidationResult",
    "NRMSEDecomposition",
    "ValidationSummaryReport",
    "OptimizerGradientReport",
    "fisher_information_matrix",
    "CrossValidator",
    "nrmse_timing_amplitude_decomposition",
    "validation_summary",
    "optimizer_gradient_report",
    # Multi-device
    "MultiDeviceResult",
    "ParetoPoint",
    "ParetoFrontResult",
    "MultiDeviceCalibrator",
    "MultiConditionResult",
    "multi_condition_validation",
]
