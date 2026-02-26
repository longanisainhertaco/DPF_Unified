"""Validation suite for DPF simulations against experimental data."""

from dpf.validation.bennett_equilibrium import (
    bennett_btheta,
    bennett_current_density,
    bennett_current_from_temperature,
    bennett_density,
    bennett_line_density,
    bennett_pressure,
    create_bennett_state,
    verify_force_balance,
)
from dpf.validation.calibration import (
    CalibrationResult,
    LeeModelCalibrator,
    calibrate_default_params,
)
from dpf.validation.experimental import (
    DEVICES,
    NX2_DATA,
    PF1000_DATA,
    UNU_ICTP_DATA,
    ExperimentalDevice,
    device_to_config_dict,
    nrmse_peak,
    validate_current_waveform,
    validate_neutron_yield,
)
from dpf.validation.lee_model_comparison import (
    LeeModel,
    LeeModelComparison,
    LeeModelResult,
)
from dpf.validation.magnetized_noh import (
    compression_ratio,
    create_noh_state,
    noh_downstream,
    noh_exact_solution,
    noh_upstream,
    shock_velocity,
    verify_rankine_hugoniot,
)
from dpf.validation.suite import (
    DEVICE_REGISTRY,
    ValidationResult,
    ValidationSuite,
    config_hash,
    nrmse_range,
    relative_error,
)

__all__ = [
    "bennett_btheta",
    "bennett_current_density",
    "bennett_current_from_temperature",
    "bennett_density",
    "bennett_line_density",
    "bennett_pressure",
    "create_bennett_state",
    "verify_force_balance",
    "CalibrationResult",
    "DEVICE_REGISTRY",
    "DEVICES",
    "ExperimentalDevice",
    "LeeModel",
    "LeeModelCalibrator",
    "LeeModelComparison",
    "LeeModelResult",
    "NX2_DATA",
    "PF1000_DATA",
    "UNU_ICTP_DATA",
    "ValidationResult",
    "ValidationSuite",
    "calibrate_default_params",
    "config_hash",
    "device_to_config_dict",
    "nrmse_peak",
    "nrmse_range",
    "relative_error",
    "validate_current_waveform",
    "validate_neutron_yield",
    "compression_ratio",
    "create_noh_state",
    "noh_downstream",
    "noh_exact_solution",
    "noh_upstream",
    "shock_velocity",
    "verify_rankine_hugoniot",
]
