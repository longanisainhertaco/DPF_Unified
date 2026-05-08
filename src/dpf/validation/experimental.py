"""Multi-device experimental validation data for DPF simulations.

Provides published experimental parameters and measured observables for
well-characterised Dense Plasma Focus devices.  These can be used to:

    1. Configure a simulation to match a real device geometry and
       electrical parameters.
    2. Validate simulated current waveforms and neutron yields against
       published measurements.

Devices included:
    - **PF-1000** (IPPLM Warsaw, Poland) -- the largest DPF in Europe.
    - **NX2** (NIE Singapore) -- compact Mather-type DPF.
    - **UNU-ICTP PFF** -- the widely-replicated training device.
    - **AECS-PF2** (Atomic Energy Commission of Syria) -- moved to _REFERENCE_ONLY (Type 2 unfittable).

Usage::

    from dpf.validation.experimental import (
        PF1000_DATA, NX2_DATA, UNU_ICTP_DATA,
        validate_current_waveform,
        validate_neutron_yield,
        device_to_config_dict,
    )

    metrics = validate_current_waveform(t_sim, I_sim, "PF-1000")
    print(f"Peak current error: {metrics['peak_current_error']:.1%}")

Units: SI throughout.

This module is a backward-compatible re-export facade.  The implementation
lives in the sub-modules:

    experimental_device.py       — ExperimentalDevice dataclass
    experimental_waveforms.py    — Digitized/reconstructed waveform arrays
    experimental_devices.py      — Device instances, DEVICES registry, helpers
    experimental_diagnostics.py  — L_p/L0, speed factor, RLC timing diagnostics
    experimental_comparison.py   — NRMSE, validation functions, device_to_config_dict
"""

from __future__ import annotations

# Re-export everything that external code imports from this module.
# Keep this list in sync with __init__.py.
from dpf.validation.experimental_comparison import (  # noqa: F401
    _find_first_peak,
    device_to_config_dict,
    normalized_rmse,
    nrmse_peak,
    validate_current_waveform,
    validate_neutron_yield,
)
from dpf.validation.experimental_device import ExperimentalDevice  # noqa: F401
from dpf.validation.experimental_devices import (  # noqa: F401
    AECS_PF2_DATA,
    DEVICES,
    FAETON_DATA,
    MJOLNIR_DATA,
    NX2_DATA,
    PF1000_16KV_DATA,
    PF1000_20KV_DATA,
    PF1000_DATA,
    PF1000_GRIBKOV_DATA,
    POSEIDON_60KV_DATA,
    POSEIDON_DATA,
    UNU_ICTP_DATA,
    _REFERENCE_ONLY,
    get_devices_by_provenance,
    get_validation_ready_devices,
)
from dpf.validation.experimental_diagnostics import (  # noqa: F401
    _S_OPTIMAL_KA_CM_TORR,  # deprecated alias, kept for external callers
    _S_TYPICAL_PF1000,
    compute_bare_rlc_timing,
    compute_lp_l0_ratio,
    compute_speed_factor,
    lp_l0_for_device,
)
