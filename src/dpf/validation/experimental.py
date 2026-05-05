"""Multi-device experimental validation data for DPF simulations.

Canonical public entry point for the ``dpf.validation.experimental*``
sub-package.  External code (and ``dpf.validation.__init__``) should
import from this module rather than from the underlying private
sub-modules; the sub-module split is an implementation detail that
may be refactored without breaking callers.

Provides published experimental parameters and measured observables
for well-characterised Dense Plasma Focus devices.  These can be used
to:

    1. Configure a simulation to match a real device geometry and
       electrical parameters.
    2. Validate simulated current waveforms and neutron yields against
       published measurements.

Devices included:
    - **PF-1000** (IPPLM Warsaw, Poland) -- the largest DPF in Europe.
    - **NX2** (NIE Singapore) -- compact Mather-type DPF.
    - **UNU-ICTP PFF** -- the widely-replicated training device.
    - **AECS-PF2** (Atomic Energy Commission of Syria) -- 2.8 kJ
      high-impedance DPF; moved to _REFERENCE_ONLY (Type 2 unfittable).
    - **FAETON-I** / **MJOLNIR** -- Fuse Energy / LLNL MA-class
      devices (synthetic waveforms; see experimental_waveforms.py).

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

Implementation lives in the private sub-modules:

    experimental_device.py       -- ExperimentalDevice dataclass
    experimental_waveforms.py    -- Digitized/reconstructed waveform arrays
    experimental_devices.py      -- Device instances, DEVICES registry, helpers
    experimental_diagnostics.py  -- L_p/L0, speed factor, RLC timing diagnostics
    experimental_comparison.py   -- NRMSE, validation functions, device_to_config_dict
"""

from __future__ import annotations

# Re-export everything that external code imports from this module.
# ``__all__`` is the canonical export list; callers should import from
# ``dpf.validation.experimental`` rather than the private sub-modules.
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
    _REFERENCE_ONLY,
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
    get_devices_by_provenance,
)
from dpf.validation.experimental_diagnostics import (  # noqa: F401
    _S_OPTIMAL_KA_CM_TORR,  # deprecated alias, kept for external callers
    _S_TYPICAL_PF1000,
    compute_bare_rlc_timing,
    compute_lp_l0_ratio,
    compute_speed_factor,
    lp_l0_for_device,
)

__all__ = [
    # Devices
    "AECS_PF2_DATA",
    "DEVICES",
    "ExperimentalDevice",
    "FAETON_DATA",
    "MJOLNIR_DATA",
    "NX2_DATA",
    "PF1000_16KV_DATA",
    "PF1000_20KV_DATA",
    "PF1000_DATA",
    "PF1000_GRIBKOV_DATA",
    "POSEIDON_60KV_DATA",
    "POSEIDON_DATA",
    "UNU_ICTP_DATA",
    "get_devices_by_provenance",
    # Comparison / validation helpers
    "_find_first_peak",
    "device_to_config_dict",
    "normalized_rmse",
    "nrmse_peak",
    "validate_current_waveform",
    "validate_neutron_yield",
    # Diagnostics
    "_S_OPTIMAL_KA_CM_TORR",
    "compute_bare_rlc_timing",
    "compute_lp_l0_ratio",
    "compute_speed_factor",
    "lp_l0_for_device",
]
