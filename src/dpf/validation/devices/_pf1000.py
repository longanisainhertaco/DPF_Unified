"""PF-1000 device preset (IPPLM Warsaw, 27 kV / 3.5 Torr D2).

Migrated from `dpf.validation.experimental_devices` per the D2 per-device
split (Option A). Numerical values are unchanged from the source module.
The canonical import path remains `from dpf.validation.experimental_devices
import PF1000_DATA`; this module is the single source of truth and the
parent module re-exports it.
"""

from __future__ import annotations

from dpf.validation.experimental_device import ExperimentalDevice
from dpf.validation.experimental_waveforms import (
    _PF1000_WAVEFORM_I_MA,
    _PF1000_WAVEFORM_T_US,
)

PF1000_DATA = ExperimentalDevice(
    name="PF-1000",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # 1.332 mF
    voltage=27e3,                  # 27 kV
    inductance=25e-9,              # 25 nH (Akel et al. 2021 Table 1 - 24 shots; text: "Bank: L0 = 25 nH")
    resistance=2.3e-3,             # 2.3 mOhm bare-bank short-circuit (Scholz 2006 Table 1). Wave-10 RCA: Akel's 6.1 mOhm includes plasma-phase resistance which is double-counted when used as bank R; plasma R enters via sheath model.
    anode_radius=0.115,            # 115 mm outer radius (IPPLM: anode OD 230mm)
    cathode_radius=0.16,           # 160 mm effective (Lee & Saw 2014; rods at 200mm)
    anode_length=0.48,             # 480 mm (Akel et al. 2021 p.1: "PF-1000 plasma focus has 480 mm long coaxial electrodes"; Table 1: z0 = 48 cm)
    fill_pressure_torr=3.5,
    fill_gas="deuterium",
    peak_current=1.87e6,           # 1.87 MA
    neutron_yield=1e11,
    current_rise_time=5.8e-6,      # 5.8 us
    reference="Scholz et al., Nukleonika 51(1), 2006",
    lee_fc=0.7, lee_fm=0.13, lee_fmr=0.35, lee_fcr=0.65,
    lee_reference=(
        "Malek et al., Plasma Physics and Technology 12(1):9 (2025) "
        "[KR: plasma-physics-and-technology-1211-9-2025.md §3 lines 177-180]: "
        "fm=0.13, fc=0.7, fmr=0.35, fcr=0.65 obtained by fitting computed and "
        "measured current waveforms at 27 kV / 3.5 Torr D2 in PF-1000."
    ),
    crowbar_resistance=1.5e-3,     # 1.5 mOhm (spark gap arc, PhD Debate #30)
    peak_current_uncertainty=0.05,     # 5% (Rogowski coil + calibration)
    rise_time_uncertainty=0.10,        # 10% (quarter-period timing)
    neutron_yield_uncertainty=0.50,    # 50% (shot-to-shot variability)
    waveform_t=_PF1000_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_PF1000_WAVEFORM_I_MA * 1e6,        # Convert MA -> A
    # Digitization uncertainty for hand-digitized Fig. 2 of Scholz et al. (2006).
    # Amplitude: ±3% (trace width ~0.06 MA on ~2 MA full scale).
    # Time: ±0.5% of full scale (~0.05 us on 10 us trace).
    # Combined current uncertainty: sqrt(5%^2 + 3%^2) = 5.8% (1-sigma).
    waveform_amplitude_uncertainty=0.03,  # 3% amplitude from trace reading
    waveform_time_uncertainty=0.005,      # 0.5% of full scale (~0.05 us)
    waveform_uncertainty_type="digitization",  # Type B: hand-digitized from published figure
    waveform_provenance="measured",
    measurement_notes=(
        "26 points hand-digitized from Scholz et al., Nukleonika 51(1), 2006, Fig. 2. "
        "Rogowski coil uncertainty ~5% (Type B, estimated — not stated in source). "
        "Digitization amplitude uncertainty ~3% (Type B, trace width / full scale). "
        "Combined waveform uncertainty: u_I = sqrt(0.05^2 + 0.03^2) = 5.8% (1-sigma). "
        "Temporal uncertainty ~0.05 us (Type B, 0.5% of 10 us trace). "
        "Effective independent data points ~5 (autocorrelation time ~1-2 us on 10 us trace). "
        "Scholz (2006) does not state measurement uncertainty; values above are estimates. "
        "Framework: ASME V&V 20-2009 for validation, GUM (JCGM 100:2008) for measurement."
    ),
)
