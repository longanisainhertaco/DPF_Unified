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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dpf.constants import k_B, m_d, m_D2

# =====================================================================
# Experimental device dataclass
# =====================================================================

@dataclass
class ExperimentalDevice:
    """Published experimental data for a Dense Plasma Focus device.

    Attributes
    ----------
    name : str
        Device name.
    institution : str
        Host institution / laboratory.
    capacitance : float
        Bank capacitance [F].
    voltage : float
        Charging voltage [V].
    inductance : float
        External (stray) inductance [H].
    resistance : float
        External (stray) resistance [Ohm].
    anode_radius : float
        Anode radius [m].
    cathode_radius : float
        Cathode radius [m].
    anode_length : float
        Anode length [m].
    fill_pressure_torr : float
        Fill gas pressure [Torr].
    fill_gas : str
        Fill gas species (e.g. ``"deuterium"``).
    peak_current : float
        Measured peak discharge current [A].
    neutron_yield : float
        Measured total DD neutron yield per shot.
    current_rise_time : float
        Measured current quarter-period (time to first peak) [s].
    reference : str
        Publication reference string.
    """

    name: str
    institution: str
    capacitance: float        # [F]
    voltage: float            # [V]
    inductance: float         # [H]
    resistance: float         # [Ohm]
    anode_radius: float       # [m]
    cathode_radius: float     # [m]
    anode_length: float       # [m]
    fill_pressure_torr: float
    fill_gas: str
    peak_current: float       # [A]
    neutron_yield: float
    current_rise_time: float  # [s]
    reference: str
    # Experimental uncertainties (1-sigma, relative)
    # Following GUM (JCGM 100:2008) and ASME V&V 20-2009 uncertainty framework.
    peak_current_uncertainty: float = 0.0   # Relative uncertainty on peak current
    rise_time_uncertainty: float = 0.0      # Relative uncertainty on rise time
    neutron_yield_uncertainty: float = 0.0  # Relative uncertainty on neutron yield
    # Digitized waveform data (optional)
    waveform_t: np.ndarray | None = None    # Time array [s]
    waveform_I: np.ndarray | None = None    # Current array [A]
    # Waveform digitization uncertainty (1-sigma, relative)
    waveform_digitization_uncertainty: float = 0.0  # Amplitude digitization error
    waveform_time_uncertainty: float = 0.0          # Temporal digitization error
    # Crowbar switch resistance [Ohm] — physical arc resistance of the
    # crowbar spark gap.  Default 0.0 for backward compatibility.
    # PF-1000: ~1-3 mOhm (spark gap arc, PhD Debate #30 Finding 4).
    crowbar_resistance: float = 0.0
    # Measurement provenance note
    measurement_notes: str = ""


# =====================================================================
# Device data
# =====================================================================

# PF-1000 digitized I(t) from Scholz et al., Nukleonika 51(1), 2006, Fig. 2
# 26 points covering 0-10 us, interpolated from published waveform
# Characteristic features: rise to ~1.87 MA at ~5.8 us, current dip at ~7 us
_PF1000_WAVEFORM_T_US = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.3, 5.6, 5.8, 6.0, 6.3, 6.5, 6.8, 7.0, 7.3,
    7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
])
_PF1000_WAVEFORM_I_MA = np.array([
    0.00, 0.15, 0.35, 0.58, 0.82, 1.05, 1.25, 1.42, 1.56, 1.67,
    1.76, 1.81, 1.85, 1.87, 1.86, 1.82, 1.75, 1.55, 1.40, 1.30,
    1.25, 1.15, 1.05, 0.95, 0.85, 0.75,
])

PF1000_DATA = ExperimentalDevice(
    name="PF-1000",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # 1.332 mF
    voltage=27e3,                  # 27 kV
    inductance=33.5e-9,            # 33.5 nH (short-circuit calibration, Scholz 2006)
    resistance=2.3e-3,             # 2.3 mOhm (short-circuit discharge, Scholz 2006 Table 1)
    anode_radius=0.115,            # 115 mm outer radius (IPPLM: anode OD 230mm)
    cathode_radius=0.16,           # 160 mm effective (Lee & Saw 2014; rods at 200mm)
    anode_length=0.60,             # 600 mm (IPPLM: anode length 60 cm)
    fill_pressure_torr=3.5,
    fill_gas="deuterium",
    peak_current=1.87e6,           # 1.87 MA
    neutron_yield=1e11,
    current_rise_time=5.8e-6,      # 5.8 us
    reference="Scholz et al., Nukleonika 51(1), 2006",
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
    waveform_digitization_uncertainty=0.03,  # 3% amplitude from trace reading
    waveform_time_uncertainty=0.005,         # 0.5% of full scale (~0.05 us)
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

NX2_DATA = ExperimentalDevice(
    name="NX2",
    institution="NIE Singapore",
    capacitance=28e-6,             # 28 uF
    voltage=11.5e3,                # 11.5 kV operating voltage (Lee & Saw 2008)
    inductance=20e-9,              # 20 nH (RADPF Module 1)
    resistance=2.3e-3,             # 2.3 mOhm (RADPF; RESF=0.1)
    anode_radius=0.019,            # 19 mm
    cathode_radius=0.041,          # 41 mm
    anode_length=0.05,             # 50 mm
    fill_pressure_torr=3.0,        # 3 Torr D2 (Lee & Saw 2008)
    fill_gas="deuterium",
    peak_current=400e3,            # 400 kA (Lee & Saw 2008)
    neutron_yield=1e8,
    current_rise_time=1.8e-6,      # 1.8 us
    reference="Lee & Saw, J. Fusion Energy 27:292, 2008; RADPF Module 1",
    peak_current_uncertainty=0.08,     # 8% (compact device, lower SNR)
    rise_time_uncertainty=0.12,        # 12%
    neutron_yield_uncertainty=0.60,    # 60% (shot-to-shot)
    measurement_notes=(
        "No digitized waveform available. Peak current and rise time from "
        "Lee & Saw, J. Fusion Energy 27:292, 2008. R0=2.3 mOhm from RADPF "
        "Module 1 preset (plasmafocus.net); actual RESF=R0/sqrt(L0/C)=0.086 "
        "(not 0.1 as sometimes stated). "
        "Fill pressure 3 Torr D2 for neutron operation. "
        "WARNING: The 400 kA peak current is likely a RADPF model output, "
        "not a direct Rogowski coil measurement. The unloaded RLC peak is "
        "402.5 kA (implying only 0.6% plasma loading, which is physically "
        "implausible for any DPF discharge). Treat as 'reference' quality, "
        "not 'experimental measurement' quality. "
        "L0 uncertainty: literature reports 15-20 nH (Sahyouni et al. 2021 "
        "DOI:10.1155/2021/6611925 vs RADPF preset). "
        "Uncertainties are Type B estimates (not stated in source)."
    ),
)

# UNU-ICTP PFF measured I(t) from IPFS "UNU ICTPPFF D2 05.15.xls"
# 45 points covering 0-5 us at 13.5 kV, 3.0 Torr D2
# Median-filtered to remove EMI spike at pinch (~2.72-2.73 us)
# Characteristic features: rise to ~169 kA at ~2.2-2.6 us, shallow 14% dip at ~2.76 us
_UNU_ICTP_WAVEFORM_T_US = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.65, 2.70, 2.73,
    2.76, 2.80, 2.85, 2.90, 2.95, 3.0, 3.1, 3.2, 3.3, 3.5,
    3.7, 4.0, 4.3, 4.5, 5.0,
])
_UNU_ICTP_WAVEFORM_I_KA = np.array([
    8.7, 18.8, 28.1, 40.6, 56.3, 65.6, 73.8, 84.4, 93.8, 103.1,
    112.5, 112.5, 121.9, 131.3, 140.6, 140.6, 150.0, 150.0, 159.4, 159.4,
    159.4, 161.9, 168.8, 168.8, 168.8, 168.8, 168.8, 164.4, 159.4, 151.3,
    145.0, 153.8, 155.0, 150.0, 150.0, 148.8, 150.0, 140.6, 140.6, 131.3,
    121.9, 112.5, 103.1, 93.8, 63.1,
])

UNU_ICTP_DATA = ExperimentalDevice(
    name="UNU-ICTP",
    institution="UNU-ICTP PFF",
    capacitance=30e-6,             # 30 uF
    voltage=13.5e3,                # 13.5 kV (IPFS measured waveform conditions)
    inductance=110e-9,             # 110 nH
    resistance=12e-3,              # 12 mOhm
    anode_radius=0.0095,           # 9.5 mm
    cathode_radius=0.032,          # 32 mm
    anode_length=0.16,             # 160 mm
    fill_pressure_torr=3.0,
    fill_gas="deuterium",
    peak_current=169e3,            # 169 kA (from digitized waveform)
    neutron_yield=1e8,
    current_rise_time=2.2e-6,      # ~2.2 us to peak (from waveform)
    reference=(
        "Lee et al., Am. J. Phys. 56, 1988; "
        "IPFS plasmafocus.net 'UNU ICTPPFF D2 05.15.xls'"
    ),
    peak_current_uncertainty=0.10,     # 10% (training device, less precise)
    rise_time_uncertainty=0.15,        # 15%
    neutron_yield_uncertainty=0.70,    # 70% (shot-to-shot)
    waveform_t=_UNU_ICTP_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_UNU_ICTP_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_digitization_uncertainty=0.016,  # GUM: 9.3 kA / (2*sqrt(3)*169 kA) = 1.6% (rectangular)
    waveform_time_uncertainty=0.002,         # 0.2% (~1 ns digitization on ~5 us trace)
    measurement_notes=(
        "45 points from IPFS 'UNU ICTPPFF D2 05.15.xls' (plasmafocus.net). "
        "Original: 5556 points at ~1 ns resolution, digitized oscilloscope trace. "
        "Quantization: 9.3 kA steps (5.5% of 169 kA peak). "
        "GUM (JCGM 100:2008) rectangular distribution: u = step/(2*sqrt(3)) = 1.6%. "
        "EMI spike at pinch time (2.72-2.73 us) removed by median filtering. "
        "Smoothed with 15-sample uniform filter + 51-sample median filter. "
        "V0=13.5 kV (from IPFS file, not 14 kV sometimes quoted). "
        "Lee model params from IPFS: fm=0.08, fc=0.7, fmr=0.16, fcr=0.7. "
        "Uncertainties are Type B estimates. Rogowski coil uncertainty ~10%. "
        "Combined waveform uncertainty: u_I = sqrt(0.10^2 + 0.016^2) = 10.0%."
    ),
)


# PF-1000 at 16 kV — Akel et al., Radiat. Phys. Chem. 188:109638, 2021
# Same device (IPPLM Warsaw), different operating conditions:
#   V0 = 16 kV (vs 27 kV), fill pressure = 1.05 Torr D2 (vs 3.5 Torr)
# Peak current measured at 1.1-1.3 MA for multiple shots.
# Waveform reconstructed from known physics scaling + published I_peak constraint:
#   - Same bank (C0, L0, R0, T/4 = 10.49 us) → identical circuit dynamics
#   - I_peak = 1.2 MA (Akel 2021 Table 1, avg of 16 shots at 1.05 Torr)
#   - Loading ratio I_peak/I_RLC = 1.2/3.19 = 0.376 (cf. 27 kV: 1.87/5.38 = 0.347)
#   - Current dip shifted earlier (~5.5 us vs ~7.0 us) due to lower fill pressure
#     (1.05 Torr → lighter sheath → faster axial rundown → earlier pinch)
#   - Scaled from 27 kV Scholz waveform shape with peak/timing adjustments
# NOTE: Replace with actual digitized data from Akel (2021) Fig. 3 when available.
_PF1000_16KV_WAVEFORM_T_US = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.3, 5.5, 5.7, 5.8, 6.0, 6.3, 6.5, 7.0, 7.5,
    8.0, 8.5, 9.0, 9.5, 10.0,
])
_PF1000_16KV_WAVEFORM_I_MA = np.array([
    0.00, 0.10, 0.23, 0.38, 0.54, 0.69, 0.82, 0.93, 1.02, 1.10,
    1.16, 1.19, 1.20, 1.18, 1.12, 1.00, 0.90, 0.85, 0.78, 0.72,
    0.66, 0.60, 0.54, 0.49, 0.44,
])

PF1000_16KV_DATA = ExperimentalDevice(
    name="PF-1000-16kV",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # Same bank
    voltage=16e3,                  # 16 kV (reduced from 27 kV)
    inductance=33.5e-9,            # Same circuit
    resistance=2.3e-3,             # Same circuit
    anode_radius=0.115,            # Same geometry
    cathode_radius=0.16,           # Same geometry
    anode_length=0.60,             # Same geometry
    fill_pressure_torr=1.05,       # 1.05 Torr D2 (Akel 2021)
    fill_gas="deuterium",
    peak_current=1.2e6,            # 1.2 MA (midpoint of 1.1-1.3 MA range)
    neutron_yield=2.33e9,          # 2.33e9 n/shot at 1.05 Torr (average of 16 shots)
    current_rise_time=6.0e-6,      # ~6 us (estimated from Lee model fit in paper)
    reference="Akel et al., Radiat. Phys. Chem. 188:109638, 2021",
    crowbar_resistance=1.5e-3,     # Same crowbar as 27 kV (PhD Debate #30)
    peak_current_uncertainty=0.10,     # 10% (range 1.1-1.3 MA = ±8.3%)
    rise_time_uncertainty=0.15,        # 15% (no explicit timing stated)
    neutron_yield_uncertainty=0.40,    # 40% (shot-to-shot, Akel Table 1)
    waveform_t=_PF1000_16KV_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_PF1000_16KV_WAVEFORM_I_MA * 1e6,        # Convert MA -> A
    waveform_digitization_uncertainty=0.05,  # 5% (physics-scaled estimate, not digitized)
    waveform_time_uncertainty=0.01,          # 1% temporal (pinch timing estimated)
    measurement_notes=(
        "PF-1000 operated at 16 kV (170.5 kJ) with 1.05 Torr D2 fill. "
        "Peak current 1.1-1.3 MA across 16 shots (Akel et al. 2021 Table 1). "
        "WAVEFORM NOTE: Reconstructed from physics scaling of 27 kV Scholz (2006) "
        "waveform, constrained by published I_peak=1.2 MA. Same bank (C0, L0, R0), "
        "so T/4=10.49 us is identical. Current dip shifted earlier (~5.5 us vs ~7.0 us) "
        "due to lower fill pressure (1.05 Torr vs 3.5 Torr → faster sheath). "
        "Waveform_digitization_uncertainty set to 5% (higher than 3% for 27 kV) to "
        "account for reconstruction uncertainty. Replace with actual digitized data "
        "from Akel (2021) Fig. 3 when paper access is obtained. "
        "DOI: 10.1016/j.radphyschem.2021.109638"
    ),
)


# PF-1000 at 27 kV — Gribkov et al., J. Phys. D: Appl. Phys. 40:3592, 2007
# INDEPENDENTLY DIGITIZED waveform from plasmafocus.net RADPF archive.
# Same device and operating conditions as Scholz (2006), but DIFFERENT shot
# and DIFFERENT digitization source. Peak 1.846 MA at 6.39 us.
# 94 data points (vs 26 for Scholz), covering -1.68 to 14.73 us.
# Source: plasmafocus.net/IPFS/machines/PF1000 05.15.xls, Sheet2
# Original: Gribkov et al., J. Phys. D: Appl. Phys. 40:3592-3607, 2007
# Use for cross-publication validation: calibrate on Scholz, predict Gribkov.
_PF1000_GRIBKOV_WAVEFORM_T_US = np.array([
    -1.682, -1.169, -0.599, -0.285, 0.000, 0.085, 0.141, 0.198, 0.310,
    0.367, 0.592, 0.648, 0.732, 0.845, 0.930, 1.072, 1.099, 1.213,
    1.354, 1.496, 1.581, 1.638, 1.837, 2.064, 2.262, 2.518, 2.660,
    2.745, 2.859, 3.143, 3.342, 3.428, 3.569, 3.712, 3.911, 4.139,
    4.338, 4.509, 4.652, 4.822, 4.908, 5.193, 5.421, 5.677, 5.905,
    6.133, 6.390, 6.590, 6.818, 6.932, 7.103, 7.331, 7.445, 7.702,
    7.874, 8.017, 8.246, 8.390, 8.477, 8.563, 8.593, 8.679, 8.737,
    8.881, 8.995, 9.109, 9.280, 9.423, 9.536, 9.651, 9.965, 10.222,
    10.365, 10.622, 10.907, 11.136, 11.136, 11.335, 11.564, 11.564,
    11.792, 12.049, 12.049, 12.534, 12.705, 12.934, 13.105, 13.362,
    13.562, 13.819, 14.019, 14.304, 14.532, 14.732,
])
_PF1000_GRIBKOV_WAVEFORM_I_KA = np.array([
    -12.188, -22.772, -11.396, -16.646, -10.959, 49.377, 93.254,
    148.090, 235.843, 290.679, 504.542, 581.295, 652.590, 707.467,
    762.323, 844.619, 899.433, 954.311, 1031.130, 1102.460, 1129.920,
    1157.360, 1201.340, 1267.260, 1349.600, 1382.660, 1421.120,
    1475.980, 1503.460, 1596.820, 1591.480, 1618.940, 1662.880,
    1701.340, 1706.970, 1734.530, 1751.110, 1784.120, 1767.780,
    1811.740, 1811.800, 1822.970, 1817.660, 1839.760, 1845.410,
    1845.580, 1845.760, 1840.430, 1840.600, 1840.000, 1829.600,
    1835.490, 1830.100, 1820.000, 1790.000, 1748.320, 1655.340,
    1584.210, 1507.560, 1430.910, 1370.660, 1315.930, 1261.180,
    1173.610, 1160.000, 1173.780, 1168.420, 1146.610, 1171.000,
    1135.820, 1130.570, 1125.280, 1081.550, 1076.250, 1049.060,
    1032.790, 1032.790, 1032.940, 1016.670, 1016.670, 1000.400,
    989.625, 989.625, 940.664, 935.310, 908.080, 897.246, 875.516,
    859.223, 848.452, 821.201, 810.450, 794.179, 794.325,
])

# Trim to t >= 0 for consistency with other waveforms
_gribkov_mask = _PF1000_GRIBKOV_WAVEFORM_T_US >= 0.0
_PF1000_GRIBKOV_T_TRIMMED = _PF1000_GRIBKOV_WAVEFORM_T_US[_gribkov_mask]
_PF1000_GRIBKOV_I_TRIMMED = _PF1000_GRIBKOV_WAVEFORM_I_KA[_gribkov_mask]

PF1000_GRIBKOV_DATA = ExperimentalDevice(
    name="PF-1000-Gribkov",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,
    voltage=27e3,
    inductance=33.5e-9,
    resistance=2.3e-3,
    anode_radius=0.115,
    cathode_radius=0.16,
    anode_length=0.60,
    fill_pressure_torr=3.5,
    fill_gas="deuterium",
    peak_current=1.846e6,           # 1.846 MA (Gribkov 2007, different shot from Scholz)
    neutron_yield=1e11,
    current_rise_time=6.39e-6,      # 6.39 us (peak timing from data)
    reference="Gribkov et al., J. Phys. D: Appl. Phys. 40:3592, 2007",
    crowbar_resistance=1.5e-3,
    peak_current_uncertainty=0.05,
    rise_time_uncertainty=0.10,
    neutron_yield_uncertainty=0.50,
    waveform_t=_PF1000_GRIBKOV_T_TRIMMED * 1e-6,    # us -> s
    waveform_I=_PF1000_GRIBKOV_I_TRIMMED * 1e3,      # kA -> A
    waveform_digitization_uncertainty=0.02,  # 2% (digital oscilloscope, not hand-digitized)
    waveform_time_uncertainty=0.003,         # 0.3% (digital acquisition)
    measurement_notes=(
        "94-point digitized waveform from plasmafocus.net RADPF archive (PF1000 05.15.xls). "
        "Original source: Gribkov et al., J. Phys. D: Appl. Phys. 40:3592-3607, 2007. "
        "Same device and conditions as Scholz (2006) PF-1000 at 27 kV, 3.5 Torr D2, "
        "but DIFFERENT shot and DIFFERENT digitization. Peak 1.846 MA at 6.39 us "
        "(vs Scholz: 1.87 MA at 5.8 us — shot-to-shot variability). "
        "Lower digitization uncertainty (2%) because this is from digital oscilloscope data "
        "archived in the Lee model RADPF package, not hand-digitized from a paper figure. "
        "DOI: 10.1088/0022-3727/40/12/008"
    ),
)


# POSEIDON (Stuttgart) — 480 kJ Mather-type DPF
# One of the largest Mather-type DPF devices, operated at IPF Stuttgart
# (now retired).  Published I(t) and neutron yield data available from
# Herold et al., Nuclear Fusion 29:33 (1989) and subsequent publications.
# Parameters from: Herold et al. (1989), Lee & Saw (2014) RADPF fitting.
# Electrode geometry: anode diameter 208 mm, cathode diameter 270 mm,
# anode length 470 mm (confirmed from multiple published sources).
POSEIDON_DATA = ExperimentalDevice(
    name="POSEIDON",
    institution="IPF Stuttgart",
    capacitance=450e-6,            # 450 uF (H. Herold, private comm.; Lee RADPF)
    voltage=40e3,                  # 40 kV typical operation (360 kJ stored)
    inductance=20e-9,              # 20 nH (very low, MA-class design)
    resistance=2e-3,               # ~2 mOhm (estimated from RESF ~0.05)
    anode_radius=0.104,            # 104 mm (208 mm diameter; Herold 1989)
    cathode_radius=0.135,          # 135 mm (270 mm diameter; Herold 1989)
    anode_length=0.47,             # 470 mm (Herold 1989)
    fill_pressure_torr=3.5,        # 3.5 Torr D2 (typical neutron operation)
    fill_gas="deuterium",
    peak_current=2.6e6,            # 2.6 MA (Herold et al. 1989, at 40 kV)
    neutron_yield=1e11,            # ~10^11 (Herold 1989)
    current_rise_time=5.0e-6,      # ~5 us (estimated from Lee model quarter-period)
    reference="Herold et al., Nucl. Fusion 29:33, 1989; Lee & Saw, J. Fusion Energy 33:319, 2014",
    peak_current_uncertainty=0.08,     # 8% (large device, Rogowski + integration)
    rise_time_uncertainty=0.15,        # 15% (not explicitly stated in source)
    neutron_yield_uncertainty=0.50,    # 50% (shot-to-shot)
    measurement_notes=(
        "POSEIDON (IPF Stuttgart): large Mather-type DPF, operated 1980s-2000s. "
        "480 kJ at 46 kV max, typically 360 kJ at 40 kV (0.5*450uF*40kV^2). "
        "Peak current ~2.6 MA at 40 kV from Herold et al. (1989). "
        "Electrode geometry: anode diameter 208 mm, cathode diameter 270 mm, "
        "anode length 470 mm (Herold 1989, confirmed by multiple published sources). "
        "L0=20 nH and R0=2 mOhm are estimates from RADPF default configuration; "
        "not directly stated in Herold (1989). "
        "Uncertainties are Type B estimates. "
        "This device has L_p/L0 >> 1 (plasma-significant). "
        "DOI (Herold 1989): 10.1088/0029-5515/29/1/005"
    ),
)

# POSEIDON at 60 kV — IPFS (plasmafocus.net) digitized I(t) waveform
# Different bank configuration from POSEIDON (40 kV): C=156 uF, V=60 kV
# Electrode geometry: a=65.5 mm, b=95 mm, zo=300 mm (Lee model fitted)
# Source: plasmafocus.net/IPFS/machines/poseidon%2005.15.xls
# Lee model fit: fm=0.275, fc=0.595, fmr=0.45, fcr=0.44
# 35 subsampled points from 103-point digitized waveform
_POSEIDON60KV_WAVEFORM_T_US = np.array([
    0.007, 0.092, 0.148, 0.205, 0.261, 0.339, 0.395, 0.452, 0.530, 0.608,
    0.686, 0.764, 0.849, 0.927, 1.027, 1.141, 1.262, 1.405, 1.577, 1.770,
    1.978, 2.186, 2.394, 2.537, 2.603, 2.675, 2.734, 2.814, 2.929, 3.123,
    3.281, 3.439, 3.619, 3.770, 3.914,
])
_POSEIDON60KV_WAVEFORM_I_KA = np.array([
    0, 267, 499, 697, 918, 1130, 1290, 1460, 1660, 1850,
    2010, 2170, 2330, 2460, 2620, 2760, 2890, 3010, 3110, 3170,
    3190, 3180, 3150, 3050, 2890, 2680, 2490, 2280, 2140, 2100,
    1990, 1890, 1800, 1700, 1580,
])

POSEIDON_60KV_DATA = ExperimentalDevice(
    name="POSEIDON-60kV",
    institution="IPF Stuttgart",
    capacitance=156e-6,            # 156 uF (IPFS Lee model fit)
    voltage=60e3,                  # 60 kV (IPFS configuration)
    inductance=17.7e-9,            # 17.7 nH (Lee model fitted value)
    resistance=1.7e-3,             # 1.7 mOhm (IPFS Lee model fit)
    anode_radius=0.0655,           # 65.5 mm (IPFS: a=6.55 cm)
    cathode_radius=0.095,          # 95 mm (IPFS: b=9.5 cm)
    anode_length=0.30,             # 300 mm (IPFS: zo=30 cm, Lee model fitted)
    fill_pressure_torr=3.8,        # 3.8 Torr D2 (IPFS)
    fill_gas="deuterium",
    peak_current=3.19e6,           # 3.19 MA (IPFS digitized peak)
    neutron_yield=1e11,            # ~10^11 (estimated, same order as 40 kV)
    current_rise_time=1.98e-6,     # 1.98 us (time of peak from waveform)
    reference="IPFS (plasmafocus.net); Herold et al., Nucl. Fusion 29:33, 1989",
    crowbar_resistance=1.5e-3,     # estimated spark gap
    peak_current_uncertainty=0.05,     # 5% (Rogowski coil)
    rise_time_uncertainty=0.05,        # 5% (well-digitized waveform)
    neutron_yield_uncertainty=0.50,    # 50% (shot-to-shot)
    waveform_t=_POSEIDON60KV_WAVEFORM_T_US * 1e-6,    # Convert us -> s
    waveform_I=_POSEIDON60KV_WAVEFORM_I_KA * 1e3,      # Convert kA -> A
    waveform_digitization_uncertainty=0.02,  # 2% (IPFS digitization, high quality)
    waveform_time_uncertainty=0.005,         # 0.5% temporal
    measurement_notes=(
        "POSEIDON at 60 kV / 156 uF (E0=280.8 kJ) with 3.8 Torr D2 fill. "
        "Digitized I(t) waveform from IPFS (plasmafocus.net) Excel file. "
        "35 subsampled points from 103-point original. Peak 3.19 MA at 1.98 us. "
        "Electrode geometry: a=65.5 mm, b=95 mm — DIFFERENT from POSEIDON 40 kV "
        "(a=104 mm, b=135 mm). This is a different bank/electrode configuration "
        "of the same physical device. Lee model fitted: fm=0.275, fc=0.595, "
        "fmr=0.45, fcr=0.44, L0=17.7 nH, R0=1.7 mOhm, zo=300 mm. "
        "Source: S. Lee, IPFS (Institute for Plasma Focus Studies). "
        "DOI (parent): 10.1088/0029-5515/29/1/005"
    ),
)


# PF-1000 at 20 kV — from PF-1000 voltage scan
# Same device, different operating conditions: V0=20 kV, 2.0 Torr D2
# Peak current estimated from Akel et al. (2021) trend and Lee model.
PF1000_20KV_DATA = ExperimentalDevice(
    name="PF-1000-20kV",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # Same bank
    voltage=20e3,                  # 20 kV
    inductance=33.5e-9,            # Same circuit
    resistance=2.3e-3,             # Same circuit
    anode_radius=0.115,            # Same geometry
    cathode_radius=0.16,           # Same geometry
    anode_length=0.60,             # Same geometry
    fill_pressure_torr=2.0,        # 2.0 Torr D2 (interpolated)
    fill_gas="deuterium",
    peak_current=1.4e6,            # 1.4 MA (estimated from voltage scaling)
    neutron_yield=5e9,             # estimated
    current_rise_time=6.3e-6,      # ~6.3 us (estimated)
    reference="Akel et al., Radiat. Phys. Chem. 188:109638, 2021 (voltage trend)",
    crowbar_resistance=1.5e-3,
    peak_current_uncertainty=0.12,     # 12% (interpolated, higher uncertainty)
    rise_time_uncertainty=0.15,
    neutron_yield_uncertainty=0.50,
    measurement_notes=(
        "PF-1000 at 20 kV / 2.0 Torr D2 — interpolated from voltage scan trend. "
        "Peak current 1.4 MA estimated from Akel et al. (2021) multi-voltage data. "
        "Not a direct measurement — higher uncertainty than 27 kV reference."
    ),
)


# FAETON-I (Fuse Energy) — 100 kV, 125 kJ, ~1 MA dense plasma focus
# Damideh et al., Scientific Reports 15:23048 (2025)
# DOI: 10.1038/s41598-025-07939-x
# 5 x 5 uF = 25 uF Marx bank, V0=100 kV direct-charge
# Static inductance L0 ~ 220 nH, R0 ~ 7.6 mOhm (estimated from damping)
# Anode radius 5 cm, cathode radius ~10 cm (5 cm A-K gap), anode length 17 cm
# Insulator: 6.5 cm MACOR
# Fill gas: D2 at 10-40 Torr (optimal 12 Torr for neutrons)
# Peak current ~1 MA at T/4 ~ 3.6 us
# No crowbar switch — current oscillates freely
# WAVEFORM: Reconstructed from damped RLC (C=25uF, L=220nH, R=7.6mOhm)
# with 4% current dip at pinch (~4.1 us). FAETON-I is extremely
# circuit-dominated (L_p/L0 = 0.107) so plasma loading is minimal.
# Replace with digitized data from Damideh (2025) Fig. 3 when available.
_FAETON_WAVEFORM_T_US = np.array([
    0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7,
    3.0, 3.2, 3.4, 3.6, 3.7, 3.8, 4.0, 4.2, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.4,
])
_FAETON_WAVEFORM_I_KA = np.array([
    0.0, 135.3, 267.0, 393.0, 511.3, 620.1, 717.6, 802.5, 873.5, 929.6,
    969.9, 987.9, 998.5, 991.4, 983.7, 973.1, 949.1, 932.3, 913.7, 829.1,
    694.8, 531.4, 346.8, 149.9, -10.5,
])

FAETON_DATA = ExperimentalDevice(
    name="FAETON-I",
    institution="Fuse Energy Technologies",
    capacitance=25e-6,             # 25 uF (5 x 5 uF Marx)
    voltage=100e3,                 # 100 kV direct-charge
    inductance=220e-9,             # 220 nH static inductance (Damideh 2025)
    resistance=7.6e-3,             # 7.6 mOhm (estimated from I_peak damping)
    anode_radius=0.05,             # 50 mm (Damideh 2025)
    cathode_radius=0.10,           # ~100 mm (estimated from 5 cm A-K gap)
    anode_length=0.17,             # 170 mm (Damideh 2025)
    fill_pressure_torr=12.0,       # 12 Torr D2 (optimal for neutron yield)
    fill_gas="deuterium",
    peak_current=1.0e6,            # ~1 MA (Damideh 2025)
    neutron_yield=2.5e10,          # 2.5e10 D-D n/shot typical (8e10 peak)
    current_rise_time=3.6e-6,      # 3.6 us (T/4 from RLC parameters)
    reference=(
        "Damideh et al., Scientific Reports 15:23048, 2025; "
        "DOI: 10.1038/s41598-025-07939-x"
    ),
    crowbar_resistance=0.0,        # No crowbar switch
    peak_current_uncertainty=0.08, # 8% (Rogowski coil + Marx jitter)
    rise_time_uncertainty=0.10,    # 10% (not precisely stated)
    neutron_yield_uncertainty=0.50,  # 50% (shot-to-shot + re-strikes)
    waveform_t=_FAETON_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_FAETON_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_digitization_uncertainty=0.08,  # 8% (reconstructed, not digitized)
    waveform_time_uncertainty=0.02,          # 2% temporal (reconstructed)
    measurement_notes=(
        "FAETON-I: 100 kV, 125 kJ DPF by Fuse Energy Technologies. "
        "Highest direct-charged voltage PF device. 5 x 5 uF Marx bank = 25 uF total. "
        "Static inductance L0 = 220 nH (Damideh 2025). R0 = 7.6 mOhm estimated from "
        "measured I_peak/I_sc ratio (I_peak ~ 1 MA vs I_sc_undamped = 1.066 MA, RESF = 0.081). "
        "Cathode radius ~10 cm estimated from stated 5 cm A-K gap. "
        "WAVEFORM: RECONSTRUCTED from damped RLC parameters, NOT digitized from paper. "
        "L_p/L0 = 0.107 — extremely circuit-dominated; plasma loading is minimal. "
        "The reconstructed waveform is essentially a bare damped sinusoid with 4% pinch dip. "
        "Damideh (2025) uses modified Lee model with two-step radial fitting for re-strikes. "
        "Replace with digitized data from Damideh (2025) Fig. 3 when full paper is obtained. "
        "Uncertainties on waveform are higher than digitized sources (8% vs 2-3%). "
        "Fill pressure 12 Torr D2 is optimal for neutron yield (range 10-40 Torr). "
        "Best pinch voltage measured at 194 kV. Peak neutron yield 8e10 at 12 Torr."
    ),
)


# MJOLNIR (LLNL) — 2 MJ MA-class deuterium DPF at 60 kV typical operation
# Schmidt et al., IEEE TPS (2021) DOI: 10.1109/TPS.2021.3106313
# Schmidt et al., IEEE TPS (2024) DOI: 10.1109/TPS.2024.3471791
# Goyon et al., Phys. Plasmas 32:033105 (2025)
# ATLAS-heritage Marx: 24 modules, 2 x 34 uF each → C_erected = 408 uF
# Typical operation 60 kV (734 kJ), peak current 2.8 MA
# Design: 100 kV (2.04 MJ), 4.5 MA target
# Anode: 228.6 mm OD (9"), implosion radius 7.6 cm
# Cathode: 24 rods, estimated inner radius ~15.7 cm (4.3 cm A-K gap)
# Anode effective length: 18.3-22.1 cm (Petrov et al. 2022)
# WAVEFORM: Reconstructed from known peak current (2.8 MA), rise time (~5 us),
# and estimated circuit parameters. Phenomenological DPF shape with
# sinusoidal rise, 22% current dip at pinch, crowbar L-R decay.
# L0 = 80 nH estimated from loading factor I_peak/I_sc ~ 0.65.
# Replace with digitized data from Schmidt (2021) or Goyon (2025) figures.
_MJOLNIR_WAVEFORM_T_US = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.3,
    4.5, 4.7, 5.0, 5.2, 5.5, 5.8, 6.0, 6.5, 7.0, 7.5,
    8.0, 8.5, 9.0, 9.5, 10.0,
])
_MJOLNIR_WAVEFORM_I_KA = np.array([
    0, 438, 865, 1271, 1646, 1980, 2265, 2495, 2663, 2733,
    2766, 2788, 2800, 2554, 2184, 2318, 2408, 2329, 2253, 2179,
    2107, 2038, 1972, 1907, 1844,
])

MJOLNIR_DATA = ExperimentalDevice(
    name="MJOLNIR",
    institution="Lawrence Livermore National Laboratory",
    capacitance=408e-6,            # 408 uF (24 Marx modules, 2 x 34 uF each, erected)
    voltage=60e3,                  # 60 kV typical operation (734 kJ stored)
    inductance=80e-9,              # 80 nH (estimated from I_peak/I_sc ~ 0.65)
    resistance=1.4e-3,             # 1.4 mOhm (RESF ~ 0.1)
    anode_radius=0.076,            # 76 mm (implosion radius, Schmidt 2021)
    cathode_radius=0.157,          # ~157 mm (estimated from 4.3 cm A-K gap + anode OD)
    anode_length=0.20,             # 200 mm (midpoint of 183-221 mm range, Petrov 2022)
    fill_pressure_torr=7.0,        # 7 Torr D2 (estimated, pressure scans performed)
    fill_gas="deuterium",
    peak_current=2.8e6,            # 2.8 MA at 60 kV (Goyon 2025)
    neutron_yield=3.8e11,          # 3.8e11 D-D at 1 MJ / 2.5 MA (Schmidt 2021)
    current_rise_time=5.0e-6,      # ~5 us (Schmidt 2024)
    reference=(
        "Schmidt et al., IEEE TPS (2021) DOI: 10.1109/TPS.2021.3106313; "
        "Goyon et al., Phys. Plasmas 32:033105, 2025; "
        "Petrov et al., Phys. Plasmas 29:062708, 2022"
    ),
    crowbar_resistance=1.5e-3,     # estimated spark gap resistance
    peak_current_uncertainty=0.08, # 8% (Rogowski coil + integration)
    rise_time_uncertainty=0.10,    # 10% (stated as ~5 us, not precise)
    neutron_yield_uncertainty=0.50,  # 50% (shot-to-shot)
    waveform_t=_MJOLNIR_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_MJOLNIR_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_digitization_uncertainty=0.10,  # 10% (reconstructed, high uncertainty)
    waveform_time_uncertainty=0.03,          # 3% temporal (reconstructed)
    measurement_notes=(
        "MJOLNIR (MegaJOuLe Neutron Imaging Radiography): MA-class DPF at LLNL. "
        "ATLAS-heritage pulsed power: 24 Marx modules, 2 x 34 uF caps each, single-stage "
        "erection. C_erected = 24 x 17 uF = 408 uF. Charged to +/- 50 kV (100 kV erected). "
        "Typical operation at 60 kV (E = 734 kJ). Design: 100 kV / 2.04 MJ / 4.5 MA. "
        "Electrode: oxygen-free copper. 228.6 mm OD anode, 24-rod cathode, 6.5 cm MACOR insulator. "
        "Implosion radius 7.6 cm (Schmidt 2021). A-K gap 4.3 cm (Petrov 2022). "
        "Anode effective lengths: 18.3-22.1 cm (multiple anodes fielded). "
        "L0 = 80 nH ESTIMATED from loading factor I_peak/I_sc ~ 0.65 (NOT measured directly). "
        "84 flexible transmission line cables from Marx towers to disk collector add stray inductance. "
        "WAVEFORM: RECONSTRUCTED (phenomenological), NOT digitized from paper. "
        "Rise: sinusoidal to 2.8 MA at 5 us. Dip: 22% at pinch (5.5 us). "
        "Post-dip: crowbar L-R decay with ~15 us effective time constant. "
        "Uncertainties are higher than digitized sources (10% amplitude, 3% temporal). "
        "Replace with digitized data from Schmidt (2021) Fig. 4-5 or Goyon (2025) when available. "
        "Performance records: 2.5 MA / 3.8e11 DD neutrons at 1 MJ (Schmidt 2021); "
        "3.7-3.8 MA / >1e12 DD neutrons with rebuilt 24-module bank (Schmidt 2024); "
        "1.84e12 DT neutrons at 2 MA (Schmidt 2024). "
        "LLNL uses Chicago PIC code for simulation, not Lee model."
    ),
)


# Registry mapping device name -> ExperimentalDevice
DEVICES: dict[str, ExperimentalDevice] = {
    "PF-1000": PF1000_DATA,
    "PF-1000-Gribkov": PF1000_GRIBKOV_DATA,
    "PF-1000-16kV": PF1000_16KV_DATA,
    "PF-1000-20kV": PF1000_20KV_DATA,
    "NX2": NX2_DATA,
    "UNU-ICTP": UNU_ICTP_DATA,
    "POSEIDON": POSEIDON_DATA,
    "POSEIDON-60kV": POSEIDON_60KV_DATA,
    "FAETON-I": FAETON_DATA,
    "MJOLNIR": MJOLNIR_DATA,
}


# =====================================================================
# L_p / L0 diagnostic (Debate #29)
# =====================================================================

def compute_lp_l0_ratio(
    L0: float,
    anode_radius: float,
    cathode_radius: float,
    anode_length: float,
) -> dict[str, float]:
    """Compute the plasma-to-circuit inductance ratio L_p/L0.

    This diagnostic determines whether a DPF device's validation is
    informative (plasma-significant) or vacuously true (circuit-dominated).

    The axial plasma inductance at the end of the anode is::

        L_p = (mu_0 / 2pi) * ln(b/a) * z_max

    where *a* is anode radius, *b* is cathode radius, *z_max* is anode
    length.

    PhD Debate #29 classification:
        - L_p/L0 > 1.0: **Plasma-significant** — physics fundamentally
          alters the waveform.  Bare RLC gives large timing error.
        - L_p/L0 < 0.5: **Circuit-dominated** — bare damped RLC gives
          reasonable timing.  Validation is vacuously true.

    Parameters
    ----------
    L0 : float
        External (circuit) inductance [H].
    anode_radius : float
        Anode radius [m].
    cathode_radius : float
        Cathode radius [m].
    anode_length : float
        Anode length [m].

    Returns
    -------
    dict
        ``L_p_axial`` : float
            Axial plasma inductance at end of anode [H].
        ``L_p_over_L0`` : float
            Ratio L_p / L0 (dimensionless).
        ``regime`` : str
            "plasma-significant" if ratio > 1.0,
            "transitional" if 0.5 <= ratio <= 1.0,
            "circuit-dominated" if ratio < 0.5.
        ``L_per_length`` : float
            Inductance per unit length [H/m].

    References
    ----------
    PhD Debate #29 (2026-02-28): L_p/L0 diagnostic for validation
    informativeness.
    """
    mu_0 = 4.0 * np.pi * 1e-7  # [H/m]
    L_per_length = (mu_0 / (2.0 * np.pi)) * np.log(cathode_radius / anode_radius)
    L_p_axial = L_per_length * anode_length
    ratio = L_p_axial / max(L0, 1e-15)

    if ratio > 1.0:
        regime = "plasma-significant"
    elif ratio >= 0.5:
        regime = "transitional"
    else:
        regime = "circuit-dominated"

    return {
        "L_p_axial": L_p_axial,
        "L_p_over_L0": ratio,
        "regime": regime,
        "L_per_length": L_per_length,
    }


def compute_bare_rlc_timing(
    C: float,
    L0: float,
    R0: float,
) -> float:
    """Compute the quarter-period of a bare damped RLC circuit.

    For a lossless RLC, the quarter-period is T/4 = pi * sqrt(L0 * C).
    With damping, the underdamped period is
    T = 2*pi / sqrt(1/(L0*C) - (R0/(2*L0))^2) and T/4 is one quarter.

    Parameters
    ----------
    C : float
        Capacitance [F].
    L0 : float
        External inductance [H].
    R0 : float
        External resistance [Ohm].

    Returns
    -------
    float
        Quarter-period [s].
    """
    omega_0_sq = 1.0 / (L0 * C)
    gamma_sq = (R0 / (2.0 * L0)) ** 2
    if omega_0_sq <= gamma_sq:
        # Overdamped — no oscillation, return RC timescale
        return np.pi * np.sqrt(L0 * C)
    omega_d = np.sqrt(omega_0_sq - gamma_sq)
    return np.pi / (2.0 * omega_d)


# =====================================================================
# Speed factor diagnostic (Debate #36)
# =====================================================================

# Optimal speed factor from Lee & Saw (2008, 2014):
# S_opt ~ 89 kA/(cm * sqrt(Torr)) for deuterium Mather-type DPF.
_S_OPTIMAL_KA_CM_TORR = 89.0


def compute_speed_factor(
    peak_current: float,
    anode_radius: float,
    fill_pressure_torr: float,
) -> dict[str, float]:
    """Compute the Lee speed factor S = I_peak / (a * sqrt(p)).

    The speed factor is a dimensionless scaling parameter that
    characterizes the drive condition of a DPF device.  Lee & Saw
    (2008) showed that neutron yield peaks at an optimal speed
    factor S_opt ~ 89 kA/(cm * sqrt(Torr)) for deuterium fill.

    Classification (PhD Debate #36):

    - S/S_opt ~ 0.8-1.2: **Optimal** — thin-sheath snowplow valid,
      Lee model fc/fm are most transferable.
    - S/S_opt < 0.8: **Sub-driven** — slow sheath, thick and diffuse,
      under-compressed pinch.
    - S/S_opt > 1.2: **Super-driven** — sheath outruns fill gas,
      snowplow approximation breaks down, fc/fm become strongly
      device-dependent.

    Parameters
    ----------
    peak_current : float
        Peak discharge current [A].
    anode_radius : float
        Anode radius [m].
    fill_pressure_torr : float
        Fill gas pressure [Torr].

    Returns
    -------
    dict
        ``S`` : float
            Speed factor [kA / (cm * sqrt(Torr))].
        ``S_over_S_opt`` : float
            Ratio S / S_opt (dimensionless).
        ``regime`` : str
            "optimal", "sub-driven", or "super-driven".

    References
    ----------
    S. Lee & S. H. Saw, J. Fusion Energy 27:292-295 (2008).
    S. Lee, J. Fusion Energy 33:319-335 (2014).
    """
    # Convert to kA/(cm * sqrt(Torr))
    I_kA = peak_current / 1e3
    a_cm = anode_radius * 100.0
    p_torr = max(fill_pressure_torr, 1e-10)

    S = I_kA / (a_cm * np.sqrt(p_torr))
    S_ratio = S / _S_OPTIMAL_KA_CM_TORR

    if 0.8 <= S_ratio <= 1.2:
        regime = "optimal"
    elif S_ratio < 0.8:
        regime = "sub-driven"
    else:
        regime = "super-driven"

    return {
        "S": S,
        "S_over_S_opt": S_ratio,
        "regime": regime,
    }


# =====================================================================
# Helpers
# =====================================================================

def _find_first_peak(signal: np.ndarray, min_prominence: float = 0.05) -> int:
    """Find the index of the first local maximum (first peak) in *signal*.

    The algorithm identifies the first point where the signal transitions
    from rising to falling, provided the peak is at least *min_prominence*
    times the global maximum.  This avoids picking up early noise spikes.

    Falls back to ``np.argmax(signal)`` if no qualifying local peak is
    found (e.g. a monotonically rising signal).

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Non-negative signal (typically ``np.abs(I)``).
    min_prominence : float
        Minimum fraction of the global max that a local peak must reach
        to qualify.  Default 0.05 (5 %).

    Returns
    -------
    int
        Index of the first qualifying local peak.
    """
    if len(signal) < 3:
        return int(np.argmax(signal))

    global_max = float(np.max(signal))
    threshold = min_prominence * global_max
    n_decline = 3  # Points ahead to verify sustained decline

    # Walk through signal and find first point where signal shows a
    # sustained decline, confirming a true local maximum rather than a
    # phase-transition plateau (common in DPF current waveforms).
    # A candidate peak at index i is confirmed if:
    # 1. Signal has been rising above threshold
    # 2. signal[i] >= signal[i-1] (local max candidate)
    # 3. All n_decline points after i are strictly below signal[i]
    #    AND the signal does not recover above signal[i] within that window
    rising = False
    for i in range(1, len(signal) - n_decline):
        if signal[i] >= threshold:
            rising = True
        if rising and signal[i] >= threshold and signal[i] >= signal[i - 1]:
            # Confirm: all following n_decline points are below peak value
            is_peak = all(
                signal[i + k + 1] < signal[i]
                for k in range(n_decline)
            )
            if is_peak:
                return i

    # Fallback: global maximum
    return int(np.argmax(signal))


# =====================================================================
# Waveform comparison
# =====================================================================

def nrmse_peak(
    t_sim: np.ndarray,
    I_sim: np.ndarray,
    t_exp: np.ndarray,
    I_exp: np.ndarray,
    truncate_at_dip: bool = False,
    max_time: float | None = None,
) -> float:
    """Compute peak-normalized RMSE between simulated and experimental waveforms.

    Resamples the simulated waveform onto the experimental time grid via
    linear interpolation, then computes NRMSE = RMSE / |I_peak_exp|.

    Parameters
    ----------
    t_sim : ndarray
        Simulated time array [s].
    I_sim : ndarray
        Simulated current waveform [A].
    t_exp : ndarray
        Experimental time array [s].
    I_exp : ndarray
        Experimental current waveform [A].
    truncate_at_dip : bool, optional
        If True, truncate comparison at the current dip (first local
        minimum of |I| after peak, searched within a limited window of
        2× the peak time).  This excludes the post-pinch region where
        frozen-L_plasma makes the model invalid.  Default False.
    max_time : float or None, optional
        If given, truncate comparison at this time [s].  Only
        experimental points with t <= max_time are included.
        Useful for windowed validation (e.g. rise-phase only).

    Returns
    -------
    float
        Peak-normalized RMSE (dimensionless).  0.0 for a perfect match.
    """
    t_e = np.asarray(t_exp, dtype=np.float64)
    I_e = np.asarray(I_exp, dtype=np.float64)

    # Explicit time window truncation
    if max_time is not None:
        mask = t_e <= max_time
        if np.sum(mask) > 2:
            t_e = t_e[mask]
            I_e = I_e[mask]

    if truncate_at_dip:
        # Find the current dip in the SIMULATED waveform — the model is
        # invalid after the dip (frozen L_plasma region).  Search only
        # within a limited window (2× peak time) to avoid picking up
        # the L-R crowbar decay tail at late times.
        abs_I_sim = np.abs(np.asarray(I_sim, dtype=np.float64))
        t_sim_arr = np.asarray(t_sim, dtype=np.float64)
        sim_peak_idx = int(np.argmax(abs_I_sim))
        t_peak = t_sim_arr[sim_peak_idx]

        # Search window: peak to 2× peak time (captures the dip but not
        # the crowbar L-R decay which can extend to 10× peak time)
        t_search_end = 2.0 * t_peak
        search_end_idx = int(np.searchsorted(t_sim_arr, t_search_end))
        search_end_idx = min(search_end_idx, len(abs_I_sim))

        post_peak_sim = abs_I_sim[sim_peak_idx:search_end_idx]
        if len(post_peak_sim) > 2:
            dip_offset = int(np.argmin(post_peak_sim))
            if dip_offset > 1:
                t_dip = t_sim_arr[sim_peak_idx + dip_offset]
                mask = t_e <= t_dip
                if np.sum(mask) > 2:
                    t_e = t_e[mask]
                    I_e = I_e[mask]

    I_sim_resampled = np.interp(t_e, t_sim, I_sim)
    residuals = I_sim_resampled - I_e
    rmse = float(np.sqrt(np.mean(residuals**2)))
    I_peak_exp = float(np.max(np.abs(I_e)))
    return rmse / max(I_peak_exp, 1e-300)


# Backward-compatible alias (CRIT-1: prefer nrmse_peak for clarity)
normalized_rmse = nrmse_peak


# =====================================================================
# Validation functions
# =====================================================================

def validate_current_waveform(
    t_sim: np.ndarray,
    I_sim: np.ndarray,
    device_name: str,
    truncate_at_dip: bool = False,
) -> dict[str, Any]:
    """Validate a simulated current waveform against experimental data.

    Compares the peak current magnitude and its timing against published
    measurements for the specified device.

    Parameters
    ----------
    t_sim : ndarray, shape (M,)
        Simulated time array [s].
    I_sim : ndarray, shape (M,)
        Simulated current waveform [A].
    device_name : str
        Key into ``DEVICES`` (e.g. ``"PF-1000"``, ``"NX2"``).

    Returns
    -------
    dict
        ``peak_current_error`` : float
            Relative error |I_peak_sim - I_peak_exp| / I_peak_exp.
        ``peak_current_sim`` : float
            Peak of simulated current [A].
        ``peak_current_exp`` : float
            Experimental peak current [A].
        ``timing_ok`` : bool
            True if simulated peak time is within 10 % of experimental
            rise time.

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    t_arr = np.asarray(t_sim, dtype=np.float64)
    I_arr = np.asarray(I_sim, dtype=np.float64)

    # Peak current: find the FIRST local maximum of |I(t)|.
    # For DPF waveforms, the first peak (before the current dip) is the
    # physically meaningful one.  Post-pinch oscillation peaks can exceed
    # the first peak and must not be mistaken for the primary peak.
    abs_I = np.abs(I_arr)
    peak_idx = _find_first_peak(abs_I)
    peak_current_sim = float(abs_I[peak_idx])
    peak_time_sim = float(t_arr[peak_idx])

    peak_current_exp = device.peak_current
    rise_time_exp = device.current_rise_time

    # Relative error on peak current
    peak_current_error = abs(peak_current_sim - peak_current_exp) / max(
        abs(peak_current_exp), 1e-300
    )

    # Timing check: peak time within 10% of experimental rise time
    timing_error = abs(peak_time_sim - rise_time_exp) / max(rise_time_exp, 1e-300)
    timing_ok = timing_error < 0.10

    # Uncertainty budget following GUM (JCGM 100:2008) and ASME V&V 20-2009.
    # Components: Rogowski coil (Type B), digitization (Type B), simulation error.
    # u_combined = sqrt(u_rogowski^2 + u_digitization^2 + u_sim^2)
    u_exp_peak = device.peak_current_uncertainty
    u_exp_timing = device.rise_time_uncertainty
    u_digitization = device.waveform_digitization_uncertainty
    # Total experimental uncertainty (Rogowski + digitization in quadrature)
    u_exp_total = np.sqrt(u_exp_peak**2 + u_digitization**2)
    # Combined uncertainty (experimental + simulation error)
    u_combined_peak = np.sqrt(u_exp_total**2 + peak_current_error**2)
    u_combined_timing = np.sqrt(u_exp_timing**2 + timing_error**2)
    # Agreement check: simulation within 2-sigma of total experimental uncertainty
    agreement_within_2sigma = peak_current_error <= 2.0 * max(u_exp_total, 0.01)

    # Waveform NRMSE: compare full I(t) trace if digitized waveform available
    waveform_available = (
        device.waveform_t is not None and device.waveform_I is not None
    )
    waveform_nrmse = float("nan")
    if waveform_available:
        waveform_nrmse = normalized_rmse(
            t_arr, I_arr, device.waveform_t, device.waveform_I,
            truncate_at_dip=truncate_at_dip,
        )

    return {
        "peak_current_error": peak_current_error,
        "peak_current_sim": peak_current_sim,
        "peak_current_exp": peak_current_exp,
        "peak_time_sim": peak_time_sim,
        "timing_ok": timing_ok,
        "timing_error": timing_error,
        "waveform_available": waveform_available,
        "waveform_nrmse": waveform_nrmse,
        "uncertainty": {
            "peak_current_exp_1sigma": u_exp_peak,
            "digitization_1sigma": u_digitization,
            "peak_current_total_exp_1sigma": float(u_exp_total),
            "rise_time_exp_1sigma": u_exp_timing,
            "peak_current_combined_1sigma": float(u_combined_peak),
            "timing_combined_1sigma": float(u_combined_timing),
            "agreement_within_2sigma": bool(agreement_within_2sigma),
        },
        "measurement_notes": device.measurement_notes,
    }


def validate_neutron_yield(
    Y_sim: float,
    device_name: str,
) -> dict[str, Any]:
    """Validate simulated neutron yield against experimental data.

    Parameters
    ----------
    Y_sim : float
        Simulated total neutron yield.
    device_name : str
        Key into ``DEVICES``.

    Returns
    -------
    dict
        ``yield_ratio`` : float
            Y_sim / Y_exp.
        ``within_order_magnitude`` : bool
            True if 0.1 < ratio < 10.
        ``yield_sim`` : float
        ``yield_exp`` : float

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    yield_exp = device.neutron_yield
    yield_ratio = Y_sim / max(yield_exp, 1e-300)

    u_exp_yield = device.neutron_yield_uncertainty
    return {
        "yield_ratio": yield_ratio,
        "within_order_magnitude": 0.1 < yield_ratio < 10.0,
        "yield_sim": float(Y_sim),
        "yield_exp": yield_exp,
        "uncertainty": {
            "neutron_yield_exp_1sigma": u_exp_yield,
        },
    }


def device_to_config_dict(device_name: str) -> dict[str, Any]:
    """Convert device parameters to a ``SimulationConfig``-compatible dict.

    Produces a configuration dictionary that can be passed directly to
    ``SimulationConfig(**device_to_config_dict("PF-1000"))`` or written
    to a JSON config file.

    The grid shape is chosen to give approximately 1 mm axial resolution
    (capped at 256 cells per dimension for tractability).

    The initial fill gas density is computed from the ideal gas law at
    room temperature (300 K)::

        n = p / (k_B * T)
        rho = n * m_D2  # D2 molecular mass at room temperature

    where *p* is the fill pressure converted from Torr to Pa.

    Parameters
    ----------
    device_name : str
        Key into ``DEVICES``.

    Returns
    -------
    dict
        Configuration dictionary with keys matching ``SimulationConfig``
        fields (``grid_shape``, ``dx``, ``sim_time``, ``circuit``, etc.).

    Raises
    ------
    KeyError
        If ``device_name`` is not in ``DEVICES``.
    """
    device = DEVICES[device_name]

    # --- Grid resolution ------------------------------------------------
    # Target ~1 mm resolution; cap at 256 cells per dimension
    target_dx = 1e-3  # 1 mm

    # Domain size: slightly larger than cathode diameter x anode length
    domain_r = device.cathode_radius * 1.5
    domain_z = device.anode_length * 1.5

    nx = min(int(np.ceil(2.0 * domain_r / target_dx)), 256)
    ny = min(int(np.ceil(2.0 * domain_r / target_dx)), 256)
    nz = min(int(np.ceil(domain_z / target_dx)), 256)

    # Ensure at least 8 cells per dimension
    nx = max(nx, 8)
    ny = max(ny, 8)
    nz = max(nz, 8)

    # Actual grid spacing from chosen cell count
    dx = 2.0 * domain_r / nx

    # --- Fill gas density from ideal gas law ----------------------------
    # Convert Torr to Pa: 1 Torr = 133.322 Pa
    pressure_Pa = device.fill_pressure_torr * 133.322
    T_room = 300.0  # K

    n_fill = pressure_Pa / (k_B * T_room)  # number density [m^-3]
    rho_fill = n_fill * m_D2                # mass density [kg/m^3] — D2 molecular

    # --- Simulation time ------------------------------------------------
    # A few quarter-periods is usually enough to capture peak current + pinch
    sim_time = 4.0 * device.current_rise_time

    return {
        "grid_shape": [nx, ny, nz],
        "dx": dx,
        "sim_time": sim_time,
        "rho0": rho_fill,
        "T0": T_room,
        "ion_mass": m_d,
        "circuit": {
            "C": device.capacitance,
            "V0": device.voltage,
            "L0": device.inductance,
            "R0": device.resistance,
            "anode_radius": device.anode_radius,
            "cathode_radius": device.cathode_radius,
        },
    }
