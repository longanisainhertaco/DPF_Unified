"""Experimental device instances and registry for DPF validation.

Each ExperimentalDevice object contains published parameters for a real
Dense Plasma Focus device, including waveform data where available.
"""

from __future__ import annotations

from dpf.validation.experimental_device import ExperimentalDevice
from dpf.validation.experimental_waveforms import (
    _FAETON_WAVEFORM_I_KA,
    _FAETON_WAVEFORM_T_US,
    _MJOLNIR_WAVEFORM_I_KA,
    _MJOLNIR_WAVEFORM_T_US,
    _PF1000_16KV_WAVEFORM_I_MA,
    _PF1000_16KV_WAVEFORM_T_US,
    _PF1000_WAVEFORM_I_MA,
    _PF1000_WAVEFORM_T_US,
    _POSEIDON60KV_WAVEFORM_I_KA,
    _POSEIDON60KV_WAVEFORM_T_US,
    _UNU_ICTP_WAVEFORM_I_KA,
    _UNU_ICTP_WAVEFORM_T_US,
    PF1000_GRIBKOV_I_TRIMMED,
    PF1000_GRIBKOV_T_TRIMMED,
)

# =====================================================================
# Device instances
# =====================================================================

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
    reference="Lee & Saw, J. Fusion Energy 27:292, 2008; RADPF Module 1 [KR: the-code-uses-a-phenomenological-mechanism-for-beam-target-production-of-fusion-.md Table 1]",
    lee_fc=0.7, lee_fm=0.10, lee_fmr=0.12, lee_fcr=0.7,
    lee_reference="Lee & Saw, J. Fusion Energy 27:292 (2008)",
    peak_current_uncertainty=0.08,     # 8% (compact device, lower SNR)
    rise_time_uncertainty=0.12,        # 12%
    neutron_yield_uncertainty=0.60,    # 60% (shot-to-shot)
    waveform_provenance="",  # No waveform data available
    measurement_notes=(
        "No digitized waveform available. Peak current and rise time from "
        "Lee & Saw, J. Fusion Energy 27:292, 2008. R0=2.3 mOhm from RADPF "
        "Module 1 preset (plasmafocus.net); actual RESF=R0/sqrt(L0/C)=0.086 "
        "(not 0.1 as sometimes stated). "
        "Fill pressure 3 Torr D2 for neutron operation. "
        "L0 uncertainty: literature reports 15-20 nH (Sahyouni et al. 2021 "
        "DOI:10.1155/2021/6611925 vs RADPF preset). "
        "Uncertainties are Type B estimates (not stated in source)."
    ),
    reliability="reference_only",
    reliability_note=(
        "400 kA peak current is a RADPF model output, not a Rogowski coil "
        "measurement. Unloaded RLC peak is 402.5 kA, implying 0.6% plasma "
        "loading — physically implausible for any DPF discharge. No digitized "
        "waveform available for NRMSE validation. fc^2/fm = 4.90, degenerate "
        "with PF-1000 (4.69) — provides no independent parameter constraint. "
        "Excluded from validation pass/fail claims."
    ),
)

# VERIFIED against KR: Lee & Saw 2014 (a-course-on-plasma-focus-numerical-experiments-
# s-lee-and-s-h-saw-part-1-basic-course.md).
# Table p.152 (KR line 12725): V0=15 kV, P0=4 Torr, L0=110 nH, C0=30 uF,
# a=0.95 cm, b=3.2 cm, z0=16 cm, Ipeak=182 kA, Ipinch=123 kA, S=96, Yn=1.2e7.
# Geometry also confirmed at KR lines 16146-16151 (same paper).
# V0 discrepancy: prior code used 13.5 kV (IPFS waveform file conditions);
# KR canonical table (p.152) states 15 kV. Per papers-are-truth, 15 kV is adopted.
# peak_current updated to 182 kA per KR p.152 Ipeak column.
# I_pinch updated to 123 kA per KR p.152 Ipinch column.
# Waveform retained from IPFS digitized file (V0=13.5 kV operating condition) as
# the only available measured trace; noted in measurement_notes.
UNU_ICTP_DATA = ExperimentalDevice(
    name="UNU-ICTP",
    institution="UNU-ICTP PFF",
    capacitance=30e-6,             # 30 uF [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12725]
    voltage=15e3,                  # 15 kV [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12725] (prior code: 13.5 kV from IPFS waveform file)
    inductance=110e-9,             # 110 nH [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12725]
    resistance=12e-3,              # 12 mOhm # UNVERIFIED: not stated in KR p.152 table; inherited from prior code
    anode_radius=0.0095,           # 9.5 mm [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md lines 16149: a=0.95e-2 m]
    cathode_radius=0.032,          # 32 mm [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md line 16150: b=3.2e-2 m]
    anode_length=0.16,             # 160 mm [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md line 16151: zo=0.16 m]
    fill_pressure_torr=4.0,        # 4 Torr [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12725] (prior code: 3.0 Torr from IPFS)
    fill_gas="deuterium",
    peak_current=182e3,            # 182 kA [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12725: Ipeak=0.182 MA]
    neutron_yield=1.2e7,           # 1.2e7 [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12725: Yn=1.2x10^7]
    current_rise_time=2.2e-6,      # ~2.2 us to peak (from waveform) # UNVERIFIED: not stated in KR table
    reference=(
        "[KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.10/152] "
        "Lee & Saw 2014; geometry also at KR lines 16146-16151"
    ),
    lee_fc=0.7, lee_fm=0.08, lee_fmr=0.16, lee_fcr=0.7,
    lee_reference="IPFS plasmafocus.net preset; Lee & Saw (2014)",  # UNVERIFIED: Lee-fit params not in KR p.152 table; inherited from IPFS
    peak_current_uncertainty=0.10,     # 10% (training device, less precise)
    rise_time_uncertainty=0.15,        # 15%
    neutron_yield_uncertainty=0.70,    # 70% (shot-to-shot)
    waveform_t=_UNU_ICTP_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_UNU_ICTP_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_amplitude_uncertainty=0.016,  # GUM: 9.3 kA / (2*sqrt(3)*169 kA) = 1.6% (rectangular)
    waveform_time_uncertainty=0.002,       # 0.2% (~1 ns digitization on ~5 us trace)
    waveform_uncertainty_type="digitization",  # Type B: quantization from digital oscilloscope
    waveform_provenance="measured",
    kr_status="verified",
    measurement_notes=(
        "45 points from IPFS 'UNU ICTPPFF D2 05.15.xls' (plasmafocus.net). "
        "Original: 5556 points at ~1 ns resolution, digitized oscilloscope trace. "
        "Quantization: 9.3 kA steps (5.5% of 169 kA peak at 13.5 kV). "
        "GUM (JCGM 100:2008) rectangular distribution: u = step/(2*sqrt(3)) = 1.6%. "
        "EMI spike at pinch time (2.72-2.73 us) removed by median filtering. "
        "Smoothed with 15-sample uniform filter + 51-sample median filter. "
        "NOTE: Waveform digitized from IPFS file at V0=13.5 kV (IPFS operating condition). "
        "Canonical device parameters per KR p.152: V0=15 kV, P0=4 Torr, Ipeak=182 kA, "
        "Ipinch=123 kA, S=96, Yn=1.2e7. "
        "Lee model params from IPFS: fm=0.08, fc=0.7, fmr=0.16, fcr=0.7. "
        "Uncertainties are Type B estimates. Rogowski coil uncertainty ~10%. "
        "Combined waveform uncertainty: u_I = sqrt(0.10^2 + 0.016^2) = 10.0%."
    ),
)


PF1000_16KV_DATA = ExperimentalDevice(
    name="PF-1000-16kV",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # Same bank
    voltage=16e3,                  # 16 kV (reduced from 27 kV)
    inductance=25e-9,              # 25 nH (Akel et al. 2021 Table 1 - these are literally Akel shots 12590-12606 at 1.05 Torr)
    resistance=2.3e-3,             # Same circuit
    anode_radius=0.115,            # Same geometry
    cathode_radius=0.16,           # Same geometry
    anode_length=0.48,             # 480 mm (Akel et al. 2021 p.1: "480 mm long coaxial electrodes"; Table 1: z0 = 48 cm)
    fill_pressure_torr=1.05,       # 1.05 Torr D2 (Akel 2021)
    fill_gas="deuterium",
    peak_current=1.165e6,          # 1.165 MA (Akel et al. 2021 Table 1, shot 12581: Ipeak = 1165 kA)
    neutron_yield=2.33e9,          # 2.33e9 n/shot at 1.05 Torr (average of 16 shots)
    current_rise_time=6.0e-6,      # ~6 us (estimated from Lee model fit in paper)
    reference="Akel et al., Radiat. Phys. Chem. 188:109633, 2021",
    crowbar_resistance=1.5e-3,     # Same crowbar as 27 kV (PhD Debate #30)
    peak_current_uncertainty=0.10,     # 10% (range 1.1-1.3 MA = ±8.3%)
    rise_time_uncertainty=0.15,        # 15% (no explicit timing stated)
    neutron_yield_uncertainty=0.40,    # 40% (shot-to-shot, Akel Table 1)
    waveform_t=_PF1000_16KV_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_PF1000_16KV_WAVEFORM_I_MA * 1e6,        # Convert MA -> A
    waveform_amplitude_uncertainty=0.05,  # 5% reconstruction model uncertainty
    waveform_time_uncertainty=0.01,       # 1% temporal (pinch timing estimated)
    waveform_uncertainty_type="reconstruction",  # Physics-scaled from 27kV Scholz waveform
    peak_current_from_shot_spread=True,  # 10% derives from 1.1-1.3 MA shot range
    waveform_provenance="reconstructed",
    lee_fc=0.70, lee_fm=0.20, lee_fmr=0.12, lee_fcr=0.48,
    # [KR: radiation-physics-and-chemistry-188-2021-109633.md §p.5 — "average
    # values as follow: 0.2, 0.7, 0.12 and 0.48" (order: fm, fc, fmr, fcr)]
    # Prior 0.47 was a 1-digit transcription error vs the verbatim KR text.
    lee_reference="Akel et al., Radiat. Phys. Chem. 188:109633, 2021 (24-shot avg at 16 kV) [KR: radiation-physics-and-chemistry-188-2021-109633.md §p.5]",
    measurement_notes=(
        "PF-1000 operated at 16 kV (170.5 kJ) with 1.05 Torr D2 fill. "
        "Peak current 1.131-1.328 MA across the 16 shots at 1.05 Torr (Akel et al. 2021 Table 1). "
        "I_peak reference = 1.165 MA from shot 12581 (Table 1: Ipeak = 1165 kA). "
        "WAVEFORM NOTE: Reconstructed from physics scaling of 27 kV Scholz (2006) "
        "waveform, rescaled by 1.165/1.20 = 0.9708 to match Akel shot 12581 peak. Same bank (C0, L0, R0), "
        "so T/4=10.49 us is identical. Current dip shifted earlier (~5.5 us vs ~7.0 us) "
        "due to lower fill pressure (1.05 Torr vs 3.5 Torr → faster sheath). "
        "Waveform_digitization_uncertainty set to 5% (higher than 3% for 27 kV) to "
        "account for reconstruction uncertainty. Replace with actual digitized data "
        "from Akel (2021) Fig. 3 when paper access is obtained. "
        "DOI: 10.1016/j.radphyschem.2021.109633"
    ),
)


PF1000_GRIBKOV_DATA = ExperimentalDevice(
    name="PF-1000-Gribkov",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,
    voltage=27e3,
    inductance=33.5e-9,
    # 6.1 mOhm: same physical PF-1000 capacitor bank as standard PF-1000 device.
    # [KR: plasma-physics-and-technology-1211-9-2025.md §Table 1 lines 256-261]
    # Malek 2025 lists Bank parameters: L0=33.5 nH, C0=1332 uF, r0=6.1 mOhm.
    # Previous value (2.3 mOhm) was inconsistent — Gribkov shot uses the same bank.
    resistance=6.1e-3,
    anode_radius=0.115,
    cathode_radius=0.16,
    anode_length=0.60,
    fill_pressure_torr=3.5,
    fill_gas="deuterium",
    peak_current=1.846e6,           # 1.846 MA (Gribkov 2007, different shot from Scholz)
    neutron_yield=1e11,
    current_rise_time=6.39e-6,      # 6.39 us (peak timing from data)
    reference="Gribkov et al., J. Phys. D: Appl. Phys. 40:1977-1989 (Part I), 2007, doi:10.1088/0022-3727/40/7/021",
    crowbar_resistance=1.5e-3,
    peak_current_uncertainty=0.05,
    rise_time_uncertainty=0.10,
    neutron_yield_uncertainty=0.50,
    waveform_t=PF1000_GRIBKOV_T_TRIMMED * 1e-6,    # us -> s
    waveform_I=PF1000_GRIBKOV_I_TRIMMED * 1e3,      # kA -> A
    waveform_amplitude_uncertainty=0.02,  # 2% (digital oscilloscope, not hand-digitized)
    waveform_time_uncertainty=0.003,      # 0.3% (digital acquisition)
    waveform_uncertainty_type="digitization",  # Type B: IPFS digital archive
    waveform_provenance="measured",
    lee_fc=0.70, lee_fm=0.08, lee_fmr=0.16, lee_fcr=0.70,
    lee_reference="Lee & Saw 2014, IPFS PF1000data.xls (same device/voltage as PF-1000 standard)",
    measurement_notes=(
        "94-point digitized waveform from plasmafocus.net/IPFS PF1000 05.15.xls Sheet2 "
        "(NOT digitized from the paper itself — the Gribkov 2007 Part I paper does not "
        "publish a tabulated I(t) curve; the xls file is the authoritative digital archive "
        "from the Lee RADPF model package). "
        "Source paper: Gribkov et al., J. Phys. D: Appl. Phys. 40:1977-1989 (Part I), 2007, "
        "doi:10.1088/0022-3727/40/7/021. "
        "(The old citation \"40:3592\" referred to Scholz 2007 Part II, not Gribkov Part I.) "
        "Same device and conditions as Scholz (2006) PF-1000 at 27 kV, 3.5 Torr D2, "
        "but DIFFERENT shot and DIFFERENT digitization. Peak 1.846 MA at 6.39 us "
        "(vs Scholz: 1.87 MA at 5.8 us - shot-to-shot variability). "
        "Lower digitization uncertainty (2%) because this is from digital oscilloscope data "
        "archived in the Lee model RADPF package, not hand-digitized from a paper figure."
    ),
)


# REFERENCE_ONLY — KR contains POSEIDON at 60 kV / 156 uF only (KR line 12736).
# The 40 kV / 450 uF variant requires Herold 1989 Nucl. Fusion 29 which is NOT on disk.
# L0=20 nH and R0=2 mOhm are RADPF default estimates, not measured values.
# Moved to _REFERENCE_ONLY registry. Do NOT promote to DEVICES until Herold 1989 is ingested.
POSEIDON_DATA = ExperimentalDevice(
    name="POSEIDON",
    institution="IPF Stuttgart",
    capacitance=450e-6,            # 450 uF (H. Herold, private comm.; Lee RADPF) # UNVERIFIED: Herold 1989 not on disk
    voltage=40e3,                  # 40 kV typical operation (360 kJ stored) # UNVERIFIED
    inductance=20e-9,              # 20 nH (RADPF default estimate) # UNVERIFIED
    resistance=2e-3,               # ~2 mOhm (estimated from RESF ~0.05) # UNVERIFIED
    anode_radius=0.104,            # 104 mm (208 mm diameter; Herold 1989) # UNVERIFIED: paper not on disk
    cathode_radius=0.135,          # 135 mm (270 mm diameter; Herold 1989) # UNVERIFIED
    anode_length=0.47,             # 470 mm (Herold 1989) # UNVERIFIED
    fill_pressure_torr=3.5,        # 3.5 Torr D2 (typical neutron operation) # UNVERIFIED
    fill_gas="deuterium",
    peak_current=2.6e6,            # 2.6 MA (Herold et al. 1989, at 40 kV) # UNVERIFIED
    neutron_yield=1e11,            # ~10^11 (Herold 1989) # UNVERIFIED
    current_rise_time=5.0e-6,      # ~5 us (estimated from Lee model quarter-period) # UNVERIFIED
    reference="Herold et al., Nucl. Fusion 29:33, 1989 (NOT on disk — requires ingestion)",
    lee_fc=0.60, lee_fm=0.275, lee_fmr=0.45, lee_fcr=0.44,
    lee_reference="Lee & Saw, J. Fusion Energy 33:319 (2014), at 60 kV — WRONG VARIANT for 40 kV",
    peak_current_uncertainty=0.08,     # 8% (large device, Rogowski + integration)
    rise_time_uncertainty=0.15,        # 15% (not explicitly stated in source)
    neutron_yield_uncertainty=0.50,    # 50% (shot-to-shot)
    waveform_provenance="",  # No waveform data available
    kr_status="reference_only",
    measurement_notes=(
        "POSEIDON 40 kV variant (IPF Stuttgart): DEMOTED to _REFERENCE_ONLY. "
        "KR has POSEIDON at 60 kV / 156 uF only (KR line 12736). "
        "This 40 kV / 450 uF variant requires Herold 1989 Nucl. Fusion 29 "
        "(DOI: 10.1088/0029-5515/29/1/005) — NOT present in KnowledgeReference/. "
        "L0=20 nH and R0=2 mOhm are RADPF default estimates, not measured values. "
        "Do not use for validation claims until Herold 1989 is ingested."
    ),
)


# VERIFIED against KR: Lee & Saw 2014 Table p.152 (KR line 12736).
# KR line 12736: Poseidon, V0=60 kV, P0=3.8 Torr, L0=18 nH, C0=156 uF,
# b=9.50 cm, a=6.55 cm, z0=30 cm, Ipeak=3.200 MA, Ipinch=1.260 MA, S=251,
# Yn=3.3e11, kmin=0.20, Ipinch/Ipeak=0.39.
# Note: KR table gives L0=18 nH; IPFS fit used 17.7 nH — within table rounding.
# 17.7 nH retained as more precise IPFS fit value; could be updated to 18 nH if
# KR is treated as the authoritative rounded value.
POSEIDON_60KV_DATA = ExperimentalDevice(
    name="POSEIDON-60kV",
    institution="IPF Stuttgart",
    capacitance=156e-6,            # 156 uF [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736]
    voltage=60e3,                  # 60 kV [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736]
    inductance=17.7e-9,            # 17.7 nH (IPFS Lee model fit; KR table rounds to 18 nH [line 12736])
    resistance=1.7e-3,             # 1.7 mOhm (IPFS Lee model fit) # UNVERIFIED: not in KR p.152 table
    anode_radius=0.0655,           # 65.5 mm [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736: a=6.55 cm]
    cathode_radius=0.095,          # 95 mm [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736: b=9.50 cm]
    anode_length=0.30,             # 300 mm [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736: z0=30 cm]
    fill_pressure_torr=3.8,        # 3.8 Torr D2 [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736]
    fill_gas="deuterium",
    peak_current=3.19e6,           # 3.19 MA (IPFS digitized peak; KR table: 3.200 MA [line 12736])
    neutron_yield=3.3e11,          # 3.3e11 [KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736: Yn=3.3x10^11]
    current_rise_time=1.98e-6,     # 1.98 us (time of peak from waveform) # UNVERIFIED: not in KR table
    reference=(
        "[KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152 line 12736] "
        "Lee & Saw 2014 Table p.152; IPFS digitized waveform"
    ),
    crowbar_resistance=1.5e-3,     # estimated spark gap # UNVERIFIED
    peak_current_uncertainty=0.05,     # 5% (Rogowski coil)
    rise_time_uncertainty=0.05,        # 5% (well-digitized waveform)
    neutron_yield_uncertainty=0.50,    # 50% (shot-to-shot)
    waveform_t=_POSEIDON60KV_WAVEFORM_T_US * 1e-6,    # Convert us -> s
    waveform_I=_POSEIDON60KV_WAVEFORM_I_KA * 1e3,      # Convert kA -> A
    waveform_amplitude_uncertainty=0.02,  # 2% (IPFS digitization, high quality)
    waveform_time_uncertainty=0.005,      # 0.5% temporal
    waveform_uncertainty_type="digitization",  # Type B: IPFS digital archive
    waveform_provenance="measured",
    kr_status="verified",
    lee_fc=0.60, lee_fm=0.275, lee_fmr=0.45, lee_fcr=0.44,
    lee_reference=(
        "Lee & Saw 2014 (fit at 60 kV); "
        "[KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md p.152]"
    ),
    measurement_notes=(
        "POSEIDON at 60 kV / 156 uF (E0=280.8 kJ) with 3.8 Torr D2 fill. "
        "KR Table p.152 (line 12736): C0=156 uF, V0=60 kV, L0=18 nH, P0=3.8 Torr, "
        "b=9.50 cm, a=6.55 cm, z0=30 cm, Ipeak=3.200 MA, Ipinch=1.260 MA, S=251, "
        "Yn=3.3e11, kmin=0.20, Ipinch/Ipeak=0.39. "
        "Digitized I(t) waveform from IPFS (plasmafocus.net) Excel file. "
        "35 subsampled points from 103-point original. Peak 3.19 MA at 1.98 us. "
        "Electrode geometry DIFFERENT from POSEIDON 40 kV (a=104 mm, b=135 mm). "
        "Lee model fitted: fm=0.275, fc=0.595, fmr=0.45, fcr=0.44, "
        "L0=17.7 nH, R0=1.7 mOhm, zo=300 mm. "
        "DOI (parent device paper): 10.1088/0029-5515/29/1/005"
    ),
)


PF1000_20KV_DATA = ExperimentalDevice(
    name="PF-1000-20kV",
    institution="IPPLM Warsaw",
    capacitance=1.332e-3,          # Same bank
    voltage=20e3,                  # 20 kV
    inductance=33.5e-9,            # Same circuit
    resistance=2.3e-3,             # Same circuit (Scholz 2006 baseline, no Akel offset)
    anode_radius=0.115,            # Same geometry
    cathode_radius=0.16,           # Same geometry
    anode_length=0.60,             # Same geometry
    fill_pressure_torr=2.0,        # 2.0 Torr D2 (lower V0 → lower optimal pressure)
    fill_gas="deuterium",
    peak_current=1.4e6,            # 1.4 MA (voltage-scaled from 27 kV Scholz data)
    neutron_yield=5e9,             # estimated
    current_rise_time=6.3e-6,      # ~6.3 us (estimated, slightly longer than 27 kV)
    reference="Akel et al., Radiat. Phys. Chem. 188:109633, 2021 (voltage trend)",
    lee_fc=0.7, lee_fm=0.08, lee_fmr=0.16, lee_fcr=0.7,
    lee_reference="Lee & Saw (2014): same device geometry, same fc/fm as 27 kV",
    crowbar_resistance=1.5e-3,
    peak_current_uncertainty=0.12,     # 12% (interpolated, higher uncertainty)
    rise_time_uncertainty=0.15,
    neutron_yield_uncertainty=0.50,
    waveform_provenance="",  # No waveform data
    measurement_notes=(
        "PF-1000 at 20 kV / 2.0 Torr D2 — interpolated from voltage scan trend. "
        "Peak current 1.4 MA estimated from Akel et al. (2021) multi-voltage data "
        "and linear voltage scaling from 27 kV reference (1.87 * 20/27 = 1.385 MA). "
        "Not a direct measurement — higher uncertainty than 27 kV reference. "
        "Lee model params (fc=0.7, fm=0.08) from 27 kV fits — same device geometry."
    ),
    reliability="estimated",
    reliability_note=(
        "Peak current is voltage-scaled from Scholz (2006) 27 kV measurement, not "
        "a direct measurement at 20 kV. Fill pressure of 2.0 Torr is interpolated. "
        "Lee model params adopted from 27 kV operating point — may need adjustment "
        "at lower voltage if sheath dynamics differ significantly."
    ),
)


FAETON_DATA = ExperimentalDevice(
    name="FAETON-I",
    institution="Fuse Energy Technologies",
    capacitance=25e-6,             # 25 uF (5 x 5 uF Marx)
    voltage=100e3,                 # 100 kV direct-charge
    inductance=220e-9,             # 220 nH static inductance (Damideh 2025)
    resistance=7.6e-3,             # 7.6 mOhm (estimated from I_peak damping)
    anode_radius=0.05,             # 50 mm (Damideh et al. 2025 Table 1: "Anode radius 5 cm")
    cathode_radius=0.106,          # 106 mm (Damideh et al. 2025 Table 1: "Cathode radius 10.6 cm"; §Apparatus: "encircle the anode with a radius of 10.6 cm")
    anode_length=0.17,             # 170 mm (Damideh et al. 2025 Table 1: "Effective anode length 17 cm")
    fill_pressure_torr=12.0,       # 12 Torr D2 (optimal for neutron yield)
    fill_gas="deuterium",
    peak_current=1.0e6,            # ~1 MA (Damideh 2025)
    neutron_yield=2.5e10,          # 2.5e10 D-D n/shot typical (8e10 peak)
    current_rise_time=3.7e-6,      # 3.7 us (Damideh et al. 2025 §III: "rise time of ~3.7 us"; transition time 3.745 us)
    reference=(
        "Damideh et al., Scientific Reports 15:23048, 2025; "
        "DOI: 10.1038/s41598-025-07939-x"
    ),
    lee_fc=0.70, lee_fm=0.70, lee_fmr=0.10, lee_fcr=0.14,
    lee_reference="Damideh et al., Sci. Rep. 15:23048 (2025); Lee co-author",
    crowbar_resistance=0.0,        # No crowbar switch
    peak_current_uncertainty=0.08, # 8% (Rogowski coil + Marx jitter)
    rise_time_uncertainty=0.10,    # 10% (not precisely stated)
    neutron_yield_uncertainty=0.50,  # 50% (shot-to-shot + re-strikes)
    waveform_t=_FAETON_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_FAETON_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_amplitude_uncertainty=0.08,  # 8% reconstruction model uncertainty
    waveform_time_uncertainty=0.02,       # 2% temporal (reconstructed)
    waveform_uncertainty_type="reconstruction",  # Reconstructed from damped RLC parameters
    waveform_provenance="reconstructed",
    measurement_notes=(
        "FAETON-I: 100 kV, 125 kJ DPF by Fuse Energy Technologies. "
        "Highest direct-charged voltage PF device. 5 x 5 uF Marx bank = 25 uF total. "
        "Static inductance L0 = 220 nH (Damideh 2025). R0 = 7.6 mOhm estimated from "
        "measured I_peak/I_sc ratio (I_peak ~ 1 MA vs I_sc_undamped = 1.066 MA, RESF = 0.081). "
        "Cathode radius 10.6 cm from Damideh et al. (2025) Table 1 (directly stated). "
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


MJOLNIR_DATA = ExperimentalDevice(
    name="MJOLNIR-1MJ",
    institution="Lawrence Livermore National Laboratory",
    # All circuit + geometry values from Schmidt et al. 2021 §III.A
    # [KR: ieee-trans-plas-sci-paper-first-experiments-and-radiographs-on-the-megajoule-neutron-imaging.md §III.A lines 145-159]
    # Verbatim: "lumped circuit capacitance of 204 µF, inductance of 67.4 nH and resistance of 12.5 mOhm"
    # Verbatim: "fielded anodes... 15.2 cm (6 inches) in diameter" (a = 0.076 m)
    # Verbatim: "anode-cathode gap is fixed at 4.3 cm" (cathode_r = a + gap = 0.076 + 0.043 = 0.119 m)
    # Verbatim: "exposed lengths varying from 18.3 to 22.1 cm" (midpoint 0.20 m used)
    capacitance=204e-6,            # 204 uF (Schmidt 2021 §III.A line 149, lumped)
    voltage=100e3,                 # 100 kV erected (±50 kV per Marx; Schmidt 2021 §III.A line 144-145)
    inductance=67.4e-9,            # 67.4 nH (Schmidt 2021 §III.A line 149, lumped)
    resistance=12.5e-3,            # 12.5 mOhm (Schmidt 2021 §III.A line 150, lumped — already includes parallel-tower combination)
    anode_radius=0.076,            # 76 mm = 15.2 cm dia / 2 (Schmidt 2021 §III.A line 156)
    cathode_radius=0.119,          # 119 mm = 76 mm + 43 mm A-K gap (Schmidt 2021 §III.A line 159)
    anode_length=0.20,             # 200 mm midpoint of 18.3-22.1 cm range (Schmidt 2021 §III.A line 157)
    fill_pressure_torr=7.0,        # 7 Torr D2 (estimated, pressure scans performed)
    fill_gas="deuterium",
    peak_current=2.5e6,            # 2.5 MA at 100 kV erected, shorted load (Schmidt 2021 §III.A line 144)
    neutron_yield=3.8e11,          # 3.8e11 D-D at 1 MJ / 2.5 MA (Schmidt 2021)
    current_rise_time=5.83e-6,     # T/4 = pi/2 * sqrt(L*C) analytic estimate; ~5 us measured
    reference=(
        "Schmidt et al., IEEE Trans. Plasma Sci. (2021) "
        "DOI: 10.1109/TPS.2021.3106313 [KR: ieee-trans-plas-sci-paper-first-"
        "experiments-and-radiographs-on-the-megajoule-neutron-imaging.md §III.A]"
    ),
    crowbar_resistance=1.5e-3,     # estimated spark gap resistance
    peak_current_uncertainty=0.08, # 8% (Rogowski coil + integration)
    rise_time_uncertainty=0.15,    # 15% (analytic estimate, not paper-pinned)
    neutron_yield_uncertainty=0.50,  # 50% (shot-to-shot)
    waveform_t=_MJOLNIR_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_MJOLNIR_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_amplitude_uncertainty=0.10,  # 10% reconstruction model uncertainty
    waveform_time_uncertainty=0.03,       # 3% temporal (reconstructed)
    waveform_uncertainty_type="reconstruction",
    waveform_provenance="reconstructed",
    lee_fc=0.70, lee_fm=0.50, lee_fmr=0.10, lee_fcr=0.14,
    lee_reference=(
        "UNVERIFIED: no published Lee model fit on disk for MJOLNIR 1-MJ "
        "configuration. Prior 'Gemini research synthesis (2026-03-13)' tag was a "
        "papers-are-truth violation -- LLM-generated, not paper-anchored. Schmidt "
        "2021 §III.A characterizes the bank/geometry but does not publish "
        "fc/fm/fmr/fcr. LLNL uses Chicago PIC code, not Lee model. To resolve: "
        "retrieve a paper publishing Lee fits for the 3-tower / 12-module 1-MJ "
        "configuration, or accept that this device cannot be Lee-validated."
    ),
    measurement_notes=(
        "MJOLNIR (MegaJOuLe Neutron Imaging Radiography): MA-class DPF at LLNL. "
        "ATLAS-heritage pulsed power: 1-MJ configuration uses three Marx towers "
        "with twelve single-stage Marx modules, each containing two 34 uF capacitors "
        "and one railgap switch (Schmidt 2021 §III.A lines 135-140). Charged to +/- 50 kV "
        "(100 kV erected). Reaches 2.5 MA into shorted load (Schmidt 2021 line 144). "
        "LUMPED CIRCUIT (Schmidt 2021 §III.A lines 148-150, verbatim): "
        "C0 = 204 uF, L0 = 67.4 nH, R0 = 12.5 mOhm. "
        "These are the bank+cable+plate values from the shorted-load calibration shot; "
        "they are ALREADY the parallel-combined lumped values. "
        "Previous code values (C=408 uF, L=80 nH, R=6.25 mOhm) double-counted a "
        "tower-parallelism halving that was already baked into Schmidt's quoted values. "
        "ELECTRODE GEOMETRY (Schmidt 2021 §III.A lines 155-159, verbatim): "
        "Oxygen-free copper. Anode diameter 15.2 cm (6 inches) -> a = 0.076 m. "
        "Anode-cathode gap fixed at 4.3 cm -> cathode_r = a + 0.043 = 0.119 m. "
        "Anode exposed lengths 18.3-22.1 cm (multiple anodes fielded; midpoint 0.20 m used). "
        "MACOR insulator exposed length 4.6 cm. Pre-drilled hollow radius 0.9 or 3.8 cm. "
        "WAVEFORM: RECONSTRUCTED (phenomenological), NOT digitized from paper. "
        "Lee fc/fm/fmr/fcr from Gemini synthesis — needs replacement once a paper-fit "
        "Lee parameter set is published. "
        "Performance records: 2.5 MA / 3.8e11 DD neutrons at 1 MJ (Schmidt 2021); "
        "3.7-3.8 MA / >1e12 DD neutrons with rebuilt 24-module bank (Schmidt 2024); "
        "1.84e12 DT neutrons at 2 MA (Schmidt 2024). "
        "LLNL uses Chicago PIC code for simulation, not Lee model."
    ),
)


# MJOLNIR 2-MJ Goyon configuration — companion to MJOLNIR_DATA (Schmidt 1-MJ).
# These are different physical configurations of the same machine: the 1-MJ
# version (3 Marx towers, 12 modules) was characterized by Schmidt 2021 §III.A;
# the 2-MJ upgrade (6 towers, 24 modules) is characterized by Goyon 2025 + Petrov 2022.
# Validation against a single "MJOLNIR" entry conflated the two configurations
# and produced apples-to-oranges error metrics. Splitting per Wave-7 S18 drift
# table HIGH-severity recommendation; preset "mjolnir" remains the 2-MJ Goyon
# configuration and now correctly pairs with MJOLNIR_2MJ_DATA.
#
# [KR: petrov-2022-mjolnir-high-low-discharges.md §II.A L228-232]
#   "Estimated lumped circuit parameters for the bank, including protective
#    resistors is 408 µF capacitance, with 46.7 nH inductance and 6.3 mOhm
#    resistance."
# [KR: neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md §II.A]
#   "228.6 mm diameter copper anode" -> a = 0.1143 m
#   "60 kV / 735 kJ stored / 2.8 MA peak / 2.1 MA at stagnation"
#   "24 cathode rods around anode"; "anode-cathode gap is fixed at 4.3 cm"
MJOLNIR_2MJ_DATA = ExperimentalDevice(
    name="MJOLNIR-2MJ",
    institution="Lawrence Livermore National Laboratory",
    capacitance=408e-6,            # 408 uF (Petrov 2022 §II.A L230, lumped)
    voltage=60e3,                  # 60 kV typical (Goyon 2025 §IV)
    inductance=46.7e-9,            # 46.7 nH (Petrov 2022 §II.A L230, lumped)
    resistance=6.3e-3,             # 6.3 mOhm (Petrov 2022 §II.A L230, lumped)
    anode_radius=0.1143,           # 114.3 mm = 228.6 mm OD / 2 (Goyon 2025 §II.A)
    cathode_radius=0.157,          # 157 mm (Goyon 2025: 24-rod cage; 4.3 cm gap)
    anode_length=0.20,             # 200 mm (Schmidt 2021 §III.A range; 2-MJ inherits)
    fill_pressure_torr=6.0,        # 6 Torr D2 (Goyon 2025 §IV operating point)
    fill_gas="deuterium",
    peak_current=2.8e6,            # 2.8 MA peak (Goyon 2025 §IV at 60 kV)
    neutron_yield=8e11,            # 8e11 D-D peak demonstrated (Goyon 2025 line 64)
    current_rise_time=8.7e-6,      # T/4 = pi/2 * sqrt(L*C) = pi/2 * sqrt(46.7e-9 * 408e-6) = 6.9 us; rounded up for damping
    reference=(
        "Goyon et al., Phys. Plasmas 32:033105 (2025); "
        "Petrov et al., LLNL-JRNL-831591 (2022) "
        "[KR: petrov-2022-mjolnir-high-low-discharges.md §II.A L228-232; "
        "KR: neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md §II.A]"
    ),
    crowbar_resistance=1.5e-3,     # estimated spark gap
    peak_current_uncertainty=0.08,
    rise_time_uncertainty=0.15,
    neutron_yield_uncertainty=0.50,
    waveform_t=_MJOLNIR_WAVEFORM_T_US * 1e-6,
    waveform_I=_MJOLNIR_WAVEFORM_I_KA * 1e3,
    waveform_amplitude_uncertainty=0.10,
    waveform_time_uncertainty=0.03,
    waveform_uncertainty_type="reconstruction",
    waveform_provenance="reconstructed",
    lee_fc=0.70, lee_fm=1.0, lee_fmr=0.10, lee_fcr=0.14,
    lee_reference=(
        "UNVERIFIED: no published Lee model fit on disk for the 2-MJ configuration. "
        "Values carried from preset 'mjolnir' calibration (papers-are-truth violation; "
        "fm=1.0 was RADPF-target-fitted to 2.8 MA at 60 kV). To resolve: retrieve a "
        "paper publishing fc/fm/fmr/fcr for the 6-tower / 24-module Goyon 2025 config "
        "(Schmidt 2024 follow-up promised in Schmidt 2021 line 626 'future work')."
    ),
    measurement_notes=(
        "MJOLNIR 2-MJ upgrade configuration (6 Marx towers, 24 single-stage modules; "
        "Goyon 2025 §II.A L107-114). 2 MJ stored energy at maximum voltage; 60 kV "
        "operating point used for §IV data with 735 kJ stored and 2.8 MA peak / 2.1 MA "
        "at stagnation (Goyon 2025 L429-431). Anode 228.6 mm OD copper, 24 cathode rods, "
        "4.3 cm A-K gap (Goyon 2025 L122). Lumped C/L/R from Petrov 2022 §II.A L228-232 "
        "(snowplow-fit estimate; no shorted-load measurement available for 2-MJ). "
        "Schmidt 2024 reports rebuilt 24-module bank delivering 3.7-3.8 MA / >1e12 DD "
        "neutrons; that paper is NOT on disk so the 2.8 MA / 8e11 figures from Goyon "
        "2025 are the canonical anchors here."
    ),
)


# REFERENCE_ONLY — DROPPED from DEVICES registry.
# Lee & Saw 2014 (KR lines 4177-4183) explicitly classifies Syrian plasma focus
# devices as "Type 2 unfittable": "2 plasma focus in Syria ... cannot be fitted by
# the model code." The AECS-PF2 is one of these Syrian devices (high inductance,
# ~200 nH or >1000 nH per KR line 4179). Using this device for Lee-model validation
# is scientifically invalid. Moved to _REFERENCE_ONLY.
AECS_PF2_DATA = ExperimentalDevice(
    name="AECS-PF2",
    institution="Atomic Energy Commission of Syria",
    capacitance=25e-6,             # 25 uF # UNVERIFIED: source not on disk
    voltage=15e3,                  # 15 kV (E = 2.8 kJ) # UNVERIFIED
    inductance=110e-9,             # 110 nH # UNVERIFIED
    resistance=30e-3,              # 30 mOhm (high impedance) # UNVERIFIED
    anode_radius=0.0095,           # 9.5 mm # UNVERIFIED
    cathode_radius=0.032,          # 32 mm # UNVERIFIED
    anode_length=0.16,             # 160 mm # UNVERIFIED
    fill_pressure_torr=2.0,        # 2 Torr D2 (midpoint of 1-4 Torr range) # UNVERIFIED
    fill_gas="deuterium",
    peak_current=90e3,             # ~90 kA (Lee & Saw AAAPT survey) # UNVERIFIED
    neutron_yield=1e6,             # ~1e6 (estimated, small device at 2 Torr) # UNVERIFIED
    current_rise_time=1.7e-6,      # ~1.7 us (T/4 from RLC params) # UNVERIFIED
    reference=(
        "DROPPED: Lee & Saw 2014 classifies Syrian PF as Type 2 unfittable "
        "[KR: a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md lines 4177-4183]"
    ),
    lee_fc=0.7, lee_fm=0.15, lee_fmr=0.0, lee_fcr=0.0,
    lee_reference="INVALID: Syrian PF is Type 2 unfittable per Lee & Saw 2014 [KR lines 4177-4183]",
    peak_current_uncertainty=0.15,     # 15% (small device, limited diagnostics)
    rise_time_uncertainty=0.20,        # 20%
    neutron_yield_uncertainty=0.70,    # 70% (shot-to-shot)
    waveform_provenance="",  # No waveform data available
    kr_status="reference_only",
    measurement_notes=(
        "AECS-PF2: DROPPED from DEVICES registry. "
        "Lee & Saw 2014 explicitly classifies the Syrian plasma focus devices as "
        "Type 2 unfittable by the Lee model code: 'There are also reports from our "
        "associates with various plasma focus which cannot be fitted. These include "
        "2 plasma focus in Syria and one in Iran.' (KR lines 4177-4183). "
        "High inductance (110 nH or higher) prevents the RADPF model from fitting "
        "the measured current waveform. All parameter values below are UNVERIFIED — "
        "source papers (Lee & Saw AAAPT survey, Lee J. Fusion Energy 33:319 2014) "
        "are NOT present in KnowledgeReference/. "
        "Do NOT use for Lee-model validation claims."
    ),
)


# =====================================================================
# Registry mapping device name -> ExperimentalDevice
# =====================================================================

DEVICES: dict[str, ExperimentalDevice] = {
    "PF-1000": PF1000_DATA,
    "PF-1000-Gribkov": PF1000_GRIBKOV_DATA,
    "PF-1000-16kV": PF1000_16KV_DATA,
    "PF-1000-20kV": PF1000_20KV_DATA,
    "NX2": NX2_DATA,
    "UNU-ICTP": UNU_ICTP_DATA,       # kr_status="verified" — Lee & Saw 2014 Table p.152
    "POSEIDON-60kV": POSEIDON_60KV_DATA,  # kr_status="verified" — Lee & Saw 2014 Table p.152
    "FAETON-I": FAETON_DATA,
    "MJOLNIR-1MJ": MJOLNIR_DATA,        # Schmidt 2021 §III.A 1-MJ baseline (3 towers)
    "MJOLNIR-2MJ": MJOLNIR_2MJ_DATA,    # Goyon 2025 + Petrov 2022 2-MJ upgrade (6 towers)
    "MJOLNIR": MJOLNIR_2MJ_DATA,        # alias: preset "mjolnir" runs 2-MJ Goyon config
                                         # so default validation pairs with that.
                                         # Use "MJOLNIR-1MJ" for Schmidt baseline validation.
}

# Devices excluded from DEVICES because their parameters cannot be sourced from
# KnowledgeReference/ or because Lee & Saw 2014 explicitly classifies them as
# unsuitable for Lee-model fitting.
#
# POSEIDON (40 kV): requires Herold 1989 Nucl. Fusion 29 (not on disk).
# AECS-PF2: Type 2 unfittable per Lee & Saw 2014 [KR lines 4177-4183].
_REFERENCE_ONLY: dict[str, ExperimentalDevice] = {
    "POSEIDON": POSEIDON_DATA,
    "AECS-PF2": AECS_PF2_DATA,
}


def get_devices_by_provenance(
    provenance: str = "measured",
) -> dict[str, ExperimentalDevice]:
    """Return devices filtered by waveform provenance.

    Args:
        provenance: One of "measured", "reconstructed", or "" (no waveform).

    Returns:
        Dict of device_name -> ExperimentalDevice for matching devices.
    """
    return {
        name: dev for name, dev in DEVICES.items()
        if dev.waveform_provenance == provenance
        and dev.waveform_t is not None
    }
