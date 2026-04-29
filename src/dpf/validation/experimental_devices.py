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
    reference="Lee & Saw, J. Fusion Energy 27:292, 2008; RADPF Module 1",
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

# UNVERIFIED — no KR source.
# Cited references (Lee et al. Am. J. Phys. 56, 1988; IPFS plasmafocus.net "UNU ICTPPFF
# D2 05.15.xls"; Lee & Saw 2014 fit parameters) are NOT present under
# /Users/anthonyzamora/dpf-unified/KnowledgeReference/. Values inherited from prior
# code revisions and cannot be re-tagged with a [KR: ...] citation.
# Per ground rule (2026-04-27), all device parameters require a KR-canonical source.
# Treat I_peak/timing/Lee-fit values here as "reference_only" until a KR source is added.
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
    lee_fc=0.7, lee_fm=0.08, lee_fmr=0.16, lee_fcr=0.7,
    lee_reference="IPFS plasmafocus.net preset; Lee & Saw (2014)",
    peak_current_uncertainty=0.10,     # 10% (training device, less precise)
    rise_time_uncertainty=0.15,        # 15%
    neutron_yield_uncertainty=0.70,    # 70% (shot-to-shot)
    waveform_t=_UNU_ICTP_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_UNU_ICTP_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_amplitude_uncertainty=0.016,  # GUM: 9.3 kA / (2*sqrt(3)*169 kA) = 1.6% (rectangular)
    waveform_time_uncertainty=0.002,       # 0.2% (~1 ns digitization on ~5 us trace)
    waveform_uncertainty_type="digitization",  # Type B: quantization from digital oscilloscope
    waveform_provenance="measured",
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
    lee_fc=0.70, lee_fm=0.20, lee_fmr=0.12, lee_fcr=0.47,
    lee_reference="Akel et al., Radiat. Phys. Chem. 188:109633, 2021 (24-shot avg at 16 kV)",
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


# UNVERIFIED — no KR source.
# Cited references (Herold et al. Nucl. Fusion 29:33, 1989; Lee & Saw J. Fusion Energy
# 33:319, 2014) are NOT present under
# /Users/anthonyzamora/dpf-unified/KnowledgeReference/. L0=20 nH and R0=2 mOhm are
# explicitly noted as RADPF default estimates, not measured values from Herold (1989).
# Per ground rule (2026-04-27), treat parameters as reference-only until a KR source
# is added.
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
    lee_fc=0.60, lee_fm=0.275, lee_fmr=0.45, lee_fcr=0.44,
    lee_reference="Lee & Saw, J. Fusion Energy 33:319 (2014), at 60 kV",
    peak_current_uncertainty=0.08,     # 8% (large device, Rogowski + integration)
    rise_time_uncertainty=0.15,        # 15% (not explicitly stated in source)
    neutron_yield_uncertainty=0.50,    # 50% (shot-to-shot)
    waveform_provenance="",  # No waveform data available
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


# UNVERIFIED — no KR source.
# Cited references (IPFS plasmafocus.net Excel file; Herold et al. Nucl. Fusion 29:33,
# 1989; Lee & Saw 2014) are NOT present under
# /Users/anthonyzamora/dpf-unified/KnowledgeReference/. Lee-fit values (fc=0.595,
# fm=0.275, fmr=0.45, fcr=0.44) and circuit parameters (L0=17.7 nH, R0=1.7 mOhm,
# zo=300 mm) come from an IPFS Lee-model fit, not a peer-reviewed paper on disk.
# Per ground rule (2026-04-27), treat parameters as reference-only until a KR source
# is added.
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
    waveform_amplitude_uncertainty=0.02,  # 2% (IPFS digitization, high quality)
    waveform_time_uncertainty=0.005,      # 0.5% temporal
    waveform_uncertainty_type="digitization",  # Type B: IPFS digital archive
    waveform_provenance="measured",
    lee_fc=0.60, lee_fm=0.275, lee_fmr=0.45, lee_fcr=0.44,
    lee_reference="IPFS (plasmafocus.net) Lee model fit; Lee & Saw 2014",
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
    name="MJOLNIR",
    institution="Lawrence Livermore National Laboratory",
    # All circuit + geometry values from Schmidt et al. 2021 §III.A
    # [KR: ieee-trans-plas-sci-paper-first-experiments-and-radiographs-on-the-megajoule-neutron-imaging.md §III.A lines 145-159]
    # Verbatim: "lumped circuit capacitance of 204 µF, inductance of 67.4 nH and resistance of 12.5 mOhm"
    # Verbatim: "fielded anodes... 15.2 cm (6 inches) in diameter" (a = 0.076 m)
    # Verbatim: "anode-cathode gap is fixed at 4.3 cm" (cathode_r = a + gap = 0.076 + 0.043 = 0.119 m)
    # Verbatim: "exposed lengths varying from 18.3 to 22.1 cm" (midpoint 0.20 m used)
    capacitance=204e-6,            # 204 uF (Schmidt 2021 §III.A line 149, lumped)
    voltage=60e3,                  # 60 kV typical operation
    inductance=67.4e-9,            # 67.4 nH (Schmidt 2021 §III.A line 149, lumped)
    resistance=12.5e-3,            # 12.5 mOhm (Schmidt 2021 §III.A line 150, lumped — already includes parallel-tower combination)
    anode_radius=0.076,            # 76 mm = 15.2 cm dia / 2 (Schmidt 2021 §III.A line 156)
    cathode_radius=0.119,          # 119 mm = 76 mm + 43 mm A-K gap (Schmidt 2021 §III.A line 159)
    anode_length=0.20,             # 200 mm midpoint of 18.3-22.1 cm range (Schmidt 2021 §III.A line 157)
    fill_pressure_torr=7.0,        # 7 Torr D2 (estimated, pressure scans performed)
    fill_gas="deuterium",
    peak_current=2.8e6,            # 2.8 MA at 60 kV (Goyon 2025)
    neutron_yield=3.8e11,          # 3.8e11 D-D at 1 MJ / 2.5 MA (Schmidt 2021)
    current_rise_time=5.0e-6,      # ~5 us (Schmidt 2024)
    reference=(
        "Schmidt et al., IEEE Trans. Plasma Sci. (2021) "
        "DOI: 10.1109/TPS.2021.3106313 [KR: ieee-trans-plas-sci-paper-first-"
        "experiments-and-radiographs-on-the-megajoule-neutron-imaging.md §III.A]"
    ),
    crowbar_resistance=1.5e-3,     # estimated spark gap resistance
    peak_current_uncertainty=0.08, # 8% (Rogowski coil + integration)
    rise_time_uncertainty=0.10,    # 10% (stated as ~5 us, not precise)
    neutron_yield_uncertainty=0.50,  # 50% (shot-to-shot)
    waveform_t=_MJOLNIR_WAVEFORM_T_US * 1e-6,      # Convert us -> s
    waveform_I=_MJOLNIR_WAVEFORM_I_KA * 1e3,        # Convert kA -> A
    waveform_amplitude_uncertainty=0.10,  # 10% reconstruction model uncertainty
    waveform_time_uncertainty=0.03,       # 3% temporal (reconstructed)
    waveform_uncertainty_type="reconstruction",
    waveform_provenance="reconstructed",
    lee_fc=0.70, lee_fm=0.50, lee_fmr=0.10, lee_fcr=0.14,
    lee_reference="Gemini research synthesis (2026-03-13); Lee model conventions",
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


# UNVERIFIED — no KR source.
# Cited references (Lee & Saw AAAPT device survey; Lee, J. Fusion Energy 33:319, 2014)
# are NOT present under /Users/anthonyzamora/dpf-unified/KnowledgeReference/.
# I_peak ~90 kA and Lee-fit values (fc=0.7, fm=0.15, fmr=0, fcr=0) are taken from
# AAAPT publications not on disk. Per ground rule (2026-04-27), treat parameters as
# reference-only until a KR source is added.
AECS_PF2_DATA = ExperimentalDevice(
    name="AECS-PF2",
    institution="Atomic Energy Commission of Syria",
    capacitance=25e-6,             # 25 uF
    voltage=15e3,                  # 15 kV (E = 2.8 kJ)
    inductance=110e-9,             # 110 nH
    resistance=30e-3,              # 30 mOhm (high impedance)
    anode_radius=0.0095,           # 9.5 mm
    cathode_radius=0.032,          # 32 mm
    anode_length=0.16,             # 160 mm
    fill_pressure_torr=2.0,        # 2 Torr D2 (midpoint of 1-4 Torr range)
    fill_gas="deuterium",
    peak_current=90e3,             # ~90 kA (Lee & Saw AAAPT survey)
    neutron_yield=1e6,             # ~1e6 (estimated, small device at 2 Torr)
    current_rise_time=1.7e-6,      # ~1.7 us (T/4 from RLC params)
    reference="Lee & Saw, AAAPT device survey; Lee, J. Fusion Energy 33:319 (2014)",
    lee_fc=0.7, lee_fm=0.15, lee_fmr=0.0, lee_fcr=0.0,
    lee_reference="Lee & Saw, AAAPT publications (Lee 2014 Review)",
    peak_current_uncertainty=0.15,     # 15% (small device, limited diagnostics)
    rise_time_uncertainty=0.20,        # 20% (not explicitly stated in source)
    neutron_yield_uncertainty=0.70,    # 70% (shot-to-shot, small device)
    waveform_provenance="",  # No waveform data available
    measurement_notes=(
        "AECS-PF2: 2.8 kJ DPF at the Atomic Energy Commission of Syria. "
        "High-impedance small device: RESF = R0/sqrt(L0/C) = 30e-3/sqrt(110e-9/25e-6) = 1.27. "
        "At RESF > 1, the circuit is overdamped without plasma loading — the DPF discharge "
        "relies on plasma inductance growth to prevent overdamping. "
        "I_peak ~90 kA from Lee & Saw AAAPT device survey and Lee (2014) Review. "
        "Fill pressure range 1-4 Torr D2; 2 Torr used as midpoint reference. "
        "Lee model fits fc=0.7, fm=0.15 from AAAPT publications. "
        "No digitized waveform available — scalar validation (I_peak, timing) only. "
        "Uncertainties are Type B estimates (not stated in source)."
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
    "UNU-ICTP": UNU_ICTP_DATA,
    "POSEIDON": POSEIDON_DATA,
    "POSEIDON-60kV": POSEIDON_60KV_DATA,
    "FAETON-I": FAETON_DATA,
    "MJOLNIR": MJOLNIR_DATA,
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
