"""Source-backed target packets for first-principles DPF development.

These packets translate local KnowledgeReference source facts into structured
engineering inputs and blocker facts. They are not validation evidence and do
not promote any whole-shot first-principles claim by themselves.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

TORR_TO_PA = 133.32236842105263

GV_ROOT = "/Users/anthonyzamora/Downloads/GV"

GV_REDUCED_MODEL_OUTPUT_COLUMNS = (
    "time_normalized_to_quarter_cycle",
    "current_normalized_to_I0",
    "capacitor_energy_fraction",
    "magnetic_energy_fraction",
    "resistive_dissipation_fraction",
    "work_done_fraction",
    "tau",
    "dynamic_dimensionless_inductance",
    "time_us",
    "current_kA",
    "total_inductance_nH",
)

GV_VERIFIED_SHOTS = (
    {
        "shot_id": "lpp_ff1_05_23_16_1",
        "device": "LPP-FF1",
        "shot_note": "shot05_23_16_1; data kindly provided by Dr. Eric Lerner",
        "input_file": "Gvinp -LPP-FF1-05_23_16_1.inp",
        "input_sha256": (
            "522795d96f4c9e6a4c4c5f6df1a369749d5490d047bd24b2ade1dc93fc92daac"
        ),
        "txt_file": "LPP-FF1-05_23_16_1.TXT",
        "txt_sha256": (
            "0e7c77da81df09ebccb26607ca3f8df3911900a64e5cedcadc3a180f1867a232"
        ),
        "xlsx_file": "LPP-FF1-05_23_16_1.xlsx",
        "xlsx_sha256": (
            "30975c452deb35828988c9bc9971cf1585e6f3269c6e181621c415e471b5c54b"
        ),
        "geometry_mm": {
            "anode_radius": 28.0,
            "anode_length": 140.0,
            "insulator_radius": 35.3,
            "insulator_length": 27.0,
            "cathode_radius": 50.0,
        },
        "circuit": {
            "capacitance_uF": 75.2,
            "inductance_nH": 34.0,
            "resistance_milliohm": 7.0,
            "voltage_kV": 39.9,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 12.0},
        "waveform_columns": {"time_us": "L", "current_kA": "M", "rows": 6789},
        "workbook_note": (
            "Columns L, M contain experimental data converted to kA and usec; "
            "fitted pressure 12.0 torr, actual pressure 18.1 mbar."
        ),
    },
    {
        "shot_id": "lpp_ff1_05_24_16_6",
        "device": "LPP-FF1",
        "shot_note": "shot05-24-16-06; data kindly provided by Dr. Eric Lerner",
        "input_file": "Gvinp - LPP-FF1-05_24_16_6.inp",
        "input_sha256": (
            "3a691e41f91caa353d5840e441fb5509212737855fb9a75bdc56e556b1d46a1a"
        ),
        "txt_file": "LPP-FF1-05_24_16_6.TXT",
        "txt_sha256": (
            "9689b356602785a984314094d10bb79497c72cbbad99f2d7fb39e5755fc0cb7a"
        ),
        "xlsx_file": "LPP-FF1-05_24_16_6.xlsx",
        "xlsx_sha256": (
            "8e5c14f2c77e0a4623f49394e4935f64693083350db3d73fca74965373c0d3d4"
        ),
        "geometry_mm": {
            "anode_radius": 28.0,
            "anode_length": 140.0,
            "insulator_radius": 35.3,
            "insulator_length": 27.0,
            "cathode_radius": 50.0,
        },
        "circuit": {
            "capacitance_uF": 75.2,
            "inductance_nH": 35.0,
            "resistance_milliohm": 5.6,
            "voltage_kV": 40.1,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 12.2},
        "waveform_columns": {"time_us": "L", "current_kA": "M", "rows": 6789},
        "workbook_note": (
            "Columns L, M contain experimental data converted to kA and usec; "
            "fitted pressure 12.2 torr, actual pressure 18.1 mbar=13.6 torr."
        ),
    },
    {
        "shot_id": "pf24_krakow_14082734",
        "device": "PF-24-KRAKOW",
        "shot_note": "shot#14082734; data kindly provided by Dr. Marek Scholz",
        "input_file": "Gvinp-PF-24-KRAKOW-14082734.inp",
        "input_sha256": (
            "f7a7d0713f34cd25cc45015cb198730c8c49141efa2e0cbb0d1b582df5184d1c"
        ),
        "txt_file": "PF-24-KRAKOW-14082734.TXT",
        "txt_sha256": (
            "63aa6bde3936a10453e5110adc7d993eb1d0c69802099f637ffc217f36addc50"
        ),
        "xlsx_file": "PF-24-KRAKOW-14082734.xlsx",
        "xlsx_sha256": (
            "5e7bd36f21ab74ee468b2b2986b39876b378e15fd2d1125b15add50a6140a398"
        ),
        "geometry_mm": {
            "anode_radius": 31.0,
            "anode_length": 172.0,
            "insulator_radius": 31.5,
            "insulator_length": 40.0,
            "cathode_radius": 49.0,
        },
        "circuit": {
            "capacitance_uF": 115.2,
            "inductance_nH": 15.0,
            "resistance_milliohm": 8.0,
            "voltage_kV": 16.0,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 1.35},
        "waveform_columns": {"time_us": "L", "current_kA": "M", "rows": 600},
        "workbook_note": (
            "Columns L, M contain experimental data as supplied; fitted pressure "
            "1.35 torr, actual pressure 2.4 mbar=1.8 torr."
        ),
    },
    {
        "shot_id": "pf24_krakow_16052007",
        "device": "PF-24-KRAKOW",
        "shot_note": "shot#16052007; data kindly provided by Dr. Marek Scholz",
        "input_file": "Gvinp - PF-24-KRAKOW-16052007.inp",
        "input_sha256": (
            "02a76b0ab0db1353970313efe4f25846ce0d6c8bb1fe57a9a8f531898557ee7f"
        ),
        "txt_file": "PF-24-KRAKOW-16052007.TXT",
        "txt_sha256": (
            "fcb8fe6bc3c637ba215eb090d6c6b7fe2c667e1576e1936f2ea5c04a7b6e172e"
        ),
        "xlsx_file": "PF-24-KRAKOW-16052007.xlsx",
        "xlsx_sha256": (
            "3a897522b30fe0d3fcea209d41ca30b94a5203f3bef6ae82595377ef1139aa7a"
        ),
        "geometry_mm": {
            "anode_radius": 31.0,
            "anode_length": 172.0,
            "insulator_radius": 31.5,
            "insulator_length": 40.0,
            "cathode_radius": 49.0,
        },
        "circuit": {
            "capacitance_uF": 115.2,
            "inductance_nH": 20.0,
            "resistance_milliohm": 14.0,
            "voltage_kV": 16.0,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 1.1},
        "waveform_columns": {"time_us": "L", "current_kA": "M", "rows": 651},
        "workbook_note": (
            "Columns L, M contain experimental data as supplied converted to usec; "
            "fitted pressure 1.1 torr, actual pressure 2.0 mbar=1.5 torr."
        ),
    },
    {
        "shot_id": "pf24_krakow_16092202",
        "device": "PF-24-KRAKOW",
        "shot_note": "shot#16092202; data kindly provided by Dr. Marek Scholz",
        "input_file": "Gvinp-PF-24-KRAKOW-16092202.inp",
        "input_sha256": (
            "6ed5aa219c6e371a384485fe163c8116680e0711668d08507ffe457969a785e2"
        ),
        "txt_file": "PF-24-KRAKOW-16092202.TXT",
        "txt_sha256": (
            "0ffe8638abd054864643e69097ca6508b69d544293ffa47a89219ffbd3f5bbbc"
        ),
        "xlsx_file": "PF-24-KRAKOW-16092202.xlsx",
        "xlsx_sha256": (
            "43ef75fd63caf1aaa4fc7be72b6e92c63851c6e9220daa6291be974a92a02e73"
        ),
        "geometry_mm": {
            "anode_radius": 31.0,
            "anode_length": 172.0,
            "insulator_radius": 31.5,
            "insulator_length": 40.0,
            "cathode_radius": 49.0,
        },
        "circuit": {
            "capacitance_uF": 115.2,
            "inductance_nH": 21.0,
            "resistance_milliohm": 22.0,
            "voltage_kV": 16.0,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 1.1},
        "waveform_columns": {"time_us": "L", "current_kA": "M", "rows": 649},
        "workbook_note": (
            "Columns L, M contain experimental data as supplied converted to usec; "
            "fitted pressure 1.1 torr, actual pressure 2.0 mbar=1.5 torr."
        ),
    },
    {
        "shot_id": "pf360_20140122_7",
        "device": "PF-360",
        "shot_note": "shot20140122_7; data kindly provided by Prof. Marek Sadowski",
        "input_file": "Gvinp - PF-360.inp",
        "input_sha256": (
            "37ccc655ad7c4447ecbcca2c87f8ce416c4f5f0a55db9b073cae039d5b918560"
        ),
        "txt_file": "PF-360.TXT",
        "txt_sha256": (
            "951d9f33bd43a069bcd6ade465f445231783473e73e1002ca4a19a2eb75e0329"
        ),
        "xlsx_file": "PF-360.xlsx",
        "xlsx_sha256": (
            "0007a509d49c753f7338b43a655221f2c67593b721a24890cc4524164ee6b072"
        ),
        "geometry_mm": {
            "anode_radius": 60.0,
            "anode_length": 304.0,
            "insulator_radius": 60.1,
            "insulator_length": 75.0,
            "cathode_radius": 75.0,
        },
        "circuit": {
            "capacitance_uF": 262.6,
            "inductance_nH": 17.0,
            "resistance_milliohm": 5.2,
            "voltage_kV": 31.0,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 10.0},
        "waveform_columns": {
            "smoothed_time_us": "L",
            "smoothed_current_kA": "M",
            "raw_time_us": "AC",
            "raw_current_kA": "AD",
            "rows": 22980,
        },
        "workbook_note": (
            "Columns L, M contain experimental data smoothed with a 400 point "
            "moving average; columns AC, AD contain unsmoothed data converted "
            "to kA and microseconds; fitted pressure 10 torr, actual pressure "
            "6.8 mbar."
        ),
    },
    {
        "shot_id": "gemini_rog_i005_20130716",
        "device": "Gemini",
        "shot_note": "DPF Data 2013/07162013/ROG_I005.dig; data kindly provided by Dr. Marek Scholz",
        "input_file": "Gvinp-Gemini.inp",
        "input_sha256": (
            "6ff50da814e0e1626d61305d6327e0ac8098af5d4cc18b621120a03db8138535"
        ),
        "txt_file": "Gemini.TXT",
        "txt_sha256": (
            "7bb2423ab847ded0527102ee5fb8a7c8cc8040f6d880f05a68188761f5bd7dba"
        ),
        "xlsx_file": "Gemini.xlsx",
        "xlsx_sha256": (
            "b62c62383ff03bb1ecce6cbd508be5054450ec21d1febd591021c32a337730d3"
        ),
        "geometry_mm": {
            "anode_radius": 76.2,
            "anode_length": 596.9,
            "insulator_radius": 77.0,
            "insulator_length": 127.0,
            "cathode_radius": 101.6,
        },
        "circuit": {
            "capacitance_uF": 432.0,
            "inductance_nH": 29.7,
            "resistance_milliohm": 2.5,
            "voltage_kV": 40.13,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 5.7},
        "waveform_columns": {
            "time_us": "L",
            "current_kA": "M",
            "raw_time_us": "AC",
            "raw_current_kA": "AD",
            "rows": 30387,
        },
        "workbook_note": (
            "Columns L and M are experimental waveform data with calibration "
            "and data reduction; columns AC and AD are raw data; fitted pressure "
            "5.7 torr, actual pressure 5.21 torr."
        ),
    },
    {
        "shot_id": "onesys_rog01004_20051208",
        "device": "OneSys",
        "shot_note": "DPF Data 2005/12082005/ROG01004.dig; data kindly provided by Dr. E.C. Hagen",
        "input_file": "Gvinp - OneSys.inp",
        "input_sha256": (
            "dfa321a545d59c3f62198c5662cb6c7da3b30d18bbeb9f5de41fe6849a498341"
        ),
        "txt_file": "OneSys.TXT",
        "txt_sha256": (
            "1d23b34a5c022a017fc517afa7327d2fd1c41570bd0a80f2661b4abf073b811f"
        ),
        "xlsx_file": "OneSys.xlsx",
        "xlsx_sha256": (
            "aca1bdaba358f0ab05c2656a21bbeb9099c275c82c74c156a490534ba6b2c2ae"
        ),
        "geometry_mm": {
            "anode_radius": 50.8,
            "anode_length": 393.7,
            "insulator_radius": 50.9,
            "insulator_length": 108.0,
            "cathode_radius": 76.2,
        },
        "circuit": {
            "capacitance_uF": 216.0,
            "inductance_nH": 46.0,
            "resistance_milliohm": 2.0,
            "voltage_kV": 35.0,
        },
        "gas": {"species": "D", "fitted_pressure_torr": 6.8},
        "waveform_columns": {
            "time_us": "L",
            "current_kA": "M",
            "raw_time_us": "AC",
            "raw_current_kA": "AD",
            "rows": 20002,
        },
        "workbook_note": (
            "Columns L and M are experimental waveform data with calibration "
            "and data reduction; columns AC and AD are raw data."
        ),
    },
)


MAY16_VALIDATED_THESES = (
    {
        "source_id": "arwinder_2015_comparative_pf_machines",
        "path": "/Users/anthonyzamora/Downloads/arwinderphdthesis.pdf",
        "sha256": (
            "2c7a8f4bd3b4d000638e4a7bd612a63d87cf1e179ee13365bd0dded40524b08c"
        ),
        "title": "Comparative Study of Plasma Focus Machines",
        "author_or_lead": "Arwinder Singh A/L Jigiri Singh",
        "document_type": "phd_thesis",
        "pages": 305,
        "text_status": "pdftotext_full_extract_available",
        "primary_uses": [
            "multi_machine_deck_registry_candidate",
            "measured_current_waveform_location_map",
            "Lee_model_baseline_and_fit_boundary_context",
            "second_scope_generalization_candidate",
        ],
        "useful_gate_ids": ["FP-10", "FP-15"],
        "candidate_facts": [
            "44 Mather-type plasma focus machines are analyzed across deuterium, neon, and argon operation",
            "PF-1000, Speed-2, and Filippov-type examples are included in the table and figure map",
            "Lee 5-phase and 6-phase fits are baseline/comparator context only",
        ],
        "not_authority_for": [
            "active first-principles closure",
            "accepted total neutron yield",
            "accepted whole-shot validation",
        ],
        "target_extraction_required": [
            "machine parameter tables",
            "measured current waveform figures",
            "device-specific operating gases and pressures",
            "baseline-only Lee fit metadata",
        ],
    },
    {
        "source_id": "talebitaher_2012_nx2_coded_aperture_imaging",
        "path": "/Users/anthonyzamora/Downloads/PhD2012AlirezaTalebitaher.pdf",
        "sha256": (
            "9b79429f0cc5b2b8a12e8e13c0331a61a354694bbe551eb51891a80b1d674af2"
        ),
        "title": "Coded Aperture Imaging of Nuclear Fusion in the Plasma Focus Device",
        "author_or_lead": "Alireza Talebitaher",
        "document_type": "phd_thesis",
        "pages": 304,
        "text_status": "pdftotext_full_extract_available",
        "primary_uses": [
            "fusion_source_spatial_image_candidate",
            "neutron_activation_detector_response_candidate",
            "neutron_anisotropy_target_candidate",
            "detector_forward_model_and_uq_method_candidate",
        ],
        "useful_gate_ids": ["FP-11", "FP-12", "FP-13", "FP-15"],
        "candidate_facts": [
            "NX2 D-D fusion source is imaged using coded aperture imaging of D(d,p)T protons",
            "CR-39 proton detectors, beryllium activation neutron detectors, and MCNP5 response calculations are described",
            "neutron-optimized NX2 operation is reported around 1-3e8 neutrons per shot at 1.6 kJ",
        ],
        "not_authority_for": [
            "PF-1000/Akel same-scope neutron acceptance",
            "accepted detector response without extraction and review",
        ],
        "target_extraction_required": [
            "NX2 machine parameters",
            "coded-mask geometry and source images",
            "beryllium activation response tables",
            "anisotropy/yield plots",
            "detector geometry and uncertainty packet",
        ],
    },
    {
        "source_id": "saw_1990_current_stepped_z_pinch",
        "path": "/Users/anthonyzamora/Downloads/sawsorheoh.pdf",
        "sha256": (
            "ad6e93b2d85363348874702c8ff55abd73ee2037eb2e5de464853c0cbb82d096"
        ),
        "title": "Experimental Studies of a Current-Stepped Z-Pinch",
        "author_or_lead": "Saw Sor Heoh",
        "document_type": "phd_thesis",
        "pages": 182,
        "text_status": "ocr_sidecar_created_from_scanned_pdf",
        "ocr_artifacts": [
            "tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.txt",
            "tmp/pdfs/may16_verified_batch/sawsorheoh_ocr.pdf",
        ],
        "primary_uses": [
            "current_step_driver_and_radial_compression_context",
            "shock_jump_and_saha_gamma_varying_model_context",
            "z_pinch_current_density_mapping_context",
            "startup_and_power_driver_method_reference",
        ],
        "useful_gate_ids": ["FP-5", "FP-6", "FP-8"],
        "candidate_facts": [
            "UMCSZP compares Z-pinch radial compression without and with current stepping",
            "current, voltage, streak photography, and radial magnetic-field mapping are used",
            "the thesis extends an energy-balance slug model with shock jump equations, EOS, Saha ionization, and a gamma-varying closure",
        ],
        "not_authority_for": [
            "DPF geometry or whole-shot DPF validation",
            "accepted DPF startup BVP",
        ],
        "target_extraction_required": [
            "OCR cleanup and page-image review",
            "UMCSZP and MLCCSZP circuit parameters",
            "current/voltage waveforms",
            "radial trajectory and current-density maps",
            "gamma-varying model equations with units and assumptions",
        ],
    },
    {
        "source_id": "serban_1995_anode_geometry_focus_characteristics",
        "path": "/Users/anthonyzamora/Downloads/A SerbanPhD1995.pdf",
        "sha256": (
            "5a19c05d03b4daf92dc6cdbcb53aecbd07a52db9939db82ff3f10136321fbdf1"
        ),
        "title": "Anode Geometry and Focus Characteristics",
        "author_or_lead": "Adrian Serban",
        "document_type": "phd_thesis",
        "pages": 271,
        "text_status": "pdftotext_full_extract_available",
        "primary_uses": [
            "anode_geometry_and_sheath_velocity_target_candidate",
            "pinch_lifetime_dimension_scaling_candidate",
            "plasma_impedance_and_focus_regime_context",
            "soft_xray_and_neutron_diagnostic_target_candidate",
        ],
        "useful_gate_ids": ["FP-5", "FP-8", "FP-11", "FP-12", "FP-15"],
        "candidate_facts": [
            "neutron-optimized devices are discussed around 10 cm per microsecond axial sheath velocity or less",
            "a 3 kJ stepped-down composite-anode plasma focus reached axial sheath speeds up to about 15 cm per microsecond",
            "optimum composite-anode operation increased neutron output by about 70 percent in the reported experiment",
        ],
        "not_authority_for": [
            "same-scope PF-1000/Akel validation",
            "accepted scaling laws without target extraction and review",
        ],
        "target_extraction_required": [
            "anode geometry tables",
            "sheath velocity measurements",
            "shadowgraph-derived pinch dimensions and lifetime",
            "neutron/hard-xray/soft-xray timing",
            "electron-temperature filter-ratio results",
        ],
    },
    {
        "source_id": "rafique_2000_deuterium_pf_compression_radiation",
        "path": "/Users/anthonyzamora/Downloads/MSR PhD thesis.pdf",
        "sha256": (
            "1eb27545f8fbaa8798278109af2a1242eb655209617270db5b832cf6278507f5"
        ),
        "title": "Compression Dynamics and Radiation Emission from a Deuterium Plasma Focus",
        "author_or_lead": "Muhammad Shahid Rafique",
        "document_type": "phd_thesis",
        "pages": 303,
        "text_status": "pdftotext_full_extract_available",
        "primary_uses": [
            "deuteron_spectrum_and_beam_target_target_candidate",
            "neutron_anisotropy_and_spectrum_context",
            "pinch_lifetime_instability_growth_candidate",
            "shadowgraph_spatial_target_candidate",
        ],
        "useful_gate_ids": ["FP-11", "FP-12", "FP-13", "FP-15"],
        "candidate_facts": [
            "deuteron spectra from 80 keV to 250 keV are correlated with total neutron yield",
            "average neutron energies are reported as about 2.48 MeV radial and 3 MeV axial",
            "neutron anisotropy averaged about 1.45 and higher-yield shots showed stronger anisotropy",
        ],
        "not_authority_for": [
            "accepted PF-1000/Akel neutron mechanism",
            "accepted beam-target closure without digitized spectra and detector UQ",
        ],
        "target_extraction_required": [
            "magnetic spectrometer geometry",
            "deuteron spectra",
            "neutron energy and anisotropy plots",
            "shadowgraph pinch radius/lifetime data",
            "soft-xray filter-ratio electron temperature traces",
        ],
    },
    {
        "source_id": "verma_2010_miniature_repetitive_pf_neutron_source",
        "path": "/Users/anthonyzamora/Downloads/PhD2010VermaRishi.pdf",
        "sha256": (
            "78b15cba0c57936cdfd24d2a8dc697abaff34f778a0f0a69ac741b80802536a5"
        ),
        "title": "Construction and Optimization of Low Energy (<240J) Miniature Repetitive Plasma Focus Neutron Source",
        "author_or_lead": "Rishi Verma",
        "document_type": "phd_thesis",
        "pages": 290,
        "text_status": "pdftotext_full_extract_available",
        "primary_uses": [
            "miniature_repetitive_pf_deck_candidate",
            "repetition_rate_and_electrode_aging_context",
            "cathode_structure_and_anode_optimization_context",
            "neutron_scaling_baseline_context",
        ],
        "useful_gate_ids": ["FP-8", "FP-10", "FP-12", "FP-15"],
        "candidate_facts": [
            "FMPF-1, FMPF-2, and FMPF-3 operate below 240 J",
            "FMPF-1 is reported at about 1.15e6 neutrons per shot at 230 J, 80 kA, and 5.5 mbar D2",
            "FMPF-3 is reported near 1.4e7 neutrons per second at 10 Hz and 200 J, 90 kA, 5.5 mbar D2",
        ],
        "not_authority_for": [
            "whole-shot single-shot DPF validation",
            "accepted electrode-ablation closure without erosion target extraction",
        ],
        "target_extraction_required": [
            "FMPF electrical and electrode tables",
            "repetition-rate yield tables",
            "shot-to-shot stability data",
            "electrode erosion and insulator aging observations",
            "current and neutron timing traces",
        ],
    },
    {
        "source_id": "avaria_2022_bayesian_sheath_diagnostics",
        "path": "/Users/anthonyzamora/Downloads/s41598-022-19764-7.pdf",
        "sha256": (
            "9ff0186062bd335802e1aa5e204e040182cbee36a04b00c3c2832c2913b6cda4"
        ),
        "title": "Bayesian inference of spectrometric data and validation with numerical simulations of plasma sheath diagnostics of a plasma focus discharge",
        "author_or_lead": "Avaria et al.",
        "document_type": "peer_reviewed_article",
        "pages": 12,
        "text_status": "pdftotext_full_extract_available",
        "primary_uses": [
            "rundown_sheath_density_target_candidate",
            "spectroscopic_diagnostic_uq_method_candidate",
            "startup_rundown_bvp_observable_candidate",
            "bayesian_target_extraction_method_reference",
        ],
        "useful_gate_ids": ["FP-5", "FP-11", "FP-13"],
        "candidate_facts": [
            "400 J hydrogen plasma focus rundown density is inferred from Stark-broadened H-alpha spectra",
            "Bayesian posterior processing is used for electron-density inference",
            "reported sheath estimates include roughly 4-20 eV temperature and 62.5 km/s velocity",
        ],
        "not_authority_for": [
            "PF-1000/Akel same-scope density validation",
            "accepted CShock validation of DPF-Unified",
        ],
        "target_extraction_required": [
            "spectrometer geometry and timing",
            "density profiles with uncertainty",
            "sheath position and velocity series",
            "Bayesian posterior settings",
            "CShock comparison data for method-only baseline",
        ],
    },
)


MAY16_KR_PROMOTION_LEDGER = {
    "markdown": "docs/USER_VALIDATED_THESES_KR_PROMOTION_2026_05_16.md",
    "json": "docs/USER_VALIDATED_THESES_KR_PROMOTION_2026_05_16.json",
    "promoted_count": 7,
    "skipped_existing_count": 0,
    "failed_count": 0,
}

MAY16_KR_RECORDS = {
    "arwinder_2015_comparative_pf_machines": {
        "markdown": "KnowledgeReference/comparative-study-of-plasma-focus-machines-2c7a8f4b.md",
        "json": "KnowledgeReference/comparative-study-of-plasma-focus-machines-2c7a8f4b.json",
        "status": "text_parity_extracted_review_needed",
    },
    "talebitaher_2012_nx2_coded_aperture_imaging": {
        "markdown": "KnowledgeReference/coded-aperture-imaging-of-nuclear-fusion-in-the-plasma-focus-device-9b79429f.md",
        "json": "KnowledgeReference/coded-aperture-imaging-of-nuclear-fusion-in-the-plasma-focus-device-9b79429f.json",
        "status": "text_parity_extracted_review_needed",
    },
    "saw_1990_current_stepped_z_pinch": {
        "markdown": "KnowledgeReference/experimental-studies-of-a-current-stepped-z-pinch-ad6e93b2.md",
        "json": "KnowledgeReference/experimental-studies-of-a-current-stepped-z-pinch-ad6e93b2.json",
        "status": "ocr_text_extracted_review_needed",
    },
    "serban_1995_anode_geometry_focus_characteristics": {
        "markdown": "KnowledgeReference/anode-geometry-and-focus-characteristics-5a19c05d.md",
        "json": "KnowledgeReference/anode-geometry-and-focus-characteristics-5a19c05d.json",
        "status": "text_parity_extracted_review_needed",
    },
    "rafique_2000_deuterium_pf_compression_radiation": {
        "markdown": "KnowledgeReference/compression-dynamics-and-radiation-emission-from-a-deuterium-plasma-focus-1eb27545.md",
        "json": "KnowledgeReference/compression-dynamics-and-radiation-emission-from-a-deuterium-plasma-focus-1eb27545.json",
        "status": "text_parity_extracted_review_needed",
    },
    "verma_2010_miniature_repetitive_pf_neutron_source": {
        "markdown": "KnowledgeReference/construction-and-optimization-of-low-energy-240j-miniature-repetitive-plasma-focus-neutron-78b15cba.md",
        "json": "KnowledgeReference/construction-and-optimization-of-low-energy-240j-miniature-repetitive-plasma-focus-neutron-78b15cba.json",
        "status": "text_parity_extracted_review_needed",
    },
    "avaria_2022_bayesian_sheath_diagnostics": {
        "markdown": "KnowledgeReference/bayesian-inference-of-spectrometric-data-and-validation-with-numerical-simulations-of-plas-9ff01860.md",
        "json": "KnowledgeReference/bayesian-inference-of-spectrometric-data-and-validation-with-numerical-simulations-of-plas-9ff01860.json",
        "status": "text_parity_extracted_review_needed",
    },
}


def may16_validated_thesis_source_targets() -> dict[str, object]:
    """Return non-promoting packets from the May 16 verified thesis/PDF batch."""

    document_targets = {
        str(row["source_id"]): _may16_document_target(row)
        for row in MAY16_VALIDATED_THESES
    }
    return {
        "batch_id": "user_validated_thesis_pdf_batch_2026_05_16",
        "source_status": "all_seven_user_verified_validated_documents_promoted_to_knowledge_reference",
        "source_ingestion_ledger": MAY16_KR_PROMOTION_LEDGER,
        "document_count": len(MAY16_VALIDATED_THESES),
        "accepted_for_whole_shot_first_principles": False,
        "document_targets": document_targets,
        "gate_coverage_candidates": {
            "FP-5_startup_bvp": [
                "saw_1990_current_stepped_z_pinch",
                "serban_1995_anode_geometry_focus_characteristics",
                "avaria_2022_bayesian_sheath_diagnostics",
            ],
            "FP-6_power_port": ["saw_1990_current_stepped_z_pinch"],
            "FP-8_physics_closure": [
                "saw_1990_current_stepped_z_pinch",
                "serban_1995_anode_geometry_focus_characteristics",
                "verma_2010_miniature_repetitive_pf_neutron_source",
            ],
            "FP-10_waveform_phase": [
                "arwinder_2015_comparative_pf_machines",
                "verma_2010_miniature_repetitive_pf_neutron_source",
            ],
            "FP-11_spatial_field_temperature": [
                "talebitaher_2012_nx2_coded_aperture_imaging",
                "serban_1995_anode_geometry_focus_characteristics",
                "rafique_2000_deuterium_pf_compression_radiation",
                "avaria_2022_bayesian_sheath_diagnostics",
            ],
            "FP-12_neutron_authority": [
                "talebitaher_2012_nx2_coded_aperture_imaging",
                "serban_1995_anode_geometry_focus_characteristics",
                "rafique_2000_deuterium_pf_compression_radiation",
                "verma_2010_miniature_repetitive_pf_neutron_source",
            ],
            "FP-13_comparator_uq": [
                "talebitaher_2012_nx2_coded_aperture_imaging",
                "rafique_2000_deuterium_pf_compression_radiation",
                "avaria_2022_bayesian_sheath_diagnostics",
            ],
            "FP-15_generalization": [
                "arwinder_2015_comparative_pf_machines",
                "talebitaher_2012_nx2_coded_aperture_imaging",
                "serban_1995_anode_geometry_focus_characteristics",
                "rafique_2000_deuterium_pf_compression_radiation",
                "verma_2010_miniature_repetitive_pf_neutron_source",
            ],
        },
        "what_it_closes": [
            "verified local source-candidate availability for several blocker families",
            "KnowledgeReference text-or-OCR source promotion with parity checks",
            "OCR availability for the scanned Saw current-stepped Z-pinch thesis",
        ],
        "what_it_does_not_close": [
            "typed target extraction with units and uncertainty",
            "same-scope PF-1000/Akel evidence",
            "mechanism-separated neutron authority",
            "detector-response and propagated UQ certificate",
            "accepted whole-shot first-principles readiness",
        ],
        "next_required_actions": [
            "extract typed tables, figures, equations, detector geometries, and uncertainty packets",
            "separate reduced-model Lee/GV/baseline material from active first-principles closure",
            "bind only reviewed same-scope targets into comparator and certificate gates",
        ],
    }


def _may16_document_target(row: dict[str, object]) -> dict[str, object]:
    kr_record = MAY16_KR_RECORDS[str(row["source_id"])]
    return {
        "path": row["path"],
        "sha256": row["sha256"],
        "knowledge_reference": kr_record,
        "title": row["title"],
        "author_or_lead": row["author_or_lead"],
        "document_type": row["document_type"],
        "pages": row["pages"],
        "text_status": row["text_status"],
        "ocr_artifacts": list(row.get("ocr_artifacts", ())),
        "source_status": "user_verified_validated_knowledge_reference_promoted",
        "accepted_for_validation": False,
        "accepted_for_whole_shot_first_principles": False,
        "primary_uses": list(row["primary_uses"]),
        "useful_gate_ids": list(row["useful_gate_ids"]),
        "candidate_facts": list(row["candidate_facts"]),
        "not_authority_for": list(row["not_authority_for"]),
        "target_extraction_required": list(row["target_extraction_required"]),
        "promotion_rule": (
            "requires typed target extraction with units, uncertainty, review "
            "status, and same-scope binding before comparator or certificate use"
        ),
    }


def may15_user_validated_source_targets() -> dict[str, object]:
    """Return non-promoting source packets from the May 15 validated PDF batch."""

    return {
        "batch_id": "user_validated_pdf_batch_2026_05_15",
        "source_status": "all_eight_user_verified_validated_research_sources",
        "accepted_for_whole_shot_first_principles": False,
        "source_ingestion_ledger": {
            "markdown": "docs/USER_VALIDATED_PDF_KR_PROMOTION_2026_05_15.md",
            "json": "docs/USER_VALIDATED_PDF_KR_PROMOTION_2026_05_15.json",
            "promoted_count": 6,
            "already_represented_count": 2,
            "parity_failed_count": 0,
        },
        "device_deck_targets": {
            "ir_mpf_100_salehizadeh_2012": _ir_mpf_100_deck_target(),
            "compact_chinese_dpf_2018": _compact_chinese_dpf_deck_target(),
            "willenborg_hendricks_1977_startup_design": (
                _willenborg_hendricks_startup_design_target()
            ),
        },
        "architecture_blocker_targets": {
            "sandia_alegra_hedp_2009": _alegra_hedp_architecture_target(),
            "gribkov_applications_2015": _gribkov_applications_target(),
        },
        "method_reference_targets": {
            "arnab_fluid_plasma_text": {
                "source": "KnowledgeReference/the-physics-of-fluids-and-plasmas-eef02f49.md",
                "role": "method_reference_only",
                "accepted_for_dpf_targets": False,
                "use": [
                    "Boltzmann/Vlasov hierarchy review",
                    "fluid moment and MHD derivation checks",
                    "transport and collision notation review",
                ],
            }
        },
        "next_required_actions": [
            "bind selected device decks into first_principles decks with source hashes",
            "extract figure and table targets with units and uncertainties",
            "review startup BVP channels against breakdown and sheath-liftoff requirements",
            "bind ALEGRA and Gribkov blocker facts into dimensionality and neutron-authority gates",
            "keep all extracted values non-accepting until independent engineering review",
        ],
    }


def gv_verified_shot_targets() -> dict[str, object]:
    """Return non-promoting packets from the verified local GV shot bundle."""

    return {
        "batch_id": "gv_verified_local_shot_bundle_2026_05_16",
        "source_status": "user_verified_local_download_not_knowledge_reference_promoted",
        "root": GV_ROOT,
        "model_document": {
            "path": f"{GV_ROOT}/Resistive Gratton-Vargas Model.pdf",
            "sha256": (
                "4fca85df65d3ea088b97528fb8c2147f1be2c514937e58563cd649f4103b59cd"
            ),
            "role": "reduced_model_usage_document_not_first_principles_authority",
        },
        "accepted_for_whole_shot_first_principles": False,
        "reduced_model_output_columns": GV_REDUCED_MODEL_OUTPUT_COLUMNS,
        "shot_count": len(GV_VERIFIED_SHOTS),
        "shot_targets": {
            str(row["shot_id"]): _gv_shot_target(row) for row in GV_VERIFIED_SHOTS
        },
        "what_it_closes": [
            "machine_geometry_candidate",
            "lumped_circuit_candidate",
            "fill_pressure_candidate",
            "measured_current_waveform_candidate",
            "reduced_model_current_baseline_candidate",
        ],
        "what_it_does_not_close": [
            "first_principles_startup_bvp",
            "preionization_or_surface_flashover_state",
            "same_scope_spatial_density_field_temperature_history",
            "mechanism_separated_neutron_history",
            "detector_response_and_uncertainty",
            "accepted_comparator_uq_matrix",
            "first_principles_validation_certificate",
        ],
        "next_required_actions": [
            "promote raw GV artifacts or verified extracts into KnowledgeReference before accepted-source use",
            "extract workbook current waveforms into typed target packets with units and point counts",
            "separate experimental waveform targets from GV reduced-model output columns",
            "bind selected shots as second-scope engineering decks only",
            "keep GV executable and reduced-model inductance trajectory out of first-principles authority",
        ],
    }


def _gv_shot_target(row: dict[str, object]) -> dict[str, object]:
    return {
        "device": row["device"],
        "shot_note": row["shot_note"],
        "accepted_for_validation": False,
        "files": {
            "input_deck": _gv_file(row["input_file"], row["input_sha256"]),
            "reduced_model_output": _gv_file(row["txt_file"], row["txt_sha256"]),
            "workbook": _gv_file(row["xlsx_file"], row["xlsx_sha256"]),
        },
        "geometry_mm": row["geometry_mm"],
        "circuit": row["circuit"],
        "gas": row["gas"],
        "experimental_waveform": {
            "status": "user_verified_workbook_candidate_not_comparator_bound",
            "columns": row["waveform_columns"],
            "note": row["workbook_note"],
        },
        "gv_baseline": {
            "status": "reduced_model_comparison_only",
            "txt_columns": GV_REDUCED_MODEL_OUTPUT_COLUMNS,
            "accepted_as_first_principles_closure": False,
        },
        "missing_for_first_principles_acceptance": [
            "startup_bvp",
            "spatial_density_field_temperature_history",
            "neutron_mechanism_separation",
            "detector_response",
            "uncertainty_and_independent_review",
        ],
    }


def _gv_file(name: object, sha256: object) -> dict[str, object]:
    return {
        "path": f"{GV_ROOT}/{name}",
        "sha256": str(sha256),
    }


def _ir_mpf_100_deck_target() -> dict[str, object]:
    source = "KnowledgeReference/original-research-f7894f85.md"
    return {
        "source": source,
        "source_sha256": (
            "f7894f85fd4d1826a5d98933453bd09664e260d46a2c9fedc4ce79491d2be4ad"
        ),
        "source_lines": {
            "abstract_deck": "30-45",
            "bank_and_current": "108-160",
            "geometry": "163-215,244-257",
            "sheath_design_timing": "200-215",
            "diagnostics_and_neutron_measurement": "269-288",
            "full_bank_projection": "300-323",
            "summary_table": "365-394",
        },
        "device": "IR-MPF-100",
        "role": "second_scope_device_deck_target",
        "accepted_for_validation": False,
        "circuit": {
            "capacitor_count": 24,
            "capacitance_each_F": 6.0e-6,
            "capacitance_F": 144.0e-6,
            "maximum_voltage_V": 40.0e3,
            "maximum_stored_energy_J": 115.0e3,
            "theoretical_peak_current_A": 1.224e6,
            "total_inductance_H": 120.0e-9,
            "capacitor_bank_inductance_H_approx": 4.0e-9,
            "spark_gap_inductance_H_approx": 40.0e-9,
            "coaxial_cable_inductance_H_approx": 60.0e-9,
            "other_connections_inductance_H_approx": 16.0e-9,
            "design_resistance_ohm": 5.0e-3,
            "design_period_s": 26.0e-6,
        },
        "geometry": {
            "anode_radius_m": 6.25e-2,
            "anode_length_m": 2.2e-1,
            "cathode_radius_m": 1.02e-1,
            "cathode_rod_count": 12,
            "cathode_rod_diameter_m": 1.2e-2,
            "cathode_rod_length_m": 2.2e-1,
            "insulator_length_m": 5.0e-2,
            "cathode_to_anode_radius_ratio": 1.63,
        },
        "gas": {
            "design_pressure_torr": 7.7,
            "design_pressure_Pa": 7.7 * TORR_TO_PA,
            "measured_shot_pressure_torr": 1.9,
            "measured_shot_pressure_Pa": 1.9 * TORR_TO_PA,
        },
        "diagnostics": [
            "ne102_scintillator",
            "rogowski_coil",
            "integrator_probe",
            "current_probe",
            "voltage_probe",
            "hard_xray",
            "current_derivative",
            "silver_activation_neutron_counter",
        ],
        "diagnostic_geometry": {
            "silver_activation_counter_distance_from_anode_top_m": 1.30,
        },
        "design_formula_context_not_closure": {
            "speed_factor_kA_per_cm_per_sqrt_torr": 50.0,
            "axial_velocity_cm_per_us": 4.037,
            "radial_velocity_cm_per_us": 6.071,
            "radial_time_s": 1.029e-6,
            "axial_time_s": 5.449e-6,
            "accepted_as_active_first_principles_closure": False,
        },
        "neutron_targets": {
            "preliminary_yield_neutrons_per_shot_at_65kJ": 1.0e9,
            "yield_neutrons_per_shot_at_29kJ_1p9torr": 1.5e9,
            "double_pinch_observed": True,
            "double_pinch_argon_voltage_V": 20.0e3,
            "double_pinch_argon_pressure_torr": 0.3,
            "double_pinch_delay_s_approx": 2.5e-6,
            "projected_full_bank_dd_yield_neutrons_per_shot_min": 1.0e10,
            "projected_full_bank_dt_yield_neutrons_per_shot": 1.0e12,
        },
        "missing_for_first_principles_acceptance": [
            "complete_startup_bvp",
            "measured_current_waveform_digitization",
            "per_point_waveform_uncertainty",
            "mechanism_separated_neutron_history",
            "detector_response_and_uq",
        ],
    }


def _compact_chinese_dpf_deck_target() -> dict[str, object]:
    source = "KnowledgeReference/high-power-laser-and-particle-beams-d1758d55.md"
    return {
        "source": source,
        "source_sha256": (
            "d1758d55ea9a32f6edb17107a86b033d8078cad337f0531ca10f18190fb220b5"
        ),
        "source_lines": {
            "abstract_targets": "38-41",
            "bank_and_current": "66-91",
            "geometry": "103-147,161-176",
            "simulation_and_measurement": "180-200,210-232",
            "english_abstract": "291-295",
        },
        "device": "compact Mather-type DPF neutron source",
        "role": "second_scope_device_deck_target",
        "accepted_for_validation": False,
        "circuit": {
            "capacitor_count": 4,
            "capacitance_each_F": 10.0e-6,
            "capacitance_total_F": 40.0e-6,
            "charging_voltage_range_V": [10.0e3, 20.0e3],
            "delivered_current_A_approx": 400.0e3,
        },
        "geometry": {
            "anode_radius_m": 17.0e-3,
            "outer_electrode_inner_radius_m": 40.0e-3,
            "radius_ratio_outer_inner": 2.3,
            "inner_electrode_length_m": 15.0e-2,
            "outer_electrode_length_m": 16.0e-2,
            "cathode_rod_count": 8,
            "cathode_rod_diameter_m": 8.0e-3,
            "cathode_rod_circle_diameter_m": 88.0e-3,
            "knife_edge_gap_m_approx": 1.0e-3,
            "insulator_inner_diameter_m": 36.0e-3,
            "insulator_outer_diameter_m": 46.0e-3,
            "insulator_height_m": 40.0e-3,
            "insulator_exposed_length_m": 45.0e-3,
            "air_side_withstand_voltage_V": 25.0e3,
        },
        "diagnostics": {
            "current_probe": "Rogowski coil",
            "neutron_detector": "photomultiplier",
            "neutron_detector_angle_deg": 90.0,
            "neutron_detector_distance_m_range": [2.0, 3.0],
            "tof_xray_speed_m_s": 3.0e8,
            "tof_2p45MeV_neutron_speed_m_s": 2.16e7,
            "tof_separation_s_per_m_approx": 43.0e-9,
            "observed_separation_s": [115.0e-9, 107.0e-9, 96.0e-9],
            "observed_distances_m": [2.5, 2.4, 2.15],
            "mean_observed_separation_s_per_m": 45.1e-9,
        },
        "operating_targets": {
            "optimum_pressure_Pa_range": [550.0, 600.0],
            "reported_pressure_Pa": 580.0,
            "focus_time_s": 1.8e-6,
            "charging_voltage_yield_threshold_V": 19.0e3,
            "average_dd_neutron_yield_per_pulse_min": 5.0e8,
            "neutron_pulse_fwhm_s": 40.0e-9,
            "neutron_pulse_fwhm_uncertainty_s": 5.0e-9,
            "simulated_pressure_yield_peak_pressure_Pa_approx": 1.0e3,
            "simulated_pressure_yield_peak_neutrons_per_pulse": 6.45e8,
        },
        "missing_for_first_principles_acceptance": [
            "visual_table_review",
            "translation_review",
            "measured_waveform_digitization",
            "axis_calibrated_pressure_yield_curve",
            "detector_response_and_uq",
        ],
    }


def _willenborg_hendricks_startup_design_target() -> dict[str, object]:
    source = (
        "KnowledgeReference/"
        "design-and-construction-of-a-dense-plasma-focus-device-12205ba4.md"
    )
    return {
        "source": source,
        "source_sha256": (
            "12205ba4bb0d1edc11b069dda4e0e084b89597a8f14ff61c3a65e0b712926a75"
        ),
        "source_lines": {
            "breakdown_and_sheath_start": "506-514,579-594",
            "symmetry_and_liftoff": "625-653",
            "mather_envelope": "1196-1209",
            "electrodes": "1382-1496",
            "insulator": "1503-1520,1545-1629",
            "timing_to_current_peak": "762-773,1365-1373,1857-1901",
            "bank": "1697-1702,1825-1830,1884-1901",
            "switching": "1961-2000,2287-2316,2372-2421",
            "gas_pressure": "2617-2639,2716-2719",
            "diagnostics": "2720-2790,2870-2924,2930-2964,3089-3133",
            "insulator_conditioning": "1545-1640,3372-3380",
            "operation_gates": "3367-3439",
            "remaining_unaccepted": "3383-3390,3446-3455",
        },
        "device": "Willenborg/Hendricks dense plasma focus device",
        "role": "startup_breakdown_and_diagnostic_design_target",
        "accepted_for_validation": False,
        "circuit": {
            "capacitance_F": 43.5e-6,
            "capacitance_status": "inferred_from_three_14p5uF_capacitors",
            "stored_energy_J_at_20kV": 8.7e3,
            "capacitor_count": 3,
            "capacitance_each_F": 14.5e-6,
            "rated_voltage_V": 20.0e3,
            "operated_voltage_V_range": [9.0e3, 19.0e3],
            "capacitor_internal_inductance_H_approx": 89.0e-9,
            "bank_ringing_frequency_Hz": 143.0e3,
            "bank_quarter_cycle_s_approx": 1.8e-6,
            "total_system_inductance_H_approx": 100.0e-9,
            "system_quarter_cycle_s_approx": 3.3e-6,
            "average_device_impedance_ohm_approx": 0.03,
        },
        "geometry": {
            "mather_inner_radius_m_approx": 5.0e-2,
            "mather_outer_radius_m_approx": 1.0e-1,
            "mather_length_m_range": [15.0e-2, 30.0e-2],
            "center_electrode_finished_diameter_m": 1.78 * 0.0254,
            "center_electrode_length_m_approx": 9.0 * 0.0254,
            "outer_electrode_rod_count": 8,
            "outer_electrode_rod_diameter_m": 0.5 * 0.0254,
            "outer_electrode_angular_spacing_deg": 45.0,
            "annular_spacing_m": 1.13 * 0.0254,
            "final_ceramic_insulator_wall_m": 0.312 * 0.0254,
            "final_ceramic_insulator_length_m": 2.93 * 0.0254,
            "final_ceramic_insulator_collar_thickness_m": 0.25 * 0.0254,
            "final_ceramic_insulator_collar_width_m": 4.0 * 0.0254,
        },
        "gas": {
            "working_pressure_torr_range": [0.1, 10.0],
            "working_pressure_Pa_range": [0.1 * TORR_TO_PA, 10.0 * TORR_TO_PA],
            "pumpdown_pressure_torr": 1.0e-6,
            "pumpdown_time_s": 8.0 * 60.0,
            "hastings_gauge_range_torr": [0.01, 20.0],
        },
        "startup_constraints": {
            "breakdown_along_insulator_determines_start": True,
            "azimuthal_symmetry_required_for_strong_focus": True,
            "conditioned_insulator_required": True,
            "conditioning_shots_max": 15,
            "conditioning_voltage_V_range": [14.0e3, 15.0e3],
            "conditioning_pressure_torr": 1.0,
            "conditioning_xray_voltage_pulse_threshold_V": 0.5,
            "focus_should_coincide_with_bank_current_peak": True,
            "sheath_velocity_m_s_approx": 5.0e6 * 0.0254,
            "gun_length_m_approx": 9.0 * 0.0254,
            "sheath_travel_time_s_approx": 1.8e-6,
            "breakdown_delay_s_approx": 1.0e-6,
            "focus_delay_s_approx": 2.8e-6,
            "current_fraction_at_focus_approx": 0.97,
            "appreciable_xrays_voltage_threshold_V": 13.0e3,
            "pressure_for_xray_operation_torr_range": [0.5, 5.0],
            "pressure_weak_xray_low_torr": 0.75,
            "pressure_weak_xray_high_torr": 1.5,
            "xray_optimum_pressure_torr_approx": 1.0,
        },
        "switch": {
            "type": "single_trigatron_air_dielectric_spark_gap",
            "main_gap_m": 0.39 * 0.0254,
            "static_breakdown_voltage_V_approx": 20.0e3,
            "trigger_pulse_voltage_V": 14.0e3,
            "jitter_s": 20.0e-9,
            "trigger_delay_s_approx": 200.0e-9,
        },
        "diagnostics": [
            "capacitive_high_voltage_divider",
            "air_core_rogowski_loop",
            "xray_detector",
            "current_voltage_xray_timing_comparison",
        ],
        "diagnostic_targets": {
            "signal_limit_V_peak_to_peak": 100.0,
            "voltage_probe_risetime_s_max": 10.0e-9,
            "voltage_probe_divider_ratio": 800.0,
            "voltage_probe_termination_ohm": 51.0,
            "voltage_probe_measures_voltage_V_min": 40.0e3,
            "xray_detector_model": "100-PIN-125",
            "xray_detector_area_m2": 100.0e-6,
            "xray_detector_depth_m": 125.0e-6,
            "xray_detector_bias_V": 200.0,
            "xray_detector_risetime_s_max": 5.0e-9,
            "non_pinch_xray_signal_V_max": 10.0e-3,
            "strong_pinch_xray_signal_V_min": 30.0,
            "xray_pulse_duration_s_range": [100.0e-9, 300.0e-9],
        },
        "missing_for_first_principles_acceptance": [
            "surface_flashover_equations",
            "secondary_emission_or_material_model",
            "preionization_state",
            "measured_breakdown_to_liftoff_state",
            "reviewed_modern_device_scope",
        ],
    }


def _alegra_hedp_architecture_target() -> dict[str, object]:
    source = "KnowledgeReference/sand2009-6373-b93aec67.md"
    return {
        "source": source,
        "source_sha256": (
            "b93aec67a34ed9cd63176dc3fdf404df4aa29ff16cf8807eb68568ed1dbc0f9c"
        ),
        "source_lines": {
            "pic_startup_import_and_3d_launch": "151-163",
            "dpf_operating_range": "317-325",
            "nonthermal_neutron_limit": "346-352,394-397,511-557",
            "three_d_need": "360-369,458-464",
            "code_physics_capabilities": "411-418",
            "eos_and_conductivity_context": "436-449",
            "circuit_and_lsp_coupling": "470-475,682-690",
            "benchmark_table_context": "640-679",
        },
        "role": "first_principles_architecture_blocker_source",
        "accepted_for_validation": False,
        "required_capabilities": [
            "2d_or_3d_ale_mhd",
            "hydrodynamics",
            "magnetics",
            "thermal_conduction",
            "radiation_physics",
            "multimaterial_multiphase_eos",
            "advanced_thermal_electrical_conductivity",
            "lumped_circuit_model",
            "neutron_yield_and_time_of_flight_diagnostics",
            "pic_to_mhd_startup_import",
        ],
        "blocker_facts": {
            "mhd_thermonuclear_neutron_yield_below_total_observed": True,
            "nonthermal_neutron_mechanisms_required": True,
            "two_d_cathode_bar_approximation_blocks_full_geometry_authority": True,
            "first_principles_startup_requires_pic_breakdown_import_or_bvp": True,
            "three_d_modeling_required_for_faithful_cathode_and_pinch_structure": True,
        },
        "numeric_context": {
            "dpf_voltage_range_V": [20.0e3, 50.0e3],
            "dpf_current_range_A": [0.6e6, 1.8e6],
            "reported_neutron_yield_max": 5.0e11,
            "bernard_long_stored_energy_J": 27.0e3,
            "bernard_short_stored_energy_J": 96.0e3,
            "tallboy_stored_energy_J": 270.0e3,
            "deuterium_eos_min_density_kg_m3": 0.01,
        },
        "missing_for_first_principles_acceptance": [
            "reviewed_pic_startup_import_payload",
            "same_scope_3d_device_evidence",
            "kinetic_nonthermal_neutron_history",
            "detector_response_and_uncertainty_packet",
        ],
    }


def _gribkov_applications_target() -> dict[str, object]:
    source = (
        "KnowledgeReference/"
        "open-access-proceedings-journal-of-physics-conference-series-ed196711.md"
    )
    return {
        "source": source,
        "source_sha256": (
            "ed1967114c762f608493bd4d049b627ed0d13165d435ed7d5c23efa92a93cc2a"
        ),
        "source_lines": {
            "current_abruption_and_plasma_diode": "93-141",
            "storage_and_matching_context": "152-190",
            "beam_material_interaction": "211-216",
            "pf1000_post_pinch_context": "391-395",
            "application_neutron_detection_context": "680-697",
            "large_dpf_mitl_and_telegraph_equations": "782-805",
            "large_device_current_and_neutron_context": "811-918",
        },
        "role": "mechanism_and_large_machine_constraint_source",
        "accepted_for_validation": False,
        "mechanism_context": {
            "current_abruption_ps_scale": True,
            "plasma_diode_transfers_energy_to_fast_particles": True,
            "fast_electrons_and_ions_can_substitute_discharge_current": True,
            "pf1000_fast_deuteron_count_approx": 1.0e18,
            "bank_to_fast_ion_beam_efficiency_approx": 0.10,
            "bad_target_density_path_product_claim": True,
        },
        "large_machine_constraints": {
            "mather_chamber_behaves_as_mitl_after_current_abruption": True,
            "simple_clr_circuit_is_insufficient_for_very_large_dpf": True,
            "mhd_plus_telegraph_equations_required_for_final_stage": True,
            "illustrative_output_current_A": 90.0e6,
            "illustrative_pinch_current_A": 20.0e6,
            "illustrative_14MeV_neutron_yield_exceeds": 1.0e17,
            "required_repetition_rate_cps_range": [3.0, 4.0],
        },
        "missing_for_first_principles_acceptance": [
            "primary_same_scope_shot_packet",
            "mechanism_separated_current_and_particle_history",
            "source_backed_mitl_boundary_model",
            "detector_response_and_uq",
        ],
    }


# ---------------------------------------------------------------------------
# S3.8 — PF-1000 / Akel same-scope source-packet hash registry
#
# Each entry maps a KnowledgeReference canonical path to the SHA-256 of the
# local markdown file.  The hash is computed at call time (fail-soft) so it
# reflects the on-disk content without embedding stale hashes in the source.
#
# Source authority:
#   KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md
#   WP-N7 spec §2.1 (Akel 2021 ingestion status and evidence inventory)
#   WP-N7 spec §3.2 (certificate channel evidence_packet_hashes)
#   handoff §S3.8 "required certificate channels — source packet hashes"
#
# These hashes are CANDIDATE evidence.  They are NOT accepted comparator
# targets and do NOT promote validation.  Every channel that can only be
# backed by text-supported scalars (scalar I_peak, scalar neutron yield,
# detector geometry text) is labeled ``candidate_comparator_only``.
# ---------------------------------------------------------------------------

#: KnowledgeReference paths for the PF-1000/Akel 16 kV, 1.05–1.2 Torr,
#: shot-12581 evidence scope.  Paths are relative to the repo root so the
#: hash function can resolve them against any checkout location.
PF1000_AKEL_KR_SOURCE_PATHS: tuple[dict[str, str], ...] = (
    {
        "source_id": "akel_2021_radiation_physics_chemistry",
        "path": (
            "KnowledgeReference/"
            "radiation-physics-and-chemistry-188-2021-109633.md"
        ),
        "scope": "pf1000_akel_16kv_1p2torr_deuterium_shot_12581",
        "evidence_type": "text_supported_scalar_candidate",
        "candidate_comparator_only": True,
        "blocking_channels": [
            "waveform_phase_packet_accepted",
            "spatial_field_temperature_packet_accepted",
        ],
        "candidate_channels": [
            "current_waveform_scalar_ipeak",
            "neutron_scalar_yield",
            "detector_geometry_text",
            "phase_timing_text",
        ],
    },
    {
        "source_id": "scholz_2006_pf1000_mega_joule",
        "path": "KnowledgeReference/scholz-2006-pf1000-mega-joule.md",
        "scope": "pf1000_full_energy_mj_scale",
        "evidence_type": "cross_scope_different_operating_point",
        "candidate_comparator_only": True,
        "scope_mismatch": True,
        "transfer_rule_required": True,
        "blocking_channels": [
            "neutron_timing_history",
            "neutron_anisotropy",
        ],
        "candidate_channels": [],
    },
    {
        "source_id": "scholz_gribkov_2007_pf1000_part2",
        "path": "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md",
        "scope": "pf1000_full_energy_1mj",
        "evidence_type": "cross_scope_inferred_temperatures",
        "candidate_comparator_only": True,
        "scope_mismatch": True,
        "transfer_rule_required": True,
        "blocking_channels": [
            "temperature_Ti_direct_measurement",
        ],
        "candidate_channels": [],
    },
    {
        "source_id": "zielinska_2011_interferometer",
        "path": (
            "KnowledgeReference/"
            "sixteenframe-interferometer-for-a-study-of-a-pinch-dynamics-in-pf1000-device-f8dc9d1b.md"
        ),
        "scope": "pf1000_different_shot_2p6hPa",
        "evidence_type": "cross_scope_density_imaging_different_shot",
        "candidate_comparator_only": True,
        "scope_mismatch": True,
        "transfer_rule_required": True,
        "blocking_channels": [
            "spatial_density_history",
        ],
        "candidate_channels": [],
    },
    {
        "source_id": "kubes_2020_closed_currents",
        "path": (
            "KnowledgeReference/"
            "characteristics-of-closed-currents-and-magnetic-fields-outside-the-dense-pinch-column-in-a-40d59f2d.md"
        ),
        "scope": "pf1000_scope_unconfirmed",
        "evidence_type": "scope_unconfirmed_magnetic_field_measurement",
        "candidate_comparator_only": True,
        "scope_mismatch": None,
        "transfer_rule_required": True,
        "blocking_channels": [
            "em_field_history",
        ],
        "candidate_channels": [],
    },
    {
        "source_id": "krauz_2012_current_sheath",
        "path": (
            "KnowledgeReference/"
            "experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md"
        ),
        "scope": "pf1000_scope_unconfirmed",
        "evidence_type": "current_sheath_structure_measurement",
        "candidate_comparator_only": True,
        "scope_mismatch": None,
        "transfer_rule_required": True,
        "blocking_channels": [
            "spatial_field_temperature_packet_accepted",
        ],
        "candidate_channels": [],
    },
)


def _sha256_of_path_soft(repo_root: Path, rel_path: str) -> str | None:
    """Return SHA-256 of a repo-relative file, or None if unreadable."""
    candidate = repo_root / rel_path
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_repo_root() -> Path:
    """Walk upward from this file to locate the dpf-unified repo root."""
    here = Path(__file__).resolve()
    # src/dpf/first_principles/source_targets.py → 3 parents = src/ → repo root
    return here.parents[3]


def pf1000_akel_source_packet_hashes(
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Return the S3.8 source-packet hash registry for the PF-1000/Akel scope.

    Each entry in the registry is a dict with:
    - ``source_id``: canonical identifier.
    - ``path``: KnowledgeReference-relative path (repo-relative).
    - ``sha256``: SHA-256 of the on-disk markdown file (``None`` if absent).
    - ``scope``: evidence scope tag.
    - ``candidate_comparator_only``: ``True`` for all channels in Sprint 3 —
      no acceptance is possible; comparator channels are labeled accordingly.
    - ``blocking_channels``: certificate channels this source CANNOT satisfy.
    - ``candidate_channels``: channels this source COULD support (text only).

    This registry is CANDIDATE evidence only.  No channel is accepted.
    No PF-1000/Akel validation certificate is emitted in Sprint 3.

    Source authority:
        ``docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/
        sprint_3/WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md`` §2.1–2.7, §3.1.
        ``docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md``
        §S3.8 required certificate channels — source packet hashes.
    """
    root = Path(repo_root) if repo_root is not None else _find_repo_root()
    entries: list[dict[str, object]] = []
    for spec in PF1000_AKEL_KR_SOURCE_PATHS:
        sha = _sha256_of_path_soft(root, str(spec["path"]))
        entry: dict[str, object] = {
            "source_id": spec["source_id"],
            "path": spec["path"],
            "sha256": sha,
            "scope": spec["scope"],
            "evidence_type": spec["evidence_type"],
            "candidate_comparator_only": spec["candidate_comparator_only"],
            "blocking_channels": list(spec["blocking_channels"]),  # type: ignore[arg-type]
            "candidate_channels": list(spec["candidate_channels"]),  # type: ignore[arg-type]
        }
        if "scope_mismatch" in spec:
            entry["scope_mismatch"] = spec["scope_mismatch"]
        if spec.get("transfer_rule_required"):
            entry["transfer_rule_required"] = True
        entries.append(entry)

    accepted_any = False  # always False in Sprint 3
    return {
        "packet_id": "pf1000_akel_source_packet_hashes_sprint3",
        "declared_scope": "pf1000_akel_16kv_1p2torr_deuterium_shot_12581",
        "sprint": "sprint_3",
        "status": "candidate_comparator_only_not_accepted",
        "source_entries": entries,
        "total_sources": len(entries),
        "accepted_any": accepted_any,
        "can_support_first_principles_acceptance": False,
        "can_support_validation_claims": False,
        "all_comparator_channels_labeled_candidate_comparator_only": True,
        "validation_blocked_reason": (
            "No same-scope digitized current waveform, density history, "
            "T_e, T_i, B_theta, or neutron spectrum/anisotropy exists for "
            "the Akel 16 kV, 1.05-1.2 Torr, shot-12581 scope in "
            "KnowledgeReference/. See WP-N7 §5 missing-parameters table."
        ),
        "source_references": [
            {
                "path": (
                    "docs/external_team_submissions/"
                    "2026_05_18_three_sprint_blocker_packet/sprint_3/"
                    "WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md"
                ),
                "lines": "1-170",
                "role": "certificate_source_packet_hash_requirement",
            },
            {
                "path": (
                    "docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md"
                ),
                "lines": "538-557",
                "role": "s3_8_required_certificate_channels",
            },
        ],
    }
