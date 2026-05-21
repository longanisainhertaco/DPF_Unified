"""Sprint 6 user-supplied source target extractions.

These packets cover the local PDFs supplied by the user on 2026-05-20.
They convert verified source availability into typed target records, while
keeping all runtime and first-principles acceptance flags false.

The Scholz et al. 2001 PF-1000 paper is now available as KR text and provides
direct PF-1000 hardware dimensions for a 24-rod large-electrode configuration.
It reduces geometry-source uncertainty, but it does not by itself certify the
runtime 3-D material masks because revision selection, wall thickness,
backplate geometry, and same-scope mask review remain open.

The Bruzzone and Bernal 2001 anomalous-resistivity paper was already promoted
from the Sprint 6 free-acquisition pass; the user-supplied file is an exact
SHA-256 duplicate and therefore is recorded here as duplicate verification
instead of a second KR record.

The later Scholz 2000, Herold 1989, Scholz 1999, Loarer 2007, Shakya
2015, Gribkov 2007, and Gribkov/Malaquias 2006 files were also classified
here.  They are useful source material, but only the direct PF-1000 hardware
sources reduce PF-1000 geometry-source uncertainty.  The foam-liner, tokamak
gas-balance, DMP applications review, and Lee-model comparison papers stay
context-only for the first-principles 3-D simulator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_EXTRACTION: Mapping[str, Any] = {
    "source_id": "scholz_2001_recent_progress_pf1000_hardware",
    "title": "Recent progress in 1 MJ Plasma-Focus research",
    "authors": (
        "Scholz, M.; Karpinski, L.; Paduch, M.; Tomaszewski, K.; "
        "Miklaszewski, R.; Szydlowski, A."
    ),
    "journal": "Nukleonika 46(1):35-39 (2001)",
    "source_pdf_sha256": (
        "d3e51f6c56f734e871f657f950486be441f75df9b75660e4524675738b002c75"
    ),
    "kr_markdown": (
        "KnowledgeReference/"
        "recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md"
    ),
    "kr_json": (
        "KnowledgeReference/"
        "recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.json"
    ),
    "scope_tag": "pf1000_2001_24_rod_large_electrode_hardware",
    "scope_caveat": (
        "PF-1000 hardware/diagnostic source for the 2001 24-rod large-electrode "
        "configuration; not an Akel 16 kV same-scope validation packet and not "
        "an accepted 3-D mask certificate."
    ),
    "kr_promotion_report": "docs/USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.md",
    "render_verification_status": "page_2_figure_1_rendered_for_geometry_labels",
    "render_artifact": (
        "docs/extractions/scholz_2001_recent_progress_render_evidence/"
        "pdf_p002_journal_p036-2.png"
    ),
    "render_artifact_sha256": (
        "4c9657f07a2a2caf6949e677f426d0ede55a3700238214e687c874b34ae60c84"
    ),
    "resolves_blockers": (
        "PF1000-BLK-004",
        "PF1000-BLK-015",
    ),
    "candidate_context_only": (
        "PF1000-BLK-009",
    ),
    "still_blocked": (
        "PF1000-BLK-010",
        "PF1000-BLK-016",
        "PF1000-BLK-017",
        "PF1000-BLK-018",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "cathode_rod_count": {
            "value": 24,
            "units": "count",
            "kr_lines": (90, 92),
            "resolves": ("PF1000-BLK-004",),
        },
        "cathode_rod_length_m": {
            "value": 0.600,
            "units": "m",
            "kr_lines": (90, 92),
            "resolves": ("PF1000-BLK-004",),
        },
        "cathode_rod_diameter_m": {
            "value": 0.032,
            "units": "m",
            "kr_lines": (90, 92),
            "resolves": (),
        },
        "outer_electrode_diameter_m": {
            "value": 0.400,
            "units": "m",
            "kr_lines": (91, 93),
            "resolves": (),
        },
        "outer_electrode_radius_m": {
            "value": 0.200,
            "units": "m",
            "kr_lines": (91, 93),
            "resolves": (),
        },
        "inner_electrode_material": {
            "value": "copper",
            "kr_lines": (93, 94),
            "resolves": (),
        },
        "inner_electrode_diameter_m": {
            "value": 0.244,
            "units": "m",
            "kr_lines": (93, 94),
            "resolves": (),
        },
        "inner_electrode_radius_m": {
            "value": 0.122,
            "units": "m",
            "kr_lines": (93, 94),
            "resolves": (),
        },
        "anode_end_face_hole_diameter_m": {
            "value": 0.030,
            "units": "m",
            "kr_lines": (94, 95),
            "resolves": (),
            "scope_note": (
                "Hardware end-face hole context only. This is not enough to "
                "accept the full hollow-anode bore radius/length runtime mask."
            ),
        },
        "anode_end_face_hole_radius_m": {
            "value": 0.015,
            "units": "m",
            "kr_lines": (94, 95),
            "resolves": (),
            "candidate_context_for": ("PF1000-BLK-009",),
        },
        "interelectrode_gap_m": {
            "value": 0.062,
            "units": "m",
            "kr_lines": (95, 96),
            "resolves": (),
        },
        "insulator_material": {
            "value": "alumina",
            "kr_lines": (96, 98),
            "resolves": (),
        },
        "insulator_outer_diameter_m": {
            "value": 0.229,
            "units": "m",
            "kr_lines": (96, 98),
            "resolves": ("PF1000-BLK-015",),
        },
        "insulator_outer_radius_m": {
            "value": 0.1145,
            "units": "m",
            "kr_lines": (96, 98),
            "resolves": ("PF1000-BLK-015",),
        },
        "insulator_length_m": {
            "value": 0.113,
            "units": "m",
            "kr_lines": (96, 98),
            "resolves": (),
        },
        "bank_module_count": {
            "value": 12,
            "units": "count",
            "kr_lines": (98, 100),
            "resolves": (),
        },
        "capacitors_per_module": {
            "value": 24,
            "units": "count",
            "kr_lines": (98, 100),
            "resolves": (),
        },
        "capacitor_voltage_kV": {
            "value": 50.0,
            "units": "kV",
            "kr_lines": (98, 100),
            "resolves": (),
        },
        "capacitance_per_capacitor_uF": {
            "value": 4.625,
            "units": "uF",
            "kr_lines": (98, 100),
            "resolves": (),
        },
        "charging_voltage_range_kV": {
            "value": (20.0, 40.0),
            "units": "kV",
            "kr_lines": (101, 103),
            "resolves": (),
        },
        "bank_energy_range_kJ": {
            "value": (266.0, 1064.0),
            "units": "kJ",
            "kr_lines": (101, 104),
            "resolves": (),
        },
        "quarter_discharge_time_us": {
            "value": 5.4,
            "units": "us",
            "kr_lines": (101, 104),
            "resolves": (),
        },
        "pin1_observation_offset_m": {
            "value": 0.020,
            "units": "m",
            "kr_lines": (116, 119),
            "resolves": (),
        },
        "pin2_pinhole_diameter_m": {
            "value": 100.0e-6,
            "units": "m",
            "kr_lines": (120, 122),
            "resolves": (),
        },
        "pin2_beryllium_filter_thickness_m": {
            "value": 20.0e-6,
            "units": "m",
            "kr_lines": (120, 122),
            "resolves": (),
        },
        "neutron_scintillator_distance_m": {
            "value": 15.0,
            "units": "m",
            "kr_lines": (149, 158),
            "resolves": (),
        },
        "good_shot_energy_kJ": {
            "value": 1070.0,
            "units": "kJ",
            "kr_lines": (166, 170),
            "resolves": (),
            "scope_note": "single good-shot context; not same-scope Akel comparator",
        },
        "good_shot_neutron_yield": {
            "value": 2.06e11,
            "units": "neutrons",
            "kr_lines": (166, 170),
            "resolves": (),
            "scope_note": "single good-shot context; not same-scope Akel comparator",
        },
        "reported_scaling_exponent": {
            "value": 3.3,
            "formula": "Y ~ Imax^3.3",
            "kr_lines": (245, 254),
            "resolves": (),
            "scope_note": "empirical scaling context only, not first-principles authority",
        },
        "no_yield_saturation_below_MA": {
            "value": 2.3,
            "units": "MA",
            "kr_lines": (251, 254),
            "resolves": (),
        },
    },
}


BRUZZONE_BERNAL_2001_DUPLICATE_VERIFICATION: Mapping[str, Any] = {
    "source_id": "bruzzone_bernal_2001_lhi_duplicate_verification",
    "title": (
        "The need of using anomalous resistivity due to Lower Hybrid "
        "Instabilities in plasma-magnetic field interfaces"
    ),
    "source_pdf_sha256": (
        "73668d0e98604959a6fcd3e20adfd5d55d757dfad943972a2b56a9595f927112"
    ),
    "user_supplied_path": (
        "/Users/anthonyzamora/Downloads/"
        "The_need_of_using_anomalous_resisti.pdf"
    ),
    "existing_kr_json": (
        "KnowledgeReference/"
        "the-need-of-using-anomalous-resistivity-due-to-lower-hybrid-"
        "instabilities-in-plasma-magnet-73668d0e.json"
    ),
    "existing_kr_markdown": (
        "KnowledgeReference/"
        "the-need-of-using-anomalous-resistivity-due-to-lower-hybrid-"
        "instabilities-in-plasma-magnet-73668d0e.md"
    ),
    "status": "exact_sha_duplicate_existing_kr_source",
    "resolves_blockers": (),
    "candidate_context_only": ("CLOSURE-BLK-ANOM-001",),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
}


SCHOLZ_2000_PF1000_DEVICE_EXTRACTION: Mapping[str, Any] = {
    "source_id": "scholz_2000_pf1000_device",
    "title": "PF-1000 device",
    "authors": "Scholz, M.; Miklaszewski, R.; Gribkov, V. A.; Mezzetti, F.",
    "journal": "Nukleonika 45(3):155-158 (2000)",
    "source_pdf_sha256": (
        "a2d6bc151ee1a3f5681c76ce00b3c60470d0c0386398b5f4033d14669279411c"
    ),
    "kr_markdown": "KnowledgeReference/pf-1000-device-a2d6bc15.md",
    "kr_json": "KnowledgeReference/pf-1000-device-a2d6bc15.json",
    "scope_tag": "pf1000_2000_facility_hardware_bank_diagnostics",
    "scope_caveat": (
        "Direct PF-1000 facility paper, but it describes an early hardware "
        "configuration and does not by itself select the Akel/Krauz/Gribkov "
        "runtime revision or provide a reviewed 3-D mask certificate."
    ),
    "kr_promotion_report": "docs/USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.md",
    "render_artifact": (
        "docs/extractions/scholz_2000_pf1000_device_render_evidence/"
        "pdf_p002_journal_p156.png"
    ),
    "render_artifact_sha256": (
        "eb6edfe94c7ce02bd0ee2adddd05df5aea353420dc026d642ef75bb457573140"
    ),
    "resolves_blockers": ("PF1000-BLK-004",),
    "candidate_context_only": (
        "PF1000 cathode-cage hardware category mismatch",
        "PF1000 bank/circuit source context",
        "PF1000 chamber source context",
    ),
    "still_blocked": (
        "PF1000-BLK-009",
        "PF1000-BLK-010",
        "PF1000-BLK-015",
        "PF1000-BLK-016",
        "PF1000-BLK-017",
        "PF1000-BLK-018",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "dpf_phase_sequence": {
            "value": (
                "initial gas breakdown/current-sheath formation; Lorentz "
                "rundown toward anode opening; radial collapse to pinch"
            ),
            "kr_lines": (20, 36),
            "resolves": (),
        },
        "breakdown_to_pinch_time_text": {
            "value": "few microseconds",
            "kr_lines": (29, 33),
            "resolves": (),
        },
        "pinch_duration_text": {
            "value": "few tens of nanoseconds",
            "kr_lines": (31, 33),
            "resolves": (),
        },
        "cathode_rod_count": {
            "value": 24,
            "units": "count",
            "kr_lines": (83, 89),
            "resolves": ("PF1000-BLK-004",),
        },
        "cathode_rod_diameter_m": {
            "value": 0.032,
            "units": "m",
            "kr_lines": (86, 88),
            "resolves": (),
        },
        "reported_oe_ce_dimension_ambiguous_m": {
            "value": (0.200, 0.1155),
            "units": "m",
            "kr_lines": (86, 90),
            "resolves": (),
            "scope_note": (
                "The text says diameters, while the stated annular spacing is "
                "consistent only with radius-like interpretation after "
                "accounting for rod radius; kept as ambiguous context."
            ),
        },
        "cathode_rod_length_m": {
            "value": 0.600,
            "units": "m",
            "kr_lines": (88, 90),
            "resolves": ("PF1000-BLK-004",),
        },
        "minimum_annular_spacing_m": {
            "value": 0.0685,
            "units": "m",
            "kr_lines": (88, 91),
            "resolves": (),
        },
        "insulator_material": {
            "value": "alumina",
            "kr_lines": (83, 86),
            "resolves": (),
        },
        "insulator_exposed_length_m": {
            "value": 0.113,
            "units": "m",
            "kr_lines": (129, 133),
            "resolves": (),
        },
        "chamber_material": {
            "value": "stainless steel",
            "kr_lines": (134, 136),
            "resolves": (),
        },
        "chamber_diameter_m": {
            "value": 1.400,
            "units": "m",
            "kr_lines": (134, 137),
            "resolves": (),
        },
        "chamber_length_m": {
            "value": 2.500,
            "units": "m",
            "kr_lines": (134, 137),
            "resolves": (),
        },
        "bank_energy_rating_kJ": {
            "value": 1200.0,
            "units": "kJ",
            "kr_lines": (138, 141),
            "resolves": (),
        },
        "bank_voltage_rating_kV": {
            "value": 40.0,
            "units": "kV",
            "kr_lines": (138, 141),
            "resolves": (),
        },
        "bank_module_count": {
            "value": 12,
            "units": "count",
            "kr_lines": (138, 141),
            "resolves": (),
        },
        "capacitors_per_module": {
            "value": 24,
            "units": "count",
            "kr_lines": (138, 141),
            "resolves": (),
        },
        "capacitance_total_F": {
            "value": 1.332e-3,
            "units": "F",
            "kr_lines": (147, 154),
            "resolves": (),
        },
        "nominal_inductance_H": {
            "value": 8.9e-9,
            "units": "H",
            "kr_lines": (147, 154),
            "resolves": (),
        },
        "quarter_discharge_time_s": {
            "value": 5.4e-6,
            "units": "s",
            "kr_lines": (147, 154),
            "resolves": (),
        },
        "short_circuit_current_A": {
            "value": 15.0e6,
            "units": "A",
            "kr_lines": (147, 154),
            "resolves": (),
        },
        "diagnostic_shot_voltage_kV": {
            "value": 32.5,
            "units": "kV",
            "kr_lines": (206, 212),
            "resolves": (),
        },
        "diagnostic_shot_current_A": {
            "value": 1.5e6,
            "units": "A",
            "kr_lines": (206, 212),
            "resolves": (),
        },
        "diagnostic_shot_fill": {
            "value": "H2 + 14% Ar at 3.4 hPa",
            "kr_lines": (206, 212),
            "resolves": (),
        },
    },
}


HEROLD_1989_POSEIDON_PF360_EXTRACTION: Mapping[str, Any] = {
    "source_id": "herold_1989_poseidon_pf360_comparative",
    "title": (
        "Comparative analysis of large plasma focus experiments performed at "
        "IPF, Stuttgart, and IPJ, Swierk"
    ),
    "authors": "Herold, H.; Jerzykiewicz, A.; Sadowski, M.; Schmidt, H.",
    "journal": "Nuclear Fusion 29(8):1255-1269 (1989)",
    "source_pdf_sha256": (
        "51a546954db969c97028db7caec866e49728cef60cb147b16208c2bcc5542a26"
    ),
    "kr_markdown": (
        "KnowledgeReference/"
        "comparative-analysis-of-large-plasma-focus-experiments-performed-"
        "at-ipf-stuttgart-and-ipj-51a54695.md"
    ),
    "scope_tag": "poseidon_pf360_cross_machine_comparative_context",
    "scope_caveat": (
        "POSEIDON/PF-360 evidence is valuable for scaling, startup, and "
        "emission-mechanism context, but it is not PF-1000 same-scope "
        "geometry, current, or validation evidence."
    ),
    "render_artifact": (
        "docs/extractions/herold_1989_poseidon_pf360_render_evidence/"
        "pdf_p004_journal_p1257.png"
    ),
    "render_artifact_sha256": (
        "27b95374f8e0e6d43f90c7eaf4951bec40093148fcdd0debae462e9f971b867f"
    ),
    "resolves_blockers": (),
    "candidate_context_only": (
        "startup radial-breakdown risk",
        "cross-machine electrode/insulator scaling",
        "cross-machine neutron and ion emission mechanisms",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "poseidon_ceramic_voltage_kV": {
            "value": 80.0,
            "units": "kV",
            "kr_lines": (323, 387),
        },
        "poseidon_ceramic_current_MA": {
            "value": 4.9,
            "units": "MA",
            "kr_lines": (323, 387),
        },
        "pf360_ceramic_current_MA": {
            "value": 2.1,
            "units": "MA",
            "kr_lines": (323, 387),
        },
        "breakdown_voltage_capacitance_trend": {
            "value": (
                "PF-360 breakdown voltage rises with storage-bank "
                "capacitance; radial breakdown above the insulator may cause "
                "current-sheath inhomogeneity or prevent sheath formation"
            ),
            "kr_lines": (445, 459),
        },
        "good_shot_radial_compression_velocity_m_s": {
            "value": 1.5e5,
            "units": "m/s",
            "kr_lines": (578, 584),
        },
        "bad_shot_radial_compression_velocity_m_s": {
            "value": (6.0e4, 8.0e4),
            "units": "m/s",
            "kr_lines": (578, 584),
        },
        "electron_density_peak_cm3": {
            "value": 1.0e19,
            "units": "cm^-3",
            "kr_lines": (714, 723),
        },
        "electron_density_axis_cm3": {
            "value": 3.0e18,
            "units": "cm^-3",
            "kr_lines": (714, 723),
        },
        "fast_ion_forward_emission_angle_deg": {
            "value": (0.0, 40.0),
            "units": "deg",
            "kr_lines": (782, 803),
        },
        "very_high_energy_ion_range_MeV": {
            "value": (1.0, 6.0),
            "units": "MeV",
            "kr_lines": (800, 803),
        },
        "neutron_pulse_fwhm_ns": {
            "value": (80.0, 150.0),
            "units": "ns",
            "kr_lines": (1182, 1208),
        },
    },
}


SCHOLZ_1999_FOAM_LINER_EXTRACTION: Mapping[str, Any] = {
    "source_id": "scholz_1999_foam_liner_current_sheath",
    "title": "Foam liner driven by a plasma focus current sheath",
    "authors": (
        "Scholz, M.; Karpinski, L.; Stepniewski, W.; Branitski, A. V.; "
        "Fedulov, M. V.; Medovschikov, S. F.; Nedoseev, S. L.; "
        "Smirnov, V. P.; Zurin, M. V.; Szydlowski, A."
    ),
    "journal": "Physics Letters A 262:453-456 (1999)",
    "source_pdf_sha256": (
        "8324d619499321fffbcbe68e3d300fb4fd2ff176e7f979de485fd59e4558653b"
    ),
    "kr_markdown": (
        "KnowledgeReference/"
        "foam-liner-driven-by-a-plasma-focus-current-sheath-8324d619.md"
    ),
    "scope_tag": "pf1000_modified_foam_liner_current_sheath_context",
    "scope_caveat": (
        "PF-1000 facility operated with a modified foam-liner target and "
        "modified tube electrodes; useful for current-sheath interaction and "
        "radiation context, not standard PF-1000 whole-shot geometry."
    ),
    "render_artifacts": (
        (
            "docs/extractions/scholz_1999_foam_liner_render_evidence/"
            "pdf_p002_journal_p454.png"
        ),
        (
            "docs/extractions/scholz_1999_foam_liner_render_evidence/"
            "pdf_p003_journal_p455.png"
        ),
    ),
    "resolves_blockers": (),
    "candidate_context_only": (
        "plasma-current-sheath/solid-load coupling",
        "VUV/soft-X-ray radiation timing",
        "active-power bookkeeping",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "liner_type_a_diameter_m": {
            "value": 0.020,
            "units": "m",
            "kr_lines": (135, 145),
        },
        "liner_type_b_diameter_m": {
            "value": 0.0054,
            "units": "m",
            "kr_lines": (135, 145),
        },
        "liner_length_m": {
            "value": 0.015,
            "units": "m",
            "kr_lines": (135, 145),
        },
        "modified_inner_electrode_diameter_m": {
            "value": 0.100,
            "units": "m",
            "kr_lines": (160, 170),
        },
        "modified_outer_electrode_diameter_m": {
            "value": 0.150,
            "units": "m",
            "kr_lines": (160, 170),
        },
        "modified_electrode_length_m": {
            "value": 0.330,
            "units": "m",
            "kr_lines": (160, 170),
        },
        "bank_capacitance_F": {
            "value": 1.0e-3,
            "units": "F",
            "kr_lines": (168, 174),
        },
        "charge_voltage_kV": {
            "value": 25.0,
            "units": "kV",
            "kr_lines": (168, 174),
        },
        "fill_pressure_hPa": {
            "value": 4.7,
            "units": "hPa",
            "kr_lines": (170, 175),
        },
        "current_amplitude_A": {
            "value": 1.0e6,
            "comparison": "greater_than",
            "units": "A",
            "kr_lines": (170, 176),
        },
        "current_rise_time_s": {
            "value": (4.0e-6, 5.0e-6),
            "units": "s",
            "source_basis": "render_verified_microsecond_glyph",
            "kr_lines": (170, 176),
        },
        "liner_border_radial_velocity_m_s": {
            "value": 2.0e4,
            "units": "m/s",
            "kr_lines": (207, 220),
        },
        "no_liner_current_sheath_velocity_m_s": {
            "value": 1.3e5,
            "units": "m/s",
            "kr_lines": (237, 256),
        },
        "vuv_soft_xray_power_W": {
            "value": 2.0e9,
            "units": "W",
            "kr_lines": (356, 360),
        },
        "hydrogen_absorption_corrected_power_W": {
            "value": 6.0e9,
            "units": "W",
            "kr_lines": (356, 360),
        },
        "active_power_peak_W": {
            "value": 3.0e10,
            "units": "W",
            "kr_lines": (369, 387),
        },
    },
}


LOARER_2007_GAS_BALANCE_EXTRACTION: Mapping[str, Any] = {
    "source_id": "loarer_2007_tokamak_gas_balance_fuel_retention",
    "title": "Gas balance and fuel retention in fusion devices",
    "source_pdf_sha256": (
        "09d09d6a8ecb8e4ba346fd038db674c492fb673ef1e8947bdaa3405723a8c30f"
    ),
    "kr_markdown": (
        "KnowledgeReference/"
        "gas-balance-and-fuel-retention-in-fusion-devices-09d09d6a.md"
    ),
    "source_doi": "10.1088/0029-5515/47/9/007",
    "scope_tag": "tokamak_pwi_gas_balance_context_not_dpf",
    "scope_caveat": (
        "Tokamak plasma-wall fuel-retention methodology. It can inform a "
        "future DPF wall/retention accounting design review, but it is not a "
        "DPF source and must not close PF-1000 physics blockers."
    ),
    "resolves_blockers": (),
    "candidate_context_only": ("plasma-wall gas-balance methodology",),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "long_term_retention_fraction_gas_balance": {
            "value": (0.10, 0.20),
            "units": "fraction",
            "kr_lines": (64, 77),
        },
        "post_mortem_retention_fraction": {
            "value": (0.03, 0.04),
            "units": "fraction",
            "kr_lines": (70, 77),
        },
        "gas_balance_equation_terms": {
            "value": (
                "integrated Qgas + QNBI + Qpellet equals plasma fuel content "
                "plus integrated vessel pumping plus integrated divertor "
                "pumping plus wall inventory"
            ),
            "kr_lines": (147, 180),
        },
        "short_timescale_reliability_range": {
            "value": "10 s to 6 min, with cited long-discharge examples up to 5 h",
            "kr_lines": (198, 205),
        },
    },
}


SHAKYA_2015_LEE_MODEL_EXTRACTION: Mapping[str, Any] = {
    "source_id": "shakya_2015_pf1000_pf400_lee_model",
    "title": "Comparison of Plasma Dynamics in Plasma Focus Devices PF1000 and PF400",
    "source_pdf_sha256": (
        "9094f12f0ead4d443592579eef082808ef91636f3e76e34b6e71de161b639be6"
    ),
    "kr_markdown": (
        "KnowledgeReference/"
        "comparison-of-plasma-dynamics-in-plasma-focus-devices-pf1000-and-"
        "pf400-9094f12f.md"
    ),
    "scope_tag": "pf1000_pf400_reduced_lee_model_context",
    "scope_caveat": (
        "This is a reduced Lee-model comparison. It may support baseline "
        "comparison and category-mismatch documentation, but it is not "
        "first-principles runtime evidence."
    ),
    "render_artifact": (
        "docs/extractions/shakya_2015_pf1000_pf400_render_evidence/"
        "pdf_p004_journal_p058.png"
    ),
    "render_artifact_sha256": (
        "5edc78a712c3f7e5d985810763b3d88ff13993762a29836daa3f6e9d5705aad0"
    ),
    "source_doi_status": (
        "no source DOI detected; reference-list DOI was sanitized from KR "
        "metadata during intake"
    ),
    "resolves_blockers": (),
    "candidate_context_only": (
        "Lee-model baseline",
        "Akel/Lee b=16 cm category-mismatch context",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "lee_model_parameter_roles": {
            "value": (
                "tube parameters b, a, z0; bank parameters L0, C0, r0; "
                "operational parameters V0, P0, fill gas; fitted fm, fc, "
                "fmr, fcr"
            ),
            "kr_lines": (133, 157),
        },
        "pf1000_voltage_kV": {
            "value": 27.0,
            "units": "kV",
            "kr_lines": (163, 193),
        },
        "pf1000_pressure_torr": {
            "value": 3.5,
            "units": "Torr",
            "kr_lines": (163, 193),
        },
        "pf1000_lee_cathode_radius_m": {
            "value": 0.160,
            "units": "m",
            "kr_lines": (186, 189),
            "scope_note": "Lee-model b parameter; not hardware metrology.",
        },
        "pf1000_anode_radius_m": {
            "value": 0.115,
            "units": "m",
            "kr_lines": (186, 189),
        },
        "pf1000_anode_length_m": {
            "value": 0.600,
            "units": "m",
            "kr_lines": (186, 189),
        },
        "pf1000_model_factors": {
            "value": {"fm": 0.13, "fc": 0.7, "fmr": 0.35, "fcr": 0.65},
            "kr_lines": (190, 193),
        },
        "pf1000_radial_phase_start_s": {
            "value": 7.415e-6,
            "units": "s",
            "kr_lines": (231, 238),
        },
        "pf1000_radial_phase_end_s": {
            "value": 9.074e-6,
            "units": "s",
            "kr_lines": (231, 238),
        },
        "pf1000_pinch_start_s": {
            "value": 9.34e-6,
            "units": "s",
            "kr_lines": (231, 238),
        },
        "pf1000_pinch_current_A": {
            "value": 826.73e3,
            "units": "A",
            "kr_lines": (320, 327),
        },
        "pf1000_peak_current_A": {
            "value": 1844.71e3,
            "units": "A",
            "kr_lines": (307, 314),
        },
        "pf1000_pinch_duration_s": {
            "value": 265.76e-9,
            "units": "s",
            "kr_lines": (291, 299),
        },
        "pf1000_peak_radial_shock_speed_m_s": {
            "value": 16.4e4,
            "units": "m/s",
            "kr_lines": (336, 343),
        },
    },
}


GRIBKOV_2007_PF1000_PART2_EXTRACTION: Mapping[str, Any] = {
    "source_id": "gribkov_2007_pf1000_part2_existing_kr_equivalent",
    "title": (
        "Plasma dynamics in the PF-1000 device under full-scale energy "
        "storage: II. Fast electron and ion characteristics versus neutron "
        "emission parameters and gun optimization perspectives"
    ),
    "source_pdf_sha256": (
        "80b44cd62c07af3343d24cc62d530f59ac46c0cbe520ffa9b2e438be5af086a3"
    ),
    "existing_kr_markdown": "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md",
    "existing_kr_status": (
        "user-supplied PDF is DOI/title-equivalent to existing KR; legacy KR "
        "metadata title remains poor but local text carries the paper title"
    ),
    "source_doi": "10.1088/0022-3727/40/12/008",
    "scope_tag": "pf1000_full_energy_fast_electron_ion_neutron_authority",
    "scope_caveat": (
        "PF-1000 Part II is strong source material for beam, neutron, and "
        "diagnostic mechanism packets, but this extraction does not by itself "
        "accept a runtime neutron model, beam-transport closure, or same-scope "
        "validation certificate."
    ),
    "resolves_blockers": (),
    "candidate_context_only": (
        "beam-target neutron mechanism",
        "fast-electron/fast-ion timing",
        "PF-1000 detector geometry and anisotropy",
        "anomalous-resistivity/virtual-diode mechanism context",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "neutron_model_features": {
            "value": (
                "hot confined plasma target; magnetized/entrapped beam ions; "
                "ion-ion fusion and Coulomb collisions"
            ),
            "kr_lines": (39, 56),
        },
        "mhd_stage_duration_text": {
            "value": "several microseconds",
            "kr_lines": (63, 69),
        },
        "pinch_confinement_formula": {
            "value": "Delta t = n tau = n r / v_i = n r / (3 k T / m_D)^0.5",
            "kr_lines": (79, 86),
        },
        "rayleigh_taylor_increment_s_inv": {
            "value": 1.0e8,
            "units": "s^-1",
            "kr_lines": (79, 90),
        },
        "kinetic_microinstability_timescale_s": {
            "value": (1.0e-13, 1.0e-10),
            "units": "s",
            "kr_lines": (87, 91),
        },
        "neutron_scaling_energy": {
            "value": "Y_n proportional to E_c^2",
            "kr_lines": (105, 119),
        },
        "neutron_scaling_pinch_current": {
            "value": "Y_n approximately 1e10 * I_p^4, with I_p in MA",
            "kr_lines": (109, 119),
        },
        "dt_neutron_yield_factor": {
            "value": 100.0,
            "units": "factor",
            "kr_lines": (117, 124),
        },
        "medium_deuteron_energy_keV": {
            "value": (50.0, 150.0),
            "units": "keV",
            "kr_lines": (129, 142),
        },
        "pinch_target_temperature_upper_eV": {
            "value": 1000.0,
            "units": "eV",
            "kr_lines": (129, 142),
        },
        "pinch_target_density_upper_cm3": {
            "value": 1.0e19,
            "units": "cm^-3",
            "kr_lines": (129, 142),
        },
        "virtual_plasma_diode_context": {
            "value": (
                "electron-MHD virtual plasma diode from anomalous resistivity "
                "and current abruption; electrons toward anode, ions toward "
                "cathode"
            ),
            "kr_lines": (149, 164),
        },
        "anode_diameter_m": {
            "value": 0.230,
            "units": "m",
            "kr_lines": (198, 201),
        },
        "anode_length_m": {
            "value": 0.600,
            "units": "m",
            "kr_lines": (198, 201),
        },
        "insulator_exposed_length_m": {
            "value": 0.113,
            "units": "m",
            "kr_lines": (223, 225),
        },
        "bank_capacitance_F": {
            "value": 1.320e-3,
            "units": "F",
            "kr_lines": (223, 230),
        },
        "shot_energy_range_kJ": {
            "value": (480.0, 850.0),
            "units": "kJ",
            "kr_lines": (223, 230),
        },
        "typical_energy_kJ": {
            "value": 810.0,
            "units": "kJ",
            "kr_lines": (223, 230),
        },
        "typical_voltage_kV": {
            "value": 35.0,
            "units": "kV",
            "kr_lines": (223, 230),
        },
        "typical_total_current_A": {
            "value": (2.5e6, 2.6e6),
            "units": "A",
            "kr_lines": (390, 421),
        },
        "best_total_current_A": {
            "value": 3.0e6,
            "comparison": "close_to",
            "units": "A",
            "kr_lines": (400, 404),
        },
        "best_neutron_yield": {
            "value": 6.0e11,
            "units": "neutrons/shot",
            "kr_lines": (182, 187),
        },
        "typical_good_neutron_yield": {
            "value": 2.0e11,
            "units": "neutrons/shot",
            "kr_lines": (182, 187),
        },
        "estimated_pinch_current_A": {
            "value": 2.0e6,
            "units": "A",
            "kr_lines": (422, 436),
        },
        "shot_3121_voltage_kV": {
            "value": 35.0,
            "units": "kV",
            "kr_lines": (445, 457),
        },
        "shot_3121_energy_kJ": {
            "value": 810.0,
            "units": "kJ",
            "kr_lines": (445, 457),
        },
        "shot_3121_y0_y90": {
            "value": 1.8,
            "units": "ratio",
            "kr_lines": (445, 457),
        },
        "neutron_pulse_fwhm_s": {
            "value": 150.0e-9,
            "units": "s",
            "kr_lines": (465, 477),
        },
    },
}


GRIBKOV_MALAQUIAS_2006_DMP_APPLICATIONS_EXTRACTION: Mapping[str, Any] = {
    "source_id": "gribkov_malaquias_2006_dmp_applications",
    "title": (
        "Dense magnetized plasma and its applications: review of the 3-year "
        "activity of the IAEA Co-ordinated Research Programme"
    ),
    "source_pdf_sha256": (
        "cca325c9ab3bda7dc2f948b2641b86ee391dd8dc51129412227cde1cf0690ef9"
    ),
    "kr_markdown": (
        "KnowledgeReference/"
        "dense-magnetized-plasma-and-its-applications-review-of-the-3-year-"
        "activity-of-the-iaea-co-cca325c9.md"
    ),
    "scope_tag": "dense_magnetized_plasma_applications_context",
    "scope_caveat": (
        "IAEA CRP review source. It supports DMP/DPF application and "
        "diagnostic-context reviews, but it does not close PF-1000 same-scope "
        "first-principles runtime blockers."
    ),
    "resolves_blockers": (),
    "candidate_context_only": (
        "DMP application taxonomy",
        "PF-1000 modernization context",
        "radiation-material interaction context",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "density_regime_position": {
            "value": (
                "between magnetic-confinement plasmas <=1e14 cm^-3 and "
                "inertial-fusion plasmas >=1e23 cm^-3"
            ),
            "kr_lines": (20, 31),
        },
        "dmp_device_attributes": {
            "value": (
                "compactness, high efficiency, reliable operation, and high "
                "brightness hard radiations"
            ),
            "kr_lines": (110, 128),
        },
        "radiation_types": {
            "value": (
                "plasma streams, fast electron and ion beams, soft/hard "
                "X-rays, and neutrons"
            ),
            "kr_lines": (124, 137),
        },
        "crp_institution_scope": {
            "value": "8 countries, 9 cities, 12 institutions",
            "kr_lines": (138, 155),
        },
        "pf1000_modernization_effect": {
            "value": (
                "one-order neutron-emission improvement and strong increase "
                "of other hard radiations"
            ),
            "kr_lines": (190, 202),
        },
        "pf1000_implosion_velocity_cm_s": {
            "value": 5.0e7,
            "units": "cm/s",
            "kr_lines": (190, 202),
        },
        "diagnostic_classes": {
            "value": (
                "Rogowski/voltage/magnetic probes, track detectors, "
                "Cerenkov detectors, optical and X-ray spectroscopy, "
                "Ross-filter X-ray detectors, neutron/scintillator probes, "
                "bolometers/calorimeters, 1 ns optical/X-ray cameras, "
                "laser interferometry, Thomson scattering"
            ),
            "kr_lines": (305, 358),
        },
        "pf1000_gyrating_particle_model_context": {
            "value": (
                "PF-1000 diagnostics checked fast electron/ion beam generation "
                "models based on plasma diode concept and Gyrating Particle "
                "Model correlation"
            ),
            "kr_lines": (451, 458),
        },
        "radiation_material_flux_regimes_W_cm2": {
            "value": {
                "implantation": (1.0e5, 1.0e7),
                "detachment": (1.0e7, 1.0e8),
                "explosive_or_broken_implantation": (1.0e8, 1.0e10),
            },
            "units": "W/cm^2",
            "kr_lines": (477, 499),
        },
    },
}


SPRINT6_USER_SUPPLIED_TARGET_EXTRACTIONS: Mapping[str, Mapping[str, Any]] = {
    "scholz_2001_recent_progress_pf1000_hardware": (
        SCHOLZ_2001_RECENT_PROGRESS_PF1000_HARDWARE_EXTRACTION
    ),
    "bruzzone_bernal_2001_lhi_duplicate_verification": (
        BRUZZONE_BERNAL_2001_DUPLICATE_VERIFICATION
    ),
    "scholz_2000_pf1000_device": SCHOLZ_2000_PF1000_DEVICE_EXTRACTION,
    "herold_1989_poseidon_pf360_comparative": (
        HEROLD_1989_POSEIDON_PF360_EXTRACTION
    ),
    "scholz_1999_foam_liner_current_sheath": SCHOLZ_1999_FOAM_LINER_EXTRACTION,
    "loarer_2007_tokamak_gas_balance_fuel_retention": (
        LOARER_2007_GAS_BALANCE_EXTRACTION
    ),
    "shakya_2015_pf1000_pf400_lee_model": SHAKYA_2015_LEE_MODEL_EXTRACTION,
    "gribkov_2007_pf1000_part2_existing_kr_equivalent": (
        GRIBKOV_2007_PF1000_PART2_EXTRACTION
    ),
    "gribkov_malaquias_2006_dmp_applications": (
        GRIBKOV_MALAQUIAS_2006_DMP_APPLICATIONS_EXTRACTION
    ),
}


def sprint6_user_supplied_target_extractions() -> dict[str, Any]:
    """Return the fail-closed user-supplied source-extraction manifest."""
    return {
        "date": "2026-05-20",
        "packets_count": len(SPRINT6_USER_SUPPLIED_TARGET_EXTRACTIONS),
        "packets": dict(SPRINT6_USER_SUPPLIED_TARGET_EXTRACTIONS),
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "guardrail": (
            "These source packets provide target-extracted evidence and "
            "duplicate verification only. They do not accept runtime geometry, "
            "anomalous resistivity closure, whole-shot simulation, or "
            "first-principles validation."
        ),
    }
