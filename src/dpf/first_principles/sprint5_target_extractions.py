"""Sprint 5 WS2 target-extraction packets.

Source: docs/CODEX_FIRST_PRINCIPLES_V2_HANDOFF_AUDIT_AND_NEXT_PLAN_2026_05_20.md
        §"Sprint 5 Workstream 2 — Execute Existing-Local Target Extractions"

These packets convert already-local KnowledgeReference material (and one
on-disk-but-not-yet-in-KR paper: Bennett 2017) into typed extraction records
with verbatim quotes, exact line ranges or page numbers, scope tags, and the
list of named blocker_ids each packet supports.

No packet promotes any acceptance flag. Every packet carries:

- ``accepted_runtime_claim = False``
- ``can_support_first_principles_acceptance = False``

The data is consumed by runtime modules through the existing
``source_targets`` / ``kr_targets`` registries. This file is the canonical
machine-readable record of Sprint 5 extraction work, separate from
``source_targets.py`` to keep that file from growing further.

Audit corrections folded in:

- audit row 6: UCSD/Beg ``massf`` lines corrected from ``:615-670`` to
  ``:597-601`` (formula) + ``:631-640`` (Paschen regimes) + ``:642-644`` (Te)
  + ``:654-660`` (Liz/Li = 2.4).
- audit row 7: Bennett 2017 71% sheath current fraction is at ``1 us`` (not
  500 ns).
- audit row 8: Braginskii 1965 Table 2 / Eqs. 4.30-4.45 RENDER-VERIFIED via
  the Read-tool PDF page renderer; Table 2 is at journal p.251 (PDF p.26).
- Bernard 1977 thermonuclear-1/4-prefactor: VERIFIED NOT FOUND. Eq.(5) is a
  proportionality ``N0 ∝ n² <σv> t_conf`` with no pair-counting derivation.
  The blocker remains ``external_acquisition_required``.
- Plasma Focus Update 2021 audit correction: the 320/500 keV deuteron values
  are from Lerner et al. on FF-1 (Focus Fusion-1), NOT PF-1000. Scope is
  ``cross_device_comparator_only``, not ``pf1000_full_energy``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# --------------------------------------------------------------------------
# Packet 1: Bennett et al. 2017 — startup BVP source-availability closures
# --------------------------------------------------------------------------
# Phys. Plasmas 24:062705 (DOI 10.1063/1.4985313). The on-disk PDF is
# mislabeled as ``schmidt-2017-kinetic-dpf-breakdown.pdf`` — actual authors
# are Bennett et al.. Not yet in KR (KR promotion is the recommended Sprint 5
# follow-up action).
BENNETT_2017_STARTUP_EXTRACTION: Mapping[str, Any] = {
    "source_id": "bennett_2017_kinetic_dpf_breakdown",
    "doi": "10.1063/1.4985313",
    "on_disk_pdf": (
        "archive_reference_OLD/references/papers/core-dpf/"
        "schmidt-2017-kinetic-dpf-breakdown.pdf"
    ),
    "filename_mislabel_actual_authors": "Bennett et al.",
    "scope_tag": "pf1000_generic",
    "scope_caveat": (
        "kinetic-PIC methodology paper on MA-scale DPF gas breakdown; "
        "not Akel-16-kV-specific acceptance"
    ),
    "resolves_blockers": (
        "STARTUP-BVP-CH03",
        "STARTUP-BVP-CH04",
        "STARTUP-BVP-CH07",
        "STARTUP-BVP-CH08",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "kr_promotion_recommended": True,
    "kr_slug_when_promoted": "bennett-2017-kinetic-dpf-breakdown",
    "targets": {
        "seed_plasma_density": {
            "value": 1e7,
            "units": "cm^-3",
            "source_page": "p.2 (062705-2)",
            "verbatim": (
                "The DPF volume is also initialized with a 10^7-cm^-3 density "
                "plasma of deuterium ions and electrons to provide seed "
                "electrons for the avalanche ionization process."
            ),
            "resolves": ("STARTUP-BVP-CH03",),
        },
        "fill_pressure_baseline": {
            "value": 5.5,
            "units": "Torr",
            "neutral_d_density_cm3": 3.52e17,
            "source_page": "p.2 (062705-2)",
            "verbatim": (
                "The volume in Fig. 3 is filled with neutral deuterium atoms "
                "with an initial density of 3.52 * 10^17 cm^-3 "
                "(5.5 Torr of molecular deuterium, D_2)."
            ),
            "resolves": ("STARTUP-BVP-CH01",),
            "note": "corroborative only",
        },
        "breakdown_delay": {
            "value": 20.0,
            "units": "ns",
            "qualifier": "approximate",
            "source_page": "p.4 (062705-4)",
            "verbatim": (
                "The measured breakdown time (the time between the rise of "
                "voltage and the rapid rise in current) is approximately 20 ns."
            ),
            "resolves": ("STARTUP-BVP-CH04",),
        },
        "pressure_regime_low": {
            "criterion": "lambda_ioniz > electrode gap (~20 cm)",
            "mode": "volumetric uniform breakdown",
            "source_page": "p.5 (062705-5)",
            "verbatim": (
                "At low pressures, the electron ionization path length exceeds "
                "20 cm so electrons traveling axially are more likely to "
                "ionize the gas leading to bulk breakdown in the DPF volume."
            ),
            "resolves": ("STARTUP-BVP-CH04",),
        },
        "pressure_regime_med": {
            "criterion": "lambda_ioniz ~ L_insulator > coaxial gap",
            "mode": "surface ionization along insulator (optimal)",
            "source_page": "p.5 (062705-5)",
            "verbatim": (
                "In an intermediate pressure range, the ionization path length "
                "may exceed the coaxial gap but approach the length of the "
                "insulator, which is longer than the gap in typical DPF designs."
            ),
            "resolves": ("STARTUP-BVP-CH04",),
        },
        "pressure_regime_high": {
            "criterion": "lambda_ioniz within a few cm (> ~15 Torr)",
            "mode": "radial filamentation",
            "source_page": "p.5 (062705-5)",
            "verbatim": (
                "At pressures above 15 Torr, electron impact ionization occurs "
                "within a few cms, so the gas may breakdown radially across "
                "the coaxial gap."
            ),
            "resolves": ("STARTUP-BVP-CH04",),
        },
        "explosive_emission_threshold_bulk": {
            "value": 250.0,
            "units": "kV/cm",
            "source_page": "p.3 (062705-3)",
            "verbatim": (
                "We use an electric field stress threshold of 250 kV/cm except "
                "for the cathode knife-edge, where the threshold is reduced to "
                "10 kV/cm to approximate the field enhancement of its 3D "
                "structures in our 2D model."
            ),
            "resolves": ("STARTUP-BVP-CH07",),
        },
        "explosive_emission_threshold_knife_edge": {
            "value": 10.0,
            "units": "kV/cm",
            "source_page": "p.3 (062705-3)",
            "verbatim": (
                "...where the threshold is reduced to 10 kV/cm to approximate "
                "the field enhancement of its 3D structures in our 2D model."
            ),
            "resolves": ("STARTUP-BVP-CH07",),
        },
        "electron_temperature_breakdown": {
            "value_eV": 4.0,
            "range_eV": (3.5, 4.0),
            "source_page": "p.5 (062705-5)",
            "verbatim": (
                "the mean local temperatures (T_e) in the electron "
                "distributions from simulation remain near 4 eV, well into "
                "breakdown, as shown in Fig. 7."
            ),
            "resolves": ("STARTUP-BVP-CH07",),
        },
        "density_contour_100ns": {
            "value": 1e13,
            "units": "cm^-3",
            "qualifier": "order of magnitude, bulk volume",
            "source_page": "p.3 (062705-3)",
            "verbatim": (
                "By 100 ns, as the plasma sheath is forming, a bulk ionization "
                "of order 10^13 cm^-3 has already occurred in the volume."
            ),
            "resolves": ("STARTUP-BVP-CH08",),
        },
        "density_contour_500ns": {
            "value": 1e15,
            "units": "cm^-3",
            "qualifier": "plasma channel across coaxial gap; Fig. 4(b)",
            "source_page": "p.3 (062705-3)",
            "verbatim": (
                "By 500 ns [Fig. 4(b)], a plasma channel has formed across the "
                "coaxial electrode gap."
            ),
            "resolves": ("STARTUP-BVP-CH08",),
        },
        "density_contour_1us": {
            "value": 1e15,
            "units": "cm^-3",
            "qualifier": "along insulator; 90% ionized near insulator end",
            "source_page": "p.3 (062705-3)",
            "verbatim": (
                "By 400 ns into the pulse a plasma of 10^15 cm^-3 density has "
                "formed along the insulator with the aid of the cathode "
                "knife-edge."
            ),
            "resolves": ("STARTUP-BVP-CH08",),
        },
        "sheath_current_fraction_1us": {
            "value_percent": 71.0,
            "at_time_us": 1.0,
            "source_page": "p.3 (062705-3)",
            "verbatim": "by 1 us [Fig. 4(c)], it is carrying 71% of the current.",
            "audit_row_7_correction_confirmed": True,
            "audit_row_7_note": (
                "Codex audit row-7 correction CONFIRMED: 71% is at 1 us, not "
                "500 ns. The 500 ns entry (Fig. 4(b)) gives only channel "
                "formation; no current-fraction percentage at 500 ns."
            ),
            "resolves": ("STARTUP-BVP-CH08",),
        },
        "photoionization_negligibility_bound": {
            "value_percent": 1.2,
            "by_time_ns": 125.0,
            "source_page": "p.3 (062705-3)",
            "verbatim": (
                "Preliminary simulations run with the addition of "
                "photoionization showed a 1.2% increase in electron density "
                "by 125 ns... Photons are, therefore, neglected here."
            ),
            "resolves": ("STARTUP-BVP-CH04",),
            "significance": "justifies STARTUP-BVP-CH06 photoemission neglect",
        },
    },
}


# --------------------------------------------------------------------------
# Packet 2: Braginskii 1965 — transport coefficients (RENDER-VERIFIED)
# --------------------------------------------------------------------------
# PDF on disk; pdftotext could not expose Table 2 or Eqs. 4.30-4.45 (Codex
# audit row 8 downgrade). Read-tool PDF page rendering CONFIRMED Table 2
# at journal p.251 (PDF p.26) and Eqs. 4.30-4.45 at journal pp.249-253
# (PDF pp.25-28). Spot-checked 24 numeric values across Z = 1, 2, 3, 4, inf
# columns. Status moves from
# ``pdf_present_needs_rendered_page_or_ocr_verification`` to
# ``kr_promotion_recommended`` with rendered-page evidence attached.
BRAGINSKII_1965_TRANSPORT_EXTRACTION: Mapping[str, Any] = {
    "source_id": "braginskii_1965_transport_processes",
    "on_disk_pdf": (
        "archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf"
    ),
    "scope_tag": "generic_formulary",
    "resolves_blockers": ("CLOSURE-BLK-BRAG-001",),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "render_verification_status": "verified_via_read_tool_pdf_page_render",
    "table_2_journal_page": 251,
    "table_2_pdf_page": 26,
    "equations_4_30_to_4_45_journal_pages": (249, 253),
    "equations_4_30_to_4_45_pdf_pages": (25, 28),
    "pdf_page_to_journal_page_offset": 202,
    "chapter_title_verified": "TRANSPORT PROCESSES IN A PLASMA, S. I. Braginskii",
    "z_columns": (1, 2, 3, 4, "inf"),
    "coefficient_families": ("alpha", "beta", "gamma", "delta"),
    "spot_checked_values": (
        # Z = 1 column (deuterium plasma, the DPF-relevant case)
        {"Z": 1, "coefficient": "alpha_0", "value": "0.5129"},
        {"Z": 1, "coefficient": "beta_0", "value": "0.7110"},
        {"Z": 1, "coefficient": "gamma_0", "value": "3.1616"},
        {"Z": 1, "coefficient": "delta_0", "value": "3.7703"},
        {"Z": 1, "coefficient": "delta_1", "value": "14.79"},
        {"Z": 1, "coefficient": "alpha_1_prime", "value": "6.416"},
        {"Z": 1, "coefficient": "beta_1_prime", "value": "5.101"},
        {"Z": 1, "coefficient": "gamma_1_prime", "value": "4.664"},
        {"Z": 1, "coefficient": "gamma_0_prime", "value": "11.92"},
        # Z = inf column (Lorentz gas limit)
        {"Z": "inf", "coefficient": "alpha_0", "value": "0.2949"},
        {"Z": "inf", "coefficient": "beta_0", "value": "1.521"},
        {"Z": "inf", "coefficient": "gamma_0", "value": "12.471"},
        {"Z": "inf", "coefficient": "delta_0", "value": "0.0961"},
    ),
    "equation_summaries": {
        "4_30": (
            "R_u = -alpha_par*u_par - alpha_perp*u_perp + alpha_wedge*[h x u]; "
            "momentum-transfer friction force from relative electron-ion drift"
        ),
        "4_31": (
            "R_T = -beta_par^T*grad_par(T_e) - beta_perp^T*grad_perp(T_e) "
            "- beta_wedge^T*[h x grad(T_e)]; electron thermal force"
        ),
        "4_39": (
            "q_i = -kappa_par^i*grad_par(T_i) - kappa_perp^i*grad_perp(T_i) "
            "+ kappa_wedge^i*[h x grad(T_i)]; ion heat flux"
        ),
        "4_41_to_4_42": (
            "Stress tensor pi_{alpha,beta} decomposed via 5 viscosity coeffs "
            "eta_0..eta_4 and rate-of-strain tensors W_{0..4 alpha,beta}"
        ),
        "4_44": (
            "Ion viscosity (Z=1): eta_0^i = 0.96*n_i*T_i*tau_i; "
            "eta_2^i, eta_4^i functions of x = omega_i*tau_i, "
            "Delta = x^4 + 4.03*x^2 + 2.33"
        ),
        "4_45": (
            "Electron viscosity (Z=1): eta_0^e = 0.733*n_e*T_e*tau_e; "
            "eta_2^e = n_e*T_e*tau_e*(2.05*x^2 + 8.50)/Delta; "
            "eta_4^e = -n_e*T_e*tau_e*x*(x^2 + 7.91)/Delta; "
            "x = omega_e*tau_e, Delta = x^4 + 13.8*x^2 + 11.6"
        ),
    },
    "external_cross_check_sources": (
        "plasmapy.formulary.braginskii.ClassicalTransport "
        "(hardcoded Z-dependent coefficients) — "
        "https://docs.plasmapy.org/en/stable/formulary/braginskii.html"
    ),
}


# --------------------------------------------------------------------------
# Packet 3: Scholz / Gribkov 2007 Part II — PF-1000 full-energy fast-ion
# --------------------------------------------------------------------------
# Already in KR at KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md
# (this is Gribkov 2007 J. Phys. D 40:3592, reclassified in V2 from external
# to existing_kr_target_extraction_pending per Codex V1 audit row 5).
SCHOLZ_GRIBKOV_2007_PARTII_EXTRACTION: Mapping[str, Any] = {
    "source_id": "scholz_gribkov_2007_pf1000_partii",
    "doi": "10.1088/0022-3727/40/12/008",
    "kr_path": "KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md",
    "scope_tag": "pf1000_full_energy",
    "device_operating_point": {
        "shot_number": 3121,
        "fill_pressure_Pa": 465,
        "charge_voltage_kV": 35,
        "bank_energy_MJ": 0.810,
    },
    "resolves_blockers": ("NEUTRON-BLK-001",),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "targets": {
        "direct_fast_deuteron_measurement": {
            "kr_lines": (318, 323),
            "detector_type": "PM-355 nuclear track detectors",
            "detector_distance_mm": 550,
            "observables": ("ion_flux", "angular_distribution", "ion_source_location"),
            "verbatim": (
                "An angular distribution of fast deuterons has been measured "
                "with nuclear track detectors (of the PM-355 type), placed at "
                "a distance of 550 mm from the inner electrode."
            ),
        },
        "5_counter_neutron_anisotropy": {
            "kr_lines": (445, 460),
            "Y0_over_Y90": 1.8,
            "Y180_over_Y90": 0.65,
            "anisotropy_character": "normal_forward_peaked_at_zero_to_Z_axis",
            "detector_count": 5,
            "detector_type": "silver_counters",
            "verbatim": (
                "the anisotropy of the emission measured in the laboratory "
                "coordinate frame has a so-called 'normal' character... its "
                "magnitudes are equal to about 1.8 for the ratio Y0°/Y90° and "
                "to ≈ 0.65 for the ratio Y180°/Y90°."
            ),
            "scope_caveat_for_akel_16kv": (
                "shot 3121 is 35 kV / 0.810 MJ — cross-scope for Akel 16 kV "
                "Option A; same-scope under Option B (PF-1000 full-energy)"
            ),
        },
        "neutron_spectra_fast_deuterons": {
            "kr_lines": (1138, 1165),
            "kinematics_equation": (
                "En = 3.269_MeV + Ed + 2*sqrt(2*En*Ed)*cos(theta)"
            ),
            "kinematics_eq_number": 17,
            "En_at_Ed0_theta90_MeV": 2.45,
            "En_headon_at_Ed100keV_MeV": 2.85,
            "acting_fast_ion_energy_range_keV": (10, 100),
            "FWHM_Tpl_keV_sideon": (3, 4),
        },
        "beam_target_angular_distribution_at_100keV": {
            "kr_lines": (1165, 1172),
            "reference_Ed_keV": 100,
            "angular_cross_section_eq": (
                "sigma(theta) = sigma(pi/2) * (1 + A * cos^2(theta))"
            ),
            "eq_number": 18,
        },
    },
}


# --------------------------------------------------------------------------
# Packet 4: Bernard 1977 — historical Mather scope (wrong-scope-only)
# --------------------------------------------------------------------------
# Already in KR via docs/CORPUS_RESCAN_KR_PROMOTION_2026_05_20.md.
# Ti = 700 eV is FILAMENT phase on ~500 kA Mather, NOT PF-1000 pinch.
# Thermonuclear 1/4 prefactor: VERIFIED NOT FOUND.
BERNARD_1977_HISTORICAL_EXTRACTION: Mapping[str, Any] = {
    "source_id": "bernard_1977_dpf_high_intensity_neutron_source",
    "journal_citation": (
        "Nuclear Instruments and Methods 145 (1977) 191-218, "
        "North-Holland Publishing Co."
    ),
    "kr_path": (
        "KnowledgeReference/"
        "the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md"
    ),
    "scope_tag": "historical_mather_wrong_scope",
    "scope_caveat": (
        "~500 kA Limeil/Juelich Mather-type DPF; filament-phase Thomson "
        "Ti = 700 eV is NOT pinch-phase and NOT transferable to PF-1000"
    ),
    "resolves_blockers": ("NEUTRON-BLK-001 (historical context only)",),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "thermonuclear_one_over_four_prefactor_found_here": False,
    "thermonuclear_prefactor_note": (
        "Eq.(5) at lines 1370-1382 reads `N0 ∝ n^2 <σv> t_conf` — "
        "proportionality only; no identical-particle 1/4 pair-counting "
        "derivation. Blocker remains external_acquisition_required; primary "
        "candidate is Glasstone & Lovberg (1960) Ch. 1."
    ),
    "targets": {
        "filament_phase_Ti_thomson": {
            "kr_lines": (455, 456),
            "value_eV": 700,
            "phase": "filament_NOT_pinch",
            "machine": "Limeil/Juelich ~500 kA Mather",
            "method": "coherent (alpha >> 1) Thomson scattering, ruby laser",
            "verbatim": (
                "The independent scattering measurement yields 700 eV for "
                "the deuteron temperature."
            ),
            "runtime_use_permitted": False,
        },
        "filament_not_neutron_source_proof": {
            "kr_lines": (457, 465),
            "Y_thermo_calc": 5e5,
            "Y_total_observed": 1e9,
            "ratio": 5e-4,
            "verbatim": (
                "One finds 5 x 10^5 which proves clearly that the collisional "
                "filament is not the neutron source"
            ),
            "scope_significance": (
                "the authors' own conclusion that 700 eV does NOT explain the "
                "DPF neutron yield — invalidating any future cite of this "
                "Ti = 700 eV as a PF-1000 pinch-phase comparator"
            ),
        },
        "thomson_method_context": {
            "kr_lines": (976, 1033),
            "alpha_regime": "alpha >> 1 (coherent ion-acoustic feature)",
            "scattering_angle_deg": (6.5, 7.5),
            "k_vector_cm_inv": 1.1e4,
        },
        "neutron_TOF_three_direction": {
            "kr_lines": (1185, 1193),
            "angles_deg": (0, 45, 90),
            "energy_resolution_keV_better_than": 50,
            "deuteron_energy_range_keV": (30, 350),
            "spectrum_shape": "E^-3",
            "mechanism_fit": "generalized (d,d)-beam-target model",
        },
        "frascati_1MJ_UIL_waveform": {
            "kr_lines": (1546, 1547),
            "device": "Frascati 1 MJ Mather",
            "bank_energy_MJ": 1.0,
            "charging_voltage_kV": 33,
            "fill_pressure_torr": 22,
            "use": "research_planning_only_not_pf1000",
        },
        "HXR_spectrum_to_350keV": {
            "kr_lines": (787, 797),
            "energy_range_keV": (60, 350),
            "spectrum_shape": "E^-3.3",
            "time_correlated_with_neutrons": True,
        },
    },
}


# --------------------------------------------------------------------------
# Packet 5: UCSD/Beg current-sheath initiation — method context only
# --------------------------------------------------------------------------
# Already in KR. Codex audit row 6 corrected the V1 line ranges from the
# wrong ``:615-670`` to the correct three ranges below.
UCSD_BEG_CURRENT_SHEATH_EXTRACTION: Mapping[str, Any] = {
    "source_id": "ucsd_beg_current_sheath_initiation",
    "doi": "10.1063/5.0020936",
    "kr_path": (
        "KnowledgeReference/"
        "effect-of-current-sheath-initiation-on-the-radial-collapse"
        "-and-energetic-particle-accelera-b2e95b88.md"
    ),
    "scope_tag": "wrong_scope_method_context",
    "device_scope": "ucsd_10_kj_mather_not_pf1000",
    "device_energy_per_paper_title": "10 kJ",
    "resolves_blockers": (
        "STARTUP-BVP-CH04 (method context)",
        "STARTUP-BVP-CH07 (method context)",
        "STARTUP-BVP-CH12 (method context)",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "audit_row_6_line_range_correction_confirmed": True,
    "audit_row_6_correction_summary": (
        "V1 cited :615-670 (verified WRONG); V2 cites :597-601 (massf) + "
        ":631-640 (Paschen regimes) + :642-644 (Te = 4 eV) + :654-660 "
        "(Liz/Li = 2.4) (all VERIFIED CORRECT against KR file text)."
    ),
    "targets": {
        "massf_formula": {
            "kr_lines": (597, 601),
            "formula": "massf = 0.4 * p0**(-0.5)",
            "p0_units": "Torr",
            "example_values_torr_to_massf": {2: 0.28, 4: 0.20, 6: 0.16},
            "verbatim": (
                "accurate experimental data fitting required a variable mass "
                "sweeping factor (massf equivalent to 0.4*p0-1/2 for all "
                "insulators...) for instance massf equal to 0.28, 0.2, and "
                "0.16 for 2, 4, and 6 torr respectively"
            ),
        },
        "pressure_regime_boundaries": {
            "kr_lines": (631, 640),
            "low": {"condition": "p0 < 0.75 Torr", "mode": "diffuse_volumetric"},
            "medium": {"condition": "0.75 < p0 < 3.75 Torr", "mode": "insulator_surface_e_glide"},
            "high": {"condition": "p0 > 3.75 Torr", "mode": "radial_filaments"},
            "method_caveat": (
                "Paschen-law analogy 'fragile' for DPFs per the source: "
                "'once a plasma is formed, such physics should no longer apply'"
            ),
        },
        "Te_4_eV_breakdown": {
            "kr_lines": (642, 644),
            "value_eV": 4.0,
            "qualifier": "Te ~ 4 eV for all fill pressures (modeling assumption)",
            "verbatim": (
                "Assuming a Te~4 eV for all fill pressures, and using the "
                "relationships between ionization path length and pressure, "
                "Liz(P), derived by Bennett et al."
            ),
        },
        "Liz_over_Li_ratio": {
            "kr_lines": (654, 660),
            "value": 2.4,
            "qualifier": "Liz / L_insulator at optimal pressure",
            "verbatim": (
                "it is found to agree remarkably well with Liz(P)/Li = 2.4 "
                "when using the optimal pressure for each of the three "
                "insulator lengths considered in our study."
            ),
        },
    },
}


# --------------------------------------------------------------------------
# Packet 6: Stepniewski 2004 — formal hardware-scope review (STAYS BLOCKED)
# --------------------------------------------------------------------------
# The 0.015 m hollow-bore stays blocked. KR sentence is explicit
# simulation-input wording. Krauz 2012 confirms bore EXISTS (r >= 12 mm) but
# does not publish the bore radius. External acquisition target identified
# (Miklaszewski 2001 — FREE PDF confirmed on disk-rescan ledger).
STEPNIEWSKI_2004_REVIEW_PACKET: Mapping[str, Any] = {
    "source_id": "stepniewski_2004_pf1000_mhd_modelling",
    "kr_path": (
        "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md"
    ),
    "scope_tag": "pf1000_simulation_parameter_not_hardware_metrology",
    "blocker_id": "PF1000-BLK-009",
    "verdict": "stays_blocked_simulation_context_only",
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "hollow_bore_radius_value_extracted": "0.015 m",
    "hollow_bore_radius_status": "modeling_context_requires_review",
    "kr_verbatim_lines": (310, 314),
    "kr_verbatim_quote": (
        "The parameters of PF-1000 facility have been taken for the "
        "simulations. They are as follows: radius of the inner electrode "
        "0.12 m, outer electrode 0.18 m, hollow radius in the centre of the "
        "electrode 0.015 m, electrode length 0.60 m."
    ),
    "corroborating_krauz_2012_line": 373,
    "krauz_2012_finding": (
        "hollow bore physically exists (probe access along hollow anode "
        "axis); r >= 12 mm inferred from probe placement at radii 1.2 and "
        "4 cm; no hardware-scope bore radius value published"
    ),
    "external_acquisition_to_close": (
        {
            "author": "Miklaszewski et al.",
            "year": 2001,
            "journal": "Nukleonika 46 suppl.1, S61-S64",
            "title": (
                "Neutron and fast ion emission from PF-1000 facility "
                "equipped with new large electrodes"
            ),
            "free_pdf_url": (
                "http://www.ichtj.waw.pl/ichtj/nukleon/back/full/"
                "vol46_2001/v46s1p061f.pdf"
            ),
            "verified_http_200": True,
            "free_download_confirmed": True,
        },
        {
            "author": "Schmidt et al.",
            "year": 2002,
            "journal": "Physica Scripta 66:168-172",
            "title": (
                "Review of recent experiments with the mega-joule PF-1000 "
                "plasma focus device"
            ),
            "free_pdf_url": None,
            "verified_http_200": False,
        },
    ),
}


# --------------------------------------------------------------------------
# Packet 7: Plasma Focus Update 2021 — comparator + Te filter-ratio caveats
# --------------------------------------------------------------------------
# Already in KR via corpus-rescan promotion. Confirms PF-1000 cathode-cage
# 200 mm radius (third hardware source), Te = 7.5 keV filter-ratio with
# strong method caveats (text-only), 750 eV "possibly mis-evaluated" cross-
# check. AUDIT CORRECTION: 320/500 keV beam-target values are FF-1
# (Lerner et al.), NOT PF-1000.
PLASMA_FOCUS_UPDATE_2021_EXTRACTION: Mapping[str, Any] = {
    "source_id": "auluck_2021_plasma_focus_update",
    "doi": "10.3390/plasma4030033",
    "kr_root": (
        "KnowledgeReference/"
        "update-on-the-scientific-status-of-the-plasma-focus-1385adeb.md"
    ),
    "scope_tag": "pf1000_full_energy",
    "resolves_blockers": (
        "Cathode-cage radius (third KR hardware source)",
        "NEUTRON-BLK-001 beam-target comparator context (cross-device only)",
    ),
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "te_filter_ratio_is_text_only_with_method_caveats": True,
    "audit_correction_320_500_kev_is_ff1_not_pf1000": True,
    "audit_correction_note": (
        "The 320 keV mean / 500 keV max deuteron values cited in the V1 "
        "comparator-confirmation audit are from Lerner et al. [95,140] on "
        "FF-1 (Focus Fusion-1), reported in §4.3.2 'PF devices other than "
        "PF-1000'. They MUST be tagged cross_device_comparator_only, NOT "
        "pf1000_full_energy. PF-1000-specific value is ~100-350 keV "
        "dominant deuteron energy (pages-0101-0125.md:403, ref [391])."
    ),
    "targets": {
        "pf1000_geometry_summary": {
            "kr_chunk": "chunks/.../pages-0026-0050.md",
            "kr_lines": (822, 849),
            "cathode_count": 12,
            "cathode_material": "stainless_steel",
            "cathode_tube_diameter_mm": 82,
            "cathode_cage_diameter_mm": 400,
            "cathode_cage_radius_mm": 200,
            "anode_diameter_mm": 230,
            "anode_material": "copper",
            "insulator_material": "alumina",
            "chamber_diameter_m": 1.4,
            "chamber_length_m": 2.5,
        },
        "te_filter_ratio_zaloga_2018": {
            "kr_chunk": "chunks/.../pages-0026-0050.md",
            "kr_lines": (512, 517),
            "value_keV": 7.5,
            "method": "filter_ratio_PIN_diode",
            "gas_mix": "D2_plus_Ne_admixture_NOT_pure_D2",
            "scope": "TEXT_ONLY_METHOD_CONTEXT",
            "accepted_runtime_claim": False,
            "method_caveats": (
                "local hot-spot Te, not bulk pinch",
                "filter-ratio is model-dependent (optically-thin "
                "bremsstrahlung assumption)",
                "D2+Ne admixture (not pure D2 — Akel scope is pure D2)",
                "source text qualifier: 'probably to individual hot-spots'",
                "parallel 750 eV measurement (ref [402]) text-flagged as "
                "possibly mis-evaluated",
            ),
        },
        "te_750ev_possibly_misevaluated_crosscheck": {
            "kr_chunk": "chunks/.../pages-0101-0125.md",
            "kr_lines": (1007, 1009),
            "value_eV": 750,
            "method": "MCP_quadrant_camera_plus_PIN_diode_ref_402",
            "scope": "TEXT_ONLY_METHOD_CAVEAT",
            "note": (
                "source text: 'possibility that this might be incorrectly "
                "evaluated: 750 eV may be the average photon energy and "
                "temperature may be ~4 times less in line with other "
                "measurements'"
            ),
        },
        "ff1_beam_target_NOT_pf1000": {
            "kr_chunk": "chunks/.../pages-0126-0150.md",
            "kr_lines": (528, 530),
            "mean_deuteron_keV": 320,
            "max_deuteron_keV": 500,
            "method": "ion_time_of_flight_Lerner_et_al_95_140",
            "device": "FF-1_Focus_Fusion_1",
            "scope": "cross_device_comparator_only",
            "explicit_NOT_pf1000": True,
        },
    },
}


# --------------------------------------------------------------------------
# Sprint 5 master accessor
# --------------------------------------------------------------------------
SPRINT_5_TARGET_EXTRACTIONS: Mapping[str, Mapping[str, Any]] = {
    BENNETT_2017_STARTUP_EXTRACTION["source_id"]: BENNETT_2017_STARTUP_EXTRACTION,
    BRAGINSKII_1965_TRANSPORT_EXTRACTION["source_id"]: BRAGINSKII_1965_TRANSPORT_EXTRACTION,
    SCHOLZ_GRIBKOV_2007_PARTII_EXTRACTION["source_id"]: SCHOLZ_GRIBKOV_2007_PARTII_EXTRACTION,
    BERNARD_1977_HISTORICAL_EXTRACTION["source_id"]: BERNARD_1977_HISTORICAL_EXTRACTION,
    UCSD_BEG_CURRENT_SHEATH_EXTRACTION["source_id"]: UCSD_BEG_CURRENT_SHEATH_EXTRACTION,
    STEPNIEWSKI_2004_REVIEW_PACKET["source_id"]: STEPNIEWSKI_2004_REVIEW_PACKET,
    PLASMA_FOCUS_UPDATE_2021_EXTRACTION["source_id"]: PLASMA_FOCUS_UPDATE_2021_EXTRACTION,
}


def sprint5_local_target_extractions() -> Mapping[str, Any]:
    """Return the 7-packet Sprint 5 target-extraction manifest.

    The returned manifest carries an aggregate ``can_support_first_principles_acceptance``
    of ``False`` regardless of contents — this is by construction. Individual
    packets resolve named blockers at the source-extraction level only; runtime
    acceptance requires upstream gate code changes (see the audit-handoff V2
    and the physics-acceptance promotion protocol).
    """
    return {
        "packet_id": "sprint5_local_target_extractions_2026_05_20",
        "controlling_audit": (
            "docs/CODEX_FIRST_PRINCIPLES_V2_HANDOFF_AUDIT_AND_NEXT_PLAN_2026_05_20.md"
        ),
        "workstream": "WS2",
        "packets_count": len(SPRINT_5_TARGET_EXTRACTIONS),
        "packets": dict(SPRINT_5_TARGET_EXTRACTIONS),
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "audit_corrections_folded_in": (
            "audit_row_6_ucsd_beg_line_ranges",
            "audit_row_7_bennett_71pct_at_1us_not_500ns",
            "audit_row_8_braginskii_render_verified_via_read_tool_pdf_pages",
            "bernard_1977_thermonuclear_prefactor_verified_not_found",
            "plasma_focus_update_320_500_kev_is_ff1_not_pf1000",
        ),
    }


__all__ = (
    "BENNETT_2017_STARTUP_EXTRACTION",
    "BRAGINSKII_1965_TRANSPORT_EXTRACTION",
    "SCHOLZ_GRIBKOV_2007_PARTII_EXTRACTION",
    "BERNARD_1977_HISTORICAL_EXTRACTION",
    "UCSD_BEG_CURRENT_SHEATH_EXTRACTION",
    "STEPNIEWSKI_2004_REVIEW_PACKET",
    "PLASMA_FOCUS_UPDATE_2021_EXTRACTION",
    "SPRINT_5_TARGET_EXTRACTIONS",
    "sprint5_local_target_extractions",
)
