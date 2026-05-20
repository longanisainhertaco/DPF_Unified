"""Fail-closed physics-closure packets for first-principles DPF runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CLOSURE_SOURCE_REFS = (
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "431-619,1210-1280",
        "role": "hybrid_closure_equations_and_limitations",
    },
    {
        "path": (
            "KnowledgeReference/"
            "unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md"
        ),
        "lines": "277-293,333-369",
        "role": "eos_radiation_material_and_pinch_scope",
    },
    {
        "path": "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json",
        "lines": "57-62",
        "role": "pf1000_two_temperature_heat_flux_ionization_equation_structure",
    },
    {
        "path": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md",
        "lines": "2996-3020,general_formulary_support",
        "role": "plasma_formula_units_transport_thermal_equilibration_radiation_support",
    },
)

REQUIRED_EFFECTS = (
    "eos_thermodynamics",
    "ionization_charge_state",
    "single_two_temperature_energy",
    "electrical_thermal_transport",
    "radiation_losses",
    "impurity_electrode_ablation",
    "hall_flr_kinetic_scope",
    "three_d_instabilities",
    "restrike_anomalous_resistance",
    "beam_target_coupling",
    # S3.5 / WP-N5: electron inertia and fast-ion stopping were unregistered
    # closures (finding F-EI). Registering them here makes "no active closure
    # is uncategorized" enforceable.
    "electron_inertia",
    "stopping_collisions",
)

REQUIRED_CLOSURE_PACKET_CHANNELS = (
    "effect_id",
    "classification",
    "source_equations_or_bound",
    "symbol_map",
    "units",
    "validity_regime",
    "implementation_reference",
    "verification_tests",
    "sensitivity_or_uq",
    "nondominance_or_claim_impact",
    "review_status",
)

# ---------------------------------------------------------------------------
# S3.5 closure registry (WP-N5 closure-registry source audit).
#
# Source-of-truth contract:
#   docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/
#     sprint_3/WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md
#
# Every active OR bounded-out physical closure in the first-principles solver
# gets exactly one explicit registry record. A record carries the eleven
# S3.5-required fields. Where the local KnowledgeReference corpus has no
# source, the record states the source absence explicitly and is BLOCKED.
# No closure may be uncategorized. Candidate closures may run engineering
# cases but cannot support acceptance. Every missing closure blocks its
# specific claim.
#
# Classification vocabulary (S3.5 / WP-N5 section 6.2), exactly one per record:
#   - active_source_backed_candidate : operator implemented; governing form
#     closed by a local KR source; runs engineering cases; cannot accept.
#   - active_blocked                 : operator implemented but the closure as
#     a whole is blocked on a missing source / coefficient / test.
#   - bounded_out_with_source        : closure intentionally excluded with a
#     cited validity bound proving it negligible / out of range.
#   - not_simulated_and_claim_blocking : closure not implemented at all; its
#     absence blocks a class of claims.
#   - external_candidate_not_authority : an external / community formula or a
#     phenomenological fit used only as cross-check / comparator baseline.
# ---------------------------------------------------------------------------
CLOSURE_CLASSIFICATIONS = (
    "active_source_backed_candidate",
    "active_blocked",
    "bounded_out_with_source",
    "not_simulated_and_claim_blocking",
    "external_candidate_not_authority",
)

# Classifications that may NOT back acceptance but ARE permitted to run an
# engineering case. Blocked / not-simulated closures cannot even run.
ENGINEERING_RUNNABLE_CLASSIFICATIONS = (
    "active_source_backed_candidate",
    "bounded_out_with_source",
    "external_candidate_not_authority",
)

REQUIRED_CLOSURE_REGISTRY_RECORD_FIELDS = (
    "closure_id",
    "classification",
    "implemented",
    "source_equations_or_absence",
    "symbol_map",
    "units",
    "validity_regime",
    "implementation_reference",
    "verification_tests",
    "sensitivity_or_uq",
    "claim_impact",
    "review_status",
)

_NRL = "KnowledgeReference/2019nrlplasma-formulary-037290d4.md"
_ALEGRA = (
    "KnowledgeReference/"
    "unlimited-release-printed-september-2009-alegra-hedp-"
    "simulations-of-the-dense-plasma-focus.md"
)
_LEE_SAW_PART1 = (
    "KnowledgeReference/"
    "a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-"
    "part-1-basic-course.md"
)
_BOSCH_HALE = "KnowledgeReference/bosch-hale-1992-fusion-reactivity.md"
_NEON_HALL = (
    "KnowledgeReference/"
    "the-hall-term-and-anomalous-resistivity-effects-in-neon-gas-puff-z-pinches.md"
)

# ---------------------------------------------------------------------------
# Sprint 4 blocker IDs for transport and closure authority gaps.
# These are named blockers referenced in the registry records below and in
# Sprint 4 tests so that every gap has a stable, searchable identifier.
# ---------------------------------------------------------------------------
CLOSURE_BLK_BRAG_001 = (
    "CLOSURE-BLK-BRAG-001-braginskii-1965-not-yet-target-extracted-in-kr"
)
CLOSURE_BLK_D2_EN_001 = (
    "CLOSURE-BLK-D2-EN-001-no-kr-source-for-d2-electron-neutral-transport"
)
CLOSURE_BLK_ION_001 = (
    "CLOSURE-BLK-ION-001-ionization-recombination-nrl-crosscheck-only-no-kr-dpf-authority"
)
CLOSURE_BLK_ANOM_001 = (
    "CLOSURE-BLK-ANOM-001-dpf-regime-anomalous-resistivity-no-kr-source-"
    "only-zpinch-candidate"
)
CLOSURE_BLK_REST_001 = (
    "CLOSURE-BLK-REST-001-dpf-restrike-equation-no-kr-source"
)


def _src(path: str, lines: str, equation: str, role: str) -> dict[str, str]:
    """One exact local KnowledgeReference citation (path + line range + eq)."""
    return {"path": path, "lines": lines, "equation": equation, "role": role}


def _absent(reason: str, missing: tuple[str, ...]) -> dict[str, Any]:
    """Explicit declaration that the local corpus has NO source for a closure."""
    return {
        "local_source_present": False,
        "reason": reason,
        "missing_from_knowledge_reference": list(missing),
    }


# Each entry is the static, source-audited registry record for one closure.
# `implemented` here is the static audit value from WP-N5; the runtime builder
# can only confirm or keep it, never silently promote it.
_CLOSURE_REGISTRY_STATIC: dict[str, dict[str, Any]] = {
    "eos_thermodynamics": {
        "closure_id": "eos_thermodynamics",
        "classification": "not_simulated_and_claim_blocking",
        "implemented": False,
        "source_equations_or_absence": {
            "in_code_form": [
                "p_i = (rho/m_i) k_B T_i  [Pa]",
                "p_e = Z (rho/m_i) k_B T_e  [Pa]",
                "e_i = p_i / ((gamma-1) rho)  [J/kg]",
                "c_s = sqrt(gamma (p_i+p_e)/rho)  [m/s]",
            ],
            "kr_supports_class_decision_only": [
                _src(
                    _ALEGRA,
                    "188,215,348-362",
                    "EOS-class",
                    "alegra_hedp_dpf_uses_sesame_tabular_and_qeos_not_constant_"
                    "gamma_ideal_gas",
                ),
            ],
            "blocking_absence": _absent(
                "no tabular/QEOS EOS closure equation set exists in the local "
                "corpus; the in-code IdealEOS is dimensionally sound but is out "
                "of validity for cold neutral fill, partially ionized rundown, "
                "degenerate pinch core, and radiation-pressure regimes",
                ("qEOS_or_tabular_EOS", "low_density_validity_floor"),
            ),
        },
        "symbol_map": {
            "rho": {"meaning": "mass density", "unit": "kg/m^3"},
            "m_i": {"meaning": "ion mass", "unit": "kg"},
            "k_B": {"meaning": "Boltzmann constant", "unit": "J/K"},
            "T_i": {"meaning": "ion temperature", "unit": "K"},
            "T_e": {"meaning": "electron temperature", "unit": "K"},
            "Z": {"meaning": "charge state", "unit": "-"},
            "gamma": {"meaning": "adiabatic index", "unit": "-"},
            "p": {"meaning": "pressure", "unit": "Pa"},
            "e": {"meaning": "specific internal energy", "unit": "J/kg"},
            "c_s": {"meaning": "sound speed", "unit": "m/s"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": [
                "fully_ionized_non_degenerate_optically_thin",
                "fixed_Z_and_fixed_gamma",
            ],
            "out_of_validity_when": [
                "cold_neutral_fill_before_breakdown",
                "partially_ionized_rundown_sheath",
                "dense_degenerate_pinch_core",
                "radiation_pressure_significant",
                "rho_below_alegra_sesame_lowest_meaningful_density",
            ],
            "regime_flag_field": "eos_regime_out_of_validity",
        },
        "implementation_reference": "src/dpf/fluid/eos.py:32-67 (IdealEOS only)",
        "verification_tests": [
            "test_eos_ideal_gas_dimensional_consistency",
            "test_eos_blocked_without_tabular_packet",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": (
            "whole_shot_thermodynamics_and_pressure_authority_blocked"
        ),
        "review_status": "not_reviewed_for_acceptance",
    },
    "ionization_charge_state": {
        "closure_id": "ionization_charge_state",
        "classification": "active_blocked",
        "implemented": False,
        "source_equations_or_absence": {
            "kr_supports": [
                _src(
                    "KnowledgeReference/"
                    "doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json",
                    "57-62",
                    "ionization-equation-structure",
                    "pf1000_two_temperature_heat_flux_ionization_equation_"
                    "structure",
                ),
            ],
            # Sprint 4 (3c): NRL formulary provides S(Z), alpha_r(Z), alpha_3,
            # and Saha equilibrium (NRL KR L4572-4659, eqs 10-17).  These are
            # CROSS-CHECK ONLY -- the NRL formulary is not a DPF-regime authority.
            # No non-NRL KR source provides deuterium ionization/recombination
            # rate coefficients.  Blocker: CLOSURE-BLK-ION-001.
            #
            # Sprint 4 (3b): D2 electron-neutral momentum-transfer cross-section
            # has no KR source.  Deuterium KR files (compression-dynamics-...,
            # deuterium-hybrid-x-pinch-..., etc.) cover DPF experiments, not
            # electron-neutral transport data.  Blocker: CLOSURE-BLK-D2-EN-001.
            "nrl_crosscheck_only": {
                "role": "nrl_formulary_ionization_recombination_cross_check_not_dpf_authority",
                "source": _src(
                    _NRL,
                    "4572-4659",
                    "eqs.(10-17)",
                    "nrl_ionization_rate_S_Z_radiative_recombination_alpha_r_"
                    "three_body_alpha3_and_saha_equilibrium",
                ),
                "authority": "cross_check_only_not_dpf_closure_authority",
                "blocker_id": CLOSURE_BLK_ION_001,
            },
            "blocking_absence": _absent(
                "no accepted ionization/recombination closure with a "
                "KR-cited rate set, charge-state transport, or "
                "conductivity/EOS charge-state feedback; the NRL formulary "
                "covers S(Z)/alpha_r/alpha_3/Saha (L4572-4659) as a "
                "cross-check only -- not DPF-regime authority; D2 "
                "electron-neutral cross-section has no KR source -- "
                f"{CLOSURE_BLK_D2_EN_001}; "
                f"{CLOSURE_BLK_ION_001}",
                (
                    "accepted_ionization_recombination_model",
                    "accepted_charge_state_transport",
                    "accepted_neutral_particle_source_coupling",
                    CLOSURE_BLK_D2_EN_001,
                    CLOSURE_BLK_ION_001,
                ),
            ),
        },
        "symbol_map": {
            "n_e": {"meaning": "electron density", "unit": "m^-3"},
            "n_0": {"meaning": "neutral density", "unit": "m^-3"},
            "Z": {"meaning": "mean charge state", "unit": "-"},
            "T_e": {"meaning": "electron temperature", "unit": "K"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": ["source_backed_rate_set_attached"],
            "out_of_validity_when": ["no_kr_cited_ionization_recombination_set"],
            "regime_flag_field": "ionization_charge_state_out_of_validity",
        },
        "implementation_reference": (
            "src/dpf/fields/ionization_transport.py (candidate channels only)"
        ),
        "verification_tests": [
            "test_ionization_three_body_rate_si_conversion",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": "breakdown_sheath_and_resistivity_authority_blocked",
        "review_status": "not_reviewed_for_acceptance",
    },
    "single_two_temperature_energy": {
        "closure_id": "single_two_temperature_energy",
        "classification": "active_blocked",
        "implemented": False,
        "source_equations_or_absence": {
            "kr_supports": [
                _src(
                    _NRL,
                    "2996-3020",
                    "thermal-equilibration",
                    "two_temperature_collisional_equilibration_form_support",
                ),
            ],
            "blocking_absence": _absent(
                "two-temperature energy split lacks an accepted electron "
                "heat-flux closure, accepted electron-ion collisional "
                "coupling, and temperature-diagnostic validation",
                (
                    "accepted_electron_heat_flux",
                    "accepted_electron_ion_collisional_coupling",
                    "temperature_diagnostic_validation",
                ),
            ),
        },
        "symbol_map": {
            "T_e": {"meaning": "electron temperature", "unit": "K"},
            "T_i": {"meaning": "ion temperature", "unit": "K"},
            "q_e": {"meaning": "electron heat flux", "unit": "W/m^2"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": ["accepted_heat_flux_and_equilibration_attached"],
            "out_of_validity_when": ["heat_flux_or_equilibration_unaccepted"],
            "regime_flag_field": "two_temperature_energy_out_of_validity",
        },
        "implementation_reference": "src/dpf/fluid/two_temperature.py",
        "verification_tests": [
            "test_two_temperature_energy_blocked_without_heat_flux",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": "hall_pressure_and_yield_authority_blocked",
        "review_status": "not_reviewed_for_acceptance",
    },
    "electrical_thermal_transport": {
        "closure_id": "electrical_thermal_transport",
        "classification": "active_source_backed_candidate",
        "implemented": True,
        "source_equations_or_absence": {
            "kr_supports": [
                _src(
                    _NRL,
                    "3024-3065",
                    "Coulomb-logarithm-definition-and-e-i-branch",
                    "coulomb_logarithm_lambda_ee_and_lambda_ei",
                ),
                _src(
                    _NRL,
                    "2701-2704",
                    "transverse-Spitzer-resistivity",
                    "eta_perp_1p03e-2_Z_lnLambda_T_pow_minus_3_2_ohm_cm",
                ),
                _src(
                    _NRL,
                    "3384-3411",
                    "weakly-ionized-collision-frequency-and-conductivity",
                    "nu_alpha_n0_sigma_sqrt_kT_over_m_and_sigma_alpha",
                ),
            ],
            # Sprint 4 (3a): Braginskii 1965 PDF exists on disk at
            # archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf
            # but has NO KnowledgeReference extract (`ls KR | grep -i braginskii`
            # returned empty).  The direct coefficient table cannot serve as
            # closure authority until a KR target extraction is completed.
            # Blocker: CLOSURE-BLK-BRAG-001
            "missing_parameter_absence": _absent(
                "the in-code Braginskii alpha(Z)/delta_e(Z) Z-dependent "
                "correction coefficients are not tabulated in the local NRL "
                "extract; Braginskii (1965) Table 1 is not in the corpus; "
                "the PDF exists at archive_reference_OLD/references/papers/"
                "mhd-numerics/braginskii_1965.pdf but has no KR extract -- "
                f"{CLOSURE_BLK_BRAG_001}",
                (
                    "braginskii_alpha_Z_and_delta_e_Z_table",
                    CLOSURE_BLK_BRAG_001,
                ),
            ),
        },
        "symbol_map": {
            "lambda": {"meaning": "Coulomb logarithm ln Lambda", "unit": "-"},
            "n_e": {"meaning": "electron density (cm^-3 in NRL eq.)",
                    "unit": "m^-3"},
            "Z": {"meaning": "charge state", "unit": "-"},
            "T_e": {"meaning": "electron temperature (eV in NRL eq.)",
                    "unit": "K"},
            "eta_perp": {"meaning": "transverse resistivity", "unit": "Ohm m"},
            "n_0": {"meaning": "neutral density", "unit": "m^-3"},
            "sigma": {"meaning": "electron-neutral cross-section",
                      "unit": "m^2"},
        },
        "units": "CGS-eV-converted",
        "validity_regime": {
            "valid_when": [
                "coulomb_log_lambda_much_greater_than_1",
                "relative_drift_u2_much_less_than_kT_over_m",
                "anomalous_microinstability_transport_negligible",
            ],
            "out_of_validity_when": [
                "lambda_near_1_strong_coupling",
                "u2_comparable_to_kT_over_m",
                "anomalous_transport_active",
            ],
            "regime_flag_field": "strong_coupling_out_of_validity",
            "regime_gate_source": [
                _src(
                    _NRL,
                    "3036-3038",
                    "coulomb-log-validity-edge",
                    "theory_good_to_10_percent_and_fails_when_lambda_near_1",
                ),
                _src(
                    _NRL,
                    "3379-3383",
                    "classical-transport-validity-criteria-3-5-6",
                    "classical_transport_valid_only_when_lambda_much_gt_1_"
                    "and_drifts_small_and_anomalous_negligible",
                ),
            ],
        },
        "implementation_reference": (
            "src/dpf/collision/spitzer.py:33-65,192-242; "
            "src/dpf/fields/conductivity.py"
        ),
        "verification_tests": [
            "test_spitzer_resistivity_order_of_magnitude",
            "test_collision_coulomb_log_branch_selection",
            "test_coulomb_log_floored_at_2",
            "test_plasmapy_coupling_regime_gate",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": (
            "field_current_coupling_remains_engineering_candidate"
        ),
        "review_status": "not_reviewed_for_acceptance",
    },
    "radiation_losses": {
        "closure_id": "radiation_losses",
        "classification": "active_blocked",
        "implemented": False,
        "source_equations_or_absence": {
            "kr_supports_volumetric_loss_term_only": [
                _src(
                    _NRL,
                    "4732-4736",
                    "eq.(30)",
                    "bremsstrahlung_free_free_hydrogen_like_volumetric_loss",
                ),
                _src(
                    _NRL,
                    "4737-4740",
                    "eq.(31)",
                    "bremsstrahlung_optical_depth",
                ),
                _src(
                    _NRL,
                    "4749-4755",
                    "eq.(33)",
                    "recombination_free_bound_radiation",
                ),
                _src(
                    _NRL,
                    "4756-4758",
                    "eq.(34)",
                    "cyclotron_radiation_volumetric_loss",
                ),
            ],
            "blocking_absence": _absent(
                "the bremsstrahlung/cyclotron volumetric LOSS terms are "
                "NRL-grounded, but the radiation loss-and-transport closure "
                "as a whole has no KR-cited Rosseland/Kramers opacity model, "
                "no opacity/diffusion decision, and no KR-cited line-emission "
                "model; transport.py self-flags the FLD opacity source as "
                "missing",
                (
                    "rosseland_kramers_opacity_closure",
                    "opacity_or_diffusion_decision",
                    "kr_cited_line_radiation_model",
                    "radiated_energy_ledger",
                ),
            ),
        },
        "symbol_map": {
            "N_e": {"meaning": "electron number density", "unit": "cm^-3"},
            "N_i": {"meaning": "ion number density", "unit": "cm^-3"},
            "T_e": {"meaning": "electron temperature", "unit": "eV"},
            "Z": {"meaning": "charge state", "unit": "-"},
            "B": {"meaning": "magnetic field (NRL CGS Gauss)", "unit": "T"},
            "tau": {"meaning": "bremsstrahlung optical depth", "unit": "-"},
            "P": {"meaning": "volumetric radiated power", "unit": "W/m^3"},
        },
        "units": "CGS-eV-converted",
        "validity_regime": {
            "valid_when": [
                "optically_thin_tau_much_less_than_1_for_eq30_loss_term",
            ],
            "out_of_validity_when": [
                "tau_approaching_1_optically_thick",
                "line_dominated_high_Z_radiation",
            ],
            "regime_flag_field": "radiation_optical_depth_out_of_validity",
        },
        "implementation_reference": (
            "src/dpf/radiation/bremsstrahlung.py (volumetric loss term only)"
        ),
        "verification_tests": [
            "test_bremsstrahlung_coeff_is_si_not_cgs",
            "test_radiation_brem_matches_nrl_eq30",
            "test_radiation_blocked_without_opacity_decision",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": "radiating_gas_or_high_z_claims_blocked",
        "review_status": "not_reviewed_for_acceptance",
    },
    "impurity_electrode_ablation": {
        "closure_id": "impurity_electrode_ablation",
        "classification": "active_blocked",
        "implemented": False,
        "source_equations_or_absence": {
            "in_code_form": [
                "P_ohmic = eta J^2  [W/m^3]",
                "S_rho = efficiency * P_ohmic  [kg/(m^3 s)]",
            ],
            "blocking_absence": _absent(
                "the Ohmic-heating driver eta*J^2 is dimensionally sound but "
                "the ablation conversion efficiency [kg/J] has NO local "
                "KnowledgeReference source; ablation.py self-declares "
                "ablation_efficiency_source_packet_missing and its references "
                "(Bruzzone, Vikhrev, Lee & Serban) are docstring-only",
                (
                    "kr_cited_ablation_efficiency_with_fluence_dependence",
                    "impurity_transport_model",
                    "electrode_material_uq",
                ),
            ),
        },
        "symbol_map": {
            "eta": {"meaning": "resistivity", "unit": "Ohm m"},
            "J": {"meaning": "current density", "unit": "A/m^2"},
            "P_ohmic": {"meaning": "volumetric Ohmic power", "unit": "W/m^3"},
            "efficiency": {"meaning": "ablation yield", "unit": "kg/J"},
            "S_rho": {"meaning": "volumetric mass source",
                      "unit": "kg/(m^3 s)"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": ["moderate_power_density_1e8_to_1e11_W_per_m2"],
            "out_of_validity_when": [
                "high_fluence_plasma_shielding_reduces_efficiency",
                "no_kr_source_bounds_this_range",
            ],
            "regime_flag_field": "ablation_out_of_validity",
        },
        "implementation_reference": (
            "src/dpf/atomic/ablation.py:122-192 (impurity-source scaffold)"
        ),
        "verification_tests": [
            "test_ablation_blocked_without_efficiency_source",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": (
            "waveform_pinch_radiation_neutron_impurity_effects_blocked"
        ),
        "review_status": "not_reviewed_for_acceptance",
    },
    "hall_flr_kinetic_scope": {
        "closure_id": "hall_flr_kinetic_scope",
        "classification": "active_blocked",
        "implemented": False,
        "source_equations_or_absence": {
            "kr_supports": [
                _src(
                    "KnowledgeReference/"
                    "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-"
                    "acb71fa9.md",
                    "431-619",
                    "hybrid-closure-equations",
                    "hybrid_hall_flr_kinetic_closure_equations_and_limitations",
                ),
            ],
            "blocking_absence": _absent(
                "the Hall term runs as a candidate channel but the "
                "FLR-validity / kinetic-handoff scope lacks an accepted "
                "electron-temperature authority and a same-scope kinetic "
                "interval review",
                (
                    "electron_temperature_authority",
                    "flr_validity_or_handoff_review",
                    "kinetic_interval_review",
                ),
            ),
        },
        "symbol_map": {
            "J": {"meaning": "current density", "unit": "A/m^2"},
            "B": {"meaning": "magnetic field", "unit": "T"},
            "n_e": {"meaning": "electron density", "unit": "m^-3"},
            "rho_i": {"meaning": "ion Larmor radius", "unit": "m"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": ["flr_handoff_reviewed_same_scope"],
            "out_of_validity_when": ["kinetic_interval_unreviewed"],
            "regime_flag_field": "hall_flr_kinetic_out_of_validity",
        },
        "implementation_reference": (
            "src/dpf/fields/hybrid_loop.py, hybrid_stepper.py (Hall candidate)"
        ),
        "verification_tests": [
            "test_hall_stays_blocked_without_include_hall",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": "late_pinch_and_acceleration_authority_blocked",
        "review_status": "not_reviewed_for_acceptance",
    },
    "three_d_instabilities": {
        "closure_id": "three_d_instabilities",
        "classification": "active_blocked",
        "implemented": True,
        "source_equations_or_absence": {
            "blocking_absence": _absent(
                "the 3D instability operator runs but lacks accepted m-mode "
                "evidence and a same-scope 3D instability packet; no KR "
                "source closes the kink/fragmentation lifetime claim",
                (
                    "accepted_m_mode_evidence",
                    "same_scope_3d_instability_packet",
                ),
            ),
        },
        "symbol_map": {
            "m": {"meaning": "azimuthal mode number", "unit": "-"},
            "gamma_inst": {"meaning": "instability growth rate", "unit": "1/s"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": ["same_scope_m_mode_evidence_attached"],
            "out_of_validity_when": ["no_same_scope_3d_instability_packet"],
            "regime_flag_field": "three_d_instability_out_of_validity",
        },
        "implementation_reference": "src/dpf/ (3D instability operator)",
        "verification_tests": [
            "test_three_d_instabilities_registered_active_blocked",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": "kink_fragmentation_and_lifetime_authority_blocked",
        "review_status": "not_reviewed_for_acceptance",
    },
    # restrike_anomalous_resistance shares one REQUIRED_EFFECTS key but covers
    # two physically distinct closures. Per WP-N5 section 6.1 they are kept
    # individually classified and individually testable via sub_closures.
    "restrike_anomalous_resistance": {
        "closure_id": "restrike_anomalous_resistance",
        "classification": "not_simulated_and_claim_blocking",
        "implemented": False,
        "source_equations_or_absence": {
            "blocking_absence": _absent(
                "shared registry effect for two distinct closures; see "
                "sub_closures for the per-closure source audit",
                ("restrike_model", "anomalous_resistivity_alpha_and_threshold"),
            ),
        },
        "symbol_map": {
            "eta_anom": {"meaning": "anomalous resistivity", "unit": "Ohm m"},
            "I": {"meaning": "circuit current", "unit": "A"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": ["per_sub_closure"],
            "out_of_validity_when": ["per_sub_closure"],
            "regime_flag_field": "restrike_anomalous_resistance_out_of_validity",
        },
        "implementation_reference": (
            "src/dpf/turbulence/anomalous.py (anomalous); restrike not "
            "implemented"
        ),
        "verification_tests": [
            "test_anomalous_and_restrike_claim_rejected",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": "current_dip_and_post_pinch_claims_blocked",
        "review_status": "not_reviewed_for_acceptance",
        "sub_closures": {
            "anomalous_resistance": {
                "closure_id": "anomalous_resistance",
                "classification": "active_blocked",
                "implemented": True,
                "source_equations_or_absence": {
                    "kr_supports_functional_form_only": [
                        _src(
                            _NRL,
                            "2706-2710",
                            "anomalous-ion-sound-collision-rate",
                            "nu_star_omega_pe_W_over_kT_5p64e4_ne_sqrt_form",
                        ),
                        _src(
                            _NRL,
                            "3382-3383",
                            "classical-transport-criterion-6",
                            "classical_transport_valid_only_when_anomalous_"
                            "transport_negligible",
                        ),
                    ],
                    # Sprint 4 (3d): Neon gas-puff Z-pinch LHDI candidate.
                    # The neon paper (PERSEUS/COBRA XMHD runs) provides
                    # eta* = m_e*nu_eff/(n_e*e^2), with nu_eff from
                    # Davidson-Gladd LHDI theory, capped at B/(n_e*e) when
                    # nu_eff >= Omega_e (KR L185-266, eq.1).  Symbol map:
                    #   eta*   [Ohm m]  anomalous resistivity
                    #   alpha  [-]      order-unity saturation parameter
                    #   v_de   [m/s]    perpendicular electron drift
                    #   v_i    [m/s]    ion thermal speed
                    #   Omega_e [rad/s] electron cyclotron frequency
                    #   B      [T]      magnetic field
                    #   n_e    [m^-3]   electron density
                    # Validity: neon gas-puff Z-pinch, XMHD, magnetized
                    # electrons (nu_eff < Omega_e required).
                    # SCOPE: neon Z-PINCH ONLY -- not DPF-regime authority.
                    # This formula is a CANDIDATE, not a DPF transport closure.
                    # Blocker: CLOSURE-BLK-ANOM-001.
                    "candidate_zpinch_formula_not_dpf_authority": {
                        "blocker_id": CLOSURE_BLK_ANOM_001,
                        "source": _src(
                            _NEON_HALL,
                            "194-266",
                            "eq.(1)",
                            "lhdi_driven_resistivity_eta_star_davidson_gladd_"
                            "neon_gaspuff_zpinch_candidate_not_dpf_authority",
                        ),
                        "formula_si": (
                            "eta_star = m_e * nu_eff / (n_e * e^2), "
                            "nu_eff ~ sqrt(pi/2) * sqrt(m_e/m_i) * alpha * "
                            "(v_de/v_i)^2 / (1 + (v_de/v_i)^2) / (eps_0 * Omega_e), "
                            "capped at B / (n_e * e) when nu_eff >= Omega_e"
                        ),
                        "symbol_map": {
                            "eta_star": {"meaning": "anomalous resistivity", "unit": "Ohm m"},
                            "m_e": {"meaning": "electron mass", "unit": "kg"},
                            "m_i": {"meaning": "ion mass", "unit": "kg"},
                            "nu_eff": {"meaning": "LHDI effective collision frequency", "unit": "rad/s"},
                            "n_e": {"meaning": "electron density", "unit": "m^-3"},
                            "e": {"meaning": "elementary charge", "unit": "C"},
                            "alpha": {"meaning": "order-unity saturation parameter", "unit": "-"},
                            "v_de": {"meaning": "perp electron drift speed", "unit": "m/s"},
                            "v_i": {"meaning": "ion thermal speed", "unit": "m/s"},
                            "Omega_e": {"meaning": "electron cyclotron frequency", "unit": "rad/s"},
                            "B": {"meaning": "magnetic field", "unit": "T"},
                        },
                        "validity": (
                            "neon_gaspuff_z_pinch_xmhd_magnetized_electrons_only"
                        ),
                        "dpf_applicability": "not_established_no_kr_source",
                        "authority": "zpinch_candidate_cross_check_not_dpf_closure",
                    },
                    "blocking_absence": _absent(
                        "the NRL row gives the functional structure "
                        "(anomalous rate proportional to omega_pe) but the DPF "
                        "turbulence parameter alpha ~ 0.01-0.1 and the "
                        "threshold-model selection (ion-acoustic, LHDI, "
                        "Buneman, CIV) are NOT KR-closed; module self-declares "
                        "microinstability_source_packets_missing; the neon "
                        "gas-puff Z-pinch LHDI formula (KR L194-266 eq.1) is "
                        "a non-DPF Z-pinch candidate only and cannot establish "
                        "same-scope DPF transport authority by itself -- "
                        f"{CLOSURE_BLK_ANOM_001}",
                        (
                            "kr_cited_alpha_saturation_amplitude",
                            "kr_cited_threshold_model_selection",
                            "kr_cited_civ_v_crit_table",
                            CLOSURE_BLK_ANOM_001,
                        ),
                    ),
                },
                "symbol_map": {
                    "alpha": {"meaning": "turbulence parameter", "unit": "-"},
                    "omega_pe": {"meaning": "electron plasma frequency",
                                 "unit": "rad/s"},
                    "n_e": {"meaning": "electron density", "unit": "m^-3"},
                    "v_d": {"meaning": "electron drift speed", "unit": "m/s"},
                    "eta_anom": {"meaning": "anomalous resistivity",
                                 "unit": "Ohm m"},
                },
                "units": "SI",
                "validity_regime": {
                    "valid_when": [
                        "microinstability_threshold_exceeded_with_kr_source",
                    ],
                    "out_of_validity_when": [
                        "alpha_band_unsourced",
                        "threshold_model_unsourced",
                        "classical_transport_simultaneously_assumed",
                    ],
                    "regime_flag_field": "anomalous_resistance_out_of_validity",
                },
                "implementation_reference": (
                    "src/dpf/turbulence/anomalous.py:312-329 "
                    "(microinstability resistivity scaffold)"
                ),
                "verification_tests": [
                    "test_anomalous_and_restrike_claim_rejected",
                ],
                "sensitivity_or_uq": "missing",
                "claim_impact": (
                    "anomalous_resistance_authority_blocked"
                ),
                "review_status": "not_reviewed_for_acceptance",
            },
            "restrike": {
                "closure_id": "restrike",
                "classification": "not_simulated_and_claim_blocking",
                "implemented": False,
                "source_equations_or_absence": {
                    # Sprint 4 (3e): `grep -nE 'restrike|secondary.pinch|
                    # post.pinch.recovery' KR/*.md` confirms restrike appears
                    # only as experimental context (current-dip side-effect)
                    # in Lee/Saw, Beresnyak, Faeton-I, MNJI, and optimization
                    # papers.  No DPF-specific restrike physics equation or
                    # governing model exists in the corpus.
                    # Blocker: CLOSURE-BLK-REST-001.
                    "blocking_absence": _absent(
                        "there is no restrike (post-pinch current-dip "
                        "recovery) closure equation in src/dpf and none in "
                        "KnowledgeReference; 'restrike' appears only as "
                        "experimental context in comparator papers (Lee/Saw "
                        "L5232, Beresnyak L113/228, Faeton-I L75-89, MNJI "
                        "L318-584, optimization L94-328); no governing "
                        "restrike physics equation extracted from any KR "
                        "source -- "
                        f"{CLOSURE_BLK_REST_001}",
                        (
                            "restrike_post_pinch_resistance_recovery_model",
                            CLOSURE_BLK_REST_001,
                        ),
                    ),
                },
                "symbol_map": {
                    "I": {"meaning": "post-pinch circuit current", "unit": "A"},
                    "dI_dt": {"meaning": "current derivative", "unit": "A/s"},
                },
                "units": "SI",
                "validity_regime": {
                    "valid_when": [],
                    "out_of_validity_when": ["not_simulated"],
                    "regime_flag_field": "restrike_out_of_validity",
                },
                "implementation_reference": None,
                "verification_tests": [
                    "test_anomalous_and_restrike_claim_rejected",
                ],
                "sensitivity_or_uq": "missing",
                "claim_impact": (
                    "current_dip_and_post_pinch_claims_blocked"
                ),
                "review_status": "not_reviewed_for_acceptance",
            },
        },
    },
    "beam_target_coupling": {
        "closure_id": "beam_target_coupling",
        "classification": "external_candidate_not_authority",
        "implemented": False,
        "source_equations_or_absence": {
            "kr_supports_phenomenological_form_only": [
                _src(
                    _LEE_SAW_PART1,
                    "5109-5145",
                    "eq.(1)",
                    "lee_saw_beam_target_single_constant_phenomenological_"
                    "form_and_Cn_calibration",
                ),
                _src(
                    _BOSCH_HALE,
                    "1-116",
                    "Table-IV",
                    "dd_fusion_cross_section_bosch_hale_1992_parametric_fit",
                ),
            ],
            "blocking_absence": _absent(
                "the Lee & Saw form is an empirical single-constant "
                "comparator baseline only; there is NO mechanism-separated "
                "thermonuclear-vs-beam-target closure, no ion-distribution "
                "transport, and no stopping closure (see stopping_collisions)",
                (
                    "mechanism_separated_thermonuclear_vs_beam_target_closure",
                    "ion_distribution_transport_stopping",
                    "spectrum_anisotropy_detector_response",
                    "beam_target_uq",
                ),
            ),
        },
        "symbol_map": {
            "Y_bt": {"meaning": "beam-target neutron yield", "unit": "neutrons"},
            "Cn": {"meaning": "calibrated proportionality constant",
                   "unit": "SI"},
            "n_i": {"meaning": "ion density", "unit": "m^-3"},
            "I_pinch": {"meaning": "pinch current", "unit": "A"},
            "z_p": {"meaning": "pinch length", "unit": "m"},
            "sigma": {"meaning": "DD cross-section", "unit": "m^2"},
            "E_beam": {"meaning": "beam ion energy", "unit": "eV"},
            "V_max": {"meaning": "peak voltage", "unit": "V"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": [
                "bosch_hale_fit_0p5_to_5000_keV_cross_section_only",
            ],
            "out_of_validity_when": [
                "used_as_mechanism_authority_not_comparator",
                "E_cm_outside_0p5_to_5000_keV",
            ],
            "regime_flag_field": "beam_target_out_of_validity",
        },
        "implementation_reference": (
            "src/dpf/diagnostics/beam_target.py:1-16,76-90 "
            "(baseline/comparator only)"
        ),
        "verification_tests": [
            "test_beam_target_bosch_hale_in_valid_range",
            "test_beam_target_stays_blocked_without_kinetic_yield",
            "test_stopping_blocked_blocks_beam_target",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": "total_neutron_yield_authority_blocked",
        "review_status": "not_reviewed_for_acceptance",
    },
    "electron_inertia": {
        "closure_id": "electron_inertia",
        "classification": "not_simulated_and_claim_blocking",
        "implemented": False,
        "source_equations_or_absence": {
            "blocking_absence": _absent(
                "the generalized-Ohm electron-inertia term (m_e dJ/dt) and "
                "the electron skin depth c/omega_pe are standard but NO local "
                "KnowledgeReference file presents the closure as a citable "
                "equation; electron inertia appears only in diagnostics, "
                "never as a closure operator (WP-N5 finding F-EI)",
                (
                    "generalized_ohm_electron_inertia_closure_equation",
                    "skin_depth_resolution_gate",
                ),
            ),
        },
        "symbol_map": {
            "m_e": {"meaning": "electron mass", "unit": "kg"},
            "J": {"meaning": "current density", "unit": "A/m^2"},
            "n_e": {"meaning": "electron density", "unit": "m^-3"},
            "omega_pe": {"meaning": "electron plasma frequency",
                         "unit": "rad/s"},
            "d_e": {"meaning": "electron skin depth c/omega_pe", "unit": "m"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": [
                "electron_skin_depth_resolved_by_grid",
                "timescales_near_one_over_omega_pe",
            ],
            "out_of_validity_when": [
                "whole_shot_mhd_deck_treats_d_e_as_sub_grid",
                "no_kr_source_to_bound_the_term",
            ],
            "regime_flag_field": "electron_inertia_out_of_validity",
        },
        "implementation_reference": None,
        "verification_tests": [
            "test_electron_inertia_registered_and_blocked",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": (
            "generalized_ohm_electron_inertia_and_skin_depth_claims_blocked"
        ),
        "review_status": "not_reviewed_for_acceptance",
    },
    "stopping_collisions": {
        "closure_id": "stopping_collisions",
        "classification": "not_simulated_and_claim_blocking",
        "implemented": False,
        "source_equations_or_absence": {
            "blocking_absence": _absent(
                "no fast-ion stopping-power closure operator exists in "
                "src/dpf and no Bethe / plasma stopping-power equation in "
                "KnowledgeReference was located as a citable closure; "
                "stopping is only named as a missing channel inside "
                "beam_target_coupling",
                ("kr_cited_bethe_or_plasma_stopping_power_closure",),
            ),
        },
        "symbol_map": {
            "f_E": {"meaning": "fast-ion energy distribution", "unit": "-"},
            "dE_dx": {"meaning": "stopping power", "unit": "eV/m"},
            "n_e": {"meaning": "background electron density", "unit": "m^-3"},
            "T_e": {"meaning": "background electron temperature", "unit": "K"},
        },
        "units": "SI",
        "validity_regime": {
            "valid_when": [],
            "out_of_validity_when": ["not_simulated"],
            "regime_flag_field": "stopping_collisions_out_of_validity",
        },
        "implementation_reference": None,
        "verification_tests": [
            "test_stopping_blocked_blocks_beam_target",
        ],
        "sensitivity_or_uq": "missing",
        "claim_impact": (
            "fast_ion_stopping_and_beam_target_neutron_authority_blocked"
        ),
        "review_status": "not_reviewed_for_acceptance",
    },
}


def _validate_registry_record(closure_id: str, record: Mapping[str, Any]) -> None:
    """Fail closed if a registry record is missing a required S3.5 field or
    carries an unknown classification. A malformed registry is a typed bug."""
    for field in REQUIRED_CLOSURE_REGISTRY_RECORD_FIELDS:
        if field not in record:
            raise ValueError(
                f"closure registry record '{closure_id}' is missing required "
                f"S3.5 field '{field}'"
            )
    classification = record["classification"]
    if classification not in CLOSURE_CLASSIFICATIONS:
        raise ValueError(
            f"closure registry record '{closure_id}' has uncategorized "
            f"classification '{classification}'; allowed: "
            f"{', '.join(CLOSURE_CLASSIFICATIONS)}"
        )


def build_closure_registry(
    *,
    runtime_implemented: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Return the explicit S3.5 per-closure registry.

    Each record carries the eleven S3.5-required fields. The registry is the
    fail-closed source-of-truth contract from
    ``WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md``: no closure is uncategorized,
    candidate closures cannot support acceptance, and every missing closure
    blocks its specific claim.

    ``runtime_implemented`` lets the runtime confirm a closure operator is
    actually wired. It may only DOWNGRADE ``implemented`` to ``False`` (an
    operator the audit thought present but the runtime cannot find), never
    promote a statically-blocked closure to ``True``. This keeps the registry
    fail-closed.
    """
    runtime_flags = dict(runtime_implemented or {})
    closures: dict[str, Any] = {}
    classification_counts: dict[str, int] = {key: 0 for key in CLOSURE_CLASSIFICATIONS}
    uncategorized: list[str] = []
    blocked_claims: dict[str, str] = {}
    engineering_runnable: list[str] = []
    acceptance_supporting: list[str] = []

    for closure_id, static_record in _CLOSURE_REGISTRY_STATIC.items():
        _validate_registry_record(closure_id, static_record)
        record = _materialize_registry_record(closure_id, static_record, runtime_flags)
        closures[closure_id] = record

        classification = record["classification"]
        classification_counts[classification] += 1
        if classification not in CLOSURE_CLASSIFICATIONS:  # pragma: no cover
            uncategorized.append(closure_id)
        if classification in ENGINEERING_RUNNABLE_CLASSIFICATIONS:
            engineering_runnable.append(closure_id)
        if classification in {
            "active_blocked",
            "not_simulated_and_claim_blocking",
        }:
            blocked_claims[closure_id] = record["claim_impact"]
        # No classification may support acceptance in Sprint 3.
        if record["can_support_first_principles_acceptance"]:  # pragma: no cover
            acceptance_supporting.append(closure_id)

    return {
        "status": "closure_registry_source_audit_not_validation",
        "registry_contract": (
            "docs/external_team_submissions/"
            "2026_05_18_three_sprint_blocker_packet/sprint_3/"
            "WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md"
        ),
        "required_record_fields": list(REQUIRED_CLOSURE_REGISTRY_RECORD_FIELDS),
        "classification_vocabulary": list(CLOSURE_CLASSIFICATIONS),
        "closures": closures,
        "registered_closure_ids": list(closures.keys()),
        "classification_counts": classification_counts,
        "uncategorized_active_closures": uncategorized,
        "engineering_runnable_closures": sorted(engineering_runnable),
        "blocked_claims": blocked_claims,
        "acceptance_supporting_closures": acceptance_supporting,
        "registry_policy": {
            "no_active_closure_uncategorized": not uncategorized,
            "candidate_closures_can_run_engineering_cases": True,
            "candidate_closures_can_support_acceptance": False,
            "every_missing_closure_blocks_its_claim": True,
            "plasmapy_can_promote_or_reject_closure": False,
        },
        "can_support_first_principles_acceptance": False,
    }


def _materialize_registry_record(
    closure_id: str,
    static_record: Mapping[str, Any],
    runtime_flags: Mapping[str, bool],
) -> dict[str, Any]:
    """Copy a static registry record and apply the fail-closed runtime flag."""
    record = {key: _deep_copy(value) for key, value in static_record.items()}
    static_implemented = bool(record["implemented"])
    if closure_id in runtime_flags:
        # Runtime confirmation may only confirm or downgrade, never promote.
        record["implemented"] = static_implemented and bool(
            runtime_flags[closure_id]
        )
        record["runtime_implementation_confirmed"] = closure_id in runtime_flags
    else:
        record["runtime_implementation_confirmed"] = False
    record["review_status"] = "not_reviewed_for_acceptance"
    record["can_support_first_principles_acceptance"] = False
    record["missing_parameters"] = _registry_missing_parameters(record)
    if "sub_closures" in record:
        for sub_id, sub in record["sub_closures"].items():
            _validate_registry_record(sub_id, sub)
            sub["review_status"] = "not_reviewed_for_acceptance"
            sub["can_support_first_principles_acceptance"] = False
            sub["missing_parameters"] = _registry_missing_parameters(sub)
    return record


def _registry_missing_parameters(record: Mapping[str, Any]) -> list[str]:
    """Collect every explicitly-declared missing-source item for a record."""
    missing: list[str] = []
    source = record.get("source_equations_or_absence", {})
    for value in source.values():
        if isinstance(value, Mapping) and value.get("local_source_present") is False:
            missing.extend(value.get("missing_from_knowledge_reference", ()))
    return missing


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_copy(item) for item in value]
    return value


# NRL Plasma Formulary citations for the classical-transport validity edge.
# These back the strong-coupling regime gate: classical Spitzer/collision
# transport is OUT of its validity range where the Coulomb logarithm is small.
PLASMAPY_REGIME_GATE_SOURCE_REFS = (
    _src(
        _NRL,
        "3036-3038",
        "coulomb-log-validity-edge",
        "coulomb_log_theory_good_to_10_percent_and_fails_when_lambda_near_1",
    ),
    _src(
        _NRL,
        "3379-3383",
        "classical-transport-validity-criteria-3-5-6",
        "classical_transport_valid_only_when_coulomb_log_lambda_much_gt_1",
    ),
)


def build_plasmapy_closure_regime_gate(
    community_formula_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the S3.5 PlasmaPy rule to the closure registry.

    The rule (S3.5 / WP-N5 section 5.3):

    - PlasmaPy is a cross-check ONLY, never a source authority.
    - A missing PlasmaPy audit cannot promote OR reject a local-source closure.
    - A PlasmaPy disagreement outside tolerance sets review-required telemetry;
      acceptance stays false until local-source review resolves it.
    - The strong-coupling Coulomb-logarithm regime is a SURFACED
      ``bounded_out_with_source`` gate citing the NRL formulary validity edge,
      never a silent floor.

    ``community_formula_audit`` is the optional PlasmaPy audit packet built by
    :func:`dpf.first_principles.plasmapy_audit.build_plasmapy_formulary_audit_packet`.
    The audit packet carries the strong-coupling detection
    (``strong_coupling_regime``); this function turns it into the registry-side
    bounded-out gate and review telemetry. It can never accept or reject.
    """
    audit = community_formula_audit
    audit_present = audit is not None and _community_formula_audit_available(audit)

    strong_coupling = (
        dict(audit.get("strong_coupling_regime", {})) if audit_present else {}
    )
    coupling_warning = bool(strong_coupling.get("coupling_warning_raised", False))
    out_of_validity = bool(
        strong_coupling.get("strong_coupling_out_of_validity", False)
    )

    # The strong-coupling bounded-out gate. When PlasmaPy raises a
    # CouplingWarning (or ln Lambda <= ~2), the classical Spitzer/collision
    # closure is out of its validity range: it is bounded out WITH the NRL
    # source, surfaced, never silently floored.
    bounded_out_gate = {
        "closure_id": "electrical_thermal_transport",
        "classification": "bounded_out_with_source",
        "triggered": out_of_validity or coupling_warning,
        "trigger_reason": (
            "plasmapy_coupling_warning_or_low_coulomb_log_strong_coupling"
            if (out_of_validity or coupling_warning)
            else "weak_coupling_classical_transport_within_validity"
        ),
        "regime_flag_field": "strong_coupling_out_of_validity",
        "source_equations_or_bound": list(PLASMAPY_REGIME_GATE_SOURCE_REFS),
        "silent_floor_forbidden": True,
        "strong_coupling_regime": strong_coupling,
    }

    # Disagreement-outside-tolerance review telemetry.
    quantities = audit.get("quantities", {}) if audit_present else {}
    outside_tolerance = sorted(
        name
        for name, record in quantities.items()
        if isinstance(record, Mapping)
        and "outside_tolerance" in str(record.get("status", ""))
    )
    review_required = bool(outside_tolerance)

    audit_status = (
        "plasmapy_regime_cross_check_present_not_authority"
        if audit_present
        else "plasmapy_regime_cross_check_absent_does_not_promote_or_reject"
    )

    return {
        "status": audit_status,
        "role": "community_formula_cross_check_not_source_authority",
        "audit_present": audit_present,
        "plasmapy_can_promote_closure": False,
        "plasmapy_can_reject_closure": False,
        "missing_audit_promotes_or_rejects_closure": False,
        "strong_coupling_bounded_out_gate": bounded_out_gate,
        "disagreement_outside_tolerance_quantities": outside_tolerance,
        "review_required": review_required,
        "review_telemetry": (
            "plasmapy_disagreement_outside_tolerance_requires_local_source_review"
            if review_required
            else "no_plasmapy_disagreement_outside_tolerance"
        ),
        "regime_gate_policy": {
            "missing_audit_blocks_engineering_run": False,
            "missing_audit_promotes_or_rejects_local_source_closure": False,
            "outside_tolerance_requires_review": True,
            "strong_coupling_is_surfaced_bounded_out_not_silent_floor": True,
        },
        "source_references": list(PLASMAPY_REGIME_GATE_SOURCE_REFS),
        "can_support_first_principles_acceptance": False,
    }


def build_physics_closure_packet(
    *,
    include_hall: bool,
    electron_energy_present: bool,
    kinetic_yield_present: bool,
    collisions_enabled: bool,
    electron_heat_flux_present: bool = False,
    electron_equilibration_audit_present: bool = False,
    ionization_charge_state_present: bool = False,
    source_backed_transport_present: bool = False,
    dimensionality: Mapping[str, Any] | None = None,
    community_formula_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return non-promoting closure status for all required physics effects."""
    effects = {
        "eos_thermodynamics": _effect(
            "blocked",
            implemented=False,
            missing=("qEOS_or_tabular_EOS", "low_density_validity", "verification_tests"),
            claim_impact="whole_shot_thermodynamics_and_pressure_authority_blocked",
        ),
        "ionization_charge_state": _effect(
            "candidate" if ionization_charge_state_present else "blocked",
            implemented=ionization_charge_state_present,
            missing=(
                "accepted_ionization_recombination_model"
                if ionization_charge_state_present
                else "ionization_recombination_model",
                "accepted_charge_state_transport"
                if ionization_charge_state_present
                else "charge_state_transport",
                "startup_link",
                "accepted_neutral_particle_source_coupling"
                if ionization_charge_state_present
                else "neutral_particle_source_coupling",
                "accepted_conductivity_eos_charge_state_feedback"
                if ionization_charge_state_present
                else "conductivity_eos_charge_state_feedback",
            ),
            claim_impact="breakdown_sheath_and_resistivity_authority_blocked",
        ),
        "single_two_temperature_energy": _effect(
            "candidate" if electron_energy_present else "blocked",
            implemented=electron_energy_present,
            missing=(
                "accepted_electron_heat_flux"
                if electron_heat_flux_present
                else "electron_heat_flux",
                "accepted_electron_ion_collisional_coupling"
                if electron_equilibration_audit_present
                else "electron_ion_collisional_coupling",
                "temperature_diagnostic_validation",
                "hall_pressure_sensitivity_uq",
            ),
            claim_impact="hall_pressure_and_yield_authority_blocked",
        ),
        "electrical_thermal_transport": _effect(
            "candidate",
            implemented=True,
            missing=(
                "accepted_transport_validity_regime"
                if source_backed_transport_present
                else "transport_validity_regime",
                "accepted_ohmic_cfl_nondominance"
                if source_backed_transport_present
                else "ohmic_cfl_nondominance",
                "accepted_thermal_conduction_closure"
                if electron_heat_flux_present
                else "thermal_conduction_closure",
                "sensitivity_uq",
            ),
            claim_impact="field_current_coupling_remains_engineering_candidate",
        ),
        "radiation_losses": _effect(
            "blocked",
            implemented=False,
            missing=("loss_model_or_bound", "opacity_or_diffusion_decision", "energy_ledger"),
            claim_impact="radiating_gas_or_high_z_claims_blocked",
        ),
        "impurity_electrode_ablation": _effect(
            "blocked",
            implemented=False,
            missing=("ablation_source_model", "impurity_transport", "electrode_material_uq"),
            claim_impact="waveform_pinch_radiation_neutron_impurity_effects_blocked",
        ),
        "hall_flr_kinetic_scope": _effect(
            "candidate" if include_hall else "blocked",
            implemented=include_hall,
            missing=(
                "electron_temperature_authority",
                "flr_validity_or_handoff",
                "kinetic_interval_review",
            ),
            claim_impact="late_pinch_and_acceleration_authority_blocked",
        ),
        "three_d_instabilities": _effect(
            "candidate",
            implemented=True,
            missing=("accepted_m_mode_evidence", "same_scope_3d_instability_packet"),
            claim_impact="kink_fragmentation_and_lifetime_authority_blocked",
        ),
        "restrike_anomalous_resistance": _effect(
            "blocked",
            implemented=False,
            missing=("restrike_model", "anomalous_resistivity_model", "post_pinch_scope"),
            claim_impact="current_dip_and_post_pinch_claims_blocked",
        ),
        "beam_target_coupling": _effect(
            "candidate" if kinetic_yield_present else "blocked",
            implemented=kinetic_yield_present,
            missing=(
                "mechanism_separation",
                "ion_distribution_transport_stopping",
                "spectrum_anisotropy_detector_response",
                "uq",
            ),
            claim_impact="total_neutron_yield_authority_blocked",
        ),
        # A8 fix: REQUIRED_EFFECTS added electron_inertia and stopping_collisions
        # (WP-N5 finding F-EI). Both are not_simulated_and_claim_blocking per the
        # static registry. They must appear in effects so that
        # closure_matrix_status_by_effect, closure_effect_status, and
        # missing_or_unaccepted_effects are symmetric with REQUIRED_EFFECTS.
        "electron_inertia": _effect(
            "blocked",
            implemented=False,
            missing=(
                "generalized_ohm_electron_inertia_closure_equation",
                "skin_depth_resolution_gate",
            ),
            claim_impact=(
                "generalized_ohm_electron_inertia_and_skin_depth_claims_blocked"
            ),
        ),
        "stopping_collisions": _effect(
            "blocked",
            implemented=False,
            missing=(
                "kr_cited_bethe_or_plasma_stopping_power_closure",
            ),
            claim_impact=(
                "fast_ion_stopping_and_beam_target_neutron_authority_blocked"
            ),
        ),
    }
    if not collisions_enabled:
        effects["electrical_thermal_transport"]["missing_channels"].append(
            "accepted_collision_parameterization"
        )
    missing_effects = [
        key
        for key, record in effects.items()
        if record["can_support_first_principles_acceptance"] is False
    ]
    active_candidate_closures = [
        key
        for key, record in effects.items()
        if record["status"] == "candidate" and record["implemented"]
    ]
    # S3.5: build the explicit per-closure registry. Runtime flags may only
    # confirm or downgrade the static audit `implemented` flag, never promote a
    # statically-blocked closure. A closure not in this map keeps its static
    # value, which keeps the registry fail-closed.
    runtime_implemented = {
        "ionization_charge_state": ionization_charge_state_present,
        "single_two_temperature_energy": electron_energy_present,
        "hall_flr_kinetic_scope": include_hall,
        "beam_target_coupling": kinetic_yield_present,
    }
    closure_registry = build_closure_registry(
        runtime_implemented=runtime_implemented,
    )
    plasmapy_regime_gate = build_plasmapy_closure_regime_gate(community_formula_audit)
    return {
        "status": "candidate_engineering_closure_packet_not_validation",
        "decision": "do_not_promote_without_complete_physics_closure_matrix",
        "required_effects": list(REQUIRED_EFFECTS),
        "required_packet_channels": list(REQUIRED_CLOSURE_PACKET_CHANNELS),
        "effects": effects,
        "closure_registry": closure_registry,
        "plasmapy_regime_gate": plasmapy_regime_gate,
        "closure_matrix_status_by_effect": {
            key: record["status"] for key, record in effects.items()
        },
        "closure_effect_status": _closure_effect_statuses(effects),
        "missing_or_unaccepted_effects": missing_effects,
        "candidate_runtime_channels": _candidate_runtime_channels(
            include_hall=include_hall,
            ionization_charge_state_present=ionization_charge_state_present,
            source_backed_transport_present=source_backed_transport_present,
            electron_energy_present=electron_energy_present,
            electron_heat_flux_present=electron_heat_flux_present,
            electron_equilibration_audit_present=electron_equilibration_audit_present,
            kinetic_yield_present=kinetic_yield_present,
            collisions_enabled=collisions_enabled,
            dimensionality=dimensionality,
            community_formula_audit=community_formula_audit,
        ),
        "active_candidate_closures": active_candidate_closures,
        "community_formula_audit": _community_formula_audit_packet(
            community_formula_audit
        ),
        "community_formula_audit_policy": {
            "optional_audit_can_support_acceptance": False,
            "local_source_truth_remains_required": True,
            "missing_or_failed_audit_blocks_engineering_run": False,
            "outside_tolerance_audit_requires_review": True,
        },
        "active_closure_policy": {
            "candidate_closures_can_run_engineering_cases": True,
            "candidate_closures_can_support_acceptance": False,
            "active_candidate_closures": active_candidate_closures,
            "required_promotion_path": (
                "each_active_or_bounded_out_effect_needs_source_equations_symbol_"
                "map_units_validity_implementation_tests_sensitivity_uq_claim_"
                "impact_and_review"
            ),
        },
        "dimensionality_acceptance_gate": _dimensionality_acceptance_gate(
            dimensionality
        ),
        "acceptance_gate": (
            "candidate_transport_ohm_electron_energy_hall_instability_and_yield_"
            "closures_cannot_support_physics_acceptance_until_every_required_"
            "effect_is_implemented_validated_or_bounded_out_with_source_equations_"
            "units_validity_tests_sensitivity_uq_claim_impact_hashes_and_review"
        ),
        "negative_test_policy": {
            "missing_effect_rejection_required": True,
            "candidate_closure_promotion_rejection_required": True,
            "hall_pressure_without_electron_temperature_rejection_required": True,
            "total_yield_without_mechanism_separation_rejection_required": True,
            "radiation_or_ablation_absent_claim_rejection_required": True,
            "anomalous_resistance_or_restrike_claim_rejection_required": True,
            "closure_sensitivity_uq_missing_rejection_required": True,
        },
        "source_references": list(CLOSURE_SOURCE_REFS),
        "dimensionality_status": None if dimensionality is None else dimensionality.get("status"),
        "source_model_limitations": (
            []
            if dimensionality is None
            else list(dimensionality.get("source_model_limitations", ()))
        ),
        "can_support_first_principles_acceptance": False,
    }


def _effect(
    status: str,
    *,
    implemented: bool,
    missing: tuple[str, ...],
    claim_impact: str,
) -> dict[str, Any]:
    missing_set = set(missing)
    return {
        "status": status,
        "implemented": implemented,
        "classification": status if status != "candidate" else "candidate_only",
        "required_packet_channels": list(REQUIRED_CLOSURE_PACKET_CHANNELS),
        "missing_channels": list(missing),
        "channel_status": _effect_channel_statuses(
            implemented=implemented,
            missing=missing_set,
        ),
        "claim_impact": claim_impact,
        "review_status": "not_reviewed_for_acceptance",
        "can_support_first_principles_acceptance": False,
    }


def _effect_channel_statuses(
    *,
    implemented: bool,
    missing: set[str],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for channel in REQUIRED_CLOSURE_PACKET_CHANNELS:
        if channel in {"effect_id", "classification"}:
            statuses[channel] = "present_non_accepting_metadata"
        elif channel == "implementation_reference" and implemented:
            statuses[channel] = "candidate_implementation_reference_not_acceptance"
        elif channel == "review_status":
            statuses[channel] = "not_reviewed_for_acceptance"
        elif channel in missing:
            statuses[channel] = "missing_or_blocked"
        else:
            statuses[channel] = "missing_or_unaccepted"
    return statuses


def _closure_effect_statuses(
    effects: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "status": str(record["status"]),
            "classification": str(record["classification"]),
            "implemented": bool(record["implemented"]),
            "claim_impact": str(record["claim_impact"]),
            "missing_channels": list(record["missing_channels"]),
            "review_status": str(record["review_status"]),
            "can_support_first_principles_acceptance": False,
        }
        for key, record in effects.items()
    }


def _dimensionality_acceptance_gate(
    dimensionality: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if dimensionality is None:
        return {
            "status": "blocked_dimensionality_packet_missing",
            "blocking_source_model_limitations": [],
        }
    limitations = list(dimensionality.get("source_model_limitations", ()))
    return {
        "status": "blocked_by_dimensionality_or_handoff_packet",
        "dimensionality_status": dimensionality.get("status"),
        "blocking_source_model_limitations": limitations,
        "can_accept_closure_without_dimensionality_acceptance": False,
    }


def _candidate_runtime_channels(
    *,
    include_hall: bool,
    ionization_charge_state_present: bool,
    source_backed_transport_present: bool,
    electron_energy_present: bool,
    electron_heat_flux_present: bool,
    electron_equilibration_audit_present: bool,
    kinetic_yield_present: bool,
    collisions_enabled: bool,
    dimensionality: Mapping[str, Any] | None,
    community_formula_audit: Mapping[str, Any] | None,
) -> list[str]:
    channels: set[str] = set()
    channels.add("candidate_electrical_transport_source_terms")
    if include_hall:
        channels.add("candidate_hall_term_enabled")
    if ionization_charge_state_present:
        channels.add("candidate_ionization_charge_state_transport")
    if source_backed_transport_present:
        channels.add("candidate_source_backed_partial_ionized_conductivity")
    if electron_energy_present:
        channels.add("candidate_electron_energy_source_terms")
    if electron_heat_flux_present:
        channels.add("candidate_braginskii_electron_heat_flux")
    if electron_equilibration_audit_present:
        channels.add("candidate_electron_ion_equilibration_audit")
    if kinetic_yield_present:
        channels.add("candidate_kinetic_yield_history")
    if collisions_enabled:
        channels.add("candidate_collision_stage_enabled")
    if _community_formula_audit_available(community_formula_audit):
        channels.add("candidate_plasmapy_community_formula_audit")
    if dimensionality is not None:
        channels.add("candidate_dimensionality_packet_linked")
        for channel in dimensionality.get("candidate_runtime_channels", ()):
            if str(channel).startswith("candidate_"):
                channels.add(f"dimensionality_{channel}")
    return sorted(channels)


def _community_formula_audit_packet(
    community_formula_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if community_formula_audit is None:
        return {
            "status": "community_formula_audit_not_requested",
            "can_support_first_principles_acceptance": False,
        }
    packet = dict(community_formula_audit)
    packet["can_support_first_principles_acceptance"] = False
    return packet


def _community_formula_audit_available(
    community_formula_audit: Mapping[str, Any] | None,
) -> bool:
    if community_formula_audit is None:
        return False
    status = str(community_formula_audit.get("status", ""))
    return status.startswith("community_formula_audit_") and "unavailable" not in status
