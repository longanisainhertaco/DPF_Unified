"""WP-5 / SSR-008 closure-matrix unit, negative, and blocker-hardening tests.

Tests assert fail-closed / honest behavior for the physics_closure packet:
  - 6 blocked closures remain BLOCKED with can_support_first_principles_acceptance=False
  - 9 active source-backed closures behave per their cited physics
  - Empirical modules (line_radiation, ablation, qmf_suppression, transport)
    stay OUT of the first-principles import graph (regression guard)

No closure values are asserted from training data; all expected values are
derived from the actual source code formulae or verified at runtime.
"""
from __future__ import annotations

import importlib.util
import re

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _packet(**kw):
    from dpf.first_principles.closure_packet import build_physics_closure_packet
    base = dict(
        include_hall=False,
        electron_energy_present=False,
        kinetic_yield_present=False,
        collisions_enabled=False,
    )
    base.update(kw)
    return build_physics_closure_packet(**base)


# ---------------------------------------------------------------------------
# Blocker-hardening: missing closures must stay blocked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("effect", [
    "eos_thermodynamics",
    "radiation_losses",
    "impurity_electrode_ablation",
    "restrike_anomalous_resistance",
])
def test_missing_closure_stays_blocked(effect):
    pkt = _packet()
    rec = pkt["effects"][effect]
    assert rec["status"] == "blocked"
    assert rec["implemented"] is False
    assert rec["can_support_first_principles_acceptance"] is False


def test_packet_never_supports_acceptance_all_flags_on():
    """Even with every optional channel present the packet cannot accept."""
    pkt = _packet(
        include_hall=True,
        electron_energy_present=True,
        kinetic_yield_present=True,
        collisions_enabled=True,
        electron_heat_flux_present=True,
        electron_equilibration_audit_present=True,
        ionization_charge_state_present=True,
        source_backed_transport_present=True,
    )
    assert pkt["can_support_first_principles_acceptance"] is False
    for rec in pkt["effects"].values():
        assert rec["can_support_first_principles_acceptance"] is False


def test_candidate_closure_cannot_be_promoted():
    pkt = _packet(electron_energy_present=True)
    assert pkt["effects"]["single_two_temperature_energy"]["status"] == "candidate"
    assert pkt["active_closure_policy"]["candidate_closures_can_support_acceptance"] is False


def test_beam_target_stays_blocked_without_kinetic_yield():
    rec = _packet()["effects"]["beam_target_coupling"]
    assert rec["status"] == "blocked"
    assert rec["can_support_first_principles_acceptance"] is False


def test_beam_target_candidate_still_cannot_accept_with_kinetic_yield():
    """beam_target goes candidate when kinetic_yield_present=True but
    still cannot support acceptance."""
    rec = _packet(kinetic_yield_present=True)["effects"]["beam_target_coupling"]
    assert rec["status"] == "candidate"
    assert rec["can_support_first_principles_acceptance"] is False


def test_hall_stays_blocked_without_include_hall():
    rec = _packet()["effects"]["hall_flr_kinetic_scope"]
    assert rec["status"] == "blocked"
    assert rec["can_support_first_principles_acceptance"] is False


# ---------------------------------------------------------------------------
# Empirical-module fence: regression guard against FP runtime leakage
#
# We check that no 'import' or 'from' line in the three FP runtime modules
# references the empirical module's dotted path component in a way that could
# constitute an import statement.  We use a precise regex to avoid
# false-positives from legitimately-named modules like ionization_transport
# or constrained_transport.
# ---------------------------------------------------------------------------

_EMPIRICAL_MODULES = [
    ("dpf.radiation.line_radiation", "line_radiation"),
    ("dpf.atomic.ablation",          "ablation"),
    ("dpf.radiation.qmf_suppression","qmf_suppression"),
    # 'dpf.radiation.transport' deliberately tested by full import pattern to
    # avoid matching 'ionization_transport' or 'constrained_transport'
    ("dpf.radiation.transport",      "radiation.transport"),
]

_FP_RUNTIME_MODULES = [
    "dpf.fields.hybrid_loop",
    "dpf.fields.hybrid_stepper",
    "dpf.first_principles.runner",
]

# Pattern: an import statement that contains the exact module leaf token
# (not as a sub-string of a longer identifier)
def _import_pattern(leaf: str) -> re.Pattern[str]:
    # Matches lines like: 'import <leaf>' or 'from <...leaf...> import'
    # The word boundary \\b prevents partial matches.
    return re.compile(r"(?:^|\s)(?:import|from)\s[^\n]*\b" + re.escape(leaf) + r"\b")


@pytest.mark.parametrize("mod_dotted,leaf", _EMPIRICAL_MODULES)
def test_empirical_modules_absent_from_first_principles_imports(mod_dotted, leaf):
    """Empirical/unknown-provenance modules must not be reachable from the
    first-principles runtime import graph."""
    pattern = _import_pattern(leaf)
    for fp_mod in _FP_RUNTIME_MODULES:
        spec = importlib.util.find_spec(fp_mod)
        assert spec is not None, f"FP runtime module {fp_mod} not found"
        with open(spec.origin, encoding="utf-8") as _fh:
            text = _fh.read()
        offenders = [
            line.strip()
            for line in text.splitlines()
            if pattern.search(line)
        ]
        assert not offenders, (
            f"{mod_dotted} leaked into first-principles module {fp_mod}: "
            + "; ".join(offenders)
        )


# ---------------------------------------------------------------------------
# Active-closure unit tests
# ---------------------------------------------------------------------------

def test_partial_ionized_conductivity_returns_positive_sigma():
    from dpf.fields.conductivity import partial_ionized_conductivity
    ne = np.full((2, 2, 2), 1e23)
    nn = np.full((2, 2, 2), 1e22)
    Te = np.full((2, 2, 2), 1.16e6)  # ~100 eV
    sigma, tel = partial_ionized_conductivity(
        electron_density_m3=ne,
        neutral_density_m3=nn,
        electron_temperature_K=Te,
    )
    assert np.all(sigma > 0.0), "conductivity must be positive for ne>0"
    assert tel.can_support_first_principles_acceptance is False


def test_partial_ionized_conductivity_rejects_bad_cross_section():
    from dpf.fields.conductivity import partial_ionized_conductivity
    ne = np.full((2, 2, 2), 1e23)
    nn = np.full((2, 2, 2), 1e22)
    Te = np.full((2, 2, 2), 1.16e6)
    with pytest.raises(ValueError, match="electron_neutral_cross_section_m2"):
        partial_ionized_conductivity(
            electron_density_m3=ne,
            neutral_density_m3=nn,
            electron_temperature_K=Te,
            electron_neutral_cross_section_m2=-1.0,
        )


def test_partial_ionized_conductivity_rejects_negative_density():
    from dpf.fields.conductivity import partial_ionized_conductivity
    ne = np.full((2, 2, 2), 1e23)
    nn = np.full((2, 2, 2), 1e22)
    Te = np.full((2, 2, 2), 1.16e6)
    with pytest.raises(ValueError):
        partial_ionized_conductivity(
            electron_density_m3=-ne,
            neutral_density_m3=nn,
            electron_temperature_K=Te,
        )


def test_spitzer_resistivity_order_of_magnitude():
    """NRL eq: eta ~ 5.2e-5 Z lnL Te_eV^-1.5 Ohm*m already includes alpha(Z).
    The code applies alpha(1)=0.5064 ON TOP of the classical nu_ei formula
    which gives ~2x the NRL value classically; the net result should be within
    ~30% of the NRL published number."""
    from dpf.collision.spitzer import spitzer_resistivity
    Te_eV = 100.0
    lnL = 10.0
    Te_K = np.array([Te_eV * 11604.518])
    ne = np.array([1e24])
    eta = float(spitzer_resistivity(ne, Te_K, lnL=lnL, Z=1.0)[0])
    nrl = 5.2e-5 * 1.0 * lnL * Te_eV ** -1.5   # NRL eq.(34) published value
    # The Braginskii alpha(1)=0.5064 is applied to classical nu_ei; the NRL
    # formula already encodes this correction, so eta ~ nrl (within 30%).
    assert 0.7 * nrl < eta < 1.3 * nrl, (
        f"eta={eta:.4e} not within 30% of NRL {nrl:.4e}"
    )


def test_coulomb_log_floored_at_2():
    """coulomb_log must return >= 2.0 even at extreme cold/dense conditions."""
    from dpf.collision.spitzer import coulomb_log
    ne = np.array([1e35])  # extreme density
    Te = np.array([100.0])  # very cold (K)
    lnL = float(coulomb_log(ne, Te)[0])
    assert lnL >= 2.0, f"Coulomb log floored below 2: {lnL}"


def test_ionization_three_body_rate_si_conversion():
    """NRL eq.(15): alpha_3 = 8.75e-27 cm^6/s -> 8.75e-39 m^6/s."""
    from dpf.fields.ionization_transport import nrl_three_body_recombination_rate
    Te_eV = 10.0
    rate = float(nrl_three_body_recombination_rate(np.array([Te_eV]))[0])
    # NRL CGS: 8.75e-27 Te_eV^-4.5  cm^6/s -> *1e-12 for m^6/s
    expected = 8.75e-27 * 1e-12 * Te_eV ** -4.5
    assert abs(rate - expected) / expected < 1e-9, (
        f"Three-body rate {rate:.6e} differs from NRL SI {expected:.6e}"
    )


def test_bremsstrahlung_coeff_is_si_not_cgs():
    """Guards the historical 1.69e-32 (CGS W/cm^3) vs 1.569e-40 (SI W/m^3) mix-up."""
    from dpf.radiation.bremsstrahlung import BREM_COEFF
    assert 1.0e-40 < BREM_COEFF < 2.0e-40, (
        f"BREM_COEFF={BREM_COEFF:.4e}: expected SI ~1.57e-40; CGS 1.69e-32 would fail"
    )


# ===========================================================================
# S3.5 closure registry and regime gates (WP-N5 closure-registry source audit).
#
# Contract:
#   docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md  (S3.5)
#   docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/
#     sprint_3/WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md
#
# Every active OR bounded-out closure has an explicit registry record; no
# active closure is uncategorized; candidate closures cannot support
# acceptance; every missing closure blocks its specific claim; PlasmaPy is a
# cross-check only and the strong-coupling Coulomb-log regime is a surfaced
# bounded_out_with_source gate, never a silent floor.
# ===========================================================================

# Closures the S3.5 handoff requires to carry an explicit registry record.
_S3_5_REQUIRED_CLOSURES = (
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
    "electron_inertia",
    "stopping_collisions",
)


def _registry(**kw):
    from dpf.first_principles.closure_packet import build_closure_registry
    return build_closure_registry(**kw)


# ---------------------------------------------------------------------------
# 5.0 Registry structure
# ---------------------------------------------------------------------------

def test_closure_registry_has_eight_closures():
    """Registry exposes every S3.5 closure, including the previously
    unregistered electron_inertia and stopping_collisions keys (WP-N5 F-EI)."""
    reg = _registry()
    for closure_id in _S3_5_REQUIRED_CLOSURES:
        assert closure_id in reg["closures"], f"{closure_id} missing from registry"
    # electron_inertia and stopping were the two unregistered closures.
    assert "electron_inertia" in reg["closures"]
    assert "stopping_collisions" in reg["closures"]


def test_every_registry_record_has_all_required_fields():
    """Each record carries the eleven S3.5-required fields; sub-closures too."""
    from dpf.first_principles.closure_packet import (
        REQUIRED_CLOSURE_REGISTRY_RECORD_FIELDS,
    )
    reg = _registry()
    for closure_id, record in reg["closures"].items():
        for field in REQUIRED_CLOSURE_REGISTRY_RECORD_FIELDS:
            assert field in record, f"{closure_id} missing S3.5 field {field}"
        for sub_id, sub in record.get("sub_closures", {}).items():
            for field in REQUIRED_CLOSURE_REGISTRY_RECORD_FIELDS:
                assert field in sub, f"{sub_id} missing S3.5 field {field}"


def test_no_active_closure_is_uncategorized():
    """Every record carries exactly one classification from the S3.5
    five-value vocabulary; the registry reports zero uncategorized."""
    from dpf.first_principles.closure_packet import CLOSURE_CLASSIFICATIONS
    reg = _registry()
    assert reg["uncategorized_active_closures"] == []
    assert reg["registry_policy"]["no_active_closure_uncategorized"] is True
    for closure_id, record in reg["closures"].items():
        assert record["classification"] in CLOSURE_CLASSIFICATIONS, (
            f"{closure_id} has uncategorized classification "
            f"{record['classification']!r}"
        )


def test_registry_never_supports_acceptance():
    """can_support_first_principles_acceptance is false for the registry and
    for every record; no closure may be acceptance-supporting in Sprint 3."""
    reg = _registry()
    assert reg["can_support_first_principles_acceptance"] is False
    assert reg["acceptance_supporting_closures"] == []
    for record in reg["closures"].values():
        assert record["can_support_first_principles_acceptance"] is False
        assert record["review_status"] == "not_reviewed_for_acceptance"
        for sub in record.get("sub_closures", {}).values():
            assert sub["can_support_first_principles_acceptance"] is False


def test_malformed_registry_record_fails_closed():
    """A record missing a required S3.5 field or with an unknown
    classification is a typed ValueError, never a silent pass."""
    from dpf.first_principles.closure_packet import _validate_registry_record
    with pytest.raises(ValueError, match="missing required S3.5 field"):
        _validate_registry_record("bad", {"closure_id": "bad"})
    with pytest.raises(ValueError, match="uncategorized classification"):
        _validate_registry_record(
            "bad",
            {
                "closure_id": "bad",
                "classification": "totally_made_up",
                "implemented": False,
                "source_equations_or_absence": {},
                "symbol_map": {},
                "units": "SI",
                "validity_regime": {},
                "implementation_reference": None,
                "verification_tests": [],
                "sensitivity_or_uq": "missing",
                "claim_impact": "x",
                "review_status": "not_reviewed_for_acceptance",
            },
        )


def test_runtime_flag_cannot_promote_blocked_closure():
    """Runtime confirmation may only confirm or downgrade `implemented`; it can
    never promote a statically-blocked closure (fail-closed registry)."""
    # eos_thermodynamics is statically not_simulated; a true runtime flag must
    # not flip implemented to True.
    reg = _registry(runtime_implemented={"eos_thermodynamics": True})
    assert reg["closures"]["eos_thermodynamics"]["implemented"] is False
    # electrical_thermal_transport is statically implemented; a False runtime
    # flag DOES downgrade it (operator the audit thought present but runtime
    # cannot find).
    reg2 = _registry(runtime_implemented={"electrical_thermal_transport": False})
    assert reg2["closures"]["electrical_thermal_transport"]["implemented"] is False


def test_registry_is_embedded_in_physics_closure_packet():
    """build_physics_closure_packet carries the registry and the PlasmaPy
    regime gate without breaking the legacy effects contract."""
    pkt = _packet()
    assert "closure_registry" in pkt
    assert "plasmapy_regime_gate" in pkt
    assert pkt["closure_registry"]["can_support_first_principles_acceptance"] is False
    # Legacy effects contract still present.
    assert "effects" in pkt
    assert pkt["status"] == "candidate_engineering_closure_packet_not_validation"


# ---------------------------------------------------------------------------
# 5.1 Positive / structural tests (values from cited KR equations / code)
# ---------------------------------------------------------------------------

def test_radiation_brem_matches_nrl_eq30():
    """bremsstrahlung_power reproduces NRL eq.(30) within the documented SI
    conversion. NRL eq.(30): P_Br = 1.69e-32 N_e T_e^(1/2) sum[Z^2 N(Z)]
    [W/cm^3], N_e in cm^-3, T_e in eV  -- KR L4732-4736."""
    from dpf.constants import e, k_B
    from dpf.radiation.bremsstrahlung import bremsstrahlung_power
    ne_m3 = 1.0e24
    Te_K = 100.0 * 11604.518  # 100 eV
    Z = 1.0
    gaunt = 1.2
    P_si = float(
        bremsstrahlung_power(
            np.array([ne_m3]), np.array([Te_K]), Z=Z, gaunt_factor=gaunt
        )[0]
    )
    # NRL eq.(30) CGS, evaluated independently from the cited equation.
    ne_cm3 = ne_m3 * 1.0e-6
    Te_eV = float(k_B * Te_K / e)
    P_nrl_cgs = 1.69e-32 * ne_cm3 * (Te_eV ** 0.5) * (Z * Z) * ne_cm3
    P_nrl_si = P_nrl_cgs * 1.0e6  # W/cm^3 -> W/m^3
    # The code carries an explicit Gaunt factor; NRL eq.(30) folds in g~1.2.
    assert abs(P_si - gaunt * P_nrl_si) / (gaunt * P_nrl_si) < 0.05, (
        f"brem P_si={P_si:.4e} not within 5% of NRL eq.(30) "
        f"{gaunt * P_nrl_si:.4e}"
    )


def test_collision_coulomb_log_branch_selection():
    """coulomb_log is finite, positive, and decreasing with density at fixed
    temperature -- consistent with NRL eq. lambda_ei branches KR L3045-3059
    (lambda = ln(r_max/r_min), r_max ~ Debye ~ n^(-1/2))."""
    from dpf.collision.spitzer import coulomb_log
    Te = np.array([1.0e6, 1.0e6])  # ~86 eV, weakly coupled
    ne_low = np.array([1.0e22, 1.0e22])
    ne_high = np.array([1.0e26, 1.0e26])
    ln_low = float(coulomb_log(ne_low, Te)[0])
    ln_high = float(coulomb_log(ne_high, Te)[0])
    assert np.isfinite(ln_low) and ln_low > 0.0
    assert np.isfinite(ln_high) and ln_high > 0.0
    # Higher density -> smaller Debye length -> smaller ln Lambda.
    assert ln_high < ln_low, (
        f"ln Lambda must decrease with density: {ln_high} !< {ln_low}"
    )


def test_beam_target_bosch_hale_in_valid_range():
    """dd_cross_section returns 0 outside the Bosch-Hale 0.5-5000 keV validity
    range and a finite positive value inside (KR bosch-hale-1992)."""
    from dpf.diagnostics.beam_target import dd_cross_section
    assert dd_cross_section(0.1) == 0.0      # below 0.5 keV
    assert dd_cross_section(1.0e5) == 0.0    # above 5000 keV
    sigma = dd_cross_section(100.0)          # inside range
    assert np.isfinite(sigma) and sigma > 0.0


def test_eos_ideal_gas_dimensional_consistency():
    """IdealEOS pressures and energies are dimensionally consistent and
    positive for a physical state (in-code form, eos.py:32-67)."""
    from dpf.fluid.eos import IdealEOS
    eos = IdealEOS(gamma=5.0 / 3.0, ion_mass=3.344e-27, Z=1.0)
    rho = np.array([1.0e-3])
    Ti = np.array([1.0e6])
    Te = np.array([1.0e6])
    p_i = eos.ion_pressure(rho, Ti)
    p_e = eos.electron_pressure(rho, Te)
    e_i = eos.ion_energy(rho, Ti)
    c_s = eos.sound_speed(rho, Ti, Te)
    assert float(p_i[0]) > 0.0 and float(p_e[0]) > 0.0
    assert float(e_i[0]) > 0.0
    assert float(c_s[0]) > 0.0


# ---------------------------------------------------------------------------
# 5.2 Fail-closed negative controls
# ---------------------------------------------------------------------------

def test_eos_blocked_without_tabular_packet():
    """eos_thermodynamics stays blocked / non-accepting with no tabular/QEOS
    packet: no KR-cited tabular EOS closure exists (WP-N5 1.1)."""
    rec = _registry()["closures"]["eos_thermodynamics"]
    assert rec["classification"] == "not_simulated_and_claim_blocking"
    assert rec["implemented"] is False
    assert rec["can_support_first_principles_acceptance"] is False
    absence = rec["source_equations_or_absence"]["blocking_absence"]
    assert absence["local_source_present"] is False
    assert "qEOS_or_tabular_EOS" in rec["missing_parameters"]


def test_radiation_blocked_without_opacity_decision():
    """radiation_losses stays blocked when the opacity/diffusion decision is
    missing, even though the bremsstrahlung volumetric term is NRL-grounded
    (WP-N5 1.2)."""
    rec = _registry()["closures"]["radiation_losses"]
    assert rec["classification"] == "active_blocked"
    assert rec["can_support_first_principles_acceptance"] is False
    assert "opacity_or_diffusion_decision" in rec["missing_parameters"]
    assert "rosseland_kramers_opacity_closure" in rec["missing_parameters"]


def test_ablation_blocked_without_efficiency_source():
    """impurity_electrode_ablation stays blocked while the ablation efficiency
    has no KR source (ablation.py self-declares the source missing)."""
    rec = _registry()["closures"]["impurity_electrode_ablation"]
    assert rec["classification"] == "active_blocked"
    assert rec["can_support_first_principles_acceptance"] is False
    absence = rec["source_equations_or_absence"]["blocking_absence"]
    assert absence["local_source_present"] is False
    assert any(
        "ablation_efficiency" in item for item in rec["missing_parameters"]
    )


def test_anomalous_and_restrike_claim_rejected():
    """Anomalous-resistance and restrike are individually classified and both
    block the current-dip / post-pinch claim. restrike is not simulated;
    anomalous resistance is implemented-but-blocked (WP-N5 1.4, 1.5, 6.1)."""
    rec = _registry()["closures"]["restrike_anomalous_resistance"]
    subs = rec["sub_closures"]
    assert set(subs) == {"anomalous_resistance", "restrike"}
    assert subs["restrike"]["classification"] == "not_simulated_and_claim_blocking"
    assert subs["restrike"]["implemented"] is False
    assert subs["anomalous_resistance"]["classification"] == "active_blocked"
    # Both block their claim and neither can support acceptance.
    for sub in subs.values():
        assert sub["can_support_first_principles_acceptance"] is False
    # The shared effect also surfaces in the packet's negative-test policy.
    pkt = _packet()
    assert pkt["negative_test_policy"][
        "anomalous_resistance_or_restrike_claim_rejection_required"
    ] is True


def test_electron_inertia_registered_and_blocked():
    """The previously unregistered electron_inertia closure (WP-N5 F-EI) now
    exists, is classified, is not simulated, and blocks its claim with an
    explicit source absence."""
    reg = _registry()
    assert "electron_inertia" in reg["closures"]
    rec = reg["closures"]["electron_inertia"]
    assert rec["classification"] == "not_simulated_and_claim_blocking"
    assert rec["implemented"] is False
    assert rec["can_support_first_principles_acceptance"] is False
    absence = rec["source_equations_or_absence"]["blocking_absence"]
    assert absence["local_source_present"] is False
    assert "electron_inertia" in reg["blocked_claims"]


def test_stopping_blocked_blocks_beam_target():
    """stopping_collisions is blocked (no KR-cited stopping-power closure) and
    its absence blocks beam-target neutron authority. beam_target_coupling
    stays an external-candidate comparator and cannot support acceptance
    (WP-N5 1.7b, 1.8)."""
    reg = _registry()
    stopping = reg["closures"]["stopping_collisions"]
    assert stopping["classification"] == "not_simulated_and_claim_blocking"
    assert stopping["implemented"] is False
    assert "stopping_collisions" in reg["blocked_claims"]
    beam = reg["closures"]["beam_target_coupling"]
    assert beam["classification"] == "external_candidate_not_authority"
    assert beam["can_support_first_principles_acceptance"] is False
    # Even with kinetic yield present at runtime, beam-target cannot promote:
    # stopping and mechanism separation remain absent.
    reg_kin = _registry(runtime_implemented={"beam_target_coupling": True})
    assert reg_kin["closures"]["beam_target_coupling"]["implemented"] is False
    assert (
        "ion_distribution_transport_stopping"
        in reg_kin["closures"]["beam_target_coupling"]["missing_parameters"]
    )


def test_closure_value_substituted_from_residual_rejected():
    """No closure may be back-derived from an energy residual. Every blocked /
    not-simulated closure declares an explicit source absence rather than a
    residual-substituted value (mirrors the Auluck no-closure rule)."""
    reg = _registry()
    for closure_id, record in reg["closures"].items():
        if record["classification"] in (
            "active_blocked",
            "not_simulated_and_claim_blocking",
        ):
            source = record["source_equations_or_absence"]
            absences = [
                v
                for v in source.values()
                if isinstance(v, dict) and v.get("local_source_present") is False
            ]
            assert absences, (
                f"{closure_id} is blocked but declares no explicit source "
                f"absence -- a closure may not be residual-substituted"
            )


def test_closure_sensitivity_uq_missing_rejected():
    """A closure without a sensitivity/UQ packet cannot be promoted: every
    record reports sensitivity_or_uq == 'missing' and cannot accept."""
    reg = _registry()
    for closure_id, record in reg["closures"].items():
        assert record["sensitivity_or_uq"] == "missing", (
            f"{closure_id} claims a sensitivity/UQ packet that does not exist"
        )
        assert record["can_support_first_principles_acceptance"] is False
    pkt = _packet()
    assert pkt["negative_test_policy"][
        "closure_sensitivity_uq_missing_rejection_required"
    ] is True


# ---------------------------------------------------------------------------
# 5.3 PlasmaPy strong-coupling regime gate (negative control)
#
# PlasmaPy is a cross-check ONLY. A missing PlasmaPy audit cannot promote or
# reject a local-source closure. A disagreement outside tolerance sets
# review-required telemetry. The strong-coupling Coulomb-log regime is a
# surfaced bounded_out_with_source gate citing the NRL formulary validity
# edge, never a silent floor.
# ---------------------------------------------------------------------------

def test_plasmapy_missing_audit_does_not_promote_or_reject():
    """A missing PlasmaPy audit can neither promote nor reject a local-source
    closure, and does not block an engineering run (S3.5 PlasmaPy rule)."""
    from dpf.first_principles.closure_packet import (
        build_plasmapy_closure_regime_gate,
    )
    gate = build_plasmapy_closure_regime_gate(None)
    assert gate["audit_present"] is False
    assert gate["plasmapy_can_promote_closure"] is False
    assert gate["plasmapy_can_reject_closure"] is False
    assert gate["missing_audit_promotes_or_rejects_closure"] is False
    assert (
        gate["regime_gate_policy"]["missing_audit_blocks_engineering_run"]
        is False
    )
    assert gate["can_support_first_principles_acceptance"] is False


def test_plasmapy_coupling_regime_gate():
    """The PlasmaPy strong-coupling regime gate.

    In a weakly coupled state (ln Lambda comfortably > 2, no CouplingWarning)
    the PlasmaPy cross-check runs as telemetry only and the bounded-out gate
    does NOT trigger. In a strongly coupled DPF pinch-core state (CouplingWarning
    raised or ln Lambda <= 2) the classical Spitzer/collision closure is OUT of
    its validity range: the gate marks it bounded_out_with_source and records
    the NRL lambda >> 1 validity citation. The CouplingWarning is never
    swallowed silently.
    """
    from dpf.first_principles.closure_packet import (
        build_plasmapy_closure_regime_gate,
    )
    from dpf.first_principles.plasmapy_audit import (
        build_plasmapy_formulary_audit_packet,
    )

    # --- Weak coupling: hot, moderate density. ln Lambda >> 2. ---
    weak_audit = build_plasmapy_formulary_audit_packet(
        {"electron_density_m3": 1.0e22, "electron_temperature_K": 1.0e6}
    )
    weak_sc = weak_audit["strong_coupling_regime"]
    assert weak_sc["coupling_warning_raised"] is False
    assert weak_sc["coulomb_log_value"] > 2.0
    assert weak_sc["strong_coupling_out_of_validity"] is False
    assert weak_sc["classification"] == (
        "weak_coupling_within_classical_transport_validity"
    )
    weak_gate = build_plasmapy_closure_regime_gate(weak_audit)
    assert weak_gate["strong_coupling_bounded_out_gate"]["triggered"] is False

    # --- Strong coupling: very dense, cold DPF pinch core. ---
    strong_audit = build_plasmapy_formulary_audit_packet(
        {"electron_density_m3": 1.0e27, "electron_temperature_K": 1.16e4}
    )
    strong_sc = strong_audit["strong_coupling_regime"]
    # The CouplingWarning must NOT be swallowed silently.
    assert strong_sc["warning_swallowed_silently"] is False
    assert strong_sc["strong_coupling_out_of_validity"] is True
    assert strong_sc["classification"] == "bounded_out_with_source"
    # Strong coupling is surfaced, never a silent floor.
    assert strong_sc["is_silent_floor"] is False

    strong_gate = build_plasmapy_closure_regime_gate(strong_audit)
    bounded_out = strong_gate["strong_coupling_bounded_out_gate"]
    assert bounded_out["triggered"] is True
    assert bounded_out["classification"] == "bounded_out_with_source"
    assert bounded_out["closure_id"] == "electrical_thermal_transport"
    assert bounded_out["silent_floor_forbidden"] is True
    # The gate must cite the NRL classical-transport validity edge.
    cited_lines = {
        ref["lines"] for ref in bounded_out["source_equations_or_bound"]
    }
    assert "3036-3038" in cited_lines      # "fails when lambda ~ 1"
    assert "3379-3383" in cited_lines      # "lambda >> 1" criterion
    for ref in bounded_out["source_equations_or_bound"]:
        assert ref["path"].endswith("2019nrlplasma-formulary-037290d4.md")
    # Acceptance still cannot be supported.
    assert strong_gate["can_support_first_principles_acceptance"] is False


def test_plasmapy_outside_tolerance_sets_review_required():
    """A PlasmaPy disagreement outside tolerance sets review-required
    telemetry but does NOT block an engineering run (S3.5 PlasmaPy rule)."""
    from dpf.first_principles.closure_packet import (
        build_plasmapy_closure_regime_gate,
    )
    # Synthetic audit packet with one quantity outside tolerance.
    audit = {
        "status": "community_formula_audit_partial_not_authority",
        "quantities": {
            "coulomb_log": {
                "status": (
                    "community_formula_cross_check_outside_tolerance_not_"
                    "authority"
                ),
                "local_value": 10.0,
                "plasmapy_value": 4.0,
            },
        },
        "strong_coupling_regime": {
            "coupling_warning_raised": False,
            "strong_coupling_out_of_validity": False,
        },
    }
    gate = build_plasmapy_closure_regime_gate(audit)
    assert gate["review_required"] is True
    assert "coulomb_log" in gate["disagreement_outside_tolerance_quantities"]
    assert gate["regime_gate_policy"]["outside_tolerance_requires_review"] is True
    assert (
        gate["regime_gate_policy"]["missing_audit_blocks_engineering_run"]
        is False
    )
    # A disagreement cannot promote or reject the local-source closure.
    assert gate["plasmapy_can_promote_closure"] is False
    assert gate["plasmapy_can_reject_closure"] is False
