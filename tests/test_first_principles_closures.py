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
