"""Sprint 8 WS5 -- tests for the Braginskii Z=1 transport candidate closure.

Covers:

- the typed source packet fails closed (acceptance flags False);
- Z=1 coefficient values match the render-verified Braginskii 1965 Table 2
  (journal p.251) -- alpha_0, beta_0, gamma_0, delta_0;
- transport-output units (Ohm*m for resistivity, W/(m*K) for conductivity)
  via a dimensionally exact PlasmaPy comparison;
- the five review-required Table-2 cells are surfaced as unavailable;
- the closure_packet.py wiring keeps the closure non-accepted and keeps
  ``CLOSURE-BLK-BRAG-001`` a non-accepted blocker.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dpf.constants import e, k_B, m_d, m_e
from dpf.first_principles.closure_packet import (
    CLOSURE_BLK_BRAG_001,
    build_braginskii_z1_transport_closure,
    build_physics_closure_packet,
)
from dpf.first_principles.sprint8_braginskii_z1_transport import (
    BRAGINSKII_Z1_REVIEW_REQUIRED_CELLS_UNAVAILABLE,
    BRAGINSKII_Z1_TABLE2_COEFFICIENTS,
    BRAGINSKII_Z1_TRANSPORT_PACKET,
    braginskii_z1_electron_parallel_conductivity,
    braginskii_z1_ion_parallel_conductivity,
    braginskii_z1_parallel_resistivity,
    braginskii_z1_transport_source_packet,
)

# Representative Z=1 deuterium-plasma cross-check point (matches the
# scripts/plasmapy_braginskii_z1_crosscheck.py reference point).
_N_E = 1.0e23
_T_E_K = 1.0e6
_TAU_E = 2.7689e-10


# ---------------------------------------------------------------------------
# Fail-closed: the packet can never set acceptance.
# ---------------------------------------------------------------------------
def test_source_packet_fails_closed() -> None:
    packet = braginskii_z1_transport_source_packet()
    assert packet["accepted_runtime_claim"] is False
    assert packet["can_support_first_principles_acceptance"] is False
    assert packet["candidate_only_engineering_evidence"] is True


def test_module_level_packet_fails_closed() -> None:
    assert BRAGINSKII_Z1_TRANSPORT_PACKET["accepted_runtime_claim"] is False
    assert (
        BRAGINSKII_Z1_TRANSPORT_PACKET["can_support_first_principles_acceptance"]
        is False
    )


# ---------------------------------------------------------------------------
# Coefficient values match the render-verified Braginskii 1965 Table 2 (Z=1).
# Citation: journal p.251, column Z=1, render
# docs/extractions/braginskii_1965_render_evidence/pdf_p026_journal_p250_p251.png
# (right half).
# ---------------------------------------------------------------------------
def test_z1_table2_coefficients_match_render_verified_source() -> None:
    coeffs = BRAGINSKII_Z1_TABLE2_COEFFICIENTS
    # Spot row, Table 2 journal p.251 column Z=1.
    assert coeffs["alpha_0"] == "0.5129"
    assert coeffs["beta_0"] == "0.7110"
    assert coeffs["gamma_0"] == "3.1616"
    assert coeffs["delta_0"] == "3.7703"
    assert coeffs["delta_1"] == "14.79"


def test_resistivity_uses_table2_alpha_0_z1_cell() -> None:
    """eta_par = m_e * alpha_0 / (e^2 n_e tau_e); alpha_0 = 0.5129 (Z=1)."""
    eta = float(braginskii_z1_parallel_resistivity(_N_E, _TAU_E))
    expected = m_e * 0.5129 / (e**2 * _N_E * _TAU_E)
    assert math.isclose(eta, expected, rel_tol=1e-12)


def test_electron_conductivity_uses_table2_gamma_0_z1_cell() -> None:
    """kappa_par_e = gamma_0 n_e (k_B T_e) tau_e k_B / m_e; gamma_0 = 3.1616."""
    kappa = float(
        braginskii_z1_electron_parallel_conductivity(_N_E, _T_E_K, _TAU_E)
    )
    expected = 3.1616 * _N_E * (k_B * _T_E_K) * _TAU_E * k_B / m_e
    assert math.isclose(kappa, expected, rel_tol=1e-12)


def test_ion_conductivity_uses_eq_4_40_coefficient() -> None:
    """kappa_par_i = 3.906 n_i (k_B T_i) tau_i k_B / m_i; 3.906 from Eq. 4.40."""
    tau_i = 2.3272e-08
    kappa = float(
        braginskii_z1_ion_parallel_conductivity(_N_E, _T_E_K, tau_i, m_i=m_d)
    )
    expected = 3.906 * _N_E * (k_B * _T_E_K) * tau_i * k_B / m_d
    assert math.isclose(kappa, expected, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Units: resistivity is Ohm*m, conductivity is W/(m*K). The check is a
# dimensionally exact comparison against PlasmaPy's quantity-aware output;
# a unit error in the closure formula would surface as a large mismatch.
# ---------------------------------------------------------------------------
def test_resistivity_and_conductivity_units_against_plasmapy() -> None:
    pytest.importorskip("plasmapy")
    u = pytest.importorskip("astropy.units")
    from astropy.constants import m_p
    from plasmapy.formulary.braginskii import ClassicalTransport

    m_i = float((2.013553 * m_p).to(u.kg).value)

    ct = ClassicalTransport(
        T_e=_T_E_K * u.K,
        n_e=_N_E * u.m**-3,
        T_i=_T_E_K * u.K,
        n_i=_N_E * u.m**-3,
        ion="D+",
        Z=1,
        B=0 * u.T,
        model="Braginskii",
        field_orientation="parallel",
    )
    # resistivity -> Ohm*m
    pp_eta = ct.resistivity.to(u.ohm * u.m)
    brag_eta = float(braginskii_z1_parallel_resistivity(_N_E, _TAU_E))
    assert math.isclose(brag_eta, float(pp_eta.value), rel_tol=0.05)
    assert pp_eta.unit.is_equivalent(u.ohm * u.m)

    # electron thermal conductivity -> W/(m*K)
    pp_kappa_e = ct.electron_thermal_conductivity.to(u.W / (u.m * u.K))
    brag_kappa_e = float(
        braginskii_z1_electron_parallel_conductivity(_N_E, _T_E_K, _TAU_E)
    )
    assert math.isclose(brag_kappa_e, float(pp_kappa_e.value), rel_tol=0.05)
    assert pp_kappa_e.unit.is_equivalent(u.W / (u.m * u.K))

    # ion thermal conductivity -> W/(m*K)
    tau_i = 2.3272e-08
    pp_kappa_i = ct.ion_thermal_conductivity.to(u.W / (u.m * u.K))
    brag_kappa_i = float(
        braginskii_z1_ion_parallel_conductivity(_N_E, _T_E_K, tau_i, m_i=m_i)
    )
    assert math.isclose(brag_kappa_i, float(pp_kappa_i.value), rel_tol=0.05)
    assert pp_kappa_i.unit.is_equivalent(u.W / (u.m * u.K))


def test_packet_records_correct_output_units() -> None:
    outputs = BRAGINSKII_Z1_TRANSPORT_PACKET["closure_outputs"]
    assert outputs["parallel_electrical_resistivity"]["units"] == "Ohm*m"
    assert (
        outputs["electron_parallel_thermal_conductivity"]["units"]
        == "W/(m*K)"
    )
    assert outputs["ion_parallel_thermal_conductivity"]["units"] == "W/(m*K)"


# ---------------------------------------------------------------------------
# Resistivity is positive and scales physically (sanity, not validation).
# ---------------------------------------------------------------------------
def test_resistivity_positive_and_array_safe() -> None:
    n_e = np.array([1.0e22, 1.0e23, 1.0e24])
    tau_e = np.array([1.0e-10, 2.0e-10, 5.0e-10])
    eta = braginskii_z1_parallel_resistivity(n_e, tau_e)
    assert eta.shape == (3,)
    assert np.all(eta > 0.0)


# ---------------------------------------------------------------------------
# The five review-required Table-2 cells are surfaced as unavailable.
# ---------------------------------------------------------------------------
def test_five_review_required_cells_marked_unavailable() -> None:
    cells = BRAGINSKII_Z1_REVIEW_REQUIRED_CELLS_UNAVAILABLE
    assert len(cells) == 5
    for cell in cells:
        assert cell["status"] == "unavailable_review_required"
    # The exact five flagged cells from the Sprint 6 WS3 extraction.
    flagged = {(c["coefficient"], c["z_column"]) for c in cells}
    assert flagged == {
        ("alpha_0_prime", 3),
        ("alpha_0_prime", 4),
        ("alpha_0_prime", "inf"),
        ("alpha_0_double_prime", 3),
        ("gamma_1_prime", "inf"),
    }
    # None of the five is a Z=1 column cell, so the Z=1 candidate is safe.
    assert all(c["z_column"] != 1 for c in cells)


def test_review_required_cells_propagate_into_closure_packet() -> None:
    closure = build_braginskii_z1_transport_closure()
    cells = closure["review_required_cells_unavailable"]
    assert len(cells) == 5
    for cell in cells:
        assert cell["status"] == "unavailable_review_required"


# ---------------------------------------------------------------------------
# closure_packet.py wiring: candidate wired, acceptance stays blocked.
# ---------------------------------------------------------------------------
def test_build_braginskii_z1_transport_closure_fails_closed() -> None:
    closure = build_braginskii_z1_transport_closure()
    assert closure["accepted_runtime_claim"] is False
    assert closure["can_support_first_principles_acceptance"] is False
    assert closure["classification"] == "active_source_backed_candidate"
    # Equations render-verified -> candidate is runnable.
    assert closure["equations_4_30_to_4_45_render_verified"] is True
    assert closure["candidate_runnable"] is True


def test_closure_blk_brag_001_remains_a_non_accepted_blocker() -> None:
    """WS5 wires the candidate but does NOT close CLOSURE-BLK-BRAG-001."""
    closure = build_braginskii_z1_transport_closure()
    assert closure["blocker_id"] == CLOSURE_BLK_BRAG_001
    assert "non_accepted_blocker_remains_open" in (
        closure["closure_blk_brag_001_status"]
    )


def test_closure_packet_surfaces_braginskii_candidate_and_stays_unaccepted() -> None:
    packet = build_physics_closure_packet(
        include_hall=False,
        electron_energy_present=False,
        kinetic_yield_present=False,
        collisions_enabled=True,
    )
    # Top-level acceptance is still hard False.
    assert packet["can_support_first_principles_acceptance"] is False
    # The WS5 candidate closure is present and non-accepting.
    brag = packet["braginskii_z1_transport_candidate"]
    assert brag["can_support_first_principles_acceptance"] is False
    assert brag["accepted_runtime_claim"] is False
    # It is surfaced as a candidate runtime channel, never an accepted one.
    assert "candidate_braginskii_z1_transport" in packet["candidate_runtime_channels"]


def test_candidate_outputs_carry_resistivity_and_conductivities() -> None:
    closure = build_braginskii_z1_transport_closure()
    outputs = closure["candidate_closure_outputs"]
    assert "parallel_electrical_resistivity_ohm_m" in outputs
    assert "electron_parallel_thermal_conductivity_w_per_m_k" in outputs
    assert "ion_parallel_thermal_conductivity_w_per_m_k" in outputs
    assert (
        outputs["parallel_electrical_resistivity_ohm_m"]["coefficient_value"]
        == "0.5129"
    )


# ---------------------------------------------------------------------------
# PlasmaPy is a cross-check lane, never authority.
# ---------------------------------------------------------------------------
def test_plasmapy_is_cross_check_lane_not_authority() -> None:
    packet = braginskii_z1_transport_source_packet()
    lane = packet["cross_check_lane_not_authority"]
    assert lane["source_equivalence_granted"] is False
    assert lane["role"] == "pinned_cross_check_lane_never_source_authority"
    assert lane["script"] == "scripts/plasmapy_braginskii_z1_crosscheck.py"
