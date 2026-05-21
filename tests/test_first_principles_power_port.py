"""WP-N1B Auluck eq. (6) six-term power-port ledger and negative tests.

Authority: docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/
sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md (verified eq. (1)-(14),
transcribed from the primary PDF) and the WP-N1B power-port acceptance proposal.

Auluck eq. (6): I(t)V(t) = I + II + III + IV + V + VI, six independent terms.
Terms I/III are Omega volume integrals; II/IV/V/VI are Sigma_p moving-boundary
surface integrals. The ledger has NO closure-by-construction residual.

These tests prove fail-closed behaviour: a missing Sigma_p face set blocks the
moving-boundary terms; a term that can only be closure-derived is rejected, not
emitted as independent; a missing eq. (1) sign-convention record fails the
ledger closed. Tests are pure functions over constructed inputs; no run
interdependency. Acceptance stays blocked regardless of residual magnitude.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any

import numpy as np
import pytest

# S3.3 structures imported by full dotted path (the package __init__ files are
# not edited under the Sprint 3 file-scope rule).
from dpf.fields.hybrid_simulator import _circuit_udpf_for_step
from dpf.fields.source_geometry import SigmaPSurfacePacket as _SigmaPSurfacePacket
from dpf.first_principles.power_port import (
    _SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED,
    _WP_N1B_LEDGER_KEYS,
    ACCEPTANCE_BLOCKING_CHANNELS,
    POWER_PORT_SOURCE_REFS,
    REQUIRED_POWER_PORT_CHANNELS,
    _sigma_p_packet_from_ledger,
    build_engineering_power_port_packet,
    build_wp_n1_auluck_power_port_ledger,
    build_wp_n1_negative_test_policy,
)

# The six Auluck eq. (6) term keys (term labels I-VI are Auluck's own).
_SIGMA_P_TERMS = (
    "term_ii_motional_magnetic_sigma_p_J",
    "term_iv_motional_electric_sigma_p_J",
    "term_v_resistive_sigma_p_J",
    "term_vi_anomalous_poloidal_sigma_p_J",
)
_STORED_TERMS = (
    "term_i_stored_magnetic_energy_rate_J",
    "term_iii_stored_electric_energy_rate_J",
)


def _omega_domain(
    *,
    interface_disjoint: bool = True,
    interface_non_empty: bool = True,
    exhaustive: bool = True,
    disjoint: bool = True,
) -> dict[str, Any]:
    """Return a synthetic Auluck Omega partition packet."""
    return {
        "partition_constraints": {
            "mutually_disjoint": disjoint,
            "exhaustive": exhaustive,
            "terminal_source_interface_non_empty": interface_non_empty,
            "terminal_source_interface_disjoint_from_omega": interface_disjoint,
        },
        "omega_volume_cells": {
            "mask_sha256": "omega_hash",
            "cell_count": 40,
            "bounds": {"non_empty": True},
            "source_refs": ["KR: auluck:203-257"],
        },
        "terminal_source_interface_faces": {
            "mask_sha256": "interface_hash",
            "cell_count": 25,
            "bounds": {"non_empty": True},
            "source_refs": ["KR: auluck:203-257"],
        },
        "wall_material_faces": {
            "mask_sha256": "wall_hash",
            "cell_count": 40,
            "bounds": {"non_empty": True},
            "source_refs": ["KR: auluck:203-257"],
        },
        "open_pml_faces": {
            "mask_sha256": "pml_hash",
            "cell_count": 20,
            "bounds": {"non_empty": True},
            "source_refs": ["KR: auluck:203-257"],
        },
        "source_interface_z_index": 0,
        "geometry_review_status": "geometry_candidate_not_reviewed",
    }


def _ledger(
    *,
    terminal: float = 100.0,
    record_sign_convention: bool = True,
    first_step_fallback: bool = False,
    first_step_udpf_source: str = "candidate_lagged_volume_j_dot_e",
    omega: dict[str, Any] | None = None,
    stored_magnetic_delta_J: float | None = None,
    stored_electric_delta_J: float | None = None,
) -> dict[str, Any]:
    """Return a synthetic simulator-emitted power_port_ledger.

    By default the runtime emits the terminal I*V cumulative work and the
    Omega partition, but NOT a Sigma_p moving-boundary face set and NOT v/eta
    on Sigma_p -- so terms II/IV/V/VI always fail closed here.

    The magnetic/electric stored-energy split (Sprint 2.3 step 1) is included
    only when ``stored_magnetic_delta_J`` / ``stored_electric_delta_J`` are
    passed. With split telemetry absent, terms I/III fail closed; with it
    present, terms I/III are computed independently.
    """
    ledger: dict[str, Any] = {
        "cumulative_terminal_port_work_J": terminal,
        "first_step_fallback": first_step_fallback,
        "first_step_udpf_source": first_step_udpf_source,
        "steps_accumulated": 2,
        "snapshot_provenance": {
            "time_centering": "candidate_step_consistent_not_accepted",
        },
        "domain_partition": omega if omega is not None else _omega_domain(),
    }
    if record_sign_convention:
        ledger["sign_convention"] = (
            "wp_n1_packet_section_3_1_into_omega_positive"
        )
    if stored_magnetic_delta_J is not None:
        ledger["stored_magnetic_energy_delta_J"] = stored_magnetic_delta_J
    if stored_electric_delta_J is not None:
        ledger["stored_electric_energy_delta_J"] = stored_electric_delta_J
    return ledger


# --- baseline: six-term structure, every term fails closed ----------------


def test_wp_n1b_ledger_is_six_term_auluck_eq6_decomposition() -> None:
    """The ledger exposes Auluck eq. (6)'s six named terms I-VI."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    assert wp["auluck_eq6_decomposition"] is True
    terms = wp["energy_ledger_terms_J"]
    assert set(terms.keys()) == set(_WP_N1B_LEDGER_KEYS)
    assert len(_WP_N1B_LEDGER_KEYS) == 6
    # There is no "electrode_interface_work" term -- Auluck's balance has none.
    assert "electrode_interface_work_J" not in terms
    assert "electrode_interface_work_J" not in wp


def test_wp_n1b_all_six_terms_fail_closed_when_no_split_no_sigma_p() -> None:
    """With neither the magnetic/electric stored-energy split nor a Sigma_p
    face set, every eq. (6) term fails closed, value None."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    terms = wp["energy_ledger_terms_J"]
    for key in _WP_N1B_LEDGER_KEYS:
        assert terms[key] is None, key
        assert wp["energy_ledger_term_status"][key] == "blocked"
    # No term was computed independently; none is in independent_terms.
    assert wp["independent_terms"] == []
    assert wp["all_six_terms_computed_independently"] is False


def test_wp_n1b_residual_is_genuine_diagnostic_not_closure() -> None:
    """The residual is a genuine I*V - sum diagnostic, never closure-derived."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    # The ledger explicitly disclaims closure-by-construction.
    assert wp["residual_is_closure_by_construction"] is False
    assert wp["residual_definition"] == (
        "I*V - (term_i + term_ii + term_iii + term_iv + term_v + term_vi)"
    )
    # With terms failing closed, the residual is None -- it is NOT papered
    # over with a derived value (no closure substitution).
    assert wp["residual_J"] is None
    assert wp["residual_fraction"] is None
    assert wp["residual_is_genuine_diagnostic"] is False
    assert wp["ledger_blocked"] is True


def test_wp_n1b_acceptance_stays_false_with_no_path_to_true() -> None:
    """Acceptance is hardcoded False at every emitted level."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    assert wp["can_support_power_port_acceptance"] is False
    assert wp["can_support_first_principles_acceptance"] is False
    assert wp["accepted_residual_tolerance"] == "not_attached"
    assert wp["scientific_status"] == "engineering_candidate_not_validation"
    # Absent-ledger branch is also non-accepting.
    blocked = build_wp_n1_auluck_power_port_ledger(None)
    assert blocked["can_support_power_port_acceptance"] is False
    assert blocked["can_support_first_principles_acceptance"] is False


def test_wp_n1b_records_auluck_eq1_sign_convention_with_leading_minus() -> None:
    """The eq. (1) sign convention -- with its load-bearing leading minus --
    is recorded verbatim in the emitted ledger."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    # Verified extract eq. (1): V_12 = -(1/I) integral_Omega d3r (J.E).
    assert wp["auluck_eq1_sign_convention"] == (
        "V_12 = -(1/I) integral_Omega d3r (J.E)"
    )
    assert wp["auluck_eq1_sign_convention"].count("-") >= 1
    assert wp["auluck_eq1_leading_minus_is_load_bearing"] is True
    assert wp["auluck_eq1_sign_convention_recorded"] is True
    sign = wp["sign_convention"]
    assert sign["auluck_eq1_relation"] == (
        "V_12 = -(1/I) integral_Omega d3r (J.E)"
    )
    assert sign["auluck_eq1_consistency_check"] == (
        "I*V_12 = -integral_Omega (J.E)"
    )


def test_wp_n1b_omega_domain_emits_four_disjoint_exhaustive_labels() -> None:
    """The four-label Omega partition is disjoint, exhaustive, interface
    excluded -- this is the volume domain for terms I/III."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    od = wp["auluck_omega_domain"]
    assert set(od["labels"].keys()) == {
        "omega_volume_cells",
        "terminal_source_interface_faces",
        "wall_material_faces",
        "open_pml_faces",
    }
    for label in od["labels"].values():
        assert label["mask_sha256"] is not None
        assert label["cell_count"] is not None
    assert od["partition_mutually_disjoint"] is True
    assert od["partition_exhaustive"] is True
    assert od["terminal_source_interface_disjoint_from_omega"] is True
    assert od["partition_valid"] is True


# --- per-term tests: each of I-VI fails closed for the documented reason --


def test_wp_n1b_terms_i_iii_fail_closed_when_split_telemetry_absent() -> None:
    """Terms I and III need the magnetic / electric stored-energy split.
    When the ledger does NOT carry the split, terms I/III fail closed. An older
    combined-only stored-EM delta must NOT be substituted for either term."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    packets = wp["energy_ledger_term_packets"]
    for key in _STORED_TERMS:
        packet = packets[key]
        assert packet["value_J"] is None, key
        assert packet["status"] == "blocked", key
        assert packet["computed_independently"] is False, key
        assert packet["blocker"] == (
            "stored_em_energy_magnetic_electric_split_not_exposed_by_runtime"
        ), key
        assert "eq. (6)" in packet["source_ref"]


# --- F1 / Sprint 2.3 step 1: terms I and III computed independently -------


def test_wp_n1b_term_i_computed_independently_from_split_telemetry() -> None:
    """F1: when the runtime emits the magnetic stored-energy delta over
    Omega, term I = d/dt integral_Omega (1/2 mu0^-1 B^2) is computed
    INDEPENDENTLY -- its value is that delta, not a closure-derived figure."""
    wp = build_wp_n1_auluck_power_port_ledger(
        _ledger(stored_magnetic_delta_J=12.5)
    )
    term_i = wp["energy_ledger_term_packets"][
        "term_i_stored_magnetic_energy_rate_J"
    ]
    assert term_i["value_J"] == 12.5
    assert term_i["status"] == "computed_independently"
    assert term_i["computed_independently"] is True
    assert term_i["blocker"] is None
    assert term_i["derived_by_closure"] is False
    # The value is sourced from the magnetic-only stored-energy telemetry.
    assert term_i["runtime_telemetry_key"] == "stored_magnetic_energy_delta_J"
    assert term_i["integrand"] == "d/dt integral_Omega d3r (1/2 mu0^-1 B^2)"
    assert wp["energy_ledger_terms_J"][
        "term_i_stored_magnetic_energy_rate_J"
    ] == 12.5
    assert (
        "term_i_stored_magnetic_energy_rate_J" in wp["independent_terms"]
    )
    assert "term_i_stored_magnetic_energy_rate_J" not in wp["blocked_terms"]


def test_wp_n1b_term_iii_computed_independently_from_split_telemetry() -> None:
    """F1: when the runtime emits the electric stored-energy delta over
    Omega, term III = d/dt integral_Omega (1/2 eps0 E^2) is computed
    INDEPENDENTLY from that delta."""
    wp = build_wp_n1_auluck_power_port_ledger(
        _ledger(stored_electric_delta_J=-3.0)
    )
    term_iii = wp["energy_ledger_term_packets"][
        "term_iii_stored_electric_energy_rate_J"
    ]
    assert term_iii["value_J"] == -3.0
    assert term_iii["status"] == "computed_independently"
    assert term_iii["computed_independently"] is True
    assert term_iii["blocker"] is None
    assert term_iii["derived_by_closure"] is False
    assert term_iii["runtime_telemetry_key"] == (
        "stored_electric_energy_delta_J"
    )
    assert term_iii["integrand"] == "d/dt integral_Omega d3r (1/2 eps0 E^2)"
    assert wp["energy_ledger_terms_J"][
        "term_iii_stored_electric_energy_rate_J"
    ] == -3.0
    assert (
        "term_iii_stored_electric_energy_rate_J" in wp["independent_terms"]
    )


def test_wp_n1b_terms_i_iii_independent_terms_ii_iv_v_vi_still_blocked(
) -> None:
    """F1: with the split present, terms I and III are independent while
    terms II/IV/V/VI stay fail-closed on Sigma_p geometry."""
    wp = build_wp_n1_auluck_power_port_ledger(
        _ledger(stored_magnetic_delta_J=8.0, stored_electric_delta_J=1.0)
    )
    assert sorted(wp["independent_terms"]) == [
        "term_i_stored_magnetic_energy_rate_J",
        "term_iii_stored_electric_energy_rate_J",
    ]
    # The four Sigma_p terms remain blocked.
    assert set(wp["blocked_terms"].keys()) == set(_SIGMA_P_TERMS)
    for key in _SIGMA_P_TERMS:
        packet = wp["energy_ledger_term_packets"][key]
        assert packet["status"] == "blocked", key
        assert packet["blocker"] == (
            "sigma_p_face_set_not_available_requires_wp_n3_reviewed_geometry"
        ), key


def test_wp_n1b_split_present_residual_stays_none_not_all_six() -> None:
    """F1: terms I/III computed does NOT unblock the residual -- it stays
    None because II/IV/V/VI are not independent. No residual is computed
    from a partial six-term set; acceptance stays False."""
    wp = build_wp_n1_auluck_power_port_ledger(
        _ledger(stored_magnetic_delta_J=8.0, stored_electric_delta_J=1.0)
    )
    assert wp["all_six_terms_computed_independently"] is False
    # Residual is NOT computed: not all six terms are independent.
    assert wp["residual_J"] is None
    assert wp["residual_fraction"] is None
    assert wp["residual_is_genuine_diagnostic"] is False
    assert wp["residual_is_closure_by_construction"] is False
    assert wp["ledger_blocked"] is True
    assert wp["ledger_blocker"] == (
        "auluck_eq6_terms_fail_closed_pending_wp_n3_geometry"
    )
    # Acceptance stays False with no path to True.
    assert wp["can_support_power_port_acceptance"] is False
    assert wp["can_support_first_principles_acceptance"] is False


def test_wp_n1b_term_i_blocks_when_only_electric_split_present() -> None:
    """F1: the split is per-term. If only the electric delta is emitted,
    term III is independent but term I still fails closed -- no term is
    faked from the other."""
    wp = build_wp_n1_auluck_power_port_ledger(
        _ledger(stored_electric_delta_J=2.0)
    )
    term_i = wp["energy_ledger_term_packets"][
        "term_i_stored_magnetic_energy_rate_J"
    ]
    term_iii = wp["energy_ledger_term_packets"][
        "term_iii_stored_electric_energy_rate_J"
    ]
    assert term_i["status"] == "blocked"
    assert term_i["value_J"] is None
    assert term_i["blocker"] == (
        "stored_em_energy_magnetic_electric_split_not_exposed_by_runtime"
    )
    assert term_iii["status"] == "computed_independently"
    assert term_iii["value_J"] == 2.0


def test_wp_n1b_runtime_emits_magnetic_electric_stored_energy_split() -> None:
    """F1 runtime: the simulator's finalized power_port_ledger carries the
    magnetic-only and electric-only stored-energy split over Omega, so
    power_port.py can compute terms I and III independently."""
    from dpf.fields.hybrid_simulator import (
        _finalize_power_port_ledger,
        _new_power_port_ledger_accumulator,
    )
    from dpf.fields.hybrid_stepper import omega_stored_em_energy_split_J

    accumulator = _new_power_port_ledger_accumulator()
    # Two steps: magnetic 2 -> 6, electric 1 -> 1.5 over Omega.
    accumulator["stored_em_energy_initial_J"] = 3.0
    accumulator["stored_em_energy_final_J"] = 7.5
    accumulator["stored_magnetic_energy_initial_J"] = 2.0
    accumulator["stored_magnetic_energy_final_J"] = 6.0
    accumulator["stored_electric_energy_initial_J"] = 1.0
    accumulator["stored_electric_energy_final_J"] = 1.5
    accumulator["steps_accumulated"] = 2
    ledger = _finalize_power_port_ledger(
        accumulator=accumulator,
        n_steps_completed=2,
        apply_circuit_boundary=True,
    )
    assert ledger is not None
    assert ledger["stored_magnetic_energy_delta_J"] == 4.0
    assert ledger["stored_electric_energy_delta_J"] == 0.5
    # The split sums to the combined delta -- consistency, not closure.
    assert (
        ledger["stored_magnetic_energy_delta_J"]
        + ledger["stored_electric_energy_delta_J"]
        == ledger["stored_em_energy_delta_J"]
    )
    # The split function keeps magnetic and electric separable.
    import numpy as np

    grid_shape = (2, 2, 2)
    omega = np.ones(grid_shape, dtype=bool)
    b_field = np.zeros((*grid_shape, 3))
    b_field[..., 2] = 1.0
    e_field = np.zeros((*grid_shape, 3))
    split = omega_stored_em_energy_split_J(
        electric_field_V_m=e_field,
        magnetic_field_T=b_field,
        omega_volume_cells=omega,
        cell_volume_m3=1.0,
    )
    # Pure magnetic field => zero electric stored energy, positive magnetic.
    assert split["electric_J"] == 0.0
    assert split["magnetic_J"] > 0.0


def test_wp_n1b_terms_ii_iv_v_vi_fail_closed_no_sigma_p_face_set() -> None:
    """Terms II/IV/V/VI are Sigma_p moving-boundary integrals; with no
    reviewed Sigma_p face set (and no v/eta on faces) each fails closed with
    the explicit Sigma_p blocker key."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    packets = wp["energy_ledger_term_packets"]
    for key in _SIGMA_P_TERMS:
        packet = packets[key]
        assert packet["value_J"] is None, key
        assert packet["status"] == "blocked", key
        assert packet["computed_independently"] is False, key
        assert packet["blocker"] == (
            "sigma_p_face_set_not_available_requires_wp_n3_reviewed_geometry"
        ), key
        # The blocked term names a verified Auluck source.
        assert "eq. (6)" in packet["source_ref"]


def test_wp_n1b_blocked_terms_map_lists_all_six_with_blockers() -> None:
    """The emitted blocked_terms map enumerates every blocked term and its
    blocker -- the fail-closed state is explicit, not silent."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    blocked = wp["blocked_terms"]
    assert set(blocked.keys()) == set(_WP_N1B_LEDGER_KEYS)
    for key in _SIGMA_P_TERMS:
        assert "sigma_p" in blocked[key]
    for key in _STORED_TERMS:
        assert "split" in blocked[key]


# --- N1 sign reversal -----------------------------------------------------


def test_negative_n1_sign_reversal_keeps_ledger_non_accepting() -> None:
    """N1: flipping the terminal I*V work sign does not unblock the ledger;
    the sign convention is a declared, load-bearing field."""
    baseline = build_wp_n1_auluck_power_port_ledger(_ledger(terminal=100.0))
    reversed_wp = build_wp_n1_auluck_power_port_ledger(_ledger(terminal=-100.0))
    # The terminal I*V work is the eq. (6) LHS; its sign flips with the input.
    assert baseline["iv_work_J"] == 100.0
    assert reversed_wp["iv_work_J"] == -100.0
    # The sign convention with the load-bearing leading minus is recorded in
    # both -- a reversed convention cannot silently pass review.
    for wp in (baseline, reversed_wp):
        assert wp["auluck_eq1_leading_minus_is_load_bearing"] is True
        assert wp["sign_convention"]["auluck_eq1_relation"] == (
            "V_12 = -(1/I) integral_Omega d3r (J.E)"
        )
    # Neither can be accepted; the residual stays None (terms fail closed).
    assert reversed_wp["can_support_power_port_acceptance"] is False
    assert reversed_wp["residual_J"] is None


def test_negative_n1_missing_sign_convention_fails_ledger_closed() -> None:
    """N1: a ledger that does not record the eq. (1) sign convention fails
    closed -- a sign-unverified ledger cannot be a genuine residual."""
    wp = build_wp_n1_auluck_power_port_ledger(
        _ledger(record_sign_convention=False)
    )
    assert wp["auluck_eq1_sign_convention_recorded"] is False
    assert wp["ledger_blocked"] is True
    assert wp["ledger_blocker"] == "auluck_eq1_sign_convention_not_recorded"
    assert wp["residual_J"] is None
    assert wp["residual_is_genuine_diagnostic"] is False
    # The eq. (1) string is still emitted (so the gap is visible), but the
    # 'recorded' flag is False.
    assert wp["sign_convention"]["auluck_eq1_sign_convention_recorded"] is False


# --- N2 wrong domain ------------------------------------------------------


def test_negative_n2_wrong_domain_fails_partition_validity() -> None:
    """N2: source interface inside Omega fails the domain-review packet."""
    corrupt_omega = _omega_domain(interface_disjoint=False)
    wp = build_wp_n1_auluck_power_port_ledger(_ledger(omega=corrupt_omega))
    od = wp["auluck_omega_domain"]
    assert od["terminal_source_interface_disjoint_from_omega"] is False
    assert od["partition_valid"] is False
    assert od["can_support_power_port_acceptance"] is False


def test_negative_n2_non_exhaustive_partition_fails_validity() -> None:
    """N2 variant: a shifted Omega that breaks exhaustiveness is detected."""
    corrupt_omega = _omega_domain(exhaustive=False)
    wp = build_wp_n1_auluck_power_port_ledger(_ledger(omega=corrupt_omega))
    assert wp["auluck_omega_domain"]["partition_exhaustive"] is False
    assert wp["auluck_omega_domain"]["partition_valid"] is False


# --- N3 omitted eq. (6) term (NOT an electrode-work term) -----------------


def test_negative_n3_omitting_an_eq6_term_blocks_genuine_residual() -> None:
    """N3 (rewritten): there is no electrode-work term. A six-term ledger
    that omits one of the genuine eq. (6) terms II/IV/V/VI cannot emit a
    genuine residual -- the residual no longer closes by construction.

    The current runtime supplies none of the six terms, so this is the
    standing state: with any term blocked, residual_J is None. This test
    asserts the structural property -- a missing term => no residual --
    rather than relying on a closure estimate to fill the gap (which the
    old five-term ledger did and which WP-N1B forbids)."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    terms = wp["energy_ledger_terms_J"]
    # Pick term V (resistive) as the representative omitted eq. (6) term.
    assert "term_v_resistive_sigma_p_J" in terms
    assert terms["term_v_resistive_sigma_p_J"] is None
    # Because term V (and the others) are not independently computed, the
    # residual is None -- it is NOT closure-derived to force a zero.
    assert wp["all_six_terms_computed_independently"] is False
    assert wp["residual_J"] is None
    assert wp["residual_is_genuine_diagnostic"] is False
    # The blocker is explicit about why the eq. (6) terms are missing.
    assert wp["ledger_blocker"] == (
        "auluck_eq6_terms_fail_closed_pending_wp_n3_geometry"
    )


def test_negative_n3_no_electrode_work_term_in_ledger_or_policy() -> None:
    """N3 closure: the stale 'electrode_interface_work' term is gone from
    both the ledger and the negative-test policy -- Auluck's balance has no
    electrode-contact-work term (verified extract)."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    assert "electrode_interface_work_J" not in wp["energy_ledger_terms_J"]
    assert "electrode_interface_work_J" not in wp
    policy = build_wp_n1_negative_test_policy(None)
    assert "N3_omitted_electrode_work" not in policy["required_negative_tests"]
    assert "N3_omitted_eq6_term" in policy["required_negative_tests"]
    assert policy["no_electrode_work_term"] is True
    assert policy["auluck_eq6_term_count"] == 6


# --- N4 low-current P/I singularity --------------------------------------


def test_negative_n4_low_current_p_over_i_singularity_is_guarded() -> None:
    """N4: I -> 0 with finite P fires the low-current guard, not P/I."""
    udpf_value, udpf_source = _circuit_udpf_for_step(
        mode="lagged_volume_j_dot_e",
        input_udpf_V=7.0,
        lagged_field_work={"j_dot_e_power_W": 1.0e9},
        current_A=0.0,
        min_current_A=1.0,
    )
    # U_DPF is NOT computed as P / I at low current.
    assert udpf_source == "input_sequence_fallback_low_current"
    assert udpf_value == 7.0
    import math

    assert math.isfinite(udpf_value)


def test_negative_n4_low_current_singularity_reported_in_packet() -> None:
    """N4: the emitted packet reports the low-current singularity feedback."""
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "input_sequence_fallback_low_current",
                "low_current_feedback": {
                    "status": (
                        "blocked_low_current_p_over_i_singularity_not_validation"
                    ),
                    "low_current_threshold_hit": True,
                    "singularity_blocked_this_step": True,
                    "can_support_first_principles_acceptance": False,
                },
                "circuit_step": {"current_A": 0.0, "udpf_V": 7.0},
            },
        },
    )
    singularity = packet["low_current_p_over_i_singularity"]
    assert singularity["low_current_threshold_hit"] is True
    assert singularity["singularity_blocked_this_step"] is True
    assert (
        packet["negative_test_policy"][
            "low_current_p_over_i_singularity_rejection_required"
        ]
        is True
    )


# --- N5 first-step fallback ----------------------------------------------


def test_negative_n5_first_step_fallback_marked_in_ledger() -> None:
    """N5: step 0 with no lagged field work is marked fallback, not closed."""
    wp = build_wp_n1_auluck_power_port_ledger(
        _ledger(
            first_step_fallback=True,
            first_step_udpf_source="input_sequence_fallback_first_step",
        )
    )
    assert wp["first_step_fallback"] is True
    assert wp["first_step_udpf_source"] == "input_sequence_fallback_first_step"
    # The fallback step does not produce an accepted ledger.
    assert wp["can_support_power_port_acceptance"] is False
    assert wp["residual_J"] is None


def test_negative_n5_first_step_udpf_source_is_explicit_fallback() -> None:
    """N5: the runtime udpf-source helper marks the uninitialized first step."""
    udpf_value, udpf_source = _circuit_udpf_for_step(
        mode="lagged_volume_j_dot_e",
        input_udpf_V=3.0,
        lagged_field_work=None,
        current_A=1.0e5,
        min_current_A=1.0,
    )
    assert udpf_source == "input_sequence_fallback_first_step"
    assert udpf_value == 3.0


# --- N6 default-mode leakage ---------------------------------------------


def test_negative_n6_default_mode_does_not_leak_into_acceptance() -> None:
    """N6: default input_sequence mode cannot be read as an accepted port."""
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "udpf_source": "input_sequence",
                "circuit_step": {"current_A": 1.0e5, "udpf_V": 1.0e3},
            },
        },
    )
    assert packet["accepted_load_power_source"] == "none"
    assert packet["active_load_relation"] == (
        "input_terminal_voltage_sequence_not_active_load_authority"
    )
    assert (
        packet["active_load_decision"]["can_support_power_port_acceptance"]
        is False
    )
    assert packet["can_support_first_principles_acceptance"] is False
    # The WP-N1B ledger is non-accepting even in default mode.
    wp = packet["wp_n1_auluck_power_port_ledger"]
    assert wp["can_support_power_port_acceptance"] is False
    assert wp["auluck_eq6_decomposition"] is True


def test_negative_n6_negative_test_policy_enumerates_all_six() -> None:
    """N6 closure: the emitted WP-N1B negative-test policy lists all six
    tests, with the rewritten N3 (no electrode-work term)."""
    packet = build_engineering_power_port_packet(None)
    policy = packet["wp_n1_negative_test_policy"]
    assert set(policy["required_negative_tests"].keys()) == {
        "N1_sign_reversal",
        "N2_wrong_domain",
        "N3_omitted_eq6_term",
        "N4_low_current_p_over_i",
        "N5_first_step_fallback",
        "N6_default_mode_leakage",
    }
    assert policy["all_six_required"] is True
    assert policy["can_support_power_port_acceptance"] is False
    assert policy["acceptance_unblocks_only_when"] == (
        "source_backed_residual_tolerance_attached AND "
        "wp_n3_reviewed_sigma_p_geometry AND all_six_tests_pass"
    )


# --- F3: no electrode-work authority wording in the power-port packet -----


def test_f3_no_electrode_work_in_required_power_port_channels() -> None:
    """F3: REQUIRED_POWER_PORT_CHANNELS carries no electrode-work channel --
    Auluck eq. (6) has no electrode-contact-work term."""
    assert "electrode_work" not in REQUIRED_POWER_PORT_CHANNELS
    # The power-balance channel is the Auluck eq. (6) six-term completeness.
    assert "auluck_eq6_power_balance" in REQUIRED_POWER_PORT_CHANNELS
    for channel in REQUIRED_POWER_PORT_CHANNELS:
        assert "electrode_work" not in channel, channel


def test_f3_no_electrode_work_in_acceptance_blocking_channels() -> None:
    """F3: ACCEPTANCE_BLOCKING_CHANNELS no longer lists
    electrode_work_partition; the blocking requirement is eq. (6) term
    completeness."""
    assert "electrode_work_partition" not in ACCEPTANCE_BLOCKING_CHANNELS
    assert "auluck_eq6_six_term_completeness" in ACCEPTANCE_BLOCKING_CHANNELS
    for channel in ACCEPTANCE_BLOCKING_CHANNELS:
        assert "electrode_work" not in channel, channel


def test_f3_acceptance_gate_states_auluck_eq6_term_completeness() -> None:
    """F3: the emitted top-level acceptance_gate string drops electrode_work
    and instead names Auluck eq. (6) term completeness (term_i..term_vi)."""
    packet = build_engineering_power_port_packet(None)
    gate = packet["acceptance_gate"]
    assert "electrode_work" not in gate
    assert "auluck_eq6_term_completeness" in gate
    assert "term_i_through_term_vi" in gate
    assert "independently_computed" in gate


def test_f3_negative_test_policy_drops_electrode_work_omission() -> None:
    """F3: the emitted negative_test_policy replaces
    electrode_work_omission_required with an Auluck-eq.(6)-term-omission
    policy."""
    packet = build_engineering_power_port_packet(None)
    policy = packet["negative_test_policy"]
    assert "electrode_work_omission_required" not in policy
    assert policy["auluck_eq6_term_omission_required"] is True
    for key in policy:
        assert "electrode_work" not in key, key


def test_f3_no_electrode_work_in_required_channels_or_policy() -> None:
    """F3 closure: no authoritative WP-N1B required channel, acceptance
    gate, or negative-test policy key contains 'electrode_work'."""
    packet = build_engineering_power_port_packet(None)
    assert all(
        "electrode_work" not in c for c in packet["required_channels"]
    )
    assert all(
        "electrode_work" not in c
        for c in packet["acceptance_blocking_channels"]
    )
    assert "electrode_work" not in packet["acceptance_gate"]
    assert all(
        "electrode_work" not in k for k in packet["negative_test_policy"]
    )
    # The WP-N1B six-negative-test policy keeps no electrode-work test.
    wp_policy = packet["wp_n1_negative_test_policy"]
    assert all(
        "electrode_work" not in k
        for k in wp_policy["required_negative_tests"]
    )
    assert wp_policy["no_electrode_work_term"] is True


# --- F5: verified Auluck extract in the top-level source references -------


def test_f5_power_port_source_refs_include_verified_auluck_extract() -> None:
    """F5: POWER_PORT_SOURCE_REFS names the verified Auluck extract with the
    role auluck_eq1_eq5_eq6_verified_power_balance."""
    roles = {ref["role"] for ref in POWER_PORT_SOURCE_REFS}
    assert "auluck_eq1_eq5_eq6_verified_power_balance" in roles
    verified = next(
        ref
        for ref in POWER_PORT_SOURCE_REFS
        if ref["role"] == "auluck_eq1_eq5_eq6_verified_power_balance"
    )
    assert "AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md" in verified[
        "path"
    ]


def test_f5_emitted_packet_source_references_include_verified_auluck() -> None:
    """F5: the emitted power-port packet's source_references include the
    verified Auluck extract, not just the OCR-garbled extract."""
    packet = build_engineering_power_port_packet(None)
    refs = packet["source_references"]
    paths = [ref["path"] for ref in refs]
    assert any(
        "AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md" in path
        for path in paths
    )
    roles = {ref["role"] for ref in refs}
    assert "auluck_eq1_eq5_eq6_verified_power_balance" in roles


# ===========================================================================
# S3.3 -- WP-N3 Sigma_p surface packet plumbing into power_port.py.
#
# Terms II/IV/V/VI consume a SigmaPSurfacePacket only -- never I*V minus the
# others. Missing Sigma_p blocks II/IV/V/VI; missing v blocks II/IV/VI;
# missing eta blocks V. S3.3 is plumbing only: it does NOT compute the surface
# integrals (Sprint 4), so even a fully-operand packet leaves the terms
# blocked and the residual None. Structures are imported by full dotted path.
# ===========================================================================


_SIGN_CONV_SENTINEL = object()  # sentinel: caller did not pass sign_convention


def _sigma_p_packet_with(
    *,
    velocity: str = "blocked",
    resistivity: str = "blocked",
    n_faces: int = 3,
    sign_convention: Any = _SIGN_CONV_SENTINEL,
    moving_classification_status: str = "available",
) -> _SigmaPSurfacePacket:
    """Return a Sigma_p packet with a present face set and tunable v/eta.

    S3R.5: now also accepts ``sign_convention`` and
    ``moving_classification_status`` so negative tests can control them
    explicitly.  Default moving_classification_status is ``"available"`` so
    tests that only care about v/eta behaviour are not affected by the S3R.5
    moving-classification gate.

    Pass ``sign_convention=None`` explicitly to create a packet with no sign
    convention (triggers the sign-convention blocker negative test).
    Default (sentinel) uses ``{"eq6_term_signs": {"term_ii": "+"}}`` so
    existing tests that don't care about sign_convention continue to pass.
    """
    if sign_convention is _SIGN_CONV_SENTINEL:
        sign_convention = {"eq6_term_signs": {"term_ii": "+"}}
    return _SigmaPSurfacePacket(
        status="candidate_sigma_p_surface_packet_not_validation",
        source_refs=("WP_N3_SIGMA_P_RUNTIME_INTERFACE_SPEC.md:1-439",),
        source_geometry_packet_id="pf1000_geometry_packet_krauz2012",
        source_geometry_hash="deadbeef",
        n_sigma_p_faces=n_faces,
        face_count_total_sigma=n_faces + 5,
        geometry_review_status="geometry_candidate_not_reviewed",
        face_ids=np.arange(n_faces),
        dS_outward_m2=np.zeros((n_faces, 3)),
        face_area_m2=np.ones(n_faces),
        outward_normal=np.zeros((n_faces, 3)),
        face_material_class=tuple("plasma" for _ in range(n_faces)),
        is_moving=np.ones(n_faces, dtype=bool),
        omega_side="omega_interior",
        excluded_interface_side="terminal_source_interface_excluded",
        outward_normal_convention="outward_from_omega",
        field_sampler_status={
            "B": "available", "E": "available", "J": "available",
        },
        velocity_status=velocity,
        resistivity_status=resistivity,
        centering={"time_centering": "candidate_step_consistent_not_accepted"},
        quadrature="midpoint_one_point_per_face",
        sign_convention=sign_convention,
        operand_blockers={
            "v": "material_velocity_v_not_available_on_sigma_p_faces",
            "eta": "resistivity_eta_not_available_on_sigma_p_faces",
        },
        moving_classification_status=moving_classification_status,
    )


def test_s33_default_ledger_emits_a_sigma_p_surface_packet() -> None:
    """The ledger always carries a Sigma_p surface packet (blocked by default)."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    assert "sigma_p_surface_packet" in wp
    assert wp["sigma_p_surface_packet_status"].startswith("blocked_")
    assert wp["sigma_p_surface_packet"]["n_sigma_p_faces"] == 0


def test_s33_missing_sigma_p_blocks_terms_ii_iv_v_vi() -> None:
    """With no Sigma_p face set all four moving-boundary terms fail closed."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    packets = wp["energy_ledger_term_packets"]
    for key in _SIGMA_P_TERMS:
        assert packets[key]["status"] == "blocked"
        assert packets[key]["value_J"] is None
        assert packets[key]["missing_operand"] == "sigma_p"
        assert packets[key]["blocker"] == (
            "sigma_p_face_set_not_available_requires_wp_n3_reviewed_geometry"
        )


def test_s33_missing_velocity_blocks_ii_iv_vi_but_not_v() -> None:
    """A Sigma_p packet with no v blocks II/IV/VI on v; V blocks on eta."""
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="blocked", resistivity="available"
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    packets = wp["energy_ledger_term_packets"]
    for key in (
        "term_ii_motional_magnetic_sigma_p_J",
        "term_iv_motional_electric_sigma_p_J",
        "term_vi_anomalous_poloidal_sigma_p_J",
    ):
        assert packets[key]["missing_operand"] == "v"
        assert packets[key]["blocker"] == (
            "material_velocity_v_not_available_on_sigma_p_faces"
        )
    # term V does not use v; with eta available it advances past v/eta.
    assert packets["term_v_resistive_sigma_p_J"]["missing_operand"] != "v"


def test_s33_missing_eta_blocks_only_term_v() -> None:
    """A Sigma_p packet with no eta blocks V on eta; II/IV/VI block on v only."""
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="blocked", resistivity="blocked"
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    packets = wp["energy_ledger_term_packets"]
    assert packets["term_v_resistive_sigma_p_J"]["missing_operand"] == "eta"
    assert packets["term_v_resistive_sigma_p_J"]["blocker"] == (
        "resistivity_eta_not_available_on_sigma_p_faces"
    )
    for key in (
        "term_ii_motional_magnetic_sigma_p_J",
        "term_iv_motional_electric_sigma_p_J",
        "term_vi_anomalous_poloidal_sigma_p_J",
    ):
        assert packets[key]["missing_operand"] == "v"


def test_s33_sigma_p_terms_consume_packet_not_iv_work() -> None:
    """Every Sigma_p term packet records that it consumes the Sigma_p packet."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    packets = wp["energy_ledger_term_packets"]
    for key in _SIGMA_P_TERMS:
        assert packets[key]["consumes"] == "sigma_p_surface_packet"
        # never derived by closure from the terminal I*V work.
        assert packets[key].get("derived_by_closure") is not True


def test_s33_full_operand_packet_still_blocks_terms_integral_is_sprint4() -> None:
    """With Sigma_p + v + eta all present the terms stay blocked (S3.3 plumbing).

    S3.3 does NOT compute the surface integral -- the integral is Sprint 4
    work, so the term value stays None and the blocker says so explicitly.
    """
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="available", resistivity="available"
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    packets = wp["energy_ledger_term_packets"]
    for key in _SIGMA_P_TERMS:
        assert packets[key]["status"] == "blocked"
        assert packets[key]["value_J"] is None
        assert packets[key]["operands_available"] is True
        assert packets[key]["blocker"] == (
            "sigma_p_surface_integral_is_sprint4_work_s3_3_is_plumbing_only"
        )


def test_s33_residual_stays_none_even_with_full_operand_sigma_p_packet() -> None:
    """The six-term residual is None while II/IV/V/VI stay blocked.

    No term is computed as I*V minus the others; with Sigma_p terms blocked
    the residual cannot be a genuine diagnostic.
    """
    ledger = _ledger(
        stored_magnetic_delta_J=10.0, stored_electric_delta_J=2.0
    )
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="available", resistivity="available"
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    assert wp["residual_J"] is None
    assert wp["all_six_terms_computed_independently"] is False
    # terms I and III remain independently computed -- S3.3 preserves them.
    packets = wp["energy_ledger_term_packets"]
    assert packets["term_i_stored_magnetic_energy_rate_J"][
        "computed_independently"
    ] is True
    assert packets["term_iii_stored_electric_energy_rate_J"][
        "computed_independently"
    ] is True


def test_s33_terms_i_iii_stay_independent_with_sigma_p_packet_present() -> None:
    """A Sigma_p packet never disturbs the independently-computed terms I/III."""
    ledger = _ledger(
        stored_magnetic_delta_J=7.0, stored_electric_delta_J=1.5
    )
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with()
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    terms = wp["energy_ledger_terms_J"]
    assert terms["term_i_stored_magnetic_energy_rate_J"] == 7.0
    assert terms["term_iii_stored_electric_energy_rate_J"] == 1.5


def test_s33_blocked_sigma_p_packet_cannot_support_acceptance() -> None:
    """A Sigma_p packet may never lift power-port or first-principles acceptance."""
    sp = _sigma_p_packet_with(velocity="available", resistivity="available")
    assert sp.can_support_power_port_acceptance is False
    assert sp.can_support_first_principles_acceptance is False
    # forcing the acceptance flag is rejected at construction.
    with pytest.raises(ValueError, match="must not claim"):
        _dc_replace(sp, can_support_first_principles_acceptance=True)


def test_s3r5_dict_form_packet_is_not_silently_ignored() -> None:
    """S3R.5 A7: a dict-form sigma_p_surface_packet must NOT be silently
    ignored. When all required fields are present it must be reconstructed into
    a SigmaPSurfacePacket; otherwise a named blocker must be emitted.
    """
    sp_instance = _sigma_p_packet_with(velocity="available", resistivity="available")
    d = sp_instance.to_dict()
    ledger = {"sigma_p_surface_packet": d}
    reconstructed = _sigma_p_packet_from_ledger(ledger)
    # Must come back as a SigmaPSurfacePacket, not None or a default block.
    assert isinstance(reconstructed, _SigmaPSurfacePacket)
    # The reconstructed packet must carry the same face count.
    assert reconstructed.n_sigma_p_faces == sp_instance.n_sigma_p_faces
    # The status must NOT be the named "not supported" blocker.
    assert reconstructed.status != _SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED


def test_s3r5_dict_form_packet_missing_required_field_emits_named_blocker() -> None:
    """S3R.5 A7: a dict-form packet missing required fields must emit a named
    'serialized_sigma_p_packet_not_supported' blocker -- not a silent discard.
    """
    sp_instance = _sigma_p_packet_with()
    d = sp_instance.to_dict()
    # Remove a required field to trigger reconstruction failure.
    del d["n_sigma_p_faces"]
    ledger = {"sigma_p_surface_packet": d}
    result = _sigma_p_packet_from_ledger(ledger)
    # Must emit the named blocker, not silently proceed.
    assert isinstance(result, _SigmaPSurfacePacket)
    assert result.status == _SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED, (
        f"expected named blocker {_SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED!r}, "
        f"got {result.status!r}"
    )


def test_s3r5_missing_sign_convention_blocks_sigma_p_terms() -> None:
    """S3R.5 A7: a Sigma_p packet with no sign_convention must block all four
    Sigma_p terms with the sign-convention blocker -- even when all other
    operands (face set, v, eta, B, E, J) are available.
    """
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="available",
        resistivity="available",
        sign_convention=None,  # no sign convention
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    packets = wp["energy_ledger_term_packets"]
    for key in _SIGMA_P_TERMS:
        packet = packets[key]
        assert packet["status"] == "blocked", key
        assert packet["missing_operand"] == "sign_convention", key
        assert "sign_convention" in packet["blocker"], (
            f"{key}: expected sign_convention blocker, got {packet['blocker']!r}"
        )
    # sign_convention gap must not produce a genuine residual.
    assert wp["residual_J"] is None


def test_s3r5_absent_moving_classification_blocks_sigma_p_terms() -> None:
    """S3R.5 A7: a Sigma_p packet whose moving_classification_status is
    'not_classified' must block all four Sigma_p terms -- Auluck p.8 requires
    stationary boundaries to contribute zero; without classification the
    distinction cannot be enforced.
    """
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="available",
        resistivity="available",
        moving_classification_status="not_classified",
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    packets = wp["energy_ledger_term_packets"]
    for key in _SIGMA_P_TERMS:
        packet = packets[key]
        assert packet["status"] == "blocked", key
        assert packet["missing_operand"] == "moving_classification", key
    # Without classification, the residual must stay None.
    assert wp["residual_J"] is None


def test_s3r5_full_operand_presence_does_not_compute_terms_before_sprint4() -> None:
    """S3R.5 A7: even with every operand (face set, v, eta, B, E, J, sign
    convention, moving classification) present the four Sigma_p terms must
    remain blocked -- the surface integral is Sprint 4 work and S3.3 is
    plumbing only.

    This is the definitive gate: full operand presence NEVER authorises a
    Sprint 3 term value.
    """
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="available",
        resistivity="available",
        moving_classification_status="available",
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    packets = wp["energy_ledger_term_packets"]
    for key in _SIGMA_P_TERMS:
        assert packets[key]["status"] == "blocked", (
            f"{key}: must stay blocked even with all operands present"
        )
        assert packets[key]["value_J"] is None, key
        # The blocker must name the Sprint 4 reason, not a missing operand.
        assert "sprint4" in packets[key]["blocker"], (
            f"{key}: blocker must cite sprint4 when all operands present"
        )
        assert packets[key].get("operands_available") is True, key
    # The residual must still be None with all four Sigma_p terms blocked.
    assert wp["residual_J"] is None
    assert wp["all_six_terms_computed_independently"] is False


# ===========================================================================
# S8-WS6 -- power-port and Sigma_p operator ledger.
#
# WS6 adds: (a) an explicit six-term-presence roster (each Auluck eq. (6) term
# independently present or fail-closed); (b) four explicit ledger-structure
# fields -- sign_convention / time_centering / domain / residual; (c) demotion
# of every active-load placeholder to engineering-only telemetry that cannot
# satisfy accepted power coupling. These tests enforce all three.
# Authority: AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (1), (6).
# ===========================================================================


def test_ws6_ledger_has_four_explicit_structure_fields() -> None:
    """WS6: the ledger structure carries the four explicit named fields
    sign_convention / time_centering / domain / residual, both at top level
    and grouped in a coherent power_port_ledger_fields block."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    for field in ("sign_convention", "time_centering", "domain", "residual"):
        assert field in wp, f"top-level ledger field {field!r} missing"
        assert field in wp["power_port_ledger_fields"], (
            f"grouped ledger field {field!r} missing"
        )


def test_ws6_four_structure_fields_present_when_no_ledger() -> None:
    """WS6: the four explicit fields exist even with no telemetry, so the
    ledger structure shape is invariant."""
    wp = build_wp_n1_auluck_power_port_ledger(None)
    for field in ("sign_convention", "time_centering", "domain", "residual"):
        assert field in wp, f"no-telemetry ledger field {field!r} missing"
        assert field in wp["power_port_ledger_fields"], field


def test_ws6_domain_field_names_omega_and_excludes_source_interface() -> None:
    """WS6 domain field: integration domain is Omega and the
    electrode/power-source interface is explicitly excluded (Auluck p.6-7)."""
    domain = build_wp_n1_auluck_power_port_ledger(_ledger())["domain"]
    assert domain["integration_domain"] == "Omega"
    assert domain["moving_boundary"] == "Sigma_p"
    assert "excluded_from_domain" in domain
    assert "interface" in domain["excluded_from_domain"].lower()
    assert domain["can_support_power_port_acceptance"] is False


def test_ws6_residual_field_tolerance_fails_closed_not_source_backed() -> None:
    """WS6 residual field: Auluck supplies no balance tolerance, so the
    accepted residual tolerance is not attached and a fail-closed blocker is
    named. The residual cannot gate acceptance."""
    residual = build_wp_n1_auluck_power_port_ledger(_ledger())["residual"]
    assert residual["accepted_residual_tolerance"] == "not_attached"
    assert "not_source_backed" in residual["residual_tolerance_blocker"]
    assert residual["is_closure_by_construction"] is False
    assert residual["can_support_power_port_acceptance"] is False


def test_ws6_six_term_presence_lists_all_six_in_eq6_order() -> None:
    """WS6: the six-term-presence roster lists exactly the six Auluck eq. (6)
    terms, each tagged with its Auluck term label I-VI."""
    presence = build_wp_n1_auluck_power_port_ledger(
        _ledger()
    )["auluck_eq6_six_term_presence"]
    assert presence["expected_term_count"] == 6
    assert set(presence["terms"]) == set(_WP_N1B_LEDGER_KEYS)
    labels = {p["auluck_term_label"] for p in presence["terms"].values()}
    assert labels == {"I", "II", "III", "IV", "V", "VI"}


def test_ws6_six_term_presence_all_blocked_no_split_no_sigma_p() -> None:
    """WS6: with no stored-EM split and no Sigma_p face set, every one of the
    six terms is `blocked_fail_closed` and the balance is not complete."""
    presence = build_wp_n1_auluck_power_port_ledger(
        _ledger()
    )["auluck_eq6_six_term_presence"]
    assert presence["independent_term_count"] == 0
    assert presence["all_six_terms_present"] is False
    for key, entry in presence["terms"].items():
        assert entry["state"] == "blocked_fail_closed", key
        assert entry["blocker"] is not None, key


def test_ws6_missing_sigma_p_face_set_blocks_terms_ii_iv_v_vi() -> None:
    """WS6 test: a missing Sigma_p face set blocks power-port acceptance --
    terms II/IV/V/VI fail closed naming the missing face set."""
    presence = build_wp_n1_auluck_power_port_ledger(
        _ledger()
    )["auluck_eq6_six_term_presence"]
    for key in _SIGMA_P_TERMS:
        entry = presence["terms"][key]
        assert entry["state"] == "blocked_fail_closed", key
        assert "sigma_p_face_set_not_available" in entry["blocker"], key


def test_ws6_missing_face_velocity_blocks_terms_ii_iv_vi() -> None:
    """WS6 test: missing face velocity v blocks terms II, IV, VI (the
    motional and anomalous Sigma_p terms) -- not V."""
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="blocked", resistivity="available"
    )
    packets = build_wp_n1_auluck_power_port_ledger(
        ledger
    )["energy_ledger_term_packets"]
    for key in (
        "term_ii_motional_magnetic_sigma_p_J",
        "term_iv_motional_electric_sigma_p_J",
        "term_vi_anomalous_poloidal_sigma_p_J",
    ):
        assert packets[key]["status"] == "blocked", key
        assert packets[key]["missing_operand"] == "v", key


def test_ws6_missing_resistivity_blocks_term_v() -> None:
    """WS6 test: missing resistivity eta blocks term V (resistive Sigma_p)."""
    ledger = _ledger()
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="available", resistivity="blocked"
    )
    packets = build_wp_n1_auluck_power_port_ledger(
        ledger
    )["energy_ledger_term_packets"]
    term_v = packets["term_v_resistive_sigma_p_J"]
    assert term_v["status"] == "blocked"
    assert term_v["missing_operand"] == "eta"


def test_ws6_active_load_placeholders_demoted_engineering_only() -> None:
    """WS6: the engineering packet demotes every active-load placeholder to
    engineering-only telemetry; none satisfies accepted power coupling."""
    packet = build_engineering_power_port_packet(
        {
            "last": {
                "circuit_step": {"current_A": 1.0e5, "udpf_V": 2.0e4},
            }
        },
        conservation={"final": {"magnetic_energy_J": 5.0e3}},
    )
    demotion = packet["active_load_placeholder_demotion"]
    assert demotion["satisfies_accepted_power_coupling"] is False
    assert demotion["any_placeholder_satisfies_accepted_power_coupling"] is False
    assert demotion["accepted_load_power_source"] == "none"
    assert demotion["can_support_first_principles_acceptance"] is False
    for name, entry in demotion["placeholders"].items():
        assert entry["channel_state"] == "excluded_not_validated", name


def test_ws6_diagnostic_inductance_cannot_be_accepted_load() -> None:
    """WS6 test: the diagnostic field inductance (L = 2 E_B / I^2) is an
    active-load fallback and CANNOT satisfy accepted power coupling."""
    packet = build_engineering_power_port_packet(
        {"last": {"circuit_step": {"current_A": 1.0e5, "udpf_V": 2.0e4}}},
        conservation={"final": {"magnetic_energy_J": 5.0e3}},
    )
    demotion = packet["active_load_placeholder_demotion"]
    ind = demotion["placeholders"]["diagnostic_field_inductance_H"]
    # The fallback value is computed (engineering telemetry) ...
    assert ind["value"] is not None
    # ... but it is excluded_not_validated and cannot couple power.
    assert ind["channel_state"] == "excluded_not_validated"
    assert demotion["satisfies_accepted_power_coupling"] is False
    # And the legacy authority tag still says diagnostic-only-not-load.
    assert (
        packet["magnetic_energy_inductance_authority"]
        == "diagnostic_only_not_circuit_load"
    )


def test_ws6_terminal_iv_product_is_not_accepted_power_coupling() -> None:
    """WS6 test: the terminal I*V product (active_power_W) is an active-load
    placeholder; it is engineering telemetry and is not accepted load power."""
    packet = build_engineering_power_port_packet(
        {"last": {"circuit_step": {"current_A": 1.0e5, "udpf_V": 2.0e4}}},
    )
    demotion = packet["active_load_placeholder_demotion"]
    iv = demotion["placeholders"]["active_power_W_terminal_iv_product"]
    assert iv["value"] == pytest.approx(1.0e5 * 2.0e4)
    assert iv["channel_state"] == "excluded_not_validated"
    # active_power_W is exposed for engineering review but the accepted load
    # source stays 'none'.
    assert packet["accepted_load_power_source"] == "none"
    assert (
        packet["active_load_decision"]["can_support_power_port_acceptance"]
        is False
    )


def test_ws6_power_port_authority_blocked_until_all_six_plus_residual() -> None:
    """WS6 exit criterion: accepted power-port authority stays blocked until
    all six terms AND the residual tolerance are source-backed. A fully
    operand Sigma_p packet does not unlock it (integral is Sprint 4; Auluck
    supplies no residual tolerance)."""
    ledger = _ledger(stored_magnetic_delta_J=10.0, stored_electric_delta_J=1.0)
    ledger["sigma_p_surface_packet"] = _sigma_p_packet_with(
        velocity="available", resistivity="available"
    )
    wp = build_wp_n1_auluck_power_port_ledger(ledger)
    presence = wp["auluck_eq6_six_term_presence"]
    # Terms I/III present, II/IV/V/VI still blocked -> not all six.
    assert presence["all_six_terms_present"] is False
    assert wp["residual"]["accepted_residual_tolerance"] == "not_attached"
    assert wp["can_support_power_port_acceptance"] is False
    assert wp["can_support_first_principles_acceptance"] is False
