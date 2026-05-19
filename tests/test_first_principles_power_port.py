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

from typing import Any

from dpf.fields.hybrid_simulator import _circuit_udpf_for_step
from dpf.first_principles.power_port import (
    _WP_N1B_LEDGER_KEYS,
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
) -> dict[str, Any]:
    """Return a synthetic simulator-emitted power_port_ledger.

    The runtime emits the terminal I*V cumulative work and the Omega
    partition, but NOT a Sigma_p moving-boundary face set, NOT v/eta on
    Sigma_p, and NOT the magnetic/electric stored-energy split -- exactly the
    runtime state investigated for WP-N1B. So this fixture cannot supply any
    eq. (6) term independently; every term must fail closed.
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


def test_wp_n1b_all_six_terms_fail_closed_pending_runtime() -> None:
    """No eq. (6) term can be computed: every term fails closed, value None."""
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


def test_wp_n1b_terms_i_iii_fail_closed_no_magnetic_electric_split() -> None:
    """Terms I and III need the magnetic / electric stored-energy split,
    which the runtime does not expose -- they fail closed, not closure-
    derived from the combined stored-EM energy."""
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
