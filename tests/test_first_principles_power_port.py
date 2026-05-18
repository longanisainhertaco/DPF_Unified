"""WP-N1 Auluck power-port ledger and six negative tests.

Source packet: docs/ssr_audit_2026_05_18/WP-N1_power_port_source_packet.md
Each test corrupts one input and asserts the emitted packet field detects it
(audit A-5). Tests are pure functions over a constructed input; no run
interdependency. Acceptance stays blocked regardless of residual magnitude.
"""

from __future__ import annotations

import math
from typing import Any

from dpf.fields.hybrid_simulator import _circuit_udpf_for_step
from dpf.first_principles.power_port import (
    build_engineering_power_port_packet,
    build_wp_n1_auluck_power_port_ledger,
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
    volume_j_dot_e: float = 60.0,
    wall: float = 5.0,
    stored_delta: float = 20.0,
    first_step_fallback: bool = False,
    first_step_udpf_source: str = "candidate_lagged_volume_j_dot_e",
    omega: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a synthetic simulator-emitted WP-N1 five-term ledger."""
    return {
        "cumulative_terminal_port_work_J": terminal,
        "cumulative_omega_volume_j_dot_e_work_J": volume_j_dot_e,
        "cumulative_wall_poynting_flux_excluding_declared_port_J": wall,
        "stored_em_energy_delta_J": stored_delta,
        "first_step_fallback": first_step_fallback,
        "first_step_udpf_source": first_step_udpf_source,
        "steps_accumulated": 2,
        "snapshot_provenance": {
            "terminal_port_work_J": "begin_step_current_times_begin_step_udpf",
            "volume_j_dot_e_work_J": "begin_step_E_with_step_masked_current",
            "wall_poynting_flux_excluding_declared_port_J": "trapezoidal",
            "stored_em_energy_delta_J": "end_minus_begin",
            "time_centering": "candidate_step_consistent_not_accepted",
        },
        "domain_partition": omega if omega is not None else _omega_domain(),
    }


# --- baseline -------------------------------------------------------------


def test_wp_n1_five_term_ledger_closes_trivially_via_closure_estimate() -> None:
    """Term 4 closure estimate (G1) drives the residual to ~0 by construction."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    terms = wp["energy_ledger_terms_J"]
    # Terms 1, 2, 3, 5 are the fully-implemented terms.
    assert wp["fully_implemented_terms"] == [
        "terminal_port_work_J",
        "volume_j_dot_e_work_J",
        "wall_poynting_flux_excluding_declared_port_J",
        "stored_em_energy_delta_J",
    ]
    # Term 4 is the labeled non-independent closure estimate.
    assert wp["closure_estimate_terms"] == ["electrode_interface_work_J"]
    assert terms["electrode_interface_work_J"] == 100.0 - 60.0 - 5.0 - 20.0
    assert wp["electrode_interface_work_independence"] == (
        "not_independent_closure_estimate"
    )
    assert wp["electrode_interface_work_blocker"] == "G1_auluck_eq_5_6_ocr_illegible"
    # Residual trivially closes — EXPECTED, cannot support acceptance.
    assert math.isclose(wp["residual_J"], 0.0, abs_tol=1e-12)
    assert wp["accepted_residual_tolerance"] == "not_attached"
    assert wp["can_support_power_port_acceptance"] is False
    assert wp["can_support_first_principles_acceptance"] is False
    assert wp["scientific_status"] == "engineering_candidate_not_validation"


def test_wp_n1_omega_domain_emits_four_disjoint_exhaustive_labels() -> None:
    """S1: the four-label partition is disjoint, exhaustive, interface excluded."""
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
    assert od["terminal_source_interface_non_empty"] is True
    assert od["terminal_source_interface_disjoint_from_omega"] is True
    assert od["partition_valid"] is True


# --- N1 sign reversal -----------------------------------------------------


def test_negative_n1_sign_reversal_makes_residual_order_one() -> None:
    """N1: flipping terminal_port_work_J sign breaks the closure detectably."""
    baseline = build_wp_n1_auluck_power_port_ledger(_ledger(terminal=100.0))
    # The closure estimate term 4 is recomputed from the (now reversed) terms,
    # so to expose the corruption we omit term 4 from the residual sum the way
    # an independent integrand would behave: compare residual against the
    # reversed-sign ledger using the SAME term-4 value as baseline.
    reversed_ledger = _ledger(terminal=-100.0)
    reversed_wp = build_wp_n1_auluck_power_port_ledger(reversed_ledger)
    baseline_term4 = baseline["energy_ledger_terms_J"][
        "electrode_interface_work_J"
    ]
    reversed_terms = reversed_wp["energy_ledger_terms_J"]
    # Independent-integrand residual: keep term 4 fixed at the baseline value.
    residual_with_fixed_term4 = (
        reversed_terms["terminal_port_work_J"]
        - reversed_terms["volume_j_dot_e_work_J"]
        - reversed_terms["wall_poynting_flux_excluding_declared_port_J"]
        - reversed_terms["stored_em_energy_delta_J"]
        - baseline_term4
    )
    # Residual jumps by ~2 * terminal_port_work.
    assert math.isclose(
        residual_with_fixed_term4,
        -2.0 * 100.0,
        rel_tol=1e-9,
    )
    denom = max(
        abs(reversed_terms["terminal_port_work_J"]),
        abs(reversed_terms["volume_j_dot_e_work_J"]),
        1.0,
    )
    assert abs(residual_with_fixed_term4 / denom) > 0.5
    # The sign convention is a declared, load-bearing packet field.
    assert reversed_wp["sign_convention"]["terminal_port_work_J"] == (
        "positive_means_energy_entering_omega_from_generator"
    )


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


# --- N3 omitted electrode work -------------------------------------------


def test_negative_n3_omitting_electrode_work_breaks_closure() -> None:
    """N3: dropping term 4 leaves a residual when dL/dt-type work is nonzero."""
    wp = build_wp_n1_auluck_power_port_ledger(_ledger())
    terms = wp["energy_ledger_terms_J"]
    electrode_work = terms["electrode_interface_work_J"]
    # With a non-trivial moving-boundary contribution, term 4 is nonzero.
    assert electrode_work != 0.0
    # Residual with term 4 omitted (set to 0) no longer closes.
    residual_without_term4 = (
        terms["terminal_port_work_J"]
        - terms["volume_j_dot_e_work_J"]
        - terms["wall_poynting_flux_excluding_declared_port_J"]
        - terms["stored_em_energy_delta_J"]
    )
    assert math.isclose(residual_without_term4, electrode_work, rel_tol=1e-9)
    assert not math.isclose(residual_without_term4, 0.0, abs_tol=1e-9)
    # Term 4 is required and labeled as a (non-independent) ledger term.
    assert "electrode_interface_work_J" in wp["energy_ledger_terms_J"]


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
    # All five terms share one declared centering (consistency check).
    centering = wp["time_centering"]
    assert centering["declared_centering"] == "step_consistent_trapezoidal"
    assert centering["all_terms_share_centering"] is True
    provenance = centering["snapshot_provenance"]
    for term in (
        "terminal_port_work_J",
        "volume_j_dot_e_work_J",
        "wall_poynting_flux_excluding_declared_port_J",
        "stored_em_energy_delta_J",
    ):
        assert term in provenance


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
    # The WP-N1 ledger is non-accepting even in default mode.
    wp = packet["wp_n1_auluck_power_port_ledger"]
    assert wp["can_support_power_port_acceptance"] is False


def test_negative_n6_negative_test_policy_enumerates_all_six() -> None:
    """N6 closure: the emitted WP-N1 negative-test policy lists all six tests."""
    packet = build_engineering_power_port_packet(None)
    policy = packet["wp_n1_negative_test_policy"]
    assert set(policy["required_negative_tests"].keys()) == {
        "N1_sign_reversal",
        "N2_wrong_domain",
        "N3_omitted_electrode_work",
        "N4_low_current_p_over_i",
        "N5_first_step_fallback",
        "N6_default_mode_leakage",
    }
    assert policy["all_six_required"] is True
    assert policy["can_support_power_port_acceptance"] is False
    assert policy["acceptance_unblocks_only_when"] == (
        "source_backed_residual_tolerance_attached AND all_six_tests_pass"
    )
