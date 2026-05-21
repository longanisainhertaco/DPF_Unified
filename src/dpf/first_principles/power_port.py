"""Fail-closed power-port packets for package-native first-principles runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace as _dc_replace
from typing import Any

import numpy as _np

from dpf.fields.source_geometry import (
    SIGMA_P_BLOCKERS,
    SIGMA_P_TERM_OPERANDS,
    SigmaPSurfacePacket,
    build_sigma_p_surface_packet,
)

POWER_PORT_SOURCE_REFS = (
    {
        "path": (
            "docs/external_team_submissions/"
            "2026_05_18_three_sprint_blocker_packet/sprint_2/"
            "AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md"
        ),
        "lines": "43-103,104-128,147-156",
        "role": "auluck_eq1_eq5_eq6_verified_power_balance",
    },
    {
        "path": "KnowledgeReference/auluck-2021-dpf-circuit-element.md",
        "lines": "151-200,206-262,426-445,1026-1047",
        "role": "field_power_contract",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "740-805,992-1005",
        "role": "hybrid_pic_circuit_pattern",
    },
    {
        "path": "KnowledgeReference/beresnyak_2018_dpf_hawk_simulations.md",
        "lines": "170-200",
        "role": "mhd_circuit_pattern",
    },
    {
        "path": "KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md",
        "lines": "44-72",
        "role": "poynting_flux_power_transmission_context",
    },
)

# WP-N1B: the required power-port channels. There is NO `electrode_work`
# channel -- the verified Auluck extract is explicit that the balance has no
# electrode-contact-work term; the electrode/power-source interface is
# EXCLUDED from Omega and its Poynting flux IS the LHS I(t)V(t). The eq. (6)
# power balance is carried by `auluck_eq6_power_balance` (term completeness:
# term_i through term_vi independently computed).
# [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (6) p.8;
# "What this source ... DOES NOT provide"]
REQUIRED_POWER_PORT_CHANNELS = (
    "interface_surface_or_volume_domain",
    "terminal_current",
    "terminal_voltage",
    "poynting_power_or_j_dot_e",
    "auluck_eq6_power_balance",
    "external_circuit_energy",
    "magnetic_energy",
    "electric_energy",
    "thermal_energy",
    "kinetic_energy",
    "particle_energy",
    "radiation_energy",
    "residual",
    "sign_convention",
    "time_centering",
    "boundary_labels",
    "startup_handoff_interval",
    "source_references",
)

# WP-N1B acceptance-blocking channels. The Auluck eq. (6) power-balance
# completeness blocker replaces the legacy `electrode_work_partition`: Auluck
# has no electrode-contact-work term, so the blocking requirement is that all
# six eq. (6) terms (term_i through term_vi) are independently computed.
ACCEPTANCE_BLOCKING_CHANNELS = (
    "named_interface_surface_or_volume_domain",
    "poynting_or_j_dot_e_power_integral",
    "auluck_eq6_six_term_completeness",
    "accepted_sign_convention",
    "accepted_time_centering",
    "residual_tolerance",
    "same_scope_power_port_review",
)

ENERGY_LEDGER_CHANNELS = (
    ("external_circuit_energy", "circuit_energy_J"),
    ("magnetic_energy", "magnetic_energy_J"),
    ("electric_energy", "electric_energy_J"),
    ("thermal_energy", "electron_internal_energy_J"),
    ("particle_energy", "particle_kinetic_energy_J"),
    ("kinetic_energy", "particle_kinetic_energy_J"),
    ("radiation_energy", "radiation_energy_J"),
)

ACCEPTED_LOAD_POWER_SOURCES = (
    "named_poynting_surface_flux",
    "reviewed_volume_j_dot_e_integral",
)

STAGE0_PACKET_IDS = (
    "power_port_source_review",
    "power_port_domain_review",
    "power_port_sign_review",
    "power_port_time_centering_review",
    "power_port_energy_ledger_review",
    "negative_test_plan",
)


def build_engineering_power_port_packet(
    circuit: Mapping[str, Any] | None,
    *,
    startup: Mapping[str, Any] | None = None,
    conservation: Mapping[str, Any] | None = None,
    simulation_telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting power-port packet for the current runner state."""
    circuit_record = _last_circuit_record(circuit)
    circuit_step = _circuit_step_from_record(circuit_record)
    udpf_source = _optional_str(circuit_record, "udpf_source")
    low_current_feedback = _mapping_or_none(
        None if circuit_record is None else circuit_record.get("low_current_feedback")
    )
    active_load_relation = _active_load_relation(circuit_step, udpf_source)
    current_A = _optional_float(circuit_step, "current_A")
    terminal_voltage_V = _optional_float(circuit_step, "udpf_V")
    final_energy = _energy_section(conservation, "final")
    field_work = _last_field_work(simulation_telemetry)
    j_dot_e_power_W = _optional_float(field_work, "j_dot_e_power_W")
    magnetic_energy_J = _optional_float(final_energy, "magnetic_energy_J")
    diagnostic_field_inductance_H = _diagnostic_field_inductance_H(
        current_A=current_A,
        magnetic_energy_J=magnetic_energy_J,
    )
    active_power_W = (
        None
        if current_A is None or terminal_voltage_V is None
        else current_A * terminal_voltage_V
    )
    residual_budget = _candidate_power_residual_budget(
        conservation=conservation,
        simulation_telemetry=simulation_telemetry,
        field_work=field_work,
        active_power_W=active_power_W,
    )
    candidate_runtime_channels = _candidate_runtime_channels(
        circuit_step=circuit_step,
        final_energy=final_energy,
        field_work=field_work,
        diagnostic_field_inductance_H=diagnostic_field_inductance_H,
        residual_budget=residual_budget,
    )
    stage0_packets = _stage0_packet_scaffolds(
        active_load_relation=active_load_relation,
        current_A=current_A,
        terminal_voltage_V=terminal_voltage_V,
        final_energy=final_energy,
        field_work=field_work,
        residual_budget=residual_budget,
        low_current_feedback=low_current_feedback,
        startup=startup,
    )
    candidate_energy_ledger = _candidate_stage0_energy_ledger(
        final_energy=final_energy,
        conservation=conservation,
        residual_budget=residual_budget,
    )
    wp_n1_ledger_telemetry = _power_port_ledger_telemetry(simulation_telemetry)
    wp_n1_power_port_ledger = build_wp_n1_auluck_power_port_ledger(
        wp_n1_ledger_telemetry
    )
    wp_n1_negative_test_policy = build_wp_n1_negative_test_policy(
        wp_n1_ledger_telemetry
    )
    missing = list(ACCEPTANCE_BLOCKING_CHANNELS)
    if circuit_step is None:
        missing.extend(("terminal_current", "terminal_voltage", "active_load_relation"))
    if startup and startup.get("whole_shot_startup_blocked") is True:
        missing.append("startup_handoff_interval")

    return {
        "status": "candidate_engineering_power_port_not_validation",
        "authority_contract": "field_power_required",
        "active_load_relation": active_load_relation,
        "accepted_load_power_source": "none",
        "diagnostic_only_relations": [
            "L_field = 2 E_B / I^2",
            "tracked_total_energy_delta",
        ],
        "diagnostic_field_inductance_H": diagnostic_field_inductance_H,
        "magnetic_energy_inductance_authority": "diagnostic_only_not_circuit_load",
        "terminal_current_A": current_A,
        "terminal_voltage_V": terminal_voltage_V,
        "active_power_W": active_power_W,
        # S8-WS6: explicit demotion of every active-load placeholder to
        # engineering-only telemetry. None of these may satisfy accepted power
        # coupling: the only accepted load-power sources are a named Poynting
        # surface flux or a reviewed volume J.E integral (ACCEPTED_LOAD_POWER_
        # SOURCES). Each placeholder carries the WS1 channel-state contract
        # string `excluded_not_validated` -- it is engineering evidence only.
        "active_load_placeholder_demotion": _active_load_placeholder_demotion(
            active_power_W=active_power_W,
            diagnostic_field_inductance_H=diagnostic_field_inductance_H,
            active_load_relation=active_load_relation,
        ),
        "power_port_step_records": _power_port_step_records(
            circuit_step=circuit_step,
            udpf_source=udpf_source,
            current_A=current_A,
            terminal_voltage_V=terminal_voltage_V,
            active_power_W=active_power_W,
            final_energy=final_energy,
            field_work=field_work,
            conservation=conservation,
            diagnostic_field_inductance_H=diagnostic_field_inductance_H,
            residual_budget=residual_budget,
            low_current_feedback=low_current_feedback,
        ),
        "interface_surface_or_volume_domain": "not_declared",
        "poynting_power_W": None,
        "j_dot_e_power_W": j_dot_e_power_W,
        "j_dot_e_domain": (
            None if field_work is None else field_work.get("domain")
        ),
        "time_centering": "candidate_runner_step_metadata_only",
        "sign_convention": "not_accepted",
        "required_channels": list(REQUIRED_POWER_PORT_CHANNELS),
        "acceptance_blocking_channels": list(ACCEPTANCE_BLOCKING_CHANNELS),
        "missing_acceptance_channels": sorted(set(missing)),
        "power_port_channel_status": _power_port_channel_statuses(
            circuit_step=circuit_step,
            final_energy=final_energy,
            field_work=field_work,
            startup=startup,
        ),
        "energy_ledger_status": _energy_ledger_status(final_energy),
        "candidate_stage0_energy_ledger": candidate_energy_ledger,
        "wp_n1_auluck_power_port_ledger": wp_n1_power_port_ledger,
        "wp_n1_negative_test_policy": wp_n1_negative_test_policy,
        "stage0_packet_scaffolds": stage0_packets,
        "stage0_packet_ids": list(STAGE0_PACKET_IDS),
        "power_port_operator_comparison": _operator_comparison_packet(),
        "sigma_quasi_tem_line_voltage_operator": (
            _sigma_quasi_tem_line_voltage_packet()
        ),
        "low_current_p_over_i_singularity": (
            dict(low_current_feedback)
            if low_current_feedback is not None
            else _missing_low_current_feedback_packet()
        ),
        "candidate_power_residual_budget": residual_budget,
        "candidate_runtime_channels": candidate_runtime_channels,
        "active_load_decision": {
            "active_load_relation": (
                active_load_relation
            ),
            "accepted_load_power_source": "none",
            "required_accepted_load_power_sources": list(ACCEPTED_LOAD_POWER_SOURCES),
            "diagnostic_relations_do_not_define_load": True,
            "candidate_volume_j_dot_e_is_not_active_load": (
                j_dot_e_power_W is not None
                and not _uses_candidate_j_dot_e_active_load(active_load_relation)
            ),
            "candidate_lagged_volume_j_dot_e_is_active_load": (
                _uses_candidate_j_dot_e_active_load(active_load_relation)
            ),
            "decision": (
                "candidate_lagged_field_power_load_not_accepted"
                if _uses_candidate_j_dot_e_active_load(active_load_relation)
                else "input_voltage_sequence_not_accepted_load_authority"
            ),
            "can_support_power_port_acceptance": False,
        },
        "acceptance_gate": (
            "terminal_current_voltage_and_energy_ledger_candidates_cannot_support_"
            "power_authority_until_named_poynting_or_j_dot_e_integral_sign_"
            "centering_auluck_eq6_term_completeness_term_i_through_term_vi_"
            "independently_computed_residual_tolerance_hashes_and_review_pass"
        ),
        "negative_test_policy": {
            "sign_convention_reversal_required": True,
            "time_centering_mismatch_required": True,
            "poynting_j_dot_e_non_equivalence_required": True,
            # Auluck has no electrode-contact-work term; the negative test
            # omits one of the genuine eq. (6) terms (term_i..term_vi) and
            # requires the residual to refuse to close by construction.
            "auluck_eq6_term_omission_required": True,
            "residual_tolerance_failure_required": True,
            "diagnostic_inductance_as_load_rejection_required": True,
            "low_current_p_over_i_singularity_rejection_required": True,
            "hidden_current_floor_or_back_emf_clip_rejection_required": True,
            "startup_handoff_gap_rejection_required": True,
            "sigma_line_voltage_as_driver_rejection_required": True,
        },
        "residual_policy": {
            "accepted_residual_tolerance": "not_attached",
            "tracked_energy_delta_is_residual": False,
            "candidate_power_residual_budget_available": (
                residual_budget["available"]
            ),
            "candidate_power_residual_budget_status": residual_budget["status"],
            "candidate_residual_channels": (
                ["tracked_total_energy_delta"]
                if conservation is not None
                and conservation.get("delta_tracked_total_energy_J") is not None
                else []
            ),
            "candidate_field_power_channels": (
                ["volume_j_dot_e_power"]
                if j_dot_e_power_W is not None
                else []
            ),
            "candidate_terminal_power_channels": (
                ["terminal_current_times_udpf_integral"]
                if residual_budget.get("cumulative_terminal_active_port_work_J")
                is not None
                else []
            ),
        },
        "source_references": list(POWER_PORT_SOURCE_REFS),
        "conservation_status": (
            None if conservation is None else conservation.get("status")
        ),
        "startup_handoff_required": (
            bool(startup.get("whole_shot_startup_blocked")) if startup else None
        ),
        "can_support_first_principles_acceptance": False,
    }


def _power_port_ledger_telemetry(
    simulation_telemetry: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return the simulator-emitted WP-N1B six-term ledger, if present."""
    if simulation_telemetry is None:
        return None
    ledger = simulation_telemetry.get("power_port_ledger")
    return ledger if isinstance(ledger, Mapping) else None


def _residual_fraction(
    residual_J: float | None,
    iv_work_J: float | None,
    term_i_J: float | None,
) -> float | None:
    """Return residual / max(|I*V work|, |term I|, 1 J).

    Denominator pattern of WP-N1 source packet S4: the dimensionless residual
    fraction is normalised against the largest of the two dominant power
    channels (the terminal I*V work and the stored-magnetic term I) so a
    near-zero numerator does not produce a spuriously large fraction.
    """
    if residual_J is None:
        return None
    denom = _residual_denominator(iv_work_J, term_i_J)
    if denom is None or denom == 0.0:
        return None
    return float(residual_J) / float(denom)


# WP-N1B: Auluck 2021 eq. (6) six-term power-balance ledger keys.
# Term labels I-VI are Auluck's own.
# [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (6), p.8]
_WP_N1B_LEDGER_KEYS = (
    "term_i_stored_magnetic_energy_rate_J",
    "term_ii_motional_magnetic_sigma_p_J",
    "term_iii_stored_electric_energy_rate_J",
    "term_iv_motional_electric_sigma_p_J",
    "term_v_resistive_sigma_p_J",
    "term_vi_anomalous_poloidal_sigma_p_J",
)

# Backward-compatibility alias. The five-term WP-N1 ledger keys (which
# included the non-Auluck "electrode_interface_work_J" closure residual) are
# superseded by the WP-N1B six-term decomposition. Retained only so importers
# of the old name resolve; the old closure-residual term is intentionally
# absent.
_WP_N1_LEDGER_KEYS = _WP_N1B_LEDGER_KEYS

# WP-N1B: the verified Auluck eq. (1) sign convention. The leading MINUS is
# load-bearing (Auluck p.6; verified extract "Sign convention" section). Any
# ledger that does not record this string fails closed.
# [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (1), p.6]
_AULUCK_EQ1_SIGN_CONVENTION = "V_12 = -(1/I) integral_Omega d3r (J.E)"

# WP-N1B: runtime-support blockers.
#
# Sigma_p (terms II/IV/V/VI): WP-N3 S3.3 introduces the `SigmaPSurfacePacket`
# data contract (`dpf.fields.source_geometry`). Terms II/IV/V/VI now consume
# that packet via `_sigma_p_surface_term`: a missing Sigma_p face set, a
# missing material velocity v, or a missing resistivity eta each fails the
# corresponding term(s) closed with a typed, operand-naming blocker. S3.3 is
# plumbing only -- it does NOT compute the surface integrals (Sprint 4 work),
# so even a fully-operand Sigma_p packet leaves the terms blocked. The runtime
# still exposes no reviewed moving-boundary face set, so the default packet is
# `SigmaPSurfacePacket.blocked()` and `_SIGMA_P_BLOCKER` remains the
# missing-face-set reason (now sourced from `SIGMA_P_BLOCKERS["sigma_p"]`).
#
# Stored-EM split (terms I/III): the runtime DOES now emit the magnetic-only
# and electric-only stored energy over Omega
# (`stored_magnetic_energy_delta_J`, `stored_electric_energy_delta_J`;
# Sprint 2.3 step 1, `omega_stored_em_energy_split_J`). Terms I and III are
# therefore computed INDEPENDENTLY from that split telemetry. The
# `_STORED_SPLIT_BLOCKER` string is retained only as the fail-closed reason
# for the case where the split telemetry is ABSENT (an older ledger that
# carries only the combined `stored_em_energy_delta_J`).
_SIGMA_P_BLOCKER = SIGMA_P_BLOCKERS["sigma_p"]
_STORED_SPLIT_BLOCKER = (
    "stored_em_energy_magnetic_electric_split_not_exposed_by_runtime"
)

# S8-WS6: the Auluck eq. (6) six-term roster, in eq. (6) order, with the
# one-line Auluck identification of each term (verified extract eq. (6) p.8;
# term identification prose p.9-10). The ledger emits this roster so a
# reviewer can confirm every one of the six terms is independently accounted
# for -- a five-term or closure-padded balance is not acceptable.
# [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (6) p.8, term ID p.9-10]
_AULUCK_EQ6_TERM_ROSTER = (
    (
        "term_i_stored_magnetic_energy_rate_J",
        "I",
        "d/dt integral_Omega d3r (1/2 mu0^-1 B^2)",
        "stored magnetic energy rate (Auluck term I)",
    ),
    (
        "term_ii_motional_magnetic_sigma_p_J",
        "II",
        "integral_Sigma_p dS.v (1/2 mu0^-1 B^2)",
        "motional magnetic Sigma_p surface integral (Auluck term II)",
    ),
    (
        "term_iii_stored_electric_energy_rate_J",
        "III",
        "d/dt integral_Omega d3r (1/2 eps0 E^2)",
        "stored electric energy rate (Auluck term III)",
    ),
    (
        "term_iv_motional_electric_sigma_p_J",
        "IV",
        "- integral_Sigma_p dS.v (1/2 eps0 E^2)",
        "motional electric Sigma_p surface integral (Auluck term IV)",
    ),
    (
        "term_v_resistive_sigma_p_J",
        "V",
        "mu0^-1 oint_Sigma_p dS.(eta J x B)",
        "resistive Sigma_p surface integral (Auluck term V)",
    ),
    (
        "term_vi_anomalous_poloidal_sigma_p_J",
        "VI",
        "- mu0^-1 oint_Sigma_p dS.B (B.v)",
        "anomalous/poloidal Sigma_p surface integral (Auluck term VI)",
    ),
)

# S8-WS6: Auluck (verified extract, "What this source ... DOES NOT provide")
# supplies NO numerical residual / energy-balance tolerance and NO
# time-centering / quadrature prescription. The residual tolerance therefore
# fails closed until a source packet attaches one; this string is the
# fail-closed reason.
_RESIDUAL_TOLERANCE_NOT_SOURCE_BACKED = (
    "residual_tolerance_not_source_backed_auluck_supplies_no_balance_tolerance"
)


def _term_blocked_packet(reason: str, *, source_ref: str) -> dict[str, Any]:
    """Return a fail-closed packet for one Auluck eq. (6) term."""
    return {
        "value_J": None,
        "status": "blocked",
        "blocker": reason,
        "computed_independently": False,
        "source_ref": source_ref,
    }


def _term_computed_packet(
    value_J: float,
    *,
    source_ref: str,
    integrand: str,
    runtime_telemetry_key: str,
) -> dict[str, Any]:
    """Return a packet for one INDEPENDENTLY-computed Auluck eq. (6) term.

    ``value_J`` is computed directly from a runtime field telemetry channel
    (named by ``runtime_telemetry_key``), NOT derived by closure from the
    terminal I*V work minus the other terms. ``integrand`` records the Auluck
    eq. (6) integrand this term evaluates.
    """
    return {
        "value_J": float(value_J),
        "status": "computed_independently",
        "blocker": None,
        "computed_independently": True,
        "source_ref": source_ref,
        "integrand": integrand,
        "runtime_telemetry_key": runtime_telemetry_key,
        "derived_by_closure": False,
    }


# WP-N3 S3.3: the integration of the Sigma_p surface integrals is Sprint 4
# work. When every operand a term needs IS available, S3.3 plumbing still does
# NOT compute the integral -- the term stays blocked with this reason, which
# is distinct from a missing-operand blocker so a reader can tell "ready for
# Sprint 4" apart from "operand absent".
# [WP_N3_SIGMA_P_RUNTIME_INTERFACE_SPEC.md section 0: "downstream computation
#  ... is Sprint 4 work"]
_SIGMA_P_INTEGRAL_IS_SPRINT4 = (
    "sigma_p_surface_integral_is_sprint4_work_s3_3_is_plumbing_only"
)


_SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED = (
    "serialized_sigma_p_packet_not_supported"
)

# S3R.5 (A7): fields required to reconstruct a SigmaPSurfacePacket from dict.
_SIGMA_P_PACKET_REQUIRED_DICT_KEYS = (
    "status",
    "source_refs",
    "source_geometry_packet_id",
    "source_geometry_hash",
    "n_sigma_p_faces",
    "face_count_total_sigma",
    "geometry_review_status",
    "face_ids",
    "dS_outward_m2",
    "face_area_m2",
    "outward_normal",
    "face_material_class",
    "is_moving",
    "omega_side",
    "excluded_interface_side",
    "outward_normal_convention",
    "field_sampler_status",
    "velocity_status",
    "resistivity_status",
    "centering",
    "quadrature",
    "sign_convention",
    "operand_blockers",
)

def _sigma_p_packet_from_dict(d: Mapping[str, Any]) -> SigmaPSurfacePacket:
    """Reconstruct a SigmaPSurfacePacket from its to_dict() form.

    S3R.5 (A7): dict-form packets must NOT be silently discarded. If any
    required field is absent the function raises KeyError so the caller can
    emit a named `serialized_sigma_p_packet_not_supported` blocker.
    """
    def _arr(v: Any) -> _np.ndarray | None:
        return None if v is None else _np.asarray(v)

    return SigmaPSurfacePacket(
        status=d["status"],
        source_refs=tuple(d["source_refs"]),
        source_geometry_packet_id=d["source_geometry_packet_id"],
        source_geometry_hash=d["source_geometry_hash"],
        n_sigma_p_faces=int(d["n_sigma_p_faces"]),
        face_count_total_sigma=int(d["face_count_total_sigma"]),
        geometry_review_status=d["geometry_review_status"],
        face_ids=_arr(d["face_ids"]),
        dS_outward_m2=_arr(d["dS_outward_m2"]),
        face_area_m2=_arr(d["face_area_m2"]),
        outward_normal=_arr(d["outward_normal"]),
        face_material_class=tuple(d["face_material_class"]),
        is_moving=_arr(d["is_moving"]),
        omega_side=d["omega_side"],
        excluded_interface_side=d["excluded_interface_side"],
        outward_normal_convention=d["outward_normal_convention"],
        field_sampler_status=dict(d["field_sampler_status"]),
        velocity_status=d["velocity_status"],
        resistivity_status=d["resistivity_status"],
        centering=dict(d["centering"]),
        quadrature=d["quadrature"],
        sign_convention=(
            None if d["sign_convention"] is None else dict(d["sign_convention"])
        ),
        operand_blockers=dict(d["operand_blockers"]),
        sigma_p_face_set_sha256=str(d.get("sigma_p_face_set_sha256", "")),
        moving_classification_sha256=str(d.get("moving_classification_sha256", "")),
        omega_partition_sha256=str(d.get("omega_partition_sha256", "")),
        material_mask_sha256_by_class=dict(
            d.get("material_mask_sha256_by_class") or {}
        ),
        moving_classification_status=str(
            d.get("moving_classification_status", "not_classified")
        ),
    )


def _sigma_p_packet_from_ledger(
    ledger: Mapping[str, Any] | None,
) -> SigmaPSurfacePacket:
    """Return the Sigma_p surface packet a ledger carries, or a blocked one.

    S3.3 plumbing. If the runtime emitted a `sigma_p_surface_packet` (a
    `SigmaPSurfacePacket` instance or its `to_dict()` form) it is used;
    otherwise a fully fail-closed packet is built. `power_port.py` never
    fabricates a Sigma_p face set.

    S3R.5 (A7): dict-form packets are reconstructed via `_sigma_p_packet_from_dict`.
    If reconstruction fails the function emits a named
    `serialized_sigma_p_packet_not_supported` blocked packet -- it does NOT
    silently discard the dict-form input and proceed as if nothing arrived.
    """
    candidate = None if ledger is None else ledger.get("sigma_p_surface_packet")
    if isinstance(candidate, SigmaPSurfacePacket):
        return candidate
    if isinstance(candidate, Mapping):
        # Attempt dict-form reconstruction. Any missing required field or
        # validation failure -> named blocker, not silent discard.
        try:
            # Verify all required keys are present before constructing.
            missing = [k for k in _SIGMA_P_PACKET_REQUIRED_DICT_KEYS if k not in candidate]
            if missing:
                raise KeyError(f"dict-form packet missing keys: {missing}")
            return _sigma_p_packet_from_dict(candidate)
        except Exception as exc:
            blocked = build_sigma_p_surface_packet(None)
            return _dc_replace(
                blocked,
                status=_SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED,
                operand_blockers={
                    **blocked.operand_blockers,
                    "serialization": str(exc),
                },
            )
    # No reviewed Sigma_p geometry from the runtime: build the blocked default.
    return build_sigma_p_surface_packet(None)


_SIGMA_P_SIGN_CONVENTION_BLOCKER = (
    "sigma_p_eq6_sign_convention_not_recorded"
)
_SIGMA_P_MOVING_CLASSIFICATION_BLOCKER = (
    "sigma_p_moving_stationary_classification_not_recorded"
)


def _sigma_p_surface_term(
    term_key: str,
    *,
    sigma_p: SigmaPSurfacePacket,
    source_ref: str,
    integrand: str,
) -> dict[str, Any]:
    """Return the fail-closed packet for one Auluck eq. (6) Sigma_p term.

    Consumes the `SigmaPSurfacePacket` ONLY (never the terminal I*V work).
    Walks the term's operand list (`SIGMA_P_TERM_OPERANDS`); the first absent
    operand sets a typed blocker that NAMES the missing operand:

      * missing Sigma_p face set -> blocks II, IV, V, VI;
      * missing v -> blocks II, IV, VI;
      * missing eta -> blocks V.

    S3R.5 additional controls (A7) — checked after all field operands pass:
      * missing sign convention -> blocks all Sigma_p terms that otherwise have
        all operands present (recorded after face-set/v/eta check)
      * absent moving/stationary classification -> blocks all Sigma_p terms
        (Auluck p.8: "stationary boundaries do not contribute"; without
        classification the integral cannot be formed correctly)

    S3.3 is plumbing only. Even with every operand present the term is NOT
    computed -- the surface integral is Sprint 4 work, so the term stays
    blocked with `_SIGMA_P_INTEGRAL_IS_SPRINT4`. No term is ever derived as
    `I*V` minus the others.
    """
    # Field operand checks (sigma_p face set, v, eta, B, E, J) -- these take
    # priority so the blocker names the most upstream missing thing.
    for operand in SIGMA_P_TERM_OPERANDS[term_key]:
        if not sigma_p.operand_status(operand):
            reason = sigma_p.operand_blockers.get(
                operand, SIGMA_P_BLOCKERS.get(operand, f"{operand}_not_available")
            )
            packet = _term_blocked_packet(reason, source_ref=source_ref)
            packet["integrand"] = integrand
            packet["missing_operand"] = operand
            packet["consumes"] = "sigma_p_surface_packet"
            return packet

    # S3R.5 (A7) sign-convention negative control: a sign-unverified surface
    # integral cannot be a genuine power-balance contribution. Checked after
    # field operands so the blocker distinguishes "face set absent" from "sign
    # convention absent when face set is present".
    if not sigma_p.has_sign_convention():
        reason = sigma_p.operand_blockers.get(
            "sign_convention",
            SIGMA_P_BLOCKERS.get("sign_convention", _SIGMA_P_SIGN_CONVENTION_BLOCKER),
        )
        packet = _term_blocked_packet(reason, source_ref=source_ref)
        packet["integrand"] = integrand
        packet["missing_operand"] = "sign_convention"
        packet["consumes"] = "sigma_p_surface_packet"
        return packet

    # S3R.5 (A7) moving/stationary classification negative control.  Checked
    # after sign convention: Auluck p.8 requires stationary boundaries to
    # contribute zero; without classification the distinction cannot be enforced.
    if sigma_p.moving_classification_status not in ("available",):
        packet = _term_blocked_packet(
            _SIGMA_P_MOVING_CLASSIFICATION_BLOCKER, source_ref=source_ref
        )
        packet["integrand"] = integrand
        packet["missing_operand"] = "moving_classification"
        packet["consumes"] = "sigma_p_surface_packet"
        packet["moving_classification_status"] = sigma_p.moving_classification_status
        return packet

    # all operands + controls satisfied: S3.3 plumbing stops here -- integral is S4.
    packet = _term_blocked_packet(
        _SIGMA_P_INTEGRAL_IS_SPRINT4, source_ref=source_ref
    )
    packet["integrand"] = integrand
    packet["missing_operand"] = None
    packet["consumes"] = "sigma_p_surface_packet"
    packet["operands_available"] = True
    return packet


def _ws6_six_term_presence(
    term_packets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the S8-WS6 explicit six-term presence roster.

    WS6 requires each of the six Auluck eq. (6) terms to be *independently
    present, or fail closed*. This walks `_AULUCK_EQ6_TERM_ROSTER` (eq. (6)
    order) and reports, per term, whether it is `present` (independently
    computed from a runtime field channel) or `blocked_fail_closed` (with the
    naming blocker). The ledger embeds this so a reviewer never has to infer
    completeness from scattered keys, and so a five-term balance is visibly
    incomplete. [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (6) p.8]
    """
    roster: dict[str, Any] = {}
    present_count = 0
    for key, label, integrand, name in _AULUCK_EQ6_TERM_ROSTER:
        packet = term_packets.get(key, {})
        independent = bool(packet.get("computed_independently"))
        if independent:
            present_count += 1
        roster[key] = {
            "auluck_term_label": label,
            "auluck_eq6_integrand": integrand,
            "name": name,
            "state": "present" if independent else "blocked_fail_closed",
            "value_J": packet.get("value_J"),
            "blocker": packet.get("blocker"),
        }
    return {
        "expected_term_count": len(_AULUCK_EQ6_TERM_ROSTER),
        "independent_term_count": present_count,
        "all_six_terms_present": present_count == len(_AULUCK_EQ6_TERM_ROSTER),
        "terms": roster,
        "fail_closed_rule": (
            "each of the six Auluck eq. (6) terms must be independently "
            "present; any term blocked fails the six-term balance closed"
        ),
    }


def _ws6_domain_field(
    auluck_omega_domain: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the S8-WS6 explicit `domain` field for the ledger structure.

    WS6 mandates an explicit `domain` field. Auluck's balance is a volume
    integral over the toroidal domain Omega (J = 0 outside it) bounded by the
    closed surface Sigma, whose *moving* part is Sigma_p; the
    electrode/power-source interface is EXCLUDED from Omega and its Poynting
    flux IS the LHS I(t)V(t). This wraps the existing Omega-domain packet and
    surfaces those facts as named fields. [verified extract Domain p.6-7]
    """
    return {
        "integration_domain": "Omega",
        "domain_topology": "toroid_not_simply_connected",
        "domain_definition": "current-carrying volume; J = 0 outside Omega",
        "bounding_surface": "Sigma",
        "moving_boundary": "Sigma_p",
        "excluded_from_domain": (
            "electrode/power-source interface; its Poynting flux IS the "
            "LHS I(t)V(t) power input"
        ),
        "domain_partition_status": auluck_omega_domain.get("status"),
        "domain_partition_valid": auluck_omega_domain.get("partition_valid"),
        "auluck_omega_domain": auluck_omega_domain,
        "source_ref": (
            "AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md Domain p.6-7"
        ),
        "can_support_power_port_acceptance": False,
    }


def _ws6_residual_field(
    *,
    residual_J: float | None,
    residual_fraction: float | None,
    residual_is_genuine: bool,
    iv_work_J: float | None,
) -> dict[str, Any]:
    """Return the S8-WS6 explicit `residual` field for the ledger structure.

    WS6 mandates an explicit `residual` field. The residual is
    `I*V - (I+II+III+IV+V+VI)`, a GENUINE (non-closure) diagnostic that is
    `None` while any term fails closed. Auluck supplies no balance tolerance
    (verified extract "DOES NOT provide"), so `accepted_residual_tolerance`
    fails closed and the residual cannot gate acceptance.
    [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (6) p.8]
    """
    return {
        "definition": (
            "I*V - (term_i + term_ii + term_iii + term_iv + term_v + term_vi)"
        ),
        "residual_J": residual_J,
        "residual_fraction": residual_fraction,
        "iv_work_lhs_J": iv_work_J,
        "is_genuine_diagnostic": residual_is_genuine,
        "is_closure_by_construction": False,
        "computable_only_when": "all six eq. (6) terms independently present",
        "accepted_residual_tolerance": "not_attached",
        "residual_tolerance_blocker": _RESIDUAL_TOLERANCE_NOT_SOURCE_BACKED,
        "tracked_energy_delta_is_residual": False,
        "can_support_power_port_acceptance": False,
    }


def build_wp_n1_auluck_power_port_ledger(
    ledger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the WP-N1B Auluck eq. (6) six-term power-balance ledger.

    Implements the WP-N1B power-port acceptance proposal. Auluck eq. (6) is

        I(t)V(t) = I + II + III + IV + V + VI

    where (verified extract eq. (6), p.8; term labels are Auluck's own):

      I   = d/dt integral_Omega d3r (1/2 mu0^-1 B^2)   stored magnetic rate
      II  = integral_Sigma_p dS.v (1/2 mu0^-1 B^2)     motional magnetic
      III = d/dt integral_Omega d3r (1/2 eps0 E^2)     stored electric rate
      IV  = - integral_Sigma_p dS.v (1/2 eps0 E^2)     motional electric
      V   = mu0^-1 oint_Sigma_p dS.(eta J x B)         resistive
      VI  = - mu0^-1 oint_Sigma_p dS.B (B.v)           anomalous/poloidal

    Each term is computed INDEPENDENTLY from runtime fields. There is NO
    closure-by-construction residual: the residual
    I*V - (I+II+III+IV+V+VI) is a genuine, non-trivial diagnostic.

    FAIL-CLOSED behaviour (WP-N1B proposal section 5/6, audit F1):
      * Terms II, IV, V, VI are surface integrals over the MOVING boundary
        Sigma_p. The runtime exposes no reviewed Sigma_p moving-boundary face
        set, and no v / eta on Sigma_p faces. These four terms fail closed
        with blocker `_SIGMA_P_BLOCKER` and value None.
      * Terms I and III need the magnetic / electric stored-energy SPLIT.
        Current runtime ledgers emit that split and compute I/III
        independently when the corresponding split channel is present. Older
        ledgers that expose no split channel, including combined-only stored
        EM ledgers, fail closed with blocker `_STORED_SPLIT_BLOCKER`.
      * If the runtime emitted no `power_port_ledger`, or did not record the
        eq. (1) sign convention, the whole ledger fails closed.

    A six-term structure with Sigma_p-dependent terms failing closed pending
    WP-N3 geometry is the honest, acceptable outcome. No invented values.
    """
    eq6_ref = "AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (6) p.8"
    eq1_ref = "AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (1) p.6"

    if ledger is None:
        return {
            "status": "blocked_wp_n1b_power_port_ledger_not_available",
            "reason": (
                "no circuit-coupled run; simulator emitted no power_port_ledger"
            ),
            "auluck_eq6_decomposition": True,
            "auluck_omega_domain": _omega_domain_unavailable_packet(),
            "auluck_eq1_sign_convention": _AULUCK_EQ1_SIGN_CONVENTION,
            "auluck_eq1_sign_convention_recorded": True,
            "energy_ledger_terms_J": {key: None for key in _WP_N1B_LEDGER_KEYS},
            # S8-WS6: all six terms fail closed when no ledger telemetry exists.
            "auluck_eq6_six_term_presence": _ws6_six_term_presence(
                {
                    key: _term_blocked_packet(
                        "no_power_port_ledger_telemetry", source_ref=eq6_ref
                    )
                    for key in _WP_N1B_LEDGER_KEYS
                }
            ),
            "ledger_blocked": True,
            "ledger_blocker": "no_power_port_ledger_telemetry",
            "iv_work_J": None,
            "residual_J": None,
            "residual_fraction": None,
            "residual_is_genuine_diagnostic": False,
            "accepted_residual_tolerance": "not_attached",
            # S8-WS6: the four explicit ledger-structure fields, present even
            # when no telemetry exists so the structure shape is invariant.
            "power_port_ledger_fields": {
                "sign_convention": _wp_n1b_sign_convention_packet(True),
                "time_centering": _wp_n1_time_centering_packet({}),
                "domain": _ws6_domain_field(_omega_domain_unavailable_packet()),
                "residual": _ws6_residual_field(
                    residual_J=None,
                    residual_fraction=None,
                    residual_is_genuine=False,
                    iv_work_J=None,
                ),
            },
            "sign_convention": _wp_n1b_sign_convention_packet(True),
            "time_centering": _wp_n1_time_centering_packet({}),
            "domain": _ws6_domain_field(_omega_domain_unavailable_packet()),
            "residual": _ws6_residual_field(
                residual_J=None,
                residual_fraction=None,
                residual_is_genuine=False,
                iv_work_J=None,
            ),
            "source_refs": [eq6_ref, eq1_ref],
            "can_support_power_port_acceptance": False,
            "can_support_first_principles_acceptance": False,
        }

    # Eq. (1) sign-convention record. Auluck p.6: V_12 carries a leading
    # minus. The runtime must record a sign-convention key for the I*V port
    # work; if it does not, the ledger fails closed (a sign-unverified
    # ledger cannot be a genuine residual diagnostic).
    sign_convention_recorded = ledger.get("sign_convention") is not None

    # Terminal I*V cumulative work -- the LHS of eq. (6). This is an
    # independent runtime channel (terminal current x voltage, time
    # integrated); it is NOT one of the six RHS terms.
    iv_work_J = _optional_float(ledger, "cumulative_terminal_port_work_J")

    # --- Terms I and III: stored magnetic / electric energy rates ---------
    # Auluck eq. (6):
    #   I   = d/dt integral_Omega d3r (1/2 mu0^-1 B^2)   stored magnetic rate
    #   III = d/dt integral_Omega d3r (1/2 eps0 E^2)     stored electric rate
    # The ledger integrates eq. (6) over the run, so the time-integrated
    # term I equals [integral_Omega (1/2 mu0^-1 B^2)]_final minus _initial,
    # i.e. the magnetic-only stored-energy DELTA over Omega; term III is the
    # electric-only delta. The runtime now emits this split directly
    # (Sprint 2.3 step 1: omega_stored_em_energy_split_J ->
    # stored_magnetic_energy_delta_J / stored_electric_energy_delta_J). Each
    # delta is computed from the runtime B / E field over Omega and is
    # INDEPENDENT of the terminal I*V work -- it is NOT a closure-derived
    # quantity. If the split telemetry is absent (an older ledger carrying
    # only the combined stored_em_energy_delta_J), terms I and III fail
    # closed: the combined delta must NOT be substituted for either term.
    stored_magnetic_delta_J = _optional_float(
        ledger, "stored_magnetic_energy_delta_J"
    )
    stored_electric_delta_J = _optional_float(
        ledger, "stored_electric_energy_delta_J"
    )
    if stored_magnetic_delta_J is not None:
        term_i = _term_computed_packet(
            stored_magnetic_delta_J,
            source_ref=eq6_ref,
            integrand="d/dt integral_Omega d3r (1/2 mu0^-1 B^2)",
            runtime_telemetry_key="stored_magnetic_energy_delta_J",
        )
    else:
        term_i = _term_blocked_packet(_STORED_SPLIT_BLOCKER, source_ref=eq6_ref)
    if stored_electric_delta_J is not None:
        term_iii = _term_computed_packet(
            stored_electric_delta_J,
            source_ref=eq6_ref,
            integrand="d/dt integral_Omega d3r (1/2 eps0 E^2)",
            runtime_telemetry_key="stored_electric_energy_delta_J",
        )
    else:
        term_iii = _term_blocked_packet(
            _STORED_SPLIT_BLOCKER, source_ref=eq6_ref
        )

    # --- Terms II, IV, V, VI: Sigma_p moving-boundary surface integrals ---
    # Auluck eq. (6):
    #   II  = integral_Sigma_p dS.v (1/2 mu0^-1 B^2)
    #   IV  = - integral_Sigma_p dS.v (1/2 eps0 E^2)
    #   V   = mu0^-1 oint_Sigma_p dS.(eta J x B)
    #   VI  = - mu0^-1 oint_Sigma_p dS.B (B.v)
    # Sigma_p is the MOVING part of the domain boundary (Auluck p.8:
    # "stationary boundaries do not contribute"). WP-N3 S3.3 plumbing: these
    # four terms consume a `SigmaPSurfacePacket` ONLY -- never the terminal
    # I*V work. Each term fails closed naming its first absent operand
    # (missing Sigma_p -> II/IV/V/VI; missing v -> II/IV/VI; missing eta -> V).
    # S3.3 does not compute the integral; even with every operand present the
    # term stays blocked pending the Sprint 4 surface-integral computation.
    sigma_p = _sigma_p_packet_from_ledger(ledger)
    term_ii = _sigma_p_surface_term(
        "term_ii_motional_magnetic_sigma_p_J",
        sigma_p=sigma_p, source_ref=eq6_ref,
        integrand="integral_Sigma_p dS.v (1/2 mu0^-1 B^2)",
    )
    term_iv = _sigma_p_surface_term(
        "term_iv_motional_electric_sigma_p_J",
        sigma_p=sigma_p, source_ref=eq6_ref,
        integrand="- integral_Sigma_p dS.v (1/2 eps0 E^2)",
    )
    term_v = _sigma_p_surface_term(
        "term_v_resistive_sigma_p_J",
        sigma_p=sigma_p, source_ref=eq6_ref,
        integrand="mu0^-1 oint_Sigma_p dS.(eta J x B)",
    )
    term_vi = _sigma_p_surface_term(
        "term_vi_anomalous_poloidal_sigma_p_J",
        sigma_p=sigma_p, source_ref=eq6_ref,
        integrand="- mu0^-1 oint_Sigma_p dS.B (B.v)",
    )

    term_packets = {
        "term_i_stored_magnetic_energy_rate_J": term_i,
        "term_ii_motional_magnetic_sigma_p_J": term_ii,
        "term_iii_stored_electric_energy_rate_J": term_iii,
        "term_iv_motional_electric_sigma_p_J": term_iv,
        "term_v_resistive_sigma_p_J": term_v,
        "term_vi_anomalous_poloidal_sigma_p_J": term_vi,
    }
    energy_ledger_terms_J = {
        key: packet["value_J"] for key, packet in term_packets.items()
    }
    independent_terms = sorted(
        key for key, packet in term_packets.items()
        if packet["computed_independently"]
    )
    blocked_terms = {
        key: packet["blocker"]
        for key, packet in term_packets.items()
        if packet["status"] == "blocked"
    }

    # --- Genuine residual: I*V - (I+II+III+IV+V+VI) ----------------------
    # This is NOT a closure-by-construction residual. It is computed ONLY
    # when every one of the six terms is an independently-computed value AND
    # the eq. (1) sign convention is recorded. With any term failing closed
    # the residual is None -- the ledger does not paper over a blocked term
    # with a derived value. (WP-N1B proposal section 5: "it no longer closes
    # by construction".)
    all_terms_independent = all(
        packet["computed_independently"] for packet in term_packets.values()
    )
    residual_J: float | None = None
    residual_is_genuine = False
    if (
        all_terms_independent
        and sign_convention_recorded
        and iv_work_J is not None
        and all(value is not None for value in energy_ledger_terms_J.values())
    ):
        residual_J = float(iv_work_J) - sum(
            float(value) for value in energy_ledger_terms_J.values()
        )
        residual_is_genuine = True

    ledger_blocked = not residual_is_genuine
    if not sign_convention_recorded:
        ledger_blocker = "auluck_eq1_sign_convention_not_recorded"
    elif blocked_terms:
        ledger_blocker = "auluck_eq6_terms_fail_closed_pending_wp_n3_geometry"
    elif iv_work_J is None:
        ledger_blocker = "terminal_iv_work_not_available"
    else:
        ledger_blocker = None

    domain_partition = ledger.get("domain_partition")
    auluck_omega_domain = _build_auluck_omega_domain_packet(
        domain_partition if isinstance(domain_partition, Mapping) else None
    )

    return {
        "status": (
            "candidate_wp_n1b_auluck_six_term_power_port_ledger_not_validation"
        ),
        "auluck_eq6_decomposition": True,
        "source_basis": [
            "Auluck 2021 DPF circuit-element eq. (6) six-term power balance",
            "Auluck 2021 eq. (1) circuit-element relation (leading minus)",
            "Auluck 2021 eq. (5) Sigma_p moving-boundary integrands",
        ],
        "source_refs": [eq6_ref, eq1_ref],
        "auluck_omega_domain": auluck_omega_domain,
        # WP-N3 S3.3 Sigma_p surface-term data contract (plumbing only). The
        # terms II/IV/V/VI packets above are derived from this packet's
        # per-operand status; the packet itself never carries a term value.
        "sigma_p_surface_packet": sigma_p.to_dict(),
        "sigma_p_surface_packet_status": sigma_p.status,
        # Eq. (1) sign convention recorded verbatim, leading minus included.
        "auluck_eq1_sign_convention": _AULUCK_EQ1_SIGN_CONVENTION,
        "auluck_eq1_sign_convention_recorded": sign_convention_recorded,
        "auluck_eq1_leading_minus_is_load_bearing": True,
        "iv_work_J": iv_work_J,
        "iv_work_is_eq6_lhs_not_a_term": True,
        "energy_ledger_terms_J": energy_ledger_terms_J,
        "energy_ledger_term_packets": term_packets,
        "energy_ledger_term_status": {
            key: packet["status"] for key, packet in term_packets.items()
        },
        "independent_terms": independent_terms,
        "blocked_terms": blocked_terms,
        "all_six_terms_computed_independently": all_terms_independent,
        "residual_J": residual_J,
        "residual_fraction": _residual_fraction(
            residual_J, iv_work_J, energy_ledger_terms_J[
                "term_i_stored_magnetic_energy_rate_J"
            ]
        ),
        "residual_definition": (
            "I*V - (term_i + term_ii + term_iii + term_iv + term_v + term_vi)"
        ),
        "residual_is_genuine_diagnostic": residual_is_genuine,
        "residual_is_closure_by_construction": False,
        "residual_interpretation": (
            "genuine_balance_diagnostic_when_all_six_terms_independent; "
            "None_while_any_term_fails_closed_no_closure_substitution"
        ),
        "ledger_blocked": ledger_blocked,
        "ledger_blocker": ledger_blocker,
        "accepted_residual_tolerance": "not_attached",
        "tracked_energy_delta_is_residual": False,
        # S8-WS6: the six explicit Auluck eq. (6) terms, each independently
        # present or fail-closed (eq. (6) order, with Auluck term labels).
        "auluck_eq6_six_term_presence": _ws6_six_term_presence(term_packets),
        # S8-WS6: the four explicit ledger-structure fields mandated by WS6 --
        # sign_convention / time_centering / domain / residual -- grouped so a
        # reviewer reads one coherent block. The standalone `sign_convention`
        # and `time_centering` keys below are retained for back-compatibility.
        "power_port_ledger_fields": {
            "sign_convention": _wp_n1b_sign_convention_packet(
                sign_convention_recorded
            ),
            "time_centering": _wp_n1_time_centering_packet(ledger),
            "domain": _ws6_domain_field(auluck_omega_domain),
            "residual": _ws6_residual_field(
                residual_J=residual_J,
                residual_fraction=_residual_fraction(
                    residual_J, iv_work_J, energy_ledger_terms_J[
                        "term_i_stored_magnetic_energy_rate_J"
                    ]
                ),
                residual_is_genuine=residual_is_genuine,
                iv_work_J=iv_work_J,
            ),
        },
        "sign_convention": _wp_n1b_sign_convention_packet(
            sign_convention_recorded
        ),
        "time_centering": _wp_n1_time_centering_packet(ledger),
        "domain": _ws6_domain_field(auluck_omega_domain),
        "residual": _ws6_residual_field(
            residual_J=residual_J,
            residual_fraction=_residual_fraction(
                residual_J, iv_work_J, energy_ledger_terms_J[
                    "term_i_stored_magnetic_energy_rate_J"
                ]
            ),
            residual_is_genuine=residual_is_genuine,
            iv_work_J=iv_work_J,
        ),
        "first_step_fallback": bool(ledger.get("first_step_fallback")),
        "first_step_udpf_source": ledger.get("first_step_udpf_source"),
        "steps_accumulated": ledger.get("steps_accumulated"),
        "scientific_status": "engineering_candidate_not_validation",
        "can_support_power_port_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _omega_domain_unavailable_packet() -> dict[str, Any]:
    return {
        "status": "blocked_auluck_omega_domain_not_available",
        "labels": [
            "omega_volume_cells",
            "terminal_source_interface_faces",
            "wall_material_faces",
            "open_pml_faces",
        ],
        "source_refs": [
            "KnowledgeReference/auluck-2021-dpf-circuit-element.md:203-257"
        ],
        "can_support_power_port_acceptance": False,
    }


def _build_auluck_omega_domain_packet(
    domain_partition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the WP-N1 S1 named Auluck Omega domain packet."""
    if domain_partition is None:
        return _omega_domain_unavailable_packet()
    constraints = domain_partition.get("partition_constraints")
    constraints = constraints if isinstance(constraints, Mapping) else {}
    label_packets = {}
    for label in (
        "omega_volume_cells",
        "terminal_source_interface_faces",
        "wall_material_faces",
        "open_pml_faces",
    ):
        entry = domain_partition.get(label)
        if isinstance(entry, Mapping):
            label_packets[label] = {
                "mask_sha256": entry.get("mask_sha256"),
                "cell_count": entry.get("cell_count"),
                "bounds": entry.get("bounds"),
                "source_refs": entry.get("source_refs"),
            }
    disjoint = bool(constraints.get("mutually_disjoint"))
    exhaustive = bool(constraints.get("exhaustive"))
    interface_non_empty = bool(
        constraints.get("terminal_source_interface_non_empty")
    )
    interface_disjoint = bool(
        constraints.get("terminal_source_interface_disjoint_from_omega")
    )
    partition_valid = (
        disjoint and exhaustive and interface_non_empty and interface_disjoint
    )
    return {
        "status": "candidate_auluck_omega_domain_not_validation",
        "source_refs": [
            "KnowledgeReference/auluck-2021-dpf-circuit-element.md:203-257"
        ],
        "labels": label_packets,
        "partition_mutually_disjoint": disjoint,
        "partition_exhaustive": exhaustive,
        "terminal_source_interface_non_empty": interface_non_empty,
        "terminal_source_interface_disjoint_from_omega": interface_disjoint,
        "partition_valid": partition_valid,
        "source_interface_z_index": domain_partition.get(
            "source_interface_z_index"
        ),
        "geometry_review_status": domain_partition.get(
            "geometry_review_status", "geometry_candidate_not_reviewed"
        ),
        "can_support_power_port_acceptance": False,
    }


def _wp_n1b_sign_convention_packet(
    sign_convention_recorded: bool,
) -> dict[str, Any]:
    """Return the WP-N1B declared sign convention, recording Auluck eq. (1).

    The verified extract is explicit (Auluck p.6, "Sign convention"): eq. (1)
    carries a leading MINUS, ``V_12 = -(1/I) integral_Omega d3r (J.E)``, and
    that minus is load-bearing. This packet records that string verbatim so a
    downstream consumer (or test) can fail closed if it is ever dropped.
    [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (1) p.6]
    """
    return {
        "status": "candidate_wp_n1b_sign_convention_not_accepted",
        "basis": "Auluck 2021 eq. (1) circuit-element relation",
        "source_ref": (
            "AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md eq. (1) p.6"
        ),
        # Verbatim eq. (1), leading minus included. This is the load-bearing
        # convention; consistency: I.V_12 = -integral_Omega J.E.
        "auluck_eq1_relation": _AULUCK_EQ1_SIGN_CONVENTION,
        "auluck_eq1_leading_minus_is_load_bearing": True,
        "auluck_eq1_consistency_check": "I*V_12 = -integral_Omega (J.E)",
        "auluck_eq1_sign_convention_recorded": bool(sign_convention_recorded),
        # Eq. (6) LHS I(t)V(t) is the power INPUT to the device, crossing the
        # excluded source interface (Auluck p.8).
        "iv_lhs": "positive_means_power_input_to_device_across_source_interface",
        "surface_flux_terms": "outward_dS_per_auluck_eq6",
        "negative_local_j_dot_e_clipped": False,
        "can_support_power_port_acceptance": False,
    }


def _wp_n1_time_centering_packet(
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the WP-N1 S3.2 step-consistent time-centering packet."""
    provenance = ledger.get("snapshot_provenance")
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    return {
        "status": "candidate_step_consistent_not_accepted",
        "declared_centering": "step_consistent_trapezoidal",
        "snapshot_provenance": provenance,
        "all_terms_share_centering": bool(provenance),
        "accuracy_order_claim": "none_no_source_in_scope_g3",
        "source_refs": [
            "KnowledgeReference/auluck-2021-dpf-circuit-element.md:173-197",
            "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1880-1888",
        ],
        "can_support_power_port_acceptance": False,
    }


def build_wp_n1_negative_test_policy(
    ledger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the WP-N1B six-negative-test policy block.

    Enumerates the six negative tests for the Auluck eq. (6) six-term
    ledger. N3 no longer references an "electrode work" term -- Auluck's
    balance has no such term (verified extract: the electrode/source
    interface is EXCLUDED from Omega and its flux IS the LHS I(t)V(t)).
    The rewritten N3 omits one of the genuine eq. (6) terms II/IV/V/VI. The
    tests themselves live in tests/test_first_principles_power_port.py and
    assert against the emitted packet fields.
    """
    return {
        "status": "candidate_wp_n1b_negative_test_policy_not_validation",
        "required_negative_tests": {
            "N1_sign_reversal": {
                "corruption": "flip sign of the terminal I*V work",
                "asserts_on": "residual_J and residual_fraction",
                "expected": "residual is sign-sensitive when computable",
            },
            "N2_wrong_domain": {
                "corruption": (
                    "include source interface in Omega or shift Omega into "
                    "current-free cells"
                ),
                "asserts_on": (
                    "auluck_omega_domain.partition_valid and "
                    "terminal_source_interface_disjoint_from_omega"
                ),
                "expected": "partition_valid becomes False",
            },
            "N3_omitted_eq6_term": {
                "corruption": (
                    "omit one of the genuine Auluck eq. (6) terms II/IV/V/VI "
                    "(there is no electrode-work term in Auluck's balance)"
                ),
                "asserts_on": (
                    "energy_ledger_terms_J completeness and residual_J"
                ),
                "expected": (
                    "a six-term ledger missing any term II/IV/V/VI cannot "
                    "emit a genuine residual; residual_J is None"
                ),
            },
            "N4_low_current_p_over_i": {
                "corruption": "drive I -> 0 while volume J.E stays finite",
                "asserts_on": "low_current_p_over_i_singularity and udpf_source",
                "expected": (
                    "low-current guard fires; U_DPF not computed as P/I; "
                    "no inf/NaN"
                ),
            },
            "N5_first_step_fallback": {
                "corruption": "run step 0 with no lagged field work",
                "asserts_on": "first_step_fallback and first_step_udpf_source",
                "expected": (
                    "step 0 marked fallback; no closed first-step residual "
                    "claimed"
                ),
            },
            "N6_default_mode_leakage": {
                "corruption": (
                    "run default input_sequence mode and read port as accepted"
                ),
                "asserts_on": (
                    "accepted_load_power_source, active_load_relation, "
                    "can_support_power_port_acceptance"
                ),
                "expected": (
                    "accepted_load_power_source stays 'none'; "
                    "can_support_power_port_acceptance stays False"
                ),
            },
        },
        "all_six_required": True,
        "auluck_eq6_term_count": 6,
        "no_electrode_work_term": True,
        "acceptance_unblocks_only_when": (
            "source_backed_residual_tolerance_attached AND "
            "wp_n3_reviewed_sigma_p_geometry AND all_six_tests_pass"
        ),
        "can_support_power_port_acceptance": False,
    }


def _last_circuit_record(
    circuit: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not circuit:
        return None
    last = circuit.get("last")
    return last if isinstance(last, Mapping) else None


def _circuit_step_from_record(
    last: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(last, Mapping):
        return None
    step = last.get("circuit_step")
    return step if isinstance(step, Mapping) else None


def _optional_float(mapping: Mapping[str, Any] | None, key: str) -> float | None:
    if mapping is None or mapping.get(key) is None:
        return None
    return float(mapping[key])


def _optional_str(mapping: Mapping[str, Any] | None, key: str) -> str | None:
    if mapping is None or mapping.get(key) is None:
        return None
    return str(mapping[key])


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _active_load_relation(
    circuit_step: Mapping[str, Any] | None,
    udpf_source: str | None,
) -> str:
    if circuit_step is None:
        return "no_active_circuit_boundary"
    if udpf_source == "candidate_lagged_volume_j_dot_e":
        return "lagged_volume_j_dot_e_voltage_not_accepted"
    if udpf_source == "candidate_lagged_auluck_volume_j_dot_e":
        return "lagged_auluck_volume_j_dot_e_voltage_not_accepted"
    return "input_terminal_voltage_sequence_not_active_load_authority"


def _uses_candidate_j_dot_e_active_load(active_load_relation: str) -> bool:
    return active_load_relation in {
        "lagged_volume_j_dot_e_voltage_not_accepted",
        "lagged_auluck_volume_j_dot_e_voltage_not_accepted",
    }


# S8-WS6: the WS1 channel-state contract string for a placeholder that is
# engineering telemetry only. It is never `accepted`; it cannot set
# accepted_runtime_claim or can_support_first_principles_acceptance.
_WS6_PLACEHOLDER_CHANNEL_STATE = "excluded_not_validated"


def _active_load_placeholder_demotion(
    *,
    active_power_W: float | None,
    diagnostic_field_inductance_H: float | None,
    active_load_relation: str,
) -> dict[str, Any]:
    """Return the S8-WS6 active-load placeholder demotion block.

    WS6: every active-load placeholder is demoted to engineering-only
    telemetry and CANNOT satisfy accepted power coupling. Accepted load power
    comes only from a named Poynting surface flux or a reviewed volume J.E
    integral (`ACCEPTED_LOAD_POWER_SOURCES`); the placeholders below are not
    in that set. Each carries the WS1 channel-state string
    `excluded_not_validated`. [verified extract: Auluck's accepted load-power
    relation is field-power over Omega, not a terminal I*V product]
    """
    placeholders = {
        "active_power_W_terminal_iv_product": {
            "value": active_power_W,
            "definition": "terminal_current_A * terminal_voltage_V",
            "channel_state": _WS6_PLACEHOLDER_CHANNEL_STATE,
            "why_not_accepted": (
                "input-voltage-sequence I*V product is not a source-derived "
                "load; Auluck's load power is field power over Omega"
            ),
        },
        "diagnostic_field_inductance_H": {
            "value": diagnostic_field_inductance_H,
            "definition": "L_field = 2 * magnetic_energy_J / I^2",
            "channel_state": _WS6_PLACEHOLDER_CHANNEL_STATE,
            "why_not_accepted": (
                "energy-derived inductance is a diagnostic, not a circuit "
                "load; eqs (13)-(14): dL_p/dt does not equal the motional "
                "Sigma_p terms"
            ),
        },
    }
    return {
        "status": "active_load_placeholders_demoted_engineering_only",
        "active_load_relation": active_load_relation,
        "accepted_load_power_source": "none",
        "required_accepted_load_power_sources": list(ACCEPTED_LOAD_POWER_SOURCES),
        "placeholders": placeholders,
        "placeholder_channel_state": _WS6_PLACEHOLDER_CHANNEL_STATE,
        "any_placeholder_satisfies_accepted_power_coupling": False,
        "satisfies_accepted_power_coupling": False,
        "can_support_power_port_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _energy_section(
    conservation: Mapping[str, Any] | None,
    section: str,
) -> Mapping[str, Any] | None:
    if conservation is None:
        return None
    value = conservation.get(section)
    return value if isinstance(value, Mapping) else None


def _last_field_work(
    simulation_telemetry: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if simulation_telemetry is None:
        return None
    last_step = simulation_telemetry.get("last_step")
    if not isinstance(last_step, Mapping):
        return None
    field_step = last_step.get("field_step")
    if not isinstance(field_step, Mapping):
        return None
    field_work = field_step.get("field_work")
    return field_work if isinstance(field_work, Mapping) else None


def _diagnostic_field_inductance_H(
    *,
    current_A: float | None,
    magnetic_energy_J: float | None,
) -> float | None:
    if current_A is None or magnetic_energy_J is None or current_A == 0.0:
        return None
    return float(2.0 * magnetic_energy_J / (current_A * current_A))


def _power_port_step_records(
    *,
    circuit_step: Mapping[str, Any] | None,
    udpf_source: str | None,
    current_A: float | None,
    terminal_voltage_V: float | None,
    active_power_W: float | None,
    final_energy: Mapping[str, Any] | None,
    field_work: Mapping[str, Any] | None,
    conservation: Mapping[str, Any] | None,
    diagnostic_field_inductance_H: float | None,
    residual_budget: Mapping[str, Any],
    low_current_feedback: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if circuit_step is None:
        return []
    return [
        {
            "status": "candidate_power_port_step_not_validation",
            "interface_surface_or_volume_domain": "not_declared",
            "terminal_current_A": current_A,
            "terminal_voltage_V": terminal_voltage_V,
            "active_power_W": active_power_W,
            "active_load_relation": _active_load_relation(circuit_step, udpf_source),
            "udpf_source": udpf_source,
            "poynting_power_W": None,
            "j_dot_e_power_W": _optional_float(field_work, "j_dot_e_power_W"),
            "j_dot_e_domain": None if field_work is None else field_work.get("domain"),
            "external_circuit_energy_J": _optional_float(
                final_energy,
                "circuit_energy_J",
            ),
            "magnetic_energy_J": _optional_float(final_energy, "magnetic_energy_J"),
            "electric_energy_J": _optional_float(final_energy, "electric_energy_J"),
            "thermal_energy_J": _optional_float(
                final_energy,
                "electron_internal_energy_J",
            ),
            "particle_kinetic_energy_J": _optional_float(
                final_energy,
                "particle_kinetic_energy_J",
            ),
            "tracked_total_energy_delta_J": (
                None
                if conservation is None
                else conservation.get("delta_tracked_total_energy_J")
            ),
            "candidate_power_residual_budget": dict(residual_budget),
            "low_current_p_over_i_singularity": (
                dict(low_current_feedback)
                if low_current_feedback is not None
                else _missing_low_current_feedback_packet()
            ),
            "residual_interpretation": (
                "tracked_energy_delta_not_accepted_power_port_residual"
            ),
            "diagnostic_field_inductance_H": diagnostic_field_inductance_H,
            "magnetic_energy_inductance_authority": (
                "diagnostic_only_not_circuit_load"
            ),
            "sign_convention": "not_accepted",
            "time_centering": "candidate_runner_step_metadata_only",
            "can_support_first_principles_acceptance": False,
        }
    ]


def _power_port_channel_statuses(
    *,
    circuit_step: Mapping[str, Any] | None,
    final_energy: Mapping[str, Any] | None,
    field_work: Mapping[str, Any] | None,
    startup: Mapping[str, Any] | None,
) -> dict[str, str]:
    candidate_channels: set[str] = set()
    if circuit_step is not None:
        candidate_channels.update(("terminal_current", "terminal_voltage"))
    if final_energy is not None:
        for channel, energy_key in ENERGY_LEDGER_CHANNELS:
            if _optional_float(final_energy, energy_key) is not None:
                candidate_channels.add(channel)
    if startup is not None:
        candidate_channels.add("startup_handoff_interval")
    if _optional_float(field_work, "j_dot_e_power_W") is not None:
        candidate_channels.add("poynting_power_or_j_dot_e")
    candidate_channels.add("source_references")

    statuses: dict[str, str] = {}
    for channel in REQUIRED_POWER_PORT_CHANNELS:
        if channel in {"auluck_eq6_power_balance", "residual"}:
            # Auluck eq. (6) is not closed: terms II/IV/V/VI fail closed on
            # Sigma_p geometry, so the six-term balance channel is blocked.
            statuses[channel] = "missing_or_blocked"
        elif channel == "poynting_power_or_j_dot_e" and channel in candidate_channels:
            statuses[channel] = "candidate_runtime_only_not_acceptance"
        elif channel == "poynting_power_or_j_dot_e":
            statuses[channel] = "missing_or_blocked"
        elif channel in {"sign_convention", "time_centering"}:
            statuses[channel] = "candidate_metadata_only_not_acceptance"
        elif channel in {"interface_surface_or_volume_domain", "boundary_labels"}:
            statuses[channel] = "missing_or_blocked"
        elif channel in candidate_channels:
            statuses[channel] = "candidate_runtime_only_not_acceptance"
        else:
            statuses[channel] = "missing_or_blocked"
    return statuses


def _energy_ledger_status(
    final_energy: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for channel, energy_key in ENERGY_LEDGER_CHANNELS:
        value = _optional_float(final_energy, energy_key)
        statuses[channel] = {
            "source_key": energy_key,
            "value_J": value,
            "status": (
                "candidate_runtime_only_not_acceptance"
                if value is not None
                else "missing_or_blocked"
            ),
            "can_support_power_port_acceptance": False,
        }
    return statuses


def _stage0_packet_scaffolds(
    *,
    active_load_relation: str,
    current_A: float | None,
    terminal_voltage_V: float | None,
    final_energy: Mapping[str, Any] | None,
    field_work: Mapping[str, Any] | None,
    residual_budget: Mapping[str, Any],
    low_current_feedback: Mapping[str, Any] | None,
    startup: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    return {
        "power_port_source_review": {
            "status": "candidate_source_review_packet_not_validation",
            "supported_by_local_sources": [
                "Auluck volume field-power voltage over declared Omega",
                "Poynting-flux power ledger at declared source interface",
                "hybrid-PIC external circuit with source-derived U_DPF",
            ],
            "deferred_or_unverified_by_local_sources": [
                "Sigma quasi-TEM line-voltage driver",
            ],
            "source_references": list(POWER_PORT_SOURCE_REFS),
            "can_support_power_port_acceptance": False,
        },
        "power_port_domain_review": {
            "status": "blocked_domain_packet_not_available",
            "active_load_relation": active_load_relation,
            "declared_runtime_domain": (
                None if field_work is None else field_work.get("domain")
            ),
            "required_for_acceptance": [
                "named integration volume or interface surface",
                "explicit source-interface exclusion",
                "boundary labels and electrode/interface partition",
            ],
            "can_support_power_port_acceptance": False,
        },
        "power_port_sign_review": {
            "status": "candidate_sign_packet_not_validation",
            "terminal_current_A": current_A,
            "terminal_voltage_V": terminal_voltage_V,
            "sign_policy": (
                "signed J.E and I*U_DPF are retained; negative local J.E is not "
                "automatically clipped or treated as a limiter condition"
            ),
            "required_for_acceptance": [
                "source-reviewed sign convention",
                "sign-reversal negative test",
            ],
            "can_support_power_port_acceptance": False,
        },
        "power_port_time_centering_review": {
            "status": "candidate_time_centering_packet_not_validation",
            "runtime_time_centering": "begin_step_or_retained_step_metadata",
            "required_for_acceptance": [
                "time-centered terminal power or field-power integral",
                "time-centering downgrade negative test",
            ],
            "can_support_power_port_acceptance": False,
        },
        "power_port_energy_ledger_review": {
            "status": "candidate_energy_ledger_packet_not_validation",
            "ledger": _candidate_stage0_energy_ledger(
                final_energy=final_energy,
                conservation=None,
                residual_budget=residual_budget,
            ),
            "can_support_power_port_acceptance": False,
        },
        "negative_test_plan": {
            "status": "candidate_negative_test_plan_not_validation",
            "required_negative_tests": [
                "sign_reversal_fails_residual_budget",
                "domain_corruption_fails_domain_review",
                "time_centering_downgrade_fails_time_review",
                "low_current_p_over_i_singularity_detected",
                "sigma_line_voltage_driver_rejected_until_source_packet_exists",
            ],
            "startup_blocker_visible": (
                bool(startup.get("whole_shot_startup_blocked"))
                if startup is not None
                else None
            ),
            "low_current_feedback_status": (
                None
                if low_current_feedback is None
                else low_current_feedback.get("status")
            ),
            "can_support_power_port_acceptance": False,
        },
    }


def _candidate_stage0_energy_ledger(
    *,
    final_energy: Mapping[str, Any] | None,
    conservation: Mapping[str, Any] | None,
    residual_budget: Mapping[str, Any],
) -> dict[str, Any]:
    initial = _energy_section(conservation, "initial")
    final = final_energy
    stored_em_delta_J = None
    if initial is not None and final is not None:
        initial_em = _sum_optional(
            _optional_float(initial, "electric_energy_J"),
            _optional_float(initial, "magnetic_energy_J"),
        )
        final_em = _sum_optional(
            _optional_float(final, "electric_energy_J"),
            _optional_float(final, "magnetic_energy_J"),
        )
        stored_em_delta_J = _difference(final_em, initial_em)
    # WP-N1B: this Stage-0 scaffold is a coarse Poynting accounting surface,
    # NOT the Auluck eq. (6) ledger. The non-Auluck "electrode_interface_work"
    # key has been removed -- the verified extract is explicit that Auluck's
    # balance has no electrode-contact-work term (the source interface is
    # excluded from Omega and its flux IS the LHS I(t)V(t)). The accepted
    # Auluck six-term decomposition is build_wp_n1_auluck_power_port_ledger.
    # [AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md, "What this source ...
    # DOES NOT provide"]
    return {
        "status": "candidate_stage0_energy_ledger_not_validation",
        "terms": {
            "terminal_port_work_J": residual_budget.get(
                "cumulative_terminal_active_port_work_J"
            ),
            "volume_j_dot_e_work_J": residual_budget.get(
                "integrated_volume_j_dot_e_work_J"
            ),
            "stored_em_energy_delta_J": stored_em_delta_J,
            "wall_poynting_flux_excluding_declared_port_J": None,
        },
        "term_status": {
            "terminal_port_work_J": _candidate_or_missing(
                residual_budget.get("cumulative_terminal_active_port_work_J")
            ),
            "volume_j_dot_e_work_J": _candidate_or_missing(
                residual_budget.get("integrated_volume_j_dot_e_work_J")
            ),
            "stored_em_energy_delta_J": _candidate_or_missing(stored_em_delta_J),
            "wall_poynting_flux_excluding_declared_port_J": "missing_or_blocked",
        },
        "auluck_six_term_ledger_is": "build_wp_n1_auluck_power_port_ledger",
        "source_basis": [
            "Poynting theorem",
            "Auluck source-interface exclusion and field-power relation",
        ],
        "interpretation": (
            "candidate accounting surface only; the Auluck eq. (6) six-term "
            "ledger is the power-balance authority and is not yet closed"
        ),
        "can_support_power_port_acceptance": False,
    }


def _candidate_or_missing(value: Any) -> str:
    return (
        "candidate_runtime_only_not_acceptance"
        if value is not None
        else "missing_or_blocked"
    )


def _operator_comparison_packet() -> dict[str, Any]:
    return {
        "status": "candidate_operator_comparison_not_validation",
        "operators": {
            "auluck_volume_j_dot_e": {
                "role": "candidate field-power voltage relation",
                "formula": "U_DPF = - integral_Omega(J.E)dV / I",
                "source_status": "supported_by_local_auluck_source",
                "runtime_status": "candidate_lagged_driver_or_ledger_only",
                "main_risk": "1/I singularity and domain/sign/time-centering review",
                "can_be_accepted_now": False,
            },
            "poynting_surface_flux": {
                "role": "candidate power ledger or future driver basis",
                "formula": "I*U_DPF equals declared source-interface Poynting flux",
                "source_status": "supported_by_auluck_and_poynting_theorem",
                "runtime_status": "not_integrated_as_active_driver",
                "main_risk": "source interface and wall/electrode partition missing",
                "can_be_accepted_now": False,
            },
            "hybrid_pic_magnetic_flux_udpf": {
                "role": "external-circuit coupling pattern",
                "formula": "circuit ODE uses U_DPF from field integration/time derivative",
                "source_status": "supported_by_local_hybrid_pic_source",
                "runtime_status": "architecture_pattern_not_pf1000_acceptance",
                "main_risk": "not same-scope PF-1000/Akel acceptance evidence",
                "can_be_accepted_now": False,
            },
            "sigma_quasi_tem_line_voltage": {
                "role": "deferred proposed boundary-port line-voltage driver",
                "formula": "U_DPF = integral_path E.dl on Sigma plane",
                "source_status": "not_verified_in_local_dpf_source",
                "runtime_status": "deferred_exploratory_telemetry_only",
                "main_risk": "assumes port plane and quasi-TEM path equivalence",
                "can_be_accepted_now": False,
            },
        },
        "decision": "do_not_replace_active_driver_with_sigma_line_voltage",
        "can_support_power_port_acceptance": False,
    }


def _sigma_quasi_tem_line_voltage_packet() -> dict[str, Any]:
    return {
        "status": "deferred_sigma_quasi_tem_driver_not_source_verified",
        "proposed_quantity": "line integral of E between terminals at Sigma plane",
        "would_require": [
            "source-verified DPF port plane definition",
            "path-independence or bounded path-spread evidence",
            "proof that omitted wall/electrode/interface work is accounted for",
            "negative tests against Auluck/Poynting domain corruption",
        ],
        "allowed_runtime_use": "exploratory_diagnostic_only",
        "disallowed_runtime_use": "accepted_or_primary_circuit_driver",
        "difference_from_auluck": (
            "Auluck derives voltage from total field power over a declared DPF "
            "domain divided by terminal current; Sigma line voltage samples an "
            "electric-field path at a boundary plane."
        ),
        "difference_from_poynting": (
            "Poynting flux accounts surface power through a declared interface; "
            "Sigma line voltage would need a separate current/power closure."
        ),
        "can_support_power_port_acceptance": False,
    }


def _missing_low_current_feedback_packet() -> dict[str, Any]:
    return {
        "status": "low_current_p_over_i_feedback_not_reported",
        "p_over_i_formula_active_candidate": None,
        "low_current_threshold_hit": None,
        "singularity_blocked_this_step": None,
        "can_support_power_port_acceptance": False,
    }


def _candidate_runtime_channels(
    *,
    circuit_step: Mapping[str, Any] | None,
    final_energy: Mapping[str, Any] | None,
    field_work: Mapping[str, Any] | None,
    diagnostic_field_inductance_H: float | None,
    residual_budget: Mapping[str, Any],
) -> list[str]:
    channels: set[str] = set()
    if circuit_step is not None:
        channels.add("candidate_terminal_current_voltage")
        channels.add("runtime_input_voltage_sequence_not_load_authority")
    if final_energy is not None:
        channels.add("candidate_tracked_energy_ledger")
    if _optional_float(final_energy, "circuit_energy_J") is not None:
        channels.add("candidate_external_circuit_energy")
    if _optional_float(field_work, "j_dot_e_power_W") is not None:
        channels.add("candidate_volume_j_dot_e_power")
    if diagnostic_field_inductance_H is not None:
        channels.add("candidate_diagnostic_field_inductance")
    if residual_budget.get("available") is True:
        channels.add("candidate_power_residual_budget")
    if residual_budget.get("cumulative_terminal_active_port_work_J") is not None:
        channels.add("candidate_cumulative_terminal_i_udpf_work")
    if residual_budget.get("full_completed_step_active_port_integral_available") is True:
        channels.add("candidate_full_completed_step_terminal_i_udpf_integral")
    return sorted(channels)


def _candidate_power_residual_budget(
    *,
    conservation: Mapping[str, Any] | None,
    simulation_telemetry: Mapping[str, Any] | None,
    field_work: Mapping[str, Any] | None,
    active_power_W: float | None,
) -> dict[str, Any]:
    delta_energy_J = (
        None
        if conservation is None
        else _optional_float(conservation, "delta_tracked_total_energy_J")
    )
    initial_total_J = _optional_float(_energy_section(conservation, "initial"), "tracked_total_energy_J")
    final_total_J = _optional_float(_energy_section(conservation, "final"), "tracked_total_energy_J")
    dt_s = _optional_float(simulation_telemetry, "dt_s")
    if dt_s is None:
        dt_s = _optional_float(conservation, "dt_s")
    n_steps_completed = _optional_float(simulation_telemetry, "n_steps_completed")
    history_stride = _optional_float(simulation_telemetry, "history_stride")
    retained_history = _retained_history(simulation_telemetry)
    retained_j_dot_e_work_J = _retained_j_dot_e_work_J(
        retained_history=retained_history,
        dt_s=dt_s,
    )
    cumulative_j_dot_e_work_J = _optional_float(
        simulation_telemetry,
        "cumulative_j_dot_e_work_J",
    )
    cumulative_j_dot_e_step_count = _optional_float(
        simulation_telemetry,
        "cumulative_j_dot_e_step_count",
    )
    cumulative_active_port_work_J = _optional_float(
        simulation_telemetry,
        "cumulative_active_port_work_J",
    )
    cumulative_active_port_step_count = _optional_float(
        simulation_telemetry,
        "cumulative_active_port_step_count",
    )
    udpf_source_counts = (
        simulation_telemetry.get("udpf_source_counts")
        if isinstance(simulation_telemetry, Mapping)
        else None
    )
    last_j_dot_e_work_J = (
        None
        if dt_s is None or _optional_float(field_work, "j_dot_e_power_W") is None
        else _optional_float(field_work, "j_dot_e_power_W") * dt_s
    )
    terminal_active_work_last_step_J = (
        None if active_power_W is None or dt_s is None else active_power_W * dt_s
    )
    denominator = _residual_denominator(
        initial_total_J,
        final_total_J,
        cumulative_j_dot_e_work_J,
        cumulative_active_port_work_J,
        retained_j_dot_e_work_J,
        last_j_dot_e_work_J,
    )
    integrated_j_dot_e_work_J = (
        cumulative_j_dot_e_work_J
        if cumulative_j_dot_e_work_J is not None
        else retained_j_dot_e_work_J
    )
    delta_minus_retained = _difference(delta_energy_J, retained_j_dot_e_work_J)
    delta_plus_retained = _sum_optional(delta_energy_J, retained_j_dot_e_work_J)
    delta_minus_integrated = _difference(delta_energy_J, integrated_j_dot_e_work_J)
    delta_plus_integrated = _sum_optional(delta_energy_J, integrated_j_dot_e_work_J)
    delta_minus_active_port = _difference(delta_energy_J, cumulative_active_port_work_J)
    delta_plus_active_port = _sum_optional(delta_energy_J, cumulative_active_port_work_J)
    active_minus_j_dot_e = _difference(
        cumulative_active_port_work_J,
        integrated_j_dot_e_work_J,
    )
    active_plus_j_dot_e = _sum_optional(
        cumulative_active_port_work_J,
        integrated_j_dot_e_work_J,
    )
    available = delta_energy_J is not None and (
        integrated_j_dot_e_work_J is not None
        or cumulative_active_port_work_J is not None
        or last_j_dot_e_work_J is not None
    )
    full_retained_history = (
        n_steps_completed is not None
        and int(n_steps_completed) == len(retained_history)
        and (history_stride is None or int(history_stride) == 1)
    )
    full_completed_step_integral = (
        n_steps_completed is not None
        and cumulative_j_dot_e_step_count is not None
        and int(cumulative_j_dot_e_step_count) == int(n_steps_completed)
    )
    full_completed_step_active_port_integral = (
        n_steps_completed is not None
        and cumulative_active_port_step_count is not None
        and int(cumulative_active_port_step_count) == int(n_steps_completed)
    )
    return {
        "status": "candidate_power_residual_budget_not_validation"
        if available
        else "candidate_power_residual_budget_missing_runtime_channels",
        "available": available,
        "tracked_energy_delta_J": delta_energy_J,
        "initial_tracked_total_energy_J": initial_total_J,
        "final_tracked_total_energy_J": final_total_J,
        "integrated_volume_j_dot_e_work_J": integrated_j_dot_e_work_J,
        "integrated_volume_j_dot_e_work_source": (
            "simulator_cumulative_all_completed_steps"
            if cumulative_j_dot_e_work_J is not None
            else (
                "retained_history_rectangular_sum"
                if retained_j_dot_e_work_J is not None
                else None
            )
        ),
        "cumulative_volume_j_dot_e_work_J": cumulative_j_dot_e_work_J,
        "cumulative_volume_j_dot_e_step_count": (
            None
            if cumulative_j_dot_e_step_count is None
            else int(cumulative_j_dot_e_step_count)
        ),
        "retained_volume_j_dot_e_work_J": retained_j_dot_e_work_J,
        "last_step_volume_j_dot_e_work_J": last_j_dot_e_work_J,
        "terminal_active_power_work_last_step_J": terminal_active_work_last_step_J,
        "cumulative_terminal_active_port_work_J": cumulative_active_port_work_J,
        "cumulative_terminal_active_port_step_count": (
            None
            if cumulative_active_port_step_count is None
            else int(cumulative_active_port_step_count)
        ),
        "udpf_source_counts": (
            dict(udpf_source_counts) if isinstance(udpf_source_counts, Mapping) else {}
        ),
        "delta_minus_active_port_work_J": delta_minus_active_port,
        "delta_plus_active_port_work_J": delta_plus_active_port,
        "delta_minus_active_port_fraction": _fraction(
            delta_minus_active_port,
            denominator,
        ),
        "delta_plus_active_port_fraction": _fraction(
            delta_plus_active_port,
            denominator,
        ),
        "active_port_minus_integrated_j_dot_e_work_J": active_minus_j_dot_e,
        "active_port_plus_integrated_j_dot_e_work_J": active_plus_j_dot_e,
        "active_port_minus_integrated_j_dot_e_fraction": _fraction(
            active_minus_j_dot_e,
            denominator,
        ),
        "active_port_plus_integrated_j_dot_e_fraction": _fraction(
            active_plus_j_dot_e,
            denominator,
        ),
        "delta_minus_integrated_j_dot_e_work_J": delta_minus_integrated,
        "delta_plus_integrated_j_dot_e_work_J": delta_plus_integrated,
        "delta_minus_integrated_j_dot_e_fraction": _fraction(
            delta_minus_integrated,
            denominator,
        ),
        "delta_plus_integrated_j_dot_e_fraction": _fraction(
            delta_plus_integrated,
            denominator,
        ),
        "delta_minus_retained_j_dot_e_work_J": delta_minus_retained,
        "delta_plus_retained_j_dot_e_work_J": delta_plus_retained,
        "delta_minus_retained_j_dot_e_fraction": _fraction(
            delta_minus_retained,
            denominator,
        ),
        "delta_plus_retained_j_dot_e_fraction": _fraction(
            delta_plus_retained,
            denominator,
        ),
        "retained_history_step_count": len(retained_history),
        "n_steps_completed": (
            None if n_steps_completed is None else int(n_steps_completed)
        ),
        "history_stride": None if history_stride is None else int(history_stride),
        "full_retained_history_available": full_retained_history,
        "full_completed_step_j_dot_e_integral_available": (
            full_completed_step_integral
        ),
        "full_completed_step_active_port_integral_available": (
            full_completed_step_active_port_integral
        ),
        "sign_convention": (
            "positive_J_dot_E_is_field_work_on_charges_candidate_not_accepted"
        ),
        "time_centering": "candidate_retained_step_history_not_accepted",
        "accepted_residual_tolerance": "not_attached",
        "interpretation": (
            "candidate_budget_for_engineering_debug_only_not_power_port_acceptance"
        ),
        "can_support_power_port_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _retained_history(
    simulation_telemetry: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if simulation_telemetry is None:
        return []
    history = simulation_telemetry.get("history_summary")
    if not isinstance(history, list):
        return []
    return [item for item in history if isinstance(item, Mapping)]


def _retained_j_dot_e_work_J(
    *,
    retained_history: list[Mapping[str, Any]],
    dt_s: float | None,
) -> float | None:
    if dt_s is None:
        return None
    powers = [
        _optional_float(item, "j_dot_e_power_W")
        for item in retained_history
        if _optional_float(item, "j_dot_e_power_W") is not None
    ]
    if not powers:
        return None
    return float(sum(powers) * dt_s)


def _residual_denominator(*values: float | None) -> float | None:
    finite = [abs(float(value)) for value in values if value is not None]
    if not finite:
        return None
    return max(max(finite), 1.0)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _sum_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) + float(right)


def _fraction(value: float | None, denominator: float | None) -> float | None:
    if value is None or denominator is None or denominator == 0.0:
        return None
    return float(value) / float(denominator)
