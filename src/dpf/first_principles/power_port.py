"""Fail-closed power-port packets for package-native first-principles runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

POWER_PORT_SOURCE_REFS = (
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

REQUIRED_POWER_PORT_CHANNELS = (
    "interface_surface_or_volume_domain",
    "terminal_current",
    "terminal_voltage",
    "poynting_power_or_j_dot_e",
    "electrode_work",
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

ACCEPTANCE_BLOCKING_CHANNELS = (
    "named_interface_surface_or_volume_domain",
    "poynting_or_j_dot_e_power_integral",
    "electrode_work_partition",
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
        "electrode_work_J": None,
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
            "centering_electrode_work_residual_tolerance_hashes_and_review_pass"
        ),
        "negative_test_policy": {
            "sign_convention_reversal_required": True,
            "time_centering_mismatch_required": True,
            "poynting_j_dot_e_non_equivalence_required": True,
            "electrode_work_omission_required": True,
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
    """Return the simulator-emitted WP-N1 five-term ledger, if present."""
    if simulation_telemetry is None:
        return None
    ledger = simulation_telemetry.get("power_port_ledger")
    return ledger if isinstance(ledger, Mapping) else None


def _residual_fraction(
    residual_J: float | None,
    terminal_port_work_J: float | None,
    volume_j_dot_e_work_J: float | None,
) -> float | None:
    """Return residual / max(|terminal|, |volume J.E|, 1 J) per packet S4."""
    if residual_J is None:
        return None
    denom = _residual_denominator(
        terminal_port_work_J,
        volume_j_dot_e_work_J,
    )
    if denom is None or denom == 0.0:
        return None
    return float(residual_J) / float(denom)


def build_wp_n1_auluck_power_port_ledger(
    ledger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the WP-N1 Auluck named-domain five-term energy ledger.

    Implements WP-N1 source packet sections 1-4. Terms 1, 2, 3, 5 are taken
    from the simulator-emitted cumulative ledger. Term 4
    (electrode_interface_work_J) is the labeled NON-INDEPENDENT closure
    estimate per source packet section 6 gap G1 option (b): the Auluck eq 5/6
    moving-boundary integrand is OCR-illegible and must not be invented, so
    term 4 is defined as
    terminal_port_work - volume_j_dot_e_work - wall_poynting - stored_em_delta.
    This makes residual_J trivially close; that is EXPECTED and means the port
    CANNOT support acceptance until an independent integrand exists.
    """
    if ledger is None:
        return {
            "status": "blocked_wp_n1_power_port_ledger_not_available",
            "reason": (
                "no circuit-coupled run; simulator emitted no power_port_ledger"
            ),
            "auluck_omega_domain": _omega_domain_unavailable_packet(),
            "energy_ledger_terms_J": {key: None for key in _WP_N1_LEDGER_KEYS},
            "residual_J": None,
            "residual_fraction": None,
            "accepted_residual_tolerance": "not_attached",
            "can_support_power_port_acceptance": False,
            "can_support_first_principles_acceptance": False,
        }

    terminal_port_work_J = _optional_float(ledger, "cumulative_terminal_port_work_J")
    volume_j_dot_e_work_J = _optional_float(
        ledger, "cumulative_omega_volume_j_dot_e_work_J"
    )
    wall_poynting_J = _optional_float(
        ledger, "cumulative_wall_poynting_flux_excluding_declared_port_J"
    )
    stored_em_delta_J = _optional_float(ledger, "stored_em_energy_delta_J")

    # Term 4 (G1): labeled non-independent closure estimate. NOT an
    # independent integrand. Defined so the five-term sum closes by
    # construction; residual is therefore trivially ~0.
    electrode_interface_work_J = None
    if (
        terminal_port_work_J is not None
        and volume_j_dot_e_work_J is not None
        and wall_poynting_J is not None
        and stored_em_delta_J is not None
    ):
        electrode_interface_work_J = (
            terminal_port_work_J
            - volume_j_dot_e_work_J
            - wall_poynting_J
            - stored_em_delta_J
        )

    # Residual policy S4: residual = term1 - term2 - term3 - term5 - term4.
    residual_J = None
    if (
        terminal_port_work_J is not None
        and volume_j_dot_e_work_J is not None
        and wall_poynting_J is not None
        and stored_em_delta_J is not None
        and electrode_interface_work_J is not None
    ):
        residual_J = (
            terminal_port_work_J
            - volume_j_dot_e_work_J
            - wall_poynting_J
            - stored_em_delta_J
            - electrode_interface_work_J
        )

    domain_partition = ledger.get("domain_partition")
    auluck_omega_domain = _build_auluck_omega_domain_packet(
        domain_partition if isinstance(domain_partition, Mapping) else None
    )

    return {
        "status": "candidate_wp_n1_auluck_five_term_power_port_ledger_not_validation",
        "source_basis": [
            "Auluck 2021 DPF circuit-element relation eq 1 and Omega domain",
            "NRL Plasma Formulary 2019 Poynting theorem",
        ],
        "source_refs": [
            "KnowledgeReference/auluck-2021-dpf-circuit-element.md:173-257",
            "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1869-1888",
        ],
        "auluck_omega_domain": auluck_omega_domain,
        "energy_ledger_terms_J": {
            "terminal_port_work_J": terminal_port_work_J,
            "volume_j_dot_e_work_J": volume_j_dot_e_work_J,
            "wall_poynting_flux_excluding_declared_port_J": wall_poynting_J,
            "stored_em_energy_delta_J": stored_em_delta_J,
            "electrode_interface_work_J": electrode_interface_work_J,
        },
        "energy_ledger_term_status": {
            "terminal_port_work_J": _candidate_or_missing(terminal_port_work_J),
            "volume_j_dot_e_work_J": _candidate_or_missing(volume_j_dot_e_work_J),
            "wall_poynting_flux_excluding_declared_port_J": _candidate_or_missing(
                wall_poynting_J
            ),
            "stored_em_energy_delta_J": _candidate_or_missing(stored_em_delta_J),
            "electrode_interface_work_J": (
                "closure_estimate_not_independent_blocked_g1"
                if electrode_interface_work_J is not None
                else "missing_or_blocked"
            ),
        },
        "electrode_interface_work_J__closure_estimate_not_independent": (
            electrode_interface_work_J
        ),
        "electrode_interface_work_independence": "not_independent_closure_estimate",
        "electrode_interface_work_blocker": "G1_auluck_eq_5_6_ocr_illegible",
        "fully_implemented_terms": [
            "terminal_port_work_J",
            "volume_j_dot_e_work_J",
            "wall_poynting_flux_excluding_declared_port_J",
            "stored_em_energy_delta_J",
        ],
        "closure_estimate_terms": ["electrode_interface_work_J"],
        "residual_J": residual_J,
        "residual_fraction": _residual_fraction(
            residual_J,
            terminal_port_work_J,
            volume_j_dot_e_work_J,
        ),
        "residual_definition": (
            "terminal_port_work - volume_j_dot_e_work - wall_poynting "
            "- stored_em_energy_delta - electrode_interface_work"
        ),
        "residual_interpretation": (
            "trivially_closes_because_term_4_is_closure_estimate_g1; "
            "engineering_debug_diagnostic_only_not_power_port_acceptance"
        ),
        "accepted_residual_tolerance": "not_attached",
        "tracked_energy_delta_is_residual": False,
        "sign_convention": _wp_n1_sign_convention_packet(),
        "time_centering": _wp_n1_time_centering_packet(ledger),
        "first_step_fallback": bool(ledger.get("first_step_fallback")),
        "first_step_udpf_source": ledger.get("first_step_udpf_source"),
        "steps_accumulated": ledger.get("steps_accumulated"),
        "scientific_status": "engineering_candidate_not_validation",
        "can_support_power_port_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


_WP_N1_LEDGER_KEYS = (
    "terminal_port_work_J",
    "volume_j_dot_e_work_J",
    "wall_poynting_flux_excluding_declared_port_J",
    "stored_em_energy_delta_J",
    "electrode_interface_work_J",
)


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


def _wp_n1_sign_convention_packet() -> dict[str, Any]:
    """Return the WP-N1 S3.1 declared sign convention."""
    return {
        "status": "candidate_wp_n1_sign_convention_not_accepted",
        "basis": "NRL Plasma Formulary 2019 Poynting theorem",
        "source_ref": "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1880-1888",
        "surface_flux_terms": "outflow_positive_dS_outward",
        "terminal_port_work_J": "positive_means_energy_entering_omega_from_generator",
        "wall_poynting_flux_J": "positive_means_energy_leaving_omega_through_walls",
        "volume_j_dot_e_work_J": "positive_means_field_work_on_charges",
        "stored_em_energy_delta_J": "positive_means_stored_em_energy_increased",
        "electrode_interface_work_J": "into_omega_positive_consistent_with_terminal",
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
    """Return the WP-N1 S5 six-negative-test policy block.

    Enumerates exactly the six audit A-5 negative tests and the emitted
    packet field each asserts on. The tests themselves live in
    tests/test_first_principles_power_port.py and assert against these
    emitted fields.
    """
    return {
        "status": "candidate_wp_n1_negative_test_policy_not_validation",
        "required_negative_tests": {
            "N1_sign_reversal": {
                "corruption": "flip sign of terminal_port_work_J",
                "asserts_on": "residual_J and residual_fraction",
                "expected": "residual jumps by ~2*terminal_port_work; fraction O(1)",
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
            "N3_omitted_electrode_work": {
                "corruption": "drop term 4 from the residual sum (set to 0)",
                "asserts_on": "residual_J with electrode_interface_work_J omitted",
                "expected": (
                    "residual no longer closes when dL/dt-type work is nonzero"
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
                    "claimed; all five terms share one centering"
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
        "acceptance_unblocks_only_when": (
            "source_backed_residual_tolerance_attached AND all_six_tests_pass"
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
            "electrode_work_J": None,
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
        if channel in {"electrode_work", "residual"}:
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
            "electrode_interface_work_J": None,
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
            "electrode_interface_work_J": "missing_or_blocked",
        },
        "source_basis": [
            "Poynting theorem",
            "Auluck source-interface exclusion and field-power relation",
        ],
        "interpretation": (
            "candidate accounting surface only; missing wall Poynting and "
            "electrode/interface work prevent power-port acceptance"
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
