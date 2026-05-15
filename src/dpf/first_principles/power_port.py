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


def build_engineering_power_port_packet(
    circuit: Mapping[str, Any] | None,
    *,
    startup: Mapping[str, Any] | None = None,
    conservation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting power-port packet for the current runner state."""
    circuit_step = _last_circuit_step(circuit)
    current_A = _optional_float(circuit_step, "current_A")
    terminal_voltage_V = _optional_float(circuit_step, "udpf_V")
    final_energy = _energy_section(conservation, "final")
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
    missing = list(ACCEPTANCE_BLOCKING_CHANNELS)
    if circuit_step is None:
        missing.extend(("terminal_current", "terminal_voltage", "active_load_relation"))
    if startup and startup.get("whole_shot_startup_blocked") is True:
        missing.append("startup_handoff_interval")

    return {
        "status": "candidate_engineering_power_port_not_validation",
        "authority_contract": "field_power_required",
        "active_load_relation": (
            "input_udpf_placeholder_times_current"
            if circuit_step is not None
            else "no_active_circuit_boundary"
        ),
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
            current_A=current_A,
            terminal_voltage_V=terminal_voltage_V,
            active_power_W=active_power_W,
            final_energy=final_energy,
            conservation=conservation,
            diagnostic_field_inductance_H=diagnostic_field_inductance_H,
        ),
        "interface_surface_or_volume_domain": "not_declared",
        "poynting_power_W": None,
        "j_dot_e_power_W": None,
        "electrode_work_J": None,
        "time_centering": "candidate_runner_step_metadata_only",
        "sign_convention": "not_accepted",
        "required_channels": list(REQUIRED_POWER_PORT_CHANNELS),
        "missing_acceptance_channels": sorted(set(missing)),
        "candidate_runtime_channels": _candidate_runtime_channels(
            circuit_step=circuit_step,
            final_energy=final_energy,
            diagnostic_field_inductance_H=diagnostic_field_inductance_H,
        ),
        "source_references": list(POWER_PORT_SOURCE_REFS),
        "conservation_status": None if conservation is None else conservation.get("status"),
        "startup_handoff_required": (
            bool(startup.get("whole_shot_startup_blocked")) if startup else None
        ),
        "can_support_first_principles_acceptance": False,
    }


def _last_circuit_step(circuit: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not circuit:
        return None
    last = circuit.get("last")
    if not isinstance(last, Mapping):
        return None
    step = last.get("circuit_step")
    return step if isinstance(step, Mapping) else None


def _optional_float(mapping: Mapping[str, Any] | None, key: str) -> float | None:
    if mapping is None or mapping.get(key) is None:
        return None
    return float(mapping[key])


def _energy_section(
    conservation: Mapping[str, Any] | None,
    section: str,
) -> Mapping[str, Any] | None:
    if conservation is None:
        return None
    value = conservation.get(section)
    return value if isinstance(value, Mapping) else None


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
    current_A: float | None,
    terminal_voltage_V: float | None,
    active_power_W: float | None,
    final_energy: Mapping[str, Any] | None,
    conservation: Mapping[str, Any] | None,
    diagnostic_field_inductance_H: float | None,
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
            "active_load_relation": "input_udpf_placeholder_times_current",
            "poynting_power_W": None,
            "j_dot_e_power_W": None,
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


def _candidate_runtime_channels(
    *,
    circuit_step: Mapping[str, Any] | None,
    final_energy: Mapping[str, Any] | None,
    diagnostic_field_inductance_H: float | None,
) -> list[str]:
    channels: set[str] = set()
    if circuit_step is not None:
        channels.add("candidate_terminal_current_voltage")
        channels.add("candidate_active_load_placeholder")
    if final_energy is not None:
        channels.add("candidate_tracked_energy_ledger")
    if _optional_float(final_energy, "circuit_energy_J") is not None:
        channels.add("candidate_external_circuit_energy")
    if diagnostic_field_inductance_H is not None:
        channels.add("candidate_diagnostic_field_inductance")
    return sorted(channels)
