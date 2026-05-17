"""Fail-closed startup breakdown audit for first-principles runner telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dpf.experimental.civ_breakdown import (
    civ_breakdown_model_metadata,
    compute_breakdown,
    compute_initial_sheath_state,
    compute_liftoff_delay,
)

STARTUP_BREAKDOWN_AUDIT_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/alfven-ionization-in-an-mhd-gas-interactions-code.md",
        "lines": "420-447",
        "role": "critical_ionization_velocity_candidate_closure_context",
    },
    {
        "path": "KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md",
        "lines": "56-74",
        "role": "pf1000_surface_discharge_avalanche_streamer_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "unlimited-release-printed-september-2009-alegra-hedp-simulations-of-the-dense-plasma-focus.md"
        ),
        "lines": "245-392,555-585",
        "role": "breakdown_liftoff_sequence_and_pic_to_mhd_startup_handoff_requirement",
    },
    {
        "path": "docs/FIRST_PRINCIPLES_BLOCKER_SOURCE_SEARCH_2026_05_15.md",
        "lines": "1-118",
        "role": "local_source_truth_startup_blocker_contract",
    },
)


def build_candidate_startup_breakdown_audit(
    *,
    device: Mapping[str, Any],
    gas: Mapping[str, Any],
    circuit: Mapping[str, Any],
    startup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return computable breakdown/liftoff telemetry without promotion authority."""

    metadata = civ_breakdown_model_metadata()
    missing = _missing_required_inputs(device=device, gas=gas, circuit=circuit)
    if missing:
        return _blocked_packet(
            metadata=metadata,
            status="candidate_civ_paschen_breakdown_audit_unavailable",
            reason="missing_required_inputs",
            missing_inputs=missing,
            device=device,
            gas=gas,
            circuit=circuit,
        )

    anode_radius_m = float(device["anode_radius_m"])
    cathode_radius_m = float(device["cathode_radius_m"])
    radial_gap_m = cathode_radius_m - anode_radius_m
    if radial_gap_m <= 0.0:
        return _blocked_packet(
            metadata=metadata,
            status="candidate_civ_paschen_breakdown_audit_unavailable",
            reason="nonpositive_radial_gap",
            missing_inputs=(),
            device=device,
            gas=gas,
            circuit=circuit,
        )

    insulator_length_m = _positive_or_none(device.get("insulator_length_m"))
    paschen_path_length_m = (
        insulator_length_m if insulator_length_m is not None else radial_gap_m
    )
    paschen_path_policy = (
        "source_declared_insulator_length"
        if insulator_length_m is not None
        else "radial_gap_fallback_because_insulator_length_missing_or_zero"
    )
    gas_species = _effective_gas_species(gas.get("species", "D2"))
    initial_current_A = _float_or_none(circuit.get("initial_current_A"))
    B_seed_T = _initial_magnetic_field_T(startup)

    try:
        breakdown = compute_breakdown(
            V0=float(circuit["voltage_V"]),
            fill_pressure_Pa=float(gas["pressure_Pa"]),
            anode_radius=anode_radius_m,
            cathode_radius=cathode_radius_m,
            insulator_length=paschen_path_length_m,
            gas_name=gas_species,
            B_seed=B_seed_T,
            I_seed=initial_current_A if _positive_or_none(initial_current_A) else None,
        )
        liftoff_delay_s = compute_liftoff_delay(breakdown)
        sheath_state = compute_initial_sheath_state(
            breakdown=breakdown,
            anode_radius=anode_radius_m,
            cathode_radius=cathode_radius_m,
            fill_pressure_Pa=float(gas["pressure_Pa"]),
        )
    except Exception as exc:  # pragma: no cover - defensive packetization.
        return _blocked_packet(
            metadata=metadata,
            status="candidate_civ_paschen_breakdown_audit_failed",
            reason=f"{type(exc).__name__}: {exc}",
            missing_inputs=(),
            device=device,
            gas=gas,
            circuit=circuit,
        )

    return {
        "status": "candidate_civ_paschen_breakdown_audit_engineering_only",
        "model_role": metadata["model_role"],
        "source_status": metadata["source_status"],
        "validation_status": metadata["validation_status"],
        "can_support_validation_claims": False,
        "can_support_whole_shot_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "decision": "do_not_promote_civ_paschen_audit_to_startup_bvp",
        "claim_limit": metadata["validity_notes"]["claim_limit"],
        "source_references": list(STARTUP_BREAKDOWN_AUDIT_SOURCE_REFS),
        "input_summary": {
            "device_name": device.get("device_name", "not_declared"),
            "anode_radius_m": anode_radius_m,
            "cathode_radius_m": cathode_radius_m,
            "radial_gap_m": radial_gap_m,
            "insulator_length_m": device.get("insulator_length_m"),
            "paschen_path_length_m": paschen_path_length_m,
            "paschen_path_policy": paschen_path_policy,
            "gas_species_requested": gas.get("species", "not_declared"),
            "gas_species_effective": gas_species,
            "fill_pressure_Pa": float(gas["pressure_Pa"]),
            "fill_temperature_K": _float_or_none(gas.get("temperature_K")),
            "bank_voltage_V": float(circuit["voltage_V"]),
            "initial_current_A": initial_current_A,
            "seed_magnetic_field_T": B_seed_T,
        },
        "breakdown": {
            "mechanism": breakdown.mechanism,
            "v_crit_m_s": breakdown.v_crit,
            "v_ExB_m_s": breakdown.v_ExB,
            "civ_ratio": breakdown.civ_ratio,
            "electric_field_V_m": breakdown.E_field,
            "seed_magnetic_field_T": breakdown.B_seed,
            "paschen_voltage_V": breakdown.paschen_voltage,
            "applied_voltage_V": breakdown.V_applied,
            "electron_mfp_m": breakdown.electron_mfp,
            "electron_larmor_radius_m": breakdown.larmor_radius_e,
            "electrons_magnetized": breakdown.is_magnetized,
            "breakdown_time_s": breakdown.breakdown_time,
            "initial_electron_temperature_K": breakdown.Te_initial,
            "initial_electron_temperature_eV": breakdown.Te_initial_eV,
            "initial_ionization_fraction": breakdown.ionization_fraction,
            "candidate_sheath_thickness_m": breakdown.sheath_thickness,
        },
        "liftoff": {
            "candidate_liftoff_delay_s": liftoff_delay_s,
            "source_status": "engineering_estimate_not_reviewed_startup_bvp",
            "can_support_handoff_acceptance": False,
        },
        "initial_sheath_state_candidate": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in sheath_state.items()
        },
        "required_promotion_path": (
            "replace_with_reviewed_surface_breakdown_bvp_or_imported_pic_"
            "field_particle_current_packet_for_same_device"
        ),
    }


def _blocked_packet(
    *,
    metadata: Mapping[str, Any],
    status: str,
    reason: str,
    missing_inputs: tuple[str, ...],
    device: Mapping[str, Any],
    gas: Mapping[str, Any],
    circuit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "model_role": metadata["model_role"],
        "source_status": metadata["source_status"],
        "validation_status": metadata["validation_status"],
        "can_support_validation_claims": False,
        "can_support_whole_shot_acceptance": False,
        "can_support_first_principles_acceptance": False,
        "decision": "do_not_promote_civ_paschen_audit_to_startup_bvp",
        "reason": reason,
        "missing_inputs": list(missing_inputs),
        "input_summary": {
            "device_name": device.get("device_name", "not_declared"),
            "gas_species_requested": gas.get("species", "not_declared"),
            "bank_voltage_V": circuit.get("voltage_V"),
        },
        "source_references": list(STARTUP_BREAKDOWN_AUDIT_SOURCE_REFS),
    }


def _missing_required_inputs(
    *,
    device: Mapping[str, Any],
    gas: Mapping[str, Any],
    circuit: Mapping[str, Any],
) -> tuple[str, ...]:
    required = (
        ("device.anode_radius_m", device.get("anode_radius_m")),
        ("device.cathode_radius_m", device.get("cathode_radius_m")),
        ("gas.pressure_Pa", gas.get("pressure_Pa")),
        ("circuit.voltage_V", circuit.get("voltage_V")),
    )
    return tuple(name for name, value in required if _positive_or_none(value) is None)


def _effective_gas_species(value: Any) -> str:
    species = str(value).strip()
    aliases = {
        "D": "D2",
        "D+": "D2",
        "DEUTERIUM": "D2",
        "H": "H2",
        "H+": "H2",
        "HYDROGEN": "H2",
    }
    return aliases.get(species.upper(), species or "D2")


def _initial_magnetic_field_T(startup: Mapping[str, Any] | None) -> float | None:
    if startup is None:
        return None
    field = startup.get("initial_magnetic_field_T")
    if isinstance(field, (tuple, list)) and len(field) >= 3:
        return _positive_or_none(field[2])
    return _positive_or_none(field)


def _positive_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed <= 0.0:
        return None
    return parsed


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
