"""Fail-closed neutron mechanism-authority packets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NEUTRON_AUTHORITY_SOURCE_REFS = (
    {
        "path": "KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md",
        "lines": "120-131,190-215,282-288,862-889",
        "role": "pf1000_akel_neutron_detector_yield_and_lee_baseline_context",
    },
    {
        "path": (
            "KnowledgeReference/"
            "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
        ),
        "lines": "952-970,1037-1040,1083-1089,1214-1266",
        "role": "hybrid_pic_fluid_yield_history_and_limitations",
    },
    {
        "path": "KnowledgeReference/fully-kinetic-simulations-of-dense-plasma-focus-z-pinch.md",
        "lines": "34-43,68-78,126-161",
        "role": "fully_kinetic_mev_ion_and_beam_target_requirement",
    },
    {
        "path": (
            "KnowledgeReference/"
            "neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md"
        ),
        "lines": "39-44,409-418,433-445,551-613",
        "role": "mechanism_separation_timing_spectrum_anisotropy_requirement",
    },
    {
        "path": (
            "KnowledgeReference/"
            "tomographic-reconstruction-of-the-neutron-time-energy-spectrum-from-a-dense-plasma-focus-b78f1154.md"
        ),
        "lines": "32-53,337-351,390-427,518-526",
        "role": "tof_tomography_detector_response_and_scatter_subtraction_schema",
    },
    {
        "path": (
            "KnowledgeReference/"
            "anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md"
        ),
        "lines": "121-137,175-204,269-288",
        "role": "pf1000_anisotropy_detector_and_scattering_schema_other_scope",
    },
)

REQUIRED_NEUTRON_AUTHORITY_CHANNELS = (
    "accepted_thermonuclear_yield_history",
    "accepted_beam_target_yield_history",
    "mechanism_separated_yield_channels",
    "ion_energy_distribution_history",
    "beam_angular_distribution_history",
    "beam_transport_stopping_model",
    "target_density_path_length_history",
    "dd_cross_section_source_and_units",
    "neutron_timing_history",
    "neutron_energy_spectrum",
    "neutron_anisotropy_angular_yield",
    "detector_response_model",
    "activation_counter_response_model",
    "direct_scattered_neutron_transport",
    "same_scope_scalar_yield",
    "yield_uncertainty_budget",
    "electron_temperature_yield_sensitivity_uq",
    "output_mapping_and_comparator",
    "source_review_certificate",
)

PF1000_AKEL_TEXT_SUPPORTED_CHANNELS = (
    "scintillator_detector_layout_0_90_180_degrees",
    "time_of_flight_mean_neutron_deuteron_energy_method",
    "silver_activation_total_yield_measurement",
    "am_be_activation_calibration_text",
    "yield_uncertainty_scalar",
    "measured_scalar_yield_shot_12581",
    "lee_thermonuclear_and_beam_target_model_text",
    "lee_beam_target_formula_context",
    "current_derivative_t0_reference_for_neutron_timing",
    "average_yield_series_fit_context",
)

BLOCKING_NEUTRON_AUTHORITY_CHANNELS = (
    "accepted_thermonuclear_yield_history",
    "accepted_beam_target_yield_history",
    "mechanism_separated_yield_channels",
    "ion_energy_distribution_history",
    "beam_angular_distribution_history",
    "beam_transport_stopping_model",
    "target_density_path_length_history",
    "neutron_energy_spectrum",
    "neutron_anisotropy_angular_yield",
    "detector_response_model",
    "activation_counter_response_model",
    "direct_scattered_neutron_transport",
    "yield_uncertainty_budget",
    "electron_temperature_yield_sensitivity_uq",
    "output_mapping_and_comparator",
    "source_review_certificate",
)

OTHER_SCOPE_SOURCE_GROUPS = (
    {
        "name": "new_2026_axisymmetric_hybrid_pic_fluid",
        "scope_mismatch": (
            "2D axisymmetric compact/LLNL-like hybrid simulation, not "
            "PF-1000/Akel shot 12581."
        ),
        "usable_for": "resolved ion-distribution yield-history requirements",
    },
    {
        "name": "llnl_fully_kinetic_dpf",
        "scope_mismatch": "LLNL low-current DPF, not PF-1000/Akel shot 12581.",
        "usable_for": "requirement for kinetic MeV ions, beam formation, and beam-target yield",
    },
    {
        "name": "mjolnir_ma_class_mechanism_separation",
        "scope_mismatch": "MA/MJ-class MJOLNIR source, not PF-1000/Akel shot 12581.",
        "usable_for": "thermonuclear vs beam-target timing, spectrum, and anisotropy schema",
    },
    {
        "name": "tof_tomography_detector_response",
        "scope_mismatch": "NNSS deuterium DPF detector setup, not PF-1000/Akel shot 12581.",
        "usable_for": "time-energy spectrum inversion and detector/scatter-response schema",
    },
    {
        "name": "pf1000_full_energy_anisotropy",
        "scope_mismatch": "PF-1000 operated at 450-500 kJ and 3.5 Torr, not Akel 16 kV.",
        "usable_for": "anisotropy, direct/scattered neutron, and detector-response schema",
    },
)


def build_mechanism_separated_neutron_packet(
    *,
    declared_scope: str,
    device_name: str | None = None,
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    accepted_channels: tuple[str, ...] | list[str] = (),
    kinetic_yield: Mapping[str, Any] | None = None,
    same_scope_source: Mapping[str, Any] | None = None,
    physics_closure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a non-promoting neutron-yield authority packet."""

    accepted = {str(channel) for channel in accepted_channels}
    accepted.update(_accepted_channels_from_targets(validation_targets))
    text_supported = (
        set(PF1000_AKEL_TEXT_SUPPORTED_CHANNELS)
        if _looks_like_pf1000_akel_scope(declared_scope, device_name)
        else set()
    )

    missing = set(REQUIRED_NEUTRON_AUTHORITY_CHANNELS) - accepted
    missing.update(BLOCKING_NEUTRON_AUTHORITY_CHANNELS)

    return {
        "status": "blocked_mechanism_separated_neutron_authority_not_available",
        "declared_scope": declared_scope,
        "device_name": device_name or "not_declared",
        "decision": "do_not_enable_total_neutron_yield_authority",
        "required_channels": list(REQUIRED_NEUTRON_AUTHORITY_CHANNELS),
        "text_supported_reference_channels": sorted(text_supported),
        "candidate_runtime_channels": _candidate_runtime_channels(kinetic_yield),
        "accepted_channels": sorted(accepted),
        "missing_acceptance_channels": sorted(missing),
        "other_scope_source_groups": list(OTHER_SCOPE_SOURCE_GROUPS),
        "source_references": list(NEUTRON_AUTHORITY_SOURCE_REFS),
        "same_scope_source_status": (
            None if same_scope_source is None else same_scope_source.get("status")
        ),
        "physics_closure_status": (
            None if physics_closure is None else physics_closure.get("status")
        ),
        "beam_target_closure_status": _beam_target_closure_status(physics_closure),
        "kinetic_yield_status": (
            None if kinetic_yield is None else kinetic_yield.get("status")
        ),
        "kinetic_yield_mechanism_separation_status": (
            None
            if kinetic_yield is None
            else kinetic_yield.get("mechanism_separation_status")
        ),
        "validation_target_count": len(validation_targets),
        "can_support_total_yield_acceptance": False,
        "can_support_first_principles_acceptance": False,
    }


def _candidate_runtime_channels(kinetic_yield: Mapping[str, Any] | None) -> list[str]:
    if kinetic_yield is None:
        return []
    channels = ["candidate_pic_ion_neutron_yield_history"]
    for channel in kinetic_yield.get("mechanism_channels", ()) or ():
        channels.append(f"candidate_{channel}")
    return sorted(set(str(channel) for channel in channels))


def _accepted_channels_from_targets(
    validation_targets: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]
) -> set[str]:
    accepted: set[str] = set()
    aliases = {
        "thermonuclear_yield_history": "accepted_thermonuclear_yield_history",
        "beam_target_yield_history": "accepted_beam_target_yield_history",
        "mechanism_separated_yield": "mechanism_separated_yield_channels",
        "ion_energy_distribution": "ion_energy_distribution_history",
        "ion_distribution_history": "ion_energy_distribution_history",
        "beam_angular_distribution": "beam_angular_distribution_history",
        "neutron_timing": "neutron_timing_history",
        "neutron_spectrum": "neutron_energy_spectrum",
        "neutron_anisotropy": "neutron_anisotropy_angular_yield",
        "detector_response": "detector_response_model",
        "activation_response": "activation_counter_response_model",
        "direct_scattered_neutron_transport": "direct_scattered_neutron_transport",
        "neutron_scalar_yield": "same_scope_scalar_yield",
        "yield_uncertainty": "yield_uncertainty_budget",
    }
    for target in validation_targets:
        status = str(target.get("status", ""))
        if status not in {
            "accepted_same_scope_source",
            "reviewed_same_scope_source",
            "accepted",
        }:
            continue
        observable = str(target.get("observable", "")).strip()
        if observable in aliases:
            accepted.add(aliases[observable])
    return accepted


def _beam_target_closure_status(physics_closure: Mapping[str, Any] | None) -> str | None:
    if physics_closure is None:
        return None
    effects = physics_closure.get("effects")
    if not isinstance(effects, Mapping):
        return None
    beam_target = effects.get("beam_target_coupling")
    if not isinstance(beam_target, Mapping):
        return None
    return None if beam_target.get("status") is None else str(beam_target["status"])


def _looks_like_pf1000_akel_scope(
    declared_scope: str,
    device_name: str | None,
) -> bool:
    haystack = f"{declared_scope} {device_name or ''}".lower()
    return "pf1000" in haystack or "pf-1000" in haystack or "akel" in haystack
