"""Package-native first-principles DPF runner surfaces."""

from dpf.first_principles.certificate_gate import (
    build_first_principles_certificate_gate_packet,
)
from dpf.first_principles.checkpoint_restart import (
    build_experimental_checkpoint_restart_family_packet,
    build_experimental_checkpoint_restart_packet,
)
from dpf.first_principles.closure_packet import build_physics_closure_packet
from dpf.first_principles.comparator_uq import build_comparator_uq_packet
from dpf.first_principles.current_waveform_comparator import (
    build_engineering_current_waveform_comparator,
)
from dpf.first_principles.deck import (
    BoundaryPolicy,
    CircuitDeck,
    ClosurePolicy,
    DeviceGeometryDeck,
    DiagnosticPolicy,
    FirstPrinciplesInputDeck,
    GasDeck,
    GridDeck,
    SourceReference,
    StartupPolicy,
    ValidationTargetReference,
    compact_chinese_dpf_engineering_deck,
    deck_hash,
    gv_verified_engineering_deck,
    gv_verified_engineering_decks,
    ir_mpf_100_engineering_deck,
    load_first_principles_input_deck,
    may15_second_scope_engineering_decks,
    minimal_engineering_deck,
    pf1000_akel_16kv_engineering_deck,
    willenborg_hendricks_engineering_deck,
)
from dpf.first_principles.dimensionality import build_dimensionality_handoff_packet
from dpf.first_principles.experimental_numerics import (
    build_experimental_numerical_family_packet,
    build_experimental_numerical_runtime_audit_packet,
    build_experimental_reproducibility_packet,
)
from dpf.first_principles.experimental_shot import (
    build_experimental_whole_shot_packet,
    build_whole_shot_duration_plan,
    stable_ohmic_cfl_dt_s,
    stable_vacuum_cfl_dt_s,
)
from dpf.first_principles.generalization import build_generalized_dpf_machine_packet
from dpf.first_principles.gv_waveforms import (
    extract_all_gv_current_waveform_packets,
    extract_gv_current_waveform_packet,
    gv_waveform_packet_summary,
)
from dpf.first_principles.inverse_calibration import (
    build_experimental_inverse_calibration_packet,
    build_source_bounded_candidate_grid,
    build_source_bounded_candidate_grid_from_parameter_scales,
    classify_inverse_calibration_results,
    score_current_history_against_targets,
)
from dpf.first_principles.inverse_parameters import (
    bank_energy_J,
    build_experimental_inverse_parameter_packet,
    current_implied_inductance_H,
    ideal_lc_peak_current_A,
    ideal_lc_quarter_cycle_s,
    quarter_cycle_implied_inductance_H,
)
from dpf.first_principles.limiter_proof import (
    build_experimental_limiter_zero_probe_packet,
)
from dpf.first_principles.limiter_readiness import build_limiter_readiness_packet
from dpf.first_principles.neutron_authority import (
    build_mechanism_separated_neutron_packet,
)
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet
from dpf.first_principles.plasmapy_audit import build_plasmapy_formulary_audit_packet
from dpf.first_principles.power_port import build_engineering_power_port_packet
from dpf.first_principles.same_scope import build_same_scope_source_packet
from dpf.first_principles.source_targets import (
    gv_verified_shot_targets,
    may15_user_validated_source_targets,
    may16_validated_thesis_source_targets,
)
from dpf.first_principles.spatial_field_temperature import (
    build_spatial_field_temperature_packet,
)
from dpf.first_principles.startup_breakdown import (
    build_candidate_startup_breakdown_audit,
)
from dpf.first_principles.startup_bvp import build_startup_bvp_packet
from dpf.first_principles.state_checkpoint import (
    write_terminal_state_checkpoint_roundtrip,
)
from dpf.first_principles.waveform_phase import build_waveform_phase_packet

try:
    from dpf.first_principles.runner import (
        FirstPrinciples3DSession,
        build_first_principles_3d_session,
        run_first_principles_3d_deck,
    )
    from dpf.first_principles.runner import (
        HybridEMPicFluidRunResult as FirstPrinciplesRunResult,
    )
except ImportError:  # pragma: no cover - runner may be unavailable during partial imports.
    FirstPrinciples3DSession = None  # type: ignore[assignment]
    FirstPrinciplesRunResult = None  # type: ignore[assignment]
    build_first_principles_3d_session = None  # type: ignore[assignment]
    run_first_principles_3d_deck = None  # type: ignore[assignment]

__all__ = [
    "CircuitDeck",
    "BoundaryPolicy",
    "ClosurePolicy",
    "DeviceGeometryDeck",
    "DiagnosticPolicy",
    "FirstPrinciplesInputDeck",
    "FirstPrinciplesRunResult",
    "FirstPrinciples3DSession",
    "GasDeck",
    "GridDeck",
    "SourceReference",
    "StartupPolicy",
    "ValidationTargetReference",
    "compact_chinese_dpf_engineering_deck",
    "deck_hash",
    "gv_verified_engineering_deck",
    "gv_verified_engineering_decks",
    "ir_mpf_100_engineering_deck",
    "load_first_principles_input_deck",
    "may15_second_scope_engineering_decks",
    "minimal_engineering_deck",
    "pf1000_akel_16kv_engineering_deck",
    "willenborg_hendricks_engineering_deck",
    "build_first_principles_certificate_gate_packet",
    "build_experimental_checkpoint_restart_packet",
    "build_experimental_checkpoint_restart_family_packet",
    "build_first_principles_3d_session",
    "build_physics_closure_packet",
    "build_comparator_uq_packet",
    "build_engineering_current_waveform_comparator",
    "build_dimensionality_handoff_packet",
    "build_engineering_power_port_packet",
    "build_experimental_numerical_family_packet",
    "build_experimental_numerical_runtime_audit_packet",
    "build_experimental_reproducibility_packet",
    "build_experimental_whole_shot_packet",
    "build_whole_shot_duration_plan",
    "stable_ohmic_cfl_dt_s",
    "build_generalized_dpf_machine_packet",
    "extract_all_gv_current_waveform_packets",
    "extract_gv_current_waveform_packet",
    "gv_waveform_packet_summary",
    "bank_energy_J",
    "build_experimental_inverse_parameter_packet",
    "current_implied_inductance_H",
    "ideal_lc_peak_current_A",
    "ideal_lc_quarter_cycle_s",
    "quarter_cycle_implied_inductance_H",
    "build_experimental_inverse_calibration_packet",
    "build_source_bounded_candidate_grid",
    "build_source_bounded_candidate_grid_from_parameter_scales",
    "classify_inverse_calibration_results",
    "score_current_history_against_targets",
    "build_limiter_readiness_packet",
    "build_experimental_limiter_zero_probe_packet",
    "build_mechanism_separated_neutron_packet",
    "build_numerical_fidelity_packet",
    "build_plasmapy_formulary_audit_packet",
    "build_same_scope_source_packet",
    "build_spatial_field_temperature_packet",
    "write_terminal_state_checkpoint_roundtrip",
    "build_candidate_startup_breakdown_audit",
    "gv_verified_shot_targets",
    "may15_user_validated_source_targets",
    "may16_validated_thesis_source_targets",
    "build_startup_bvp_packet",
    "build_waveform_phase_packet",
    "stable_vacuum_cfl_dt_s",
    "run_first_principles_3d_deck",
]
