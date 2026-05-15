"""Package-native first-principles DPF runner surfaces."""

from dpf.first_principles.deck import (
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
    deck_hash,
    load_first_principles_input_deck,
    minimal_engineering_deck,
    pf1000_akel_16kv_engineering_deck,
)
from dpf.first_principles.certificate_gate import (
    build_first_principles_certificate_gate_packet,
)
from dpf.first_principles.closure_packet import build_physics_closure_packet
from dpf.first_principles.comparator_uq import build_comparator_uq_packet
from dpf.first_principles.dimensionality import build_dimensionality_handoff_packet
from dpf.first_principles.generalization import build_generalized_dpf_machine_packet
from dpf.first_principles.limiter_readiness import build_limiter_readiness_packet
from dpf.first_principles.neutron_authority import (
    build_mechanism_separated_neutron_packet,
)
from dpf.first_principles.numerical_fidelity import build_numerical_fidelity_packet
from dpf.first_principles.power_port import build_engineering_power_port_packet
from dpf.first_principles.same_scope import build_same_scope_source_packet
from dpf.first_principles.spatial_field_temperature import (
    build_spatial_field_temperature_packet,
)
from dpf.first_principles.startup_bvp import build_startup_bvp_packet
from dpf.first_principles.waveform_phase import build_waveform_phase_packet

try:
    from dpf.first_principles.runner import (
        HybridEMPicFluidRunResult as FirstPrinciplesRunResult,
    )
    from dpf.first_principles.runner import (
        run_first_principles_3d_deck,
    )
except ImportError:  # pragma: no cover - runner may be unavailable during partial imports.
    FirstPrinciplesRunResult = None  # type: ignore[assignment]
    run_first_principles_3d_deck = None  # type: ignore[assignment]

__all__ = [
    "CircuitDeck",
    "ClosurePolicy",
    "DeviceGeometryDeck",
    "DiagnosticPolicy",
    "FirstPrinciplesInputDeck",
    "FirstPrinciplesRunResult",
    "GasDeck",
    "GridDeck",
    "SourceReference",
    "StartupPolicy",
    "ValidationTargetReference",
    "deck_hash",
    "load_first_principles_input_deck",
    "minimal_engineering_deck",
    "pf1000_akel_16kv_engineering_deck",
    "build_first_principles_certificate_gate_packet",
    "build_physics_closure_packet",
    "build_comparator_uq_packet",
    "build_dimensionality_handoff_packet",
    "build_engineering_power_port_packet",
    "build_generalized_dpf_machine_packet",
    "build_limiter_readiness_packet",
    "build_mechanism_separated_neutron_packet",
    "build_numerical_fidelity_packet",
    "build_same_scope_source_packet",
    "build_spatial_field_temperature_packet",
    "build_startup_bvp_packet",
    "build_waveform_phase_packet",
    "run_first_principles_3d_deck",
]
