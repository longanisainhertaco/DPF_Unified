"""Diagnostics test-lane manifest.

This module classifies diagnostics-oriented pytest files by what their passing
status can claim. It prevents engineering, synthetic, and source-component
checks from being mistaken for source-backed validation tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DiagnosticTestLane = Literal[
    "engineering-smoke",
    "synthetic-only",
    "source-component-check",
    "source-blocked",
    "source-backed-validation",
]


@dataclass(frozen=True)
class DiagnosticTestLaneEntry:
    """Classification for one diagnostics-oriented pytest file."""

    test_file: str
    lane: DiagnosticTestLane
    markers: tuple[str, ...]
    source_status: str
    validation_status: str = "not_validation_evidence"
    can_support_validation_claims: bool = False
    blockers: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict for reports/tests."""
        return asdict(self)


_TEST_LANES: tuple[DiagnosticTestLaneEntry, ...] = (
    DiagnosticTestLaneEntry(
        test_file="test_beam_tracker.py",
        lane="engineering-smoke",
        markers=("diagnostics_engineering", "diagnostics_source_blocked"),
        source_status="beam_tracker_estimate_not_same_scope_validation",
        blockers=("accepted_beam_target_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_diagnostics_evidence_manifest.py",
        lane="engineering-smoke",
        markers=("diagnostics_engineering",),
        source_status="claim_control_manifest_regression",
        blockers=("not_a_physics_validation_test",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_diagnostics_test_lanes.py",
        lane="engineering-smoke",
        markers=("diagnostics_engineering",),
        source_status="test_lane_manifest_regression",
        blockers=("not_a_physics_validation_test",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_energy_balance.py",
        lane="engineering-smoke",
        markers=("diagnostics_engineering",),
        source_status="state_accounting_regression_not_experimental_validation",
        blockers=("circuit_coupled_energy_validation_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_filamentation.py",
        lane="source-blocked",
        markers=("diagnostics_engineering", "diagnostics_source_blocked"),
        source_status="filamentation_formula_source_packet_missing",
        blockers=("local_filamentation_source_closure_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_neutron_yield.py",
        lane="source-component-check",
        markers=("diagnostics_source_component", "diagnostics_source_blocked"),
        source_status="component_reactivity_checks_not_total_dpf_validation",
        blockers=("same_scope_neutron_validation_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_pb11_yield.py",
        lane="source-blocked",
        markers=("diagnostics_engineering", "diagnostics_source_blocked"),
        source_status="pb11_reactivity_tables_missing_from_verified_local_corpus",
        blockers=("pb11_source_tables_and_dpf_feasibility_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_plasmoid.py",
        lane="source-blocked",
        markers=("diagnostics_engineering", "diagnostics_source_blocked"),
        source_status="plasmoid_formula_source_packet_missing",
        blockers=("same_scope_plasmoid_diagnostic_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_regime_classifier.py",
        lane="source-blocked",
        markers=("diagnostics_engineering", "diagnostics_source_blocked"),
        source_status="regime_classifier_source_or_training_packet_missing",
        blockers=("regime_classifier_source_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_scaling_laws.py",
        lane="engineering-smoke",
        markers=("diagnostics_engineering", "diagnostics_source_blocked"),
        source_status="empirical_scaling_regression_not_solver_validation",
        blockers=("same_scope_scaling_validation_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_synthetic_diagnostics.py",
        lane="synthetic-only",
        markers=("diagnostics_synthetic",),
        source_status="synthetic_outputs_without_detector_response_validation",
        blockers=("same_scope_detector_response_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_thomson_api.py",
        lane="synthetic-only",
        markers=("diagnostics_synthetic", "diagnostics_source_blocked"),
        source_status="synthetic_thomson_api_without_calibrated_diagnostic_packet",
        blockers=("thomson_source_and_detector_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_thomson_scattering.py",
        lane="synthetic-only",
        markers=("diagnostics_synthetic", "diagnostics_source_blocked"),
        source_status="synthetic_thomson_formulas_without_same_scope_calibration",
        blockers=("thomson_source_and_detector_packet_missing",),
    ),
    DiagnosticTestLaneEntry(
        test_file="test_yield_tracker.py",
        lane="engineering-smoke",
        markers=("diagnostics_engineering", "diagnostics_source_blocked"),
        source_status="mechanism_labeled_yield_summary_not_total_validation",
        blockers=("same_scope_neutron_yield_packet_missing",),
    ),
)


def diagnostics_test_lane_entries() -> tuple[DiagnosticTestLaneEntry, ...]:
    """Return immutable diagnostics test-lane entries."""
    return _TEST_LANES


def diagnostics_test_lane_manifest() -> list[dict[str, object]]:
    """Return the diagnostics test-lane manifest as plain dictionaries."""
    return [entry.to_dict() for entry in _TEST_LANES]


def diagnostics_test_lane_for_file(test_file: str) -> DiagnosticTestLaneEntry | None:
    """Return the lane entry for a pytest filename, if it is diagnostics-scoped."""
    for entry in _TEST_LANES:
        if entry.test_file == test_file:
            return entry
    return None


def diagnostics_test_lane_counts() -> dict[str, int]:
    """Count diagnostics-oriented test files by lane."""
    counts: dict[str, int] = {}
    for entry in _TEST_LANES:
        counts[entry.lane] = counts.get(entry.lane, 0) + 1
    return counts
