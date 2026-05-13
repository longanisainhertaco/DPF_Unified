"""Fail-closed evidence manifest for diagnostics surfaces.

This manifest is a claim-control surface. It classifies diagnostic outputs by
current evidence lane, but it does not validate the underlying physics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

EvidenceClass = Literal[
    "accepted",
    "blocked-by-review",
    "missing",
    "engineering-probe",
    "synthetic-only",
]


@dataclass(frozen=True)
class DiagnosticEvidenceEntry:
    """Evidence classification for one diagnostics module/output group."""

    entry_id: str
    module: str
    output: str
    symbols: tuple[str, ...]
    evidence_class: EvidenceClass
    source_status: str
    validation_status: str = "not_validation_evidence"
    can_support_validation_claims: bool = False
    blockers: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict for API/export callers."""
        return asdict(self)


_MANIFEST: tuple[DiagnosticEvidenceEntry, ...] = (
    DiagnosticEvidenceEntry(
        entry_id="DIA-BEAM-TARGET",
        module="beam_target",
        output="DD beam-target yield, disruption, anisotropy, and event split helpers",
        symbols=(
            "dd_cross_section",
            "dd_cross_section_array",
            "beam_target_yield_lee_saw",
            "beam_target_yield_rate",
            "detect_pinch_disruption",
            "neutron_anisotropy",
            "decompose_neutron_events",
        ),
        evidence_class="blocked-by-review",
        source_status="partial_local_lee_saw_formula_support_needs_same_scope_packet",
        blockers=(
            "same_scope_beam_energy_and_dwell_evidence_missing",
            "anisotropy_and_detector_response_packet_missing",
        ),
        notes="Lee/Saw-form pieces stay non-validation until tied to accepted same-scope neutron evidence.",
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-BEAM-TRACKER",
        module="beam_tracker",
        output="beam particle tracking summary and kinetic beam-target yield estimate",
        symbols=("BeamTrackerResult", "BeamTracker"),
        evidence_class="engineering-probe",
        source_status="engineering_particle_tracker_not_same_scope_validation",
        blockers=("accepted_beam_generation_and_target_coupling_packet_missing",),
        notes="Result metadata labels yield as engineering_estimate_not_validation.",
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-CHECKPOINT",
        module="checkpoint",
        output="checkpoint save/load state persistence",
        symbols=("save_checkpoint", "load_checkpoint"),
        evidence_class="engineering-probe",
        source_status="io_state_bookkeeping_not_physics_validation",
        blockers=("checkpoint_replay_validation_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-DERIVED",
        module="derived",
        output="derived MHD scalars and Bennett-radius helper",
        symbols=(
            "current_density_magnitude",
            "plasma_beta",
            "mach_number",
            "alfven_speed",
            "fast_magnetosonic_speed",
            "bennett_radius",
        ),
        evidence_class="engineering-probe",
        source_status="formula_helpers_need_per_output_kr_closure",
        blockers=("per_formula_source_line_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-ENERGY-BALANCE",
        module="energy_balance",
        output="energy snapshots, reports, and accounting tracker",
        symbols=("EnergySnapshot", "EnergyReport", "EnergyTracker"),
        evidence_class="engineering-probe",
        source_status="state_accounting_diagnostic_not_experimental_validation",
        blockers=("circuit_coupled_energy_validation_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-FILAMENTATION",
        module="filamentation",
        output="filament detection result and helper",
        symbols=("FilamentResult", "detect_filaments"),
        evidence_class="missing",
        source_status="local_formula_source_closure_missing",
        blockers=("filamentation_diagnostic_source_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-HDF5",
        module="hdf5_writer",
        output="HDF5 scalar/field diagnostics including rough max_div_B",
        symbols=("HDF5Writer",),
        evidence_class="engineering-probe",
        source_status="export_bookkeeping_with_rough_array_metric_labels",
        blockers=("geometry_aware_divergence_diagnostic_missing",),
        notes="max_div_B is labeled rough_array_metric_not_physical_divergence.",
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-INSTABILITY",
        module="instability",
        output="m=0, tearing, state-growth, and plasmoid instability helpers",
        symbols=(
            "m0_growth_rate",
            "tearing_mode_growth_rate",
            "m0_growth_rate_from_state",
            "detect_plasmoids",
        ),
        evidence_class="missing",
        source_status="local_instability_formula_source_closure_missing",
        blockers=("instability_source_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-INTERFEROMETRY",
        module="interferometry",
        output="Abel transform/inversion and optical phase/fringe synthetic helpers",
        symbols=("abel_transform", "abel_inversion", "phase_shift", "fringe_shift"),
        evidence_class="synthetic-only",
        source_status="synthetic_optical_diagnostic_without_detector_validation_packet",
        blockers=("same_scope_interferometry_response_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-NEUTRON-TOF",
        module="neutron_tof",
        output="thermonuclear, beam-target, and combined neutron TOF spectra",
        symbols=("thermonuclear_spectrum", "beam_target_spectrum", "combined_tof_spectrum"),
        evidence_class="synthetic-only",
        source_status="synthetic_spectrum_without_same_scope_detector_response",
        blockers=("neutron_spectrum_and_detector_response_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-NEUTRON-YIELD",
        module="neutron_yield",
        output="DD reactivity and thermonuclear neutron-yield integration helpers",
        symbols=(
            "dd_reactivity",
            "dd_reactivity_array",
            "neutron_yield_rate",
            "integrate_neutron_yield",
        ),
        evidence_class="blocked-by-review",
        source_status="component_reactivity_supported_but_total_dpf_yield_unvalidated",
        blockers=(
            "same_scope_density_temperature_volume_packet_missing",
            "mechanism_separated_neutron_validation_packet_missing",
        ),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-PB11",
        module="pb11_yield",
        output="p-B11 reactivity, yield, alpha-power, and metadata helpers",
        symbols=(
            "pb11_model_metadata",
            "pb11_reactivity",
            "pb11_reactivity_array",
            "pb11_yield_rate",
            "pb11_alpha_power_density",
        ),
        evidence_class="missing",
        source_status="pb11_reactivity_table_source_missing_from_verified_local_corpus",
        blockers=("pb11_reactivity_tables_and_dpf_feasibility_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-PEASE-BRAGINSKII",
        module="pease_braginskii",
        output="Pease-Braginskii current and threshold check",
        symbols=("pease_braginskii_current", "check_pease_braginskii"),
        evidence_class="engineering-probe",
        source_status="formula_helper_needs_claim_scope_and_kr_line_closure",
        blockers=("pease_braginskii_scope_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-PIC-YIELD",
        module="pic_yield",
        output="PIC particle-sampled neutron-yield rate estimate",
        symbols=("pic_neutron_yield_rate",),
        evidence_class="engineering-probe",
        source_status="particle_sampling_estimate_without_validated_beam_distribution",
        blockers=("accepted_pic_distribution_and_detector_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-PLASMA-REGIME",
        module="plasma_regime",
        output="plasma parameter and regime validity helpers",
        symbols=(
            "plasma_parameter_ND",
            "magnetic_reynolds_number",
            "debye_length",
            "ion_skin_depth",
            "regime_validity",
        ),
        evidence_class="missing",
        source_status="regime_formula_source_closure_missing",
        blockers=("regime_validity_source_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-PLASMOID",
        module="plasmoid",
        output="flux function, critical points, plasmoid detection, and force-free diagnostics",
        symbols=(
            "PlasmoidResult",
            "ForceFreeDiag",
            "compute_flux_function",
            "find_critical_points",
            "detect_plasmoids",
            "force_free_diagnostic",
        ),
        evidence_class="missing",
        source_status="plasmoid_formula_and_detector_source_closure_missing",
        blockers=("same_scope_plasmoid_diagnostic_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-REGIME-CLASSIFIER",
        module="regime_classifier",
        output="heuristic plasma-regime classifier",
        symbols=("RegimeResult", "classify_regime"),
        evidence_class="engineering-probe",
        source_status="heuristic_classifier_not_source_backed_validation",
        blockers=("classifier_training_or_source_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-RUNAWAY-ELECTRONS",
        module="runaway_electrons",
        output="Dreicer field, runaway fraction, and hard-xray power helpers",
        symbols=("dreicer_field", "runaway_fraction", "hard_xray_power"),
        evidence_class="missing",
        source_status="runaway_formula_source_closure_missing",
        blockers=("runaway_electron_and_hard_xray_source_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-SCALING-LAWS",
        module="scaling_laws",
        output="empirical scaling estimates and narrative",
        symbols=("ScalingResult", "compute_scaling", "scaling_narrative"),
        evidence_class="engineering-probe",
        source_status="empirical_scaling_context_not_solver_validation",
        blockers=("same_scope_scaling_validation_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-SHEAR-STABILIZATION",
        module="shear_stabilization",
        output="shear stabilization margin helper",
        symbols=("compute_shear_margin",),
        evidence_class="missing",
        source_status="shear_formula_source_closure_missing",
        blockers=("shear_stabilization_source_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-THOMSON",
        module="thomson_scattering",
        output="Thomson scattering spectra, line integration, and fit helper",
        symbols=(
            "spectral_density_salpeter",
            "thomson_spectrum",
            "thomson_line_integrated",
            "fit_te_ne_v",
        ),
        evidence_class="synthetic-only",
        source_status="synthetic_thomson_diagnostic_without_same_scope_calibration",
        blockers=("thomson_scattering_source_and_detector_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-XRAY",
        module="xray_imaging",
        output="x-ray emissivity, filtered image, geometry, and B-dot synthetic helpers",
        symbols=(
            "bremsstrahlung_emissivity",
            "filtered_emissivity",
            "synthetic_xray_image",
            "radiating_pinch_geometry_from_image",
            "synthetic_bdot_probe",
        ),
        evidence_class="synthetic-only",
        source_status="synthetic_xray_and_bdot_outputs_without_detector_validation",
        blockers=("xray_filter_emissivity_and_detector_packet_missing",),
    ),
    DiagnosticEvidenceEntry(
        entry_id="DIA-YIELD-TRACKER",
        module="yield_tracker",
        output="mechanism-separated neutron-yield history and summary",
        symbols=("YieldTimepoint", "YieldResult", "YieldTracker"),
        evidence_class="engineering-probe",
        source_status="component_labeled_yield_summary_not_total_validation",
        blockers=("same_scope_neutron_yield_timing_spectrum_detector_uq_packet_missing",),
    ),
)


def diagnostics_evidence_entries() -> tuple[DiagnosticEvidenceEntry, ...]:
    """Return immutable manifest entries for internal callers/tests."""
    return _MANIFEST


def diagnostics_evidence_manifest() -> list[dict[str, object]]:
    """Return diagnostics evidence entries as serializable dictionaries."""
    return [entry.to_dict() for entry in _MANIFEST]


def diagnostics_evidence_by_module() -> dict[str, list[dict[str, object]]]:
    """Group manifest entries by diagnostics module name."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in _MANIFEST:
        grouped.setdefault(entry.module, []).append(entry.to_dict())
    return grouped


def diagnostics_manifest_status_counts() -> dict[str, int]:
    """Count manifest entries by evidence class."""
    counts: dict[str, int] = {}
    for entry in _MANIFEST:
        counts[entry.evidence_class] = counts.get(entry.evidence_class, 0) + 1
    return counts


def diagnostics_manifest_entry(entry_id: str) -> dict[str, object]:
    """Return one manifest entry by ID or raise ``KeyError``."""
    for entry in _MANIFEST:
        if entry.entry_id == entry_id:
            return entry.to_dict()
    raise KeyError(entry_id)
