"""Machine-readable v1 export bridge scope decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExportScopeDecision(BaseModel):
    """One v1 export bridge scope decision."""

    format_id: str
    display_name: str
    status: Literal["accepted", "deferred", "rejected"]
    rationale: str
    acceptance_evidence: list[str] = Field(default_factory=list)
    guardrail: str


def export_scope_decisions() -> list[ExportScopeDecision]:
    """Return the reviewed v1 export bridge decisions."""

    return [
        ExportScopeDecision(
            format_id="hdf5_diagnostics",
            display_name="DPF HDF5 diagnostics",
            status="accepted",
            rationale="Native diagnostics path with schema/time-base attributes, units, and run-manifest sidecar provenance.",
            acceptance_evidence=[
                "tests/test_export_scope.py::test_hdf5_diagnostics_export_carries_units_and_schema",
                "tests/test_validation_artifacts.py::test_engine_run_emits_preview_classification_and_sidecar_manifest",
            ],
            guardrail="HDF5 file creation alone is not scientific validation; use run manifest classification and readiness metadata.",
        ),
        ExportScopeDecision(
            format_id="well_hdf5",
            display_name="The Well HDF5 training format",
            status="accepted",
            rationale="Well exporter has schema/unit tests and the engine adapter now passes grid spacing, geometry, and simulation provenance metadata.",
            acceptance_evidence=[
                "tests/test_export_scope.py::test_engine_well_adapter_preserves_spacing_geometry_and_provenance",
                "tests/test_walrus_consolidated.py::test_finalize_field_units_stored_as_attrs",
            ],
            guardrail="Accepted for training-data interchange, not as validation evidence without source-gated labels.",
        ),
        ExportScopeDecision(
            format_id="vtk_vtu",
            display_name="VTK/VTU",
            status="deferred",
            rationale="Reader/import utilities exist for backend output, but v1 export correctness is not covered by a writer smoke test.",
            guardrail="Do not advertise VTK/VTU export as supported until writer, units, and readability tests exist.",
        ),
        ExportScopeDecision(
            format_id="cgns_hdf5",
            display_name="CGNS/HDF5",
            status="deferred",
            rationale="No SRS-grade CGNS writer or external readability test is present.",
            guardrail="Do not imply CGNS compatibility from generic HDF5 support.",
        ),
        ExportScopeDecision(
            format_id="openfoam",
            display_name="OpenFOAM",
            status="deferred",
            rationale="No OpenFOAM export writer, mesh mapping, units test, or external smoke test is present.",
            guardrail="Do not list OpenFOAM as a v1 accepted export bridge.",
        ),
        ExportScopeDecision(
            format_id="ansys_pymapdl",
            display_name="Ansys/PyMAPDL",
            status="deferred",
            rationale="No Ansys/PyMAPDL export bridge, license-aware test path, or external smoke test is present.",
            guardrail="Do not promise Ansys/PyMAPDL support until legal/tooling constraints and tests are resolved.",
        ),
    ]


def accepted_export_formats() -> list[str]:
    """Return accepted v1 export format IDs."""

    return [
        decision.format_id
        for decision in export_scope_decisions()
        if decision.status == "accepted"
    ]
