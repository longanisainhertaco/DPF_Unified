# IO/Export Module Audit

Status date: 2026-05-09

This note is advisory. It is not source-of-truth science and must be re-checked
against current code and reviewed `KnowledgeReference/` evidence before any
implementation work.

## Scope

Primary files reviewed:

- `src/dpf/io/`
- `src/dpf/diagnostics/hdf5_writer.py`
- export wiring in `src/dpf/engine/core.py`
- validation artifact helpers
- export scope, SRS traceability, project lifecycle, airgap, and WALRUS tests

## Intended Behavior

The v1 export scope is intentionally narrow. Accepted formats are:

- DPF HDF5 diagnostics
- The Well HDF5 training-data format

VTK/VTU, CGNS, OpenFOAM, and Ansys/PyMAPDL are deferred.

HDF5 diagnostics should provide schema/versioned scalar and field output with
units, timebase, and run-manifest sidecar provenance.

Run artifacts should carry Preview/source-gated classification and link into
manifests/certificates rather than implying validation from file creation.

## Source-Of-Truth Support Found Or Missing

Supported locally:

- Local KR material supports general HDF5 simulation dumps/history/visualization
  as an engineering output pattern.

Missing or not yet strong enough:

- No local KR authority was found for this project claiming The Well schema,
  PolymathicAI compatibility, WALRUS compatibility, or DPF-specific validation
  meaning.
- The local Well spec cites external sources and is therefore suspect under the
  current source-of-truth rule until acquired/promoted.
- SRS traceability is staged, not fully imported into Doorstop.

## Current Implementation Summary

`src/dpf/io/export_scope.py` mirrors the v1 export decision table and exposes
accepted IDs. It is a scope-control module, not a writer or validator.

`src/dpf/diagnostics/hdf5_writer.py` records scalar histories and optional field
snapshots, then writes root attributes including schema version, timebase units,
and record count.

`SimulationEngine` creates HDF5 and optional Well exporters, embeds artifact
metadata/manifest sidecars, and finalizes diagnostics on normal completion.

`src/dpf/io/well_exporter.py` is a buffered adapter around the AI Well exporter.
It buffers numpy-array state, infers grid shape from `rho`, and flushes to the
full exporter.

2026-05-09 implementation update:

- `SimulationEngine.run()` now flushes the buffered Well exporter on normal
  completion and attempts the same after run errors.
- The buffered Well adapter now forwards circuit scalars to the full Well
  exporter.
- The full AI Well exporter now writes cylindrical root metadata as
  `grid_type="cylindrical"` instead of always labeling files as Cartesian.
- The full AI Well exporter now writes owner/distribution artifact
  classification metadata as root attributes and JSON while preserving
  `validation_status="not_validation_evidence"` and `result_label="Preview"`.
  `dpf export-well` accepts matching artifact classification flags, and
  engine-flushed Well files keep the same fail-closed training-data labels.
- `SimulationConfig.diagnostics` now carries owner/classification/distribution
  artifact fields. The engine uses those fields for run manifests, embedded
  HDF5 governance metadata, and engine-flushed Well files.
- Batch-generated Well trajectories now use the same config-driven artifact
  classification metadata instead of emitting unclassified training files.
- Checkpoint/restart HDF5 files now carry fail-closed artifact role,
  validation status, Preview label, and config-derived classification metadata.
- Batch runs now write `dataset_manifest.json` with fail-closed labels, artifact
  classification metadata, file hashes when trajectory files exist, counts,
  parameter ranges, and a training-candidate guardrail. REST simulation creation
  preserves artifact classification fields supplied through the config payload.
- `docs/EXPORT_SCOPE_V1.md`, the SRS draft, and the candidate requirements
  baseline now agree that accepted HDF5/Well paths carry fail-closed
  classification/provenance labels, while deferred external bridges still need
  non-manifest classification propagation before acceptance.

## Why It Likely Fails Or Is Unverifiable

- `SimulationEngine.run()` now flushes Well output on normal completion and
  attempts to flush after run errors, and `dpf export-well` now forwards
  explicit artifact classification metadata. API/config-level classification
  propagation is covered through `SimulationConfig` payloads, and batch dataset
  manifests now carry the same fail-closed artifact metadata.
- Strict Well validation now covers scalar-history NaN/Inf checks, energy
  evidence, monotonic time, geometry/root consistency, provenance/classification
  attrs, sanitized-dataset rejection, saturation-scale detection, and all-zero
  magnetic-field rejection. This is still local integrity checking, not external
  Well/WALRUS compatibility proof.
- Existing local generated HDF5 training artifacts are not validation evidence.
  The WALRUS data review records non-finite current/voltage, zero magnetic
  field, missing manifests, and other defects.

## Stale Or Inaccurate Assumptions

- "Well schema-tested" means locally smoke-tested, not externally validated
  compatibility.
- The full AI exporter now labels cylindrical files as cylindrical, but external
  Well/WALRUS compatibility is still not locally source-closed.
- Well files now carry fail-closed artifact classification metadata, but that
  metadata is governance/provenance only. It is not evidence that The Well,
  WALRUS, or local generated training data are source-backed.
- Config-driven classification now reaches engine HDF5/Well/run-manifest
  artifacts, batch Well trajectories, checkpoint HDF5 files, and batch dataset
  manifests. Readiness summaries in validation certificates remain a validation
  artifact task, not an IO/export classification blocker.
- Baseline docs and the SRS draft now agree that accepted HDF5/Well paths carry
  fail-closed classification/provenance, while deferred bridges still need their
  own non-manifest classification propagation before acceptance.

## Trustworthy Tests Versus Suspect Tests

More trustworthy for narrow behavior:

- `tests/test_export_scope.py`
- `tests/test_validation_artifacts.py`
- `tests/test_srs_traceability_export.py`
- `tests/test_project_lifecycle.py`
- `tests/test_airgap_gate.py`

Limited or suspect:

- Well tests in `tests/test_walrus_consolidated.py` validate this repo's local
  schema assumptions, not KR-backed or external reader compatibility.
- Engine integration tests that manually close the engine do not prove CLI or
  normal-run exporter flushing.

## Future-Agent Notes

These notes are not authority. Re-check all line numbers and current behavior
before editing.

- Treat all non-`KnowledgeReference/` Well/WALRUS/The Well claims as hypotheses.
- Do not use ignored local `.h5` or `training_data/` files as validation
  artifacts without manifests, hashes, source authority, and strict validator
  results.
- File export success is not a scientific result. Keep result classification,
  provenance, and validation status attached.

## Backlog Links

See `BACKLOG.md` entries `IO-001` through `IO-008`.
