# DPF-Unified v1 Export Scope

Status: candidate v1 scope decision
Date: 2026-05-08

## Accepted

| Format | Scope | Acceptance evidence | Guardrail |
| --- | --- | --- | --- |
| DPF HDF5 diagnostics | Native scalar and field diagnostics with schema/time-base attributes, dataset units, and run-manifest sidecar provenance. | `tests/test_export_scope.py`; `tests/test_validation_artifacts.py`. | File readability is not scientific validation; use result classification and readiness metadata. |
| The Well HDF5 | Training-data interchange path with field units and engine adapter metadata for grid spacing, geometry, and simulation provenance. | `tests/test_export_scope.py`; `tests/test_walrus_consolidated.py`. | Accepted for training-data interchange only; validation claims still require source-gated evidence. |

## Classification Propagation

Accepted v1 export paths are fail-closed:

- HDF5 diagnostics are accepted only with schema/time-base/unit metadata plus
  run-manifest sidecar provenance and result classification.
- Well HDF5 is accepted only as a training-data interchange adapter. Engine
  exports now flush without manual `engine.close()`, forward circuit scalars,
  preserve cylindrical `grid_type`, and carry fail-closed validation/result
  labels through adapter simulation metadata.
- Deferred bridge formats remain outside v1.0 until writer/readability tests
  and non-manifest classification propagation are designed.

## Deferred

| Format | Reason | Guardrail |
| --- | --- | --- |
| VTK/VTU | Backend reader/import utilities exist, but no v1 writer/readability acceptance test exists. | Do not advertise as supported export until writer, units, and smoke tests exist. |
| CGNS/HDF5 | No SRS-grade CGNS writer or external readability test exists. | Do not imply CGNS compatibility from generic HDF5 support. |
| OpenFOAM | No OpenFOAM writer, mesh mapping, units test, or external smoke test exists. | Do not list as accepted v1 bridge. |
| Ansys/PyMAPDL | No bridge, license-aware test path, or external smoke test exists. | Do not promise support until legal/tooling constraints and tests are resolved. |

The machine-readable mirror of this decision is `src/dpf/io/export_scope.py`.
