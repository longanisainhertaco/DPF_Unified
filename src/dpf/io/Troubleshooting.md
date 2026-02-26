# IO Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 1 file (178 LOC)

---

## HIGH

### ✅ FIXED — HIGH-1: Dead Loop in _prepare_well_arrays
- **File:Line**: `well_exporter.py:125-132`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Fix**: Removed dead loop and unused `dpf_scalar_to_well` import + dead `n_steps`/`n_traj`/`traj_idx` variables. `np.stack` path is now the only code path.
- **Impact**: Wasted computation on every flush. Dead code obscures intent.

### HIGH-2: Duplicate of ai/well_exporter.py
- **File:Line**: Entire module
- **Found by**: py-ai-diag
- **Cross-review verdict**: MODIFIED (see `src/dpf/ai/Troubleshooting.md` CRIT-3)
- **Description**: This module IS used by `engine.py:214` (production code path). The `ai/well_exporter.py` is used by the CLI. Two separate production paths with different HDF5 formats.
- **Impact**: Format inconsistency between engine-generated and CLI-generated Well exports.

---

## REJECTED FINDINGS

### REJECTED: "Never used by any production code path"
- **Reason**: `engine.py:214` imports `from dpf.io.well_exporter import WellExporter`. This IS a production path. Finding upgraded to HIGH-2 with corrected assessment.
