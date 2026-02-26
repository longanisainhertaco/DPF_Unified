# CLI Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 1 file (401 LOC)

---

## No critical issues found.

---

## LOW

### LOW-1: Private Attribute Access on WellExporter ✅ FIXED
- **File:Line**: `main.py:251`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED (minor)
- **Description**: `len(exporter._times)` accesses a private attribute from outside the class.
- **Fix applied**: Added `n_snapshots` property to `WellExporter` (`well_exporter.py:72`); updated `main.py:251` to use `exporter.n_snapshots`.
- **Impact**: Coupling to internal implementation detail. Minor maintenance concern.

---

## REJECTED FINDINGS

None.
