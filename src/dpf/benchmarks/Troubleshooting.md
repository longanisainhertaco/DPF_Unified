# Benchmarks Domain — Cross-Review Troubleshooting Guide

**Cross-reviewer**: xreview-metal
**Source reviews**: py-metal (Python expert), phys-metal (Physics expert)
**Files reviewed**: metal_benchmark.py (827 lines), apple_silicon.py (270 lines)
**Date**: 2026-02-25

---

## CRITICAL

(None identified.)

---

## HIGH

(None identified.)

---

## MEDIUM

(None identified.)

---

## LOW

### XR-BEN-L1: metal_benchmark.py MPS Availability Guards Are Correct

- **File:Line**: `metal_benchmark.py` (multiple locations)
- **Found by**: xreview-metal (positive finding during cross-review)
- **Cross-review verdict**: CORRECT — all 8 benchmarks properly check `torch.backends.mps.is_available()` before attempting MPS operations.

**Description**: The benchmark suite (elementwise, reduction, Laplacian, CT update, HLL flux, full MHD step, memory transfer, WALRUS inference) consistently guards MPS operations behind availability checks. Benchmarks gracefully skip or run CPU-only when MPS is unavailable. The timing methodology uses `torch.mps.synchronize()` before timing GPU operations, which is the correct pattern for Apple Silicon (avoids measuring only kernel launch latency).

**Evidence**:
- Each benchmark function checks MPS availability
- `torch.mps.synchronize()` called before `time.perf_counter()` end markers
- CPU baselines provided for all GPU benchmarks

**Impact**: Positive — benchmarks are robust and produce meaningful timing numbers.

### XR-BEN-L2: apple_silicon.py Is CPU-Only Despite Name ✅ FIXED

- **File:Line**: `apple_silicon.py` (entire file, 270 lines)
- **Found by**: xreview-metal (observation during cross-review)
- **Cross-review verdict**: Not a bug — the file benchmarks Python-engine MHD operations on Apple Silicon CPU, not GPU. The name refers to the hardware platform, not the compute target.

**Description**: `apple_silicon.py` runs CPU-only benchmarks: viscosity stencil, CT update, Sod shock tube, and cylindrical MHD. It uses NumPy, not PyTorch MPS. This is complementary to `metal_benchmark.py` which covers GPU benchmarks. The file name could be clearer (e.g., `cpu_benchmark.py` or `python_engine_benchmark.py`) but this is a naming convention issue, not a defect.

**Fix applied**: Updated module docstring to explicitly state "CPU performance benchmarks" and note that this module benchmarks the Python/NumPy/Numba engine only, directing users to `metal_benchmark.py` for GPU benchmarks.

**Impact**: None. Potential confusion from file name only.

### XR-BEN-L3: No Warm-Up Runs Before Timing in apple_silicon.py ✅ FIXED

- **File:Line**: `apple_silicon.py:~60-80`
- **Found by**: xreview-metal (new finding during cross-review)
- **Cross-review verdict**: Minor benchmark methodology concern.

**Description**: The CPU benchmarks in `apple_silicon.py` do not perform warm-up runs before timing. For Numba-accelerated functions, the first call includes JIT compilation time. For NumPy, the first call may trigger memory allocation that subsequent calls avoid. `metal_benchmark.py` correctly uses warm-up runs.

**Fix applied**: `benchmark_sod` and `benchmark_cylindrical_mhd` both used `warmup=1`; increased to `warmup=2` to ensure Numba JIT compilation and initial memory allocation are excluded from timing. (`benchmark_viscosity` and `benchmark_ct` already used `warmup=2`.)

**Impact**: Benchmark numbers may overestimate runtime due to cold-start effects. No physics impact.

---

## REJECTED

(None identified.)

---

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| CRITICAL | 0 | — |
| HIGH | 0 | — |
| MEDIUM | 0 | — |
| LOW | 3 | XR-BEN-L1, L2, L3 |
| REJECTED | 0 | — |

**Overall benchmarks domain assessment**: 8.0/10. Both benchmark files are well-structured with proper MPS availability guards, correct GPU synchronization timing, and clean result formatting. `metal_benchmark.py` is production-quality with warm-up runs and multi-iteration averaging. `apple_silicon.py` is simpler but functional. No physics or correctness issues found by either expert review. The only concerns are naming clarity and warm-up methodology in the CPU benchmarks.
