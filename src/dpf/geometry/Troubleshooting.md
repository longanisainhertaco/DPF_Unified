# Geometry Domain — Cross-Review Troubleshooting Guide

**Cross-reviewer**: xreview-metal
**Source reviews**: py-metal (Python expert), phys-metal (Physics expert)
**Files reviewed**: cylindrical.py (357 lines)
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

### XR-GEO-L1: Numba @njit _safe_inv_r May Not JIT on First Import ✅ NO ACTION NEEDED

- **File:Line**: `cylindrical.py:~35-45`
- **Found by**: xreview-metal (new finding during cross-review)
- **Cross-review verdict**: Minor performance concern, not a correctness issue.
- **Resolution**: No code change needed per cross-review recommendation. `cache=True` mitigates repeat costs. Documented Numba behavior per CLAUDE.md lesson #6. Ruff check passes.

**Description**: The `_safe_inv_r(r, r_min)` function uses `@njit(cache=True)` which triggers Numba JIT compilation on first call. In test environments where Numba is cold, this adds 1-5 seconds latency to the first geometry operation. The `cache=True` flag mitigates repeat costs but the initial compilation penalty remains.

**Evidence**:
- `cylindrical.py:~38`: `@njit(cache=True)` decorator on `_safe_inv_r`
- Known Numba JIT latency pattern documented in CLAUDE.md lesson #6

**Proposed fix**: No code change needed. This is documented Numba behavior. If latency is problematic, consider a pure NumPy fallback: `inv_r = np.where(r > r_min, 1.0/r, 1.0/r_min)`.

**Impact**: 1-5 second first-call latency. No correctness impact.

### XR-GEO-L2: Cylindrical Geometry Uses SI Units Consistently ✅ CONFIRMED CORRECT

- **File:Line**: `cylindrical.py` (entire file)
- **Found by**: xreview-metal (positive finding during cross-review)
- **Cross-review verdict**: CORRECT — this file imports `MU_0` from `dpf.constants` and uses SI units throughout, unlike the Metal solver which uses mixed HL/SI. This is the correct reference implementation.
- **Resolution**: No action needed. Positive finding confirmed. Ruff check passes.

**Description**: The `CylindricalGeometry` class consistently uses SI electromagnetic units with explicit `mu_0` from `dpf.constants`. The curl, divergence, and gradient operators all handle the 1/r geometric factors correctly. The geometric source terms for momentum (hoop stress `p + B^2/(2*mu_0)`) and B-field induction use the proper cylindrical identities.

**Evidence**:
- Uses `from dpf.constants import MU_0` (or equivalent)
- Geometric source terms match Stone & Norman (1992) and Mignone (2007)
- Divergence includes the required 1/r * d(r*f_r)/dr form

**Impact**: Positive — this module is a clean reference for how cylindrical MHD geometry should be implemented.

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
| LOW | 2 | XR-GEO-L1, L2 |
| REJECTED | 0 | — |

**Overall geometry domain assessment**: 8.5/10. The `cylindrical.py` module is clean, well-structured, and physically correct. It uses SI units consistently (unlike the Metal solver), implements proper 1/r geometric factors, and correctly handles the axis singularity at r=0. No bugs identified by either the Python or Physics expert reviews. The only note is the standard Numba JIT latency on first call, which is a known platform characteristic, not a code defect.
