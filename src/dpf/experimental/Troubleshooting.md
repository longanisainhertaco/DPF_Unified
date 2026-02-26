# Experimental Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 4 files (~2,260 LOC)

---

## HIGH

### ✅ FIXED HIGH-1: species.py np.roll Gives Periodic BC — Wrong for DPF
- **File:Line**: `species.py:293-294, 301`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: Species advection uses `np.roll` for flux computation, which wraps array boundaries periodically. DPF geometry uses reflecting/outflow boundaries, not periodic.
- **Fix applied**: Replaced both `np.roll` calls with explicit boundary slicing using zero-gradient outflow BC at the high boundary and zero inflow flux at the low boundary. Works for arbitrary axis `d` via dynamic slice lists.
- **Impact**: Ablated material that reaches the domain boundary wraps to the opposite side. Unphysical mass transport in boundary regions.

---

## MEDIUM

### ✅ FIXED MOD-1: AMR grid.py _add_buffer O(N^2) Performance
- **File:Line**: `amr/grid.py:264-291`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: Buffer expansion uses nested Python loops. For 256x256 grid with buffer=4: ~1M iterations in pure Python.
- **Fix applied**: Replaced with `scipy.ndimage.binary_dilation` using a cross-shaped (4-connected) structuring element that matches the original 4-neighbor logic. O(N) performance.

### ✅ FIXED MOD-2: AMR grid.py Single Bounding Box for Disconnected Regions
- **File:Line**: `amr/grid.py:299-335`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `_find_tagged_bbox` returns one box encompassing ALL tagged cells. Disconnected regions get a single oversized patch.
- **Fix applied**: Replaced with `scipy.ndimage.label` for connected-component analysis. Each disconnected region now gets its own tight bounding box.

### MOD-3: Boris Pusher Missing Coulomb Collisions
- **File:Line**: `pic/hybrid.py` (entire module)
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED as known limitation
- **Description**: Collisionless Boris pusher. For DPF beam-target, beam slowing-down in dense target requires a collision operator (Fokker-Planck or Monte Carlo).
- **Impact**: PIC module is not integrated into engine.py (gated by `config.kinetic.enabled=False`). No production impact currently.
- **Action taken**: Added explicit `.. note::` in `push_particles` docstring documenting the limitation, its physics impact, and deferral to a future phase.

### ✅ FIXED MOD-4: gpu_backend.py Dead on Apple Silicon
- **File:Line**: `gpu_backend.py` (entire module, 119 LOC)
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: CuPy-only GPU backend. Apple Silicon has no CUDA. Production GPU path uses PyTorch MPS via MetalMHDSolver.
- **Fix applied**: Added `.. deprecated::` docstring section and a `warnings.warn(DeprecationWarning)` at module import time pointing users to `MetalMHDSolver`.

---

## INFORMATIONAL

### INFO-1: Species Advection First-Order Upwind Only
- **Found by**: py-ai-diag
- **Assessment**: Acknowledged in docstring. Higher-order reconstruction planned for future phase.

### INFO-2: PIC Module Not Integrated
- **Found by**: py-ai-diag
- **Assessment**: Boris pusher and HybridPIC are complete but never called from `engine.py`. Gated by `config.kinetic.enabled` (default False).

---

## REJECTED FINDINGS

None.
