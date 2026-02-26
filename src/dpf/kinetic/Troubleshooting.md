# Kinetic Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 2 files (~1,105 LOC)

---

## CRITICAL

### ✅ FIXED — CRIT-1: hybrid.py Is Near-Identical Duplicate of experimental/pic/hybrid.py
- **File:Line**: `hybrid.py` (978 LOC → now 33 LOC thin re-export)
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED (see `src/dpf/ai/Troubleshooting.md` CRIT-2 for full details)
- **Description**: ~978 LOC duplicated with RNG divergence at beam injection.
- **Fix applied**: Replaced 978-LOC copy with thin re-export from `dpf.experimental.pic.hybrid`.
  All public symbols (`HybridPIC`, `ParticleSpecies`, `boris_push`, `deposit_density`,
  `deposit_current`, `interpolate_field_to_particles`, `detect_instability`) are re-exported.
  Canonical implementation uses `np.random.default_rng()` (modern, local RNG).

---

## MEDIUM

### MOD-1: manager.py Isotropic Grid Assumption
- **File:Line**: `manager.py:33-34`
- **Found by**: py-ai-diag, phys-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `dy=config.dx, dz=config.dx` hardcodes isotropic grid. For cylindrical DPF geometry, dr != dz in general.
- **Impact**: Particle dynamics incorrect for non-cubic grids. Module is not integrated into engine (low production impact).

### MOD-2: manager.py Hardcoded Constants
- **File:Line**: `manager.py:36, 126`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `dt=1e-9` placeholder timestep (line 36) and `spread=0.1` hardcoded beam angular spread (line 126).
- **Impact**: Non-configurable physics parameters. Module not in production use.

### ✅ FIXED — MOD-3: manager.py Missing `Any` Import
- **File:Line**: `manager.py:60`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED (works at runtime due to `from __future__ import annotations`)
- **Fix applied**: Added `from typing import Any` to manager.py imports.

---

## REJECTED FINDINGS

None.
