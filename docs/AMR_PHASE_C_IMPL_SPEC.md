# AMR Phase C: Cylindrical Flux Correction — Implementation Spec

**Date**: 2026-03-26  |  **Status**: IMPLEMENTATION-READY
**Refs**: Berger & Colella (1989) JCP 82:64 §2.2–2.4; Stone et al. (2020) ApJS 249:4
§3.3 (Athena++ flux_correction_cc.cpp); Keppens et al. (2023) A&A 673:A66 §4.3
(MPI-AMRVAC mod_fix_conserve); Vaidya et al. (2007) JCoPh 226:925 (AstroBEAR CT)

---

## 1. The Math: Cylindrical Flux Correction

### Geometry primitives (factor-of-2pi cancellation shown explicitly)

The cylindrical volume element is `dV = 2*pi*r*dr*dz`. The factor `2*pi` appears
in every area and volume, so it cancels in all correction ratios. The code uses
the **reduced forms** (with `2*pi` divided out) throughout:

```
Full face area, radial:   A_r_full  = 2*pi * r_face * dz
Reduced face area, r:     A_r       =        r_face * dz            (1)

Full face area, axial:    A_z_full  = 2*pi * 0.5 * (r_hi^2 - r_lo^2)
Reduced face area, z:     A_z       =        0.5 * (r_hi^2 - r_lo^2) (2)

Full cell volume:         V_full    = 2*pi * 0.5 * (r_hi^2 - r_lo^2) * dz
Reduced cell volume:      V         =        0.5 * (r_hi^2 - r_lo^2) * dz  (3)
```

With these reduced forms, the conservation update is identical to Cartesian:

```
delta_U_c = (sum_fine F_f * A_f - F_c * A_c) * dt / V_c             (4)
```

The `2*pi` in the numerator (`2*pi * F * A_r`) and denominator (`2*pi * V`)
cancel exactly. This is the "geometry-agnostic" property used by MPI-AMRVAC
and Athena++.

### Correction formula expanded for ratio=2

Consider one coarse cell at `(ir, iz)` adjacent to the fine level on its `r_hi`
face. The coarse face is at `r_face = r_hi = r_lo + dr_c`. The fine level covers
the same spatial region with two faces in the z-direction:

```
Fine face 1 (iz*2):      r_face_f = r_hi   (same radial position)
Fine face 2 (iz*2 + 1):  r_face_f = r_hi   (same radial position)
```

For a radial CF boundary (faces in the r-direction), all fine faces sharing that
coarse face are at the **same** r_face. Reduced areas:

```
A_coarse_r = r_hi * dz_c
A_fine_r1  = r_hi * dz_f   (dz_f = dz_c / 2)
A_fine_r2  = r_hi * dz_f

A_fine_r1 + A_fine_r2 = r_hi * dz_c = A_coarse_r                   (5)
```

The areas sum correctly. The correction is:

```
delta_U_c = dt / V_c * [r_hi * (F_f1*dz_f + F_f2*dz_f) - r_hi * F_c * dz_c]
          = dt / V_c * r_hi * dz_c * [(F_f1 + F_f2)/2 - F_c]       (6)
```

For an axial CF boundary (faces in the z-direction), the two fine faces are at
**different** r positions: `r_f1 = r_lo + dr_f/2` and `r_f2 = r_lo + 3*dr_f/2`.
The reduced areas:

```
A_coarse_z = 0.5 * (r_hi^2 - r_lo^2)  = r_mid * dr_c           (midpoint approx)
A_fine_z1  = 0.5 * ((r_lo + dr_f)^2 - r_lo^2)     = r_lo*dr_f + (dr_f^2)/2
A_fine_z2  = 0.5 * (r_hi^2 - (r_lo + dr_f)^2)     = r_hi*dr_f - (dr_f^2)/2

A_fine_z1 + A_fine_z2 = (r_hi^2 - r_lo^2) / 2 = A_coarse_z              (7)
```

The areas still sum to the coarse area, but the individual values differ. The
correction must use the exact A_fine for each sub-face — NOT simply halved:

```
delta_U_c = dt / V_c * [A_fine_z1*F_f1 + A_fine_z2*F_f2 - A_coarse_z*F_c]  (8)
```

This is the key cylindrical gotcha: **axial fine areas are asymmetric in r**.

### Axis boundary (r=0)

At r=0, `r_face = 0`, so `A_r = r_face * dz = 0`. The radial flux through the
axis is identically zero by the axisymmetric boundary condition. If a coarse cell
abuts the axis on its `r_lo` face, that face carries zero flux — no refluxing
contribution from or to the axis. The `V_c > 0` guard in `apply_correction`
handles this: for the innermost cell, `r_lo = 0` and `V_c = 0.5 * r_hi^2 * dz > 0`
(not zero), so the cell is updated normally. Only the face area at r=0 is zero,
which is correct physics (no mass/momentum flux through axis).

---

## 2. mhd_rhs Interface Change

`mhd_rhs` currently returns `dU_dt: mx.array` only. The fluxes `F_r` and `F_z`
are already computed internally (lines 796, 826) and discarded after the divergence
step. The change is minimal: expose them via a keyword argument.

### Exact signature change (`mlx_riemann.py`)

```python
def mhd_rhs(
    U: mx.array,
    grid: object,
    gamma: float = 5.0 / 3.0,
    dr: float = 1.0,
    dz: float = 1.0,
    method: str = "weno5z",
    riemann: str = "hlld",
    precision: str = "float32",
    return_fluxes: bool = False,                              # NEW
) -> mx.array | tuple[mx.array, mx.array, mx.array]:         # NEW
```

Dispatch to `_mhd_rhs_cylindrical` with the flag; that function returns
`(dU_dt, F_r, F_z)` when `return_fluxes=True`, otherwise `dU_dt` only.

### Change inside `_mhd_rhs_cylindrical` (no signature change to `compute_fluxes`)

```python
# After computing F_r and F_z (lines 796, 826) — no other code changes.
# At the very end, before the return:
    if return_fluxes:
        return dU_dt, F_r, F_z      # F_r: (NVAR, n_ifaces_r, nz)
    return dU_dt                    # backward compatible
```

**F_r shape**: `(NVAR, nr+1, nz)` — one interface per cell boundary in r.
The slice convention matches existing code: `F_r[:, k, :]` is the flux through
the face between cells `k-1` and `k` (zero-indexed from the ghost region start).

**F_z shape**: `(NVAR, nr, nz+1)` — one interface per cell boundary in z.

Callers that do not pass `return_fluxes=True` receive exactly the same object
they receive today. No callsite changes required unless refluxing is enabled.

---

## 3. FluxRegister Implementation

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FluxRegisterCylindrical:
    """Stores flux*area*dt sums on both sides of each coarse-fine face.

    One register per (coarse_level, face_id) pair. face_id is an integer
    assigned by build_cf_face_map() and stable across subcycle sub-steps.
    """
    # Keys: face_id (int)
    # Values: accumulated flux*area*dt, shape (NVAR,)
    fine_sum: dict[int, np.ndarray] = field(default_factory=dict)
    coarse_val: dict[int, np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        self.fine_sum.clear()
        self.coarse_val.clear()

    def accumulate_fine(
        self,
        face_id: int,
        F_normal: np.ndarray,   # (NVAR,) — flux at this fine face (normal component)
        area: float,            # reduced area A (no 2*pi), pre-computed
        dt: float,
    ) -> None:
        """Sum fine flux*area*dt. Called once per fine sub-step per face."""
        contribution = F_normal * area * dt
        if face_id not in self.fine_sum:
            self.fine_sum[face_id] = np.zeros_like(contribution)
        self.fine_sum[face_id] += contribution

    def accumulate_coarse(
        self,
        face_id: int,
        F_normal: np.ndarray,   # (NVAR,) — flux at this coarse face
        area: float,            # reduced area A of the coarse face
        dt: float,
    ) -> None:
        """Record coarse flux*area*dt. Called once per coarse step."""
        self.coarse_val[face_id] = F_normal * area * dt

    def apply_correction(
        self,
        U_coarse: np.ndarray,   # (NVAR, nr, nz), in-place update
        face_id: int,
        ir: int,
        iz: int,
        V_c: float,             # reduced cell volume (no 2*pi)
        sign: float,            # +1 if fine region is on "hi" side, -1 if "lo"
    ) -> None:
        """Apply Berger-Colella correction to one coarse cell.

        delta_U = sign * (fine_sum - coarse_val) / V_c
        Sign: +1 if the CF face is on the hi side of the coarse cell
              (the fine region is to the right/top of the coarse cell),
              -1 if on the lo side.

        Convention matches Athena++ flux_correction_cc.cpp:
            U[i] -= dt * (fine_FA - coarse_FA) / vol
        where the sign is absorbed into the direction of flux.
        """
        if face_id not in self.fine_sum or face_id not in self.coarse_val:
            return
        if V_c < 1e-30:   # axis singularity guard
            return
        delta = (self.fine_sum[face_id] - self.coarse_val[face_id]) / V_c
        U_coarse[:, ir, iz] += sign * delta
```

### Pre-computed face map

`build_cf_face_map` runs once after each regrid and returns a list of
`CFace` objects, each carrying all data needed by the accumulate/apply calls:

```python
@dataclass
class CFace:
    face_id: int
    coarse_block_idx: tuple[int, int]
    ir: int               # coarse cell index
    iz: int
    face_dir: str         # "r" or "z"
    face_side: str        # "lo" or "hi"
    sign: float           # +1 hi, -1 lo
    coarse_face_pos: int  # interface index in F_r or F_z for coarse block
    coarse_area: float    # pre-computed reduced area
    coarse_V: float       # pre-computed reduced cell volume
    fine_faces: list[tuple[
        tuple[int, int],  # fine block_idx
        int,              # ir in fine block
        int,              # iz in fine block
        int,              # interface index in fine F_r or F_z
        float,            # fine face reduced area (differs for z-faces)
    ]]
```

This pre-computation eliminates all r-arithmetic from the hot path.

---

## 4. Conservation Proof and Acceptance Test

### Proof

Let `Omega_c` be the coarse cell and `Omega_f1`, `Omega_f2` be the two fine
cells covering the same spatial volume. After one coarse step + two fine
sub-steps + refluxing:

```
integral(U_c, Omega_c) - integral(U_c^n, Omega_c)
  = -dt * [F_c * A_c - (F_f1*A_f1 + F_f2*A_f2)*dt_f/dt_c term...]
  + reflux correction

After refluxing:
  = -dt * F_c * A_c + correction
  = -dt * F_c * A_c + (fine_FA - coarse_FA) / V_c * V_c
  = -dt * (sum F_fine * A_fine)   <- same as uniform fine grid
```

The composite solution matches a uniform fine grid to the precision of
floating-point arithmetic in the accumulated fine_sum.

### Test (`tests/test_amr_phase_c.py`)

```python
def test_refluxing_conservation_1d_sod():
    """Global mass conserved to machine precision after one coarse step.

    Setup: 1D Sod shock tube mapped onto a 2-level AMR hierarchy.
    The contact discontinuity is positioned to cross the CF boundary
    at exactly step 5. Without refluxing, dm/m ~ 1e-4. With refluxing,
    dm/m < 1e-12 (double precision) or < 1e-6 (float32).
    """
    # Build 2-level hierarchy: coarse 16 cells, fine 32 cells (left half)
    hier = build_sod_hierarchy(n_coarse=16, refinement_ratio=2)
    register = FluxRegisterCylindrical()

    m0 = total_mass(hier)
    for step in range(10):
        dU_c, F_r_c, F_z_c = mhd_rhs(hier.coarse.U, ..., return_fluxes=True)
        register.accumulate_coarse(...)
        # Two fine sub-steps
        for _ in range(2):
            dU_f, F_r_f, F_z_f = mhd_rhs(hier.fine.U, ..., return_fluxes=True)
            register.accumulate_fine(...)
        register.apply_correction(hier.coarse.U, ...)
        register.reset()

    m1 = total_mass(hier)
    assert abs(m1 - m0) / m0 < 1e-12   # float64 path
    # OR
    assert abs(m1 - m0) / m0 < 1e-6    # float32 path (MLX)
```

`total_mass` sums `rho * V_c` over all cells using the reduced volume formula
(eq. 3). The test must use the cylindrical geometry (not Cartesian) because the
asymmetric z-face areas (eq. 7) are the failure mode.

---

## 5. Cylindrical Gotchas

### Gotcha A: Radial faces at CF boundary in r

One coarse r-face maps to `ratio` fine r-faces stacked in z. All fine faces
are at the **same** r_face value. The reduced areas satisfy eq. (5) exactly.
No asymmetry — the sum of fine areas equals the coarse area with no remainder.

Implementation: loop over `iz_fine in range(ratio)`, accumulate `F_fine * r_face * dz_f`.

### Gotcha B: Axial faces at CF boundary in z

One coarse z-face maps to `ratio` fine z-faces stacked in r. The fine faces
are at **different** r positions, so their reduced areas (eq. 2) differ. The
areas still sum to the coarse area (eq. 7), but only if computed exactly from
the actual r bounds — not by dividing the coarse area by `ratio`.

**Wrong** (tempting shortcut, violates conservation):
```python
A_fine_each = A_coarse_z / ratio   # WRONG — ignores r variation
```

**Correct** (must use exact r bounds for each sub-face):
```python
for dr_offset in range(ratio):
    r_lo_f = r_lo_coarse + dr_offset * dr_fine
    r_hi_f = r_lo_f + dr_fine
    A_fine = 0.5 * (r_hi_f**2 - r_lo_f**2)    # exact, not A_coarse/ratio
    register.accumulate_fine(fid, F_fine, A_fine, dt_fine)
```

The residual `A_fine_z1 - A_fine_z2 = dr_f * (r_lo + r_hi)/2 * ... ` is
O(dr_fine/r_mid), which is small but non-zero and exactly the truncation error
that refluxing must remove.

### Gotcha C: Sign convention

The sign of the correction depends on which side of the coarse cell the fine
region occupies. Convention (consistent with Athena++ `pmy_block->pcoord->GetFace1Area`):

```
face_side = "hi"  →  fine cells are to the right/top of coarse cell
                   →  sign = +1   (fine flux entering coarse cell from hi face)
face_side = "lo"  →  fine cells are to the left/bottom
                   →  sign = -1
```

A sign error here adds mass instead of conserving it (FMEA risk C1, RPN 135).
The acceptance test in Section 4 will catch this within 1 step — the mass
error will grow rather than shrink.

### Gotcha D: Fluxes are at interfaces, not cell centers

`F_r` from `compute_fluxes(..., dim=0)` has shape `(NVAR, n_ifaces_r, nz)`.
The interface at index `k` in F_r lies between coarse cells `k-1` and `k`
(counting from the ghost region boundary). The CF face index must be extracted
from the correct column, not from a cell-center index:

```python
# Coarse cell at ir=N is adjacent to CF boundary at face k = N+1 (hi side)
F_coarse_at_cf = np.asarray(F_r_coarse[:, N + 1, :])  # NOT F_r[:, N, :]
```

---

## 6. Risk Assessment

Without refluxing, the concerns analysis quantified 0.3–3% cumulative mass error
over a 20,000-step PF-1000 discharge. The range spans:
- Low (0.08%): slow sheath, narrow shock relative to block, WENO5-Z captures it cleanly
- Nominal (0.32%): PLM reconstruction, 32 effective CF crossings
- High (3.2%): multiple crossings, PLM, strong shock (radial implosion phase)

After refluxing, the residual error comes from:

1. **Float32 accumulation** in `fine_sum`: Each fine sub-step adds a float32
   contribution. For ratio=2 there are only 2 additions per coarse step —
   no significant rounding. For ratio=4 (if implemented), 4 additions accumulate
   ~4*eps32 ~ 5e-7 relative error per step. Cumulative over 20,000 steps: ~1e-2.
   **Recommendation**: always accumulate in float64 even when the solver runs
   float32. `np.float64` for register arrays costs negligible memory.

2. **Prolongation error at CF ghost cells**: Before the fine RHS is evaluated,
   coarse data is prolongated into fine ghost zones. Prolongation is 2nd-order
   (van Leer slopes in Phase B). The flux across the ghost layer carries O(dr_c^2)
   error — but this enters both `fine_sum` and `coarse_val` symmetrically, so it
   cancels in the difference. Residual is O(dr_c^3) — negligible.

3. **Geometric area mismatch if axial shortcut used**: Using `A_fine = A_coarse/2`
   instead of exact areas introduces O(dr_f/r) error per crossing. At r=5 mm,
   dr_f=0.12 mm: error = 2.4%. This does NOT cancel — it is the dominant source
   of residual if Gotcha B is mishandled.

**Expected residual after correct refluxing**:
- With float64 accumulation + exact cylindrical areas: `dm/m < 1e-10` per step,
  `< 2e-6` cumulative over 20,000 steps (dominated by float64 round-off).
- With float32 accumulation: `dm/m < 1e-5` per step, `< 2e-1` cumulative —
  unacceptable. Do not accumulate in float32.

---

## Implementation Checklist

| Step | File | LOC | Risk |
|------|------|-----|------|
| Add `return_fluxes` kwarg to `mhd_rhs` | `mlx_riemann.py` | +8 | Low (default False) |
| Thread flag into `_mhd_rhs_cylindrical` | `mlx_riemann.py` | +4 | Low |
| `FluxRegisterCylindrical` class | `amr_reflux.py` (new) | ~80 | Medium |
| `CFace` dataclass + `build_cf_face_map` | `amr_reflux.py` | ~70 | High (index arithmetic) |
| Wire into `amr_step_recursive` (Phase D) | `amr_phase_d.py` | ~25 | Medium |
| Conservation test | `tests/test_amr_phase_c.py` | ~60 | — |
| Float64 accumulation in register | `amr_reflux.py` | 0 (use np.float64 default) | Low |

Total new code: ~240 LOC excluding tests.

Gate criteria before Phase D integration:
- `test_refluxing_conservation_1d_sod` passes with `dm/m < 1e-6`
- Sign test: manually flip sign in `apply_correction`, verify mass grows monotonically
- Axis test: place CF boundary at `ir=0` (innermost block), verify no NaN, `V_c > 0`
