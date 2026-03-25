# Sprint 7 Feature Parity Research Specification

**Date**: 2026-03-24
**Methodology**: Six Sigma DMAIC
**Scope**: MLX Cartesian 3D Support, Dedner/Powell Divergence Cleaning, Braginskii Viscosity
**Author**: dpf-mhd-physicist agent

---

## Table of Contents

1. [Item 1: MLX Cartesian 3D Support](#item-1-mlx-cartesian-3d-support)
2. [Item 2: Dedner/Powell Divergence Cleaning for MLX](#item-2-dednerpowell-divergence-cleaning-for-mlx)
3. [Item 3: Braginskii Viscosity for MLX](#item-3-braginskii-viscosity-for-mlx)
4. [Cross-Item Dependencies](#cross-item-dependencies)
5. [Risk Register](#risk-register)

---

## Item 1: MLX Cartesian 3D Support

### 1.1 Define

**Problem Statement**: The MLX MHD solver (`MLXMHDSolver`) is hard-coded for axisymmetric cylindrical (r,z) geometry with `ny=1`. Cartesian 3D support is required for feature parity with `MetalMHDSolver`, which supports both `coordinates="cartesian"` and `coordinates="cylindrical"` with full 3D grids.

**Customer Requirement**: Users running Sod 3D, Brio-Wu 3D, and Orszag-Tang 2D vortex tests need a Cartesian MLX backend for cross-backend validation and performance benchmarking against the PyTorch Metal solver.

**Scope**: Add `coordinates="cartesian"` mode to `MLXMHDSolver` with 3D grid support (nx, ny, nz all >= 1), no geometric source terms, outflow/periodic boundary conditions, and a third flux sweep in the y-direction.

### 1.2 Measure

#### 1.2.1 Current State Analysis

**Where `ny=1` is enforced**:
- `src/dpf/metal/mlx_solver.py:131-135` -- Constructor raises `ValueError` if `ny != 1`:
  ```python
  nr, ny, nz = grid_shape
  if ny != 1:
      raise ValueError(
          f"MLXMHDSolver is axisymmetric (ny=1 required), got ny={ny}."
      )
  ```

**Grid class is cylindrical-only**:
- `src/dpf/metal/mlx_grid.py:19` -- `CylindricalGrid` computes radial cell volumes `pi*(r_{i+1/2}^2 - r_{i-1/2}^2)*dz`, face areas `2*pi*r_face*dz`, and `inv_r`. These are meaningless in Cartesian geometry.

**`mhd_rhs` assumes 2D (nr, nz)**:
- `src/dpf/metal/mlx_riemann.py:251-429` -- The `mhd_rhs` function performs two flux sweeps:
  - `dim=0` (radial): r-weighted flux divergence `-(r_R*F_R - r_L*F_L) / (r_c * dr)` at line 352
  - `dim=1` (axial): standard flux divergence `-(F_R - F_L) / dz` at line 373
  - No `dim=2` sweep exists.

**Geometric source terms are cylindrical-specific**:
- `src/dpf/metal/mlx_kernels.py:886-1087` -- `cylindrical_source_mlx` computes three source terms:
  - `S_mr = (p_tot - Btheta^2) / r + rho * vtheta^2 / r` (centripetal + hoop stress)
  - `S_mt = -(rho * vr * vtheta - Br * Btheta) / r` (angular momentum transport)
  - `S_Bt = -(vr * Btheta - Br * vtheta) / r` (induction geometric term)
  - In Cartesian geometry, ALL geometric source terms are identically zero.

**Electrode BCs are cylindrical-only**:
- `src/dpf/metal/mlx_solver.py:226-375` -- `_pad_electrode_ghost` sets `B_theta = mu0*I/(2*pi*r)` at the cathode face. This BC is physically meaningful only in cylindrical coordinates.

**Conservative variable layout assumes 2D storage**:
- `src/dpf/metal/mlx_solver.py:13-15` -- State shape is `(10, nr, nz)`. For 3D Cartesian, this must become `(10, nx, ny, nz)` or `(NVAR, nx, ny, nz)`.

**HLLD kernel assumes 2D interface layout**:
- `src/dpf/metal/mlx_kernels.py:363-882` -- The Metal HLLD kernel indexes `stride = n_ifaces * nz` and uses `thread_position_in_grid.x` for interface index and `.y` for transverse index. For 3D, the transverse dimension becomes 2D (ny, nz), requiring a flattened or 3D grid dispatch.

**How MetalMHDSolver handles 3D Cartesian**:
- `src/dpf/metal/metal_solver.py:217` -- Accepts `coordinates: str = "cartesian"` (default).
- `src/dpf/metal/metal_solver.py:311-322` -- Cylindrical grid arrays (`_r`, `_inv_r`, `_r_face`) are only built when `coordinates == "cylindrical"`; otherwise they stay `None`.
- `src/dpf/metal/metal_solver.py:627-651` -- The `_compute_rhs` method branches:
  - Cylindrical: calls `mhd_rhs_cylindrical_mps()` with `r_cell`, `r_face` arguments.
  - Cartesian: calls `mhd_rhs_mps()` which does standard 3-axis flux sweeps without geometric source terms.
- `src/dpf/metal/metal_solver.py:1422-1425` -- Electrode BC application checks `self.coordinates == "cartesian"` and decomposes `B_theta` into `(Bx, By)` components for Cartesian grids (not applicable for MLX since we skip electrode BCs in Cartesian mode).

#### 1.2.2 Quantitative Gap

| Feature | MLX Solver | Metal (PyTorch) Solver | Gap |
|---------|-----------|----------------------|-----|
| Cartesian 3D grid | Not supported | Full 3D (nx, ny, nz) | Missing |
| y-direction flux sweep | Not implemented | `dim=1` in `mhd_rhs_mps` | Missing |
| Geometric source terms | Cylindrical only | None for Cartesian | N/A (simpler) |
| Electrode ghost BCs | Cylindrical only | Cylindrical only | N/A |
| Outflow/periodic BCs | Ghost pad (electrode) | Per-axis BC tuple | Missing for Cartesian |
| State shape | `(NVAR, nr, nz)` | `(nx, ny, nz)` per field | Different convention |

### 1.3 Analyze

#### 1.3.1 Root Cause

The MLX solver was purpose-built during Phase B for DPF cylindrical simulations. Cartesian 3D was never a design goal because:

1. The primary use case is the Dense Plasma Focus, which has axial symmetry.
2. The `CylindricalGrid` class, geometric source kernel, and electrode ghost-cell BC were all designed around the (r, z) coordinate system.
3. The HLLD Metal kernel dispatches threads over a 2D grid `(n_ifaces, nz)`, not 3D.

#### 1.3.2 Impact Analysis

Adding Cartesian 3D affects these modules:

| Module | Change Type | Complexity |
|--------|------------|------------|
| `mlx_solver.py` | Remove `ny=1` guard, add coordinate branching | Medium |
| `mlx_grid.py` | New `CartesianGrid` class (or generalize) | Medium |
| `mlx_riemann.py` | Add `dim=2` (y-direction) sweep in `mhd_rhs` | Medium |
| `mlx_kernels.py` | HLLD kernel: 3D thread dispatch (or flatten transverse) | High |
| `mlx_kernels.py` | Ghost pad: Cartesian outflow/periodic (no electrode) | Medium |
| `mlx_timestepper.py` | CFL: add y-direction fast speed | Low |
| `mlx_primitives.py` | No change (scalar operations, shape-agnostic) | None |
| `mlx_reconstruction.py` | Add `dim=2` reconstruction | Low-Medium |
| `mlx_transport.py` | No change (1D Thomas solver along z; add y if needed) | Low |
| `mlx_ct.py` | Full 3D CT or skip (Dedner preferred for 3D) | High (defer) |

### 1.4 Improve

#### 1.4.1 Implementation Plan

**Phase 1: Grid abstraction (~80 LOC)**

Create a `CartesianGrid` class in `mlx_grid.py` that provides the same attribute interface as `CylindricalGrid` but with Cartesian geometry:

```
File: src/dpf/metal/mlx_grid.py
New class: CartesianGrid
Attributes:
  nx, ny, nz: int
  dx, dy, dz: float
  cell_volume: mx.array  # dx * dy * dz (uniform)
  r_cell: None           # Not applicable
  r_face: None           # Not applicable
  inv_r: None            # Not applicable
```

**Phase 2: Remove ny=1 guard, add coordinate branching (~40 LOC)**

```
File: src/dpf/metal/mlx_solver.py
Line 131-135: Replace ValueError with coordinate-based branching:
  - cylindrical: keep ny=1 requirement, build CylindricalGrid
  - cartesian: allow arbitrary ny, build CartesianGrid
Line 196-212: _build_internals() branches on self.coordinates
Line 592-600: _ghost_active condition: skip electrode BC for cartesian
```

**Phase 3: 3D flux computation (~120 LOC)**

```
File: src/dpf/metal/mlx_riemann.py
Function: mhd_rhs()
Changes:
  - Accept grid that may be CartesianGrid (no r_cell/r_face)
  - dim=0 (x): standard flux divergence -(F_R - F_L) / dx (no r-weighting)
  - dim=1 (y): standard flux divergence -(F_R - F_L) / dy (NEW)
  - dim=2 (z): standard flux divergence -(F_R - F_L) / dz (currently dim=1)
  - Skip geometric source terms when grid has no r_cell attribute
```

The key insight: Cartesian flux divergence is SIMPLER than cylindrical because there is no r-weighting. The existing axial (dim=1) flux divergence code at `mlx_riemann.py:369-378` is already the correct template:
```python
div_Fz = -(F_R_z - F_L_z) / dz_eff
```

For 3D, we repeat this pattern for all three dimensions.

**Phase 4: 3D reconstruction (~30 LOC)**

```
File: src/dpf/metal/mlx_reconstruction.py
Function: reconstruct(U, dim, method)
Changes:
  - Currently handles dim=0 (axis 1 of state) and dim=1 (axis 2 of state)
  - State shape changes from (NVAR, nr, nz) to (NVAR, nx, ny, nz)
  - dim=0: reconstruct along axis 1 (x)
  - dim=1: reconstruct along axis 2 (y)  -- NEW
  - dim=2: reconstruct along axis 3 (z)
  - For WENO5-Z and PLM, the stencil operation is 1D along the chosen axis
```

**Phase 5: HLLD kernel 3D dispatch (~60 LOC)**

The HLLD Metal kernel currently dispatches over `(n_ifaces, nz)`. For 3D:
- dim=0 (x-sweep): interfaces have shape `(NVAR, n_ifaces_x, ny, nz)` -- flatten ny*nz as transverse
- dim=1 (y-sweep): interfaces have shape `(NVAR, nx, n_ifaces_y, nz)` -- transpose to `(NVAR, n_ifaces_y, nx*nz)`
- dim=2 (z-sweep): interfaces have shape `(NVAR, nx, ny, n_ifaces_z)` -- transpose to `(NVAR, n_ifaces_z, nx*ny)`

The existing transpose pattern in `compute_fluxes()` at `mlx_riemann.py:236-241` already handles dim=1 by transposing axes. Extending to dim=2 follows the same pattern.

**Phase 6: Ghost cell BCs for Cartesian (~50 LOC)**

```
File: src/dpf/metal/mlx_kernels.py (or new mlx_bc.py)
New function: ghost_pad_cartesian_mlx(U, ng, bc_type)
  - bc_type = "outflow": zero-gradient (copy boundary values)
  - bc_type = "periodic": wrap around
  - Applied per dimension before reconstruction
```

**Phase 7: CFL update (~15 LOC)**

```
File: src/dpf/metal/mlx_timestepper.py
Function: compute_dt_cfl()
Changes:
  - Add y-direction fast magnetosonic speed to CFL criterion
  - dt = cfl * min(dx/cf_x, dy/cf_y, dz/cf_z)
```

#### 1.4.2 Estimated Effort

| Component | New LOC | Modified LOC | Files Touched |
|-----------|---------|-------------|---------------|
| CartesianGrid class | 60 | 0 | mlx_grid.py |
| Solver coordinate branching | 20 | 30 | mlx_solver.py |
| 3D mhd_rhs + dim=2 sweep | 80 | 40 | mlx_riemann.py |
| dim=2 reconstruction | 20 | 10 | mlx_reconstruction.py |
| HLLD 3D dispatch (flatten) | 30 | 30 | mlx_riemann.py |
| Cartesian ghost pad | 50 | 0 | mlx_kernels.py or mlx_bc.py |
| CFL 3D update | 10 | 5 | mlx_timestepper.py |
| **Total** | **~270** | **~115** | **5-6 files** |

Estimated effort: 4-6 hours implementation, 2-3 hours testing.

### 1.5 Control

#### 1.5.1 Test Plan

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| `test_mlx_cartesian_instantiation` | Create MLXMHDSolver with coordinates="cartesian", grid (16,16,16) | No error, solver.coordinates == "cartesian" |
| `test_mlx_cartesian_uniform_preservation` | Step uniform state, verify no change | max(abs(U_new - U_old)) < 1e-6 |
| `test_mlx_sod_3d` | 3D Sod shock tube along x-axis, ny=nz=4 | L1(rho) < 0.05 vs exact Riemann solution |
| `test_mlx_brio_wu_3d` | 3D Brio-Wu MHD shock along x-axis | No NaN, compound wave structure visible |
| `test_mlx_orszag_tang_2d` | Orszag-Tang vortex on (128,128,1) grid | Energy conservation dE/E < 1e-4 at t=0.5 |
| `test_mlx_cartesian_conservation` | Mass, momentum, energy conservation | Relative drift < 1e-6 over 100 steps |
| `test_mlx_cartesian_vs_metal` | Compare MLX Cartesian vs MetalMHDSolver(cartesian) on Sod | L1(rho) parity < 5% |
| `test_mlx_cartesian_periodic_bc` | Advection test with periodic BCs | Wave returns to initial position |

#### 1.5.2 Monitoring

- div(B) tracked per step (should be ~0 for Cartesian without CT; Dedner/Powell for cleanup)
- Conservation monitor: mass, momentum, energy totals reported at each step

---

## Item 2: Dedner/Powell Divergence Cleaning for MLX

### 2.1 Define

**Problem Statement**: The MLX solver uses constrained transport (CT) for div(B) = 0 enforcement in cylindrical geometry (`mlx_ct.py`). However, CT is limited to 2D axisymmetric configurations. For 3D Cartesian support and as an alternative to CT in cylindrical mode, Dedner GLM divergence cleaning and/or Powell 8-wave source terms are needed. Both methods exist in the Python engine (`mhd_solver.py:949-1142`) but have never been ported to MLX.

**Customer Requirement**: Feature parity with the Python MHD solver's Dedner + Powell divergence cleaning, usable in both Cartesian 3D and cylindrical geometries.

### 2.2 Measure

#### 2.2.1 Existing Implementation in Python Engine

**Dedner GLM (original, Dedner et al. 2002)**:
- `src/dpf/fluid/mhd_solver.py:949-986` -- `_dedner_source(psi, B, ch, cp, dx)`
- Governing equations:
  ```
  dpsi/dt = -ch^2 * div(B) - (ch^2 / cp^2) * psi
  dB/dt  += -grad(psi)
  ```
- `ch`: hyperbolic cleaning speed (propagation speed of divergence errors)
- `cp`: parabolic damping speed (controls psi decay rate)

**Dedner GLM (Mignone & Tzeferacos 2010 tuning)**:
- `src/dpf/fluid/mhd_solver.py:989-1033` -- `_dedner_source_mt2010(psi, B, ch, cr, dx)`
- Improved formulation with separate damping coefficient:
  ```
  dpsi/dt = -ch^2 * div(B) - cr * psi
  dB/dt  += -grad(psi)
  ```
- `cr = ch / dx` is the M&T2010 optimal damping rate prescription

**Powell 8-wave source terms**:
- `src/dpf/fluid/mhd_solver.py:1040-1099` -- `powell_source_terms(state, dx, dy, dz)`
- Source vector (conservative variables):
  ```
  S_Powell = -div(B) * [0, B_x, B_y, B_z, v.B, v_x, v_y, v_z, 0]^T
  ```
- Proportionally corrects momentum, energy, and induction equations

**Powell cylindrical variant**:
- `src/dpf/fluid/mhd_solver.py:1102-1142` -- `powell_source_terms_cylindrical(state_2d, geom)`
- Uses cylindrical div(B): `(1/r) * d(r*Br)/dr + dBz/dz`

**Integration point in Python engine**:
- `src/dpf/fluid/mhd_solver.py:1876-1896` -- Dedner is applied within `_euler_stage()`:
  - Skipped when `self.use_ct = True` (CT and Dedner are mutually exclusive)
  - `ch` auto-computed from max fast magnetosonic speed: `ch = max(|v| + c_f)`
  - `cr = ch / dx` (M&T2010 prescription)
  - `psi` is evolved alongside B in the RK stages

**Integration in RK stepping**:
- `src/dpf/fluid/mhd_solver.py:2155-2249` -- SSP-RK3 evolves `psi` alongside all other variables:
  ```python
  psi_1 = psi_n + dt * rhs1["dpsi_dt"]
  psi_2 = 0.75 * psi_n + 0.25 * psi_1  # (after second stage)
  psi_new = (1/3) * psi_n + (2/3) * psi_3e  # (after third stage)
  ```

#### 2.2.2 Current MLX div(B) Strategy

- `src/dpf/metal/mlx_ct.py` -- CT using face-centred B-fields and corner EMF
- `src/dpf/metal/mlx_solver.py:619-720` -- `_apply_ct_correction()` called after RK step
- CT maintains div(B) = 0 to machine precision in 2D axisymmetric geometry
- No psi variable exists in the MLX state vector (slot IEE is used for `e_electron`)
- No Dedner or Powell implementation exists in any MLX module

#### 2.2.3 Quantitative Gap

| Feature | MLX Solver | Python Engine | Gap |
|---------|-----------|--------------|-----|
| Dedner GLM psi evolution | Not implemented | `_dedner_source_mt2010` | Missing |
| Powell 8-wave sources | Not implemented | `powell_source_terms` | Missing |
| CT | Implemented (2D axisymmetric) | Implemented (optional) | Parity for 2D |
| div(B) monitor | Not implemented | Via Powell return dict | Missing |
| psi in state vector | No slot (IEE = e_electron) | Yes, in state dict | Design needed |

### 2.3 Analyze

#### 2.3.1 Literature Review

**Dedner et al. (2002)**, "Hyperbolic Divergence Cleaning for the MHD Equations", JCP 175:645-673.

The core idea: extend the MHD system with a scalar field psi that couples to div(B) via a generalized Lagrange multiplier (GLM). The modified induction equation becomes:

```
dB/dt + grad(psi) = curl(v x B)
dpsi/dt + ch^2 * div(B) = -(ch^2/cp^2) * psi
```

where `ch` is the fastest signal speed (typically max fast magnetosonic speed) and `cp` controls parabolic damping. The system is hyperbolic: divergence errors propagate at speed `ch` and are damped at rate `ch^2/cp^2`.

**Key properties**:
- psi acts as a 9th conserved variable (added to the flux, not as a source term)
- The Bn flux gains a psi contribution: `F_Bn = psi` (at interfaces)
- The psi flux gains a Bn contribution: `F_psi = ch^2 * Bn`
- This makes div(B) errors propagate out of the domain at speed ch
- The damping term `-(ch^2/cp^2)*psi` is a source term applied after fluxes

**Mignone & Tzeferacos (2010)**, "A Second-order Unsplit Godunov Scheme for Cell-centered MHD", JCP 229:5896-5920.

Simplified tuning: replace `ch^2/cp^2` with a single damping rate `cr`:
```
dpsi/dt = -ch^2 * div(B) - cr * psi
```
where `cr = ch / dx` gives optimal damping (divergence errors decay in ~1 cell crossing time).

**Powell et al. (1999)**, "A Solution-Adaptive Upwind Scheme for Ideal Magnetohydrodynamics", JCP 154:284-309.

Alternative approach: add source terms proportional to div(B) to ALL equations:
```
drho/dt += 0
d(rho*v)/dt += -div(B) * B
dE/dt += -div(B) * (v . B)
dB/dt += -div(B) * v
```

**Tradeoffs**:
- Powell terms are NOT conservative (they add energy proportional to div(B))
- But they prevent div(B) from growing by feeding it back as a restoring force
- Simpler to implement than Dedner (no extra variable, no flux modification)
- Can be combined with Dedner for maximum robustness

**Comparison: CT vs Dedner vs Powell**:

Referencing Toth (2000) "The div(B)=0 Constraint in Shock-Capturing MHD", JCP 161:605-652:

| Method | div(B) Control | Conservative? | Extra Variable | Complexity | Best For |
|--------|---------------|---------------|----------------|------------|----------|
| CT | Machine precision | Yes | None (staggered B) | High | 2D structured |
| Dedner GLM | ~dx^2 (2nd order) | Yes (with flux form) | psi (scalar) | Medium | 3D, AMR |
| Powell 8-wave | Controlled growth | No | None | Low | Quick fix |
| Dedner + Powell | ~dx^2 + restoring | Partially | psi | Medium | Maximum robustness |

**Recommendation for DPF/MLX**: Implement Dedner GLM (M&T2010 tuning) as the primary method. Powell is optional but easy to add as a bonus. CT remains the default for cylindrical 2D axisymmetric runs.

#### 2.3.2 Root Cause

MLX uses CT because the Metal solver used CT, and Phase B replicated that strategy. Dedner/Powell were in the Python engine from Phase D but were never ported because:
1. CT was sufficient for the 2D axisymmetric DPF use case
2. Dedner requires an extra state variable (psi) which complicates the 10-variable layout
3. No 3D Cartesian support existed to motivate Dedner

#### 2.3.3 State Variable Design Decision

The current MLX conservative variable ordering has 10 slots:
```
[rho, rho*vr, rho*vz, rho*vt, E, S*rho, Br, Bz, Btheta, e_electron]
 IDN   IMR     IMZ     IMT    IEN  ISR   IBR  IBZ   IBT      IEE
```

Options for adding psi:
1. **Expand to NVAR=11**: Add `IPSI = 10` at the end. Cleanest but requires updating all kernel dispatch sizes, strides, and reconstruction loops.
2. **Overload IEE**: Use slot 9 for psi when e_electron is not active. Risk: cannot use dual-energy and Dedner simultaneously.
3. **Separate psi array**: Evolve psi outside the main state vector as a side-car. The flux contributions are added manually.

**Recommendation**: Option 3 (separate psi array). Rationale:
- No kernel changes needed for NVAR
- psi only participates in the induction equation flux and its own evolution
- The M&T2010 formulation treats the damping term as a source, not a flux
- The flux modification (adding psi to Bn flux) can be applied post-reconstruction

### 2.4 Improve

#### 2.4.1 Governing Equations (Dedner GLM, M&T2010)

The modified MHD system in conservative form:

**Induction equation** (modified):
```
dBx/dt = d(vy*Bx - vx*By)/dy + d(vz*Bx - vx*Bz)/dz - dpsi/dx
dBy/dt = d(vx*By - vy*Bx)/dx + d(vz*By - vy*Bz)/dz - dpsi/dy
dBz/dt = d(vx*Bz - vz*Bx)/dx + d(vy*Bz - vz*By)/dy - dpsi/dz
```

**psi evolution**:
```
dpsi/dt = -ch^2 * div(B) - cr * psi
```

where:
- `ch = max(|v| + c_f)` over all cells (fastest signal speed)
- `cr = ch / dx` (M&T2010 optimal damping)
- `c_f` = fast magnetosonic speed

**Flux form**: The psi terms can be incorporated into the Riemann flux:
```
F_Bn += psi        (at each interface, add psi to the normal-B flux)
F_psi = ch^2 * Bn  (psi flux through each interface)
```

This makes the Dedner system hyperbolic and compatible with the existing Riemann solver framework.

#### 2.4.2 Powell Source Terms (all 8 MHD equations)

In conservative variable form with div(B) computed from the current state:

| Variable | Powell Source Term |
|----------|--------------------|
| rho (mass) | 0 |
| rho*vx (x-momentum) | `-div(B) * Bx` |
| rho*vy (y-momentum) | `-div(B) * By` |
| rho*vz (z-momentum) | `-div(B) * Bz` |
| E (total energy) | `-div(B) * (v . B)` |
| S*rho (entropy tracer) | 0 |
| Bx (x-induction) | `-div(B) * vx` |
| By (y-induction) | `-div(B) * vy` |
| Bz (z-induction) | `-div(B) * vz` |
| e_electron | 0 |

For cylindrical coordinates, div(B) = `(1/r) * d(r*Br)/dr + dBz/dz + (1/r) * dBtheta/dtheta`, where the last term vanishes for axisymmetry.

#### 2.4.3 Implementation Plan

**Approach**: Implement Dedner as an operator-split source term (not modifying the HLLD flux). This is simpler and matches the Python engine's approach.

**Step 1: Dedner source function (~60 LOC)**

```
File: src/dpf/metal/mlx_divb.py (NEW)

def dedner_source_mlx(
    psi: mx.array,            # shape (nr, nz) or (nx, ny, nz)
    B: tuple[mx.array, ...],  # (Br, Bz, Bt) or (Bx, By, Bz)
    ch: float,                # cleaning speed
    cr: float,                # damping rate = ch / dx
    dr: float, dz: float,     # grid spacings
    dy: float = 0.0,          # for 3D
    coordinates: str = "cylindrical",
    r_cell: mx.array | None = None,
) -> tuple[mx.array, tuple[mx.array, ...]]:
    """Dedner GLM divergence cleaning (M&T2010).

    Returns (dpsi_dt, dB_corrections).
    """
    # Compute div(B) using central differences
    if coordinates == "cylindrical":
        # (1/r) * d(r*Br)/dr + dBz/dz
        ...
    else:
        # dBx/dx + dBy/dy + dBz/dz
        ...

    dpsi_dt = -ch**2 * div_B - cr * psi

    # grad(psi) correction to B
    dBr_dt = -gradient(psi, dr, axis=0)
    dBz_dt = -gradient(psi, dz, axis=1)  # or axis=2 for 3D

    return dpsi_dt, (dBr_dt, dBz_dt, ...)
```

**Step 2: Powell source function (~40 LOC)**

```
File: src/dpf/metal/mlx_divb.py (same file)

def powell_source_mlx(
    U: mx.array,          # conserved state
    gamma: float,
    dr: float, dz: float,
    dy: float = 0.0,
    coordinates: str = "cylindrical",
    r_cell: mx.array | None = None,
) -> mx.array:
    """Powell 8-wave source terms for div(B) control.

    Returns source array same shape as U.
    """
```

**Step 3: Integration into RK stepping (~30 LOC)**

```
File: src/dpf/metal/mlx_timestepper.py
Function: _rk_stage() or ssp_rk3_step()
Changes:
  - Accept optional psi array
  - After computing L(U), apply Dedner correction to B components
  - Evolve psi alongside U in each RK stage
  - Compute ch from max fast magnetosonic speed in compute_dt_cfl
```

**Step 4: Solver integration (~20 LOC)**

```
File: src/dpf/metal/mlx_solver.py
Changes:
  - Add enable_dedner: bool parameter (default: False for cylindrical/CT, True for Cartesian)
  - Maintain psi array as instance variable (separate from U)
  - Pass psi through step() pipeline
  - Option to use Dedner instead of CT when use_ct=False
```

#### 2.4.4 Where in the RK Pipeline

**Dedner (flux modification form)**: Apply after reconstruction, before Riemann solve:
1. Reconstruct UL, UR at interfaces
2. Add psi to normal-B flux: `F_Bn += psi` at interface
3. Solve Riemann problem (HLLD/HLL)
4. After flux divergence, apply psi source: `dpsi_dt = -ch^2 * div(B) - cr * psi`

**Dedner (source term form, recommended)**: Apply as operator-split source after each RK stage:
1. Compute L(U) via standard mhd_rhs (no psi in flux)
2. Update U via RK combination
3. Apply Dedner correction: `B -= dt * grad(psi)`, `psi += dt * (-ch^2 * div(B) - cr * psi)`

The source-term form is recommended because:
- Does not require modifying the HLLD kernel
- Matches the Python engine's implementation
- Operator-split Dedner is 2nd-order accurate when applied symmetrically (Strang splitting)

**Powell**: Apply as source terms within `mhd_rhs`, added to the flux-divergence result. This is the same location as geometric source terms.

#### 2.4.5 Estimated Effort

| Component | New LOC | Modified LOC | Files |
|-----------|---------|-------------|-------|
| `mlx_divb.py` (new) | 120 | 0 | 1 new |
| RK pipeline integration | 30 | 20 | mlx_timestepper.py |
| Solver wiring | 20 | 15 | mlx_solver.py |
| CFL update for ch | 5 | 5 | mlx_timestepper.py |
| **Total** | **~175** | **~40** | **3 files** |

Estimated effort: 3-4 hours implementation, 1-2 hours testing.

### 2.5 Control

#### 2.5.1 Test Plan

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| `test_mlx_dedner_uniform` | Uniform state + psi=0: no change | max(abs(dpsi_dt)) < 1e-12 |
| `test_mlx_dedner_monopole_decay` | Initialize psi = Gaussian, B = 0: psi decays | max(psi) < 0.01 * initial after 100 steps |
| `test_mlx_dedner_divb_reduction` | Initialize non-zero div(B): measure reduction | div(B) decreases by >10x in 50 steps |
| `test_mlx_powell_uniform` | Uniform B: Powell sources = 0 | max(abs(S_powell)) < 1e-12 |
| `test_mlx_dedner_vs_ct` | Compare div(B) for cylindrical problem | Both < 1e-8 after 100 steps |
| `test_mlx_dedner_cartesian_3d` | Orszag-Tang 2D: div(B) stays bounded | max(abs(div(B))) < 1e-4 |

---

## Item 3: Braginskii Viscosity for MLX

### 3.1 Define

**Problem Statement**: The MLX solver has `enable_braginskii_viscosity` as a constructor parameter (`mlx_solver.py:117`) but the docstring explicitly states "not yet wired -- future use" (`mlx_solver.py:88`). Braginskii viscosity is fully implemented in the PyTorch Metal solver via `apply_braginskii_viscosity_mps()` in `metal_transport.py:396-574` and is called from `metal_solver.py:1373-1378`. The MLX solver needs a port of this physics.

**Customer Requirement**: Feature parity for viscous transport physics between MLX and Metal backends, enabling viscous DPF simulations on the MLX solver.

### 3.2 Measure

#### 3.2.1 Existing PyTorch Metal Implementation

**Location**: `src/dpf/metal/metal_transport.py:396-574`

**Function signature**:
```python
def apply_braginskii_viscosity_mps(
    velocity: torch.Tensor,       # (3, nx, ny, nz)
    rho: torch.Tensor,            # (nx, ny, nz)
    pressure: torch.Tensor,       # (nx, ny, nz)
    B: torch.Tensor,              # (3, nx, ny, nz)
    Ti: torch.Tensor,             # (nx, ny, nz)
    dt: float,
    dx: float, dy: float, dz: float,
    ion_mass: float = M_D,
    full_braginskii: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**Physics implemented**:

1. **Strain rate tensor** (lines 446-463):
   ```
   S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)
   S_trace = S_xx + S_yy + S_zz
   ```

2. **Ion collision time** (NRL Formulary shorthand, lines 476-486):
   ```
   tau_i = 2.09e7 * Ti_eV^{3/2} * sqrt(A) / (ni_cm3 * Z^4 * lnL)  [s]
   ```
   where Coulomb log `lnL = clamp(23 - ln(sqrt(ni_cm3) / Ti_eV^1.5), min=2)`

3. **Parallel viscosity coefficient** (line 488-489):
   ```
   eta_0 = 0.96 * n_i * k_B * T_i * tau_i
   ```

4. **Full anisotropic mode** (`full_braginskii=True`, lines 491-525):
   - Gyroviscosity: `eta_1 = 0.3 * ni * kB * Ti / (omega_ci^2 * tau_i)`
   - Parallel projection: `W0_ij = (b_i*b_j - delta_ij/3) * (b_k*b_l*S_kl)`
   - Stress tensor: `sigma_ij = eta_0 * W0_ij + eta_1 * (S_ij - W0_ij)`

5. **Isotropic mode** (`full_braginskii=False`, lines 526-533):
   - Traceless stress: `sigma_ij = eta_0 * (S_ij - delta_ij * S_trace / 3)`

6. **Viscous acceleration** (lines 554-558):
   ```
   a_i = div(sigma)_i / rho
   v_new = v + dt * a
   ```

7. **Viscous heating** (lines 560-568):
   ```
   Q_visc = sigma_ij * S_ij  (full tensor contraction)
   p_new = p + dt * (gamma - 1) * Q_visc
   ```

**Integration point in Metal solver**:
- `src/dpf/metal/metal_solver.py:1373-1378` -- Called as operator-split step AFTER the hyperbolic RK advance and AFTER resistive diffusion, within the transport physics block.

#### 3.2.2 MLX Transport Module Status

- `src/dpf/metal/mlx_transport.py` -- Contains `apply_resistive_diffusion()` and `apply_thermal_conduction()` using the Thomas tridiagonal solver on CPU in float64.
- No viscosity function exists in this module.
- The thermal conduction is wired in `mlx_solver.py:452-496` (`_do_thermal_conduction`).
- The resistive diffusion is wired in `mlx_solver.py:399-446` (`_do_resistive_diffusion`).

#### 3.2.3 Quantitative Gap

| Feature | MLX Solver | Metal (PyTorch) Solver | Gap |
|---------|-----------|----------------------|-----|
| Parallel viscosity (eta_0) | Flag stored, not wired | Implemented | Missing |
| Gyroviscosity (eta_1) | Not implemented | Implemented (full_braginskii) | Missing |
| Viscous stress tensor | Not implemented | Full 6-component symmetric | Missing |
| Viscous heating | Not implemented | sigma_ij * S_ij contraction | Missing |
| CFL viscous limit | Not implemented | Not explicit (relies on dt_MHD) | Both need it |

### 3.3 Analyze

#### 3.3.1 Physics Reference

**Braginskii (1965)**, Reviews of Plasma Physics, Vol. 1, pp. 205-311.

The viscous stress tensor for a magnetized plasma has 5 independent coefficients (eta_0 through eta_4). For the ordering relevant to DPF (strongly magnetized ions, `omega_ci * tau_i >> 1`):

- **eta_0** (parallel viscosity): `eta_0 = 0.96 * n_i * k_B * T_i * tau_i`
  - Dominates along the magnetic field direction
  - Independent of B-field strength
  - This is the largest coefficient

- **eta_1** (perpendicular viscosity): `eta_1 = 3 * n_i * k_B * T_i / (10 * omega_ci^2 * tau_i)`
  - Perpendicular to B, proportional to `1/(omega_ci^2 * tau_i)`
  - Small in strongly magnetized regime

- **eta_2** = `4 * eta_1` (related perpendicular coefficient)

- **eta_3, eta_4** (gyroviscous): `~ n_i * k_B * T_i / omega_ci`
  - Odd in B-field direction (gyro-viscosity)
  - Important for drift ordering but often neglected in MHD

**For DPF conditions** (T_i ~ 1-10 keV, B ~ 0.1-10 T, n ~ 10^24-10^26 m^-3):
- `omega_ci * tau_i ~ 10-100` (moderately to strongly magnetized)
- `eta_0 / eta_1 ~ (omega_ci * tau_i)^2 ~ 100-10000`
- **Parallel viscosity dominates** by orders of magnitude

**Simplification for MLX implementation**: In the initial port, implementing only the isotropic parallel viscosity (eta_0) is sufficient. The full anisotropic tensor can be added as `full_braginskii=True` option matching the Metal solver.

#### 3.3.2 Axisymmetric Stress Tensor

For the (r, z) axisymmetric geometry used by the MLX cylindrical solver, the strain rate tensor simplifies. With v = (vr, vtheta, vz) and derivatives only in r and z:

```
S_rr = dvr/dr
S_zz = dvz/dz
S_tt = vr / r                               (geometric term)
S_rz = 0.5 * (dvr/dz + dvz/dr)
S_rt = 0.5 * (dvtheta/dr - vtheta/r)        (geometric term)
S_tz = 0.5 * dvtheta/dz
```

The trace: `S_trace = S_rr + S_tt + S_zz = dvr/dr + vr/r + dvz/dz`

In the isotropic (traceless) limit:
```
sigma_ij = eta_0 * (S_ij - delta_ij * S_trace / 3)
```

The divergence of the stress tensor in cylindrical coordinates:
```
(div sigma)_r = dsigma_rr/dr + dsigma_rz/dz + (sigma_rr - sigma_tt) / r
(div sigma)_z = dsigma_rz/dr + dsigma_zz/dz + sigma_rz / r
(div sigma)_t = dsigma_rt/dr + dsigma_tz/dz + 2 * sigma_rt / r
```

Note the geometric terms `1/r` in the divergence -- these are analogous to the cylindrical geometric source terms in the momentum equation.

#### 3.3.3 Stability Constraint

The explicit viscous diffusion timestep:
```
dt_visc < dx^2 * rho / (2 * D * eta_visc)
```
where `D` is the number of spatial dimensions (2 for axisymmetric, 3 for 3D).

For DPF conditions:
- `eta_0 ~ 0.96 * n * kB * T * tau_i`
- With n ~ 10^25 m^-3, T ~ 1 keV, tau_i ~ 10^-9 s: eta_0 ~ 1.3 Pa.s
- With rho ~ 3.3e-2 kg/m^3, dx ~ 1e-3 m: dt_visc ~ dx^2 * rho / (4 * eta_0) ~ 6e-9 s
- Typical MHD CFL timestep: dt_MHD ~ 1e-9 to 1e-8 s

This means viscous CFL is often comparable to or more restrictive than MHD CFL at DPF temperatures. **Sub-cycling may be needed** when `dt_visc < dt_MHD`.

#### 3.3.4 Root Cause

The MLX solver's `enable_braginskii_viscosity` flag was added as a placeholder during Phase B construction. Thermal conduction was prioritized because it has a stronger effect on DPF electron temperature evolution. Viscosity was deferred because:
1. The viscous stress tensor requires 6 components (more complex than scalar conduction)
2. Cylindrical geometry adds extra geometric terms to the divergence
3. The viscous CFL constraint could reduce the timestep significantly

### 3.4 Improve

#### 3.4.1 Implementation Plan

**Step 1: Viscosity coefficients (~40 LOC)**

```
File: src/dpf/metal/mlx_transport.py
New function: braginskii_viscosity_coefficients(rho, Ti, B_mag, ion_mass)
Returns: eta_0, eta_1 (or just eta_0 for isotropic mode)

Uses NRL Formulary shorthand for tau_i (same as Metal solver):
  tau_i = 2.09e7 * Ti_eV^1.5 * sqrt(A) / (ni_cm3 * lnL)
  eta_0 = 0.96 * ni * kB * Ti * tau_i
```

**Step 2: Strain rate computation (~60 LOC)**

```
File: src/dpf/metal/mlx_transport.py
New function: compute_strain_rate_cylindrical(vr, vz, vt, dr, dz, r_cell)
Returns: S_rr, S_zz, S_tt, S_rz, S_rt, S_tz, S_trace

Uses central differences via mx operations.
Geometric terms (vr/r, vtheta/r) use inv_r with L'Hopital at axis.
```

For Cartesian 3D:
```
New function: compute_strain_rate_cartesian(vx, vy, vz, dx, dy, dz)
Returns: S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, S_trace
```

**Step 3: Stress tensor + divergence (~80 LOC)**

```
File: src/dpf/metal/mlx_transport.py
New function: apply_braginskii_viscosity_mlx(
    U, dt, dr, dz, r_cell, inv_r, gamma, ion_mass,
    coordinates="cylindrical", full_braginskii=False
)
Returns: Updated U with viscous acceleration and heating.

Pipeline:
  1. Extract primitives: rho, vr, vz, vt, p, B from U
  2. Compute Ti from p, rho, ion_mass
  3. Compute eta_0 (and eta_1 if full)
  4. Compute strain rate tensor
  5. Build stress tensor (isotropic or anisotropic)
  6. Compute div(sigma) in cylindrical or Cartesian coordinates
  7. Update velocity: v_new = v + dt * div(sigma) / rho
  8. Compute viscous heating: Q = sigma_ij * S_ij
  9. Update energy: E_new = E + dt * Q (or equivalently p_new += dt * (gamma-1) * Q)
  10. Rebuild conserved state U from updated primitives
```

**Step 4: Wiring into solver (~20 LOC)**

```
File: src/dpf/metal/mlx_solver.py
Function: step()
Changes:
  - After resistive diffusion (step 5) and thermal conduction (step 6):
  - if self.enable_braginskii_viscosity:
        U = self._do_braginskii_viscosity(U, dt)
```

**Step 5: Sub-cycling safety (~15 LOC)**

```
File: src/dpf/metal/mlx_transport.py
Add viscous CFL check:
  dt_visc = dx^2 * min(rho) / (2 * D * max(eta_0))
  if dt > dt_visc:
      n_sub = min(ceil(dt / dt_visc), 20)
      dt_sub = dt / n_sub
      for _ in range(n_sub):
          apply viscous step with dt_sub
```

This matches the sub-cycling approach already used for resistive diffusion in the MLX solver.

#### 3.4.2 Estimated Effort

| Component | New LOC | Modified LOC | Files |
|-----------|---------|-------------|-------|
| Viscosity coefficients | 40 | 0 | mlx_transport.py |
| Strain rate (cylindrical + Cartesian) | 80 | 0 | mlx_transport.py |
| Stress tensor + divergence | 80 | 0 | mlx_transport.py |
| Viscous heating | 20 | 0 | mlx_transport.py |
| Sub-cycling | 15 | 0 | mlx_transport.py |
| Solver wiring | 20 | 5 | mlx_solver.py |
| **Total** | **~255** | **~5** | **2 files** |

Estimated effort: 4-5 hours implementation, 2-3 hours testing.

### 3.5 Control

#### 3.5.1 Test Plan

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| `test_mlx_viscosity_coefficients` | eta_0 for T=1keV, n=1e25: compare with NRL Formulary | Relative error < 1% |
| `test_mlx_viscosity_uniform_no_change` | Uniform velocity field: no viscous force | max(abs(dv)) < 1e-10 |
| `test_mlx_viscous_shock_tube` | Viscous Sod problem: shock width broadens | Shock width > inviscid case |
| `test_mlx_viscous_heating` | Shear flow: Q_visc > 0 | Energy increases monotonically |
| `test_mlx_viscosity_vs_metal` | Same initial conditions, compare vel/p after 10 steps | L1 parity < 5% |
| `test_mlx_viscous_cfl` | Large eta_0: verify sub-cycling activates | n_subcycles > 1 for eta_0 > dt*2*D/dx^2*rho |
| `test_mlx_viscosity_cylindrical` | Axisymmetric shear: geometric terms correct | Compare with analytical Taylor-Couette |
| `test_mlx_viscosity_energy_conservation` | Total energy (KE + thermal) conserved | dE_total/E < 1e-6 (viscosity redistributes, not creates) |

---

## Cross-Item Dependencies

```
Item 1 (Cartesian 3D)  ──depends-on──>  Item 2 (Dedner)
   [3D needs div(B) cleaning since         [Dedner is the preferred
    CT is 2D-only in MLX]                   method for 3D]

Item 1 (Cartesian 3D)  ──enables──>  Item 3 (Viscosity, Cartesian path)
   [Cartesian strain rate uses              [compute_strain_rate_cartesian
    standard partial derivatives]            only makes sense with 3D grid]
```

**Recommended implementation order**:
1. **Item 3 (Braginskii Viscosity)** -- No dependencies on other items. Can be implemented and tested in the existing cylindrical framework immediately.
2. **Item 2 (Dedner/Powell)** -- No hard dependency on Item 1, but most useful with it. Can be tested in cylindrical mode as alternative to CT.
3. **Item 1 (Cartesian 3D)** -- Benefits from Item 2 being available for div(B) cleaning. Largest scope.

---

## Risk Register

| ID | Risk | Probability | Impact | Mitigation |
|----|------|------------|--------|------------|
| R1 | HLLD 3D dispatch NaN in float32 | Medium | High | Flatten transverse dims; fallback to HLL for 3D |
| R2 | Viscous CFL too restrictive | Medium | Medium | Sub-cycling with cap at 20 (proven pattern from resistive diffusion) |
| R3 | Dedner psi oscillation in float32 | Low | Medium | cr = ch/dx damping (M&T2010); P_FLOOR on psi |
| R4 | CartesianGrid + CylindricalGrid API divergence | Low | Low | Define common `GridBase` protocol with shared attributes |
| R5 | State variable layout change (NVAR=11) | N/A | N/A | Mitigated: psi as side-car array, no NVAR change |
| R6 | Cylindrical viscosity geometric terms wrong sign | Medium | High | Compare with Athena++ `hydro/hydro_diffusion/` implementation |
| R7 | Performance regression from 3D (3x flux sweeps) | Certain | Low | Expected ~50% slowdown vs 2D; acceptable for correctness |

---

## Summary of Deliverables

| Item | New Files | Modified Files | Total New LOC | Effort (h) |
|------|-----------|----------------|---------------|------------|
| 1. Cartesian 3D | 0-1 (mlx_bc.py) | mlx_solver.py, mlx_grid.py, mlx_riemann.py, mlx_reconstruction.py, mlx_timestepper.py | ~270 | 6-9 |
| 2. Dedner/Powell | 1 (mlx_divb.py) | mlx_solver.py, mlx_timestepper.py | ~175 | 4-6 |
| 3. Braginskii Viscosity | 0 | mlx_transport.py, mlx_solver.py | ~255 | 6-8 |
| **Total** | **1-2** | **6-7** | **~700** | **16-23** |

---

## References

1. Borges R., Carmona M., Costa B., Don W.S., "An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws", JCP 227:3191 (2008).
2. Braginskii S.I., "Transport Processes in a Plasma", Reviews of Plasma Physics Vol. 1:205 (1965).
3. Dedner A., Kemm F., Kroener D., Munz C.-D., Schnitzer T., Wesenberg M., "Hyperbolic Divergence Cleaning for the MHD Equations", JCP 175:645 (2002).
4. Gardiner T.A. & Stone J.M., "An unsplit Godunov method for ideal MHD via constrained transport", JCP 205:509 (2005).
5. Mignone A. & Tzeferacos P., "A second-order unsplit Godunov scheme for cell-centred MHD: The CTU-GLM scheme", JCP 229:5896 (2010).
6. Miyoshi T. & Kusano K., "A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics", JCP 208:315 (2005).
7. Popovas A., Nordlund A., Ramsey J.P., "DISPATCH HLLS -- an entropy-stable method for MHD in float32", arXiv:2211.02438 (2025).
8. Powell K.G., Roe P.L., Linde T.J., Gombosi T.I., De Zeeuw D.L., "A Solution-Adaptive Upwind Scheme for Ideal Magnetohydrodynamics", JCP 154:284 (1999).
9. Shu C.-W. & Osher S., "Efficient implementation of essentially non-oscillatory shock-capturing schemes", JCP 77:439 (1988).
10. Stone J.M., Tomida K., White C.J., Felker K.G., "The Athena++ adaptive mesh refinement framework for astrophysical magnetohydrodynamics", ApJS 249:4 (2020).
11. Toth G., "The div(B)=0 Constraint in Shock-Capturing Magnetohydrodynamics Codes", JCP 161:605 (2000).
