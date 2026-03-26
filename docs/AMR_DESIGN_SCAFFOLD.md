# AMR Design Scaffold for DPF-Unified

Design-only document. No implementation accompanies this file.

**Author**: Engine Architect Agent
**Date**: 2026-03-26
**Status**: DRAFT -- requires review before Phase implementation
**Prerequisite reading**: `src/dpf/experimental/static_refinement.py` (562 LOC, existing SMR)

---

## 1. Architecture Options

### 1A. Patch-Based AMR (Berger-Oliger / AMReX style)

Arbitrary rectangular patches at each level. Patches can overlap and tile
irregularly. AMReX (LBNL) is the reference implementation.

- **Pros**: Maximum flexibility; fewest wasted cells; mature theory (Berger &
  Colella 1989).
- **Cons**: Complex bookkeeping (patch list management, load balancing); variable
  array shapes incompatible with `mx.compile()` static-shape requirement; patch
  creation/destruction forces Python-level memory allocation every regrid.

### 1B. Octree AMR

Cell-by-cell refinement organized in a tree. RAMSES (Teyssier 2002) is the
reference.

- **Pros**: Cell-level adaptivity; minimal wasted cells.
- **Cons**: Pointer-chasing memory access pattern is worst-case for GPU; not
  representable as contiguous arrays; 2D cylindrical is not a natural octree
  geometry.

### 1C. Block-Structured AMR (Athena++ / Parthenon style)

Fixed-size blocks (e.g., 32x64) organized into refinement levels. Each level is
a collection of identically-shaped blocks. Athena++ (Stone et al. 2020) and
Parthenon (Grete et al. 2022) use this approach.

- **Pros**: Every block is the same shape -- compatible with `mx.compile()`;
  batch all blocks at same level into a single MLX call via `mx.vmap()` or
  leading batch dimension; simple ghost exchange (fixed stencil offsets);
  maps directly onto existing `MLXMHDSolver` which already operates on
  `(NVAR, nr, nz)` arrays.
- **Cons**: Wastes cells at block boundaries (buffer zones); refinement
  granularity limited to block size; potential Amdahl bottleneck during regrid
  (Parthenon-VIBE 2025, arXiv:2509.19701).

### Recommendation: Block-Structured AMR

Block-structured is the only viable option for MLX. The static-shape constraint
of `mx.compile()` eliminates patch-based and octree. Block-structured also maps
naturally onto the existing solver: each block is a `(NVAR, nr_block, nz_block)`
array, identical to the current solver input. The Parthenon-VIBE Amdahl
bottleneck is mitigable at 2 levels (see Section 6).

---

## 2. Refinement Criteria

### Options Evaluated

| Criterion | Formula | Sensitivity to Sheath | False Positives | Cost |
|-----------|---------|----------------------|-----------------|------|
| Lohner (density) | `\|d2rho/dx2\| / (\|drho/dx\|/dx + eps*\|rho\|/dx^2)` | HIGH | Medium (any shock) | LOW |
| Current density | `\|curl B\| * dx / (\|B\| + eps)` | HIGHEST | Low | Medium |
| Pressure jump | `\|dp\|/p` across faces | Medium | High (acoustic waves) | LOW |
| div(B) error | `\|div B\| * dx / \|B\|` | Low | Irrelevant noise | LOW |
| Density gradient | `\|grad rho\| / rho` | High | Medium | LOW |

### Recommendation: Current Density (primary) + Lohner (secondary)

For DPF, the current density `|J| = |curl B|/mu_0` is the most physically
motivated sensor. The sheath IS the current sheet -- J peaks exactly where
refinement is needed. The Lohner indicator catches additional features (rarefaction
waves, contact discontinuities) that J misses.

Both sensors already exist in `static_refinement.py` (lines 321-441) as
`lohner_error_indicator()` and `current_density_sensor()`. The combined
criterion:

```python
def needs_refinement(block: BlockState, thresholds: tuple[float, float]) -> bool:
    j_sensor = current_density_sensor(block.B, block.dr, block.dz)
    lohner = lohner_error_indicator(block.rho, block.dr, block.dz)
    return np.max(j_sensor) > thresholds[0] or np.max(lohner) > thresholds[1]
```

Threshold tuning: `j_threshold = 0.3`, `lohner_threshold = 0.2` based on
typical DPF sheath profiles. These should be configurable.

---

## 3. Data Structure

### Block Definition

```python
@dataclass
class AMRBlock:
    """Single block in the AMR hierarchy."""
    level: int                    # 0 = coarsest
    index: tuple[int, int]        # (ir, iz) position in level grid
    U: mx.array                   # conserved state (NVAR, nr_block, nz_block)
    psi: mx.array | None          # Dedner cleaning scalar (nr_block, nz_block)
    r_min: float                  # physical left edge [m]
    z_min: float                  # physical bottom edge [m]
    dr: float                     # cell spacing at this level
    dz: float                     # cell spacing at this level
    parent: tuple[int, int] | None
    children: list[tuple[int, int]]  # up to 4 children (2x2 in r,z)
    needs_update: bool            # flagged for refinement/coarsening
```

### Level Container

```python
@dataclass
class AMRLevel:
    """All blocks at one refinement level."""
    level: int
    blocks: dict[tuple[int, int], AMRBlock]
    dr: float
    dz: float
    dt: float                     # timestep for this level (subcycled)

    def as_batch(self) -> mx.array:
        """Stack all blocks into (N_blocks, NVAR, nr_block, nz_block)."""
        return mx.stack([b.U for b in self.blocks.values()])
```

### 2-Level Layout (Phase A Target)

```
Level 0 (base):  4 blocks of (NVAR, 32, 64) covering full domain
Level 1 (fine):  1-4 blocks of (NVAR, 32, 64) covering sheath region
                 (factor-2 refinement: dr1 = dr0/2, dz1 = dz0/2)

Total cells:
  Level 0: 4 * 32 * 64 = 8,192
  Level 1: 4 * 32 * 64 = 8,192 (worst case, typically 1-2 blocks)
  Effective: equiv. to 128x256 at sheath, 64x128 elsewhere
```

Each block has ghost zones (2 cells for PLM, 3 for WENO5) filled from:
- Neighboring blocks at same level (copy)
- Parent block at coarser level (prolongation / interpolation)
- Physical boundaries (electrode BCs, axis, outflow)

### Memory Budget (M3 Pro, 36 GB)

```
Per block: NVAR * nr * nz * 4 bytes = 10 * 32 * 64 * 4 = 80 KB
8 blocks (2 levels): 640 KB
Ghost zones add ~25%: ~800 KB
Workspace (RHS, fluxes): 3x state = 2.4 MB
Total: < 4 MB -- negligible on 36 GB
```

---

## 4. Flux Conservation at Coarse-Fine Boundaries

At level boundaries, the coarse cell face sees a single flux, but the fine
cells compute 2 (or 4 in 3D) sub-fluxes along the same physical face. Without
correction, conservation fails at O(dx) -- the dominant error in AMR.

### Refluxing Algorithm (Berger & Colella 1989)

After advancing both levels, correct the coarse flux at each coarse-fine face:

```
delta_F = sum(F_fine * A_fine) - F_coarse * A_coarse
U_coarse[adjacent_cell] += delta_F * dt / V_coarse
```

For cylindrical coordinates, the face areas and cell volumes carry `r` factors:

```python
def reflux_correction_cylindrical(
    U_coarse: mx.array,     # (NVAR, nr_c, nz_c)
    F_coarse_r: mx.array,   # (NVAR, nr_c+1, nz_c) radial flux at faces
    F_fine_r: mx.array,     # (NVAR, nr_f+1, nz_f) fine radial flux
    r_face_coarse: mx.array, # radial face positions, coarse
    r_face_fine: mx.array,   # radial face positions, fine
    dt: float,
    dr_c: float,
    dz_c: float,
) -> mx.array:
    """Apply refluxing correction at coarse-fine radial boundary.

    For each coarse face that abuts fine blocks:
      dF = (r_f[j]*F_f[j]*dz_f + r_f[j+1]*F_f[j+1]*dz_f) - r_c*F_c*dz_c
      U_c[adjacent] += dF * dt / (r_c * dr_c * dz_c)
    """
    # Implementation: loop over boundary faces, accumulate fine fluxes,
    # subtract coarse flux, apply correction to adjacent coarse cell.
    # This is O(N_boundary) -- cheap relative to interior update.
    ...
```

The cylindrical `r` weighting is critical. Standard Cartesian refluxing
(AMReX default) does not account for `2*pi*r*dr*dz` volume elements. Our
solver already uses `r`-weighted finite volumes (see MEMORY.md:
"r-weighted finite volume > operator-split for cylindrical MHD"), so the
refluxing must be consistent.

### Constrained Transport at Boundaries

If CT is active, the magnetic flux through coarse-fine faces must also be
corrected to maintain div(B) = 0. This is the hardest part of AMR for MHD
and the reason most GPU MHD codes (including AthenaK) defer it.

For Phase A-B, use Dedner GLM cleaning (already implemented) instead of CT
at level boundaries. CT remains active within each block. This is the
approach used by MPI-AMRVAC (Keppens et al. 2023).

---

## 5. Subcycling

### The Question

Should the fine level (level 1, dt_fine = dt_coarse / 2) take 2 steps per
coarse step?

### Option A: Subcycling (standard)

```
t=0          t=dt_c/2       t=dt_c
L0: |------ full step ------|
L1: |-- half --|-- half --|
    ^          ^          ^
    sync       sync       sync + reflux
```

- **Pros**: CFL-correct on both levels; fine level resolves fast waves at
  fine scale; standard in Athena++, FLASH, AMReX.
- **Cons**: 2x the fine-level work; synchronization points add latency;
  on GPU, the fine blocks alone may underutilize the hardware.

### Option B: Global Timestep (no subcycling)

Both levels advance with `dt = min(dt_coarse, dt_fine)`.

- **Pros**: Simpler implementation; no synchronization; all blocks advance
  in one batched MLX call; avoids the Parthenon-VIBE Amdahl bottleneck.
- **Cons**: Coarse level takes unnecessarily small steps; overall speedup
  from AMR reduced (coarse level is CFL-limited by fine dt).

### Recommendation for DPF: Global Timestep (Phase A-B), Subcycling (Phase C+)

For 2-level AMR with factor-2, the coarse level only wastes a factor of 2 in
timestep. The implementation simplicity and GPU utilization advantages dominate.
At 3+ levels (Phase D), subcycling becomes necessary to avoid the coarse level
taking 4x or 8x smaller timesteps.

---

## 6. MLX Constraints and Mitigations

### Constraint 1: `mx.compile()` Requires Static Shapes

**Impact**: Cannot change the number of blocks or block size at runtime inside
a compiled function.

**Mitigation**: Pre-allocate a fixed maximum number of blocks per level.
Inactive blocks are masked with a boolean flag. The compiled kernel processes
all slots; inactive slots produce zero flux (branch-free masking).

```python
MAX_BLOCKS_PER_LEVEL = 16  # pre-allocated
active_mask: mx.array       # (MAX_BLOCKS_PER_LEVEL,) bool

# Batch update: all blocks at once, masked
U_batch = level.as_padded_batch()  # (MAX_BLOCKS, NVAR, nr, nz)
U_new = compiled_rhs(U_batch) * active_mask[:, None, None, None]
```

### Constraint 2: float32 Only on Metal GPU

**Impact**: Conservation at coarse-fine boundaries accumulates float32 rounding.
Refluxing correction adds O(eps_32) ~ 1e-7 error per step.

**Mitigation**: Acceptable for DPF. The dual-energy entropy tracer (already
implemented) handles the dominant float32 failure mode (pressure recovery).
Refluxing errors are sub-dominant. For production V&V, use `precision="float64"`
(CPU fallback) which bypasses Metal entirely.

### Constraint 3: No Dynamic Memory Allocation in Metal Kernels

**Impact**: Cannot create new blocks inside a kernel.

**Mitigation**: Regridding (block creation/destruction) happens in Python on
CPU. Only the RHS evaluation and time integration run on Metal. The regrid
frequency is low (every 10-50 steps), so the CPU overhead is small.

### Constraint 4: Parthenon-VIBE Amdahl Bottleneck

The Parthenon-VIBE paper (arXiv:2509.19701) found that AMR overhead
(regridding, ghost exchange, load balancing) can exceed compute savings
on GPU, especially for small block counts. Their measurement: regrid cost
scales as O(N_blocks * N_ghost) and dominates when blocks < 64.

**Mitigation for DPF**: With 2 levels and max 16 blocks, we have at most
~32 blocks total. Ghost exchange is a memory copy, not a kernel launch.
Regrid every 10-50 steps amortizes the cost. The DPF sheath is spatially
coherent (not fragmented), so block count stays low. The bottleneck is
real for 3D cosmology sims with thousands of blocks -- not for 2D
cylindrical DPF.

---

## 7. Implementation Phases

### Phase A: 2-Level SMR with Manual Region Selection

Extend `static_refinement.py` to use block-structured layout instead of
independent fine-grid re-simulation.

- Replace single fine grid with `AMRBlock` / `AMRLevel` data structures
- Manual region selection (user specifies which blocks are refined)
- Ghost exchange between blocks at same level (copy)
- Prolongation from level 0 to level 1 ghost zones (bilinear)
- Restriction from level 1 to level 0 (volume-weighted average)
- Global timestep (no subcycling)
- No refluxing yet (conservation error accepted)
- Wire into `MLXMHDSolver` as an optional mode

**Dependencies**: None (builds on existing static_refinement.py)
**LOC estimate**: 400-500
**Files**: `src/dpf/experimental/amr.py` (new), edits to `mlx_solver.py`

### Phase B: Automatic Refinement with Indicators

- Implement block-level refinement tagging using `current_density_sensor()`
  and `lohner_error_indicator()` (both already exist)
- Automatic regrid every N steps (configurable, default 20)
- Proper nesting constraint: refined block must have parent
- Buffer zone: tag neighbors of flagged blocks to avoid refinement fronts
- Add `AMRConfig` to Pydantic config with thresholds

**Dependencies**: Phase A
**LOC estimate**: 250-350
**Files**: `src/dpf/experimental/amr.py` (extend), `src/dpf/config.py` (add AMRConfig)

### Phase C: Refluxing + Optional Subcycling

- Implement Berger-Colella refluxing with cylindrical `r`-weighting
- Add subcycling as configurable option (default off)
- Conservation verification: total mass/energy/momentum before and after
  correction step
- Dedner GLM at level boundaries (not CT)

**Dependencies**: Phase A, Phase B
**LOC estimate**: 300-400
**Files**: `src/dpf/experimental/amr.py` (extend)

### Phase D: 3+ Levels + Performance

- Generalize to N levels with recursive subcycling
- `mx.vmap()` over blocks at each level for batch processing
- Profiling and tuning: regrid frequency, block size, max levels
- Coarsening criterion (de-refine when indicator drops below threshold)

**Dependencies**: Phase C
**LOC estimate**: 200-300
**Files**: `src/dpf/experimental/amr.py` (extend)

---

## 8. LOC Estimates Summary

| Phase | New LOC | Cumulative | Calendar Estimate | Key Risk |
|-------|---------|------------|-------------------|----------|
| A | 400-500 | 400-500 | 2-3 sessions | Ghost exchange bugs |
| B | 250-350 | 650-850 | 1-2 sessions | Over-refinement |
| C | 300-400 | 950-1250 | 2-3 sessions | Conservation error at boundaries |
| D | 200-300 | 1150-1550 | 1-2 sessions | Performance regression |
| **Total** | **1150-1550** | | **6-10 sessions** | |

Test LOC (not included above): ~300-400 per phase, ~1200-1600 total.

---

## 9. Risk Assessment

### Risk 1: Flux Conservation at Coarse-Fine Boundaries (HIGHEST)

**Probability**: High. This is the hardest part of AMR for MHD.
**Impact**: Mass/energy drift proportional to sheath velocity * boundary area.
**Detection**: Monitor `delta_mass / mass_total` per step. Threshold: 1e-8.
**Mitigation**: Phase C refluxing. Until then, accept O(dx) conservation error.
**FMEA RPN**: 315 (from DMAIC_FORWARD_PLAN.md, item 7.1).

### Risk 2: GPU Memory Fragmentation

**Probability**: Low for 2D cylindrical (blocks are < 1 MB each).
**Impact**: OOM or performance cliff from non-contiguous allocations.
**Detection**: Monitor `mx.metal.get_active_memory()` during regrid.
**Mitigation**: Pre-allocate fixed MAX_BLOCKS. Reuse slots.

### Risk 3: Performance Regression from Regridding (Amdahl)

**Probability**: Medium for Phase D (3+ levels with frequent regrid).
**Impact**: AMR slower than uniform fine grid.
**Detection**: Wall-clock comparison: AMR vs uniform at matched effective resolution.
**Mitigation**: Regrid every 20-50 steps (not every step). Global timestep
avoids subcycling overhead for 2 levels. Profile before adding complexity.
**FMEA RPN**: 336 (from DMAIC_FORWARD_PLAN.md, item 7.3).

### Risk 4: Breaking Existing Tests

**Probability**: Medium. 4,900+ tests assume `(NVAR, nr, nz)` uniform arrays.
**Impact**: CI failures; wasted debugging time.
**Detection**: Run full test suite after Phase A.
**Mitigation**: AMR is opt-in (`use_amr=True` in config). Default path unchanged.
AMR blocks expose the same `(NVAR, nr, nz)` interface per block.

### Risk 5: CT Divergence at Level Boundaries

**Probability**: High if CT is used across levels.
**Impact**: Growing div(B) at coarse-fine faces.
**Detection**: Monitor max(|div B|) at level boundaries.
**Mitigation**: Use Dedner GLM at boundaries (already implemented). CT within
blocks only. This is explicitly the MPI-AMRVAC approach.

### Risk Matrix

| Risk | Probability | Impact | Phase Affected | Mitigation Ready? |
|------|-------------|--------|----------------|-------------------|
| Flux conservation | High | High | C | Yes (refluxing) |
| CT at boundaries | High | Medium | C | Yes (Dedner) |
| Amdahl bottleneck | Medium | Medium | D | Partial (global dt) |
| Test breakage | Medium | Low | A | Yes (opt-in) |
| Memory fragmentation | Low | Medium | D | Yes (pre-alloc) |

---

## 10. References

1. **Berger, M.J. & Colella, P.**, "Local adaptive mesh refinement for shock
   hydrodynamics," *JCP* 82:64-84 (1989).
   -- Foundational AMR algorithm. Refluxing at coarse-fine boundaries.

2. **Stone, J.M. et al.**, "The Athena++ adaptive mesh refinement framework,"
   *ApJS* 249:4 (2020).
   -- Block-structured AMR for MHD with CT. Reference architecture.

3. **Grete, P. et al.**, "Parthenon -- a performance portable block-structured
   adaptive mesh refinement framework," *IJHPCA* 37:465-486 (2022).
   -- GPU-native block AMR built on Kokkos. Basis for AthenaK.

4. **Parthenon-VIBE** (arXiv:2509.19701, 2025).
   -- Found severe Amdahl bottleneck: AMR overhead exceeds compute savings
   on GPU when regrid frequency is high. Key constraint for our design.

5. **Lohner, R.**, "An adaptive finite element scheme for transient problems
   in CFD," *CMAME* 61:323-338 (1987).
   -- Second-derivative error indicator. Already implemented in our codebase.

6. **Keppens, R. et al.**, "MPI-AMRVAC 3.0," *A&A* 673:A66 (2023).
   -- Block AMR with GLM div(B) cleaning at level boundaries (not CT).
   Validates our approach of Dedner at boundaries, CT within blocks.

7. **Teyssier, R.**, "Cosmological hydrodynamics with adaptive mesh
   refinement," *A&A* 385:337-364 (2002).
   -- Octree AMR (RAMSES). Included for comparison; not recommended for GPU.

8. **Zhang, W. et al.**, "AMReX: a framework for block-structured adaptive
   mesh refinement," *JOSS* 4:1370 (2019).
   -- Patch-based AMR. Reference for refluxing implementation details.

9. **Miyoshi, T. & Kusano, K.**, "A multi-state HLL approximate Riemann
   solver for ideal MHD," *JCP* 208:315-344 (2005).
   -- HLLD solver used within each block.

10. **Popovas, A. et al.**, "DISPATCH HLLS: entropy-stable MHD in float32,"
    *arXiv:2211.02438* (2025).
    -- Dual-energy entropy tracer. Critical for float32 AMR where
    refluxing errors compound with cancellation errors.

---

## Appendix: Decision Log

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| AMR type | Block-structured | Patch, Octree | `mx.compile()` static shapes |
| Primary sensor | Current density | Density gradient | J peaks at sheath; physically motivated |
| Subcycling (Phase A-B) | Global dt | Subcycled | Simpler; 2-level penalty is only 2x |
| div(B) at boundaries | Dedner GLM | CT | CT across levels is unsolved for GPU |
| Block size | 32x64 | 16x32, 64x128 | Matches typical nr/nz ratio; fits L1 cache |
| Max blocks pre-alloc | 16 per level | Dynamic | `mx.compile()` needs static shapes |
