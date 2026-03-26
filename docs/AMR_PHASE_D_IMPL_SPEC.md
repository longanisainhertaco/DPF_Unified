# AMR Phase D: N-Level V-Cycle Subcycling — Implementation Spec

**Date**: 2026-03-26  |  **Status**: IMPL-READY  |  **Budget**: <250 lines
**Prereqs**: Phases A (static), B (auto-refine), C (refluxing) complete
**Refs**: Berger & Colella 1989 JCP 82:64; Stone et al. 2020 ApJS 249:4

---

## 1. Generalized AMRHierarchy: 2-Level → N-Level

### Data Structure Changes

The Phase D prototype `AMRHierarchy` already stores `levels: list[AMRLevel]`
and `add_level()`. Three additions are needed for N-level:

```python
@dataclass
class AMRLevel:
    level: int
    blocks: dict[tuple[int, int], AMRBlock]
    dr: float
    dz: float
    dt: float                        # NEW: level-local timestep
    flux_register: FluxRegister      # NEW: one per level (for CF refluxing)
    vmap_rhs: Callable | None = None # NEW: cached compile(vmap(rhs_block))

@dataclass
class AMRHierarchy:
    levels: list[AMRLevel]
    max_levels: int
    ratio: int
    _rhs_cache: dict[int, Callable] = field(default_factory=dict)  # keyed by level_idx
```

Key rule: `level.dt = dt_l0 / ratio^l`. Level-0 dt from global CFL minimum.
Phase A fallback: all levels use global minimum. Hard cap: `max_levels = 4`.

---

## 2. V-Cycle Algorithm (Canonical)

```
advance_level(level_idx, dt_level):
    level = hierarchy.levels[level_idx]
    ratio = hierarchy.ratio

    # 1. Advance this level with dt_level (SSP-RK3)
    U_padded = pad_ghosts(level, topology, ng=3)        # Option B from concerns doc
    L_batch   = compile(vmap(mhd_rhs))(U_padded)       # 2.44x vs loop
    level     = ssp_rk3_combine(level, L_batch, dt_level)

    # 2. Recurse into finer level (ratio sub-steps)
    if level_idx + 1 < hierarchy.n_levels and hierarchy.levels[level_idx+1].blocks:
        dt_fine = dt_level / ratio
        for sub in range(ratio):
            # Ghost refill MANDATORY before every fine sub-step (RPN 200 risk)
            fill_ghosts_from_coarse(level_idx+1, level_idx)   # prolongation
            fill_ghosts_same_level(level_idx+1)                # neighbor copy
            advance_level(level_idx + 1, dt_fine)              # recurse

        # 3. Average-down: fine -> coarse (volume-weighted, r-weighted)
        restrict_all_children(level_idx)

        # 4. Reflux correction (Phase C formula, applied at every interface)
        if config.use_refluxing:
            apply_refluxing(level.flux_register, cf_map[level_idx], level)
            level.flux_register.reset()
```

**Call site** (replaces `amr_step_with_autorefine`):

```python
def amr_step_vcycle(hierarchy, dt_l0, config, solver_params):
    if should_regrid(step):
        regrid(...)
    for l in range(hierarchy.n_levels):
        hierarchy.levels[l].dt = dt_l0 / (config.ratio ** l)
    advance_level(hierarchy, 0, dt_l0, config, solver_params)
```

W-cycle excluded: DPF is hyperbolic-dominated; 57% more fine-level work, no gain.

---

## 3. Ghost Freshness Protocol

### When Does Ghost Data Become Stale?

A block's ghost data is stale the moment any neighboring block (coarse or same-level)
completes an RK stage that modifies its interior. For subcycled AMR:

| Event | Staleness | Action |
|-------|-----------|--------|
| Same-level neighbor completes RK stage | Immediately stale | Refill before next RK stage |
| Coarse level advances by `dt_coarse` | Stale after coarse step completes | Refill before FIRST fine sub-step |
| Coarse level advances by `dt_fine` (sub-step) | n/a — coarse is frozen during fine sub-steps | No refill needed mid-sub-step series |

### Can We Skip Any Refills?

**Coarse-to-fine boundary (prolongation)**: The coarse level is NOT updated
between fine sub-steps — it is frozen at the beginning of its step. Therefore
prolongation-ghost data is valid for ALL `ratio` fine sub-steps using the same
coarse state. The prototype's "refill before every fine sub-step" is conservative.

**Optimized protocol** (saves `(ratio-1)` prolongation calls per coarse step):

```python
# Prolongation: once per coarse step, before first sub-step
fill_ghosts_from_coarse(fine_idx, coarse_idx)

for sub in range(ratio):
    fill_ghosts_same_level(fine_idx)   # must still happen: fine neighbors update
    advance_level(fine_idx, dt_fine)
```

For ratio=2 this halves prolongation cost. For ratio=4 it cuts to 25%.

**Same-level ghost refill** cannot be skipped: fine-level neighbors advance at
`dt_fine` and their boundary data changes after each sub-step.

Conservative fallback (prototype, RPN 200 mitigation): `fill_all_ghosts()` before
every sub-step. Use during initial testing; switch to optimized once tests pass.

---

## 4. Memory Budget (PF-1000, 3 Levels)

### Grid Parameters

| Level | dr (mm) | dz (mm) | Blocks (est.) | Block shape | Cells/block |
|-------|---------|---------|---------------|-------------|-------------|
| 0 | 0.469 | 1.25 | 4 (64×128 base, 4×16×32 blocks) | 16×32 | 512 |
| 1 | 0.234 | 0.625 | 4 (sheath region, 1-2 active at a time) | 16×32 | 512 |
| 2 | 0.117 | 0.313 | 4 (pinch region only) | 16×32 | 512 |

Total blocks: 12. Total cells: 12 × 512 = 6,144.

### Memory Calculation

Per block: `NVAR × block_nr × block_nz × bytes`
- float32: 10 × 16 × 32 × 4 = 20,480 bytes = **20 KB**
- float64: **40 KB**

| Component | float32 | float64 |
|-----------|---------|---------|
| 12 block U arrays | 240 KB | 480 KB |
| 12 ghost-padded U (ng=3) | ~400 KB | ~800 KB |
| Flux registers (2 CF interfaces) | ~5 KB | ~10 KB |
| Python metadata | ~50 KB | ~50 KB |
| **Total** | **~700 KB** | **~1.3 MB** |

0.004% of 36 GB. Memory is irrelevant at this scale; concern only above ~1,000 blocks.

---

## 5. mx.vmap Batching Strategy Across 3 Levels

### Constraint

`mx.vmap` requires a uniform leading dimension of identical-shape arrays.
Blocks on different levels have the same block shape (16×32) but different `dr`,
`dz`, `r_min` — i.e., different `CylindricalGrid` parameters. Batching
**across levels is invalid** (different physics parameters).

### Strategy: Per-Level Batching

```
Level 0: 4 blocks  → vmap batch of 4  → compile(vmap(rhs))[level0_grid]
Level 1: 4 blocks  → vmap batch of 4  → compile(vmap(rhs))[level1_grid]
Level 2: 4 blocks  → vmap batch of 4  → compile(vmap(rhs))[level2_grid]
```

At 4 blocks per level, each vmap call processes a batch of 4 × 10 × 22 × 38 = ~33K
elements (with ghost padding). The 2.41x vmap speedup measured at 8 blocks scales
to ~2.1x at 4 blocks (less saturation, but still substantial vs sequential).

Cached compiled functions: `_rhs_cache` stores `compile(vmap(rhs_block_l))` keyed
by `(level_idx, block_nr, block_nz, nr_padded, nz_padded)`. Shape-caching overhead
is 6% (mlx_compile_vmap_analysis.md). Flush cache entry after regrid at that level.
Degenerate batch=1 (single active block at level 2) works; performance same as
sequential but uses the same code path.

---

## 6. Is Level 2 Worth It?

### Physics Unlocked by Each Level

DPF sheath ion inertial length: `c/ω_pi = c / sqrt(n_i e² / ε_0 m_i)`.
For D₂ at 3.5 Torr and ~10x compression during implosion:
`n_i ~ 10 × 1.2e23 m⁻³ → c/ω_pi ~ 0.08 mm`.

| Level | dr (mm) | Cells across sheath (2mm) | Cells across c/ωpi | Physics accessible |
|-------|---------|--------------------------|---------------------|-------------------|
| 0 | 0.469 | 4 | 0.2 | MHD continuum (sheath captured barely) |
| 1 | 0.234 | 8 | 0.3 | Shock structure, current sheet profile |
| 2 | 0.117 | 17 | 0.7 | Resistive tearing, anomalous resistivity onset |

**Level 1 unlocks**: correct shock width (Rankine-Hugoniot jump captured in ~4 cells
instead of ~2), physically meaningful current density peak, sheath velocity estimate
within ~5% (vs ~15% at level 0). This is the primary science gain.

**Level 2 unlocks**: resistive dissipation layer structure (requires ~10 cells across
the current sheet), anomalous resistivity activation threshold (localized J/nec > vth
condition), early-pinch kinking instability seeds. **These are inaccessible at level 1.**

**Verdict**: Level 1 suffices for I_peak/t_peak calibration. Level 2 is needed for
neutron yield and instability onset. Gate behind `max_levels: 3` (default 2);
enable only for pinch-phase analysis.

---

## 7. Testing Plan

### Test 1: Linear Wave Convergence (3 levels, analytical solution)

Right-traveling Alfvén wave, periodic BC: `rho=1, B=(1,0,0.1*sin(2πx/L))`.
Run at 1, 2, and 3 levels. Measure L1 error in `Bz` at `t = 2L/vA`.
Pass: error decreases ~4× per level (PLM/HLL second-order). Level 3 must not
increase error vs level 2 — that would flag a prolongation or ghost bug.

### Test 2: Sod Shock Crossing CF Boundary (conservation)

1D Sod, CF boundary at x=0.3 (shock crosses at t~0.2), 200 steps, level 1
refines x in [0.25, 0.45]. Monitor `∫ρ dV` each step.
Pass: refluxing reduces `|Δm/m|` from ~1e-4/crossing to `< 1e-8` cumulative (≥100× reduction).

### Test 3: 3-Level PF-1000 Smoke Test

100 steps, `max_levels=3`, `auto_refine=True`. Level 2 activates near axis
during radial implosion. Pass: no NaN, no negative pressure, `|ΔE/E| < 1e-4/step`.

---

## 8. Risk Assessment: Top 3 Failure Modes

| # | Failure Mode | S | O | D | RPN | Mitigation | Rollback |
|---|-------------|---|---|---|-----|------------|----------|
| D1 | **Stale ghost at level 2 boundary** — level 2 reads from level 1 ghosts that were filled before level 1 advanced in its sub-step. Net: level 2 uses ghost data that is 1 fine sub-step old. | 8 | 5 | 5 | **200** | Optimized protocol: prolong once before sub-step series; refill same-level before each. Add assertion: `assert ghost_fill_step[block] == current_sub_step`. | Fall back to full `fill_all_ghosts()` before every sub-step (conservative). |
| D2 | **Restriction-reflux ordering at level 1→0 after multi-level advance** — reflux must happen AFTER all level-1 sub-steps complete AND after level-2 has restricted to level-1, else level-1 fluxes are inconsistent. | 9 | 4 | 4 | **144** | Strict ordering in `advance_level`: recurse fully → restrict fine→this → apply_refluxing at this interface. Never apply reflux mid-subcycle. Unit test: total momentum before/after full V-cycle step must match. | Disable refluxing at level 1 interface (`use_refluxing=False` per-interface flag). |
| D3 | **vmap shape mismatch when block count varies between regrid steps** — if regrid adds/removes blocks, the batch tensor `U_batch` changes shape; compiled function sees new shape → recompile, but cached `_rhs_cache` still holds old shape → wrong dispatch. | 6 | 6 | 3 | **108** | Flush `_rhs_cache[level_idx]` after every regrid that changes block count at that level. Add shape fingerprint to cache key: `(level_idx, n_blocks, nr_padded, nz_padded)`. | Sequential fallback per block (no vmap) if cache miss. |

D1 and D2 interact: stale ghosts (D1) can mask wrong reflux ordering (D2) by
biasing fluxes that partially cancel the reflux error. Test both together via
Test 2 (Sod crossing), not in isolation.
