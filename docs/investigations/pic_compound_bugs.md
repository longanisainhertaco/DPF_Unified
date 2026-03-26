# PIC-MHD Hybrid Module: Compound Bug Analysis

**Date**: 2026-03-26
**Investigator**: dpf-validation-engineer (Opus 4.6)
**File under analysis**: `src/dpf/experimental/pic/hybrid.py` (1716 LOC, 16 functions + 2 classes)

---

## 1. Test Coverage Gap Analysis

### Functions and Their Test Status

| Function | LOC | Tested? | Test Location |
|----------|-----|---------|---------------|
| `_nanbu_scatter_kernel` | 249 | YES | test_pic_hybrid.py (8 tests) |
| `_coulomb_scatter` | 103 | PARTIAL | test_pic_hybrid.py (indirectly via push_particles fallback) |
| `_boris_push_kernel` | 80 | YES | test_pic_validation.py (gyration, return-to-origin) |
| `_deposit_density_kernel` | 75 | YES | test_pic_validation.py (charge conservation, multiple particles) |
| `_deposit_current_kernel` | 104 | NO | -- |
| `_deposit_current_esirkepov_kernel` | 226 | NO | -- |
| `_interpolate_scalar_kernel` | 56 | NO | -- |
| `_interpolate_vector_kernel` | 65 | NO | -- |
| `boris_push` (wrapper) | 9 | YES | via kernel test |
| `deposit_density` (wrapper) | 10 | YES | via kernel test |
| `deposit_current` (wrapper) | 12 | NO | -- |
| `deposit_current_esirkepov` (wrapper) | 13 | NO | -- |
| `interpolate_field_to_particles` (wrapper) | 11 | PARTIAL | test_pic_validation.py (shape test only) |
| `ParticleSpecies` | 16 | NO | -- (only used transitively) |
| `HybridPIC.__init__` | 18 | YES | test_pic_hybrid.py |
| `HybridPIC.enable_collisions` | 4 | NO | -- |
| `HybridPIC.add_species` | 12 | NO | -- |
| `HybridPIC.push_particles` | 86 | PARTIAL | test_pic_hybrid.py (NaN check only, no correctness) |
| `HybridPIC._apply_reflecting_bc` | 12 | NO | -- |
| `HybridPIC.deposit` | 47 | NO | -- |
| `HybridPIC.inject_beam` | 89 | NO | -- |
| `detect_instability` | 17 | NO | -- |

**Summary**: 10 of 22 functions/methods have zero test coverage. 3 more have only partial coverage (shape checks or NaN checks, no correctness validation).

---

## 2. Esirkepov Kernel Deep Analysis (lines 668-894)

### 2.1 Charge Conservation Identity

The Esirkepov algorithm guarantees `div(J)*dt + delta_rho = 0` by constructing J from the *difference* of shape functions `dS = S1 - S0` rather than from particle velocities. The W-factors use a running-sum accumulation along each axis, which is the discrete analog of the fundamental theorem of calculus.

**Assessment**: The mathematical structure is correct for the 2-node CIC case. The running sum `Wx_running += dSx[li] * wy_jk * wz_jk` followed by accumulation into `Jx[ii, jj, kk]` implements Esirkepov's Equation 17 correctly for the first-order (CIC) shape.

**BUT**: The charge conservation identity holds ONLY when the index mapping between old and new positions is exact. The clamping at lines 738-741 and 755-757 breaks this guarantee.

### 2.2 Off-by-One and Boundary Errors

**BUG 1 (CRITICAL) -- Clamping destroys charge conservation**

```python
# Line 738-741: Old position clamped
ix0 = max(0, min(ix0, nx - 2))
# Line 755-757: New position clamped
ix1 = max(0, min(ix1, nx - 2))
```

When a particle's old position is outside the grid (negative coordinate or beyond grid), `ix0` gets clamped but `fx0 = xo - ix0` is computed BEFORE clamping (line 744, then clamped to [0,1] at 747). The issue: for a particle at x=-0.5/dx, `ix0 = floor(-0.5) = -1`, then clamped to 0, then `fx0 = -0.5 - 0 = -0.5`, then clamped to 0.0. This means `S0x = (1.0, 0.0)` -- the particle is assigned entirely to node 0.

Similarly for the new position. If old and new positions are on opposite sides of a boundary, the dS computation is correct *within the clamped region* but the actual charge movement is not captured. The particle's charge appears/disappears at the boundary without corresponding current. **This breaks div(J)*dt + delta_rho = 0 at boundaries.**

**Severity**: High for particles near domain edges. Benign when all particles stay well within the grid.

**BUG 2 (MODERATE) -- Multi-cell crossing silently zeroed**

```python
# Lines 787-789:
else:
    S1x_0 = 0.0
    S1x_1 = 0.0
```

When `|offset_x| > 1` (particle crosses more than one cell in a single timestep), BOTH S1 values are set to zero. This means `dSx = (0 - S0x_0, 0 - S0x_1)`, which deposits current as if the particle disappeared from the old cell and appeared nowhere. The charge is lost.

For CIC with a 2-node stencil, Esirkepov requires `|offset| <= 1` per axis per step. A particle crossing 2+ cells violates the CFL-like constraint `dx/dt > v_max`. This is documented in Esirkepov (2001) but there is NO runtime check or warning.

**Severity**: Will trigger whenever a fast particle (beam ions at 100 keV) crosses more than one cell per step. At 100 keV, a deuteron has v ~ 3.1e6 m/s. With dx = 0.01 m and dt = 1e-9 s, displacement per step = 3.1e-3 m = 0.31 cells. Safe. But with dt = 1e-8 s, displacement = 3.1 cells. **Silent charge loss.**

### 2.3 Division by Zero

```python
# Line 718:
prefactor = charge / (cell_volume * dt) if (cell_volume > 0.0 and dt > 0.0) else 0.0
```

This guards against zero dt correctly. No other division-by-zero paths exist in the kernel.

### 2.4 The dt Bug (line 1561)

```python
# deposit() method, line 1561:
jx, jy, jz = deposit_current_esirkepov(
    sp.positions_old, sp.positions, sp.weights, sp.charge,
    self.grid_shape, self.dx, self.dy, self.dz, self.dt,  # <-- self.dt
)
```

```python
# push_particles() method, line 1431-1432:
if dt is None:
    dt = self.dt
```

**The problem**: `push_particles(dt=some_value)` moves particles using `some_value`, but `deposit()` always uses `self.dt` for the Esirkepov prefactor. If the caller passes a different dt to push_particles (e.g., a sub-cycled PIC timestep), the prefactor `charge / (cell_volume * dt)` uses the wrong dt.

The Esirkepov identity `div(J)*dt + delta_rho = 0` requires that the dt in the prefactor matches the dt used for the push. A mismatch scales J incorrectly:

```
J_computed = J_correct * (dt_push / self.dt)
```

**Severity**: Breaks charge conservation by a factor of `dt_push / self.dt` whenever sub-cycling is used. Currently benign because no caller sub-cycles, but this is a latent defect that will manifest the moment adaptive timestepping is attempted.

### 2.5 Array Bounds

All stencil accesses are guarded by bounds checks:
```python
if jj < 0 or jj >= ny or kk < 0 or kk >= nz:
    continue
```

No out-of-bounds access is possible. The cost is that boundary particles silently lose current contributions to clipped stencil nodes, which again breaks charge conservation near boundaries.

---

## 3. Push-Deposit Cycle Analysis

### 3.1 What Happens When a Particle Leaves the Domain?

**Reflecting BCs** (`_apply_reflecting_bc`, lines 1490-1519):

```python
# Reflect off lower boundary (x < 0)
mask_lo = positions[:, d] < 0.0
positions[mask_lo, d] = -positions[mask_lo, d]
velocities[mask_lo, d] = -velocities[mask_lo, d]

# Reflect off upper boundary (x > L)
mask_hi = positions[:, d] > limits[d]
positions[mask_hi, d] = 2.0 * limits[d] - positions[mask_hi, d]
velocities[mask_hi, d] = -velocities[mask_hi, d]
```

**BUG 3 (MODERATE) -- Double reflection possible**: If a particle exits through a corner (e.g., x<0 AND y<0), both reflections are applied independently. This is correct for orthogonal boundaries. But if a particle is reflected in x from `x = -0.1L` to `x = 0.1L`, and then also exits in y, the combined reflection is correct only for small excursions. For large excursions (v*dt >> L), a single reflection is insufficient -- the particle could bounce multiple times. No multi-bounce handling exists.

**BUG 4 (CRITICAL for Esirkepov) -- Reflected position breaks Esirkepov stencil**: `positions_old` is saved BEFORE the push (line 1447), and `positions` is the reflected post-push position. Esirkepov computes current from the displacement `positions_new - positions_old`. When a particle near the boundary is reflected, the displacement vector is WRONG:

Example: particle at x = 0.99*L with v > 0. After push, x = 1.01*L. After reflection, x = 0.99*L. Displacement = 0.99L - 0.99L = 0. But the particle actually moved from 0.99L to 1.01L and back. Esirkepov sees zero displacement, deposits zero current. The actual charge moved to the boundary and back, but no current is recorded.

**This is a fundamental incompatibility between reflecting BCs and Esirkepov current deposition.** The fix requires depositing current in two segments: pre-reflection and post-reflection.

### 3.2 Negative rho from Deposition

The `deposit()` method (line 1525) computes rho as:

```python
rho_grid += sp.charge * n
```

where `n` is the CIC-deposited number density (always >= 0). For positive ions (charge > 0), rho >= 0. For electrons (which are NOT tracked as particles in this hybrid scheme -- electrons are a neutralizing fluid), rho is always non-negative from the ion species.

**However**: If someone adds a species with negative charge (electrons as particles), rho can go negative at any cell. No floor or check exists.

**Assessment**: Not a concern for the intended hybrid PIC use (ions only), but a latent issue if the module is extended.

### 3.3 Weight Conservation

`inject_beam` (line 1578) appends particles with `macro_weight = weight_total / n_beam`. The total injected weight is `n_beam * weight_total / n_beam = weight_total`. Correct.

`push_particles` does not create or destroy particles. Correct.

`_apply_reflecting_bc` does not change weights. Correct.

**No particle removal mechanism exists**. Particle count grows monotonically with every `inject_beam` call. There is no particle merging, splitting, or absorption. Weight is trivially conserved because nothing changes it.

---

## 4. Compound Bug Interaction Analysis

### 4.1 Interacting Untested Function Pairs

**PAIR 1 (HIGHEST RISK): push_particles + deposit (via Esirkepov)**

- `push_particles` saves `positions_old`, calls `_boris_push_kernel`, calls `_apply_reflecting_bc`
- `deposit` calls `_deposit_current_esirkepov_kernel` using `positions_old` and the reflected `positions`
- **Compound bug**: Reflected particles produce wrong displacement vectors (Bug 4) AND the dt mismatch (Bug in line 1561) compounds the error. These two bugs can partially cancel or amplify depending on the sign of the reflection.

**PAIR 2 (HIGH RISK): inject_beam + deposit**

- `inject_beam` sets `sp.positions_old = sp.positions.copy()` (line 1665)
- `deposit` checks `np.array_equal(sp.positions_old, sp.positions)` to decide Esirkepov vs CIC fallback (line 1556)
- **Compound bug**: If `inject_beam` is called between push and deposit, newly injected particles have `positions_old == positions`, correctly falling back to CIC. BUT the existing particles in the same species still use Esirkepov. The J contributions from CIC and Esirkepov use different conventions (CIC: J = q*n*v, Esirkepov: J from shape function differences). Mixing them in the same species array produces inconsistent current that does NOT satisfy charge conservation.

Wait -- actually re-reading lines 1554-1557: `_esirkepov_usable` is PER-SPECIES, not per-particle. If `inject_beam` is called, the new particles make `np.array_equal(positions_old, positions)` return False (because old particles have positions_old != positions), so ALL particles (including newly injected ones) go through Esirkepov. Newly injected particles have zero displacement, so their Esirkepov contribution is zero. This is actually correct behavior. The fallback only activates on the FIRST call before any push.

**Revised assessment**: This pair is safe IF inject_beam is not called between push and deposit for the same species.

**PAIR 3 (HIGH RISK): Boris push + interpolation at ghost cells**

- `interpolate_field_to_particles` clamps indices to `[0, nx-2]` (line 934)
- Boris push uses these interpolated E and B fields
- **Compound bug**: A particle at position x ~ 0 (near lower boundary) gets `ix = 0`, which reads `field[0, ...]` and `field[1, ...]`. If the MHD solver does not fill ghost cells at index 0 with physical values, the interpolated field is garbage (typically zero or initial conditions). Boris push with garbage E-field produces garbage velocity. Garbage velocity produces large displacement. Large displacement triggers multi-cell crossing. Multi-cell crossing zeroes Esirkepov S1 (Bug 2). Charge is lost. Deposited density drops. On next step, E-field from Ohm's law changes. Feedback loop.

**This is the most likely NaN chain.**

**PAIR 4 (MODERATE RISK): Nanbu collisions + Boris push**

- `push_particles` calls Boris FIRST, then Nanbu scattering
- Nanbu modifies velocities in-place but NOT positions
- **Compound bug**: The Nanbu kernel passes `vel_a = new_vel, vel_b = new_vel` (same array twice, line 1472). This means it scatters the species against ITSELF. But the Fisher-Yates shuffle uses `idx_a` and `idx_b` independently, so a particle can be paired with itself (`idx_a[k] == idx_b[k]`). Self-scattering is a no-op (COM frame velocity is zero, p_star = 0, `continue` at line 176), so this is just wasted work, not a bug. But if N_a != N_b (which can't happen here since both are the same array), `min(N_a, N_b)` silently drops particles.

### 4.2 State Inconsistency Scenarios

| Scenario | Functions Involved | Inconsistency |
|----------|-------------------|---------------|
| Particle reflected + Esirkepov | _apply_reflecting_bc + _deposit_current_esirkepov_kernel | Wrong displacement vector, broken charge conservation |
| Fast beam particle + large dt | inject_beam + boris_push + Esirkepov | Multi-cell crossing, silent charge loss |
| Ghost cell E-field + push | interpolate_field_to_particles + _boris_push_kernel | Garbage acceleration, runaway velocity |
| Sub-cycled push + deposit | push_particles(dt=dt_sub) + deposit() | J scaled by wrong dt, charge conservation broken by dt_push/self.dt |

---

## 5. First End-to-End NaN Prediction

### 5.1 Particle Count Growth

With `inject_beam` called every step, injecting `n_beam` particles:
- After 1000 steps: `N_initial + 1000 * n_beam` particles
- Typical n_beam = 100: 100,000 new + N_initial
- Each particle: 3 doubles position + 3 doubles velocity + 1 double weight + 3 doubles positions_old = 10 doubles = 80 bytes
- 100,000 particles = 8 MB. Not a memory problem.
- At n_beam = 10000: 10M particles = 800 MB. Becomes a concern on 36 GB system, especially with Numba kernel overhead.

### 5.2 Boris Push Relativistic Runaway

The Boris pusher (lines 398-478) is NON-RELATIVISTIC. There is no gamma factor:

```python
qdt_over_2m = charge * dt / (2.0 * mass)
```

The relativistic Boris push uses `q*dt / (2*m*gamma)`. Without the gamma correction, a strong E-field accelerates the particle without limit. Each step adds `delta_v = q*E*dt/m` regardless of current velocity. After enough steps, `v > c`.

**Consequence**: When `v > c`, the position update `x += v*dt` moves the particle by more than `dx` per step. If the grid is coarse enough, this triggers multi-cell crossing in Esirkepov (Bug 2) and eventual domain exit.

The Nanbu kernel has subluminal clamping (line 131-138), but Boris does not. **The Boris pusher will produce superluminal particles in DPF conditions.**

DPF pinch E-fields: ~10^7 V/m. Deuteron: q=e, m=m_D. Acceleration: a = eE/m_D ~ 1.602e-19 * 1e7 / 3.34e-27 = 4.8e14 m/s^2. At dt=1e-10 s, delta_v = 4.8e4 m/s per step. After 6000 steps, v = 2.9e8 m/s ~ c. **Superluminal in ~6000 steps of Boris push at DPF-relevant E-fields.**

### 5.3 E-Field Interpolation at Ghost Cells

The MHD solver's E-field grid typically has ghost cells initialized to zero or ambient values. When `interpolate_field_to_particles` clamps a particle at position x~0 to `ix=0`, it reads `field[0,...]` and `field[1,...]`. If index 0 is a ghost cell with E=0 while index 1 has the physical E-field, the interpolated value is a weighted average. This produces a spurious gradient at the boundary.

More critically: if the MHD state has NaN in any ghost cell (common during pinch phase, see MEMORY.md ghost-cell NaN root cause), the interpolation returns NaN. Boris push with NaN E produces NaN velocity. NaN velocity produces NaN position. NaN position in Esirkepov produces NaN J. NaN J in Ohm's law produces NaN E for the next step. **Total simulation NaN in one step.**

### 5.4 Most Likely NaN Chain

```
Step 1:    MHD ghost cells contain NaN (known issue, see ghost-cell RHS fix)
Step 2:    Particle near boundary, interpolation reads ghost cell NaN
Step 3:    Boris push: NaN E -> NaN velocity -> NaN position
Step 4:    Esirkepov: NaN positions -> NaN in normalized coords -> NaN in J
Step 5:    J fed back to MHD Ohm's law -> NaN in E everywhere
Step 6:    ALL particles get NaN E -> complete simulation death
```

**Time to NaN: 1 step after PIC coupling activates, if any particle is within 1 cell of the boundary.**

### 5.5 Second Most Likely NaN Chain

```
Step 1:    Strong E-field from pinch (10^7 V/m)
Step 2:    Non-relativistic Boris push accumulates v > c over ~6000 steps
Step 3:    Position update: x += v*dt where v ~ 3e8, dt ~ 1e-10
             displacement = 0.03 m per step, dx = 0.01 m -> 3 cells per step
Step 4:    Esirkepov: |offset_x| = 3 > 1 -> S1 = (0, 0), dS = -S0
Step 5:    Charge deposited as leaving but not arriving -> rho drops to zero
Step 6:    Ohm's law: J/rho -> division by near-zero rho -> E spike
Step 7:    E spike accelerates all nearby particles -> exponential runaway
Step 8:    NaN from overflow
```

**Time to NaN: ~6000 Boris steps, but in practice likely fewer due to localized E-field spikes.**

---

## 6. Prioritized Test List

### MUST TEST FIRST (NaN prevention)

**Priority 1: `interpolate_field_to_particles` with NaN in ghost cells**
- Test: Create a (8,8,8,3) E-field array with NaN at index 0 and 7. Place particle at x ~ 0. Verify interpolation returns NaN (or a guarded finite value).
- Why first: This is the #1 NaN entry point. One step is enough to kill the simulation.
- Expected finding: NaN propagates. No guard exists.

**Priority 2: `_deposit_current_esirkepov_kernel` charge conservation test**
- Test: Single particle moving from (0.5, 0.5, 0.5) to (0.6, 0.5, 0.5) on a 4x4x4 grid. Compute div(J)*dt numerically (finite differences). Compute delta_rho from deposit_density at old and new positions. Verify `div(J)*dt + delta_rho = 0` to machine precision.
- Why: This is the core mathematical identity. If it fails, ALL current deposition is wrong.
- Expected finding: Passes for interior particles. Fails for particles near boundaries (Bug 1).

**Priority 3: `_deposit_current_esirkepov_kernel` multi-cell crossing**
- Test: Single particle moving from (0.5, 0.5, 0.5) to (2.7, 0.5, 0.5) (offset_x = 2). Verify J is not zero.
- Why: Bug 2 silently zeroes charge for multi-cell crossings.
- Expected finding: J incorrectly zeroed. Charge lost.

**Priority 4: `_apply_reflecting_bc` + Esirkepov interaction**
- Test: Particle at x = 0.99*Lx with vx > 0. Push, reflect, then run Esirkepov. Verify charge conservation.
- Why: Bug 4. Reflected displacement is wrong.
- Expected finding: Charge conservation violation at boundary.

### HIGH PRIORITY (correctness)

**Priority 5: `_boris_push_kernel` velocity growth without relativistic correction**
- Test: Single deuteron in E = (1e7, 0, 0) V/m, B = 0. Push for 10000 steps at dt=1e-10. Check if |v| exceeds c.
- Why: Non-relativistic pusher will produce superluminal particles in DPF conditions.
- Expected finding: v > c after ~6000 steps. No clamping.

**Priority 6: `deposit` method dt mismatch**
- Test: Create HybridPIC with self.dt = 1e-10. Call push_particles(dt=1e-11). Call deposit(). Verify J magnitude matches the dt=1e-11 push, not dt=1e-10.
- Why: The deposit method hardcodes self.dt for Esirkepov prefactor.
- Expected finding: J is 10x too large (scaled by self.dt/dt_push = 10).

**Priority 7: `_deposit_current_kernel` (non-Esirkepov) correctness**
- Test: Single particle with known charge, velocity, weight. Verify sum(J)*cell_volume = q*w*v.
- Why: Zero tests exist. This is the fallback when Esirkepov is off.

**Priority 8: `inject_beam` energy and direction**
- Test: Inject beam at 100 keV along z. Verify speed = sqrt(2*E/m) and direction matches within spread tolerance.
- Why: Zero tests. Incorrect speed would invalidate all beam-target physics.

### MODERATE PRIORITY (robustness)

**Priority 9: `_apply_reflecting_bc` corner reflection**
- Test: Particle exits through corner (x<0 AND y<0). Verify it stays in domain after reflection.

**Priority 10: `detect_instability` false positive/negative rates**
- Test: Uniform density (no instability) and peaked density with B_z sign change (instability). Verify correct detection.

**Priority 11: `ParticleSpecies` positions_old initialization**
- Test: Verify `positions_old` is a copy (not a reference) after __post_init__.

**Priority 12: `HybridPIC.add_species` + `enable_collisions` state management**
- Test: Add species, enable collisions, verify _collision_enabled flag propagates to push_particles.

---

## 7. Minimum Viable Fix Set Assessment

The original "38 LOC for V2" fix set likely targets the most obvious bugs. Based on this analysis, the compound interactions require more:

| Fix | LOC Est. | Prevents |
|-----|----------|----------|
| NaN guard in interpolation (return 0.0 for NaN cells) | 8 | NaN chain #1 |
| Pass dt explicitly through deposit() | 5 | dt mismatch bug |
| Velocity clamping in Boris push (0.99c) | 6 | Superluminal runaway |
| Runtime warning for multi-cell crossing in Esirkepov | 10 | Silent charge loss |
| Two-segment deposition for reflected particles | 40 | Boundary charge conservation |
| Relativistic gamma in Boris push | 15 | Physical correctness |
| **Total** | **~84** | |

The 38 LOC fix set is insufficient to handle the compound interaction effects. The ghost-cell NaN chain (Priority 1) alone can kill the simulation in a single step and requires fixes in BOTH the MHD ghost cell handling AND the PIC interpolation.

---

## 8. Recommendations

1. **Do not attempt end-to-end PIC-MHD coupling** until Priority 1-4 tests are written and passing. The ghost-cell NaN chain will produce instant simulation death.

2. **Add a PIC CFL check**: `max(v) * dt < dx` must hold for ALL particles, checked every step. Violation should either sub-cycle the PIC push or raise an error.

3. **The Esirkepov kernel is mathematically correct for interior particles** in the single-cell-crossing regime. It does NOT need a rewrite. It needs boundary handling and a CFL guard.

4. **The Boris pusher needs a relativistic upgrade** before DPF use. The non-relativistic version will produce superluminal particles within microseconds of DPF-relevant fields.

5. **The dt bug on line 1561 is a ticking time bomb**. It will manifest the moment anyone tries adaptive timestepping. Fix it now while it is cheap (5 LOC).
