# Sprint 4 FMEA: MLX Solver Final Validation

**Date**: 2026-03-24
**Author**: dpf-validation-engineer (Cortana)
**Scope**: All MLX solver modules -- failure modes that could prevent Sprint 4 validation pass
**Method**: Code-level inspection of each module against PF-1000 discharge conditions
**Context**: Sprint 4 is final validation. No deferments. Every failure mode below must be resolved or test-gated.

---

## Severity / Occurrence / Detection Scale

| Score | Severity | Occurrence | Detection |
|-------|----------|-----------|-----------|
| 1 | Negligible | Extremely unlikely | Always caught by existing tests |
| 2 | Minor cosmetic | Very unlikely | Caught by standard test suite |
| 3 | Minor functional | Unlikely | Caught by targeted test |
| 5 | Moderate degradation | Occasional | Requires specific diagnostic |
| 7 | Major physics error | Likely | Hard to distinguish from noise |
| 8 | Simulation invalid | Very likely | Only visible in long runs |
| 10 | Crash / NaN / hang | Certain | Silent corruption |

---

## Module 1: mlx_solver.py (Top-Level Solver)

### FM-1.1: eta_field Not Passed Through to Resistive Diffusion

**Input condition**: Engine passes `eta_field` as a NumPy array or scalar via `kwargs`.
**Code path**: `mlx_solver.py:481-489` -- `eta_raw = kwargs.get("eta_field")`, then conversion to `mx.array` or `float`.
**Finding**: The passthrough is CORRECT. `eta_raw` is extracted from kwargs, converted appropriately, and passed to `_do_resistive_diffusion`. No failure mode here.

**Status**: PASS (no action needed)

### FM-1.2: Electrode BC Does Not Handle Negative Current (Post-Pinch Reversal)

**Input condition**: After pinch collapse, current reverses direction (I < 0). `abs(current) > 1e-10` still evaluates True.
**Code path**: `mlx_solver.py:249` -- `Bt_electrode = (_MU0 * current) / (2.0 * math.pi * r_outer)`.
**Finding**: The formula correctly computes negative Bt for negative current. The `abs(current) > 1e-10` gate at line 464 only skips the BC when current is essentially zero, not when it's negative. This is CORRECT behavior -- a reversed current produces reversed Bt, which is physical.

**Status**: PASS (no action needed)

### FM-1.3: Coupling State Does Not Carry Lp or R_plasma

**Input condition**: Engine calls `coupling_interface()` expecting inductance/resistance data for circuit feedback.
**Code path**: `mlx_solver.py:504` -- `self._coupling = CouplingState(current=current, voltage=voltage)`.
**Finding**: Only current and voltage are stored. The `CouplingState` is a pass-through of the values the solver RECEIVED, not computed plasma quantities. If the engine expects `Lp` or `R_plasma` from the solver, it will get default/zero values.
**Failure mode**: Wrong answer -- circuit solver uses stale or zero Lp.
**Effect on PF-1000**: If Lp is zero, back-EMF is zero, current never dips at pinch. The simulation produces a flat I(t) with no pinch signature.

| Metric | Value |
|--------|-------|
| Severity | 8 |
| Occurrence | 3 (depends on engine calling pattern; engine may compute Lp externally) |
| Detection | 5 (shows up as missing current dip, but could be mistaken for wrong fc/fm) |
| **RPN** | **120** |

**Action**: Verify engine.py computes Lp independently from the MHD state dict rather than relying on `coupling_interface()`. If it does, this is a non-issue. If it does not, add Lp/R_plasma computation to the MLX solver step. **Test gate**: Sprint 4 acceptance test M2/S2 (I_peak + dip) will catch this.

---

## Module 2: mlx_timestepper.py (SSP-RK3 + CFL)

### FM-2.1: CFL Returns Infinite dt When Grid Is Entirely at RHO_FLOOR

**Input condition**: Initialization with vacuum state (rho = RHO_FLOOR = 1e-12 everywhere), B = 0, v = 0.
**Code path**: `mlx_timestepper.py:383-404` -- `cons_to_prim` returns p = P_FLOOR, then `fast_magnetosonic` computes `a_sq = gamma * P_FLOOR / RHO_FLOOR = 5/3 * 1e-12 / 1e-12 = 1.67`. `cf = sqrt(1.67) ~ 1.29 m/s`. `max_r = 1.29`. Then `dt = cfl * dx / 1.29`.
**Finding**: This is finite and correct. The CFL does NOT return infinite dt because P_FLOOR/RHO_FLOOR gives a finite sound speed. However, the dt will be very large (cfl * dx / 1.29 ~ 0.3 * 0.001 / 1.29 ~ 2.3e-4 s), which is too large once the electrode BC injects a strong B-field. The first step will overshoot.

**Failure mode**: First timestep is too large because CFL is computed BEFORE electrode BC is applied.
**Effect on PF-1000**: NaN on step 1 or 2 as the electrode B-field creates a shock that the oversized dt cannot resolve.

| Metric | Value |
|--------|-------|
| Severity | 10 |
| Occurrence | 6 (happens whenever solver starts from vacuum with electrode BC) |
| Detection | 2 (immediate NaN crash is obvious) |
| **RPN** | **120** |

**Action**: Fix before Sprint 4. `compute_dt` in `mlx_solver.py:369-390` should apply electrode BC before computing CFL, or apply a safety cap `dt = min(dt_cfl, dt_max_initial)` for the first step. Alternatively, the engine should call `compute_dt` AFTER calling `step` once with a tiny dt to establish the electrode field.

### FM-2.2: Entropy Tracer Advected Correctly Through SSP-RK3 (Not Forward Euler)

**Input condition**: Entropy tracer (ISR) must be advanced with the same SSP-RK3 combination coefficients as all other conserved variables.
**Code path**: `mlx_timestepper.py:557-569` -- SSP-RK3 applies `U + dt*L1`, `0.75*U + 0.25*(U1+dt*L2)`, `1/3*U + 2/3*(U2+dt*L3)` to the FULL state vector U which includes ISR at index 5.
**Finding**: CORRECT. The entropy tracer is part of the packed state vector and receives the same SSP-RK3 combination as all other variables. The `_resync_energy` at each stage (line 552) resets IEN from entropy-derived pressure but does NOT modify ISR itself. The `_apply_floors` (line 427) only clamps ISR >= 0, which is correct.

**Status**: PASS (no action needed)

### FM-2.3: _apply_floors Uses mx.split Which Creates (1,nr,nz) Slices, Then [0] Indexing

**Input condition**: Any state vector.
**Code path**: `mlx_timestepper.py:424-428` -- `rows = list(mx.split(U, NVAR, axis=0))` produces list of `(1, nr, nz)` arrays. Then `rows[IDN] = mx.maximum(rows[IDN], RHO_FLOOR)`. Finally `mx.stack([r[0] for r in rows], axis=0)`.
**Finding**: The `r[0]` indexing removes the leading dimension from each `(1, nr, nz)` slice back to `(nr, nz)`, then `mx.stack` produces `(10, nr, nz)`. This is correct BUT fragile: if `mx.split` changes behavior or if `rows[IDN]` is replaced by a result without the leading dimension (which `mx.maximum` preserves), the `r[0]` indexing would fail.

More critically: `rows[ISR] = mx.maximum(rows[ISR], 0.0)` is applied to the `(1, nr, nz)` shape, and the scalar 0.0 broadcasts correctly. The `[0]` indexing then reduces it. This works.

**Failure mode**: Dimension mismatch if floor logic produces arrays of different shapes.

| Metric | Value |
|--------|-------|
| Severity | 10 |
| Occurrence | 1 (currently works; only breaks on refactoring) |
| Detection | 1 (immediate shape error crash) |
| **RPN** | **10** |

**Action**: Low priority. Consider refactoring to use direct indexing (`U[IDN]`) instead of `mx.split` for clarity, but no Sprint 4 urgency.

### FM-2.4: _clamp_velocity Uses Same Pattern -- Potential Shape Issue

**Input condition**: Velocity exceeds `_V_CLAMP_FACTOR * cf`.
**Code path**: `mlx_timestepper.py:472-477` -- `rows[IMR] = (rho * vr_c)[None]`. The `[None]` adds a leading dim to make `(1, nr, nz)` matching the `mx.split` output shape. Then `r[0]` in stack removes it.
**Finding**: CORRECT. The `[None]` indexing is intentional to match the `(1, nr, nz)` shape from `mx.split`.

**Status**: PASS

---

## Module 3: mlx_riemann.py (Spatial Operator / Flux Divergence)

### FM-3.1: Two Conflicting mhd_rhs Implementations

**Input condition**: Import resolution between `mlx_timestepper.py` and `mlx_riemann.py`.
**Code path**: `mlx_timestepper.py:59` defines `mhd_rhs()`. `mlx_riemann.py:130` also defines `mhd_rhs()`.
**Finding**: `mlx_timestepper.py` imports from `mlx_kernels` (line 39: `hlld_flux_mlx`) and uses its OWN `mhd_rhs` at line 557. `mlx_riemann.py` has a separate `mhd_rhs` that uses `compute_fluxes` from itself. The `ssp_rk3_step` function calls the local `mhd_rhs` in `mlx_timestepper.py`, NOT the one in `mlx_riemann.py`.

**Failure mode**: Two different MHD RHS implementations may produce different results. If the wrong one is called, physics is wrong.
**Effect on PF-1000**: Depends on which path is active. The `mlx_timestepper.py` version (used by SSP-RK3) has a simpler flux divergence that uses zero-padding for boundary cells, while `mlx_riemann.py` has a more careful r-weighted cylindrical divergence with proper face-radius weighting.

| Metric | Value |
|--------|-------|
| Severity | 7 |
| Occurrence | 8 (always active -- whichever is used, the other is dead code or inconsistent) |
| Detection | 7 (produces subtly different results, not NaN; would only show in careful convergence study) |
| **RPN** | **392** |

**Action**: **Fix before Sprint 4.** Determine which `mhd_rhs` is correct and delete the other. The `mlx_riemann.py` version appears more carefully implemented (proper face-radius weighting, proper ghost-cell handling, explicit r-weighted divergence). The `mlx_timestepper.py` version is simpler but handles the HLL fallback path. Recommendation: make `mlx_timestepper.py` import and use `mhd_rhs` from `mlx_riemann.py`, and delete the duplicate.

### FM-3.2: Flux Divergence Boundary Cell Handling with WENO5-Z

**Input condition**: WENO5-Z produces `nr-5` interfaces from `nr` cells. For a 16-cell grid, only 11 interfaces, updating 10 interior cells. The outermost 3 cells on each side get zero flux divergence.
**Code path**: `mlx_timestepper.py:100-132` -- Pads F_r to `nr-1` interfaces with zeros, then pads rF with zeros at face 0 and face nr.
**Finding**: The zero-padded boundary interfaces mean the 2-3 outermost cells on each side receive NO flux divergence update. They only evolve via geometric source terms. Over many timesteps, these cells drift from the interior solution, creating a boundary layer artifact.

In `mlx_riemann.py:183-241`, the approach is different: the flux is NOT padded to `nr-1`. Instead, only `n_updated_r = n_ifaces_r - 1` interior cells are updated, and the boundary cells are explicitly left at zero dU/dt. This is cleaner but has the same physical limitation: boundary cells are frozen except for sources.

**Failure mode**: Boundary cell stagnation creates unphysical density/pressure gradients near the electrode that contaminate the radial flux divergence at adjacent interior cells.
**Effect on PF-1000**: The electrode is at the outer boundary. If the 2-3 cells near the cathode don't evolve via fluxes, the B_theta profile near the electrode is wrong, affecting the J x B force and current distribution.

| Metric | Value |
|--------|-------|
| Severity | 6 |
| Occurrence | 7 (always happens with WENO5-Z on grids < ~32 radial cells) |
| Detection | 6 (subtle error in B-field profile near boundary; shows as wrong I_peak) |
| **RPN** | **252** |

**Action**: **Fix before Sprint 4.** For boundary cells where WENO5-Z doesn't have enough stencil, fall back to PLM reconstruction. This is standard practice (Athena++ does this). Implementation: in `mhd_rhs`, compute PLM fluxes for the 2 boundary cells on each side and merge with WENO5-Z interior fluxes.

### FM-3.3: mlx_riemann.py Geometric Source Conversion Error

**Input condition**: Cylindrical geometry with non-zero vtheta and Btheta.
**Code path**: `mlx_riemann.py:262-293` -- Converts to primitive, calls `cylindrical_source_mlx`, then converts back.
**Finding**: The source kernel returns primitive-space accelerations (velocity sources, not momentum sources). The code at line 270 converts: `dmr = rho * src[IMR]`. This is correct IF `src[IMR]` is an acceleration (m/s^2). But `cylindrical_source_mlx` returns sources in the primitive variable layout where index 1 is vr, and the source for momentum-per-unit-mass IS acceleration.

However, `src[IBT]` at line 272 is used directly as `dBt = src[IBT]`. The source for Bt in cylindrical coordinates is `d(Bt)/dt = ...` which is already in the correct units. Line 132 applies `U[IBT] + src[IBT] * dt` in `mlx_sources.py:apply_geometric_sources`, but in `mlx_riemann.py:293` it's added as `dU_dt`, so the dt multiplication happens in the RK integrator. This is correct.

But there is an inconsistency: `mlx_riemann.py` adds geometric sources as dU/dt terms to the RHS (no dt multiplication), while `mlx_sources.py:apply_geometric_sources` multiplies by dt internally. These are two different calling conventions. If both are called, the geometric source is double-counted.

**Failure mode**: Geometric sources double-counted or applied in wrong units.
**Effect on PF-1000**: Incorrect centrifugal/hoop stress balance at the pinch column. Could cause premature or delayed pinch, affecting I_dip timing.

| Metric | Value |
|--------|-------|
| Severity | 7 |
| Occurrence | 3 (only if both source paths are called; need to verify call chain) |
| Detection | 7 (subtle timing error in I_dip, not obvious crash) |
| **RPN** | **147** |

**Action**: **Verify before Sprint 4.** Trace the call chain from `ssp_rk3_step` to confirm which geometric source path is used. If `mhd_rhs` in `mlx_timestepper.py` is the active path, it calls `_geometric_sources` at line 159 (its own version). If `mhd_rhs` from `mlx_riemann.py` is used, it calls `cylindrical_source_mlx`. These may produce different results. Unify to one path.

---

## Module 4: mlx_primitives.py (Dual-Energy Recovery)

### FM-4.1: recover_pressure_dual_energy Returns Wrong Pressure When E < 0

**Input condition**: Total energy E becomes negative due to float32 accumulation errors or excessive radiation cooling.
**Code path**: `mlx_primitives.py:186-198` -- `p_E = gm1 * (E - KE - ME)`. If E < KE + ME (which happens at beta < 1e-6), `p_E` is negative. `p_S = Srho * rho^(gm1)` is always positive (Srho >= 0 from floor). The blend: `p = w * p_E + (1-w) * p_S`. When eta is small (low beta), `w -> 0`, so `p -> p_S` (correct). When eta is large (high beta), `w -> 1`, so `p -> p_E` (could be negative, but P_FLOOR catches it at line 197).
**Finding**: CORRECT. The P_FLOOR at line 197 prevents negative pressure output. The switching criterion correctly falls back to entropy at low beta where p_E is corrupted.

**Status**: PASS

### FM-4.2: Entropy Tracer Srho Can Be Negative After Bremsstrahlung Cooling

**Input condition**: Strong bremsstrahlung removes energy from total E but NOT from Srho (bremsstrahlung in `mlx_sources.py:289-355` only modifies U[IEN], not U[ISR]).
**Code path**: `mlx_sources.py:343-354` -- `U[ISR]` is unchanged after bremsstrahlung.
**Finding**: Srho is NOT updated for radiation losses. Over time, Srho overestimates the true entropy (it reflects the pre-radiation state). When the switching criterion selects entropy-derived pressure (`w -> 0`), `p_S` is higher than the true post-radiation pressure. This causes the simulation to see too-high pressure in cooled regions.

**Failure mode**: Wrong answer -- pressure overestimated in regions with strong bremsstrahlung.
**Effect on PF-1000**: The pinch column has the strongest bremsstrahlung. Overestimated pressure resists compression, producing a wider pinch column and lower neutron yield.

| Metric | Value |
|--------|-------|
| Severity | 6 |
| Occurrence | 7 (always happens when bremsstrahlung is enabled) |
| Detection | 8 (requires comparing p_S vs p_E in the pinch region; not caught by standard tests) |
| **RPN** | **336** |

**Action**: **Fix before Sprint 4.** After bremsstrahlung modifies U[IEN], update U[ISR] consistently: `Srho_new = p_new * rho^(1-gamma)` where `p_new` is the post-radiation pressure. Add ~5 LOC to `apply_bremsstrahlung`.

---

## Module 5: mlx_kernels.py (HLLD Metal Kernel)

### FM-5.1: HLLD Entropy Flux Uses Upwind vn, Not Contact Speed SM

**Input condition**: Entropy tracer at a contact discontinuity.
**Code path**: `mlx_kernels.py:471-472` -- `FL[5] = Srho_L * vn_L; FR[5] = Srho_R * vn_R`.
**Finding**: The physical flux for entropy is `F_S = Srho * v_n`. The HLLD STAR state entropy at line 484 is `UsL[5] = Srho_L * (SL - vn_L) / denom_L`. This is the Rankine-Hugoniot jump condition for a passively advected scalar, which is correct. The star-state entropy is carried through the contact at speed SM via the flux selection logic at lines 513-518.

**Status**: PASS (correct passive scalar advection through HLLD)

### FM-5.2: HLLD D_L Denominator Near-Zero Produces Large Tangential Velocities

**Input condition**: At the Alfven resonance point where `rho * (S - vn) * (S - SM) ~ Bn^2`, D_L -> 0.
**Code path**: `mlx_kernels.py:409-413` -- `D_L = rho_L * (SL - vn_L) * (SL - SM) - Bn * Bn`. If `|D_L| < TINY`, `inv_DL = 0.0`. This zeros out the correction terms at lines 415-425, falling back to the input tangential velocities.
**Finding**: When `inv_DL = 0`, the star-state tangential velocities equal the input values: `vt1_sL = vt1_L`, `Bt1_sL = Bt1_L * fL * 0 = 0`. Wait -- line 422: `Bt1_sL = Bt1_L * fL * inv_DL`. If `inv_DL = 0`, then `Bt1_sL = 0`. But `fL = rho_L * (SL-vn_L)^2 - Bn^2`, which is close to D_L (same expression with `SL-vn_L` instead of `SL-SM`). When D_L -> 0, fL may also be near zero, so `Bt1_sL -> 0/0 = 0` via the floor.

**Failure mode**: Tangential B-field set to zero in cells near Alfven resonance. This is overly diffusive but not catastrophic.
**Effect on PF-1000**: At the pinch column boundary, Alfven resonance can occur. Zeroing Bt there removes magnetic pressure support, potentially causing over-compression.

| Metric | Value |
|--------|-------|
| Severity | 5 |
| Occurrence | 4 (only at Alfven resonance, which is a narrow region) |
| Detection | 7 (requires examining Bt profile at pinch boundary; not obvious in I(t)) |
| **RPN** | **140** |

**Action**: Test gate during Sprint 4. Monitor the LF fallback trigger rate. If > 1% of interfaces trigger the NaN/Inf fallback, investigate further.

### FM-5.3: HLLD Physical Flux F[5] (Entropy) Set to 0.0 Inside physical_flux()

**Input condition**: All states.
**Code path**: `mlx_kernels.py:341` -- `F[5] = 0.0f`.
**Finding**: The physical flux function sets the entropy flux to zero. The actual entropy flux is computed OUTSIDE `physical_flux()` at lines 471-472: `FL[5] = Srho_L * vn_L`. The physical_flux function computes the Euler+MHD flux but NOT the entropy flux. This is correct because entropy is a passive scalar and its upwind flux is handled separately through the star states.

**Status**: PASS

### FM-5.4: HLLD dim=1 Tangential Index Mapping for Axial Direction

**Input condition**: Axial flux computation (dim=1).
**Code path**: `mlx_kernels.py:362-366` -- For `dim=1`: `im_n=2, im_t1=3, im_t2=1, ib_n=7, ib_t1=8, ib_t2=6`.
**Finding**: The axial normal is IMZ=2, tangent1 is IMT=3, tangent2 is IMR=1. For B: normal is IBZ=7, tangent1 is IBT=8, tangent2 is IBR=6. This maps (vz, vtheta, vr) and (Bz, Btheta, Br) as (normal, tangent1, tangent2).

This is INCONSISTENT with the `mlx_timestepper.py:_hll_flux` at line 219 which uses `im_n=2, im_t1=1, im_t2=3` (tangent1=vr, tangent2=vtheta). The HLLD and HLL use different tangential ordering for dim=1.

**Failure mode**: Wrong answer when HLL fallback is triggered for axial fluxes -- the tangential velocity/field components are swapped.

| Metric | Value |
|--------|-------|
| Severity | 7 |
| Occurrence | 3 (only when HLL fallback is used in axial direction, which requires NaN in HLLD) |
| Detection | 8 (extremely subtle; only produces wrong answer if both HLL and HLLD are used in same run, and only for axial direction) |
| **RPN** | **168** |

**Action**: **Fix before Sprint 4.** Unify the tangential index mapping between HLLD and HLL. The HLLD mapping (im_t1=3=IMT, im_t2=1=IMR for dim=1) appears physically correct (tangent to z-axis is theta first, then r). The HLL in `mlx_timestepper.py` uses the opposite order. Correct HLL to match HLLD.

---

## Module 6: mlx_reconstruction.py (WENO5-Z)

### FM-6.1: WENO5-Z Falls Back to PLM for Grids < 6 Cells

**Input condition**: Grid with < 6 cells in one dimension (e.g., 4x32 for a coarse radial test).
**Code path**: `mlx_reconstruction.py:269-270` -- `if n < 6: return plm_reconstruct(Q, dim=dim, limiter="mc")`.
**Finding**: Correct graceful degradation. PLM with MC limiter is 2nd-order and stable.

**Status**: PASS

### FM-6.2: WENO5-Z eps=1e-6 Too Large for Smooth Data in Float32

**Input condition**: Smooth data (e.g., during rundown phase with no shocks).
**Code path**: `mlx_reconstruction.py:228-233` -- `a0 = d0 * (1.0 + (tau5 / (eps + beta0)) ** 2)`. With eps=1e-6 and smooth data where beta ~ 1e-12, the weight becomes `d0 * (1 + (tau5 / 1e-6)^2)`. Since tau5 ~ beta ~ 1e-12 for smooth data, `tau5/(eps+beta) ~ 1e-12/1e-6 = 1e-6`, and `(1e-6)^2 = 1e-12`. So `alpha ~ d0 * (1 + 1e-12) ~ d0`. The ideal weights are recovered. No issue.

For discontinuous data where beta0 ~ 1 and beta2 ~ 1e-4: `tau5 = |beta0 - beta2| ~ 1`. Then `alpha_0 = d0 * (1 + (1/(1e-6+1))^2) ~ d0 * 2`. And `alpha_2 = d2 * (1 + (1/(1e-6+1e-4))^2) ~ d2 * (1 + 1e8) ~ d2 * 1e8`. So alpha_2 dominates, which is the stencil that doesn't cross the discontinuity. This is correct ENO-like behavior.

**Status**: PASS

---

## Module 7: mlx_sources.py (Source Terms)

### FM-7.1: Bremsstrahlung Coefficient Correctly Handles Float32 Subnormal

**Input condition**: The coefficient 1.42e-40 is subnormal in float32 (smallest normal float32 is ~1.18e-38).
**Code path**: `mlx_sources.py:323-332` -- Computation is done in NumPy float64 via `np.asarray(...).astype(np.float64)`, then cast back to float32 at line 331.
**Finding**: CORRECT. The subnormal coefficient is handled in float64 where it is well within normal range. The final result `Q_rad` is in float32, but by then the coefficient has been multiplied by `ne^2 * sqrt(Te)` which produces values in the W/m^3 range (representable in float32).

**Status**: PASS

### FM-7.2: Geometric Source at Axis (r=0) with inv_r

**Input condition**: First radial cell at r = dr/2 (e.g., r = 0.5 mm for dr = 1 mm).
**Code path**: `mlx_sources.py:100-137` calls `cylindrical_source_mlx(Q, r_cell, inv_r, gamma)`. The `inv_r` array is precomputed from `CylindricalGrid` which uses L'Hopital at r=0.
**Finding**: Need to verify the `CylindricalGrid.inv_r` implementation. The `mlx_timestepper.py:342` uses `grid.inv_r[:, None]` which broadcasts correctly. The key question is what value `inv_r[0]` takes.

If `r_inner = 0`, then `r_cell[0] = dr/2`. `inv_r[0] = 1 / (dr/2) = 2/dr`. This is a large but finite value (e.g., 2000 for dr = 1 mm). The source terms `S_mr = (rho*vt^2 - Bt^2) * inv_r[0]` scale as 2000 * (rho*vt^2 - Bt^2). For a uniform initial state with vt=0 and Bt=0, this is zero. For pinch conditions with Bt ~ 100 (HL) at r = 0.5 mm: `S_mr = -100^2 * 2000 = -2e7`. This is a large inward force, which is physical (hoop stress).

The axis singularity is avoided because the first cell center is at dr/2, not at r=0. No NaN possible from division.

**Status**: PASS (contingent on CylindricalGrid setting r_cell[0] = dr/2, not 0)

### FM-7.3: apply_geometric_sources Multiplies Source by dt Internally

**Input condition**: Any geometric source application.
**Code path**: `mlx_sources.py:118-120` -- `dmr = rho * src[1] * dt`.
**Finding**: This function applies the source as a forward-Euler update: `U_new = U + S * dt`. This is NOT compatible with the SSP-RK3 approach where dU/dt terms are returned and the time integrator handles dt multiplication. If `apply_geometric_sources` is called during the RK stages, the dt multiplication is doubled (once here, once in the RK combination).

This function is NOT called by the SSP-RK3 path (which uses `_geometric_sources` in `mlx_timestepper.py:314` or `cylindrical_source_mlx` in `mlx_riemann.py:266`). It appears to be a standalone operator-split interface.

**Failure mode**: Double dt application if accidentally called within RK stages.

| Metric | Value |
|--------|-------|
| Severity | 8 |
| Occurrence | 2 (only if called from wrong location) |
| Detection | 3 (produces obviously wrong results -- factor-of-dt error) |
| **RPN** | **48** |

**Action**: Low priority. Add a warning docstring or rename to `apply_geometric_sources_operator_split` to prevent misuse.

### FM-7.4: Ohmic Heating Entropy Update Uses Inconsistent Formula

**Input condition**: Ohmic heating with nonzero eta and J.
**Code path**: `mlx_sources.py:270-272` -- `dSrho = Q_ohm * dt * (gamma-1) * inv_p * rho`.
**Finding**: The entropy tracer Srho = p * rho^(1-gamma). Taking d(Srho)/dt from ohmic heating:
`d(Srho)/dt = dp/dt * rho^(1-gamma) + p * (1-gamma) * rho^(-gamma) * drho/dt`.
For pure heating (no density change): `d(Srho)/dt = dp/dt * rho^(1-gamma)`.
With `dp/dt = (gamma-1) * Q_ohm` (from energy equation): `d(Srho)/dt = (gamma-1) * Q_ohm * rho^(1-gamma)`.
So `dSrho = (gamma-1) * Q_ohm * dt * rho^(1-gamma)`.

The code computes: `dSrho = Q_ohm * dt * (gamma-1) * (1/p) * rho`.
Since `Srho = p * rho^(1-gamma)`, we have `1/p = rho^(1-gamma) / Srho`.
So the code's formula is: `dSrho = (gamma-1) * Q_ohm * dt * rho / p`.
Since `rho / p = rho / (Srho * rho^(gamma-1)) = 1 / (Srho * rho^(gamma-2))`.

The correct formula is: `dSrho = (gamma-1) * Q_ohm * dt * rho^(1-gamma)`.
The code's formula is: `dSrho = (gamma-1) * Q_ohm * dt * rho / p`.
These are equal ONLY IF `rho / p = rho^(1-gamma)`, i.e., `p = rho^gamma`, which is true only for the specific normalization Srho = 1. In general, `p = Srho * rho^(gamma-1)`, so `rho / p = rho / (Srho * rho^(gamma-1)) = 1 / (Srho * rho^(gamma-2))`.

The correct formula should be `rho^(1-gamma)`, not `rho / p`. These differ by a factor of `Srho`.

**Failure mode**: Wrong entropy update magnitude -- Srho increments are wrong by a factor proportional to the current entropy value.
**Effect on PF-1000**: In high-entropy regions (hot plasma), the entropy increment from ohmic heating is underestimated (because p is large, so rho/p is small). In low-entropy regions, it is overestimated. This biases the dual-energy pressure recovery.

| Metric | Value |
|--------|-------|
| Severity | 5 |
| Occurrence | 5 (whenever ohmic heating is active, which is always during compression) |
| Detection | 8 (only visible by comparing p_S vs p_E in ohmic heating regions; no obvious crash) |
| **RPN** | **200** |

**Action**: **Fix before Sprint 4.** Replace `dSrho = Q_ohm * dt * (gamma - 1.0) * inv_p * rho` with `dSrho = Q_ohm * dt * (gamma - 1.0) * mx.power(mx.maximum(rho, 1e-30), 1.0 - gamma)`.

---

## Module 8: mlx_ct.py (Constrained Transport)

### FM-8.1: CT Module Not Wired Into SSP-RK3 Pipeline

**Input condition**: Any simulation with use_ct=True.
**Code path**: `mlx_solver.py` -- search for "ct" or "constrained". Neither `compute_emf`, `apply_ct`, nor any CT function is called in the `step()` method or in `ssp_rk3_step()`.
**Finding**: The CT module exists as standalone functions but is NOT called anywhere in the solver pipeline. The solver evolves cell-centered B-fields through the Riemann solver's induction equation, which does NOT maintain div(B)=0.

**Failure mode**: div(B) accumulates over time, producing unphysical magnetic monopoles.
**Effect on PF-1000**: Magnetic monopoles create spurious J x B forces. Over a full discharge (~10,000 steps), div(B) errors accumulate and can create unphysical current channels, destabilizing the pinch.

| Metric | Value |
|--------|-------|
| Severity | 7 |
| Occurrence | 10 (always -- CT is never called) |
| Detection | 3 (div(B) is easy to compute as a diagnostic; test M8 in DoD will catch this) |
| **RPN** | **210** |

**Action**: **Fix before Sprint 4.** Wire CT into the SSP-RK3 pipeline. After each RK stage, extract face-centered B from cell-centered B (averaging), compute EMF from velocities, and apply CT update. Then project face-centered B back to cell centers. This is ~50 LOC in the timestepper. Alternatively, accept cell-centered B with Powell 8-wave cleaning as a simpler alternative, but this does NOT satisfy DoD criterion M8 (`max |div(B)| * dx / max |B| < 1e-6`).

### FM-8.2: CT div_B_cylindrical at Axis (r=0) Uses inv_r with 1e-30 Floor

**Input condition**: Computing div(B) for verification.
**Code path**: `mlx_ct.py:224` -- `inv_r = 1.0 / mx.maximum(r_cell_col, mx.array(1e-30, dtype=mx.float32))`.
**Finding**: At the axis cell (r_cell = dr/2 ~ 5e-4 m), `inv_r ~ 2000`, which is finite. The 1e-30 floor only activates if r_cell is identically zero, which should not happen with the standard grid construction. No NaN risk.

**Status**: PASS

---

## Module 9: mlx_transport.py (Thomas Solver / Resistive Diffusion)

### FM-9.1: Thomas Solver Boundary Conditions (Neumann/Zero-Flux)

**Input condition**: B-field diffusion along z or r.
**Code path**: `mlx_transport.py:136-137` -- `alpha_left = np.concatenate([[0.0], alpha_face])` and `alpha_right = np.concatenate([alpha_face, [0.0]])`. This sets the boundary face diffusivity to zero, implementing zero-flux (Neumann) BCs.
**Finding**: CORRECT for axial boundaries (z=0 anode wall, z=zmax open end treated as insulating). For the radial direction, the cylindrical system (`_build_cylindrical_diffusion_system` at line 180) uses `alpha_m = np.concatenate([[alpha[0]], ...])` and `alpha_p = np.concatenate([..., [alpha[-1]]])`. This extrapolates the boundary diffusivity rather than zeroing it.

At the outer radial boundary (cathode), the zero-gradient BC on B is correct (the electrode prescribes B_theta, not its gradient). At the axis (r=0), the reflecting BC requires dB/dr = 0 for Bz and Bt, and B_r = 0. The current implementation uses Neumann BCs for both boundaries (via extrapolation), which is approximately correct.

**Status**: PASS (acceptable for Neumann BCs)

### FM-9.2: Resistive Diffusion Applied to Already-Diffused Fields

**Input condition**: Both z-diffusion and r-diffusion in sequence.
**Code path**: `mlx_transport.py:278-293` -- z-diffusion loop uses `Br_np` (original) to build the system but writes to `Br_new`. Then r-diffusion loop at line 289 uses `Br_np` (original again, NOT `Br_new`).
**Finding**: The r-diffusion at line 289 uses the ORIGINAL `Br_np` (line 289: `for field, field_new in [(Br_np, Br_new), ...]`), not the z-diffused result. This means z-diffusion and r-diffusion are applied to the same INPUT, and the results are written to the same OUTPUT. The second (r) diffusion OVERWRITES the z-diffusion result.

**Failure mode**: z-direction diffusion is effectively ignored. Only r-direction diffusion survives.
**Effect on PF-1000**: Axial diffusion of B-field is missing. The pinch column's axial B-field structure (which sets the axial pressure gradient at the pinch ends) is not diffused, leading to artificially sharp axial gradients.

| Metric | Value |
|--------|-------|
| Severity | 6 |
| Occurrence | 8 (always when both nr > 1 and nz > 1, which is the normal case) |
| Detection | 8 (no crash; only visible by comparing diffusion convergence in z-direction, which is unlikely to be tested) |
| **RPN** | **384** |

**Action**: **Fix before Sprint 4.** Change the r-diffusion loop to use `Br_new` (the z-diffused result) as input:
```python
# Line 289-293: change from Br_np to Br_new
for field, field_new in [(Br_new, Br_new), (Bz_new, Bz_new), (Bt_new, Bt_new)]:
```
This makes the diffusion sequential (z first, then r on the z-diffused result), which is the standard Strang-split approach.

### FM-9.3: Ohmic Heating Uses dB^2 Not J^2

**Input condition**: Ohmic heating computation after resistive diffusion.
**Code path**: `mlx_transport.py:302-310` -- `dB_sq = (Br_new - Br_np)^2 + ...` then `Q_ohmic = 0.5 * dB_sq * MU_0`.
**Finding**: This estimates ohmic heating from the change in magnetic energy, which is energy-conserving by construction (energy removed from B goes to thermal). This is acceptable as an approximation, though it slightly underestimates heating when the field topology changes (e.g., reconnection). The `MU_0` factor converts from HL to SI units correctly.

However, due to FM-9.2 above, `Br_new - Br_np` only reflects r-direction diffusion (z-direction result was overwritten). The ohmic heating from z-direction diffusion is lost.

| Metric | Value |
|--------|-------|
| Severity | 4 |
| Occurrence | 8 (same as FM-9.2) |
| Detection | 8 (same as FM-9.2) |
| **RPN** | **256** |

**Action**: Fixed by fixing FM-9.2.

### FM-9.4: Thomas Solver Division by Zero When b[0] = 0

**Input condition**: If the main diagonal b[0] = 0 (should not happen with the floor `b = np.maximum(b, 1.0)` at line 140/194).
**Code path**: `mlx_transport.py:72` -- `c_prime[0] = c[0] / b[0]`.
**Finding**: With `b = np.maximum(b, 1.0)`, `b[0] >= 1.0` always. No division by zero possible.

However, the forward sweep at line 76-78: `denom = b[i] - a[i-1] * c_prime[i-1]`. If `a[i-1] * c_prime[i-1] >= b[i]`, denom could be zero or negative. With implicit diffusion, `b[i] = 1 + coeff_p + coeff_m >= 1`, `|a[i-1]| = coeff_m[i] <= b[i] - 1`, and `|c_prime[i-1]| < 1` by construction of the Thomas algorithm for diagonally dominant systems. So `|a[i-1] * c_prime[i-1]| < |a[i-1]| < b[i]`, ensuring denom > 0.

**Status**: PASS (diagonally dominant system guarantees no division by zero)

---

## Module 10: mlx_state.py (State Dict Conversion)

### FM-10.1: State Dict Conversion Handles Negative Pressure

**Input condition**: Engine passes `state["pressure"]` with negative values (e.g., from a corrupted Python engine state).
**Code path**: `mlx_state.py` (not fully read, but `prim_to_cons` in `mlx_primitives.py:122-123` applies `p_safe = mx.maximum(p, P_FLOOR)`).
**Finding**: Negative pressure input is floored to P_FLOOR. The entropy tracer is computed from the floored pressure: `Srho = P_FLOOR * rho^(1-gamma)`. This is a low but positive entropy value. Subsequent evolution will recover the correct pressure from the MHD dynamics.

**Status**: PASS

### FM-10.2: State Dict Conversion Handles Negative Density

**Input condition**: Engine passes `state["rho"]` with negative values.
**Code path**: `mlx_primitives.py:122` -- `rho_safe = mx.maximum(rho, RHO_FLOOR)`.
**Finding**: Negative density is floored to RHO_FLOOR = 1e-12. The energy and entropy are computed from the floored density. No NaN or crash.

**Status**: PASS

---

## Risk Priority Number (RPN) Summary

| Rank | ID | Module | Failure Mode | S | O | D | RPN | Action |
|------|----|--------|-------------|---|---|---|-----|--------|
| 1 | FM-3.1 | mlx_riemann/timestepper | Two conflicting mhd_rhs implementations | 7 | 8 | 7 | **392** | Fix before Sprint 4 |
| 2 | FM-9.2 | mlx_transport | z-diffusion overwritten by r-diffusion | 6 | 8 | 8 | **384** | Fix before Sprint 4 |
| 3 | FM-4.2 | mlx_sources | Bremsstrahlung does not update entropy tracer | 6 | 7 | 8 | **336** | Fix before Sprint 4 |
| 4 | FM-3.2 | mlx_riemann/timestepper | WENO5-Z boundary cells get zero flux | 6 | 7 | 6 | **252** | Fix before Sprint 4 |
| 5 | FM-9.3 | mlx_transport | Ohmic heating misses z-diffusion component | 4 | 8 | 8 | **256** | Fixed by FM-9.2 |
| 6 | FM-8.1 | mlx_ct | CT not wired into SSP-RK3 pipeline | 7 | 10 | 3 | **210** | Fix before Sprint 4 |
| 7 | FM-7.4 | mlx_sources | Ohmic entropy update formula wrong | 5 | 5 | 8 | **200** | Fix before Sprint 4 |
| 8 | FM-5.4 | mlx_kernels | HLL/HLLD tangential index mismatch for dim=1 | 7 | 3 | 8 | **168** | Fix before Sprint 4 |
| 9 | FM-3.3 | mlx_riemann | Geometric source double-count risk | 7 | 3 | 7 | **147** | Verify before Sprint 4 |
| 10 | FM-5.2 | mlx_kernels | HLLD D_L near-zero zeros tangential B | 5 | 4 | 7 | **140** | Test gate during Sprint 4 |
| 11 | FM-2.1 | mlx_timestepper | CFL too large on first step from vacuum | 10 | 6 | 2 | **120** | Fix before Sprint 4 |
| 12 | FM-1.3 | mlx_solver | CouplingState missing Lp/R_plasma | 8 | 3 | 5 | **120** | Verify before Sprint 4 |
| 13 | FM-7.3 | mlx_sources | apply_geometric_sources multiplies dt internally | 8 | 2 | 3 | **48** | Low priority |
| 14 | FM-2.3 | mlx_timestepper | mx.split fragile shape pattern | 10 | 1 | 1 | **10** | Low priority |

---

## Critical Path: Must-Fix Before Sprint 4 Starts

### Priority 1: Structural Issues (RPN > 300)
1. **FM-3.1 (RPN 392)**: Eliminate duplicate `mhd_rhs` -- unify on `mlx_riemann.py` version.
2. **FM-9.2 (RPN 384)**: Fix z-then-r diffusion overwrite in Thomas solver.
3. **FM-4.2 (RPN 336)**: Add entropy tracer update to bremsstrahlung cooling.

### Priority 2: Physics Correctness (RPN 200-300)
4. **FM-3.2 (RPN 252)**: Add PLM fallback for WENO5-Z boundary cells.
5. **FM-8.1 (RPN 210)**: Wire CT into SSP-RK3 pipeline.
6. **FM-7.4 (RPN 200)**: Fix ohmic heating entropy formula.

### Priority 3: Edge Cases (RPN 100-200)
7. **FM-5.4 (RPN 168)**: Fix HLL tangential index order for dim=1.
8. **FM-2.1 (RPN 120)**: Cap initial CFL dt for vacuum starts.

### Verify Only (no code change expected)
9. **FM-1.3 (RPN 120)**: Verify engine computes Lp independently.
10. **FM-3.3 (RPN 147)**: Verify geometric source is not double-counted.

---

## Test Gates for Sprint 4

These failure modes should be monitored during Sprint 4 validation, not necessarily fixed beforehand:

| FM | Test | Pass Criterion |
|----|------|----------------|
| FM-5.2 | LF fallback rate | < 1% of interfaces trigger NaN fallback |
| FM-8.1 | div(B) diagnostic | max |div(B)| * dx / max |B| < 1e-6 (DoD M8) |
| FM-4.2 | p_S vs p_E in pinch | |p_S - p_E| / p_E < 10% in pinch column |
| FM-2.1 | First-step stability | No NaN on step 1 with vacuum IC + electrode BC |
| FM-3.2 | Boundary cell drift | Electrode-adjacent cell density stable over 100 steps |

---

## Relationship to Existing Risk Analysis (PHASE_B_RISK_ANALYSIS.md)

| Existing Risk | FMEA Confirmation | Status |
|---------------|-------------------|--------|
| N1 (WENO5-Z+HLLD contacts) | Not directly found; HLLD implementation is robust | Mitigated by LF fallback |
| N2 (Float32 HLLD overflow) | FM-5.2 confirms D_L near-zero case | Mitigated by TINY floor + LF fallback |
| N3 (Entropy at shocks) | FM-4.2 and FM-7.4 are NEW findings beyond N3 | ELEVATED -- entropy not updated for brem/ohmic |
| N4 (CFL dual-energy) | FM-2.1 found a specific WORSE case: vacuum start | ELEVATED -- vacuum CFL is immediate crash |
| N5 (Axis singularity) | FM-7.2 confirms axis is handled correctly | Mitigated by dr/2 cell center |
| N6 (Blending drift) | Not confirmed as a problem in code review | Acceptable |
| I1 (Backend naming) | Not in scope for FMEA (design issue) | Deferred |
| V1 (fc/fm recalibration) | Still applicable | Sprint 4 task |
| NEW: FM-3.1 | Two conflicting mhd_rhs -- not in original risk analysis | **CRITICAL NEW FINDING** |
| NEW: FM-9.2 | Diffusion overwrite -- not in original risk analysis | **CRITICAL NEW FINDING** |
| NEW: FM-8.1 | CT not wired -- not in original risk analysis | **HIGH NEW FINDING** |
