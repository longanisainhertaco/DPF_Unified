# Phase B Risk Analysis: MLX MHD Solver Remaining Sprints

**Date**: 2026-03-24
**Author**: dpf-validation-engineer (Cortana)
**Scope**: Sprints 2-4 (time integration, solver assembly, DPF validation)
**Status**: Risk assessment for implementation decisions
**Context**: Sprint 1 complete (WENO5-Z reconstruction + dual-energy primitives built)

---

## 1. Numerical Risks

### N1. WENO5-Z + HLLD Interaction: Oscillation at Contact Discontinuities

**Likelihood**: Medium
**Impact**: High

**Root cause**: WENO5-Z reconstructs each conserved variable independently (component-wise). At contact discontinuities where density jumps but pressure is continuous, the independently-reconstructed momentum and energy can produce inconsistent left/right states for the HLLD solver. The HLLD contact wave speed S_M depends on the pressure balance across the contact -- if the reconstructed states violate this balance, the solver generates spurious oscillations that WENO-Z's smoothness indicators cannot suppress because they operate per-variable.

This is specifically worse than PLM because PLM's slope limiter is more aggressive near discontinuities (minmod/MC kills the slope entirely when signs differ), while WENO5-Z preserves more of the high-order reconstruction. The extra accuracy near smooth regions becomes a liability near contacts where the component-wise approach introduces inconsistencies between variables.

**Detection**: Monitor the Brio-Wu shock tube compound fast-rarefaction/contact/slow-shock structure at N=256. If the contact discontinuity (rho jump at x~0.5-0.6) shows >10% overshoot/undershoot relative to the exact solution, this risk is active. Also check Sod contact: exact solution has a clean density plateau between contact and tail -- wiggles here indicate the problem.

**Mitigation**: The HLLD Metal kernel (mlx_kernels.py lines 520-531) already has a Lax-Friedrichs fallback triggered by NaN/Inf detection. Extend this to also trigger on `|p_star - p_L| / p_L > 10` or `rho_star < 0` (unphysical intermediate states). This targets the specific failure mode without adding the ~500 LOC characteristic decomposition that was already rejected.

Additionally, the entropy tracer helps here: S_rho is upwinded through the contact wave (line 484 in the HLLD kernel), providing a second pressure estimate that can detect when the total-energy-derived pressure at the contact is corrupted.

**Fallback**: If oscillations persist beyond what the LF fallback can handle, reduce to PLM reconstruction for the 3 cells adjacent to detected contacts (hybrid PLM/WENO5-Z). Detection criterion: `|rho[i+1] - rho[i-1]| / rho[i] > 0.5` flags a contact. This is ~30 LOC.

---

### N2. Float32 HLLD Intermediate States: Specific Overflow Points

**Likelihood**: High
**Impact**: High

**Root cause**: The HLLD solver computes intermediate quantities that can overflow float32 (max ~3.4e38) or underflow to zero at DPF-relevant conditions. The specific dangerous computations in `mlx_kernels.py`:

1. **Total pressure `pt_star`** (line 399): `pt_L + rho_L * (SL - vn_L) * (SM - vn_L)`. At the electrode boundary with `rho ~ 1e-3 kg/m^3`, `SL ~ -2.8e6 m/s` (Alfven speed at 100 T), `vn_L ~ 0`, `SM ~ 0`: the product `rho_L * SL^2 ~ 1e-3 * 7.8e12 ~ 7.8e9`. Combined with `pt_L ~ 3.98e9` (magnetic pressure at 100T): `pt_star ~ 1.2e10`. Safe in float32 but only 3 decades of headroom.

2. **Double-star denominators `D_L`, `D_R`** (lines 409-410): `rho * (S - vn) * (S - SM) - Bn^2`. At the pinch column with `I = 500 kA`, `r = 1 mm`: `Bn ~ 100 T` in HL units means `Bn^2 ~ 1e4`. Meanwhile `rho * (S - vn)^2 ~ 1e-3 * (2.8e6)^2 ~ 7.8e9`. These terms don't cancel dangerously here. BUT at the electrode where `Bn` is small and `rho` is near the floor (1e-12): `D ~ 1e-12 * (2.8e6)^2 - 0 ~ 7.8`, and `1/D` is fine. The dangerous case is intermediate `Bn` where `rho * (S-vn)*(S-SM) ~ Bn^2` -- then `D -> 0` and the tangential velocity/field computations (lines 415-425) produce inf. The code guards with `TINY = 1e-20` floor (line 412), but `1/1e-20 = 1e20` multiplied by `Bt ~ 100` gives `1e22` -- still within float32 range.

3. **Energy star states** (lines 429-433): `e_sL = ((SL-vn)*E_L - pt*vn + pt_star*SM + Bn*(vB_L - vB_sL)) / denom_L`. Each term can be ~1e10. The subtraction of terms of similar magnitude (e.g., `(SL-vn)*E_L ~ 2.8e6 * 1e10 = 2.8e16` and `pt*vn` could be similar) risks losing significant digits. With 7 decimal digits in float32, subtracting 1e16-scale quantities loses precision below 1e9 -- which is the pressure scale. This IS the catastrophic cancellation that dual-energy is designed to handle. But the HLLD flux itself (not just the final pressure recovery) uses this corrupted energy.

4. **ACTUAL overflow risk**: The `sqrt_rho` computation (lines 435-436) with `rho_sL` at the density floor (1e-12) gives `sqrt(1e-12) = 1e-6`. Then `SL_star = SM - |Bn| / 1e-6`. With `Bn = 100` in HL units: `SL_star = SM - 1e8`. This is fine arithmetically but means the Alfven star waves are extremely spread -- the double-star region occupies most of the Riemann fan, making the solution degenerate toward HLL quality anyway.

**Detection**: Add a per-timestep diagnostic that counts cells where `|flux[IEN]| > 1e30` or where the Lax-Friedrichs fallback was triggered. If >1% of interfaces trigger the fallback at 128x512, the HLLD solver is not providing its intended benefit over HLL.

**Mitigation**: Three layers already exist in the kernel:
1. `P_FLOOR = 1e-12` and `RHO_FLOOR = 1e-12` prevent zero-division (line 286-287)
2. `V_MAX = 1e6` velocity clamp prevents velocity runaway (line 316-318)
3. Lax-Friedrichs fallback on NaN/Inf (lines 520-531)

Add a fourth layer: before computing double-star states, check if `|D_L| < 1e-6 * |rho_L * (SL-vn_L)^2|` and if so, skip to single-star (HLL-like) flux. This degrades gracefully at the specific cells where HLLD intermediate states are unreliable, without penalizing the rest of the domain.

**Fallback**: If float32 HLLD proves fundamentally unreliable at DPF electrode conditions (>5% of interfaces hitting fallback), switch to HLL for the first 3 radial cells near the cathode and HLLD everywhere else. The electrode boundary is a region where the flow is simple (nearly 1D radial compression), so HLL suffices there.

---

### N3. Entropy Tracer at Shocks: Rankine-Hugoniot Violation Accumulation

**Likelihood**: Medium
**Impact**: Medium

**Root cause**: The entropy tracer S_rho is passively advected through the Riemann solver's contact wave. At shocks, the Rankine-Hugoniot conditions require entropy to INCREASE (2nd law). The advected S_rho does not capture this jump -- it is only correct in smooth flow. The resync mechanism (`entropy_resync` in mlx_primitives.py) resets S_rho from the total-energy-derived pressure at detected shocks. But shock detection is approximate (div_v + pressure gradient threshold), and the resync uses the very subtraction (`E - KE - ME`) that dual-energy exists to avoid.

The specific failure mode: at the converging cylindrical shock front (Phase 3), the shock is curved and moves radially inward. The compression ratio is ~4 (gamma = 5/3 limit). The entropy jump across the shock is `Delta_S = p_post / (rho_post^gamma) - p_pre / (rho_pre^gamma)`. If resync misses one cell at the shock front, the entropy deficit accumulates. After 100 timesteps, this manifests as pressure underestimate of ~5-10% in the post-shock region, which affects the compression dynamics.

The secondary risk: the resync criterion (mlx_primitives.py line 336) checks `reliable = (|p_S| / |E|) > sync_threshold`. At the shock front in low-beta plasma (beta ~ 0.01), `p_S / E` is small, and the criterion may block resync precisely where it's needed most. The `sync_threshold = 1e-2` default may be too aggressive.

**Detection**: Compare S_rho-derived pressure and E-derived pressure in post-shock cells over 100 timesteps. If the difference grows monotonically and exceeds 5%, resync is failing. Also run a cylindrical Sedov blast at beta=1 (where both pressure estimates are reliable) and compare -- if the entropy tracer drifts even at high beta, the advection itself has a bug.

**Mitigation**: Lower `sync_threshold` to 1e-4 (from 1e-2). At the shock front, beta may be 0.01 but the total-energy subtraction IS reliable there because shocks convert KE/ME to thermal energy -- the post-shock state has higher beta than the pre-shock state. The current threshold of 1e-2 is unnecessarily conservative; the criterion should check the POST-shock reliability, not the pre-shock.

Additionally, apply a one-sided resync: only reset S_rho from total energy when `p_E > p_S` (entropy should increase at shocks, never decrease). This prevents the resync from accidentally REDUCING entropy at expansion waves where p_E might be unreliable.

**Fallback**: If entropy drift exceeds 5% despite tuning, switch to the Popovas (2025) "residual entropy production" approach: at each RK stage, compute the entropy residual `R_S = d(S_rho)/dt + div(S_rho * v) - source` and add it as a correction. This is more expensive (~20% overhead) but provably consistent with the conservation law.

---

### N4. CFL with Dual-Energy: Pressure Recovery Changing Wave Speeds Mid-Stage

**Likelihood**: Low
**Impact**: Medium

**Root cause**: SSP-RK3 computes 3 stages. At each stage, the fast magnetosonic speed `c_f = sqrt(0.5*(a^2 + va^2 + sqrt(disc)))` depends on pressure via `a^2 = gamma*p/rho`. If the dual-energy switching criterion selects a different pressure source (entropy vs total energy) at different stages for the same cell, the effective wave speed can change discontinuously within a single RK step. This could violate the CFL condition computed at the beginning of the step.

Specifically: at the electrode boundary, Stage 1 might use entropy-derived pressure (p_S ~ 160 Pa, giving c_f ~ 5e3 m/s) because eta < eta1. After Stage 1 modifies the state, the Stage 2 entropy tracer may have been resynced (from entropy_resync), producing a different p_S that pushes eta above eta2, switching to total-energy pressure (which in float32 could be corrupted, giving a very different c_f). The CFL was computed using Stage 1's wave speed, but Stage 2 sees a different effective speed.

**Detection**: Log the CFL number `|v + c_f| * dt / dx` for each RK stage at 3 representative cells (axis, mid-domain, cathode). If any stage shows CFL > 1.5 (50% above the target CFL of 0.3/0.5), the mid-stage wave speed change is dangerous.

**Mitigation**: Compute CFL using the MAXIMUM fast magnetosonic speed across both pressure estimates: `c_f_max = max(c_f(p_E), c_f(p_S))`. This is conservative (smaller dt) but guarantees CFL safety regardless of which pressure the switching criterion selects. The cost is ~5% smaller timesteps, which is negligible.

**Fallback**: Not needed. The conservative CFL estimate handles this.

---

### N5. Cylindrical Axis Singularity: L'Hopital in Geometric Sources vs Reconstruction

**Likelihood**: Medium
**Impact**: High

**Root cause**: The cylindrical coordinate system has a 1/r singularity at r=0. The geometric source terms (hoop stress `p/r`, magnetic tension `B_theta^2 / (mu_0 * r)`) diverge as r -> 0. The spec (METAL_V2_SPEC.md section 7.2) places the inner cell boundary at `r_min = dr/2`, never at r=0. The Metal kernel for geometric sources (mlx_kernels.py, not yet shown but described in the spec) applies L'Hopital's rule: `p/r -> dp/dr` at the first active cell.

The risk is that the WENO5-Z reconstruction and the geometric source kernel use DIFFERENT treatments of the axis. WENO5-Z reconstruction (mlx_reconstruction.py) reconstructs interface values using the 5-point stencil. The innermost 2-3 cells are within the ghost region (reflecting BC). If the reconstructed interface value at the axis face implies a pressure gradient inconsistent with L'Hopital's rule in the source term, the two operators fight each other, producing oscillations at the axis.

Specific scenario: reflecting BC negates v_r and B_r in ghost cells. WENO5-Z sees a symmetric profile and reconstructs p at the axis face. The geometric source at cell 0 computes `dp/dr` via one-sided difference. If the ghost cell padding is off by half a cell (which is common in reflecting BCs -- should the mirror be about the face or the cell center?), the effective L'Hopital approximation is wrong.

**Detection**: Initialize a uniform-pressure, zero-velocity cylindrical state and evolve for 100 steps. Any non-zero velocity at the axis indicates a spurious force from source-reconstruction inconsistency. The velocity at r=0 should remain < 1e-10 * c_s.

**Mitigation**: Ensure the ghost cell kernel (mlx_kernels.py lines 96-103) mirrors about the cell FACE (r=0), not the cell center. The current implementation uses `mirror = -r_interior - 1`, which maps ghost index `ng-1-ig` to interior index `ig`. For ng=3: ghost index 2 maps to interior index 0 (correct face reflection), ghost index 1 maps to interior index 1, ghost index 0 maps to interior index 2. This is correct.

Additionally, the geometric source kernel must use `r = r_min + (i + 0.5) * dr` for cell centers, ensuring the first cell center is at `dr/2`, not `dr`. Verify this in the kernel implementation.

**Fallback**: If axis oscillations persist, switch the innermost 2 cells to PLM reconstruction (the weno5z_reconstruct function in mlx_reconstruction.py line 270 already falls back to PLM for n<6 cells, but the stencil extends INTO the ghost region, so this may not trigger). An explicit axis flag that forces PLM for cells i < 3 would cost ~5 LOC.

---

### N6. Energy Conservation with Dual-Energy: Blending Drift

**Likelihood**: Medium
**Impact**: Medium

**Root cause**: The dual-energy switching blends p_E and p_S with a cubic Hermite function (mlx_primitives.py line 195): `p = w * p_E + (1-w) * p_S`. The blended pressure is used for flux computation (via cons_to_prim inside HLLD). But the total energy equation evolves E conservatively through the Riemann solver, while the entropy tracer S_rho is advected separately. The two give DIFFERENT pressure estimates, and the blend is NOT a conserved quantity itself.

Over many timesteps, the blend introduces a systematic drift: cells that oscillate near the switching boundary (eta ~ eta1-eta2) get a slightly different pressure each way. This is the dual-energy "blending zone drift" documented in Bryan et al. (2014, Section 4.2). Enzo mitigates this by resetting the internal energy from total energy whenever eta > eta2 (outside the magnetically-dominated regime). We do the analogous thing with entropy resync at shocks (entropy_resync).

The specific DPF concern: during the radial compression phase, the plasma behind the sheath transitions from beta ~ 1 (rundown) through beta ~ 0.01 (compression) -- sweeping through the entire blending zone. Every cell in the post-shock region passes through the blending zone TWICE: once during compression, once during post-pinch expansion. Each passage introduces ~0.1% energy error. Over the full discharge (~1000 steps through the blending zone), this accumulates to ~1% total energy drift.

**Detection**: Track `E_total = integral(E * r * dr * dz)` and `E_circuit = integral(V*I*dt)` and `E_rad = integral(Q_rad*dV*dt)` every timestep. If `|E_total - E_circuit + E_rad| / E_circuit > 0.05` (5%), the drift is problematic. The DoD threshold is 10% (M4), so 5% provides margin.

**Mitigation**: After each full RK step (not each stage), reset the entropy tracer from total energy in cells where eta > 2*eta2 (deep in the high-beta regime where E - KE - ME is fully reliable). This prevents the entropy tracer from accumulating drift in the region where it's not needed. Cost: ~10 LOC in the timestepper, no performance impact.

**Fallback**: If drift exceeds 5%, narrow the blending zone: set eta1 = 1e-6, eta2 = 1e-3 (from current 1e-5/1e-2). This makes the transition sharper but reduces the number of cells in the blending zone. The risk is switching artifacts, which can be smoothed by applying the cubic Hermite over a wider stencil (3-cell wide instead of pointwise).

---

## 2. Framework Risks

### F1. MLX Lazy Evaluation: Memory Spikes for 128x512 Grids

**Likelihood**: Medium
**Impact**: Medium

**Root cause**: MLX uses lazy evaluation -- operations are recorded in a computation graph and only executed when results are needed (via `mx.eval()`, `.item()`, or `np.array()`). During one RK stage of the MHD solver, the graph accumulates: ghost padding, WENO5-Z reconstruction (many intermediate arrays), HLLD flux computation, flux divergence, geometric source terms, entropy resync, and floor enforcement. For a (10, 128, 512) state, each intermediate is ~2.5 MB. A single RK stage might create 50-100 intermediates, accumulating ~250 MB of deferred computation.

With 3 RK stages per timestep, the peak graph could reach ~750 MB before evaluation. On a 36 GB M3 Pro, this is manageable but wastes memory that could be used for larger grids.

The deeper risk: the Metal kernels (ghost pad, HLLD, geometric sources) are NOT lazy -- they execute immediately via Metal dispatch. But the MLX ops wrapping them (slicing for reconstruction, elementwise for source terms) ARE lazy. The interaction between eager Metal kernels and lazy MLX ops could cause synchronization stalls where `mx.eval()` is called implicitly at kernel boundaries.

**Detection**: Profile memory usage during one timestep with `mx.metal.get_peak_memory()` before and after. If peak memory exceeds 4x the state array size (4 * 10 * 128 * 512 * 4 bytes = 10 MB * 4 = 40 MB), the graph is accumulating too many intermediates. Current expectation: ~100-200 MB peak.

**Mitigation**: Insert explicit `mx.eval()` after each major computational phase:
1. After ghost padding (force evaluation before reconstruction reads padded array)
2. After flux computation (force evaluation before flux divergence)
3. After each RK stage (force evaluation before next stage reads updated state)

This limits the graph to one phase at a time (~50 MB peak). The cost is 3 GPU synchronization points per timestep (~100 microseconds total on M3 Pro), which is negligible compared to the ~10 ms compute time per step at 128x512.

**Fallback**: If memory still spikes, wrap the entire RK step in `mx.compile()` which forces MLX to optimize the graph and reuse memory for intermediates. The research doc (PHASE_B_RESEARCH.md line 75-79) notes that `mx.compile()` cannot fuse stencil operations, but it CAN optimize memory allocation for the elementwise chains between stencils.

---

### F2. mx.compile() with Custom Metal Kernels: Graph Tracing Issues

**Likelihood**: Medium
**Impact**: Low

**Root cause**: `mx.compile()` traces a function by recording operations as a graph. Custom Metal kernels created via `mx.fast.metal_kernel()` are opaque to the tracer -- the tracer sees them as black-box functions with known input/output shapes. This means:
1. The compiled graph cannot fuse operations across a kernel boundary
2. The compiled graph must evaluate all kernel inputs before the kernel call
3. If a kernel output is used in a subsequent elementwise chain, the chain STARTS a new fusion group

This is functionally correct but suboptimal: the compiler cannot reorder or fuse across kernel boundaries. The risk is that a `mx.compile()`-wrapped function that interleaves custom kernels and MLX ops produces a graph that thrashes between Metal kernel dispatch and elementwise fusion, with synchronization overhead at each boundary.

**Detection**: Benchmark the RK step with and without `mx.compile()`. If the compiled version is slower or equal, the graph tracing overhead exceeds the fusion benefit.

**Mitigation**: Apply `mx.compile()` ONLY to pure-MLX-ops sections (pressure recovery, entropy resync, floor enforcement, CFL computation). Do NOT compile the full RK step that includes custom kernels. This isolates the benefit (elementwise fusion) from the cost (kernel boundary synchronization).

**Fallback**: Skip `mx.compile()` entirely. The Metal kernels already handle the hotspots. The elementwise ops are ~10% of total compute time -- optimizing them further has diminishing returns.

---

### F3. Metal Kernel JIT: Compilation Time for Complex Kernels

**Likelihood**: Low
**Impact**: Low

**Root cause**: The HLLD kernel source is ~190 lines of MSL (lines 349-534 in mlx_kernels.py). Metal shader compilation scales superlinearly with shader complexity. First-time compilation of the HLLD kernel could take 1-5 seconds. MLX caches compiled kernels, so subsequent calls are fast (~microseconds). But if the kernel needs recompilation (e.g., grid size changes the dispatch parameters), the delay recurs.

**Detection**: Time the first invocation of each kernel. If HLLD first-call takes >5 seconds, it may impact interactive use (Gradio frontend).

**Mitigation**: Already handled by the kernel caching pattern (`_hlld_kernel_cache` at line 536). The `mx.fast.metal_kernel()` call compiles once; subsequent calls to the returned function just dispatch with new data. Ensure the caching is module-level (global) not per-instance, so solver re-instantiation doesn't trigger recompilation. The current implementation is correct (module-level cache).

**Fallback**: Pre-warm kernels during solver `__init__` by running a single-cell dummy computation. This moves the compilation cost to startup rather than first timestep.

---

### F4. MLX Memory Management: Garbage Collection Between RK Stages

**Likelihood**: Low
**Impact**: Medium

**Root cause**: MLX allocates GPU memory from a pool. Intermediate arrays from Stage 1 of SSP-RK3 should be freed before Stage 2 starts. Python's reference counting handles this IF no references are held. But common patterns like `U_old = U` (aliasing, not copying) can keep Stage 1 intermediates alive through Stage 2.

The SSP-RK3 formula requires `U_n` (original state) through all 3 stages:
```
U1 = Un + dt*L(Un)
U2 = 3/4*Un + 1/4*(U1 + dt*L(U1))
U3 = 1/3*Un + 2/3*(U2 + dt*L(U2))
```
So `Un` must survive all 3 stages. But `U1` only needs to survive until `U2` is computed. If the implementation holds all 3 stage results simultaneously, peak memory triples.

**Detection**: Track `mx.metal.get_active_memory()` at the start and end of each RK stage. Active memory should not exceed 3x the state array size (Un + current stage + L(current)).

**Mitigation**: Explicitly overwrite stage variables:
```python
U1 = Un + dt * L(Un)
mx.eval(U1)
U2 = 0.75 * Un + 0.25 * (U1 + dt * L(U1))
mx.eval(U2)
del U1  # explicitly release Stage 1
U_new = (1.0/3.0) * Un + (2.0/3.0) * (U2 + dt * L(U2))
mx.eval(U_new)
del U2  # explicitly release Stage 2
```

**Fallback**: Call `mx.metal.clear_cache()` between stages if memory pressure exceeds a threshold. This forces memory reclamation but may trigger reallocation.

---

### F5. Thread Safety: mx.eval() Blocking and Race Conditions

**Likelihood**: Low
**Impact**: Low

**Root cause**: `mx.eval()` is a synchronous barrier that forces all pending computations to complete before returning. In a single-threaded MHD solver, this is unambiguous. The risk arises if the Gradio frontend or circuit solver runs on a different Python thread that also calls MLX operations.

MLX uses a single Metal command queue per process. If two threads submit work concurrently, the operations are serialized on the GPU but may interleave in unexpected ways. For example, the circuit solver extracting `Lp` via `.item()` (which calls `mx.eval()` internally) while the MHD solver is building a computation graph could force premature evaluation of an incomplete graph.

**Detection**: This would manifest as non-deterministic results between runs, or as `BAD_ACCESS` crashes if arrays are modified during evaluation.

**Mitigation**: Run the MHD solver and circuit coupling on the same thread. The circuit solver already runs on CPU (float64) with scalar extraction via `.item()`. Ensure `.item()` calls happen between RK stages (after `mx.eval()`), not during graph construction.

**Fallback**: Add a mutex around the MLX computation path. Cost: negligible (mutex acquire/release is ~100 ns vs ~10 ms computation).

---

## 3. Integration Risks

### I1. engine.py Backend Cascade: "mlx" vs "metal" Naming Collision

**Likelihood**: High
**Impact**: Medium

**Root cause**: The current engine.py uses `backend="metal"` to select the PyTorch-MPS-based MetalMHDSolver. Phase B adds an MLX-based solver. If both use `backend="metal"`, the engine must disambiguate. If the new backend uses `backend="mlx"`, the naming is clear but breaks the existing backend priority cascade: `athenak > athena > metal > python`.

The spec (METAL_V2_SPEC.md section 2) calls the new solver "Phase B MLX solver" but doesn't specify the backend string. The engine integration work unit (WU-3.3 in PHASE_B_RESEARCH.md) adds `backend="mlx"` to the selection cascade. But the existing `device.py` has `has_metal()` (PyTorch MPS) and the new `mlx_device.py` has `has_mlx()` -- these test different things.

The collision scenario: a user sets `backend="metal"` expecting the new MLX solver (because it runs on Metal GPU) but gets the old PyTorch solver. Or the auto-resolution picks "metal" (PyTorch) when "mlx" (MLX) would be better.

**Detection**: This is a design-time risk, not a runtime risk. It becomes a bug when users report confusion about which solver they're running.

**Mitigation**: Use `backend="mlx"` for the new solver. Update the auto-resolution priority to: `athenak > athena > mlx > metal > python`. The old `metal` backend becomes deprecated (kept for backward compatibility but not auto-selected when MLX is available). Add a log message when the solver is selected: `"Using MLX MHD solver (Metal GPU)"` vs `"Using PyTorch Metal MHD solver (MPS)"`.

**Fallback**: If naming causes ongoing confusion, rename `metal` to `pytorch_mps` in the next major version. This is a breaking change but clarifies the actual framework being used.

---

### I2. Circuit Coupling: Scalar Extraction from MLX Arrays

**Likelihood**: Low
**Impact**: High

**Root cause**: The circuit solver (rlc_solver.py) runs in CPU float64. It needs two scalars per timestep: plasma inductance `Lp` and plasma resistance `R_plasma`. These are computed as volume integrals over the MHD state (density-weighted radius, Spitzer resistivity integral).

The extraction path: `mx.array` -> `mx.sum()` (reduction) -> `mx.eval()` -> `.item()` -> Python float -> float64 promotion. The spec estimates ~20-50 microseconds per extraction, which is acceptable. The precision risk: the volume integral is computed in float32 on GPU. For Lp, the integral is `integral(r * rho * r * dr * dz) / integral(rho * r * dr * dz)`. With rho values spanning 6 orders of magnitude (1e-3 to 1e3 kg/m^3) and r from 0.005 to 0.16 m, the sum involves adding terms of very different magnitudes. Float32 summation of 128*512 = 65536 terms with 6 decades of dynamic range can lose 2-3 digits.

**Detection**: Compare Lp computed in float32 (MLX sum) vs float64 (NumPy sum on the same data extracted via zero-copy). If they differ by >0.1%, the float32 summation is insufficient.

**Mitigation**: Use Kahan compensated summation for the volume integrals. MLX doesn't have a built-in Kahan sum, but it can be implemented as:
```python
def kahan_sum_2d(arr):
    # Sum along axis 0 first (radial), then axis 1 (axial)
    # This limits the number of terms per reduction
    return mx.sum(mx.sum(arr, axis=0), axis=0)
```
Alternatively, extract the full 2D arrays to NumPy (zero-copy) and sum in float64:
```python
rho_np = np.array(U[IDN])  # zero-copy
r_np = np.array(r_centers)  # zero-copy
Lp = compute_Lp_f64(rho_np, r_np)  # float64 sum
```
This adds ~10 microseconds for the zero-copy transfer (which is actually just a pointer cast on unified memory) plus ~50 microseconds for the float64 sum. Total: ~100 microseconds per timestep, negligible.

**Fallback**: Always use the float64 NumPy path for circuit coupling scalars. The zero-copy transfer makes this nearly free on Apple Silicon.

---

### I3. State Dict Conversion Overhead

**Likelihood**: Medium
**Impact**: Low

**Root cause**: The DPF engine uses `dict[str, np.ndarray]` state dicts. The MLX solver uses `(10, nr, nz)` packed `mx.array`. Every call to `solver.step()` requires:
1. Pack: dict -> mx.array (primitive -> conserved conversion, 10 array allocations + 1 stack)
2. Unpack: mx.array -> dict (conserved -> primitive conversion, 8 dict entries)

At 128x512 = 65536 cells, each variable is 256 KB. The full state is 10 * 256 KB = 2.5 MB. On Apple Silicon unified memory, `mx.array(np_arr)` for C-contiguous float32 is zero-copy (pointer sharing). The actual computation cost is the primitive-to-conserved conversion (5 multiplications + 1 square root per cell).

The timing concern: the engine calls `solver.step()` potentially thousands of times per discharge. If pack/unpack takes 1 ms each, that's 2 ms per step overhead. At ~10 ms per step compute time, this is 20% overhead.

**Detection**: Time the `from_state_dict()` and `to_state_dict()` calls separately from the solver step. If combined overhead exceeds 10% of step time, optimization is needed.

**Mitigation**: Keep the packed `mx.array` as the solver's internal state between steps. Only convert to/from state dict at the engine interface boundary (start and end of the simulation, and at diagnostic output intervals). During the inner loop, pass the packed array directly. This requires the engine to support "opaque state" passthrough -- the solver stores its state internally and the engine only requests a state dict when it needs diagnostics.

**Fallback**: If the engine interface cannot support opaque state, cache the packed array as `solver._U` and only repack when the state dict has actually changed (e.g., circuit coupling modified a field).

---

### I4. Cross-Backend Parity: Acceptable Tolerance Between PyTorch and MLX

**Likelihood**: Medium
**Impact**: Medium

**Root cause**: The PyTorch Metal solver (Phase A/M) and the MLX solver (Phase B) implement the same algorithms (WENO5-Z + HLLD + SSP-RK3) in different frameworks. Floating-point arithmetic is not associative: `(a + b) + c != a + (b + c)` in float32. Different operation ordering between PyTorch and MLX will produce bitwise-different results even with identical inputs. Additionally, the Metal HLLD kernel (MLX) is a custom MSL shader, while the PyTorch HLLD is pure PyTorch tensor ops -- the operation decomposition is fundamentally different.

The question is: how much difference is acceptable? The DoD (METAL_V2_DOD.md S5) specifies L1(rho) < 15% on Sod. This is very loose. The PHASE_B_RESEARCH.md R10 notes that this is a "should-have." The real concern is PF-1000: if fc/fm parameters were calibrated on the PyTorch solver and the MLX solver produces a different I(t) waveform, the calibration is invalid.

**Detection**: Run the Sod shock tube on both solvers at N=256 for 100 steps. Compute L1(rho) between them. Also run PF-1000 for 100 steps on both and compare I(t) peak and dip timing.

**Mitigation**: The acceptable tolerance should be:
- Sod L1(rho) < 1% (same algorithm, same precision, only op ordering differs)
- PF-1000 I_peak < 2% difference (same parameters)
- If L1 > 1% on Sod, there is an algorithmic difference (bug), not just floating-point noise.

Achieve this by ensuring both solvers use identical:
- WENO5-Z coefficients (FD point-value: 3/8, -10/8, 15/8 etc.)
- HLLD wave speed estimates (Davis bounds)
- SSP-RK3 coefficients (1, 3/4, 1/4, 1/3, 2/3)
- CFL number (same default)
- Floor values (RHO_FLOOR, P_FLOOR)

**Fallback**: If PF-1000 I_peak differs by >5% between solvers with identical parameters, recalibrate fc/fm for the MLX solver independently. Document the calibration as solver-specific. This is undesirable but physically valid -- the numerical diffusion of two different implementations can legitimately require different effective mass/current factors.

---

## 4. Performance Risks

### P1. Memory Bandwidth Ceiling: 150 GB/s Shared

**Likelihood**: High
**Impact**: Medium

**Root cause**: The M3 Pro has ~150 GB/s memory bandwidth shared between CPU and GPU. The MHD solver is bandwidth-bound (low arithmetic intensity -- most operations are 1-2 FLOP per byte loaded). At 128x512 with 10 variables, the state is 2.5 MB. Each RK stage reads the state ~5 times (ghost pad, reconstruct, HLLD, flux divergence, source terms) and writes ~3 times (new state, intermediate fluxes, entropy resync). That is ~8 * 2.5 MB = 20 MB of traffic per stage, or 60 MB per timestep.

At 150 GB/s, 60 MB takes 0.4 microseconds -- far below the observed ~10 ms per step. The actual bottleneck is kernel launch overhead and synchronization, not bandwidth. But at 256x1024 (4x the grid), traffic is 240 MB per step, still only ~1.6 microseconds. Bandwidth is NOT the bottleneck at these grid sizes.

The real performance risk is that the GPU compute utilization is low because the kernels are too small to saturate the GPU's execution units. At 128x512 = 65K threads, the M3 Pro's GPU (10 cores, ~1280 execution units) has only ~50 threads per execution unit -- barely enough to hide memory latency.

**Detection**: Use `mx.metal.get_peak_memory()` and wall-clock timing to compute effective bandwidth. If effective bandwidth is <10% of peak (i.e., <15 GB/s), the GPU is underutilized.

**Mitigation**: Fuse operations to reduce kernel launch count. Currently: ghost pad (1 kernel) + HLLD (1 kernel per dimension = 2) + geometric sources (1 kernel) = 4 custom kernel launches per stage, 12 per timestep. Each launch has ~5-10 microsecond overhead on Metal, totaling ~120 microseconds -- small relative to ~10 ms compute.

The real mitigation for small grids is the backend auto-selection: use MLX only when grid > 64x256. Below that, Athena++ or Python engine is faster.

**Fallback**: Accept that MLX won't beat Athena++ at 128x512. The DoD S9 ("faster than Athena++") is a should-have, not a must-have. The value of MLX is zero-copy circuit coupling and Python ecosystem integration, not raw throughput.

---

### P2. Kernel Launch Overhead for Many Small Kernels per RK Stage

**Likelihood**: Medium
**Impact**: Medium

**Root cause**: Each custom Metal kernel launch involves: argument encoding, command buffer submission, GPU scheduling, execution, completion signaling. On Apple Silicon, this pipeline takes ~5-15 microseconds per launch. The MLX solver per RK stage dispatches:

- 1 ghost pad kernel (both r and z directions embedded)
- 2 HLLD kernels (radial + axial sweeps)
- 1 geometric source kernel
- Plus ~10-20 MLX elementwise operations (reconstruction, pressure recovery, entropy sync, CFL)

The MLX elementwise ops may be fused by lazy evaluation into 2-3 Metal dispatches. Total: ~7-10 Metal dispatches per stage, ~25-30 per timestep. At 10 microseconds each: ~300 microseconds overhead per timestep.

At 128x512, each kernel processes 65K cells. The HLLD kernel (~200 FLOPs per cell) takes ~65K * 200 / (5 TFLOPS) ~ 2.6 microseconds of compute. The launch overhead (10 us) dominates the compute (2.6 us) by 4x.

**Detection**: Profile with `METAL_DEVICE_WRAPPER_TYPE=1` environment variable (Apple Metal frame capture). Count total kernel dispatches and measure dispatch-to-completion latency.

**Mitigation**: Three strategies, ordered by implementation effort:

1. **Batch the directional sweeps** (low effort): Instead of separate r-sweep and z-sweep HLLD kernels, fuse into a single kernel that computes both fluxes. This halves the HLLD launch count. Requires a more complex kernel but saves ~10 us/stage.

2. **Use mx.compile() for elementwise chains** (medium effort): Wrapping the reconstruction + pressure recovery + entropy sync in `mx.compile()` lets MLX fuse them into 1-2 Metal dispatches instead of 10-20. Already planned in Sprint 2.

3. **Increase grid size** (no effort): At 256x1024, the HLLD kernel takes ~10 microseconds of compute, matching the launch overhead. At 512x2048, compute dominates. The solver naturally becomes efficient at larger grids.

**Fallback**: Accept the overhead at 128x512. The total wall time (~10-15 ms per step) is still acceptable for interactive simulation with Gradio frontend.

---

### P3. Thomas Solver CPU Fallback: Serialization Point

**Likelihood**: High
**Impact**: Low

**Root cause**: The implicit resistive diffusion step uses a Thomas (tridiagonal) solver. Thomas is inherently sequential along each radial line: the forward sweep depends on the previous cell's result. This cannot be parallelized on GPU. The solver must run on CPU.

For a 128x512 grid: 512 independent tridiagonal systems (one per z-column), each of size 128. Each Thomas solve is ~5 * 128 = 640 FLOPs. Total: 512 * 640 = 328K FLOPs. On a single M3 Pro core at ~50 GFLOPS: ~6.5 microseconds. With memory access overhead, realistically ~50-100 microseconds.

The data transfer: extract B-field components from MLX to NumPy (zero-copy on unified memory), run Thomas in NumPy/Numba, write back (zero-copy). The extraction requires `mx.eval()` to materialize the lazy arrays, adding a synchronization point.

**Detection**: Time the Thomas solver step separately. If it exceeds 1 ms, it's a meaningful fraction of the ~10 ms total step time.

**Mitigation**: The 512 independent systems can be parallelized across CPU cores using Numba `@njit(parallel=True)` with `prange`. On 8 P-cores of M3 Pro, this reduces to ~50/8 ~ 6 microseconds + thread launch overhead (~50 us). Total: ~100 microseconds, negligible.

Alternatively, the Thomas solver can be batched as a cyclic reduction algorithm on GPU. Cyclic reduction parallelizes the tridiagonal solve into O(log N) steps. For N=128, that is 7 steps, each fully parallel across 512 systems. This is implementable as a custom Metal kernel (~100 LOC MSL) but the effort is not justified unless the Thomas solver becomes a bottleneck.

**Fallback**: Keep Thomas on CPU. Even at 500 microseconds, it is <5% of total step time.

---

### P4. MHD Sub-Stepping: Correct CFL with MLX

**Likelihood**: Medium
**Impact**: High

**Root cause**: The CFL timestep `dt = CFL * dx / (|v| + c_f)` must be computed correctly. The fast magnetosonic speed `c_f` depends on pressure, which in dual-energy mode depends on the switching criterion. The risk is a specific MLX implementation bug: if `fast_magnetosonic()` (mlx_primitives.py lines 202-260) receives the wrong pressure (e.g., p_S when it should receive the blended p, or p_E from the corrupted subtraction), the CFL estimate is wrong.

The lesson from Phase A (CLAUDE.md lesson): the Metal v1 solver had a CFL bug where `dt` was computed from state BEFORE ghost cells were applied, missing the extreme B-field at the electrode boundary. The electrode has the fastest wave speed in the entire domain (c_f ~ 2.8e6 m/s at 100 T, 1 mm from anode). Missing this cell makes dt too large, causing the solver to blow up after ~10 steps.

**Detection**: Print the CFL-limiting cell location and its wave speed for the first 10 timesteps. It should always be near the electrode boundary (outer radial cells, near anode). If the CFL-limiting cell is in the interior, something is wrong.

**Mitigation**: The `fast_magnetosonic()` function MUST receive the dual-energy blended pressure, not p_E. Verify by tracing the call chain from the CFL computation backward to pressure recovery.

Also: compute CFL AFTER ghost padding, so the electrode ghost cells (with prescribed B_theta = mu0*I/(2*pi*r)) are included in the maximum wave speed search. This is critical -- the ghost cell B_theta at r = cathode radius with I = 1 MA gives B ~ 12.5 T, which translates to c_f ~ 3.5e5 m/s at typical cathode density. Missing this underpredicts the CFL constraint.

**Fallback**: Apply a safety factor: `dt = 0.5 * CFL * dx / max_wave_speed` instead of the standard `CFL * dx / max_wave_speed`. The extra factor of 0.5 provides margin for wave speed underestimation at the cost of 2x more timesteps.

---

## 5. Validation Risks

### V1. PF-1000 Calibration: fc/fm Parameters Tuned for PyTorch Solver

**Likelihood**: High
**Impact**: High

**Root cause**: The Lee model mass-swept fraction (fc) and current factor (fm) are effective parameters that absorb numerical diffusion, grid resolution effects, and model approximation errors into two scalar coefficients. The current calibrated values (fc = 0.70, fm = 0.20 per Akel 2021) were validated against the PyTorch Metal solver at specific grid resolutions (16x32 to 64x256).

The MLX solver has different numerical diffusion because:
1. The HLLD implementation is a custom Metal kernel (MSL) rather than PyTorch tensor ops -- different operation ordering affects the amount of numerical diffusion at contacts and shocks
2. WENO5-Z is implemented as pure MLX vectorized ops rather than PyTorch -- different slicing patterns may affect the effective reconstruction at boundaries
3. The dual-energy switching introduces a pressure difference that PyTorch's simple `(gamma-1)*(E-KE-ME)` does not have

These differences change the effective snowplow dynamics: more diffusion -> slower compression -> lower peak current -> need to increase fm to compensate. Changing fm affects the current dip timing and neutron yield.

The historical precedent (from MEMORY.md): "Compensating errors: parameters calibrated with a bug compensate for it. Fixing the bug without recalibrating makes results WORSE. (Back-EMF fix: I_peak jumped 47% because fc/fm were tuned with double-counting.)"

**Detection**: Run PF-1000 at 16 kV / 1.2 Torr with Akel parameters (fc=0.70, fm=0.20) on the MLX solver. Compare I_peak against the PyTorch result (should be ~1.2 MA). If I_peak differs by >10%, recalibration is needed.

**Mitigation**: Plan for recalibration as Sprint 4 WU-4.1 work. Budget 1-2 days for fc/fm optimization using the Akel 24-shot dataset. Use the same optimization approach: sweep fc in [0.5, 0.9], fm in [0.1, 0.35], minimize NRMSE against published I(t).

Key insight: the dual-energy formulation should IMPROVE physics accuracy (no pressure corruption), which means the MLX solver needs LESS compensating from fc/fm. Expect fm to increase slightly (less numerical diffusion in the sheath -> less mass swept -> need higher fm to match experimental I_peak).

**Fallback**: If recalibration produces physically unreasonable parameters (fm > 0.5 or fm < 0.05), the solver has a physics bug that fc/fm cannot compensate. Escalate to dpf-mhd-physicist for circuit coupling review.

---

### V2. Back-EMF Double-Counting: Interaction with MLX Solver

**Likelihood**: Medium
**Impact**: High

**Root cause**: The back-EMF double-counting bug (rlc_solver.py:52 adds dLp_dt to R_star, AND coupler.py:194 computes back_emf = I * dLp_dt) is documented in METAL_V2_SPEC.md section 5.2 and marked as a known v1 bug. The DoD (METAL_V2_DOD.md section 4.2) explicitly calls for fixing this.

The risk for the MLX solver: if this bug is fixed for the MLX solver but NOT for the PyTorch solver, cross-backend comparisons become meaningless (they're comparing different circuit models). If it's fixed for both, the calibrated fc/fm parameters are invalidated (see V1 above).

The specific interaction with dual-energy: the back-EMF `V_back = -I * dLp/dt` depends on dLp/dt, which depends on the sheath position, which depends on the density field, which depends on pressure recovery. If the MLX solver's dual-energy formulation gives a different density profile at the sheath front (because pressure is more accurate at the electrode), dLp/dt changes, and the back-EMF changes, and the circuit current changes. This is a nonlinear coupling: fixing float32 pressure corruption changes the snowplow dynamics, which changes the circuit feedback.

**Detection**: Compare the back-EMF voltage time series between PyTorch (with bug) and MLX (without bug). If they differ by >50%, the fix has a large dynamical effect that will propagate through the I(t) waveform.

**Mitigation**: Fix the back-EMF bug in BOTH solvers simultaneously (as part of Sprint 3 engine integration). This ensures cross-backend comparisons are meaningful. Then recalibrate fc/fm for both solvers against the Akel dataset.

2026-05-11 update: the formulary/local-KR audit chose the other equivalent
ownership path. `RLCSolver` keeps `dLp_dt` in `R_star`, and `CircuitCoupler`
returns `back_emf=0.0` for the inductive term so `I*dLp/dt` is not counted
twice. A future distinct motional-EMF model must use the separate `back_emf`
channel.

**Fallback**: If fixing the bug breaks both solvers' I(t) waveforms beyond recalibration, investigate whether the "double-counting" is actually compensating for a missing term in the circuit model (e.g., the motional back-EMF from radial compression velocity).

---

### V3. High-Resolution Grid (128x512): Wall Time Feasibility

**Likelihood**: Medium
**Impact**: Medium

**Root cause**: The PF-1000 discharge covers ~12 microseconds. At the pinch phase (Phase 4), the CFL constraint gives `dt ~ CFL * dr / c_f ~ 0.3 * 9e-4 / 2.8e6 ~ 1e-10 s`. Total steps to complete the discharge: `12e-6 / 1e-10 = 120,000 steps`. At 128x512, each step takes ~10-15 ms on M3 Pro (estimated). Total wall time: 120,000 * 12.5 ms = 1,500 seconds = 25 minutes.

This is feasible for validation but too slow for interactive parameter sweeps. The Phase A PyTorch solver at 16x32 takes ~2 minutes for a full discharge. The MLX solver at 128x512 will be ~12x slower per step (larger grid) but the same number of steps (CFL scales with dx, and 128x512 has smaller dx).

Actually, refining from 16x32 to 128x512 means dx shrinks by 8x in r and 16x in z. CFL dt shrinks by 8x (radial direction dominates). So 120,000 / 8 * 8 = same number of steps BUT each step at 128x512 has 256x more cells. If compute scales linearly with cells: 2 minutes * 256 = 512 minutes = 8.5 hours. This is too long for a single validation run.

The saving grace: the pinch phase (with the smallest dt) lasts only ~200 ns out of 12 us. The rundown phase has dt ~ 1e-8 s (much larger dx at the sheath position). So the step count is dominated by the pinch phase: ~200 ns / 1e-10 s = 2,000 steps at the minimum dt, plus ~10,000 steps for rundown at larger dt. Total: ~12,000 steps. At 15 ms per step: 180 seconds = 3 minutes.

**Detection**: Run the first 100 steps and extrapolate based on the adaptive dt profile. If estimated total wall time exceeds 30 minutes, the grid is too fine for routine validation.

**Mitigation**: Use adaptive grid sizing:
1. Rundown phase (0-6 us): 64x256 grid (sufficient, no fine structure)
2. Compression phase (6-7 us): 128x512 grid (need to resolve the converging shock)
3. Pinch phase (7-7.5 us): 128x512 grid (need fine radial resolution at r ~ 1 mm)
4. Post-pinch (7.5-12 us): 64x256 grid (expanding plasma, coarsening is fine)

This is a form of manual AMR and reduces the number of fine-grid steps by ~60%. Implementation: the engine reduces grid resolution between phases by coarsening the state arrays (conservative restriction). This is ~50 LOC.

**Fallback**: Run full-discharge validation at 64x256 (acceptable resolution, ~5 minute wall time) and reserve 128x512 for convergence studies that run only the pinch phase (~30 seconds for 2,000 steps).

---

## Risk Summary Matrix

| ID | Risk | Likelihood | Impact | Priority |
|----|------|-----------|--------|----------|
| N2 | Float32 HLLD intermediate overflow | High | High | CRITICAL |
| V1 | fc/fm recalibration needed | High | High | CRITICAL |
| N1 | WENO5-Z + HLLD contact oscillation | Medium | High | HIGH |
| N5 | Axis singularity source/recon mismatch | Medium | High | HIGH |
| V2 | Back-EMF double-count interaction | Medium | High | HIGH |
| P4 | CFL computation with dual-energy | Medium | High | HIGH |
| I1 | Backend naming collision | High | Medium | HIGH |
| N3 | Entropy Rankine-Hugoniot accumulation | Medium | Medium | MEDIUM |
| N6 | Dual-energy blending drift | Medium | Medium | MEDIUM |
| F1 | MLX lazy eval memory spikes | Medium | Medium | MEDIUM |
| I4 | Cross-backend parity tolerance | Medium | Medium | MEDIUM |
| V3 | 128x512 wall time feasibility | Medium | Medium | MEDIUM |
| P1 | Memory bandwidth ceiling | High | Medium | MEDIUM |
| P2 | Kernel launch overhead | Medium | Medium | MEDIUM |
| I2 | Circuit scalar extraction precision | Low | High | MEDIUM |
| N4 | CFL mid-stage wave speed change | Low | Medium | LOW |
| F4 | GC between RK stages | Low | Medium | LOW |
| P3 | Thomas solver serialization | High | Low | LOW |
| I3 | State dict conversion overhead | Medium | Low | LOW |
| F2 | mx.compile() + custom kernels | Medium | Low | LOW |
| F3 | Metal kernel JIT compile time | Low | Low | LOW |
| F5 | Thread safety / mx.eval() races | Low | Low | LOW |

---

## Sprint-Specific Risk Mapping

### Sprint 2 (Time Integration + Source Terms): 6 active risks
- **N1** (WENO5-Z+HLLD contacts): First full timestep exposes this
- **N3** (entropy resync at shocks): First shock test exposes this
- **N4** (CFL with dual-energy): First CFL computation exposes this
- **N5** (axis singularity): First cylindrical test exposes this
- **N6** (blending drift): First multi-step test exposes this
- **F1** (lazy eval memory): First full RK step exposes this

### Sprint 3 (Solver Assembly + Engine Integration): 5 active risks
- **I1** (backend naming): Engine integration work unit
- **I2** (circuit coupling precision): First circuit-coupled test
- **I3** (state dict overhead): First end-to-end step measurement
- **I4** (cross-backend parity): First cross-backend comparison
- **V2** (back-EMF fix): Circuit coupling work unit

### Sprint 4 (DPF Validation + Performance): 5 active risks
- **V1** (fc/fm recalibration): PF-1000 validation
- **V3** (128x512 wall time): Production grid benchmark
- **P1** (bandwidth ceiling): Performance measurement
- **P2** (kernel launch overhead): Performance measurement
- **P4** (CFL correctness): Full discharge stability

---

## Recommended Pre-Sprint Actions

1. **Before Sprint 2**: Run the WENO5-Z + HLLD pipeline (built in Sprint 1) on Sod and Brio-Wu shock tubes for 100 steps. Measure L1(rho) and check for contact oscillations. This exercises risks N1 and N2 before building the time integrator on top.

2. **Before Sprint 3**: Run a uniform-pressure cylindrical state for 100 steps with the Sprint 2 timestepper. Verify zero spurious velocity at axis (risk N5) and zero energy drift (risk N6). This proves the source terms + reconstruction + time integration are consistent before adding engine complexity.

3. **Before Sprint 4**: Fix the back-EMF double-counting bug in BOTH solvers (risk V2) and recalibrate fc/fm on the PyTorch solver first. This establishes a clean baseline for the MLX solver validation.

4. **Day 1 of Sprint 4**: Run PF-1000 at 16x32 on both solvers with identical parameters. If I_peak differs by >5%, recalibrate before attempting 128x512 runs. Do not waste 30-minute runs on a miscalibrated solver.
