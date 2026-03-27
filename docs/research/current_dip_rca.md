# Root Cause Analysis: test_current_dip_present Failure

**Test**: `tests/test_validation_consolidated.py::TestMetalEngineVsExperiment::test_current_dip_present`
**Error**: `Current dip 0.0% too small (expected > 5%) in 23935 post-peak samples`
**Status**: Pre-existing failure on clean main and Sprint S-1. Physics issue, not a regression.

## 1. What the Test Checks

The test runs a full Metal engine simulation of PF-1000 (12 us, 16x1x32 grid, dx=1cm, HLL+PLM, float32) and measures the current dip after peak as a pinch signature.

```python
# Test configuration (test_validation_consolidated.py:4422-4452)
grid_shape = [16, 1, 32]
dx = 1e-2          # 1 cm
sim_time = 12e-6   # 12 us
backend = "metal"
fc = 0.816, fm = 0.142
radiation/collision disabled
handoff_mode = "lee_only" (default)
```

The dip is computed as:
```python
peak_idx = _find_first_peak(abs_I)
post_peak = abs_I[peak_idx : peak_idx + 3us_window]
dip = (abs_I[peak_idx] - min(post_peak)) / abs_I[peak_idx]
assert dip > 0.05  # 5% minimum
```

## 2. Root Cause: CFL Catastrophic Collapse at Radial Phase Onset

**The simulation freezes at t=6.756 us when the MHD timestep collapses to zero.**

### Evidence (diagnostic run, float64 CPU for traceability)

| Step | Time (us) | dt (s)   | Phase   | |B|_max (T) | rho_min     | cf_max (m/s) |
|------|-----------|----------|---------|------------|-------------|--------------|
| 1137 | 6.7538    | 2.68e-9  | radial  | 2.73       | 9.67e-8     | 6.98e6       |
| 1138 | 6.7564    | 1.17e-11 | radial  | **1,055**  | **5.18e-9** | **1.11e10**  |
| 1139 | 6.7565    | 2.69e-12 | radial  | 4,604      | 5.62e-8     | 1.17e10      |
| 1140 | 6.7565    | 2.68e-13 | radial  | 46,120     | 7.38e-8     | 1.51e11      |
| 1141 | 6.7565    | 1.46e-14 | radial  | 846,900    | 7.51e-8     | 2.11e12      |
| 1142 | 6.7565    | 1.00e-15 | radial  | 1.23e7     | 7.52e-8     | 4.01e13      |
| 1148 | 6.7565    | 0        | radial  | >1e9       | -           | -            |

Between steps 1137 and 1138:
- |B|_max jumps from 2.73 T to 1,055 T (386x in one step)
- CFL timestep drops from 2.68 ns to 11.7 ps (229x collapse)
- Subsequent steps: B grows exponentially, dt collapses to 0

### Causal Chain

```
1. Snowplow enters radial phase at t=6.534 us (step 1095)
2. _apply_electrode_bc applies snowplow zipper BC:
   - B_theta = 0 for z > z_sheath (axial zipper)
   - B_theta = 0 for r < r_shock (radial zipper, line 615)
3. This creates a sharp B_theta discontinuity at r = r_shock
4. On the coarse 16x1x32 grid (dr = 1 cm), the discontinuity
   spans only 1-2 cells
5. The MHD solver's flux computation produces extreme gradients
   at the discontinuity boundary
6. Ghost-cell electrode BC injects B_theta ~ mu0*I/(2*pi*r)
   into outer ghost cells, amplifying the discontinuity
7. Cells near the discontinuity develop extreme B values
   while maintaining low density (near floor)
8. va = B/sqrt(mu0*rho) -> infinity
9. CFL dt -> 0, simulation freezes
10. Snowplow stuck in "radial" phase (never reaches pinch)
11. No pinch -> no dL/dt -> no current dip
```

### Why dt_circuit Does Not Help

dt_circuit = 0.1 * min(L/R, sqrt(LC)) ~ 0.67 us (for PF-1000 parameters).
dt_fluid collapses to 0, so dt = min(0, 0.67us) = 0.

### Why Vacuum Masking Does Not Help

The Metal solver masks cells with `rho < 1e-4 * rho_max` from CFL. At step 1138:
- rho_max = 4.58e-3, threshold = 4.58e-7
- The extreme-B cell has rho = 5.18e-9 < threshold (masked)
- But dt_fluid is still 1.17e-11, meaning a NON-vacuum cell also has extreme signal speed
- The B-field amplification contaminates neighboring cells above the vacuum threshold

## 3. Why the Standalone Circuit+Snowplow Works

The standalone RLC+Snowplow path (no MHD grid) produces a 76% current dip because:
- The snowplow's ODE solver advances through radial -> reflected -> pinch
- No MHD CFL constraint
- dL/dt from post-pinch expansion feeds directly into the circuit ODE
- The natural LC sinusoidal decay provides the primary dip

## 4. Where the Dip Signal Is Expected to Come From

In a real DPF or the Lee model:

```
V_cap = L_total * dI/dt + I * dL/dt + I * R_total
```

The current dip comes from two sources:
1. **dL/dt during radial compression**: L_plasma increases as the sheath compresses,
   creating a back-EMF (I * dL/dt) that opposes current flow
2. **Post-pinch expansion dL/dt**: After pinch, the column disrupts and expands,
   with dL/dt modeled by the Goyon expansion model in `snowplow.py:505-541`

In the failing test:
- Source 1 is never activated because the MHD CFL freezes before radial compression completes
- Source 2 is never activated because pinch is never reached (snowplow stuck in "radial")

## 5. Code Path Trace

```
test_current_dip_present (line 4601)
  -> _get_metal_result() (line 4478)
    -> _run_metal_pf1000() (line 4455)
      -> SimulationEngine(config).step() x 25000
        -> _compute_dt() (core.py:522)
          -> fluid._compute_dt(state) = MetalMHDSolver.compute_dt()
            -> _compute_dt_cfl() (metal_solver.py:470)  [returns ~0]
          -> dt = min(dt_fluid, dt_circuit) -> ~0
        -> _step_circuit_subcycle(dt=~0, ...) (circuit_coupling.py:28)
          -> snowplow.step(dt_sub=~0, current) [dt too small to advance]
          -> circuit.step(coupling, back_emf, dt_sub=~0) [current unchanged]
        -> _step_fluid_advance(dt=~0, ...) (core.py:927)
          -> fluid.step(state, dt_mhd=~0, ...) [state unchanged]
```

Result: 24000 steps at dt=0, time frozen at 6.756 us, current flat at 1.8825 MA.

## 6. Recommended Fixes

### Fix A: Minimum dt floor (immediate, low-risk)

Add a minimum dt floor in `_compute_dt()` to prevent CFL collapse:

```python
# src/dpf/engine/core.py:522
def _compute_dt(self) -> float:
    dt_fluid = self.fluid._compute_dt(self.state)
    ...
    dt = min(dt_fluid, dt_circuit)
    # Floor: prevent CFL collapse from freezing the simulation.
    # 1e-12 s (1 ps) is the minimum meaningful MHD timestep for DPF
    # (sub-picosecond dynamics are not physical at MHD scales).
    dt = max(dt, 1e-12)
    return dt
```

**Risk**: Could violate CFL stability if the floor is above the true CFL limit.
**Benefit**: Prevents infinite loops. The simulation would advance (possibly with some accuracy loss) rather than freezing.

### Fix B: Decouple MHD CFL from engine timestep during radial phase (medium-risk)

When the snowplow is in radial phase, use `dt_circuit` as the engine timestep and let the MHD sub-stepping handle the CFL internally:

```python
# src/dpf/engine/core.py:522
def _compute_dt(self) -> float:
    dt_fluid = self.fluid._compute_dt(self.state)
    ...
    dt = min(dt_fluid, dt_circuit)
    # During radial phase, the MHD CFL can collapse due to electrode BC
    # discontinuities on coarse grids. Use circuit dt and rely on MHD
    # sub-stepping (capped at 100) to handle the fast MHD timescale.
    if (self.snowplow is not None
        and self.snowplow.phase in ("radial", "reflected")
        and dt < 1e-12):
        dt = dt_circuit
    return dt
```

**Risk**: MHD sub-stepping cap of 100 may not be enough.

### Fix C: Fix the root cause -- stabilize electrode BC on coarse grids (correct fix)

The real issue is the sharp B_theta discontinuity from the radial zipper BC on a 1-2 cell grid. Options:

1. **Smooth the radial zipper**: Instead of `B[1, :ir_shock, :, :] = 0.0`, taper B_theta over 2-3 cells using a smooth profile (e.g., tanh ramp). This prevents the single-cell discontinuity that triggers the instability.

   ```python
   # src/dpf/engine/circuit_coupling.py:610-615
   if self.snowplow.phase in ("radial", "reflected"):
       r_shock = self.snowplow.r_shock
       dr = self.config.dx
       for ir in range(nx):
           r = (ir + 0.5) * dr
           # Smooth taper over 3 cells instead of sharp cutoff
           alpha = 0.5 * (1 + np.tanh((r - r_shock) / (1.5 * dr)))
           B[1, ir, :, :] *= alpha
   ```

2. **Skip electrode BC during radial phase on coarse grids**: If grid resolution is < 32 radial cells, the electrode BC introduces more instability than physics. Let the snowplow Lp handle the coupling.

3. **Increase vacuum density floor**: Cells near the radial zipper boundary should maintain a minimum density to cap the Alfven speed:

   ```python
   # In _apply_electrode_bc, after radial zipper:
   rho_floor_zipper = 1e-4 * self.state["rho"].max()
   self.state["rho"] = np.maximum(self.state["rho"], rho_floor_zipper)
   ```

### Fix D: Increase test grid resolution (test-level fix)

Use 32x1x64 instead of 16x1x32 to resolve the radial shock front with more cells:

```python
# tests/test_validation_consolidated.py:4435
preset["grid_shape"] = [32, 1, 64]
preset["dx"] = 5e-3  # 5 mm
```

**Risk**: Test becomes 4x slower. May still exhibit the instability at a later time.

## 7. Recommended Fix Order

1. **Immediate**: Fix A (dt floor) -- prevents the test from spinning at dt=0
2. **Short-term**: Fix C.1 (smooth radial zipper) -- addresses root cause
3. **Validation**: Fix D (finer grid) -- confirms the physics works at higher resolution
4. **Long-term**: Fix C.3 (density floor near zipper) -- defensive measure

## 8. Files Requiring Changes

| File | Line(s) | Change |
|------|---------|--------|
| `src/dpf/engine/core.py` | 522-542 | Add dt floor (Fix A) |
| `src/dpf/engine/circuit_coupling.py` | 610-615 | Smooth radial zipper (Fix C.1) |
| `tests/test_validation_consolidated.py` | 4435-4436 | Increase grid resolution (Fix D) |

## 9. Verification

After applying fixes, verify:
1. Snowplow reaches pinch phase (phase transitions: rundown -> radial -> reflected -> pinch)
2. Current dip > 5% in the 3 us post-peak window
3. B-field remains bounded (|B|_max < 100 T throughout simulation)
4. dt remains > 1e-12 s throughout simulation
5. Full 12 us simulation completes in < 2000 MHD steps
