# Post-Pinch Column Expansion: Current Dip Physics and Root Cause Analysis

**Date**: 2026-03-26
**Status**: Research complete, fix options ranked
**Fidelity target**: 60% current dip depth (PF-1000 experimental)

---

## 1. Physics of the Current Dip

After the plasma column reaches maximum compression (pinch), it expands outward due to:
1. **Thermal pressure**: The shock-heated plasma at T ~ 1-10 keV has pressure p >> B^2/(2*mu_0) at stagnation
2. **m=0 sausage instability**: Fastest-growing MHD mode disrupts the column on timescale tau_m0 ~ 30-100 ns
3. **Reflected shock**: The inward shock bounces off the axis, creating an outward-propagating reflected shock

The expansion reduces the plasma inductance:
```
Lp = (mu_0 / 2*pi) * z_f * ln(b / r_column)
```
As r_column increases toward the cathode radius b, ln(b/r) decreases, so Lp decreases. The rate of change dLp/dt is negative (inductance dropping).

The circuit equation:
```
(L0 + Lp) * dI/dt = V_cap - R*I - I * dLp/dt
```

The term `-I * dLp/dt` becomes `+I * |dLp/dt|` when dLp/dt < 0, which acts as a **positive voltage source** trying to maintain the current. However, the total inductance L_total = L0 + Lp is also decreasing, which means the stored magnetic energy 0.5 * L * I^2 is being released. Whether the current dips or rises depends on the competition between:

- **Resistive dissipation**: R * I (removes energy)
- **Inductive release**: -I * dLp/dt (adds energy if dLp/dt < 0)
- **Anomalous resistance**: R_anom from m=0 disruption micro-instabilities

For a 60% dip (PF-1000 observed), the anomalous resistance must dominate:
```
R_anom + R_0 >> |dLp/dt|
```

Quantitative estimate for PF-1000 at 27 kV:
- I_pinch ~ 600 kA, V_cap ~ 5 kV (near zero crossing)
- Required dLp/dt for dip: ~ V/I ~ 5000/600000 ~ 8 mOhm/us
- Column expansion velocity: v_expand ~ r_pinch / tau_m0 ~ 2e-3 / 50e-9 ~ 4e4 m/s
- dLp/dt = -(mu_0/2pi) * z_f * v_expand / r = -(2e-7) * 0.05 * 4e4 / 2e-3 = -2 mOhm/us
- So pure inductive dLp/dt gives only ~2 mOhm/us. The additional 6-10 mOhm/us must come from anomalous resistance.

---

## 2. Current Implementation: Three Code Paths

### Path A: Snowplow Model (src/dpf/fluid/snowplow.py)

The snowplow has a complete post-pinch expansion model:

1. **Phase detection**: When reflected shock terminates (reaches cathode or re-stagnates), `_pinch_complete = True` (line 839)
2. **Expansion velocity**: `v_expand = 3 * r_pinch / tau_m0` (line 862). The factor 3 accounts for non-uniform m=0 axial bulges (Haines 2011).
3. **Column radius evolution**: `r(t) = r_pinch + v_expand * (t - t_pinch)` (line 531)
4. **dL/dt computation**: `dL/dt = -(mu_0/2pi) * z_f * v_expand / r(t)` (line 539). Correctly negative.
5. **Anomalous resistance**: R_anom = 2 * |dL/dt_peak| * exp(-t/3*tau_m0) (lines 561-565). EMPIRICAL factor 2 ensures R_anom > |dL/dt|.

This is the model that produces the 51.8% dip claimed in the Phase R notes. It works.

### Path B: MLX Solver (src/dpf/metal/mlx_solver.py)

The MLX solver computes Lp from the MHD density field:

1. **Lp computation**: Lee formula with density-weighted r_eff (lines 1228-1233)
2. **CRITICAL BUG -- Monotonicity enforcement** (lines 1235-1239):
   ```python
   if Lp > self._Lp_max:
       self._Lp_max = Lp
   else:
       Lp = self._Lp_max  # <-- Lp can NEVER decrease!
   ```
   This clamps Lp to its historical maximum. Post-pinch, the column expands, r_eff increases, and the Lee formula gives a LOWER Lp. But the clamp forces Lp = Lp_max, so **dL/dt = 0 always after pinch**.
3. **dL/dt**: Computed as backward difference (line 1244), but since Lp is clamped, dL/dt = 0 post-pinch.

**Result**: Zero current dip from the MLX path.

### Path C: Engine Circuit Coupling (src/dpf/engine/circuit_coupling.py)

The engine's `_step_circuit_subcycle` function has three branches for Lp/dLdt source:

1. **Lines 85-137**: Snowplow active (radial phase). Blends snowplow Lp with MHD density-weighted Lp. dL/dt comes from either snowplow or MHD feedback.
2. **Lines 139-160**: Snowplow inactive (post-pinch). Uses `snowplow._frozen_result()` which provides the correct expanding column dL/dt and R_anom.
3. **Lines 162-167**: No snowplow, density-weighted coupler only.

In the post-pinch branch (lines 139-160), there's a priority check:
```python
if self.coupling_mode == "density_weighted" and feedback is not None and feedback.Lp > 0:
    coupling.Lp = feedback.Lp          # MHD field Lp
    coupling.dL_dt = feedback.dLp_dt   # MHD field dLp/dt
    back_emf = feedback.back_emf
else:
    coupling.Lp = sp_Lp               # snowplow Lp
    coupling.dL_dt = sp_dLdt          # snowplow dLp/dt (correct!)
```

When using the MLX backend with `coupling_mode="density_weighted"`, the MHD feedback takes priority. But the MLX solver's Lp is monotonically clamped (Bug in Path B), so dLp/dt = 0. The snowplow's correct post-pinch dL/dt is **overridden** by the MLX solver's zero.

---

## 3. Root Cause: Signal Path Is Broken

The chain of failure:

```
Snowplow computes correct post-pinch dL/dt (negative, ~2 mOhm/us)
    |
    v
Engine checks: "is density_weighted mode active AND does MHD feedback exist?"
    |
    YES (MLX solver provides feedback with Lp > 0)
    |
    v
Engine uses MLX feedback.dLp_dt instead of snowplow dLdt
    |
    v
MLX solver's Lp is monotonically clamped --> dLp_dt = 0
    |
    v
Circuit sees dLp_dt = 0, back_emf = 0
    |
    v
No current dip. I(t) follows pure RLC decay.
```

For the pure snowplow path (no MHD backend), the dip works (51.8%). The bug only manifests when the MLX/Metal solver provides MHD field feedback that overrides the snowplow's post-pinch model.

---

## 4. What Lee Model Does (Reference)

Lee's 5-phase model (Lee & Saw, Phys. Plasmas 21, 072501, 2014):

- **Phase 1**: Axial rundown (snowplow)
- **Phase 2**: Radial inward shock (slug model)
- **Phase 3**: Reflected shock (outward bounce)
- **Phase 4**: Slow compression (quasi-equilibrium)
- **Phase 5**: Post-pinch column expansion

Phase 5 uses:
```
r(t) = r_min * (1 + (t - t_pinch) / tau_exp)
Lp(t) = (mu_0/2pi) * z_f * ln(b / r(t))
dLp/dt = -(mu_0/2pi) * z_f * dr/dt / r(t)
```

The expansion timescale tau_exp is derived from the reflected shock speed and thermal expansion. Lee fits this empirically for each device. For PF-1000: tau_exp ~ 50-100 ns gives the correct dip depth.

Our JAX Lee model (`src/dpf/jax/lee_model.py`) implements this with sigmoid soft-switching for differentiability. It correctly blends phases 2-3 using `w_inward` and `w_reflected` weights. The dLp/dt is continuous across phase transitions.

---

## 5. What Production Codes Do

### GORGON (Imperial College)

Full 3D resistive MHD. The column expansion and current dip emerge **self-consistently** from the MHD equations without a post-pinch model. The m=0 instability grows naturally, disrupts the column, and the resulting inductance change drives the current dip. No artificial expansion velocity needed.

Key: GORGON resolves the instability (3D), so it captures the non-uniform expansion. Our axisymmetric solver can't see m=1 kink modes, and m=0 growth is artificially symmetric.

### MACH2 (Air Force Research Lab)

2D MHD code for Z-pinches. Uses operator-split circuit coupling where the circuit solver receives Lp from volume integration of B-field energy:
```
W_B = integral(B^2 / (2*mu_0)) dV
Lp = 2 * W_B / I^2
```
This Lp naturally decreases post-pinch as the field diffuses and the column expands. No monotonicity clamp.

### PERSEUS (Princeton/NRL)

2D extended MHD. Self-consistent circuit coupling. The plasma inductance is computed from the magnetic field distribution at each timestep. Post-pinch dynamics (expansion, disruption) emerge from the MHD solution. Uses Boris correction for vacuum regions which improves field evolution behind the sheath.

### Common Pattern

All production codes let the **MHD solution determine Lp directly**. They do NOT clamp Lp to be monotonically increasing. The Lp decrease post-pinch is physical and essential for the current dip.

---

## 6. Fix Options (Ranked by Complexity)

### Option A: Remove Monotonicity Clamp Post-Pinch (Simple, 10 LOC)

**Location**: `src/dpf/metal/mlx_solver.py:1235-1239`

Replace the unconditional monotonicity clamp with a phase-aware version:

```python
# Monotonicity enforcement ONLY during compression (pre-pinch).
# Post-pinch, Lp MUST be allowed to decrease for current dip.
if not self._post_pinch:
    if Lp > self._Lp_max:
        self._Lp_max = Lp
    else:
        Lp = self._Lp_max
```

**Pinch detection**: When dLp/dt turns negative for N consecutive steps after reaching maximum, set `_post_pinch = True`. Or: when the density-weighted r_eff starts increasing.

**Pros**: Minimal code change. Lets MHD field evolution drive the Lp decrease.
**Cons**: The MHD solution may not capture the correct expansion rate in 2D axisymmetric. The monotonicity clamp existed to prevent noisy Lp oscillations during compression from destabilizing the circuit.
**Risk**: Without the clamp, noisy density fields during compression can cause spurious Lp oscillations. Need to keep the clamp pre-pinch but remove it post-pinch.

### Option B: Use Snowplow Post-Pinch dL/dt When MHD Lp Is Clamped (Medium, 30 LOC)

**Location**: `src/dpf/engine/circuit_coupling.py:139-160`

When in post-pinch mode, prefer the snowplow's expansion model over the MHD field feedback:

```python
# Post-pinch: snowplow expansion model provides dL/dt and R_anom.
# MHD field Lp is monotonically clamped (can't decrease), so its
# dLp_dt = 0 and is useless for the current dip.
# Always use snowplow post-pinch until MHD monotonicity is fixed.
if self.snowplow is not None and self.snowplow.pinch_complete:
    coupling.Lp = sp_Lp
    coupling.dL_dt = sp_dLdt
    coupling.R_plasma = max(coupling.R_plasma, sp_R)
```

**Pros**: Immediate fix. Uses the already-validated snowplow expansion model (51.8% dip).
**Cons**: Does not use MHD field information post-pinch. The snowplow expansion is a 0D model, not spatially resolved.
**Risk**: Low. The snowplow already produces the correct dip for PF-1000.

### Option C: MHD-Derived Expansion Velocity for Lp Evolution (Medium-Full, 80 LOC)

Compute the column expansion velocity from the MHD density/velocity fields:

```python
# Post-pinch: measure radial expansion from MHD velocity field
# v_expand = mass-weighted radial velocity at density peak
v_expand = sum(v_r * rho * dV) / sum(rho * dV)  # in expansion region
dLp_dt = -(mu_0/2pi) * z_f * v_expand / r_eff
```

**Pros**: Uses spatially-resolved MHD information. Self-consistent.
**Cons**: Requires identifying the expanding column region. The MHD expansion rate may be too slow if numerical diffusion damps the outward shock.
**Risk**: Moderate. Need to validate that the MHD solver produces a physically reasonable expansion velocity.

### Option D: Self-Consistent MHD Expansion with Dynamic Circuit Coupling (Full, 200+ LOC)

Remove the monotonicity clamp entirely. Compute Lp from B-field energy (volume integral) rather than the Lee formula:

```python
# Lp from magnetic field energy (MACH2 approach)
W_B = sum(B**2 / (2*mu_0) * dV)  # volume integral
Lp = 2 * W_B / I**2
```

This eliminates the density-weighted Lee formula and its associated noise issues. The B-field is smoother than the density field, so Lp from B-field energy is less noisy and doesn't need monotonicity clamping.

**Pros**: Physically correct. No clamp needed. B-field energy is a robust integral quantity.
**Cons**: Requires careful treatment of the external B-field (electrode BC). The B-field includes both plasma and vacuum contributions, so the Lp includes the vacuum inductance which must be subtracted.
**Risk**: Electrode boundary conditions inject B_theta, which contributes to W_B. Need to isolate the plasma contribution.

---

## 7. Recommended Fix Strategy

**Phase 1 (immediate)**: Implement Option B. Force snowplow post-pinch dL/dt to override MHD feedback. This unblocks the 51.8% dip for all engine configurations immediately.

**Phase 2 (next sprint)**: Implement Option A. Add phase-aware monotonicity control in the MLX solver. Keep the clamp during compression, release it post-pinch.

**Phase 3 (future)**: Implement Option D. Move to B-field energy Lp (MACH2/GORGON approach) for fully self-consistent circuit coupling without any monotonicity hack.

---

## 8. Anomalous Resistance: The Missing Ingredient

Even with correct dLp/dt, the current dip requires anomalous resistance. The physics:

1. m=0 instability creates necking of the plasma column
2. High-impedance necks develop ion-acoustic and lower-hybrid drift instabilities
3. These micro-instabilities provide anomalous resistivity: eta_anom ~ 100-1000x eta_Spitzer
4. The effective column resistance R_anom ~ eta_anom * z_f / (pi * r_neck^2) ~ 5-20 mOhm

Our snowplow model (lines 561-565 of snowplow.py) computes:
```python
dL_dt_peak = (mu_0 / 2pi) * z_f * v_expand / r_pinch
R_anom_peak = 2.0 * dL_dt_peak  # EMPIRICAL: ensures R_anom > |dL/dt|
R_anom = R_anom_peak * exp(-dt_since_pinch / (3 * tau_m0))
```

This is physically motivated but empirically tuned. The factor 2 and the 3*tau_m0 decay are not derived from first principles.

Production codes (GORGON, PERSEUS) achieve this naturally through spatially-resolved anomalous resistivity models (e.g., drift-velocity threshold: when v_drift > v_thermal, eta jumps). Our CIV/drift-velocity resistivity model in the MLX solver could provide this if activated post-pinch, potentially replacing the snowplow's empirical R_anom entirely.

---

## 9. Validation Targets

| Metric | PF-1000 Experimental | Current (MHD mode) | Target |
|--------|---------------------|-------------------|--------|
| Dip depth (I_min/I_peak) | ~40% of I_peak remains (60% dip) | ~0% dip (flat) | > 40% dip |
| Dip timing | ~1-2 us after peak | N/A | Within 0.5 us |
| Post-dip recovery | Partial current recovery | N/A | Qualitative match |
| Crowbar decay | Exponential L-R | Correct | Maintain |

The 51.8% dip from the snowplow-only path is already within 15% of the experimental 60% dip for PF-1000. The immediate goal is to propagate this to the full MHD engine path.

---

## 10. References

1. Lee, S. & Saw, S.H., "Plasma focus ion beam fluence and flux," Phys. Plasmas 21, 072501 (2014).
2. Goyon, C. et al., "MJOLNIR: A mega-joule pulsed-power driver," Phys. Plasmas 32, 033105 (2025).
3. Haines, M.G., "A review of the dense Z-pinch," Plasma Phys. Control. Fusion 53, 093001 (2011).
4. Angus, J.R. et al., "Shock scaling in the gas-puff Z-pinch," Phys. Plasmas 28, 012705 (2021).
5. Giuliani, J.L. & Commisso, R.J., "A review of the gas-puff Z-pinch as an x-ray and neutron source," IEEE Trans. Plasma Sci. 43, 2385 (2015).
6. Soto, L., "New trends and future perspectives on plasma focus research," Plasma Phys. Control. Fusion 47, A361 (2005).

---

## Appendix A: Code Locations

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Snowplow post-pinch model | `src/dpf/fluid/snowplow.py` | 505-572, 836-875 | Working (51.8% dip) |
| MLX Lp monotonicity clamp (BUG) | `src/dpf/metal/mlx_solver.py` | 1235-1239 | Blocks dip |
| Engine post-pinch Lp routing | `src/dpf/engine/circuit_coupling.py` | 139-160 | MHD overrides snowplow |
| Circuit solver (receives dLp/dt) | `src/dpf/circuit/rlc_solver.py` | 249-358 | Correct |
| JAX Lee model (reference) | `src/dpf/jax/lee_model.py` | 170-310 | Correct |
| Pinch physics diagnostics | `src/dpf/validation/pinch_physics.py` | 108-165 | Tau_m0, tau_exp formulas |
| Test: current dip | `tests/test_validation_consolidated.py` | ~2752 | Lee model test |
| Test: snowplow dip | `tests/test_snowplow_consolidated.py` | ~569-590 | Snowplow-only test |

## Appendix B: Numerical Estimates for PF-1000 (27 kV, 3 Torr D2)

```
I_peak = 1.87 MA
I_pinch = 600 kA (at moment of maximum compression)
r_pinch = 2 mm (10:1 compression ratio, a = 20 mm)
z_f = 50 mm (pinch column length, ~L_anode/4)
b = 32 mm (cathode radius)
tau_m0 = 31 * (2.0)^2 * sqrt(3.0) / (10 * 1.87) = 11.5 ns (Goyon Eq. 4)
tau_exp = 31.5 * (2.0)^2 * sqrt(3.0) / (10 * 1.87) = 11.7 ns (Goyon Eq. 3)
v_expand = 3 * r_pinch / tau_m0 = 3 * 0.002 / 11.5e-9 = 5.2e5 m/s

dLp/dt = -(mu_0/2pi) * z_f * v_expand / r_pinch
       = -(2e-7) * 0.05 * 5.2e5 / 0.002
       = -2.6 mOhm/us (from expansion)

R_anom = 2 * |dLp_dt_peak| = 5.2 mOhm (at pinch)
         decays as exp(-t / 34.5 ns)

Required for 60% dip: R_total ~ 10-15 mOhm/us over ~2 us
Snowplow model provides: ~5.2 mOhm initial, decaying
Gap: ~5-10 mOhm -- partially from Spitzer (R_spitzer ~ 1-3 mOhm at T_pinch)
                   partially from external circuit R0 = 2.3 mOhm

Total effective: R_anom + R_spitzer + R_0 ~ 8-10 mOhm
This is close to the required 10-15 mOhm. The 51.8% dip (vs 60% experimental)
is consistent with the model being ~15% short on total resistance.
```
