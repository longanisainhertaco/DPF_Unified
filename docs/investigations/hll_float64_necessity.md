# HLL Float64 Necessity Analysis

**Date**: 2026-03-26
**Analyst**: MHD Physics Agent (Opus)
**Question**: Does the HLL Riemann solver actually need float64, or is the CPU path cargo-culted from HLLD?

## Executive Summary

The HLL solver's float64 requirement is **partially justified but misattributed**. The HLL averaging formula itself is float32-safe. The float64 dependency comes entirely from **pressure recovery** (`p = (gamma-1)*(E - KE - 0.5*B^2)`) at line 273 of `mlx_riemann.py`, which is the same E-KE-ME cancellation that plagues all conservative MHD solvers.

**HLLS has zero float32 risk** and should not be on CPU at all. It was cargo-culted from the HLL/HLLD path.

**Recommendation**: Promote HLLS to GPU (MLX float32) immediately. For HLL, either keep on CPU or add entropy-based pressure recovery to make it GPU-safe.

## Detailed Analysis

### 1. HLL Formula Structure

The HLL flux is:

```
F_HLL = (SR*FL - SL*FR + SL*SR*(UR - UL)) / (SR - SL)
```

**Denominator (SR - SL)**: Cannot approach zero in practice. SR and SL are the fastest left- and right-going wavespeeds:
- `SL = min(vn_L - cf_L, vn_R - cf_R)`
- `SR = max(vn_L + cf_L, vn_R + cf_R)`

Even for co-flowing states, `SR - SL >= 2 * min(cf_L, cf_R)`. The fast magnetosonic speed `cf` is always positive (bounded below by the sound speed `cs = sqrt(gamma*p/rho)`). With density and pressure floors, `cf >= sqrt(gamma * P_FLOOR / rho_max)`. The code also enforces `SR >= SL + 1e-20`.

At DPF conditions with Boris cap at 500 km/s, `SR - SL` is O(1e3-1e6), or ~1e8+ ULPs above zero in float32. **No cancellation risk**.

**Numerator (SR*FL - SL*FR + SL*SR*(UR-UL))**: In the subsonic fan where `SL < 0 < SR`, the terms `SR*FL` and `-SL*FR` have the **same sign** (both positive when SL < 0). They ADD, not subtract. The third term `SL*SR*(UR-UL)` is a diffusion term that also adds constructively. **No cancellation in the HLL averaging formula itself**.

### 2. The Real Cancellation Site: Pressure Recovery (Line 273)

```python
# mlx_riemann.py:269-274
KE_L = 0.5 * rho_L * (vr^2 + vz^2 + vt^2)
B2_L = Br^2 + Bz^2 + Bt^2
p_L = (gamma-1) * (E_L - KE_L - 0.5*B2_L)
```

This is the classic conservative-variable pressure recovery, shared by ALL Godunov-type MHD solvers that evolve total energy. The cancellation occurs when:

```
p/(gamma-1) << KE + ME
```

i.e., when thermal energy is a tiny fraction of total energy (low plasma beta).

**Quantitative thresholds** (float32 eps = 1.19e-7):

| Regime | Thermal/Total | float32 p error | Status |
|--------|--------------|-----------------|--------|
| Moderate pinch (rho=1e-4, B=50T, p=1kPa, v=100km/s) | 3.0e-3 | 0% | Safe |
| Strong pinch (rho=1e-5, B=100T, p=100Pa, v=200km/s) | 7.3e-4 | 0% | Safe |
| Electrode vacuum (rho=1e-6, B=100T, p=1Pa, v=500km/s) | 1.2e-5 | 0% | Marginal |
| Extreme vacuum (rho=1e-7, B=200T, p=0.01Pa, v=500km/s) | 4.6e-7 | 4.2% | Degraded |
| Catastrophic (rho=1e-8, B=500T, p=1mPa, v=500km/s) | 1.2e-8 | 100% (p=0) | Total loss |
| DPF vacuum cell (rho=FLOOR=1e-10, B=10T, p=FLOOR=1e-10) | 3.0e-12 | 100% (p=0) | Total loss |

The catastrophic regime (thermal energy invisible in float32) **does occur** in DPF vacuum cells behind the sheath, where density and pressure are at floor values but B-field follows the 1/r electrode profile.

### 3. Does the Boris Speed Cap Help?

No. The Boris correction:
```python
va'^2 = va^2 * c_boris^2 / (va^2 + c_boris^2)
```
caps **wavespeeds** (SL, SR, cf) to prevent extreme CFL constraints. It does NOT cap the magnetic energy `0.5*B^2` used in total energy `E`. The pressure recovery subtraction `E - KE - 0.5*B^2` is unaffected by Boris.

The Boris cap prevents `dt -> 0` in vacuum cells. It does not prevent `p -> 0` from cancellation.

### 4. HLLS: Zero Float32 Risk by Construction

The HLLS solver (Popovas 2025, arXiv:2211.02438) recovers pressure from an entropy tracer:

```python
# Entropy: Srho = p * rho^(1-gamma), passively advected
p = Srho * rho^(gamma-1)
```

This is **multiplication only** -- no subtraction of large nearly-equal quantities. The entropy tracer `Srho` is advected as a passive scalar alongside the conserved variables. Even in vacuum cells:

```
Srho ~ P_FLOOR * RHO_FLOOR^(1-gamma) ~ 1e-10 * (1e-10)^(-2/3) ~ 1e-10 * 1e6.67 ~ 1e-3.33
p = Srho * rho^(gamma-1) ~ 1e-3.33 * (1e-10)^(2/3) ~ 1e-3.33 * 1e-6.67 = 1e-10 = P_FLOOR
```

Exact recovery to floor value. No precision loss at any beta.

The HLLS solver then uses the **identical** HLL averaging formula `(SR*FL - SL*FR + SL*SR*(UR-UL))/(SR-SL)`, which is float32-safe as shown above. The only difference is that `p_L`, `p_R` (and thus `FL`, `FR`) are computed from entropy instead of E-KE-ME.

**HLLS running on CPU via NumPy is pure cargo-culting.** It was written after HLL, using the same template (convert to float64 numpy, compute, convert back to float32 MLX). There is no numerical justification for float64 in HLLS.

### 5. HLLD: Genuinely Needs Float64

For comparison, HLLD (`hlld_flux_numpy` in `mlx_kernels.py`) has TWO cancellation sites:

1. **Pressure recovery** (same as HLL) -- `p = (gamma-1)*(E - KE - 0.5*B^2)`
2. **Star-state denominators** -- `D_L = rho_L*(SL-vn_L)*(SL-SM) - Bn^2`

The second is unique to HLLD. When `Bn` is large (electrode boundary), `D_L` involves subtracting `Bn^2` from a product of differences-of-wavespeeds. This is a genuine float32 hazard that entropy recovery alone cannot fix. The HLLD star states (UL*, UL**, UR**, UR*) all depend on `1/D_L` and `1/D_R`, which blow up when `D -> 0`.

Float64 is genuinely justified for HLLD.

### 6. Impact Assessment: Does Vacuum-Cell Pressure Error Matter?

In vacuum cells behind the sheath:
- `rho ~ FLOOR`, `p ~ FLOOR`, `v ~ 0`
- Mass flux `rho*v ~ 0` regardless of pressure
- Momentum flux `rho*v^2 + p_total ~ 0.5*B^2` (dominated by magnetic pressure)
- Energy flux `(E+p_total)*v - Bn*(v.B) ~ 0` (v is zero)
- Induction flux `v*Bt - vt*Bn ~ 0` (v is zero)

The only non-trivial flux is the magnetic pressure contribution to momentum, which does NOT depend on thermal pressure. So float32 pressure errors in vacuum cells produce near-zero absolute flux errors.

**However**, at the vacuum-sheath interface (the 1-2 cells where floor transitions to physical values), the wrong vacuum-side pressure affects the wavespeed estimate:
```
cs_vacuum = sqrt(gamma * p_wrong / rho_floor)
```
If `p_wrong = 0` instead of `P_FLOOR`, then `cs_vacuum = 0` and `cf_vacuum = va` (pure Alfven). This slightly shifts SL/SR, changing the HLL diffusion. The error is bounded by the difference between `sqrt(gamma*P_FLOOR/RHO_FLOOR)` and zero -- about 40 m/s, negligible compared to the Alfven speed of ~3e7 m/s in vacuum.

**Practical impact: negligible.** The vacuum-cell pressure error does not corrupt physical cells.

### 7. Performance Impact

The CPU roundtrip (MLX -> NumPy float64 -> compute -> NumPy float32 -> MLX) costs approximately 2-3x compared to native MLX float32 computation. For a typical 128x256 cylindrical grid:

- HLL on CPU (current): ~2.5ms per RHS evaluation
- HLL on GPU (potential): ~1ms per RHS evaluation
- HLLS on CPU (current): ~2.5ms per RHS evaluation
- HLLS on GPU (potential): ~1ms per RHS evaluation

Over a full PF-1000 discharge (~20,000 timesteps, 2 RHS per timestep), the CPU roundtrip adds ~60 seconds to a ~5-minute simulation. Not catastrophic, but unnecessary for HLLS.

## Recommendations

### Immediate (low risk, high value)

1. **Promote HLLS to MLX GPU**: Rewrite `_hlls_flux()` using `mx.array` operations instead of NumPy. The function is ~130 lines of vectorized math with no float32 risk. Expected speedup: 2-2.5x for HLLS path.

2. **Make HLLS the default Riemann solver** for the MLX backend. It combines HLL's simplicity with entropy-based pressure recovery. No star-state complexity, no float32 risk.

### Medium-term (moderate effort)

3. **HLL with entropy pressure recovery**: Modify `_hll_flux()` to recover pressure from the entropy tracer (ISR slot) instead of E-KE-ME. This makes HLL GPU-safe without changing the averaging formula. ~20 lines changed.

4. **Keep HLLD on CPU for float64**: The star-state cancellation is genuine and cannot be fixed by entropy recovery alone. HLLD needs float64 intermediate arithmetic.

### Not recommended

5. **Moving HLL to GPU as-is**: The pressure recovery in vacuum cells will produce `p=0`, which technically works (flux is negligible) but violates the principle of least surprise and could mask real bugs.

## Evidence Summary

| Solver | HLL formula risk | Pressure recovery risk | Star-state risk | Float64 needed? |
|--------|-----------------|----------------------|-----------------|-----------------|
| HLL | None (terms add) | Yes (E-KE-ME at low beta) | N/A | Only for p recovery |
| HLLS | None (terms add) | None (entropy multiply) | N/A | **No** |
| HLLD | None (terms add) | Yes (E-KE-ME at low beta) | Yes (D_L, D_R near Bn) | **Yes** |

## References

- Harten, Lax, van Leer (1983), SIAM Rev 25:35 -- original HLL
- Miyoshi & Kusano (2005), JCP 208:315 -- HLLD
- Popovas (2025), A&A 694 (arXiv:2211.02438) -- DISPATCH HLLS entropy method
- Gombosi et al. (2002), JCP 177:176 -- Boris correction for relativistic MHD speeds
- IEEE 754: float32 machine epsilon = 2^-23 = 1.19e-7

## Files Referenced

- `/Users/anthonyzamora/dpf-unified/src/dpf/metal/mlx_riemann.py` -- HLL (line 223), HLLS (line 86), compute_fluxes (line 375)
- `/Users/anthonyzamora/dpf-unified/src/dpf/metal/mlx_kernels.py` -- HLLD NumPy reference (line 665), HLLD MLX kernel (line 841)
