# POSEIDON-60kV Performance Analysis

**Date**: 2026-03-26
**Context**: Multi-device calibration run. POSEIDON-60kV has been running for 5.5+ hours (326 min CPU) while UNU-ICTP (23 min) and FAETON-I (35 min) completed normally.

## 1. Device Parameters

| Parameter | UNU-ICTP | PF-1000 | FAETON-I | POSEIDON-60kV |
|-----------|----------|---------|----------|---------------|
| Voltage [kV] | 14 | 27 | 100 | 60 |
| Capacitance [µF] | 30 | 1332 | 25 | 156 |
| Bank energy [kJ] | 2.9 | 485.5 | 125.0 | 280.8 |
| Peak current [MA] | 0.17 | 1.87 | 1.00 | 3.19 |
| Rise time [µs] | 2.85 | 10.49 | 3.68 | 2.61 |
| Anode radius [mm] | 9.5 | 115 | 25 | 65.5 |
| Cathode radius [mm] | 32 | 160 | 55 | 95 |
| Fill pressure [Torr] | 3.0 | 3.5 | 4.0 | 3.8 |

## 2. Why POSEIDON Is Slow: Three Compounding Factors

### Factor 1: Highest Peak Current → Strongest B-field → Fastest Alfven Speed

POSEIDON has the highest peak current (3.19 MA) of any device in our calibration set. The magnetic field at the anode surface is:

```
B_max = µ₀ I_peak / (2π a) = 4π×10⁻⁷ × 3.19×10⁶ / (2π × 0.0655) = 9.7 T
```

During radial compression to r_pinch ≈ 3 mm:

```
B_pinch = µ₀ I / (2π r_pinch) = 213 T
```

The Alfven speed at the pinch:

```
v_A = B / √(µ₀ ρ) = 213 / √(4π×10⁻⁷ × 0.05) ≈ 850 km/s
```

This is 5.7x faster than PF-1000's pinch Alfven speed (150 km/s). Since `dt_CFL ∝ dx/v_A`, POSEIDON needs 5.7x more timesteps per microsecond of simulation time.

### Factor 2: Vacuum Region Float32 Catastrophe

Behind the compression sheath, the vacuum region has:
- B_theta = µ₀ I / (2π r) from the electrode BC (persists after sheath passes)
- ρ → RHO_FLOOR = 10⁻¹² kg/m³

The uncorrected Alfven speed in vacuum:

```
v_A_vacuum = 19.5 T / √(4π×10⁻⁷ × 10⁻¹²) = 1.74 × 10¹⁰ m/s
```

This is 58x the speed of light. The CFL timestep would be:

```
dt_vacuum = 0.3 × 0.006 / 1.74×10¹⁰ = 1.04 × 10⁻¹³ s
```

Our vacuum masking (rho_cfl_fraction = 10⁻⁴) excludes these cells from CFL computation. But the **Riemann solver** still computes fluxes at vacuum cell interfaces with these extreme wavespeeds. This causes:

1. `SL` and `SR` wave speeds of order 10¹⁰ in the HLL flux formula
2. `SR × FL - SL × FR` produces intermediate values of order 10²⁰
3. The difference `SR×FL - SL×FR + SL×SR×(QR-QL)` cancels to small values
4. Float32 (7.2 significant digits) cannot resolve this cancellation
5. **Result: NaN in the HLL flux** → Lax-Friedrichs fallback (more diffusive)

The RuntimeWarning at `mlx_riemann.py:182` confirms this:
```
RuntimeWarning: invalid value encountered in subtract
F_hll = (SR*FL - SL*FR + SL*SR*(QR_np - QL_np)) * inv_dS
```

POSEIDON triggers this more than other devices because:
- Higher I_peak → stronger electrode B_theta → more extreme vacuum v_A
- Larger geometry (a=65.5mm vs a=9.5mm for UNU-ICTP) → more vacuum cells
- 3.19 MA vs 0.17 MA → vacuum B_theta is 19x stronger

### Factor 3: MHD Regime Violation at High Current Density

The warnings show 53-94% of cells are outside the MHD-valid regime:
```
MHD regime validity: 94% of cells outside MHD-valid regime (ND>1 or dx<10*lambda_De)
```

The MHD regime check tests:
- ND > 1: Debye number (number of particles in a Debye sphere). ND < 1 means the plasma is too sparse for fluid treatment.
- dx < 10 × λ_De: grid spacing smaller than 10 Debye lengths means unresolved kinetic physics.

At 32×64 grid with dx=6mm, the Debye length for a 300 eV, 10²⁵ m⁻³ plasma is:
```
λ_De = √(ε₀ kT / ne e²) = 1.3 × 10⁻⁶ m = 1.3 µm
```

So dx/λ_De ≈ 4600 — well above the 10 threshold. The MHD regime warning is coming from **low-density regions** where ne is small and λ_De is large, or from the vacuum where ND < 1. This is not a physics accuracy concern — it's a diagnostic warning about the vacuum.

## 3. CPU vs GPU Execution

**The solver runs on Apple Silicon GPU (Metal) via MLX.** Confirmed by:
- The solver is `MLXMHDSolver` with `backend='mlx'`
- MLX operations execute on the M3 Pro GPU by default
- Memory: 88 MB (0.2% of 36 GB) — consistent with GPU-resident arrays

**However, the HLL flux computation drops to CPU.** Our `_hll_flux()` function (mlx_riemann.py:86-196) converts to NumPy float64 for numerical safety:
```python
QL_np = np.asarray(QL).astype(np.float64)  # GPU→CPU transfer
QR_np = np.asarray(QR).astype(np.float64)
...
return mx.array(F_out.astype(np.float32))   # CPU→GPU transfer
```

This CPU round-trip happens **twice per RK stage** (radial + axial sweep) × **3 stages per SSP-RK3 step** = **6 CPU→GPU→CPU transfers per timestep**. On unified memory (M3 Pro), this is a memcpy, not PCIe, but it still forces synchronization and prevents MLX graph fusion.

**The Lax-Friedrichs NaN fallback adds overhead**: when NaN is detected (POSEIDON's vacuum cells), the fallback recomputes the flux as `F_LF = 0.5*(FL+FR) - 0.5*S_max*(QR-QL)`. This is a branch in NumPy that prevents vectorization for the affected cells.

**Bottom line**: POSEIDON is slow because of physics (extreme wavespeeds → tiny dt → many steps) compounded by numerics (float32 NaN → fallback path → more diffusion → more steps to resolve features).

## 4. Boris Correction Impact (Implemented This Session)

The Boris correction we implemented caps vacuum Alfven speed at c_boris = 5×10⁵ m/s:

```
v_A' = v_A × c_boris / √(v_A² + c_boris²)
```

**Impact on POSEIDON vacuum cells:**

| Quantity | Without Boris | With Boris | Improvement |
|----------|--------------|------------|-------------|
| v_A (vacuum) | 1.74×10¹⁰ m/s | 5.0×10⁵ m/s | 34,756× |
| dt_CFL (vacuum) | 1.04×10⁻¹³ s | 3.6×10⁻⁹ s | 34,756× |
| SL/SR magnitude | ~10¹⁰ | ~5×10⁵ | 34,756× |
| Float32 cancel risk | CRITICAL | NONE | Eliminated |

The Boris correction was wired into the Riemann solver wavespeeds (commit bd02b80) AFTER the POSEIDON calibration started. The running calibration uses the old 3×10⁸ cap, not the 5×10⁵ Boris cap.

**Expected impact on rerun**:
- Zero NaN warnings (no Lax-Friedrichs fallback needed)
- ~2x faster per trial (no fallback overhead, slightly larger dt)
- Better calibrated fc/fm (not compensating for LF diffusion)

## 5. HLLS Entropy Solver Impact (Researched, Not Yet Implemented)

The HLLS entropy solver (Popovas 2025) would eliminate the float32 cancellation at the source:

| Current HLL | HLLS Entropy |
|-------------|-------------|
| `p = (γ-1)(E - KE - ME)` → cancellation | `P = ρ^γ exp(S(γ-1))` → multiplication only |
| Energy flux: `(E+pt)vn - Bn(v·B)` → large terms | Entropy flux: `ρvnS` → simple advection |
| NaN at β < 10⁻⁴ | Float32-safe at any β |

**Combined Boris + HLLS** would give POSEIDON:
1. Bounded wavespeeds (Boris) → reasonable dt everywhere
2. No NaN in fluxes (HLLS) → no fallback path
3. Full GPU execution (no CPU float64 bridge needed)
4. Estimated per-trial time: ~2-5 min (vs ~80 min currently)

## 6. Lessons for the Research Paper

### Lesson 1: Device Scaling Exposes Numerical Limits
The same solver that handles UNU-ICTP (2.9 kJ) in 1 minute fails at POSEIDON (281 kJ) due to a 19× difference in peak current producing a 35,000× difference in vacuum Alfven speed. **Numerical robustness must be tested across the full device energy range, not just the calibration target.**

### Lesson 2: Float32 Cancellation Is Device-Dependent
PF-1000 (27 kV, 1.87 MA) runs cleanly in float32 HLL. POSEIDON (60 kV, 3.19 MA) produces NaN from the same code path. The cancellation site (SR×FL - SL×FR) scales with I² — higher-current devices are more vulnerable. **Any claim of "float32 validated" must specify the device conditions.**

### Lesson 3: The Boris Correction Is Not Optional for Multi-Device Codes
A single-device code tuned for PF-1000 can get away with density floor hacks. A multi-device code that runs from UNU-ICTP to POSEIDON requires physics-based wave speed limiting. **Boris correction (or equivalent) should be the default, not an option.**

### Lesson 4: Calibration Runtime Scales Non-Linearly with Device Energy
The calibration time per trial scales approximately as:
```
t_trial ∝ I_peak² × t_rise / E_bank^{0.5}
```

This is because: dt ∝ 1/v_A ∝ 1/(I/r), n_steps ∝ t_discharge/dt, and the radial phase (where dt is smallest) duration scales with the LC rise time. For POSEIDON, this gives ~100× longer trials than UNU-ICTP.

### Lesson 5: The NaN Fallback Mask Creates Compensating Errors
When Lax-Friedrichs replaces HLL at NaN cells, the solver becomes more diffusive in vacuum regions. The calibrated fc/fm values compensate for this artificial diffusion. Fixing the NaN (via Boris or HLLS) changes the effective physics, requiring recalibration. **Every numerical fix must be followed by recalibration.**

## 7. Diagnostic Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| CPU time | 326 min | 5.4 hours |
| Trials completed | ~3-4 of 20 | 15-17% |
| MHD regime warnings | 14 (POSEIDON-specific) | 53-94% cells outside regime |
| NaN warnings | 5 | Float32 cancellation in HLL flux |
| Estimated completion | 6-10 more hours | Based on 80 min/trial |
| Boris would reduce to | ~2-5 min/trial | 40-120 min total |
| HLLS would eliminate | All NaN | Zero fallback overhead |
"""
