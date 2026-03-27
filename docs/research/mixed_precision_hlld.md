# Mixed-Precision HLLD Riemann Solver for MLX Backend

Research document: Where catastrophic cancellation occurs in HLLD float32, what
production codes do about it, and a practical mixed-precision strategy for DPF-Unified.

Date: 2026-03-26
Status: Research only (no source changes)

---

## 1. Where Float32 Cancellation Occurs in HLLD

The HLLD solver (Miyoshi & Kusano 2005, JCP 208:315) computes ~40 intermediate
values per cell interface. Most are safe in float32. The following 6 operations
are the cancellation sites, traced from the Metal kernel in `mlx_kernels.py:422-607`
and the NumPy reference in `mlx_kernels.py:665-838`.

### 1.1 Pressure Recovery: `p = (gamma-1) * (E - KE - ME)`

**Location**: `cons_to_prim` in `_HLLD_HEADER` (line 398)

```metal
p = max((gamma - 1.0f) * (E - ke - mag), P_FLOOR);
```

**Cancellation mechanism**: In magnetically dominated cells (plasma beta << 1),
E ~ ME >> p/(gamma-1). The subtraction E - KE - ME cancels most significant
digits. At the DPF electrode boundary: E ~ 1e8, ME ~ 1e8, p/(gamma-1) ~ 1e2.
In float32 (24-bit mantissa, ~7 decimal digits), the result has 0-1 significant
digits.

**Severity**: CRITICAL in DPF. At the sheath front (r ~ 1-5 mm), B_theta ~ 1-10 T
creates beta ~ 0.001-0.01. The electrode ghost cells have B_theta set by 1/r
boundary conditions, reaching 50-100 T at the inner electrode surface where
beta ~ 1e-5.

**Current mitigation**: The HLL and HLLS solvers use entropy-derived pressure
`p = Srho * rho^(gamma-1)` for wavespeeds, avoiding this subtraction entirely.
The HLLD solver still uses the E-KE-ME subtraction for p_L and p_R.

### 1.2 Contact Wave Speed: `S_M` Denominator

**Location**: Metal kernel line 467-470

```metal
float denom_SM = rho_R * (SR - vn_R) - rho_L * (SL - vn_L);
float SM = (rho_R * vn_R * (SR - vn_R) - rho_L * vn_L * (SL - vn_L) + pt_L - pt_R) / denom_SM;
```

**Cancellation mechanism**: When left and right states are nearly identical (weak
discontinuity or smooth flow), `rho_R*(SR-vn_R) ~ rho_L*(SL-vn_L)` and the
denominator cancels. The numerator has the same issue plus `pt_L - pt_R ~ 0`.
The result is an indeterminate form 0/0.

**Severity**: MODERATE. Protected by the `TINY` floor on the denominator, which
prevents division by zero but returns an inaccurate SM. In smooth regions this
produces more diffusion than necessary but doesn't cause NaN. Becomes dangerous
when the floor activates AND the resulting SM feeds into subsequent star-state
computations.

### 1.3 Star-State Denominators: `D_L` and `D_R`

**Location**: Metal kernel lines 482-486

```metal
float D_L = rho_L * (SL - vn_L) * (SL - SM) - Bn * Bn;
float D_R = rho_R * (SR - vn_R) * (SR - SM) - Bn * Bn;
```

**Cancellation mechanism**: This is THE critical HLLD cancellation. The first
term `rho * (S_wave - vn) * (S_wave - SM)` is positive and of order
`rho * cf^2 ~ rho * (B^2/rho) = B^2`. The second term is `Bn^2`. When
Bn is large relative to Bt (nearly parallel B-field), these two terms are
nearly equal and cancel.

Specifically: when the flow approaches the switch-on degeneracy (where the
slow magnetosonic speed equals the Alfven speed), D_L -> 0 from either side.
In float32:
- `rho_L * (SL - vn_L) * (SL - SM)` ~ 1e6 (B_theta^2 / mu_0 in HL units)
- `Bn^2` ~ 1e6

The subtraction loses ~7 digits in float32, leaving noise.

**Severity**: CRITICAL. D_L and D_R appear in the denominators of the
transverse velocity star-states (vt1_sL, vt2_sL) and transverse B-field
star-states (Bt1_sL, Bt2_sL). A noisy D produces noisy star states, which
propagate into the energy star-state e_sL, which propagates into the
double-star energy e_dsL, which becomes the flux. One bad D_L can corrupt
the entire solution at that interface.

**Current mitigation**: The Metal kernel uses `inv_DL = (|D_L| < TINY) ? 0 : 1/D_L`,
which zeros out the correction when D is small. This is equivalent to falling
back to HLL at those interfaces (the star-state corrections vanish), but it's
discontinuous -- D can be small-but-nonzero and still yield garbage.

### 1.4 Total Pressure Star-State: `pt*`

**Location**: Metal kernel line 472

```metal
float pt_star = max(pt_L + rho_L * (SL - vn_L) * (SM - vn_L), P_FLOOR);
```

**Cancellation mechanism**: When SM ~ vn_L (contact wave speed matches left
velocity), the second term vanishes, but the floating-point evaluation of
`(SL - vn_L) * (SM - vn_L)` may accumulate round-off from the noisy SM
(Section 1.2). Additionally, both `SL - vn_L` and `SM - vn_L` are differences
of similar-magnitude velocities.

**Severity**: LOW-MODERATE. The `P_FLOOR` clamp prevents negative pt*, and pt*
appears in the energy star-state numerator where it's additive rather than
multiplicative. Errors here add an O(epsilon) perturbation to the energy flux.

### 1.5 Energy Star-State: `e_sL`, `e_sR`

**Location**: Metal kernel lines 502, 506

```metal
float e_sL = ((SL - vn_L) * E_L - pt_L * vn_L + pt_star * SM + Bn * (vB_L - vB_sL)) / denom_L;
```

**Cancellation mechanism**: The numerator is a sum of 4 terms that should yield
the energy density in the star region. When the star state is close to the
original state (weak wave), this is a small difference of large numbers.
Furthermore, the last term `Bn * (vB_L - vB_sL)` involves the difference of
two dot products that are similar when the Alfven wave correction is small.

**Severity**: MODERATE. Inherits all accumulated errors from SM, pt*, D_L, and
the star-state velocities/B-fields. The division by denom_L (= SL - SM, which
can be small) amplifies the error.

### 1.6 Double-Star Energy: `e_dsL`, `e_dsR`

**Location**: Metal kernel lines 532-534

```metal
float e_dsL = e_sL - sqrt_rho_sL * (vB_sL - vB_dsL) * sign_Bn;
```

**Cancellation mechanism**: When Bn is small (the common case in cylindrical DPF
where B_theta dominates), the double-star states equal the star states, and this
term vanishes correctly. When Bn is large, the `vB_sL - vB_dsL` difference
involves star-state values that are already contaminated by D_L errors.

**Severity**: LOW for DPF (B_theta-dominated geometry). Would be CRITICAL for
problems with strong B_r or B_z (e.g., spherical blast, linear wave tests).

### Summary: The Cancellation Cascade

```
cons_to_prim (E-KE-ME) ─── corrupted p_L, p_R
           │
           ├──► pt_L, pt_R
           │        │
           ├──► SM denominator (rho*(S-v) subtraction)
           │        │
           │        ├──► SM ──► pt* ──► e_sL numerator
           │        │                        │
           │        └──► D_L = rho*(SL-vn)*(SL-SM) - Bn^2  ◄── THE CRITICAL ONE
           │                  │
           │                  ├──► vt1_sL, vt2_sL, Bt1_sL, Bt2_sL
           │                  │        │
           │                  │        └──► vB_sL ──► e_sL ──► e_dsL
           │                  │
           │                  └──► (if D_L ~ 0, entire star state is garbage)
           │
           └──► Fast magnetosonic cf (OK -- numerically stable form used)
```

**Key insight**: The cascade has ONE root cause (pressure recovery) and ONE
critical amplifier (D_L/D_R). Fixing those two eliminates ~90% of the float32
risk.

---

## 2. What Production Codes Do

### 2.1 Athena++ (Stone et al. 2020, ApJS 249:4)

Athena++ uses float64 throughout. No mixed-precision option. Their HLLD
implementation (src/hydro/rsolvers/mhd/hlld.cpp) handles the D_L degeneracy
with:

```cpp
// If Bn = 0, set star tangential quantities equal to input
if (0.5*Bn_sq < TINY_NUMBER * pt_star) {
    vt1_sL = vt1_L;  vt2_sL = vt2_L;
    Bt1_sL = Bt1_L;  Bt2_sL = Bt2_L;
    // (same for right)
}
```

This is the same `bn_small` branch in our Metal kernel. They also clamp:
```cpp
SL_star = SM - fabs(Bn)/sqrt(rho_sL);  // can't be less than SL
SL_star = fmax(SL_star, SL);
```

**Key point**: Athena++ never runs in float32 for production MHD. Their GPU port
(AthenaK) uses Kokkos `Real` type which defaults to float64. The only Athena
float32 path is for I/O compression.

### 2.2 DISPATCH / HLLS (Popovas 2025, A&A 694, arXiv:2211.02438)

DISPATCH avoids the HLLD cancellation problem entirely by using HLLS -- an
entropy-based HLL solver. Their key insight:

> "The HLLS solver uses the entropy equation to compute pressure, eliminating
> the catastrophic cancellation in E - KE - ME. The wavespeeds are computed
> from entropy-derived pressure, and the HLL averaging formula has no
> subtractions of similar-magnitude quantities."

DISPATCH runs in float32 on GPU with HLLS and achieves comparable results to
float64 HLLD on their test suite (Orszag-Tang, MHD blast, rotor). They accept
the additional diffusion of a 2-wave solver (no contact or Alfven resolution)
as the cost of float32 safety.

**Our implementation**: `_hlls_flux_gpu()` and `_hll_flux_gpu()` in
`mlx_riemann.py` already implement this approach. HLL uses entropy for
wavespeeds but conserved E for fluxes. HLLS uses entropy for everything.

### 2.3 FLASH (Fryxell et al. 2000)

FLASH uses a dual-energy formalism (Bryan et al. 2014) for pressure recovery
but runs HLLD in float64. Their switching criterion `e_int/E` for dual-energy
has a circular dependency -- the numerator IS the corrupted float32 subtraction.
We already identified and avoided this (see `mlx_primitives.py:213-214`
comments).

### 2.4 Enzo / Enzo-E

Same dual-energy as FLASH, same float64 HLLD. Enzo-E (the rewrite) supports
float32 mode but recommends float64 for MHD.

### 2.5 MPI-AMRVAC (Xia et al. 2018)

Uses HLLD in float64. For float32, they offer an "entropy fix" where HLLD
is replaced by HLL near degenerate points (where D_L or D_R is small). This
is essentially what our `inv_DL = 0 when |D_L| < TINY` achieves.

---

## 3. The Minimal Float64 Set

Of the ~40 intermediate values in HLLD per cell interface, only a subset
needs float64 to eliminate the cancellation cascade. Traced from the analysis
in Section 1:

### Tier 1: MUST be float64 (cancellation sites)

| Variable | Computation | Why |
|----------|-------------|-----|
| `p_L`, `p_R` | `(gamma-1) * (E - KE - ME)` | Root of the cascade. 7+ digits cancel at beta < 0.01. |
| `D_L`, `D_R` | `rho*(SL-vn)*(SL-SM) - Bn^2` | Critical amplifier. Near switch-on, 7+ digits cancel. |
| `denom_SM` | `rho_R*(SR-vn_R) - rho_L*(SL-vn_L)` | 0/0 form at weak discontinuities. |
| `SM` (numerator) | `rho_R*vn_R*(SR-vn_R) - rho_L*vn_L*(SL-vn_L) + pt_L - pt_R` | Same cancellation as denominator. |

**Count**: 6 values (p_L, p_R, D_L, D_R, SM_num, SM_den) = the critical path.

### Tier 2: SHOULD be float64 (error amplification)

| Variable | Computation | Why |
|----------|-------------|-----|
| `pt_L`, `pt_R` | `p + 0.5*B^2` | Depends on corrupted p. |
| `pt_star` | `pt_L + rho_L*(SL-vn_L)*(SM-vn_L)` | Depends on noisy SM. |
| `e_sL`, `e_sR` | `((SL-vn_L)*E_L - pt_L*vn_L + ...)/denom_L` | Accumulates Tier 1 errors, divided by small denom. |
| `vt1_sL`, `vt2_sL` | `vt1_L - Bn*Bt1_L*(SM-vn_L)/D_L` | Divides by noisy D_L. |
| `Bt1_sL`, `Bt2_sL` | `Bt1_L * f_L / D_L` | Divides by noisy D_L. |

**Count**: ~12 values on each side (L and R).

### Tier 3: Safe in float32

| Variable | Computation | Why |
|----------|-------------|-----|
| `rho_L`, `rho_R` | Direct from state | No subtraction. |
| `vn_L`, `vn_R` | `mom / rho` | Division, not subtraction. |
| `Bn`, `Bt1`, `Bt2` | Direct from state | No subtraction. |
| `SL`, `SR` | `min/max(vn +/- cf)` | Min/max are safe. |
| `cf_L`, `cf_R` | `sqrt(0.5*(a2+va2+disc))` | Stable discriminant form. |
| `rho_sL`, `rho_sR` | `rho*(S-v)/denom` | Ratio, clamped by floor. |
| `sqrt_rho_sL/R` | `sqrt(rho_s)` | Always positive after floor. |
| Physical flux `FL`, `FR` | Products and sums | No near-cancellation. |
| Final flux selection | `where` on wave speeds | Comparison, not arithmetic. |
| Double-star quantities | Additive corrections | Small when Bn small (DPF). |

**Count**: ~20 values. These can stay float32.

### Minimal float64 strategy

Promote **Tier 1 + Tier 2** to float64 = ~18 intermediate scalars per interface.
Keep **Tier 3** in float32 = ~20 values + all flux assembly.

The pattern is:
1. Upcast `E_L, KE_L, ME_L, E_R, KE_R, ME_R` to float64 for pressure recovery
2. Compute SM, D_L, D_R, star-state velocities/B-fields/energies in float64
3. Downcast star states to float32 for flux assembly (additive operations, safe)
4. Continue in float32 for physical flux, HLL averaging, wave-speed selection

---

## 4. MLX Implementation Path

### 4.1 MLX float64 support

MLX supports `mx.float64` on CPU. On Apple Metal GPU, only float32 is available.
This creates three implementation options:

**Option A: CPU float64 for critical path (current approach)**

The `_hlld_flux_cpu64()` function already does this: converts to NumPy float64,
runs the full HLLD in float64, casts back to float32. Cost: ~2.5x slower than
GPU due to CPU execution + GPU-CPU sync.

**Sync cost**: Each call to `np.asarray(mx.array)` triggers a GPU->CPU transfer
and sync. Each `mx.array(np_result)` triggers CPU->GPU transfer. For a 128x256
grid with 2 dimensions, that's 4 round-trips per RK substep, 12 per timestep.
At ~50us per sync on M3 Pro, total overhead is ~600us/step, which is ~10-15%
of a typical 4ms timestep.

**Option B: Pure MLX float64 on CPU (no GPU)**

```python
QL_64 = QL.astype(mx.float64)  # forces CPU execution
# ... all HLLD in float64 ...
F = F_64.astype(mx.float32)    # back to GPU-capable type
```

MLX dispatches float64 ops to CPU automatically. No explicit `np.asarray` needed.
But the entire HLLD runs on CPU, losing the GPU parallelism advantage.

**Option C: Mixed-precision Metal kernel (proposed)**

Encode the float64 subset directly in the Metal kernel using the insight from
Section 3. Metal Shading Language does not support `double` on Apple Silicon
GPUs (no hardware FP64). However, we can emulate the critical operations using
**double-float arithmetic** (Dekker 1971, Priest 1991):

```metal
// Double-float: represent x as (hi, lo) where x = hi + lo
// |lo| <= 0.5 * ulp(hi)
struct df { float hi; float lo; };

inline df df_add(df a, df b) {
    float s = a.hi + b.hi;
    float v = s - a.hi;
    float t = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    return {s, t};
}

inline df df_sub(df a, df b) {
    return df_add(a, {-b.hi, -b.lo});
}

inline df df_from(float x) { return {x, 0.0f}; }
```

This gives ~48 bits of mantissa (vs 24 for float32, 53 for float64) at the cost
of 2x the register pressure and ~4-6x the arithmetic ops for promoted operations.

**Cost estimate**: Only the 6 Tier-1 variables use df arithmetic. At 4-6x per
df operation and ~10 operations for the critical path, the overhead is
~50 extra FLOPS per interface out of ~200 total. Expected slowdown: 20-30%
vs pure float32, but 2-3x faster than full CPU float64.

### 4.2 Recommended implementation: Option C (double-float on GPU)

Rationale:
- Keeps everything on GPU (no sync overhead)
- Only Tier-1 operations use df arithmetic (6 values, ~10 subtractions)
- Tier-2 values inherit improved precision from Tier-1 without needing df
- Tier-3 stays native float32
- `mx.compile()` compatibility: the kernel is MSL, not Python -- compile
  doesn't affect it. The kernel is already compiled once and cached.

### 4.3 Alternative: Entropy-augmented HLLD (Option D -- see Section 6)

Instead of double-float arithmetic in the kernel, replace the pressure recovery
with entropy-derived pressure. This eliminates the root cause (Section 1.1)
without any precision change. D_L/D_R cancellation (Section 1.3) would still
need the `bn_small` fallback, but with correct pressure feeding into SM,
the cascade is broken at the root.

---

## 5. Alternative: HLLD with Entropy-Derived Pressure (HLLD-S Hybrid)

### 5.1 Concept

Use entropy for pressure recovery in HLLD star states, conservative E for fluxes.
This is a hybrid that keeps the 4-wave resolution of HLLD while avoiding the
worst cancellation site.

The modification is surgical -- only `cons_to_prim` changes:

```python
# INSTEAD OF (current HLLD, cancellation-prone):
p = (gamma - 1) * (E - KE - ME)

# USE (entropy-derived, cancellation-free):
p = Srho * rho^(gamma-1)
```

Everything downstream (SM, D_L, D_R, star states) receives accurate pressure.

### 5.2 What this fixes

- **Section 1.1 (pressure recovery)**: ELIMINATED. Entropy recovery is purely
  multiplicative -- no catastrophic cancellation possible.
- **Section 1.2 (SM denominator)**: IMPROVED. With correct p -> correct pt ->
  correct SM numerator and denominator. The near-cancellation still exists when
  left/right states are similar, but the inputs are now correct.
- **Section 1.3 (D_L, D_R)**: PARTIALLY IMPROVED. The `rho*(SL-vn)*(SL-SM)`
  term uses wavespeeds from correct pressure. The Bn^2 subtraction is still
  inherent to the algebra. When Bn ~ sqrt(rho*cf^2 - B_perp^2), this is
  unavoidable -- but it only triggers near the switch-on degeneracy, which
  is geometrically rare in DPF (B_theta dominates, B_r is small).
- **Section 1.4-1.6**: IMPROVED by cascade effect from better inputs.

### 5.3 What this doesn't fix

The D_L/D_R cancellation (Section 1.3) is algebraic, not numerical. It occurs
when `rho*(SL-vn)*(SL-SM) ~ Bn^2`, which is a PHYSICAL degeneracy (the slow
and Alfven waves coincide). No amount of precision helps at exact degeneracy --
the HLLD intermediate states simply don't exist there. The correct response is
the `bn_small` branch: fall back to HLL-like behavior where the Alfven wave
vanishes.

In DPF geometry: B_theta >> B_r, B_z in the pinch region. So Bn (the
normal-direction B) is small relative to Bt, and D_L ~ rho*(SL-vn)*(SL-SM)
(large) minus Bn^2 (small). The cancellation is mild. The exception is at the
electrode surfaces where B_r can be comparable to B_theta.

### 5.4 Energy conservation with entropy pressure

Using entropy-derived pressure for STAR STATES while keeping conserved E in
the FLUX computation preserves exact energy conservation. The star-state
pressure only determines the intermediate-state structure (wave speeds, tangential
fields). The actual energy flux is:

```
F_E = (E + pt) * vn - Bn * (v . B)
```

where E comes from the conserved state (or the Rankine-Hugoniot jump relation),
not from the entropy estimate. The HLL formula `F = (SR*FL - SL*FR + SL*SR*(UR-UL))/(SR-SL)`
uses the conserved UR, UL directly.

**Key point**: The entropy pressure only enters through SM and the star-state
structure. The final flux is assembled from conserved quantities. This is
exactly the same separation that DISPATCH uses for HLLS -- entropy for wave
structure, conserved variables for flux.

### 5.5 Implementation cost

Modify `cons_to_prim` in the Metal kernel (6 lines of change):

```metal
// Current (cancellation-prone):
float E  = max(U[4 * stride + idx], P_FLOOR);
float ke = 0.5f * rho * (vn*vn + vt1*vt1 + vt2*vt2);
float mag = 0.5f * (Bn*Bn + Bt1*Bt1 + Bt2*Bt2);
p = max((gamma - 1.0f) * (E - ke - mag), P_FLOOR);

// HLLD-S (cancellation-free):
float Srho = max(U[5 * stride + idx], P_FLOOR);
p = max(Srho * pow(rho, gamma - 1.0f), P_FLOOR);
```

No changes to the star-state algebra, flux assembly, or wave-speed selection.
The physical flux function still uses `E` (conserved total energy) directly.

### 5.6 Risk: Entropy tracer accuracy at shocks

The entropy tracer is passively advected and doesn't satisfy the Rankine-Hugoniot
jump conditions at shocks. After a strong shock, the entropy value is WRONG --
it reflects the pre-shock entropy, not the post-shock entropy.

Our `entropy_resync()` function (mlx_primitives.py:391-471) handles this by
resynchronizing the entropy tracer at detected shocks. The resync uses the
total-energy-derived pressure (which IS correct at shocks in float64 / smooth
regions).

**For HLLD-S**: The entropy tracer at the cell interface (post-reconstruction)
carries information from cells that may include unresync'd shock artifacts.
However:

1. WENO5-Z reconstruction doesn't generate new shocks -- it interpolates.
2. At real shocks, the wave speeds SL, SR bracket the shock, and the HLLD
   flux formula reduces to the standard Rankine-Hugoniot flux regardless of
   the star-state pressure estimate.
3. The star-state pressure only matters in the FAN region (between SL* and SR*),
   which is the smooth part of the solution.

**Assessment**: This is safe for production use. The entropy tracer is accurate
in the FAN region where star-state pressure matters. At shocks, the HLLD flux
formula correctly reduces to the shock jump relation regardless of star-state
details.

---

## 6. Recommended Strategy: Two-Phase Implementation

### Phase 1: HLLD-S Hybrid (low risk, high impact, ~50 LOC)

1. Modify `cons_to_prim` in `_HLLD_HEADER` to use entropy-derived pressure
2. Keep conserved E for physical flux and energy star-state
3. Test on Sod, Brio-Wu, Orszag-Tang, DPF full discharge
4. Verify: energy conservation, shock structure, convergence rate

This eliminates the ROOT CAUSE (Section 1.1) and breaks the cascade at its
origin. Expected result: HLLD runs stable in float32 for DPF simulations
without the CPU float64 fallback.

### Phase 2: Double-Float D_L/D_R (moderate risk, medium impact, ~100 LOC)

Only needed if Phase 1 reveals residual D_L/D_R issues (unlikely for DPF
but possible for problems with strong B_n):

1. Implement `df_sub()` for the `D_L = rho*(SL-vn)*(SL-SM) - Bn^2` computation
2. Use df arithmetic for the ~6 Tier-1 values
3. Test specifically on the switch-on degeneracy (B_n = B_total, B_t = 0)

### Decision criterion for Phase 2

After Phase 1, run the DPF full discharge with HLLD-S at 128x256 resolution.
If the Lax-Friedrichs fallback (`has_nan` branch in the Metal kernel) activates
at more than 0.1% of interfaces in any timestep, proceed to Phase 2.

---

## 7. Expected Impact on DPF Simulations

### 7.1 Where HLLD matters vs HLL

HLLD resolves 7 MHD wave families (2 fast, 2 Alfven, 2 slow, 1 entropy).
HLL resolves only 2 (fast-L and fast-R), smearing everything between them.
The practical difference shows up at:

**Contact discontinuity at sheath front**: The DPF current sheath is a contact
discontinuity separating the swept-up plasma from the fill gas. HLLD resolves
this sharply (1-2 cells). HLL smears it over 5-10 cells. This matters for:
- Accurate sheath mass calculation (which feeds back to circuit coupling)
- Neutron yield estimation (depends on pinch density, which depends on
  how much mass the sheath sweeps)
- Instability seeding (sharp sheath = correct Rayleigh-Taylor growth rate)

**Rotational discontinuity at pinch**: The Z-pinch column has a rotational
discontinuity where B_theta changes direction. HLLD resolves the Alfven waves
that maintain this structure. HLL smears the rotational discontinuity, causing
artificial reconnection and premature pinch disruption.

**Slow shocks in the pinch column**: During radial compression, slow MHD shocks
form behind the sheath. HLLD resolves these separately from the contact
discontinuity. HLL merges them into a single smeared structure.

### 7.2 When HLL is sufficient

- **Axial acceleration phase** (before radial compression): Flow is largely 1D
  along z-axis. No transverse B-field discontinuities. HLL with PLM is adequate.
- **Post-pinch expansion**: Flow is smooth and expanding. No strong
  discontinuities. Even Lax-Friedrichs would work.
- **Vacuum regions**: No waves to resolve. HLL + Boris correction bounds
  wave speeds without D_L/D_R issues.

### 7.3 Quantitative impact estimate

Based on Phase R validation (32 commits, 75 CI tests):
- HLL+PLM at 128x256: I_peak error ~6-8% vs experiment, ~15% L1(rho) on Sod
- HLLD(float64)+WENO5Z: I_peak error ~3-5%, ~5% L1(rho) on Sod
- HLLD-S(float32)+WENO5Z (projected): Same accuracy as float64 HLLD, 2-3x
  faster due to GPU execution

The HLLD advantage is primarily in the radial compression phase (pinch),
where the sheath sharpness directly affects the circuit coupling through
plasma inductance Lp.

---

## 8. Implementation Checklist (for future sprint)

- [ ] Modify `cons_to_prim` in `_HLLD_HEADER` (Metal kernel) to read ISR slot
- [ ] Pass ISR slot index to Metal kernel input buffer (currently only reads E)
- [ ] Update `hlld_flux_numpy` reference to use entropy pressure
- [ ] Add test: HLLD-S vs HLLD(float64) on Brio-Wu, L1 error < 1%
- [ ] Add test: HLLD-S energy conservation on Orszag-Tang, dE/E < 1e-6
- [ ] Add test: DPF full discharge HLLD-S, no NaN fallback activation
- [ ] Benchmark: HLLD-S(float32 GPU) vs HLLD(float64 CPU) timing
- [ ] Update `compute_fluxes()` to accept `riemann="hlld_s"` option
- [ ] Document in CLAUDE.md lessons learned section

---

## References

1. Miyoshi, T., Kusano, K. (2005). "A multi-state HLL approximate Riemann
   solver for ideal magnetohydrodynamics." JCP 208:315-344.

2. Popovas, A. et al. (2025). "DISPATCH methods: An approximate,
   entropy-based Riemann solver for ideal MHD." A&A 694. arXiv:2211.02438.

3. Stone, J.M. et al. (2020). "The Athena++ Adaptive Mesh Refinement
   Framework: Design and Magnetohydrodynamic Solvers." ApJS 249:4.

4. Bryan, G.L. et al. (2014). "ENZO: An Adaptive Mesh Refinement Code for
   Astrophysics." ApJS 211:19. (Dual-energy formalism.)

5. Borges, R. et al. (2008). "An improved weighted essentially non-oscillatory
   scheme for hyperbolic conservation laws." JCP 227:3191-3211.

6. Gombosi, T.I. et al. (2002). "Semirelativistic magnetohydrodynamics and
   physics-based convergence acceleration." JCP 177:176-205. (Boris correction.)

7. Dekker, T.J. (1971). "A floating-point technique for extending the
   available precision." Numerische Mathematik 18:224-242. (Double-float.)

8. Priest, D.M. (1991). "Algorithms for arbitrary precision floating point
   arithmetic." Proc. 10th IEEE Symposium on Computer Arithmetic, pp. 132-143.

9. Minoshima, T. et al. (2019). "A Boris correction in the MHD solver."
   ApJ 874:37. (Boris-corrected HLLD.)
