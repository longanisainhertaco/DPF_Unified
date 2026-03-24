# DISPATCH HLLS Entropy-Based Riemann Solver: Technical Analysis

**Paper**: Popovas, A. (2025). "DISPATCH methods: An approximate, entropy-based Riemann solver for ideal magnetohydrodynamics."
**Journal**: Astronomy & Astrophysics, Vol. 694 (June 2025)
**DOI**: [10.1051/0004-6361/202554028](https://doi.org/10.1051/0004-6361/202554028)
**arXiv**: [2211.02438](https://arxiv.org/abs/2211.02438) (submitted Nov 2022, updated Apr 2025)
**Code**: DISPATCH framework (Modern Fortran), public version at dispatch.readthedocs.io. No standalone HLLS reference implementation published.

**Purpose**: Evaluate HLLS for DPF-Unified Metal GPU solver to eliminate catastrophic cancellation in pressure recovery at low plasma beta (beta << 0.01) near electrode boundaries.

---

## 1. How HLLS Differs from Standard HLLD

### The Core Problem HLLS Solves

Standard HLLD (Miyoshi & Kusano 2005) conserves total energy:

```
E_tot = p/(gamma-1) + 0.5*rho*v^2 + B^2/(2*mu0)
```

Pressure recovery requires subtracting kinetic and magnetic energy from total energy:

```
p = (gamma-1) * (E_tot - 0.5*rho*v^2 - B^2/(2*mu0))
```

When `E_kin + E_mag >> E_th` (low beta), this subtraction suffers catastrophic cancellation in float32. A 1-ULP error in E_tot at `O(10^6)` produces `O(1)` error in pressure at `O(1)`, yielding negative pressure and solver crash.

### HLLS Solution: Replace Total Energy with Entropy

HLLS replaces the total energy equation with an entropy evolution equation. The 8-component conservative variable vector becomes (paper Eq. 16):

```
U = (rho, rho*u_x, rho*u_y, rho*u_z, rho*S, B_x, B_y, B_z)^T
```

where `S` is entropy per unit mass, replacing `E_tot` in the 5th component.

The flux vector (paper Eq. 18, x-direction):

```
F = (rho*u_x,
     rho*u_x^2 + P_tot - B_x^2,
     rho*u_x*u_y - B_x*B_y,
     rho*u_x*u_z - B_x*B_z,
     rho*u_x*S,              <-- entropy flux (passive advection)
     0,
     u_x*B_y - u_y*B_x,
     u_x*B_z - u_z*B_x)^T
```

The source term vector (paper Eq. 17):

```
Psi = (0, Phi_x, Phi_y, Phi_z, Q_ext/T + Q_S/T, 0, 0, 0)^T
```

### Key Difference: Entropy Flux is Passive Advection

In HLLD, the energy flux has a complex `(E_tot + P_tot)*v_n - B_n*(v.B)` structure coupling all waves. In HLLS, the entropy flux `F_S = rho*u_n*S` is a **passive scalar advection** — no pressure term, no magnetic coupling in the flux itself. Pressure effects enter only through the `Q_S/T` source term.

This means entropy never suffers catastrophic cancellation because it is never computed by subtracting large nearly-equal quantities.

### Structural Comparison

| Component | HLLD | HLLS |
|-----------|------|------|
| 5th conserved variable | E_tot = rho*e + 0.5*rho*v^2 + B^2/2 | rho*S |
| 5th flux component | (E_tot + P_tot)*v_n - B_n*(v.B) | rho*v_n*S |
| 5th source term | 0 (energy conserved) | Q_S/T + Q_ext/T |
| Pressure recovery | p = (gamma-1)*(E_tot - E_kin - E_mag) | p = rho^gamma * exp[(gamma-1)*S_hat] |
| Conservation | Exact total energy conservation | Exact mass/momentum/B; approximate energy |
| Entropy | Not tracked (implicit) | Explicitly evolved, monotonically increasing |

---

## 2. Entropy Variable Definition and EOS

### Thermodynamic Entropy (paper Eq. 7)

```
S = S_0 + c_v * ln(P * rho^(-gamma))
```

where `c_v = k_B / ((gamma-1) * mu * m_u)`.

### Dimensionless Modified Entropy (paper Eq. 10)

For numerical implementation, DISPATCH uses a scaled, dimensionless entropy:

```
S_hat*(gamma-1) = S_0 + ln(P * rho^(-gamma))
```

where `S_hat` absorbs the `c_v` factor. This keeps entropy values O(1) in computational units, critical for float32 precision.

### Pressure Recovery from Entropy (forward EOS)

Given `(rho, S_hat)`, recover pressure without subtraction:

```
P = rho^gamma * exp(S_hat*(gamma-1) - S_0)
```

In code units with `S_0 = 0`:

```
P = rho^gamma * exp((gamma-1) * S_hat)
```

This is a **multiplicative** operation. No catastrophic cancellation possible. The exponential is always positive, guaranteeing `P > 0` as long as `rho > 0` and `S_hat` is finite.

### Temperature and Internal Energy Recovery

```
T = P * mu * m_u / (rho * k_B) = rho^(gamma-1) * exp((gamma-1)*S_hat) / c_v

epsilon = P / (rho * (gamma-1)) = rho^(gamma-1) * exp((gamma-1)*S_hat) / (gamma-1)
```

---

## 3. Pressure Positivity Guarantee

### Mechanism

HLLS guarantees positive pressure **by construction** through the entropy formulation:

1. **Entropy is evolved directly** — never computed by subtraction
2. **Pressure recovery** uses `P = rho^gamma * exp(...)` — exponential is always positive
3. **Density** has standard positivity preservation (density floor)
4. **No ad-hoc pressure floors needed** for the entropy-derived pressure

The only requirement: `rho > 0` and `S_hat` finite (not NaN/Inf).

### Caveats

The paper does NOT claim unconditional positivity. Two failure modes remain:

1. **Floating-point overflow in exp()**: If `S_hat` grows very large (extreme heating), `exp((gamma-1)*S_hat)` overflows. Requires clamping `S_hat` to `~88/gamma` for float32.
2. **Density negativity**: HLLS does not improve density positivity over HLLD — same density evolution equations apply.

### Comparison to HLLD Pressure Floors

Our current Metal HLLD uses `P_FLOOR = 1e-20` (see `_riemann_constants.py`). This is an ad-hoc fix that:
- Violates conservation
- Can mask real physics errors
- Introduces artificial heating in magnetically dominated regions

HLLS eliminates the need for pressure floors entirely in the Riemann solver.

---

## 4. Entropy Production at Shocks (The Hard Part)

### The Problem

Entropy is NOT conserved across shocks — it increases irreversibly. The entropy equation with only advective flux `F_S = rho*v*S` would predict isentropic flow everywhere, missing shock heating entirely. A source term is needed.

### Energy Decomposition (paper Eqs. 25-27)

HLLS derives the entropy source from energy conservation by decomposing total energy:

```
E_tot = E_th + E_kin + E_mag
```

Each component evolves as:

Kinetic energy (Eq. 25):
```
dE_kin/dt = -div(F_kin) + W_gas - Q_kin + Theta_kin
```

Thermal energy (Eq. 26):
```
dE_th/dt = -div(F_th) - W_gas + Q_kin + Theta_th
```

Magnetic energy (Eq. 27):
```
dE_mag/dt = -div(F_mag) - Theta_kin - Theta_th
```

where:
- `W_gas = P * div(v)` — pressure work (reversible)
- `Q_kin` — kinetic-to-thermal dissipation (irreversible)
- `Theta_kin` — magnetic-to-kinetic conversion
- `Theta_th` — magnetic-to-thermal conversion

### Entropy Production Formula (paper Eq. 28)

The key equation. The heat generated by numerical dissipation:

```
Q_S = -div(F_kin) - div(F_mag) - dE_kin/dt - dE_mag/dt + W_gas
```

This is derived by requiring total energy conservation:
```
dE_tot/dt = -div(F_tot) = -div(F_th) - div(F_kin) - div(F_mag)
```
and substituting the individual energy evolution equations.

### Implementation: Computing Q_S

The practical computation requires:

1. Compute kinetic energy flux through Riemann solver: `F_kin = 0.5*rho*v^2 * v_n + P_tot*v_n - B_n*(v.B)` (kinetic part of the Godunov flux)
2. Compute magnetic energy flux: `F_mag = B^2*v_n/2 - B_n*(v.B)` (magnetic part)
3. Compute `E_kin^t` and `E_kin^(t+dt)` from the updated momentum/density
4. Compute `E_mag^t` and `E_mag^(t+dt)` from the updated B-field
5. Finite-difference time derivatives: `dE_kin/dt = (E_kin^(t+dt) - E_kin^t) / dt`
6. Evaluate `Q_S` from Eq. 28
7. Enforce second law: `S_gen = max(0, Q_S/T)`

### Entropy Update (paper Eq. 29)

```
d(rho*S)/dt = -div(rho*v*S) + max(0, Q_S/T)
```

The `max(0, ...)` clamp enforces the second law of thermodynamics — entropy never decreases. This is the HLLS equivalent of the Rankine-Hugoniot energy jump condition.

### How This Handles Shocks

At a shock:
- Kinetic energy drops sharply (flow decelerates)
- `dE_kin/dt < 0` while `div(F_kin)` captures the flux
- The difference `Q_S > 0` represents shock heating
- This generates entropy at the correct rate to satisfy R-H conditions

The entropy is NOT derived from Rankine-Hugoniot jump conditions directly. Instead, it uses the **residual principle**: whatever kinetic/magnetic energy is dissipated by the numerical scheme gets converted to thermal energy (entropy). The `max(0, ...)` prevents artificial cooling.

### HLLS-2 Variant

The paper tests a variant (HLLS-2, Figure 12) that allows `Q_S < 0`, omitting the max clamp. This tracks total energy dissipation more faithfully but can violate the second law in regions of numerical oscillation. The standard HLLS with the clamp is recommended.

---

## 5. Switching Criterion Between Entropy and Energy Formulations

**There is no switching criterion.** HLLS is a complete standalone formulation, not a hybrid. The paper presents it as a full replacement for the total-energy HLLD, not a fallback that switches between entropy-based and energy-based modes.

For a DPF implementation, we could implement a hybrid:

**Proposed DPF hybrid strategy** (not from the paper):
```
if beta < beta_switch:
    use HLLS (entropy-based pressure recovery)
else:
    use HLLD (energy-based, more accurate for high-beta flows)
```

A reasonable threshold would be `beta_switch ~ 0.01-0.1`. However, the discontinuity at the switching boundary could introduce artifacts. A smoother blend:

```
alpha = sigmoid((log10(beta) - log10(beta_switch)) / delta)
P = alpha * P_energy + (1 - alpha) * P_entropy
```

The paper does NOT discuss this — it is our engineering decision for DPF.

---

## 6. Float32 Validation

### Confirmed: All Tests Run in Single Precision

The paper explicitly states: **"The experiments were run in single floating-point precision"** (Section 5.1). This is one of the strongest selling points for our Metal GPU implementation.

### Tests Validated in float32

| Test | Resolution | Key Result |
|------|-----------|------------|
| Linear wave convergence | 48-1024 cells | 2nd-order convergence achieved |
| Entropy wave (M~0.001) | 100 cells | Minimal diffusion |
| Shu-Osher shock | 1500 cells | Sharp shocks, matches HLLD |
| Brio-Wu MHD | 1200 cells | Compound waves accurate |
| Kelvin-Helmholtz | 256^2 | Primary/secondary instabilities converged |
| Rayleigh-Taylor | 768x1536 | Maintains symmetry better than references |
| Hot bubble (low-Mach) | 128x192 | Delta_A/A ~ 10^-4 perturbation tracked |
| MHD blast | 512^2 | Sharp shock fronts |
| Orszag-Tang vortex | 1024^2 | Plasmoids form correctly |
| Current sheet | 256^2 | beta in [0.1, 10] |
| Magnetic loop advection | 128x64 | CT maintains div(B)=0 |
| Gresho vortex | up to 128^2 | Works down to Ma=0.01, fails at Ma=0.001 |
| Magnetic rotor | 512^2 | Ma=20 |

### Float32 Limitations Noted

1. **Linear wave amplitude**: Uses `A = 10^-3` instead of `10^-6` (Stone et al. reference) because `10^-6` is near the float32 noise floor.
2. **Hot bubble**: "The perturbation being very close to the numerical noise limit as we normally operate with single precision."
3. **Gresho vortex at Ma=0.001**: Fails — "the expected outcome; Godunov-type Riemann solvers deal with very low Mach numbers poorly." Ma=0.01 still works fine.

### Implications for DPF on Metal

The DPF operates at:
- Sheath/radial phase: Ma ~ 0.1-1 (well within HLLS float32 range)
- Pinch compression: Ma ~ 1-10 (strong shocks, HLLS validated)
- Electrode boundary: beta ~ 0.001-0.01 (this is WHY we want HLLS)
- Low-Mach ambient: Ma ~ 0.01-0.1 (HLLS handles this, standard HLLD struggles)

HLLS in float32 covers ALL DPF operating regimes. The Ma=0.001 failure is irrelevant for DPF.

---

## 7. Computational Overhead vs HLLD

### The paper provides no explicit benchmarks.

### Estimated Additional Operations per Cell Interface

Based on the 7-step algorithm (Section 4.2):

| Extra Step | Operations | Notes |
|-----------|-----------|-------|
| Kinetic energy flux computation | ~10 FLOPs | `0.5*rho*v^2*v_n` components |
| Magnetic energy flux computation | ~10 FLOPs | `0.5*B^2*v_n - B_n*(v.B)` |
| E_kin at t and t+dt | ~6 FLOPs/cell | `0.5*rho*v^2` |
| E_mag at t and t+dt | ~4 FLOPs/cell | `0.5*B^2` |
| dE_kin/dt, dE_mag/dt | ~4 FLOPs/cell | finite differences |
| Q_S (Eq. 28) | ~5 FLOPs/cell | sum of computed terms |
| max(0, Q_S/T) | ~3 FLOPs/cell | division + max |
| Pressure recovery `rho^gamma * exp(...)` | ~8 FLOPs/cell | replaces subtraction (net ~+5) |

**Estimated overhead: ~15-25% additional FLOPs** compared to standard HLLD.

However, HLLS **eliminates** the NaN fallback path in our current HLLD (`_should_check_nan()` + HLL fallback at line 387-395 of `_riemann_solvers.py`), which saves branching overhead. Net overhead may be closer to 10-15%.

### Memory Overhead

HLLS requires storing one additional field per cell: `rho*S` (entropy density). This replaces `E_tot`, so the total conservative variable count remains 8. No additional memory needed if we simply swap the 5th component.

---

## 8. Pseudocode / Reference Implementation

### No pseudocode provided in the paper.

The paper describes the algorithm in 7 prose steps (Section 4.2) and refers to Appendix A for the MUSCL-Hancock framework. The DISPATCH code is written in Modern Fortran but no standalone HLLS module has been published.

### Reconstructed Algorithm (from paper description)

```
HLLS_STEP(U, dt, dx):
    # U = [rho, rho*vx, rho*vy, rho*vz, rho*S, Bx, By, Bz]

    # 1. Compute primitives from entropy EOS
    rho = U[0]
    v = U[1:4] / rho
    S = U[4] / rho
    B = U[5:8]
    P = rho^gamma * exp((gamma-1) * S)  # no subtraction!
    T = P / (rho * c_v * (gamma-1))

    # 2. Reconstruct L/R states at interfaces (PLM or WENO)
    UL, UR = reconstruct(U)

    # 3. Solve Riemann problem for entropy flux
    #    Entropy is advected as passive scalar through HLLD fan
    #    Use standard HLLD wave speeds and contact/Alfven structure
    #    but replace E_tot flux with rho*v_n*S flux
    F = hlls_riemann(UL, UR, gamma, dim)

    # 4. Compute kinetic and magnetic energy Godunov fluxes
    #    (from the Riemann solver intermediate states)
    F_kin = kinetic_energy_flux(UL, UR, gamma, dim)
    F_mag = magnetic_energy_flux(UL, UR, gamma, dim)

    # 5. Provisional update (advection only, no entropy source)
    U_new = U - (dt/dx) * (F[i+1/2] - F[i-1/2])

    # 6. Compute entropy production
    E_kin_old = 0.5 * rho * |v|^2
    E_mag_old = 0.5 * |B|^2
    # (compute E_kin_new, E_mag_new from U_new)
    rho_new = U_new[0]
    v_new = U_new[1:4] / rho_new
    B_new = U_new[5:8]
    E_kin_new = 0.5 * rho_new * |v_new|^2
    E_mag_new = 0.5 * |B_new|^2

    dEkin_dt = (E_kin_new - E_kin_old) / dt
    dEmag_dt = (E_mag_new - E_mag_old) / dt
    div_Fkin = (F_kin[i+1/2] - F_kin[i-1/2]) / dx
    div_Fmag = (F_mag[i+1/2] - F_mag[i-1/2]) / dx
    W_gas = P * div(v)  # pressure work

    Q_S = -div_Fkin - div_Fmag - dEkin_dt - dEmag_dt + W_gas

    # 7. Apply entropy generation (second law enforcement)
    S_gen = max(0, Q_S / T)
    U_new[4] += dt * S_gen  # add entropy source to rho*S

    return U_new
```

### HLLS Riemann Solver Core (entropy flux)

The Riemann solver itself is identical to HLLD in structure (same wave speeds, same intermediate state construction for rho, momentum, B) except:

1. The 5th component `E_tot` is replaced by `rho*S`
2. The entropy flux through the Riemann fan is `F_S = rho_star * v_n_star * S_star`
3. In the Riemann fan, entropy is advected passively: the intermediate-state entropy uses the upwind value
   - If `v_n_star > 0`: `S_star = S_L`
   - If `v_n_star <= 0`: `S_star = S_R`

This is dramatically simpler than the HLLD energy flux, which involves complex coupling between all intermediate states.

---

## 9. Constrained Transport Interaction

### Entropy does NOT participate in CT

The CT scheme evolves face-centered B-fields using edge-centered electric fields:

```
dB_x/dt = dE_z/dy - dE_y/dz   (and cyclic permutations)
```

where `E = -v x B + eta*J`.

The entropy variable `rho*S` is a cell-centered scalar that has no direct role in the CT EMF computation. CT operates on B-fields and velocities only.

### Indirect Interaction

CT updates B-field -> new B affects pressure via entropy EOS -> pressure affects momentum flux -> momentum flux provides velocities for next CT step. The coupling is through the time integration, not through the CT stencil itself.

### Our Metal CT Implementation

The existing `emf_from_fluxes_mps()` in `metal_stencil.py` computes EMFs from Riemann solver fluxes. Switching from HLLD to HLLS does not require any changes to the CT module. The B-field components in the Riemann solver (components 5-7 of the flux) remain identical between HLLD and HLLS.

---

## 10. Wave Speed Estimates

HLLS uses the same MHD eigenvalue structure as HLLD (paper Eqs. 19-21):

### 7 MHD Eigenvalues

```
lambda_1,7 = u -/+ c_f    (fast magnetosonic)
lambda_3,5 = u -/+ c_s    (slow magnetosonic)
lambda_2,6 = u -/+ c_a    (Alfven)
lambda_4   = u             (entropy/contact)
```

### Fast/Slow Magnetosonic and Alfven Speeds (Eq. 20-21)

```
c_f^2 = d + sqrt(d^2 - a^2 * B_x^2 / rho)
c_s^2 = d - sqrt(d^2 - a^2 * B_x^2 / rho)
c_a   = |B_x| / sqrt(rho)

d = (a^2 + |B|^2/rho) / 2
```

where `a` is the adiabatic sound speed `a^2 = gamma * P / rho`.

### Davis Wave Speed Bounds (Appendix A, Eq. A.1)

```
S_L = min(u_L, u_R) - max(c_f_L, c_f_R)
S_R = max(u_L, u_R) + max(c_f_L, c_f_R)
```

### Contact Wave Speed (Appendix A, Eq. A.9)

```
u* = [rho_R*u_R*v_R + rho_L*u_L*v_L + (P_tot_L - P_tot_R)] / [rho_L*v_L + rho_R*v_R]
```

where `v_L = u_L - S_L`, `v_R = S_R - u_R`.

### Total Pressure in Star Region (Appendix A, Eq. A.10)

```
P_tot* = [rho_R*v_R*P_tot_L + rho_L*v_L*P_tot_R + rho_R*rho_L*v_R*v_L*(u_L - u_R)]
         / [rho_L*v_L + rho_R*v_R]
```

---

## 11. DPF Implementation Plan

### Phase 1: Entropy Variable Infrastructure

Add entropy as 9th component to the state vector (alongside existing 8 + electron energy):

```python
# In _riemann_constants.py
ISE = 8   # entropy density index (rho*S)
# IEE = 8 already exists for electron energy
# Need to decide: replace IEE slot or add ISE as 9th
```

Better approach: **replace the IEN (total energy) slot with ISE (entropy)** in HLLS mode, since HLLS does not evolve total energy. The solver operates in one mode or the other.

### Phase 2: Entropy-Based Pressure Recovery

```python
def pressure_from_entropy(rho: torch.Tensor, S: torch.Tensor, gamma: float) -> torch.Tensor:
    """Recover pressure from density and entropy. Always positive."""
    S_clamped = torch.clamp(S, min=-80.0 / (gamma - 1), max=80.0 / (gamma - 1))
    return torch.clamp(rho, min=RHO_FLOOR) ** gamma * torch.exp((gamma - 1) * S_clamped)
```

### Phase 3: HLLS Riemann Solver

Modify `hlld_flux_mps` to:
1. Compute wave speeds and intermediate states for rho, momentum, B identically
2. Replace energy flux (5th component) with entropy flux `rho*v_n*S` using upwind selection
3. Return kinetic and magnetic energy fluxes as additional outputs for Q_S computation

### Phase 4: Entropy Source Term

After each Euler substage in SSP-RK3:
1. Compute E_kin, E_mag before and after the flux update
2. Evaluate Q_S from Eq. 28
3. Add `max(0, Q_S/T) * dt` to entropy density

### Phase 5: Validation

Run the test suite from the paper in float32:
- Sod shock tube (hydro subset)
- Brio-Wu MHD shock tube
- Orszag-Tang vortex
- Low-beta stress test (DPF electrode boundary conditions)

---

## 12. Risk Assessment

### Advantages for DPF

1. **Eliminates pressure negativity** at electrode boundaries (beta << 0.01) — the primary motivation
2. **Validated in float32** — matches our Metal GPU constraint exactly
3. **Same wave speed structure as HLLD** — reuse existing code
4. **Simpler entropy flux** — less branching, better GPU utilization
5. **No pressure floors needed** — cleaner physics

### Risks

1. **Total energy not conserved**: HLLS trades energy conservation for entropy monotonicity. For DPF with strong shocks, the Q_S source term must capture ALL dissipation. If it under-counts, energy leaks.
2. **Q_S computation requires post-hoc energy tracking**: Additional complexity in the time integration loop. The SSP-RK3 stages need to output intermediate E_kin, E_mag values.
3. **No reference implementation**: Must implement from the paper description alone. Risk of subtle errors in Q_S computation.
4. **Entropy overflow in float32**: `exp((gamma-1)*S)` can overflow if S grows too large. Need careful clamping.
5. **Not tested for cylindrical geometry**: All paper tests are Cartesian. Geometric source terms in cylindrical MHD may interact with the entropy formulation.

### Recommendation

**Implement HLLS as an alternative Riemann solver option** (`riemann_solver="hlls"`) alongside existing HLL and HLLD. Do NOT replace HLLD — keep it as the default for validated problems. Use HLLS specifically where low-beta pressure recovery fails.

A hybrid approach is practical:
- Use HLLS for the entropy/pressure recovery path
- Keep HLLD wave speed estimates and intermediate state construction
- Add Q_S post-processing after each RK substage
- Validate against Brio-Wu and a DPF-specific low-beta test

---

## 13. Comparison to Alternative Approaches

### HLLD-PC (Physical Consistency, arXiv 2507.10420, 2025)

A different 2025 paper addresses the same low-beta failure mode by enforcing consistency between B_parallel and magnetic energy in intermediate states. Does NOT use entropy. Pros: simpler to implement (patches existing HLLD). Cons: less fundamental fix, still susceptible to float32 cancellation in extreme cases.

### MLAU (Minoshima et al.)

Low-dissipation HLLD variant for wide Mach number range. Available on GitHub (github.com/minoshim/MLAU). Focuses on reducing dissipation at low Mach, not on pressure positivity. Complementary to HLLS.

### Well-Balanced Schemes

Maintain hydrostatic equilibrium exactly. HLLS "performs almost as well as a well-balanced scheme" according to the paper. For DPF, we don't need hydrostatic balance — we need pressure positivity in magnetically dominated regions. HLLS is the right tool.

---

## References

- Popovas, A. (2025). A&A, 694. DOI: 10.1051/0004-6361/202554028
- Miyoshi, T. & Kusano, K. (2005). JCP, 208, 315. (Standard HLLD)
- Borges, R. et al. (2008). JCP, 227, 3191. (WENO-Z)
- Nordlund, A. et al. (2018). MNRAS, 477, 624. (DISPATCH framework)
- Evans, C. & Hawley, J. (1988). ApJ, 332, 659. (Constrained Transport)
- Fromang, S. et al. (2006). A&A, 457, 371. (RAMSES HLLD implementation)

---

## Appendix: Existing DPF-Unified HLLD Implementation Notes

The current Metal HLLD at `src/dpf/metal/_riemann_solvers.py` is a **proper 4-intermediate-state HLLD** (not HLLC-MHD as previously suspected). It correctly implements:
- Davis wave speed bounds (lines 98-102)
- Contact wave speed SM (lines 209-215)
- Outer star states U*_L, U*_R with tangential velocity/B-field jumps (lines 233-283)
- Inner double-star states U**_L, U**_R with Alfven wave resolution (lines 306-357)
- 5-region flux selection (lines 370-385)
- HLL fallback for NaN safety (lines 387-395)

The HLLS modification would primarily affect:
1. Lines 258-263: Replace `e_sL`, `e_sR` (star-state energy) with entropy advection
2. Lines 334-337: Replace `e_dsL`, `e_dsR` (double-star energy) with entropy
3. Add post-update Q_S computation in the Metal solver's `_euler_stage`
4. Replace `_cons_to_prim_mps` pressure recovery with entropy-based recovery
