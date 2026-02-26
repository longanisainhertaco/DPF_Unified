# CESE-MHD Feasibility Report for DPF-Unified

**Date**: 2026-02-26
**Researcher**: CESE-MHD Research Agent (dpf-mhd-physicist)
**Task**: #3 — Research 4th-order CESE method for DPF cylindrical MHD
**Status**: Complete

---

## Executive Summary

This report provides a deep technical analysis of the Conservation Element and Solution Element (CESE) method for MHD, evaluating its suitability as an alternative numerical engine for DPF-Unified's cylindrical Z-pinch simulations. The analysis covers the mathematical formulation, 4th-order extension, cylindrical geometry handling, resistive MHD capability, implementation roadmap, and honest comparison with our current WENO5+HLLD+SSP-RK3 approach.

**Bottom-line recommendation**: **Prototype a 1D CESE Euler solver** (~400 LOC) to build institutional knowledge, but **defer full CESE-MHD implementation** until either (a) Jiang & Zhang release their 4th-order code, or (b) we have exhausted improvements to our current WENO5+HLLD pipeline. The risk-to-reward ratio does not justify a full rewrite now.

---

## 1. The CESE Method: Mathematical Formulation

### 1.1 Governing Equation

Consider a system of conservation laws in d spatial dimensions:

```
∂u/∂t + ∂f₁(u)/∂x₁ + ∂f₂(u)/∂x₂ + ... + ∂fₐ(u)/∂xₐ = 0
```

where **u** ∈ ℝᵐ is the vector of conserved variables and **fᵢ(u)** are the flux functions.

The CESE method treats this in (d+1)-dimensional space-time. Define the space-time flux vector:

```
h = (f₁, f₂, ..., fₐ, u)  ∈ ℝ^(d+1)
```

Then the conservation law is equivalent to:

```
∇_(x,t) · h = 0
```

and in integral form over any space-time volume V with boundary S:

```
∮_S h · n̂ dS = 0                                    (1)
```

This is the **starting point** for CESE — not the differential form.

### 1.2 Conservation Elements (CEs) and Solution Elements (SEs)

**Conservation Element (CE)**: A non-overlapping space-time volume over which the integral conservation law (1) is enforced. In the 1D case on a staggered mesh:

```
CE_j^(n+1/2) = [x_{j-1/2}, x_{j+1/2}] × [t_n, t_{n+1}]    (half-step CE)
```

The CEs tile the entire space-time domain without overlap, ensuring **global conservation** by construction.

**Solution Element (SE)**: A collection of line segments in the (x,t) plane associated with each mesh point (x_j, t_n), through which fluxes are evaluated. In the 2nd-order scheme:

```
SE_j^n = { (x, t_n) : x ∈ [x_{j-1/2}, x_{j+1/2}] }      (horizontal segment)
         ∪ { (x_j, t) : t ∈ [t_{n-1/2}, t_{n+1/2}] }      (vertical segment)
```

The key insight: the SE surfaces form the **boundaries** of the CEs. By computing fluxes along SE boundaries, we enforce conservation on CEs.

### 1.3 Solution Representation via Taylor Expansion

At each mesh point (x_j, t_n), the solution is represented by a Taylor expansion. For the **2nd-order** scheme:

```
u(x, t) ≈ u_j^n + (u_x)_j^n · (x - x_j) + (u_t)_j^n · (t - t_n)       (2)
```

where:
- `u_j^n` = solution value at mesh point
- `(u_x)_j^n` = spatial derivative (independent unknown)
- `(u_t)_j^n` = temporal derivative (computed via Cauchy-Kowalewski: `u_t = -f_x = -A·u_x`, where A = ∂f/∂u)

**Critical difference from FV/FD methods**: Both `u` and `u_x` are treated as **independent marching unknowns**. The solution at each point carries both the value and its gradient. This is what enables the compact stencil.

### 1.4 Marching Equations (2nd-Order, 1D Scalar)

For `∂u/∂t + ∂f(u)/∂x = 0` with `f = au` (linear advection, wave speed a):

**Step 1 — Compute u at the new time level** by enforcing flux conservation on the CE:

```
u_j^(n+1/2) = (1/2)(u_{j-1/2}^n + u_{j+1/2}^n)
             + (1/2)(Δx/2)[(u_x)_{j-1/2}^n - (u_x)_{j+1/2}^n]
             + (Δt/Δx)[f_{j-1/2}^n - f_{j+1/2}^n]                        (3)
             + (Δt/2)[(f_x)_{j-1/2}^n + (f_x)_{j+1/2}^n]
```

where the fluxes f and f_x are evaluated from the Taylor expansion at neighboring points.

**Step 2 — Compute u_x at the new time level** using finite differences of the updated solution:

```
(u_x)_j^(n+1/2) = [u_{j+1}^(n+1/2) - u_{j-1}^(n+1/2)] / (2·Δx)        (4)
```

(In the dissipative scheme, a weighted average is used instead of centered differences.)

The staggered mesh means: at time level n, data lives at integer indices j; at time level n+1/2, data lives at half-integer indices j+1/2. Each half-step uses only the **nearest neighbors** from the previous half-step.

### 1.5 Dissipation Mechanism

The non-dissipative core scheme (Eq. 3-4) is neutrally stable. To handle shocks, **numerical dissipation** is added via a weighted average in the u_x update:

```
(u_x)_j^(n+1/2) = (1/2)[(u_x)_{j-1/2}^n + (u_x)_{j+1/2}^n]  · W       (5)
                 + [(u_{j+1/2}^n - u_{j-1/2}^n) / Δx]          · (1-W)
```

where W ∈ [0, 1] is a blending parameter:
- W = 0: purely finite-difference gradient → maximum dissipation
- W = 1: average of transported gradients → zero dissipation (non-dissipative core)

An automatic shock-sensing mechanism adjusts W locally:

```
W_j = α / (α + |Δu_x|/|u_x|_max)                                        (6)
```

where α is a tuning parameter (typically α ∈ [1, 3]) and |Δu_x| measures the jump in spatial derivatives (a shock indicator).

### 1.6 CFL Condition

The CESE method has a CFL condition:

```
ν = |a| · Δt / Δx ≤ 1                                                    (7)
```

**Crucially, this CFL limit does not degrade with increasing order of accuracy.** Both the 2nd-order and 4th-order CESE schemes have the same CFL ≤ 1 constraint. This is a significant advantage over DG methods where the CFL limit scales as ~1/(2p+1) for order p.

However, the CESE scheme can become **excessively dissipative** for small CFL numbers (ν << 1). Best accuracy is achieved near ν ≈ 0.5-0.9. CFL-insensitive variants exist but add complexity.

---

## 2. The 4th-Order CESE Extension

### 2.1 Key Idea: Higher-Order Taylor Expansion

The 4th-order CESE scheme (Bilyeu 2014, Jiang & Zhang 2025) extends the Taylor expansion to 3rd order:

```
u(x, t) ≈ u_j^n
         + (u_x)_j^n · (x - x_j)  + (u_t)_j^n · (t - t_n)
         + (1/2)(u_xx)_j^n · (x - x_j)²  + (u_xt)_j^n · (x-x_j)(t-t_n)  + (1/2)(u_tt)_j^n · (t-t_n)²
         + (1/6)(u_xxx)_j^n · (x-x_j)³ + (1/2)(u_xxt)_j^n · (x-x_j)²(t-t_n) + ...         (8)
```

The independent marching unknowns are now:

| Order | 2nd-order CESE | 4th-order CESE |
|-------|---------------|----------------|
| Unknowns per point (1D) | u, u_x (2) | u, u_x, u_xx, u_xxx (4) |
| Unknowns per point (2D) | u, u_x, u_y (3) | u, u_x, u_y, u_xx, u_xy, u_yy, u_xxx, u_xxy, u_xyy, u_yyy (10) |
| Unknowns per point (3D) | u, u_x, u_y, u_z (4) | u + 3 + 6 + 10 = 20 per variable |

### 2.2 Temporal Derivatives via Cauchy-Kowalewski

All temporal derivatives (u_t, u_tt, u_xt, u_xxt, ...) are **not independent unknowns**. They are computed from the spatial derivatives using the **Cauchy-Kowalewski (Lax-Wendroff) procedure**:

```
u_t = -f_x = -A · u_x                                           (from PDE)
u_tt = -f_xt = -(A · u_x)_t = A² · u_xx + A' · (u_x)² · A     (chain rule)
u_xt = -(f_x)_x = -(A · u_x)_x                                 (mixed)
```

For a nonlinear system, this involves the Jacobian A = ∂f/∂u and its derivatives. The 4th-order scheme requires computing these up to 3rd-order temporal derivatives.

### 2.3 Spatial Derivative Marching

The crucial step: how are the higher spatial derivatives (u_xx, u_xxx) computed at the new time level?

**In Jiang & Zhang (2025)**, the approach is:
1. March u forward using the space-time flux integral (same CE/SE structure as 2nd-order)
2. Compute u_x from u at the new time level via **finite differences** of the marched u values
3. Compute u_xx from u_x via finite differences
4. Compute u_xxx from u_xx via finite differences

The key insight from their paper: **p-th order derivatives are computed from (p-1)-th order derivatives** using finite differences on the staggered mesh. This hierarchical approach means the stencil remains compact (same as 2nd-order) because each finite difference only uses nearest neighbors.

### 2.4 CE/SE Geometry: Unchanged from 2nd-Order

A remarkable property: the 4th-order scheme uses the **exact same CEs, SEs, and stencil** as the 2nd-order scheme. Only the Taylor expansion (and thus the flux evaluation along SE boundaries) changes. This means:
- Same mesh structure
- Same data communication pattern
- Same CFL limit (ν ≤ 1)
- Same parallel decomposition

The additional cost is purely in the computation at each mesh point (evaluating more Taylor terms, more Cauchy-Kowalewski derivatives).

### 2.5 Computational Cost Comparison

For 8-component 3D ideal MHD:

| Scheme | Unknowns/cell | Flux eval cost | Memory/cell |
|--------|--------------|----------------|-------------|
| 2nd-order CESE | 8 × 4 = 32 | Low | 32 doubles |
| 4th-order CESE | 8 × 20 = 160 | High (Cauchy-Kowalewski) | 160 doubles |
| WENO5 + HLLD + SSP-RK3 | 8 | 3 RK stages × 3 dims × HLLD | ~24 doubles (with stages) |

The 4th-order CESE stores **20× more data per variable** than a standard FV scheme. However, Jiang & Zhang (2025) showed that a **4th-order CESE on N cells matches 2nd-order CESE on 4N cells**, so the effective cost is much lower for a given accuracy target.

---

## 3. CESE for MHD Equations

### 3.1 The 8-Component Ideal MHD System

The ideal MHD equations in conservation form:

```
∂ρ/∂t + ∇·(ρv) = 0                                             (mass)
∂(ρv)/∂t + ∇·(ρv⊗v + P*I - BB/μ₀) = 0                         (momentum)
∂E/∂t + ∇·((E + P*)v - B(v·B)/μ₀) = 0                         (energy)
∂B/∂t + ∇·(v⊗B - B⊗v) = 0                                     (induction)
```

where P* = p + B²/(2μ₀) is the total pressure and E = p/(γ-1) + ρv²/2 + B²/(2μ₀).

In the CESE framework, these are written as:

```
∂U/∂t + ∂F(U)/∂x + ∂G(U)/∂y + ∂H(U)/∂z = 0
```

where U = (ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz)ᵀ is the 8-component conservative state vector.

### 3.2 Divergence-Free B in CESE

The ∇·B = 0 constraint is a fundamental challenge for any MHD scheme. In the CESE framework, three approaches have been published:

**Approach 1: No special treatment (Zhang et al. 2006)**

Zhang, Yu et al. found that the **original CESE method naturally maintains ∇·B to acceptable levels** without any special treatment, because the space-time conservation enforced by the CE/SE structure implicitly respects the induction equation structure. In their tests (rotated MHD shock tube, MHD vortex), results with and without divergence cleaning were indistinguishable.

**Approach 2: Least-squares constrained derivatives (Yang et al. 2017)**

For the high-order scheme, the spatial derivatives of B (B_x, B_y, B_z and their higher derivatives) are constrained during the finite-difference step. Given the over-determined system of equations for the derivatives, a least-squares solve enforces:

```
∂Bx/∂x + ∂By/∂y + ∂Bz/∂z = 0                                  (9)
```

at every mesh point, at every time level. This is applied when computing (B_x)_x, (B_y)_y, (B_z)_z so that their sum vanishes. The key advantage: this is **local** (no global Poisson solve) and **cheap** (least-squares on a 3×N_unknowns system).

**Approach 3: WLS-ENO reconstruction (Yang et al. 2018)**

For the upwind CESE variant, a Weighted Least Squares Essentially Non-Oscillatory (WLS-ENO) reconstruction is applied that simultaneously maintains conservation, the ENO (non-oscillatory) property, and the divergence-free constraint. This is more sophisticated but not needed for the central CESE scheme.

### 3.3 CESE-MHD Published Results

| Test Problem | Reference | CESE Performance |
|-------------|-----------|-----------------|
| Rotated MHD shock tube | Zhang et al. 2006 | Good agreement, small oscillations in B |
| Brio-Wu shock tube | Jiang et al. 2017, 2025 | Captures all 7 wave families |
| Orszag-Tang vortex | Jiang et al. 2017, 2025 | Fine structure resolved at lower resolution |
| MHD blast wave | Jiang & Zhang 2025 | Robust, no carbuncle |
| Solar wind CME | Jiang et al. 2010 | Successful 3D AMR simulation |
| Current sheet | Feng et al. 2006 | Resistive reconnection captured |

---

## 4. Cylindrical Geometry Analysis

### 4.1 The Cylindrical MHD Equations

In cylindrical coordinates (r, φ, z), the ideal MHD equations have **geometric source terms** due to curvature:

```
∂U/∂t + (1/r)∂(r·Fr)/∂r + (1/r)∂Fφ/∂φ + ∂Fz/∂z = S(U, r)    (10)
```

where the geometric source vector S contains terms like:

```
S_ρvr = (ρvφ² + p + Bφ²/(2μ₀) - Br²/(2μ₀)) / r               (centrifugal + hoop stress)
S_ρvφ = -(ρvr·vφ - Br·Bφ/μ₀) / r                               (Coriolis-like)
S_Br = -(vr·Bφ - vφ·Br) / r                                     (curvature induction)
```

The **1/r singularity at the axis** (r = 0) is the primary challenge for cylindrical geometry.

### 4.2 CESE in Curvilinear Coordinates

The published approach (Jiang et al. 2010; Yang et al. 2018) handles curvilinear coordinates by **coordinate transformation**:

```
Physical domain: (r, φ, z) → Computational domain: (ξ, η, ζ)
```

The transformed equations retain conservation form:

```
∂Û/∂t + ∂F̂/∂ξ + ∂Ĝ/∂η + ∂Ĥ/∂ζ = 0                         (11)
```

where Û = J·U (J = Jacobian determinant) and the transformed fluxes include the metric terms. The CESE method is then applied directly to the transformed system in the rectangular computational domain.

**Advantages for cylindrical geometry:**
1. The geometric source terms (1/r terms) are absorbed into the metric Jacobian — no separate source term discretization needed
2. The space-time conservation is enforced in physical space, not computational space, avoiding "faux conservation" issues
3. The compact stencil means fewer ghost cells near the axis

**Challenges:**
1. The Jacobian J = r for cylindrical coords → J = 0 at axis, requiring regularization
2. Metric terms (∂r/∂ξ, etc.) must be computed to the same order as the solution
3. The least-squares div(B) constraint must be formulated in physical coordinates

### 4.3 Axis Singularity Treatment

For the r = 0 axis in DPF cylindrical geometry, two approaches exist:

1. **Regularized coordinates**: Use ξ = r² near the axis so J = ∂r/∂ξ = 1/(2√ξ) is bounded. This requires a non-uniform mesh in the computational domain.

2. **L'Hôpital limits**: The geometric source terms S ~ 1/r are matched by flux gradients that also vanish at r = 0 (by symmetry). The actual limit is finite:
   ```
   lim_{r→0} (1/r)∂(rFr)/∂r = 2·∂Fr/∂r |_{r=0}
   ```
   This can be enforced analytically in the CESE framework by modifying the CE/SE definitions at the axis cells.

For DPF: The axis singularity is manageable because DPF cylindrical grids typically start at r_min > 0 (the inner electrode has finite radius). The singularity only matters if we extend the grid to r = 0.

---

## 5. Resistive MHD Extension

### 5.1 Current State: Ideal MHD Only (Mostly)

The vast majority of CESE-MHD publications use **ideal MHD**. The sole exception is **Feng, Hu & Wei (2006)**, who applied the 2nd-order CESE to 2.5D resistive MHD for magnetic reconnection in Cartesian geometry.

Their approach: add the resistive diffusion term directly into the flux:

```
∂B/∂t = ∇×(v×B) - ∇×(η·∇×B)                                   (12)
       = ∇×(v×B) - ∇×(η·J)
```

In conservation form, the induction equation with resistivity becomes:

```
∂B/∂t + ∂/∂x[(v⊗B - B⊗v) - η(∂B/∂x)] = 0                    (13)
```

The η·∂²B/∂x² diffusion term is a parabolic addition to the hyperbolic system. This changes the character of the PDE from purely hyperbolic to **mixed hyperbolic-parabolic**.

### 5.2 Approaches for Adding Resistivity to CESE

**Option A: Include diffusion in the space-time integral (preferred for CESE)**

Modify the flux function to include the diffusive flux:

```
f_total(u, u_x) = f_convective(u) + f_diffusive(u_x)
```

where `f_diffusive = -η · u_x` for the magnetic field components. Since the CESE method already evolves u_x as an independent unknown, the diffusive flux can be evaluated **directly** without any additional discretization. This is a natural advantage of CESE — the gradient information needed for diffusion is already available.

The modified space-time flux vector becomes:

```
h = (f_convective(u) - η·u_x, u)                                (14)
```

and the conservation integral is applied as before. This does **not** require operator splitting.

**Pros**: Clean, no splitting error, maintains space-time conservation, naturally uses the already-available u_x.
**Cons**: Changes the CFL condition. The diffusive CFL adds a parabolic stability constraint:

```
ν_diff = η · Δt / Δx² ≤ 1/2                                     (15)
```

For large η (as in anomalous resistivity near the pinch), this can be more restrictive than the hyperbolic CFL.

**Option B: Operator splitting (Strang)**

Keep the CESE hyperbolic solver unchanged and apply resistive diffusion as a separate operator-split step:

```
U^{n+1} = D(Δt/2) · H(Δt) · D(Δt/2) · U^n                     (16)
```

where H is the CESE hyperbolic step and D is the diffusion step (explicit or implicit).

**Pros**: Keeps CESE solver clean, can use implicit diffusion for stiff η.
**Cons**: Splitting error O(Δt²), breaks space-time conservation (the main CESE selling point).

**Option C: Implicit-explicit (IMEX)**

Treat the diffusive terms implicitly within the CESE framework. This has not been published for CESE but is theoretically possible by modifying the marching equations to include an implicit solve for the diffusive contribution.

**Pros**: No parabolic CFL constraint, no splitting error.
**Cons**: Requires solving a linear system at each time step, significantly more complex to implement.

### 5.3 DPF Resistivity Requirements

For DPF simulations, we need:

| Resistivity Type | Typical η | Where | CESE Impact |
|-----------------|-----------|-------|-------------|
| Spitzer (classical) | ~10⁻⁷ Ω·m at 1 keV | Everywhere | Mild diffusion, Option A works |
| Anomalous (turbulent) | ~10⁻⁴ - 10⁻² Ω·m | Pinch column, current sheet | Severe parabolic CFL, need Option B or C |
| Enhanced (Chodura) | ~10⁻³ Ω·m | Sheath region | Localized, Option B with subcycling |

**Assessment**: For DPF, the Spitzer resistivity in most of the domain is small enough that Option A (include in flux) works fine. The anomalous resistivity in the pinch column is highly localized and stiff — this would require either subcycling or an operator-split approach (Option B).

Our current DPF code already uses **Strang splitting** for resistive diffusion (see `engine.py`), so Option B would map naturally onto the existing architecture.

---

## 6. Implementation Roadmap

### Stage 1: 1D CESE Euler Proof of Concept

**Goal**: Build institutional knowledge, verify equations, test shock capturing.

**Scope**:
- 1D Euler equations (3 components: ρ, ρv, E)
- 2nd-order CESE with dissipation
- Staggered mesh, periodic + outflow BCs
- Sod shock tube verification
- Convergence study (smooth problem)

**LOC estimate**: ~400 Python LOC
- Core CESE engine: ~200 LOC (CE/SE definitions, Taylor expansion, marching)
- Euler flux & Cauchy-Kowalewski: ~80 LOC
- Boundary conditions: ~40 LOC
- Test problems (Sod, smooth wave): ~80 LOC

**Timeline**: 2-3 days for an experienced developer
**Risk**: Low — well-documented in literature

### Stage 2: 1D 4th-Order CESE Euler

**Goal**: Validate 4th-order extension, measure convergence rate.

**Scope**:
- Extend Stage 1 to 4th-order Taylor expansion
- Add u_xx, u_xxx as marching unknowns
- Cauchy-Kowalewski up to 3rd temporal derivatives
- Verify 4th-order convergence on smooth problems

**LOC estimate**: ~250 additional LOC (on top of Stage 1)
- Higher-order Taylor terms: ~100 LOC
- Cauchy-Kowalewski chain rule: ~80 LOC
- Higher-order derivative marching: ~70 LOC

**Timeline**: 2-3 days
**Risk**: Medium — Cauchy-Kowalewski for nonlinear Euler is algebraically involved

### Stage 3: 2D Cartesian Ideal MHD CESE

**Goal**: Prove CESE works for MHD with div(B) handling.

**Scope**:
- 2D Cartesian MHD (8 components)
- 2nd-order CESE (simplicity over 4th-order)
- Least-squares div(B) constraint
- Brio-Wu and Orszag-Tang tests

**LOC estimate**: ~800-1000 LOC
- 2D CESE engine (CE/SE in 2+1 dimensions): ~300 LOC
- MHD flux functions (8-component): ~150 LOC
- Cauchy-Kowalewski for MHD Jacobian: ~150 LOC
- Div(B) least-squares constraint: ~100 LOC
- Boundary conditions (2D): ~100 LOC
- Test problems: ~100 LOC

**Timeline**: 1-2 weeks
**Risk**: Medium-High — 2D CE/SE geometry is significantly more complex than 1D

### Stage 4: 2D Cylindrical Ideal MHD CESE

**Goal**: DPF-relevant geometry.

**Scope**:
- Extend Stage 3 to (r, z) cylindrical coordinates
- Metric terms and Jacobian (J = r)
- Axis treatment (r → 0)
- Geometric source terms via coordinate transform
- Bennett equilibrium verification
- Magnetized Noh verification

**LOC estimate**: ~500-700 additional LOC
- Coordinate transformation and metrics: ~150 LOC
- Axis regularization: ~100 LOC
- Modified CE/SE for cylindrical: ~150 LOC
- Test problems (Bennett, Noh): ~100 LOC
- Additional boundary conditions (electrode, axis): ~100 LOC

**Timeline**: 1-2 weeks
**Risk**: High — axis singularity + cylindrical metrics are where implementations often fail

### Stage 5: Full Resistive Cylindrical MHD + Circuit Coupling

**Goal**: Production-ready DPF CESE solver.

**Scope**:
- Add Spitzer resistivity (in-flux, Option A)
- Add anomalous resistivity (operator-split, Option B)
- Circuit coupling (L_plasma, back-EMF)
- Two-temperature model (Te, Ti)
- Bremsstrahlung radiation
- Full DPF engine integration

**LOC estimate**: ~1000-1500 additional LOC
- Resistive diffusion (in-flux): ~150 LOC
- Operator-split anomalous resistivity: ~150 LOC
- Circuit interface: ~100 LOC
- Two-temperature model: ~200 LOC
- Radiation: ~100 LOC
- Engine integration (config, state dict, diagnostics): ~300 LOC
- Additional tests: ~200 LOC

**Timeline**: 3-4 weeks
**Risk**: Very High — untested territory (no published resistive cylindrical CESE-MHD)

### Stage 6: 4th-Order Upgrade

**Goal**: Maximize accuracy advantage of CESE.

**Scope**:
- Upgrade Stage 5 from 2nd-order to 4th-order
- Higher-order Cauchy-Kowalewski for MHD in cylindrical coords
- Higher-order metric terms
- Verification of 4th-order convergence

**LOC estimate**: ~500-800 additional LOC
**Timeline**: 2-3 weeks
**Risk**: Very High — 4th-order cylindrical MHD is unprecedented in the literature

### Total LOC and Timeline Summary

| Stage | LOC | Cumulative | Timeline | Risk |
|-------|-----|------------|----------|------|
| 1. 1D Euler (2nd order) | ~400 | 400 | 2-3 days | Low |
| 2. 1D Euler (4th order) | ~250 | 650 | 2-3 days | Medium |
| 3. 2D Cartesian MHD | ~900 | 1,550 | 1-2 weeks | Medium-High |
| 4. 2D Cylindrical MHD | ~600 | 2,150 | 1-2 weeks | High |
| 5. Resistive + circuit | ~1,250 | 3,400 | 3-4 weeks | Very High |
| 6. 4th-order upgrade | ~650 | 4,050 | 2-3 weeks | Very High |

**Total**: ~4,000 LOC, ~10-14 weeks for a full 4th-order resistive cylindrical CESE-MHD solver.

### Metal GPU Compatibility

**Can CESE run on Apple Metal via PyTorch tensors?** Yes, in principle:

1. **Data layout**: CESE stores (u, u_x, u_y, u_z, u_xx, ...) at each point. These map naturally to multi-channel tensors: `[N_unknowns × N_vars, nx, ny, nz]`.

2. **Stencil operations**: The CESE marching equations use only **nearest-neighbor** data, which maps to 3×3 (2D) or 3×3×3 (3D) convolution stencils — highly efficient on GPU.

3. **No branching**: Unlike HLLD (which has complex if/else for wave speeds), the CESE marching formulas are **pure arithmetic** — ideal for GPU vectorization.

4. **Memory**: 4th-order CESE stores 20 unknowns × 8 MHD vars = 160 floats per cell. For a 128³ grid in float32: 160 × 128³ × 4 bytes ≈ 1.3 GB — fits in M3 Pro's 36GB unified memory.

5. **Float32 precision**: The CESE method's space-time conservation is exact to machine precision. In float32, this means conservation to ~10⁻⁷, same as our current Metal solver.

**Assessment**: CESE is **more GPU-friendly than WENO5+HLLD** because it avoids the complex branching in the HLLD Riemann solver and the nonlinear WENO weights. The regular stencil and pure arithmetic make it ideal for tensor operations.

---

## 7. Honest Comparison: CESE vs. Current WENO5+HLLD

### 7.1 What We Gain

| Advantage | Significance for DPF | Confidence |
|-----------|---------------------|------------|
| No Riemann solver needed | Eliminates HLLD NaN/stability issues in float32 | High |
| Compact stencil (nearest neighbor) | Simpler BCs at electrodes, axis | High |
| CFL ≤ 1 independent of order | Larger timesteps at 4th order | High |
| Genuinely multidimensional | No splitting artifacts in (r,φ,z) | High |
| Simultaneous space-time accuracy | 4th-order in time without SSP-RK stages | High |
| Natural div(B) handling | May eliminate need for CT or Dedner | Medium |
| GPU-friendly (no branching) | Better Metal GPU utilization | Medium |
| 4th-order at cost of 2nd-order (4× resolution) | Dramatic efficiency gain if true | Medium (unverified for our case) |

### 7.2 What We Lose

| Disadvantage | Significance for DPF | Confidence |
|-------------|---------------------|------------|
| Mature, tested codebase | Our WENO5+HLLD has 2000+ tests, months of debugging | High |
| Known behavior at DPF conditions | CESE-MHD only tested for solar wind | High |
| Athena++ cross-validation | Can't compare CESE against Athena++ easily | High |
| Community support | CESE-MHD is a niche method, few practitioners | High |
| Resistive MHD track record | Only one paper (Feng 2006), 2nd-order, Cartesian | High |
| Open-source code | No 4th-order CESE-MHD code available | High |
| Shock capturing robustness | CESE dissipation mechanism is less battle-tested than HLLD | Medium |
| Memory overhead (4th order) | 20× more unknowns per variable per cell | Medium |
| Cauchy-Kowalewski complexity | Algebraically involved for nonlinear MHD (8-component Jacobian) | Medium |

### 7.3 Risk Assessment

**Technical risks:**
1. **Axis singularity**: No published CESE-MHD work on cylindrical coordinates with r=0 axis treatment. This is uncharted territory.
2. **Resistive MHD stability**: Adding diffusion to CESE can change the stability character. No published analysis for CESE + anomalous resistivity.
3. **Two-temperature physics**: No published CESE work with separate Te/Ti evolution. The energy equation coupling is complex.
4. **Circuit coupling**: The CESE time-stepping is locked to the CFL; circuit coupling may need adaptive time stepping.
5. **Shock capturing in cylindrical converging flows**: The CESE dissipation mechanism has not been tested for the extreme compression ratios (>100×) seen in DPF pinch.

**Schedule risks:**
1. LOC estimates have historically been **underestimated by 2-3×** in this project (P0 Bennett: estimated 50, actual 180).
2. The 10-14 week estimate could easily become 6+ months with debugging and verification.
3. Meanwhile, our current WENO5+HLLD pipeline continues to improve incrementally.

**Opportunity cost:**
- Time spent on CESE cannot be spent on:
  - Athena++ PPM+characteristic (proven, adds ~0.1 to accuracy score)
  - WALRUS fine-tuning on DPF data (immediate AI capabilities)
  - Phase AB+ improvements to existing solvers
  - Additional verification benchmarks (P2, P3)

---

## 8. Recommendation

### Tier 1 (Do Now): Learn CESE Fundamentals
- **Implement Stage 1** (1D 2nd-order CESE Euler, ~400 LOC, 2-3 days)
- This builds institutional knowledge at minimal cost
- Validates our understanding of the equations
- Provides a reference implementation we can point to

### Tier 2 (Do If Stage 1 Succeeds): Explore 4th-Order
- **Implement Stage 2** (1D 4th-order CESE Euler, ~250 LOC, 2-3 days)
- Verifies 4th-order convergence claims
- Tests Cauchy-Kowalewski for nonlinear systems

### Tier 3 (Defer): Full CESE-MHD
- **Defer Stages 3-6** until one of:
  (a) Jiang & Zhang release their 4th-order CESE-MHD code (most likely path)
  (b) Our WENO5+HLLD pipeline hits a fundamental accuracy ceiling
  (c) A specific DPF physics case fails with WENO5+HLLD but succeeds with CESE (axis singularity, extreme CFL, etc.)

### Decision Matrix

| Condition | Action |
|-----------|--------|
| Jiang & Zhang release code | Fork it, adapt to cylindrical, add resistivity |
| WENO5+HLLD accuracy ceiling hit | Investigate CESE as alternative engine |
| DPF axis singularity problems emerge | Prototype CESE cylindrical (Stage 4) |
| None of the above | Continue improving current pipeline |

### Score Impact Assessment

**CESE is not expected to improve the PhD debate score in the near term.** The score is currently limited by:
1. Validation breadth (more benchmarks needed) — addressed by QSSS
2. Open bugs (C2, C3, M9) — addressed by bug fixes
3. Resistive MHD + circuit coupling correctness — addressed by D1/D2 fixes

A CESE implementation would only help if:
- It enables higher-fidelity simulations than our current triple-engine approach
- It can be validated against the same benchmark suite
- It demonstrates clear superiority for DPF-specific problems

None of these benefits are achievable in the near term (<3 months).

---

## 9. Key References

### Foundational CESE
1. Chang S.C., "The Method of Space-Time Conservation Element and Solution Element," *J. Comput. Phys.* 119, 295-324 (1995). — Original method.
2. Chang S.C., "The Method of Space-Time Conservation Element and Solution Element: A New High-Resolution and Genuinely Multidimensional Paradigm for Solving Conservation Laws," *J. Comput. Phys.* 156, 89-136 (1999). — Multi-D extension.
3. Wen C.-Y. et al., *Space-Time Conservation Element and Solution Element Method: Advances and Applications in Engineering Sciences*, Springer (2023). — Comprehensive textbook.

### Higher-Order CESE
4. Bilyeu D.L., "A higher-order conservation element solution element method for solving hyperbolic differential equations on unstructured meshes," PhD Thesis, Ohio State (2014).
5. Bilyeu D.L. et al., "A Two-Dimensional Fourth-Order Unstructured-Meshed Euler Solver Based on the CESE Method," *J. Comput. Phys.* 257, 981-999 (2014).
6. Jiang C., Zhang L., "A New Implementation of a Fourth-Order CESE Scheme for 3D MHD Simulations," *Solar Physics* 300, 39 (2025). — **Most relevant: 4th-order MHD.**

### CESE for MHD
7. Zhang M. et al., "Solving the MHD equations by the space-time CESE method," *J. Comput. Phys.* 214, 599-617 (2006). — First CESE-MHD.
8. Feng X. et al., "Modeling the Resistive MHD by the CESE Method," *Solar Physics* 235, 235-257 (2006). — **Only resistive CESE-MHD paper.**
9. Yang Y. et al., "A high-order CESE scheme with a new divergence-free method for MHD," *J. Comput. Phys.* 330, 280-300 (2017). — High-order + div(B).
10. Yang Y. et al., "An upwind CESE scheme for 2D and 3D MHD in general curvilinear coordinates," *J. Comput. Phys.* 371, 850-869 (2018). — Curvilinear coordinates.
11. Jiang C. et al., "AMR Simulations of MHD Problems by the CESE Method in Curvilinear Coordinates," *Solar Physics* 267, 463-491 (2010). — AMR + curvilinear.

### Open-Source Implementations
12. SOLVCON: https://github.com/solvcon/solvcon — 2nd-order CESE framework (Python/C++, ~1600 commits, 18 stars). **No MHD support.** Code moved to https://github.com/solvcon/modmesh.
13. **No open-source 4th-order CESE-MHD code exists.** All implementations are research codes held by the Jiang/Feng group at NSSC (Chinese Academy of Sciences).

---

## Appendix A: CESE vs. DPF Current Solver — Quick Reference

| Feature | Python Engine | Metal Engine | CESE (proposed) |
|---------|--------------|--------------|-----------------|
| Formulation | Non-conservative (p) | Conservative (E) | Conservative (E) |
| Reconstruction | WENO5-Z | PLM/WENO5 | None (Taylor) |
| Riemann solver | HLLD | HLL/HLLD | None |
| Time integration | SSP-RK2/RK3 | SSP-RK2/RK3 | Implicit in CE/SE |
| Spatial order | 5th (smooth) | 2nd-5th | 2nd or 4th |
| Temporal order | 2nd (RK2)/3rd (RK3) | 2nd/3rd | Same as spatial |
| CFL limit | ~0.5 | ~0.5 | ~1.0 |
| Stencil width | 5 cells | 2-5 cells | 1 cell (nearest) |
| Div(B) | Dedner | CT | Least-squares |
| Directional splitting | Yes | Yes | No |
| GPU friendly | No (NumPy) | Yes (MPS) | Yes (pure arithmetic) |
| LOC | ~5,300 | ~1,500 | ~4,000 (est.) |

## Appendix B: Jiang & Zhang (2025) Key Claims

From "A New Implementation of a Fourth-Order CESE Scheme for 3D MHD Simulations" (Solar Physics, 2025):

1. 4th-order accuracy in both space and time simultaneously
2. Compact stencil identical to 2nd-order CESE
3. CFL ≤ 1 for all orders
4. General framework extensible to arbitrary order (just add Taylor terms)
5. 4th-order on N cells ≈ 2nd-order on 4N cells in accuracy, at ~5% compute cost
6. Spatial derivatives computed hierarchically: p-th order from (p-1)-th order via finite differences
7. Successfully tested on Brio-Wu, blast wave, and other standard MHD benchmarks
8. **No code release announced**
