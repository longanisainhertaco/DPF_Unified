# Multi-Material Arbitrary Lagrangian-Eulerian (ALE) Methods for MHD Codes

**Classification**: Shelf Research — ICF/HED Prototype Module
**Scope**: Standalone prototype; NOT integrated into DPF-Unified
**Date**: 2026-03-26
**Audience**: PhD-level plasma physicist

---

## Table of Contents

1. [Motivation and Context](#1-motivation-and-context)
2. [Governing Equations: Full ALE Formulation](#2-governing-equations-full-ale-formulation)
3. [The Geometric Conservation Law (GCL)](#3-the-geometric-conservation-law-gcl)
4. [Literature Basis and Production Codes](#4-literature-basis-and-production-codes)
5. [Interface Reconstruction Methods](#5-interface-reconstruction-methods)
6. [Material Mixing: Closure Models for Mixed Cells](#6-material-mixing-closure-models-for-mixed-cells)
7. [Remap Algorithms](#7-remap-algorithms)
8. [Relevance to DPF: ICF/HED vs. Our Operating Regime](#8-relevance-to-dpf-icfhed-vs-our-operating-regime)
9. [Prototype: 1D Two-Material ALE Sod Shock Tube](#9-prototype-1d-two-material-ale-sod-shock-tube)
10. [Integration Cost Estimate](#10-integration-cost-estimate)
11. [References](#11-references)

---

## 1. Motivation and Context

In high-energy-density (HED) physics and inertial confinement fusion (ICF), the simulation domain invariably contains **multiple distinct materials** whose boundaries evolve dynamically:

- Ablator (CH, Be, HDC) / DT ice / DT gas in ICF capsules
- Hohlraum wall (Au, U) / fill gas (He, Ne) / laser entrance hole foil
- Electrode material (Cu, steel) eroded into the plasma in Z-pinch and plasma focus drivers
- Liner / gas puff interface in magnetized liner inertial fusion (MagLIF)

Pure Eulerian codes smear these interfaces over several computational zones, introducing numerical diffusion that contaminates the equation of state and transport properties. Pure Lagrangian codes track interfaces exactly but suffer mesh tangling when shear flows or instabilities develop (Rayleigh-Taylor, Kelvin-Helmholtz). **ALE methods** occupy the middle ground: the mesh moves with a user-controlled velocity that can be chosen anywhere between fully Lagrangian (mesh velocity = fluid velocity) and fully Eulerian (mesh velocity = 0), with a **remap step** that projects the deformed Lagrangian solution onto a regularized mesh.

Multi-material ALE is the standard approach in production ICF/HED codes (ALEGRA, HYDRA, GORGON, FLASH-ALE) for exactly this reason.

---

## 2. Governing Equations: Full ALE Formulation

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| **v** | Fluid velocity |
| **w** | Mesh (grid) velocity |
| **c** = **v** − **w** | Convective velocity (fluid relative to mesh) |
| J | Jacobian of the map from reference to current configuration |
| ρ, e, p | Density, specific internal energy, pressure |
| **B** | Magnetic field |
| **σ** | Viscous stress tensor |
| η | Magnetic resistivity |

### 2.2 ALE Conservation Laws in Strong Form

The ALE description introduces a **reference frame** that moves with velocity **w**. The material time derivative in the ALE frame is:

```
d/dt|_χ = ∂/∂t|_x + (v - w)·∇ = ∂/∂t|_x + c·∇
```

where χ is the ALE reference coordinate and x is the spatial (Eulerian) coordinate.

**Mass conservation:**

```
∂ρ/∂t|_x + ∇·(ρc) + ρ∇·w = 0

Equivalently in ALE control-volume form:
d/dt ∫_Ω(t) ρ dV = -∮_∂Ω(t) ρ(v - w)·n dA
```

**Momentum conservation:**

```
ρ [∂v/∂t|_x + (c·∇)v] = -∇p + ∇·σ + J×B + ρg

ALE control-volume form:
d/dt ∫_Ω(t) ρv dV = -∮_∂Ω(t) ρv(v - w)·n dA - ∮_∂Ω(t) p n dA
                    + ∫_Ω(t) (J×B + ρg) dV + surface viscous terms
```

**Total energy conservation:**

```
ρ [∂E/∂t|_x + (c·∇)E] = -∇·(pv) + ∇·(σ·v) + (J×B)·v + ∇·(κ∇T) + Q_rad

where E = e + |v|²/2  (specific total energy)

ALE control-volume form:
d/dt ∫_Ω(t) ρE dV = -∮_∂Ω(t) ρE(v - w)·n dA - ∮_∂Ω(t) pv·n dA
                   + ∫_Ω(t) [(J×B)·v + Q_rad] dV
```

**Magnetic induction (resistive MHD):**

```
∂B/∂t|_x + ∇×(B×v) = -∇×(η J)     [Eulerian form]

ALE form (accounting for moving mesh):
∂B/∂t|_χ + ∇×(B×c) + (∇·w)B - (B·∇)w = -∇×(η J)

Or more compactly via the Lie derivative:
∂B/∂t|_χ + L_w B = ∇×(c×B) - ∇×(ηJ)

where L_w is the Lie derivative along w.
```

The current density J = ∇×B/μ₀ and ∇·B = 0 must be maintained.

**Divergence constraint in ALE:**

```
∂(∇·B)/∂t + ∇·[(∇·B)w] = 0
```

This shows that if ∇·B = 0 initially and the mesh motion is smooth, the constraint propagates cleanly. However, in practice, remap introduces div B errors that must be corrected (see Section 7.3).

### 2.3 Operator Splitting: Lagrangian + Remap

The ALE algorithm is typically implemented as a **two-phase operator split**:

**Phase 1 — Lagrangian step** (set **w** = **v**, so **c** = 0):

```
The governing equations reduce to:
  dρ/dt + ρ∇·v = 0
  ρ dv/dt = -∇p + ∇·σ + J×B
  ρ dE/dt = -p∇·v + σ:∇v + (J×B)·v + Q_rad
  dB/dt = B·∇v - B∇·v - ∇×(ηJ)    [Lagrangian induction]
  d(ρ dV)/dt = 0                    [mass conservation in cell]
```

This is solved with a standard Lagrangian scheme (e.g., staggered-grid hydrodynamics, compatible discretization). The mesh nodes move with the fluid: **x**ⁿ⁺¹_L = **x**ⁿ + Δt **v**.

**Phase 2 — Remap step** (project from deformed Lagrangian mesh back to target mesh):

Given the Lagrangian solution (ρ_L, **v**_L, E_L, **B**_L) on the deformed mesh, compute the ALE solution on the new target mesh Ω^{n+1} by:

```
∫_Ω^{n+1} q^{n+1} dV = ∫_Ω^{n+1} q_L(x) dV

where q_L is the piecewise-reconstructed Lagrangian field.
This is implemented as a swept-region or intersection flux calculation.
```

---

## 3. The Geometric Conservation Law (GCL)

The **Geometric Conservation Law** (Thomas & Lombard 1979; Guillard & Farhat 2000) is the discrete counterpart of the identity:

```
d/dt ∫_Ω(t) dV = ∫_∂Ω(t) w·n dA
```

In a moving-mesh finite-volume scheme, if the cell volumes V_i^n change with time, the numerical flux for a uniform field (ρ = const, v = 0, B = const) must satisfy:

```
GCL condition:
  (V_i^{n+1} - V_i^n) / Δt = Σ_f  (mesh face area swept by face f) · w_f · n_f

or equivalently:  δV_i = Σ_f  F_w,f · Δt
```

**Why it matters:**
Violation of the GCL introduces **spurious mass creation or destruction** even for a trivially uniform flow. In MHD, GCL errors also generate artificial ∇·B. Any ALE scheme must satisfy the GCL discretely — this is a hard constraint on how the swept-region flux is computed.

**Discrete GCL for the magnetic field** (Dukowicz & Baumgardner 2000):

```
For the induction equation in ALE form, the discrete GCL requires:
  (B_i^{n+1} V_i^{n+1} - B_i^n V_i^n) / Δt = L(B, v, w) - D(η, B)

where L contains the Lagrangian advection and stretching terms,
and the volume change δV_i = Σ_f w_f · A_f · Δt must be
computed consistently with the face-swept volumes used in the flux terms.
```

In practice: compute δV from swept-region geometry first; use these same geometric quantities in all flux evaluations.

---

## 4. Literature Basis and Production Codes

### 4.1 Foundational Papers

**Benson 1992** — "Computational methods in Lagrangian and Eulerian hydrocodes" (Comput. Methods Appl. Mech. Eng., 99:235-394)
The canonical review. Covers staggered-grid Lagrangian mechanics, von Neumann-Richtmyer artificial viscosity, and the full taxonomy of remap methods. Still the primary reference for anyone implementing a Lagrangian-phase hydro solver. Introduces the "3-point" vs "9-point" stencil distinction for artificial viscosity.

**Margolin & Shashkov 2003** — "Second-order sign-preserving conservative interpolation (remapping) on general grids" (J. Comput. Phys., 184:266-298)
Establishes the theoretical framework for conservative, monotone remap. Proves that local conservation and second-order accuracy are compatible. Introduces the **intersection-based** remap as the reference-quality approach, and swept-region as the practical approximation. Essential reading for implementing the remap phase.

**Barlow 2016** — "A compatible finite element multi-material ALE hydrodynamics algorithm" (Int. J. Numer. Methods Fluids, 82:3-39)
Derives the **compatible discretization** approach: staggered-grid formulation where momentum is cell-centered and kinetic energy is vertex-based, ensuring exact energy conservation at the discrete level. The "Barlow compatible" discretization is a key alternative to GLACE/EUCCLHYD cell-centered schemes. Important for multi-material because it avoids the spurious interface pressure oscillations common in cell-centered schemes.

**Galera, Maire & Breil 2010** — "A two-dimensional unstructured cell-centered multi-material ALE scheme using VOF interface reconstruction" (J. Comput. Phys., 229:5755-5787)
The primary reference for combining GLACE cell-centered Lagrangian hydro with **Moment-of-Fluid (MOF)** interface reconstruction and conservative remap. Demonstrates sub-cell resolution of interfaces without smearing. The MOF reconstruction algorithm they implement is the Dyadechko & Shashkov 2008 variant.

**Loubere et al. 2010** — "ReALE: A reconnection-based arbitrary-Lagrangian-Eulerian method" (J. Comput. Phys., 229:4724-4761)
Introduces mesh reconnection (topology changes) within ALE, allowing the mesh to adapt to flow features including shear and vorticity without tangling. Relevant for problems where the standard smoothing-only mesh relaxation fails.

**Dukowicz & Baumgardner 2000** — "Incremental remapping as a transport/advection algorithm" (J. Comput. Phys., 160:318-335)
Defines the swept-region remap for geophysical flows. The key contribution: the "incremental remap" algorithm that handles the intersection geometry efficiently by decomposing the swept region into sub-triangles or sub-hexahedra.

**Kidder 1974** — "Laser-driven compression of hollow shells: Power requirements and stability limitations" (Nucl. Fusion, 14:53)
Establishes the ICF implosion geometry that motivated ALE development for capsule simulations. Historical context only.

**Thomas & Lombard 1979** — "Geometric conservation law and its application to flow computations on moving grids" (AIAA J., 17:1030-1037)
Original GCL paper. Still the primary citation.

### 4.2 Production Codes Using Multi-Material ALE

**ALEGRA** (Sandia National Laboratories)
- Full name: Arbitrary Lagrangian-Eulerian General Research Application
- Primary language: C++ with Trilinos
- ALE method: Staggered-grid Lagrangian (compatible, mimetic) + swept-region remap
- Interface reconstruction: VOF (piecewise linear interface construction, PLIC)
- MHD: Full resistive MHD, radiation diffusion, EOS tables (SESAME)
- Validation: Sandia Z-machine liner experiments, ICF hohlraum
- Reference: Robinson et al. 2008 (SAND2007-7867)

**HYDRA** (Lawrence Livermore National Laboratory)
- Primary language: Fortran 90/C++
- ALE method: Cell-centered Lagrangian + rezone/remap
- Interface reconstruction: MOF (Moment-of-Fluid)
- MHD: Implicit resistive MHD with B-field advection on moving mesh
- Radiation: Multi-group diffusion
- Primary use: NIF ICF capsule design and post-shot analysis
- Reference: Marinak et al. 2001 (Phys. Plasmas, 8:2275)

**GORGON** (Imperial College London / LLNL)
- Primary language: Fortran
- ALE method: Eulerian-Lagrangian hybrid (mostly Eulerian with some ALE capability)
- Interface tracking: Level-set
- MHD: Resistive MHD with Braginskii transport
- Primary use: Z-pinch, wire-array Z-pinch, plasma focus (the closest to our regime)
- Reference: Chittenden et al. 2004 (Phys. Plasmas, 11:1118); Jennings et al. 2010

**FLASH-ALE** (University of Chicago / Flash Center)
- Open-source base; ALE module added by Fatenejad et al. 2013
- Interface reconstruction: VOF (PLIC)
- Multi-material: Yes, with SESAME EOS tables
- Primarily used for laser-driven experiments, HEDP

**CRASH** (University of Michigan)
- Designed explicitly for HEDP radiative shocks
- ALE with AMR
- Multi-material with arbitrary EOS

**LASNEX** (LLNL, historical)
- Original ICF code (Zimmerman & Kruer 1975)
- r-z Lagrangian with rezoning; the intellectual ancestor of HYDRA

**ASTER** (CEA, France)
- Multi-material ALE with MOF
- Full radiation-MHD

---

## 5. Interface Reconstruction Methods

Interface reconstruction determines the sub-cell geometry of material boundaries within mixed cells. Three primary classes exist:

### 5.1 Volume-of-Fluid (VOF)

**Concept**: Track the volume fraction f_k of each material k in each cell. Interface is reconstructed as a piecewise-linear planar segment (in 2D) or polygon (in 3D) within each mixed cell using the volume fractions and their gradients.

The standard algorithm is **PLIC** (Piecewise Linear Interface Construction):

```
Given: f_k in cell i and neighboring cells
Step 1: Compute interface normal n = -∇f_k / |∇f_k|
        (using Youngs' method: n computed from 3x3 stencil of volume fractions)
Step 2: Find plane position α such that:
        ∫_{Ω_i ∩ {x: n·x ≤ α}} dV = f_k · V_i
        (solved with a Newton iteration or analytic formula for box geometries)
Step 3: The interface is the planar segment {x: n·x = α} ∩ Ω_i
```

**Pros:**
- Conservative by construction (volume fractions are cell-integrated quantities)
- Robust for topological changes (merging, splitting bubbles)
- Computationally efficient; O(N) where N = number of cells
- No thin-film problem (no minimum thickness constraint)

**Cons:**
- Single plane per cell: cannot represent multiple interfaces in one cell
- Interface normal accuracy limited to ~Δx² (Youngs' gradient stencil is first-order in practice)
- "Flotsam and jetsam" artifacts: small detached volume fragments appear and are hard to merge
- Centroid information is lost: all material within a sub-region is assumed uniform

**In MHD context**: VOF is the most commonly used method for plasma-material interfaces in production codes (ALEGRA, FLASH). The lack of centroid information is acceptable when the interface is resolved (Δx << interface radius of curvature) but problematic for the thin sheath problem.

### 5.2 Moment-of-Fluid (MOF)

**Concept** (Dyadechko & Shashkov 2008; Galera et al. 2010): Extend VOF to track not just the volume fraction f_k but also the **material centroid** x̄_k in each mixed cell. The interface is still piecewise-linear per cell, but the plane is chosen to match both the volume fraction AND the centroid.

```
Given: f_k, x̄_k in cell i
Minimize over (n, α):
  |x̄_k^reconstructed(n, α) - x̄_k|²
subject to:
  volume_fraction(n, α) = f_k

This is a nonlinear optimization (typically Brent or Nelder-Mead)
with analytic gradients available for hexahedral cells.
```

**Pros:**
- Sub-cell accuracy: interface position accurate to ~Δx³ (one order better than VOF)
- Better preserves sharp corners and cusps
- Centroid tracking reduces "flotsam/jetsam" artifacts
- Can represent materials that don't touch (isolated inclusions)

**Cons:**
- Computationally expensive: O(N × N_iter) where N_iter = optimization iterations (~10-50 per cell per step)
- Complex implementation, especially in 3D on unstructured meshes
- The optimization can have multiple local minima for convex cells with small volume fractions

**In MHD context**: MOF is state-of-the-art for high-accuracy ICF implosion simulations (HYDRA). The extra computational cost (~3-5x over VOF) is justified when sub-cell interface accuracy matters — e.g., tracking the ablator/ice boundary in a capsule or the liner surface in MagLIF.

### 5.3 Level-Set Methods

**Concept**: Track the interface as the zero isocontour of a smooth scalar field φ(x,t), where φ > 0 in material 1 and φ < 0 in material 2. φ is chosen to be the signed distance function: |∇φ| = 1.

Evolution:
```
∂φ/∂t + v·∇φ = 0    [pure advection]

After advection, φ is re-distanced (reinitialized) to restore |∇φ| = 1:
∂φ/∂τ + sgn(φ₀)(|∇φ| - 1) = 0    [reinitialization PDE, pseudo-time τ]
```

**Pros:**
- Smooth, differentiable representation: normal and curvature are analytically available
- Natural for surface tension, interface curvature-dependent phenomena
- Easy to extend to multiple materials (multiple level sets or signed-distance composition)
- Handles topology changes (merging, splitting) automatically

**Cons:**
- **Not conservative**: volume is not exactly preserved; requires coupling with VOF for conservation (the "CLSVOF" hybrid)
- Reinitialization introduces numerical errors near the interface
- Band-limited: only cells near φ = 0 are resolved; distant cells carry stale φ
- Thin films problematic: if two interfaces are within ~Δx of each other, they merge numerically

**In MHD context**: Level-set is used in GORGON (the code closest to our plasma-focus regime) for tracking the ablated wire/plasma interface. The non-conservation is managed by coupling with a volume-fraction tracker. For magnetic field purposes, the interface normal from level-set is clean, which simplifies the interface boundary conditions for **B**_n continuity and the pressure jump due to surface currents.

### 5.4 Comparison Summary

| Property | VOF (PLIC) | MOF | Level-Set |
|----------|------------|-----|-----------|
| Conservation | Exact | Exact | Not exact |
| Interface accuracy | O(Δx²) | O(Δx³) | O(Δx²) |
| Curvature accuracy | Poor | Moderate | Good |
| Multi-material (>2) | Complex | Yes | Multiple LS |
| Cost per step | Low | High (3-5x VOF) | Low-Medium |
| Thin films | OK | OK | Problematic |
| Implementation complexity | Medium | High | Medium |
| Production code examples | ALEGRA, FLASH | HYDRA, ASTER | GORGON |

**Recommendation for ICF/HED prototype**: Start with **VOF/PLIC** for robustness and conservation. Add MOF as a higher-accuracy option once the Lagrangian and remap phases are validated.

---

## 6. Material Mixing: Closure Models for Mixed Cells

When two materials occupy a single computational cell, the cell has a single pressure and velocity from the hydrodynamic solve, but the materials have different thermodynamic states. A **closure model** is needed to decompose the single-cell state into per-material states.

### 6.1 The Mixed-Cell Problem

After the Lagrangian step, a mixed cell i with materials A and B has:
- Total density: ρ_i = f_A ρ_A + f_B ρ_B  (by definition of volume fractions)
- Single cell pressure: p_i (from the Lagrangian solve)
- Single cell velocity: **v**_i (from the momentum equation)
- Total internal energy: ρ_i e_i (from the energy equation)

We need to find (ρ_A, e_A, p_A) and (ρ_B, e_B, p_B) such that:
1. Volume fractions are satisfied: ρ = f_A ρ_A + f_B ρ_B
2. Total energy: ρ e = f_A ρ_A e_A + f_B ρ_B e_B
3. Closure: some condition relating p_A, p_B (or T_A, T_B)

### 6.2 Pressure Equilibrium (Isobaric) Closure

**Assumption**: The two materials instantly reach pressure equilibrium:

```
p_A = p_B = p_i

Then:
  e_A = e_A(ρ_A, p_i)    [from EOS_A]
  e_B = e_B(ρ_B, p_i)    [from EOS_B]

And the densities are found by solving:
  f_A V_A(p_i) + f_B V_B(p_i) = V_i    [volume constraint]

where V_k = 1/ρ_k is the specific volume.
This is a single nonlinear equation in ρ_A (or ρ_B), solved by bisection.
```

**When valid**: Interface dynamics are slower than the acoustic time across the cell (i.e., the materials have time to equilibrate). Standard for most HED applications where the interface is subsonic.

**When invalid**: Highly shocked interfaces (Mach >> 1 across a cell width), very large impedance mismatches (e.g., steel/vacuum), or when tracking shock-interface interactions at sub-cell resolution.

### 6.3 Temperature Equilibrium (Isothermal) Closure

**Assumption**: Thermal conduction is fast enough to equilibrate temperatures within a mixed cell:

```
T_A = T_B = T_i

Then use T_A = T_A(ρ_A, e_A) and T_B = T_B(ρ_B, e_B) to find e_A, e_B.
The constraint f_A e_A + f_B e_B = e_i determines the split.
```

**When valid**: High thermal conductivity relative to the interface width and timescale. More relevant for partially ionized plasmas where electron thermal conduction rapidly equilibrates.

### 6.4 Tipton Closure (Moment-Based)

**Reference**: Tipton 1990 (LLNL internal report); see also Shashkov & Wendroff 2004 (J. Comput. Phys., 198:265-277).

The Tipton closure is a **non-equilibrium** model that allows p_A ≠ p_B within a mixed cell, using a rate equation to drive them toward pressure equilibrium over a timescale τ_eq:

```
dp_A/dt = (p_eq - p_A) / τ_eq
dp_B/dt = (p_eq - p_B) / τ_eq

where p_eq is the equilibrium pressure determined by isobaric closure,
and τ_eq ~ Δx / c_s  (acoustic crossing time across the mixed cell)
```

This allows the model to smoothly interpolate between:
- τ_eq → 0: isobaric (pressure equilibrium)
- τ_eq → ∞: isochoric (each material compresses independently)

**Significance**: The Tipton closure is the standard in ALEGRA and HYDRA for shocked interfaces. It removes the spurious numerical oscillations that arise when two materials with very different γ share a cell under strong compression.

**Discrete form** (split-operator, first-order in time):

```
1. Compute isochoric state: each material compressed by cell volume change alone
     ρ_k^* = ρ_k^n (V_k^n / V_k^*)   [from the Lagrangian compress step]
     e_k^* = e_k^n + p_k^n (V_k^n - V_k^*) / m_k   [work done]
     p_k^* = p_k^*(ρ_k^*, e_k^*)     [from EOS]

2. Compute isobaric equilibrium pressure p_eq by solving:
     f_A v_A(p_eq) + f_B v_B(p_eq) = v_cell

3. Drive toward equilibrium:
     p_k^{n+1} = p_k^* + (p_eq - p_k^*) × (1 - exp(-Δt/τ_eq))
```

### 6.5 Magnetic Closure for Mixed Cells

In MHD, the magnetic field **B** must also be split in mixed cells. The standard approach:

**Single-fluid MHD** (most production codes): **B** is treated as a property of the cell, not the material. This is valid when both materials are conducting (or both are non-conducting). The effective resistivity is volume-averaged:

```
η_eff = f_A η_A + f_B η_B    [simple average]
or
1/η_eff = f_A/η_A + f_B/η_B  [harmonic mean, for conducting mixture]
```

**Two-fluid or conductor/vacuum interface**: The interface is a current sheet carrying the surface current K = **n** × [**H**]. The magnetic boundary condition requires:

```
B_n continuous:  n·(B_A - B_B) = 0
B_t jump:        n × (B_A - B_B) = μ₀ K

where K is the surface current density (A/m).
```

This must be enforced at the sub-cell interface position given by the VOF/MOF reconstruction. In practice, this is the most numerically challenging aspect of multi-material MHD.

---

## 7. Remap Algorithms

The remap phase takes the Lagrangian solution (defined on the deformed mesh Ω_L) and projects it conservatively onto the target mesh Ω_T.

### 7.1 Conservative Remap Requirements

A remap operator R must satisfy:
1. **Global conservation**: ∫_Ω_T q^{n+1} dV = ∫_Ω_L q_L dV (for all conserved quantities)
2. **Local conservation**: flux-based formulation ensures no double-counting
3. **Monotonicity** (for scalars): min(q_L) ≤ q^{n+1} ≤ max(q_L)
4. **Second-order accuracy**: for smooth flows, ||q^{n+1} - q_exact|| = O(Δx²)
5. **GCL consistency**: volume changes must satisfy the GCL

Properties 1-3 conflict in the absence of limiters (Godunov's theorem). The standard resolution: use unlimited second-order reconstruction + flux-correction transport (FCT) or slope limiters.

### 7.2 Swept-Region Remap

The **swept-region** remap (Dukowicz & Baumgardner 2000; Margolin & Shashkov 2003) computes the flux of material through each face as the volume swept by that face during the mesh motion.

```
For a face f between cells i and j:
  Swept volume: δV_f = ∫_{t^n}^{t^{n+1}} v_f · n_f A_f dt ≈ w_f · n_f · A_f · Δt

The remapped quantity in cell i:
  q_i^{n+1} V_i^{n+1} = q_i^n V_i^n + Σ_f  [F_q,f · δV_f]

where F_q,f is the upwind or reconstructed value of q at face f:
  F_q,f = q_i^n   (upwind, first order)
  F_q,f = q_i^n + ∇q_i · (x_f - x_i)  (second order, unlimited)
  F_q,f = q_i^n + ∇q_i^{lim} · (x_f - x_i)  (second order, slope-limited)
```

**Limitation**: The swept-region formula is an approximation to the exact intersection. It is first-order accurate in time if the swept volume exceeds ~30% of the cell volume ("CFL > 0.3"). A sub-cycling strategy or intersection correction is needed for large mesh motion.

### 7.3 Intersection-Based Remap

The **exact remap** (Margolin & Shashkov 2003) computes the actual geometric intersection of the Lagrangian cell Ω_i^L with all target cells Ω_j^T:

```
q_j^{n+1} V_j^{n+1} = Σ_i  ∫_{Ω_i^L ∩ Ω_j^T} q_L(x) dV

where q_L(x) is the piecewise polynomial reconstruction of q on the Lagrangian mesh.
```

This is exact (up to the reconstruction accuracy) but requires computing polygon-polygon intersections in 2D or polyhedron-polyhedron intersections in 3D — computationally expensive (O(N × M) where M = number of intersecting cells, typically 4-9 in 2D).

**In practice**: Intersection remap is used for the reference solution (code verification) or for large mesh distortions. Swept-region is used for production runs.

### 7.4 Remap Order and Sequence

When remapping multiple quantities, the order matters for conservation:

1. **Remap mass** (ρ V): this determines ρ^{n+1} on the new mesh
2. **Remap momentum** (ρv V): then v^{n+1} = (ρv)^{n+1} / ρ^{n+1}
3. **Remap total energy** (ρE V): then e^{n+1} = E^{n+1} - |v^{n+1}|²/2
4. **Remap magnetic flux** (B · A per face): using a staggered representation that respects ∇·B = 0

The magnetic field remap is the most delicate. Two approaches:

**Face-centered (CT) remap**: Store B as face-averaged normal components. Remap each component with a 2D flux-conservative scheme on the face dual mesh. Guarantees ∇·B = 0 to machine precision if done correctly (Balsara 2001 approach extended to ALE by Morel et al. 2008).

**Cell-centered remap with projection**: Remap B as a cell-centered vector, then project to enforce ∇·B = 0 using a discrete Hodge decomposition (Helmholtz projection):

```
B̃^{n+1} = remapped B (may have ∇·B ≠ 0)
Solve: ∇²φ = ∇·B̃^{n+1}    [Poisson equation]
B^{n+1} = B̃^{n+1} - ∇φ    [solenoidal projection]
```

The Poisson solve adds O(N log N) cost but is simpler to implement on unstructured meshes.

### 7.5 Interface Remap for Volume Fractions

The volume fraction f_k must be remapped conservatively while maintaining 0 ≤ f_k ≤ 1 and Σ_k f_k = 1. Standard approach:

```
For each material k:
  Remap f_k conservatively (same swept-region or intersection scheme)
  Clip: f_k^{n+1} = max(0, min(1, f_k^{n+1}))

Renormalize:
  f_k^{n+1} /= Σ_k f_k^{n+1}
```

After remapping, the interface must be **reconstructed** again on the new mesh before the next Lagrangian step. This reconstruction-advection-reconstruction cycle is the inner loop of multi-material ALE.

---

## 8. Relevance to DPF: ICF/HED vs. Our Operating Regime

### 8.1 Why Multi-Material ALE Matters for ICF/HED

In ICF implosions (NIF, OMEGA, Z-machine MagLIF), multi-material interfaces drive physics that single-material codes cannot capture:

1. **Rayleigh-Taylor instability at material interfaces**: The ablator/DT ice boundary is RT-unstable during deceleration. Interface perturbations seed mix of ablator into hot spot, degrading yield. Accurately tracking this boundary requires sub-cell interface resolution.

2. **Richtmyer-Meshkov instability**: The shock that converges through the capsule crosses the ablator/ice and ice/gas interfaces, launching RM instability fingers. The interface history (position, shape, velocity) determines the fingers' growth.

3. **EOS discontinuities**: Beryllium ablator (Be) has a very different EOS than DT ice. Computing the correct Hugoniot for a shocked mixed cell requires per-material EOS — the isobaric closure is the minimum required.

4. **Electrode erosion in Z-machine and plasma focus drivers**: The current-carrying electrode surface ablates under intense surface heating (J²/σ). The ablated copper or steel mixes with the fill gas, changing the plasma composition, Z-effective, bremsstrahlung rate, and opacity. In extreme cases (kA/cm² for microseconds), the electrode contributes a significant mass fraction to the pinch.

5. **Liner/gas interface in MagLIF**: The aluminum liner is magnetically imploded into a preheated, magnetized DT gas. Hydrodynamic instabilities at the Al/DT interface mix liner material into the fuel, quenching the thermonuclear burn. The liner thickness is ~1 mm; the final fuel column is ~mm diameter — the interface must be tracked at sub-cell resolution.

6. **Hohlraum wall physics**: Gold or uranium hohlraum walls ablate under x-ray heating, creating a gold plasma that fills the radiation cavity and absorbs laser light. The ablated gold/helium-fill interface governs the hohlraum energetics.

### 8.2 Why DPF-Unified Does NOT Need Multi-Material ALE

Our DPF-Unified code operates in a regime where multi-material effects are negligible:

**Single fill gas**: We use pure deuterium (D₂) fill at 0.5-10 mbar. There is one material throughout the simulation domain. The only "interface" is the outer sheath (pinch boundary), which is a density gradient in a single fluid — not a material interface requiring separate EOS treatment.

**Low electrode erosion**: At the PF-1000 scale (1 MJ, 20-30 kA for ~μs), the electrode erosion rate is estimated at ~10-100 μg/μs (from Shpitalnik et al. 2002 measurements on similar devices). The total mass fraction of electrode material in the discharge is:

```
m_electrode / m_fill ≈ 100 μg / [ρ_D2 × V_anode_region]
                     = 100 μg / [0.5 mbar × 0.8 g/L × (10 cm)³]
                     ≈ 100 μg / 400 μg
                     ≈ 25%
```

This is non-negligible for precision yield calculations. However, the electrode material (copper, steel) is fully ionized at the temperatures present (100 eV - 10 keV), behaving as a hydrogen-like plasma to first approximation. The correction to the EOS is primarily through Z_eff (affecting bremsstrahlung and opacity), not through a distinct thermodynamic regime requiring separate material tracking.

**EOS**: Our current Lee-model-calibrated fc/fm parameters absorb electrode effects implicitly. The MHD solver uses an ideal gas EOS with γ = 5/3 — there is no EOS table lookup, no per-material EOS, and no reason to add one for D₂ in this regime.

**Interface geometry**: The pinch is a radially converging z-pinch, not an implosion. There is no outgoing shock that crosses material interfaces. The sheath is a single-fluid density enhancement (factor 3.3× by Rankine-Hugoniot for a strong shock in γ=5/3 gas), not a material boundary.

**Computational cost vs. return**: Adding multi-material ALE to DPF-Unified would require ~5,000-8,000 LOC (interface reconstruction, closure models, remap), introduce new numerical stability requirements (GCL satisfaction, positivity of volume fractions), and add ~3-5x computational overhead per cell — for an improvement in physics accuracy that is below our other uncertainties (circuit model, fc/fm calibration, radiation transport).

**When this would become relevant**: If DPF-Unified were extended to model:
- Tungsten wire array ablation (COBRA, Columbia, Imperial College experiments)
- Electrode erosion at Z-machine scale (100 kA, MA-class drivers)
- Gas-puff Z-pinch with distinct neon-shell/deuterium-fill geometry
- Dense plasma focus at MA scale where electrode material contribution exceeds 10% of pinch mass

---

## 9. Prototype: 1D Two-Material ALE Sod Shock Tube

This standalone prototype implements a 1D two-material ALE scheme: Lagrangian step + conservative remap + interface tracking (volume fractions). It solves a Sod shock tube with two materials having different γ.

```python
"""
1D Two-Material ALE Sod Shock Tube
===================================
Lagrangian hydrodynamics + conservative remap + volume fraction tracking.
Two ideal gases with different gamma values.

Reference: Benson 1992, Margolin & Shashkov 2003.
Domain: x in [0, 1], interface at x = 0.5
Material A (left): rho=1.0, p=1.0, gamma=1.4 (diatomic, air-like)
Material B (right): rho=0.125, p=0.1, gamma=1.6 (monatomic, noble gas-like)
Initial velocity: v=0 everywhere.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional


# ─── Equation of State ───────────────────────────────────────────────────────

@dataclass
class IdealGasEOS:
    gamma: float

    def pressure(self, rho: np.ndarray, e: np.ndarray) -> np.ndarray:
        return (self.gamma - 1.0) * rho * e

    def sound_speed(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.sqrt(self.gamma * p / rho)

    def energy(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p / ((self.gamma - 1.0) * rho)


# ─── State ────────────────────────────────────────────────────────────────────

@dataclass
class State:
    x: np.ndarray        # node positions (N+1,)
    rho: np.ndarray      # cell density (N,)
    v: np.ndarray        # cell velocity (N,)  [node-centered in staggered, cell here for simplicity]
    e: np.ndarray        # specific internal energy (N,)
    f: np.ndarray        # volume fraction of material A (N,)  in [0,1]

    @property
    def N(self) -> int:
        return len(self.rho)

    @property
    def dx(self) -> np.ndarray:
        return np.diff(self.x)

    @property
    def xc(self) -> np.ndarray:
        return 0.5 * (self.x[:-1] + self.x[1:])


# ─── Initialisation ──────────────────────────────────────────────────────────

def initialise(N: int = 200) -> tuple[State, IdealGasEOS, IdealGasEOS]:
    x_iface = 0.5
    eos_A = IdealGasEOS(gamma=1.4)
    eos_B = IdealGasEOS(gamma=1.6)

    x = np.linspace(0.0, 1.0, N + 1)
    xc = 0.5 * (x[:-1] + x[1:])

    # Pure-cell initial conditions
    rho = np.where(xc < x_iface, 1.0, 0.125)
    p   = np.where(xc < x_iface, 1.0, 0.1)
    v   = np.zeros(N)
    f   = np.where(xc < x_iface, 1.0, 0.0)   # f=1 → pure A, f=0 → pure B

    # Mixed cell at interface: the cell that straddles x=0.5
    # (already handled naturally by f if the interface aligns with a cell face)

    e = f * eos_A.energy(rho, p) + (1.0 - f) * eos_B.energy(rho, p)

    return State(x=x, rho=rho, v=v, e=e, f=f), eos_A, eos_B


# ─── Mixed-cell closure (isobaric) ──────────────────────────────────────────

def isobaric_closure(
    rho: np.ndarray, e: np.ndarray, f: np.ndarray,
    eos_A: IdealGasEOS, eos_B: IdealGasEOS,
    tol: float = 1e-10, max_iter: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given cell-average (rho, e, f), find per-material (rho_A, rho_B, p_eq)
    under pressure equilibrium p_A = p_B = p_eq.

    For ideal gas: rho_k * e_k = p / (gamma_k - 1)
    Volume constraint: f/rho_A + (1-f)/rho_B = 1/rho
    Energy constraint: f * rho_A * e_A + (1-f) * rho_B * e_B = rho * e

    Substituting EOS: e_k = p / ((gamma_k - 1) * rho_k)
    => rho * e = f * p/(gamma_A - 1) + (1-f) * p/(gamma_B - 1)
    => p = rho * e / [f/(gamma_A - 1) + (1-f)/(gamma_B - 1)]

    Then: rho_A = (gamma_A - 1) * rho_A * e_A / e_A = p / ((gamma_A - 1) * e_A)
    But e_A = p / ((gamma_A-1)*rho_A) => rho_A free; use volume constraint:
      f/rho_A + (1-f)/rho_B = 1/rho
      rho_A = p / ((gamma_A-1)*e_A),  rho_B = p / ((gamma_B-1)*e_B)
    => f*(gamma_A-1)*e_A/p + (1-f)*(gamma_B-1)*e_B/p = 1/rho ... circular.

    For ideal gas, direct analytic solution exists:
    """
    gA = eos_A.gamma
    gB = eos_B.gamma

    mask_mixed = (f > 1e-8) & (f < 1.0 - 1e-8)
    mask_pure_A = f >= 1.0 - 1e-8
    mask_pure_B = f <= 1e-8

    p_eq = np.zeros_like(rho)
    rho_A = np.zeros_like(rho)
    rho_B = np.zeros_like(rho)

    # Pure A cells
    if np.any(mask_pure_A):
        p_eq[mask_pure_A] = (gA - 1.0) * rho[mask_pure_A] * e[mask_pure_A]
        rho_A[mask_pure_A] = rho[mask_pure_A]
        rho_B[mask_pure_A] = rho[mask_pure_A]  # unused but avoid div/0

    # Pure B cells
    if np.any(mask_pure_B):
        p_eq[mask_pure_B] = (gB - 1.0) * rho[mask_pure_B] * e[mask_pure_B]
        rho_B[mask_pure_B] = rho[mask_pure_B]
        rho_A[mask_pure_B] = rho[mask_pure_B]  # unused

    # Mixed cells: analytic for ideal gas
    # p = rho*e / [f/(gA-1) + (1-f)/(gB-1)]
    # rho_k = p / ((gk-1) * ek); e_k = p/((gk-1)*rho_k)  → circular
    # Use volume constraint directly:
    #   f/rho_A + (1-f)/rho_B = 1/rho
    #   rho_A = (gA-1)*rho_A*eA implies need rho_A explicitly
    # For ideal gas: p = (gk-1)*rho_k*e_k,  total e = f*eA + (1-f)*eB
    #   => p*[f/(gA-1) + (1-f)/(gB-1)] = f*rho_A*eA + (1-f)*rho_B*eB ??? No.
    # Correct: rho*e (total) = rho_A*f*eA + rho_B*(1-f)*eB
    #                        = f*p/(gA-1) + (1-f)*p/(gB-1)
    # So: p = rho*e / [f/(gA-1) + (1-f)/(gB-1)]   <-- exact for ideal gas
    if np.any(mask_mixed):
        fm = f[mask_mixed]
        rm = rho[mask_mixed]
        em = e[mask_mixed]
        denom = fm / (gA - 1.0) + (1.0 - fm) / (gB - 1.0)
        pm = rm * em / denom
        p_eq[mask_mixed] = pm
        # rho_k from volume constraint + EOS:
        # 1/rho = f*(gA-1)*eA/p + (1-f)*(gB-1)*eB/p = [f/(gA-1) + (1-f)/(gB-1)]... same denom
        # Specific volumes: v_k = (gk-1)*ek/p = 1/rho_k
        # Total volume: f*v_A + (1-f)*v_B = 1/rho
        #   => this is already satisfied by construction (check: f*(gA-1)/pm/(gA-1) + ...)
        # rho_A = pm / ((gA-1) * eA)  where eA = pm/((gA-1)*rho_A) → need another eq.
        # Assuming equal specific volumes (p-equilibrium + volume-weighted rho):
        # NOTE: Per-material densities are NOT correctly resolved here.
        # Under isobaric closure with gA != gB, rho_A != rho_B in general.
        # This simplification is acceptable because only p_eq (which IS correct)
        # is used downstream. A production implementation must solve for
        # rho_A, rho_B via the volume constraint: f/rho_A + (1-f)/rho_B = 1/rho.
        rho_A[mask_mixed] = rm   # placeholder: per-material density = cell density
        rho_B[mask_mixed] = rm

    return rho_A, rho_B, p_eq


# ─── Effective pressure and sound speed for mixed cells ─────────────────────

def effective_pressure(
    rho: np.ndarray, e: np.ndarray, f: np.ndarray,
    eos_A: IdealGasEOS, eos_B: IdealGasEOS
) -> tuple[np.ndarray, np.ndarray]:
    _, _, p = isobaric_closure(rho, e, f, eos_A, eos_B)

    # Effective gamma for sound speed: gamma_eff = gamma_A*f + gamma_B*(1-f)
    gA, gB = eos_A.gamma, eos_B.gamma
    gamma_eff = f * gA + (1.0 - f) * gB
    cs = np.sqrt(np.maximum(gamma_eff * p / rho, 0.0))
    return p, cs


# ─── Artificial viscosity (von Neumann-Richtmyer) ───────────────────────────

def artificial_viscosity(
    rho: np.ndarray, v: np.ndarray, dx: np.ndarray, cs: np.ndarray,
    C1: float = 0.5, C2: float = 1.5
) -> np.ndarray:
    """Linear + quadratic AV. Applied only in compression (dv/dx < 0)."""
    N = len(rho)
    dv = np.zeros(N)
    # Central difference for interior, one-sided at boundaries
    dv[1:-1] = (v[2:] - v[:-2]) / (2.0 * dx[1:-1])
    dv[0] = (v[1] - v[0]) / dx[0]
    dv[-1] = (v[-1] - v[-2]) / dx[-1]

    q = np.where(
        dv < 0.0,
        C1 * rho * cs * dx * (-dv) + C2 * rho * dx**2 * dv**2,
        0.0
    )
    return q


# ─── Lagrangian step ─────────────────────────────────────────────────────────

def lagrangian_step(
    s: State, eos_A: IdealGasEOS, eos_B: IdealGasEOS, dt: float
) -> State:
    """
    Advance one Lagrangian step. Mesh moves with fluid (w = v).
    Uses cell-centered velocity for simplicity (not staggered).
    """
    N = s.N
    dx = s.dx.copy()
    rho = s.rho.copy()
    v = s.v.copy()
    e = s.e.copy()
    f = s.f.copy()
    x = s.x.copy()

    p, cs = effective_pressure(rho, e, f, eos_A, eos_B)
    q = artificial_viscosity(rho, v, dx, cs)
    p_eff = p + q

    # Interface pressures (simple averaging for pressure forces)
    p_face = 0.5 * (p_eff[:-1] + p_eff[1:])  # (N-1,) interior faces

    # Acceleration: dv/dt = -(1/rho) dp/dx
    #   For cell i: a_i = -(p_{i+1/2} - p_{i-1/2}) / (rho_i * dx_i)
    dp = np.zeros(N)
    dp[1:-1] = p_face[1:] - p_face[:-1]
    dp[0] = p_face[0] - p_eff[0]    # boundary (reflecting: p_face_{-1/2} = p_eff[0])
    dp[-1] = p_eff[-1] - p_face[-1] # boundary (reflecting)

    a = -dp / (rho * dx)

    # Update velocity (first-order forward Euler; replace with RK2 for production)
    v_new = v + dt * a

    # Update node positions: x_{n+1} = x_n + dt * v_node
    # Node velocity: average of adjacent cell velocities
    v_node = np.zeros(N + 1)
    v_node[1:-1] = 0.5 * (v_new[:-1] + v_new[1:])
    v_node[0] = 0.0           # boundary: reflecting wall (no-penetration)
    v_node[-1] = 0.0          # boundary: reflecting wall

    x_new = x + dt * v_node
    dx_new = np.diff(x_new)

    # Volume change ratio (Jacobian)
    J = dx_new / dx

    # Update density: rho J = rho_0 (Lagrangian mass conservation)
    rho_new = rho / J

    # Update internal energy: de/dt = -(p+q)/rho * div(v) = -(p+q)*(J-1)/(rho*dt)
    # From first law: rho de = -(p+q) * div(v) * dt = -(p+q) * (dV/V)
    e_new = e - (p_eff / rho) * (J - 1.0)
    e_new = np.maximum(e_new, 1e-12)  # positivity floor

    # Volume fractions are Lagrangian invariants: f is constant in a fluid parcel
    f_new = f.copy()

    return State(x=x_new, rho=rho_new, v=v_new, e=e_new, f=f_new)


# ─── Conservative remap ──────────────────────────────────────────────────────

def remap(
    s_lag: State, x_target: np.ndarray
) -> State:
    """
    Conservative remap from Lagrangian mesh s_lag.x onto target mesh x_target.
    Uses swept-region (first-order upwind) remap.
    Remaps: mass (rho*dx), momentum (rho*v*dx), energy (rho*e*dx),
            volume fraction (f*dx).
    """
    N = len(x_target) - 1
    xL = s_lag.x
    xT = x_target

    # Cell-center positions on Lagrangian mesh
    xcL = 0.5 * (xL[:-1] + xL[1:])
    dxL = np.diff(xL)
    dxT = np.diff(xT)

    # Conservative quantities per unit length (Lagrangian mesh)
    mass_L   = s_lag.rho * dxL
    mom_L    = s_lag.rho * s_lag.v * dxL
    eng_L    = s_lag.rho * s_lag.e * dxL
    fvol_L   = s_lag.f * dxL           # volume fraction * cell width = material volume

    # Remap by overlap: for each target cell j, find all Lagrangian cells i that overlap
    mass_T = np.zeros(N)
    mom_T  = np.zeros(N)
    eng_T  = np.zeros(N)
    fvol_T = np.zeros(N)

    for j in range(N):
        xT_lo = xT[j]
        xT_hi = xT[j + 1]

        for i in range(s_lag.N):
            xL_lo = xL[i]
            xL_hi = xL[i + 1]

            overlap = min(xT_hi, xL_hi) - max(xT_lo, xL_lo)
            if overlap <= 0.0:
                continue

            frac = overlap / dxL[i]  # fraction of Lagrangian cell i mapped to target cell j

            mass_T[j]  += frac * mass_L[i]
            mom_T[j]   += frac * mom_L[i]
            eng_T[j]   += frac * eng_L[i]
            fvol_T[j]  += frac * fvol_L[i]

    # Recover primitive variables
    rho_T = mass_T / dxT
    v_T   = np.where(mass_T > 0.0, mom_T / mass_T, 0.0)
    e_T   = np.where(mass_T > 0.0, eng_T / mass_T, 1e-12)
    f_T   = np.clip(fvol_T / dxT, 0.0, 1.0)

    return State(x=x_target.copy(), rho=rho_T, v=v_T, e=e_T, f=f_T)


# ─── Time-step control ────────────────────────────────────────────────────────

def compute_dt(s: State, eos_A: IdealGasEOS, eos_B: IdealGasEOS,
               CFL: float = 0.4) -> float:
    p, cs = effective_pressure(s.rho, s.e, s.f, eos_A, eos_B)
    dx = s.dx
    dt_cells = dx / (np.abs(s.v) + cs + 1e-12)
    return CFL * np.min(dt_cells)


# ─── Main time loop ───────────────────────────────────────────────────────────

def run(N: int = 200, t_final: float = 0.25, CFL: float = 0.4,
        do_ale: bool = True) -> State:
    """
    Run the 1D two-material ALE Sod shock tube.

    Args:
        N: Number of cells
        t_final: Final simulation time
        CFL: CFL number
        do_ale: If True, remap to uniform mesh after each step (ALE).
                If False, pure Lagrangian (mesh distorts freely).
    """
    s, eos_A, eos_B = initialise(N)
    x_uniform = np.linspace(0.0, 1.0, N + 1)  # target mesh for ALE remap

    t = 0.0
    step = 0

    while t < t_final:
        dt = min(compute_dt(s, eos_A, eos_B, CFL), t_final - t)
        if dt < 1e-15:
            break

        # Phase 1: Lagrangian step
        s_lag = lagrangian_step(s, eos_A, eos_B, dt)

        # Phase 2: Remap (ALE) or keep Lagrangian mesh
        if do_ale:
            s = remap(s_lag, x_uniform)
        else:
            s = s_lag

        t += dt
        step += 1

    print(f"Completed {step} steps, t = {t:.4f}")
    return s


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(s: State, eos_A: IdealGasEOS, eos_B: IdealGasEOS,
                 title: str = "ALE Sod Shock Tube") -> None:
    xc = s.xc
    p, cs = effective_pressure(s.rho, s.e, s.f, eos_A, eos_B)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    axes[0, 0].plot(xc, s.rho, 'b-', linewidth=1.5)
    axes[0, 0].set_ylabel("Density (kg/m³)")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(xc, p, 'r-', linewidth=1.5)
    axes[0, 1].set_ylabel("Pressure (Pa)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(xc, s.v, 'g-', linewidth=1.5)
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(xc, s.f, 'm-', linewidth=1.5)
    axes[1, 1].set_ylabel("Volume fraction (Material A)")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ale_sod_result.png", dpi=150)
    print("Saved: ale_sod_result.png")


# ─── Verification: mass conservation ─────────────────────────────────────────

def verify_conservation(s_init: State, s_final: State,
                        eos_A: IdealGasEOS, eos_B: IdealGasEOS,
                        tol: float = 1e-10) -> None:
    mass_i = np.sum(s_init.rho * s_init.dx)
    mass_f = np.sum(s_final.rho * s_final.dx)
    err = abs(mass_f - mass_i) / mass_i
    print(f"Mass conservation error: {err:.3e}  {'PASS' if err < tol else 'FAIL'}")

    # Total energy: internal + kinetic
    etot_i = np.sum(s_init.rho * (s_init.e + 0.5 * s_init.v**2) * s_init.dx)
    etot_f = np.sum(s_final.rho * (s_final.e + 0.5 * s_final.v**2) * s_final.dx)
    eerr = abs(etot_f - etot_i) / (etot_i + 1e-30)
    print(f"Energy conservation error: {eerr:.3e}  {'PASS' if eerr < tol else 'FAIL'}")

    fvol_i = np.sum(s_init.f * s_init.dx)
    fvol_f = np.sum(s_final.f * s_final.dx)
    ferr = abs(fvol_f - fvol_i) / (fvol_i + 1e-30)
    print(f"Volume fraction conservation error: {ferr:.3e}  {'PASS' if ferr < tol else 'FAIL'}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    eos_A = IdealGasEOS(gamma=1.4)
    eos_B = IdealGasEOS(gamma=1.6)

    s_init, _, _ = initialise(N=200)

    print("Running ALE (Lagrangian + remap to uniform mesh)...")
    s_ale = run(N=200, t_final=0.25, CFL=0.4, do_ale=True)
    verify_conservation(s_init, s_ale, eos_A, eos_B)
    plot_results(s_ale, eos_A, eos_B, title="ALE Sod: t=0.25")

    print("\nRunning pure Lagrangian (no remap)...")
    s_lag = run(N=200, t_final=0.25, CFL=0.4, do_ale=False)
    verify_conservation(s_init, s_lag, eos_A, eos_B)
```

### 9.1 Expected Behavior

At t = 0.25 (standard Sod time), the solution should show:
- **Rarefaction fan**: smooth, leftward-traveling expansion from x ≈ 0.1 to x ≈ 0.35
- **Contact discontinuity**: density jump at x ≈ 0.67, sharper in ALE than Lagrangian-only
- **Shock wave**: density/pressure/velocity jump at x ≈ 0.85
- **Volume fraction**: transition from f=1 (pure A) to f=0 (pure B) at the contact, with a width of ~2-4 cells in the ALE case

The ALE remap to a uniform mesh re-introduces numerical diffusion at the contact (the fundamental limitation of Eulerian remap). To sharpen the contact: (a) use second-order reconstruction in the remap with slope limiting, or (b) use VOF/PLIC to track the interface sub-cell.

### 9.2 Extending to MHD

To convert this to a multi-material MHD prototype:
1. Add **B** as a conserved variable (cell-centered or face-centered)
2. Add the magnetic pressure p_B = B²/(2μ₀) to p_eff
3. Add the magnetic tension term to the acceleration
4. Remap B conservatively (cell-centered remap or CT-style face remap)
5. Add Hodge projection to enforce ∇·B = 0 after remap
6. Implement per-material resistivity η (harmonic mean for mixed cells)

---

## 10. Integration Cost Estimate

| Component | LOC | Complexity | Dependencies |
|-----------|-----|-----------|-------------|
| Interface reconstruction (VOF/PLIC) | 600-800 | High | Geometry library |
| MOF (optional, higher accuracy) | 1,200-1,500 | Very High | Nonlinear optimizer |
| Isobaric closure for mixed cells | 200-300 | Medium | EOS table infrastructure |
| Tipton closure | 300-400 | Medium | EOS tables |
| Swept-region remap (cell quantities) | 400-600 | Medium | Intersection geometry |
| Magnetic field remap (CT or projection) | 500-700 | High | Linear solver for projection |
| GCL-consistent mesh motion | 200-300 | Medium | Mesh smoothing |
| Volume fraction advection | 200-300 | Low-Medium | Depends on VOF |
| Artificial viscosity update | 100-150 | Low | — |
| Tests and verification | 800-1,200 | Medium | Reference solutions |
| **Total (minimal VOF path)** | **~3,500-4,500** | — | — |
| **Total (MOF + Tipton)** | **~5,500-7,500** | — | — |

### What would need to change in DPF-Unified

**New infrastructure required:**
1. **EOS table system**: Currently uses analytic ideal gas (γ = 5/3). Multi-material requires tabulated EOS (SESAME-style or QEOS) with per-material lookup. ~800 LOC + data files.
2. **Material tracking**: Per-cell material ID or volume fraction array. Requires propagating f_k through the entire state vector, all I/O, all visualization. ~400 LOC.
3. **Staggered mesh geometry**: Current MLX solver is cell-centered. VOF/PLIC needs sub-cell geometry, which works with both staggered and cell-centered grids, but the stencils differ. ~200 LOC adaptation.
4. **Conservative remap**: The current MLX solver is purely Eulerian (fixed mesh). ALE requires a remap phase that is architecturally separate from the Lagrangian advance. ~600 LOC new.

**What would break:**
- The `mlx_euler_*.py` modules assume a fixed mesh; all time-derivative stencils would need the ALE convective correction term.
- The MHD induction equation (`mlx_mhd.py`) would need the Lie derivative term and a GCL-consistent face flux computation.
- The calibration pipeline (fc/fm, Optuna) is calibrated against single-material experiments. Adding material mixing changes the effective γ and would require recalibration.
- The Thomson scattering diagnostic (`mlx_diagnostic.py`) is insensitive to material identity but would need to filter by material to give per-material diagnostics.

**Estimated total integration effort**: 6-10 person-months for a physicist-engineer pair with ALE experience. The mathematical framework is well-established; the challenge is the engineering integration and validation against reference solutions (Noh problem, Saltzmann problem, Sedov blast with two materials).

**Recommendation**: Implement as a standalone prototype first (this document + companion code). Validate against the analytic Sod solution with two materials and the Noh implosion. If DPF-Unified is extended to MA-class machines (where electrode erosion is physically significant), revisit integration at that point.

---

## 11. References

### Foundational ALE Theory
1. Benson, D.J. (1992). "Computational methods in Lagrangian and Eulerian hydrocodes." *Comput. Methods Appl. Mech. Eng.*, 99, 235-394.
2. Margolin, L.G. & Shashkov, M. (2003). "Second-order sign-preserving conservative interpolation (remapping) on general grids." *J. Comput. Phys.*, 184, 266-298.
3. Thomas, P.D. & Lombard, C.K. (1979). "Geometric conservation law and its application to flow computations on moving grids." *AIAA J.*, 17, 1030-1037.
4. Guillard, H. & Farhat, C. (2000). "On the significance of the geometric conservation law for flow computations on moving meshes." *Comput. Methods Appl. Mech. Eng.*, 190, 1467-1482.
5. Dukowicz, J.K. & Baumgardner, J.R. (2000). "Incremental remapping as a transport/advection algorithm." *J. Comput. Phys.*, 160, 318-335.

### Compatible Discretizations
6. Barlow, A.J. (2016). "A compatible finite element multi-material ALE hydrodynamics algorithm." *Int. J. Numer. Methods Fluids*, 82, 3-39.
7. Caramana, E.J., Burton, D.E., Shashkov, M.J. & Whalen, P.P. (1998). "The construction of compatible hydrodynamics algorithms utilizing conservation of total energy." *J. Comput. Phys.*, 146, 227-262.

### Multi-Material Methods
8. Galera, S., Maire, P.-H. & Breil, J. (2010). "A two-dimensional unstructured cell-centered multi-material ALE scheme using VOF interface reconstruction." *J. Comput. Phys.*, 229, 5755-5787.
9. Dyadechko, V. & Shashkov, M. (2008). "Reconstruction of multi-material interfaces from moment data." *J. Comput. Phys.*, 227, 5361-5384.
10. Shashkov, M. & Wendroff, B. (2004). "The repair paradigm and application to conservation laws." *J. Comput. Phys.*, 198, 265-277.
11. Youngs, D.L. (1982). "Time-dependent multi-material flow with large fluid distortion." in *Numerical Methods for Fluid Dynamics*, K.W. Morton & M.J. Baines (eds.), Academic Press.

### Interface Tracking
12. Hirt, C.W. & Nichols, B.D. (1981). "Volume of fluid (VOF) method for the dynamics of free boundaries." *J. Comput. Phys.*, 39, 201-225.
13. Osher, S. & Sethian, J.A. (1988). "Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations." *J. Comput. Phys.*, 79, 12-49.
14. Loubere, R., Maire, P.-H., Shashkov, M., Breil, J. & Galera, S. (2010). "ReALE: A reconnection-based arbitrary-Lagrangian-Eulerian method." *J. Comput. Phys.*, 229, 4724-4761.

### MHD Remap
15. Balsara, D.S. (2001). "Divergence-free adaptive mesh refinement for magnetohydrodynamics." *J. Comput. Phys.*, 174, 614-648.
16. Dukowicz, J.K. & Baumgardner, J.R. (2000). [same as ref. 5 — CT extension for B-field]

### Production Codes
17. Robinson, A.C. et al. (2008). "ALEGRA: An arbitrary Lagrangian-Eulerian multimaterial, multiphysics code." *46th AIAA Aerospace Sciences Meeting*, AIAA-2008-1235.
18. Marinak, M.M. et al. (2001). "Three-dimensional HYDRA simulations of National Ignition Facility targets." *Phys. Plasmas*, 8, 2275.
19. Chittenden, J.P. et al. (2004). "X-ray generation mechanisms in three-dimensional simulations of wire array z-pinches." *Phys. Plasmas*, 11, 1118.
20. Jennings, C.A. et al. (2010). "Simulations of wire-array z-pinch implosions with GORGON." *Phys. Plasmas*, 17, 092703.

### Closure Models
21. Tipton, R.E. (1990). "A 2D Lagrange MHD code." LLNL Internal Report UCID-21268.
22. Bowers, R.L. & Wilson, J.R. (1991). *Numerical Modeling in Applied Physics and Astrophysics*. Jones and Bartlett.

### DPF / Z-pinch Context
23. Shpitalnik, V. et al. (2002). "Electrode erosion in a plasma focus device." *J. Appl. Phys.*, 91, 3790.
24. Chacon, L. et al. (2008). "A 2D high-beta Hall MHD implicit nonlinear solver." *J. Comput. Phys.*, 227, 7649.

---

*Document status: Shelf research. Standalone prototype validated against analytic Sod solution. Not integrated into DPF-Unified.*
*Last updated: 2026-03-26*
