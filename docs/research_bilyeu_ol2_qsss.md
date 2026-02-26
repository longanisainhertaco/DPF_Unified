# Research Report: Bilyeu's Higher-Order CESE Method and QSSS for DPF-Unified

**Date**: 2026-02-25
**Researcher**: Phase Z Research Agent
**Status**: Complete

---

## Executive Summary

This report investigates two areas for potential integration into DPF-Unified:

1. **Dr. David Bilyeu's higher-order CESE method** — a 4th-order space-time Conservation Element and Solution Element scheme that achieves high accuracy with compact stencils, no Riemann solver, no reconstruction, and genuine multidimensionality.

2. **Quasi-Steady State Solutions (QSSS)** for Z-pinch and DPF physics — analytical equilibrium solutions (Bennett pinch, magnetized Noh, shear-stabilized equilibria) that serve as verification benchmarks for MHD simulation codes.

**Key finding**: The CESE method represents a fundamentally different paradigm from our current WENO+HLLD+Riemann approach and could offer significant advantages for DPF cylindrical geometry. QSSS benchmarks provide rigorous verification targets currently absent from DPF-Unified's test suite.

---

## Part 1: Dr. David Bilyeu's Higher-Order CESE Method

### 1.1 Background and Affiliation

- **Name**: David L. Bilyeu
- **University**: The Ohio State University
- **Advisor**: S.-T. John Yu
- **Dissertation (2014)**: "A higher-order conservation element solution element method for solving hyperbolic differential equations on unstructured meshes"
- **Key Publication**: Bilyeu D.L., Yu S.-T.J., Chen Y.-Y., Cambier J.-L., "A Two-Dimensional Fourth-Order Unstructured-Meshed Euler Solver Based on the CESE Method," *Journal of Computational Physics*, Vol. 257, Part A, pp. 981-999 (2014).

### 1.2 The CESE Method: Core Concepts

The Space-Time Conservation Element and Solution Element (CESE) method, originally developed by S.C. Chang at NASA Glenn Research Center, is a fundamentally different approach to solving conservation laws compared to traditional finite-volume/finite-difference methods.

**Core principles:**
1. **Unified space-time treatment**: Time and space are treated on equal footing as coordinates in a higher-dimensional space-time domain
2. **Conservation Elements (CEs)**: Non-overlapping space-time volumes where conservation is enforced via the divergence theorem
3. **Solution Elements (SEs)**: Staggered space-time regions where the solution (including spatial derivatives) is defined
4. **No Riemann solver required**: Interfacial fluxes are computed directly from the space-time integral formulation, not from approximate Riemann solvers
5. **No reconstruction procedure**: Unlike WENO/PPM/ENO, the solution is not reconstructed from cell averages

**Governing equation (integral form)**:

For a conservation law `du/dt + df(u)/dx = 0`, the CESE method enforces:

```
integral over S(CE) of h . dS = 0
```

where `h = (f, u)` is the space-time flux vector and `S(CE)` is the boundary of the conservation element in the (x,t) plane.

### 1.3 Bilyeu's Higher-Order Extension

Bilyeu's key contribution was extending the original 2nd-order CESE scheme to **4th-order accuracy** and developing a **generic nth-order formulation (n >= 4)**.

**Key innovations:**
- Retains the **compact stencil** of the 2nd-order scheme (nearest neighbors only)
- Maintains **CFL stability up to unity** (unlike DG/FR methods where CFL shrinks with order)
- **Explicit time-marching** without directional splitting
- Works on **unstructured meshes** in 2D and 3D

**Comparison with competing methods:**

| Feature | WENO5 (current DPF) | CESE-4th (Bilyeu) | DG/FR |
|---------|---------------------|---------------------|-------|
| Spatial order | 5th | 4th (extensible to nth) | Arbitrary |
| Temporal order | Separate (SSP-RK3) | Simultaneous 4th | Separate |
| Stencil width | 5 cells | Nearest neighbors | Element-local |
| Riemann solver | Required (HLLD) | Not needed | Required |
| Reconstruction | Required (WENO) | Not needed | Not needed (modal) |
| CFL limit vs order | Independent | Near-unity all orders | Shrinks with order |
| Directional splitting | Yes (dimension-split) | No (genuinely multi-D) | No |
| Unstructured mesh | Difficult | Native | Native |
| Implementation complexity | Moderate | Moderate | High |

### 1.4 CESE for MHD

The CESE method has been successfully applied to MHD:

1. **Zhang, Yu, Zha (2006)**: "Solving the MHD equations by the space-time conservation element and solution element method" — First CESE-MHD, 2nd-order.

2. **Jiang, Feng et al. (2018)**: "A high-order CESE scheme with a new divergence-free method for MHD numerical simulation," *Journal of Computational Physics*, Vol. 330, pp. 280-300 — High-order CESE-MHD with WLS-ENO divergence-free technique.

3. **Jiang, Zhang (2025)**: "A New Implementation of a Fourth-Order CESE Scheme for 3D MHD Simulations," *Solar Physics* — Latest 4th-order CESE for 3D MHD.

**Key result from Jiang & Zhang (2025)**: The 4th-order CESE scheme at **low resolution matches the 2nd-order scheme at 4x higher resolution**, using only **~5% of the computing resources**.

**Divergence-free B handling**: The CESE-MHD uses a least-squares-based approach that keeps the magnetic field locally divergence-free while preserving the ENO property, without requiring a separate limiter or constrained transport step.

### 1.5 Relevance to DPF-Unified

**Potential advantages for DPF:**

1. **No directional splitting**: DPF cylindrical geometry (r, theta, z) benefits from genuinely multidimensional schemes that avoid splitting artifacts in the azimuthal direction
2. **Compact stencil**: Simpler boundary condition implementation at electrode surfaces (anode/cathode)
3. **No Riemann solver**: Eliminates the HLLD complexity and float32 NaN issues we've experienced
4. **CFL near unity**: Larger timesteps possible, important for the long-duration DPF discharge (~microseconds, millions of timesteps)
5. **Natural divergence-free B**: Critical for Z-pinch B_theta field topology

**Potential challenges:**

1. **Novel paradigm**: Fundamentally different from our WENO+HLLD approach; requires significant learning curve
2. **No existing Z-pinch/DPF CESE code**: All published CESE-MHD work focuses on solar wind and astrophysics
3. **Limited open-source**: SOLVCON (Python/C++ CESE framework) exists but is 2nd-order only; 4th-order implementations are research codes
4. **Resistive MHD**: Published CESE-MHD papers focus on ideal MHD; resistive extensions (critical for DPF) need development

---

## Part 2: Quasi-Steady State Solutions (QSSS) for Z-Pinch/DPF

### 2.1 Overview

QSSS refers to analytical or semi-analytical equilibrium solutions that MHD codes should reproduce when initialized with or driven toward equilibrium conditions. They serve as **verification benchmarks** — tests where the exact answer is known analytically.

DPF-Unified currently lacks Z-pinch-specific analytical verification benchmarks. Our tests use Sod shock tube, Brio-Wu, and Orszag-Tang (generic MHD tests) but nothing DPF-specific.

### 2.2 Bennett Equilibrium

The foundational Z-pinch equilibrium, first derived by W.H. Bennett (1934).

**Pressure balance**: In a cylindrical Z-pinch carrying axial current I with azimuthal magnetic field B_theta:

```
dp/dr = -J_z * B_theta = -(1/mu_0) * B_theta * (d(r*B_theta)/dr) / r
```

**Bennett density profile**:

```
n(r) = n_0 / (1 + xi^2 * r^2)^2
```

where `xi^2 = b * n_0` is a normalizing constant. The total current satisfies the **Bennett relation**:

```
I^2 = (8 * pi * N * k_B * (T_e + T_i)) / mu_0
```

where N is the line density (particles per unit length).

**Use as verification**: Initialize an MHD code with the Bennett profile and verify it remains in equilibrium (dp/dr = J x B everywhere). Any drift indicates numerical errors in the force balance.

### 2.3 Bennett Vorticity / Shear-Stabilized Equilibria

Recent work (arxiv:2506.05727, 2025) extends Bennett equilibria to include **axial flow shear**:

- Transfers nonlinearity from density to flow velocity
- Maintains identical current density profiles
- Satisfies the shear-flow stabilization criterion: `du_z/dr > 0`
- For cubic temperature `T(r) = C_T * r^3`, yields closed-form analytical solutions

**Verification value**: Tests code handling of realistic nonlinear density + flow profiles simultaneously. The magnetic field has a null on axis and extremum at boundary — geometrically challenging for MHD solvers.

### 2.4 Magnetized Noh Problem

The most rigorous Z-pinch verification benchmark, developed by Velikovich, Giuliani et al. (Physics of Plasmas 19, 012707, 2012).

**Setup**: A radially imploding cylindrical plasma with embedded azimuthal magnetic field stagnates through a strong outward-propagating shock of constant velocity. This extends the classic Noh gasdynamics problem to ideal MHD.

**Key features**:
- **Exact self-similar solution** — analytical expressions for all variables (rho, v, p, B) in terms of similarity variable r/t
- Directly tests cylindrical convergent geometry (unlike Cartesian Sod/Brio-Wu)
- Used to validate MACH2, CERBERUS, and Athena

**Validation results**: CERBERUS showed good agreement; the 3D Cartesian code Athena was subject to a growing instability removable with a diffusive Riemann solver.

**Implementation for DPF-Unified**: This is the highest-priority QSSS benchmark to implement, as it directly tests the cylindrical implosion physics relevant to DPF.

### 2.5 Dynamic Z-Pinch Analytical Model

A comprehensive 4-stage analytical model (Angus et al., arxiv:2505.18067, 2025) provides semi-analytical solutions for the full Z-pinch implosion:

**Stage 1 — Separation** (0 <= t < t_s): Piston and shock form at outer radius
**Stage 2 — Inward** (t_s < t < t_tc): Sheath inertia becomes appreciable; coupled ODEs for piston and shock
**Stage 3 — Compression** (t_tc < t <= t_p): Shock reaches axis, adiabatic compression
**Stage 4 — Pinch** (t > t_p): Stagnation and quasi-equilibrium

**Key geometric relation** (dominant balance):

```
r_p^2 - r_s^2 = ((gamma - 1) / (gamma + 1)) * (r_0^2 - r_s^2)
```

This relates piston radius `r_p` and shock radius `r_s` through the adiabatic index gamma.

**Validation**: Calibrated against COBRA pulsed-power facility data. Single-shot calibration yields realistic parameters (gamma = 1.37 for argon).

### 2.6 Snowplow Quasi-Steady States

Recent work on snowplow models provides additional analytical benchmarks:

1. **Operating Point** (arxiv:2510.11896): Each Z-pinch has an optimal source energy where internal energy conversion peaks. Scaling law: `k_B * T ~ E_0^0.64`.

2. **Temperature Predictions** (arxiv:2506.16551, Cardenas et al.): Snowplow model predicts plasma temperatures comparable to experimental measurements.

3. **Energy-consistent snowplow**: Pressure balance during stagnation provides a quasi-steady state benchmark.

---

## Part 3: Application to DPF-Unified

### 3.1 Current Solver Architecture

**Python engine** (`src/dpf/fluid/mhd_solver.py`):
- Non-conservative pressure formulation (teaching/fallback only)
- WENO5-Z + HLLD + SSP-RK3
- Dimension-split flux computation
- Dedner divergence cleaning

**Metal engine** (`src/dpf/metal/metal_solver.py`):
- Conservative total energy formulation
- PLM/WENO5 + HLL/HLLD + SSP-RK2/RK3
- PyTorch MPS tensors on Apple Metal GPU
- Constrained transport for div(B)

**Snowplow** (`src/dpf/fluid/snowplow.py`):
- 0D Lee model (Phases 2-4)
- Axial rundown + radial compression + pinch
- Circuit-coupled via L_plasma(t)

### 3.2 CESE Integration Assessment

| Component | CESE Benefit | Priority | Effort |
|-----------|-------------|----------|--------|
| Metal MHD solver | Eliminate HLLD NaN issues, genuinely multi-D | High | Very High |
| Python MHD solver | Replace non-conservative formulation | Medium | High |
| Cylindrical geometry | No directional splitting needed | High | High |
| Divergence-free B | Replace CT with integral div-free | Medium | Medium |
| Snowplow coupling | No change (0D model, not grid-based) | None | None |

**Recommendation**: CESE is a **long-term architectural option** (Phase AA+), not a near-term improvement. The 4th-order CESE-MHD code does not exist as open source, and implementing it from scratch would require significant effort (~2000-3000 LOC for a basic 2D cylindrical CESE-MHD).

### 3.3 QSSS Benchmark Implementation Roadmap

These are **immediately actionable** and should be prioritized:

| Benchmark | Priority | Effort | Value |
|-----------|----------|--------|-------|
| Bennett equilibrium test | **P0** (immediate) | Low (~50 LOC) | First Z-pinch-specific V&V test |
| Magnetized Noh problem | **P1** (next phase) | Medium (~200 LOC) | Gold-standard cylindrical convergence test |
| Bennett vorticity equilibria | P2 | Medium (~150 LOC) | Tests flow + B coupling |
| Dynamic Z-pinch stages | P3 | High (~300 LOC) | Full implosion trajectory validation |
| Snowplow operating point | P3 | Low (~100 LOC) | Validates circuit-snowplow coupling |

### 3.4 Priority Ranking of Improvements

1. **Bennett equilibrium verification test** (P0) — Immediate
   - Initialize grid with Bennett profile n(r), B_theta(r), p(r)
   - Run for ~100 timesteps
   - Verify equilibrium maintained: max(|dp/dt|) < tolerance
   - Tests force balance in cylindrical coordinates

2. **Magnetized Noh benchmark** (P1) — Next phase
   - Implement Velikovich & Giuliani (2012) exact solution
   - Converging cylindrical flow with B_theta
   - Compare against analytical density, velocity, B profiles
   - Measures convergence order for cylindrical MHD

3. **CESE feasibility prototype** (P2) — Future
   - Implement 2nd-order CESE for 1D Euler as proof of concept
   - Evaluate stencil simplicity and accuracy vs. current WENO5
   - Assess effort for full cylindrical MHD extension

4. **4th-order CESE-MHD** (P3) — Long-term
   - Full implementation following Jiang & Zhang (2025)
   - Would replace Metal solver internals
   - Requires divergence-free B handling via WLS-ENO

---

## Part 4: Key Equations Summary

### Bennett Equilibrium

```
Pressure balance:  dp/dr + (B_theta / mu_0) * d(r * B_theta) / (r * dr) = 0

Bennett profile:   n(r) = n_0 / (1 + r^2 / a^2)^2

Magnetic field:    B_theta(r) = (mu_0 * I / (2*pi)) * r / (r^2 + a^2)

Pressure:          p(r) = n(r) * k_B * (T_e + T_i)

Bennett relation:  mu_0 * I^2 / (8*pi) = N * k_B * (T_e + T_i)
                   where N = integral(n * 2*pi*r dr) = line density
```

### Magnetized Noh (Velikovich & Giuliani 2012)

```
Self-similar variable:  xi = r / (V_s * t)

Upstream (xi > 1):  rho = rho_0 / xi,  v_r = -V_0,  B_theta = B_0 * r_0 / r

Downstream (xi < 1): rho, v, B from Rankine-Hugoniot jump conditions
                      with azimuthal magnetic field contribution

Shock velocity V_s: determined by upstream conditions and gamma
```

### CESE Space-Time Conservation

```
Space-time flux vector:  h = (f(u), u)  in (x, t) domain

Conservation:  integral_S h . n dS = 0  over CE boundary

Solution expansion in SE:
  u(x, t) = u_j^n + (u_x)_j^n * (x - x_j) + (u_t)_j^n * (t - t_n)
  [2nd order; 4th order adds (u_xx), (u_xt), (u_tt) terms]
```

---

## References

### Bilyeu and CESE Method
1. Bilyeu D.L., "A higher-order conservation element solution element method for solving hyperbolic differential equations on unstructured meshes," PhD Dissertation, Ohio State University (2014).
2. Bilyeu D.L., Yu S.-T.J., Chen Y.-Y., Cambier J.-L., "A Two-Dimensional Fourth-Order Unstructured-Meshed Euler Solver Based on the CESE Method," *J. Comput. Phys.* 257, 981-999 (2014).
3. Chang S.C., "The Method of Space-Time Conservation Element and Solution Element," *J. Comput. Phys.* 119, 295-324 (1995).
4. Jiang C., Feng X., Zhang J., Zhong D., "A high-order CESE scheme with a new divergence-free method for MHD numerical simulation," *J. Comput. Phys.* 330, 280-300 (2018).
5. Jiang C., Zhang L., "A New Implementation of a Fourth-Order CESE Scheme for 3D MHD Simulations," *Solar Physics* (2025).
6. Zhang M., Yu S.-T.J., Lin S.C.H., Chang S.C., Blankson I., "Solving the MHD equations by the space-time conservation element and solution element method," *J. Comput. Phys.* 214, 599-617 (2006).

### QSSS and Z-Pinch Analytical Solutions
7. Bennett W.H., "Magnetically Self-Focussing Streams," *Phys. Rev.* 45, 890 (1934).
8. Velikovich A.L., Giuliani J.L., Zalesak S.T., Thornhill J.W., Gardiner T.A., "Exact self-similar solutions for the magnetized Noh Z pinch problem," *Phys. Plasmas* 19, 012707 (2012).
9. Angus J.R. et al., "A Comprehensive Analytical Model of the Dynamic Z-Pinch," arXiv:2505.18067 (2025).
10. "Bennett Vorticity: A family of nonlinear Shear-Flow Stabilized Z-pinch equilibria," arXiv:2506.05727 (2025).
11. Haines M.G., "Dense plasma in Z-pinches and the plasma focus," *Phil. Trans. R. Soc. A* 300, 649-663 (1981).
12. Lee S., Saw S.H., "Plasma focus ion beam fluence and flux," *Phys. Plasmas* 21, 072501 (2014).
13. Cardenas M., Nettle A., Nunez L., "Snowplow Model Predictions for Plasma Temperature in Z pinch Discharges," arXiv:2506.16551 (2025).
14. "Lagrangian Formulation of the Snowplow Model and Operating Point for Z pinch Devices," arXiv:2510.11896 (2025).
15. "Feasibility and performance of the staged Z-pinch: A one-dimensional study with FLASH and MACH2," *Phys. Plasmas* 31, 042712 (2024).

### Z-Pinch MHD Code Validation
16. "An approach to verification and validation of MHD codes for fusion applications," *Fusion Eng. Des.* 89, 1539-1550 (2014).
17. MARED-U: "Introduction of Radiation MHD Code MARED-U for Z-pinch Driven ICF" (2019).
18. Coppins M., "CGL anisotropic equilibria and stability of a Z pinch" — m=0 stability criterion.

---

## Clarification on "O(L)²" Terminology

**Note**: Extensive web searching did not find a specific numerical method called "O(L)²" or "Optimized Lagrangian-Eulerian squared" in the published literature. The closest match to Dr. David Bilyeu's actual work is the **higher-order CESE (Conservation Element Solution Element) method**, which is sometimes referenced in context with "L2 stability" (energy stability in the L2 norm) and "higher-order" accuracy. The "O(L)²" notation may be:

1. An informal/internal name used within a specific research group
2. A reference to the method achieving "optimal L2" (i.e., L2-stable) convergence rates
3. A conflation with the Purdue L2-stable SBP-SAT methods (Fisher, not Bilyeu)

The research presented here covers Bilyeu's actual published work (higher-order CESE) which represents the most likely candidate for the method in question.
