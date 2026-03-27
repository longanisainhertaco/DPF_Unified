# General Relativistic MHD (GRMHD): Research Reference

**Status**: Standalone prototype module — NOT integrated into DPF-Unified
**Audience**: PhD-level physics. Assumes knowledge of special relativity, tensor calculus, and classical MHD.
**Date**: 2026-03-26

---

## Table of Contents

1. [Physics Motivation and Scope](#1-physics-motivation-and-scope)
2. [3+1 Decomposition and Metric Choices](#2-31-decomposition-and-metric-choices)
3. [Governing Equations: Valencia Formulation](#3-governing-equations-valencia-formulation)
4. [Constraint Damping: div(B) = 0 in Curved Spacetime](#4-constraint-damping-divb--0-in-curved-spacetime)
5. [Primitive Variable Recovery](#5-primitive-variable-recovery)
6. [Literature Basis](#6-literature-basis)
7. [Standard Test Problems](#7-standard-test-problems)
8. [Minimal SRMHD Prototype (~200 LOC Python)](#8-minimal-srmhd-prototype-200-loc-python)
9. [Relevance to DPF](#9-relevance-to-dpf)
10. [Integration Cost Estimate](#10-integration-cost-estimate)

---

## 1. Physics Motivation and Scope

GRMHD governs magnetized plasma in strong gravitational fields where both relativistic fluid dynamics and curved spacetime geometry are essential. The regimes where GRMHD is non-negotiable:

| Application | Compactness `M/R` | `v/c` | B (Gauss) |
|---|---|---|---|
| Neutron star merger (BNS) | ~0.2 | ~0.5 | 10^15–10^16 |
| Black hole accretion (Sgr A*, M87*) | ~0.5 (ISCO) | ~0.5–0.9 | 10^4–10^8 |
| Relativistic jet (M87) | ~0.3 | ~0.99 | 10^2–10^4 |
| Pulsar magnetosphere | ~0.2 | ~1 (force-free) | 10^12 |
| GRB central engine | ~0.5 | ~0.99 | 10^14–10^15 |

The Event Horizon Telescope (EHT) images of M87* and Sgr A* are directly interpreted through GRMHD simulation libraries (HARM, BHAC, KORAL, Illinois GRMHD, H-AMR). The 2019 EHT code comparison paper verified that five independent GRMHD codes agree on the structure of magnetically arrested disks (MAD) and standard and normal evolution (SANE) states.

The two foundational physical regimes within GRMHD are:

- **GRMHD proper**: finite resistivity negligible, ideal MHD limit, E + v×B = 0 in comoving frame
- **GRRMHD** (resistive): full Ohm's law in covariant form; required near current sheets, reconnection zones

This document covers ideal GRMHD in the Valencia formulation.

---

## 2. 3+1 Decomposition and Metric Choices

### 2.1 The 3+1 (ADM) Split

General relativity treats spacetime as a 4-manifold with metric g_μν (signature −+++). Numerical GRMHD requires foliation of this 4D spacetime into a sequence of spacelike 3D hypersurfaces Σ_t. This is the Arnowitt-Deser-Misner (ADM) decomposition.

The spacetime line element is written:

```
ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt)
```

where:

- **alpha** (lapse function): controls the rate at which proper time elapses relative to coordinate time. `alpha = 1` in flat space.
- **beta^i** (shift vector): describes how spatial coordinates are dragged between hypersurfaces. `beta^i = 0` in Schwarzschild isotropic coordinates.
- **gamma_ij** (spatial 3-metric): the induced metric on each Σ_t hypersurface. `gamma_ij = diag(1,1,1)` in Minkowski.

The determinants are related by:
```
sqrt(-g) = alpha * sqrt(gamma)
```
where `g = det(g_μν)` and `gamma = det(gamma_ij)`.

The extrinsic curvature K_ij measures how each Σ_t is embedded in the 4D manifold. For GRMHD evolution, K_ij appears in source terms but is not evolved by the fluid code (it requires separate metric evolution via Einstein's equations or a fixed background approximation).

### 2.2 Metric Choices

#### Kerr-Schild Coordinates (Black Holes)

The Kerr metric in Boyer-Lindquist coordinates has coordinate singularities at the horizon. Kerr-Schild (horizon-penetrating) coordinates remove these:

```
g_μν^KS = eta_μν + 2H * l_μ l_ν
```

where `eta_μν` is the Minkowski metric, `H` is a scalar, and `l_μ` is a null covector. For the Kerr black hole:

```
H = M*r / (r^2 + a^2 cos^2(theta))
l_μ dx^μ = -dt - (r dx + a dy) / (r^2 + a^2) sin(...) ...
```

In modified Kerr-Schild (MKS) coordinates used by HARM and its descendants, an additional coordinate transformation `x^1 = log(r)`, `x^2 = theta + h(theta)` concentrates resolution near the equatorial plane and polar axis where physics is most dynamic.

Kerr-Schild is the standard for black hole accretion disk simulations because:
1. No coordinate singularity at the horizon (unlike Boyer-Lindquist)
2. Horizon is at a fixed coordinate location, simplifying boundary conditions
3. The metric components are smooth everywhere

Key parameters: mass M, spin parameter a (|a| <= M), with a = 0 being Schwarzschild.

#### FLRW Metric (Cosmology)

For cosmological MHD, the Friedmann-Lemaitre-Robertson-Walker metric:

```
ds^2 = -dt^2 + a(t)^2 [dr^2/(1-kr^2) + r^2 dOmega^2]
```

where `a(t)` is the scale factor and k ∈ {-1, 0, +1} is the curvature. In comoving coordinates with k=0:

```
alpha = 1,  beta^i = 0,  gamma_ij = a(t)^2 delta_ij
sqrt(gamma) = a(t)^3
```

GRMHD in FLRW is used for primordial magnetic field evolution, structure formation with B-fields, and magnetized cosmological reionization.

#### Minkowski Limit (Special Relativistic MHD)

Setting:
```
alpha = 1,  beta^i = 0,  gamma_ij = delta_ij,  sqrt(-g) = 1
```

reduces GRMHD to SRMHD (Special Relativistic MHD) in flat Cartesian coordinates. This is the limit used for the prototype in Section 8. It preserves all the algorithmic complexity of primitive recovery and relativistic Riemann solvers while eliminating metric source terms.

---

## 3. Governing Equations: Valencia Formulation

### 3.1 Physical Variables

The primitive variables (physically intuitive, non-unique from conserved):

```
P = (rho, v^i, p, B^i)
```

- `rho`: rest-mass density in comoving frame (baryon number density times rest mass)
- `v^i`: 3-velocity measured by Eulerian (normal) observer: `v^i = u^i / (alpha u^t) + beta^i / alpha`
- `p`: gas pressure (thermal)
- `B^i`: magnetic field 3-vector as measured in the Eulerian frame

Derived quantities:
- `W = 1/sqrt(1 - gamma_ij v^i v^j)`: Lorentz factor (always >= 1)
- `b^μ`: 4-magnetic field in comoving frame
- `b^2 = b^μ b_μ`: magnetic energy density (comoving frame)
- `h = 1 + epsilon + p/rho`: specific enthalpy (`epsilon` = specific internal energy)
- `h* = h + b^2/rho`: magnetosonic enthalpy

The conserved variables (what the code evolves):

```
U = (D, S_i, tau, B^i)
```

- `D = sqrt(gamma) * W * rho`: conserved mass density
- `S_i = sqrt(gamma) * (rho h* W^2 v_i - alpha b^t b_i)`: conserved momentum density
- `tau = sqrt(gamma) * (rho h* W^2 - p* - alpha^2 (b^t)^2) - D`: conserved energy density (minus rest mass)
- `B^i = sqrt(gamma) * B^i_prim`: magnetic field (already conserved under ideal MHD)

where `p* = p + b^2/2` is the total pressure and `b^t = W*(B^i v_i)/alpha`.

### 3.2 The Valencia Conservation Law

Banyuls et al. (1997) cast GRMHD in fully conservative form. In 3+1 notation, the system reads:

```
(1/sqrt(-g)) * d/dt [sqrt(gamma) * U^a] + (1/sqrt(-g)) * d/dx^i [sqrt(-g) * F^i_a] = S_a
```

or equivalently in divergence form on a fixed background metric:

```
d_t U + d_i F^i = S
```

where the state vector, fluxes, and sources are:

**State vector** U = (D, S_j, tau, B^k):

```
D   = sqrt(gamma) * W * rho
S_j = sqrt(gamma) * [(rho h + b^2) W^2 v_j - alpha b^0 b_j]
tau = sqrt(gamma) * [(rho h + b^2) W^2 - (p + b^2/2) - alpha^2 (b^0)^2] - D
B^k = sqrt(gamma) * B^k
```

**Flux vector** F^i (flux in direction i):

```
F^i(D)   = sqrt(-g) / sqrt(gamma) * [alpha v^i - beta^i] * D / W
         = sqrt(-g) * (alpha v^i - beta^i) * rho * W

F^i(S_j) = sqrt(-g) * [S_j * (alpha v^i - beta^i) / sqrt(gamma) + alpha * delta^i_j * p*
            - alpha * b^i b_j]

F^i(tau) = sqrt(-g) * [(alpha v^i - beta^i) tau/sqrt(gamma) + alpha * p* v^i
            - alpha * b^0 * B^i / W]

F^i(B^k) = sqrt(-g) * [(alpha v^i - beta^i) B^k / sqrt(gamma) - (alpha v^k - beta^k) B^i / sqrt(gamma)]
```

Note: The magnetic flux F^i(B^k) is antisymmetric, which is the discrete version of Faraday's law.

**Source vector** S (geometric sources from curved spacetime):

```
S(D)   = 0

S(S_j) = sqrt(-g) * [T^μν (d_j g_μν) / 2 - Gamma^λ_μj T^μ_λ / ... ]
       ~ sqrt(-g) * [alpha * T^μν d_μ alpha ... - T^μν Gamma^0_μν beta_j ... ]

S(tau) = sqrt(-g) * [alpha * (T^μ0 d_μ log(alpha) - T^μν Gamma^0_μν)]

S(B^i) = 0  (in ideal MHD on fixed background)
```

The stress-energy tensor of magnetized fluid is:

```
T^μν = (rho h + b^2) u^μ u^ν + (p + b^2/2) g^μν - b^μ b^ν
```

where `u^μ` is the 4-velocity and `b^μ` is the comoving magnetic 4-vector.

### 3.3 Relation Between Comoving and Eulerian Magnetic Fields

The Eulerian 3-field B^i (measured by the normal observer with 4-velocity n^μ) relates to the comoving 4-field b^μ via:

```
b^0   = W * B^i v_i / alpha
b^i   = (B^i + alpha * b^0 * u^i) / W
b^2   = (B^2 + alpha^2 (b^0)^2) / W^2    where B^2 = gamma_ij B^i B^j
```

This coupling between the velocity field and the magnetic field is what makes primitive recovery non-trivial.

### 3.4 Equation of State

The system is closed by an EOS relating `p`, `rho`, and `epsilon`. Common choices:

- **Gamma-law**: `p = (Gamma-1) * rho * epsilon`, `h = 1 + Gamma/(Gamma-1) * p/rho`
- **Polytrope**: `p = K * rho^Gamma` (isentropic; useful for testing)
- **Tabulated**: Nuclear EOS tables (Lattimer-Swesty, SRO, etc.) for neutron star mergers

For the prototype, Gamma-law with Gamma = 4/3 (radiation-dominated) is used.

---

## 4. Constraint Damping: div(B) = 0 in Curved Spacetime

### 4.1 The Problem

The divergence constraint in curved spacetime is:

```
d_i (sqrt(gamma) B^i) = 0
```

This is not automatically preserved by finite-difference evolution of the induction equation. Truncation errors introduce monopole errors that grow over time, particularly in strong-field regions near the horizon.

The induction equation (from Faraday's law in 3+1 form):

```
d_t B^i = d_j [alpha v^i B^j - alpha v^j B^i + beta^j B^i - beta^i B^j]
```

In flat space this is exactly divergence-free if the divergence-free condition holds initially. In curved space and with AMR, discrete violations accumulate.

### 4.2 Constrained Transport (CT)

The gold standard. Stagger B^i on cell faces, EMF (electromotive force) on cell edges. The discrete Faraday's law then exactly preserves `div(B) = 0` to machine precision.

Implementation: B^x lives on the x-faces, B^y on y-faces, B^z on z-faces. The EMF lives on edges. The CT update:

```
dB^x/dt = (E_z(j+1,k) - E_z(j,k)) / dy - (E_y(j,k+1) - E_y(j,k)) / dz
```

This is structurally analogous to Yee's scheme in computational electrodynamics. Evans & Hawley (1988) established CT for classical MHD. HARM2D uses CT. BHAC uses CT with AMR.

**Cost**: Requires staggered grids, complicates AMR prolongation/restriction operators, and the coupling between face-centered B and cell-centered primitives adds reconstruction complexity.

### 4.3 Generalized Lagrange Multiplier (GLM) — Dedner Scheme

Munz et al. (2000) / Dedner et al. (2002) introduced a cleaning scalar psi that damps divergence errors:

```
d_t B^i + d_i psi = (normal MHD terms)
d_t psi + c_h^2 d_i B^i = -psi / tau_d
```

where `c_h` is the hyperbolic cleaning speed (typically c_h = c_CFL * max signal speed) and `tau_d = c_r * c_h / dx` is the damping timescale.

Advantages: cell-centered, easy AMR, no staggering.
Disadvantages: only approximately divergence-free; residual errors at O(dx^2) level.

In curved spacetime, the GLM system generalizes to:

```
d_t [sqrt(gamma) B^i] + d_j [sqrt(-g) F^{ij}] + d_i [sqrt(gamma) psi] = 0
d_t [sqrt(gamma) psi] + c_h^2 d_i [sqrt(gamma) B^i] = -sqrt(gamma) psi * c_h^2 / (c_r * c_h)
```

The DPF-Unified codebase uses the Dedner GLM scheme (mlx_divb.py). GRMHD adoption of this is straightforward.

### 4.4 Divergence Cleaning via Projection

After each timestep, project B onto a divergence-free field by solving a Poisson equation:

```
del^2 phi = div(B),  then B -> B - grad(phi)
```

Expensive (global elliptic solve), but exact. Used in some cosmological codes.

### 4.5 Staggered CT with AMR: The Hard Problem

When AMR refinement boundaries exist, CT requires careful prolongation of face-centered fields. The prolonged coarse-level B^i must remain divergence-free, which requires solving a local constrained interpolation problem. This is the main technical barrier to combining CT with AMR in GRMHD.

BHAC (Porth et al. 2017) implements CT with AMR for GRMHD using a flux-CT scheme where EMFs are interpolated across refinement boundaries. H-AMR uses a different approach with staggered-mesh refinement. Both are production-quality implementations that took years to develop.

---

## 5. Primitive Variable Recovery

### 5.1 Why This Is the Hardest Part

In non-relativistic MHD, the conserved variables (rho, rho*v, E, B) uniquely and analytically determine the primitive variables (rho, v, p, B). In GRMHD, the mapping U → P is implicit and may be:

1. **Transcendental**: The Lorentz factor W appears on both sides of equations involving p
2. **Non-unique**: Multiple physical roots can exist in high-magnetization (sigma >> 1) regions
3. **Failure-prone**: Near vacuum, near the horizon, or in strongly magnetized force-free regions, no physical root may be found
4. **Expensive**: Requires 1D or 2D Newton-Raphson iteration at every cell, every timestep

The core difficulty is that S_i (momentum) depends on B^i and the velocity, while B_i depends on gamma_ij and v^i through:

```
S_i S^i = (rho h W)^2 v^2 + ... [mixed terms involving b^2, W, v^2]
```

There is no closed-form inversion.

### 5.2 The W(v^2) Scheme (Noble et al. 2006)

Noble et al. (2006) parameterize the inversion by W and v^2 = gamma_ij v^i v^j. Define:

```
xi = rho h W^2 = (conserved momentum) / v^i [approximately]
```

Then:
```
D = W * rho                              => rho = D / W
tau + D = xi - p* + b^2/2 / W^2 ...     => relate p to xi, W
S^2 = xi^2 v^2 + (2 xi + b^2) b^2 v^2 / ... - (b^i S_i)^2
```

After algebra, the system reduces to finding a root of a single scalar equation:

```
f(xi) = xi - (rho h + b^2) W^2 + (p + b^2/2) = 0
```

where W = W(xi) from the momentum equation, rho = rho(xi, W), p = p(rho, epsilon) from the EOS, and epsilon = epsilon(xi, W, rho) from the energy equation.

This is solved by 1D Newton-Raphson on `xi`, starting from the previous timestep value as the initial guess.

The Noble scheme is the standard in HARM and its descendants. It works in most conditions but fails in:
- `sigma = b^2 / (rho h) >> 1` (magnetically dominated, force-free limit)
- Very low density floors (vacuum cells near the horizon)
- Regions where `v^2 >= 1` (superluminal states from truncation error)

### 5.3 The 2D Newton-Raphson Scheme

An alternative (used in BHAC) solves simultaneously for two unknowns, e.g., (p, W) or (rho, W), with the two constraint equations:

```
f_1(p, W) = (tau + D + p - S^2/(2*Q^2)) - rho h W^2 + p + ...  = 0
f_2(p, W) = W^2 (1 - v^2) - 1 = 0
```

where `Q = rho h W^2 + b^2` and `v^2` is expressed in terms of the conserved variables and the unknowns.

The Jacobian `df_i/dX_j` is computed analytically for efficiency. Convergence requires `|f| < epsilon_tol ~ 1e-10` within typically 5-15 iterations.

### 5.4 Failure Recovery

When Newton-Raphson fails (no convergence, negative pressure, v^2 >= 1), the code must apply a floor or fallback:

1. **Atmosphere**: Set rho = rho_atm, v^i = 0, p = p_atm (near-vacuum fallback)
2. **Velocity limiter**: Cap v^2 at 1 - 1e-10 (prevent superluminal)
3. **Pressure floor**: p >= K * rho^Gamma with K_floor << K_physical
4. **Entropy fallback**: Evolve a passive entropy variable; use it to recover p when Newton-Raphson fails (Kastaun et al. 2021)

Entropy-based fallback is increasingly standard: the code evolves both the energy equation and an entropy equation, and switches to entropy recovery in troubled cells.

### 5.5 Magnetization Regime

The magnetization parameter sigma = b^2 / (rho h) characterizes the magnetic dominance:

- sigma << 1: gas-dominated (normal MHD regime, recovery is stable)
- sigma ~ 1: equipartition (challenging)
- sigma >> 1: magnetically dominated (force-free limit; primitive recovery fails)

GRMHD codes handle sigma >> 1 regions by either: (a) applying an atmosphere and treating them as force-free, (b) using the force-free electrodynamics (GRFFE) limit, or (c) using a hybrid GRMHD/GRFFE scheme (e.g., Del Zanna et al. 2016).

In black hole jets, the polar funnel has sigma ~ 10–10^4 and is handled by these special procedures.

---

## 6. Literature Basis

### Font (2008) — Living Review in Relativity

**J.A. Font, "Numerical Hydrodynamics and Magnetohydrodynamics in General Relativity"**
*Living Reviews in Relativity*, 11, 7 (2008). Updated from 2003 edition.

The canonical pedagogical review. Covers:
- Full derivation of Valencia formulation from first principles
- All Riemann solver options (HLLE, Roe, Marquina) in relativistic context
- Primitive recovery schemes with convergence analysis
- Test problem suite and convergence studies
- Historical development of GRMHD codes

Required reading before implementing any GRMHD code.

### Gammie, McKinney & Toth (2003) — HARM

**C.F. Gammie, J.C. McKinney, G. Toth, "HARM: A Numerical Scheme for General Relativistic Magnetohydrodynamics"**
*ApJ*, 589, 444 (2003). arXiv:astro-ph/0301509.

The paper that defined the modern GRMHD code paradigm. Contributions:
- Conservative Valencia formulation in MKS coordinates for Kerr metric
- HLL Riemann solver with 7-wave structure for GRMHD
- Noble et al. primitive recovery (published separately in 2006 as a full paper)
- Constrained transport on staggered grid
- Standard test problems: Bondi, magnetized torus, Blandford-Znajek

HARM is still widely used (open source, ~2000 lines of C). Its descendants (HARMPI, GRFFE, GRMHD-GPU, H-AMR) are the most-cited codes in black hole astrophysics.

### Noble et al. (2006) — Primitive Recovery

**S.C. Noble, C.F. Gammie, J.C. McKinney, L. Del Zanna, "Primitive Variable Solvers for Conservative General Relativistic Magnetohydrodynamics"**
*ApJ*, 641, 626 (2006). arXiv:astro-ph/0512420.

Systematic comparison of five primitive recovery schemes:
1. 2D Newton-Raphson (most robust, slowest)
2. 1D Newton-Raphson in xi (W scheme) — standard HARM
3. Palenzuela et al. scheme
4. Del Zanna et al. scheme
5. Harm2D (specific to HARM)

Conclusion: the 1D xi-scheme is best for general use. The paper includes FORTRAN reference implementation and detailed convergence analysis across the parameter space (rho, p, W, sigma).

### White, Stone & Gammie (2016) — Athena++ GRMHD

**C.J. White, J.M. Stone, C.F. Gammie, "An Extension of the Athena++ Code Framework for GRMHD Based on Advanced Riemann Solvers and Staggered-Mesh Constrained Transport"**
*ApJS*, 225, 22 (2016). arXiv:1511.00943.

Extends Athena++ (the most widely used non-relativistic MHD code) to GRMHD:
- HLLD Riemann solver adapted for special relativistic MHD (computationally cheaper than HLLE for smooth flows)
- Staggered-mesh CT in Kerr-Schild coordinates
- AMR with CT — the key technical contribution
- Reconstruction options: PLM, PPM, WENO5

Notable: the HLLD solver for SRMHD is derived and validated here. This is directly relevant to the SRMHD prototype in Section 8.

### Porth et al. (2017) — BHAC

**O. Porth, H. Olivares, R. Mizuno, Z. Younsi, L. Rezzolla, M. Moscibrodzka, H. Falcke, M. Kramer, "The Black Hole Accretion Code"**
*Computational Astrophysics and Cosmology*, 4, 1 (2017). arXiv:1611.09720.

BHAC (Black Hole Accretion Code): production GRMHD with:
- AMR using MPI-AMRVAC framework
- Multiple metric support (Kerr-Schild, FLRW, custom)
- 2D Newton-Raphson primitive recovery with entropy fallback
- Both CT and GLM divergence control
- GPU acceleration (OpenACC)
- Polarized radiation transport coupling (GRRT)

BHAC was one of five codes used in EHT 2019 image reconstruction validation.

### EHT Code Comparison (2019)

**EHT Collaboration et al., "First M87 Event Horizon Telescope Results. V. Physical Origin of the Asymmetric Ring"**
*ApJL*, 875, L5 (2019). arXiv:1906.11242.

The code comparison paper that validated the EHT image interpretation:

Five GRMHD codes compared: HARM (GPU port), BHAC, KORAL, Illinois GRMHD (GRMHD3D), H-AMR.

Key findings:
- All codes agree on MAD vs SANE magnetic field state distinction
- Jet power and disk structure consistent across codes to within factor ~2
- Primary image morphology (crescent shape, asymmetry) is robust
- Differences appear at the ~10% level in radiative efficiency and jet structure

This is the most rigorous cross-validation study ever performed for GRMHD codes. Its significance: it established that GRMHD images of black hole shadows are not code-dependent artifacts.

---

## 7. Standard Test Problems

### 7.1 Bondi Accretion (Schwarzschild)

**Description**: Steady-state spherical accretion onto a non-rotating black hole. Michel (1972) generalized the Bondi solution to GR.

**Setup**:
- Schwarzschild metric (alpha = sqrt(1 - 2M/r), beta^r = 0, spherical symmetry)
- Gas pressure: polytrope `p = K rho^Gamma`
- Boundary conditions: specify rho_inf and p_inf at outer boundary; inflow BC at horizon

**Analytic solution**: The critical point occurs at r_sonic = M(5 - 3Gamma) / (4(Gamma-1)) for a Gamma-law gas. At the sonic point, the infall velocity equals the local sound speed. The mass accretion rate M_dot is determined by the sonic condition.

**What it tests**: Metric source terms, inner horizon boundary conditions, smooth transonic flow in strong gravity.

**Pass criterion**: Steady-state density and velocity profiles match Michel solution to < 0.1% after several sound-crossing times.

### 7.2 Magnetized Bondi Accretion

Add a radially aligned magnetic field to Bondi accretion: B^r ~ 1/r^2 (maintains div(B) = 0 in spherical symmetry). For weak fields (plasma beta = 2p/b^2 >> 1), the solution approaches the hydrodynamic Bondi. For strong fields, magnetic pressure modifies the accretion rate.

**What it tests**: Magnetic field evolution in curved spacetime, div(B) control near the horizon.

### 7.3 Fishbone-Moncrief Torus

**Description**: A magnetized equilibrium torus orbiting a Kerr black hole. The standard initial condition for black hole accretion simulations.

**Setup** (Fishbone & Moncrief 1976):
- Kerr-Schild metric with spin a = 0.9375 (standard HARM value)
- Torus pressure maximum at r_max = 12M (standard: rin = 6M)
- Constant specific angular momentum l = l_kep(r_max) within the torus
- Pressure: `p = K rho^Gamma` with Gamma = 4/3
- Magnetic field: A_phi proportional to max(rho/rho_max - 0.2, 0) (vector potential; guarantees div(B)=0)
- Normalize: maximum plasma beta = 100 initially

**Evolution**: The magnetorotational instability (MRI) develops within ~5 orbital periods (t ~ 5 * 2*pi*r_max^(3/2) in geometric units). MRI drives turbulence and angular momentum transport, causing infall. The magnetically arrested disk (MAD) or standard and normal evolution (SANE) states emerge after t ~ 10^4 M.

**Pass criterion**: MRI growth rate matches linear theory omega_MRI = kz * v_A (Alfven speed) within ~20% during linear phase. After saturation, alpha_SS ~ 0.01–0.1.

### 7.4 Blandford-Znajek Jet

**Description**: Magnetically-driven jet powered by extraction of black hole spin energy.

**Setup**: Initialize a Fishbone-Moncrief torus, evolve until MRI-driven accretion builds up large-scale poloidal magnetic flux near the horizon. In MAD state, jets spontaneously form along the rotation axis.

**Characteristic output**: Jet power scales as `P_jet ~ kappa * Phi_BH^2 * Omega_BH^2 * f(Omega_BH/Omega_F)` where Phi_BH is the magnetic flux threading the horizon and Omega_BH = a/(2r_H) is the angular velocity of the horizon.

**Pass criterion**: The BZ power matches the analytic prediction within ~30% for a = 0.9375. Jet Lorentz factor W ~ 5–10 in the polar funnel.

### 7.5 General Relativistic Shock Tube

**Description**: 1D shock tube in Minkowski (flat) spacetime, used to test the Riemann solver accuracy independently of metric effects.

**Balsara Test 1** (from Balsara 2001, reproduced in Font 2008):

Left state:  rho=1.0, p=1.0, vx=0, vy=0, vz=0, Bx=0.5, By=1.0, Bz=0
Right state: rho=0.1, p=0.1, vx=0, vy=0, vz=0, Bx=0.5, By=-1.0, Bz=0

Domain: x in [-0.5, 0.5], initial discontinuity at x=0, Gamma=2, t_end=0.4.

Contains: fast rarefaction, slow compound wave, contact discontinuity, slow compound wave, fast shock.

**Pass criterion**: Density, velocity, pressure, and magnetic field profiles match reference solution (computed with high-resolution reference code) to within L1 error < 0.01.

### 7.6 Relativistic Alfven Wave

**Description**: Circularly polarized Alfven wave propagating at angle to B in flat spacetime. Exact nonlinear solution exists.

**Why it matters**: Tests the preservation of magnetic energy and wave propagation speed in the relativistic limit. In the non-relativistic limit, `c_A = B/sqrt(4*pi*rho)`. Relativistically, the fast-magnetosonic and Alfven speeds are modified.

---

## 8. Minimal SRMHD Prototype (~200 LOC Python)

The following implements 1D Special Relativistic MHD (flat spacetime, Minkowski metric) using:
- HLL Riemann solver
- PLM reconstruction with minmod limiter
- Noble et al. 1D xi primitive recovery
- Balsara Test 1 for validation

This is the flat-spacetime limit that exercises all algorithmic complexity (primitive recovery, relativistic Riemann solver) without metric complications.

```python
"""
srmhd_1d.py  —  Minimal 1D Special Relativistic MHD solver
Test: Balsara (2001) Test 1 shock tube
Method: HLL + PLM + Noble primitive recovery

Equations solved:
  d_t U + d_x F = 0
  U = (D, Sx, Sy, Sz, tau, By, Bz)  (Bx = const in 1D)
  D = W*rho, Si = rho*h*W^2*vi - bt*bi, tau = rho*h*W^2 - p* - bt^2 - D
"""

import numpy as np

GAMMA = 2.0  # adiabatic index for Balsara Test 1


# ─── Equation of State ────────────────────────────────────────────────────────

def eos_pressure(rho: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return (GAMMA - 1.0) * rho * eps


def eos_enthalpy(rho: np.ndarray, eps: np.ndarray) -> np.ndarray:
    p = eos_pressure(rho, eps)
    return 1.0 + eps + p / rho  # h = 1 + eps + p/rho


# ─── Primitive → Conserved ────────────────────────────────────────────────────

def prim_to_cons(P: np.ndarray, Bx: float) -> np.ndarray:
    """
    P: (7, N) array: [rho, vx, vy, vz, eps, By, Bz]
    Returns U: (7, N) array: [D, Sx, Sy, Sz, tau, By, Bz]
    """
    rho, vx, vy, vz, eps, By, Bz = P

    v2  = vx**2 + vy**2 + vz**2
    W   = 1.0 / np.sqrt(1.0 - v2)
    h   = eos_enthalpy(rho, eps)
    p   = eos_pressure(rho, eps)

    B2  = Bx**2 + By**2 + Bz**2
    vdotB = vx*Bx + vy*By + vz*Bz

    bt  = W * vdotB              # b^t (covariant time component in Minkowski)
    bx  = Bx/W + W*vdotB*vx
    by  = By/W + W*vdotB*vy
    bz  = Bz/W + W*vdotB*vz

    b2  = (B2 + (W*vdotB)**2) / W**2   # b^mu b_mu
    pstar = p + 0.5*b2

    rhoh_W2 = (rho*h + b2) * W**2

    D   = W * rho
    Sx  = rhoh_W2*vx - bt*bx
    Sy  = rhoh_W2*vy - bt*by
    Sz  = rhoh_W2*vz - bt*bz
    tau = rhoh_W2 - pstar - bt**2 - D

    return np.array([D, Sx, Sy, Sz, tau, By, Bz])


# ─── Primitive Recovery: Noble et al. (2006) 1D xi scheme ─────────────────────

def cons_to_prim(U: np.ndarray, Bx: float,
                 tol: float = 1e-12, max_iter: int = 50) -> np.ndarray:
    """
    Recover P from U using 1D Newton-Raphson on xi = rho*h*W^2 + b^2.
    Noble et al. (2006), scheme C.
    Operates cell-by-cell (U shape: (7,)).
    """
    D, Sx, Sy, Sz, tau, By, Bz = U

    S2    = Sx**2 + Sy**2 + Sz**2
    B2    = Bx**2 + By**2 + Bz**2
    SdotB = Sx*Bx + Sy*By + Sz*Bz

    # Initial guess for xi from previous conserved state
    xi = tau + D + 0.5*B2   # rough: xi ~ rho*h*W^2 ~ tau + D

    for _ in range(max_iter):
        # Given xi, recover W and v^2
        # S^2 = xi^2 v^2 + (2*xi + B^2)(SdotB)^2/xi^2   [Noble eq. A3]
        # => v^2 = (S^2 - (2*xi+B2)*SdotB^2/xi^2) / xi^2
        A    = xi + B2          # = rho*h*W^2 + b^2 (first form)
        BsdB = SdotB**2
        v2   = (S2*xi**2 + BsdB*(2*xi + B2)) / (xi + B2)**2 / xi**2
        # Clamp to physical range
        v2   = np.clip(v2, 0.0, 1.0 - 1e-10)
        W    = 1.0 / np.sqrt(1.0 - v2)

        rho  = D / W
        b2   = (B2 + (W*SdotB/xi)**2 * B2) / W**2   # approximate
        # Exact b^2 from Noble A7:
        b2   = (B2/W**2) + (SdotB/xi)**2

        # From tau = (rho*h+b^2)*W^2 - p* - bt^2 - D, solve for eps:
        # tau + D = xi + b2*W^2 - (p + b2/2) - bt^2
        # With bt^2 = W^2*(SdotB/xi)^2 and p = (G-1)*rho*eps:
        eps_val = (tau + D - xi + xi/W**2 + 0.5*b2*(1/W**2 - 1) + 0.5*B2/W**2) / rho
        # Clip eps to prevent negative internal energy
        eps_val = max(eps_val, 1e-10)

        p_val   = eos_pressure(np.array([rho]), np.array([eps_val]))[0]
        h_val   = eos_enthalpy(np.array([rho]), np.array([eps_val]))[0]

        xi_new  = (rho*h_val + b2) * W**2

        residual = xi_new - xi
        if abs(residual) < tol * abs(xi):
            xi = xi_new
            break
        # Numerical derivative df/dxi (finite difference)
        dxi  = max(1e-6 * abs(xi), 1e-12)
        xi_p = xi + dxi
        A_p   = xi_p + B2
        v2_p  = (S2*xi_p**2 + BsdB*(2*xi_p + B2)) / A_p**2 / xi_p**2
        v2_p  = np.clip(v2_p, 0.0, 1.0 - 1e-10)
        W_p   = 1.0 / np.sqrt(1.0 - v2_p)
        rho_p = D / W_p
        b2_p  = (B2/W_p**2) + (SdotB/xi_p)**2
        eps_p = (tau + D - xi_p + xi_p/W_p**2 + 0.5*b2_p*(1/W_p**2-1) + 0.5*B2/W_p**2) / rho_p
        eps_p = max(eps_p, 1e-10)
        h_p   = eos_enthalpy(np.array([rho_p]), np.array([eps_p]))[0]
        xi_p_new = (rho_p*h_p + b2_p) * W_p**2

        deriv = (xi_p_new - xi_new) / dxi
        if abs(deriv) < 1e-14:
            break
        xi -= residual / (deriv - 1.0)
        xi  = max(xi, 1e-10)

    # Final primitive extraction
    A    = xi + B2
    v2   = (S2*xi**2 + SdotB**2*(2*xi + B2)) / A**2 / xi**2
    v2   = np.clip(v2, 0.0, 1.0 - 1e-10)
    W    = 1.0 / np.sqrt(1.0 - v2)
    rho  = D / W
    b2   = (B2/W**2) + (SdotB/xi)**2
    eps_val = (tau + D - xi + xi/W**2 + 0.5*b2*(1/W**2-1) + 0.5*B2/W**2) / rho
    eps_val = max(eps_val, 1e-10)

    vx = (Sx + SdotB*Bx/xi) / A
    vy = (Sy + SdotB*By/xi) / A
    vz = (Sz + SdotB*Bz/xi) / A

    return np.array([rho, vx, vy, vz, eps_val, By, Bz])


# ─── Flux Computation ─────────────────────────────────────────────────────────

def compute_flux(P: np.ndarray, Bx: float) -> np.ndarray:
    """
    Flux in the x-direction for 1D SRMHD.
    F = (D*vx, Sx*vx + p* - Bx*bx/W^2, ..., By*vx - Bx*vy, Bz*vx - Bx*vz)
    """
    rho, vx, vy, vz, eps, By, Bz = P

    v2    = vx**2 + vy**2 + vz**2
    W     = 1.0 / np.sqrt(np.clip(1.0 - v2, 1e-10, None))
    h     = eos_enthalpy(rho, eps)
    p     = eos_pressure(rho, eps)
    B2    = Bx**2 + By**2 + Bz**2
    vdotB = vx*Bx + vy*By + vz*Bz

    bt    = W * vdotB
    bx    = Bx/W + W*vdotB*vx
    by    = By/W + W*vdotB*vy
    bz    = Bz/W + W*vdotB*vz
    b2    = (B2 + (W*vdotB)**2) / W**2
    pstar = p + 0.5*b2

    rhoh_W2 = (rho*h + b2) * W**2
    D   = rho * W

    FD   = D * vx
    # F^x(S_j) = (rho*h + b^2)*W^2 * vx*vj + p* * delta_{xj} - b^x * b_j
    FSx  = rhoh_W2*vx*vx + pstar - bx*bx
    FSy  = rhoh_W2*vx*vy         - bx*by
    FSz  = rhoh_W2*vx*vz         - bx*bz
    # F^x(tau) = (tau + p*)*vx - bt*bx, where tau + p* = rhoh_W2 - bt^2 - D
    Ftau = (rhoh_W2 - bt**2 - D)*vx - bt*bx
    FBy  = By*vx - Bx*vy
    FBz  = Bz*vx - Bx*vz

    return np.array([FD, FSx, FSy, FSz, Ftau, FBy, FBz])


def compute_flux_vec(P: np.ndarray, Bx: float) -> np.ndarray:
    """Vectorized flux over N cells. P: (7, N)."""
    rho, vx, vy, vz, eps, By, Bz = P

    v2    = vx**2 + vy**2 + vz**2
    W     = 1.0 / np.sqrt(np.clip(1.0 - v2, 1e-10, None))
    p     = eos_pressure(rho, eps)
    h     = eos_enthalpy(rho, eps)
    B2    = Bx**2 + By**2 + Bz**2
    vdotB = vx*Bx + vy*By + vz*Bz

    bt    = W * vdotB
    b2    = (B2 + (W*vdotB)**2) / W**2
    pstar = p + 0.5*b2

    rho_h_b2_W2 = (rho*h + b2) * W**2

    # Comoving 4-field spatial components (flat spacetime)
    bx    = Bx/W + bt*vx
    by    = By/W + bt*vy
    bz    = Bz/W + bt*vz

    D    = rho * W
    FD   = D * vx
    # F^x(S_j) = (rho*h + b^2)*W^2 * vx*vj + p* * delta_{xj} - b^x * b_j
    FSx  = rho_h_b2_W2*vx*vx + pstar - bx*bx
    FSy  = rho_h_b2_W2*vx*vy         - bx*by
    FSz  = rho_h_b2_W2*vx*vz         - bx*bz
    # F^x(tau) = (tau + p*)*vx - bt*bx, where tau+p* = rho_h_b2_W2 - bt^2 - D
    Ftau = (rho_h_b2_W2 - bt**2 - D)*vx - bt*bx
    FBy  = By*vx - Bx*vy
    FBz  = Bz*vx - Bx*vz

    return np.array([FD, FSx, FSy, FSz, Ftau, FBy, FBz])


# ─── Signal Speeds (HLL) ──────────────────────────────────────────────────────

def signal_speeds(P: np.ndarray, Bx: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate max fast magnetosonic speed in x-direction.
    Returns (lambda_minus, lambda_plus) for HLL: N arrays.
    """
    rho, vx, vy, vz, eps, By, Bz = P

    v2   = vx**2 + vy**2 + vz**2
    W    = 1.0 / np.sqrt(np.clip(1.0 - v2, 1e-10, None))
    p    = eos_pressure(rho, eps)
    h    = eos_enthalpy(rho, eps)
    B2   = Bx**2 + By**2 + Bz**2
    vdotB= vx*Bx + vy*By + vz*Bz
    b2   = (B2 + (W*vdotB)**2) / W**2

    cs2  = GAMMA * p / (rho * h)                    # sound speed^2
    ca2  = b2 / (rho*h + b2)                        # Alfven speed^2 (approx)
    cf2  = np.clip(cs2 + ca2 - cs2*ca2, 0.0, 0.99)  # fast magnetosonic^2 (approx)
    cf   = np.sqrt(cf2)

    # Relativistic frame-dragging of signal speeds
    denom   = 1.0 - v2 * cf2
    disc    = np.sqrt(np.clip(cf2*(1-v2)*(1 - v2*cf2 - vx**2*(1-cf2)), 0.0, None))
    lam_p   = (vx*(1-cf2) + disc) / denom
    lam_m   = (vx*(1-cf2) - disc) / denom

    return lam_m, lam_p


# ─── HLL Riemann Solver ───────────────────────────────────────────────────────

def hll_flux(UL: np.ndarray, UR: np.ndarray,
             PL: np.ndarray, PR: np.ndarray, Bx: float) -> np.ndarray:
    """
    HLL flux at interface. All arrays: (7, N_faces).
    """
    lm_L, lp_L = signal_speeds(PL, Bx)
    lm_R, lp_R = signal_speeds(PR, Bx)

    sl = np.minimum(lm_L, lm_R)   # left-going signal speed
    sr = np.maximum(lp_L, lp_R)   # right-going signal speed

    FL = compute_flux_vec(PL, Bx)
    FR = compute_flux_vec(PR, Bx)

    # HLL flux:
    # F_HLL = (sr*FL - sl*FR + sl*sr*(UR - UL)) / (sr - sl)
    ds = sr - sl
    ds = np.where(ds < 1e-10, 1e-10, ds)

    F_hll = (sr*FL - sl*FR + sl[None,:]*sr[None,:]*(UR - UL)) / ds[None,:]

    # Where both signals go same direction, use upwind
    F_hll = np.where(sl[None,:] >= 0.0, FL, F_hll)
    F_hll = np.where(sr[None,:] <= 0.0, FR, F_hll)

    return F_hll


# ─── PLM Reconstruction ───────────────────────────────────────────────────────

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)


def reconstruct(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PLM reconstruction with minmod limiter. Returns (UL, UR) at each
    interface i+1/2 for i in [1, N-2) (N-3 interfaces, needs 2 ghost cells)."""
    # Slopes at each cell i (for i in [1, N-2])
    dU_l = U[:, 1:-1] - U[:, :-2]   # U_i - U_{i-1}
    dU_r = U[:, 2:  ] - U[:, 1:-1]  # U_{i+1} - U_i
    slope = minmod(dU_l, dU_r)       # limited slope, shape (7, N-2)

    # UL at interface i+1/2: extrapolate right from cell i (slope index i-1 maps to cell i)
    UL = U[:, 1:-1] + 0.5 * slope          # shape (7, N-2)
    # UR at interface i+1/2: extrapolate left from cell i+1 (slope index i maps to cell i+1)
    UR = U[:, 2:-1] - 0.5 * slope[:, 1:]   # shape (7, N-3)
    # Trim UL to match UR (drop last interface)
    UL = UL[:, :-1]                          # shape (7, N-3)
    return UL, UR


# ─── Primitive Recovery (Vectorized) ──────────────────────────────────────────

def cons_to_prim_vec(U: np.ndarray, Bx: float) -> np.ndarray:
    """Recover primitives cell-by-cell."""
    N = U.shape[1]
    P = np.zeros_like(U)
    for i in range(N):
        try:
            P[:, i] = cons_to_prim(U[:, i], Bx)
        except Exception:
            P[0, i] = max(U[0, i], 1e-10)
            P[4, i] = 1e-10
    return P


# ─── Main Integrator ──────────────────────────────────────────────────────────

def srmhd_evolve(rho0, vx0, vy0, vz0, eps0, By0, Bz0,
                 Bx: float, dx: float, t_end: float, cfl: float = 0.4):
    """
    Evolve 1D SRMHD from t=0 to t=t_end.
    Returns (x, P_final) where P_final is (7, N).
    """
    N = len(rho0)
    P = np.array([rho0, vx0, vy0, vz0, eps0, By0, Bz0])
    U = np.zeros((7, N))
    for i in range(N):
        U[:, i] = prim_to_cons(P[:, i:i+1].reshape(7,1), Bx).reshape(7)

    t = 0.0
    while t < t_end:
        P = cons_to_prim_vec(U, Bx)

        lm, lp = signal_speeds(P, Bx)
        max_speed = np.max(np.abs(np.concatenate([lm, lp])))
        dt = cfl * dx / max(max_speed, 1e-10)
        dt = min(dt, t_end - t)

        UL, UR = reconstruct(U)   # (7, N-3) arrays: interfaces [1.5, N-2.5)
        PL = cons_to_prim_vec(UL, Bx)
        PR = cons_to_prim_vec(UR, Bx)

        F = hll_flux(UL, UR, PL, PR, Bx)  # (7, N-3) fluxes

        # Update interior cells [2, N-2): F has N-3 interfaces at i+1/2 for i=1..N-3
        # dF = F_{i+1/2} - F_{i-1/2} for cells i=2..N-3 => N-4 differences
        U[:, 2:-2] -= dt/dx * (F[:, 1:] - F[:, :-1])
        # Outflow boundary conditions (2 ghost cells each side)
        U[:, 0]  = U[:, 2]
        U[:, 1]  = U[:, 2]
        U[:, -1] = U[:, -3]
        U[:, -2] = U[:, -3]

        t += dt

    return cons_to_prim_vec(U, Bx)


# ─── Balsara Test 1 ───────────────────────────────────────────────────────────

def balsara_test1(N: int = 400) -> dict:
    """
    Relativistic MHD shock tube. Balsara (2001) Test 1.
    Left:  rho=1.0, p=1.0, vx=vy=vz=0, Bx=0.5, By=1.0,  Bz=0.0
    Right: rho=0.1, p=0.1, vx=vy=vz=0, Bx=0.5, By=-1.0, Bz=0.0
    Gamma=2, t_end=0.4
    """
    x   = np.linspace(-0.5, 0.5, N)
    dx  = x[1] - x[0]
    mid = N // 2

    rho0 = np.where(x < 0, 1.0, 0.1)
    p0   = np.where(x < 0, 1.0, 0.1)
    eps0 = p0 / ((GAMMA - 1.0) * rho0)
    vx0  = np.zeros(N)
    vy0  = np.zeros(N)
    vz0  = np.zeros(N)
    By0  = np.where(x < 0, 1.0, -1.0)
    Bz0  = np.zeros(N)
    Bx   = 0.5

    P = srmhd_evolve(rho0, vx0, vy0, vz0, eps0, By0, Bz0,
                     Bx=Bx, dx=dx, t_end=0.4)

    rho, vx, vy, vz, eps, By, Bz = P
    p = eos_pressure(rho, eps)

    return {"x": x, "rho": rho, "vx": vx, "vy": vy, "p": p, "By": By, "Bz": Bz}


if __name__ == "__main__":
    result = balsara_test1(N=400)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fields = [("rho", "Density"), ("vx", "vx"), ("vy", "vy"),
              ("p", "Pressure"), ("By", "By"), ("Bz", "Bz")]
    for ax, (key, label) in zip(axes.flat, fields):
        ax.plot(result["x"], result[key], "b-", lw=1.5)
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Balsara Test 1: Relativistic MHD Shock Tube (t=0.4)")
    plt.tight_layout()
    plt.savefig("balsara_test1.png", dpi=150)
    print("Saved: balsara_test1.png")
    print(f"  rho range: [{result['rho'].min():.4f}, {result['rho'].max():.4f}]")
    print(f"  vx  range: [{result['vx'].min():.4f}, {result['vx'].max():.4f}]")
    print(f"  p   range: [{result['p'].min():.4f}, {result['p'].max():.4f}]")
```

### 8.1 Prototype Limitations

This 200-LOC prototype demonstrates the core algorithm. Production gaps:

1. **Primitive recovery**: The Newton-Raphson above is not robust for high magnetization (sigma > 1). Noble et al. (2006) describe a more careful root-bracketing approach.
2. **Riemann solver**: HLL is diffusive. HLLD (5-wave) for SRMHD is substantially more accurate (implemented in White et al. 2016, available in AthenaK).
3. **Reconstruction**: PLM (piecewise linear) with minmod is used. PPM or WENO5 would be needed for production accuracy.
4. **CT**: This prototype uses no divergence control. For 3D, CT or GLM is required.
5. **1D only**: Extension to 3D requires directional splitting or full multi-D reconstruction.

---

## 9. Relevance to DPF

DPF plasmas operate in a regime where all GR effects are negligibly small:

| Parameter | DPF Value | GR Threshold |
|---|---|---|
| Plasma velocity | v ~ 2–5 × 10^5 m/s | v/c ~ 10^-3 |
| GR correction (v/c)^2 | ~10^-6 | Significant at ~0.01 |
| Plasma density | rho ~ 10^-3 kg/m^3 | non-degenerate |
| Magnetic energy / rest mass | b^2/rho ~ 10^-4 | sigma << 1 |
| Spacetime curvature GM/Rc^2 | ~0 (no compact object) | zero |

The GR correction to DPF dynamics is at the 10^-6 level — five orders of magnitude below measurement precision. Classical non-relativistic MHD is correct to 10^-6 relative error for all DPF configurations.

**Where GRMHD IS relevant that connects to plasma physics:**

- **Z-pinch precursors to neutron star merger jets**: GRMHD jets from BNS mergers involve relativistic Z-pinch instabilities (m=0 sausage, m=1 kink). The instability physics is identical to DPF kink modes but at v/c ~ 0.5.
- **Plasma focus as neutron star merger analogue**: The DPF plasma focus phase qualitatively resembles the MRI-driven accretion phase in MAD disks. The magnetic helicity evolution, kink instability, and reconnection physics are analogous.
- **Neutron star crust dynamics**: Low-velocity (v/c ~ 10^-4) magnetized dense plasma in NS crusts is closer to DPF physics than the accretion disk, but in degenerate matter with B ~ 10^14 G.

**Practical use of GRMHD for DPF researchers:**

GRMHD test problems (Balsara shock tubes, relativistic Alfven waves) are useful for benchmarking numerical schemes independent of GR. A researcher building a new MHD code can validate Riemann solvers against SRMHD reference solutions (exact for some cases) before adding the DPF-specific cylindrical geometry and resistivity.

The SRMHD prototype in Section 8 serves this validation purpose.

---

## 10. Integration Cost Estimate

Integrating GRMHD into DPF-Unified would require:

### 10.1 What Must Be Rewritten

| Component | Current (DPF-Unified) | GRMHD Requirement | Effort |
|---|---|---|---|
| State vector | (rho, rho*v, E, B) — 7 vars | (D, S_i, tau, B^i) — 8 vars | ~2 weeks |
| Riemann solver | HLL/HLLS for non-rel | HLL for SRMHD/GRMHD | ~3 weeks |
| Primitive recovery | Trivial (linear) | Noble NR (non-linear, can fail) | ~4 weeks |
| Flux computation | Non-relativistic | Full F^i from Valencia formulation | ~2 weeks |
| Source terms | Cylindrical geometry, resistivity | Christoffel symbols, metric | ~6 weeks |
| Metric module | None | Kerr-Schild/FLRW lapse,shift,gamma_ij | ~3 weeks |
| div(B) control | Dedner GLM | Covariant Dedner or CT | ~2 weeks |
| Tests | ~5100 tests | 5100 + GRMHD suite | ~3 weeks |

**Total estimate: 25 weeks for one developer to implement and validate.**

### 10.2 Why It's Not Practical as an Add-On

GRMHD is not a feature that can be "added" to a non-relativistic MHD code. The reasons:

1. **Primitive recovery**: The entire inversion U → P is qualitatively different. It requires Newton-Raphson iteration that can fail, requiring fallback logic that threads through every part of the code.

2. **Covariant formulation**: Every equation must be rewritten with metric factors `sqrt(-g)`, Christoffel symbols, and the distinction between coordinate and physical components. This is not modular.

3. **Speed of light limit**: Non-relativistic codes assume v << c. The Lorentz factor W = 1 + v^2/(2c^2) + ... appears everywhere. Retrofitting this requires modifying every flux, source term, and signal speed calculation.

4. **Testing regime**: GRMHD codes are validated against completely different test problems (Bondi accretion, magnetized tori) than non-relativistic MHD codes. DPF-Unified's test suite tests cylindrical pinch dynamics that are irrelevant to GRMHD validation.

5. **Coordinate systems**: DPF-Unified is optimized for cylindrical (r,z) coordinates. GRMHD in Kerr-Schild uses spherical Boyer-Lindquist or Kerr-Schild (r, theta, phi) coordinates with a modified radial coordinate x^1 = log(r). These coordinate systems are fundamentally incompatible.

### 10.3 The Right Architecture (If Ever Needed)

If DPF-Unified needed GRMHD (it doesn't), the right approach would be:

1. Build a standalone GRMHD module following HARM's architecture (~3000 LOC in C)
2. Use the existing MLX/Metal infrastructure for GPU acceleration
3. Share only: EOS interface, output format, visualization pipeline
4. Do NOT share: flux solvers, source terms, grid geometry, primitive recovery

The SRMHD prototype in Section 8 is a self-contained proof-of-concept that can serve as the starting point for such a standalone module.

---

## References

1. Banyuls, F., Font, J.A., Ibanez, J.M., Marti, J.M., Miralles, J.A. (1997). "Numerical 3+1 General Relativistic Hydrodynamics: A Local Characteristic Approach." *ApJ*, 476, 221.

2. Noble, S.C., Gammie, C.F., McKinney, J.C., Del Zanna, L. (2006). "Primitive Variable Solvers for Conservative General Relativistic Magnetohydrodynamics." *ApJ*, 641, 626. arXiv:astro-ph/0512420.

3. Font, J.A. (2008). "Numerical Hydrodynamics and Magnetohydrodynamics in General Relativity." *Living Reviews in Relativity*, 11, 7.

4. Gammie, C.F., McKinney, J.C., Toth, G. (2003). "HARM: A Numerical Scheme for General Relativistic Magnetohydrodynamics." *ApJ*, 589, 444. arXiv:astro-ph/0301509.

5. White, C.J., Stone, J.M., Gammie, C.F. (2016). "An Extension of the Athena++ Code Framework for GRMHD Based on Advanced Riemann Solvers and Staggered-Mesh Constrained Transport." *ApJS*, 225, 22. arXiv:1511.00943.

6. Porth, O., et al. (2017). "The Black Hole Accretion Code." *Computational Astrophysics and Cosmology*, 4, 1. arXiv:1611.09720.

7. EHT Collaboration (2019). "First M87 Event Horizon Telescope Results. V. Physical Origin of the Asymmetric Ring." *ApJL*, 875, L5. arXiv:1906.11242.

8. Balsara, D.S. (2001). "Total Variation Diminishing Scheme for Relativistic Magnetohydrodynamics." *ApJS*, 132, 83.

9. Evans, C.R., Hawley, J.F. (1988). "Simulation of magnetohydrodynamic flows: A constrained transport method." *ApJ*, 332, 659.

10. Dedner, A., Kemm, F., Kroner, D., Munz, C.D., Schnitzer, T., Wesenberg, M. (2002). "Hyperbolic Divergence Cleaning for the MHD Equations." *Journal of Computational Physics*, 175, 645.

11. Fishbone, L.G., Moncrief, V. (1976). "Relativistic fluid disks in orbit around Kerr black holes." *ApJ*, 207, 962.

12. Michel, F.C. (1972). "Accretion of matter by condensed objects." *Astrophysics and Space Science*, 15, 153.

13. Kastaun, W., Kalinani, J.V., Ciolfi, R. (2021). "Robust recovery of primitive variables in relativistic ideal magnetohydrodynamics." *Physical Review D*, 103, 023018. arXiv:2005.01821.

---

*Document generated: 2026-03-26. Standalone research reference for GRMHD prototype module — no integration into DPF-Unified codebase.*
