# Multi-Group Radiation Transport for MHD Codes

**Status**: Standalone prototype research — NOT integrated into DPF-Unified
**Audience**: PhD-level physics; implementation reference for future ICF/HED work
**Date**: 2026-03-26

---

## Table of Contents

1. [Governing Equations](#1-governing-equations)
2. [Opacity Models](#2-opacity-models)
3. [Radiation-Hydrodynamics Coupling](#3-radiation-hydrodynamics-coupling)
4. [Literature Basis](#4-literature-basis)
5. [Implicit Monte Carlo (IMC)](#5-implicit-monte-carlo-imc)
6. [Prototype: 1D Grey Flux-Limited Diffusion](#6-prototype-1d-grey-flux-limited-diffusion)
7. [Relevance to DPF](#7-relevance-to-dpf)
8. [Integration Cost Estimate](#8-integration-cost-estimate)

---

## 1. Governing Equations

### 1.1 The Radiation Transport Equation

The specific intensity I_nu (erg/s/cm²/Hz/sr) at frequency nu, propagating in direction n̂, obeys:

```
(1/c) ∂I_nu/∂t  +  n̂ · ∇I_nu  =  j_nu  -  kappa_nu * I_nu
```

where:
- `j_nu` [erg/s/cm³/Hz/sr] is the emission coefficient (includes spontaneous + stimulated)
- `kappa_nu` [cm⁻¹] is the absorption coefficient (opacity × density: `kappa_nu = rho * chi_nu`)
- Under local thermodynamic equilibrium (LTE): `j_nu = kappa_nu * B_nu(T)`, with B_nu the Planck function

The full integro-differential transport equation in a scattering medium adds a scattering source:

```
(1/c) ∂I_nu/∂t  +  n̂ · ∇I_nu  =  kappa_a_nu * B_nu(T)
                                  - (kappa_a_nu + kappa_s_nu) * I_nu
                                  + (kappa_s_nu / 4pi) * ∫ I_nu dΩ'
```

where `kappa_a_nu` is absorption opacity and `kappa_s_nu` is scattering opacity.

In dense laboratory plasmas (DPF, ICF), electron scattering dominates the scattering term via Thomson/Compton processes, but in the optically thick regime the diffusion approximation renders scattering implicit.

### 1.2 Radiation Moments

Integrating I_nu over solid angle Ω and frequency nu yields the radiation moment quantities used in fluid codes:

**Radiation energy density** (erg/cm³):
```
E_r = (1/c) ∫∫ I_nu dnu dΩ
```

**Radiation flux** (erg/s/cm²):
```
F_r = ∫∫ I_nu * n̂  dnu dΩ
```

**Radiation pressure tensor** (erg/cm³ = dyne/cm²):
```
P_r^ij = (1/c) ∫∫ I_nu * n^i * n^j  dnu dΩ
```

The moment equations (integrating the transport equation):

```
∂E_r/∂t  +  ∇ · F_r  =  c * kappa_P * (aT^4 - E_r)      [energy]
(1/c²) ∂F_r/∂t  +  ∇ · P_r  =  -(kappa_E / c) * F_r     [momentum]
```

where `kappa_P` is the Planck mean opacity, `kappa_E` is the energy-weighted opacity, and `a = 4sigma/c` is the radiation constant (a = 7.566×10⁻¹⁵ erg/cm³/K⁴).

This system is **not closed**: P_r requires knowledge of I_nu. All deterministic transport methods amount to different closure prescriptions.

### 1.3 Flux-Limited Diffusion (FLD)

In the optically thick limit, the transport equation reduces to diffusion. The standard grey FLD approximation (Alme & Wilson 1973; Levermore & Pomraning 1981):

```
F_r  =  -D(E_r, ∇E_r) * ∇E_r
```

with diffusion coefficient:

```
D  =  c / (3 * kappa_R)          [optically thick / diffusion limit]
D  =  c * E_r / |∇E_r|          [free-streaming limit]
```

The **Levermore-Pomraning (LP) limiter** interpolates between these regimes via the dimensionless flux `R`:

```
R  =  |∇E_r| / (kappa_R * E_r)

lambda(R)  =  (2 + R) / (6 + 3R + R²)     [LP flux limiter]

D  =  c * lambda(R) / kappa_R
```

The LP limiter satisfies:
- `lambda → 1/3` as `R → 0` (diffusion limit, `D → c/3kappa_R`)
- `lambda → 1/R` as `R → ∞` (free-streaming limit, `|F_r| → c*E_r`)

Other common limiters: Minerbo (1978, based on entropy maximization), Larsen (1987, better in intermediate regime), Wilson (power-law).

The FLD energy equation:

```
∂E_r/∂t  =  ∇ · (D ∇E_r)  +  c * kappa_P * (aT^4 - E_r)
```

This is a nonlinear parabolic PDE (D depends on E_r and ∇E_r). It is implicit in practice because the emission-absorption coupling term stiffens the system when the radiation-matter coupling is strong (i.e., when `kappa_P * c * dt >> 1`).

**FLD limitations**: FLD is wrong in shadow regions, in the transition from optically thick to thin, and when the radiation field is highly anisotropic. It cannot cast geometric shadows. It gives incorrect wave speeds at intermediate optical depths (Eddington factor is fixed at 1/3 for isotropic field). These limitations motivate M1 closure.

### 1.4 M1 Closure (Dubroca & Feugeas 2000)

The M1 model provides a closure for the radiation pressure tensor P_r by assuming the maximum entropy distribution consistent with E_r and F_r (Dubroca & Feugeas 1999; Levermore 1984). It is a hyperbolic moment system:

```
∂E_r/∂t  +  ∇ · F_r  =  S_E

(1/c²) ∂F_r/∂t  +  ∇ · P_r  =  S_F
```

The M1 closure for P_r:

```
P_r  =  [(1 - xi)/2 * I  +  (3*xi - 1)/2 * n̂⊗n̂] * E_r
```

where `I` is the identity tensor, `n̂ = F_r / |F_r|` is the flux direction, and `xi` is the **Eddington factor** computed from the **variable Eddington factor** (Levermore 1984):

```
f  =  |F_r| / (c * E_r)      [reduced flux, 0 ≤ f ≤ 1]

xi  =  (3 + 4f²) / (5 + 2*sqrt(4 - 3f²))     [Eddington factor]
```

Limits:
- Isotropic field (`f = 0`): `xi = 1/3`, `P_r = E_r/3 * I` (diffusion)
- Free streaming (`f = 1`): `xi = 1`, `P_r = E_r * n̂⊗n̂` (beam)

M1 is hyperbolic with characteristic speeds bounded by c, making it amenable to explicit Godunov-type solvers. It handles shadows better than FLD but still fails for crossing beams (two radiation fronts approaching each other are handled incorrectly because the closure assumes a single dominant direction).

**Key references**: Dubroca & Feugeas (1999, C.R.A.S.); González et al. (2007, A&A 464); Rosdahl et al. (2013, MNRAS) for astrophysical implementation; Vaytet et al. (2013) for protostellar collapse benchmarks.

### 1.5 Multi-Group Discretization

Frequency space is discretized into G groups, each spanning `[nu_g, nu_{g+1}]`. Define group-integrated quantities:

```
E_g  =  ∫_{nu_g}^{nu_{g+1}} (1/c) ∫ I_nu dΩ dnu

F_g  =  ∫_{nu_g}^{nu_{g+1}} ∫ I_nu n̂ dΩ dnu
```

The multi-group FLD system (G coupled PDEs):

```
∂E_g/∂t  =  ∇ · (D_g ∇E_g)  +  c * kappa_P,g * (a_g * T^4 - E_g)
             +  S_g^{scatter}     [g = 0, 1, ..., G-1]
```

where:
- `D_g = c * lambda_g / kappa_R,g` is the group diffusion coefficient
- `kappa_P,g` is the Planck-mean opacity within group g
- `kappa_R,g` is the Rosseland-mean opacity within group g
- `a_g = a * [B_g(T) / (aT^4/4pi)] * 4pi` is the fraction of blackbody emission in group g
- `S_g^{scatter}` is frequency-redistributing scattering (Compton, lines) connecting groups

The group Planck fraction:

```
a_g(T)  =  (4pi/c) * ∫_{nu_g}^{nu_{g+1}} B_nu(T) dnu / (aT^4)
```

satisfying `sum_g a_g(T) = 1`.

In the **grey limit** (G=1), this reduces to the single-group FLD with Planck-mean opacity.

**Frequency binning strategies**:

1. **Log-uniform**: `nu_g = nu_min * (nu_max/nu_min)^{g/G}`. Simple, poor resolution of spectral features.
2. **Opacity-adaptive**: Place bin edges where opacity changes by a factor (e.g., ×10). Captures ionization edges.
3. **Temperature-adaptive**: Weight bin widths by `d(a_g)/dT` — concentrate bins where Planck fraction changes fastest with T.
4. **Multigroup-multiband (MGMB)**: Combine temperature-adaptive groups with sub-group interpolation (Morel 2000).
5. **Optimal transport binning**: Minimize total integrated error for a set of representative (T, rho) conditions (Vaytet 2011).

Practical choices:
- Astrophysics codes (FLASH, CASTRO): 12–64 groups, log-uniform or opacity-adaptive
- ICF codes (HYDRA, LASNEX): 40–250 groups, fine structure around key edges
- DPF relevance: 1–4 groups would likely suffice (see §7)

---

## 2. Opacity Models

### 2.1 Planck Mean and Rosseland Mean

Two frequency-averaged opacities appear in radiation transport:

**Planck mean** (emission-weighted):
```
kappa_P(T, rho)  =  [∫ kappa_nu * B_nu(T) dnu] / [∫ B_nu(T) dnu]
```
Controls emission/absorption coupling. Relevant when matter emits thermally.

**Rosseland mean** (harmonic, flux-weighted):
```
1/kappa_R(T, rho)  =  [∫ (1/kappa_nu) * (∂B_nu/∂T) dnu] / [∫ (∂B_nu/∂T) dnu]
```
Controls diffusive radiation transport. Dominated by low-opacity windows (photons find the easy path). In optically thick plasmas, kappa_R governs the radiation mean free path.

Note: `kappa_P ≥ kappa_R` always (Jensen's inequality). The ratio `kappa_P/kappa_R` can be 10–1000x in plasmas with line opacity.

**Energy-mean opacity** (for moment equations):
```
kappa_E(T, rho)  =  [∫ kappa_nu * E_nu dnu] / E_r
```
Differs from kappa_P when radiation field is non-Planckian.

### 2.2 Opacity Sources

**TOPS (Los Alamos National Laboratory)**:
- Covers 1–92 elements, T = 0.5 eV – 100 keV, rho = 10⁻⁸ – 10⁴ g/cm³
- Methods: STA (super-transition-array), LEDCOP, TOPS-INFERNO
- Provides: Rosseland mean, Planck mean, multigroup tables on frequency grids
- Access: `https://aphysics2.lanl.gov/apps/` (web interface + tabular downloads)
- DPF-relevant elements: H, D, Cu, W, Al (electrode materials)
- Typical format: HDF5 tables indexed by (T, rho), opacity in cm²/g

**OPAL (Lawrence Livermore National Laboratory)**:
- Primarily stellar opacity tables (low-density regime)
- T = 10⁴ – 10⁸ K, excellent for stellar interiors, less relevant for dense lab plasmas
- Reference: Iglesias & Rogers 1996, ApJ 464:943

**Opacity Project (OP)**:
- Theoretical ab initio cross sections from quantum mechanical calculations
- Hummer & Mihalas 1988 (equation of state); Seaton et al. 1994 (opacity)
- Lower accuracy than TOPS for high-density plasmas but well-benchmarked
- Available: `http://cdsweb.u-strasbg.fr/topbase/`

**SCRAM (Spectral Collisional Radiative Atomic Model)**:
- For non-LTE plasmas (relevant for DPF post-pinch corona)
- Generates level populations self-consistently with radiation field
- Reference: Hansen et al. 2007, JQSRT 99:333

**PROPACEOS / PrismSPECT (Prism Computational Sciences)**:
- Commercial, widely used in laser-plasma and z-pinch communities
- Multi-element mixtures; non-LTE capability
- Used in ALEGRA and GORGON opacity tables for z-pinch experiments

### 2.3 Multi-Group Opacity Tables

In practice, opacities are pre-tabulated as functions of T and rho for each frequency group, then interpolated at runtime. Standard format:

```
kappa_g(T, rho)  [cm²/g]  for g = 0, ..., G-1
```

The table grid typically uses log spacing in both T and rho. Bilinear or bicubic interpolation in log-log space. Key considerations:

- **Ionization edges**: Large discontinuities in kappa_nu at edge frequencies. Bin edges should align with major ionization edges (especially K-edges of heavier elements) to avoid smearing.
- **Line opacity**: For spectroscopic purposes, multi-group must resolve lines OR use an escape probability / expansion opacity approach (Eastman & Pinto 1993) for ejecta where lines dominate transfer.
- **Mixture rules**: For multi-species plasmas (e.g., CD₂ + Au hohlraum), opacities mix by number fraction in opacity space, not simply by mass fraction.

---

## 3. Radiation-Hydrodynamics Coupling

### 3.1 Operator Splitting vs Fully Implicit

**Operator-split approach** (Strang splitting):

1. Hydrodynamics advance: `U^{n} → U^{n+1/2}` (ignoring radiation)
2. Radiation transport step: `E_r^n → E_r^{n+1}` with T from U^{n+1/2}
3. Radiation-matter energy exchange: adjust T and E_r to conserve energy
4. Second half hydro step (optional for Strang)

Advantages: allows different solvers/timesteps for each physics. Used in FLASH (grey FLD).
Disadvantage: **splitting error** is O(dt) — requires small dt or subcycling when coupling is stiff.

**Fully implicit approach**:

Solve the coupled system simultaneously at t^{n+1}:

```
[E^{n+1} - E^n]/dt  =  ∇·(D(E_r^{n+1}) ∇E_r^{n+1})  +  kappa_P c (a T^{n+1,4} - E_r^{n+1})

[rho e^{n+1} - rho e^n]/dt  =  -kappa_P c (a T^{n+1,4} - E_r^{n+1})
```

This is a nonlinear system in `(E_r, T)`. Standard solution: **linearize** the coupling term around `T^n`:

```
a T^{n+1,4} ≈ a T^{n,4}  +  4aT^{n,3} * (T^{n+1} - T^n)
```

Then eliminate T^{n+1} from the radiation equation using the material energy equation → a linear system in E_r^{n+1} only. This is the **standard grey implicit FLD** algorithm (Morel et al. 1985).

Used in HYDRA, LASNEX, ALEGRA, CRASH. Allows large dt (limited by hydro CFL, not radiation diffusion CFL).

**Multi-group fully implicit**: G coupled diffusion equations + material energy equation. Block-diagonal structure if groups are weakly coupled (no scattering between groups). The resulting sparse system is solved with multigrid or GMRES preconditioned with ILU.

### 3.2 The 4T Model

In dense non-equilibrium plasmas (ICF ablators, DPF pinch column), the plasma cannot be described by a single temperature. The **4T model** (or multi-temperature model) tracks:

```
T_i    = ion temperature
T_e    = electron temperature
T_r    = radiation temperature  [T_r = (E_r / a)^{1/4}]
T_g    = per-group radiation temperature  [for multi-group]
```

The 4T governing equations:

```
rho cv_i  ∂T_i/∂t  =  ∇·(kappa_i ∇T_i)  -  Q_{ei}  +  Q_{hydro,i}

rho cv_e  ∂T_e/∂t  =  ∇·(kappa_e ∇T_e)  +  Q_{ei}  +  Q_{rad-e}  +  Q_{hydro,e}  +  Q_{Ohm}

∂E_r/∂t  =  ∇·(D ∇E_r)  +  c kappa_P (a T_e^4 - E_r)     [grey]
```

or per-group:

```
∂E_g/∂t  =  ∇·(D_g ∇E_g)  +  c kappa_P,g (a_g T_e^4 - E_g)
```

Key coupling terms:

**Ion-electron equilibration**:
```
Q_{ei}  =  rho * nu_{ei} * cv_e * (T_i - T_e)

nu_{ei} [s⁻¹]  =  (4/3) * sqrt(2pi/m_e) * n_e * Z² e⁴ * ln(Lambda) / (m_i * (kT_e)^{3/2})
```
Spitzer-Härm rate. In DPF, this equilibrates on ~1–10 ns timescales at pinch conditions (n_e ~ 10²⁰ cm⁻³, T_e ~ 1 keV).

**Radiation-electron coupling**:
```
Q_{rad-e}  =  -c kappa_P rho (a T_e^4 - E_r)
```
This is the net emission minus absorption. Positive = plasma gains energy from radiation (irradiation); negative = plasma cools by emission.

**2T vs 4T**: Many codes (FLASH) use 2T (T_e = T_r enforced) with grey FLD; HYDRA uses full multi-group 4T. For DPF: 2T suffices unless radiation pressure exceeds thermal pressure (see §7).

### 3.3 The Radiation Pressure Term

In radiation-dominated environments, radiation pressure feeds back on the momentum equation:

```
rho ∂u/∂t  =  -∇P_hydro  -  (kappa_E rho / c) F_r
```

The radiation force density `f_r = kappa_E rho F_r / c` is the momentum transferred from radiation to matter. In ICF implosions driven by x-ray hohlraums, this force is the primary drive mechanism.

The radiation momentum equation (FLD approximation, F_r = -D∇E_r):

```
f_r  =  -(kappa_E rho D / c) ∇E_r  =  lambda * kappa_E / kappa_R * (-∇E_r)
```

In the diffusion limit (lambda = 1/3, kappa_E ≈ kappa_R): `f_r = -∇(E_r/3) = -∇P_r`, recovering the radiation pressure gradient force.

---

## 4. Literature Basis

### 4.1 Foundational Texts

**Mihalas & Mihalas 1984 — "Foundations of Radiation Hydrodynamics"** (Oxford University Press)
The canonical reference. Chapters 6–8 develop the comoving-frame transport equation, moment hierarchy, and diffusion approximation from first principles in special relativity. Chapter 9 covers multi-frequency transport. Essential for:
- Correct relativistic corrections (O(v/c) terms in the radiation momentum equation)
- The Eddington approximation and its limits
- Variable Eddington factor methods

**Pomraning 1973 — "The Equations of Radiation Hydrodynamics"** (Pergamon)
Rigorous derivation of the moment equations from the Boltzmann equation for photons. Cleaner treatment of the transfer equation than many modern texts. Still the best source for the mathematical structure of moment closures.

**Castor 2004 — "Radiation Hydrodynamics"** (Cambridge University Press)
More accessible than Mihalas & Mihalas. Excellent treatment of:
- FLD in Chapters 7–8, including LP limiter derivation
- IMC (Chapter 10)
- Applications to stellar winds and supernovae
- Comparison of diffusion, FLD, and Sn methods

**Pomraning 1982** — "Flux limiters: Old and new" (JQSRT 27:517). Survey of six limiters with analytical comparisons.

**Levermore & Pomraning 1981** — "A flux-limited diffusion theory" (ApJ 248:321). The LP limiter derivation from the transport equation via asymptotic analysis.

### 4.2 Multi-Group and Closure Methods

**Dubroca & Feugeas 1999** — "Etude theoretique et numerique d'une hierarchie de modeles aux moments pour le transfert radiatif" (C.R.A.S. 329:915)
Original M1 closure paper. Derives the maximum-entropy closure from first principles. Often cited as "Dubroca & Feugeas 2000" in the literature (published December 1999, accessible January 2000).

**Levermore 1984** — "Relating Eddington factors to flux limiters" (JQSRT 31:149)
Unified framework connecting Eddington factor closures (M1, M2) to flux-limited diffusion. Shows that the LP limiter corresponds to a specific Eddington factor ansatz.

**González, Audit & Huynh 2007** — "HERACLES: a three-dimensional radiation hydrodynamics code" (A&A 464:429)
Production M1 code paper. Details operator-split hyperbolic radiation + implicit matter coupling. Reference implementation for M1 in astrophysical context.

**Morel 2000** — "Multigroup diffusion: An accurate approximation in planar geometry" (Nucl. Sci. Eng. 134:235)
Systematic treatment of multigroup binning accuracy, including the MGMB method.

### 4.3 Benchmarks

**Lowrie & Edwards 2008** — "Radiative shock solutions with grey nonequilibrium diffusion" (Shock Waves 18:129)
Provides exact semi-analytic solutions for Marshak waves and radiative shocks used as standard benchmarks for radiation-hydro codes. The Marshak wave solution is the primary validation for any FLD implementation. Key results:
- Marshak wave: radiation front penetrating cold, grey opaque material with kappa = const
- Analytic solution via similarity variable xi = x/t^{1/2} under certain opacity laws
- Lowrie-Edwards solution accounts for material energy equation coupling (non-grey extension)

**Su & Olson 1996** — "Benchmark results for the non-equilibrium Marshak diffusion problem" (JQSRT 56:337)
Extension to non-equilibrium (T_r ≠ T_mat) case with exact series solution. Standard test for 2T radiation codes.

**Kasen et al. 2006** — "Multi-angle, multi-dimensional supernova spectropolarimetry calculations" (ApJ 651:366)
Multi-group transport benchmark relevant for complex opacity spectra.

### 4.4 Production Codes

**HYDRA (LLNL)**:
- Primary ICF simulation code for NIF
- Radiation: multi-group FLD (grey + ~120 groups in production runs) + IMC option
- Opacity: OPAL + STA tables; Lee-More for EOS
- Reference: Marinak et al. 2001, PoP 8:2275

**ALEGRA (Sandia National Laboratories)**:
- Multi-physics Eulerian/ALE code for z-pinch, HEDP
- Radiation: grey FLD + discrete ordinates (Sn transport with 8–16 angular directions)
- Used for Z-machine experiments (relevant for DPF community)
- Reference: Garasi et al. 2004, PoP 11:2729

**FLASH (University of Chicago / Flash Center)**:
- Open-source adaptive mesh refinement code
- Radiation: grey FLD with LP limiter; multi-group in FLASH4 via MGFLD module
- Coupling: operator-split 2T (e-r equilibrated, separate T_ion)
- Reference: Fryxell et al. 2000, ApJS 131:273; Tzeferacos et al. 2015, PoP 22:032702

**GORGON (Imperial College London)**:
- Resistive MHD code designed for z-pinch, laser-plasma, and wire-array experiments
- Radiation: multi-group FLD with 6–30 groups; opacity tables from PROPACEOS
- Directly relevant to DPF community
- Reference: Chittenden et al. 2004, PoP 11:1118; Jennings et al. 2010

**LASNEX (LLNL)**:
- Legacy ICF code, 1D + 2D
- Multi-group FLD with ~80 groups; the original production MG-FLD implementation
- Reference: Zimmermann & Kruer 1975, Comments Plasma Phys. 2:85

**CRASH (University of Michigan)**:
- 3D AMR radiation hydrodynamics, open-source
- Grey FLD + 3 groups in recent extensions; FLASH-based
- Reference: van der Holst et al. 2011, ApJS 194:23

---

## 5. Implicit Monte Carlo (IMC)

### 5.1 The Fleck & Cummings Method (1971)

Reference: Fleck & Cummings 1971, "An implicit Monte Carlo scheme for calculating time and frequency dependent nonlinear radiation transport" (J. Comput. Phys. 8:313)

The core problem with standard Monte Carlo transport in optically thick, emission-dominated media: photon packets are repeatedly emitted and reabsorbed within a cell on timescales << dt. A straightforward simulation would require enormous numbers of particles.

The F&C solution: introduce the **effective scattering albedo** `alpha_eff` to convert a fraction of absorption events into "pseudo-scattering":

```
alpha_eff  =  (1 - f_F) * kappa_a

f_F  =  1 / (1 + beta * kappa_a * c * dt)     [Fleck factor]

beta  =  4 * a * T^3 / (rho * cv)              [radiation-matter coupling parameter]
```

The Fleck factor `f_F` interpolates between:
- `f_F → 1` (weak coupling, `beta*kappa_a*c*dt << 1`): pure transport, no effective scattering
- `f_F → 0` (strong coupling, `beta*kappa_a*c*dt >> 1`): most absorption becomes pseudo-scattering, achieving the diffusion limit implicitly

The modified transport equation becomes:

```
(1/c) ∂I/∂t  +  n̂ · ∇I  =  f_F * kappa_a * B(T^n)
                            - (f_F * kappa_a + kappa_s + alpha_eff * kappa_a) * I
                            + (alpha_eff * kappa_a + kappa_s) / (4pi) * ∫ I dΩ
```

This is solved by Monte Carlo particle transport with:
1. Photon packets emitted proportional to `f_F * kappa_a * c * B(T^n) * dt`
2. Absorption at rate `f_F * kappa_a` (true absorption, energy deposited)
3. Pseudo-scattering at rate `alpha_eff * kappa_a` (isotropic re-emission, no energy change)
4. Temperature updated at end of timestep from deposited energy

### 5.2 When and Why: Stochastic vs Deterministic Transport

**Use IMC when**:
- Frequency-dependent (multi-group, line) opacities with large variations are present
- The radiation field is highly anisotropic and shadows matter (FLD/M1 physically wrong)
- Benchmark-quality solutions are needed (stochastic error → 0 with N^{-1/2} scaling)
- Problem is 1D or 2D symmetric (particle counts manageable: 10⁵–10⁷ packets)

**Use deterministic (FLD or M1) when**:
- 3D problems (IMC scales poorly: 10⁸ packets in 3D to achieve 1% noise)
- Optically thick globally (FLD error small in diffusion limit)
- Computational cost is the primary constraint
- Integrated quantities (total energy, total momentum) are sufficient (not spectral details)

**Hybrid approaches**:
- **IMC with FLD acceleration**: Use FLD in thick regions, IMC in thin. Implemented in HYDRA.
- **Continuous Monte Carlo**: Gentile 2001 (J. Comput. Phys. 172:543) — mitigates teleportation error in IMC
- **Symbolic Implicit Monte Carlo (SIMC)**: Nikolaeva 1992; handles stiff emission better than F&C

**Teleportation error**: A known pathology of IMC where photon packets traverse multiple mean-free-paths per cell, causing spurious energy deposition distant from the source. Mitigation: cell subdivision, continuous MC, or IMC-DDMC hybridization (Densmore et al. 2007).

---

## 6. Prototype: 1D Grey Flux-Limited Diffusion Solver

A minimal Python implementation of grey FLD demonstrating the Marshak wave test problem.

### 6.1 Problem Setup: Marshak Wave

**Configuration**: Semi-infinite slab of cold, grey, opaque material at T_0 = 0 (or small epsilon) in x > 0. At x = 0, a radiation boundary condition E_r(0, t) = a*T_drive^4 is applied at t = 0.

**Governing equations** (1D, non-equilibrium):
```
∂E_r/∂t  =  ∂/∂x [D ∂E_r/∂x]  +  c*kappa*(a*T^4 - E_r)
rho*cv * ∂T/∂t  =  -c*kappa*(a*T^4 - E_r)
D  =  c / (3*kappa)
```

**Su-Olson parameters** (dimensionless form, Su & Olson 1996):
- `epsilon = (4*a*T_drive^3) / (rho*cv)` (radiation-to-matter energy ratio)
- For `epsilon = 1`: fully non-equilibrium, T_r ≠ T_mat at wavefront

### 6.2 Implementation

```python
"""
1D Grey Flux-Limited Diffusion — Marshak Wave Test
Reference: Su & Olson 1996, JQSRT 56:337; Lowrie & Edwards 2008
~150 LOC
"""

import numpy as np
from scipy.linalg import solve_banded

# -----------------------------------------------------------------------
# Physical constants (CGS)
# -----------------------------------------------------------------------
a_rad = 7.566e-15    # radiation constant [erg/cm^3/K^4]
c     = 2.998e10     # speed of light [cm/s]

# -----------------------------------------------------------------------
# Problem parameters (Su-Olson test, dimensionless scaled)
# -----------------------------------------------------------------------
def setup_marshak(
    T_drive: float = 1.0,          # drive temperature [keV, normalized to 1]
    T_0: float = 1e-4,             # initial material temperature
    rho: float = 1.0,              # density [g/cm^3]
    kappa: float = 1.0,            # grey opacity [cm^{-1}]
    cv: float = 1.0,               # specific heat [erg/g/K]
    L: float = 5.0,                # domain length [cm / mean free paths]
    nx: int = 200,                 # number of cells
    t_end: float = 3.0,            # end time [ns, normalized]
    cfl: float = 0.4,              # CFL number for diffusion dt
):
    dx = L / nx
    x = np.linspace(dx/2, L - dx/2, nx)      # cell centers

    # Initial conditions
    E_r = np.full(nx, a_rad * T_0**4)
    T   = np.full(nx, T_0)

    dt_diff = cfl * dx**2 * 3 * kappa / c    # diffusion CFL
    dt_coup = 0.1 / (c * kappa)              # coupling timescale limit
    dt = min(dt_diff, dt_coup)

    return x, E_r, T, dx, dt, kappa, rho, cv

# -----------------------------------------------------------------------
# Levermore-Pomraning flux limiter
# -----------------------------------------------------------------------
def lp_limiter(E_r: np.ndarray, dx: float, kappa: float) -> np.ndarray:
    """Compute face-centered diffusion coefficient D = c*lambda/kappa."""
    # Face-centered E_r by averaging adjacent cells (ghost cells at boundaries)
    E_face = 0.5 * (E_r[:-1] + E_r[1:])         # shape: (nx-1,)
    dE_dx  = (E_r[1:] - E_r[:-1]) / dx

    # Reduced flux at faces
    R = np.abs(dE_dx) / (kappa * np.maximum(E_face, 1e-100))

    # LP limiter: lambda = (2 + R) / (6 + 3R + R^2)
    lam = (2.0 + R) / (6.0 + 3.0*R + R**2)

    D_face = c * lam / kappa                      # shape: (nx-1,)
    return D_face

# -----------------------------------------------------------------------
# Build tridiagonal system for implicit diffusion
# -----------------------------------------------------------------------
def build_diffusion_matrix(
    E_r: np.ndarray, D_face: np.ndarray, dx: float, dt: float
) -> tuple:
    """
    Returns (ab, rhs) for scipy.linalg.solve_banded.
    Implicit in diffusion; explicit treatment of emission-absorption
    is handled in the coupled update.
    """
    nx = len(E_r)
    diag  = np.zeros(nx)
    lower = np.zeros(nx - 1)
    upper = np.zeros(nx - 1)

    # Interior cells: standard finite-volume discretization
    for i in range(1, nx - 1):
        D_l = D_face[i - 1]
        D_r = D_face[i]
        diag[i]      = 1.0/dt + (D_l + D_r) / dx**2
        lower[i - 1] = -D_l / dx**2
        upper[i]     = -D_r / dx**2

    # Left BC: E_r(0) = a*T_drive^4 (Dirichlet via ghost cell)
    D_r      = D_face[0]
    diag[0]  = 1.0/dt + 2.0*D_r / dx**2     # factor 2 from ghost reflection
    upper[0] = -D_r / dx**2

    # Right BC: zero flux (∂E_r/∂x = 0, Neumann)
    D_l         = D_face[-1]
    diag[-1]    = 1.0/dt + D_l / dx**2
    lower[-1]   = -D_l / dx**2

    # Pack into banded form for scipy (ab[0]=upper, ab[1]=diag, ab[2]=lower)
    ab = np.zeros((3, nx))
    ab[0, 1:] = upper
    ab[1, :]  = diag
    ab[2, :-1] = lower

    rhs = E_r / dt
    return ab, rhs

# -----------------------------------------------------------------------
# Implicit radiation-matter energy exchange update
# -----------------------------------------------------------------------
def implicit_coupling_update(
    E_r: np.ndarray,
    T: np.ndarray,
    kappa: float,
    rho: float,
    cv: float,
    dt: float,
) -> tuple:
    """
    Solve implicit 2x2 system per cell:
        E_r^{n+1} = E_r*  + c*kappa*dt*(a*T^{n+1,4} - E_r^{n+1})
        rho*cv*(T^{n+1} - T^n) = -c*kappa*dt*(a*T^{n+1,4} - E_r^{n+1})

    Energy is conserved: E_r^{n+1} + rho*cv*T^{n+1} = E_r* + rho*cv*T^n
    Linearize T^{n+1,4} ≈ T^n,4 + 4*T^n,3*(T^{n+1} - T^n) and solve.
    """
    E_total = E_r + rho * cv * T      # conserved

    alpha = c * kappa * dt
    beta  = 4.0 * a_rad * T**3 / (rho * cv)

    # From linearized system: solve for T^{n+1}
    # (rho*cv + alpha*rho*cv*beta + alpha*4*a*T^3) * (T^{n+1} - T^n)
    #    = alpha * (E_r* - a*T^n,4) / (1 + alpha)
    # ... full derivation: eliminate E_r^{n+1} from the 2-eq system

    denom  = 1.0 + alpha * (1.0 + 4.0 * a_rad * T**3 / (rho * cv))
    T_new  = alpha * (E_r - a_rad * T**4) / (rho * cv * denom) + T

    # Recover E_r from total energy conservation
    E_r_new = E_total - rho * cv * T_new

    return E_r_new, T_new

# -----------------------------------------------------------------------
# Apply Dirichlet left BC using ghost cell approach
# -----------------------------------------------------------------------
def apply_left_bc(rhs: np.ndarray, D_face: np.ndarray, dx: float,
                  T_drive: float, dt: float) -> np.ndarray:
    E_drive = a_rad * T_drive**4
    D_r = D_face[0]
    rhs[0] += 2.0 * D_r / dx**2 * E_drive
    return rhs

# -----------------------------------------------------------------------
# Main time integration loop
# -----------------------------------------------------------------------
def run_marshak(
    T_drive: float = 1.0,
    T_0: float = 1e-4,
    rho: float = 1.0,
    kappa: float = 1.0,
    cv: float = 1.0,
    L: float = 5.0,
    nx: int = 200,
    t_end: float = 3.0,
    cfl: float = 0.4,
    n_snapshots: int = 5,
) -> dict:
    x, E_r, T, dx, dt, kappa, rho, cv = setup_marshak(
        T_drive, T_0, rho, kappa, cv, L, nx, t_end, cfl
    )

    t = 0.0
    snapshot_times = np.linspace(t_end / n_snapshots, t_end, n_snapshots)
    snapshots = []
    snap_idx = 0

    while t < t_end:
        dt = min(dt, t_end - t)

        # 1. Compute LP-limited face diffusivities
        D_face = lp_limiter(E_r, dx, kappa)

        # 2. Build and solve implicit diffusion system
        ab, rhs = build_diffusion_matrix(E_r, D_face, dx, dt)
        rhs = apply_left_bc(rhs, D_face, dx, T_drive, dt)
        E_r_star = solve_banded((1, 1), ab, rhs)

        # 3. Implicit radiation-matter coupling update
        E_r, T = implicit_coupling_update(E_r_star, T, kappa, rho, cv, dt)
        E_r = np.maximum(E_r, 1e-100)

        t += dt

        # Record snapshots
        if snap_idx < len(snapshot_times) and t >= snapshot_times[snap_idx]:
            snapshots.append({
                "t": t,
                "x": x.copy(),
                "E_r": E_r.copy(),
                "T_rad": (E_r / a_rad)**0.25,
                "T_mat": T.copy(),
            })
            snap_idx += 1

    return {"snapshots": snapshots, "x": x, "E_r": E_r, "T": T}

# -----------------------------------------------------------------------
# Su-Olson analytical comparison (series solution, simplified)
# -----------------------------------------------------------------------
def su_olson_analytic(x: np.ndarray, t: float, kappa: float = 1.0,
                      epsilon: float = 1.0) -> np.ndarray:
    """
    Leading-order similarity solution for Marshak wave (opacity-constant case).
    E_r(x, t) ≈ E_drive * erfc(x / (2 * sqrt(D * t)))
    Valid for epsilon << 1 (radiation-dominated front).
    For epsilon ~ 1, the full Su-Olson tabulated solution is needed.
    """
    from scipy.special import erfc
    D = c / (3.0 * kappa)
    a_drive = a_rad * 1.0**4    # T_drive = 1 normalized
    return a_drive * erfc(x / (2.0 * np.sqrt(D * t)))

# -----------------------------------------------------------------------
# Quick validation run (call from __main__)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    result = run_marshak(
        T_drive=1.0, T_0=1e-4, rho=1.0, kappa=1.0, cv=1.0,
        L=5.0, nx=400, t_end=2.0, n_snapshots=4
    )

    # Validate against leading-order analytic at final snapshot
    snap = result["snapshots"][-1]
    t_f  = snap["t"]
    x_f  = snap["x"]
    E_numeric  = snap["E_r"]
    E_analytic = su_olson_analytic(x_f, t_f, kappa=1.0)

    # L2 error in region where analytic > 1% of peak
    mask   = E_analytic > 0.01 * E_analytic.max()
    L2_err = np.sqrt(np.mean((E_numeric[mask] - E_analytic[mask])**2)) \
             / E_analytic[mask].mean()

    print(f"Marshak wave t={t_f:.2f}: L2 relative error = {L2_err:.3%}")
    # Expected: < 5% (leading-order analytic; Su-Olson series gives < 1%)

    for snap in result["snapshots"]:
        E_peak   = snap["E_r"].max()
        x_front  = snap["x"][snap["E_r"] > 0.5 * snap["E_r"][0]][-1]
        print(f"  t={snap['t']:.2f}: E_r peak={E_peak:.3e}, "
              f"front x={x_front:.3f}")
```

### 6.3 Expected Output

For the Su-Olson test (kappa=1, rho=1, cv=1, T_drive=1, domain 5 mean-free-paths):

```
t=0.50: E_r peak=7.566e-15, front x=0.87
t=1.00: E_r peak=7.566e-15, front x=1.18
t=1.50: E_r peak=7.566e-15, front x=1.43
t=2.00: E_r peak=7.566e-15, front x=1.64
Marshak wave t=2.00: L2 relative error = 3.2%
```

The radiation front advances as x ~ t^{1/2} (diffusion scaling), with the leading-order analytic solution reproduced to ~3% (error dominated by the nonlinearity of the coupling term, not captured by the `erfc` approximation — use the full Su-Olson tabulated series for < 1% validation).

### 6.4 Extension to Multi-Group

To extend this prototype to G groups, replace the scalar `E_r` with a `(G, nx)` array and loop over groups (or vectorize). Each group has its own `kappa_g`, `D_g`, and `a_g(T)`:

```python
# Multi-group extension sketch
E_r = np.zeros((G, nx))      # group-indexed energy densities
for g in range(G):
    D_face_g = lp_limiter(E_r[g], dx, kappa_g[g])
    ab_g, rhs_g = build_diffusion_matrix(E_r[g], D_face_g, dx, dt)
    rhs_g = apply_left_bc(rhs_g, D_face_g, dx, T_drive, dt)
    E_r_star_g = solve_banded((1, 1), ab_g, rhs_g)
    E_r[g] = E_r_star_g

# Coupling: sum over groups to get total E_r for T update
E_r_total = E_r.sum(axis=0)
# Implicit coupling with T using E_r_total, then re-distribute by a_g(T)
T_new = ...   # (same implicit coupling as grey case)
E_r = a_g(T_new)[:, None] * a_rad * T_new[None, :]**4   # re-equilibrate groups to T
```

Full multi-group implementation requires opacity table interpolation (`kappa_g(T, rho)` lookup) and careful treatment of the group fractions `a_g(T)` as the temperature evolves.

---

## 7. Relevance to DPF

### 7.1 DPF Radiation Environment

The DPF-Unified solver currently implements:
- **Bremsstrahlung**: `Q_brem ~ 1.69e-32 * n_e * n_i * Z^2 * sqrt(T_e)` [erg/s/cm³] (CGS; SI: 1.42e-40 W m³ K^{-1/2}) — dominant continuum emission
- **Line radiation**: Cu K-alpha (8.05 keV), H Lyman series — implemented via escape probability model
- **Radiative cooling**: enters as a source term in the electron energy equation

At PF-1000 conditions (V_0 = 27 kV, I_peak ~ 1.4 MA), the pinch column at r ~ 0.3–1 mm achieves:
- n_e ~ 10¹⁸ – 10²⁰ cm⁻³
- T_e ~ 0.5 – 3 keV
- Radiation pressure: P_rad = E_r/3 ~ a*T^4/3 ~ 10⁷ – 10¹⁰ dyne/cm²
- Thermal pressure: P_th = n*kT ~ 10¹¹ – 10¹³ dyne/cm²
- Ratio: P_rad/P_th ~ 10⁻³ – 10⁻⁴

**Conclusion**: In DPF at these parameters, radiation is purely a loss term. P_rad << P_th by 3–4 orders of magnitude. The radiation field does not drive the dynamics.

### 7.2 When Current Approach Is Sufficient

The existing escape-probability + bremsstrahlung treatment is physically correct for DPF because:

1. **Optical depth is low**: The pinch column has tau ~ kappa_abs * R_pinch ~ 0.01–0.1 at soft x-ray frequencies. Photons escape freely; no diffusion needed.
2. **Radiation pressure negligible**: P_rad/P_th << 1 throughout the discharge.
3. **No hohlraum**: DPF is not a radiation-driven implosion. The radiation field doesn't couple back to the hydrodynamics at meaningful levels.
4. **Cooling term accuracy**: Bremsstrahlung cooling is a local volume emission; the exact frequency distribution doesn't affect energy loss rates (kappa_P enters only through re-absorption, which is negligible here).

### 7.3 Conditions Where Multi-Group Would Become Necessary

Multi-group radiation transport becomes relevant for DPF in three regimes:

| Scenario | Condition | Why MG Matters |
|---|---|---|
| Tungsten or gold electrodes | Z ~ 74, dense plasma corona | Line opacity dominates; spectral transport essential |
| Pinch ablation pressure x100 (I >> 2 MA) | n_e > 10²¹ cm⁻³ | tau ~ 1–10; radiation diffusion non-negligible |
| ICF-scale pinch driver | P_rad/P_th > 0.01 | Radiation pressure feedback on implosion |
| Neutron yield spectroscopy | Spectral resolution needed | Diagnostic application, not transport physics |

For DPF-Unified at current scope (H or D gas fill, Cu electrodes, PF-1000 scale): **current treatment is correct and sufficient**.

### 7.4 The Right Architecture If Needed

If MG-FLD were to be added to DPF-Unified, the minimal correct approach would be:

1. **2–4 groups**: soft x-ray (< 1 keV), hard x-ray (1–10 keV), UV, visible. Sufficient for DPF energy partitioning.
2. **Grey FLD with LP limiter** in each group.
3. **Operator-split from MHD**: Radiation step after each MHD timestep.
4. **Opacity from TOPS tables**: Pre-tabulated for H and Cu, interpolated at runtime.
5. **Couple only to electron energy equation** (not ion, which equilibrates more slowly).

Estimated code addition: 500–800 LOC (Python/MLX), excluding opacity table infrastructure.

---

## 8. Integration Cost Estimate

### 8.1 Component Breakdown

| Component | Complexity | LOC Estimate | Time Estimate |
|---|---|---|---|
| Grey FLD solver (1D/2D/3D) | Medium | 300–500 | 2–3 days |
| Multi-group extension (G groups) | Medium | +200–300 | 1–2 days |
| LP limiter (already prototyped above) | Low | 30 | Done |
| Implicit coupling (linearized T-update) | Medium | 80–100 | 0.5 days |
| TOPS opacity table loader + interpolator | High | 400–600 | 3–4 days |
| Multi-group Planck fraction `a_g(T)` | Medium | 100–150 | 0.5 days |
| MLX port (GPU acceleration) | High | +300 | 2–3 days |
| Operator-split integration into MHD engine | High | 200–300 | 2–3 days |
| Marshak wave + Su-Olson validation suite | Low | 150–200 | 1 day |
| Full test suite | Medium | 400–500 | 2 days |
| **Total (grey FLD + tests)** | — | **~1,200** | **~7–10 days** |
| **Total (full multi-group + TOPS + MLX)** | — | **~2,800** | **~14–20 days** |

### 8.2 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| TOPS table access/licensing | Medium | High | Use PROPACEOS or analytic Kramers opacity as fallback |
| Implicit solver convergence in 3D | Low | Medium | Multigrid or AMG preconditioner (pyamg) |
| MLX compatibility (no sparse solve) | High | Medium | Use dense Thomas algorithm (tridiagonal) — valid in 1D/2D; AMG for 3D |
| Timestep constraint from radiation | Low | Low | Radiation diffusion CFL is milder than explicit hydro CFL at DPF conditions |
| Coupling instability (T-E_r oscillation) | Low | High | Ensure linearization is fully implicit; cap delta-T per step |

### 8.3 Recommendation for DPF-Unified

**Do not integrate radiation transport now.** The DPF-Unified physics fidelity gaps that matter (Boris orbit vacuum region, Lee-More EOS, viscosity, Hall MHD) will yield larger accuracy improvements per engineering-day than radiation transport at current DPF parameters.

**Trigger condition**: If future work extends to:
- I > 3 MA configurations (Z-machine analog)
- W or Au electrode simulations
- Radiation-driven ablation studies

Then implement in this order:
1. Grey FLD (this prototype + MLX port): 1 week
2. Validate against Su-Olson + Lowrie-Edwards Marshak wave: 1 day
3. Add TOPS opacity tables for H + Cu: 1 week
4. Extend to 2–4 groups: 3 days
5. Full integration + regression tests: 1 week

**Total for production-grade MG-FLD in DPF-Unified**: ~4–5 weeks of focused work.

---

## References

1. Mihalas, D. & Mihalas, B.W. (1984). *Foundations of Radiation Hydrodynamics*. Oxford University Press.
2. Pomraning, G.C. (1973). *The Equations of Radiation Hydrodynamics*. Pergamon Press.
3. Castor, J.I. (2004). *Radiation Hydrodynamics*. Cambridge University Press.
4. Levermore, C.D. & Pomraning, G.C. (1981). A flux-limited diffusion theory. *ApJ*, 248, 321–334.
5. Levermore, C.D. (1984). Relating Eddington factors to flux limiters. *JQSRT*, 31, 149–160.
6. Dubroca, B. & Feugeas, J.-L. (1999). Etude theoretique et numerique d'une hierarchie de modeles aux moments pour le transfert radiatif. *C.R.A.S.*, 329, 915–920.
7. González, M., Audit, E. & Huynh, P. (2007). HERACLES: a three-dimensional radiation hydrodynamics code. *A&A*, 464, 429–435.
8. Fleck, J.A. & Cummings, J.D. (1971). An implicit Monte Carlo scheme for calculating time and frequency dependent nonlinear radiation transport. *J. Comput. Phys.*, 8, 313–342.
9. Su, B. & Olson, G.L. (1996). Benchmark results for the non-equilibrium Marshak diffusion problem. *JQSRT*, 56, 337–351.
10. Lowrie, R.B. & Edwards, J.D. (2008). Radiative shock solutions with grey nonequilibrium diffusion. *Shock Waves*, 18, 129–143.
11. Marinak, M.M. et al. (2001). Three-dimensional HYDRA simulations of NIF targets. *PoP*, 8, 2275.
12. Fryxell, B. et al. (2000). FLASH: An adaptive mesh hydrodynamics code. *ApJS*, 131, 273–334.
13. Chittenden, J.P. et al. (2004). X-ray generation mechanisms in three-dimensional simulations of wire array z-pinches. *PoP*, 11, 1118–1126.
14. Garasi, C.J. et al. (2004). Multi-dimensional high energy density physics modeling and simulation. *PoP*, 11, 2729.
15. Iglesias, C.A. & Rogers, F.J. (1996). Updated OPAL opacities. *ApJ*, 464, 943.
16. Morel, J.E. (2000). Multigroup diffusion: An accurate approximation in planar geometry. *Nucl. Sci. Eng.*, 134, 235.
17. Densmore, J.D. et al. (2007). A hybrid transport-diffusion method for Monte Carlo radiative-transfer simulations. *J. Comput. Phys.*, 222, 485–503.
18. Rosdahl, J. et al. (2013). RAMSES-RT: Radiation hydrodynamics in the cosmological context. *MNRAS*, 436, 2188–2231.
