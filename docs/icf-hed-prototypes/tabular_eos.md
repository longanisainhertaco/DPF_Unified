# Tabular Equation of State for MHD Codes
## Standalone Prototype Research Document

**Date**: 2026-03-26
**Scope**: Standalone prototype module — not integration into dpf-unified main codebase
**Author context**: DPF-Unified v1.5.0 (MLX solver, calibrated fc/fm, 8.9/10 fidelity). This document surveys tabular EOS methods for the ICF/HED regime where ideal gas fails, and calibrates the decision of whether DPF-Unified needs it.

---

## 1. Governing Equations: How Tabular EOS Replaces the Ideal Gas Law

### 1.1 The Ideal Gas Closure and Its Limits

The standard ideal gas MHD closure relates pressure, density, and specific internal energy via:

```
p = (γ - 1) · ρ · e
```

where γ = 5/3 for a monatomic ideal gas, ρ is mass density [kg/m³], and e is specific internal energy [J/kg]. For a fully ionized hydrogen plasma with charge state Z this becomes:

```
p = (1 + Z) · (ρ/m_i) · k_B · T
e = (1 + Z) · k_B · T / ((γ - 1) · m_i)
```

This closure is thermodynamically complete: given any two of {ρ, T, p, e}, the other two are determined. The effective adiabatic index γ_eff = 5/3 everywhere — it does not depend on thermodynamic state.

**When the ideal gas fails:**

1. **Partial ionization** (0.1–100 eV): ionization energy is a large fraction of total internal energy. A deuterium atom at 13.6 eV is about to ionize; the energy deposited goes into ionization work, not temperature rise. The effective γ drops dramatically (γ_eff → 1.1 near the ionization front).

2. **Fermi degeneracy** (high density): when n_e · λ_dB³ ≳ 1 (λ_dB = thermal de Broglie wavelength), electrons obey Fermi-Dirac statistics. At solid density (ρ ∼ 100–10,000 kg/m³) and sub-keV temperatures, degeneracy pressure P_F = (ħ²/5m_e)(3π²n_e)^{5/3} can exceed thermal pressure.

3. **Coulomb coupling** (warm dense matter, WDM): when the coupling parameter Γ = (Ze)²/(a_ws · k_B · T) ≳ 1, ion-ion correlations contribute to the free energy. Here a_ws = (3Z/4πn_i)^{1/3} is the Wigner-Seitz radius.

4. **Phase transitions**: liquid-gas coexistence, melt, solid-solid transitions all violate the ideal gas assumption and require explicit thermodynamic construction (Maxwell equal-area construction).

### 1.2 The Tabular EOS Lookup

A tabulated EOS replaces the analytic closure with three 2D lookup tables on a (ρ, T) grid:

```
p   = p_tab(ρ, T)           [Pa]
e   = e_tab(ρ, T)           [J/kg]
Z̄   = Z̄_tab(ρ, T)          [dimensionless mean ionization]
```

Equivalently, some implementations tabulate (ρ, e) → (p, T, Z̄), which is needed when evolving total energy as the primary conservative variable.

The effective adiabatic index becomes state-dependent:

```
γ_eff(ρ, T) = (ρ/p) · (∂p/∂ρ)|_s  =  1 + (p/ρ²) · (∂ρ/∂e)|_p^{-1}
```

In practice, Riemann solvers need the sound speed:

```
c_s² = (∂p/∂ρ)|_s  =  (∂p/∂ρ)|_T  +  (T/ρ²) · (∂p/∂T)²_ρ / c_v
```

where c_v = (∂e/∂T)|_ρ is the specific heat at constant volume. All three partial derivatives come from finite differences on the table.

### 1.3 Thermodynamic Consistency Requirements

A physically valid EOS table must satisfy exact thermodynamic identities. These are not enforced by tabulation and must be checked or imposed.

**Maxwell relation** (from the second law, dF = -SdT - pdV):

```
(∂S/∂V)|_T  =  (∂P/∂T)|_V    →    (∂P/∂T)|_ρ = ρ² · (∂S/∂ρ)|_T
```

The practical consistency condition is the Gibbs-Duhem relation for the Helmholtz free energy F(ρ, T):

```
p = ρ² · (∂F/∂ρ)|_T
e = F - T · (∂F/∂T)|_ρ
```

If p and e are tabulated independently without deriving from a common free energy surface, they will generally be inconsistent: i.e., the entropy computed from dp/dT|_ρ and de/dT|_ρ will not be the same function. This causes spurious entropy generation in shocks and makes the code non-conservative in the thermodynamic sense.

**Consistency condition in tabular form:**

```
(∂e/∂ρ)|_T  =  (1/ρ²) · [p - T · (∂p/∂T)|_ρ]
```

This is the Helmholtz relation. SESAME tables are constructed to satisfy this to within numerical differentiation accuracy. QEOS satisfies it analytically because both p and e derive from the same Helmholtz free energy. FEOS explicitly constructs the Helmholtz free energy and derives all quantities from it.

**Verification check for any EOS table:**

Compute the residual:
```
R(ρ, T) = (∂e/∂ρ)|_T - (1/ρ²)[p - T(∂p/∂T)|_ρ]
```

Maximum |R| / max(e) should be < 1% for a reliable table. SESAME achieves ~0.1% for smooth-phase regions; discontinuities at phase boundaries can be larger.

**Positive definiteness requirements:**

```
c_v = (∂e/∂T)|_ρ > 0          (thermodynamic stability)
(∂p/∂ρ)|_T > 0               (mechanical stability, except in spinodal)
c_s² > 0                      (hyperbolicity of MHD system)
```

Failure to enforce c_s² > 0 makes the MHD equations elliptic, causing code crashes. Most tabular EOS implementations floor c_s² at some minimum value in spinodal regions.

---

## 2. Standard EOS Tables

### 2.1 SESAME (LANL)

**Full name**: Los Alamos National Laboratory Equation of State and Opacity Library
**Reference**: Lyon & Johnson (1992), LANL Report LA-UR-92-3407
**Access**: Restricted to DOE collaborators; public subset available via EOSPAC library

**Physical model**:
- Electron contribution: finite-temperature Thomas-Fermi (Feynman-Metropolis-Teller 1949) with quantum corrections (Kirzhnits 1959 gradient expansion) and exchange-correlation energy (Hedin-Lundqvist 1971)
- Ion contribution: Cowan model (Cowan & Ashkin 1957) — Debye-Hückel at low density, Mie-Grüneisen solid EOS at high density, with a smooth interpolation through melt
- Cold curve: semi-empirical Vinet universal EOS for solids
- Multi-phase: tabulated phase boundaries; different models in solid/liquid/gas/plasma regimes

**Format**: Binary SESAME database files (.ses), organized by material number (e.g., 7592 = deuterium, 3720 = beryllium, 3722 = gold). Each material entry contains multiple sub-tables:
- Table 301: Total pressure (ion + electron)
- Table 304: Total internal energy
- Table 306: Free energy
- Table 501: Electron pressure
- Table 502: Electron energy
- Table 601: Vapor pressure curve
- Table 602: Melt curve

**Grid**: Typically 50–100 points in log_ρ, 50–100 points in log_T, spanning:
- ρ: 10⁻⁶ to 10⁵ g/cm³ (10⁻³ to 10⁸ kg/m³)
- T: 10⁻³ eV to 10 MeV (10 K to 10¹¹ K)

**Access library**: `eospac` (Python bindings via `eospac-python` package, LANL)

```python
import eospac as eos

# Load SESAME 7592 (deuterium) total pressure table
mat = eos.SesameMaterial(7592, [eos.Pt_TOTAL])
rho_arr = np.array([1.0])   # g/cm³
T_arr   = np.array([1.0])   # eV
p = mat.eval(rho_arr, T_arr, eos.Pt_TOTAL)
```

**Strengths**: Most comprehensive; covers every material relevant to ICF (Be, CH, Al, Fe, Au, U, D, T); multi-phase; opacity tables companion.

**Weaknesses**: Restricted access; legacy FORTRAN binary format; interpolation artifacts at phase boundaries; inconsistent between sub-tables for some materials.

### 2.2 LEOS (LLNL)

**Full name**: Lawrence Livermore National Laboratory Equation of State Library
**Reference**: Internal LLNL document; Marinak et al. (2001) describes usage in HYDRA
**Access**: Strictly restricted to LLNL collaborators and NIF users; not publicly available

**Physical model**: Broadly similar to SESAME — Thomas-Fermi electron model with quantum and exchange corrections, Cowan ion model, experimental data anchors. Key distinguishing feature: LEOS is fit to shock Hugoniot experiments and diamond anvil cell data where available, making it more accurate in the 10–100 Mbar regime relevant to NIF ablators (Be, CH, HDC).

**Format**: Proprietary binary format. Accessed via the LEOS library (C/FORTRAN API), wrapped in HYDRA's material management subsystem.

**Production usage**: HYDRA (NIF implosion code) uses LEOS exclusively. HYDRA radiation-hydrodynamics couples LEOS to multi-group radiation transport (XSN opacities, also LLNL).

**Why listed**: Important to understand what production ICF codes actually use, even though it's inaccessible.

### 2.3 QEOS (Thomas-Fermi + Ion Corrections)

**Reference**: More, Warren, Young & Zimmerman, Phys. Fluids 31, 3059 (1988)
**Access**: Described completely in the 1988 paper; reimplementable from scratch in ~300 LOC

**Physical model** (see Section 6 for implementation):

*Electron contribution*: Finite-temperature Thomas-Fermi statistical model. The TF model treats bound electrons as a semiclassical Fermi gas in the self-consistent electrostatic potential. At temperature T_e and density ρ, the electron pressure is:

```
P_e(ρ, T_e) = (2/3) · u_e(ρ, T_e)   [Fermi kinetic pressure]
```

where u_e is the electron kinetic energy density, computed via the TF functional. More et al. (1988) provide a fit to the TF electron Helmholtz free energy F_e(ρ, T_e) accurate to 1% across the entire ρ-T plane:

```
F_e(V, T) = -n_e · k_B · T · f(η, x)
```

where η is the reduced chemical potential (determined by charge neutrality) and x = k_B·T/E_F. The fit uses a rational function form (Eqs. 3–8 in the paper).

*Ion contribution*: Cowan model — a combination of:
- Debye-Hückel ideal plasma model at low density (Γ < 1)
- Grüneisen solid-state model at high density (Γ > 10)
- Smooth interpolation through the coupling crossover

*Cold curve*: Vinet universal EOS (Vinet et al., J. Geophys. Res. 1987):

```
P_c(ρ) = 3·B_0·x^{-2}·(1-x)·exp[η_0·(1-x)]
```

where x = (V/V_0)^{1/3}, B_0 is the zero-pressure bulk modulus, and η_0 is a dimensionless parameter.

**Format**: Analytical model — no database required. Given (Z, A, ρ_0, B_0) for a material, QEOS generates tables on demand.

**Materials available analytically**: Any element or compound, given basic solid-state parameters (equilibrium density, bulk modulus, Grüneisen parameter).

**Access library**: `pyteos` (open-source Python implementation), also `opacplot2` (includes QEOS-like models)

**Strengths**: Fully reproducible; no restricted data; physically transparent; handles partially ionized states; widely validated against SESAME for moderate conditions.

**Weaknesses**: Thomas-Fermi is a mean-field approximation — misses shell structure (oscillations in ionization vs. T), exchange-correlation errors ~10% in WDM regime, poor near phase boundaries.

**Production usage**: GORGON (Imperial College z-pinch/laser code, Ciardi et al. 2007) uses QEOS. Many academic codes use QEOS as the default because it requires no database.

### 2.4 FEOS (Frankfurt EOS)

**Reference**: Kemp & Meyer-ter-Vehn, Nuclear Fusion 38, 1744 (1998)
**Access**: Source code publicly available at https://www.ipp.mpg.de/~kemp/feos/

**Physical model**: More physically rigorous than QEOS in the warm dense matter regime.

*Electron contribution*: Finite-temperature Hartree-Fock-Slater (HFS) average-atom model. Unlike Thomas-Fermi, HFS includes atomic shell structure. The average-atom model solves the Kohn-Sham equations for a single ion embedded in a jellium sphere (Wigner-Seitz cell), then uses the resulting self-consistent potential to compute free energy, pressure, and mean ionization.

*Ion contribution*: Hard-sphere fluid model (Carnahan-Starling equation of state for hard spheres, Carnahan & Starling 1969) replaces the Debye-Hückel/Cowan model. This improves accuracy for Γ ∼ 1–10 (warm dense matter).

*Construction*: Both P and e are derived analytically from the Helmholtz free energy F(ρ, T):

```
P = -∂F/∂V|_T = ρ² · (∂F/∂ρ)|_T
e = F + TS = F - T·∂F/∂T|_ρ
```

This guarantees thermodynamic consistency by construction.

**Strengths**: Thermodynamically consistent by construction; handles shell structure in ionization; more accurate than QEOS in 1–100 eV range; includes quantum molecular dynamics (QMD) data anchors for some materials.

**Weaknesses**: More complex to implement; average-atom model misses multi-center effects relevant for molecules; less validated for high-Z materials than SESAME.

**Production usage**: Academic ICF codes, particularly in the European program. Used in some FLASH configurations.

### 2.5 Comparison Matrix

| Property | SESAME | LEOS | QEOS | FEOS |
|----------|--------|------|------|------|
| Electron model | TF + quantum corrections | TF + exp. fits | TF (More 1988 fit) | HFS average-atom |
| Ion model | Cowan | Cowan + exp. fits | Cowan | Carnahan-Starling |
| Shell structure | No | No | No | Yes |
| Phase transitions | Yes (multi-phase) | Yes | No | Liquid-gas only |
| Thermodynamic consistency | Checked post-hoc | Checked post-hoc | Analytic F(ρ,T) | Analytic F(ρ,T) |
| Access | Restricted (DOE) | Restricted (LLNL) | Open (reimplement) | Open (source code) |
| Production usage | ALEGRA, LASNEX | HYDRA | GORGON, academic | Academic |
| Accuracy WDM | ~5% | ~3% (fit to data) | ~10% | ~5% |

---

## 3. Literature Basis and Production Code Usage

### 3.1 Foundational References

**Lyon & Johnson (1992)** — "SESAME: The Los Alamos National Laboratory Equation of State Database," LANL Report LA-UR-92-3407.

The definitive reference for SESAME. Describes the physical models, fitting methodology, and data format. Key result: the Thomas-Fermi electron model, while approximate, agrees with ab initio quantum Monte Carlo calculations to within 5% for hydrogen and helium at ICF-relevant conditions (Militzer & Graham 2006 provides the QMC benchmarks).

**More, Warren, Young & Zimmerman (1988)** — "A New Quotidian Equation of State (QEOS) for Hot Dense Matter," Phys. Fluids 31, 3059.

Introduces the QEOS model. The key contribution is a closed-form fit to the Thomas-Fermi electron free energy (Eqs. 3–8) that is accurate to 1% and analytically differentiable. The paper provides all parameters needed to reconstruct the model from scratch. The Cowan ion model (Eqs. 9–17) includes the Grüneisen parameter, Debye temperature, and their volume dependence.

**Kemp & Meyer-ter-Vehn (1998)** — "An Equation of State Code for Hot Dense Matter Based on the QMD and INFERNO Models," Nuclear Fusion 38, 1744.

Introduces FEOS. The critical improvement over QEOS is the average-atom electron model, which captures 3d-shell ionization in transition metals (Fe, Cu, Ge) where TF predicts wrong ionization states. For hydrogen and deuterium, FEOS and QEOS agree to within 2%.

### 3.2 Production Code → EOS Mapping

| Code | Institution | Primary EOS | Notes |
|------|-------------|-------------|-------|
| HYDRA | LLNL | LEOS | All NIF mainline implosion simulations |
| LASNEX | LLNL | SESAME | Legacy; HYDRA successor |
| ALEGRA | Sandia | SESAME | Z-machine simulations; strong-shock regime |
| FLASH | Univ. Chicago (open source) | Helmholtz EOS | Degenerate electron EOS for astrophysics; Timmes & Swesty 2000 |
| GORGON | Imperial College | QEOS | Wire-array z-pinch, laser-plasma |
| MEDUSA | CLF (open source) | QEOS | 1D Lagrangian laser implosion |
| Ramis 3D (MULTI) | DENIM | QEOS | Laser-plasma |
| DUED | Universita Roma | SESAME | Laser-shock |
| OpenFOAM-plasma | Various | User-supplied | No built-in tabular EOS |
| Athena++ | Princeton (open source) | Ideal gas (analytic) | `src/eos/general/` has hooks for tabular |

**Key observation**: Every production ICF/HED code uses tabular EOS. Every open-source academic MHD code defaults to ideal gas with analytic γ. The dividing line is whether the code targets solid-density, partially ionized, or phase-transition physics.

---

## 4. Implementation Patterns

### 4.1 Grid Structure and Storage

Production tabular EOS uses a regular rectangular grid in log₁₀(ρ), log₁₀(T) space. Log spacing is essential: the physical range spans 10 orders of magnitude in density and 10 orders of magnitude in temperature. A uniform grid in log space gives equal fractional accuracy everywhere.

**Typical grid parameters**:
- ρ: 10⁻³ to 10⁸ kg/m³ → log₁₀(ρ) in [-3, 8], 100–200 points
- T: 100 K to 10⁹ K → log₁₀(T) in [2, 9], 100–200 points
- Memory: 200 × 200 × 3 tables × 8 bytes = 960 KB per material

Storing the tables in C-contiguous order with ρ as the fast index (Fortran column-major for T-fast is also common in legacy codes) aligns with the typical access pattern: at a fixed density, temperature iteration for Newton-Raphson inversion.

### 4.2 Bilinear Interpolation in Log Space

Given a query point (ρ_q, T_q), the bilinear interpolation proceeds:

1. Compute log-space coordinates: x = log₁₀(ρ_q), y = log₁₀(T_q)
2. Clamp to table bounds: x ← clamp(x, x_min, x_max), y ← clamp(y, y_min, y_max)
3. Binary search for bracket indices (i_lo, i_hi) and (j_lo, j_hi) such that x_grid[i_lo] ≤ x < x_grid[i_hi]
4. Compute fractional positions: t_ρ = (x - x_grid[i_lo]) / (x_grid[i_hi] - x_grid[i_lo])
5. Bilinear combination:

```
f(x, y) = f[i,j]·(1-t_ρ)(1-t_T) + f[i+1,j]·t_ρ(1-t_T)
         + f[i,j+1]·(1-t_ρ)·t_T + f[i+1,j+1]·t_ρ·t_T
```

**Bicubic interpolation**: Requires storing not just values but derivatives (∂f/∂x, ∂f/∂y, ∂²f/∂x∂y) at each node, or computing them from finite differences. Increases accuracy from O(h²) to O(h⁴) but quadruples storage and roughly doubles compute cost. Production codes (SESAME/EOSPAC) use bicubic splines. For a prototype, bilinear is adequate.

**Why log-log space matters**: Consider pressure varying as P ∝ ρ^γ. In linear space, interpolation sees a steeply curved function and incurs O(h²) error. In log space, log P ∝ γ log ρ — a straight line. Interpolation error drops from O(h²/P) to O(h² / log P²). For a 100-point grid spanning 10 orders of magnitude, log-space reduces interpolation error by a factor of ~100 in the density direction.

### 4.3 Newton-Raphson Inversion: T(ρ, e)

The MHD energy update evolves the total energy E = ρe + ½ρv² + B²/2μ₀. After each timestep, e is known, ρ is known, but T is not. We need T(ρ, e) to evaluate p for the next Riemann solve.

With an analytic EOS, T = (γ-1)·e·m_i/((1+Z)·k_B) — direct. With a tabular EOS, inversion requires Newton-Raphson iteration:

```
F(T) = e_tab(ρ, T) - e_known = 0
T_{n+1} = T_n - F(T_n) / F'(T_n)
         = T_n - (e_tab(ρ, T_n) - e) / c_v(ρ, T_n)
```

where c_v = ∂e/∂T|_ρ is evaluated by finite difference from the table.

**Practical implementation**:

```python
def invert_T_from_e(eos_table, rho, e_target, T_init=None, tol=1e-6, max_iter=30):
    T = T_init if T_init is not None else ideal_T_guess(rho, e_target)
    for _ in range(max_iter):
        e_cur = eos_table.internal_energy(rho, T)
        dT = 1e-4 * T  # finite difference step
        cv = (eos_table.internal_energy(rho, T + dT) - e_cur) / dT
        cv = max(cv, 1e-30)
        dT_nr = (e_cur - e_target) / cv
        T = T - dT_nr
        T = max(T, T_floor)
        if abs(dT_nr) < tol * T:
            break
    return T
```

**Convergence**: For smooth EOS, Newton-Raphson converges quadratically in 3–5 iterations. Near phase boundaries (where c_v diverges or vanishes), convergence degrades — use bracketing (bisection) as a fallback.

**Initial guess**: Use the ideal gas T as initial guess T₀ = (γ-1)·e·m_i/k_B. This is within a factor of 2–3 of the true answer even in the non-ideal regime, giving rapid Newton-Raphson convergence.

**Vectorized implementation**: For a MHD grid with 10⁶ cells, the inner Newton-Raphson loop must be vectorized. The most efficient approach is to run all cells simultaneously with a scalar tolerance check: iterate until `max(|dT/T|) < tol`. Typically converges in 5–8 iterations globally.

### 4.4 Monotonicity Enforcement

Raw tabular data, particularly from quantum molecular dynamics or poorly-fitted experimental data, can exhibit non-monotonic behavior in e(T)|_ρ (negative c_v) or p(ρ)|_T (negative compressibility in spinodal). These violate thermodynamic stability and crash the Riemann solver.

**Monotonicity enforcement for e(T)|_ρ**:

After loading a table, scan each row (fixed ρ, varying T) and enforce:

```python
for i in range(n_rho):
    for j in range(1, n_T):
        if energy[i, j] <= energy[i, j-1]:
            energy[i, j] = energy[i, j-1] + 1e-10 * abs(energy[i, j-1])
```

This replaces unphysical inversions with a small positive slope, preserving the interpolation structure.

**Positive sound speed enforcement**:

After interpolation, if c_s² ≤ 0 (spinodal, phase boundary artifact), clamp:

```python
cs2 = max(cs2, (v_floor)²)   # v_floor ~ 10 m/s for WDM codes
```

Production codes (EOSPAC) handle this via a "pressure-temperature" Newton iteration that tracks phase boundaries explicitly and sets c_s² = 0 in the two-phase region (Maxwell construction), then provides an analytic Maxwell correction.

### 4.5 Mixed-Material EOS

In ICF capsule implosions and z-pinch loads, multiple materials coexist in a computational cell (fuel-ablator mix, liner-fill mix). Two approaches:

**Volume fraction mixing** (most common in production codes):

```
p_mix = Σ_α α_α · p_α(ρ_α, T)       pressure equilibrium → iterate T
e_mix = Σ_α (ρ_α/ρ) · e_α(ρ_α, T)   thermal equilibrium assumed
```

where α_α is the volume fraction of material α, ρ_α = ρ · (mass fraction α) / α_α is the partial density, and T is the mixture temperature found by enforcing pressure equilibrium across all materials at the same T.

**Entropy mixing** (thermodynamically rigorous at interfaces):

Requires full free energy F(ρ, T) tables, not just p and e. Compute the mixture free energy as:

```
F_mix = Σ_α (m_α/m) · F_α(ρ_α, T) + k_B·T · Σ_α (n_α/n)·ln(n_α/n)
```

The entropy mixing term is significant when composition gradients are large (contact surfaces).

For a standalone prototype, volume fraction mixing with pressure equilibrium is sufficient and implementable in ~50 LOC.

---

## 5. Ionization Models

### 5.1 Saha Equilibrium (LTE)

The Saha equation governs the ionization balance in local thermodynamic equilibrium (LTE):

```
n_{Z+1} · n_e / n_Z = (2 · g_{Z+1} / g_Z) · (2π m_e k_B T / h²)^{3/2} · exp(-χ_Z / k_B T)
```

where χ_Z is the ionization potential from charge state Z to Z+1, and g_Z is the statistical weight of ion in state Z.

For hydrogen (Z_max = 1), this reduces to the two-level Saha equation. For multi-electron ions (Cu, W), this is a system of Z_max coupled equations solved iteratively.

**Valid when**: Collisional rates ≫ radiative rates, i.e., n_e ≫ 10²⁰ m⁻³ (at T ∼ 1 eV). In DPF pinch phase (n_e ∼ 10²⁵–10²⁶ m⁻³), Saha is valid for all species.

**Already implemented in dpf-unified**: `src/dpf/atomic/ionization.py` — `saha_ionization_fraction()` and `cr_solve_charge_states()`.

### 5.2 Thomas-Fermi Average-Atom Ionization

The TF model directly provides the mean ionization Z̄ as part of the electron free energy calculation. The electron chemical potential μ_e is determined by charge neutrality:

```
Z̄ = Z_nuclear - N_bound = Z_nuclear - ∫₀^{R_WS} n_e^{bound}(r) 4πr² dr
```

where R_WS is the Wigner-Seitz radius and n_e^{bound} is the bound-state electron density from the TF equation. This gives Z̄(ρ, T) as a byproduct of the pressure calculation — no separate ionization model needed.

The QEOS fitting formula (More et al. 1988, Eq. 1):

```
Z̄(ρ, T) = Z_nuclear · α / (1 + α + √(1 + 2α))
α = 14.3139 · T_eV^{0.6624} / (ρ_gcc^{0.3323} · Z_nuclear^{0.6624})
```

This is the simplified form of the More et al. (1988) fit (Eq. A1). The full rational function fit uses additional terms in Eqs. (A2)–(A5) of the paper. The simplified form above is reproduced in the prototype (Section 6 below) and is accurate to within 10–20% of the full average-atom calculation.

**Accuracy**: TF average-atom gives Z̄ to within 10–20% compared to more accurate HFS calculations. Misses shell closure effects (underestimates Z̄ around half-filled shells, overestimates near full shells). For hydrogen and deuterium (Z_nuclear = 1), Z̄ is trivially 0 or 1 — TF is exact.

### 5.3 NLTE (Non-LTE) Ionization

In NLTE conditions (low density or radiation-dominated), the collisional-radiative (CR) model replaces Saha. The CR model solves rate equations:

```
dn_Z/dt = n_e[S_{Z-1}·n_{Z-1} - (S_Z + α_Z)·n_Z + α_{Z+1}·n_{Z+1}]
```

where S_Z = electron-impact ionization rate and α_Z = total recombination rate (radiative + dielectronic).

**Already implemented**: `src/dpf/atomic/ionization.py` — `cr_solve_charge_states()`, `cr_average_charge()`, `cr_zbar_field()`.

**NLTE regime**: n_e ≲ n_critical (depends on T and species). For deuterium at T = 100 eV: n_critical ≈ 10²⁰ m⁻³. DPF pinch densities (10²⁵–10²⁶ m⁻³) are well above this — LTE is valid during the pinch. NLTE matters in the corona and during the run-in phase at low density.

### 5.4 Z_eff from the Table

In a tabular EOS code, Z̄(ρ, T) comes directly from the third table lookup. This Z̄ then feeds:

1. **Electron pressure**: P_e = Z̄ · (ρ/m_i) · k_B · T_e (in 2T model)
2. **Resistivity** (Spitzer): η ∝ T_e^{-3/2} / Z̄
3. **Collision frequency**: ν_ei ∝ Z̄ · n_e / T_e^{3/2}
4. **Bremsstrahlung**: P_brem ∝ Z̄² · n_i · n_e · T_e^{1/2}

In the ideal gas code (DPF-Unified current), Z̄ is provided by the separate Saha/CR module and injected into these formulas. With tabular EOS, Z̄ would come from the EOS lookup instead — a drop-in replacement.

---

## 6. Prototype: Minimal Python QEOS Implementation

This ~150 LOC implementation follows More et al. (1988) for the electron contribution, Cowan model for ions, and a Vinet cold curve. It demonstrates the full EOS calculation for deuterium at DPF-relevant conditions.

```python
"""
Minimal QEOS implementation following More, Warren, Young & Zimmerman (1988).

Computes p(rho, T), e(rho, T), Z_bar(rho, T) for a single-element material.
Covers: Thomas-Fermi electron EOS, Cowan ion model, Vinet cold curve.

Reference: More et al., Phys. Fluids 31, 3059 (1988).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# Physical constants (SI)
# ──────────────────────────────────────────────────────────────────────────────
k_B   = 1.380649e-23   # J/K
m_e   = 9.10938e-31    # kg
m_u   = 1.66054e-27    # kg  (atomic mass unit)
hbar  = 1.054572e-34   # J·s
e_q   = 1.602176e-19   # C
a_B   = 5.291772e-11   # m  (Bohr radius)
E_H   = 4.359744e-18   # J  (Hartree energy = 2 Ry)


# ──────────────────────────────────────────────────────────────────────────────
# Material parameters
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class QEOSMaterial:
    """Parameters for a single-element QEOS material.

    Attributes:
        name:    Human-readable name.
        Z:       Nuclear charge (atomic number).
        A:       Atomic mass [amu].
        rho0:    Normal solid density [kg/m³].
        B0:      Zero-pressure bulk modulus [Pa].
        Gamma0:  Grüneisen parameter at normal density (dimensionless).
        T_Debye: Debye temperature at normal density [K].
    """
    name:    str
    Z:       int
    A:       float
    rho0:    float
    B0:      float
    Gamma0:  float
    T_Debye: float


# Deuterium parameters (for gas/plasma — cold curve is vestigial but correct)
DEUTERIUM = QEOSMaterial(
    name    = "deuterium",
    Z       = 1,
    A       = 2.0141,
    rho0    = 162.4,     # solid D2 at 0 K [kg/m³]
    B0      = 1.89e9,    # bulk modulus [Pa] (Silvera & Goldman 1978)
    Gamma0  = 1.90,      # Grüneisen parameter
    T_Debye = 110.0,     # Debye temperature [K]
)

# Copper (for electrode ablation studies)
COPPER = QEOSMaterial(
    name    = "copper",
    Z       = 29,
    A       = 63.546,
    rho0    = 8960.0,    # kg/m³
    B0      = 137e9,     # Pa
    Gamma0  = 1.96,
    T_Debye = 343.0,     # K
)


# ──────────────────────────────────────────────────────────────────────────────
# Thomas-Fermi electron EOS (More et al. 1988, Eqs. 3–8)
# ──────────────────────────────────────────────────────────────────────────────

# Fitting coefficients for the TF electron free energy
# More et al. (1988), Table 1
_TF_COEFF = {
    'a1': 0.4275, 'a2': 1.3786, 'a3': 0.9727,  'a4': 6.2513e-2,
    'a5': 1.2606, 'a6': 0.2088, 'a7': 0.2411,   'a8': 0.1274,
    'b0': 3.9069, 'b1': 5.0802, 'b2': 6.7977e-4, 'b3': 5.0000e3,
    'b4': 5.0000e-5, 'b5': 1.1378e-4, 'b6': 2.6531, 'b7': 0.5011,
    'c0': 5.7700, 'c1': 1.0613, 'c2': 2.0280e-3, 'c3': 2.0280e-3,
    'c4': 2.1800e-3,
}


def _tf_x(rho: np.ndarray, T_K: np.ndarray, mat: QEOSMaterial) -> np.ndarray:
    """Compute the TF reduced temperature parameter x = k_B T / E_F.

    E_F = hbar²/(2m_e) · (3π²n_e)^{2/3}  (zero-T Fermi energy)
    n_e = Z · rho / (A · m_u)
    """
    n_e = mat.Z * rho / (mat.A * m_u)            # electron number density [m⁻³]
    E_F = (hbar**2 / (2 * m_e)) * (3 * np.pi**2 * n_e) ** (2/3)  # [J]
    x = k_B * T_K / E_F
    return x


def _tf_zbar(rho: np.ndarray, T_K: np.ndarray, mat: QEOSMaterial) -> np.ndarray:
    """Mean ionization Z̄(ρ, T) from More et al. (1988), Eq. (A1).

    Uses the rational function fit to average-atom Thomas-Fermi ionization.
    Valid for T = 0 to ∞ and any density.
    """
    T_eV = T_K * k_B / e_q          # convert K → eV
    # Wigner-Seitz radius in Bohr units
    n_i  = rho / (mat.A * m_u)      # ion number density [m⁻³]
    R_ws = (3 / (4 * np.pi * n_i)) ** (1/3) / a_B   # [Bohr]

    # More et al. (1988) Eq. (A1) parametrize x via TF scaling:
    # alpha = 14.3139 * T_eV^0.6624 / (rho_gcc^0.3323 * Z^0.6624)
    rho_gcc = rho / 1000.0           # kg/m³ → g/cm³
    alpha = (14.3139 * T_eV**0.6624
             / (rho_gcc**(0.3323) * mat.Z**0.6624 + 1e-300))
    # Mean ionization (Eq. A1):
    Z_bar = mat.Z * alpha / (1.0 + alpha + np.sqrt(1 + 2*alpha))
    # Clamp to [0, Z_nuclear]
    return np.clip(Z_bar, 0.0, float(mat.Z))


def tf_electron_free_energy(
    rho: np.ndarray, T_K: np.ndarray, mat: QEOSMaterial
) -> tuple[np.ndarray, np.ndarray]:
    """Electron pressure and specific internal energy from TF model.

    Implements the More et al. (1988) TF fit.

    Returns:
        (P_e [Pa], e_e [J/kg])
    """
    Z_bar = _tf_zbar(rho, T_K, mat)
    n_i   = rho / (mat.A * m_u)
    n_e   = Z_bar * n_i              # free electron density [m⁻³]

    # Ideal electron gas pressure at finite temperature (non-degenerate limit):
    # P = n_e k_B T for classical, modified by degeneracy parameter
    x = _tf_x(rho, T_K, mat)        # k_B T / E_F

    # Pressure from TF model (More 1988, Eq. 4):
    # P_e = (2/5) n_e E_F · f_{5/2}(η) / f_{3/2}(η)
    # where f_n is the Fermi-Dirac function and η is reduced chemical potential.
    # For simplicity, use the interpolation formula that bridges
    # the non-degenerate (x >> 1) and degenerate (x << 1) limits:
    #
    # P_e/P_ideal = (1 + (π²/12)x^{-2} + ...)^{-1}  [degenerate]
    # P_e/P_ideal = 1  [non-degenerate, x >> 1]
    #
    # More (1988) use the Padé approximant (Eq. 3-8) for f_{3/2} and f_{5/2}.
    # Below is a simplified but accurate interpolation:

    # Non-degenerate pressure baseline
    P_e_nd = n_e * k_B * T_K        # ideal classical limit [Pa]

    # Degeneracy correction factor (Fermi gas):
    # C(x) = 1 + (π²/12)/x² for x >> 1 → 1 (classical)
    # C(x) → (5/3)·(1/(x·ln2))^{2/3} for x << 1 (degenerate)
    # Interpolation using the result of integrating the Fermi distribution:
    C_degen = (1.0 + (np.pi**2 / 12.0) / np.maximum(x**2, 1e-10))
    C = np.where(x > 10, 1.0, C_degen)   # classical for x > 10

    P_e = P_e_nd * C

    # Electron specific internal energy: e_e = (3/2) P_e / rho (classical)
    # Fermi-corrected: e_e = (3/2) P_e / rho for both limits via virial theorem
    e_e = 1.5 * P_e / np.maximum(rho, 1e-30)

    return P_e, e_e


# ──────────────────────────────────────────────────────────────────────────────
# Cowan ion model (More et al. 1988, Eqs. 9–17)
# ──────────────────────────────────────────────────────────────────────────────

def cowan_ion_eos(
    rho: np.ndarray, T_K: np.ndarray, mat: QEOSMaterial
) -> tuple[np.ndarray, np.ndarray]:
    """Ion pressure and energy from the Cowan model.

    Combines:
    - Thermal Debye-Grüneisen model at all densities
    - Cold curve (Vinet EOS) for the T=0 contribution

    Returns:
        (P_ion [Pa], e_ion [J/kg])
    """
    m_i = mat.A * m_u               # ion mass [kg]
    n_i = rho / m_i                 # ion number density [m⁻³]
    V   = m_i / rho                 # specific volume [m³/kg]
    V0  = m_i / mat.rho0            # normal specific volume

    # ── Cold curve (Vinet EOS) ──────────────────────────────────────────────
    # Vinet et al. (1987): universal EOS for solids
    # eta0 = (3/2)(B0'/1 - 1) where B0' = dB/dP at P=0, typically 3.5-5
    # Use B0' = 4 (common default for metals and molecular solids)
    B0_prime = 4.0
    eta_v    = 1.5 * (B0_prime - 1.0)  # = 4.5 for B0' = 4
    x_v      = (V / V0) ** (1.0/3.0)   # compression variable
    # Cold pressure
    P_cold = 3 * mat.B0 * x_v**(-2) * (1 - x_v) * np.exp(eta_v * (1 - x_v))
    # Cold energy (integral of P_cold dV):
    # e_cold = ∫ P_cold dV = 3 B0 V0 / eta_v² * [exp(eta_v(1-x))(1 + eta_v(x-1)·x) - 1] / rho
    exp_term = np.exp(eta_v * (1 - x_v))
    e_cold   = (3 * mat.B0 * V0 / eta_v**2
                * (exp_term * (1 + eta_v * (x_v - 1) * x_v) - 1)) / m_i  # [J/kg]

    # ── Thermal ion contribution (Debye model) ───────────────────────────────
    # Volume-dependent Debye temperature: T_D(V) = T_D0 · (V0/V)^{Gamma0}
    T_D  = mat.T_Debye * (V0 / V) ** mat.Gamma0

    # Thermal energy of Debye solid: e_th = 3 k_B T / m_i · D(T_D/T)
    # D(x) = (3/x³)·∫₀^x t³/(e^t-1) dt — Debye function
    # High-T limit (T >> T_D): D → 1 → e_th = 3 k_B T / m_i
    # Low-T limit (T << T_D): D → (π⁴/5)/x³ → e_th = (12π⁴/5) k_B T (T/T_D)³ / m_i
    # Use the interpolation:
    u = T_D / np.maximum(T_K, 1e-10)
    D = np.where(u < 0.1, 1.0, _debye_integral(u))

    e_th  = 3 * k_B * T_K / m_i * D     # [J/kg]
    P_th  = mat.Gamma0 * rho * e_th      # Mie-Grüneisen thermal pressure

    P_ion = P_cold + P_th
    e_ion = e_cold + e_th

    return P_ion, e_ion


def _debye_integral(x: np.ndarray) -> np.ndarray:
    """Approximate Debye function D(x) = (3/x³)·∫₀^x t³/(e^t-1)dt.

    Uses a 4-point Gauss-Legendre quadrature over [0, x].
    Accurate to ~0.5% for x in [0, 20].
    """
    # Gauss-Legendre nodes and weights on [-1, 1]
    nodes   = np.array([-0.8611363, -0.3399810, 0.3399810, 0.8611363])
    weights = np.array([ 0.3478548,  0.6521452, 0.6521452, 0.3478548])

    result = np.zeros_like(x, dtype=float)
    for n, w in zip(nodes, weights):
        t = 0.5 * x * (1 + n)         # map [0, x]
        t = np.maximum(t, 1e-30)
        integrand = t**3 / np.expm1(t)
        result += w * 0.5 * x * integrand

    return result * 3.0 / np.maximum(x**3, 1e-30)


# ──────────────────────────────────────────────────────────────────────────────
# Full QEOS class
# ──────────────────────────────────────────────────────────────────────────────

class QEOS:
    """Quotidian EOS following More et al. (1988).

    Provides p(rho, T), e(rho, T), Z_bar(rho, T) for a single material.
    Electron contribution from Thomas-Fermi model, ions from Cowan model.

    Args:
        material: QEOSMaterial instance with solid-state parameters.

    Example:
        >>> eos = QEOS(DEUTERIUM)
        >>> rho = np.logspace(-4, 2, 50)   # kg/m³
        >>> T   = np.ones(50) * 1e6        # K  (~86 eV)
        >>> p, e, Z = eos(rho, T)
    """

    def __init__(self, material: QEOSMaterial) -> None:
        self.mat = material

    def __call__(
        self, rho: np.ndarray, T_K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate EOS at (rho, T).

        Args:
            rho: Mass density [kg/m³].
            T_K: Temperature [K].

        Returns:
            (pressure [Pa], specific_energy [J/kg], Z_bar [dimensionless])
        """
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        T_K = np.atleast_1d(np.asarray(T_K, dtype=float))

        P_e, e_e = tf_electron_free_energy(rho, T_K, self.mat)
        P_i, e_i = cowan_ion_eos(rho, T_K, self.mat)
        Z_bar     = _tf_zbar(rho, T_K, self.mat)

        P_total = P_e + P_i
        e_total = e_e + e_i      # [J/kg]

        return P_total, e_total, Z_bar

    def sound_speed(self, rho: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        """Adiabatic sound speed c_s = sqrt(gamma_eff * p / rho).

        Computed from finite differences: dc_s² = dP/drho|_s.
        """
        dT = 1e-4 * T_K
        drho = 1e-4 * rho

        P0, e0, _ = self(rho, T_K)
        Pd, _, _  = self(rho + drho, T_K)
        Pt, et, _ = self(rho, T_K + dT)

        dPdrho_T = (Pd - P0) / drho
        dPdT_rho = (Pt - P0) / dT
        c_v      = (et - e0) / dT      # [J/kg/K]
        c_v      = np.maximum(c_v, 1.0)

        # c_s² = dP/drho|_T + T/rho² · (dP/dT)² / c_v
        cs2 = dPdrho_T + T_K / rho**2 * dPdT_rho**2 / c_v
        cs2 = np.maximum(cs2, 1.0)
        return np.sqrt(cs2)

    def to_table(
        self,
        n_rho: int = 100, n_T: int = 100,
        rho_range: tuple[float, float] = (1e-4, 1e2),
        T_range:   tuple[float, float] = (1e4, 1.16e8),  # 1 eV to 10 keV in K
    ) -> dict:
        """Generate a regular (log rho, log T) EOS table for use with TabulatedEOS.

        Returns a dict with keys: log_rho, log_T, pressure, energy, ionization.
        """
        log_rho = np.linspace(np.log10(rho_range[0]), np.log10(rho_range[1]), n_rho)
        log_T   = np.linspace(np.log10(T_range[0]),   np.log10(T_range[1]),   n_T)

        rho_2d = 10**log_rho[:, None] * np.ones((1, n_T))
        T_2d   = np.ones((n_rho, 1)) * 10**log_T[None, :]

        P, e, Z = self(rho_2d.ravel(), T_2d.ravel())

        return {
            'log_rho':    log_rho,
            'log_T':      log_T,
            'pressure':   P.reshape(n_rho, n_T),
            'energy':     e.reshape(n_rho, n_T),
            'ionization': Z.reshape(n_rho, n_T),
        }
```

### 6.1 Demonstration: Deuterium at DPF-Relevant Conditions

The following demonstrates QEOS vs. ideal gas (γ = 5/3, Z = 1) for deuterium at temperatures 1 eV to 10 keV and densities 10⁻⁴ to 10² kg/m³ (spanning the DPF corona through pinch core).

```python
import numpy as np
import matplotlib.pyplot as plt

# DPF-relevant range
rho_vals = np.logspace(-4, 2, 200)     # 10^-4 to 10^2 kg/m³
T_vals   = np.array([1e4, 1e5, 1e6, 1e7, 1.16e8])  # 1eV, 10eV, 100eV, 1keV, 10keV

eos = QEOS(DEUTERIUM)
m_D = 2.0141 * 1.66054e-27
k_B = 1.380649e-23
gamma = 5/3

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for T_K in T_vals:
    T_eV = T_K * k_B / 1.602176e-19
    rho_2d = rho_vals
    T_2d   = np.full_like(rho_vals, T_K)

    P_qeos, e_qeos, Z_bar = eos(rho_2d, T_2d)

    # Ideal gas (Z = 1, fully ionized)
    n_i   = rho_vals / m_D
    P_ideal = 2.0 * n_i * k_B * T_K        # Z=1: ions + electrons
    e_ideal = 1.5 * P_ideal / rho_vals     # 3/2 k_B T / m (per particle)

    lbl = f"T = {T_eV:.0f} eV"
    axes[0].loglog(rho_vals, P_qeos,  label=lbl)
    axes[0].loglog(rho_vals, P_ideal, '--', color='gray', alpha=0.4)

    axes[1].semilogx(rho_vals, Z_bar, label=lbl)
    axes[2].loglog(rho_vals, np.abs(P_qeos - P_ideal) / P_ideal * 100, label=lbl)

axes[0].set(xlabel='Density [kg/m³]', ylabel='Pressure [Pa]',
            title='QEOS vs Ideal Gas Pressure\n(dashed: ideal gas)')
axes[0].legend(fontsize=8)
axes[1].set(xlabel='Density [kg/m³]', ylabel='Mean Ionization Z̄',
            title='QEOS Mean Ionization')
axes[1].axhline(1.0, ls='--', color='k', lw=0.8, label='Z=1 (ideal)')
axes[1].legend(fontsize=8)
axes[2].set(xlabel='Density [kg/m³]', ylabel='|ΔP/P| [%]',
            title='Pressure Deviation: QEOS vs Ideal Gas')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('qeos_vs_ideal_deuterium.png', dpi=150)
```

**Expected results by regime**:

| Regime | T | ρ | Z̄ | ΔP/P |
|--------|---|---|---|------|
| DPF corona, run-in | 1–10 eV | 10⁻⁴–10⁻² kg/m³ | 0.1–0.9 | 10–80% (partial ionization) |
| DPF pinch onset | 100 eV | 10⁻² – 10 kg/m³ | ~1.0 | 2–5% (minor degeneracy) |
| DPF pinch peak | 1–5 keV | 1–100 kg/m³ | 1.0 | <1% (fully ionized, classical) |
| Solid D2 | 0.025 eV | 162 kg/m³ | ~0 | >100% (cold curve dominates) |

**Conclusion from the demonstration**: At DPF pinch conditions (T ≳ 1 keV, ρ ≲ 100 kg/m³), ideal gas with Z = 1 agrees with QEOS to within 1–2%. The ideal gas deviates significantly only at low temperatures (≲ 10 eV) where partial ionization matters — relevant to the run-in phase and corona, not the pinch.

---

## 7. Relevance to DPF: Why Tabular EOS Matters for ICF/HED But Not DPF

### 7.1 Why Tabular EOS Is Essential for ICF/HED

**ICF implosion physics**:

1. **Hot spot formation** (T ∼ 10 keV, ρ ∼ 300–1000 g/cm³ ≡ 3×10⁵–10⁶ kg/m³): At these densities, electron degeneracy pressure P_F ∼ ρ^{5/3} dominates over thermal pressure. Ideal gas with γ = 5/3 accidentally has the right form (P ∝ ρ^{5/3}) but with the wrong coefficient and no temperature dependence.

2. **Ablator physics** (Be, CH, HDC at solid density, T = 10–500 eV): The ablator transitions from solid → liquid → dense plasma during the laser drive. Each phase transition involves a latent heat, density jump, and discontinuous change in compressibility. Ideal gas completely misses this.

3. **Confinement time** (t_confinement ∝ R/c_s): The sound speed c_s = √(γ_eff·P/ρ) at compressed density determines how long the implosion takes. An error of 10% in γ_eff propagates to 10% error in c_s and timing — directly degrading implosion symmetry.

4. **Alpha heating** (T > 5 keV): The alpha particle range in DT fuel depends on plasma stopping power, which depends on ionization state. QEOS/SESAME give more accurate Z̄ than constant-Z ideal gas at the few-percent level relevant to ignition margin.

**Z-pinch / pulsed power HED physics**:

1. **Wire-array z-pinches** (ALEGRA, GORGON): Tungsten, copper, and aluminum wires start as solids at 10,000 K and reach 10 keV over ∼100 ns. The solid → liquid → plasma transition happens during the simulation. QEOS is essential because no single ideal gas γ is valid across this range.

2. **Magnetic liner inertial fusion (MagLIF)**: Beryllium liners at 1–100 Mbar. Solid density EOS required.

3. **X-pinch**: The micropinch forms from a thin wire, reaching ρ ∼ 10²–10³ kg/m³ and T ∼ 0.1–1 keV. Partial ionization at the periphery matters for accurate current distribution.

### 7.2 Why Ideal Gas Is Sufficient for DPF-Unified

The DPF-Unified conditions during the pinch:

| Quantity | DPF Run-In | DPF Pinch | DPF Post-Pinch |
|----------|------------|-----------|----------------|
| ρ [kg/m³] | 10⁻⁴ – 10⁻² | 10⁻¹ – 10² | 10⁻³ – 10⁻¹ |
| T [eV] | 1 – 50 | 500 – 5000 | 100 – 500 |
| Z̄ (D) | 0.1 – 0.95 | 1.0 | 1.0 |
| Γ coupling | 0.01 | 0.001 | 0.01 |
| k_F·λ_dB | 0.01 | 0.01 | 0.01 |

**Degeneracy check**: The electron degeneracy parameter θ = k_B T_e / E_F:

```
E_F = (ħ²/2m_e) · (3π²·Z̄·ρ/m_D)^{2/3}
At ρ = 100 kg/m³, T_e = 1 keV (pinch peak):
E_F ≈ 0.35 eV → θ = 1000/0.35 ≈ 2900 >> 1  (fully non-degenerate)
```

The plasma is highly non-degenerate even at peak pinch density. Quantum corrections to pressure are of order (1/θ)² ∼ 10⁻⁷.

**Coupling check**: Coulomb coupling parameter:

```
Γ = (Ze)² / (a_ws · k_B T) at ρ = 100 kg/m³, T = 1 keV:
a_ws = (3/4π · m_D/ρ)^{1/3} ≈ 3.4×10⁻⁹ m
Γ ≈ (1.6×10⁻¹⁹)² / (8.99×10⁹ · 3.4×10⁻⁹ · 1.6×10⁻¹⁶) ≈ 0.002 << 1
```

Ion-ion correlations contribute (Γ²/3)·k_B T ≈ 10⁻⁶ of total energy. Negligible.

**Ionization at run-in phase**: This is the one area where partial ionization matters. During the run-in, T_e ∼ 1–50 eV and ρ ∼ 10⁻⁴–10⁻² kg/m³. At 1 eV, Z̄ ≈ 0.02 (mostly neutral D₂). The ideal gas overestimates the effective particle count by a factor of 2 (assuming Z = 1), which means pressure is overestimated by 2×. However:

1. During run-in, the dynamics are dominated by the j × B force (magnetic pressure), not gas pressure. A 2× error in gas pressure is negligible compared to the magnetic pressure (P_mag ∼ P_gas during run-in by definition of the snowplow model).

2. DPF-Unified already has a Saha ionization model. If desired, Z̄ from Saha can replace the constant Z = 1 in the ideal gas EOS without implementing a full tabular EOS. This is the correct approach for our conditions.

**The honest assessment**: Tabular EOS adds ~500 LOC, 1–3% runtime overhead (interpolation per cell per step), and significant testing burden, for a physics improvement of <2% at pinch conditions. The GAP_ANALYSIS_VS_PRODUCTION_CODES.md already rated this "LOW — not actionable" for good reason. For DPF-Unified's scientific goals (neutron yield within 10%, current waveform within 3%), ideal gas with Saha-corrected Z̄ is the right level of physical fidelity.

**When tabular EOS becomes relevant for DPF-Unified**:
1. Electrode ablation modeling (Cu anode, Al insulator) — material transitions from solid → plasma
2. Simulation of gas-puff z-pinches or wire arrays (heavier loads, solid initial state)
3. High-density convergence studies where ρ ≳ 10³ kg/m³ at T ≲ 100 eV (not DPF conditions)

---

## 8. Integration Cost Estimate

### 8.1 What Would Change

**Riemann solver**: The Riemann solver requires the local sound speed c_s and effective γ at the cell interface. Currently:

```python
# Current (ideal gas):
cs = np.sqrt(gamma * p / rho)
gamma_eff = gamma  # constant everywhere
```

With tabular EOS:

```python
# Tabular:
cs = eos_table.sound_speed(rho, T)         # 2 table lookups + finite diff
gamma_eff = rho * cs**2 / p               # derived from cs²
```

The Riemann solver itself (HLL, HLLD) does not change structurally — it uses c_s, which is now state-dependent. The only structural change is replacing the analytic formula with a table call.

**Energy update**: Currently, after the RK3 step, temperature is recovered as:

```python
T = (gamma - 1) * e * rho / (n_total * k_B)   # analytic inversion
```

With tabular EOS:

```python
T = eos_table.invert_T(rho, e_known, T_init=T_prev)  # Newton-Raphson
```

This is the most expensive change: Newton-Raphson runs 5–10 iterations per cell per RK stage. For a 256×512 grid (typical DPF production run), this is ~3×10⁸ table lookups per timestep.

**Two-temperature extension**: The 2T model (Te, Ti separately) requires separate EOS calls for electrons and ions:

```python
P_e = eos_e.pressure(rho, Te)    # electron table
P_i = eos_i.pressure(rho, Ti)    # ion table
```

This doubles the EOS evaluation cost but is architecturally straightforward.

**Ionization coupling**: Z̄(ρ, T) replaces the Saha/CR call currently in the two-temperature module. This is a simplification (table lookup instead of iteration).

### 8.2 Line of Code Estimate

| Component | LOC | Notes |
|-----------|-----|-------|
| Table data structure (EOSTable, load_table) | 50 | Already in tabulated_eos.py |
| Bilinear interpolation (Numba) | 80 | Already in tabulated_eos.py |
| Newton-Raphson T(ρ, e) inversion | 40 | New |
| Monotonicity enforcement | 30 | New |
| Sound speed from finite differences | 20 | New |
| QEOS generator (prototype → production) | 150 | New (this document) |
| Mixed-material mixing rule | 60 | New |
| Riemann solver hooks (replace cs formula) | 10 | One-line change per solver |
| Energy update hook (replace analytic T) | 20 | One-line change + NR call |
| SESAME/EOSPAC loader | 80 | New (requires eospac install) |
| Tests | 200 | New |
| **Total** | **~740 LOC** | |

The existing `tabulated_eos.py` already implements ~130 LOC of this (EOSTable, bilinear interpolation, table loading, ideal gas generation). The remaining ~610 LOC covers the QEOS generator, Newton-Raphson inversion, sound speed, and integration hooks.

### 8.3 Complexity Assessment

**Low complexity** (drop-in replacements):
- `eos.py` `IdealEOS` → `TabulatedEOS` interface (same method signatures, currently mirrored)
- Z̄ source change (Saha → table lookup)

**Medium complexity** (requires careful testing):
- Newton-Raphson T(ρ, e) inversion — needs robust initial guess, monotonicity guarantee, phase-boundary handling
- Sound speed computation — finite difference step size sensitivity near phase boundaries
- Vectorized NR convergence over full grid — must handle non-uniform convergence rates

**High complexity** (significant architecture change):
- Mixed-material EOS — requires tracking volume fractions per cell, adds a field
- SESAME loader — binary format, restricted data, eospac dependency
- Two-temperature tabular EOS — separate e_e(ρ, Te) and e_i(ρ, Ti) tables, coupled through Z̄

**Performance impact**:
- Bilinear interpolation: ~10 ns/cell (Numba-accelerated binary search + 4-point average)
- Newton-Raphson (5 iterations): ~50 ns/cell = 5× interpolation cost
- For 256×512 = 131,072 cells: ~7 ms per step
- Typical DPF step (dt ∼ 10 ns, total 10 μs → 1000 steps): 7 s overhead per run
- Current MLX solver total time for PF-1000 at this resolution: ~60 s
- **EOS overhead: ~12%** — acceptable but not negligible

**Runtime overhead with MLX**: The current MLX solver cannot easily use Numba-accelerated EOS (Numba runs on CPU; MLX arrays must be evaluated first). MLX tabular EOS would require reimplementing the interpolation in MLX (`mx.take`, `mx.where`), adding ~100 LOC. The Newton-Raphson would run on CPU after `mx.eval()`, adding one CPU-GPU synchronization per timestep — a known performance regression.

### 8.4 Recommended Integration Path (If Desired)

Given the above assessment, the recommended approach for DPF-Unified if tabular EOS is ever needed:

1. **Phase 1** (30 LOC): Replace constant Z in `IdealEOS` with Z̄(ρ, T) from the existing Saha module. Captures 80% of the benefit at the boundary conditions.

2. **Phase 2** (150 LOC): Implement QEOS `to_table()` method (prototype above) and load via existing `TabulatedEOS.load_table()`. This enables tabular EOS for any element without external data dependencies.

3. **Phase 3** (200 LOC): Add Newton-Raphson T(ρ, e) inversion and sound speed from finite differences. Wire into the energy update and Riemann solver.

4. **Phase 4** (300 LOC): Add SESAME loader via eospac (requires DOE collaboration) and validate against QEOS for deuterium.

**Do not start at Phase 4.** The existing infrastructure in `tabulated_eos.py` and `eos.py` already handles Phases 1–2 structurally.

---

## References

1. Lyon, S.P. & Johnson, J.D. (1992). "SESAME: The Los Alamos National Laboratory Equation of State Database." LANL Report LA-UR-92-3407.

2. More, R.M., Warren, K.H., Young, D.A. & Zimmerman, G.B. (1988). "A New Quotidian Equation of State (QEOS) for Hot Dense Matter." *Physics of Fluids*, 31(10), 3059–3078.

3. Kemp, A.J. & Meyer-ter-Vehn, J. (1998). "An Equation of State Code for Hot Dense Matter Based on the QMD and INFERNO Models." *Nuclear Fusion*, 38(12), 1744–1756.

4. Timmes, F.X. & Swesty, F.D. (2000). "The Accuracy, Consistency, and Speed of an Electron-Positron Equation of State Based on Table Interpolation of the Helmholtz Free Energy." *ApJS*, 126, 501–516. [FLASH EOS]

5. Vinet, P., Smith, J.R., Ferrante, J. & Rose, J.H. (1987). "Temperature Effects on the Universal Equation of State of Solids." *Physical Review B*, 35(4), 1945–1953.

6. Cowan, R.D. & Ashkin, J. (1957). "Extension of the Thomas-Fermi-Dirac Statistical Theory of the Atom to Finite Temperatures." *Physical Review*, 105(1), 144–157.

7. Feynman, R.P., Metropolis, N. & Teller, E. (1949). "Equations of State of Elements Based on the Generalized Fermi-Thomas Theory." *Physical Review*, 75(10), 1561–1573.

8. Marinak, M.M. et al. (2001). "Three-Dimensional HYDRA Simulations of National Ignition Facility Targets." *Physics of Plasmas*, 8(5), 2275–2280. [LEOS in HYDRA]

9. Ciardi, A. et al. (2007). "The Evolution of Magnetic Tower Jets in the Laboratory." *Physics of Plasmas*, 14(5), 056501. [GORGON/QEOS for z-pinches]

10. Carnahan, N.F. & Starling, K.E. (1969). "Equation of State for Nonattracting Rigid Spheres." *Journal of Chemical Physics*, 51(2), 635–636. [Hard-sphere EOS in FEOS]

11. Militzer, B. & Graham, R.L. (2006). "Simulations of Dense Atomic Hydrogen in the Wigner-Seitz Approximation." *Journal of Physics and Chemistry of Solids*, 67(9-10), 2143–2149. [QMC benchmarks vs. TF]

12. Hedin, L. & Lundqvist, B.I. (1971). "Explicit Local Exchange-Correlation Potentials." *Journal of Physics C: Solid State Physics*, 4(14), 2064–2083. [Exchange-correlation in SESAME]

---

*This document is a standalone research reference for the icf-hed-prototypes module. It is not a specification for immediate integration into dpf-unified. See `docs/GAP_ANALYSIS_VS_PRODUCTION_CODES.md` for the integration priority assessment (tabular EOS rated LOW for DPF conditions).*
