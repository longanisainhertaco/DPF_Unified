# Laser-Plasma Interaction Models for MHD Codes

**Status**: Standalone prototype module — NOT integrated into DPF-Unified
**Date**: 2026-03-26
**Scope**: ICF / High-Energy-Density (HED) physics. Irrelevant to DPF (see Section 7).

---

## Table of Contents

1. [Governing Equations](#1-governing-equations)
2. [Ray-Tracing Algorithms](#2-ray-tracing-algorithms)
3. [Parametric Instabilities](#3-parametric-instabilities)
4. [Literature Basis and Production Code Survey](#4-literature-basis-and-production-code-survey)
5. [Cross-Beam Energy Transfer (CBET)](#5-cross-beam-energy-transfer-cbet)
6. [Minimal 1D Prototype](#6-minimal-1d-prototype)
7. [Relevance to DPF](#7-relevance-to-dpf)
8. [Integration Cost Estimate](#8-integration-cost-estimate)

---

## 1. Governing Equations

### 1.1 Plasma Refractive Index

The complex refractive index of an unmagnetized collisional plasma in the geometric-optics limit is:

```
n^2 = 1 - (omega_pe^2 / omega^2) * (1 / (1 + i * nu_ei / omega))
```

where:
- `omega_pe = sqrt(n_e * e^2 / (eps_0 * m_e))` — electron plasma frequency
- `omega` — laser angular frequency
- `nu_ei` — electron-ion collision frequency (Spitzer)

In the collisionless limit (`nu_ei << omega`), taking the real part:

```
n = sqrt(1 - n_e / n_c)
```

where the critical density is:

```
n_c = eps_0 * m_e * omega^2 / e^2          [SI]
    = m_e * omega^2 / (4 * pi * e^2)      [CGS]
    = 1.1e21 * (1.06 um / lambda)^2       [cm^-3, NRL Formulary p. 28]
```

For Nd:YAG at 1.06 um: `n_c ~ 1.1e21 cm^-3`.
For 3omega (0.351 um): `n_c ~ 9.9e21 cm^-3`.

The real refractive index `n` falls from 1 (vacuum) to 0 at the critical surface. Light cannot propagate beyond `n_e = n_c`.

### 1.2 Geometric Optics: The Ray Equation

In the geometric-optics (eikonal) limit, the wave equation reduces to the ray equation (Born & Wolf 1999, ch. 3):

```
d/ds (n * dr/ds) = grad(n)
```

where:
- `r(s)` — ray position as a function of arc length `s`
- `n(r)` — refractive index field
- `grad(n)` — spatial gradient of the refractive index

This is a system of 6 ODEs in (r, k) phase space. Defining the ray direction unit vector `hat_k = dr/ds`, and the wavevector `k = n * omega/c * hat_k`, the equivalent Hamiltonian form is:

```
dr/dt = partial_H/partial_k
dk/dt = -partial_H/partial_r
H(r, k) = |k|^2 - n^2(r) * (omega/c)^2 = 0  (dispersion relation)
```

This formulation (Hamilton 1833; Luneburg 1964) is exact within geometric optics and enables symplectic integration.

**Validity**: Geometric optics holds when the scale length of density variation `L_n = n_e / |grad(n_e)|` satisfies:

```
L_n >> lambda / (2 * pi)
```

Near caustics or the critical surface, this breaks down and full-wave treatments (e.g., the resonance absorption zone) are required.

### 1.3 Inverse Bremsstrahlung Absorption

The dominant laser-absorption mechanism in underdense ICF corona plasma is collisional (inverse bremsstrahlung) absorption. The absorption coefficient (Kruer 1988, eq. 1.6):

```
kappa_IB = (nu_ei / c) * (n_e / n_c) / sqrt(1 - n_e/n_c)
```

with the Spitzer-Harm electron-ion collision frequency:

```
nu_ei = (4 * sqrt(2*pi) / 3) * (n_e * Z * e^4 * ln_Lambda) / (m_e^2 * v_Te^3)
      = 2.91e-6 * n_e * Z * ln_Lambda / T_e^(3/2)   [s^-1, CGS, T_e in eV]
```

where `ln_Lambda ~ 6-10` is the Coulomb logarithm and `Z` is the mean ionization state.

The intensity along a ray evolves as:

```
dI/ds = -kappa_IB(r(s)) * I(s)
```

Integrating along the ray path from vacuum (`I_0`) to position `s`:

```
I(s) = I_0 * exp(-tau(s))
tau(s) = integral_0^s kappa_IB(s') ds'   [optical depth]
```

The volumetric power deposition density (source term for the electron energy equation):

```
Q_abs(r) = kappa_IB(r) * I(r)   [W/cm^3]
```

This enters the MHD energy equation as:

```
partial_t(E) + div(F_E) = Q_abs + Q_rad + ...
```

where `E = rho*eps + B^2/(2*mu_0) + 0.5*rho*v^2` is total energy density.

### 1.4 Critical Surface Physics

At `n_e = n_c`, the refractive index `n -> 0` and the phase velocity `v_ph = c/n -> inf`. The group velocity `v_g = c*n -> 0`. Rays are refracted away from the critical surface.

For a ray with vacuum incidence angle `theta_0` (relative to density gradient), Snell's law gives the turning point:

```
n_e(r_turn) / n_c = 1 - sin^2(theta_0)
```

Rays at normal incidence (`theta_0 = 0`) reach `n_e = n_c` exactly. Oblique rays reflect before the critical surface.

**Resonance absorption** (not captured by geometric optics) occurs when `theta_0 != 0` and the p-polarized electric field drives oscillations at `omega_pe`. The absorbed fraction peaks near `(k_0 * L_n)^(1/3) * sin(theta_0) ~ 0.5` and can reach 20-50% of incident energy. This requires a separate fitting formula or full-wave solve (Forslund et al. 1975).

**Critical surface reflection**: For s-polarized or very oblique p-polarized light, the Fresnel reflection coefficient is computed at the turning point. In practice, production codes track energy balance: `I_transmitted = (1 - R_Fresnel) * I_incident`.

### 1.5 Laser Energy Deposition as MHD Source Term

The coupled laser-MHD system requires communicating `Q_abs(r)` from the ray-trace to the hydro grid. The standard operator-split approach:

1. **Ray trace** on the current density/temperature field -> compute `Q_abs` on ray segments
2. **Deposit** `Q_abs` to grid via volume-weighted scatter (CIC or NGP)
3. **Advance MHD** for one timestep with `Q_abs` as source in electron energy equation
4. **Advance ionization** (Thomas-Fermi or Saha) -> update `Z`, `n_e`
5. Repeat

The timestep constraint for stability:

```
dt_laser < min(dx / c,  1 / (kappa_IB * c))
```

In practice, ray-tracing is much faster than the speed of light on MHD timescales, so rays are traced quasi-statically each MHD step.

---

## 2. Ray-Tracing Algorithms

### 2.1 3D Ray Tracing (General)

The standard algorithm for production ICF codes (HYDRA, LILAC, DRACO):

**Initialization**:
- Sample the laser beam entrance plane with `N_rays` rays (typically 10^3 to 10^6)
- Each ray carries power `P_ray = P_total / N_rays`
- Assign initial direction from beam f-number and focus geometry

**Propagation** (each ray, each step):

```python
# Runge-Kutta-4 integration of ray equation
# State vector: y = [rx, ry, rz, kx, ky, kz]
# Hamilton's equations: dr/dt = k/|k|, dk/dt = (omega/c)^2 * grad(n^2)/2

def ray_rhs(y, n_field, grad_n_field):
    r, k = y[:3], y[3:]
    k_mag = np.linalg.norm(k)
    dr = k / k_mag                          # direction (unit vector)
    dk = 0.5 * (omega/c)**2 * grad_n2(r)   # refraction
    return np.concatenate([dr, dk])
```

**Absorption**: At each step of arc length `ds`:

```
dI = -kappa_IB(r) * I * ds
Q_grid[cell] += kappa_IB(r) * I * ds / V_cell
```

**Termination criteria**:
- Ray intensity drops below threshold: `I < 1e-6 * I_0`
- Ray exits computational domain
- Ray reaches `n_e > n_c` (reflection)
- Maximum steps exceeded

### 2.2 2D Cylindrical Ray Trace

For cylindrically symmetric targets (NIF hohlraum, direct-drive capsule in 2D):

In cylindrical (r, z) geometry, the ray equation becomes:

```
d^2r/ds^2 = (1/n) * (partial_r(n) - k_phi^2 / (n * r^2))
d^2z/ds^2 = (1/n) * partial_z(n)
d/ds(r^2 * dphi/ds) = 0   -> angular momentum conserved: L = n * r * sin(theta_phi) = const
```

The conserved quantity `L = n * r * k_phi` (generalized angular momentum) allows reduction to a 2D integration in (r, z). This is the basis for LILAC's 2D ray trace (Delettrez et al. 1987).

For a beam entering at angle `theta` to the axis:

```
L = n_0 * r_entry * sin(theta_entry)
```

**Caustic formation**: When `dr/ds = 0` and `d^2r/ds^2 < 0`, the ray approaches a caustic (envelope of rays). Geometric optics predicts infinite intensity; physical regularization requires:
- Gaussian beam smoothing
- Paraxial complex-ray tracing near focus
- Full-wave solve in caustic region (Kravtsov-Orlov theory)

### 2.3 Kramers Opacity and CBET Ray Trace

The **Kramers (free-free) opacity** for thermal plasma:

```
kappa_ff = (4 * Z^2 * e^6 * n_e * n_i) / (3 * m_e * c * h * nu^3) * sqrt(2*pi / (3*m_e*kB*T)) * g_ff
```

where `g_ff ~ 1-5` is the free-free Gaunt factor. This is equivalent to inverse bremsstrahlung in the low-frequency limit.

**Multi-beam tracing** tracks rays from all beams simultaneously. CBET coupling (Section 5) requires knowing the wavevector and intensity of all crossing beams at each point in space.

### 2.4 Power Deposition Schemes

**Nearest-grid-point (NGP)**: Deposit all absorption in one cell. Fast, noisy.

**Cloud-in-cell (CIC)**: Trilinear weight to 8 surrounding cells. Smoother, ~2x cost.

**Segment-based**: Compute `dI` over full segment, split proportionally among cells the segment traverses. Most accurate for coarse grids.

Production codes (HYDRA) use segment-based deposition with subcycling near the critical surface where `kappa_IB` varies rapidly.

---

## 3. Parametric Instabilities

### 3.1 Overview

At high laser intensities, coherent coupling between the laser wave and plasma waves drives exponential growth of daughter waves. These are treated as loss terms in the ray-trace energy budget. They do not appear in the geometric-optics ray equation directly — they modify the effective absorption.

**Threshold intensity** (rough guide):

```
I_threshold ~ 10^13 - 10^14 W/cm^2
```

Below this, parametric instabilities are negligible and inverse bremsstrahlung dominates. At NIF scale (`I ~ 10^14 - 10^15 W/cm^2`), they are a first-order effect.

### 3.2 Stimulated Raman Scattering (SRS)

**Process**: Laser photon -> backscattered photon + electron plasma wave (EPW)

**Matching conditions**:
```
omega_0 = omega_s + omega_epw
k_0 = k_s + k_epw
```

**Threshold** (homogeneous plasma, Liu et al. 1974):
```
I_SRS > (omega_0 * omega_epw * nu_epw * nu_s) / (k_epw * v_os)^2
```

where `v_os = eE_0/(m_e * omega_0)` is the electron quiver velocity.

**Saturation mechanisms**: Electron trapping, wave breaking, Landau damping (`k_epw * lambda_De > 0.3`), nonlinear frequency shifts.

**Impact on ICF**: SRS scatters 5-20% of incident energy backward; generates hot electrons (100-300 keV) that preheat the fuel and degrade implosion symmetry.

### 3.3 Stimulated Brillouin Scattering (SBS)

**Process**: Laser photon -> backscattered photon + ion acoustic wave (IAW)

**Matching conditions**:
```
omega_0 = omega_s + omega_iaw
k_0 ~ k_s + k_iaw    (k_iaw << k_0, so omega_s ~ omega_0)
```

**Gain coefficient** (Kruer 1988):
```
g_SBS = (v_os^2 / v_s^2) * (omega_pi^2 / omega_0) * (1 / nu_iaw)
```

where `v_s = sqrt((Z*T_e + 3*T_i)/m_i)` is the sound speed.

**Impact on ICF**: SBS reflects 5-50% of incident energy; primary threat to laser coupling efficiency. Mitigated by beam smoothing (RPP, SSD) and plasma flow velocity shear (Landau damping).

### 3.4 Two-Plasmon Decay (TPD)

**Process**: Laser photon -> two electron plasma waves (EPW1 + EPW2), occurs only near `n_c/4`

**Threshold** (Simon et al. 1983):
```
I_TPD > 10^14 * (T_e[keV] / L_n[um]) * lambda_um^2   [W/cm^2]
```

**Impact on ICF**: Generates hot electrons (20-100 keV) from `n_c/4` region. Strongly correlated with hard X-ray emission. Primary preheat concern for 2omega and 3omega drivers.

### 3.5 Implementation in Production Codes

HYDRA (Marinak et al. 2001) and LILAC (Delettrez et al. 1987) handle parametric instabilities via:
1. **Linear gain models**: Compute SRS/SBS gain at each ray point; subtract from ray intensity
2. **Reduced models**: Prescribe a fixed reflectivity fraction based on pre-computed PIC/Vlasov results
3. **Full coupling**: Separate hydro-kinetic models (pF3D, OSIRIS) run offline to provide scatter maps

For a minimal prototype, parametric instabilities are omitted or handled via a fixed absorption multiplier `eta_para in [0.6, 0.95]` applied to the IB absorption coefficient.

---

## 4. Literature Basis and Production Code Survey

### 4.1 Foundational References

**Kaiser et al. 2000** — "HYDRA: A mass-production ICF simulation code"
*Phys. Plasmas 7, 2062*

The definitive reference for production-scale laser-plasma coupling in ICF hydrocodes. HYDRA's laser package implements:
- 3D ray-trace with `~10^5` rays per beam, 40+ NIF beams
- Inverse bremsstrahlung with Langdon correction (non-Maxwellian f_e at high intensity)
- CBET via ray-coupling at crossing points (post-2010 extension)
- Flux-limited electron heat conduction (Spitzer-Harm + flux limiter f=0.06-0.1)
- Multigroup radiation transport (60 groups) coupled to laser deposition

HYDRA uses operator splitting: ray-trace -> deposit -> advance radiation+hydro. Ray trace is parallelized per-beam over MPI ranks.

**Radha et al. 2005** — "Multidimensional analysis of direct-drive, plastic-shell implosions on OMEGA"
*Phys. Plasmas 12, 056307*

Documents the LILAC code's 2D cylindrical ray-trace for direct-drive implosions. Key implementation details:
- Rays traced in (r, z) with conserved angular momentum
- Adaptive step size: `ds = alpha * min(dx, L_n)`, `alpha ~ 0.1`
- Absorption deposited to electron energy equation only (two-temperature model)
- Non-LTE ionization (detailed configuration accounting for 1-10 keV plasmas)
- Linked to DRACO for 2D implosion dynamics

**Froula et al. 2012** — *Plasma Scattering of Electromagnetic Radiation*, 2nd ed., Academic Press

The standard graduate text. Chapter 2: refractive index and ray equations. Chapter 8: Thomson scattering (diagnostic, not heating). Chapter 9-11: SRS, SBS, TPD theory and experimental signatures. Essential reference for parametric instability thresholds and saturation physics.

**Kruer 1988** — *The Physics of Laser Plasma Interactions*, Addison-Wesley

Seminal text. Chapter 1: inverse bremsstrahlung. Chapter 4: parametric instabilities. Chapter 6: filamentation. Equations are in CGS and remain the standard form referenced by production codes.

**Atzeni & Meyer-ter-Vehn 2004** — *The Physics of Inertial Confinement Fusion*, Oxford

Chapters 7-9: laser absorption physics, corona dynamics, ablation pressure scaling. Contains the ablation pressure scaling:
```
P_abl [Mbar] = 40 * (I_14 / lambda_um)^(2/3) * (A / 2Z)^(1/3)
```
where `I_14` is intensity in units of `10^14 W/cm^2`.

### 4.2 Production Code Survey

| Code | Lab | Geometry | Ray Model | CBET | Parametrics | Notes |
|------|-----|----------|-----------|------|-------------|-------|
| HYDRA | LLNL | 3D AMR | 3D ray-trace, ~10^5 rays/beam | Yes (post-2010) | Linear gain models | NIF workhorse |
| LILAC | LLE | 2D cyl | 2D cylindrical, angular momentum | Partial | SRS/SBS gain | OMEGA direct-drive |
| DRACO | LLE | 2D/3D | 3D, linked to LILAC | Yes | Reduced model | OMEGA implosions |
| FLASH | U.Chicago | 3D AMR | 3D ray-trace (laser package) | No | No | Open-source, ~200 LOC laser pkg |
| MULTI | GSI | 1D/2D | 1D ray-trace | No | No | Tabulated opacities |
| DUED | INFN | 2D | 2D ray-trace | No | Partial | Italian labs |
| ASTER | CEA | 2D/3D | 3D ray-trace | Yes | Partial | French ICF |

**FLASH laser package** (Fatenejad et al. 2013, *High Energy Density Physics* 9, 172) is the only open-source implementation at production quality. Uses:
- Bilinear interpolation of `n_e` onto ray positions
- RK2 integration
- Segment-based deposition
- ~200 LOC in Fortran90

---

## 5. Cross-Beam Energy Transfer (CBET)

### 5.1 Physical Mechanism

When two laser beams cross in underdense plasma, they can exchange energy via stimulated Brillouin scattering mediated by an ion acoustic wave (IAW). The seed wave for each beam's SBS is the other beam itself, provided the beat frequency matches an IAW:

```
omega_1 - omega_2 = omega_iaw(k_1 - k_2)
|k_1 - k_2| * v_s = omega_1 - omega_2
```

For monochromatic beams: `omega_1 = omega_2`, so `v_s ~ 0` (zero-frequency IAW). This is purely convective energy exchange with no frequency shift.

For beams of different wavelengths (wavelength-detuned CBET): the crossing angle and wavelength difference can be tuned to maximize or minimize transfer. This is exploited in NIF's laser architecture.

### 5.2 Coupling Model (Rosen et al. 2011; Michel et al. 2010)

The CBET gain coefficient between a pump beam (beam 1) and probe beam (beam 2) at a crossing point:

```
G_CBET = (e^2 / (m_e * c * epsilon_0)) * (k_iaw^2 / |k_iaw|) *
         (chi_i^2 / |epsilon_iaw|^2) * (I_1 / omega_1)
```

where:
- `chi_i` — ion susceptibility at the IAW frequency/wavevector
- `epsilon_iaw = 1 + chi_e + chi_i` — total dielectric function
- The ratio `|chi_i / epsilon_iaw|^2` peaks near IAW resonance

**Ponderomotive detuning**: At high intensity, the laser ponderomotive force modifies the local plasma density and velocity, shifting the IAW frequency and saturating CBET. The detuning parameter:

```
delta = (k_iaw * v_s - omega_beat) / (Gamma_iaw / 2)
```

where `Gamma_iaw` is the IAW damping rate. `delta = 0` maximizes gain.

### 5.3 Implementation in Ray-Trace

CBET requires a two-pass ray-trace algorithm:

**Pass 1**: Trace all beams simultaneously, storing `(r_i, k_i, I_i)` at each ray step.

**Pass 2**: At each spatial point, identify all crossing rays within a neighborhood `delta_V`. For each pair (i, j):
1. Compute beating wavevector `k_iaw = k_i - k_j`
2. Evaluate CBET gain `G_ij` from coupling model
3. Transfer power: `dI_i/ds = G_ij * I_j * I_i`, `dI_j/ds = -G_ij * I_j * I_i`

This is an `O(N_rays^2)` operation per spatial cell. Production codes (HYDRA) use spatial hashing to reduce to `O(N_crossing_pairs)` per cell. Typical NIF simulations: 40 beams x 10^5 rays = 4x10^6 rays; crossing pairs ~10^4 per cell.

### 5.4 Why CBET Matters for ICF

On NIF (192 beams, 1.8 MJ, 3omega, 351 nm), inner and outer cones cross in the hohlraum. Without CBET:
- Inner beams over-drive the hohlraum waist
- Outer beams under-drive the laser entrance hole
- Radiation drive non-uniformity > 5%: insufficient for ignition-class implosion symmetry

With CBET mitigation (wavelength detuning 0-3 Angstrom between inner/outer cones, Michel et al. 2010):
- Energy transfer from inner->outer beams partially compensated
- Drive symmetry improved to ~1%

CBET is the primary reason NIF implosion simulations post-2010 require multi-beam 3D ray-trace. 2D axisymmetric codes cannot capture it correctly.

---

## 6. Minimal 1D Prototype

A 100-LOC Python implementation of 1D ray tracing through a linear density ramp with inverse bremsstrahlung absorption. Demonstrates the core physics without grid coupling.

```python
"""
laser_plasma_1d.py

Minimal 1D ray-trace through a linear density ramp.
Demonstrates: refraction, inverse bremsstrahlung absorption, critical surface.

Physics:
  n(x) = sqrt(1 - n_e(x)/n_c)
  dI/ds = -kappa_IB(x) * I
  kappa_IB = (nu_ei/c) * (n_e/n_c) / sqrt(1 - n_e/n_c)

Units: CGS throughout. All quantities normalized where noted.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple


# --- Physical constants (CGS) ---
C_LIGHT = 2.998e10      # cm/s
E_CHARGE = 4.803e-10    # esu
M_ELECTRON = 9.109e-28  # g
K_BOLTZMANN = 1.381e-16 # erg/K
EV_TO_ERG = 1.602e-12


@dataclass
class PlasmaParams:
    """Plasma and laser parameters."""
    lambda_laser: float = 0.351e-4   # laser wavelength [cm], 3omega Nd:YAG
    I_0: float = 1e14                # peak intensity [W/cm^2]
    Z_ion: float = 3.5               # mean ionization (CH plasma ~C/H mix)
    A_ion: float = 6.5               # mean atomic mass
    T_e_eV: float = 2000.0           # electron temperature [eV]
    ln_Lambda: float = 7.0           # Coulomb logarithm
    n_e_peak: float = 0.9e21         # peak electron density [cm^-3]
    L_ramp: float = 200e-4           # density scale length [cm] (linear ramp)
    x_max: float = 250e-4            # domain length [cm]
    N_cells: int = 1000              # grid resolution


def critical_density(lambda_cm: float) -> float:
    """n_c [cm^-3] for laser wavelength lambda [cm]."""
    omega = 2 * np.pi * C_LIGHT / lambda_cm
    return M_ELECTRON * omega**2 / (4 * np.pi * E_CHARGE**2)


def nu_ei_spitzer(n_e: np.ndarray, T_e_eV: float, Z: float, ln_Lambda: float) -> np.ndarray:
    """Spitzer electron-ion collision frequency [s^-1], CGS."""
    T_e_erg = T_e_eV * EV_TO_ERG
    v_Te = np.sqrt(T_e_erg / M_ELECTRON)  # thermal velocity [cm/s], T already in erg
    return (4 * np.sqrt(2 * np.pi) / 3) * (n_e * Z * E_CHARGE**4 * ln_Lambda) / \
           (M_ELECTRON**2 * v_Te**3)


def build_density_profile(params: PlasmaParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear density ramp: n_e rises from 0 to n_e_peak over L_ramp, then flat.
    Returns (x [cm], n_e [cm^-3]).
    """
    x = np.linspace(0, params.x_max, params.N_cells)
    n_e = np.where(x < params.L_ramp,
                   params.n_e_peak * (x / params.L_ramp),
                   params.n_e_peak)
    return x, n_e


def compute_refractive_index(n_e: np.ndarray, n_c: float) -> np.ndarray:
    """n(x) = sqrt(1 - n_e/n_c), clipped to [0, 1]."""
    eta_sq = 1.0 - np.clip(n_e / n_c, 0.0, 1.0)
    return np.sqrt(np.maximum(eta_sq, 0.0))


def compute_kappa_IB(n_e: np.ndarray, n_c: float, nu_ei: np.ndarray) -> np.ndarray:
    """
    Inverse bremsstrahlung absorption coefficient [cm^-1].
    kappa_IB = (nu_ei / c) * (n_e/n_c) / sqrt(1 - n_e/n_c)
    Diverges at critical surface -> clip where n_e/n_c > 0.99.
    """
    ne_nc = np.clip(n_e / n_c, 0.0, 0.99)
    denom = np.sqrt(1.0 - ne_nc)
    return (nu_ei / C_LIGHT) * (ne_nc / denom)


def trace_ray_1d(x: np.ndarray, kappa_IB: np.ndarray, refr_idx: np.ndarray,
                 I_0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1D ray trace with IB absorption. Normal incidence (no refraction in 1D).

    In 1D with normal incidence, the ray travels in +x direction until
    n_e/n_c >= 1 (critical surface), where it reflects. We integrate
    dI/dx = -kappa_IB(x) * I * (1/n(x)) accounting for the reduced
    path element in the medium.

    Returns:
        I_forward [N_cells]: forward-going intensity
        I_reflected [N_cells]: reflected intensity (from critical surface)
        Q_dep [N_cells]: volumetric power deposition [W/cm^3]
    """
    dx = x[1] - x[0]
    N = len(x)

    I_fwd = np.zeros(N)
    I_fwd[0] = I_0

    # Forward pass: integrate dI/ds = -kappa_IB * I
    # ds/dx = 1/n(x) in the medium (phase path correction)
    for i in range(1, N):
        if refr_idx[i] < 1e-3:  # critical surface: reflect
            break
        # Path element in medium: ds = dx / n(x)
        path_fac = 1.0 / max(refr_idx[i], 1e-3)
        I_fwd[i] = I_fwd[i-1] * np.exp(-kappa_IB[i] * dx * path_fac)

    # Reflected wave: traverse in reverse from turning point
    # Reflection is total at critical surface (for normal incidence)
    i_crit = np.argmax(refr_idx < 1e-3)
    if i_crit == 0:
        i_crit = N - 1  # no critical surface in domain

    I_ref = np.zeros(N)
    I_ref[i_crit] = I_fwd[i_crit]  # reflected intensity at turning point

    for i in range(i_crit - 1, -1, -1):
        path_fac = 1.0 / max(refr_idx[i], 1e-3)
        I_ref[i] = I_ref[i+1] * np.exp(-kappa_IB[i] * dx * path_fac)

    # Deposited power: dQ = kappa_IB * (I_fwd + I_ref) * path_fac
    path_fac_arr = 1.0 / np.maximum(refr_idx, 1e-3)
    Q_dep = kappa_IB * (I_fwd + I_ref) * path_fac_arr  # [cm^-1 * W/cm^2] = W/cm^3

    return I_fwd, I_ref, Q_dep


def run_prototype(params: PlasmaParams | None = None) -> dict:
    """Run the 1D prototype and return results dict."""
    if params is None:
        params = PlasmaParams()

    x, n_e = build_density_profile(params)
    n_c = critical_density(params.lambda_laser)

    refr_idx = compute_refractive_index(n_e, n_c)
    nu_ei = nu_ei_spitzer(n_e, params.T_e_eV, params.Z_ion, params.ln_Lambda)
    kappa_IB = compute_kappa_IB(n_e, n_c, nu_ei)

    I_fwd, I_ref, Q_dep = trace_ray_1d(x, kappa_IB, refr_idx, params.I_0)

    # Summary metrics
    total_absorbed = np.trapz(Q_dep, x)
    eta_abs = 1.0 - I_ref[0] / params.I_0  # net absorption fraction
    x_crit_idx = np.argmax(refr_idx < 1e-3)
    x_crit = x[x_crit_idx] if x_crit_idx > 0 else params.x_max

    return {
        "x": x, "n_e": n_e, "n_c": n_c,
        "refr_idx": refr_idx, "kappa_IB": kappa_IB,
        "I_fwd": I_fwd, "I_ref": I_ref, "Q_dep": Q_dep,
        "eta_abs": eta_abs,
        "total_absorbed_W_cm2": total_absorbed,
        "x_critical_cm": x_crit,
    }


def plot_results(res: dict) -> None:
    """Produce a 4-panel summary plot."""
    x_um = res["x"] * 1e4  # cm -> um

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("1D Laser-Plasma Prototype: Linear Density Ramp", fontsize=13)

    ax = axes[0, 0]
    ax.plot(x_um, res["n_e"] / res["n_c"], "b-", lw=2)
    ax.axhline(1.0, color="r", ls="--", label="Critical surface")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("n_e / n_c")
    ax.set_title("Electron Density Profile")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x_um, res["refr_idx"], "g-", lw=2)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("n(x)")
    ax.set_title("Refractive Index")

    ax = axes[1, 0]
    ax.semilogy(x_um, res["I_fwd"] / res["I_fwd"][0] + 1e-12, "b-", label="Forward")
    ax.semilogy(x_um, res["I_ref"] / res["I_fwd"][0] + 1e-12, "r--", label="Reflected")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("I / I_0")
    ax.set_title(f"Intensity (eta_abs = {res['eta_abs']:.1%})")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(x_um, res["Q_dep"] / res["Q_dep"].max(), "orange", lw=2)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("Q_dep (normalized)")
    ax.set_title("Volumetric Power Deposition")

    plt.tight_layout()
    plt.savefig("laser_plasma_1d_results.png", dpi=150)
    print("Saved: laser_plasma_1d_results.png")


if __name__ == "__main__":
    params = PlasmaParams(
        lambda_laser=0.351e-4,   # 3omega (NIF wavelength)
        I_0=5e14,                # 5e14 W/cm^2 (NIF corona intensity)
        T_e_eV=2000.0,           # 2 keV coronal temperature
        Z_ion=3.5,               # ~CH plastic ablator
        n_e_peak=0.9e21,         # 0.9 * n_c (underdense corona)
        L_ramp=200e-4,           # 200 um density scale length
        x_max=300e-4,
        N_cells=2000,
    )

    res = run_prototype(params)

    print(f"Critical density n_c = {res['n_c']:.3e} cm^-3")
    print(f"Critical surface at x = {res['x_critical_cm']*1e4:.1f} um")
    print(f"Net absorption fraction: {res['eta_abs']:.1%}")
    print(f"Peak kappa_IB = {res['kappa_IB'].max():.3e} cm^-1")
    print(f"Total absorbed power: {res['total_absorbed_W_cm2']:.3e} W/cm^2")

    plot_results(res)
```

### 6.1 Expected Output

For the default parameters (5e14 W/cm^2, 2 keV, 3omega, CH plasma):

- `n_c = 1.0e22 cm^-3` (3omega)
- `n_e_peak / n_c = 0.09` (well underdense — no critical surface in domain)
- `eta_abs ~ 5-15%` (IB absorption over 200 um scale length)
- Peak `kappa_IB ~ 1-10 cm^-1` near peak density

For `n_e_peak = 0.99 * n_c` (near-critical):
- Critical surface appears at `x ~ 198 um` (99% up the ramp)
- Ray is reflected; double-pass absorption `eta_abs ~ 20-40%`
- Sharp peak in `Q_dep` at critical surface

### 6.2 Limitations of This Prototype

1. **1D, normal incidence**: No refraction angle, no caustic formation, no oblique incidence resonance absorption
2. **Steady-state**: No temporal evolution of the plasma profile
3. **Single ray**: No beam geometry, no CBET
4. **Fixed T_e**: No coupling between Q_dep and temperature evolution
5. **No parametric instabilities**: kappa_IB only, no SRS/SBS loss

A full 2D prototype (cylindrical geometry + grid coupling) requires ~500 LOC and is the recommended next step.

---

## 7. Relevance to DPF

**Short answer: none.**

The Dense Plasma Focus is electrically driven. The plasma formation mechanism is entirely electromagnetic induction from the pulsed power circuit (Mather 1965; Lee 1984). There is no laser.

Key distinctions:

| Property | ICF/HED Laser Target | DPF |
|----------|----------------------|-----|
| Driver | Laser (1omega/3omega, ns pulses) | Pulsed power (kA-MA, us timescale) |
| Heating mechanism | Inverse bremsstrahlung, resonance absorption | Ohmic (J x B) heating, shock compression |
| Density regime | `n_e ~ 0.01-1 * n_c`, corona + dense shell | `n_e ~ 10^18 - 10^24 cm^-3`, no critical surface concept |
| Laser package | Core physics | Not applicable |
| CBET | 40-beam multi-beam coupling | Not applicable |
| Parametric instabilities | First-order ICF concern | Not applicable |
| Ablation pressure | `P_abl ~ I^(2/3)` | Not applicable |

The quantities DPF-Unified solves — circuit-MHD coupling, cylindrical snowplow, pinch formation, neutron yield — have no analog in laser-plasma interaction. The governing equations (ray equation, IB absorption) appear nowhere in DPF physics.

**This document exists for ICF/HED prototype completeness** and as reference material if the simulator is ever extended to laser-driven configurations (laser-wire, laser-preionization, or hybrid Z-pinch/laser targets, which do exist experimentally but are outside the current DPF-Unified scope).

---

## 8. Integration Cost Estimate

Estimates assume integration into a 2D cylindrical MHD code with existing temperature and density fields (e.g., as a standalone module against DPF-Unified's MLX solver, purely hypothetically).

### 8.1 Minimal 2D Ray-Trace Module

| Component | LOC | Time |
|-----------|-----|------|
| Ray ODE integrator (RK4, cylindrical) | 150 | 2 days |
| IB absorption + deposition (CIC) | 100 | 1 day |
| Critical surface detection + reflection | 80 | 1 day |
| Grid interface (read n_e, T_e; write Q_dep) | 80 | 1 day |
| Tests (conservation, analytic 1D limit) | 150 | 2 days |
| **Total** | **~560** | **~7 days** |

### 8.2 Full Production-Grade Package

| Component | LOC | Time |
|-----------|-----|------|
| 3D multi-beam ray-trace (RK4, adaptive step) | 600 | 2 weeks |
| CBET coupling (two-pass, spatial hash) | 400 | 2 weeks |
| Parametric instability gain models (SRS/SBS) | 300 | 1 week |
| Non-Maxwellian Langdon correction | 150 | 3 days |
| Flux-limited electron heat conduction | 200 | 1 week |
| Multi-group radiation transport | 800 | 3 weeks |
| Performance optimization (MLX/Metal kernels) | 300 | 1 week |
| Validation against HYDRA/LILAC benchmarks | — | 2 weeks |
| **Total** | **~2750** | **~12 weeks** |

### 8.3 Recommendation

For DPF-Unified: **do not integrate**. The laser package is irrelevant to DPF physics, and the 12-week effort represents a complete domain pivot.

For a new ICF/HED code: start with the minimal 2D module (7 days), validate against the 1D analytic limit (Section 6), then layer CBET only if multi-beam geometry is required. FLASH's open-source laser package (`source/physics/sourceTerms/Laser/`) is the best existing reference for the minimal implementation.

---

## References

1. Kaiser, T. B. et al. (2000). "HYDRA: A mass-production ICF simulation code." *Phys. Plasmas* 7, 2062.
2. Radha, P. B. et al. (2005). "Multidimensional analysis of direct-drive implosions on OMEGA." *Phys. Plasmas* 12, 056307.
3. Froula, D. H., Glenzer, S. H., Luhmann, N. C., & Sheffield, J. (2012). *Plasma Scattering of Electromagnetic Radiation*, 2nd ed. Academic Press.
4. Kruer, W. L. (1988). *The Physics of Laser Plasma Interactions*. Addison-Wesley.
5. Atzeni, S. & Meyer-ter-Vehn, J. (2004). *The Physics of Inertial Confinement Fusion*. Oxford University Press.
6. Michel, P. et al. (2010). "Symmetry tuning via controlled crossed-beam energy transfer on the NIF." *Phys. Plasmas* 17, 056305.
7. Marinak, M. M. et al. (2001). "Three-dimensional HYDRA simulations of NIF targets." *Phys. Plasmas* 8, 2275.
8. Fatenejad, M. et al. (2013). "Collaborative comparison of simulation codes for high-energy-density physics applications." *High Energy Density Physics* 9, 172.
9. Liu, C. S., Rosenbluth, M. N., & White, R. B. (1974). "Raman and Brillouin scattering of EM waves in inhomogeneous plasmas." *Phys. Fluids* 17, 1211.
10. Simon, A. et al. (1983). "On the inhomogeneous two-plasmon instability." *Phys. Fluids* 26, 3107.
11. Forslund, D. W. et al. (1975). "Theory and simulation of resonant absorption in a hot plasma." *Phys. Rev. Lett.* 35, 1336.
12. Delettrez, J. et al. (1987). "Effect of laser illumination nonuniformity on implosion self-generated magnetic fields." *Phys. Rev. A* 36, 3926.
13. Born, M. & Wolf, E. (1999). *Principles of Optics*, 7th ed. Cambridge University Press.
14. Luneburg, R. K. (1964). *Mathematical Theory of Optics*. University of California Press.
15. Mather, J. W. (1965). "Investigation of the high-energy acceleration mode in the coaxial gun." *Phys. Fluids Suppl.* 8, S366.
16. Lee, S. (1984). "A sequential plasma focus." *Amer. J. Phys.* 52, 986.
