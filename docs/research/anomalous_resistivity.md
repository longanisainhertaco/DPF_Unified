# Anomalous Resistivity Models for DPF Z-Pinch: MLX Implementation Research

**Date**: 2026-03-26
**Purpose**: Design document for anomalous resistivity in the MLX MHD solver
**Target**: `src/dpf/metal/mlx_transport.py` (new function) + `src/dpf/metal/mlx_solver.py` (wiring)

---

## 1. Why Anomalous Resistivity Matters for DPF

Classical resistivity models (Spitzer, Lee-More) compute electron-ion scattering
from Coulomb collisions. At the DPF pinch, this dramatically underestimates the
effective resistivity because micro-instabilities dominate scattering.

The mechanism: at pinch, current density J concentrates into a narrow column
(r ~ 1-2 mm). The electron drift velocity

    v_d = |J| / (n_e * e)

exceeds the ion acoustic speed c_s = sqrt(k_B * T_e / m_i), triggering
collective plasma instabilities that scatter electrons far more effectively
than binary Coulomb collisions.

### Quantitative estimate for PF-1000 at pinch

| Parameter | Value | Source |
|-----------|-------|--------|
| Peak current I | 1.8 MA | Scholz 2006 |
| Pinch radius r_p | 1.5 mm | MHD simulation |
| J = I / (pi * r_p^2) | 2.5e11 A/m^2 | Derived |
| n_e at pinch | 1e24 - 1e25 m^-3 | Lee model + MHD |
| v_d = J / (n_e * e) | 1.6e6 m/s (at n_e=1e24) | Derived |
| T_e at pinch | 0.5 - 2 keV | Experimental |
| c_s = sqrt(k_B * T_e / m_i) | 2.2e5 m/s (at 1 keV, D) | Derived |
| v_d / c_s | ~7 | **Well above threshold** |
| v_ti = sqrt(k_B * T_i / m_i) | 2.2e5 m/s (at 1 keV) | Derived |

At n_e = 1e24 and T_e = 1 keV:
- **Spitzer**: eta ~ 1.03e-4 * Z * ln(Lambda) / T_eV^1.5 ~ 3.3e-5 Ohm*m
- **Lee-More**: eta ~ 1e-6 Ohm*m (reduced by saturation at high density)
- **Anomalous (Sagdeev, alpha=0.05)**: eta ~ 5e-5 Ohm*m
- **Anomalous (drift-velocity, v_d=7*v_ti)**: eta ~ 5e-4 Ohm*m

The drift-velocity model gives ~10-100x higher resistivity than Lee-More at
pinch. This controls:
1. **Magnetic field diffusion rate** through the pinch column
2. **Ohmic heating** Q = eta * J^2 (dominant heating mechanism at pinch)
3. **Pinch lifetime** (faster diffusion = shorter confinement)
4. **Current dip depth** (back-EMF from resistive dissipation)
5. **Neutron yield** (determines ion temperature and confinement time)

---

## 2. Models in the Literature

### 2.1 Sagdeev (1966) -- Simple Threshold

The earliest anomalous resistivity model. Step function at v_d > v_crit:

    eta_anom = alpha * m_e * omega_pe / (n_e * e^2)   for v_d > v_crit
            = 0                                         otherwise

where omega_pe = sqrt(n_e * e^2 / (eps_0 * m_e)) is the electron plasma
frequency and alpha ~ 0.01-0.1 is a free turbulence parameter.

- **Pros**: Simple, widely used, single free parameter
- **Cons**: Step function discontinuity; alpha is not self-consistent;
  magnitude independent of HOW MUCH v_d exceeds threshold

**Used by**: MACH2 (NRL) with alpha=0.01-0.1, threshold v_d > c_s.

Reference: Sagdeev, Rev. Plasma Phys. 4:23 (1966).

### 2.2 LHDI -- Lower Hybrid Drift Instability (Davidson & Gladd 1975, Huba 1977)

Lower threshold than ion-acoustic. Triggers first in DPF sheaths:

    Threshold: v_d > (m_e / m_i)^{1/4} * v_ti

For deuterium, (m_e/m_i)^{1/4} ~ 0.129, so the LHDI threshold is ~8x lower
than the ion-acoustic threshold.

The LHDI growth rate peaks at k * rho_e ~ 1 (electron gyroradius scale),
producing anomalous resistivity:

    eta_LHDI ~ (m_e / (n_e * e^2)) * gamma_LHDI

where gamma_LHDI ~ omega_LH * (v_d / v_ti) is the linear growth rate and
omega_LH = sqrt(omega_ce * omega_ci) is the lower hybrid frequency.

Rososhek et al. (2026, Phys. Plasmas) showed that an LHDI-driven model with
saturation bound B/(n_e*e) and quadratic scaling:

    eta_LHDI ~ (v_d/v_ti)^2 / (1 + (v_d/v_ti)^2)

reproduces experimental B-field structure in neon z-pinches when combined with
the Hall term.

**Key insight from Rososhek**: The alpha parameter should SCALE with v_d/c_s,
not be constant. This is exactly what the drift-velocity model (Sec 2.4) does.

References:
- Davidson & Gladd, Phys. Fluids 18:1327 (1975)
- Huba et al., Phys. Fluids B 5:3779 (1993)
- Rososhek, Seyler, Lavine & Hammer, Phys. Plasmas (2026)

### 2.3 Ion-Acoustic Turbulence

When v_d > c_s, ion-acoustic waves are destabilized. The effective collision
frequency from quasi-linear theory:

    nu_eff ~ omega_pe * (W / n_e * k_B * T_e)

where W is the wave energy density. In steady-state saturation:

    nu_eff ~ omega_pi * (v_d / v_th_e)

This gives eta that scales linearly with v_d/v_th_e rather than quadratically.
Less commonly used in modern z-pinch codes because the LHDI and drift-velocity
models capture the physics more accurately at DPF conditions where v_d >> c_s.

Reference: Kadomtsev, Plasma Turbulence (1965), Chapter 6.

### 2.4 Drift-Velocity Model (Faerder 2024, Bychenkov 1988)

The model recommended for DPF implementation. Based on current-driven
instabilities where the effective collision frequency scales quadratically
with the drift velocity ratio:

    v_d = |J| / (n_e * e)                              [m/s]
    v_ti = sqrt(k_B * T_i / m_i)                       [m/s]
    omega_pi = sqrt(n_i * e^2 / (eps_0 * m_i))         [rad/s]

    nu_eff = omega_pi * min((v_d / v_ti)^2, 1.0)       [1/s]
    eta_dv = m_e * nu_eff / (n_e * e^2)                [Ohm*m]

    Applied only when v_d > v_ti (threshold).

The saturation cap min((v_d/v_ti)^2, 1.0) corresponds to the Bohm diffusion
limit: nu_eff cannot exceed omega_pi. This prevents unphysical runaway at
extreme drift velocities.

**Key advantages over Sagdeev**:
1. **No free alpha parameter** -- the (v_d/v_ti)^2 factor self-regulates
2. **Smooth quadratic scaling** instead of step function
3. **Self-consistent magnitude** -- higher drift = higher resistivity
4. **Temperature coupling** -- v_ti ~ sqrt(T_i), creating negative feedback:
   high eta -> more ohmic heating -> higher T_i -> lower v_d/v_ti -> lower eta

Faerder et al. (2024) upgraded MACH2 to use this model and showed better
agreement with experimental x-ray images for DPF z-pinch experiments.

Bychenkov et al. (1988) derived the theoretical basis: the effective collision
frequency from lower-hybrid turbulence saturates as omega_pi * (v_d/v_ti)^2,
providing the (v_d/v_ti)^2 scaling law.

References:
- Faerder et al., Phys. Plasmas (2024) -- DPF drift-velocity implementation
- Bychenkov et al., Sov. J. Plasma Phys. 14:28 (1988) -- theoretical basis

### 2.5 What Production Z-Pinch Codes Use

| Code | Lab | Anomalous Model | Notes |
|------|-----|-----------------|-------|
| MACH2 | NRL | Sagdeev (alpha=0.01-0.1) -> drift-velocity (Faerder 2024) | Upgraded to drift-velocity for better x-ray agreement |
| HYDRA | LLNL | Spitzer + saturation model: nu_anom = min(nu_ii, omega_pi*(v_d/v_ti)^2) | nu_ii upper bound prevents extreme values |
| GORGON | Imperial | No explicit model; HLL numerical resistivity at grid scale | Relies on numerical dissipation |
| SpECTRE | SXS | None (ideal MHD only) | Resistive MHD under development |
| Athena++ | Princeton | User callback CalcMagDiffCoeff_ | Hook exists, no built-in model |
| ZEUS-2D | LANL | Sagdeev with alpha=0.01 | Classic implementation |

---

## 3. Implementation Design for MLX

### 3.1 Data Flow

```
mlx_solver.py step()
  |
  +-- cons_to_prim(U) -> rho, p
  |
  +-- compute_current_density(U, dr, dz, r_cell) -> J_sq  [ALREADY EXISTS]
  |
  +-- compute_resistivity(Te, rho, model="lee_more") -> eta_classical
  |                                                       [ALREADY EXISTS in mlx_transport.py]
  |
  +-- anomalous_resistivity_mlx(J_sq, rho, p, gamma, Z_eff, ion_mass, model)
  |     -> eta_anom                                       [NEW]
  |
  +-- eta_total = max(eta_classical, eta_classical + eta_anom)
  |     -> passed to _do_resistive_diffusion()            [EXISTING pathway]
```

### 3.2 Interface: `anomalous_resistivity_mlx()`

All inputs are 2D arrays (nr, nz) already available in the solver. All
operations are elementwise -- naturally GPU-parallel on MLX.

```python
def anomalous_resistivity_mlx(
    J_sq: np.ndarray,           # |J|^2 [A^2/m^4], shape (nr, nz)
    rho: np.ndarray,            # mass density [kg/m^3]
    p: np.ndarray,              # gas pressure [Pa]
    gamma: float = 5.0 / 3.0,
    Z_eff: float = 1.0,
    ion_mass: float = 3.34358377e-27,  # deuterium
    model: str = "drift_velocity",
    alpha: float = 0.05,        # only used for "sagdeev" model
) -> np.ndarray:
    """Compute anomalous resistivity from micro-instabilities.

    Returns eta_anom [Ohm*m], shape (nr, nz). Zero where threshold
    is not exceeded.
    """
```

### 3.3 Why `max(eta_classical, eta_classical + eta_anom)` Not Just `max`

Anomalous resistivity is ADDITIVE to classical -- they represent independent
scattering mechanisms (Coulomb collisions vs wave-particle interactions).
The total scattering rate is the sum:

    nu_total = nu_Spitzer + nu_anomalous
    eta_total = m_e * nu_total / (n_e * e^2) = eta_Spitzer + eta_anom

This is how MACH2, HYDRA, and our existing `total_resistivity()` in
`anomalous.py` implement it. The `max()` form is sometimes used as a simpler
approximation:

    eta_total = max(eta_classical, eta_anom)

which is fine because eta_anom >> eta_classical when anomalous is active,
and eta_anom = 0 when below threshold.

For the MLX implementation, use the additive form for physical correctness:

    eta_total = eta_classical + eta_anom

### 3.4 Float32 Safety Analysis

| Quantity | Range at DPF pinch | Float32 safe? |
|----------|-------------------|---------------|
| J_sq | 1e20 - 1e22 | Yes (max float32 = 3.4e38) |
| n_e = rho / m_i | 1e20 - 1e26 | Yes |
| v_d = sqrt(J_sq) / (n_e * e) | 1e3 - 1e7 | Yes |
| v_ti = sqrt(k_B*T_i/m_i) | 1e4 - 1e6 | Yes |
| (v_d/v_ti)^2 | 0 - ~100 (capped at 1 by saturation) | Yes |
| omega_pi = sqrt(n_i * e^2 / (eps0 * m_i)) | 1e8 - 1e14 | Yes |
| eta_anom | 0 - 1e-3 | Yes |

No subnormal or overflow concerns. All intermediate quantities are in
comfortable float32 range. No log-space arithmetic needed (unlike
bremsstrahlung where the coefficient 1.42e-40 is subnormal).

---

## 4. Critical Thresholds for PF-1000

### 4.1 Drift Velocity at Pinch

For PF-1000 at peak compression (t ~ 5.4 us):

    I_peak ~ 1.8 MA
    r_pinch ~ 1.5 mm (from MHD)
    J = I / (pi * r^2) = 1.8e6 / (pi * (1.5e-3)^2) = 2.5e11 A/m^2

    At n_e = 1e24 m^-3 (moderate pinch density):
    v_d = J / (n_e * e) = 2.5e11 / (1e24 * 1.6e-19) = 1.56e6 m/s

    At n_e = 1e25 m^-3 (high pinch density):
    v_d = 1.56e5 m/s

### 4.2 Ion Acoustic Speed

    At T_e = 1 keV (1.16e7 K), deuterium:
    c_s = sqrt(k_B * T_e / m_i) = sqrt(1.38e-23 * 1.16e7 / 3.34e-27)
        = 2.22e5 m/s

    v_d / c_s = 1.56e6 / 2.22e5 = 7.0  (at n_e = 1e24)
    v_d / c_s = 0.70                     (at n_e = 1e25)

At moderate pinch densities, v_d/c_s ~ 7 -- well above the threshold.
At very high densities, the threshold may NOT be exceeded.
This is physically correct: dense pinches are more resistive classically
and less driven by anomalous mechanisms.

### 4.3 Ion Thermal Speed (for drift-velocity model)

    At T_i = 0.3 keV (ions cooler than electrons during compression):
    v_ti = sqrt(k_B * T_i / m_i) = 1.28e5 m/s

    v_d / v_ti = 1.56e6 / 1.28e5 = 12.2 (at n_e = 1e24)

    (v_d/v_ti)^2 = 149 -> capped at 1.0 (Bohm limit)
    -> nu_eff = omega_pi (saturated)

### 4.4 Anomalous Resistivity Magnitude

    omega_pi = sqrt(n_i * e^2 / (eps_0 * m_i))
             = sqrt(1e24 * (1.6e-19)^2 / (8.85e-12 * 3.34e-27))
             = 9.3e10 rad/s

    At saturation (v_d >> v_ti):
    eta_anom = m_e * omega_pi / (n_e * e^2)
             = 9.1e-31 * 9.3e10 / (1e24 * (1.6e-19)^2)
             = 3.3e-6 Ohm*m

    Unsaturated at v_d/v_ti = 3:
    eta_anom = 3.3e-6 * 9 = 3.0e-5 Ohm*m

    Compare Lee-More at same conditions: eta_LM ~ 1e-6 Ohm*m
    -> Anomalous is 3-30x larger at pinch

### 4.5 CFL Impact

    eta_anom = 3e-5 Ohm*m, dr = 1.5e-4 m (at 100 radial cells over 1.5 cm):
    dt_resistive = dr^2 * mu_0 / (2 * eta_anom)
                 = (1.5e-4)^2 * 1.26e-6 / (2 * 3e-5)
                 = 4.7e-10 s

    Typical MHD CFL: dt_MHD ~ 1e-9 s
    -> Sub-cycling factor ~ 2-3 (manageable)

    At saturation (eta = 3.3e-6):
    dt_resistive = 4.3e-9 s > dt_MHD -> no sub-cycling needed

---

## 5. Prototype Implementation

```python
import numpy as np

# Physical constants (SI)
_M_E = 9.10938e-31       # electron mass [kg]
_E_CHARGE = 1.602176634e-19
_EPS_0 = 8.854187817e-12
_K_B = 1.380649e-23
_M_D = 3.34358377e-27    # deuterium mass [kg]


def anomalous_resistivity_mlx(
    J_sq: np.ndarray,
    rho: np.ndarray,
    p: np.ndarray,
    gamma: float = 5.0 / 3.0,
    Z_eff: float = 1.0,
    ion_mass: float = _M_D,
    model: str = "drift_velocity",
    alpha: float = 0.05,
) -> np.ndarray:
    """Compute anomalous resistivity from current-driven micro-instabilities.

    Models available:
        "drift_velocity" (default): Faerder 2024 / Bychenkov 1988.
            eta = (m_e * omega_pi / (n_e * e^2)) * min((v_d/v_ti)^2, 1)
            Threshold: v_d > v_ti. Quadratic scaling, self-regulating.
        "sagdeev": Sagdeev 1966.
            eta = alpha * m_e * omega_pe / (n_e * e^2)
            Threshold: v_d > c_s. Step function, free alpha parameter.
        "lhdi": Lower Hybrid Drift Instability.
            eta = alpha * m_e * omega_pe / (n_e * e^2)
            Threshold: v_d > (m_e/m_i)^{1/4} * v_ti. Lower than ion-acoustic.

    All inputs/outputs in SI units. Operates elementwise on 2D arrays.

    Args:
        J_sq: |J|^2 from curl(B), [A^2/m^4], shape (nr, nz).
        rho: Mass density [kg/m^3], shape (nr, nz).
        p: Gas pressure [Pa], shape (nr, nz).
        gamma: Adiabatic index.
        Z_eff: Effective ion charge.
        ion_mass: Ion mass [kg].
        model: Anomalous resistivity model name.
        alpha: Turbulence parameter for "sagdeev"/"lhdi" models.

    Returns:
        eta_anom: Anomalous resistivity [Ohm*m], shape (nr, nz).
            Zero where threshold is not exceeded.

    References:
        Faerder et al., Phys. Plasmas (2024).
        Bychenkov et al., Sov. J. Plasma Phys. 14:28 (1988).
        Sagdeev, Rev. Plasma Phys. 4:23 (1966).
        Davidson & Gladd, Phys. Fluids 18:1327 (1975).
    """
    rho_safe = np.maximum(rho, 1e-20)
    p_safe = np.maximum(p, 1e-12)

    # Derived quantities
    n_i = rho_safe / ion_mass
    n_e = Z_eff * n_i

    # Drift velocity: v_d = |J| / (n_e * e)
    J_mag = np.sqrt(np.maximum(J_sq, 0.0))
    v_d = J_mag / np.maximum(n_e * _E_CHARGE, 1e-30)

    # Ion temperature: T_i = p * m_i / (2 * rho * k_B) [K]
    # Factor 2: pressure = (n_e + n_i) * k_B * T for Z=1
    T_i = p_safe * ion_mass / (2.0 * rho_safe * _K_B)

    # Ion thermal speed
    v_ti = np.sqrt(_K_B * np.maximum(T_i, 1.0) / ion_mass)

    if model == "drift_velocity":
        # Faerder 2024 / Bychenkov 1988
        # nu_eff = omega_pi * min((v_d/v_ti)^2, 1.0)
        # eta = m_e * nu_eff / (n_e * e^2)
        omega_pi = np.sqrt(n_i * _E_CHARGE**2 / (_EPS_0 * ion_mass))

        ratio_sq = np.minimum((v_d / np.maximum(v_ti, 1.0))**2, 1.0)
        nu_eff = omega_pi * ratio_sq

        eta_anom = _M_E * nu_eff / np.maximum(n_e * _E_CHARGE**2, 1e-60)

        # Threshold: only active where v_d > v_ti
        mask = v_d > v_ti
        eta_anom = np.where(mask, eta_anom, 0.0)

    elif model == "sagdeev":
        # Sagdeev 1966: eta = alpha * m_e * omega_pe / (n_e * e^2)
        # Threshold: v_d > c_s = sqrt(k_B * Te / m_i)
        # For this model, use Te ~ Ti (assume rough equipartition)
        T_e = T_i  # approximation; caller can supply 2T state
        c_s = np.sqrt(_K_B * np.maximum(T_e, 1.0) / ion_mass)

        omega_pe = np.sqrt(n_e * _E_CHARGE**2 / (_EPS_0 * _M_E))
        eta_anom = alpha * _M_E * omega_pe / np.maximum(n_e * _E_CHARGE**2, 1e-60)

        mask = v_d > c_s
        eta_anom = np.where(mask, eta_anom, 0.0)

    elif model == "lhdi":
        # LHDI: v_d > (m_e/m_i)^{1/4} * v_ti
        lhdi_factor = (_M_E / ion_mass) ** 0.25
        v_crit = lhdi_factor * v_ti

        omega_pe = np.sqrt(n_e * _E_CHARGE**2 / (_EPS_0 * _M_E))
        eta_anom = alpha * _M_E * omega_pe / np.maximum(n_e * _E_CHARGE**2, 1e-60)

        mask = v_d > v_crit
        eta_anom = np.where(mask, eta_anom, 0.0)

    else:
        raise ValueError(
            f"Unknown anomalous resistivity model: {model!r}. "
            "Options: 'drift_velocity', 'sagdeev', 'lhdi'."
        )

    # Global cap: eta_anom should not exceed ~1e-2 Ohm*m (Bohm limit)
    return np.clip(eta_anom, 0.0, 1e-2)
```

---

## 6. Integration Plan: Wiring Into mlx_solver.py

### 6.1 Current Resistivity Pathway

In `mlx_solver.py` step(), lines 760-785:

```python
# Case A: eta_field passed externally by engine
eta_raw = kwargs.get("eta_field")
_eta_arg = ...  # parse to mx.array or float

# Case B: self-computed from resistivity_model
elif self._resistivity_model != "constant":
    eta_computed = compute_resistivity(Te_eV, rho_np, model=self._resistivity_model, ...)
    _eta_arg = mx.array(eta_computed.astype(np.float32))
```

### 6.2 Proposed Changes

**Step 1: Add `anomalous_model` kwarg to MLXMHDSolver.__init__()**

```python
self._anomalous_model: str | None = kwargs.get("anomalous_resistivity", None)
# Options: None (disabled), "drift_velocity", "sagdeev", "lhdi"
self._anomalous_alpha: float = float(kwargs.get("anomalous_alpha", 0.05))
```

**Step 2: In step(), after computing eta_classical, add anomalous contribution**

```python
# After line 785 (existing compute_resistivity block):
if self._anomalous_model is not None and _eta_arg is not None:
    J_sq_np = np.asarray(compute_current_density(U, self.dr, self.dz, self._r_cell))
    rho_np = np.asarray(mx.maximum(U[IDN], 1e-12))
    p_np = np.asarray(...)  # recover pressure from conserved state

    eta_anom = anomalous_resistivity_mlx(
        J_sq_np, rho_np, p_np,
        gamma=self.gamma, Z_eff=self.Z_eff, ion_mass=self.ion_mass,
        model=self._anomalous_model, alpha=self._anomalous_alpha,
    )
    # Additive: eta_total = eta_classical + eta_anom
    _eta_arg = _eta_arg + mx.array(eta_anom.astype(np.float32))
```

**Step 3: Add `anomalous_resistivity_mlx()` to `mlx_transport.py`**

Place it alongside `compute_resistivity()`. The function signature uses
numpy arrays (not mx.array) because the transport module operates on CPU
in float64 (same as Lee-More and Spitzer).

**Step 4: Config integration**

In `config.py`, add to `FluidConfig` or a new `ResistivityConfig`:

```python
anomalous_resistivity: str | None = None  # "drift_velocity", "sagdeev", "lhdi", None
anomalous_alpha: float = 0.05            # turbulence parameter (sagdeev/lhdi only)
```

### 6.3 Files to Modify

| File | Change | LOC |
|------|--------|-----|
| `src/dpf/metal/mlx_transport.py` | Add `anomalous_resistivity_mlx()` | ~80 |
| `src/dpf/metal/mlx_solver.py` | Wire anomalous into eta computation in step() | ~20 |
| `src/dpf/config.py` | Add anomalous resistivity config fields | ~5 |
| `tests/test_mlx_transport.py` (new or append) | Unit tests for the three models | ~60 |
| **Total** | | **~165** |

### 6.4 Test Plan

1. **Unit: threshold check** -- v_d below threshold -> eta_anom = 0
2. **Unit: saturation cap** -- v_d >> v_ti -> eta_anom capped at omega_pi scaling
3. **Unit: model comparison** -- drift_velocity > sagdeev > lhdi at same v_d/v_ti
4. **Unit: dimensional check** -- output units [Ohm*m] verified against NRL Formulary
5. **Integration: Sod shock** -- anomalous off, verify no regression
6. **Integration: DPF pinch** -- anomalous on, verify eta_field spatial structure
   (high in pinch column, zero in vacuum)
7. **CFL: sub-cycling** -- verify resistive CFL constraint is still respected with
   higher eta from anomalous contribution

### 6.5 Risks

1. **CFL constraint tightening**: At saturation, eta_anom ~ 3e-6 Ohm*m. The
   existing sub-cycling in `apply_resistive_diffusion()` handles this (capped
   at 20 sub-steps). At drift-velocity saturation: dt_res ~ 4e-9 s vs
   dt_MHD ~ 1e-9 s -> ratio ~ 2-3 sub-cycles. Manageable.

2. **Self-regulating feedback oscillation**: High eta -> more ohmic heating ->
   higher T_i -> lower v_d/v_ti -> lower eta. This is physically correct
   (negative feedback) but can cause oscillations within a single timestep if
   eta and temperature are updated explicitly. Mitigation: the operator-split
   Strang scheme already decouples the resistive step from source terms.

3. **Vacuum cell divergence**: In vacuum (rho -> 0), n_e -> 0 and v_d -> inf.
   The floor on n_e (via rho_safe = max(rho, 1e-20)) and the saturation cap
   (ratio_sq capped at 1.0) prevent divergence. The resulting eta_anom in
   vacuum is bounded by m_e * omega_pi / (n_e_floor * e^2) which is large
   but clipped at 1e-2 by the global cap.

---

## 7. Existing Codebase Assets

The `src/dpf/turbulence/anomalous.py` module already implements all three
models (ion-acoustic, LHDI, Buneman) plus CIV and drift-velocity (scalar).
However, it uses Numba @njit and operates on 1D/scalar inputs for the engine's
volume-averaged snowplow model.

The MLX implementation needs:
- Pure numpy (no Numba) for compatibility with the mlx_transport.py pattern
- 2D array inputs matching (nr, nz) grid shape
- Operates alongside Lee-More in the same code path
- Returns spatial eta_anom field, not a scalar

The prototype in Section 5 is the MLX-ready version. The existing
`anomalous.py` code validates the physics; the new code provides the
GPU-friendly interface.

---

## 8. Summary: Recommended Implementation Path

1. **Add `anomalous_resistivity_mlx()` to `mlx_transport.py`** -- the prototype
   from Section 5, with float64 computation for consistency with Lee-More.

2. **Default model: `"drift_velocity"`** -- Faerder 2024 is the most physically
   motivated and self-regulating. No free alpha parameter.

3. **Wire into `mlx_solver.py` step()** -- additive to classical resistivity,
   after the existing `compute_resistivity()` call.

4. **Keep it off by default** -- `anomalous_resistivity=None` in config.
   Enable explicitly for DPF simulations where pinch physics matters.

5. **Validate against the existing `anomalous.py`** -- cross-check that the
   MLX and Numba implementations agree at identical inputs.

6. **Post-implementation**: Run PF-1000 calibration smoke test (fc=0.7, fm=0.08,
   32x64). Verify I_peak within 20% of reference. The anomalous resistivity
   will change the pinch dynamics, so fc/fm may need re-calibration.
