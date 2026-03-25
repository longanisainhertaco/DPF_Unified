# Sprint 5 Implementation Research

## Executive Summary

This document is the validated implementation specification for two critical gaps in the MLX MHD solver that prevent full circuit-plasma coupling and two-temperature physics:

1. **Gap 1 — Plasma Inductance Coupling**: `MLXMHDSolver.coupling_interface()` returns a default `CouplingState` with `Lp=0.0` and `dL_dt=None`. The circuit solver receives zero plasma inductance from the MLX backend, meaning back-EMF is never generated and the circuit sees a fixed external inductance only. The PyTorch Metal solver already computes Lp from the spatially-resolved B-field at `metal_solver.py:2210-2264`.

2. **Gap 2 — Two-Temperature Source Terms**: The engine skips its `_apply_collision_radiation()` step for the MLX backend (correct — the EOS pressure overwrite would clobber dual-energy-recovered pressure). But the Metal solver applies two-temperature source terms inline at `metal_solver.py:1580-1604` after its GPU step. The MLX solver has no equivalent. Electron-ion equilibration, Ohmic heating, and bremsstrahlung cooling are never applied to the electron energy channel.

Both gaps are architectural, not physics, errors. The governing equations and reference implementations exist in the codebase. The fix for each is 50-80 lines of MLX code.

---

## Methodology (DMAIC)

Each gap is analyzed through the Six Sigma DMAIC cycle:

- **Define**: What is broken, what does the customer (circuit solver / physics fidelity) need?
- **Measure**: Current vs expected behavior, quantified with specific field values.
- **Analyze**: Root cause via 5-Whys. Traceable to sprint history and code structure.
- **Improve**: Exact code changes, governing equations, edge case handling.
- **Control**: Validation criteria, regression tests, monitoring.

---

## Gap 1: Plasma Inductance Coupling

### 1.1 Define

**Problem Statement**: The MLX solver's `coupling_interface()` method at `mlx_solver.py:725-733` returns a `CouplingState` initialized only with `current` and `voltage` (set at line 650):

```python
# mlx_solver.py:650
self._coupling = CouplingState(current=current, voltage=voltage)
```

The `CouplingState` dataclass (`core/bases.py:60-87`) has 7 fields:

| Field | Type | Default | What MLX provides | What circuit needs |
|-------|------|---------|-------------------|--------------------|
| `Lp` | float | 0.0 | 0.0 (default) | Plasma inductance [H] from B-field |
| `emf` | float | 0.0 | 0.0 (default) | Unused by circuit (it computes back_emf separately) |
| `current` | float | 0.0 | Set correctly | Current passthrough |
| `voltage` | float | 0.0 | Set correctly | Voltage passthrough |
| `dL_dt` | float or None | None | None (default) | Rate of change [H/s]; None triggers BDF2 fallback |
| `R_plasma` | float | 0.0 | 0.0 (default) | Set by engine, not solver |
| `Z_bar` | float | 1.0 | 1.0 (default) | Set by engine, not solver |

**Impact**: When the engine calls `self.fluid.coupling_interface()` at `engine.py:1045`, it gets `Lp=0.0`. The circuit solver at `rlc_solver.py:283` computes `L_total = L_ext + Lp = L_ext + 0.0`, meaning the plasma column has no inductive load on the circuit. There is no back-EMF (since `dL_dt=0` via BDF2 on constant `Lp=0`). The current waveform will be an undamped LC oscillation with no plasma coupling, completely decoupled from the MHD dynamics.

**Customer**: The `RLCSolver.step()` method at `rlc_solver.py:247-348` and the engine's `CircuitCoupler` at `coupler.py:48-273`.

### 1.2 Measure

**Current behavior** (MLX backend):
- `coupling.Lp = 0.0` every timestep
- `coupling.dL_dt = None` every timestep
- Circuit sees `L_total = L_ext` (constant)
- Back-EMF = 0 always

**Expected behavior** (PyTorch Metal reference):
- `coupling.Lp` computed from B-field at `metal_solver.py:2233-2250`
- `coupling.dL_dt` computed via 1st-order backward difference at `metal_solver.py:2253-2256`
- At peak current (I ~ 400 kA, PF-1000): Lp ~ 10-50 nH, dL_dt ~ 1-10 H/s

**PyTorch Metal implementation** (`metal_solver.py:2210-2264`):

The Metal solver uses a B-field volume-integral approach:
```python
# metal_solver.py:2233-2248
B_theta_sq = B[0] ** 2 + B[1] ** 2              # Cartesian Bx,By -> azimuthal
B_theta_avg = float(torch.mean(torch.sqrt(B_theta_sq)).item())
radial_extent = self.dx * nx
axial_length = self.dx * nz
Lp_est = B_theta_avg * radial_extent * axial_length / (abs(current) + 1e-30)
```

This is the flux-linkage definition: `Lp = Phi / I = <B_theta> * A / I`.

**However**, the engine also has `CircuitCoupler` (`coupler.py:48-273`) which uses the density-weighted Lee formula approach. When `coupling_mode="density_weighted"` and a `CircuitCoupler` instance exists (`engine.py:1071-1077, 1145-1150`), the engine computes Lp from the MHD state dict directly, bypassing the solver's `coupling_interface()`. The solver's Lp is still used as a fallback and for consistency.

### 1.3 Analyze

**Root Cause (5-Whys)**:

1. **Why is Lp=0?** Because `MLXMHDSolver.step()` at line 650 creates `CouplingState(current=current, voltage=voltage)` without computing Lp.
2. **Why was Lp computation omitted?** Sprint 3 focused on the MHD transport pipeline (WENO5-Z, HLLD, SSP-RK3, dual-energy, CT). Circuit coupling was deferred to Sprint 4.
3. **Why wasn't it added in Sprint 4?** Sprint 4 was blocked by the electrode ghost-cell NaN issue (`docs/SPRINT4_FMEA.md`). The HLLD NaN at 1e8 pressure ratio consumed the entire sprint.
4. **Why does the NaN block Lp computation?** It doesn't directly — Lp computation is independent of the Riemann solver. But Sprint 4's scope was consumed by the NaN fix, and Lp was never prioritized.
5. **Why wasn't CircuitCoupler used instead?** CircuitCoupler operates at the engine level and does work for MLX (it reads the state dict). But the solver's own `coupling_interface()` returning Lp=0 means the engine has inconsistent coupling data when the coupler isn't active (e.g., `coupling_mode="lee_only"`).

**Contributing factors**:
- The MLX solver is axisymmetric (r,z) while the Metal solver is 3D Cartesian. The B_theta field is stored in index `IBT` (the 9th conserved variable) for MLX, but as `B[0],B[1]` (Bx,By) for Metal. The Lp formula must be adapted.
- The MLX solver works in Heaviside-Lorentz units (mu_0=1) when `convert_b_si_to_hl=True`. The Lp formula uses mu_0 explicitly, so B-field values must be converted back to SI before computing Lp.

### 1.4 Improve

#### Governing Equation

The Lee model plasma inductance formula (Lee & Saw, Phys. Plasmas 21, 072501, 2014; research DB formula #75):

```
Lp = (mu_0 / 2*pi) * z_sheath * ln(b / r_eff)
```

where:
- `mu_0 = 4*pi*1e-7` H/m (permeability of free space)
- `z_sheath` = sheath axial position [m], from density peak (argmax of column density along z)
- `b` = cathode radius [m] (outer electrode)
- `r_eff` = density-weighted effective radius [m]

The density-weighted radius:

```
r_eff = integral(r * rho * 2*pi*r * dr * dz) / integral(rho * 2*pi*r * dr * dz)
```

integrated over the region `z in [0, z_sheath]`.

For `dL/dt`, the 1st-order backward difference:

```
dL_dt = (Lp(t) - Lp(t - dt)) / dt
```

The engine's `CircuitCoupler` already implements this at `coupler.py:106-203` with BDF2 and monotonicity enforcement. The solver-level computation should use the simpler B-field method (matching the Metal solver) as a consistency check, while the engine-level `CircuitCoupler` provides the authoritative value.

#### Alternative: Volume-integral B-field method (matching Metal solver)

For the MLX cylindrical solver, B_theta is stored directly in `U[IBT]` (conserved variable index 8). The volume integral:

```
W_B_theta = integral(B_theta^2 / (2*mu_0) * dV)
Lp = 2 * W_B_theta / I^2
```

In HL units (mu_0=1): `W_B_theta = integral(B_theta^2 / 2 * dV)`. Convert back to SI: `W_B_theta_SI = W_B_theta * mu_0`.

For the cylindrical grid: `dV = 2*pi*r * dr * dz` at each cell center.

#### Recommended approach: Density-weighted Lee formula (matching CircuitCoupler)

The B-field volume integral is sensitive to electrode BC artifacts on coarse grids (documented in `coupler.py:1-8`). The density-weighted method is more robust. Since the MLX solver has direct access to `rho` and the grid, implement the Lee formula inline.

#### Exact Code Changes

**File**: `src/dpf/metal/mlx_solver.py`

**Change 1**: Add instance variables for Lp tracking after line 170.

```python
# After line 170 (self._coupling = CouplingState())
self._prev_Lp: float = 0.0
self._Lp_max: float = 0.0  # Monotonicity enforcement
self._cathode_radius: float = kwargs.get("cathode_radius", 0.025)  # Default 25mm
```

**Change 2**: Add `_update_coupling()` method (new method, insert before `coupling_interface()` at line 725).

```python
def _update_coupling(
    self,
    U: Any,
    current: float,
    voltage: float,
    dt: float,
) -> None:
    """Update plasma-circuit coupling state from conserved MHD variables.

    Computes plasma inductance using the density-weighted Lee formula:
        Lp = (mu_0 / 2*pi) * z_sheath * ln(b / r_eff)

    where r_eff is the density-weighted effective radius and z_sheath
    is the sheath axial position from the density peak.

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz) in solver units.
    current : float
        Circuit current [A].
    voltage : float
        Capacitor voltage [V].
    dt : float
        Timestep [s].

    References
    ----------
    Lee, S. & Saw, S.H., Phys. Plasmas 21, 072501 (2014).
    """
    import numpy as np
    from dpf.metal.mlx_kernels import IDN, IBT

    rho_np = np.array(U[IDN], copy=False)  # (nr, nz)
    nr, nz = rho_np.shape
    dr = self._grid.dr
    dz = self._grid.dz

    # Radial cell centres
    r_arr = self._r_inner + (np.arange(nr) + 0.5) * dr  # (nr,)

    # Step 1: Sheath position from column density peak
    col_density = np.sum(rho_np * r_arr[:, np.newaxis], axis=0) * dr  # (nz,)
    iz_sheath = int(np.argmax(col_density))
    z_sheath = (iz_sheath + 0.5) * dz

    # Step 2: Density-weighted effective radius
    rho_region = rho_np[:, :iz_sheath + 1]
    r_col = r_arr[:, np.newaxis]
    dV = 2.0 * math.pi * r_col * dr * dz
    mass = rho_region * dV
    total_mass = float(np.sum(mass))

    if total_mass > 0:
        r_eff = float(np.sum(r_col * mass) / total_mass)
    else:
        r_eff = 0.5 * self._cathode_radius

    r_eff = max(r_eff, 1e-6)
    r_eff = min(r_eff, self._cathode_radius * 0.999)

    # Step 3: Lee formula
    if r_eff > 0 and z_sheath > 0:
        Lp = (_MU0 / (2.0 * math.pi)) * z_sheath * math.log(
            self._cathode_radius / r_eff
        )
    else:
        Lp = 0.0

    # Step 4: Monotonicity enforcement
    if Lp > self._Lp_max:
        self._Lp_max = Lp
    else:
        Lp = self._Lp_max

    # Step 5: dL/dt via backward difference
    if self._prev_Lp > 0 and dt > 0:
        dL_dt: float | None = (Lp - self._prev_Lp) / dt
    else:
        dL_dt = None
    self._prev_Lp = Lp

    self._coupling = CouplingState(
        Lp=Lp,
        current=current,
        voltage=voltage,
        dL_dt=dL_dt,
    )
```

**Change 3**: Call `_update_coupling()` in `step()` before returning, at line 649 (before the current `self._coupling = ...` line).

Replace:
```python
# mlx_solver.py:650
self._coupling = CouplingState(current=current, voltage=voltage)
```

With:
```python
self._update_coupling(U, current, voltage, dt)
```

#### Sheath Detection Strategy

The density-peak method (argmax of column density along z) is used because:
- **B_theta extent fails**: In cylindrical MHD, B_theta fills the entire domain outside the sheath, making it unusable for sheath detection (documented in MEMORY.md).
- **Contiguous density fails**: Density decreases behind the sheath at advanced compression, breaking contiguous-region detection.
- **Column density peak is robust**: Integrating `rho * r * dr` at each z gives the mass-weighted axial profile. The peak is the sheath front.

This matches `CircuitCoupler.compute_feedback()` at `coupler.py:145-151`.

#### Edge Cases

1. **No sheath detected** (uniform density): `iz_sheath` points to an arbitrary z. `r_eff` defaults to `0.5 * cathode_radius`. Lp is small but non-zero. The monotonicity clamp prevents it from decreasing later.

2. **Lp decreases** (physically impossible during compression): The `_Lp_max` clamp at lines enforcing `Lp = max(Lp, _Lp_max)` prevents this. Noisy `z_sheath` oscillations (documented in MEMORY.md) would cause back-EMF instability without this clamp.

3. **HL unit correction**: The density `U[IDN]` is in SI (kg/m^3) regardless of B-field unit convention. The Lee formula uses only geometry and density, not B-field, so no HL conversion is needed.

4. **Very early timesteps** (pre-sheath): Before current ramp-up, column density is nearly uniform. Lp is near zero. `dL_dt = None` triggers the circuit solver's internal BDF2 estimate, which returns 0.0 from an empty history.

### 1.5 Control

**Validation criterion**: Lp(t) computed by the MLX solver should match the PyTorch Metal solver's output within 10% for identical initial conditions and grid resolution. Specifically:

- Run a 100-step cylindrical discharge simulation with both MLX and Metal backends
- Compare Lp at each step: `max(|Lp_mlx - Lp_metal| / max(Lp_metal, 1e-12)) < 0.10`
- The Metal solver uses the B-field volume-integral method, so exact agreement is not expected. The CircuitCoupler (density-weighted) is the authoritative source.

**Regression tests** (add to `tests/test_mlx_solver.py`):

1. `test_coupling_interface_returns_nonzero_Lp`: After one step with I=100kA, verify `coupling.Lp > 0`.
2. `test_coupling_Lp_monotonicity`: Run 10 steps with increasing current. Verify Lp is non-decreasing.
3. `test_coupling_dL_dt_finite`: After 2+ steps, verify `coupling.dL_dt is not None` and finite.
4. `test_coupling_Lp_matches_coupler`: Compare solver Lp with `CircuitCoupler.compute_feedback()` Lp on the same state dict. Within 20%.

---

## Gap 2: Two-Temperature Source Terms

### 2.1 Define

**Problem Statement**: The engine's Strang-split collision step at `engine.py:2362-2386` is skipped for `backend in ("metal", "mlx")`:

```python
# engine.py:2369
if self.backend not in ("metal", "mlx"):
    col_cfg = self.config.collision
    ...  # temperature relaxation, pressure overwrite
```

This skip is correct because the collision step overwrites pressure via `self.eos.total_pressure()` at line 2384, which would clobber the dual-energy-recovered pressure that both Metal and MLX solvers compute internally. (Documented in `feedback_engine_physics_bypass.md`.)

But the Metal solver compensates by running two-temperature source terms inline at `metal_solver.py:1580-1604`:

```python
# metal_solver.py:1580-1604
e_electron_in = result.get("e_electron", state.get("e_electron"))
if e_electron_in is not None:
    from dpf.fluid.two_temperature import step_electron_energy
    ...
    e_e_new, Te_new, Ti_new = step_electron_energy(...)
    result["Te"] = np.maximum(Te_new, 1.0)
    result["Ti"] = np.maximum(Ti_new, 1.0)
    result["e_electron"] = e_e_new
```

The MLX solver has no equivalent. After its `step()` returns at `mlx_solver.py:651`:
- `e_electron` is advected by the MHD transport (via `IEE` in the conserved array)
- But no source terms are applied: no electron-ion equilibration, no Ohmic heating to electrons, no bremsstrahlung cooling
- `Te` and `Ti` in the output state dict are derived from single-fluid pressure at `mlx_state.py:234-236`:
  ```python
  Ti = mx.maximum(T_ion, 0.0)
  Te = Ti  # single-fluid default; IEE overrides if non-zero
  ```
  The `IEE` override is noted but not implemented in `to_state_dict()`.

**Impact**: Without two-temperature source terms:
- Electrons and ions never equilibrate (Te = Ti always, from single-fluid EOS)
- Ohmic heating goes into total energy but not specifically into electrons
- Bremsstrahlung cooling never removes energy from the electron channel
- The `e_electron` field is advected but physically meaningless
- Radiation diagnostics report zero bremsstrahlung loss

**Customer**: Physics fidelity for DPF pinch phase. At ne ~ 1e24 m^-3 and Te ~ 100 eV (1.16e6 K), the Spitzer equilibration time is ~100 ns, comparable to the pinch lifetime (~200 ns). Two-temperature effects are physically significant.

### 2.2 Measure

**Current behavior** (MLX backend):
- `e_electron` is advected by MHD transport (upwind passive scalar)
- `Te = Ti = p * m_i / (2 * rho * kB)` (single-fluid, from `mlx_state.py:232-236`)
- No electron-ion energy exchange
- No Ohmic heating to electron channel
- No bremsstrahlung cooling

**Expected behavior** (Metal backend reference, `metal_solver.py:1580-1604`):
- After MHD step, calls `step_electron_energy()` from `two_temperature.py:287-383`
- Applies: compressional work (-p_e * div(v)), Ohmic heating (eta * J^2), bremsstrahlung loss, implicit Spitzer equilibration
- Updates Te, Ti, and e_electron in the result dict

**Quantitative gap** (PF-1000 conditions):
- At ne = 1e24 m^-3, Te = 1.16e6 K (100 eV):
  - Spitzer nu_ei ~ 2.7e10 s^-1 (from `spitzer.py:54-71`)
  - Equilibration rate alpha = nu_ei * dt * 2*m_e/m_d ~ 0.004 per ns
  - Relaxation timescale: ~250 ns (Te-Ti separation persists through pinch)
- Bremsstrahlung power density: ~1.42e-40 * ne^2 * sqrt(Te) ~ 4.3e7 W/m^3
  - Over 200 ns pinch in ~1 cm^3 volume: ~0.86 J radiated

### 2.3 Analyze

**Root Cause (5-Whys)**:

1. **Why are 2T source terms missing?** The MLX `step()` returns at line 651 without calling any source term function on `e_electron`.
2. **Why wasn't it added during Sprint 3?** Sprint 3 focused on the core MHD pipeline. The `e_electron` field was added as a passive scalar (advection only) with source terms planned for Sprint 4.
3. **Why wasn't it added in Sprint 4?** Sprint 4 was consumed by the electrode NaN blocker. Two-temperature source terms were deprioritized.
4. **Why can't the engine handle it?** The engine's `_apply_collision_radiation()` at `engine.py:2362-2386` explicitly skips collision physics for `backend in ("metal", "mlx")` to avoid clobbering dual-energy pressure. This is correct but creates a gap.
5. **Why does Metal work but MLX doesn't?** Metal applies 2T source terms in its `step()` method after GPU computation, on CPU (`metal_solver.py:1580-1604`). This is an inline operator split that runs outside the engine's collision step. MLX needs the same pattern.

**Architectural insight**: The engine skip at line 2369 is a necessary guard, not a bug. The fix must be inside the MLX solver (or immediately after it in the engine, before state dict is consumed). The Metal solver's approach of running `step_electron_energy()` on CPU after GPU transport is the correct pattern.

### 2.4 Improve

#### Governing Equations

**Electron energy density evolution** (Braginskii 1965, Reviews of Plasma Physics Vol. 1):

```
d(rho_e_e)/dt + div(rho_e_e * v) = -p_e * div(v) + Q_ohm + Q_ei - Q_rad
```

where:
- `rho_e_e = (3/2) * n_e * kB * Te` is the electron energy density [J/m^3]
- `p_e = n_e * kB * Te` is the electron partial pressure [Pa]
- `Q_ohm = eta * |J|^2` is Ohmic heating deposited into electrons [W/m^3]
- `Q_ei = 3 * n_e * kB * (Ti - Te) * (m_e / m_i) * nu_ei` is electron-ion equilibration [W/m^3]
- `Q_rad = C_brem * n_e^2 * Z^2 * g_ff * sqrt(Te)` is bremsstrahlung cooling [W/m^3]

The advection term `div(rho_e_e * v)` is already handled by the MHD transport step (IEE is advected as a passive scalar in the conserved array).

**Spitzer electron-ion collision frequency** (NRL Plasma Formulary 2019, p. 34; `spitzer.py:54-71`):

```
nu_ei = (4 * sqrt(2*pi) * n_e * Z * e^4 * ln(Lambda)) /
        (3 * (4*pi*epsilon_0)^2 * sqrt(m_e) * (kB * Te)^(3/2))
```

**Implicit temperature relaxation** (`spitzer.py:298-339`):

```
Te_new = T_eq + (Te - T_eq) * exp(-2 * alpha)
Ti_new = T_eq + (Ti - T_eq) * exp(-2 * alpha)

where:
  alpha = nu_ei * dt * 2 * m_e / m_d
  T_eq = (Z * Te + Ti) / (Z + 1)
```

This is unconditionally stable for all dt (exponential decay toward equilibrium).

#### Where to Add the Source Step

Insert after the SSP-RK3 hyperbolic step and before the state dict output, matching the Metal solver's pattern. Specifically, in `MLXMHDSolver.step()`, after line 641 (Braginskii conduction) and before line 644 (`self._U = U`).

The source terms operate on NumPy arrays (the existing `step_electron_energy()` function is NumPy-based), so convert from MLX to NumPy, apply sources, and convert back. This matches the Metal solver's approach of running 2T on CPU.

#### Exact Code Changes

**File**: `src/dpf/metal/mlx_solver.py`

**Change**: Add a `_do_two_temperature_sources()` method and call it in `step()`.

```python
def _do_two_temperature_sources(
    self,
    U: Any,
    dt: float,
    eta: float | Any | None = None,
) -> Any:
    """Operator-split two-temperature source terms on electron energy.

    Applies electron-ion equilibration, Ohmic heating, and
    bremsstrahlung cooling to the e_electron channel (IEE).

    Runs on CPU via NumPy (reuses two_temperature.step_electron_energy)
    to avoid re-implementing Spitzer physics in MLX.

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz).
    dt : float
        Timestep [s].
    eta : float or mx.array or None
        Resistivity [Ohm*m]. If None, no Ohmic heating.

    Returns
    -------
    mx.array
        Updated U with modified IEE, same shape.
    """
    mx = require_mlx()
    from dpf.metal.mlx_kernels import IDN, IEE, IMR, IMZ, IMT, IEN, IBR, IBZ, IBT, NVAR
    from dpf.metal.mlx_primitives import P_FLOOR, cons_to_prim
    from dpf.fluid.two_temperature import step_electron_energy

    # Extract primitives
    rho, vr, vz, vt, p, Br, Bz, Bt = cons_to_prim(U, self.gamma)

    # Convert to NumPy for two_temperature module
    rho_np = np.array(rho, copy=False)
    vr_np = np.array(vr, copy=False)
    vz_np = np.array(vz, copy=False)
    vt_np = np.array(vt, copy=False)
    p_np = np.array(p, copy=False)
    ee_np = np.array(U[IEE], copy=False)

    # Number densities (fully ionized, Z=1)
    n_i = np.maximum(rho_np / self.ion_mass, 1e-10)
    n_e = n_i  # Z=1

    # Temperatures
    # T = p * m_i / (2 * rho * kB) for fully ionized Z=1
    T_single = p_np * self.ion_mass / (2.0 * np.maximum(rho_np, 1e-30) * _K_B)
    Te = np.maximum(T_single, 1.0)
    Ti = np.maximum(T_single, 1.0)

    # If e_electron is non-trivial, derive Te from it
    ee_sum = float(np.sum(np.abs(ee_np)))
    if ee_sum > 0:
        Te = (2.0 / 3.0) * ee_np / np.maximum(n_e * _K_B, 1e-300)
        Te = np.maximum(Te, 1.0)
        # Ti from total internal energy minus electron energy
        gm1 = self.gamma - 1.0
        KE = 0.5 * rho_np * (vr_np**2 + vz_np**2 + vt_np**2)
        Br_np = np.array(Br, copy=False)
        Bz_np = np.array(Bz, copy=False)
        Bt_np = np.array(Bt, copy=False)
        ME = 0.5 * (Br_np**2 + Bz_np**2 + Bt_np**2)
        E_np = np.array(U[IEN], copy=False)
        e_int = np.maximum(E_np - KE - ME, 0.0)  # total internal energy density
        e_ion = np.maximum(e_int - ee_np, 0.0)
        Ti = (2.0 / 3.0) * e_ion / np.maximum(n_i * _K_B, 1e-300)
        Ti = np.maximum(Ti, 1.0)

    # Resistivity for Ohmic heating
    if eta is not None:
        if hasattr(eta, '__len__'):
            eta_np = np.array(eta, copy=False)
        else:
            eta_np = np.full_like(rho_np, float(eta))
    else:
        eta_np = np.zeros_like(rho_np)

    # J^2 estimate from curl(B) (simplified for cylindrical)
    # J_theta ~ dBr/dz - dBz/dr, J_z ~ (1/r)*d(r*Bt)/dr, J_r ~ -dBt/dz
    # For Ohmic heating, use Q_ohm = eta * J^2
    # Approximate with zero if eta is zero
    J_sq = np.zeros_like(rho_np)
    if np.any(eta_np > 0):
        Br_np = np.array(Br, copy=False)
        Bz_np = np.array(Bz, copy=False)
        Bt_np = np.array(Bt, copy=False)
        # HL→SI if needed (curl(B)/mu0 = J, but in HL mu0=1)
        scale = _SQRT_MU0 if self._convert_b_si_to_hl else 1.0
        Br_si = Br_np * scale
        Bz_si = Bz_np * scale
        Bt_si = Bt_np * scale
        # J_theta = dBr/dz - dBz/dr
        J_theta = np.gradient(Br_si, self.dz, axis=1) - np.gradient(Bz_si, self.dx, axis=0)
        # J_z = (1/r) * d(r*Bt)/dr  (simplified)
        r_arr = np.array(self._grid.r_cell, copy=False)
        r_2d = r_arr[:, np.newaxis] if r_arr.ndim == 1 else r_arr
        rBt = np.maximum(np.abs(r_2d), 1e-10) * Bt_si
        J_z = np.gradient(rBt, self.dx, axis=0) / np.maximum(np.abs(r_2d), 1e-10)
        # J_r = -dBt/dz
        J_r = -np.gradient(Bt_si, self.dz, axis=1)
        J_sq = (J_theta**2 + J_z**2 + J_r**2) / (_MU0**2)

    # Velocity array for div(v) computation
    velocity = np.stack([vr_np, np.zeros_like(vr_np), vz_np], axis=0)

    # Call step_electron_energy
    e_e_new, Te_new, Ti_new = step_electron_energy(
        rho_e_e=ee_np, rho=rho_np,
        velocity=velocity, eta=eta_np,
        J_sq=J_sq, Te=Te, Ti=Ti,
        n_e=n_e, n_i=n_i,
        dx=self.dx, dt=dt,
        Z=self.Z_eff, gaunt_factor=self.gaunt_factor,
        gamma=self.gamma,
    )

    # Update IEE in conserved array
    rows = [U[i] for i in range(NVAR)]
    rows[IEE] = mx.array(e_e_new.astype(np.float32))
    return mx.stack(rows, axis=0)
```

**In `step()`, add the call** after the Braginskii conduction block (after line 641) and before unpacking:

```python
# ── 6.5. Two-temperature source terms ────────────────────────
if self._use_dual_energy and self.enable_bremsstrahlung:
    eta_for_2t = kwargs.get("eta_field")
    U = self._do_two_temperature_sources(U, dt, eta=eta_for_2t)
    mx.eval(U)
```

**In `mlx_state.py:to_state_dict()`**, update the Te/Ti derivation to use IEE when non-zero. After line 236:

```python
# Override Te from IEE if non-trivial
ee = U[IEE]
ee_sum = float(mx.sum(mx.abs(ee)).item())
if ee_sum > 0:
    n_e = rho / self.ion_mass  # Z=1
    Te = (2.0 / 3.0) * ee / mx.maximum(n_e * _K_B, 1e-300)
    Te = mx.maximum(Te, 1.0)
    # Ti from remainder
    gm1 = self.gamma - 1.0
    e_int_total = mx.maximum(p / gm1, 0.0)  # p/(gamma-1) = total internal
    e_ion = mx.maximum(e_int_total - ee, 0.0)
    n_i = rho / self.ion_mass
    Ti = (2.0 / 3.0) * e_ion / mx.maximum(n_i * _K_B, 1e-300)
    Ti = mx.maximum(Ti, 1.0)
```

#### How to Avoid the EOS Pressure Overwrite Problem

The engine's collision step at `engine.py:2384` calls `self.eos.total_pressure(rho, Ti_new, Te_new)` which overwrites the state dict pressure with an EOS-derived value. This clobbers the dual-energy-recovered pressure because:
1. The EOS uses `p = (n_e + n_i) * kB * T_avg` which is independent of the solver's energy equation
2. At shocks, the dual-energy entropy tracer provides a more accurate pressure than the EOS

The MLX two-temperature source terms modify only `e_electron` (IEE) in the conserved array, not the total energy (IEN) or entropy tracer (ISR). The total energy is conserved by the MHD transport. The pressure is recovered from (E, Srho) via the dual-energy switch in `cons_to_prim()`. This avoids the EOS overwrite entirely.

After the source terms run, `Te` and `Ti` are derived from `e_electron` and `p/(gamma-1)` respectively, not from an EOS call. The pressure in the state dict comes from the dual-energy switch, not from the collision step.

#### How e_electron Stays Consistent with Pressure

`step_electron_energy()` (`two_temperature.py:287-383`) modifies `rho_e_e` (electron energy density) via:
1. Compressional work: `rho_e_e -= dt * p_e * div(v)` (explicit)
2. Ohmic heating: `rho_e_e += dt * eta * J^2` (explicit)
3. Radiation loss: `rho_e_e -= dt * Q_rad` (explicit)
4. Equilibration: implicit Spitzer relaxation (exponential decay toward T_eq)

After this, `Te_new` is recovered from the updated `rho_e_e`, and `Ti_new` is derived from total internal energy minus electron energy. The total energy `E` in the conserved array is not modified by the source terms -- it was already updated by the MHD transport step. The pressure `p` is recovered from `E` via dual-energy in the next `cons_to_prim()` call.

This means `e_electron` and `p` can temporarily be inconsistent (e_electron reduced by radiation, but E unchanged). This is acceptable because:
- `e_electron < p/(gamma-1)` is always true (electrons have less energy than the total)
- The electron energy fraction is small (~30-50% of total internal energy for Z=1)
- The inconsistency is resolved at the next timestep when the MHD transport step re-advects both E and e_electron

### 2.5 Control

**Validation criterion**: Te relaxation timescale matches Spitzer formula within 20%.

Specifically:
- Initialize a uniform plasma at ne = 1e22 m^-3, Te = 2e6 K, Ti = 1e5 K, B = 0
- Run 100 steps with dt = 1 ns
- Measure Te decay toward T_eq = (Te + Ti) / 2 = 1.05e6 K
- Fit exponential: `Te(t) = T_eq + (Te_0 - T_eq) * exp(-t/tau)`
- Compare `tau` with Spitzer prediction: `tau = m_d / (2 * m_e * nu_ei)` ~ 250 ns
- Require `|tau_measured - tau_spitzer| / tau_spitzer < 0.20`

**Regression tests** (add to `tests/test_mlx_solver.py`):

1. `test_two_temperature_equilibration`: Verify Te relaxes toward Ti over ~100 steps.
2. `test_two_temperature_bremsstrahlung`: At high Te (1e7 K), verify e_electron decreases from radiation.
3. `test_two_temperature_Te_Ti_different`: After source terms, verify Te != Ti when initial conditions differ.
4. `test_two_temperature_energy_conservation`: Total energy (E in conserved array) is unchanged by source terms (source terms only redistribute between electron and ion channels).
5. `test_two_temperature_matches_metal`: Run identical IC on MLX and Metal. Te, Ti within 15% after 10 steps.

---

## Implementation Checklist (ordered)

Priority order based on dependencies and impact:

### Phase A: Gap 1 -- Plasma Inductance (estimated 30 min)

- [ ] **A.1** Add `_prev_Lp`, `_Lp_max`, `_cathode_radius` instance variables to `__init__` in `mlx_solver.py`
- [ ] **A.2** Implement `_update_coupling()` method in `mlx_solver.py` (Lee formula with density-weighted r_eff)
- [ ] **A.3** Replace `CouplingState(current=current, voltage=voltage)` at line 650 with `self._update_coupling(U, current, voltage, dt)`
- [ ] **A.4** Write 4 regression tests for coupling_interface
- [ ] **A.5** Run existing MLX test suite to verify no regression

### Phase B: Gap 2 -- Two-Temperature Sources (estimated 45 min)

- [ ] **B.1** Implement `_do_two_temperature_sources()` method in `mlx_solver.py`
- [ ] **B.2** Add call in `step()` after Braginskii conduction, gated on `_use_dual_energy`
- [ ] **B.3** Update `mlx_state.py:to_state_dict()` to derive Te/Ti from IEE when non-zero
- [ ] **B.4** Write 5 regression tests for two-temperature physics
- [ ] **B.5** Run full test suite including cross-backend parity

### Phase C: Integration Validation (estimated 20 min)

- [ ] **C.1** Run 50-step cylindrical discharge with MLX backend, verify Lp > 0 at peak current
- [ ] **C.2** Compare I(t) waveform between MLX and Metal backends (should match within 5%)
- [ ] **C.3** Verify Te != Ti during pinch phase with MLX backend
- [ ] **C.4** Check total energy conservation: dE/E < 1e-6 per step

---

## Validation Plan

### Unit Tests

| Test | File | What it verifies |
|------|------|-----------------|
| `test_coupling_interface_returns_nonzero_Lp` | `test_mlx_solver.py` | Gap 1: Lp > 0 after step with I > 0 |
| `test_coupling_Lp_monotonicity` | `test_mlx_solver.py` | Gap 1: Lp non-decreasing over 10 steps |
| `test_coupling_dL_dt_finite` | `test_mlx_solver.py` | Gap 1: dL_dt computed after 2+ steps |
| `test_coupling_Lp_matches_coupler` | `test_mlx_solver.py` | Gap 1: solver Lp within 20% of CircuitCoupler |
| `test_two_temperature_equilibration` | `test_mlx_solver.py` | Gap 2: Te relaxes toward Ti |
| `test_two_temperature_bremsstrahlung` | `test_mlx_solver.py` | Gap 2: e_electron decreases from radiation at high Te |
| `test_two_temperature_Te_Ti_different` | `test_mlx_solver.py` | Gap 2: Te != Ti after source terms |
| `test_two_temperature_energy_conservation` | `test_mlx_solver.py` | Gap 2: Total E unchanged by source terms |
| `test_two_temperature_matches_metal` | `test_mlx_solver.py` | Gap 2: Te/Ti within 15% of Metal |

### Integration Tests

| Test | What it verifies |
|------|-----------------|
| 50-step cylindrical discharge (MLX) | Lp grows during compression, I(t) shows plasma loading |
| Cross-backend I(t) comparison | MLX and Metal current waveforms within 5% |
| Pinch phase Te/Ti separation | Te > Ti during Ohmic heating, Te < Ti during rapid compression |

### Acceptance Criteria

1. All 9 unit tests pass
2. Lp(t) > 0 during compression phase
3. dL_dt > 0 during axial phase (sheath advancing)
4. Te != Ti when two-temperature is enabled
5. Total energy conservation: dE/E < 1e-6 per step
6. No new NaN or negative pressure failures
7. No regression in existing 370+ MLX tests

---

## References

### Primary Sources

1. **Lee, S. & Saw, S.H.** (2014). "Plasma focus ion beam fluence and flux - scaling with stored energy." Phys. Plasmas 21, 072501. -- Lee model Lp formula (research DB formula #75).

2. **Auluck, S.K.H.** (2024). "Poloidal magnetic field in the dense plasma focus." -- Poloidal B-field contributions to inductance (research DB paper #28, #280). 13 equations for B-field topology in DPF.

3. **Braginskii, S.I.** (1965). "Transport Processes in a Plasma." Reviews of Plasma Physics, Vol. 1, pp. 205-311. -- Electron-ion equilibration rate, thermal conductivity coefficients, transport in magnetized plasma.

4. **NRL Plasma Formulary** (2019). pp. 34, 58. -- Spitzer resistivity, collision frequencies, equilibration timescales.

5. **Popovas, A. et al.** (2025). "DISPATCH HLLS: a dual-energy MHD solver for float32 GPU computing." arXiv:2211.02438. -- Entropy-based dual-energy switching criterion used in the MLX solver.

### Codebase References

| File | Lines | Content |
|------|-------|---------|
| `src/dpf/metal/mlx_solver.py` | 650, 725-733 | MLX coupling_interface (Gap 1 location) |
| `src/dpf/metal/metal_solver.py` | 2210-2264 | Metal _update_coupling (reference implementation) |
| `src/dpf/metal/metal_solver.py` | 1580-1604 | Metal two-temperature inline step (reference) |
| `src/dpf/core/bases.py` | 60-87 | CouplingState dataclass |
| `src/dpf/circuit/rlc_solver.py` | 247-348 | RLCSolver.step() consuming Lp/dL_dt |
| `src/dpf/circuit/coupler.py` | 106-203 | CircuitCoupler density-weighted Lp |
| `src/dpf/fluid/two_temperature.py` | 287-383 | step_electron_energy() source terms |
| `src/dpf/collision/spitzer.py` | 54-71, 298-339 | nu_ei, relax_temperatures |
| `src/dpf/engine.py` | 2362-2386 | Engine collision skip for Metal/MLX |
| `src/dpf/engine.py` | 1045-1178 | Circuit subcycle consuming coupling |
| `src/dpf/metal/mlx_kernels.py` | 22 | NVAR=10, IDN..IEE index definitions |
| `src/dpf/metal/mlx_state.py` | 94-169, 200-259 | from_state_dict / to_state_dict |
| `src/dpf/metal/mlx_primitives.py` | 15, 33 | IEE index, prim_to_cons with e_electron |

### Research Database Queries

| Query | Result |
|-------|--------|
| `formulas WHERE name LIKE '%inductance%'` | Formula #75: `dL_p/dt = (mu_0/(2*pi)) * [ln(b/r_p)*v_a + (z/r_p)*v_r]` |
| `formulas WHERE name LIKE '%inductance%'` | Formula #198: `dLp/dt = (mu0/2pi) * [ln(b/rp)*va + (z/rp)*vr]` (duplicate) |
| `papers WHERE authors LIKE '%Auluck%'` | 8 papers, including #28 "Poloidal magnetic field in the DPF" (2024) |
| `papers WHERE authors LIKE '%Lee%'` | #730 "Dimensions and lifetime of the plasma focus pinch" (1996), #738-744 Lee model series |
