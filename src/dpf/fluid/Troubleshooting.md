# Fluid Module Troubleshooting Guide

## Cross-Review Panel Consensus (Date: 2026-02-25)

### Review Panel
- **Python Expert (py-fluid)**: Code quality, bugs, performance — 28 findings (2C, 6H, 12M, 8L)
- **Physics Expert (phys-fluid)**: MHD equations, transport physics, conservation — 31 findings (6C, 8H, 10M, 7L; 6 self-retracted)
- **Cross-Review Synthesizer (xreview-fluid)**: Verification, adjudication, consensus

### Final Verdict Summary

| Severity | Count | Notes |
|----------|-------|-------|
| CRITICAL | 3     | 2 sub-cycling bugs (confirmed), 1 tau_e coefficient (confirmed) |
| HIGH     | 7     | HLL wave speed, missing RK3, EOS defaults, cylindrical CT, Nernst div(B), gyroviscosity |
| MEDIUM   | 11    | Non-conservative pressure, missing 3/2 factor, high-Mach recovery, etc. |
| LOW      | 11    | Eps parameter, naming, caching, sweep order, etc. |
| REJECTED | 9     | 6 phys-fluid self-retractions + 3 xreview rejections |

---

## CRITICAL Issues (Must Fix)

### [C1] ✅ FIXED — Sub-cycling reuses stale `div_q` in `_braginskii_heat_flux` — zero stability benefit
- **File**: `mhd_solver.py:1227-1235`
- **Found by**: py-fluid
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The sub-cycling loop for Braginskii heat conduction reuses `div_q` computed once before the loop (line 1204). Since `div_q` depends on `Te` (via `grad(Te)` and `kappa(Te)`), each sub-step applies a stale operator. The n sub-cycles are mathematically equivalent to a single Forward Euler step of size `dt` — sub-cycling provides **zero** additional stability.
- **Evidence**:
  ```python
  # Line 1204-1211: div_q computed ONCE from initial Te
  div_q = (
      np.gradient(heat_flux[0], dx, axis=0)
      + np.gradient(heat_flux[1], dx, axis=1)
      + np.gradient(heat_flux[2], dx, axis=2)
  )

  # Line 1227-1235: Sub-cycle loop reuses stale div_q
  n_sub = max(1, int(np.ceil(dt / dt_diff)))
  n_sub = min(n_sub, 100)
  dt_sub = dt / n_sub
  Te_new = Te.copy()
  for _ in range(n_sub):
      Te_new += dt_sub * div_q / (ne_safe * k_B)  # div_q is STALE
  ```
- **Mathematical proof**: `T_new = T + n_sub * dt_sub * f(T_initial) = T + dt * f(T_initial)` — identical to a single Forward Euler step regardless of `n_sub`.
- **Correct behavior**: Recompute heat flux and `div_q` from `Te_new` at each sub-step:
  ```python
  Te_new = Te.copy()
  for _ in range(n_sub):
      kappa_par_sub, kappa_perp_sub = braginskii_kappa(ne, Te_new, B_mag)
      grad_Te_sub = np.array([np.gradient(Te_new, dx, axis=i) for i in range(3)])
      # Recompute q_par, grad_perp, heat_flux, div_q from Te_new
      # ...
      Te_new += dt_sub * div_q_sub / (1.5 * ne_safe * k_B)  # NOTE: also fix 3/2 factor (see M2)
  ```
- **Justification**: Any simulation with `dt > dt_diff` (common in DPF — hot plasma, high kappa_par) will be unstable. The sub-cycling cap at 100 further limits effective stability to `100 * dt_diff`.
- **Impact**: All simulations using the Python engine with Braginskii heat conduction enabled.

---

### [C2] ✅ FIXED — Identical sub-cycling bug in standalone `anisotropic_thermal_conduction`
- **File**: `anisotropic_conduction.py:266-274`
- **Found by**: py-fluid
- **Cross-review verdict**: **CONFIRMED**
- **Description**: Exact same stale `div_q` bug as C1 — the Sharma-Hammett limited conduction function reuses `div_q` computed at line 245 in the sub-cycling loop at lines 271-274.
- **Evidence**:
  ```python
  # Line 245-250: div_q computed ONCE
  div_q = (
      np.gradient(heat_flux[0], dx, axis=0)
      + np.gradient(heat_flux[1], dy, axis=1)
      + np.gradient(heat_flux[2], dz, axis=2)
  )

  # Line 266-274: Sub-cycle reuses stale div_q
  n_sub = max(1, int(np.ceil(dt / dt_diff)))
  n_sub = min(n_sub, 100)
  dt_sub = dt / n_sub
  Te_new = Te.copy()
  for _ in range(n_sub):
      Te_new += dt_sub * div_q / (1.5 * ne_safe * k_B)  # div_q is STALE
  ```
- **Proposed fix**: Same as C1 — recompute kappa, grad(Te), heat flux, and div_q at each sub-step.
- **Justification**: This is the standalone module-level equivalent, likely copy-pasted. Both copies must be fixed together.
- **Impact**: All simulations using the standalone anisotropic conduction module.

---

### [C3] ✅ FIXED — tau_e coefficient ~4x too large in `anisotropic_conduction.py` (wrong unit convention)
- **File**: `anisotropic_conduction.py:78` (also line 118)
- **Found by**: phys-fluid (A1)
- **Cross-review verdict**: **CONFIRMED** after independent unit conversion
- **Description**: The electron collision time uses `tau_e = 3.44e5 * Te^{1.5} / (ne * lnL)`. The coefficient `3.44e5` comes from the NRL Plasma Formulary formula which expects `Te` in **eV** and `ne` in **cm^-3**. However, the code passes `Te_safe` in **Kelvin** and `ne_safe` in **m^-3**. Converting to SI units gives:
  ```
  tau_e[K,m^-3] = 3.44e5 * (8.617e-5)^{1.5} * 1e6 * Te[K]^{1.5} / (ne[m^-3] * lnL)
                = 8.71e4 * Te[K]^{1.5} / (ne[m^-3] * lnL)
  ```
  The coefficient should be `8.71e4`, not `3.44e5` — a factor of ~3.95 error.
- **Evidence**:
  ```python
  # Line 78: Uses NRL eV/cm^-3 coefficient on Kelvin/m^-3 inputs
  tau_e = 3.44e5 * Te_safe ** 1.5 / np.maximum(ne_safe * lnL, 1e-30)
  ```
  Compare with `nernst.py:58` which correctly uses explicit physical constants:
  ```python
  numerator = 3.0 * np.sqrt(2.0 * pi) * epsilon_0**2 * np.sqrt(m_e) * (k_B * Te) ** 1.5
  denominator = ne * Z * e_charge**4 * lnL
  ```
- **Correct behavior**: Either use the SI coefficient `8.71e4` or use explicit physical constants as in nernst.py:
  ```python
  # Option A: Corrected shortcut coefficient
  tau_e = 8.71e4 * Te_safe ** 1.5 / np.maximum(ne_safe * Z_eff * lnL, 1e-30)

  # Option B: Full Spitzer formula (preferred — no unit ambiguity)
  from dpf.constants import epsilon_0, m_e
  tau_e = (3.0 * np.sqrt(2 * np.pi) * epsilon_0**2 * np.sqrt(m_e)
           * (k_B * Te_safe)**1.5
           / np.maximum(ne_safe * Z_eff * e_charge**4 * lnL, 1e-300))
  ```
- **Justification**: tau_e is ~4x too large → kappa_parallel is ~4x too large → heat conduction is 4x too fast. This affects all simulations using the standalone `anisotropic_conduction` module. The same error appears at line 118 for `kappa_perp`.
- **Reference**: NRL Plasma Formulary (2019), p.28, Eq. 31. SI conversion: Te[eV] = 8.617e-5 * Te[K], ne[cm^-3] = ne[m^-3] * 1e-6.
- **Impact**: Parallel and perpendicular thermal conductivity both wrong by ~4x in the standalone module. The `_braginskii_heat_flux` in mhd_solver.py uses `dpf.collision.spitzer.braginskii_kappa()` instead, which is not affected.

---

## HIGH Priority Issues

### [H1] ✅ FIXED — HLL wave speed ignores transverse magnetic field — underestimates fast magnetosonic speed
- **File**: `mhd_solver.py:192-199`
- **Found by**: py-fluid (H1) + phys-fluid (M2)
- **Cross-review verdict**: **CONFIRMED** — upgraded from phys-fluid's MEDIUM to HIGH
- **Description**: The HLL Riemann solver computes `cf = sqrt(a^2 + Bn^2/(mu_0*rho))` using only the normal B component. The correct fast magnetosonic speed depends on total |B|. When transverse B is significant (e.g., guide field problems), this underestimates cf, reducing HLL numerical diffusion below what's needed to suppress post-shock oscillations.
- **Evidence**:
  ```python
  # Lines 192-199: Only normal B used
  B_sq_L = Bn_L ** 2       # Missing Bt contribution
  B_sq_R = Bn_R ** 2
  va2_L = B_sq_L / (mu_0 * np.maximum(rho_L, 1e-30))
  cf_L = np.sqrt(a2_L + va2_L)  # This is NOT the fast magnetosonic speed
  ```
  The function signature (lines 183-184) only receives `Bn_L, Bn_R` — transverse B is not passed.
- **Correct behavior**: The upper bound fast speed is `cf^2 = a^2 + |B|^2/(mu_0*rho)`. Pass total B magnitude or transverse B components:
  ```python
  def _hll_flux_1d_core(..., Bn_L, Bn_R, Bt_L, Bt_R, ...):
      B_sq_L = Bn_L**2 + Bt_L**2
      B_sq_R = Bn_R**2 + Bt_R**2
  ```
- **Justification**: With HLLD as default, HLL is a fallback path. But when HLL is used (small grids, NaN fallback), the wave speed error can cause oscillations near MHD shocks with strong transverse fields.
- **Reference**: Miyoshi & Kusano, JCP 208, 315 (2005), Eq. 3.

---

### [H2] ✅ FIXED — `CylindricalMHDSolver` only supports SSP-RK2, no SSP-RK3
- **File**: `cylindrical_mhd.py:486-564`
- **Found by**: py-fluid (H2)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The Cartesian `MHDSolver` supports both `ssp_rk2` and `ssp_rk3` (default). The cylindrical solver hardcodes SSP-RK2. Since cylindrical is the primary DPF coordinate system, this limits temporal accuracy for the primary use case.
- **Evidence**: The `step()` method at lines 538-564 implements only the 2-stage SSP-RK2 combination: `rho_new = 0.5 * rho_n + 0.5 * (rho_1 + dt * rhs2["drho_dt"])`. No `time_integrator` parameter or RK3 path exists.
- **Proposed fix**: Add `time_integrator` parameter to `__init__` and implement `_step_ssp_rk3` following the Cartesian pattern at `mhd_solver.py:2078-2146`.
- **Justification**: API inconsistency between solvers sharing the same base class. Users expect the same options.

---

### [H3] ✅ FIXED — `IdealEOS` defaults to `m_p` (proton mass) instead of `m_d` (deuterium)
- **File**: `eos.py:27`
- **Found by**: py-fluid (H3)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: `IdealEOS.__init__` defaults to `ion_mass=m_p` (1.67e-27 kg). Every other module in `src/dpf/fluid/` uses `m_d` (3.34e-27 kg). The `MHDSolver.__init__` at line 1350 creates `self.eos = IdealEOS(gamma=gamma)` with default `m_p`, while line 1346 sets `self.ion_mass = m_d` — the EOS and solver disagree on ion mass.
- **Evidence**:
  ```python
  # eos.py:15,27 — defaults to proton mass
  from dpf.constants import k_B, m_p
  class IdealEOS:
      def __init__(self, gamma=5.0/3.0, ion_mass: float = m_p, ...):

  # mhd_solver.py:63 — uses deuterium
  _DEFAULT_ION_MASS = m_d
  # mhd_solver.py:1350 — creates EOS with proton mass
  self.eos = IdealEOS(gamma=gamma)  # Gets m_p default
  ```
- **Proposed fix**:
  ```python
  # eos.py
  from dpf.constants import k_B, m_d
  class IdealEOS:
      def __init__(self, gamma=5.0/3.0, ion_mass: float = m_d, ...):
  ```
  Also fix mhd_solver.py:1350: `self.eos = IdealEOS(gamma=gamma, ion_mass=self.ion_mass)`
- **Justification**: Temperature computations via `eos.temperature_from_energy()` will return values 2x too high if this method is ever called with default mass.
- **Impact**: Latent bug — affects any code path that uses `solver.eos` methods. The MHD solver's hot path computes temperature directly, bypassing the EOS.

---

### [H4] ✅ FIXED — `temperature_from_energy` ignores ionization state Z
- **File**: `eos.py:59-62`
- **Found by**: py-fluid (H4) + phys-fluid (E2)
- **Cross-review verdict**: **CONFIRMED** — consensus between both reviewers
- **Description**: The formula `T = (gamma-1)*e*rho / (n*k_B)` where `n = rho/m_i` is missing the `(1+Z)` factor. For a fully ionized plasma, `e = (1+Z)*n_i*k_B*T / ((gamma-1)*rho)`, so the inverse is `T = (gamma-1)*e*rho / ((1+Z)*n_i*k_B)`.
- **Evidence**:
  ```python
  # Line 59-62: Missing (1+Z) divisor
  def temperature_from_energy(self, rho, e_int):
      n = rho / self.mi
      return (self.gamma - 1.0) * e_int * rho / (np.maximum(n, 1e-30) * k_B)
  ```
  The `total_pressure()` method correctly includes Z (via `electron_pressure`), but the inverse `temperature_from_energy` does not, breaking round-trip consistency.
- **Proposed fix**:
  ```python
  def temperature_from_energy(self, rho, e_int):
      n = rho / self.mi
      return (self.gamma - 1.0) * e_int * rho / (np.maximum(n, 1e-30) * (1.0 + self.Z) * k_B)
  ```
- **Justification**: For Z=1 deuterium, temperature is overestimated by 2x. The method breaks the identity: `T_recovered = temperature_from_energy(rho, ion_energy(rho,T) + electron_energy(rho,T))` gives `(1+Z)*T` instead of `T`.
- **Reference**: Huba, NRL Plasma Formulary (2019).

---

### [H5] ✅ FIXED — Cylindrical CT implementation is non-functional (Cartesian CT on cylindrical geometry)
- **File**: `cylindrical_mhd.py:583-615`
- **Found by**: py-fluid (M1, M2, M12) + phys-fluid (C3)
- **Cross-review verdict**: **CONFIRMED** — upgraded from MEDIUM to HIGH (consensus: non-functional code)
- **Description**: Three compounding errors make the cylindrical CT completely non-functional:
  1. **Wrong EMF construction** (lines 600-602): All 3 E-field components are averaged into `E_face_x` only. `E_face_y` and `E_face_z` remain zeros. Correct CT requires mapping tangential E-components to the appropriate faces.
  2. **Cartesian metric** (line 593): `dy=self.dr` uses Cartesian face areas. Cylindrical div(B)=0 requires `(1/r)*d(r*B_r)/dr + dB_z/dz = 0` with r-dependent face areas.
  3. **Unpopulated faces**: E_face_z is never filled, making the axial CT contribution always zero.
- **Evidence**:
  ```python
  # Lines 591-602: Cartesian CT module called with wrong metric
  staggered = cell_centered_to_face(B_3d[0], B_3d[1], B_3d[2],
      dx=self.dr, dy=self.dr, dz=self.dz)  # Cartesian faces, not cylindrical
  E_face_x = np.zeros((self.nr + 1, 1, self.nz))
  E_face_y = np.zeros((self.nr, 2, self.nz))   # Never populated
  E_face_z = np.zeros((self.nr, 1, self.nz + 1))  # Never populated
  for d in range(3):
      E_face_x[:-1, :, :] += 0.5 * E_3d[d, :, :, :] / 3.0  # Wrong: all 3 components
      E_face_x[1:, :, :] += 0.5 * E_3d[d, :, :, :] / 3.0
  ```
- **Proposed fix**: Either implement proper cylindrical CT with r-dependent face areas per Gardiner & Stone (2008), or disable CT in cylindrical mode (force `enable_ct=False`).
- **Justification**: Default is `enable_ct=False` (Dedner cleaning is used instead), so this only affects users who explicitly enable CT in cylindrical mode. Those users get silently incorrect div(B) control.
- **Reference**: Gardiner & Stone, JCP 227, 4123 (2008) — cylindrical CT method.

---

### [H6] ✅ FIXED — Nernst advection violates div(B) = 0
- **File**: `nernst.py:252-306`
- **Found by**: phys-fluid (N1)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The Nernst update advects each B component independently via scalar upwind transport: `B_new[i] = upwind_advect(B[i], v_N, ...)`. Independent component-wise advection does NOT preserve div(B) = 0. The correct approach is to evolve B via the induction equation form: `dB/dt = curl(v_N x B)`, which preserves div(B) by construction.
- **Evidence**:
  ```python
  # Lines from apply_nernst_advection (calls _upwind_advect_component per component):
  Bx_new = _upwind_advect_component(Bx, vN[0], vN[1], vN[2], ...)
  By_new = _upwind_advect_component(By, vN[0], vN[1], vN[2], ...)
  Bz_new = _upwind_advect_component(Bz, vN[0], vN[1], vN[2], ...)
  ```
- **Proposed fix**: Replace with `dB/dt = -curl(E_Nernst)` where `E_Nernst = -v_N x B`:
  ```python
  E_nernst = -cross(v_N, B)  # Already computed in nernst_electric_field()
  dB_dt_nernst = -curl(E_nernst)
  B_new = B + dt * dB_dt_nernst
  ```
- **Justification**: The Nernst effect is a correction to the main induction equation. Introducing div(B) errors in a transport-level correction undermines the main solver's divergence control (whether Dedner or CT).
- **Reference**: Ridgers et al., Phys. Plasmas 15, 092311 (2008), Eq. 8.

---

### [H7] ✅ FIXED — Gyroviscosity `omega_ci` missing Z factor (code contradicts docstring)
- **File**: `viscosity.py:215`
- **Found by**: phys-fluid (V3)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The gyroviscosity function computes `omega_ci = e_charge * B_mag / m_ion`. The docstring at line 202 states `omega_ci = Z * e * B / m_ion`, but the code uses `e_charge` without Z. For Z=1 deuterium this is correct, but for Z>1 ions the cyclotron frequency is Z times higher, making eta_3 Z times too large.
- **Evidence**:
  ```python
  # Line 202 (docstring): "omega_ci = Z * e * B / m_ion"
  # Line 215 (code):
  omega_ci = e_charge * B_mag / m_ion  # Missing Z factor
  ```
- **Proposed fix**: Add Z parameter:
  ```python
  def braginskii_eta3(ni, Ti, B_mag, m_ion=m_d, Z_eff=1.0):
      omega_ci = Z_eff * e_charge * B_mag / m_ion
  ```
- **Justification**: Code contradicts its own documentation. For DPF with electrode ablation (Cu, W), the gyroviscosity would be wrong by a factor of Z (29 for Cu).
- **Reference**: Braginskii (1965), Eq. 2.23.

---

## MEDIUM Priority Issues

### [M1] Non-conservative pressure equation in cylindrical solver (no total energy path)
- **File**: `cylindrical_mhd.py:384`
- **Found by**: phys-fluid (M1/C1)
- **Cross-review verdict**: **CONFIRMED as MEDIUM** (downgraded from phys-fluid's CRITICAL)
- **Description**: The cylindrical solver exclusively uses the primitive pressure equation `dp/dt = -gamma*p*div(v) + (gamma-1)*eta*J^2`, which gives incorrect Rankine-Hugoniot jump conditions at shocks. The Cartesian solver has both conservative and non-conservative paths.
- **Evidence**: `cylindrical_mhd.py:384-385`:
  ```python
  dp_dt = -self.gamma * p * div_v + (self.gamma - 1.0) * ohmic_heating
  ```
- **Correct behavior**: Implement total energy evolution: `dE/dt = -div[(E+p_total)*v - B(v.B)/mu_0] + eta*J^2`.
- **Justification**: Downgraded because the Python engine is documented as a "teaching engine" (CLAUDE.md, MEMORY.md: "MITIGATED R.2: demoted to teaching engine"). Production simulations should use the Metal or Athena++ backends. The Cartesian solver's WENO5 path correctly uses conservative energy.
- **Reference**: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics," Ch. 3 (2009).

---

### [M2] ✅ FIXED — `_braginskii_heat_flux` in mhd_solver.py missing 3/2 factor in energy denominator
- **File**: `mhd_solver.py:1235`
- **Found by**: **xreview-fluid (NEW)** — both reviewers missed this
- **Cross-review verdict**: **NEW FINDING**
- **Description**: The temperature update uses `Te_new += dt_sub * div_q / (ne_safe * k_B)`, but the electron thermal energy is `u_e = (3/2)*n_e*k_B*Te`, so the correct update is `dTe/dt = div(q) / ((3/2)*n_e*k_B)`. The missing 3/2 factor causes temperature to change 50% too fast.
- **Evidence**:
  ```python
  # mhd_solver.py:1235 — WRONG: missing 3/2
  Te_new += dt_sub * div_q / (ne_safe * k_B)

  # anisotropic_conduction.py:274 — CORRECT: has 3/2
  Te_new += dt_sub * div_q / (1.5 * ne_safe * k_B)
  ```
  These are two implementations of the same physics with different denominators. The anisotropic_conduction.py version is correct.
- **Proposed fix**:
  ```python
  # mhd_solver.py:1235
  Te_new += dt_sub * div_q / (1.5 * ne_safe * k_B)
  ```
- **Justification**: The Braginskii heat conduction in mhd_solver.py applies 1.5x the correct temperature change per timestep, in addition to the stale sub-cycling bug (C1). These errors compound: wrong rate AND no sub-cycling stability. The anisotropic_conduction.py module has the correct denominator but shares the sub-cycling bug.
- **Impact**: Simulations using mhd_solver's built-in Braginskii conduction (via `enable_braginskii=True`).

---

### [M3] Pressure recovery from total energy is catastrophically ill-conditioned at high Mach
- **File**: `mhd_solver.py:1726-1731`
- **Found by**: phys-fluid (M5)
- **Cross-review verdict**: **CONFIRMED as MEDIUM** (downgraded from phys-fluid's CRITICAL)
- **Description**: The conservative energy approach recovers pressure via `dp_dt = (gamma-1)*(dE_total_dt - v.dmom + 0.5*v^2*drho - B.dB/mu_0)`. In high-Mach flow (kinetic >> thermal), this subtraction is catastrophically ill-conditioned. Standard fix is dual-energy switching (Bryan et al. 1995).
- **Evidence**: Lines 1726-1731 show the direct subtraction without any conditioning check.
- **Proposed fix**: Implement dual-energy switching: use total energy in high-beta regions, primitive pressure in low-beta.
- **Justification**: Downgraded because the Python engine is a teaching engine. Production high-Mach DPF simulations should use Athena++ which has robust energy handling.
- **Reference**: Bryan G. et al., Computer Physics Communications 89, 149 (1995).

---

### [M4] `curl(E)` uses 2nd-order `np.gradient` even with WENO5 active
- **File**: `mhd_solver.py:1652-1657`
- **Found by**: py-fluid (M4)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The induction equation `dB/dt = -curl(E)` uses 2nd-order central differences for all cells, creating an order mismatch with 5th-order WENO5 density/momentum fluxes. The HLLD B-flux correction (lines 1665-1695) partially mitigates this for interior cells.
- **Evidence**:
  ```python
  curl_E = np.array([
      np.gradient(E_field[2], self.dx, axis=1) - np.gradient(E_field[1], self.dx, axis=2),
      # ... 2nd-order everywhere
  ])
  ```
- **Proposed fix**: Document as known limitation. For higher-order B evolution, use Metal or Athena++ backends.
- **Justification**: The HLLD B-flux correction provides higher-order contributions for interior cells, partially addressing the mismatch. The overall solver convergence order is ~1.86 (limited by MHD nonlinearity), so this doesn't materially degrade accuracy.

---

### [M5] Reflected shock uses frozen slug mass (no mass pickup)
- **File**: `snowplow.py:466`
- **Found by**: py-fluid (L1) + phys-fluid (S3)
- **Cross-review verdict**: **CONFIRMED as MEDIUM** (compromise: py-fluid LOW, phys-fluid HIGH)
- **Description**: During the reflected shock phase, `M_slug = self._M_slug_pinch` is constant. As the reflected shock expands, it should sweep additional fill gas mass: `M = M_pinch + f_mr * rho0 * pi * (r^2 - r_min^2) * z_f`.
- **Proposed fix**:
  ```python
  # In _step_reflected:
  M_slug = self._M_slug_pinch + (self.f_m * self.rho0 * np.pi
           * (r_s**2 - self.r_pinch_min**2) * z_f)
  ```
- **Justification**: The reflected phase is brief and the current treatment is consistent with simplified Lee model descriptions. The missing mass pickup overestimates reflected shock velocity but has minimal impact on the overall current waveform. Splitting the difference between the two reviewers at MEDIUM.
- **Reference**: Lee & Saw, Phys. Plasmas 21, 072501 (2014).

---

### [M6] `_emf_from_fluxes_kernel` is not parallelized (serial bottleneck)
- **File**: `constrained_transport.py:369`
- **Found by**: py-fluid (M6)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The EMF kernel uses `@njit(cache=True)` without `parallel=True`, while the companion `_ct_update_kernel` (line 292) uses `parallel=True`. For 64^3 grids, the triple-nested loop iterates ~260K times in serial.
- **Proposed fix**: Add `parallel=True` and change outer loop to `prange`.
- **Justification**: Amdahl's law bottleneck. CT performance is limited by the serial EMF computation.

---

### [M7] Viscosity boundary stencil order discontinuity
- **File**: `viscosity.py:260-277`
- **Found by**: py-fluid (M8)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The strain rate uses 1st-order one-sided differences at boundaries and 2nd-order central in the interior. This creates O(dx) truncation error spikes at i=1 and i=nx-2.
- **Evidence**:
  ```python
  if i == 0:
      dvx_dx = (vx[1, j, k] - vx[0, j, k]) / dx           # O(dx)
  else:
      dvx_dx = (vx[i + 1, j, k] - vx[i - 1, j, k]) / (2.0 * dx)  # O(dx^2)
  ```
- **Proposed fix**: Use 2nd-order one-sided stencils: `(-1.5*v[0] + 2*v[1] - 0.5*v[2]) / dx`.
- **Justification**: For viscosity-dominated boundary layers (Hartmann layers), the order reduction creates non-physical velocity profiles.

---

### [M8] Nernst advection uses 1st-order upwind — excessively diffusive
- **File**: `nernst.py:253-306`
- **Found by**: py-fluid (M9)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The `_upwind_advect_component` kernel uses first-order upwind differencing. The O(dx) numerical diffusion can exceed the physical Nernst transport on coarse grids.
- **Proposed fix**: Use 2nd-order upwind or TVD (minmod limiter). Note: this is secondary to H6 (the component-wise advection should be replaced entirely with curl-based induction).
- **Justification**: Moot if H6 is fixed (replacing component advection with curl form). Listed for completeness.

---

### [M9] ADI splitting is first-order in 3D
- **File**: `implicit_diffusion.py:219-245`
- **Found by**: phys-fluid (I1) + py-fluid (M7)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: The ADI applies Crank-Nicolson sequentially along x→y→z. In 3D, this Douglas-Gunn ADI is first-order, not second-order (which requires symmetric Strang splitting: x-y-z-z-y-x).
- **Proposed fix**: Either accept first-order temporal accuracy (documented) or implement symmetric Strang splitting.
- **Reference**: Strang, SIAM J. Numer. Anal. 5, 506 (1968).

---

### [M10] ✅ FIXED — SSP-RK3 CT correction uses wrong reference state at stages 2 and 3
- **File**: `mhd_solver.py:2125-2126, 2142-2143`
- **Found by**: py-fluid (M10)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: At RK3 Stage 2, CT is applied as `_apply_ct_correction_numpy(B_2, B_n, dt)` — using `B_n` as the reference for all stages. The CT correction should use the stage-appropriate reference: `B_1` for Stage 2, `B_2` for Stage 3.
- **Evidence**:
  ```python
  # Line 2125-2126 (Stage 2): B_n is reference — should be B_1
  if self.use_ct:
      B_2 = self._apply_ct_correction_numpy(B_2, B_n, dt)

  # Line 2142-2143 (Stage 3): B_n is reference — should be B_2
  if self.use_ct:
      B_new = self._apply_ct_correction_numpy(B_new, B_n, dt)
  ```
- **Proposed fix**:
  ```python
  # Stage 2:
  B_2 = self._apply_ct_correction_numpy(B_2, B_1, dt)
  # Stage 3:
  B_new = self._apply_ct_correction_numpy(B_new, B_2, dt)
  ```
- **Justification**: Introduces O(dt) div(B) errors at each step. Masked by Dedner cleaning (default, mutually exclusive with CT). Only affects users who explicitly enable `use_ct=True`.

---

### [M11] Coulomb logarithm formula inconsistency across modules
- **File**: `anisotropic_conduction.py:74`, `viscosity.py:72`, `nernst.py:96`
- **Found by**: phys-fluid (cross-cutting observation)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: Three different Coulomb logarithm formulas exist:
  - `viscosity.py:72-73`: `lnL = 23 - ln(sqrt(ni_cm3)*Z/Ti_eV^{1.5})` — NRL ion-ion formula
  - `anisotropic_conduction.py:74`: Same ion-ion formula used for electron-ion collisions
  - `nernst.py:96`: Uses centralized `coulomb_log()` from `dpf.collision.spitzer`
  The first two use the ion-ion formula for electron collisions. NRL distinguishes:
  - Electron-ion: `lnL = 24 - ln(sqrt(ne_cm3) / Te_eV)` (for Te > 10*Z^2 eV)
  - Ion-ion: `lnL = 23 - ln(Z^3*sqrt(2*ni_cm3) / Ti_eV^{1.5})`
- **Proposed fix**: Use the centralized `coulomb_log()` from `dpf.collision.spitzer` everywhere, ensuring it selects the correct formula (electron-ion vs ion-ion) based on the collision type.
- **Justification**: Coulomb log varies slowly (10-20 range), so the practical impact is ~10% error in collision times. Still, using the wrong formula is a physics categorization error.

---

## LOW Priority Issues / Improvements

### [L1] ⏸ DEFERRED — WENO-Z epsilon parameter too large for float64 (changing eps destabilizes hybrid scheme)
- **File**: `mhd_solver.py:102`
- **Found by**: py-fluid (L2) + phys-fluid (M3)
- **Cross-review verdict**: **CONFIRMED** — consensus between both reviewers
- **Description**: `eps = 1e-6` reduces WENO-Z effectiveness at critical points. Borges et al. (2008) recommend `eps ~ dx^2` or `1e-36` for float64.
- **Proposed fix**: `eps = 1e-36` for the Python engine (always float64). Keep `1e-6` for Metal engine (float32).
- **Reference**: Borges R. et al., JCP 227, 3191 (2008), Sec. 3.

### [L2] Thomas algorithm zero-pivot produces garbage instead of error
- **File**: `implicit_diffusion.py:62-63, 70-77`
- **Found by**: py-fluid (H5)
- **Cross-review verdict**: **DOWNGRADED from HIGH to LOW**
- **Description**: Zero diagonal pivot causes `continue` (corrupts factorization) and `x[i] = 0.0` (arbitrary). However, for physical diffusion coefficients (positive eta), the Crank-Nicolson diagonal is always `1 + sigma > 0`, making zero pivots impossible in practice. The exact `== 0.0` floating-point comparison also makes triggering extremely unlikely.
- **Proposed fix**: Replace `== 0.0` with `< 1e-300` and return input unchanged as safe fallback.
- **Justification**: Theoretically incorrect but practically unreachable with physical inputs.

### [L3] `tabulated_eos.py` hardcodes `ion_mass=3.34e-27` (magic number)
- **File**: `tabulated_eos.py:393`
- **Found by**: py-fluid (L3)
- **Cross-review verdict**: **CONFIRMED**
- **Proposed fix**: Use `ion_mass: float = m_d` (already imported).

### [L4] Dedner cp=ch is non-optimal in cylindrical solver
- **File**: `cylindrical_mhd.py:402`
- **Found by**: phys-fluid (M6)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: Uses `cp = ch` (damping rate 1). Mignone & Tzeferacos (2010) recommend `cr = ch/dx` for better parabolic damping.
- **Proposed fix**: `cr = ch / min(self.dr, self.dz)` as in the Cartesian solver.
- **Reference**: Mignone & Tzeferacos, JCP 229, 5896 (2010).

### [L5] Nernst function parameter naming: `Z` vs `Z_eff`
- **File**: `nernst.py:42`
- **Found by**: py-fluid (L5)
- **Cross-review verdict**: **CONFIRMED**
- **Proposed fix**: Rename to `Z_eff` consistently.

### [L6] RKL2 coefficients recomputed every call (no caching)
- **File**: `super_time_step.py:41-130`
- **Found by**: py-fluid (L6)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: `rkl2_coefficients(s)` is O(s) with s~4-16, negligible vs diffusion operator cost.
- **Proposed fix**: Add manual cache dict (arrays aren't hashable for `lru_cache`).

### [L7] ADI sweep order bias (always x→y→z)
- **File**: `implicit_diffusion.py:219-245`
- **Found by**: py-fluid (M7)
- **Cross-review verdict**: **DOWNGRADED from MEDIUM to LOW**
- **Description**: Fixed x→y→z ordering creates small directional bias. For isotropic DPF grids, negligible.
- **Proposed fix**: Alternate sweep order based on step count.

### [L8] Electrode BC z-loop is redundant
- **File**: `mhd_solver.py:1879-1883`
- **Found by**: py-fluid (M11)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: `B_th_local` is k-independent, recomputed `nz` times.
- **Proposed fix**: Vectorize over z-dimension using broadcasting.

### [L9] WENO5 boundary zeroing freezes dB_dt at boundary cells
- **File**: `mhd_solver.py:1812-1813`
- **Found by**: py-fluid (M5)
- **Cross-review verdict**: **CONFIRMED as design trade-off**
- **Description**: Zeros dB_dt at first/last 2 cells for WENO5 boundary consistency. Creates frozen-B boundary layer.
- **Proposed fix**: Document the behavior. For production use, route through backends with proper ghost zones.

### [L10] eta_1 and kappa_par coefficients only valid for Z=1 (undocumented)
- **File**: `viscosity.py:149`, `anisotropic_conduction.py:80`
- **Found by**: phys-fluid (V2, A3)
- **Cross-review verdict**: **CONFIRMED**
- **Description**: eta_1 coefficient 0.3 and kappa_par coefficient 3.16 are Z=1 values. For Z>1, these change significantly (Braginskii Table 2). Phase S.6 added Z-dependent kappa in `dpf.collision.spitzer` but the standalone modules still use hardcoded values.
- **Proposed fix**: Accept Z parameter and use Z-dependent Braginskii tables, or delegate to `dpf.collision.spitzer`.
- **Reference**: Braginskii (1965), Table 1-2.

### [L11] Hall term hardcodes Z=1 electron density
- **File**: `mhd_solver.py:1641`
- **Found by**: py-fluid (H6)
- **Cross-review verdict**: **DOWNGRADED from HIGH to LOW**
- **Description**: `ne = rho / self.ion_mass` assumes Z=1. For deuterium this is correct; only wrong for Z>1 materials.
- **Proposed fix**: `ne = Z * rho / self.ion_mass`.
- **Justification**: DPF primary fill gas is deuterium (Z=1). Downgraded because the code comment explicitly acknowledges `# Z=1` and the primary use case is correct.

---

## REJECTED Findings

### [R1] HLLD star transverse velocity formula has wrong mu_0 (phys-fluid M4)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Self-retracted** after careful dimensional analysis. The factor `mu_0 * denom_star_L` correctly converts SI units. The reviewer verified: numerator [T^2 * m/s] / denominator → [m/s]. Formula at `mhd_solver.py:406` is dimensionally correct.

### [R2] Rusanov diffusion has wrong dx normalization (phys-fluid M7)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Self-retracted** after dimensional analysis. The Rusanov diffusion term `0.5 * alpha * (U[i+1]-2U[i]+U[i-1]) / dx` is the standard form for the flux-difference dissipation. Units check out: [m/s * kg/m^3 / m] = [kg/(m^3*s)].

### [R3] Radial force missing factor of 2 (phys-fluid S1)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Self-retracted** after first-principles derivation. `F_rad = mu_0*(f_c*I)^2*z_f/(4*pi*r_s)` matches the magnetic pressure force `P_mag * 2*pi*r_s*z_f` where `P_mag = B_theta^2/(2*mu_0)` and `B_theta = mu_0*I/(2*pi*r_s)`.

### [R4] Adiabatic back-pressure exponent wrong (phys-fluid S2)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Self-retracted**. For 2D cylindrical compression, `V/V0 = r^2/b^2`, so `p = p0*(b/r)^(2*gamma) = p0*(b/r)^{10/3}` for gamma=5/3. This matches the code at `snowplow.py:338`.

### [R5] Ion collision time Z^4 should be Z^2 (phys-fluid V1)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Self-retracted**. Z^4 is correct for like-particle (ion-ion) self-collisions per NRL Plasma Formulary. The distinction is between unlike-particle (e-i) collisions (Z^2) and like-particle (i-i) collisions (Z^4).

### [R6] Viscous heating formula missing factor of 2 (phys-fluid V4)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Self-retracted**. The formula `Q = eta0 * |S^{traceless}|^2` is correct. The factor of 2 on off-diagonals accounts for tensor symmetry: `2*Sxy^2 = Sxy^2 + Syx^2`.

### [R7] Cylindrical induction missing geometric 1/r source (phys-fluid C2)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Rejected by xreview** after verifying `CylindricalGeometry.curl()` at `geometry/cylindrical.py:156-185`. The z-component correctly uses `(1/r)*d(r*B_theta)/dr`:
  ```python
  rBtheta = self.r_2d * B_theta
  curl[2] = self.inv_r_2d * np.gradient(rBtheta, self.dr, axis=0)
  ```
  This IS the `(1/r)*d(rF_theta)/dr` form, not a bare `dF_theta/dr`. The geometric source is correctly included.

### [R8] Missing magnetic tension in momentum equation (phys-fluid C5)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Rejected by xreview** after verifying `geometric_source_momentum()` at `geometry/cylindrical.py:249-256`. The radial source correctly includes:
  ```python
  source[0] = inv_r_2d * (rho*v_theta^2 + pressure + B_theta^2/(2*mu_0) - B_r*B_theta/mu_0)
  ```
  The `B_theta^2/(2*mu_0*r)` magnetic hoop stress and `-B_r*B_theta/(mu_0*r)` tension are both present, matching Stone & Norman (1992) Eq. A5.

### [R9] Energy equation denominator sign in anisotropic_conduction (phys-fluid A2)
- **Originally found by**: phys-fluid
- **Reason for rejection**: **Self-retracted**. The sign convention is internally consistent: `heat_flux = kappa * grad(T)` (positive from hot to cold), `div_q > 0` at cold spots where heat converges, `Te_new += dt * div_q / (1.5*ne*kB)` increases Te at convergence points. Correct.

---

## Cross-Cutting Observations

### O1. Inconsistent floor/ceiling constants across modules
Multiple modules use different floor values: `rho` floor ranges from `1e-20` (mhd_solver) to `1e-30` (eos, nernst). Recommend defining project-wide constants in `dpf/constants.py`.

### O2. Two copies of Braginskii heat conduction logic
`mhd_solver.py:1130-1242` and `anisotropic_conduction.py:1-281` implement the same physics with:
- Different denominators (mhd_solver missing 3/2 — see M2)
- Same sub-cycling bug (C1, C2)
- Different kappa sources (mhd_solver uses `dpf.collision.spitzer`, standalone uses its own with wrong tau_e — see C3)
These should be consolidated into a single implementation.

### O3. Z-dependence neglected throughout transport modules
Viscosity, conduction, Nernst, and Hall all hardcode Z=1 coefficients. For deuterium DPF this is correct. For electrode ablation (Cu Z=29, W Z=74), all transport will be wrong. This limitation should be documented prominently.

### O4. Code quality is consistently high
All 11 files have thorough docstrings, NumPy-style documentation, and clear inline comments. Physics references are cited. This is the strongest aspect of the codebase and significantly aids review.

---

## Priority Fix Order

1. **IMMEDIATE** (C1 + C2 + M2): Fix sub-cycling in both heat conduction implementations, and fix the 3/2 denominator in mhd_solver. Consider consolidating into a single implementation (O2).
2. **IMMEDIATE** (C3): Fix tau_e coefficient in `anisotropic_conduction.py` (3.44e5 → 8.71e4, or use explicit constants).
3. **HIGH** (H1): Fix HLL wave speed to include total B.
4. **HIGH** (H3 + H4): Fix IdealEOS defaults (m_p→m_d) and add (1+Z) to `temperature_from_energy`.
5. **HIGH** (H5): Disable CT in cylindrical mode or implement proper cylindrical CT.
6. **HIGH** (H6): Replace Nernst component-wise advection with curl-based induction.
7. **HIGH** (H2): Add SSP-RK3 to CylindricalMHDSolver.
8. **HIGH** (H7): Add Z factor to gyroviscosity omega_ci.
9. **MEDIUM** (M1-M11): Address in order of impact.
10. **LOW** (L1-L11): Address during routine maintenance.

---

*End of cross-review — xreview-fluid*
*Panel: py-fluid (Python Expert) + phys-fluid (Physics Expert) + xreview-fluid (Cross-Review Synthesizer)*
