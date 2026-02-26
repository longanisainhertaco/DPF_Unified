# Metal GPU Domain — Cross-Review Troubleshooting Guide

**Cross-reviewer**: xreview-metal
**Source reviews**: py-metal (Python expert), phys-metal (Physics expert)
**Files reviewed**: metal_solver.py, metal_riemann.py, metal_stencil.py, metal_transport.py, metal_kernel.py, mlx_surrogate.py, device.py
**Date**: 2026-02-25

---

## CRITICAL

### XR-MTL-C1: Unit System Mismatch Between Transport and Solver Modules — ✅ FIXED

- **File:Line**: `metal_transport.py:69-72` vs `metal_solver.py:832`
- **Found by**: py-metal (C1)
- **Cross-review verdict**: CONFIRMED — verified against source code. The inconsistency is real and affects all Hall/Braginskii + resistive diffusion combinations.
- **Fix applied**: Changed `curl_B_mps()` default `mu_0` parameter from `MU_0` (SI, 4π×10⁻⁷) to `1.0` (HL code units). Now J = curl(B) in HL units, consistent with `metal_solver.py`.

**Description**: `metal_transport.py` computes current density in SI units as `J = curl(B) / mu_0` (lines 69-72: `J[0] = (dBz_dy - dBy_dz) / mu_0`), while `metal_solver.py` computes current density in Heaviside-Lorentz code units as `J = curl(B)` (line 832: `Jx = (dBz_dy - dBy_dz)`, no mu_0 division). The resistive diffusion operator in `metal_solver.py` compensates by converting eta_SI to eta_code via `eta_eff = eta / mu_0_si` at line 822, making resistive diffusion internally self-consistent in isolation. However, when Hall MHD or Braginskii transport operators (which use SI J) are active simultaneously with resistive diffusion (which uses HL J), the B-field updates receive terms with inconsistent magnitudes. The Hall electric field `E_Hall = -J x B / (n_e * e)` computed in transport uses J that is mu_0 (~1.26e-6) times smaller than what the solver would produce. When both operators contribute to dB/dt in the same timestep, the result is physically wrong.

**Evidence**:
- `metal_transport.py:69`: `J[0] = (dBz_dy - dBy_dz) / mu_0` (SI)
- `metal_transport.py:72`: `J[2] = (dBy_dx - dBx_dy) / mu_0` (SI)
- `metal_solver.py:832`: `Jx = (dBz_dy - dBy_dz)` (HL, no mu_0)
- `metal_solver.py:822`: `eta_eff = eta / mu_0_si` (conversion that makes resistive diffusion self-consistent)
- `metal_solver.py:857`: `dB[0] -= dt * eta_eff * lap_B[0]` (uses HL eta_eff)

**Correct behavior**: All electromagnetic quantities should use a single consistent unit system throughout the solver. Either:
(a) Everything in SI: J = curl(B)/mu_0, eta in Ohm-m, Faraday = dB/dt = -curl(E)
(b) Everything in HL (code units): J = curl(B), eta_code = eta_SI/mu_0, same Faraday law

**Proposed fix**: Unify on HL code units throughout. In `metal_transport.py`, remove the `/mu_0` from J computation. Adjust Hall and Braginskii formulas to use HL J consistently. Add a unit system constant at module level (`UNIT_SYSTEM = "heaviside_lorentz"`) and assert consistency at init time. Alternatively, add a `mu_0` parameter to the transport functions and let the solver pass `mu_0=1.0` (HL) or `mu_0=4*pi*1e-7` (SI).

**Impact**: When Hall MHD or Braginskii conduction/viscosity are enabled alongside resistive diffusion, B-field evolution receives terms differing by factor mu_0 ~ 1.26e-6. This produces catastrophically wrong magnetic field dynamics. Pure resistive MHD (no Hall/Braginskii) is unaffected because the solver's internal HL conversion is self-consistent.

---

## HIGH

### XR-MTL-H1: HLLD Star-State Denominator Uses torch.abs() — Sign Flip — ✅ FIXED

- **File:Line**: `metal_riemann.py:911`
- **Found by**: phys-metal (M1)
- **Cross-review verdict**: CONFIRMED — verified against Miyoshi & Kusano (2005) Eq. 43-44. Standard implementations do not take absolute value.
- **Fix applied**: Replaced `torch.abs()` with sign-preserving clamp using `torch.where(torch.abs(D) < 1e-20, 1e-20, D)` for both D_L and D_R.

**Description**: The HLLD solver computes the inverse denominator for transverse star-state velocities as `inv_rhoL_dSL = 1.0 / torch.clamp(torch.abs(rhoL * (SL - vnL) * (SL - SM) - Bn_sq), min=1e-20)`. The `torch.abs()` forces the denominator positive, which is correct for preventing division by zero but incorrect for the physics. The denominator `D_L = rho_L * (S_L - v_n,L) * (S_L - S_M) - B_n^2` can legitimately be negative when `B_n^2 > rho_L * (S_L - v_n,L) * (S_L - S_M)`, i.e., in moderate-to-strong normal magnetic field regimes. Taking abs() flips the sign of transverse velocity and magnetic field star states, producing incorrect intermediate states.

**Evidence**:
- `metal_riemann.py:911`: `inv_rhoL_dSL = 1.0 / torch.clamp(torch.abs(rhoL * (SL - vnL) * (SL - SM) - Bn_sq), min=1e-20)`
- Miyoshi & Kusano (2005), Eq. 43: `v_{t,L}^* = v_{t,L} - B_n * B_{t,L} * (S_M - v_{n,L}) / D_L` where D_L can be negative
- Same pattern at line ~940 for the right state

**Correct behavior**: The denominator D_L should preserve its sign. Use `torch.sign(D_L) * torch.clamp(torch.abs(D_L), min=1e-20)` or equivalently `torch.where(torch.abs(D_L) < 1e-20, torch.ones_like(D_L) * 1e-20, D_L)` to clamp magnitude without flipping sign.

**Proposed fix**:
```python
D_L = rhoL * (SL - vnL) * (SL - SM) - Bn_sq
safe_D_L = torch.where(torch.abs(D_L) < 1e-20, torch.full_like(D_L, 1e-20), D_L)
inv_rhoL_dSL = 1.0 / safe_D_L
```
Apply same pattern to D_R on the right side.

**Impact**: Incorrect transverse star states in moderate B_n regimes. The HLLD solver degrades to less accurate than HLL for oblique MHD shocks. The existing NaN→HLL fallback (lines 1082-1091) may mask some failures, but incorrect (non-NaN) results pass through silently.

---

## MEDIUM

### XR-MTL-M1: Simplified CT Projection, Not True Gardiner-Stone CT

- **File:Line**: `metal_stencil.py:212-315`, `metal_solver.py:1130-1242`
- **Found by**: py-metal (M1), phys-metal (M2) — both reviewers agree
- **Cross-review verdict**: CONFIRMED — the code comments at metal_solver.py:1143-1146 acknowledge this is "simple CT averaging" not the full Gardiner-Stone (2005) upwind CT.

**Description**: The CT implementation uses arithmetic averaging of face EMFs to compute edge EMFs: `Ez_edge = 0.25 * (Ez_xface[:-1,:,:] + Ez_xface[1:,:,:] + Ez_yface[:,:-1,:] + Ez_yface[:,1:,:])` (metal_stencil.py:258-261). True Gardiner-Stone CT uses upwind-biased EMF averaging that accounts for wave propagation direction, providing better shock-capturing and div(B) control. The current implementation is the "simple CT" of Balsara & Spicer (1999), which is first-order in time and can introduce spurious oscillations near MHD shocks.

**Evidence**:
- `metal_stencil.py:258-261`: Simple arithmetic average of 4 face EMFs
- `metal_solver.py:1143`: Comment acknowledges simplified nature
- Gardiner & Stone (2005), JCP 205: Algorithm 4 describes upwind CT with contact/rotational wave speed weighting

**Correct behavior**: Upwind CT (Gardiner & Stone 2005) uses characteristic speeds at cell edges to weight the EMF contributions, providing 2nd-order accuracy and better shock-capturing.

**Proposed fix**: Implement upwind CT with wave-speed weighting. This requires passing HLL wave speeds (S_L, S_R) from the Riemann solver to the CT step. Estimated ~150 LOC addition. Alternatively, document the limitation and recommend Athena++ backend for applications requiring rigorous div(B) control near shocks.

**Impact**: Spurious magnetic monopoles accumulate near MHD discontinuities. For smooth flows and weak shocks, the simple CT is adequate. For strong MHD shocks (Brio-Wu, Orszag-Tang at late times), div(B) errors can reach ~1% of |B|, potentially affecting shock structure.

### XR-MTL-M2: Viscous Heating Uses Isotropic Formula in Anisotropic Mode — ✅ FIXED

- **File:Line**: `metal_transport.py:516-519`
- **Found by**: phys-metal (M3)
- **Cross-review verdict**: CONFIRMED — when `full_braginskii=True`, the stress tensor uses both eta_0 (parallel) and eta_1 (gyroviscous), but the heating term `Q_visc = eta0 * strain_magnitude_sq` only includes the isotropic eta_0 contribution.
- **Fix applied**: Replaced isotropic `eta0 * |S|²` with full tensor contraction `Q_visc = sigma_ij * S_ij`. Works correctly for both isotropic and anisotropic modes.

**Description**: In `full_braginskii` mode, the viscous stress tensor correctly uses anisotropic coefficients (parallel eta_0 and gyroviscous eta_1). However, the viscous heating returned at line 516 is `Q_visc = eta0 * strain_rate_sq`, which is the isotropic dissipation rate. The correct anisotropic heating is `Q = sigma_{ij} * S_{ij}` where sigma is the full anisotropic stress tensor. This underestimates heating perpendicular to B and overestimates along B.

**Evidence**:
- `metal_transport.py:516`: `Q_visc = eta0 * strain_rate_sq` (isotropic formula)
- `metal_transport.py:460-510`: Full anisotropic stress tensor computation with eta_0, eta_1
- Braginskii (1965), Rev. Plasma Phys. 1:205: Eq. 2.21 gives Q = pi_ij * W_ij with full tensor contraction

**Correct behavior**: `Q_visc = sum(sigma_ij * S_ij)` where sigma is the full anisotropic stress tensor already computed in the function.

**Proposed fix**: After computing the full stress tensor `stress` and strain rate `S_ij`, compute `Q_visc = torch.sum(stress * S_ij)` (element-wise product and sum). This is ~3 lines of code.

**Impact**: Viscous heating budget is wrong when `full_braginskii=True` and the magnetic field is not aligned with the flow. The error is bounded by the ratio eta_1/eta_0, which is typically small in strongly magnetized plasmas but can be order-unity in weakly magnetized regions.

### XR-MTL-M3: Braginskii Conduction Sub-Cycling Reuses Stale Flux — ✅ FIXED

- **File:Line**: `metal_transport.py:290-310`
- **Found by**: py-metal (M5)
- **Cross-review verdict**: CONFIRMED — the sub-cycling loop computes `div_q` once outside the loop and applies it repeatedly.
- **Fix applied**: Moved temperature gradient, flux limiter, heat flux, and div_q computation inside the sub-cycling loop. Each sub-step now recomputes from updated Te.

**Description**: The Braginskii thermal conduction sub-cycling computes `div_q = div(kappa * grad(T))` once, then applies `Te += (dt_sub) * div_q` for each sub-step. Since the temperature field changes after each sub-step, the heat flux should be recomputed. Reusing the initial flux is equivalent to Forward Euler with a larger timestep, defeating the purpose of sub-cycling for stability.

**Evidence**:
- `metal_transport.py:~295`: `div_q` computed once from initial Te
- `metal_transport.py:~305`: Loop applies `Te += dt_sub * div_q` without recomputing div_q

**Correct behavior**: Recompute `div_q` from the updated Te at each sub-step. This makes the sub-cycling a proper multi-step explicit integration.

**Proposed fix**: Move the `div_q` computation inside the sub-cycling loop. Performance cost: N_sub gradient evaluations instead of 1. If N_sub is typically 2-4, this is acceptable.

**Impact**: Sub-cycling provides no additional stability over a single step with the same total dt. For stiff conduction problems (high kappa_parallel), the explicit step can go unstable despite nominally respecting the sub-cycled CFL.

### XR-MTL-M4: SSP-RK3 Operates on Primitive Variables

- **File:Line**: `metal_solver.py:1078-1124`
- **Found by**: py-metal (M2)
- **Cross-review verdict**: CONFIRMED as design limitation, not a bug. Many production codes (including early FLASH versions) use primitive-variable formulations. The SSP property is strictly only guaranteed for conservation-form updates.

**Description**: The SSP-RK3 stages in `_step_ssp_rk3()` operate on primitive variables (rho, v, p, B) rather than conservative variables (rho, rho*v, E, B). The strong-stability-preserving property of Shu-Osher (1988) RK methods is proven only for total-variation-diminishing schemes in conservation form. Operating on primitives can introduce conservation errors at shocks and lose the TVD guarantee.

**Evidence**:
- `metal_solver.py:1078-1124`: RK stages compute `rhs = self._compute_rhs(state)` where state contains primitive (rho, v, p, B)
- `metal_solver.py:1090`: `state1[key] = state_n[key] + dt * rhs[key]` — direct update of primitives
- Shu & Osher (1988): SSP property proven for `U^{n+1} = U^n + dt * L(U^n)` in conservation form

**Correct behavior**: Convert primitives → conservatives before RK stages, apply RK in conservation form, convert back after each stage.

**Proposed fix**: Add `_prim_to_cons()` and `_cons_to_prim()` methods to MetalMHDSolver. Wrap RK stages in conversion calls. Estimated ~80 LOC. Low priority since current approach works well for smooth flows and weak shocks.

**Impact**: Conservation errors at strong shocks. The Rankine-Hugoniot jump conditions may not be satisfied exactly. For weak shocks (Mach < 3), the error is small. For strong shocks (Mach > 10), this can produce incorrect shock speeds.

### XR-MTL-M5: Hardcoded mu_0_si in Resistive Diffusion — ✅ FIXED

- **File:Line**: `metal_solver.py:821`
- **Found by**: py-metal (M3)
- **Cross-review verdict**: CONFIRMED — the value `mu_0_si = 4 * np.pi * 1e-7` is hardcoded locally rather than imported from `dpf.constants`.
- **Fix applied**: Replaced hardcoded value with `from dpf.constants import mu_0` and use canonical `mu_0` from scipy.constants.

**Description**: The resistive diffusion operator defines `mu_0_si = 4 * np.pi * 1e-7` as a local constant at line 821 instead of importing from `dpf.constants.MU_0`. This creates a maintenance risk if the constants module is ever updated (e.g., to use the exact 2019 SI redefinition value).

**Evidence**:
- `metal_solver.py:821`: `mu_0_si = 4 * np.pi * 1e-7`
- `dpf/constants.py` defines `MU_0` as the canonical source of truth

**Proposed fix**: Replace with `from dpf.constants import MU_0` and use `MU_0` in place of `mu_0_si`.

**Impact**: Low. The numerical values are identical to ~15 significant figures. But the code duplication creates a maintenance hazard.

---

## LOW

### XR-MTL-L1: tau_e Coefficient Mismatch (~25% Error) — ✅ FIXED

- **File:Line**: `metal_transport.py:217`
- **Found by**: phys-metal (L1)
- **Cross-review verdict**: CONFIRMED — the coefficient 3.44e5 corresponds to NRL Formulary with Te in eV and ne in cm^-3, but the function receives Te in Kelvin and ne in m^-3.
- **Fix applied**: Now uses `Te_eV` and `ne_cm3` (already computed for Coulomb logarithm) with the NRL coefficient, and includes `Z_eff` in the denominator.

**Description**: The electron collision time `tau_e = 3.44e5 * Te_safe**1.5 / (ne_safe * lnLambda * Z_eff)` uses the NRL Formulary shorthand coefficient. However, the inputs are in SI (Kelvin, m^-3). The correct SI coefficient involves `(4*pi*epsilon_0)^2 * (2*k_B)^{3/2}` etc. The NRL shorthand absorbs unit conversions that don't match the input units, producing ~25% systematic error in the collision time.

**Evidence**:
- `metal_transport.py:217`: `tau_e = 3.44e5 * Te_safe ** 1.5 / (ne_safe * lnLambda * Z_eff)`
- NRL Formulary (2019), p. 34: `tau_e = 3.44e5 * T_eV^{3/2} / (n_cm3 * ln_Lambda)` where T in eV, n in cm^-3
- SI formula: tau_e = (12 * pi^{3/2} * epsilon_0^2 * m_e^{1/2} * (k_B * T_K)^{3/2}) / (n_m3 * Z * e^4 * ln_Lambda)

**Proposed fix**: Use the full SI formula with explicit constants, or convert inputs to NRL units (Te_eV = Te_K / 11604.5, ne_cm3 = ne_m3 * 1e-6) before applying the NRL coefficient.

**Impact**: ~25% systematic error in all Braginskii transport coefficients (kappa_parallel, kappa_perp, eta_visc). Affects conduction and viscosity magnitudes but not their qualitative behavior.

### XR-MTL-L2: P_FLOOR Inconsistency Between Modules — ✅ FIXED

- **File:Line**: `metal_riemann.py:61` vs `metal_transport.py:33`
- **Found by**: py-metal (m3)
- **Cross-review verdict**: CONFIRMED — `metal_riemann.py` uses `P_FLOOR = 1e-12` while `metal_transport.py` uses `P_FLOOR = 1e-20`.
- **Fix applied**: Changed `metal_transport.py` P_FLOOR from 1e-20 to 1e-12 to match `metal_riemann.py` (safe for float32).

**Description**: Two different pressure floor values are used in the Metal domain. The Riemann solver uses 1e-12 (more conservative, appropriate for float32 where 1e-20 would underflow). The transport module uses 1e-20 (appropriate for float64 but problematic in float32 mode where it underflows to subnormal/zero).

**Proposed fix**: Define `P_FLOOR` once in a shared constants location (e.g., `metal/__init__.py` or `dpf/constants.py`). Use 1e-12 for float32 mode and 1e-20 for float64 mode.

**Impact**: In float32 mode, `P_FLOOR = 1e-20` in transport underflows to zero, providing no floor protection. Could allow division-by-zero in transport coefficient calculations.

### XR-MTL-L3: CT Correction Per-Stage Uses B_n as Fixed Reference

- **File:Line**: `metal_solver.py:1095-1100`
- **Found by**: phys-metal (L2)
- **Cross-review verdict**: CONFIRMED — the CT correction after each RK stage uses `B_n` (the state at the beginning of the timestep) as the reference for computing div(B) corrections, rather than the current stage's B.

**Description**: In SSP-RK3, each stage produces an intermediate B. The CT correction is applied using `B_n` as the solenoidal reference throughout all stages. This means stages 2 and 3 correct div(B) relative to the initial state, not relative to the evolving intermediate state. The error is O(dt) per stage.

**Proposed fix**: Pass the current stage's B as the reference for CT correction. Requires storing B at each intermediate stage.

**Impact**: Small. The CT correction already accumulates O(dt^2) errors from the simplified averaging. The reference-state error is subdominant.

### XR-MTL-L4: Duplicated Utility Functions

- **File:Line**: `metal_stencil.py` and `metal_riemann.py` both define `_ensure_mps()` and `_check_no_nan()`
- **Found by**: py-metal (M4)
- **Cross-review verdict**: CONFIRMED — pure code quality issue, no physics impact.

**Description**: Both `metal_stencil.py` and `metal_riemann.py` independently define `_ensure_mps()` (device transfer) and `_check_no_nan()` (NaN detection) with identical logic.

**Proposed fix**: Move to `metal/__init__.py` or a `metal/utils.py` module and import in both files.

**Impact**: None (functional). Maintenance burden only.

### XR-MTL-L5: metal_kernel.py Is Dead Code

- **File:Line**: `metal_kernel.py` (entire file, 197 lines)
- **Found by**: py-metal (m1)
- **Cross-review verdict**: CONFIRMED — the file requires a compiled `.metallib` at `src/dpf/metal/build/default.metallib` which does not exist. The `MetalKernelWrapper` class is never instantiated by any other module in the codebase.

**Description**: `metal_kernel.py` provides a PyObjC-based wrapper for native Metal Shading Language compute kernels. It loads a `.metallib` file, creates compute pipelines, and dispatches kernel groups. However, the `.metallib` file was never compiled (no build script produces it), and no module imports `MetalKernelWrapper`. The `_compute_rhs_native()` method in `metal_solver.py` references it but is itself dead code (guarded by `if self._use_native_metal` which is always False).

**Proposed fix**: Either delete the file or document it as a prototype for future native Metal kernel development. If kept, add a `# STATUS: PROTOTYPE — not functional` header.

**Impact**: None. The file is never executed. It adds 197 lines of unmaintained dead code.

### XR-MTL-L6: Dead Code in metal_solver.py

- **File:Line**: `metal_solver.py:464-662`
- **Found by**: py-metal (m2)
- **Cross-review verdict**: CONFIRMED — `_compute_rhs_native()` and `_dispatch_sweep()` are only reachable if `self._use_native_metal` is True, which requires the non-existent `.metallib` file.

**Description**: ~200 lines of native Metal dispatch code that is never executed. It references `MetalKernelWrapper` from the dead `metal_kernel.py`.

**Proposed fix**: Remove or gate behind an explicit `EXPERIMENTAL_NATIVE_METAL` flag with a clear warning.

**Impact**: None functionally. Code bloat only.

---

## REJECTED

### XR-MTL-R1: py-metal C2 — "HLLD Missing Final NaN Fallback"

- **File:Line**: `metal_riemann.py:1082-1091`
- **Found by**: py-metal (C2) — claimed as CRITICAL
- **Cross-review verdict**: **REJECTED** — the NaN fallback exists and is functional.

**Description**: The py-metal reviewer claimed that the HLLD solver lacks a final NaN guard and can return NaN fluxes to the caller. This is incorrect. Lines 1082-1091 of `metal_riemann.py` clearly implement an HLL fallback:

```python
# Final NaN guard — fall back to HLL for any remaining NaN
has_nan = torch.isnan(F_HLLD)
if has_nan.any():
    F_HLL = self._hll_flux(UL, UR, FL, FR, dim)
    nan_mask = has_nan.any(dim=0)
    F_HLLD[:, nan_mask] = F_HLL[:, nan_mask]
```

This guard detects NaN in any component of the HLLD flux and substitutes the corresponding HLL flux. The py-metal reviewer either missed this code or reviewed an earlier version before the guard was added.

**Impact**: N/A — the finding is factually wrong. The HLLD solver has a robust NaN→HLL fallback chain.

### XR-MTL-R2: HLLD Star-State Density Going Negative Before Clamping

- **File:Line**: `metal_riemann.py:~920`
- **Found by**: py-metal (m4)
- **Cross-review verdict**: **REJECTED** — this is normal HLLD behavior. The star-state density `rho_L* = rho_L * (S_L - v_n,L) / (S_L - S_M)` is algebraically guaranteed non-negative when S_L < v_n,L < S_M (the physical regime where HLLD applies). The clamping is a safety net for numerical edge cases (float32 round-off), not an indication of a formula error. Miyoshi & Kusano (2005) Section 3 proves positivity of star states in the physical domain.

---

## Summary

| Severity | Count | Fixed | IDs |
|----------|-------|-------|-----|
| CRITICAL | 1 | 1 | XR-MTL-C1 ✅ |
| HIGH | 1 | 1 | XR-MTL-H1 ✅ |
| MEDIUM | 5 | 3 | XR-MTL-M2 ✅, M3 ✅, M5 ✅ (M1, M4 open) |
| LOW | 6 | 2 | XR-MTL-L1 ✅, L2 ✅ (L3-L6 open) |
| REJECTED | 2 | — | XR-MTL-R1, R2 |

**Overall Metal domain assessment**: 7.0/10 → improved. All CRITICAL and HIGH issues resolved. The unit-system mismatch (C1) is fixed — transport and solver both use HL code units. The HLLD sign issue (H1) uses sign-preserving clamping per Miyoshi & Kusano. Sub-cycling (M3), viscous heating (M2), tau_e units (L1), P_FLOOR (L2), and mu_0 import (M5) also fixed. Remaining open items are design tradeoffs (simplified CT, primitive-variable RK) and code quality (dead code, duplicated utilities).
