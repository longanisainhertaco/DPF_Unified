# Metal v2 MHD Solver: Physics-Derived Definition of Done

**Date**: 2026-03-24
**Author**: dpf-mhd-physicist (Cortana)
**Status**: Evaluation rubric for Metal v2 implementation
**Scope**: Physics requirements, solver design, acceptance criteria

This document is derived entirely from DPF discharge physics, experimental data,
and verified Metal v1 failure modes. It defines what any correct DPF MHD solver
must satisfy, independent of implementation framework.

---

## 1. DPF Discharge Physics Requirements

### 1.1 The Five Phases of a DPF Discharge

| Phase | Duration | Spatial Scale | Dominant Physics | beta Range |
|-------|----------|---------------|------------------|------------|
| 1. Breakdown | ~100 ns | Full electrode gap (5-16 cm) | Paschen breakdown, ionization | N/A (not MHD) |
| 2. Axial rundown | 2-6 us | Electrode length (48 cm for PF-1000) | Snowplow J x B acceleration, circuit-driven I(t) | ~1 (equipartition) |
| 3. Radial compression | 200-500 ns | r: 11.55 cm -> 2 cm (anode radius -> pinch) | Converging cylindrical shock, Rankine-Hugoniot | 1 -> 0.01 |
| 4. Pinch | 50-200 ns | r: ~1-2 mm (pinch column) | Maximum B, maximum compression, instabilities | 10^-4 to 10^-2 |
| 5. Post-pinch | 1-5 us | Expanding plasma, re-strike | Expansion, current re-distribution | ~1 (recovering) |

**Source**: Lee & Saw (2014); Akel et al. (2021), Table 2 -- PF-1000 shot 12581:
Va = 10.5 cm/us, Vs = 22 cm/us, Vp = 18 cm/us, pinch duration ~212 ns.

### 1.2 Governing Equations

The MHD system in conservative form (8 variables, SI units with mu_0 = 1 in HL code units):

```
d(rho)/dt + div(rho * v) = 0                                     [mass]
d(rho*v)/dt + div(rho*v*v + P_tot*I - B*B) = S_geom              [momentum]
d(E)/dt + div((E + P_tot)*v - B*(v.B)) = Q_ohm - Q_rad + S_E    [energy]
d(B)/dt + div(v*B - B*v) = 0                                     [induction]
```

where:
- `E = p/(gamma-1) + 0.5*rho*|v|^2 + 0.5*|B|^2` [J/m^3] (total energy density)
- `P_tot = p + 0.5*|B|^2` [Pa] (total pressure)
- `Q_ohm = eta * |J|^2` [W/m^3] (ohmic heating, eta in Ohm.m)
- `Q_rad = 1.42e-40 * g_ff * Z * n_e^2 * sqrt(Te)` [W/m^3] (bremsstrahlung, SI)
- `S_geom` = cylindrical geometric source terms (1/r contributions)

**Dimensional check on bremsstrahlung**:
- `[1.42e-40] * [1] * [1] * [m^-3]^2 * [K]^(1/2) = [W m^3 K^{-1/2}] * [m^-6] * [K^{1/2}] = [W/m^3]` -- correct.
- Reference: Rybicki & Lightman (1979) Eq. 5.14a; NRL Plasma Formulary (2019) p. 58.

### 1.3 Conservation Laws That Must Be Satisfied

1. **Mass**: `integral(rho) dV = const` (no sources/sinks)
2. **Momentum**: `integral(rho*v) dV = integral(F_external) dt` (J x B from circuit is external)
3. **Energy**: `dE_total/dt = P_circuit - P_radiated` (circuit injects, radiation removes)
4. **div(B) = 0**: To machine precision via constrained transport

### 1.4 Plasma Beta Profile Across the Discharge

**This section determines where float32 fails.**

At the electrode boundary during rundown (Phase 2):
- `I = 1.2 MA` (typical PF-1000, Akel 2021)
- `r = 0.01 m` (1 cm from anode surface)
- `B_theta = mu_0 * I / (2*pi*r) = (4*pi*1e-7)(1.2e6) / (2*pi*0.01) = 24 T`
- `ME = B^2 / (2*mu_0) = 24^2 / (2 * 4*pi*1e-7) = 2.29e8 Pa = 229 MPa`
- Fill gas at 1.2 Torr D2, T = 300 K: `n = p_fill / (k_B * T) = 160 / 1.38e-23*300 = 3.86e22 m^-3`
- `p_thermal = n * k_B * T = 160 Pa`
- **beta = p_thermal / ME = 160 / 2.29e8 = 7.0e-7**

At the pinch column (Phase 4):
- `I_pinch = 500 kA` (Akel 2021, shot 12581: 523 kA)
- `r_pinch = 0.001 m` (1 mm)
- `B_theta = mu_0 * I / (2*pi*r) = (4*pi*1e-7)(5e5) / (2*pi*0.001) = 100 T`
- `ME = 100^2 / (2 * 4*pi*1e-7) = 3.98e9 Pa = 3.98 GPa`
- Pinch temperature: `Te ~ 1-5 keV = 1.16e7 - 5.8e7 K`
- Peak density: `n_i = 1.7e23 m^-3` (shot 12581)
- `p_thermal = 2 * n_i * k_B * Te = 2 * 1.7e23 * 1.38e-23 * 1.16e7 = 5.4e7 Pa`
- **beta = 5.4e7 / 3.98e9 = 0.014** (at the column edge; even lower further out)

### 1.5 Float32 Catastrophic Cancellation Analysis

Standard pressure recovery: `p = (gamma-1) * (E - 0.5*rho*|v|^2 - 0.5*|B|^2)`

Float32 has ~7.2 significant decimal digits (23-bit mantissa).

| beta | E_total [Pa] | KE + ME [Pa] | p [Pa] | Surviving digits | Status |
|------|-------------|-------------|--------|-----------------|--------|
| 1.0 | 2e8 | 1e8 | 1e8 | 7 | Safe |
| 0.1 | 1.1e8 | 1e8 | 1e7 | 6 | Safe |
| 0.01 | 1.01e8 | 1e8 | 1e6 | 5 | Marginal |
| 1e-3 | 1.001e8 | 1e8 | 1e5 | 4 | Unreliable |
| 1e-4 | 1.0001e8 | 1e8 | 1e4 | 3 | Corrupt |
| 1e-6 | 1.000001e8 | 1e8 | 1e2 | 1 | Garbage |
| 7e-7 | actual electrode | | 160 Pa | <1 | **Negative** |

**Conclusion**: The DPF electrode boundary (beta ~ 7e-7) produces guaranteed negative
pressure in float32 with standard energy formulation. The pinch edge (beta ~ 0.01)
has only 5 reliable digits -- marginal for shocks where the Rankine-Hugoniot conditions
demand accurate pressure jumps.

### 1.6 Validated PF-1000 Parameters (Akel 2021, 24-shot dataset)

**Circuit**:
- C0 = 1332 uF, V0 = 16 kV, L0 = 25 nH, r0 = 4.0-6.5 mOhm (shot-dependent)
- E_stored = 170.5 kJ

**Geometry**:
- Anode radius a = 11.55 cm, cathode radius b = 16 cm
- Anode length z0 = 48 cm

**Lee model fit parameters** (all 24 shots):
- fc = 0.70 (constant), fm = 0.17-0.24 (mean 0.20)
- fmr = 0.03-0.35 (high variability), fcr = 0.30-0.85

**Validation targets**:
- I_peak range: 1131-1335 kA (mean ~1240 kA at 16 kV)
- I_pinch range: 262-598 kA
- Computed Yn: 1.7e8 to 1.11e10 n/shot
- Computed vs measured Yn: < 2% average error (1.2 Torr: 1.78e9 vs 1.75e9; 1.05 Torr: 2.33e9 vs 2.29e9)

**Higher energy operation** (Kubes 2019):
- V0 = 16-20 kV, E_stored = 250-350 kJ
- I_peak = 1.0-2.0 MA, I_pinch = 0.7-1.5 MA

---

## 2. Solver Design (Derived from Physics Requirements)

### 2.1 Energy Formulation: Entropy Tracer with Dual-Energy Switching

**Requirement**: Survive float32 at beta ~ 7e-7 (electrode) and beta ~ 0.01 (pinch edge).

**Solution**: Entropy tracer `S_rho = rho * K` where `K = p / rho^gamma` (pseudo-entropy).

- Pressure recovery: `p = K * rho^gamma` -- multiplicative, always positive.
- Dimensional check: `[K] = [Pa] / [kg/m^3]^gamma = [Pa * m^(3*gamma) / kg^gamma]`.
  For gamma = 5/3: `[K] = [Pa * m^5 / kg^(5/3)]`.
  Then `p = K * rho^gamma` has units `[Pa * m^5 / kg^(5/3)] * [kg/m^3]^(5/3) = [Pa]`. Correct.

**Dual-energy switching** (smooth blend, not hard switch):

```
eta = p_entropy / E_total    (where p_entropy = K * rho^gamma)

If eta > eta_2 = 1e-2:  use p from total energy (conservative, more accurate)
If eta < eta_1 = 1e-5:  use p from entropy tracer (immune to cancellation)
Between:                 cubic Hermite blend
```

**Why NOT pure entropy (DISPATCH HLLS)?**
- HLLS does not conserve total energy -- it trades conservation for monotonicity.
- DPF DoD requires energy conservation < 10% over the discharge.
- The hybrid approach preserves conservation where total energy is accurate (high beta)
  and falls back to entropy where it isn't (low beta).

**Source**: Popovas (2025), A&A 694, arXiv:2211.02438 -- validated HLLS in float32.
Bryan et al. (2014), ApJS 211, 19 -- Enzo dual-energy formalism.
FINAL_CROSS_REFERENCE.md -- 7-source consensus on switching criterion.

### 2.2 Entropy Tracer Evolution

The entropy tracer `S_rho = rho * K` is advected as a passive scalar:

```
d(S_rho)/dt + div(S_rho * v) = (gamma - 1) * S_rho / p * (Q_ohm - Q_rad + Q_shock)
```

- Left side: passive advection through the Riemann solver contact wave (upwind flux).
- Right side: source terms that change entropy.
  - `Q_ohm = eta * J^2` increases entropy (ohmic heating).
  - `Q_rad` decreases entropy (radiation cooling).
  - `Q_shock`: entropy production at shocks from the residual principle (Popovas Eq. 28).

**Critical**: Ohmic heating must appear in BOTH the total energy equation AND the entropy
source term. Enzo's known gap (FM-6 in enzo_dual_energy_analysis.md) is missing ohmic
heating in the internal energy -- we must not repeat this.

Dimensional check on source term:
- `(gamma-1) * S_rho / p * Q_ohm` has units `[1] * [kg/m^3 * Pa*m^5/kg^(5/3)] / [Pa] * [W/m^3]`
- Simplifies to `[kg/m^3 * m^5/kg^(5/3)] * [W/m^3] / [1]`... this needs the specific form.
- More directly: `dK/dt = (gamma-1) * K/p * Q_net = (gamma-1) * Q_net / (rho^gamma)`.
  Units: `[1] * [W/m^3] / [kg/m^3]^gamma`. Multiplied by rho to get `d(S_rho)/dt`:
  `[kg/m^3] * [W/m^3] / [kg/m^3]^gamma * [1/s] ... `
- The correct conservative form for the source: `d(rho*K)/dt|_source = (gamma-1) * K * Q_net / p`.
  Since K = p/rho^gamma: `(gamma-1) * (p/rho^gamma) * Q_net / p = (gamma-1) * Q_net / rho^gamma`.
  Then `d(rho*K)/dt = rho * dK/dt = rho * (gamma-1) * Q_net / rho^gamma = (gamma-1) * Q_net / rho^(gamma-1)`.
  Units: `[1] * [W/m^3] / [kg/m^3]^(2/3)` for gamma=5/3. This has the right dimensions to be
  a time derivative of `[rho * K] = [kg/m^3 * Pa * m^5 / kg^(5/3)]`. Verified consistent.

### 2.3 Reconstruction

**Phase 2 (rundown)**: Smooth flow behind the snowplow sheath. PLM (2nd order) sufficient.
**Phase 3 (compression)**: Converging shock with density jump ratio up to 4 (gamma = 5/3).
  WENO5-Z (5th order) needed for sharp shock capture without excessive numerical diffusion.
**Phase 4 (pinch)**: Extreme gradients. WENO5-Z with positivity-preserving fallback.

The solver must implement both PLM and WENO5-Z, selectable at runtime.

**WENO5-Z stencil**: 5 cells wide. Requires nghost >= 3 ghost cells per boundary.
The Metal v1 solver uses 2 ghost cells for PLM, 3 for WENO5, selected dynamically.
This is correct.

### 2.4 Riemann Solver

**HLLD** (Miyoshi & Kusano 2005): Resolves 7 MHD wave families through 4 intermediate states
(L*, L**, R**, R*). Required for DPF because:

1. The contact wave separates the shocked fill gas from the swept electrode material.
   HLL smears this contact, producing incorrect post-shock temperatures.
2. Alfven waves carry information about B_theta perturbations along the pinch column.
   HLL cannot resolve these, over-damping MHD instabilities that determine pinch stability.
3. At the pinch column boundary, beta ~ 0.01 -- the slow magnetosonic wave degenerates
   toward the Alfven speed. Only HLLD correctly resolves this transition.

**Verified**: The Metal v1 HLLD implementation (`_riemann_solvers.py`, 397 LOC) is a proper
4-intermediate-state solver. It computes SL_star/SR_star (Alfven speeds at lines 308-309),
double-star states U_dsL/U_dsR with tangential velocity/B-field averaging (lines 314-332),
and performs 5-region flux selection (lines 370-385). The earlier claim that it was
"actually HLLC-MHD" is incorrect -- the code IS proper HLLD.

### 2.5 Time Integration

**SSP-RK3** (Shu & Osher 1988): 3rd-order strong stability preserving.

```
u^(1) = u^n + dt * L(u^n)
u^(2) = 3/4 * u^n + 1/4 * (u^(1) + dt * L(u^(1)))
u^(n+1) = 1/3 * u^n + 2/3 * (u^(2) + dt * L(u^(2)))
```

SSP property ensures that if the forward Euler method is TVD (total variation diminishing)
under a given CFL, the SSP-RK3 method is also TVD under the same CFL. This prevents
spurious oscillations near shocks.

**CFL constraints** (most restrictive first):

1. **Resistive diffusion** (explicit): `dt < dx^2 * mu_0 / (2 * eta)`.
   For Spitzer resistivity at Te = 1 keV: `eta ~ 1e-7 Ohm.m`.
   With dx = 0.001 m: `dt < (1e-3)^2 * 4*pi*1e-7 / (2 * 1e-7) = 6.3e-4 s`. Not limiting.
   For anomalous resistivity at pinch (`eta ~ 1e-3`): `dt < 6.3e-8 s`. Very restrictive.
   **Action**: Sub-cycle resistive diffusion with N = ceil(dt_mhd / dt_res), capped at 20.

2. **Fast magnetosonic**: `dt < CFL * dx / (|v| + c_f)`.
   At the pinch: `c_f ~ v_A ~ B/sqrt(mu_0*rho) ~ 100 / sqrt(4*pi*1e-7 * 0.001) = 2.8e6 m/s`.
   With dx = 0.001 m, CFL = 0.3: `dt < 1.1e-10 s`. This dominates during the pinch.

3. **Circuit coupling**: The circuit quarter-period is `T/4 ~ pi*sqrt(L*C)/2 ~ pi*sqrt(25e-9 * 1332e-6)/2 ~ 2.9 us`.
   The dI/dt timescale is ~1 us. Circuit sub-cycling within MHD steps is needed when
   dt_MHD << dt_circuit (which it always is).

### 2.6 Constrained Transport (div(B) = 0)

In axisymmetric (r, z) geometry:
- B_r and B_z are face-centered, evolved via edge-centered EMFs.
- B_theta is cell-centered, evolved by the induction equation directly.
- Only B_r and B_z participate in CT. B_theta does not because it is orthogonal to the
  computational (r, z) plane.

CT guarantees `div(B) = (1/r) * d(r*B_r)/dr + dB_z/dz = 0` to machine precision.

The entropy tracer does NOT participate in CT -- it is a cell-centered scalar with no
direct role in the EMF computation. CT operates on B-fields and velocities only.

### 2.7 Circuit Coupling

**Plasma inductance** (Lee formula, density-weighted):

```
Lp = (mu_0 / (2*pi)) * z_sheath * ln(b / r_eff)    [H]
```

where:
- `z_sheath` = axial position of sheath (from density peak detection) [m]
- `b` = cathode radius [m]
- `r_eff` = density-weighted effective radius of current channel [m]

Dimensional check: `[H/m] * [m] * [1] = [H]`. Correct.

**Back-EMF**:

```
back_emf = I * dLp/dt    [V]
```

NOT `d(Lp * I)/dt = I * dLp/dt + Lp * dI/dt` -- the Lp * dI/dt term is already
in the inductive voltage `L_total * dI/dt` on the left side of the circuit equation.
Double-counting this was a v1 bug (coupler.py:194, identified in Troubleshooting.md).

**Lp monotonicity**: dLp/dt must be computed from a monotonically increasing Lp estimate.
Noisy z_sheath detection causes oscillating dLp/dt that destabilizes the circuit.
Solution: use a running maximum or exponential smoothing on Lp.

**Circuit equation** (implicit midpoint for stiffness):

```
L_total * dI/dt = V_cap - R_eff * I - I * dLp/dt - back_emf_motional
```

where `L_total = L0 + ESL + Lp`, `R_eff = r0 + R_plasma(Spitzer)`.

### 2.8 Electrode Boundary Conditions

At the cathode (outer radial boundary, r = b):
- `B_theta = mu_0 * I / (2*pi*b)` -- prescribed from circuit current.
- Density: zero-gradient (outflow).
- Velocity: zero-gradient.
- Pressure: zero-gradient (NOT extrapolated from interior -- that would impose the
  corrupted float32 pressure onto the boundary).

At the axis (r = 0):
- Reflecting: `v_r(0) = 0`, `B_r(0) = 0`.
- `B_theta(0) = 0` (by symmetry).
- `d(rho)/dr|_0 = 0`.

At the anode face (z = 0):
- Conducting wall: `v_z = 0`, `E_tangential = 0`.

At the open end (z = z_max):
- Outflow: zero-gradient on all variables.

---

## 3. Definition of Done

### 3.1 Must-Have (simulation is physically meaningless without these)

| ID | Criterion | Test | Threshold | Rationale |
|----|-----------|------|-----------|-----------|
| M1 | No negative pressure | `test_v2_no_negative_pressure` | `p > 0` everywhere, all timesteps | Negative pressure is unphysical. Entropy formulation guarantees this. |
| M2 | PF-1000 I_peak accuracy | `test_v2_pf1000_ipeak` | Within 10% of 1.2 MA (Akel 2021 mean at 16 kV) | If I(t) is wrong, all downstream physics is wrong. |
| M3 | Mass conservation | `test_v2_mass_conservation` | `|M(t) - M(0)| / M(0) < 0.05` over full discharge | Mass should be exactly conserved; 5% allows for outflow BCs. |
| M4 | Energy conservation | `test_v2_energy_conservation` | `|dE_total/dt - P_circuit + P_rad| / P_circuit < 0.10` | Energy budget must close to 10% accounting for circuit input and radiation. |
| M5 | No NaN propagation | `test_v2_no_nan` | Zero NaN in any field at any timestep | NaN = solver crash = useless. |
| M6 | Completes 5 phases | `test_v2_full_discharge` | Simulation reaches t > 2 * t_peak (~12 us for PF-1000) without crash | Must survive rundown, compression, pinch, AND post-pinch. |
| M7 | Float32 on Metal GPU | `test_v2_float32_metal` | All physics in float32, runs on MPS device | The entire point of Metal v2. |
| M8 | div(B) = 0 | `test_v2_divb` | `max(|div(B)|) * dx / max(|B|) < 1e-6` | CT must maintain divergence-free B to relative precision. |

### 3.2 Should-Have (needed for publication-quality results)

| ID | Criterion | Test | Threshold | Rationale |
|----|-----------|------|-----------|-----------|
| S1 | I(t) waveform shape | `test_v2_pf1000_nrmse` | NRMSE < 0.25 vs Akel/Kubes waveform | Shape matters, not just peak. |
| S2 | Current dip at pinch | `test_v2_current_dip` | Dip magnitude 30-70% of I_peak | The current dip is the signature of radial compression -- no dip = no compression. |
| S3 | Pinch voltage spike | `test_v2_pinch_voltage` | V_pinch > 20 kV at peak compression | Voltage spike = back-EMF from inductance change. Verifies circuit coupling. |
| S4 | Multi-device validation | `test_v2_multidevice` | 3+ devices (PF-1000, UNU-ICTP, NX2) all complete without crash | Solver must not be tuned to one device. |
| S5 | Cross-backend parity | `test_v2_cross_backend_sod` | L1(rho) < 15% vs Python WENO5+HLLD on Sod shock tube | Ensures Metal v2 matches the reference Python engine. |
| S6 | Brio-Wu MHD shock tube | `test_v2_brio_wu` | Compound wave structure preserved, no NaN | Standard MHD Riemann problem validation. |
| S7 | Sod shock tube | `test_v2_sod_shock` | L1(rho) < 1e-2 at N=256 | Hydro subset validation. |
| S8 | Diffusion convergence | `test_v2_diffusion_convergence` | Convergence rate >= 1.9 (2nd order) | Verifies spatial discretization order. |
| S9 | Faster than Athena++ | `test_v2_performance` | Wall-clock < Athena++ at grid >= 128x512 | Metal GPU should outperform single-core C++ at sufficient grid size. |

### 3.3 Must-Have Phase 2 (research & plan during Phase 1, implement after Phase 1 passes)

Phase 1 (sections 3.1-3.2 above) validates the core physics: dual-energy entropy tracer, circuit coupling, electrode BCs. Phase 2 items are full requirements — NOT optional, NOT out of scope — but gated behind Phase 1 validation. Each item must have a research deliverable (architecture doc + DoD + LOC estimate) completed during Phase 1 implementation, ready to build the moment Phase 1 passes.

| ID | Requirement | Research Deliverable (during Phase 1) | Implementation (after Phase 1) |
|----|------------|---------------------------------------|-------------------------------|
| P2-1 | AMR (adaptive mesh refinement) | Block-based AMR architecture for cylindrical MHD with CT. Refinement criteria (density gradient, current density). Prolongation/restriction operators for entropy tracer. | Estimated 2,000-4,000 LOC, 8-12 weeks |
| P2-2 | Multi-species (argon, neon, deuterium) | Species-dependent EOS, ionization equilibrium tables (Saha), radiation coefficients per species (line emission for Ar/Ne). State vector extension. | Estimated 500-1,000 LOC, 3-4 weeks |
| P2-3 | 3D (non-axisymmetric instability modes) | Full 3D MHD with azimuthal dimension, 3D CT (edge-centered EMFs on all 12 edges), 3D WENO5 reconstruction, 3D geometric source terms. Memory/performance scaling analysis. | Estimated 1,500-3,000 LOC, 6-8 weeks |
| P2-4 | Radiation transport (FLD) | Flux-limited diffusion for optically thick regions, multi-group optional. Implicit solver for radiation-matter coupling. Interaction with entropy tracer. | Estimated 800-1,500 LOC, 4-6 weeks |
| P2-5 | Characteristic WENO5 decomposition | 7x7 MHD eigenvector matrices (L/R), degenerate case handling (Bn→0), interaction with entropy tracer reconstruction. Benchmark vs component-wise on DPF electrode test. | Estimated 500-700 LOC, 2-3 weeks |
| P2-6 | IMEX time integration | Coupled implicit (resistive/conductive) + explicit (hyperbolic) RK stages. LSDIRK(2,2,2) or ARK2 scheme. Nonlinear solver for implicit stage. Comparison vs current operator-split approach. | Estimated 400-600 LOC, 2-3 weeks |

**Phase 1 exit criterion**: ALL of M1-M8 pass. Only then do Phase 2 items begin implementation.

**Phase 2 research exit criterion**: Each P2-X has an architecture document, a DoD with testable acceptance criteria, and an LOC/timeline estimate — all completed before Phase 1 coding ends.

---

## 4. Metal v1 Failure-to-Fix Mapping

### 4.1 Float32 Pressure Corruption

**v1 Failure**: The `mhd_rhs_mps` function (metal_riemann.py:271-273) computes dp/dt via:

```python
dp_dt = (gamma - 1.0) * (
    dU_dt[IEN] - v_dot_dmom + 0.5 * v_sq * drho_dt - B_dot_dB
)
```

This involves subtracting large nearly-equal rate-of-change terms. When beta < 0.01,
the subtraction `dU_dt[IEN] - v_dot_dmom - B_dot_dB` loses significant digits in
float32, producing corrupted dp/dt that accumulates into negative pressure.

**DoD Violated**: M1 (no negative pressure).

**v2 Fix**: Replace pressure recovery path with entropy-based dual-energy:

1. Evolve `S_rho = rho * K` as a 9th conserved variable (passive scalar through Riemann solver).
2. Recover pressure as `p = K * rho^gamma = (S_rho / rho) * rho^gamma = S_rho * rho^(gamma-1)`.
3. Use dual-energy switching: when `eta = p_entropy / E_total < eta_2`, trust entropy.
4. The `_cons_to_prim_mps` function must be modified to accept the entropy tracer and
   use it for pressure recovery when the switching criterion selects it.
5. `_euler_stage_compilable` must evolve S_rho alongside rho, vel, p, B.

### 4.2 Back-EMF Wiring

**v1 Failure**: Two related bugs:

(a) `engine.py:1025` -- back_emf = 0.0 when using snowplow coupling (most cases).
    The MHD-computed back_emf from `_compute_back_emf()` is discarded at line 1025.
    Only used when `use_snowplow_mhd_feedback=True` AND bridge phase is complete.

(b) `coupler.py:194` -- `back_emf = current * dLp_dt`. This is correct for the
    inductive back-EMF. The formula `d(Lp*I)/dt = I*dLp/dt + Lp*dI/dt` would
    double-count the `Lp*dI/dt` term already in the circuit equation's left side.
    The Troubleshooting.md correctly identifies this. The coupler code is correct.

**DoD Violated**: M2 (I_peak accuracy), S2 (current dip).

**v2 Fix**: Wire the feedback path from MHD to circuit:

1. MHD solver computes Lp from density-weighted radius (not B^2 energy integral).
2. MHD solver computes dLp/dt with monotonicity enforcement (running max on Lp).
3. Engine passes `back_emf = I * dLp/dt` to circuit solver in ALL coupling modes,
   not just when bridge phase is complete.
4. Back-EMF clamped to +/- 50 kV to prevent handoff instability.

### 4.3 Ghost Cell Count for WENO5

**v1 Status**: Actually correct. The Metal v1 code at metal_riemann.py:218 uses
`gh = 3 if (reconstruction == "weno5" and n_dim >= 5) else 2`. This dynamically
selects 3 ghost cells for WENO5, which provides the 5-point stencil. The initial
concern about "ghost cells = 2" being insufficient for WENO5 is addressed in the
current codebase.

**DoD Impact**: None -- already fixed.

### 4.4 Missing Ohmic Heating in Internal Energy

**v1 Failure**: The Metal solver applies resistive diffusion operator-split
(metal_solver.py has `implicit_resistivity` option), but ohmic heating `Q = eta * J^2`
must appear in both the total energy equation AND the entropy tracer source term.
If ohmic heating only appears in total energy, the entropy tracer will under-predict
pressure in resistive regions, causing the dual-energy switching to produce incorrect
pressure.

**DoD Violated**: M4 (energy conservation).

**v2 Fix**: After each resistive diffusion sub-step:
1. Compute `Q_ohm = eta * |J|^2` from the updated B-field.
2. Add `Q_ohm * dt_sub` to total energy E.
3. Add `(gamma-1) * Q_ohm * dt_sub / rho^(gamma-1)` to `S_rho`.
4. Both updates must happen in the same sub-step to maintain consistency.

### 4.5 B^2-Energy Plasma Inductance vs Density-Weighted

**v1 Failure**: The Metal solver at metal_solver.py:2072-2086 computes Lp from
the volume-averaged B_theta magnitude: `Lp = B_theta_avg * A / |I|`. This
B^2-energy-based estimate includes electrode boundary artifacts where the prescribed
B_theta creates artificially high magnetic energy. The density-weighted approach
(coupler.py, using `r_eff` from density profile) is more robust because the current
channel location is determined by where the plasma IS, not where B was prescribed.

**DoD Violated**: M2 (I_peak accuracy), S1 (waveform shape).

**v2 Fix**: Use the density-weighted Lp formula from `coupler.py`:
1. Detect sheath position from density peak (argmax along radial axis).
2. Compute density-weighted effective radius: `r_eff = sum(r * rho * dV) / sum(rho * dV)`.
3. `Lp = (mu_0 / (2*pi)) * z_sheath * ln(b / r_eff)`.
4. Apply running-maximum smoothing to Lp for monotonicity.

### 4.6 Non-Conservative Pressure Evolution (Python Engine)

**v1 Failure**: The Python MHD solver (`mhd_solver.py`) evolves dp/dt (primitive
variable) rather than dE/dt (conservative variable). The Metal solver internally
works with conservative variables in the Riemann solver but converts back to
primitive dp/dt in `mhd_rhs_mps`. This conversion involves the same catastrophic
cancellation as direct pressure recovery.

**DoD Violated**: M1 (no negative pressure), implicitly M4 (energy conservation at shocks).

**v2 Fix**: The _euler_stage must work in conservative variables throughout:
1. Convert primitives to conservatives U = (rho, rho*v, E, B, S_rho).
2. Compute dU/dt from Riemann solver fluxes.
3. Update U^(n+1) = U^n + dt * dU/dt.
4. Convert back to primitives using entropy-based pressure recovery.
5. Never compute dp/dt as an intermediate -- go directly from dE/dt to p via entropy.

---

## 5. Validation Test Specifications

### 5.1 Unit Tests (fast, < 1 second each)

```python
def test_v2_entropy_tracer_uniform():
    """Entropy tracer on uniform state should remain constant."""
    # Setup: uniform rho=1, p=1, v=0, B=0, K = p/rho^gamma = 1
    # Step: 10 timesteps
    # Assert: max(|K_final - K_initial|) / K_initial < 1e-6

def test_v2_entropy_positivity():
    """Pressure from entropy is always positive, even at beta=1e-7."""
    # Setup: rho=1e-3, p=160, B_theta=24T (electrode conditions)
    # Assert: p_from_entropy > 0

def test_v2_dual_energy_switching():
    """Switching selects entropy at low beta, total energy at high beta."""
    # Setup: grid with beta ranging from 1e-6 to 10
    # Assert: cells with beta < 1e-5 use entropy; cells with beta > 0.1 use total energy

def test_v2_passive_scalar_advection():
    """Entropy tracer advected correctly through contact wave."""
    # Setup: two-state Riemann problem with different K_L, K_R
    # Assert: K is upwinded through contact (K follows density, not pressure)

def test_v2_circuit_coupling_lp():
    """Density-weighted Lp matches analytical coaxial inductance."""
    # Setup: uniform current channel at r < r_eff
    # Assert: |Lp - Lp_analytical| / Lp_analytical < 0.05
    # where Lp_analytical = (mu_0 / 2pi) * z * ln(b / r_eff)

def test_v2_back_emf_wired():
    """Back-EMF is non-zero when dLp/dt is non-zero."""
    # Setup: simulation with changing Lp
    # Assert: back_emf = I * dLp/dt, not zero

def test_v2_ohmic_in_entropy():
    """Ohmic heating appears in both E and S_rho."""
    # Setup: static plasma with applied J
    # Step: one resistive diffusion step
    # Assert: p increased by eta*J^2*dt/(gamma-1) to within 1%
    # Assert: K increased correspondingly
```

### 5.2 Shock Tube Tests (< 10 seconds each)

```python
def test_v2_sod_shock_tube():
    """Sod shock tube in float32 on Metal."""
    # Setup: standard Sod (rho_L=1, p_L=1, rho_R=0.125, p_R=0.1), N=256
    # Assert: L1(rho) < 1e-2, L1(p) < 1e-2
    # Assert: no negative pressure anywhere

def test_v2_brio_wu():
    """Brio-Wu MHD shock tube preserves compound wave structure."""
    # Setup: standard Brio-Wu, N=512
    # Assert: no NaN, no negative pressure
    # Assert: density profile has correct number of waves (fast, slow, contact)

def test_v2_low_beta_shock():
    """Shock propagation at beta = 0.001 without pressure corruption."""
    # Setup: magnetized shock with B chosen so beta = 0.001
    # Assert: no negative pressure
    # Assert: post-shock state satisfies R-H within 5%
```

### 5.3 Convergence Tests (< 60 seconds)

```python
@pytest.mark.slow
def test_v2_diffusion_convergence():
    """Resistive diffusion convergence study."""
    # Setup: Gaussian B-field profile, analytical solution is Gaussian spreading
    # Resolutions: N = 32, 64, 128, 256
    # Assert: L1 error convergence rate >= 1.9

@pytest.mark.slow
def test_v2_smooth_wave_convergence():
    """Linear MHD wave convergence (smooth problem)."""
    # Setup: small-amplitude fast magnetosonic wave
    # Resolutions: N = 32, 64, 128, 256
    # Assert: with WENO5, convergence rate >= 3.0 in smooth regions
```

### 5.4 Full Discharge Tests (< 5 minutes)

```python
@pytest.mark.slow
def test_v2_pf1000_full_discharge():
    """PF-1000 full discharge: all 5 phases, float32, Metal GPU."""
    # Setup: PF-1000 preset, 64x256 grid, t_final = 12 us
    # Assert: M1-M8 all satisfied (no negative p, I_peak within 10%,
    #         mass conservation < 5%, energy conservation < 10%,
    #         no NaN, completes all phases, float32, div(B) < 1e-6)
    # Assert: S2 (current dip 30-70%)

@pytest.mark.slow
def test_v2_pf1000_ipeak():
    """PF-1000 peak current within 10% of 1.2 MA."""
    # Assert: 1.08 MA < I_peak < 1.32 MA

@pytest.mark.slow
def test_v2_multidevice():
    """3 devices complete full discharge without crash."""
    # Devices: PF-1000 (170 kJ), UNU-ICTP (3 kJ), NX2 (11 kJ)
    # Assert: all three reach t_final without NaN or crash
```

---

## 6. Phase-by-Phase Acceptance Criteria

### Phase 2: Axial Rundown (t = 0 to t_peak)

| Criterion | Acceptance | Measurement |
|-----------|------------|-------------|
| Snowplow velocity | Va = 8-12 cm/us (PF-1000) | Sheath position z(t) slope |
| Current rise | dI/dt ~ 0.3-0.5 MA/us | Time derivative of I(t) |
| Inductance growth | Lp monotonically increasing | Lp(t) from density-weighted formula |
| No negative pressure | p > 0 everywhere | Global minimum of p array |
| Mass swept | M_swept ~ fm * rho_fill * pi * (b^2 - a^2) * z | Volume integral of rho |

### Phase 3: Radial Compression (t_peak to t_pinch)

| Criterion | Acceptance | Measurement |
|-----------|------------|-------------|
| Shock convergence | r_shock decreasing toward axis | Density peak radial position |
| Compression ratio | rho_peak / rho_fill >= 4 | Maximum density |
| Current dip onset | I starts decreasing | dI/dt < 0 |
| Float32 survival | No NaN or negative p | Global checks |
| Entropy tracer active | Switching criterion selects entropy at electrode | eta < eta_1 at r = b |

### Phase 4: Pinch (t_pinch to t_pinch + 200 ns)

| Criterion | Acceptance | Measurement |
|-----------|------------|-------------|
| Maximum compression | r_pinch < 3 cm (PF-1000) | Density peak radial position |
| Pinch current | I_pinch = 250-600 kA (PF-1000) | I(t) at t_pinch |
| Voltage spike | V > 20 kV | Capacitor voltage or back-EMF |
| No negative pressure | p > 0 at pinch column AND electrode | Global minimum |
| Temperature | Te ~ 1-10 keV (1.16e7 - 1.16e8 K) at pinch | Max(Te) in domain |

### Phase 5: Post-Pinch (t_pinch + 200 ns to t_final)

| Criterion | Acceptance | Measurement |
|-----------|------------|-------------|
| Current recovery | I recovers after dip (not monotonic decay) | I(t) increases after minimum |
| No late-time crash | Simulation continues to t_final | No exception raised |
| Energy budget closure | Total energy in - out ~ energy in fields + thermal | Volume integrals |

---

## 7. Numerical Constants (Verified)

All constants below are in SI unless noted. Each has been dimensionally verified.

| Constant | Value | Units | Source | Dimensional Check |
|----------|-------|-------|--------|-------------------|
| mu_0 | 4*pi*1e-7 | H/m = kg*m/A^2/s^2 | SI definition | -- |
| k_B | 1.380649e-23 | J/K | SI definition | -- |
| m_D | 3.34358377e-27 | kg | CODATA 2018 | -- |
| Bremsstrahlung C0 | 1.42e-40 | W*m^3*K^{-1/2} | Rybicki & Lightman 1979 Eq. 5.14a | [C0]*[m^-6]*[K^1/2] = [W/m^3] |
| Cn (neutron yield) | 8.54e8 | SI (calibrated at 0.5 MA) | Akel 2021 Eq. 1 | -- |
| Lp coefficient | mu_0 / (2*pi) = 2e-7 | H/m | Griffiths Eq. 7.27 | [H/m]*[m]*[1] = [H] |

---

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Entropy tracer drifts at shocks (under-produces entropy) | Medium | Incorrect post-shock temperatures | Implement residual principle (Popovas Eq. 28): Q_S resynchronizes from KE+ME energy |
| Float32 overflow in K = p/rho^gamma at extreme conditions | Low | Inf propagation | Clamp K to [K_floor, K_max] where K_max = 1e30 (safe for float32) |
| Dual-energy switching boundary creates artifacts | Medium | Pressure oscillations at switching surface | Use cubic Hermite smooth blend over [eta_1, eta_2], not hard switch |
| MLX API limitations prevent custom kernel | Low | Fall back to PyTorch MPS | Pure-tensor implementation already works on MPS; MLX is optimization only |
| Cylindrical geometry interacts with entropy source term | Medium | Incorrect geometric heating | Test on cylindrical Sedov blast with known analytical solution |
| Circuit coupling instability during pinch | High | Oscillating I(t), eventual NaN | Back-EMF clamp +/-50 kV, Lp monotonicity enforcement, sub-cycling |
