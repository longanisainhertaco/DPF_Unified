# DPF Unified — Physics Verification & User Acceptance Test Plan

## Table of Contents

1. [Physics Verification Plan](#1-physics-verification-plan)
2. [UAT-A: PhD-Level Plasma Physicist](#2-uat-a-phd-level-plasma-physicist)
3. [UAT-B: New Student / Intern](#3-uat-b-new-student--intern)

---

## 1. Physics Verification Plan

### 1.1 Verification Philosophy

The DPF simulator implements multi-physics MHD with circuit coupling, radiation transport, and fusion diagnostics across three computational backends (Python, Athena++, AthenaK) plus a Metal GPU backend for Apple Silicon. Verification proceeds in five tiers:

| Tier | Scope | Method | Pass Criteria |
|------|-------|--------|---------------|
| **T1 — Unit** | Individual physics modules | Analytical comparison | < 1% error vs. exact solution |
| **T2 — Integration** | Coupled subsystems | Manufactured solutions, convergence | 2nd-order convergence rate |
| **T3 — System** | Full DPF simulation | Experimental data comparison | Qualitative agreement + correct scaling |
| **T4 — Surrogate** | WALRUS vs. physics | Cross-validation, rollout divergence | < 10% L2 error over 50 steps |
| **T5 — Metal GPU** | Metal backend parity + physics accuracy | Cross-backend L1 norm, convergence order, solver accuracy | < 5% L1 error vs Python; 5th-order WENO5 verified |

### 1.2 Tier 1 — Unit Verification

#### 1.2.1 Resistive Diffusion (EXISTING — Phase C)

| Test | Reference | Grid | Tolerance |
|------|-----------|------|-----------|
| Gaussian diffusion (explicit) | B(x,t) = B0/sqrt(1+4Dt/w^2) * exp(-x^2/(w^2+4Dt)) | 32, 64, 128 | Convergence order > 1.5 |
| Gaussian diffusion (Crank-Nicolson) | Same | Same | Order > 1.8 |
| Gaussian diffusion (RKL2 STS) | Same | Same | Order > 1.5 |

**Status**: PASSING (test_phase_c_verification.py)

#### 1.2.2 Sod Shock Tube (EXISTING — Phase F.3)

| Test | Reference | Grid | Tolerance |
|------|-----------|------|-----------|
| Python backend | Exact Riemann (gamma=1.4) | 256 cells | L1(rho) < 5%, L1(p) < 5% |
| Athena++ subprocess | Same | 256 cells | L1(rho) < 5%, L1(v) < 10% |

**Status**: PASSING

#### 1.2.3 Brio-Wu MHD Shock (EXISTING — Phase F.3)

| Test | Reference | Grid | Tolerance |
|------|-----------|------|-----------|
| 7-wave MHD structure | Bx = 0.75 const, >5 density jumps | 256 cells | Positivity + conservation |

**Status**: PASSING

#### 1.2.4 NEW — Spitzer Resistivity Verification

| Test | Reference | Method |
|------|-----------|--------|
| eta(Te) scaling | eta = 1.03e-4 * Z * ln(Lambda) * sqrt(m_e) / Te^(3/2) | Compute eta at 10, 100, 1000 eV; compare to tabulated NRL values |
| Coulomb log | ln(Lambda) = 23.5 - ln(sqrt(n_e) / Te^(5/4)) | Compare to NRL Formulary Table 2 |

**Pass criteria**: < 5% relative error at all temperatures.

#### 1.2.5 NEW — Bremsstrahlung Power Verification

| Test | Reference | Method |
|------|-----------|--------|
| P_brem(Te, ne) | P = 1.69e-32 * Z^2 * n_e * n_i * sqrt(Te) [W/m^3] | Evaluate at ne=1e24, Te=1keV, compare to NRL Formulary |
| Gaunt factor g_ff | g_ff(Te) tabulated values | Compare to Karzas-Latter tables |

**Pass criteria**: < 10% error with Gaunt factor correction.

#### 1.2.6 NEW — DD Fusion Reactivity

| Test | Reference | Method |
|------|-----------|--------|
| <sigma*v>(Ti) | Bosch-Hale (1992) parameterization | Evaluate at Ti = 1, 5, 10, 50, 100 keV |
| Neutron yield integral | dY/dt = (1/4) * n_D^2 * <sv> * V_pinch | Compare to Lee model estimates |

**Pass criteria**: < 2% error vs. Bosch-Hale tabulated values.

#### 1.2.7 NEW — Circuit RLC Solver

| Test | Reference | Method |
|------|-----------|--------|
| Underdamped RLC (no plasma) | I(t) = V0*sqrt(C/L)*sin(omega*t)*exp(-R*t/(2L)) | V0=27kV, C=1.332mF, L=15nH, R=2mOhm |
| Overdamped RLC | I(t) = V0/(L*(r1-r2))*(exp(-r1*t)-exp(-r2*t)) | R >> 2*sqrt(L/C) |
| Critical damping | I(t) = V0/L * t * exp(-R*t/(2L)) | R = 2*sqrt(L/C) |

**Pass criteria**: < 0.1% error over 5 oscillation periods (analytical RLC is exact).

### 1.3 Tier 2 — Integration Verification

#### 1.3.1 Orszag-Tang Vortex (EXISTING)

| Test | Reference | Grid | Tolerance |
|------|-----------|------|-----------|
| 2D MHD vortex | Energy conservation + positive density | 128x128 | dE/E < 5% over t_final |

**Status**: PASSING

#### 1.3.2 Sedov-Taylor Blast (EXISTING)

| Test | Reference | Grid | Tolerance |
|------|-----------|------|-----------|
| Cylindrical blast | Similarity solution R_shock ~ (E*t^2/rho)^(1/5) | 64x128 | Shock position < 30% error |

**Status**: PASSING

#### 1.3.3 NEW — MHD Convergence Study

| Test | Method | Grids | Expected |
|------|--------|-------|----------|
| Linear MHD wave (Alfven) | Smooth IC, measure L2 error decay | 32, 64, 128, 256 | 2nd order (SSP-RK2 + WENO5) |
| Linear MHD wave (Fast magnetosonic) | Same | Same | 2nd order |

**Pass criteria**: Measured convergence rate > 1.8 (theoretical: 2.0 for SSP-RK2).

#### 1.3.4 NEW — Anisotropic Conduction Ring Test

| Test | Reference | Method |
|------|-----------|--------|
| Circular B-field + temperature ring | Heat should diffuse along B, not across | Initialize T-ring perpendicular to B, measure cross-field leakage |

**Pass criteria**: Cross-field diffusion < 1% of parallel after 100 diffusion times.

#### 1.3.5 NEW — Coupled Circuit-MHD Energy Balance

| Test | Method | Tolerance |
|------|--------|-----------|
| E_cap(0) = E_cap(t) + E_ind(t) + E_plasma(t) + E_rad(t) + E_ohmic(t) | Track all energy channels | Sum within 2% of initial |

**Pass criteria**: Energy partitioning accounts for > 98% of initial capacitor energy.

### 1.4 Tier 3 — System Verification (DPF Device Comparison)

#### 1.4.1 Lee Model Comparison (EXISTING — Phase C)

| Device | Reference | Metrics | Tolerance |
|--------|-----------|---------|-----------|
| PF-1000 | Lee (2014), 1.2 MA peak | I_peak, t_pinch | 20% |
| NX2 | Lee & Serban, 400 kA | I_peak, t_pinch | 20% |

**Status**: PASSING

#### 1.4.2 NEW — PF-1000 Experimental Comparison

| Metric | Experimental (IPPLM) | Simulation Target | Source |
|--------|----------------------|-------------------|--------|
| Peak current | 1.8-2.0 MA | 1.5-2.5 MA | Scholz et al. (2006) |
| Pinch time | 5-7 us | 4-9 us | Sadowski et al. (2004) |
| Neutron yield | 10^10-10^11 | 10^9-10^12 (order-of-magnitude) | Scholz et al. (2006) |
| Pinch radius | 2-5 mm | Qualitative compression | X-ray imaging |

**Pass criteria**: All metrics within factor of 2-3 of experimental range (appropriate for MHD-level modeling without kinetic effects).

#### 1.4.3 NEW — Scaling Law Verification

| Scaling | Expected | Method |
|---------|----------|--------|
| Y_n vs I_peak^4 | Y_n proportional to I^4 (Lee scaling) | Sweep V0 from 10-50 kV, measure Y_n vs I_peak |
| I_peak vs V0 | I_peak ~ V0 * sqrt(C/L) | Sweep V0 at fixed C, L |
| t_pinch vs sqrt(LC) | Quarter-period scaling | Vary L and C independently |

**Pass criteria**: Power-law exponents within 30% of theoretical values.

#### 1.4.4 NEW — Metal GPU Cross-Backend Verification

| Test | Method | Tolerance |
|------|--------|-----------|
| Sod shock (Metal vs Python) | Run identical IC on both backends, compare L1(rho) | < 10% relative difference |
| Long-run energy conservation | 100+ step Metal simulation, track cumulative energy drift | < 2% drift |
| Float32 vs float64 stencil | CT update with same input, compare results | Relative error < 1e-5 |

**Pass criteria**: Metal backend reproduces Python engine physics to within float32 tolerance for standard benchmarks.

#### 1.4.5 Phase O — Metal GPU Physics Accuracy (COMPLETE)

| Test | Method | Tolerance | Status |
|------|--------|-----------|--------|
| HLLD Riemann solver | Miyoshi & Kusano (2005) 8-component solver, contact+Alfven resolution | No NaN, stable for Brio-Wu | PASSING |
| WENO5-Z reconstruction | Borges et al. (2008) with point-value FD formulas | 5th-order interior convergence (5.47-5.79) | PASSING |
| SSP-RK3 time integration | Shu-Osher (1988) 3-stage 3rd-order SSP | Lower error than SSP-RK2 on smooth problems | PASSING |
| Float64 precision mode | CPU float64 fallback (MPS only supports float32) | Eliminates round-off accumulation | PASSING |
| Maximum accuracy config | WENO5-Z + HLLD + SSP-RK3 + float64 + CT + MC limiter | Stable for 50+ steps on Brio-Wu | PASSING |
| Overall convergence order | Richardson extrapolation on smooth MHD wave | Order >= 1.7 (measured 1.86, limited by MHD nonlinearity) | PASSING |

**45 tests total** in `test_phase_o_physics_accuracy.py`. All passing.

#### 1.4.6 NEW — AthenaK Cross-Backend Verification

| Test | Method | Tolerance |
|------|--------|-----------|
| Blast wave (AthenaK vs Python) | Run MHD blast on both backends, compare density evolution | Qualitative agreement: density ratio > 2 in both |
| State dict parity | Compare state dict keys and shapes between backends | Identical keys, compatible shapes |

**Pass criteria**: AthenaK subprocess produces physically correct results matching Python engine qualitatively.

### 1.5 Tier 4 — WALRUS Surrogate Verification

#### 1.5.1 Surrogate vs. Physics (Single Step)

| Test | Method | Tolerance |
|------|--------|-----------|
| State prediction fidelity | Given 4-step history from physics, predict step 5 | L2(rho) < 5%, L2(Te) < 10% |
| Conservation check | E_surrogate vs E_physics | < 15% energy drift per step |
| Positivity | rho > 0, T > 0 in surrogate output | 100% positive |

#### 1.5.2 Rollout Divergence (Multi-Step)

| Test | Method | Tolerance |
|------|--------|-----------|
| 10-step rollout | Auto-regressive, compare to physics | L2(rho) < 15% at step 10 |
| 50-step rollout | Same | L2(rho) < 30% at step 50 |
| Drift monitoring | Track max(Te) trajectory | Qualitative agreement (same peak timing) |

#### 1.5.3 Parameter Sweep Fidelity

| Test | Method | Tolerance |
|------|--------|-----------|
| V0 sweep (10 points) | Compare surrogate vs physics peak metrics | Rank ordering preserved |
| Sensitivity analysis | dY/dV0 sign and magnitude | Same sign, magnitude within 50% |

#### 1.5.4 Ensemble Uncertainty

| Test | Method | Tolerance |
|------|--------|-----------|
| In-distribution confidence | PF-1000 parameters | Confidence > 0.8 |
| Out-of-distribution detection | Parameters far from training | OOD score > 0.5 |
| Calibration | Confidence vs actual error correlation | Positive correlation (r > 0.3) |

#### 1.5.5 Hybrid Engine Fallback

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Divergence triggers fallback | Inject noise into surrogate output | Engine reverts to physics within 5 steps |
| Physics phase accuracy | First N steps match full-physics run | L2 < 1% |
| Surrogate-to-physics transition | State continuity at handoff | No density/pressure jumps > 10% |

---

## 2. UAT-A: PhD-Level Plasma Physicist

### Persona

**Dr. Maria Vasquez** — Postdoc in pulsed-power plasma physics at a national lab. She has:
- Published on z-pinch implosions and DPF scaling laws
- Used FLASH, Gorgon, and Lee model codes
- Strong intuition for MHD behavior, energy balance, and diagnostic interpretation
- Skeptical of new tools until she can reproduce known results
- Comfortable with command line but prefers GUI for parameter exploration

### Test Environment

- DPF Unified GUI running (Electron app, `npm run dev`)
- Python backend started (`dpf serve --port 8765`)
- All three backends available (Python primary; Athena++/AthenaK optional)
- WALRUS checkpoint loaded (if available) or offline mode

---

### UAT-A.1 — Launch & Backend Verification

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| A1.1 | Launch the Electron app | App connects, TopBar shows green dot + "Idle" | |
| A1.2 | Check TopBar backend badges | At least "Python" badge visible; optionally "Athena++" or "AthenaK" | |
| A1.3 | Open the AI Co-Pilot sidebar | Sidebar slides in with Advisory, Sweep, Inverse, Chat panels | |
| A1.4 | Type "status" in chat | Response shows surrogate/ensemble status (loaded or offline) | |
| A1.5 | Type "help" in chat | Lists all 7 supported question types with examples | |

---

### UAT-A.2 — PF-1000 Baseline Simulation

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| A2.1 | Select "PF-1000" preset from dropdown | All parameters populate: V0=27kV, C=1.332mF, L0=15nH, cylindrical 64x1x128 grid | |
| A2.2 | Hover over "CAPACITANCE" label | Tooltip: "Energy storage capacitance of the capacitor bank. E = 1/2 CV^2." | |
| A2.3 | Hover over "CFL NUMBER" label | Tooltip: "Courant stability factor..." | |
| A2.4 | Click ARM | Advisory panel populates with heuristic checks; Button becomes FIRE (pulsing cyan) | |
| A2.5 | Review advisories | Should see "Configuration looks nominal" or relevant physics warnings | |
| A2.6 | Click FIRE | Status changes to "Running"; oscilloscope shows live I(t) and max_Te(t) traces | |
| A2.7 | Observe current waveform shape | Damped sinusoidal oscillation; peak current 1-2 MA; quarter-period consistent with sqrt(LC) | |
| A2.8 | Wait for completion | Status changes to "Finished"; PostShotPanel replaces oscilloscope | |
| A2.9 | Check PINCH TIME | Value in microseconds range (4-9 us for PF-1000), not "0.00" or "Not reached" | |
| A2.10 | Check PEAK CURRENT | Value should be 1-2 MA (PF-1000 range) | |
| A2.11 | Check NEUTRON YIELD | Non-zero value; order-of-magnitude 10^8-10^11 | |
| A2.12 | Check ENERGY PARTITION | Bank energy = 1/2 * C * V0^2 = 485 J ... verify sum makes sense | |

---

### UAT-A.3 — Physics Intuition Checks

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| A3.1 | Reset, increase V0 from 27kV to 40kV, ARM+FIRE | Peak current increases (I ~ V0*sqrt(C/L)); neutron yield increases | |
| A3.2 | Reset, decrease fill pressure by 10x | Faster pinch time, possibly higher Te but lower neutron yield (less fuel) | |
| A3.3 | Reset, enable Nernst effect toggle | Simulation runs without crash; max_Te may differ from baseline | |
| A3.4 | Reset, enable Full Braginskii viscosity | Simulation runs; energy partition may show increased kinetic dissipation | |
| A3.5 | Reset, disable Bremsstrahlung | Radiated energy should drop to near zero; Te should be higher | |
| A3.6 | Reset, set CFL = 0.9 (high) | ARM should generate advisory warning about high CFL | |
| A3.7 | Reset, set grid to 8x1x8 (very coarse) | ARM should generate "very coarse grid" advisory | |

---

### UAT-A.4 — Cross-Backend Comparison

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| A4.1 | Select Python backend, run PF-1000 | Record I_peak, t_pinch, Y_n | |
| A4.2 | Select Athena++ backend (if available), same config | Run completes; I_peak within 2x of Python result | |
| A4.3 | Select AthenaK backend (if available), same config | Run completes; I_peak within 2x of Python result | |
| A4.4 | Compare all three runs | Qualitative agreement on current waveform shape and pinch timing | |

---

### UAT-A.5 — AI Surrogate Interaction

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| A5.1 | Open Sweep panel, select V0, range 10kV-50kV, 10 points | Sweep runs (if backend available) or shows guidance message | |
| A5.2 | If sweep completes, check scaling curve | V0 vs max_Te relationship should be monotonically increasing | |
| A5.3 | Change sweep metric to "neutron_rate" | Curve updates; should show strong I^4-like dependence on V0 | |
| A5.4 | Open Inverse Design, target max_Te = 5e8 K | Optimization runs or shows guidance for loading checkpoint | |
| A5.5 | Type "what maximizes neutron yield?" in chat | Response explains inverse design approach with actionable steps | |
| A5.6 | Type "explain bennett equilibrium" in chat | Glossary response: "Bennett equilibrium balances magnetic pinch pressure..." | |
| A5.7 | Type "explain Hall MHD" in chat | Should match glossary or return "not in glossary" with suggestions | |

---

### UAT-A.6 — Energy Conservation Audit

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| A6.1 | Run PF-1000 baseline | Note final energy_conservation value in PostShot | |
| A6.2 | Verify 0.90 < energy_conservation < 1.10 | Within 10% (some dissipation expected from radiation/resistivity) | |
| A6.3 | Disable all dissipation (radiation off, resistivity off) | energy_conservation should be closer to 1.0 (< 5% drift) | |
| A6.4 | Enable radiation, check energy partition | Radiated energy + plasma energy + residual should equal bank energy | |

---

### UAT-A.7 — Edge Cases and Failure Modes

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| A7.1 | Set V0 = 0 V, ARM | Should warn: "Voltage may be too low for breakdown" | |
| A7.2 | Set anode_radius > cathode_radius | Should warn about geometry mismatch | |
| A7.3 | Set negative capacitance | Validation error; ARM fails | |
| A7.4 | Set CFL = 0 | Validation error or warning | |
| A7.5 | Set simulation time very short (1 ns) | Simulation finishes quickly; few steps; pinch "Not reached" | |
| A7.6 | During a running simulation, click STOP | Simulation halts; status changes to Finished; partial data available | |
| A7.7 | Rapidly click ARM-FIRE-RESET-ARM-FIRE | No crashes; state machine correctly transitions | |

---

### UAT-A.8 — Physicist Satisfaction Criteria

| Criterion | Description | Weight |
|-----------|-------------|--------|
| **Physical plausibility** | Current waveforms, pinch dynamics, and neutron yields are in the right ballpark | Critical |
| **Scaling law fidelity** | Y_n vs I^4, I_peak vs V0 scalings qualitatively correct | High |
| **Energy conservation** | Total energy tracked and reported; drift < 10% | Critical |
| **Parameter sensitivity** | Changing V0, pressure, geometry produces expected directional changes | High |
| **Transparency** | Tooltips explain physics; chat glossary is accurate | Medium |
| **Backend agreement** | Multiple backends produce consistent results for same problem | High |
| **Surrogate utility** | WALRUS predictions (when available) are faster and qualitatively correct | Medium |
| **Professional presentation** | Engineering notation, correct units, clean oscilloscope traces | Medium |

---

## 3. UAT-B: New Student / Intern

### Persona

**Alex Chen** — Incoming summer intern with a B.S. in Mechanical Engineering. Alex has:
- Taken intro plasma physics but never worked with DPF devices
- Used MATLAB/Python for classwork but no experience with MHD codes
- No familiarity with terms like WENO5, HLLD, Braginskii, or CFL
- Learns by exploring, clicking buttons, and reading tooltips
- Gets frustrated by unclear error messages or unexplained failures

### Test Environment

- DPF Unified GUI running
- Python backend started (minimal setup)
- WALRUS backend offline (typical for a new user)

---

### UAT-B.1 — First Launch Experience

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| B1.1 | Launch app for the first time | Clean interface loads; no errors; left panel visible with parameters | |
| B1.2 | Read the TopBar | Sees "DPF SIMULATOR v1.0.0" and status "Idle" | |
| B1.3 | Scroll through left panel sections | 11 collapsible sections visible; all labels readable | |
| B1.4 | Notice the parameter values have units | Units displayed next to each input (F, V, H, Ohm, Pa, etc.) | |
| B1.5 | Click the ARM button without changing anything | Either validation passes (with default config) or clear error message appears | |

---

### UAT-B.2 — Tooltip Discovery & Learning

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| B2.1 | Hover over "CAPACITANCE" label | Tooltip appears: "Energy storage capacitance of the capacitor bank. E = 1/2 CV^2." | |
| B2.2 | Hover over "CFL NUMBER" | Tooltip: "Courant stability factor — lower is more stable, higher is faster. Must be < 1." | |
| B2.3 | Hover over "NERNST EFFECT" toggle | Tooltip explains what the Nernst effect is in accessible language | |
| B2.4 | Hover over "BREMSSTRAHLUNG" toggle | Tooltip: "Free-free radiation from electron-ion collisions..." | |
| B2.5 | Hover over "ANOMALOUS ALPHA" | Tooltip: "Anomalous resistivity coefficient — enhances Ohmic heating near the pinch column." | |
| B2.6 | Student understands at least 5 parameters from tooltips alone | Tooltips provide sufficient context for a newcomer | |

---

### UAT-B.3 — Guided First Simulation (Preset)

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| B3.1 | Click the preset dropdown | List of device presets appears (PF-1000, NX2, etc.) | |
| B3.2 | Select "PF-1000" | All parameters auto-fill; student doesn't need to know correct values | |
| B3.3 | Click ARM | Advisory panel shows feedback; button changes to FIRE | |
| B3.4 | Read advisories | Messages are in plain English; student understands the severity levels (info/warning) | |
| B3.5 | Click FIRE | Oscilloscope view appears with live traces | |
| B3.6 | Watch the traces update in real-time | Cyan line (current) oscillates; student can read the axis labels and units | |
| B3.7 | Wait for simulation to finish | "Finished" status shown; PostShotPanel appears with metrics | |
| B3.8 | Read PostShot metrics | Pinch time, peak current, neutron yield displayed with engineering notation (MA, us, etc.) | |
| B3.9 | Understand energy partition | Bar chart or table showing where the energy went | |
| B3.10 | Click RESET | Returns to Idle state; ready for new run | |

---

### UAT-B.4 — Exploration & Experimentation

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| B4.1 | Change voltage from 27kV to 15kV | Input accepts the value; no error | |
| B4.2 | ARM and read advisories | May see "Voltage may be too low for breakdown" warning | |
| B4.3 | FIRE anyway | Simulation runs; peak current is lower than before | |
| B4.4 | Compare to previous run mentally | Student observes cause-and-effect (lower voltage = lower current) | |
| B4.5 | Try changing gas species from D2 to Argon | Dropdown changes species; ion mass updates automatically | |
| B4.6 | Try entering an invalid value (negative capacitance) | Clear validation error message; ARM button does not proceed | |
| B4.7 | Try entering a very large voltage (1 MV) | Either warning advisory or simulation runs (student sees dramatic results) | |

---

### UAT-B.5 — WALRUS Chat Discovery

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| B5.1 | Click "AI Co-Pilot" button in TopBar | Sidebar opens showing panels | |
| B5.2 | Find the Chat panel (scroll down in sidebar) | "Ask WALRUS..." input field visible with Send button | |
| B5.3 | Type "help" and press Enter | Response lists all question types with examples; student now knows what to ask | |
| B5.4 | Type "what is a z-pinch?" | Glossary response explains z-pinch in 2-3 sentences | |
| B5.5 | Type "what is bremsstrahlung?" | Clear explanation accessible to a student | |
| B5.6 | Type "explain CFL" | Explains CFL condition in understandable terms | |
| B5.7 | Click a suggestion chip | Corresponding question is auto-sent and answered | |
| B5.8 | Type "what maximizes neutron yield?" | Response explains inverse design concept and next steps | |
| B5.9 | Type something random like "banana" | Friendly fallback: "I didn't understand that question. Try asking about..." | |
| B5.10 | Click "Clear" button | Chat history cleared; fresh start | |

---

### UAT-B.6 — Gas Species & Unit Understanding

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| B6.1 | Open GAS section | See gas species dropdown, fill pressure, density, ion mass, temperature | |
| B6.2 | Change gas from D2 to Helium | Ion mass auto-updates to 4 amu | |
| B6.3 | Change fill pressure | Density recalculates automatically (ideal gas) | |
| B6.4 | Hover over "FILL PRESSURE" | Tooltip explains what fill pressure means | |
| B6.5 | Note the units shown (Pa, kg/m^3, amu, K) | All parameters have visible units | |

---

### UAT-B.7 — Error Recovery

| Step | Action | Expected Result | Pass/Fail |
|------|--------|-----------------|-----------|
| B7.1 | Delete the voltage value entirely (empty field) | Field shows validation state; ARM may fail with clear message | |
| B7.2 | Set capacitance to 0 | Warning or error; simulation would be meaningless | |
| B7.3 | Click FIRE during a running simulation | Button shows STOP (red); clicking it stops the simulation cleanly | |
| B7.4 | After stopping, click RESET | Returns to Idle; student can try again | |
| B7.5 | Close and reopen the AI sidebar | Sidebar toggles correctly; state preserved or cleanly reset | |

---

### UAT-B.8 — Learning Outcome Assessment

| Criterion | Description | Weight |
|-----------|-------------|--------|
| **Time to first simulation** | Student runs a preset simulation within 5 minutes | Critical |
| **Tooltip comprehension** | Student learns 3+ physics concepts from tooltips alone | High |
| **Chat utility** | Student uses chat to learn 2+ physics terms | High |
| **Cause-and-effect** | Student observes that changing voltage changes peak current | High |
| **No dead ends** | Student never gets stuck with no way forward | Critical |
| **Error clarity** | All error messages suggest what to fix | High |
| **Self-guided exploration** | Student tries 3+ parameter changes without instruction | Medium |
| **Confidence** | Student feels comfortable running a second simulation | Critical |
| **Fun factor** | Student describes the experience as interesting or engaging | Medium |

---

## Appendix A — Test Matrix Summary

### Physics Coverage Map

| Physics Module | T1 Unit | T2 Integration | T3 System | T4 Surrogate |
|----------------|---------|----------------|-----------|--------------|
| MHD core (WENO5 + HLLD) | Sod, Brio-Wu | Orszag-Tang, Alfven wave | PF-1000, NX2 | Rollout divergence |
| Circuit coupling (RLC) | Analytical RLC | Energy balance | Lee model | Sweep fidelity |
| Resistive diffusion | Gaussian | Convergence study | Ohmic heating | -- |
| Spitzer resistivity | NRL tables | -- | R_plasma evolution | -- |
| Bremsstrahlung | Power formula | -- | Radiated energy | -- |
| DD fusion yield | Bosch-Hale | -- | Y_n scaling | Parameter sweep |
| Braginskii transport | -- | Anisotropic ring | -- | -- |
| Nernst effect | -- | -- | Te modification | -- |
| Anomalous resistivity | -- | -- | Pinch dynamics | -- |
| Metal GPU (HLLD+WENO5-Z+SSP-RK3+CT) | Sod shock parity, HLLD stability, WENO5-Z convergence | Energy conservation, SSP-RK3 accuracy | Cross-backend L1, max accuracy config | -- |
| Metal GPU float64 | Float64 precision mode | Convergence order >= 1.7 | -- | -- |
| AthenaK (Kokkos) | -- | Blast parity | State dict parity | -- |

### UAT Coverage Map

| Feature | UAT-A (PhD) | UAT-B (Student) |
|---------|-------------|-----------------|
| Preset selection | A2.1 | B3.1-B3.2 |
| Tooltips | A2.2-A2.3 | B2.1-B2.6 |
| ARM/FIRE/RESET cycle | A2.4-A2.12 | B3.3-B3.10 |
| Advisories | A2.5, A3.6-A3.7 | B3.4 |
| Live oscilloscope | A2.6-A2.7 | B3.5-B3.6 |
| PostShot analysis | A2.9-A2.12 | B3.8-B3.9 |
| Physics sensitivity | A3.1-A3.5 | B4.1-B4.4 |
| Cross-backend | A4.1-A4.4 | -- |
| AI Sweep | A5.1-A5.3 | -- |
| AI Inverse Design | A5.4 | -- |
| Chat glossary | A5.5-A5.7 | B5.3-B5.10 |
| Energy conservation | A6.1-A6.4 | -- |
| Edge cases | A7.1-A7.7 | B7.1-B7.5 |
| Gas species | -- | B6.1-B6.5 |
| Error recovery | A7.7 | B7.1-B7.5 |

---

## Appendix B — Verification Test Implementation Checklist

| ID | Test | File | Status |
|----|------|------|--------|
| T1.1 | Resistive diffusion convergence | test_phase_c_verification.py | EXISTING |
| T1.2 | Sod shock tube | test_phase_f_verification.py | EXISTING |
| T1.3 | Brio-Wu MHD shock | test_phase_f_verification.py | EXISTING |
| T1.4 | Spitzer resistivity vs NRL | NEW — test_verification_spitzer.py | TODO |
| T1.5 | Bremsstrahlung power | NEW — test_verification_bremsstrahlung.py | TODO |
| T1.6 | DD fusion reactivity (Bosch-Hale) | NEW — test_verification_fusion.py | TODO |
| T1.7 | Analytical RLC circuit | NEW — test_verification_rlc.py | TODO |
| T2.1 | Orszag-Tang vortex | test_phase_c_verification.py | EXISTING |
| T2.2 | Sedov-Taylor blast | test_phase_c_verification.py | EXISTING |
| T2.3 | MHD convergence (Alfven wave) | NEW — test_verification_convergence.py | TODO |
| T2.4 | Anisotropic conduction ring | NEW — test_verification_aniso_cond.py | TODO |
| T2.5 | Circuit-MHD energy balance | NEW — test_verification_energy.py | TODO |
| T3.1 | Lee model comparison | test_phase_c_verification.py | EXISTING |
| T3.2 | PF-1000 experimental | NEW — test_verification_pf1000.py | TODO |
| T3.3 | Scaling law verification | NEW — test_verification_scaling.py | TODO |
| T4.1 | Surrogate single-step | NEW — test_verification_surrogate.py | TODO |
| T4.2 | Rollout divergence | NEW — test_verification_rollout.py | TODO |
| T4.3 | Sweep fidelity | NEW — test_verification_sweep.py | TODO |
| T4.4 | Ensemble uncertainty | NEW — test_verification_ensemble.py | TODO |
| T4.5 | Hybrid engine fallback | NEW — test_verification_hybrid.py | TODO |
| T5.1 | Metal Sod shock parity | test_phase_n_cross_backend.py | PASSING |
| T5.2 | Metal energy conservation (100+ steps) | test_phase_n_cross_backend.py | PASSING |
| T5.3 | AthenaK blast parity | test_phase_n_cross_backend.py | PASSING |
| T5.4 | HLLD solver instantiation + single step | test_phase_o_physics_accuracy.py | PASSING |
| T5.5 | HLLD Brio-Wu MHD shock stability | test_phase_o_physics_accuracy.py | PASSING |
| T5.6 | WENO5-Z reconstruction (5th-order convergence) | test_phase_o_physics_accuracy.py | PASSING |
| T5.7 | WENO5-Z interior convergence order 5.47-5.79 | test_phase_o_physics_accuracy.py | PASSING |
| T5.8 | Float64 precision mode (CPU fallback) | test_phase_o_physics_accuracy.py | PASSING |
| T5.9 | SSP-RK3 instantiation + single step | test_phase_o_physics_accuracy.py | PASSING |
| T5.10 | SSP-RK3 Sod shock stability | test_phase_o_physics_accuracy.py | PASSING |
| T5.11 | SSP-RK3 lower error than SSP-RK2 | test_phase_o_physics_accuracy.py | PASSING |
| T5.12 | SSP-RK3 + WENO5-Z + float64 convergence (order >= 1.7) | test_phase_o_physics_accuracy.py | PASSING |
| T5.13 | Maximum accuracy config (WENO5-Z + HLLD + SSP-RK3 + float64 + CT) | test_phase_o_physics_accuracy.py | PASSING |

---

## Appendix C — Running the UATs

### Prerequisites

```bash
# Start the backend
cd dpf-unified
pip install -e ".[dev]"
dpf serve --port 8765

# In another terminal, start the GUI
cd gui
npm install
npm run dev
```

### Recording Results

For each UAT step:
1. Execute the action described
2. Compare actual behavior to "Expected Result"
3. Mark Pass/Fail in the rightmost column
4. Note any deviations, errors, or unexpected behavior
5. Record timestamps for time-to-completion metrics

### Reporting

After completing all tests, compile:
- Total pass/fail counts per section
- Critical failures (any step marked Critical that fails)
- Screenshots of unexpected behavior
- Student feedback (UAT-B only): qualitative impressions, confusion points
- Physicist feedback (UAT-A only): physics accuracy concerns, missing features
