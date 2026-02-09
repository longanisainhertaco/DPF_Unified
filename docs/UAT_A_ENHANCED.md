# UAT-A Enhanced: Hyper-Critical PhD Plasma Physicist Evaluation

## DPF Unified -- User Acceptance Test Plan (Expert Persona)

**Document version**: 1.0
**Date**: 2026-02-09
**Scope**: End-to-end physics fidelity, numerical integrity, and professional usability of the DPF Unified simulator from the perspective of an experienced pulsed-power experimentalist.

---

## 1. Persona: Dr. Maria Vasquez

### 1.1 Backstory

Dr. Maria Vasquez holds a PhD in plasma physics from the Institute of Nuclear Physics, Krakow (2013). She spent six years at IPPLM Warsaw operating the PF-1000, the largest Mather-type DPF device in Europe (266 uF at 40 kV, up to 1.2 MJ stored energy). She led the diagnostic campaign that correlated neutron anisotropy with axial plasma sheath asymmetry (Vasquez et al., J. Fusion Energy, 2018 -- fictitious for persona purposes). She has subsequently worked at Sandia National Laboratories on pulsed-power z-pinch experiments (Z machine) and currently holds a senior research position at a DOE lab, where she evaluates computational tools for NNSA pulsed-power programs.

She has direct hands-on experience with:
- Lee model code (S. Lee, IAEA, 2014 revision)
- GORGON (Imperial College, 3D resistive MHD)
- FLASH (University of Chicago, AMR MHD)
- MACH2 (AFRL, 2D Eulerian MHD)
- HYDRA (LLNL, ALE radiation-hydrodynamics)

### 1.2 Motivation for Testing

Dr. Vasquez has been asked by her division to evaluate DPF Unified as a potential rapid-prototyping and parameter-exploration tool for upcoming DPF experiments. She has three weeks to determine whether the code is:

1. **Physically trustworthy**: Does it reproduce the MHD phenomenology she has personally observed on PF-1000 and published in the literature?
2. **Numerically sound**: Are the discretization schemes, conservation properties, and convergence behaviors consistent with what she expects from a competent MHD code?
3. **Operationally useful**: Can she set up a simulation, explore parameter space, and trust the diagnostics output without babysitting every run?

She is not looking for a production-grade HEDP code. She is looking for a tool that does not silently produce garbage. Her threshold for rejection is low: one physically nonsensical result without a clear warning, and she will not recommend the tool.

### 1.3 Testing Philosophy

Dr. Vasquez follows a strict evaluation protocol:

- **Trust nothing that is not verified against an analytical solution or published data.**
- **Every numerical result must have stated units; unitless outputs are unacceptable.**
- **Conservation violations must be reported, even if small. "5% energy drift" is not acceptable for a conservative MHD scheme on a clean test problem.**
- **Every physics approximation must be stated and justified. Unstated approximations are bugs.**
- **The code must fail gracefully on unphysical inputs. Silent acceptance of garbage parameters is a disqualifying defect.**

---

## 2. Pre-Test Checklist

Before running a single simulation, Dr. Vasquez reviews the following. Each item is scored PASS/FAIL. Two or more FAILs in this section will cause her to delay testing until resolved.

### 2.1 Documentation Review

| ID | Check | Expected | Pass/Fail |
|----|-------|----------|-----------|
| P-1 | Is the coordinate system documented? (Cartesian vs. cylindrical, axis orientation) | Clear statement of r-z vs. x-y-z conventions, axis labeling, cell-centering vs. face-centering for B | |
| P-2 | Is the equation set written down explicitly? | Full MHD equations in conservation form with source terms (circuit, radiation, resistive, Hall) clearly enumerated | |
| P-3 | Are the numerical schemes identified? | WENO5 reconstruction, HLL/HLLD Riemann solver, SSP-RK2 time integration, Dedner cleaning stated explicitly | |
| P-4 | Are units documented for every state variable? | rho [kg/m^3], v [m/s], p [Pa], B [T], Te [K], Ti [K], psi [T*m/s] | |
| P-5 | Is the Spitzer resistivity formula written out with constants? | eta = 1.03e-4 * Z_eff * ln(Lambda) / Te^(3/2) [Ohm*m], with Te in eV | |
| P-6 | Is the anomalous resistivity model documented? | Functional form, threshold criterion, smoothing, and the alpha parameter meaning | |
| P-7 | Is the circuit model documented as a circuit diagram or coupled ODE? | V_cap = V0 - (1/C)*integral(I*dt), L_total*dI/dt + R_total*I = V_cap, with L_total = L0 + L_plasma(t) | |
| P-8 | Are boundary conditions enumerated for each boundary? | r=0 (axis symmetry), r=r_max (outflow or wall), z=0 (electrode), z=z_max (electrode or outflow) | |
| P-9 | Is there a statement about divergence-free B enforcement? | Dedner hyperbolic cleaning with ch and cr parameters, or constrained transport, or Powell 8-wave | |
| P-10 | Is the CFL condition stated with the signal speed definition? | CFL = max(|v| + c_f) * dt / dx, where c_f is the fast magnetosonic speed | |

### 2.2 Code Inspection (Spot Checks)

| ID | Check | Expected | Pass/Fail |
|----|-------|----------|-----------|
| P-11 | Open `mhd_solver.py`, verify WENO5 weights are (0.1, 0.6, 0.3) for ideal weights | Matches standard Jiang-Shu (1996) WENO5 weights | |
| P-12 | Verify Riemann solver uses correct signal speed estimates | HLL: S_L = min(v_L - c_f_L, v_R - c_f_R), S_R = max(v_L + c_f_L, v_R + c_f_R) | |
| P-13 | Check that the energy equation uses total energy E = p/(gamma-1) + 0.5*rho*v^2 + B^2/(2*mu_0) | Not internal energy only -- that would be non-conservative | |
| P-14 | Verify that the induction equation has the correct sign: dB/dt = curl(v x B) - curl(eta*J) | Missing minus sign on resistive term is a classic bug | |
| P-15 | Check that the Bosch-Hale parameterization for D-D reactivity is used, not the simpler Duane fit | Bosch-Hale (1992) coefficients for D(d,n)He3 branch | |

---

## 3. Detailed Test Scenarios

### Scenario 1: Free-Ringing RLC Circuit (No Plasma Load)

**Purpose**: Validate the circuit solver in isolation before coupling to plasma.

**Setup**:
- V0 = 27 kV, C = 1.332 mF, L0 = 15 nH, R0 = 3 mOhm
- Plasma load disabled (R_plasma = 0, L_plasma = 0)
- Run for 5 quarter-periods: t_final = 5 * pi/2 * sqrt(L0 * C) = 5 * pi/2 * sqrt(15e-9 * 1.332e-3) = ~1.11 us

**Expected behavior**:
- Damped sinusoidal current: I(t) = V0 * sqrt(C/L0) * sin(omega*t) * exp(-R0*t/(2*L0))
- omega = sqrt(1/(L*C) - (R/(2L))^2) = ~223 krad/s
- I_peak = V0 * sqrt(C/L0) * exp(-R0*t_peak/(2*L0)) = ~8.0 MA (!!!)

**Red flags**:
- Current amplitude growing in time (energy creation)
- Frequency not matching sqrt(1/LC)
- Oscillation not decaying at the correct rate R/(2L)
- Any non-smooth features in the current trace (kinks, jumps)

**Quantitative acceptance criteria**:
- Phase error < 0.1% over 5 cycles relative to analytical solution
- Amplitude error < 0.5% at each peak
- Total energy E_cap + E_ind = (1/2)*C*V(t)^2 + (1/2)*L*I(t)^2 conserved to < 0.01% (minus Ohmic losses which must match integral of I^2*R*dt)

**Follow-up probing questions**:
- "What happens at exactly t=0? Is there a current spike from the initial dV/dt discontinuity? How is that handled numerically?"
- "If I set R0=0 exactly, does the circuit conserve energy to machine precision? It should, since there is no dissipation."
- "Is the circuit integrator symplectic? If not, why not?"

---

### Scenario 2: PF-1000 Baseline -- Current Waveform Validation

**Purpose**: Compare the simulated current trace against published PF-1000 experimental data.

**Setup**:
- PF-1000 preset: V0 = 27 kV, C = 1.332 mF, L0 = 15 nH, R0 = 3 mOhm
- Cylindrical geometry: 64 (r) x 1 (theta) x 128 (z), dx = 0.5 mm
- Anomalous alpha = 0.05
- Bremsstrahlung ON, sheath BC ON
- t_final = 5 us

**Expected behavior** (references: Lee & Serban 1996, Scholz et al. 2004, Scholz et al. 2006):
- Axial run-down phase: current rising approximately as damped sinusoid from 0 to ~4-5 us
- Current dip (kink) near pinch time (~5-7 us at PF-1000 operating conditions)
- Peak current before pinch: 1.5-2.0 MA
- Post-pinch current oscillation or decay

**Red flags**:
- Peak current exceeding 5 MA (unphysical for PF-1000 at 27 kV)
- No current dip visible at pinch -- the current dip is the most characteristic DPF signature
- Pinch occurring before 2 us or after 10 us (outside physical range for PF-1000)
- Negative current values during the first half-cycle (should be strictly positive until pinch)
- Perfectly smooth current waveform with no pinch-related features (suggests plasma is not actually compressing)

**Quantitative acceptance criteria**:
- I_peak within [1.0, 3.0] MA (broad range acceptable for MHD-level code)
- t_pinch within [3.0, 10.0] us
- Current dip magnitude: dI/I_peak > 5% (visible dip)
- Quarter-period t_qp = pi/2 * sqrt(L0*C) = ~0.22 us sets the timescale -- initial current rise should be consistent

**Follow-up probing questions**:
- "The Lee model predicts I_peak = 1.87 MA for these exact parameters. How does your result compare?"
- "What is the effective plasma resistance at pinch time? Is it consistent with Spitzer at the computed Te?"
- "Show me dL_plasma/dt during the compression. Is it positive and peaking at pinch?"
- "How is the sheath mass computed? Snow-plow model? If so, what is the mass sweep-up efficiency?"

---

### Scenario 3: Mesh Convergence Study (Richardson Extrapolation)

**Purpose**: Verify that the spatial discretization converges at the expected order.

**Setup**: Run the PF-1000 configuration at four resolutions:
- 16 x 1 x 32 (dx = 2.0 mm)
- 32 x 1 x 64 (dx = 1.0 mm)
- 64 x 1 x 128 (dx = 0.5 mm)
- 128 x 1 x 256 (dx = 0.25 mm)

Hold CFL = 0.4 constant (so dt scales with dx). Use identical physics settings.

**Expected behavior**:
- Peak current should converge monotonically as grid is refined
- The convergence rate should be at least 2nd order for smooth flows (SSP-RK2 + WENO5 limited to 2nd at discontinuities)
- Richardson extrapolation: if f(h) = f_exact + C*h^p, then p = log2((f_2h - f_h)/(f_h - f_{h/2}))
- Energy conservation should improve with resolution

**Red flags**:
- Non-monotonic convergence (result at 64x128 differs from 32x64 in a non-systematic direction)
- Convergence order < 1.0 (first-order behavior indicates the WENO5 is clipping to first-order everywhere, probably a bug in the smoothness indicators)
- Convergence order > 3.0 (super-convergence usually indicates a bug, not a feature)
- The coarsest grid producing results that differ by more than a factor of 2 from the finest grid
- Wall-clock time not scaling as expected: each doubling of resolution in 2D should increase time by ~8x (4x cells, 2x time steps from CFL)

**Quantitative acceptance criteria**:
- Measured convergence order p in [1.5, 2.5] for I_peak
- Measured convergence order p in [1.5, 2.5] for E_total at t_final
- Richardson-extrapolated I_peak within 10% of finest-grid result
- Energy conservation improving with resolution (dE/E at finest grid < dE/E at coarsest grid)

**Follow-up probing questions**:
- "What is your effective convergence order for the magnetic field at the axis? Is it polluted by the coordinate singularity at r=0?"
- "Is the geometric source term (1/r terms in cylindrical MHD) treated as a centered difference or is there a correction for the cell-averaged 1/r?"
- "At 128x256, are you resolving the current sheet thickness? What is the Sweet-Parker thickness at your Lundquist number?"
- "Have you checked for the carbuncle instability on the axis? Grid-aligned shocks in cylindrical coordinates are notorious for this."

---

### Scenario 4: Energy Conservation Audit (Long-Run Stress Test)

**Purpose**: Verify that the conservative MHD scheme actually conserves total energy when there are no explicit dissipation sources.

**Setup**:
- PF-1000 geometry, 64 x 1 x 128
- Disable ALL dissipation: bremsstrahlung OFF, resistivity OFF, viscosity OFF, Nernst OFF, anomalous resistivity OFF (alpha=0), sheath OFF
- Run for t_final = 10 us (twice the normal PF-1000 run)

**Expected behavior**:
- Without dissipation, total energy E = E_kinetic + E_thermal + E_magnetic + E_capacitor should be conserved
- For a conservative finite-volume scheme, conservation should be to machine precision (roundoff only), modulo boundary fluxes
- Energy conservation should be independent of CFL number (run at CFL=0.2 and CFL=0.8 and compare)

**Red flags**:
- Energy drift > 0.1% over the full run with no dissipation sources -- this indicates a non-conservative discretization or a source-term bug
- Energy drift that depends on CFL number (suggests time-integration error dominating)
- Energy suddenly jumping at a specific time (boundary condition injecting/removing energy)
- Negative pressures or densities appearing (positivity failure in the Riemann solver)
- The Dedner cleaning scalar psi carrying significant energy that is not accounted for

**Quantitative acceptance criteria**:
- |E(t_final) - E(0)| / E(0) < 1e-6 for ideal MHD (no dissipation, periodic BCs)
- |E(t_final) - E(0)| / E(0) < 1e-3 for ideal MHD with realistic BCs (some boundary flux expected)
- div(B) norm decreasing or bounded (Dedner cleaning working)
- No negative densities or pressures at any time step

**Follow-up probing questions**:
- "Is div(B) actually cleaning to zero, or is it just being damped to some plateau? Show me max(|div(B)|) vs. time."
- "What is the total energy in the psi field? If it is significant (> 1% of E_total), your Dedner cleaning is acting as an energy sink."
- "How do you handle the geometric source terms in the energy equation for cylindrical coordinates? They must be exactly balanced or you get O(1) energy errors."
- "What is your positivity limiter? Do you have one? If not, what happens when rho goes negative at the axis?"

---

### Scenario 5: Spitzer Resistivity vs. Lee-More Comparison

**Purpose**: Evaluate the resistivity model at extreme conditions relevant to the DPF pinch.

**Setup**:
- PF-1000 baseline
- Run twice: (a) with Spitzer resistivity, (b) with anomalous resistivity (alpha = 0.05)
- Record eta(r, z, t) at the pinch axis at pinch time

**Expected behavior**:
- Spitzer eta = 1.03e-4 * Z * ln(Lambda) / Te^{3/2} [Ohm*m] (with Te in eV)
- At Te = 1 keV, n_e = 1e25 m^-3: eta_Spitzer ~ 4.5e-8 Ohm*m
- At Te = 100 eV, n_e = 1e24 m^-3: eta_Spitzer ~ 1.4e-5 Ohm*m
- Anomalous resistivity should dominate at low Te or when the drift velocity exceeds the ion-acoustic speed
- Magnetic Reynolds number Rm = mu_0 * v * L / eta should be reported

**Red flags**:
- Resistivity not updating with Te (indicates a static resistivity bug)
- Spitzer eta at Te = 10 eV giving values below 1e-6 Ohm*m (too low -- calculation error in Te units)
- No difference between Spitzer-only and anomalous runs at high density / low Te (anomalous should matter there)
- Coulomb logarithm being held fixed when dynamic_coulomb_log is True

**Quantitative acceptance criteria**:
- Spitzer eta at Te = 1 keV within [1e-8, 1e-7] Ohm*m
- Spitzer eta at Te = 100 eV within [1e-6, 1e-4] Ohm*m
- Anomalous eta exceeding Spitzer eta by at least 10x in the low-Te sheath region
- Magnetic diffusion time t_diff = mu_0 * L^2 / eta consistent with pinch dynamics timescale

**Follow-up probing questions**:
- "Why are you using Spitzer and not Lee-More? Lee-More is more accurate at high densities where the plasma is partially degenerate."
- "What is your effective Z for deuterium? Are you using Z=1 or are you accounting for impurities sputtered from the electrodes?"
- "Does your Coulomb logarithm have the correct density dependence? At very high density the classical Coulomb log formula breaks down."
- "What happens to the resistivity in the neutral gas region ahead of the sheath? You need a neutral collision model there, not Spitzer."

---

### Scenario 6: Bremsstrahlung vs. Line Radiation

**Purpose**: Check that the radiation model is physically consistent and correctly implemented.

**Setup**:
- PF-1000 baseline
- Run three cases:
  - (a) Bremsstrahlung only (default)
  - (b) Bremsstrahlung + line radiation with 1% copper impurity (impurity_Z=29, impurity_fraction=0.01)
  - (c) All radiation off

**Expected behavior**:
- P_brem = 1.69e-32 * Z^2 * n_e * n_i * sqrt(Te) [W/m^3] (NRL Formulary)
- At Te = 1 keV, n_e = 1e25: P_brem ~ 5.3e12 W/m^3 per unit volume
- For copper impurities (Z=29), line radiation dominates bremsstrahlung below ~1 keV by orders of magnitude
- Line radiation + bremsstrahlung should significantly reduce peak Te compared to bremsstrahlung alone
- Post's criterion: high-Z impurities cause radiation collapse when Z_eff exceeds ~2-3 at DPF temperatures

**Red flags**:
- Adding copper impurities having no effect on Te (line radiation not actually being computed)
- Negative temperatures appearing with strong radiation
- Radiated power exceeding the total thermal energy in a single time step (radiation subcycling needed)
- Bremsstrahlung power at 100 eV being comparable to line radiation power for copper -- it should be far less

**Quantitative acceptance criteria**:
- Bremsstrahlung power within 20% of NRL Formulary at reference (Te, n_e) pairs
- Line radiation power for Cu at 100 eV exceeding bremsstrahlung by at least 10x
- Peak Te with impurities lower than without (radiation cooling working)
- Total radiated energy not exceeding initial stored energy (energy conservation)

**Follow-up probing questions**:
- "What cooling function are you using for copper line radiation? Is it from ADAS, FLYCHK, or some simpler parametric model?"
- "Are you accounting for opacity? At high density the plasma may be optically thick to certain line transitions."
- "How do you handle the radiation timestep constraint? If the cooling time is shorter than the hydro timestep, you get negative temperatures."
- "Is the Gaunt factor temperature-dependent? A fixed g_ff = 1.2 is only valid over a narrow range."

---

### Scenario 7: Neutron Yield Scaling Law

**Purpose**: Verify that the simulated neutron yield scales correctly with peak current (Lee scaling: Y_n proportional to I_pinch^4).

**Setup**: Voltage sweep at constant geometry:
- V0 = [10, 15, 20, 27, 35, 45] kV (6 points)
- All other PF-1000 parameters held fixed
- Record I_peak and Y_n for each run

**Expected behavior** (Lee & Serban 1996, Lee 2008):
- Y_n proportional to I_pinch^alpha where alpha is in [3.5, 4.5]
- Plot log(Y_n) vs. log(I_peak): slope should be ~4
- The absolute Y_n values should bracket 1e8 to 1e11 across this range
- Low-voltage runs may not achieve pinch at all (Y_n = 0)

**Red flags**:
- Y_n scaling exponent < 2 or > 6 (unphysical -- something is wrong with the fusion model)
- Y_n not depending on I_peak at all (suggests neutron calculation is independent of plasma conditions)
- Y_n at 10 kV being comparable to Y_n at 45 kV (no physics sensitivity)
- Negative or imaginary neutron yields
- Y_n at 27 kV being above 1e13 (would exceed total D-D fusion energy available in the fuel)

**Quantitative acceptance criteria**:
- Power-law exponent alpha in [3.0, 5.5] (broader range acceptable for MHD code vs. experiment)
- Y_n at 27 kV within [1e8, 1e12]
- Y_n at 10 kV < Y_n at 27 kV (monotonic with voltage)
- At least one run (lowest voltage) shows Y_n = 0 or negligible (no pinch achieved)

**Follow-up probing questions**:
- "Is the neutron yield computed as a volume integral of (1/4)*n_D^2*<sigma*v>*dV over the pinch volume, or is it a simpler estimate?"
- "Are you using the Bosch-Hale parameterization for <sigma*v>? If so, which branch: D(d,n)He3 or D(d,p)T?"
- "What is the effective ion temperature in the pinch at 27 kV? Is it thermonuclear or beam-target?"
- "Does your fusion model account for the beam-target contribution? PF-1000 neutrons are known to be anisotropic, suggesting beam-target dominates."

---

### Scenario 8: Low Fill Pressure Edge Case

**Purpose**: Test behavior at operating conditions that are physically extreme but experimentally accessible.

**Setup**:
- PF-1000 geometry
- Fill pressure = 0.5 Torr (0.05 of baseline), corresponding to rho0 ~ 4e-5 kg/m^3
- V0 = 27 kV (standard)

**Expected behavior**:
- Much faster sheath propagation (less mass to sweep up)
- Earlier pinch time (potentially 2-3 us instead of 5-7 us)
- Higher peak temperatures (less mass to heat)
- Lower neutron yield (less fuel despite higher Ti)
- Potentially incomplete sheath formation or sheath leaking

**Red flags**:
- Negative densities in the evacuated region behind the sheath (positivity failure)
- Te exceeding 100 keV (runaway heating with no physical basis at these energies)
- The code crashing or producing NaN (should handle gracefully)
- Identical results to the baseline (no sensitivity to fill pressure indicates a bug)
- Sheath speed exceeding the speed of light (no relativistic limit in the MHD equations, but v/c > 0.01 should trigger a warning)

**Quantitative acceptance criteria**:
- Pinch time earlier than baseline by at least 30%
- Peak Te higher than baseline
- No NaN or Inf values in any output field
- All densities remain positive
- Energy conservation maintained (radiation losses may differ)

**Follow-up probing questions**:
- "At very low pressure, your mean free path exceeds the device dimensions. Is MHD even valid here? Do you warn the user?"
- "What is the Knudsen number Kn = lambda_mfp / L at these conditions? If Kn > 0.01, you need at least a two-fluid correction."
- "How does your code handle the near-vacuum region behind the sheath? Is there a density floor? If so, what is it and how does it affect conservation?"
- "At low pressure, the Hall parameter omega_ce * tau_ei can exceed 100. Are you including the Hall term? If not, why not?"

---

### Scenario 9: High Voltage Stress Test

**Purpose**: Test the upper bound of operating parameters.

**Setup**:
- PF-1000 geometry
- V0 = 80 kV (well above normal PF-1000 operating range)
- All other parameters at baseline

**Expected behavior**:
- Very high peak current: I_peak > 3 MA
- Stored energy E = 0.5 * C * V0^2 = 0.5 * 1.332e-3 * (80e3)^2 = 4.26 MJ (4x normal)
- Extremely fast dynamics -- CFL may require very small dt
- Potential numerical difficulties from strong shocks

**Red flags**:
- Code accepting this without any advisory warning (a physicist would know this is outside the validated range)
- Temperatures exceeding 1 MeV (at which point pair production becomes relevant -- well outside MHD validity)
- The CFL condition being violated (explicit time step too large for the fast magnetosonic speed at these energies)
- Grid-scale oscillations (odd-even decoupling from the central difference operators in the source terms)
- Carbuncle instability on the axis (grid-aligned strong shock + low dissipation)

**Quantitative acceptance criteria**:
- Simulation either completes with physically bounded results OR fails gracefully with a clear error message
- If it completes: I_peak in [2, 10] MA, Te_max < 100 keV
- No grid-scale checkerboard patterns in any field
- Advisory system warns about extreme parameters before the run starts

**Follow-up probing questions**:
- "At V0 = 80 kV, the electric field in the breakdown region exceeds 10 kV/cm. Are you modeling the initial breakdown phase or just assuming instant ionization?"
- "What is your largest CFL-limited timestep at peak compression? Is it resolving the ion cyclotron frequency?"
- "Do you have any numerical viscosity to suppress carbuncle? If so, is it documented and does it contaminate the physical dissipation?"

---

### Scenario 10: Asymmetric Electrode Geometry

**Purpose**: Test sensitivity to geometric configuration beyond the standard Mather type.

**Setup**:
- Anode radius = 0.03 m (half of PF-1000 standard 0.0575 m)
- Cathode radius = 0.08 m (same as standard)
- This gives a radius ratio of 0.375 (vs. standard 0.719)
- All other PF-1000 parameters at baseline

**Expected behavior**:
- Higher inductance per unit length: L' = (mu_0 / 2*pi) * ln(r_cathode / r_anode)
- At r_a = 0.03, r_c = 0.08: L' = 1.96e-7 H/m (vs. 6.56e-8 H/m for standard)
- Higher magnetic pressure at the anode surface (B_theta ~ I / (2*pi*r_a))
- Faster pinch but potentially less stable (stronger sausage instability drive)
- Lower I_peak due to higher inductance

**Red flags**:
- No change in dynamics compared to baseline (geometry not actually feeding into the inductance calculation)
- Peak current identical to baseline (L_plasma not being computed correctly from the geometry)
- The anode_radius = 0.03 being rejected by the validator (it should be valid, just unusual)

**Quantitative acceptance criteria**:
- I_peak lower than baseline by at least 20% (higher inductance reduces peak current)
- L_plasma contribution to total inductance clearly visible in diagnostics
- Pinch radius closer to the (smaller) anode radius
- Bennett equilibrium pressure balance: mu_0 * I^2 / (8*pi) = n_line * k_B * (Te + Ti)

**Follow-up probing questions**:
- "How do you compute dL_plasma/dt? Is it from the actual field configuration or from an approximate formula?"
- "For a Filippov-type geometry (large radius ratio), the sheath dynamics are qualitatively different. Does your code handle both Mather and Filippov topologies?"
- "What happens if I set anode_radius > cathode_radius? The validator should catch this, but does it?"

---

### Scenario 11: Dedner Divergence Cleaning Verification

**Purpose**: Verify that the div(B) = 0 constraint is maintained to acceptable levels.

**Setup**:
- PF-1000 baseline
- Record max(|div(B)|) at every output timestep
- Also record the total energy in the psi (Dedner scalar) field

**Expected behavior**:
- div(B) errors should be generated at shocks and current sheets
- Dedner cleaning should transport these errors to the boundary and damp them
- max(|div(B)|) should oscillate but remain bounded (not growing secularly)
- The Dedner scalar field should not carry more than 0.1% of total energy

**Red flags**:
- div(B) growing monotonically (cleaning not working)
- Large psi energy (> 1% of E_total) indicating that the cleaning is acting as an energy reservoir
- div(B) errors concentrated at the axis (coordinate singularity issue)
- Cleaning speed ch not being set correctly (ch should be > c_fast)

**Quantitative acceptance criteria**:
- max(|div(B)|) * dx / max(|B|) < 0.01 at all times (1% relative div(B))
- psi energy / E_total < 0.001 at all times
- No secular growth in div(B) norm over the simulation
- ch parameter automatically set to max(c_fast) across the domain

**Follow-up probing questions**:
- "Why Dedner cleaning and not constrained transport (CT)? CT maintains div(B) = 0 to machine precision."
- "What are your ch and cr parameters? Are they adaptive or fixed? If fixed, how do you know they are appropriate?"
- "Does the psi field have its own boundary conditions? What happens to divergence errors that reach the boundary?"
- "Have you compared div(B) errors between Python, Athena++, and AthenaK backends? Athena++ uses CT -- does it do better?"

---

### Scenario 12: GUI Display Validation (Physical Ranges)

**Purpose**: Verify that the GUI does not display physically nonsensical values.

**Setup**:
- Run PF-1000 baseline through the GUI
- Examine every displayed quantity in the oscilloscope and PostShot panels

**Expected behavior**:
- Temperatures: 300 K to 1e9 K (no negatives, no values above pair-production threshold)
- Densities: rho > 0 everywhere (never negative or zero)
- Pressures: p > 0 everywhere
- Current: monotonically rising until first peak, then may oscillate
- Neutron yield: non-negative, finite
- Energy conservation fraction: between 0.8 and 1.2 (outside this range indicates a problem)
- All values displayed with correct SI units and engineering notation

**Red flags**:
- Temperature displayed as negative or as "NaN"
- Pressure displayed as 0.00 (total vacuum is unphysical in MHD)
- Current displayed in wrong units (A vs. MA -- off by 1e6)
- Neutron yield displayed without units or with wrong units (should be total count or count/shot)
- Any quantity displayed as "Infinity" or "#VALUE!"
- Oscilloscope axes not auto-scaling properly (trace invisible because axis range is wrong)
- Time axis in seconds when microseconds would be appropriate (5e-6 vs. 5 us)

**Quantitative acceptance criteria**:
- All displayed temperatures in [0, 1e10] K
- All displayed densities > 0
- All displayed pressures > 0
- Current in MA (not A or kA), time in us (not s)
- Energy conservation displayed and within [0.80, 1.20]
- Engineering notation used for values > 1e4 or < 1e-3

**Follow-up probing questions**:
- "What is the minimum density in the simulation? Is there a density floor? If so, how does it affect the displayed values?"
- "When the simulation is running, are the oscilloscope traces showing instantaneous values or time-averaged values? For rapidly fluctuating quantities, this matters."
- "Is the neutron yield displayed as the time-integrated total or the instantaneous rate? For experimental comparison, I need the total."
- "Can I export the raw data behind these displays? I want to analyze it in my own plotting tools."

---

### Scenario 13: Cross-Backend Consistency

**Purpose**: Verify that different computational backends produce consistent results for the same physical problem.

**Setup**:
- Run the PF-1000 preset on all available backends: Python, Athena++, AthenaK
- Identical parameters, identical grid (as close as possible given backend limitations)

**Expected behavior**:
- All backends should produce qualitatively similar current waveforms
- Peak current should agree within 50% (different numerical schemes will give different results, but the physics should be the same)
- Pinch timing should agree within a factor of 2
- Energy conservation should be comparable or better on Athena++ (CT vs. Dedner)

**Red flags**:
- One backend producing a completely different qualitative result (e.g., no pinch when others show pinch)
- I_peak differing by more than a factor of 3 between backends
- One backend crashing while others complete
- AthenaK (Cartesian only) producing qualitatively wrong results for what should be a cylindrical problem

**Quantitative acceptance criteria**:
- I_peak agreement within factor of 2 across all backends
- t_pinch agreement within factor of 2
- Same qualitative current waveform shape (damped sinusoid with dip)
- Energy conservation within 10% on all backends

**Follow-up probing questions**:
- "AthenaK is Cartesian only. How are you mapping the cylindrical PF-1000 problem to a Cartesian grid? Is this a full 3D Cartesian run or a 2D slice?"
- "Athena++ uses HLLD while Python uses HLL. The HLLD solver is more accurate for MHD. Do you see a measurable difference in the current sheet structure?"
- "What reconstruction does each backend use? If Athena++ is using PPM and Python is using WENO5, the effective resolution is different."

---

### Scenario 14: Time Integration of Diagnostics

**Purpose**: Verify that neutron yield and radiated energy are properly time-integrated (not just snapshot values).

**Setup**:
- PF-1000 baseline
- Run with two different output intervals: (a) output every 10 steps, (b) output every 100 steps
- Compare final neutron yield and total radiated energy between the two runs

**Expected behavior**:
- Time-integrated quantities (Y_n, E_rad) should be independent of output interval
- They should be computed from the internal solver timestep, not the output timestep
- The peak values of instantaneous quantities (Te_max, I_peak) should be identical regardless of output interval (captured at the actual peak, not just at output times)

**Red flags**:
- Y_n differing by more than 5% between the two output intervals (indicates integration is tied to output, not solver timestep)
- E_rad changing with output interval (same issue)
- I_peak being missed because the output interval is too coarse (the actual peak falls between output snapshots)

**Quantitative acceptance criteria**:
- |Y_n(10-step) - Y_n(100-step)| / Y_n(10-step) < 1%
- |E_rad(10-step) - E_rad(100-step)| / E_rad(10-step) < 1%
- I_peak identical to 4 significant figures between the two runs

**Follow-up probing questions**:
- "Is the neutron yield computed at every internal timestep or only at output times?"
- "How do you compute the time-integrated radiated energy? Is it a trapezoidal rule over the internal timesteps?"
- "If I set the output interval to 1 (every step), does that change the physics at all? It should not."

---

## 4. Post-Test Evaluation Rubric

Dr. Vasquez scores the simulator on the following dimensions, each rated 1 (unacceptable) to 5 (excellent). A score of 2 or below on any Critical dimension is disqualifying.

| # | Dimension | Weight | 1 (Unacceptable) | 3 (Adequate) | 5 (Excellent) | Score |
|---|-----------|--------|-------------------|---------------|----------------|-------|
| 1 | **Conservation properties** | Critical | Energy drifts > 10% on clean problems | Conservation within 5% on most runs; some drift with strong shocks | Machine-precision conservation on ideal tests; < 1% on production runs | |
| 2 | **Convergence behavior** | Critical | No convergence or erratic convergence | 2nd-order convergence on smooth problems | Demonstrates both 2nd-order smooth and 1st-order shocked convergence with correct transition | |
| 3 | **Experimental agreement** | Critical | Results differ by > 10x from published PF-1000 data | Within factor of 3 of published data on key metrics | Within 50% of published data with stated assumptions | |
| 4 | **Numerical robustness** | Critical | Crashes or NaN on routine parameter variations | Handles most parameter ranges; occasional failures on extremes | Graceful degradation on all tested inputs; clear error messages | |
| 5 | **Physics transparency** | High | No documentation of approximations; opaque source terms | Key approximations stated; some missing details | Every approximation documented, justified, and referenceable | |
| 6 | **Unit consistency** | High | Mixed units or missing units in output | Consistent SI units throughout; occasional formatting issues | Engineering notation, tooltips with unit derivations, export with units in headers | |
| 7 | **Divergence-free B** | High | div(B) growing without bound | div(B) bounded but > 1% of |B| | div(B) < 0.01% of |B| or CT-based (machine-precision) | |
| 8 | **Diagnostic fidelity** | High | Neutron yield off by > 100x; not time-integrated | Within factor of 10; properly integrated | Within factor of 3 of experimental data; uncertainty quantified | |
| 9 | **Edge case handling** | Medium | Silent garbage on extreme inputs | Warnings on most extreme inputs; some undetected failures | Comprehensive parameter validation with physics-informed warnings | |
| 10 | **Resistivity model** | Medium | Constant resistivity only | Spitzer with fixed Coulomb log | Spitzer with dynamic Coulomb log + anomalous with justified threshold + impurity correction | |
| 11 | **Radiation model** | Medium | No radiation or wrong scaling | Bremsstrahlung with correct scaling | Bremsstrahlung + line radiation + opacity estimate + documented Gaunt factor | |
| 12 | **Multi-backend consistency** | Medium | Backends produce contradictory results | Qualitative agreement; quantitative differences understood | Quantitative agreement within stated numerical error bars | |
| 13 | **GUI professional quality** | Low | Missing units, confusing layout, broken interactions | Functional with correct units; some layout issues | Publication-quality displays with engineering notation, correct axis labels, exportable data | |
| 14 | **Surrogate model utility** | Low | Surrogate worse than random; no uncertainty | Surrogate within 2x of physics; basic confidence estimate | Calibrated uncertainty bounds; clear OOD detection; useful for parameter exploration | |
| 15 | **Performance** | Low | 64x128 run takes > 1 hour | 64x128 completes in < 10 minutes | 128x256 completes in < 10 minutes on standard hardware | |

### Scoring Thresholds

| Overall Score | Recommendation |
|---------------|----------------|
| Average >= 4.0, no Critical < 3 | **Recommend for use** with minor caveats |
| Average >= 3.0, no Critical < 2 | **Conditional recommendation** pending identified fixes |
| Average >= 2.5, one Critical < 3 | **Do not recommend** until Critical issues resolved |
| Average < 2.5 or any Critical = 1 | **Reject** -- fundamental issues with physics fidelity or numerical integrity |

---

## 5. "Gotcha" Tests: Traps for Common Physics Simulation Bugs

These tests are specifically designed to catch bugs that Dr. Vasquez has seen in other codes throughout her career.

### Gotcha 1: The Silent Unit Mismatch

**Test**: Set Te = 1 eV manually in the initial conditions, then read back Te from the simulation output.
**Expected**: Te should be reported as 11,604.5 K (1 eV = 11604.5 K). If it is reported as 1.0, the code is treating Te as eV internally but displaying it as K without conversion. If it is reported as 1.16e4, it is correct.
**Common bug**: Mixed eV/K units in different modules, with no conversion at the interfaces.

### Gotcha 2: The Off-by-mu_0 Magnetic Pressure

**Test**: Initialize a uniform plasma with B = 1 T, rho = 1 kg/m^3, v = 0, p = 1 Pa. Compute B^2/(2*mu_0).
**Expected**: Magnetic pressure = 1 / (2 * 4*pi*1e-7) = 3.979e5 Pa. The total pressure should be p_gas + p_mag = 1 + 3.979e5 Pa.
**Common bug**: Missing mu_0 factor gives magnetic pressure = 0.5 Pa (off by 10^6). The simulation runs but the magnetic field does nothing. Another variant: using mu_0 in CGS (= 1) while the rest of the code is in SI.

### Gotcha 3: The Wrong Sign on the Hall Term

**Test**: Initialize a Harris current sheet and enable the Hall term. The magnetic flux should exhibit quadrupole reconnection (out-of-plane B perturbation with 4 lobes of alternating sign).
**Expected**: Quadrupole B_y structure around the X-point.
**Common bug**: Wrong sign on J x B / (n_e * e) gives anti-reconnection (flux pile-up instead of reconnection). The simulation may look plausible at first glance but produces exactly the wrong physics.

### Gotcha 4: The Density Floor Contamination

**Test**: Run a strong shock problem (Mach > 10) and check the total mass in the domain at t=0 and t_final.
**Expected**: Total mass exactly conserved (finite volume should guarantee this).
**Common bug**: A density floor (rho_min = 1e-10) silently adds mass to near-vacuum regions. Over many timesteps, total mass can increase by 10% or more. The code "works" but the conservation is broken by the floor.

### Gotcha 5: The Axial Singularity (r=0)

**Test**: In cylindrical coordinates, examine the solution at the first radial cell (r closest to 0).
**Expected**: B_r = 0 at r = 0 (antisymmetry), B_theta = 0 at r = 0 (antisymmetry), dB_z/dr = 0 at r = 0 (symmetry). No 1/r singularity in any computed quantity.
**Common bug**: Division by r in geometric source terms without L'Hopital correction at r=0. This produces Inf or NaN at the axis cell, which then propagates outward. A less severe version: the code clamps r to r_min > 0, which introduces O(r_min) error in the axis solution.

### Gotcha 6: The CFL Violation During Pinch

**Test**: Run PF-1000 and monitor the actual CFL number (max signal speed * dt / dx) at every timestep.
**Expected**: CFL should always remain below the specified limit (e.g., 0.4). During pinch, the fast magnetosonic speed increases dramatically, so dt must decrease.
**Common bug**: dt is computed from the initial conditions and never updated (fixed timestep). As the pinch compresses and B increases, the actual CFL exceeds 1.0, and the scheme becomes unstable. The code either crashes or produces oscillating garbage without a clear error message.

### Gotcha 7: The Odd-Even Decoupling

**Test**: Run a 1D shock tube on a staggered grid. Plot the solution at every cell (not just every other cell).
**Expected**: Smooth solution with no oscillation at the grid scale.
**Common bug**: Central difference operators on a collocated grid produce decoupled even and odd modes. The solution looks smooth when plotted at every other cell but shows sawtooth oscillations at every cell. This is masked by most plotting routines that default to interpolated rendering.

### Gotcha 8: The Gaunt Factor of Doom

**Test**: Compute bremsstrahlung power at Te = 10 eV and Te = 10 keV with the code's Gaunt factor.
**Expected**: The Gaunt factor g_ff should be order unity (typically 1.0-1.5) for both temperatures. At low Te, quantum corrections increase it slightly; at high Te, relativistic corrections decrease it slightly.
**Common bug**: Gaunt factor formula produces g_ff > 10 or g_ff < 0 at extreme temperatures, leading to catastrophically wrong radiation losses. The code may use a fit valid only in a narrow range.

### Gotcha 9: The Invisible Boundary Energy Leak

**Test**: Run a periodic box with a smooth MHD wave (e.g., circularly polarized Alfven wave). Monitor total energy vs. time.
**Expected**: Energy exactly conserved (periodic BCs, smooth solution, no shocks).
**Common bug**: Boundary condition implementation has a one-cell offset error, causing ghost cell values to be incorrect. This injects small energy perturbations at each BC application. The total energy drifts linearly in time, proportional to the boundary flux error.

### Gotcha 10: The Frozen Coulomb Logarithm

**Test**: Run a simulation where Te varies from 10 eV to 10 keV across the domain. Print the Coulomb logarithm at both extremes.
**Expected**: ln(Lambda) at 10 eV ~ 7-10; ln(Lambda) at 10 keV ~ 15-20. The ratio matters for resistivity scaling.
**Common bug**: ln(Lambda) initialized once at t=0 from initial conditions and never updated. All subsequent resistivity calculations use a single constant value. This produces wrong resistivity scaling across the domain and over time.

---

## 6. Comparison Matrix: Published DPF Data vs. Simulation

### 6.1 PF-1000 (IPPLM Warsaw)

| Metric | Experimental Value | Source | Simulated Value | Agreement | Notes |
|--------|-------------------|--------|-----------------|-----------|-------|
| Peak current (27 kV, 3.5 Torr D2) | 1.8 +/- 0.2 MA | Scholz et al. 2004, J. Phys. D | | | |
| Pinch time | 5.5 +/- 1.0 us | Scholz et al. 2004 | | | |
| Neutron yield (27 kV) | (0.5-2) x 10^10 | Scholz et al. 2006, Plasma Sources Sci. Technol. | | | |
| Current dip magnitude | 10-30% of I_peak | Lee & Serban 1996, IEEE Trans. Plasma Sci. | | | |
| Pinch radius | 3-8 mm | Scholz et al. 2004 (X-ray imaging) | | | |
| Pinch length | 50-100 mm | Sadowski et al. 2004, Nukleonika | | | |
| Electron temperature (pinch) | 0.5-2.0 keV | Czaus et al. 2016, Phys. Plasmas | | | |
| Ion temperature (pinch) | 1-5 keV (anisotropic) | Scholz et al. 2006 (neutron spectrum) | | | |
| Magnetic field at pinch | 20-80 T | Estimated from I_pinch / (2*pi*r_pinch) | | | |
| Stored energy (27 kV) | 486 kJ | 0.5 * 1.332e-3 * (27e3)^2 | | | Should be exact |
| Quarter-period | 0.22 us | pi/2 * sqrt(15e-9 * 1.332e-3) | | | Should be exact |

### 6.2 NX2 (NIE Singapore)

| Metric | Experimental Value | Source | Simulated Value | Agreement | Notes |
|--------|-------------------|--------|-----------------|-----------|-------|
| Peak current (12 kV) | 350-420 kA | Lee & Serban 1996 | | | |
| Pinch time | 0.8-1.5 us | Lee et al. 2009, J. Fusion Energy | | | |
| Neutron yield (12 kV, D2) | (0.5-5) x 10^6 | Lee et al. 2009 | | | |
| Stored energy | 64.8 J | 0.5 * 0.9e-6 * (12e3)^2 | | | |

### 6.3 Lee Model Benchmarks

| Device | Lee Model I_peak | Lee Model Y_n | Lee Model t_pinch | Source |
|--------|------------------|---------------|--------------------| -------|
| PF-1000 (27 kV) | 1.87 MA | 1.1 x 10^10 | 5.8 us | Lee (2014), IAEA |
| NX2 (12 kV) | 393 kA | 1.0 x 10^6 | 1.0 us | Lee & Serban 1996 |
| LLNL-DPF (24 kV) | ~1.0 MA | ~10^9 | ~3 us | Estimated from scaling |

### 6.4 Scaling Law Verification

| Scaling Relation | Theoretical Exponent | Measured Exponent | Reference | Status |
|------------------|---------------------|-------------------|-----------|--------|
| Y_n vs. I_pinch | 4.0 | | Lee & Serban 1996 | |
| I_peak vs. V0 | 1.0 | | Basic circuit theory | |
| t_pinch vs. sqrt(LC) | 1.0 | | Basic circuit theory | |
| Y_n vs. E_stored | 2.0 | | Lee scaling (Y ~ I^4, I ~ V, E ~ V^2) | |
| Pinch Te vs. I_pinch^2 | 2.0 | | Bennett equilibrium | |
| r_pinch vs. n_line^(-1/2) | -0.5 | | Bennett equilibrium | |

---

## 7. Uncertainty Quantification Requirements

Dr. Vasquez insists that every simulation output include an uncertainty estimate. Sources of uncertainty in a DPF simulation include:

### 7.1 Numerical Uncertainty

| Source | How to Quantify | Acceptable Level |
|--------|----------------|------------------|
| Spatial discretization | Richardson extrapolation from 3+ grid levels | Reported convergence order, extrapolated error estimate |
| Temporal discretization | CFL sensitivity study (run at CFL = 0.2, 0.4, 0.8) | Results insensitive to CFL within 5% |
| Riemann solver | Compare HLL vs. HLLD (if available) | Results agree within 10% |
| Reconstruction | Compare WENO5 vs. PLM (if available) | Quantify diffusion difference |
| Divergence cleaning | Monitor div(B) and psi energy | div(B)*dx/|B| < 0.01 |

### 7.2 Physics Model Uncertainty

| Source | How to Quantify | Acceptable Level |
|--------|----------------|------------------|
| Resistivity model (Spitzer vs. Lee-More vs. anomalous) | Run with different models, report spread | Spread documented in output |
| Radiation model (bremsstrahlung only vs. +line) | Compare with/without line radiation | Effect on Te quantified |
| Gaunt factor | Compare fixed g_ff = 1.2 vs. temperature-dependent | Effect on radiated power quantified |
| Equation of state (ideal vs. tabulated) | State assumption; note where ideal EOS breaks | T > 100 eV ideal is acceptable; T < 10 eV needs neutral correction |
| Hall term (included vs. excluded) | Run both; report difference | Quantify effect on reconnection and pinch structure |
| Nernst effect | Run both; report difference | Quantify effect on Te profile |

### 7.3 Input Parameter Uncertainty

| Parameter | Typical Experimental Uncertainty | How to Propagate |
|-----------|----------------------------------|------------------|
| V0 (charging voltage) | +/- 2% | Run at V0 +/- 2%, report output spread |
| C (capacitance) | +/- 5% | Include in parameter sweep |
| L0 (external inductance) | +/- 20% (hard to measure) | This is the dominant input uncertainty for most DPF experiments |
| R0 (external resistance) | +/- 30% | Includes cable, switch, and contact resistance |
| Fill pressure | +/- 10% | Pressure gauge calibration uncertainty |
| Electrode geometry | +/- 0.5 mm | Manufacturing tolerance |
| Gas composition | Variable | D2 purity, air leaks, electrode erosion products |

---

## 8. Appendix: Key References

The following publications form Dr. Vasquez's primary comparison dataset. She expects the simulation team to be familiar with these.

1. Lee, S. & Serban, A. (1996). "Dimensions and lifetime of the plasma focus pinch." IEEE Trans. Plasma Sci., 24(3), 1101-1105. -- Foundational scaling laws for DPF: Y_n ~ I_pinch^4, pinch dimensions.

2. Scholz, M. et al. (2004). "Experimental studies of neutron emission from the PF-1000 facility." J. Phys. D: Appl. Phys., 37, 2500-2506. -- PF-1000 experimental data: current waveforms, neutron yields, pinch timing.

3. Scholz, M. et al. (2006). "Status of the PF-1000 device." Plasma Sources Sci. Technol., 15, S162. -- Updated PF-1000 performance data.

4. Lee, S. (2008). "Current and neutron scaling for megajoule plasma focus machines." Plasma Phys. Control. Fusion, 50, 105005. -- Generalized scaling laws across multiple DPF devices.

5. Lee, S. (2014). "Plasma Focus Radiative Model." IAEA report. -- Lee model code documentation and validation dataset.

6. Bosch, H.-S. & Hale, G. M. (1992). "Improved formulas for fusion cross-sections and thermal reactivities." Nucl. Fusion, 32(4), 611-631. -- Standard D-D and D-T fusion reactivity parameterizations.

7. Braginskii, S. I. (1965). "Transport processes in a plasma." Reviews of Plasma Physics, Vol. 1, p. 205. -- Anisotropic transport coefficients.

8. Spitzer, L. & Harm, R. (1953). "Transport phenomena in a completely ionized gas." Phys. Rev., 89(5), 977. -- Classical resistivity.

9. Sadowski, M. J. et al. (2004). "Research on dense magnetized plasmas in the PF-1000 facility." Nukleonika, 49, S49-S54. -- Pinch geometry and dynamics.

10. Bennett, W. H. (1934). "Magnetically Self-Focussing Streams." Phys. Rev., 45(12), 890-897. -- Bennett equilibrium: I^2 = (8*pi/mu_0) * N_line * k_B * (Te + Ti).

---

## 9. Final Verdict Template

After completing all scenarios, Dr. Vasquez fills in the following:

```
EVALUATOR: Dr. Maria Vasquez
DATE: ___________
DPF UNIFIED VERSION: ___________
BACKENDS TESTED: Python [ ] Athena++ [ ] AthenaK [ ]

SUMMARY SCORES:
  Conservation:         ___/5
  Convergence:          ___/5
  Experimental agree.:  ___/5
  Numerical robustness: ___/5
  Physics transparency: ___/5
  Unit consistency:     ___/5
  Div(B) quality:       ___/5
  Diagnostic fidelity:  ___/5
  Edge case handling:   ___/5
  Resistivity model:    ___/5
  Radiation model:      ___/5
  Multi-backend:        ___/5
  GUI quality:          ___/5
  Surrogate utility:    ___/5
  Performance:          ___/5

  OVERALL AVERAGE: ___/5
  CRITICAL MINIMUM: ___/5

RECOMMENDATION: [ ] Recommend  [ ] Conditional  [ ] Do Not Recommend  [ ] Reject

CRITICAL ISSUES (must be fixed before deployment):
  1. ___
  2. ___
  3. ___

HIGH-PRIORITY ISSUES (should be fixed within 3 months):
  1. ___
  2. ___
  3. ___

PHYSICS CONCERNS (limitations to document for users):
  1. ___
  2. ___
  3. ___

COMMENTS:
_______________________________________________
_______________________________________________
_______________________________________________
```

---

*This enhanced UAT plan was designed to simulate the evaluation methodology of an experienced pulsed-power plasma physicist. The test scenarios are ordered from simple (isolated circuit) to complex (full multi-physics DPF), with each scenario building trust incrementally. The "gotcha" tests are drawn from real bugs that have been found in production MHD codes. The comparison matrix uses only published, peer-reviewed experimental data. No test scenario requires physics beyond the stated capabilities of the DPF Unified simulator.*
