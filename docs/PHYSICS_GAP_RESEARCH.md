# Deep Research: 5 Priority Physics Gaps for DPF-Unified

**Date**: 2026-03-25
**Purpose**: Sprint-ready research for MHD simulator physics upgrades
**Scope**: Post-pinch expansion, anomalous resistivity, beam-target Yn, radiation, Bell-Plesset RTI

---

## Gap 1: Post-Pinch Column Expansion Model

### Problem Statement

After stagnation the pinch column expands, producing dL/dt that drives the current dip. Our model gives either no dip (dL_dt=0 when snowplow deactivates, line 152 of `circuit_coupling.py`) or a 90% dip (uncalibrated density-weighted feedback). Experimental dip is ~60% for PF-1000 at 27 kV.

### Current Codebase State

In `src/dpf/engine/circuit_coupling.py:139-153`, when the snowplow is no longer active (post-pinch), the code either:
1. Holds inductance constant (`dL_dt = 0.0`) -- produces NO current dip
2. Uses `density_weighted` coupling mode with `feedback.dLp_dt` -- produces an UNCALIBRATED dip (90%)

The problem is that option (2) passes through the raw MHD expansion rate, which is too fast because MHD instabilities (m=0 sausage) disrupt the column faster than thermal expansion, and the effective inductance change depends on the pinch column geometry in a way that the density-weighted integral does not properly capture during disruption.

### State of the Art

**Lee & Saw (2014), Phys. Plasmas 21, 072501** -- The Lee model Phase 4c (column expansion):
- After pinch stagnation, the pinch column expands as a slug with initial radius r_min
- Expansion driven by: (a) residual kinetic energy from reflected shock, (b) thermal pressure, (c) magnetic diffusion
- Lee's expansion velocity: `v_exp = v_piston_reflected = (gamma-1)/(gamma+1) * v_implosion`
- For strong shock (gamma=5/3): `v_exp ~ v_imp / 4`
- Column radius evolves: `r(t) = r_min + v_exp * (t - t_pinch)`
- Inductance: `L_p(t) = (mu_0/2pi) * z_f * ln(b/r(t))`
- `dL/dt = -(mu_0/2pi) * z_f * v_exp / r(t)` (NEGATIVE, L decreases as column expands)

**Critical insight**: dL/dt is negative during expansion (L decreases as r grows, since L ~ ln(b/r) and r is increasing). The back-EMF `V = I * dL/dt` is negative, which means the circuit equation `L*dI/dt + I*dL/dt + ... = 0` drives a current DIP. The depth of the dip depends on how fast r expands.

**Lee model calibration**: Lee uses a parameter `fm_pinch` (mass fraction during expansion) that controls how much mass participates in the expansion, and thereby the deceleration/expansion rate. Typical fm_pinch ~ 0.1-0.2.

**Goyon et al. (2025), Phys. Plasmas 32, 033105** -- MJOLNIR MA-class DPF:
- Expansion timescale: `tau_exp ~ 31.5 * R_cm^2 * sqrt(P_Torr) / (CR * I_MA)` [ns]
- For PF-1000 (R=8.3 cm, P=3.5 Torr, CR=10, I=1.8 MA): tau_exp ~ 27 ns
- m=0 timescale: `tau_m0 ~ 31.0 * R_cm^2 * sqrt(P_Torr) / (CR * I_MA)` [ns]
- tau_m0 / tau_exp ~ 1.0 -- m=0 disrupts on the SAME timescale as expansion
- This means the column doesn't expand smoothly; it is disrupted by m=0, which creates plasmoids

**What controls expansion velocity?**:
1. **Thermal expansion**: `v_th ~ c_s_post_shock = sqrt(gamma * k_B * T_stag / m_i)`. For T_stag=1 keV D2: c_s ~ 2.8e5 m/s. This is the sound speed of the stagnated plasma.
2. **Magnetic pressure**: `v_A ~ B_theta / sqrt(mu_0 * rho)`. At pinch: B~50T, rho~1e-3 kg/m^3 -> v_A ~ 5e4 m/s. Slower than thermal.
3. **m=0 instability**: Growth time tau_m0 ~ 10-50 ns. Disruption happens faster than smooth expansion. Creates hot-spots and plasmoid ejection that add irregular dL/dt.

**MRT instability timescale vs thermal expansion timescale**: From Goyon (2025), the ratio tau_m0/tau_exp ~ 1, meaning both processes compete. For high-current devices (I>1 MA), m=0 disruption dominates because Alfven speed scales with I. For low-current devices (I<100 kA), thermal expansion is faster.

**Dip shape universality**: The dip shape is NOT universal. It depends on:
- Fill pressure (higher P -> deeper dip, more mass to expand against)
- Voltage (higher V0 -> higher v_imp -> faster expansion -> shallower dip in time units)
- Geometry (anode radius, cathode radius, anode length)
- Device class: MA-class devices show multiple mini-dips from successive m=0 events; kA-class show a single smooth dip

**Calibration data -- PF-1000 dip shapes**:
- Scholz 2006 (27 kV, 3.5 Torr): Current dip from ~1.8 MA to ~0.7 MA (61% dip), duration ~500 ns, single dip
- Gribkov 2007 (16 kV, 1.2 Torr): Current dip from ~0.5 MA to ~0.25 MA (50% dip), duration ~400 ns, broader dip
- Scholz 2006 at 23 kV: ~55% dip
- The dip depth scales roughly as sqrt(V0) because higher voltage gives higher v_imp and thus higher v_exp

### How Production Codes Handle It

**MACH2 (NRL)**: Uses a 2D MHD code that naturally captures column expansion through the momentum equation. No special post-pinch model needed -- the MHD evolution handles it. However, MACH2 uses anomalous resistivity to dissipate the pinch, and the expansion rate depends sensitively on the resistivity model (Faerder 2024).

**GORGON (Imperial College)**: 3D MHD code. Post-pinch expansion is captured self-consistently. Column disruption by m=0/m=1 instabilities is resolved on the grid. No lumped model needed.

**Lee model code**: Uses the Phase 4c expansion with fm_pinch parameter. This is the approach most applicable to our lumped circuit model.

**FLASH (Rochester)**: Resolves instabilities on the grid. No expansion model. But requires AMR to capture pinch neck at r_min ~ 1 mm.

### Governing Equations

The Lee Phase 4c expansion model for a cylindrical pinch column:

```
dr/dt = v_exp                                          (1)
dv_exp/dt = (p_thermal - B_theta^2/(2*mu_0)) / (rho * r)  (2) -- net pressure drives expansion
L_p(t) = (mu_0/(2*pi)) * z_f * ln(b/r(t))            (3)
dL_p/dt = -(mu_0/(2*pi)) * z_f * v_exp / r(t)        (4)
V_back = I * dL_p/dt                                  (5)
```

where:
- r(t) is the column radius (starts at r_min, expands outward)
- v_exp is the expansion velocity
- p_thermal is the post-shock thermal pressure: p = n*k_B*T_stag (both species)
- B_theta = mu_0 * I / (2*pi*r) -- azimuthal field at column boundary
- b is the cathode radius (outer electrode)
- z_f is the effective pinch length (axial extent)

Initial conditions at t = t_pinch:
- r(t_pinch) = r_min (minimum pinch radius from snowplow)
- v_exp(t_pinch) = v_reflected = ((gamma-1)/(gamma+1)) * v_imp (reflected shock piston)
- T_stag from Rankine-Hugoniot: T ~ 21 * I_MA^2 / (R_cm^2 * P_Torr) [keV]

### Implementation Complexity for MLX (float32)

**LOW complexity.** This is a simple ODE system (2 equations for r, v_exp) solved at the circuit sub-cycle level. No spatial discretization needed.

Float32 concerns: None significant. r varies from ~1 mm to ~10 mm; inductance from ~1 nH to ~100 nH. All well within float32 range. The ln() function is smooth and well-conditioned.

### Risks and Gotchas

1. **Sign convention**: dL/dt is NEGATIVE during expansion. The code in `circuit_coupling.py` currently expects positive dL_dt in some paths. Must trace all sign conventions carefully.
2. **v_exp calibration**: The reflected shock velocity formula gives v_exp ~ v_imp/4, but the effective expansion rate depends on how much mass participates (fm_pinch). This is a calibration parameter.
3. **Expansion termination**: The column can't expand past the cathode radius b. Need r(t) <= b check.
4. **Multiple disruptions**: MA-class devices show multiple m=0 events. A simple monotonic expansion doesn't capture this. For PF-1000 at 27 kV (single dip), the monotonic model is adequate.
5. **Interaction with density-weighted Lp**: If MHD is active during expansion, the density-weighted Lp already captures some expansion. Must not double-count.

### Implementation Recommendation

**Approach**: Add Lee Phase 4c expansion ODE to the snowplow model. When snowplow reaches pinch phase, switch to expansion mode with the ODE system above. The expansion provides dL/dt to the circuit solver.

**Files to modify**:
- `src/dpf/fluid/snowplow.py` -- Add expansion state (r_column, v_expand) and step_expansion() method (~60 LOC)
- `src/dpf/engine/circuit_coupling.py` lines 139-153 -- Replace the `dL_dt=0.0` with snowplow expansion dL/dt (~15 LOC)
- `src/dpf/config.py` -- Add `expansion_velocity_factor` (default 0.25) and `fm_pinch` (default 0.15) to SnowplowConfig (~5 LOC)
- `tests/test_post_pinch_expansion.py` -- New test file (~120 LOC)

**LOC estimate**: ~200 LOC total (80 snowplow, 20 coupling, 10 config, 120 tests)

**Data needed**: PF-1000 current waveforms from Scholz 2006 and Gribkov 2007 for calibration. Already available in `src/dpf/validation/experimental_comparison.py`.

**No new research data needed** -- can be coded from existing literature (Lee & Saw 2014, Goyon 2025).

---

## Gap 2: Anomalous Resistivity (Drift-Velocity Model)

### Problem Statement

Current implementation in `src/dpf/turbulence/anomalous.py` uses three threshold models (ion-acoustic, LHDI, Buneman classic) with a fixed alpha parameter. The resistivity magnitude is `eta_anom = alpha * m_e * omega_pe / (n_e * e^2)`, applied as a step function above threshold. Faerder 2024 argues that drift-velocity resistivity with a smooth transition is the dominant mechanism for DPF.

### Current Codebase State

The anomalous resistivity module (`anomalous.py`) implements:
- `eta_anom = alpha * m_e * omega_pe / (n_e * e^2)` with alpha=0.05 (fixed)
- Threshold: `v_d > c_s` (ion-acoustic), `v_d > (m_e/m_i)^{1/4} * v_ti` (LHDI), or `v_d > v_te` (Buneman)
- Step-function activation: zero below threshold, full value above
- No spatial smoothing or ramp-up

### State of the Art

**Rososhek, Seyler, Lavine & Hammer (2026), Phys. Plasmas** -- "The Hall Term and Anomalous Resistivity Effects in Neon Gas-Puff Z-Pinches":
- Compare Spitzer, anomalous (Sagdeev-Galeev), and turbulent-cascade resistivity models in 2D Hall-MHD
- Find that anomalous resistivity with v_d > c_s threshold produces most physical results
- Hall term + anomalous resistivity together are essential for reproducing experimental B-field structure
- Key finding: the alpha parameter should scale with v_d/c_s (not be constant)

**Faerder et al. (2024), Phys. Plasmas** -- Drift-velocity resistivity for DPF:
The drift-velocity model is based on current-driven instabilities where the effective collision frequency scales with the drift velocity ratio:

```
eta_dv = (m_e / (n_e * e^2)) * nu_eff                    (1)
nu_eff = omega_pi * (v_d / v_ti)^2  for v_d > v_ti      (2)
       = 0                           for v_d <= v_ti
```

where:
- omega_pi = sqrt(n_i * e^2 / (epsilon_0 * m_i)) is the ion plasma frequency
- v_d = |J| / (n_e * e) is the electron drift velocity
- v_ti = sqrt(k_B * T_i / m_i) is the ion thermal speed

This gives:
```
eta_dv = (m_e * omega_pi) / (n_e * e^2) * (v_d / v_ti)^2    (3)
```

The key difference from our current model:
- **Smooth quadratic scaling** with (v_d/v_ti)^2 instead of step function
- Threshold is v_ti (ion thermal speed), not c_s (ion sound speed)
- No free alpha parameter -- the (v_d/v_ti)^2 factor provides natural self-regulation
- At high drift (v_d >> v_ti), this can be orders of magnitude larger than the Sagdeev model

**Comparison of anomalous resistivity models at DPF conditions**:

| Model | Threshold | Magnitude (at v_d=10*v_ti, ne=1e24) | Temperature dependence |
|-------|-----------|--------------------------------------|----------------------|
| Spitzer | None | eta ~ 1e-5 * Z * T_e^{-3/2} ~ 1e-7 Ohm*m (1 keV) | Strong: T^{-3/2} |
| Ion-acoustic (current) | v_d > c_s | eta ~ alpha * m_e * omega_pe / (n_e * e^2) ~ 5e-5 Ohm*m | Weak (via c_s threshold) |
| LHDI (current) | v_d > (m_e/m_i)^{1/4} * v_ti | Same magnitude as above, lower threshold | Weak (via v_ti threshold) |
| Buneman | v_d > v_te | Same magnitude, much higher threshold | Through v_te |
| Drift-velocity (Faerder) | v_d > v_ti | eta ~ (m_e * omega_pi / n_e e^2) * (v_d/v_ti)^2 ~ 1e-3 Ohm*m | Through v_ti ~ sqrt(T_i) |
| Spitzer + effective Z | None | eta_Spitzer * Z_eff (Z_eff >> Z for turbulent scattering) | T^{-3/2} |

**Key observation**: The drift-velocity model gives MUCH higher resistivity at high drift speeds because of the (v_d/v_ti)^2 scaling. At v_d = 10*v_ti, it is ~100x the Sagdeev model. This is why it may be needed for DPF where v_d/v_ti can reach 10-100 during pinch disruption.

### How Z-Pinch Codes Handle It

**MACH2 (NRL)**: Uses Sagdeev formula with alpha=0.01-0.1, threshold v_d > c_s. Applied pointwise (cell-by-cell). Faerder 2024 upgraded MACH2 to use the drift-velocity model and showed better agreement with experimental x-ray images.

**HYDRA (LLNL)**: Uses a combination of Spitzer + anomalous. The anomalous part uses a saturation model where nu_anom = min(nu_ii, omega_pi * (v_d/v_ti)^2). The ion-ion collision frequency nu_ii provides an upper bound.

**GORGON (Imperial)**: Uses Spitzer resistivity with a floor. No explicit anomalous model, but the numerical resistivity of the HLL solver provides implicit dissipation at the grid scale.

**SpECTRE (SXS)**: No anomalous resistivity (ideal MHD only in current release). Resistive MHD is under development.

**Athena++ (Princeton)**: The `field_diffusion` module supports user-defined resistivity functions via `CalcMagDiffCoeff_` callback. No built-in anomalous model, but the hook is there.

### Governing Equations

The complete drift-velocity anomalous resistivity:

```
v_d = |J| / (n_e * e)                                    [m/s]
v_ti = sqrt(k_B * T_i / m_i)                              [m/s]
omega_pi = sqrt(n_i * e^2 / (epsilon_0 * m_i))            [rad/s]

if v_d > v_ti:
    nu_eff = omega_pi * (v_d / v_ti)^2                    [1/s]
    nu_eff = min(nu_eff, omega_pi)   # saturation cap     [1/s]
    eta_anom = m_e * nu_eff / (n_e * e^2)                  [Ohm*m]
else:
    eta_anom = 0

eta_total = eta_Spitzer + eta_anom                         [Ohm*m]
```

The saturation cap `nu_eff <= omega_pi` prevents unphysically large resistivity when v_d >> v_ti. This corresponds to the Bohm diffusion limit.

For the MLX solver, this enters as a source term in the induction equation:
```
dB/dt = ... + curl(eta * J)
```
where `J = curl(B) / mu_0` and eta includes both Spitzer and anomalous contributions.

### Implementation Complexity for MLX (float32)

**LOW-MEDIUM complexity.** The computation is pointwise (no spatial derivatives needed beyond what already exists for J = curl(B)/mu_0). Main concern is float32 precision:

- `omega_pi` at ne=1e24: omega_pi ~ 4.2e10 rad/s -- fine for float32
- `v_d/v_ti` ratio: can be 0.1 to 100 -- fine for float32
- `eta_anom` values: 1e-7 to 1e-3 Ohm*m -- fine for float32
- **Gotcha**: `(v_d/v_ti)^2` can overflow float32 if v_d/v_ti > 1e19 (impossible physically, but worth clamping)

### Risks and Gotchas

1. **CFL constraint**: The resistive CFL is `dt < dx^2 * mu_0 / (2*eta)`. For eta_anom=1e-3 and dx=1mm: dt < 6.3e-10 s. This is ~100x smaller than the MHD CFL. **Sub-cycling is essential** (already implemented in the codebase for Spitzer resistivity).
2. **Stability at high drift**: The (v_d/v_ti)^2 scaling can produce extremely large eta in a single cell, destabilizing the solver. The saturation cap at omega_pi is critical.
3. **Temperature coupling**: v_ti depends on T_i, which changes due to ohmic heating (J^2*eta). This creates a feedback loop: high eta -> more heating -> higher T_i -> lower v_d/v_ti -> lower eta. This is self-regulating but can cause oscillations without implicit coupling.
4. **Step function vs smooth**: A step function at v_d = v_ti creates a discontinuity in the resistivity field. Smooth ramp-up (e.g., using tanh) over a width of ~0.1*v_ti improves stability.

### Implementation Recommendation

**Approach**: Add `drift_velocity` as a new `threshold_model` option in `anomalous_resistivity_field()` and `anomalous_resistivity_scalar()`. Keep the existing models for backward compatibility.

**Files to modify**:
- `src/dpf/turbulence/anomalous.py` -- Add `drift_velocity_resistivity()` function and integrate into `anomalous_resistivity_field()` dispatch (~80 LOC)
- `src/dpf/metal/mlx_sources.py` -- Add drift-velocity eta computation in the MLX source term path (~40 LOC)
- `src/dpf/config.py` -- Add `"drift_velocity"` to threshold_model enum, add `eta_saturation_cap` parameter (~5 LOC)
- `tests/test_drift_velocity_resistivity.py` -- New test file (~100 LOC)

**LOC estimate**: ~225 LOC total

**No new research data needed** -- equations from Faerder 2024 and HYDRA saturation model are fully specified.

---

## Gap 3: Beam-Target Neutron Yield from MHD Fields

### Problem Statement

The beam-target neutron yield calculation in `diagnostics/beam_target.py` and `yield_tracker.py` requires `V_pinch` (pinch voltage), which drives beam energy via `E_beam = e * V_pinch`. Currently, V_pinch is computed in `state_management.py:197` as:

```python
_V_pinch = abs(self._coupling.current) * (max(_dL_mhd, _dL_sp) + _sp_R)
```

After the snowplow deactivates, `_dL_sp = 0` and `_sp_R = 0`. If the MHD coupling is not in density-weighted mode, `_dL_mhd = 0` too. Result: `V_pinch = 0`, and beam-target yield drops to zero.

This is the same root cause as Gap 1 -- the post-pinch expansion model must provide dL/dt to feed V_pinch.

### State of the Art

**Lee model beam-target mechanism** (Lee 2014, J. Fusion Energy 33:319):
- V_pinch arises from the inductive voltage across the expanding pinch: `V_pinch = I * dL/dt`
- This accelerates deuterons to `E_beam = e * V_pinch` (typically 50-200 keV)
- Beam deuterons traverse the dense pinch column (n_target ~ 1e24-1e26 m^-3, L ~ 5-20 mm)
- Yield rate: `dY/dt = f_beam * (I/e) * n_target * sigma_DD(E_cm) * L_target`
- f_beam ~ 0.14 (fraction of current converted to beam)
- V_pinch from Lee model: `V_pinch = (mu_0/(2*pi)) * z_f * v_A * I / r_min`
- At PF-1000 conditions: V_pinch ~ 20-50 kV, E_beam ~ 20-50 keV

**How to extract V_pinch from MHD state**:

Three approaches, in order of physical fidelity:

1. **Inductive voltage (current approach + Gap 1 fix)**: `V_pinch = I * dL/dt` from the expansion model. This is the Lee model approach and is most consistent with the circuit coupling.

2. **E x B drift velocity**: Extract the radial electric field from Ohm's law and compute the drift velocity at the pinch boundary:
   ```
   E_r = -v_z * B_theta + eta * J_z        (simplified Ohm's law in cylindrical coords)
   V_pinch ~ integral(E_r * dr) over pinch radius
   ```
   This requires spatially resolved fields and is more physical but harder to implement.

3. **Ion velocity at pinch boundary**: Extract `v_r` from the MHD momentum equation at the pinch radius. The radial velocity at the column boundary is the expansion velocity, and the associated electric field gives:
   ```
   V_pinch ~ v_r * B_theta * z_f
   ```
   where z_f is the effective pinch length.

**Schmidt 2017 argument about MHD limitations**:
Schmidt (2017, IEEE Trans. Plasma Sci.) argues that beam-target yield is fundamentally kinetic -- the beam deuterons are non-thermal (50-200 keV while bulk is 1-10 keV) and their production requires kinetic effects (anomalous resistive acceleration, m=0 disruption). MHD cannot capture:
- The beam-target energy spectrum (non-Maxwellian)
- The spatial localization of beam production (at m=0 necks)
- The beam transit through the target column (requires kinetic transport)

**However**: The Lee model successfully predicts beam-target yield to within a factor of 2-3 using V_pinch from the circuit. This is because:
- V_pinch provides the correct ORDER OF MAGNITUDE of beam energy
- sigma_DD is exponentially sensitive to E_beam at low energies but only weakly dependent above 20 keV
- The spatial averaging (f_beam * I_pinch as beam flux, n_target * L as target) captures the gross physics

**Thermonuclear vs beam-target fraction**:

| Voltage | E_beam | T_stag | Y_bt/Y_thermo | Dominant mechanism |
|---------|--------|--------|---------------|-------------------|
| <15 kV  | <15 keV | <1 keV | ~1-2 | Mixed |
| 15-30 kV | 15-50 keV | 1-5 keV | ~10-100 | Beam-target |
| 30-100 kV | 50-200 keV | 5-20 keV | ~5-20 | Beam-target |
| >100 kV | >200 keV | >20 keV | ~1-5 | Both significant |

At PF-1000 (27 kV), beam-target dominates by ~10-50x. Thermonuclear alone gives ~1e6 neutrons; total measured is ~1e8.

**V_pinch sensitivity**: The DD cross section has a Gamow factor: `sigma ~ exp(-B_G/sqrt(E_cm))` with B_G=31.4 keV^{1/2}. At E_cm=10 keV (V_pinch=20 kV): sigma ~ 0.28 mbarn. At E_cm=50 keV (V_pinch=100 kV): sigma ~ 16 mbarn. A factor of 2 error in V_pinch translates to ~3-10x error in yield at low energies, but only ~2x at high energies. The exponential sensitivity relaxes above ~30 keV.

### Governing Equations

The beam-target model (already implemented in `beam_target.py`, needs V_pinch fix):

```
E_beam = e * V_pinch                                       [J]
E_cm = E_beam / 2                                          [J]  (equal-mass DD)
sigma_DD(E_cm) = S(E_cm) / (E_cm * exp(B_G/sqrt(E_cm)))   [m^2]
dY/dt = f_beam * (I_pinch/e) * n_target * sigma_DD * L_target  [1/s]
Y_bt = integral(dY/dt * dt, t_pinch, t_end)                [count]
```

V_pinch from MHD fields (approach 3, extractable from state):
```
v_r_boundary = radial velocity at r = r_pinch (from MHD state)
B_theta_boundary = B_theta at r = r_pinch (from MHD state)
V_pinch = v_r_boundary * B_theta_boundary * z_eff          [V]
```

Alternatively, from the expansion model (Gap 1):
```
V_pinch = I * |dL/dt| = I * (mu_0/(2*pi)) * z_f * v_exp / r(t)  [V]
```

### Implementation Complexity for MLX (float32)

**LOW.** This is a diagnostic computation, not part of the MHD solve. V_pinch extraction requires reading MHD state arrays (already available) and computing a few scalar quantities. No float32 concerns.

### Risks and Gotchas

1. **V_pinch from MHD fields can be noisy**: v_r at the pinch boundary oscillates due to numerical noise. Temporal smoothing (running average over ~5 timesteps) is essential.
2. **Pinch boundary detection**: Defining "the pinch boundary" from MHD fields requires a density contour or gradient criterion. The density-peak method in `coupler.py` works for the axial direction but needs a radial analog.
3. **f_beam uncertainty**: f_beam=0.14 is empirical. It varies with device and operating conditions. For relative comparisons this is fine; for absolute Yn prediction, f_beam is the dominant uncertainty.
4. **Coupling to Gap 1**: If the expansion model (Gap 1) provides dL/dt, then V_pinch = I * |dL/dt| is automatically available. This is the simplest path.

### Implementation Recommendation

**Approach**: Two-pronged:
1. **Primary (quick win)**: Once Gap 1 is implemented, V_pinch automatically becomes nonzero post-pinch via `V_pinch = I * |dL/dt|` from the expansion ODE. This requires only ~10 LOC change in `state_management.py` to use the expansion dL/dt.
2. **Secondary (higher fidelity)**: Extract V_pinch from MHD fields (v_r * B_theta * z_eff) during the MHD-coupled phase. This provides a cross-check and works even when the snowplow model is inactive.

**Files to modify**:
- `src/dpf/engine/state_management.py` lines 192-197 -- Use expansion dL/dt for V_pinch (~10 LOC)
- `src/dpf/circuit/coupler.py` -- Add `extract_v_pinch(state, config)` method using v_r * B_theta (~40 LOC)
- `tests/test_beam_target_vpinch.py` -- New tests (~80 LOC)

**LOC estimate**: ~130 LOC total (mostly tests)

**Dependency**: Gap 1 (expansion model) should be implemented first.

---

## Gap 4: Radiation Beyond Bremsstrahlung

### Problem Statement

Only free-free bremsstrahlung is active in the energy equation (`bremsstrahlung.py`). The `improved_radiation.py` module adds recombination and cyclotron but not line radiation. The `line_radiation.py` module EXISTS but is not wired into the engine's energy operator. For Cu electrodes (Z=29), line radiation can exceed bremsstrahlung by 10-100x at 50-200 eV.

### Current Codebase State

We have three radiation modules:
1. `radiation/bremsstrahlung.py` -- P_ff = 1.42e-40 * g_ff * Z * ne^2 * sqrt(Te). Active in engine.
2. `radiation/improved_radiation.py` -- Adds recombination P_fb and cyclotron P_cyc. Has `total_radiation_power()`. Not wired into engine.
3. `radiation/line_radiation.py` -- Piecewise power-law cooling functions for H, Ne, Ar, Cu, W. Has `_cooling_copper()` with 21-point log-log interpolation table from Post et al. (1977). Not wired into engine.

The Cu cooling function peaks at Lambda ~ 3e-30 W m^3 at ~100 eV. Bremsstrahlung at the same conditions (ne=1e24, Te=100 eV = 1.16e6 K): P_ff = 1.42e-40 * 1.2 * 29 * (1e24)^2 * sqrt(1.16e6) ~ 5.3e19 W/m^3. Line radiation: P_line = ne * n_Cu * Lambda(100 eV). If n_Cu/ne = 0.01 (1% Cu impurity): P_line = 1e24 * 1e22 * 3e-30 = 3e16 W/m^3. So at 1% Cu, bremsstrahlung dominates. But at 10% Cu or higher (electrode erosion late in discharge), line radiation becomes comparable.

### State of the Art

**Tabulated cooling curves -- data sources**:

1. **ADAS/OpenADAS** (open-adas.ac.uk): Atomic Data and Analysis Structure. Free access after registration. Provides ADF11 (ionization/recombination rates) and ADF15 (photon emissivity coefficients) for Cu. Coronal equilibrium cooling functions can be computed from these.

2. **CHIANTI** (chiantidatabase.org): Primarily astrophysical (H through Zn, Z<=30). Cu data available but limited compared to lower-Z elements. Version 11 (2025) includes updated Cu collision strengths.

3. **FLYCHK** (NLTE spectral code, LLNL): Computes NLTE opacities and emissivities for arbitrary Z and conditions. Web interface at fly.llnl.gov. Can generate cooling curves for Cu at arbitrary ne/Te.

4. **AtomDB/APEC** (atomdb.org): X-ray spectral database. Cu data available. Used primarily for astrophysical plasmas but applicable to DPF conditions.

5. **Post, Jensen, Tarter, Grasberger & Lokke (1977), At. Data Nucl. Data Tables 20:397**: The original coronal equilibrium cooling curves. This is what our `line_radiation.py` uses. Accuracy: factor of 2-3 for most elements, factor of 3-5 for Cu (fewer atomic data available in 1977).

**Optically thin approximation -- validity for DPF**:

The optical depth for line radiation: `tau = n_Z * sigma_line * L`
where sigma_line ~ pi * r_0^2 * f * (c/Delta_nu) for a Doppler-broadened line.

For Cu M-shell lines at ne=1e24, Te=100 eV, L=1 cm:
- sigma_line ~ 1e-20 m^2 (strong resonance line)
- n_Cu ~ 1e22 (1% impurity)
- tau ~ 1e22 * 1e-20 * 0.01 ~ 1

**tau ~ 1 at DPF pinch conditions.** This means the optically thin approximation is MARGINAL. At peak compression (ne=1e25, L=1 mm), tau ~ 10 (optically thick). At earlier times (ne=1e23, L=1 cm), tau ~ 0.01 (thin).

For practical purposes:
- **Axial phase** (ne ~ 1e22-1e23): Optically thin, coronal equilibrium valid
- **Radial compression** (ne ~ 1e23-1e24): Marginally thin, coronal still usable with escape factor
- **Pinch** (ne ~ 1e24-1e26): Optically thick for strong lines. Need escape factor or full transport.

**Escape factor approximation**: P_line_effective = P_line * f_escape, where:
```
f_escape = (1 - exp(-tau)) / tau     (Holstein approximation for uniform slab)
```
This reduces to f_escape -> 1 for tau << 1 (thin) and f_escape -> 1/tau for tau >> 1 (thick, trapped radiation escapes only from surface).

**Line radiation dominance temperature**:
For Cu (Z=29): Line radiation exceeds bremsstrahlung below ~1 keV:
- At 50 eV: Lambda_line ~ 3e-30 W m^3, P_brem coefficient ~ 1.42e-40 * 1.2 * 29 * sqrt(50*11604) ~ 3.6e-36 W m^3 per ne^2. For n_Cu/ne = 0.01: P_line/P_brem ~ (n_Cu * Lambda_line) / (ne * 3.6e-36) = (0.01 * 3e-30) / 3.6e-36 ~ 8. Line dominates by 8x.
- At 500 eV: Lambda_line ~ 1e-30, P_brem coefficient ~ 1.14e-35. Ratio ~ (0.01 * 1e-30) / 1.14e-35 ~ 0.9. About equal.
- At 5 keV: Lambda_line ~ 3e-31, P_brem coefficient ~ 3.6e-35. Ratio ~ 0.08. Bremsstrahlung dominates.

**NLTE vs LTE regime**:
At DPF pinch conditions (ne ~ 1e24 m^-3, Te ~ 100-1000 eV):
- LTE validity: ne > ne_crit ~ 1e20 * Z^7 * (Te/eV)^{1/2} m^-3 (Griem criterion)
- For Cu: ne_crit ~ 1e20 * 29^7 * sqrt(100) ~ 2e30 m^-3. **DPF is NOT in LTE for Cu.**
- For D (Z=1): ne_crit ~ 1e20 * sqrt(100) ~ 1e21. **DPF IS in LTE for D at pinch.**
- Conclusion: DPF Cu plasma is in **coronal equilibrium** (NLTE), not LTE. Coronal cooling functions are the correct tool.

**Impact on energy balance**: Missing line radiation means Te is overestimated at pinch. For a DPF with 1% Cu impurity and Te ~ 100 eV, line radiation adds ~8x the bremsstrahlung losses. This cools the pinch faster, reduces peak Te, and can affect:
- Neutron yield (thermonuclear component scales exponentially with T)
- Pinch duration (faster cooling -> shorter confinement)
- Current dip depth (radiation losses reduce plasma pressure -> faster expansion)

At 0.1% Cu (clean discharge), line radiation adds ~1x bremsstrahlung -- important but not dominant.

### How Production Codes Handle It

**FLASH (Rochester)**: Uses tabulated NLTE opacities from IONMIX or PrOpacEOS. Full multi-group radiation transport with flux-limited diffusion. This is far more sophisticated than needed for DPF.

**GORGON (Imperial)**: Uses Post-Jensen cooling curves (same source as our `line_radiation.py`) with optically thin assumption. Adds escape factor for high-density regions.

**MACH2 (NRL)**: Uses coronal cooling curves from ADAS. Includes charge-state evolution (time-dependent ionization balance) for multi-species.

**PLUTO (Torino)**: Provides tabulated cooling functions via `radloss.c`. Users supply Lambda(T) tables. No built-in Cu data but easy to add.

**Athena++ (Princeton)**: The `Cooling` module supports tabulated cooling functions Lambda(T). Loaded from external files. No built-in high-Z data.

### Governing Equations

Total radiation power density:

```
P_rad = P_brem + P_line + P_rec + P_cyc                   [W/m^3]

P_brem = 1.42e-40 * g_ff(Te) * Z_bar * ne^2 * sqrt(Te)   [W/m^3]
P_line = ne * n_Z * Lambda(Te, Z)                          [W/m^3]
P_rec  = C_rec * ne^2 * Z^2 * sqrt(chi/(kB*Te)) * exp(-chi/(kB*Te))  [W/m^3]
P_cyc  = 6.21e-28 * B^2 * ne * Te                         [W/m^3]

Lambda(Te, Z) = coronal cooling function from tabulated data  [W m^3]
```

With escape factor for optically thick lines:
```
P_line_eff = P_line * f_escape
f_escape = (1 - exp(-tau)) / tau
tau = n_Z * sigma_line_eff * L_char
L_char = min(r_pinch, z_pinch)   (characteristic escape length)
sigma_line_eff ~ 1e-20 * (13.6/Te_eV)^{1/2} [m^2]  (crude estimate for resonance lines)
```

Energy equation source term:
```
dTe/dt = ... - P_rad / (1.5 * ne * kB)    [K/s]
```

### Implementation Complexity for MLX (float32)

**MEDIUM.** The cooling function lookup is a log-log interpolation (already coded in `line_radiation.py`). The main work is wiring it into the engine and adding the escape factor. Float32 concerns:

- Lambda values span 1e-40 to 3e-30: this range (1e10) is within float32 (1e-38 to 1e38), but marginal at the low end. Use log-space interpolation (already done in `_cooling_copper`).
- The `exp(-tau)` in the escape factor can underflow for tau > 87 (float32 min ~ 1.2e-38). Clamp: if tau > 80, f_escape = 1/tau.

### Risks and Gotchas

1. **Cu impurity fraction is uncertain**: DPF electrode erosion deposits Cu into the plasma, but the amount depends on discharge energy, electrode geometry, and conditioning. Typical estimates: 0.1-5% by number. This is a calibration parameter.
2. **Coronal equilibrium assumption**: Valid during pinch (density high enough for collisional ionization to dominate over transport). May break during expansion when density drops rapidly.
3. **Overcooling instability**: If line radiation is too strong, it can cool the plasma to Te < 10 eV where recombination cascades dump energy as UV/visible photons. This creates a thermal instability (cooling rate increases as Te drops). The implicit solver in `bremsstrahlung.py` handles this for P_ff ~ sqrt(Te) but not for the more complex Lambda(Te) dependence. Need to verify stability.
4. **Post et al. data accuracy for Cu**: Factor of 2-3 uncertainty. For better accuracy, use ADAS data (requires registration and data download).

### Implementation Recommendation

**Approach**: Wire `line_radiation.py` into the engine's radiation operator. Add escape factor. Add Cu impurity fraction as a config parameter.

**Files to modify**:
- `src/dpf/engine/physics_operators.py` -- Add line radiation call alongside bremsstrahlung in the radiation step (~30 LOC)
- `src/dpf/radiation/line_radiation.py` -- Add `line_radiation_power(ne, Te, Z, n_impurity)` public function and escape factor (~50 LOC, the cooling functions are already there)
- `src/dpf/config.py` -- Add `impurity_fraction` and `impurity_Z` to RadiationConfig (~5 LOC)
- `src/dpf/metal/mlx_sources.py` -- Add line radiation source term for MLX solver (~30 LOC)
- `tests/test_line_radiation.py` -- New tests (~100 LOC)

**LOC estimate**: ~215 LOC total

**Data**: The 21-point Cu cooling function table is already in `line_radiation.py`. For higher accuracy, download ADAS ADF11/ADF15 data for Cu and generate a 50-point table. This is a ~2 hour data task, not a coding task.

---

## Gap 5: Bell-Plesset Convergent RTI Growth

### Problem Statement

The MRT growth rate in `pinch_physics.py` uses the planar formula: `gamma = sqrt(g*k*A - magnetic_term)`. In cylindrical converging geometry (DPF radial implosion), the convergence amplifies perturbations by a factor that depends on the convergence ratio R0/R(t). At DPF convergence ratios of 10:1 to 100:1, this amplification is significant.

### Current Codebase State

`src/dpf/validation/pinch_physics.py` has:
- `mrti_growth_rate()` -- planar MRT with Atwood number and B-field stabilization (Bian 2026)
- `classical_rt_growth_rate()` -- planar RT (no B)
- `critical_magnetic_field()` -- B_c for mode stabilization
- `mrti_saturated_growth_rate()` -- maximum growth rate in strong-B limit
- `mrti_diagnostics()` -- composite diagnostic with e-folding count

`src/dpf/diagnostics/instability.py` has:
- `m0_growth_rate()` -- Kruskal-Schwarzschild sausage instability
- `tearing_mode_growth_rate()` -- FKR tearing mode

None of these include convergent geometry effects (Bell-Plesset).

### State of the Art

**Bell (1951), "Taylor instability on cylinders and spheres"**, Proc. R. Soc. A 214:573:
- Derived the modified RT growth rate for converging cylindrical and spherical geometries
- Key result: in converging geometry, the perturbation amplitude grows as:
  ```
  xi(t) ~ xi_0 * (R_0/R(t))^{bell_exponent} * exp(integral(gamma_planar * dt))
  ```
- The `(R_0/R(t))^{bell_exponent}` factor is the geometric amplification
- For a thin shell converging in cylindrical geometry, bell_exponent depends on mode number m and convergence velocity profile

**Plesset (1954), "On the Stability of Fluid Flows with Spherical Symmetry", J. Appl. Phys. 25:96:
- Extended Bell's analysis to arbitrary deceleration profiles
- The full equation for perturbation amplitude on a converging cylinder:

```
d^2(xi)/dt^2 + 2*(dR/dt)/R * d(xi)/dt - [n/R * (d^2R/dt^2) - n*(n^2-1)/(R^3) * sigma] * xi = 0
```

where:
- xi(t) is the perturbation amplitude
- R(t) is the interface radius (decreasing during implosion)
- n is the mode number (azimuthal)
- sigma is the surface tension (0 for MHD, but B-field provides an effective tension)

For the DPF case (no surface tension, MHD with B-field), the Bell-Plesset equation becomes:

```
d^2(xi)/dt^2 + (2*v_R/R) * d(xi)/dt + (g_eff(t) * k_eff(t) * A - k^2 * v_A^2) * xi = 0
```

where:
- v_R = dR/dt (implosion velocity, negative during compression)
- g_eff = -d^2R/dt^2 (effective gravity from deceleration)
- k_eff = n/R(t) (effective wavenumber increases as R shrinks)
- The term `(2*v_R/R) * d(xi)/dt` is the **Bell-Plesset damping** (or amplification) term

**Key physics**: Two convergence effects:

1. **Wavenumber stretching**: k_eff = n/R(t) increases as R decreases. Higher k means shorter wavelength perturbations become important later in the implosion. The growth rate gamma ~ sqrt(g*k) increases as k increases.

2. **Geometric amplification**: The `(2*v_R/R) * d(xi)/dt` term. When v_R < 0 (compression): if d(xi)/dt > 0 (growing perturbation), this term is negative (DAMPING during compression). When v_R > 0 (expansion/deceleration): this term amplifies. The NET effect over a full compression-deceleration cycle is amplification.

**Amplification factor for typical DPF conditions**:

For convergence ratio CR = R_0/R_min:
- CR = 10: Amplification ~ CR^{1/2} to CR^1 ~ 3-10x (depends on deceleration profile)
- CR = 30: Amplification ~ 5-30x
- CR = 100: Amplification ~ 10-100x

The amplification exponent depends on the implosion velocity profile:
- Constant velocity (free-fall): amplification ~ CR^{1/2} (minimal Bell-Plesset)
- Decelerating implosion (snowplow): amplification ~ CR^1 (maximum Bell-Plesset)
- DPF is between these: initial constant velocity, then deceleration near axis

**Bian et al. (2026), Phys. Plasmas 33, 012303** -- MRT in DPF:
- Analyzed MRT in the context of dense plasma focus specifically
- Found that axial magnetic field (B_z) stabilizes the dominant m=0 mode
- Without B_z: growth rate gamma ~ sqrt(g * k * A)
- With B_z: gamma = sqrt(g*k*A - 2*B_z^2*k^2/(mu_0*(rho_h+rho_l)))
- Critical wavelength: lambda_c = 4*pi*B_z^2 / (mu_0*(rho_h-rho_l)*g)
- Did NOT include Bell-Plesset effects (analyzed a planar slab model)
- Conclusion: "convergent geometry effects may further modify the growth rates" -- acknowledged as a gap in their analysis

**Does Bell-Plesset affect pinch timing?**:
YES, indirectly:
- Higher instability growth -> earlier m=0 disruption -> shorter pinch duration -> lower Yn
- At CR=30 (typical PF-1000): Bell-Plesset increases the number of e-foldings from ~5 (planar) to ~8-10
- With 10 e-foldings, any seed perturbation of 1% grows to e^10 = 22,000x -- certainly past nonlinear threshold
- This means the pinch disrupts ~2-3 e-folding times earlier than planar prediction
- At tau_m0 ~ 30 ns per e-folding: pinch is ~60-90 ns shorter (significant compared to ~200-300 ns pinch lifetime)

### How Production Codes Handle It

**FLASH (Rochester)**: Resolves perturbations on the grid. Bell-Plesset emerges naturally from the cylindrical/spherical geometry of the mesh. No explicit growth rate model needed (but requires 2D/3D simulation).

**GORGON (Imperial)**: Same -- resolves on grid in 3D cylindrical/Cartesian.

**ICF codes (HYDRA, LASNEX)**: Use explicit Bell-Plesset growth rate modifications in their 1D implosion models. The HYDRA perturbation model tracks individual modes with the Bell-Plesset ODE. This is the closest analog to what we need.

**Athena++ / SpECTRE**: No built-in Bell-Plesset model. These are grid codes that resolve instabilities directly.

**Semi-analytic codes (Lee model, DELPHI)**: Do not include Bell-Plesset. Use planar growth rates or no instability model at all.

### Governing Equations

The Bell-Plesset ODE for perturbation amplitude in cylindrical converging geometry:

```
d^2(xi)/dt^2 + (2*v_R/R) * d(xi)/dt + omega^2(t) * xi = 0

omega^2(t) = -g_eff(t) * k_eff(t) * A + k_eff^2 * v_A^2

where:
  g_eff(t) = -d^2R/dt^2       (effective gravity; positive during deceleration)
  k_eff(t) = n / R(t)          (mode number n, stretching wavenumber)
  v_A(t)  = B_z / sqrt(mu_0 * rho_avg(t))  (Alfven speed for stabilizing B_z)
  A = (rho_h - rho_l) / (rho_h + rho_l)    (Atwood number)
  v_R = dR/dt                  (interface velocity)
```

For DPF: R(t) comes from the snowplow model (r_shock trajectory). v_R = dr_shock/dt. g_eff = -d^2r_shock/dt^2.

The instability-driven mixing width (for sub-grid modeling without resolving individual modes):

```
h_mix(t) = alpha_RT * A * integral(integral(g_eff * dt) * dt)    (Dimonte-Schneider model)
```

where alpha_RT ~ 0.05-0.07 is the RT mixing coefficient. This gives the RMS amplitude of density perturbations without resolving modes.

Alternatively, track a few dominant modes (n=1,2,3) independently and take the max amplitude:
```
xi_n(t) from Bell-Plesset ODE for each n
Disruption when max_n(xi_n) > r_pinch / 2
```

### Implementation Complexity for MLX (float32)

**LOW.** The Bell-Plesset ODE is a simple 2nd-order ODE solved alongside the snowplow trajectory. Only a few modes need to be tracked (n=1,2,3). This adds ~3 extra ODEs to the snowplow sub-cycle. No spatial discretization needed for the sub-grid model.

Float32 concerns: None. All quantities (R ~ mm, xi ~ um to mm, growth rates ~ 1e8 s^-1) are well within float32 range.

### Risks and Gotchas

1. **Initial perturbation amplitude**: The seed perturbation xi_0 is unknown and varies shot-to-shot. Typically assume xi_0 ~ 1% of initial radius (xi_0 = 0.01 * R_0). This is a calibration parameter.
2. **Nonlinear saturation**: The Bell-Plesset ODE is linear. When xi ~ R/2, the perturbation is nonlinear and the ODE over-predicts growth. Need a saturation model: `xi_sat = min(xi_linear, C_sat * R)` with C_sat ~ 0.3-0.5.
3. **Mode coupling**: In reality, different modes interact nonlinearly (mode n generates modes n+1, n-1, 2n, etc.). The independent-mode approximation underestimates late-time mixing.
4. **Interaction with expansion model**: The Bell-Plesset analysis gives the instability growth during compression. During expansion (Gap 1), the modes continue to grow but with reversed sign of v_R. The expansion phase may be more important for pinch disruption timing.
5. **Resolved vs sub-grid**: For 2D/3D MHD (MLX solver), perturbations ARE resolved on the grid. The Bell-Plesset model would only be needed for the 0D/1D snowplow model. For the MHD solver, the convergent geometry effects emerge naturally from the cylindrical coordinates.

### Implementation Recommendation

**Approach**: Add Bell-Plesset ODE to the snowplow model. Track modes n=1,2,3. Use disruption criterion (xi > 0.3 * r_pinch) to trigger pinch termination and transition to expansion (Gap 1). For the MHD solver, no changes needed -- convergent effects are captured by the cylindrical grid.

**Files to modify**:
- `src/dpf/fluid/snowplow.py` -- Add `_step_bell_plesset()` method tracking 3 modes via the BP ODE alongside the radial trajectory (~80 LOC)
- `src/dpf/validation/pinch_physics.py` -- Add `bell_plesset_growth_rate()` and `bell_plesset_amplification()` diagnostic functions (~60 LOC)
- `src/dpf/config.py` -- Add `bp_seed_amplitude` (default 0.01) and `bp_modes` (default [1,2,3]) to SnowplowConfig (~5 LOC)
- `tests/test_bell_plesset.py` -- New test file (~100 LOC)

**LOC estimate**: ~245 LOC total

**No new research data needed** -- equations from Bell 1951, Plesset 1954, and the HYDRA perturbation model are fully specified in the literature.

---

## Summary Table

| Gap | Priority | LOC | Dependency | Data Needed | Float32 Risk |
|-----|----------|-----|------------|-------------|-------------|
| 1. Post-pinch expansion | HIGHEST | ~200 | None | PF-1000 waveforms (have) | None |
| 2. Drift-velocity resistivity | HIGH | ~225 | None | None | Low (CFL constraint) |
| 3. Beam-target V_pinch | HIGH | ~130 | Gap 1 | None | None |
| 4. Line radiation + opacity | MEDIUM | ~215 | None | ADAS Cu data (optional) | Low (log-space) |
| 5. Bell-Plesset RTI | MEDIUM | ~245 | None | None | None |

**Recommended implementation order**: Gap 1 -> Gap 3 -> Gap 2 -> Gap 4 -> Gap 5

Rationale:
- Gap 1 directly fixes the current dip (most visible deficiency) and unblocks Gap 3
- Gap 3 is trivial once Gap 1 is done
- Gap 2 improves pinch physics independent of others
- Gap 4 and 5 are diagnostic/calibration improvements, lower urgency

**Total LOC for all 5 gaps**: ~1,015 LOC (including ~500 LOC of tests)
