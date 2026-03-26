# PIC-MHD Hybrid Module Validation Plan

**Module**: `src/dpf/experimental/pic/hybrid.py` (1717 LOC)
**Support**: `src/dpf/kinetic/manager.py`, `src/dpf/kinetic/initialize.py`
**Diagnostics**: `src/dpf/diagnostics/beam_target.py`, `src/dpf/diagnostics/pic_yield.py`
**Date**: 2026-03-26
**Status**: Design document (no implementation)

---

## 1. Module Inventory

### Numba Kernels (hybrid.py)

| Function | Lines | Purpose | Test Status |
|----------|-------|---------|-------------|
| `_nanbu_scatter_kernel` | 38-287 | Nanbu-Perez (2012) relativistic binary Coulomb collisions | TESTED: empty, subluminal, speed conservation, large s12 deflection (test_pic_hybrid.py) |
| `_coulomb_scatter` | 290-394 | Takizuka-Abe (1977) simplified Coulomb scattering | UNTESTED: no direct tests; exercised only via HybridPIC fallback path |
| `_boris_push_kernel` | 397-478 | Boris algorithm for charged-particle push in E+B | TESTED: gyration radius, return-to-origin (test_pic_validation.py) |
| `_deposit_density_kernel` | 481-557 | CIC trilinear density deposition | TESTED: charge conservation, multi-particle (test_pic_validation.py) |
| `_deposit_current_kernel` | 560-665 | CIC trilinear current deposition J=qnv | UNTESTED: no direct current deposition tests |
| `_deposit_current_esirkepov_kernel` | 668-894 | Esirkepov (2001) charge-conserving current deposition | UNTESTED: no tests at all |
| `_interpolate_scalar_kernel` | 897-954 | Inverse CIC scalar field -> particles | UNTESTED: tested indirectly via `interpolate_field_to_particles` |
| `_interpolate_vector_kernel` | 957-1023 | Inverse CIC vector field -> particles | TESTED: uniform field (test_pic_validation.py) |

### Public API Wrappers (hybrid.py)

| Function | Lines | Purpose | Test Status |
|----------|-------|---------|-------------|
| `boris_push` | 1031-1077 | Thin wrapper around `_boris_push_kernel` | TESTED |
| `deposit_density` | 1080-1118 | Wrapper around `_deposit_density_kernel` | TESTED |
| `deposit_current` | 1121-1159 | Wrapper around `_deposit_current_kernel` | UNTESTED |
| `deposit_current_esirkepov` | 1162-1203 | Wrapper around Esirkepov kernel | UNTESTED |
| `interpolate_field_to_particles` | 1206-1242 | Dispatch to scalar/vector interpolation | TESTED (vector only) |

### Classes (hybrid.py)

| Class/Method | Lines | Purpose | Test Status |
|--------------|-------|---------|-------------|
| `ParticleSpecies` | 1250-1284 | Dataclass: positions, velocities, weights | UNTESTED directly |
| `HybridPIC.__init__` | 1308-1335 | Driver init: grid, timestep, species list | TESTED (via test_pic_hybrid.py) |
| `HybridPIC.enable_collisions` | 1337-1350 | Configure Coulomb collision background | TESTED (binary flag wiring) |
| `HybridPIC.add_species` | 1356-1396 | Register a particle species | UNTESTED directly |
| `HybridPIC.push_particles` | 1402-1488 | Boris push + BC + collision for all species | UNTESTED as integration |
| `HybridPIC._apply_reflecting_bc` | 1490-1519 | Reflecting walls at domain edges | UNTESTED |
| `HybridPIC.deposit` | 1525-1572 | Deposit all species (rho, Jx, Jy, Jz) | UNTESTED as integration |
| `HybridPIC.inject_beam` | 1578-1665 | Inject beam particles into existing species | UNTESTED directly |
| `detect_instability` | 1673-1717 | Heuristic m=0 sausage instability detector | UNTESTED |

### Support Modules

| Module | Purpose | Test Status |
|--------|---------|-------------|
| `kinetic/manager.py` (163 LOC) | KineticManager: engine-PIC bridge, beam injection, J_kin coupling | UNTESTED |
| `kinetic/initialize.py` (96 LOC) | Initialize PIC particles from MHD state | TESTED: shapes, Maxwellian velocity (test_pic_validation.py) |
| `diagnostics/beam_target.py` | DD cross section (Bosch-Hale 1992) + beam-target yield | TESTED: cross section only (test_pic_validation.py) |
| `diagnostics/pic_yield.py` | PIC-based neutron yield rate from macro-particles | TESTED: nonzero/zero yield (test_pic_validation.py) |

### Summary

- **Tested**: 10 functions/methods (kernels for Boris, density deposition, vector interp, Nanbu scatter)
- **Untested**: 14 functions/methods (Esirkepov, current deposition, reflecting BC, HybridPIC integration, detect_instability, KineticManager)
- **Never run end-to-end**: PIC has never been activated during a full DPF discharge simulation

---

## 2. Unit Test Plan

### 2.1 `_coulomb_scatter` (Takizuka-Abe)

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_coulomb_scatter_preserves_speed` | N=200 deuterons at 1e6 m/s, n_bg=1e25, T=100 eV, dt=1e-10 | Mean speed change < 5% (elastic scattering preserves speed) | rel 0.05 |
| `test_coulomb_scatter_empty` | N=0 array | Returns unchanged empty array | exact |
| `test_coulomb_scatter_low_energy_skip` | Particles with E < 1 eV | No scattering applied (guard at line 336) | exact |
| `test_coulomb_scatter_nu_cap` | Large n_bg=1e30, dt=1e-6 | nu capped at 0.5/dt, deflection < pi/2 per step | theta_rms < 0.71 rad |

### 2.2 `_deposit_current_kernel` (CIC J)

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_current_density_single_particle` | 1 particle at cell center, vx=1e6 m/s, q=e, w=1e15 | Jx integrates to q*w*vx (total current contribution) | rel 1e-6 |
| `test_current_density_zero_velocity` | Stationary particles | J = 0 everywhere | exact |
| `test_current_density_sum_rule` | N=100 particles with random positions | sum(J)*dV = q * sum(w*v) | rel 1e-6 |

### 2.3 `_deposit_current_esirkepov_kernel` (charge-conserving)

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_esirkepov_continuity_equation` | 1 particle moving from (2.5,2.5,2.5)*dx to (3.5,2.5,2.5)*dx | div(J)*dt + delta_rho = 0 for every cell | abs 1e-10 |
| `test_esirkepov_matches_cic_total` | N=50 particles, small displacement | Total deposited current integral matches CIC within 10% | rel 0.1 |
| `test_esirkepov_stationary_zero` | positions_old == positions_new | J = 0 everywhere | exact |
| `test_esirkepov_multi_cell_crossing` | Particle moves 2+ cells in one step | No index-out-of-bounds; J remains finite | no NaN/Inf |

### 2.4 `_apply_reflecting_bc`

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_reflect_lower_boundary` | Particle at x=-0.001 with vx=-1e5 | x = +0.001, vx = +1e5 | exact |
| `test_reflect_upper_boundary` | Particle at x=Lx+0.001 with vx=+1e5 | x = Lx-0.001, vx = -1e5 | exact |
| `test_interior_unchanged` | Particle well inside domain | Position and velocity unchanged | exact |
| `test_corner_reflection` | Particle outside in all 3 dimensions | All components reflected | exact |

### 2.5 `detect_instability`

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_uniform_no_instability` | Uniform rho, uniform Bz | Returns False | exact |
| `test_compressed_with_sign_change` | rho peaked at center (10x mean), Bz reverses sign | Returns True | exact |
| `test_compressed_no_sign_change` | High density compression but uniform Bz | Returns False (needs both criteria) | exact |
| `test_zero_density` | rho = 0 everywhere | Returns False (mean_rho <= 0 guard) | exact |

### 2.6 `inject_beam`

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_beam_particle_count` | inject n_beam=100 into empty species | species.n_particles() == 100 | exact |
| `test_beam_energy_correct` | E=100 keV deuterons | 0.5*m*v^2 / eV = 100e3 for each particle (spread=0) | rel 1e-6 |
| `test_beam_direction` | direction=[0,0,1], spread=0 | All velocities along z-axis | abs 1e-10 on vx,vy |
| `test_beam_spread_cone` | spread=0.1 rad | All velocity vectors within 0.1 rad of direction | max angle < 0.1 |
| `test_beam_weights` | weight_total=1e16, n_beam=100 | Each macro-particle weight = 1e14 | exact |

### 2.7 `KineticManager`

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| `test_init_creates_empty_species` | Default KineticConfig | species[0].n_particles() == 0 | exact |
| `test_step_before_start_time` | time < kc.start_time | Returns {"status": "waiting"}, no particles pushed | exact |
| `test_update_mhd_state` | rho with peak 1e-3 kg/m^3, Te with peak 1e7 K | _n_bg ~ 3e23 m^-3, _T_bg_eV ~ 862 eV | rel 0.01 |
| `test_beam_injection_trigger` | time >= start_time, inject_beam=True | beam_injected=True, n_particles > 0 | exact |

---

## 3. Integration Test Plan

### 3.1 Landau Damping (electrostatic, 1D)

**Purpose**: Validate Boris push + field interpolation in the electrostatic limit.

**Setup**:
- Grid: 128 x 1 x 1, dx = L/128, L = 2*pi/k with k = 0.4/lambda_D
- Species: electrons (m_e, -e), 10,000 particles per cell
- Initial perturbation: delta_n/n0 = 0.01 * cos(k*x), Maxwellian v_th
- E-field: Poisson solve from deposited charge (add simple 1D Poisson solver for test)
- B = 0 (purely electrostatic)
- dt = 0.1 / omega_pe

**Expected**: Electric field amplitude decays as exp(-gamma*t) with gamma/omega_pe = 0.0661 (Landau damping rate for k*lambda_D = 0.4).

**Pass/Fail**: Measured damping rate within 10% of analytical value over 10 plasma periods.

**LOC estimate**: ~120 (Poisson solver + test setup + analysis)

**Note**: This test requires adding a 1D electrostatic Poisson solver (not currently in the codebase). This is a standard PIC validation test from Birdsall & Langdon (1985).

### 3.2 Two-Stream Instability

**Purpose**: Validate beam injection + CIC deposition produce correct instability growth.

**Setup**:
- Grid: 128 x 1 x 1, periodic
- Two counter-streaming beams: v0 = +/- 3*v_th, 5000 particles each
- Small perturbation on positions (1e-3 * dx * sin(2*pi*x/L))
- B = 0
- dt = 0.2 / omega_pe

**Expected**: Fastest growing mode at k = omega_pe / v0. Growth rate gamma = omega_pe * sqrt(3)/2 * (n_beam/n_total)^{1/3} for cold beams.

**Pass/Fail**: Electric field energy grows exponentially at the predicted rate within 20% during the linear phase (t < 10/omega_pe).

**LOC estimate**: ~100 (test setup + growth rate measurement)

### 3.3 Ion Cyclotron Gyration (extended Boris validation)

**Purpose**: Validate long-term phase-space conservation of Boris integrator.

**Setup**:
- Single deuteron, B = (0, 0, 1) T, v_perp = 1e6 m/s, E = 0
- Run for 1000 gyroperiods
- dt = T_c / 32 (32 steps per gyroperiod)

**Expected**: Larmor radius constant to machine precision. Phase error grows linearly (not exponentially) with time. Energy conserved to < 1e-10.

**Pass/Fail**: |r_L(t=1000*T_c) - r_L(t=0)| / r_L(0) < 1e-8. Energy drift < 1e-10 per gyroperiod.

**LOC estimate**: ~60

### 3.4 Ion Acoustic Wave (collision operator)

**Purpose**: Validate that the Nanbu-Perez collision operator produces correct thermalization.

**Setup**:
- 2000 deuterons initialized at 200 keV mono-energetic beam
- Background: n_bg = 1e25 m^-3, T_bg = 100 eV
- Run for 1000 collision times: tau_coll = v^3 * 4*pi*eps0^2 * m^2 / (n*q^4*ln_Lambda)
- dt = tau_coll / 100

**Expected**: Beam energy distribution thermalizes from delta-function to Maxwellian. Mean energy decreases as E(t) = E0 * exp(-t/tau_slow), where tau_slow = m_beam * v^3 / (2 * n * q^4 * ln_Lambda / (4*pi*eps0^2 * m_reduced)).

**Pass/Fail**: After 5*tau_slow, the energy distribution is consistent with a Maxwellian (chi-squared test on velocity histogram, p > 0.05).

**LOC estimate**: ~100

---

## 4. End-to-End Test Plan

### Phase E2E-1: PIC on static MHD state (no coupling)

**Purpose**: Verify particle push works with realistic DPF field structure.

**Setup**:
1. Load a saved MHD state from mid-discharge (t ~ 5 us, PF-1000 at 27 kV)
2. Initialize 10,000 particles from MHD state using `initialize_particles_from_mhd()`
3. Push for 100 PIC steps at dt = 1e-10 s
4. E-field: E = -v x B + eta*J (from MHD state)
5. B-field: from MHD state
6. No feedback to MHD (one-way coupling)

**Expected**:
- All particles remain within domain (reflecting BCs work)
- No NaN in positions or velocities
- Deposited J_kin has reasonable magnitude (< 1e12 A/m^2)
- Total kinetic energy conserved to < 1% (no collisions)

**Pass/Fail**: Zero NaN, all particles in domain, energy conservation < 1%.

**LOC estimate**: ~80

### Phase E2E-2: PIC with MHD feedback (J_kin -> MHD source)

**Purpose**: Verify energy conservation with bidirectional PIC-MHD coupling.

**Setup**:
1. Small grid: 16 x 1 x 32, PF-1000 parameters
2. Enable kinetic module: `config.kinetic.enabled = True, start_time = 0`
3. Initialize thermal particles (no beam)
4. Run 10 engine steps
5. Monitor: total energy = MHD energy + particle kinetic energy

**Expected**:
- J_kin appears in `_current_source_terms` dict
- MHD state evolves (J_kin modifies B via Faraday's law)
- Total energy (MHD + kinetic) conserved to < 5%

**Pass/Fail**: Energy conservation < 5%. No crashes. J_kin non-zero.

**LOC estimate**: ~120

### Phase E2E-3: Full DPF discharge with PIC beam injection

**Purpose**: First-ever end-to-end DPF simulation with kinetic beam ions.

**Setup**:
1. PF-1000 at 27 kV, 3.5 Torr D2
2. `config.kinetic.enabled = True`
3. `config.kinetic.start_time = 5.5e-6` (near pinch)
4. `config.kinetic.beam_energy = 100e3` (100 keV)
5. `config.kinetic.n_particles = 10000`
6. Run full discharge: 0 to 8 us
7. Monitor: I(t), beam particle count, neutron yield rate

**Expected**:
- Simulation completes without crash
- Beam injected at pinch time
- Non-zero neutron yield from PIC particles
- I(t) waveform not significantly perturbed by PIC feedback

**Pass/Fail**: Completes. Yn > 0. I_peak within 10% of PIC-off baseline.

**LOC estimate**: ~150

**Mark**: `@pytest.mark.slow` (estimated 5-15 minutes)

---

## 5. Reference Benchmarks

### 5.1 Published PIC-DPF Results

| Source | Device | Method | Key Result | Usable For |
|--------|--------|--------|------------|------------|
| Schmidt et al., PRL 109:205003 (2012) | 1 MJ DPF | Fully kinetic (LSP code) | Yn ~ 3e11, beam-target dominates | Order-of-magnitude Yn comparison |
| Pasternak et al., (2024) | PF-1000 | PIC | Beam energy spectra | Beam energy distribution shape |
| Lee & Saw (2014) | Multiple | Lee model beam-target | Yn ~ I_pinch^4 scaling | Scaling law cross-check |
| Gribkov et al. (2007) | PF-1000 27kV | Experiment | Yn = 0.5-2.0 x 10^11 | Target range for end-to-end |

### 5.2 Expected Yn for PF-1000 at 27 kV

- **Experimental range**: 0.5 - 2.0 x 10^11 neutrons per shot (Gribkov 2007, high shot-to-shot variability)
- **Schmidt (2012) LSP result**: ~3 x 10^11 for a 1 MJ device (PF-1000 is 1 MJ class)
- **Lee model estimate**: Yn ~ 10^10 - 10^11 depending on fc/fm
- **Our target**: Order-of-magnitude agreement (10^10 - 10^12). Within factor-of-3 is excellent for a first PIC-MHD hybrid.

### 5.3 Beam-Target vs Thermonuclear Ratio

For PF-1000 class devices, beam-target contribution should dominate:
- Yn_beam / Yn_thermo ~ 10-100 (Schmidt 2012)
- Our PIC module should show this ratio. If thermonuclear dominates, something is wrong with beam injection or collision operator.

---

## 6. Known Issues

### 6.1 Bugs Found During Code Review

1. **Nanbu self-collision uses same array for both species** (line 1471-1478): `_nanbu_scatter_kernel(new_vel, new_vel, ...)` passes the same velocity array for both species a and b. The kernel modifies in-place (lines 280-285), so species-b reads already-modified species-a velocities. This introduces a systematic bias -- particles processed earlier in the loop see pre-scatter velocities while later particles see post-scatter. Standard practice is to copy one array or use half-step synchronization.

2. **Esirkepov dt uses `self.dt` not the passed `dt`** (line 1561): `deposit()` always uses `self.dt` for Esirkepov deposition, even though `push_particles()` accepts an arbitrary `dt` parameter. If the engine passes a different dt to push vs deposit, the charge conservation guarantee breaks.

3. **E-field computation in `_step_pic` is approximate** (engine/core.py line 871): `E = -v x B` uses the MHD bulk velocity, which omits the Hall term `J x B / (n*e)` and the electron pressure gradient `-grad(P_e)/(n*e)`. For DPF pinch conditions with strong density gradients, this can be 10-100% wrong.

4. **No particle removal**: Particles that lose all energy are never removed. Over long simulations, thousands of thermalized particles consume memory and CPU in the Boris push without contributing to beam-target yield.

5. **Reflecting BC is unphysical for DPF**: Real DPF has conducting electrodes (absorbing for particles hitting the anode/cathode) and open boundaries along the axis. Reflecting BC artificially traps beam ions, overestimating confinement time and potentially yield.

6. **Binary collision self-pairing is unphysical**: Line 1471 pairs species with itself for "self-collisions." Nanbu-Perez is designed for inter-species collisions (a != b). Self-collisions should use the Takizuka-Abe scheme with random pairing, or the Nanbu self-collision variant with weight correction.

7. **`weight_total` default 1e16 is device-independent**: The beam weight should scale with I_pinch and pinch duration. PF-1000 at 27 kV has I_pinch ~ 500 kA for ~100 ns, giving ~3e17 ions. The default 1e16 underestimates by 30x.

8. **No sub-cycling PIC within MHD timestep**: MHD dt ~ 1e-9 s, but the beam cyclotron period for 100 keV deuterons in 10 T is T_c ~ 6.5e-9 s (need ~30 steps per gyroperiod => dt_pic ~ 2e-10 s). Currently PIC and MHD use the same dt, which under-resolves gyration by ~5x.

### 6.2 Stubs and Placeholders

- `detect_instability()`: Purely heuristic. Density compression + Bz sign change is a crude proxy for m=0. No Fourier analysis, no growth rate estimation, no comparison to Kruskal-Shafranov.
- No electron kinetics: Module docstring says "electrons as massless neutralising fluid" but there is no actual Ohm's law solver or generalized Ohm's law implementation.
- No field solve: PIC module relies entirely on MHD fields. For self-consistent PIC-MHD, need at least an induction equation update from J_kin.

---

## 7. Implementation Phases

### Phase V1: Unit Tests for Standalone Components (no coupling)

**Scope**: Test every untested kernel and method in isolation.

| Test File | Tests | Target |
|-----------|-------|--------|
| `test_pic_unit_coulomb_scatter.py` | 4 tests for `_coulomb_scatter` | Speed preservation, empty guard, low-energy skip, nu cap |
| `test_pic_unit_current_deposition.py` | 3 tests for `_deposit_current_kernel` | Single particle, zero velocity, sum rule |
| `test_pic_unit_esirkepov.py` | 4 tests for Esirkepov kernel | Continuity equation, total match, stationary zero, multi-cell |
| `test_pic_unit_reflecting_bc.py` | 4 tests for `_apply_reflecting_bc` | Lower, upper, interior, corner |
| `test_pic_unit_instability.py` | 4 tests for `detect_instability` | Uniform, compressed+sign, compressed-only, zero density |
| `test_pic_unit_inject_beam.py` | 5 tests for `inject_beam` | Count, energy, direction, spread, weights |
| `test_pic_unit_kinetic_manager.py` | 4 tests for `KineticManager` | Init, wait, mhd_state, injection trigger |

**Total**: 28 new tests
**LOC estimate**: ~500
**Dependencies**: None beyond existing dpf imports + numba
**Duration**: All tests < 1s each (pure kernel tests)

### Phase V2: Integration Tests (standard PIC benchmarks)

**Scope**: Classical PIC validation problems.

| Test File | Tests | Target |
|-----------|-------|--------|
| `test_pic_integ_gyration.py` | 2 tests | 1000-gyroperiod Boris conservation |
| `test_pic_integ_two_stream.py` | 2 tests | Two-stream growth rate |
| `test_pic_integ_thermalization.py` | 2 tests | Nanbu-Perez thermalization |

**Total**: 6 new tests
**LOC estimate**: ~350 (includes 1D Poisson solver utility, ~80 LOC)
**Dependencies**: May need simple 1D Poisson solver for Landau damping test
**Duration**: ~5-30s per test (mark @pytest.mark.slow for > 1s)

**Note**: Landau damping test deferred to V2b because it requires a 1D Poisson solver not present in the codebase. The two-stream and thermalization tests use fixed fields or self-contained setups.

### Phase V3: MHD-Coupled Tests (static + one-way + two-way)

**Scope**: PIC operating on realistic MHD fields with and without feedback.

| Test File | Tests | Target |
|-----------|-------|--------|
| `test_pic_mhd_static.py` | 3 tests | Push on static state, no NaN, energy conservation |
| `test_pic_mhd_coupled.py` | 3 tests | J_kin feedback, total energy conservation, no crash |

**Total**: 6 new tests
**LOC estimate**: ~300
**Dependencies**: Saved MHD state or simple analytic field initialization
**Duration**: 5-60s per test (@pytest.mark.slow)

### Phase V4: Full Discharge End-to-End

**Scope**: PF-1000 full discharge with PIC beam injection.

| Test File | Tests | Target |
|-----------|-------|--------|
| `test_pic_e2e_pf1000.py` | 3 tests | Completes, Yn > 0, I(t) not perturbed |

**Total**: 3 new tests
**LOC estimate**: ~250
**Dependencies**: Working PF-1000 preset, MLX or Python backend
**Duration**: 5-15 min per test (@pytest.mark.slow)

---

## 8. LOC Estimates Summary

| Phase | New Test LOC | New Utility LOC | Total |
|-------|-------------|-----------------|-------|
| V1: Unit tests | 500 | 0 | 500 |
| V2: Integration | 270 | 80 (Poisson solver) | 350 |
| V3: MHD-coupled | 300 | 0 | 300 |
| V4: End-to-end | 250 | 0 | 250 |
| **Total** | **1320** | **80** | **1400** |

Bug fixes discovered during validation (estimated):

| Fix | LOC | Priority |
|-----|-----|----------|
| Esirkepov dt mismatch (issue 6.1.2) | 5 | HIGH |
| Self-collision copy (issue 6.1.1) | 10 | HIGH |
| PIC sub-cycling (issue 6.1.8) | 40 | MEDIUM |
| Absorbing electrode BC (issue 6.1.5) | 60 | MEDIUM |
| Particle removal (issue 6.1.4) | 30 | LOW |
| Weight scaling (issue 6.1.7) | 15 | LOW |

---

## 9. Risk Assessment

### Blocking Risks (will prevent end-to-end)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **PIC timestep too small** | HIGH | MHD dt ~ 1e-9, PIC needs dt ~ 2e-10 for 100 keV in 10 T. Without sub-cycling, gyration is under-resolved and Boris push becomes inaccurate. | Implement PIC sub-cycling: N_pic = ceil(dt_mhd / dt_pic), cap at 50. ~40 LOC in `push_particles()`. |
| **Memory exhaustion** | MEDIUM | 10,000 particles x 7 arrays x float64 = 4.3 MB (fine). But 1M particles (needed for convergence) = 430 MB. Full 3D deposit on 64^3 grid = 2 MB. Manageable on 36 GB. | Start with 10K particles. Profile memory. Scale to 100K max for V4. |
| **Numba JIT compilation time** | MEDIUM | First call to each kernel takes 1-5s. 8 kernels = 8-40s JIT overhead. Makes rapid iteration painful. | Use `cache=True` (already set). Pre-compile in conftest fixture. |
| **E-field approximation** | HIGH | `E = -v x B` missing Hall and pressure gradient terms. At pinch, Hall term E_Hall ~ J x B/(ne) can equal or exceed convective E. Beam ions see wrong E-field, wrong trajectories. | Phase V3 should measure E_Hall magnitude. If > 10% of E_conv, add Hall term to `_step_pic`. ~30 LOC. |

### Non-Blocking Risks (degrade accuracy, don't prevent running)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Reflecting BC inflates confinement | HIGH | Beam ions bounce off walls instead of escaping. Overestimates Yn by ~2-5x. | Replace with absorbing BC on electrodes (anode r=0, cathode r=R_outer) in Phase V3. |
| Self-collision bias (issue 6.1.1) | HIGH | Systematic directional bias in scattering. Under-estimates isotropization by ~10-30%. | Copy velocity array before passing to Nanbu kernel. 10 LOC fix. |
| CIC noise for < 100 particles/cell | MEDIUM | Deposited J_kin noisy, especially on 16^3 grid with 10K particles (~2.4/cell). | Smooth J_kin with 3-point Gaussian before injecting as MHD source. |
| No electron response | LOW for V1-V3 | Electrons assumed perfectly neutralizing. OK for ions > 10 keV. Breaks below ~1 keV where electron drag matters. | Acceptable for beam-target physics. Flag for future electron PIC. |

### Critical Path

```
V1 (unit tests, 1-2 days)
  |
  v
V2 (integration, 2-3 days)
  |
  v
Bug fixes: Esirkepov dt, self-collision copy, sub-cycling (1 day)
  |
  v
V3 (MHD-coupled, 2-3 days)
  |
  v
V4 (end-to-end, 1-2 days)
```

**Total estimated calendar time**: 7-11 days of focused work.
**Longest single test**: V4 full discharge at ~15 minutes wall-clock.
**Most likely blocker**: PIC sub-cycling not implemented (issue 6.1.8). Without it, V3/V4 will produce garbage because Boris push under-resolves gyration at DPF field strengths.

---

## Appendix A: Key Physical Parameters

| Quantity | PF-1000 27 kV Value | Formula |
|----------|---------------------|---------|
| B at pinch | 10-50 T | mu0 * I / (2*pi*r), r ~ 1 mm |
| Deuteron cyclotron period | 6.5 ns (at 10 T) | 2*pi*m_D / (e*B) |
| 100 keV deuteron speed | 3.1e6 m/s | sqrt(2*E/m_D) |
| Larmor radius (100 keV, 10 T) | 6.5 mm | m_D*v / (e*B) |
| Slowing-down time | ~50 ns | m*v^3 / (n*q^4*ln_L / (4*pi*eps0^2)) at n=1e25 |
| PIC timestep (30 steps/gyro) | 0.2 ns | T_c / 30 |
| MHD timestep | ~1 ns | dx / v_fast |
| Sub-cycle ratio | ~5 | dt_mhd / dt_pic |
| Grid cells crossed per PIC step | ~0.06 | v * dt_pic / dx at dx=1mm |

## Appendix B: Existing Test Coverage Map

```
hybrid.py (1717 LOC)
  [TESTED]   _nanbu_scatter_kernel      (4 tests in test_pic_hybrid.py)
  [UNTESTED] _coulomb_scatter           (0 tests)
  [TESTED]   _boris_push_kernel         (2 tests in test_pic_validation.py)
  [TESTED]   _deposit_density_kernel    (2 tests in test_pic_validation.py)
  [UNTESTED] _deposit_current_kernel    (0 tests)
  [UNTESTED] _deposit_current_esirkepov (0 tests)
  [TESTED]   _interpolate_scalar_kernel (indirect via vector)
  [TESTED]   _interpolate_vector_kernel (1 test)
  [UNTESTED] _apply_reflecting_bc       (0 tests)
  [UNTESTED] detect_instability         (0 tests)
  [UNTESTED] inject_beam                (0 tests)
  [UNTESTED] push_particles (full)      (0 tests)
  [UNTESTED] deposit (full)             (0 tests)

manager.py (163 LOC)
  [UNTESTED] KineticManager             (0 tests)

initialize.py (96 LOC)
  [TESTED]   initialize_particles_from_mhd (2 tests)

Existing test files:
  test_pic_hybrid.py      — 7 tests  (Nanbu kernel + HybridPIC collision flag)
  test_pic_validation.py  — 12 tests (Boris, CIC, interpolation, DD xsec, yield, init)
  test_beam_tracker.py    — 5 tests  (BeamTracker separate diagnostic, not HybridPIC)
```
