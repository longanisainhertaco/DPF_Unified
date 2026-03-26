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

---

## Six Sigma Refinement -- Bug Fix Specifications

**Date**: 2026-03-26
**Purpose**: Raise PIC readiness from 5/10 to 7/10 via DMAIC on all 8 known bugs.
**Review source**: `SCAFFOLD_REVIEW_SIX_SIGMA.md` Section 3 (PIC)

---

### D -- Define: Bug Fix Specifications

#### Bug 1: Nanbu Self-Collision In-Place Mutation Bias

- **Location**: `hybrid.py:1471-1478`
- **Root cause**: `_nanbu_scatter_kernel(new_vel, new_vel, ...)` passes the same array as both `vel_a` and `vel_b`. The kernel writes back at lines 280-285 (`vel_a[ia] = ...`, `vel_b[ib] = ...`). When `ia == ib` (self-collision) or when the loop processes pair (i, i+1) and later (i+1, i+2), species-b reads already-scattered species-a velocities. This introduces ordering-dependent directional bias.
- **Impact**: Scattering isotropization rate off by ~10-30%. Beam slowing-down time systematically wrong. Non-blocking for V1/V2 but corrupts V3/V4 collision physics.
- **Fix**: Copy `new_vel` before passing as the second argument:
  ```python
  vel_b_copy = new_vel.copy()
  _nanbu_scatter_kernel(
      new_vel, vel_b_copy,
      sp.weights, sp.weights,
      ...
  )
  # Average result: new_vel already modified as species a;
  # no need to write back vel_b_copy (self-collision symmetry).
  ```
- **LOC**: 3 (add copy, adjust call)
- **Regression risk**: LOW -- only changes collision behavior, no effect on collisionless tests
- **Test**: `test_nanbu_self_collision_no_ordering_bias` -- run 2000 particles with identical initial conditions, compare RMS scattering angle against the same test using separate arrays. Must agree within 5%.

#### Bug 2: Esirkepov dt Mismatch

- **Location**: `hybrid.py:1561` -- `self.dt` used instead of the dt passed to `push_particles()`
- **Root cause**: `deposit()` method at line 1561 hardcodes `self.dt` in the Esirkepov call. But `push_particles()` at line 1431 accepts `dt` parameter and may be called with a different timestep (sub-cycling, adaptive dt). The positions move by `dt_push * v` but the charge-conservation formula divides by `self.dt`, breaking `div(J)*dt + delta_rho = 0`.
- **Impact**: When `dt_push != self.dt`, the Esirkepov continuity equation is violated. Charge conservation broken. BLOCKING for any sub-cycling implementation (Bug 8 depends on this fix).
- **Fix**: Store the actual push dt and use it in deposit:
  ```python
  # In push_particles(), after dt resolution:
  self._last_push_dt = dt

  # In deposit(), line 1561:
  # BEFORE: self.grid_shape, self.dx, self.dy, self.dz, self.dt,
  # AFTER:  self.grid_shape, self.dx, self.dy, self.dz, self._last_push_dt,
  ```
- **LOC**: 5 (add attribute init, store dt, change reference)
- **Regression risk**: LOW -- when `dt == self.dt` (current behavior), output is identical
- **Dependency**: Must be fixed BEFORE Bug 8 (sub-cycling)
- **Test**: `test_esirkepov_dt_consistency` -- push with `dt=0.5*self.dt`, deposit, verify `div(J)*dt_push + delta_rho == 0` to 1e-10.

#### Bug 3: E-Field Missing Hall Term

- **Location**: `engine/core.py:871` -- `E_fld = -np.cross(v, B_fld)`
- **Root cause**: The MHD Ohm's law gives `E = -v x B + eta*J + J x B/(n_e*e) - grad(P_e)/(n_e*e)`. Lines 871-874 only include the convective term and resistive term. The Hall term `J x B/(n_e*e)` and electron pressure gradient are omitted.
- **Impact**: At pinch conditions (n_e ~ 10^25, B ~ 10T, J ~ 10^12 A/m^2), `E_Hall ~ J*B/(n_e*e) ~ 10^12 * 10 / (10^25 * 1.6e-19) ~ 6 kV/m`. Convective `E_conv ~ v*B ~ 10^5 * 10 ~ 10^6 V/m`. So Hall is ~0.6% of convective -- smaller than feared in the Six Sigma review. However, during early pinch when v is small and J/n_e is large, Hall can dominate. This is a physics accuracy issue, not a correctness bug.
- **Fix**: Add Hall term when available:
  ```python
  # After line 874:
  if J_field is not None:
      rho = self.state["rho"]
      m_i = 3.34e-27  # deuterium
      n_e = rho / m_i  # quasi-neutrality
      n_e = np.maximum(n_e, 1e18)  # vacuum floor
      ne_3d = n_e[..., np.newaxis]  # broadcast for cross product
      J_fld = np.moveaxis(J_field, 0, -1)
      E_hall = np.cross(J_fld, B_fld) / (ne_3d * 1.602e-19)
      E_fld = E_fld + E_hall
  ```
- **LOC**: 10
- **Regression risk**: MEDIUM -- changes particle trajectories. Must validate that I(t) is not perturbed.
- **Dependency**: Independent. Can be deferred to V3 phase. Requires Hall MHD scaffold (H1) for full correctness, but this simplified version works standalone.
- **Test**: `test_E_field_hall_term_magnitude` -- compute E_hall / E_conv ratio at pinch conditions. Verify it is O(0.01-0.1). `test_E_field_hall_included` -- verify E_fld differs from pure `E = -v x B` when J is non-zero.

#### Bug 4: No Particle Removal

- **Location**: `push_particles()` method, lines 1402-1488 -- no removal logic
- **Root cause**: Particles that thermalize (energy < thermal background) are never removed. They continue to be pushed, interpolated, and deposited every step.
- **Impact**: Performance degradation only. After 10^4 steps with continuous injection, particle count grows linearly. At 100 particles/step, that is 10^6 particles = 430 MB. Non-blocking but wastes compute.
- **Fix**: Add energy-based culling at the end of `push_particles()`:
  ```python
  # After sp.velocities = new_vel (line 1488):
  if self._collision_enabled:
      v_sq = np.sum(sp.velocities**2, axis=1)
      E_kin_eV = 0.5 * sp.mass * v_sq / 1.602e-19
      keep = E_kin_eV > self._T_background_eV  # above thermal
      if not np.all(keep):
          sp.positions = sp.positions[keep]
          sp.velocities = sp.velocities[keep]
          sp.weights = sp.weights[keep]
          sp.positions_old = sp.positions_old[keep]
  ```
- **LOC**: 12
- **Regression risk**: LOW -- only removes particles that contribute nothing to beam-target yield
- **Dependency**: None
- **Test**: `test_particle_removal_culls_thermalized` -- inject 100 particles at 1 eV into 100 eV background. After push, verify they are removed. `test_particle_removal_preserves_energetic` -- inject 100 at 100 keV. Verify all survive.

#### Bug 5: Reflecting BC Unphysical for DPF

- **Location**: `_apply_reflecting_bc()`, lines 1490-1519
- **Root cause**: Real DPF has conducting electrodes (absorbing) at anode (r=0) and cathode (r=R_outer), and open boundaries along the axis. Reflecting BC traps beam ions, overestimating confinement time by 2-5x.
- **Impact**: Yn overestimated. Direct impact on V4 validation against Gribkov (2007). HIGH priority for V4.
- **Fix**: Add BC mode selection:
  ```python
  def __init__(self, ..., bc_mode: str = "reflecting"):
      self.bc_mode = bc_mode

  def _apply_bc(self, positions, velocities):
      if self.bc_mode == "absorbing":
          return self._apply_absorbing_bc(positions, velocities)
      return self._apply_reflecting_bc(positions, velocities)

  def _apply_absorbing_bc(self, positions, velocities):
      """Remove particles that exit the domain."""
      inside = np.ones(len(positions), dtype=bool)
      limits = [self._Lx, self._Ly, self._Lz]
      for d in range(3):
          inside &= (positions[:, d] >= 0.0) & (positions[:, d] <= limits[d])
      return positions[inside], velocities[inside]
      # NOTE: caller must also filter weights and positions_old
  ```
- **LOC**: 25 (new method + wiring + weight/positions_old filtering)
- **Regression risk**: MEDIUM -- changes particle count dynamically. Must update deposit() to handle shrinking arrays.
- **Dependency**: None, but the weight/positions_old filtering requires touching `push_particles()` loop structure.
- **Test**: `test_absorbing_bc_removes_escaped` -- place particle outside domain, verify removed. `test_absorbing_bc_preserves_interior` -- verify interior particles unchanged. `test_reflecting_vs_absorbing_yn` -- compare yield with both BCs (absorbing should give lower Yn).

#### Bug 6: Binary Collision Self-Pairing Unphysical

- **Location**: `hybrid.py:1462-1478`
- **Root cause**: Lines 1471-1478 call `_nanbu_scatter_kernel(new_vel, new_vel, sp.weights, sp.weights, sp.mass, sp.mass, ...)`. Nanbu-Perez (2012) is designed for inter-species collisions where species a and b have different distribution functions. For self-collisions (a == b), Nanbu (1997) Section IV specifies random pairing with weight correction: randomly shuffle indices, pair (0,1), (2,3), etc. The current implementation pairs particle i with particle i (via array index), which is not random pairing.
- **Impact**: Coupled with Bug 1 (in-place mutation). Self-collision thermalization rate is systematically biased. The Nanbu self-collision variant gives correct relaxation rate; the current code does not.
- **Fix**: Use random pairing for self-collisions:
  ```python
  if self.use_binary_collisions:
      n_sp = self._n_background
      ln_lam = ...  # existing code
      cell_vol = self.dx * self.dy * self.dz
      # Random pairing for self-collisions
      idx = np.random.permutation(len(new_vel))
      vel_shuffled = new_vel[idx].copy()
      wt_shuffled = sp.weights[idx].copy()
      _nanbu_scatter_kernel(
          new_vel, vel_shuffled,
          sp.weights, wt_shuffled,
          sp.mass, sp.mass,
          sp.charge, sp.charge,
          n_sp, n_sp,
          ln_lam, dt, cell_vol,
      )
  ```
- **LOC**: 8 (add permutation + copy)
- **Regression risk**: LOW -- changes scattering details but not the overall framework
- **Dependency**: Supersedes Bug 1 fix (if Bug 6 is fixed with random pairing + copy, Bug 1 is automatically resolved)
- **Test**: `test_self_collision_random_pairing_thermalization` -- mono-energetic beam should thermalize to Maxwellian. Chi-squared test on velocity distribution after 5*tau_slow. Compare rate against analytical Spitzer thermalization time.

#### Bug 7: Default `weight_total` Device-Independent

- **Location**: `inject_beam()` line 1586 -- `weight_total: float = 1e16`
- **Root cause**: Default 1e16 is hardcoded regardless of device. PF-1000 at 27 kV has I_pinch ~ 500 kA for ~100 ns, giving ~3e17 ions via `N = I * t / q`. Default underestimates by 30x.
- **Impact**: Deposited J_kin is 30x too small. MHD feedback negligible even if coupling is correct. Non-blocking for V1/V2 (unit tests use explicit weights) but makes V4 validation meaningless.
- **Fix**: Compute default from device parameters:
  ```python
  def inject_beam(self, ..., weight_total: float | None = None):
      if weight_total is None:
          # Estimate from pinch current and duration
          # I_pinch ~ 500 kA, tau_pinch ~ 100 ns for PF-1000
          # N_ions = I * tau / q
          weight_total = 1e16  # fallback
      ...
  ```
  Better: pass from KineticManager which has access to circuit state (I_pinch, pinch duration).
- **LOC**: 5 (change default to None, add fallback)
- **Regression risk**: ZERO if default fallback equals current value
- **Dependency**: Full fix depends on KineticManager wiring (not in scope for V1)
- **Test**: `test_beam_weight_matches_pinch_current` -- given I=500kA, tau=100ns, verify weight_total ~ 3e17.

#### Bug 8: No PIC Sub-Cycling

- **Location**: `push_particles()` method, lines 1402-1488 -- single Boris push per call
- **Root cause**: MHD dt ~ 1e-9 s, but for 100 keV deuterons in 10 T, cyclotron period T_c ~ 6.5e-9 s. Need ~30 steps per gyroperiod, so dt_pic ~ 2e-10 s. Sub-cycle ratio N = ceil(dt_mhd / dt_pic) ~ 5. Without sub-cycling, Boris push takes one giant step that under-resolves gyration, introducing artificial energy drift and incorrect trajectories.
- **Impact**: BLOCKING. V3/V4 produce garbage without this. Boris integrator is only symplectic when dt << T_c. At dt = T_c (no sub-cycling), energy error per step is O((omega_c * dt)^2) ~ O(1), i.e., total energy conservation breaks.
- **Fix**: Add sub-cycling loop inside `push_particles()`:
  ```python
  def push_particles(self, E, B, dt=None):
      if dt is None:
          dt = self.dt

      # Compute PIC sub-cycling
      for sp in self.species:
          if sp.n_particles() == 0:
              continue

          # Sub-cycle computation
          omega_c = abs(sp.charge) * np.max(np.linalg.norm(B_at_p, axis=-1)) / sp.mass
          if omega_c > 0:
              dt_pic = min(dt, 2.0 * np.pi / (omega_c * 30.0))  # 30 steps/gyro
              n_sub = max(1, min(int(np.ceil(dt / dt_pic)), 50))  # cap at 50
          else:
              n_sub = 1
              dt_pic = dt

          # ... interpolation ...

          for _ in range(n_sub):
              E_at_p = interpolate_field_to_particles(E, sp.positions, ...)
              B_at_p = interpolate_field_to_particles(B, sp.positions, ...)
              sp.positions_old = sp.positions.copy()
              new_pos, new_vel = boris_push(
                  sp.positions, sp.velocities, E_at_p, B_at_p,
                  sp.charge, sp.mass, dt_pic,
              )
              new_pos, new_vel = self._apply_reflecting_bc(new_pos, new_vel)
              sp.positions = new_pos
              sp.velocities = new_vel

          # Store actual sub-step dt for Esirkepov
          self._last_push_dt = dt_pic

          # Collisions use full dt (collision is operator-split)
          # ... collision code here with dt, not dt_pic ...
  ```
- **LOC**: 25 (loop structure + omega_c computation + dt_pic + cap)
- **Regression risk**: MEDIUM -- changes the push loop structure. Must verify that n_sub=1 (weak B) reproduces current behavior exactly.
- **Dependency**: Requires Bug 2 fix first (Esirkepov dt mismatch)
- **Test**: `test_subcycling_energy_conservation` -- 100 keV deuteron in 10 T for 100 gyroperiods. With sub-cycling: energy drift < 1e-8. Without: energy drift > 0.1. `test_subcycling_n_sub_computation` -- verify n_sub=5 for dt=1ns, B=10T, m=m_D. `test_subcycling_cap_at_50` -- verify n_sub capped at 50 even if omega_c is extreme.

---

### M -- Measure: Fix Effort and Dependency Matrix

#### LOC and Risk Summary

| Bug # | Description | LOC | Risk | Priority | Blocks |
|-------|-------------|-----|------|----------|--------|
| 1 | Nanbu self-collision in-place mutation | 3 | LOW | HIGH | V3 collision accuracy |
| 2 | Esirkepov dt mismatch | 5 | LOW | **CRITICAL** | Bug 8 (sub-cycling) |
| 3 | E-field missing Hall term | 10 | MEDIUM | MEDIUM | V3 trajectory accuracy |
| 4 | No particle removal | 12 | LOW | LOW | Performance only |
| 5 | Reflecting BC unphysical | 25 | MEDIUM | HIGH (for V4) | V4 Yn validation |
| 6 | Self-collision random pairing | 8 | LOW | HIGH | V3 collision accuracy |
| 7 | weight_total device-independent | 5 | ZERO | LOW (for V1) | V4 J_kin magnitude |
| 8 | No PIC sub-cycling | 25 | MEDIUM | **CRITICAL** | V3, V4 entirely |
| **Total** | | **93** | | | |

#### Fix Dependency Graph

```
Bug 2 (Esirkepov dt) ──────────────> Bug 8 (sub-cycling)
                                         |
                                         v
Bug 6 (random pairing) ──supersedes──> Bug 1 (in-place mutation)
                                         |
                                         v
                                    V3 collision tests
                                         |
Bug 5 (absorbing BC) ──────────────> V4 Yn validation
Bug 7 (weight scaling) ────────────> V4 J_kin magnitude
Bug 3 (Hall E-field)  ────────────> V3 trajectory accuracy
Bug 4 (particle removal) ─────────> Performance (independent)

Dependency chains:
  Chain A: Bug 2 -> Bug 8 -> V3/V4 (CRITICAL PATH, 30 LOC)
  Chain B: Bug 6 -> (Bug 1 resolved) -> V3 collisions (8 LOC)
  Chain C: Bug 5 -> V4 Yn (25 LOC)
  Independent: Bug 3, Bug 4, Bug 7
```

---

### A -- Analyze: Compounding Bugs and Minimum Viable Fix Set

#### Compounding Bug Interactions

1. **Bug 2 + Bug 8 (dt mismatch + no sub-cycling)**: These compound multiplicatively. Without sub-cycling, Boris push uses dt_mhd ~ 1ns which under-resolves gyration by 5x. If sub-cycling is added but Bug 2 is unfixed, Esirkepov uses self.dt (1ns) while positions moved by dt_pic (0.2ns) -- the charge conservation factor `q/dt` is 5x wrong, and deposited J is 5x too large. Net effect: doubly wrong trajectories AND charge non-conservation. **Must fix Bug 2 first, then Bug 8.**

2. **Bug 1 + Bug 6 (in-place mutation + no random pairing)**: These compound. In-place mutation means early particles see pre-scatter velocities and late particles see post-scatter. Without random pairing, particles are always paired by index (0 with 0, which is self). Combined: every particle scatters off its own already-modified velocity, which is physically nonsensical. **Bug 6 fix (random pairing with copy) resolves both simultaneously.**

3. **Bug 5 + Bug 7 (reflecting BC + wrong weight)**: Reflecting BC overestimates confinement (2-5x too many particles in domain). Low weight_total underestimates each particle's contribution (30x too low). These partially compensate: 5x too many particles * 30x too little weight per particle = still 6x too low total current. **Both must be fixed for V4.**

4. **Bug 3 + Bug 8 (wrong E + wrong dt)**: Wrong E-field directs particles along wrong trajectories. Wrong dt means each step's error is amplified. These don't cancel -- they compound as uncorrelated errors adding in quadrature. At pinch: ~5% trajectory error from E-field + ~50% energy error from no sub-cycling = sub-cycling dominates.

#### Minimum Viable Fix Set for V1 (Unit Tests Passing)

V1 tests are standalone kernel tests. No coupling, no MHD, no sub-cycling needed. **Zero bug fixes required for V1.** All 28 unit tests can be written and pass against the current code because:
- Unit tests test kernels in isolation with controlled inputs
- Esirkepov dt test will EXPOSE Bug 2 (test written to verify the bug, then fix applied)
- Self-collision tests will EXPOSE Bugs 1+6

**Recommended approach**: Write V1 tests first, use them to expose bugs, then fix.

#### Minimum Viable Fix Set for V2 (Integration Tests Passing)

V2 tests (gyration, two-stream, thermalization) need:
- Bug 8 (sub-cycling) for the 1000-gyroperiod test at strong B
- Bug 6 (random pairing) for thermalization test correctness

**V2 minimum fixes**: Bug 2 + Bug 8 + Bug 6 = 38 LOC

#### Minimum Viable Fix Set for V3 (MHD-Coupled Tests)

All V2 fixes plus:
- Bug 3 (Hall E-field) for trajectory accuracy at pinch

**V3 minimum fixes**: V2 fixes + Bug 3 = 48 LOC

#### Minimum Viable Fix Set for V4 (End-to-End)

All V3 fixes plus:
- Bug 5 (absorbing BC) for realistic confinement
- Bug 7 (weight scaling) for correct J_kin magnitude

**V4 minimum fixes**: V3 fixes + Bug 5 + Bug 7 = 78 LOC

---

### I -- Improve: Updated Implementation Plan

#### Revised Critical Path

```
V1 Unit Tests (write tests, expose bugs)
  |  [1-2 days, 500 LOC tests]
  v
Bug Fix Sprint A: Bug 2 (5 LOC) + Bug 6 (8 LOC) + Bug 8 (25 LOC)
  |  [1 day, 38 LOC production + 40 LOC tests for fixes]
  v
V2 Integration Tests
  |  [2-3 days, 350 LOC tests]
  v
Bug Fix Sprint B: Bug 3 (10 LOC) + Bug 5 (25 LOC) + Bug 7 (5 LOC)
  |  [1 day, 40 LOC production + 30 LOC tests]
  v
V3 MHD-Coupled Tests
  |  [2-3 days, 300 LOC tests]
  v
Bug Fix Sprint C: Bug 4 (12 LOC) -- performance only, lowest priority
  |  [0.5 days]
  v
V4 End-to-End Tests
  |  [2-3 days, 250 LOC tests]
  v
DONE
```

#### Revised Time Estimate

| Phase | Original | Revised | Confidence Interval |
|-------|----------|---------|---------------------|
| V1 (unit tests) | 1-2 days | 1-2 days | HIGH (no code changes needed) |
| Bug Sprint A | (not in original) | 1 day | HIGH (38 LOC, clear specs) |
| V2 (integration) | 2-3 days | 2-3 days | MEDIUM (Poisson solver needed) |
| Bug Sprint B | (not in original) | 1 day | MEDIUM (Hall term needs validation) |
| V3 (MHD-coupled) | 2-3 days | 3-4 days | LOW (MHD coupling is complex) |
| Bug Sprint C | (not in original) | 0.5 days | HIGH (simple filter) |
| V4 (end-to-end) | 1-2 days | 2-3 days | LOW (first-ever run, unknowns) |
| **Total** | **7-11 days** | **12-16 days** | P50=14, P90=18 |

The Six Sigma review's estimate of 12-18 days is confirmed. The original 7-11 days excluded bug fix time and underestimated V3/V4 integration complexity.

#### Simpler Boris Validation: Uniform E-field Drift

Landau damping requires a Poisson solver (~80 LOC). A simpler Boris validation that requires zero infrastructure:

**E x B Drift Test** (~15 LOC):
```python
def test_ExB_drift():
    """Single particle in crossed E and B drifts at v_D = E x B / B^2."""
    E = np.array([[0.0, 1e4, 0.0]])   # Ey = 10 kV/m
    B = np.array([[0.0, 0.0, 1.0]])    # Bz = 1 T
    v_drift_expected = 1e4 / 1.0       # v_Dx = Ey/Bz = 10 km/s

    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0]])  # start at rest
    dt = 1e-9
    for _ in range(1000):
        pos, vel = boris_push(pos, vel, E, B, Q_E, M_D, dt)

    # After transient gyration, net drift in x-direction
    v_drift_measured = pos[0, 0] / (1000 * dt)
    assert v_drift_measured == pytest.approx(v_drift_expected, rel=0.05)
```

This validates Boris push correctness (E x B drift is exact for Boris at any dt/omega_c) without requiring any solver infrastructure. It complements the existing gyration tests by testing the electric field response.

**Magnetic mirror test** (~20 LOC):
```python
def test_magnetic_mirror():
    """Particle in converging B reflects at mirror point."""
    # Uniform Bz with gradient: B = B0 * (1 + z/L)
    # Particle with pitch angle > mirror angle is reflected
    # mu = m*v_perp^2 / (2*B) conserved
    # This is too complex for a unit test; skip for V1.
```

Recommendation: Use E x B drift as the "10-line Boris validation" for V1. Defer magnetic mirror to V2.

#### Sub-Cycling Wrapper (5-Line Version)

The minimal sub-cycling can indeed be a wrapper:

```python
def push_particles_subcycled(self, E, B, dt_mhd):
    """Wrapper: sub-cycle Boris push within MHD timestep."""
    for sp in self.species:
        if sp.n_particles() == 0:
            continue
        B_max = np.max(np.linalg.norm(
            interpolate_field_to_particles(B, sp.positions, self.dx, self.dy, self.dz),
            axis=1,
        ))
        omega_c = abs(sp.charge) * B_max / sp.mass if B_max > 0 else 0
        n_sub = max(1, min(int(np.ceil(omega_c * dt_mhd / (2 * np.pi / 30))), 50))
        dt_pic = dt_mhd / n_sub
        for _ in range(n_sub):
            self.push_particles(E, B, dt=dt_pic)
```

This is 10 lines, not 5, because the omega_c computation requires field interpolation. But it can be simplified to 5 lines if we pre-compute omega_c from MHD state (B_max on grid):

```python
def push_subcycled(self, E, B, dt_mhd, B_max):
    omega_c = self.species[0].charge * B_max / self.species[0].mass
    n_sub = max(1, min(int(omega_c * dt_mhd * 30 / (2 * np.pi)), 50))
    dt_pic = dt_mhd / n_sub
    for _ in range(n_sub):
        self.push_particles(E, B, dt=dt_pic)
```

5 lines. Uses B_max from the MHD grid (already available). Assumes single species (DPF beam ions). Cap at 50 sub-cycles.

---

### C -- Control: Regression Gates

#### CI Test Gates Per Fix

| Bug Fix | Gate Test | Pass Criterion | Run After |
|---------|-----------|----------------|-----------|
| Bug 2 (Esirkepov dt) | `test_esirkepov_continuity_equation` | `div(J)*dt + delta_rho < 1e-10` for any dt | Every commit to hybrid.py |
| Bug 6 (random pairing) | `test_nanbu_self_collision_no_ordering_bias` | RMS angle agrees with reference within 5% | Every commit to hybrid.py |
| Bug 8 (sub-cycling) | `test_subcycling_energy_conservation` | Energy drift < 1e-8 over 100 gyroperiods | Every commit to hybrid.py |
| Bug 3 (Hall E) | `test_E_field_hall_included` | E_fld != -v x B when J != 0 | Every commit to engine/core.py |
| Bug 5 (absorbing BC) | `test_absorbing_bc_removes_escaped` | Escaped particles removed | Every commit to hybrid.py |
| Bug 7 (weight) | `test_beam_weight_matches_pinch_current` | weight_total = I*tau/q within 10% | Every commit to hybrid.py |
| Bug 4 (removal) | `test_particle_removal_culls_thermalized` | Particles below thermal energy removed | Every commit to hybrid.py |

#### Existing Test Regression Gate

All 24 existing PIC tests must remain green:
- `test_pic_hybrid.py` (7 tests)
- `test_pic_validation.py` (12 tests)
- `test_beam_tracker.py` (5 tests)

Command: `pytest tests/test_pic_hybrid.py tests/test_pic_validation.py tests/test_beam_tracker.py -v`

#### Phase-Level Quality Gates

| Phase | Gate | Command |
|-------|------|---------|
| V1 complete | 28 new unit tests pass + 24 existing green | `pytest tests/test_pic_*.py -v` |
| Bug Sprint A | Esirkepov continuity + sub-cycling energy + collision pairing | `pytest tests/test_pic_*.py -k "esirkepov or subcycl or collision" -v` |
| V2 complete | 6 integration tests pass, gyration energy < 1e-8 | `pytest tests/test_pic_integ_*.py -v` |
| Bug Sprint B | Hall E-field test + absorbing BC + weight scaling | `pytest tests/test_pic_*.py -k "hall or absorbing or weight" -v` |
| V3 complete | 6 MHD-coupled tests, total energy < 5% | `pytest tests/test_pic_mhd_*.py -v` |
| V4 complete | 3 E2E tests, Yn in [1e10, 1e12], completes without crash | `pytest tests/test_pic_e2e_*.py -v -m slow` |

---

### Research Findings

#### Published PIC-DPF Results for Benchmarking

Search of `docs/research-reference/` and the research database found limited PIC-DPF references:

1. **Schmidt et al., PRL 109:205003 (2012)**: Fully kinetic LSP simulation of 1 MJ DPF. Key result: Yn ~ 3e11, beam-target dominates over thermonuclear by 10-100x. This is our primary benchmark for V4.

2. **Pasternak et al. (2024)**: PF-1000 PIC simulation. Referenced in scaffold Section 5.1 but no specific data extracted. Need to digitize beam energy spectra for comparison.

3. **Damideh (2025)**: FAETON-I PIC-MHD hybrid geometry. Referenced in `ANALYSIS_REPORT.md` as a gap: "No kinetic effects in any MHD backend." Our PIC module addresses this gap.

4. **Auluck (2023)**: Explains Yn ~ I^2.96 scaling (not I^4) because beam-target saturates at high current. Our validation should reproduce this sub-quartic scaling.

5. No Python-based PIC-DPF code exists in the open-source ecosystem. The closest references are Chicago/LSP (commercial, Fortran/C++) and EPOCH (open-source PIC, but not DPF-specific). Our implementation is novel.

#### Simpler PIC Tests Than Landau Damping

In addition to E x B drift (described above), three other zero-infrastructure tests:

1. **Uniform B gyration** (already implemented in test_pic_validation.py) -- validates Boris push core.

2. **E x B drift** (15 LOC, described above) -- validates E-field response. Exact for Boris at any dt.

3. **Gradient-B drift** (25 LOC) -- particle in non-uniform B drifts perpendicular to both B and grad(B). Validates field interpolation accuracy. Requires a field with known gradient.

4. **Particle-in-uniform-field energy conservation** (10 LOC) -- purely magnetic field, verify |v| is constant to machine precision after 10^4 steps. This is the simplest possible Boris validation.

Recommendation: Add tests 2 and 4 to V1 (25 LOC total). These provide stronger Boris validation than the existing gyration tests with zero infrastructure overhead.

#### Sub-Cycling Pattern from Existing Codebase

The resistive diffusion sub-cycling pattern in `mlx_sources.py` uses:
```python
N = ceil(dt_mhd / dt_res)
N = min(N, 20)  # cap
dt_sub = dt_mhd / N
for _ in range(N):
    U = apply_diffusion(U, dt_sub)
```

The PIC sub-cycling is structurally identical. The 5-line wrapper described above follows this pattern exactly, with the cap raised to 50 (PIC sub-cycles are cheaper than diffusion because they operate on particles, not the full grid).

---

### Summary: Path from 5/10 to 7/10

| Action | Readiness Gain | Rationale |
|--------|---------------|-----------|
| Write V1 unit tests (28 tests) | +0.5 | Establishes test baseline, exposes bugs |
| Fix Bug 2 + Bug 8 (Esirkepov dt + sub-cycling) | +0.5 | Unblocks V2/V3/V4 entirely |
| Fix Bug 6 (random pairing, supersedes Bug 1) | +0.25 | Collision physics correctness |
| Write V2 integration tests (6 tests) | +0.25 | Classical PIC benchmarks pass |
| Fix Bug 5 (absorbing BC) | +0.25 | Realistic confinement for V4 |
| Revised time estimate (12-16 days) | +0.25 | Honest scheduling = higher confidence |
| **Total** | **+2.0** | **5/10 -> 7/10** |

Remaining items for 7/10 -> 9/10 (future):
- Fix Bug 3 (Hall E-field) -- requires Hall MHD scaffold
- Fix Bug 7 (weight scaling) -- requires KineticManager wiring
- Fix Bug 4 (particle removal) -- performance only
- V3 + V4 tests passing
- Particle count convergence study (10K -> 100K -> 1M)
- Comparison against Schmidt (2012) Yn values
