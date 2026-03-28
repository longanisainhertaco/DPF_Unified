# Boris Vacuum Correction: Calibration Impact DMAIC Investigation

**Date**: 2026-03-27
**Sprint**: S-1 post-mortem
**Estimated effort**: ~3 hours (single engineer, single session)
**Status**: PLANNED

---

## DEFINE

### Investigation Question

Sprint S-1 replaced the density-floor vacuum stabilization (`rho = max(rho, B^2/va_max^2)`)
with Boris vacuum correction (`f_boris = c_boris^2 / (v_A^2 + c_boris^2)`) in three locations:
1. Geometric source terms (`mlx_kernels.py:938-951`, Metal kernel + NumPy reference)
2. Wave speed computation (`mlx_primitives.py:294-355`, `fast_magnetosonic_boris()`)
3. Velocity clamping / floor enforcement (`mlx_timestepper.py:65-111`, `_stage_post_impl()`)

This shifted the optimal calibration parameters:

| Parameter | Pre-Boris | Post-Boris | Shift |
|-----------|-----------|------------|-------|
| fc | 0.797 | 0.649 | -18.6% |
| fm | 0.084 | 0.062 | -26.2% |
| I_peak error | 6.4% | 4.8% | improved |
| t_peak error | 10.0% | 11.9% | degraded |
| NRMSE | -- | 0.147 | -- |
| Objective | -- | 0.099 | -- |

The fc*fm product shifted from 0.0669 to 0.0404 (-40%). This is not a small perturbation.

### Three Hypotheses

- **A (better physics)**: Boris removes fake mass injection that inflated the effective
  current-coupling fraction. Lower fc is the true physical value because the solver no
  longer needs to compensate for artificial mass in vacuum cells behind the sheath.
- **B (new error)**: Boris over-reduces magnetic pressure at the sheath front (where
  v_A is legitimately high), weakening snowplow compression, forcing the optimizer to
  compensate with lower fc to reduce sheath velocity.
- **C (insufficient convergence)**: 30 Optuna trials (7.3 min wall time) may not have
  explored the basin adequately. The pre-Boris optimum used 69 trials. The landscape
  may be multi-modal.

### Exit Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Convergence check | Two starting points converge to same basin (|delta_fc| < 0.02) | Gradient refinement from both seeds |
| Waveform quality | NRMSE improves OR stays within 0.01 | Full-waveform comparison vs Scholz data |
| Mass conservation | Relative mass error < 1e-6 at pinch time | Conservation monitor output |
| Hypothesis selected | One hypothesis clearly supported by >= 3/5 measurements | Decision matrix below |

### What "Better Physics" Looks Like

If hypothesis A is correct, we expect:
1. Gradient refinement from fc=0.797 descends toward fc~0.65 (not stuck at 0.797)
2. The post-Boris waveform matches the rundown slope better (density floor was adding fake mass during rundown)
3. Mass is conserved better with Boris (no artificial mass injection)
4. Species fractions (s_rho / rho) stay closer to 1.0 in vacuum cells
5. The fc*fm product is NOT an invariant (shifts because the physics changed, not because of parameter degeneracy)

---

## MEASURE

### M1: Dual-Seed Gradient Refinement

**Purpose**: Test whether the two optima are distinct basins or the same basin found from different starting points.

**Script**:
```bash
# Seed 1: pre-Boris optimum
python3 scripts/calibrate_gradient.py --device pf1000 --fc0 0.797 --fm0 0.084 \
    --max-iters 15 --tol 0.001

# Seed 2: post-Boris optimum
python3 scripts/calibrate_gradient.py --device pf1000 --fc0 0.649 --fm0 0.062 \
    --max-iters 15 --tol 0.001
```

**Runtime**: ~15 min per seed (each iter = 5 forward evals, ~15 sec each, 15 iters).

**Decision criteria**:

| Outcome | Supports |
|---------|----------|
| Both converge to fc ~ 0.65 +/- 0.02 | A (single basin, Boris found the true minimum) |
| Both converge to fc ~ 0.80 +/- 0.02 | B (fc=0.649 was a local minimum, true physics needs higher fc) |
| They converge to different basins (|delta_fc| > 0.05) | C (landscape is multi-modal, need more trials) |
| Seed 1 oscillates / fails to converge | B or C (gradient is near-zero plateau near 0.797) |

### M2: Full-Waveform NRMSE Decomposition

**Purpose**: Identify WHERE in the I(t) waveform each calibration disagrees with experiment.
The rundown phase (0-3 us), radial phase (3-5 us), and post-pinch (>5 us) probe different
physics. If Boris fixes the rundown slope, that confirms fake mass was corrupting early dynamics.

**Script**:
```python
#!/usr/bin/env python3
"""M2: Waveform shape decomposition for Boris calibration study."""
import numpy as np
from dpf.validation.mlx_calibration import run_mlx_forward_model
from dpf.validation.experimental import DEVICES

# Run both calibrations
r_old = run_mlx_forward_model(fc=0.797, fm=0.084, preset_name="pf1000")
r_new = run_mlx_forward_model(fc=0.649, fm=0.062, preset_name="pf1000")

# Load experimental waveform
exp = DEVICES["pf1000"]
t_exp, I_exp = exp.time_us, exp.current_MA

# Interpolate both to experimental time base and compute phase-by-phase NRMSE
# Phase 1: rundown (t < 3 us)
# Phase 2: radial (3 us < t < 5 us)
# Phase 3: post-pinch (t > 5 us)
# Report per-phase NRMSE for both calibrations
```

Save as: `scripts/boris_waveform_decomp.py`

**Runtime**: ~30 sec (2 forward evals).

**Decision criteria**:

| Outcome | Supports |
|---------|----------|
| fc=0.649 has lower rundown NRMSE, similar radial NRMSE | A (Boris fixed the rundown mass error) |
| fc=0.649 has higher radial NRMSE, lower elsewhere | B (Boris weakens radial compression) |
| Both have similar per-phase NRMSE | C (indeterminate, need finer metrics) |

### M3: Mass Conservation Audit

**Purpose**: Quantify whether Boris actually improves mass conservation relative to the
density-floor approach. The old density floor injected mass by setting `rho = max(rho, B^2/va_max^2)`.
Boris should eliminate this artificial mass source.

**Script**:
```python
#!/usr/bin/env python3
"""M3: Mass conservation comparison -- Boris vs density floor."""
import numpy as np
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.presets import get_preset

def run_with_mass_tracking(fc, fm, label):
    preset = get_preset("pf1000")
    preset["snowplow"]["current_fraction"] = fc
    preset["snowplow"]["mass_fraction"] = fm
    preset["fluid"]["backend"] = "mlx"
    preset["fluid"]["riemann_solver"] = "hlls"
    preset["fluid"]["reconstruction"] = "plm"
    preset["fluid"]["time_integrator"] = "ssp_rk2"

    config = SimulationConfig.from_preset(preset)
    engine = SimulationEngine(config)
    engine.initialize()

    # Track total mass at each output step
    masses = []
    times = []
    for step in range(5000):
        engine.step()
        if step % 500 == 0:
            state = engine.get_state()
            rho = state["rho"]
            # Volume-weighted mass integral (cylindrical: dV = 2*pi*r*dr*dz)
            r = engine.solver.grid.r_cell
            dr = engine.solver.grid.dr
            dz = engine.solver.grid.dz
            mass = 2 * np.pi * np.sum(rho * r[:, np.newaxis] * dr * dz)
            masses.append(mass)
            times.append(engine.time)
    return times, masses

t_old, m_old = run_with_mass_tracking(0.797, 0.084, "pre-Boris")
t_new, m_new = run_with_mass_tracking(0.649, 0.062, "post-Boris")

# Compare: delta_m / m_0 for each
print(f"Pre-Boris mass drift:  {(m_old[-1] - m_old[0]) / m_old[0]:.2e}")
print(f"Post-Boris mass drift: {(m_new[-1] - m_new[0]) / m_new[0]:.2e}")
```

**Runtime**: ~2 min (2 x 5000 steps at ~15 ms/step).

**Decision criteria**:

| Outcome | Supports |
|---------|----------|
| Post-Boris mass drift < pre-Boris by > 10x | A (Boris genuinely improves conservation) |
| Both have similar mass drift | Neither (mass conservation is not the discriminator) |
| Post-Boris mass drift is WORSE | B (Boris introduced a new conservation issue) |

### M4: Species Contamination Check

**Purpose**: The density floor hack set `rho = max(rho, B^2/va_max^2)` which injected
unphysical mass into vacuum cells. This corrupted the species tracer `s_rho` (ISR slot)
because `s_rho / rho` should equal the initial species fraction everywhere.
Boris should eliminate this contamination.

**Script**:
```python
#!/usr/bin/env python3
"""M4: Species fraction contamination in vacuum cells."""
import numpy as np
from dpf.engine import SimulationEngine
from dpf.config import SimulationConfig
from dpf.presets import get_preset

def species_contamination(fc, fm):
    preset = get_preset("pf1000")
    preset["snowplow"]["current_fraction"] = fc
    preset["snowplow"]["mass_fraction"] = fm
    preset["fluid"]["backend"] = "mlx"
    preset["fluid"]["riemann_solver"] = "hlls"
    preset["fluid"]["reconstruction"] = "plm"
    preset["fluid"]["time_integrator"] = "ssp_rk2"

    config = SimulationConfig.from_preset(preset)
    engine = SimulationEngine(config)
    engine.initialize()

    for _ in range(2000):
        engine.step()

    state = engine.get_state()
    rho = state["rho"]
    # Access raw conserved state for entropy tracer
    U = engine.solver.U  # (NVAR, nr, nz) MLX array
    s_rho = np.array(U[5])  # ISR = 5
    rho_arr = np.array(U[0])

    # Species fraction in vacuum cells (rho < 1e-4 * rho_max)
    rho_max = rho_arr.max()
    vacuum_mask = rho_arr < 1e-4 * rho_max
    s_frac_vac = s_rho[vacuum_mask] / np.maximum(rho_arr[vacuum_mask], 1e-30)

    # Should be ~1.0 if species is pure deuterium everywhere
    return np.std(s_frac_vac), np.mean(np.abs(s_frac_vac - 1.0))

std_old, mae_old = species_contamination(0.797, 0.084)
std_new, mae_new = species_contamination(0.649, 0.062)
print(f"Pre-Boris vacuum species std:  {std_old:.4e}, MAE from 1.0: {mae_old:.4e}")
print(f"Post-Boris vacuum species std: {std_new:.4e}, MAE from 1.0: {mae_new:.4e}")
```

**Runtime**: ~1 min.

**Decision criteria**:

| Outcome | Supports |
|---------|----------|
| Post-Boris species std < pre-Boris by > 5x | A (Boris preserves species purity) |
| Similar species contamination | Neither (species tracer not the main effect) |

### M5: Landscape Flatness / Parameter Coupling

**Purpose**: Determine if the loss landscape is flat (degenerate) along a fc-fm curve,
which would mean the optimizer found a different point on the same valley rather than
a different valley.

**Script**:
```python
#!/usr/bin/env python3
"""M5: Loss landscape cross-section along fc*fm = const."""
import numpy as np
from dpf.validation.mlx_calibration import run_mlx_forward_model

# Cross-section 1: along old fc*fm product (0.0669)
product_old = 0.797 * 0.084  # 0.0669
fcs = np.linspace(0.55, 0.85, 7)
losses_old_product = []
for fc in fcs:
    fm = product_old / fc
    if 0.03 <= fm <= 0.20:
        r = run_mlx_forward_model(fc=fc, fm=fm, preset_name="pf1000")
        losses_old_product.append((fc, fm, r.objective, r.peak_error, r.timing_error))
        print(f"fc={fc:.3f} fm={fm:.4f} obj={r.objective:.4f}")

# Cross-section 2: along new fc*fm product (0.0404)
product_new = 0.649 * 0.062  # 0.0404
losses_new_product = []
for fc in fcs:
    fm = product_new / fc
    if 0.03 <= fm <= 0.20:
        r = run_mlx_forward_model(fc=fc, fm=fm, preset_name="pf1000")
        losses_new_product.append((fc, fm, r.objective, r.peak_error, r.timing_error))
        print(f"fc={fc:.3f} fm={fm:.4f} obj={r.objective:.4f}")

# Cross-section 3: fc scan at fixed fm=0.062
losses_fc_scan = []
for fc in np.linspace(0.50, 0.85, 8):
    r = run_mlx_forward_model(fc=fc, fm=0.062, preset_name="pf1000")
    losses_fc_scan.append((fc, 0.062, r.objective, r.peak_error, r.timing_error))
    print(f"fc={fc:.3f} fm=0.062 obj={r.objective:.4f}")

# Cross-section 4: fm scan at fixed fc=0.649
losses_fm_scan = []
for fm in np.linspace(0.03, 0.15, 7):
    r = run_mlx_forward_model(fc=0.649, fm=fm, preset_name="pf1000")
    losses_fm_scan.append((0.649, fm, r.objective, r.peak_error, r.timing_error))
    print(f"fc=0.649 fm={fm:.3f} obj={r.objective:.4f}")
```

**Runtime**: ~7 min (30 forward evals at ~15 sec each).

**Decision criteria**:

| Outcome | Supports |
|---------|----------|
| Loss is flat along fc*fm = const (variation < 0.01) | C (parameter degeneracy, fc shift is an optimizer artifact) |
| Loss has clear minimum at fc ~ 0.65 regardless of cross-section | A (true optimum shifted) |
| Loss has clear minimum at fc ~ 0.80 along fc*fm=0.067 | B (old optimum was correct, Boris degraded physics) |
| Multiple local minima visible | C (landscape is multi-modal) |

---

## ANALYZE

### Fishbone: What Else Changed Between Phase Q and Sprint S-1?

| Category | Factor | Changed? | Impact Assessment |
|----------|--------|----------|-------------------|
| **Machine** | Grid resolution (32x64) | NO | Same grid for both calibrations |
| **Machine** | MLX version / Metal driver | POSSIBLY | Minor; same Apple Silicon, same float32 |
| **Method** | Riemann solver (HLLS) | NO | Same for both |
| **Method** | Reconstruction (PLM) | NO | Same for both |
| **Method** | Time integrator (SSP-RK2) | NO | Same for both |
| **Method** | Density floor in `_stage_post_impl` | **YES** | Old: `rho = max(rho, B^2/va_max^2)`. New: `rho = max(rho, RHO_FLOOR=1e-12)`. This is the primary change. |
| **Method** | Wave speed computation (CFL) | **YES** | Old: `fast_magnetosonic()`. New: `fast_magnetosonic_boris()`. Bounds wave speeds at 5e5 m/s. |
| **Method** | Geometric source terms | **YES** | Old: no Boris factor. New: `f_boris` multiplies magnetic pressure + tension in source terms. |
| **Method** | Ghost cell / electrode BCs | NO | Unchanged |
| **Material** | Experimental reference data | NO | Same Gribkov PF-1000 waveform |
| **Material** | Preset parameters (V0, L0, C0) | NO | Same pf1000 preset |
| **Measurement** | Objective function weights | **VERIFY** | Phase Q used (0.4, 0.3, 0.3). Sprint S-1 calibration used the same? Check `calibrate_multi_device.py` defaults. |
| **Measurement** | n_trials | **YES** | Phase Q: 69 trials. Sprint S-1: 30 trials. Half the budget. |
| **Manpower** | Optimizer (Optuna TPE) | NO | Same algorithm |
| **Mother Nature** | OS / hardware changes | NO | Same M3 Pro |

### Key Confounders

1. **Trial budget halved**: 30 vs 69 trials. This alone could explain the shift if the landscape
   has a narrow valley. The TPE surrogate may not have had enough samples to model the basin.

2. **Three simultaneous changes**: Boris was applied to (a) source terms, (b) wave speeds, and
   (c) floor enforcement. If the fc shift is from Boris, which of the three mechanisms caused it?
   A partial-Boris experiment (enable one at a time) would isolate this, but costs 3x the calibration budget.

3. **Objective weights**: The calibration objective `J = 0.4*|peak_err| + 0.3*|timing_err| + 0.3*NRMSE`
   may weight timing too heavily. If Boris improves peak but degrades timing, the optimizer may
   have traded fc to rebalance. Confirm the gradient refinement script uses the same weights
   (`calibrate_gradient.py:62-64` shows peak=0.5, timing=0.3, waveform=0.2 -- **different weights!**).

### Parameter Coupling Analysis

The snowplow Lee model current is `I(t) ~ V0 * C * exp(-t/tau) * f(fc, fm)` where:
- fc controls how much circuit current reaches the plasma sheath
- fm controls how much fill gas is swept into the sheath

These are coupled: higher fc (more current) drives the sheath faster, sweeping more mass.
Lower fm (less mass) also produces a faster sheath. The observable (I_peak) constrains
a curve in (fc, fm) space, not a point. Multiple (fc, fm) pairs can produce the same I_peak.

The NRMSE term breaks this degeneracy by fitting the waveform shape, but at 32x64 resolution
with PLM+HLLS, the waveform is diffusive enough that the shape constraint is weak.

**Prediction**: If M5 shows the loss is flat along a curve, then the fc shift is partly
parameter degeneracy (hypothesis C) and partly physics (hypothesis A). The gradient
refinement (M1) will distinguish: if both seeds converge to the same point, it's real physics.

### Decision Matrix

| Measurement | Supports A | Supports B | Supports C |
|-------------|-----------|-----------|-----------|
| M1: Dual-seed gradient | Both converge to ~0.65 | Both converge to ~0.80 | Different convergence points |
| M2: Waveform decomposition | Rundown NRMSE improves | Radial NRMSE degrades | Similar everywhere |
| M3: Mass conservation | Boris mass drift < old | Boris mass drift > old | Similar |
| M4: Species contamination | Boris species purer | -- | -- |
| M5: Landscape flatness | Sharp minimum at 0.65 | Sharp minimum at 0.80 | Flat valley |

**Decision rule**: Hypothesis with >= 3 supporting measurements wins. If tied, run 100-trial
Optuna sweep as tiebreaker.

---

## IMPROVE

### If A (Better Physics) -- fc=0.649 is correct

Actions:
1. Update `src/dpf/presets.py` PF-1000 default `current_fraction=0.649`, `mass_fraction=0.062`
2. Update `scripts/calibrate_multi_device.py` DEVICE_SEEDS to `(0.65, 0.06)` for pf1000
3. Run multi-device recalibration (UNU-ICTP, POSEIDON, FAETON) with Boris enabled
4. Update `docs/FC_FM_CALIBRATION_DMAIC.md` with post-Boris results table
5. Add observation to `memory/observations.md`: Boris vacuum correction shifts fc by -18% because
   density floor was injecting artificial mass that inflated current coupling
6. Recalibrate WALRUS surrogate with new fc/fm defaults (deferred to Phase J.2)

Files to modify:
- `src/dpf/presets.py` -- update pf1000 fc/fm
- `scripts/calibrate_multi_device.py` -- update DEVICE_SEEDS
- `results/multi_device_calibration.json` -- will be overwritten by recalibration

### If B (New Error) -- Boris over-reduces magnetic pressure

Actions:
1. Diagnose WHERE Boris over-reduces: plot `f_boris` field at pinch time, verify it's ~1.0
   in the sheath and << 1 only in vacuum
2. Potential fix: increase `c_boris` from 5e5 to 1e6 m/s (reduces Boris effect in the
   sheath while still bounding vacuum). Recalibrate.
3. Alternative fix: apply Boris only in geometric source terms, not in wave speed
   computation (Minoshima 2019 applies it everywhere, but DPF sheath is not a typical
   astrophysical vacuum problem)
4. If fix identified, recalibrate and verify fc returns to ~0.80

Files to investigate:
- `src/dpf/metal/mlx_kernels.py:900` -- `C_BORIS_SQ = 2.5e11` (500 km/s)^2
- `src/dpf/metal/mlx_primitives.py:53` -- `_C_BORIS_DEFAULT = 5e5`
- `src/dpf/metal/mlx_timestepper.py:109` -- `_C_BORIS_SQ = 2.5e11` (hardcoded duplicate)

**Note**: `c_boris` is hardcoded in three places (DRY violation). Should be a single config
parameter passed through the solver. Fix regardless of hypothesis outcome.

### If C (Insufficient Convergence) -- need more trials

Actions:
1. Run 100-trial Optuna sweep: `python3 scripts/calibrate_multi_device.py --devices pf1000 --trials 100`
2. Follow with gradient refinement from the Optuna best
3. Compare against pre-Boris 69-trial result
4. If the 100-trial result is between 0.65 and 0.80, the landscape is genuinely flat
   and both calibrations are statistically equivalent

**Runtime**: ~25 min for 100 trials.

### Regardless of Outcome

1. **Fix c_boris DRY violation**: Extract `C_BORIS_SQ` into `mlx_primitives.py` and import
   in `mlx_kernels.py` and `mlx_timestepper.py`. Single source of truth.
   - `mlx_primitives.py:53` -- already has `_C_BORIS_DEFAULT`
   - `mlx_kernels.py:900` -- hardcoded `2.5e11f` in Metal shader (cannot import Python)
   - `mlx_timestepper.py:109` -- hardcoded `2.5e11` (can import from primitives)

2. **Equalize objective weights**: `calibrate_gradient.py` uses (0.5, 0.3, 0.2) while
   `calibrate_multi_device.py` uses the `run_mlx_forward_model` defaults (0.4, 0.3, 0.3).
   Standardize to one set. Recommend (0.4, 0.3, 0.3) since waveform shape is important
   for breaking fc/fm degeneracy.

3. **Document Boris c_boris sensitivity**: Run 3 forward evals at c_boris = {2e5, 5e5, 1e6}
   with fc=0.649, fm=0.062 to quantify how sensitive the result is to this parameter.

---

## CONTROL

### C1: Calibration Stability Gate (CI)

Add a regression test that runs a single forward eval with the current best (fc, fm) and
asserts the objective is within 20% of the stored baseline. This catches any physics change
that silently shifts calibration.

```python
# tests/test_calibration_stability.py
@pytest.mark.slow
def test_pf1000_calibration_stable():
    """Regression gate: physics changes that shift calibration must be caught."""
    from dpf.validation.mlx_calibration import run_mlx_forward_model

    # Current best calibration (update when recalibrating)
    EXPECTED_FC = 0.649
    EXPECTED_FM = 0.062
    EXPECTED_OBJ = 0.099  # from results/multi_device_calibration.json
    TOLERANCE = 0.20  # 20% objective shift triggers investigation

    result = run_mlx_forward_model(
        fc=EXPECTED_FC, fm=EXPECTED_FM, preset_name="pf1000"
    )
    assert result.success, "Forward model failed"
    assert abs(result.objective - EXPECTED_OBJ) / EXPECTED_OBJ < TOLERANCE, (
        f"Calibration shifted: obj={result.objective:.4f} vs expected {EXPECTED_OBJ:.4f}. "
        f"A physics change may have invalidated the calibration. "
        f"Run scripts/calibrate_multi_device.py to recalibrate."
    )
```

**Runtime**: ~15 sec. Tag with `@pytest.mark.slow` to exclude from fast CI.

### C2: Post-Physics-Change Smoke Test Protocol

After ANY change to:
- `mlx_kernels.py` (source terms, Riemann solver)
- `mlx_primitives.py` (pressure recovery, wave speeds)
- `mlx_timestepper.py` (floors, clamping, RK stages)
- `mlx_riemann.py` (flux computation)
- `mlx_transport.py` (diffusion, conduction)

Run:
```bash
python3 -c "
from dpf.validation.mlx_calibration import run_mlx_forward_model
r = run_mlx_forward_model(fc=0.7, fm=0.08, preset_name='pf1000')
print(f'I_peak={r.I_peak_A/1e6:.3f} MA, t_peak={r.t_peak_s*1e6:.1f} us, obj={r.objective:.4f}')
assert r.success and r.I_peak_A > 1e6, 'Calibration smoke test failed'
"
```

If I_peak changes by > 20% from the known baseline (~1.8 MA), the physics change
shifted calibration and must be documented + recalibrated.

### C3: Boris Factor Diagnostic Output

Add `f_boris` field to the solver's diagnostic output so future investigations can
inspect where Boris is active without re-deriving it:

```python
# In mlx_solver.py get_diagnostics() or similar:
def _boris_diagnostic(self) -> np.ndarray:
    """Return Boris factor field f_boris(r,z) in [0,1]. 1.0 = physical, <<1 = vacuum."""
    rho = np.array(self.U[IDN])
    B_sq = np.array(self.U[IBR]**2 + self.U[IBZ]**2 + self.U[IBT]**2)
    c_sq = 2.5e11  # (500 km/s)^2
    va_sq = B_sq / np.maximum(rho, 1e-30)
    return c_sq / (va_sq + c_sq)
```

### C4: Calibration Provenance Record

Every calibration result must record:
- Physics configuration (Boris on/off, c_boris value, riemann solver, reconstruction)
- Grid resolution
- Number of Optuna trials
- Objective function weights
- Git commit hash of the solver code

Update `results/multi_device_calibration.json` schema:
```json
{
  "preset": "pf1000",
  "physics_config": {
    "boris_correction": true,
    "c_boris": 5e5,
    "riemann_solver": "hlls",
    "reconstruction": "plm",
    "time_integrator": "ssp_rk2"
  },
  "objective_weights": {"peak": 0.4, "timing": 0.3, "waveform": 0.3},
  "git_commit": "abc123",
  "best_fc": 0.649,
  "best_fm": 0.062,
  "..."
}
```

---

## Execution Order

| Step | Measurement | Time | Dependencies |
|------|------------|------|--------------|
| 1 | M5: Landscape scan (30 evals) | 7 min | None |
| 2 | M1: Gradient refinement seed 1 (fc=0.797) | 15 min | None (parallel with M5) |
| 3 | M1: Gradient refinement seed 2 (fc=0.649) | 15 min | None (parallel with M5) |
| 4 | M2: Waveform decomposition | 30 sec | M1 results for best points |
| 5 | M3: Mass conservation audit | 2 min | None |
| 6 | M4: Species contamination check | 1 min | None |
| 7 | Analysis + decision matrix | 15 min | All measurements complete |
| 8 | Implement IMPROVE actions | 30-60 min | Decision made |
| 9 | Implement CONTROL gates | 30 min | IMPROVE complete |

**Total**: ~2.5-3 hours including implementation.

Steps 1-3 can run in parallel (separate terminal sessions). Steps 5-6 can run in parallel.

---

## References

- Gombosi et al. (2002), JCP 177:176 -- Semi-relativistic MHD, Boris correction origin
- Minoshima et al. (2019), ApJ 874:37 -- Boris correction for HLLD Riemann solver
- `src/dpf/metal/mlx_kernels.py:938-951` -- Boris factor in geometric source terms
- `src/dpf/metal/mlx_primitives.py:294-388` -- `fast_magnetosonic_boris()`, `boris_factor()`
- `src/dpf/metal/mlx_timestepper.py:56-111` -- Boris-aware floor enforcement
- `results/multi_device_calibration.json` -- Post-Boris calibration results (30 trials)
- `docs/FC_FM_CALIBRATION_DMAIC.md` -- Pre-Boris calibration DMAIC (69 trials, fc=0.797)
- `scripts/calibrate_gradient.py` -- Gradient refinement script (NOTE: different objective weights)
- `scripts/calibrate_multi_device.py` -- Optuna TPE calibration sweep
