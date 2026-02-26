# Circuit Solver Troubleshooting

**Cross-review by**: xreview-circuit (synthesizing py-circuit-val + phys-circuit-val)
**Files reviewed**: `rlc_solver.py` (335 LOC)
**Date**: 2026-02-25

---

## CRITICAL Findings

### ✅ FIXED — CRIT-1: `initial_energy()` returns CURRENT capacitor energy, not initial

- **File**: `rlc_solver.py:332-334`
- **Found by**: py-circuit-val (C-RLC-1), phys-circuit-val (BUG-CV-1)
- **Cross-review verdict**: **CONFIRMED** (both reviewers independently identified, code verified)

**Description**: The method `initial_energy()` computes `0.5 * self.C * (self.state.charge / self.C) ** 2` which simplifies to `0.5 * self.state.charge**2 / self.C = 0.5 * C * V_current**2`. Since `self.state.charge` is updated every timestep (line 309: `self.state.charge = self.C * V_new`), this returns the **current** capacitor energy, identical to `self.state.energy_cap`.

**Evidence**:
```python
# rlc_solver.py:332-334
def initial_energy(self) -> float:
    """Return the initial energy stored in the capacitor."""
    return 0.5 * self.C * (self.state.charge / self.C) ** 2
```

**Correct behavior**: Should return the energy at t=0, i.e., `0.5 * C * V0**2`. This is a constant that never changes during simulation.

**Proposed fix**:
```python
# In __init__, add after line 107:
self._initial_energy_J = 0.5 * C * V0**2

# Replace lines 332-334:
def initial_energy(self) -> float:
    """Return the initial energy stored in the capacitor."""
    return self._initial_energy_J
```

**Justification**: V0 is available in `__init__` as a parameter. The initial capacitor energy `0.5*C*V0^2` is a physical constant of the circuit.

**Impact**: Any energy conservation audit using `initial_energy()` as denominator will compare `E_total / E_cap_current` instead of `E_total / E_initial`, giving a spuriously varying "conservation" ratio. The `total_energy()` method (line 323-330) is correct and unaffected.

---

## HIGH Findings

*None.*

---

## MEDIUM Findings

### ✅ FIXED — MED-1: `dL_dt != 0.0` float equality check is fragile

- **File**: `rlc_solver.py:242-245`
- **Found by**: py-circuit-val (C-RLC-3)
- **Cross-review verdict**: **CONFIRMED** — exact float equality is fragile

**Description**: The code uses `coupling.dL_dt != 0.0` to decide whether to use the coupling-provided dL/dt or the internal BDF2 estimate. If the plasma solver legitimately returns `dL_dt = 0.0` (e.g., at pinch stagnation when inductance is momentarily constant), this code overrides it with the BDF2 estimate, which may be nonzero due to numerical noise.

**Evidence**:
```python
# rlc_solver.py:242-245
if coupling.dL_dt != 0.0:
    dLp_dt = coupling.dL_dt
else:
    dLp_dt = self.compute_dLp_dt(Lp)
```

**Correct behavior**: Use a sentinel value (`None` or `float('nan')`) for "not provided" in `CouplingState.dL_dt`, and test `if coupling.dL_dt is not None:`. Alternatively, use an explicit boolean flag like `coupling.dL_dt_provided`.

**Proposed fix**:
```python
if coupling.dL_dt is not None:
    dLp_dt = coupling.dL_dt
else:
    dLp_dt = self.compute_dLp_dt(Lp)
```
(Requires updating `CouplingState` default from `0.0` to `None`.)

**Impact**: At pinch stagnation, the BDF2 estimate will inject numerical noise into dL/dt when the true value is zero, potentially causing small spurious current oscillations. This is a moderate-severity edge case.

### ✅ FIXED — MED-2: BDF2 uniform-spacing tolerance is extremely tight

- **File**: `rlc_solver.py:170`
- **Found by**: py-circuit-val (C-RLC-4)
- **Cross-review verdict**: **CONFIRMED** — `1e-20` relative tolerance essentially requires bit-exact equality

**Description**: The check `abs(dt1 - dt2) < 1e-20 * max(dt1, dt2, 1e-30)` determines whether to use the uniform or non-uniform BDF2 formula. For typical DPF timesteps (~1e-9 to 1e-6 s), any floating-point rounding will trigger the non-uniform branch even when timesteps are intended to be uniform.

**Evidence**:
```python
# rlc_solver.py:170
if abs(dt1 - dt2) < 1e-20 * max(dt1, dt2, 1e-30):
```

**Correct behavior**: Use a relative tolerance of `1e-6` to `1e-8` for the uniform-spacing test. The non-uniform formula is mathematically equivalent to the uniform formula when `r = dt2/dt1 = 1`, so using it unnecessarily is not a bug per se, but it involves extra operations and can amplify rounding errors when `r` deviates from 1.0 by tiny amounts.

**Proposed fix**:
```python
if abs(dt1 - dt2) < 1e-8 * max(dt1, dt2, 1e-30):
```

**Impact**: Low in practice (the non-uniform formula reduces to BDF2 when r=1), but could introduce O(epsilon) errors that accumulate over many timesteps.

### MED-3: Back-EMF treated as constant in midpoint rule (latent until C1 engine bug is fixed)

- **File**: `rlc_solver.py:257-264` (post-crowbar) and `rlc_solver.py:275-278` (non-crowbar)
- **Found by**: py-circuit-val (C-RLC-2)
- **Cross-review verdict**: **DOWNGRADED from CRITICAL to MEDIUM** — currently masked by hardcoded `back_emf = 0.0` in engine.py (known bug C1)

**Description**: The implicit midpoint method is nominally 2nd-order, but `back_emf` is treated as a constant over the timestep (not evaluated at the midpoint). For a truly 2nd-order method, the back_emf should also be midpoint-evaluated: `back_emf_mid = 0.5 * (back_emf_n + back_emf_{n+1})`. However, since back_emf depends on the MHD state which is computed externally, the solver has no way to evaluate it at the midpoint without a predictor-corrector loop.

**Cross-review note**: phys-circuit-val verified the implicit midpoint derivation as CORRECT for the linear terms (KVL, capacitor ODE). The back-emf is an external forcing term, and treating it as constant is the standard approach in operator-split codes. This is a deliberate simplification, not a bug. The method remains globally 2nd-order for the linear circuit part and 1st-order for the back-emf coupling.

**Impact**: Once back_emf is non-zero (after C1 fix), the circuit solver's temporal accuracy during radial compression (when back_emf varies rapidly) will be O(dt) rather than O(dt^2). This matters for rapid pinch dynamics but is acceptable for the current code maturity level.

---

## LOW Findings

### LOW-1: I_mid computed twice redundantly

- **File**: `rlc_solver.py:281` and `rlc_solver.py:295`
- **Found by**: py-circuit-val (C-RLC-5)
- **Cross-review verdict**: **CONFIRMED** — cosmetic

`I_mid = 0.5 * (I_n + I_new)` is computed at line 281 for the voltage update (inside the non-crowbar branch) and again at line 295 for energy accounting (outside both branches). The second computation shadows the first. Values are identical; this is redundant but harmless.

### LOW-2: Energy dissipation integral uses midpoint rule (O(dt) for I^2)

- **File**: `rlc_solver.py:300`
- **Found by**: py-circuit-val (C-RLC-6)
- **Cross-review verdict**: **CONFIRMED but trivial**

The dissipated energy `R * I_mid^2 * dt` uses the midpoint `I_mid` for the integral of `I(t)^2`. The midpoint rule is 2nd-order for linear functions but only 1st-order for the integral of `I^2` (which is quadratic in I). A Simpson-type formula `(I_n^2 + 4*I_mid^2 + I_new^2)/6 * dt` would be 3rd-order.

**Impact**: Negligible. The cumulative dissipation error is O(dt * T_sim) where T_sim is the total simulation time. For typical DPF timesteps (~1e-9 s) and simulation times (~10 us), this is ~0.001% error.

### ✅ FIXED — LOW-3: Module docstring says "central difference" but code uses backward difference

- **File**: `rlc_solver.py:17-19`
- **Found by**: phys-circuit-val (BUG-CV-4)
- **Cross-review verdict**: **CONFIRMED** — documentation mismatch only

The module-level docstring says "2nd-order central difference" but the method docstring (line 143-145) correctly says BDF2 (backward). The code implements BDF2 correctly. Only the top-level docstring is wrong.

---

## REJECTED Findings

### REJ-1: py-circuit-val C-RLC-2 severity overrated

The Python reviewer initially rated the back-EMF treatment as CRITICAL. Cross-review downgrades to MEDIUM because: (a) it's currently masked by the known C1 engine bug (`back_emf = 0.0`), (b) the physics reviewer verified the implicit midpoint derivation as correct for all other terms, and (c) treating external forcing as constant is the standard approach in operator-split DPF codes (Lee model, RADPFM all do this).

---

## Positive Observations

1. **Implicit midpoint method**: Correct derivation verified by physics reviewer against Hairer & Wanner (1996). Unconditionally A-stable, 2nd-order, symplectic.
2. **BDF2 dL/dt**: Both uniform and non-uniform spacing formulas verified against Gear (1971). Correct Lagrange interpolation derivative for non-uniform grids.
3. **Crowbar model**: Correctly implements voltage-zero trigger, capacitor freeze, L-R decay. Physics-correct per Lee (2005).
4. **Coaxial inductance**: `L_p = (mu_0 / 2pi) * length * ln(b/r)` matches Griffiths (4th ed.) Eq. 7.27.
5. **Clean dataclass design**: `CircuitState` separates capacitor, inductor, and resistive energy tracking. `energy_res_plasma` correctly excluded from `total_energy()` to avoid double-counting.
