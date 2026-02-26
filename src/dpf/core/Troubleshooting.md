# Core Module Troubleshooting Guide

## Cross-Review Panel Consensus (Date: 2026-02-25)

### Review Panel
- **Python Expert (py-engine)**: Code quality, state management, error handling, API contracts
- **Physics Expert (phys-engine)**: Operator splitting, coupling physics, EMF, inductance computation
- **Cross-Review Synthesizer (xreview-engine)**: Source verification, adjudication, severity calibration

### Scope
Files reviewed: `core/bases.py`, `core/field_manager.py`

---

## MEDIUM Priority Issues

### [M1] ✅ FIXED — FieldManager Uses Cartesian Vector Calculus for Cylindrical Geometry
- **File**: `field_manager.py:79-129`
- **Found by**: py-engine (H2)
- **Cross-review verdict**: **CONFIRMED but DOWNGRADED from HIGH to MEDIUM**
- **Fix**: Added `if self.geometry == "cylindrical":` branches to `divergence()` and `curl()` using correct cylindrical operators `(1/r)*d(r*F_r)/dr` and `(1/r)*d(r*F_θ)/dr` with `np.maximum(r, 1e-30)` axis-singularity guard.

- **Description**: `FieldManager.divergence()`, `curl()`, and `gradient()` use Cartesian finite-difference formulas regardless of the `self.geometry` setting. When `geometry="cylindrical"`, these produce incorrect results because cylindrical div/curl/grad have additional 1/r geometric terms.

- **Evidence**:
  ```python
  # field_manager.py:88-92 — Cartesian divergence always used
  def divergence(self, field: np.ndarray) -> np.ndarray:
      return (
          np.gradient(field[0], self.dx, axis=0)   # Should be (1/r)*d(r*F_r)/dr
          + np.gradient(field[1], self.dy, axis=1)  # dy unused in cylindrical
          + np.gradient(field[2], self.dz, axis=2)
      )
  ```

  Engine sets `self.field_manager.geometry = "cylindrical"` at `engine.py:297` but this flag is **never checked** in any computation method.

- **Downgrade justification**: The engine does NOT use `FieldManager.divergence()`, `curl()`, or `gradient()` for any physics computation. The cylindrical J-field computation (engine.py:706) uses `self.fluid.geom.curl()` from `CylindricalMHDSolver`, which correctly handles cylindrical geometry. The only FieldManager call is `compute_plasma_inductance()` (which correctly handles cylindrical volume elements at line 59-69) and `max_div_B()` for diagnostics (which will report wrong values in cylindrical geometry).

- **Impact**: Wrong `max_div_B()` diagnostic values for cylindrical simulations. No impact on physics computations.

- **Proposed fix**:
  ```python
  def divergence(self, field: np.ndarray) -> np.ndarray:
      if self.geometry == "cylindrical":
          r = (np.arange(self.nx) + 0.5) * self.dx
          r = r[:, np.newaxis, np.newaxis]
          # div(F) = (1/r)*d(r*F_r)/dr + dF_z/dz  (axisymmetric)
          rFr = field[0] * r
          dFr_dr = np.gradient(rFr, self.dx, axis=0) / np.maximum(r, 1e-30)
          dFz_dz = np.gradient(field[2], self.dz, axis=2)
          return dFr_dr + dFz_dz
      return (
          np.gradient(field[0], self.dx, axis=0)
          + np.gradient(field[1], self.dy, axis=1)
          + np.gradient(field[2], self.dz, axis=2)
      )
  ```

  Similarly for `curl()`:
  ```python
  def curl(self, field: np.ndarray) -> np.ndarray:
      if self.geometry == "cylindrical":
          r = (np.arange(self.nx) + 0.5) * self.dx
          r = r[:, np.newaxis, np.newaxis]
          # curl_r = -dF_theta/dz
          # curl_theta = dF_r/dz - dF_z/dr
          # curl_z = (1/r)*d(r*F_theta)/dr
          curl_r = -np.gradient(field[1], self.dz, axis=2)
          curl_theta = (np.gradient(field[0], self.dz, axis=2)
                       - np.gradient(field[2], self.dx, axis=0))
          rFt = field[1] * r
          curl_z = np.gradient(rFt, self.dx, axis=0) / np.maximum(r, 1e-30)
          return np.array([curl_r, curl_theta, curl_z])
      # ... existing Cartesian curl ...
  ```

---

### [M2] ✅ FIXED — _compute_dt Uses Private Method Not in PlasmaSolverBase
- **File**: `engine.py:472` referencing `bases.py`
- **Found by**: py-engine (M1)
- **Cross-review verdict**: **CONFIRMED**
- **Fix**: Added `compute_dt(self, state) -> float` to `PlasmaSolverBase` with a 1e-6 s safe default fallback. Engine-side call-site update (`_compute_dt` → `compute_dt`) is a separate engine.py change.

- **Description**: `self.fluid._compute_dt(self.state)` calls a private method not defined in `PlasmaSolverBase`. If a new backend doesn't implement `_compute_dt`, it fails at runtime with `AttributeError`.

- **Evidence**:
  ```python
  # engine.py:472
  dt_fluid = self.fluid._compute_dt(self.state)

  # bases.py:81-108 — PlasmaSolverBase only defines:
  #   step() — abstract
  #   coupling_interface() — returns default CouplingState
  # No _compute_dt() method
  ```

- **Proposed fix**: Add to `PlasmaSolverBase`:
  ```python
  def compute_dt(self, state: dict[str, np.ndarray]) -> float:
      """Compute CFL-limited timestep for this solver.

      Args:
          state: Current simulation state.

      Returns:
          Maximum stable timestep [s].
      """
      return 1e-6  # Safe default fallback
  ```

  Update engine to use public method: `dt_fluid = self.fluid.compute_dt(self.state)`

---

## LOW Priority Issues

### [L1] PlasmaSolverBase.coupling_interface Returns Empty CouplingState
- **File**: `bases.py:106-108`
- **Found by**: xreview-engine (cross-review observation)
- **Cross-review verdict**: **CONFIRMED LOW**

- **Description**: The default `coupling_interface()` returns `CouplingState()` with all zeros. Backends that don't override this (e.g., AthenaK subprocess mode) always report zero R_plasma and zero L_plasma, causing the circuit solver to use only the FieldManager inductance.

- **Impact**: AthenaK simulations may have incorrect plasma-circuit coupling if the VTK output doesn't include resistivity/inductance data. Currently mitigated by the FieldManager fallback inductance computation.

---

### [L2] ✅ FIXED — CouplingState Missing Validation
- **File**: `bases.py:58-78`
- **Found by**: xreview-engine (cross-review observation)
- **Cross-review verdict**: **CONFIRMED LOW**

- **Description**: `CouplingState` is a plain dataclass with no validation. Negative inductance (Lp < 0), negative resistance (R_plasma < 0), or NaN values would propagate silently to the circuit solver.

- **Fix**: Added `__post_init__` that clamps `Lp` and `R_plasma` to 0.0 if negative, preventing non-physical values from reaching the circuit solver.

---

## Verified Correct

### compute_plasma_inductance (field_manager.py:37-75)
- **Verified by**: phys-engine (M1, self-retracted after analysis)
- **Cross-review verdict**: **CORRECT**

- The cylindrical volume integration uses correct volume elements: `dV = 2*pi*r * dr * dz`.
- The inductance formula `L = 2 * W_B / I^2` where `W_B = sum(B^2/(2*mu_0) * dV)` is algebraically consistent with the engine's separate computation at lines 770-776.
- Both yield `L = sum(B^2/mu_0 * dV) / I^2`.

### PlasmaSolverBase Interface (bases.py:81-108)
- Clean ABC with `step()` and `coupling_interface()`.
- `step()` signature accepts `source_terms` for kinetic coupling.
- All three production backends (MHDSolver, CylindricalMHDSolver, MetalMHDSolver) correctly implement this interface.

### StepResult Dataclass (bases.py:20-55)
- Comprehensive diagnostics with sensible defaults.
- All 15 fields are used by engine.py and tests.

---

## Summary Table

| ID | Severity | File | Line | Category | Issue | Verdict |
|----|----------|------|------|----------|-------|---------|
| M1 | MEDIUM | field_manager.py | 79-129 | Physics | Cartesian math for cylindrical | ✅ FIXED |
| M2 | MEDIUM | bases.py | — | API | _compute_dt not in base | ✅ FIXED |
| L1 | LOW | bases.py | 106 | API | Empty default CouplingState | CONFIRMED (by design) |
| L2 | LOW | bases.py | 58 | Validation | No CouplingState guards | ✅ FIXED |

## Cross-References

- The double circuit step (C1 in `src/dpf/Troubleshooting.md`) affects how `CouplingState` is used — two coupling states with different R_plasma and back_emf are passed to the circuit in a single engine step.
- The FieldManager.B aliasing issue (L1 in `src/dpf/Troubleshooting.md`) interacts with `compute_plasma_inductance()` — if operation order changes, the inductance could be computed from post-MHD B instead of pre-MHD B.
