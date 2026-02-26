# Engine/Config/Presets/Constants/Units Troubleshooting Guide

## Cross-Review Panel Consensus (Date: 2026-02-25)

### Review Panel
- **Python Expert (py-engine)**: Code quality, state management, error handling, API contracts
- **Physics Expert (phys-engine)**: Operator splitting, coupling physics, EMF, Lee model parameters
- **Cross-Review Synthesizer (xreview-engine)**: Source verification, adjudication, severity calibration

### Scope
Files reviewed: `engine.py`, `config.py`, `presets.py`, `constants.py`, `units.py`

---

## CRITICAL Issues (Must Fix)

### [C1] Double Circuit Step in Python/Metal/AthenaK Backends
- **File**: `engine.py:646-649` + `engine.py:812-817`
- **Found by**: py-engine (C1) + phys-engine (C1) independently
- **Cross-review verdict**: **CONFIRMED** (3-0 consensus, highest-impact bug found)

- **Description**: The `step()` method calls `self.circuit.step()` at **two** locations for all non-Athena++ backends. Each call mutates the circuit's internal state (current, voltage, time). The net effect: the circuit advances by `2*dt` per engine timestep while the simulation only advances by `dt`.

- **Evidence** (verified from engine.py and rlc_solver.py):
  ```python
  # engine.py:646-649 — FIRST circuit step (R_plasma=0.0, back_emf from MHD)
  back_emf = self._compute_back_emf(dt)
  coupling = CouplingState(Lp=L_p, dL_dt=dL_dt, R_plasma=0.0)
  new_coupling = self.circuit.step(coupling, back_emf, dt)  # MUTATES state

  # engine.py:812-817 — SECOND circuit step (R_plasma from Spitzer, back_emf=0)
  back_emf = 0.0
  new_coupling = self.circuit.step(coupling, back_emf, dt)  # MUTATES state again
  self._coupling = new_coupling
  ```

  Each call to `circuit.step()` executes (rlc_solver.py:307-310):
  ```python
  self.state.current = I_new
  self.state.voltage = V_new
  self.state.time += dt      # ← time advances by dt EACH call
  ```

  Additional corruption: rlc_solver.py:313 appends to `_Lp_history` on EACH call, recording two entries per engine step. This corrupts the BDF2 `dL/dt` computation.

- **Correct behavior**: The circuit should be stepped exactly **once** per engine timestep, with the full coupling state (R_plasma, back_emf, L_plasma). The Athena++ path (`_step_athena()`, line 1276) correctly calls `circuit.step()` only once.

- **Proposed fix**: Remove the first circuit step entirely (lines 628-653). Use the previous timestep's current for electrode BCs. Merge back_emf into the second call:
  ```python
  # REMOVE lines 628-653 entirely.
  # At line 652, use previous current for electrode BC:
  if self.boundary_cfg.electrode_bc:
      self._apply_electrode_bc(self._coupling.current)

  # At line 816, include motional back-EMF:
  back_emf = self._compute_back_emf(dt)
  new_coupling = self.circuit.step(coupling, back_emf, dt)
  ```

- **Justification**: The Athena++ single-step path is the reference implementation. The double-step desynchronizes circuit time from simulation time, inflates current ramp rate by ~2x, and corrupts the BDF2 inductance derivative history.

- **Impact**: Circuit current grows ~2x too fast. Current dip timing is wrong. Energy accounting double-counts dissipation. BDF2 dL/dt history is corrupted. Validation against PF-1000 I_peak = 1.87 MA is unreliable because the bug is partially masked by snowplow damping and back-pressure.

---

### [C2] Motional Back-EMF Hardcoded to Zero (Compounds C1)
- **File**: `engine.py:812-816`
- **Found by**: phys-engine (C2), py-engine (noted as part of C1)
- **Cross-review verdict**: **CONFIRMED** (known bug C1 in MEMORY.md, physics analysis verified)

- **Description**: The comment at line 812-815 claims `back_emf = 0` is correct because "I * dL/dt is already in R_star." This is partially correct for the **inductive** back-EMF but ignores the **motional** back-EMF from resolved MHD velocity fields.

  Two distinct back-EMF terms exist in a DPF:
  1. **Inductive**: `I * dL/dt` — correctly handled in `R_star = R_eff + dLp_dt` (rlc_solver.py:272)
  2. **Motional**: `integral(v x B) . dl` — work done by Lorentz force on plasma against circuit current. NOT captured by `dL/dt`.

  The `_compute_back_emf()` method (lines 1059-1090) correctly computes the motional EMF from the MHD state, but the result is discarded at line 816.

- **Evidence**:
  ```python
  # engine.py:1076-1078 — correctly computes motional EMF
  if self.geometry_type == "cylindrical":
      emf_density = -(velocity[0] * B[1])  # -(v_r * B_theta)

  # engine.py:816 — throws it away
  back_emf = 0.0  # <-- motional EMF discarded
  ```

- **Correct behavior**: For the **snowplow-only** (0D) case, `back_emf = 0` IS correct because `I*dL/dt` captures the inductance change. For the **coupled MHD-circuit** case with resolved B and v fields, the motional EMF should feed back to the circuit. Reference: Lee & Saw (2014), Eq. (4).

- **Proposed fix** (combined with C1 fix — single circuit step):
  ```python
  # After removing the first circuit step, include motional back-EMF in the single call:
  motional_emf = self._compute_back_emf(dt) if self.backend != "python" else 0.0
  # For snowplow-only runs (no resolved MHD fields), keep back_emf=0
  # For MHD-coupled runs, include the motional term
  if np.any(self.state["velocity"] != 0):
      back_emf = motional_emf
  else:
      back_emf = 0.0
  new_coupling = self.circuit.step(coupling, back_emf, dt)
  ```

  Note: The decision on whether to include motional EMF should depend on whether the MHD solver is actively evolving fields (velocity non-zero), not on the backend type.

- **Impact**: Current quench dynamics are wrong for MHD-coupled simulations. The current doesn't feel the drag from plasma motion against the magnetic field. This primarily affects the post-pinch phase where v_r is large.

---

## HIGH Priority Issues

### [H1] Metal Electrode BC Applies B_theta Inside Anode (1/r Singularity)
- **File**: `engine.py:1107-1111`
- **Found by**: py-engine (H1)
- **Cross-review verdict**: **CONFIRMED** (Metal backend only)

- **Description**: The Metal backend electrode BC applies `B_theta = mu_0 * I / (2*pi*r)` for ALL r > 0, including inside the anode. Due to Python operator precedence, the condition at line 1109 evaluates as `True` for all r > 0.

- **Evidence**:
  ```python
  # engine.py:1109 — operator precedence bug
  if cc.anode_radius <= r <= cc.cathode_radius and r > 0 or r < cc.anode_radius and r > 0:
  #    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  #    (True for annulus)                        OR         (True for r < anode_radius!)
  ```

  At the first cell (r = 0.5 * dr), for PF-1000 with I = 1.87 MA and dr = 7.5e-4 m:
  `B_theta = mu_0 * 1.87e6 / (2*pi*3.75e-4) = 997 T` — physically absurd inside copper anode.

  The **Python backend** uses `CylindricalMHDSolver.apply_electrode_bfield_bc()` (cylindrical_mhd.py:454-484) which handles this correctly by only setting B at specific radial indices (anode, cathode, and between).

- **Proposed fix**:
  ```python
  if cc.anode_radius <= r <= cc.cathode_radius and r > 0:
      B[1, ir, :, :] = _mu_0 * current / (2.0 * pi * r)
  elif r < cc.anode_radius:
      B[1, ir, :, :] = 0.0  # No field inside solid anode
  ```

- **Impact**: Unphysical ~1000 T magnetic field inside the anode creates extreme magnetic pressure in the first few grid cells. Affects Metal backend cylindrical simulations only.

---

### [H2] Checkpoint Save/Load Missing Snowplow State
- **File**: `engine.py:492-537`
- **Found by**: py-engine (H3)
- **Cross-review verdict**: **CONFIRMED**

- **Description**: `save_checkpoint()` and `load_from_checkpoint()` preserve circuit state and field arrays but **not** snowplow dynamics state (phase, z, v, r_shock, swept_mass, rundown_complete). A simulation restarted from checkpoint with snowplow enabled resets to the initial axial phase.

- **Evidence**: Lines 492-510 (save) save only `{current, voltage, energy_cap, energy_ind, energy_res}`. Lines 512-537 (load) restore only circuit state. No snowplow attributes are persisted.

- **Proposed fix**:
  ```python
  # In save_checkpoint():
  snowplow_state = None
  if self.snowplow is not None:
      snowplow_state = {
          "z": self.snowplow.z, "v": self.snowplow.v,
          "r_shock": self.snowplow.r_shock, "v_r": self.snowplow.v_r,
          "phase": self.snowplow.phase,
          "swept_mass": self.snowplow.swept_mass,
          "rundown_complete": self.snowplow.rundown_complete,
      }

  # In load_from_checkpoint():
  if self.snowplow is not None and "snowplow" in data:
      sp = data["snowplow"]
      for attr, val in sp.items():
          setattr(self.snowplow, attr, val)
  ```

- **Justification**: The snowplow phase (axial -> radial -> pinch -> reflected) is the defining state of a DPF simulation. Losing it on restart makes checkpointing unreliable for production runs.

---

### [H3] PF-1000 Anode Length Default = 0.16 m (Should Be 0.6 m)
- **File**: `config.py:314` + `presets.py:41-69`
- **Found by**: phys-engine (M2)
- **Cross-review verdict**: **CONFIRMED** (upgraded from MODERATE to HIGH — affects primary validation device)

- **Description**: `SnowplowConfig.anode_length` defaults to 0.16 m. The PF-1000 preset does NOT override this. PF-1000 anode length is **600 mm = 0.6 m** per Scholz (2006) Table 1 and Lee & Saw (2014) z_0 = 0.6 m.

- **Evidence**:
  ```python
  # config.py:314
  anode_length: float = Field(0.16, gt=0, description="Anode length [m]")

  # presets.py:41-69 (pf1000) — no snowplow override
  # No "snowplow": {"anode_length": 0.6} in the preset
  ```

- **Impact**: With 0.16 m instead of 0.6 m:
  - Axial rundown time ~4x too short (t_rundown ~ sqrt(z))
  - Swept mass at end of rundown ~4x too small
  - Plasma inductance at radial entry ~4x too small (L ~ z_anode)
  - Current dip magnitude and timing dramatically wrong

- **Proposed fix**: Add snowplow overrides to each preset:
  ```python
  # presets.py — pf1000 preset
  "snowplow": {"anode_length": 0.6},      # Scholz 2006: 600 mm

  # presets.py — nx2 preset
  "snowplow": {"anode_length": 0.05},     # ~50 mm for NX2

  # presets.py — llnl_dpf preset
  "snowplow": {"anode_length": 0.08},     # ~80 mm for LLNL DPF
  ```

- **Reference**: Scholz (2006) Table 1; Lee & Saw (2014) Sec. III.

---

## MEDIUM Priority Issues

### [M1] Electrode BC Forces B_theta at Open Electrode End (z=nz-1)
- **File**: `cylindrical_mhd.py:477-480`
- **Found by**: phys-engine (C3)
- **Cross-review verdict**: **CONFIRMED but DOWNGRADED from CRITICAL to MEDIUM**

- **Description**: The Python backend electrode BC sets `B_theta = mu_0*I/(2*pi*r)` at both z=0 (insulator/closed end) and z=nz-1 (open end) for cells between the electrodes. In DPF geometry, z=nz-1 is the open end where the sheath exits. Setting a forced B_theta there is non-physical — it should be a free boundary.

- **Evidence**:
  ```python
  # cylindrical_mhd.py:477-480
  for iz in [0, self.nz - 1]:  # Both ends!
      for ir in range(idx_anode, min(idx_cath + 1, self.nr)):
          r_local = max(r[ir], 1e-10)
          B[1, ir, iz] = mu_0 * current / (2.0 * np.pi * r_local)
  ```

- **Mitigating factor**: The axial zipper BC (engine.py:1121-1122) zeros B_theta for z > z_sheath AFTER the electrode BC is applied, partially correcting this during the axial rundown phase. During the radial phase, the full z-axis has current flow, so B_theta at z=nz-1 is physically appropriate.

- **Proposed fix**: Only apply electrode BC at z=0 (closed end), or implement a gradient-free (extrapolation) BC at z=nz-1:
  ```python
  for iz in [0]:  # Only closed end (insulator face)
      for ir in range(idx_anode, min(idx_cath + 1, self.nr)):
          r_local = max(r[ir], 1e-10)
          B[1, ir, iz] = mu_0 * current / (2.0 * np.pi * r_local)
  # Open end (z=nz-1): zero-gradient extrapolation
  B[1, :, -1] = B[1, :, -2]
  ```

- **Reference**: Lee (1984), Scholz (2006): downstream face is open; Mather-type geometry.

---

### [M2] Stale Z_bar for Second Strang Half-Step
- **File**: `engine.py:664-670` + `engine.py:907`
- **Found by**: py-engine (H5)
- **Cross-review verdict**: **CONFIRMED but DOWNGRADED from HIGH to MEDIUM**

- **Description**: Z_bar and Z_bar_field are computed once (lines 664-670) using the pre-MHD state. After the MHD step modifies rho and Te, the second Strang half-step at line 907 uses stale ionization state.

- **Evidence**:
  ```python
  # Line 667: Z_bar computed BEFORE MHD step
  Z_bar_field = saha_ionization_fraction_array(Te_flat, ne_flat).reshape(Te.shape)

  # Line 907: second Strang half uses STALE Z_bar_field
  self._apply_collision_radiation(dt / 2.0, Z_bar, Z_bar_field=Z_bar_field)
  ```

- **Downgrade justification**: The operator splitting is already 1st-order due to the non-standard splitting order (see [M3] below). Recomputing Z_bar would not improve the overall temporal accuracy. Z_bar changes slowly relative to a single MHD timestep. The fix is straightforward but the impact is marginal until the splitting order is corrected.

- **Proposed fix** (apply when splitting order is fixed):
  ```python
  # Before line 907:
  Te_post = self.state["Te"]
  ne_post = self.state["rho"] / self.ion_mass
  Z_bar_field_post = saha_ionization_fraction_array(
      Te_post.ravel(), ne_post.ravel()
  ).reshape(Te_post.shape)
  Z_bar_post = max(float(np.mean(Z_bar_field_post)), 0.01)
  self._apply_collision_radiation(dt / 2.0, Z_bar_post, Z_bar_field=Z_bar_field_post)
  ```

---

### [M3] Strang Splitting Order Is Non-Standard (1st Order Coupling)
- **File**: `engine.py` step() method (lines 628-907)
- **Found by**: phys-engine (P1)
- **Cross-review verdict**: **CONFIRMED**

- **Description**: The operator splitting order places the circuit step, snowplow, ablation, Powell sources, Nernst, and sheath BC between the two Strang half-steps. Only collision+radiation are symmetrically split. This gives O(dt) temporal accuracy for circuit-MHD coupling instead of O(dt^2).

- **Current order**: S(dt/2) -> Circuit -> Snowplow -> MHD -> Ablation -> Powell -> Nernst -> Sheath -> S(dt/2)
- **Standard Strang**: S(dt/2) -> F(dt) -> S(dt/2) where S = all source terms, F = hyperbolic flux

- **Impact**: O(dt) instead of O(dt^2) temporal accuracy for coupled physics. For typical DPF timesteps (dt ~ ns), the error is small but prevents formal convergence verification.

---

### [M4] Silent NaN Repair in _sanitize_state
- **File**: `engine.py:543-572`
- **Found by**: py-engine (H4)
- **Cross-review verdict**: **CONFIRMED but DOWNGRADED from HIGH to MEDIUM**

- **Description**: `_sanitize_state()` silently replaces NaN/Inf with floor values and only logs a warning. Persistent NaN from solver instabilities would be "repaired" every step, producing smooth-looking but incorrect trajectories.

- **Downgrade justification**: This is a standard defensive programming pattern in research codes (e.g., FLASH, Athena++ have similar guards). NaN repair at low frequency (1-2 cells per step from boundary artifacts) is legitimate. The concern is about persistent high-frequency NaN that indicates real instability.

- **Proposed fix**: Add cumulative tracking and fail-fast threshold:
  ```python
  def _sanitize_state(self, label: str) -> int:
      ...
      if count > 0:
          self._cumulative_repairs = getattr(self, '_cumulative_repairs', 0) + count
          if self._cumulative_repairs > 10000:
              raise RuntimeError(
                  f"Solver instability: {self._cumulative_repairs} cumulative NaN/Inf repairs. "
                  f"Latest: {count} in '{key}' {label}."
              )
  ```

---

### [M5] Inconsistent mu_0 Between units.py and constants.py
- **File**: `units.py:34` vs `constants.py:12`
- **Found by**: py-engine (H6)
- **Cross-review verdict**: **CONFIRMED but DOWNGRADED from HIGH to MEDIUM**

- **Description**: Two different mu_0 values are used:
  - `constants.py`: `scipy.constants.mu_0` = **1.25663706127e-06** (CODATA 2022 measured value, scipy 1.17.0)
  - `units.py`: `4.0e-7 * pi` = **1.25663706144e-06** (traditional exact definition)

- **Evidence** (runtime verification):
  ```
  scipy.constants.mu_0 = 1.25663706127000005128e-06
  units.py MU_0        = 1.25663706143591728851e-06
  Relative difference  = 1.320e-10
  ```

  The difference is ~1.3e-10 relative (verified). This arises because scipy 1.17.0 uses the CODATA 2022 measured value (post-2019 SI redefinition where mu_0 is no longer exactly 4*pi*1e-7).

- **Impact**: Negligible for all practical MHD simulations (~13th decimal place). However, violates the "single source of truth" principle stated in `constants.py` docstring.

- **Proposed fix**:
  ```python
  # units.py — use the project's single source of truth
  from dpf.constants import mu_0 as MU_0
  SQRT_MU_0 = np.sqrt(MU_0)
  ```

---

### [M6] FluidConfig Missing Validators for String Fields
- **File**: `config.py:142-166`
- **Found by**: py-engine (M5)
- **Cross-review verdict**: **CONFIRMED**

- **Description**: Four `FluidConfig` string fields accept arbitrary values without validation: `diffusion_method`, `time_integrator`, `reconstruction`, `precision`. Invalid values like `diffusion_method="banana"` pass silently.

- **Proposed fix**: Use `Literal` type hints or `@model_validator`:
  ```python
  from typing import Literal
  diffusion_method: Literal["explicit", "sts", "implicit"] = "explicit"
  time_integrator: Literal["ssp_rk2", "ssp_rk3"] = "ssp_rk3"
  precision: Literal["float32", "float64"] = "float32"
  ```

---

### [M7] fill_pressure_Pa Default Wrong for NX2
- **File**: `config.py:310-312` + `presets.py` (nx2 preset)
- **Found by**: py-engine (M8), phys-engine (P5)
- **Cross-review verdict**: **CONFIRMED** (same root cause as H3)

- **Description**: `fill_pressure_Pa = 400 Pa` is the default. NX2 operates at ~0.5-2 Torr (67-267 Pa), not 3 Torr. The NX2 preset does not override this. Fix alongside H3 by adding `fill_pressure_Pa` to each device preset.

---

### [M8] units.py Unconditional `import torch`
- **File**: `units.py:31`
- **Found by**: py-engine (C2)
- **Cross-review verdict**: **CONFIRMED but DOWNGRADED from CRITICAL to MEDIUM**

- **Description**: `units.py` imports `torch` at module level. Any import of `dpf.units` without PyTorch fails with `ImportError`.

- **Downgrade justification**: Verified that the **only** consumer of `dpf.units` is `src/dpf/metal/metal_solver.py`, which itself requires PyTorch. No transitive import path from the core engine touches `units.py`. The torch import does not break any non-Metal workflow.

- **Proposed fix** (still recommended for defensive coding):
  ```python
  import numpy as np
  # Remove: import torch

  from dpf.constants import mu_0 as MU_0
  SQRT_MU_0 = np.sqrt(MU_0)

  def to_code_units(B_si):
      """Convert B from SI (Tesla) to Code Units (HL)."""
      return B_si / SQRT_MU_0  # Works for both numpy and torch via __truediv__
  ```

---

## LOW Priority Issues

### [L1] FieldManager.B Aliasing Risk
- **File**: `engine.py:630`
- **Found by**: py-engine (M2)
- **Cross-review verdict**: **CONFIRMED LOW**

- **Description**: `self.field_manager.B = self.state["B"]` creates a reference alias, not a copy. Currently safe because `compute_plasma_inductance()` is called before the fluid step modifies B. Any reordering would silently compute inductance from wrong B.

- **Proposed fix**: `self.field_manager.B = self.state["B"].copy()`

---

### [L2] compute_plasma_inductance Discontinuity at I = 1e-3 A
- **File**: `field_manager.py:49`
- **Found by**: py-engine (M3)
- **Cross-review verdict**: **CONFIRMED LOW**

- **Description**: Hard threshold at |I| < 1e-3 A returns fixed `min_inductance = 1e-9 H`. Crossing this threshold during ramp-up creates a discontinuity in L_p that spikes dL/dt. Only relevant during the first ~ns of simulation.

---

### [L3] Missing Kinetic CFL Constraint
- **File**: `engine.py:470-490`
- **Found by**: py-engine (M4)
- **Cross-review verdict**: **CONFIRMED LOW** (kinetic module is experimental/dormant)

---

### [L4] Diagnostic State Dict Aliases Live State
- **File**: `engine.py:975-1031`
- **Found by**: py-engine (M6)
- **Cross-review verdict**: **CONFIRMED LOW**

- **Description**: `{**self.state, ...}` is a shallow copy — numpy arrays are shared. Safe as long as `diagnostics.record()` doesn't modify arrays, which HDF5Writer typically doesn't.

---

### [L5] Duplicate "# Coupling" Comment
- **File**: `engine.py:228-229`
- **Found by**: py-engine (L1)
- **Cross-review verdict**: **CONFIRMED LOW** (trivial)

---

### [L6] Snowplow Initial z = 1e-4 m Is Arbitrary
- **File**: `snowplow.py:104`
- **Found by**: phys-engine (P2)
- **Cross-review verdict**: **CONFIRMED LOW**

- **Description**: The 0.1 mm offset avoids the zero-mass singularity. For PF-1000 (L=0.6 m), this is 0.017% — negligible. Could be improved with a physics-based breakdown initialization.

---

### [L7] Reflected Shock Uses Constant Slug Mass
- **File**: `snowplow.py:436-520`
- **Found by**: phys-engine (P4)
- **Cross-review verdict**: **CONFIRMED LOW** (model limitation, not a bug)

- **Description**: During reflected shock phase, slug mass is frozen at pinch value. In reality, the reflected shock sweeps additional mass. This is a known simplification of the 0D model.
- **Reference**: Auluck (2016) notes this is non-negligible but difficult to model in 0D.

---

### [L8] rho0 = 4e-4 for PF-1000 Needs Documentation
- **File**: `presets.py:51`
- **Found by**: phys-engine (m2)
- **Cross-review verdict**: **CONFIRMED LOW** (documentation gap, not a bug)

- **Description**: `rho0 = 4e-4 kg/m^3` matches **atomic** deuterium density at ~3 Torr, not molecular D2. The value is physically correct (post-dissociation assumption for DPF fill gas) but should be documented.

---

## REJECTED Findings

### [R1] py-engine H6 Severity Claim: mu_0 "5.5e-17 absolute" Difference
- **Rejection reason**: The actual runtime difference is ~1.66e-16 absolute (1.3e-10 relative), not the claimed 5.5e-17 (~4.4e-11 relative). The finding IS real (see M5 above) but the magnitude was misstated. The py-engine underestimated the actual discrepancy by ~3x in absolute terms but overestimated the severity as HIGH — the impact is negligible for MHD.

### [R2] phys-engine P3: Adiabatic Back-Pressure Formula Error
- **Rejection reason**: The physics expert initially flagged this as a concern but then self-corrected during the review. `p = p_0 * (b/r_s)^(2*gamma)` is **correct** for cylindrical compression with gamma = 5/3. Reference: Lee & Saw (2014), Section III.B. **Verified correct.**

### [R3] phys-engine M1: FieldManager Inductance Computation Bug
- **Rejection reason**: The physics expert self-retracted this finding after verifying both inductance computations (FieldManager and engine) are algebraically consistent: `L = sum(B^2/mu_0 * dV) / I^2`. **Verified correct.**

### [R4] py-engine M9: "AthenaK Goes Through Python Path" as Bug
- **Rejection reason**: This is **intentional design**, not a bug. AthenaK uses subprocess mode (no pybind11), so it goes through the Python operator-split loop to get collision, radiation, and coupling physics that are not available in the AthenaK binary. The Athena++ fast path skips these because Athena++ supports custom C++ source terms.

### [R5] py-engine M7: Powell In-Place + Reassign as Bug
- **Rejection reason**: The `np.squeeze()` returns a **view** of the original array. The in-place `+=` modifies the original through the view, then the explicit reassignment is redundant but harmless. This is not a bug — just slightly redundant code. The concern about `np.squeeze` returning a copy for non-contiguous arrays is valid in theory but doesn't apply here (the arrays ARE contiguous).

### [R6] py-engine L3: Snowplow Enabled by Default as Bug
- **Rejection reason**: This is a design choice, not a bug. Snowplow is the core DPF identity physics (PhD Debate #3 identified its absence as the primary "identity gap"). Having it enabled by default is the correct default for a DPF simulator. Non-DPF use cases (Cartesian demos, Sod tests) are unaffected because snowplow requires cylindrical geometry and active circuit current.

---

## Verified Correct (Commendations)

### Circuit Equation Formulation (rlc_solver.py:211-321)
- The implicit midpoint method is well-formulated, A-stable, and 2nd-order accurate.
- `R_star = R_eff + dLp_dt` correctly handles inductive back-EMF.
- Energy accounting separates external and plasma dissipation.
- **Reference**: Matches Lee (1984) Eq. 1-3; Hairer & Wanner "Solving ODEs II".

### BDF2 dL/dt (rlc_solver.py:138-189)
- 2nd-order backward difference with non-uniform timestep support and 1st-order fallback.
- History via `deque(maxlen=3)`. Correct for O(dt^2) temporal accuracy.

### Snowplow Force Coefficients (snowplow.py:92-101)
- `L_coeff = 2 * F_coeff` relationship correct (Phase T bug fix verified).
- **Reference**: Lee (1984) Eq. 5; Lee & Saw (2014) Eq. 3.

### Physical Constants (constants.py)
- All 7 physical constants verified against CODATA 2018: mu_0, k_B, m_e, m_p, m_d, e, c.
- m_d = 3.34358377e-27 kg matches to 9 significant figures.

### PF-1000 Circuit Parameters (presets.py)
- V0, C0, L0, R0, anode_radius, cathode_radius all verified against Scholz (2006) and Lee & Saw (2014).

### `int()` -> `round()` Fix for ir_shock
- **Verified at engine.py:1128**: `ir_shock = round(r_shock / dr)` — correctly uses `round()`.

### Radial Zipper BC Now Backend-Agnostic
- **Verified at engine.py:1113-1130**: The zipper BC writes directly to `self.state["B"]` which is a NumPy array for all backends. The original concern about `pass` for Athena++/AthenaK appears resolved.

---

## Summary Table

| ID | Severity | File | Line | Category | Issue | Verdict |
|----|----------|------|------|----------|-------|---------|
| C1 | CRITICAL | engine.py | 649+817 | Bug | Double circuit step (2x rate) | ✅ FIXED (already single-step) |
| C2 | CRITICAL | engine.py | 816 | Bug | Motional back-EMF discarded | ✅ FIXED |
| H1 | HIGH | engine.py | 1109 | Bug | Metal electrode BC 1/r inside anode | ✅ FIXED |
| H2 | HIGH | engine.py | 512 | State | Checkpoint misses snowplow state | ✅ FIXED |
| H3 | HIGH | config.py | 314 | Params | PF-1000 anode_length=0.16 vs 0.6 m | ✅ FIXED |
| M1 | MEDIUM | cyl_mhd.py | 477 | Physics | Electrode BC at open end (z=nz-1) | CONFIRMED |
| M2 | MEDIUM | engine.py | 907 | Bug | Stale Z_bar for 2nd Strang half | CONFIRMED |
| M3 | MEDIUM | engine.py | 628-907 | Design | Non-standard Strang splitting (O(dt)) | CONFIRMED |
| M4 | MEDIUM | engine.py | 564 | Design | Silent NaN repair masks instability | ✅ FIXED |
| M5 | MEDIUM | units.py | 34 | Constants | mu_0 differs by 1.3e-10 from scipy | ✅ FIXED |
| M6 | MEDIUM | config.py | 142 | Config | Unvalidated string fields | ✅ FIXED |
| M7 | MEDIUM | config.py | 310 | Params | fill_pressure wrong for NX2 | ✅ FIXED |
| M8 | MEDIUM | units.py | 31 | Import | Unconditional torch at module level | ✅ FIXED |
| L1 | LOW | engine.py | 630 | State | FieldManager.B aliasing | CONFIRMED |
| L2 | LOW | field_mgr.py | 49 | Numerics | Inductance discontinuity at 1mA | CONFIRMED |
| L3 | LOW | engine.py | 470 | CFL | No kinetic particle CFL | CONFIRMED |
| L4 | LOW | engine.py | 975 | State | Diagnostic state aliases | CONFIRMED |
| L5 | LOW | engine.py | 228 | Quality | Duplicate comment | ✅ FIXED |
| L6 | LOW | snowplow.py | 104 | Physics | Arbitrary z_init=1e-4 | CONFIRMED |
| L7 | LOW | snowplow.py | 466 | Model | Constant slug mass in reflected | CONFIRMED |
| L8 | LOW | presets.py | 51 | Docs | rho0=4e-4 undocumented assumption | CONFIRMED |
| R1-R6 | — | — | — | — | 6 findings rejected | REJECTED |
