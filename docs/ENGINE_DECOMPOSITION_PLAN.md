# Engine.py Decomposition Plan

**Document**: ENGINE_DECOMPOSITION_PLAN.md
**Date**: 2026-03-24
**Methodology**: Six Sigma DMAIC
**Risk Level**: P2 Tech Debt — behavior-preserving refactor only
**Current LOC**: 2,908 (limit: 400 per coding-style.md)

---

## Table of Contents

1. [DMAIC Phase 1: Define](#1-define)
2. [DMAIC Phase 2: Measure](#2-measure)
3. [DMAIC Phase 3: Analyze](#3-analyze)
4. [DMAIC Phase 4: Improve — Proposed Decomposition](#4-improve)
5. [DMAIC Phase 5: Control — Migration Plan](#5-control)
6. [Appendix A: Complete Method Map](#appendix-a-complete-method-map)
7. [Appendix B: Test Dependency Matrix](#appendix-b-test-dependency-matrix)
8. [Appendix C: Import Graph](#appendix-c-import-graph)

---

## 1. Define

### Problem Statement

`src/dpf/engine.py` is 2,908 LOC in a single file containing a single class (`SimulationEngine`) with 37 methods. The project's coding standard mandates a 400-LOC maximum per file. The file is 7.3x over the limit.

### Business Case

- **Readability**: New contributors (and AI agents) must load the entire 2,908-line file to understand any single responsibility.
- **Merge conflict surface**: Every physics change, backend addition, or diagnostics tweak touches the same file.
- **Test isolation**: Cannot unit-test circuit coupling without importing radiation, viscosity, and all 5 backend solvers.
- **Agent delegation**: Parallel agent work on engine features causes file-level conflicts. Splitting enables independent module work.

### Scope

- Split `SimulationEngine` into a thin orchestrator + 6 focused modules.
- All 4,861 existing tests must pass without modification (or with import-path-only changes).
- No physics changes. No behavioral changes. No performance changes.
- The public API (`from dpf.engine import SimulationEngine`) must remain unchanged.

### Success Criteria

- Every new file < 400 LOC.
- `pytest tests/ -x -q` passes (all 4,861 tests).
- `from dpf.engine import SimulationEngine` still works (backward compat).
- `ruff check src/dpf/engine/` passes.

---

## 2. Measure

### 2.1 Current State Metrics

| Metric | Value |
|--------|-------|
| Total LOC | 2,908 |
| Methods | 37 |
| Class attributes set in `__init__` | 58 |
| Import statements | 43 (lines 19-71) |
| Direct consumers (src/) | 6 files |
| Test files (active, non-archive) | 18 files |
| Test files (archive) | 47 files |
| Cyclomatic complexity (estimated) | High — `step()` dispatches to 8 sub-methods, each with geometry/backend branching |

### 2.2 Method Inventory with Line Ranges

| # | Method | Lines | LOC | Domain |
|---|--------|-------|-----|--------|
| 1 | `__init__` | 94-497 | 403 | Initialization |
| 2 | `engine_tier` (property) | 498-508 | 10 | Backend dispatch |
| 3 | `_resolve_backend` (static) | 511-605 | 94 | Backend dispatch |
| 4 | `_initial_state` | 607-631 | 24 | Initialization |
| 5 | `_compute_dt` | 633-653 | 20 | Step pipeline |
| 6 | `save_checkpoint` | 655-685 | 30 | State management |
| 7 | `load_from_checkpoint` | 687-725 | 38 | State management |
| 8 | `_sanitize_state` | 731-771 | 40 | State management |
| 9 | `step` | 777-859 | 82 | Step pipeline |
| 10 | `_step_init_fields` | 865-871 | 6 | Step pipeline |
| 11 | `_step_ionization_and_resistivity` | 873-1032 | 159 | Physics operators |
| 12 | `_step_circuit_subcycle` | 1034-1178 | 144 | Circuit coupling |
| 13 | `_step_pic` | 1180-1202 | 22 | Step pipeline |
| 14 | `_step_fluid_advance` | 1204-1301 | 97 | Step pipeline |
| 15 | `_step_post_fluid_corrections` | 1303-1357 | 54 | Step pipeline |
| 16 | `_step_diagnostics_and_yield` | 1359-1473 | 114 | State management |
| 17 | `_step_record_and_checkpoint` | 1475-1611 | 136 | State management |
| 18 | `_compute_J_from_B` | 1613-1637 | 24 | Physics operators |
| 19 | `_measure_ohmic_gap` | 1639-1666 | 27 | Circuit coupling |
| 20 | `_compute_ohmic_correction` | 1668-1693 | 25 | Circuit coupling |
| 21 | `_compute_snowplow_source_terms` | 1695-1883 | 188 | Circuit coupling |
| 22 | `_should_use_coupler` | 1885-1904 | 19 | Circuit coupling |
| 23 | `_compute_grid_mass` | 1906-1915 | 9 | State management |
| 24 | `_compute_back_emf` | 1917-1948 | 31 | Circuit coupling |
| 25 | `_apply_electrode_bc` | 1950-2015 | 65 | Circuit coupling |
| 26 | `_initialize_radial_bfield` | 2017-2083 | 66 | Circuit coupling |
| 27 | `_initialize_radial_state` | 2085-2160 | 75 | Circuit coupling |
| 28 | `_dynamic_sheath_pressure` | 2162-2200 | 38 | Circuit coupling |
| 29 | `_step_athena` | 2206-2313 | 107 | Backend dispatch |
| 30 | `_make_step_result` | 2315-2336 | 21 | State management |
| 31 | `_apply_collision_radiation` | 2342-2510 | 168 | Physics operators |
| 32 | `_apply_nernst` | 2516-2557 | 41 | Physics operators |
| 33 | `_apply_powell_sources` | 2563-2617 | 54 | Physics operators |
| 34 | `_apply_viscosity` | 2623-2700 | 77 | Physics operators |
| 35 | `_apply_diffusion` | 2706-2799 | 93 | Physics operators |
| 36 | `close` | 2805-2808 | 3 | State management |
| 37 | `get_field_snapshot` | 2810-2816 | 6 | State management |
| 38 | `run` | 2822-2908 | 86 | Step pipeline |

### 2.3 LOC by Responsibility Domain

| Domain | Methods | Total LOC | % of File |
|--------|---------|-----------|-----------|
| **Initialization** | 2 | 427 | 14.7% |
| **Step pipeline** | 7 | 367 | 12.6% |
| **Circuit coupling** | 9 | 653 | 22.5% |
| **Physics operators** | 6 | 616 | 21.2% |
| **State management** | 7 | 397 | 13.7% |
| **Backend dispatch** | 3 | 211 | 7.3% |
| **Imports + docstring** | - | 93 | 3.2% |
| **Blank lines / comments** | - | ~144 | 5.0% |
| **Total** | **37** | **2,908** | **100%** |

### 2.4 Shared State Analysis

The `SimulationEngine.__init__` sets 58 instance attributes. These are the state dependencies that make extraction non-trivial:

**Core state** (used by nearly all methods):
- `self.state` — MHD field dict
- `self.config` — SimulationConfig
- `self.circuit` — RLCSolver
- `self.fluid` — PlasmaSolverBase
- `self.time`, `self.step_count`
- `self.geometry_type`, `self.ion_mass`
- `self._coupling` — CouplingState

**Circuit coupling state** (used only by circuit methods):
- `self.snowplow` — SnowplowModel
- `self.coupler` — CircuitCoupler
- `self.coupling_mode`
- `self._prev_L_plasma`, `self._lp_blend_alpha`, `self._lp_blend_active`
- `self._prev_swept_mass`, `self._prev_radial_swept_mass`
- `self._radial_bfield_initialized`, `self._skip_next_fluid_step`
- `self._last_feedback`, `self._coupler_decision_cache`
- `self._ohmic_gap_history`, `self._last_ohmic_gap`

**Diagnostics state** (used only by diagnostics/recording):
- `self.diagnostics` — HDF5Writer
- `self.well_exporter` — WellExporter
- `self.diag_interval`, `self.well_interval`
- `self._energy_tracker`, `self._yield_tracker`
- `self._last_R_plasma`, `self._last_Z_bar`, `self._last_eta_anom`
- `self._last_div_B`, `self._last_neutron_rate`, etc.
- `self._last_pb_result`, `self._last_m0_result`
- `self._last_regime_result`, `self._last_fringe_shifts`
- `self.total_radiated_energy`, `self.total_neutron_yield`

**Physics operator state** (used only by collision/radiation/diffusion):
- `self.eos` — IdealEOS
- `self.rad_cfg` — RadiationConfig
- `self.sheath_cfg` — SheathConfig
- `self.boundary_cfg` — BoundaryConfig
- `self._nan_check_stride`

**Caching state** (performance optimization):
- `self._coupling_cache_stride`
- `self._cached_R_plasma`, `self._cached_L_plasma`, `self._cached_Z_bar`
- `self._cached_Z_bar_field`, `self._cached_eta_field`, `self._cached_eta_anom`

---

## 3. Analyze

### 3.1 Root Cause Analysis — Why Did engine.py Grow to 2,908 LOC?

**Root cause: Incremental feature accretion without periodic refactoring.**

The engine grew through 16+ development phases (A through P), each adding physics modules, backend support, or coupling mechanisms. Key growth events:

1. **Phase B (wire physics)**: Added collision, radiation, bremsstrahlung operators (+~200 LOC).
2. **Phase F (Athena++)**: Added `_step_athena()` and `_resolve_backend()` (+~200 LOC).
3. **Phase J (AthenaK)**: Extended backend dispatch for AthenaK (+~50 LOC).
4. **Phase M (Metal GPU)**: Extended `__init__` for Metal/MLX backend initialization (+~100 LOC).
5. **Phase O (physics accuracy)**: Added Powell sources, viscosity, diffusion operators (+~300 LOC).
6. **Phase S (snowplow)**: Added `_compute_snowplow_source_terms()` — 188 LOC alone, plus circuit subcycling and handoff logic (+~400 LOC).
7. **Phase U (coupling)**: Added ohmic gap correction, electrode BC, radial initialization (+~250 LOC).
8. **Diagnostics growth**: `_step_diagnostics_and_yield` and `_step_record_and_checkpoint` expanded as neutron yield, instability detection, interferometry, Pease-Braginskii checks, and Well export were added (+~250 LOC).

**Contributing factors:**

- **"Just add a method" path of least resistance**: Adding a new `_apply_X()` method to the engine class is always easier than creating a new module with proper interfaces.
- **State sharing**: All methods access `self.state`, `self.config`, and `self.circuit`, making extraction feel harder than it is.
- **No file-size CI gate**: The 400-LOC rule was documented but not enforced by CI.
- **Single-class architecture**: The engine was designed as one monolithic orchestrator rather than composed of focused subsystems.

### 3.2 Natural Seam Analysis

Analyzing method clusters by shared state access reveals 6 clean extraction boundaries:

**Seam 1 — Backend Dispatch** (low coupling to engine state):
- `_resolve_backend()` is already `@staticmethod` — zero coupling.
- `engine_tier` only reads `self.backend`.
- Backend-specific `__init__` blocks (lines 131-276) only need config.

**Seam 2 — Circuit Coupling** (tight internal coupling, loose external coupling):
- 9 methods share `self.snowplow`, `self.coupler`, `self._coupling`, `self._prev_L_plasma`, etc.
- Only interface with engine: reads `self.state`, `self.circuit`, `self.config`.
- The snowplow source term computation (188 LOC) is completely self-contained physics.

**Seam 3 — Physics Operators** (stateless — operate on state dict):
- `_apply_collision_radiation`, `_apply_nernst`, `_apply_powell_sources`, `_apply_viscosity`, `_apply_diffusion` are pure operator-split steps.
- Read `self.state` + `self.config`, write back into `self.state`.
- `_step_ionization_and_resistivity` reads state and produces coupling quantities.
- Only shared state: `self.eos`, `self.rad_cfg`, `self.sheath_cfg`, caching fields.

**Seam 4 — State Management** (diagnostics recording, checkpointing, sanitization):
- `save_checkpoint`, `load_from_checkpoint`, `_sanitize_state`, `_make_step_result`, `_step_diagnostics_and_yield`, `_step_record_and_checkpoint`, `close`, `get_field_snapshot`, `_compute_grid_mass`.
- Shares: `self.diagnostics`, `self.well_exporter`, `self._energy_tracker`, `self._yield_tracker`, many `self._last_*` tracking fields.

**Seam 5 — Initialization** (one-shot, no runtime coupling):
- `__init__` and `_initial_state` are construction-time only.
- The massive `__init__` naturally decomposes into: circuit init, solver factory (backend dispatch), diagnostics init, state init, coupling init.

**Seam 6 — Athena Fast Path** (self-contained alternative step):
- `_step_athena()` is a complete alternative timestep for Athena++/AthenaK backends.
- Only shared interface: `self.state`, `self.circuit`, `self._coupling`, `self.config`.

---

## 4. Improve — Proposed Decomposition

### 4.1 Target File Structure

```
src/dpf/engine/
    __init__.py              # Re-export SimulationEngine (backward compat)
    core.py                  # SimulationEngine class skeleton + step() + run()
    backend_dispatch.py      # Backend resolution, solver factory, engine_tier
    circuit_coupling.py      # Lp, back-EMF, electrode BC, snowplow source terms
    physics_operators.py     # Collision, radiation, Nernst, Powell, viscosity, diffusion
    state_management.py      # Sanitize, checkpoint, diagnostics recording, step result
    athena_step.py           # Athena++/AthenaK fast-path timestep
```

### 4.2 Detailed File Specifications

---

#### `src/dpf/engine/__init__.py`

**Responsibility**: Backward-compatible re-export.

**Contents**:
```python
from dpf.engine.core import SimulationEngine

__all__ = ["SimulationEngine"]
```

**LOC estimate**: 5

**Migration risk**: None. All existing `from dpf.engine import SimulationEngine` statements work unchanged.

---

#### `src/dpf/engine/core.py`

**Responsibility**: SimulationEngine class definition, `__init__`, `step()`, `run()`, and the step sub-method dispatch. This is the orchestrator — it calls into the other modules but contains no physics logic itself.

**Methods that stay here**:

| Method | Lines (current) | Reason |
|--------|-----------------|--------|
| `__init__` | 94-497 | Calls into backend_dispatch + state_management for setup |
| `_initial_state` | 607-631 | Part of initialization |
| `_compute_dt` | 633-653 | Central to step pipeline |
| `step` | 777-859 | Main orchestrator dispatch |
| `_step_init_fields` | 865-871 | Pre-step hook |
| `_step_pic` | 1180-1202 | Thin dispatch to KineticManager |
| `_step_fluid_advance` | 1204-1301 | MHD sub-stepping orchestration |
| `_step_post_fluid_corrections` | 1303-1357 | Post-MHD orchestration |
| `run` | 2822-2908 | Batch loop calling step() |
| `close` | 2805-2808 | Cleanup |
| `get_field_snapshot` | 2810-2816 | State access |

**LOC estimate**: ~380

The `__init__` will be shortened by extracting:
- Backend selection into `backend_dispatch.create_solver(config)` (~100 LOC saved)
- Diagnostics init into `state_management.init_diagnostics(config)` (~30 LOC saved)
- The long warning blocks (lines 454-496) into a helper

**Dependencies**:
- Imports: `backend_dispatch`, `circuit_coupling`, `physics_operators`, `state_management`, `athena_step`
- Imported by: `__init__.py` (re-export), all external consumers

---

#### `src/dpf/engine/backend_dispatch.py`

**Responsibility**: Backend resolution logic, solver factory, engine tier classification.

**Methods that move here**:

| Method | Lines (current) | LOC |
|--------|-----------------|-----|
| `_resolve_backend` (static) | 511-605 | 94 |
| `engine_tier` (property helper) | 498-508 | 10 |
| `create_solver(config, geometry_type)` | Extracted from `__init__` lines 131-276 | ~145 |
| `_log_backend_warnings(config, backend)` | Extracted from `__init__` lines 454-496 | ~42 |

**LOC estimate**: ~310

**New public interface**:
```python
def resolve_backend(requested: str) -> str:
    """Resolve backend string to canonical name."""

def create_solver(
    config: SimulationConfig,
    geometry_type: str,
) -> tuple[PlasmaSolverBase, float | np.ndarray]:
    """Create the fluid solver and return (solver, cell_volume)."""

def engine_tier(backend: str) -> str:
    """Return 'production' or 'teaching' based on backend."""

def log_backend_warnings(config: SimulationConfig, backend: str) -> None:
    """Log warnings about skipped physics modules for non-Python backends."""
```

**Dependencies**:
- Imports: `dpf.config.SimulationConfig`, all solver classes (lazy imports preserved)
- Imported by: `core.py`

---

#### `src/dpf/engine/circuit_coupling.py`

**Responsibility**: All circuit-plasma coupling logic — Lp computation, back-EMF, electrode boundary conditions, snowplow source terms, ohmic gap correction.

**Methods that move here**:

| Method | Lines (current) | LOC |
|--------|-----------------|-----|
| `_step_circuit_subcycle` | 1034-1178 | 144 |
| `_compute_snowplow_source_terms` | 1695-1883 | 188 |
| `_should_use_coupler` | 1885-1904 | 19 |
| `_compute_grid_mass` | 1906-1915 | 9 |
| `_compute_back_emf` | 1917-1948 | 31 |
| `_apply_electrode_bc` | 1950-2015 | 65 |
| `_initialize_radial_bfield` | 2017-2083 | 66 |
| `_initialize_radial_state` | 2085-2160 | 75 |
| `_dynamic_sheath_pressure` | 2162-2200 | 38 |
| `_compute_J_from_B` | 1613-1637 | 24 |
| `_measure_ohmic_gap` | 1639-1666 | 27 |
| `_compute_ohmic_correction` | 1668-1693 | 25 |

**LOC estimate**: ~390 (including class definition and docstrings)

**Note**: At 390 LOC this is right at the limit. If it exceeds 400 during implementation, split `_compute_snowplow_source_terms` (188 LOC) into a separate `snowplow_coupling.py`.

**Design pattern**: Mixin class or composed helper. The mixin pattern preserves `self.` access to engine state:

```python
class CircuitCouplingMixin:
    """Circuit-plasma coupling methods for SimulationEngine.

    Mixed into SimulationEngine to provide:
    - Circuit sub-cycling with snowplow dynamics
    - Electrode boundary conditions
    - Back-EMF computation
    - Ohmic gap measurement and correction
    - Snowplow source terms for MHD grid
    """

    def _step_circuit_subcycle(self, dt, R_plasma, L_plasma, Z_bar):
        ...
```

**Dependencies**:
- Imports: `numpy`, `dpf.constants`, `dpf.circuit.coupler`, `dpf.core.bases.CouplingState`
- Imported by: `core.py` (via mixin inheritance)

---

#### `src/dpf/engine/physics_operators.py`

**Responsibility**: All operator-split physics steps — collision/radiation, Nernst, Powell, viscosity, diffusion, ionization/resistivity.

**Methods that move here**:

| Method | Lines (current) | LOC |
|--------|-----------------|-----|
| `_step_ionization_and_resistivity` | 873-1032 | 159 |
| `_apply_collision_radiation` | 2342-2510 | 168 |
| `_apply_nernst` | 2516-2557 | 41 |
| `_apply_powell_sources` | 2563-2617 | 54 |
| `_apply_viscosity` | 2623-2700 | 77 |
| `_apply_diffusion` | 2706-2799 | 93 |

**LOC estimate**: ~390 (borderline — see note)

**Note**: At 390 LOC this is right at the limit. If it exceeds 400, split into:
- `physics_operators.py` — collision/radiation/ionization (~370 LOC)
- `diffusion_operators.py` — Nernst, Powell, viscosity, diffusion (~220 LOC)

**Design pattern**: Mixin class.

```python
class PhysicsOperatorsMixin:
    """Operator-split physics steps for SimulationEngine.

    Mixed into SimulationEngine to provide Strang-split:
    - Collision (electron-ion temperature relaxation)
    - Radiation (bremsstrahlung, line, FLD transport)
    - Nernst B-field advection
    - Powell 8-wave div(B) sources
    - Braginskii ion viscosity
    - Implicit/STS magnetic and thermal diffusion
    - Ionization state and resistivity computation
    """

    def _step_ionization_and_resistivity(self):
        ...

    def _apply_collision_radiation(self, dt_sub, Z_bar, *, Z_bar_field=None):
        ...
```

**Dependencies**:
- Imports: `numpy`, `dpf.constants`, `dpf.collision.spitzer`, `dpf.radiation.*`, `dpf.fluid.viscosity`, `dpf.fluid.implicit_diffusion`, `dpf.fluid.super_time_step`, `dpf.fluid.anisotropic_conduction`, `dpf.fluid.nernst`, `dpf.fluid.ionization`, `dpf.fluid.mhd_solver` (Powell), `dpf.atomic.ionization`, `dpf.turbulence.anomalous`, `dpf.sheath.bohm`
- Imported by: `core.py` (via mixin inheritance)

---

#### `src/dpf/engine/state_management.py`

**Responsibility**: State sanitization, checkpoint save/load, diagnostics recording, step result construction, yield/energy tracking.

**Methods that move here**:

| Method | Lines (current) | LOC |
|--------|-----------------|-----|
| `_sanitize_state` | 731-771 | 40 |
| `save_checkpoint` | 655-685 | 30 |
| `load_from_checkpoint` | 687-725 | 38 |
| `_step_diagnostics_and_yield` | 1359-1473 | 114 |
| `_step_record_and_checkpoint` | 1475-1611 | 136 |
| `_make_step_result` | 2315-2336 | 21 |

**LOC estimate**: ~390

**Design pattern**: Mixin class.

```python
class StateManagementMixin:
    """State management, diagnostics, and checkpointing for SimulationEngine.

    Mixed into SimulationEngine to provide:
    - NaN/Inf sanitization with cumulative repair tracking
    - HDF5 checkpoint save/load
    - Diagnostics recording (energy, yield, interferometry, regime validity)
    - StepResult construction
    """

    def _sanitize_state(self, label: str) -> int:
        ...
```

**Dependencies**:
- Imports: `numpy`, `dpf.diagnostics.*`, `dpf.core.bases.StepResult`
- Imported by: `core.py` (via mixin inheritance)

---

#### `src/dpf/engine/athena_step.py`

**Responsibility**: The Athena++/AthenaK/Hybrid backend fast-path timestep.

**Methods that move here**:

| Method | Lines (current) | LOC |
|--------|-----------------|-----|
| `_step_athena` | 2206-2313 | 107 |

**LOC estimate**: ~130 (including imports, class definition, docstring)

**Design pattern**: Mixin class.

```python
class AthenaStepMixin:
    """Athena++/AthenaK backend timestep for SimulationEngine."""

    def _step_athena(self, dt, sim_time, _max_steps):
        ...
```

**Dependencies**:
- Imports: `contextlib`, `numpy`, `dpf.diagnostics.pease_braginskii`
- Imported by: `core.py` (via mixin inheritance)

---

### 4.3 Final Class Assembly in core.py

```python
from dpf.engine.circuit_coupling import CircuitCouplingMixin
from dpf.engine.physics_operators import PhysicsOperatorsMixin
from dpf.engine.state_management import StateManagementMixin
from dpf.engine.athena_step import AthenaStepMixin


class SimulationEngine(
    CircuitCouplingMixin,
    PhysicsOperatorsMixin,
    StateManagementMixin,
    AthenaStepMixin,
):
    """Dense Plasma Focus simulation engine.

    Orchestrates the coupled circuit-plasma simulation loop.
    Physics operators, circuit coupling, and state management are
    provided by mixin classes in sibling modules.
    """

    def __init__(self, config: SimulationConfig) -> None:
        ...  # Shortened __init__ using backend_dispatch helpers

    def step(self, *, _max_steps: int | None = None) -> StepResult:
        ...  # Orchestrator — calls mixin methods

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        ...
```

### 4.4 LOC Summary

| File | LOC (estimated) | Under 400? |
|------|-----------------|------------|
| `__init__.py` | 5 | Yes |
| `core.py` | ~380 | Yes |
| `backend_dispatch.py` | ~310 | Yes |
| `circuit_coupling.py` | ~390 | Yes (marginal) |
| `physics_operators.py` | ~390 | Yes (marginal) |
| `state_management.py` | ~390 | Yes (marginal) |
| `athena_step.py` | ~130 | Yes |
| **Total** | **~1,995** | — |

**Note**: The total is lower than 2,908 because imports are consolidated (each module imports only what it needs, eliminating duplicate imports), and the 93-line module-level docstring is replaced by per-module docstrings.

### 4.5 Contingency: If Mixin Files Exceed 400 LOC

Three files are marginal (~390 LOC). If any exceeds 400 during implementation:

- **circuit_coupling.py** > 400: Extract `_compute_snowplow_source_terms` (188 LOC) into `snowplow_coupling.py`.
- **physics_operators.py** > 400: Split into `collision_radiation.py` (ionization + collision + radiation) and `diffusion_operators.py` (Nernst + Powell + viscosity + diffusion).
- **state_management.py** > 400: Extract `_step_record_and_checkpoint` (136 LOC) into `diagnostics_recording.py`.

---

## 5. Control — Migration Plan

### 5.1 Migration Strategy: Mixin Extraction

The mixin pattern was chosen over composition (helper objects) for a critical reason: **zero changes to method signatures or `self.` access patterns**. Every extracted method continues to access `self.state`, `self.config`, `self.circuit`, etc. exactly as before. The only difference is which file the method's source code lives in.

This makes the refactor a pure file reorganization with deterministic correctness — if the file compiles and the class resolves, the behavior is identical.

### 5.2 Step-by-Step Migration Checklist

Each step is independently testable and independently revertible.

#### Pre-flight

- [ ] **Baseline test run**: `pytest tests/ -x -q` — record pass count (expected: 4,861)
- [ ] **Baseline ruff**: `ruff check src/dpf/engine.py` — clean
- [ ] **Git branch**: `git checkout -b refactor/engine-decomposition`
- [ ] **Snapshot**: `cp src/dpf/engine.py src/dpf/engine.py.bak`

#### Step 1: Create package structure (no code moves yet)

```bash
mkdir -p src/dpf/engine
touch src/dpf/engine/__init__.py
```

Write `__init__.py`:
```python
# Backward compatibility: re-export from the monolithic module
# during migration. Will switch to core.py import after migration.
from dpf.engine_monolith import SimulationEngine  # temporary

__all__ = ["SimulationEngine"]
```

Rename: `mv src/dpf/engine.py src/dpf/engine_monolith.py`

Update all `from dpf.engine import SimulationEngine` to work via the `__init__.py` re-export. Since Python resolves `dpf.engine` as the package now, the `__init__.py` handles the redirect.

- [ ] **Test**: `pytest tests/ -x -q` — all pass
- [ ] **Commit**: `refactor: create engine package with backward-compat re-export`

#### Step 2: Extract `backend_dispatch.py`

Move `_resolve_backend` and `engine_tier` to `src/dpf/engine/backend_dispatch.py`. Extract solver creation logic from `__init__` lines 131-276 into `create_solver()`. Extract warning logic from lines 454-496 into `log_backend_warnings()`.

In `engine_monolith.py`:
- Import and call `resolve_backend()`, `create_solver()`, `log_backend_warnings()` from `backend_dispatch`
- Delete the moved code blocks
- Keep `engine_tier` as a thin property calling `backend_dispatch.engine_tier(self.backend)`

- [ ] **Test**: `pytest tests/ -x -q` — all pass
- [ ] **Ruff**: `ruff check src/dpf/engine/`
- [ ] **Commit**: `refactor: extract backend_dispatch from engine`

#### Step 3: Extract `athena_step.py`

Move `_step_athena` to `AthenaStepMixin` in `src/dpf/engine/athena_step.py`.

In `engine_monolith.py`:
- Add `AthenaStepMixin` to `SimulationEngine` base classes
- Delete `_step_athena` from the monolith

- [ ] **Test**: `pytest tests/ -x -q` — all pass
- [ ] **Commit**: `refactor: extract athena_step mixin from engine`

#### Step 4: Extract `state_management.py`

Move `_sanitize_state`, `save_checkpoint`, `load_from_checkpoint`, `_step_diagnostics_and_yield`, `_step_record_and_checkpoint`, `_make_step_result` to `StateManagementMixin`.

- [ ] **Test**: `pytest tests/ -x -q` — all pass
- [ ] **Commit**: `refactor: extract state_management mixin from engine`

#### Step 5: Extract `circuit_coupling.py`

Move all 12 circuit coupling methods to `CircuitCouplingMixin`.

- [ ] **Test**: `pytest tests/ -x -q` — all pass
- [ ] **Commit**: `refactor: extract circuit_coupling mixin from engine`

#### Step 6: Extract `physics_operators.py`

Move all 6 physics operator methods to `PhysicsOperatorsMixin`.

- [ ] **Test**: `pytest tests/ -x -q` — all pass
- [ ] **Commit**: `refactor: extract physics_operators mixin from engine`

#### Step 7: Finalize — rename monolith to core.py

Rename `engine_monolith.py` to `core.py`. Update `__init__.py`:

```python
from dpf.engine.core import SimulationEngine

__all__ = ["SimulationEngine"]
```

Delete `engine.py.bak` backup.

- [ ] **Test**: `pytest tests/ -x -q` — all pass (4,861)
- [ ] **Ruff**: `ruff check src/dpf/engine/`
- [ ] **LOC check**: Every file < 400 LOC
- [ ] **Import check**: `python3 -c "from dpf.engine import SimulationEngine; print('OK')"`
- [ ] **Commit**: `refactor: finalize engine package decomposition`

#### Step 8: Clean up archive test imports (optional, low priority)

The 47 archived test files in `tests/_archive/` import `from dpf.engine import SimulationEngine`. These continue to work via the `__init__.py` re-export. No changes needed unless archive tests are un-archived.

### 5.3 Rollback Procedure

At any step, if tests fail:

```bash
# Full rollback to monolithic engine.py
git checkout main -- src/dpf/engine.py
rm -rf src/dpf/engine/
rm -f src/dpf/engine_monolith.py
pytest tests/ -x -q  # verify baseline restored
```

If only a single extraction step fails:

```bash
git revert HEAD  # revert the last extraction commit
pytest tests/ -x -q  # verify previous state restored
```

### 5.4 CI Integration

After migration is complete, add a file-size check to prevent re-growth:

```yaml
# .github/workflows/file-size-check.yml
- name: Check file sizes
  run: |
    find src/dpf/ -name "*.py" -exec wc -l {} + | \
      awk '$1 > 400 && !/^[[:space:]]*[0-9]+ total/ { print "OVER LIMIT:", $0; exit 1 }'
```

### 5.5 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Circular import between mixin modules | Low | Medium | Mixins only import from `dpf.*` packages, never from each other |
| `self.` attribute not found at runtime | Low | High | Mixins are mixed into SimulationEngine which sets all attrs in `__init__` |
| Type checker confused by mixin pattern | Medium | Low | Add `TYPE_CHECKING` protocol or `# type: ignore` annotations |
| Archive tests break | None | None | `__init__.py` re-export covers all import paths |
| Performance regression from mixin MRO | None | None | Python MRO resolution is O(1) cached after first lookup |
| External consumers break | None | None | Only 6 src/ files import engine, all via `from dpf.engine import SimulationEngine` |

### 5.6 Estimated Effort

| Step | Time (estimated) |
|------|------------------|
| Pre-flight + Step 1 (package setup) | 15 min |
| Step 2 (backend_dispatch) | 30 min |
| Step 3 (athena_step) | 15 min |
| Step 4 (state_management) | 30 min |
| Step 5 (circuit_coupling) | 45 min |
| Step 6 (physics_operators) | 30 min |
| Step 7 (finalize) | 15 min |
| Step 8 (cleanup, optional) | 15 min |
| **Total** | **~3 hours** |

This estimate assumes no test failures during migration. Budget an additional 1-2 hours for debugging if issues arise.

---

## Appendix A: Complete Method Map

Every method in the current `engine.py` mapped to its destination:

| Method | Current Lines | Destination | Domain |
|--------|--------------|-------------|--------|
| `__init__` | 94-497 | `core.py` (shortened) | Init |
| `engine_tier` | 498-508 | `core.py` (delegates to `backend_dispatch`) | Backend |
| `_resolve_backend` | 511-605 | `backend_dispatch.py` | Backend |
| `_initial_state` | 607-631 | `core.py` | Init |
| `_compute_dt` | 633-653 | `core.py` | Step |
| `save_checkpoint` | 655-685 | `state_management.py` | State |
| `load_from_checkpoint` | 687-725 | `state_management.py` | State |
| `_sanitize_state` | 731-771 | `state_management.py` | State |
| `step` | 777-859 | `core.py` | Step |
| `_step_init_fields` | 865-871 | `core.py` | Step |
| `_step_ionization_and_resistivity` | 873-1032 | `physics_operators.py` | Physics |
| `_step_circuit_subcycle` | 1034-1178 | `circuit_coupling.py` | Circuit |
| `_step_pic` | 1180-1202 | `core.py` | Step |
| `_step_fluid_advance` | 1204-1301 | `core.py` | Step |
| `_step_post_fluid_corrections` | 1303-1357 | `core.py` | Step |
| `_step_diagnostics_and_yield` | 1359-1473 | `state_management.py` | State |
| `_step_record_and_checkpoint` | 1475-1611 | `state_management.py` | State |
| `_compute_J_from_B` | 1613-1637 | `circuit_coupling.py` | Circuit |
| `_measure_ohmic_gap` | 1639-1666 | `circuit_coupling.py` | Circuit |
| `_compute_ohmic_correction` | 1668-1693 | `circuit_coupling.py` | Circuit |
| `_compute_snowplow_source_terms` | 1695-1883 | `circuit_coupling.py` | Circuit |
| `_should_use_coupler` | 1885-1904 | `circuit_coupling.py` | Circuit |
| `_compute_grid_mass` | 1906-1915 | `circuit_coupling.py` | Circuit |
| `_compute_back_emf` | 1917-1948 | `circuit_coupling.py` | Circuit |
| `_apply_electrode_bc` | 1950-2015 | `circuit_coupling.py` | Circuit |
| `_initialize_radial_bfield` | 2017-2083 | `circuit_coupling.py` | Circuit |
| `_initialize_radial_state` | 2085-2160 | `circuit_coupling.py` | Circuit |
| `_dynamic_sheath_pressure` | 2162-2200 | `circuit_coupling.py` | Circuit |
| `_step_athena` | 2206-2313 | `athena_step.py` | Backend |
| `_make_step_result` | 2315-2336 | `state_management.py` | State |
| `_apply_collision_radiation` | 2342-2510 | `physics_operators.py` | Physics |
| `_apply_nernst` | 2516-2557 | `physics_operators.py` | Physics |
| `_apply_powell_sources` | 2563-2617 | `physics_operators.py` | Physics |
| `_apply_viscosity` | 2623-2700 | `physics_operators.py` | Physics |
| `_apply_diffusion` | 2706-2799 | `physics_operators.py` | Physics |
| `close` | 2805-2808 | `core.py` | Step |
| `get_field_snapshot` | 2810-2816 | `core.py` | Step |
| `run` | 2822-2908 | `core.py` | Step |

---

## Appendix B: Test Dependency Matrix

### Active Test Files (18 files importing SimulationEngine)

All import via `from dpf.engine import SimulationEngine`. The `__init__.py` re-export ensures zero changes needed.

| Test File | Test Count (approx) | Engine Methods Exercised |
|-----------|---------------------|--------------------------|
| `test_mhd_solver_consolidated.py` | ~80 | step, run, _compute_dt |
| `test_validation_consolidated.py` | ~60 | run, step, save/load_checkpoint |
| `test_verification_consolidated.py` | ~40 | step, run, _apply_collision_radiation |
| `test_circuit_consolidated.py` | ~50 | step, _step_circuit_subcycle |
| `test_circuit_coupler.py` | ~20 | _should_use_coupler, _step_circuit_subcycle |
| `test_snowplow_consolidated.py` | ~40 | _compute_snowplow_source_terms, _step_circuit_subcycle |
| `test_metal_gpu_consolidated.py` | ~35 | step, run (Metal backend) |
| `test_mlx_pf1000.py` | ~10 | run (MLX backend) |
| `test_mlx_cross_backend.py` | ~15 | step (cross-backend parity) |
| `test_mlx_engine_integration.py` | ~20 | step, _apply_collision_radiation |
| `test_athena_consolidated.py` | ~30 | _step_athena, _resolve_backend |
| `test_physics.py` | ~25 | _apply_collision_radiation, _apply_viscosity |
| `test_two_temperature.py` | ~15 | step, _apply_collision_radiation |
| `test_calibration_consolidated.py` | ~20 | run |
| `test_research_consolidated.py` | ~10 | run |
| `test_infrastructure_consolidated.py` | ~30 | save/load_checkpoint, close |
| `test_web_ui_consolidated.py` | ~15 | step (via server) |
| `test_walrus_consolidated.py` | ~20 | run (hybrid backend) |

### Archive Test Files (47 files)

Located in `tests/_archive/`. All import via `from dpf.engine import SimulationEngine`. No changes needed — the `__init__.py` re-export handles them.

### Source Consumers (6 files)

| File | Import Statement | Impact |
|------|------------------|--------|
| `src/dpf/server/simulation.py:19` | `from dpf.engine import SimulationEngine` | None (re-export) |
| `src/dpf/cli/main.py:51,219,313` | `from dpf.engine import SimulationEngine` | None (re-export) |
| `src/dpf/ai/hybrid_engine.py:68` | `from dpf.engine import SimulationEngine` | None (re-export) |
| `src/dpf/ai/batch_runner.py:191` | `from dpf.engine import SimulationEngine` | None (re-export) |

---

## Appendix C: Import Graph

### engine.py Imports (43 imports, lines 19-71)

**Standard library**: `contextlib`, `logging`, `time`
**Typing**: `Any` from `typing`

**dpf.atomic**: `saha_ionization_fraction_array`
**dpf.circuit**: `CircuitCoupler`, `FeedbackResult`, `RLCSolver`
**dpf.collision**: `coulomb_log`, `nu_ei`, `relax_temperatures`, `spitzer_resistivity`
**dpf.config**: `SimulationConfig`
**dpf.constants**: `eV`, `k_B`, `m_e`, `pi`, `mu_0`
**dpf.core**: `CouplingState`, `StepResult`, `FieldManager`
**dpf.diagnostics**: `load_checkpoint`, `save_checkpoint`, `EnergyTracker`, `HDF5Writer`, `abel_transform`, `fringe_shift`, `check_pease_braginskii`, `regime_validity`, `YieldTracker`
**dpf.fluid**: `anisotropic_thermal_conduction`, `CylindricalMHDSolver`, `IdealEOS`, `implicit_resistive_diffusion`, `implicit_thermal_diffusion`, `coronal_z_eff`, `MHDSolver`, `powell_source_terms`, `powell_source_terms_cylindrical`, `apply_nernst_advection`, `SnowplowModel`, `rkl2_diffusion_3d`, `rkl2_thermal_step`, `braginskii_eta0`, `braginskii_eta1`, `ion_collision_time`, `viscous_heating_rate`, `viscous_stress_rate`
**dpf.kinetic**: `KineticManager`
**dpf.radiation**: `apply_bremsstrahlung_losses`, `apply_line_radiation_losses`, `apply_radiation_transport`
**dpf.sheath**: `apply_sheath_bc`, `floating_potential`
**dpf.turbulence**: `anomalous_resistivity_field`, `anomalous_resistivity_scalar`, `total_resistivity`

**Lazy imports** (inside methods):
- `dpf.athena_wrapper.AthenaPPSolver` (lines 136, 150)
- `dpf.athenak_wrapper.AthenaKSolver` (line 145)
- `dpf.metal.metal_solver.MetalMHDSolver` (line 160)
- `dpf.metal.mlx_solver.MLXMHDSolver` (line 200)
- `dpf.geometry.cylindrical.CylindricalGeometry` (lines 189, 219)
- `dpf.io.well_exporter.WellExporter` (line 292)
- `dpf.fluid.two_temperature.initialize_electron_energy` (line 626)
- `dpf.atomic.ablation.ablation_source_array` (line 1280)
- `dpf.experimental.poloidal_bfield.add_poloidal_field` (line 1320)
- `dpf.diagnostics.instability.m0_growth_rate_from_state` (line 1430)
- `dpf.ai.hybrid_engine.HybridEngine` (line 2834)
- `dpf.ai.surrogate.DPFSurrogate` (line 2835)

### What Imports engine.py (Reverse Dependencies)

```
src/dpf/server/simulation.py    → SimulationEngine
src/dpf/cli/main.py             → SimulationEngine (3 locations, all lazy)
src/dpf/ai/hybrid_engine.py     → SimulationEngine (lazy)
src/dpf/ai/batch_runner.py      → SimulationEngine (lazy)
tests/ (18 active + 47 archive) → SimulationEngine
app_engine.py                   → SimulationEngine
```

All external consumers import only `SimulationEngine` — the class name. No external code imports internal methods. The `__init__.py` re-export is sufficient for 100% backward compatibility.

---

## Summary

| Aspect | Current | After Decomposition |
|--------|---------|---------------------|
| Files | 1 (2,908 LOC) | 7 files (max 390 LOC each) |
| Max file size | 2,908 LOC | ~390 LOC |
| Public API | `from dpf.engine import SimulationEngine` | Same (via `__init__.py`) |
| Test changes needed | N/A | Zero |
| Migration steps | N/A | 8 (each independently testable + revertible) |
| Estimated effort | N/A | ~3 hours |
| Design pattern | Monolithic class | Mixin composition |
| Risk | N/A | Low (pure code reorganization, no behavior change) |
