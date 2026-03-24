# Phase B Sprint 1 (Sprint 0 per Research Doc): MLX Foundation + Metal Kernels

**Date**: 2026-03-24
**Author**: dpf-engine-architect
**Status**: Implementation-ready
**Duration**: Week 1-2 (estimated 5-7 working days)
**Depends on**: Phase A complete (15/16 Metal validation tests passing)

---

## 1. Objective

Sprint 1 delivers the MLX foundation layer for the clean-room MHD solver:

1. **MLX device manager** -- detection, dtype helpers, stream management
2. **Cylindrical grid geometry** -- face areas, cell volumes, radii as cached `mx.array`
3. **State dict bridge** -- pack/unpack between DPF `dict[str, np.ndarray]` and `(10, nr, nz)` `mx.array`
4. **All 3 Metal kernels** integrated and tested against NumPy references:
   - Ghost cell padding with electrode BCs
   - HLLD Riemann solver (Miyoshi & Kusano 2005)
   - Cylindrical geometric source terms with L'Hopital at axis

At the end of this sprint, the kernel layer is validated and ready for Sprint 2 (WENO5-Z reconstruction + Riemann pipeline).

---

## 2. File Manifest

### New Production Files

| # | Path | Purpose | Est. LOC |
|---|------|---------|----------|
| 1 | `src/dpf/metal/mlx_device.py` | MLX availability detection, dtype helpers, Metal stream management | 120 |
| 2 | `src/dpf/metal/mlx_grid.py` | Cylindrical grid: radii, face areas, cell volumes as `mx.array` | 100 |
| 3 | `src/dpf/metal/mlx_state.py` | State dict <-> packed `(10, nr, nz)` mx.array conversion | 150 |
| 4 | `src/dpf/metal/mlx_kernels.py` | 3 Metal kernels: ghost pad, HLLD, cylindrical source + Python wrappers | 400 |

**Subtotal production**: ~770 LOC

### New Test Files

| # | Path | Purpose | Est. LOC |
|---|------|---------|----------|
| 5 | `tests/test_mlx_device.py` | MLX availability, dtype selection, graceful fallback | 80 |
| 6 | `tests/test_mlx_grid.py` | Grid geometry correctness, face area / volume consistency | 100 |
| 7 | `tests/test_mlx_state.py` | Round-trip state packing, zero-copy verification, edge cases | 120 |
| 8 | `tests/test_mlx_kernels.py` | All 3 kernels vs NumPy reference at multiple grid sizes | 350 |

**Subtotal tests**: ~650 LOC

### Modified Files

| Path | Change | LOC Delta |
|------|--------|-----------|
| `src/dpf/metal/device.py` | Add `has_mlx()` function + MLX version detection | +25 |
| `src/dpf/metal/__init__.py` | Export new MLX modules (guarded by `HAS_MLX`) | +10 |

**Grand total**: ~1,455 LOC

---

## 3. Implementation Order

### Day 1: Device Layer + Grid Geometry

**Task 1.1**: `mlx_device.py` (no dependencies)

Create the MLX device abstraction layer. This file has zero dependencies on other new files and can be written and tested immediately.

**Task 1.2**: `mlx_grid.py` (depends on 1.1)

Create the cylindrical grid geometry with all arrays pre-computed as `mx.array`. Depends only on `mlx_device` for dtype helpers.

**Task 1.3**: `device.py` modification

Add `has_mlx()` to the existing device detection module.

**Task 1.4**: `test_mlx_device.py` + `test_mlx_grid.py`

Unit tests for both modules. Run immediately after implementation.

### Day 2: State Dict Bridge

**Task 2.1**: `mlx_state.py` (depends on 1.2)

Build the state dict packing/unpacking layer. Depends on `mlx_grid` for geometry validation.

**Task 2.2**: `test_mlx_state.py`

Round-trip tests, shape validation, zero-copy verification.

### Day 3-4: Metal Kernels

**Task 3.1**: `mlx_kernels.py` -- Ghost Cell Kernel (depends on 1.1)

Port the ghost cell padding kernel from `docs/metal_v2_research/kernels/ghost_cell_kernel.py`. Include both the MSL source and the NumPy reference implementation in a single module.

**Task 3.2**: `mlx_kernels.py` -- HLLD Kernel (depends on 1.1)

Port the HLLD Riemann solver kernel from `docs/metal_v2_research/kernels/hlld_kernel.py`.

**Task 3.3**: `mlx_kernels.py` -- Cylindrical Source Kernel (depends on 1.1)

Port the cylindrical source kernel from `docs/metal_v2_research/kernels/cylindrical_source_kernel.py`.

**Task 3.4**: `test_mlx_kernels.py`

Full kernel validation suite: each kernel at 3 grid sizes, edge cases, NaN safety.

### Day 5: Integration + Polish

**Task 4.1**: `__init__.py` update

Export new modules with `HAS_MLX` guards.

**Task 4.2**: Run full test suite

Verify no regressions in existing 1475 tests. All new tests pass.

**Task 4.3**: Ruff + type checking

Clean lint. Type hints on all public APIs.

### Dependency DAG

```
mlx_device (1.1)
  |
  +-- mlx_grid (1.2)
  |     |
  |     +-- mlx_state (2.1)
  |
  +-- mlx_kernels (3.1, 3.2, 3.3)

device.py mod (1.3) -- independent
__init__.py mod (4.1) -- after all production code
```

---

## 4. MLXState Class Design

### File: `src/dpf/metal/mlx_state.py`

```python
"""State dict <-> packed mx.array conversion for cylindrical MHD.

The MLX solver internally uses a packed (10, nr, nz) mx.array:
  [rho, rho*vr, rho*vz, rho*vtheta, E, S*rho, Br, Bz, Btheta, e_electron]

The DPF engine uses dict[str, np.ndarray] with keys:
  {rho, velocity, pressure, B, Te, Ti, psi}

MLXState handles the conversion, including:
  - Primitive -> conserved variable transformation (pack)
  - Conserved -> primitive transformation (unpack)
  - Zero-copy NumPy <-> MLX transfer on Apple Silicon unified memory
  - Unit conversion (SI -> Heaviside-Lorentz for B fields)
"""

from __future__ import annotations

import numpy as np

from dpf.metal.mlx_device import require_mlx

mx = require_mlx()  # lazy import with error message


# -- Variable index constants (conserved) --
NVAR = 10
IDN = 0   # density
IMR = 1   # radial momentum
IMZ = 2   # axial momentum
IMT = 3   # azimuthal momentum
IEN = 4   # total energy
ISR = 5   # entropy tracer (S * rho)
IBR = 6   # Br
IBZ = 7   # Bz
IBT = 8   # Btheta
IEE = 9   # electron energy density

# DPF state dict keys consumed/produced
DPF_KEYS = ("rho", "velocity", "pressure", "B", "Te", "Ti", "psi")


class MLXState:
    """Manages conversion between DPF state dicts and packed MLX arrays.

    Parameters
    ----------
    nr : int
        Number of radial cells.
    nz : int
        Number of axial cells.
    gamma : float
        Adiabatic index (default 5/3).
    ion_mass : float
        Ion mass [kg] for temperature derivation (default deuterium).

    Attributes
    ----------
    U : mx.array
        Packed conserved state, shape (10, nr, nz), float32.
    nr, nz : int
        Grid dimensions.
    """

    def __init__(
        self,
        nr: int,
        nz: int,
        gamma: float = 5.0 / 3.0,
        ion_mass: float = 3.34358377e-27,
    ) -> None:
        self.nr = nr
        self.nz = nz
        self.gamma = gamma
        self.ion_mass = ion_mass
        self._k_B = 1.380649e-23
        self.U: mx.array = mx.zeros((NVAR, nr, nz), dtype=mx.float32)

    def from_state_dict(
        self,
        state: dict[str, np.ndarray],
        convert_b_si_to_hl: bool = False,
    ) -> mx.array:
        """Pack a DPF state dict into conserved (10, nr, nz) mx.array.

        Performs primitive -> conserved conversion:
          rho*v = rho * velocity
          E = p/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2
          S*rho = rho * (p / rho^gamma)  [entropy tracer]

        Parameters
        ----------
        state : dict[str, np.ndarray]
            DPF state dict with keys: rho, velocity, pressure, B.
            Optional: Te, Ti, psi (passed through as auxiliary).
        convert_b_si_to_hl : bool
            If True, divide B by sqrt(mu_0) for HL units.

        Returns
        -------
        mx.array
            Packed conserved state, shape (10, nr, nz), float32.
        """
        ...

    def to_state_dict(
        self,
        U: mx.array,
        convert_b_hl_to_si: bool = False,
    ) -> dict[str, np.ndarray]:
        """Unpack conserved (10, nr, nz) mx.array to DPF state dict.

        Performs conserved -> primitive conversion:
          velocity = momentum / rho
          p = (gamma-1) * (E - 0.5*rho*v^2 - 0.5*B^2)
          Te = Ti = p * ion_mass / (rho * k_B)

        Parameters
        ----------
        U : mx.array
            Packed conserved state, shape (10, nr, nz), float32.
        convert_b_hl_to_si : bool
            If True, multiply B by sqrt(mu_0) for SI units.

        Returns
        -------
        dict[str, np.ndarray]
            DPF state dict with float64 NumPy arrays.
        """
        ...

    def entropy_from_primitives(
        self,
        rho: mx.array,
        p: mx.array,
    ) -> mx.array:
        """Compute entropy tracer S = p / rho^gamma.

        Parameters
        ----------
        rho : mx.array
            Density, shape (nr, nz).
        p : mx.array
            Pressure, shape (nr, nz).

        Returns
        -------
        mx.array
            S * rho, shape (nr, nz).
        """
        ...

    @staticmethod
    def zero_copy_to_mlx(arr: np.ndarray) -> mx.array:
        """Transfer NumPy array to MLX with zero-copy on Apple Silicon.

        Parameters
        ----------
        arr : np.ndarray
            Must be float32, C-contiguous for zero-copy.

        Returns
        -------
        mx.array
        """
        ...

    @staticmethod
    def zero_copy_to_numpy(arr: mx.array) -> np.ndarray:
        """Transfer MLX array to NumPy with zero-copy on Apple Silicon.

        Parameters
        ----------
        arr : mx.array

        Returns
        -------
        np.ndarray
        """
        ...
```

### Key Design Decisions

1. **Float32 internal, float64 external**: All MLX computation is float32 (Metal GPU constraint). The `to_state_dict()` method returns float64 NumPy arrays for compatibility with the DPF engine contract.

2. **Zero-copy path**: `mx.array(np_arr)` is zero-copy when `np_arr` is float32 + C-contiguous. `np.array(mx_arr)` is zero-copy back. The `MLXState` enforces contiguity before transfer.

3. **Entropy tracer (S*rho)**: Variable index 5 stores S*rho where S = p/rho^gamma. This is the conservatively advected entropy used for dual-energy pressure recovery. Computed during `from_state_dict()`.

4. **Electron energy (e_electron)**: Variable index 9. Initially set from Te via `e_e = 0.5 * rho * k_B * Te / ion_mass` (simple thermal energy). The two-temperature model evolves this independently.

5. **Unit conversion**: B-field conversion between SI (explicit mu_0) and Heaviside-Lorentz (mu_0=1) is handled at the pack/unpack boundary, not inside kernels. Factor: `B_HL = B_SI / sqrt(mu_0)`.

---

## 5. First Metal Kernel: Ghost Cell Padding

### Why Ghost Pad First

1. **Simplest kernel** -- single input array, simple index arithmetic, no physics coupling
2. **Foundation for HLLD** -- HLLD requires padded arrays for reconstruction stencils
3. **Testable in isolation** -- no dependency on other kernels
4. **Prototype is validated** -- `docs/metal_v2_research/kernels/ghost_cell_kernel.py` passes all 9 tests

### Exact API

The kernel will be ported from the prototype with minimal modification:

```python
def ghost_cell_pad(
    state: mx.array,
    current: float,
    r_inner: float,
    dr: float,
    ng: int = 3,
) -> mx.array:
    """Pad (10, nr, nz) state with ghost cells and electrode BCs.

    Inner boundary (axis, r->0):
      - Reflecting: rho, p, E, S*rho, Bz, e_e copied from mirror cell
      - Sign flip: vr, vtheta, Br, Btheta negated (axisymmetry)

    Outer boundary (cathode):
      - Zero-gradient for all variables
      - Conducting wall: vr = 0, Br = 0
      - Btheta = mu0 * I / (2*pi*r) from circuit current

    Parameters
    ----------
    state : mx.array
        Conserved state, shape (10, nr, nz), float32.
    current : float
        Circuit current [A].
    r_inner : float
        Inner radial boundary position [m].
    dr : float
        Radial cell spacing [m].
    ng : int
        Number of ghost cells per side (default 3 for WENO5-Z).

    Returns
    -------
    mx.array
        Padded state, shape (10, nr + 2*ng, nz), float32.
    """
```

### NumPy Reference

Every kernel includes a NumPy reference implementation with identical API (substituting `np.ndarray` for `mx.array`). This reference is:
- Used as ground truth in tests
- Used as CPU fallback when MLX is unavailable
- Kept in the same file for co-maintenance

```python
def ghost_cell_pad_numpy(
    state: np.ndarray,
    current: float,
    r_inner: float,
    dr: float,
    ng: int = 3,
) -> np.ndarray:
    """NumPy reference implementation of ghost cell padding."""
```

### MSL Source Porting

The MSL source from `docs/metal_v2_research/kernels/ghost_cell_kernel.py` lines 49-137 is used verbatim. Changes:
- None. The prototype MSL is production-ready.

### Thread Group Configuration

From the prototype, optimized for M3 Pro (14 GPU cores, SIMD width 32):

```python
tg_r = min(32, nr_g)
tg_z = min(8, nz)
grid_r = ((nr_g + tg_r - 1) // tg_r) * tg_r
grid_z = ((nz + tg_z - 1) // tg_z) * tg_z
```

### Test Plan for Ghost Cell Kernel

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| `test_ghost_uniform_interior` | Uniform state: verify interior cells unchanged after padding | `max_err < 1e-7` |
| `test_ghost_axis_reflecting` | Check IMR, IBR, IBT, IMT sign-flipped in inner ghosts | All flipped values negative for positive interior |
| `test_ghost_cathode_wall` | Check IMR=0, IBR=0 in outer ghosts | Exact zero |
| `test_ghost_btheta_current` | Verify Btheta = mu0*I/(2*pi*r) in outer ghosts for I=100kA | `rel_err < 1e-4` |
| `test_ghost_zero_current` | Zero current: all Btheta in ghosts should be ~0 | `< 1e-10` |
| `test_ghost_vs_numpy_16x16` | Compare Metal vs NumPy at 16x16 grid | `max_err < 1e-5` |
| `test_ghost_vs_numpy_64x128` | Compare Metal vs NumPy at 64x128 grid | `max_err < 1e-5` |
| `test_ghost_vs_numpy_128x512` | Compare Metal vs NumPy at 128x512 grid | `max_err < 1e-5` |
| `test_ghost_large_current` | I = 2 MA (extreme case): no NaN, physical Btheta | No NaN, no Inf |
| `test_ghost_output_shape` | Verify output shape is (10, nr+2*ng, nz) for various ng | Shape correct |

---

## 6. Integration Points

### How the MLX Solver Will Plug Into engine.py

The MLX solver will be a new backend option (`backend="mlx"`) alongside the existing PyTorch Metal solver (`backend="metal"`). The integration follows this pattern:

```python
# In engine.py, backend resolution cascade:
# athenak > athena > mlx > metal > python

class DPFEngine:
    def _create_solver(self, config: SimConfig) -> PlasmaSolverBase:
        backend = config.fluid.backend
        if backend == "auto":
            backend = self._auto_select_backend()

        if backend == "mlx":
            from dpf.metal.mlx_solver import MLXMHDSolver  # Sprint 3
            return MLXMHDSolver(
                nr=config.geometry.nr,
                nz=config.geometry.nz,
                dr=config.geometry.dx,
                dz=config.geometry.dz,
                gamma=config.fluid.gamma,
                cfl=config.fluid.cfl,
                r_inner=config.geometry.r_inner,
            )
```

### Sprint 1 Integration Scope (Foundation Only)

Sprint 1 does NOT wire into `engine.py`. The integration points are:

1. **`device.py` update**: Add `has_mlx() -> bool` so the auto-selection cascade can detect MLX availability in Sprint 3.

2. **Module imports**: New files follow the existing `src/dpf/metal/` package structure. All MLX imports are guarded:
   ```python
   try:
       import mlx.core as mx
       HAS_MLX = True
   except ImportError:
       HAS_MLX = False
       mx = None
   ```

3. **State dict contract**: `MLXState.to_state_dict()` returns `dict[str, np.ndarray]` matching the `PlasmaSolverBase.step()` contract. Same keys as all other backends: `{rho, velocity, pressure, B, Te, Ti, psi}`.

4. **Existing PyTorch Metal solver**: Untouched. `backend="metal"` continues to use `MetalMHDSolver` (PyTorch MPS). The MLX solver is a parallel path, not a replacement (yet).

### File Placement Within `src/dpf/metal/`

```
src/dpf/metal/
  __init__.py           (modify: add MLX exports)
  device.py             (modify: add has_mlx())
  mlx_device.py         (NEW: MLX device layer)
  mlx_grid.py           (NEW: cylindrical grid)
  mlx_state.py          (NEW: state dict bridge)
  mlx_kernels.py        (NEW: 3 Metal kernels)
  mlx_surrogate.py      (EXISTING: WALRUS, untouched)
  metal_solver.py       (EXISTING: PyTorch, untouched)
  metal_riemann.py      (EXISTING: PyTorch, untouched)
  metal_stencil.py      (EXISTING: PyTorch, untouched)
  metal_transport.py    (EXISTING: PyTorch, untouched)
  ...
```

---

## 7. Testing Strategy

### Test Infrastructure

All tests use `pytest.importorskip("mlx.core")` at module level to skip gracefully on non-Apple-Silicon CI runners.

```python
# tests/test_mlx_kernels.py (top of file)
import pytest
mx = pytest.importorskip("mlx.core")
import numpy as np
from dpf.metal.mlx_kernels import (
    ghost_cell_pad, ghost_cell_pad_numpy,
    hlld_flux, hlld_flux_numpy,
    cylindrical_source, cylindrical_source_numpy,
)
```

### Test Categories

#### A. Unit Tests: `test_mlx_device.py` (~80 LOC, 6 tests)

| Test | What It Checks |
|------|----------------|
| `test_has_mlx` | `has_mlx()` returns True on Apple Silicon |
| `test_mlx_version` | `get_mlx_version()` returns string >= "0.30.0" |
| `test_default_dtype` | `get_mlx_dtype()` returns `mx.float32` |
| `test_float64_fallback` | Requesting float64 returns `mx.float32` with warning |
| `test_device_info` | `mlx_device_info()` returns dict with "metal_available" key |
| `test_import_guard` | Module importable even when MLX not installed (returns HAS_MLX=False) |

#### B. Unit Tests: `test_mlx_grid.py` (~100 LOC, 8 tests)

| Test | What It Checks |
|------|----------------|
| `test_grid_radii` | Cell-center radii: r[i] = r_inner + (i + 0.5) * dr |
| `test_grid_face_radii` | Face radii: r_face[i] = r_inner + i * dr |
| `test_radial_face_areas` | A_r = 2*pi*r_face * dz |
| `test_axial_face_areas` | A_z = pi * (r_face[i+1]^2 - r_face[i]^2) |
| `test_cell_volumes` | V = pi * (r_face[i+1]^2 - r_face[i]^2) * dz |
| `test_volume_sum` | Sum of volumes = pi * (r_outer^2 - r_inner^2) * L_z |
| `test_grid_shapes` | All arrays have correct shapes for (nr, nz) |
| `test_grid_dtype` | All arrays are mx.float32 |

#### C. Unit Tests: `test_mlx_state.py` (~120 LOC, 10 tests)

| Test | What It Checks |
|------|----------------|
| `test_pack_uniform` | Uniform state packs to correct conserved variables |
| `test_unpack_uniform` | Unpack of packed uniform = original (round-trip) |
| `test_round_trip_random` | Random state: pack -> unpack preserves rho, v, p, B to 1e-5 |
| `test_output_keys` | `to_state_dict()` returns all DPF_KEYS |
| `test_output_shapes` | All arrays in state dict have correct shapes |
| `test_output_dtype` | All arrays in state dict are float64 |
| `test_entropy_tracer` | S*rho computed correctly: rho * p / rho^gamma |
| `test_electron_energy` | e_e = 0.5 * rho * k_B * Te / m_ion |
| `test_zero_copy_float32` | Float32 C-contiguous -> mx.array is zero-copy (data_ptr check) |
| `test_b_unit_conversion` | SI -> HL -> SI round-trip preserves B to 1e-10 |

#### D. Kernel Tests: `test_mlx_kernels.py` (~350 LOC, 28 tests)

**Ghost Cell Kernel** (10 tests -- see Section 5 test plan)

**HLLD Kernel** (10 tests):

| Test | What It Checks |
|------|----------------|
| `test_hlld_uniform_zero_flux` | Identical L/R states produce identical Metal vs NumPy flux |
| `test_hlld_sod_no_nan` | Sod shock: no NaN, no Inf |
| `test_hlld_sod_vs_numpy` | Sod shock: Metal vs NumPy max_err < 1e-3 |
| `test_hlld_brio_wu_no_nan` | Brio-Wu (strong Bt discontinuity): no NaN |
| `test_hlld_entropy_advection` | Entropy tracer advects with contact, not diffuses |
| `test_hlld_electron_energy` | Electron energy advects with contact wave |
| `test_hlld_axial_sweep` | dim=2 (axial) produces correct index mapping |
| `test_hlld_vs_numpy_64x128` | Grid 64x128: Metal vs NumPy < 1e-4 |
| `test_hlld_vs_numpy_128x512` | Grid 128x512: Metal vs NumPy < 1e-4 |
| `test_hlld_lax_friedrichs_fallback` | Degenerate input triggers LF fallback without NaN |

**Cylindrical Source Kernel** (8 tests):

| Test | What It Checks |
|------|----------------|
| `test_cyl_uniform_p_over_r` | Uniform p: S_mr = p/r (except ir=0) |
| `test_cyl_lhopital` | ir=0: S_mr = dp/dr, S_mt = S_Bt = 0 |
| `test_cyl_centrifugal` | vtheta > 0: S_mr includes rho*vtheta^2/r |
| `test_cyl_magnetic_tension` | Btheta > 0: hoop stress term present |
| `test_cyl_zero_sources` | Zero B, zero vtheta: only p/r source |
| `test_cyl_vs_numpy_32x64` | Metal vs NumPy at 32x64: < 1e-4 |
| `test_cyl_vs_numpy_128x512` | Metal vs NumPy at 128x512: < 1e-4 |
| `test_cyl_energy_source_zero` | Energy source (index 4) always zero |

### Cross-Check Against Prototype

Every kernel is validated against two references:
1. The NumPy reference in the same `mlx_kernels.py` file
2. The prototype's reference in `docs/metal_v2_research/kernels/` (for traceability)

A test helper function runs both and asserts agreement:

```python
def assert_kernel_matches_prototype(kernel_fn, prototype_fn, *args, tol=1e-5):
    """Verify production kernel matches prototype output."""
    result = np.array(kernel_fn(*[mx.array(a) if isinstance(a, np.ndarray) else a for a in args]))
    reference = prototype_fn(*args)
    np.testing.assert_allclose(result, reference, atol=tol, rtol=tol)
```

---

## 8. Risk Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | `mx.fast.metal_kernel()` API change in MLX update | Low | Medium | Pin `mlx>=0.30.0,<1.0`. The `metal_kernel` API has been stable since v0.30.0 (7+ releases). Test on each MLX upgrade. |
| R2 | Ghost cell kernel produces wrong sign flips at axis | Medium | High | Test with non-uniform state where sign flip is observable. Compare every variable index independently against NumPy reference. The prototype already validates this. |
| R3 | HLLD kernel NaN on extreme MHD states | Medium | High | Lax-Friedrichs fallback built into MSL source (lines 353-365 of prototype). Test with Brio-Wu (strong Bt discontinuity) and degenerate zero-density input. |
| R4 | Zero-copy assumption fails on non-Apple-Silicon | Low | Low | `zero_copy_to_mlx()` checks `arr.flags.c_contiguous` and dtype. Falls back to explicit copy with warning. MLX on non-Apple-Silicon (CUDA backend) copies anyway. |
| R5 | Thread group sizing suboptimal for M3 Pro | Low | Medium | Use prototype defaults (32x8 = 256 threads). Profile with `mx.eval()` timing in kernel benchmarks. Optimization deferred to Sprint 4. |
| R6 | CI runners lack Apple Silicon / MLX | HIGH | Low | All MLX tests use `pytest.importorskip("mlx.core")`. Tests skip gracefully. NumPy reference tests always run. No CI gate on MLX-specific tests. |
| R7 | State dict round-trip loses precision (float64 -> float32 -> float64) | Medium | Medium | Document that MLX solver internally uses float32. Round-trip test with `atol=1e-5` (not machine epsilon). Entropy tracer S*rho is particularly sensitive -- test with values spanning 6 orders of magnitude. |
| R8 | Kernel build fails silently (MSL compilation error) | Medium | Medium | Call each kernel with `verbose=True` during first test run to log generated MSL. Wrap kernel build in try/except with informative error message pointing to Metal GPU requirements. |

---

## 9. Exit Criteria

Sprint 1 is complete when ALL of the following pass:

### Must-Have (Blocking)

- [ ] `mlx_device.py` importable and functional on Apple Silicon with MLX >= 0.30.0
- [ ] `mlx_grid.py` produces correct cylindrical geometry (verified against analytical formulas)
- [ ] `mlx_state.py` round-trip test passes: pack -> unpack -> pack agrees to < 1e-5
- [ ] Ghost cell kernel: Metal vs NumPy max error < 1e-5 at 3 grid sizes (16x32, 64x128, 128x512)
- [ ] HLLD kernel: Metal vs NumPy max error < 1e-3 at 3 grid sizes (float32 tolerance)
- [ ] HLLD kernel: No NaN on Sod, Brio-Wu, and degenerate inputs
- [ ] Cylindrical source kernel: Metal vs NumPy max error < 1e-4 at 3 grid sizes
- [ ] Cylindrical source kernel: L'Hopital at ir=0 matches dp/dr to < 0.1
- [ ] `has_mlx()` returns correct boolean in `device.py`
- [ ] All existing tests pass (no regressions): `pytest tests/ -x -q --ignore=tests/test_mlx_*`
- [ ] All new tests pass: `pytest tests/test_mlx_device.py tests/test_mlx_grid.py tests/test_mlx_state.py tests/test_mlx_kernels.py -v`
- [ ] `ruff check src/dpf/metal/mlx_*.py tests/test_mlx_*.py` clean
- [ ] Total new tests >= 40

### Should-Have (Non-blocking)

- [ ] Kernel benchmark shows < 1 ms per kernel call at 128x512
- [ ] Zero-copy transfer verified (no unnecessary data copies in pack/unpack)
- [ ] MLX modules gracefully importable on non-Apple-Silicon (HAS_MLX=False, no crash)

### Verification Command

```bash
# Full sprint 1 verification in one command
python3 -m pytest tests/ -x -q && \
python3 -m pytest tests/test_mlx_device.py tests/test_mlx_grid.py \
                   tests/test_mlx_state.py tests/test_mlx_kernels.py -v && \
ruff check src/dpf/metal/mlx_*.py tests/test_mlx_*.py
```

---

## 10. Module Specifications

### `mlx_device.py` -- Detailed Design

```python
"""MLX device detection, dtype helpers, and Metal stream management.

Provides:
  - has_mlx() -> bool
  - require_mlx() -> module  (raises ImportError with install instructions)
  - get_mlx_version() -> str
  - get_mlx_dtype(precision: str) -> mx.Dtype
  - mlx_device_info() -> dict[str, Any]
  - ensure_contiguous(arr: mx.array) -> mx.array
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment]


def has_mlx() -> bool:
    """Return True if MLX is installed and Metal GPU is available."""
    if not HAS_MLX:
        return False
    try:
        # Verify Metal is functional (not just importable)
        test = mx.array([1.0, 2.0, 3.0])
        mx.eval(test)
        return True
    except Exception:
        return False


def require_mlx():
    """Import and return mlx.core, raising ImportError if unavailable."""
    if not HAS_MLX:
        raise ImportError(
            "MLX is required for the MLX MHD solver. "
            "Install with: pip install 'mlx>=0.30.0'"
        )
    return mx


def get_mlx_version() -> str:
    """Return MLX version string, or 'not installed'."""
    if not HAS_MLX:
        return "not installed"
    return mx.__version__


def get_mlx_dtype(precision: str = "float32") -> Any:
    """Map precision string to MLX dtype.

    Note: Metal GPU does not support float64. Requesting float64
    logs a warning and returns float32.
    """
    if not HAS_MLX:
        return None
    if precision == "float64":
        logger.warning(
            "MLX on Metal GPU does not support float64. Using float32."
        )
    return mx.float32


def mlx_device_info() -> dict[str, Any]:
    """Return dict of MLX/Metal device capabilities."""
    if not HAS_MLX:
        return {"metal_available": False, "mlx_version": "not installed"}
    return {
        "metal_available": True,
        "mlx_version": get_mlx_version(),
        "default_dtype": "float32",
    }


def ensure_contiguous(arr) -> Any:
    """Ensure mx.array is row-contiguous for kernel dispatch."""
    if not HAS_MLX:
        return arr
    return mx.contiguous(arr) if hasattr(mx, "contiguous") else arr
```

### `mlx_grid.py` -- Detailed Design

```python
"""Cylindrical grid geometry for MLX MHD solver.

Pre-computes and caches radial/axial geometry arrays as mx.array:
  - Cell-center radii
  - Face radii (half-cell offsets)
  - Radial face areas: A_r(i,j) = 2*pi*r_face(i) * dz
  - Axial face areas: A_z(i,j) = pi * (r_face(i+1)^2 - r_face(i)^2)
  - Cell volumes: V(i,j) = pi * (r_face(i+1)^2 - r_face(i)^2) * dz

All arrays are float32 and cached on first access.
"""

from __future__ import annotations

import math

import numpy as np

from dpf.metal.mlx_device import require_mlx

mx = require_mlx()


class CylindricalGrid:
    """Pre-computed cylindrical grid geometry as MLX arrays.

    Parameters
    ----------
    nr : int
        Number of radial cells.
    nz : int
        Number of axial cells.
    dr : float
        Radial cell spacing [m].
    dz : float
        Axial cell spacing [m].
    r_inner : float
        Inner radial boundary [m] (default 0.0).
    """

    def __init__(
        self,
        nr: int,
        nz: int,
        dr: float,
        dz: float,
        r_inner: float = 0.0,
    ) -> None:
        self.nr = nr
        self.nz = nz
        self.dr = float(dr)
        self.dz = float(dz)
        self.r_inner = float(r_inner)

        # Cell-center radii: r[i] = r_inner + (i + 0.5) * dr
        r_np = np.arange(nr, dtype=np.float32) * dr + r_inner + 0.5 * dr
        self.r_cell: mx.array = mx.array(r_np)

        # Face radii: r_face[i] = r_inner + i * dr  (nr+1 values)
        r_face_np = np.arange(nr + 1, dtype=np.float32) * dr + r_inner
        self.r_face: mx.array = mx.array(r_face_np)

        # Radial face areas: A_r(i) = 2*pi*r_face(i) * dz
        self.A_r: mx.array = 2.0 * math.pi * self.r_face * dz

        # Axial face areas: A_z(i) = pi * (r_face[i+1]^2 - r_face[i]^2)
        rf2 = self.r_face * self.r_face
        self.A_z: mx.array = math.pi * (rf2[1:] - rf2[:-1])

        # Cell volumes: V(i) = A_z(i) * dz
        self.volume: mx.array = self.A_z * dz

        # Precompute 1/r for source terms (with floor at axis)
        self.inv_r: mx.array = 1.0 / mx.maximum(self.r_cell, mx.array(1e-30))

        # Force evaluation to cache on device
        mx.eval(
            self.r_cell, self.r_face, self.A_r, self.A_z,
            self.volume, self.inv_r,
        )

    @property
    def r_outer(self) -> float:
        """Outer radial boundary [m]."""
        return self.r_inner + self.nr * self.dr

    @property
    def L_z(self) -> float:
        """Axial domain length [m]."""
        return self.nz * self.dz
```

### `mlx_kernels.py` -- Structure

Single file containing all 3 Metal kernels + their NumPy references. Organized as:

```python
"""Custom Metal kernels for cylindrical MHD on Apple Silicon.

Contains:
  1. ghost_cell_pad / ghost_cell_pad_numpy
  2. hlld_flux / hlld_flux_numpy
  3. cylindrical_source / cylindrical_source_numpy

Each kernel is built lazily on first call via mx.fast.metal_kernel()
and cached at module level.

MSL source strings are ported from:
  docs/metal_v2_research/kernels/{ghost_cell,hlld,cylindrical_source}_kernel.py
"""

# --- Section 1: Constants and Variable Indices ---
# (shared between all kernels)

# --- Section 2: Ghost Cell Padding ---
# _GHOST_HEADER, _GHOST_SOURCE
# ghost_cell_pad(state, current, r_inner, dr, ng) -> mx.array
# ghost_cell_pad_numpy(state, current, r_inner, dr, ng) -> np.ndarray

# --- Section 3: HLLD Riemann Solver ---
# _HLLD_HEADER, _HLLD_SOURCE
# hlld_flux(UL, UR, gamma, dim) -> mx.array
# hlld_flux_numpy(UL, UR, gamma, dim) -> np.ndarray

# --- Section 4: Cylindrical Geometric Sources ---
# _CYL_HEADER, _CYL_SOURCE
# cylindrical_source(prim, r_cell, dr) -> mx.array
# cylindrical_source_numpy(prim, r_cell, dr) -> np.ndarray
```

The MSL sources are taken verbatim from the prototype kernels in `docs/metal_v2_research/kernels/`. The Python wrapper functions follow the same API as the prototype but import from `dpf.metal.mlx_device` instead of raw `mlx.core`.

---

## 11. Relationship to Subsequent Sprints

Sprint 1 outputs feed directly into Sprint 2:

| Sprint 1 Output | Sprint 2 Consumer |
|------------------|-------------------|
| `mlx_kernels.ghost_cell_pad` | `mlx_reconstruction.py` -- pads state before WENO5-Z stencil |
| `mlx_kernels.hlld_flux` | `mlx_riemann.py` -- called after reconstruction to produce interface fluxes |
| `mlx_kernels.cylindrical_source` | `mlx_sources.py` -- geometric source terms in RHS computation |
| `mlx_state.MLXState` | `mlx_primitives.py` -- cons/prim conversion using packed array |
| `mlx_grid.CylindricalGrid` | `mlx_riemann.py` -- face areas for flux divergence |

Sprint 2 (WENO5-Z + Riemann pipeline) cannot begin until all 3 kernels pass validation.
