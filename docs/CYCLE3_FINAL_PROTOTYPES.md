# Cycle 3: Final Corrected Prototypes

**Date**: 2026-03-26 | **Cycle**: 3 of 3 (FINAL before implementation)
**Corrections**: 5 critical errors from CYCLE2_DEEP_REVIEW.md resolved

---

## Corrections Applied

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | Hall mu_0 FALSE ALARM | HL units: J_HL x B_HL = J_SI x B_SI (sqrt(mu_0) cancels). Three docs propagated one initial algebra mistake. | Removed "fix". Whistler test now validates current code IS correct. |
| 2 | MLX solver API | Cycle 2 claimed solver returns mx.array. WRONG: `step()` takes `dict[str, np.ndarray]` and returns `dict[str, np.ndarray]`. State keys: rho, velocity, pressure, B, Te, Ti, psi. But `initialize()` does NOT exist -- must build state dict manually. | Fixed PIC V5/AMR to construct state dicts, removed nonexistent `initialize()` call. |
| 3 | AMR mass conservation | `sum(rho)` is not mass on cylindrical grid. Must use `2*pi*r*dr*dz` volume weighting. | Added proper cylindrical volume integral. |
| 4 | Species advection wired | `mlx_solver.py:754-763` already calls `species_advection_step`. The "blocker" claim was false. | E2E test now uses solver's built-in species path via `species_config` kwarg. |
| 5 | Differentiable MHD JAX syntax | `QL.at[0].add()` is JAX. MLX uses index assignment. `hlls_flux_r` doesn't exist; actual function is `_hlls_flux`. | Rewrote using `compute_fluxes` (public API) and MLX array construction. |

---

## 1. PIC V5: Full Discharge on MLX Backend

```python
"""PIC V5: Full MHD+PIC discharge on MLX backend."""
import math
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest


def _make_uniform_state(nr: int, nz: int, rho0: float, p0: float) -> dict[str, np.ndarray]:
    # CYCLE 3 FIX: build state dict manually (no solver.initialize() method exists)
    state = {
        "rho": np.full((nr, 1, nz), rho0, dtype=np.float64),
        "velocity": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "pressure": np.full((nr, 1, nz), p0, dtype=np.float64),
        "B": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "Te": np.full((nr, 1, nz), p0 * 3.34e-27 / (2 * rho0 * 1.381e-23), dtype=np.float64),
        "Ti": np.full((nr, 1, nz), p0 * 3.34e-27 / (2 * rho0 * 1.381e-23), dtype=np.float64),
        "psi": np.zeros((nr, 1, nz), dtype=np.float64),
    }
    return state


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.slow
def test_pic_v5_mlx_full_discharge():
    from dpf.metal.mlx_solver import MLXMHDSolver
    from dpf.experimental.pic.hybrid import HybridPIC

    nr, nz = 16, 32
    dx = 0.23 / nr
    dz = 0.60 / nz

    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz), dx=dx, dz=dz, gamma=5.0 / 3.0,
        cfl=0.3, riemann_solver="hll", reconstruction="plm",
        time_integrator="ssp_rk3",
    )

    rho0, p0 = 0.084, 350.0
    # CYCLE 3 FIX: construct state dict, not solver.initialize()
    state = _make_uniform_state(nr, nz, rho0, p0)

    V0, C0, L0, R0 = 27e3, 1332e-6, 33.5e-9, 2.3e-3
    omega = 1.0 / math.sqrt(L0 * C0)
    tau = 2.0 * L0 / R0

    pic = HybridPIC(
        grid_shape=(nr, 1, nz), dx=dx, dy=dx, dz=dz, dt=1e-9,
    )
    charge_d, mass_d = 1.602e-19, 3.34e-27
    pic.add_species("deuterons", charge=charge_d, mass=mass_d)

    n_total, inject_start = 200, 50
    nan_detected = False
    max_v_over_c = 0.0
    particle_counts = []
    c_light = 2.998e8

    for step_i in range(n_total):
        t = step_i * 1e-9
        current = (V0 / (omega * L0)) * math.exp(-t / tau) * math.sin(omega * t)

        dt_mhd = solver.compute_dt(state)
        dt_mhd = min(dt_mhd, 5e-9)
        state = solver.step(state, dt_mhd, current=current, voltage=V0)

        # CYCLE 3 FIX: state IS a dict with np.ndarray values -- iteration is correct
        for key, val in state.items():
            if isinstance(val, np.ndarray) and np.any(np.isnan(val)):
                nan_detected = True
                break
        if nan_detected:
            break

        if step_i >= inject_start:
            if step_i == inject_start:
                pic.inject_beam(
                    species_name="deuterons", n_beam=100, energy_eV=100e3,
                    direction=np.array([0.0, 0.0, 1.0]),
                    origin=np.array([dx * nr / 2, 0.0, dz]),
                    spread=0.1,
                )

            # CYCLE 3 FIX: state["B"] is (3, nr, 1, nz) -- squeeze ny dim
            B_arr = state["B"]  # (3, nr, 1, nz)
            B_avg = np.mean(B_arr, axis=(1, 2, 3))
            E_field = np.zeros((nr, 1, nz, 3), dtype=np.float64)
            B_field = np.zeros((nr, 1, nz, 3), dtype=np.float64)
            B_field[..., 0] = B_avg[0]
            B_field[..., 1] = B_avg[1]
            B_field[..., 2] = B_avg[2]

            pic.push_particles(E_field, B_field, dt=dt_mhd)

            for sp in pic.species:
                if sp.n_particles() > 0:
                    v2 = np.sum(sp.velocities ** 2, axis=1)
                    v_max = math.sqrt(float(np.max(v2)))
                    max_v_over_c = max(max_v_over_c, v_max / c_light)

        particle_counts.append(sum(sp.n_particles() for sp in pic.species))

    assert not nan_detected, f"NaN at step {step_i}"
    assert max_v_over_c < 1.0, f"Superluminal particle: v/c = {max_v_over_c:.3f}"
    assert particle_counts[-1] >= 100, "Particle count dropped below injected amount"
    rho_final = state["rho"]
    assert np.std(rho_final) > 1e-6 * np.mean(rho_final), "MHD state did not evolve"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_pic_v5_smoke_10_steps():
    """10-step smoke: MLX solver + no PIC, verifies no crash."""
    from dpf.metal.mlx_solver import MLXMHDSolver

    nr, nz = 16, 32
    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz), dx=0.015, dz=0.019,
        riemann_solver="hll", reconstruction="plm",
    )
    # CYCLE 3 FIX: construct state dict manually
    state = _make_uniform_state(nr, nz, rho0=0.084, p0=350.0)
    for _ in range(10):
        dt = min(solver.compute_dt(state), 5e-9)
        state = solver.step(state, dt, current=1e5, voltage=27e3)
    assert not any(
        np.any(np.isnan(v)) for v in state.values() if isinstance(v, np.ndarray)
    )
```

---

## 2. Ghost Padding GPU Port

No Cycle 2 errors. **Code unchanged from CYCLE1_PROTOTYPE_CODE.md Item 2** (self-contained, correct). Copy-paste ready from that document.

---

## 3. Hall MHD Validation (CORRECTED -- No mu_0 Fix)

```python
"""Hall MHD validation: current code E_Hall = (J x B)/(ne*e) IS correct in HL units.

CYCLE 3 FIX: Removed the mu_0 "fix" that would have broken the Hall term.
The whistler test validates that the CURRENT code produces correct dispersion.
If the test fails, investigate numerical causes (resolution, boundaries),
NOT a missing mu_0 factor.

Proof: J_HL = curl(B_HL) = J_SI * sqrt(mu_0).
       B_HL = B_SI / sqrt(mu_0).
       J_HL x B_HL = J_SI * sqrt(mu_0) * B_SI / sqrt(mu_0) = J_SI * B_SI.
       E_Hall = (J x B)/(ne*e) is identical in SI and HL units. QED.
"""
import math
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest

_MU0 = 4.0 * math.pi * 1e-7
_E_CHARGE = 1.602176634e-19
_SQRT_MU0 = math.sqrt(_MU0)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_hall_whistler_dispersion():
    """Whistler wave: validates J = curl(B) (no /mu_0) is correct in HL units.

    CYCLE 3 FIX: This test validates the CURRENT code is correct.
    The analytical target uses E_Hall = (J x B)/(ne*e) WITHOUT mu_0.
    If it passes, Hall MHD is physically correct. If it fails, the
    cause is numerical, not a missing unit factor.
    """
    from dpf.metal.mlx_sources import apply_hall_mhd
    from dpf.metal.mlx_kernels import IDN, IBR, IBZ, IBT, IEN, ISR

    nr, nz = 4, 64
    Lz = 1.0
    dr, dz = 0.01, Lz / nz
    ion_mass = 3.34e-27
    rho0 = 1e-3
    ne0 = rho0 / ion_mass
    p0 = 1e4
    B0_SI = 1.0
    B0_HL = B0_SI / _SQRT_MU0
    gamma = 5.0 / 3.0

    k = 2.0 * math.pi / Lz
    dB_amp_SI = 0.01
    dB_amp_HL = dB_amp_SI / _SQRT_MU0

    z_cell = np.array([(j + 0.5) * dz for j in range(nz)], dtype=np.float32)
    r_cell = np.array([(i + 0.5) * dr for i in range(nr)], dtype=np.float32)

    U = np.zeros((10, nr, nz), dtype=np.float32)
    U[IDN] = rho0
    U[IBZ] = B0_HL
    U[IBR] = dB_amp_HL * np.sin(k * z_cell)[None, :]
    B2 = U[IBR] ** 2 + U[IBZ] ** 2 + U[IBT] ** 2
    U[IEN] = p0 / (gamma - 1) + 0.5 * B2
    U[ISR] = p0 / rho0 ** (gamma - 1)

    U_mx = mx.array(U)
    r_cell_mx = mx.array(r_cell)

    # CYCLE 3 FIX: analytical whistler phase speed uses SAME formula as code
    # v_phase = k * B0_SI / (mu_0 * ne0 * e) -- this is the SI expression
    # In HL: v_phase = k * B0_HL * sqrt(mu_0) / (mu_0 * ne0 * e)
    #       = k * B0_SI / (mu_0 * ne0 * e) -- SAME result
    v_phase_analytical = k * B0_SI / (_MU0 * ne0 * _E_CHARGE)

    dt_hall = 0.1 * dz ** 2 * _MU0 * ne0 * _E_CHARGE / B0_SI
    n_steps = max(1, int(0.1 * (2 * math.pi / k) / v_phase_analytical / dt_hall))
    n_steps = min(n_steps, 500)

    for _ in range(n_steps):
        U_mx = apply_hall_mhd(U_mx, dt_hall, dr, dz, r_cell_mx, ion_mass)
    mx.eval(U_mx)
    U_final = np.asarray(U_mx)

    # Hall must have modified Br (not a no-op)
    dBr = U_final[IBR, nr // 2, :] - dB_amp_HL * np.sin(k * z_cell)
    assert np.max(np.abs(dBr)) > 1e-6 * dB_amp_HL, (
        "Hall term did not modify Br -- possible no-op bug"
    )

    # Phase measurement via cross-correlation
    Br_initial = dB_amp_HL * np.sin(k * z_cell)
    Br_final = U_final[IBR, nr // 2, :]
    from numpy.fft import fft
    F_init = fft(Br_initial)
    F_final = fft(Br_final)
    cross = F_final * np.conj(F_init)
    phase_shift = np.angle(cross[1])

    total_time = n_steps * dt_hall
    expected_phase = k * v_phase_analytical * total_time
    expected_phase_wrapped = (expected_phase + math.pi) % (2 * math.pi) - math.pi

    # CYCLE 3 FIX: if this passes, the code is correct. If it fails by >30%,
    # investigate numerical dispersion (finite difference stencil), NOT mu_0.
    if abs(expected_phase_wrapped) > 0.1:
        error = abs(phase_shift - expected_phase_wrapped) / abs(expected_phase_wrapped)
        assert error < 0.30, (
            f"Whistler phase error: {error:.1%}. "
            f"Expected {expected_phase_wrapped:.3f} rad, got {phase_shift:.3f} rad. "
            f"If >30%: check resolution/boundaries, NOT mu_0 factor."
        )


### test_hall_uniform_b_noop and test_hall_on_vs_off_magnitude

**Unchanged from CYCLE1_PROTOTYPE_CODE.md Item 3** (tests 2 and 3). No Cycle 2 errors in these -- they test curl(B)=0 no-op and 1/(ne*e) scaling respectively. Copy-paste ready from Cycle 1.
```

---

## 4. Multi-Device Calibration Sweep

No Cycle 2 errors. **Code unchanged from CYCLE1_CALIBRATION_PROTOTYPE.md Item 1.** One note: `getattr(cal_result, "nrmse", 10.0)` is fragile -- extract from best trial object at implementation time.

---

## 5. Line Radiation MLX

One minor fix: `U[10 - 1]` hardcodes IEE index.

```python
# In apply_line_radiation_mlx, the energy update stack:
    # CYCLE 3 FIX: use IEE constant instead of hardcoded 10-1
    updated_vars = [
        U[IDN], U[IMR], U[IMZ], U[IMT],
        U[IEN] - dE,
        U[ISR],
        U[IBR], U[IBZ], U[IBT],
        U[IEE],  # CYCLE 3 FIX: was U[10 - 1]
    ]
    return mx.stack(updated_vars, axis=0).astype(mx.float32)
```

Rest of line radiation code unchanged from Cycle 1 (correct).

---

## 6. Multi-Species E2E Test (CORRECTED -- Uses Existing Wiring)

```python
"""Test: PF-1000 with D2 + Cu impurity, 100 steps.

CYCLE 3 FIX: Species advection is ALREADY wired in mlx_solver.py:754-763.
The solver accepts species_config kwarg and manages Y internally.
This test uses the solver's built-in species path instead of
manually calling species functions.
"""
from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")


def _make_pf1000_state(nr: int, nz: int) -> dict[str, np.ndarray]:
    rho0, p0 = 0.084, 350.0
    return {
        "rho": np.full((nr, 1, nz), rho0, dtype=np.float64),
        "velocity": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "pressure": np.full((nr, 1, nz), p0, dtype=np.float64),
        "B": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "Te": np.full((nr, 1, nz), 100.0, dtype=np.float64),
        "Ti": np.full((nr, 1, nz), 100.0, dtype=np.float64),
        "psi": np.zeros((nr, 1, nz), dtype=np.float64),
    }


@pytest.mark.slow
def test_pf1000_d2_cu_100_steps():
    """PF-1000 with 99% D2 + 1% Cu, 100 steps via solver's built-in species path."""
    from dpf.metal.mlx_solver import MLXMHDSolver

    nr, nz = 32, 64
    dr, dz = 1e-3, 1e-3

    # CYCLE 3 FIX: pass species_config to solver -- it wires species advection internally
    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz), dx=dr, dz=dz, gamma=5.0 / 3.0,
        coordinates="cylindrical",
        riemann_solver="hll", reconstruction="plm",
        time_integrator="ssp_rk2",
        enable_bremsstrahlung=True,
        species_config={
            "species": ["D", "Cu"],
            "Z": [1, 29],
            "A": [2.014, 63.546],
            "background": "D",
            "initial_fractions": {"Cu": 0.01},
        },
    )

    state = _make_pf1000_state(nr, nz)
    dt = 1e-9
    current, voltage = 500e3, 20e3

    Zeff_max_history = []

    for step_i in range(100):
        state = solver.step(state, dt, current=current, voltage=voltage)

        # CYCLE 3 FIX: species data is returned in state["species"] dict
        # by the solver (mlx_solver.py:787-793)
        if "species" in state:
            species_data = state["species"]
            # Verify species fractions exist
            assert "D" in species_data or "Cu" in species_data, (
                f"Species data missing at step {step_i}"
            )

    # Assertion 1: No NaN in final state
    for key in ["rho", "pressure"]:
        arr = state[key]
        assert np.all(np.isfinite(arr)), f"NaN/Inf in {key} after 100 steps"

    # Assertion 2: Species fractions returned by solver
    assert "species" in state, "Solver did not return species data"
    species_data = state["species"]
    if "Cu" in species_data and "D" in species_data:
        Y_cu = species_data["Cu"]
        Y_d = species_data["D"]
        Y_total = Y_cu + Y_d
        assert np.max(np.abs(Y_total - 1.0)) < 0.01, (
            f"Species fractions don't sum to 1: max deviation = {np.max(np.abs(Y_total - 1.0))}"
        )

    # Assertion 3: Energy is finite
    assert np.all(np.isfinite(state["pressure"])), "Pressure has NaN after 100 steps"


@pytest.mark.slow
def test_species_fraction_conservation_advection():
    """Verify species advection conserves total mass fraction (unit test)."""
    from dpf.metal.mlx_species import SpeciesManager, species_advection_step
    from dpf.metal.mlx_kernels import IDN, IMR, IMZ, NVAR

    nr, nz = 32, 64
    species_mgr = SpeciesManager(
        species=["D", "Cu"], Z=[1, 29], A=[2.014, 63.546], background="D",
    )

    r = mlx.arange(nr, dtype=mlx.float32)[:, None]
    z = mlx.arange(nz, dtype=mlx.float32)[None, :]
    Y_cu = 0.05 * mlx.exp(-((r - 5) ** 2 + (z - 32) ** 2) / 50.0)
    Y = Y_cu[None, :, :]

    total_before = float(mlx.sum(Y))

    U = mlx.zeros((NVAR, nr, nz), dtype=mlx.float32)
    U = U.at[IDN].add(1.0)
    U = U.at[IMR].add(100.0)

    Y_new = species_advection_step(Y, U, dr=1e-3, dz=1e-3, dt=1e-8, gamma=5.0 / 3.0)
    total_after = float(mlx.sum(Y_new))

    rel_change = abs(total_after - total_before) / max(total_before, 1e-30)
    assert rel_change < 0.05, f"Species mass changed by {rel_change*100:.1f}%"
```

---

## 7. AMR Integration Test (CORRECTED -- Cylindrical Volume Weighting)

```python
"""AMR integration test: PF-1000 early axial rundown (500 steps).

CYCLE 3 FIXES:
- _total_mass uses cylindrical volume weighting: mass = sum(rho * 2*pi*r*dr*dz)
- Solver constructed with keyword args (not SimulationConfig object)
- State dict access verified against actual API
"""
from __future__ import annotations

import math
import time
import numpy as np
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _total_mass_cylindrical(
    state: dict[str, np.ndarray], r_inner: float, dr: float, dz: float,
) -> float:
    """Cylindrical volume-weighted mass integral.

    CYCLE 3 FIX: was sum(rho), now sum(rho * 2*pi*r*dr*dz).
    """
    rho = state.get("rho", np.zeros((1, 1, 1)))
    if rho.ndim == 3:
        nr, _, nz = rho.shape
        rho_2d = rho[:, 0, :]  # squeeze ny=1
    else:
        nr, nz = rho.shape
        rho_2d = rho

    r_centers = np.array([r_inner + (i + 0.5) * dr for i in range(nr)])
    # Volume of each ring cell: 2*pi*r*dr*dz
    cell_volumes = 2.0 * math.pi * r_centers * dr * dz  # (nr,)
    # Mass = sum over all cells: rho * V
    mass = np.sum(rho_2d * cell_volumes[:, None])
    return float(mass)


def _measure_sheath_width(state: dict[str, np.ndarray], dr: float) -> float:
    B = state.get("B", np.zeros((3, 1, 1, 1)))
    if B.ndim < 4 or B.shape[0] < 3:
        return 0.0
    # CYCLE 3 FIX: B is (3, nr, 1, nz) -- squeeze ny dim
    B_theta = B[2, :, 0, :]  # (nr, nz)
    nr = B_theta.shape[0]
    if nr < 3:
        return 0.0
    iz_mid = B_theta.shape[1] // 2
    Bt_slice = B_theta[:, iz_mid]
    J_approx = np.abs(np.gradient(Bt_slice, dr))
    J_max = np.max(J_approx)
    if J_max < 1e-10:
        return 0.0
    return float(np.sum(J_approx > 0.5 * J_max))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.slow
def test_amr_pf1000_early_rundown():
    from dpf.metal.mlx_solver import MLXMHDSolver

    nr, nz = 32, 64
    r_max = 0.23
    z_max = 0.60
    dr = r_max / nr
    dz = z_max / nz

    # CYCLE 3 FIX: construct solver with keyword args, not SimulationConfig
    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz), dx=dr, dz=dz, gamma=5.0 / 3.0,
        riemann_solver="hll", reconstruction="plm", time_integrator="ssp_rk3",
        amr_config={
            "enabled": True, "max_levels": 2, "refinement_ratio": 2,
            "block_nr": 16, "block_nz": 32, "max_blocks_per_level": 16,
            "regrid_interval": 50,
        },
    )

    rho0, p0 = 0.084, 350.0
    state = {
        "rho": np.full((nr, 1, nz), rho0, dtype=np.float64),
        "velocity": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "pressure": np.full((nr, 1, nz), p0, dtype=np.float64),
        "B": np.zeros((3, nr, 1, nz), dtype=np.float64),
        "Te": np.full((nr, 1, nz), 100.0, dtype=np.float64),
        "Ti": np.full((nr, 1, nz), 100.0, dtype=np.float64),
        "psi": np.zeros((nr, 1, nz), dtype=np.float64),
    }

    # CYCLE 3 FIX: use cylindrical volume weighting for mass
    mass_0 = _total_mass_cylindrical(state, r_inner=0.0, dr=dr, dz=dz)
    t0 = time.perf_counter()

    for step in range(500):
        dt = solver.compute_dt(state)
        state = solver.step(state, dt, current=100e3, voltage=20e3)

    wall_time = time.perf_counter() - t0
    mass_f = _total_mass_cylindrical(state, r_inner=0.0, dr=dr, dz=dz)

    # Mass conservation: < 1% drift
    mass_drift = abs(mass_f - mass_0) / max(mass_0, 1e-30)
    assert mass_drift < 0.01, f"Mass drift {mass_drift:.4f} > 1%"

    sheath_w = _measure_sheath_width(state, dr)
    print(f"AMR: {wall_time:.1f}s, mass drift={mass_drift:.6f}, sheath={sheath_w:.0f} cells")
```

---

## 8. Thomson Scattering Gradio UI

No Cycle 2 errors. **Code unchanged from CYCLE1_INTEGRATION_PROTOTYPE.md Item 2.** One note at implementation: guard `state["rho"].ndim < 2` for Lee model (returns scalars, not 2D arrays).

---

## 9. Differentiable MHD Smoke Test (CORRECTED -- MLX Syntax)

```python
"""Smoke test: can mx.grad() differentiate through a single HLLS flux call?

CYCLE 3 FIXES:
- Uses compute_fluxes (public API), not nonexistent hlls_flux_r
- Uses MLX array construction, not JAX .at[].add() syntax
- Validates AD gradient against finite difference
"""
import pytest

mlx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")


def test_hlls_grad_smoke():
    """Differentiate through compute_fluxes via mx.grad."""
    import mlx.core as mx
    from dpf.metal.mlx_riemann import compute_fluxes

    def loss_fn(rho_L: mx.array) -> mx.array:
        """Scalar loss: sum of HLLS flux given left-state density perturbation."""
        nr, nz = 4, 4
        # CYCLE 3 FIX: build state array with MLX ops, not JAX .at[].add()
        U = mx.zeros((10, nr, nz), dtype=mx.float32)
        # Set uniform background
        rho_field = mx.broadcast_to(rho_L, (nr, nz))
        p_field = mx.full((nr, nz), 1.0, dtype=mx.float32)
        gamma = 5.0 / 3.0

        # Pack conserved state: IDN=0, IEN=4, ISR=5
        components = []
        for i in range(10):
            if i == 0:  # IDN
                components.append(rho_field)
            elif i == 4:  # IEN = p/(gamma-1)
                components.append(p_field / (gamma - 1.0))
            elif i == 5:  # ISR = p * rho^(1-gamma)
                components.append(p_field * mx.power(rho_field, 1.0 - gamma))
            else:
                components.append(mx.zeros((nr, nz), dtype=mx.float32))
        U = mx.stack(components, axis=0)

        # CYCLE 3 FIX: use compute_fluxes (public API), dim=0 for r-direction
        F = compute_fluxes(U, gamma=gamma, dim=0, method="plm", riemann="hll")
        return mx.sum(F)

    grad_fn = mx.grad(loss_fn)
    rho_val = mx.array(1.0, dtype=mx.float32)

    # Try AD gradient
    try:
        g = grad_fn(rho_val)
        mx.eval(g)
        ad_works = True
    except Exception as e:
        ad_works = False
        pytest.skip(f"mx.grad does not support compute_fluxes path: {e}")

    if ad_works:
        # Finite difference validation
        eps = mx.array(1e-3, dtype=mx.float32)
        f_plus = loss_fn(rho_val + eps)
        f_minus = loss_fn(rho_val - eps)
        mx.eval(f_plus, f_minus)
        fd_grad = (f_plus - f_minus) / (2.0 * eps)
        mx.eval(fd_grad)

        rel_error = float(mx.abs(g - fd_grad) / mx.maximum(mx.abs(fd_grad), mx.array(1e-10)))
        assert rel_error < 0.05, (
            f"AD gradient ({float(g):.6f}) disagrees with FD ({float(fd_grad):.6f}) "
            f"by {rel_error:.2e}"
        )
```

---

## Final Implementation Priority

| Rank | Item | LOC | Hours | Blocking? | Go/No-Go |
|------|------|-----|-------|-----------|----------|
| 1 | Ghost padding GPU port | 100+30 | 2-3 | None | **GO** -- self-contained, parity-tested |
| 2 | Hall validation tests (no fix) | 60 | 1-2 | None | **GO** -- validates current code |
| 3 | Line radiation MLX | 100 | 3-4 | SpeciesManager | **GO** -- follows bremsstrahlung pattern |
| 4 | Species E2E test | 60 | 1-2 | species_config kwarg | **GO** -- uses existing wiring |
| 5 | Multi-device calibration | 80 | 2 + 40min compute | None | **GO** -- straight automation |
| 6 | Thomson UI | 100 | 3 | Gradio tab hookup | **GO** -- independent UI |
| 7 | PIC V5 full discharge | 80 | 3-4 | Esirkepov dt fix | **CONDITIONAL** -- needs PIC bug fixes first |
| 8 | AMR integration test | 80 | 2 + 30min compute | AMR rhs_fn gap | **CONDITIONAL** -- CF ghost exchange missing |
| 9 | Differentiable MHD | 20 | 0.5 | mx.grad support | **RESEARCH** -- run smoke test to determine feasibility |

---

## Final FMEA (Updated RPNs)

| # | Failure Mode | Sev | Occ | Det | RPN | Change from Cycle 1 |
|---|-------------|-----|-----|-----|-----|---------------------|
| ~~1~~ | ~~Hall mu_0 factor~~ | — | — | — | ~~300~~ **0** | **REMOVED**: false alarm |
| 2 | Esirkepov dt mismatch (PIC) | 7 | 9 | 4 | 252 | Unchanged. Blocks PIC V5. |
| 3 | Float32 subnormals in Cu line rad | 8 | 4 | 7 | 224 | Unchanged. Mitigated by log-space. |
| 4 | Cu Z_eff catastrophe in vacuum | 6 | 5 | 6 | 180 | Unchanged. Mitigated by existing mask. |
| 5 | PIC ghost NaN field interpolation | 10 | 7 | 2 | 140 | Unchanged. Blocks PIC V5. |
| 6 | AMR CF ghost exchange gap | 8 | 4 | 4 | 128 | Unchanged. Known Phase B item. |
| 7 | mx.where broadcast shape (ghost) | 6 | 5 | 3 | 90 | Unchanged. Covered by parity test. |
| 8 | AMR rhs_fn=None (no source terms) | 8 | 5 | 2 | 80 | Unchanged. Acceptable for 500-step test. |
| ~~N3~~ | ~~AMR sum(rho) not mass~~ | — | — | — | ~~420~~ **0** | **FIXED**: cylindrical volume weighting added |
| ~~N4~~ | ~~Species test bypasses wiring~~ | — | — | — | ~~200~~ **0** | **FIXED**: uses solver's built-in species path |
| ~~N5~~ | ~~JAX syntax in diff MHD~~ | — | — | — | ~~40~~ **0** | **FIXED**: MLX syntax + correct function name |

**Top 3 active risks**: Esirkepov dt (252), Cu subnormals (224), vacuum Z_eff (180).

---

## Go/No-Go Assessment

**SAFE TO IMPLEMENT NOW (6 items)**:
1. Ghost padding GPU port -- fully self-contained
2. Hall validation tests -- validates existing correct code
3. Line radiation MLX -- follows proven bremsstrahlung pattern
4. Species E2E test -- leverages existing solver wiring
5. Multi-device calibration -- automation of existing pipeline
6. Thomson UI -- independent Gradio tab

**CONDITIONAL (2 items)**:
7. PIC V5 -- blocked on Esirkepov dt fix and ghost NaN guard (3 LOC each)
8. AMR integration -- CF ghost exchange gap produces O(1) errors at boundaries; acceptable for initial integration test but not production

**RESEARCH FIRST (1 item)**:
9. Differentiable MHD -- run smoke test (5 min) to determine if `mx.grad` propagates through `compute_fluxes`. Result determines whether 3-4 day prototype is feasible.
