"""Integration tests for the MLX Metal v2 backend wired into SimulationEngine.

All tests are skipped automatically when MLX is not installed or no Metal GPU
is available, so the suite remains green on non-Apple hardware.

Test inventory
--------------
1.  ``test_config_accepts_mlx_backend`` — FluidConfig validator accepts "mlx".
2.  ``test_engine_selects_mlx_solver`` — engine.fluid is MLXMHDSolver when
    backend="mlx" (requires Metal GPU; skipped otherwise).
3.  ``test_mlx_solver_is_available_flag`` — is_available() returns bool, no crash.
4.  ``test_mlx_solver_instantiation`` — direct construction, no engine needed.
5.  ``test_mlx_solver_step_no_crash`` — 3 steps, no NaN, keys present.
6.  ``test_mlx_solver_compute_dt`` — compute_dt returns positive finite float.
7.  ``test_engine_step_mlx_three_steps`` — run engine.step() 3 times via MLX.
8.  ``test_pf1000_preset_mlx_construction`` — PF-1000 preset + backend="mlx".
9.  ``test_mlx_backend_unavailable_raises`` — informative RuntimeError when MLX absent.
10. ``test_state_dict_keys`` — output has all required DPF state keys.
11. ``test_engine_tier_mlx_is_production`` — engine_tier == "production".
12. ``test_cell_volume_set_on_mlx_backend`` — _cell_volume is positive float.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Skip the entire module (at collection time) when MLX is not installed.
# Tests that additionally need a Metal GPU skip individually.
mlx = pytest.importorskip("mlx.core", reason="MLX not installed")  # noqa: E402, I001

from dpf.config import FluidConfig, SimulationConfig  # noqa: E402
from dpf.metal.mlx_device import HAS_MLX  # noqa: E402
from dpf.metal.mlx_solver import MLXMHDSolver  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METAL_GPU_AVAILABLE = HAS_MLX and (mlx.default_device().type == mlx.gpu)

requires_metal = pytest.mark.skipif(
    not _METAL_GPU_AVAILABLE,
    reason="Metal GPU not available (non-Apple or simulator)",
)

_CIRCUIT = {
    "C": 1e-6,
    "V0": 1e3,
    "L0": 1e-7,
    "R0": 0.01,
    "anode_radius": 0.005,
    "cathode_radius": 0.01,
}

_EXPECTED_KEYS = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}


def _mlx_config(grid=(8, 1, 8), **overrides) -> SimulationConfig:
    defaults: dict = {
        "grid_shape": grid,
        "dx": 1e-3,
        "sim_time": 1e-7,
        "dt_init": 1e-10,
        "circuit": _CIRCUIT,
        "fluid": {"backend": "mlx"},
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _make_minimal_state(nr: int = 8, nz: int = 8) -> dict[str, np.ndarray]:
    """Minimal uniform DPF state dict suitable for MLX solver."""
    shape = (nr, 1, nz)
    rho = np.full(shape, 1e-4, dtype=np.float64)
    velocity = np.zeros((3, nr, 1, nz), dtype=np.float64)
    pressure = np.full(shape, 1e2, dtype=np.float64)
    B = np.zeros((3, nr, 1, nz), dtype=np.float64)
    B[2] = 1e-3  # small Btheta seed
    Te = np.full(shape, 1e4, dtype=np.float64)
    Ti = np.full(shape, 1e4, dtype=np.float64)
    psi = np.zeros(shape, dtype=np.float64)
    return {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": psi,
    }


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


def test_config_accepts_mlx_backend():
    fc = FluidConfig(backend="mlx")
    assert fc.backend == "mlx"


# ---------------------------------------------------------------------------
# 2. Engine selects MLXMHDSolver
# ---------------------------------------------------------------------------


@requires_metal
def test_engine_selects_mlx_solver():
    from dpf.engine import SimulationEngine

    engine = SimulationEngine(_mlx_config())
    assert isinstance(engine.fluid, MLXMHDSolver), (
        f"Expected MLXMHDSolver, got {type(engine.fluid)}"
    )


# ---------------------------------------------------------------------------
# 3. is_available flag
# ---------------------------------------------------------------------------


def test_mlx_solver_is_available_flag():
    result = MLXMHDSolver.is_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 4. Direct instantiation (no engine)
# ---------------------------------------------------------------------------


@requires_metal
def test_mlx_solver_instantiation():
    solver = MLXMHDSolver(
        grid_shape=(8, 1, 8),
        dx=1e-3,
        dz=1e-3,
        gamma=5.0 / 3.0,
        cfl=0.3,
        riemann_solver="hlld",
        reconstruction="weno5z",
        time_integrator="ssp_rk3",
        coordinates="cylindrical",
        r_inner=0.005,
        convert_b_si_to_hl=True,
    )
    assert solver.nr == 8
    assert solver.nz == 8
    assert solver._riemann == "hlld"
    assert solver._method == "weno5z"


# ---------------------------------------------------------------------------
# 5. Step: no crash, expected keys, no NaN
# ---------------------------------------------------------------------------


@requires_metal
def test_mlx_solver_step_no_crash():
    solver = MLXMHDSolver(
        grid_shape=(8, 1, 8),
        dx=1e-3,
        dz=1e-3,
        convert_b_si_to_hl=True,
    )
    state = _make_minimal_state(nr=8, nz=8)
    dt = 1e-10

    for _ in range(3):
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

    assert set(state.keys()) == _EXPECTED_KEYS
    assert not np.any(np.isnan(state["rho"])), "NaN in rho after 3 MLX steps"
    assert not np.any(np.isnan(state["pressure"])), "NaN in pressure after 3 MLX steps"
    assert np.all(state["rho"] > 0), "Non-positive density after 3 MLX steps"


# ---------------------------------------------------------------------------
# 6. compute_dt
# ---------------------------------------------------------------------------


@requires_metal
def test_mlx_solver_compute_dt():
    solver = MLXMHDSolver(
        grid_shape=(8, 1, 8),
        dx=1e-3,
        dz=1e-3,
        convert_b_si_to_hl=True,
    )
    state = _make_minimal_state(nr=8, nz=8)
    dt = solver.compute_dt(state)
    assert isinstance(dt, float)
    assert math.isfinite(dt)
    assert dt > 0.0


# ---------------------------------------------------------------------------
# 7. Engine step — 3 steps
# ---------------------------------------------------------------------------


@requires_metal
def test_engine_step_mlx_three_steps():
    from dpf.engine import SimulationEngine

    engine = SimulationEngine(_mlx_config())
    for _ in range(3):
        result = engine.step()
    assert result.step == 3
    assert math.isfinite(result.time)
    assert result.time > 0.0


# ---------------------------------------------------------------------------
# 8. PF-1000 preset with backend="mlx"
# ---------------------------------------------------------------------------


@requires_metal
def test_pf1000_preset_mlx_construction():
    from dpf.presets import get_preset

    preset = get_preset("pf1000")
    preset.setdefault("fluid", {})
    preset["fluid"]["backend"] = "mlx"
    config = SimulationConfig(**preset)
    assert config.fluid.backend == "mlx"

    from dpf.engine import SimulationEngine

    engine = SimulationEngine(config)
    assert isinstance(engine.fluid, MLXMHDSolver)


# ---------------------------------------------------------------------------
# 9. Backend unavailable raises informative error
# ---------------------------------------------------------------------------


def test_mlx_backend_unavailable_raises(monkeypatch):
    """When MLX is absent, _resolve_backend raises RuntimeError with instructions."""
    import dpf.metal.mlx_solver as _mod

    monkeypatch.setattr(_mod.MLXMHDSolver, "is_available", staticmethod(lambda: False))

    from dpf.engine import SimulationEngine

    config = SimulationConfig(
        grid_shape=(4, 1, 4),
        dx=1e-3,
        sim_time=1e-7,
        dt_init=1e-10,
        circuit=_CIRCUIT,
        fluid={"backend": "mlx"},
    )
    with pytest.raises(RuntimeError, match="MLX Metal backend"):
        SimulationEngine(config)


# ---------------------------------------------------------------------------
# 10. State dict keys
# ---------------------------------------------------------------------------


@requires_metal
def test_state_dict_keys():
    solver = MLXMHDSolver(
        grid_shape=(4, 1, 4),
        dx=1e-3,
        dz=1e-3,
        convert_b_si_to_hl=True,
    )
    state = _make_minimal_state(nr=4, nz=4)
    out = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
    assert set(out.keys()) == _EXPECTED_KEYS, (
        f"Missing keys: {_EXPECTED_KEYS - set(out.keys())}"
    )


# ---------------------------------------------------------------------------
# 11. engine_tier is "production" for MLX backend
# ---------------------------------------------------------------------------


@requires_metal
def test_engine_tier_mlx_is_production():
    from dpf.engine import SimulationEngine

    engine = SimulationEngine(_mlx_config())
    assert engine.engine_tier == "production"


# ---------------------------------------------------------------------------
# 12. _cell_volume is set and positive
# ---------------------------------------------------------------------------


@requires_metal
def test_cell_volume_set_on_mlx_backend():
    from dpf.engine import SimulationEngine

    engine = SimulationEngine(_mlx_config())
    assert engine._cell_volume is not None
    assert engine._cell_volume > 0.0
