"""Engine timestep safety tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dpf.core.bases import CouplingState
from dpf.engine.core import SimulationEngine


class _FixedDtFluid:
    def __init__(self, dt: float) -> None:
        self.dt = dt

    def _compute_dt(self, state: object) -> float:
        return self.dt


def _engine_with_fluid_dt(dt: float, backend: str = "mlx") -> SimulationEngine:
    engine = SimulationEngine.__new__(SimulationEngine)
    engine.fluid = _FixedDtFluid(dt)
    engine.state = object()
    engine.circuit = SimpleNamespace(L_ext=1e-9, R_total=1.0, C=1e-6)
    engine._coupling = CouplingState(Lp=0.0)
    engine.config = SimpleNamespace(
        dt_init=None,
        sim_time=1e-6,
        fluid=SimpleNamespace(backend=backend),
    )
    engine.step_count = 1
    return engine


@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_engine_dt_does_not_floor_gpu_cfl(backend: str) -> None:
    engine = _engine_with_fluid_dt(5.0e-13, backend=backend)

    assert engine._compute_dt() == pytest.approx(5.0e-13)
