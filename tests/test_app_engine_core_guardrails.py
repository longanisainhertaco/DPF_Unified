from __future__ import annotations

import numpy as np
import pytest

import app_engine


class _RecordingDiagnostics:
    def record(self, state: dict, t: float) -> None:
        del state, t


class _RecordingEngine:
    last_max_steps: int | None = None

    def __init__(self, config) -> None:
        self.config = config
        self.time = 0.0
        self.diagnostics = _RecordingDiagnostics()
        self.snowplow = object()
        self.state = {
            "rho": np.ones((2, 1, 2)),
            "velocity": np.zeros((3, 2, 1, 2)),
            "pressure": np.ones((2, 1, 2)),
            "B": np.zeros((3, 2, 1, 2)),
            "Te": np.full((2, 1, 2), 300.0),
            "Ti": np.full((2, 1, 2), 300.0),
            "psi": np.zeros((2, 1, 2)),
        }

    def step(self, *args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        return None

    def run(self, max_steps: int | None = None) -> dict[str, float | int]:
        type(self).last_max_steps = max_steps
        self.time = 1.0e-9
        self.diagnostics.record(
            {
                "circuit": {
                    "current": 1.0e6,
                    "voltage": 1.0e3,
                    "energy_cap": 2.0e3,
                    "energy_ind": 3.0e3,
                    "energy_res": 4.0e3,
                },
                "snowplow": {
                    "z_sheath": 1.0e-3,
                    "r_shock": 2.0e-3,
                    "phase": "axial",
                },
                "coupler": {"Lp": 5.0e-9},
                "Te": self.state["Te"],
                "Ti": self.state["Ti"],
            },
            self.time,
        )
        return {
            "steps": max_steps or 3,
            "peak_current_A": 1.0e6,
            "peak_current_time_s": self.time,
            "energy_conservation": 1.0,
            "total_neutron_yield": 0.0,
        }


class _FailingEngine:
    def __init__(self, config) -> None:
        del config
        raise ValueError("engine setup failed")


def test_run_mhd_simulation_core_honors_n_steps(monkeypatch) -> None:
    import dpf.engine

    monkeypatch.setattr(dpf.engine, "SimulationEngine", _RecordingEngine)
    result = app_engine.run_mhd_simulation_core(
        "tutorial",
        sim_time_us=1.0,
        n_steps=7,
        grid_nx=2,
    )

    assert _RecordingEngine.last_max_steps == 7
    assert result["requested_max_steps"] == 7
    assert result["n_steps"] == 7
    assert result["backend"] == "python"


def test_run_mhd_simulation_core_rejects_non_positive_n_steps() -> None:
    with pytest.raises(ValueError, match="n_steps must be positive"):
        app_engine.run_mhd_simulation_core("tutorial", sim_time_us=1.0, n_steps=0)


def test_run_mhd_simulation_core_raises_engine_failure_by_default(monkeypatch) -> None:
    import dpf.engine

    monkeypatch.setattr(dpf.engine, "SimulationEngine", _FailingEngine)

    with pytest.raises(RuntimeError, match="Lee fallback is disabled") as exc_info:
        app_engine.run_mhd_simulation_core("tutorial", sim_time_us=1.0, grid_nx=2)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_run_mhd_simulation_core_allows_explicit_lee_fallback(monkeypatch) -> None:
    import dpf.engine

    monkeypatch.setattr(dpf.engine, "SimulationEngine", _FailingEngine)
    monkeypatch.setattr(
        app_engine,
        "run_simulation_core",
        lambda *args, **kwargs: {"backend": "lee", "n_steps": 11},
    )

    result = app_engine.run_mhd_simulation_core(
        "tutorial",
        sim_time_us=1.0,
        grid_nx=2,
        allow_engine_fallback=True,
    )

    assert result["backend"] == "lee (fallback from engine:python)"
    assert result["engine_status"] == "failed"
    assert result["engine_fallback"] == "lee"
    assert result["engine_fallback_allowed"] is True
    assert result["engine_error_type"] == "ValueError"
    assert "engine setup failed" in result["engine_error"]
