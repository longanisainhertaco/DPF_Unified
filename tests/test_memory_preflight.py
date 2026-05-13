from __future__ import annotations

from unittest.mock import patch

import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.engine.memory_preflight import estimate_run_memory_bytes, run_memory_preflight
from dpf.engine.runtime_telemetry import RuntimeMemoryTelemetry


def _config(*, grid_shape: list[int] | None = None, diagnostics: dict | None = None):
    return SimulationConfig(
        grid_shape=grid_shape or [4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        diagnostics={"hdf5_filename": ":memory:", **(diagnostics or {})},
    )


def test_memory_estimate_scales_with_grid_size() -> None:
    small = estimate_run_memory_bytes(_config(grid_shape=[4, 4, 4]), "python")
    large = estimate_run_memory_bytes(_config(grid_shape=[8, 8, 8]), "python")

    assert large == small * 8


def test_memory_preflight_blocks_over_limit() -> None:
    config = _config(grid_shape=[512, 512, 64])

    result = run_memory_preflight(config, "python", available_bytes=1_000_000)

    assert result.passed is False
    assert result.override is False
    assert "exceeds" in result.reason


def test_memory_preflight_allows_explicit_overcommit() -> None:
    config = _config(
        grid_shape=[512, 512, 64],
        diagnostics={"allow_memory_overcommit": True},
    )

    result = run_memory_preflight(config, "python", available_bytes=1_000_000)

    assert result.passed is True
    assert result.override is True


def test_engine_blocks_before_allocating_unsafe_grid() -> None:
    config = _config(grid_shape=[512, 512, 64])

    with (
        patch("dpf.engine.memory_preflight.system_available_memory_bytes", return_value=1_000_000),
        pytest.raises(MemoryError, match="Memory preflight blocked launch"),
    ):
        SimulationEngine(config)


def test_engine_summary_includes_memory_preflight() -> None:
    config = _config()

    summary = SimulationEngine(config).run(max_steps=1)

    assert "memory_preflight" in summary
    assert summary["memory_preflight"]["passed"] is True
    assert summary["memory_preflight"]["projected_bytes"] > 0


def test_runtime_memory_telemetry_tracks_peak_rss(monkeypatch) -> None:
    samples = iter([100, 125, 110])

    monkeypatch.setattr("dpf.engine.runtime_telemetry.process_rss_bytes", lambda: next(samples))

    telemetry = RuntimeMemoryTelemetry.start("python")
    telemetry.sample()
    telemetry.finish()

    assert telemetry.telemetry_supported is True
    assert telemetry.process_start_rss_bytes == 100
    assert telemetry.process_peak_rss_bytes == 125
    assert telemetry.process_end_rss_bytes == 110
    assert telemetry.sample_count == 3


def test_engine_summary_includes_runtime_memory_telemetry() -> None:
    config = _config(diagnostics={"memory_telemetry_interval_steps": 1})

    summary = SimulationEngine(config).run(max_steps=1)

    assert "runtime_memory_telemetry" in summary
    telemetry = summary["runtime_memory_telemetry"]
    assert telemetry["backend"] == "python"
    assert telemetry["sample_count"] >= 2
    assert telemetry["telemetry_supported"] is True
    assert telemetry["process_peak_rss_bytes"] is not None


def test_engine_runtime_memory_telemetry_can_be_disabled() -> None:
    config = _config(diagnostics={"runtime_memory_telemetry_enabled": False})

    summary = SimulationEngine(config).run(max_steps=1)

    assert "runtime_memory_telemetry" not in summary
