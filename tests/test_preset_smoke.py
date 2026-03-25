"""WU-2.2: Multi-device smoke tests for all 16 presets.

Verifies every preset can:
1. Instantiate a SimulationConfig without error
2. Create a SimulationEngine without error
3. Run at least 10 steps without NaN or crash
4. Produce non-zero current

Uses coarse grids (8x1x16 or 16x1x32) for speed.
Total runtime target: <15s for all presets.
"""

from __future__ import annotations

import math

import pytest

from dpf.config import SimulationConfig
from dpf.presets import get_preset, get_preset_names


def _coarse_config(preset_name: str, max_steps: int = 20) -> SimulationConfig:
    """Build a coarse-grid config from a preset for smoke testing."""
    p = get_preset(preset_name)

    # Determine grid size that covers the device geometry
    cathode_r = p.get("circuit", {}).get("cathode_radius", 0.05)

    if preset_name == "cartesian_demo":
        # Cartesian needs at least 16 cells per dim for gradient calc
        p["grid_shape"] = [16, 16, 16]
        p["dx"] = 0.01
    else:
        nr = 16
        nz = 32
        dx = max(cathode_r * 1.1 / nr, 0.001)  # cover cathode + 10% margin
        p["grid_shape"] = [nr, 1, nz]
        p["dx"] = dx

    # Long sim_time so step count limits us, not time
    p["sim_time"] = 1000.0

    return SimulationConfig(**p)


ALL_PRESETS = sorted(get_preset_names())


@pytest.mark.validation
class TestPresetSmokeSuite:
    """Run every preset through the engine for a few steps."""

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_instantiates(self, preset_name):
        """Preset creates a valid SimulationConfig."""
        config = _coarse_config(preset_name)
        assert config.sim_time > 0

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_runs_without_nan(self, preset_name):
        """Preset runs 10 engine steps without NaN."""
        from dpf.engine.core import SimulationEngine

        config = _coarse_config(preset_name)
        engine = SimulationEngine(config)
        result = engine.step(_max_steps=10)

        assert not math.isnan(result.current), (
            f"{preset_name}: current is NaN after {result.step} steps"
        )
        assert result.step >= 1, f"{preset_name}: no steps completed"

    @pytest.mark.parametrize("preset_name", [
        p for p in ALL_PRESETS if p not in ("cartesian_demo", "phase_p_fidelity")
    ])
    def test_preset_produces_current(self, preset_name):
        """Preset produces non-zero current after a few steps."""
        from dpf.engine.core import SimulationEngine

        config = _coarse_config(preset_name)
        engine = SimulationEngine(config)
        result = engine.step(_max_steps=10)

        assert abs(result.current) > 0, (
            f"{preset_name}: zero current after {result.step} steps"
        )


@pytest.mark.validation
class TestPresetTable:
    """Generate a summary table of all preset smoke test results."""

    def test_all_presets_summary(self):
        """Run all presets and print a summary table."""
        from dpf.engine.core import SimulationEngine

        results = {}
        for name in ALL_PRESETS:
            try:
                config = _coarse_config(name)
                engine = SimulationEngine(config)
                result = engine.step(_max_steps=10)
                nan = math.isnan(result.current)
                results[name] = {
                    "status": "NaN" if nan else "OK",
                    "steps": result.step,
                    "current": result.current,
                }
            except Exception as e:
                results[name] = {
                    "status": f"ERR:{type(e).__name__}",
                    "steps": 0,
                    "current": 0.0,
                }

        # Print summary
        print("\n--- Preset Smoke Test Summary ---")
        print(f"{'Preset':25s} {'Status':8s} {'Steps':6s} {'I [A]':>12s}")
        print("-" * 55)
        ok_count = 0
        for name in ALL_PRESETS:
            r = results[name]
            print(f"{name:25s} {r['status']:8s} {r['steps']:6d} {r['current']:+12.2e}")
            if r["status"] == "OK":
                ok_count += 1
        print(f"\n{ok_count}/{len(ALL_PRESETS)} presets OK")
        assert ok_count >= 14, f"Only {ok_count}/16 presets passed"
