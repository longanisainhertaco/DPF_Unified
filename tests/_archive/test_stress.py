"""Stress tests for DPF simulation robustness.

These tests run more steps on larger grids to verify:
1. No NaN/Inf creep into state arrays
2. Energy conservation holds
3. All physics modules work together
4. Neutron yield and interferometry are wired in
5. Cylindrical geometry works end-to-end

Mark with @pytest.mark.slow for optional skipping.
"""

from __future__ import annotations

import numpy as np

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine


def _make_config(**overrides):
    """Build a SimulationConfig with optional overrides."""
    defaults = {
        "grid_shape": [8, 8, 8],
        "dx": 1e-3,
        "sim_time": 1e-7,
        "dt_init": 1e-10,
        "circuit": {
            "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
            "anode_radius": 0.005, "cathode_radius": 0.01,
        },
    }
    defaults.update(overrides)
    return SimulationConfig(**defaults)


# ====================================================
# Neutron Yield Integration Tests
# ====================================================


class TestNeutronYieldIntegration:
    """Tests that neutron yield is computed inside the engine."""

    def test_step_result_has_neutron_fields(self):
        """StepResult includes neutron_rate and total_neutron_yield."""
        from dpf.core.bases import StepResult

        r = StepResult()
        assert hasattr(r, "neutron_rate")
        assert hasattr(r, "total_neutron_yield")
        assert r.neutron_rate == 0.0
        assert r.total_neutron_yield == 0.0

    def test_engine_tracks_neutron_yield(self):
        """Engine accumulates total_neutron_yield over steps."""
        engine = SimulationEngine(_make_config())
        engine.run(max_steps=5)
        # At cold gas temperatures (300K), yield should be zero
        assert engine.total_neutron_yield >= 0.0

    def test_neutron_yield_in_summary(self):
        """run() summary includes total_neutron_yield."""
        engine = SimulationEngine(_make_config())
        summary = engine.run(max_steps=5)
        assert "total_neutron_yield" in summary

    def test_step_result_neutron_rate_nonnegative(self):
        """neutron_rate is always non-negative."""
        engine = SimulationEngine(_make_config())
        for _ in range(5):
            result = engine.step()
            assert result.neutron_rate >= 0.0
            assert result.total_neutron_yield >= 0.0


# ====================================================
# Interferometry Integration Tests
# ====================================================


class TestInterferometryIntegration:
    """Tests that interferometry is computed for cylindrical geometry."""

    def test_cylindrical_has_fringe_shifts(self):
        """Cylindrical engine computes fringe shifts."""
        config = _make_config(
            grid_shape=[8, 1, 8],
            geometry={"type": "cylindrical"},
        )
        engine = SimulationEngine(config)
        engine.step()
        assert engine._last_fringe_shifts is not None
        assert len(engine._last_fringe_shifts) == 8  # nr

    def test_cartesian_no_fringe_shifts(self):
        """Cartesian engine does NOT compute fringe shifts."""
        engine = SimulationEngine(_make_config())
        engine.step()
        assert engine._last_fringe_shifts is None

    def test_fringe_shifts_finite(self):
        """Fringe shifts are finite numbers."""
        config = _make_config(
            grid_shape=[8, 1, 8],
            geometry={"type": "cylindrical"},
        )
        engine = SimulationEngine(config)
        for _ in range(3):
            engine.step()
        assert np.all(np.isfinite(engine._last_fringe_shifts))


# ====================================================
# Stress: Cartesian with all physics
# ====================================================


class TestCartesianStress:
    """Cartesian stress tests with all physics enabled."""

    def test_8_cubed_50_steps(self):
        """8^3 grid, up to 50 steps — no NaN, energy conserved."""
        config = _make_config(
            grid_shape=[8, 8, 8],
            sim_time=1e-5,
            radiation={"bremsstrahlung_enabled": True},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=50)

        assert summary["steps"] >= 10  # Should complete many steps
        assert 0.95 <= summary["energy_conservation"] <= 1.05
        # Verify no NaN in state
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf found in {key}"

    def test_all_physics_combined(self):
        """All optional physics enabled simultaneously."""
        config = _make_config(
            grid_shape=[8, 8, 8],
            sim_time=1e-5,
            radiation={"bremsstrahlung_enabled": True, "fld_enabled": True},
            sheath={"enabled": True, "boundary": "z_high"},
            collision={"dynamic_coulomb_log": True},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=20)

        assert summary["steps"] >= 10
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf in {key} with all physics"


# ====================================================
# Stress: Cylindrical geometry
# ====================================================


class TestCylindricalStress:
    """Cylindrical stress tests with radiation and sheath."""

    def test_8x16_50_steps(self):
        """8x1x16 cylindrical, up to 50 steps — no NaN, energy conserved."""
        config = _make_config(
            grid_shape=[8, 1, 16],
            sim_time=1e-5,
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": True},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=50)

        assert summary["steps"] >= 10
        assert 0.95 <= summary["energy_conservation"] <= 1.05
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf in {key}"

    def test_cylindrical_with_sheath(self):
        """Cylindrical with sheath BCs enabled."""
        config = _make_config(
            grid_shape=[8, 1, 8],
            sim_time=1e-5,
            geometry={"type": "cylindrical"},
            radiation={"bremsstrahlung_enabled": True},
            sheath={"enabled": True, "boundary": "z_high"},
        )
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=20)

        assert summary["steps"] >= 10
        for key, arr in engine.state.items():
            if isinstance(arr, np.ndarray):
                assert np.all(np.isfinite(arr)), f"NaN/Inf in {key}"

    def test_cylindrical_neutron_yield_tracked(self):
        """Cylindrical engine tracks neutron yield."""
        config = _make_config(
            grid_shape=[8, 1, 8],
            sim_time=1e-5,
            geometry={"type": "cylindrical"},
        )
        engine = SimulationEngine(config)
        engine.run(max_steps=10)
        assert engine.total_neutron_yield >= 0.0


# ====================================================
# Presets: Smoke tests
# ====================================================


class TestPresetSmoke:
    """Smoke tests: each preset creates an engine and runs a few steps."""

    def test_tutorial_preset_runs(self):
        """Tutorial preset runs 5 steps without error."""
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("tutorial"))
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=5)
        assert summary["steps"] == 5

    def test_pf1000_preset_runs(self):
        """PF-1000 preset runs 3 steps without error."""
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("pf1000"))
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=3)
        assert summary["steps"] == 3

    def test_nx2_preset_runs(self):
        """NX2 preset runs 3 steps without error."""
        from dpf.presets import get_preset

        config = SimulationConfig(**get_preset("nx2"))
        engine = SimulationEngine(config)
        summary = engine.run(max_steps=3)
        assert summary["steps"] == 3

    def test_all_presets_no_nan(self):
        """All presets produce finite state after 3 steps."""
        from dpf.presets import get_preset, get_preset_names

        for name in get_preset_names():
            config = SimulationConfig(**get_preset(name))
            engine = SimulationEngine(config)
            engine.run(max_steps=3)
            for key, arr in engine.state.items():
                if isinstance(arr, np.ndarray):
                    assert np.all(np.isfinite(arr)), (
                        f"NaN/Inf in {key} for preset '{name}'"
                    )
