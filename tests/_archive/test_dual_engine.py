"""Tests for the dual-engine architecture (Sprint F.2).

These tests validate:
1. Backend field in FluidConfig (python, athena, auto)
2. Engine backend selection and fallback logic
3. Python backend unchanged behavior (no regressions)
4. Athena++ backend engine integration
5. Auto backend selection

Note: Athena++ cannot be re-initialized in the same process (it uses
global state), so all tests that create an Athena++ engine share a single
module-scoped engine instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.config import FluidConfig, SimulationConfig
from dpf.core.bases import PlasmaSolverBase

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def python_config(default_circuit_params):
    """Config with explicit Python backend."""
    return SimulationConfig(
        grid_shape=[8, 8, 8],
        dx=1e-2,
        sim_time=1e-6,
        circuit=default_circuit_params,
        fluid={"backend": "python"},
    )


@pytest.fixture
def auto_config(default_circuit_params):
    """Config with auto backend selection."""
    return SimulationConfig(
        grid_shape=[8, 8, 8],
        dx=1e-2,
        sim_time=1e-6,
        circuit=default_circuit_params,
        fluid={"backend": "auto"},
    )


# Module-scoped Athena++ engine fixture.
# Athena++ uses global state that does not support re-initialization in the
# same process.  All Athena++-specific tests share this single engine.
@pytest.fixture(scope="module")
def athena_engine():
    """Create a single Athena++ engine for the entire test module."""
    from dpf.athena_wrapper import is_available

    if not is_available():
        pytest.skip("Athena++ not compiled")

    from dpf.engine import SimulationEngine

    config = SimulationConfig(
        grid_shape=[16, 1, 32],
        dx=1e-3,
        sim_time=1e-7,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "ESR": 0.0,
            "ESL": 0.0,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        geometry={"type": "cylindrical"},
        fluid={"backend": "athena"},
    )
    engine = SimulationEngine(config)
    return engine


# ============================================================
# Test: FluidConfig backend field
# ============================================================


class TestFluidConfigBackend:
    """Tests for the backend field in FluidConfig."""

    def test_default_backend_is_python(self):
        fc = FluidConfig()
        assert fc.backend == "python"

    def test_python_backend_valid(self):
        fc = FluidConfig(backend="python")
        assert fc.backend == "python"

    def test_athena_backend_valid(self):
        fc = FluidConfig(backend="athena")
        assert fc.backend == "athena"

    def test_auto_backend_valid(self):
        fc = FluidConfig(backend="auto")
        assert fc.backend == "auto"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be"):
            FluidConfig(backend="invalid")

    def test_backend_in_config_dict(self, python_config):
        d = python_config.model_dump()
        assert d["fluid"]["backend"] == "python"

    def test_backend_serialization_roundtrip(self, python_config):
        """Backend survives JSON serialization."""
        json_str = python_config.to_json()
        restored = SimulationConfig.model_validate_json(json_str)
        assert restored.fluid.backend == "python"


# ============================================================
# Test: Engine backend resolution
# ============================================================


class TestEngineBackendResolution:
    """Tests for SimulationEngine._resolve_backend()."""

    def test_resolve_python(self):
        from dpf.engine import SimulationEngine
        assert SimulationEngine._resolve_backend("python") == "python"

    def test_resolve_auto_returns_valid(self):
        from dpf.engine import SimulationEngine
        result = SimulationEngine._resolve_backend("auto")
        assert result in ("python", "athena")

    def test_resolve_athena_when_available(self):
        from dpf.athena_wrapper import is_available
        from dpf.engine import SimulationEngine
        if not is_available():
            pytest.skip("Athena++ not compiled")
        assert SimulationEngine._resolve_backend("athena") == "athena"

    def test_resolve_athena_raises_when_unavailable(self, monkeypatch):
        """If Athena++ unavailable, explicit 'athena' should raise."""
        # Temporarily make is_available return False
        import dpf.athena_wrapper as aw
        from dpf.engine import SimulationEngine
        original = aw._AVAILABLE
        monkeypatch.setattr(aw, "_AVAILABLE", False)
        try:
            with pytest.raises(RuntimeError, match="not compiled"):
                SimulationEngine._resolve_backend("athena")
        finally:
            monkeypatch.setattr(aw, "_AVAILABLE", original)

    def test_resolve_invalid_raises(self):
        from dpf.engine import SimulationEngine
        with pytest.raises(ValueError, match="Unknown backend"):
            SimulationEngine._resolve_backend("cuda")


# ============================================================
# Test: Python backend unchanged
# ============================================================


class TestPythonBackendUnchanged:
    """Verify Python backend works exactly as before."""

    def test_engine_creates_with_python_backend(self, python_config):
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(python_config)
        assert engine.backend == "python"

    def test_python_backend_single_step(self, python_config):
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(python_config)
        result = engine.step()
        assert result.time > 0
        assert result.step == 1
        assert not result.finished

    def test_python_backend_default_config(self, default_circuit_params):
        """Default config (no backend specified) uses Python."""
        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-2,
            sim_time=1e-6,
            circuit=default_circuit_params,
        )
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(config)
        assert engine.backend == "python"

    def test_python_backend_cylindrical(self, default_circuit_params):
        config = SimulationConfig(
            grid_shape=[8, 1, 8],
            dx=1e-2,
            sim_time=1e-7,
            circuit=default_circuit_params,
            geometry={"type": "cylindrical"},
            fluid={"backend": "python"},
        )
        from dpf.engine import SimulationEngine
        engine = SimulationEngine(config)
        assert engine.backend == "python"
        result = engine.step()
        assert result.step == 1


# ============================================================
# Test: Athena++ backend engine integration
#
# NOTE: Athena++ cannot be re-initialized in the same process.
# All tests in this class share a single module-scoped engine.
# ============================================================


class TestAthenaBackendEngine:
    """Integration tests for Athena++ backend in SimulationEngine.

    All tests use a shared module-scoped engine because Athena++
    does not support re-initialization (global state limitation).
    """

    def test_engine_creates_with_athena_backend(self, athena_engine):
        assert athena_engine.backend == "athena"
        assert isinstance(athena_engine.fluid, PlasmaSolverBase)

    def test_athena_backend_single_step(self, athena_engine):
        result = athena_engine.step()
        assert result.time > 0
        assert result.step >= 1
        assert np.isfinite(result.current)

    def test_athena_backend_multiple_steps(self, athena_engine):
        initial_step = athena_engine.step_count
        for _ in range(3):
            result = athena_engine.step()
        assert result.step == initial_step + 3
        assert result.time > 0

    def test_athena_backend_energy_conservation(self, athena_engine):
        """Energy should be approximately conserved (circuit only)."""
        for _ in range(5):
            result = athena_engine.step()
        # Energy conservation should be reasonable (no radiation/collision)
        assert 0.5 < result.energy_conservation < 2.0

    def test_athena_state_has_required_keys(self, athena_engine):
        # Ensure at least one step has been run
        if athena_engine.step_count == 0:
            athena_engine.step()
        required = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        assert required.issubset(set(athena_engine.state.keys()))

    def test_athena_step_result_has_all_fields(self, athena_engine):
        """Athena++ StepResult should have all expected fields."""
        result = athena_engine.step()
        assert hasattr(result, "time")
        assert hasattr(result, "step")
        assert hasattr(result, "dt")
        assert hasattr(result, "current")
        assert hasattr(result, "voltage")
        assert hasattr(result, "energy_conservation")
        assert hasattr(result, "max_Te")
        assert hasattr(result, "max_rho")
        assert hasattr(result, "finished")


# ============================================================
# Test: Auto backend selection
# ============================================================


class TestAutoBackend:
    """Tests for auto backend selection.

    Since Athena++ cannot be re-initialized, we test the resolution
    logic without creating additional engines.
    """

    def test_auto_resolves_to_valid_backend(self):
        """Auto should resolve to either python or athena."""
        from dpf.engine import SimulationEngine
        result = SimulationEngine._resolve_backend("auto")
        assert result in ("python", "athena")

    def test_auto_prefers_athena_when_available(self):
        """When Athena++ is available, auto should prefer it."""
        from dpf.athena_wrapper import is_available
        from dpf.engine import SimulationEngine
        result = SimulationEngine._resolve_backend("auto")
        if is_available():
            assert result == "athena"
        else:
            assert result == "python"


# ============================================================
# Test: Backend compatibility
# ============================================================


class TestBackendCompatibility:
    """Verify both backends produce compatible outputs."""

    def test_step_result_fields_match(self, default_circuit_params):
        """Python backend StepResult has all required fields."""
        from dpf.engine import SimulationEngine

        py_config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-2,
            sim_time=1e-6,
            circuit=default_circuit_params,
            fluid={"backend": "python"},
        )
        engine = SimulationEngine(py_config)
        result = engine.step()

        # Check all StepResult fields are populated
        assert hasattr(result, "time")
        assert hasattr(result, "step")
        assert hasattr(result, "dt")
        assert hasattr(result, "current")
        assert hasattr(result, "voltage")
        assert hasattr(result, "energy_conservation")
        assert hasattr(result, "max_Te")
        assert hasattr(result, "max_rho")
        assert hasattr(result, "finished")

    def test_athena_and_python_state_keys_match(self, athena_engine, default_circuit_params):
        """Both backends should produce state dicts with same keys."""
        from dpf.engine import SimulationEngine

        py_config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-2,
            sim_time=1e-6,
            circuit=default_circuit_params,
            fluid={"backend": "python"},
        )
        py_engine = SimulationEngine(py_config)
        py_engine.step()

        # Ensure Athena engine has been stepped
        if athena_engine.step_count == 0:
            athena_engine.step()

        # Both should have the same state keys
        py_keys = set(py_engine.state.keys())
        ath_keys = set(athena_engine.state.keys())
        assert py_keys == ath_keys
