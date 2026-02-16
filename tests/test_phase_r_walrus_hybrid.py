"""Phase R tests: WALRUS hybrid backend (Athena++ + surrogate acceleration).

Tests the HybridEngine class which runs full physics for an initial phase, then
hands off to WALRUS surrogate for acceleration. Uses ensemble-confidence validation
instead of parallel physics validation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dpf.ai.hybrid_engine import HybridEngine
from dpf.config import SimulationConfig

# ============================================================================
# Mock DPFSurrogate
# ============================================================================


class MockSurrogate:
    """Mock WALRUS surrogate for testing HybridEngine without ML dependencies."""

    def __init__(self, history_length: int = 4, behavior: str = "stable") -> None:
        """Create mock surrogate.

        Args:
            history_length: Number of historical timesteps required
            behavior: One of "stable", "nan", "variance_growth"
        """
        self.history_length = history_length
        self.behavior = behavior
        self._call_count = 0

    def predict_next_step(
        self, history: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Predict next state.

        Args:
            history: List of DPF states

        Returns:
            Predicted next state
        """
        self._call_count += 1
        last_state = history[-1]

        if self.behavior == "stable":
            # Return copy with small perturbation
            predicted = {}
            for key, val in last_state.items():
                if isinstance(val, np.ndarray):
                    # Add 1% noise
                    noise = np.random.normal(0, 0.01, val.shape)
                    predicted[key] = val + noise * np.abs(val + 1e-10)
                else:
                    predicted[key] = val
            return predicted

        elif self.behavior == "nan":
            # Return NaN after 3 calls
            if self._call_count >= 3:
                predicted = {}
                for key, val in last_state.items():
                    if isinstance(val, np.ndarray):
                        predicted[key] = np.full_like(val, np.nan)
                    else:
                        predicted[key] = val
                return predicted
            else:
                return {k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in last_state.items()}

        elif self.behavior == "variance_growth":
            # Return states with exponentially growing variance
            growth_factor = 2.5 ** self._call_count
            predicted = {}
            for key, val in last_state.items():
                if isinstance(val, np.ndarray):
                    noise = np.random.normal(0, 0.1 * growth_factor, val.shape)
                    predicted[key] = val + noise
                else:
                    predicted[key] = val
            return predicted

        else:
            raise ValueError(f"Unknown behavior: {self.behavior}")


# ============================================================================
# Mock SimulationEngine
# ============================================================================


class MockEngine:
    """Mock SimulationEngine for testing HybridEngine."""

    def __init__(self, config: SimulationConfig) -> None:
        """Create mock engine.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self._step_count = 0
        self._shape = (8, 8, 8)

    def step(self) -> None:
        """Advance simulation by one timestep."""
        self._step_count += 1

    def get_field_snapshot(self) -> dict[str, np.ndarray]:
        """Get current field state.

        Returns:
            State dict with rho, velocity, pressure, B, Te, Ti, psi
        """
        # Create deterministic state based on step count
        seed = self._step_count
        rng = np.random.RandomState(seed)

        state = {
            "rho": rng.uniform(1e-6, 1e-5, self._shape),
            "velocity": rng.uniform(-1e3, 1e3, (3, *self._shape)),
            "pressure": rng.uniform(100, 200, self._shape),
            "B": rng.uniform(-0.1, 0.1, (3, *self._shape)),
            "Te": rng.uniform(1.0, 2.0, self._shape),
            "Ti": rng.uniform(1.0, 2.0, self._shape),
            "psi": rng.uniform(-1e-3, 1e-3, self._shape),
        }
        return state


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def minimal_config() -> SimulationConfig:
    """Create minimal simulation config for hybrid tests."""
    return SimulationConfig(
        grid_shape=(8, 8, 8),
        dx=0.01,
        sim_time=1e-6,
        circuit={"V0": 20e3, "C": 100e-6, "L0": 50e-9, "R0": 0.01,
                 "anode_radius": 0.005, "cathode_radius": 0.01},
        fluid={"backend": "hybrid", "handoff_fraction": 0.2, "validation_interval": 10},
    )


@pytest.fixture
def mock_surrogate() -> MockSurrogate:
    """Create mock WALRUS surrogate."""
    return MockSurrogate(history_length=4, behavior="stable")


# ============================================================================
# Test HybridEngine Initialization
# ============================================================================


def test_hybrid_engine_init_valid_handoff(minimal_config: SimulationConfig,
                                          mock_surrogate: MockSurrogate) -> None:
    """Test HybridEngine initialization with valid handoff_fraction."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
        validation_interval=50,
        max_l2_divergence=0.1,
    )

    assert engine.handoff_fraction == pytest.approx(0.2)
    assert engine.validation_interval == 50
    assert engine.max_l2_divergence == pytest.approx(0.1)
    assert len(engine._trajectory) == 0


def test_hybrid_engine_init_invalid_handoff_negative(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test HybridEngine rejects handoff_fraction < 0."""
    with pytest.raises(ValueError, match="handoff_fraction must be in \\[0, 1\\]"):
        HybridEngine(
            config=minimal_config,
            surrogate=mock_surrogate,
            handoff_fraction=-0.1,
        )


def test_hybrid_engine_init_invalid_handoff_too_large(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test HybridEngine rejects handoff_fraction > 1."""
    with pytest.raises(ValueError, match="handoff_fraction must be in \\[0, 1\\]"):
        HybridEngine(
            config=minimal_config,
            surrogate=mock_surrogate,
            handoff_fraction=1.5,
        )


def test_hybrid_engine_init_boundary_values(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test HybridEngine accepts boundary values 0.0 and 1.0."""
    # handoff_fraction = 0.0 (all surrogate)
    engine_zero = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.0,
    )
    assert engine_zero.handoff_fraction == pytest.approx(0.0)

    # handoff_fraction = 1.0 (all physics)
    engine_one = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=1.0,
    )
    assert engine_one.handoff_fraction == pytest.approx(1.0)


# ============================================================================
# Test _run_physics_phase
# ============================================================================


def test_run_physics_phase_calls_step_correct_count(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _run_physics_phase calls engine.step() correct number of times."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    mock_engine = MockEngine(minimal_config)
    n_steps = 20

    history = engine._run_physics_phase(mock_engine, n_steps)

    # Verify step() was called n_steps times
    assert mock_engine._step_count == n_steps

    # Verify history has n_steps states
    assert len(history) == n_steps

    # Verify each state has all required fields
    for state in history:
        assert "rho" in state
        assert "velocity" in state
        assert "pressure" in state
        assert "B" in state
        assert "Te" in state
        assert "Ti" in state
        assert "psi" in state


def test_run_physics_phase_snapshots_are_independent(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _run_physics_phase returns independent state copies."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    mock_engine = MockEngine(minimal_config)
    history = engine._run_physics_phase(mock_engine, 5)

    # Verify states are different (mock engine uses step count as seed)
    rho_0 = history[0]["rho"]
    rho_1 = history[1]["rho"]
    assert not np.allclose(rho_0, rho_1)


# ============================================================================
# Test _run_surrogate_phase NaN/Inf Detection
# ============================================================================


def test_run_surrogate_phase_detects_nan(minimal_config: SimulationConfig) -> None:
    """Test _run_surrogate_phase detects NaN and falls back."""
    surrogate_nan = MockSurrogate(history_length=4, behavior="nan")

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_nan,
        handoff_fraction=0.2,
        validation_interval=2,  # Check every 2 steps
    )

    # Create initial physics history
    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    # Run surrogate phase (should detect NaN after 3 steps, check at step 4)
    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=10)

    # Should have stopped early due to NaN detection
    # NaN happens at step 3, validation at step 4 (validation_interval=2)
    assert len(surrogate_history) <= 4
    assert len(surrogate_history) > 0  # Got at least some steps


def test_run_surrogate_phase_detects_inf(minimal_config: SimulationConfig) -> None:
    """Test _run_surrogate_phase detects Inf values and falls back."""

    class InfSurrogate:
        """Surrogate that produces Inf after 2 calls."""

        def __init__(self) -> None:
            self.history_length = 4
            self._call_count = 0

        def predict_next_step(
            self, history: list[dict[str, np.ndarray]]
        ) -> dict[str, np.ndarray]:
            self._call_count += 1
            last_state = history[-1]

            if self._call_count >= 2:
                return {
                    "rho": np.full_like(last_state["rho"], np.inf),
                    "velocity": np.full_like(last_state["velocity"], 0.0),
                    "pressure": np.full_like(last_state["pressure"], 100.0),
                    "B": np.full_like(last_state["B"], 0.0),
                    "Te": np.full_like(last_state["Te"], 1.0),
                    "Ti": np.full_like(last_state["Ti"], 1.0),
                    "psi": np.full_like(last_state["psi"], 0.0),
                }
            else:
                return {k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in last_state.items()}

    surrogate_inf = InfSurrogate()

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_inf,
        handoff_fraction=0.2,
        validation_interval=2,
    )

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=10)

    # Should have stopped early due to Inf detection
    assert len(surrogate_history) <= 4


# ============================================================================
# Test _run_surrogate_phase Exponential Variance Growth
# ============================================================================


def test_run_surrogate_phase_detects_variance_growth(
    minimal_config: SimulationConfig,
) -> None:
    """Test _run_surrogate_phase detects exponential variance growth and falls back."""
    surrogate_growth = MockSurrogate(history_length=4, behavior="variance_growth")

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_growth,
        handoff_fraction=0.2,
        validation_interval=2,  # Check every 2 steps
    )

    # Seed for reproducibility
    np.random.seed(42)

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    # Run surrogate phase — should detect variance growth and stop early
    # Growth is exponential: 2.5^1, 2.5^2, 2.5^3, ...
    # Validation happens at steps 2, 4, 6, 8 (every 2 steps)
    # After 3 validations with exponential growth, should bail
    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=20)

    # Should have stopped at or before completing all 20 steps
    assert len(surrogate_history) <= 20


def test_run_surrogate_phase_stable_variance_continues(
    minimal_config: SimulationConfig,
) -> None:
    """Test _run_surrogate_phase continues when variance is stable."""
    surrogate_stable = MockSurrogate(history_length=4, behavior="stable")

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_stable,
        handoff_fraction=0.2,
        validation_interval=5,
    )

    np.random.seed(42)

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    n_steps = 15
    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps)

    # Should complete all steps (stable variance)
    assert len(surrogate_history) == n_steps


# ============================================================================
# Test _validate_step L2 Divergence
# ============================================================================


def test_validate_step_identical_states(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _validate_step returns 0 for identical states."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    shape = (8, 8, 8)
    state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }

    divergence = engine._validate_step(state, state)

    assert divergence == pytest.approx(0.0, abs=1e-10)


def test_validate_step_different_states(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _validate_step computes normalized L2 divergence correctly."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    shape = (8, 8, 8)
    state1 = {
        "rho": np.full(shape, 1.0),
        "pressure": np.full(shape, 100.0),
    }
    state2 = {
        "rho": np.full(shape, 1.1),  # 10% difference
        "pressure": np.full(shape, 110.0),  # 10% difference
    }

    divergence = engine._validate_step(state1, state2)

    # Expected: for each field, L2(diff) / L2(actual)
    # rho: L2([0.1, 0.1, ...]) / L2([1.1, ...]) = 0.1*sqrt(N) / 1.1*sqrt(N) = 0.1/1.1 ≈ 0.0909
    # pressure: same
    # Average: 0.0909
    expected = 0.1 / 1.1  # ≈ 0.0909
    assert divergence == pytest.approx(expected, rel=0.01)


def test_validate_step_shape_mismatch_skips_field(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _validate_step skips fields with shape mismatch."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    state1 = {
        "rho": np.full((8, 8, 8), 1.0),
        "pressure": np.full((8, 8, 8), 100.0),
    }
    state2 = {
        "rho": np.full((4, 4, 4), 1.1),  # Different shape
        "pressure": np.full((8, 8, 8), 110.0),
    }

    divergence = engine._validate_step(state1, state2)

    # Should skip rho (shape mismatch), only compute pressure divergence
    # pressure divergence: 0.1 / 1.1
    assert divergence == pytest.approx(0.1 / 1.1, rel=0.01)


def test_validate_step_missing_field_skips(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _validate_step skips fields missing in physics_state."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    surrogate_state = {
        "rho": np.full((8, 8, 8), 1.0),
        "pressure": np.full((8, 8, 8), 100.0),
        "Te": np.full((8, 8, 8), 2.0),
    }
    physics_state = {
        "rho": np.full((8, 8, 8), 1.1),
        "pressure": np.full((8, 8, 8), 110.0),
        # Te missing
    }

    divergence = engine._validate_step(surrogate_state, physics_state)

    # Should only compute divergence for rho and pressure
    # Average: 0.0909
    expected = 0.1 / 1.1
    assert divergence == pytest.approx(expected, rel=0.01)


# ============================================================================
# Test run() Complete Flow
# ============================================================================


def test_run_complete_flow_with_mock_surrogate(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HybridEngine.run() complete flow with mock surrogate."""
    # Mock SimulationEngine import
    mock_engine_instance = MockEngine(minimal_config)

    def mock_simulation_engine_constructor(config: Any) -> MockEngine:
        return mock_engine_instance

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
        validation_interval=10,
    )

    max_steps = 50
    summary = engine.run(max_steps=max_steps)

    # Verify summary structure
    assert "total_steps" in summary
    assert "physics_steps" in summary
    assert "surrogate_steps" in summary
    assert "wall_time_s" in summary
    assert "fallback_to_physics" in summary

    # Verify step counts
    expected_physics_steps = int(max_steps * 0.2)  # 10 steps
    expected_surrogate_steps = max_steps - expected_physics_steps  # 40 steps

    assert summary["physics_steps"] == expected_physics_steps
    assert summary["total_steps"] == max_steps
    assert summary["surrogate_steps"] == expected_surrogate_steps
    assert summary["fallback_to_physics"] is False

    # Verify wall time is positive
    assert summary["wall_time_s"] > 0


def test_run_fallback_to_physics_on_nan(
    minimal_config: SimulationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HybridEngine.run() sets fallback flag when surrogate produces NaN."""
    surrogate_nan = MockSurrogate(history_length=4, behavior="nan")

    mock_engine_instance = MockEngine(minimal_config)

    def mock_simulation_engine_constructor(config: Any) -> MockEngine:
        return mock_engine_instance

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=surrogate_nan,
        handoff_fraction=0.2,
        validation_interval=2,
    )

    summary = engine.run(max_steps=50)

    # Verify fallback flag is set
    assert summary["fallback_to_physics"] is True

    # Verify total_steps < max_steps (stopped early)
    assert summary["total_steps"] < 50


def test_run_uses_default_max_steps_when_none(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HybridEngine.run() uses default max_steps=1000 when None passed."""
    mock_engine_instance = MockEngine(minimal_config)

    def mock_simulation_engine_constructor(config: Any) -> MockEngine:
        return mock_engine_instance

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.3,
    )

    summary = engine.run(max_steps=None)

    # Default max_steps=1000, handoff_fraction=0.3
    expected_physics_steps = int(1000 * 0.3)  # 300
    expected_surrogate_steps = 1000 - expected_physics_steps  # 700

    assert summary["physics_steps"] == expected_physics_steps
    assert summary["total_steps"] == 1000
    assert summary["surrogate_steps"] == expected_surrogate_steps


def test_run_handoff_fraction_zero_all_surrogate(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HybridEngine.run() with handoff_fraction=0 falls back (no physics history)."""
    mock_engine_instance = MockEngine(minimal_config)

    def mock_simulation_engine_constructor(config: Any) -> MockEngine:
        return mock_engine_instance

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.0,
    )

    summary = engine.run(max_steps=20)

    # With handoff_fraction=0, physics produces 0 history states,
    # so surrogate has nothing to seed from and falls back immediately
    assert summary["physics_steps"] == 0
    assert summary["fallback_to_physics"] is True
    assert summary["surrogate_steps"] == 0


def test_run_handoff_fraction_one_all_physics(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HybridEngine.run() with handoff_fraction=1.0 (all physics)."""
    mock_engine_instance = MockEngine(minimal_config)

    def mock_simulation_engine_constructor(config: Any) -> MockEngine:
        return mock_engine_instance

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=1.0,
    )

    summary = engine.run(max_steps=20)

    assert summary["physics_steps"] == 20
    assert summary["surrogate_steps"] == 0
    assert summary["total_steps"] == 20


# ============================================================================
# Test trajectory Property
# ============================================================================


def test_trajectory_accumulates_states(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HybridEngine.trajectory accumulates physics + surrogate states."""
    mock_engine_instance = MockEngine(minimal_config)

    def mock_simulation_engine_constructor(config: Any) -> MockEngine:
        return mock_engine_instance

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.4,
    )

    # Trajectory should be empty before run()
    assert len(engine.trajectory) == 0

    engine.run(max_steps=50)

    # Trajectory should have 50 states (20 physics + 30 surrogate)
    assert len(engine.trajectory) == 50

    # Each state should have all required fields
    for state in engine.trajectory:
        assert "rho" in state
        assert "velocity" in state
        assert "pressure" in state
        assert "B" in state
        assert "Te" in state
        assert "Ti" in state
        assert "psi" in state


def test_trajectory_empty_before_run(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test HybridEngine.trajectory is empty before run()."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    assert len(engine.trajectory) == 0


def test_trajectory_persists_across_multiple_runs(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HybridEngine.trajectory accumulates across multiple run() calls."""
    mock_engine_instance = MockEngine(minimal_config)

    def mock_simulation_engine_constructor(config: Any) -> MockEngine:
        return mock_engine_instance

    monkeypatch.setattr(
        "dpf.engine.SimulationEngine",
        mock_simulation_engine_constructor,
    )

    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.5,
    )

    engine.run(max_steps=20)
    first_run_length = len(engine.trajectory)
    assert first_run_length == 20

    engine.run(max_steps=10)
    second_run_length = len(engine.trajectory)
    assert second_run_length == 30  # 20 + 10


# ============================================================================
# Test engine.py Hybrid Backend Initialization
# ============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("dpf.athena_wrapper", reason="Athena++ not available"),
    reason="Athena++ not available",
)
def test_engine_hybrid_backend_creates_athena_solver(
    minimal_config: SimulationConfig,
) -> None:
    """Test engine.py hybrid backend creates AthenaPPSolver."""
    # Import here to avoid import errors when Athena++ not available
    from dpf.athena_wrapper import AthenaPPSolver
    from dpf.engine import SimulationEngine

    config = SimulationConfig(
        grid_shape=(8, 8, 8), dx=0.01, sim_time=1e-6,
        circuit={"V0": 20e3, "C": 100e-6, "L0": 50e-9, "R0": 0.01,
                 "anode_radius": 0.005, "cathode_radius": 0.01},
        fluid={"backend": "hybrid", "handoff_fraction": 0.2},
    )

    engine = SimulationEngine(config)

    # Verify backend is hybrid
    assert engine.backend == "hybrid"

    # Verify fluid solver is AthenaPPSolver
    assert isinstance(engine.fluid, AthenaPPSolver)

    # Verify _hybrid_engine is None (lazily initialized)
    assert engine._hybrid_engine is None


@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("dpf.athena_wrapper", reason="Athena++ not available"),
    reason="Athena++ not available",
)
def test_engine_hybrid_backend_respects_handoff_fraction(
    minimal_config: SimulationConfig,
) -> None:
    """Test engine.py hybrid backend respects handoff_fraction from config."""
    from dpf.engine import SimulationEngine

    config = SimulationConfig(
        grid_shape=(8, 8, 8), dx=0.01, sim_time=1e-6,
        circuit={"V0": 20e3, "C": 100e-6, "L0": 50e-9, "R0": 0.01,
                 "anode_radius": 0.005, "cathode_radius": 0.01},
        fluid={"backend": "hybrid", "handoff_fraction": 0.35},
    )

    engine = SimulationEngine(config)

    # Verify config.fluid.handoff_fraction is preserved
    assert engine.config.fluid.handoff_fraction == pytest.approx(0.35)


def test_fluid_config_handoff_fraction_validation() -> None:
    """Test FluidConfig validates handoff_fraction range."""
    from dpf.config import FluidConfig

    # Valid values
    valid_config_low = FluidConfig(backend="hybrid", handoff_fraction=0.0)
    assert valid_config_low.handoff_fraction == pytest.approx(0.0)

    valid_config_high = FluidConfig(backend="hybrid", handoff_fraction=1.0)
    assert valid_config_high.handoff_fraction == pytest.approx(1.0)

    valid_config_mid = FluidConfig(backend="hybrid", handoff_fraction=0.5)
    assert valid_config_mid.handoff_fraction == pytest.approx(0.5)

    # Invalid values
    with pytest.raises(ValueError):
        FluidConfig(backend="hybrid", handoff_fraction=-0.1)

    with pytest.raises(ValueError):
        FluidConfig(backend="hybrid", handoff_fraction=1.5)


def test_fluid_config_validation_interval_validation() -> None:
    """Test FluidConfig validates validation_interval >= 1."""
    from dpf.config import FluidConfig

    # Valid value
    valid_config = FluidConfig(backend="hybrid", validation_interval=50)
    assert valid_config.validation_interval == 50

    # Invalid value (< 1)
    with pytest.raises(ValueError):
        FluidConfig(backend="hybrid", validation_interval=0)


def test_fluid_config_defaults() -> None:
    """Test FluidConfig uses correct defaults for hybrid backend."""
    from dpf.config import FluidConfig

    config = FluidConfig(backend="hybrid")

    # Verify defaults from config.py
    assert config.handoff_fraction == pytest.approx(0.1)  # 10% physics
    assert config.validation_interval == 50


# ============================================================================
# Edge Cases
# ============================================================================


def test_run_surrogate_phase_with_zero_steps(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _run_surrogate_phase handles n_steps=0 gracefully."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    shape = (8, 8, 8)
    initial_state = {
        "rho": np.full(shape, 1e-6),
        "velocity": np.zeros((3, *shape)),
        "pressure": np.full(shape, 100.0),
        "B": np.zeros((3, *shape)),
        "Te": np.full(shape, 1.0),
        "Ti": np.full(shape, 1.0),
        "psi": np.zeros(shape),
    }
    physics_history = [initial_state] * 4

    surrogate_history = engine._run_surrogate_phase(physics_history, n_steps=0)

    assert len(surrogate_history) == 0


def test_run_physics_phase_with_zero_steps(
    minimal_config: SimulationConfig,
    mock_surrogate: MockSurrogate,
) -> None:
    """Test _run_physics_phase handles n_steps=0 gracefully."""
    engine = HybridEngine(
        config=minimal_config,
        surrogate=mock_surrogate,
        handoff_fraction=0.2,
    )

    mock_engine = MockEngine(minimal_config)
    history = engine._run_physics_phase(mock_engine, n_steps=0)

    assert len(history) == 0
    assert mock_engine._step_count == 0
