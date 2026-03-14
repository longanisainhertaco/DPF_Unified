from __future__ import annotations

import importlib.util

import pytest

from dpf.ai.inverse_design import InverseDesigner, InverseResult

# Check optional dependencies
HAS_OPTUNA = importlib.util.find_spec("optuna") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class MockSurrogate:
    """Mock surrogate that returns predictable results."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True

    def parameter_sweep(self, configs, n_steps=100):
        """
        Return results that scale predictably with input parameters.

        Uses simple normalized relationships:
        - max_rho = (V0/1e4) * (C/1e-6)
        - max_Te = V0 * 0.1
        - max_Ti = V0 * 0.05
        - max_B = (V0/1e4) * (C/1e-6) * 0.01

        Returns nested format matching real DPFSurrogate._extract_summary():
        {"config": {...}, "metrics": {...}}
        """
        results = []
        for config in configs:
            V0 = config.get("V0", 1e4)
            C = config.get("C", 1e-6)
            results.append({
                "config": config,
                "metrics": {
                    "max_rho": (V0 / 1e4) * (C / 1e-6),
                    "max_Te": V0 * 0.1,
                    "max_Ti": V0 * 0.05,
                    "max_B": (V0 / 1e4) * (C / 1e-6) * 0.01,
                    "n_steps": n_steps,
                },
            })
        return results


class FailingSurrogate:
    """Mock surrogate that always fails."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True

    def parameter_sweep(self, configs, n_steps=100):
        """Return error results."""
        return [{"error": "Mock failure"}]


class ExceptionSurrogate:
    """Mock surrogate that raises exceptions."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True

    def parameter_sweep(self, configs, n_steps=100):
        """Raise an exception."""
        raise RuntimeError("Mock exception")


@pytest.fixture
def mock_surrogate():
    """Create a mock surrogate model."""
    return MockSurrogate()


@pytest.fixture
def parameter_ranges():
    """Standard parameter ranges for testing."""
    return {
        "V0": (5e3, 2e4),
        "C": (5e-7, 2e-6),
    }


# ============================================================================
# InverseResult Tests
# ============================================================================


def test_inverse_result_defaults():
    """Test InverseResult default initialization."""
    result = InverseResult()
    assert result.best_params == {}
    assert result.best_score == float("inf")
    assert result.all_trials == []
    assert result.n_trials == 0


def test_inverse_result_custom_values():
    """Test InverseResult stores custom values."""
    params = {"V0": 1e4, "C": 1e-6}
    score = 0.123
    trials = [({"V0": 8e3}, 0.5), ({"V0": 1.2e4}, 0.2)]
    n_trials = 10

    result = InverseResult(
        best_params=params,
        best_score=score,
        all_trials=trials,
        n_trials=n_trials,
    )

    assert result.best_params == params
    assert result.best_score == score
    assert result.all_trials == trials
    assert result.n_trials == n_trials


# ============================================================================
# InverseDesigner Initialization Tests
# ============================================================================


def test_inverse_designer_init(mock_surrogate, parameter_ranges):
    """Test InverseDesigner stores surrogate and parameter_ranges."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    assert designer.surrogate is mock_surrogate
    assert designer.parameter_ranges == parameter_ranges


# ============================================================================
# Objective Function Tests
# ============================================================================


def test_objective_exact_match(mock_surrogate, parameter_ranges):
    """Test _objective returns 0.0 when prediction exactly matches targets."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)

    # Use params that will produce specific outputs
    params = {"V0": 1e4, "C": 1e-6}
    # Expected outputs: max_rho=1.0, max_Te=1000.0, max_Ti=500.0, max_B=0.01
    targets = {
        "max_rho": 1.0,  # (1e4/1e4) * (1e-6/1e-6) = 1.0
        "max_Te": 1000.0,  # 1e4 * 0.1 = 1000.0
    }

    score = designer._objective(params, targets, None)
    assert score == pytest.approx(0.0, abs=1e-10)


def test_objective_mismatch(mock_surrogate, parameter_ranges):
    """Test _objective returns positive value for mismatched targets."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    # Target values that don't match prediction
    targets = {
        "max_rho": 2.0,  # Prediction is 1.0
        "max_Te": 2000.0,  # Prediction is 1000.0
    }

    score = designer._objective(params, targets, None)
    # Normalized error for max_rho: |1.0 - 2.0| / 2.0 = 0.5
    # Normalized error for max_Te: |1000.0 - 2000.0| / 2000.0 = 0.5
    expected = 0.5 + 0.5
    assert score == pytest.approx(expected)


def test_objective_surrogate_failure():
    """Test _objective returns large penalty (1e10) when surrogate fails."""
    failing_surrogate = FailingSurrogate()
    parameter_ranges = {"V0": (5e3, 2e4), "C": (5e-7, 2e-6)}
    designer = InverseDesigner(failing_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"max_rho": 1.0}

    score = designer._objective(params, targets, None)
    assert score == 1e10


def test_objective_surrogate_exception():
    """Test _objective returns large penalty (1e10) when surrogate raises exception."""
    exception_surrogate = ExceptionSurrogate()
    parameter_ranges = {"V0": (5e3, 2e4), "C": (5e-7, 2e-6)}
    designer = InverseDesigner(exception_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"max_rho": 1.0}

    score = designer._objective(params, targets, None)
    assert score == 1e10


def test_objective_constraint_violation(mock_surrogate, parameter_ranges):
    """Test _objective applies constraint penalty for violated constraints."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    # Prediction: max_Te=1000.0
    targets = {"max_rho": 1.0}
    constraints = {"max_Te": 500.0}  # Violated: predicted 1000.0 > 500.0

    score = designer._objective(params, targets, constraints)

    # Base score from targets: 0.0 (exact match)
    # Constraint violation: (1000.0 - 500.0) / 500.0 = 1.0
    # Penalty: 10.0 * 1.0^2 = 10.0
    expected = 0.0 + 10.0
    assert score == pytest.approx(expected)


def test_objective_constraint_satisfied(mock_surrogate, parameter_ranges):
    """Test _objective does not add penalty when constraints are satisfied."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    # Prediction: max_Te=1000.0
    targets = {"max_rho": 1.0}
    constraints = {"max_Te": 2000.0}  # Satisfied: predicted 1000.0 < 2000.0

    score = designer._objective(params, targets, constraints)
    # Only base score, no constraint penalty
    assert score == pytest.approx(0.0, abs=1e-10)


def test_objective_missing_metric(mock_surrogate, parameter_ranges):
    """Test _objective handles missing metric in prediction."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)

    params = {"V0": 1e4, "C": 1e-6}
    targets = {"nonexistent_metric": 100.0}

    score = designer._objective(params, targets, None)
    # Should get large penalty for missing metric
    assert score == pytest.approx(1e6)


# ============================================================================
# find_config Tests
# ============================================================================


def test_find_config_invalid_method(mock_surrogate, parameter_ranges):
    """Test find_config raises ValueError for invalid method."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.0}

    with pytest.raises(ValueError, match="Invalid method"):
        designer.find_config(targets, method="invalid_method")


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_find_config_evolutionary(mock_surrogate, parameter_ranges):
    """Test find_config with method='evolutionary' returns InverseResult."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.5}  # Target: V0 * C * 1e6 = 1.5

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,  # Small for fast testing
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert "V0" in result.best_params
    assert "C" in result.best_params
    assert result.best_score < float("inf")
    assert result.n_trials > 0
    assert len(result.all_trials) > 0


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
def test_find_config_bayesian(mock_surrogate, parameter_ranges):
    """Test find_config with method='bayesian' returns InverseResult."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="bayesian",
        n_trials=20,  # Small for fast testing
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert "V0" in result.best_params
    assert "C" in result.best_params
    assert result.best_score < float("inf")
    assert result.n_trials == 20
    assert len(result.all_trials) == 20


def test_bayesian_search_no_optuna(mock_surrogate, parameter_ranges, monkeypatch):
    """Test _bayesian_search raises ImportError when optuna unavailable."""
    # Monkeypatch to simulate optuna not being available
    import dpf.ai.inverse_design

    monkeypatch.setattr(dpf.ai.inverse_design, "HAS_OPTUNA", False)

    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.0}

    with pytest.raises(ImportError, match="Optuna required"):
        designer._bayesian_search(targets, None, 10, 42)


def test_evolutionary_search_no_scipy(mock_surrogate, parameter_ranges, monkeypatch):
    """Test _evolutionary_search raises ImportError when scipy unavailable."""
    # Monkeypatch to simulate scipy not being available
    import dpf.ai.inverse_design

    monkeypatch.setattr(dpf.ai.inverse_design, "HAS_SCIPY", False)

    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.0}

    with pytest.raises(ImportError, match="SciPy required"):
        designer._evolutionary_search(targets, None, 10, 42)


# ============================================================================
# Parameter Bounds Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_parameters_within_bounds(mock_surrogate, parameter_ranges):
    """Test parameters stay within specified ranges during optimization."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    # Check best params are within bounds
    for param_name, value in result.best_params.items():
        low, high = parameter_ranges[param_name]
        assert low <= value <= high

    # Check all trials are within bounds
    for params, _ in result.all_trials:
        for param_name, value in params.items():
            low, high = parameter_ranges[param_name]
            assert low <= value <= high


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_multiple_parameter_ranges(mock_surrogate):
    """Test multiple parameter ranges handled correctly."""
    parameter_ranges = {
        "V0": (5e3, 2e4),
        "C": (5e-7, 2e-6),
        "L0": (1e-8, 1e-7),  # Extra parameter (not used by mock, but should work)
    }

    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.0}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    # Should have all three parameters
    assert len(result.best_params) == 3
    assert "V0" in result.best_params
    assert "C" in result.best_params
    assert "L0" in result.best_params


# ============================================================================
# Optimization History Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_all_trials_populated(mock_surrogate, parameter_ranges):
    """Test InverseResult.all_trials populated after optimization."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert len(result.all_trials) > 0
    # Each trial should be (params_dict, score)
    for params, score in result.all_trials:
        assert isinstance(params, dict)
        assert isinstance(score, (int, float))
        assert score >= 0.0  # Scores should be non-negative


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_n_trials_matches_actual_count(mock_surrogate, parameter_ranges):
    """Test InverseResult.n_trials matches actual trial count."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.5}

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert result.n_trials == len(result.all_trials)


# ============================================================================
# Multi-target and Multi-constraint Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_multiple_targets(mock_surrogate, parameter_ranges):
    """Test optimization with multiple target metrics."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {
        "max_rho": 1.5,
        "max_Te": 1200.0,
    }

    result = designer.find_config(
        targets=targets,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert result.best_score < float("inf")
    # With two targets, exact match is harder but should find reasonable solution
    assert result.best_score < 10.0


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_multiple_constraints(mock_surrogate, parameter_ranges):
    """Test optimization with multiple constraints."""
    designer = InverseDesigner(mock_surrogate, parameter_ranges)
    targets = {"max_rho": 1.0}
    constraints = {
        "max_Te": 1500.0,
        "max_Ti": 600.0,
    }

    result = designer.find_config(
        targets=targets,
        constraints=constraints,
        method="evolutionary",
        n_trials=30,
        seed=42,
    )

    assert isinstance(result, InverseResult)
    assert result.best_score < float("inf")
