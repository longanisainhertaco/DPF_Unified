from __future__ import annotations

import numpy as np
import pytest

from dpf.ai.instability_detector import InstabilityDetector, InstabilityEvent


class MockSurrogate:
    """Mock surrogate for testing InstabilityDetector."""

    def __init__(self, noise_level: float = 0.0):
        self.history_length = 4
        self.noise_level = noise_level

    def predict_next_step(self, history: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Return last state plus controlled noise."""
        last = history[-1]
        return {k: v.copy() + np.random.normal(0, self.noise_level, v.shape) for k, v in last.items()}


def _make_state(shape: tuple[int, ...] = (8, 8, 8), value: float = 1.0) -> dict[str, np.ndarray]:
    """Create standard physics state dict for testing."""
    return {
        "rho": np.ones(shape) * value,
        "velocity": np.zeros((3, *shape)),
        "pressure": np.ones(shape) * 100.0,
        "B": np.zeros((3, *shape)),
        "Te": np.ones(shape) * 1e4,
        "Ti": np.ones(shape) * 1e4,
        "psi": np.zeros(shape),
    }


class TestInstabilityEvent:
    """Tests for InstabilityEvent dataclass."""

    def test_event_stores_all_fields(self):
        """InstabilityEvent stores step, time, divergence, field divergences, and severity."""
        field_divergences = {"rho": 0.1, "pressure": 0.2}
        event = InstabilityEvent(
            step=42,
            time=1.5e-6,
            l2_divergence=0.15,
            field_divergences=field_divergences,
            severity="medium",
        )

        assert event.step == 42
        assert event.time == pytest.approx(1.5e-6)
        assert event.l2_divergence == pytest.approx(0.15)
        assert event.field_divergences == field_divergences
        assert event.severity == "medium"


class TestInstabilityDetectorInit:
    """Tests for InstabilityDetector initialization."""

    def test_init_stores_threshold_values(self):
        """InstabilityDetector __init__ stores surrogate and threshold values."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(
            surrogate, threshold_low=0.1, threshold_medium=0.2, threshold_high=0.4
        )

        assert detector.surrogate is surrogate
        assert detector.threshold_low == pytest.approx(0.1)
        assert detector.threshold_medium == pytest.approx(0.2)
        assert detector.threshold_high == pytest.approx(0.4)

    def test_init_default_thresholds(self):
        """InstabilityDetector uses default thresholds when not specified."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate)

        assert detector.threshold_low == pytest.approx(0.05)
        assert detector.threshold_medium == pytest.approx(0.15)
        assert detector.threshold_high == pytest.approx(0.3)


class TestClassifySeverity:
    """Tests for severity classification."""

    def test_classify_severity_high(self):
        """classify_severity returns 'high' when divergence >= threshold_high."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate, threshold_high=0.3)

        assert detector.classify_severity(0.3) == "high"
        assert detector.classify_severity(0.5) == "high"
        assert detector.classify_severity(1.0) == "high"

    def test_classify_severity_medium(self):
        """classify_severity returns 'medium' when threshold_medium <= divergence < threshold_high."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate, threshold_medium=0.15, threshold_high=0.3)

        assert detector.classify_severity(0.15) == "medium"
        assert detector.classify_severity(0.2) == "medium"
        assert detector.classify_severity(0.29) == "medium"

    def test_classify_severity_low(self):
        """classify_severity returns 'low' when threshold_low <= divergence < threshold_medium."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(
            surrogate, threshold_low=0.05, threshold_medium=0.15, threshold_high=0.3
        )

        assert detector.classify_severity(0.05) == "low"
        assert detector.classify_severity(0.1) == "low"
        assert detector.classify_severity(0.14) == "low"


class TestComputeDivergence:
    """Tests for divergence computation."""

    def test_compute_divergence_identical_states(self):
        """compute_divergence returns 0.0 for identical states."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate)

        state = _make_state()
        overall, field_divs = detector.compute_divergence(state, state)

        assert overall == pytest.approx(0.0, abs=1e-10)
        for field_div in field_divs.values():
            assert field_div == pytest.approx(0.0, abs=1e-10)

    def test_compute_divergence_different_states(self):
        """compute_divergence returns positive value for different states."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate)

        state1 = _make_state(value=1.0)
        state2 = _make_state(value=1.5)

        overall, field_divs = detector.compute_divergence(state1, state2)

        assert overall > 0.0
        # Only rho changes with value parameter, so it should have positive divergence
        assert field_divs["rho"] > 0.0
        # Other fields are constant in _make_state, so zero divergence
        assert field_divs["pressure"] == pytest.approx(0.0, abs=1e-10)
        assert field_divs["Te"] == pytest.approx(0.0, abs=1e-10)
        assert field_divs["Ti"] == pytest.approx(0.0, abs=1e-10)

    def test_compute_divergence_missing_fields(self):
        """compute_divergence handles missing fields via empty intersection."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate)

        predicted = {"rho": np.ones((8, 8, 8))}
        actual = {"pressure": np.ones((8, 8, 8)) * 100.0}

        overall, field_divs = detector.compute_divergence(predicted, actual)

        # No common fields, so overall should be 0.0 and field_divs empty
        assert overall == pytest.approx(0.0)
        assert len(field_divs) == 0

    def test_compute_divergence_shape_mismatch(self, caplog):
        """compute_divergence logs warning and skips field for shape mismatch."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate)

        predicted = {"rho": np.ones((8, 8, 8)), "pressure": np.ones((8, 8, 8)) * 100.0}
        actual = {"rho": np.ones((4, 4, 4)), "pressure": np.ones((8, 8, 8)) * 100.0}

        overall, field_divs = detector.compute_divergence(predicted, actual)

        # rho should be skipped due to shape mismatch, only pressure should be computed
        assert "rho" not in field_divs
        assert "pressure" in field_divs
        assert "Shape mismatch for field rho" in caplog.text


class TestCheck:
    """Tests for single-step instability checking."""

    def test_check_returns_none_below_threshold(self):
        """check returns None when divergence < threshold_low with zero-noise mock."""
        surrogate = MockSurrogate(noise_level=0.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        state = _make_state()
        history = [state] * 4

        event = detector.check(history, state, step=10, time=1e-6)

        assert event is None

    def test_check_returns_event_above_threshold(self):
        """check returns InstabilityEvent when divergence > threshold_low with high-noise mock."""
        np.random.seed(42)
        surrogate = MockSurrogate(noise_level=10.0)  # High noise to ensure divergence > 0.05
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        state = _make_state()
        history = [state] * 4

        event = detector.check(history, state, step=10, time=1e-6)

        assert event is not None
        assert isinstance(event, InstabilityEvent)

    def test_check_event_correct_severity(self):
        """check event has correct severity classification."""
        np.random.seed(42)
        # Use high noise to push divergence into medium/high range
        surrogate = MockSurrogate(noise_level=50.0)
        detector = InstabilityDetector(
            surrogate, threshold_low=0.05, threshold_medium=0.15, threshold_high=0.3
        )

        state = _make_state()
        history = [state] * 4

        event = detector.check(history, state, step=10, time=1e-6)

        assert event is not None
        # Verify severity matches the divergence thresholds
        if event.l2_divergence >= 0.3:
            assert event.severity == "high"
        elif event.l2_divergence >= 0.15:
            assert event.severity == "medium"
        else:
            assert event.severity == "low"

    def test_check_event_correct_step_and_time(self):
        """check event has correct step and time values."""
        np.random.seed(42)
        surrogate = MockSurrogate(noise_level=10.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        state = _make_state()
        history = [state] * 4

        event = detector.check(history, state, step=42, time=2.5e-6)

        assert event is not None
        assert event.step == 42
        assert event.time == pytest.approx(2.5e-6)


class TestMonitorTrajectory:
    """Tests for trajectory monitoring."""

    def test_monitor_trajectory_empty_for_short_trajectory(self):
        """monitor_trajectory returns empty list for trajectory too short."""
        surrogate = MockSurrogate()
        detector = InstabilityDetector(surrogate)

        # Trajectory with only 3 states, but history_length is 4
        trajectory = [_make_state()] * 3

        events = detector.monitor_trajectory(trajectory)

        assert events == []

    def test_monitor_trajectory_detects_increasing_divergence(self):
        """monitor_trajectory detects instabilities in trajectory with increasing divergence."""
        np.random.seed(42)
        surrogate = MockSurrogate(noise_level=0.0)  # Zero noise, so divergence comes from trajectory
        detector = InstabilityDetector(surrogate, threshold_low=0.01)  # Lower threshold

        # Create trajectory where all fields change significantly
        trajectory = []
        for i in range(10):
            factor = 1.0 + i * 0.5
            state = {
                "rho": np.ones((8, 8, 8)) * factor,
                "velocity": np.ones((3, 8, 8, 8)) * factor * 0.1,
                "pressure": np.ones((8, 8, 8)) * 100.0 * factor,
                "B": np.ones((3, 8, 8, 8)) * factor * 0.01,
                "Te": np.ones((8, 8, 8)) * 1e4 * factor,
                "Ti": np.ones((8, 8, 8)) * 1e4 * factor,
                "psi": np.ones((8, 8, 8)) * factor * 0.001,
            }
            trajectory.append(state)

        events = detector.monitor_trajectory(trajectory)

        # Should detect instabilities once divergence exceeds threshold
        assert len(events) > 0
        # Events should be in chronological order
        for i in range(len(events) - 1):
            assert events[i].step < events[i + 1].step

    def test_monitor_trajectory_no_events_for_stable_trajectory(self):
        """monitor_trajectory returns empty list for stable trajectory."""
        surrogate = MockSurrogate(noise_level=0.0)
        detector = InstabilityDetector(surrogate, threshold_low=0.05)

        # Constant trajectory with zero noise should have zero divergence
        trajectory = [_make_state(value=1.0)] * 10

        events = detector.monitor_trajectory(trajectory)

        assert events == []
