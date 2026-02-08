from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dpf.ai.surrogate import DPFSurrogate

logger = logging.getLogger(__name__)


@dataclass
class InstabilityEvent:
    """
    Detected instability event from surrogate-physics divergence.

    Attributes:
        step: Simulation step number
        time: Physical time (s)
        l2_divergence: Overall normalized L2 divergence
        field_divergences: Per-field divergence map
        severity: Classification ("low", "medium", "high")
    """

    step: int
    time: float
    l2_divergence: float
    field_divergences: dict[str, float]
    severity: str


class InstabilityDetector:
    """
    Detects instability onset by comparing WALRUS predictions to actual physics.

    Monitors divergence between surrogate predictions and ground truth physics
    states. Classifies instability severity and provides per-field diagnostics.

    Args:
        surrogate: Trained WALRUS surrogate model
        threshold_low: Minimum divergence to flag as instability
        threshold_medium: Divergence threshold for medium severity
        threshold_high: Divergence threshold for high severity
    """

    def __init__(
        self,
        surrogate: DPFSurrogate,
        threshold_low: float = 0.05,
        threshold_medium: float = 0.15,
        threshold_high: float = 0.3,
    ) -> None:
        self.surrogate = surrogate
        self.threshold_low = threshold_low
        self.threshold_medium = threshold_medium
        self.threshold_high = threshold_high

        logger.info(
            f"InstabilityDetector initialized: low={threshold_low}, "
            f"medium={threshold_medium}, high={threshold_high}"
        )

    def check(
        self,
        history: list[dict[str, np.ndarray]],
        actual_next: dict[str, np.ndarray],
        step: int = 0,
        time: float = 0.0,
    ) -> InstabilityEvent | None:
        """
        Check for instability at a single timestep.

        Args:
            history: Recent physics history for surrogate
            actual_next: Ground truth next state
            step: Simulation step number
            time: Physical time (s)

        Returns:
            InstabilityEvent if divergence exceeds threshold_low, else None
        """
        predicted_next = self.surrogate.predict_next_step(history)

        overall_divergence, field_divergences = self.compute_divergence(predicted_next, actual_next)

        if overall_divergence < self.threshold_low:
            return None

        severity = self.classify_severity(overall_divergence)

        logger.info(
            f"Instability detected at step {step}, time {time:.3e}s: "
            f"divergence={overall_divergence:.4f}, severity={severity}"
        )

        return InstabilityEvent(
            step=step,
            time=time,
            l2_divergence=overall_divergence,
            field_divergences=field_divergences,
            severity=severity,
        )

    def compute_divergence(
        self, predicted: dict[str, np.ndarray], actual: dict[str, np.ndarray]
    ) -> tuple[float, dict[str, float]]:
        """
        Compute normalized L2 divergence for all scalar fields.

        Args:
            predicted: Surrogate prediction
            actual: Ground truth state

        Returns:
            Tuple of (overall_divergence, per_field_divergences)
        """
        field_divergences = {}

        # Only compute for fields present in both states
        common_fields = set(predicted.keys()) & set(actual.keys())

        for field in common_fields:
            pred_array = predicted[field]
            actual_array = actual[field]

            if pred_array.shape != actual_array.shape:
                logger.warning(
                    f"Shape mismatch for field {field}: {pred_array.shape} vs {actual_array.shape}"
                )
                continue

            diff_norm = np.linalg.norm(pred_array - actual_array)
            actual_norm = max(np.linalg.norm(actual_array), 1e-10)
            field_divergence = diff_norm / actual_norm

            field_divergences[field] = float(field_divergence)

        # Overall divergence is mean of per-field divergences
        if field_divergences:
            overall = float(np.mean(list(field_divergences.values())))
        else:
            overall = 0.0
            logger.warning("No common fields found for divergence computation")

        return overall, field_divergences

    def classify_severity(self, l2_divergence: float) -> str:
        """
        Classify instability severity based on divergence magnitude.

        Args:
            l2_divergence: Overall normalized L2 divergence

        Returns:
            Severity string: "low", "medium", or "high"
        """
        if l2_divergence >= self.threshold_high:
            return "high"
        elif l2_divergence >= self.threshold_medium:
            return "medium"
        else:
            return "low"

    def monitor_trajectory(
        self, trajectory: list[dict[str, np.ndarray]]
    ) -> list[InstabilityEvent]:
        """
        Monitor full trajectory for instability events.

        Slides a window of history_length over the trajectory, checking each
        step for surrogate-physics divergence.

        Args:
            trajectory: Complete simulation trajectory

        Returns:
            List of detected InstabilityEvents (chronological order)
        """
        history_length = self.surrogate.history_length
        events = []

        if len(trajectory) <= history_length:
            logger.warning(
                f"Trajectory too short for monitoring: {len(trajectory)} states "
                f"vs history_length={history_length}"
            )
            return events

        logger.info(
            f"Monitoring trajectory: {len(trajectory)} states, "
            f"history_length={history_length}"
        )

        # Slide window over trajectory
        for i in range(history_length, len(trajectory)):
            history = trajectory[i - history_length : i]
            actual_next = trajectory[i]

            # Use index as step number, placeholder time
            event = self.check(history, actual_next, step=i, time=float(i))

            if event is not None:
                events.append(event)

        logger.info(
            f"Trajectory monitoring complete: {len(events)} instability events detected"
        )

        return events
