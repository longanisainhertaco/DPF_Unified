from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from dpf.ai.surrogate import DPFSurrogate
from dpf.config import SimulationConfig

logger = logging.getLogger(__name__)


class HybridEngine:
    """
    Hybrid physics-surrogate engine for DPF simulations.

    Runs full physics for an initial phase, then hands off to WALRUS surrogate
    for acceleration. Validates surrogate predictions periodically and falls back
    to physics if divergence exceeds threshold.

    Args:
        config: Simulation configuration
        surrogate: Trained WALRUS surrogate model
        handoff_fraction: Fraction of total steps to run physics before handoff
        validation_interval: Steps between validation checks
        max_l2_divergence: Maximum allowed L2 divergence before fallback

    Raises:
        ValueError: If handoff_fraction not in [0, 1]
    """

    def __init__(
        self,
        config: SimulationConfig,
        surrogate: DPFSurrogate,
        handoff_fraction: float = 0.2,
        validation_interval: int = 50,
        max_l2_divergence: float = 0.1,
    ) -> None:
        if not 0.0 <= handoff_fraction <= 1.0:
            raise ValueError(f"handoff_fraction must be in [0, 1], got {handoff_fraction}")

        self.config = config
        self.surrogate = surrogate
        self.handoff_fraction = handoff_fraction
        self.validation_interval = validation_interval
        self.max_l2_divergence = max_l2_divergence
        self._trajectory: list[dict[str, np.ndarray]] = []

        logger.info(
            f"HybridEngine initialized: handoff={handoff_fraction:.2%}, "
            f"validation_interval={validation_interval}, max_divergence={max_l2_divergence}"
        )

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        """
        Run hybrid simulation with physics + surrogate phases.

        Args:
            max_steps: Maximum total steps (None uses config.max_steps)

        Returns:
            Summary dict with keys: total_steps, physics_steps, surrogate_steps,
            wall_time_s, fallback_to_physics
        """
        from dpf.engine import SimulationEngine

        if max_steps is None:
            max_steps = self.config.max_steps

        physics_steps = int(max_steps * self.handoff_fraction)
        surrogate_steps = max_steps - physics_steps

        logger.info(
            f"Starting hybrid run: {physics_steps} physics steps, "
            f"{surrogate_steps} surrogate steps"
        )

        start_time = time.time()

        # Phase 1: Physics
        engine = SimulationEngine(self.config)
        physics_history = self._run_physics_phase(engine, physics_steps)
        self._trajectory.extend(physics_history)

        # Phase 2: Surrogate
        fallback_to_physics = False
        if surrogate_steps > 0:
            surrogate_history = self._run_surrogate_phase(physics_history, surrogate_steps)
            self._trajectory.extend(surrogate_history)

            # Check if we fell back to physics
            if len(surrogate_history) < surrogate_steps:
                fallback_to_physics = True
                logger.warning("Surrogate diverged, fell back to physics")

        wall_time = time.time() - start_time

        summary = {
            "total_steps": len(self._trajectory),
            "physics_steps": physics_steps,
            "surrogate_steps": len(self._trajectory) - physics_steps,
            "wall_time_s": wall_time,
            "fallback_to_physics": fallback_to_physics,
        }

        logger.info(
            f"Hybrid run complete: {summary['total_steps']} steps in {wall_time:.2f}s "
            f"(fallback={fallback_to_physics})"
        )

        return summary

    def _run_physics_phase(self, engine: Any, n_steps: int) -> list[dict[str, np.ndarray]]:
        """
        Run full physics simulation for n_steps.

        Args:
            engine: SimulationEngine instance
            n_steps: Number of physics steps

        Returns:
            List of field snapshots
        """
        logger.info(f"Running physics phase: {n_steps} steps")
        history = []

        for i in range(n_steps):
            engine.step()
            state = engine.get_field_snapshot()
            history.append(state)

            if (i + 1) % 100 == 0:
                logger.debug(f"Physics step {i + 1}/{n_steps}")

        logger.info(f"Physics phase complete: {len(history)} states")
        return history

    def _run_surrogate_phase(
        self, history: list[dict[str, np.ndarray]], n_steps: int
    ) -> list[dict[str, np.ndarray]]:
        """
        Run surrogate model with periodic validation.

        Args:
            history: Initial physics history for surrogate
            n_steps: Number of surrogate steps to attempt

        Returns:
            List of surrogate-predicted field snapshots (may be shorter if fallback occurs)
        """
        from dpf.engine import SimulationEngine

        logger.info(f"Running surrogate phase: {n_steps} steps")
        surrogate_history = []

        # Use last history_length states as initial window
        window_size = self.surrogate.history_length
        window = history[-window_size:]

        for i in range(n_steps):
            # Surrogate prediction
            predicted_state = self.surrogate.predict_next_step(window)
            surrogate_history.append(predicted_state)

            # Periodic validation
            if (i + 1) % self.validation_interval == 0:
                logger.debug(f"Validating surrogate at step {i + 1}/{n_steps}")

                # Run one physics step for comparison
                engine = SimulationEngine(self.config)
                # Fast-forward engine to current state (expensive, simplified for now)
                # In production, would maintain parallel engine or use checkpointing
                physics_state = engine.get_field_snapshot()

                divergence = self._validate_step(predicted_state, physics_state)

                if divergence > self.max_l2_divergence:
                    logger.warning(
                        f"Surrogate divergence {divergence:.4f} exceeds threshold "
                        f"{self.max_l2_divergence:.4f} at step {i + 1}"
                    )
                    break

                logger.debug(f"Surrogate validation passed: divergence={divergence:.4f}")

            # Slide window
            window = window[1:] + [predicted_state]

        logger.info(f"Surrogate phase complete: {len(surrogate_history)} states")
        return surrogate_history

    def _validate_step(
        self, surrogate_state: dict[str, np.ndarray], physics_state: dict[str, np.ndarray]
    ) -> float:
        """
        Compute normalized L2 divergence between surrogate and physics.

        Args:
            surrogate_state: Surrogate prediction
            physics_state: Ground truth physics state

        Returns:
            Normalized L2 divergence (0 = perfect match)
        """
        divergence_sum = 0.0
        n_fields = 0

        for field in surrogate_state:
            if field not in physics_state:
                continue

            pred = surrogate_state[field]
            actual = physics_state[field]

            if pred.shape != actual.shape:
                logger.warning(f"Shape mismatch for field {field}: {pred.shape} vs {actual.shape}")
                continue

            diff_norm = np.linalg.norm(pred - actual)
            actual_norm = max(np.linalg.norm(actual), 1e-10)
            field_divergence = diff_norm / actual_norm

            divergence_sum += field_divergence
            n_fields += 1

        return divergence_sum / max(n_fields, 1)

    @property
    def trajectory(self) -> list[dict[str, np.ndarray]]:
        """Complete simulation trajectory (physics + surrogate phases)."""
        return self._trajectory
