from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dpf.ai import HAS_TORCH
from dpf.ai.field_mapping import SCALAR_FIELDS
from dpf.ai.surrogate import DPFSurrogate

logger = logging.getLogger(__name__)


@dataclass
class PredictionWithConfidence:
    """
    Ensemble surrogate prediction with uncertainty quantification.

    Attributes:
        mean_state: Mean prediction across ensemble models
        std_state: Standard deviation per field
        confidence: Overall confidence score in [0, 1]
        ood_score: Out-of-distribution detection score
        n_models: Number of ensemble members
    """

    mean_state: dict[str, np.ndarray] = field(default_factory=dict)
    std_state: dict[str, np.ndarray] = field(default_factory=dict)
    confidence: float = 1.0
    ood_score: float = 0.0
    n_models: int = 1


class EnsemblePredictor:
    """
    Ensemble of WALRUS models for uncertainty-aware predictions.

    Uses multiple trained checkpoints to quantify prediction uncertainty
    and detect out-of-distribution states.

    Args:
        checkpoint_paths: Paths to trained WALRUS model checkpoints
        device: PyTorch device ("cpu", "cuda", "mps")
        history_length: Number of past states for prediction
        confidence_threshold: Minimum confidence for reliable predictions

    Raises:
        ImportError: If PyTorch not available
    """

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        device: str = "cpu",
        history_length: int = 4,
        confidence_threshold: float = 0.8,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch required for EnsemblePredictor. Install with: pip install torch"
            )

        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.device = device
        self.history_length = history_length
        self.confidence_threshold = confidence_threshold

        # Load all models
        self._models: list[DPFSurrogate] = []
        for ckpt_path in self.checkpoint_paths:
            try:
                model = DPFSurrogate(ckpt_path, device=device)
                self._models.append(model)
                logger.info(f"Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")

        if not self._models:
            raise ValueError("No valid checkpoints loaded")

        logger.info(
            f"EnsemblePredictor initialized: {len(self._models)} models, "
            f"device={device}, confidence_threshold={confidence_threshold}"
        )

    @property
    def n_models(self) -> int:
        """Number of ensemble members."""
        return len(self._models)

    def predict(self, history: list[dict[str, np.ndarray]]) -> PredictionWithConfidence:
        """
        Generate ensemble prediction with uncertainty quantification.

        Args:
            history: Recent physics history

        Returns:
            PredictionWithConfidence with mean, std, and confidence
        """
        if len(history) < self.history_length:
            logger.warning(
                f"History length {len(history)} < required {self.history_length}, "
                f"padding with zeros"
            )

        # Collect predictions from all models
        predictions: list[dict[str, np.ndarray]] = []
        for model in self._models:
            pred = model.predict_next_step(history)
            predictions.append(pred)

        # Stack predictions for each field
        # Assume all predictions have same fields and shapes
        all_fields = predictions[0].keys()
        stacked: dict[str, np.ndarray] = {}

        for field_name in all_fields:
            field_predictions = [pred[field_name] for pred in predictions]
            stacked[field_name] = np.stack(field_predictions, axis=0)

        # Compute mean and std per field
        mean_state = {}
        std_state = {}

        for field_name, field_stack in stacked.items():
            mean_state[field_name] = np.mean(field_stack, axis=0)
            std_state[field_name] = np.std(field_stack, axis=0)

        # Compute overall confidence
        confidence = self._compute_confidence(std_state)

        logger.debug(
            f"Ensemble prediction: {len(predictions)} models, confidence={confidence:.4f}"
        )

        return PredictionWithConfidence(
            mean_state=mean_state,
            std_state=std_state,
            confidence=confidence,
            ood_score=0.0,  # Computed separately via ood_detection()
            n_models=len(predictions),
        )

    def is_confident(self, prediction: PredictionWithConfidence) -> bool:
        """
        Check if ensemble prediction meets confidence threshold.

        Args:
            prediction: Ensemble prediction with confidence score

        Returns:
            True if confidence >= threshold
        """
        return prediction.confidence >= self.confidence_threshold

    def ood_detection(
        self,
        state: dict[str, np.ndarray],
        training_stats: dict[str, dict[str, float]] | None = None,
    ) -> float:
        """
        Detect out-of-distribution states via Mahalanobis-like distance.

        Args:
            state: Current simulation state
            training_stats: Per-field statistics (keys: mean, std)

        Returns:
            OOD score (0 = in-distribution, higher = more OOD)
        """
        if training_stats is None:
            logger.debug("No training stats provided, OOD score = 0.0")
            return 0.0

        distances = []

        for field_name in SCALAR_FIELDS:
            if field_name not in state or field_name not in training_stats:
                continue

            field_array = state[field_name]
            field_stats = training_stats[field_name]

            field_mean_val = np.mean(field_array)
            training_mean = field_stats.get("mean", 0.0)
            training_std = field_stats.get("std", 1.0)

            # Normalized distance
            distance = abs(field_mean_val - training_mean) / max(training_std, 1e-10)
            distances.append(distance)

        if not distances:
            logger.warning("No common fields for OOD detection")
            return 0.0

        ood_score = float(np.mean(distances))

        logger.debug(f"OOD detection: score={ood_score:.4f}")

        return ood_score

    def _compute_confidence(self, std_state: dict[str, np.ndarray]) -> float:
        """
        Convert ensemble std to confidence score.

        Args:
            std_state: Per-field standard deviation maps

        Returns:
            Confidence in [0, 1] (1 = high confidence, 0 = low confidence)
        """
        # Compute mean relative std across all fields
        relative_stds = []

        for _field_name, std_array in std_state.items():
            # Mean std for this field
            mean_std = np.mean(std_array)
            relative_stds.append(mean_std)

        if not relative_stds:
            logger.warning("No fields for confidence computation, returning 0.0")
            return 0.0

        mean_std = float(np.mean(relative_stds))

        # Convert to confidence: low std → high confidence
        # Use 1 / (1 + mean_std) mapping
        confidence = 1.0 / (1.0 + mean_std)

        # Clip to [0, 1]
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return confidence
