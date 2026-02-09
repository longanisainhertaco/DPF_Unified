"""DPF AI module — surrogate models, training data pipeline, inverse design.

Provides WALRUS-based surrogate models for fast parameter sweeps,
inverse design, and real-time prediction. All torch/walrus dependencies
are optional — modules degrade gracefully when not installed.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Optional dependency flags
HAS_TORCH = False
HAS_WALRUS = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    logger.info("torch not available; AI surrogate features disabled")

try:
    from walrus.models import IsotropicModel  # noqa: F401

    HAS_WALRUS = True
except (ImportError, ModuleNotFoundError):
    logger.info("walrus not available; WALRUS model loading disabled")


def torch_available() -> bool:
    """Return True if PyTorch is installed."""
    return HAS_TORCH


def walrus_available() -> bool:
    """Return True if the WALRUS package is installed."""
    return HAS_WALRUS
