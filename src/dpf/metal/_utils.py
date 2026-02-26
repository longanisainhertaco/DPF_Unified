"""Shared utility functions for the Metal GPU domain.

These helpers are used by both ``metal_stencil.py`` and ``metal_riemann.py``
to validate device placement and detect NaN values.
"""

from __future__ import annotations

import torch


def _ensure_mps(t: torch.Tensor, name: str = "tensor") -> None:
    """Validate that a tensor resides on the MPS or CPU device.

    CPU is accepted to support float64 precision mode, which PyTorch MPS
    does not support and therefore forces the Metal solver to run on CPU.

    Args:
        t: Tensor to check.
        name: Human-readable label for error messages.

    Raises:
        ValueError: If the tensor is not on an MPS or CPU device.
    """
    if t.device.type not in ("mps", "cpu"):
        raise ValueError(
            f"{name} must be on MPS or CPU device, got {t.device}"
        )


def _check_no_nan(t: torch.Tensor, label: str = "result") -> None:
    """Assert that a tensor contains no NaN values.

    Args:
        t: Tensor to validate.
        label: Context label for the assertion error message.

    Raises:
        AssertionError: If any element is NaN.
    """
    assert not torch.isnan(t).any(), f"NaN detected in {label}"
