"""JAX-based differentiable DPF models.

Provides gradient-enabled Lee model for inverse design, sensitivity
analysis, and gradient-based calibration of Dense Plasma Focus devices.

Requires: jax, jaxlib, optax
"""

from dpf.jax.lee_model import calibrate, loss_fn, sensitivity, simulate

__all__ = ["simulate", "loss_fn", "calibrate", "sensitivity"]
