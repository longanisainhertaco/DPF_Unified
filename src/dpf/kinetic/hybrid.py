"""Hybrid Fluid-PIC solver — re-exported from ``experimental.pic.hybrid``.

This module was previously a ~978-LOC duplicate of
``dpf.experimental.pic.hybrid`` with a divergent RNG (legacy
``np.random.random()`` vs. the canonical ``np.random.default_rng()``).

The duplicate has been removed.  All public symbols are now imported
directly from the canonical implementation so that ``kinetic.hybrid``
consumers keep working without changes.
"""

from __future__ import annotations

from dpf.experimental.pic.hybrid import (  # noqa: F401
    HybridPIC,
    ParticleSpecies,
    boris_push,
    deposit_current,
    deposit_density,
    detect_instability,
    interpolate_field_to_particles,
)

__all__ = [
    "HybridPIC",
    "ParticleSpecies",
    "boris_push",
    "deposit_current",
    "deposit_density",
    "detect_instability",
    "interpolate_field_to_particles",
]
