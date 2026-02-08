"""Adaptive mesh refinement for 2D axisymmetric DPF simulations.

Provides patch-based AMR with gradient-based tagging, bilinear prolongation,
and volume-weighted restriction for cylindrical (r, z) geometry.
"""

from dpf.experimental.amr.grid import (
    AMRConfig,
    AMRGrid,
    AMRPatch,
    prolong_patch,
    restrict_patch,
    tag_cells_gradient,
)

__all__ = [
    "AMRConfig",
    "AMRGrid",
    "AMRPatch",
    "prolong_patch",
    "restrict_patch",
    "tag_cells_gradient",
]
