"""Electrode boundary conditions for the MLX cylindrical MHD solver.

Extracted from mlx_solver.py (God Class decomposition, Sprint S-3).

Handles ghost cell padding for electrode B_theta BCs:
- Inner ghosts (axis): reflecting with sign-flip on B_theta, B_r, mom_r, mom_t
- Outer ghosts (cathode): zero-gradient base + B_theta = mu0*I/(2*pi*r)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from dpf.metal.constants import MU_0, P_FLOOR, SQRT_MU0
from dpf.metal.mlx_device import require_mlx

logger = logging.getLogger(__name__)


def electrode_bt_fixup_mlx(
    U_padded: Any,
    r_cell: Any,
    current: float,
    ng: int,
    nr_phys: int,
    convert_si_to_hl: bool = True,
) -> Any:
    """Apply electrode B_theta BC in pure MLX (no np.asarray sync).

    Sets B_theta = mu0*I/(2*pi*r) [/ sqrt(mu0) if HL] at:
    From Ampere's law: Chen (2016), Intro to Plasma Physics, 3rd ed., p.189, Eq. (6.2).
    PDF on disk: references/papers/textbooks/chen-intro-plasma-physics-controlled-fusion-3ed.pdf
      - Outer ghost cells: indices [ng+nr_phys, ng+nr_phys+ng)
      - Outermost ng interior cells: blend with max(existing, electrode)
    Updates total energy for magnetic energy consistency throughout.
    """
    mx = require_mlx()
    from dpf.metal.constants import IBR, IBT, IBZ, IDN, IEE, IEN, IMR, IMT, IMZ, ISR

    _GAMMA_LOCAL = 5.0 / 3.0
    divisor = SQRT_MU0 if convert_si_to_hl else 1.0
    nr_g = U_padded.shape[1]

    r_safe = mx.maximum(mx.abs(r_cell), 1e-10)
    Bt_electrode = mx.array(
        MU_0 * current / (2.0 * math.pi) / divisor, dtype=mx.float32
    ) / r_safe
    Bt_electrode_2d = Bt_electrode[:, None] * mx.ones((1, U_padded.shape[2]), dtype=mx.float32)

    idx = mx.arange(nr_g)
    outer_ghost_mask = (idx >= ng + nr_phys) & (idx < ng + nr_phys + ng)
    outer_ghost_mask_2d = outer_ghost_mask[:, None]

    B2_old = U_padded[IBR] ** 2 + U_padded[IBZ] ** 2 + U_padded[IBT] ** 2
    Bt_new_outer = mx.where(outer_ghost_mask_2d, Bt_electrode_2d, U_padded[IBT])
    B2_new_outer = U_padded[IBR] ** 2 + U_padded[IBZ] ** 2 + Bt_new_outer ** 2
    dE_outer = 0.5 * (B2_new_outer - B2_old)

    E_updated = U_padded[IEN] + mx.where(outer_ghost_mask_2d, dE_outer, 0.0)

    p_mag_outer = 0.5 * B2_new_outer
    beta_floor = 1e-4
    p_min = beta_floor * mx.maximum(p_mag_outer, mx.array(P_FLOOR, dtype=mx.float32))
    E_floor = p_min / (_GAMMA_LOCAL - 1.0) + 0.5 * B2_new_outer
    E_updated = mx.where(outer_ghost_mask_2d, mx.maximum(E_updated, E_floor), E_updated)

    rho_updated = mx.where(outer_ghost_mask_2d, mx.maximum(U_padded[IDN], 1e-4), U_padded[IDN])

    interior_blend_mask = (idx >= ng + nr_phys - ng) & (idx < ng + nr_phys)
    interior_blend_mask_2d = interior_blend_mask[:, None]

    Bt_blended = mx.where(
        mx.abs(Bt_new_outer) > mx.abs(Bt_electrode_2d),
        Bt_new_outer,
        Bt_electrode_2d,
    )
    Bt_final = mx.where(interior_blend_mask_2d, Bt_blended, Bt_new_outer)

    B2_blend = U_padded[IBR] ** 2 + U_padded[IBZ] ** 2 + Bt_final ** 2
    dE_blend = 0.5 * (B2_blend - B2_old)
    E_final = E_updated + mx.where(interior_blend_mask_2d, dE_blend, 0.0)

    p_mag_blend = 0.5 * B2_blend
    p_min_b = beta_floor * mx.maximum(p_mag_blend, mx.array(P_FLOOR, dtype=mx.float32))
    E_floor_b = p_min_b / (_GAMMA_LOCAL - 1.0) + 0.5 * B2_blend
    E_final = mx.where(interior_blend_mask_2d, mx.maximum(E_final, E_floor_b), E_final)

    return mx.stack([
        rho_updated,
        U_padded[IMR],
        U_padded[IMZ],
        U_padded[IMT],
        E_final,
        U_padded[ISR],
        U_padded[IBR],
        U_padded[IBZ],
        Bt_final,
        U_padded[IEE],
    ], axis=0).astype(mx.float32)


def strip_ghost(U: Any, ng: int) -> Any:
    """Strip ghost cells from padded state, returning interior only."""
    return U[:, ng:-ng, :]
