"""Electrode boundary conditions for the MLX cylindrical MHD solver.

Two BC modes:

1. **Inlet BC (Sun 2025)**: B_theta = mu0*I/(2*pi*r) at z=0 (insulator end).
   The current enters through the inlet boundary. B_theta only exists at
   z-positions where current has flowed — the MHD dynamics determine how
   far B propagates axially. Cathode: conducting wall (dB/dn = 0).

   Sun et al. (2025), Acta Physica Sinica 74:115201, Eq. (19), Fig. 2.
   PDF: references/papers/core-dpf/2025_Theoretical_and_numerical_studies_
   on_motion_process_of_dense_plasma_focus.pdf

2. **Cathode BC (legacy)**: B_theta = mu0*I/(2*pi*r) at outer-r ghost cells.
   Injects B at the cathode for all z simultaneously. Only valid when
   combined with z-dependent masking or vacuum resistivity.

Default: inlet BC (mode 1). The cathode BC is preserved for testing.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from dpf.metal.constants import MU_0, P_FLOOR, SQRT_MU0
from dpf.metal.mlx_device import require_mlx

logger = logging.getLogger(__name__)


def inlet_bt_bc_mlx(
    U: Any,
    r_cell: Any,
    current: float,
    ng_z: int,
    convert_si_to_hl: bool = True,
) -> Any:
    """Apply B_theta BC at inlet boundary (z=0) — Sun 2025 method.

    Sets B_theta = mu0*I/(2*pi*r) [/ sqrt(mu0) if HL] at the first ng_z
    z-cells (ghost zone at z=0, the insulator/inlet end).

    The current enters through the insulator base. B_theta at the inlet
    represents the total azimuthal field from Ampere's law at that plane.
    The MHD dynamics (Alfven wave propagation along z) determine how far
    B_theta extends axially — no sheath detection needed.

    Cathode (outer r): conducting wall, dB/dn = 0 (zero-gradient, handled
    by the existing ghost cell padding).

    References
    ----------
    Sun et al. (2025), Acta Physica Sinica 74:115201, Eq. (19):
        B = mu0*I/(2*pi*r) at inlet boundary AB.
    Auluck (2021), Phys. Plasmas 28:030703, Eq. (13):
        dPhi/dt coupling is exact for 2D axisymmetric (B purely azimuthal).

    Parameters
    ----------
    U : mx.array
        Conserved state (NVAR, nr, nz), NOT padded with r-ghosts.
    r_cell : mx.array
        Cell-center radii (nr,).
    current : float
        Circuit current [A].
    ng_z : int
        Number of ghost cells at z=0 to set (typically 2-3).
    convert_si_to_hl : bool
        If True, divide B by sqrt(mu0) for Heaviside-Lorentz units.

    Returns
    -------
    U : mx.array
        State with B_theta set at z=0:ng_z cells. Energy updated for consistency.
    """
    mx = require_mlx()
    from dpf.metal.constants import IBR, IBT, IBZ, IEN

    _GAMMA = 5.0 / 3.0
    divisor = SQRT_MU0 if convert_si_to_hl else 1.0
    nr = U.shape[1]

    r_safe = mx.maximum(mx.abs(r_cell), 1e-10)
    # B_theta = mu0 * I / (2*pi*r), converted to HL if needed
    Bt_inlet = mx.array(
        MU_0 * current / (2.0 * math.pi) / divisor, dtype=mx.float32
    ) / r_safe  # shape (nr,)

    # Smooth ramp over first ng_z cells to avoid NaN from float32
    # catastrophic cancellation at the B=0 → B=1500 HL discontinuity.
    # The ramp ensures the Riemann solver sees a gradual transition.
    # Same technique as the cathode BC blend (lines 71-116 of legacy BC).
    nz = U.shape[2]
    ng_z_safe = min(ng_z, nz)

    # Build smooth ramp: 1.0 at z=0, linear decay to 0 at z=ng_z
    z_ramp = mx.zeros((nz,), dtype=mx.float32)
    for iz in range(ng_z_safe):
        weight = 1.0 - float(iz) / max(ng_z_safe, 1)
        z_ramp = z_ramp.at[iz].add(weight)

    Bt_target = Bt_inlet[:, None] * z_ramp[None, :]  # (nr, nz)

    # Blend: use max(existing, target) so B_theta grows monotonically
    # from the inlet without overwriting MHD-evolved B ahead of the ramp.
    Bt_old = U[IBT]
    Bt_new = mx.where(
        mx.abs(Bt_target) > mx.abs(Bt_old),
        Bt_target,
        Bt_old,
    )

    # Update total energy for B_theta change (energy consistency)
    B2_old = U[IBR] ** 2 + U[IBZ] ** 2 + Bt_old ** 2
    B2_new = U[IBR] ** 2 + U[IBZ] ** 2 + Bt_new ** 2
    dE = 0.5 * (B2_new - B2_old)
    E_new = mx.maximum(U[IEN] + dE, P_FLOOR)

    # Reconstruct state
    rows = [U[i] for i in range(U.shape[0])]
    rows[IBT] = Bt_new
    rows[IEN] = E_new

    return mx.stack(rows, axis=0).astype(mx.float32)


def electrode_bt_fixup_mlx(
    U_padded: Any,
    r_cell: Any,
    current: float,
    ng: int,
    nr_phys: int,
    convert_si_to_hl: bool = True,
    z_sheath_frac: float = 1.0,
) -> Any:
    """Legacy cathode BC: B_theta at outer-r ghost cells (kept for testing).

    See inlet_bt_bc_mlx for the physically correct Sun 2025 method.
    """
    mx = require_mlx()
    from dpf.metal.constants import IBR, IBT, IBZ, IDN, IEE, IEN, IMR, IMT, IMZ, ISR

    _GAMMA_LOCAL = 5.0 / 3.0
    divisor = SQRT_MU0 if convert_si_to_hl else 1.0
    nr_g = U_padded.shape[1]
    nz_g = U_padded.shape[2]

    r_safe = mx.maximum(mx.abs(r_cell), 1e-10)
    Bt_electrode = mx.array(
        MU_0 * current / (2.0 * math.pi) / divisor, dtype=mx.float32
    ) / r_safe

    iz_sheath = max(1, int(z_sheath_frac * nz_g))
    z_mask = mx.zeros((nz_g,), dtype=mx.float32)
    z_mask = z_mask.at[:iz_sheath].add(1.0)
    Bt_electrode_2d = Bt_electrode[:, None] * z_mask[None, :]

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

    # No interior blend — ghost cells only. Interior cells evolve freely
    # via MHD, allowing the plasma compression to modify B_theta structure.
    # The old interior blend (max(existing, electrode)) prevented the sheath
    # from reducing B below the vacuum coaxial value, making Phi constant
    # and U_PF insensitive to plasma motion.
    Bt_final = Bt_new_outer
    E_final = E_updated

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
