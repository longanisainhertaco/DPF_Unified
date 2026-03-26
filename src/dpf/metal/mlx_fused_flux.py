"""Fused PLM reconstruction + HLL Riemann flux for MLX.

Provides a unified API for computing PLM-reconstructed HLL flux.
Currently delegates to separate PLM + HLL calls. A future Metal
kernel implementation will fuse both into a single GPU pass to
eliminate intermediate UL/UR array materialization (~2.6 MB savings
per RK stage on 128x256 grid).

References:
    van Leer B., JCP 23:276 (1977) — Monotonic reconstruction.
    Harten, Lax & van Leer, SIAM Rev. 25:35 (1983) — HLL solver.
"""

from __future__ import annotations

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def fused_plm_hll_flux(
    Q: mx.array,
    gamma: float,
    dim: int,
) -> mx.array:
    """Compute PLM-reconstructed HLL flux.

    Currently uses separate PLM reconstruction + HLL flux computation.
    A fused Metal kernel is planned for a future sprint.

    Args:
        Q: Conserved state, shape (NVAR, nr, nz).
        gamma: Adiabatic index.
        dim: Normal direction. 0=radial (axis 1), 1=axial (axis 2).

    Returns:
        Numerical flux F at interfaces.
        For dim=0: shape (NVAR, nr-1, nz).
        For dim=1: shape (NVAR, nr, nz-1).
    """
    from dpf.metal.mlx_reconstruction import plm_reconstruct
    from dpf.metal.mlx_riemann import _hll_flux

    UL, UR = plm_reconstruct(Q, dim, limiter="mc")
    return _hll_flux(UL, UR, gamma, dim)
