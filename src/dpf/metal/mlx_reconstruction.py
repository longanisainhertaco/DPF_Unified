"""WENO5-Z and PLM reconstruction for the MLX MHD solver.

Implements 5th-order WENO-Z (Borges et al. 2008) and 2nd-order PLM
reconstruction at cell interfaces. All operations are vectorized
mx.array operations suitable for mx.compile() fusion.

References:
    Shu C.-W., SIAM Rev. 51, 82-126 (2009) — FD point-value WENO5 formulas.
    Borges R. et al., JCP 227, 3191-3211 (2008) — WENO-Z nonlinear weights.
    Jiang G.-S. & Shu C.-W., JCP 126, 202-228 (1996) — Smoothness indicators.
    van Leer B., JCP 23, 276-299 (1977) — MC limiter.
"""

from __future__ import annotations

import mlx.core as mx

# ---------------------------------------------------------------------------
# Compile cache — populated lazily on first call.
# ---------------------------------------------------------------------------

_COMPILED: dict[str, object] = {}


def _compile_if_available(fn: object) -> object:
    """Wrap *fn* with mx.compile if MLX supports it, else return it unchanged."""
    try:
        return mx.compile(fn)  # type: ignore[attr-defined]
    except Exception:
        return fn


# ============================================================
# Internal helpers
# ============================================================


def _take(arr: mx.array, axis: int, start: int, length: int) -> mx.array:
    """Extract `length` elements along `axis` starting at `start`.

    Uses direct slice notation (zero-copy) rather than mx.take with an
    index array, eliminating per-call mx.array allocations.

    Args:
        arr: Source array.
        axis: Axis to slice along (0, 1, or 2).
        start: First index (inclusive).
        length: Number of elements to extract.

    Returns:
        Sliced array; shape is identical to `arr` except the selected
        axis has size `length`.
    """
    end = start + length
    if axis == 0:
        return arr[start:end]
    if axis == 1:
        return arr[:, start:end]
    if axis == 2:
        return arr[:, :, start:end]
    return arr[:, :, :, start:end]


# ============================================================
# Slope limiters (PLM)
# ============================================================


def _minmod(a: mx.array, b: mx.array) -> mx.array:
    """Minmod slope limiter.

    Returns sign(a) * min(|a|, |b|) when sign(a) == sign(b), else 0.

    Args:
        a: Left slope.
        b: Right slope.

    Returns:
        Limited slope, same shape as inputs.
    """
    same_sign = (a * b) > 0.0
    limited = mx.sign(a) * mx.minimum(mx.abs(a), mx.abs(b))
    return mx.where(same_sign, limited, mx.zeros_like(a))


def _mc_limit(a: mx.array, b: mx.array) -> mx.array:
    """Monotonized Central (MC, van Leer) slope limiter.

    Defined as the minmod of (2a, (a+b)/2, 2b) — equivalently:
        mc(a, b) = sign(a) * min(2|a|, 2|b|, |a+b|/2)   if sign(a) == sign(b)
                 = 0                                       otherwise

    This is the standard "minmod of three" form (LeVeque 2002, §6.10).

    Args:
        a: Left slope.
        b: Right slope.

    Returns:
        Limited slope, same shape as inputs.
    """
    same_sign = (a * b) > 0.0
    limited = mx.sign(a) * mx.minimum(
        mx.minimum(2.0 * mx.abs(a), 2.0 * mx.abs(b)),
        0.5 * mx.abs(a + b),
    )
    return mx.where(same_sign, limited, mx.zeros_like(a))


# ============================================================
# PLM reconstruction
# ============================================================


def plm_reconstruct(
    Q: mx.array,
    dim: int,
    limiter: str = "mc",
) -> tuple[mx.array, mx.array]:
    """PLM reconstruction with MC or minmod limiter.

    For each interior cell i, computes a limited slope and extrapolates
    to the left/right faces::

        QL[i+1/2] = Q[i]   + 0.5 * slope[i]
        QR[i+1/2] = Q[i+1] - 0.5 * slope[i+1]

    Boundary cells use zero slope (constant extrapolation).

    Args:
        Q: Conserved state, shape (NVAR, nr, nz).
        dim: Reconstruction dimension. 0=radial (axis 1), 1=axial (axis 2).
        limiter: "mc" or "minmod".

    Returns:
        (QL, QR) at interfaces.
        For dim=0: shapes (NVAR, nr-1, nz).
        For dim=1: shapes (NVAR, nr, nz-1).
    """
    axis = dim + 1
    n = Q.shape[axis]

    if n < 2:
        raise ValueError(
            f"PLM requires at least 2 cells along dim={dim}, got {n}"
        )

    limit_fn = _mc_limit if limiter == "mc" else _minmod

    # Forward differences along axis: shape has n-1 entries along axis
    q_lo = _take(Q, axis, 0, n - 1)
    q_hi = _take(Q, axis, 1, n - 1)
    fwd = q_hi - q_lo  # differences between adjacent cells

    if n >= 3:
        # Interior cells [1 .. n-2] have both left and right differences
        fwd_l = _take(fwd, axis, 0, n - 2)  # diff[i-1] for cells 1..n-2
        fwd_r = _take(fwd, axis, 1, n - 2)  # diff[i]   for cells 1..n-2
        slope_interior = limit_fn(fwd_l, fwd_r)

        # Build zero pad shape (same as Q but size-1 on axis)
        pad_shape = list(Q.shape)
        pad_shape[axis] = 1
        zero_pad = mx.zeros(pad_shape, dtype=Q.dtype)

        # Full slope: [0, slope_interior, 0] along axis
        slope = mx.concatenate([zero_pad, slope_interior, zero_pad], axis=axis)
    else:
        slope = mx.zeros_like(Q)

    # Extrapolate to interfaces
    QL = _take(Q, axis, 0, n - 1) + 0.5 * _take(slope, axis, 0, n - 1)
    QR = _take(Q, axis, 1, n - 1) - 0.5 * _take(slope, axis, 1, n - 1)

    return QL, QR


# ============================================================
# WENO5-Z reconstruction kernel
# ============================================================


def _weno5z_left_biased(
    qm2: mx.array,
    qm1: mx.array,
    q0: mx.array,
    qp1: mx.array,
    qp2: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Left-biased WENO5-Z reconstruction at interface i+1/2.

    Uses the finite-difference point-value candidate polynomials (Shu 2009,
    SIAM Rev. 51, Sec. 2.2) with WENO-Z (Borges et al. 2008) nonlinear
    weights. Ideal weights are d0=1/16, d1=10/16, d2=5/16 (FD point-value).

    These FD formulas give 5th-order accuracy on point values. The FV
    Jiang-Shu (1996) formulas (coefficients (2,-7,11)/6, ideal weights
    0.1/0.6/0.3) only give 5th order on cell averages; on point values they
    degrade to 2nd order (CLAUDE.md lesson #51).

    Note on eps: Borges (2008) suggests eps=1e-40, but float32 underflows
    below ~1e-38. For float32 stability, eps=1e-6 is used (matching the
    PyTorch reference implementation in _riemann_reconstruction.py).
    This does not degrade order of accuracy on smooth data.

    Args:
        qm2: Q[i-2], shape (NVAR, ...).
        qm1: Q[i-1], shape (NVAR, ...).
        q0:  Q[i],   shape (NVAR, ...).
        qp1: Q[i+1], shape (NVAR, ...).
        qp2: Q[i+2], shape (NVAR, ...).
        eps: Denominator floor to prevent division by zero.

    Returns:
        Reconstructed value at i+1/2, same shape as inputs.
    """
    # Candidate polynomials (Shu 2009 FD point-value, Lagrange interpolation at u=+0.5)
    # S0 = {i-2, i-1, i}: coefficients (3/8, -10/8, 15/8)
    p0 = (3.0 * qm2 - 10.0 * qm1 + 15.0 * q0) / 8.0
    # S1 = {i-1, i, i+1}: coefficients (-1/8, 6/8, 3/8)
    p1 = (-qm1 + 6.0 * q0 + 3.0 * qp1) / 8.0
    # S2 = {i, i+1, i+2}: coefficients (3/8, 6/8, -1/8)
    p2 = (3.0 * q0 + 6.0 * qp1 - qp2) / 8.0

    # Ideal weights (FD point-value, Shu SIAM Rev. 2009 Sec. 2.2)
    d0 = 1.0 / 16.0
    d1 = 10.0 / 16.0
    d2 = 5.0 / 16.0

    # Smoothness indicators (Jiang & Shu 1996, Eq. 2.62)
    beta0 = (
        (13.0 / 12.0) * (qm2 - 2.0 * qm1 + q0) ** 2
        + 0.25 * (qm2 - 4.0 * qm1 + 3.0 * q0) ** 2
    )
    beta1 = (
        (13.0 / 12.0) * (qm1 - 2.0 * q0 + qp1) ** 2
        + 0.25 * (qm1 - qp1) ** 2
    )
    beta2 = (
        (13.0 / 12.0) * (q0 - 2.0 * qp1 + qp2) ** 2
        + 0.25 * (3.0 * q0 - 4.0 * qp1 + qp2) ** 2
    )

    # WENO-Z global smoothness indicator (Borges et al. 2008, Eq. 25)
    tau5 = mx.abs(beta0 - beta2)

    # WENO-Z nonlinear weights: alpha_k = d_k * (1 + (tau5/(eps+beta_k))^2)
    a0 = d0 * (1.0 + (tau5 / (eps + beta0)) ** 2)
    a1 = d1 * (1.0 + (tau5 / (eps + beta1)) ** 2)
    a2 = d2 * (1.0 + (tau5 / (eps + beta2)) ** 2)

    a_sum = mx.maximum(a0 + a1 + a2, 1e-30)

    return (a0 / a_sum) * p0 + (a1 / a_sum) * p1 + (a2 / a_sum) * p2


def weno5z_reconstruct(
    Q: mx.array,
    dim: int,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """WENO5-Z reconstruction (Borges et al. 2008).

    5th-order reconstruction at cell interfaces using WENO-Z nonlinear
    weights. Requires at least 6 cells in the reconstruction dimension;
    falls back to PLM with MC limiter if fewer.

    Both left-biased (QL) and right-biased (QR) reconstructions require a
    full 5-point stencil. For them to cover the same interfaces, n ≥ 6 is
    required and only n-5 interior interfaces are produced:

        Interface j (j=0..n-6) sits between cells j+2 and j+3.
        QL stencil: cells {j, j+1, j+2, j+3, j+4}
        QR stencil: cells {j+5, j+4, j+3, j+2, j+1}  (mirrored, centered on j+3)

    Args:
        Q: Conserved state, shape (NVAR, nr, nz).
        dim: Reconstruction dimension. 0=radial (axis 1), 1=axial (axis 2).
        eps: Small number for WENO-Z weight denominator (default 1e-6).

    Returns:
        (QL, QR) at interfaces.
        For dim=0: shapes (NVAR, nr-5, nz) — 5 cells consumed overall.
        For dim=1: shapes (NVAR, nr, nz-5).
    """
    axis = dim + 1
    n = Q.shape[axis]

    if n < 6:
        return plm_reconstruct(Q, dim=dim, limiter="mc")

    n_iface = n - 5  # interfaces where both QL and QR have full stencils

    # Left-biased: interface j is right face of cell j+2.
    # Stencil: {j, j+1, j+2, j+3, j+4}
    qm2 = _take(Q, axis, 0, n_iface)
    qm1 = _take(Q, axis, 1, n_iface)
    q0 = _take(Q, axis, 2, n_iface)
    qp1 = _take(Q, axis, 3, n_iface)
    qp2 = _take(Q, axis, 4, n_iface)

    if "weno5z_left_biased" not in _COMPILED:
        _COMPILED["weno5z_left_biased"] = _compile_if_available(_weno5z_left_biased)
    _kernel = _COMPILED["weno5z_left_biased"]

    QL = _kernel(qm2, qm1, q0, qp1, qp2, eps)  # type: ignore[operator]

    # Right-biased: same interface j, reconstructed from the right cell j+3.
    # Mirrored stencil: {j+5, j+4, j+3, j+2, j+1}
    rm2 = _take(Q, axis, 5, n_iface)
    rm1 = _take(Q, axis, 4, n_iface)
    r0 = _take(Q, axis, 3, n_iface)
    rp1 = _take(Q, axis, 2, n_iface)
    rp2 = _take(Q, axis, 1, n_iface)

    QR = _kernel(rm2, rm1, r0, rp1, rp2, eps)  # type: ignore[operator]

    return QL, QR


# ============================================================
# Dispatch
# ============================================================


def reconstruct(
    Q: mx.array,
    dim: int,
    method: str = "weno5z",
    **kwargs,
) -> tuple[mx.array, mx.array]:
    """Dispatch to PLM or WENO5-Z reconstruction.

    Args:
        Q: Conserved state, shape (NVAR, nr, nz).
        dim: Reconstruction dimension. 0=radial, 1=axial.
        method: "weno5z" or "plm".
        **kwargs: Forwarded to the chosen function.
            PLM accepts: limiter ("mc" or "minmod").
            WENO5-Z accepts: eps (float).

    Returns:
        (QL, QR) at interfaces.
    """
    if method == "weno5z":
        return weno5z_reconstruct(Q, dim=dim, **kwargs)
    if method == "plm":
        return plm_reconstruct(Q, dim=dim, **kwargs)
    raise ValueError(f"Unknown reconstruction method: {method!r}. Choose 'weno5z' or 'plm'.")
