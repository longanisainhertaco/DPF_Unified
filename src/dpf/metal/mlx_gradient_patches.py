"""Pure-MLX replacements for numpy call sites in the MLX solver.

100% mlx.core, zero numpy, compatible with mx.grad.

References:
    Spitzer & Härm, Phys. Rev. 89, 977 (1953).
    Gericke et al., Phys. Plasmas 9, 818 (2002) — ln_Lambda.
    Thomas L.H. (1949) — tridiagonal algorithm.
"""

from __future__ import annotations

import math

import mlx.core as mx

# ── Constants ────────────────────────────────────────────────────────────────

_K_B: float = 1.380649e-23
_EV: float = 1.602176634e-19
_MU_0: float = 4.0 * math.pi * 1e-7
_TWO_PI: float = 2.0 * math.pi

IBT: int = 8  # Azimuthal B-field index in conserved state vector

# ── 1. Ghost Cell Padding ────────────────────────────────────────────────────


def ghost_pad_mlx(U: mx.array, ng: int, bc: str = "outflow") -> mx.array:
    """Pad conserved state with ng ghost cells on each side of axis 1.

    Inner (axis): reflecting — sign-flip IMR=1, IMT=3, IBR=6, IBT=8.
    Outer: outflow — zero-gradient copy, IMR and IBR zeroed.

    Args:
        U:  State array, shape (NVAR, nr, nz).
        ng: Ghost cell layers per side.
        bc: Outer BC — "outflow" (default) or pass-through for other callers.

    Returns:
        Array shape (NVAR, nr + 2*ng, nz).
    """
    nvar = U.shape[0]
    ndim = U.ndim

    # Inner ghosts: reversed copy of first ng interior cells
    inner_ghosts = U[:, :ng, ...][:, ::-1, ...]

    # Build sign vector for reflecting components
    signs = mx.ones((nvar,), dtype=U.dtype)
    for v in (1, 3, 6, 8):
        if v < nvar:
            signs = mx.concatenate([
                signs[:v],
                mx.array([-1.0], dtype=U.dtype),
                signs[v + 1:],
            ])
    signs = signs.reshape([nvar] + [1] * (ndim - 1))
    inner_ghosts = inner_ghosts * signs

    # Outer ghosts: zero-gradient from last interior cell
    outer_ghosts = mx.broadcast_to(U[:, -1:, ...], (nvar, ng) + U.shape[2:])

    if bc == "outflow":
        def _zero_var(arr: mx.array, v: int) -> mx.array:
            if v >= nvar:
                return arr
            z = mx.zeros((1, ng) + arr.shape[2:], dtype=arr.dtype)
            return mx.concatenate([arr[:v], z, arr[v + 1:]], axis=0)

        outer_ghosts = _zero_var(outer_ghosts, 1)  # IMR
        outer_ghosts = _zero_var(outer_ghosts, 6)  # IBR

    return mx.concatenate([inner_ghosts, U, outer_ghosts], axis=1)


# ── 2. Thomas Tridiagonal Solver ─────────────────────────────────────────────


def thomas_solve_mlx(
    a: mx.array,
    b: mx.array,
    c: mx.array,
    d: mx.array,
) -> mx.array:
    """Solve tridiagonal system Ax = d (Thomas algorithm, sequential mx ops).

    Args:
        a: Lower diagonal, shape (n,). a[0] unused.
        b: Main diagonal,  shape (n,).
        c: Upper diagonal, shape (n,). c[n-1] unused.
        d: Right-hand side, shape (n,).

    Returns:
        x: Solution vector, shape (n,).
    """
    n = b.shape[0]
    if n == 1:
        return mx.reshape(d[0] / b[0], (1,))

    zero = mx.array(0.0, dtype=b.dtype)
    c_prime: list[mx.array] = [zero] * n
    d_prime: list[mx.array] = [zero] * n

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else zero
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x: list[mx.array] = [zero] * n
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return mx.stack(x)


# ── 3. Spitzer Resistivity ───────────────────────────────────────────────────


def spitzer_eta_mlx(
    rho: mx.array,
    pressure: mx.array,
    ion_mass: float,
    Z_eff: float = 1.0,
) -> mx.array:
    """Compute Spitzer resistivity [Ohm·m] from density and pressure.

    Uses PARALLEL Spitzer resistivity (5.2e-5), see NRL Formulary 2019 p.37.
    Note: mlx_transport.py uses PERPENDICULAR (1.03e-4). These differ by 1.96x.

    T_eV = p * m_i / (2 * rho * eV)  (ionised plasma, Z=1)
    ln_Lambda from NRL Formulary 2019, p.34.
    eta = 5.2e-5 * Z_eff * ln_Lambda / T_eV^1.5

    Args:
        rho:      Mass density [kg/m^3].
        pressure: Thermal pressure [Pa].
        ion_mass: Ion mass [kg].
        Z_eff:    Effective charge (default 1.0).

    Returns:
        eta: Resistivity [Ohm·m], same shape as rho.
    """
    rho_safe = mx.maximum(rho, mx.array(1e-30, dtype=rho.dtype))

    T_eV = mx.maximum(
        pressure * float(ion_mass) / (2.0 * rho_safe * float(_EV)),
        mx.array(0.1, dtype=rho.dtype),
    )

    # NRL Formulary 2019, p.34, electron-ion Coulomb logarithm:
    #   T_e < 10*Z^2 eV: lnL = 23 - 0.5*ln(n_e_cgs) - ln(Z) + 1.5*ln(T_eV)
    #   T_e > 10*Z^2 eV: lnL = 24 - 0.5*ln(n_e_cgs) + 1.0*ln(T_eV)
    # SI conversion (n_e in m^-3): constant += 0.5*ln(1e6) ≈ 6.9 → 29.9 / 30.9
    # Low-T regime includes -ln(Z); high-T does not.
    n_e = rho_safe / float(ion_mass)
    ln_Z = mx.log(mx.array(max(Z_eff, 1.0), dtype=rho.dtype))
    lnL_low = 29.9 - 0.5 * mx.log(n_e) - ln_Z + 1.5 * mx.log(T_eV)
    lnL_high = 30.9 - 0.5 * mx.log(n_e) + 1.0 * mx.log(T_eV)
    threshold = 10.0 * Z_eff * Z_eff
    ln_lambda = mx.where(T_eV > threshold, lnL_high, lnL_low)
    ln_lambda = mx.maximum(ln_lambda, mx.array(2.0, dtype=rho.dtype))
    ln_lambda = mx.minimum(ln_lambda, mx.array(20.0, dtype=rho.dtype))

    return 5.2e-5 * float(Z_eff) * ln_lambda / (T_eV ** 1.5)


# ── 4. Electrode B_theta Injection ──────────────────────────────────────────


def electrode_bt_mlx(
    U: mx.array,
    current: float,
    r_cells: mx.array,
    fc: float,
    mu_0: float = _MU_0,
) -> mx.array:
    """Replace IBT row in U with electrode B_theta = mu_0*fc*I/(2*pi*r).

    Applies to the entire padded radial axis. Does NOT update total energy —
    use the full ghost_pad pipeline for energy-consistent injection.

    Args:
        U:        State array (NVAR, nr_padded, nz), float32.
        current:  Circuit current [A].
        r_cells:  Cell-centre radii, shape (nr_padded,).
        fc:       Current fraction coupling coefficient.
        mu_0:     Permeability of free space [H/m].

    Returns:
        U with IBT slice replaced.
    """
    nr_g = U.shape[1]
    nz = U.shape[2]

    r_safe = mx.maximum(r_cells, mx.array(1e-10, dtype=r_cells.dtype))
    Bt_field = float(mu_0 * fc * current) / (float(_TWO_PI) * r_safe)
    Bt_row = mx.reshape(mx.broadcast_to(Bt_field[:, None], (nr_g, nz)), (1, nr_g, nz))

    return mx.concatenate([U[:IBT], Bt_row, U[IBT + 1:]], axis=0)
