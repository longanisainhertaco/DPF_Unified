"""RKL2 Super Time-Stepping for the MLX solver.

Provides a generic RKL2 integrator that advances any parabolic operator
(resistive diffusion, thermal conduction, viscosity, Hall MHD) using
an s-stage explicit method whose stability region extends ~0.25*s^2
times the standard explicit CFL.

The coefficient computation is delegated to the Python-engine module
(dpf.fluid.super_time_step.rkl2_coefficients) which uses exact
Chebyshev polynomial recursion. The step loop runs on MLX arrays.

References:
    Meyer, Balsara & Aslam, JCP 231:2963 (2012) — RKL2 method.
    Vaidya et al., MNRAS 472:3147 (2017) — PLUTO implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import numpy as np

from dpf.fluid.super_time_step import rkl2_coefficients


def rkl2_step_mlx(
    U: mx.array,
    rhs_fn: Callable[[mx.array], mx.array],
    dt: float,
    s_stages: int = 8,
) -> mx.array:
    """Advance U by one RKL2 super-step using a provided RHS function.

    This is the generic MLX RKL2 integrator. The caller provides the
    parabolic RHS function L(U) → dU/dt. The integrator applies the
    s-stage RKL2 recursion entirely in MLX (GPU).

    The recursion (Meyer et al. 2012, eq. 2.15):
        Y_0 = U
        Y_1 = Y_0 + mu_tilde_1 * dt * L(Y_0)
        Y_j = mu_j * Y_{j-1} + nu_j * Y_{j-2} + (1-mu_j-nu_j) * Y_0
              + mu_tilde_j * dt * L(Y_{j-1}) + gamma_tilde_j * dt * L(Y_0)
        U_new = Y_s

    Args:
        U: State array (any shape, MLX).
        rhs_fn: Function L(U) → dU/dt for the parabolic operator.
            Must accept and return mx.array of the same shape as U.
        dt: Super-timestep [s]. Can be up to ~0.25*s^2 * dt_explicit.
        s_stages: Number of RKL stages (default 8). Must be >= 2.

    Returns:
        Updated U after one RKL2 super-step.
    """
    mu_np, nu_np, mu_t_np, gamma_t_np = rkl2_coefficients(s_stages)

    Y0 = U
    L0 = rhs_fn(Y0)

    # Stage 1: forward Euler
    Y_prev2 = Y0
    Y_prev1 = Y0 + float(mu_t_np[1]) * dt * L0

    # Stages 2..s
    for j in range(2, s_stages + 1):
        L_prev1 = rhs_fn(Y_prev1)
        Y_curr = (
            float(mu_np[j]) * Y_prev1
            + float(nu_np[j]) * Y_prev2
            + (1.0 - float(mu_np[j]) - float(nu_np[j])) * Y0
            + float(mu_t_np[j]) * dt * L_prev1
            + float(gamma_t_np[j]) * dt * L0
        )
        Y_prev2 = Y_prev1
        Y_prev1 = Y_curr

    mx.eval(Y_prev1)
    return Y_prev1


def compute_sts_stages(
    dt_mhd: float,
    dt_parabolic: float,
    max_stages: int = 20,
) -> int:
    """Compute the number of RKL2 stages needed to cover dt_mhd.

    The RKL2 stability limit is ~0.25 * s^2 * dt_parabolic.
    We solve: 0.25 * s^2 * dt_parabolic >= dt_mhd
    => s >= sqrt(dt_mhd / (0.25 * dt_parabolic))

    Args:
        dt_mhd: MHD (hyperbolic) timestep [s].
        dt_parabolic: Explicit parabolic CFL limit [s].
        max_stages: Maximum allowed stages (default 20).

    Returns:
        Number of RKL2 stages (clamped to [2, max_stages]).
    """
    if dt_parabolic <= 0 or dt_mhd <= 0:
        return 2
    ratio = dt_mhd / (0.25 * dt_parabolic)
    s = int(np.ceil(np.sqrt(max(ratio, 1.0))))
    return max(2, min(s, max_stages))
