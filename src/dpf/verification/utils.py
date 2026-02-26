"""Shared utilities for DPF verification tests."""

from __future__ import annotations

import numpy as np


def estimate_convergence_order(
    resolutions: list[int],
    errors: list[float],
) -> float:
    """Estimate the order of convergence from (resolution, error) pairs.

    Uses a least-squares fit in log-log space:
    ``log(error) ~ -order * log(N) + const``.

    If any error is zero or negative, or if there are fewer than two
    usable data points, returns ``0.0``.

    Args:
        resolutions: Grid sizes (N values).
        errors: Corresponding error norms.

    Returns:
        Estimated convergence order (positive means error decreases with
        increasing resolution).
    """
    if len(resolutions) < 2:
        return 0.0

    # Filter out zero / negative / NaN errors
    log_N: list[float] = []
    log_e: list[float] = []
    for N, err in zip(resolutions, errors, strict=False):
        if np.isfinite(err) and err > 0 and N > 0:
            log_N.append(np.log(float(N)))
            log_e.append(np.log(err))

    if len(log_N) < 2:
        return 0.0

    # Least-squares: log_e = slope * log_N + intercept
    log_N_arr = np.array(log_N)
    log_e_arr = np.array(log_e)
    A = np.vstack([log_N_arr, np.ones(len(log_N_arr))]).T
    result = np.linalg.lstsq(A, log_e_arr, rcond=None)
    slope = result[0][0]

    # Convergence order is the negative of the slope:
    #   error ~ N^{-order}  =>  log(e) = -order * log(N) + const
    return float(-slope)
