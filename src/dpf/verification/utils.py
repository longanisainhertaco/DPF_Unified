"""Shared utilities for DPF verification tests."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def estimate_convergence_order(
    resolutions: list[int],
    errors: list[float],
) -> float:
    """Estimate the order of convergence from (resolution, error) pairs.

    Uses a least-squares fit in log-log space:
    ``log(error) ~ -order * log(N) + const``.

    This is a least-squares analogue of the classical Grid Convergence
    Index (GCI) ratio ``log(e1/e2) / log(r)`` for two refinements; the
    GCI itself is defined in Roache, P.J. (1994) "Perspective: A Method
    for Uniform Reporting of Grid Refinement Studies", J. Fluids Eng.
    116:405-413. [UNVERIFIED-CITATION: Roache 1994 PDF not on disk; the
    formula here is the textbook log-log slope, which is the form most
    code-V&V references quote.]

    Returns ``NaN`` (with a logged warning) when the order cannot be
    estimated — i.e. fewer than two usable ``(finite, positive err,
    positive N)`` points remain after filtering.  A previous version
    returned ``0.0`` in these cases, which was indistinguishable from
    a legitimately measured zeroth-order result and allowed failing
    convergence studies to silently pass.

    Args:
        resolutions: Grid sizes (N values).
        errors: Corresponding error norms.

    Returns:
        Estimated convergence order (positive means error decreases
        with increasing resolution).  ``float('nan')`` when the
        estimate is not well-defined.
    """
    if len(resolutions) < 2:
        logger.warning(
            "estimate_convergence_order: need >= 2 resolutions, got %d; "
            "returning NaN",
            len(resolutions),
        )
        return float("nan")

    # Filter out zero / negative / NaN errors
    log_N: list[float] = []
    log_e: list[float] = []
    for N, err in zip(resolutions, errors, strict=False):
        if np.isfinite(err) and err > 0 and N > 0:
            log_N.append(np.log(float(N)))
            log_e.append(np.log(err))

    if len(log_N) < 2:
        logger.warning(
            "estimate_convergence_order: only %d usable (N, err) pairs "
            "after filtering non-finite / non-positive entries "
            "(resolutions=%s, errors=%s); returning NaN",
            len(log_N), resolutions, errors,
        )
        return float("nan")

    # Least-squares: log_e = slope * log_N + intercept
    log_N_arr = np.array(log_N)
    log_e_arr = np.array(log_e)
    A = np.vstack([log_N_arr, np.ones(len(log_N_arr))]).T
    result = np.linalg.lstsq(A, log_e_arr, rcond=None)
    slope = result[0][0]

    # Convergence order is the negative of the slope:
    #   error ~ N^{-order}  =>  log(e) = -order * log(N) + const
    return float(-slope)
