"""WALRUS surrogate validation against analytical benchmarks.

Generates physics trajectories from Bennett equilibrium and magnetized Noh
exact solutions, then validates surrogate predictions against them.  This
bridges the QSSS analytical benchmarks in ``dpf.validation`` with the
WALRUS surrogate validation pipeline in ``dpf.ai.surrogate``.

Usage::

    from dpf.ai.benchmark_validation import (
        create_bennett_trajectory,
        create_noh_trajectory,
        validate_surrogate_against_bennett,
        validate_surrogate_against_noh,
    )

    trajectory = create_bennett_trajectory(n_steps=10)
    report = validate_surrogate_against_bennett(surrogate, n_steps=10)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dpf.validation.bennett_equilibrium import create_bennett_state
from dpf.validation.magnetized_noh import create_noh_state

# Deuterium ion mass [kg]
_M_DEUTERIUM = 3.344e-27


def create_bennett_trajectory(
    n_steps: int = 10,
    nr: int = 32,
    nz: int = 8,
    r_max: float = 0.05,
    dz: float = 0.01,
    n_0: float = 1e23,
    a: float = 0.005,
    Te: float = 1e7,
    Ti: float = 1e7,
    m_ion: float = _M_DEUTERIUM,
) -> list[dict[str, np.ndarray]]:
    """Generate a trajectory of Bennett equilibrium states.

    Since the Bennett profile is a stationary equilibrium, every state in the
    trajectory is identical.  A perfect surrogate should reproduce each step
    exactly.

    Parameters
    ----------
    n_steps : int
        Number of trajectory steps (all identical).
    nr, nz : int
        Grid dimensions.
    r_max, dz : float
        Domain size [m].
    n_0 : float
        On-axis number density [m^-3].
    a : float
        Bennett radius [m].
    Te, Ti : float
        Electron and ion temperatures [K].
    m_ion : float
        Ion mass [kg].

    Returns
    -------
    list[dict[str, np.ndarray]]
        Trajectory of ``n_steps`` identical DPF state dicts.
    """
    state, _I_total, _r_centers = create_bennett_state(
        nr=nr, nz=nz, r_max=r_max, dz=dz,
        n_0=n_0, a=a, Te=Te, Ti=Ti, m_ion=m_ion,
    )
    # Deep-copy the state for each step to avoid aliasing
    return [{k: v.copy() for k, v in state.items()} for _ in range(n_steps)]


def create_noh_trajectory(
    n_steps: int = 10,
    nr: int = 32,
    nz: int = 8,
    r_max: float = 2.0,
    rho_0: float = 1.0,
    V_0: float = 1.0,
    B_0: float = 0.0,
    gamma: float = 5.0 / 3.0,
    t_start: float = 0.5,
    t_end: float = 2.0,
) -> list[dict[str, np.ndarray]]:
    """Generate a trajectory of magnetized Noh states at different times.

    The magnetized Noh solution has an outward-propagating shock whose
    position scales as ``r_shock = V_s * t``.  Each state in the trajectory
    corresponds to a different time, so the profiles evolve.

    Parameters
    ----------
    n_steps : int
        Number of trajectory steps.
    nr, nz : int
        Grid dimensions.
    r_max : float
        Outer radial boundary [m].
    rho_0, V_0, B_0, gamma : float
        Problem parameters.
    t_start, t_end : float
        Time range [s] (must be positive).

    Returns
    -------
    list[dict[str, np.ndarray]]
        Trajectory of ``n_steps`` DPF state dicts at uniformly spaced times.
    """
    times = np.linspace(t_start, t_end, n_steps)
    trajectory: list[dict[str, np.ndarray]] = []
    for t in times:
        state, _info = create_noh_state(
            nr=nr, nz=nz, r_max=r_max, t=float(t),
            rho_0=rho_0, V_0=V_0, B_0=B_0, gamma=gamma,
        )
        trajectory.append(state)
    return trajectory


def validate_surrogate_against_bennett(
    surrogate: Any,
    n_steps: int = 10,
    fields: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate surrogate predictions on a stationary Bennett equilibrium.

    A perfect surrogate should return L2 = 0 on this benchmark because
    the equilibrium does not evolve — next state equals current state.

    Parameters
    ----------
    surrogate : DPFSurrogate
        Loaded surrogate model instance.
    n_steps : int
        Trajectory length (must be > surrogate.history_length).
    fields : list[str] or None
        Fields to validate (default: all).
    **kwargs
        Extra keyword arguments forwarded to ``create_bennett_trajectory``.

    Returns
    -------
    dict[str, Any]
        Validation report from ``surrogate.validate_against_physics()``.
    """
    hl = getattr(surrogate, "history_length", 4)
    effective_steps = max(n_steps, hl + 2)
    trajectory = create_bennett_trajectory(n_steps=effective_steps, **kwargs)
    return surrogate.validate_against_physics(trajectory, fields=fields)


def validate_surrogate_against_noh(
    surrogate: Any,
    n_steps: int = 10,
    fields: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate surrogate predictions on an evolving magnetized Noh solution.

    The Noh trajectory has a time-evolving shock, so non-zero L2 error is
    expected even for a good surrogate.  This benchmark quantifies how well
    the surrogate tracks shock propagation dynamics.

    Parameters
    ----------
    surrogate : DPFSurrogate
        Loaded surrogate model instance.
    n_steps : int
        Trajectory length (must be > surrogate.history_length).
    fields : list[str] or None
        Fields to validate (default: all).
    **kwargs
        Extra keyword arguments forwarded to ``create_noh_trajectory``.

    Returns
    -------
    dict[str, Any]
        Validation report from ``surrogate.validate_against_physics()``.
    """
    hl = getattr(surrogate, "history_length", 4)
    effective_steps = max(n_steps, hl + 2)
    trajectory = create_noh_trajectory(n_steps=effective_steps, **kwargs)
    return surrogate.validate_against_physics(trajectory, fields=fields)
