"""Implicit induction equation for vacuum cells in the MLX cylindrical solver.

Motivation
----------
The explicit SSP-RK3 integrator is constrained by the fast-magnetosonic CFL
across ALL cells.  In the vacuum region behind the compression sheath (low
rho, frozen-in B_theta from electrode BCs), the Alfven speed v_A = |B|/sqrt(rho)
can be enormous even after the va_max=1e6 density floor, because that floor
only caps the *wavespeed contribution to dt* — it does not remove the frozen-in
field from the induction equation.

The floor approach works for stability but it introduces a fictitious mass
source.  A cleaner treatment is to recognise that in true vacuum (rho below
some threshold), the induction equation degenerates:

    dB/dt = curl(v x B) - curl(eta/mu0 * J)

and the "velocity" term vanishes (rho -> 0 => v -> 0 for finite momentum).
What remains is pure resistive diffusion — a parabolic operator — which is
already handled implicitly in mlx_transport.py via the Thomas ADI solver.

This module extends that idea to cover the full induction equation in vacuum
cells.  The operator-split approach is:

    1. Explicit SSP-RK3 on *physical* cells (rho >= threshold * rho_max).
    2. Implicit induction solve on *vacuum* cells (rho < threshold * rho_max).
    3. Merge: physical cells take the explicit result; vacuum cells take the
       implicit result.  The merge is a simple masked selection — no
       interpolation at the boundary because operator splitting treats the
       two regions sequentially.

Implementation plan (Phase S-3)
--------------------------------
The Thomas ADI infrastructure from mlx_transport.apply_resistive_diffusion
is reused directly.  The implicit step solves the linearised induction equation
in each vacuum cell column using the existing _build_cylindrical_diffusion_system
/ thomas_solve pair.  The velocity advection term is dropped (vacuum => v=0)
so the system is exactly resistive diffusion.

When to use
-----------
- Enable when ``ImplicitMHDConfig.threshold > 0`` (disabled by default to
  preserve current Phase Q behavior until Phase S-3 is complete).
- Recommended threshold: 1e-3 (denser than the CFL mask at 1e-4, so it covers
  more of the under-resolved region while leaving physical cells explicit).
- Not needed if the resistive-diffusion sub-cycle (mlx_transport) already
  damps the vacuum B on a timescale shorter than dt_MHD.  Profile both before
  enabling.

References
----------
Stone J.M. & Norman M.L., ApJS 80:753 (1992) — operator-split MHD.
Ryu D. et al., ApJ 452:364 (1995) — implicit B diffusion in tenuous regions.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

from dpf.metal.mlx_grid import CylindricalGrid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ImplicitMHDConfig:
    """Configuration for the implicit induction solver on vacuum cells.

    Parameters
    ----------
    threshold : float
        Fraction of rho_max below which a cell is treated as vacuum and
        handed to the implicit solver.  0.0 disables the module entirely
        (default: disabled to preserve Phase Q behaviour until Phase S-3).
    method : str
        Implicit method to use.  Currently only ``"adi"`` (Alternating
        Direction Implicit via Thomas tridiagonal) is planned.
    max_iterations : int
        Maximum iterations for iterative implicit solvers (reserved for
        future Gauss-Seidel / multigrid methods).  Not used by the ADI path.
    eta_vacuum : float | None
        Override resistivity [Ohm*m] used in vacuum cells.  If None, the
        same eta array as the rest of the domain is used.  A higher value
        in vacuum damps unphysical currents faster.
    sub_cycle : bool
        If True, sub-cycle the implicit step with dt_implicit = dt / n_sub
        where n_sub is chosen to maintain stability of the linear system.
        Recommended for large eta_vacuum values.
    """

    threshold: float = 0.0
    method: str = "adi"
    max_iterations: int = 1
    eta_vacuum: float | None = None
    sub_cycle: bool = False

    def __post_init__(self) -> None:
        if self.threshold < 0.0 or self.threshold >= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1); got {self.threshold}"
            )
        if self.method not in {"adi"}:
            raise ValueError(
                f"method must be 'adi'; got '{self.method}'"
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1; got {self.max_iterations}"
            )


# ---------------------------------------------------------------------------
# Vacuum cell identification
# ---------------------------------------------------------------------------


def identify_vacuum_cells(
    U: mx.array,
    rho_threshold: float,
) -> mx.array:
    """Identify vacuum cells where rho < rho_threshold * rho_max.

    Vacuum cells are those whose density is below a fixed fraction of the
    domain maximum.  These are the cells that cause extreme Alfven speeds and
    would otherwise throttle the CFL timestep to sub-picosecond values at
    pinch time.

    The mask is computed on-device (MLX) to avoid a GPU->CPU round-trip.
    It is re-evaluated every timestep because the sheath position moves.

    Parameters
    ----------
    U : mx.array
        Conserved state array, shape (NVAR, nr, nz), float32.
    rho_threshold : float
        Fraction of rho_max.  Cells with rho < rho_threshold * rho_max are
        marked as vacuum.  Typical value: 1e-3.

    Returns
    -------
    mask : mx.array
        Boolean mask, shape (nr, nz).  True where the cell is vacuum.

    Notes
    -----
    Uses IDN=0 slice of U for density.  Safe to call before or after floor
    enforcement; a post-floor call will reflect the true physical density.
    """
    raise NotImplementedError("Phase S-3: implement after AMR + species validation")


# ---------------------------------------------------------------------------
# Implicit induction step
# ---------------------------------------------------------------------------


def implicit_induction_step(
    U: mx.array,
    dt: float,
    eta: mx.array | float,
    grid: CylindricalGrid,
    mask: mx.array,
    gamma: float = 5.0 / 3.0,
    config: ImplicitMHDConfig | None = None,
) -> mx.array:
    """Advance the induction equation implicitly in masked (vacuum) cells.

    For cells where ``mask`` is True (vacuum), the induction equation reduces
    to resistive diffusion because v -> 0 as rho -> 0.  This function applies
    the Thomas ADI solver from mlx_transport to those cells, treating the
    velocity advection term as zero.

    The solve proceeds in two ADI sweeps:
    - z-sweep: solve each radial column along z (uniform spacing, Thomas).
    - r-sweep: solve each axial column along r (cylindrical, Thomas with 1/r).

    Only the B-field components (IBR, IBZ, IBT) are updated.  The density,
    momentum, energy, and entropy tracer are left unchanged — the implicit
    step is only responsible for the induction equation.

    Parameters
    ----------
    U : mx.array
        Conserved state, shape (NVAR, nr, nz), float32.  Input is NOT
        modified in-place; a new array is returned.
    dt : float
        Timestep [s].  The implicit solver is unconditionally stable so
        dt can be the full MHD CFL step (no sub-cycling required for
        moderate eta).
    eta : mx.array or float
        Resistivity [Ohm*m].  Scalar or shape (nr, nz).
    grid : CylindricalGrid
        Grid geometry (provides dr, dz, r_cell).
    mask : mx.array
        Boolean vacuum mask, shape (nr, nz).  Only cells where mask=True
        are solved; physical cells are passed through unchanged.
    gamma : float
        Adiabatic index (unused in the pure diffusion path, carried for
        interface consistency).
    config : ImplicitMHDConfig or None
        Configuration object.  If None, defaults are used.  ``config.eta_vacuum``
        overrides eta in vacuum cells when set.

    Returns
    -------
    U_implicit : mx.array
        Updated conserved state with B-field advanced in vacuum cells,
        shape (NVAR, nr, nz), float32.

    Notes
    -----
    The Thomas solver runs on CPU in float64 (see mlx_transport.thomas_solve).
    This requires one GPU->CPU copy of the B-field columns that intersect the
    vacuum mask, and one CPU->GPU copy of the updated columns.  For typical
    DPF grids (nr=64, nz=256) with ~20% vacuum cells, this is ~150 KB per
    timestep — negligible compared to the simulation state.

    Ohmic heating from the implicit step is NOT applied here to avoid double-
    counting with mlx_transport.apply_resistive_diffusion.  If the implicit
    MHD module replaces (rather than supplements) the transport step in vacuum
    cells, the caller is responsible for routing Ohmic heating correctly.
    """
    raise NotImplementedError("Phase S-3: implement after AMR + species validation")


# ---------------------------------------------------------------------------
# Merge explicit and implicit results
# ---------------------------------------------------------------------------


def merge_explicit_implicit(
    U_explicit: mx.array,
    U_implicit: mx.array,
    mask: mx.array,
) -> mx.array:
    """Combine explicit (physical) and implicit (vacuum) state arrays.

    Constructs the merged state by selecting cell-wise:
    - Physical cells (mask=False): take values from U_explicit.
    - Vacuum cells (mask=True): take values from U_implicit.

    The mask is broadcast from (nr, nz) to (NVAR, nr, nz) for a single
    mx.where call.  This keeps the merge entirely on-device with no Python
    loop over variables.

    Parameters
    ----------
    U_explicit : mx.array
        State after explicit SSP-RK3 step, shape (NVAR, nr, nz), float32.
    U_implicit : mx.array
        State after implicit induction step, shape (NVAR, nr, nz), float32.
        For physical cells this should equal U_explicit (so the merge is
        a no-op there), but that is NOT enforced — the caller controls it.
    mask : mx.array
        Boolean vacuum mask, shape (nr, nz).  True selects U_implicit.

    Returns
    -------
    U_merged : mx.array
        Cell-wise merged state, shape (NVAR, nr, nz), float32.

    Notes
    -----
    If U_explicit and U_implicit were produced by independent solves that
    diverge significantly at the physical/vacuum boundary, spurious gradients
    can form.  The caller should ensure that both solves start from the same
    initial state and that the boundary region is at least 1 cell wide in
    the physical direction (i.e., avoid mask=True cells immediately adjacent
    to inflowing boundaries unless resistive diffusion smooths the transition).
    """
    raise NotImplementedError("Phase S-3: implement after AMR + species validation")


# ---------------------------------------------------------------------------
# High-level operator-split step
# ---------------------------------------------------------------------------


def apply_implicit_mhd_split(
    U_explicit: mx.array,
    U_pre_explicit: mx.array,
    dt: float,
    eta: mx.array | float,
    grid: CylindricalGrid,
    config: ImplicitMHDConfig,
    gamma: float = 5.0 / 3.0,
) -> mx.array:
    """Full operator-split implicit MHD correction.

    Convenience wrapper that executes the three-step sequence:
      1. identify_vacuum_cells(U_pre_explicit, config.threshold)
      2. implicit_induction_step(U_pre_explicit, dt, eta, grid, mask, gamma, config)
      3. merge_explicit_implicit(U_explicit, U_implicit, mask)

    This function is the intended integration point in mlx_solver.py.
    It is called AFTER the explicit SSP-RK3 step and BEFORE the transport
    operator (resistive diffusion / thermal conduction).

    The vacuum mask is computed from U_pre_explicit (state at the START of
    the timestep) rather than U_explicit to avoid classifying cells that
    were only temporarily depleted during an RK stage as permanent vacuum.

    Parameters
    ----------
    U_explicit : mx.array
        State after the full explicit SSP-RK3 step, shape (NVAR, nr, nz).
    U_pre_explicit : mx.array
        State at the START of the timestep (before RK3), shape (NVAR, nr, nz).
        Used only for mask computation.
    dt : float
        Timestep [s].
    eta : mx.array or float
        Resistivity [Ohm*m].
    grid : CylindricalGrid
        Grid geometry.
    config : ImplicitMHDConfig
        Configuration.  If config.threshold == 0.0, returns U_explicit
        unchanged (fast path, no cost).
    gamma : float
        Adiabatic index.

    Returns
    -------
    U_out : mx.array
        Merged state, shape (NVAR, nr, nz), float32.
    """
    raise NotImplementedError("Phase S-3: implement after AMR + species validation")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def vacuum_cell_stats(
    U: mx.array,
    mask: mx.array,
) -> dict[str, float]:
    """Compute diagnostic statistics for the vacuum cell population.

    Returns a dict suitable for logging or time-series storage.  Intended
    for use in the solver's diagnostic step, not the hot path.

    Parameters
    ----------
    U : mx.array
        Conserved state, shape (NVAR, nr, nz).
    mask : mx.array
        Boolean vacuum mask, shape (nr, nz).

    Returns
    -------
    stats : dict[str, float]
        Keys:
        - ``n_vacuum``: total number of vacuum cells.
        - ``frac_vacuum``: vacuum cells as a fraction of total cells.
        - ``rho_min_vacuum``: minimum density in vacuum region.
        - ``B_max_vacuum``: maximum |B| in vacuum region (HL units).
        - ``va_max_vacuum``: maximum Alfven speed in vacuum region [m/s].
    """
    raise NotImplementedError("Phase S-3: implement after AMR + species validation")
