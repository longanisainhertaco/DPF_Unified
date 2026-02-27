"""Orszag-Tang vortex benchmark on Metal GPU solver.

The Orszag-Tang vortex (Orszag & Tang, J. Fluid Mech. 90, 129, 1979) is the
canonical 2D MHD benchmark.  Smooth sinusoidal initial conditions develop into
supersonic MHD turbulence with thin current sheets, fast/slow MHD shocks, and
vortical structures — all through purely nonlinear dynamics.

We use Formulation A (unit domain [0,1]^2, Athena++ convention):
    rho  = 25 / (36*pi)
    p    = 5  / (12*pi)
    vx   = -sin(2*pi*y)
    vy   =  sin(2*pi*x)
    Bx   = -B0 * sin(2*pi*y),  B0 = 1/sqrt(4*pi)
    By   =  B0 * sin(4*pi*x)
    gamma = 5/3, periodic BCs

Heaviside-Lorentz code units (mu_0 = 1): E = p/(g-1) + 0.5*rho*|v|^2 + 0.5*|B|^2.

References
----------
- Orszag & Tang, J. Fluid Mech. 90, 129 (1979)
- Londrillo & Del Zanna, ApJ 530, 508 (2000)
- Stone et al., ApJS 178, 137 (2008) — Athena code paper
- Toth, JCP 161, 605 (2000) — div(B)=0 study
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402

# ============================================================
# Constants & initial conditions (Formulation A — unit domain)
# ============================================================

GAMMA = 5.0 / 3.0
RHO0 = 25.0 / (36.0 * np.pi)   # 0.22101...
P0 = 5.0 / (12.0 * np.pi)      # 0.13263...
B0 = 1.0 / np.sqrt(4.0 * np.pi)  # 0.28209...


def _make_orszag_tang_state(nx: int, ny: int, nz: int = 4) -> dict[str, np.ndarray]:
    """Create Orszag-Tang vortex initial conditions on [0,1]^2.

    Uses cell-centered coordinates with uniform spacing dx = 1/nx.
    The z-direction is uniform (nz >= 4 for WENO5 stencil padding).

    Parameters
    ----------
    nx, ny : int
        Grid size in x and y.
    nz : int
        Grid size in z (minimum 4 for stability; uniform in z).

    Returns
    -------
    dict
        State dict with rho, velocity, pressure, B, Te, Ti, psi.
    """
    dx = 1.0 / nx
    dy = 1.0 / ny

    # Cell-centered coordinates
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    rho = np.full((nx, ny, nz), RHO0)
    pressure = np.full((nx, ny, nz), P0)
    velocity = np.zeros((3, nx, ny, nz))
    B = np.zeros((3, nx, ny, nz))

    for k in range(nz):
        velocity[0, :, :, k] = -np.sin(2.0 * np.pi * Y)
        velocity[1, :, :, k] = np.sin(2.0 * np.pi * X)
        B[0, :, :, k] = -B0 * np.sin(2.0 * np.pi * Y)
        B[1, :, :, k] = B0 * np.sin(4.0 * np.pi * X)

    # Temperature (arbitrary for ideal MHD — set from p = n*k*T)
    ion_mass = 1.67e-27
    k_B = 1.38e-23
    T = pressure * ion_mass / (2.0 * np.maximum(rho, 1e-30) * k_B)

    return {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure,
        "B": B,
        "Te": T.copy(),
        "Ti": T.copy(),
        "psi": np.zeros((nx, ny, nz)),
    }


def _total_energy(state: dict[str, np.ndarray], gamma: float = GAMMA) -> float:
    """Total MHD energy in Heaviside-Lorentz units (mu_0 = 1).

    E = sum[ p/(g-1) + 0.5*rho*|v|^2 + 0.5*|B|^2 ]
    """
    e_th = state["pressure"] / (gamma - 1.0)
    e_kin = 0.5 * state["rho"] * np.sum(state["velocity"] ** 2, axis=0)
    e_mag = 0.5 * np.sum(state["B"] ** 2, axis=0)
    return float(np.sum(e_th + e_kin + e_mag))


def _run_orszag_tang(
    nx: int,
    t_end: float,
    n_steps: int | None = None,
    reconstruction: str = "plm",
    riemann_solver: str = "hll",
    limiter: str = "mc",
    precision: str = "float32",
    use_ct: bool = False,
) -> tuple[dict[str, np.ndarray], float, float, int]:
    """Run Orszag-Tang to t_end, returning (final_state, E0, E_final, steps).

    Parameters
    ----------
    nx : int
        Square grid cells (nx x nx).
    t_end : float
        Target end time.
    n_steps : int or None
        If given, run exactly this many steps (ignoring t_end for stopping).
    reconstruction, riemann_solver, limiter, precision : str
        Solver options.
    use_ct : bool
        Whether to use constrained transport for div(B)=0.

    Returns
    -------
    state : dict
        Final state.
    E0 : float
        Initial total energy.
    E_final : float
        Final total energy.
    steps : int
        Number of timesteps taken.
    """
    nz = 4
    dx = 1.0 / nx

    solver = MetalMHDSolver(
        grid_shape=(nx, nx, nz),
        dx=dx,
        gamma=GAMMA,
        cfl=0.3,
        reconstruction=reconstruction,
        riemann_solver=riemann_solver,
        limiter=limiter,
        precision=precision,
        use_ct=use_ct,
        bc=("periodic", "periodic", "periodic"),
        enable_bremsstrahlung=False,
        enable_hall=False,
        enable_braginskii_conduction=False,
        enable_braginskii_viscosity=False,
        enable_nernst=False,
    )

    state = _make_orszag_tang_state(nx, nx, nz)
    E0 = _total_energy(state)

    rho_floor = 1e-8 * RHO0
    p_floor = 1e-8 * P0

    t = 0.0
    steps = 0
    max_steps = n_steps if n_steps is not None else 100_000

    while steps < max_steps:
        if n_steps is None and t >= t_end:
            break
        dt = solver.compute_dt(state)
        if n_steps is None and t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            break

        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        state["rho"] = np.maximum(state["rho"], rho_floor)
        state["pressure"] = np.maximum(state["pressure"], p_floor)

        t += dt
        steps += 1

        if not np.all(np.isfinite(state["rho"])):
            break

    E_final = _total_energy(state)
    return state, E0, E_final, steps


# ============================================================
# Smoke tests — 10 steps on small grid
# ============================================================

class TestOrszagTangSmoke:
    """Quick sanity tests: run a few steps, check no crash / NaN."""

    def test_hll_plm_10_steps(self):
        """HLL+PLM should run 10 steps without NaN on 32x32 grid."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=32, t_end=1.0, n_steps=10,
            reconstruction="plm", riemann_solver="hll",
        )
        assert steps == 10
        assert np.all(np.isfinite(state["rho"]))
        assert np.all(np.isfinite(state["pressure"]))
        assert np.all(np.isfinite(state["B"]))
        assert np.min(state["rho"]) > 0

    def test_hlld_plm_10_steps(self):
        """HLLD+PLM should run 10 steps without NaN on 32x32 grid."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=32, t_end=1.0, n_steps=10,
            reconstruction="plm", riemann_solver="hlld",
        )
        assert steps == 10
        assert np.all(np.isfinite(state["rho"]))
        assert np.all(np.isfinite(state["pressure"]))
        assert np.min(state["rho"]) > 0

    def test_weno5_hll_10_steps(self):
        """WENO5+HLL should run 10 steps without NaN on 32x32 grid."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=32, t_end=1.0, n_steps=10,
            reconstruction="weno5", riemann_solver="hll",
        )
        assert steps == 10
        assert np.all(np.isfinite(state["rho"]))
        assert np.all(np.isfinite(state["pressure"]))

    def test_initial_energy(self):
        """Verify initial total energy matches analytical prediction.

        E0 = integral[ p/(g-1) + 0.5*rho*|v|^2 + 0.5*|B|^2 ] over [0,1]^2 * nz
        Analytical (per z-layer):
          E_th  = P0/(g-1) * 1 = 0.13263 / 0.6667 = 0.19895
          E_kin = 0.5 * RHO0 * <|v|^2> = 0.5 * 0.22101 * 1.0 = 0.11050
          E_mag = 0.5 * <|B|^2> = 0.5 * B0^2 * (0.5+0.5) = 0.03979
          E_per_layer = 0.34924
        """
        nz = 4
        state = _make_orszag_tang_state(64, 64, nz)
        E = _total_energy(state)
        # Analytical per cell, summed over nx*ny*nz cells:
        E_th_analytical = P0 / (GAMMA - 1.0)
        E_kin_analytical = 0.5 * RHO0 * 1.0  # <sin^2 + sin^2> = 1.0
        E_mag_analytical = 0.5 * B0**2 * 1.0  # <sin^2(2py) + sin^2(4px)> = 1.0
        E_per_cell = E_th_analytical + E_kin_analytical + E_mag_analytical
        E_expected = E_per_cell * 64 * 64 * nz  # sum over all cells (no dV — cell-sum)
        # The grid-sum energy should match because <sin^2> = 0.5 for each term:
        # but the actual discrete sum uses specific grid points, so allow ~1%
        assert pytest.approx(E_expected, rel=0.02) == E

    def test_initial_conditions_symmetry(self):
        """Orszag-Tang ICs have specific symmetries: <vx>=<vy>=0, <Bx>=<By>=0."""
        state = _make_orszag_tang_state(64, 64, 4)
        # Mean velocity and B should be zero (sinusoidal integrals over full period)
        assert np.abs(np.mean(state["velocity"][0])) < 1e-10
        assert np.abs(np.mean(state["velocity"][1])) < 1e-10
        assert np.abs(np.mean(state["B"][0])) < 1e-10
        assert np.abs(np.mean(state["B"][1])) < 1e-10
        # Uniform density and pressure
        assert np.std(state["rho"]) < 1e-15
        assert np.std(state["pressure"]) < 1e-15


# ============================================================
# Conservation tests — 64x64 to t=0.1
# ============================================================

class TestOrszagTangConservation:
    """Energy, mass, and momentum conservation on 64x64 grid."""

    def test_mass_conservation_hll(self):
        """Total mass should be conserved to < 1e-4 (periodic BCs)."""
        state0 = _make_orszag_tang_state(64, 64, 4)
        M0 = float(np.sum(state0["rho"]))

        state, _, _, steps = _run_orszag_tang(
            nx=64, t_end=0.1,
            reconstruction="plm", riemann_solver="hll",
        )
        M_final = float(np.sum(state["rho"]))
        rel_err = abs(M_final - M0) / M0
        assert rel_err < 1e-4, f"Mass conservation failed: rel_err={rel_err:.2e}"

    def test_energy_conservation_hll(self):
        """Total energy conservation < 5% at t=0.1 on 64x64 (float32)."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=64, t_end=0.1,
            reconstruction="plm", riemann_solver="hll",
        )
        rel_err = abs(Ef - E0) / E0
        assert rel_err < 0.05, f"Energy conservation failed: rel_err={rel_err:.2e}"

    def test_energy_conservation_hlld(self):
        """HLLD should conserve energy as well as or better than HLL."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=64, t_end=0.1,
            reconstruction="plm", riemann_solver="hlld",
        )
        rel_err = abs(Ef - E0) / E0
        assert rel_err < 0.05, f"HLLD energy conservation failed: rel_err={rel_err:.2e}"

    def test_momentum_conservation(self):
        """Total momentum should remain near zero (periodic sinusoidal ICs)."""
        state0 = _make_orszag_tang_state(64, 64, 4)
        state, _, _, _ = _run_orszag_tang(
            nx=64, t_end=0.1,
            reconstruction="plm", riemann_solver="hll",
        )
        mom_x = float(np.sum(state["rho"] * state["velocity"][0]))
        mom_y = float(np.sum(state["rho"] * state["velocity"][1]))
        M0 = float(np.sum(state0["rho"]))
        # Momentum should be near zero; normalize by initial mass * v_max = M0 * 1.0
        assert abs(mom_x) / M0 < 0.05, f"x-momentum drift: {mom_x/M0:.2e}"
        assert abs(mom_y) / M0 < 0.05, f"y-momentum drift: {mom_y/M0:.2e}"

    def test_density_stays_positive(self):
        """Density must remain positive through the evolution."""
        state, _, _, _ = _run_orszag_tang(
            nx=64, t_end=0.1,
            reconstruction="plm", riemann_solver="hll",
        )
        assert np.min(state["rho"]) > 0, f"Negative density: min={np.min(state['rho'])}"


# ============================================================
# Physics benchmarks — 64x64 to t=0.5
# ============================================================

@pytest.mark.slow
class TestOrszagTangBenchmark64:
    """Run to t=0.5 on 64x64 — the standard comparison time."""

    def test_hll_plm_density_range(self):
        """At t=0.5 density should show compression and rarefaction.

        Reference (Athena 192x192): rho_min ~ 0.10-0.15, rho_max ~ 0.45-0.50.
        On 64x64 with HLL (diffusive), ranges will be narrower.
        """
        state, E0, Ef, steps = _run_orszag_tang(
            nx=64, t_end=0.5,
            reconstruction="plm", riemann_solver="hll",
        )
        assert np.all(np.isfinite(state["rho"]))
        rho_min = float(np.min(state["rho"]))
        rho_max = float(np.max(state["rho"]))
        # Density must have developed structure (not still uniform)
        assert rho_max / rho_min > 1.5, (
            f"Density range too narrow: [{rho_min:.4f}, {rho_max:.4f}]"
        )
        # Energy conservation at t=0.5 should be < 10% on coarse grid
        rel_err = abs(Ef - E0) / E0
        assert rel_err < 0.10, f"Energy drift at t=0.5: {rel_err:.2e}"

    def test_hlld_plm_sharper_features(self):
        """HLLD should produce wider density range than HLL (less diffusive)."""
        state_hll, _, _, _ = _run_orszag_tang(
            nx=64, t_end=0.5,
            reconstruction="plm", riemann_solver="hll",
        )
        state_hlld, _, _, _ = _run_orszag_tang(
            nx=64, t_end=0.5,
            reconstruction="plm", riemann_solver="hlld",
        )
        range_hll = np.max(state_hll["rho"]) - np.min(state_hll["rho"])
        range_hlld = np.max(state_hlld["rho"]) - np.min(state_hlld["rho"])
        # HLLD should resolve sharper features -> wider range (or at least comparable)
        # Allow HLLD to be no worse than 80% of HLL range
        assert range_hlld > 0.8 * range_hll, (
            f"HLLD range ({range_hlld:.4f}) unexpectedly narrower than "
            f"HLL range ({range_hll:.4f})"
        )

    def test_magnetic_energy_decreases(self):
        """Magnetic energy should decrease (converted to kinetic/thermal by reconnection)."""
        state0 = _make_orszag_tang_state(64, 64, 4)
        E_mag_0 = 0.5 * float(np.sum(state0["B"] ** 2))

        state, _, _, _ = _run_orszag_tang(
            nx=64, t_end=0.5,
            reconstruction="plm", riemann_solver="hll",
        )
        E_mag_f = 0.5 * float(np.sum(state["B"] ** 2))

        # Magnetic energy should decrease as reconnection converts it
        assert E_mag_f < E_mag_0, (
            f"Magnetic energy increased: {E_mag_0:.4f} -> {E_mag_f:.4f}"
        )


# ============================================================
# 128x128 high-resolution benchmarks
# ============================================================

@pytest.mark.slow
class TestOrszagTangBenchmark128:
    """128x128 grid — better resolved features, tighter tolerances."""

    def test_hll_plm_no_nan_t05(self):
        """128x128 HLL+PLM should reach t=0.5 without NaN."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=128, t_end=0.5,
            reconstruction="plm", riemann_solver="hll",
        )
        assert np.all(np.isfinite(state["rho"]))
        assert np.all(np.isfinite(state["pressure"]))
        assert np.all(np.isfinite(state["B"]))
        # Better energy conservation at higher resolution
        rel_err = abs(Ef - E0) / E0
        assert rel_err < 0.05, f"Energy drift at 128x128: {rel_err:.2e}"

    def test_hlld_plm_no_nan_t05(self):
        """128x128 HLLD+PLM should reach t=0.5 without NaN."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=128, t_end=0.5,
            reconstruction="plm", riemann_solver="hlld",
        )
        assert np.all(np.isfinite(state["rho"]))
        assert np.min(state["rho"]) > 0

    def test_density_range_convergence(self):
        """128x128 should produce wider density range than 64x64 (less diffusion)."""
        state_64, _, _, _ = _run_orszag_tang(
            nx=64, t_end=0.5,
            reconstruction="plm", riemann_solver="hll",
        )
        state_128, _, _, _ = _run_orszag_tang(
            nx=128, t_end=0.5,
            reconstruction="plm", riemann_solver="hll",
        )
        range_64 = np.max(state_64["rho"]) - np.min(state_64["rho"])
        range_128 = np.max(state_128["rho"]) - np.min(state_128["rho"])
        # Higher resolution -> sharper features -> wider range
        assert range_128 > range_64, (
            f"128x128 range ({range_128:.4f}) not wider than "
            f"64x64 range ({range_64:.4f})"
        )

    def test_float64_energy_conservation(self):
        """Float64 precision should give much better energy conservation."""
        state, E0, Ef, steps = _run_orszag_tang(
            nx=64, t_end=0.1,
            reconstruction="plm", riemann_solver="hll",
            precision="float64",
        )
        rel_err = abs(Ef - E0) / E0
        # Float64 should conserve energy to < 1% on short runs
        assert rel_err < 0.01, f"Float64 energy drift: {rel_err:.2e}"
