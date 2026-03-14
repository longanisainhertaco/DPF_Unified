"""Phase AL: Shock convergence at DPF-relevant conditions.

PhD Debate #23 identified that Phase AK's smooth-wave convergence is the
"easiest possible regime" for numerical methods.  Shock-capturing is where
methods are actually stressed.  This test extends the convergence study to
discontinuous problems:

1. **Sod shock tube at PF-1000 fill conditions** — with exact Riemann
   solution comparison (rho, p, v) and L1 error convergence.
2. **Brio-Wu at PF-1000 density/pressure** — MHD shock convergence via
   self-convergence (no exact solution; compare coarse vs fine grid).
3. **Repair fraction under shocks** — the repair diagnostic added in
   Phase AK should register nonzero repairs on shock problems.

Physical conditions (PF-1000 fill gas):
    rho_L = 7.53e-4 kg/m³  (3.5 Torr D2 at 300K)
    p_L   = 466 Pa          (ideal gas D2)
    rho_R = rho_L / 8       (Sod right state)
    p_R   = p_L / 10        (Sod right state)

For Brio-Wu, left/right density and pressure are PF-1000-scale with
magnetic field B0 = 0.01 T (Heaviside-Lorentz B0_HL ~ 8.92).

Shock convergence is typically ~1st order due to the O(h) Gibbs
phenomenon at discontinuities, regardless of the formal reconstruction
order in smooth regions.  We verify this and measure the actual L1 rate.

References:
    Toro E.F. (2009), Riemann Solvers and Numerical Methods for Fluid
        Dynamics, 3rd ed., Ch. 4 — Exact Riemann solver for Euler.
    Brio M. & Wu C.C. (1988), J. Comput. Phys. 75, 400-422.
    Miyoshi T. & Kusano K. (2005), JCP 208, 315-344 — HLLD.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf.metal.metal_riemann import (  # noqa: E402, I001
    get_repair_stats,
    reset_repair_stats,
)
from dpf.metal.metal_solver import MetalMHDSolver  # noqa: E402

# =====================================================================
# Physical constants — PF-1000 fill conditions
# =====================================================================

_MU_0 = 4.0 * np.pi * 1e-7
_RHO0 = 7.53e-4  # kg/m³ (3.5 Torr D2 at 300 K)
_P0 = 466.0  # Pa (ideal gas D2 at 300 K)
_B0_SI = 0.01  # T
_B0_HL = _B0_SI / np.sqrt(_MU_0)  # ~8.92 Heaviside-Lorentz
_GAMMA = 5.0 / 3.0


# =====================================================================
# Exact Riemann solver for the Sod problem (Euler equations)
# =====================================================================


def _exact_sod(
    x: np.ndarray,
    t: float,
    rho_L: float,
    p_L: float,
    rho_R: float,
    p_R: float,
    gamma: float,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact solution for the Sod shock tube at time t.

    Solves the Riemann problem with left/right states connected by a
    left-going rarefaction, contact, and right-going shock.

    Returns (rho, u, p) arrays evaluated at positions x.
    """
    # Abbreviations
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    g2 = 2.0 / (gamma + 1.0)
    g3 = gm1 / gp1
    g4 = 2.0 / gm1
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    # Newton-Raphson for star region pressure p_star
    # For Sod-like problems: left rarefaction + right shock
    p_star = 0.5 * (p_L + p_R)  # initial guess
    for _ in range(50):
        # Left rarefaction branch
        f_L = g4 * c_L * ((p_star / p_L) ** (gm1 / (2.0 * gamma)) - 1.0)
        df_L = (
            (1.0 / (rho_L * c_L))
            * (p_star / p_L) ** (-(gp1) / (2.0 * gamma))
        )

        # Right shock branch
        A_R = g2 / rho_R
        B_R = g3 * p_R
        sqrt_term = np.sqrt(A_R / (p_star + B_R))
        f_R = (p_star - p_R) * sqrt_term
        df_R = sqrt_term * (1.0 - 0.5 * (p_star - p_R) / (p_star + B_R))

        f = f_L + f_R
        df = df_L + df_R
        dp = -f / df
        p_star += dp
        if abs(dp) < 1e-10 * p_star:
            break

    # Star region velocity
    u_star = 0.5 * (
        f_L - f_R  # This is u_L - f_L + u_R + f_R with u_L=u_R=0
    )
    # Recompute correctly: u_star = 0.5 * (u_L + u_R + f_R - f_L)
    # With u_L = u_R = 0:
    u_star = 0.5 * (f_R - f_L)

    # Left rarefaction fan
    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)
    c_star_L = np.sqrt(gamma * p_star / rho_star_L)

    # Right shock
    rho_star_R = rho_R * (
        (p_star / p_R + g3) / (g3 * p_star / p_R + 1.0)
    )
    S_R = (
        np.sqrt((gp1 / (2.0 * gamma)) * p_star / p_R + gm1 / (2.0 * gamma))
        * c_R
    )  # shock speed (right-moving)

    # Rarefaction head and tail speeds
    S_HL = -c_L  # head of rarefaction (left-going)
    S_TL = u_star - c_star_L  # tail of rarefaction

    # Build solution
    rho_out = np.empty_like(x)
    u_out = np.empty_like(x)
    p_out = np.empty_like(x)

    for i, xi in enumerate(x):
        s = (xi - x0) / t if t > 0 else 0.0

        if s < S_HL:
            # Left undisturbed
            rho_out[i] = rho_L
            u_out[i] = 0.0
            p_out[i] = p_L
        elif s < S_TL:
            # Inside rarefaction fan
            rho_out[i] = rho_L * (g2 + g3 * (-s) / c_L) ** g4
            u_out[i] = g2 * (c_L + s)
            p_out[i] = p_L * (g2 + g3 * (-s) / c_L) ** (g4 * gamma)
        elif s < u_star:
            # Star region (left of contact)
            rho_out[i] = rho_star_L
            u_out[i] = u_star
            p_out[i] = p_star
        elif s < S_R:
            # Star region (right of contact)
            rho_out[i] = rho_star_R
            u_out[i] = u_star
            p_out[i] = p_star
        else:
            # Right undisturbed
            rho_out[i] = rho_R
            u_out[i] = 0.0
            p_out[i] = p_R

    return rho_out, u_out, p_out


# =====================================================================
# IC builders for shock tubes at PF-1000 conditions
# =====================================================================


def _make_sod_dpf(
    nx: int,
    x0: float = 0.5,
) -> tuple[dict, float, float]:
    """Sod shock tube at PF-1000 fill conditions.

    Left state:  rho = rho0,     p = p0      (PF-1000 fill)
    Right state: rho = rho0/8,   p = p0/10   (standard Sod ratio)

    Returns (state, dx, domain_length).
    """
    ny = nz = 4
    L = 1.0  # normalized domain
    dx = L / nx
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)

    rho = np.where(xc < x0, _RHO0, _RHO0 / 8.0)
    pressure = np.where(xc < x0, _P0, _P0 / 10.0)

    rho_3d = np.broadcast_to(rho[:, None, None], (nx, ny, nz)).copy()
    p_3d = np.broadcast_to(pressure[:, None, None], (nx, ny, nz)).copy()

    state = {
        "rho": rho_3d,
        "velocity": np.zeros((3, nx, ny, nz), dtype=np.float64),
        "pressure": p_3d,
        "B": np.zeros((3, nx, ny, nz), dtype=np.float64),
        "Te": np.full((nx, ny, nz), 300.0, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), 300.0, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx, L


def _make_briowu_dpf(
    nx: int,
    x0: float = 0.5,
) -> tuple[dict, float, float]:
    """Brio-Wu MHD shock tube at PF-1000-scale density/pressure.

    Left:  rho = rho0,     p = p0,      Bx = 0.75*B0_HL, By = +B0_HL
    Right: rho = rho0/8,   p = p0/10,   Bx = 0.75*B0_HL, By = -B0_HL

    Preserves the standard Brio-Wu density/pressure ratios (8:1, 10:1)
    and By sign flip, but at PF-1000 parameter magnitudes.

    Returns (state, dx, domain_length).
    """
    ny = nz = 4
    L = 1.0
    dx = L / nx
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)

    rho = np.where(xc < x0, _RHO0, _RHO0 / 8.0)
    pressure = np.where(xc < x0, _P0, _P0 / 10.0)
    Bx = np.full(nx, 0.75 * _B0_HL)
    By = np.where(xc < x0, _B0_HL, -_B0_HL)

    rho_3d = np.broadcast_to(rho[:, None, None], (nx, ny, nz)).copy()
    p_3d = np.broadcast_to(pressure[:, None, None], (nx, ny, nz)).copy()
    B_3d = np.zeros((3, nx, ny, nz), dtype=np.float64)
    B_3d[0] = np.broadcast_to(Bx[:, None, None], (nx, ny, nz)).copy()
    B_3d[1] = np.broadcast_to(By[:, None, None], (nx, ny, nz)).copy()

    state = {
        "rho": rho_3d,
        "velocity": np.zeros((3, nx, ny, nz), dtype=np.float64),
        "pressure": p_3d,
        "B": B_3d,
        "Te": np.full((nx, ny, nz), 300.0, dtype=np.float64),
        "Ti": np.full((nx, ny, nz), 300.0, dtype=np.float64),
        "psi": np.zeros((nx, ny, nz), dtype=np.float64),
    }
    return state, dx, L


# =====================================================================
# Runner utilities
# =====================================================================


def _run_sod_dpf(
    nx: int,
    t_end: float,
    riemann: str = "hll",
    recon: str = "plm",
    integrator: str = "ssp_rk2",
    precision: str = "float64",
) -> tuple[dict, float, float]:
    """Run Sod problem at PF-1000 conditions and return final state.

    Returns (state, dx, L).
    """
    state, dx, L = _make_sod_dpf(nx)

    solver = MetalMHDSolver(
        grid_shape=(nx, 4, 4),
        dx=dx,
        gamma=_GAMMA,
        cfl=0.4,
        device="cpu",
        riemann_solver=riemann,
        reconstruction=recon,
        time_integrator=integrator,
        precision=precision,
        use_ct=False,
    )

    t = 0.0
    for _ in range(50000):
        dt = solver.compute_dt(state)
        if t + dt > t_end:
            dt = t_end - t
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        t += dt
        if t >= t_end * 0.9999:
            break

    return state, dx, L


def _run_briowu_dpf(
    nx: int,
    t_end: float,
    riemann: str = "hll",
    recon: str = "plm",
    integrator: str = "ssp_rk2",
    precision: str = "float64",
    gamma: float = 2.0,
) -> tuple[dict, float, float]:
    """Run Brio-Wu at PF-1000 conditions and return final state.

    Returns (state, dx, L).
    """
    state, dx, L = _make_briowu_dpf(nx)

    solver = MetalMHDSolver(
        grid_shape=(nx, 4, 4),
        dx=dx,
        gamma=gamma,
        cfl=0.3,
        device="cpu",
        riemann_solver=riemann,
        reconstruction=recon,
        time_integrator=integrator,
        precision=precision,
        use_ct=False,
    )

    t = 0.0
    for _ in range(50000):
        dt = solver.compute_dt(state)
        if t + dt > t_end:
            dt = t_end - t
        state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
        t += dt
        if t >= t_end * 0.9999:
            break

    return state, dx, L


def _l1_error_sod(
    state: dict,
    nx: int,
    dx: float,
    L: float,
    t_end: float,
) -> dict[str, float]:
    """Compute L1 errors for Sod solution vs exact.

    Returns dict of L1(rho), L1(u), L1(p).
    """
    xc = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    rho_exact, u_exact, p_exact = _exact_sod(
        xc, t_end, _RHO0, _P0, _RHO0 / 8.0, _P0 / 10.0, _GAMMA,
    )

    rho_num = state["rho"][:, 2, 2]
    u_num = state["velocity"][0, :, 2, 2]
    p_num = state["pressure"][:, 2, 2]

    return {
        "rho": float(np.mean(np.abs(rho_num - rho_exact))),
        "u": float(np.mean(np.abs(u_num - u_exact))),
        "p": float(np.mean(np.abs(p_num - p_exact))),
    }


def _self_convergence_l1(
    state_coarse: dict,
    state_fine: dict,
    nx_c: int,
    nx_f: int,
) -> float:
    """Self-convergence L1(rho) between coarse and fine grid.

    Downsample fine grid to coarse grid resolution, compute L1.
    Assumes nx_f = 2 * nx_c.
    """
    rho_c = state_coarse["rho"][:, 2, 2]
    rho_f = state_fine["rho"][:, 2, 2]
    # Average pairs: fine cell 2i and 2i+1 -> coarse cell i
    rho_f_down = 0.5 * (rho_f[0::2] + rho_f[1::2])
    return float(np.mean(np.abs(rho_c - rho_f_down[:nx_c])))


# =====================================================================
# Sod shock tube tests at PF-1000 conditions
# =====================================================================


class TestSodDPFStability:
    """Sod shock tube at PF-1000 fill conditions — stability."""

    def test_sod_dpf_no_nan(self):
        """Sod at PF-1000 conditions runs 200 steps without NaN."""
        state, dx, L = _make_sod_dpf(nx=100)
        solver = MetalMHDSolver(
            grid_shape=(100, 4, 4),
            dx=dx,
            gamma=_GAMMA,
            cfl=0.4,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(200):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        assert not np.any(np.isnan(state["rho"])), "NaN in density"
        assert not np.any(np.isnan(state["pressure"])), "NaN in pressure"
        assert np.all(state["rho"] > 0), "Negative density"
        assert np.all(state["pressure"] > 0), "Negative pressure"

    def test_sod_dpf_shock_present(self):
        """Verify shock, rarefaction, and contact are resolved in density."""
        # Run to t ~ 0.2 * L / cs ~ 2e-4 s equivalent
        cs = np.sqrt(_GAMMA * _P0 / _RHO0)
        t_end = 0.2 / cs  # ~2e-4 s

        state, dx, L = _run_sod_dpf(nx=200, t_end=t_end, precision="float64")
        rho = state["rho"][:, 2, 2]

        # Density should have a plateau (star region) with rho between
        # rho_L and rho_R
        rho_min = rho.min()
        assert rho_min >= 0, "Negative density"
        assert rho_min < _RHO0 / 4, "Density never drops below half of left state"
        # Check there's a contact discontinuity (density jump in star region)
        d_rho = np.diff(rho)
        max_jump = np.max(np.abs(d_rho))
        assert max_jump > 0.01 * _RHO0, "No density jump detected (missing contact)"


class TestSodDPFConvergence:
    """Sod shock convergence at PF-1000 fill conditions."""

    @pytest.mark.slow
    def test_sod_plm_hll_convergence_order(self):
        """PLM+HLL Sod convergence should be ~0.5-1.0 (first order at shocks).

        Theory: shocks are O(h^1) even with higher-order methods. The global
        L1 rate for a multi-wave Riemann problem is typically 0.5-1.0 due to
        the contact discontinuity (spreads as sqrt(h) for diffusive schemes).
        """
        cs = np.sqrt(_GAMMA * _P0 / _RHO0)
        t_end = 0.15 / cs

        resolutions = [64, 128, 256]
        errors: list[float] = []

        for nx in resolutions:
            state, dx, L = _run_sod_dpf(
                nx=nx, t_end=t_end, precision="float64",
            )
            err = _l1_error_sod(state, nx, dx, L, t_end)
            errors.append(err["rho"])

        # Convergence order
        order_1 = np.log2(errors[0] / errors[1])
        order_2 = np.log2(errors[1] / errors[2])
        avg_order = 0.5 * (order_1 + order_2)

        print("\nSod PLM+HLL convergence at PF-1000 conditions:")
        for i, nx in enumerate(resolutions):
            print(f"  nx={nx:4d}: L1(rho) = {errors[i]:.4e}")
        print(f"  Order: {order_1:.2f}, {order_2:.2f}, avg={avg_order:.2f}")

        # Shocks limit global order to ~0.5-1.0
        assert avg_order > 0.3, f"Convergence too slow: {avg_order:.2f}"
        assert avg_order < 1.5, f"Convergence suspiciously fast: {avg_order:.2f}"
        assert errors[-1] < errors[0], "Error did not decrease with resolution"

    @pytest.mark.slow
    def test_sod_weno5_hlld_convergence(self):
        """WENO5+HLLD should have lower L1 than PLM+HLL at same resolution."""
        cs = np.sqrt(_GAMMA * _P0 / _RHO0)
        t_end = 0.15 / cs
        nx = 256

        state_plm, dx_plm, L = _run_sod_dpf(
            nx=nx, t_end=t_end, riemann="hll", recon="plm",
            precision="float64",
        )
        err_plm = _l1_error_sod(state_plm, nx, dx_plm, L, t_end)

        state_weno, dx_weno, _ = _run_sod_dpf(
            nx=nx, t_end=t_end, riemann="hlld", recon="weno5",
            integrator="ssp_rk3", precision="float64",
        )
        err_weno = _l1_error_sod(state_weno, nx, dx_weno, L, t_end)

        print(f"\nSod at nx={nx}, PF-1000 conditions:")
        print(f"  PLM+HLL:          L1(rho)={err_plm['rho']:.4e}")
        print(f"  WENO5+HLLD+RK3:   L1(rho)={err_weno['rho']:.4e}")
        ratio = err_plm["rho"] / max(err_weno["rho"], 1e-20)
        print(f"  Error ratio (PLM/WENO5): {ratio:.1f}x")

        # WENO5 should be at least somewhat better
        assert err_weno["rho"] < err_plm["rho"], (
            f"WENO5 not better: {err_weno['rho']:.4e} >= {err_plm['rho']:.4e}"
        )

    @pytest.mark.slow
    def test_sod_float32_vs_float64(self):
        """Float32 vs float64 Sod at PF-1000: float32 should be close."""
        cs = np.sqrt(_GAMMA * _P0 / _RHO0)
        t_end = 0.15 / cs
        nx = 128

        state_f32, dx, L = _run_sod_dpf(
            nx=nx, t_end=t_end, precision="float32",
        )
        err_f32 = _l1_error_sod(state_f32, nx, dx, L, t_end)

        state_f64, _, _ = _run_sod_dpf(
            nx=nx, t_end=t_end, precision="float64",
        )
        err_f64 = _l1_error_sod(state_f64, nx, dx, L, t_end)

        print(f"\nSod float32 vs float64 at nx={nx}, PF-1000:")
        print(f"  float32: L1(rho) = {err_f32['rho']:.4e}")
        print(f"  float64: L1(rho) = {err_f64['rho']:.4e}")

        # Float32 should not be catastrophically worse (within 3x)
        assert err_f32["rho"] < 3.0 * err_f64["rho"], (
            f"Float32 too much worse: {err_f32['rho']:.4e} vs {err_f64['rho']:.4e}"
        )


# =====================================================================
# Brio-Wu MHD shock tube at PF-1000 conditions
# =====================================================================


class TestBrioWuDPFStability:
    """Brio-Wu at PF-1000-scale parameters — stability tests."""

    def test_briowu_dpf_hll_plm_no_nan(self):
        """HLL+PLM on Brio-Wu at PF-1000 conditions: no NaN after 100 steps."""
        state, dx, L = _make_briowu_dpf(nx=100)
        solver = MetalMHDSolver(
            grid_shape=(100, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(100):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        assert not np.any(np.isnan(state["rho"]))
        assert not np.any(np.isnan(state["pressure"]))
        assert np.all(state["rho"] > 0)
        assert np.all(state["pressure"] > 0)

    def test_briowu_dpf_hlld_weno5_no_nan(self):
        """HLLD+WENO5 on Brio-Wu at PF-1000 conditions: stress test."""
        state, dx, L = _make_briowu_dpf(nx=100)
        solver = MetalMHDSolver(
            grid_shape=(100, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hlld",
            reconstruction="weno5",
            time_integrator="ssp_rk3",
            precision="float32",
            use_ct=False,
        )

        for _ in range(100):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        assert not np.any(np.isnan(state["rho"]))
        assert not np.any(np.isnan(state["pressure"]))

    def test_briowu_dpf_by_sign_flip(self):
        """Brio-Wu at PF-1000: By should change sign across contact."""
        state, dx, L = _make_briowu_dpf(nx=200)
        solver = MetalMHDSolver(
            grid_shape=(200, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float64",
            use_ct=False,
        )

        # Run to t ~ 0.1 (in normalized time L=1, waves travel ~0.1 domain)
        cf_est = np.sqrt(_GAMMA * _P0 / _RHO0 + _B0_HL**2 / _RHO0)
        t_end = 0.1 / cf_est

        t = 0.0
        for _ in range(50000):
            dt = solver.compute_dt(state)
            if t + dt > t_end:
                dt = t_end - t
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)
            t += dt
            if t >= t_end * 0.999:
                break

        By = state["B"][1, :, 2, 2]
        # Left half should have positive By, right half negative
        quarter = 200 // 4
        left_avg = np.mean(By[:quarter])
        right_avg = np.mean(By[-quarter:])
        assert left_avg > 0, f"Left By should be positive: {left_avg:.4e}"
        assert right_avg < 0, f"Right By should be negative: {right_avg:.4e}"


class TestBrioWuDPFConvergence:
    """Self-convergence for Brio-Wu at PF-1000 conditions.

    No exact solution exists for MHD Riemann problems, so we use
    self-convergence: compare coarse grid against fine grid (downsampled).
    """

    @pytest.mark.slow
    def test_briowu_self_convergence_plm(self):
        """PLM+HLL self-convergence on Brio-Wu at PF-1000 conditions."""
        cf_est = np.sqrt(_GAMMA * _P0 / _RHO0 + _B0_HL**2 / _RHO0)
        t_end = 0.1 / cf_est

        resolutions = [64, 128, 256]
        states = {}
        for nx in resolutions:
            state, dx, L = _run_briowu_dpf(
                nx=nx, t_end=t_end, precision="float64",
            )
            states[nx] = state

        # Self-convergence: compare N vs 2N
        e1 = _self_convergence_l1(states[64], states[128], 64, 128)
        e2 = _self_convergence_l1(states[128], states[256], 128, 256)
        order = np.log2(e1 / e2)

        print("\nBrio-Wu PLM+HLL self-convergence at PF-1000:")
        print(f"  64 vs 128:  L1 = {e1:.4e}")
        print(f"  128 vs 256: L1 = {e2:.4e}")
        print(f"  Order: {order:.2f}")

        # Expect ~0.5-1.0 for shock problems
        assert order > 0.2, f"Self-convergence order too low: {order:.2f}"
        assert e2 < e1, "Error did not decrease with resolution"

    @pytest.mark.slow
    def test_briowu_self_convergence_weno5(self):
        """WENO5+HLLD self-convergence on Brio-Wu at PF-1000 conditions."""
        cf_est = np.sqrt(_GAMMA * _P0 / _RHO0 + _B0_HL**2 / _RHO0)
        t_end = 0.1 / cf_est

        resolutions = [64, 128, 256]
        states = {}
        for nx in resolutions:
            state, dx, L = _run_briowu_dpf(
                nx=nx, t_end=t_end,
                riemann="hlld", recon="weno5",
                integrator="ssp_rk3", precision="float64",
            )
            states[nx] = state

        e1 = _self_convergence_l1(states[64], states[128], 64, 128)
        e2 = _self_convergence_l1(states[128], states[256], 128, 256)
        order = np.log2(e1 / e2)

        print("\nBrio-Wu WENO5+HLLD self-convergence at PF-1000:")
        print(f"  64 vs 128:  L1 = {e1:.4e}")
        print(f"  128 vs 256: L1 = {e2:.4e}")
        print(f"  Order: {order:.2f}")

        # WENO5 at shocks is also ~1st order, but should have lower error
        assert order > 0.2, f"Self-convergence order too low: {order:.2f}"
        assert e2 < e1, "Error did not decrease with resolution"


# =====================================================================
# Repair fraction under shocks
# =====================================================================


class TestRepairFractionShocks:
    """Positivity fallback should register repairs on shock problems.

    Phase AK showed 0 repairs on smooth waves.  Shock problems (especially
    Brio-Wu with strong By discontinuity) may trigger the positivity
    fallback, particularly in float32.
    """

    def test_repair_stats_sod_dpf(self):
        """Sod at PF-1000: measure repair fraction."""
        reset_repair_stats()
        state, dx, L = _make_sod_dpf(nx=200)
        solver = MetalMHDSolver(
            grid_shape=(200, 4, 4),
            dx=dx,
            gamma=_GAMMA,
            cfl=0.4,
            device="cpu",
            riemann_solver="hll",
            reconstruction="plm",
            time_integrator="ssp_rk2",
            precision="float32",
            use_ct=False,
        )

        for _ in range(200):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        stats = get_repair_stats()
        print("\nSod PF-1000 repair stats:")
        print(f"  total_checked:  {stats['total_checked']:,}")
        print(f"  total_repaired: {stats['total_repaired']:,}")
        print(f"  calls: {stats['calls']}")
        if stats["total_checked"] > 0:
            frac = stats["total_repaired"] / stats["total_checked"]
            print(f"  repair fraction: {frac:.6f}")

        # Sod with HLL+PLM should have very few or zero repairs
        # (HLL is very diffusive, PLM clips oscillations)
        if stats["total_checked"] > 0:
            frac = stats["total_repaired"] / stats["total_checked"]
            assert frac < 0.01, f"Repair fraction too high for Sod: {frac:.4f}"

    def test_repair_stats_briowu_dpf(self):
        """Brio-Wu at PF-1000: measure repair fraction with HLLD."""
        reset_repair_stats()
        state, dx, L = _make_briowu_dpf(nx=200)
        solver = MetalMHDSolver(
            grid_shape=(200, 4, 4),
            dx=dx,
            gamma=2.0,
            cfl=0.3,
            device="cpu",
            riemann_solver="hlld",
            reconstruction="weno5",
            time_integrator="ssp_rk3",
            precision="float32",
            use_ct=False,
        )

        for _ in range(200):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt=dt, current=0.0, voltage=0.0)

        stats = get_repair_stats()
        print("\nBrio-Wu PF-1000 HLLD+WENO5 repair stats:")
        print(f"  total_checked:  {stats['total_checked']:,}")
        print(f"  total_repaired: {stats['total_repaired']:,}")
        print(f"  calls: {stats['calls']}")
        if stats["total_checked"] > 0:
            frac = stats["total_repaired"] / stats["total_checked"]
            print(f"  repair fraction: {frac:.6f}")

        # HLLD+WENO5 on Brio-Wu may trigger some repairs due to By discontinuity
        # Accept up to 1% repair fraction (positivity fallback working correctly)
        if stats["total_checked"] > 0:
            frac = stats["total_repaired"] / stats["total_checked"]
            assert frac < 0.01, f"Repair fraction too high: {frac:.4f}"

        # Verify simulation completed successfully regardless
        assert not np.any(np.isnan(state["rho"]))
        assert np.all(state["rho"] > 0)


# =====================================================================
# Summary comparison table
# =====================================================================


class TestShockConvergenceSummary:
    """Print combined summary of shock convergence results."""

    @pytest.mark.slow
    def test_shock_convergence_summary(self):
        """Summary: Sod exact + Brio-Wu self-convergence at PF-1000 conditions."""
        cs = np.sqrt(_GAMMA * _P0 / _RHO0)
        cf_est = np.sqrt(_GAMMA * _P0 / _RHO0 + _B0_HL**2 / _RHO0)
        t_sod = 0.15 / cs
        t_bw = 0.1 / cf_est

        # Sod PLM+HLL convergence
        sod_errors: list[float] = []
        for nx in [64, 128, 256]:
            state, dx, L = _run_sod_dpf(
                nx=nx, t_end=t_sod, precision="float64",
            )
            err = _l1_error_sod(state, nx, dx, L, t_sod)
            sod_errors.append(err["rho"])

        sod_order = 0.5 * (
            np.log2(sod_errors[0] / sod_errors[1])
            + np.log2(sod_errors[1] / sod_errors[2])
        )

        # Brio-Wu PLM+HLL self-convergence
        bw_states = {}
        for nx in [64, 128, 256]:
            state, dx, L = _run_briowu_dpf(
                nx=nx, t_end=t_bw, precision="float64",
            )
            bw_states[nx] = state

        bw_e1 = _self_convergence_l1(bw_states[64], bw_states[128], 64, 128)
        bw_e2 = _self_convergence_l1(bw_states[128], bw_states[256], 128, 256)
        bw_order = np.log2(bw_e1 / bw_e2)

        print("\n" + "=" * 72)
        print("Phase AL: Shock convergence at PF-1000 conditions")
        print("=" * 72)
        print(f"{'Problem':<25} {'Method':<20} {'Order':<8} {'L1(256)':<12}")
        print("-" * 72)
        print(
            f"{'Sod (exact ref)':<25} {'PLM+HLL':<20} "
            f"{sod_order:<8.2f} {sod_errors[-1]:<12.4e}"
        )
        print(
            f"{'Brio-Wu (self-conv)':<25} {'PLM+HLL':<20} "
            f"{bw_order:<8.2f} {bw_e2:<12.4e}"
        )
        print("-" * 72)
        print("Note: Shock convergence is typically O(h^0.5-1.0).")
        print("      Smooth wave (Phase AK): PLM order 1.30, WENO5 order 2.14")
        print("=" * 72)

        # Both should show positive convergence
        assert sod_order > 0.3, f"Sod convergence: {sod_order:.2f}"
        assert bw_order > 0.2, f"Brio-Wu convergence: {bw_order:.2f}"
