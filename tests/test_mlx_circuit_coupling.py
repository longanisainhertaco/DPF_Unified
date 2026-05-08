"""Tests proving MHD solver feeds back into circuit via density-weighted Lp.

Demonstrates that the MLX MHD solver:
1. Propagates B_theta from electrode boundary conditions into the domain
2. Develops J×B-driven radial compression of the plasma
3. Computes plasma inductance from density-weighted Lee formula
4. Returns MHD Lp time series from the engine for coupling verification

These tests answer the "is the MHD physics real?" question: the solver
computes B-field propagation, density compression, and inductance from
first principles — not from the snowplow ODE.

References
----------
Lee & Saw, Phys. Plasmas 21, 072501 (2014) — Lee model inductance formula.
"""
from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="MLX not available")


@pytest.fixture
def pf1000_mhd_state():
    """Set up an MLX solver with PF-1000 geometry and run 50 steps."""
    from dpf.metal.mlx_solver import MLXMHDSolver
    from dpf.presets import get_preset

    preset = get_preset("pf1000")
    cc = preset["circuit"]
    sp_cfg = preset.get("snowplow", {})
    _kB = 1.380649e-23
    _m_D2 = 6.69e-27
    _p_Pa = sp_cfg.get("fill_pressure_Pa", 400.0)
    rho0 = _p_Pa * _m_D2 / (_kB * 300.0)

    r_anode = cc["anode_radius"]
    r_cathode = cc["cathode_radius"]
    nr, nz = 32, 64
    dr = (r_cathode - r_anode) / nr
    dz = sp_cfg.get("anode_length", 0.6) / nz

    solver = MLXMHDSolver(
        grid_shape=(nr, 1, nz), dx=dr, dz=dz,
        riemann_solver="hlls", reconstruction="plm",
        time_integrator="ssp_rk2", coordinates="cylindrical",
        r_inner=r_anode, cathode_radius=r_cathode,
        ion_mass=_m_D2 / 2.0,
    )

    state = {
        "rho": np.full((nr, 1, nz), rho0, dtype=np.float32),
        "velocity": np.zeros((3, nr, 1, nz), dtype=np.float32),
        "pressure": np.full((nr, 1, nz), _p_Pa, dtype=np.float32),
        "B": np.zeros((3, nr, 1, nz), dtype=np.float32),
        "Te": np.full((nr, 1, nz), 300.0, dtype=np.float32),
        "Ti": np.full((nr, 1, nz), 300.0, dtype=np.float32),
    }

    I_test = 500_000.0  # 500 kA
    for _ in range(50):
        dt = solver._compute_dt(state)
        state = solver.step(
            state, dt, current=I_test, voltage=20000.0,
            apply_electrode_bc=True,
        )

    return solver, state, rho0


class TestBThetaPropagation:
    """Verify electrode B_theta propagates into the MHD domain."""

    def test_btheta_nonzero_in_interior(self, pf1000_mhd_state):
        solver, state, _ = pf1000_mhd_state
        bt = state["B"][2, :, 0, :]  # B_theta component
        # B_theta should have propagated from cathode into interior cells
        assert np.max(np.abs(bt)) > 0.01, "B_theta never propagated from electrode BC"

    def test_btheta_increases_inward(self, pf1000_mhd_state):
        """B_theta ~ 1/r should be stronger at smaller radius."""
        _, state, _ = pf1000_mhd_state
        bt = state["B"][2, :, 0, :]
        mid_z = bt.shape[1] // 2
        bt_profile = np.abs(bt[:, mid_z])
        outer_avg = np.mean(bt_profile[-8:])
        inner_avg = np.mean(bt_profile[:8])
        assert inner_avg > outer_avg, (
            f"B_theta not stronger at smaller radius: inner={inner_avg:.4f} outer={outer_avg:.4f}"
        )


class TestJxBCompression:
    """Verify J x B force drives radial density compression."""

    def test_density_not_uniform(self, pf1000_mhd_state):
        _, state, rho0 = pf1000_mhd_state
        rho = state["rho"][:, 0, :]
        rho_range = np.max(rho) - np.min(rho)
        assert rho_range > 0.01 * rho0, (
            f"Density range {rho_range:.3e} too small vs initial {rho0:.3e}"
        )

    def test_radial_velocity_inward(self, pf1000_mhd_state):
        """J x B force (B_theta gradient) drives gas radially inward."""
        _, state, _ = pf1000_mhd_state
        vr = state["velocity"][0, :, 0, :]
        assert np.min(vr) < -1.0, (
            f"No inward radial velocity: min(vr)={np.min(vr):.2f} m/s"
        )


class TestCouplingInterface:
    """Verify coupling_interface returns physically meaningful Lp."""

    def test_coupling_lp_positive(self, pf1000_mhd_state):
        solver, _, _ = pf1000_mhd_state
        coupling = solver.coupling_interface()
        assert coupling.Lp >= 0, f"Negative Lp: {coupling.Lp}"

    def test_coupling_has_current(self, pf1000_mhd_state):
        solver, _, _ = pf1000_mhd_state
        coupling = solver.coupling_interface()
        assert abs(coupling.current) > 0, "Coupling current is zero"


class TestEngineCircuitFeedback:
    """Verify the mlx_engine MHD mode produces MHD Lp time series."""

    def test_mhd_mode_returns_lp_mhd_series(self):
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge(
            "pf1000", max_steps=200, mode="mhd", grid_shape=(16, 1, 32),
        )
        assert "Lp_mhd_nH" in r, "Missing Lp_mhd_nH in engine output"
        mhd_lps = r["Lp_mhd_nH"]
        assert len(mhd_lps) == r["n_steps"]

    def test_mhd_lp_nonzero(self):
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge(
            "pf1000", max_steps=200, mode="mhd", grid_shape=(16, 1, 32),
        )
        mhd_lps = r["Lp_mhd_nH"]
        nonzero = [x for x in mhd_lps if x > 0.001]
        assert len(nonzero) > 0, "MHD Lp never became nonzero"

    def test_mhd_lp_is_not_snowplow_alias(self):
        """MHD Lp must be computed from MHD state, not copied from snowplow."""
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge(
            "pf1000", max_steps=200, mode="mhd", grid_shape=(16, 1, 32),
        )
        snowplow_lps = np.asarray(r["Lp_snowplow_nH"])
        mhd_lps = np.asarray(r["Lp_mhd_nH"])
        assert len(snowplow_lps) == len(mhd_lps) == r["n_steps"]
        assert np.any(np.abs(mhd_lps - snowplow_lps) > 1.0e-6), (
            "MHD Lp series is identical to snowplow Lp; coupling signal is aliased"
        )

    def test_lee_mode_no_mhd_overhead(self):
        """Lee mode should still work and have zero MHD Lp."""
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge("pf1000", max_steps=200, mode="lee")
        mhd_lps = r["Lp_mhd_nH"]
        assert all(x == 0.0 for x in mhd_lps), "Lee mode should have zero MHD Lp"

    def test_engine_reports_coupling_source(self):
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge(
            "pf1000", max_steps=200, mode="mhd", grid_shape=(16, 1, 32),
        )
        assert "coupling_source" in r
        assert len(r["coupling_source"]) == r["n_steps"]

    def test_blend_alpha_in_output(self):
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge(
            "pf1000", max_steps=200, mode="mhd", grid_shape=(16, 1, 32),
        )
        assert "blend_alpha" in r, "Missing blend_alpha in engine output"
        assert 0.0 <= r["blend_alpha"] <= 1.0


class TestSheathDetectionBennettPinch:
    """Verify update_coupling on synthetic Bennett pinch with known L_p.

    Root cause fix for MHD-driven I_peak error: the sheath detection
    algorithm previously used column density argmax, which was dominated
    by large-r fill-gas cells and picked the fill/vacuum boundary instead
    of the compressed sheath.  One wrong z-slice shifted r_eff from 7 to
    19 mm and L_p error from 4.5% to 27%.

    Fix: use on-axis density profile for sheath detection.

    References
    ----------
    Lee & Saw, J. Fusion Energy 33:319 (2014), Eq. L_p = (mu0/2pi)*z*ln(b/r_p).
    Bennett pinch: rho(r) = rho0 / (1 + (r/a)^2)^2, analytic r_eff = a*pi/2.
    """

    def test_bennett_pinch_lp_within_10_percent(self):
        """L_p from update_coupling on Bennett pinch matches analytic to <10%."""
        import math

        from dpf.metal.mlx_coupling import update_coupling
        from dpf.metal.mlx_grid import CylindricalGrid

        nr, nz = 64, 128
        dr, dz = 0.005, 0.005
        a_pinch = 0.005  # 5 mm
        rho_0 = 1e-3
        cathode_radius = 0.160
        MU_0 = 4e-7 * math.pi

        r_arr = (np.arange(nr) + 0.5) * dr
        rho_2d = np.zeros((nr, nz), dtype=np.float32)
        iz_sheath = 60  # z_sheath = 302.5 mm
        for iz in range(iz_sheath + 1):
            rho_2d[:, iz] = rho_0 / (1.0 + (r_arr / a_pinch) ** 2) ** 2
        rho_2d[:, iz_sheath + 1 :] = 1e-6

        U_np = np.zeros((10, nr, nz), dtype=np.float32)
        U_np[0] = rho_2d
        U = mlx.array(U_np)
        grid = CylindricalGrid(nr, nz, dr, dz, r_inner=0.0)

        coupling, _, _ = update_coupling(
            U,
            current=1e6,
            voltage=1e4,
            dt=1e-9,
            grid=grid,
            cathode_radius=cathode_radius,
            r_inner=0.0,
            prev_Lp=0.0,
            Lp_max=0.0,
            coordinates="cylindrical",
            Lp_history=[],
            sim_time=1e-6,
        )

        # Analytic: r_eff = a*pi/2, L_p = (mu0/2pi)*z*ln(b/r_eff)
        r_eff_analytic = a_pinch * math.pi / 2
        z_sheath_m = (iz_sheath + 0.5) * dz
        Lp_analytic = (
            (MU_0 / (2 * math.pi))
            * z_sheath_m
            * math.log(cathode_radius / r_eff_analytic)
        )

        rel_error = abs(coupling.Lp - Lp_analytic) / Lp_analytic
        assert rel_error < 0.10, (
            f"Bennett pinch L_p error {rel_error:.1%}: "
            f"code={coupling.Lp * 1e9:.1f} nH vs analytic={Lp_analytic * 1e9:.1f} nH"
        )
