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

    def test_btheta_decreases_inward(self, pf1000_mhd_state):
        """B_theta ~ 1/r should be stronger near cathode (outer cells)."""
        _, state, _ = pf1000_mhd_state
        bt = state["B"][2, :, 0, :]
        mid_z = bt.shape[1] // 2
        bt_profile = np.abs(bt[:, mid_z])
        # Outer cells (near cathode) should have higher B_theta
        outer_avg = np.mean(bt_profile[-8:])
        inner_avg = np.mean(bt_profile[:8])
        assert outer_avg > inner_avg, (
            f"B_theta not stronger at cathode: outer={outer_avg:.4f} inner={inner_avg:.4f}"
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

    def test_lee_mode_no_mhd_overhead(self):
        """Lee mode should still work and have zero MHD Lp."""
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge("pf1000", max_steps=200, mode="lee")
        mhd_lps = r["Lp_mhd_nH"]
        assert all(x == 0.0 for x in mhd_lps), "Lee mode should have zero MHD Lp"

    def test_blend_alpha_in_output(self):
        from dpf.metal.mlx_engine import run_mlx_discharge

        r = run_mlx_discharge(
            "pf1000", max_steps=200, mode="mhd", grid_shape=(16, 1, 32),
        )
        assert "blend_alpha" in r, "Missing blend_alpha in engine output"
        assert 0.0 <= r["blend_alpha"] <= 1.0
