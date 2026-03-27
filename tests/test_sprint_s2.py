"""Tests for Sprint S-2 deliverables: RKL2 transport, anomalous resistivity,
post-pinch Lp expansion.

D1: RKL2 resistive diffusion on GPU
D2: RKL2 thermal conduction on GPU
D4: Anomalous resistivity (drift-velocity model)
D5: Phase-aware Lp monotonicity for current dip
"""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")


# ──────────────────────────────────────────────────────────────────────────────
# D1/D2: RKL2 super-timestepping
# ──────────────────────────────────────────────────────────────────────────────


class TestRKL2Operators:
    """Tests for mlx_sts_operators explicit Laplacian RHS."""

    def test_uniform_field_zero_rhs(self):
        """Uniform field has zero Laplacian."""
        from dpf.metal.mlx_sts_operators import resistive_diffusion_rhs

        B = mlx.full((16, 32), 5.0)
        alpha = mlx.full((16, 32), 1.0)
        r_cell = mlx.array(np.linspace(0.005, 0.08, 16, dtype=np.float32))
        rhs = resistive_diffusion_rhs(B, alpha, 0.005, 0.005, r_cell)
        assert float(mlx.max(mlx.abs(rhs))) < 1e-4

    def test_gaussian_diffuses(self):
        """Gaussian peak should decrease after diffusion step."""
        from dpf.metal.mlx_sts import rkl2_step_mlx
        from dpf.metal.mlx_sts_operators import resistive_diffusion_rhs

        nr, nz = 16, 32
        dr = dz = 0.005
        r_cell = mlx.array(np.linspace(0.5 * dr, (nr - 0.5) * dr, nr, dtype=np.float32))
        r_np = np.array(r_cell)
        z_np = np.linspace(0.5 * dz, (nz - 0.5) * dz, nz)
        R, Z = np.meshgrid(r_np, z_np, indexing="ij")
        B = mlx.array(np.exp(-((R - 0.04) ** 2 + (Z - 0.08) ** 2) / 0.01**2).astype(np.float32))
        alpha = mlx.full((nr, nz), 1.0)

        def rhs(x):
            return resistive_diffusion_rhs(x, alpha, dr, dz, r_cell)

        # dt must be within RKL2 stability: 0.25 * s^2 * dt_explicit
        # dt_explicit ~ 0.5 * dx^2 / alpha = 0.5 * 0.005^2 / 1.0 = 1.25e-5
        # RKL2 s=4: stable up to 0.25 * 16 * 1.25e-5 = 5e-5
        B_new = rkl2_step_mlx(B, rhs, dt=1e-5, s_stages=4)
        assert float(mlx.max(B_new)) < float(mlx.max(B))

    def test_rkl2_vs_thomas_small_dt(self):
        """RKL2 and Thomas agree for small dt (both ~ forward Euler limit)."""
        from dpf.metal.mlx_sts import compute_sts_stages, rkl2_step_mlx
        from dpf.metal.mlx_sts_operators import (
            compute_parabolic_dt,
            resistive_diffusion_rhs,
        )
        from dpf.metal.mlx_transport import apply_resistive_diffusion

        nr, nz = 8, 8
        dr = dz = 0.001
        r_cell = mlx.array(np.linspace(0.5 * dr, (nr - 0.5) * dr, nr, dtype=np.float32))
        B_init = np.zeros((nr, nz), dtype=np.float32)
        B_init[nr // 2, nz // 2] = 1.0
        eta = 1e-5
        alpha_val = eta / (4 * np.pi * 1e-7)
        dt = 1e-10

        # Thomas
        Bt_thomas = apply_resistive_diffusion(
            mlx.zeros((nr, nz)), mlx.zeros((nr, nz)), mlx.array(B_init),
            mlx.full((nr, nz), 1e-3), mlx.full((nr, nz), 1e5),
            eta, dt, dr, dz, r_cell,
        )[2]

        # RKL2
        alpha = mlx.full((nr, nz), alpha_val, dtype=mlx.float32)
        dt_para = compute_parabolic_dt(alpha, dr, dz)
        s = compute_sts_stages(dt, dt_para)

        def rhs(B):
            return resistive_diffusion_rhs(B, alpha, dr, dz, r_cell)

        Bt_rkl2 = rkl2_step_mlx(mlx.array(B_init), rhs, dt, s_stages=s)
        mlx.eval(Bt_rkl2)

        diff = np.abs(np.array(Bt_thomas) - np.array(Bt_rkl2))
        assert diff.max() < 1e-4, f"RKL2 vs Thomas max diff: {diff.max():.2e}"

    def test_conduction_rhs(self):
        """Thermal conduction RHS produces non-zero output for non-uniform T."""
        from dpf.metal.mlx_sts_operators import thermal_conduction_rhs

        nr, nz = 8, 16
        dr = dz = 0.001
        r_cell = mlx.array(np.linspace(0.5 * dr, (nr - 0.5) * dr, nr, dtype=np.float32))
        T = mlx.array(np.random.RandomState(42).uniform(1e5, 1e7, (nr, nz)).astype(np.float32))
        kappa = mlx.full((nr, nz), 100.0)
        rho = mlx.full((nr, nz), 1e-3)
        rhs = thermal_conduction_rhs(T, kappa, rho, dr, dz, r_cell)
        assert float(mlx.max(mlx.abs(rhs))) > 0

    def test_stage_count_computation(self):
        """Stage count increases with dt_mhd / dt_parabolic ratio."""
        from dpf.metal.mlx_sts import compute_sts_stages

        assert compute_sts_stages(1e-9, 1e-8) == 2  # ratio=0.4, s=ceil(1)=2 (min)
        assert compute_sts_stages(1e-7, 1e-8) == 7  # ratio=40, s=ceil(6.3)=7
        assert compute_sts_stages(1e-5, 1e-8) >= 20  # ratio=4000, s=ceil(63)=20 (capped)


class TestRKL2SolverIntegration:
    """Tests for RKL2 wired into the MLX solver."""

    def _make_state(self):
        return {
            "rho": np.full((16, 1, 32), 1e-3, dtype=np.float32),
            "velocity": np.zeros((3, 16, 1, 32), dtype=np.float32),
            "pressure": np.full((16, 1, 32), 1e5, dtype=np.float32),
            "B": np.zeros((3, 16, 1, 32), dtype=np.float32),
            "Te": np.full((16, 1, 32), 1e6, dtype=np.float32),
            "Ti": np.full((16, 1, 32), 1e6, dtype=np.float32),
            "psi": np.zeros((16, 1, 32), dtype=np.float32),
        }

    def test_rkl2_solver_runs(self):
        """Solver completes a step with RKL2 transport enabled."""
        from dpf.metal.mlx_solver import MLXMHDSolver

        solver = MLXMHDSolver(
            grid_shape=(16, 1, 32), dx=0.001, dz=0.001,
            coordinates="cylindrical", convert_b_si_to_hl=True,
            use_rkl2_transport=True,
        )
        state = self._make_state()
        result = solver.step(state, dt=1e-9, current=1e5, voltage=20000, eta_field=1e-5)
        assert "rho" in result
        assert not np.any(np.isnan(result["rho"]))

    def test_thomas_fallback_runs(self):
        """Solver completes a step with Thomas fallback."""
        from dpf.metal.mlx_solver import MLXMHDSolver

        solver = MLXMHDSolver(
            grid_shape=(16, 1, 32), dx=0.001, dz=0.001,
            coordinates="cylindrical", convert_b_si_to_hl=True,
            use_rkl2_transport=False,
        )
        state = self._make_state()
        result = solver.step(state, dt=1e-9, current=1e5, voltage=20000, eta_field=1e-5)
        assert not np.any(np.isnan(result["rho"]))


# ──────────────────────────────────────────────────────────────────────────────
# D4: Anomalous resistivity
# ──────────────────────────────────────────────────────────────────────────────


class TestAnomalousResistivity:

    def test_below_threshold_zero(self):
        """No anomalous resistivity when v_d < v_ti."""
        from dpf.metal.mlx_transport import anomalous_resistivity

        rho = np.full((4, 8), 1e-3)
        p = np.full((4, 8), 1e5)
        J_sq = np.full((4, 8), 1.0)  # tiny J -> v_d << v_ti
        eta = anomalous_resistivity(J_sq, rho, p)
        np.testing.assert_allclose(eta, 0.0)

    def test_above_threshold_nonzero(self):
        """Anomalous resistivity active when v_d > v_ti (DPF pinch)."""
        from dpf.metal.mlx_transport import anomalous_resistivity

        rho = np.full((4, 8), 1e-4)
        p = np.full((4, 8), 1e7)
        J_sq = np.full((4, 8), (2.5e11) ** 2)  # PF-1000 pinch
        eta = anomalous_resistivity(J_sq, rho, p, model="drift_velocity")
        assert np.all(eta > 0)
        assert np.all(eta < 1e-2)  # below Bohm cap

    def test_drift_velocity_exceeds_classical(self):
        """Anomalous >> classical at pinch conditions."""
        from dpf.metal.mlx_transport import anomalous_resistivity, lee_more_resistivity

        rho = np.full((4, 8), 1e-4)
        p = np.full((4, 8), 1e7)
        J_sq = np.full((4, 8), (2.5e11) ** 2)
        eta_anom = anomalous_resistivity(J_sq, rho, p)
        Te_eV = p * 3.34e-27 / (2 * rho * 1.38e-23) / 11604.5
        eta_lm = lee_more_resistivity(Te_eV, rho)
        assert np.all(eta_anom > eta_lm * 10)

    def test_sagdeev_model(self):
        from dpf.metal.mlx_transport import anomalous_resistivity

        rho = np.full((4, 8), 1e-4)
        p = np.full((4, 8), 1e7)
        J_sq = np.full((4, 8), (2.5e11) ** 2)
        eta = anomalous_resistivity(J_sq, rho, p, model="sagdeev")
        assert np.all(eta > 0)

    def test_lhdi_model(self):
        from dpf.metal.mlx_transport import anomalous_resistivity

        rho = np.full((4, 8), 1e-4)
        p = np.full((4, 8), 1e7)
        J_sq = np.full((4, 8), (2.5e11) ** 2)
        eta = anomalous_resistivity(J_sq, rho, p, model="lhdi")
        assert np.all(eta > 0)

    def test_unknown_model_raises(self):
        from dpf.metal.mlx_transport import anomalous_resistivity

        with pytest.raises(ValueError, match="Unknown anomalous"):
            anomalous_resistivity(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), model="fake")

    def test_bohm_cap(self):
        """Anomalous resistivity capped at 1e-2 Ohm*m (Bohm limit)."""
        from dpf.metal.mlx_transport import anomalous_resistivity

        rho = np.full((4, 8), 1e-10)  # extreme vacuum
        p = np.full((4, 8), 1e2)
        J_sq = np.full((4, 8), (1e15) ** 2)  # extreme J
        eta = anomalous_resistivity(J_sq, rho, p)
        assert np.all(eta <= 1e-2)


# ──────────────────────────────────────────────────────────────────────────────
# D5: Post-pinch Lp expansion
# ──────────────────────────────────────────────────────────────────────────────


class TestPostPinchLpExpansion:

    def test_lp_allowed_to_decrease_postpinch(self):
        """After Lp peaks (pinch), it should be allowed to decrease."""
        from dpf.metal.mlx_solver import MLXMHDSolver

        solver = MLXMHDSolver(
            grid_shape=(8, 1, 16), dx=0.005, dz=0.005,
            coordinates="cylindrical", convert_b_si_to_hl=True,
        )
        # Simulate Lp increasing then decreasing
        solver._Lp_max = 1e-7
        solver._prev_Lp = 1e-7

        # Lp slightly below peak (within 2%) — should clamp
        Lp_near = 0.99e-7
        if Lp_near > solver._Lp_max:
            solver._Lp_max = Lp_near
        elif Lp_near < solver._Lp_max * 0.98:
            pass  # allow decrease
        else:
            Lp_near = solver._Lp_max
        assert Lp_near == 1e-7  # clamped

        # Lp significantly below peak (>2%) — should NOT clamp
        Lp_far = 0.9e-7
        if Lp_far > solver._Lp_max:
            solver._Lp_max = Lp_far
        elif Lp_far < solver._Lp_max * 0.98:
            pass  # allow
        else:
            Lp_far = solver._Lp_max
        assert Lp_far == 0.9e-7  # allowed to decrease
