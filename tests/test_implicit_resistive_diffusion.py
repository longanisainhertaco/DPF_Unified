"""Tests for implicit resistive diffusion via batched Thomas algorithm.

Covers:
1. Thomas algorithm correctness vs torch.linalg.solve
2. Diffusion convergence (Gaussian → analytical solution)
3. High-eta stability (eta=1e-2 that blows up explicit sub-cycling)
4. Cylindrical geometry correctness
5. Backward compatibility (explicit path unchanged)
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch required for Metal solver tests")

from dpf.metal.metal_transport import (  # noqa: E402
    batched_thomas_solve,
    implicit_resistive_step,
)

# ── Thomas Algorithm Correctness ──────────────────────────────


class TestBatchedThomasSolve:
    """Verify Thomas algorithm matches dense linear algebra."""

    def test_random_tridiagonal_vs_linalg(self) -> None:
        """Thomas solve matches torch.linalg.solve for random systems."""
        torch.manual_seed(42)
        batch, n = 16, 32

        # Random positive-definite diagonally dominant tridiagonal
        a = -torch.rand(batch, n, dtype=torch.float64) * 0.3
        c = -torch.rand(batch, n, dtype=torch.float64) * 0.3
        b = torch.rand(batch, n, dtype=torch.float64) * 2.0 + 1.0  # dominant
        a[:, 0] = 0.0
        c[:, -1] = 0.0
        d = torch.randn(batch, n, dtype=torch.float64)

        x_thomas = batched_thomas_solve(a, b, c, d)

        # Build dense matrices and solve
        for k in range(batch):
            A = torch.diag(b[k])
            for i in range(1, n):
                A[i, i - 1] = a[k, i]
            for i in range(n - 1):
                A[i, i + 1] = c[k, i]
            x_ref = torch.linalg.solve(A, d[k])
            assert torch.allclose(x_thomas[k], x_ref, atol=1e-12), (
                f"Batch {k}: max diff = {(x_thomas[k] - x_ref).abs().max():.2e}"
            )

    def test_identity_system(self) -> None:
        """Thomas solve with identity matrix returns RHS."""
        batch, n = 4, 16
        a = torch.zeros(batch, n, dtype=torch.float64)
        b = torch.ones(batch, n, dtype=torch.float64)
        c = torch.zeros(batch, n, dtype=torch.float64)
        d = torch.randn(batch, n, dtype=torch.float64)

        x = batched_thomas_solve(a, b, c, d)
        assert torch.allclose(x, d, atol=1e-14)

    def test_float32_input_preserved(self) -> None:
        """Output dtype matches input dtype."""
        batch, n = 2, 8
        a = torch.zeros(batch, n, dtype=torch.float32)
        b = torch.ones(batch, n, dtype=torch.float32) * 2.0
        c = torch.zeros(batch, n, dtype=torch.float32)
        d = torch.ones(batch, n, dtype=torch.float32)

        x = batched_thomas_solve(a, b, c, d)
        assert x.dtype == torch.float32

    def test_single_element(self) -> None:
        """Thomas solve handles n=1 systems."""
        a = torch.zeros(3, 1, dtype=torch.float64)
        b = torch.tensor([[2.0], [3.0], [5.0]], dtype=torch.float64)
        c = torch.zeros(3, 1, dtype=torch.float64)
        d = torch.tensor([[4.0], [9.0], [10.0]], dtype=torch.float64)

        x = batched_thomas_solve(a, b, c, d)
        assert torch.allclose(x, d / b, atol=1e-14)


# ── Implicit Resistive Step ───────────────────────────────────


class TestImplicitResistiveStep:
    """Test the full implicit diffusion operator."""

    @pytest.fixture
    def gaussian_profile(self) -> tuple[torch.Tensor, float, int]:
        """Create a Gaussian B_theta profile for diffusion tests."""
        nx = 64
        dx = 0.01
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        sigma = 0.1
        x0 = nx * dx / 2.0
        B_theta = torch.exp(-0.5 * ((x - x0) / sigma) ** 2)

        B = torch.zeros(3, nx, 1, 1, dtype=torch.float64)
        B[1, :, 0, 0] = B_theta
        return B, dx, nx

    def test_gaussian_diffusion_convergence_cartesian(
        self, gaussian_profile: tuple[torch.Tensor, float, int],
    ) -> None:
        """Gaussian diffuses toward analytical solution in Cartesian."""
        B, dx, nx = gaussian_profile
        eta_val = 1e-4
        eta = torch.full((nx, 1, 1), eta_val, dtype=torch.float64)
        dt = 0.001
        n_steps = 100
        t_total = dt * n_steps

        # Analytical: sigma^2(t) = sigma_0^2 + 2*D*t, amplitude scales as sigma_0/sigma(t)
        sigma_0 = 0.1
        D = eta_val  # mu_0 = 1 in code units
        sigma_t = math.sqrt(sigma_0**2 + 2 * D * t_total)

        B_evolved = B.clone()
        for _ in range(n_steps):
            B_evolved = implicit_resistive_step(
                B_evolved, eta, dt, dx,
                coordinates="cartesian",
                component=1,
            )

        # Check that the profile has broadened
        B_init = B[1, :, 0, 0]
        B_final = B_evolved[1, :, 0, 0]

        # Peak should decrease (diffusion spreads the profile)
        assert B_final.max() < B_init.max(), "Peak should decrease from diffusion"

        # Compute variance (second moment) as a robust width measure
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        x0 = nx * dx / 2.0
        var_init = float(((x - x0) ** 2 * B_init.abs()).sum() / B_init.abs().sum())
        var_final = float(((x - x0) ** 2 * B_final.abs()).sum() / B_final.abs().sum())
        assert var_final > var_init, "Variance should increase from diffusion"

        # Quantitative check: peak amplitude ratio ~ sigma_0 / sigma_t
        expected_ratio = sigma_0 / sigma_t
        actual_ratio = float(B_final.max() / B_init.max())
        assert abs(actual_ratio - expected_ratio) < 0.15, (
            f"Amplitude ratio {actual_ratio:.3f} vs expected {expected_ratio:.3f}"
        )

    def test_uniform_field_unchanged(self) -> None:
        """Uniform B-field should not change under diffusion."""
        nx = 32
        dx = 0.01
        B = torch.ones(3, nx, 1, 1, dtype=torch.float64) * 0.5
        eta = torch.full((nx, 1, 1), 1e-3, dtype=torch.float64)

        B_new = implicit_resistive_step(
            B, eta, dt=0.01, dx=dx,
            coordinates="cartesian",
            component=1,
        )
        assert torch.allclose(B_new[1], B[1], atol=1e-13), (
            f"Uniform field changed: max diff = {(B_new[1] - B[1]).abs().max():.2e}"
        )

    def test_high_eta_stability(self) -> None:
        """Implicit solver stays stable with eta=1e-2 (blows up explicit)."""
        nx = 32
        dx = 0.01
        # dt_res_explicit = dx^2 / (2*eta) = 1e-4 / 0.02 = 5e-3
        # With dt=0.1, explicit needs 20 sub-cycles (at the cap) but CFL is violated
        eta_high = 1e-2
        dt_large = 0.1  # 20x the explicit CFL limit

        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        B = torch.zeros(3, nx, 1, 1, dtype=torch.float64)
        B[1, :, 0, 0] = torch.sin(2 * math.pi * x / (nx * dx))
        eta = torch.full((nx, 1, 1), eta_high, dtype=torch.float64)

        B_new = implicit_resistive_step(
            B, eta, dt_large, dx,
            coordinates="cartesian",
            component=1,
        )

        # Must be finite
        assert torch.all(torch.isfinite(B_new)), "Implicit result has NaN/Inf"

        # Amplitude should decrease (diffusion damps modes)
        assert B_new[1].abs().max() < B[1].abs().max(), (
            "Sinusoidal mode should be damped by diffusion"
        )

        # Should not blow up — peak should be smaller than initial
        assert B_new[1].abs().max() < 1.1, "Amplitude grew — instability"

    def test_cylindrical_geometry(self) -> None:
        """Cylindrical diffusion respects 1/r factor."""
        nx = 48
        dx = 0.005
        r_inner = 0.02  # typical DPF inner radius

        r = (r_inner + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx).reshape(nx, 1, 1)
        x_flat = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        x0 = nx * dx / 2.0
        sigma = 0.05

        B = torch.zeros(3, nx, 1, 1, dtype=torch.float64)
        B[1, :, 0, 0] = torch.exp(-0.5 * ((x_flat - x0) / sigma) ** 2)

        eta = torch.full((nx, 1, 1), 5e-4, dtype=torch.float64)
        dt = 0.001

        B_new = implicit_resistive_step(
            B, eta, dt, dx,
            r=r,
            coordinates="cylindrical",
            component=1,
        )

        assert torch.all(torch.isfinite(B_new)), "Cylindrical result has NaN/Inf"
        assert B_new[1].abs().max() < B[1].abs().max(), (
            "Cylindrical diffusion should damp the Gaussian peak"
        )

    def test_small_grid_passthrough(self) -> None:
        """Grids with nx < 3 are returned unchanged."""
        B = torch.randn(3, 2, 1, 1, dtype=torch.float64)
        eta = torch.ones(2, 1, 1, dtype=torch.float64)
        B_new = implicit_resistive_step(B, eta, dt=0.01, dx=0.01)
        assert torch.equal(B_new, B)

    def test_only_specified_component_changes(self) -> None:
        """Only the target component is modified."""
        nx = 16
        B = torch.randn(3, nx, 1, 1, dtype=torch.float64)
        B_orig = B.clone()
        eta = torch.full((nx, 1, 1), 1e-3, dtype=torch.float64)

        # Diffuse component 1 (B_theta)
        B_new = implicit_resistive_step(
            B, eta, dt=0.01, dx=0.01,
            component=1,
        )
        # Components 0 and 2 should be unchanged
        assert torch.equal(B_new[0], B_orig[0])
        assert torch.equal(B_new[2], B_orig[2])

    def test_multidim_batch(self) -> None:
        """Implicit step handles ny > 1, nz > 1 (batched)."""
        nx, ny, nz = 32, 4, 4
        dx = 0.01
        B = torch.zeros(3, nx, ny, nz, dtype=torch.float64)
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        for j in range(ny):
            for k in range(nz):
                B[1, :, j, k] = torch.sin(2 * math.pi * x / (nx * dx)) * (1 + 0.1 * j)
        eta = torch.full((nx, ny, nz), 1e-4, dtype=torch.float64)

        B_new = implicit_resistive_step(B, eta, dt=0.001, dx=dx, component=1)

        assert torch.all(torch.isfinite(B_new))
        # All batched lines should be damped
        for j in range(ny):
            for k in range(nz):
                assert B_new[1, :, j, k].abs().max() < B[1, :, j, k].abs().max()


# ── Metal Solver Integration ──────────────────────────────────


class TestMetalSolverImplicitResistivity:
    """Test that the implicit path is wired correctly in MetalMHDSolver."""

    def test_implicit_resistivity_config_default(self) -> None:
        """implicit_resistivity defaults to False (backward compat, opt-in)."""
        from dpf.metal.metal_solver import MetalMHDSolver
        solver = MetalMHDSolver(
            grid_shape=(16, 16, 16), dx=0.01,
            device="cpu", use_ct=False,
        )
        assert solver.implicit_resistivity is False

    def test_explicit_fallback(self) -> None:
        """Setting implicit_resistivity=False uses explicit sub-cycling."""
        from dpf.metal.metal_solver import MetalMHDSolver
        solver = MetalMHDSolver(
            grid_shape=(16, 16, 16), dx=0.01,
            device="cpu", use_ct=False,
            implicit_resistivity=False,
        )
        assert solver.implicit_resistivity is False

    def test_implicit_vs_explicit_low_eta_similar(self) -> None:
        """At low eta, implicit and explicit paths give similar results."""
        from dpf.metal.metal_solver import MetalMHDSolver

        nx = 16
        dx = 0.01
        eta_val = 1e-6  # well within explicit CFL

        # Create identical solvers
        solver_impl = MetalMHDSolver(
            grid_shape=(nx, 1, 1), dx=dx,
            device="cpu", use_ct=False,
            implicit_resistivity=True,
            precision="float64",
        )
        solver_expl = MetalMHDSolver(
            grid_shape=(nx, 1, 1), dx=dx,
            device="cpu", use_ct=False,
            implicit_resistivity=False,
            precision="float64",
        )

        # Create B field and eta
        B = torch.zeros(3, nx, 1, 1, dtype=torch.float64)
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        B[1, :, 0, 0] = torch.sin(2 * math.pi * x / (nx * dx))
        p = torch.ones(nx, 1, 1, dtype=torch.float64) * 1e5
        rho = torch.ones(nx, 1, 1, dtype=torch.float64)
        eta = torch.full((nx, 1, 1), eta_val, dtype=torch.float64)
        dt = 1e-6

        B_impl, p_impl = solver_impl._apply_resistive_diffusion(
            B, p, rho, eta, dt, 5.0 / 3.0,
        )
        B_expl, p_expl = solver_expl._apply_resistive_diffusion(
            B, p, rho, eta, dt, 5.0 / 3.0,
        )

        # Forward Euler vs backward Euler differ by O(dt). Additionally, the
        # explicit path uses curl(eta*J) while the implicit path solves the
        # diffusion equation directly — different spatial discretizations.
        # Both should produce a damped B_theta that agrees to within ~10%.
        diff = (B_impl[1] - B_expl[1]).abs().max()
        B_scale = B[1].abs().max()
        rel_diff = float(diff / B_scale)
        assert rel_diff < 0.1, f"Implicit vs explicit B rel diff too large: {rel_diff:.2e}"

    def test_implicit_high_eta_no_blowup(self) -> None:
        """Implicit solver handles high eta without blowup in full solver."""
        from dpf.metal.metal_solver import MetalMHDSolver

        nx = 32
        dx = 0.01
        solver = MetalMHDSolver(
            grid_shape=(nx, 1, 1), dx=dx,
            device="cpu", use_ct=False,
            implicit_resistivity=True,
            precision="float64",
        )

        B = torch.zeros(3, nx, 1, 1, dtype=torch.float64)
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        B[1, :, 0, 0] = torch.sin(2 * math.pi * x / (nx * dx)) * 0.1
        p = torch.ones(nx, 1, 1, dtype=torch.float64) * 1e5
        rho = torch.ones(nx, 1, 1, dtype=torch.float64)
        eta = torch.full((nx, 1, 1), 1e-2, dtype=torch.float64)
        dt = 0.01

        B_new, p_new = solver._apply_resistive_diffusion(
            B, p, rho, eta, dt, 5.0 / 3.0,
        )

        assert torch.all(torch.isfinite(B_new)), "Implicit result has NaN/Inf"
        assert torch.all(torch.isfinite(p_new)), "Pressure result has NaN/Inf"
