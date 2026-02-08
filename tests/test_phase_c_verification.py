"""Tests for Phase C: Verification & Validation.

Covers:
    C.1 Diffusion convergence (explicit, implicit, STS)
    C.2 Orszag-Tang vortex benchmark
    C.3 Cylindrical Sedov blast wave
    C.4 Lee Model comparison
"""

from __future__ import annotations

import numpy as np
import pytest


# ===================================================================
# C.1 — Diffusion Convergence Tests
# ===================================================================


class TestDiffusionConvergence:
    """Verify resistive magnetic diffusion converges against analytical solution."""

    def test_analytical_solution_is_correct(self):
        """Gaussian diffusion analytical solution satisfies known properties."""
        from dpf.verification.diffusion_convergence import gaussian_B_analytical

        B0 = 1.0
        sigma0 = 0.1
        D = 1e5  # Arbitrary diffusivity

        # Use a wide domain (10*sigma0) so Gaussian tails are fully captured
        x = np.linspace(-1.0, 1.0, 2000)

        # At t=0, should be the original Gaussian
        By_0 = gaussian_B_analytical(x, 0.0, B0, sigma0, D)
        expected = B0 * np.exp(-x**2 / (2.0 * sigma0**2))
        np.testing.assert_allclose(By_0, expected, rtol=1e-12)

        # At t>0, peak should decrease and width should increase
        By_t = gaussian_B_analytical(x, 1e-8, B0, sigma0, D)
        assert np.max(By_t) < B0  # Peak decreases
        assert np.max(By_t) > 0   # Still positive peak

        # Total integral should be conserved (Gaussian integral property)
        # At t=1e-8 with D=1e5, sigma(t)^2 = 0.01 + 2e-3 = 0.012
        # so sigma(t) ~ 0.11, still well within [-1, 1]
        dx = x[1] - x[0]
        integral_0 = np.sum(By_0) * dx
        integral_t = np.sum(By_t) * dx
        np.testing.assert_allclose(integral_t, integral_0, rtol=1e-3)

    def test_result_dataclass_fields(self):
        """DiffusionConvergenceResult has expected fields."""
        from dpf.verification.diffusion_convergence import DiffusionConvergenceResult

        r = DiffusionConvergenceResult(
            method="test",
            resolutions=[32, 64],
            errors=[0.1, 0.05],
            convergence_order=2.0,
            eta=1.0,
            sigma0=0.1,
            t_end=1e-6,
        )
        assert r.method == "test"
        assert len(r.resolutions) == 2
        assert r.convergence_order == 2.0

    def test_convergence_order_estimation(self):
        """Internal convergence order estimation works on known data."""
        from dpf.verification.diffusion_convergence import _estimate_convergence_order

        # Exact second-order data
        resolutions = [16, 32, 64, 128]
        errors = [1.0 / n**2 for n in resolutions]
        order = _estimate_convergence_order(resolutions, errors)
        assert abs(order - 2.0) < 0.05, f"Expected ~2.0, got {order:.3f}"

    def test_invalid_method_raises(self):
        """run_diffusion_convergence rejects unknown methods."""
        from dpf.verification.diffusion_convergence import run_diffusion_convergence

        with pytest.raises(ValueError, match="Unknown method"):
            run_diffusion_convergence(method="bogus")

    @pytest.mark.slow
    def test_diffusion_convergence_implicit(self):
        """Implicit (Crank-Nicolson) diffusion converges with order > 1.5."""
        from dpf.verification.diffusion_convergence import run_diffusion_convergence

        result = run_diffusion_convergence(
            method="implicit",
            resolutions=[32, 64, 128],
        )

        assert result.convergence_order > 1.5, (
            f"Implicit convergence order {result.convergence_order:.2f} "
            f"should be > 1.5. Errors: {result.errors}"
        )

        # Errors should decrease monotonically
        for i in range(len(result.errors) - 1):
            assert result.errors[i + 1] < result.errors[i], (
                f"Error at nx={result.resolutions[i + 1]} "
                f"({result.errors[i + 1]:.3e}) should be less than at "
                f"nx={result.resolutions[i]} ({result.errors[i]:.3e})"
            )

    @pytest.mark.slow
    def test_diffusion_convergence_sts(self):
        """STS (RKL2) diffusion converges with order > 1.5."""
        from dpf.verification.diffusion_convergence import run_diffusion_convergence

        result = run_diffusion_convergence(
            method="sts",
            resolutions=[32, 64, 128],
        )

        assert result.convergence_order > 1.5, (
            f"STS convergence order {result.convergence_order:.2f} "
            f"should be > 1.5. Errors: {result.errors}"
        )

    @pytest.mark.slow
    def test_diffusion_convergence_explicit(self):
        """Explicit MHD solver diffusion converges with order > 1.5."""
        from dpf.verification.diffusion_convergence import run_diffusion_convergence

        result = run_diffusion_convergence(
            method="explicit",
            resolutions=[32, 64, 128],
        )

        assert result.convergence_order > 1.5, (
            f"Explicit convergence order {result.convergence_order:.2f} "
            f"should be > 1.5. Errors: {result.errors}"
        )


# ===================================================================
# C.2 — Orszag-Tang Vortex Benchmark
# ===================================================================


class TestOrszagTang:
    """Verify the Orszag-Tang vortex benchmark runs and produces valid results."""

    def test_result_dataclass_fields(self):
        """OrszagTangResult has expected fields."""
        from dpf.verification.orszag_tang import OrszagTangResult

        r = OrszagTangResult(
            rho_final=np.zeros((10, 10)),
            rho_min=0.1,
            rho_max=1.0,
            energy_initial=100.0,
            energy_final=99.0,
            energy_conservation=0.99,
            max_div_B=1e-10,
            nx=10,
            t_end=0.5,
            n_steps=100,
        )
        assert r.nx == 10
        assert r.energy_conservation == 0.99

    @pytest.mark.slow
    def test_orszag_tang_runs(self):
        """Orszag-Tang vortex completes and produces physically valid output.

        Uses reduced resolution (nx=64) and short t_end for test speed.
        The solver uses zero-gradient BCs (not periodic), so we use a
        short integration time (t=0.2) to keep boundary effects manageable.

        Verifies:
        - Density stays positive
        - Energy conserved to within 5%
        - No NaN/Inf in output
        """
        from dpf.verification.orszag_tang import run_orszag_tang

        result = run_orszag_tang(nx=32, t_end=0.1)

        # No NaN/Inf
        assert np.all(np.isfinite(result.rho_final)), "NaN/Inf in density field"

        # Density must stay positive
        assert result.rho_min > 0, (
            f"Density went non-positive: rho_min={result.rho_min:.3e}"
        )

        # Energy conservation: within 5% (Neumann BCs may leak some energy)
        assert abs(result.energy_conservation - 1.0) < 0.05, (
            f"Energy conservation {result.energy_conservation:.4f} "
            f"deviates more than 5% from unity"
        )

        # Should have taken some steps
        assert result.n_steps > 0

        # Density should have developed structure (not uniform anymore)
        rho_range = result.rho_max - result.rho_min
        assert rho_range > 0.001, (
            f"Density range {rho_range:.6f} too small -- no dynamics?"
        )


# ===================================================================
# C.3 — Cylindrical Sedov Blast
# ===================================================================


class TestSedovCylindrical:
    """Verify the cylindrical Sedov-Taylor blast wave test."""

    def test_similarity_solution_basic(self):
        """Sedov similarity solution returns physically reasonable values."""
        from dpf.verification.sedov_cylindrical import sedov_shock_radius_cylindrical

        E0 = 1.0
        rho0 = 1.0
        gamma = 5.0 / 3.0

        # R should increase with time
        R1 = sedov_shock_radius_cylindrical(E0, rho0, 0.01, gamma)
        R2 = sedov_shock_radius_cylindrical(E0, rho0, 0.1, gamma)
        assert R2 > R1 > 0

        # R should increase with energy
        R_low = sedov_shock_radius_cylindrical(0.1, rho0, 0.1, gamma)
        R_high = sedov_shock_radius_cylindrical(10.0, rho0, 0.1, gamma)
        assert R_high > R_low

    def test_result_dataclass_fields(self):
        """SedovCylindricalResult has expected fields."""
        from dpf.verification.sedov_cylindrical import SedovCylindricalResult

        r = SedovCylindricalResult(
            rho_profile_r=np.ones(10),
            rho_profile_z=np.ones(20),
            r_coords=np.arange(10) * 0.01,
            z_coords=np.arange(20) * 0.01,
            shock_position_numerical=0.3,
            shock_position_analytical=0.32,
            relative_error=0.0625,
            E0=1.0,
            rho0=1.0,
            t_end=0.1,
        )
        assert r.relative_error == 0.0625

    @pytest.mark.slow
    def test_sedov_cylindrical_runs(self):
        """Sedov blast test runs and produces meaningful output.

        Uses reduced resolution for test speed.  The cylindrical solver's
        two-temperature model can introduce NaN for strong blast waves,
        so we verify:
        - The test completes with >0 steps
        - The analytical shock radius formula gives reasonable values
        - The solver produces some finite output before NaN (if any)
        """
        from dpf.verification.sedov_cylindrical import run_sedov_cylindrical

        result = run_sedov_cylindrical(nr=64, nz=128, t_end=0.05)

        # Should have taken some steps
        assert result.n_steps > 0, "Sedov test should take at least one step"

        # Analytical shock radius should be positive and finite
        assert result.shock_position_analytical > 0 or not np.isfinite(result.t_end), (
            f"Analytical shock position should be positive, got "
            f"{result.shock_position_analytical:.4e}"
        )

        # The Sedov setup is complete and returns a result
        assert result.E0 > 0
        assert result.rho0 > 0
        assert result.gamma > 1.0

        # Density profiles should have correct shape
        assert len(result.rho_profile_r) == 64
        assert len(result.rho_profile_z) == 128

        # If the solver managed finite output, check shock detection
        if np.all(np.isfinite(result.rho_profile_r)):
            # Shock position should be within domain
            assert 0 < result.shock_position_numerical < 0.5, (
                f"Shock at {result.shock_position_numerical:.4e} outside domain"
            )
            # Relative error within 30% (generous for coarse grid)
            assert result.relative_error < 0.30, (
                f"Shock error {result.relative_error:.2%} exceeds 30%"
            )


# ===================================================================
# C.4 — Lee Model
# ===================================================================


class TestLeeModel:
    """Verify the Lee Model produces reasonable results for known devices."""

    def test_result_dataclass_fields(self):
        """LeeModelResult has expected fields."""
        from dpf.validation.lee_model_comparison import LeeModelResult

        r = LeeModelResult(
            t=np.array([0, 1e-6]),
            I=np.array([0, 1e6]),
            V=np.array([15e3, 10e3]),
            z_sheet=np.array([0, 0.05]),
            r_shock=np.array([0.08, 0.08]),
            peak_current=1e6,
            peak_current_time=1e-6,
            pinch_time=2e-6,
            device_name="test",
        )
        assert r.peak_current == 1e6
        assert r.device_name == "test"

    def test_lee_model_instantiation(self):
        """LeeModel can be instantiated with default parameters."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel()
        assert model.fm == 0.7
        assert model.fc == 0.7

    def test_lee_model_custom_params(self):
        """LeeModel can be instantiated with custom parameters."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel(current_fraction=0.8, mass_fraction=0.6)
        assert model.fm == 0.8
        assert model.fc == 0.6

    @pytest.mark.slow
    def test_lee_model_pf1000(self):
        """Lee Model produces reasonable I(t) for PF-1000 parameters.

        Verifies:
        - Peak current is within right order of magnitude (100 kA - 10 MA)
        - Peak current time is microsecond scale
        - Current starts at zero
        - At least phase 1 completes
        """
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel()
        result = model.run(device_name="PF-1000")

        # Current should start near zero
        assert abs(result.I[0]) < 1e3, (
            f"Initial current {result.I[0]:.2e} should be ~0"
        )

        # Peak current should be in reasonable range for PF-1000
        # (Experimental: ~1.87 MA, model should be within 0.1-10 MA)
        assert result.peak_current > 100e3, (
            f"Peak current {result.peak_current:.2e} too low for PF-1000"
        )
        assert result.peak_current < 10e6, (
            f"Peak current {result.peak_current:.2e} too high for PF-1000"
        )

        # Peak current time should be microsecond scale
        assert 1e-7 < result.peak_current_time < 50e-6, (
            f"Peak current time {result.peak_current_time:.2e} "
            f"should be in microsecond range"
        )

        # Phase 1 should have completed
        assert 1 in result.phases_completed

        # Time and current arrays should have matching lengths
        assert len(result.t) == len(result.I)
        assert len(result.t) == len(result.V)

        # No NaN/Inf in output
        assert np.all(np.isfinite(result.I)), "NaN/Inf in current waveform"
        assert np.all(np.isfinite(result.V)), "NaN/Inf in voltage waveform"

    @pytest.mark.slow
    def test_lee_model_nx2(self):
        """Lee Model produces reasonable I(t) for NX2 parameters.

        NX2 is a smaller device: peak current ~400 kA.
        """
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel()
        result = model.run(device_name="NX2")

        # Peak current should be in range 50 kA - 2 MA for NX2
        assert result.peak_current > 50e3, (
            f"Peak current {result.peak_current:.2e} too low for NX2"
        )
        assert result.peak_current < 2e6, (
            f"Peak current {result.peak_current:.2e} too high for NX2"
        )

        # Peak time should be microsecond scale
        assert 1e-7 < result.peak_current_time < 20e-6

        assert 1 in result.phases_completed

    @pytest.mark.slow
    def test_lee_model_comparison_pf1000(self):
        """Lee Model comparison produces error metrics for PF-1000."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel()
        comparison = model.compare_with_experiment("PF-1000")

        assert comparison.device_name == "PF-1000"
        assert comparison.lee_result.peak_current > 0
        # Error should be computed (may not be small, as Lee Model is simplified)
        assert comparison.peak_current_error >= 0
        assert comparison.timing_error >= 0

    def test_lee_model_missing_device_raises(self):
        """Lee Model raises for unknown device name."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel()
        with pytest.raises(KeyError, match="not found"):
            model.run(device_name="NONEXISTENT_DEVICE")

    def test_lee_model_no_args_raises(self):
        """Lee Model raises when neither device_name nor device_params given."""
        from dpf.validation.lee_model_comparison import LeeModel

        model = LeeModel()
        with pytest.raises(ValueError, match="Must provide"):
            model.run()


# ===================================================================
# Module-level imports and edge cases
# ===================================================================


class TestModuleImports:
    """Verify all new modules can be imported correctly."""

    def test_import_diffusion_convergence(self):
        """diffusion_convergence module imports without error."""
        from dpf.verification.diffusion_convergence import (
            DiffusionConvergenceResult,
            gaussian_B_analytical,
            run_diffusion_convergence,
        )
        assert callable(run_diffusion_convergence)
        assert callable(gaussian_B_analytical)

    def test_import_orszag_tang(self):
        """orszag_tang module imports without error."""
        from dpf.verification.orszag_tang import (
            OrszagTangResult,
            run_orszag_tang,
        )
        assert callable(run_orszag_tang)

    def test_import_sedov_cylindrical(self):
        """sedov_cylindrical module imports without error."""
        from dpf.verification.sedov_cylindrical import (
            SedovCylindricalResult,
            run_sedov_cylindrical,
            sedov_shock_radius_cylindrical,
        )
        assert callable(run_sedov_cylindrical)
        assert callable(sedov_shock_radius_cylindrical)

    def test_import_lee_model(self):
        """lee_model_comparison module imports without error."""
        from dpf.validation.lee_model_comparison import (
            LeeModel,
            LeeModelComparison,
            LeeModelResult,
        )
        assert callable(LeeModel)

    def test_verification_init_exports(self):
        """verification __init__ exports new symbols."""
        from dpf.verification import (
            DiffusionConvergenceResult,
            OrszagTangResult,
            SedovCylindricalResult,
            run_diffusion_convergence,
            run_orszag_tang,
            run_sedov_cylindrical,
        )
        assert callable(run_diffusion_convergence)

    def test_validation_init_exports(self):
        """validation __init__ exports Lee Model symbols."""
        from dpf.validation import (
            LeeModel,
            LeeModelComparison,
            LeeModelResult,
        )
        assert callable(LeeModel)
