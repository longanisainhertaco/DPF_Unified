"""Tests for the implicit MHD module (mlx_implicit_mhd).

All test bodies are stubs — they are marked xfail until Phase S-3 implements
the module.  The stubs document the acceptance criteria and serve as the
test plan for the implementation sprint.

Test structure
--------------
- Config validation: ImplicitMHDConfig construction and guard rails.
- Vacuum identification: identify_vacuum_cells correctness and edge cases.
- Implicit induction step: B-field evolution, conservation, stability.
- Merge: explicit/implicit blending correctness.
- Integration: full operator-split round-trip with the MLX solver.
- Performance: GPU->CPU->GPU copy overhead within acceptable bounds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Mark all tests in this module as expected-fail until implementation.
# Remove the module-level mark and switch individual tests to xpass once
# Phase S-3 is underway.
pytestmark = pytest.mark.xfail(
    reason="Phase S-3: mlx_implicit_mhd not yet implemented",
    strict=False,
)

# Skip entire module if MLX is not available (e.g., CI without Apple Silicon).
mlx = pytest.importorskip("mlx.core", reason="MLX not available")

from dpf.metal.mlx_grid import CylindricalGrid  # noqa: E402
from dpf.metal.mlx_implicit_mhd import (  # noqa: E402
    ImplicitMHDConfig,
    apply_implicit_mhd_split,
    identify_vacuum_cells,
    implicit_induction_step,
    merge_explicit_implicit,
    vacuum_cell_stats,
)
from dpf.metal.mlx_kernels import IBT, IDN, NVAR  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_grid() -> CylindricalGrid:
    """8x16 cylindrical grid for fast unit tests."""
    return CylindricalGrid(nr=8, nz=16, dr=1e-2, dz=2e-3)


@pytest.fixture()
def medium_grid() -> CylindricalGrid:
    """32x64 cylindrical grid for integration tests."""
    return CylindricalGrid(nr=32, nz=64, dr=1e-2, dz=2e-3)


def _uniform_state(nr: int, nz: int, rho: float = 1.0, B_theta: float = 0.1) -> mlx.array:
    """Construct a uniform conserved state for testing.

    Sets rho, B_theta to the given values; all other components to floors/zero.
    """
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    U[IDN] = rho
    gamma = 5.0 / 3.0
    p0 = 1e-4
    U[4] = p0 / (gamma - 1.0) + 0.5 * B_theta**2  # IEN = p/(gamma-1) + ME
    U[IBT] = B_theta
    return mlx.array(U)


def _sheath_state(nr: int, nz: int) -> mlx.array:
    """Construct a state with a physical sheath and a vacuum region.

    Physical cells: r-index 0..nr//2-1, rho=1.0
    Vacuum cells:   r-index nr//2..nr-1, rho=1e-6
    B_theta uniform across both regions (mimics electrode BC injection).
    """
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    rho_phys = 1.0
    rho_vac = 1e-6
    U[IDN, : nr // 2, :] = rho_phys
    U[IDN, nr // 2 :, :] = rho_vac
    U[IBT] = 0.5  # uniform B_theta in HL units
    gamma = 5.0 / 3.0
    p0 = 1e-4
    U[4] = p0 / (gamma - 1.0) + 0.5 * 0.5**2
    return mlx.array(U)


# ---------------------------------------------------------------------------
# ImplicitMHDConfig tests
# ---------------------------------------------------------------------------


class TestImplicitMHDConfig:
    def test_default_disabled(self) -> None:
        """Default config has threshold=0.0 (module disabled).

        Ensures the module does not activate in the existing solver without
        an explicit opt-in.  This is the Phase Q compatibility gate.
        """
        cfg = ImplicitMHDConfig()
        assert cfg.threshold == 0.0

    def test_valid_threshold(self) -> None:
        """Threshold in (0, 1) is accepted without error."""
        cfg = ImplicitMHDConfig(threshold=1e-3)
        assert cfg.threshold == 1e-3

    def test_threshold_zero_is_valid(self) -> None:
        """Threshold=0.0 means disabled; should construct without error."""
        cfg = ImplicitMHDConfig(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_threshold_negative_raises(self) -> None:
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            ImplicitMHDConfig(threshold=-0.1)

    def test_threshold_one_raises(self) -> None:
        """threshold >= 1.0 makes every cell vacuum; should be rejected."""
        with pytest.raises(ValueError, match="threshold"):
            ImplicitMHDConfig(threshold=1.0)

    def test_unsupported_method_raises(self) -> None:
        """Unsupported method strings should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            ImplicitMHDConfig(method="crank_nicolson")

    def test_max_iterations_zero_raises(self) -> None:
        """max_iterations=0 is nonsensical; should raise ValueError."""
        with pytest.raises(ValueError, match="max_iterations"):
            ImplicitMHDConfig(max_iterations=0)

    def test_eta_vacuum_optional(self) -> None:
        """eta_vacuum=None (default) and a positive float are both valid."""
        cfg_none = ImplicitMHDConfig(eta_vacuum=None)
        cfg_val = ImplicitMHDConfig(eta_vacuum=1e-4)
        assert cfg_none.eta_vacuum is None
        assert cfg_val.eta_vacuum == pytest.approx(1e-4)

    def test_sub_cycle_flag(self) -> None:
        """sub_cycle toggles without error."""
        cfg = ImplicitMHDConfig(sub_cycle=True)
        assert cfg.sub_cycle is True


# ---------------------------------------------------------------------------
# identify_vacuum_cells tests
# ---------------------------------------------------------------------------


class TestIdentifyVacuumCells:
    def test_all_physical(self) -> None:
        """Uniform-density state should produce an all-False mask."""
        U = _uniform_state(8, 16, rho=1.0)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        assert mask.shape == (8, 16)
        assert not bool(mlx.any(mask))

    def test_all_vacuum(self) -> None:
        """State with rho << threshold should be all-True mask."""
        U = _uniform_state(8, 16, rho=1e-8)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        assert bool(mlx.all(mask))

    def test_sheath_split(self) -> None:
        """Sheath state: lower half physical, upper half vacuum.

        Checks that the mask boundary aligns with the density step in the
        fixture.
        """
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        mask_np = np.asarray(mask)
        assert not mask_np[: nr // 2, :].any(), "physical half should not be masked"
        assert mask_np[nr // 2 :, :].all(), "vacuum half should be fully masked"

    def test_threshold_zero_returns_no_vacuum(self) -> None:
        """threshold=0.0 disables vacuum identification; mask should be all False."""
        U = _sheath_state(8, 16)
        mask = identify_vacuum_cells(U, rho_threshold=0.0)
        assert not bool(mlx.any(mask))

    def test_output_dtype_bool(self) -> None:
        """Mask dtype should be boolean (not float)."""
        U = _uniform_state(8, 16, rho=1.0)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        assert mask.dtype == mlx.bool_

    def test_shape_matches_spatial(self) -> None:
        """Output shape is (nr, nz), not (NVAR, nr, nz)."""
        U = _uniform_state(12, 24, rho=1.0)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        assert mask.shape == (12, 24)

    def test_single_vacuum_cell(self) -> None:
        """Exactly one vacuum cell in a physical domain."""
        U_np = np.ones((NVAR, 8, 16), dtype=np.float32)
        U_np[IDN] = 1.0
        U_np[IDN, 4, 8] = 1e-10  # single vacuum cell
        U = mlx.array(U_np)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        mask_np = np.asarray(mask)
        assert mask_np.sum() == 1
        assert mask_np[4, 8]


# ---------------------------------------------------------------------------
# implicit_induction_step tests
# ---------------------------------------------------------------------------


class TestImplicitInductionStep:
    def test_no_vacuum_cells_unchanged(self, small_grid: CylindricalGrid) -> None:
        """All-False mask: state should be returned unchanged."""
        nr, nz = 8, 16
        U = _uniform_state(nr, nz)
        mask = mlx.zeros((nr, nz), dtype=mlx.bool_)
        U_out = implicit_induction_step(U, dt=1e-9, eta=1e-5, grid=small_grid, mask=mask)
        assert np.allclose(np.asarray(U_out), np.asarray(U), atol=1e-7)

    def test_b_field_diffuses_in_vacuum(self, small_grid: CylindricalGrid) -> None:
        """B_theta gradient in vacuum cells should diffuse toward uniform.

        Sets up a vacuum slab with a B_theta step and verifies that after
        one implicit step the gradient is reduced (not increased).
        """
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        U_out = implicit_induction_step(U, dt=1e-9, eta=1.0, grid=small_grid, mask=mask)
        U_np = np.asarray(U)
        U_out_np = np.asarray(U_out)
        # Gradient along r in B_theta should decrease
        grad_in = np.abs(np.diff(U_np[IBT, :, 0]))
        grad_out = np.abs(np.diff(U_out_np[IBT, :, 0]))
        assert grad_out.max() <= grad_in.max() + 1e-7

    def test_density_unchanged(self, small_grid: CylindricalGrid) -> None:
        """Implicit step must NOT modify density (rho is not in the induction equation)."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        U_out = implicit_induction_step(U, dt=1e-9, eta=1e-5, grid=small_grid, mask=mask)
        assert np.allclose(
            np.asarray(U_out[IDN]), np.asarray(U[IDN]), atol=1e-8
        ), "density should not change during implicit induction step"

    def test_momentum_unchanged(self, small_grid: CylindricalGrid) -> None:
        """Momentum components are not touched by the induction solver."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        U_out = implicit_induction_step(U, dt=1e-9, eta=1e-5, grid=small_grid, mask=mask)
        for slot in (1, 2, 3):  # IMR, IMZ, IMT
            assert np.allclose(
                np.asarray(U_out[slot]), np.asarray(U[slot]), atol=1e-8
            )

    def test_no_nan_output(self, small_grid: CylindricalGrid) -> None:
        """No NaN or Inf in output for any reasonable input."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        U_out = implicit_induction_step(U, dt=1e-9, eta=1e-5, grid=small_grid, mask=mask)
        U_out_np = np.asarray(U_out)
        assert np.isfinite(U_out_np).all(), "implicit step produced NaN or Inf"

    def test_unconditional_stability_large_dt(self, small_grid: CylindricalGrid) -> None:
        """Implicit solver should remain stable even at 100x the explicit CFL.

        Explicit resistive CFL for these params: dt_exp = dr^2 * mu0 / (2*eta).
        With dr=1e-2, eta=1e-4: dt_exp ~ 6e-7 s.  Test at 100 * dt_exp.
        """
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        dt_large = 6e-5  # 100x explicit CFL
        U_out = implicit_induction_step(
            U, dt=dt_large, eta=1e-4, grid=small_grid, mask=mask
        )
        U_out_np = np.asarray(U_out)
        assert np.isfinite(U_out_np).all(), "large-dt implicit step blew up"

    def test_eta_vacuum_override(self, small_grid: CylindricalGrid) -> None:
        """ImplicitMHDConfig.eta_vacuum overrides global eta in vacuum cells.

        Higher eta_vacuum should produce stronger diffusion of B_theta in
        the vacuum region compared to using the global eta.
        """
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        cfg_low = ImplicitMHDConfig(threshold=1e-3, eta_vacuum=1e-6)
        cfg_high = ImplicitMHDConfig(threshold=1e-3, eta_vacuum=1.0)
        U_low = implicit_induction_step(U, dt=1e-9, eta=1e-5, grid=small_grid, mask=mask, config=cfg_low)
        U_high = implicit_induction_step(U, dt=1e-9, eta=1e-5, grid=small_grid, mask=mask, config=cfg_high)
        diff_low = float(mlx.max(mlx.abs(U_low[IBT] - U[IBT])))
        diff_high = float(mlx.max(mlx.abs(U_high[IBT] - U[IBT])))
        assert diff_high > diff_low, "higher eta_vacuum should cause more B diffusion"

    def test_output_shape(self, small_grid: CylindricalGrid) -> None:
        """Output shape must match input shape (NVAR, nr, nz)."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = mlx.ones((nr, nz), dtype=mlx.bool_)
        U_out = implicit_induction_step(U, dt=1e-9, eta=1e-5, grid=small_grid, mask=mask)
        assert U_out.shape == U.shape


# ---------------------------------------------------------------------------
# merge_explicit_implicit tests
# ---------------------------------------------------------------------------


class TestMergeExplicitImplicit:
    def test_all_physical_selects_explicit(self) -> None:
        """All-False mask: merged result equals U_explicit everywhere."""
        nr, nz = 8, 16
        U_exp = _uniform_state(nr, nz, rho=1.0, B_theta=0.2)
        U_imp = _uniform_state(nr, nz, rho=1.0, B_theta=0.5)
        mask = mlx.zeros((nr, nz), dtype=mlx.bool_)
        U_merged = merge_explicit_implicit(U_exp, U_imp, mask)
        assert np.allclose(np.asarray(U_merged), np.asarray(U_exp), atol=1e-8)

    def test_all_vacuum_selects_implicit(self) -> None:
        """All-True mask: merged result equals U_implicit everywhere."""
        nr, nz = 8, 16
        U_exp = _uniform_state(nr, nz, rho=1.0, B_theta=0.2)
        U_imp = _uniform_state(nr, nz, rho=1.0, B_theta=0.5)
        mask = mlx.ones((nr, nz), dtype=mlx.bool_)
        U_merged = merge_explicit_implicit(U_exp, U_imp, mask)
        assert np.allclose(np.asarray(U_merged), np.asarray(U_imp), atol=1e-8)

    def test_half_half_split(self) -> None:
        """Mask selects explicit in lower half, implicit in upper half.

        Verifies cell-wise selection correctness and no cross-contamination.
        """
        nr, nz = 8, 16
        U_exp = _uniform_state(nr, nz, rho=1.0, B_theta=0.2)
        U_imp = _uniform_state(nr, nz, rho=1.0, B_theta=0.5)
        mask_np = np.zeros((nr, nz), dtype=bool)
        mask_np[nr // 2 :, :] = True
        mask = mlx.array(mask_np)
        U_merged = merge_explicit_implicit(U_exp, U_imp, mask)
        U_merged_np = np.asarray(U_merged)
        exp_np = np.asarray(U_exp)
        imp_np = np.asarray(U_imp)
        assert np.allclose(U_merged_np[:, : nr // 2, :], exp_np[:, : nr // 2, :], atol=1e-8)
        assert np.allclose(U_merged_np[:, nr // 2 :, :], imp_np[:, nr // 2 :, :], atol=1e-8)

    def test_output_shape(self) -> None:
        """Merged output has the same shape as inputs."""
        nr, nz = 8, 16
        U_exp = _uniform_state(nr, nz)
        U_imp = _uniform_state(nr, nz)
        mask = mlx.zeros((nr, nz), dtype=mlx.bool_)
        U_merged = merge_explicit_implicit(U_exp, U_imp, mask)
        assert U_merged.shape == (NVAR, nr, nz)

    def test_no_nan_output(self) -> None:
        """Merge of two finite states should be finite."""
        nr, nz = 8, 16
        U_exp = _uniform_state(nr, nz, rho=1.0)
        U_imp = _uniform_state(nr, nz, rho=1e-6)
        mask_np = np.zeros((nr, nz), dtype=bool)
        mask_np[4:, :] = True
        mask = mlx.array(mask_np)
        U_merged = merge_explicit_implicit(U_exp, U_imp, mask)
        assert np.isfinite(np.asarray(U_merged)).all()


# ---------------------------------------------------------------------------
# apply_implicit_mhd_split integration tests
# ---------------------------------------------------------------------------


class TestApplyImplicitMHDSplit:
    def test_disabled_config_returns_explicit(self, small_grid: CylindricalGrid) -> None:
        """threshold=0.0 fast-path returns U_explicit unchanged."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        cfg = ImplicitMHDConfig(threshold=0.0)
        U_out = apply_implicit_mhd_split(
            U_explicit=U,
            U_pre_explicit=U,
            dt=1e-9,
            eta=1e-5,
            grid=small_grid,
            config=cfg,
        )
        assert np.allclose(np.asarray(U_out), np.asarray(U), atol=1e-8)

    def test_enabled_config_modifies_vacuum(self, small_grid: CylindricalGrid) -> None:
        """With threshold=1e-3 and eta=1.0, vacuum B_theta should change."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        cfg = ImplicitMHDConfig(threshold=1e-3)
        U_out = apply_implicit_mhd_split(
            U_explicit=U,
            U_pre_explicit=U,
            dt=1e-9,
            eta=1.0,
            grid=small_grid,
            config=cfg,
        )
        U_np = np.asarray(U)
        U_out_np = np.asarray(U_out)
        vacuum_slice = slice(nr // 2, None)
        assert not np.allclose(
            U_out_np[IBT, vacuum_slice, :], U_np[IBT, vacuum_slice, :], atol=1e-8
        ), "implicit step should have modified B_theta in vacuum cells"

    def test_physical_cells_unchanged(self, small_grid: CylindricalGrid) -> None:
        """Physical cells must not be altered by the implicit correction."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        cfg = ImplicitMHDConfig(threshold=1e-3)
        U_out = apply_implicit_mhd_split(
            U_explicit=U,
            U_pre_explicit=U,
            dt=1e-9,
            eta=1e-5,
            grid=small_grid,
            config=cfg,
        )
        phys_slice = slice(None, nr // 2)
        assert np.allclose(
            np.asarray(U_out)[:, phys_slice, :],
            np.asarray(U)[:, phys_slice, :],
            atol=1e-8,
        )

    def test_no_nan_full_step(self, small_grid: CylindricalGrid) -> None:
        """Full operator-split step must not produce NaN."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        cfg = ImplicitMHDConfig(threshold=1e-3)
        U_out = apply_implicit_mhd_split(
            U_explicit=U,
            U_pre_explicit=U,
            dt=1e-9,
            eta=1e-5,
            grid=small_grid,
            config=cfg,
        )
        assert np.isfinite(np.asarray(U_out)).all()

    def test_mask_from_pre_explicit_not_post(self, small_grid: CylindricalGrid) -> None:
        """Mask is evaluated on U_pre_explicit, not U_explicit.

        Construct a case where the explicit step partially replenishes a
        vacuum cell.  The implicit step should still treat that cell as vacuum
        because it was vacuum at the start of the step.
        """
        nr, nz = 8, 16
        U_pre = _sheath_state(nr, nz)
        # Simulate that the explicit step added mass to the first vacuum cell
        U_post_np = np.asarray(U_pre).copy()
        U_post_np[IDN, nr // 2, :] = 0.5  # now dense in post, but was vacuum
        U_post = mlx.array(U_post_np)
        cfg = ImplicitMHDConfig(threshold=1e-3)
        # Should not raise; the implicit step sees the cell as vacuum
        # (from U_pre), applies implicit B diffusion, then merges.
        U_out = apply_implicit_mhd_split(
            U_explicit=U_post,
            U_pre_explicit=U_pre,
            dt=1e-9,
            eta=1e-5,
            grid=small_grid,
            config=cfg,
        )
        assert np.isfinite(np.asarray(U_out)).all()


# ---------------------------------------------------------------------------
# vacuum_cell_stats tests
# ---------------------------------------------------------------------------


class TestVacuumCellStats:
    def test_keys_present(self) -> None:
        """Stats dict must contain all required keys."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        stats = vacuum_cell_stats(U, mask)
        for key in ("n_vacuum", "frac_vacuum", "rho_min_vacuum", "B_max_vacuum", "va_max_vacuum"):
            assert key in stats, f"missing key '{key}' in stats"

    def test_n_vacuum_correct(self) -> None:
        """n_vacuum should equal the number of True cells in the mask."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        stats = vacuum_cell_stats(U, mask)
        expected = int(np.asarray(mask).sum())
        assert stats["n_vacuum"] == expected

    def test_frac_vacuum_bounds(self) -> None:
        """Vacuum fraction must be in [0, 1]."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        stats = vacuum_cell_stats(U, mask)
        assert 0.0 <= stats["frac_vacuum"] <= 1.0

    def test_no_vacuum_cells_returns_zeros(self) -> None:
        """All-False mask should yield n_vacuum=0 and frac_vacuum=0."""
        nr, nz = 8, 16
        U = _uniform_state(nr, nz, rho=1.0)
        mask = mlx.zeros((nr, nz), dtype=mlx.bool_)
        stats = vacuum_cell_stats(U, mask)
        assert stats["n_vacuum"] == 0
        assert stats["frac_vacuum"] == pytest.approx(0.0)

    def test_va_max_vacuum_positive(self) -> None:
        """Alfven speed in vacuum should be finite and positive for nonzero B."""
        nr, nz = 8, 16
        U = _sheath_state(nr, nz)
        mask = identify_vacuum_cells(U, rho_threshold=1e-3)
        stats = vacuum_cell_stats(U, mask)
        assert stats["va_max_vacuum"] > 0.0
        assert math.isfinite(stats["va_max_vacuum"])
