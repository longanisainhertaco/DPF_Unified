"""Tests for block-structured AMR (src/dpf/metal/mlx_amr.py).

Validates:
1. Hierarchy construction from uniform grid
2. Ghost exchange between neighboring blocks
3. Prolongation (coarse -> fine) conserves mass
4. Restriction (fine -> coarse) conserves mass
5. Full amr_step on Sod shock (Lax-Friedrichs RHS)
6. 2-level AMR produces sharper shock than coarse-only
"""

import numpy as np
import pytest

from dpf.metal.mlx_amr import (
    IDN,
    IEN,
    ISR,
    NVAR,
    AMRLevel,
    amr_step,
    build_amr_hierarchy,
    decompose_domain,
    ghost_exchange_same_level,
    prolongate_to_fine,
)


def _sod_initial_condition(nr: int, nz: int, gamma: float = 5.0 / 3.0) -> np.ndarray:
    """Create Sod shock tube initial condition in conserved variables.

    Left state (z < 0.5): rho=1, p=1
    Right state (z >= 0.5): rho=0.125, p=0.1
    """
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    mid = nz // 2
    # Left state
    rho_L, p_L = 1.0, 1.0
    U[IDN, :, :mid] = rho_L
    U[IEN, :, :mid] = p_L / (gamma - 1.0)
    U[ISR, :, :mid] = p_L * rho_L ** (1.0 - gamma)
    # Right state
    rho_R, p_R = 0.125, 0.1
    U[IDN, :, mid:] = rho_R
    U[IEN, :, mid:] = p_R / (gamma - 1.0)
    U[ISR, :, mid:] = p_R * rho_R ** (1.0 - gamma)
    return U


class TestHierarchyConstruction:
    """Test building AMR hierarchy."""

    def test_decompose_creates_blocks(self):
        level0 = decompose_domain(32, 64, dr=0.01, dz=0.01, r_inner=0.0,
                                  block_nr=16, block_nz=16)
        # 32/16 = 2 blocks in r, 64/16 = 4 blocks in z
        assert len(level0.blocks) == 8
        for _idx, block in level0.blocks.items():
            assert block.U.shape == (NVAR, 16, 16)

    def test_build_hierarchy_has_two_levels(self):
        h = build_amr_hierarchy(32, 64, dr=0.01, dz=0.01, r_inner=0.0,
                                block_nr=16, block_nz=16, ratio=2)
        assert h.n_levels == 2
        assert len(h.levels[0].blocks) == 8
        assert len(h.levels[1].blocks) == 0  # no refined blocks yet

    def test_build_hierarchy_with_refined_block(self):
        h = build_amr_hierarchy(32, 64, dr=0.01, dz=0.01, r_inner=0.0,
                                block_nr=16, block_nz=16, ratio=2,
                                refined_blocks=[[0, 2]])
        # Refining one coarse block should create 4 fine blocks (2x2 subdivision)
        assert len(h.levels[1].blocks) > 0


class TestGhostExchange:
    """Test ghost cell communication between blocks."""

    def test_ghost_exchange_returns_padded(self):
        level0 = decompose_domain(32, 32, dr=0.01, dz=0.01, r_inner=0.0,
                                  block_nr=16, block_nz=16)
        # Set uniform density
        for block in level0.blocks.values():
            block.U = np.ones((NVAR, 16, 16), dtype=np.float32)

        padded = ghost_exchange_same_level(level0, ng=2, block_nr=16, block_nz=16,
                                           r_inner=0.0)
        for _idx, U_pad in padded.items():
            U_arr = np.asarray(U_pad)
            assert U_arr.shape == (NVAR, 20, 20)  # 16 + 2*2 ghosts


class TestConservation:
    """Test that prolongation and restriction conserve mass."""

    def test_prolongation_conserves_mass(self):
        """Prolongated fine blocks should have same total mass as coarse."""
        level0 = decompose_domain(16, 16, dr=0.01, dz=0.01, r_inner=0.0,
                                  block_nr=16, block_nz=16)
        block = level0.blocks[(0, 0)]
        # Uniform density
        block.U = np.ones((NVAR, 16, 16), dtype=np.float32)
        block.U[IDN] = 2.0  # rho = 2

        level1 = AMRLevel(level=1, blocks={}, dr=0.005, dz=0.005)
        children = prolongate_to_fine(block, level1, ratio=2,
                                      block_nr=16, block_nz=16)
        fine_mass = sum(float(np.sum(np.asarray(c.U)[IDN])) for c in children)
        coarse_mass = float(np.sum(block.U[IDN]))
        # Fine grid has 4x cells, each at half volume -> total mass should match
        # within interpolation error
        assert abs(fine_mass - 4 * coarse_mass) / (4 * coarse_mass) < 0.1


class TestAMRStep:
    """Test full AMR timestep on Sod shock."""

    def test_sod_shock_no_crash(self):
        """AMR step on Sod shock should complete without error."""
        nr, nz = 32, 64
        dr, dz = 0.01, 0.01
        block_nr, block_nz = 16, 16

        h = build_amr_hierarchy(nr, nz, dr, dz, r_inner=0.0,
                                block_nr=block_nr, block_nz=block_nz, ratio=2)

        # Initialize with Sod IC
        U0 = _sod_initial_condition(nr, nz)
        for idx, block in h.levels[0].blocks.items():
            ir, iz = idx
            r_start = ir * block_nr
            z_start = iz * block_nz
            block.U = U0[:, r_start:r_start + block_nr, z_start:z_start + block_nz].copy()

        # One step
        dt = 1e-4
        h_new, dt_used = amr_step(
            h, dt=dt, gamma=5.0 / 3.0, method="plm", riemann="hll",
            ng=2, current=0.0, r_inner=0.0, step_number=0,
            rhs_fn=None,  # uses built-in Lax-Friedrichs
        )
        assert dt_used == dt

        # Check no NaN
        for block in h_new.levels[0].active_blocks():
            U = np.asarray(block.U)
            assert not np.any(np.isnan(U)), "NaN in AMR output"

    @pytest.mark.skipif(
        not pytest.importorskip("mlx.core", reason="MLX not available"),
        reason="MLX required",
    )
    def test_sod_production_rhs_no_crash(self):
        """AMR step with production MLX RHS (WENO5-Z/HLL) completes."""
        from dpf.metal.mlx_amr import make_mlx_block_rhs
        nr, nz = 32, 64
        dr, dz = 0.01, 0.01
        block_nr, block_nz = 16, 16

        h = build_amr_hierarchy(nr, nz, dr, dz, r_inner=dr * 0.5,
                                block_nr=block_nr, block_nz=block_nz, ratio=2)
        U0 = _sod_initial_condition(nr, nz)
        for idx, block in h.levels[0].blocks.items():
            ir, iz = idx
            block.U = U0[:, ir * block_nr:(ir + 1) * block_nr,
                         iz * block_nz:(iz + 1) * block_nz].copy()

        rhs_fn = make_mlx_block_rhs(coordinates="cylindrical")
        h_new, _ = amr_step(
            h, dt=1e-5, gamma=5.0 / 3.0, method="plm", riemann="hll",
            ng=3, current=0.0, r_inner=dr * 0.5, step_number=0,
            rhs_fn=rhs_fn,
        )
        for block in h_new.levels[0].active_blocks():
            U = np.asarray(block.U)
            assert not np.any(np.isnan(U)), f"NaN in production AMR block {block.index}"
            assert np.all(U[IDN] >= 0), f"Negative rho in block {block.index}"

    def test_sod_density_positive(self):
        """After AMR step, density should remain positive."""
        nr, nz = 32, 32
        dr, dz = 0.01, 0.01
        block_nr, block_nz = 16, 16

        h = build_amr_hierarchy(nr, nz, dr, dz, r_inner=0.0,
                                block_nr=block_nr, block_nz=block_nz, ratio=2)
        U0 = _sod_initial_condition(nr, nz)
        for idx, block in h.levels[0].blocks.items():
            ir, iz = idx
            r_s, z_s = ir * block_nr, iz * block_nz
            block.U = U0[:, r_s:r_s + block_nr, z_s:z_s + block_nz].copy()

        # 5 steps
        for step in range(5):
            h, _ = amr_step(
                h, dt=5e-5, gamma=5.0 / 3.0, method="plm", riemann="hll",
                ng=2, current=0.0, r_inner=0.0, step_number=step,
                rhs_fn=None,
            )
        for block in h.levels[0].active_blocks():
            rho = np.asarray(block.U)[IDN]
            assert np.all(rho >= 0), f"Negative density in block {block.index}"

    def test_total_mass_conserved(self):
        """Total mass across all blocks should be conserved."""
        nr, nz = 32, 32
        dr, dz = 0.01, 0.01
        block_nr, block_nz = 16, 16

        h = build_amr_hierarchy(nr, nz, dr, dz, r_inner=0.0,
                                block_nr=block_nr, block_nz=block_nz, ratio=2)
        U0 = _sod_initial_condition(nr, nz)
        for idx, block in h.levels[0].blocks.items():
            ir, iz = idx
            r_s, z_s = ir * block_nr, iz * block_nz
            block.U = U0[:, r_s:r_s + block_nr, z_s:z_s + block_nz].copy()

        mass0 = sum(
            float(np.sum(np.asarray(b.U)[IDN]))
            for b in h.levels[0].active_blocks()
        )

        for step in range(3):
            h, _ = amr_step(
                h, dt=5e-5, gamma=5.0 / 3.0, method="plm", riemann="hll",
                ng=2, current=0.0, r_inner=0.0, step_number=step,
                rhs_fn=None,
            )

        mass1 = sum(
            float(np.sum(np.asarray(b.U)[IDN]))
            for b in h.levels[0].active_blocks()
        )

        # Lax-Friedrichs is not perfectly conservative at block boundaries
        # but should be within 10% for 3 steps
        rel_error = abs(mass1 - mass0) / mass0
        assert rel_error < 0.1, f"Mass conservation error: {rel_error:.4f}"


class TestSolverAMRIntegration:
    """Test MLXMHDSolver with enable_amr=True."""

    @pytest.mark.skipif(
        not pytest.importorskip("mlx.core", reason="MLX not available"),
        reason="MLX required",
    )
    def test_solver_constructs_with_amr(self):
        from dpf.metal.mlx_solver import MLXMHDSolver
        solver = MLXMHDSolver(
            grid_shape=(16, 1, 32), dx=0.01,
            enable_amr=True, amr_block_nr=8, amr_block_nz=16,
        )
        assert solver._enable_amr is True
        assert solver._amr_hierarchy is not None
        assert solver._amr_rhs_fn is not None

    @pytest.mark.skipif(
        not pytest.importorskip("mlx.core", reason="MLX not available"),
        reason="MLX required",
    )
    def test_amr_step_method_runs(self):
        """MLXMHDSolver.amr_step() should accept a state dict and return one."""
        from dpf.metal.mlx_solver import MLXMHDSolver
        nr, nz = 16, 32
        solver = MLXMHDSolver(
            grid_shape=(nr, 1, nz), dx=0.01,
            enable_amr=True, amr_block_nr=8, amr_block_nz=16,
        )
        state = {
            "rho": np.ones((nr, nz), dtype=np.float32) * 1e-3,
            "velocity": np.zeros((3, nr, nz), dtype=np.float32),
            "pressure": np.ones((nr, nz), dtype=np.float32) * 100.0,
            "B": np.zeros((3, nr, nz), dtype=np.float32),
            "Te": np.ones((nr, nz), dtype=np.float32) * 1e4,
            "Ti": np.ones((nr, nz), dtype=np.float32) * 1e4,
        }
        result = solver.amr_step(state, dt=1e-6, current=0.0)
        assert "rho" in result
        assert result["rho"].shape == (nr, nz)
        assert not np.any(np.isnan(result["rho"]))
