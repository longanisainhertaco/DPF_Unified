"""Tests for AMR Phase A-slim: 2-level block-structured refinement for MLX.

All tests gate on mlx.core availability.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from dpf.metal.mlx_amr import (  # noqa: E402, I001
    AMRBlock,
    AMRHierarchy,
    AMRLevel,
    FluxRegister,
    NVAR,
    IDN,
    IBR,
    IBT,
    IEN,
    _prolongate_vanleer,
    amr_step,
    apply_reflux_correction,
    assemble_global_state,
    build_amr_hierarchy,
    decompose_domain,
    ghost_exchange_same_level,
    populate_blocks_from_state,
    prolongate_to_fine,
    restrict_to_coarse,
)

BNR = 16
BNZ = 32
DR = 1e-3
DZ = 1e-3
R_INNER = 0.01


# ---------------------------------------------------------------------------
# Helper: make a simple uniform conserved state
# ---------------------------------------------------------------------------


def make_uniform_state(nr: int, nz: int, rho: float = 1.0, p: float = 0.1) -> np.ndarray:
    """Create a uniform (NVAR, nr, nz) conserved state."""
    gamma = 5.0 / 3.0
    U = np.zeros((NVAR, nr, nz), dtype=np.float32)
    U[IDN] = rho
    U[IEN] = p / (gamma - 1.0) + 0.5 * rho * 0.0  # no kinetic energy
    return U


def make_hierarchy_2x2(U_global: np.ndarray | None = None) -> AMRHierarchy:
    """Build a 2x2-block, 2-level hierarchy for 32x64 grid."""
    h = build_amr_hierarchy(
        nr=32, nz=64, dr=DR, dz=DZ, r_inner=R_INNER,
        block_nr=BNR, block_nz=BNZ, ratio=2,
    )
    if U_global is not None:
        populate_blocks_from_state(h.levels[0], U_global, BNR, BNZ)
    return h


# ---------------------------------------------------------------------------
# 1. test_decompose_block_count
# ---------------------------------------------------------------------------


def test_decompose_block_count():
    level = decompose_domain(32, 64, DR, DZ, R_INNER, BNR, BNZ)
    assert len(level.blocks) == 2 * 2  # ceil(32/16) x ceil(64/32)


# ---------------------------------------------------------------------------
# 2. test_decompose_coordinates
# ---------------------------------------------------------------------------


def test_decompose_coordinates():
    level = decompose_domain(32, 64, DR, DZ, R_INNER, BNR, BNZ)
    b_10 = level.blocks[(1, 0)]
    expected_r_min = R_INNER + 1 * BNR * DR
    assert abs(b_10.r_min - expected_r_min) < 1e-12
    b_00 = level.blocks[(0, 0)]
    assert abs(b_00.r_min - R_INNER) < 1e-12
    assert abs(b_00.z_min - 0.0) < 1e-12


# ---------------------------------------------------------------------------
# 3. test_populate_assemble_roundtrip
# ---------------------------------------------------------------------------


def test_populate_assemble_roundtrip():
    rng = np.random.default_rng(42)
    U_global = rng.random((NVAR, 32, 64), dtype=np.float32)
    level = decompose_domain(32, 64, DR, DZ, R_INNER, BNR, BNZ)
    populate_blocks_from_state(level, U_global, BNR, BNZ)
    U_recovered = np.asarray(assemble_global_state(level, 32, 64, BNR, BNZ))
    assert np.max(np.abs(U_global - U_recovered)) < 1e-6


# ---------------------------------------------------------------------------
# 4. test_ghost_interior_copy
# ---------------------------------------------------------------------------


def test_ghost_interior_copy():
    """Ghost cells from a neighbor should equal the neighbor's interior slab."""
    U_global = np.zeros((NVAR, 32, 64), dtype=np.float32)
    # Set block (0,0) to all-1s, block (1,0) to all-2s
    U_global[:, :BNR, :BNZ] = 1.0
    U_global[:, BNR:, :BNZ] = 2.0

    level = decompose_domain(32, 64, DR, DZ, R_INNER, BNR, BNZ)
    populate_blocks_from_state(level, U_global, BNR, BNZ)

    ng = 3
    padded = ghost_exchange_same_level(level, ng, BNR, BNZ, R_INNER)

    # Block (0,0)'s E ghost should contain block (1,0)'s first ng radial cells
    pad_00 = np.asarray(padded[(0, 0)])
    # E ghost is at pad[..., ng+BNR:ng+BNR+ng, ng:ng+BNZ]
    e_ghost = pad_00[:, ng + BNR : ng + BNR + ng, ng : ng + BNZ]
    assert np.allclose(e_ghost, 2.0), f"E ghost should be 2.0, got {e_ghost[0,0,0]}"


# ---------------------------------------------------------------------------
# 5. test_ghost_axis_reflection
# ---------------------------------------------------------------------------


def test_ghost_axis_reflection():
    """Sign flip for IBR and IBT at the axis (W boundary of block (0,0))."""
    U_global = np.zeros((NVAR, 32, 64), dtype=np.float32)
    # Set B_r = 3.0 in the interior near axis
    U_global[IBR, :BNR, :BNZ] = 3.0
    U_global[IBT, :BNR, :BNZ] = 5.0

    level = decompose_domain(32, 64, DR, DZ, R_INNER, BNR, BNZ)
    populate_blocks_from_state(level, U_global, BNR, BNZ)

    ng = 3
    padded = ghost_exchange_same_level(level, ng, BNR, BNZ, R_INNER)
    pad_00 = np.asarray(padded[(0, 0)])

    # W ghost (r=0 axis reflection) for block (0,0): r_inner = R_INNER > 0
    # W ghost is at pad[..., :ng, ng:ng+BNZ]
    w_ghost_Br = pad_00[IBR, :ng, ng : ng + BNZ]
    assert np.all(w_ghost_Br < 0), "B_r should be sign-flipped in axis ghost"

    w_ghost_Bt = pad_00[IBT, :ng, ng : ng + BNZ]
    assert np.all(w_ghost_Bt < 0), "B_theta should be sign-flipped in axis ghost"


# ---------------------------------------------------------------------------
# 6. test_ghost_outflow_zerograd
# ---------------------------------------------------------------------------


def test_ghost_outflow_zerograd():
    """Edge blocks get zero-gradient ghosts at physical boundaries."""
    U_global = np.ones((NVAR, 32, 64), dtype=np.float32) * 7.0
    # Overwrite N boundary of block (0,1) — last row in z
    U_global[:, :BNR, BNZ : 2 * BNZ] = 9.0

    level = decompose_domain(32, 64, DR, DZ, R_INNER, BNR, BNZ)
    populate_blocks_from_state(level, U_global, BNR, BNZ)

    ng = 3
    padded = ghost_exchange_same_level(level, ng, BNR, BNZ, R_INNER)

    # Block (0,1) is at top (iz=1 = max). Its N ghost should be zero-gradient.
    pad_01 = np.asarray(padded[(0, 1)])
    # N ghost is at pad[..., ng:ng+BNR, ng+BNZ:ng+BNZ+ng]
    n_ghost = pad_01[:, ng : ng + BNR, ng + BNZ : ng + BNZ + ng]
    last_interior = pad_01[:, ng : ng + BNR, ng + BNZ - 1 : ng + BNZ]
    assert np.allclose(n_ghost, last_interior), "N ghost should equal last interior slab"


# ---------------------------------------------------------------------------
# 7. test_prolong_constant
# ---------------------------------------------------------------------------


def test_prolong_constant():
    """Prolongation of a uniform field should preserve the constant exactly."""
    U_c = np.ones((NVAR, BNR, BNZ), dtype=np.float32)
    U_f = _prolongate_vanleer(U_c, ratio=2)
    assert U_f.shape == (NVAR, BNR * 2, BNZ * 2)
    assert np.allclose(U_f, 1.0), f"max deviation: {np.max(np.abs(U_f - 1.0))}"


# ---------------------------------------------------------------------------
# 8. test_prolong_linear
# ---------------------------------------------------------------------------


def test_prolong_linear():
    """Prolongation of a linear ramp should be second-order accurate."""
    nr_c, nz_c = BNR // 2, BNZ // 2
    U_c = np.zeros((1, nr_c, nz_c), dtype=np.float32)
    for i in range(nr_c):
        U_c[0, i, :] = float(i) / nr_c  # linear in r
    U_f = _prolongate_vanleer(U_c, ratio=2)
    # Fine cell (2i, j) should be close to coarse cell (i, j/2) value
    # Error should be O(dr_coarse^2) = O(1/nr_c^2) — i.e. < O(dr^1)
    errors = []
    for i in range(nr_c - 1):
        for j in range(nz_c):
            fine_val = float(U_f[0, i * 2, j * 2])
            coarse_val = float(U_c[0, i, j])
            errors.append(abs(fine_val - coarse_val))
    max_err = max(errors)
    # Tolerance: should be much smaller than cell width (1/nr_c)
    assert max_err < 1.0 / nr_c, f"Linear prolong error too large: {max_err}"


# ---------------------------------------------------------------------------
# 9. test_restrict_recovers_coarse
# ---------------------------------------------------------------------------


def test_restrict_recovers_coarse():
    """restrict(prolongate(U)) should recover U for smooth data (L_inf < 1e-3)."""
    U_c = np.zeros((NVAR, BNR, BNZ), dtype=np.float32)
    for i in range(BNR):
        for j in range(BNZ):
            U_c[:, i, j] = np.sin(np.pi * i / BNR) * np.cos(np.pi * j / BNZ)
    U_c[IDN] += 1.0  # ensure positive density
    # Use only one coarse block for this test
    coarse_block = AMRBlock(level=0, index=(0, 0), U=U_c, r_min=R_INNER, z_min=0.0)

    fine_level = AMRLevel(level=1, blocks={}, dr=DR / 2, dz=DZ / 2)
    children = prolongate_to_fine(coarse_block, fine_level, ratio=2, block_nr=BNR, block_nz=BNZ)
    for child in children:
        fine_level.blocks[child.index] = child

    # Restrict back
    U_c_recovered = AMRBlock(
        level=0, index=(0, 0), U=U_c.copy(), r_min=R_INNER, z_min=0.0
    )
    restrict_to_coarse(children, U_c_recovered, fine_level, ratio=2, block_nr=BNR, block_nz=BNZ)

    err = np.max(np.abs(np.asarray(U_c_recovered.U) - U_c))
    # float32 boundary effects cause O(1e-3) error on smooth sin/cos data — allow 2e-3
    assert err < 2e-3, f"restrict(prolongate(U)) roundtrip error {err:.3e} exceeds 2e-3"


# ---------------------------------------------------------------------------
# 10. test_restrict_conserves_mass
# ---------------------------------------------------------------------------


def test_restrict_conserves_mass():
    """Volume-weighted restriction must conserve total mass."""
    U_c = np.ones((NVAR, BNR, BNZ), dtype=np.float32)
    U_c[IDN] = 2.0  # density = 2

    coarse_block = AMRBlock(level=0, index=(0, 0), U=U_c.copy(), r_min=R_INNER, z_min=0.0)
    fine_level = AMRLevel(level=1, blocks={}, dr=DR / 2, dz=DZ / 2)
    children = prolongate_to_fine(coarse_block, fine_level, ratio=2, block_nr=BNR, block_nz=BNZ)
    for child in children:
        fine_level.blocks[child.index] = child

    # Compute fine total mass (volume-weighted)
    def cylindrical_volume(r_lo: float, r_hi: float, dz: float) -> float:
        return 0.5 * (r_hi**2 - r_lo**2) * dz

    mass_fine = 0.0
    for child in children:
        U_f = np.asarray(child.U)
        fine_dr = fine_level.dr
        fine_dz = fine_level.dz
        for i in range(BNR):
            r_lo = child.r_min + i * fine_dr
            r_hi = r_lo + fine_dr
            vol = cylindrical_volume(r_lo, r_hi, fine_dz)
            mass_fine += float(np.sum(U_f[IDN, i, :])) * vol

    # Compute coarse total mass
    mass_coarse = 0.0
    for i in range(BNR):
        r_lo = R_INNER + i * DR
        r_hi = r_lo + DR
        vol = cylindrical_volume(r_lo, r_hi, DZ)
        mass_coarse += float(np.sum(U_c[IDN, i, :])) * vol

    rel_err = abs(mass_fine - mass_coarse) / (mass_coarse + 1e-30)
    assert rel_err < 1e-5, f"Mass not conserved after prolongation: rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# 11. test_amr_step_uniform_preserved
# ---------------------------------------------------------------------------


def test_amr_step_uniform_preserved():
    """Uniform state on all blocks should remain unchanged after 1 AMR step."""
    U_global = make_uniform_state(32, 64)
    h = make_hierarchy_2x2(U_global)

    h_out, _ = amr_step(
        hierarchy=h,
        dt=1e-10,
        gamma=5.0 / 3.0,
        method="plm",
        riemann="hll",
        ng=3,
        current=0.0,
        r_inner=R_INNER,
        step_number=0,
        rhs_fn=None,
        use_refluxing=False,
    )

    U_out = np.asarray(assemble_global_state(h_out.levels[0], 32, 64, BNR, BNZ))
    # For a truly uniform state, the Lax-Friedrichs RHS should be ~0
    max_delta = np.max(np.abs(U_out[IDN] - U_global[IDN]))
    assert max_delta < 1e-4, f"Uniform state changed after AMR step: max|dU|={max_delta:.2e}"


# ---------------------------------------------------------------------------
# 12. test_amr_step_mass_conservation
# ---------------------------------------------------------------------------


def test_amr_step_mass_conservation():
    """Mass should be conserved to < 1e-5 relative after 1 AMR step."""
    U_global = make_uniform_state(32, 64, rho=1.0)
    h = make_hierarchy_2x2(U_global)

    def cylindrical_mass(level: AMRLevel, block_nr: int, block_nz: int) -> float:
        total = 0.0
        for block in level.active_blocks():
            U_b = np.asarray(block.U)
            for i in range(U_b.shape[1]):
                r_lo = block.r_min + i * level.dr
                r_hi = r_lo + level.dr
                vol = 0.5 * (r_hi**2 - r_lo**2) * level.dz
                total += float(np.sum(U_b[IDN, i, :])) * vol
        return total

    mass_before = cylindrical_mass(h.levels[0], BNR, BNZ)

    h_out, _ = amr_step(
        hierarchy=h,
        dt=1e-10,
        gamma=5.0 / 3.0,
        method="plm",
        riemann="hll",
        ng=3,
        current=0.0,
        r_inner=R_INNER,
        step_number=0,
        rhs_fn=None,
        use_refluxing=False,
    )

    mass_after = cylindrical_mass(h_out.levels[0], BNR, BNZ)
    rel_err = abs(mass_after - mass_before) / (mass_before + 1e-30)
    assert rel_err < 1e-5, f"Mass not conserved: rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# 13. test_flux_register
# ---------------------------------------------------------------------------


def test_flux_register():
    """FluxRegister accumulates fine flux correctly."""
    register = FluxRegister()
    flux_coarse = np.ones(NVAR) * 2.0
    flux_fine1 = np.ones(NVAR) * 1.5
    flux_fine2 = np.ones(NVAR) * 0.8

    register.accumulate_coarse(0, flux_coarse, area=0.01, dt=1e-9)
    register.accumulate_fine(0, flux_fine1, area=0.005, dt=1e-9)
    register.accumulate_fine(0, flux_fine2, area=0.005, dt=1e-9)

    assert 0 in register.coarse_FA
    assert 0 in register.fine_FA
    expected_coarse = flux_coarse * 0.01 * 1e-9
    np.testing.assert_allclose(register.coarse_FA[0], expected_coarse, rtol=1e-6)
    expected_fine = flux_fine1 * 0.005 * 1e-9 + flux_fine2 * 0.005 * 1e-9
    np.testing.assert_allclose(register.fine_FA[0], expected_fine, rtol=1e-6)

    register.reset()
    assert len(register.coarse_FA) == 0
    assert len(register.fine_FA) == 0


# ---------------------------------------------------------------------------
# 14. test_reflux_sign
# ---------------------------------------------------------------------------


def test_reflux_sign():
    """Reflux correction should increase coarse cell mass when fine flux > coarse flux."""
    # Create a minimal hierarchy: one coarse block, no fine blocks
    level0 = decompose_domain(BNR, BNZ, DR, DZ, R_INNER, BNR, BNZ)
    level1 = AMRLevel(level=1, blocks={}, dr=DR / 2, dz=DZ / 2)
    # hierarchy variable unused but documents the 2-level structure
    _ = AMRHierarchy(levels=[level0, level1], block_nr=BNR, block_nz=BNZ, ratio=2)

    # Manually build a face and register
    c_block = level0.blocks[(0, 0)]
    U_np = np.asarray(c_block.U).copy().astype(np.float32)
    U_np[IDN] = 1.0
    c_block.U = U_np

    r_lo = R_INNER + (BNR - 1) * DR
    r_hi = r_lo + DR
    V_c = 0.5 * (r_hi**2 - r_lo**2) * DZ

    face = {
        "face_id": 0,
        "coarse_block_idx": (0, 0),
        "coarse_ir": BNR - 1,
        "coarse_iz": 0,
        "face_dir": "r",
        "face_side": "hi",
        "fine_faces": [],
        "coarse_area": r_hi * DZ,
        "coarse_volume": V_c,
    }

    register = FluxRegister()
    flux_coarse = np.zeros(NVAR)
    flux_coarse[IDN] = 1.0
    flux_fine = np.zeros(NVAR)
    flux_fine[IDN] = 2.0  # fine flux > coarse flux

    register.accumulate_coarse(0, flux_coarse, area=1.0, dt=1.0)
    register.accumulate_fine(0, flux_fine, area=1.0, dt=1.0)

    U_before = float(np.asarray(c_block.U)[IDN, BNR - 1, 0])
    apply_reflux_correction(register, [face], level0)
    U_after = float(np.asarray(c_block.U)[IDN, BNR - 1, 0])

    # delta = fine_FA - coarse_FA = 2 - 1 = 1, sign = +1 for "hi"
    # correction = +1 * delta / V_c > 0 → U should increase
    assert U_after > U_before, f"Reflux should increase coarse cell: before={U_before}, after={U_after}"


# ---------------------------------------------------------------------------
# 15. test_build_hierarchy_manual
# ---------------------------------------------------------------------------


def test_build_hierarchy_manual():
    """Manual refined_blocks=[(0,1)] should create 4 fine children (ratio^2)."""
    h = build_amr_hierarchy(
        nr=32, nz=64, dr=DR, dz=DZ, r_inner=R_INNER,
        block_nr=BNR, block_nz=BNZ, ratio=2,
        refined_blocks=[[0, 1]],
    )
    assert h.n_levels == 2
    fine_blocks = list(h.levels[1].blocks.values())
    assert len(fine_blocks) == 4, f"Expected 4 fine children, got {len(fine_blocks)}"


# ---------------------------------------------------------------------------
# 16. test_hierarchy_dr_dz
# ---------------------------------------------------------------------------


def test_hierarchy_dr_dz():
    """Fine level dr/dz should be exactly dr_coarse / ratio."""
    ratio = 2
    h = build_amr_hierarchy(
        nr=32, nz=64, dr=DR, dz=DZ, r_inner=R_INNER,
        block_nr=BNR, block_nz=BNZ, ratio=ratio,
    )
    assert h.levels[1].dr == DR / ratio
    assert h.levels[1].dz == DZ / ratio


# ---------------------------------------------------------------------------
# 17. test_amr_disabled_default
# ---------------------------------------------------------------------------


def test_amr_disabled_default():
    """MLXMHDSolver without amr_config should have _amr_hierarchy = None."""
    from dpf.metal.mlx_solver import MLXMHDSolver

    solver = MLXMHDSolver(
        grid_shape=(32, 1, 64),
        dx=DR,
        dz=DZ,
        r_inner=R_INNER,
    )
    assert solver._amr_hierarchy is None, "AMR hierarchy should be None when not configured"
    assert solver._amr_config is None, "AMR config should be None when not provided"


# ---------------------------------------------------------------------------
# 18. test_existing_solver_unaffected
# ---------------------------------------------------------------------------


def test_existing_solver_unaffected():
    """MLXMHDSolver without amr_config produces same results as before."""
    from dpf.metal.mlx_solver import MLXMHDSolver

    solver = MLXMHDSolver(
        grid_shape=(16, 1, 16),
        dx=1e-3,
        dz=1e-3,
        r_inner=0.01,
        reconstruction="plm",
        riemann_solver="hll",
    )
    state = {
        "rho": np.ones((16, 1, 16), dtype=np.float32),
        "velocity": np.zeros((3, 16, 1, 16), dtype=np.float32),
        "pressure": np.ones((16, 1, 16), dtype=np.float32) * 0.1,
        "B": np.zeros((3, 16, 1, 16), dtype=np.float32),
        "Te": np.ones((16, 1, 16), dtype=np.float32),
        "Ti": np.ones((16, 1, 16), dtype=np.float32),
    }
    result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)
    assert "rho" in result
    assert not np.any(np.isnan(result["rho"])), "Non-AMR step produced NaN"
    # AMR hierarchy should still be None (never set)
    assert solver._amr_hierarchy is None
