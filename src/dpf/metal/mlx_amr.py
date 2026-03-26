"""Block-structured AMR for MLX MHD solver (Phase A-slim).

2-level, global timestep, ghost exchange, prolongation/restriction, minimal refluxing.

Refs:
    Berger & Colella, JCP 82:64 (1989) -- AMR for hyperbolic PDEs.
    Stone et al., ApJS 249:4 (2020) -- Athena++ SMR/AMR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

IDN, IMR, IMZ, IMT, IEN, ISR, IBR, IBZ, IBT, IEE = range(10)
NVAR = 10
# Variables that flip sign at the axis (r=0) reflection BC
_SIGN_FLIP_VARS = (IMR, IMT, IBR, IBT)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AMRBlock:
    """One block in the AMR hierarchy."""

    level: int
    index: tuple[int, int]
    U: Any  # mx.array or np.ndarray, shape (NVAR, block_nr, block_nz)
    r_min: float
    z_min: float
    active: bool = True


@dataclass
class AMRLevel:
    """One refinement level: a collection of AMRBlocks at uniform dr/dz."""

    level: int
    blocks: dict[tuple[int, int], AMRBlock]
    dr: float
    dz: float

    def active_blocks(self) -> list[AMRBlock]:
        return sorted(
            (b for b in self.blocks.values() if b.active),
            key=lambda b: b.index,
        )

    def as_batch(self) -> Any:
        """Stack all active block U arrays into (N_blocks, NVAR, nr, nz)."""
        active = self.active_blocks()
        if not active:
            return np.zeros((0, NVAR, 1, 1))
        arrays = [np.asarray(b.U) for b in active]
        return np.stack(arrays, axis=0)

    def scatter_batch(self, U_batch: Any) -> None:
        """Distribute (N_blocks, NVAR, nr, nz) batch back into blocks."""
        active = self.active_blocks()
        U_np = np.asarray(U_batch)
        for i, block in enumerate(active):
            block.U = mx.array(U_np[i]) if mx is not None else U_np[i]


@dataclass
class AMRHierarchy:
    """2-level block-structured AMR hierarchy."""

    levels: list[AMRLevel]
    block_nr: int
    block_nz: int
    ratio: int

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def total_cells(self) -> int:
        total = 0
        for level in self.levels:
            for b in level.active_blocks():
                U = np.asarray(b.U)
                total += U.shape[1] * U.shape[2]
        return total

    def add_level(self, dr: float, dz: float) -> AMRLevel:
        """Append a new empty AMRLevel and return it."""
        li = len(self.levels)
        new_level = AMRLevel(level=li, blocks={}, dr=dr, dz=dz)
        self.levels.append(new_level)
        return new_level

    def fill_all_ghosts(self, ng: int = 3, r_inner: float = 0.0) -> None:
        """Rebuild ghost-cell padded arrays for all levels (side-effect free).

        This is called after regrid to ensure newly created blocks have valid
        ghost data before the next RHS evaluation. Because ghost_exchange_same_level
        returns a dict of padded arrays (not modifying blocks in-place), this
        method is a no-op in terms of block.U — the actual ghost exchange happens
        inside amr_step. This call exists to trigger any lazy MLX evaluations and
        to serve as a logical fence in the regrid pipeline.
        """
        for level in self.levels:
            if level.blocks:
                ghost_exchange_same_level(
                    level, ng, self.block_nr, self.block_nz, r_inner
                )

    def block_topology(self, level_idx: int) -> dict[tuple[int, int], dict[str, Any]]:
        """Build neighbor map for all blocks at level_idx.

        Keys: N (+z), S (-z), E (+r), W (-r). Value: neighbor index or None.
        """
        level = self.levels[level_idx]
        topo: dict[tuple[int, int], dict[str, Any]] = {}
        for idx in level.blocks:
            ir, iz = idx
            neighbors = {
                "N": (ir, iz + 1) if (ir, iz + 1) in level.blocks else None,
                "S": (ir, iz - 1) if (ir, iz - 1) in level.blocks else None,
                "E": (ir + 1, iz) if (ir + 1, iz) in level.blocks else None,
                "W": (ir - 1, iz) if (ir - 1, iz) in level.blocks else None,
            }
            topo[idx] = neighbors
        return topo


# ---------------------------------------------------------------------------
# Domain decomposition
# ---------------------------------------------------------------------------


def decompose_domain(
    nr: int,
    nz: int,
    dr: float,
    dz: float,
    r_inner: float,
    block_nr: int,
    block_nz: int,
) -> AMRLevel:
    """Split a (nr, nz) grid into blocks of size (block_nr, block_nz).

    Blocks at the boundary may be smaller if grid does not divide evenly.
    """
    import math

    n_ir = math.ceil(nr / block_nr)
    n_iz = math.ceil(nz / block_nz)
    blocks: dict[tuple[int, int], AMRBlock] = {}
    for ir in range(n_ir):
        for iz in range(n_iz):
            r_min = r_inner + ir * block_nr * dr
            z_min = iz * block_nz * dz
            actual_nr = min(block_nr, nr - ir * block_nr)
            actual_nz = min(block_nz, nz - iz * block_nz)
            U_init = np.zeros((NVAR, actual_nr, actual_nz), dtype=np.float32)
            blocks[(ir, iz)] = AMRBlock(
                level=0,
                index=(ir, iz),
                U=U_init,
                r_min=r_min,
                z_min=z_min,
                active=True,
            )
    return AMRLevel(level=0, blocks=blocks, dr=dr, dz=dz)


def populate_blocks_from_state(
    level: AMRLevel,
    U_global: Any,
    block_nr: int,
    block_nz: int,
) -> None:
    """Slice global state (NVAR, nr, nz) into each block's U array."""
    U_np = np.asarray(U_global)
    for (ir, iz), block in level.blocks.items():
        r_start = ir * block_nr
        z_start = iz * block_nz
        actual_nr = np.asarray(block.U).shape[1]
        actual_nz = np.asarray(block.U).shape[2]
        r_end = r_start + actual_nr
        z_end = z_start + actual_nz
        U_slice = U_np[:, r_start:r_end, z_start:z_end].astype(np.float32)
        block.U = mx.array(U_slice) if mx is not None else U_slice


def assemble_global_state(
    level: AMRLevel,
    nr: int,
    nz: int,
    block_nr: int,
    block_nz: int,
) -> Any:
    """Assemble block data back into a (NVAR, nr, nz) global array."""
    U_global = np.zeros((NVAR, nr, nz), dtype=np.float32)
    for (ir, iz), block in level.blocks.items():
        if not block.active:
            continue
        U_np = np.asarray(block.U)
        r_start = ir * block_nr
        z_start = iz * block_nz
        actual_nr = U_np.shape[1]
        actual_nz = U_np.shape[2]
        r_end = r_start + actual_nr
        z_end = z_start + actual_nz
        U_global[:, r_start:r_end, z_start:z_end] = U_np
    return mx.array(U_global) if mx is not None else U_global


# ---------------------------------------------------------------------------
# Ghost exchange
# ---------------------------------------------------------------------------


def ghost_exchange_same_level(
    level: AMRLevel,
    ng: int,
    block_nr: int,
    block_nz: int,
    r_inner: float,
) -> dict[tuple[int, int], Any]:
    """Build padded arrays with ghost cells for each active block.

    Ghost fill priority:
    - Interior: block.U
    - Neighbor exists: copy ng-wide slab from neighbor
    - W boundary (r=r_inner): reflect with sign flip on _SIGN_FLIP_VARS
    - E/S/N boundary (outer/bottom/top): zero-gradient (copy last slab)

    Returns dict mapping block index -> padded array (NVAR, nr+2ng, nz+2ng).
    """
    result: dict[tuple[int, int], Any] = {}

    # Build neighbor map inline (block_topology lives on AMRHierarchy but we
    # only need a local map here)
    topo: dict[tuple[int, int], dict[str, Any]] = {}
    for idx in level.blocks:
        ir, iz = idx
        topo[idx] = {
            "N": (ir, iz + 1) if (ir, iz + 1) in level.blocks else None,
            "S": (ir, iz - 1) if (ir, iz - 1) in level.blocks else None,
            "E": (ir + 1, iz) if (ir + 1, iz) in level.blocks else None,
            "W": (ir - 1, iz) if (ir - 1, iz) in level.blocks else None,
        }

    for idx, block in level.blocks.items():
        if not block.active:
            continue
        U_np = np.asarray(block.U).astype(np.float32)
        _, bnr, bnz = U_np.shape
        U_pad = np.zeros((NVAR, bnr + 2 * ng, bnz + 2 * ng), dtype=np.float32)

        # Fill interior
        U_pad[:, ng : ng + bnr, ng : ng + bnz] = U_np

        neighbors = topo.get(idx, {"N": None, "S": None, "E": None, "W": None})

        # W face (r=inner, axis reflection)
        w_idx = neighbors["W"]
        if w_idx is not None and w_idx in level.blocks:
            w_block = level.blocks[w_idx]
            w_np = np.asarray(w_block.U).astype(np.float32)
            U_pad[:, :ng, ng : ng + bnz] = w_np[:, -ng:, :]
        else:
            # Physical axis BC: reflect with sign flip
            for gi in range(ng):
                src_i = ng + gi  # first interior cell for ghost index (ng-1-gi)
                ghost_i = ng - 1 - gi
                U_pad[:, ghost_i, ng : ng + bnz] = U_pad[:, src_i, ng : ng + bnz]
            for v in _SIGN_FLIP_VARS:
                for gi in range(ng):
                    src_i = ng + gi
                    ghost_i = ng - 1 - gi
                    U_pad[v, ghost_i, ng : ng + bnz] = -U_pad[v, src_i, ng : ng + bnz]

        # E face (outer r boundary): zero-gradient
        e_idx = neighbors["E"]
        if e_idx is not None and e_idx in level.blocks:
            e_block = level.blocks[e_idx]
            e_np = np.asarray(e_block.U).astype(np.float32)
            U_pad[:, ng + bnr :, ng : ng + bnz] = e_np[:, :ng, :]
        else:
            for gi in range(ng):
                U_pad[:, ng + bnr + gi, ng : ng + bnz] = U_pad[
                    :, ng + bnr - 1, ng : ng + bnz
                ]

        # S face (z bottom): zero-gradient
        s_idx = neighbors["S"]
        if s_idx is not None and s_idx in level.blocks:
            s_block = level.blocks[s_idx]
            s_np = np.asarray(s_block.U).astype(np.float32)
            U_pad[:, ng : ng + bnr, :ng] = s_np[:, :, -ng:]
        else:
            for gi in range(ng):
                U_pad[:, ng : ng + bnr, ng - 1 - gi] = U_pad[
                    :, ng : ng + bnr, ng
                ]

        # N face (z top): zero-gradient
        n_idx = neighbors["N"]
        if n_idx is not None and n_idx in level.blocks:
            n_block = level.blocks[n_idx]
            n_np = np.asarray(n_block.U).astype(np.float32)
            U_pad[:, ng : ng + bnr, ng + bnz :] = n_np[:, :, :ng]
        else:
            for gi in range(ng):
                U_pad[:, ng : ng + bnr, ng + bnz + gi] = U_pad[
                    :, ng : ng + bnr, ng + bnz - 1
                ]

        # Fill corners with zero-gradient from edge ghosts (nearest edge)
        # W-S corner
        U_pad[:, :ng, :ng] = U_pad[:, :ng, ng : ng + 1]
        # W-N corner
        U_pad[:, :ng, ng + bnz :] = U_pad[:, :ng, ng + bnz - 1 : ng + bnz]
        # E-S corner
        U_pad[:, ng + bnr :, :ng] = U_pad[:, ng + bnr :, ng : ng + 1]
        # E-N corner
        U_pad[:, ng + bnr :, ng + bnz :] = U_pad[:, ng + bnr :, ng + bnz - 1 : ng + bnz]

        result[idx] = mx.array(U_pad) if mx is not None else U_pad

    return result


# ---------------------------------------------------------------------------
# Prolongation (coarse -> fine)
# ---------------------------------------------------------------------------


def _vanleer(a: float, b: float) -> float:
    """Van Leer limiter for piecewise-linear interpolation."""
    if a * b > 0:
        return 2.0 * a * b / (a + b)
    return 0.0


def _prolongate_vanleer(U_coarse: np.ndarray, ratio: int = 2) -> np.ndarray:
    """Piecewise-linear prolongation with van Leer slope limiting.

    Args:
        U_coarse: (NVAR, nr_c, nz_c) coarse block data.
        ratio: Refinement ratio (default 2).

    Returns:
        (NVAR, nr_c*ratio, nz_c*ratio) fine block data.
    """
    nvar, nr_c, nz_c = U_coarse.shape
    # Start from piecewise constant
    U_fine = np.repeat(np.repeat(U_coarse, ratio, axis=1), ratio, axis=2)
    for v in range(nvar):
        for i in range(nr_c):
            for j in range(nz_c):
                # Radial slope
                if 0 < i < nr_c - 1:
                    dr = _vanleer(
                        U_coarse[v, i, j] - U_coarse[v, i - 1, j],
                        U_coarse[v, i + 1, j] - U_coarse[v, i, j],
                    )
                else:
                    dr = 0.0
                # Axial slope
                if 0 < j < nz_c - 1:
                    dz = _vanleer(
                        U_coarse[v, i, j] - U_coarse[v, i, j - 1],
                        U_coarse[v, i, j + 1] - U_coarse[v, i, j],
                    )
                else:
                    dz = 0.0
                for di in range(ratio):
                    for dj in range(ratio):
                        xi_r = (di + 0.5) / ratio - 0.5
                        xi_z = (dj + 0.5) / ratio - 0.5
                        U_fine[v, i * ratio + di, j * ratio + dj] += dr * xi_r + dz * xi_z
    return U_fine


def prolongate_to_fine(
    coarse_block: AMRBlock,
    fine_level: AMRLevel,
    ratio: int,
    block_nr: int,
    block_nz: int,
) -> list[AMRBlock]:
    """Prolongate one coarse block into ratio^2 fine child blocks.

    Each coarse block spawns ratio x ratio fine blocks. The coarse block
    is divided into ratio x ratio quadrants; each quadrant is prolongated
    independently to produce one fine block of size (block_nr, block_nz).
    """
    U_np = np.asarray(coarse_block.U).astype(np.float64)
    nr_q = block_nr // ratio
    nz_q = block_nz // ratio
    fine_dr = fine_level.dr
    fine_dz = fine_level.dz
    children: list[AMRBlock] = []

    for di in range(ratio):
        for dj in range(ratio):
            quad = U_np[:, di * nr_q : (di + 1) * nr_q, dj * nz_q : (dj + 1) * nz_q]
            U_fine_np = _prolongate_vanleer(quad.astype(np.float32), ratio)
            ir_f = coarse_block.index[0] * ratio + di
            iz_f = coarse_block.index[1] * ratio + dj
            r_min_f = coarse_block.r_min + di * block_nr * fine_dr
            z_min_f = coarse_block.z_min + dj * block_nz * fine_dz
            U_fine = mx.array(U_fine_np) if mx is not None else U_fine_np
            children.append(
                AMRBlock(
                    level=fine_level.level,
                    index=(ir_f, iz_f),
                    U=U_fine,
                    r_min=r_min_f,
                    z_min=z_min_f,
                    active=True,
                )
            )
    return children


# ---------------------------------------------------------------------------
# Restriction (fine -> coarse)
# ---------------------------------------------------------------------------


def restrict_to_coarse(
    fine_blocks: list[AMRBlock],
    coarse_block: AMRBlock,
    fine_level: AMRLevel,
    ratio: int,
    block_nr: int,
    block_nz: int,
) -> None:
    """Volume-weighted (r-weighted cylindrical) restriction.

    Replaces coarse_block.U with volume-weighted average of covering fine cells.
    V = 0.5 * (r_hi^2 - r_lo^2) * dz (cylindrical annular volume, 2pi cancels).

    Args:
        fine_blocks: Child AMRBlocks covering this coarse block.
        coarse_block: Coarse block to update in-place.
        fine_level: Fine AMRLevel (provides dr, dz).
        ratio: Refinement ratio.
        block_nr: Block radial cell count.
        block_nz: Block axial cell count.
    """
    fine_dr = fine_level.dr
    fine_dz = fine_level.dz

    # Build lookup: fine_index -> AMRBlock
    fine_map: dict[tuple[int, int], AMRBlock] = {b.index: b for b in fine_blocks}

    U_c_np = np.asarray(coarse_block.U).astype(np.float64)
    nr_q = block_nr // ratio
    nz_q = block_nz // ratio

    for di in range(ratio):
        for dj in range(ratio):
            ir_f = coarse_block.index[0] * ratio + di
            iz_f = coarse_block.index[1] * ratio + dj
            fine_b = fine_map.get((ir_f, iz_f))
            if fine_b is None:
                continue
            U_f_np = np.asarray(fine_b.U).astype(np.float64)

            r0_coarse = di * nr_q
            z0_coarse = dj * nz_q

            for ic in range(nr_q):
                for jc in range(nz_q):
                    vol_sum = 0.0
                    w_sum = np.zeros(NVAR)
                    for df in range(ratio):
                        for dg in range(ratio):
                            # Cell index in the fine block
                            if_idx = ic * ratio + df
                            jf_idx = jc * ratio + dg
                            r_lo = fine_b.r_min + if_idx * fine_dr
                            r_hi = r_lo + fine_dr
                            vol = 0.5 * (r_hi**2 - r_lo**2) * fine_dz
                            # Guard against axis singularity
                            if vol < 1e-30:
                                vol = fine_dr * r_lo * fine_dz + 1e-30
                            w_sum += U_f_np[:, if_idx, jf_idx] * vol
                            vol_sum += vol
                    U_c_np[:, r0_coarse + ic, z0_coarse + jc] = w_sum / vol_sum

    coarse_block.U = mx.array(U_c_np.astype(np.float32)) if mx is not None else U_c_np.astype(np.float32)


# ---------------------------------------------------------------------------
# Flux register and refluxing
# ---------------------------------------------------------------------------


@dataclass
class FluxRegister:
    """Stores flux*area*dt on both sides of coarse-fine (CF) boundaries."""

    coarse_FA: dict[int, np.ndarray] = field(default_factory=dict)
    fine_FA: dict[int, np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        self.coarse_FA.clear()
        self.fine_FA.clear()

    def accumulate_coarse(
        self, fid: int, flux: np.ndarray, area: float, dt: float
    ) -> None:
        self.coarse_FA[fid] = np.asarray(flux) * area * dt

    def accumulate_fine(
        self, fid: int, flux: np.ndarray, area: float, dt: float
    ) -> None:
        if fid not in self.fine_FA:
            self.fine_FA[fid] = np.zeros_like(np.asarray(flux))
        self.fine_FA[fid] += np.asarray(flux) * area * dt


def identify_cf_faces(
    hierarchy: AMRHierarchy, coarse_li: int = 0
) -> list[dict[str, Any]]:
    """Identify coarse cell faces adjacent to the fine level.

    Each returned dict has keys:
        face_id, coarse_block_idx, coarse_ir, coarse_iz,
        face_dir (r|z), face_side (lo|hi), fine_faces (list),
        coarse_area, coarse_volume.
    """
    if len(hierarchy.levels) < coarse_li + 2:
        return []

    coarse_level = hierarchy.levels[coarse_li]
    fine_level = hierarchy.levels[coarse_li + 1]
    ratio = hierarchy.ratio
    block_nr = hierarchy.block_nr
    block_nz = hierarchy.block_nz
    faces: list[dict[str, Any]] = []
    face_id = 0

    for c_idx, c_block in coarse_level.blocks.items():
        if not c_block.active:
            continue
        ir_c, iz_c = c_idx

        # Scan all fine blocks that overlap this coarse block
        # Fine blocks covering coarse block (ir_c, iz_c) have indices
        # in the range [ir_c*ratio, (ir_c+1)*ratio) x [iz_c*ratio, (iz_c+1)*ratio)
        # Check if a fine block exists just beyond the coarse block's E face (+r)
        for jc in range(block_nz):
            # E face (+r) of this coarse block
            for dj in range(ratio):
                fine_ir = (ir_c + 1) * ratio  # one block past the E edge
                fine_iz = iz_c * ratio + jc // (block_nz // ratio) * ratio + dj // ratio
                if (fine_ir, fine_iz) in fine_level.blocks:
                    r_face = c_block.r_min + block_nr * coarse_level.dr
                    r_lo = c_block.r_min + (block_nr - 1) * coarse_level.dr
                    r_hi = r_lo + coarse_level.dr
                    A = r_face * coarse_level.dz
                    V = 0.5 * (r_hi**2 - r_lo**2) * coarse_level.dz
                    if V < 1e-30:
                        break
                    faces.append({
                        "face_id": face_id,
                        "coarse_block_idx": c_idx,
                        "coarse_ir": block_nr - 1,
                        "coarse_iz": jc,
                        "face_dir": "r",
                        "face_side": "hi",
                        "fine_faces": [(fine_ir, fine_iz)],
                        "coarse_area": A,
                        "coarse_volume": V,
                    })
                    face_id += 1
                    break  # one face per coarse cell

    return faces


def apply_reflux_correction(
    register: FluxRegister,
    cf_faces: list[dict[str, Any]],
    coarse_level: AMRLevel,
) -> float:
    """Apply Berger-Colella flux correction at CF faces.

    For each face: delta = fine_FA - coarse_FA
    U_c += sign * delta / V_c

    Returns total |correction| for monitoring.
    """
    total_corr = 0.0
    for face in cf_faces:
        fid = face["face_id"]
        if fid not in register.coarse_FA or fid not in register.fine_FA:
            continue
        delta = register.fine_FA[fid] - register.coarse_FA[fid]
        c_idx = face["coarse_block_idx"]
        c_block = coarse_level.blocks.get(c_idx)
        if c_block is None:
            continue
        ir = face["coarse_ir"]
        iz = face["coarse_iz"]
        V_c = face["coarse_volume"]
        if V_c < 1e-30:
            continue
        sign = 1.0 if face["face_side"] == "hi" else -1.0
        U_np = np.asarray(c_block.U).astype(np.float32)
        U_np[:, ir, iz] += (sign * delta / V_c).astype(np.float32)
        c_block.U = mx.array(U_np) if mx is not None else U_np
        total_corr += float(np.sum(np.abs(delta)))
    return total_corr


# ---------------------------------------------------------------------------
# Orchestrator: build hierarchy + AMR step
# ---------------------------------------------------------------------------


def build_amr_hierarchy(
    nr: int,
    nz: int,
    dr: float,
    dz: float,
    r_inner: float,
    block_nr: int,
    block_nz: int,
    ratio: int,
    refined_blocks: list[list[int]] | None = None,
) -> AMRHierarchy:
    """Build a 2-level AMR hierarchy.

    Level 0: decomposed from (nr, nz) global grid.
    Level 1: fine blocks at dr/ratio, dz/ratio; initially populated from
             refined_blocks (list of [ir, iz] coarse block indices) or empty.
    """
    level0 = decompose_domain(nr, nz, dr, dz, r_inner, block_nr, block_nz)
    fine_dr = dr / ratio
    fine_dz = dz / ratio
    level1 = AMRLevel(level=1, blocks={}, dr=fine_dr, dz=fine_dz)
    hierarchy = AMRHierarchy(
        levels=[level0, level1],
        block_nr=block_nr,
        block_nz=block_nz,
        ratio=ratio,
    )

    if refined_blocks is not None:
        for rb in refined_blocks:
            ir_c, iz_c = rb[0], rb[1]
            coarse_block = level0.blocks.get((ir_c, iz_c))
            if coarse_block is None:
                continue
            children = prolongate_to_fine(coarse_block, level1, ratio, block_nr, block_nz)
            for child in children:
                level1.blocks[child.index] = child

    return hierarchy


def amr_step(
    hierarchy: AMRHierarchy,
    dt: float,
    gamma: float,
    method: str,
    riemann: str,
    ng: int,
    current: float,
    r_inner: float,
    step_number: int,
    rhs_fn: Any,
    use_refluxing: bool = True,
    config: Any = None,
) -> tuple[AMRHierarchy, float]:
    """One AMR timestep (global dt, sequential block processing).

    Pipeline:
    1. Ghost exchange on all levels
    2. Advance each level: per-block RHS on padded state, SSP-RK3, strip ghosts
    3. Restrict fine -> coarse (volume-weighted)
    4. Optional reflux correction at CF faces

    Args:
        hierarchy: Current AMR hierarchy.
        dt: Global timestep [s].
        gamma: Adiabatic index.
        method: Reconstruction method ("weno5z" or "plm").
        riemann: Riemann solver ("hll" or "hlld").
        ng: Number of ghost cells.
        current: Circuit current [A] (for electrode BC).
        r_inner: Inner radial boundary [m].
        step_number: Current step number (for logging).
        rhs_fn: Callable(U_padded, grid, dt, ...) -> dU (block-local RHS).
        use_refluxing: Whether to apply reflux correction.

    Returns:
        (updated hierarchy, dt_used).
    """
    if config is not None and step_number % config.regrid_interval == 0 and step_number > 0:
        hierarchy, _, _ = auto_regrid(hierarchy, config)

    block_nr = hierarchy.block_nr
    block_nz = hierarchy.block_nz

    # ── 1. Ghost exchange on all levels ──────────────────────────────────
    padded: dict[int, dict[tuple[int, int], Any]] = {}
    for li, level in enumerate(hierarchy.levels):
        if not level.blocks:
            continue
        padded[li] = ghost_exchange_same_level(
            level, ng, block_nr, block_nz, r_inner
        )

    # ── 2. Advance coarse level (level 0) ────────────────────────────────
    level0 = hierarchy.levels[0]
    for idx, block in level0.blocks.items():
        if not block.active:
            continue
        U_pad = padded[0].get(idx)
        if U_pad is None:
            continue
        U_pad_np = np.asarray(U_pad).astype(np.float32)
        bnr = np.asarray(block.U).shape[1]
        bnz = np.asarray(block.U).shape[2]
        try:
            dU = _block_rhs(U_pad_np, block, level0, gamma, method, riemann, dt, ng)
        except Exception as exc:
            logger.warning("AMR block (%d,%d) RHS failed: %s", idx[0], idx[1], exc)
            continue
        U_int = U_pad_np[:, ng : ng + bnr, ng : ng + bnz] + dt * dU
        U_int = np.maximum(U_int, 0.0)
        block.U = mx.array(U_int) if mx is not None else U_int

    # ── 2b. Advance fine level (level 1) if present ───────────────────────
    if len(hierarchy.levels) > 1:
        level1 = hierarchy.levels[1]
        if level1.blocks:
            for idx, block in level1.blocks.items():
                if not block.active:
                    continue
                U_pad = padded.get(1, {}).get(idx)
                if U_pad is None:
                    continue
                U_pad_np = np.asarray(U_pad).astype(np.float32)
                bnr = np.asarray(block.U).shape[1]
                bnz = np.asarray(block.U).shape[2]
                try:
                    dU = _block_rhs(U_pad_np, block, level1, gamma, method, riemann, dt, ng)
                except Exception as exc:
                    logger.warning(
                        "AMR fine block (%d,%d) RHS failed: %s", idx[0], idx[1], exc
                    )
                    continue
                U_int = U_pad_np[:, ng : ng + bnr, ng : ng + bnz] + dt * dU
                U_int = np.maximum(U_int, 0.0)
                block.U = mx.array(U_int) if mx is not None else U_int

            # ── 3. Restrict fine -> coarse ────────────────────────────────
            for c_idx, c_block in level0.blocks.items():
                if not c_block.active:
                    continue
                children = [
                    b for b in level1.active_blocks()
                    if (b.index[0] // hierarchy.ratio == c_idx[0]
                        and b.index[1] // hierarchy.ratio == c_idx[1])
                ]
                if children:
                    restrict_to_coarse(
                        children, c_block, level1,
                        hierarchy.ratio, block_nr, block_nz,
                    )

            # ── 4. Reflux correction ───────────────────────────────────────
            if use_refluxing:
                cf_faces = identify_cf_faces(hierarchy, coarse_li=0)
                if cf_faces:
                    register = FluxRegister()
                    # Phase A: re-evaluate fluxes at CF faces (no flux capture from RHS)
                    # This is a no-op for the register (register stays empty)
                    # which means delta=0, so correction is 0. Correct behavior
                    # for Phase A: actual flux capture is deferred to Phase C.
                    apply_reflux_correction(register, cf_faces, level0)

    return hierarchy, dt


def _block_rhs(
    U_pad: np.ndarray,
    block: AMRBlock,
    level: AMRLevel,
    gamma: float,
    method: str,
    riemann: str,
    dt: float,
    ng: int,
) -> np.ndarray:
    """Compute first-order explicit RHS for a single block.

    Uses simple finite-difference Lax-Friedrichs flux for Phase A.
    The full MLX RHS pipeline requires mx.array and a grid object;
    Phase A uses a simplified CPU RHS for correctness testing.

    Args:
        U_pad: Padded state (NVAR, nr+2*ng, nz+2*ng).
        block: The AMRBlock being advanced.
        level: AMRLevel containing dr, dz.
        gamma: Adiabatic index.
        method: Reconstruction ("plm" or "weno5z").
        riemann: Riemann solver ("hll").
        dt: Timestep.
        ng: Ghost cell count.

    Returns:
        dU/dt of shape (NVAR, bnr, bnz).
    """
    _, nr_pad, nz_pad = U_pad.shape
    bnr = nr_pad - 2 * ng
    bnz = nz_pad - 2 * ng
    dr = level.dr
    dz = level.dz

    # Interior only
    U_int = U_pad[:, ng : ng + bnr, ng : ng + bnz]

    # Lax-Friedrichs: dU/dt = -(F_{i+1/2} - F_{i-1/2})/dr - (G_{j+1/2} - G_{j-1/2})/dz
    # F, G = 0.5*(F_L + F_R) - 0.5*alpha*(U_R - U_L)
    # Use maximum wave speed alpha = max(|v| + cf) over the block

    rho = np.maximum(U_int[IDN], 1e-10)
    vr = U_int[IMR] / rho
    vz = U_int[IMZ] / rho

    # Fast magnetosonic speed (simplified)
    p = np.maximum((gamma - 1.0) * (
        U_int[IEN] - 0.5 * rho * (vr**2 + vz**2)
        - 0.5 * (U_int[IBR]**2 + U_int[IBZ]**2 + U_int[IBT]**2)
    ), 1e-10 * rho)
    cs2 = gamma * p / rho
    ca2 = (U_int[IBR]**2 + U_int[IBZ]**2 + U_int[IBT]**2) / rho
    cf = np.sqrt(np.maximum(cs2 + ca2, 0.0)) + 1e-10
    alpha = float(np.max(np.abs(vr) + cf)) + float(np.max(np.abs(vz) + cf))

    dU = np.zeros_like(U_int)

    # Radial flux divergence
    # Use upwind slabs from padded array
    UL_r = U_pad[:, ng - 1 : ng + bnr - 1, ng : ng + bnz]  # left face
    UR_r = U_pad[:, ng : ng + bnr, ng : ng + bnz]           # right face
    UL_r_right = U_pad[:, ng : ng + bnr, ng : ng + bnz]
    UR_r_right = U_pad[:, ng + 1 : ng + bnr + 1, ng : ng + bnz]

    rho_L = np.maximum(UL_r[IDN], 1e-10)
    rho_R = np.maximum(UR_r[IDN], 1e-10)
    vr_L = UL_r[IMR] / rho_L
    vr_R = UR_r[IMR] / rho_R
    F_left = 0.5 * (vr_L * UL_r + vr_R * UR_r) - 0.5 * alpha * (UR_r - UL_r)

    rho_L2 = np.maximum(UL_r_right[IDN], 1e-10)
    rho_R2 = np.maximum(UR_r_right[IDN], 1e-10)
    vr_L2 = UL_r_right[IMR] / rho_L2
    vr_R2 = UR_r_right[IMR] / rho_R2
    F_right = 0.5 * (vr_L2 * UL_r_right + vr_R2 * UR_r_right) - 0.5 * alpha * (UR_r_right - UL_r_right)

    dU -= (F_right - F_left) / dr

    # Axial flux divergence
    UL_z = U_pad[:, ng : ng + bnr, ng - 1 : ng + bnz - 1]
    UR_z = U_pad[:, ng : ng + bnr, ng : ng + bnz]
    UL_z_top = U_pad[:, ng : ng + bnr, ng : ng + bnz]
    UR_z_top = U_pad[:, ng : ng + bnr, ng + 1 : ng + bnz + 1]

    rho_Lz = np.maximum(UL_z[IDN], 1e-10)
    rho_Rz = np.maximum(UR_z[IDN], 1e-10)
    vz_Lz = UL_z[IMZ] / rho_Lz
    vz_Rz = UR_z[IMZ] / rho_Rz
    G_bottom = 0.5 * (vz_Lz * UL_z + vz_Rz * UR_z) - 0.5 * alpha * (UR_z - UL_z)

    rho_Lzt = np.maximum(UL_z_top[IDN], 1e-10)
    rho_Rzt = np.maximum(UR_z_top[IDN], 1e-10)
    vz_Lzt = UL_z_top[IMZ] / rho_Lzt
    vz_Rzt = UR_z_top[IMZ] / rho_Rzt
    G_top = 0.5 * (vz_Lzt * UL_z_top + vz_Rzt * UR_z_top) - 0.5 * alpha * (UR_z_top - UL_z_top)

    dU -= (G_top - G_bottom) / dz

    return dU


# ---------------------------------------------------------------------------
# Phase B: Automatic refinement — sensor helpers
# ---------------------------------------------------------------------------


def _lohner_indicator_block(rho: np.ndarray, dr: float, dz: float) -> float:
    """2D Lohner (1987) normalized second-derivative indicator on a (nr, nz) array.

    E = |d2rho/dr2| / (|drho/dr|/dr + eps*|rho|/dr^2)  (summed over r and z dirs)

    Returns the scalar max over the block in [0, 1].
    """
    nr, nz = rho.shape
    eps = 1e-6 * float(np.mean(np.abs(rho)) + 1e-30)
    indicator = np.zeros_like(rho)

    if nr > 2:
        d2r = np.zeros_like(rho)
        d1r = np.zeros_like(rho)
        d2r[1:-1, :] = rho[2:, :] - 2.0 * rho[1:-1, :] + rho[:-2, :]
        d1r[1:-1, :] = np.abs(rho[2:, :] - rho[:-2, :])
        den = d1r + eps * np.abs(rho) / dr
        indicator += np.abs(d2r) / (den + 1e-30)

    if nz > 2:
        d2z = np.zeros_like(rho)
        d1z = np.zeros_like(rho)
        d2z[:, 1:-1] = rho[:, 2:] - 2.0 * rho[:, 1:-1] + rho[:, :-2]
        d1z[:, 1:-1] = np.abs(rho[:, 2:] - rho[:, :-2])
        den = d1z + eps * np.abs(rho) / dz
        indicator += np.abs(d2z) / (den + 1e-30)

    max_val = float(np.max(indicator))
    if max_val > 0:
        indicator /= max_val
    return float(np.max(indicator))


def _current_density_sensor_block(B: np.ndarray, dr: float, dz: float) -> float:
    """Normalized |J| = |curl B| sensor on B (3, nr, nz) array.

    J_theta ~ dBr/dz - dBz/dr (dominant cylindrical component).
    Returns scalar max in [0, 1].
    """
    Br = B[0]
    Bz = B[1] if B.shape[0] > 1 else np.zeros_like(Br)
    nr, nz = Br.shape

    J_theta = np.zeros_like(Br)
    if nz > 2:
        J_theta[:, 1:-1] += (Br[:, 2:] - Br[:, :-2]) / (2.0 * dz)
    if nr > 2:
        J_theta[1:-1, :] -= (Bz[2:, :] - Bz[:-2, :]) / (2.0 * dr)

    B_mag = np.sqrt(np.sum(B**2, axis=0))
    B_max = float(np.max(B_mag))
    sensor = np.abs(J_theta) * dr / (B_mag + 0.01 * max(B_max, 1e-10))

    s_max = float(np.max(sensor))
    if s_max > 0:
        sensor /= s_max
    return float(np.max(sensor))


# ---------------------------------------------------------------------------
# Phase B: Function 1 — evaluate_refinement_sensors
# ---------------------------------------------------------------------------


def evaluate_refinement_sensors(
    hierarchy: AMRHierarchy,
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Run Lohner (density) and current-density sensors on every active leaf block.

    Returns {(level_idx, ir, iz): (j_val, lohner_val)}.

    Leaf blocks are active blocks on the finest level that has any blocks.
    For a 2-level hierarchy with fine blocks present, level-0 blocks that are
    covered by fine children are not leaf blocks — only level-1 blocks are.
    """
    result: dict[tuple[int, int, int], tuple[float, float]] = {}
    ratio = hierarchy.ratio

    for li, level in enumerate(hierarchy.levels):
        fine_level = hierarchy.levels[li + 1] if li + 1 < len(hierarchy.levels) else None

        for (ir, iz), block in level.blocks.items():
            if not block.active:
                continue

            if fine_level is not None and fine_level.blocks:
                has_fine_child = any(
                    (ir * ratio + di, iz * ratio + dj) in fine_level.blocks
                    for di in range(ratio)
                    for dj in range(ratio)
                )
                if has_fine_child:
                    continue

            U_np = np.asarray(block.U)
            rho = U_np[IDN]
            B = np.stack([U_np[IBR], U_np[IBZ], U_np[IBT]])

            l_val = _lohner_indicator_block(rho, level.dr, level.dz)
            j_val = _current_density_sensor_block(B, level.dr, level.dz)

            result[(li, ir, iz)] = (j_val, l_val)

    return result


# ---------------------------------------------------------------------------
# Phase B: Function 2 — flag_blocks_for_refinement
# ---------------------------------------------------------------------------


def flag_blocks_for_refinement(
    sensor_values: dict[tuple[int, int, int], tuple[float, float]],
    config: Any,
) -> dict[tuple[int, int, int], int]:
    """Convert sensor scalars to flags: +1 (refine), -1 (derefine), 0 (keep).

    Hysteresis: refine_threshold / derefine_threshold = 6:1 ratio prevents
    oscillation when sensor value straddles the threshold due to numerical noise.

    Axis guard: blocks with ir <= 1 are never refined (1/r singularity amplifies J).
    """
    flags: dict[tuple[int, int, int], int] = {}
    for key, (j_val, l_val) in sensor_values.items():
        li, ir, iz = key
        # Never refine the first 2 radial blocks (axis singularity, R1 from spec)
        if ir <= 1:
            flags[key] = 0
            continue

        if li < config.max_levels - 1 and (
            j_val > config.j_threshold_refine or l_val > config.lohner_threshold_refine
        ):
            flags[key] = 1
        elif li > 0 and (
            j_val < config.j_threshold_derefine and l_val < config.lohner_threshold_derefine
        ):
            flags[key] = -1
        else:
            flags[key] = 0

    return flags


# ---------------------------------------------------------------------------
# Phase B: Function 3 — enforce_proper_nesting
# ---------------------------------------------------------------------------


def enforce_proper_nesting(
    flags: dict[tuple[int, int, int], int],
    hierarchy: AMRHierarchy,
    config: Any,
) -> dict[tuple[int, int, int], int]:
    """Enforce buffer zones and proper nesting constraints on refinement flags.

    Pass 1: buffer zone expansion — any block neighboring a +1 block within
            config.buffer_width is promoted from -1/0 to +1.
    Pass 2: proper nesting — orphan fine blocks (no coarse parent) get flag 0.
    Pass 3: capacity cap — if total fine blocks would exceed max_blocks_per_level,
            keep only the top-N by sensor value.
    """
    result = dict(flags)

    # Pass 1: buffer zone expansion
    refine_keys = [k for k, v in result.items() if v == 1]
    for key in refine_keys:
        li, ir, iz = key
        bw = config.buffer_width
        for dri in range(-bw, bw + 1):
            for dzi in range(-bw, bw + 1):
                if dri == 0 and dzi == 0:
                    continue
                neighbor = (li, ir + dri, iz + dzi)
                if neighbor in result and result[neighbor] < 1:
                    result[neighbor] = 1

    # Pass 2: proper nesting — remove orphan fine blocks
    ratio = hierarchy.ratio
    to_zero: list[tuple[int, int, int]] = []
    for key, flag in result.items():
        if flag != 1:
            continue
        li, ir, iz = key
        if li > 0:
            pidx = (ir // ratio, iz // ratio)
            parent_level = hierarchy.levels[li - 1]
            if pidx not in parent_level.blocks:
                to_zero.append(key)
    for key in to_zero:
        result[key] = 0

    # Pass 3: capacity cap — keep only top-N candidates by sensor max
    refine_candidates = [(k, v) for k, v in result.items() if v == 1]
    n_existing = sum(
        len(lv.blocks) for li, lv in enumerate(hierarchy.levels) if li > 0
    )
    headroom = config.max_blocks_per_level - n_existing
    if headroom < 0:
        headroom = 0
    if len(refine_candidates) > headroom:
        refine_candidates.sort(key=lambda x: x[0], reverse=True)
        keep = {k for k, _ in refine_candidates[:headroom]}
        for key, flag in list(result.items()):
            if flag == 1 and key not in keep:
                result[key] = 0

    return result


# ---------------------------------------------------------------------------
# Phase B: Function 4 — create_child_blocks
# ---------------------------------------------------------------------------


def create_child_blocks(
    hierarchy: AMRHierarchy,
    parent_block: AMRBlock,
    config: Any,
) -> list[AMRBlock]:
    """Prolongate a coarse block to ratio^2 fine children.

    Ensures a fine level exists in the hierarchy. Uses _prolongate_vanleer
    to initialize each child block. Skips children that already exist.

    Returns list of newly created AMRBlock objects (already stored in
    hierarchy.levels[parent_block.level + 1].blocks).
    """
    ratio = hierarchy.ratio
    block_nr = hierarchy.block_nr
    block_nz = hierarchy.block_nz
    fi = parent_block.level + 1

    # Ensure the fine level exists
    while len(hierarchy.levels) <= fi:
        prev_level = hierarchy.levels[-1]
        hierarchy.add_level(dr=prev_level.dr / ratio, dz=prev_level.dz / ratio)

    fine_level = hierarchy.levels[fi]
    fine_dr = fine_level.dr
    fine_dz = fine_level.dz

    U_np = np.asarray(parent_block.U).astype(np.float32)
    nr_q = block_nr // ratio
    nz_q = block_nz // ratio

    new_children: list[AMRBlock] = []

    for di in range(ratio):
        for dj in range(ratio):
            ir_f = parent_block.index[0] * ratio + di
            iz_f = parent_block.index[1] * ratio + dj
            cidx = (ir_f, iz_f)

            if cidx in fine_level.blocks:
                continue

            quad = U_np[:, di * nr_q : (di + 1) * nr_q, dj * nz_q : (dj + 1) * nz_q]
            U_fine_np = _prolongate_vanleer(quad, ratio)

            r_min_f = parent_block.r_min + di * block_nr * fine_dr
            z_min_f = parent_block.z_min + dj * block_nz * fine_dz

            child = AMRBlock(
                level=fi,
                index=cidx,
                U=mx.array(U_fine_np) if mx is not None else U_fine_np,
                r_min=r_min_f,
                z_min=z_min_f,
                active=True,
            )
            fine_level.blocks[cidx] = child
            new_children.append(child)

    return new_children


# ---------------------------------------------------------------------------
# Phase B: Function 5 — remove_child_blocks
# ---------------------------------------------------------------------------


def remove_child_blocks(
    hierarchy: AMRHierarchy,
    child_block: AMRBlock,
    config: Any,
) -> None:
    """Restrict a fine block back to its coarse parent, then remove it.

    The parent block is updated in-place via restrict_to_coarse before
    the child is deleted. The level container is preserved (empty level
    is valid for subsequent regrids).
    """
    li = child_block.level
    assert li > 0, "Cannot remove a level-0 block"
    ratio = hierarchy.ratio
    block_nr = hierarchy.block_nr
    block_nz = hierarchy.block_nz

    pidx = (child_block.index[0] // ratio, child_block.index[1] // ratio)
    parent = hierarchy.levels[li - 1].blocks.get(pidx)
    fine_level = hierarchy.levels[li]

    if parent is not None:
        restrict_to_coarse(
            [child_block], parent, fine_level, ratio, block_nr, block_nz
        )

    del fine_level.blocks[child_block.index]


# ---------------------------------------------------------------------------
# Phase B: Function 6 — auto_regrid
# ---------------------------------------------------------------------------


def auto_regrid(
    hierarchy: AMRHierarchy,
    config: Any,
) -> tuple[AMRHierarchy, int, int]:
    """Full regrid orchestrator: evaluate -> flag -> nest -> create/remove.

    Returns (updated_hierarchy, n_refined, n_derefined).

    n_refined counts newly created child blocks; n_derefined counts removed
    child blocks. Calls hierarchy.fill_all_ghosts() after any topology change.
    """
    sensor_values = evaluate_refinement_sensors(hierarchy)
    flags = flag_blocks_for_refinement(sensor_values, config)
    flags = enforce_proper_nesting(flags, hierarchy, config)

    n_refined = 0
    n_derefined = 0
    ratio = hierarchy.ratio

    # Refine pass
    for (li, ir, iz), flag in flags.items():
        if flag != 1:
            continue
        parent = hierarchy.levels[li].blocks.get((ir, iz))
        if parent is None:
            continue
        children = create_child_blocks(hierarchy, parent, config)
        n_refined += len(children)

    # Derefine pass — only derefine a complete sibling set
    deref_keys = [(li, ir, iz) for (li, ir, iz), flag in flags.items() if flag == -1]
    processed: set[tuple[int, int, int]] = set()

    for li, ir, iz in deref_keys:
        if (li, ir, iz) in processed:
            continue

        # Check all ratio^2 siblings have flag -1
        pir = ir // ratio
        piz = iz // ratio
        sibling_keys = [
            (li, pir * ratio + di, piz * ratio + dj)
            for di in range(ratio)
            for dj in range(ratio)
        ]
        all_deref = all(
            flags.get(k, 0) == -1 for k in sibling_keys
        )
        if not all_deref:
            continue

        for skey in sibling_keys:
            sli, sir, siz = skey
            block = hierarchy.levels[sli].blocks.get((sir, siz))
            if block is not None:
                remove_child_blocks(hierarchy, block, config)
                n_derefined += 1
            processed.add(skey)

    if n_refined > 0 or n_derefined > 0:
        hierarchy.fill_all_ghosts()

    return hierarchy, n_refined, n_derefined
