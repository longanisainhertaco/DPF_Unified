"""Multi-species impurity tracking for the MLX MHD solver.

Advects species mass fractions as sidecar arrays alongside the MHD
conserved state. Uses PLM-reconstructed upwind flux with the bulk
MHD velocity — no new Metal kernels required.

The background species (deuterium) is derived from the constraint
sum(Y_k) = 1, not evolved. Only impurity species are advected.

References:
    Stone et al. (2008), ApJS 178:137, p.3 Eq.(1) — conservation law framework.
        Passive scalar advection d(rho*Y)/dt + div(rho*Y*v) = 0 is a direct
        consequence of the conservation form with flux F = rho*Y*v.
        PDF on disk: references/papers/mhd-numerics/stone_2008_athena.pdf
    Poinsot & Veynante, Theoretical and Numerical Combustion (2005) —
        multi-species transport framework. NOT ON DISK.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from dpf.metal.mlx_kernels import IDN, IMR, IMZ
from dpf.metal.mlx_primitives import RHO_FLOOR
from dpf.metal.mlx_reconstruction import plm_reconstruct


@dataclass
class SpeciesManager:
    """Manages multi-species mass fractions on the MHD grid."""

    species: list[str]
    Z: list[int]
    A: list[float]
    background: str = "D"

    def __post_init__(self) -> None:
        self._idx = {name: i for i, name in enumerate(self.species)}
        self._bg_idx = self._idx[self.background]
        self._evolved_idx = [i for i in range(len(self.species)) if i != self._bg_idx]
        self.species_Z_mx = mx.array(self.Z, dtype=mx.float32)
        self.species_A_mx = mx.array(self.A, dtype=mx.float32)

    @property
    def n_evolved(self) -> int:
        return len(self._evolved_idx)

    @property
    def species_idx(self) -> dict[str, int]:
        return self._idx

    def init_mass_fractions(
        self, nr: int, nz: int, initial_fractions: dict[str, float] | None = None,
    ) -> mx.array:
        """Initialize evolved species mass densities (Y_k * rho, but stored as Y_k)."""
        Y = mx.zeros((self.n_evolved, nr, nz), dtype=mx.float32)
        if initial_fractions:
            for name, frac in initial_fractions.items():
                if name != self.background and name in self._idx:
                    evolved_pos = self._evolved_idx.index(self._idx[name])
                    Y = Y.at[evolved_pos].add(frac)
        return Y

    def recover_background(self, Y_evolved: mx.array) -> mx.array:
        """Reconstruct full species array including background from constraint."""
        Y_bg = 1.0 - mx.sum(Y_evolved, axis=0, keepdims=True)
        Y_bg = mx.maximum(Y_bg, 0.0)
        parts = []
        ev_idx = 0
        for i in range(len(self.species)):
            if i == self._bg_idx:
                parts.append(Y_bg)
            else:
                parts.append(Y_evolved[ev_idx : ev_idx + 1])
                ev_idx += 1
        return mx.concatenate(parts, axis=0)


def species_rhs(
    Y_rho: mx.array,
    vr: mx.array,
    vz: mx.array,
    dr: float,
    dz: float,
    r_cell: mx.array | None = None,
    r_face: mx.array | None = None,
) -> mx.array:
    """Compute dY_rho/dt = -div(Y_rho * v) for all species.

    Uses PLM-reconstructed upwind flux. Vectorized over species
    (axis 0 treated as variable dimension by plm_reconstruct).

    Args:
        Y_rho: Species mass densities, shape (N_sp, nr, nz).
        vr: Radial velocity [m/s], shape (nr, nz).
        vz: Axial velocity [m/s], shape (nr, nz).
        dr, dz: Cell spacings [m].
        r_cell: Cell-center radii for cylindrical weighting. None = Cartesian.
        r_face: Face radii. None = Cartesian.

    Returns:
        dY_rho/dt, shape (N_sp, nr, nz).
    """
    n_sp, nr, nz = Y_rho.shape
    dY = mx.zeros_like(Y_rho)

    # --- Radial sweep ---
    QL_r, QR_r = plm_reconstruct(Y_rho, dim=0, limiter="mc")
    n_if_r = QL_r.shape[1]
    vr_face = 0.5 * (vr[:n_if_r] + vr[1 : n_if_r + 1])
    vr_pos = mx.maximum(vr_face, 0.0)[None]
    vr_neg = mx.minimum(vr_face, 0.0)[None]
    F_r = QL_r * vr_pos + QR_r * vr_neg

    n_upd_r = n_if_r - 1
    if r_cell is not None and r_face is not None:
        r_L = r_face[1 : 1 + n_upd_r, None]
        r_R = r_face[2 : 2 + n_upd_r, None]
        r_c = r_cell[1 : 1 + n_upd_r, None]
        div_Fr = -(r_R * F_r[:, 1:, :] - r_L * F_r[:, :n_upd_r, :]) / (r_c * dr)
    else:
        div_Fr = -(F_r[:, 1:, :] - F_r[:, :n_upd_r, :]) / dr

    pad_l = mx.zeros((n_sp, 1, nz))
    pad_r_arr = mx.zeros((n_sp, nr - 1 - n_upd_r, nz))
    dY = dY + mx.concatenate([pad_l, div_Fr, pad_r_arr], axis=1)

    # --- Axial sweep ---
    QL_z, QR_z = plm_reconstruct(Y_rho, dim=1, limiter="mc")
    n_if_z = QL_z.shape[2]
    vz_face = 0.5 * (vz[:, :n_if_z] + vz[:, 1 : n_if_z + 1])
    vz_pos = mx.maximum(vz_face, 0.0)[None]
    vz_neg = mx.minimum(vz_face, 0.0)[None]
    F_z = QL_z * vz_pos + QR_z * vz_neg

    n_upd_z = n_if_z - 1
    div_Fz = -(F_z[:, :, 1:] - F_z[:, :, :n_upd_z]) / dz
    pad_l_z = mx.zeros((n_sp, nr, 1))
    pad_r_z = mx.zeros((n_sp, nr, nz - 1 - n_upd_z))
    dY = dY + mx.concatenate([pad_l_z, div_Fz, pad_r_z], axis=2)

    return dY


def species_advection_step(
    Y: mx.array,
    U: mx.array,
    dr: float,
    dz: float,
    dt: float,
    gamma: float,
    r_cell: mx.array | None = None,
    r_face: mx.array | None = None,
) -> mx.array:
    """SSP-RK2 species advection using MHD bulk velocity.

    Args:
        Y: Species mass fractions, shape (N_sp, nr, nz).
        U: MHD conserved state (NVAR, nr, nz) for velocity extraction.
        dr, dz: Cell spacings.
        dt: Timestep.
        gamma: Adiabatic index (unused, for API compatibility).
        r_cell, r_face: Cylindrical geometry arrays.

    Returns:
        Updated Y, shape (N_sp, nr, nz).
    """
    rho = mx.maximum(U[IDN], RHO_FLOOR)
    inv_rho = 1.0 / rho
    vr = U[IMR] * inv_rho
    vz = U[IMZ] * inv_rho

    # Convert Y (mass fractions) to Y_rho (mass densities) for conservative advection
    Y_rho = Y * rho[None]

    # SSP-RK2 (simpler, sufficient for passive scalars)
    L0 = species_rhs(Y_rho, vr, vz, dr, dz, r_cell, r_face)
    Y_rho_1 = Y_rho + dt * L0
    L1 = species_rhs(Y_rho_1, vr, vz, dr, dz, r_cell, r_face)
    Y_rho_new = 0.5 * Y_rho + 0.5 * (Y_rho_1 + dt * L1)

    # Convert back to mass fractions
    rho_new = mx.maximum(rho, RHO_FLOOR)
    Y_new = Y_rho_new / rho_new[None]
    return mx.maximum(Y_new, 0.0)


def apply_ablation_sources(
    Y: mx.array,
    dt: float,
    ablation_rate: mx.array,
    cu_idx: int,
) -> mx.array:
    """Apply electrode ablation source to Cu species fraction.

    Args:
        Y: Evolved species fractions, shape (N_evolved, nr, nz).
        dt: Timestep [s].
        ablation_rate: Cu mass injection rate [1/s], shape (nr, nz).
        cu_idx: Index of Cu in the evolved species array.

    Returns:
        Updated Y with Cu fraction increased at electrode cells.
    """
    delta = ablation_rate * dt
    Y = Y.at[cu_idx].add(delta)
    return mx.maximum(Y, 0.0)


def compute_zeff_field(
    Y_full: mx.array,
    species_Z: mx.array,
    species_A: mx.array,
) -> mx.array:
    """Compute Zeff from species composition (fully ionized approximation).

    Zeff = sum(n_k * Z_k^2) / sum(n_k * Z_k)
    where n_k proportional to Y_k / A_k.

    Args:
        Y_full: All species fractions, shape (N_species, nr, nz).
        species_Z: Atomic numbers, shape (N_species,).
        species_A: Atomic masses [amu], shape (N_species,).

    Returns:
        Zeff field, shape (nr, nz).
    """
    n_weight = Y_full / species_A[:, None, None]
    Z = species_Z[:, None, None]
    num = mx.sum(n_weight * Z * Z, axis=0)
    den = mx.sum(n_weight * Z, axis=0)
    zeff = num / mx.maximum(den, 1e-30)

    # Mask vacuum cells: where total mass fraction is negligible,
    # Z_eff from trace Cu impurities climbs to 24+ → catastrophic radiation.
    # Return Z_eff=1 (pure hydrogen) in vacuum regions.
    Y_total = mx.sum(Y_full, axis=0)
    return mx.where(Y_total < 1e-4, 1.0, zeff)


def pad_species_ghost(Y: mx.array, ng: int) -> mx.array:
    """Zero-gradient ghost padding for species array."""
    inner = mx.repeat(Y[:, :1, :], ng, axis=1)
    outer = mx.repeat(Y[:, -1:, :], ng, axis=1)
    return mx.concatenate([inner, Y, outer], axis=1)
