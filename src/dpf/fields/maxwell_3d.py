"""Minimal 3-D full-Maxwell field core on the repo's Yee/CT layout.

The arXiv:2604.09032v1 KnowledgeReference source makes the full 3-D
hybrid-PIC finish line explicit: electromagnetic fields must evolve in
plasma and vacuum, including boundary semantics for conductors and open/PML
regions.  This module is the first independently testable field slice for
that path.  It is not a complete DPF simulator and does not promote neutron
yield or same-scope validation claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np

from dpf.fluid.constrained_transport import (
    StaggeredBField,
    compute_div_B,
    ct_update,
    face_to_cell_centered,
)

EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
SPEED_OF_LIGHT = 1.0 / sqrt(EPSILON_0 * MU_0)

HYBRID_PIC_3D_SOURCE = (
    "KnowledgeReference/"
    "fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md"
)


@dataclass(frozen=True)
class Maxwell3DGrid:
    """Uniform Cartesian grid for a Yee-layout full-Maxwell field solve."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or len(self.spacing) != 3:
            raise ValueError("shape and spacing must be 3-tuples")
        if any(int(n) != n or n < 2 for n in self.shape):
            raise ValueError("all grid dimensions must be integers >= 2")
        if any(float(d) <= 0.0 for d in self.spacing):
            raise ValueError("all grid spacings must be positive")

    @property
    def nx(self) -> int:
        return int(self.shape[0])

    @property
    def ny(self) -> int:
        return int(self.shape[1])

    @property
    def nz(self) -> int:
        return int(self.shape[2])

    @property
    def dx(self) -> float:
        return float(self.spacing[0])

    @property
    def dy(self) -> float:
        return float(self.spacing[1])

    @property
    def dz(self) -> float:
        return float(self.spacing[2])

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz


@dataclass(frozen=True)
class Maxwell3DBoundaries:
    """Boundary controls for the first 3-D Maxwell field slice.

    ``conductor_cells`` marks cells whose adjacent electric-field edges are
    held at zero.  ``pml_cells`` and ``pml_strength`` provide deterministic
    per-step damping near logical grid boundaries; they are engineering
    boundary semantics, not a validated PML coefficient model.
    """

    conductor_cells: np.ndarray | None = None
    pml_cells: int = 0
    pml_strength: float = 0.0
    open_boundary: bool = True

    def __post_init__(self) -> None:
        if int(self.pml_cells) != self.pml_cells or self.pml_cells < 0:
            raise ValueError("pml_cells must be a non-negative integer")
        if float(self.pml_strength) < 0.0:
            raise ValueError("pml_strength must be non-negative")


@dataclass
class YeeElectricField:
    """Electric field on Yee-grid edges."""

    Ex_edge: np.ndarray
    Ey_edge: np.ndarray
    Ez_edge: np.ndarray

    def copy(self) -> YeeElectricField:
        return YeeElectricField(
            Ex_edge=np.array(self.Ex_edge, copy=True),
            Ey_edge=np.array(self.Ey_edge, copy=True),
            Ez_edge=np.array(self.Ez_edge, copy=True),
        )


@dataclass
class Maxwell3DState:
    """Field state with edge-centered E and face-centered B."""

    E: YeeElectricField
    B: StaggeredBField

    @property
    def Ex_edge(self) -> np.ndarray:
        return self.E.Ex_edge

    @property
    def Ey_edge(self) -> np.ndarray:
        return self.E.Ey_edge

    @property
    def Ez_edge(self) -> np.ndarray:
        return self.E.Ez_edge

    def copy(self) -> Maxwell3DState:
        return Maxwell3DState(
            E=self.E.copy(),
            B=StaggeredBField(
                Bx_face=np.array(self.B.Bx_face, copy=True),
                By_face=np.array(self.B.By_face, copy=True),
                Bz_face=np.array(self.B.Bz_face, copy=True),
                dx=self.B.dx,
                dy=self.B.dy,
                dz=self.B.dz,
            ),
        )


@dataclass(frozen=True)
class Maxwell3DDiagnostics:
    """Energy and divergence diagnostics for a field state."""

    electric_energy_J: float
    magnetic_energy_J: float
    total_energy_J: float
    max_abs_div_B_T_per_m: float
    stable_vacuum_dt_s: float
    grid_shape: tuple[int, int, int]
    source: str = HYBRID_PIC_3D_SOURCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Maxwell3DFieldCore:
    """First 3-D full-Maxwell field core for the hybrid-PIC finish line."""

    capability_id = "full_maxwell_vacuum_plasma_fields"

    def __init__(
        self,
        grid: Maxwell3DGrid,
        boundaries: Maxwell3DBoundaries | None = None,
    ) -> None:
        self.grid = grid
        self.boundaries = boundaries or Maxwell3DBoundaries()
        self._validate_boundary_shape()
        self._conductor_edges = self._build_conductor_edge_masks()
        self._E_damping = (
            _pml_factor(_ex_shape(grid), self.boundaries.pml_cells, self.boundaries.pml_strength),
            _pml_factor(_ey_shape(grid), self.boundaries.pml_cells, self.boundaries.pml_strength),
            _pml_factor(_ez_shape(grid), self.boundaries.pml_cells, self.boundaries.pml_strength),
        )
        self._B_damping = (
            _pml_factor(_bx_shape(grid), self.boundaries.pml_cells, self.boundaries.pml_strength),
            _pml_factor(_by_shape(grid), self.boundaries.pml_cells, self.boundaries.pml_strength),
            _pml_factor(_bz_shape(grid), self.boundaries.pml_cells, self.boundaries.pml_strength),
        )

    @classmethod
    def zeros(
        cls,
        grid: Maxwell3DGrid,
        boundaries: Maxwell3DBoundaries | None = None,
    ) -> Maxwell3DState:
        return cls(grid, boundaries).empty_state()

    def empty_state(self) -> Maxwell3DState:
        grid = self.grid
        return Maxwell3DState(
            E=YeeElectricField(
                Ex_edge=np.zeros(_ex_shape(grid), dtype=float),
                Ey_edge=np.zeros(_ey_shape(grid), dtype=float),
                Ez_edge=np.zeros(_ez_shape(grid), dtype=float),
            ),
            B=StaggeredBField(
                Bx_face=np.zeros(_bx_shape(grid), dtype=float),
                By_face=np.zeros(_by_shape(grid), dtype=float),
                Bz_face=np.zeros(_bz_shape(grid), dtype=float),
                dx=grid.dx,
                dy=grid.dy,
                dz=grid.dz,
            ),
        )

    def stable_vacuum_dt(self, cfl: float = 0.99) -> float:
        """Return the 3-D vacuum Courant limit for the Yee FDTD update."""
        if cfl <= 0.0 or cfl > 1.0:
            raise ValueError("cfl must satisfy 0 < cfl <= 1")
        grid = self.grid
        inverse_length = sqrt(
            1.0 / grid.dx**2 + 1.0 / grid.dy**2 + 1.0 / grid.dz**2
        )
        return cfl / (SPEED_OF_LIGHT * inverse_length)

    def validate_state(self, state: Maxwell3DState) -> None:
        grid = self.grid
        _require_shape("Ex_edge", state.Ex_edge, _ex_shape(grid))
        _require_shape("Ey_edge", state.Ey_edge, _ey_shape(grid))
        _require_shape("Ez_edge", state.Ez_edge, _ez_shape(grid))
        _require_shape("Bx_face", state.B.Bx_face, _bx_shape(grid))
        _require_shape("By_face", state.B.By_face, _by_shape(grid))
        _require_shape("Bz_face", state.B.Bz_face, _bz_shape(grid))
        if (state.B.dx, state.B.dy, state.B.dz) != self.grid.spacing:
            raise ValueError("magnetic field spacing does not match grid")

    def curl_B_to_edges(self, B: StaggeredBField) -> YeeElectricField:
        """Return discrete curl(B) sampled on electric-field edges."""
        grid = self.grid
        _require_shape("Bx_face", B.Bx_face, _bx_shape(grid))
        _require_shape("By_face", B.By_face, _by_shape(grid))
        _require_shape("Bz_face", B.Bz_face, _bz_shape(grid))

        curl_x = np.zeros(_ex_shape(grid), dtype=float)
        curl_y = np.zeros(_ey_shape(grid), dtype=float)
        curl_z = np.zeros(_ez_shape(grid), dtype=float)

        nx, ny, nz = grid.shape
        curl_x[:, 1:ny, 1:nz] = (
            (B.Bz_face[:, 1:ny, 1:nz] - B.Bz_face[:, 0 : ny - 1, 1:nz]) / grid.dy
            - (B.By_face[:, 1:ny, 1:nz] - B.By_face[:, 1:ny, 0 : nz - 1]) / grid.dz
        )
        curl_y[1:nx, :, 1:nz] = (
            (B.Bx_face[1:nx, :, 1:nz] - B.Bx_face[1:nx, :, 0 : nz - 1]) / grid.dz
            - (B.Bz_face[1:nx, :, 1:nz] - B.Bz_face[0 : nx - 1, :, 1:nz]) / grid.dx
        )
        curl_z[1:nx, 1:ny, :] = (
            (B.By_face[1:nx, 1:ny, :] - B.By_face[0 : nx - 1, 1:ny, :]) / grid.dx
            - (B.Bx_face[1:nx, 1:ny, :] - B.Bx_face[1:nx, 0 : ny - 1, :]) / grid.dy
        )

        return YeeElectricField(curl_x, curl_y, curl_z)

    def step_electric(
        self,
        state: Maxwell3DState,
        dt: float,
        current_density: YeeElectricField | None = None,
    ) -> Maxwell3DState:
        """Advance E by Ampere-Maxwell with optional edge current density."""
        self.validate_state(state)
        if dt < 0.0:
            raise ValueError("dt must be non-negative")

        curl_B = self.curl_B_to_edges(state.B)
        E = state.E.copy()
        E.Ex_edge += SPEED_OF_LIGHT**2 * dt * curl_B.Ex_edge
        E.Ey_edge += SPEED_OF_LIGHT**2 * dt * curl_B.Ey_edge
        E.Ez_edge += SPEED_OF_LIGHT**2 * dt * curl_B.Ez_edge

        if current_density is not None:
            _require_shape("Jx_edge", current_density.Ex_edge, _ex_shape(self.grid))
            _require_shape("Jy_edge", current_density.Ey_edge, _ey_shape(self.grid))
            _require_shape("Jz_edge", current_density.Ez_edge, _ez_shape(self.grid))
            E.Ex_edge -= dt * current_density.Ex_edge / EPSILON_0
            E.Ey_edge -= dt * current_density.Ey_edge / EPSILON_0
            E.Ez_edge -= dt * current_density.Ez_edge / EPSILON_0

        next_state = Maxwell3DState(E=E, B=state.copy().B)
        return self.apply_boundary_conditions(next_state)

    def step_magnetic(self, state: Maxwell3DState, dt: float) -> Maxwell3DState:
        """Advance B by Faraday's law with constrained transport."""
        self.validate_state(state)
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if dt == 0.0:
            next_B = state.copy().B
        else:
            next_B = ct_update(
                state.B,
                state.Ex_edge,
                state.Ey_edge,
                state.Ez_edge,
                dt,
            )
        damped_B = StaggeredBField(
            Bx_face=next_B.Bx_face * self._B_damping[0],
            By_face=next_B.By_face * self._B_damping[1],
            Bz_face=next_B.Bz_face * self._B_damping[2],
            dx=self.grid.dx,
            dy=self.grid.dy,
            dz=self.grid.dz,
        )
        return Maxwell3DState(E=state.E.copy(), B=damped_B)

    def step(
        self,
        state: Maxwell3DState,
        dt: float,
        current_density: YeeElectricField | None = None,
    ) -> Maxwell3DState:
        """Advance E and B by one explicit full-Maxwell field step."""
        state_with_E = self.step_electric(state, dt, current_density=current_density)
        return self.step_magnetic(state_with_E, dt)

    def apply_boundary_conditions(self, state: Maxwell3DState) -> Maxwell3DState:
        """Apply conductor electric constraints and deterministic PML damping."""
        self.validate_state(state)
        E = state.E.copy()
        E.Ex_edge *= self._E_damping[0]
        E.Ey_edge *= self._E_damping[1]
        E.Ez_edge *= self._E_damping[2]

        if self._conductor_edges is not None:
            E.Ex_edge[self._conductor_edges[0]] = 0.0
            E.Ey_edge[self._conductor_edges[1]] = 0.0
            E.Ez_edge[self._conductor_edges[2]] = 0.0

        return Maxwell3DState(E=E, B=state.copy().B)

    def edge_E_to_cell_centered(
        self,
        E: YeeElectricField,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average edge-centered electric fields to cell centers."""
        grid = self.grid
        _require_shape("Ex_edge", E.Ex_edge, _ex_shape(grid))
        _require_shape("Ey_edge", E.Ey_edge, _ey_shape(grid))
        _require_shape("Ez_edge", E.Ez_edge, _ez_shape(grid))
        Ex = 0.25 * (
            E.Ex_edge[:, :-1, :-1]
            + E.Ex_edge[:, 1:, :-1]
            + E.Ex_edge[:, :-1, 1:]
            + E.Ex_edge[:, 1:, 1:]
        )
        Ey = 0.25 * (
            E.Ey_edge[:-1, :, :-1]
            + E.Ey_edge[1:, :, :-1]
            + E.Ey_edge[:-1, :, 1:]
            + E.Ey_edge[1:, :, 1:]
        )
        Ez = 0.25 * (
            E.Ez_edge[:-1, :-1, :]
            + E.Ez_edge[1:, :-1, :]
            + E.Ez_edge[:-1, 1:, :]
            + E.Ez_edge[1:, 1:, :]
        )
        return Ex, Ey, Ez

    def electric_energy_J(self, state: Maxwell3DState) -> float:
        self.validate_state(state)
        Ex, Ey, Ez = self.edge_E_to_cell_centered(state.E)
        density = 0.5 * EPSILON_0 * (Ex**2 + Ey**2 + Ez**2)
        return float(np.sum(density) * self.grid.cell_volume)

    def magnetic_energy_J(self, state: Maxwell3DState) -> float:
        self.validate_state(state)
        Bx, By, Bz = face_to_cell_centered(state.B)
        density = 0.5 * (Bx**2 + By**2 + Bz**2) / MU_0
        return float(np.sum(density) * self.grid.cell_volume)

    def divergence_B(self, state: Maxwell3DState) -> np.ndarray:
        self.validate_state(state)
        return compute_div_B(state.B)

    def diagnostics(self, state: Maxwell3DState) -> Maxwell3DDiagnostics:
        electric = self.electric_energy_J(state)
        magnetic = self.magnetic_energy_J(state)
        div_B = self.divergence_B(state)
        return Maxwell3DDiagnostics(
            electric_energy_J=electric,
            magnetic_energy_J=magnetic,
            total_energy_J=electric + magnetic,
            max_abs_div_B_T_per_m=float(np.max(np.abs(div_B))),
            stable_vacuum_dt_s=self.stable_vacuum_dt(),
            grid_shape=self.grid.shape,
        )

    def _validate_boundary_shape(self) -> None:
        mask = self.boundaries.conductor_cells
        if mask is not None:
            _require_shape("conductor_cells", np.asarray(mask), self.grid.shape)

    def _build_conductor_edge_masks(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        mask = self.boundaries.conductor_cells
        if mask is None:
            return None
        cells = np.asarray(mask, dtype=bool)

        Ex = np.zeros(_ex_shape(self.grid), dtype=bool)
        Ex[:, :-1, :-1] |= cells
        Ex[:, 1:, :-1] |= cells
        Ex[:, :-1, 1:] |= cells
        Ex[:, 1:, 1:] |= cells

        Ey = np.zeros(_ey_shape(self.grid), dtype=bool)
        Ey[:-1, :, :-1] |= cells
        Ey[1:, :, :-1] |= cells
        Ey[:-1, :, 1:] |= cells
        Ey[1:, :, 1:] |= cells

        Ez = np.zeros(_ez_shape(self.grid), dtype=bool)
        Ez[:-1, :-1, :] |= cells
        Ez[1:, :-1, :] |= cells
        Ez[:-1, 1:, :] |= cells
        Ez[1:, 1:, :] |= cells

        return Ex, Ey, Ez


def maxwell_3d_field_capability_evidence(
    *,
    passed: bool,
    test_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Build gated evidence for only the full-Maxwell field capability."""
    status = "accepted" if passed else "blocked"
    return {
        "passed": bool(passed),
        "status": status,
        "capability": Maxwell3DFieldCore.capability_id,
        "source": HYBRID_PIC_3D_SOURCE,
        "implementation": "src/dpf/fields/maxwell_3d.py",
        "evidence_type": "engineering_component_verification",
        "test_ids": list(test_ids),
        "limitations": [
            "Does not supply ion PIC push/deposition evidence.",
            "Does not supply electron-fluid generalized Ohm closure evidence.",
            "Does not supply same-scope 3-D DPF validation evidence.",
        ],
    }


def _ex_shape(grid: Maxwell3DGrid) -> tuple[int, int, int]:
    return (grid.nx, grid.ny + 1, grid.nz + 1)


def _ey_shape(grid: Maxwell3DGrid) -> tuple[int, int, int]:
    return (grid.nx + 1, grid.ny, grid.nz + 1)


def _ez_shape(grid: Maxwell3DGrid) -> tuple[int, int, int]:
    return (grid.nx + 1, grid.ny + 1, grid.nz)


def _bx_shape(grid: Maxwell3DGrid) -> tuple[int, int, int]:
    return (grid.nx + 1, grid.ny, grid.nz)


def _by_shape(grid: Maxwell3DGrid) -> tuple[int, int, int]:
    return (grid.nx, grid.ny + 1, grid.nz)


def _bz_shape(grid: Maxwell3DGrid) -> tuple[int, int, int]:
    return (grid.nx, grid.ny, grid.nz + 1)


def _require_shape(name: str, value: np.ndarray, expected: tuple[int, int, int]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} != expected {expected}")


def _pml_factor(
    shape: tuple[int, int, int],
    pml_cells: int,
    strength: float,
) -> np.ndarray:
    if pml_cells <= 0 or strength <= 0.0:
        return np.ones(shape, dtype=float)

    factor = np.ones(shape, dtype=float)
    for axis, length in enumerate(shape):
        idx = np.arange(length, dtype=float)
        distance = np.minimum(idx, length - 1 - idx)
        depth = np.clip((pml_cells - distance) / max(pml_cells, 1), 0.0, 1.0)
        axis_factor = np.exp(-float(strength) * depth**2)
        reshape = [1, 1, 1]
        reshape[axis] = length
        factor *= axis_factor.reshape(tuple(reshape))
    return factor
