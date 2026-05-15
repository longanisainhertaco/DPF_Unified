"""Source-scoped external-circuit magnetic boundary for the 3-D hybrid path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi
from typing import Any

import numpy as np

from dpf.fields.maxwell_3d import (
    HYBRID_PIC_3D_SOURCE,
    MU_0,
    Maxwell3DGrid,
    Maxwell3DState,
)


@dataclass(frozen=True)
class CircuitParameters:
    """Lumped RLC parameters from the local hybrid-PIC source."""

    inductance_H: float = 1.1e-7
    voltage_V: float = 1.5e4
    resistance_ohm: float = 1.2e-2
    capacitance_F: float = 2.0e-5

    def __post_init__(self) -> None:
        if self.inductance_H <= 0.0:
            raise ValueError("inductance_H must be positive")
        if self.capacitance_F <= 0.0:
            raise ValueError("capacitance_F must be positive")
        if self.resistance_ohm < 0.0:
            raise ValueError("resistance_ohm must be non-negative")


@dataclass(frozen=True)
class CircuitState:
    """External circuit state used by the explicit source update."""

    current_A: float = 1.773e4
    charge_C: float = 0.218


@dataclass(frozen=True)
class CircuitStepTelemetry:
    """Telemetry for the source Eq. 37-38 explicit circuit update."""

    status: str
    source: str
    source_lines: str
    current_A: float
    charge_C: float
    udpf_V: float
    dI_dt_A_s: float
    next_current_A: float
    next_charge_C: float
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CircuitBoundaryTelemetry:
    """Telemetry for applying the azimuthal magnetic injection boundary."""

    status: str
    source: str
    source_lines: str
    current_A: float
    z_index: int
    radius_floor_m: float
    radius_max_m: float | None
    blend: float
    bx_faces_updated: int
    by_faces_updated: int
    btheta_abs_min_T: float
    btheta_abs_max_T: float
    can_support_first_principles_acceptance: bool = False

    @property
    def faces_updated(self) -> int:
        return self.bx_faces_updated + self.by_faces_updated

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["faces_updated"] = self.faces_updated
        return payload


class CircuitMagneticBoundaryDrive:
    """RLC current update and azimuthal magnetic boundary condition.

    The local source gives a cylindrical boundary formula, B_theta =
    mu0 I / (2 pi r), and an explicit RLC update for the generator current.
    This class applies those equations to the Cartesian 3-D engineering grid
    as a candidate boundary component; it is not a validated DPF geometry map.
    """

    capability_id = "rlc_circuit_magnetic_boundary_drive"

    def __init__(
        self,
        grid: Maxwell3DGrid,
        *,
        parameters: CircuitParameters | None = None,
        axis_origin_m: tuple[float, float] = (0.0, 0.0),
        radius_floor_m: float | None = None,
        radius_max_m: float | None = None,
    ) -> None:
        self.grid = grid
        self.parameters = parameters or CircuitParameters()
        self.axis_origin_m = (float(axis_origin_m[0]), float(axis_origin_m[1]))
        self.radius_floor_m = (
            0.5 * min(grid.dx, grid.dy)
            if radius_floor_m is None
            else float(radius_floor_m)
        )
        self.radius_max_m = None if radius_max_m is None else float(radius_max_m)
        if self.radius_floor_m <= 0.0:
            raise ValueError("radius_floor_m must be positive")
        if self.radius_max_m is not None and self.radius_max_m <= 0.0:
            raise ValueError("radius_max_m must be positive when supplied")

    @staticmethod
    def azimuthal_B_T(current_A: float, radius_m: float) -> float:
        """Return B_theta from the local source Eq. 34."""
        if radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        return MU_0 * float(current_A) / (2.0 * pi * float(radius_m))

    def step_circuit(
        self,
        state: CircuitState,
        *,
        dt_s: float,
        udpf_V: float = 0.0,
    ) -> tuple[CircuitState, CircuitStepTelemetry]:
        """Advance current and charge with the source Eq. 37-38 scheme."""
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        params = self.parameters
        dI_dt = (
            params.voltage_V
            - params.resistance_ohm * state.current_A
            - state.charge_C / params.capacitance_F
            - float(udpf_V)
        ) / params.inductance_H
        next_state = CircuitState(
            current_A=float(state.current_A + dt_s * dI_dt),
            charge_C=float(state.charge_C + dt_s * state.current_A),
        )
        telemetry = CircuitStepTelemetry(
            status="candidate_engineering_rlc_circuit_step",
            source=HYBRID_PIC_3D_SOURCE,
            source_lines="740-792",
            current_A=float(state.current_A),
            charge_C=float(state.charge_C),
            udpf_V=float(udpf_V),
            dI_dt_A_s=float(dI_dt),
            next_current_A=next_state.current_A,
            next_charge_C=next_state.charge_C,
        )
        return next_state, telemetry

    def cell_centered_azimuthal_B_T(self, current_A: float) -> np.ndarray:
        """Return a Cartesian cell-centered projection of source B_theta."""
        x = _cell_centers(self.grid.nx, self.grid.dx) - self.axis_origin_m[0]
        y = _cell_centers(self.grid.ny, self.grid.dy) - self.axis_origin_m[1]
        X, Y = np.meshgrid(x, y, indexing="ij")
        Bx, By, _ = self._azimuthal_components(current_A, X, Y)
        field = np.zeros(self.grid.shape + (3,), dtype=float)
        field[..., 0] = Bx[:, :, None]
        field[..., 1] = By[:, :, None]
        return field

    def apply_injection_port_boundary(
        self,
        state: Maxwell3DState,
        *,
        current_A: float,
        z_index: int = 0,
        blend: float = 1.0,
    ) -> tuple[Maxwell3DState, CircuitBoundaryTelemetry]:
        """Apply the azimuthal B boundary to one axial magnetic-field plane."""
        if int(z_index) != z_index or z_index < 0 or z_index >= self.grid.nz:
            raise ValueError("z_index must address a Bx/By axial plane")
        if blend < 0.0 or blend > 1.0:
            raise ValueError("blend must satisfy 0 <= blend <= 1")

        Bx_slice, by_slice, bx_mask, by_mask, btheta_values = (
            self._boundary_face_components(current_A)
        )
        if not np.any(bx_mask) and not np.any(by_mask):
            raise ValueError("radius_max_m excludes every boundary face")

        next_state = state.copy()
        k = int(z_index)
        next_state.B.Bx_face[:, :, k][bx_mask] = (
            (1.0 - blend) * next_state.B.Bx_face[:, :, k][bx_mask]
            + blend * Bx_slice[bx_mask]
        )
        next_state.B.By_face[:, :, k][by_mask] = (
            (1.0 - blend) * next_state.B.By_face[:, :, k][by_mask]
            + blend * by_slice[by_mask]
        )
        abs_btheta = np.abs(btheta_values)
        telemetry = CircuitBoundaryTelemetry(
            status="candidate_engineering_magnetic_injection_boundary",
            source=HYBRID_PIC_3D_SOURCE,
            source_lines="740-792",
            current_A=float(current_A),
            z_index=k,
            radius_floor_m=float(self.radius_floor_m),
            radius_max_m=self.radius_max_m,
            blend=float(blend),
            bx_faces_updated=int(np.count_nonzero(bx_mask)),
            by_faces_updated=int(np.count_nonzero(by_mask)),
            btheta_abs_min_T=float(np.min(abs_btheta)),
            btheta_abs_max_T=float(np.max(abs_btheta)),
        )
        return next_state, telemetry

    def _boundary_face_components(
        self,
        current_A: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid = self.grid
        x_face = _face_coordinates(grid.nx, grid.dx) - self.axis_origin_m[0]
        y_center = _cell_centers(grid.ny, grid.dy) - self.axis_origin_m[1]
        X_bx, Y_bx = np.meshgrid(x_face, y_center, indexing="ij")
        Bx, _, radius_bx = self._azimuthal_components(current_A, X_bx, Y_bx)
        bx_mask = self._radius_mask(radius_bx)

        x_center = _cell_centers(grid.nx, grid.dx) - self.axis_origin_m[0]
        y_face = _face_coordinates(grid.ny, grid.dy) - self.axis_origin_m[1]
        X_by, Y_by = np.meshgrid(x_center, y_face, indexing="ij")
        _, By, radius_by = self._azimuthal_components(current_A, X_by, Y_by)
        by_mask = self._radius_mask(radius_by)

        btheta_values = np.concatenate((
            np.abs(self._btheta_on_radius(current_A, radius_bx[bx_mask])),
            np.abs(self._btheta_on_radius(current_A, radius_by[by_mask])),
        ))
        return Bx, By, bx_mask, by_mask, btheta_values

    def _azimuthal_components(
        self,
        current_A: float,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius = np.sqrt(X**2 + Y**2)
        radius_eff = np.maximum(radius, self.radius_floor_m)
        btheta = self._btheta_on_radius(current_A, radius_eff)
        Bx = -btheta * np.divide(Y, radius_eff)
        By = btheta * np.divide(X, radius_eff)
        mask = self._radius_mask(radius)
        return np.where(mask, Bx, 0.0), np.where(mask, By, 0.0), radius

    def _btheta_on_radius(self, current_A: float, radius_m: np.ndarray) -> np.ndarray:
        return MU_0 * float(current_A) / (2.0 * pi * np.maximum(radius_m, self.radius_floor_m))

    def _radius_mask(self, radius_m: np.ndarray) -> np.ndarray:
        if self.radius_max_m is None:
            return np.ones(radius_m.shape, dtype=bool)
        return radius_m <= self.radius_max_m


def circuit_boundary_candidate_evidence(
    telemetry: CircuitBoundaryTelemetry | CircuitStepTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for the source circuit/boundary slice."""
    return {
        "passed": telemetry.status.startswith("candidate_engineering_"),
        "status": "candidate",
        "capability": CircuitMagneticBoundaryDrive.capability_id,
        "source": telemetry.source,
        "source_lines": telemetry.source_lines,
        "implementation": "src/dpf/fields/circuit_boundary.py",
        "evidence_type": "engineering_circuit_magnetic_boundary_component",
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "The B_theta formula and RLC update are source-scoped, but this is a Cartesian engineering projection.",
            "UDPF is an input placeholder; the full magnetic-flux integral and differentiation closure is not implemented.",
            "No accepted 3-D DPF injection-port geometry or same-scope validation packet is attached.",
        ],
    }


def _cell_centers(n: int, spacing: float) -> np.ndarray:
    return (np.arange(n, dtype=float) + 0.5) * spacing - 0.5 * n * spacing


def _face_coordinates(n: int, spacing: float) -> np.ndarray:
    return np.arange(n + 1, dtype=float) * spacing - 0.5 * n * spacing
