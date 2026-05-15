"""Candidate PIC-current coupling utilities for the 3-D Maxwell field core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.fields.maxwell_3d import (
    HYBRID_PIC_3D_SOURCE,
    Maxwell3DGrid,
    YeeElectricField,
)


@dataclass(frozen=True)
class PICCurrentSourceTelemetry:
    """Telemetry for a PIC-to-Maxwell current source conversion."""

    status: str
    source: str
    deposition_method: str
    finite: bool
    input_shape: tuple[int, int, int]
    edge_shapes: dict[str, tuple[int, int, int]]
    continuity_status: str
    continuity_linf_A_per_m3: float | None = None
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PICCurrentSourcePort:
    """Convert PIC deposition output into Yee edge current density."""

    def __init__(self, grid: Maxwell3DGrid) -> None:
        self.grid = grid

    def from_cell_centered_current(
        self,
        Jx: np.ndarray,
        Jy: np.ndarray,
        Jz: np.ndarray,
        *,
        deposition_method: str,
        rho_previous: np.ndarray | None = None,
        rho_current: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[YeeElectricField, PICCurrentSourceTelemetry]:
        """Map cell-centered current density arrays to Yee edge arrays."""
        Jx = np.asarray(Jx, dtype=float)
        Jy = np.asarray(Jy, dtype=float)
        Jz = np.asarray(Jz, dtype=float)
        _require_cell_shape("Jx", Jx, self.grid.shape)
        _require_cell_shape("Jy", Jy, self.grid.shape)
        _require_cell_shape("Jz", Jz, self.grid.shape)

        edge_current = YeeElectricField(
            Ex_edge=_average_cells_to_x_edges(Jx),
            Ey_edge=_average_cells_to_y_edges(Jy),
            Ez_edge=_average_cells_to_z_edges(Jz),
        )
        finite = bool(
            np.all(np.isfinite(edge_current.Ex_edge))
            and np.all(np.isfinite(edge_current.Ey_edge))
            and np.all(np.isfinite(edge_current.Ez_edge))
        )

        continuity_status = "unverified_no_prior_charge_state"
        residual = None
        if rho_previous is not None or rho_current is not None or dt is not None:
            if rho_previous is None or rho_current is None or dt is None or dt <= 0.0:
                continuity_status = "blocked_incomplete_continuity_inputs"
            else:
                rho_previous = np.asarray(rho_previous, dtype=float)
                rho_current = np.asarray(rho_current, dtype=float)
                _require_cell_shape("rho_previous", rho_previous, self.grid.shape)
                _require_cell_shape("rho_current", rho_current, self.grid.shape)
                residual_array = (
                    (rho_current - rho_previous) / dt
                    + _cell_centered_divergence(Jx, Jy, Jz, self.grid)
                )
                residual = float(np.max(np.abs(residual_array)))
                continuity_status = "measured_not_accepted"

        telemetry = PICCurrentSourceTelemetry(
            status="candidate_engineering_coupling" if finite else "blocked_nonfinite",
            source=HYBRID_PIC_3D_SOURCE,
            deposition_method=deposition_method,
            finite=finite,
            input_shape=self.grid.shape,
            edge_shapes={
                "Jx_edge": tuple(edge_current.Ex_edge.shape),
                "Jy_edge": tuple(edge_current.Ey_edge.shape),
                "Jz_edge": tuple(edge_current.Ez_edge.shape),
            },
            continuity_status=continuity_status,
            continuity_linf_A_per_m3=residual,
        )
        return edge_current, telemetry


def pic_current_port_candidate_evidence(
    telemetry: PICCurrentSourceTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for PIC current plumbing."""
    return {
        "passed": telemetry.finite,
        "status": "candidate",
        "capability": "kinetic_ion_pic_push_deposition",
        "source": telemetry.source,
        "implementation": "src/dpf/fields/pic_coupling.py",
        "evidence_type": "engineering_current_source_port",
        "deposition_method": telemetry.deposition_method,
        "continuity_status": telemetry.continuity_status,
        "continuity_linf_A_per_m3": telemetry.continuity_linf_A_per_m3,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Cell-to-edge averaging is coupling plumbing, not charge-conserving acceptance.",
            "Electron-fluid generalized Ohm closure is not supplied.",
            "Same-scope 3-D DPF validation is not supplied.",
        ],
    }


def _require_cell_shape(
    name: str,
    value: np.ndarray,
    expected: tuple[int, int, int],
) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} != expected {expected}")


def _average_cells_to_x_edges(Jx: np.ndarray) -> np.ndarray:
    nx, ny, nz = Jx.shape
    edge = np.zeros((nx, ny + 1, nz + 1), dtype=float)
    count = np.zeros_like(edge)
    edge[:, :-1, :-1] += Jx
    count[:, :-1, :-1] += 1.0
    edge[:, 1:, :-1] += Jx
    count[:, 1:, :-1] += 1.0
    edge[:, :-1, 1:] += Jx
    count[:, :-1, 1:] += 1.0
    edge[:, 1:, 1:] += Jx
    count[:, 1:, 1:] += 1.0
    return edge / count


def _average_cells_to_y_edges(Jy: np.ndarray) -> np.ndarray:
    nx, ny, nz = Jy.shape
    edge = np.zeros((nx + 1, ny, nz + 1), dtype=float)
    count = np.zeros_like(edge)
    edge[:-1, :, :-1] += Jy
    count[:-1, :, :-1] += 1.0
    edge[1:, :, :-1] += Jy
    count[1:, :, :-1] += 1.0
    edge[:-1, :, 1:] += Jy
    count[:-1, :, 1:] += 1.0
    edge[1:, :, 1:] += Jy
    count[1:, :, 1:] += 1.0
    return edge / count


def _average_cells_to_z_edges(Jz: np.ndarray) -> np.ndarray:
    nx, ny, nz = Jz.shape
    edge = np.zeros((nx + 1, ny + 1, nz), dtype=float)
    count = np.zeros_like(edge)
    edge[:-1, :-1, :] += Jz
    count[:-1, :-1, :] += 1.0
    edge[1:, :-1, :] += Jz
    count[1:, :-1, :] += 1.0
    edge[:-1, 1:, :] += Jz
    count[:-1, 1:, :] += 1.0
    edge[1:, 1:, :] += Jz
    count[1:, 1:, :] += 1.0
    return edge / count


def _cell_centered_divergence(
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    grid: Maxwell3DGrid,
) -> np.ndarray:
    dJx_dx = np.gradient(Jx, grid.dx, axis=0, edge_order=1)
    dJy_dy = np.gradient(Jy, grid.dy, axis=1, edge_order=1)
    dJz_dz = np.gradient(Jz, grid.dz, axis=2, edge_order=1)
    return dJx_dx + dJy_dy + dJz_dz
