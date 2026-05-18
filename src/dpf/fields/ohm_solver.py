"""Candidate 3-D generalized Ohm-Ampere current solver.

The local arXiv:2604.09032v1 KnowledgeReference source derives an algebraic
midpoint solve for current density from generalized Ohm's law coupled to
Ampere's law.  This module implements that cell-centered algebraic closure as
engineering component evidence for the 3-D hybrid PIC-fluid path, with a
backward-Euler option for the stiff resistive limit of the same equations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.constants import e
from dpf.fields.maxwell_3d import (
    EPSILON_0,
    HYBRID_PIC_3D_SOURCE,
    SPEED_OF_LIGHT,
    Maxwell3DGrid,
)


@dataclass(frozen=True)
class GeneralizedOhmTelemetry:
    """Telemetry for a generalized Ohm current solve."""

    status: str
    source: str
    include_hall: bool
    include_pressure: bool
    pressure_active_fraction: float
    min_electron_density_m3: float
    max_conductivity_S_m: float
    max_current_A_m2: float
    max_algebraic_residual_A_m2: float
    ohm_time_centering_theta: float
    electric_update_scheme: str
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneralizedOhmSolver:
    """Solve the source-derived midpoint Ohm-Ampere current closure."""

    capability_id = "electron_fluid_generalized_ohm_solver"

    def __init__(self, grid: Maxwell3DGrid) -> None:
        self.grid = grid

    def pressure_gradient_term(
        self,
        electron_pressure_Pa: np.ndarray,
        electron_density_m3: np.ndarray,
        *,
        density_threshold_m3: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return P = grad(pe)/(e ne), with optional low-density suppression."""
        pe = np.asarray(electron_pressure_Pa, dtype=float)
        ne = np.asarray(electron_density_m3, dtype=float)
        _require_scalar_shape("electron_pressure_Pa", pe, self.grid.shape)
        _require_scalar_shape("electron_density_m3", ne, self.grid.shape)
        _require_positive_density(ne)
        if density_threshold_m3 < 0.0:
            raise ValueError("density_threshold_m3 must be non-negative")

        grad = np.stack(
            (
                np.gradient(pe, self.grid.dx, axis=0, edge_order=1),
                np.gradient(pe, self.grid.dy, axis=1, edge_order=1),
                np.gradient(pe, self.grid.dz, axis=2, edge_order=1),
            ),
            axis=-1,
        )
        active = ne >= density_threshold_m3
        pressure_term = np.zeros(self.grid.shape + (3,), dtype=float)
        pressure_term[active] = grad[active] / (
            e * ne[active, np.newaxis]
        )
        telemetry = {
            "status": "candidate_pressure_gradient_term",
            "source": HYBRID_PIC_3D_SOURCE,
            "source_lines": "1107-1185",
            "density_threshold_m3": float(density_threshold_m3),
            "active_fraction": float(np.count_nonzero(active) / active.size),
            "can_support_first_principles_acceptance": False,
        }
        return pressure_term, telemetry

    def solve_current(
        self,
        *,
        electric_field_V_m: np.ndarray,
        magnetic_field_T: np.ndarray,
        curl_B_T_m: np.ndarray,
        ion_current_A_m2: np.ndarray,
        conductivity_S_m: np.ndarray | float,
        electron_density_m3: np.ndarray,
        dt_s: float,
        pressure_term_V_m: np.ndarray | None = None,
        include_hall: bool = True,
        ohm_time_centering_theta: float = 0.5,
    ) -> tuple[np.ndarray, GeneralizedOhmTelemetry]:
        """Solve for current density using the source algebraic form.

        ``ohm_time_centering_theta=0.5`` gives the midpoint/Crank-Nicolson
        current used by the original candidate path. ``theta=1`` gives a
        backward-Euler resistive Ampere-Ohm solve for stiff source-backed
        conductivity, avoiding an artificial explicit Ohmic conductivity cap.
        """
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        theta = float(ohm_time_centering_theta)
        if theta < 0.5 or theta > 1.0:
            raise ValueError("ohm_time_centering_theta must be in [0.5, 1.0]")
        E = _as_vector("electric_field_V_m", electric_field_V_m, self.grid.shape)
        B = _as_vector("magnetic_field_T", magnetic_field_T, self.grid.shape)
        curl_B = _as_vector("curl_B_T_m", curl_B_T_m, self.grid.shape)
        Ji = _as_vector("ion_current_A_m2", ion_current_A_m2, self.grid.shape)
        ne = np.asarray(electron_density_m3, dtype=float)
        _require_scalar_shape("electron_density_m3", ne, self.grid.shape)
        _require_positive_density(ne)
        sigma = np.asarray(conductivity_S_m, dtype=float)
        if sigma.shape == ():
            sigma = np.full(self.grid.shape, float(sigma), dtype=float)
        _require_scalar_shape("conductivity_S_m", sigma, self.grid.shape)
        if np.any(sigma < 0.0):
            raise ValueError("conductivity_S_m must be non-negative")

        if pressure_term_V_m is None:
            pressure = np.zeros(self.grid.shape + (3,), dtype=float)
            include_pressure = False
            pressure_active_fraction = 0.0
        else:
            pressure = _as_vector("pressure_term_V_m", pressure_term_V_m, self.grid.shape)
            include_pressure = True
            pressure_active_fraction = float(
                np.count_nonzero(np.linalg.norm(pressure, axis=-1) > 0.0)
                / np.prod(self.grid.shape)
            )

        D = 1.0 + theta * sigma * dt_s / EPSILON_0
        alpha = sigma / (e * ne)
        known_velocity_like = (
            theta * SPEED_OF_LIGHT**2 * dt_s * curl_B
            + Ji / (e * ne[..., np.newaxis])
        )
        A = sigma[..., np.newaxis] * (E + np.cross(known_velocity_like, B) + pressure)

        if include_hall:
            J = _solve_hall_system(A, B, D, alpha)
        else:
            J = A / D[..., np.newaxis]

        residual = _ohm_algebraic_residual(J, A, B, D, alpha, include_hall=include_hall)
        telemetry = GeneralizedOhmTelemetry(
            status="candidate_engineering_closure",
            source=HYBRID_PIC_3D_SOURCE,
            include_hall=include_hall,
            include_pressure=include_pressure,
            pressure_active_fraction=pressure_active_fraction,
            min_electron_density_m3=float(np.min(ne)),
            max_conductivity_S_m=float(np.max(sigma)),
            max_current_A_m2=float(np.max(np.linalg.norm(J, axis=-1))),
            max_algebraic_residual_A_m2=float(np.max(np.linalg.norm(residual, axis=-1))),
            ohm_time_centering_theta=theta,
            electric_update_scheme=(
                "backward_euler_resistive_ampere_ohm"
                if theta == 1.0
                else "midpoint_crank_nicolson_resistive_ampere_ohm"
            ),
        )
        return J, telemetry


def generalized_ohm_candidate_evidence(
    telemetry: GeneralizedOhmTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for the generalized Ohm closure."""
    return {
        "passed": telemetry.status == "candidate_engineering_closure",
        "status": "candidate",
        "capability": GeneralizedOhmSolver.capability_id,
        "source": telemetry.source,
        "implementation": "src/dpf/fields/ohm_solver.py",
        "evidence_type": "engineering_ohm_ampere_component",
        "include_hall": telemetry.include_hall,
        "include_pressure": telemetry.include_pressure,
        "ohm_time_centering_theta": telemetry.ohm_time_centering_theta,
        "electric_update_scheme": telemetry.electric_update_scheme,
        "max_algebraic_residual_A_m2": telemetry.max_algebraic_residual_A_m2,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Cell-centered closure is integrated into the candidate Yee/PIC loop through HybridPIC3DFieldStepper, but it remains engineering evidence.",
            "Pressure and Hall terms remain qualitative without a separate electron-energy closure.",
            "Same-scope 3-D validation is not supplied.",
        ],
    }


def _solve_hall_system(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    shape = A.shape[:-1]
    matrix = np.zeros(shape + (3, 3), dtype=float)
    matrix[..., 0, 0] = D
    matrix[..., 1, 1] = D
    matrix[..., 2, 2] = D
    matrix[..., 0, 1] = alpha * B[..., 2]
    matrix[..., 0, 2] = -alpha * B[..., 1]
    matrix[..., 1, 0] = -alpha * B[..., 2]
    matrix[..., 1, 2] = alpha * B[..., 0]
    matrix[..., 2, 0] = alpha * B[..., 1]
    matrix[..., 2, 1] = -alpha * B[..., 0]
    solved = np.linalg.solve(
        matrix.reshape((-1, 3, 3)),
        A.reshape((-1, 3, 1)),
    )
    return solved.reshape(shape + (3,))


def _ohm_algebraic_residual(
    J: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    alpha: np.ndarray,
    *,
    include_hall: bool,
) -> np.ndarray:
    left = D[..., np.newaxis] * J
    if include_hall:
        left = left + alpha[..., np.newaxis] * np.cross(J, B)
    return left - A


def _as_vector(
    name: str,
    value: np.ndarray,
    cell_shape: tuple[int, int, int],
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    expected = cell_shape + (3,)
    if arr.shape != expected:
        raise ValueError(f"{name} shape {arr.shape} != expected {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _require_scalar_shape(
    name: str,
    value: np.ndarray,
    expected: tuple[int, int, int],
) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} shape {value.shape} != expected {expected}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive_density(ne: np.ndarray) -> None:
    if np.any(ne <= 0.0):
        raise ValueError("electron_density_m3 must be strictly positive")
