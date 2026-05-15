"""Candidate Marder/Gauss-law electric-field correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.fields.maxwell_3d import EPSILON_0, HYBRID_PIC_3D_SOURCE, Maxwell3DGrid


@dataclass(frozen=True)
class MarderCorrectionTelemetry:
    """Telemetry for one Marder correction application."""

    status: str
    source: str
    marder_factor_m2: float
    residual_before_linf: float
    residual_after_linf: float
    residual_reduction_fraction: float
    electric_field_linf_V_m: float
    correction_linf_V_m: float
    relative_correction_linf: float
    nondominance_threshold: float | None
    nondominance_status: str
    charge_density_mode: str
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarderCorrection:
    """Apply E <- E + d grad(div(E) - rho/eps0) on cell-centered fields."""

    capability_id = "gauss_law_or_marder_control"

    def __init__(self, grid: Maxwell3DGrid) -> None:
        self.grid = grid

    def gauss_residual(
        self,
        electric_field_V_m: np.ndarray,
        charge_density_C_m3: np.ndarray | None = None,
    ) -> np.ndarray:
        E = _as_vector("electric_field_V_m", electric_field_V_m, self.grid.shape)
        if charge_density_C_m3 is None:
            rho = np.zeros(self.grid.shape, dtype=float)
        else:
            rho = np.asarray(charge_density_C_m3, dtype=float)
            _require_scalar_shape("charge_density_C_m3", rho, self.grid.shape)
        return _divergence(E, self.grid) - rho / EPSILON_0

    def apply(
        self,
        electric_field_V_m: np.ndarray,
        *,
        charge_density_C_m3: np.ndarray | None = None,
        marder_factor_m2: float,
        nondominance_threshold: float | None = None,
    ) -> tuple[np.ndarray, MarderCorrectionTelemetry]:
        if marder_factor_m2 < 0.0:
            raise ValueError("marder_factor_m2 must be non-negative")
        if nondominance_threshold is not None and nondominance_threshold < 0.0:
            raise ValueError("nondominance_threshold must be non-negative")
        E = _as_vector("electric_field_V_m", electric_field_V_m, self.grid.shape)
        residual_before = self.gauss_residual(E, charge_density_C_m3)
        correction = marder_factor_m2 * _gradient(residual_before, self.grid)
        corrected = E + correction
        residual_after = self.gauss_residual(corrected, charge_density_C_m3)
        before_linf = float(np.max(np.abs(residual_before)))
        after_linf = float(np.max(np.abs(residual_after)))
        if before_linf > 0.0:
            reduction = (before_linf - after_linf) / before_linf
        else:
            reduction = 0.0
        field_linf = float(np.max(np.linalg.norm(E, axis=-1)))
        correction_linf = float(np.max(np.linalg.norm(correction, axis=-1)))
        relative_correction = (
            correction_linf / field_linf if field_linf > 0.0 else 0.0
        )
        if nondominance_threshold is None:
            nondominance_status = "not_evaluated"
        elif relative_correction <= nondominance_threshold:
            nondominance_status = "candidate_within_bound"
        else:
            nondominance_status = "candidate_dominant_correction"
        telemetry = MarderCorrectionTelemetry(
            status="candidate_engineering_marder_correction",
            source=HYBRID_PIC_3D_SOURCE,
            marder_factor_m2=float(marder_factor_m2),
            residual_before_linf=before_linf,
            residual_after_linf=after_linf,
            residual_reduction_fraction=float(reduction),
            electric_field_linf_V_m=field_linf,
            correction_linf_V_m=correction_linf,
            relative_correction_linf=float(relative_correction),
            nondominance_threshold=(
                None
                if nondominance_threshold is None
                else float(nondominance_threshold)
            ),
            nondominance_status=nondominance_status,
            charge_density_mode=(
                "quasi_neutral_zero_charge"
                if charge_density_C_m3 is None
                else "explicit_charge_density"
            ),
        )
        return corrected, telemetry


def marder_candidate_evidence(
    telemetry: MarderCorrectionTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for Marder/Gauss-law control."""
    return {
        "passed": telemetry.status == "candidate_engineering_marder_correction",
        "status": "candidate",
        "capability": MarderCorrection.capability_id,
        "source": telemetry.source,
        "implementation": "src/dpf/fields/marder.py",
        "evidence_type": "engineering_gauss_law_control_component",
        "residual_before_linf": telemetry.residual_before_linf,
        "residual_after_linf": telemetry.residual_after_linf,
        "residual_reduction_fraction": telemetry.residual_reduction_fraction,
        "relative_correction_linf": telemetry.relative_correction_linf,
        "nondominance_threshold": telemetry.nondominance_threshold,
        "nondominance_status": telemetry.nondominance_status,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Cell-centered component is only coupled to Yee edges in the candidate stepper.",
            "Nondominance telemetry is component/loop engineering evidence, not accepted zmax/sheath sensitivity.",
            "Same-scope 3-D validation is not supplied.",
        ],
    }


def _divergence(E: np.ndarray, grid: Maxwell3DGrid) -> np.ndarray:
    return (
        np.gradient(E[..., 0], grid.dx, axis=0, edge_order=1)
        + np.gradient(E[..., 1], grid.dy, axis=1, edge_order=1)
        + np.gradient(E[..., 2], grid.dz, axis=2, edge_order=1)
    )


def _gradient(scalar: np.ndarray, grid: Maxwell3DGrid) -> np.ndarray:
    return np.stack(
        (
            np.gradient(scalar, grid.dx, axis=0, edge_order=1),
            np.gradient(scalar, grid.dy, axis=1, edge_order=1),
            np.gradient(scalar, grid.dz, axis=2, edge_order=1),
        ),
        axis=-1,
    )


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
