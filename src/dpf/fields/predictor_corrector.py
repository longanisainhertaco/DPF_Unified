"""Candidate current predictor-corrector for the 3-D hybrid PIC-fluid loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid
from dpf.fields.ohm_solver import GeneralizedOhmSolver


@dataclass(frozen=True)
class CurrentPredictorCorrectorTelemetry:
    """Telemetry for a source-derived current predictor-corrector slice."""

    status: str
    source: str
    first_step_initialization: bool
    predictor_delta_linf_A_m2: float
    corrected_max_current_A_m2: float
    corrected_max_residual_A_m2: float
    include_hall: bool
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CurrentPredictorCorrector:
    """Implement current extrapolation and end-step Ohm correction primitives."""

    capability_id = "current_predictor_corrector"

    def __init__(self, grid: Maxwell3DGrid) -> None:
        self.grid = grid
        self.ohm_solver = GeneralizedOhmSolver(grid)

    def predict_end_step_current(
        self,
        midpoint_current_A_m2: np.ndarray,
        previous_current_A_m2: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Return J*_{n+1} = 2 J_{n+1/2} - J_n, or J_{1/2} on first step."""
        J_mid = _as_vector("midpoint_current_A_m2", midpoint_current_A_m2, self.grid.shape)
        if previous_current_A_m2 is None:
            return np.array(J_mid, copy=True), True
        J_previous = _as_vector("previous_current_A_m2", previous_current_A_m2, self.grid.shape)
        return 2.0 * J_mid - J_previous, False

    def correct_end_step_current(
        self,
        *,
        midpoint_current_A_m2: np.ndarray,
        previous_current_A_m2: np.ndarray | None,
        electric_field_next_V_m: np.ndarray,
        magnetic_field_next_T: np.ndarray,
        predicted_ion_current_A_m2: np.ndarray,
        conductivity_S_m: np.ndarray | float,
        electron_density_m3: np.ndarray,
        pressure_term_V_m: np.ndarray | None = None,
        include_hall: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, CurrentPredictorCorrectorTelemetry]:
        """Predict J* and solve the source end-of-step generalized Ohm law."""
        predicted_total, first_step = self.predict_end_step_current(
            midpoint_current_A_m2,
            previous_current_A_m2,
        )
        corrected, ohm_telemetry = self.ohm_solver.solve_current(
            electric_field_V_m=electric_field_next_V_m,
            magnetic_field_T=magnetic_field_next_T,
            curl_B_T_m=np.zeros(self.grid.shape + (3,), dtype=float),
            ion_current_A_m2=predicted_ion_current_A_m2,
            conductivity_S_m=conductivity_S_m,
            electron_density_m3=electron_density_m3,
            dt_s=0.0,
            pressure_term_V_m=pressure_term_V_m,
            include_hall=include_hall,
        )
        predictor_delta = predicted_total - _as_vector(
            "midpoint_current_A_m2",
            midpoint_current_A_m2,
            self.grid.shape,
        )
        telemetry = CurrentPredictorCorrectorTelemetry(
            status="candidate_engineering_predictor_corrector",
            source=HYBRID_PIC_3D_SOURCE,
            first_step_initialization=first_step,
            predictor_delta_linf_A_m2=float(np.max(np.abs(predictor_delta))),
            corrected_max_current_A_m2=ohm_telemetry.max_current_A_m2,
            corrected_max_residual_A_m2=ohm_telemetry.max_algebraic_residual_A_m2,
            include_hall=ohm_telemetry.include_hall,
        )
        return predicted_total, corrected, telemetry


def predictor_corrector_candidate_evidence(
    telemetry: CurrentPredictorCorrectorTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for current predictor-corrector plumbing."""
    return {
        "passed": telemetry.status == "candidate_engineering_predictor_corrector",
        "status": "candidate",
        "capability": CurrentPredictorCorrector.capability_id,
        "source": telemetry.source,
        "implementation": "src/dpf/fields/predictor_corrector.py",
        "evidence_type": "engineering_current_predictor_corrector",
        "first_step_initialization": telemetry.first_step_initialization,
        "corrected_max_residual_A_m2": telemetry.corrected_max_residual_A_m2,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Does not perform the provisional ion push itself.",
            "Does not rebuild ne, Te, and Ji from particle state.",
            "Integrated only as a candidate end-step current solve, not accepted full-loop authority.",
        ],
    }


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
