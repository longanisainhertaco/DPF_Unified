"""Candidate integrated 3-D hybrid field-current stepper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.fields.conductivity import PlasmaVacuumConductivityBlend
from dpf.fields.marder import MarderCorrection
from dpf.fields.maxwell_3d import (
    HYBRID_PIC_3D_SOURCE,
    Maxwell3DBoundaries,
    Maxwell3DFieldCore,
    Maxwell3DGrid,
    Maxwell3DState,
)
from dpf.fields.ohm_solver import GeneralizedOhmSolver
from dpf.fields.pic_coupling import PICCurrentSourcePort
from dpf.fields.predictor_corrector import CurrentPredictorCorrector
from dpf.fluid.constrained_transport import face_to_cell_centered


@dataclass(frozen=True)
class HybridPIC3DStepTelemetry:
    """Telemetry for one candidate 3-D hybrid field-current step."""

    status: str
    source: str
    conductivity: dict[str, Any]
    ohm_solver: dict[str, Any]
    predictor_corrector: dict[str, Any] | None
    marder: dict[str, Any] | None
    current_port: dict[str, Any]
    field_work: dict[str, Any]
    diagnostics_before: dict[str, Any]
    diagnostics_after: dict[str, Any]
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridPIC3DStepResult:
    """Result from one candidate field-current step."""

    state: Maxwell3DState
    total_current_A_m2: np.ndarray
    end_step_current_A_m2: np.ndarray
    conductivity_S_m: np.ndarray
    telemetry: HybridPIC3DStepTelemetry


class HybridPIC3DFieldStepper:
    """Couple source conductivity, generalized Ohm current, and Maxwell fields."""

    def __init__(
        self,
        grid: Maxwell3DGrid,
        *,
        boundaries: Maxwell3DBoundaries | None = None,
    ) -> None:
        self.grid = grid
        self.maxwell = Maxwell3DFieldCore(grid, boundaries=boundaries)
        self.conductivity = PlasmaVacuumConductivityBlend(grid)
        self.ohm_solver = GeneralizedOhmSolver(grid)
        self.predictor_corrector = CurrentPredictorCorrector(grid)
        self.marder = MarderCorrection(grid)
        self.current_port = PICCurrentSourcePort(grid)
        self.previous_total_current_A_m2: np.ndarray | None = None

    def step(
        self,
        state: Maxwell3DState,
        *,
        dt_s: float,
        ion_current_A_m2: np.ndarray,
        electron_density_m3: np.ndarray,
        sigma0_S_m: np.ndarray | float,
        background_density_m3: float,
        ohmic_cfl_safety: float,
        pressure_term_V_m: np.ndarray | None = None,
        include_hall: bool = True,
        use_predictor_corrector: bool = False,
        marder_factor_m2: float = 0.0,
        marder_nondominance_threshold: float | None = None,
        charge_density_C_m3: np.ndarray | None = None,
        apply_density_conductivity_blend: bool = True,
        apply_ohmic_cfl_limit: bool = True,
    ) -> HybridPIC3DStepResult:
        """Advance one candidate field-current step without particle pushing."""
        self.maxwell.validate_state(state)
        E_cell = np.stack(self.maxwell.edge_E_to_cell_centered(state.E), axis=-1)
        B_cell = np.stack(face_to_cell_centered(state.B), axis=-1)
        curl_B_edges = self.maxwell.curl_B_to_edges(state.B)
        curl_B_cell = np.stack(
            self.maxwell.edge_E_to_cell_centered(curl_B_edges),
            axis=-1,
        )
        sigma_eff, conductivity_telemetry = self.conductivity.effective_conductivity(
            sigma0_S_m=sigma0_S_m,
            electron_density_m3=electron_density_m3,
            background_density_m3=background_density_m3,
            dt_s=dt_s,
            ohmic_cfl_safety=ohmic_cfl_safety,
            apply_density_blend=apply_density_conductivity_blend,
            apply_ohmic_cfl_limit=apply_ohmic_cfl_limit,
        )
        total_current, ohm_telemetry = self.ohm_solver.solve_current(
            electric_field_V_m=E_cell,
            magnetic_field_T=B_cell,
            curl_B_T_m=curl_B_cell,
            ion_current_A_m2=ion_current_A_m2,
            conductivity_S_m=sigma_eff,
            electron_density_m3=electron_density_m3,
            dt_s=dt_s,
            pressure_term_V_m=pressure_term_V_m,
            include_hall=include_hall,
        )
        edge_current, current_telemetry = self.current_port.from_cell_centered_current(
            total_current[..., 0],
            total_current[..., 1],
            total_current[..., 2],
            deposition_method="generalized_ohm_total_current",
        )
        field_work = _field_work_telemetry(
            total_current_A_m2=total_current,
            electric_field_V_m=E_cell,
            cell_volume_m3=self.grid.cell_volume,
        )
        before = self.maxwell.diagnostics(state)
        next_state = self.maxwell.step(
            state,
            dt_s,
            current_density=edge_current,
        )
        marder_telemetry: dict[str, Any] | None = None
        if marder_factor_m2 > 0.0:
            E_next_cell = np.stack(
                self.maxwell.edge_E_to_cell_centered(next_state.E),
                axis=-1,
            )
            corrected_E_cell, correction_telemetry = self.marder.apply(
                E_next_cell,
                charge_density_C_m3=charge_density_C_m3,
                marder_factor_m2=marder_factor_m2,
                nondominance_threshold=marder_nondominance_threshold,
            )
            delta_E_cell = corrected_E_cell - E_next_cell
            delta_E_edges, _ = self.current_port.from_cell_centered_current(
                delta_E_cell[..., 0],
                delta_E_cell[..., 1],
                delta_E_cell[..., 2],
                deposition_method="marder_cell_centered_delta",
            )
            corrected_E = next_state.E.copy()
            corrected_E.Ex_edge += delta_E_edges.Ex_edge
            corrected_E.Ey_edge += delta_E_edges.Ey_edge
            corrected_E.Ez_edge += delta_E_edges.Ez_edge
            next_state = self.maxwell.apply_boundary_conditions(
                Maxwell3DState(E=corrected_E, B=next_state.B)
            )
            marder_telemetry = correction_telemetry.to_dict()
        end_step_current = total_current
        predictor_corrector_telemetry: dict[str, Any] | None = None
        if use_predictor_corrector:
            E_next = np.stack(
                self.maxwell.edge_E_to_cell_centered(next_state.E),
                axis=-1,
            )
            B_next = np.stack(face_to_cell_centered(next_state.B), axis=-1)
            _, end_step_current, pc_telemetry = (
                self.predictor_corrector.correct_end_step_current(
                    midpoint_current_A_m2=total_current,
                    previous_current_A_m2=self.previous_total_current_A_m2,
                    electric_field_next_V_m=E_next,
                    magnetic_field_next_T=B_next,
                    predicted_ion_current_A_m2=ion_current_A_m2,
                    conductivity_S_m=sigma_eff,
                    electron_density_m3=electron_density_m3,
                    pressure_term_V_m=pressure_term_V_m,
                    include_hall=include_hall,
                )
            )
            predictor_corrector_telemetry = pc_telemetry.to_dict()
            self.previous_total_current_A_m2 = np.array(end_step_current, copy=True)
        after = self.maxwell.diagnostics(next_state)
        telemetry = HybridPIC3DStepTelemetry(
            status="candidate_engineering_field_current_step",
            source=HYBRID_PIC_3D_SOURCE,
            conductivity=conductivity_telemetry.to_dict(),
            ohm_solver=ohm_telemetry.to_dict(),
            predictor_corrector=predictor_corrector_telemetry,
            marder=marder_telemetry,
            current_port=current_telemetry.to_dict(),
            field_work=field_work,
            diagnostics_before=before.to_dict(),
            diagnostics_after=after.to_dict(),
        )
        return HybridPIC3DStepResult(
            state=next_state,
            total_current_A_m2=total_current,
            end_step_current_A_m2=end_step_current,
            conductivity_S_m=sigma_eff,
            telemetry=telemetry,
        )


def hybrid_stepper_candidate_evidence(
    telemetry: HybridPIC3DStepTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for the integrated field-current step."""
    return {
        "passed": telemetry.status == "candidate_engineering_field_current_step",
        "status": "candidate",
        "capability": "hybrid_field_current_step",
        "source": telemetry.source,
        "implementation": "src/dpf/fields/hybrid_stepper.py",
        "evidence_type": "engineering_integrated_field_current_step",
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Does not push particles or rebuild ion deposition from particle state.",
            "Marder correction is candidate-coupled to Yee edges with nondominance telemetry only.",
            "Predictor-corrector is an optional candidate end-step current solve, not a full accepted provisional ion push.",
            "Does not supply same-scope 3-D DPF validation.",
        ],
    }


def _field_work_telemetry(
    *,
    total_current_A_m2: np.ndarray,
    electric_field_V_m: np.ndarray,
    cell_volume_m3: float,
) -> dict[str, Any]:
    power_density = np.sum(total_current_A_m2 * electric_field_V_m, axis=-1)
    j_dot_e_power_W = float(np.sum(power_density) * cell_volume_m3)
    return {
        "status": "candidate_engineering_volume_j_dot_e_power_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "Auluck 2021 lines 151-200; hybrid source lines 740-805",
        "domain": "full_cartesian_grid_volume",
        "integral": "sum_cell_centered_J_dot_E_times_cell_volume",
        "j_dot_e_power_W": j_dot_e_power_W,
        "max_abs_power_density_W_m3": float(np.max(np.abs(power_density))),
        "cell_volume_m3": float(cell_volume_m3),
        "sign_convention": (
            "positive_J_dot_E_is_field_work_on_charges_candidate_not_accepted"
        ),
        "time_centering": "begin_step_E_with_midpoint_candidate_current",
        "can_support_first_principles_acceptance": False,
    }
