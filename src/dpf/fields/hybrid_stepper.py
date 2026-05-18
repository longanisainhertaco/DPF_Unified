"""Candidate integrated 3-D hybrid field-current stepper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.fields.conductivity import PlasmaVacuumConductivityBlend
from dpf.fields.marder import MarderCorrection
from dpf.fields.maxwell_3d import (
    EPSILON_0,
    HYBRID_PIC_3D_SOURCE,
    MU_0,
    Maxwell3DBoundaries,
    Maxwell3DFieldCore,
    Maxwell3DGrid,
    Maxwell3DState,
)
from dpf.fields.ohm_solver import GeneralizedOhmSolver
from dpf.fields.pic_coupling import PICCurrentSourcePort
from dpf.fields.predictor_corrector import CurrentPredictorCorrector
from dpf.fluid.constrained_transport import face_to_cell_centered

_NUMERICAL_ELECTRON_DENSITY_FLOOR_M3 = 1.0


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
        ohm_time_centering_theta: float = 0.5,
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
            ohm_time_centering_theta=ohm_time_centering_theta,
        )
        total_current, current_domain = mask_current_to_resolved_plasma(
            total_current,
            electron_density_m3,
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
            electron_density_m3=electron_density_m3,
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
                    dt_s=dt_s,
                    ohm_time_centering_theta=ohm_time_centering_theta,
                )
            )
            predictor_corrector_telemetry = pc_telemetry.to_dict()
            end_step_current, end_step_domain = mask_current_to_resolved_plasma(
                end_step_current,
                electron_density_m3,
            )
            predictor_corrector_telemetry["current_domain"] = end_step_domain
            self.previous_total_current_A_m2 = np.array(end_step_current, copy=True)
        after = self.maxwell.diagnostics(next_state)
        ohm_solver_packet = ohm_telemetry.to_dict()
        ohm_solver_packet["current_domain"] = current_domain
        telemetry = HybridPIC3DStepTelemetry(
            status="candidate_engineering_field_current_step",
            source=HYBRID_PIC_3D_SOURCE,
            conductivity=conductivity_telemetry.to_dict(),
            ohm_solver=ohm_solver_packet,
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


def mask_current_to_resolved_plasma(
    current_A_m2: np.ndarray,
    electron_density_m3: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Zero generalized-Ohm conduction current outside resolved plasma cells."""

    current = np.asarray(current_A_m2, dtype=float)
    density = np.asarray(electron_density_m3, dtype=float)
    threshold = _NUMERICAL_ELECTRON_DENSITY_FLOOR_M3 * (1.0 + 1.0e-12)
    resolved = density > threshold
    excluded = ~resolved
    current_norm = np.linalg.norm(current, axis=-1)
    masked = np.where(resolved[..., np.newaxis], current, 0.0)
    masked_norm = np.linalg.norm(masked, axis=-1)
    return masked, {
        "status": "candidate_resolved_plasma_current_domain_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "criterion": (
            "deposit generalized-Ohm conduction current only where n_e exceeds "
            "the numerical electron-density floor"
        ),
        "numerical_electron_density_floor_m3": (
            _NUMERICAL_ELECTRON_DENSITY_FLOOR_M3
        ),
        "resolved_cell_count": int(np.count_nonzero(resolved)),
        "excluded_numerical_floor_cell_count": int(np.count_nonzero(excluded)),
        "total_cell_count": int(density.size),
        "max_unmasked_current_A_m2": float(np.max(current_norm)),
        "max_masked_current_A_m2": float(np.max(masked_norm)),
        "max_excluded_numerical_floor_current_A_m2": (
            float(np.max(current_norm[excluded])) if np.any(excluded) else 0.0
        ),
        "can_support_first_principles_acceptance": False,
        "limitations": (
            "Candidate runtime domain guard only; vacuum-floor cells are not a physical electron-fluid conductor.",
        ),
    }


def omega_volume_j_dot_e_power_W(
    *,
    total_current_A_m2: np.ndarray,
    electric_field_V_m: np.ndarray,
    omega_volume_cells: np.ndarray,
) -> dict[str, Any]:
    """Return the Auluck volume J.E power over the named Omega domain.

    WP-N1 source packet S2 term 2. Re-scopes the volume J.E integral from the
    density-threshold mask to the Auluck omega_volume_cells partition label.
    Poynting theorem RHS is - integral_V (J.E) dV
    [KR: 2019nrlplasma-formulary-037290d4.md:1886]; Auluck eq 1 uses this same
    J.E as the numerator of I*V [KR: auluck-2021-dpf-circuit-element.md:173-197].
    Sign convention: positive J.E is field work on charges. Negative local
    J.E is physical (dynamo regions) and is NEVER clipped
    [KR: auluck-2021-dpf-circuit-element.md:158-160].
    """
    current = np.asarray(total_current_A_m2, dtype=float)
    e_field = np.asarray(electric_field_V_m, dtype=float)
    omega = np.asarray(omega_volume_cells, dtype=bool)
    power_density = np.sum(current * e_field, axis=-1)
    if omega.shape != power_density.shape:
        raise ValueError("omega_volume_cells shape does not match field grid")
    if np.any(omega):
        omega_power_density = power_density[omega]
        j_dot_e_power_W = float(np.sum(omega_power_density))
        max_abs_power_density = float(np.max(np.abs(omega_power_density)))
        min_signed_power_density = float(np.min(omega_power_density))
    else:
        j_dot_e_power_W = 0.0
        max_abs_power_density = 0.0
        min_signed_power_density = 0.0
    return {
        "status": "candidate_auluck_omega_volume_j_dot_e_power_not_validation",
        "domain": "omega_volume_cells",
        "integral": "sum_cell_centered_J_dot_E_over_omega_volume_cells",
        "j_dot_e_power_density_summed_W_m3": j_dot_e_power_W,
        "omega_cell_count": int(np.count_nonzero(omega)),
        "max_abs_power_density_W_m3": max_abs_power_density,
        "min_signed_power_density_W_m3": min_signed_power_density,
        "sign_convention": (
            "positive_J_dot_E_is_field_work_on_charges_candidate_not_accepted"
        ),
        "negative_local_j_dot_e_clipped": False,
        "source_refs": [
            "KnowledgeReference/auluck-2021-dpf-circuit-element.md:173-197",
            "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1880-1888",
        ],
        "can_support_first_principles_acceptance": False,
    }


def _poynting_face_flux_W(
    *,
    electric_field_V_m: np.ndarray,
    magnetic_field_T: np.ndarray,
    face_mask: np.ndarray,
    grid: Maxwell3DGrid,
) -> float:
    """Return outward Poynting flux N = E x H integrated over masked faces.

    WP-N1 source packet S2 terms 1/3. Poynting theorem surface term
    +integral_S (N.dS), N = E x H, dS outward
    [KR: 2019nrlplasma-formulary-037290d4.md:1880-1888]. E and B are
    interpolated to cell centers (a candidate numerical step, source packet
    section 2). The flux is the outflow-positive integral over the cell-label
    set: N projected on the axial (z) normal times the cell cross-section,
    summed over the labeled cells. This is an engineering surface estimate.
    """
    e_field = np.asarray(electric_field_V_m, dtype=float)
    b_field = np.asarray(magnetic_field_T, dtype=float)
    mask = np.asarray(face_mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    # H = B / mu0 ; N = E x H. Axial (z) component carries flux through the
    # axial cell-label slabs used by the partition.
    h_field = b_field / MU_0
    n_z = e_field[..., 0] * h_field[..., 1] - e_field[..., 1] * h_field[..., 0]
    cell_cross_section_m2 = float(grid.dx * grid.dy)
    return float(np.sum(n_z[mask]) * cell_cross_section_m2)


def wall_poynting_flux_W(
    *,
    electric_field_V_m: np.ndarray,
    magnetic_field_T: np.ndarray,
    wall_material_cells: np.ndarray,
    open_pml_cells: np.ndarray,
    grid: Maxwell3DGrid,
) -> dict[str, Any]:
    """Return wall + open-boundary Poynting outflow excluding the source port.

    WP-N1 source packet S2 term 3. Integrates +integral_S (N.dS) over
    Sigma minus the declared source-interface port, i.e. over
    wall_material_faces UNION open_pml_faces
    [KR: 2019nrlplasma-formulary-037290d4.md:1882-1888;
    auluck-2021-dpf-circuit-element.md:215-224,426-443]. Outflow-positive:
    positive means EM energy leaving Omega through walls / open boundary.
    """
    wall_flux_W = _poynting_face_flux_W(
        electric_field_V_m=electric_field_V_m,
        magnetic_field_T=magnetic_field_T,
        face_mask=wall_material_cells,
        grid=grid,
    )
    pml_flux_W = _poynting_face_flux_W(
        electric_field_V_m=electric_field_V_m,
        magnetic_field_T=magnetic_field_T,
        face_mask=open_pml_cells,
        grid=grid,
    )
    return {
        "status": "candidate_wall_poynting_flux_not_validation",
        "domain": "wall_material_faces_union_open_pml_faces",
        "wall_material_poynting_flux_W": wall_flux_W,
        "open_pml_poynting_flux_W": pml_flux_W,
        "wall_poynting_flux_excluding_declared_port_W": wall_flux_W + pml_flux_W,
        "sign_convention": "outflow_positive_energy_leaving_omega",
        "interpolation_note": (
            "E and B interpolated to cell centers; candidate numerical step"
        ),
        "geometry_review_status": "geometry_candidate_not_reviewed",
        "source_refs": [
            "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1880-1888",
            "KnowledgeReference/auluck-2021-dpf-circuit-element.md:215-224,426-443",
        ],
        "can_support_first_principles_acceptance": False,
    }


def omega_stored_em_energy_J(
    *,
    electric_field_V_m: np.ndarray,
    magnetic_field_T: np.ndarray,
    omega_volume_cells: np.ndarray,
    cell_volume_m3: float,
) -> float:
    """Return stored EM energy W = (1/2) integral_Omega (H.B + E.D) dV.

    WP-N1 source packet S2 term 5 component. W per the formulary
    [KR: 2019nrlplasma-formulary-037290d4.md:1869-1879], evaluated on
    omega_volume_cells. Returned as an instantaneous energy; the ledger
    forms Delta W = W(end) - W(begin).
    """
    e_field = np.asarray(electric_field_V_m, dtype=float)
    b_field = np.asarray(magnetic_field_T, dtype=float)
    omega = np.asarray(omega_volume_cells, dtype=bool)
    e_density = 0.5 * EPSILON_0 * np.sum(e_field * e_field, axis=-1)
    b_density = 0.5 * np.sum(b_field * b_field, axis=-1) / MU_0
    if omega.shape != e_density.shape:
        raise ValueError("omega_volume_cells shape does not match field grid")
    if not np.any(omega):
        return 0.0
    return float(np.sum((e_density + b_density)[omega]) * float(cell_volume_m3))


def _field_work_telemetry(
    *,
    total_current_A_m2: np.ndarray,
    electric_field_V_m: np.ndarray,
    electron_density_m3: np.ndarray,
    cell_volume_m3: float,
) -> dict[str, Any]:
    power_density = np.sum(total_current_A_m2 * electric_field_V_m, axis=-1)
    electron_density = np.asarray(electron_density_m3, dtype=float)
    resolved_plasma = electron_density > (1.0 + 1.0e-12)
    if np.any(resolved_plasma):
        j_dot_e_power_W = float(np.sum(power_density[resolved_plasma]) * cell_volume_m3)
        max_abs_resolved_power_density = float(
            np.max(np.abs(power_density[resolved_plasma]))
        )
    else:
        j_dot_e_power_W = 0.0
        max_abs_resolved_power_density = 0.0
    unmasked_j_dot_e_power_W = float(np.sum(power_density) * cell_volume_m3)
    return {
        "status": "candidate_engineering_volume_j_dot_e_power_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "Auluck 2021 lines 151-200; hybrid source lines 740-805",
        "domain": "resolved_plasma_current_carrying_cells",
        "integral": "sum_cell_centered_J_dot_E_times_cell_volume_over_resolved_plasma",
        "j_dot_e_power_W": j_dot_e_power_W,
        "unmasked_full_grid_j_dot_e_power_W": unmasked_j_dot_e_power_W,
        "max_abs_power_density_W_m3": max_abs_resolved_power_density,
        "max_abs_unmasked_power_density_W_m3": float(np.max(np.abs(power_density))),
        "power_domain_gate": {
            "status": "candidate_resolved_plasma_power_domain_not_validation",
            "source": "KnowledgeReference/auluck-2021-dpf-circuit-element.md",
            "source_lines": "151-209",
            "criterion": (
                "include cells with electron_density_m3 above the numerical "
                "electron-density floor"
            ),
            "included_cell_count": int(np.count_nonzero(resolved_plasma)),
            "excluded_numerical_floor_cell_count": int(
                resolved_plasma.size - np.count_nonzero(resolved_plasma)
            ),
            "total_cell_count": int(resolved_plasma.size),
            "electron_density_min_m3": float(np.min(electron_density)),
            "electron_density_max_m3": float(np.max(electron_density)),
            "numerical_electron_density_floor_m3": 1.0,
            "unmasked_minus_resolved_power_W": float(
                unmasked_j_dot_e_power_W - j_dot_e_power_W
            ),
            "can_support_first_principles_acceptance": False,
        },
        "cell_volume_m3": float(cell_volume_m3),
        "sign_convention": (
            "positive_J_dot_E_is_field_work_on_charges_candidate_not_accepted"
        ),
        "time_centering": "begin_step_E_with_midpoint_candidate_current",
        "can_support_first_principles_acceptance": False,
    }
