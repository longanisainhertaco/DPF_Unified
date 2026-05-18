"""Candidate electron-energy closure for the 3-D hybrid PIC-fluid path.

The local hybrid PIC-fluid source identifies ``Te = Ti`` as a simplifying
closure and states that pressure-gradient/Hall runs need a separate electron
temperature evolution with collisional coupling and heat-flux models before
they can be quantitative.  This module wraps the repo's existing
two-temperature source-term implementation behind explicit nonaccepting
telemetry for the 3-D first-principles gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from dpf.collision.spitzer import braginskii_kappa
from dpf.constants import c, e, k_B, m_e
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid
from dpf.fluid.implicit_diffusion import diffuse_field_1d_backward_euler
from dpf.fluid.two_temperature import (
    electron_energy_from_temperature,
    equilibration_convention_audit,
    step_electron_energy,
    temperature_from_electron_energy,
    two_temperature_model_metadata,
)

_NUMERICAL_ELECTRON_DENSITY_FLOOR_M3 = 1.0


@dataclass
class ElectronEnergyState:
    """Electron and ion temperature state for the candidate 3-D loop."""

    electron_energy_J_m3: np.ndarray
    electron_temperature_K: np.ndarray
    ion_temperature_K: np.ndarray

    def electron_pressure_Pa(self, electron_density_m3: np.ndarray) -> np.ndarray:
        return electron_density_m3 * k_B * self.electron_temperature_K


@dataclass(frozen=True)
class ElectronEnergyTelemetry:
    """Telemetry for one candidate electron-energy source update."""

    status: str
    source: str
    source_lines: str
    model_metadata: dict[str, Any]
    min_electron_temperature_K: float
    max_electron_temperature_K: float
    min_ion_temperature_K: float
    max_ion_temperature_K: float
    max_abs_delta_electron_temperature_K: float
    max_current_A_m2: float
    max_resistivity_ohm_m: float
    include_ohmic_heating: bool
    include_electron_velocity_pressure_work: bool
    include_equilibration: bool
    include_bremsstrahlung_loss: bool
    include_heat_flux: bool
    heat_flux: dict[str, Any]
    density_reconciliation: dict[str, Any]
    equilibration_audit: dict[str, Any]
    max_electron_current_drift_m_s: float
    closure_validity: dict[str, Any]
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ElectronEnergyClosure:
    """Operator-split candidate electron-energy source update."""

    capability_id = "separate_electron_energy_closure"

    def __init__(self, grid: Maxwell3DGrid) -> None:
        self.grid = grid

    def initialize(
        self,
        *,
        electron_temperature_K: np.ndarray | float,
        ion_temperature_K: np.ndarray | float,
        electron_density_m3: np.ndarray,
    ) -> ElectronEnergyState:
        ne = _scalar("electron_density_m3", electron_density_m3, self.grid.shape)
        Te = _scalar_or_full(
            "electron_temperature_K",
            electron_temperature_K,
            self.grid.shape,
        )
        Ti = _scalar_or_full("ion_temperature_K", ion_temperature_K, self.grid.shape)
        _require_positive("electron_density_m3", ne)
        _require_positive("electron_temperature_K", Te)
        _require_positive("ion_temperature_K", Ti)
        return ElectronEnergyState(
            electron_energy_J_m3=electron_energy_from_temperature(Te, ne),
            electron_temperature_K=Te,
            ion_temperature_K=Ti,
        )

    def step_sources(
        self,
        state: ElectronEnergyState,
        *,
        electron_density_m3: np.ndarray,
        ion_density_m3: np.ndarray,
        mass_density_kg_m3: np.ndarray,
        velocity_m_s: np.ndarray,
        resistivity_ohm_m: np.ndarray | float,
        current_A_m2: np.ndarray,
        dt_s: float,
        charge_state_Z: float = 1.0,
        gaunt_factor: float = 1.2,
        temperature_floor_K: float = 1.0,
        magnetic_field_T: np.ndarray | None = None,
        heat_flux_subcycles_max: int = 1000,
    ) -> tuple[ElectronEnergyState, ElectronEnergyTelemetry]:
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        if charge_state_Z <= 0.0:
            raise ValueError("charge_state_Z must be positive")
        if temperature_floor_K <= 0.0:
            raise ValueError("temperature_floor_K must be positive")
        if int(heat_flux_subcycles_max) != heat_flux_subcycles_max:
            raise ValueError("heat_flux_subcycles_max must be an integer")
        if heat_flux_subcycles_max < 1:
            raise ValueError("heat_flux_subcycles_max must be positive")

        ne = _scalar("electron_density_m3", electron_density_m3, self.grid.shape)
        ni = _scalar("ion_density_m3", ion_density_m3, self.grid.shape)
        rho = _scalar("mass_density_kg_m3", mass_density_kg_m3, self.grid.shape)
        eta = _scalar_or_full("resistivity_ohm_m", resistivity_ohm_m, self.grid.shape)
        current = _vector("current_A_m2", current_A_m2, self.grid.shape)
        velocity = _vector("velocity_m_s", velocity_m_s, self.grid.shape)
        _require_positive("electron_density_m3", ne)
        _require_positive("ion_density_m3", ni)
        _require_positive("mass_density_kg_m3", rho)
        if np.any(eta < 0.0):
            raise ValueError("resistivity_ohm_m must be non-negative")

        before_Te = _scalar(
            "state.electron_temperature_K",
            state.electron_temperature_K,
            self.grid.shape,
        )
        before_Ti = _scalar(
            "state.ion_temperature_K",
            state.ion_temperature_K,
            self.grid.shape,
        )
        energy = _scalar(
            "state.electron_energy_J_m3",
            state.electron_energy_J_m3,
            self.grid.shape,
        )
        energy, density_reconciliation = _reconcile_energy_density_to_temperature(
            stored_energy_J_m3=energy,
            electron_temperature_K=before_Te,
            electron_density_m3=ne,
        )
        working_energy, working_Te, heat_flux_telemetry = (
            _apply_braginskii_heat_flux_candidate(
                electron_energy_J_m3=energy,
                electron_temperature_K=before_Te,
                electron_density_m3=ne,
                magnetic_field_T=magnetic_field_T,
                grid=self.grid,
                dt_s=dt_s,
                charge_state_Z=charge_state_Z,
                temperature_floor_K=temperature_floor_K,
                max_subcycles=int(heat_flux_subcycles_max),
            )
        )
        electron_current_drift, resolved_plasma, electron_fluid_domain = (
            _electron_current_drift_on_resolved_plasma(
                current_A_m2=current,
                electron_density_m3=ne,
            )
        )
        J_sq = np.sum(current * current, axis=-1)
        J_sq_for_sources = np.where(resolved_plasma, J_sq, 0.0)
        eta_for_sources = np.where(resolved_plasma, eta, 0.0)
        max_drift_m_s = float(
            electron_fluid_domain["max_resolved_current_drift_m_s"]
        )
        pre_source_validity = _electron_closure_validity_packet(
            electron_temperature_K=before_Te,
            max_electron_current_drift_m_s=max_drift_m_s,
            electron_fluid_domain=electron_fluid_domain,
        )
        if pre_source_validity["status"].startswith("blocked_"):
            metadata = two_temperature_model_metadata()
            equilibration_audit = equilibration_convention_audit(
                electron_temperature_K=before_Te,
                ion_temperature_K=before_Ti,
                electron_density_m3=ne,
                ion_density_m3=ni,
                Z=charge_state_Z,
            )
            telemetry = ElectronEnergyTelemetry(
                status=pre_source_validity["status"],
                source=HYBRID_PIC_3D_SOURCE,
                source_lines="1074-1097, 1226-1240, 1267-1278",
                model_metadata=metadata,
                min_electron_temperature_K=float(np.min(before_Te)),
                max_electron_temperature_K=float(np.max(before_Te)),
                min_ion_temperature_K=float(np.min(before_Ti)),
                max_ion_temperature_K=float(np.max(before_Ti)),
                max_abs_delta_electron_temperature_K=0.0,
                max_current_A_m2=float(np.max(np.linalg.norm(current, axis=-1))),
                max_resistivity_ohm_m=float(np.max(eta)),
                include_ohmic_heating=False,
                include_electron_velocity_pressure_work=False,
                include_equilibration=False,
                include_bremsstrahlung_loss=False,
                include_heat_flux=False,
                heat_flux={
                    "status": "not_applied_blocked_electron_closure_validity",
                    "applied": False,
                    "subcycles": 0,
                },
                density_reconciliation=density_reconciliation,
                equilibration_audit=equilibration_audit,
                max_electron_current_drift_m_s=max_drift_m_s,
                closure_validity=pre_source_validity,
            )
            return state, telemetry
        electron_velocity = velocity - electron_current_drift
        electron_velocity = np.where(
            resolved_plasma[..., np.newaxis],
            electron_velocity,
            0.0,
        )
        velocity_fluid_layout = np.moveaxis(electron_velocity, -1, 0)
        new_energy, new_Te, new_Ti = step_electron_energy(
            working_energy,
            rho,
            velocity_fluid_layout,
            eta_for_sources,
            J_sq_for_sources,
            working_Te,
            before_Ti,
            ne,
            ni,
            min(self.grid.spacing),
            dt_s,
            Z=charge_state_Z,
            gaunt_factor=gaunt_factor,
            Te_floor=temperature_floor_K,
        )
        next_state = ElectronEnergyState(
            electron_energy_J_m3=new_energy,
            electron_temperature_K=new_Te,
            ion_temperature_K=new_Ti,
        )
        closure_validity = _electron_closure_validity_packet(
            electron_temperature_K=new_Te,
            max_electron_current_drift_m_s=max_drift_m_s,
            electron_fluid_domain=electron_fluid_domain,
        )
        if closure_validity["status"].startswith("blocked_"):
            new_energy = working_energy
            new_Te = working_Te
            new_Ti = before_Ti
            next_state = ElectronEnergyState(
                electron_energy_J_m3=new_energy,
                electron_temperature_K=new_Te,
                ion_temperature_K=new_Ti,
            )
        metadata = two_temperature_model_metadata()
        equilibration_audit = equilibration_convention_audit(
            electron_temperature_K=working_Te,
            ion_temperature_K=before_Ti,
            electron_density_m3=ne,
            ion_density_m3=ni,
            Z=charge_state_Z,
        )
        telemetry = ElectronEnergyTelemetry(
            status=(
                closure_validity["status"]
                if closure_validity["status"].startswith("blocked_")
                else "candidate_engineering_electron_energy_closure"
            ),
            source=HYBRID_PIC_3D_SOURCE,
            source_lines="1074-1097, 1226-1240, 1267-1278",
            model_metadata=metadata,
            min_electron_temperature_K=float(np.min(new_Te)),
            max_electron_temperature_K=float(np.max(new_Te)),
            min_ion_temperature_K=float(np.min(new_Ti)),
            max_ion_temperature_K=float(np.max(new_Ti)),
            max_abs_delta_electron_temperature_K=float(np.max(np.abs(new_Te - before_Te))),
            max_current_A_m2=float(np.max(np.linalg.norm(current, axis=-1))),
            max_resistivity_ohm_m=float(np.max(eta)),
            include_ohmic_heating=True,
            include_electron_velocity_pressure_work=True,
            include_equilibration=True,
            include_bremsstrahlung_loss=True,
            include_heat_flux=bool(heat_flux_telemetry["applied"]),
            heat_flux=heat_flux_telemetry,
            density_reconciliation=density_reconciliation,
            equilibration_audit=equilibration_audit,
            max_electron_current_drift_m_s=max_drift_m_s,
            closure_validity=closure_validity,
        )
        return next_state, telemetry


def electron_energy_candidate_evidence(
    telemetry: ElectronEnergyTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for the electron-energy closure."""
    return {
        "passed": telemetry.status == "candidate_engineering_electron_energy_closure",
        "status": "candidate",
        "capability": ElectronEnergyClosure.capability_id,
        "source": telemetry.source,
        "source_lines": telemetry.source_lines,
        "implementation": "src/dpf/fields/electron_energy.py",
        "evidence_type": "engineering_electron_energy_source_update",
        "max_abs_delta_electron_temperature_K": (
            telemetry.max_abs_delta_electron_temperature_K
        ),
        "heat_flux": telemetry.heat_flux,
        "density_reconciliation": telemetry.density_reconciliation,
        "equilibration_audit": telemetry.equilibration_audit,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Uses the repo two-temperature source terms; Braginskii heat flux is candidate-only and relaxation conventions remain source-audit blocked.",
            "Electron pressure work uses the current velocity u_e = u_i - J/(e n_e), but the closure is still operator-split and candidate-only.",
            "No same-scope electron-temperature diagnostic or neutron-yield UQ packet is attached.",
        ],
    }


def extended_ohm_temperature_authority_status(
    *,
    include_hall: bool,
    include_pressure: bool,
    electron_energy_evidence: ElectronEnergyTelemetry | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail-closed authority check for Te-sensitive pressure/Hall Ohm terms."""
    requires_separate_te = bool(include_hall or include_pressure)
    if not requires_separate_te:
        return {
            "status": "not_required_for_baseline_resistive_ohm",
            "source": HYBRID_PIC_3D_SOURCE,
            "source_lines": "1226-1240",
            "requires_separate_te": False,
            "include_hall": bool(include_hall),
            "include_pressure": bool(include_pressure),
            "can_support_pressure_hall_quantitative_claims": True,
            "can_support_first_principles_acceptance": False,
        }

    evidence = _evidence_mapping(electron_energy_evidence)
    if evidence is None:
        return {
            "status": "blocked_te_equal_ti_or_missing_separate_te",
            "source": HYBRID_PIC_3D_SOURCE,
            "source_lines": "1226-1240",
            "requires_separate_te": True,
            "include_hall": bool(include_hall),
            "include_pressure": bool(include_pressure),
            "can_support_pressure_hall_quantitative_claims": False,
            "can_support_first_principles_acceptance": False,
            "blocker": (
                "Pressure-gradient/Hall terms are qualitative only until a "
                "reviewed separate Te equation, accepted heat flux, collisional "
                "coupling, diagnostics, and UQ packet are accepted."
            ),
        }

    status = str(evidence.get("status") or "").strip().lower()
    accepted = evidence.get("passed") is True and status in {"accepted", "validated"}
    if accepted:
        authority_status = "accepted_separate_te_authority"
    else:
        authority_status = "candidate_separate_te_still_blocked"
    return {
        "status": authority_status,
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "1226-1240",
        "requires_separate_te": True,
        "include_hall": bool(include_hall),
        "include_pressure": bool(include_pressure),
        "evidence_status": status or "unset",
        "can_support_pressure_hall_quantitative_claims": accepted,
        "can_support_first_principles_acceptance": accepted,
    }


def _evidence_mapping(
    evidence: ElectronEnergyTelemetry | Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if evidence is None:
        return None
    if isinstance(evidence, ElectronEnergyTelemetry):
        return evidence.to_dict()
    if isinstance(evidence, Mapping):
        return evidence
    return None


def _electron_closure_validity_packet(
    *,
    electron_temperature_K: np.ndarray,
    max_electron_current_drift_m_s: float,
    electron_fluid_domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_temperature = float(np.max(electron_temperature_K))
    max_thermal_speed = float(
        np.sqrt(2.0 * k_B * max(max_temperature, 0.0) / m_e)
    )
    drift_ratio = max_electron_current_drift_m_s / c
    thermal_ratio = max_thermal_speed / c
    if drift_ratio >= 1.0:
        status = "blocked_superluminal_electron_current_drift"
    elif thermal_ratio >= 1.0:
        status = "blocked_relativistic_electron_temperature"
    else:
        status = "candidate_nonrelativistic_electron_closure_in_range"
    return {
        "status": status,
        "model_scope": "nonrelativistic_braginskii_ohm_electron_energy",
        "max_electron_temperature_K": max_temperature,
        "max_electron_thermal_speed_m_s": max_thermal_speed,
        "max_electron_current_drift_m_s": float(max_electron_current_drift_m_s),
        "thermal_speed_to_c": thermal_ratio,
        "current_drift_to_c": drift_ratio,
        "electron_fluid_domain": electron_fluid_domain,
        "can_support_first_principles_acceptance": False,
    }


def _electron_current_drift_on_resolved_plasma(
    *,
    current_A_m2: np.ndarray,
    electron_density_m3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return J/(e n_e) only on cells where an electron fluid is resolved."""

    density = np.asarray(electron_density_m3, dtype=float)
    current = np.asarray(current_A_m2, dtype=float)
    threshold = _NUMERICAL_ELECTRON_DENSITY_FLOOR_M3 * (1.0 + 1.0e-12)
    resolved = density > threshold
    drift = np.divide(
        current,
        e * density[..., np.newaxis],
        out=np.zeros_like(current),
        where=resolved[..., np.newaxis],
    )
    current_norm = np.linalg.norm(current, axis=-1)
    drift_norm = np.linalg.norm(drift, axis=-1)
    resolved_count = int(np.count_nonzero(resolved))
    excluded = ~resolved
    max_resolved_current = (
        float(np.max(current_norm[resolved])) if resolved_count else 0.0
    )
    max_resolved_drift = (
        float(np.max(drift_norm[resolved])) if resolved_count else 0.0
    )
    max_excluded_current = (
        float(np.max(current_norm[excluded])) if np.any(excluded) else 0.0
    )
    return drift, resolved, {
        "status": "candidate_resolved_plasma_electron_fluid_domain_not_validation",
        "source": HYBRID_PIC_3D_SOURCE,
        "criterion": (
            "evaluate nonrelativistic electron-fluid drift and source work only "
            "where n_e exceeds the numerical electron-density floor"
        ),
        "numerical_electron_density_floor_m3": (
            _NUMERICAL_ELECTRON_DENSITY_FLOOR_M3
        ),
        "min_electron_density_m3": float(np.min(density)),
        "max_electron_density_m3": float(np.max(density)),
        "resolved_cell_count": resolved_count,
        "excluded_numerical_floor_cell_count": int(np.count_nonzero(excluded)),
        "total_cell_count": int(density.size),
        "all_cells_at_numerical_floor": resolved_count == 0,
        "max_resolved_current_A_m2": max_resolved_current,
        "max_resolved_current_drift_m_s": max_resolved_drift,
        "max_excluded_numerical_floor_current_A_m2": max_excluded_current,
        "can_support_first_principles_acceptance": False,
        "limitations": (
            "Candidate runtime domain guard only; floor cells are vacuum bookkeeping, not a physical electron fluid.",
        ),
    }


def _reconcile_energy_density_to_temperature(
    *,
    stored_energy_J_m3: np.ndarray,
    electron_temperature_K: np.ndarray,
    electron_density_m3: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep electron internal-energy density consistent with current n_e and T_e."""

    reconciled = electron_energy_from_temperature(
        electron_temperature_K,
        electron_density_m3,
    )
    delta = reconciled - stored_energy_J_m3
    scale = np.maximum.reduce([
        np.abs(stored_energy_J_m3),
        np.abs(reconciled),
        np.ones_like(reconciled),
    ])
    max_relative_delta = float(np.max(np.abs(delta) / scale))
    status = (
        "candidate_density_temperature_energy_reconciled"
        if max_relative_delta > 1.0e-12
        else "density_temperature_energy_already_consistent"
    )
    return (
        reconciled,
        {
            "status": status,
            "source": HYBRID_PIC_3D_SOURCE,
            "formula": "u_e = 3/2 n_e k_B T_e",
            "preserves_temperature_field": True,
            "stored_energy_min_J_m3": float(np.min(stored_energy_J_m3)),
            "stored_energy_max_J_m3": float(np.max(stored_energy_J_m3)),
            "reconciled_energy_min_J_m3": float(np.min(reconciled)),
            "reconciled_energy_max_J_m3": float(np.max(reconciled)),
            "max_abs_delta_energy_J_m3": float(np.max(np.abs(delta))),
            "max_relative_delta": max_relative_delta,
            "electron_density_min_m3": float(np.min(electron_density_m3)),
            "electron_density_max_m3": float(np.max(electron_density_m3)),
            "can_support_first_principles_acceptance": False,
            "limitations": (
                "Candidate density/temperature consistency repair only.",
                "Ionization potential energy and molecular chemistry energy sinks remain separate unresolved closures.",
            ),
        },
    )


def _scalar(name: str, value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != expected {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _scalar_or_full(
    name: str,
    value: np.ndarray | float,
    shape: tuple[int, int, int],
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(shape, float(arr), dtype=float)
    return _scalar(name, arr, shape)


def _vector(name: str, value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    expected = shape + (3,)
    if arr.shape != expected:
        raise ValueError(f"{name} shape {arr.shape} != expected {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _vector_optional(
    name: str,
    value: np.ndarray | None,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    expected_cell_last = shape + (3,)
    expected_component_first = (3,) + shape
    if arr.shape == expected_cell_last:
        vector = arr
    elif arr.shape == expected_component_first:
        vector = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(
            f"{name} shape {arr.shape} != expected {expected_cell_last} "
            f"or {expected_component_first}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _require_positive(name: str, value: np.ndarray) -> None:
    if np.any(value <= 0.0):
        raise ValueError(f"{name} must be positive")


def _apply_braginskii_heat_flux_candidate(
    *,
    electron_energy_J_m3: np.ndarray,
    electron_temperature_K: np.ndarray,
    electron_density_m3: np.ndarray,
    magnetic_field_T: np.ndarray | None,
    grid: Maxwell3DGrid,
    dt_s: float,
    charge_state_Z: float,
    temperature_floor_K: float,
    max_subcycles: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply source-backed Braginskii heat flux as non-promoting telemetry."""
    base = {
        "source": (
            "KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.json"
        ),
        "source_lines": "57-62",
        "transport_source": "src/dpf/collision/spitzer.py::braginskii_kappa",
        "boundary_condition": "candidate_zero_normal_heat_flux",
        "can_support_first_principles_acceptance": False,
    }
    if magnetic_field_T is None or dt_s == 0.0:
        return (
            electron_energy_J_m3,
            electron_temperature_K,
            {
                **base,
                "status": "not_applied_missing_magnetic_field_or_zero_dt",
                "applied": False,
                "subcycles": 0,
            },
        )

    density_gate = _heat_flux_density_gate(electron_density_m3)
    if density_gate["all_cells_at_numerical_floor"]:
        return (
            electron_energy_J_m3,
            electron_temperature_K,
            {
                **base,
                "status": "not_applied_no_resolved_plasma_electron_density",
                "applied": False,
                "subcycles": 0,
                "density_gate": density_gate,
                "temperature_floor_contact_count": 0,
            },
        )

    B = _vector_optional("magnetic_field_T", magnetic_field_T, grid.shape)
    assert B is not None
    Te = np.array(electron_temperature_K, dtype=float, copy=True)
    ne = np.array(electron_density_m3, dtype=float, copy=False)
    energy = np.array(electron_energy_J_m3, dtype=float, copy=True)
    B_mag = np.sqrt(np.sum(B * B, axis=-1))
    kappa_par, kappa_perp = braginskii_kappa(ne, Te, B_mag, charge_state_Z)
    kappa_par = np.maximum(np.where(np.isfinite(kappa_par), kappa_par, 0.0), 0.0)
    kappa_perp = np.maximum(np.where(np.isfinite(kappa_perp), kappa_perp, 0.0), 0.0)
    max_kappa = float(np.max(kappa_par))
    min_ne = float(np.min(np.maximum(ne, 1.0)))
    if max_kappa <= 0.0:
        return (
            energy,
            Te,
            {
                **base,
                "status": "not_applied_zero_conductivity",
                "applied": False,
                "subcycles": 0,
                "density_gate": density_gate,
                "max_kappa_parallel_W_m_K": max_kappa,
            },
        )

    diffusivity = max_kappa / max(1.5 * min_ne * k_B, 1e-300)
    dx_min = min(grid.spacing)
    n_dim = 3
    dt_stable = 0.25 * dx_min * dx_min / max(n_dim * diffusivity, 1e-300)
    if not np.isfinite(dt_stable) or dt_stable <= 0.0:
        return (
            energy,
            Te,
            {
                **base,
                "status": "blocked_nonfinite_heat_flux_stability_limit",
                "applied": False,
                "subcycles": 0,
                "density_gate": density_gate,
                "max_kappa_parallel_W_m_K": max_kappa,
            },
        )
    required_subcycles = max(1, int(np.ceil(dt_s / dt_stable)))
    if required_subcycles > max_subcycles:
        return _apply_implicit_braginskii_heat_flux_candidate(
            base=base,
            electron_energy_J_m3=energy,
            electron_density_m3=ne,
            magnetic_field_T=B,
            kappa_parallel_W_m_K=kappa_par,
            kappa_perpendicular_W_m_K=kappa_perp,
            grid=grid,
            dt_s=dt_s,
            temperature_floor_K=temperature_floor_K,
            explicit_required_subcycles=required_subcycles,
            explicit_max_subcycles=max_subcycles,
            explicit_dt_stable_s=dt_stable,
            explicit_max_kappa_parallel_W_m_K=max_kappa,
            density_gate=density_gate,
        )

    dt_sub = dt_s / required_subcycles
    max_abs_source = 0.0
    for _ in range(required_subcycles):
        B_mag = np.sqrt(np.sum(B * B, axis=-1))
        kappa_par, kappa_perp = braginskii_kappa(ne, Te, B_mag, charge_state_Z)
        kappa_par = np.maximum(np.where(np.isfinite(kappa_par), kappa_par, 0.0), 0.0)
        kappa_perp = np.maximum(np.where(np.isfinite(kappa_perp), kappa_perp, 0.0), 0.0)
        heat_source = _braginskii_heat_source(
            electron_temperature_K=Te,
            magnetic_field_T=B,
            kappa_parallel_W_m_K=kappa_par,
            kappa_perpendicular_W_m_K=kappa_perp,
            spacing_m=grid.spacing,
        )
        max_abs_source = max(max_abs_source, float(np.max(np.abs(heat_source))))
        energy = energy + dt_sub * heat_source
        energy_floor = 1.5 * ne * k_B * temperature_floor_K
        energy = np.maximum(energy, energy_floor)
        Te = temperature_from_electron_energy(energy, ne, temperature_floor_K)

    return (
        energy,
        Te,
        {
            **base,
            "status": "candidate_braginskii_anisotropic_heat_flux_applied",
            "applied": True,
            "subcycles": required_subcycles,
            "dt_stable_s": dt_stable,
            "max_kappa_parallel_W_m_K": float(np.max(kappa_par)),
            "max_kappa_perpendicular_W_m_K": float(np.max(kappa_perp)),
            "max_abs_heat_flux_source_W_m3": max_abs_source,
            "net_heat_flux_power_W": float(np.sum(heat_source) * grid.cell_volume),
            "density_gate": density_gate,
        },
    )


def _heat_flux_density_gate(electron_density_m3: np.ndarray) -> dict[str, Any]:
    density = np.asarray(electron_density_m3, dtype=float)
    threshold = _NUMERICAL_ELECTRON_DENSITY_FLOOR_M3 * (1.0 + 1.0e-12)
    resolved = density > threshold
    return {
        "status": "candidate_resolved_plasma_electron_density_gate",
        "source": HYBRID_PIC_3D_SOURCE,
        "criterion": "apply electron heat flux only where n_e exceeds the numerical electron-density floor",
        "numerical_electron_density_floor_m3": (
            _NUMERICAL_ELECTRON_DENSITY_FLOOR_M3
        ),
        "min_electron_density_m3": float(np.min(density)),
        "max_electron_density_m3": float(np.max(density)),
        "resolved_cell_count": int(np.count_nonzero(resolved)),
        "total_cell_count": int(density.size),
        "all_cells_at_numerical_floor": int(np.count_nonzero(resolved)) == 0,
        "can_support_first_principles_acceptance": False,
        "limitations": (
            "Candidate runtime guard only; true startup still needs accepted breakdown/avalanche and neutral/electron energy exchange.",
        ),
    }


def _apply_implicit_braginskii_heat_flux_candidate(
    *,
    base: dict[str, Any],
    electron_energy_J_m3: np.ndarray,
    electron_density_m3: np.ndarray,
    magnetic_field_T: np.ndarray,
    kappa_parallel_W_m_K: np.ndarray,
    kappa_perpendicular_W_m_K: np.ndarray,
    grid: Maxwell3DGrid,
    dt_s: float,
    temperature_floor_K: float,
    explicit_required_subcycles: int,
    explicit_max_subcycles: int,
    explicit_dt_stable_s: float,
    explicit_max_kappa_parallel_W_m_K: float,
    density_gate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Backward-Euler frozen-coefficient tensor heat-flux solve."""

    heat_capacity = 1.5 * electron_density_m3 * k_B
    energy_floor = heat_capacity * temperature_floor_K
    rhs = np.asarray(electron_energy_J_m3, dtype=float).ravel()
    scale = max(float(np.linalg.norm(rhs)), 1.0)
    shape = electron_energy_J_m3.shape

    def matvec(flat_energy: np.ndarray) -> np.ndarray:
        energy_trial = np.asarray(flat_energy, dtype=float).reshape(shape)
        temperature_trial = energy_trial / heat_capacity
        source = _braginskii_heat_source(
            electron_temperature_K=temperature_trial,
            magnetic_field_T=magnetic_field_T,
            kappa_parallel_W_m_K=kappa_parallel_W_m_K,
            kappa_perpendicular_W_m_K=kappa_perpendicular_W_m_K,
            spacing_m=grid.spacing,
        )
        return (energy_trial - dt_s * source).ravel()

    operator = LinearOperator(
        (rhs.size, rhs.size),
        matvec=matvec,
        dtype=np.float64,
    )
    iteration_count = 0

    def _count_iteration(_residual: float) -> None:
        nonlocal iteration_count
        iteration_count += 1

    solution, info = gmres(
        operator,
        rhs,
        x0=rhs,
        rtol=1.0e-8,
        atol=1.0e-12 * scale,
        restart=min(50, rhs.size),
        maxiter=200,
        callback=_count_iteration,
        callback_type="pr_norm",
    )
    residual_abs = float(np.linalg.norm(operator.matvec(solution) - rhs))
    residual_rel = residual_abs / scale
    if info != 0 or not np.all(np.isfinite(solution)) or residual_rel > 1.0e-6:
        return _apply_diagonal_adi_braginskii_heat_flux_candidate(
            base=base,
            electron_energy_J_m3=electron_energy_J_m3,
            electron_density_m3=electron_density_m3,
            magnetic_field_T=magnetic_field_T,
            kappa_parallel_W_m_K=kappa_parallel_W_m_K,
            kappa_perpendicular_W_m_K=kappa_perpendicular_W_m_K,
            grid=grid,
            dt_s=dt_s,
            temperature_floor_K=temperature_floor_K,
            explicit_required_subcycles=explicit_required_subcycles,
            explicit_max_subcycles=explicit_max_subcycles,
            explicit_dt_stable_s=explicit_dt_stable_s,
            explicit_max_kappa_parallel_W_m_K=explicit_max_kappa_parallel_W_m_K,
            density_gate=density_gate,
            tensor_solver_info={
                "implicit_solver_info": int(info),
                "implicit_iterations": int(iteration_count),
                "implicit_residual_abs": residual_abs,
                "implicit_residual_rel": residual_rel,
            },
        )

    implicit_energy = np.maximum(solution.reshape(shape), energy_floor)
    implicit_temperature = temperature_from_electron_energy(
        implicit_energy,
        electron_density_m3,
        temperature_floor_K,
    )
    heat_source = (implicit_energy - electron_energy_J_m3) / dt_s
    return (
        implicit_energy,
        implicit_temperature,
        {
            **base,
            "status": "candidate_braginskii_anisotropic_heat_flux_implicit_applied",
            "applied": True,
            "required_subcycles": explicit_required_subcycles,
            "subcycles": 0,
            "max_subcycles": explicit_max_subcycles,
            "dt_stable_s": explicit_dt_stable_s,
            "max_kappa_parallel_W_m_K": explicit_max_kappa_parallel_W_m_K,
            "max_kappa_perpendicular_W_m_K": float(
                np.max(kappa_perpendicular_W_m_K)
            ),
            "implicit_scheme": "frozen_coefficient_backward_euler_tensor_conduction",
            "implicit_solver": "scipy.sparse.linalg.gmres",
            "implicit_solver_info": int(info),
            "implicit_iterations": int(iteration_count),
            "implicit_residual_abs": residual_abs,
            "implicit_residual_rel": residual_rel,
            "max_abs_heat_flux_source_W_m3": float(np.max(np.abs(heat_source))),
            "net_heat_flux_power_W": float(np.sum(heat_source) * grid.cell_volume),
            "density_gate": density_gate,
            "temperature_floor_contact_count": int(
                np.count_nonzero(implicit_energy <= energy_floor)
            ),
        },
    )


def _apply_diagonal_adi_braginskii_heat_flux_candidate(
    *,
    base: dict[str, Any],
    electron_energy_J_m3: np.ndarray,
    electron_density_m3: np.ndarray,
    magnetic_field_T: np.ndarray,
    kappa_parallel_W_m_K: np.ndarray,
    kappa_perpendicular_W_m_K: np.ndarray,
    grid: Maxwell3DGrid,
    dt_s: float,
    temperature_floor_K: float,
    explicit_required_subcycles: int,
    explicit_max_subcycles: int,
    explicit_dt_stable_s: float,
    explicit_max_kappa_parallel_W_m_K: float,
    density_gate: dict[str, Any],
    tensor_solver_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    heat_capacity = 1.5 * electron_density_m3 * k_B
    temperature = electron_energy_J_m3 / heat_capacity
    B_mag = np.sqrt(np.sum(magnetic_field_T * magnetic_field_T, axis=-1))
    b_hat = np.divide(
        magnetic_field_T,
        B_mag[..., np.newaxis],
        out=np.zeros_like(magnetic_field_T),
        where=B_mag[..., np.newaxis] > 0.0,
    )
    field = np.array(temperature, dtype=float, copy=True)
    if not np.all(np.isfinite(field)):
        return (
            electron_energy_J_m3,
            temperature,
            {
                **base,
                "status": "blocked_implicit_heat_flux_solver_failed",
                "applied": False,
                "required_subcycles": explicit_required_subcycles,
                "subcycles": 0,
                "max_subcycles": explicit_max_subcycles,
                "max_kappa_parallel_W_m_K": explicit_max_kappa_parallel_W_m_K,
                "dt_stable_s": explicit_dt_stable_s,
                "implicit_scheme": (
                    "frozen_coefficient_backward_euler_tensor_conduction"
                ),
                "implicit_solver": "scipy.sparse.linalg.gmres",
                "density_gate": density_gate,
                **tensor_solver_info,
            },
        )

    for axis, spacing in enumerate(grid.spacing):
        kappa_axis = kappa_perpendicular_W_m_K + (
            kappa_parallel_W_m_K - kappa_perpendicular_W_m_K
        ) * b_hat[..., axis] ** 2
        diffusivity = kappa_axis / np.maximum(heat_capacity, 1.0e-300)
        field = _apply_axis_diffusion(
            field=field,
            diffusivity=diffusivity,
            dt_s=dt_s,
            spacing_m=spacing,
            axis=axis,
        )
        if not np.all(np.isfinite(field)):
            return (
                electron_energy_J_m3,
                temperature,
                {
                    **base,
                    "status": "blocked_diagonal_adi_heat_flux_nonfinite",
                    "applied": False,
                    "required_subcycles": explicit_required_subcycles,
                    "subcycles": 0,
                    "max_subcycles": explicit_max_subcycles,
                    "max_kappa_parallel_W_m_K": (
                        explicit_max_kappa_parallel_W_m_K
                    ),
                    "dt_stable_s": explicit_dt_stable_s,
                    "implicit_scheme": (
                        "diagonal_braginskii_backward_euler_adi_fallback"
                    ),
                    "density_gate": density_gate,
                    **tensor_solver_info,
                },
            )

    raw_min_temperature = float(np.min(field))
    temperature_floor_contact_count = int(np.count_nonzero(field <= temperature_floor_K))
    implicit_temperature = np.maximum(field, temperature_floor_K)
    implicit_energy = electron_energy_from_temperature(
        implicit_temperature,
        electron_density_m3,
    )
    heat_source = (implicit_energy - electron_energy_J_m3) / dt_s
    return (
        implicit_energy,
        implicit_temperature,
        {
            **base,
            "status": "candidate_braginskii_diagonal_adi_heat_flux_applied",
            "applied": True,
            "required_subcycles": explicit_required_subcycles,
            "subcycles": 0,
            "max_subcycles": explicit_max_subcycles,
            "dt_stable_s": explicit_dt_stable_s,
            "max_kappa_parallel_W_m_K": explicit_max_kappa_parallel_W_m_K,
            "max_kappa_perpendicular_W_m_K": float(np.max(kappa_perpendicular_W_m_K)),
            "implicit_scheme": "diagonal_braginskii_backward_euler_adi_fallback",
            "implicit_solver": (
                "dpf.fluid.implicit_diffusion.diffuse_field_1d_backward_euler"
            ),
            "positivity_scheme": "backward_euler_m_matrix_diagonal_adi",
            "positivity_preserving_for_nonnegative_temperature": True,
            "nonlinear_kappa_frozen": True,
            "omitted_cross_derivative_terms": True,
            "fallback_after_tensor_solver_failure": True,
            **tensor_solver_info,
            "max_abs_heat_flux_source_W_m3": float(np.max(np.abs(heat_source))),
            "net_heat_flux_power_W": float(np.sum(heat_source) * grid.cell_volume),
            "raw_min_temperature_before_floor_K": raw_min_temperature,
            "density_gate": density_gate,
            "temperature_floor_contact_count": temperature_floor_contact_count,
        },
    )


def _apply_axis_diffusion(
    *,
    field: np.ndarray,
    diffusivity: np.ndarray,
    dt_s: float,
    spacing_m: float,
    axis: int,
) -> np.ndarray:
    moved_field = np.moveaxis(field, axis, 0)
    moved_diffusivity = np.moveaxis(diffusivity, axis, 0)
    updated = np.empty_like(moved_field)
    for transverse_index in np.ndindex(moved_field.shape[1:]):
        index = (slice(None),) + transverse_index
        updated[index] = diffuse_field_1d_backward_euler(
            moved_field[index].astype(np.float64, copy=False),
            moved_diffusivity[index].astype(np.float64, copy=False),
            float(dt_s),
            float(spacing_m),
        )
    return np.moveaxis(updated, 0, axis)


def _braginskii_heat_source(
    *,
    electron_temperature_K: np.ndarray,
    magnetic_field_T: np.ndarray,
    kappa_parallel_W_m_K: np.ndarray,
    kappa_perpendicular_W_m_K: np.ndarray,
    spacing_m: tuple[float, float, float],
) -> np.ndarray:
    grad_T = np.stack(
        [
            _center_gradient(electron_temperature_K, spacing_m[axis], axis)
            for axis in range(3)
        ],
        axis=-1,
    )
    B_mag = np.sqrt(np.sum(magnetic_field_T * magnetic_field_T, axis=-1))
    b_hat = np.divide(
        magnetic_field_T,
        B_mag[..., np.newaxis],
        out=np.zeros_like(magnetic_field_T),
        where=B_mag[..., np.newaxis] > 0.0,
    )
    b_dot_grad = np.sum(b_hat * grad_T, axis=-1)
    grad_parallel = b_dot_grad[..., np.newaxis] * b_hat
    grad_perpendicular = grad_T - grad_parallel
    heat_flux_positive_to_cold = (
        kappa_parallel_W_m_K[..., np.newaxis] * grad_parallel
        + kappa_perpendicular_W_m_K[..., np.newaxis] * grad_perpendicular
    )
    heat_flux_positive_to_cold = np.where(
        np.isfinite(heat_flux_positive_to_cold),
        heat_flux_positive_to_cold,
        0.0,
    )
    source = np.zeros_like(electron_temperature_K)
    for axis, dx in enumerate(spacing_m):
        source += _face_divergence_zero_boundary(
            heat_flux_positive_to_cold[..., axis],
            dx,
            axis,
        )
    return np.where(np.isfinite(source), source, 0.0)


def _center_gradient(value: np.ndarray, dx: float, axis: int) -> np.ndarray:
    moved = np.moveaxis(value, axis, 0)
    grad = np.zeros_like(moved)
    grad[1:-1] = (moved[2:] - moved[:-2]) / (2.0 * dx)
    if moved.shape[0] > 1:
        grad[0] = (moved[1] - moved[0]) / dx
        grad[-1] = (moved[-1] - moved[-2]) / dx
    return np.moveaxis(grad, 0, axis)


def _face_divergence_zero_boundary(
    centered_flux: np.ndarray,
    dx: float,
    axis: int,
) -> np.ndarray:
    moved = np.moveaxis(centered_flux, axis, 0)
    faces = np.zeros((moved.shape[0] + 1,) + moved.shape[1:], dtype=float)
    faces[1:-1] = 0.5 * (moved[:-1] + moved[1:])
    divergence = (faces[1:] - faces[:-1]) / dx
    return np.moveaxis(divergence, 0, axis)
