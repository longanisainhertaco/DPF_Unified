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
from scipy.constants import Boltzmann as K_B

from dpf.collision.spitzer import braginskii_kappa
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid
from dpf.fluid.two_temperature import (
    electron_energy_from_temperature,
    equilibration_convention_audit,
    step_electron_energy,
    temperature_from_electron_energy,
    two_temperature_model_metadata,
)


@dataclass
class ElectronEnergyState:
    """Electron and ion temperature state for the candidate 3-D loop."""

    electron_energy_J_m3: np.ndarray
    electron_temperature_K: np.ndarray
    ion_temperature_K: np.ndarray

    def electron_pressure_Pa(self, electron_density_m3: np.ndarray) -> np.ndarray:
        return electron_density_m3 * K_B * self.electron_temperature_K


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
    include_equilibration: bool
    include_bremsstrahlung_loss: bool
    include_heat_flux: bool
    heat_flux: dict[str, Any]
    equilibration_audit: dict[str, Any]
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
        J_sq = np.sum(current * current, axis=-1)
        velocity_fluid_layout = np.moveaxis(velocity, -1, 0)
        new_energy, new_Te, new_Ti = step_electron_energy(
            working_energy,
            rho,
            velocity_fluid_layout,
            eta,
            J_sq,
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
        metadata = two_temperature_model_metadata()
        equilibration_audit = equilibration_convention_audit(
            electron_temperature_K=working_Te,
            ion_temperature_K=before_Ti,
            electron_density_m3=ne,
            ion_density_m3=ni,
            Z=charge_state_Z,
        )
        telemetry = ElectronEnergyTelemetry(
            status="candidate_engineering_electron_energy_closure",
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
            include_equilibration=True,
            include_bremsstrahlung_loss=True,
            include_heat_flux=bool(heat_flux_telemetry["applied"]),
            heat_flux=heat_flux_telemetry,
            equilibration_audit=equilibration_audit,
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
        "equilibration_audit": telemetry.equilibration_audit,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Uses the repo two-temperature source terms; Braginskii heat flux is candidate-only and relaxation conventions remain source-audit blocked.",
            "Not yet coupled into the accepted 3-D Yee/PIC pressure-gradient/Hall loop.",
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
                "max_kappa_parallel_W_m_K": max_kappa,
            },
        )

    diffusivity = max_kappa / max(1.5 * min_ne * K_B, 1e-300)
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
                "max_kappa_parallel_W_m_K": max_kappa,
            },
        )
    required_subcycles = max(1, int(np.ceil(dt_s / dt_stable)))
    if required_subcycles > max_subcycles:
        return (
            energy,
            Te,
            {
                **base,
                "status": "blocked_heat_flux_subcycle_limit_exceeded",
                "applied": False,
                "required_subcycles": required_subcycles,
                "subcycles": 0,
                "max_subcycles": max_subcycles,
                "max_kappa_parallel_W_m_K": max_kappa,
                "dt_stable_s": dt_stable,
            },
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
        energy_floor = 1.5 * ne * K_B * temperature_floor_K
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
        },
    )


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
