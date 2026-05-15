"""Candidate electron-energy closure for the 3-D hybrid PIC-fluid path.

The local hybrid PIC-fluid source identifies ``Te = Ti`` as a simplifying
closure and states that pressure-gradient/Hall runs need a separate electron
temperature evolution with collisional coupling and heat-flux models before
they can be quantitative.  This module wraps the repo's existing
two-temperature source-term scaffold behind explicit nonaccepting telemetry for
the 3-D first-principles gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.constants import Boltzmann as K_B

from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid
from dpf.fluid.two_temperature import (
    electron_energy_from_temperature,
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
    ) -> tuple[ElectronEnergyState, ElectronEnergyTelemetry]:
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        if charge_state_Z <= 0.0:
            raise ValueError("charge_state_Z must be positive")
        if temperature_floor_K <= 0.0:
            raise ValueError("temperature_floor_K must be positive")

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
        J_sq = np.sum(current * current, axis=-1)
        velocity_fluid_layout = np.moveaxis(velocity, -1, 0)
        new_energy, new_Te, new_Ti = step_electron_energy(
            energy,
            rho,
            velocity_fluid_layout,
            eta,
            J_sq,
            before_Te,
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
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Uses the repo two-temperature scaffold; heat flux and relaxation conventions remain source-audit blocked.",
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
                "reviewed separate Te equation, heat flux, collisional coupling, "
                "diagnostics, and UQ packet are accepted."
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


def _require_positive(name: str, value: np.ndarray) -> None:
    if np.any(value <= 0.0):
        raise ValueError(f"{name} must be positive")
