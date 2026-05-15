"""Source-derived plasma-vacuum conductivity blending component."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.fields.maxwell_3d import EPSILON_0, HYBRID_PIC_3D_SOURCE, Maxwell3DGrid


@dataclass(frozen=True)
class ConductivityBlendTelemetry:
    """Telemetry for plasma-vacuum conductivity blending and Ohmic CFL limiting."""

    status: str
    source: str
    background_density_m3: float
    ohmic_cfl_safety: float
    sigma_cfl_S_m: float
    max_sigma_raw_S_m: float
    max_sigma_effective_S_m: float
    vacuum_fraction: float
    transition_fraction: float
    plasma_fraction: float
    cfl_limited_fraction: float
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlasmaVacuumConductivityBlend:
    """Apply the source conductivity transition and explicit Ohmic CFL cap."""

    capability_id = "plasma_vacuum_conductivity_blending"

    def __init__(self, grid: Maxwell3DGrid) -> None:
        self.grid = grid

    def effective_conductivity(
        self,
        *,
        sigma0_S_m: np.ndarray | float,
        electron_density_m3: np.ndarray,
        background_density_m3: float,
        dt_s: float,
        ohmic_cfl_safety: float,
    ) -> tuple[np.ndarray, ConductivityBlendTelemetry]:
        if background_density_m3 <= 0.0:
            raise ValueError("background_density_m3 must be positive")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if ohmic_cfl_safety <= 0.0:
            raise ValueError("ohmic_cfl_safety must be positive")

        ne = np.asarray(electron_density_m3, dtype=float)
        _require_scalar_shape("electron_density_m3", ne, self.grid.shape)
        if np.any(ne < 0.0):
            raise ValueError("electron_density_m3 must be non-negative")
        sigma0 = np.asarray(sigma0_S_m, dtype=float)
        if sigma0.shape == ():
            sigma0 = np.full(self.grid.shape, float(sigma0), dtype=float)
        _require_scalar_shape("sigma0_S_m", sigma0, self.grid.shape)
        if np.any(sigma0 < 0.0):
            raise ValueError("sigma0_S_m must be non-negative")

        ratio = ne / background_density_m3
        raw = np.zeros(self.grid.shape, dtype=float)
        transition = (ratio >= 0.1) & (ratio < 1.0)
        plasma = ratio >= 1.0
        raw[transition] = ratio[transition] ** 3 * sigma0[transition]
        raw[plasma] = sigma0[plasma]

        sigma_cfl = ohmic_cfl_safety * EPSILON_0 / dt_s
        effective = np.minimum(raw, sigma_cfl)
        cfl_limited = raw > sigma_cfl
        total = raw.size
        telemetry = ConductivityBlendTelemetry(
            status="candidate_engineering_conductivity_blend",
            source=HYBRID_PIC_3D_SOURCE,
            background_density_m3=float(background_density_m3),
            ohmic_cfl_safety=float(ohmic_cfl_safety),
            sigma_cfl_S_m=float(sigma_cfl),
            max_sigma_raw_S_m=float(np.max(raw)),
            max_sigma_effective_S_m=float(np.max(effective)),
            vacuum_fraction=float(np.count_nonzero(ratio < 0.1) / total),
            transition_fraction=float(np.count_nonzero(transition) / total),
            plasma_fraction=float(np.count_nonzero(plasma) / total),
            cfl_limited_fraction=float(np.count_nonzero(cfl_limited) / total),
        )
        return effective, telemetry


def conductivity_blend_candidate_evidence(
    telemetry: ConductivityBlendTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for conductivity blending."""
    return {
        "passed": telemetry.status == "candidate_engineering_conductivity_blend",
        "status": "candidate",
        "capability": PlasmaVacuumConductivityBlend.capability_id,
        "source": telemetry.source,
        "implementation": "src/dpf/fields/conductivity.py",
        "evidence_type": "engineering_plasma_vacuum_conductivity_component",
        "sigma_cfl_S_m": telemetry.sigma_cfl_S_m,
        "cfl_limited_fraction": telemetry.cfl_limited_fraction,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Component is only integrated into candidate Maxwell/Ohm steps.",
            "No accepted DPF sensitivity packet shows the limiter is weakly active.",
            "Same-scope 3-D validation is not supplied.",
        ],
    }


def _require_scalar_shape(
    name: str,
    value: np.ndarray,
    expected: tuple[int, int, int],
) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} shape {value.shape} != expected {expected}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
