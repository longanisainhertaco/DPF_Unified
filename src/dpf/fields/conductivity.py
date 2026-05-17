"""Source-derived plasma-vacuum conductivity blending component."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.collision.spitzer import coulomb_log, spitzer_resistivity
from dpf.constants import e, k_B, m_e
from dpf.fields.maxwell_3d import EPSILON_0, HYBRID_PIC_3D_SOURCE, Maxwell3DGrid

NRL_SPITZER_CONDUCTIVITY_SOURCE = (
    "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:2660-2725"
)
NRL_WEAKLY_IONIZED_CONDUCTIVITY_SOURCE = (
    "KnowledgeReference/2019nrlplasma-formulary-037290d4.md:3379-3425"
)
NRL_TYPICAL_ELECTRON_NEUTRAL_CROSS_SECTION_M2 = 5.0e-19


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
    ohmic_cfl_limit_applied: bool = True
    density_blend_applied: bool = True
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartialIonizedConductivityTelemetry:
    """Telemetry for source-backed scalar conductivity from ion and neutral collisions."""

    status: str
    source: str
    weakly_ionized_source: str
    electron_neutral_cross_section_m2: float
    min_electron_density_m3: float
    max_electron_density_m3: float
    min_neutral_density_m3: float
    max_neutral_density_m3: float
    min_sigma_S_m: float
    max_sigma_S_m: float
    max_resistivity_ohm_m: float
    max_electron_ion_resistivity_ohm_m: float
    max_electron_neutral_resistivity_ohm_m: float
    limitations: tuple[str, ...]
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
        apply_density_blend: bool = True,
        apply_ohmic_cfl_limit: bool = True,
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
        transition = (ratio >= 0.1) & (ratio < 1.0)
        plasma = ratio >= 1.0
        if apply_density_blend:
            raw = np.zeros(self.grid.shape, dtype=float)
            raw[transition] = ratio[transition] ** 3 * sigma0[transition]
            raw[plasma] = sigma0[plasma]
        else:
            raw = np.array(sigma0, copy=True, dtype=float)

        sigma_cfl = ohmic_cfl_safety * EPSILON_0 / dt_s
        cfl_limited = raw > sigma_cfl
        effective = (
            np.minimum(raw, sigma_cfl)
            if apply_ohmic_cfl_limit
            else np.array(raw, copy=True, dtype=float)
        )
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
            ohmic_cfl_limit_applied=bool(apply_ohmic_cfl_limit),
            density_blend_applied=bool(apply_density_blend),
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


def partial_ionized_conductivity(
    *,
    electron_density_m3: np.ndarray,
    neutral_density_m3: np.ndarray,
    electron_temperature_K: np.ndarray,
    electron_neutral_cross_section_m2: float = (
        NRL_TYPICAL_ELECTRON_NEUTRAL_CROSS_SECTION_M2
    ),
) -> tuple[np.ndarray, PartialIonizedConductivityTelemetry]:
    """Return scalar conductivity including Spitzer and electron-neutral drag."""
    if electron_neutral_cross_section_m2 <= 0.0:
        raise ValueError("electron_neutral_cross_section_m2 must be positive")
    ne = np.asarray(electron_density_m3, dtype=float)
    nn = np.asarray(neutral_density_m3, dtype=float)
    Te = np.asarray(electron_temperature_K, dtype=float)
    if ne.shape != nn.shape or ne.shape != Te.shape:
        raise ValueError("electron, neutral, and temperature arrays must match")
    if np.any(ne < 0.0) or np.any(nn < 0.0):
        raise ValueError("densities must be non-negative")
    if np.any(Te <= 0.0):
        raise ValueError("electron_temperature_K must be positive")
    ne_safe = np.maximum(ne, 1.0)
    lnL = coulomb_log(ne_safe, Te)
    eta_ei = spitzer_resistivity(ne_safe, Te, lnL=lnL, Z=1.0)
    vte = np.sqrt(k_B * Te / m_e)
    nu_en = nn * float(electron_neutral_cross_section_m2) * vte
    eta_en = m_e * nu_en / (ne_safe * e**2)
    eta_total = eta_ei + eta_en
    sigma = np.divide(
        1.0,
        eta_total,
        out=np.zeros_like(eta_total),
        where=eta_total > 0.0,
    )
    sigma = np.where(ne > 0.0, sigma, 0.0)
    telemetry = PartialIonizedConductivityTelemetry(
        status="candidate_source_backed_partial_ionized_conductivity",
        source=NRL_SPITZER_CONDUCTIVITY_SOURCE,
        weakly_ionized_source=NRL_WEAKLY_IONIZED_CONDUCTIVITY_SOURCE,
        electron_neutral_cross_section_m2=float(electron_neutral_cross_section_m2),
        min_electron_density_m3=float(np.min(ne)),
        max_electron_density_m3=float(np.max(ne)),
        min_neutral_density_m3=float(np.min(nn)),
        max_neutral_density_m3=float(np.max(nn)),
        min_sigma_S_m=float(np.min(sigma)),
        max_sigma_S_m=float(np.max(sigma)),
        max_resistivity_ohm_m=float(np.max(eta_total)),
        max_electron_ion_resistivity_ohm_m=float(np.max(eta_ei)),
        max_electron_neutral_resistivity_ohm_m=float(np.max(eta_en)),
        limitations=(
            "Candidate scalar conductivity; magnetized tensor Pedersen/Hall conductivity is not yet solved as a tensor transport packet.",
            "Electron-neutral cross section uses the NRL typical weakly ionized estimate, not same-gas reviewed cross-section data.",
            "No accepted conductivity sensitivity or nondominance packet is attached.",
        ),
    )
    return sigma, telemetry


def _require_scalar_shape(
    name: str,
    value: np.ndarray,
    expected: tuple[int, int, int],
) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} shape {value.shape} != expected {expected}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
