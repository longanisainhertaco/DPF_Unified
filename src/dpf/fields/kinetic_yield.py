"""Candidate kinetic ion neutron-yield history for the 3-D hybrid loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dpf.diagnostics.pic_yield import pic_neutron_yield_rate
from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.maxwell_3d import HYBRID_PIC_3D_SOURCE, Maxwell3DGrid


@dataclass(frozen=True)
class KineticYieldTelemetry:
    """One candidate D-D yield-history sample from PIC ion state."""

    status: str
    source: str
    source_lines: str
    dt_s: float
    time_s: float
    neutron_rate_per_s: float
    neutron_increment: float
    cumulative_neutrons: float
    included_species: tuple[str, ...]
    n_macroparticles: int
    mechanism_channels: tuple[str, ...]
    mechanism_separation_status: str
    target_density_min_m3: float
    target_density_max_m3: float
    can_support_first_principles_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KineticIonYieldHistory:
    """Accumulate candidate D-D neutron yield from PIC ion distributions."""

    capability_id = "kinetic_ion_neutron_yield_history"

    def __init__(
        self,
        grid: Maxwell3DGrid,
        *,
        deuterium_species_names: tuple[str, ...] = ("d", "deuterium", "beam_d"),
    ) -> None:
        self.grid = grid
        self.deuterium_species_names = tuple(
            name.strip().lower() for name in deuterium_species_names
        )
        self.cumulative_neutrons = 0.0
        self.time_s = 0.0

    def step(
        self,
        pic: HybridPIC,
        *,
        target_density_m3: np.ndarray,
        dt_s: float,
    ) -> KineticYieldTelemetry:
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        if tuple(pic.grid_shape) != self.grid.shape:
            raise ValueError("PIC grid shape does not match Maxwell grid")
        target = np.asarray(target_density_m3, dtype=float)
        if target.shape != self.grid.shape:
            raise ValueError(
                f"target_density_m3 shape {target.shape} != expected {self.grid.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_density_m3 must be finite")
        if np.any(target < 0.0):
            raise ValueError("target_density_m3 must be non-negative")

        rate = 0.0
        included: list[str] = []
        n_particles = 0
        for species in pic.species:
            if species.name.strip().lower() not in self.deuterium_species_names:
                continue
            if species.n_particles() == 0:
                continue
            included.append(species.name)
            n_particles += species.n_particles()
            rate += float(
                pic_neutron_yield_rate(
                    species.positions,
                    species.velocities,
                    species.weights,
                    target,
                    self.grid.dx,
                    self.grid.dy,
                    self.grid.dz,
                    species.mass,
                )
            )

        increment = rate * dt_s
        self.cumulative_neutrons += increment
        self.time_s += dt_s
        return KineticYieldTelemetry(
            status="candidate_engineering_kinetic_yield_history",
            source=HYBRID_PIC_3D_SOURCE,
            source_lines="952-963, 1083-1089, 1259-1266",
            dt_s=float(dt_s),
            time_s=float(self.time_s),
            neutron_rate_per_s=float(rate),
            neutron_increment=float(increment),
            cumulative_neutrons=float(self.cumulative_neutrons),
            included_species=tuple(included),
            n_macroparticles=int(n_particles),
            mechanism_channels=("dd_particle_distribution_total",),
            mechanism_separation_status="not_mechanism_separated",
            target_density_min_m3=float(np.min(target)),
            target_density_max_m3=float(np.max(target)),
        )


def kinetic_yield_candidate_evidence(
    telemetry: KineticYieldTelemetry,
) -> dict[str, Any]:
    """Build non-promoting evidence for kinetic ion neutron-yield history."""
    return {
        "passed": telemetry.status == "candidate_engineering_kinetic_yield_history",
        "status": "candidate",
        "capability": KineticIonYieldHistory.capability_id,
        "source": telemetry.source,
        "source_lines": telemetry.source_lines,
        "implementation": "src/dpf/fields/kinetic_yield.py",
        "evidence_type": "engineering_pic_ion_yield_history_sample",
        "neutron_rate_per_s": telemetry.neutron_rate_per_s,
        "cumulative_neutrons": telemetry.cumulative_neutrons,
        "included_species": telemetry.included_species,
        "mechanism_channels": telemetry.mechanism_channels,
        "mechanism_separation_status": telemetry.mechanism_separation_status,
        "can_support_first_principles_acceptance": False,
        "limitations": [
            "Uses candidate PIC D-D yield diagnostic without same-scope detector response.",
            "No mechanism-separated validation packet, angular spectrum, or UQ is attached.",
            "Electron-temperature closure sensitivity remains unresolved for yield authority.",
        ],
    }


def kinetic_neutron_yield_authority_status(
    *,
    kinetic_yield_evidence: Mapping[str, Any] | KineticYieldTelemetry | None,
    mechanism_evidence: Mapping[str, Any] | None = None,
    detector_response_evidence: Mapping[str, Any] | None = None,
    uncertainty_evidence: Mapping[str, Any] | None = None,
    temperature_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed authority check for total neutron-yield claims."""
    missing: list[str] = []
    kinetic = _evidence_mapping(kinetic_yield_evidence)
    if not _accepted(kinetic):
        missing.append("accepted_kinetic_yield_history")
    if not _accepted(mechanism_evidence):
        missing.append("mechanism_separated_yield_channels")
    if not _accepted(detector_response_evidence):
        missing.append("same_scope_detector_response")
    if not _accepted(uncertainty_evidence):
        missing.append("yield_uncertainty_budget")

    temp_status = ""
    if temperature_authority is not None:
        temp_status = str(temperature_authority.get("status") or "")
        if (
            temperature_authority.get(
                "can_support_pressure_hall_quantitative_claims"
            )
            is False
        ):
            missing.append("electron_temperature_authority")

    accepted = not missing
    return {
        "status": "accepted" if accepted else "blocked",
        "source": HYBRID_PIC_3D_SOURCE,
        "source_lines": "952-963, 1083-1089, 1259-1266",
        "missing_evidence": sorted(set(missing)),
        "temperature_authority_status": temp_status or "not_attached",
        "can_support_total_yield_acceptance": accepted,
        "can_support_first_principles_acceptance": accepted,
        "validity_note": (
            "The local source treats integral yield as order-of-magnitude and "
            "Te-sensitive; scalar cumulative PIC yield cannot be accepted "
            "without mechanism separation, detector response, and UQ."
        ),
    }


def _evidence_mapping(
    evidence: Mapping[str, Any] | KineticYieldTelemetry | None,
) -> Mapping[str, Any] | None:
    if evidence is None:
        return None
    if isinstance(evidence, KineticYieldTelemetry):
        return evidence.to_dict()
    if isinstance(evidence, Mapping):
        return evidence
    return None


def _accepted(evidence: Mapping[str, Any] | None) -> bool:
    if evidence is None:
        return False
    status = str(evidence.get("status") or "").strip().lower()
    return evidence.get("passed") is True and status in {"accepted", "validated"}
