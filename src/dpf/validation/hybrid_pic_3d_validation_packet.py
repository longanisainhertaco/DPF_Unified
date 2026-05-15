"""Same-scope validation packet gate for the 3-D hybrid PIC-fluid core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from dpf.validation.hybrid_pic_3d import (
    HYBRID_PIC_3D_SOURCE,
    hybrid_pic_3d_readiness_status,
)


@dataclass(frozen=True)
class HybridPIC3DValidationPacket:
    """Evidence packet required before a 3-D hybrid run can be accepted."""

    geometry_dimensionality: str
    source_scope: str
    hybrid_pic_3d_evidence: Mapping[str, object] = field(default_factory=dict)
    same_scope_target_evidence: Mapping[str, object] | None = None
    detector_response_evidence: Mapping[str, object] | None = None
    uncertainty_evidence: Mapping[str, object] | None = None
    conservation_evidence: Mapping[str, object] | None = None
    nondominance_evidence: Mapping[str, object] | None = None
    backend_evidence: Mapping[str, object] | None = None
    source: str = HYBRID_PIC_3D_SOURCE
    source_lines: str = "942-951, 974-991, 1215-1225, 1259-1266"

    def readiness_input(self) -> dict[str, object]:
        return {
            "geometry_dimensionality": self.geometry_dimensionality,
            "hybrid_pic_3d_evidence": dict(self.hybrid_pic_3d_evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_hybrid_pic_3d_validation_packet(
    packet: HybridPIC3DValidationPacket,
) -> dict[str, Any]:
    """Evaluate same-scope first-principles validation readiness."""
    capability_status = hybrid_pic_3d_readiness_status(packet.readiness_input())
    missing: list[str] = []
    if capability_status["can_support_first_principles_acceptance"] is not True:
        missing.append("accepted_hybrid_pic_3d_capabilities")
    evidence_requirements = {
        "accepted_same_scope_targets": packet.same_scope_target_evidence,
        "accepted_detector_response": packet.detector_response_evidence,
        "accepted_uncertainty_budget": packet.uncertainty_evidence,
        "accepted_conservation_packet": packet.conservation_evidence,
        "accepted_nondominance_packet": packet.nondominance_evidence,
        "accepted_backend_scaling_packet": packet.backend_evidence,
    }
    for requirement, evidence in evidence_requirements.items():
        if not _accepted(evidence):
            missing.append(requirement)

    accepted = not missing
    return {
        "status": "accepted" if accepted else "blocked",
        "source": packet.source,
        "source_lines": packet.source_lines,
        "source_scope": packet.source_scope,
        "geometry_dimensionality": packet.geometry_dimensionality,
        "missing_evidence": sorted(set(missing)),
        "hybrid_pic_3d_readiness": capability_status,
        "can_support_first_principles_acceptance": accepted,
        "validity_note": (
            "A true first-principles 3-D DPF claim requires accepted capability "
            "evidence plus same-scope targets, detector response, UQ, "
            "conservation/nondominance, and backend-scaling packets."
        ),
    }


def candidate_packet_from_source_geometry(
    geometry: object,
) -> HybridPIC3DValidationPacket:
    """Build a blocked packet from a source-geometry object or mapping."""
    if isinstance(geometry, Mapping):
        source_scope = str(geometry.get("source_scope") or "unknown")
        dimensionality = str(geometry.get("coordinate_system") or "unset")
    else:
        source_scope = str(getattr(geometry, "source_scope", "unknown"))
        dimensionality = str(getattr(geometry, "coordinate_system", "unset"))
    return HybridPIC3DValidationPacket(
        geometry_dimensionality=dimensionality,
        source_scope=source_scope,
    )


def _accepted(evidence: Mapping[str, object] | None) -> bool:
    if evidence is None:
        return False
    status = str(evidence.get("status") or "").strip().lower()
    return evidence.get("passed") is True and status in {"accepted", "validated"}
