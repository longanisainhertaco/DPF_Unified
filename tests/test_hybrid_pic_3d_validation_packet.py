from dpf.fields.source_geometry import HybridPICSourceGeometry
from dpf.validation.hybrid_pic_3d import HYBRID_PIC_3D_CAPABILITY_IDS
from dpf.validation.hybrid_pic_3d_validation_packet import (
    HybridPIC3DValidationPacket,
    candidate_packet_from_source_geometry,
    evaluate_hybrid_pic_3d_validation_packet,
)


def _accepted_evidence() -> dict[str, object]:
    return {
        "passed": True,
        "status": "accepted",
        "source": "unit.test",
    }


def test_source_geometry_validation_packet_remains_blocked() -> None:
    import dpf.validation as validation

    packet = candidate_packet_from_source_geometry(HybridPICSourceGeometry())

    status = evaluate_hybrid_pic_3d_validation_packet(packet)

    assert "HybridPIC3DValidationPacket" in validation.__all__
    assert "evaluate_hybrid_pic_3d_validation_packet" in validation.__all__
    assert packet.geometry_dimensionality == "2d_axisymmetric_rz"
    assert status["status"] == "blocked"
    assert status["can_support_first_principles_acceptance"] is False
    assert "accepted_hybrid_pic_3d_capabilities" in status["missing_evidence"]
    assert "accepted_same_scope_targets" in status["missing_evidence"]
    assert "accepted_detector_response" in status["missing_evidence"]
    assert "accepted_uncertainty_budget" in status["missing_evidence"]
    assert "explicit_3d_geometry" in (
        status["hybrid_pic_3d_readiness"]["missing_capabilities"]
    )


def test_validation_packet_accepts_only_complete_same_scope_evidence() -> None:
    capability_evidence = {
        capability_id: _accepted_evidence()
        for capability_id in HYBRID_PIC_3D_CAPABILITY_IDS
    }
    packet = HybridPIC3DValidationPacket(
        geometry_dimensionality="3d",
        source_scope="unit_same_scope_3d",
        hybrid_pic_3d_evidence=capability_evidence,
        same_scope_target_evidence=_accepted_evidence(),
        detector_response_evidence=_accepted_evidence(),
        uncertainty_evidence=_accepted_evidence(),
        conservation_evidence=_accepted_evidence(),
        nondominance_evidence=_accepted_evidence(),
        backend_evidence=_accepted_evidence(),
    )

    status = evaluate_hybrid_pic_3d_validation_packet(packet)

    assert status["status"] == "accepted"
    assert status["missing_evidence"] == []
    assert status["can_support_first_principles_acceptance"] is True


def test_validation_packet_blocks_when_extra_packets_are_missing() -> None:
    capability_evidence = {
        capability_id: _accepted_evidence()
        for capability_id in HYBRID_PIC_3D_CAPABILITY_IDS
    }
    packet = HybridPIC3DValidationPacket(
        geometry_dimensionality="3d",
        source_scope="unit_same_scope_3d",
        hybrid_pic_3d_evidence=capability_evidence,
        same_scope_target_evidence=_accepted_evidence(),
    )

    status = evaluate_hybrid_pic_3d_validation_packet(packet)

    assert status["status"] == "blocked"
    assert "accepted_hybrid_pic_3d_capabilities" not in status["missing_evidence"]
    assert "accepted_detector_response" in status["missing_evidence"]
    assert "accepted_uncertainty_budget" in status["missing_evidence"]
    assert "accepted_conservation_packet" in status["missing_evidence"]
    assert "accepted_nondominance_packet" in status["missing_evidence"]
    assert "accepted_backend_scaling_packet" in status["missing_evidence"]
