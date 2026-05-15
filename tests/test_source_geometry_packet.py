from dpf.fields.source_geometry import (
    HybridPICSourceGeometry,
    source_geometry_candidate_evidence,
)
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status


def test_source_geometry_packet_records_local_source_values() -> None:
    geometry = HybridPICSourceGeometry()

    assert geometry.source_scope == "llnl_like_180ka_axisymmetric_hybrid_pic"
    assert geometry.coordinate_system == "2d_axisymmetric_rz"
    assert geometry.anode_length_m == 0.05
    assert geometry.anode_radius_m == 0.01
    assert geometry.axial_pml_layers == 20
    assert geometry.background_particles == 500_000
    assert geometry.sheath_particles == 26_060
    assert geometry.can_support_first_principles_acceptance is False


def test_source_geometry_smoke_grid_is_cartesian_engineering_projection() -> None:
    geometry = HybridPICSourceGeometry()

    grid = geometry.smoke_grid(shape=(6, 6, 10))

    assert grid.shape == (6, 6, 10)
    assert grid.dx == grid.dy
    assert grid.dz == geometry.physical_length_m / 10


def test_source_geometry_candidate_evidence_does_not_satisfy_gate() -> None:
    evidence = source_geometry_candidate_evidence(HybridPICSourceGeometry())
    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "same_scope_3d_validation_packet": evidence,
        },
    })

    assert evidence["status"] == "candidate"
    assert evidence["can_support_first_principles_acceptance"] is False
    assert "same_scope_3d_validation_packet" in status["missing_capabilities"]
