import numpy as np

from dpf.experimental.pic.hybrid import HybridPIC
from dpf.fields.maxwell_3d import Maxwell3DFieldCore, Maxwell3DGrid
from dpf.fields.pic_coupling import (
    PICCurrentSourcePort,
    pic_current_port_candidate_evidence,
)
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status

M_D = 3.344e-27
Q_E = 1.602e-19


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def test_pic_current_port_maps_hybrid_pic_deposit_to_yee_edges() -> None:
    grid = _grid()
    pic = HybridPIC(
        grid_shape=grid.shape,
        dx=grid.dx,
        dy=grid.dy,
        dz=grid.dz,
        dt=1.0e-12,
        use_esirkepov=False,
    )
    pic.add_species(
        "d",
        M_D,
        Q_E,
        positions=np.array([
            [1.25e-3, 1.5e-3, 1.75e-3],
            [2.25e-3, 2.0e-3, 1.25e-3],
        ]),
        velocities=np.array([
            [1.0e5, 2.0e5, 0.0],
            [-0.5e5, 1.0e5, 0.75e5],
        ]),
        weights=np.array([2.0e8, 3.0e8]),
    )
    rho, Jx, Jy, Jz = pic.deposit()

    edge_current, telemetry = PICCurrentSourcePort(grid).from_cell_centered_current(
        Jx,
        Jy,
        Jz,
        deposition_method="hybrid_pic.deposit:cic_current",
        rho_current=rho,
    )

    assert edge_current.Ex_edge.shape == (4, 5, 5)
    assert edge_current.Ey_edge.shape == (5, 4, 5)
    assert edge_current.Ez_edge.shape == (5, 5, 4)
    assert telemetry.finite is True
    assert telemetry.status == "candidate_engineering_coupling"
    assert telemetry.continuity_status == "blocked_incomplete_continuity_inputs"
    assert np.max(np.abs(edge_current.Ex_edge)) > 0.0


def test_pic_current_port_feeds_ampere_source_term_without_promoting_validation() -> None:
    grid = _grid()
    core = Maxwell3DFieldCore(grid)
    port = PICCurrentSourcePort(grid)
    Jx = np.full(grid.shape, 2.0e4)
    Jy = np.zeros(grid.shape)
    Jz = np.zeros(grid.shape)

    current, telemetry = port.from_cell_centered_current(
        Jx,
        Jy,
        Jz,
        deposition_method="synthetic_uniform_current",
    )
    stepped = core.step_electric(core.empty_state(), dt=1.0e-18, current_density=current)

    assert np.all(stepped.Ex_edge < 0.0)
    assert np.count_nonzero(stepped.Ey_edge) == 0
    assert np.count_nonzero(stepped.Ez_edge) == 0
    assert telemetry.can_support_first_principles_acceptance is False


def test_pic_current_port_reports_continuity_residual_as_measured_not_accepted() -> None:
    grid = _grid()
    port = PICCurrentSourcePort(grid)
    rho0 = np.zeros(grid.shape)
    rho1 = np.zeros(grid.shape)
    Jx = np.ones(grid.shape)
    Jy = np.ones(grid.shape)
    Jz = np.ones(grid.shape)

    _, telemetry = port.from_cell_centered_current(
        Jx,
        Jy,
        Jz,
        deposition_method="synthetic_constant_current",
        rho_previous=rho0,
        rho_current=rho1,
        dt=1.0e-12,
    )

    assert telemetry.continuity_status == "measured_not_accepted"
    assert telemetry.continuity_linf_A_per_m3 == 0.0
    assert telemetry.can_support_first_principles_acceptance is False


def test_pic_current_port_candidate_evidence_does_not_satisfy_hybrid_pic_gate() -> None:
    grid = _grid()
    Jx = np.ones(grid.shape)
    Jy = np.zeros(grid.shape)
    Jz = np.zeros(grid.shape)
    _, telemetry = PICCurrentSourcePort(grid).from_cell_centered_current(
        Jx,
        Jy,
        Jz,
        deposition_method="synthetic_current",
    )
    candidate = pic_current_port_candidate_evidence(telemetry)

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "kinetic_ion_pic_push_deposition": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert status["status"] == "blocked"
    assert "kinetic_ion_pic_push_deposition" in status["missing_capabilities"]
