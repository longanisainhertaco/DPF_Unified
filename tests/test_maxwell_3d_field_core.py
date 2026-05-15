import numpy as np

from dpf.fields.maxwell_3d import (
    SPEED_OF_LIGHT,
    Maxwell3DBoundaries,
    Maxwell3DFieldCore,
    Maxwell3DGrid,
    maxwell_3d_field_capability_evidence,
)
from dpf.validation.hybrid_pic_3d import (
    HYBRID_PIC_3D_SOURCE,
    hybrid_pic_3d_readiness_status,
)


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(4, 5, 6), spacing=(1.0e-3, 1.2e-3, 1.5e-3))


def test_stable_vacuum_dt_uses_3d_yee_courant_limit() -> None:
    grid = _grid()
    core = Maxwell3DFieldCore(grid)

    dt = core.stable_vacuum_dt(cfl=0.5)

    assert dt > 0.0
    assert dt < min(grid.spacing) / SPEED_OF_LIGHT


def test_zero_field_step_remains_zero_and_reports_diagnostics() -> None:
    core = Maxwell3DFieldCore(_grid())
    state = core.empty_state()
    dt = core.stable_vacuum_dt(cfl=0.2)

    stepped = core.step(state, dt)
    diagnostics = core.diagnostics(stepped)

    assert diagnostics.electric_energy_J == 0.0
    assert diagnostics.magnetic_energy_J == 0.0
    assert diagnostics.total_energy_J == 0.0
    assert diagnostics.max_abs_div_B_T_per_m == 0.0
    assert diagnostics.source == HYBRID_PIC_3D_SOURCE


def test_constrained_transport_magnetic_step_preserves_divergence_free_B() -> None:
    core = Maxwell3DFieldCore(_grid())
    state = core.empty_state()
    rng = np.random.default_rng(1701)
    state.E.Ez_edge[1:-1, 1:-1, :] = rng.normal(scale=2.0e3, size=(3, 4, 6))

    stepped = core.step_magnetic(state, dt=2.0e-12)
    div_B = core.divergence_B(stepped)

    assert float(np.max(np.abs(div_B))) < 1.0e-9


def test_ampere_update_advances_edge_E_from_face_centered_B_curl() -> None:
    core = Maxwell3DFieldCore(_grid())
    state = core.empty_state()
    for j in range(_grid().ny):
        state.B.Bz_face[:, j, :] = j * 2.0e-6

    stepped = core.step_electric(state, dt=1.0e-15)

    assert np.max(np.abs(stepped.Ex_edge[:, 1:-1, 1:-1])) > 0.0
    assert np.max(np.abs(stepped.Ey_edge)) == 0.0
    assert np.max(np.abs(stepped.Ez_edge)) == 0.0


def test_conductor_cells_zero_adjacent_electric_edges() -> None:
    grid = Maxwell3DGrid(shape=(3, 3, 3), spacing=(1.0e-3, 1.0e-3, 1.0e-3))
    core = Maxwell3DFieldCore(
        grid,
        Maxwell3DBoundaries(conductor_cells=np.ones(grid.shape, dtype=bool)),
    )
    state = core.empty_state()
    state.E.Ex_edge.fill(5.0)
    state.E.Ey_edge.fill(6.0)
    state.E.Ez_edge.fill(7.0)

    bounded = core.apply_boundary_conditions(state)

    assert np.count_nonzero(bounded.Ex_edge) == 0
    assert np.count_nonzero(bounded.Ey_edge) == 0
    assert np.count_nonzero(bounded.Ez_edge) == 0


def test_pml_damping_reduces_electric_energy_near_boundaries() -> None:
    grid = Maxwell3DGrid(shape=(4, 4, 4), spacing=(1.0e-3, 1.0e-3, 1.0e-3))
    core = Maxwell3DFieldCore(
        grid,
        Maxwell3DBoundaries(pml_cells=1, pml_strength=0.75),
    )
    state = core.empty_state()
    state.E.Ex_edge.fill(3.0)
    state.E.Ey_edge.fill(3.0)
    state.E.Ez_edge.fill(3.0)
    before = core.electric_energy_J(state)

    after = core.electric_energy_J(core.apply_boundary_conditions(state))

    assert after < before


def test_full_maxwell_component_evidence_does_not_complete_hybrid_pic_gate() -> None:
    evidence = maxwell_3d_field_capability_evidence(
        passed=True,
        test_ids=(
            "tests/test_maxwell_3d_field_core.py::test_zero_field_step_remains_zero_and_reports_diagnostics",
            "tests/test_maxwell_3d_field_core.py::test_constrained_transport_magnetic_step_preserves_divergence_free_B",
        ),
    )

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "full_maxwell_vacuum_plasma_fields": evidence,
        },
    })

    assert evidence["status"] == "accepted"
    assert evidence["source"] == HYBRID_PIC_3D_SOURCE
    assert status["status"] == "blocked"
    assert "full_maxwell_vacuum_plasma_fields" in status["satisfied_capabilities"]
    assert "explicit_3d_geometry" not in status["missing_capabilities"]
    assert "kinetic_ion_pic_push_deposition" in status["missing_capabilities"]
