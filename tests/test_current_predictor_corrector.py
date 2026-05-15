import numpy as np

from dpf.fields.maxwell_3d import Maxwell3DGrid
from dpf.fields.predictor_corrector import (
    CurrentPredictorCorrector,
    predictor_corrector_candidate_evidence,
)
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status


def _grid() -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=(3, 3, 3), spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def _vector(grid: Maxwell3DGrid, values: tuple[float, float, float]) -> np.ndarray:
    out = np.zeros(grid.shape + (3,), dtype=float)
    out[..., 0] = values[0]
    out[..., 1] = values[1]
    out[..., 2] = values[2]
    return out


def test_predictor_corrector_uses_source_linear_current_extrapolation() -> None:
    grid = _grid()
    pc = CurrentPredictorCorrector(grid)
    midpoint = _vector(grid, (4.0, -2.0, 1.0))
    previous = _vector(grid, (1.0, 0.5, -1.0))

    predicted, first_step = pc.predict_end_step_current(midpoint, previous)

    np.testing.assert_allclose(predicted, 2.0 * midpoint - previous)
    assert first_step is False


def test_predictor_corrector_first_step_initializes_from_midpoint_current() -> None:
    grid = _grid()
    pc = CurrentPredictorCorrector(grid)
    midpoint = _vector(grid, (4.0, -2.0, 1.0))

    predicted, first_step = pc.predict_end_step_current(midpoint)

    np.testing.assert_allclose(predicted, midpoint)
    assert first_step is True


def test_predictor_corrector_end_step_ohm_solve_has_small_residual() -> None:
    grid = _grid()
    pc = CurrentPredictorCorrector(grid)
    midpoint = _vector(grid, (2.0e4, 1.0e4, -0.5e4))
    previous = _vector(grid, (1.0e4, -0.5e4, 0.25e4))
    E_next = _vector(grid, (3.0, -1.0, 0.5))
    B_next = _vector(grid, (0.0, 0.0, 0.75))
    Ji_star = _vector(grid, (1.0e3, 2.0e3, 0.0))
    ne = np.full(grid.shape, 1.0e21)

    predicted, corrected, telemetry = pc.correct_end_step_current(
        midpoint_current_A_m2=midpoint,
        previous_current_A_m2=previous,
        electric_field_next_V_m=E_next,
        magnetic_field_next_T=B_next,
        predicted_ion_current_A_m2=Ji_star,
        conductivity_S_m=25.0,
        electron_density_m3=ne,
        include_hall=True,
    )

    np.testing.assert_allclose(predicted, 2.0 * midpoint - previous)
    assert corrected.shape == grid.shape + (3,)
    assert telemetry.status == "candidate_engineering_predictor_corrector"
    assert telemetry.corrected_max_residual_A_m2 < 1.0e-8
    assert telemetry.can_support_first_principles_acceptance is False


def test_predictor_corrector_candidate_evidence_does_not_satisfy_hybrid_gate() -> None:
    grid = _grid()
    pc = CurrentPredictorCorrector(grid)
    zero = np.zeros(grid.shape + (3,), dtype=float)
    E_next = _vector(grid, (1.0, 0.0, 0.0))
    _, _, telemetry = pc.correct_end_step_current(
        midpoint_current_A_m2=zero,
        previous_current_A_m2=None,
        electric_field_next_V_m=E_next,
        magnetic_field_next_T=zero,
        predicted_ion_current_A_m2=zero,
        conductivity_S_m=1.0,
        electron_density_m3=np.full(grid.shape, 1.0e20),
        include_hall=False,
    )
    candidate = predictor_corrector_candidate_evidence(telemetry)

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "current_predictor_corrector": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert "current_predictor_corrector" in status["missing_capabilities"]
