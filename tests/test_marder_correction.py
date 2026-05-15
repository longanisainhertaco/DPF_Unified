import numpy as np
import pytest

from dpf.fields.marder import MarderCorrection, marder_candidate_evidence
from dpf.fields.maxwell_3d import Maxwell3DGrid
from dpf.validation.hybrid_pic_3d import hybrid_pic_3d_readiness_status


def _grid(shape: tuple[int, int, int] = (16, 4, 4)) -> Maxwell3DGrid:
    return Maxwell3DGrid(shape=shape, spacing=(1.0e-3, 1.0e-3, 1.0e-3))


def test_marder_correction_reduces_smooth_quasineutral_gauss_residual() -> None:
    grid = _grid()
    marder = MarderCorrection(grid)
    E = np.zeros(grid.shape + (3,), dtype=float)
    x = np.arange(grid.nx, dtype=float) * grid.dx
    E[:, :, :, 0] = np.sin(2.0 * np.pi * x[:, np.newaxis, np.newaxis] / (grid.nx * grid.dx))
    before = marder.gauss_residual(E)

    corrected, telemetry = marder.apply(
        E,
        marder_factor_m2=0.02 * grid.dx**2,
        nondominance_threshold=0.25,
    )
    after = marder.gauss_residual(corrected)

    assert corrected.shape == E.shape
    assert telemetry.status == "candidate_engineering_marder_correction"
    assert telemetry.charge_density_mode == "quasi_neutral_zero_charge"
    assert telemetry.residual_after_linf < telemetry.residual_before_linf
    assert telemetry.relative_correction_linf <= telemetry.nondominance_threshold
    assert telemetry.nondominance_status == "candidate_within_bound"
    assert float(np.max(np.abs(after))) < float(np.max(np.abs(before)))


def test_marder_correction_accepts_explicit_charge_density_shape() -> None:
    grid = _grid(shape=(4, 4, 4))
    marder = MarderCorrection(grid)
    E = np.zeros(grid.shape + (3,), dtype=float)
    rho = np.full(grid.shape, 1.0e-8)

    _, telemetry = marder.apply(
        E,
        charge_density_C_m3=rho,
        marder_factor_m2=0.0,
    )

    assert telemetry.charge_density_mode == "explicit_charge_density"
    assert telemetry.residual_before_linf > 0.0
    assert telemetry.residual_after_linf == telemetry.residual_before_linf


def test_marder_correction_rejects_negative_factor() -> None:
    grid = _grid(shape=(4, 4, 4))
    marder = MarderCorrection(grid)
    E = np.zeros(grid.shape + (3,), dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        marder.apply(E, marder_factor_m2=-1.0e-8)


def test_marder_correction_flags_dominant_candidate_correction() -> None:
    grid = _grid()
    marder = MarderCorrection(grid)
    E = np.zeros(grid.shape + (3,), dtype=float)
    x = np.arange(grid.nx, dtype=float) * grid.dx
    E[:, :, :, 0] = np.sin(
        2.0 * np.pi * x[:, np.newaxis, np.newaxis] / (grid.nx * grid.dx)
    )

    _, telemetry = marder.apply(
        E,
        marder_factor_m2=0.5 * grid.dx**2,
        nondominance_threshold=0.01,
    )

    assert telemetry.relative_correction_linf > telemetry.nondominance_threshold
    assert telemetry.nondominance_status == "candidate_dominant_correction"


def test_marder_candidate_evidence_does_not_satisfy_hybrid_gate() -> None:
    grid = _grid(shape=(4, 4, 4))
    marder = MarderCorrection(grid)
    E = np.zeros(grid.shape + (3,), dtype=float)
    _, telemetry = marder.apply(E, marder_factor_m2=0.0)
    candidate = marder_candidate_evidence(telemetry)

    status = hybrid_pic_3d_readiness_status({
        "geometry_dimensionality": "3d",
        "hybrid_pic_3d_evidence": {
            "gauss_law_or_marder_control": candidate,
        },
    })

    assert candidate["status"] == "candidate"
    assert candidate["can_support_first_principles_acceptance"] is False
    assert candidate["nondominance_status"] == "not_evaluated"
    assert "gauss_law_or_marder_control" in status["missing_capabilities"]
