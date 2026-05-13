from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.engine.backend_capabilities import backend_feature_diagnostics
from dpf.engine.backend_dispatch import backend_authority_labels


def _config(
    *,
    fluid: dict | None = None,
    radiation: dict | None = None,
    **overrides: object,
) -> SimulationConfig:
    return SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-2,
        sim_time=1e-9,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        fluid=fluid or {},
        radiation=radiation or {},
        diagnostics={"hdf5_filename": ":memory:"},
        **overrides,
    )


def test_backend_feature_diagnostics_warn_for_athenak_skipped_physics() -> None:
    config = _config(
        fluid={
            "backend": "athenak",
            "enable_viscosity": True,
            "enable_nernst": True,
            "enable_anisotropic_conduction": True,
            "diffusion_method": "sts",
        },
        radiation={"bremsstrahlung_enabled": True},
    )

    diagnostics = backend_feature_diagnostics(config, "athenak")

    features = {diagnostic.feature for diagnostic in diagnostics}
    severities = {diagnostic.severity for diagnostic in diagnostics}
    assert "Braginskii viscosity" in features
    assert "Nernst effect" in features
    assert "anisotropic thermal conduction" in features
    assert "radiation transport (bremsstrahlung/line)" in features
    assert "RKL2 super time-stepping" in features
    assert severities == {"warning"}


def test_backend_feature_diagnostics_note_gpu_diffusion_fallback() -> None:
    config = _config(fluid={"backend": "mlx", "diffusion_method": "implicit"})

    diagnostics = backend_feature_diagnostics(config, "mlx")
    diffusion = next(
        item for item in diagnostics
        if item.feature == "implicit diffusion"
    )

    assert diffusion.severity == "info"
    assert diffusion.behavior == "explicit_fallback"
    assert "implicit" in diffusion.feature


def test_backend_feature_diagnostics_report_gpu_operator_ownership() -> None:
    config = _config(
        fluid={
            "backend": "mlx",
            "enable_nernst": True,
            "enable_anisotropic_conduction": True,
            "enable_viscosity": True,
        },
        radiation={
            "bremsstrahlung_enabled": True,
            "line_radiation_enabled": True,
        },
    )

    diagnostics = backend_feature_diagnostics(config, "mlx")
    ownership = {item.feature: item.behavior for item in diagnostics}

    assert ownership["Nernst effect"] == "backend_owned"
    assert ownership["anisotropic thermal conduction"] == "backend_owned"
    assert ownership["Braginskii viscosity"] == "backend_owned"
    assert ownership["bremsstrahlung radiation"] == "backend_owned"
    assert ownership["line radiation"] == "python_operator_owned"


def test_backend_authority_labels_do_not_treat_tier_as_validation() -> None:
    labels = backend_authority_labels("mlx")

    assert labels["implementation_tier"] == "production"
    assert labels["validation_status"] == "not_validation_evidence"
    assert "does not certify scientific readiness" in labels["validation_note"]


def test_engine_summary_exposes_backend_authority_labels() -> None:
    config = _config()
    engine = SimulationEngine(config)

    summary = engine.run(max_steps=1)

    assert summary["backend"] == "python"
    assert summary["backend_implementation_tier"] == "teaching"
    assert summary["backend_validation_status"] == "not_validation_evidence"
    assert summary["backend_authority"]["implementation_tier"] == "teaching"
    assert summary["nonfinite_state_evidence"]["classification"] == "engineering_probe"
    assert summary["nonfinite_state_evidence"]["first_event"] is None
    assert summary["result_classification"]["label"] == "Preview"


def test_engine_summary_marks_breakdown_config_as_not_applied() -> None:
    config = _config()
    assert config.breakdown.enabled is True

    engine = SimulationEngine(config)
    summary = engine.run(max_steps=1)

    breakdown = summary["breakdown_authority"]
    assert breakdown["enabled"] is True
    assert breakdown["status"] == "config_only_not_applied"
    assert breakdown["applied_to_initial_state"] is False
    assert breakdown["validation_status"] == "not_validation_evidence"


def test_engine_records_first_nonfinite_event_before_repair() -> None:
    config = _config(nonfinite_event_history_limit=2)
    engine = SimulationEngine(config)
    engine.state["pressure"][0, 0, 0] = np.nan

    repaired = engine._sanitize_state("unit test")

    evidence = engine.nonfinite_state_evidence
    first_event = evidence["first_event"]
    assert repaired == 1
    assert evidence["classification"] == "engineering_probe"
    assert evidence["cumulative_repairs"] == 1
    assert first_event["label"] == "unit test"
    assert first_event["fields"][0]["field"] == "pressure"
    assert first_event["fields"][0]["first_index"] == [0, 0, 0]
    assert first_event["fields"][0]["first_value"] == "nan"
    assert np.isfinite(engine.state["pressure"][0, 0, 0])


def test_engine_fail_fast_preserves_first_nonfinite_state() -> None:
    config = _config(fail_fast_on_nonfinite=True)
    engine = SimulationEngine(config)
    engine.state["rho"][1, 1, 1] = np.inf

    with pytest.raises(RuntimeError, match="Non-finite state detected before repair"):
        engine._sanitize_state("probe")

    evidence = engine.nonfinite_state_evidence
    assert evidence["fail_fast_on_nonfinite"] is True
    assert evidence["first_event"]["fields"][0]["field"] == "rho"
    assert evidence["first_event"]["fields"][0]["first_value"] == "inf"
    assert np.isinf(engine.state["rho"][1, 1, 1])


def test_gpu_owned_nernst_is_not_applied_by_python_operator() -> None:
    from dpf.core.bases import CouplingState

    config = _config(fluid={"enable_nernst": True})
    engine = SimulationEngine(config)
    engine.backend = "mlx"

    with patch.object(engine, "_apply_nernst") as apply_nernst:
        engine._step_post_fluid_corrections(1e-12, 1.0, CouplingState())

    apply_nernst.assert_not_called()


def test_gpu_diffusion_fallback_is_not_applied_by_python_operator() -> None:
    config = _config(fluid={"diffusion_method": "implicit", "enable_resistive": True})
    engine = SimulationEngine(config)
    engine.backend = "mlx"

    with patch.object(engine, "_apply_diffusion") as apply_diffusion:
        engine._apply_collision_radiation(1e-12, 1.0)

    apply_diffusion.assert_not_called()


def test_mlx_engine_passes_requested_transport_flags() -> None:
    config = _config(
        fluid={
            "backend": "mlx",
            "enable_hall": True,
            "enable_anisotropic_conduction": True,
            "enable_viscosity": True,
            "enable_nernst": True,
            "precision": "float64",
        }
    )

    with (
        patch("dpf.metal.mlx_solver.MLXMHDSolver.is_available", return_value=True),
        patch("dpf.metal.mlx_solver.MLXMHDSolver.__init__", return_value=None) as init,
    ):
        engine = SimulationEngine(config)

    kwargs = init.call_args.kwargs
    assert engine.backend == "mlx"
    assert kwargs["enable_hall"] is True
    assert kwargs["enable_braginskii_conduction"] is True
    assert kwargs["enable_braginskii_viscosity"] is True
    assert kwargs["enable_nernst"] is True
    assert kwargs["precision"] == "float64"
