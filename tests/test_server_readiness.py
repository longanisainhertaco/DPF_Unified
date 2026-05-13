from __future__ import annotations

from dpf.server.readiness import api_readiness_payload
from dpf.validation import (
    FIRST_PRINCIPLES_MHD_MODE,
    PF1000_AKEL_SOURCE_SCOPE,
    PF1000_AKEL_VALIDATION_SCOPE,
)


def test_api_readiness_payload_exposes_fail_closed_authority_and_blockers() -> None:
    payload = api_readiness_payload(backend="mlx")

    assert payload["validation_status"] == "not_evaluated"
    assert payload["result_classification"]["label"] == "Preview"
    assert payload["result_classification"]["can_support_validation_claims"] is False
    assert payload["predictive_readiness"]["ready"] is False
    assert payload["high_fidelity_readiness"]["ready"] is False
    assert payload["digitization_status"]["failed_task_count"] == 1
    assert payload["readiness_scope"]["run_validation_scope"] == "not_declared"
    assert payload["readiness_scope"]["digitization_validation_scope"] == (
        "pf1000_16kv_2021_akel"
    )
    assert payload["readiness_scope"]["digitization_applies_to_run"] is False
    assert payload["readiness_scope"]["source_blocker_scope"] == "global_source_queue"
    assert "akel_2021_fig1_current_waveform_shot_12581" in payload["source_blockers"]
    assert "independent_review_missing" in payload["source_blockers"]


def test_api_readiness_payload_marks_same_scope_digitization_blockers() -> None:
    payload = api_readiness_payload(
        backend="mlx",
        validation_scope="pf1000_16kv_2021_akel",
    )

    assert payload["readiness_scope"]["run_validation_scope"] == "pf1000_16kv_2021_akel"
    assert payload["readiness_scope"]["digitization_applies_to_run"] is True
    assert payload["readiness_scope"]["source_blocker_scope"] == "run_scope"
    assert "Akel digitization blockers apply" in (
        payload["readiness_scope"]["source_blocker_scope_note"]
    )


def test_api_readiness_payload_exports_first_principles_blockers() -> None:
    payload = api_readiness_payload(
        backend="python",
        result={
            "run_mode": FIRST_PRINCIPLES_MHD_MODE,
            "source_scope": PF1000_AKEL_SOURCE_SCOPE,
            "source_scope_status": "same_scope_blocked_by_review",
            "I_MA": [0.0, 0.1],
            "V_kV": [16.0, 15.0],
            "Lp_mhd_nH": [1.0, 1.1],
            "dLp_dt": [0.0, 1.0],
            "phases": ["axial"],
            "rho": [1.0],
            "B": [0.0],
            "Te": [300.0],
            "z_sheath_cm": [0.0],
            "E_cap_kJ": [1.0],
            "E_ind_kJ": [0.0],
            "E_res_kJ": [0.0],
            "circuit_energy_residual_kJ": [0.0],
        },
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
    )

    readiness = payload["first_principles_mhd_readiness"]
    assert readiness["ready"] is False
    assert readiness["status"] == "blocked"
    assert readiness["source_scope"] == PF1000_AKEL_SOURCE_SCOPE
    assert readiness["validation_scope"] == PF1000_AKEL_VALIDATION_SCOPE
    assert "accepted_same_scope_akel_digitization" in readiness["missing_evidence"]
    assert "field_coupled_energy_accounting" in readiness["missing_evidence"]
    assert "first_principles_startup_initialization" in readiness["missing_evidence"]
    assert payload["first_principles_energy_accounting"]["status"] == "incomplete"
    assert payload["first_principles_startup_initialization"]["status"] == "incomplete"
    assert any("blocked_by_review" in item for item in payload["source_blockers"])


def test_simulation_info_includes_api_readiness_fields() -> None:
    from dpf.config import SimulationConfig
    from dpf.server.simulation import SimulationManager

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-7,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
    )
    manager = SimulationManager(config)
    manager.create_engine()

    info = manager.info()

    assert info["validation_status"] == "not_evaluated"
    assert info["result_classification"]["label"] == "Preview"
    assert info["predictive_readiness"]["status"] == "not_predictive_ready"
    assert info["high_fidelity_readiness"]["ready"] is False
    assert info["digitization_status"]["task_count"] == 6
    assert info["readiness_scope"]["run_validation_scope"] == "not_declared"
    assert info["readiness_scope"]["digitization_applies_to_run"] is False
    assert "review_status_not_accepted" in info["source_blockers"]


def test_simulation_info_includes_first_principles_readiness_fields() -> None:
    from dpf.config import SimulationConfig
    from dpf.server.simulation import SimulationManager

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-7,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
    )
    manager = SimulationManager(
        config,
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
        source_scope_status="same_scope_blocked_by_review",
        preset_name="pf1000_akel",
        run_mode=FIRST_PRINCIPLES_MHD_MODE,
    )
    manager.create_engine()

    info = manager.info()

    readiness = info["first_principles_mhd_readiness"]
    assert readiness["run_mode"] == FIRST_PRINCIPLES_MHD_MODE
    assert readiness["source_scope"] == PF1000_AKEL_SOURCE_SCOPE
    assert readiness["validation_scope"] == PF1000_AKEL_VALIDATION_SCOPE
    assert readiness["ready"] is False
    assert "accepted_same_scope_akel_digitization" in readiness["missing_evidence"]


def test_simulation_info_preserves_declared_validation_scope() -> None:
    from dpf.config import SimulationConfig
    from dpf.server.simulation import SimulationManager

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-7,
        circuit={
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
    )
    manager = SimulationManager(
        config,
        validation_scope="pf1000_16kv_2021_akel",
    )
    manager.create_engine()

    info = manager.info()

    assert info["readiness_scope"]["run_validation_scope"] == "pf1000_16kv_2021_akel"
    assert info["readiness_scope"]["digitization_applies_to_run"] is True


def test_rest_simulation_response_includes_readiness_fields() -> None:
    from fastapi.testclient import TestClient

    from dpf.server.app import _simulations, app

    _simulations.clear()
    client = TestClient(app)

    response = client.post("/api/simulations", json={"config": {}, "preset": "tutorial"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["result_classification"]["label"] == "Preview"
    assert payload["predictive_readiness"]["ready"] is False
    assert payload["digitization_status"]["failed_task_count"] == 1
    assert payload["readiness_scope"]["run_validation_scope"] == "not_declared"
    assert payload["readiness_scope"]["digitization_applies_to_run"] is False
    assert "independent_review_missing" in payload["source_blockers"]


def test_rest_simulation_response_exposes_first_principles_preset_scope() -> None:
    from fastapi.testclient import TestClient

    from dpf.server.app import _simulations, app

    _simulations.clear()
    client = TestClient(app)

    response = client.post(
        "/api/simulations",
        json={
            "preset": "pf1000_akel",
            "config": {},
            "run_mode": FIRST_PRINCIPLES_MHD_MODE,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    readiness = payload["first_principles_mhd_readiness"]
    assert readiness["source_scope"] == PF1000_AKEL_SOURCE_SCOPE
    assert readiness["validation_scope"] == PF1000_AKEL_VALIDATION_SCOPE
    assert readiness["status"] == "blocked"
    assert "accepted_same_scope_akel_digitization" in readiness["missing_evidence"]


def test_rest_simulation_response_preserves_declared_validation_scope() -> None:
    from fastapi.testclient import TestClient

    from dpf.server.app import _simulations, app

    _simulations.clear()
    client = TestClient(app)

    response = client.post(
        "/api/simulations",
        json={
            "config": {
                "validation_scope": "pf1000_16kv_2021_akel",
                "grid_shape": [4, 4, 4],
                "dx": 1e-3,
                "sim_time": 1e-7,
                "circuit": {
                    "C": 1e-6,
                    "V0": 1e3,
                    "L0": 1e-7,
                    "R0": 0.01,
                    "anode_radius": 0.005,
                    "cathode_radius": 0.01,
                },
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["readiness_scope"]["run_validation_scope"] == "pf1000_16kv_2021_akel"
    assert payload["readiness_scope"]["digitization_applies_to_run"] is True


def test_rest_simulation_preserves_artifact_classification_config() -> None:
    from fastapi.testclient import TestClient

    from dpf.server.app import _simulations, app

    _simulations.clear()
    client = TestClient(app)

    response = client.post(
        "/api/simulations",
        json={
            "config": {
                "grid_shape": [4, 4, 4],
                "dx": 1e-3,
                "sim_time": 1e-7,
                "circuit": {
                    "C": 1e-6,
                    "V0": 1e3,
                    "L0": 1e-7,
                    "R0": 0.01,
                    "anode_radius": 0.005,
                    "cathode_radius": 0.01,
                },
                "diagnostics": {
                    "hdf5_filename": ":memory:",
                    "artifact_owner": "api-owner",
                    "artifact_classification": "internal",
                    "artifact_distribution": "api-only",
                    "artifact_handling_notes": "api metadata",
                },
            },
        },
    )

    assert response.status_code == 200
    sim_id = response.json()["sim_id"]
    diagnostics = _simulations[sim_id].config.diagnostics
    assert diagnostics.artifact_owner == "api-owner"
    assert diagnostics.artifact_classification == "internal"
    assert diagnostics.artifact_distribution == "api-only"
