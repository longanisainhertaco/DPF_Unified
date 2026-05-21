from __future__ import annotations

import json

from dpf.first_principles.deck import PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE
from dpf.first_principles.runtime_demonstrator_scope import SELECTED_SCOPE_LABEL
from dpf.server.readiness import api_readiness_payload
from dpf.validation.first_principles_mhd import (
    PF1000_AKEL_SOURCE_SCOPE,
    PF1000_AKEL_VALIDATION_SCOPE,
)

FIRST_PRINCIPLES_MHD_MODE = "first_principles_mhd"

# PF-1000 Scholz 2000/2001 24-rod large-electrode full-energy scope labels.
PF1000_FULL_ENERGY_VALIDATION_SCOPE = SELECTED_SCOPE_LABEL
PF1000_FULL_ENERGY_SOURCE_SCOPE = PF1000_SCHOLZ_2001_24ROD_SOURCE_SCOPE


def _first_principles_payload(
    *,
    validation_scope: str,
    source_scope: str,
) -> dict[str, object]:
    """Build a first-principles api_readiness_payload for a declared scope."""

    return api_readiness_payload(
        backend=FIRST_PRINCIPLES_MHD_MODE,
        result={
            "run_mode": FIRST_PRINCIPLES_MHD_MODE,
            "validation_scope": validation_scope,
            "source_scope": source_scope,
        },
        validation_scope=validation_scope,
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
    assert readiness["package_native_runner"] == (
        "first_principles_3d_hybrid_em_pic_fluid"
    )
    assert "same_scope_source" in readiness["missing_evidence"]
    assert "startup_bvp" in readiness["missing_evidence"]
    assert payload["first_principles_energy_accounting"]["status"] == (
        "engineering_candidate_conservation_telemetry_not_validation"
    )
    assert payload["first_principles_startup_initialization"]["status"] == (
        "rejected_startup_mode_for_first_principles"
    )
    assert payload["first_principles_neutron_yield_authority"]["status"] == (
        "blocked_mechanism_separated_neutron_authority_not_available"
    )
    assert any(
        "same-scope source-truth acceptance packet" in item
        for item in payload["source_blockers"]
    )


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
    assert "same_scope_source" in readiness["missing_evidence"]
    assert readiness["execution_mode"] == "first_principles_3d_hybrid_em_pic_fluid"


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
    assert "same_scope_source" in readiness["missing_evidence"]
    assert readiness["package_native_runner"] == (
        "first_principles_3d_hybrid_em_pic_fluid"
    )


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


# --------------------------------------------------------------------------
# SS11-2 / S10-A2: scope-safe server readiness.
#
# Closes the Akel/full-energy mixing path where the readiness layer always ran
# the Akel 16 kV seed-layer deck and then stamped the caller-supplied scope.
# --------------------------------------------------------------------------


def test_full_energy_readiness_uses_full_energy_deck_not_akel() -> None:
    """A PF-1000 full-energy request must run the full-energy deck.

    The readiness payload must contain no Akel startup mode, no Akel source
    scope, and no Akel validation-packet blocker text.
    """

    payload = _first_principles_payload(
        validation_scope=PF1000_FULL_ENERGY_VALIDATION_SCOPE,
        source_scope=PF1000_FULL_ENERGY_SOURCE_SCOPE,
    )
    readiness = payload["first_principles_mhd_readiness"]

    assert readiness["ready"] is False
    assert readiness["status"] == "blocked"
    assert readiness["can_support_first_principles_acceptance"] is False
    # The runtime deck is the Scholz 24-rod full-energy deck, NOT the Akel deck.
    assert "scholz" in readiness["runtime_deck_id"]
    assert "akel" not in readiness["runtime_deck_id"].lower()
    assert (
        readiness["actual_runtime_validation_scope"]
        == PF1000_FULL_ENERGY_VALIDATION_SCOPE
    )
    assert (
        readiness["actual_runtime_source_scope"]
        == PF1000_FULL_ENERGY_SOURCE_SCOPE
    )

    # No Akel source scope on the readiness payload.
    assert "akel" not in str(readiness["source_scope"]).lower()
    assert "akel" not in str(readiness["validation_scope"]).lower()
    assert (
        "akel"
        not in str(readiness["runtime_deck_internal_source_scope"]).lower()
    )

    # No Akel startup mode: the Akel seed-layer startup telemetry carries an
    # Akel-tagged source scope; the full-energy deck's startup must not.
    startup = payload["first_principles_startup_initialization"]
    assert "akel" not in json.dumps(startup).lower()

    # No Akel validation-packet blocker text.
    assert not any("akel" in str(item).lower() for item in readiness["blockers"])


def test_akel_readiness_uses_akel_deck() -> None:
    """A genuine Akel 16 kV request must still run the Akel deck."""

    payload = _first_principles_payload(
        validation_scope=PF1000_AKEL_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
    )
    readiness = payload["first_principles_mhd_readiness"]

    assert readiness["ready"] is False
    assert readiness["status"] == "blocked"
    # The runtime deck is the Akel 16 kV engineering deck.
    assert "akel" in readiness["runtime_deck_id"].lower()
    assert "scholz" not in readiness["runtime_deck_id"].lower()
    assert (
        readiness["actual_runtime_validation_scope"]
        == PF1000_AKEL_VALIDATION_SCOPE
    )
    assert readiness["actual_runtime_source_scope"] == PF1000_AKEL_SOURCE_SCOPE
    # The Akel deck's startup telemetry carries the Akel seed-layer scope.
    startup = payload["first_principles_startup_initialization"]
    assert "akel" in json.dumps(startup).lower()


def test_readiness_payload_exposes_requested_and_actual_runtime_scope() -> None:
    """The readiness payload must expose BOTH requested and actual scope.

    A reviewer must always be able to see which deck was actually executed
    versus what the caller requested.
    """

    for validation_scope, source_scope in (
        (PF1000_AKEL_VALIDATION_SCOPE, PF1000_AKEL_SOURCE_SCOPE),
        (
            PF1000_FULL_ENERGY_VALIDATION_SCOPE,
            PF1000_FULL_ENERGY_SOURCE_SCOPE,
        ),
    ):
        payload = _first_principles_payload(
            validation_scope=validation_scope,
            source_scope=source_scope,
        )
        readiness = payload["first_principles_mhd_readiness"]

        # Requested scope is echoed verbatim.
        assert readiness["requested_validation_scope"] == validation_scope
        assert readiness["requested_source_scope"] == source_scope
        # Actual runtime scope is reported alongside it.
        assert "actual_runtime_validation_scope" in readiness
        assert "actual_runtime_source_scope" in readiness
        assert "runtime_deck_id" in readiness
        # For a matched request the runtime family equals the requested family.
        assert readiness["scope_match"] is True
        assert (
            readiness["actual_runtime_validation_scope"] == validation_scope
        )


def test_readiness_scope_mismatch_is_blocked_and_never_ready() -> None:
    """A requested/runtime scope mismatch must fail closed.

    A request that names a full-energy validation scope but an Akel source
    scope is internally contradictory: readiness must report it as blocked,
    must not run any deck, and must not present it as ready.
    """

    payload = _first_principles_payload(
        validation_scope=PF1000_FULL_ENERGY_VALIDATION_SCOPE,
        source_scope=PF1000_AKEL_SOURCE_SCOPE,
    )
    readiness = payload["first_principles_mhd_readiness"]

    assert readiness["ready"] is False
    assert readiness["status"] == "blocked"
    assert readiness["scope_match"] is False
    assert readiness["can_support_first_principles_acceptance"] is False
    # No deck was executed: a contradictory request cannot ride any deck.
    assert readiness["runtime_deck_id"] == "not_run"
    assert readiness["actual_runtime_validation_scope"] == "not_run"
    assert readiness["actual_runtime_source_scope"] == "not_run"
    # The mismatch is named in the blockers.
    assert any(
        "does not resolve to a known runtime deck" in str(item)
        for item in readiness["blockers"]
    )
    # No deck telemetry leaked into the energy/startup/neutron payloads.
    assert payload["first_principles_energy_accounting"] == {}
    assert payload["first_principles_startup_initialization"] == {}
    assert payload["first_principles_neutron_yield_authority"] == {}


def test_undeclared_scope_first_principles_readiness_fails_closed() -> None:
    """A not_declared / unknown scope must fail closed, never default a deck.

    An undeclared scope must not silently default to the Akel deck (or any
    deck): readiness must be blocked with no runtime deck executed.
    """

    # No validation_scope declared at all.
    payload = api_readiness_payload(backend=FIRST_PRINCIPLES_MHD_MODE)
    readiness = payload["first_principles_mhd_readiness"]

    assert readiness["ready"] is False
    assert readiness["status"] == "blocked"
    assert readiness["scope_match"] is False
    assert readiness["runtime_deck_id"] == "not_run"
    assert readiness["can_support_first_principles_acceptance"] is False
    # No Akel deck was run: no Akel content in the first-principles runtime
    # sections.  (The global Akel digitization-source queue is a separate,
    # always-present part of the payload and is intentionally excluded here.)
    first_principles_sections = {
        "first_principles_mhd_readiness": readiness,
        "first_principles_energy_accounting": (
            payload["first_principles_energy_accounting"]
        ),
        "first_principles_startup_initialization": (
            payload["first_principles_startup_initialization"]
        ),
        "first_principles_neutron_yield_authority": (
            payload["first_principles_neutron_yield_authority"]
        ),
    }
    assert "akel" not in json.dumps(first_principles_sections).lower()

    # An unknown/unsupported scope label also fails closed.
    unknown = _first_principles_payload(
        validation_scope="some_unsupported_scope",
        source_scope="some_unsupported_source",
    )
    unknown_readiness = unknown["first_principles_mhd_readiness"]
    assert unknown_readiness["ready"] is False
    assert unknown_readiness["status"] == "blocked"
    assert unknown_readiness["runtime_deck_id"] == "not_run"


def test_readiness_never_promotes_first_principles_acceptance() -> None:
    """No readiness path may promote a first-principles acceptance flag."""

    for validation_scope, source_scope in (
        (PF1000_AKEL_VALIDATION_SCOPE, PF1000_AKEL_SOURCE_SCOPE),
        (
            PF1000_FULL_ENERGY_VALIDATION_SCOPE,
            PF1000_FULL_ENERGY_SOURCE_SCOPE,
        ),
        (PF1000_FULL_ENERGY_VALIDATION_SCOPE, PF1000_AKEL_SOURCE_SCOPE),
        ("not_declared", "not_declared"),
    ):
        payload = _first_principles_payload(
            validation_scope=validation_scope,
            source_scope=source_scope,
        )
        readiness = payload["first_principles_mhd_readiness"]
        assert readiness["ready"] is False
        assert readiness["status"] == "blocked"
        assert readiness["can_support_first_principles_acceptance"] is False
        blob = json.dumps(payload).lower()
        assert '"accepted_runtime_claim": true' not in blob
        assert '"promotes_acceptance": true' not in blob
        assert '"can_support_first_principles_acceptance": true' not in blob
        assert '"can_support_first_principles_acceptance":true' not in blob
