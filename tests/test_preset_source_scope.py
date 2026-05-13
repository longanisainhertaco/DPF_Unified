"""Preset source-scope labels for product/API guardrails."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dpf.presets import (
    get_preset,
    list_presets,
    preset_authority_manifest,
    preset_value_authority,
)


def _load_monitor_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_source_truth_simulation_monitor.py"
    spec = importlib.util.spec_from_file_location("source_truth_monitor", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _presets_by_name() -> dict[str, dict[str, object]]:
    return {str(item["name"]): item for item in list_presets()}


def _value_paths(data: dict[str, object], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.update(_value_paths(value, path))
        else:
            paths.add(path)
    return paths


def test_pf1000_presets_expose_distinct_source_scope_labels() -> None:
    presets = _presets_by_name()

    broad = presets["pf1000"]
    akel = presets["pf1000_akel"]
    trend = presets["pf1000_20kv"]

    assert broad["source_scope"] == "pf1000_standard_27kv_lee_malek"
    assert broad["source_scope_status"] == "same_scope_source_reviewed_not_certificate"
    assert broad["validation_scope"] == ""
    assert "shot-12581" in str(broad["source_scope_note"])
    assert "accepted evidence" in str(broad["source_scope_note"])

    assert akel["source_scope"] == "pf1000_16kv_2021_akel_shot12581"
    assert akel["source_scope_status"] == "same_scope_blocked_by_review"
    assert akel["validation_scope"] == "pf1000_16kv_2021_akel"
    assert "digitization packet is accepted" in str(akel["source_scope_note"])

    assert trend["source_scope"] == "pf1000_20kv_derived_operating_point"
    assert trend["source_scope_status"] == "derived_operating_point_not_validation_evidence"
    assert trend["validation_scope"] == ""


def test_pf1000_akel_registry_is_same_shot_as_preset() -> None:
    from dpf.validation.experimental_devices import DEVICES

    dev = DEVICES["PF-1000-16kV"]
    preset = get_preset("pf1000_akel")

    assert dev.fill_pressure_torr == 1.20
    assert dev.resistance == 6.1e-3
    assert dev.neutron_yield == 6.1e9
    assert dev.lee_fm == 0.17
    assert dev.lee_fmr == 0.26
    assert dev.lee_fcr == 0.75

    assert preset["circuit"]["R0"] == dev.resistance
    assert preset["snowplow"]["fill_pressure_Pa"] == 160.0
    assert preset["snowplow"]["mass_fraction"] == dev.lee_fm
    assert preset["snowplow"]["radial_mass_fraction"] == dev.lee_fmr
    assert preset["snowplow"]["radial_current_fraction"] == dev.lee_fcr


def test_poseidon_60kv_preset_tracks_registry_fit_without_certifying() -> None:
    from dpf.validation.experimental_devices import DEVICES

    presets = _presets_by_name()
    summary = presets["poseidon_60kv"]
    preset = get_preset("poseidon_60kv")
    dev = DEVICES["POSEIDON-60kV"]

    assert summary["source_scope"] == "poseidon_60kv_lee_saw_2014_ipfs"
    assert summary["source_scope_status"] == (
        "same_scope_source_reviewed_waveform_unverified_not_certificate"
    )
    assert summary["validation_scope"] == ""
    assert summary["can_support_validation_claims"] is False

    assert preset["circuit"]["C"] == dev.capacitance
    assert preset["circuit"]["V0"] == dev.voltage
    assert preset["circuit"]["L0"] == dev.inductance
    assert preset["circuit"]["R0"] == dev.resistance
    assert preset["snowplow"]["fill_pressure_Pa"] == 506.624
    assert preset["snowplow"]["current_fraction"] == dev.lee_fc
    assert preset["snowplow"]["mass_fraction"] == dev.lee_fm
    assert preset["snowplow"]["radial_mass_fraction"] == dev.lee_fmr
    assert preset["snowplow"]["radial_current_fraction"] == dev.lee_fcr


def test_unu_ictp_preset_tracks_registry_scope_without_certifying() -> None:
    from dpf.validation.experimental_devices import DEVICES

    presets = _presets_by_name()
    summary = presets["unu_ictp"]
    preset = get_preset("unu_ictp")
    dev = DEVICES["UNU-ICTP"]

    assert summary["source_scope"] == "unu_ictp_lee_saw_2014_table_p152_ipfs"
    assert summary["source_scope_status"] == (
        "same_scope_source_reviewed_waveform_unverified_not_certificate"
    )
    assert summary["validation_scope"] == ""
    assert summary["can_support_validation_claims"] is False

    assert preset["circuit"]["C"] == dev.capacitance
    assert preset["circuit"]["V0"] == dev.voltage
    assert preset["circuit"]["L0"] == dev.inductance
    assert preset["circuit"]["anode_radius"] == dev.anode_radius
    assert preset["circuit"]["cathode_radius"] == dev.cathode_radius
    assert preset["snowplow"]["anode_length"] == dev.anode_length
    assert preset["snowplow"]["fill_pressure_Pa"] == 533.288
    assert preset["snowplow"]["current_fraction"] == dev.lee_fc
    assert preset["snowplow"]["mass_fraction"] == dev.lee_fm
    assert preset["snowplow"]["radial_mass_fraction"] == dev.lee_fmr


def test_nx2_preset_tracks_reference_only_registry_without_certifying() -> None:
    from dpf.validation.experimental_devices import DEVICES

    presets = _presets_by_name()
    summary = presets["nx2"]
    preset = get_preset("nx2")
    dev = DEVICES["NX2"]

    assert summary["source_scope"] == "nx2_reference_only_lee_radpf"
    assert summary["source_scope_status"] == "reference_only_not_validation_evidence"
    assert summary["validation_scope"] == ""
    assert summary["can_support_validation_claims"] is False

    assert dev.reliability == "reference_only"
    assert preset["circuit"]["C"] == dev.capacitance
    assert preset["circuit"]["V0"] == dev.voltage
    assert preset["circuit"]["L0"] == dev.inductance
    assert preset["circuit"]["R0"] == dev.resistance
    assert preset["snowplow"]["current_fraction"] == dev.lee_fc
    assert preset["snowplow"]["mass_fraction"] == dev.lee_fm
    assert preset["snowplow"]["radial_mass_fraction"] == dev.lee_fmr
    assert preset["snowplow"]["radial_current_fraction"] == dev.lee_fcr


def test_mjolnir_preset_tracks_partial_source_registry_without_certifying() -> None:
    from dpf.validation.experimental_devices import DEVICES

    presets = _presets_by_name()
    summary = presets["mjolnir"]
    preset = get_preset("mjolnir")
    dev = DEVICES["MJOLNIR"]

    assert summary["source_scope"] == "mjolnir_schmidt_2021_1mj"
    assert summary["source_scope_status"] == (
        "same_scope_partial_source_review_waveform_reconstructed_not_certificate"
    )
    assert summary["validation_scope"] == ""
    assert summary["can_support_validation_claims"] is False

    assert dev.waveform_provenance == "reconstructed"
    assert preset["circuit"]["C"] == dev.capacitance
    assert preset["circuit"]["V0"] == dev.voltage
    assert preset["circuit"]["L0"] == dev.inductance
    assert preset["circuit"]["R0"] == dev.resistance
    assert preset["circuit"]["anode_radius"] == dev.anode_radius
    assert preset["circuit"]["cathode_radius"] == dev.cathode_radius
    assert preset["snowplow"]["fill_pressure_Pa"] == 933.254
    assert preset["snowplow"]["mass_fraction"] == dev.lee_fm
    assert preset["snowplow"]["radial_mass_fraction"] == dev.lee_fmr
    assert preset["snowplow"]["radial_current_fraction"] == dev.lee_fcr


def test_faeton_preset_tracks_table3_two_step_scope_without_certifying() -> None:
    from dpf.validation.experimental_devices import DEVICES
    from dpf.validation.kr_targets import faeton_i_high_voltage_dpf_targets

    presets = _presets_by_name()
    summary = presets["faeton"]
    preset = get_preset("faeton")
    dev = DEVICES["FAETON-I"]
    targets = faeton_i_high_voltage_dpf_targets()
    table3_rows = targets["current_waveform_targets"]["table_3_shots"]
    source_pairs = {(row["fcr"], row["fcr2"]) for row in table3_rows}

    assert summary["source_scope"] == "faeton_i_damideh_2025_table3_shot1027_two_step_restrike"
    assert summary["source_scope_status"] == (
        "same_scope_partial_source_review_waveform_reconstructed_not_certificate"
    )
    assert summary["validation_scope"] == ""
    assert summary["can_support_validation_claims"] is False

    assert dev.waveform_provenance == "reconstructed"
    assert preset["snowplow"]["radial_current_fraction"] == dev.lee_fcr
    assert preset["snowplow"]["radial_current_fraction_2"] == dev.lee_fcr2
    assert (
        preset["snowplow"]["radial_current_fraction"],
        preset["snowplow"]["radial_current_fraction_2"],
    ) in source_pairs
    assert preset["snowplow"]["radial_transition_time"] == 7.0e-6


def test_source_truth_monitor_explains_remaining_nonaccepting_gaps() -> None:
    from dpf.validation.experimental_devices import DEVICES

    monitor = _load_monitor_module()

    nx2 = DEVICES["NX2"]
    nx2_state = monitor._source_state("NX2", nx2)
    nx2_gaps = monitor._source_gap_flags("NX2", nx2, nx2_state)
    assert "reference_only_device_not_scientific_validation_target" in nx2_gaps
    assert any("not_same_shot_deuterium" in flag for flag in nx2_gaps)

    mjolnir = DEVICES["MJOLNIR"]
    mjolnir_model_gaps = monitor._model_coverage_flags("MJOLNIR", mjolnir)
    assert mjolnir_model_gaps == [
        "mjolnir_restrike_current_trace_model_required_by_kr_"
        "but_no_accepted_timing_or_magnitude_parameters"
    ]

    faeton_flags = monitor._source_config_flags(get_preset("faeton"), DEVICES["FAETON-I"])
    assert faeton_flags == [
        "snowplow.radial_transition_time_not_in_faeton_kr_extract_observed=7e-06"
    ]


def test_source_truth_monitor_top_level_dashboard_categories() -> None:
    monitor = _load_monitor_module()

    assert monitor.TOP_LEVEL_MONITOR_CATEGORIES == (
        "operational_failure",
        "source_gap",
        "model_coverage_gap",
        "numerical_verification_gap",
        "validation_ready_accuracy_failure",
    )
    categories = monitor._top_level_monitor_categories({
        "workflow_status": "broken",
        "error": "solver failed",
        "source_gap_flags": ["missing_waveform"],
        "model_coverage_flags": ["restrike_missing"],
        "numerical_verification_flags": ["backend_parity_missing"],
        "source_state": {"validation_ready": True},
        "accuracy_flags": ["nrmse_full>0.35_pipeline_fence"],
    })

    assert categories == list(monitor.TOP_LEVEL_MONITOR_CATEGORIES)


def test_list_presets_fails_closed_for_value_authority() -> None:
    for preset in list_presets():
        assert preset["validation_status"] == "not_validation_evidence"
        assert preset["can_support_validation_claims"] is False
        assert preset["value_source_status"]
        assert preset["source_scope_status"]


def test_preset_value_authority_covers_every_config_leaf() -> None:
    for summary in list_presets():
        name = str(summary["name"])
        config = get_preset(name)
        expected_paths = _value_paths(config)
        records = preset_value_authority(name)
        observed_paths = {str(record["path"]) for record in records}

        assert observed_paths == expected_paths
        for record in records:
            assert record["validation_status"] == "not_validation_evidence"
            assert record["can_support_validation_claims"] is False
            assert record["value_source_status"]


def test_preset_authority_manifest_includes_all_presets() -> None:
    manifest = preset_authority_manifest()
    expected_names = {str(item["name"]) for item in list_presets()}

    assert set(manifest) == expected_names
    assert all(manifest[name] for name in expected_names)


def test_get_preset_does_not_promote_source_scope_into_simulation_config() -> None:
    preset = get_preset("pf1000_akel")

    assert "_meta" not in preset
    assert "source_scope" not in preset
    assert "source_scope_status" not in preset
    assert "validation_scope" not in preset


def test_rest_preset_list_includes_source_scope_metadata() -> None:
    from fastapi.testclient import TestClient

    from dpf.server.app import app

    response = TestClient(app).get("/api/presets")

    assert response.status_code == 200
    presets = {item["name"]: item for item in response.json()}
    assert presets["pf1000_akel"]["validation_scope"] == "pf1000_16kv_2021_akel"
    assert presets["pf1000"]["source_scope_status"] == (
        "same_scope_source_reviewed_not_certificate"
    )


def test_preset_request_scope_mapping_is_source_scoped_only() -> None:
    from dpf.server.app import _validation_scope_from_request
    from dpf.server.models import CreateSimulationRequest

    akel_request = CreateSimulationRequest(config={}, preset="pf1000_akel")
    broad_request = CreateSimulationRequest(config={}, preset="pf1000")

    assert _validation_scope_from_request(akel_request) == "pf1000_16kv_2021_akel"
    assert _validation_scope_from_request(broad_request) is None
