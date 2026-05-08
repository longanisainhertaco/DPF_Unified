from dpf.validation.experimental import (
    PF1000_DATA,
    POSEIDON_60KV_DATA,
    get_validation_ready_devices,
    validate_current_waveform,
    validate_neutron_yield,
)


def test_validate_current_waveform_reports_kr_ready_source_authority() -> None:
    metrics = validate_current_waveform(
        PF1000_DATA.waveform_t,
        PF1000_DATA.waveform_I,
        "PF-1000",
    )

    authority = metrics["source_authority"]
    assert authority["validation_ready"] is True
    assert authority["validation_role"] == "tier1_circuit_evidence_candidate"
    assert authority["waveform_kr_status"] == "verified"


def test_validate_current_waveform_marks_external_archive_as_numeric_only() -> None:
    metrics = validate_current_waveform(
        POSEIDON_60KV_DATA.waveform_t,
        POSEIDON_60KV_DATA.waveform_I,
        "POSEIDON-60kV",
    )

    authority = metrics["source_authority"]
    assert metrics["waveform_nrmse"] == 0.0
    assert authority["kr_status"] == "verified"
    assert authority["waveform_provenance"] == "measured"
    assert authority["waveform_kr_status"] == "unverified"
    assert authority["validation_ready"] is False
    assert authority["validation_role"] == "numeric_comparison_only"


def test_validate_neutron_yield_is_numeric_only_not_tier_five() -> None:
    metrics = validate_neutron_yield(PF1000_DATA.neutron_yield, "PF-1000")

    assert metrics["within_order_magnitude"] is True
    authority = metrics["source_authority"]
    assert authority["kr_status"] == "verified"
    assert authority["validation_ready"] is False
    assert authority["validation_role"] == "numeric_yield_comparison_only"
    assert "not_neutron_physics_validation" in metrics["validity_notes"]


def test_validation_ready_device_registry_is_source_gated() -> None:
    ready = get_validation_ready_devices()

    assert set(ready) == {"PF-1000"}
    assert ready["PF-1000"].waveform_kr_status == "verified"
