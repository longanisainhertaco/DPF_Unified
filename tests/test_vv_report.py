from dpf.validation.vv_report import _get_device_validation, generate_vv_report


def _status_by_device() -> dict[str, dict]:
    return {entry["device"]: entry for entry in _get_device_validation()}


def test_device_validation_summary_is_source_authority_based() -> None:
    statuses = _status_by_device()

    assert statuses["PF-1000"]["status"] == "VALIDATION_READY"
    assert statuses["PF-1000"]["kr_status"] == "verified"
    assert statuses["PF-1000"]["waveform_provenance"] == "measured"
    assert statuses["PF-1000"]["waveform_kr_status"] == "verified"

    assert statuses["PF-1000-16kV"]["status"] == "RECONSTRUCTED_ONLY"
    assert statuses["NX2"]["status"] == "REFERENCE_ONLY"
    assert statuses["POSEIDON-60kV"]["status"] == "WAVEFORM_KR_UNVERIFIED"
    assert statuses["UNU-ICTP"]["status"] == "WAVEFORM_KR_UNVERIFIED"


def test_vv_report_does_not_hardcode_device_pass_claims() -> None:
    report = generate_vv_report()

    assert "devices validation-ready" in report
    assert "Devices validation-ready" in report
    assert "Best single-device accuracy" not in report
    assert "devices PASS" not in report
    assert "| Statistical validation | 24-shot PF-1000 (r=0.9899) | PASS |" not in report
    assert "| Statistical validation | Requires explicit KR-sourced comparison bundle | Source-gated |" in report
    assert "Line radiation (empirical coronal fits)" in report
    assert "Line radiation (CHIANTI-style)" not in report
