from pathlib import Path


def test_readme_validation_claims_are_source_gated() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "not yet an end-to-end predictive DPF simulator" in readme
    assert "Predictive-readiness gate" in readme
    assert "only the standard PF-1000 Scholz waveform record" in readme
    assert "validated against experimental current waveforms from four devices" not in readme
    assert "waveforms from 6 devices with zero parameter calibration" not in readme
    assert "V&V campaign (2,026 shots)" not in readme
    assert "24-shot validation data" not in readme
