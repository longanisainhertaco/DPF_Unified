from pathlib import Path


def test_scope_claims_are_source_gated() -> None:
    scope = Path("docs/SCOPE.md").read_text(encoding="utf-8")

    assert "source-gated validation evidence" in scope
    assert "PF-1000 Scholz waveform record" in scope
    assert "validation-ready" in scope
    assert "not an end-to-end DPF validation claim" in scope
    assert "validated against 6 devices" not in scope
    assert "6/7 PASS" not in scope
    assert "Validated against published experimental data with zero calibration" not in scope
