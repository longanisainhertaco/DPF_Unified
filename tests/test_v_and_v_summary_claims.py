from pathlib import Path


def test_v_and_v_summary_uses_source_gated_validation_language() -> None:
    summary = Path("docs/V_AND_V_SUMMARY.md").read_text(encoding="utf-8")

    assert "SOURCE-GATED, PARTIAL" in summary
    assert "1/9 registered devices validation-ready" in summary
    assert "not spatial\nMHD validation" in summary
    assert "6/7 devices PASS" not in summary
    assert "Zero calibration" not in summary
    assert "24-shot" not in summary
    assert "Circuit-Level (Lee Snowplow Model) — VALIDATED" not in summary
