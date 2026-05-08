from pathlib import Path


def test_ai_disclosure_uses_current_source_gated_validation_status() -> None:
    disclosure = Path("docs/AI_DISCLOSURE.md").read_text(encoding="utf-8")

    assert "source-gated against local `KnowledgeReference/` records" in disclosure
    assert "only registered\n   validation-ready circuit waveform record" in disclosure
    assert "published data for 6 devices" not in disclosure
    assert "validated statistically across 24" not in disclosure
    assert "1.27% mean I_peak error" not in disclosure
