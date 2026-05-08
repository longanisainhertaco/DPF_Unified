from pathlib import Path


def test_joss_draft_marks_validation_claims_stale() -> None:
    draft = Path("docs/joss-paper-draft.md").read_text(encoding="utf-8")

    assert "stale paper draft" in draft
    assert "Source-Gated Status" in draft
    assert "validated against published experimental data for seven DPF devices" not in draft
    assert "mean peak current error across 24 shots" not in draft
    assert "Current waveforms were compared against published experimental data for six" not in draft
    assert "Statistical Validation: PF-1000 24-Shot Campaign" not in draft
    assert "validated against experimental data" not in draft
