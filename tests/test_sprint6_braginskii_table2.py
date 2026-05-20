"""Tests for the Sprint 6 WS3 Braginskii Table 2 target-extracted packet."""

from __future__ import annotations

from pathlib import Path

from dpf.first_principles.sprint6_braginskii_table2_target_extraction import (
    BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION,
    sprint6_ws3_braginskii_target_extraction,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = REPO_ROOT / (
    "archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf"
)
RENDER_DIR = REPO_ROOT / "docs/extractions/braginskii_1965_render_evidence"
HUMAN_PACKET = REPO_ROOT / (
    "docs/extractions/BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md"
)


def test_packet_fails_closed() -> None:
    p = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION
    assert p["accepted_runtime_claim"] is False
    assert p["can_support_first_principles_acceptance"] is False


def test_manifest_fails_closed() -> None:
    m = sprint6_ws3_braginskii_target_extraction()
    assert m["accepted_runtime_claim"] is False
    assert m["can_support_first_principles_acceptance"] is False
    assert m["source_equivalence_granted_for_cross_check_lanes"] is False


def test_pdf_exists_and_sha256_matches() -> None:
    """The PDF must be on disk; the recorded SHA-256 must match the file."""
    import hashlib

    assert PDF_PATH.exists(), f"PDF missing at {PDF_PATH}"
    sha = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    assert sha == BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION["pdf_sha256"], (
        f"PDF SHA-256 drift: file={sha}, packet="
        f"{BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION['pdf_sha256']}"
    )


def test_render_manifest_exists() -> None:
    """The render-evidence directory must contain the manifest + 4 PNGs."""
    assert RENDER_DIR.exists(), f"Render-evidence dir missing: {RENDER_DIR}"
    manifest = RENDER_DIR / "render_manifest.json"
    assert manifest.exists(), f"Render manifest missing: {manifest}"
    pngs = sorted(RENDER_DIR.glob("*.png"))
    assert len(pngs) == 4, f"Expected 4 rendered PNGs; found {len(pngs)}"


def test_primary_render_png_exists_and_sha_prefix_matches() -> None:
    import hashlib

    p = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION
    primary = REPO_ROOT / p["primary_render_path"]
    assert primary.exists(), f"Primary render missing: {primary}"
    sha = hashlib.sha256(primary.read_bytes()).hexdigest()
    assert sha.startswith(p["primary_render_sha256_prefix"]), (
        f"Primary-render SHA prefix drift: file={sha[:16]}, "
        f"packet={p['primary_render_sha256_prefix']}"
    )


def test_z_columns_complete() -> None:
    p = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION
    assert p["z_columns"] == (1, 2, 3, 4, "inf")


def test_z1_spot_checks_present() -> None:
    """The DPF deuterium-plasma case (Z=1) must have all 17 coefficient rows."""
    z1 = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION["spot_checked_table_2"]["z1"]
    required_rows = {
        "alpha_0",
        "beta_0",
        "gamma_0",
        "delta_0",
        "delta_1",
        "alpha_1_prime",
        "alpha_0_prime",
        "alpha_1_double_prime",
        "alpha_0_double_prime",
        "beta_1_prime",
        "beta_0_prime",
        "beta_1_double_prime",
        "beta_0_tilde_double_prime",
        "gamma_1_prime",
        "gamma_0_prime",
        "gamma_1_double_prime",
        "gamma_0_double_prime",
    }
    assert set(z1.keys()) == required_rows


def test_z1_canonical_values_match_render() -> None:
    """The Z=1 canonical values must match the render. The Sprint 5 WS2
    audit verified Z=1 alpha_0 = 0.5129, beta_0 = 0.7110, gamma_0 = 3.1616
    via PyMuPDF page render at PDF p.26 = journal p.251 right half.
    """
    z1 = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION["spot_checked_table_2"]["z1"]
    assert z1["alpha_0"] == "0.5129"
    assert z1["beta_0"] == "0.7110"
    assert z1["gamma_0"] == "3.1616"
    assert z1["delta_0"] == "3.7703"


def test_status_transition_records_render_verification_closure() -> None:
    p = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION
    assert "pdf_present_needs_rendered_page_or_ocr_verification" in p["status_transition"]
    assert "target_extracted_source_supported" in p["status_transition"]


def test_blocker_id_targets_brag_001() -> None:
    p = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION
    assert p["blocker_id"] == "CLOSURE-BLK-BRAG-001"


def test_cross_check_lanes_are_not_substitutes() -> None:
    """PlasmaPy is a cross-check lane, NOT a source-equivalent substitute."""
    lanes = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION[
        "cross_check_lanes_not_substitutes"
    ]
    assert len(lanes) >= 1
    for lane in lanes:
        assert lane["source_equivalence_granted"] is False
        assert "review_packet" in lane
        # The PlasmaPy review packet path must exist
        review_path = REPO_ROOT / lane["review_packet"]
        assert review_path.exists(), (
            f"Cross-check lane review packet missing: {review_path}"
        )


def test_human_readable_packet_present() -> None:
    assert HUMAN_PACKET.exists(), (
        f"Human-readable extraction packet missing: {HUMAN_PACKET}"
    )


def test_audit_corrections_carried_forward() -> None:
    m = sprint6_ws3_braginskii_target_extraction()
    folded = set(m["audit_corrections_carried_forward"])
    assert "codex_v1_row_8_braginskii_render_verification_closed" in folded


def test_2up_spread_layout_noted() -> None:
    """The page-mapping clarification (PDF is 2-up scanned) must be recorded
    so future readers do not re-encounter the offset confusion."""
    p = BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION
    assert p["table_2_position_in_2up_spread"] == "right_half_of_2up_spread"
    assert "2-up scanned spread" in p["pdf_layout_note"]
