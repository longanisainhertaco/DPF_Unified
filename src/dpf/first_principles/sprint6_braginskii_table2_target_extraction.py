"""Sprint 6 WS3 — Braginskii 1965 Table 2 target-extracted KR packet.

This module encodes the render-verified Braginskii 1965 Table 2 target
extraction documented in
``docs/extractions/BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md``
with rendered-page evidence under
``docs/extractions/braginskii_1965_render_evidence/``.

The packet promotes ``CLOSURE-BLK-BRAG-001`` from
``pdf_present_needs_rendered_page_or_ocr_verification`` (Codex V1 audit
row 8) to
``target_extracted_source_supported_pending_runtime_consumption_and_review``.

Promotion is a source-availability transition, NOT a runtime acceptance
transition. Per the Codex Sprint 5 WS2 audit:

- ``accepted_runtime_claim = False``
- ``can_support_first_principles_acceptance = False``

Runtime acceptance still requires (i) the closure-packet code to cite this
target extraction, (ii) numerical-fidelity tests that exercise the Z=1 row
coefficients through the resistivity / heat-conduction formulas, (iii) a
same-scope comparator run, and (iv) certificate-gate pass for the broader
transport-closure chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION: Mapping[str, Any] = {
    "source_id": "braginskii_1965_table_2_target_extracted",
    "blocker_id": "CLOSURE-BLK-BRAG-001",
    "status_transition": (
        "pdf_present_needs_rendered_page_or_ocr_verification"
        " -> target_extracted_source_supported_pending_runtime_consumption_and_review"
    ),
    "scope_tag": "generic_formulary",
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    # PDF
    "pdf_path": (
        "archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf"
    ),
    "pdf_sha256": (
        "9687440676b43b02758373ef83d5a471730e9ca4bc207cb98f248fcafcb9d404"
    ),
    "pdf_size_bytes": 5089370,
    "pdf_total_pages": 56,
    # Render evidence
    "primary_render_path": (
        "docs/extractions/braginskii_1965_render_evidence/"
        "pdf_p026_journal_p250_p251.png"
    ),
    "primary_render_sha256_prefix": "c914283871fbf6f1",
    "render_manifest_path": (
        "docs/extractions/braginskii_1965_render_evidence/render_manifest.json"
    ),
    "renderer": "PyMuPDF 1.27.2.3 (fitz) at 200 dpi",
    # Page mapping (corrected from Sprint 5 packet)
    "table_2_journal_page": 251,
    "table_2_pdf_page_one_indexed": 26,
    "table_2_position_in_2up_spread": "right_half_of_2up_spread",
    "pdf_layout_note": (
        "Each PDF page is a 2-up scanned spread containing two consecutive "
        "journal pages. PDF p.26 = journal pp.250 (left) + 251 (right). The "
        "Sprint 5 packet's pdf_page_to_journal_page_offset = 202 describes "
        "the LEFT-page offset of the spread; Table 2 is on the RIGHT half."
    ),
    # Table 2 columns
    "z_columns": (1, 2, 3, 4, "inf"),
    "z1_two_pass_verified": True,
    "z_inf_two_pass_verified": True,
    "z2_z3_z4_render_visible_but_review_required_at_consumption": True,
    "cells_flagged_review_required": (
        ("alpha_0_prime", 3),
        ("alpha_0_prime", 4),
        ("alpha_0_prime", "inf"),
        ("alpha_0_double_prime", 3),
        ("gamma_1_prime", "inf"),
    ),
    # Spot-checked numeric values (verbatim from render). Z=1 column is the
    # DPF deuterium-plasma case.
    "spot_checked_table_2": {
        "z1": {
            "alpha_0": "0.5129",
            "beta_0": "0.7110",
            "gamma_0": "3.1616",
            "delta_0": "3.7703",
            "delta_1": "14.79",
            "alpha_1_prime": "6.416",
            "alpha_0_prime": "1.837",
            "alpha_1_double_prime": "1.704",
            "alpha_0_double_prime": "0.7796",
            "beta_1_prime": "5.101",
            "beta_0_prime": "2.681",
            "beta_1_double_prime": "3/2",
            "beta_0_tilde_double_prime": "3.053",
            "gamma_1_prime": "4.664",
            "gamma_0_prime": "11.92",
            "gamma_1_double_prime": "5/2",
            "gamma_0_double_prime": "21.67",
        },
        "z_inf": {
            "alpha_0": "0.2949",
            "beta_0": "1.521",
            "gamma_0": "12.471",
            "delta_0": "0.0961",
            "delta_1": "7.482",
            "alpha_1_prime": "4.63",
            "alpha_1_double_prime": "1.704",
            "alpha_0_double_prime": "0.0940",
            "beta_1_prime": "3.798",
            "beta_0_prime": "0.1461",
            "beta_1_double_prime": "3/2",
            "beta_0_tilde_double_prime": "0.877",
            "gamma_0_prime": "1.20",
            "gamma_1_double_prime": "5/2",
            "gamma_0_double_prime": "10.23",
        },
    },
    # Equation cross-references
    "equation_region_pdf_pages": (25, 28),
    "equation_region_journal_pages": (248, 255),
    "equations_4_30_to_4_45_present_in_render_manifest": True,
    # Cross-check lanes (NOT source-equivalent substitutes)
    "cross_check_lanes_not_substitutes": (
        {
            "name": "plasmapy_formulary_braginskii_z_table",
            "url": (
                "https://docs.plasmapy.org/en/stable/formulary/braginskii.html"
            ),
            "review_packet": (
                "docs/source_equivalence_review/"
                "PLASMAPY_BRAGINSKII_CROSS_CHECK_REVIEW_PACKET_2026_05_20.md"
            ),
            "source_equivalence_granted": False,
        },
    ),
    # Provenance and audit basis
    "audit_basis": (
        "docs/CODEX_SPRINT5_WS2_AUDIT_2026_05_20.md (HEAD 558de6f); "
        "docs/CODEX_FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_2026_05_20.md "
        "(row 8 downgrade closed by render verification)"
    ),
    "extraction_date": "2026-05-20",
    "extraction_method": (
        "PyMuPDF page render at 200 dpi + two-pass visual confirmation of "
        "Z=1 and Z=inf columns; Z=2, Z=3, Z=4 columns visible at rendered "
        "evidence but require per-cell visual re-verification at consumption"
    ),
    "human_readable_packet_path": (
        "docs/extractions/BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md"
    ),
    "kr_mirror_path": (
        "KnowledgeReference/braginskii-1965-table-2-TARGET-EXTRACTED.md"
    ),
}


# Sprint 6 WS3 master manifest — currently single-packet but structured to
# accept additional WS3 target extractions in future sub-sprints.
SPRINT_6_WS3_TARGET_EXTRACTIONS: Mapping[str, Mapping[str, Any]] = {
    BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION["source_id"]: (
        BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION
    ),
}


def sprint6_ws3_braginskii_target_extraction() -> Mapping[str, Any]:
    """Return the Sprint 6 WS3 Braginskii Table 2 target-extraction manifest.

    The aggregate manifest enforces:

    - ``accepted_runtime_claim = False``
    - ``can_support_first_principles_acceptance = False``
    - ``source_equivalence_granted_for_cross_check_lanes = False``

    These flags are not negotiable by data content. They are runtime-acceptance
    boundary markers and require code+test+certificate-gate work in a later
    sprint to be relaxed (and even then never via this packet alone).
    """
    return {
        "packet_id": "sprint6_ws3_braginskii_table_2_target_extraction_2026_05_20",
        "controlling_goal": (
            "Sprint 6 Goal — WS3 Braginskii Table 2 target extraction"
        ),
        "workstream": "WS3",
        "packets_count": len(SPRINT_6_WS3_TARGET_EXTRACTIONS),
        "packets": dict(SPRINT_6_WS3_TARGET_EXTRACTIONS),
        "accepted_runtime_claim": False,
        "can_support_first_principles_acceptance": False,
        "source_equivalence_granted_for_cross_check_lanes": False,
        "audit_corrections_carried_forward": (
            "codex_v1_row_8_braginskii_render_verification_closed",
            "sprint5_ws2_a1_per_target_resolves_subset_invariant_preserved",
            "sprint5_ws2_a3_no_aggressive_closure_language",
        ),
    }


__all__ = (
    "BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION",
    "SPRINT_6_WS3_TARGET_EXTRACTIONS",
    "sprint6_ws3_braginskii_target_extraction",
)
