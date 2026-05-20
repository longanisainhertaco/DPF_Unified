# Braginskii 1965 Table 2 — Target-Extracted KR Packet (2026-05-20)

Sprint: Sprint 6 WS3.
Blocker: `CLOSURE-BLK-BRAG-001` source-availability (target-extraction lane;
runtime acceptance still requires code consumption and review — see §6).

## 1. Source identity

- **Citation:** Braginskii, S. I. (1965). *"Transport processes in a plasma"*,
  in M. A. Leontovich (ed.), *Reviews of Plasma Physics, Vol. 1*, Consultants
  Bureau, New York, pp. 205-311. Translated by Herbert Lashinsky.
- **PDF on disk:** `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf`
- **PDF SHA-256:** `9687440676b43b02758373ef83d5a471730e9ca4bc207cb98f248fcafcb9d404`
- **PDF size:** 5,089,370 bytes (4.85 MiB)
- **PDF total pages:** 56 (scanned 2-up; each PDF page holds two journal pages)
- **scope_tag:** `generic_formulary` (Z-dependent transport-coefficient table
  for fully-ionized plasmas; the Z=1 column is the DPF deuterium-plasma case)
- **accepted_runtime_claim:** `false`
- **can_support_first_principles_acceptance:** `false`

## 2. Render evidence

Render evidence files live under
`docs/extractions/braginskii_1965_render_evidence/`. Each PDF page renders
as a 2-up scanned spread of two consecutive journal pages.

Render manifest (`render_manifest.json` in that directory) records the
following:

| PDF page (1-idx) | journal pages | render filename | SHA-256 | contains Table 2 |
| ---: | --- | --- | --- | --- |
| 25 | 248-249 | `pdf_p025_journal_p248_p249.png` | `26bb8a89de86970c…` | no |
| 26 | **250-251** | `pdf_p026_journal_p250_p251.png` | **`c914283871fbf6f1…`** | **yes — right half** |
| 27 | 252-253 | `pdf_p027_journal_p252_p253.png` | `9588cc5bcd24af8d…` | no |
| 28 | 254-255 | `pdf_p028_journal_p254_p255.png` | `71ccbd5ae73ee386…` | no |

- **Primary render for Table 2:** `pdf_p026_journal_p250_p251.png`
  (2195 × 1695 px, 200 dpi).
- **Table 2 position in the render:** right half of the 2-up spread
  (journal page 251).
- **Renderer:** PyMuPDF (`fitz`) 1.27.2.3 in `.venv312` (Python 3.12.13).
- **Re-derivation command:** `scripts/render_braginskii_table2.py` would
  re-produce the manifest deterministically from the on-disk PDF; this packet
  was produced by the inline rendering script logged in
  `docs/extractions/braginskii_1965_render_evidence/render_manifest.json`.

The Codex V1 audit row 8 had downgraded `CLOSURE-BLK-BRAG-001` to
`pdf_present_needs_rendered_page_or_ocr_verification` because `pdftotext` did
not expose Table 2 (the PDF is image-scanned, not text-layered). The
PyMuPDF render at 200 dpi resolves Table 2 legibly; this packet is the
render-verified target extraction. Codex independently re-rendered page 26
and confirmed the table layout in the Sprint 5 WS2 audit at HEAD `558de6f`.

## 3. Page-mapping clarification

V1 prose and the Sprint 5 WS2 packet used a single PDF-to-journal offset
"PDF p.26 → journal p.251" with implicit offset 225. The render evidence
shows the PDF is **2-up scanned**: each PDF page contains TWO consecutive
journal pages. The corrected mapping for the Sprint 5 packet's
`pdf_page_to_journal_page_offset = 202` should be read as a left-page
offset for the spread; the right-page offset is +203. For Table 2 the
correct citation is:

- PDF p.26 (1-indexed) → journal p.250 (left of spread) / **journal p.251 (right of spread, contains Table 2)**

The Sprint 5 packet's `sprint5_target_extractions.py::BRAGINSKII_1965_TRANSPORT_EXTRACTION`
field `pdf_page_to_journal_page_offset = 202` describes the left-page offset
of the spread, not a single per-page constant; downstream consumers reading
that field should compute the right-page (Table 2) journal page as
`pdf_page_one_indexed + 202 + 23 = 251`. A clean re-render with this packet's
manifest avoids that arithmetic.

## 4. Table 2 — Z-dependent transport coefficient families

Header: blank · Z=1 · Z=2 · Z=3 · Z=4 · Z=∞.

The five columns are the principal Z values in the source. The Z=1 column
applies to the DPF deuterium-plasma case (single-charged ions).

| coefficient | Z=1 | Z=2 | Z=3 | Z=4 | Z=∞ |
| --- | --- | --- | --- | --- | --- |
| α₀ = 1 − (α₀'/δ₀) | **0.5129** | 0.4408 | 0.3965 | 0.3752 | **0.2949** |
| β₀ = β₀'/δ₀ | **0.7110** | 0.9052 | 1.016 | 1.064 | **1.521** |
| γ₀ = γ₀'/δ₀ | **3.1616** | 4.890 | 6.064 | 6.920 | **12.471** |
| δ₀ | **3.7703** | 1.0465 | 0.5814 | 0.4106 | **0.0961** |
| δ₁ | **14.79** | 10.80 | 9.618 | 9.055 | **7.482** |
| α₁' | **6.416** | 5.523 | 5.226 | 5.077 | 4.63 |
| α₀' | **1.837** | 0.5956 | 0.3555 (review-required) | 0.2566 (review-required) | (review-required) |
| α₁'' | **1.704** | 1.704 | 1.704 | 1.704 | 1.704 |
| α₀'' | **0.7796** | 0.3429 | 0.7400 (review-required — anomalous Z trend) | 0.1957 | 0.0940 |
| β₁' | **5.101** | 4.450 | 4.233 | 4.124 | 3.798 |
| β₀' | **2.681** | 0.9473 | 0.5905 | 0.4478 | 0.1461 |
| β₁'' | **3/2** | 3/2 | 3/2 | 3/2 | 3/2 |
| β̃₀'' | **3.053** | 1.784 | 1.442 | 1.285 | 0.877 |
| γ₁' | **4.664** | 3.957 | 3.721 | 3.604 | 3.25 (review-required low-confidence) |
| γ₀' | **11.92** | 5.118 | 3.525 | 2.941 | 1.20 |
| γ₁'' | **5/2** | 5/2 | 5/2 | 5/2 | 5/2 |
| γ₀'' | **21.67** | 15.37 | 13.53 | 12.65 | 10.23 |

**Bolded** values are spot-confirmed via two independent reads of
`pdf_p026_journal_p250_p251.png` (the first read during the Sprint 5 WS2
extraction agent's PDF page render; the second during this Sprint 6 WS3
re-render). The Z=2 / Z=3 / Z=4 inner columns are render-visible at the
cited image but were not individually two-pass verified in this packet;
cells flagged `(review-required)` are visually ambiguous and MUST be
re-rendered and re-read at consumption time. Code that consumes any cell
flagged `(review-required)` MUST NOT promote it through a target-extracted
gate without a second-reader confirmation.

## 5. Equation-region cross-references

Equations 4.30 - 4.45 (transport-closure relations binding the table to
runtime physics) appear in pp. 249-253 (PDF pp. 25-28 spreads):

- **Eq. 4.30** — friction force decomposition
  `R_u = -α_∥ u_∥ - α_⊥ u_⊥ + α_∧ [h × u]` where `u = V_e − V_i`.
- **Eq. 4.31** — thermal force decomposition
  `R_T = -β_∥^T ∇_∥ T_e - β_⊥^T ∇_⊥ T_e - β_∧^T [h × ∇T_e]`.
- **Eq. 4.32, 4.33** — electron heat flux `q_e` decomposed into frictional
  and thermal contributions.
- **Eq. 4.34 - 4.38** — `α, β, κ` coefficients in terms of the Z-dependent
  table entries and `x = ω_e τ_e`, `Δ = x⁴ + δ₁ x² + δ₀`.
- **Eq. 4.39** — ion heat flux `q_i` with `κ_∥^i`, `κ_⊥^i`, `κ_∧^i`.
- **Eq. 4.40** — ion thermal conductivities
  `κ_∥^i = 3.906 n_i T_i τ_i / m_i` (visible on render p.26 left half);
  `κ_⊥^i = (n_i T_i τ_i / m_i)(2 x² + 2.645)/Δ`;
  `κ_∧^i = (n_i T_i τ_i / m_i) x (5/2 x² + 4.65)/Δ`
  with `x = ω_i τ_i`, `Δ = x⁴ + 2.70 x² + 0.677`.
- **Eq. 4.41 - 4.42** — viscosity tensor π_{αβ} via five coefficients
  `η_0..η_4` and rate-of-strain tensors `W_{0αβ} .. W_{4αβ}`.
- **Eq. 4.44** — ion viscosities `η_0^i = 0.96 n_i T_i τ_i`,
  `η_2^i = n_i T_i τ_i (6/5 x² + 2.23)/Δ`,
  `η_4^i = n_i T_i τ_i x (x² + 2.38)/Δ`,
  `Δ = x⁴ + 4.03 x² + 2.33`.
- **Eq. 4.45** — electron viscosities (Z=1):
  `η_0^e = 0.733 n_e T_e τ_e`,
  `η_2^e = n_e T_e τ_e (2.05 x² + 8.50)/Δ`,
  `η_4^e = -n_e T_e τ_e x (x² + 7.91)/Δ`,
  with `x = ω_e τ_e`, `Δ = x⁴ + 13.8 x² + 11.6`.

Equations 4.32 - 4.40 and 4.41 - 4.45 are visible on render pages 26-28 of
this packet's manifest. A future Sprint 6+ promotion may add equation-level
target-extraction records; this Sprint 6 WS3 packet promotes Table 2 only.

## 6. Acceptance posture — explicit non-promotion

**This packet is the source-availability target extraction. It is NOT a
runtime closure.** Per the Codex V2 audit verdict and the Sprint 6 goal:

- `accepted_runtime_claim = false`.
- `can_support_first_principles_acceptance = false`.
- `CLOSURE-BLK-BRAG-001` moves from
  `pdf_present_needs_rendered_page_or_ocr_verification` →
  `target_extracted_source_supported_pending_runtime_consumption_and_review`.
- Runtime acceptance still requires (i) the closure-packet code in
  `src/dpf/first_principles/closure_packet.py::electrical_thermal_transport`
  to cite this packet by path, (ii) a numerical-fidelity test that exercises
  the Z=1 row coefficients through the resistivity / heat-conduction
  formulas, (iii) a same-scope comparator run, and (iv) a certificate gate
  pass for the broader transport-closure chain.

Cross-check sources accepted as cross-check lanes only (NOT
source-equivalent replacements):

- PlasmaPy `formulary.braginskii.ClassicalTransport` hardcoded Z-table
  (https://docs.plasmapy.org/en/stable/formulary/braginskii.html). Use as a
  second-witness against this packet's Z=1 row; do not treat PlasmaPy as a
  source-equivalent substitute pending a separate source-equivalence review
  packet (Sprint 6 WS4 queue).

## 7. Provenance summary (machine-readable)

```python
BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION = {
    "source_id": "braginskii_1965_table_2_target_extracted",
    "blocker_id": "CLOSURE-BLK-BRAG-001",
    "status_transition": (
        "pdf_present_needs_rendered_page_or_ocr_verification"
        " -> target_extracted_source_supported_"
        "pending_runtime_consumption_and_review"
    ),
    "scope_tag": "generic_formulary",
    "accepted_runtime_claim": False,
    "can_support_first_principles_acceptance": False,
    "pdf_path": (
        "archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf"
    ),
    "pdf_sha256": "9687440676b43b02758373ef83d5a471730e9ca4bc207cb98f248fcafcb9d404",
    "pdf_size_bytes": 5089370,
    "table_2_render_path": (
        "docs/extractions/braginskii_1965_render_evidence/"
        "pdf_p026_journal_p250_p251.png"
    ),
    "table_2_render_sha256_prefix": "c914283871fbf6f1",
    "render_manifest_path": (
        "docs/extractions/braginskii_1965_render_evidence/render_manifest.json"
    ),
    "renderer": "PyMuPDF 1.27.2.3 (fitz) at 200 dpi",
    "table_2_journal_page": 251,
    "table_2_pdf_page_one_indexed": 26,
    "table_2_position_in_2up_spread": "right half",
    "z_columns": (1, 2, 3, 4, "inf"),
    "z1_coefficients_spot_checked_two_pass": True,
    "z2_z3_z4_cells_render_visible_but_review_required": True,
    "cells_flagged_review_required": (
        ("alpha_0_prime", 3),
        ("alpha_0_prime", 4),
        ("alpha_0_prime", "inf"),
        ("alpha_0_double_prime", 3),  # anomalous Z trend; re-verify
        ("gamma_1_prime", "inf"),
    ),
    "cross_check_lane_not_substitute": "plasmapy_formulary_braginskii_z_table",
    "audit_basis": "docs/CODEX_SPRINT5_WS2_AUDIT_2026_05_20.md (HEAD 558de6f)",
    "extraction_date": "2026-05-20",
    "extraction_method": (
        "PyMuPDF page render at 200 dpi + two-pass visual confirmation of "
        "Z=1 and Z=inf columns + spot-check against PlasmaPy "
        "formulary.braginskii hardcoded Z table (cross-check lane only, "
        "not source-equivalent)"
    ),
}
```

This dict will be added to `src/dpf/first_principles/sprint6_target_extractions.py`
in the same Sprint 6 commit; the structural-invariant test
`test_per_target_resolves_subset_of_top_level` (from Sprint 5 WS2) applies
to it. No runtime consumer is modified by this Sprint 6 commit — that's
Sprint 5+ WS4 work.
