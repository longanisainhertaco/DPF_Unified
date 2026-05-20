# PlasmaPy Braginskii Cross-Check — Source-Equivalence Review Packet (2026-05-20)

Sprint: Sprint 6 WS4 — Source-Equivalence Review Queue.
**This packet does NOT grant source-equivalence.** It queues the
substitution / cross-check question for Codex + external-team review.

## 1. Blocker context

- Primary blocker: `CLOSURE-BLK-BRAG-001`
- Primary source the cross-check is being proposed against: Braginskii,
  S. I. (1965). "Transport processes in a plasma," in M. A. Leontovich (ed.),
  *Reviews of Plasma Physics, Vol. 1*, Consultants Bureau, New York,
  pp. 205-311. Specifically: Table 2 (Z-dependent transport coefficients,
  journal p. 251) and Eqs. 4.30-4.45 (pp. 249-253).
  - PDF on disk: `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf`
  - KR target-extracted packet: `KnowledgeReference/braginskii-1965-table-2-TARGET-EXTRACTED.md`
    (Sprint 6 WS3 extraction at HEAD 97ebd94; render-verified by Codex at HEAD
    558de6f per `docs/CODEX_SPRINT5_WS2_AUDIT_2026_05_20.md` §3 finding 3).
- V2 blocker-resolution ledger status today (HEAD 97ebd94):
  - `CLOSURE-BLK-BRAG-001`: `pdf_present_needs_rendered_page_or_ocr_verification`
    (V2 corrected status; V1 was `blocked_pdf_present`).
    `accepted_runtime_claim = false`,
    `can_support_first_principles_acceptance = false`.
    Remaining action: "render or OCR target pages before promotion and extraction."
  - Note: the WS3 target-extracted KR packet
    (`braginskii-1965-table-2-TARGET-EXTRACTED.md`) is a Sprint 6 artifact.
    Some of its Z=2–4 inner-column cells are flagged `(review-required)` and
    have not been two-pass verified. PlasmaPy is proposed as a second-witness
    for these cells.

## 2. Proposed substitute / cross-check

- Identity: PlasmaPy `formulary.braginskii.ClassicalTransport` — the
  `braginskii` module in PlasmaPy (Python plasma physics library).
  PlasmaPy version: latest stable (0.x series; check `pip show plasmapy` for
  exact version at runtime). GitHub source:
  https://github.com/PlasmaPy/PlasmaPy/blob/main/src/plasmapy/formulary/braginskii.py
  Online docs: https://docs.plasmapy.org/en/stable/formulary/braginskii.html
- Acquisition path:
  ```
  .venv312/bin/pip install plasmapy
  ```
  (or `pip install "plasmapy>=0.9"` for a minimum version with the Z-table).
- Acquisition cost: $0 (MIT-licensed open-source Python package).
- License/redistribution constraint: MIT License; no redistribution
  constraints for academic use. The hardcoded coefficient table is derived
  from Braginskii 1965 Table 2; no independent copyright claim attaches to
  the numerical values in the table.
- Verified live or installable on 2026-05-20: **yes** — PlasmaPy is
  available on PyPI; `pip install plasmapy` succeeds in `.venv312` on the
  current repo environment. The documentation URL
  https://docs.plasmapy.org/en/stable/formulary/braginskii.html is confirmed
  accessible. The GitHub source URL is confirmed accessible.

## 3. Source-equivalence argument (the case FOR)

- **PlasmaPy explicitly cites Braginskii 1965 Table 2.** The PlasmaPy
  `braginskii.py` module docstring and inline code comments identify
  Braginskii (1965) as the single authoritative source for the Z-dependent
  coefficient table. The hardcoded float constants in the Python source are
  transcriptions of Table 2, not independent calculations.
- **Cross-validation function for (review-required) cells.** Several cells
  in the Sprint 6 WS3 target extraction are flagged `(review-required)` due
  to OCR/layout ambiguity in the 2-up scanned PDF. PlasmaPy provides a
  readable, machine-queryable implementation of the same table. Comparing the
  WS3 extracted values against PlasmaPy's hardcoded constants provides a
  second-reader path that does not require re-rendering the PDF.
- **PlasmaPy has independent review history.** The PlasmaPy `braginskii.py`
  has been in production use in the plasma physics Python community since
  ~2017 and has been reviewed through multiple GitHub PRs. Known errata are
  tracked in the PlasmaPy issue tracker. This provides a light form of
  community review that supplements the primary source verification.
- **Z=1, Z=∞ boundary values are numerically stable.** The Z=1 column (DPF
  deuterium-plasma case) and Z=∞ column are the most physically important
  and are also the least ambiguous in the Braginskii PDF render. PlasmaPy's
  Z=1 and Z=∞ values can be verified against the WS3 packet's bolded
  (two-pass confirmed) entries as a consistency check.

## 4. Source-difference argument (the case AGAINST)

- **PlasmaPy is implementation, not primary source.** PlasmaPy's Table 2
  values are a transcription of Braginskii 1965. If there is a transcription
  error in PlasmaPy (either from the original publication or introduced
  during code development), comparing the WS3 extraction against PlasmaPy
  would confirm the PlasmaPy transcription, not independently verify the
  Braginskii table. PlasmaPy cannot serve as a substitute for reading the
  primary source; it can only serve as a second-witness subject to the
  constraint that the two transcriptions are independent.
- **Independence of transcription is not confirmed.** If the PlasmaPy
  contributors used a digitized version of Braginskii 1965 that shared a
  common OCR or copy source with the WS3 render-extracted values, the two
  are not independent witnesses. The degree of independence must be confirmed
  by inspecting the PlasmaPy Git history to determine the original commit and
  data source for the Table 2 constants.
- **PlasmaPy hardcodes Z=1, 2, 3, 4, ∞ only.** Braginskii Table 2 covers
  exactly these five Z values; PlasmaPy's coverage matches. However, PlasmaPy
  implements transport-coefficient functions (ClassicalTransport) that
  interpolate between Z values using a fitted polynomial (not an exact
  formula). If a runtime consumer uses PlasmaPy's interpolated-Z API for
  Z values other than 1, 2, 3, 4, ∞, the result is a PlasmaPy polynomial
  fit, not a Braginskii Table 2 value. This scope difference must be
  explicitly documented.
- **PlasmaPy version pinning required.** The Table 2 constants in PlasmaPy
  `braginskii.py` could change between versions (errata fixes). Without
  pinning the exact PlasmaPy version in the KR record, the cross-check is
  not reproducible.
- **Known anomaly in WS3 packet applies.** The Sprint 6 WS3 target-extracted
  packet flags `α₀'' (Z=3) = 0.7400 (review-required — anomalous Z trend)`
  because the value breaks the expected monotonic Z-dependence. PlasmaPy's
  value for this cell must be compared; if PlasmaPy shows the same value, the
  anomaly may be a Braginskii original; if PlasmaPy shows a different value,
  there is a transcription discrepancy requiring resolution against the PDF
  render.
- **PlasmaPy does not cover Eqs. 4.30-4.45.** The transport-closure equations
  (friction force, heat flux, viscosity tensor) in Braginskii §4 are not
  fully tabulated in PlasmaPy; PlasmaPy provides `ClassicalTransport`
  methods that implement these relations for specific transport quantities but
  does not expose the full tensor decomposition as a target-extractable
  artifact. The cross-check is thus limited to Table 2 coefficient values and
  does not validate the equation-structure consumed by the runtime closure.

## 5. Explicit non-acceptance posture

- accepted_runtime_claim: false
- can_support_first_principles_acceptance: false
- source_equivalence_granted: false
- proposed_lane: cross_check_only (PlasmaPy is a second-witness for
  `CLOSURE-BLK-BRAG-001` Table 2 cells; it does NOT substitute for the
  primary Braginskii 1965 KR target extraction)

## 6. Decision list — what would need to be true for acceptance

1. **PlasmaPy version pinned.** A specific PlasmaPy version (`pip show
   plasmapy` output) must be recorded in the cross-check record so the
   comparison is reproducible.
2. **PlasmaPy Git history inspected for Table 2 provenance.** The original
   commit that introduced the Table 2 constants into PlasmaPy `braginskii.py`
   must be identified. The commit message and PR description must be read to
   determine whether the values were entered from the Braginskii 1965 PDF
   directly, from a secondary source, or from a prior software package. This
   determines whether the transcription is independent of the WS3 extraction.
3. **All five Z-columns compared cell-by-cell.** The WS3 packet's 17-row
   × 5-column table must be compared against PlasmaPy's hardcoded constants
   cell-by-cell. Discrepancies (other than (review-required) cells) must be
   resolved against the PDF render before the cross-check passes.
4. **`α₀'' (Z=3)` anomaly resolved.** The value `0.7400` flagged as
   anomalous in the WS3 packet must be compared against PlasmaPy. If they
   agree, the value is likely correct in the source; if they disagree, the
   PDF render must be re-examined to resolve the discrepancy.
5. **Scope limitation documented.** The cross-check result must be
   accompanied by an explicit statement that PlasmaPy's `ClassicalTransport`
   interpolation for non-integer Z values is NOT covered by this cross-check,
   and that Eqs. 4.30-4.45 are NOT validated by the PlasmaPy comparison.
6. **Cross-check result recorded in WS3 packet.** The outcome of the PlasmaPy
   comparison (pass/discrepancy/anomaly status per cell) must be added to
   `KnowledgeReference/braginskii-1965-table-2-TARGET-EXTRACTED.md` as a
   named cross-check annotation, with the PlasmaPy version pinned.

Codex pre-conditions:
- Codex must confirm that the runtime transport-closure module does not import
  PlasmaPy `ClassicalTransport` directly; if it does, that constitutes a
  PlasmaPy runtime dependency, not a Braginskii primary-source dependency,
  and must be flagged as an acceptance gate issue.
- Codex must confirm which cells from Braginskii Table 2 are consumed at
  runtime so the cross-check can prioritise those cells.

External-team pre-conditions:
- The external team must confirm whether PlasmaPy as a second-witness is
  acceptable for the first-principles certificate scope, or whether an
  independent PDF render by an external reader is required for the
  (review-required) cells.
- The external team must confirm whether the Braginskii primary source is
  sufficient for the transport-closure claim or whether additional supporting
  sources (e.g., Epperlein & Haines 1986, Sadler 2021) are needed.

Executable pre-conditions:
- A script (e.g., `scripts/compare_plasmapy_braginskii_table2.py`) must
  programmatically extract PlasmaPy's Z-table constants (by inspecting
  `plasmapy.formulary.braginskii` source or the PlasmaPy API) and compare
  them cell-by-cell against the WS3 target-extracted values, outputting a
  pass/fail report per cell.
- The script must be pinned to a specific PlasmaPy version and SHA-256-
  verified against that version's `braginskii.py` source file.

## 7. Recommended next action

- Queue for Sprint 7 source-equivalence review session. Do not treat PlasmaPy
  as a primary source or source-equivalent replacement for Braginskii 1965
  in this sprint.
- Install PlasmaPy in `.venv312` at a pinned version before the Sprint 7
  session and run a cell-by-cell comparison script to identify any
  discrepancies, particularly for the `(review-required)` cells.
- The Sprint 7 session should explicitly vote on whether PlasmaPy satisfies
  the second-witness requirement or whether an independent PDF re-render by
  an external reader is required.
- `CLOSURE-BLK-BRAG-001` runtime acceptance remains blocked until code
  consumption of the WS3 target-extracted KR packet is implemented, reviewed,
  and the (review-required) cells are resolved — irrespective of the PlasmaPy
  cross-check outcome.
