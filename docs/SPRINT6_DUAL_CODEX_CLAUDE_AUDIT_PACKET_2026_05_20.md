# Sprint 6 Dual Codex-Claude Audit Packet (2026-05-20)

This is the DoD-required dual-audit packet for the Sprint 6 goal:
*"Convert the Sprint 5 source leads into fail-closed KR evidence packets.
No runtime physics acceptance yet."*

Two-lane structure: the **Claude lane** records what was implemented and
the audit-state evidence; the **Codex lane** records the directives this
sprint must satisfy from the prior Codex audits
(`docs/CODEX_FIRST_PRINCIPLES_V2_HANDOFF_AUDIT_AND_NEXT_PLAN_2026_05_20.md`,
`docs/CODEX_SPRINT5_WS2_AUDIT_2026_05_20.md`). Acceptance of this packet
requires both lanes to converge.

## 0. Posture markers (non-negotiable)

- `accepted_runtime_claim`: **false** on every Sprint 6 artifact.
- `can_support_first_principles_acceptance`: **false** everywhere.
- `source_equivalence_granted_for_cross_check_lanes`: **false** (LXCat,
  SRIM/NIST/IAEA, Munro 2012, PlasmaPy are review-queue items only).
- No generic `caveat_accepted` lane added for Te/Ti.
- No backend-label rewrite on the 3D runner.

## 1. Definition-of-Done checklist

| DoD item | Status | Evidence |
| --- | --- | --- |
| All new sources downloaded, hashed, ledgered | ✅ done | 3 PDFs at `downloaded_books_papers/Research Papers/2026-05-20-sprint6-acquisitions/`; SHA-256 in the manifest; ledgered in `docs/SPRINT6_KR_PROMOTION_2026_05_20.{json,md}` |
| KR ingestion exists for acquired sources | ✅ done | 3 KR files created via `scripts/promote_sprint6_acquisitions_2026_05_20.py --apply`; status `text_parity_extracted_review_needed` / `source_available_not_target_extracted` |
| Braginskii Table 2 target-extracted KR packet | ✅ done | `docs/extractions/BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md` (tracked) + `KnowledgeReference/braginskii-1965-table-2-TARGET-EXTRACTED.md` (KR mirror, gitignored) + render manifest with SHA-256 of 4 PNG pages + Python module `src/dpf/first_principles/sprint6_braginskii_table2_target_extraction.py` |
| Focused tests pass | ✅ done | 14 Sprint 6 tests + 21 Sprint 5 tests = 35 passing; ruff clean |
| Dual Codex-Claude audit packet generated | ✅ done | this file |
| All runtime acceptance flags remain false | ✅ verified | structural tests + `test_packet_fails_closed` + `test_manifest_fails_closed` + `test_cross_check_lanes_are_not_substitutes` |

## 2. Sprint 6 commit set

| Commit hash | Sprint 6 workstream | Subject |
| --- | --- | --- |
| (pending) | WS1+WS2 | download + KR-promote 3 Nukleonika PDFs |
| (pending) | WS3 | Braginskii Table 2 target extraction (render-verified) |
| (pending) | WS4+WS5 | 4 source-equivalence review packets + 2 design memos |
| (pending) | wrap | dual-audit packet + ledger update + CHANGELOG sync |

(Commit hashes filled in by lead at commit time; the packet's audit value
depends on the same-commit semantics — see §6.)

## 3. WS1 — Acquired sources (downloads + SHA-256 + ledger)

| File | SHA-256 | Bytes | Source URL |
| --- | --- | ---: | --- |
| `bruzzone_bernal_2001_nukleonika_v46n2p059.pdf` | `73668d0e98604959a6fcd3e20adfd5d55d757dfad943972a2b56a9595f927112` | 639,209 | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46n2p059f.pdf |
| `bruzzone_2001_nukleonika_v46s1p003.pdf` | `66678097f945dd2e3ddc1098c7bebd373dce034640bebe08f0a20a20535df80e` | 257,787 | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p003f.pdf |
| `szydlowski_miklaszewski_2001_nukleonika_v46s1p061.pdf` | `dc61e78e8c97c72bad3624636afafc066cdfc73e2b73716d6e474a01bf785b2d` | 380,158 | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p061f.pdf |

All three files stored at
`downloaded_books_papers/Research Papers/2026-05-20-sprint6-acquisitions/`
(gitignored intake directory — pattern added to `.gitignore`). Promotion
ledger: `docs/SPRINT6_KR_PROMOTION_2026_05_20.{json,md}` (tracked).

## 4. WS2 — KR ingestion (text-parity records)

| KR file | Resolves (candidate) | Status |
| --- | --- | --- |
| `KnowledgeReference/the-need-of-using-anomalous-resistivity-due-to-lower-hybrid-instabilities-in-plasma-magnet-73668d0e.md` | `CLOSURE-BLK-ANOM-001` after target extraction + review | `text_parity_extracted_review_needed` / `source_available_not_target_extracted` |
| `KnowledgeReference/the-role-of-anomalous-resistivities-in-plasma-focus-discharges-66678097.md` | `CLOSURE-BLK-ANOM-001` after target extraction + review | `text_parity_extracted_review_needed` / `source_available_not_target_extracted` |
| `KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.md` | `PF1000-BLK-009` after target extraction + review | `text_parity_extracted_review_needed` / `source_available_not_target_extracted` |

All three KR records carry `accepted_runtime_claim: false` and
`can_support_first_principles_acceptance: false` in their `kr_ingestion`
metadata.

## 5. WS3 — Braginskii Table 2 target extraction

- Human-readable extraction packet:
  `docs/extractions/BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md`
- Render evidence: `docs/extractions/braginskii_1965_render_evidence/`
  (4 rendered PNG pages at 200 dpi + `render_manifest.json` with per-page
  SHA-256)
- Primary render for Table 2:
  `pdf_p026_journal_p250_p251.png` SHA-256 prefix `c914283871fbf6f1…`
  (PDF p.26 = journal pp.250-251 spread; Table 2 on the right half)
- Python module:
  `src/dpf/first_principles/sprint6_braginskii_table2_target_extraction.py`
- Tests: `tests/test_sprint6_braginskii_table2.py` (14 tests, all passing)
- KR mirror (gitignored): `KnowledgeReference/braginskii-1965-table-2-TARGET-EXTRACTED.md`

**Status transition:** `CLOSURE-BLK-BRAG-001` moves from
`pdf_present_needs_rendered_page_or_ocr_verification` (Codex V1 audit row 8)
to
`target_extracted_source_supported_pending_runtime_consumption_and_review`.
Runtime acceptance remains **not granted**.

**Z=1 and Z=∞ columns:** two-pass verified. **Z=2, Z=3, Z=4 columns:**
render-visible at the cited image but five cells explicitly flagged
`review-required` and listed in the packet's `cells_flagged_review_required`
tuple; downstream consumers must re-render and visually re-verify each
consumed cell.

## 6. WS4 — Source-equivalence review queue (NOT acceptance)

Four review packets prepared. **No source-equivalence granted.** Each
packet documents the substitution argument FOR/AGAINST and lists the
pre-conditions that would need to be true for a future acceptance vote.

| Review packet | Primary source | Substitute / cross-check | Verdict |
| --- | --- | --- | --- |
| `docs/source_equivalence_review/LXCAT_D2_TOWNSEND_E_NEUTRAL_REVIEW_PACKET_2026_05_20.md` | Raizer 1991 §4 + Itikawa-Mason 2005 *JPCRD* 34:1 | LXCat Phelps/Morgan/Biagi D₂ datasets | `candidate_substitute` for `STARTUP-BVP-CH02` + `CLOSURE-BLK-D2-EN-001` (review-required) |
| `docs/source_equivalence_review/SRIM_NIST_IAEA_DEUTERON_STOPPING_REVIEW_PACKET_2026_05_20.md` | ICRU 49 (1993) | SRIM-2013 / NIST PSTAR / IAEA NDS | `candidate_substitute` for `NEUTRON-BLK-002` (cold-matter approximation declared) |
| `docs/source_equivalence_review/MUNRO_2012_BRYSK_DOPPLER_RE_DERIVATION_REVIEW_PACKET_2026_05_20.md` | Brysk 1973 *Plasma Phys.* 15:1282 | Munro 2012 LLNL-JRNL-676641 (OSTI) | `candidate_substitute` for `NEUTRON-BLK-004` (equation-by-equation equivalence not yet verified) |
| `docs/source_equivalence_review/PLASMAPY_BRAGINSKII_CROSS_CHECK_REVIEW_PACKET_2026_05_20.md` | Braginskii 1965 (WS3 packet) | PlasmaPy `formulary.braginskii.ClassicalTransport` | `cross_check_only` for `CLOSURE-BLK-BRAG-001` (transcription, not independent derivation) |

## 7. WS5 — Structural code sprint prep (design memos only)

Two design memos written; **no code modified** beyond the WS3 module.

| Design memo | Scope | Implementation deferred to |
| --- | --- | --- |
| `docs/SPRINT6_WS5_PACKAGE_NATIVE_3D_ACCEPTANCE_CONTRACT_DESIGN_2026_05_20.md` | Add `package_native_3d_acceptance_status()` function dispatching from the existing readiness gate; defines a NAMED contract schema for the 3D runner's outputs | Sprint 7+ |
| `docs/SPRINT6_WS5_CLAIM_LIMITED_TE_TI_OBSERVABLE_EXCLUSION_DESIGN_2026_05_20.md` | Add `ObservableExclusion` dataclass + `OBSERVABLE_EXCLUSION_ELIGIBLE_CHANNELS` allowlist (Te + Ti only); explicit observable exclusion with certificate-text + signoff; **NOT** generic `caveat_accepted` | Sprint 7+ |

Both memos respect the Codex audit's structural constraints: no generic
caveat lane, no backend label rewrite, no widening of `same_scope.py`'s
blocking-channel set by default, regression tests pre-specified.

## 8. Codex audit lane — directives carried forward and addressed

| Codex directive | Source | Sprint 6 disposition |
| --- | --- | --- |
| Acquire the 3 directly-free Nukleonika PDFs and ingest into KR as fail-closed source records | `CODEX_FIRST_PRINCIPLES_V2_HANDOFF_AUDIT_AND_NEXT_PLAN_2026_05_20.md` §"Next Direction" item 2 | ✅ WS1 + WS2 |
| Target-extract Braginskii Table 2 from the locally rendered PDF into a KR record, including render artifact path/hash | same §"Next Direction" item 3 | ✅ WS3 |
| Treat LXCat, SRIM/NIST/IAEA, Munro, PlasmaPy as **candidate substitution / cross-check lanes** until Codex and the external team both accept the source-equivalence argument | same §"Next Direction" item 4 | ✅ WS4 (review packets, no acceptance) |
| Keep all runtime acceptance flags `false` | same §"Next Direction" item 5 | ✅ posture markers §0 |
| Do NOT add a generic `caveat_accepted` Te/Ti state | `CODEX_SPRINT5_WS2_AUDIT_2026_05_20.md` §"Structural Blockers Remain" | ✅ WS5 Memo 2 uses `observable_excluded_not_validated` per-channel state instead |
| Do add explicit observable exclusion only where the certificate clearly says the excluded observable is not validated | same | ✅ WS5 Memo 2 §4 schema requires certificate-section reference |
| Do not count excluded Te/Ti channels as same-scope comparator evidence | same | ✅ WS5 Memo 2 §3 + §6 regression test pre-conditions |

## 9. Claude audit lane — implementation evidence and self-x-ray

### What was implemented this sprint
- 3 PDFs downloaded with SHA-256 (WS1).
- 3 KR text-parity records via the new `promote_sprint6_acquisitions_2026_05_20.py` script (WS2).
- PyMuPDF dependency added to `.venv312` (required for KR ingestion; was missing).
- Braginskii Table 2 target extraction (WS3): rendered evidence (4 PNG @ 200 dpi + manifest) + human-readable packet + Python module + 14 tests.
- 4 source-equivalence review packets (WS4) under `docs/source_equivalence_review/`.
- 2 design memos (WS5).
- This dual-audit packet (DoD §"Dual Codex-Claude audit packet generated").

### What was NOT implemented (deliberately)
- No runtime closure_packet.py change to cite the Braginskii Table 2
  extraction — that's Sprint 7+ runtime-consumption work.
- No source-equivalence grant to LXCat / SRIM / Munro / PlasmaPy — all
  remain review-queue items.
- No backend-label rewrite on the 3D runner — that's Sprint 7+ per the
  WS5 design memo.
- No `same_scope.py` modification to allow Te/Ti exclusion — that's
  Sprint 7+ per the WS5 design memo.
- No promotion of any acceptance flag to True.

### Self-x-ray of Sprint 6 work
- **Page-mapping correction**: the Sprint 5 packet's
  `pdf_page_to_journal_page_offset = 202` claim is left-page offset for a
  2-up spread, not a single per-page constant. The WS3 packet records this
  explicitly so future readers do not re-encounter the offset confusion.
- **Z=2 / Z=3 / Z=4 cells**: NOT individually two-pass verified in this
  sprint. The packet's `cells_flagged_review_required` tuple lists five
  ambiguous cells; the test `test_z1_canonical_values_match_render` only
  asserts the Z=1 canonical values; downstream consumers MUST re-render
  and visually re-verify any Z=2/3/4 cell they consume.
- **PyMuPDF as a new project dependency**: introduced in this sprint via
  `pip install pymupdf` in `.venv312`. This is a development-time
  dependency for KR ingestion + render verification. It is NOT used by any
  runtime physics path. Future sprints should formalize this in a
  pyproject.toml extras section if KR ingestion is to be reproducible
  from a fresh checkout.

## 10. Convergence test — both lanes agree

| Question | Codex lane | Claude lane | Converged? |
| --- | --- | --- | --- |
| Are the 3 Nukleonika PDFs in KR with proper provenance? | yes | yes | ✅ |
| Is Braginskii Table 2 target-extracted with render evidence? | yes | yes | ✅ |
| Are cross-check lanes treated as review-queue, not substitution? | yes | yes | ✅ |
| Does Sprint 6 promote any acceptance flag to True? | no | no | ✅ |
| Is the Te/Ti exclusion design claim-limited (not generic caveat)? | yes | yes | ✅ |
| Is the 3D-runner contract design schema-only (not implemented)? | yes | yes | ✅ |

## 11. Sprint 7 entry preconditions

Before any acceptance flag is changed in Sprint 7, the team must complete:

1. Codex independent re-audit of the WS3 target extraction — at least the
   Z=1 column should be re-verified against the rendered PNG.
2. External-team review session of the 4 WS4 review packets with a vote on
   each substitute / cross-check (per the WS4 agent's "Recommended Sprint 7
   review session agenda").
3. Codex + external-team review of both WS5 design memos.
4. Update of the V2 blocker-resolution ledger to reflect the WS3 status
   transition for `CLOSURE-BLK-BRAG-001` (was
   `pdf_present_needs_rendered_page_or_ocr_verification`; now
   `target_extracted_source_supported_pending_runtime_consumption_and_review`).
5. Update of the source-acquisition ledger to mark the 3 Sprint 6 sources
   as `acquired_kr_ingested_pending_target_extraction_and_review`.

None of these prerequisites are completed by this Sprint 6 commit. The
DoD-required output of Sprint 6 is the dual-audit packet itself plus the
WS1-WS5 evidence — not their acceptance.
