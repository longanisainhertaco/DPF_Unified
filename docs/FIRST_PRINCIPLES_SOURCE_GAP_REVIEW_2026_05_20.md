# First-Principles Source-Gap Review Of Other-Team Findings

Date: 2026-05-20

Repository: `/Users/anthonyzamora/dpf-unified`

Context:
The external team reported that multiple Sprint 3 blockers were overstated as
"no KR source" when the local disk or existing text-parity KR corpus already
contains useful source material. This review checks the claim without promoting
any source to scientific authority.

## Verdict

The core correction is accepted:

`no KR source` is too broad for several blockers. The more accurate blocker
class for many items is:

`source_available_not_target_extracted`

That distinction matters. It means the next useful parallel lane is not broad
literature discovery; it is a controlled source-target extraction sprint over
files already present on disk or already text-parity promoted into
`KnowledgeReference/`.

This does not close validation. It does not make any raw PDF, text-parity KR
file, or shallow extracted file accepted physics authority. Runtime code may not
consume these values until they become typed target packets with units, source
line ranges, scope labels, uncertainty/status, and fail-closed tests.

## Inventory Checks

Observed local inventory:

- `archive_reference_OLD/references/papers/`: 952 PDF files.
- `downloaded_books_papers/`: 144 PDF files.
- `docs/research-reference/dpf_research.db`: 772 rows in `papers`.
- `archive_reference_OLD/references/papers/Research_Database.sqlite`: 410 rows
  in `papers`.

These counts support the team's main point: the machine already has substantial
raw source material. Raw possession is not authority; target extraction is the
missing step.

## Spot-Checked Claims

### Chamber Wall Material And Thickness

Historical finding, superseded by the target-extraction execution below:
The other team is correct that the current blocker wording in
`src/dpf/fields/source_geometry.py` was inaccurate before the 2026-05-20
target-extraction pass.

Pre-extraction code:

- `source_geometry.py:953-959` blocks `chamber_wall_material` and
  `chamber_wall_thickness_m` with reasons
  `PF1000-BLK-021-chamber-wall-material-no-kr-source` and
  `PF1000-BLK-022-chamber-wall-thickness-no-kr-source`.

Local KR evidence:

- `KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons-caused-by-the-plasma-focus-vessel-527cc533.md:114-115`
  states the PF-1000 vessel is stainless steel and gives average wall thickness
  of 10 mm.
- The promotion ledger shows that source as
  `text_parity_extracted_review_needed`:
  `docs/RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.md:22`.

Correct status before this pass:

- Keep the fields blocked for accepted/source-supported runtime geometry until
  target extraction is complete.
- Replace the blocker reason from `no-kr-source` to
  `source_available_not_target_extracted`.
- Add a target-extraction task for chamber wall material and wall thickness.

2026-05-20 execution:

- Target extraction is complete for the Krasa 2008 chamber wall material and
  wall thickness values.
- `PF1000GeometryPacket` now marks `chamber_wall_material` and
  `chamber_wall_thickness_m` as source-supported geometry-context fields.
- The chamber-wall mask still remains a candidate in the default Krauz packet
  because the cathode-cage radial split remains a conflict.

### Beam-Target Coupling

Finding:
The other team is correct to reject any claim that beam-target is "fully
covered."

Current code:

- `src/dpf/first_principles/closure_packet.py:754-816` cites Lee/Saw and
  Bosch-Hale support for phenomenological/cross-section pieces.
- The same block correctly preserves the real blockers:
  mechanism separation, ion-distribution transport, stopping, spectrum,
  anisotropy, detector response, and UQ.

Correct status:

- Bosch-Hale cross-section/reactivity support is a useful source foundation.
- It is not a complete beam-target authority chain.
- The honest blocker remains fast-ion stopping plus ion distribution plus
  detector/spectrum/anisotropy separation.

### Stepniewski 2004 Anode Bore

Finding:
The team's direction is useful, but the lane label should be corrected.

Observed source:

- `downloaded_books_papers/Research Papers/2026-05-11-user-ingest/stepniewski2004.pdf`
  exists on disk.
- It is also already text-parity promoted into KR:
  `docs/RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.md:46`.
- `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md:310-314`
  gives PF-1000 simulation parameters including hollow radius 0.015 m.

Correct status:

- This is not "not in KR." It is already in KR as text-parity extraction.
- It still needs target extraction and scope review before code can treat the
  bore radius as a supported PF-1000 geometry field.
- The Stepniewski value appears in an MHD modeling context; scope and revision
  conflicts must remain explicit.

### PF-1000 16 kV Comparator Channels

Finding:
The team's "real wall is validation, not physics inputs" statement is mostly
correct.

Existing packet evidence already says:

- Akel 16 kV shot-set current/scalar yield evidence exists:
  `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md:38-47`.
- Missing same-scope channels include `T_e`, directly measured `T_i`, neutron
  spectrum, anisotropy, startup/preionization, and spectroscopy at the 16 kV
  point:
  `WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md:65-71,185-189,298-308`.
- The same-scope source-search doc says the Akel 16 kV material is a candidate
  input/scalar-output source, not a complete accepted same-scope packet:
  `docs/FIRST_PRINCIPLES_SAME_SCOPE_SOURCE_SEARCH_2026_05_15.md:18-33,63-88`.

Correct status:

- Same-scope certificate completion is blocked unless the validation target is
  re-scoped or new same-scope measurements are obtained.
- Ingesting more existing PF-1000 full-energy material can improve physics
  foundations and cross-scope comparator design, but cannot manufacture missing
  16 kV `V(t)`, `T_e/T_i`, or X-ray evidence.

## Corrections To The Current Sprint 3R Framing

Sprint 3R remains necessary. The other-team findings do not remove the
runtime/control-plane bugs already identified:

- startup acceptance spoofing;
- scalar neutron authority spoofing;
- NumPy 2 beam-target failure;
- heuristic PF-1000 material masks;
- incomplete `Sigma_p` packet schema and dict ingestion;
- closure matrix omissions;
- restart-merge ledger omissions;
- stale packet ledgers and path drift.

But Sprint 3R should add one explicit parallel lane:

`S3R-X: Source Target Extraction Backlog`

This lane can run in parallel with code remediation because it is source-review
work, not runtime promotion.

## S3R-X Parallel Lane: Source Target Extraction Backlog

Goal:
Convert "no KR source" blockers into precise states:

- `target_extracted_source_supported`;
- `source_available_not_target_extracted`;
- `source_conflict_requires_review`;
- `genuinely_absent`;
- `wrong_scope_not_same_scope`.

Allowed work:

1. Inventory the 952 archived PDFs, 144 downloaded PDFs, and 772-row
   `docs/research-reference/dpf_research.db` against the Sprint 3 blocker list.
2. For files already in KR as `text_parity_extracted_review_needed`, produce
   target-extraction packets rather than duplicate ingestion.
3. For files on disk but not in KR, ingest as fail-closed text-parity KR
   records first, then target-extract only the needed quantities.
4. For genuinely absent sources, create acquisition records with exact missing
   equation/data target.
5. Preserve same-scope status. PF-1000 full-energy sources cannot become Akel
   16 kV validation evidence without a reviewed transfer rule.

Priority target packets:

| Priority | Target | Current reviewed status |
| --- | --- | --- |
| P0 | PF-1000 chamber wall material and wall thickness | Extracted 2026-05-20; geometry-context source-supported, not Akel validation. |
| P0 | PF-1000 hollow-anode bore radius from Stepniewski 2004 | Extracted 2026-05-20 as modeling context; runtime authority still blocked pending hardware-scope review. |
| P0 | Klir 2011 detector response | Already target-extracted in `kr_targets.py`; now referenced in the S4 extraction packet. |
| P0 | Krasa 2008 direct/scattered neutron transport and vessel geometry | Extracted 2026-05-20; new KR target added. |
| P1 | Current-sheath initiation source packet | Extracted 2026-05-20 as wrong-scope startup method context. |
| P1 | Hall/Ohm/transport sources | Neon gas-puff Hall/LHDI formula extracted 2026-05-20 as non-DPF closure candidate; NRL 2019 transport core extracted as formulary cross-check. Direct Braginskii 1965 coefficient extraction remains separate. |
| P1 | Talebitaher ion distribution / anisotropy material | Extracted 2026-05-20 as NX2 detector/anisotropy context, not PF-1000/Akel authority. |
| P1 | EOS/radiation/opacity sources | On-disk sources likely exist; authority depends on extracting exact equations, units, regimes, and validity limits. |
| P2 | Restrike and D2 Townsend/Paschen constants | Still likely genuine external acquisition unless local search finds exact equations/data. |

2026-05-20 target-extraction execution:

- Added `docs/FIRST_PRINCIPLES_TARGET_EXTRACTIONS_2026_05_20.md` as the
  durable extraction report.
- Added `sprint4_source_available_target_extractions()` with seven typed,
  line-referenced records for Krasa 2008 PF-1000 vessel geometry/scatter,
  Stepniewski 2004 PF-1000 hollow-bore simulation context, UCSD/Beg startup
  method context, the neon gas-puff Hall/LHDI anomalous-resistivity formula,
  the NRL 2019 transport formulary cross-check, Talebitaher 2012 NX2
  detector/anisotropy context, and the already-coded Klir 2011 ToF
  detector-response target.
- Added `pf1000_krasa_vessel_scatter_anisotropy_targets()` to the KR target
  manifest.
- Promoted only the Krasa chamber-wall material/thickness fields in
  `PF1000GeometryPacket` to source-supported geometry context. Stepniewski's
  hollow bore remains blocked as
  `target_extracted_modeling_context_requires_review`.
- Validation state did not change: Akel 16 kV same-scope field, temperature,
  X-ray, spectrum, anisotropy, and whole-shot certificate channels remain
  blocked.

## Required Output From The Other Team

The team should not return prose-only claims. The acceptable output is a
source-target extraction packet with:

1. blocker ID;
2. old status;
3. proposed new status;
4. local file path;
5. exact KR path and line range, or PDF path and page/figure/table;
6. extracted value/equation;
7. units and symbol map;
8. scope tag;
9. uncertainty or "uncertainty not supplied";
10. runtime claim impact;
11. tests that must keep non-target-extracted data blocked.

## Codex Audit Position

Accepted:

- The team found a real taxonomy error: several blockers are
  `source_available_not_target_extracted`, not `no KR source`.
- The chamber-wall correction is concrete and should be fixed.
- The beam-target blocker is correctly preserved as incomplete despite
  Bosch-Hale support.
- A source-target extraction lane should begin immediately in parallel with
  Sprint 3R code remediation.

Not accepted:

- Raw PDFs cannot be treated as KR authority.
- Text-parity KR cannot be treated as target-extracted authority.
- PF-1000 full-energy sources cannot validate Akel 16 kV shot 12581 without a
  scope decision or transfer rule.
- Beam-target is not fully covered.

Bottom line:

The other-team finding changes the source-work plan, not the validation state.
It should reduce blocker-search waste and convert many "missing source" fields
into exact extraction tasks. It does not remove the Sprint 3R remediation gate.
